# M2 iteration 2 — native Q8_1 + DP4A IQ2_XXS indexed gate/up

Decision: **IQ2 gate/up fragment passes; proceed to M3 Q2_K down.** This is
not full M2 completion yet: dense Q8/wo_a prototype and the graph-captured
layer slice remain.

## Corrected production shape

Pinned vLLM source proves routed experts are within-expert TP sharded:
`intermediate_size_per_partition = 2048 / TP4 = 512`
(`fused_moe/config.py:1333`; compressed-tensors WNA16 shapes
`compressed_tensors_moe_wna16.py:173-238`). Every rank holds all 256 experts:

- gate/up: K4096→N512/rank, all selected top-6 experts;
- down: K512/rank→N4096 partial output, reduced across TP.

The earlier N2048 single-GPU benchmark was a conservative wrong-shape test,
not production placement. PCIe/NVLink do not constrain these local weight
reads; server60's PCIe cost remains in the separately measured collective
pool (M0: 19.74%).

## Implementation

- Caller-owned bf16→Q8_1 quantizer: one fp16 scale/32 values + int8 codes;
  exact DwarfStar rounding (`amax/127`, `roundf`), separately timed.
- Native IQ2_XXS×Q8_1 DP4A preserves group-level integer
  `sumi * ((aux32>>27)|1) / 8` truncation.
- Final schedule follows antirez/ds4's measured prototype: one lane owns one
  32-weight pair and executes eight DP4As; 32 lanes cover four 256-weight
  blocks/pass. Lookup tables use ordinary read-only device memory (constant
  memory serialized divergent indices).
- Indexed op handles top-k ids plus gate+up in one launch; no allocations,
  current stream, CUDA-graph safe. No ggml linkage.

## Correctness/build evidence

- RTX 3090: **15/15 tests pass** — exact quant scales/codes, independent
  packed-integer output oracle, raw/aligned equality, selected expert mapping,
  M=1/2/4, CUDA graph replay.
- Bugs caught red→green: missing warp-max broadcast; signed overflow in parity
  byte replication (`uint32_t` required); indexed-output rank validation.
- SM86 stable extension built/packaged; latest extension SHA-256 at test gate
  `bf752202…` before the final validation-only relink.

## Exclusive five-trial result (K4096→N512/rank, top6 gate+up)

Protocol: 5 independent runs, 5,000 warmup + 10,000 measured calls each;
canonical service stopped; zero compute processes before launch; process
sampled every 100 ms; at most one process and GPU0 only across 248 samples;
max clock exactly 1650 MHz; canonical restored healthy/zero-swap.

| Metric | Result |
|---|---:|
| indexed gate+up mean | **26.231 µs** |
| range | 26.124–26.386 µs |
| CV | **0.343%** |
| logical bandwidth mean | **247.35 GB/s** |
| range | 245.89–248.36 GB/s |
| quantize mean (direct launch) | 8.363 µs |
| captured quantize+indexed pipeline | **27.309 µs** |
| pipeline CV | 0.541% |

Evidence: `evidence/m2-iq2-iteration2/`.

## Interference audit

The earlier 256.86 GB/s provisional run's host journal shows only our
benchmark container between canonical stop/restart; no overlapping Docker or
SSH session. The controlled rerun is ~3.7% lower, so the provisional result
was optimistic due short timing/clock settling, **not external interference**.
The five-trial result supersedes it.

## Causal decision

- Scalar BF16 iteration: 40.9 GB/s (rejected).
- Q8 DP4A single matrix: 190.5 GB/s.
- Indexed exact TP4 gate+up: **247.35 GB/s / 27.31 µs including shared
  quantization** — enough to justify Q2_K down and the layer-slice gate.
- Aligned layout is **not selected for IQ2 DP4A**: at the exact N512
  single-matrix shape it was slower than raw (9.90 vs 8.89 µs); indexed raw
  already exposes concurrency. Keep aligned code only as reference evidence
  until final cleanup; production candidate uses raw GGUF blocks directly.
