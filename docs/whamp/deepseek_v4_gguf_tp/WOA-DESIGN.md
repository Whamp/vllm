<!-- markdownlint-disable MD060 -->

# WOA-DESIGN.md — Q8_0 wo_a output projection design (M1 scope, M2 gate)

## Problem

`wo_a` is GGUF `blk.l.attn_output_a.weight`, Q8_0 [4096, 8192]
(ne0=4096=K, ne1=8192=N = n_groups 8 × o_lora_rank 1024), 43 layers,
**0.3833 GiB/rank after TP=4 column sharding** (N=2048/rank). In vLLM it is
a `ColumnParallelLinear` flagged `is_bmm=True` with
`bmm_batch_size=n_local_groups` (attention.py:284-293) — a batched matmul
over 2 local groups at TP=4.

Precedent (WNA16, measured): naive dequant-to-BF16 caching cost **688
MiB/rank**; disabling the cache collapsed decode to **34.01 tok/s** (per-
token full dequant); the production answer was the FP8 Marlin-**diagonal**
path (Whamp/vLLM `7b39c930`, patch 0010): keep weights quantized, flatten
the BMM groups into one N=2048 GEMM, select diagonal outputs. That path is
FP8-block specific and does not apply to Q8_0 bytes.

## Design: int8-g32 Marlin-diagonal for Q8_0

1. **Repack** (shared with all dense Q8_0): Q8_0 → symmetric int8 codes +
   fp16 group-32 scales (2 B per 32 weights — byte-neutral vs 34 B blocks),
   Marlin tile-packing, uint8b128 offset. Documented last-bits loss vs
   dequant+GEMM; class-B window. All wo_a ne0 values (4096) divisible by 32
   — no partial blocks (verified from inventory).
2. **Kernel route**: the dense compressed-tensors WNA16 Marlin path
   (`vllm/model_executor/layers/quantization/compressed_tensors/schemes/
   compressed_tensors_wNa16.py`: bits=8 → `scalar_types.uint8b128`, Marlin
   input/scales helpers) with the **diagonal trick from patch 0010**:
   flatten [2 groups, 1024] BMM into one [2048, 4096] GEMM per rank, run
   Marlin, select the group-diagonal output rows. Existing fused-output-
   selection helpers from the FP8 path (`marlin.py` @ 6354125a) are the
   adaptation seam.
3. **Memory**: packed int8 stays in place; **no BF16 cache** — recovers the
   688 MiB/rank the naive path would cost. Post-transform wo_a stays
   **0.3833 GiB/rank exactly**: Q8_0 = 32 int8 codes + 2-byte fp16 scale,
   identical to int8-g32 storage (byte-neutral, excluding tile padding).
4. **Graph capture**: Marlin launches are stream-parameterized and capture-
   safe in the current stack (the FP8-diagonal path runs under graphs in
   production); M2 must verify the int8-flavored path identically (replay
   M=1–4 + aliasing sweep, per PLAN §8 M2).

## Alternatives rejected

- BF16 cache: +688 MiB/rank → roughly **−100K to −120K context tokens**
  (688 MiB / 5.832 KiB/tok gives 118K linear; measured precedent was ~92K
  because heterogeneous-cache admission is nonlinear). It is fatal to the
  ≥140K target, not a fallback.
- Per-token dequant: measured 34.01 tok/s — fatal, rejected.
- fp16-scaled cuBLAS GEMM on repacked codes: not capture-safe workspace-
  wise without more work; Marlin is the proven lane.

## Kill gate (M2)

If the int8 Marlin-diagonal prototype at real serving shapes (TP=4, M=1–4,
groups=2×1024) either (a) fails correctness beyond the class-B window,
(b) breaks graph capture after the ABI fix, or (c) costs > ~0.9 ms/token
vs the FP8 path's ~23% dense share budget → **stop per PLAN §8** (dense/
wo_a miss → dense redesign or stop).

## VRAM delta summary

| Option | wo_a resident/rank | Cache | Notes |
|---|---:|---:|---|
| BF16 cache (rejected) | 0.3833 GiB packed + 0.672 GiB bf16 | yes | ≈−100K to −120K ctx |
| int8 Marlin-diagonal (design) | 0.3833 GiB + tile padding | **no** | byte-neutral before tile padding |
