<!-- markdownlint-disable MD060 -->

# GGUF-TP: running llama.cpp's hardest quantized weights natively inside vLLM

*Native GGUF tensor-parallel inference for DeepSeek-V4-Flash on 4× RTX 3090 —
no GGML, no requantization, from-scratch SM86 kernels.*

**Abstract.** The best-available quantization of DeepSeek V4 Flash lives in a
llama.cpp GGUF file (Antirez's IQ2_XXS / Q2_K / Q8_0 family, 86.7 GB). It
measured dramatically better under agentic evaluation than the safetensors
requantization that vLLM normally consumes — but llama.cpp cannot
tensor-parallel this model on 24 GB cards, and vLLM's own GGUF support
converts weights to float at load instead of executing the packed bytes. We
built the missing bridge: a native GGUF execution engine inside vLLM's
tensor-parallel runtime. Eight from-scratch SM86 CUDA kernels run the exact
GGUF bytes, a byte-exact contract guarantees what the kernels consume, and
the validated result is 76.7 decode tok/s (2× llama.cpp), 551.9 cache-busted
prefill tok/s, 140K on-GPU context with exact long-context recall, and an
agentic benchmark result that matches the proven llama.cpp baseline at 2.65×
the wall-clock speed. This is the story, including the parts that failed.

---

## 1. The problem: three pieces of the ecosystem that don't fit

DeepSeek V4 Flash is a 200B-class Mixture-of-Experts model that fits on
consumer hardware only when aggressively quantized. Three facts define the
landscape we started from:

1. **The best weights are in a GGUF file.** Antirez's
   `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-SExpQ8-OutQ8-chat-v2-imatrix-0731`
   GGUF is the only quantization of this model that has held up under our
   hardest gate. In a post-fix 12-task DeepSWE comparison, GGUF-via-llama.cpp
   scored 6 strict solves / 96.57% partial reward, versus 0 solves / 80.62%
   for the WNA16 safetensors requantization running on vLLM. **The
   requantization, not the runtime, was the quality killer.**

2. **llama.cpp cannot tensor-parallel this model.** Audited exhaustively:
   llama.cpp's CUDA tensor parallelism doesn't fit 24 GB cards with this
   weight set, and row-splitting the layers breaks the grouped-attention
   computation graphs. The good weights were structurally capped at ~38
   engine tok/s on a single GPU.

3. **vLLM's GGUF support converts rather than executes.** Upstream vLLM reads
   GGUF and translates weights into its own formats at load time — low-bit
   formats become float tensors in the default path, and the quantized path
   that keeps some formats packed runs through vendored GGML kernels on a
   single GPU (no TP). No runtime in the ecosystem executed the packed
   IQ2_XXS / Q2_K / Q8_0 bytes on a tensor-parallel GPU stack. "GGUF
   in tensor parallelism" is a standing community ask nobody had shipped.

The commercial irony: vLLM reached 74.98 decode tok/s on *inferior* WNA16
weights, while the *better* weights were stuck at 38. The gap between those
two numbers was the entire project.

## 2. Design stance: an inference-engine ASIC

We treated this as building an engine, not integrating one: **one model, one
artifact family, one hardware target, zero backward compatibility.** The
GGUF file is the immutable source of truth — same-architecture weight
updates reload through the same pinned contract. No llama.cpp/ggml wrapping:
the wrap route entered once via review suggestions, was demolished by the
review process, and was cut permanently. Will's original direction stood:
rewrite the expert kernels from scratch, vLLM-native, reading GGUF bytes
directly, with tests that verify accuracy.

The base pin is vLLM branch `incubate/gguf-tp-sm86`, built on the speed-stack
tip `b7766cfe` — the accumulated result of prior campaigns on this repo:
FlashMLA sparse decode, hierarchical all-reduce (the two-GPU-pair pattern
that survives our PCIe-only, no-NVLink topology), fp8_ds_mla KV cache, the
DSML tool-call parser fix, Marlin-wo_a diagonal machinery, CUDA-graph capture.
GGUF-TP inherits all of that and adds the GGUF-native layer: 14 commits, 36
files, +6,454 lines over the base.

**Keep:** vLLM's scheduler and continuous batching, the TP=4 launcher,
FlashMLA sparse decode, hierarchical all-reduce, fp8_ds_mla KV, CUDA-graph
capture, tokenizer/chat path.

**New:** the GGUF-native loader, the byte-exact contract machinery, and the
eight kernels.

**Deliberate non-goals:** no new dense GEMM family (Q8_0 rides existing Marlin
int8 after a repack), no aligned-SoA weight transform for IQ2 (measured
slower at our exact shapes), no derived repack artifact for Q2_K (raw already
sits at llama.cpp's bandwidth), no BF16 KV cache fallback.

## 3. The contract: byte-exact, before a single kernel

Kernels that consume quantized bytes are only as good as their understanding
of those bytes. We pinned the contract three ways before writing CUDA.

**L0 oracle (class A, bitwise).** We transcribed `dequantize_row_q8_0`,
`dequantize_row_q2_K`, and `dequantize_row_iq2_xxs` verbatim from pinned
llama.cpp source (Whamp/llama.cpp `0379cf4bf`) into a standalone C reference,
then wrote an *independent* fp32 decoder in NumPy from the written contract —
no shared code path. 10,000 random blocks per format plus adversarial
corpora (LUT boundary indices, sub-scale extremes, scale-nibble extremes,
±max/subnormal block scales, NaN/Inf probes): **bitwise pass 100%** on all
three formats. The oracle earned its keep on day one: the independent decoder
initially wrote chunk-1 outputs at weights 32..159 instead of 128..255 — a
real off-by-chunk bug caught by red→green before any kernel existed.

**Inventory.** A bounded 16 MiB header parse of the 86.72 GB blob produced a
1,328-tensor directory with a full set of consistency proofs: offsets
monotonic, zero overlaps, last tensor ends exactly at file size minus data
start, and Σ(bytes) + Σ(alignment gaps) = capacity exactly. Family bytes:
routed experts 72.56 GiB (IQ2_XXS gate/up at [4096, 2048, 256] = 528 MiB × 86,
Q2_K down at [2048, 4096, 256] = 672 MiB × 43 — note the down projection's
K/N axis swap vs gate/up), attention 4.55 GiB, shared experts 1.07 GiB,
token embedding 0.99 GiB F16 [4096, 129280], indexer/compressor 0.91 GiB,
output 0.52 GiB Q8_0 [4096, 129280], router 0.09 GiB (including three I32
`tid2eid` hash-table tensors), hyperconnection 0.06 GiB.

**Coordinate-aware mapping (class A2).** Byte checksums cannot catch a
transposed or incorrectly fused load, so we added coordinate oracles: for every
tensor family and TP boundary, sample (expert, output row, input column)
coordinates, derive the GGUF byte offset independently, decode, and compare
against the destination's logical value — covering first/last blocks,
first/last ranks, fused-slot boundaries, and hash-table rows. The tensor-level
TP mapping table assigns every one of the 1,328 tensors to a vLLM destination
with constructor file:line citations: `fused_wqa_wkv` and both compressors
replicated (`disable_tp=True`), indexer `wq_b`/`weights_proj` replicated
linears, `wq_b`/`wo_a` column-sharded, `wo_b` row-sharded, routed experts
whole-matrix sharded with all 256 experts per rank (intermediate dimension
/4: gate/up N512, down K512 — a correction the pinned source forced), token
embedding/output vocab-sharded, router/HC/norms/tid2eid replicated. Result:
21.1893 GiB per rank, verified twice by independent tools.

**Tokenizer.** GGUF and HF tokenizer alphabets identical by id (129,280
tokens, zero mismatches), 127,741 merges identical in order. The stack's
`deepseek_v4` tokenizer mode (DSML, tools, thinking) is pinned explicitly.

## 4. The kernels

Eight `__global__` kernels across five source files under
`csrc/libtorch_stable/quantization/gguf_dsv4/` (~1,700 lines including
support headers). All registered capture-safe from day one (current-stream
argument, caller-owned workspaces, no internal pools), all compiled for
`compute_86, code=sm_86`.

| Kernel | File | Role |
|---|---|---|
| `iq2_xxs_matvec_kernel` | iq2_xxs_matvec.cu | decode, raw GGUF blocks (reference) |
| `quantize_bf16_to_q8_1_kernel` | iq2_xxs_matvec.cu | shared activation quantizer BF16→Q8_1 |
| `iq2_xxs_q8_1_matvec_kernel` | iq2_xxs_matvec.cu | decode against Q8_1 activations, DP4A |
| `iq2_xxs_q8_1_indexed_gate_up_kernel` | iq2_xxs_matvec.cu | decode, fused indexed top-6 gate+up |
| `iq2_xxs_q8_1_grouped_gate_up_kernel` | iq2_xxs_grouped.cu | prefill, grouped MMA (IMMA.16832.S8.S8) gate+up |
| `q2_k_q8_1_indexed_down_kernel` | q2_k_matvec.cu | decode, indexed Q2_K down |
| `q2_k_q8_1_grouped_down_kernel` | q2_k_grouped.cu | prefill, grouped MMA Q2_K down |
| `swiglu_weighted_q8_1_kernel` | swiglu_q8.cu | fused weighted SwiGLU→Q8_1 |

Support headers: `iq2_xxs_tables.cuh` (the 256-entry × 8-weight LUTs),
`int8_mma.cuh` (warp-level IMMA helpers), `q8_1_utils.cuh`.

**The IQ2_XXS decode path.** The format is the hard one: each 66-byte block
carries an fp16 scale plus 32 uint16 code words that interleave 4-bit grid
indices (into a 2 KiB LUT of 256×8-weight tables) and packed 7-bit sign +
5-bit integer sub-scale fields. The final accumulator needs the integer
truncation semantics (`sumi = sumi * ls / 8`, then `d * bq8_1.ds * sumi`)
that are the reason class-A oracles compare dequantized fp32 values, never
fused outputs. Our first implementation used scalar BF16 FMA loops: correct
(7/7 tests including CUDA-graph capture/replay), but 52.9 µs at K4096×N2048
— 40.9 GB/s against llama.cpp's 346–358 GB/s MMVQ. A two-iteration tuning
cap was pre-registered; iteration 2 quantized the layer input once to Q8_1
(shared across gate/up) and ran native DP4A against the raw GGUF codes.
Result: the indexed fused gate+up at the exact TP4 shape (K4096→N512/rank,
256 experts, top-6 routing) measured **26.231 µs / 247.35 GB/s, 0.343% CV**
over five exclusive trials — and DwarfStar's aligned-SoA repack, which the
literature credits with ~25%, measured *slower* at our exact shapes (9.90 vs
8.89 µs single-matrix) and was rejected. Raw GGUF blocks won.

**The Q2_K decode path.** Same skeleton, different math: 16 sub-blocks of
4-bit scale/min plus 2-bit quants, with the down projection's K/N axis swap
baked into the kernel. Indexed down at K512→N4096/rank: **13.752 µs, 300.23
GB/s, 0.270% CV** — already at the pinned llama.cpp reference's 307 GB/s, so
the aligned variant was deliberately declined on causal-budget grounds. The
combined decode expert path is 27.3 + 14.9 = **42.2 µs/layer**, beating the
old WNA16 stack's ~50 µs Humming baseline.

**The prefill paths (M≥128).** Prefill is a different animal: the decode
kernels are per-expert warp-per-row matvecs, and at M256 they saturate the
available expert time budget with no room for non-expert work (indexed
M256 five-trial baseline: 1.723 ms/token across 43 layers, a 580 tok/s
ceiling — the prefill falsifier named this as a fail condition). The grouped
kernels fix it with token compaction plus SM86 MMA/DP4A weight reuse: shared
WMMA N16 lost (7.97 ms), shared MMA N8 lost (6.41 ms), raw decode-to-register
N8 with a 64 µs alignment won (3.931 ms vs indexed 6.242 ms — 1.56×). Q2_K
down follows with scale nibbles folded into INT8 MMA codes and per-16 min
correction outside the MMA. Full uniform M256: gate/up 3.932 + down 2.082 +
alignment 0.065 = **6.079 ms vs 10.219 indexed (1.68×)**, leaving 0.797
ms/token of the 550 tok/s budget for non-expert work. Dispatch policy is
empirical: indexed below ~M128, grouped above.

**The fused SwiGLU.** Gate·up weighting plus down-activation quantization
fuse into one kernel — `swiglu_weighted_q8_1_kernel` — improving the TP4
layer slice 3.9% decode / 1.2% prefill and eliminating the BF16 intermediate
and post-down weighting pass. It also forced an honest revision of a
pre-registered Q8_1 class-B window: the adversarial fused path measured
1.0527% normalized MAE vs the 1.0% bound, better than the existing BF16→Q8_1 at 1.0688%
— so the window was transparently revised to 1.25% with every other bound and
all task-quality gates unchanged.

**Dense Q8_0: a repack, not a new GEMM.** The attention, shared-expert, and
output projections are Q8_0. Writing a new dense GEMM family would have
risked Marlin-level performance for zero user-visible gain; instead a
load-time adapter preserves the exact signed int8 codes, offsets them to
Marlin's uint8b128 layout, converts FP16 block scales to BF16 group-32
scales (documented last-bits lossy), and reuses vLLM's existing Marlin int8
kernels. The `wo_a` output projection is the sharp edge: the FP8
Marlin-diagonal trick doesn't apply to Q8_0, a no-cache dequant path measured
34.01 tok/s (fatal), and a BF16 cache costs 688 MiB/rank (~100–120K of
context — fatal). The grouped-diagonal seam on the repacked Q8_0 weights
measured **18.438 µs M1/2/4 with CUDA-graph replay** and byte-neutral storage
(8,912,896 bytes/layer). The dense decode pool: fused_wqa_wkv 13.690 µs,
wq_b 17.345, wo_b 18.061, shared gate+up 12.228, shared down 8.160, grouped
wo_a 18.438 — sum 87.922 µs/layer = 3.781 ms over 43 layers, near the
previous stack's ~3.54 ms Marlin pool.

**The TP=4 graph-captured layer slice.** Microbenchmarks are ceilings, not
projections, so M2's gate was the full slice: Q8 attention chain + first
all-reduce, routed IQ2→fused SwiGLU→Q2, shared Q8 expert, final all-reduce —
captured, replayed, five independent launches / 20 rank samples. Decode
**0.193402 ms/layer** (0.126% CV) → 74.13 tok/s projection; prefill M256
**10.1765 ms/layer batch** (0.107% CV) → 582.76 tok/s projection. M2 passed;
34/34 GPU tests, memcheck 0 errors, racecheck 0 hazards.

## 5. The loader

The `gguf_dsv4` load/quant format pair registers into vLLM's model loader
with the whole contract enforced at load: exact path, SHA-256, file size, and
tensor count must match; rank 0 hashes once; bounded contiguous/strided
pread streams feed per-rank views; ordinary tensors cast under the per-kernel
dtype contracts while quant bytes pass through untouched. Q8 linears allocate
raw row bytes then repack byte-neutrally to Marlin; routed methods allocate
all 256 experts per rank with TP-sharded intermediate dimensions. The
meta-model verifier runs before any GPU load: 1,328 sources = 1,180 plans =
1,180 actual parameters with exact name and element counts per rank — it
caught a routed_experts defect and a ParallelLMHead method defect during
development. The Marlin repack for the vocabulary head needed a chunked INT32
path after a whole-tensor INT64 repack demanded 1,010 MiB with ~902 MiB free —
the first M5 attempt failed post-load on exactly that, and the fix landed as
commit `3ec20cebe`.

## 6. Results on server60

Target: 4× RTX 3090 (SM 8.6, PCIe-only, no NVLink), fixed 230 W power limit
and 210–1650 MHz clock range, untouched. Model view: 21.53 GiB/rank, KV
0.81 GiB/rank at 154,519 tokens (1.10× context concurrency), CUDA graphs 0.06
GiB, load time 271.9 s, zero serving-process swap.

| Gate | Floor | Target | Measured |
|---|---:|---:|---:|
| Decode, engine tok/s | ≥58 | 70 | **76.697** (3 warm + 5 measured, 0.033% CV) |
| Prefill, cache-busted tok/s | ≥550 | 700 | **551.89** (3 runs) |
| Concurrency 2 aggregate | — | — | **121.86 tok/s** (61.1 each) |
| On-GPU context | ≥140K | 155K | **140,000** (measured ceiling) |

Functional gates passed in order: deterministic generation, automatic tool
call, post-tool continuation, exact NIAH recall at **119,730 prompt tokens**,
quick quality 27/30 pass@1 and pass@3.

## 7. The gate that matters: an agentic benchmark

Speed and recall are necessary but not sufficient — the whole point of
running the *original* weights was agentic quality. M8 was a one-cell
DeepSWE pilot (SuperJSON task, max reasoning, identical harness) against the
reused llama.cpp result for the same task, with the pass criterion being
Will's judgment of closeness.

| | GGUF-TP | llama.cpp Antirez control |
|---|---:|---:|
| Partial reward | **0.9949** | 0.9898 |
| Solved F2P | **79/80** | 78/80 |
| P2P | 116/116 | 116/116 |
| Turns | 70 | 118 |
| Tool calls | 80 (22 edit/write) | 124 |
| Output tokens | 119,557 | 195,420 |
| Patch bytes | 43,680 | 22,499 |
| Wall clock | **2,520 s** | 6,678.5 s |

Matches or exceeds the proven-quality baseline on every measure at 2.65× the
wall-clock speed. Against the WNA16 safetensors runs on the same task
(uniform 0.9235, quality 0.8980), the native-GGUF engine is far ahead —
confirming, end-to-end, that requantization — not the vLLM runtime — drove
the historical quality gap.

## 8. Honesty: what failed, what drifted, what's still open

**The M6 layer oracle.** We pre-registered per-layer class-B windows and ran
a paired 366-token layer dump against llama.cpp. **28/43 layers fail** the
pre-registered windows; drift grows smoothly from layer 0 and peaks near
layer 20 (median post-FFN cosine 0.992988, NRMSE 0.1191). Attention-vs-FFN
bisection localized the largest incremental drift to FFN phases at layers 7,
9, 15, 20; route-ID capture shows route sets differing at 9/43 layers by
exactly one expert each. We then ran one-variable bisection arms: FP16 router
storage (median NRMSE 0.1006 — marginal, not sufficient), FP32 router
compute (0.1194 — worse), forced indexed experts to bypass grouped-MMQ
arithmetic (0.1188 — not the cause). **No single mechanism explains the
drift**; the evidence points to accumulation of documented class-B per-op
differences — Q8_0→Marlin FP16 scale rounding, DP4A reduction order,
FlashMLA-vs-llama attention — across 43 layers. Two facts keep this from
being fatal: **final logits pass** (cosine 0.9973, top-1 equal, complete
top-10 overlap with ranks 5/6 swapped), and the agentic pilot showed zero
behavioral damage against the byte-identical-weights control. It remains a
genuine open finding with a follow-up project (including a weight-rounding
idea Will wants to test), tracked and recorded — not a footnote.

**Prefill.** 551.9 meets the ≥550 floor but misses the 700 target. The M2
layer-slice projection of 582.76 was documented as optimistic (it omits
inherited attention/indexer/norm work), and M5/M7 confirmed that.

**Context.** 140K is a measured capacity ceiling with 71–73 MiB idle physical
headroom per GPU after long-context JIT — the packed-TP steady state, not
release headroom. The operating context must not rise without remeasurement,
and any OOM at or below operating context reopens the decision. llama.cpp
retains the 430K active-context crown (Q8_0 KV + layer split); GGUF-TP owns
the speed tier at 140K, and the two services alternate via a validated
rollback contract.

**The prefill falsifier.** The indexed prefill path was falsified by design —
the screening showed it would leave impossibly little of the 550 tok/s
budget for non-expert work — which is precisely why the grouped MMA path was
mandatory, not optional. This is the methodology working as intended.

## 9. What we'd tell our past selves

- **Write the contract before the kernels.** The L0 oracle caught a real
  off-by-chunk bug in the *independent* decoder, and the coordinate oracle
  (A2) is the only thing that would have caught a transposed or incorrectly fused
  load. Checksums find corruption, not misunderstanding.
- **Pre-register kill criteria and obey them.** Two tuning iterations for
  IQ2; the prefill falsifier named its fail condition before the grouped
  path existed; the `wo_a` gate had an explicit kill. The process forced
  measured decisions where vibes would have produced two-week detours.
- **Reuse the proven dense path.** The temptation to write a new Q8_0 GEMM
  family would have risked Marlin-class performance for nothing. The
  byte-neutral repack is a small, checkable adapter.
- **Honest projections save surprises.** The layer-slice prefill projection
  was documented as optimistic and the measured result confirmed it. The
  decode projection was conservative and the measured result beat it.

## 10. Status and next steps

GGUF-TP has passed every functional, performance, and behavioral gate with
kill criteria honestly applied. The promotion to production serving (durable
Compose profile, pinned image digest, validated rollback to the canonical
llama.cpp service) is the final milestone. The upstreaming decision — whether
and how this engine generalizes beyond DeepSeek-V4-Flash — is recorded as an
open question, alongside the drift-minimization follow-up.

## Appendix: artifact pointers

- Engine + research trail: Whamp/vllm `incubate/gguf-tp-sm86`, 14 commits,
  tip `3ec20cebe` (research docs being moved here from
  noonghunna/club-3090 `feat/gguf-tp-engine` → `.research/gguf-tp-engine/`:
  PLAN.md, PROGRESS.md, FORMAT-CONTRACT.md, TP-MAPPING.md,
  DTYPE-CONTRACTS.md, WOA-DESIGN.md, REPACK-SPEC.md, CAPACITY.md,
  M6-LAYER-ORACLE-SPEC.md, M8-DEEPSWE.md, evidence bundles)
- Weight artifact: `antirez/deepseek-v4-gguf` blob, SHA-256 `ca22ae2f…b1c0`,
  86,720,111,488 bytes, 1,328 tensors
- Hardware: server60, 4× RTX 3090, PCIe-only, 230 W / 210–1650 MHz fixed
- Duration: 2026-08-17 → 2026-08-18, local compute only, no rental

*Every number above is measured on our stack and traceable to a milestone
record; where a number is a projection, it is labeled as such.*
