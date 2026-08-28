# Blog/Twitter notes — GGUF-TP engine

Context: Will asked (2026-08-18) for (1) a kernel inventory, (2) the significant
accomplishments, (3) a naming recommendation ("should we call this GGUF running
on vLLM? something else?"), as prep for a technical research blogpost, starting
with a long-form X post (premium account). Every number below is from
PROGRESS.md / PLAN.md / milestone docs or direct source inventory, not memory.

## 1. The one-sentence pitch

llama.cpp's exact quantized DeepSeek bytes (IQ2_XXS + Q2_K + Q8_0 GGUF),
executed natively — no GGML, no requantization — by from-scratch SM86 CUDA
kernels under vLLM's tensor-parallel runtime, on 4× RTX 3090s: 76.7 engine
tok/s decode (~2× llama.cpp), 551.9 cache-busted prefill tok/s, 140K on-GPU
context, and a DeepSWE agentic pilot that matches/exceeds the llama.cpp control
at 2.65× the wall-clock speed.

## 2. Kernel inventory (the direct answer)

**8 new native CUDA kernels** (`__global__`), 5 source files, plus 3 support
headers. All in `csrc/libtorch_stable/quantization/gguf_dsv4/` at
Whamp/vllm commit `3ec20cebe` (branch `incubate/gguf-tp-sm86`):

| # | Kernel | File | Path |
|---|---|---|---|
| 1 | `iq2_xxs_matvec_kernel` | iq2_xxs_matvec.cu | decode raw IQ2_XXS matvec (initial reference) |
| 2 | `quantize_bf16_to_q8_1_kernel` | iq2_xxs_matvec.cu | activation quantizer BF16→Q8_1 (shared) |
| 3 | `iq2_xxs_q8_1_matvec_kernel` | iq2_xxs_matvec.cu | decode IQ2_XXS against Q8_1 activations (DP4A) |
| 4 | `iq2_xxs_q8_1_indexed_gate_up_kernel` | iq2_xxs_matvec.cu | decode: indexed top-6 gate+up fused (raw GGUF blocks) |
| 5 | `iq2_xxs_q8_1_grouped_gate_up_kernel` | iq2_xxs_grouped.cu | prefill/batched: grouped MMA (IMMA.16832.S8.S8) gate+up |
| 6 | `q2_k_q8_1_indexed_down_kernel` | q2_k_matvec.cu | decode: indexed Q2_K down (scale nibbles folded into INT8 codes) |
| 7 | `q2_k_q8_1_grouped_down_kernel` | q2_k_grouped.cu | prefill/batched: grouped MMA Q2_K down |
| 8 | `swiglu_weighted_q8_1_kernel` | swiglu_q8.cu | fused weighted SwiGLU→Q8_1 (gate·up weighting + down quantization in one pass) |

Support headers: `iq2_xxs_tables.cuh` (256-entry × 8-weight LUTs, 2 KiB),
`int8_mma.cuh` (warp-level IMMA.16832.S8.S8 helpers), `q8_1_utils.cuh`.

Dense-path honesty: the Q8_0 attention/shared/output projections deliberately
reuse vLLM's **existing Marlin int8 kernels** after a byte-neutral load-time
repack (Q8_0 → int8 group-32, FP16→BF16 scales) — `q8_0_marlin.py` (186 lines).
The grouped-diagonal `wo_a` also rides Marlin's grouped seam. So the novel
kernels are the low-bit routed-expert family + fused SwiGLU + quantizer; the
dense GEMM family is a repack + reuse story, which is a feature (kept dense
perf at the proven Marlin level).

Sizes: new CUDA ~1,691 lines (incl. headers), loader/planner/io Python 1,532,
config+Marlin adapter 552, tests 1,630, 6 benchmark scripts. Branch: 14
commits, 36 files, +6,454 insertions over base pin `b7766cfe`.

Also inherited (NOT new here — prior WNA16 speed-stack campaigns): FlashMLA
sparse decode, hierarchical all-reduce, fp8_ds_mla KV cache, DSML tool-call
parser fix, SwiGLU semantics fix, Marlin-wo_a diagonal machinery, CUDA-graph
capture path, TP=4 launcher.

## 3. Significant accomplishments

1. **Byte-exact GGUF contract** — L0 oracle: pinned llama.cpp CPU
   `dequantize_row_*` (Whamp/llama.cpp `0379cf4bf`) vs independent
   NumPy-fp32 decoders, 10,000 random + adversarial blocks/format, **bitwise
   100%** (q8_0 / q2_K / iq2_xxs). Red→green: caught a real q2_K chunk-index
   bug in the independent decoder, not the contract.
2. **1,328-tensor inventory → 1,180 runtime targets** with exact per-rank
   name/element counts (TP4 verifier: 1,328 sources = 1,180 plans = 1,180
   actual parameters), 22,751,844,636 bytes = 21.1893 GiB/rank, zero
   overlap; GGUF SHA-256 `ca22ae2f…b1c0` (86.72 GB blob) verified at load,
   hash-once-on-rank-0, fail-closed.
3. **8 from-scratch SM86 kernels** for the formats llama.cpp's TP can't
   express and vLLM's GGUF path can't execute: decode indexed + prefill
   grouped paths for IQ2_XXS and Q2_K, fused weighted SwiGLU→Q8_1, shared
   BF16→Q8_1 activation quantizer. 34/34 GPU tests, memcheck/racecheck clean.
4. **Decode expert path faster than the old stack**: IQ2 gate+up 27.309 µs +
   Q2 down 14.898 µs = 42.2 µs/layer vs ~50 µs WNA16 Humming baseline.
5. **Q8_0 dense runs on Marlin** via byte-neutral repack (fatal gate passed:
   no BF16 cache — 688 MiB/rank would have cost ~100–120K context; the
   18.4 µs grouped-diagonal wo_a avoided the 34 tok/s no-cache dequant path).
6. **TP=4 graph-captured layer slice** projection 74.13 decode / 582.76
   prefill tok/s → **measured 76.697 decode (3 warm + 5 measured, 0.033% CV)
   and 551.89 cache-busted prefill** (floor 550 pass, target 700 miss).
7. **140K on-GPU context** (154,519 KV tokens @ 1.10× concurrency, 0.81 GiB
   KV/rank, 21.53 GiB model/rank, load 271.9 s), **exact NIAH recall at
   119,730 prompt tokens**, deterministic canaries, auto tool-call,
   post-tool continuation, zero swap.
8. **M8 one-cell DeepSWE pilot (the quality gate): GGUF-TP partial 0.9949
   (F2P 79/80, P2P 116/116) vs llama.cpp control 0.9898 (78/80, 116/116)**
   — matches/exceeds on every measure, 2.65× faster wall-clock (2,520 s vs
   6,678 s). Vs WNA16 same-task runs (0.9235 / 0.8980): the requantization
   was the quality killer, not the runtime.
9. **Validation ladder executed honestly**: L0 bitwise → A2 coordinate-aware
   mapping oracle → per-kernel class-B windows → TP4 layer slice → full-model
   per-layer oracle vs llama.cpp → NIAH → agentic pilot. M6 layer-oracle gate
   FAILED per pre-registered windows (see §4) and was recorded, not hidden.
10. **2-day campaign** (2026-08-17 → 08-18, from goal start to pilot result)
   on local compute, no rental, at fixed 230 W / 1650 MHz GPU safety policy.

## 4. Honest caveats (must appear in the blog post)

- **M6 per-layer drift**: 28/43 layers outside pre-registered class-B windows
  (median post-FFN cos 0.992988, NRMSE 0.1191), growing smoothly from layer 0;
  attention-vs-FFN bisect localized largest increments to FFN phases at
  layers 7/9/15/20; route sets differ at 9/43 layers by exactly one expert.
  One-variable bisect arms (FP16 router storage, FP32 router compute, forced
  indexed experts) all rejected. Conclusion: accumulation of documented
  class-B per-op differences (Q8_0→Marlin FP16 scale rounding, DP4A reduction
  order, FlashMLA-vs-llama attention). **Yet final logits pass** (cos 0.9973,
  top-1 equal, complete top-10 overlap with ranks 5/6 swapped) and the
  agentic pilot shows no behavioral damage vs byte-identical-weight llama.cpp.
- **Prefill**: 551.89 meets the ≥550 floor but misses the 700 target; the
  582.76 layer-slice projection was optimistic as documented (omitted
  inherited attention/indexer/norm work).
- **Context is a measured ceiling, not headroom**: 71–73 MiB idle physical
  headroom per GPU after long-context JIT (accepted by Will as normal for a
  packed TP profile); operating context must not rise without remeasurement.
- llama.cpp retains the 430K active-context advantage (Q8_0 KV + layer split);
  GGUF-TP owns the speed tier at 140K.
- Q8_0→Marlin conversion is documented last-bits-lossy in scales (FP16→BF16
  rounding); IQ2_XXS/Q2_K bytes are never re-encoded.

## 5. Naming recommendation

**Keep the engine name GGUF-TP** (already the project name; precise about the
two things that are actually new: native GGUF execution + tensor parallelism).
In prose, describe it as **"a native GGUF inference engine inside vLLM"** —
never plain "GGUF on vLLM", because upstream vLLM already has GGUF support and
that phrase reads as the existing converter path (dequantize-to-float), which
is exactly what this is not.

Recommended one-liners:
- "Native GGUF tensor-parallel inference for DeepSeek-V4-Flash" (plan title)
- "llama.cpp's quantized bytes, vLLM's runtime, from-scratch kernels"
- "GGUF executed, not converted"

Distinguishing claims (all true here, false for the alternatives):
- GGML/llama.cpp: cannot TP this model (CUDA-TP doesn't fit 24 GB cards;
  row-split breaks grouped-attention graphs — audited).
- Upstream vLLM GGUF support: loads the file but dequantizes weights into
  its normal float formats (fp16 internal kernels); officially "highly
  experimental," migrated to the OOT vllm-gguf-plugin; tested set is
  Q6_K/Q8_0/IQ4_XS/Q4_K_M/Q4_0 — IQ2_XXS/Q2_K not in it; TP is supported in
  the plugin, so the claim is "converts, doesn't execute the packed bytes,"
  not "single-GPU only" (verified via web 2026-08-18).
- DwarfStar (antirez/ds4): from-scratch engine, same GGUF family, but its
  TP topology needs 48 GB cards; ours is the TP-by-quarters design for 24 GB.
- WNA16 safetensors on vLLM: same runtime, requantized weights, measurably
  worse agentic quality (0.92–0.90 vs 0.99 partial on identical tasks).

## 6. X post draft (long-form, premium) — see TWITTER-POST.md
## 7. Blog post draft — see BLOG-POST.md
