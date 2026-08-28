# PLAN: Native GGUF tensor-parallel inference for DeepSeek-V4-Flash on 4× RTX 3090

Status: execution v7. M0/M1 passed. M2 IQ2 gate/up fragment passed: native
Q8_1+DP4A indexed top-6 gate/up at the corrected TP4 shape K4096→N512/rank
measured 27.309 µs for captured quantize+compute and 247.35 GB/s over five
exclusive trials (0.343% kernel CV); 15/15 SM86 correctness/graph tests pass.
Aligned IQ2 is rejected (slower at N512); raw GGUF blocks are selected. Pinned
source corrected the loader contract: all 256 experts exist on each rank with
the intermediate dimension /4 (w13 N512; w2 K512), not 64 whole experts/rank.
Proceed to M3 Q2_K down, then return for dense/wo_a and the M2 layer-slice
completion gate. Earlier plan changes: two changes from v4: (1) **the "wrap llama.cpp kernels"
route is cut entirely** — Will's original direction stands as the only route:
rewrite the expert kernels from scratch, vLLM-native, reading GGUF bytes
directly, with testing to verify accuracy. The wrap route entered via review
suggestions, never matched the requirement, and was demolished by the third
review. (2) **DwarfStar (antirez/ds4) absorbed as source material** with
exact code pointers (§4.6) — primarily its aligned-SoA weight repack, which
attacks the exact IQ2_XXS bandwidth deficit we measured on our own 3090s.

Review lineage: Grok 4.6 xhigh, Claude Fable 5 medium, GPT-5.6 Sol xhigh (all
2026-08-17, archived under `reviews/`). Findings that survive the route cut —
tensor-level TP mapping, coordinate-aware load oracles, tokenizer bootstrap
pin, wo_a kill-gating, executable DeepSWE statistic, capacity-from-table —
are retained.

## 1. Objective and decision frame

Run the **exact proven-good Antirez GGUF bytes** (IQ2_XXS routed gate/up, Q2_K
routed down, Q8_0 attention/shared/output, F16/F32/I32 control tensors)
inside a **vLLM-style tensor-parallel runtime** at vLLM-class speed, on
server60's four RTX 3090s. No requantization, no lossy conversion of the
routed experts. This is an inference-engine ASIC: one model, one artifact
family, one hardware target, zero backward compatibility.

**Why this is worth doing:** the model is the most intelligent thing that
fits our VRAM; the GGUF encoding is the only quantization of it that has
held up under our hardest gate — the post-DSML-fix final 12-task DeepSWE
comparison measured GGUF-via-llama.cpp at 6 strict solves / 96.57% partial
reward versus WNA16 requant at 0 solves / 80.62%; llama.cpp cannot
tensor-parallel it (audited: CUDA-TP doesn't fit 24 GiB cards, row-split
breaks grouped-attention graphs); vLLM reached 74.98 decode tok/s on
inferior weights while llama.cpp is structurally capped at ~38–39 engine
(measured). Same-architecture weight updates load through the same pinned
contract (a new inventory pass, not a rewrite). And "GGUF in tensor
parallelism" is a standing community ask nobody has shipped.

**Precedent that this class of engine works:** DwarfStar (antirez/ds4) is a
from-scratch one-model-family engine that consumes the same GGUF family
natively — not a GGML derivative — and ships its own IQ2_XXS/Q2_K/Q8_0 CUDA
kernel tree, a load-time aligned-repack pipeline, and CUDA tensor
parallelism. Its exact topology fails on our 24 GiB cards (§4.6 negatives),
but its existence and its kernel/repack designs are directly usable
reference material.

**Decision thresholds** (decode = engine tokens/s; client-wall ~12% lower,
reported alongside):

| Outcome | Minimum success | Target | Stretch |
|---|---:|---:|---:|
| Single-stream decode (engine) | ≥ 58 tok/s (~1.5× llama.cpp engine 38.4) | 70 tok/s | 75 tok/s (WNA16 stack parity) |
| Cache-busted prefill | ≥ 550 tok/s | 700 tok/s | 887 tok/s (inherited-stack parity) |
| **On-GPU unique-request context** | ≥ 140K *point estimate subject to §10 sensitivity* | 155K | 170K+ via levers |
| Prefix-reuse offload tier | present, measured hit-rate ≥ 60% on repeated-prefix workload | — | — |
| Quality | **one-cell DeepSWE pilot** §6 / `M8-DEEPSWE.md`: Will judgment vs reused llama.cpp baseline | quick-pack within noise of GGUF baseline | historical 12-task 6/12 anchor (reference only) |
| Correctness | class-A/A2 dequant+mapping oracles; known-delta paths within pre-registered windows | deterministic canaries | NIAH exact recall at achieved on-GPU context |

**Context, stated honestly:** the 16 GiB host tier is an eviction/prefix-
restore tier — the promoted compose records it "is not part of the measured
230K performance path." It does **not** buy active context on top of the
on-GPU pool. 430K-class active context remains llama.cpp's exclusive
advantage (Q8_0 KV + layer split). During evaluation the two services
alternate via the validated rollback contract.

Guardrails (inherited, non-negotiable): GPU safety policy 230 W / 210–1650 MHz
untouched; one causal variable per experiment; zero-swap final states;
verified rollback to the canonical llama.cpp service; every claim measured.

## 2. Evidence inventory — what exists, correctly labeled

Three evidence grades: **proven-here** (measured on our stack),
**proven-adjacent** (real measurement, different kernel/stack — supports
feasibility, not performance), **unmeasured** (assumption to retire).

| Component | Grade | Evidence |
|---|---|---|
| IQ2_XXS/Q2_K dequant semantics **in the pinned llama.cpp source** | proven-here (source); plan transcription gated by the L0 oracle | pinned Whamp/llama.cpp `0379cf4bf`; the contract is generated from source and gated by oracle, never trusted from prose |
| IQ2_XXS decode matvec leaving bandwidth on the table | proven-here | our exact-shape MMVQ microbench on a 1650 MHz 3090: IQ2_XXS 346–358 GB/s, Q2_K 307 GB/s, vs Q8_0 713 GB/s; Nsight Compute: IQ2_XXS 64% SM throughput / 36.2% occupancy |
| **Root cause of that deficit = 2-byte alignment + split loads** | proven-adjacent (his hardware) — hypothesis for ours | DwarfStar `cuda/mmq/test/proto_iq2_aligned.cu:1-22`: block_iq2_xxs is 66 bytes → code stream 2-byte aligned → every 32-bit weight word costs two 16-bit loads (`get_int_b2`); his measured ~142 GB/s vs ~200 ceiling |
| **Aligned-SoA repack recovers it** | proven-adjacent | DwarfStar `ds4_cuda.cu` MMQ-tier comment (~line 1081): aligned-SoA/D2R/producer-quantized tiers worth "a further ~25% on top of the ~2.5x this tier gives"; repack pipeline is production code (`cuda/mmq/ds4_repack.cu`) |
| Batched (n_tok ≥ 2) IQ2_XXS/Q2_K/Q8_0 MoE + dense kernels in production | proven-adjacent | DwarfStar `ds4_cuda.cu:~1075-1088` ported Entrpi/ds4 MMQ tier: dense Q8_0 GEMMs + IQ2_XXS gate/up + Q2_K down routed-MoE for prefill/batched-verify, validated against official continuation vectors |
| TP-sharded indexed grouped-MoE with in-mainloop unpack | proven-here for int-unpack W2/W4 | Humming extensions `e5a8452c7`, `dd2d1fd6`; 7-case SM86 oracle |
| Whole non-MoE stack at speed on FP8 weights | proven-here | 74.98 tok/s composite |
| Same stack on Q8_0 weights (dense + `wo_a`) | **unmeasured — largest unmeasured kernel share** | measured at M2 |
| Correctness-oracle methodology | proven-here | WNA16 oracle ladder; DwarfStar's continuation-vector regression is a convergent independent precedent |
| KV-offload tier inertness | proven-here as an unused reservation | eviction-pressure behavior unmeasured |

**Scope reduction:** only IQ2_XXS and Q2_K need codebook paths. Q8_0 becomes
int8 group-32 via a last-bits-lossy repack (§4.3). F16/F32/I32 control
tensors pass through **subject to per-kernel dtype contracts** (§4.5) and
the tensor-level TP mapping (§4.7).

## 3. The artifact contract (pinned; M1 verifies byte-level)

GGUF: `antirez/deepseek-v4-gguf`, `DeepSeek-V4-Flash-IQ2XXS-w2Q2K-AProjQ8-
SExpQ8-OutQ8-chat-v2-imatrix-0731.gguf`, blob SHA-256 `ca22ae2f…b1c0`,
86,720,111,488 bytes, verified on server60. MTP is a separate 3.6 GiB file,
not in this blob — no MTP lever.

Tensor classes (M1 produces the authoritative per-tensor inventory):

| Class | Format | Notes |
|---|---|---|
| Routed gate/up (`ffn_gate_exps`/`ffn_up_exps`) | IQ2_XXS | GGUF axis order `{K, N, E}` |
| Routed down (`ffn_down_exps`) | Q2_K | **GGUF axis order `{N, K, E}`** — differs from gate/up |
| Attention projections, shared experts, `output` | Q8_0 | `wq_a` and `attn_kv` stored **separately** in GGUF |
| `token_embd` | F16 | **vocab-sharded at runtime** (VocabParallelEmbedding) |
| Router, indexer, compressor, HC | F16 | per-tensor TP rules from §4.7 table |
| Norms, sinks, biases | F32 | no downcast |
| `ffn_gate_tid2eid` (early layers) | I32 | `{n_expert_used, n_vocab}` hash table |

**IQ2_XXS layout (source: `vecdotq.cuh:985-1014`):** block = fp16 `d` +
32 uint16. Per 32 weights (8 bytes): two uint16 = **4 grid-index bytes**
(256-entry × 8-weight LUT, 2 KiB); the following uint32 = **four 7-bit sign
fields** plus a **5-bit integer sub-scale** `ls = aux32 >> 27 | 1`, applied
with integer truncation: `sumi = sumi * ls / 8`, then `d * bq8_1.ds * sumi`.
Q2_K = fp16 `d`+`dmin`, 16 sub-blocks of 4-bit scale/min + 2-bit quants.
Q8_0 = fp16 `d` + 32×int8. CPU `dequantize_row_*` is the golden oracle;
integer-truncation semantics are why class A covers **dequantized values in
fp32** while fused outputs live in class B. M1 also pins the tokenizer/config
contract (§4.4).

## 4. Architecture: keep / adapt / new

**Base pin:** branch `incubate/gguf-tp-sm86` in Whamp/vllm from the
speed-stack tip `b7766cfe` (tree `6354125a`) — DSML fix + SwiGLU + Marlin-wo_a
+ FlashMLA + hierarchical all-reduce + KV-offload repair, measured 74.98/887.
club-3090 `feat/gguf-tp-engine` (this worktree) owns deployment, evidence,
plans.

**Keep (proven on the pinned base):** vLLM scheduler + continuous batching;
TP=4 launcher; FlashMLA sparse decode; hierarchical all-reduce; DSML parser
fix; SwiGLU semantics; fp8_ds_mla KV + offload tier; CUDA-graph capture path;
tokenizer/chat path **subject to §4.4**.

**Adapt / new:**

1. **GGUF-native loader**: mmap, verify blob SHA-256, GGUF tensor-name →
   vLLM module mapping via the §4.7 tensor-level table, per-rank packed
   views, checksum fail-closed against the M1 inventory. IQ2_XXS/Q2_K/
   F16/F32/I32 bytes are never re-encoded by the loader.
2. **Aligned-SoA is evidence-gated, not mandatory.** DwarfStar's layout and
   checksum discipline remain the tested comparison. IQ2 results reject it
   for production: aligned DP4A at exact TP4 N512 was slower than raw (9.90 vs
   8.89 µs single-matrix), while indexed raw reaches 247.35 GB/s. Therefore
   IQ2 loads raw GGUF blocks directly. M3 independently A/Bs Q2_K against
   `proto_m2_q2k.cu`; only a measured Q2 winner may create a derived,
   checksum-gated repack. The original GGUF is never mutated.
3. **Q8_0 → int8 group-32 repack (documented lossy-in-last-bits):** exact
   int8 codes; fp16 block scale → CT scale (fp16 where the kernel allows,
   else bf16-rounded), Marlin tile-packing, uint8b128 offset. Tolerance
   oracle vs Q8_0 dequant+GEMM; excluded from the bit-exact ladder. Dense
   GEMV/GEMM perf measured at M2. **A minimal repacker + `wo_a`
   serving-shape prototype is measured in M2** so the go/no-go uses
   measured numbers, not sketches.
4. **`wo_a` Q8_0 output projection:** FP8 Marlin-diagonal doesn't apply to
   Q8_0; no-cache fallback measured 34.01 tok/s (fatal); BF16 cache 688
   MiB/rank. M1 scopes a Q8 grouped-Marlin (or equivalent) diagonal path
   with VRAM delta; **M2 measures a real serving-shape prototype.** Kill:
   if the only working options are BF16 cache or ~34 tok/s dequant, stop.
5. **Routed-expert kernels — single route: rewrite from scratch,
   vLLM-native.** `FusedMoEExperts`-style `torch.ops` operations, registered
   capture-safe from day one (current-stream argument, caller-owned
   workspaces, no internal pools — the wrap-era ABI lessons are design
   requirements now). The public input contract is bf16. M2 evidence permits
   one explicit internal representation change: quantize that layer input once
   to Q8_1 and share it across gate/up experts, then run native DP4A against
   GGUF/aligned-SoA codes. Quantization is separately timed/class-B-gated and
   is never hidden. Correctness reference: llama.cpp CPU
   `dequantize_row_*` + reference GEMM (class A/B oracles). Kernel-by-kernel
   with DwarfStar's vendored kernel tree as permissively-licensed
   reference for grid/LUT/dp4a math and tiling (`cuda/mmq/vecdotq.cuh`,
   `mmq.cuh`, `mmvq.cuh`, `mmid.cu`, `quantize.cuh`, `iq2_host_tables.h`).
   Separate decode (M=1–4, warp-per-row matvec style) and prefill/batched
   (n_tok ≥ 2, MMA path) kernels, mirroring DwarfStar's split.
6. **Per-kernel dtype contracts for replicated families:** M1 inventories
   the dtype/layout contract of every kernel consuming each family
   (compressor fp32 assert; indexer fused-quant; merged-GEMM fp32 hand-off;
   activation dtype bf16). Any transform or cast becomes a documented
   conversion with a class-B window and a capacity-table line. The
   **router** cast policy is explicit (F16 → bf16 is lossy where top-6
   tie-breaks live; prefer fp32/fp16-native if the kernel accepts it).
7. **Tensor-level TP mapping table (replaces family rules):** M1 produces,
   per GGUF tensor: logical shape and axis order, destination parameter
   (incl. fused slots such as `fused_wqa_wkv` = GGUF `wq_a` + `attn_kv`
   stacked in fixed order), TP axis (e.g. token_embd vocab-sharded;
   `fused_wqa_wkv` `disable_tp=True` replicated; wq_b/wo_a/wo_b distinct
   column/row rules), rank slice, quant-block row axis, runtime dtype,
   post-load storage. Capacity is derived from this table.

**Quant-method config**: `gguf_dsv4` expressing experts = GGUF-native,
dense Q8_0 = CT int8-g32, control = per-kernel-contract passthrough.

**Retained:** the compressed-tensors expert path stays for the same-tree
WNA16 A/B that attributes misses to the new route rather than to
FlashMLA/graphs/AR regressions.

### 4.4 Tokenizer/config source of truth

The pinned stack auto-selects `tokenizer_mode="deepseek_v4"` from the
architecture and that tokenizer overrides `apply_chat_template` with custom
`encode_messages` (DSML, tools, thinking) — **the GGUF bootstrap must pin
this mode explicitly before module/tokenizer construction**, and M1 adds
text/API→token-ID golden tests for ordinary chat, high/max reasoning, tool
calls, and post-tool continuation. Token-ID-pinned probes remain for kernel
isolation only, never as tokenizer evidence. M1 also diffs GGUF KV metadata
vs HF config (RoPE theta/YaRN, compress ratios, SWA window) and names the
authoritative source.

### 4.5 Router cast policy

F16 → bf16 loses tie-break precision in top-6 routing; the default is to
keep router math in fp16/fp32 where the consuming kernel accepts it, and
any cast is a documented class-B window, not an accident.

### 4.6 DwarfStar borrowings — exact pointers and negatives

Pinned at antirez/ds4 commit `84cc882` (current main at time of study; our
Aug 13 audit pinned the same commit). MIT license with retained GGML
copyright notice — any adapted kernel carries comment-level attribution to
both ds4 and llama.cpp/GGML.

**Borrow (with pointer — what we take):**

| Pointer | What it is | What we take |
|---|---|---|
| `cuda/mmq/test/proto_iq2_aligned.cu:1-22` | The alignment hypothesis: 66-byte IQ2_XXS blocks → 2-byte-aligned code stream → two 16-bit loads per 32-bit word (`get_int_b2`) → ~142 of ~200 GB/s; aligned SoA (d[] halves separate, qs[] 64B-aligned per block) restores full-width loads at identical byte count; A/B protocol at the exact production decode shape; correctness = integer math identical, float accumulation-order tolerance | The hypothesis, the A/B microbench protocol, and the correctness framing for our §6 class A/B split |
| `cuda/mmq/ds4_repack.cu` + `ds4_repack.h` | Production aligned-artifact layout library (extracted from a weight-server); contract: producers must stay bit-identical, FNV repack-hash lines are the gate | The load-time-repack design and its hash-gate discipline for our §4.2; we produce ours in-process at model load |
| `ds4_cuda.cu:~1075-1088` | Production MMQ tier (ported from Entrpi/ds4 commits `39d3877c`, `a56e07a5`, `944482d5`): dense Q8_0 GEMMs + IQ2_XXS gate/up + Q2_K down routed-MoE for n_tok ≥ 2; decode untouched; FP32 reduction-order drift validated against official continuation vectors; `DS4_CUDA_MMQ=0` escape | Precedent that batched IQ2_XXS/Q2_K MoE kernels are production-viable; the continuation-vector drift-validation method; env-escape pattern |
| `ds4_cuda.cu:~1081` | "aligned-SoA / D2R / producer-quantized tiers … a further ~25% on top of the ~2.5x this tier gives" | Sized expectation for the aligned tier — proven-adjacent, must be re-measured on SM86 |
| `cuda/mmq/` tree (`vecdotq.cuh`, `mmq.cuh`, `mmvq.cuh/.cu`, `mmid.cu/.cuh`, `quantize.cu/.cuh`, `common.cuh`, `mma.cuh`, `ggml-common.h`, `test/iq2_host_tables.h`, `test/test_mmq_parity.cu`, `test/proto_m2_q2k.cu`, `test/proto_q8_aligned.cu`, `test/proto_q8_warp8.cu`) | Vendored, adapted, MIT-licensed kernel implementations of exactly our three formats, with parity tests | Reference implementations for grid/LUT/dp4a math, tiling, activation quantization, and parity-test structure; adapted, not linked |
| `tests/test-vectors/` + `QA_BEFORE_RELEASES.md` | Official-model continuation vectors as regression checks | Method reinforcement for our class-B full-model forward windows |
| `ds4-server` micro-batching (README "CUDA multi-GPU" section) | Grouped decode across resident sessions for aggregate throughput | Deferred: informs the later parallel-2/4 aggregate work, out of scope for single-stream milestones |

**Avoid (design negatives, established by our 2026-08-13 audit at the same
commit — off-box source audit, never run on our hardware):**

- **DP=2 × TP=2 topology:** two pipeline stages, each a GPU pair; routed
  experts split 50/50 within the pair, but attention/router/shared-expert
  weights **replicated to both GPUs of each pair** → 87.446 GiB aggregate
  weight residency vs 80.759 GiB payload (+6.686 GiB duplication) on cards
  that already barely hold a quarter-share each.
- **Hard 2 GiB per-GPU slab reserve** (`ds4_cuda.cu:~26002` reserve =
  max(2 GiB, 5% free)) → projected slabs 22.09–22.13 GiB exceed a 24 GiB
  board before any context is chosen. Documented four-card Q2 endpoint is
  48 GB cards, not ours.
- **F32 compressed-KV rows on CUDA** (`ds4.c:14945-14958`,
  `DS4_GPU_ATTN_COMP_CACHE_F16` is Apple-only) → ~4× our Q8 KV footprint;
  long context on discrete CUDA needs smaller chunks.
- **Q2 grouped routed shapes use an exact-but-slow fallback** in current
  ds4 (native grouped kernels exist for Q4_K) — consistent with our plan
  writing native Q2_K kernels rather than accepting fallbacks.

Our TP-by-quarters design (all weight families sharded, no pair
replication) is the deliberate inversion of these negatives.

## 5. Performance reasoning

- **Linear time-mix projection is a screening bound only.** The replacement
  swaps expert *and* dense kernels simultaneously inside stream-parallel,
  graph-captured, NCCL-overlapped decode; the base stack's own benchmark
  history shows M-regime transfer errors up to 5.6×. The real go/no-go is a
  **TP=4 graph-captured decoder-layer slice** containing the Q8 dense
  kernels, the rewritten expert operation, and the real all-reduce. For
  prefill, benchmark the **observed M distribution** from the fresh trace,
  not one nominal M≈256 point.
- Screening bound arithmetic (re-anchored on a fresh nsys trace of the
  running 74.98 stack at M0): experts ~15–18% of kernel time (~2.0–2.4 ms
  of 13.3 ms/token), dense Q8_0 replacement ~23% share unmeasured — if
  dense runs 30% slower, ~0.9 ms of budget is gone before experts spend
  anything. Realistic expert tolerance to hold 58 tok/s: **~2.2–2.6×
  slower than W2 Humming**, contingent on dense-path near-parity.
- **Aligned-SoA is conditional, not assumed.** Iteration 1 measured only
  +2.0% for scalar BF16 FMA (52.86 µs, 40.9 GB/s at K4096×N2048), proving
  execution/lookup — not raw alignment — dominated that kernel. Iteration 2
  repeats raw/aligned under Q8_1 DP4A, the path to which DwarfStar's ~+25%
  adjacent evidence actually applies. If DP4A does not move the mediator
  enough, the two-iteration kill criterion triggers.
- **M2 measured components (all on a 3090):** fresh-trace mix; rewritten
  expert-op time at decode shapes (raw-layout and aligned-SoA variants
  A/B'd per the proto protocol); dense Q8_0-g32 Marlin GEMV; `wo_a`
  prototype time.
- **Prefill falsifier:** dense Q8_0 GEMM across the observed M distribution;
  projection < 550 tok/s is a named failure before bring-up.
- Overall decode estimate band: **55–75 tok/s**, estimate until M7.

## 6. Correctness doctrine

- **A. Bit-exact (no tolerance):** IQ2_XXS/Q2_K **dequantized weight values
  in fp32** vs llama.cpp CPU `dequantize_row_*`, random + adversarial
  corpora (sign patterns, extreme scales, LUT boundary indices, sub-scale
  extremes). The aligned-SoA repack joins this class: **repacked artifact
  must decode to identical fp32 values** as the raw layout (same integer
  math, layout transform only), gated by the repack content hash.
  Fused outputs are class B by construction.
- **A2. Coordinate-aware mapping oracle:** byte checksums cannot catch a
  transposed or mis-fused load (routed down `{N,K,E}` vs gate/up `{K,N,E}`;
  `fused_wqa_wkv` slot order). For every tensor family and TP boundary,
  sample `(expert, output row, input column)` coordinates, derive the GGUF
  byte offset independently, decode, and compare the destination's logical
  value — covering first/last block, first/last rank, fused-slot
  boundaries, and hash-table (`tid2eid`) rows.
- **B. Known-delta (pre-registered windows):** Q8_0-repack GEMM vs dequant+
  GEMM; rewritten expert op vs reference GEMM; full-model forward vs
  llama.cpp on fixed prompt sets (KL bound stated before first run;
  continuation-vector style regression as a second anchor, per DwarfStar
  `ds4_cuda.cu:~1086`); fp8_ds_mla KV / FlashMLA / hier-AR /
  replicated-family casts each with their own window.
- **C. Determinism:** CUDA-graph replay vs eager equality; AR rank-order
  consistency; graph-size sweep for pointer aliasing.
- **D. End-to-end:** deterministic canaries; tool round-trip and post-tool
  continuation; NIAH exact recall at achieved on-GPU context; **DeepSWE
  one-cell pilot** (`M8-DEEPSWE.md`): GGUF-TP runs **one task, one seed
  (rep0)** on the locked SuperJSON harness; compare against **reused**
  llama.cpp results for the same task. Pass = Will judges closeness adequate
  given single-run variance. The **≥72-cell multi-seed grid is cancelled**
  (Will 2026-08-18); do not run or schedule it. A single SuperJSON run is
  the M8 quality gate for this project, not smoke-only.

Ladder L0→L6, adversarial review loops (1 implementer + 2 reviewers on diff +
format contract), checksums on every tensor view, oracle failures batched as
work queue — unchanged.

## 7. Execution methodology

Unchanged: FORMAT-CONTRACT.md generated from source before code and gated by
the L0+A2 oracles; trial-first; loader cutover on its own branch; process
fixes over hand fixes; worktree/commit discipline; one causal variable;
Nsight attribution before kernel tuning; unprofiled end-to-end numbers for
claims. All kernel gates on a server60 RTX 3090. **No rental compute
planned.**

## 8. Milestones, gates, kill criteria

| # | Deliverable | Gate | Kill / pivot | Est. |
|---|---|---|---|---|
| M0 | Worktrees, base pin `b7766cfe`, fresh nsys trace of the 74.98 stack, baseline re-anchor | pins + trace recorded | — | 1 d |
| M1 | `FORMAT-CONTRACT.md` from source; per-tensor inventory; §4.7 tensor-level TP table; per-kernel dtype contracts; tokenizer bootstrap pin + text-level golden tests; `wo_a` design + VRAM delta; aligned-SoA repack spec (layout + hash gate); capacity table with every delta sized in MiB → tokens | L0 class-A oracle 100%; A2 mapping-oracle design; inventory matches blob; capacity table shows ≥140K or levers whose summed size closes the gap | inventory mismatch → re-scope; `wo_a` no viable design → **stop**; capacity gap unclosable → re-decide scope with Will | 2–4 d |
| M2 | Rewritten expert kernels (IQ2_XXS first, decode + batched), **raw-layout vs aligned-SoA A/B** at exact serving shapes; capture-safe `torch.ops` registration, capture/replay M=1–4 TP=4 + aliasing sweep; minimal Q8 repacker + dense GEMV + `wo_a` prototype measured; prefill M-distribution bench; **TP=4 graph-captured decoder-layer slice** as the real projection | screening projection ≥ 58 decode and ≥ 550 prefill from the layer slice, dense and `wo_a` components measured | causal matrix: expert-kernel miss after **2 tuning iterations** (incl. aligned-SoA variant) → **stop**; dense/`wo_a` miss → dense redesign or **stop**; one-metric-only miss → explicit decision with Will | 1–2 wk |
| M3 | Q2_K kernels on the same skeleton + aligned variant; productionize the repack artifact path | same gates as M2 per kernel | same | 1 wk |
| M4 | Production GGUF loader + repack + Q8_0 repack + `gguf_dsv4` config + `wo_a` path; CPU tests; checksum fail-closed; **calendar kill: 10 working days** | full-tensor mapping test incl. A2 boundaries; byte-identity assert on packed IQ2_XXS/Q2_K views; repack hash gate | calendar breach → descope review | 1–2 wk |
| M5 | server60 TP=4 bring-up (authorized window; validated rollback) | class-A/B full-path oracle on-GPU; NCU dispatch; readiness | repeated OOM/instability → capacity re-plan | 0.5–1 wk |
| M6 | Per-layer vs llama.cpp (class B); canaries + NIAH at achieved context | pre-registered windows pass | unexplained divergence → bisect | 3–5 d |
| M7 | Matched perf campaign; same-tree WNA16 A/B attribution | ≥58 engine decode, ≥550 prefill, ≥140K on-GPU, zero swap | miss → keep llama.cpp canonical; publish | 3–5 d |
| M8 | Quality: quick pack + **one-cell DeepSWE pilot** (`M8-DEEPSWE.md`) | Will judges GGUF-TP close enough vs reused llama.cpp baseline on pilot task | divergence → component bisect; **do not run cancelled 72-cell grid** | pilot run only |
| M9 | Promotion package; open-source decision | Will's approval; healthy final service | — | 2–3 d |

Effort envelope: **7–10 weeks if gates pass first-try**; M2/M3 iterations,
M6 bisects, `wo_a` redesign are the contingency sources — kills bound the
downside, not the calendar. (DeepSWE 72-cell grid cost removed per Will
2026-08-18.)

## 9. Risk register

| Risk | Likelihood | Impact | Mitigation |
|---|---|---|---|
| `wo_a` Q8_0 path slow or fat | medium-high | fatal if unsolved | M1 design + M2 prototype; kill |
| Dense Q8_0 path slower than FP8 Marlin | medium | eats the 58 budget | measured at M2; feeds go/no-go |
| Aligned-SoA gains don't transfer from his silicon to SM86 | medium | expert budget stays tight | raw-vs-aligned A/B is an M2 gate, not an assumption |
| Expert kernels too slow at decode M≤8 even aligned | medium | misses 58 | 2-iteration tuning cap then stop; ~2.2–2.6× tolerance vs W2 Humming |
| Rewrite bugs corrupt weights silently | medium | quality damage | class-A/A2 oracles; DwarfStar-style continuation-vector regression as second anchor |
| Capture-unsafe patterns slip in | low-medium | eager fallback ≈ 5.5 tok/s | capture/replay + aliasing sweep is an M2 gate |
| Linear projection misleads the go/no-go | medium | wrong decision | screening only; layer-slice + M-distribution are the gates |
| Capacity: 140K point estimate; ~17.6K context per 100 MiB/rank unmodeled | certain (sensitivity) | context floor | M1 sizes every delta from the §4.7 table |
| Byte-valid but logically wrong load (transpose/fused-slot/rank offset) | medium | silent quality damage | class-A2 coordinate oracle |
| Tokenizer bootstrap falls back to generic HF mode | medium | contaminated comparisons | §4.4 explicit pin + text-level golden tests |
| Router cast degrades top-6 tie-breaks | medium | silent quality damage | §4.5 explicit policy; class-B window |
| DeepSWE single-run variance | certain | false pass/fail on one cell | **72-cell grid cancelled**; Will judgment gate; bisect on material regression only |
| Effort overrun | medium | opportunity cost | M1/M2/M4 kills; calendar caps |

## 10. Capacity plan

Method: M1's table from §4.7 — per-rank registered weights by tensor (sharded
vs replicated incl. `fused_wqa_wkv` replication and vocab-sharded
token_embd), post-transform sizes, graph pool (~0.19 GiB measured), Humming/
NVRTC workspace, loader/repack scratch, Marlin tile padding, KV pool,
headroom. Anchors: 78.74 GiB WNA16 artifact reached 230,144 ctx with 1.28 GiB
available KV; KV density ≈ 5.832 KiB/token/rank (independently recomputed by
the third review) → **~17.6K context per 100 MiB/rank**. The sharded GGUF tax
(~0.5 GiB/rank) gives ~139.1K — a point estimate with the replicated-family
delta at zero; precedent (indexer 191→767 MiB/rank after transforms) says it
must be measured, not assumed small. The aligned-SoA repack is byte-count
neutral by construction (§4.2) and must stay so in the capacity table. The
M1 gate requires the completed table to show ≥140K or levers whose summed MiB
close the gap explicitly. The 16 GiB host tier ships as prefix-reuse only,
gated by its hit-rate line. 430K active stays llama.cpp's exclusive advantage.

## 11. What "done" means

server60 serves `deepseek-v4-flash-0731-gguf-tp` from the pinned GGUF blob at
≥58/70 engine decode, ≥550/700 prefill, ≥140K on-GPU unique context (per the
M1 capacity table), zero swap, safety policy intact, M8 one-cell DeepSWE
pilot with Will's closeness judgment per `M8-DEEPSWE.md`, validated rollback,
everything committed and pushed, evidence bundled, upstreaming decision
recorded. The llama.cpp service remains canonical until M8 passes.

## 12. Immediate next actions (on approval)

1. M0: worktrees, base pin `b7766cfe`, fresh nsys trace of the 74.98 stack,
   baseline re-anchor.
2. M1: FORMAT-CONTRACT + L0/A2 oracle design + inventory + §4.7 TP table +
   dtype contracts + tokenizer pin + `wo_a` design + aligned-SoA repack
   spec + capacity table.
3. M2: IQ2_XXS kernel rewrite with raw-vs-aligned A/B, capture-safe
   registration, dense/`wo_a` prototypes, decoder-layer slice — the first
   hard end-to-end number.
4. **Resolved 2026-08-17:** Will accepts the measured 140–142K on-GPU context
   floor with approximately 0.52 GiB projected headroom as this service's
   initial contract. M5 must still measure residency and obey its falsifier;
   expanding context is a separate follow-up if promotion succeeds.
5. **Resolved 2026-08-18:** Will accepts the measured M5 idle headroom
   (71–73 MiB after long-context JIT) as normal for a packed vLLM TP profile.
   The 1 GiB physical-headroom release guard is scoped to dynamically sized
   profiles and is **not a promotion gate for this engine**; release evidence
   is zero swap + verify-stress-class boundary tests at the operating context.
   Reopen condition: any OOM at or below operating context. Full text:
   `CAPACITY.md` → "Will's headroom decision (2026-08-18)".
6. **Resolved 2026-08-18:** M8 DeepSWE = **one-cell pilot only**; the ≥72-cell
   multi-seed grid is **cancelled**. Will **approves executing** the locked
   GGUF-TP pilot (`M8-DEEPSWE.md`, plan `sha256:7ac3e4c4…`). Pass = Will's
   closeness judgment vs reused llama.cpp baseline on the pilot task. M6 must
   still pass before M8 counts toward promotion.
