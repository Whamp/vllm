# Adversarial review — Claude Fable 5 medium (claude-fable-5-medium), 2026-08-17

- Target: .research/gguf-tp-engine/PLAN.md v2 @ commit 24d1f973d2e0ad0529b807647c4759c306efbad1
- Envelope: Cursor Agent CLI --print, --yolo --trust, read-only instruction, same workspace/add-dir set as the Grok run; explicit one-time user override of the standing Claude-unavailable policy for this CLI-routed review only (recorded in conversation 2026-08-17); independence guard: instructed NOT to read the archived Grok review.
- Prompt SHA-256: cfdd879f162313fc62e620422ae0ed8a3ab2469350f087a9938e57f134e63e86
- Worktree after run: unchanged (clean at 24d1f973, empty diff).
- Delivery-owner reconciliation: all 10 findings (F1-F10) accepted. F1, F2, F5 source-verified against pinned trees before acceptance: vecdotq.cuh:985-1014 (IQ2_XXS grid bytes + aux32 7-bit signs + ls sub-scale, integer truncation), mmid.cu:5-28 + mmvq.cu:699/766/823-825/394-443/775-787 (device-side id compaction, dedicated MoE MMVQ launch, gate/bias fusion, DSV4 small-K), compressor.py fp32 assert + attention.py fp32/fused-quant contracts. F6 binomial arithmetic independently recomputed (P(X<=4|n=12,p=.5)=794/4096=19.4%). Plan v3 incorporates all accepted findings.

---

All checks are done. Here is the independent adversarial review of plan v2.

# Adversarial review — GGUF-TP engine plan v2 (independent second opinion)

## Findings

### F1 [MAJOR] The plan's IQ2_XXS block-layout description is wrong, which falsifies its own "read line-by-line" evidence grade

Attacked text (§3): *"IQ2_XXS = fp16 `d` + 32×uint16, each uint16 = grid-LUT byte (256-entry LUT, 2 KiB) + 8 sign bits"*; and §2 row 1 grading dequant semantics as **proven-here** on the basis of *"pinned Whamp/llama.cpp `0379cf4bf` source read line-by-line."*

Evidence: the actual kernel (`vec_dot_iq2_xxs_q8_1`) shows the real layout. Per group of four uint16s: the first two uint16s hold **four grid-index bytes**, and the next uint32 (`aux32`) holds **four 7-bit sign fields** plus a **5-bit per-group sub-scale** applied as integer `ls = aux32 >> 27 | 1; sumi = sumi * ls / 8`:

```1011:1014:/home/will/projects/llama.cpp-ds4-study/ggml/src/ggml-cuda/vecdotq.cuh
    const int ls = aux32 >> 27 | 1; // (scale * 2 + 1)
    sumi = sumi * ls / 8;           // (sumi * scale + sumi / 2) / 4
    const float d = __half2float(bq2->d) * __low2float(bq8_1[iqs/2].ds);
    return d * sumi;
```

The plan's description omits the per-group sub-scale entirely and misstates the sign packing (7-bit fields shared in `aux32`, not "8 sign bits per uint16"). A FORMAT-CONTRACT written from §3 as-is dequantizes garbage. The class-A oracle would catch it at L0, but the point stands: the one row §2 grades "proven-here" from a source read is demonstrably misdescribed in the same document. Required change: fix §3's layout text (grid bytes + 7-bit signs + 5-bit sub-scale + integer truncation semantics), and downgrade the §2 confidence language — the semantics are proven *in the pinned source*, not yet proven *in the plan's understanding of it*.

### F2 [MAJOR] The wrapped-ggml counter-analysis (§4, review-F13 arm) is factually false about the pinned tree — the fallback is far cheaper than represented, possibly cheap enough to be primary

Attacked text: *"those are single-GEMM kernels; vLLM needs a graph-capturable fused indexed grouped GEMM with device-side top-6 routing over 256 experts — exactly the machinery Humming provides and llama.cpp solves with launch-bound CPU-side per-expert dispatch. The hidden cost is reimplementing indexed dispatch, not the vec_dot."*

Evidence from the pinned study tree (`0379cf4`):

- `ggml/src/ggml-cuda/mmid.cu` implements **device-side** id compaction for `mul_mat_id`: `mm_ids_helper` converts routing ids to expert-sorted `ids_src1`/`ids_dst`/`expert_bounds` on the GPU — no CPU-side per-expert dispatch.
- `ggml/src/ggml-cuda/mmvq.cu:824-830`: `if (has_ids && ncols_dst > 1)` dispatches a **dedicated MoE MMVQ kernel** (`mul_mat_vec_q_moe_launch`); the single-token path takes device `ids` directly (`mmvq.cu:425-426`). MMQ also supports `mul_mat_id` (`mmq.cu`).
- The launcher carries **fusion args including gate fusion** (`fusion.gate`, `mmvq.cu:765`) and this fork already has **DeepSeek-V4-specific decode tuning** for exactly the routed up/gate IQ2_XXS matvec shape (`DSV4_MMVQ_SMALLK`, `mmvq.cu:777-787` — the comment names IQ2_XXS at `n_embd=4096`).

So the "hidden cost" the plan cites — indexed dispatch — already exists in the kernels it would wrap, device-side and graph-capturable, with DSV4 shape tuning done. What actually remains for the wrapped arm is id-format/TP-offset plumbing and vLLM op integration, which is much smaller than "reimplementing indexed dispatch." Required change: rewrite the counter-analysis against the real `mmid.cu`/`mul_mat_vec_q_moe_launch` machinery, and re-decide primary-vs-fallback ordering on corrected facts. At minimum, M2 should microbench the wrapped-ggml MoE path *first* (it is nearly free to measure) so the Humming-fragment effort has a measured bar to beat rather than a strawman.

### F3 [MAJOR] The §5 tolerance arithmetic uses a stale time mix and budgets zero slowdown for the entire Q8_0 dense path — the "3× headroom" is more like ~2.2–2.6×

Attacked text (§5): *"at 74.98 tok/s (13.3 ms/token) with experts ≈15% ≈ 2.0 ms, holding everything else constant: 58 tok/s tolerates experts ~3× slower"*; and the risk register's *"§5 tolerance math (3× headroom to floor)."*

Two errors compound:

1. **Stale mix applied to a new denominator.** The 15% expert share comes from the *pre-FlashMLA* trace, on a stack where NCCL (~19%) and sparse decode (~14%) were larger. The 74.98 stack "attacks the top two," so the surviving kernels' shares renormalize *upward*. If hier-AR and FlashMLA roughly halved those two, experts are ~18% of the new mix, i.e. ~2.4 ms, not 2.0 ms. The tolerance to the 3.9 ms budget becomes (2.4+3.9)/2.4 ≈ **2.6×**, not 3×.
2. **"Holding everything else constant" is false by the plan's own admission.** §2 grades "Same stack on Q8_0 weights" as **unmeasured**, yet §5 assigns the Q8_0 dense replacement (all attention projections, shared experts, output — the Marlin ~23% share, *larger* than the expert share) zero slowdown. Q8_0→int8-g32 means group-32 scales (more scale traffic than the FP8 path's 128-block scales) at 8.5 bpw vs ~8 bpw. If the dense path runs even 30% slower, ~0.9 ms of the 3.9 ms budget is gone before experts spend any, and expert tolerance drops toward ~2.2×.

Required change: (a) take a **fresh nsys decode trace of the running 74.98 stack** — it exists and is cheap, there is no reason M2's gate arithmetic should rest on a pre-FlashMLA trace; (b) the M2 projection must include *measured* Q8_0-g32 Marlin dense GEMV numbers and the measured `wo_a` number at serving shapes (all obtainable on one 3090 without bring-up), not just the expert fragment; (c) restate the risk-register headroom honestly.

### F4 [MAJOR] The ~140K capacity number is called a "floor" and "certain" but is an unmargined midpoint with ~17K-context sensitivity per 100 MiB of unmodeled residency

Attacked text (§10): *"the honest on-GPU projection is ~140K, floor 140K"*; risk register: *"Capacity: ~140K on-GPU honest floor … likelihood: certain."*

Arithmetic: 1.28 GiB available KV at 230,144 ctx → ~5.8 KiB/token/rank. The stated ~0.5 GiB/rank sharded tax alone gives (1.28−0.5) GiB / 5.8 KiB ≈ **137K** — i.e. the "floor" is the projection *with the replicated-family delta at zero*, which §10 itself admits is unquantified ("**plus** replicated-family deltas"). The precedent the plan cites cuts against it: on the WNA16 stack the replicated indexer went **191→767 MiB/rank after transforms**. Every additional 100 MiB/rank of replicated delta, loader/repack scratch, Marlin tile padding (`marlin_padded_nk`, `marlin_utils.py:221`), or Humming NVRTC workspace costs ~17K context. A 0.5 GiB/rank replicated delta puts the real floor near **50–55K**, below any success threshold. Required change: relabel 140K as a *point estimate*, state the KiB/token sensitivity explicitly in §1's threshold table, and make the M1 capacity gate quantify the replicated-family delta and scratch terms *before* the ≥140K minimum-success line is treated as achievable.

### F5 [MAJOR] "F16/F32 control tensors load as-is — no dtype changes at all" conflicts with the consuming kernels' dtype/layout contracts

Attacked text (§2): *"F16/F32/I32 control tensors load as-is — no BF16 downcast, no dtype changes at all outside the two documented conversions."*

Evidence: on the base stack these families do not run raw. The indexer path goes through fused-quant transforms (`SparseAttnIndexer`, `fused_n_q_rope_quant`, MXFP4 block handling in `vllm/models/deepseek_v4/attention.py`) — that is precisely why the measured residency was 191→767 MiB/rank "after transforms"; the compressor asserts fp32 state (`compressor.py:174`); and the model's activation dtype is bf16. So the F16 replicated families face exactly two outcomes, both of which the plan disclaims: (a) the same transforms the WNA16 stack applies (an undeclared conversion with a ~0.5 GiB/rank-class replicated capacity hit — feeding F4), or (b) an F16→BF16 cast (mantissa 10→7 bits, lossy) — on the **router**, where top-6 tie-breaks are sensitive, this is a silent-quality vector the oracle classes don't cover because the plan declared the family conversion-free. Required change: M1 must inventory the dtype/layout contract of every kernel consuming a replicated family; any transform or cast becomes a third/fourth *documented* conversion with a class-B window and a line in the capacity table.

### F6 [MAJOR] The DeepSWE class-D window has almost no statistical power at n=1 on 12 tasks

Attacked text (§1/§6): *"strict solves ≥ baseline−1 and mean partial ≥ baseline−2.0 pp"* against a baseline of 6/12 strict solves, single SuperJSON worker, temperature 0.6.

Binomial reality: if the new engine is *identical* (per-task solve p=0.5), P(X≤4) ≈ 19% — a one-in-five false failure. If it genuinely degraded to p=0.35 (a real, promotion-blocking regression), P(X≥5) ≈ 42% — nearly a coin-flip false pass. A gate with 19%/42% error rates cannot carry the plan's hardest quality claim, especially with stochastic sampling adding run-to-run variance the window doesn't model. Required change: make the comparison **paired per-task** (same tasks, discordant-pair analysis), run ≥3 seeds or fix seeds/greedy where the harness allows, and pre-register the decision rule on the paired statistic — or honestly demote DeepSWE from "gate" to "smoke signal" and let per-layer KL + NIAH + canaries carry the promotion decision.

### F7 [MAJOR] The M2 and M1 gates are self-graded: projections graded by the plan's own (flawed) arithmetic, and an escape hatch that makes M1 unfailable

Attacked text (§8): M2 gate *"projected decode ≥ 58 via §5 arithmetic; fallback arm costed"*; §5 falsifier *"the wrapped-ggml fallback arm also **projects** < 58"*; M1 gate *"capacity table yields ≥140K projection **or a named lever**."*

Three problems: (a) the M2 go/no-go runs through the §5 mapping that F3 shows is built on a stale trace and omits the dense-path term — the gate inherits the arithmetic's optimism; (b) the fallback arm is only "costed"/"projected" on both sides of the kill condition — no measurement is required to keep the project alive, and per F2 measuring it is nearly free; (c) "or a named lever" means M1's capacity gate cannot fail — one can always name a lever. First hard measured gate for decode is M7 and for prefill there is *no* falsifier at all before M7 (the 550 floor is never kernel-gated; the M2 microbench covers M=1–8 expert shapes only, while the dense Q8_0 GEMMs at prefill M≈256 are unmeasured). Required change: M2 gate = projection from a *fresh* trace with *measured* expert-fragment + wrapped-ggml + dense-Q8_0 + `wo_a` kernel times; M1 gate = quantified capacity table with levers *sized in MiB and context tokens*, not merely named; add a prefill-shape microbench line to M2 or M4.

### F8 [MINOR] The 6–9 week envelope is a zero-contingency straight sum that contradicts the plan's own anticipated iteration

Attacked text (§8): *"Effort envelope: 6–9 focused weeks."* Summing the milestone estimates gives ~6 weeks (all-low) to ~9.5 weeks (all-high) — the envelope *is* the sum, with zero allowance for the "two tuning iterations" M2 explicitly budgets, M6 "unexplained divergence → bisect," M8 "component bisect," or the wo_a redesign loop the risk register rates medium-high. A plan that anticipates iteration in its gates but not in its calendar is optimistic by construction. Also inconsistent: M4's calendar kill is 10 working days against a 1-week estimate. Required change: either state the envelope as "6–9 weeks if every gate passes first-try; kills bound the downside" or add explicit iteration contingency.

### F9 [MINOR] Unnamed risks: tokenizer/chat-template parity, GGUF-metadata-vs-HF-config source of truth, per-load hashing and host-RAM budget

The plan "keeps" the vLLM tokenizer/chat path (§4) while every class-B/D comparison target (llama.cpp canonical service, the DeepSWE 6/12 baseline) ran through the **GGUF-embedded tokenizer, llama.cpp's chat template, and llama.cpp's sampler chain**. Any divergence (byte-merge edge cases, added-token handling, template whitespace, top-k/top-p application order) produces different token streams before the first kernel executes, contaminating the fixed-prompt KL oracle (which must be pinned to token *IDs*, not text) and the DeepSWE delta. Similarly unstated: where the vLLM model config comes from for the GGUF service (RoPE theta / YaRN factors, compress ratio, SWA window — GGUF KV metadata vs the HF config the WNA16 stack used; M1 should diff them). And operationally: "verify blob SHA-256" on an 80.76 GiB mmap'd file at every load is minutes of I/O per restart, and the host-RAM budget (80 GiB page cache churn + 16 GiB pinned offload tier + repack scratch) is nowhere in the capacity plan. Required change: add tokenizer-parity verification to M1, name the config source of truth, and state the load-time/host-RAM contract.

### F10 [MINOR] Class A's "fragment dequant bit-exact, no tolerance" is ambiguous about the comparison point, and the reference kernel itself is not bit-equal to the CPU oracle

Attacked text (§6): *"A. Bit-exact (must hold exactly): IQ2_XXS/Q2_K **fragment dequant** vs llama.cpp CPU `dequantize_row_*` … No tolerance."*

The GPU reference path the plan's kernels would emulate uses integer arithmetic with truncation (`sumi * ls / 8`, F1's citation) — its *dot-product results* are not bit-equal to CPU float dequant-then-FMA, by design. Bit-exactness is only achievable for **dequantized weight values computed in fp32**; the fused fragment's GEMM *output* necessarily lives in class B with a window (the L2 full-kernel oracle exists in the ladder but its class assignment is unstated). As written, class A is either trivially satisfiable (values only — fine, but say so) or impossible (fused output — would fail on day one and invite "re-classify until green"). Required change: pin class A to dequantized values in fp32, and explicitly assign fused fragment output to a pre-registered class-B window.

## Verdict

**PROCEED-WITH-REVISIONS.** The plan's core bet is defensible — the artifact's quality edge is real, the base stack is measured, and the two new codebook fragments are a genuinely narrow scope — but v2 still contains one factual error that would misdirect week one (the IQ2_XXS layout in §3), one wrong counter-analysis that may have the primary/fallback routes backwards (the pinned tree already ships device-side indexed MoE MMVQ with DSV4 tuning), a decode-tolerance argument that silently exempts the largest unmeasured kernel share (the Q8_0 dense path), a capacity "floor" that is actually a zero-margin midpoint, an undeclared F16-family conversion problem, and a quality gate with coin-flip error rates. None of these is unfixable, and most are repairable inside M1–M2 at low cost (a fresh nsys trace, a wrapped-ggml microbench, a quantified capacity table, a paired DeepSWE protocol) — but the M1/M2 gates as currently written would grade the plan's own optimism, so the revisions must land in the gate definitions before M0, not be discovered mid-flight.
