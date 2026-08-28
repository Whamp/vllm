# M0-TRACE.md — fresh post-optimization WNA16 baseline (server60)

Status: M0 complete. Observational attribution only — profiled throughput is
invalid and never used as benchmark evidence.

## Identity

- Plan SHA-256: `b13ce445d79fb5b2e3031b2ec1bcae4f43c530d9b4d75358f8ecb0f7113f072f`
  (`evidence/m0-trace/plan.json`).
- Harness: Whamp/club-3090 `bfb1f9c4af073a9d49b748b93c293b6197e1eae1`
  (added `trace-flashmla-hier` + plan-bound long-warmup rollback).
- Runtime tree: Whamp/vllm `b7766cfe4d15d9b68acea43097ceff221e8a739f`,
  tree `6354125afd1306c9286f734d1c47c23c767d77a9`.
- Speed image: `sha256:eb2884fc60ee…`; Nsight image:
  `sha256:8359712d1fa4…`; quality artifact revision `12035985…`.
- Profile: `max_model_len=230144`, max_num_seqs=2,
  max_num_batched_tokens=256, gpu_memory_utilization=0.98,
  KV_OFFLOADING_SIZE=0.001 (trace-only minimum), FlashMLA=1,
  hierarchical islands `0,1;2,3`.
- Raw trace stays on server60:
  `/home/will/inference/runtime/deepseek-v4-wna16-sm86/gguf-tp-m0-20260817/
  nsight-flashmla-hier-1/profile/deepseek-v4-decode-flashmla-hier.nsys-rep`;
  size 62,024,647 bytes; SHA-256
  `92ee80ff4368ba7067a106be850034422c05b41d9aa0bca9f9cafdb1ed5b2e04`.

## Gates

- FlashMLA: upstream numerical suite 17/17; seven cuobjdump-visible sm_86
  cubins; startup `Using native Ampere FlashMLA sparse decode`.
- Hierarchical all-reduce: BF16 oracle/timing across 4K–262K elements;
  hierarchical median 75.98–85.32% of NCCL; startup dispatch
  `['HIERARCHICAL','PYNCCL']` for TP/EP.
- Profile request: warmed, 256-token non-streamed response, finish=length,
  1,125 content chars, nonempty.

## Summed GPU kernel-time mix (M2 screening budget)

Source: `evidence/m0-trace/cuda-gpu-kernel-summary.csv` and the mutually
exclusive substring classification in `category-summary.json`. **This is
summed GPU kernel time, not critical-path attribution**; kernels can overlap.

| Pool | Fraction | M2 implication |
|---|---:|---|
| Marlin dense projections (FP8 baseline) | **26.63%** | Q8_0→int8-g32 dense/wo_a replacement is the largest changed pool; must stay near baseline |
| Collectives (hier kernels + NCCL allgather) | **19.74%** | unchanged inherited stack; local trace confirms it remains material |
| Humming W2 experts | **15.41%** | primary IQ2_XXS/Q2_K rewrite comparison |
| Humming W4 experts | **0.92%** | included in total expert replacement budget |
| **Total expert pool** | **16.33%** | confirms PLAN's 15–18% estimate |
| Hyperconnection | 6.69% | unchanged; class-B preservation gate |
| Sparse indexer | 6.04% | unchanged; F16→bf16 window remains sensitive |
| FlashMLA sparse decode | **4.41%** | post-optimization drop from the old ~14% mix; not the M2 focus |
| Other | 20.16% | CUTLASS BF16, fused ops, norms, cache writes, elementwise |

Plan update: the old dense estimate (~23%) becomes **26.63%**; expert estimate
remains 16.33%. M2's graph-captured layer-slice gate, not this linear mix,
remains the actual keep/stop decision.

## Rollback / safety evidence

- Pre-run safety cap re-verified under live generation: 800 GPU-clock samples,
  maximum exactly 1650 MHz, zero samples over; power service enabled/active,
  230 W requested/current on all cards.
- Canonical llama.cpp service restored: container
  `llama-cpp-deepseek-v4-fast-prefill`, image `sha256:a96bd947…`, healthy,
  restart count 0, serving-process swap 0, no experiment container remains.
- Independent 90-minute restore watchdog armed before the switch and cancelled
  only after all final checks passed.
- Compact final evidence: `evidence/m0-trace/`; raw trace retained by path+hash.
