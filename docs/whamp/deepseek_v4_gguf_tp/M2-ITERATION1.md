<!-- markdownlint-disable MD060 -->

# M2 iteration 1 — native BF16 IQ2_XXS scalar matvec

Decision: **reject as the production expert kernel; keep as the correctness/
layout reference and proceed to the second permitted tuning iteration.**

## Change

Whamp/vllm branch `incubate/gguf-tp-sm86`, base `b7766cfe`:

- New stable-ABI CUDA ops `gguf_iq2_xxs_raw_matvec` and
  `gguf_iq2_xxs_aligned_matvec` (current stream, caller-owned output, no
  allocations, CUDA-graph safe).
- One warp/output row; BF16 activation input, in-loop IQ2 grid/sign decode,
  fp32 FMA accumulation; raw 66-byte blocks vs byte-neutral aligned streams.
- Immutable format tables copied/adapted from antirez/ds4 `84cc882` under
  MIT attribution; no llama.cpp/ggml linkage.

## Evidence

- CUDA 13 / SM86 extension build: pass; extension SHA-256
  `6148b0e4b754603f40a028a43a4976eb1b97a71bbc7fd6a69ddecdefbff4c7cc`;
  cuobjdump shows both raw/aligned IQ2 kernel symbols in packaged SM86 code.
- RTX 3090 numerical + graph tests: **7 passed in 5.10 s** (M=1/2/4,
  K=256/512, random pinned-grid/sign/scale corpus, independent fp32 reference,
  raw==aligned bitwise, CUDA graph replay).
- Exact serving-shape benchmark: K=4096, N=2048, 100 warmup + 1,000 measured
  calls; `evidence/m2-iq2-iteration1/benchmark.json`.

| M | raw µs | aligned µs | aligned/raw | aligned logical GB/s |
|---:|---:|---:|---:|---:|
| 1 | 53.95 | 52.86 | 0.9798 | 40.91 |
| 2 | 88.20 | 86.94 | 0.9858 | 24.88 |
| 4 | 163.47 | 161.36 | 0.9871 | 13.40 |

Graph replay ratios are nearly identical (M1 aligned/raw 0.9799).

## Causal result

- **Hypothesis falsified for this kernel:** the 2-byte raw-block alignment is
  not the dominant constraint; aligned streams recover only 1.3–2.0%.
- Achieved 40.9 GB/s is only ~11–12% of our measured llama.cpp IQ2_XXS MMVQ
  (346–358 GB/s), so the scalar BF16 FMA loop is execution/lookup limited.
- One M=1 expert matrix at 52.86 µs cannot meet the full top-6 gate/up expert
  budget if repeated serially; no indexed-MoE integration is justified yet.

## Iteration-2 move (final permitted tuning iteration)

Explicitly quantize each BF16 layer input once to Q8_1, then use a native
IQ2_XXS×Q8_1 DP4A matvec. This is a rewrite, not a wrapper. Source references:
antirez/ds4 `cuda/mmq/quantize.cu`, `vecdotq.cuh`, `mmvq.cuh` and the pinned
llama.cpp integer identity in FORMAT-CONTRACT.md. The conversion cost is
measured separately and shared across all gate/up experts; it is never hidden.
Aligned/raw A/B repeats under DP4A, where DwarfStar's alignment mechanism
actually applies. If the full quantize+DP4A path still misses the M2 budget,
the PLAN §8 two-iteration kill criterion triggers.
