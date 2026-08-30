# Qwen3.8 concurrent PLE fan-out result

## Decision

Reject concurrent four-rank PLE fan-out on server60's current PCIe topology. Preserve it default-off for one matched retest after the planned BIOS correction changes GPU 0 from PCIe 3.0 x4 to x8.

The x4 link masking a gain is an unverified hypothesis, not the diagnosed cause. The relevant transfers are small, and fixed latency, synchronization, scheduling, or other costs may remain dominant after the link-width correction.

Do not promote this candidate from the current evidence.

## Preserved candidate

- Branch: `perf/qwen38-ple-fanout-production`
- Implementation commit: `f8b16676e9f6208ab5bcdf2728e0a5905554b07d`
- Control image: `sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef`
- Candidate image on server60: `sha256:e4200bc033f1ebbb592e1f21281e97a1c89f58111f17d58145e3ec681b1c33f7`
- Control worker: `8d6d71de8b56f32e27850ba24dcd194482808e6b7f0dc27f9312d39eb3c0fa32`
- Candidate worker: `ba30a269be70eaea376d0535f0c3bb950e2979a9928ab9161b7513c5c50bc04c`
- Enablement: `VLLM_PLE_CONCURRENT_FANOUT=1`

Unset or `0` preserves serial production behavior. The candidate remains outside `main`.

## Current-topology evidence

The candidate started on all four RTX 3090 GPUs with:

- the expected image and worker hashes;
- native NVFP4 lookup enabled;
- `Concurrent PLE output fan-out enabled for TP=4` in the worker log;
- 425,497 aggregate KV-cache tokens;
- healthy status and zero restarts;
- no CUDA, stream, semaphore, deadlock, or PLE-worker error during C1, C2, and C4 request smoke.

The short smoke reached 62.34, 87.20, and 156.92 decode tok/s at C1, C2, and C4. These were correctness and stability probes, not promotion measurements.

The final comparison gave each variant the same 128-token C2 preconditioning harness immediately before its 256-token C1/C2/C4 matrix:

| Concurrency | Control | Candidate | Change |
| ---: | ---: | ---: | ---: |
| C1 | 63.615925 tok/s | 61.154960 tok/s | -3.87% |
| C2 | 105.205337 tok/s | 101.111325 tok/s | -3.89% |
| C4 | 184.011676 tok/s | 179.749824 tok/s | -2.32% |

Prefill changed by -0.34%, +0.01%, and -0.09% at C1, C2, and C4. The decode result misses both requirements: at least 1% improvement at C2 or C4 and no more than 1% loss at C1.

The candidate therefore failed before the full deterministic, tools, multimodal, and 261,544-token acceptance battery. Skipping that expensive battery after the performance gate failed is intentional; it must not be represented as passed.

## Attribution boundary

Nsight Systems was unavailable on the host and candidate image, so the planned request-to-all-ranks-ready trace was not collected. Process-level measurements were sensitive to warm state and did not establish Python thread scheduling, PCIe width, or another mechanism as the cause of the loss.

A readiness-only ablation reached healthy startup but was not benchmarked after Will fixed the decision boundary at the original candidate. No ablation result enters this verdict.

## Restoration

The experiment stopped and removed all fan-out containers, then restored the promoted service through:

```text
/home/will/inference/runtime/qwen38-qsa-fp8-candidate/restore-fp8.sh
```

The final receipt records the exact control image, healthy status, zero restarts, `unless-stopped`, 262,144-token model length, the control worker hash, zero swap, `vm.overcommit_memory=0`, and no stray experiment jobs.

The checksum-bound server60 archive is:

```text
/home/will/build/qwen38-ple-fanout/EVIDENCE.SHA256SUMS
```

It contains 143 files. The manifest SHA-256 is:

```text
1abed6ffaf7979c46666e1b16681c6e803d47784c5b2e4cea9142f0ade3bc244
```

## Authorized post-BIOS retest

Run one matched retest only after verifying under load that the intended GPU link changed from x4 to x8 and the other links and storage remain healthy. Preserve the model, image, worker hashes, 0.98 memory utilization, FP8 KV cache, hierarchical all-reduce grouping, CUDA Graph mode, request inputs, warmup, and power limits.

Use the same matched-warm protocol and current thresholds. Do not promote unless C2 or C4 improves by at least 1%, C1 loses no more than 1%, capacity remains at least 425,497 tokens, and no correctness or operational gate fails. If the retest still loses, retain the current rejection without another tuning round.
