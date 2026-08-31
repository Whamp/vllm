# Qwen3.8 concurrent PLE fan-out result

## Decision

Do not promote unconditional four-rank PLE fan-out on server60. The post-BIOS x8 crossover confirmed a 3.95% C4 decode gain and a 1.25% C2 gain, but C1 decode lost 5.54%.

Keep this candidate default-off. The next candidate should retain serial TP delivery for single-sequence decode and use concurrent delivery for multi-sequence decode and prefill.

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

## Post-BIOS x8 crossover

The BIOS change moved GPU 0 from PCIe Gen3 x4 to x8. The Samsung NVMe device remained available, and all four GPUs negotiated x8/x16/x8/x16 under load.

An initial health-gated attempt stopped during the control because the GPU 0 root port produced recurrent correctable PCIe errors. Will then authorized a full run that recorded correctable errors but stopped on fatal or uncorrectable PCIe events, NVIDIA Xids, CUDA or NCCL failures, request failures, or container restarts.

Two matched pairs ran in opposite orders:

1. control, then fan-out candidate;
2. fan-out candidate, then control.

The pooled values are arithmetic means of the order-balanced legs:

| Concurrency | Control decode | Fan-out decode | Decode delta | Prefill delta | Mean TTFT delta |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 | 66.0317 tok/s | 62.3709 tok/s | -5.5440% | +1.0522% | -69.4700% |
| C2 | 106.8013 tok/s | 108.1360 tok/s | +1.2497% | +0.8496% | -47.7222% |
| C4 | 182.3439 tok/s | 189.5454 tok/s | +3.9494% | +0.0827% | -59.2749% |

The C4 gain repeated at 3.8996% and 3.9993% in the two orders. C2 measured +2.3323% and +0.1699%. C1 lost 3.2601% and 7.7938%.

Candidate C4 decode population CV was 0.9234% and 1.1100%. Control C4 CV was 11.3514% and 9.8862%, but the control pair means remained close at 182.5485 and 182.1394 tok/s. Candidate means repeated at 189.6671 and 189.4236 tok/s.

The unconditional candidate fails the preregistered gate because C1 loses more than 1%. The measurements support a request-shape gate: keep serial delivery for single-sequence decode, then use concurrent delivery for multi-sequence decode and prefill.

Both variants exercised the noisy x8 riser path. The control accumulated 123 correctable errors across both legs and the candidate accumulated 177. The second leg had more errors in both orders, and total pair counts were nearly identical at 148 and 152, so the candidate's 54-error excess is not clean feature attribution. No fatal or uncorrectable PCIe event, NVIDIA Xid, container restart, CUDA or NCCL failure, or measured-request failure occurred.

Both variants retained 425,497 aggregate KV-cache tokens. The crossover did not rerun the full behavior battery because the unconditional candidate failed the C1 gate; the same immutable candidate image passed that battery in the earlier campaign.

The combined checksum-bound archive is:

```text
/home/will/build/qwen38-ple-fanout-post-bios-x8-crossover
```

It contains 15 files. The manifest SHA-256 is:

```text
eceb5664869811631f46169c152b485c035dc5a9dde4892c1132399c3e98c90e
```

The forward and reverse pair manifests are `f0edb74779faeaffa9034b942be1a3ae55635a470ebee892fe68df2f584f5374` and `23d7ae0883e6ae2dea67722e6be6c16846eb6747a8b3465da0e5f811066d9a1c`.

The final restore returned pinned production to healthy status with zero restarts, 425,497 KV-cache tokens, native NVFP4 lookup enabled, zero swap, `vm.overcommit_memory=0`, and no experiment containers or jobs.
