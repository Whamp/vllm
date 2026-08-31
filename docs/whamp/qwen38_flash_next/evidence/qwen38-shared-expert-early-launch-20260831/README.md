# Qwen3.8 shared-expert early-launch evidence

This bundle records the bounded server60 validation of the default-off CUDA
shared-expert early-launch experiment at Whamp/vLLM commit
`0a5e081188f5384bcf79cfd201f418906de8f083`.

## Decision

Keep the mechanism default-off and retest it after the planned server60 BIOS
maintenance. Do not promote it from the current evidence, but do not treat the
experiment as terminally rejected.

The same-image unprofiled A/B did not clear the preregistered 3% decode gate:

| Metric | Selector off | Selector on | Change |
| --- | ---: | ---: | ---: |
| C1 decode | 59.3775 tok/s | 58.8669 tok/s | -0.8598% |
| C2 aggregate decode | 103.2025 tok/s | 104.5260 tok/s | +1.2825% |
| C1 cache-busted prefill | 1,614.3558 tok/s | 1,618.7557 tok/s | +0.2725% |
| C2 aggregate prefill | 1,629.2663 tok/s | 1,632.5239 tok/s | +0.1999% |

The positive C2 and prefill results justify retaining the candidate for another
matched test. The C1 regression remains part of the decision.

One RTX 3090 currently runs through a PCIe Gen3 x4 link. The result is therefore
specific to the present topology. This test does not show that the x4 link
caused any measured gain or regression.

## CUDA and serving gates

- Eight eager and CUDA Graph operator tests passed on one RTX 3090. They cover
  M=1, M=2, the 256-token threshold, the fallback above the threshold, and both
  DBO slots.
- Compute Sanitizer memcheck passed all eight tests with zero errors.
- Compute Sanitizer racecheck passed all eight tests with zero hazards.
- `run/gpu-gate/executed-test.py.gz` preserves the exact source used for the GPU
  run. The committed test uses the repository-required accelerator-neutral name
  for the same synchronization operation.
- The TP=4 candidate passed deterministic output, automatic tool use,
  post-tool continuation, multimodal input, two-stream decode, and exact
  `VIOLET ORBIT 9137` retrieval from a 261,544-token API prompt.
- The candidate stayed healthy with zero restarts, zero serving-process swap,
  and no logged allocator failure.

## Matched trace result

Both Nsight Systems 2025.5.2 traces used the same candidate image, model,
command, profiler settings, and compact C1/C2 workload. The only runtime change
was `VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH=0` versus `1`. Both profiled services
reported 423,164 KV-cache tokens and retained the 262,144-token API limit. The
unprofiled production service reports 425,497 KV-cache tokens; that 0.5483%
profiler difference is recorded, not treated as a failure.

CUDA Graph correlation analysis shows that the selector changed execution on
all four ranks:

| Trace measure | C1 | C2 |
| --- | ---: | ---: |
| Streams per replay | 2 -> 3 | 2 -> 3 |
| Mean median graph-span change | -2.4881% | +8.9127% |
| Mean overlap-time change | +2.6520% | -10.0426% |
| Mean overlap fraction | 10.3422% -> 10.6481% | 9.6944% -> 8.6063% |

The trace therefore does not show a phase-independent critical-path
improvement. C1 moved in the intended direction; C2 did not. Node-level Nsight
instrumentation materially changes timing, so the graph-span figures are
attribution evidence, not serving throughput measurements.

`analyze_trace.py` reproduces `trace-analysis.json` byte-for-byte from the raw
SQLite pair.

## Exact candidate identity

| Item | Identity |
| --- | --- |
| Candidate image ID | `sha256:39de8fdfb787592cf06819268a817c8a4087d84658e2300d7adb5ad136b59bb3` |
| Base production image | `sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef` |
| Candidate Git commit | `0a5e081188f5384bcf79cfd201f418906de8f083` |
| Selector | `VLLM_CUDA_SHARED_EXPERTS_EARLY_LAUNCH=1` |
| Full-model candidate Compose SHA-256 | `864c4fa77f475f0c48670876b7fb540e5a84f1398621238d848fea8d985989ac` |
| Full-model control Compose SHA-256 | See `run/full-model/control-compose.sha256` |

`run/build/` contains the image inspection, compressed build log, extracted
source hashes, and runtime patch manifest. The manifest-pinned Dockerfile remains
in the experiment delivery directory. `run/full-model/` contains the exact
resolved control/candidate Compose files and benchmark results.

## Raw trace archive

The raw traces are hard-linked into the durable server runtime tree; removing
the build-tree links does not remove this archive.

| File | Bytes | SHA-256 |
| --- | ---: | --- |
| `/home/will/inference/runtime/qwen38-shared-expert-overlap/evidence/20260831/control/control.nsys-rep` | 115,599,376 | `1448bc8c81b79870365d7be406351c33c322d9aac0332d01c7aa75aeec9c1526` |
| `/home/will/inference/runtime/qwen38-shared-expert-overlap/evidence/20260831/control/control.sqlite` | 499,650,560 | `c402c88d1c247f7abc04e437853c240e5152da60dfead4589aa0cf589696cd4b` |
| `/home/will/inference/runtime/qwen38-shared-expert-overlap/evidence/20260831/candidate/candidate.nsys-rep` | 118,052,122 | `bfcc1a01b053361de44ad3139f59c2cd3ff070a6014d3927c497517b93fd52f6` |
| `/home/will/inference/runtime/qwen38-shared-expert-overlap/evidence/20260831/candidate/candidate.sqlite` | 507,805,696 | `7674ad8a142cf6903dd4abeeb58f2d256f1c42346c2f8140207400d4ded1f2df` |

## Post-BIOS matched retest

1. Confirm the GPU device order and loaded PCIe link width. Record topology and
   require the repaired slot to negotiate at least x8 before interpreting the
   run as the planned retest.
2. Verify the pinned production rollback, zero swap, and the fixed 230 W / 210-
   1650 MHz GPU safety policy. Arm the restore watchdog.
3. Rebuild or reuse the exact candidate above. Keep the model, image, context,
   max sequences, batch-token budget, cache format, collectives, PLE path, and
   sampling fixed. Change only the selector.
4. Run matched selector-off/on pairs in reverse order as well as forward order.
   Use three warmups and five measured runs for C1 and C2 decode and
   cache-busted prefill. Add C4 as a diagnostic, not as a substitute for C1/C2.
5. Repeat functional, long-context, zero-swap, allocator, and memory-stability
   checks. Capture another matched trace only if the service results support a
   causal promotion decision.
6. Promote only if the preregistered decode and regression thresholds pass with
   the corrected topology. Otherwise retain or retire the selector based on the
   new evidence.

## Final state

The experiment containers were removed. Server60 was restored to production
container `qwen38-qsa-fp8-candidate` on image
`sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef`
with native 262,144-token context, `restart: unless-stopped`, zero restarts,
zero serving-process swap, active GPU safety controls, and no restore timer.
