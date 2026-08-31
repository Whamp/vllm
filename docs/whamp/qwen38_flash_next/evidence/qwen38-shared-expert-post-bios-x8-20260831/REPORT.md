# Qwen3.8 shared-expert post-BIOS x8 retest

Decision: **default off; current early-launch mechanism is a post-BIOS no-go**.

The BIOS change raised GPU0 from x4 to x8. The complete topology is now x8/x16/x8/x16. Expected PCIe riser errors were non-blocking by user instruction.

## Unprofiled same-image result

| Metric | Control | Early launch | Delta | Forward | Reverse |
| --- | ---: | ---: | ---: | ---: | ---: |
| C1 decode | 61.588 | 55.450 | -9.967% | -9.593% | -10.341% |
| C2 aggregate decode | 104.646 | 105.588 | +0.900% | +2.161% | -0.360% |
| C4 aggregate decode (diagnostic) | 188.895 | 188.382 | -0.272% | -0.252% | -0.291% |
| C1 prefill | 1684.410 | 1673.877 | -0.625% | -0.695% | -0.556% |
| C2 aggregate prefill | 1692.938 | 1687.643 | -0.313% | -0.332% | -0.294% |
| C4 aggregate prefill (diagnostic) | 1695.037 | 1694.027 | -0.060% | -0.080% | -0.040% |

Each control and candidate mean pools ten measured runs across two independent startups. Each startup used three decode warmups, five measured decode runs, one prefill warmup, and three measured cache-busted prefill runs.

## Fresh Nsight result

The trace used 0.979 GPU memory utilization for both arms because profiler startup overhead narrowly failed the 0.98 reservation check. Both arms retained the 262,144-token API limit. Profiled throughput is attribution-only.

| Phase | Graph-span delta | GPU-busy-union delta | Overlap-time delta | Streams |
| --- | ---: | ---: | ---: | ---: |
| C1 | +7.306% | +0.832% | +2.927% | 2 -> 3 |
| C2 | +6.291% | +1.622% | -9.883% | 2 -> 3 |

The selector still changes scheduling, but it does not shorten the critical path. C1 and C2 graph spans both grow while GPU busy time moves only slightly. The post-BIOS result therefore falsifies the x4-link explanation for the prior mixed outcome under this workload.

## Guardrails

All four benchmark arms ran at zero worker swap with no matched allocator or EngineCore failure signatures. Clocks reached the fixed 1650 MHz cap, power stayed below the fixed 230 W limit, and temperatures were comparable between arms.

Raw traces are checksum-bound under `/home/will/inference/runtime/qwen38-shared-expert-overlap/evidence/post-bios-x8-20260831`.
