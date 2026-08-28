<!-- markdownlint-disable MD060 -->

# M2 — dense Q8_0 Marlin decode screen

Decision: **pass the decode-dense component screen.** This is not the M2 layer-slice or prefill gate.

All rank-local Q8_0 shapes use the same byte-neutral INT8/group-32 Marlin adapter validated in `M2-Q8-WOA.md`. Numerical coverage now spans K=256/512/1024/2048/4096; 14/14 RTX 3090 tests pass, including M=1/4 original-Q8 and transformed-weight windows plus exact grouped `wo_a` M=1/2/4 graph replay.

## Exclusive RTX 3090 benchmark

Five trials per shape and M, 2,500 warm + 5,000 measured graph replays, GPU0 only, ≤1 process/sample, max clock 1650 MHz, canonical final zero-swap. Prepared weight+scale bytes equal raw Q8_0 bytes for every shape.

| Rank-local projection | K→N | M=1 graph time |
|---|---:|---:|
| fused wq_a+wkv | 4096→1536 | 13.690 µs |
| wq_b | 1024→8192 | 17.345 µs |
| wo_b | 2048→4096 | 18.061 µs |
| shared gate+up | 4096→1024 | 12.228 µs |
| shared down | 512→4096 | 8.160 µs |
| `wo_a` grouped diagonal | 4096→2048 | 18.438 µs |
| vocabulary head, once/token | 4096→32320 | 199.394 µs |

The six per-layer calls total **87.922 µs/layer**, or **3.781 ms across 43 layers**. Including the once-per-token vocabulary head gives **3.980 ms/token**. The M0 optimized-WNA16 trace assigned 26.63% of its approximately 13.3 ms/token summed-kernel screen to Marlin dense work (approximately 3.54 ms), so the isolated Q8 screen is in the same range rather than triggering the dense redesign/stop gate. That cross-method comparison is observational only: the trace pool and exclusive microbenchmark have different scheduling and cache conditions. The TP4 graph-captured layer slice remains the only projection input.

M=2/4 times are retained in `evidence/m2-q8-dense/`; they remain close to M=1 and do not substitute for the required batched-prefill M distribution.
