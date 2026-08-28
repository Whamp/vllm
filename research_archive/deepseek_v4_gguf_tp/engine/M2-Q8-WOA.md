# M2 — Q8_0 Marlin-diagonal `wo_a` prototype

Decision: **pass the fatal `wo_a` gate.** Dense-family coverage and the full decoder-layer slice remain open.

## Mechanism

The load-time adapter preserves every Q8_0 signed code, offsets it by 128 for Marlin `uint8b128`, converts each FP16 block scale to BF16, and prepares symmetric INT8/group-32 Marlin storage. The existing `_apply_dsv4_wo_a_marlin_diagonal` seam flattens the two rank-local groups, executes one efficient projection, and selects matching group outputs. No BF16 weight cache or steady-state dequantization exists.

## Correctness and capacity

- CPU literal and geometry tests cover code byte order, signed offset, scale orientation, and fail-closed shapes.
- RTX 3090: 6/6 tests pass, including M=1/2/4 grouped diagonal outputs and CUDA Graph replay.
- Kernel output passes the separately pre-registered transformed-weight and original-Q8 normalized class-B windows. The initial elementwise relative assertion failed only near zero because it compared BF16-rounded runtime scales with original FP16 scales; the independent normalized windows distinguish that documented representation delta from kernel/repack errors.
- Exact TP4 layer storage is byte-neutral: raw Q8_0 = prepared weight+scale = **8,912,896 bytes**. Workspace is separate fixed runtime state; no 16 MiB BF16 layer cache is retained.

## Exclusive RTX 3090 benchmark

Five trials per M, 2,500 warm + 5,000 measured graph replays, GPU0 only, ≤1 process/sample, max clock 1650 MHz, canonical service restored zero-swap:

| Input tokens M | Mean graph time | CV |
|---:|---:|---:|
| 1 | **18.438 µs/layer** | 0.198% |
| 2 | **18.415 µs/layer** | 0.028% |
| 4 | **18.466 µs/layer** | 0.023% |

At M=1, 43 layers project to **0.793 ms/token**, below M2's approximately 0.9 ms/token `wo_a` kill threshold. This is an exclusive microbenchmark ceiling, not a serving claim; NCCL/SM contention and the complete layer work graph remain for the TP4 slice.

Evidence: `evidence/m2-q8-woa/`.
