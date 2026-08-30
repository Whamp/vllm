# Qwen3.8 Kernel2 SM86 experiment

## Decision

Do not integrate a hyperconnection replacement from this experiment.

The native BF16 kernel improved the merged-down projection but could not meet
the preregistered 0.8 ms per generated token gate after combining both
projections. W8A16 was a net loss. W4A16 accelerated the up projection, but all
97 real up tensors failed the numerical gate.

Production dispatch did not change.

## Environment

- GPU: NVIDIA GeForce RTX 3090, compute capability 8.6
- GPUs used by microbenchmarks: GPU 0 only
- Power and clock safety policy: 230 W, 210-1650 MHz
- Torch: 2.13.0+cu130
- Control image: `sha256:4b59067e269f313a78f0a698e79261230fb02e3712f42ffd54b3e9ec9be9705a`
- Model: Intel Qwen3.8 Flash Next AutoRound W4A16
- Shapes: merged down `[336,10240]`, up `[10240,320]`
- Token batches: M=1 and M=2
- Timing: 20 graph warmups, 21 alternating samples, 20 graph replays per sample,
  16 pointer-distinct weights

The benchmark excluded weight preparation, packing, allocation, compilation,
and random input generation from timing.

## CuTe result

The existing CuTe skinny GEMM failed before execution with
`CONFIG_UNSUPPORTED_ARCH`. Its implementation accepts SM90 and newer, not
SM86. The experiment did not bypass the architecture guard.

## Marlin result

| Arm | M=1 result | M=2 result | Decision |
| --- | ---: | ---: | --- |
| W8A16 group-128 down | -0.322 ms/step | -0.542 ms/step | Slower than BF16 |
| W8A16 per-channel up | +0.289 ms/step | +0.290 ms/step | Too small; W8 pair is a net loss |
| W4A16 group-32 up | +0.500 ms/step | +0.503 ms/step | Fast but fails quality |
| W4A16 group-64 up | +0.511 ms/step | +0.512 ms/step | Fast but fails quality |
| W4A16 group-128 down | -0.115 ms/step | -0.339 ms/step | Slower and inaccurate |

Positive numbers are projected savings after multiplying per-call latency by
96 down or 97 up calls. These are projection-screen estimates, not serving
speedups.

The W8A16 pair projected -0.033 ms at M=1 and -0.252 ms at M=2. It fails the
mechanism gate.

## Real-weight W4 screen

The CPU screen streamed all 97 production hyperconnection up tensors and used
signed INT4 codes with FP16 scales.

| Scheme | Aggregate NRMSE | Cosine | Passing tensors |
| --- | ---: | ---: | ---: |
| Group 32 | 0.112371 | 0.993717 | 0/97 |
| Group 64 | 0.131970 | 0.991372 | 0/97 |
| Per channel | 0.180234 | 0.984078 | 0/97 |

The gate required NRMSE at most 0.02 and cosine at least 0.9999 for every
tensor. W4 failed by a wide margin. The property suite caught and fixed a scale
floor bug before this scan: `1e-8` rounds to zero in FP16, so the implementation
now uses the exact FP16 minimum subnormal `2^-24`.

## Native BF16 result

The standalone CUDA extension loads two BF16 values per lane, accumulates in
FP32, uses one complete output-row assignment, and reuses weights across M=2.
It retains no additional weight representation.

| Projection | M | Best plan | BF16 control | Native | Speedup | Projected saving |
| --- | ---: | --- | ---: | ---: | ---: | ---: |
| Down | 1 | 256 threads, 1 row/block | 14.848 us | 9.574 us | 1.551x | 0.506 ms |
| Up | 1 | 32 threads, 4 rows/block | 10.765 us | 9.658 us | 1.115x | 0.107 ms |
| Down | 2 | 256 threads, 1 row/block | 14.646 us | 9.763 us | 1.500x | 0.469 ms |
| Up | 2 | 32 threads, 8 rows/block | 10.720 us | 11.949 us | 0.897x | -0.119 ms |

Combined savings were 0.614 ms at M=1 and 0.350 ms at M=2. Both miss the
0.8 ms gate. The M=2 result is the more relevant concurrency discriminator.

All 26 native cases passed deterministic CUDA Graph replay. The selected
results had cosine at least 0.999997 and NRMSE at most 0.002478 against the BF16
control. The extension contains an SM86 cubin, uses 33-48 registers depending
on specialization, 512 bytes of shared memory, and no local-memory spills.

## Final state

The exact Qwen3.8 FP8-QSA plus hierarchical-all-reduce production service was
restored after the experiment. `final-state.txt` records:

- healthy container;
- zero restarts;
- `unless-stopped` restart policy;
- 262,144-token model identity;
- zero host and serving-process swap;
- active GPU safety service and 230 W limits;
- no remaining experiment restore timer.

## Reproduce

From this directory:

```bash
python analyze_results.py
sha256sum -c SHA256SUMS
```

`analyze_results.py` reconstructs `summary.json` from the raw benchmark files.
The maintained benchmark and CUDA sources live in `benchmarks/kernels/`.
