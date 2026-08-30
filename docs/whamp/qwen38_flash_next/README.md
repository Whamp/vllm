# Qwen3.8 Flash Next on server60

This directory records the RTX 3090 work behind server60's Intel AutoRound
Qwen3.8-Flash-Next service. Each report separates source eligibility, kernel
validation, serving measurements, rejected experiments, and the accepted
production contract.

## Accepted production stack

| Item | Accepted value |
| --- | --- |
| Model | Intel `Qwen3.8-Flash-Next-W4A16-AutoRound` |
| Model revision | `861536dda5bcb208376fc4cd879b2bf76bece9fe` |
| PLE | Primitive NVFP4 sidecar, revision `da8b39586016d8325ac619be28ad77d6296625ec` |
| Main QSA cache | Calibrated per-layer E4M3 |
| Context | 262,144 tokens |
| Maximum sequences | 2 |
| Batch-token budget | 1,024 |
| GPU-memory utilization | 0.98 |
| Collectives | Hierarchical islands `0,1;2,3`, PYNCCL fallback |
| Hyperconnection path | Native SM86 BF16 Kernel2 for exact M=1/2 shapes |
| Production image | `sha256:acff9d8e08096a2265b23e50f5ff0d52a3f1e95ffa91e2fb099346e274a9b735` |
| Rollback | Pre-Kernel2 hierarchical image and BF16-QSA service contracts |

The Kernel2 same-image ablation measured +7.84% c=1 decode and +2.69% c=2
aggregate decode. Prefill changed by less than 0.3%. BenchLocal quick scored
28/30, and the established tool, post-tool, multimodal, two-stream, and
261,492-token NIAH checks passed. See
[the Kernel2 report](KERNEL2-HYPERCONNECTION.md).

## Accepted changes

- [GPU memory baseline](GPU-MEMORY-BASELINE.md) attributes registered and
  unregistered allocations before optimization.
- [QSA top-k buffer budget](QSA-TOPK-BUFFER-1024.md) reduced the batch-token
  budget from 2,048 to 1,024 and recovered context without reducing decode.
- [Runtime-bounded RoPE](QSA-ROPE-BOUND.md) stopped materializing unused rows.
- [GPU utilization ceiling](GPU-MEMORY-UTILIZATION-0968.md) records the earlier
  stable 202,400-token profile and its safety evidence.
- [Calibrated FP8 QSA](QSA-FP8-CACHE.md) reached native 262,144-token context
  while retaining BF16 rollback.
- [Hierarchical all-reduce](HIERARCHICAL-ALL-REDUCE.md) replaced decode-sized
  PCIe ring collectives on the measured path.
- [Native SM86 Kernel2](KERNEL2-HYPERCONNECTION.md) specializes the repeated
  Qwen hyperconnection BF16 projections for M=1 and M=2.

## Rejected experiments

- [Capacity kernel gates](CAPACITY-KERNEL-GATES.md) close the generic
  hyperconnection FP8/INT8 candidates that failed quality or performance.
- [QSA INT8 cache](QSA-INT8-CACHE.md) passed correctness but was too slow on
  RTX 3090.
- [QSA INT4 cache](QSA-INT4-CACHE.md) passed numerical and graph checks but
  missed the decode reader-time limit.
- [Direct QSA Q8-K/Q4-V](QSA-INT4-DIRECT-DESIGN.md) reduced bytes but failed its
  serving-shape reader-performance gate.
- [Kernel2 investigation](KERNEL2-HYPERCONNECTION.md) records why CuTe, W8A16,
  W4A16, and the original service-budget gate were rejected before Will approved
  the smaller native BF16 gain for a production A/B.

## Completion and evidence

[QWEN38-MEMORY-CAPACITY-COMPLETION.md](QWEN38-MEMORY-CAPACITY-COMPLETION.md)
records the earlier memory-capacity audit. Later FP8-QSA, hierarchical
all-reduce, stability, utilization, and Kernel2 reports supersede its production
identity while retaining its rejected-experiment history.

Raw evidence lives under [`evidence/`](evidence/). The Kernel2 production bundle
is checksum-bound and reproducible at
[`evidence/qwen38-kernel2-production-20260830/`](evidence/qwen38-kernel2-production-20260830/README.md).
