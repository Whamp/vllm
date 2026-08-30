# Qwen3.8 hyperconnection `Kernel2` investigation

Status: source and preserved-trace investigation. No GPU work has run for this
study.

## Decision

Investigate the repeated BF16 hyperconnection projections as the next Qwen3.8
decode target. Do not change production until an RTX 3090 gate proves numerical
correctness, CUDA Graph replay, runtime dispatch, and at least a 20% reduction in
the combined M=2 projection time. The intended end-to-end threshold is at least
5% higher concurrency-2 aggregate decode throughput.

Collective elimination and overlap are tracked separately as a deferred
long-shot. The promoted hierarchical all-reduce remains unchanged.

## Production path

Qwen3.8 has 48 decoder layers. Each layer constructs one attention and one MLP
`GatedResidual`. The model adds one final mixer, for 97 hyperconnection groups
per tensor-parallel rank.

The checkpoint configuration fixes:

- hidden size: 2,560;
- hyperconnection streams: 4;
- flattened hyperconnection width: 10,240;
- low-rank width: 320;
- merged down plus injection width: 324 logical rows, padded to 336 physical
  rows.

`GatedResidual.mix()` and `combine_and_mix()` execute this repeated chain:

1. grouped RMS normalization or combined residual plus RMS normalization;
2. BF16 merged down/injection projection, `[M,10240] x [336,10240]^T`;
3. clamped hyperconnection SiLU over the 320 low-rank values;
4. BF16 up projection, `[M,320] x [10240,320]^T`;
5. sigmoid gate mixing over the four residual streams.

Both projections are standard unquantized vLLM linear modules. On RTX 3090
they dispatch through Torch's `F.linear` path to SM80 CUTLASS BF16 WMMA
kernels. The Qwen low-latency selector is enabled only on SM103.

Source owners:

- `vllm/models/qwen4_exp/nvidia/model.py`, decoder construction and execution;
- `vllm/models/qwen4_exp/nvidia/hyperconnection.py`, projection shapes and
  dependencies;
- `vllm/models/qwen4_exp/nvidia/low_latency_gemm.py`, SM103-only Qwen selector;
- `vllm/model_executor/kernels/linear/cute_dsl/_skinny_gemm.py`, existing
  shape-dynamic BF16/FP16 kernel.

## Existing trace evidence

The preserved Nsight Systems trace is:

`/home/will/inference/runtime/qwen38-qsa-fp8-trace/output/qwen38-fp8-c1-c2.nsys-rep`

Its SQLite export is:

`/home/will/inference/runtime/qwen38-qsa-fp8-trace/output/qwen38-fp8-c1-c2.sqlite`

The phase-separated six-episode analysis measured:

| Decode mode | Generated tokens | Wall throughput | Summed GPU time/token | `Kernel2` time/token |
| --- | ---: | ---: | ---: | ---: |
| c=1 | 762 | 43.717 tok/s | 16.381 ms | 3.112 ms |
| c=2 | 1,427 | 65.980 tok/s | 9.745 ms | 2.665 ms |

The c=2 episodes used 762 decode scheduler steps, or 1.8727 generated tokens per
step. If the c=1 `Kernel2` work amortized perfectly across those rows, it would
cost about 1.662 ms per generated token. The measured 2.665 ms leaves an
estimated 1.003 ms/token concurrency-specific gap. This estimate assumes the
same kernel mixture in both modes. A generation-1 versus generation-2 kernel
breakdown must verify that assumption before implementation.

The c=1 trace resolves the two dominant `Kernel2` variants:

| Projection | Grid | Calls in 127 steps | Median-range latency | Weight bytes/call | Effective weight bandwidth |
| --- | --- | ---: | ---: | ---: | ---: |
| merged down/injection | `(8,3,32)` | 12,279 | 12.864-14.816 us | 6,881,280 | about 503 GB/s at 13.68 us |
| up | `(8,80,1)` | 12,278 | 10.752-12.608 us | 6,553,600 | about 587 GB/s at 11.16 us |

The counts equal about 96.7 calls per decode step, matching the 96 decoder-layer
hyperconnections plus the final mixer. The two main families consume about
2.384 ms per c=1 token before smaller `Kernel2` variants.

The 97 groups own 1,302,855,680 bytes, or 1.2134 GiB, of BF16 projection
weights per rank:

| Storage | Bytes/rank | GiB/rank |
| --- | ---: | ---: |
| 96 padded down/injection matrices | 660,602,880 | 0.6152 |
| final unpadded down matrix | 6,553,600 | 0.0061 |
| 97 up matrices | 635,699,200 | 0.5920 |

This is a weight-streaming problem at M=1 and M=2. The c=1 kernels already move
weights at roughly 500-590 GB/s, so a replacement must improve M=2 reuse or
remove surrounding work. Launch reduction alone is unlikely to meet the gate.

## Existing implementation candidates

### Qwen CuTe skinny GEMM

`QWEN4_EXP_GEMM_PLANS` already contains measured M=1 and M=2 plans for the
`(N=336,K=10240)` down/injection projection. Production enables the selector
only on SM103. The underlying CuTe kernel uses register loads, FP32 accumulation,
and a block reduction. Its PDL path disables itself on architectures without
PDL.

This makes an SM86 experiment source-feasible, not supported. Required evidence
is still missing:

- successful SM86 compilation and packaged cubin;
- numerical agreement and deterministic CUDA Graph replay;
- register and spill counts;
- comparison with Torch/CUTLASS using pointer-distinct weights;
- end-to-end dispatch and serving gains.

The table has no `(N=10240,K=320)` up-projection plan. K=320 is not divisible by
the kernel's usual 32-thread by four- or eight-element vector tiles. Any CuTe
candidate needs a narrower vector width or a separate exact-shape kernel.

### Projection epilogues

The dependency chain permits two narrower fusions without changing model math:

- fold hyperconnection SiLU into the down/injection projection epilogue;
- fold sigmoid and four-stream gate mixing into the up projection epilogue.

The trace assigns about 0.2 ms/token each to the standalone SiLU and gate-mix
families at c=1. Their combined upper bound is smaller than the projection
family, but the up epilogue also avoids writing and rereading the 10,240-element
gate tensor.

A one-kernel full hyperconnection is not the first move. The up projection
requires all 320 low-rank outputs, which creates a global dependency between the
two matrix operations. A cooperative persistent kernel would add a large
synchronization and maintenance burden before the simpler exact-shape paths are
measured.

## Gated hypothesis

Outcome: raise warmed concurrency-2 aggregate decode by at least 5% while
preserving single-stream decode, prefill, 262,144-token context, vision,
capability, and memory stability.

Critical segment: 96 repeated pairs of SM80 CUTLASS BF16 hyperconnection
projections per decode step.

Evidence: the preserved trace assigns 2.665 ms per generated c=2 token to
`Kernel2`, and c=2 amortizes this family much less than its 1.8727 rows per step
would predict.

Move: specialize the exact M=1 and M=2 projection shapes for SM86, starting with
a benchmark-only comparison against Torch/CUTLASS. Consider the two epilogue
fusions only after the projection timing identifies their remaining value.

Architecture and runtime gate: RTX 3090, SM86 cubin, BF16 inputs and outputs,
FP32 accumulation, packed row-major weights, CUDA Graph replay, and the current
48-layer Qwen3.8 model contract.

Predicted mediator: at least 20% lower combined M=2 down-plus-up projection time
with no extra persistent weight copy.

Lose conditions:

- M=1 regresses enough to reduce single-stream decode;
- M=2 gains come from warm L2 reuse absent from the 1.2 GiB production stream;
- spills, extra workspace, or graph capture erase the kernel gain;
- changed BF16 accumulation causes an end-to-end capability regression;
- prefill dispatch changes.

Falsifier: the best correct exact-shape candidate improves the pointer-distinct
M=2 projection pair by less than 20%, or a measured kernel gain produces less
than 5% end-to-end concurrency-2 improvement.

## Validation plan

GPU execution is deferred while server60 runs the long evaluation.

The later maintenance-window gate must run in this order:

1. regenerate the c=1 versus c=2 `Kernel2` breakdown from the preserved trace;
2. compare Torch/CUTLASS and each exact-shape candidate on M=1 and M=2 with
   pointer-distinct weights and CUDA Graph replay;
3. compare against an independent FP32 reference and the existing BF16 result;
4. require deterministic graph replay;
5. run Compute Sanitizer memcheck and racecheck;
6. inspect the SM86 cubin, register count, spills, and relevant SASS;
7. run the two projections inside the full hyperconnection chain;
8. run a matched production serving A/B with zero swap and unchanged GPU safety
   controls.

No production selector or CUDA kernel should change before steps 1-3 establish
the mechanism.
