# Qwen3.8 hyperconnection INT8 exact-shape screen

## Decision

Reject vLLM's generic Cutlass W8A8 dense-linear path for Qwen3.8
hyperconnections on RTX 3090. It passed deterministic CUDA Graph replay, but it
was 2.5 to 3.8 times slower than BF16. The merged down-and-injection projection
also missed the preregistered numerical bound.

This result does not reject INT8 storage. It rejects per-channel weight scaling
through the generic dynamic-activation Cutlass path. Any next attempt needs
separate component or K-block scales and a kernel specialized for the two skinny
hyperconnection shapes.

## Environment

- GPU: one server60 RTX 3090, compute capability 8.6
- Power policy: 230 W, 210-1650 MHz
- Runtime image: `sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`
- Model: Intel Qwen3.8 Flash Next AutoRound, layer 0 real weights
- Kernel: `CutlassInt8ScaledMMLinearKernel`
- Quantization: symmetric per-output-channel INT8 weights and dynamic symmetric
  per-token INT8 activations
- Shapes: merged down/injection `[336, 10240]` and up `[10240, 320]`
- Token counts: M=1, M=2, and M=256
- Samples: five CUDA-event trials after 20 warmups

## Acceptance

- normalized RMSE at most 0.02
- cosine similarity at least 0.9999
- candidate throughput at least 90% of BF16 at every tested shape
- finite, bitwise-equal CUDA Graph replay

## Results

| Matrix | M | BF16 us | INT8 us | BF16/INT8 speed ratio | NRMSE | Cosine |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| merged down/injection | 1 | 27.375 | 87.015 | 0.315 | 0.02918 | 0.999575 |
| merged down/injection | 2 | 27.536 | 89.018 | 0.309 | 0.02504 | 0.999687 |
| merged down/injection | 256 | 48.681 | 123.464 | 0.394 | 0.02584 | 0.999666 |
| up | 1 | 23.013 | 87.384 | 0.263 | 0.01274 | 0.999919 |
| up | 2 | 22.788 | 88.092 | 0.259 | 0.01417 | 0.999900 |
| up | 256 | 34.447 | 89.620 | 0.384 | 0.01366 | 0.999907 |

The two layer-0 matrices use 6,759,744 bytes of INT8 weights and scales versus
13,434,880 BF16 weight bytes, a 49.7% reduction. That storage result is not
useful with the measured latency and merged-down error.

## Files

- `gate.py.gz`: exact script executed for the shape gate, compressed without timestamps
- `result.json`: clean machine-readable result
- `runtime.log.gz`: complete runtime output, including selected-kernel evidence
- `SHA256SUMS`: archive hashes
