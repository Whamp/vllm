# Qwen3.8 hyperconnection FP8 screen

## Decision

Reject vLLM's existing online 128x128 block-FP8 plus Marlin path for Qwen3.8
hyperconnections on RTX 3090. It saves 0.492449 GiB of registered storage per
rank and passes the bounded numerical and CUDA-Graph gates, but it is 2.9 to
5.7 times slower than BF16 on the exact layer-0 hyperconnection matrix shapes.
The full model was not launched with this method.

This result rejects the generic Marlin implementation for these shapes. It does
not prove that every purpose-built compressed hyperconnection kernel is slow.

## Real-weight reconstruction

`reconstruct_fp8_error.py` streams all 97 hyperconnection groups from the exact
Intel model view and compares E4M3 per-tensor quantization with 128x128 block
quantization. It also calculates the real Marlin tile padding and expanded
BF16 scale storage.

The source index was
`e8893bbecf33dc7f9cdc27f927adbb3886d41531756e8e46cf9cee85499a1201`
with 222,587 tensors.

| Scheme | Aggregate NRMSE | Cosine | Worst tensor NRMSE |
| --- | ---: | ---: | ---: |
| Per tensor | 0.026486 | 0.999623 | 0.028390 |
| Block 128x128 | 0.026313 | 0.999630 | 0.026885 |

Marlin padding changes the storage estimate:

| Allocation | Bytes/rank | GiB/rank |
| --- | ---: | ---: |
| Current BF16 hyperconnections | 1,304,842,240 | 1.215229 |
| Block-FP8 Marlin candidate | 776,079,360 | 0.722780 |
| Reclaimed registered storage | 528,762,880 | 0.492449 |
| Additional Marlin workspaces | 63,632 | 0.000059 |

The earlier 0.61 GiB estimate assumed one byte for every logical weight and
ignored Marlin's padded physical dimensions and scale layout. This measured
estimate replaces it.

## RTX 3090 gate

`gpu-gate.py.txt` loads the real layer-0 matrices, quantizes and repacks them through
vLLM's production `Fp8PerBlockOnlineLinearMethod`, requires
`MarlinFP8ScaledMMLinearKernel`, compares outputs with BF16, captures and
replays CUDA graphs, and times M=1, M=2, and M=256. The preregistered gates were:

- normalized RMSE at most 0.04;
- cosine similarity at least 0.999;
- deterministic finite CUDA-Graph replay;
- FP8 speed at least 90 percent of BF16 for every measured shape.

The numerical and graph gates passed. Performance failed every shape.

| Matrix | M | BF16 us | FP8 Marlin us | BF16/FP8 speed ratio |
| --- | ---: | ---: | ---: | ---: |
| Merged down/injection, 336x10240 | 1 | 26.784 | 112.748 | 0.238 |
| Merged down/injection, 336x10240 | 2 | 26.551 | 138.023 | 0.192 |
| Merged down/injection, 336x10240 | 256 | 48.937 | 139.489 | 0.351 |
| Up, 10240x320 | 1 | 22.218 | 127.177 | 0.175 |
| Up, 10240x320 | 2 | 21.990 | 125.662 | 0.175 |
| Up, 10240x320 | 256 | 34.970 | 127.518 | 0.274 |

The tested image selected packaged SM86 Marlin at runtime. Both matrices passed
bitwise-deterministic CUDA-Graph replay. Numerical NRMSE ranged from 0.02349 to
0.02786, and cosine similarity ranged from 0.999612 to 0.999724.

Three bounded attempts occurred. The first omitted the absolute Hub-cache mount
needed by model-view symlinks. The second omitted vLLM's compilation-config
context in the standalone custom-op test. Neither reached kernel execution. The
third corrected both harness defects and produced the result above. Automatic
rollback restored the original service after every attempt.

## Files

- `reconstruct_fp8_error.py` is the CPU-only all-weight screen.
- `reconstruction-result.json` is its complete per-tensor result.
- `gpu-gate.py.txt` is the exact executed RTX 3090 numerical, graph, and timing
  gate. It is archived as text so repository CUDA API hooks do not treat it as
  maintained test code.
- `gpu-gate-result.json` is the clean machine-readable GPU result.
- `gpu-gate-runtime.log.txt` includes vLLM's runtime-selection messages followed by
  the same result.
- `SHA256SUMS` binds the evidence directory.

The experimental source flag and model changes were reverted after the no-go.
The production Qwen service was restored on image
`sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`
with zero restarts and zero serving-process swap.
