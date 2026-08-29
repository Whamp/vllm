# Qwen3.8 QSA INT8 cache evaluation

## Decision

Reject per-token-head INT8 for the Qwen3.8 QSA main K/V cache on server60's
RTX 3090s. Keep the promoted BF16 cache unchanged.

The implementation reused vLLM's existing dynamic per-token-head INT8 writer,
inline FP32 scale layout, and CUDA-Graph-safe scale views. It added only an INT8
branch to QSA's sparse reader. Writer arithmetic, numerical accuracy, M=1
latency, and CUDA-Graph replay passed. M=256 reader time was 26.32 times BF16,
so no full-model INT8 launch was attempted.

## Capacity model

Each QSA layer currently stores 1,024 BF16 main-cache bytes per token. Dynamic
INT8 stores 512 data bytes plus one FP32 scale for K and V, or 520 bytes per
layer. Across twelve QSA layers, the main cache falls from 12,288 to 6,240 bytes
per token. The unchanged compressed-indexer side cache adds 768 bytes, giving
7,008 bytes per token per rank.

That is a 46.32% reduction from the complete 13,056-byte BF16 QSA cache. Scaling
the measured 156,400-token BF16 allocation by bytes per token gives about
291,375 tokens, above the model's native 262,144-token length. The capacity
benefit is large enough to justify the kernel experiment, but not a 26-fold
prefill-kernel regression.

## Implementation boundary

The rejected implementation changed only the QSA main cache:

- `Qwen4ExpQSAFlashAttentionImpl` inherited `TritonAttentionImpl` so QSA reused
  the existing writer and inline scale views.
- `Qwen4ExpQSAFlashAttentionBackend` published the existing 520-byte INT8 row
  layout.
- The sparse reader loaded each selected token's K/V scales, converted INT8 to
  BF16, and retained the existing attention math.
- BF16 remained the default and unchanged fallback.
- QSA raw-key and compressed-key side caches, MRoPE, top-k selection, model
  weights, and PLE were unchanged.

## Property-based layout search

A 100-example Hypothesis property searched:

- HND and NHD physical layouts;
- one to four blocks and KV heads;
- block sizes 16, 32, and 64;
- head sizes 16, 32, 64, 128, and 256;
- positive and negative scale payloads.

For every generated geometry, it checked the declared 520-byte-style row
arithmetic generalized to the drawn head size, exact K/V scale-tail byte
locations, and non-overlap between data writes and scale views. The final search
had 100 passes and no invalid cases.

A temporary counterfeit replaced the token stride with the head stride. The
property found and shrank two failures:

- an NHD corruption at one block, two heads, block size 16, and head size 16;
- an HND out-of-bounds view at one block, one head, block size 16, and head size
  16.

The counterfeit was removed and the same property passed. The property test is
archived as evidence because the rejected QSA INT8 code is not retained.

## RTX 3090 gate

The one-GPU gate fixed these limits:

- normalized RMSE at most 0.02 versus BF16;
- cosine similarity at least 0.999;
- exact dynamic writer codes on a controlled non-boundary input;
- matching random-input scales;
- finite, bitwise-deterministic CUDA-Graph replay of write plus attend;
- INT8 sparse-reader time at most 1.25 times BF16 at M=1 and M=256.

### Numerical and graph results

| Metric | Result | Bound |
| --- | ---: | ---: |
| Normalized RMSE | 0.010372 | <= 0.02 |
| Cosine similarity | 0.999946 | >= 0.999 |
| Maximum absolute error | 0.009102 | recorded only |
| Controlled writer codes | exact | exact |
| Writer scales | match | match |
| CUDA-Graph replay | bitwise equal, finite | required |

### Kernel results

| Shape | BF16 reader | INT8 reader | INT8/BF16 | Decision |
| --- | ---: | ---: | ---: | --- |
| M=1 | 141.00 us | 144.69 us | 1.026x | pass |
| M=256 | 572.86 us | 15,080.24 us | 26.324x | reject |

The M=256 measurements were stable across five samples. INT8 ranged from
15,059.86 to 15,105.47 microseconds. BF16 ranged from 572.14 to 573.47.

The dynamic writer cost was smaller but still measurable:

| Shape | BF16 writer | INT8 writer | INT8/BF16 |
| --- | ---: | ---: | ---: |
| M=1 setup | 62.46 us | 83.01 us | 1.329x |
| M=256 setup | 60.65 us | 83.32 us | 1.374x |

## Why it loses

The selected QSA sparse reader converts every selected INT8 K and V value to
FP32, applies its per-token scale, then converts to BF16 before the existing
Tensor-Core dot products. M=1 remains launch and latency dominated, so the work
adds only 2.6%. At M=256, the conversion repeats over every row and each of the
2,051 selected tokens, producing the same large prefill failure seen in the
software E4M3 experiment.

This rejects the direct dequantize-then-BF16 QSA reader. It does not reject a
purpose-built integer attention kernel whose query representation and dot
products avoid that conversion. Building such a kernel is not justified before
Q4 is evaluated because Q4 already has a dedicated split-dot reader in vLLM.

## Delivery state

All INT8 source and test changes are reverted after this report. Production
remains on the original image, BF16 QSA cache, 156,400-token fitted context,
`max_num_batched_tokens=1024`, zero restarts, and zero serving-process swap.

The checksum-bound rejected patch, runtime overlay, GPU result, and property
search evidence are under
[evidence/qwen38-qsa-int8-20260829/](evidence/qwen38-qsa-int8-20260829/).
