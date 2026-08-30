# Qwen3.8 QSA INT4 cache evaluation

## Decision

Reject the tested per-token-head INT4 QSA cache for server60's RTX 3090s. Keep
the promoted BF16 cache unchanged.

Q4 passed generated layout and semantic properties, fixed-shape numerical bounds,
and CUDA-Graph replay. It was 8.8% faster than BF16 at M=256 but 3.44 times
slower at M=1. Replacing the multi-operation Hadamard transforms with cached
BF16 Tensor-Core matrices reduced M=1 to 2.03 times BF16. Four decode split-K
schedules remained at 2.06 to 2.09 times BF16, so the fixed 1.25 limit was not
met. No full-model Q4 launch was attempted.

This rejects the tested RHT, asymmetric INT4, float split-dot implementation. It
does not establish that every possible Q4 attention algorithm is slow. Reaching
the gate would require a different quantized-query and quantized-probability
attention design, with new numerical and quality risks beyond a bounded cache
adaptation.

## Capacity model

Each QSA layer currently stores 1,024 BF16 main-cache bytes per token. Packed
INT4 uses 128 bytes each for K and V plus one FP32 scale for each, or 264 bytes
per layer. Twelve QSA layers therefore use 3,168 bytes. The unchanged
compressed-indexer side cache adds 768 bytes, giving 3,936 bytes per token per
rank.

This is a 69.85% reduction from the complete 13,056-byte BF16 QSA cache. Scaling
the measured 156,400-token BF16 allocation by bytes per token gives about
518,790 tokens, well above the model's native 262,144-token limit. Q4 would make
native context capacity easy, but the decode regression failed before a model
launch could test that estimate.

## Preserved contract

The rejected implementation changed only the QSA main cache:

- BF16 remained the default and kept its original FlashAttention cache-update
  path.
- The existing INT4 writer applied the same deterministic randomized Hadamard
  transform, asymmetric quantization, low-nibble-first packing, and hidden
  zero-point scale format already used by vLLM.
- QSA used a separate packed sparse reader. The BF16 reader was not modified.
- Raw and compressed QSA indexer caches, top-k selection, MRoPE, model weights,
  PLE, vision, and scheduling metadata were unchanged.
- The packed reader applied the attention scale once, subtracted K and V zero
  points, merged split-K outputs, then applied the inverse transform once.

## Property-based tests

### Layout and aliasing

A 150-example CPU Hypothesis property generated HND and NHD layouts, one to four
blocks and heads, block sizes 16, 32, and 64, and head sizes 16 through 256. It
checked:

- exact packed row and scale-tail sizes;
- exact returned K/V data-view shapes;
- K/V scale-byte offsets;
- non-overlap between packed data writes and scale views.

The first version of this property failed its adversarial audit. A counterfeit
that returned the scale tail as part of packed data survived because the test
checked only scale placement. Adding an exact returned-shape assertion killed
and shrank the counterfeit to one block, one head, block size 16, and head size
16. A second counterfeit that used the head stride as the token stride also
shrunk to that smallest geometry and failed out of bounds.

### Matrix RHT

A 100-example CPU property checked cached matrix addresses, matrix dimensions,
and `forward * inverse / head_dim` round trips for head sizes 16 through 256 and
one to eight generated rows.

### Sparse attention semantics

A 30-example RTX 3090 property generated one or two requests, one to four pages,
one to four query rows, duplicate selected tokens, and `-1` sentinels. It
compared the complete packed Q4 path against an independent BF16 sparse-attention
reference over the original K/V tensors. The largest observed normalized RMSE
was 0.1566.

The same property killed three test-only GPU counterfeits at its decisive
numerical assertion:

| Counterfeit | Observed failing NRMSE |
| --- | ---: |
| Swap low/high key nibbles | 0.5254, target search reached 1.1194 |
| Omit K zero-point correction | 0.2616, target search reached 0.7231 |
| Omit `1/head_dim` RHT normalization | 0.5627, target search reached 1.1455 |

The third defect was a semantic holdout that did not shape the initial property.
All counterfeit images compiled and reached the oracle; setup or compilation
failures were not counted as kills.

## Fixed-shape RTX 3090 gate

The gate used Qwen TP=4 geometry with six query heads, one KV head, head
dimension 256, and 2,051 selected tokens. Bounds were fixed before execution:

- normalized RMSE at most 0.18;
- cosine similarity at least 0.98;
- finite, bitwise-deterministic CUDA-Graph replay;
- Q4 reader time at most 1.25 times BF16 at M=1 and M=256.

### Initial implementation

| Shape | BF16 | Q4 | Q4/BF16 | NRMSE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| M=1 | 141.26 us | 485.99 us | 3.440x | 0.14832 | reject |
| M=256 | 575.54 us | 531.35 us | 0.923x | 0.15838 | pass |

Cosine similarity was 0.98924 at M=1 and 0.98770 at M=256. CUDA-Graph replay
was finite and bitwise deterministic.

### Decode attribution

A timing-only ablation measured:

| Component | Median time |
| --- | ---: |
| Forward RHT | 137.55 us |
| Inverse RHT | 134.76 us |
| Packed sparse core with RHT bypassed | 258.98 us |
| Complete reader | 483.53 us |

The RHT helper expanded each small transform into several operations. This was a
measured optimization target rather than a guess.

### Cached matrix RHT

A cached BF16 forward/inverse matrix pair replaced the multi-operation helper for
QSA only. Generated semantic properties still passed.

| Shape | BF16 | Q4 matrix | Q4/BF16 | NRMSE | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| M=1 | 137.16 us | 278.43 us | 2.030x | 0.14823 | reject |
| M=256 | 575.95 us | 508.98 us | 0.884x | 0.15837 | pass |

The matrix path removed about 208 microseconds from M=1, but the packed core
still missed the limit.

### Decode retile screen

Four Q4-only schedules changed `BLOCK_N` and split count while leaving the BF16
schedule and all semantics unchanged:

| Schedule | Q4 M=1 | BF16 M=1 | Ratio |
| --- | ---: | ---: | ---: |
| `BLOCK_N=32`, 32 splits | 281.24 us | 134.50 us | 2.091x |
| `BLOCK_N=64`, 32 splits | 283.29 us | 137.83 us | 2.055x |
| `BLOCK_N=64`, 16 splits | 282.88 us | 135.48 us | 2.088x |
| `BLOCK_N=128`, 16 splits | 283.80 us | 137.21 us | 2.068x |

All were noise-level regressions from the 278.43-microsecond matrix baseline.
This falsified split scheduling as the remaining cause.

## Why the bounded design stops here

The Q4 core performs two transformed K dot products and two transformed V dot
products after nibble unpacking and zero-point handling. BF16 performs one K and
one V Tensor-Core dot. The retile screen shows the remaining gap is not empty
splits or narrow tiles.

A faster path would need to quantize transformed queries for integer score dots
and probably quantize softmax probabilities for integer value dots. That changes
the numerical algorithm and adds at least two more dynamic quantization
boundaries. It is a new attention-kernel project, not a reasonable extension of
vLLM's existing Q4 cache format. The fixed gate prevents trading roughly half of
single-stream decode throughput for unused context above the model maximum.

## Delivery state

All Q4 source and test changes are reverted after this report. Production
remains on the original BF16 image with 156,400-token fitted context,
`max_num_batched_tokens=1024`, zero restarts, and zero serving-process swap.

The checksum-bound source patch, overlays, generated tests, counterfeit results,
GPU gates, attribution, and retile measurements are under
[evidence/qwen38-qsa-int4-20260829/](evidence/qwen38-qsa-int4-20260829/).
