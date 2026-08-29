# Direct quantized QSA INT4 design for SM86

## Status

This document preregisters a new Qwen3.8 QSA cache reader. No implementation or
performance claim exists yet.

The prior INT4 cache reader is rejected. It unpacked INT4 values into floating
point, performed two transformed K dots and two transformed V dots, and remained
2.03 times slower than BF16 at M=1 after its best bounded tuning. See
[QSA-INT4-CACHE.md](QSA-INT4-CACHE.md).

The replacement keeps the proven packed cache writer and row layout but changes
the attention arithmetic. It quantizes transformed queries and value-weighted
softmax probabilities so the score and value products execute as integer dot
products on RTX 3090 Tensor Cores.

## Fixed environment and result boundary

| Item | Contract |
| --- | --- |
| GPU | NVIDIA RTX 3090, SM86 |
| Model | Intel Qwen3.8 Flash Next AutoRound |
| Main cache | `int4_per_token_head`, asymmetric packed INT4 |
| Cache row | packed K, FP32 K scale/ZP, packed V, FP32 V scale/ZP |
| Query geometry | 6 query heads, 1 KV head, head dimension 256 |
| Selection width | 2,051 tokens |
| Decode shape | M=1 and M=2 |
| Prefill shape | M=256 |
| Fallback | unchanged BF16 QSA path |
| Power policy | 230 W, 210-1650 MHz |

The kernel result becomes useful when it returns one sparse-attention output for
all query heads. Timings include query transformation, dynamic quantization,
sparse score computation, softmax, value accumulation, inverse transformation,
and split merging.

The fixed microbenchmark limit is 1.25 times the matched BF16 reader at M=1 and
M=256. A candidate that misses either shape does not advance to a model launch.

## Preserved storage and writer contract

The implementation must reuse vLLM's existing per-token-head INT4 writer and
inline scale views in
[`vllm/v1/attention/ops/int4_per_token_head.py`](../../../vllm/v1/attention/ops/int4_per_token_head.py).
It must preserve:

- randomized Hadamard transformation before K/V quantization;
- asymmetric low-nibble-first INT4 packing;
- one FP32 scale word per token and head with the zero point in its low nibble;
- HND and NHD paging, block tables, duplicate selected tokens, and `-1`
  sentinels;
- raw and compressed QSA indexer caches, top-k selection, MRoPE, and scheduling;
- byte-for-byte BF16 behavior when INT4 is disabled.

For head dimension 256, each main-cache token uses 264 bytes per QSA layer
instead of 1,024 BF16 bytes. Across 12 QSA layers, the main cache uses 3,168
bytes per token. The unchanged compressed-indexer side cache adds 768 bytes, for
3,936 bytes per token and rank.

## Quantized score contract

Let `H` denote the existing unnormalized randomized Hadamard transform, `d=256`,
`q_h = H(q)`, and let a packed key reconstruct as
`k_h ~= (k_code - k_zp) * k_scale`.

For each query head:

1. Compute `q_h` with the existing transform signs and ordering.
2. Choose one positive symmetric query scale `q_scale`.
3. Quantize `q_code = round(clamp(q_h / q_scale, -127, 127))`.
4. Compute the integer score
   `s_int = dot(q_code, k_code) - k_zp * sum(q_code)`.
5. Convert once with
   `score = s_int * q_scale * k_scale / (d * sqrt(d))`.

The `1/d` factor compensates for the unnormalized transform. The attention scale
is applied exactly once. Invalid cache entries receive negative infinity before
softmax.

The SM86 implementation must lower the inner product to integer Tensor-Core or
DP4A instructions. Unpacking INT4 to floating point before the dot is not this
design.

## Quantized value contract

A packed transformed value reconstructs as
`v_h ~= (v_code - v_zp) * v_scale`. After the online softmax produces a
nonnegative unnormalized probability `p_t` for token `t`, define
`r_t = p_t * v_scale_t`.

For each split and query head:

1. Choose one positive `r_scale` over the split's valid `r_t` values.
2. Quantize `r_code_t = round(clamp(r_t / r_scale, 0, 127))`.
3. Compute
   `o_int = sum_t r_code_t * v_code_t`.
4. Compute the zero-point correction
   `z_int = sum_t r_code_t * v_zp_t`.
5. Accumulate `(o_int - z_int) * r_scale` in FP32.
6. Apply ordinary online-softmax max and normalization updates.
7. Merge split outputs in fixed split order.
8. Apply the inverse randomized Hadamard transform and `1/d` exactly once.

Folding `v_scale_t` into the quantized probability is required. Quantizing `p_t`
without the token's V scale changes the value equation and is a counterfeit.

## Kernel schedule

The first implementation uses three caller-stream operations:

1. fused randomized-Hadamard query transform plus symmetric INT8 quantization;
2. paged sparse INT8-query by packed-INT4-key score, online softmax, and
   INT8-weighted-probability by packed-INT4-value accumulation;
3. fused inverse randomized-Hadamard transform, normalization, and BF16 store.

All temporary tensors use caller-owned, address-stable workspace sized during
profile warmup. Decode must allocate nothing after CUDA Graph capture. M=256 may
use a separate schedule but must implement the same equations.

If cached BF16 matrix transforms alone keep M=1 above the speed limit, the only
permitted transform tuning step is a fused 256-element SM86 Hadamard kernel.
Retiling the rejected floating-point core is closed by prior evidence.

## Numerical gates

The direct path must pass every gate below before a full-model launch:

- normalized RMSE at most 0.17 against an independent BF16 sparse-attention
  reference;
- cosine similarity at least 0.985;
- no more than 0.02 additional NRMSE and no more than 0.002 lower cosine than the
  archived matrix-RHT INT4 reader on the same seeded cases;
- finite, bitwise-equal CUDA Graph replay;
- exact caller-stream ordering without default-stream dependence;
- stable lower-index handling for duplicate and sentinel selections;
- Compute Sanitizer memcheck and racecheck with zero findings.

These kernel bounds only decide whether a model test is warranted. Promotion
still requires deterministic, streaming, tool, post-tool, reasoning, vision,
near-ceiling NIAH, full BenchLocal, concurrency-two, and matched performance and
VRAM evidence against BF16.

## Property search

Property-based tests will generate one or two requests, one to four pages, one
to four query rows, duplicate selected tokens, `-1` sentinels, random packed
codes, independent key/value scales and zero points, and split counts from 1 to
32. The independent oracle dequantizes literals and evaluates ordinary BF16
attention without calling production pack, transform, score, or merge helpers.

The property must kill these compiled counterfeits at the numerical assertion:

1. omit query scaling;
2. omit key zero-point correction;
3. omit the `1/d` transformed-score normalization;
4. quantize `p_t` without multiplying by `v_scale_t`;
5. omit value zero-point correction;
6. reuse one probability scale across splits while merging as if scales were
   independent;
7. apply the inverse transform or attention sink more than once.

Hypothesis runs at least 50 CUDA cases with shrinking and replay. Named exact
Qwen shapes remain separate tests because generated small cases do not establish
serving-shape dispatch or performance.

## Dispatch and mechanism proof

Acceptance requires all of the following on server60:

- packaged SM86 code in the exact runtime image;
- runtime selection of the direct INT4 QSA path only when explicitly enabled;
- SASS containing integer Tensor-Core or DP4A instructions in the score and
  value kernels;
- no float expansion of packed K/V before the named integer dots;
- measured M=1 and M=256 kernel timing within 1.25 times BF16;
- an ablation showing the gain disappears when the direct integer core is
  replaced by the archived floating-point split-dot core.

If the integer instructions dispatch but the end-to-end reader misses the speed
limit, the candidate is rejected unless a profile identifies one bounded,
mechanism-specific operation with enough causal budget to cross the gate.
