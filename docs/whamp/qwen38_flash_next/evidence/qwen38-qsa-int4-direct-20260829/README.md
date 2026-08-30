# Direct quantized QSA arithmetic screen

## Decision

Use Q8 K plus Q4 V for the first direct-integer QSA implementation. Do not
implement uniform Q4 K/V under the current numerical gate.

The screen used the fixed QSA geometry of six query heads, one KV head, head
dimension 256, 2,051 selected tokens, and 32 splits over 20 deterministic seeds.
It compared ordinary BF16 attention, float-unpack quantized attention, and the
proposed INT8-query plus INT8-weighted-probability arithmetic.

This is CPU arithmetic evidence. It does not establish exact production-writer
parity, SM86 dispatch, CUDA Graph behavior, or performance.

## Uniform Q4 K/V

Uniform Q4 K/V failed the fixed absolute NRMSE limit:

| Metric | Float-unpack Q4 | Direct-integer Q4 | Limit |
| --- | ---: | ---: | ---: |
| Maximum NRMSE | 0.174093 | 0.174598 | 0.17 |
| Minimum cosine | 0.985291 | 0.985208 | at least 0.985 |
| Maximum added NRMSE | n/a | 0.001109 | 0.02 |
| Maximum cosine loss | n/a | 0.000181 | 0.002 |

The direct integer boundaries added little error. Q4 cache quantization dominated
the failed absolute result.

## Q8 K plus Q4 V

Symmetric per-token/head Q8 keys remove most score error while values retain the
capacity-dominant packed Q4 representation.

| Metric | Float-unpack Q8-K/Q4-V | Direct-integer Q8-K/Q4-V | Limit |
| --- | ---: | ---: | ---: |
| Maximum NRMSE | 0.118028 | 0.119039 | 0.17 |
| Minimum cosine | 0.993060 | 0.992941 | at least 0.985 |
| Maximum added NRMSE | n/a | 0.001516 | 0.02 |
| Maximum cosine loss | n/a | 0.000170 | 0.002 |

All fixed arithmetic gates passed.

The mixed main-cache row uses 256 INT8 K bytes, one FP32 K scale, 128 packed
INT4 V bytes, and one FP32 V scale/ZP word, or 392 bytes per QSA layer. Twelve
layers use 4,704 bytes per token. The unchanged compressed-indexer side cache
adds 768 bytes, for 5,472 bytes per token and rank. BF16 uses 13,056 bytes, so
the mixed format reduces complete QSA cache storage by 58.1%.

## Arithmetic

The selected score path quantizes the transformed query to symmetric INT8 and
computes an INT8 by INT8 key dot. The selected value path folds each token's V
scale into its softmax probability, quantizes that weighted probability to
INT8, and computes an INT8 by packed-INT4 value dot with zero-point correction.
The inverse transform and normalization run once after split accumulation.

## Limitations

- Inputs are deterministic Gaussian tensors, not captured model activations.
- The screen uses an explicit randomized Hadamard matrix rather than the
  production writer implementation.
- Matrix products emulate exact integer arithmetic in FP32 on CPU.
- No packed-row paging, sentinel, duplicate-token, or split-merge property is
  exercised here.
- The fixed GPU tests, counterfeits, SASS proof, sanitizers, and performance gate
  remain mandatory.

## Files

- `screen_direct_int4_numerics.py`: uniform Q4 K/V arithmetic screen
- `screen-result.json`: uniform Q4 K/V result
- `screen_q8k_q4v_numerics.py`: selected mixed-cache arithmetic screen
- `screen-result-q8k-q4v.json`: selected mixed-cache result
- `screen-stdout.txt`: uniform-screen status
- `screen-stdout-q8k-q4v.txt`: mixed-screen status
- `SHA256SUMS`: evidence hashes
