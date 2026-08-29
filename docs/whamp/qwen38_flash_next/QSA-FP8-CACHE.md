# Qwen3.8 QSA FP8 cache evaluation

## Decision

Reject FP8 for the Qwen3.8 QSA main K/V cache on server60's RTX 3090s.
Keep the promoted BF16 cache unchanged.

E4M3 storage passed the writer, numerical, and CUDA-Graph gates and was neutral
for M=1 decode. Its software-decoded M=256 kernel was 28.46 times slower than
BF16. E5M2 compiled natively on SM86 but missed the preregistered numerical
bound. Neither format reached full-model loading because the kernel gates had
already rejected it.

This result is specific to the pinned QSA Triton implementation, RTX 3090 SM86,
and the measured M=1 and M=256 shapes. It does not say that FP8 KV cache is bad
on architectures with native E4M3 conversion support.

## Capacity model

The twelve QSA layers currently store 12,288 bytes of BF16 main K/V per token
and 768 bytes of BF16 compressed-indexer side cache, for 13,056 bytes per token
per rank.

An FP8 main cache would reduce the main K/V term to 6,144 bytes while leaving
the side cache unchanged, for 6,912 bytes per token. That is a 47.06% reduction
in the complete QSA cache payload. It would likely make the model's native
262,144-token context fit, but no capacity estimate can justify a kernel that is
28 times slower on the prefill shape.

## Acceptance bounds

The one-GPU RTX 3090 gate fixed these limits before the final run:

- normalized RMSE at most 0.05 versus BF16 attention output;
- cosine similarity at least 0.995;
- bitwise-deterministic, finite CUDA-Graph replay;
- FP8 kernel time at most 1.25 times BF16 at M=1 and M=256.

The gate used TP=4 QSA geometry with six local query heads, one local KV head,
head dimension 256, and a 2,051-token sparse selection for timing.

## Format results

### Typed E4M3

The installed Triton compiler rejected E4M3FN on SM86:

```text
ValueError: type fp8e4nv not supported in this architecture.
The supported fp8 dtypes are ('fp8e4b15', 'fp8e5')
```

This is a compiler and architecture-format gate, not a timing result.

### Typed E5M2

E5M2 compiled, but its BF16-relative output error was:

| Metric | Result | Bound |
| --- | ---: | ---: |
| Normalized RMSE | 0.059410 | <= 0.05 |
| Cosine similarity | 0.998239 | >= 0.995 |
| Maximum absolute error | 0.011963 | recorded only |

The NRMSE gate rejected E5M2. The bound was not relaxed.

### Software-decoded E4M3

The E4M3 reader loaded raw bytes and reconstructed FP32 values from sign,
exponent, and mantissa bits inside Triton. This preserved the better E4M3
mantissa without using the unsupported typed conversion.

Two pipeline stages required 102,400 bytes of shared memory at M=256, exceeding
RTX 3090's 101,376-byte launch limit by 1,024 bytes. Reducing only the FP8 path
to one pipeline stage cleared that architecture gate.

The final E4M3 result was:

| Metric | Result | Bound |
| --- | ---: | ---: |
| Writer bytes | exact E4M3 match | exact |
| Normalized RMSE | 0.031408 | <= 0.05 |
| Cosine similarity | 0.999507 | >= 0.995 |
| Maximum absolute error | 0.007324 | recorded only |
| CUDA-Graph replay | bitwise equal, finite | required |

Performance rejected it:

| Shape | BF16 | E4M3 | FP8/BF16 time |
| --- | ---: | ---: | ---: |
| M=1 | 139.55 us | 140.34 us | 1.006x |
| M=256 | 588.54 us | 16,750.52 us | 28.461x |

The M=256 result is stable across five samples. FP8 samples ranged from
16,744.27 to 16,757.86 microseconds; BF16 ranged from 588.34 to 588.75.

## Why it loses

SM86 can store E4M3 bytes, but the selected Triton path cannot convert E4M3FN
values directly. The software reader performs sign, exponent, mantissa, bitcast,
scale, and BF16 conversion work for every selected K and V element. At M=1 the
kernel remains latency dominated and the added work is hidden. At M=256 the
conversion multiplies across every row and selected token, overwhelming the
bytes saved.

E5M2 avoids that software conversion but gives up enough mantissa precision to
miss the numerical gate.

## Delivery state

All FP8 source changes were experimental and are reverted after this report.
The production service remains on:

- image `sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`;
- BF16 QSA cache;
- 156,400-token fitted context;
- `max_num_batched_tokens=1024`;
- zero serving-process swap and zero restarts.

The full evidence and rejected source patch are under
[evidence/qwen38-qsa-fp8-20260829/](evidence/qwen38-qsa-fp8-20260829/).
The next cache-format experiment is Q4, with a separate numerical and
performance design rather than a renamed FP8 path.
