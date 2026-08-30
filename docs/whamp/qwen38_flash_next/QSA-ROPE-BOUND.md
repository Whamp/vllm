# Qwen3.8 runtime-bounded RoPE cache

## Decision

Promote a runtime-bounded Qwen3.8 RoPE cache on server60.

The model previously materialized 1,048,576 BF16 MRoPE rows because the generic
Qwen2.5-VL implementation multiplies the model maximum by four for video input.
Qwen3.8's inherited Qwen3-VL position construction is token bounded. Capping
materialized rows at the supported 262,144-token model length therefore
preserves every legal text, image, video, and continuation position.

The change reclaimed exactly 96 MiB per GPU and increased automatically fitted
context from 156,400 to 167,600 tokens. Exact-final-image decode retained 98.06%
of the promoted control, prefill retained 99.42%, and concurrency-2 aggregate
throughput was 0.78% higher. Exact retrieval passed at 160,035 API prompt tokens.

## Mechanism

`MRotaryEmbedding` receives the model's 262,144-position maximum, multiplies it
by four, and builds 1,048,576 rows. The promoted change leaves that construction
and its frequency calculation untouched, then clones only the first 262,144
rows into new storage. This matters: a slice alone would keep the original
128 MiB storage alive.

The helper fails closed when:

- the requested bound is zero or negative;
- the source cache has fewer rows than the model supports.

It returns without allocating when another QSA layer has already applied the
same bound. The twelve full-attention layers share one cached rotary embedding,
so the first call performs the clone and later calls are idempotent.

## Multimodal safety proof

The Qwen3.8 multimodal class inherits Qwen3-VL position generation. For each
media block, the position range advances by `max(height, width)` while the
placeholder consumes `height * width` tokens. Both dimensions are positive, so
media positions cannot advance faster than token count. Prefix and suffix text
advance one position per token. The initial MRoPE delta is therefore nonpositive,
and continuation positions also stay below sequence length.

A 150-example Hypothesis property generated:

- image and video requests;
- one to eight video frames;
- grid heights and widths from 1 to 12;
- merge sizes 1 and 2;
- prefix and suffix lengths from 0 to 32;
- continuation lengths from 1 to 64.

Every case had nonnegative positions, maximum position below input-token count,
nonpositive continuation delta, and continuation positions below the resulting
sequence length. There were no invalid examples.

Two counterfeits proved the tests were live:

1. returning a cache slice without cloning kept the original 4,096-byte test
   storage rather than the required 1,024 bytes;
2. shifting Qwen3-VL positions by sequence length shrank to a three-token 1×1
   image and failed with maximum position 5.

Both counterfeits were removed before final validation.

## Measured memory effect

The storage-deduplicated diagnostic produced 40 rank/stage reports. The raw
reports and relative-path SHA-256 manifest verify byte-for-byte after copying
off server60.

| Registered category | Previous | Candidate | Change |
| --- | ---: | ---: | ---: |
| RoPE cache | 128 MiB | 32 MiB | -96 MiB |
| QSA top-k buffers | 192.28 MiB baseline | 96.14 MiB | previously promoted |

The candidate's total registered-storage change against the original 2,048-token
baseline is 192.14 MiB per rank because that comparison includes both accepted
changes. Category accounting isolates the RoPE change at exactly 96 MiB.

| Capacity metric | QSA-1024 control | RoPE bound | Change |
| --- | ---: | ---: | ---: |
| Auto-fitted context | 156,400 | 167,600 | +11,200, +7.16% |
| RoPE materialized rows | 1,048,576 | 262,144 | -75% |
| Final physical free VRAM, GPU 0 | not remeasured here | 1,836 MiB | recorded |
| Final physical free VRAM, GPUs 1–3 | not remeasured here | 1,880 MiB | recorded |

The final profile still uses BF16 QSA K/V. No cache precision or model weight
changed.

## Matched performance

The exact-final-image benchmark used three decode warmups plus five measured
256-token streams and one prefill warmup plus three cache-busted 17,825-token
prompts.

| Metric | Promoted QSA-1024 control | Final RoPE image | Retention |
| --- | ---: | ---: | ---: |
| Decode | 43.9422 tok/s | 43.0912 tok/s | 98.06% |
| Cache-busted prefill | 1,549.8940 tok/s | 1,540.8430 tok/s | 99.42% |
| Concurrency-2 aggregate | 54.7205 tok/s | 55.1491 tok/s | 100.78% |

All results used the same Intel AutoRound artifact, Primitive quantized PLE,
vision stack, EP=4, BF16 QSA cache, `max_num_seqs=2`, and
`max_num_batched_tokens=1024`.

## Correctness and quality

The exact final image passed:

- deterministic `PARIS` output;
- automatic tool choice and post-tool continuation;
- synthetic image inference;
- two concurrent streamed requests;
- exact `VIOLET ORBIT 9137` retrieval at 160,035 API prompt tokens;
- BenchLocal quick at 26/30, with ToolCall-15 12/15 and InstructFollow-15
  14/15;
- zero residual running or waiting requests after acceptance;
- zero serving-process swap and zero restarts.

The BenchLocal total is unchanged from the QSA-1024 control.

## Operational incident

The first accepted promotion was unintentionally rolled back after validation.
The cleanup command stopped the timer successfully, then returned code 5 for an
already-unloaded service unit. `set -e` triggered the failure trap, which
correctly restored the original image. No unsafe or unknown service state
occurred.

The identical validated image was promoted again. Final cleanup stops timer and
service units separately and tolerates absent-unit errors. Fresh checks then
confirmed the promoted image, 167,600-token context, deterministic and
multimodal output, zero swap, zero restarts, active GPU safety service, and no
active rollback timer.

## Final service

The live service uses:

- image
  `sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b`;
- Intel AutoRound revision
  `861536dda5bcb208376fc4cd879b2bf76bece9fe`;
- Primitive PLE revision
  `da8b39586016d8325ac619be28ad77d6296625ec`;
- BF16 QSA cache;
- 167,600-token fitted context;
- `max_num_seqs=2` and `max_num_batched_tokens=1024`;
- the fixed 230 W and 210–1650 MHz GPU safety policy.

The checksum-bound reports, raw diagnostic reports, tests, runtime contracts,
benchmarks, quality results, and final state are under
[evidence/qwen38-rope-bound-20260829/](evidence/qwen38-rope-bound-20260829/).
