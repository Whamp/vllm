# Qwen3.8 QSA FP8 cache

## Decision

Promote the calibrated direct-E4M3 QSA cache as server60's Qwen3.8 production default.
Keep the BF16 profile as the rollback path. The current production image also
enables island-aware hierarchical all-reduce. See
[Qwen3.8 hierarchical all-reduce](HIERARCHICAL-ALL-REDUCE.md).

The promoted profile stores the twelve QSA layers' main K/V cache as E4M3 bytes,
decodes those bytes in the sparse QSA reader, and applies an exact calibrated
scale for each layer's K and V tensors. Raw and compressed indexer caches remain
unchanged. The current 0.98 profile reaches the native 262,144-token model limit
with 421,608 aggregate KV tokens and 1.61x maximum concurrency.

The earlier FP8 no-go in this file tested a generic software conversion path. It
was valid for that implementation, which took 28.46 times BF16 reader time at
M=256. It does not describe the promoted reader. The direct reader folds cache
scales into score and output scaling and selects SM86 profiles by request shape.
Its final one-GPU gate stayed within 1.25 times BF16 reader time at every tested
shape.

## Production identity

| Item | Value |
| --- | --- |
| Model | Intel `Qwen3.8-Flash-Next-W4A16-AutoRound` |
| Model revision | `861536dda5bcb208376fc4cd879b2bf76bece9fe` |
| Derived config SHA-256 | `932cbf4d5dc50efa395db095ea3664fd6ef7672886332b2d0307cd9aa28ac9cf` |
| Derived index SHA-256 | `e8893bbecf33dc7f9cdc27f927adbb3886d41531756e8e46cf9cee85499a1201` |
| Primitive PLE revision | `da8b39586016d8325ac619be28ad77d6296625ec` |
| Accepted base image | `sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b` |
| Base FP8 image | `sha256:61971a78222e89335d84a7f4d72b0e8842619a4a29564582a58ff328af48abb9` |
| Current hierarchical image | `sha256:4b59067e269f313a78f0a698e79261230fb02e3712f42ffd54b3e9ec9be9705a` |
| Scale-file SHA-256 | `554d68aa917bdcac3ec7e5c14a8ca1421182d0b899da30b6c54501b72aefdcf3` |
| Cache dtype | `fp8_e4m3` |
| Context | 262,144 tokens |
| Maximum sequences | 2 |
| Batch-token budget | 1,024 |
| GPU-memory utilization | 0.98 |
| CUDA graphs | `FULL_DECODE_ONLY` |
| Endpoint | `http://server60:30002/v1` |

The production Compose file uses `restart: unless-stopped`. It pins the image by
digest and mounts the scale file read-only. `restore-bf16.sh` stops the FP8
Compose project before running the preserved BF16 rollback. `restore-fp8.sh`
recreates the promoted service and verifies image identity, model identity,
context, restart policy, health, and zero swap.

## Scale calibration

The runtime rejects missing layers, nonpositive scales, malformed files, and
fallback or global scales. It reads twelve layer-specific K/V entries from
`qsa-fp8-scales.json`.

Calibration used the Intel checkpoint and the live Primitive PLE path. The live
workload included text and two real image prompts. Every rank reported every
QSA layer. The merge took the maximum K and V absolute value across TP ranks,
applied a 1.125 safety margin, and divided by E4M3's maximum finite magnitude of
448.

The merged scale file and four rank reports are under
[`evidence/qwen38-qsa-fp8-e4m3-20260829`](evidence/qwen38-qsa-fp8-e4m3-20260829/README.md).

## Kernel gates

The RTX 3090 gate used six local query heads, one local KV head, head dimension
256, and 2,051 selected tokens. It passed:

- all 256 E4M3 byte patterns, including signed zero, subnormals, infinities, and
  NaNs under the declared reference semantics;
- finite numerical comparison against BF16 at M=1, 4, 64, and 512;
- bitwise-equal CUDA Graph replay at M=1 and M=256;
- the 1.25x BF16 reader-time limit at M=1, 8, 32, 256, and 512.

| Shape | Cosine | NRMSE | FP8/BF16 reader time |
| --- | ---: | ---: | ---: |
| M=1 | 0.999277 | 0.038033 | 1.043x |
| M=8 | not separately scored | not separately scored | 1.043x |
| M=32 | not separately scored | not separately scored | 1.051x |
| M=64 | 0.999281 | 0.037910 | not timed in the final table |
| M=256 | graph replay passed | graph replay passed | 1.174x |
| M=512 | 0.999276 | 0.038064 | 1.218x |

These are reader microbenchmarks. They do not explain the model's decode gap.

## Serving gates

The initial `gpu_memory_utilization=0.95` profile reported 2.08 GiB available
for KV and allocated 314,261 tokens. It passed:

- deterministic text generation;
- automatic tool selection and post-tool continuation;
- multimodal inference;
- two simultaneous short generations;
- exact retrieval of `VIOLET ORBIT 9137` from a 261,544-token API prompt;
- zero serving-process swap;
- the fixed 230 W server60 power policy.

The initial PYNCCL production benchmark measured 43.77 decode tokens/s and
1,529.25 cache-busted prefill tokens/s. The preceding BF16 run measured 43.98
and 1,531.33. The differences were minus 0.47% and minus 0.14%.

After trace-guided hierarchical all-reduce promotion, the exact-final service
measured 50.34 decode and 1,538.14 prefill tokens/s. Concurrency-2 aggregate
throughput rose from 53.25 to 59.00 tokens/s. The model, QSA cache, PLE,
context, batch-token budget, and GPU policy stayed unchanged.

The later 0.98 production setting reported 2.79 GiB available for KV and
421,608 cache tokens. It preserved the same 262,144-token context and passed
image, tool, post-tool, concurrency-2, and exact 261,544-token NIAH checks. The
first acceptance warmed another 460 to 500 MiB per GPU. Three more
concurrency-2 rounds produced no further NVML growth, OOM, allocator retry,
restart, or swap. See
[`evidence/qwen38-gpu-util-098-20260830`](evidence/qwen38-gpu-util-098-20260830/README.md).

## llama-benchy interpretation

Will's llama-benchy files are preserved byte-for-byte in the evidence directory.
The cached runs are the usable decode-scaling evidence:

| Workload | Aggregate decode | Mean per request |
| --- | ---: | ---: |
| Cached c=1 | 48.21 tokens/s | 48.21 tokens/s |
| Cached c=2 | 65.68 tokens/s | 36.12 tokens/s |

Cached c=2 gives 1.36x aggregate speedup and 68.1% parallel efficiency.

The cold c=2 `tg_throughput` value of 11.39 tokens/s is not steady concurrent
decode. llama-benchy computes batch generation throughput over the interval from
the earliest first token to the latest last token. In this run one request was
decoding while the other was still prefilling. Per-request decode alternated
between about 40.7 and 5.9 tokens/s, while time to first response alternated
between about 21 and 40 seconds. Treat this as a mixed concurrent-prefill and
decode interference workload.

## Open performance work

The matched c=1/c=2 trace separated mixed concurrent prefill from steady decode
and identified BF16 ring all-reduce as the first measured critical segment.
Island-aware hierarchical all-reduce shortened that segment and raised
single-stream decode by 15.0%. The next trace-grounded target is the repeated
BF16 hyperconnection projection family documented in
[`KERNEL2-HYPERCONNECTION.md`](KERNEL2-HYPERCONNECTION.md).

Mixed concurrent-prefill starvation remains separate. The current exact-final
service costs 19.865 ms/token at 50.34 tokens/s. Alesha's 64.31-token/s
near-maximum result costs 15.550 ms/token, leaving a 4.315 ms/token external
comparison gap. Checkpoint differences, split projection structure, companion
FP8 weights, and host configuration still prevent attributing that gap to one
runtime mechanism.
