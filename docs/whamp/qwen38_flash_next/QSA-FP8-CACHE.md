# Qwen3.8 QSA FP8 cache

## Decision

Promote the calibrated direct-E4M3 QSA cache as server60's Qwen3.8 production default.
Keep the BF16 profile as the rollback path.

The promoted profile stores the twelve QSA layers' main K/V cache as E4M3 bytes,
decodes those bytes in the sparse QSA reader, and applies an exact calibrated
scale for each layer's K and V tensors. Raw and compressed indexer caches remain
unchanged. The profile reached the native 262,144-token model limit with 314,261
aggregate KV tokens and 1.20x maximum concurrency.

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
| Production FP8 image | `sha256:61971a78222e89335d84a7f4d72b0e8842619a4a29564582a58ff328af48abb9` |
| Scale-file SHA-256 | `554d68aa917bdcac3ec7e5c14a8ca1421182d0b899da30b6c54501b72aefdcf3` |
| Cache dtype | `fp8_e4m3` |
| Context | 262,144 tokens |
| Maximum sequences | 2 |
| Batch-token budget | 1,024 |
| GPU-memory utilization | 0.95 |
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

At `gpu_memory_utilization=0.95`, the service reported 2.08 GiB available for
KV and allocated 314,261 tokens. It passed:

- deterministic text generation;
- automatic tool selection and post-tool continuation;
- multimodal inference;
- two simultaneous short generations;
- exact retrieval of `VIOLET ORBIT 9137` from a 261,544-token API prompt;
- zero serving-process swap;
- the fixed 230 W server60 power policy.

A matched internal benchmark measured 43.77 decode tokens/s and 1,529.25
cache-busted prefill tokens/s. The preceding BF16 run measured 43.98 and
1,531.33. The differences were minus 0.47% and minus 0.14%.

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

Two targets remain separate:

1. Mixed concurrent-prefill starvation or serialization in the cold c=2 run.
2. Cached steady decode scaling and the single-stream gap to Alesha's reported
   63 to 68 tokens/s.

The local cached c=1 result costs 20.744 ms/token. Alesha's 64.31-token/s
near-maximum result costs 15.550 ms/token. The unexplained budget is 5.194
ms/token. `max_num_seqs`, GPU-memory utilization, and benchmark prompt method do
not explain that matched single-stream interval. Before changing code, capture
matched c=1 and c=2 timelines and attribute scheduler phase admission, 1,024-token
chunking, expert-parallel all-to-all and imbalance, PLE, QSA, GDN,
hyperconnection, shared-expert, and routed-MoE work.

The next implementation must name one measured critical segment and shorten it.
Checkpoint changes and Alesha's thin-v2 FP8 companion projections remain a
separate one-variable comparison rather than an assumed cause.
