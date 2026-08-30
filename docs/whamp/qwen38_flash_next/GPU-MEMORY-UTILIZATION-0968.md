# Qwen3.8 GPU memory utilization ceiling

## Decision

Promote `gpu_memory_utilization=0.968` for the Intel AutoRound Qwen3.8 service on
server60.

The prior 0.95 profile left about 1.8 GiB physically free after validation. A
strict one-variable search tested 0.97 first. That arm fit 206,400 tokens and
passed correctness and performance, but failed the fixed 1 GiB post-NIAH
headroom floor by 4–8 MiB. It was rolled back.

The 0.968 arm fit 202,400 tokens and retained at least 1,098 MiB per GPU after
190K NIAH and the final quality run. It passed every gate and is now the live
production profile.

## Fixed contract

Both arms kept these fields unchanged:

- image
  `sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b`;
- Intel AutoRound and Primitive PLE revisions;
- vision enabled;
- tensor parallel size 4 with expert parallelism;
- BF16 QSA cache;
- 262,144 materialized RoPE rows;
- `max_num_batched_tokens=1024`;
- `max_num_seqs=2`;
- automatic context sizing;
- full-decode-only CUDA graphs;
- 230 W and 210–1650 MHz GPU safety policy.

Only `gpu_memory_utilization` changed.

## Preregistered gates

The accepted profile had to meet all of these:

- at least 30,000 fitted context tokens above the 167,600-token control;
- at least 95% decode retention;
- at least 90% cache-busted prefill retention;
- at least 1,024 MiB physically free on every GPU after near-ceiling NIAH;
- zero serving-process swap and zero restarts;
- deterministic output, tools and post-tool continuation, multimodal input,
  concurrency two, and long-context retrieval;
- BenchLocal quick quality equal to the accepted 26/30 total;
- complete rollback to the 0.95 service.

## Rejected 0.97 boundary

The 0.97 arm reached 206,400 tokens, a gain of 38,800. Its matched measurements
were strong:

| Metric | Result |
| --- | ---: |
| Decode | 44.3338 tok/s |
| Cache-busted prefill | 1,536.0469 tok/s |
| Concurrency-2 aggregate | 53.4096 tok/s |
| Exact NIAH | 195,044 API prompt tokens |
| Minimum post-NIAH free VRAM | 1,016 MiB |

GPU 0 had 1,016 MiB free and GPUs 1–3 had 1,020 MiB. The hard floor was 1,024
MiB, so the arm was rejected and the 0.95 service restored automatically. The
floor was not relaxed after seeing the result.

## Accepted 0.968 result

The 0.968 candidate reached 202,400 tokens, a gain of 34,800 over the 0.95
control and 54,000 over the original 148,400-token QSA profile.

### Candidate A/B

| Metric | 0.95 control | 0.968 candidate | Retention or gain |
| --- | ---: | ---: | ---: |
| Fitted context | 167,600 | 202,400 | +34,800, +20.76% |
| Decode | 43.0912 tok/s | 43.9783 tok/s | 102.06% |
| Cache-busted prefill | 1,540.8430 tok/s | 1,531.3296 tok/s | 99.38% |
| Concurrency-2 aggregate | 55.1491 tok/s | 55.9203 tok/s | 101.40% |
| Exact NIAH | 160,035 | 190,047 API prompt tokens | +30,012 |
| Minimum post-NIAH free VRAM | 1,836 MiB control final | 1,100 MiB | pass |

### Exact final production image

The recreated production profile passed again:

| Metric | Result | Retention vs 0.95 |
| --- | ---: | ---: |
| Decode | 43.5405 tok/s | 101.04% |
| Cache-busted prefill | 1,542.7705 tok/s | 100.13% |
| Concurrency-2 aggregate | 53.2523 tok/s | 96.56% |
| Exact NIAH | 190,047 API prompt tokens | pass |

Post-NIAH physical headroom was 1,216 MiB on GPU 0 and 1,100 MiB on GPUs 1–3.
After the final BenchLocal run, the service had 1,214 MiB on GPU 0 and 1,098 MiB
on GPUs 1–3.

## Correctness and quality

The exact final profile passed:

- deterministic output;
- automatic tool choice and post-tool continuation;
- synthetic image inference;
- two concurrent streamed requests;
- exact `VIOLET ORBIT 9137` retrieval at 190,047 API prompt tokens;
- BenchLocal quick 26/30, with ToolCall-15 12/15 and InstructFollow-15 14/15;
- zero running or waiting requests after validation;
- zero swap for all 12 serving processes;
- zero restarts;
- active GPU safety service;
- inactive rollback timer.

## Stable ceiling

The paired 0.97 and 0.968 results establish the current parameter ceiling under
the 1 GiB release margin. Raising utilization by another 0.002 recovered about
4,000 context tokens but consumed 78–82 MiB too much physical margin after a
195K request. The accepted 0.968 profile retains 74–76 MiB above the floor after
a 190K request and roughly the same margin after the quality pack.

Further context gains now require reducing persistent or cache bytes rather than
raising the executor budget again.

The checksum-bound plans, Compose files, rollback scripts, benchmarks, NIAH,
quality outputs, GPU states, metrics, and final-state record are under
[evidence/qwen38-util-0968-20260829/](evidence/qwen38-util-0968-20260829/).
