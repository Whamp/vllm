# Qwen3.8 async scheduling result

## Decision

Keep the promoted service unchanged. It already uses async scheduling through
vLLM's automatic default. Explicitly disabling async scheduling lost 24.5% to
36.8% pooled decode throughput at C1, C2, and C4.

This was not a clean all-metric win. Async scheduling increased mean TTFT by
111 ms at C2 and 176 ms at C4, which failed the preregistered 5% TTFT limit.
It improved C1 TTFT by 48 ms. The current service remains the right profile for
the throughput campaign, but a latency-first deployment may make a different
tradeoff.

No code or production configuration changed. Concurrent PLE fan-out remained
disabled throughout the experiment.

## What production was already doing

The promoted image is:

```text
sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef
```

Its `async_scheduling` argument defaults to `None`. For this generation model,
no-speculation configuration and multiprocessing executor,
`VllmConfig.__post_init__` resolves that value to `True`.
`MultiprocExecutor.supports_async_scheduling()` also returns `True`.

The production command passes neither scheduling flag. The experiment therefore
compared the existing behavior against an explicit disable:

- disabled control: `--no-async-scheduling`
- enabled candidate: `--async-scheduling`

The enabled and disabled worker snapshots had 234 and 230
`VLLM::Worker` threads. The restored production service had 234, matching the
enabled variant's one extra async output-copy thread per TP worker.

## Fixed configuration

Both variants used the same immutable image and preserved:

- TP4 and expert parallelism;
- `max_num_seqs=4` and `max_num_batched_tokens=1024`;
- `gpu_memory_utilization=0.98`;
- FP8 E4M3 KV cache;
- Mamba align mode and prefix caching;
- CPU PLE offload with native NVFP4 lookup;
- hierarchical all-reduce `0,1;2,3`;
- FULL_DECODE_ONLY CUDA Graphs;
- tool and reasoning parsers;
- the multimodal model path.

The resolved Compose profiles differed only in project, container, port,
experiment label, and scheduling flag. Their SHA-256 values were:

```text
e5b7c45a852c1667c7eeef4d8e620f29b036db03aadf1542b046acd68f60759e  disabled.yml
eff1657cfa3fd016cf24ef9851c64868524df0d61e5443aab271cd31b4c0f2ed  enabled.yml
```

## Measurement contract

The campaign ran two fresh-start pairs in reverse order:

1. disabled, then enabled;
2. enabled, then disabled.

Each service ran three decode warmups and five measured 256-token rounds at C1,
C2, and C4. Each concurrency also ran one prefill warmup and three measured
cache-busted prefill rounds. The experiment required at least a 1% C2 or C4
decode gain, no C1 decode loss above 1%, and no prefill or TTFT regression above
5%.

## Decode throughput

Values are aggregate generated tokens per second.

| Pair | Concurrency | Disabled | Enabled | Delta |
| --- | ---: | ---: | ---: | ---: |
| 1 | C1 | 44.589 | 60.897 | +36.57% |
| 1 | C2 | 80.081 | 103.630 | +29.41% |
| 1 | C4 | 142.063 | 178.822 | +25.88% |
| 2 | C1 | 44.153 | 60.475 | +36.97% |
| 2 | C2 | 81.449 | 102.792 | +26.20% |
| 2 | C4 | 145.990 | 179.895 | +23.22% |

The pooled result used all ten measured decode rounds per variant and
concurrency:

| Concurrency | Disabled | Enabled | Delta |
| --- | ---: | ---: | ---: |
| C1 | 44.371 | 60.686 | +36.77% |
| C2 | 80.765 | 103.211 | +27.79% |
| C4 | 144.026 | 179.359 | +24.53% |

Both pairs favored async scheduling at every concurrency.

## Prefill and TTFT

Pooled prefill throughput improved by 1.79% at C1, 1.25% at C2, and 1.29% at
C4.

| Concurrency | Disabled TTFT | Enabled TTFT | Absolute delta | Relative delta |
| --- | ---: | ---: | ---: | ---: |
| C1 | 470 ms | 422 ms | -48 ms | -10.27% |
| C2 | 595 ms | 706 ms | +111 ms | +18.67% |
| C4 | 669 ms | 845 ms | +176 ms | +26.36% |

The C2 and C4 TTFT rows failed the preregistered 5% limit. The performance gate
therefore failed as a whole even though decode and prefill passed.

## Capacity and runtime safety

Every fresh launch reported 425,497 aggregate KV-cache tokens. Enabled and
disabled runs had identical NVML residency on every GPU:

```text
VLLM worker: 22,796 MiB per GPU
PLE helper:      276 MiB per GPU
```

Each launch used the expected native-PLE worker, helper, and shared-library
hashes. All services remained healthy with zero restarts. No CUDA, NCCL, PLE,
out-of-memory, assertion, or deadlock failure appeared. Each launch captured
all four decode graphs.

The first attempted launch was rejected by the experiment runner because its log
pattern matched two known Transformers docstring warnings under the
`PleOffloadWorker` prefix. The fail-closed restore completed. The corrected
pattern rejected a synthetic PLE failure but ignored those warnings before the
four valid matrices began.

## Behavior acceptance

After the matrices, the exact pinned production service was restored. A fresh
acceptance run passed:

- deterministic output: `PARIS`;
- one tool call and successful tool-result continuation;
- multimodal image input;
- two concurrent decode streams;
- 261,544-token NIAH retrieval: `VIOLET ORBIT 9137`;
- healthy service with zero restarts.

A separate prefix-cache probe repeated an identical 22,971-token request. The
second request reused 21,600 tokens, a 94.03% hit ratio, and completed in 1.08 s
versus 14.52 s for the first request.

These checks validate the retained production behavior. They do not establish
sampling-distribution equivalence between enabled and disabled scheduling, and
no production behavior changed as a result of this experiment.

## Evidence

The checksum-bound campaign files are under:

```text
/home/will/build/qwen38-async-scheduling
```

Important files include:

- `PRE-REGISTRATION.md`
- `PROFILES.SHA256SUMS`
- `RESOLVED.SHA256SUMS`
- `INPUTS.SHA256SUMS`
- `evidence/pair1-disabled/benchmark.json`
- `evidence/pair1-enabled/benchmark.json`
- `evidence/pair2-enabled/benchmark.json`
- `evidence/pair2-disabled/benchmark.json`
- `evidence/summary.json`
- `evidence/production-acceptance.json`
- `evidence/production-prefix-cache.json`
- `runner-failure-1/`

This experiment measured service behavior. It did not collect a fresh Nsight
Systems trace, so it does not claim why async scheduling produced these deltas.
The exact-production trace remains the next campaign step.
