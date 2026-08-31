# Direct BF16 SSD PLE production evidence

This bundle records the order-balanced server60 comparison and production
switch from Primitive NVFP4 PLE mappings to the original Intel BF16 PLE shard.
The model, runtime, QSA cache, collectives, Kernel2 path, context, batching,
sampling, and GPU safety controls stayed fixed.

## Identities

| Item | Value |
| --- | --- |
| vLLM branch commit | `9a2bdd3e39b3ad692f4b4d3c9a5f9d2bde91a3a4` |
| Runtime image | `sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef` |
| BF16 PLE file | Intel `model-00016-of-00017.safetensors` |
| BF16 PLE SHA-256 | `59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf` |
| BF16 PLE bytes | `102400512256` |
| NVFP4 rollback revision | `da8b39586016d8325ac619be28ad77d6296625ec` |
| API context | `262144` tokens |
| GPU safety | 230 W, driver-managed clocks within the host policy |

## Order-balanced A/B

The arm order was BF16 A, NVFP4 A, NVFP4 B, BF16 B. Each decode arm used
three warmups and five measured 256-token runs. Each prefill arm used one
warmup and three measured runs. The committed analyzer pools both arms of each
table format.

| Metric | NVFP4 | Direct BF16 | Change |
| --- | ---: | ---: | ---: |
| Concurrency-1 decode | 63.4460 tok/s | 67.8257 tok/s | +6.9030% |
| Concurrency-2 aggregate decode | 111.8125 tok/s | 113.4911 tok/s | +1.5013% |
| Concurrency-1 prefill | 1683.1305 tok/s | 1677.1220 tok/s | -0.3570% |
| Concurrency-2 aggregate prefill | 1688.7556 tok/s | 1691.4904 tok/s | +0.1619% |

The NVFP4 control showed a large cache-order effect at concurrency 1, rising
from 60.3463 tok/s in arm A to 66.5457 tok/s in arm B. Direct BF16 measured
67.8140 and 67.8374 tok/s in its two arms. Pooling the order-balanced arms
keeps that cache effect in the comparison.

Across the two arms, the PLE worker used 40.29 CPU seconds with BF16 versus
53.60 seconds with NVFP4. BF16 recorded 281362432 process-read bytes and
285216768 host-NVMe read bytes. NVFP4 recorded 12918558720 and 12924092416
bytes, respectively. BF16 therefore moved about 45 times fewer storage bytes
in this workload, despite its larger source file. BF16 incurred more major
faults, so page-fault count alone does not explain the outcome.

## Correctness and quality

The BF16 candidate passed:

- deterministic arithmetic and explicit high-effort reasoning;
- automatic tool selection and coherent post-tool continuation;
- repeated-prefix cache reuse;
- multimodal inference;
- two-stream generation;
- exact `VIOLET ORBIT 9137` retrieval from the 261492-token NIAH prompt;
- BenchLocal quick at 30/30, versus the accepted NVFP4 record at 28/30.

## PLE stage timing

A separate diagnostic-only restart measured 256 operations. It added an
explicit result-ready synchronization and did not contribute benchmark data.
The cumulative means were:

| Stage | Mean |
| --- | ---: |
| Request launch to worker handling | 12596 us |
| BF16 row gather | 2581 us |
| H2D plus semaphore submission | 498 us |
| H2D plus semaphore result ready | 578 us |
| Prior output-buffer reuse wait | 226 us |

These stages overlap. Their values must not be summed into a token latency.
They show that direct BF16 row gathering is not the largest remaining PLE
interval.

## Production decision

Direct BF16 passed the registered performance, capability, long-context, and
quality gates. It now runs on server60 under the existing production identity
and port with `restart: unless-stopped`. Final checks confirmed the expected
model identity, 262144-token API limit, deterministic output, zero process
swap, strict overcommit restored to 0, and the 230 W GPU safety service.

The Primitive NVFP4 profile remains the exact rollback. Startup still emits
the inherited expandable-segment mapping warnings seen on this tight-memory
profile. The service reached health and completed all registered workloads.

`SHA256SUMS` covers the readable summaries, production Compose, raw archive,
and `RAW-SHA256SUMS`. The gzip archive contains 113 original evidence files.
Extract it and run `sha256sum -c RAW-SHA256SUMS` from the extraction directory
to verify each one. Raw Docker inspect payloads were excluded because they
contain resolved environment values rather than reproducible source inputs.
