# Direct BF16 SSD PLE

## Decision

Server60 now reads Qwen3.8's PLE directly from Intel's original BF16
safetensors shard. The previous Primitive NVFP4 table is now a cold rollback
that must be re-downloaded before use.

This result was counterintuitive. The BF16 table is 95.37 GiB, more than three
times the NVFP4 sidecar's size. In the measured workload, BF16 still read about
45 times fewer bytes from storage, used 25% less PLE-worker CPU time, and
improved concurrency-1 decode by 6.90%. Its rows are contiguous 320-byte spans.
NVFP4 reads separate code and scale spans from 128 mappings, which caused more
read amplification under the tested access pattern.

## Fixed runtime contract

The comparison kept these values fixed:

- Intel AutoRound W4A16 model and vision path;
- calibrated E4M3 QSA cache;
- hierarchical all-reduce islands `0,1;2,3`;
- native SM86 Kernel2 hyperconnection path;
- `262144` API context, `max_num_seqs=4`, and `max_num_batched_tokens=1024`;
- FULL_DECODE_ONLY CUDA graphs;
- 230 W GPU safety controls.

Only the PLE table and gather implementation changed.

The BF16 path binds Intel `model-00016-of-00017.safetensors` by content hash:

```text
SHA-256  59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf
Bytes     102400512256
Tensors   128
Rows      320001536
Width     160 BF16 values
```

The worker validates the bounded safetensors header, tensor names, row geometry,
file size, and content-addressed blob name. It maps the file read-only with
`MADV_RANDOM`, gathers requested rows into the existing pinned BF16 output
buffer, and keeps the established CUDA IPC fanout and semaphore protocol.

## Full-model A/B

The order was BF16 A, NVFP4 A, NVFP4 B, BF16 B. Both formats started fresh in
each arm. Decode used three warmups and five measured 256-token runs. Prefill
used one warmup and three measured runs.

| Metric | NVFP4 | Direct BF16 | Change |
| --- | ---: | ---: | ---: |
| Concurrency-1 decode | 63.4460 tok/s | 67.8257 tok/s | +6.9030% |
| Concurrency-2 aggregate decode | 111.8125 tok/s | 113.4911 tok/s | +1.5013% |
| Concurrency-1 prefill | 1683.1305 tok/s | 1677.1220 tok/s | -0.3570% |
| Concurrency-2 aggregate prefill | 1688.7556 tok/s | 1691.4904 tok/s | +0.1619% |

Cache order mattered for NVFP4. Its two concurrency-1 arms measured 60.3463 and
66.5457 tok/s. BF16 measured 67.8140 and 67.8374 tok/s. The pooled comparison
retains both control states.

| Two-arm telemetry | NVFP4 | Direct BF16 |
| --- | ---: | ---: |
| PLE worker CPU | 53.60 s | 40.29 s |
| PLE process-read bytes | 12918558720 | 281362432 |
| Host NVMe-read bytes | 12924092416 | 285216768 |
| Host NVMe-read time | 9717 ms | 5985 ms |
| Major faults | 54724 | 68695 |

The higher BF16 major-fault count and lower byte count show why page-fault count
alone is misleading here. Each format faults and reads different spans.

## Capability and quality

The BF16 candidate passed deterministic arithmetic, explicit high-effort
reasoning, automatic tool selection, post-tool continuation, repeated-prefix
cache reuse, multimodal inference, and two-stream generation. It retrieved the
exact needle from a 261492-token prompt.

BenchLocal quick scored 30/30. The accepted NVFP4 record scored 28/30.

A separate diagnostic-only restart measured 2.58 ms mean BF16 gather time over
256 operations. Request launch to worker handling averaged 12.60 ms, H2D plus
semaphore submission 0.50 ms, and result readiness 0.58 ms. These stages
overlap, so they are attribution data rather than additive token latency.

## Production and rollback

Production uses the existing service identity and port 30002 with
`restart: unless-stopped`. Final checks confirmed the expected model,
262144-token limit, deterministic output, zero process swap,
`vm.overcommit_memory=0`, and active GPU safety controls.

Production copies the exact legacy `ple_layer.py` into its checksum-bound
runtime directory. That source computes n-gram row IDs, invokes the table gather,
and applies the PLE projection, gate, normalization, and short-convolution
state update. The current legacy image needs it, but production no longer reads
it from the NVFP4 Hugging Face cache.

After a self-contained restart passed, the 129 NVFP4 table files were deleted.
This reclaimed 28,800,757,760 filesystem bytes. The old restore script now
fails before touching production until the exact revision and byte count have
been re-downloaded. Startup on this tight-memory profile still emits the
inherited expandable-segment mapping warnings seen with the control runtime,
but the BF16 service reached health and completed every registered workload.

The complete 113-file archive, analyzer output, raw measurements, functional
results, BenchLocal report, diagnostic timing, production Compose, and final
health record are in
[`evidence/qwen38-ple-bf16-ssd-production-20260831/`](evidence/qwen38-ple-bf16-ssd-production-20260831/README.md).
