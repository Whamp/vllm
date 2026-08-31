# Direct BF16 SSD PLE for Qwen3.8

## Status

This is a default-off CPU-validated candidate. It has not loaded the full model or changed server60 production.

The current service already reads its 28.8 GB Primitive NVFP4 PLE sidecar through file-backed mappings. The question is therefore whether direct reads from the original 95.37 GiB BF16 table can preserve exact PLE values without slowing serving enough to matter.

The first server60 result supports a full service A/B. Direct BF16 reads with `MADV_RANDOM` moved fewer physical bytes and completed faster than the NVFP4 sidecar under the same cold random-row probe. That does not establish end-to-end performance.

## Artifact identity

The downloaded file is Intel `model-00016-of-00017.safetensors`:

```text
LFS SHA-256  59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf
File bytes   102,400,512,256
Payload      102,400,491,520 bytes / 95.3679 GiB
Header       20,728 bytes
Tensors      128
```

Each tensor is BF16 `[2,500,012, 160]`. The file stores shard 0 through shard 127 contiguously in numeric order. Together they form `[320,001,536, 160]`.

The object is byte-identical to the LFS identity recorded at the pinned Intel model revision `861536dda5bcb208376fc4cd879b2bf76bece9fe`. The successful download updated the local `main` snapshot to `4c67bf686b7f7fd386bae6b07ab59e8ff1d5b897`, but both revisions name the same PLE object. A one-time full-file SHA-256 pass using idle-class direct I/O matched `59d1ce2d...f35abf` exactly.

## Measured storage behavior

### Current production sidecar

The production PLE worker maps 128 Primitive NVFP4 files totaling 28,800,188,416 bytes. A live `/proc/<pid>/smaps` inspection found 1,331,880 KiB resident. Its mappings had no random-access advice.

During one warmup and three 256-token generations, the worker recorded:

| Run | Worker CPU | Major faults | Process read bytes | Read bytes/token |
| ---: | ---: | ---: | ---: | ---: |
| Warmup | 1.47 s | 6,400 | 1,626,439,680 | 6.06 MiB |
| 1 | 1.07 s | 3,635 | 900,931,584 | 3.36 MiB |
| 2 | 0.96 s | 2,975 | 726,376,448 | 2.71 MiB |
| 3 | 0.94 s | 2,547 | 617,922,560 | 2.30 MiB |

The 102.4 GB Intel download was active during these runs. Treat their wall time as contaminated by storage contention. The process counters still prove that production performs substantial SSD reads and readahead.

### Cold exact-shape probes

The bounded probes used 64 token steps. Each step selected one random row from each of the 16 disjoint PLE head regions. Each BF16 run evicted only the BF16 file before starting. The NVFP4 runs used distinct row seeds without evicting production's shared file cache, so their fault rate reflects the cache state at that moment rather than a guaranteed cold start.

| Table | Advice | Mean latency/token | Read bytes/token | Major faults/token |
| --- | --- | ---: | ---: | ---: |
| BF16 | default | 2.83 ms | 2,048 KiB | 15.98 |
| BF16 | `MADV_RANDOM` | 1.35 ms | 67.75 KiB | 16.94 |
| NVFP4 | default | 5.06 ms | 4,597-4,929 KiB | 19.7-21.0 |
| NVFP4 | `MADV_RANDOM` | 1.71 ms | 79.8-84.1 KiB | 20.0-21.0 |

`MADV_RANDOM` cut BF16 read amplification about 30 times and cold latency about 2.1 times. It cut NVFP4 read amplification about 58 times and cold latency about 2.9 times. The committed Python benchmark repeated the BF16 cold arm at 1.468 ms/token and 69,760 bytes/token.

BF16 can read fewer pages despite its larger file. One BF16 row is a contiguous 320-byte span. One NVFP4 row reads an 80-byte code span and a separate 10-byte scale span, so cold access usually faults two pages per selected row. The larger BF16 table can still have a lower cache-hit rate under sustained workloads. Only the service A/B can settle the net result.

The direct BF16 implementation also passed an independent oracle over 260 real rows spanning 111 source shards. Its BF16 bytes matched bounded `pread` reference reads exactly. The checksum-bound results are under [`evidence/bf16-ssd-ple-cpu-20260831/`](evidence/bf16-ssd-ple-cpu-20260831/README.md).

## Candidate mechanism

The candidate keeps the existing asynchronous PLE worker, pinned output buffer, CUDA IPC fanout, semaphore contract, and CUDA Graph behavior. It changes only the table behind the worker's existing `_ple_quant.gather_into()` call.

When all three variables are set, the worker:

1. resolves the checkpoint path to its content-addressed blob;
2. requires the resolved Hugging Face blob filename to match the configured SHA-256;
3. parses at most 16 MiB of safetensors header data;
4. validates every tensor name, dtype, shape, byte range, shard index, payload order, and final file size;
5. maps the source file read-only and applies `MADV_RANDOM`;
6. replaces the unused 95 GiB model parameter with an empty BF16 stub;
7. copies selected BF16 rows into the existing pinned worker output through one native call.

```text
VLLM_PLE_BF16_MMAP_FILE=/ple/59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf
VLLM_PLE_BF16_MMAP_SHA256=59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf
VLLM_PLE_BF16_MMAP_LIBRARY=/opt/vllm/libvllm_ple_nvfp4_gather.so
```

The BF16 mode is mutually exclusive with `VLLM_PLE_QUANT_DIR` and `VLLM_PLE_DISK_OFFLOAD_DIR`. With BF16 unset, the generated worker retains the legacy rule that the quantized sidecar takes precedence when both older variables are present. Startup does not reread 95 GiB to recompute SHA-256. Delivery therefore requires the Hugging Face download receipt or a separately recorded full-file hash in addition to the content-addressed filename check.

This design does not use vLLM PR [#54129](https://github.com/vllm-project/vllm/pull/54129)'s in-forward mmap path. That path copies IDs from GPU to CPU and requires a piecewise graph split. The existing worker already performs request staging and GPU fanout off the model thread, so retaining it is the smaller change for server60.

It also avoids vLLM PR [#54070](https://github.com/vllm-project/vllm/pull/54070)'s second 95 GiB `.bin` file. The Intel shard already stores the table in the exact contiguous order the worker needs.

## Gated hypothesis

```text
OUTCOME: preserve exact BF16 PLE values with no more than 10% c=1 decode loss,
         no more than 15% c=2 aggregate decode loss, and no more than 5% prefill loss
CRITICAL SEGMENT: request submission through all-rank PLE result readiness
EVIDENCE: cold server60 BF16 gathers used 69 KiB and 1.35 ms/token;
          the current NVFP4 format used 82-86 KiB and 1.61-1.76 ms/token under MADV_RANDOM
MOVE: map the original BF16 safetensors payload directly in the existing worker
GATE: exact artifact layout, independent row oracle, and CPU contract tests pass
LOSE CONDITION: the 3.3-times larger table lowers cache hit rate enough to dominate,
                or BF16 file faults interfere with concurrent requests
SHIFTED COST: 95.37 GiB disk file, a larger page-cache working set, and more cold-tail risk
FALSIFIER: matched serving loses more than the thresholds or PLE readiness becomes slower
CONTRACTS: model, QSA, PLE hashing, row order, CUDA Graphs, context, vision, tools,
           sampling, safety controls, zero swap, and rollback remain unchanged
```

## Required service A/B

Do not promote from the CPU evidence.

1. Freeze the live image, Compose, model, scale file, context, cache capacity, GPU safety state, and worker hashes.
2. Build a thin child image containing only the four checksum-bound overlay files.
3. Arm the exact production rollback watchdog before stopping production.
4. Run a non-restarting BF16 candidate at the same 262,144-token API limit.
5. Verify startup dispatch and the exact PLE source SHA-256.
6. Run deterministic text, tool plus post-tool, reasoning, multimodal, two-stream, prefix-cache, and 261K NIAH checks.
7. Run reverse-ordered matched control and candidate pairs. Use three decode warmups plus five measured 256-token runs, and one prefill warmup plus three measured runs, at c=1 and c=2.
8. Record PLE-worker CPU, major faults, process read bytes, NVMe reads, queue time, lookup time, H2D submission, semaphore completion, VRAM, swap, and allocator failures.
9. Run BenchLocal quick if performance stays inside the thresholds.
10. Restore and verify the exact production service whether the candidate passes or fails.

A passing result establishes one server60 option. It does not prove that every host should prefer BF16 over a quantized PLE table.

## Source basis

- vLLM PR [#54070](https://github.com/vllm-project/vllm/pull/54070) implements worker-side BF16 disk mapping through a second file. Its RTX PRO 6000 measurements reported about 8% c=1 and 17% c=32 loss against BF16 RAM inside a 48 GB memory cap.
- vLLM PR [#54129](https://github.com/vllm-project/vllm/pull/54129) maps original checkpoint shards in the model forward and now supports Intel BF16 PLE tensors.
- Primitive AI's [quantized PLE repository](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) supplies the accepted NVFP4 sidecar and the production worker source.
