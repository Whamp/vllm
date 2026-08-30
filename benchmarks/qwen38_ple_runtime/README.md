# Qwen3.8 native NVFP4 PLE gather

This experiment targets the PLE worker that the accepted server60 Compose profile bind-mounts over the vLLM image. It does not target the newer in-tree `qwen4_exp` sidecar path.

The completed server60 campaign and promotion decision are recorded in [RESULTS.md](RESULTS.md).

The production worker source must have SHA-256:

```text
e85a2d599a422b0b2451f7ad74e408688f53e59e3a77ba17afb4fdffd0bcebad
```

That file is `worker_image_quant.py` from Primitive AI revision `da8b39586016d8325ac619be28ad77d6296625ec`.

## Behavior

The candidate replaces the worker's per-shard Torch gather and dequantization loop with one raw CPU call. The kernel:

- reads the existing mmap-backed E2M1 code and E4M3 scale tensors;
- preserves low-nibble-first decoding and caller row order;
- writes BF16 rows into the existing pinned output buffer;
- keeps the original Torch path when the output dtype is not BF16;
- leaves the original path active when `VLLM_PLE_NVFP4_GATHER_LIBRARY` is unset.

An invalid library path fails worker startup. It does not silently disable an explicitly requested candidate.

## Build the overlay

Run this command from the repository root:

```bash
.venv/bin/python \
  benchmarks/qwen38_ple_runtime/build_native_gather_overlay.py \
  /path/to/worker_image_quant.py \
  /path/to/output
```

The builder verifies the production worker hash before it writes:

```text
worker_image_quant.py
nvfp4_native_gather.py
libvllm_ple_nvfp4_gather.so
SHA256SUMS
```

## Wire a candidate Compose profile

Keep the accepted image and all model arguments unchanged. Replace the existing worker mount with the generated worker, then add these mounts:

```yaml
volumes:
  - type: bind
    source: /path/to/output/worker_image_quant.py
    target: /usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/worker.py
    read_only: true
  - type: bind
    source: /path/to/output/nvfp4_native_gather.py
    target: /usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/nvfp4_native_gather.py
    read_only: true
  - type: bind
    source: /path/to/output/libvllm_ple_nvfp4_gather.so
    target: /opt/vllm/libvllm_ple_nvfp4_gather.so
    read_only: true
environment:
  VLLM_PLE_NVFP4_GATHER_LIBRARY: /opt/vllm/libvllm_ple_nvfp4_gather.so
```

Do not change the accepted production profile in place. Run the candidate sequentially, collect its evidence, stop it, and restore the exact accepted profile.

## CPU evidence

The committed tests compile the raw ABI and compare it against the production Torch arithmetic. They cover:

- 1, 2, 7, and 128 shards;
- widths 16, 32, and the production width 160;
- zero, one, 16, and 32 gathered rows;
- shard boundaries, duplicate IDs, caller order, invalid IDs, and fallback routing.

Local hot-data runs at 128 shards and width 160 measured:

```text
Rows  Torch mean  Native mean  Speedup
16    0.445449 ms 0.006178 ms  72.10x
32    0.869581 ms 0.009115 ms  95.40x
```

Reproduce these rows with `benchmark_native_gather.py` and the generated shared library.

This microbenchmark excludes mmap page faults, worker scheduling, GPU delivery, and model execution. It proves that the native call removes the measured Torch gather overhead; it does not predict end-to-end throughput.

## Required server60 gates

Do not promote this candidate from CPU evidence. Use the production state accepted immediately before this experiment as the control. Freeze its image digest, Compose SHA-256, source revision, cache capacity, process residency, and service matrix before starting the candidate.

Run these gates in order:

1. Verify the three generated artifact hashes inside the candidate runtime.
2. Verify the PLE worker starts with the native route enabled.
3. Compare at least 100 actual sidecar rows, including shard boundaries and duplicates, against the Python fallback. Require bit-exact BF16 output.
4. Benchmark actual mmap-backed gathers at 16 and 32 rows. Require at least a 2x mean speedup at each shape.
5. Require aggregate KV-cache capacity to equal or exceed the control. Reject any unexplained GPU-process residency increase above 64 MiB on any rank.
6. Collect matched baseline and candidate traces. Require request-to-all-ranks PLE readiness to improve by at least 0.25 ms at either C1 or C2, with no regression above 0.10 ms at the other concurrency.
7. Run five paired C1/C2 rounds with alternating control/candidate order. Require at least a 1% decode-throughput gain at one concurrency and no more than a 1% loss at the other. Treat prefill and TTFT as diagnostic unless either regresses by more than 5%.
8. Re-run context, deterministic output, tools, reasoning, multimodal, concurrency, prefix-cache, CUDA-Graph, and memory-capacity acceptance.
9. Restore the exact accepted service and verify its image, profile, health, restart count, restart policy, zero swap, and absence of rollback timers.

Reject the candidate if it misses a gate. Do not combine it with fan-out or async scheduling until this isolated verdict is recorded.
