# Qwen3.8 native NVFP4 PLE gather

This experiment targets the PLE worker that the accepted server60 Compose profile bind-mounts over the vLLM image. It does not target the newer in-tree `qwen4_exp` sidecar path.

The completed server60 campaign and promotion decision are recorded in [RESULTS.md](RESULTS.md).
The separate scheduling A/B is recorded in [ASYNC-SCHEDULING-RESULTS.md](ASYNC-SCHEDULING-RESULTS.md).
The exact-production trace blocker is recorded in [NSYS-TRACE-BLOCKER.md](NSYS-TRACE-BLOCKER.md).
The direct BF16 SSD production path is recorded in [BF16-SSD-PLE.md](BF16-SSD-PLE.md).

The production worker source must have SHA-256:

```text
e85a2d599a422b0b2451f7ad74e408688f53e59e3a77ba17afb4fdffd0bcebad
```

That file is `worker_image_quant.py` from Primitive AI revision `da8b39586016d8325ac619be28ad77d6296625ec`.

A self-contained legacy-image overlay also pins `ple_layer.py` to SHA-256:

```text
1cb682b53f024b2060c5fe205fa0f6eca7c8df2cfbca3d21bea94d832b4db16a
```

Copying this 51,675-byte source into the overlay prevents production restarts
from depending on the removable NVFP4 Hugging Face cache. This option is only
needed by the accepted legacy `qwen3_8_flash_next` image. Current `qwen4_exp`
source owns the BF16 mmap path directly and does not need this source override.

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
  /path/to/output \
  --ple-layer /path/to/legacy/ple_layer.py
```

The builder verifies the production worker hash before it writes:

```text
worker_image_quant.py
nvfp4_native_gather.py
bf16_ple_mmap_gather.py
libvllm_ple_nvfp4_gather.so
ple_layer.py
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

## Direct BF16 SSD candidate

The same generated worker can map the original Intel BF16 PLE shard directly. Do not set `VLLM_PLE_QUANT_DIR` or `VLLM_PLE_DISK_OFFLOAD_DIR` in this mode.

Mount the generated BF16 helper beside the existing helper and bind the content-addressed Intel shard read-only:

```yaml
volumes:
  - type: bind
    source: /path/to/output/bf16_ple_mmap_gather.py
    target: /usr/local/lib/python3.12/dist-packages/vllm/v1/ple_offload/bf16_ple_mmap_gather.py
    read_only: true
  - type: bind
    source: /path/to/output/ple_layer.py
    target: /usr/local/lib/python3.12/dist-packages/vllm/models/qwen3_8_flash_next/nvidia/ple_layer.py
    read_only: true
  - type: bind
    source: /path/to/intel/model-00016-of-00017.safetensors
    target: /ple/59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf
    read_only: true
environment:
  VLLM_PLE_BF16_MMAP_FILE: /ple/59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf
  VLLM_PLE_BF16_MMAP_SHA256: 59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf
  VLLM_PLE_BF16_MMAP_LIBRARY: /opt/vllm/libvllm_ple_nvfp4_gather.so
```

This path passed the server60 full-model A/B and is now the production default. See [BF16-SSD-PLE.md](BF16-SSD-PLE.md) for the artifact contract, measurements, and cold NVFP4 rollback.

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

Run a cold direct-BF16 SSD probe with:

```bash
python benchmarks/qwen38_ple_runtime/benchmark_bf16_ssd_gather.py \
  --checkpoint /path/to/model-00016-of-00017.safetensors \
  --expected-sha256 59d1ce2df8a9e4441e0d6328b5fd620f427734274bf559ba4f15a8f98bf35abf \
  --library /path/to/output/libvllm_ple_nvfp4_gather.so \
  --drop-file-cache
```

The cold option evicts only the specified checkpoint's clean file-cache pages. Do not run it while another process depends on that same table's cache state.

These microbenchmarks exclude worker scheduling, GPU delivery, and model execution. They validate the local gather mechanisms, not end-to-end throughput.

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
