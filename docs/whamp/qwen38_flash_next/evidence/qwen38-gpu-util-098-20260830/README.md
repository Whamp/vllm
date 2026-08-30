# Qwen3.8 GPU utilization 0.98 acceptance

## Decision

Server60 now runs the existing Qwen3.8 FP8-QSA plus hierarchical-all-reduce profile with `gpu_memory_utilization=0.98`. The change kept the same image, container, Docker network, port 30002, model, 262,144-token context, multimodal settings, and rollback path.

The 0.95 Compose file remains in `production-095-before-098.yml`. `rollback-095-from-098.sh` restores it in place.

## Multimodal reservation check

The deployed vLLM source has no fixed image-memory option analogous to `max_num_seqs`. Startup reported an encoder budget of 16,384 tokens and profiled one maximum-size image.

`--limit-mm-per-prompt` controls item count. The current budget already limits profiling to one image, so setting the count to one would not reclaim memory. Its optional width and height fields affect dummy profiling. A safe resolution reduction would also need a real processor bound such as `--mm-processor-kwargs max_pixels`. This change preserved the existing image contract.

## Capacity and acceptance

At 0.98, startup reported:

- 2.79 GiB available KV memory;
- 421,608 aggregate KV tokens;
- 1.61x maximum concurrency at 262,144 tokens.

The acceptance run passed deterministic generation, automatic tool selection, post-tool continuation, image input, two simultaneous short generations, and exact retrieval of `VIOLET ORBIT 9137` from a 261,544-token API prompt. Concurrency-2 produced 44.25 aggregate tokens/s in this functional run.

The first full acceptance warmed another 460 to 500 MiB per GPU. Three additional concurrency-2 rounds left NVML usage byte-stable at 23,577 MiB on GPU 0 and 23,473 MiB on GPUs 1 through 3. No OOM, allocator-retry, restart, process-swap, or host-swap signal appeared. Final free VRAM was 550 MiB on GPU 0 and 654 MiB on the other GPUs. These figures are diagnostic headroom, not a fixed release threshold.

## Files

- `production.yml`: active 0.98 Compose contract.
- `production-095-before-098.yml`: retained 0.95 rollback.
- `acceptance-098.json`: capability, concurrency, and 261K NIAH results.
- `gpu-util-098-capacity.txt`: startup encoder and KV sizing lines.
- `gpu-util-098-*-before.csv` and `gpu-util-098-*-after.csv`: initial and warm-state NVML readings.
- `gpu-util-098-final-state.txt`: final service, swap, safety, and VRAM state.
- `rollback-095-from-098.sh.gz`: exact in-place rollback procedure, stored as deterministic gzip.
