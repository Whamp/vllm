# Qwen3.8 concurrent PLE output fan-out

## Status

CPU implementation complete. GPU validation has not started.

This candidate stacks only concurrent TP output delivery on the promoted native lookup baseline. The control remains image:

```text
sha256:5f3da087ea29d8122e0ac83dc6dc7b60b4dda59d3f532b9569b984c2d5b013ef
```

The builder accepts only the promoted worker with SHA-256:

```text
8d6d71de8b56f32e27850ba24dcd194482808e6b7f0dc27f9312d39eb3c0fa32
```

The current generated fan-out worker has SHA-256:

```text
ba30a269be70eaea376d0535f0c3bb950e2979a9928ab9161b7513c5c50bc04c
```

## Behavior

Set this environment variable to enable the candidate:

```text
VLLM_PLE_CONCURRENT_FANOUT=1
```

Unset or `0` keeps serial target operations. Any other value fails startup. TP1 always uses the direct serial path.

For TP greater than one, the worker owns one persistent thread per TP target. It submits all target operations before waiting for any future. Each thread enters its target GPU's device context before it:

1. waits for the target's previous copy and RESET semaphore;
2. or submits the new pinned-host result copy and DONE semaphore.

The PLE worker still computes each layer once. It keeps the same per-layer pinned buffers, target streams, semaphore order, and layer-first request order. Shutdown joins the executor before the process releases its IPC resources.

## Build

First generate the promoted native worker:

```bash
.venv/bin/python \
  benchmarks/qwen38_ple_runtime/build_native_gather_overlay.py \
  /path/to/worker_image_quant.py \
  /tmp/qwen38-native-worker
```

Then add the fan-out patch:

```bash
.venv/bin/python \
  benchmarks/qwen38_ple_runtime/build_output_fanout_overlay.py \
  /tmp/qwen38-native-worker/worker_image_quant.py \
  /tmp/qwen38-fanout-worker
```

The second builder verifies the promoted worker hash and emits its own `SHA256SUMS` file.

## CPU evidence

The focused tests prove:

- all TP operations are submitted before the first wait;
- each operation enters the matching target device context;
- TP1 and disabled mode remain serial;
- only `0` and `1` are accepted;
- both readiness and delivery use the helper;
- shutdown joins and clears the executor;
- another worker revision fails the builder hash gate.

The combined CPU PLE, meta-construction, worker, and model regression set passes 76 tests. CPU tests cannot prove that the production CUDA streams, imported IPC tensors, and semaphores are safe across Python threads. That requires server60.

## Preregistered server60 gates

Use the promoted image and `max_num_seqs=4` profile as the control. Do not change the native gather, model, cache, graph, collective, or scheduler configuration.

Run these gates in order:

1. Verify the candidate worker hash inside the runtime and require the concurrent fan-out startup receipt.
2. Run a short C1, C2, and C4 smoke matrix. Reject any CUDA context, stream, semaphore, data race, deadlock, restart, or output error.
3. Collect matched control and candidate PLE timing. Require request-to-all-ranks-ready mean or p95 to improve by at least 0.25 ms at C2 or C4, with no regression above 0.10 ms at C1.
4. Run reverse-ordered service matrices. Require at least 1% decode improvement at C2 or C4 and no more than 1% loss at C1. Reject a prefill or TTFT regression above 5%.
5. Require aggregate KV capacity to remain at least 425,497 tokens. Reject any unexplained per-rank GPU residency increase above 64 MiB.
6. Re-run deterministic output, tools, reasoning, multimodal, C1, C2, C4, prefix-cache, CUDA Graph, and 261,544-token retrieval checks.
7. Restore the promoted service and verify its image, profile, health, restart policy, zero swap, and absence of timers.

Reject the candidate if it misses a gate. Do not combine it with async scheduling before recording its isolated result.
