# Qwen3.8 GPU utilization evidence

This directory supports
[../../GPU-MEMORY-UTILIZATION-0968.md](../../GPU-MEMORY-UTILIZATION-0968.md).

- `rejected-097/` preserves the 0.97 plan, Compose, rollback, benchmark,
  acceptance, NIAH, and GPU state that failed the 1 GiB margin by 4–8 MiB.
- `accepted-0968/` preserves candidate and production plans, control and
  promoted Compose files, rollback scripts, matched benchmarks, 190K NIAH,
  BenchLocal quick results, metrics, GPU states, and the final zero-swap service
  record.

The only experimental variable was `gpu_memory_utilization`. Both profiles used
the same immutable image, model and PLE revisions, BF16 QSA cache, 262,144 RoPE
rows, and scheduler settings.
