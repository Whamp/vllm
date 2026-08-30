# Warmed stability evidence

This directory closes the allocator and workload-stability gate for the promoted
Qwen3.8 FP8-QSA hierarchical-all-reduce profile.

## Result

- Deterministic, tool, post-tool, multimodal, concurrency-2, and 261,544-token
  NIAH checks passed.
- A separate first-block-nonced 261,549-token NIAH check passed in 188.55 seconds.
- All recorded allocator counters were unchanged from execution step 500 to 750
  on every rank.
- During the cold long-context check, 162 NVML samples per GPU showed no memory
  growth. Minimum free memory was 1,370 MiB on rank 0 and 1,394 MiB on ranks 1-3.
- Serving-process swap remained zero.
- The normal production image was restored healthy with zero restarts, swap
  disabled, the 230 W safety service active, and no rollback timer.

The allocator did make a one-time first-use reservation before reaching its
stable state. Reserved memory increased by 360-380 MiB per rank from startup
warmup to execution step 500, while active allocation increased by 5.37 MiB.
The exact counters are in `stability-summary.json`.

## Files

- `stability-summary.json`: deterministic reduced result.
- `analyze_stability.py`: analyzer that regenerates the summary after extracting
  `allocator-reports.tar.gz` into `reports/`.
- `allocator-reports.tar.gz`: all 72 rank-local stage reports and their internal
  checksum manifest.
- `diagnostic-context.tar.gz`: exact diagnostic image and launch/restore context.
- `nvml-timeseries.csv` and `process-swap-timeseries.csv`: cold long-context
  telemetry.
- `acceptance.json` and `niah-cold.json`: functional and long-context results.
- `metrics-before.txt.gz` and `metrics-after.txt.gz`: scheduler and KV state.
- `final-production-*`: restored production identity and safety state.

`SHA256SUMS` covers every file in this directory. `stability-archives.sha256`
also binds the two deterministic archives.
