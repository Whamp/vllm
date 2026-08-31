# BF16 SSD PLE CPU evidence

This bundle records the CPU-only server60 evidence collected before any full-model BF16 PLE launch.

`summary.json` contains:

- the Intel PLE artifact's pinned and downloaded revision identities;
- the full-file direct-I/O SHA-256 result;
- the bounded 260-row independent oracle;
- the live NVFP4 mapping inventory;
- the default and `MADV_RANDOM` SSD probes;
- the result from `benchmark_bf16_ssd_gather.py`.

The BF16 runs explicitly evicted only the BF16 file's clean cache pages before each cold probe. The NVFP4 runs used distinct row seeds without eviction because the production service mapped those same files. The NVFP4 results therefore reflect the observed cache state and are not guaranteed-cold measurements.

No GPU model changed during this evidence collection. The running service remained the accepted NVFP4 PLE profile.
