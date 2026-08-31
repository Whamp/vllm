# Qwen3.8 shared-expert post-BIOS x8 evidence

This compact bundle preserves the forward/reverse same-image benchmark, fresh
matched Nsight analysis, exact candidate and production identities, topology,
telemetry, and final restoration record for the post-BIOS x8 retest.

The result is a no-go for the current early-launch mechanism: C1 decode regressed
by 9.97%, C2 aggregate decode improved by only 0.90%, C4 was flat, and fresh
CUDA Graph traces showed longer C1 and C2 graph spans.

Raw Nsight reports remain on server60 at:

```text
/home/will/inference/runtime/qwen38-shared-expert-overlap/evidence/post-bios-x8-20260831
```

Their identities are pinned by `RAW-TRACE-SHA256SUMS`. Archived scripts and logs
are gzip-compressed with timestamps removed; decompression recovers the exact
server-side bytes. `SHA256SUMS` verifies this compact bundle.
