# Qwen3.8 memory baseline evidence

This directory preserves the staged four-rank model-memory capture used by
[`GPU-MEMORY-BASELINE.md`](../../GPU-MEMORY-BASELINE.md).

## Contents

- `raw/` contains 40 atomic JSON reports, ten stages for each TP rank.
- `raw/SHA256SUMS` binds every raw report.
- `analyze_reports.py` verifies the manifest, exact stage/rank inventory,
  storage-category coverage, and then writes the deterministic summary.
- `baseline-summary.json` is the generated machine-readable result.

The reports contain tensor names, shapes, dtypes, storage identities local to
the process, and allocator/device counters. They contain no model values or
credentials.

## Reproduce

From this directory:

```bash
python3 analyze_reports.py raw /tmp/qwen38-memory-baseline.json
cmp baseline-summary.json /tmp/qwen38-memory-baseline.json
(cd raw && sha256sum -c SHA256SUMS)
sha256sum baseline-summary.json raw/SHA256SUMS
```

Expected hashes:

```text
f0281c9896c32d70cfde9b349e047a9e2e8a24ec94f7d2e119cc4b1e2534760a  baseline-summary.json
690ae384e4385a1cea3db814c78be6c1ccd9612bbd11299bd44ac853565c78ee  raw/SHA256SUMS
```

The raw reports came from server60 image
`sha256:59e1df5a8023f7a9c8ee331321826efd6c68ea1bb165740e9a7f48d4e13200ec`,
which applied the opt-in diagnostics from Whamp/vLLM commit
`1833ca8579be3075bbe4c89d24f9e32ceb275ce1` to base image
`sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`.
