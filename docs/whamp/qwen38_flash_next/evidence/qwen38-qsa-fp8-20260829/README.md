# Qwen3.8 QSA FP8 evidence

This directory preserves the rejected FP8 QSA cache experiment summarized in
[../../QSA-FP8-CACHE.md](../../QSA-FP8-CACHE.md).

## Contents

- `attempt-history.json` records the typed E4M3, typed E5M2, two-stage software
  E4M3, and final one-stage software E4M3 decisions.
- `fp8-source.patch.txt.gz` is the complete rejected canonical source and test diff.
- `runtime-overlay.tar.gz` contains the final Qwen3.8 semantic port, Dockerfile,
  and checksum-bound image manifest.
- `gpu-gate/gpu-gate.py.txt` is the exact executed gate source.
- `gpu-gate/gpu-gate.json` is the complete writer, numerical, graph, and timing
  result.
- `production-final-state.txt` records the restored BF16 service.

The source patch is evidence only. It must not be applied or promoted because
the M=256 FP8 kernel failed its performance gate by a factor of 22.77 over the
maximum allowed ratio and by a factor of 28.46 against BF16.
