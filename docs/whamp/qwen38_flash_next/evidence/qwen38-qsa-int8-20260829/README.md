# Qwen3.8 QSA INT8 evidence

This directory preserves the rejected per-token-head INT8 QSA cache experiment
summarized in [../../QSA-INT8-CACHE.md](../../QSA-INT8-CACHE.md).

## Contents

- `int8-source.patch.txt.gz` contains the rejected canonical QSA source and
  example-test changes.
- `runtime-overlay.tar.gz` contains the final Qwen3.8 semantic port, Dockerfile,
  and image manifest.
- `gpu-gate/` contains the exact executed gate and complete RTX 3090 result.
- `property-tests/` contains the Hypothesis property, its final green search,
  and the shrunk failures from the temporary stride counterfeit.
- `production-final-state.txt` records the restored BF16 service.

The implementation is evidence only. Do not apply it. M=256 sparse-reader time
was 26.32 times BF16 against a fixed maximum ratio of 1.25.
