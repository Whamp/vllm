# Qwen3.8 QSA INT4 evidence

This directory preserves the rejected Q4 QSA cache experiment summarized in
[../../QSA-INT4-CACHE.md](../../QSA-INT4-CACHE.md).

## Contents

- `int4-source.patch.txt.gz` contains the rejected canonical implementation.
- `matrix-runtime-overlay.tar.gz` contains the matrix-RHT Qwen3.8 image overlay.
- `property-tests/` contains generated test source, red-green evidence, the
  initially surviving packed-width counterfeit, its repaired kill, and the
  token-stride kill.
- `server-results/property*` and `matrix-property*` contain the generated RTX
  3090 semantic searches.
- `server-results/counterfeit-*` contain the three GPU counterfeit kills.
- `server-results/gpu-gate*` and `matrix-gpu-gate*` contain the fixed numerical,
  CUDA-Graph, and performance gates.
- `server-results/decode-attribution.json` records RHT and packed-core timing.
- `server-results/retile-*` contain four decode schedule A/Bs.
- `gates/` contains exact executed gate and attribution scripts.
- `production-final-state.txt` records the restored BF16 service.

The code is evidence only. Do not apply it. Numerical, property, and M=256 gates
passed, but the best M=1 reader remained 2.03 times BF16 against a fixed maximum
ratio of 1.25.
