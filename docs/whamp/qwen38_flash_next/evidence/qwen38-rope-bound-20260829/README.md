# Qwen3.8 runtime-bounded RoPE evidence

This directory supports [../../QSA-ROPE-BOUND.md](../../QSA-ROPE-BOUND.md).

## Contents

- `memory-reports/` contains all 40 diagnostic reports and a relocatable
  SHA-256 manifest.
- `memory-summary.json` is the deterministic analyzer output.
- `property-tests/` contains the QSA bound examples, the generated multimodal
  property in the existing Qwen3-VL suite, red-green runs, and both counterfeit
  kills.
- `rope-source.patch.txt.gz` contains the canonical one-file implementation.
- `runtime-overlay.tar.gz` contains production and diagnostic Dockerfiles plus
  the exact generated Qwen3.8 runtime source and manifest.
- `server-runtime/` contains executed Compose and rollback contracts, matched
  benchmarks, functional and NIAH results, BenchLocal quick results, metrics,
  and final GPU/model state.

The final live image is
`sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b`.
It serves 167,600-token context with BF16 QSA cache, zero process swap, and no
active rollback timer.
