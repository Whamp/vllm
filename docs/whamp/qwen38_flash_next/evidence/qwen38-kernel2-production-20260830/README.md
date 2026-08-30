# Qwen3.8 Kernel2 production evidence

This directory records the server60 acceptance of the native SM86 BF16
hyperconnection projection path at Whamp/vLLM commit
`42b918e36fa3bdd04e3d7bd7ad4a9c7695b9624f`.

## Result

The same-image flag ablation isolates the native selector from stable-extension
rebuild drift.

| Measurement | Kernel2 disabled | Kernel2 enabled | Change |
| --- | ---: | ---: | ---: |
| c=1 decode | 49.2643 tok/s | 53.1284 tok/s | +7.84% |
| c=2 aggregate decode | 79.4293 tok/s | 81.5672 tok/s | +2.69% |
| c=1 cache-busted prefill | 1,536.69 tok/s | 1,532.68 tok/s | -0.26% |
| c=2 aggregate prefill | 1,552.69 tok/s | 1,556.74 tok/s | +0.26% |

The earlier cross-image comparison measured +7.26% c=1 decode, -0.12% c=2
decode, -0.06% c=1 prefill, and +0.85% c=2 prefill. The exact final
production benchmark measured 51.8565 decode tokens/s and 1,538.9902 prefill
tokens/s.

BenchLocal quick scored 28/30: 14/15 ToolCall and 14/15 InstructFollow. The
established deterministic, automatic-tool, post-tool, multimodal, two-stream,
and 261,492-token NIAH checks also passed.

## Production identity

| Item | Value |
| --- | --- |
| Image | `sha256:acff9d8e08096a2265b23e50f5ff0d52a3f1e95ffa91e2fb099346e274a9b735` |
| `production.yml` SHA-256 | `bfd7267653464d604f92a8b27f1965467139dc8543760a53b86b844667207d17` |
| Source commit | `42b918e36fa3bdd04e3d7bd7ad4a9c7695b9624f` |
| Stable extension SHA-256 | `91118abf4f8b94e1b41dc4226dfd1ef9cf32bd69a610156543c649b57e523381` |
| Runtime selector count | 193 Qwen3.8 linears per TP rank |
| Model context | 262,144 tokens |
| Endpoint | `http://server60:30002/v1` |

The archived final-state record shows health `healthy`, zero restarts, restart
policy `unless-stopped`, zero host and serving-process swap, active 230 W GPU
power controls, and no rollback timer.

## Reproduce the summary

Run:

```bash
python3 analyze_kernel2_production.py > reproduced-summary.json
cmp summary.json reproduced-summary.json
sha256sum -c SHA256SUMS
```

The analyzer reads the raw service matrices, exact-final benchmark, and
BenchLocal result. It does not contain measured values beyond immutable
provenance identifiers.

## File map

- `gpu-gate.json`: three-seed numerical and deterministic CUDA Graph results.
- `memcheck.log.gz`, `racecheck.log.gz`: Compute Sanitizer acceptance.
- `extension-elf.txt`, `extension.sha256`: packaged SM86 extension evidence.
- `baseline-matrix.json`: earlier production-image service matrix.
- `candidate-matrix.json`: selector-enabled service matrix.
- `ablation-matrix.json`: same-final-image selector-disabled matrix.
- `final-production-benchmark.json`: exact production-named c=1 benchmark.
- `acceptance.json`: functional, multimodal, NIAH, and concurrency checks.
- `benchlocal-quick.json`, `benchlocal-quick.md.gz`: 28/30 quality result.
- `production.yml`, `restore-kernel2.sh`: accepted deployment contract.
- `production-before-kernel2.yml`, `restore-pre-kernel2.sh`: rollback contract.
- `build-exact-extension.sh`, `patch-legacy-runtime.py.gz`, and
  `promote-production.py.gz`: exact legacy-runtime build and promotion scripts.
- `create-ablation.py.gz`: same-image selector-ablation Compose generator.
- `final-container-inspect.json`, `final-gpu.csv`, `final-state.txt`: final state.
