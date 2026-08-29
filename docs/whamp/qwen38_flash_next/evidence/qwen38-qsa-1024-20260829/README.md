# Qwen3.8 QSA 1,024-token A/B evidence

This directory preserves the complete control, diagnostic, acceptance, and
promotion record for lowering `max_num_batched_tokens` from 2,048 to 1,024 on
the Intel AutoRound Qwen3.8 EP=4 profile.

The decision and measurements are summarized in
[../../QSA-TOPK-BUFFER-1024.md](../../QSA-TOPK-BUFFER-1024.md).

## Reproduce the analysis

From this directory:

```bash
python3 analyze_qsa_ab.py
sha256sum -c SHA256SUMS
```

The analyzer also reads the sibling baseline reports under
`../qwen38-memory-baseline-20260829/raw`.

## File groups

- `plan.json` binds the one-variable experiment, control measurement, thresholds,
  image, model, PLE, and rollback identities.
- `benchmark.py.txt` is the exact executed matched benchmark source.
- `acceptance.py.txt` is the exact executed behavior and long-context source.
- `control-benchmark.json`, `candidate-benchmark.json`, and
  `promoted-benchmark.json` contain the matched 3-warmup/5-measured decode and
  1-warmup/3-measured prefill runs.
- `memory-reports/` contains 40 rank-local diagnostic reports plus their original
  checksum manifest.
- `acceptance.json.gz` and `promoted-acceptance.json.gz` contain deterministic, tool,
  post-tool, multimodal, concurrency-2, and 145K retrieval results.
- `benchlocal-quick.json.gz` is the no-retry 30-case BenchLocal result;
  `benchlocal-quick.md.txt` is its rendered report.
- `candidate.yml`, `production.yml`, and `control.yml` are the exact diagnostic,
  promoted, and rollback Compose contracts.
- `restore-production.sh.txt` and `promotion-rollback.sh.txt` are the tested recovery
  scripts.
- `production-startup-summary.log.txt`, `production-inspect.json`, and
  `production-final-state.txt` record the final image, launch settings, context,
  KV allocation, health, safety service, swap, and VRAM state.
- `analysis.json` is the deterministic summary emitted by
  `analyze_qsa_ab.py`.

The final service runs the original non-diagnostic image with BF16 QSA cache.
Only `max_num_batched_tokens=1024` changed from the control configuration.
