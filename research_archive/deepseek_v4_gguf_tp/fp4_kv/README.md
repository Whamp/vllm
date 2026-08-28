# DeepSeek V4 GGUF-TP FP4 DS-MLA cache

This directory owns the design, acceptance evidence, and promotion decision for
the optional `fp4_ds_mla` cache added to the server60 GGUF-TP runtime.

- [REPORT.md](REPORT.md) — format, implementation, causal gates, matched results,
  defects found during bring-up, and remaining risk.
- [`evidence/server60-20260820/`](evidence/server60-20260820/) — compact raw logs,
  matched result JSON, image-build output, and `SHA256SUMS`.

Implementation sources:

- `Whamp/forks-flash-mla-int@81a06aa6feb608bcba687a40acf60ee87d14f2da`
  — native SM86 FP4 sparse-MLA decode and prefill.
- `Whamp/vllm@633815f6889d9d033aefa04bf40cb270d5b6a3f1`
  — cache allocation, writers, readers, paging, attention dispatch, and tests.
- `models/deepseek-v4-flash-0731/vllm/gguf-tp/FP4-MANIFEST.json`
  — complete thin-image build identity.

The FP8 production profile remains `compose/multi4/gguf-tp/base.yml`. The FP4
profile is the validated opt-in `fp4.yml` because its 148K configuration remains
a capacity-ceiling profile below the normal 1 GiB VRAM release margin.
