# FP8-indexer comparison provenance

The canonical benchmark and BenchLocal quick run were captured on 2026-08-20 against the `fp4_ds_mla` GGUF-TP candidate before enabling MXFP4 indexer caching.

- Served model: `deepseek-v4-flash-0731-gguf-tp`
- Benchmark endpoint: `http://127.0.0.1:8034`
- Benchmark runtime image: `club-3090/deepseek-v4-gguf-tp:fp4-ds-mla-dev6`
- Benchmark runtime digest: `sha256:6ec61abbbf4e00b59c5711431b75868abe06e264d4e9a949767f190222e3092c`
- vLLM source: `Whamp/vllm@633815f6889d9d033aefa04bf40cb270d5b6a3f1`
- FlashMLA source: `Whamp/forks-flash-mla-int@81a06aa6feb608bcba687a40acf60ee87d14f2da`
- Main MLA cache: `fp4_ds_mla`
- Sparse indexer cache: FP8, 132-byte physical rows
- Context: 148,000
- `max_num_seqs`: 2
- `max_num_batched_tokens`: 256
- GPU safety policy: 230 W and 210-1650 MHz
- Serving-process swap: zero during both measurements

`image-inspect.json` binds the benchmarked dev6 image identity. `fp4-final-equivalent.yml` is the subsequently published equivalent launch contract at Whamp/club-3090 commit `32263ba5`; it uses the reproducibly rebuilt final image `sha256:eb94d5049bf4d8d55c335ac1d2445382a811b7312d28e3e73088011a8103e181`, not the dev6 benchmark image. The source-level FP4 behavior is the same; the file is included to document launch arguments, not to relabel the benchmarked image.

`bench-canonical.log.gz` is the exact 3-warmup/5-measured benchmark output. `quality-quick.json` is the exact BenchLocal result used for the 27/30 comparison.
