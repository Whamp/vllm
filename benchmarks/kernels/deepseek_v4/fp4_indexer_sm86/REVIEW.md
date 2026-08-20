# Independent review record

## Scope

- Fixed point: `Whamp/vllm@81593507f`
- Accounting commit: `3f54512de`
- Initial indexer commit: `187116cef`
- Review axes: repository/Python standards, implementation spec, and adversarial GPU-kernel safety

## Reviewers

- OpenAI Codex GPT-5.6 Sol High: standards and spec review
- Cursor Grok 4.6 Extra High: adversarial kernel review
- Z.ai GLM 5.3 Max: unavailable; the provider returned repeated HTTP 429 quota-exhaustion responses and produced no review
- OpenAI Codex GPT-5.6 Sol Medium: fix-verification synthesis

## Findings and resolution

### Fail-closed platform handling

**Finding:** `assert` guarded the general architecture boundary, so optimized Python could remove it outside the earlier SM8x model-selection check.

**Resolution:** `supports_mxfp4_indexer_cache` now also requires an NVIDIA CUDA platform and accepts only SM86 or SM100-family devices. Metadata construction raises `ValueError` unconditionally on unsupported platforms. The unfused MXFP4 insertion branch raises `NotImplementedError` rather than relying on `assert`. Seven architecture cases cover SM86, SM100, SM80, SM89, SM90, SM120, and non-CUDA platforms.

### Top-k and downstream-output evidence

**Finding:** The initial test permitted one wrong top-k ID, tolerated broad order disagreement, and used a weak four-dimensional periodic output proxy.

**Resolution:** The test now requires exact top-k set equality. It requires pairwise order agreement whenever the reference margin exceeds twice the observed maximum logit error, leaving only numerically ambiguous near ties unordered. It gathers deterministic random 64-dimensional value rows through the selected set and requires bit-exact set-reduction equality. A separate tied-boundary case remains.

### Deployed `clean_logits=False` path

**Finding:** The paged test only exercised `clean_logits=True`, while production leaves out-of-range tail storage uninitialized and relies on sequence-length-bounded top-k.

**Resolution:** Tests now cover both modes for `next_n=1` and `next_n=4`. The false mode poisons unwritten logits with maximum FP32 values, checks valid logits, then passes the unsliced poisoned tensor through the real `_C.top_k_per_row_decode` operation with production sequence lengths and requires exact selected sets. CUDA-Graph replay remains deterministic.

### Oracle independence

**Finding:** The first test reference copied the production E2M1 threshold cascade.

**Resolution:** The reference now selects from a hard-coded E2M1 value table by minimum distance and resolves exact ties by even code parity. This is algorithmically independent of the production threshold cascade. UE8M0 decode and downstream FP32 accumulation remain explicit in the reference.

### Fused insertion-to-gather integration

**Finding:** Direct logits and writer tests did not prove that the existing cache gather operation interpreted the new segregated page correctly.

**Resolution:** Every FP4 fused indexer-writer case now gathers through `cp_gather_indexer_k_quant_cache` and requires byte-exact packed-value and UE8M0-scale equality with the independent writer reference.

### Baseline provenance

**Finding:** Initial durable evidence omitted the exact FP8-indexer benchmark and BenchLocal inputs used for reported deltas.

**Resolution:** `evidence/baseline_comparison/` now contains the exact canonical benchmark log, raw BenchLocal JSON, benchmarked image inspection, final-equivalent resolved profile, provenance note, and SHA-256 manifest. The note distinguishes the benchmarked dev6 image from the later reproducibly rebuilt equivalent image.

### Report claims

**Finding:** The first report over-attributed the increase in available pool memory to smaller profiling/gather workspaces and did not narrowly scope sanitizer evidence.

**Resolution:** The report now separates the exact 58,834,944-byte logical indexer saving from the 111,214,592-byte observed available-pool increase and labels the residual attribution unresolved. Sanitizer claims are limited to the paged-decode and fused-query-writer tests actually run. NIAH is labeled integration evidence, not a numerical oracle. The kernels are explicitly described as reusing FP8 autotune configurations and not as tuned SM86 kernels.

### Public documentation and logging

**Finding:** The flag docstring still said unsupported, and the full allocation JSON logged at INFO.

**Resolution:** The flag now documents a backend-gated MXFP4 indexer cache. Full allocation JSON logs at DEBUG; normal capacity summaries remain at INFO.

### Commit trailers

**Finding:** The pushed accounting commit lacked the repository's requested attribution trailers.

**Resolution:** Published history was not rewritten. Subsequent local commits use `Assisted-by` and `Signed-off-by` trailers. The original process violation remains recorded here rather than hidden by force-pushing a published commit.

## Findings not changed

- The backend's supported head sizes describe semantic attention widths, not physical quantized row bytes; adding physical width 68 would mix two interfaces.
- `next_n > 2` flattening on SM86 and inherited FP8 autotune choices are performance opportunities, not correctness defects. Measured deep-prefill loss keeps the feature opt-in.
- Full allocation-report construction remains available for deterministic accounting, while its large serialized form is DEBUG-only.

## Verdict

Proceed as an explicit capacity experiment after the strengthened GPU tests, repository gates, packaging checks, and healthy final-state audit pass. Do not promote it as the default sparse-indexer format.
