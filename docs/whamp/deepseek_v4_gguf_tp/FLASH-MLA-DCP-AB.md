<!-- markdownlint-disable MD060 -->

# FlashMLA partial-decode and narrow-prefill A/B

Date: 2026-08-21

## Decision

Keep the merged FlashMLA branch as the DCP kernel base. Do not promote it as a standalone production speed change.

The branch passed every RTX 3090 correctness gate. It did not change production throughput beyond measurement noise because the current FP8 service uses the combined FlashMLA decode operation and Triton prefill. Partial decode is inactive until DCP is enabled, and the fused BLOCK_M=16 prefill operation is not wired into this production path.

## Pinned artifacts

- FlashMLA source: `Whamp/forks-flash-mla-int`, branch `feat/dcp-partial-fp4`, commit `2921831`
- Wheel: `flash_mla-2.0.0-cp39-abi3-linux_x86_64.whl`
- Wheel SHA-256: `8de43339487ebbfbb06afc95a4bf48f306e755830500aaa1e3bdbcc635d3070c`
- Candidate image: `club-3090/deepseek-v4-gguf-tp:dcp-ab-2921831`
- Baseline image: `club-3090/deepseek-v4-gguf-tp@sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf`
- Model: `deepseek-v4-flash-0731-gguf-tp`
- Serving profile: TP=4, 148,000 context, `max_num_seqs=2`, `max_num_batched_tokens=256`, FP8 DS-MLA KV
- Hardware policy: 230 W per RTX 3090, 210–1650 MHz graphics-clock range

The first SM86 wheel build caught a missing `fp4_ds_mla.cuh` include in the unified prefill translation unit. Commit `2921831` fixes that build defect. The final wheel contains seven `sm_86` cubins.

## Correctness evidence

All tests ran on server60 RTX 3090 GPUs against the final wheel:

- DCP partial decode and narrow-tile prefill: 9/9 passed
    - partial decode versus independent oracle
    - two-way and four-way partial merge versus combined decode
    - prefill oracle at 8, 16, 32, and 64 heads
    - BLOCK_M=16 versus BLOCK_M=32 agreement
- FP8, INT8, and FP4 decode/prefill regression set: 50/50 passed
- Compute Sanitizer memcheck: 9/9 passed, zero errors
- Compute Sanitizer racecheck: 9/9 passed, zero hazards, errors, or warnings
- CUDA Graph determinism is covered by the partial and native-format regression tests.

## Matched service benchmark

Both arms used the same model, Compose configuration, server state, three warmups, five measured narrative runs, five measured code runs, and three cache-busted runs at each prefill depth. Serving-process swap was zero. All cards reached but did not exceed the fixed 1650 MHz clock ceiling.

| Metric | Baseline | Candidate | Delta |
| --- | ---: | ---: | ---: |
| Narrative decode TPS | 79.82 | 79.76 | -0.08% |
| Code decode TPS | 79.86 | 79.81 | -0.06% |
| 10K prefill TPS | 541.22 | 539.60 | -0.30% |
| 90K prefill TPS | 520.73 | 521.06 | +0.06% |

The deltas are too small and inconsistent to support a performance claim. The branch is accepted only as the validated kernel prerequisite for DCP.

## Final server state

After the A/B, server60 returned to the digest-pinned production image with restart policy `unless-stopped`. The service was healthy with zero restarts, zero serving-process swap, the GPU safety service active, and all four cards at the 230 W / 1650 MHz limits under the deterministic canary.

## Reproduction evidence

Raw logs and checksums are in `evidence/flash-mla-dcp-ab-20260821/`.
