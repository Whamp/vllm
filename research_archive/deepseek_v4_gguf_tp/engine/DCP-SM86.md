# DeepSeek V4 SM86 decode context parallelism

Date: 2026-08-28

## Decision

The native GGUF-TP runtime now has a working DeepSeek V4 SM86 decode-context-parallel path. Keep it experimental. Do not replace the 148K production service.

The accepted DCP profile uses TP=4, DCP=4, `dcp_comm_backend=a2a`, FP8 DS-MLA KV, prefix caching disabled, `max_num_seqs=2`, `max_num_batched_tokens=256`, FULL_DECODE_ONLY graphs, and an explicit 400 MB KV pool. It passes deterministic, tool, multi-turn, coding, reasoning, and exact-recall tests through 136K context. It decodes at 37.5 TPS, about 53% below the 79.8 TPS production baseline.

A 262,144-context profile starts with 373,421 KV tokens and 1.42x declared concurrency using a 700 MB pool, but it leaves only 75 MiB at idle and 11 MiB under the 240K probe. The 240K request hit the wrapper's 900-second timeout without crashing the engine. That is neither a recall pass nor a safe release profile.

## Implementation

Whamp/vLLM branch `feat/gguf-tp-dcp-sm86`, head `00793b3e5`, implements:

- compressed-entry ownership in one shared module:
  - `owner(e) = (e // I) % W`
  - `local(e) = (e // (I * W)) * I + e % I`
  - invalid entry `-1` passes through
- replicated `SlidingWindowSpec` and `SlidingWindowMLASpec` groups for SWA and compressor state
- DCP-aware scheduler, manager, worker block-table, and hash geometry
- compressed cache writes filtered by owner and translated into rank-local entry coordinates
- deterministic global sparse-indexer top-k with lower global index as the tie-break
- C128 rank-local metadata with fixed graph-safe row width
- byte-preserving compressed-cache all-gather for eager prefill, then the unchanged prefill attention math
- FlashMLA partial decode, query-head all-gather, fp32 LSE merge, and one-time attention-sink application
- persistent output, LSE, and SWA-length buffers for FULL_DECODE_ONLY graphs
- bounded indexer prefill workspace through `VLLM_DSV4_INDEXER_PREFILL_BUFFER_TOKENS`
- opt-in full-score, ownership, and global-cache comparison diagnostics under `VLLM_SM86_DCP_VALIDATE_TOPK`

Current fail-closed boundaries:

- `cp_kv_cache_interleave_size` must be 1
- cache format must be `fp8_ds_mla`; FP4 DCP partial decode is unsupported
- `dcp_comm_backend` must be `a2a`
- the patched partial FlashMLA operator must be present
- prefix caching stays disabled in the accepted profile
- prefill stays eager; only decode is captured

## Correctness defects found and fixed

1. Replicated SWA was initially included in every rank's partial attention, counting the same window four times. The fix keeps storage replicated but assigns each query's SWA computation to exactly one DCP rank by absolute position.
2. C128 metadata intentionally carries rank-local entry coordinates. The first port passed those integers to FlashMLA as physical cache slots. The fix translates C128 entries through the same local block table used by C4.
3. Partial FlashMLA dense index tails must contain an in-bounds value even though `extra_lens` masks them. The adapter now zero-fills inactive tail columns.
4. A NaN comparator initially failed open because comparisons with NaN return false. The diagnostics now reject non-finite query, partial output, LSE, merge output, and reference output explicitly.
5. Automatic KV allocation left 11 MiB physical headroom, so first-request JIT could fail while loading a 20–64 MiB module and leave a sticky CUDA error. The accepted 148K profile reserves 400 MB explicitly and leaves about 467 MiB.

The fixed 9,830-token discriminator returns `CRIMSON PLATYPUS 47` under both DCP=1 and DCP=4. Before the C128 fix, DCP=4 returned only `CR` and emitted NaN logprobs.

## GPU evidence

Kernel prerequisite, FlashMLA commit `2921831`, wheel SHA-256 `8de43339487ebbfbb06afc95a4bf48f306e755830500aaa1e3bdbcc635d3070c`:

- 9/9 DCP partial and narrow-prefill tests
- 50/50 FP8, INT8, and FP4 regressions
- Compute Sanitizer memcheck: zero errors
- Compute Sanitizer racecheck: zero hazards
- seven packaged SM86 cubins

vLLM CPU and structural evidence:

- 46 direct DCP contract tests
- 142 direct plus adjacent cache-manager/input-batch tests on the final tree
- Ruff check and format pass on all changed Python files
- `ty` passes on the new standalone DCP layout and merge modules
- whole-file `ty` remains red on pre-existing `gpu_model_runner.py` union/type diagnostics
- CodeGraph found no signature changes; repository-wide cycle/boundary warnings are pre-existing

## Measured profiles

| Profile | KV tokens | Physical free | Decode TPS | Result |
| --- | ---: | ---: | ---: | --- |
| Production DCP=1, 148K | 156,738 historical | tight | 79.82 narrative / 79.86 code | Current production |
| DCP=4, 148K, automatic KV | 353,403 | about 11 MiB | 37.82 pre-fix | Unsafe and correctness-rejected |
| DCP=4, 148K, 400 MB KV | 155,810 | about 467 MiB | 37.58 narrative / 37.53 code | Correct experimental profile |
| DCP=4, 262,144, 700 MB KV | 373,421 | 75 MiB idle, 11 MiB under 240K | not rebenchmarked | Startup passes; 240K timed out |

Corrected 148K stress results:

- exact recall at 9,213 and 27,513 tokens
- tool-prefill, IDE-agent, multi-turn, coding, and reasoning probes pass
- exact recall at 94K and 136K
- prefill: 460.9 TPS at 9K, 328.4 at 27K, 296.7 at 94K, 332.6 at 136K
- VRAM change across 94K to 136K: -4 MiB
- serving-process swap: zero

The 262K profile passed the short functional probes before the ceiling request. The 240K rung timed out after 900 seconds; no request remained and the engine stayed healthy. The normal 1 GiB VRAM release guard fails for every DCP profile tested.

## Server60 final state

The experiment container was removed. `dsv4-gguf-tp-prod` was restored on image `sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf` with restart policy `unless-stopped`, zero restarts, zero serving-process swap, and a successful deterministic Paris response. `gpu-power-limit.service` is active with every RTX 3090 fixed at 230 W and the 210–1650 MHz safety range. No restore timer remains.

## Remaining work

- Performance: 37.5 TPS is too slow for default promotion. The short-context identity merge did not improve the captured graph path.
- Capacity safety: reclaim at least 1 GiB per rank before treating 262K as sustained-agent capable.
- Prefix caching: the accepted profile disables it. Supporting it requires the P11 lookback/hash-geometry validation and a separate correctness campaign.
- Prefill: delta gather and fused FP8 prefill remain unported; the current full-prefix all-gather is correct but slows as context grows.

Raw logs and hashes are in `evidence/dcp-sm86-20260828/`.
