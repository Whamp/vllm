<!-- markdownlint-disable MD060 -->

# GGUF-TP cold-expert offload route study

Status: complete. The uniform H=224 cache fails its preregistered coverage gate. This experiment did not benchmark offloaded decode speed.

## Decision

The preregistered gate required at least 99% routed-expert visit coverage with no more than 224 of 256 experts resident in every layer. The full 43-layer capture fails:

| Workload | Captured token rows per layer | Median H99 | Worst H99 | Worst layer |
| --- | ---: | ---: | ---: | ---: |
| SuperJSON pilot final context | 41,987 | 209 | 250 | 0 and 2 |
| 12 completed coding-agent sessions | 926,529 | 216 | 251 | 0, 1, and 2 |

The decision is NO-GO because any layer at H99 of 248 or higher rejects the design. The corpus needs 251 resident experts in its worst layers.

At H=224, corpus coverage falls to 92.66% in layer 0. A fixed hot set would send roughly 7.3% of that layer's expert visits to the cold path before counting transfer latency or cache-management work.

## Workloads

The pilot request reconstructs the passed one-worker SuperJSON DeepSWE trajectory after Pi compaction. The server rendered 41,943 prompt tokens and generated one token. Boundary requests used to publish complete snapshots account for the remaining captured rows.

The second workload replays 12 completed GGUF-TP coding-agent sessions. These sessions represent 8.70 agent-hours of real work. All 12 requests returned HTTP 200 at their full server-rendered sizes. Prompt lengths ranged from 25,141 to 125,307 tokens and totaled 926,485 tokens. The service generated one token per request.

The capture preserved production prefix-cache behavior. This is intentional. Expert offload must serve routes the engine executes after cache hits, not routes it would execute with caching disabled.

## Full coverage result

The complete H=1..256 curves are in `dynamic-capture-20260820/analysis.json`. Selected results follow.

| Layer group | Pilot H99 range | Corpus H99 range | Corpus minimum coverage at H=224 |
| --- | ---: | ---: | ---: |
| Static hash-routed layers 0–2 | 249–250 | 251 | 92.66% |
| Dynamic activation-routed layers 3–42 | 184–228 | 199–236 | 98.02% |
| All 43 layers | 184–250 | 199–251 | 92.66% |

The dynamic layers are more concentrated than the static hash-routed layers, but several still exceed H=224. Corpus layers 3, 5, 6, 7, 9, 10, 11, 19, and 31 need more than 224 experts for 99% coverage.

## Cross-workload stability

The pilot and corpus do not share one stable H=224 hot set:

| Metric across 43 layers | Minimum | Median | Maximum |
| --- | ---: | ---: | ---: |
| H=224 hot-set Jaccard | 0.867 | 0.939 | 0.974 |
| Pilot H=224 set coverage on corpus | 91.64% | 98.81% | 99.71% |
| Corpus H=224 set coverage on pilot | 92.85% | 99.39% | 99.88% |

A fixed set learned from one workload would perform worse than each workload's own best set in the layers that already fail the gate.

## Consecutive reuse

The immutable GGUF stores exact token-ID routing tables for layers 0–2. Those tables provide an exact temporal-locality proxy without adding GPU instrumentation.

| Workload | H=224 LRU hit rate | H=248 LRU hit rate | Mean overlap between consecutive top-6 sets |
| --- | ---: | ---: | ---: |
| SuperJSON final context | 95.49–95.69% | 99.02–99.07% | 2.2–3.0% |
| 12-task corpus | 95.26–95.50% | 99.04–99.07% | 2.2–3.0% |

These LRU rates cover only layers 0–2. At H=224 they imply about 0.8 misses per token from those three layers' 18 expert visits. They must not be extrapolated to all 43 layers.

Across all 43 layers, an oracle fixed set selected from the same complete workload would miss 2.17 expert visits per pilot token and 3.11 per corpus token at H=224. This is a best-case frequency result, not a decode-speed measurement. All-layer LRU behavior remains unmeasured.

The optional all-layer decode ring was not used. Its 34 MiB per-rank allocation made the 148K production profile fail the KV fit gate. The uniform H=224 design had already failed its preregistered gate, so the ring was not allowed to consume the production KV margin.

## Nonuniform cache implication

The result rejects a uniform limit of H≤224 in every layer. It does not reject every allocation with the same total memory budget.

For the corpus, assigning each layer its own H99 requires an average H of 219.77 and leaves 1,558 expert slots cold. Uniform H=224 leaves 1,376 slots cold. Taking the union of the pilot and corpus H99 hot sets requires an average H of 223.91, leaves 1,380 slots cold, and covers at least 99% of both measured workloads by construction. Individual layers range as high as H=254.

This nonuniform allocation preserves approximately the same aggregate expert-slot saving as uniform H=224. It is a plausible follow-up, but it still needs all-layer temporal-locality evidence and a real offloaded-decode benchmark. The present experiment does not establish that its decode loss would be below 10%.

## Capture and provenance

Whamp/vLLM branch `research/gguf-tp-route-stats` owns the diagnostic implementation:

- `e0646f991` added per-layer histograms and an optional decode ring.
- `761b48a44` separated ring allocation from histogram capture.
- `7ef128567` added a validated histogram flush interval while keeping the 300-second default.

The final server60 capture used image `sha256:5fab88440740a6033bcacda473ffaeed7a4f4e386d494b516432487f0df09729` at 148K context, max_num_seqs=2, and max_num_batched_tokens=256. It retained the production FlashMLA decode, hierarchical all-reduce, Marlin-diagonal wo_a, Q8 KV, and exact GGUF bytes.

Every retained boundary has four rank-local snapshots. The extractor rejects missing ranks, metadata disagreement, counter rollback, partial top-k rows, unequal layer token totals, or rank histogram disagreement. All checks passed. Serving-process swap stayed at zero throughout accepted capture.

The runtime image and capture evidence are under `dynamic-capture-20260820/`. `SHA256SUMS` binds the compact archive.

## Reproduction

- `build_deepswe_route_replay.py` reconstructs Pi sessions and requires exact agreement with captured provider requests.
- `render_deepswe_route_tokens.py` performs CPU-only DeepSeek rendering.
- `extract_dynamic_route_histograms.py` validates and merges rank-local boundary snapshots.
- `analyze_route_skew.py` emits coverage curves, cross-workload transfer, and the preregistered decision.
- `build_static_route_workload.py` computes exact static-layer reuse from GGUF routing tables.

## Next work

The uniform H=224 implementation stops here. A nonuniform cache is a separate design decision rather than a passing result for the preregistered design. Part B of the active goal is independent: capture the TP=4 layer-slice Nsight trace, verify the launch-latency hypothesis, and only then decide whether to implement the bounded decode fusion pass.
