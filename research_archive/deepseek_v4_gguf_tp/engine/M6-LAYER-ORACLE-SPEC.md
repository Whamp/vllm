# M6 class-B layer oracle — pre-registered specification

Status: thresholds and exact prompt fixed **before the first layer-dump run**.

## Question

Does native GGUF-TP preserve the pinned Antirez llama.cpp model's numerical path closely enough at every decoder-layer boundary and final output projection to proceed to M8?

This gate tests the same IQ2_XXS/Q2_K/Q8_0 bytes with exact shared token IDs. It does not compare chat templates, sampling, or generated text.

## Fixed input

`m6-layer-oracle/render-request.json` contains one OpenAI conversation with:

1. a weather-tool request;
2. a completed assistant tool call;
3. its JSON tool result;
4. a final user continuation.

The DSML-fixed GGUF-TP `/v1/chat/completions/render` endpoint produced the exact 366 IDs in `m6-layer-oracle/token-ids.txt`. Both engines must consume that file directly at positions 0–365. The token file's SHA-256 is part of each run receipt. No engine may re-tokenize the text.

## Observation point

For each of the 43 layers, capture the final prompt token after DeepSeek V4's post-FFN hyperconnection reconstruction (`hc_ffn_post` in llama.cpp; `mhc_post_tilelang` in vLLM). The logical shape is `[hc_mult=4, hidden_size=4096]`, converted to FP32 only for comparison.

Capture the final-token 129,280-element vocabulary logits after output projection as a second observation.

The llama.cpp implementation uses its existing scheduler evaluation callback in a standalone diagnostic executable. The vLLM implementation is diagnostic-only and requires all of:

- quantization method `gguf_dsv4`;
- eager execution;
- TP rank 0;
- an exact token-ID match;
- a new or empty output directory.

Normal serving remains inert.

Pinned diagnostic implementations:

- Whamp/vLLM `41a672a0be912c23e97a4230d124f9d55f50a4cb` (tree `e9ede11c6b32581b12afb3f35e6beb3c8d93b1d9`), based on validated GGUF-TP `3ec20cebe`;
- Whamp/llama.cpp `04636336e8bb0be49cbb45d00bd215bc9f124ff2` (tree `426c87947153b2ab2a78d3b60e1331bcdf48a2d6`), based on canonical `0379cf4bf889f3d28038a005210c4bc193fc8ba1`.

`m6-layer-oracle/implementation.json` checksum-binds the diagnostic and comparator sources. The vLLM CPU contract has 10 passing tests. The llama translation unit compiles; the pinned fork's known unconditional CUDA symbol prevents a CUDA-off full link, so final build proof must use the real CUDA build.

## Metrics

For reference vector `r` (llama.cpp) and candidate `c` (GGUF-TP):

- normalized RMSE: `sqrt(mean((c-r)^2)) / sqrt(mean(r^2))`;
- normalized MAE: `mean(abs(c-r)) / mean(abs(r))`;
- cosine similarity;
- final-logit top-1 equality;
- final-logit top-10 set overlap.

All denominators must be finite and nonzero. Every file must match its manifest size and SHA-256 before comparison.

## Pass window

Layer boundaries must satisfy:

- every layer cosine similarity **≥ 0.995**;
- every layer normalized RMSE **≤ 10%**;
- every layer normalized MAE **≤ 10%**;
- median layer cosine similarity **≥ 0.999**;
- median layer normalized RMSE **≤ 3%**;
- median layer normalized MAE **≤ 3%**.

Final logits must satisfy:

- cosine similarity **≥ 0.995**;
- normalized RMSE **≤ 10%**;
- normalized MAE **≤ 10%**;
- exact top-1 token equality;
- top-10 set overlap **≥ 8/10**.

The per-component class-B windows are 1% normalized RMSE/MAE. The 10% per-layer ceiling permits bounded accumulation and different reduction order across 43 layers without accepting a large mapping, routing, projection, or cache error. The 3% median requirement prevents that broad ceiling from hiding systematic drift.

## Decision

- **Pass:** all layer and final-logit requirements pass; combine with the existing deterministic/tool/post-tool, NIAH, quick-pack, graph, and kernel evidence and proceed to M8.
- **Fail:** stop M8 and bisect the earliest failing layer. Do not relax thresholds after seeing results.
- **Infrastructure failure:** preserve both dumps and logs, repair the harness, and rerun the same token file and thresholds.
