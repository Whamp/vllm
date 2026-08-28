# DeepSeek V4 native GGUF research

This directory is the permanent record for native DeepSeek V4 GGUF execution in Whamp/vLLM.

Start with [the runtime history](DEEPSEEK-V4-GGUF-RUNTIME-HISTORY.md). It compares the Antirez and Unsloth models, llama.cpp and vLLM, FP8 and FP4 KV caches, validated context lengths, speed, quality, and failed experiments.

The implementation is specific to the Whamp fork. It is not upstream vLLM support.

## What is in Whamp/vLLM main

Whamp/vLLM main contains the native Antirez GGUF tensor-parallel loader and kernels, plus the FP4 DeepSeek MLA cache. The main implementation lives in:

- `vllm/model_executor/layers/quantization/gguf_dsv4/`
- `vllm/model_executor/model_loader/gguf_dsv4.py`
- `vllm/model_executor/model_loader/gguf_dsv4_io.py`
- `vllm/model_executor/model_loader/gguf_dsv4_plan.py`
- `csrc/libtorch_stable/quantization/gguf_dsv4/`
- `vllm/models/deepseek_v4/cache_layout.py`

Unsloth IQ1 support and decode context parallelism remain on separate branches until they receive human review and are intentionally merged.

## Main reports

- [Runtime history and result table](DEEPSEEK-V4-GGUF-RUNTIME-HISTORY.md)
- [Implementation progress](PROGRESS.md)
- [Original implementation plan](PLAN.md)
- [First full-model acceptance](M5-M7-RUNTIME.md)
- [DeepSWE behavioral test](M8-DEEPSWE.md)
- [FP4 KV report](fp4_kv/REPORT.md)
- [Unsloth IQ1 performance record](iq1/PERF-PREP-20260821.md)
- [Decode context parallelism](DCP-SM86.md)
- [Capacity accounting](CAPACITY.md)
- [Cross-engine numerical test](M6-LAYER-ORACLE-SPEC.md)
- [Raw evidence archive](EVIDENCE.md)

## Technical reports

- [GGUF format contract](FORMAT-CONTRACT.md)
- [Tensor-parallel mapping](TP-MAPPING.md)
- [Runtime dtype contracts](DTYPE-CONTRACTS.md)
- [Q8_0 output-projection design](WOA-DESIGN.md)
- [Loader report](M4-LOADER.md)
- [Grouped expert prefill](M2-GROUPED-PREFILL.md)
- [Layer-slice performance](M2-LAYER-SLICE.md)
- [Fusion trace and no-go decision](FUSION-TRACE.md)
- [Cold-expert route study](route-offload/ROUTE-OFFLOAD.md)
- [SM86 DCP import audit](SM86-DCP-IMPORT-AUDIT.md)

`BLOG-POST.md`, `BLOGPOST-NOTES.md`, and `TWITTER-POST.md` are preserved drafts. Their numbers reflect the point in time when each draft was written. Use the runtime history for final results.
