# DeepSeek V4 GGUF-TP research

Start with [DEEPSEEK-V4-GGUF-RUNTIME-HISTORY.md](DEEPSEEK-V4-GGUF-RUNTIME-HISTORY.md). It is the consolidated reference for Antirez, Unsloth UD-IQ1_S, Unsloth UD-IQ1_M, FP8 and FP4 KV, llama.cpp, native GGUF tensor parallelism, DCP, throughput, context, and promotion decisions.

## Main reports

- [Runtime history and result matrix](DEEPSEEK-V4-GGUF-RUNTIME-HISTORY.md)
- [Implementation progress](PROGRESS.md)
- [Original implementation plan](PLAN.md)
- [First full-model runtime acceptance](M5-M7-RUNTIME.md)
- [DeepSWE behavioral gate](M8-DEEPSWE.md)
- [FP4 DS-MLA report at commit `32263ba5`](https://github.com/Whamp/club-3090/blob/32263ba51cf421c2e4785f200654d160af143b91/.research/gguf-tp-q4-kv/REPORT.md)
- [SM86 decode context parallelism](DCP-SM86.md)
- [Capacity accounting](CAPACITY.md)
- [Cross-engine numerical oracle](M6-LAYER-ORACLE-SPEC.md)

## Lower-level implementation evidence

- [GGUF format contract](FORMAT-CONTRACT.md)
- [Tensor-parallel mapping](TP-MAPPING.md)
- [Runtime dtype contracts](DTYPE-CONTRACTS.md)
- [Q8_0 `wo_a` design](WOA-DESIGN.md)
- [Loader report](M4-LOADER.md)
- [Grouped expert prefill](M2-GROUPED-PREFILL.md)
- [Layer-slice performance](M2-LAYER-SLICE.md)
- [Fusion trace and no-go decision](FUSION-TRACE.md)
- [Cold-expert route study](route-offload/ROUTE-OFFLOAD.md)
- [SM86 DCP import audit](SM86-DCP-IMPORT-AUDIT.md)

The evidence directories carry `SHA256SUMS` where raw logs were preserved. Historical runtime numbers describe the exact recorded profile, not the current server state.
