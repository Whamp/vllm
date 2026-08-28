# M4 — native GGUF loader and runtime ownership

Decision: **M4 passes.** Proceed to M5 guarded TP4 full-model bring-up. The 10-working-day kill was not approached.

## Delivered runtime contract

Whamp/vLLM `incubate/gguf-tp-sm86` through `741b3abfb` now provides:

- load format and quantization method `gguf_dsv4`;
- required exact GGUF path, SHA-256, file size, and tensor count;
- bounded 16 MiB GGUF-v3 header parser with duplicate/type/range/overlap checks;
- rank-0 full-file SHA verification broadcast and successful-result caching;
- O(tensor) TP coordinate plans using contiguous or counted-strided spans;
- bounded `pread` streaming, vectorized strided gathering, raw quant-byte copy, and ordinary dtype casts;
- Q8 raw-row ownership followed by byte-neutral INT8/group-32 Marlin preparation;
- all-256-expert IQ2 gate/up and Q2 down parameters with within-expert TP sharding;
- indexed M<128 and grouped-MMA M≥128 runtime dispatch;
- weighted clamped-SwiGLU→Q8 fusion;
- quantized `ParallelLMHead` and production `wo_a` diagonal compatibility.

No generic GGUF support, model compatibility layer, or llama.cpp linkage was introduced.

## Full-inventory and coordinate evidence

The verified 1,328-entry inventory (`1cadb51c…`) plans on every TP rank as:

- 1,328 source plans;
- 1,180 runtime targets;
- 1,328 span descriptors;
- **22,751,844,636 bytes = 21.1893065 GiB/rank**;
- no target overlap.

The first implementation represented every down row as one Python object (~45 million descriptors across the model). The final counted-strided representation is O(1) descriptors/tensor while preserving source offset, target offset, width, count, and both strides.

A durable CPU/meta verifier constructs the complete Ampere DeepSeek V4 graph without GPU allocation and checks plans against real named parameters. Its first run found and corrected two defects:

1. actual routed parameters live below `.experts.routed_experts`;
2. `ParallelLMHead` is an embedding subclass and needed explicit quant-method selection.

Final TP4 torchrun/Gloo result, on all ranks independently:

- 1,328 inventory tensors;
- 1,180 planned target names;
- 1,180 actual named parameters;
- exact name-set equality;
- exact element-count equality.

See `evidence/m4-loader/meta-model-tp4-report.json`.

## Byte identity, repack, and lifecycle

- IQ2/Q2 rank spans copy raw bytes only. Synthetic strided/contiguous IO tests verify source/target coordinates; M1 L0 proves source decode; TP4 full-plan spans cover every expert and first/last rank block.
- No aligned IQ2/Q2 artifact is produced: aligned IQ2 lost at exact shape and Q2 was declined by causal budget. Therefore there is no derived low-bit repack hash to maintain; the immutable source hash remains the identity gate.
- Q8 changes representation only after loading: signed code preservation, group-scale conversion, prepared storage byte equality, numerical windows, graph replay, and exact method lifecycle all pass. The final RTX 3090 method test replaces `weight_raw`, retains byte-neutral weight+scale storage, and executes against the dequantized reference.

## Validation

- Loader/planner/IO/allocation suite: 11 focused CPU tests.
- Final Q8 lifecycle file: 11 RTX 3090 tests; canonical service restored healthy and zero-swap.
- Full target verifier: TP1 and TP4 meta passes.
- Pre-commit Ruff/format/mypy/forbidden-import/config checks pass.
- CodeGraph cycles/signatures pass. Its only earlier boundary warning was the pre-existing `engine/arg_utils → config/load` edge surfaced by documenting the new format.
- aislop’s new-module complexity findings were resolved. Remaining dependency-manifest errors are false positives for established vLLM Torch/NumPy/Pydantic/regex imports; large DeepSeek model warnings are pre-existing.

## M5 risks

M4 proves mapping, allocation, and local lifecycle—not full 86.7 GB transfer, post-load aggregate residency, API readiness, or task correctness. M5 must still enforce the 22.78 GiB/rank pre-KV falsifier, verify runtime dispatch, and preserve rollback.
