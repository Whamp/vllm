# Qwen3.8-Flash-Next on vLLM and RTX 3090

Date: 2026-08-28

## Decision

Start with an existing W4A16 safetensors backbone, not a native GGUF loader.

The best-supported server60 candidate combines:

1. [`aixiaoma/Qwen3.8-Flash-Next-W4A16`](https://huggingface.co/aixiaoma/Qwen3.8-Flash-Next-W4A16) or its byte-identical [`VnimanieAI`](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16) copy for the GPU backbone.
2. A quantized PLE table from [`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) for host-memory lookup.

The W4A16 backbone has a 66.34 GiB text payload after excluding PLE, MTP, and vision weights. Its author reports successful serving on four RTX 3090s through vLLM's INT4 Marlin path. The checkpoint's published 95.43 GiB BF16 PLE cannot fit server60, but Primitive publishes 32 GB INT4 and 28.8 GB software-NVFP4 PLE sidecars with a vLLM offload implementation.

Use the INT4 PLE first. It has plain signed integer dequantization, a measured 32.9 GB host RSS, and a reported successful run inside a 48 GB container. Test the 28.8 GB software-NVFP4 table if server60 needs the extra 3.2 GB of host margin. Neither PLE path needs native FP4 tensor-core support because the CPU worker dequantizes only selected rows before transfer.

Keep the current UD-Q4_K_XL GGUF service as the quality and performance control. Do not build a new GGUF loader unless the safetensors path fails a named gate.

## Server60 requirements

The deployment must preserve these constraints:

- the full PLE table remains outside GPU VRAM;
- the resident PLE representation remains quantized;
- no process swaps;
- tensor parallelism uses all four RTX 3090s;
- concurrency 2 must work, with concurrency 4 as the target;
- the existing 230 W GPU safety policy remains unchanged;
- the current ik_llama service remains available for rollback and matched comparison.

## Model and live reference

The official config at revision [`de4b8e4d`](https://huggingface.co/Qwen/Qwen3.8-Flash-Next/blob/de4b8e4d43b917e7706784d8bb445c9af86a3540/config.json) defines:

- 48 layers;
- hidden size 2,560;
- 36 Gated DeltaNet layers and 12 Qwen Sparse Attention layers;
- 512 routed experts, top 10, intermediate size 640;
- one shared expert per layer;
- four residual streams with rank-320 gated residual projections;
- one layer-2 PLE table with 16 n-gram heads of width 160;
- native context 262,144.

The [official report](https://github.com/QwenLM/Qwen3.8-Flash-Next/blob/69885871a64393807d988b27b1b5e380e8f28526/tech_report.pdf) describes the architecture and PLE offloading intent.

The live server60 control is the exact [`unsloth/UD-Q4_K_XL`](https://huggingface.co/unsloth/Qwen3.8-Flash-Next-GGUF/tree/c8b5954a88c2775c546b92593eda40ea041d3176/UD-Q4_K_XL) artifact on ik_llama.cpp commit [`15dddc60`](https://github.com/ikawrakow/ik_llama.cpp/commit/15dddc60b3fc937a9e2a210359ecce392ccdf446). Its relevant settings are:

- four-GPU layer split `1.01,0.97,0.99,1.03`;
- `per_layer_token_embd=CPU`;
- 400,000 configured context;
- two slots;
- Q8_0 K and V cache;
- no speculative decoder.

The inspected process used 37.9 GB RSS with zero process swap and 22.3 to 23.5 GiB per GPU. Recent logs showed about 31 decode tokens/s and 350 to 450 prefill tokens/s. Those log samples are not a matched benchmark.

## Safetensors candidate audit

All sizes below come from immutable Hugging Face revisions and safetensors headers. Repository totals include small metadata files. Tensor payload figures exclude headers.

### Recommended backbone: aixiaoma or Vnimanie W4A16

Pinned revisions:

- `aixiaoma/Qwen3.8-Flash-Next-W4A16` at `75234a1d675cc7dd70569689872feb3d8aa1aca4`;
- `VnimanieAI/Qwen3.8-Flash-Next-W4A16` at `9236d703b25f25eb5c17e9640204f84fa1ce0c6e`.

The 38 safetensors and JSON objects common to both repositories have matching sizes and object hashes.

Exact tensor payload:

| Component | GiB |
| --- | ---: |
| Complete checkpoint | 167.458 |
| BF16 PLE | 95.429 |
| MTP | 4.856 |
| Vision | 0.836 |
| Text backbone without PLE, MTP, or vision | 66.337 |
| Even TP=4 share of that text payload | 16.584 |

The checkpoint uses compressed-tensors symmetric INT4 group-128 for routed experts and selected attention projections. It preserves shared experts, GDN, routers, indexers, residual projections, embeddings, and output in BF16.

The author reports:

- 4×RTX 3090 serving at 98,304 context with expert parallelism;
- 8×RTX 3090 serving at 262,144 context;
- 67 decode tokens/s without MTP on the tested 8×RTX 3090 system;
- 100 to 114 decode tokens/s with MTP;
- decode-only CUDA graphs because Inductor hung on the tested Ampere system.

These are publisher results, not server60 measurements. The card does not pin a runtime image digest. Reproduction needs an exact source and image contract.

The published checkpoint cannot run on server60 unchanged because its BF16 PLE requires at least 110 GB free host RAM. Replacing the PLE is mandatory.

### Second candidate: local-inference-lab mixed checkpoint

[`local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89`](https://huggingface.co/local-inference-lab/Qwen3.8-Flash-Next-NVFP4-4p89) at `b184bb5650367c3e934c7849407be9da3671e7f5` is the smallest complete safetensors candidate found.

Exact tensor payload:

| Component | GiB |
| --- | ---: |
| Complete checkpoint | 98.533 |
| Quantized PLE | 26.883 |
| Non-PLE payload | 71.650 |
| Text backbone without PLE, MTP, or vision | about 69.8 |

Its PLE stores packed 4-bit values plus group-16 E4M3 scales. The backbone uses ModelOpt mixed precision:

- MXFP8 for attention, shared experts, and vision;
- NVFP4 for routed experts;
- W4A16 NVFP4 for MTP routed experts.

Current vLLM source has Marlin and Humming implementations that declare SM75+ support for NVFP4 and MXFP8. Marlin provides a weight-only fallback on GPUs without native FP4. This establishes source-level eligibility, not packaged dispatch or performance on RTX 3090.

The repository card says only `WIP`. It has no serving command, quality results, runtime pin, or hardware validation. Use it only after the W4A16 path, or as a bounded loader and kernel experiment.

### RadixArk NVFP4

[`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4) at `7b719225242aacd3dbd3f9407468c2ee9a9d2594` is 135.25 GB on disk.

It quantizes only routed experts to ModelOpt NVFP4 W4A4. Attention, GDN, residual projections, shared experts, routers, vision, and MTP remain BF16. Its 51.2 GB FP8 PLE is dequantized to BF16 at load time. That load contract does not fit server60.

The publisher validated SGLang only on Blackwell. Current vLLM may route its NVFP4 experts through Marlin or Humming on SM86, but the exact Qwen checkpoint has no such validation. It is a useful source checkpoint for PLE sidecars, not the first server60 backbone.

### Inferact NVFP4

[`Inferact/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/Inferact/Qwen3.8-Flash-Next-NVFP4) at `103a7608316173ca6edd49929544244de7ffda70` is 182.84 GB on disk. Safetensors metadata shows a 102.40 GB BF16 PLE shard. It fails the host-RAM gate unchanged.

### Other complete checkpoints

- `lovedheart/Qwen3.8-Flash-Next-NVFP4-FP8` is 132.58 GB and retains the same 51.2 GB FP8 PLE family. Its card requires Blackwell.
- `primitive-ai/Qwen3.8-Flash-Next-mixed-NVFP4-FP8` is 183.73 GB and retains a roughly 95 GB BF16 PLE unless an external sidecar is used. Its card requires about 100 GB free host RAM without that sidecar.
- `axiomofmind/Qwen3.8-Flash-Next-W4A16-NVFP4` is 186.44 GB. It offers no server60 advantage over the smaller, Ampere-tested INT4 checkpoint.
- The official FP8 checkpoint and other BF16 or FP8 derivatives are too large or lack an SM86 execution path.

## Quantized PLE options

### Primitive vLLM sidecars

[`primitive-ai/Qwen3.8-Flash-Next-PLE-quant`](https://huggingface.co/primitive-ai/Qwen3.8-Flash-Next-PLE-quant) at `da8b39586016d8325ac619be28ad77d6296625ec` publishes three PLE formats derived from the original BF16 table.

| Format | Stored size | Measured host RSS | Single-stream result on publisher system |
| --- | ---: | ---: | ---: |
| FP8 per row | 49 GB | 52.6 GB | 80.1 to 80.3 tokens/s |
| INT4 group-16 | 32 GB | 32.9 GB | 80.1 to 80.2 tokens/s |
| software NVFP4 group-16 | 28.8 GB | 29.8 GB | 80.1 to 80.3 tokens/s |
| BF16 baseline | 95.4 GB | about 95 GB | 84.4 to 84.5 tokens/s |

The publisher measured the INT4 table inside a 48 GB container and states that a 64 GB RAM host can serve it. Server60 has 60 GiB, about 64.4 decimal GB, so it meets that reported capacity narrowly. Zero-swap validation remains mandatory.

The implementation memory-maps 128 table shards and dequantizes only selected rows. The INT4 path performs nibble unpacking and FP16 group-scale multiplication on CPU tensors. The software-NVFP4 path uses an E2M1 lookup and E4M3 scales in software. Neither path depends on SM100 FP4 instructions.

Publisher results on one RTX PRO 6000 Blackwell showed:

- about 5% lower single-stream throughput than the BF16 in-RAM table;
- knowledge scores in the same measured band;
- tool-calling scores within the reported repeat spread;
- working MTP composition;
- page-cache-backed mapped files rather than an anonymous full-table allocation.

Those results support a server60 experiment. They do not prove Threadripper 2950X lookup latency or 4×RTX 3090 overlap.

### Lewfkrad SGLang sidecar

[`Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE`](https://huggingface.co/Lewfkrad/Qwen3.8-Flash-Next-NVFP4-W4-PLE) at `8bf4dd3779b15732b303c0931e64961a332a0c78` contains:

- 25,600,122,880 bytes of packed signed INT4 values;
- 3,200,015,360 bytes of E4M3 group scales;
- a strict source and layout manifest.

It is 26.822 GiB and source-binds RadixArk revision `7b719225`. Its runtime streams the files into pinned CPU tensors, gathers rows through UVA, and dequantizes on GPU. The author qualified only TP=1 on an SM120 RTX PRO 6000.

This is less direct for vLLM and server60 than Primitive's existing vLLM overlay. Preserve its manifest discipline and representation as references.

## Fit estimate for the recommended path

### Host RAM

Primitive measured 32.9 GB host RSS for the INT4 PLE and 29.8 GB for software NVFP4. The current ik_llama process uses 37.9 GB RSS with a 26.82 GiB IQ4_NL PLE and its runtime state.

A single-copy quantized PLE plus vLLM worker overhead should fit 60 GiB, but the margin is an instance value. The preflight must reject launch unless it leaves an 8 GiB host reserve and can normalize swap to zero.

Do not page-lock the whole table. Pin only transfer buffers. Keep the table memory-mapped and let the page cache reclaim cold rows.

### GPU VRAM

The first-order W4A16 text-weight floor is 16.584 GiB per rank. Expert parallelism changes ownership, so exact per-rank residency requires a meta planner and runtime measurement.

The current upstream QSA path uses BF16 K and V. At TP=4, each rank holds one KV head across 12 QSA layers:

```text
12 layers x K and V x 256 values x 2 bytes = 12,288 bytes/token/rank
```

| Context | BF16 QSA K/V per rank |
| ---: | ---: |
| 96K | 1.125 GiB |
| 128K | 1.500 GiB |
| 148K | 1.734 GiB |
| 262,144 | 3.000 GiB |

The sparse indexer adds about 768 bytes per token per rank. Gated DeltaNet temporal state costs about 27 MiB per active request across its 36 layers, before smaller recurrent buffers.

Start at 96K, the published 4×3090 profile. Search toward 128K or 148K only after exact model, graph, recurrent-state, and KV residency is measured. Quantized QSA KV is a later capacity change.

## Existing vLLM work to reuse

The model work remains upstream and unmerged:

- [PR #53896](https://github.com/vllm-project/vllm/pull/53896), Qwen3.8-Flash-Next model support;
- [PR #53899](https://github.com/vllm-project/vllm/pull/53899), PLE CPU offload;
- [PR #53909](https://github.com/vllm-project/vllm/pull/53909), QSA, gated residual, and PLE state kernels;
- [PR #54129](https://github.com/vllm-project/vllm/pull/54129), disk-backed PLE experiments;
- [PR #54070](https://github.com/vllm-project/vllm/pull/54070), BF16 disk PLE loading.

As inspected, the model and PLE PR branches conflict with current upstream main. The PLE worker accepts the standard checkpoint loader and the dummy loader, not a custom GGUF loader. It owns one PLE model per data-parallel group and asynchronously copies results to TP-rank output buffers.

Primitive's two-file overlay adds quantized, memory-mapped PLE tables. It is currently a vendored model-file replacement rather than an upstream vLLM PR.

Whamp/vLLM already has reusable SM86 machinery:

- compressed-tensors W4A16 and Marlin;
- NVFP4 Marlin and Humming eligibility on SM75+;
- native GGUF split identity and bounded loading, if later needed;
- SM86 CUDA graph, sanitizer, and cubin tests;
- hierarchical all-reduce for server60's PCIe topology.

## Recommended implementation sequence

### Phase 1: reproduce the published Ampere backbone

Semantically port the Qwen4Exp model and PLE work onto current Whamp/vLLM. Load the aixiaoma/Vnimanie W4A16 checkpoint on a CPU/meta model first. Verify exact parameter names, compressed-tensors target resolution, expert-parallel ownership, and the PLE exclusion.

Do not involve the PLE table yet. A dummy PLE result is enough to prove model construction and GPU-weight fit.

### Phase 2: integrate one quantized PLE

Start with Primitive INT4 group-16. Preserve its 128-shard order, exact row count, nibble layout, group scales, source identity, and mmap behavior.

Required tests:

- row IDs match the official and ik_llama behavior;
- decoded rows match an independent INT4 oracle;
- batched requests preserve separate token histories;
- prefix-cache resume preserves PLE and recurrent state;
- one PLE copy exists on the host;
- PLE compute and transfer overlap with layers 0 and 1;
- no process swaps.

If host reserve is too small, test the 28.8 GB software-NVFP4 table with the same gates.

### Phase 3: bounded server60 launch

Start with:

- TP=4 and expert parallelism;
- max_num_seqs 2;
- max_num_batched_tokens 256;
- 96K context;
- BF16 QSA KV;
- decode-only CUDA graphs;
- MTP disabled;
- prefix caching disabled until growing-conversation state tests pass;
- a non-restarting candidate and exact rollback timer.

### Phase 4: correctness and performance

Require:

- deterministic token parity against ik_llama for a fixed rendered prompt;
- layer or final-logit comparison before accepting large output drift;
- reasoning, tool, streaming, and post-tool continuation checks;
- exact NIAH near the configured ceiling;
- matched warm single-stream decode;
- matched cache-busted prefill;
- concurrency 2 and 4 throughput and latency;
- no VRAM growth and zero swap.

Only after that should MTP, longer context, quantized QSA KV, or the local-inference-lab checkpoint enter the experiment queue.

## Stop conditions

Stop or redesign if:

- the W4A16 checkpoint cannot reproduce its published SM86 Marlin dispatch;
- exact TP=4 weight residency leaves too little room for a useful 96K cache;
- the quantized PLE serializes decode on the Threadripper host;
- PLE state diverges under concurrency or prefix-cache resume;
- QSA or GDN falls back to CPU or repeated host synchronization;
- quality fails before any KV-cache precision change;
- the path does not materially beat the current ik_llama service in aggregate throughput.

## Next action

Build the CPU/meta integration on current Whamp/vLLM using the W4A16 checkpoint and Primitive INT4 PLE contract. Produce an exact 4×3090 residency plan and a source-pinned, non-restarting server60 launch plan before touching the live service.
