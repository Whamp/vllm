<!-- markdownlint-disable MD060 -->

# DTYPE-CONTRACTS.md — per-family runtime dtype requirements (M1)

Pinned tree `6354125a` (branch incubate/gguf-tp-sm86). Model dtype: bfloat16
(the stack's serving dtype). GGUF storage per the verified inventory.

| Family | GGUF storage | Runtime requirement (citation) | Contract |
|---|---|---|---|
| Attention wq_b/wo_a/wo_b | Q8_0 | bf16 activations; weights via quant method (int8-g32 repack) | repack; class-B window vs dequant+GEMM |
| Attention fused_wqa_wkv (wq_a+wkv slots) | Q8_0 | `disable_tp=True` ReplicatedLinear (attention.py:265-272) | repack; replicated |
| wo_a runtime cache | — | fast path requires **bf16** weights or the dequant cache path is used (attention.py:443-447 requires `dtype == torch.bfloat16` for merged/indexer GEMMs) | see WOA-DESIGN.md |
| Compressor fused_wkv_wgate, ape | F16 / F32 | `quant_config=None`, `disable_tp=True` (compressor.py:285-294); **merged-GEMM fast path requires bf16 weights** (attention.py:443-447); compressor state/cache and scratch are **fp32** (`assert self.dtype == torch.float32` compressor.py:174; state_dtype fp32 :275; scratch fp32 :271) | cast F16→bf16 at load; **lossy (10→7 mantissa bits)**; class-B window; fp32-upcast fallback option if M6 divergence |
| Indexer wq_b, weights_proj | F16 | ReplicatedLinear `quant_config=None` (attention.py:949-957); bf16 required for the merged fast path (attention.py:443-447) | cast F16→bf16; lossy; class-B window; fp32-upcast fallback option (affects top-k selection → treat as sensitive) |
| Indexer/compressor norm | F32 | RMSNorm fp32-friendly | load as fp32, no cast |
| Router ffn_gate_inp | F16 | GateLinear `out_dtype=float32` (model.py:564-571); weight follows model dtype | cast F16→bf16; lossy in top-6 tie-breaks; **fp32 weight upcast is the default if the GEMM accepts it** (PLAN §4.5) — decide at M2 kernel bring-up; 4 MiB/layer cost is negligible |
| exp_probs_b bias | F32 | fp32 param | none |
| tid2eid | I32 | int hash table (model.py:573+) | none; replicated |
| Hyperconnection hc_fn | F16 [16384,24/4] | nn.Parameter **fp32** (model.py:878-917, 1139-1147) | F16→fp32 upcast: **lossless** |
| hc base/scale | F32 | fp32 params | none |
| Norms | F32 | RMSNorm | none |
| token_embd | F16 [4096,129280] | VocabParallelEmbedding, model dtype bf16 | F16→bf16 cast at load (lossy) or fp16 table with row-cast on lookup (same loss); class-B window; embedding lookup fidelity affects every token |
| output head | Q8_0 | ParallelLMHead via quant method | int8-g32 repack; class-B window |
| Routed experts | IQ2_XXS/Q2_K | new kernels consume packed bytes directly; bf16 activations in, accum fp32 | **no cast — bit-exact bytes** (class-A/A2) |

## Standing rule

Every F16→bf16 cast above is a *documented lossy conversion* with a
pre-registered class-B tolerance window measured at M6 (per-layer forward vs
llama.cpp on fixed prompts). If any window fails, the recorded fallback is
an fp32 (or fp16-native) upcast for that family with its capacity cost —
not a silent acceptance. No other implicit casts are permitted; the loader
fail-closes on unlisted dtype combinations.
