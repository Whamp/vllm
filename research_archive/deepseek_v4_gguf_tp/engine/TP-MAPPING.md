# TP-MAPPING.md — GGUF tensor → vLLM destination → TP rule (§4.7 table, M1)

Derived from pinned Whamp/vllm tree `6354125a` (branch incubate/gguf-tp-sm86,
base b7766cfe) and the verified GGUF inventory
(`evidence/gguf-inventory.json`, blob `ca22ae2f…`). Byte totals are raw-GGUF
bytes; the Q8_0→int8-g32 repack is size-neutral by construction (34 B/32 w →
32 B codes + 2 B fp16 scale per 32). Every rule cites its constructor.

## 1. Per-family mapping

| GGUF tensor (per layer l) | dims (ne0..) | vLLM destination | Constructor (file:line @ 6354125a) | TP rule @ TP=4 |
|---|---|---|---|---|
| `blk.l.attn_q_a.weight` | [4096,1024] | `attn.fused_wqa_wkv` slot 0 | MergedColumnParallelLinear, `disable_tp=True` (attention.py:265-272) | **replicated** |
| `blk.l.attn_kv.weight` | [4096,512] | `attn.fused_wqa_wkv` slot 1 (stacked after wq_a) | same | **replicated** |
| `blk.l.attn_q_b.weight` | [1024,32768] | `attn.wq_b` | ColumnParallelLinear (attention.py:274-282) | shard ÷4 (heads) |
| `blk.l.attn_output_a.weight` | [4096,8192] | `attn.wo_a` | ColumnParallelLinear, `is_bmm` groups (attention.py:284-293) | shard ÷4 (groups) |
| `blk.l.attn_output_b.weight` | [8192,4096] | `attn.wo_b` | RowParallelLinear (attention.py:298+) | shard ÷4 (rows) |
| `blk.l.attn_k_norm.weight` | F32 [512] | `attn.kv_norm` RMSNorm | — | replicated |
| `blk.l.attn_compressor_kv/gate.weight` | [4096,coff·512]×2 | `compressor.fused_wkv_wgate` slots wkv=0, wgate=1 | MergedColumnParallelLinear, `disable_tp=True` (compressor.py:285-294) | **replicated** |
| `blk.l.attn_compressor_ape.weight` | [coff·512,4] | compressor APE param | — | replicated |
| `blk.l.attn_compressor_norm.weight` | F32 [512] | compressor RMSNorm | — | replicated |
| `blk.l.indexer.attn_q_b.weight` | [1024,8192] | `indexer.wq_b` | **ReplicatedLinear** (attention.py:949-955) | **replicated** |
| `blk.l.indexer.proj.weight` | [4096,64] | `indexer.weights_proj` | **ReplicatedLinear** (attention.py:956+) | **replicated** |
| `blk.l.indexer_compressor_*` | as compressor ×coff 128 | `indexer.compressor.*` | same compressor constructors | **replicated** |
| `blk.l.ffn_gate_exps.weight` | [4096,2048,256] IQ2_XXS | routed expert w13 gate slot | FusedMoEConfig computes `intermediate_size_per_partition = intermediate_size / tp_size` (fused_moe/config.py:1333); WNA16 shape uses it for every expert (compressed_tensors_moe_wna16.py:173-238) | **all 256 experts; N-shard 2048/4=512 rows/rank** |
| `blk.l.ffn_up_exps.weight` | [4096,2048,256] IQ2_XXS | routed expert w13 up slot | same | all experts; N=512/rank |
| `blk.l.ffn_down_exps.weight` | [2048,4096,256] Q2_K (**K/N swapped vs gate/up**) | routed expert w2 | same | **all experts; K-shard 2048/4=512/rank**, partial N4096 outputs reduced across TP |
| `blk.l.ffn_gate_shexp/up_shexp.weight` | [4096,2048] | `shared_experts.gate_up_proj` | MergedColumnParallelLinear (model.py:119) | shard ÷4 |
| `blk.l.ffn_down_shexp.weight` | [2048,4096] | `shared_experts.down_proj` | RowParallelLinear (model.py:127) | shard ÷4 |
| `blk.l.ffn_gate_inp.weight` | [4096,256] F16 | `moe.gate` (fp32 out) | GateLinear `out_dtype=float32` (model.py:564-571) | **replicated** (cast policy §4.5: keep fp16/fp32 math) |
| `blk.l.ffn_gate_tid2eid.weight` | I32 [6,129280] (layers 0-2 only) | `gate.tid2eid` hash table | nn param (model.py:573+) | **replicated** |
| `blk.l.exp_probs_b.bias` | F32 [256] | `gate.e_score_correction_bias` | — | replicated |
| `blk.l.hc_attn_fn/hc_ffn_fn.weight` | F16 [16384,24]/[16384,4-head] | `hc_*_fn` params (model.py:878-914, 1139-1147) | nn.Parameter | replicated (MHC broadcast) |
| `blk.l.hc_*_base/scale.weight` | F32 | `hc_*_base/scale` | nn.Parameter | replicated |
| `blk.l.ffn_norm.weight`, `output_norm.weight` | F32 [4096] | RMSNorm | — | replicated |
| `token_embd.weight` | F16 [4096,129280] | `model.embed_tokens` | VocabParallelEmbedding (model.py:1113) | **vocab-shard ÷4** |
| `output.weight` | Q8_0 [4096,129280] | `lm_head` | ParallelLMHead (model.py:1551) | **vocab-shard ÷4** (last PP rank) |

Note: no MTP/nextn tensors exist in this GGUF (metadata
`nextn_predict_layers=1` only) — consistent with the separate-MTP-file
contract; nothing to load.

## 2. Per-rank weight arithmetic (TP=4, raw-byte basis)

| Family | Total GiB | Rule | Per-rank GiB |
|---|---:|---|---:|
| routed experts | 72.5625 | ÷4 intermediate dimension (all experts present) | 18.1406 |
| attention sharded (wq_b, wo_a, wo_b) | 4.2832 | ÷4 | 1.0708 |
| attention replicated (fused_wqa_wkv) | 0.2677 | ×1 | 0.2677 |
| shared experts | 1.0708 | ÷4 | 0.2677 |
| token_embd | 0.9863 | ÷4 vocab | 0.2466 |
| output head | 0.5240 | ÷4 vocab | 0.1310 |
| indexer+compressor | 0.9075 | ×1 | 0.9075 |
| router (gate/bias/tid2eid) | 0.0927 | ×1 | 0.0927 |
| hyperconnection | 0.0631 | ×1 | 0.0631 |
| norms | 0.0016 | ×1 | 0.0016 |
| **Total per rank** | 80.7594 | | **21.1893** |

Replicated per rank: 1.3326 GiB. Sharded per rank: 19.8567 GiB.

## 3. Capacity anchor

- Measured WNA16-quality anchor: 78.739 GiB artifact → 20.69 GiB registered
  per rank at 156K warm readiness (residency audit); 230,144 ctx with
  1.28 GiB available KV at the promoted profile.
- This GGUF: +2.02 GiB total, **+0.50 GiB per rank** over WNA16-quality
  (21.189 vs 20.69) → projected KV headroom ≈ 1.28 − 0.50 = 0.78 GiB
  → ≈ 0.78 GiB ÷ 5.832 KiB/token/rank ≈ **≈141.9K tokens**, consistent with
  PLAN §10's 139.1K point estimate (which also budgets the graph-pool delta).
- Deltas not yet in this number (M1 capacity-table remainder): Marlin tile
  padding on int8-g32 dense paths, aligned-SoA repack (byte-neutral by
  construction), graph pool (~0.19 GiB measured on WNA16), activation/
  workspace/JIT pools (WNA16-measured ~0.53 GiB non-Torch + allocator
  cache), 32-byte loader alignment (~1.7 KiB total — negligible).

## 4. Loader implications (feeding M4)

- Fused-slot order is load-order significant: `fused_wqa_wkv` = [wq_a ‖ wkv]
  columns; `compressor.fused_wkv_wgate` = [wkv ‖ wgate] — the class-A2
  coordinate oracle must sample across each slot boundary.
- Routed experts use **within-expert TP**, not 64 whole experts/rank. Gate/up:
  every rank loads a contiguous 512-row N slice for every expert (GGUF ne0=K,
  so each rank slice is contiguous inside one expert). Down: every rank loads
  a K slice of 512 values inside every N=4096 row (two Q2_K blocks/row), so
  loader extraction is strided across output rows and must be class-A2 checked
  at first/last block for every rank.
- Vocab shards: token_embd/output are [4096, 129280] with ne0=4096 — the
  vocab axis is the SLOW axis; a rank's vocab slice is a contiguous range of
  32,320 rows. tid2eid [6, 129280] replicates whole.
- Q8_0 → int8-g32 repack applies to: attention 5/layer + shared 3/layer +
  output head. Q8_0 rows are 34·ceil(ne0/32) bytes; ne0 ∈ {4096, 1024, 512,
  8192? (wo_b rows are 8192 columns → ne0=8192)} — all divisible by 32, so
  no partial blocks in this artifact (fail-closed check belongs in the
  repacker).
