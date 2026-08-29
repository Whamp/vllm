# Qwen3.8 Flash Next INT4 MoE: tensor parallelism versus expert parallelism

## Decision

Do not write a new TP-sharded expert kernel yet.

On the exact four-GPU server60 deployment, `--enable-expert-parallel` is not
using a token all-to-all backend. It changes expert weight ownership from
intra-expert TP to whole-expert placement, computes only the 128 local experts
on each rank, and performs a final hidden-state all-reduce. Plain TP also needs
a final hidden-state all-reduce. The measured EP path is both faster and smaller.

The no-EP fallback exposes a real vLLM deficiency, but it is not evidence that
EP wastes VRAM:

- AutoRound group-128 Marlin rejects the rank-local expert width of 160.
- vLLM falls back to Triton WNA16.
- The fallback changes the effective group size from 128 to 32 and repeats every
  scale four times.
- That scale expansion predicts 1.318 GiB of extra scale storage per rank across
  48 layers. The measured model-load increase was 1.21 GiB per rank.

A custom TP kernel could remove the fallback penalty. At best it would first
recover the EP path's expert storage and speed. It would not reduce the packed
expert weights below EP's current per-rank allocation. Further work needs to
measure the remaining EP residency and its critical path before choosing a new
kernel.

## Scope and evidence versions

The tested artifact and runtime were:

- Intel checkpoint revision
  `861536dda5bcb208376fc4cd879b2bf76bece9fe`.
- Text and vision payload after omitting PLE and MTP:
  73,581,002,608 file bytes, reported by vLLM as a 68.53 GiB checkpoint.
- Primitive quantized PLE revision
  `da8b39586016d8325ac619be28ad77d6296625ec`.
- vLLM image
  `vllm/vllm-openai@sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3`.
- The image reports vLLM `0.1.dev20073+g8e685d198`, but that commit is not
  published upstream.
- The image's MoE parallel config, no-DP/EP prepare/finalize, MoE runner,
  expert-map manager, generic WNA16 loader, INC resolver, Marlin checks, and
  WNA16 oracle files match upstream PR #53899 head
  [`a5530b90cab09b187463396a99612a486ba91d6f`](https://github.com/vllm-project/vllm/tree/a5530b90cab09b187463396a99612a486ba91d6f)
  byte-for-byte. The Qwen model-state file differs only by the class/module
  rename from `Qwen3_8FlashNext` to `Qwen4Exp`.
- Current upstream main inspected for later fixes:
  [`4c6c9d569f83d4dc5509e77e3458734a1fda2c2d`](https://github.com/vllm-project/vllm/tree/4c6c9d569f83d4dc5509e77e3458734a1fda2c2d).

This report separates source facts, server60 measurements, estimates, and open
questions. It makes no performance claim for an unimplemented kernel.

## Measured server60 A/B

Both arms used the Intel artifact, Primitive PLE, vision tower, TP=4,
`max_num_seqs=2`, automatic context fitting, `gpu_memory_utilization=0.95`,
FULL_DECODE_ONLY CUDA graphs, no MTP, zero host swap, and the fixed 230 W GPU
safety policy. The only runtime-semantic change was
`--enable-expert-parallel`.

| Configuration | Routed backend | Model load/rank | Runtime model + non-Torch/rank | KV pool | Auto-fit context | Matched wall decode |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| EP=4 | Marlin | 18.55 GiB | 19.51-19.55 GiB | 1.85-1.89 GiB | 148,000 | 39.94 tok/s |
| Plain TP=4 | Triton WNA16 | 19.76 GiB | 20.31-20.35 GiB | 1.04-1.09 GiB | 82,000 | 33.28 tok/s |

The decode discriminator used one warmup and three measured requests. Each
request generated 256 tokens at temperature 0.6 and top-p 0.95 from the same
short code prompt. This is enough to distinguish the 20% backend difference,
but it is not a promotion benchmark.

Both arms reached API readiness and returned the exact deterministic `PARIS`
canary. The EP arm also passed automatic tool calling, post-tool continuation,
and a synthetic-image vision canary.

The EP Compose SHA-256 was
`ba8dfa451a4f8dca55710c3ef2a1847cc5847ba8fadd83c722c290a3856216da`.
The no-EP Compose SHA-256 was
`3f6d14e67849de76c7be5b61d1ea55ba7ed17fb7d3e09265a46b1a26acaa4c27`.

## How vLLM maps TP and EP here

### Plain TP

Without EP, `FusedMoEParallelConfig.make` keeps the flattened tensor-parallel
size. `FusedMoEConfig` then divides the 640-wide intermediate dimension by four,
producing 160 columns per rank:

- [`FusedMoEParallelConfig.make`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/fused_moe/config.py#L1128-L1255)
- [`FusedMoEConfig.__post_init__`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/fused_moe/config.py#L1339-L1361)

Every rank owns all 512 expert identities, but only one quarter of each expert's
intermediate dimension. Every rank processes every selected expert and returns
a partial output. The outputs must be reduced across ranks.

### EP on this deployment

With EP enabled, the same function explicitly resets expert TP to one and turns
the four flattened TP ranks into four EP ranks. The source comment is direct:
"each device owns a set of experts fully. There is no tensor parallel."

`ExpertMapManager` assigns 512 / 4 = 128 complete experts to each rank and maps
all non-local expert IDs to `-1`:

- [`determine_expert_map`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/fused_moe/expert_map_manager.py#L22-L113)

The rank-local expert width stays 640. Group-128 quantization and Marlin's thread
tiles both fit exactly.

### Why this is not an all-to-all path

`use_all2all_kernels` is true only when EP is combined with DP, prefill-context
parallelism, or sequence parallelism. This deployment has DP=1, PCP=1, and SP=1:

- [`FusedMoEParallelConfig.use_all2all_kernels`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/fused_moe/config.py#L1051-L1059)

The modular-kernel factory therefore selects
`MoEPrepareAndFinalizeNoDPEPModular`, whose `prepare` function does no dispatch:

- [`maybe_make_prepare_finalize`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/fused_moe/all2all_utils.py#L118-L153)
- [`MoEPrepareAndFinalizeNoDPEPModular`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/fused_moe/prepare_finalize/no_dp_ep.py#L40-L97)

After each rank computes its local experts, `MoERunner` performs the late
hidden-state all-reduce whenever either TP or EP has more than one rank:

- [`MoERunner._maybe_reduce_final_output`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/fused_moe/runner/moe_runner.py#L455-L489)

Plain TP and this EP configuration therefore both reduce one hidden-state-sized
output per MoE layer. They distribute computation differently, but EP does not
add the DeepEP-style token dispatch that I previously assumed.

## Why plain TP expands storage

### Marlin rejection

The INC AutoRound resolver selects Marlin only when the rank-local shape passes
`check_moe_marlin_supports_layer`:

- [`_resolve_gptq_moe`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/quantization/inc/schemes/inc_wna16_scheme.py#L113-L157)

The merged Marlin padding work still rejects a quantization group that crosses a
TP boundary. Even with tile padding allowed, the rank-local intermediate must be
divisible by the checkpoint group size:

- [`check_moe_marlin_supports_config`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/quantization/utils/marlin_utils.py#L334-L396)

For Intel W4A16:

```text
moe_intermediate_size = 640
TP size               = 4
rank-local width       = 160
checkpoint group size  = 128
160 % 128              = 32
```

Marlin cannot preserve the checkpoint's five 128-value groups when equal TP
boundaries occur at 0, 160, 320, 480, and 640.

### Generic fallback behavior

`MoeWNA16Method.create_weights` repeatedly halves the effective group size until
both the hidden size and rank-local intermediate size divide evenly. For 160 it
chooses 32. The loader then repeats checkpoint scales four times:

- [`MoeWNA16Method.create_weights`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/quantization/moe_wna16.py#L243-L323)
- [`MoeWNA16Method.get_weight_loader`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/model_executor/layers/quantization/moe_wna16.py#L579-L635)

Repeating a group-128 scale over four group-32 windows preserves the dequantized
weight values. It also expands scale storage for both projections, even though
gate/up's K dimension is the 2,560-wide hidden size and never needed the smaller
group.

### Storage calculation

For 48 layers, BF16 scales, 512 experts, hidden size 2,560, and intermediate
size 640:

```text
EP, 128 experts/rank, N=640, group=128:
  w13 scales = 128 * 2*640 * (2560/128) * 2 bytes
  w2  scales = 128 * 2560  * (640/128)  * 2 bytes
  total over 48 layers = 0.439453 GiB/rank

Plain TP, 512 experts/rank, N=160, effective group=32:
  w13 scales = 512 * 2*160 * (2560/32) * 2 bytes
  w2  scales = 512 * 2560  * (160/32)  * 2 bytes
  total over 48 layers = 1.757813 GiB/rank

Predicted delta = 1.318359 GiB/rank
Measured model-load delta = 19.76 - 18.55 = 1.21 GiB/rank
```

The estimate and measurement are close enough to identify scale expansion as
the dominant no-EP residency penalty. Exact attribution still needs a
storage-deduplicated runtime parameter inventory.

## What the remaining EP memory means

The selected checkpoint files total 68.527649 GiB. An ideal byte-even quarter is
17.131912 GiB per rank. EP loads 18.55 GiB per rank, about 1.42 GiB above that
naive floor.

That 1.42 GiB is not explained by routed-expert scale expansion. EP keeps the
original group-128 scale count. Likely contributors include replicated routers,
norms, hyperconnection and recurrent control tensors, vision weights or
buffers, packed-layout metadata, and backend transformations. The current
research has not measured their exact shares.

The 0.99 GiB startup "peak activation" and 0.10 GiB CUDA-graph figures are
separate from model loading. Automatic KV fitting reserves against them. They
may offer more context capacity than expert-kernel work, especially because the
profile currently uses `max_num_batched_tokens=2048` rather than a
capacity-tuned value.

A model-memory inventory and a one-variable batched-token A/B should precede any
claim that the remaining VRAM is wasted.

## Where the apparent context capacity went

The intuitive calculation starts from about 68.53 GiB of selected checkpoint
files and four 24 GB cards. The runtime does not have the resulting difference
available for KV cache.

The live EP startup reported 23.56 GiB of usable device memory per rank and the
following exact profile:

| Allocation | Per rank | Four ranks |
| --- | ---: | ---: |
| Selected checkpoint file payload | 17.13 GiB if perfectly divided | 68.53 GiB |
| Loaded model | 18.55 GiB | 74.20 GiB |
| Loaded model plus non-Torch state | 19.51-19.55 GiB | about 78.1 GiB |
| Peak activation reservation | 0.99 GiB | 3.96 GiB |
| CUDA graphs | 0.10 GiB | 0.40 GiB |
| Five-percent utilization reserve | 1.18 GiB | 4.71 GiB |
| Actual KV pool | 1.85-1.89 GiB | about 7.5 GiB |

Starting with the measured 94.24 GiB visible across four GPUs, the checkpoint
appears to leave 25.71 GiB. Runtime model and post-load state consume about
9.51 GiB beyond the file payload. Activation profiling, CUDA graphs, and the
five-percent memory reserve consume another 9.07 GiB. That leaves about 7.1 GiB
inside the configured budget, close to the measured aggregate KV allocation.

### Decomposing the 9.51 GiB runtime expansion

The 15 selected safetensors files contain 68.499272 GiB of tensor payload plus
about 0.028 GiB of headers and alignment. A perfectly byte-even TP=4 partition
would therefore hold 17.124818 GiB of tensors per rank. vLLM's model-load
profiler measured 18.55 GiB per rank, an increase of about 1.425 GiB/rank or
5.70 GiB aggregate.

Most of that model-load increase follows directly from module ownership and
registered runtime buffers:

| Source-derived model-load increase | GiB/rank |
| --- | ---: |
| Replicated hyperconnection weights and norms | 0.895 |
| QSA top-k output buffers for 2,048 batched tokens | 0.188 |
| Replicated MoE routers | 0.088 |
| Replicated PLE key/value projections | 0.046 |
| Materialized 262K-position BF16 RoPE cache | 0.031 |
| Replicated QSA indexer projection and norms | 0.027 |
| Hyperconnection merged-linear alignment padding | 0.022 |
| Replicated QSA K/V projection heads at TP=4 | 0.015 |
| Replicated GDN a/b projections | 0.012 |
| Non-TP vision position/patch/bias/norm tensors | 0.007 |
| **Explained subtotal** | **1.331** |
| Remaining model-load difference | **about 0.094** |

The two largest items are easy to miss:

- Qwen's hyperconnection low-rank up/down projections are intentionally
  `ReplicatedLinear`; the combined down/injection projection is also
  `disable_tp=True` and pads its 324 logical rows to 336 physical rows.
- Every QSA layer registers an `[max_num_batched_tokens, indexer.output_width]`
  INT32 top-k buffer. With 12 QSA layers, 2,048 batched tokens, and output width
  2,051, those buffers occupy 0.187775 GiB/rank before any request runs.

The relevant owners are
[`Qwen4ExpHyperConnection`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/hyperconnection.py#L60-L125),
[`Qwen4ExpQSAAttention`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/qsa.py#L270-L318),
[`QSAIndexer`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/indexer_qsa.py#L100-L170),
and the replicated PLE projections in
[`Qwen4ExpPLELayer`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/ple_layer.py#L915-L950).

This leaves the difference from 18.55 GiB model loading to 19.51-19.55 GiB
"consumed memory". That roughly 0.96-1.00 GiB/rank is not a clean NCCL figure
and should not be named `non-Torch` without qualification:

- `DeviceMemoryProfiler` records the free-memory delta around model loading.
- vLLM initializes NCCL before its startup memory snapshot, so initial NCCL and
  CUDA-context state are excluded from the later consumed-memory delta.
- `memory_profiling` then measures all persistent consumption after model
  construction, including post-load Torch runner buffers, PLE IPC/output
  buffers, backend allocations invisible to the Torch allocator, and allocator
  rounding or retained storage.
- The startup log compresses that total into the label "weights + non-torch";
  it does not publish an operator-level breakdown.

The profiler definitions are in
[`MemorySnapshot` and `memory_profiling`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/utils/mem_utils.py#L108-L325),
and the Qwen image takes its baseline after distributed initialization in
[`GPUWorker.init_device`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/v1/worker/gpu_worker.py#L455-L515).

An exact split of the final roughly 1 GiB/rank requires opt-in staged runtime
instrumentation: storage-deduplicated named parameters and buffers after model
load, after model-state/PLE connector initialization, and after profile warmup,
paired with Torch allocator and device-free-memory counters. It cannot be
reconstructed exactly from the existing rounded startup log alone.

The KV pool is also not one aggregate cache striped over 96 GB. Qwen has 12 full
QSA layers, two KV heads, and head dimension 256, as pinned in the
[Intel config](https://huggingface.co/Intel/Qwen3.8-Flash-Next-W4A16-AutoRound/blob/861536dda5bcb208376fc4cd879b2bf76bece9fe/config.json).
With TP=4, vLLM cannot assign two whole KV heads evenly to four ranks. It sets
`num_kv_heads=max(1, 2//4)=1`, so every rank stores one BF16 K head and one BF16
V head for every full-attention layer:

- [`Qwen4ExpQSAAttention`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/qsa.py#L165-L215)

```text
Main QSA KV per token per rank:
  12 layers * 1 KV head * (K + V) * 256 dimensions * 2 BF16 bytes
  = 12,288 bytes
```

The sparse indexer adds a compressed BF16 key side cache. It stores one
128-value key every four tokens in each of the 12 QSA layers:

- [`QSACompressedKeyCache`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/common/qsa_cache.py#L797-L808)

```text
Indexer side cache per token per rank:
  12 layers * 128 dimensions * 2 BF16 bytes / compression ratio 4
  = 768 bytes

Total theoretical cache payload:
  12,288 + 768 = 13,056 bytes = 12.75 KiB/token/rank
```

At 148,400 tokens, that is 1.804 GiB per rank before paging, block rounding, raw
indexer ring state, and other cache metadata. The measured pool was 1.85-1.89
GiB per rank. The source-derived calculation therefore explains the fitted
context closely.

The current implementation explicitly rejects both lower-precision QSA cache
and context parallelism:

- the main QSA cache accepts only BF16;
- a quantization scheme with KV-cache settings is rejected;
- QSA reports `supports_dcp=False` and rejects decode or prefill context
  parallelism greater than one;
- the QSA side caches are also BF16 and report `supports_dcp=False`.

These restrictions are enforced in
[`qsa.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/qsa.py#L60-L215)
and
[`qsa_cache.py`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/common/qsa_cache.py#L655-L808).
The image's corresponding Qwen3.8 files have the same behavior and differ from
these PR-head sources only by the Qwen3.8-to-Qwen4Exp rename.

This makes cache format and cache sharding the large context levers:

- A one-byte cache format would roughly halve the 13,056-byte payload and put
  the native 262K context in range if its writers, sparse indexer, attention
  readers, quality, and SM86 performance were validated.
- Context-parallel cache ownership could remove the current TP replication, but
  the QSA implementation does not support it and it would add cross-rank decode
  communication.
- Raising utilization to the physical limit would increase KV from about 1.85
  to only 2.34-2.39 GiB per rank, enough for roughly 190K rather than 262K.
- Reducing the 2,048-token prefill chunk can reclaim part of the 0.99 GiB
  activation reservation, but it trades away prefill throughput and requires a
  measured A/B. Native 262K needs about 3.19 GiB of cache per rank at the current
  BF16 layout.

The 148K result is therefore not primarily an expert-parallelism limitation.
It is the combination of runtime residency, a large prefill reservation, and a
BF16 QSA cache replicated across TP ranks.

## Existing fixes and related work

### Official Qwen and community guidance

The official vLLM recipe says plain TP8 is incompatible with Qwen3.8's
128-wide quantization blocks and prescribes TEP8 on H200:

- [Qwen3.8 Flash Next recipe](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next)
  inspected with content SHA-256
  `cc1565f2949e07b31e6b0b2bc2ad228b84581ba3982f9f940ef2a739212a4b80`.

Vnimanie's W4A16 checkpoint documents the same 640/128 geometry and requires EP
for four or eight RTX 3090s:

- [Vnimanie W4A16 README at revision `9236d703`](https://huggingface.co/VnimanieAI/Qwen3.8-Flash-Next-W4A16/blob/9236d703b25f25eb5c17e9640204f84fa1ce0c6e/README.md#the-640128-problem--why-expert-parallelism-is-mandatory)

This is not proof that no better implementation can exist. It confirms that EP
is the intended current vLLM solution, not an accidental workaround unique to
our setup.

### Merged Marlin thread-tile padding

PR [#45703](https://github.com/vllm-project/vllm/pull/45703), merged as
[`37a682d392330da665690ee0a77c9d0a875f315f`](https://github.com/vllm-project/vllm/commit/37a682d392330da665690ee0a77c9d0a875f315f),
pads WNA16 MoE intermediate dimensions to Marlin thread tiles. It explicitly
leaves strict checks in place when a quantization group crosses a TP boundary.
It cannot solve 160/group-128 without changing the checkpoint partition.

### Group-aligned padded partitions

Open PR [#49951](https://github.com/vllm-project/vllm/pull/49951) addresses the
same root class for ModelOpt NVFP4. It changes Nemotron TP8 shards from 232 to
240 so every boundary aligns to group-16. The final rank loads the remainder and
zero-fills the tail.

That approach is economical for 232 to 240, a 3.4% increase. Applied directly
to Qwen's 160/group-128 case, each rank would need a 256-wide aligned slot. Four
slots would represent 1,024 positions for a real width of 640, a 60% expert
weight expansion. It is not a viable server60 default. The PR is also open and
merge-conflicting at head
`8d83404c1138f2948c35b5664bc47dddccd41315`.

### Scale-sharding bug

Open issue [#41511](https://github.com/vllm-project/vllm/issues/41511) shows a
compressed-tensors W4A16 case where weights shard along K but the complete scale
tensor remains on every rank. Its suggested scale slicing works when the number
of source groups divides across TP ranks. Qwen has five groups across four
ranks. At least one group must cross a rank boundary, so equal scale slicing is
not sufficient.

### Generic WNA16 kernel fixes

PR [#44563](https://github.com/vllm-project/vllm/pull/44563) fixes legal
`BLOCK_SIZE_K / group_size` selection inside the generic WNA16 kernel. It does
not preserve a group that crosses a rank boundary and does not remove the
fallback loader's scale repetition.

Issue [#17604](https://github.com/vllm-project/vllm/issues/17604) records the
same family of AWQ MoE TP failures and the practical EP workaround.

### Uneven tensor parallelism

Open draft PR [#47759](https://github.com/vllm-project/vllm/pull/47759)
implements model-specific uneven Qwen3.5/Qwen3Next attention, GDN, and dense-MLP
partitions. It excludes MoE/EP experiments and does not solve quantized routed
experts.

A group-aligned uneven MoE split of five groups across four ranks would assign
2/1/1/1 groups. One rank would own twice the routed-expert width and weight
storage of the others. On four 24 GiB cards, the heavy rank would not fit.

### Pipeline parallelism

The vLLM scaling guide recommends pipeline parallelism when TP splits are uneven
or GPUs lack NVLink:

- [Parallelism and scaling](https://github.com/vllm-project/vllm/blob/4c6c9d569f83d4dc5509e77e3458734a1fda2c2d/docs/serving/parallelism_scaling.md)

Qwen3.8's current model state rejects PP greater than one because non-first
pipeline ranks do not receive the raw token IDs needed for PLE:

- [`Qwen4ExpModelState`](https://github.com/vllm-project/vllm/blob/a5530b90cab09b187463396a99612a486ba91d6f/vllm/models/qwen4_exp/nvidia/model_state.py#L20-L45)

PP is therefore a separate model-plumbing project, not a launch-flag solution.

### SGLang and FlashInfer

SGLang's older WNA16 loader uses the same group-halving and scale-repetition
fallback. Its newer quantized MoE work generally pads group/tile-aligned shapes
or uses EP. SGLang issue
[#30887](https://github.com/sgl-project/sglang/issues/30887) describes gated
MoE padding for non-aligned NVFP4 shapes.

FlashInfer issue
[#3206](https://github.com/flashinfer-ai/flashinfer/issues/3206) reports Gemma4
NVFP4 working with TP4+EP but failing with TP-only when its rank-local
intermediate shape is incompatible. No reviewed solution there handles an
arbitrary group-128 boundary at N=160 on SM86.

### TensorRT-LLM

TensorRT-LLM exposes separate MoE TP and EP sizes, including hybrid ETP. Its
current TRTLLM-Gen path still requires each local TP shard to be a complete
multiple of the quantization alignment and fails closed otherwise:

- [TensorRT-LLM parallel strategy](https://nvidia.github.io/TensorRT-LLM/latest/features/parallel-strategy.html)
- [`fused_moe_trtllm_gen.py`](https://github.com/NVIDIA/TensorRT-LLM/blob/main/tensorrt_llm/_torch/modules/fused_moe/fused_moe_trtllm_gen.py)

TensorRT-LLM PR
[#11618](https://github.com/NVIDIA/TensorRT-LLM/pull/11618) also adds
quantization-aware sharding for NVFP4. It does not establish an existing
SM86 W4A16 group-straddling kernel that vLLM can adopt.

## Candidate directions after source review

### 1. Keep EP and reduce measured non-expert residency

This is the current leader.

EP already uses the original group-128 scales, complete 640-wide experts,
Marlin, no token all-to-all, and one final all-reduce. It reached 148K context
and beat no-EP decode by 20% in the bounded A/B.

The next evidence should be:

1. A storage-deduplicated parameter and buffer inventory for the exact Intel EP
   runtime.
2. Separation of routed experts, shared experts, attention/GDN, vision,
   embeddings/head, router/hyperconnection/norm state, and non-Torch memory.
3. A one-variable `max_num_batched_tokens` capacity/perfill A/B to test the
   0.99 GiB activation reservation.
4. An Nsight Systems decode trace and per-rank expert-token counts before
   attributing latency to EP imbalance or collectives.

### 2. Improve the plain-TP Triton fallback without a new GEMM

This is narrower than a new Marlin-like kernel.

The fallback uses one effective group size for both projections. Only the down
projection's local K=160 forces the smaller group. Gate/up's K=2,560 remains
natively divisible by 128, yet its scales are expanded fourfold too.

Keeping gate/up at group-128 while leaving down at the existing group-32
subdivision would reduce estimated no-EP scale storage from 1.758 GiB to about
0.879 GiB per rank, recovering about 0.879 GiB without changing dequantized
weights. Current `FusedMoEQuantConfig` documents that most kernels assume both
weights use the same group shape, and `TritonWNA16Experts` passes one
`block_shape` to both GEMMs. This requires a real interface and kernel contract
change, not a loader-only edit.

The no-EP log also reports no tuned Triton configuration for
`E=512,N=160,int4_w4a16` on RTX 3090. Tuning that exact shape may improve the
33.28 tok/s fallback. It will not remove scale storage by itself.

### 3. Add a group-phase-aware TP kernel

This remains technically possible, but it is not justified yet.

A correct kernel would need to preserve the global group-128 coordinates across
local K ranges `[0,160)`, `[160,320)`, `[320,480)`, and `[480,640)`. Three rank
boundaries split a quantization group. The kernel and loader would need an
explicit global K offset or per-rank group-phase contract, scale ownership for
partial boundary groups, and separate gate/up and down quantization descriptors.

Required gates would include:

- independent dequantized-value and MoE-output oracles;
- all four TP offsets, boundary groups, top-10 routing, and M=1/M>1 shapes;
- CUDA Graph replay;
- Compute Sanitizer memcheck and racecheck;
- packaged sm_86 code and runtime dispatch proof;
- exact parameter-storage accounting;
- an unprofiled end-to-end A/B against EP/Marlin;
- evidence that the gain exceeds EP despite both paths requiring the final
  hidden-state reduction.

The likely benefit is better rank-balanced compute for a single token. The
likely loss is a new kernel and loader contract for no packed-weight reduction
below EP's current floor. Measure EP rank imbalance first.

### 4. Add separate intra-expert TP and EP sizes

TensorRT-LLM supports hybrid ETP, while current vLLM collapses expert TP to one
when EP is enabled. A two-by-two split would produce local width 320, which still
crosses group-128 boundaries and falls back to group-64 without additional
work. It may reduce EP rank imbalance, but it does not remove the quantization
contract problem.

## Final assessment

The current Intel EP profile is not obviously wasting expert VRAM. It is the
only tested path that simultaneously keeps checkpoint-native group-128 scales,
uses Marlin, balances packed expert storage across four ranks, avoids token
all-to-all on this deployment, and preserves 148K context.

Plain TP revealed two valid improvement targets:

- vLLM's fallback unnecessarily expands gate/up scales because one shared group
  size governs both GEMMs.
- The exact E=512, N=160 Triton shape is untuned on RTX 3090.

Neither target proves that a new TP GEMM is the best next move. First explain the
remaining 1.42 GiB/rank above the ideal checkpoint quarter and profile the EP
critical path. If those measurements show routed-expert storage or EP compute
imbalance is still the dominant constraint, then a group-phase-aware TP kernel
has a concrete gate. Without that evidence, it would recreate machinery that EP
already avoids.
