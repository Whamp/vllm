# Qwen3.8 Flash Next GPU memory baseline

## Decision

The Intel AutoRound EP=4 profile has three measured model-side memory targets:

1. Hyperconnections occupy 1.215229 GiB per rank. They are the largest target,
   but ordinary tensor sharding would add latency-sensitive collectives at up to
   96 layer boundaries per token. Test weight-only Q8 before distributed
   sharding.
2. QSA registers 0.187775 GiB per rank of top-k output buffers from the default
   2,048-token batch budget. Test 1,024 before 512 because this budget also
   controls chunked-prefill throughput.
3. The shared QSA RoPE cache materializes 1,048,576 positions and occupies
   0.125 GiB per rank. A lower bound needs a vision/MRoPE position audit before
   implementation because multimodal position IDs need not equal text length.

Post-load Marlin processing is not an expansion source. It removes 204,472,320
registered bytes per rank. The quantized CPU PLE path adds 23,176,704 Torch
bytes during initialization and releases part of that setup state before worker
load completes.

No optimization is accepted by this report. It establishes the exact baseline
and the measurements required for the first A/Bs.

## Environment

The capture ran on server60 across four RTX 3090 GPUs under the persistent
230 W power limit. Host swap was disabled and serving-process swap remained
zero.

| Item | Identity |
| --- | --- |
| Model | Intel `Qwen3.8-Flash-Next-W4A16-AutoRound` |
| Model revision | `861536dda5bcb208376fc4cd879b2bf76bece9fe` |
| Derived model config | SHA-256 `932cbf4d5dc50efa395db095ea3664fd6ef7672886332b2d0307cd9aa28ac9cf` |
| Primitive PLE | revision `da8b39586016d8325ac619be28ad77d6296625ec` |
| Base image | `sha256:0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3` |
| Diagnostic source | Whamp/vLLM `1833ca8579be3075bbe4c89d24f9e32ceb275ce1` |
| Diagnostic image | `sha256:59e1df5a8023f7a9c8ee331321826efd6c68ea1bb165740e9a7f48d4e13200ec` |
| Runtime profile | TP=4, EP=4, `max_num_seqs=2`, `gpu_memory_utilization=0.95` |
| Context | automatic fit to 148,400 tokens |
| Main cache | BF16 QSA K/V, unchanged |
| Vision | enabled |

The diagnostic captured storage-deduplicated registered parameters and buffers,
Torch allocator counters, device-used memory, and residuals at ten startup
stages. It is disabled unless `VLLM_MODEL_MEMORY_REPORT_DIR` is set and remains
absent from compilation cache factors.

## Reproduction

The checksum-bound raw reports and deterministic analyzer are in
[`evidence/qwen38-memory-baseline-20260829`](evidence/qwen38-memory-baseline-20260829/README.md).
The raw manifest SHA-256 is
`690ae384e4385a1cea3db814c78be6c1ccd9612bbd11299bd44ac853565c78ee`.
The generated summary SHA-256 is
`f0281c9896c32d70cfde9b349e047a9e2e8a24ec94f7d2e119cc4b1e2534760a`.

## Staged memory

Values are the four-rank means. Registered storage is deduplicated by device,
storage pointer, byte offset, and storage size, so parameter aliases count once.

| Stage | Registered GiB/rank | Device used GiB/rank |
| --- | ---: | ---: |
| Distributed initialized | 0 | 0.434753 |
| Model runner initialized | 0 | 0.436707 |
| Module initialized | 18.740460 | 19.819519 |
| Weights loaded | 18.740460 | 19.898941 |
| Postprocessed | 18.550030 | 21.996597 |
| PLE offload initialized | 18.550030 | 19.519455 |
| Worker model loaded | 18.550030 | 19.512222 |
| Profile complete | 18.550030 | 19.950684 |
| KV cache allocated | 18.550030 | 21.649902 |
| Warmup complete | 18.550030 | 21.755371 |

The postprocessing stage's 21.996597 GiB device use is transient. Device use
falls to 19.519455 GiB after PLE setup while registered storage remains fixed.
The durable postprocessed storage is 0.190430 GiB smaller than the loaded
module storage.

The automatic cache allocator adds 1.848440 GiB of Torch allocations per rank.
That agrees with the runtime's reported approximately 1.85 GiB cache pool and
148,400-token fitted context.

At profile completion, device use beyond startup baseline and registered model
storage is 1,037,127,144 bytes, or 0.965900 GiB per rank. Its measured parts are:

| Residual at profile completion | GiB/rank |
| --- | ---: |
| Unregistered persistent Torch allocation | 0.225082 |
| Torch allocator cache | 0.437778 |
| Non-Torch growth above distributed startup | 0.303040 |
| **Total** | **0.965900** |

The allocator-cache value is observational, not a reclaim estimate. vLLM's
memory-profiling context empties the allocator before its authoritative
`after_profile` snapshot. This diagnostic stage runs after later graph-memory
estimation, so a separate free-memory A/B is required before changing allocator
lifecycle.

## Registered model storage

Every postprocessed rank has the same deduplicated category totals.

| Owner | GiB/rank |
| --- | ---: |
| Routed experts | 14.611908 |
| Hyperconnections | 1.215229 |
| GDN and linear attention | 0.983974 |
| Embedding and LM head | 0.592041 |
| QSA attention | 0.292980 |
| Vision | 0.216048 |
| QSA top-k buffers | 0.187775 |
| RoPE cache | 0.125000 |
| MoE routers | 0.117188 |
| Shared experts | 0.110092 |
| PLE projections and buffers | 0.061169 |
| QSA indexer | 0.036627 |
| **Total** | **18.550030** |

This exact inventory replaces the earlier source-only estimate. In particular,
the RoPE cache is 128 MiB per rank, not 32 MiB. Its storage spans 1,048,576
positions even though model auto-fit selected 148,400 tokens.

## Gated hypotheses

### H1: Q8 hyperconnection weights

**Observed cost.** Hyperconnection parameters and buffers occupy 1,304,842,240
bytes per rank. Their source modules use replicated up projections and
`disable_tp=True` merged down/injection projections.

**Move.** Keep the communication contract unchanged and represent eligible BF16
hyperconnection matrix weights with an SM86-supported W8A16 path. A raw
one-byte-per-weight upper-bound estimate reclaims about 0.61 GiB per rank before
scale and layout overhead.

**Gate.** Prove that the exact skinny shapes dispatch a supported SM86 kernel and
that it does not regress decode or prefill. Compare numerical output at each
changed projection, then run deterministic, tool, reasoning, vision, NIAH, and
benchlocal checks.

**Lose condition.** Reject the path if repacking, scales, activation conversion,
or a slow skinny-GEMM fallback erases capacity or throughput gains. Q4 is a
later quality-gated arm, not part of the first experiment.

**Why not shard first.** Perfect four-way storage sharding could reclaim about
0.911 GiB per rank, but an ordinary row- or column-parallel implementation adds
small collectives at latency-sensitive hyperconnection boundaries. Do not pay
that coordination cost without a design that reuses or fuses existing
collectives.

**Result: reject the generic block-FP8 Marlin path.** The real 128x128 block-FP8
layout would use 776,079,360 bytes per rank and reclaim 528,762,880 bytes, or
0.492449 GiB, after exact Marlin padding, scale, and workspace accounting. A
97-group real-weight screen measured aggregate normalized RMSE 0.026313 and
cosine 0.999630. The exact-shape RTX 3090 gate selected
`MarlinFP8ScaledMMLinearKernel`, passed numerical and deterministic CUDA-Graph
checks, but ran 2.9 to 5.7 times slower than BF16 across M=1, M=2, and M=256.
The full model was not launched with this method. The experimental source was
reverted. See
[evidence/qwen38-hyperconnection-fp8-20260829/README.md](evidence/qwen38-hyperconnection-fp8-20260829/README.md).

This result rejects vLLM's existing generic Marlin path for these skinny shapes.

**Result: reject the generic Cutlass W8A8 path.** A real-weight layer-0 screen
used per-output-channel INT8 weights and dynamic per-token INT8 activations. It
cut the two tested matrices' weight-and-scale storage by 49.7% and passed
bitwise-equal finite CUDA Graph replay, but ran 2.5 to 3.8 times slower than
BF16. The merged down-and-injection projection also missed the 0.02 normalized
RMSE and 0.9999 cosine bounds. See
[evidence/qwen38-hyperconnection-int8-20260829/README.md](evidence/qwen38-hyperconnection-int8-20260829/README.md).

The two generic compressed-linear paths are now closed. An all-weight screen
selects K-group-128 scales for down and injection plus per-row scales for up as
the custom-kernel input contract. It has 0.010262 aggregate weight-reconstruction
NRMSE, 0.999912 cosine, 0.016962 worst-tensor NRMSE, and projects 603.31 MiB of
registered storage savings per rank. It still needs a purpose-built kernel for
the `[336, 10240]` and `[10240, 320]` shapes, grouped-activation output oracles,
and exact-shape speed evidence. The projected saving leaves about 67 MiB of the
670 MiB capacity target to recover elsewhere.

### H2: lower the QSA top-k batch allocation

**Observed cost.** Twelve QSA layers each register an INT32 buffer shaped from
2,048 batched tokens and output width 2,051. The exact total is 201,621,504
bytes per rank.

**Move.** Set `max_num_batched_tokens=1024` as a one-variable A/B. This halves
the buffer and reclaims 96.14 MiB per rank. A later 512-token arm would reclaim
144.21 MiB relative to the baseline.

**Gate.** The reclaimed registered bytes must match the estimate. Cache-busted
prefill, decode, concurrency two, fitted context, activation peak, and graph
memory must all be measured on the same artifact and image.

**Lose condition.** Reject any setting whose prefill loss is not justified by
the fitted-context gain. The current 2,048-token budget may be serving real
chunked-prefill work rather than accidental slack.

**Result: promote 1,024 tokens.** The registered QSA top-k allocation fell by
exactly 100,810,752 bytes per rank, with no other registered-storage change.
Auto-fit context rose from 148,400 to 156,400 tokens. On the final production
image, decode retained 99.71% of control and cache-busted prefill retained
94.70%. Exact 145,041-token retrieval, deterministic text, streaming, tools,
post-tool continuation, vision, concurrency two, and BenchLocal quick 26/30 all
passed with zero swap. See
[QSA-TOPK-BUFFER-1024.md](QSA-TOPK-BUFFER-1024.md).

### H3: bound RoPE materialization

**Observed cost.** The shared BF16 RoPE storage is 134,217,728 bytes per rank,
which is 128 bytes for each of 1,048,576 cached positions.

**Move.** Bound only materialized rows while preserving the model's RoPE
frequency and MRoPE semantics. A 262,144-row cache would reclaim exactly 96 MiB
per rank. A 148,400-row cache would reclaim about 109.88 MiB per rank.

**Gate.** First prove the maximum Qwen multimodal position ID for the supported
image and video paths. Text sequence length alone is not a valid bound for
MRoPE. The implementation must fail closed or grow safely before an index can
exceed materialized rows, remain CUDA-Graph compatible, and preserve long-text
and multimodal outputs.

**Lose condition.** Do not implement a text-length cap that breaks legal video
position IDs or adds runtime reallocations to graph replay.

**Result: promote 262,144 rows.** A 150-example generated image/video property
proved Qwen3.8 MRoPE positions remain token bounded and killed a shifted-position
semantic counterfeit. Cloning the first 262,144 rows reclaimed exactly
100,663,296 bytes per rank, increased fitted context from 156,400 to 167,600,
and preserved exact retrieval at 160,035 prompt tokens. The exact final image
retained 98.06% decode and 99.42% cache-busted prefill, passed vision, tools,
concurrency two, and BenchLocal quick 26/30, and remained zero-swap. See
[QSA-ROPE-BOUND.md](QSA-ROPE-BOUND.md).

### H4: native Q8 and Q4 QSA cache

The current QSA path explicitly requires BF16 main K/V storage and does not
support cache quantization. Q8 and Q4 therefore need new write, read, attention,
allocation, graph, and numerical contracts. They are not launch-flag tests.

The baseline cache costs 13,056 bytes per token per rank across the twelve QSA
layers and compressed indexer side cache. Q8 could nearly halve the dominant
main K/V term and Q4 could nearly quarter it, but neither estimate includes
scales, side-cache treatment, conversion work, or kernel efficiency. Start this
work only after H1 to H3 establish the fixed-residency baseline and available
headroom for kernel bring-up.

**FP8 result: reject.** Typed E4M3 does not compile for SM86 in the installed
Triton. Typed E5M2 missed the numerical bound at 0.059410 normalized RMSE.
Integer-decoded E4M3 passed numerical and M=1 timing gates but ran 28.46 times
slower than BF16 at M=256. No full-model FP8 launch was attempted. See
[QSA-FP8-CACHE.md](QSA-FP8-CACHE.md).

**INT8 result: reject.** Per-token-head INT8 passed its writer, numerical,
CUDA-Graph, and M=1 reader gates. Its M=256 sparse reader ran 26.32 times slower
than BF16. The generated cache-layout property also passed 100 valid HND/NHD
cases and caught two shrunk failures under a temporary stride counterfeit. See
[QSA-INT8-CACHE.md](QSA-INT8-CACHE.md).

**INT4 result: reject.** Packed Q4 passed generated layout, RHT, and sparse
attention properties, three semantic counterfeit kills, numerical bounds,
CUDA-Graph replay, and the M=256 performance gate. Matrix RHT reduced M=1 from
3.44 to 2.03 times BF16, but four Q4-only decode schedules remained at 2.06 to
2.09 times BF16 against a fixed maximum of 1.25. No full-model launch was
attempted. See [QSA-INT4-CACHE.md](QSA-INT4-CACHE.md).

### H5: executor budget ceiling

**Result: promote 0.968.** The bounded-RoPE service still had 1.8 GiB physical
headroom at `gpu_memory_utilization=0.95`. A preregistered 0.97 arm fit 206,400
tokens and passed correctness and performance, but failed the 1 GiB post-NIAH
margin by 4–8 MiB and rolled back. The 0.968 arm fit 202,400 tokens, passed exact
190,047-token retrieval, retained 101.04% decode and 100.13% cache-busted prefill
on the final production identity, scored BenchLocal quick 26/30, and retained at
least 1,098 MiB free with zero swap. See
[GPU-MEMORY-UTILIZATION-0968.md](GPU-MEMORY-UTILIZATION-0968.md).

## Restored service

After capture, the diagnostic container was removed and the original service
was restored on the exact base image. Fresh checks showed:

- HTTP health returned 200;
- `/v1/models` reported the original model and 148,400-token context;
- a deterministic request returned exactly `PARIS` with a normal stop;
- restart count was zero;
- every serving process had zero swap;
- `vm.overcommit_memory` was restored to `0`;
- `gpu-power-limit.service` was active.

The rollback script's PLE-log predicate timed out after the service had become
healthy because the expected registration line was absent from the fresh log.
The predicate was replaced with direct health, process, image, model, swap,
host-policy, and deterministic-output checks. The block-FP8 gate used this
repaired rollback after each bounded attempt. Fresh final checks again found the
original image healthy with zero restarts and zero serving-process swap. The
remaining restore timer was cancelled to prevent an unnecessary second restart.
