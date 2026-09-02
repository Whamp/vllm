<!-- markdownlint-disable MD060 -->

# Four RTX 3090s, two giant models, and the optimizations that survived contact with hardware

*How we turned a PCIe-only consumer GPU box into a useful long-context inference server, including the ideas that worked, the ones that failed, and the bugs that sent us in the wrong direction.*

## The short version

Server60 has four 24 GB RTX 3090s, no active NVLink, uneven PCIe Gen3 links, 60 GiB of system RAM, and a fixed 230 W safety limit. It is not the machine most modern inference software targets. That made it a useful test of a simple rule: an optimization is not real until it works on the actual model, hardware, runtime, and workload.

Over this thread we worked on two models:

- DeepSeek-V4-Flash-0731, first through low-bit safetensors and later through the original Antirez GGUF;
- Qwen3.8-Flash-Next, using Intel AutoRound weights and a quantized CPU-resident PLE table.

The largest successful DeepSeek result was a native GGUF tensor-parallel engine inside vLLM. It executed the original IQ2_XXS, Q2_K, and Q8_0 bytes with custom SM86 kernels, reached 76.70 decode tokens/s, and matched the proven llama.cpp model on an agentic coding task at 2.65 times the wall-clock speed. Later work pushed decode to about 80 tokens/s, added a 368-byte FP4 MLA cache, and validated exact recall at 173,058 tokens. Some capacity variants fit much more cache but were too slow or too close to the memory limit to promote.

The final Qwen3.8 production service reached the model's native 262,144-token context. Its direct calibrated E4M3 QSA cache, runtime-bounded RoPE, smaller persistent top-k workspace, and island-aware hierarchical all-reduce produced 50.34 decode tokens/s and 1,538.14 cache-busted prefill tokens/s in the exact-final benchmark. The current `gpu_memory_utilization=0.98` setting allocates 421,608 cache tokens and passed an exact 261,544-token retrieval, image input, tools, post-tool continuation, concurrency two, and warmed memory-stability checks.

Those wins came from removing specific measured costs. Most generic ideas failed. CPU offload destroyed decode. Generic FP8 and INT8 skinny GEMMs were slower than BF16. Several Q4 and Q8 cache readers were numerically correct but far too slow. Decode context parallelism doubled cache capacity and cut speed by more than half. A proposed fusion pass disappeared after the trace showed only a 3.5% optimistic ceiling.

That pattern is the real story.

## The machine shaped every decision

Server60 is a single-NUMA Threadripper 2950X host with four RTX 3090s. CUDA peer reads and writes work across every pair, but the topology has two local PCIe islands and no active NVLink. The observed links are x4, x16, x8, and x16. The host has 60 GiB RAM, which is enough for services and a compressed PLE table but not enough to pin a 140 GiB routed-expert pool.

The GPU safety policy stayed fixed throughout accepted work:

- 230 W power limit per card;
- 210 to 1650 MHz graphics-clock range;
- no performance result accepted by raising power or clocks.

That last rule mattered. A 1995 MHz clock experiment improved llama.cpp decode by more than 10%, but it violated the operating policy and increased idle draw. We removed the controller and restored the safety cap. It remains a rejected historical result, not an optimization.

We also treated service recovery as part of correctness. GPU experiments stopped the live model, armed an exact rollback timer, used non-restarting candidates, normalized swap, and verified the restored image, model identity, context, restart policy, power service, and per-process swap. A fast kernel is not useful if the server ends the test in an unknown state.

## The method: one causal claim at a time

Two working rules governed the project.

First, every optimization needed a gate. We named the expected mechanism, the cost it should remove, the measurement that could confirm it, and the result that would kill it. We did not promote a method because a README said "SM86 supported" or because a cubin contained `sm_86`. We checked the package, generated device code, runtime dispatch, numerical output, CUDA Graph replay, serving behavior, and end-to-end performance.

Second, correctness evidence had to challenge the likely bug. The test ladder included:

1. independent format decoders and randomized block oracles;
2. coordinate-aware tensor mapping and exact shard inventories;
3. property tests that killed counterfeit strides, nibble order, position offsets, and layout assumptions;
4. CUDA Graph replay and caller-stream ordering;
5. Compute Sanitizer memcheck and racecheck;
6. full-model deterministic, tool, post-tool, streaming, reasoning, and vision probes;
7. long-context needle retrieval;
8. BenchLocal or DeepSWE when the change could affect model capability.

This was slower than toggling flags. It was faster than debugging silent corruption after a 250,000-token request.

# Part one: DeepSeek V4 Flash

## We first made low-bit safetensors run on Ampere

The starting point was an apparent opportunity. DeepSeek V4 Flash fit in 96 GiB when heavily quantized, llama.cpp already ran it, and vLLM could be much faster if we could feed the model into its tensor-parallel engine.

The first artifact was an imatrix-weighted W2A16 safetensors model. We built a resumable conversion pipeline, generated the artifact on a rented A100, published 45 checksum-verified shards, and added a compressed-tensors to Humming MoE path for RTX 3090. Bring-up found a series of independent runtime faults:

- compressed-tensors metadata accidentally replaced the model's native block-FP8 declaration;
- generic FP8 handling transformed `wo_a` into the wrong layout;
- the Humming factory failed to forward the layer object;
- A100-oriented sparse prefill and split-K decode kernels exceeded the RTX 3090 shared-memory limit;
- DeepSeek's DSML parser did not stop at the outer tool-call marker;
- conservative eager bring-up settings disabled the path vLLM needed for speed.

The early service decoded at only 5.55 tokens/s. That number was real, but it described the correctness configuration, not vLLM's steady-state capability. The key A/B changed one flag: breakable CUDA graphs versus `--enforce-eager`. Eager measured 4.96 tokens/s. Graph execution measured about 60. The graph path was a roughly 12 times decode improvement and the first major win.

The lesson was blunt: a valid benchmark can still answer the wrong question.

## We recovered context by removing a real duplicate

The first graph-enabled profile stopped near 131K context. File-size arithmetic suggested several GiB should still exist, so we instrumented storage rather than trimming more scheduler knobs.

The trace found a 16 MiB BF16 `wo_a` output-projection tensor retained in every one of 43 attention layers. The SM86 path dequantized the block-FP8 weights once and kept the BF16 result for `einsum`. That cost 688 MiB per rank.

The naive fix removed the cache and dequantized on every token. It reached 230,144 context, but decode collapsed from 61.56 to 34.01 tokens/s. Capacity improved for the wrong reason: we moved work into the latency path.

The accepted design kept the original FP8 weight and scales, packed them for Marlin, flattened the two local groups into one projection, and selected the matching diagonal outputs. This removed the 688 MiB duplicate without repeated dequantization.

The result at 230,144 context was:

| Metric | 131K baseline | Marlin `wo_a` |
| --- | ---: | ---: |
| Decode | 61.56 tok/s | 61.91 tok/s |
| Cache-busted prefill | 920.91 tok/s | 875.93 tok/s |
| Aggregate KV capacity | 148,290 | 275,238 tokens |
| Exact long-context recall | shorter gate | 211,031 tokens |

The service also handled two simultaneous 90,029-token retrieval requests. The cost was physical headroom: only about 93 MiB per GPU remained after the long request. We treated 230K as a measured ceiling, not a comfortable operating point.

## The speed stack that worked

Once the model fit, Nsight showed where decode time went. Two mechanisms survived matched A/Bs.

### Native Ampere FlashMLA sparse decode

The AppMana FlashMLA fork supplied an SM86 sparse MLA path over the existing `fp8_ds_mla` cache. Kernel oracles passed on RTX 3090, generated cubins contained SM86 code, and the end-to-end service improved from about 62.06 to 70.84 decode tokens/s while prefill remained flat.

### Island-aware hierarchical all-reduce

The four cards form two PCIe islands: ranks `0,1` and `2,3`. A valid Nsight trace assigned about 17% to 21% of each GPU's decode span to NCCL with no same-GPU overlap. The existing hierarchical backend passed BF16 numerical and CUDA Graph tests and beat NCCL across the serving-size payloads.

Combining FlashMLA and hierarchical all-reduce produced 74.98 decode tokens/s and 887.52 cache-busted prefill tokens/s. Tool use, coding, deep reasoning, and exact recall through 211,551 tokens passed.

## The parser bug that looked like a bad quant

Both WNA16 artifacts sometimes behaved normally for two tool turns and then emitted one enormous response with hundreds of repeated reads. The pattern looked like quantization damage. It was not.

The tokenizer rendered valid DSML tool calls, but vLLM failed to add the outer tool-call closing marker as a stop sequence. Commit `9a2ffbb4` fixed the parser. Post-tool continuations became normal immediately.

That correction did not prove the WNA16 artifacts were good. A later 12-task DeepSWE comparison still showed a large quality gap:

| Model path | Strict solves | Mean partial reward | Agent-hours |
| --- | ---: | ---: | ---: |
| WNA16 safetensors | 0/12 | 80.62% | 9.65 |
| Antirez IQ2_XXS GGUF | 6/12 | 96.57% | 19.85 |

WNA16 was about twice as fast in that comparison but materially less capable. Two things had been true at once: the parser was broken, and the requantized model was worse.

That finding changed the direction of the project.

## We tried to make llama.cpp much faster

The proven-quality fallback was Antirez's IQ2_XXS/Q2_K/Q8_0 GGUF in a specialized llama.cpp fork. It served a 430,080-token Q8 KV context, recalled a needle at 395,282 tokens, prefetched that deep prompt at about 913 tokens/s, and decoded around 33 to 38 tokens/s.

Nsight showed quantized matrix-vector kernels dominated. Exact-shape microbenchmarks measured roughly 346 to 358 GB/s for IQ2_XXS, 307 GB/s for Q2_K, and 713 GB/s for Q8_0. We tested the obvious kernel knobs and found little room:

- `DSV4_MOE_GEMV_RPB` did nothing;
- Q2_K accumulation cleanup changed bandwidth by about 1%;
- Q8 rows-per-block was a wash;
- larger VDR values looked fast only because they dropped contributions and changed checksums;
- a semantics-preserving VDR rewrite produced shape-dependent gains worth only an estimated 2% to 4% end to end;
- clock locking helped, but violated the server safety policy and was removed.

The remaining legal whole-service improvement bound was about 1.5%, below the 10% campaign target. The campaign closed as measured infeasible.

This explained the larger architectural gap. llama.cpp split layers serially across four GPUs. vLLM tensor-parallelized every layer and reduced its communication tax. Better MMVQ kernels could not remove the pipeline structure.

## The pivot: execute GGUF bytes natively inside vLLM

The best weights lived in GGUF. The fastest runtime was vLLM. Instead of converting one into the other, we built a model-specific native GGUF execution path.

The scope was intentionally narrow: one architecture, one artifact family, one hardware target. No GGML linkage, no generic model support, no float expansion of the low-bit experts.

The implementation added:

- a bounded, checksum-bound GGUF parser and exact 1,328-tensor inventory;
- coordinate-aware TP planning with 1,180 runtime targets;
- native IQ2_XXS gate/up and Q2_K down kernels for indexed decode;
- grouped SM86 integer-MMA kernels for prefill;
- fused clamped SwiGLU, route weighting, and Q8_1 requantization;
- byte-neutral Q8_0 to Marlin preparation for dense linears and `wo_a`;
- CUDA Graph-safe operators with caller-owned workspaces.

The format work caught real bugs before serving. The independent Q2_K decoder had a chunk offset error. Later, Unsloth IQ1 tests passed by construction until a real-weight replay proved that both the kernel and synthetic oracle used the wrong lane-interleaved nibble order. The corrected oracle failed first, then the corrected kernel passed.

The first complete Antirez GGUF-TP profile measured:

| Gate | Result |
| --- | ---: |
| Decode | 76.697 tok/s |
| Cache-busted prefill | 551.89 tok/s |
| Concurrency two | 121.86 aggregate tok/s |
| Configured context | 140,000 tokens |
| Exact recall | 119,730 tokens |
| Quick quality | 27/30 |

The decisive quality gate was a one-cell SuperJSON DeepSWE run:

| Result | GGUF-TP | llama.cpp control |
| --- | ---: | ---: |
| Partial reward | 0.9949 | 0.9898 |
| Feature tests | 79/80 | 78/80 |
| Preservation tests | 116/116 | 116/116 |
| Wall time | 2,520 s | 6,678.5 s |

The native engine preserved the model's capability and completed the task 2.65 times faster.

## DeepSeek extensions that worked, and those that did not

### FP4 MLA cache worked, but the highest-capacity profile was unsafe

The native `fp4_ds_mla` format reduced each stored token row from 584 to 368 bytes. It kept BF16 RoPE and used native AppMana FP4 sparse decode and prefill.

At 148K, decode remained about 80 tokens/s. Prefill fell by 3% to 5%. A later 175K profile passed exact retrieval at 173,058 tokens and the full functional stress suite, but left only 27 MiB free per card. It was a functional success and a production no-go.

### Unsloth IQ1 support became functional but remained slow

We added native IQ1_S, IQ1_M, IQ3_XXS, MXFP4, Q4_K, Q5_K, and Q6_K execution plus split-GGUF loading. After fixing the lane-interleaving defect:

- UD-IQ1_S reached 40.80 engine decode tokens/s and about 251 prefill tokens/s;
- UD-IQ1_M reached 44.79 engine decode tokens/s and about 275 prefill tokens/s.

Both models loaded, generated coherent text, used tools, and passed bounded long-context checks. Both were roughly half the Antirez engine's speed, so neither replaced it.

### Decode context parallelism traded too much speed for cache

DCP sharded compressed MLA history across the four ranks. Correctness required global top-k merging, local slot translation, fixed-order LSE merging, and applying the attention sink exactly once. Several bugs appeared only at long context, including rank-local compressed indices being passed as physical FlashMLA slots.

After repair, DCP=4 passed the 148K functional suite but decoded at about 37.5 tokens/s, versus about 79.8 without DCP. A 262,144-token service reached readiness but timed out at the 240K stress rung with only 11 to 12 MiB free per GPU. It stayed experimental.

### Fusion and cold-expert offload failed their gates

A route study found the simple uniform 224-hot-expert cache covered only about 95% to 96% of accesses rather than the required 99%. That rejected only the uniform design, not every possible offloader, but it removed the easy version.

A proposed decode fusion pass also died after measurement. The production-semantics Nsight trace showed the standalone nodes available to fuse totaled only 10.496 microseconds per layer. That translated to an optimistic 3.5% whole-token ceiling, not the expected 9% to 12%. We wrote no fusion kernel.

# Part two: Qwen3.8 Flash Next

## The model brought a different memory problem

Qwen3.8 Flash Next has 48 hybrid recurrent and full-attention layers, 512 routed experts with top-10 routing, a vision encoder, and a 20-million-entry per-layer token embedding table. The PLE table is about 51 billion parameters by itself.

The first usable vLLM profile combined:

- Intel's symmetric group-128 AutoRound W4A16 model;
- Primitive AI's quantized NVFP4 PLE sidecar on CPU;
- TP=4 plus EP=4;
- vision enabled;
- BF16 QSA main cache;
- MTP disabled.

Expert parallelism was not an arbitrary preference. With EP disabled, each TP rank received a 160-wide expert shard that did not divide the group-128 quantization geometry. vLLM fell back from Marlin to generic Triton and reduced the effective group size to 32, quadrupling routed scale storage. The no-EP profile used 19.76 GiB model memory per rank, fit only 82K context, and decoded at 33.28 tokens/s. EP=4 used 18.55 GiB, fit 148K, and decoded at 39.94 tokens/s.

## We made the PLE path production-safe

The original sidecar integration constructed a 102.4 GB BF16 PLE parameter and replaced it later. It needed temporary overcommit and was easy to break under a 48 GiB cgroup.

The durable path attaches the sidecar before full-table materialization. It validates all 128 shards and their metadata, preserves per-shard outer scales, gathers requested rows with order and duplicates intact, and dequantizes into caller-owned CPU output. A strict-overcommit, no-network, no-GPU test covered all 320,001,536 rows with about 526 MiB peak RSS and zero swap.

We also replaced PyTorch CUDA-tensor pickling with raw CUDA-driver IPC descriptors. That removed the `SYS_PTRACE` dependency while preserving exact H2D results and explicit mapping cleanup.

These changes did not raise tokens per second. They removed fragile startup behavior, which is what made later performance work reproducible.

## Memory accounting found two cheap wins

The first staged diagnostic reported 18.550 GiB of registered model storage per rank. The largest owners were routed experts at 14.612 GiB, hyperconnections at 1.215 GiB, GDN and linear attention at 0.984 GiB, and QSA top-k buffers at 0.188 GiB.

Two persistent allocations were removable without changing model arithmetic.

### Halve the QSA top-k workspace

Each of 12 QSA layers allocated an INT32 top-k buffer sized by `max_num_batched_tokens`. Reducing the budget from 2,048 to 1,024 reclaimed exactly 100,810,752 bytes per rank and raised auto-fitted context from 148,400 to 156,400.

Decode moved from 44.07 to 43.94 tokens/s. Prefill fell 5.3%, from 1,636.62 to 1,549.89 tokens/s. We accepted that cost for 8,000 more context tokens.

### Bound the RoPE cache to legal positions

The inherited vision implementation materialized 1,048,576 BF16 RoPE rows, four times the model's 262,144-token maximum. A generated multimodal property proved that Qwen3.8 image, video, text, and continuation positions remain token bounded.

Cloning only the first 262,144 rows reclaimed exactly 96 MiB per rank. Context rose from 156,400 to 167,600 while decode retained 98.06% and prefill retained 99.42%. Exact recall passed at 160,035 tokens.

A later executor-budget increase to 0.968 raised the BF16-cache profile to 202,400 tokens with exact 190,047-token recall, 43.54 decode tokens/s, 1,542.77 prefill tokens/s, and a 26/30 BenchLocal quick score.

## The hyperconnection compression campaign was a complete no-go

Hyperconnections occupied 1.215 GiB per rank, so they looked like the obvious next target. We tested the idea harder than most production features receive.

- generic block-FP8 Marlin passed numerical and CUDA Graph checks but ran 2.9 to 5.7 times slower than BF16;
- generic Cutlass W8A8 was 2.5 to 3.8 times slower and missed one numerical bound;
- a purpose-built Triton INT8 path was 2 to 5 times slower;
- a native DP4A kernel passed graph replay but remained slower;
- a native SM86 IMMA path made the per-row up projection slightly faster at M=1/2, but the quality-required group-128 merged-down path was about 9.5 times slower.

The problem was structural. Eighty separately scaled merged-down groups prevented accumulation across the full K dimension. Scale handling erased the integer Tensor Core advantage.

We closed the entire compression route without launching a full model. The storage estimate was attractive. The exact shapes were not.

## Most low-precision QSA cache designs also failed

QSA stores a large BF16 main K/V cache and smaller raw and compressed side caches. Reducing the main cache was the largest path to native 262K context, but the generic implementations were poor on SM86.

### Rejected cache paths

| Candidate | Correctness | Performance result | Decision |
| --- | --- | --- | --- |
| Typed E5M2 FP8 | Failed numerical bound | Not relevant | Reject |
| Software E4M3, generic reader | Passed numerical | 28.46x BF16 at M=256 | Reject |
| INT8 per-token-head | Passed numerical and graphs | 26.32x BF16 at M=256 | Reject |
| Packed INT4 | Passed properties, numerics, graphs | Best M=1 still 2.03x BF16 | Reject |
| Direct Q8-K/Q4-V Triton | Passed numerical and graphs | 4.206x BF16 at M=1, 1.717x at M=256 | Reject |

Property testing paid for itself here. Counterfeit nibble order, omitted zero-point correction, missing RHT normalization, and broken HND/NHD strides all failed with small generated examples. The kernels were slow for real reasons, not because the tests were weak.

## The direct calibrated E4M3 path worked

An external four-3090 recipe pointed to a different mechanism. Instead of a generic float8 tensor path, it stored E4M3 bytes, decoded them in registers on SM86, folded calibrated K/V scales into score and output scaling, and selected narrow shape-specific kernel profiles.

We adapted only that cache mechanism. We did not copy the external checkpoint or companion FP8 weight changes.

The calibration ran 24 frozen text, code, math, multilingual, and long prompts plus two real images across all four ranks. It produced positive finite per-layer K/V scales for all 12 QSA layers with a 1.125 safety margin.

The final standalone RTX 3090 gate covered all 256 E4M3 byte patterns, numerical output, finite bitwise-equal CUDA Graph replay, and reader timing at M=1, 8, 32, 256, and 512. Every tested shape stayed within 1.25 times BF16 reader time.

The full model reached 262,144-token readiness with native vision and tools. Before collective optimization it measured 43.77 decode tokens/s and 1,529.25 prefill tokens/s, essentially equal to the BF16-cache service while providing enough capacity for exact retrieval at 261,544 tokens.

This is an important distinction. "FP8 QSA failed" and "FP8 QSA worked" both appear in the record. The generic software reader failed. The direct calibrated reader with SM86-specific profiles worked. The representation was not the recommendation. The mechanism was.

## Nsight found the next win in communication

The first c=1/c=2 trace showed BF16 ring all-reduce consumed 62.5% of summed kernel time in the c=1 segment and 65.8% in c=2. Across six measured decode episodes, collectives cost 4.221 ms of summed GPU kernel time per generated token.

The deployed Qwen image lacked the existing Whamp island-aware backend, so we added a narrow compatibility overlay and configured ranks `0,1;2,3`.

The four-GPU mechanism gate beat NCCL at every tested payload. The exact-final serving result was:

| Metric | PYNCCL | Hierarchical | Change |
| --- | ---: | ---: | ---: |
| Decode | 43.77 tok/s | 50.34 tok/s | +15.0% |
| Cache-busted prefill | 1,529.25 tok/s | 1,538.14 tok/s | +0.6% |
| Concurrency two aggregate | 53.25 tok/s | 59.00 tok/s | +10.8% |

The model retained vision, automatic tools, post-tool continuation, 261K exact recall, BenchLocal quick 26/30, zero swap, and zero restarts.

A compile-mode A/B was less interesting. `VLLM_COMPILE` with full and piecewise graphs measured 44.19 decode and 1,534.46 prefill against the then-current 43.77 and 1,529.25 baseline. It was a wash and was not promoted.

## The final 0.98 production setting

The last change raised `gpu_memory_utilization` from 0.95 to 0.98 without changing the model, image, network, port, vision contract, scheduler limits, or cache format.

We first checked multimodal reservation. vLLM already profiled one maximum-size image with a 16,384-token encoder budget. `--limit-mm-per-prompt` controls item count and would not reduce that one-image profile. Dummy width and height options alone would under-profile larger inputs. A safe image-memory reduction would need an enforced processor bound such as `max_pixels`, so we left the vision contract unchanged.

At 0.98, startup reported:

- 2.79 GiB available KV memory;
- 421,608 aggregate cache tokens;
- 1.61 times maximum concurrency at 262,144 tokens.

The service passed deterministic generation, tools, post-tool continuation, image input, concurrency two, and exact 261,544-token retrieval. The first acceptance warmed another 460 to 500 MiB per GPU. Three further concurrency-two rounds produced no additional NVML growth. Final free memory was 550 MiB on GPU 0 and 654 MiB on GPUs 1 through 3, with no OOM, allocator retry, restart, host swap, or process swap.

That is the current production state.

# What worked

The successful changes share one property: each removed a measured cost without moving comparable work into a hotter path.

| Change | Measured effect |
| --- | --- |
| DeepSeek breakable CUDA graphs | About 4.96 to 60 tok/s decode |
| DeepSeek runtime-bounded RoPE | About 407 MiB/rank reclaimed at 215K |
| DeepSeek FP8 Marlin `wo_a` | Removed 688 MiB/rank, 131K to 230K context without decode loss |
| DeepSeek FlashMLA plus hierarchical AR | About 61.17 to 74.98 tok/s decode |
| Native DeepSeek GGUF-TP | 76.70 tok/s and llama.cpp-level agentic quality |
| DeepSeek FP4 MLA cache | 584 to 368 bytes/token-row, exact recall at 173,058 tokens |
| Qwen QSA top-k budget 2,048 to 1,024 | 96.14 MiB/rank reclaimed, +8K context |
| Qwen runtime-bounded RoPE | 96 MiB/rank reclaimed, +11.2K context |
| Qwen direct calibrated E4M3 QSA | Native 262,144 context at near-BF16 performance |
| Qwen hierarchical AR | +15.0% decode, +10.8% concurrency-two aggregate |
| Qwen utilization 0.95 to 0.98 | 315K to 421,608 cache tokens with warmed stability |

# What did not work

The dead ends were useful because they removed broad categories of speculation.

| Idea | Why it failed |
| --- | --- |
| DeepSeek CPU/UVA weight offload | Decode fell to 12.68 to 19.54 tok/s |
| DeepSeek no-cache `wo_a` | Repeated dequantization cut decode to 34.01 tok/s |
| DeepSeek batch-token budget 128 | Prefill fell to 465.90 tok/s |
| DeepSeek llama.cpp MMVQ retuning | Correct variants projected below the 10% service threshold |
| DeepSeek uniform hot-expert cache | Only about 95% to 96% coverage at 224 experts |
| DeepSeek indexed-decode fusion | Trace showed only a 3.5% optimistic ceiling |
| DeepSeek DCP=4 | Correct at 148K but about 53% slower; 262K stress failed |
| DeepSeek Unsloth IQ1 variants | Functional, but decode and prefill were about half the Antirez path |
| Qwen no expert parallelism | Triton fallback, 82K context, 16.7% lower decode |
| Qwen generic hyperconnection compression | Every FP8/INT8/DP4A/IMMA route missed speed or quality gates |
| Qwen generic FP8/INT8/INT4 QSA | Correct implementations remained 2x to 28x too slow at key shapes |
| Qwen direct Q8-K/Q4-V Triton | 4.206x BF16 at M=1 and 1.717x at M=256 |
| Qwen compile-mode change | Performance wash |
| GPU clock locking | Fast, but rejected because it violated the safety policy |

# What we learned

## "Supported" is not an execution result

A package can advertise SM86, compile an SM86 cubin, and still dispatch the wrong path or run too slowly. We required the whole chain: source support, packaged code, runtime selection, numerical output, mechanism timing, and model-level behavior.

## Memory must be attributed by owner

The useful capacity wins were not "use less memory" ideas. They were named allocations: 688 MiB of retained BF16 `wo_a`, 96 MiB of excess Qwen RoPE, 96 MiB of QSA top-k workspace. Once the owner was known, the implementation could preserve the rest of the system.

## A smaller representation can be slower

This happened repeatedly. INT4 and INT8 caches saved bytes but paid for RHT, unpacking, scaling, split merges, and extra launches. Hyperconnection INT8 saved hundreds of MiB on paper but lost on skinny matrix shapes. Storage arithmetic is not a throughput model.

## Synthetic tests can agree on the same mistake

The IQ1 nibble-order bug is the clearest example. Kernel and test passed because both encoded sequential quartets. A replay against real Unsloth bytes and pinned llama.cpp exposed the true lane-interleaved contract. Independent oracles need independent assumptions, not only separate functions.

## One-turn quality checks miss state-machine bugs

The DSML stop bug survived ordinary generation and tool selection. It failed after a tool result entered history. That is why the acceptance ladder now includes post-tool continuation and, for model changes, an early coding-agent task.

## A negative trace can save more time than a positive microbenchmark

The fusion study is the best case. Source inspection suggested a 9% to 12% decode opportunity. The production-semantics trace cut the ceiling to 3.5%. We stopped before writing a kernel.

## The hardware topology belongs in the design

Hierarchical all-reduce worked because server60 has two PCIe islands. The same code would be pointless on NVSwitch and may be wrong for a different PCIe tree. Performance decisions belong to an instance, not a GPU model name.

# Where we landed

The durable result is not one perfect service. It is a set of measured choices.

For DeepSeek V4:

- specialized llama.cpp remains the proven high-context path at 430K reserved context and about 33 to 38 decode tokens/s;
- native Antirez GGUF-TP is the fast path at about 80 decode tokens/s;
- FP4 KV reached a validated 175K context but was not promoted because 27 MiB headroom was too small;
- IQ1 and DCP remain useful experimental branches, not defaults.

For Qwen3.8 Flash Next, the production service now uses:

- Intel AutoRound W4A16 weights;
- Primitive quantized CPU PLE;
- TP=4 plus EP=4;
- vision enabled;
- calibrated direct E4M3 QSA cache;
- runtime-bounded RoPE;
- `max_num_batched_tokens=1024`;
- `max_num_seqs=2`;
- island-aware hierarchical all-reduce over `0,1;2,3`;
- native 262,144-token context;
- `gpu_memory_utilization=0.98`;
- 421,608 aggregate KV tokens;
- zero swap and the fixed 230 W safety policy.

The remaining Qwen performance gap is not yet pinned to one mechanism. Mixed concurrent prefill still queues or starves decode, and an external four-3090 recipe remains faster under a different checkpoint, host-memory budget, and projection layout. The next useful optimization must start from a matched trace, not from copying that result.

That is the standard this thread established: measure the real cost, make one change, and keep the dead ends in the record.

## Evidence and deeper reports

DeepSeek V4 native GGUF reports in this repository:

- [runtime history](deepseek_v4_gguf_tp/DEEPSEEK-V4-GGUF-RUNTIME-HISTORY.md)
- [native engine implementation](deepseek_v4_gguf_tp/BLOG-POST.md)
- [first full-model acceptance](deepseek_v4_gguf_tp/M5-M7-RUNTIME.md)
- [agentic DeepSWE gate](deepseek_v4_gguf_tp/M8-DEEPSWE.md)
- [FP4 KV cache](deepseek_v4_gguf_tp/fp4_kv/REPORT.md)
- [DCP experiment](deepseek_v4_gguf_tp/DCP-SM86.md)
- [fusion no-go](deepseek_v4_gguf_tp/FUSION-TRACE.md)
- [cold-expert route study](deepseek_v4_gguf_tp/route-offload/ROUTE-OFFLOAD.md)

Qwen3.8 reports at immutable Whamp/vLLM commit `ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa`:

- [memory baseline](https://github.com/Whamp/vllm/blob/ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa/docs/whamp/qwen38_flash_next/GPU-MEMORY-BASELINE.md)
- [QSA top-k reduction](https://github.com/Whamp/vllm/blob/ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa/docs/whamp/qwen38_flash_next/QSA-TOPK-BUFFER-1024.md)
- [runtime-bounded RoPE](https://github.com/Whamp/vllm/blob/ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa/docs/whamp/qwen38_flash_next/QSA-ROPE-BOUND.md)
- [capacity kernel gates](https://github.com/Whamp/vllm/blob/ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa/docs/whamp/qwen38_flash_next/CAPACITY-KERNEL-GATES.md)
- [direct E4M3 QSA cache](https://github.com/Whamp/vllm/blob/ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa/docs/whamp/qwen38_flash_next/QSA-FP8-CACHE.md)
- [hierarchical all-reduce](https://github.com/Whamp/vllm/blob/ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa/docs/whamp/qwen38_flash_next/HIERARCHICAL-ALL-REDUCE.md)
- [0.98 production acceptance](https://github.com/Whamp/vllm/tree/ed3a839ae6e857aa2d315e2de7adf82c10f8c1fa/docs/whamp/qwen38_flash_next/evidence/qwen38-gpu-util-098-20260830)
