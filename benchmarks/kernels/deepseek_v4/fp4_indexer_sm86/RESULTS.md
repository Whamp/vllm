# DeepSeek V4 SM86 MXFP4 indexer results

## Decision

The SM86 MXFP4 sparse-indexer cache is a validated opt-in capacity mode. It must not replace the existing FP8 indexer default.

It increases tested long-context capacity materially and preserves the focused behavioral-quality result. It also reduces prefill throughput, especially at deep context, and every tested capacity profile remains below the project's 1 GiB free-VRAM release gate.

## Compared configurations

Both arms use the same DeepSeek V4 GGUF-TP model, four RTX 3090 GPUs, TP=4, `fp4_ds_mla` for the main MLA cache with its retained BF16 RoPE section, `max_num_seqs=2`, `max_num_batched_tokens=256`, CUDA graphs, native Ampere FlashMLA, hierarchical all-reduce, and the 230 W / 210-1650 MHz GPU safety policy.

The only cache-format difference is the 21 compression-ratio-4 sparse-indexer caches:

- Baseline: 128 FP8 values plus one FP32 scale, 132 bytes per row.
- Candidate: 64 packed E2M1 bytes plus four UE8M0 scale bytes, 68 bytes per row.

The candidate uses dedicated SM86 Triton prefill and paged-decode logits kernels. Existing FP8 and DeepGEMM branches remain unchanged.

## Allocation accounting

### Indexer row and page

| Quantity | FP8 indexer | MXFP4 indexer | Change |
| --- | ---: | ---: | ---: |
| Semantic key width | 128 values | 128 values | Equal |
| Physical row | 132 bytes | 68 bytes | -48.5% |
| Real 64-token page | 8,448 bytes | 4,352 bytes | -48.5% |
| Allocated page | 8,704 bytes | 4,608 bytes | -47.1% |
| Per-layer logical bytes at 175K | 5,953,536 | 3,151,872 | -47.1% |
| All 21 indexer layers at 175K | 125,024,256 | 66,189,312 | -58,834,944 bytes |

The 256-byte page-padding quantum remains unchanged.

### Packed groups

The packed layout still has five groups and an 872,256-byte global block stride. Group 0 shrinks from 699,168 to 613,152 bytes per block, but another group still determines the global stride. The indexer row change therefore does not reduce bytes per allocated packed block.

The 175K request's logical model-length requirement falls from 478,230,912 to 419,395,968 bytes per rank, an exact 58,834,944-byte saving. Available KV-pool memory rises by 111,214,592 bytes, from 815,005,369 to 926,219,961 bytes per rank.

The accounting proves both changes but does not separately attribute the remaining 52,379,648-byte increase in available pool memory. Smaller FP4 indexer profiling and gather workspaces are a source-level candidate, not a measured attribution. Because available pool memory increases while the block stride stays fixed, vLLM allocates more 872,256-byte blocks.

### Runtime reconciliation

| Profile | Indexer | Available pool | Blocks | Allocated pool | KV tokens | Planned versus observed |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| 175K baseline | FP8 | 815,005,369 B | 934 | 814,687,104 B | 178,050 | exact |
| 175K candidate | MXFP4 | 926,219,961 B | 1,061 | 925,463,616 B | 202,260 | exact |
| 200K candidate | MXFP4 | 884,276,921 B | 1,013 | 883,595,328 B | 199,409 | exact |

At matched 175K, reported KV-token capacity rises **13.60%**. The larger configured context consumes more non-KV runtime memory, so the 200K profile has fewer available blocks than the 175K profile.

## Long-context correctness

Before the indexer port, the 175K baseline recalled the exact needle at 173,058 prompt tokens. The 200K MXFP4-indexer candidate recalled exact needles at 194,812 and 195,812 prompt tokens. This is a **13.15%** increase in the highest directly validated prompt length and a **14.29%** increase in configured context.

The 200K profile is a functional capacity ceiling, not a release-safe operating point:

- 25-26 MiB free VRAM per RTX 3090 after near-ceiling work
- zero serving-process swap after normalization
- zero VRAM growth across the ceiling ladder
- exact needle recall through 195,812 tokens

The normal release gate requires 1 GiB free VRAM per card. None of the pre-indexer 160K, 170K, or 175K profiles, nor the post-indexer 175K or 200K profiles, meets that margin.

## Performance

Canonical protocol: three warmups and five measured narrative/code runs, followed by three cache-busted runs at each prefill depth.

| Metric | FP8-indexer baseline | MXFP4-indexer candidate | Change |
| --- | ---: | ---: | ---: |
| Narrative decode | 80.36 tok/s | 77.32 tok/s | -3.78% |
| Code decode | 80.37 tok/s | 77.32 tok/s | -3.80% |
| 10K prefill | 524.87 tok/s | 499.84 tok/s | -4.77% |
| 90K prefill | 495.79 tok/s | 348.35 tok/s | -29.74% |
| Concurrency-2 aggregate | 127.27 tok/s | 122.30 tok/s | -3.91% |
| Concurrency-2 per stream | 63.74 tok/s | 61.24 tok/s | -3.92% |
| Post-run VRAM growth | 0 MiB | 0 MiB | equal |

The deep-prefill regression is the main reason not to promote MXFP4 indexer caching as the default. The kernel is correctness-oriented SM86 enablement rather than a tuned prefill implementation. The concurrency-2 comparison uses the identical three-warmup/five-measured 512-token pair harness in both arms; every measured completion ended by the length cap.

## Behavioral quality

The matched BenchLocal quick gate produced:

| Indexer | Pass@1 | Pass@3 |
| --- | ---: | ---: |
| FP8 baseline | 27/30 | 27/30 |
| MXFP4 candidate | 26/30 | 27/30 |

The candidate had one additional pass@1 miss, while pass@3 remained equal. A single 30-case sampled run does not establish a quality difference. This result supports opt-in testing but not a stronger quality-equivalence claim.

## Kernel and runtime gates

Passed on RTX 3090 SM86:

- E2M1/UE8M0 numerical comparison using a table-distance, ties-to-even reference independent of the production threshold cascade
- Partial sequence, non-block-aligned context, `next_n=4`, and paged block-table cases
- Top-k set and pairwise-order comparison, downstream gathered-output comparison, and a tied-boundary case
- Software E2M1 query and cache writers
- Deterministic CUDA-Graph replay
- Targeted paged-decode and fused-query-writer Compute Sanitizer memcheck: zero errors
- Targeted paged-decode and fused-query-writer Compute Sanitizer racecheck: zero hazards
- Runtime-generated `_mxfp4_mqa_logits_kernel.sm_86.cubin` and `_mxfp4_paged_mqa_logits_kernel.sm_86.cubin`
- Full TP=4 model load and runtime dispatch
- `verify-full.sh`
- `verify-stress.sh`, including tool, reasoning, coding-agent, and exact long-context integration probes; NIAH is integration evidence, not a numerical-format oracle

The MXFP4 logits kernels reuse the inherited FP8 autotune configuration and are not described as tuned SM86 kernels. The measured deep-prefill regression remains the performance evidence for that limitation.

## Release recommendation

Keep the FP8 indexer as the production default. Ship the MXFP4 indexer as an explicit, documented capacity experiment for users who value approximately 13% more validated context enough to accept about 4% decode/shallow-prefill loss and about 30% deep-prefill loss.

Do not describe the 200K profile as release-safe. Its 25-26 MiB physical VRAM margin is a measured ceiling. A future default requires either a tuned MXFP4 prefill kernel or another VRAM reclaim that restores the 1 GiB margin without sacrificing the demonstrated context gain.

## Evidence

- `evidence/capacity_before_indexer/`: 160K, 170K, and 175K startup, allocation, and stress evidence.
- `evidence/baseline_comparison/`: exact pre-indexer benchmark, BenchLocal result, image identity, and launch provenance.
- `evidence/after_indexer/`: kernel, sanitizer, cubin, TP=4 startup, allocation, quality, benchmark, 175K stress, 200K stress, swap, safety, and release evidence.
- Each evidence directory contains a `SHA256SUMS` manifest.
