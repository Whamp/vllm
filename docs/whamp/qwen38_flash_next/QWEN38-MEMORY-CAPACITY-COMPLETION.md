# Qwen3.8 memory-capacity completion audit

## Final result

The server60 Intel AutoRound Qwen3.8 service now runs at 202,400 fitted context,
up from 148,400 at the first healthy EP=4 baseline. The final profile keeps the
same model, quantized CPU PLE, vision stack, BF16 QSA cache, expert parallelism,
and CUDA-Graph mode.

Two persistent allocations were reduced:

- QSA top-k buffers: 100,810,752 bytes per rank;
- RoPE materialization: 100,663,296 bytes per rank.

Together they reclaim 201,474,048 bytes per rank, or 192.14 MiB. The accepted
executor budget then raises stable context to 202,400 while preserving a 1 GiB
physical safety margin after 190K NIAH.

## Requirement matrix

| Goal requirement | Evidence | Verdict |
| --- | --- | --- |
| Storage-deduplicated staged residency baseline | [GPU-MEMORY-BASELINE.md](GPU-MEMORY-BASELINE.md), 40 reports and deterministic analyzer under `evidence/qwen38-memory-baseline-20260829/` | Verified |
| Parameters, buffers, post-load state, activations, CUDA graphs, backend/non-Torch, allocator, and KV/cache families | Baseline stage/category tables plus residual accounting in the same report | Verified |
| Hyperconnection replication/padding first | [GPU-MEMORY-BASELINE.md](GPU-MEMORY-BASELINE.md) H1 and `evidence/qwen38-hyperconnection-fp8-20260829/` | Measured no-go |
| Reduce QSA/indexer allocation | [QSA-TOPK-BUFFER-1024.md](QSA-TOPK-BUFFER-1024.md), commit `2bb737583` | Promoted |
| Reduce another measured owner | [QSA-ROPE-BOUND.md](QSA-ROPE-BOUND.md), commit `f186b5e02` | Promoted |
| Evaluate Q8/FP8 cache | [QSA-FP8-CACHE.md](QSA-FP8-CACHE.md) and [QSA-INT8-CACHE.md](QSA-INT8-CACHE.md) | Measured no-go |
| Evaluate Q4 cache | [QSA-INT4-CACHE.md](QSA-INT4-CACHE.md), generated properties and GPU counterfeit matrix | Measured no-go |
| Preserve BF16 rollback | Every cache experiment used the unchanged BF16 service as automatic rollback; final profile still uses BF16 | Verified |
| Maximize stable context | [GPU-MEMORY-UTILIZATION-0968.md](GPU-MEMORY-UTILIZATION-0968.md); 0.97 rejected, 0.968 promoted | Verified stable ceiling |
| Per-rank and aggregate VRAM | Baseline reports, candidate/final GPU CSVs, final-state records | Verified |
| Fitted and validated context | 202,400 fitted; exact retrieval at 190,047 API prompt tokens | Verified |
| Physical headroom | 1,214 MiB on GPU 0 and 1,098 MiB on GPUs 1–3 after final quality validation | Verified |
| Single-stream decode | 43.5405 tok/s on exact final profile | Verified |
| Cache-busted prefill | 1,542.7705 tok/s on exact final profile | Verified |
| Concurrency up to supported maximum | Concurrency two passed at 53.2523 aggregate tok/s | Verified |
| Deterministic/API/streaming/tools/reasoning/multimodal | Exact-final acceptance JSON under `evidence/qwen38-util-0968-20260829/accepted-0968/` | Verified |
| Long-context NIAH | Exact `VIOLET ORBIT 9137` at 190,047 API prompt tokens | Verified |
| BenchLocal quality | Quick 26/30, ToolCall-15 12/15, InstructFollow-15 14/15 | Verified |
| Zero serving-process swap | Final-state artifact lists all 12 PIDs at `swap_kib=0` | Verified |
| CUDA-Graph behavior | Full-decode-only graphs reached readiness and passed runtime probes; rejected cache kernels also passed graph replay where applicable | Verified |
| SM86 dispatch and numerical parity for changed kernels | Accepted changes add no CUDA kernel. Rejected Q8/Q4 kernels ran on RTX 3090 SM86 and were numerically gated before performance rejection | Not applicable to accepted code; rejected paths measured |
| GPU safety policy | Final state records active safety service, 230 W, and clocks within the 210–1650 MHz policy | Verified |
| Exact artifact, PLE, and vision provenance | All Compose contracts retain pinned model/PLE revisions and vision; multimodal probes passed | Verified |
| Tested rollback | Every outage used a guarded rollback. Rejected 0.97 and one cleanup failure restored the prior healthy service | Verified |
| Focused and adjacent tests | 115 tests passed before review; moved final suites passed 101 tests, including 150 generated cases | Verified |
| Property discrimination | Packed-layout, RHT, sparse-attention, and MRoPE properties killed boundary and semantic counterfeits | Verified |
| Lint, format, type/static, secrets, diff, and repository checks | Applicable pre-commit hooks, gitleaks, diff checks, and CodeGraph signature checks passed; pre-existing QSA structural edges were accounted for | Verified |
| Independent review | Standards and Spec follow-up reviews both reported zero findings | Verified |
| Reproducible pushed implementation and evidence | Branch `feat/qwen38-memory-capacity` through commit `004847f75`, plus this audit commit | Verified after push |
| Healthy final server | Exact image, 202,400 context, zero restarts, zero swap, active safety, inactive rollback timer | Verified |

## Accepted changes

### QSA top-k allocation

Reducing `max_num_batched_tokens` from 2,048 to 1,024 removed exactly
100,810,752 bytes per rank and raised fitted context from 148,400 to 156,400.
The final one-variable A/B retained 99.71% decode and 94.70% prefill.

### Runtime-bounded RoPE

Qwen3-VL image/video positions were proven token bounded over 150 generated
cases. Materializing 262,144 instead of 1,048,576 rows removed exactly 96 MiB
per rank and raised fitted context to 167,600. Exact-final performance retained
98.06% decode and 99.42% prefill.

### Executor budget

The 0.97 arm reached 206,400 tokens but failed the preregistered 1 GiB floor by
4–8 MiB after 195K NIAH. The floor was not moved. The 0.968 arm reached 202,400,
passed 190K NIAH, and retained 1,098 MiB minimum free VRAM after the final
quality pack.

## Rejected paths

- Generic block-FP8 Marlin hyperconnections saved a modeled 0.492 GiB per rank
  but were 2.9–5.7 times slower than BF16 at exact shapes.
- Typed E4M3 cache did not compile on SM86 Triton. E5M2 failed the numerical
  bound. Software E4M3 was 28.46 times slower than BF16 at M=256.
- Per-token-head INT8 passed numerical and decode gates but was 26.32 times
  slower at M=256.
- Packed INT4 passed generated properties, numerical bounds, CUDA graphs, and
  M=256 performance. Its best M=1 path remained 2.03 times BF16 after matrix
  RHT; four split-K schedules did not improve it.
- `gpu_memory_utilization=0.97` passed behavior and performance but failed the
  physical-margin gate and rolled back.

## Why work stops here

The largest remaining registered owner is replicated hyperconnection storage.
The existing generic compressed kernel is decisively too slow. True TP sharding
would replace replication with new per-layer collective boundaries on the
server's PCIe topology. No source or measured path removes those collectives,
so implementing it would violate the goal's evidence-backed-change rule.

A faster Q4 cache would require a new integer attention algorithm that quantizes
queries and probably softmax probabilities. That adds new numerical boundaries
rather than extending the tested cache representation. It is a separate kernel
project, not a low-risk remaining step.

The 0.97 failure establishes that more executor budget cannot safely recover
context under the fixed 1 GiB margin. No measured low-risk owner remains within
this goal's constraints.

## Final service identity

- Image:
  `sha256:1b4577a1b6f11029bb0c06e8051b7a3b360b5834b65e84fae09ff2f5485c6c0b`
- Model: `qwen3.8-flash-next-intel-autoround-w4a16`
- Context: 202,400
- `gpu_memory_utilization`: 0.968
- `max_num_batched_tokens`: 1,024
- `max_num_seqs`: 2
- QSA cache: BF16
- RoPE rows: 262,144
- Swap: disabled, every serving process at zero
- Restarts: zero
- Rollback timer: inactive
