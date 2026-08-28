<!-- markdownlint-disable MD060 -->

# GGUF-TP decode fusion trace

Status: complete, measured no-go. The preregistered launch/dependency-latency premise is falsified. No fusion kernel was implemented or shipped.

## Decision

The production-semantic TP=4 decoder-layer graph is kernel and collective dominated, not launch-gap dominated.

| Stable replay metric | Median | Range |
| --- | ---: | ---: |
| First graph node to final graph node | 182.529 µs | 180.835–184.546 µs |
| GPU busy-time union | 181.793 µs | 180.099–183.810 µs |
| Internal idle time | 0.736 µs | 0.736–0.736 µs |
| Gap before the next graph replay | 4.576 µs | 4.480–4.864 µs |
| First-node to next first-node period | 187.090 µs | 185.411–189.058 µs |

The preregistration expected 60–100 µs/layer between summed standalone kernels and the measured layer slice, substantially caused by launch or dependency latency. The corrected trace finds 0.736 µs inside the graph and 4.576 µs between graph replays. CUDA Graph node gaps inside a replay are generally below 0.1 µs.

That result fails the explicit implementation gate. Rewriting the indexed IQ2/Q2 kernels would test a different hypothesis, so this bounded pass stops without a production change.

## Correction to the first trace

The initial trace used the historical M2 layer-slice harness from Whamp/vLLM `0ef05fe53`. It reconstructed 23 graph nodes and correctly falsified the large launch-gap premise, but a source audit caught two synthetic-tail differences before any fusion implementation:

- the benchmark expressed shared-expert clamped SwiGLU as six eager PyTorch kernels, while production `DeepseekV4MLP` already uses `SiluAndMulWithClamp`;
- the benchmark converted routed and shared outputs to FP32 before adding and casting, while production converts the routed reduction to BF16 and performs one BF16 add.

Whamp/vLLM `6f4f658ab` corrects the benchmark to the production operation and dtype contracts. The corrected 17-node trace is the decision record. The original 23-node trace remains archived as historical evidence rather than being silently replaced.

## Where the corrected layer time goes

Median node-duration sums across 49 stable replays on each of four GPUs:

| Group | Time | Share of 182.529 µs graph span |
| --- | ---: | ---: |
| Six dense Q8 Marlin projections | 84.160 µs | 46.1% |
| Two hierarchical all-reduces | 45.409 µs | 24.9% |
| Indexed IQ2 gate/up and Q2 down | 38.304 µs | 21.0% |
| Original F1+F2 removable standalone nodes | 10.496 µs | 5.8% |
| Existing fused shared activation | 1.376 µs | 0.8% |
| Existing BF16 routed/shared add | 1.504 µs | 0.8% |

The original F1+F2 upper bound is the complete duration of the five standalone nodes it proposed to absorb: routed input quantization, weighted SwiGLU requantization, top-k reduction, routed FP32-to-BF16 conversion, and routed/shared BF16 addition. Their median sum is 10.496 µs. Real savings must be lower because the fused producer kernels still have to perform that arithmetic and may pay additional register or occupancy cost.

Deleting all 10.496 µs from all 43 layers would save 0.451 ms/token before any epilogue cost. Against the M2 decode budget of about 13.04 ms/token, that is a 3.5% optimistic whole-token ceiling. The measured target is therefore too small for the proposed multi-kernel rewrite under this pass's evidence gate.

The largest remaining pools are dense Marlin execution and hierarchical collectives. They are real kernel/communication work, not launch gaps, and neither is a legal substitute for the preregistered numerics-preserving fusion hypothesis.

## Method

The final trace used the production-semantic TP=4 graph-captured layer-slice harness at Whamp/vLLM `6f4f658ab` and capture image `sha256:5fab88440740a6033bcacda473ffaeed7a4f4e386d494b516432487f0df09729` on server60.

Configuration:

- four RTX 3090 GPUs under the unchanged 230 W / 210–1650 MHz safety policy;
- `VLLM_HIER_ALL_REDUCE=0,1;2,3`;
- vLLM custom all-reduce disabled on the PCIe-only topology;
- 10 warmup graph replays;
- 50 captured indexed-decode replays per rank;
- CUDA Graph trace granularity `node`;
- Nsight Systems 2025.3.1;
- grouped M=256 prefill run after capture, unchanged from M2.

The first replay on each rank is excluded because the profiler-start barrier perturbs its first collective. The stable sample contains 196 complete layer replays. The profiled benchmark reported 0.1893–0.1905 ms/layer across ranks; timeline first-node spans are lower because event timing includes graph-launch and surrounding measurement overhead.

The corrected grouped M=256 path remained functional at 9.93–10.13 ms/layer in the one-iteration post-capture check. Those timings are smoke evidence, not a performance comparison.

## Evidence

`evidence/fusion-trace-20260820/` contains both trace generations:

- root files: original 23-node synthetic-tail trace and analysis;
- `production-semantics/`: authoritative 17-node report, SQLite timeline, `analysis.json`, standard Nsight summaries, rank results, logs, and exact corrected harness;
- `analyze_trace.py`: deterministic analyzer with explicit `synthetic-v1` and `production-v2` profiles;
- `SHA256SUMS`: checksum manifest for the complete compact archive.

Both 2.5 GB target-side `.qdstrm` streams were deleted only after successful report import and SQLite export. The compact `.nsys-rep` and SQLite files preserve the queryable evidence.

## Acceptance accounting

- Trace and attribution: **passed**, and premise falsified.
- Fusion implementation: **not entered**, as required by the failed gate.
- New-kernel numerical, graph, sanitizer, cubin, layer-oracle, and serving A/B gates: **not applicable because no kernel or runtime path changed**.
- Model bytes, grouped prefill, Q8 KV, 148K context, maximum sequence count, and production Compose: **unchanged**.
- Final service state: **passed**. `dsv4-gguf-tp-prod` is healthy with zero restarts, restart policy `unless-stopped`, exact image `sha256:f91e8283e7ad116b8664b4a936dba88ebafcb8910a968dce2a3c34420f010adf`, exact model identity, deterministic `PILOT-READY-8421` output, idle scheduler/KV metrics, zero swap for every container process, one DeepSeek container, no restore timer, and the active 230 W / 210–1650 MHz safety policy. Physical headroom remains the previously documented 99–100 MiB per GPU.
