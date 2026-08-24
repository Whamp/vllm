# Expert-cache hybrid execution planning

Design notes for measurement-driven MoE expert-cache decisions: how to
divide GPU memory between an expert cache and the KV pool, how to divide
a decode step's cache misses between H2D fetches and host-side expert
execution, and how to choose a cache eviction policy from captured
routing traces before running any GPU experiment.

Provenance: these mechanisms were evaluated against
[FreeToken](https://github.com/FlashML-org/FreeToken), which pairs a GPU
expert cache with direct CPU expert execution for consumer-GPU MoE
serving. The ideas adopted here are its bandwidth-calibrated scheduling
and joint memory budgeting; the implementation is new and deliberately
diverges where FreeToken's operational behavior is unsound.

## Bandwidth profiles

`vllm/model_executor/offloader/bandwidth_profile.py` defines
`HybridBandwidthProfile`, a JSON artifact recording two measurements:

- `host_moe_gbps`: CPU expert-execution throughput for one expert weight
  format (`quant_format`).
- `pcie_h2d_gbps`: pinned-memory H2D gather bandwidth over one
  interconnect.

Both are meaningful only in combination with the hardware they were
measured on, so the artifact carries the full identity (`gpu_name`,
`cpu_model`, `interconnect`, `quant_format`) and
`profile_matches_hardware()` requires **every supplied identity field**
to match before a plan may trust the numbers. A profile benched on an
EPYC workstation must not silently set fetch/host policy for a desktop
Ryzen: CPU-execution throughput varies by more than the entire PCIe
bandwidth across host CPUs, so keying on GPU name alone (as FreeToken
does) can invert the decision.

Measurements come from existing tooling (`ft bench bw`-style microbench:
streaming expert math from page-locked host memory for the first figure,
a pinned H2D gather of expert-sized blocks for the second). The profile
format is engine-neutral; producing one is a benchmark task, not a
serving-time task.

## Balanced miss split

`vllm/model_executor/offloader/hybrid_budget.py::balanced_miss_split`
answers the per-step question: of the experts a decode step needs that
are not cached, how many should stream over PCIe into the GPU cache and
how many should execute directly on the CPU?

Fetching `f` experts costs `f * bytes_per_expert / pcie_h2d_gbps`;
executing the other `h` on the host costs `h * bytes_per_expert /
host_moe_gbps`. The two paths run concurrently, so total miss cost is
the max of the two and is minimized when they are equal, giving

    f = N * pcie_h2d_gbps / (pcie_h2d_gbps + host_moe_gbps)

for `N` misses. This is FreeToken's `q*` insight, kept as a pure
function so it is unit-testable without a device and reusable by any
caller that knows the miss set size.

## KV-floor-first budget planning

`plan_expert_kv_budget()` divides a GPU memory budget between
expert-cache slots and KV pages with an invariant FreeToken lacks: **the
requested context capacity is honored before any expert spend**.
FreeToken's auto-sizing fills expert slots greedily and hands KV the
remains, which left a 96 GB card with roughly 8k context tokens at
defaults and queued longer prompts forever
([FreeToken #111](https://github.com/FlashML-org/FreeToken/issues/111)).
Here the KV floor is subtracted first; experts take the remainder up to
`num_experts`; leftover after the expert cap flows back into extra KV
pages; a slot demand that cannot fit relaxes rather than evicting the
floor; and a budget below the floor raises instead of misconfiguring.

## Cache-policy simulation

`vllm/benchmarks/expert_cache_sim.py` replays captured routing traces
against LRU, FIFO, LFU, LFRU (frequency-protected LRU), and the Belady
oracle upper bound. Traces are `(num_steps, num_layers, top_k)` expert-id
arrays — the concatenation of `RoutedExpertsLists` batches this fork's
routed-experts capture already produces — and cache keys are
`(layer_index, expert_id)` pairs because expert ids repeat across layers
while weights differ.

Run it without a trace:

```sh
python benchmarks/expert_cache_policy_sim.py --slots 1500 --num-steps 600
```

which prints, for a seeded hot/cold synthetic workload (256 experts,
21 layers, top-8, 25% hot):

| policy | hits | hit rate |
| --- | ---: | ---: |
| belady | 65477 | 0.6496 |
| lfu | 41534 | 0.4120 |
| lfru | 32129 | 0.3187 |
| lru | 31892 | 0.3164 |
| fifo | 24481 | 0.2429 |

Findings so far, all on synthetic traffic:

- With capacity above the hot working set every policy saturates near
  100% and comparisons are meaningless; discrimination requires slot
  counts below the touched-key span of a step.
- Frequency-aware policies separate from recency-only ones only once
  traffic has a stable hot set; LFRU's fixed two-hit protection
  threshold underperforms plain LFU here because moderately-hot experts
  lose protection while still being reused.

Neither finding should be trusted beyond the synthetic generator. The
harness exists so real captured sessions decide the policy question.

## Runtime integration status

The planners above are complete and CPU-validated. What remains needs a
GPU and is intentionally not implemented blind:

1. Device-resident miss classification. During CUDA-graph capture the
   miss count exists only on-device, so the split decision must run in
   a graph-compatible kernel (FreeToken does this in Triton) or the
   graph must be cut at the decision point. Kernel choice and capture
   strategy require profiling on target hardware.
2. Host-side expert execution path. vLLM has no CPU MoE execution branch
   for offloaded experts today; UVA and prefetch offloading both move
   weights to the GPU. Adding one touches fused-MoE method dispatch and
   must be validated for numerical parity.
3. End-to-end latency evidence. The balanced split minimizes modeled
   miss time; whether the model holds depends on overlap between the
   two paths, measurable only with Nsight on hardware.

## Recurrent-state anchors for replay serving

Separate from expert caching, FreeToken checkpoints recurrent state at
tool-call boundaries rather than arbitrary positions. This matters to
this fork's relayered Qwen3.5 serving (`qwen3_5_replay.py`): each
replayed layer span occupies its own logical KV/GDN slot ids, so a
prefix-cache hit that reuses recurrent state must line up slot-for-slot
with the replay layout, and agent workloads rewrite prompts precisely at
tool-call boundaries — making those boundaries the highest-value
snapshot points.

Design sketch for a later change:

- Anchor selection: snapshot the per-slot GDN/recurrent state when a
  generated message ends a turn with a tool call (the assistant message
  plus tool results will be re-prefixed identically on the next turn).
- Slot alignment: snapshots record the full slot layout (original layers
  `0..L-1` plus each replay occurrence's contiguous block), so a resumed
  request validates its `replay_spans` against the snapshot before
  reuse; mismatched spans invalidate rather than partially reuse.
- Eviction: anchors live in host memory keyed by prefix hash like other
  prefix-cache blocks, evicted LRU; restoring streams state back into
  device pools at a scheduler idle point.

This is a design precedent, not implemented here; it interacts with
mamba prefix caching and the replay weight-tying contract and needs its
own correctness story (exact state equality after restore) before any
implementation.
