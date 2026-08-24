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
to match before a plan may trust the numbers (optionally including
``numa_node`` for multi-socket hosts, where locality shifts both
bandwidths). A profile benched on an
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

## GPU-window execution plan

The steps below are written so a future hardware session can execute
them directly, in order, with each step gated on the previous one's
evidence. Nothing here should be coded before its step's hardware
prerequisite exists.

### Step 1 — Miss-classification kernel (graph-compatible)

Contract (fixed here so the planner seam and kernel agree):

- Inputs: the active expert-id tensor after top-k rewrite
  `(num_slots_per_rank,)`-style layout used by fused MoE, a device-side
  cached-key set (sorted id array + count, updated in-graph), the
  balanced fetch count `f` from `balanced_miss_split` (host-computed per
  profile, passed as a captured scalar), and LRU metadata.
- Outputs: a per-miss fetch/host bitmap, rewritten expert ids (GPU slot
  or host-execution sentinel), an in-graph updated miss count for the
  scheduler, and eviction victims.

Two capture strategies to A/B, since they trade graph compatibility
against split accuracy:

1. **In-graph Triton kernel** (FreeToken's approach): classification,
   eviction, and rewrite all run inside capture; no graph cuts; the
   split is exact every step. Cost: sorted-set update and LRU bookkeeping
   must be atomic-friendly single-pass kernels — the main engineering
   risk.
2. **Graph cut at the decision point**: capture two subgraphs and choose
   between them on a host-synchronized miss count. Simpler kernels, but
   pays a sync per decode step and bounds the win at low batch exactly
   where hybrid execution matters.

Decision rule: implement 1 only if 2's measured per-step sync overhead
exceeds ~2% of step time at the target batch size; otherwise ship 2 and
revisit.

### Step 2 — Host execution path

Add a CPU branch to fused MoE dispatch that executes nonresident misses'
expert math on pinned host weights, producing partial outputs summed in
the same order as the all-GPU reference (expert-split rounding: sum per
expert then accumulate token-side, matching FreeToken's finding that
order determines bit-exactness). Gate: max-abs-diff parity vs the GPU
reference within quantization tolerance on randomized routing, plus
identical argmax/top-k behavior on downstream samples.

### Step 3 — Overlap and latency evidence

Nsight trace of the balanced split vs all-fetch and all-host baselines:
confirm H2D gather and host expert math actually overlap (the model
`balanced_miss_split` minimizes assumes concurrent paths), record
effective bandwidths, and regenerate the profile artifact from the same
machine so planning uses measured numbers. Report p50/p99 step time and
tok/s at batch sizes 1–8 against PR #51710 prefetch alone.

### Step 4 — Policy decision from real traces

Capture routed-expert traces (`RoutedExpertsLists` concatenation) from
representative agent sessions, replay through
`benchmarks/expert_cache_policy_sim.py`, and pick the production policy
on measured hit-rate deltas — not on the synthetic table above.

## Runtime integration status

The planners above are complete and CPU-validated. What remains —
device-resident miss classification, the host expert-execution path,
and overlap/latency evidence — needs a GPU and is intentionally not
implemented blind; each is specified step by step in the GPU-window
execution plan above, ordered by hardware prerequisite.

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
