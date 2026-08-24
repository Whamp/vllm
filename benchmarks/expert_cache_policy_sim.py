# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Run expert-cache policy comparison over a captured routing trace.

Replays a trace (or a synthetic stand-in) against every supported cache
policy and prints a markdown hit-rate table, e.g.:

    python benchmarks/expert_cache_policy_sim.py \
        --trace routed_experts_session.npz --slots 4096

Without ``--trace``, a synthetic trace with configurable locality is
generated so the harness runs end-to-end anywhere.
"""

import argparse

from vllm.benchmarks.expert_cache_sim import (
    POLICY_NAMES,
    format_results_table,
    load_routing_trace,
    simulate_expert_cache,
    synth_trace,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trace", type=str, default=None, help="trace .npz")
    parser.add_argument("--slots", type=int, default=1024)
    parser.add_argument(
        "--num-steps", type=int, default=512, help="synthetic trace length"
    )
    parser.add_argument("--num-layers", type=int, default=21)
    parser.add_argument("--num-experts", type=int, default=256)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument(
        "--hot-fraction", type=float, default=0.25, help="hot expert share"
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if args.trace is not None:
        trace = load_routing_trace(args.trace).routing_data
        source = args.trace
    else:
        trace = synth_trace(
            args.num_steps,
            args.num_layers,
            args.num_experts,
            args.top_k,
            args.hot_fraction,
            args.seed,
        )
        source = "synthetic"
        print(
            f"synthetic trace: {args.num_steps} steps x {args.num_layers} "
            f"layers x top-{args.top_k}, {args.num_experts} experts, "
            f"hot_fraction={args.hot_fraction}"
        )

    results = [simulate_expert_cache(trace, args.slots, name) for name in POLICY_NAMES]
    print(f"source: {source}")
    print(format_results_table(results, args.slots), end="")


if __name__ == "__main__":
    main()
