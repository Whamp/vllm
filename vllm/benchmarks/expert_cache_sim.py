# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Offline expert-cache policy simulation over captured routing traces.

Replays a sequence of routed-expert touch sets against candidate cache
policies — LRU, FIFO, LFU, LFRU (frequency-protected LRU), and the Belady
oracle upper bound — and reports hit rates, so cache-policy questions are
answered from captured traces before any GPU experiment is run.

A trace is an ``.npz`` file with one array: ``routing_data`` of shape
``(num_steps, num_layers, num_experts_per_tok)`` holding per-layer expert
ids for every scheduled step. This is the per-session concatenation of
:class:`~vllm.v1.outputs.RoutedExpertsLists` batches. Cache keys are
``(layer_index, expert_id)`` pairs because expert ids repeat across
layers while the weights differ. Within a step all layer lookups count
against the pre-step cache state and misses are inserted as one batch,
so intra-step ordering never distorts results; a step whose distinct
misses exceed the cache capacity is treated as streaming traffic and
inserts nothing.
"""

from __future__ import annotations

from bisect import bisect_right
from dataclasses import dataclass
from pathlib import Path

import numpy as np

POLICY_NAMES = ("lru", "fifo", "lfu", "lfru", "belady")
_LFRU_HOT_HITS = 2


def synth_trace(
    num_steps: int,
    num_layers: int,
    num_experts: int,
    top_k: int,
    hot_fraction: float,
    seed: int,
) -> np.ndarray:
    """Synthetic decode-step routing: a hot expert subset plus uniform
    cold traffic, parameterized by ``hot_fraction``."""
    rng = np.random.default_rng(seed)
    num_hot = max(1, int(num_experts * hot_fraction))
    hot_ids = rng.choice(num_experts, size=num_hot, replace=False)
    from_hot = rng.choice(num_hot, size=(num_steps, num_layers, top_k))
    routing = hot_ids[from_hot]
    cold_mask = rng.random((num_steps, num_layers, top_k)) < 1.0 - hot_fraction
    cold_ids = rng.integers(0, num_experts, size=routing.shape)
    return np.where(cold_mask, cold_ids, routing).astype(np.int32)


def format_results_table(results: list[PolicySimResult], num_slots: int) -> str:
    """Markdown table of simulation results, ordered by hits descending."""
    lines = [
        "| policy | slots | touches | hits | misses | hit rate |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for r in sorted(results, key=lambda x: -x.hits):
        lines.append(
            f"| {r.policy} | {r.num_slots} | {r.touches} | {r.hits} | "
            f"{r.misses} | {r.hit_rate:.4f} |"
        )
    return "\n".join(lines) + "\n"


@dataclass(frozen=True)
class RoutingTrace:
    """Captured routed experts; see module docstring for shapes."""

    routing_data: np.ndarray


@dataclass(frozen=True)
class PolicySimResult:
    """Hit/miss totals for one policy over one trace."""

    policy: str
    num_slots: int
    touches: int
    hits: int

    @property
    def misses(self) -> int:
        return self.touches - self.hits

    @property
    def hit_rate(self) -> float:
        return self.hits / self.touches if self.touches else 0.0


def write_routing_trace(path: Path, routing_data: np.ndarray) -> None:
    """Persist a 3-D ``(num_steps, num_layers, top_k)`` expert-id array."""
    if routing_data.ndim != 3:
        raise ValueError(
            f"routing_data must be 3-D (num_steps, num_layers, top_k), "
            f"got shape {routing_data.shape}"
        )
    np.savez(path, routing_data=routing_data.astype(np.int32))


def load_routing_trace(path: Path) -> RoutingTrace:
    """Load a trace written by :func:`write_routing_trace`."""
    with np.load(path) as data:
        if "routing_data" not in data:
            raise ValueError(
                f"trace file {path} is missing 'routing_data'; regenerate "
                "the capture with write_routing_trace"
            )
        routing_data = data["routing_data"]
        if (rank := routing_data.ndim) != 3:
            raise ValueError(
                f"trace file {path}: expected rank 3 "
                f"(steps, layers, top_k) routing_data, got rank {rank}"
            )
        if not np.issubdtype(routing_data.dtype, np.integer):
            raise ValueError(
                f"trace file {path}: expected integer expert ids, got "
                f"dtype {routing_data.dtype}"
            )
        return RoutingTrace(routing_data=routing_data)


def simulate_expert_cache(
    routing_data: np.ndarray,
    num_slots: int,
    policy: str,
) -> PolicySimResult:
    """Replay ``routing_data`` against ``policy`` with a global cache of
    ``num_slots`` expert slots."""
    if policy not in POLICY_NAMES:
        raise ValueError(
            f"unknown expert-cache policy {policy!r}; expected one of "
            f"{', '.join(POLICY_NAMES)}"
        )
    if num_slots <= 0:
        raise ValueError("num_slots must be positive")

    steps = _step_touch_sets(routing_data)
    future_uses = _next_use_steps(routing_data) if policy == "belady" else None

    cached: set[tuple[int, int]] = set()
    first_used_step: dict[tuple[int, int], int] = {}
    last_used_step: dict[tuple[int, int], int] = {}
    use_count: dict[tuple[int, int], int] = {}

    hits = 0
    touches = 0
    top_k = routing_data.shape[2]
    flat_steps = routing_data.reshape(routing_data.shape[0], -1)
    for step_index, (row, touched) in enumerate(zip(flat_steps, steps)):
        # Every layer lookup counts against the pre-step cache state.
        hits += sum(
            1
            for position, expert_id in enumerate(row)
            if (position // top_k, int(expert_id)) in cached
        )
        touches += row.size

        # A step needing more distinct experts than the cache holds is
        # served streaming: count its lookups but insert nothing.
        if len(touched) > num_slots:
            continue

        missed = sorted(touched - cached)

        inserted_this_step: set[tuple[int, int]] = set()
        for expert_key in missed:
            while len(cached) + 1 > num_slots:
                victim_expert = _choose_victim(
                    policy,
                    cached=cached,
                    first_used_step=first_used_step,
                    last_used_step=last_used_step,
                    use_count=use_count,
                    step_index=step_index,
                    protected=inserted_this_step,
                    next_use_steps=future_uses,
                )
                cached.remove(victim_expert)
            cached.add(expert_key)
            inserted_this_step.add(expert_key)
        for expert_key in touched:
            first_used_step.setdefault(expert_key, step_index)
            last_used_step[expert_key] = step_index
            use_count[expert_key] = use_count.get(expert_key, 0) + 1

    return PolicySimResult(
        policy=policy,
        num_slots=num_slots,
        touches=touches,
        hits=hits,
    )


def _step_touch_sets(
    routing_data: np.ndarray,
) -> list[frozenset[tuple[int, int]]]:
    """Distinct ``(layer_index, expert_id)`` cache keys touched per step.

    Expert ids are per-layer: layer 3's expert 7 and layer 9's expert 7
    are different weights and occupy different cache slots, so the layer
    index participates in every key. Entries of the same (layer, expert)
    pair within one step collapse to one key; lookups are still counted
    per raw entry in :func:`simulate_expert_cache`.
    """
    top_k = routing_data.shape[2]
    flat = routing_data.reshape(routing_data.shape[0], -1)
    steps: list[frozenset[tuple[int, int]]] = []
    for row in flat:
        keys = {
            (position // top_k, int(expert_id))
            for position, expert_id in enumerate(row)
        }
        steps.append(frozenset(keys))
    return steps


def _next_use_steps(
    routing_data: np.ndarray,
) -> dict[tuple[int, int], list[int]]:
    """Per ``(layer, expert)`` key, the sorted steps using it (oracle)."""
    top_k = routing_data.shape[2]
    flat = routing_data.reshape(routing_data.shape[0], -1)
    uses: dict[tuple[int, int], list[int]] = {}
    for step_index, row in enumerate(flat):
        for position, expert_id in enumerate(row):
            key = (position // top_k, int(expert_id))
            uses.setdefault(key, []).append(step_index)
    return uses


def _choose_victim(
    policy: str,
    *,
    cached: set[tuple[int, int]],
    first_used_step: dict[tuple[int, int], int],
    last_used_step: dict[tuple[int, int], int],
    use_count: dict[tuple[int, int], int],
    step_index: int,
    protected: set[tuple[int, int]],
    next_use_steps: dict[tuple[int, int], list[int]] | None,
) -> tuple[int, int]:
    """Pick the cached ``(layer, expert)`` key to evict per ``policy``.

    ``protected`` holds experts inserted earlier in the current step's
    miss batch; they are never eviction candidates.
    """
    candidates = [e for e in cached if e not in protected]
    assert candidates, (
        "every cached expert was inserted in the current miss batch; "
        "num_slots is smaller than one step's distinct misses"
    )

    def least_recently_used(pool: list[tuple[int, int]]) -> tuple[int, int]:
        return min(pool, key=lambda e: last_used_step[e])

    if policy == "lru":
        return least_recently_used(candidates)
    if policy == "fifo":
        return min(candidates, key=lambda e: first_used_step[e])
    if policy == "lfu":
        return min(candidates, key=lambda e: (use_count[e], last_used_step[e]))
    if policy == "lfru":
        cold = [e for e in candidates if use_count[e] < _LFRU_HOT_HITS]
        return least_recently_used(cold or candidates)
    if policy == "belady":
        assert next_use_steps is not None

        def next_use(expert_key: tuple[int, int]) -> float:
            steps_list = next_use_steps[expert_key]
            pos = bisect_right(steps_list, step_index)
            return steps_list[pos] if pos < len(steps_list) else float("inf")

        return max(candidates, key=next_use)
    raise AssertionError(f"unhandled expert-cache policy {policy!r}")
