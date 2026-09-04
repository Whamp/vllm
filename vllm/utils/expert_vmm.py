# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Planning primitives for mixed GPU/host virtual-memory expert weights."""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from functools import cache
from pathlib import Path


@dataclass(frozen=True)
class ExpertPermutation:
    """Hot-first local expert numbering and its global routing table."""

    hot_local_ids: tuple[int, ...]
    new_to_old: tuple[int, ...]
    old_to_new: tuple[int, ...]
    expert_map: tuple[int, ...]


@dataclass(frozen=True)
class VMMTierBytes:
    """Driver-aligned physical memory assigned to one virtual tensor."""

    mapped_bytes: int
    device_bytes: int
    host_bytes: int


def _round_up(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


@cache
def load_expert_rankings(path: str | Path) -> Mapping[str, tuple[int, ...]]:
    """Load immutable expert rankings keyed by exact ``RoutedExperts`` prefix."""
    rankings_path = Path(path)
    raw_rankings = json.loads(rankings_path.read_text())
    if not isinstance(raw_rankings, dict):
        raise ValueError("expert rankings must be a JSON object")

    rankings: dict[str, tuple[int, ...]] = {}
    for layer_name, expert_ids in raw_rankings.items():
        if not isinstance(layer_name, str) or not layer_name:
            raise ValueError("expert ranking keys must be non-empty layer names")
        if not isinstance(expert_ids, list) or any(
            type(expert_id) is not int for expert_id in expert_ids
        ):
            raise ValueError(
                f"expert ranking for {layer_name!r} must contain integer expert IDs"
            )
        rankings[layer_name] = tuple(expert_ids)
    return rankings


def plan_expert_permutation(
    ranked_global_ids: Sequence[int],
    expert_map: Sequence[int],
    hot_experts: int,
) -> ExpertPermutation:
    """Plan a hot-first permutation while preserving global expert routing.

    ``expert_map`` maps global expert IDs to this rank's local IDs, with ``-1``
    for experts owned by another rank. Ranked local experts are selected first;
    unseen local experts fill any remaining hot capacity in local-ID order.
    """
    if hot_experts < 0:
        raise ValueError("hot_experts must be non-negative")

    local_ids = sorted(local_id for local_id in expert_map if local_id >= 0)
    expected_local_ids = list(range(len(local_ids)))
    if local_ids != expected_local_ids:
        raise ValueError("expert_map local IDs must be unique and contiguous from zero")

    local_experts = len(local_ids)
    hot_experts = min(hot_experts, local_experts)
    hot_local_ids: list[int] = []
    seen: set[int] = set()
    if hot_experts > 0:
        for global_id in ranked_global_ids:
            if global_id < 0 or global_id >= len(expert_map):
                continue
            local_id = expert_map[global_id]
            if local_id < 0 or local_id in seen:
                continue
            hot_local_ids.append(local_id)
            seen.add(local_id)
            if len(hot_local_ids) == hot_experts:
                break

    if len(hot_local_ids) < hot_experts:
        for local_id in range(local_experts):
            if local_id in seen:
                continue
            hot_local_ids.append(local_id)
            seen.add(local_id)
            if len(hot_local_ids) == hot_experts:
                break

    new_to_old = tuple(
        hot_local_ids
        + [local_id for local_id in range(local_experts) if local_id not in seen]
    )
    old_to_new_list = [0] * local_experts
    for new_local_id, old_local_id in enumerate(new_to_old):
        old_to_new_list[old_local_id] = new_local_id
    old_to_new = tuple(old_to_new_list)
    new_expert_map = tuple(
        -1 if old_local_id < 0 else old_to_new[old_local_id]
        for old_local_id in expert_map
    )
    return ExpertPermutation(
        hot_local_ids=tuple(hot_local_ids),
        new_to_old=new_to_old,
        old_to_new=old_to_new,
        expert_map=new_expert_map,
    )


def plan_vmm_tier_bytes(
    *,
    total_bytes: int,
    row_bytes: int,
    hot_experts: int,
    granularity: int,
) -> VMMTierBytes:
    """Split a tensor into driver-aligned device and host mappings."""
    if total_bytes <= 0:
        raise ValueError("total_bytes must be positive")
    if row_bytes <= 0:
        raise ValueError("row_bytes must be positive")
    if hot_experts < 0:
        raise ValueError("hot_experts must be non-negative")
    if granularity <= 0:
        raise ValueError("granularity must be positive")
    hot_bytes = hot_experts * row_bytes
    if hot_bytes > total_bytes:
        raise ValueError("hot expert rows exceed the tensor size")

    mapped_bytes = _round_up(total_bytes, granularity)
    device_bytes = min(_round_up(hot_bytes, granularity), mapped_bytes)
    return VMMTierBytes(
        mapped_bytes=mapped_bytes,
        device_bytes=device_bytes,
        host_bytes=mapped_bytes - device_bytes,
    )
