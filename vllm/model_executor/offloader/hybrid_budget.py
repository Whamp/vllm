# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Hybrid expert-cache budget and miss-split planning.

Two pure-arithmetic decisions for hybrid MoE execution, where a decode
step's expert-cache misses can either stream weights from host memory to
the GPU cache or execute directly on the CPU:

- :func:`balanced_miss_split` divides the miss set so the H2D fetch path
  and the CPU execution path finish at the same time instead of
  serializing behind the slower one.
- :func:`plan_expert_kv_budget` divides GPU memory between expert-cache
  slots and KV pages with the KV context floor honored FIRST. An
  expert-first split is what lets a large auto-sized expert cache crowd a
  96 GB card down to a few thousand context tokens, so this planner
  guarantees the requested context capacity before spending anything on
  experts.
"""

from __future__ import annotations

from vllm.model_executor.offloader.bandwidth_profile import HybridBandwidthProfile


def balanced_miss_split(
    num_missed_experts: int, bytes_per_expert: int, profile: HybridBandwidthProfile
) -> tuple[int, int]:
    """Split cache misses into ``(fetched_to_gpu, executed_on_host)``.

    The split equalizes transfer time and host execution time: fetching
    ``f`` experts takes ``f * bytes_per_expert / pcie_h2d_gbps`` while
    hosting the other ``h`` takes ``h * bytes_per_expert / host_moe_gbps``,
    and both paths run concurrently. Returns counts summing to
    ``num_missed_experts``.
    """
    if num_missed_experts < 0:
        raise ValueError("num_missed_experts must be non-negative")
    if num_missed_experts == 0:
        return 0, 0
    fetched = _balanced_fetch_count(num_missed_experts, profile)
    return fetched, num_missed_experts - fetched


def _balanced_fetch_count(
    num_missed_experts: int, profile: HybridBandwidthProfile
) -> int:
    """Fetch count equalizing H2D transfer and host execution time."""
    fetch_fraction = profile.pcie_h2d_gbps / (
        profile.pcie_h2d_gbps + profile.host_moe_gbps
    )
    return max(0, min(num_missed_experts, round(num_missed_experts * fetch_fraction)))


def split_miss_keys(
    missed_expert_keys: list[tuple[int, int]], profile: HybridBandwidthProfile
) -> tuple[list[tuple[int, int]], list[tuple[int, int]]]:
    """Partition ``(layer, expert)`` miss keys into fetch and host lists.

    The fetch-list length comes from :func:`balanced_miss_split`; keys
    are taken in sorted order so callers get a deterministic plan for a
    given miss set. This is the seam contract the device-side wrapper
    will implement: classify misses, then route each key to either the
    H2D gather or the host execution path.
    """
    fetched_count = _balanced_fetch_count(len(missed_expert_keys), profile)
    ordered = sorted(missed_expert_keys)
    return ordered[:fetched_count], ordered[fetched_count:]


def plan_expert_kv_budget(
    *,
    budget_bytes: int,
    per_expert_bytes: int,
    kv_page_bytes: int,
    num_experts: int,
    kv_floor_pages: int,
    min_expert_slots: int = 0,
) -> tuple[int, int]:
    """Divide ``budget_bytes`` between KV pages and expert-cache slots.

    The KV pool receives at least ``kv_floor_pages`` before any expert
    spend; experts then take the remainder up to ``num_experts``, and any
    leftover after the expert cap flows back into additional KV pages.
    Returns ``(expert_slots, kv_pages)`` whose footprint fits the budget.

    ``min_expert_slots`` is enforced only when affordable without touching
    the KV floor: a slot demand that cannot fit relaxes to what remains
    after the floor instead of evicting context capacity.
    """
    if per_expert_bytes <= 0 or kv_page_bytes <= 0:
        raise ValueError("per_expert_bytes and kv_page_bytes must be positive")
    if num_experts < 0 or kv_floor_pages <= 0 or min_expert_slots < 0:
        raise ValueError(
            "num_experts must be non-negative; kv_floor_pages and "
            "min_expert_slots must be positive"
        )

    kv_floor_bytes = kv_floor_pages * kv_page_bytes
    if budget_bytes < kv_floor_bytes:
        raise ValueError(
            f"budget {budget_bytes} B is below the required KV context floor "
            f"({kv_floor_pages} pages x {kv_page_bytes} B = {kv_floor_bytes} B); "
            "raise the budget or lower kv_floor_pages"
        )

    remaining = budget_bytes - kv_floor_bytes
    slots = min(remaining // per_expert_bytes, num_experts)
    if slots < min_expert_slots:
        slots = min(min_expert_slots, remaining // per_expert_bytes)

    leftover = remaining - slots * per_expert_bytes
    pages = kv_floor_pages + leftover // kv_page_bytes
    assert slots * per_expert_bytes + pages * kv_page_bytes <= budget_bytes
    return slots, pages
