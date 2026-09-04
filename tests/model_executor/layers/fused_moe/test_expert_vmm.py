# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import itertools

import pytest

from vllm.utils.expert_vmm import (
    load_expert_rankings,
    plan_expert_permutation,
    plan_vmm_tier_bytes,
)


def test_expert_rankings_require_an_exact_layer_prefix(tmp_path):
    rankings_path = tmp_path / "rankings.json"
    rankings_path.write_text('{"language_model.model.layers.0.mlp.experts": [7, 3, 1]}')

    rankings = load_expert_rankings(rankings_path)

    assert rankings["language_model.model.layers.0.mlp.experts"] == (7, 3, 1)
    assert "language_model.mtp.layers.0.mlp.experts" not in rankings


def test_expert_rankings_reject_non_integer_ids(tmp_path):
    rankings_path = tmp_path / "rankings.json"
    rankings_path.write_text('{"language_model.model.layers.0.mlp.experts": [1, "2"]}')

    with pytest.raises(ValueError, match="integer expert IDs"):
        load_expert_rankings(rankings_path)


def test_expert_permutation_prioritizes_ranked_local_experts_then_fills_capacity():
    plan = plan_expert_permutation(
        ranked_global_ids=[4, 1, 4, 99, -1],
        expert_map=[-1, 2, 0, -1, 1],
        hot_experts=3,
    )

    assert plan.hot_local_ids == (1, 2, 0)
    assert plan.new_to_old == (1, 2, 0)
    assert plan.old_to_new == (2, 0, 1)
    assert plan.expert_map == (-1, 1, 2, -1, 0)


def test_expert_permutation_rejects_non_contiguous_local_ids():
    with pytest.raises(ValueError, match="contiguous"):
        plan_expert_permutation(
            ranked_global_ids=[0, 1],
            expert_map=[0, 2],
            hot_experts=1,
        )


def test_expert_permutation_preserves_global_routing_exhaustively():
    """Every small valid EP map still resolves each global expert to itself.

    This exhaustively enumerates the finite domain with up to five global
    experts, every non-empty local subset, every local numbering, every global
    ranking, and every valid hot capacity. It is a property search rather than
    a table of selected examples.
    """
    for global_experts in range(1, 6):
        global_ids = tuple(range(global_experts))
        for local_experts in range(1, global_experts + 1):
            for local_globals in itertools.combinations(global_ids, local_experts):
                for local_numbering in itertools.permutations(range(local_experts)):
                    expert_map = [-1] * global_experts
                    for global_id, local_id in zip(
                        local_globals, local_numbering, strict=True
                    ):
                        expert_map[global_id] = local_id
                    for ranking in itertools.permutations(global_ids):
                        for hot_experts in range(local_experts + 1):
                            plan = plan_expert_permutation(
                                ranking, expert_map, hot_experts
                            )

                            assert sorted(plan.new_to_old) == list(range(local_experts))
                            assert len(plan.hot_local_ids) == hot_experts
                            assert plan.new_to_old[:hot_experts] == plan.hot_local_ids
                            for global_id, old_local_id in enumerate(expert_map):
                                if old_local_id < 0:
                                    assert plan.expert_map[global_id] == -1
                                    continue
                                new_local_id = plan.expert_map[global_id]
                                assert plan.new_to_old[new_local_id] == old_local_id


def test_vmm_tier_bytes_rounds_each_mapping_to_driver_granularity():
    tier = plan_vmm_tier_bytes(
        total_bytes=300 * 1024**2,
        row_bytes=2340 * 1024,
        hot_experts=110,
        granularity=2 * 1024**2,
    )

    assert tier.mapped_bytes == 300 * 1024**2
    assert tier.device_bytes == 252 * 1024**2
    assert tier.host_bytes == 48 * 1024**2
    assert tier.device_bytes + tier.host_bytes == tier.mapped_bytes


@pytest.mark.parametrize(
    ("total_bytes", "row_bytes", "hot_experts", "granularity"),
    [
        (0, 1, 0, 1),
        (1, 0, 0, 1),
        (1, 1, -1, 1),
        (1, 1, 0, 0),
        (8, 4, 3, 2),
    ],
)
def test_vmm_tier_bytes_rejects_invalid_geometry(
    total_bytes, row_bytes, hot_experts, granularity
):
    with pytest.raises(ValueError):
        plan_vmm_tier_bytes(
            total_bytes=total_bytes,
            row_bytes=row_bytes,
            hot_experts=hot_experts,
            granularity=granularity,
        )
