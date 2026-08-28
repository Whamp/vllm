# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

import vllm.v1.worker.block_table as block_table_module
from vllm.v1.worker.block_table import MultiGroupBlockTable


def test_multigroup_block_table_replicates_dcp_exempt_groups(monkeypatch) -> None:
    group = SimpleNamespace(world_size=4, rank_in_group=2)
    monkeypatch.setattr(block_table_module, "get_dcp_group", lambda: group)
    monkeypatch.setattr(
        block_table_module,
        "get_pcp_group",
        lambda: SimpleNamespace(world_size=1, rank_in_group=0),
    )

    tables = MultiGroupBlockTable(
        max_num_reqs=2,
        max_num_batched_tokens=16,
        pin_memory=False,
        device=torch.device("cpu"),
        block_sizes=[16, 4],
        kernel_block_sizes=[16, 4],
        max_num_blocks=[8, 8],
        cp_kv_cache_interleave_size=1,
        dcp_exempt=[False, True],
    )

    sharded, replicated = tables.block_tables
    assert (sharded.dcp_world_size, sharded.dcp_rank) == (4, 2)
    assert (replicated.dcp_world_size, replicated.dcp_rank) == (1, 0)


def test_multigroup_block_table_validates_exemption_cardinality() -> None:
    try:
        MultiGroupBlockTable(
            max_num_reqs=2,
            max_num_batched_tokens=16,
            pin_memory=False,
            device=torch.device("cpu"),
            block_sizes=[16, 4],
            kernel_block_sizes=[16, 4],
            max_num_blocks=[8, 8],
            dcp_exempt=[True],
        )
    except ValueError as error:
        assert "dcp_exempt length" in str(error)
    else:
        raise AssertionError("mismatched dcp_exempt length was accepted")
