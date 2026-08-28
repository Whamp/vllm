# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.common.ops.cache_utils import (
    _sm86_dcp_virtual_block_table,
)
from vllm.models.deepseek_v4.sparse_mla import (
    build_sm86_dcp_c128_decode_entries,
)


def _owned_count(num_entries: int, rank: int, world: int) -> int:
    return sum(entry % world == rank for entry in range(num_entries))


def test_c128_decode_metadata_uses_contiguous_rank_local_entries() -> None:
    positions = torch.tensor([127, 255, 383, 511, 639], dtype=torch.int64)
    width = 4
    indices, lengths = build_sm86_dcp_c128_decode_entries(
        positions,
        compress_ratio=128,
        decode_width=width,
        dcp_world_size=4,
        dcp_rank=0,
        cp_interleave=1,
        max_global_entries=8,
    )

    expected_lengths = torch.tensor(
        [_owned_count((int(pos) + 1) // 128, 0, 4) for pos in positions],
        dtype=torch.int32,
    )
    assert torch.equal(lengths, expected_lengths)
    for row, length in zip(indices, expected_lengths, strict=True):
        count = int(length)
        assert row[:count].tolist() == list(range(count))
        assert row[count:].tolist() == [-1] * (width - count)


def test_dcp_virtual_block_table_addresses_rank_major_staging() -> None:
    table = _sm86_dcp_virtual_block_table(
        max_entries=4,
        max_local_entries=3,
        num_reqs=2,
        dcp_world_size=2,
        cp_interleave=1,
        device=torch.device("cpu"),
    )
    assert table.tolist() == [[0, 6, 1, 7], [3, 9, 4, 10]]
