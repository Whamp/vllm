# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from vllm.v1.attention.backends.mla.sm86_dcp_layout import (
    sm86_dcp_global_to_local,
    sm86_dcp_local_count,
    sm86_dcp_local_to_global,
    sm86_dcp_owner,
    sm86_dcp_owns,
    sm86_dcp_replicated_swa_owner,
)


def _enumerate_owned_entries(
    num_entries: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> list[int]:
    return [
        entry
        for entry in range(num_entries)
        if (entry // cp_interleave) % dcp_world_size == dcp_rank
    ]


@pytest.mark.parametrize("dcp_world_size", range(1, 9))
@pytest.mark.parametrize("cp_interleave", (1, 2, 4, 8))
def test_sm86_dcp_layout_matches_enumerated_entry_ownership(
    dcp_world_size: int,
    cp_interleave: int,
) -> None:
    entry_counts = (0, 1, cp_interleave - 1, cp_interleave, 31, 32, 255, 256, 257)
    for num_entries in entry_counts:
        global_entries = torch.arange(num_entries, dtype=torch.int64)
        owners = sm86_dcp_owner(global_entries, dcp_world_size, cp_interleave)
        for dcp_rank in range(dcp_world_size):
            expected = _enumerate_owned_entries(
                num_entries, dcp_rank, dcp_world_size, cp_interleave
            )
            assert sm86_dcp_local_count(
                num_entries, dcp_rank, dcp_world_size, cp_interleave
            ) == len(expected)
            assert global_entries[owners == dcp_rank].tolist() == expected
            assert sm86_dcp_owns(
                global_entries,
                dcp_rank,
                dcp_world_size,
                cp_interleave,
            ).tolist() == [entry in expected for entry in range(num_entries)]

            local_entries = torch.arange(len(expected), dtype=torch.int64)
            round_trip_global = sm86_dcp_local_to_global(
                local_entries,
                dcp_rank,
                dcp_world_size,
                cp_interleave,
            )
            assert round_trip_global.tolist() == expected
            assert sm86_dcp_global_to_local(
                round_trip_global,
                dcp_rank,
                dcp_world_size,
                cp_interleave,
            ).tolist() == local_entries.tolist()


def test_sm86_dcp_layout_preserves_invalid_entry_sentinel() -> None:
    invalid = torch.tensor([-1], dtype=torch.int64)
    assert sm86_dcp_owner(invalid, 4, 1).item() == -1
    assert sm86_dcp_local_to_global(invalid, 2, 4, 1).item() == -1
    assert sm86_dcp_global_to_local(invalid, 2, 4, 1).item() == -1
    assert not sm86_dcp_owns(invalid, 2, 4, 1).item()


def test_replicated_swa_owner_assigns_each_query_to_one_rank() -> None:
    positions = torch.tensor([0, 1, 2, 3, 4, 1023, 1024], dtype=torch.int64)
    owners = sm86_dcp_replicated_swa_owner(
        positions,
        compressed_block_size=256,
        dcp_world_size=4,
        cp_interleave=1,
    )
    assert owners.tolist() == [0, 1, 2, 3, 0, 3, 0]


def test_compressed_length_is_localized_after_compression() -> None:
    # Global tokens [0, 12) contain three compression-ratio-4 entries.
    # DCP rank 0 owns global entries 0 and 2. Localizing token count first
    # would produce 1 instead of 2 and silently under-size the cache metadata.
    assert sm86_dcp_local_count(12 // 4, 0, 2, 1) == 2
