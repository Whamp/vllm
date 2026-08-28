# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

import vllm.model_executor.layers.sparse_attn_indexer as indexer_module
from vllm.model_executor.layers.sparse_attn_indexer import _sm86_dcp_global_topk


class _FakeDCPGroup:
    def __init__(self, values: torch.Tensor, indices: torch.Tensor) -> None:
        self.values = values
        self.indices = indices

    def all_gather(self, tensor: torch.Tensor, dim: int) -> torch.Tensor:
        assert dim == 1
        return self.values if tensor.dtype.is_floating_point else self.indices


def test_sm86_dcp_global_topk_is_score_ordered_with_index_ties(monkeypatch) -> None:
    gathered_values = torch.tensor(
        [[0.8, 0.7, -float("inf"), 0.8, 0.6, 0.55]], dtype=torch.float32
    )
    gathered_indices = torch.tensor([[5, 1, -1, 2, 6, 3]], dtype=torch.int32)
    monkeypatch.setattr(
        indexer_module,
        "get_dcp_group",
        lambda: _FakeDCPGroup(gathered_values, gathered_indices),
    )

    local_values = gathered_values[:, :3]
    local_indices = gathered_indices[:, :3]
    values, indices = _sm86_dcp_global_topk(local_values, local_indices, 3)

    assert torch.equal(indices, torch.tensor([[2, 5, 1]], dtype=torch.int32))
    assert torch.allclose(values, torch.tensor([[0.8, 0.8, 0.7]]))


def test_sm86_dcp_global_topk_keeps_invalid_padding_out(monkeypatch) -> None:
    gathered_values = torch.full((1, 4), -float("inf"), dtype=torch.float32)
    gathered_indices = torch.full((1, 4), -1, dtype=torch.int32)
    monkeypatch.setattr(
        indexer_module,
        "get_dcp_group",
        lambda: _FakeDCPGroup(gathered_values, gathered_indices),
    )

    values, indices = _sm86_dcp_global_topk(
        gathered_values[:, :2], gathered_indices[:, :2], 2
    )

    assert torch.equal(indices, torch.full((1, 2), -1, dtype=torch.int32))
    assert torch.isneginf(values).all()
