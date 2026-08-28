# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch
from torch import nn

from vllm.models.deepseek_v4.attention import DeepseekV4Attention


class _ProfiledCompressorLinear(nn.Module):
    """Mimic the return_bias=False compressor after GGUF raw-weight loading."""

    def __init__(self, expected: torch.Tensor) -> None:
        super().__init__()
        self.expected = expected

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        assert hidden_states.shape == self.expected.shape
        return self.expected


def test_profiled_compressor_projection_returns_tensor_without_unpacking() -> None:
    expected = torch.randn(3, 8)
    compressor = nn.Module()
    compressor.fused_wkv_wgate = _ProfiledCompressorLinear(expected)

    output = DeepseekV4Attention._project_compressor_kv_score(
        compressor, torch.zeros_like(expected)
    )

    assert output is expected
