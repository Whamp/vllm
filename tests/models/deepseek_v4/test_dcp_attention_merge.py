# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.models.deepseek_v4.common.ops.dcp import (
    DCP_LSE_SENTINEL,
    apply_attn_sink,
    softmax_stats_to_lse,
)


def test_softmax_stats_to_lse_uses_finite_empty_shard_sentinel() -> None:
    rowmax = torch.tensor([[2.0, -3.0]], dtype=torch.float32)
    sumexp = torch.tensor([[4.0, 0.0]], dtype=torch.float32)
    actual = softmax_stats_to_lse(rowmax, sumexp)
    assert torch.equal(actual[:, 1], torch.tensor([DCP_LSE_SENTINEL]))
    assert torch.isfinite(actual).all()
    expected_lse = torch.tensor([2.0 + torch.log(torch.tensor(4.0))])
    assert torch.allclose(actual[:, 0], expected_lse)


def test_apply_attn_sink_matches_direct_global_softmax() -> None:
    # Two rank-local attention shards plus one learned attention sink.
    scores = torch.tensor([1.0, -2.0, 0.5, 3.0], dtype=torch.float64)
    values = torch.tensor(
        [[1.0, 2.0], [3.0, -1.0], [-2.0, 0.5], [4.0, 3.0]],
        dtype=torch.float64,
    )
    sink = torch.tensor([0.75], dtype=torch.float64)

    weights_without_sink = torch.softmax(scores, dim=0)
    pre_sink_out = (weights_without_sink[:, None] * values).sum(dim=0)
    lse_without_sink = torch.logsumexp(scores, dim=0)

    actual = apply_attn_sink(
        pre_sink_out.reshape(1, 1, 2),
        lse_without_sink.reshape(1, 1),
        sink,
    ).reshape(2)

    sink_weight = torch.exp(sink[0])
    expected = (torch.exp(scores)[:, None] * values).sum(dim=0) / (
        torch.exp(scores).sum() + sink_weight
    )
    assert torch.allclose(actual, expected, rtol=1e-12, atol=1e-12)


def test_empty_attention_shard_contributes_zero_before_sink() -> None:
    empty_out = torch.ones((1, 1, 2), dtype=torch.float32)
    empty_lse = torch.full((1, 1), DCP_LSE_SENTINEL, dtype=torch.float32)
    sink = torch.tensor([0.0], dtype=torch.float32)
    actual = apply_attn_sink(empty_out, empty_lse, sink)
    assert torch.equal(actual, torch.zeros_like(actual))
