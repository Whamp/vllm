# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 decode-context-parallel sparse-attention merge operations."""

from typing import TYPE_CHECKING

import torch

from vllm.v1.attention.ops.common import cp_lse_ag_out_rs
from vllm.v1.attention.ops.dcp_alltoall import dcp_a2a_lse_reduce

if TYPE_CHECKING:
    from vllm.distributed.parallel_state import GroupCoordinator

# Empty shards must remain finite. Using -inf makes an all-empty merge compute
# -inf - -inf and poison the softmax with NaN.
DCP_LSE_SENTINEL = -1.0e30


def softmax_stats_to_lse(
    rowmax: torch.Tensor,
    sumexp: torch.Tensor,
) -> torch.Tensor:
    """Convert pre-sink softmax statistics to natural-log LSE values."""
    rowmax = rowmax.to(torch.float32)
    sumexp = sumexp.to(torch.float32)
    lse = rowmax + torch.log(torch.clamp(sumexp, min=torch.finfo(torch.float32).tiny))
    return torch.where(
        sumexp > 0,
        lse,
        torch.full_like(lse, DCP_LSE_SENTINEL),
    )


def apply_attn_sink(
    out: torch.Tensor,
    lse: torch.Tensor,
    attn_sink: torch.Tensor,
) -> torch.Tensor:
    """Apply the DeepSeek attention sink once after global shard merging."""
    sink = attn_sink[: out.shape[1]].to(dtype=lse.dtype)
    output_lse = torch.logaddexp(lse, sink.unsqueeze(0))
    scale = torch.exp(lse - output_lse).to(dtype=out.dtype)
    return out * scale.unsqueeze(-1)


def dcp_merge_flashmla_output(
    local_out: torch.Tensor,
    local_lse: torch.Tensor,
    attn_sink: torch.Tensor,
    output: torch.Tensor,
    group: "GroupCoordinator",
    use_a2a: bool = True,
) -> None:
    """Merge pre-sink rank-local FlashMLA partials into local-head output."""
    if use_a2a:
        out, lse = dcp_a2a_lse_reduce(
            local_out,
            local_lse,
            group,
            return_lse=True,
        )
    else:
        out, lse = cp_lse_ag_out_rs(
            local_out,
            local_lse,
            group,
            return_lse=True,
        )
    output[:, : out.shape[1], :].copy_(apply_attn_sink(out, lse, attn_sink))


def dcp_softmax_reduce(
    local_max: torch.Tensor,
    local_sum: torch.Tensor,
    local_weighted_value: torch.Tensor,
    group: "GroupCoordinator",
) -> torch.Tensor:
    """Merge fp32 stable-softmax statistics across DCP ranks."""
    valid = local_sum > 0
    local_max = torch.where(
        valid,
        local_max,
        torch.full_like(local_max, -float("inf")),
    )
    gathered_max = group.all_gather(local_max, dim=0).reshape(
        (group.world_size,) + local_max.shape
    )
    global_max = gathered_max.max(dim=0).values

    scale = torch.exp(local_max - global_max)
    scale = torch.where(valid, scale, torch.zeros_like(scale))
    reduce_payload = torch.stack(
        (
            torch.where(valid, local_sum * scale, torch.zeros_like(local_sum)),
            torch.where(
                valid,
                local_weighted_value * scale,
                torch.zeros_like(local_weighted_value),
            ),
        )
    )
    global_sum, global_weighted_value = group.all_reduce(reduce_payload).unbind(0)
    return torch.where(
        global_sum > 0,
        global_weighted_value / global_sum,
        torch.zeros_like(global_weighted_value),
    )
