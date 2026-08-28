# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 SM86 decode-context-parallel compressed-entry layout.

Every DCP consumer must use these helpers after converting token positions to
compressed-entry coordinates. For interleave ``I`` and DCP world size ``W``:

    owner(e)       = (e // I) % W
    local_entry(e) = (e // (I * W)) * I + e % I
    global(r, j)   = (j // I) * (I * W) + r * I + j % I

The invalid entry ``-1`` passes through every mapping unchanged.
"""

import torch


def sm86_dcp_local_count(
    num_entries: int,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> int:
    """Count entries owned by one DCP rank in global range ``[0, n)``."""
    full_cycles = num_entries // (cp_interleave * dcp_world_size)
    base = full_cycles * cp_interleave
    remainder = num_entries - full_cycles * cp_interleave * dcp_world_size
    extra = min(max(remainder - dcp_rank * cp_interleave, 0), cp_interleave)
    return base + extra


def sm86_dcp_local_to_global(
    local_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Map rank-local compressed-entry indices to global entry indices."""
    safe = torch.clamp(local_indices, min=0)
    global_indices = (
        (safe // cp_interleave) * (cp_interleave * dcp_world_size)
        + dcp_rank * cp_interleave
        + safe % cp_interleave
    )
    return torch.where(
        local_indices >= 0,
        global_indices,
        torch.full_like(global_indices, -1),
    )


def sm86_dcp_owner(
    global_indices: torch.Tensor,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Return the owning DCP rank for each global compressed-entry index."""
    safe = torch.clamp(global_indices, min=0)
    owners = (safe // cp_interleave) % dcp_world_size
    return torch.where(
        global_indices >= 0,
        owners,
        torch.full_like(owners, -1),
    )


def sm86_dcp_owns(
    global_indices: torch.Tensor,
    dcp_rank: int,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Return whether one DCP rank owns each valid global entry index."""
    return (global_indices >= 0) & (
        sm86_dcp_owner(global_indices, dcp_world_size, cp_interleave) == dcp_rank
    )


def sm86_dcp_global_to_local(
    global_indices: torch.Tensor,
    dcp_rank: int | torch.Tensor,
    dcp_world_size: int,
    cp_interleave: int,
) -> torch.Tensor:
    """Map global compressed entries to rank-local prefix coordinates.

    For entries owned by ``dcp_rank`` this is the exact inverse of
    :func:`sm86_dcp_local_to_global`. For other entries it returns the local
    prefix count. Callers that require ownership must mask with
    :func:`sm86_dcp_owns` first.
    """
    safe = torch.clamp(global_indices, min=0)
    rank_stride = dcp_world_size * cp_interleave
    full_cycles = safe // rank_stride
    base = full_cycles * cp_interleave
    remainder = safe - full_cycles * rank_stride
    extra = torch.clamp(
        remainder - dcp_rank * cp_interleave,
        min=0,
        max=cp_interleave,
    )
    local_indices = base + extra
    return torch.where(
        global_indices >= 0,
        local_indices,
        torch.full_like(local_indices, -1),
    )
