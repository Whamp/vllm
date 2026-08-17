# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Prepare native GGUF Q8_0 weights for symmetric INT8 Marlin execution."""

from dataclasses import dataclass
from importlib import import_module
from types import ModuleType

import torch

GGUF_Q8_0_BLOCK_ELEMENTS = 32
GGUF_Q8_0_BLOCK_BYTES = 34


def _load_gguf_q8_marlin_utils() -> ModuleType:
    """Load the linear-kernel package before its Marlin utility dependency."""
    import_module("vllm.model_executor.kernels.linear.mixed_precision.marlin")
    return import_module("vllm.model_executor.layers.quantization.utils.marlin_utils")


def unpack_gguf_q8_0_to_gptq(
    raw_weights: torch.Tensor,
    *,
    input_columns: int,
    scale_dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert row-major Q8_0 blocks to GPTQ INT8 words and group-32 scales.

    The returned packed weights have ``[K / 4, N]`` orientation expected by
    ``gptq_marlin_repack``. Q8_0 signed codes are offset by 128 for Marlin's
    symmetric ``uint8b128`` type. The returned scales have ``[K / 32, N]``
    orientation and the caller-selected runtime dtype.
    """
    if raw_weights.dtype is not torch.uint8 or raw_weights.ndim != 2:
        raise ValueError("GGUF Q8_0 weights must be a rank-2 uint8 tensor")
    if not raw_weights.is_contiguous():
        raise ValueError("GGUF Q8_0 weights must be contiguous")
    if input_columns % GGUF_Q8_0_BLOCK_ELEMENTS != 0:
        raise ValueError("GGUF Q8_0 input columns must be a multiple of 32")
    block_count = input_columns // GGUF_Q8_0_BLOCK_ELEMENTS
    expected_row_bytes = block_count * GGUF_Q8_0_BLOCK_BYTES
    if raw_weights.shape[1] != expected_row_bytes:
        raise ValueError(
            "GGUF Q8_0 row byte count mismatch: "
            f"expected {expected_row_bytes}, got {raw_weights.shape[1]}"
        )

    output_rows = raw_weights.shape[0]
    blocks = raw_weights.reshape(output_rows, block_count, GGUF_Q8_0_BLOCK_BYTES)
    q8_scales = (
        blocks[:, :, :2]
        .contiguous()
        .view(torch.float16)
        .reshape(output_rows, block_count)
    )
    signed_codes = (
        blocks[:, :, 2:]
        .contiguous()
        .view(torch.int8)
        .reshape(output_rows, input_columns)
    )
    unsigned_codes_kn = (signed_codes.to(torch.int16) + 128).to(torch.int64).T
    code_groups = unsigned_codes_kn.reshape(input_columns // 4, 4, output_rows)
    packed_weights = (
        code_groups[:, 0]
        | (code_groups[:, 1] << 8)
        | (code_groups[:, 2] << 16)
        | (code_groups[:, 3] << 24)
    ).to(torch.int32)
    return packed_weights.contiguous(), q8_scales.T.to(scale_dtype).contiguous()


@dataclass(frozen=True)
class GGUFQ8MarlinWeights:
    """Prepared Q8_0 Marlin storage and its fixed rank-local shape contract."""

    weight: torch.Tensor
    scales: torch.Tensor
    workspace: torch.Tensor
    empty_indices: torch.Tensor
    input_columns: int
    output_rows: int


def prepare_gguf_q8_0_marlin(
    raw_weights: torch.Tensor,
    *,
    input_columns: int,
    scale_dtype: torch.dtype,
) -> GGUFQ8MarlinWeights:
    """Repack immutable Q8_0 rows once for group-32 INT8 Marlin execution."""
    from vllm import _custom_ops as ops

    marlin_utils = _load_gguf_q8_marlin_utils()
    gptq_weights, scales = unpack_gguf_q8_0_to_gptq(
        raw_weights, input_columns=input_columns, scale_dtype=scale_dtype
    )
    output_rows = raw_weights.shape[0]
    padded_n, padded_k = marlin_utils.marlin_padded_nk(
        output_rows, input_columns, GGUF_Q8_0_BLOCK_ELEMENTS
    )
    padded_weights = marlin_utils.marlin_pad_qweight(
        gptq_weights,
        output_rows,
        input_columns,
        padded_n,
        padded_k,
    )
    marlin_weights = ops.gptq_marlin_repack(
        b_q_weight=padded_weights,
        perm=torch.empty(0, dtype=torch.int32, device=raw_weights.device),
        size_k=padded_k,
        size_n=padded_n,
        num_bits=8,
    )
    padded_scales = marlin_utils.marlin_pad_scales(
        scales,
        output_rows,
        input_columns,
        padded_n,
        padded_k,
        GGUF_Q8_0_BLOCK_ELEMENTS,
    )
    marlin_scales = marlin_utils.marlin_permute_scales(
        padded_scales,
        size_k=padded_k,
        size_n=padded_n,
        group_size=GGUF_Q8_0_BLOCK_ELEMENTS,
    )
    empty_indices = marlin_utils.marlin_make_empty_g_idx(raw_weights.device)
    return GGUFQ8MarlinWeights(
        weight=marlin_weights,
        scales=marlin_scales,
        workspace=marlin_utils.marlin_make_workspace_new(raw_weights.device),
        empty_indices=empty_indices,
        input_columns=input_columns,
        output_rows=output_rows,
    )


def apply_gguf_q8_0_marlin(
    inputs: torch.Tensor, prepared: GGUFQ8MarlinWeights
) -> torch.Tensor:
    """Apply prepared Q8_0 weights without steady-state dequantization."""
    from vllm.scalar_type import scalar_types

    marlin_utils = _load_gguf_q8_marlin_utils()
    if inputs.shape[-1] != prepared.input_columns:
        raise ValueError(
            "GGUF Q8_0 Marlin input width mismatch: "
            f"expected {prepared.input_columns}, got {inputs.shape[-1]}"
        )
    return marlin_utils.apply_gptq_marlin_linear(
        input=inputs,
        weight=prepared.weight,
        weight_scale=prepared.scales,
        weight_zp=prepared.empty_indices,
        g_idx=prepared.empty_indices,
        g_idx_sort_indices=prepared.empty_indices,
        workspace=prepared.workspace,
        wtype=scalar_types.uint8b128,
        output_size_per_partition=prepared.output_rows,
        input_size_per_partition=prepared.input_columns,
        is_k_full=True,
    )
