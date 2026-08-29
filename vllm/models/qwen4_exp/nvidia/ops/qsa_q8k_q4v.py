# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen QSA writer and reader for symmetric Q8 K plus asymmetric Q4 V."""

import torch

from vllm.model_executor.layers.quantization.utils.int8_utils import (
    per_token_quant_int8,
    round_int8,
)
from vllm.triton_utils import tl, triton
from vllm.v1.attention.ops.int4_per_token_head import (
    pack_int4_nibbles,
    single_rht,
)


@triton.jit
def _reshape_cache_q8k_q4v_kernel(
    key_ptr,
    value_ptr,
    key_cache_ptr,
    key_scale_cache_ptr,
    value_cache_ptr,
    value_scale_cache_ptr,
    slot_mapping_ptr,
    stride_key_token: tl.int64,
    stride_key_head: tl.int64,
    stride_value_token: tl.int64,
    stride_value_head: tl.int64,
    stride_key_cache_block: tl.int64,
    stride_key_cache_slot: tl.int64,
    stride_key_cache_head: tl.int64,
    stride_key_scale_block: tl.int64,
    stride_key_scale_slot: tl.int64,
    stride_key_scale_head: tl.int64,
    stride_value_cache_block: tl.int64,
    stride_value_cache_slot: tl.int64,
    stride_value_cache_head: tl.int64,
    stride_value_scale_block: tl.int64,
    stride_value_scale_slot: tl.int64,
    stride_value_scale_head: tl.int64,
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    head_size_padded: tl.constexpr,
    packed_head_padded: tl.constexpr,
) -> None:
    token_index = tl.program_id(0)
    head_index = tl.program_id(1)
    slot = tl.load(slot_mapping_ptr + token_index).to(tl.int64)
    if slot < 0:
        return
    block_index = slot // block_size
    slot_index = slot % block_size

    key_offsets = tl.arange(0, head_size_padded)
    key_mask = key_offsets < head_size
    key = tl.load(
        key_ptr
        + token_index * stride_key_token
        + head_index * stride_key_head
        + key_offsets,
        mask=key_mask,
        other=0.0,
    ).to(tl.float32)
    key_absolute_maximum = tl.max(tl.abs(key))
    key_scale = tl.where(
        key_absolute_maximum == 0,
        1.0,
        key_absolute_maximum / 127.0,
    )
    key_codes = round_int8(tl.clamp(key / key_scale, -127.0, 127.0))
    tl.store(
        key_cache_ptr
        + block_index * stride_key_cache_block
        + slot_index * stride_key_cache_slot
        + head_index * stride_key_cache_head
        + key_offsets,
        key_codes,
        mask=key_mask,
    )
    tl.store(
        key_scale_cache_ptr
        + block_index * stride_key_scale_block
        + slot_index * stride_key_scale_slot
        + head_index * stride_key_scale_head,
        key_scale,
    )

    packed_offsets = tl.arange(0, packed_head_padded)
    even_offsets = packed_offsets * 2
    odd_offsets = even_offsets + 1
    even_mask = even_offsets < head_size
    odd_mask = odd_offsets < head_size
    value_base = (
        value_ptr + token_index * stride_value_token + head_index * stride_value_head
    )
    value_even = tl.load(
        value_base + even_offsets,
        mask=even_mask,
        other=0.0,
    ).to(tl.float32)
    value_odd = tl.load(
        value_base + odd_offsets,
        mask=odd_mask,
        other=0.0,
    ).to(tl.float32)
    value_minimum = tl.minimum(
        tl.min(tl.where(even_mask, value_even, float("inf"))),
        tl.min(tl.where(odd_mask, value_odd, float("inf"))),
    )
    value_maximum = tl.maximum(
        tl.max(tl.where(even_mask, value_even, float("-inf"))),
        tl.max(tl.where(odd_mask, value_odd, float("-inf"))),
    )
    value_scale = tl.maximum((value_maximum - value_minimum) / 15.0, 1e-6)
    value_zero_point = tl.clamp(
        round_int8(-value_minimum / value_scale).to(tl.float32),
        0.0,
        15.0,
    )
    inverse_value_scale = 1.0 / value_scale
    value_even_codes = tl.clamp(
        round_int8(value_even * inverse_value_scale + value_zero_point).to(tl.float32),
        0.0,
        15.0,
    )
    value_odd_codes = tl.clamp(
        round_int8(value_odd * inverse_value_scale + value_zero_point).to(tl.float32),
        0.0,
        15.0,
    )
    value_scale_bits = value_scale.to(tl.int32, bitcast=True)
    value_scale_and_zero_point = (
        (value_scale_bits & -16) | (value_zero_point.to(tl.int32) & 0xF)
    ).to(tl.float32, bitcast=True)
    value_packed = pack_int4_nibbles(
        value_even_codes.to(tl.uint8),
        value_odd_codes.to(tl.uint8),
    )
    tl.store(
        value_cache_ptr
        + block_index * stride_value_cache_block
        + slot_index * stride_value_cache_slot
        + head_index * stride_value_cache_head
        + packed_offsets,
        value_packed,
        mask=packed_offsets < head_size // 2,
    )
    tl.store(
        value_scale_cache_ptr
        + block_index * stride_value_scale_block
        + slot_index * stride_value_scale_slot
        + head_index * stride_value_scale_head,
        value_scale_and_zero_point,
    )


def reshape_and_cache_q8k_q4v(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_cache: torch.Tensor,
    value_scale_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
) -> None:
    """Transform and store Q8 keys plus packed asymmetric Q4 values."""
    if key.shape != value.shape or key.ndim != 3:
        raise ValueError("QSA Q8-K/Q4-V writer requires matching 3D K/V")
    if key.dtype != torch.bfloat16 or value.dtype != torch.bfloat16:
        raise TypeError("QSA Q8-K/Q4-V writer requires BF16 K/V")
    if key.shape[2] % 2:
        raise ValueError("QSA Q8-K/Q4-V writer requires an even head size")
    expected_cache_prefix = (key_cache.shape[0], key_cache.shape[1], key.shape[1])
    if key_cache.shape != (*expected_cache_prefix, key.shape[2]):
        raise ValueError("QSA Q8-K/Q4-V key cache shape mismatch")
    if value_cache.shape != (*expected_cache_prefix, key.shape[2] // 2):
        raise ValueError("QSA Q8-K/Q4-V value cache shape mismatch")
    if key_scale_cache.shape != expected_cache_prefix:
        raise ValueError("QSA Q8-K/Q4-V key scale shape mismatch")
    if value_scale_cache.shape != expected_cache_prefix:
        raise ValueError("QSA Q8-K/Q4-V value scale shape mismatch")
    if key_cache.dtype != torch.int8 or value_cache.dtype != torch.uint8:
        raise TypeError("QSA Q8-K/Q4-V cache storage dtype mismatch")
    if (
        key_scale_cache.dtype != torch.float32
        or value_scale_cache.dtype != torch.float32
    ):
        raise TypeError("QSA Q8-K/Q4-V scale storage must be FP32")
    if slot_mapping.shape != (key.shape[0],) or slot_mapping.dtype != torch.int64:
        raise ValueError("QSA Q8-K/Q4-V slot mapping is invalid")

    transformed_key = single_rht(key.float()).to(key.dtype)
    transformed_value = single_rht(value.float()).to(value.dtype)
    head_size = key.shape[2]
    head_size_padded = triton.next_power_of_2(head_size)
    packed_head_padded = triton.next_power_of_2(head_size // 2)
    num_warps = min(8, max(1, head_size_padded // 32))
    _reshape_cache_q8k_q4v_kernel[(key.shape[0], key.shape[1])](
        transformed_key,
        transformed_value,
        key_cache,
        key_scale_cache,
        value_cache,
        value_scale_cache,
        slot_mapping,
        transformed_key.stride(0),
        transformed_key.stride(1),
        transformed_value.stride(0),
        transformed_value.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_scale_cache.stride(0),
        key_scale_cache.stride(1),
        key_scale_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_scale_cache.stride(0),
        value_scale_cache.stride(1),
        value_scale_cache.stride(2),
        block_size=key_cache.shape[1],
        head_size=head_size,
        head_size_padded=head_size_padded,
        packed_head_padded=packed_head_padded,
        num_warps=num_warps,
    )


@triton.jit
def _qsa_q8k_q4v_split_kernel(
    query_ptr,
    query_scale_ptr,
    key_cache_ptr,
    key_scale_ptr,
    value_cache_ptr,
    value_scale_ptr,
    indices_ptr,
    block_table_ptr,
    token_to_request_ptr,
    partial_output_ptr,
    partial_lse_ptr,
    output_ptr,
    stride_query_row,
    stride_query_head,
    stride_query_scale_row,
    stride_query_scale_head,
    stride_key_block,
    stride_key_token,
    stride_key_head,
    stride_key_scale_block,
    stride_key_scale_token,
    stride_key_scale_head,
    stride_value_block,
    stride_value_token,
    stride_value_head,
    stride_value_scale_block,
    stride_value_scale_token,
    stride_value_scale_head,
    stride_indices_row,
    stride_table_request,
    stride_output_row,
    stride_output_head,
    num_rows,
    num_cache_blocks,
    num_requests,
    topk: tl.constexpr,
    page_size: tl.constexpr,
    page_table_width: tl.constexpr,
    group_size: tl.constexpr,
    head_dim: tl.constexpr,
    num_query_heads: tl.constexpr,
    num_splits: tl.constexpr,
    num_tiles: tl.constexpr,
    block_m: tl.constexpr,
    block_n: tl.constexpr,
) -> None:
    row = tl.program_id(0)
    kv_head = tl.program_id(1)
    split_index = tl.program_id(2)
    request = tl.load(token_to_request_ptr + row)
    safe_request = tl.minimum(tl.maximum(request, 0), num_requests - 1)

    head_offsets = tl.arange(0, block_m)
    dimension_offsets = tl.arange(0, head_dim)
    token_offsets = tl.arange(0, block_n)
    first_head = kv_head * group_size
    query_mask = head_offsets[:, None] < group_size
    query = tl.load(
        query_ptr
        + row * stride_query_row
        + (first_head + head_offsets[:, None]) * stride_query_head
        + dimension_offsets[None, :],
        mask=query_mask,
        other=0,
    )
    query_scale = tl.load(
        query_scale_ptr
        + row * stride_query_scale_row
        + (first_head + head_offsets) * stride_query_scale_head,
        mask=head_offsets < group_size,
        other=0.0,
    )

    maximum = tl.full((block_m,), -1.0e20, dtype=tl.float32)
    normalizer = tl.zeros((block_m,), dtype=tl.float32)
    accumulator = tl.zeros((block_m, head_dim), dtype=tl.float32)
    score_scale_log2: tl.constexpr = ((head_dim**-0.5) / head_dim) * 1.4426950408889634

    split_tile_start = split_index * num_tiles // num_splits
    split_tile_end = (split_index + 1) * num_tiles // num_splits
    for tile_index in range(split_tile_start, split_tile_end):
        selected_offsets = tile_index * block_n + token_offsets
        logical_token = tl.load(
            indices_ptr + row * stride_indices_row + selected_offsets,
            mask=selected_offsets < topk,
            other=-1,
        )
        safe_token = tl.maximum(logical_token, 0)
        logical_page = safe_token // page_size
        page_offset = safe_token % page_size
        valid = (
            (request >= 0)
            & (request < num_requests)
            & (logical_token >= 0)
            & (logical_page < page_table_width)
        )
        physical_page = tl.load(
            block_table_ptr
            + safe_request * stride_table_request
            + tl.minimum(logical_page, page_table_width - 1),
            mask=valid,
            other=-1,
        )
        valid &= (physical_page >= 0) & (physical_page < num_cache_blocks)
        safe_page = tl.maximum(physical_page, 0).to(tl.int64)

        key = tl.load(
            key_cache_ptr
            + safe_page[None, :] * stride_key_block
            + page_offset[None, :] * stride_key_token
            + kv_head * stride_key_head
            + dimension_offsets[:, None],
            mask=valid[None, :],
            other=0,
        )
        key_scale = tl.load(
            key_scale_ptr
            + safe_page * stride_key_scale_block
            + page_offset * stride_key_scale_token
            + kv_head * stride_key_scale_head,
            mask=valid,
            other=0.0,
        )
        integer_score = tl.dot(query, key, out_dtype=tl.int32)
        scores = (
            integer_score.to(tl.float32)
            * query_scale[:, None]
            * key_scale[None, :]
            * score_scale_log2
        )
        scores = tl.where(valid[None, :], scores, -1.0e20)
        next_maximum = tl.maximum(maximum, tl.max(scores, axis=1))
        alpha = tl.math.exp2(maximum - next_maximum)
        probabilities = tl.where(
            valid[None, :],
            tl.math.exp2(scores - next_maximum[:, None]),
            0.0,
        )

        value_scale_raw = tl.load(
            value_scale_ptr
            + safe_page * stride_value_scale_block
            + page_offset * stride_value_scale_token
            + kv_head * stride_value_scale_head,
            mask=valid,
            other=0.0,
        )
        value_scale_bits = value_scale_raw.to(tl.int32, bitcast=True)
        value_zero_point = value_scale_bits & 0xF
        value_scale = (value_scale_bits & -16).to(tl.float32, bitcast=True)
        weighted_probability = probabilities * value_scale[None, :]
        probability_absolute_maximum = tl.max(weighted_probability, axis=1)
        probability_scale = tl.where(
            probability_absolute_maximum == 0,
            1.0,
            probability_absolute_maximum / 127.0,
        )
        probability_codes = round_int8(
            tl.clamp(weighted_probability / probability_scale[:, None], 0.0, 127.0)
        )

        packed_dimensions = dimension_offsets // 2
        value_packed = tl.load(
            value_cache_ptr
            + safe_page[:, None] * stride_value_block
            + page_offset[:, None] * stride_value_token
            + kv_head * stride_value_head
            + packed_dimensions[None, :],
            mask=valid[:, None],
            other=0,
        )
        value_unsigned = tl.where(
            dimension_offsets[None, :] % 2 == 0,
            value_packed & 0xF,
            (value_packed >> 4) & 0xF,
        ).to(tl.int32)
        value_codes = (value_unsigned - value_zero_point[:, None]).to(tl.int8)
        integer_output = tl.dot(
            probability_codes,
            value_codes,
            out_dtype=tl.int32,
        )
        accumulator = (
            accumulator * alpha[:, None]
            + integer_output.to(tl.float32) * probability_scale[:, None]
        )
        normalizer = normalizer * alpha + tl.sum(probabilities, axis=1)
        maximum = next_maximum

    has_values = normalizer > 0
    normalized_output = tl.where(
        has_values[:, None],
        accumulator / tl.maximum(normalizer[:, None], 1.0e-20),
        0.0,
    )
    output_mask = head_offsets[:, None] < group_size
    if num_splits == 1:
        tl.store(
            output_ptr
            + row * stride_output_row
            + (first_head + head_offsets[:, None]) * stride_output_head
            + dimension_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
    else:
        partial_lse = tl.where(
            has_values,
            maximum + tl.math.log2(tl.maximum(normalizer, 1.0e-20)),
            -float("inf"),
        )
        tl.store(
            partial_output_ptr
            + (
                (split_index * num_rows + row) * num_query_heads
                + first_head
                + head_offsets[:, None]
            )
            * head_dim
            + dimension_offsets[None, :],
            normalized_output,
            mask=output_mask,
        )
        tl.store(
            partial_lse_ptr
            + (split_index * num_rows + row) * num_query_heads
            + first_head
            + head_offsets,
            partial_lse,
            mask=head_offsets < group_size,
        )


def _qsa_q8k_q4v_launch_config(
    num_rows: int,
    num_kv_heads: int,
    group_size: int,
    selection_width: int,
) -> tuple[int, int, int, int, int]:
    block_m = max(16, triton.next_power_of_2(group_size))
    base_programs = num_rows * num_kv_heads
    if base_programs <= 8:
        block_n, target_splits, num_warps = 32, 32, 4
    elif base_programs <= 256:
        block_n, target_splits, num_warps = 64, 8, 4
    else:
        block_n, target_splits, num_warps = 64, 1, 4
    num_tiles = triton.cdiv(selection_width, block_n)
    maximum_useful_splits = 1 << (num_tiles.bit_length() - 1)
    num_splits = min(maximum_useful_splits, target_splits)
    return block_m, block_n, num_splits, num_tiles, num_warps


def qsa_sparse_paged_attention_q8k_q4v(
    query: torch.Tensor,
    key_cache: torch.Tensor,
    key_scale_cache: torch.Tensor,
    value_cache: torch.Tensor,
    value_scale_cache: torch.Tensor,
    logical_indices: torch.Tensor,
    block_table: torch.Tensor,
    token_to_request: torch.Tensor,
    output: torch.Tensor | None = None,
) -> torch.Tensor:
    """Run direct integer sparse QSA over mixed Q8-K/Q4-V cache rows."""
    if query.ndim != 3 or key_cache.ndim != 4 or value_cache.ndim != 4:
        raise ValueError("QSA Q8-K/Q4-V reader received invalid Q/K/V shapes")
    if query.shape[2] != key_cache.shape[3]:
        raise ValueError("QSA Q8-K/Q4-V query and key widths differ")
    if value_cache.shape[:3] != key_cache.shape[:3]:
        raise ValueError("QSA Q8-K/Q4-V K/V cache prefixes differ")
    if value_cache.shape[3] * 2 != query.shape[2]:
        raise ValueError("QSA Q8-K/Q4-V packed value width is invalid")
    if key_scale_cache.shape != key_cache.shape[:3]:
        raise ValueError("QSA Q8-K/Q4-V key scale shape mismatch")
    if value_scale_cache.shape != key_cache.shape[:3]:
        raise ValueError("QSA Q8-K/Q4-V value scale shape mismatch")
    if query.dtype != torch.bfloat16 or key_cache.dtype != torch.int8:
        raise TypeError("QSA Q8-K/Q4-V query or key dtype is invalid")
    if value_cache.dtype != torch.uint8:
        raise TypeError("QSA Q8-K/Q4-V value cache must be packed uint8")
    if (
        key_scale_cache.dtype != torch.float32
        or value_scale_cache.dtype != torch.float32
    ):
        raise TypeError("QSA Q8-K/Q4-V scales must be FP32")
    if logical_indices.ndim != 2 or logical_indices.shape[0] != query.shape[0]:
        raise ValueError("QSA Q8-K/Q4-V indices must have one row per query")
    if token_to_request.shape != (query.shape[0],) or block_table.ndim != 2:
        raise ValueError("QSA Q8-K/Q4-V metadata shape mismatch")
    if output is None:
        output = torch.empty_like(query)
    if output.shape != query.shape or output.dtype != query.dtype:
        raise ValueError("QSA Q8-K/Q4-V output must match query")
    if query.shape[0] == 0:
        return output

    from vllm.models.qwen4_exp.nvidia.ops.qsa import _qsa_merge_splitk_kernel

    transformed_query = single_rht(query.float()).to(query.dtype)
    quantized_query, query_scale = per_token_quant_int8(transformed_query)
    group_size = query.shape[1] // key_cache.shape[2]
    block_m, block_n, num_splits, num_tiles, num_warps = _qsa_q8k_q4v_launch_config(
        query.shape[0],
        key_cache.shape[2],
        group_size,
        logical_indices.shape[1],
    )
    transformed_output = torch.empty_like(query)
    if num_splits == 1:
        partial_output = transformed_output
        partial_lse = transformed_output
    else:
        partial_output = torch.empty(
            (num_splits, *query.shape),
            dtype=torch.float32,
            device=query.device,
        )
        partial_lse = torch.empty(
            (num_splits, query.shape[0], query.shape[1]),
            dtype=torch.float32,
            device=query.device,
        )

    _qsa_q8k_q4v_split_kernel[
        (
            query.shape[0],
            key_cache.shape[2],
            num_splits,
        )
    ](
        quantized_query,
        query_scale,
        key_cache,
        key_scale_cache,
        value_cache,
        value_scale_cache,
        logical_indices,
        block_table,
        token_to_request,
        partial_output,
        partial_lse,
        transformed_output,
        quantized_query.stride(0),
        quantized_query.stride(1),
        query_scale.stride(0),
        query_scale.stride(1),
        key_cache.stride(0),
        key_cache.stride(1),
        key_cache.stride(2),
        key_scale_cache.stride(0),
        key_scale_cache.stride(1),
        key_scale_cache.stride(2),
        value_cache.stride(0),
        value_cache.stride(1),
        value_cache.stride(2),
        value_scale_cache.stride(0),
        value_scale_cache.stride(1),
        value_scale_cache.stride(2),
        logical_indices.stride(0),
        block_table.stride(0),
        transformed_output.stride(0),
        transformed_output.stride(1),
        query.shape[0],
        key_cache.shape[0],
        block_table.shape[0],
        topk=logical_indices.shape[1],
        page_size=key_cache.shape[1],
        page_table_width=block_table.shape[1],
        group_size=group_size,
        head_dim=query.shape[2],
        num_query_heads=query.shape[1],
        num_splits=num_splits,
        num_tiles=num_tiles,
        block_m=block_m,
        block_n=block_n,
        num_warps=num_warps,
        num_stages=2,
    )
    if num_splits != 1:
        _qsa_merge_splitk_kernel[(query.shape[0], query.shape[1])](
            partial_output,
            partial_lse,
            transformed_output,
            transformed_output.stride(0),
            transformed_output.stride(1),
            query.shape[0],
            HEAD_DIM=query.shape[2],
            NUM_QUERY_HEADS=query.shape[1],
            NUM_SPLITS=num_splits,
            BLOCK_SPLITS=triton.next_power_of_2(num_splits),
            num_warps=2,
            num_stages=1,
        )
    restored = single_rht(transformed_output.float(), inverse=True) / query.shape[2]
    output.copy_(restored.to(query.dtype))
    return output


__all__ = [
    "qsa_sparse_paged_attention_q8k_q4v",
    "reshape_and_cache_q8k_q4v",
]
