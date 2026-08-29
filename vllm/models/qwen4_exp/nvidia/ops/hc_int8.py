# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compressed INT8 projection kernels for Qwen4Exp hyperconnections."""

import torch

from vllm.model_executor.layers.quantization.utils.int8_utils import round_int8
from vllm.triton_utils import tl, triton
from vllm.utils.torch_utils import direct_register_custom_op

_INT8_MAX = 127.0
_DOT_K = 32
_DOT_M = 16
_DOT_N = 16


@triton.jit
def _quantize_hyperconnection_activation_kernel(
    input_ptr,
    quantized_ptr,
    scale_ptr,
    input_columns: tl.constexpr,
    group_size: tl.constexpr,
    group_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    token_index = tl.program_id(0)
    group_index = tl.program_id(1)
    group_offsets = tl.arange(0, block_size)
    column_offsets = group_index * group_size + group_offsets
    mask = group_offsets < group_size
    values = tl.load(
        input_ptr + token_index * input_columns + column_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    absolute_maximum = tl.max(tl.abs(values))
    scale = tl.where(absolute_maximum == 0, 1.0, absolute_maximum / _INT8_MAX)
    codes = round_int8(tl.maximum(-_INT8_MAX, tl.minimum(_INT8_MAX, values / scale)))
    tl.store(
        quantized_ptr + token_index * input_columns + column_offsets,
        codes,
        mask=mask,
    )
    tl.store(scale_ptr + token_index * group_count + group_index, scale)


@triton.jit
def _hyperconnection_int8_decode_kernel(
    activation_ptr,
    activation_scale_ptr,
    weight_ptr,
    weight_scale_ptr,
    output_ptr,
    token_count: tl.constexpr,
    output_rows: tl.constexpr,
    input_columns: tl.constexpr,
    group_size: tl.constexpr,
    group_count: tl.constexpr,
) -> None:
    token_offsets = tl.program_id(0) * _DOT_M + tl.arange(0, _DOT_M)
    output_offsets = tl.program_id(1) * _DOT_N + tl.arange(0, _DOT_N)
    token_mask = token_offsets < token_count
    output_mask = output_offsets < output_rows
    output_accumulator = tl.zeros((_DOT_M, _DOT_N), dtype=tl.float32)

    for group_index in tl.range(0, group_count, loop_unroll_factor=1):
        integer_accumulator = tl.zeros((_DOT_M, _DOT_N), dtype=tl.int32)
        group_start = group_index * group_size
        for tile_start in tl.range(0, group_size, _DOT_K, loop_unroll_factor=1):
            column_offsets = group_start + tile_start + tl.arange(0, _DOT_K)
            activation = tl.load(
                activation_ptr
                + token_offsets[:, None] * input_columns
                + column_offsets[None, :],
                mask=token_mask[:, None],
                other=0,
            )
            weight = tl.load(
                weight_ptr
                + output_offsets[None, :] * input_columns
                + column_offsets[:, None],
                mask=output_mask[None, :],
                other=0,
            )
            integer_accumulator += tl.dot(
                activation,
                weight,
                out_dtype=tl.int32,
            )

        activation_scale = tl.load(
            activation_scale_ptr + token_offsets * group_count + group_index,
            mask=token_mask,
            other=0.0,
        )
        weight_scale = tl.load(
            weight_scale_ptr + output_offsets * group_count + group_index,
            mask=output_mask,
            other=0.0,
        )
        output_accumulator += (
            integer_accumulator.to(tl.float32)
            * activation_scale[:, None]
            * weight_scale[None, :]
        )

    tl.store(
        output_ptr + token_offsets[:, None] * output_rows + output_offsets[None, :],
        output_accumulator,
        mask=token_mask[:, None] & output_mask[None, :],
    )


@triton.jit
def _dequantize_hyperconnection_weight_kernel(
    weight_ptr,
    weight_scale_ptr,
    output_ptr,
    element_count: tl.constexpr,
    input_columns: tl.constexpr,
    group_size: tl.constexpr,
    group_count: tl.constexpr,
    block_size: tl.constexpr,
) -> None:
    offsets = tl.program_id(0) * block_size + tl.arange(0, block_size)
    mask = offsets < element_count
    output_rows = offsets // input_columns
    input_offsets = offsets % input_columns
    group_offsets = input_offsets // group_size
    codes = tl.load(weight_ptr + offsets, mask=mask, other=0).to(tl.float32)
    scales = tl.load(
        weight_scale_ptr + output_rows * group_count + group_offsets,
        mask=mask,
        other=0.0,
    ).to(tl.float32)
    tl.store(output_ptr + offsets, codes * scales, mask=mask)


def _hyperconnection_int8_decode(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    if inputs.ndim != 2 or weight.ndim != 2 or weight_scale.ndim != 2:
        raise ValueError("Qwen hyperconnection INT8 decode requires 2D tensors")
    token_count, input_columns = inputs.shape
    output_rows, weight_columns = weight.shape
    if weight_columns != input_columns:
        raise ValueError("Qwen hyperconnection INT8 input width mismatch")
    if token_count not in (1, 2):
        raise ValueError("Qwen hyperconnection INT8 decode supports one or two tokens")
    if group_size <= 0 or input_columns % group_size:
        raise ValueError("Qwen hyperconnection INT8 group size must divide input width")
    group_count = input_columns // group_size
    if weight_scale.shape != (output_rows, group_count):
        raise ValueError("Qwen hyperconnection INT8 weight scale shape mismatch")
    if weight.dtype is not torch.int8 or weight_scale.dtype is not torch.float16:
        raise TypeError("Qwen hyperconnection INT8 weight storage is invalid")
    if inputs.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Qwen hyperconnection INT8 activation dtype is invalid")

    contiguous_inputs = inputs.contiguous()
    contiguous_weight = weight.contiguous()
    quantized_inputs = torch.empty_like(contiguous_inputs, dtype=torch.int8)
    activation_scale = torch.empty(
        (token_count, group_count),
        dtype=torch.float32,
        device=inputs.device,
    )
    quantization_block = triton.next_power_of_2(group_size)
    quantization_warps = min(max(quantization_block // 256, 1), 8)
    _quantize_hyperconnection_activation_kernel[(token_count, group_count)](
        contiguous_inputs,
        quantized_inputs,
        activation_scale,
        input_columns=input_columns,
        group_size=group_size,
        group_count=group_count,
        block_size=quantization_block,
        num_warps=quantization_warps,
    )

    output = inputs.new_empty((token_count, output_rows))
    _hyperconnection_int8_decode_kernel[
        (triton.cdiv(token_count, _DOT_M), triton.cdiv(output_rows, _DOT_N))
    ](
        quantized_inputs,
        activation_scale,
        contiguous_weight,
        weight_scale,
        output,
        token_count=token_count,
        output_rows=output_rows,
        input_columns=input_columns,
        group_size=group_size,
        group_count=group_count,
        num_warps=4,
    )
    return output


def _dequantize_hyperconnection_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    group_size: int,
) -> None:
    if weight.shape != output.shape or weight.ndim != 2:
        raise ValueError("Qwen hyperconnection INT8 dequant output shape mismatch")
    output_rows, input_columns = weight.shape
    if group_size <= 0 or input_columns % group_size:
        raise ValueError("Qwen hyperconnection INT8 group size must divide input width")
    group_count = input_columns // group_size
    if weight_scale.shape != (output_rows, group_count):
        raise ValueError("Qwen hyperconnection INT8 weight scale shape mismatch")
    if weight.dtype is not torch.int8 or weight_scale.dtype is not torch.float16:
        raise TypeError("Qwen hyperconnection INT8 weight storage is invalid")
    if output.dtype not in (torch.bfloat16, torch.float16):
        raise TypeError("Qwen hyperconnection INT8 dequant output dtype is invalid")

    block_size = 256
    element_count = weight.numel()
    _dequantize_hyperconnection_weight_kernel[
        (triton.cdiv(element_count, block_size),)
    ](
        weight,
        weight_scale,
        output,
        element_count=element_count,
        input_columns=input_columns,
        group_size=group_size,
        group_count=group_count,
        block_size=block_size,
        num_warps=4,
    )


def _decode_fake(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    del weight_scale, group_size
    return inputs.new_empty((*inputs.shape[:-1], weight.shape[0]))


def _dequantize_fake(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    group_size: int,
) -> None:
    del weight, weight_scale, output, group_size


direct_register_custom_op(
    op_name="qwen4_exp_hyperconnection_int8_decode",
    op_func=_hyperconnection_int8_decode,
    fake_impl=_decode_fake,
)
direct_register_custom_op(
    op_name="qwen4_exp_dequantize_hyperconnection_int8_weight",
    op_func=_dequantize_hyperconnection_weight,
    mutates_args=["output"],
    fake_impl=_dequantize_fake,
)


def hyperconnection_int8_decode(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    """Run the one- or two-token compressed hyperconnection projection."""
    return torch.ops.vllm.qwen4_exp_hyperconnection_int8_decode(
        inputs,
        weight,
        weight_scale,
        group_size,
    )


def dequantize_hyperconnection_int8_weight(
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    output: torch.Tensor,
    group_size: int,
) -> None:
    """Dequantize one compressed projection into caller-owned workspace."""
    torch.ops.vllm.qwen4_exp_dequantize_hyperconnection_int8_weight(
        weight,
        weight_scale,
        output,
        group_size,
    )


__all__ = [
    "dequantize_hyperconnection_int8_weight",
    "hyperconnection_int8_decode",
]
