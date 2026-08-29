# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from enum import Enum

import torch
import torch.nn.functional as F

from vllm.model_executor.layers.linear import (
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.utils import replace_parameter

_HYPERCONNECTION_INT8_GROUP_SIZE = 128
_HYPERCONNECTION_INT8_MAX = 127


class HyperconnectionInt8ScaleLayout(str, Enum):
    """Scale layout for Qwen hyperconnection INT8 weights."""

    PER_ROW = "per_row"
    K_GROUP_128 = "k_group_128"


class Qwen4ExpHyperconnectionInt8LinearMethod(LinearMethodBase):
    """Quantize Qwen hyperconnection BF16 weights after checkpoint loading."""

    def __init__(
        self,
        *,
        scale_layout: HyperconnectionInt8ScaleLayout,
        valid_output_rows: int | None = None,
    ) -> None:
        self.scale_layout = scale_layout
        self.valid_output_rows = valid_output_rows
        self._unquantized_loader = UnquantizedLinearMethod()

    def create_weights(
        self,
        layer: torch.nn.Module,
        input_size_per_partition: int,
        output_partition_sizes: list[int],
        input_size: int,
        output_size: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        """Create the ordinary BF16 load parameter before post-load conversion."""
        self._unquantized_loader.create_weights(
            layer,
            input_size_per_partition,
            output_partition_sizes,
            input_size,
            output_size,
            params_dtype,
            **extra_weight_attrs,
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        """Replace the loaded BF16 weight with INT8 codes and FP16 scales."""
        if getattr(layer, "_qwen_hyperconnection_int8_processed", False):
            return

        weight = layer.weight
        if weight.ndim != 2:
            raise ValueError(
                "Qwen hyperconnection INT8 weight must be two-dimensional, "
                f"got shape {tuple(weight.shape)}"
            )
        if weight.dtype not in (torch.bfloat16, torch.float16, torch.float32):
            raise TypeError(
                "Qwen hyperconnection INT8 source weight must be floating point, "
                f"got {weight.dtype}"
            )

        output_rows, input_columns = weight.shape
        valid_output_rows = (
            output_rows if self.valid_output_rows is None else self.valid_output_rows
        )
        if not 0 < valid_output_rows <= output_rows:
            raise ValueError(
                "Qwen hyperconnection INT8 valid output rows must be in "
                f"[1, {output_rows}], got {valid_output_rows}"
            )

        valid_weight = weight[:valid_output_rows].float()
        if self.scale_layout is HyperconnectionInt8ScaleLayout.K_GROUP_128:
            if input_columns % _HYPERCONNECTION_INT8_GROUP_SIZE:
                raise ValueError(
                    "Qwen hyperconnection K-group INT8 requires the input width "
                    f"to be divisible by {_HYPERCONNECTION_INT8_GROUP_SIZE}, "
                    f"got {input_columns}"
                )
            group_count = input_columns // _HYPERCONNECTION_INT8_GROUP_SIZE
            grouped_weight = valid_weight.reshape(
                valid_output_rows,
                group_count,
                _HYPERCONNECTION_INT8_GROUP_SIZE,
            )
            valid_scale = grouped_weight.abs().amax(dim=2) / _HYPERCONNECTION_INT8_MAX
            valid_scale = torch.where(
                valid_scale == 0,
                torch.ones_like(valid_scale),
                valid_scale,
            ).to(torch.float16)
            valid_codes = (
                grouped_weight.div(valid_scale.float().unsqueeze(-1))
                .round()
                .clamp(-_HYPERCONNECTION_INT8_MAX, _HYPERCONNECTION_INT8_MAX)
                .to(torch.int8)
                .reshape(valid_output_rows, input_columns)
            )
            scale_columns = group_count
        else:
            valid_scale = (
                valid_weight.abs().amax(dim=1, keepdim=True) / _HYPERCONNECTION_INT8_MAX
            )
            valid_scale = torch.where(
                valid_scale == 0,
                torch.ones_like(valid_scale),
                valid_scale,
            ).to(torch.float16)
            valid_codes = (
                valid_weight.div(valid_scale.float())
                .round()
                .clamp(-_HYPERCONNECTION_INT8_MAX, _HYPERCONNECTION_INT8_MAX)
                .to(torch.int8)
            )
            scale_columns = 1

        quantized_weight = torch.zeros_like(weight, dtype=torch.int8)
        weight_scale = torch.ones(
            (output_rows, scale_columns),
            dtype=torch.float16,
            device=weight.device,
        )
        quantized_weight[:valid_output_rows].copy_(valid_codes)
        weight_scale[:valid_output_rows].copy_(valid_scale)

        replace_parameter(layer, "weight", quantized_weight)
        replace_parameter(layer, "weight_scale", weight_scale)
        layer._qwen_hyperconnection_int8_processed = True

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Run direct INT8 decode or workspace-backed BF16 prefill."""
        if bias is not None:
            raise ValueError("Qwen hyperconnection INT8 projections do not use bias")

        from vllm.platforms import current_platform

        if (
            not current_platform.is_cuda()
            or current_platform.get_device_capability() != (8, 6)
        ):
            raise RuntimeError("Qwen hyperconnection INT8 execution requires CUDA SM86")
        if not getattr(layer, "_qwen_hyperconnection_int8_processed", False):
            raise RuntimeError(
                "Qwen hyperconnection INT8 weight was not processed after loading"
            )
        if x.shape[-1] != layer.weight.shape[1]:
            raise ValueError(
                "Qwen hyperconnection INT8 input width mismatch: "
                f"expected {layer.weight.shape[1]}, got {x.shape[-1]}"
            )

        from vllm.models.qwen4_exp.nvidia.ops.hc_int8 import (
            dequantize_hyperconnection_int8_weight,
            hyperconnection_int8_decode,
        )

        original_shape = x.shape
        flat_input = x.reshape(-1, original_shape[-1]).contiguous()
        group_size = (
            _HYPERCONNECTION_INT8_GROUP_SIZE
            if self.scale_layout is HyperconnectionInt8ScaleLayout.K_GROUP_128
            else layer.weight.shape[1]
        )
        if flat_input.shape[0] <= 2:
            output = hyperconnection_int8_decode(
                flat_input,
                layer.weight,
                layer.weight_scale,
                group_size,
            )
        else:
            from vllm.v1.worker.workspace import current_workspace_manager

            (dequantized_weight,) = current_workspace_manager().get_simultaneous(
                (tuple(layer.weight.shape), x.dtype)
            )
            dequantize_hyperconnection_int8_weight(
                layer.weight,
                layer.weight_scale,
                dequantized_weight,
                group_size,
            )
            output = F.linear(flat_input, dequantized_weight)
        return output.reshape(*original_shape[:-1], layer.weight.shape[0])
