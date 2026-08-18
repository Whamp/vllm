# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization ownership for native DeepSeek V4 GGUF weights."""

from dataclasses import dataclass
from typing import Any

import torch

from vllm.model_executor.layers.fused_moe import (
    FusedMoEConfig,
    FusedMoEMethodBase,
    RoutedExperts,
    SharedExperts,
)
from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
    moe_align_block_size,
)
from vllm.model_executor.layers.linear import (
    LinearBase,
    LinearMethodBase,
    UnquantizedLinearMethod,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
    QuantizeMethodBase,
)
from vllm.model_executor.layers.quantization.gguf_dsv4.q8_0_marlin import (
    GGUFQ8MarlinWeights,
    apply_gguf_q8_0_marlin,
    prepare_gguf_q8_0_marlin,
)
from vllm.model_executor.utils import set_weight_attrs

_GROUPED_EXPERT_MIN_TOKENS = 128
_IQ2_BLOCK_ELEMENTS = 256
_IQ2_BLOCK_BYTES = 66
_Q2_BLOCK_ELEMENTS = 256
_Q2_BLOCK_BYTES = 84
_Q8_BLOCK_ELEMENTS = 32
_Q8_BLOCK_BYTES = 34


@dataclass(frozen=True)
class _GateUpResult:
    gate: torch.Tensor
    up: torch.Tensor
    schedule: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None


_Q8_LINEAR_SUFFIXES = (
    ".attn.fused_wqa_wkv",
    ".attn.wq_b",
    ".attn.wo_a",
    ".attn.wo_b",
    ".ffn.shared_experts.gate_up_proj",
    ".ffn.shared_experts.down_proj",
)


class GGUFDSV4QuantConfig(QuantizationConfig):
    """Select native GGUF methods only for routed experts and Q8 linears."""

    @classmethod
    def get_name(cls):
        return "gguf_dsv4"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return [torch.bfloat16]

    @classmethod
    def get_min_capability(cls) -> int:
        return 86

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        return []

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> "GGUFDSV4QuantConfig":
        return cls()

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        if isinstance(layer, RoutedExperts):
            return GGUFDSV4MoEMethod(self, layer.moe_config)
        if isinstance(layer, LinearBase):
            if prefix == "lm_head" or prefix.endswith(_Q8_LINEAR_SUFFIXES):
                return GGUFDSV4LinearMethod()
            return UnquantizedLinearMethod()
        return None


class GGUFDSV4LinearMethod(LinearMethodBase):
    """Own raw Q8_0 rows until one byte-neutral load-time Marlin repack."""

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
        if input_size_per_partition % _Q8_BLOCK_ELEMENTS:
            raise ValueError("GGUF Q8 linear input partition must be divisible by 32")
        output_size_per_partition = sum(output_partition_sizes)
        row_bytes = input_size_per_partition // _Q8_BLOCK_ELEMENTS * _Q8_BLOCK_BYTES
        weight_raw = torch.nn.Parameter(
            torch.empty(output_size_per_partition, row_bytes, dtype=torch.uint8),
            requires_grad=False,
        )
        layer.register_parameter("weight_raw", weight_raw)
        set_weight_attrs(weight_raw, extra_weight_attrs)
        layer.input_size = input_size
        layer.output_size = output_size
        layer.input_size_per_partition = input_size_per_partition
        layer.output_size_per_partition = output_size_per_partition
        layer.output_partition_sizes = output_partition_sizes
        layer.params_dtype = params_dtype

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        prepared = prepare_gguf_q8_0_marlin(
            layer.weight_raw,
            input_columns=layer.input_size_per_partition,
            scale_dtype=layer.params_dtype,
        )
        delattr(layer, "weight_raw")
        layer.register_parameter(
            "weight",
            torch.nn.Parameter(prepared.weight, requires_grad=False),
        )
        layer.register_parameter(
            "weight_scale",
            torch.nn.Parameter(prepared.scales, requires_grad=False),
        )
        layer.register_buffer("workspace", prepared.workspace, persistent=False)
        layer.register_buffer("empty_indices", prepared.empty_indices, persistent=False)

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("GGUF Q8 linear does not support bias")
        prepared = GGUFQ8MarlinWeights(
            weight=layer.weight,
            scales=layer.weight_scale,
            workspace=layer.workspace,
            empty_indices=layer.empty_indices,
            input_columns=layer.input_size_per_partition,
            output_rows=layer.output_size_per_partition,
        )
        return apply_gguf_q8_0_marlin(x, prepared)


class GGUFDSV4MoEMethod(FusedMoEMethodBase):
    """Execute all 256 raw GGUF experts with TP-sharded intermediate widths."""

    def __init__(self, quant_config: GGUFDSV4QuantConfig, moe: FusedMoEConfig) -> None:
        super().__init__(moe)
        self.quant_config = quant_config

    @property
    def topk_indices_dtype(self) -> torch.dtype | None:
        return torch.int32

    def maybe_roundup_sizes(
        self,
        hidden_size: int,
        intermediate_size_per_partition: int,
        act_dtype: torch.dtype,
        moe_parallel_config,
    ) -> tuple[int, int]:
        return hidden_size, intermediate_size_per_partition

    def create_weights(
        self,
        layer: RoutedExperts,
        num_experts: int,
        hidden_size: int,
        intermediate_size_per_partition: int,
        params_dtype: torch.dtype,
        **extra_weight_attrs,
    ) -> None:
        del num_experts, params_dtype
        global_num_experts = int(extra_weight_attrs.pop("global_num_experts"))
        if hidden_size % _IQ2_BLOCK_ELEMENTS:
            raise ValueError("GGUF IQ2 hidden size must be divisible by 256")
        if intermediate_size_per_partition % _Q2_BLOCK_ELEMENTS:
            raise ValueError("GGUF Q2 intermediate partition must be divisible by 256")
        gate_row_bytes = hidden_size // _IQ2_BLOCK_ELEMENTS * _IQ2_BLOCK_BYTES
        down_row_bytes = (
            intermediate_size_per_partition // _Q2_BLOCK_ELEMENTS * _Q2_BLOCK_BYTES
        )
        for name in ("gate_raw", "up_raw"):
            parameter = torch.nn.Parameter(
                torch.empty(
                    global_num_experts,
                    intermediate_size_per_partition,
                    gate_row_bytes,
                    dtype=torch.uint8,
                ),
                requires_grad=False,
            )
            layer.register_parameter(name, parameter)
            set_weight_attrs(parameter, extra_weight_attrs)
        down_raw = torch.nn.Parameter(
            torch.empty(
                global_num_experts,
                hidden_size,
                down_row_bytes,
                dtype=torch.uint8,
            ),
            requires_grad=False,
        )
        layer.register_parameter("down_raw", down_raw)
        set_weight_attrs(down_raw, extra_weight_attrs)

    def get_fused_moe_quant_config(self, layer: RoutedExperts):
        return None

    def _run_gate_up(
        self,
        layer: RoutedExperts,
        hidden_states: torch.Tensor,
        topk_ids: torch.Tensor,
    ) -> _GateUpResult:
        token_count = hidden_states.shape[0]
        topk = topk_ids.shape[1]
        intermediate_size = layer.intermediate_size_per_partition
        gate_scales = torch.empty(
            token_count,
            hidden_states.shape[1] // _Q8_BLOCK_ELEMENTS,
            device=hidden_states.device,
            dtype=torch.float16,
        )
        gate_codes = torch.empty_like(hidden_states, dtype=torch.int8)
        gate_output = torch.empty(
            token_count,
            topk,
            intermediate_size,
            device=hidden_states.device,
            dtype=torch.float32,
        )
        up_output = torch.empty_like(gate_output)
        torch.ops._C.gguf_quantize_bf16_to_q8_1(hidden_states, gate_scales, gate_codes)
        if token_count < _GROUPED_EXPERT_MIN_TOKENS:
            torch.ops._C.gguf_iq2_xxs_q8_1_indexed_gate_up(
                gate_scales,
                gate_codes,
                layer.gate_raw,
                layer.up_raw,
                topk_ids,
                gate_output,
                up_output,
            )
            return _GateUpResult(gate_output, up_output, None)

        schedule = moe_align_block_size(
            topk_ids=topk_ids,
            block_size=8,
            num_experts=layer.global_num_experts,
        )
        torch.ops._C.gguf_iq2_xxs_q8_1_grouped_gate_up(
            gate_scales,
            gate_codes,
            layer.gate_raw,
            layer.up_raw,
            *schedule,
            gate_output,
            up_output,
            topk,
        )
        return _GateUpResult(gate_output, up_output, schedule)

    def _run_down(
        self,
        layer: RoutedExperts,
        hidden_states: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        gate_up: _GateUpResult,
    ) -> torch.Tensor:
        token_count, topk, intermediate_size = gate_up.gate.shape
        assignment_count = token_count * topk
        down_scales = torch.empty(
            assignment_count,
            intermediate_size // _Q8_BLOCK_ELEMENTS,
            device=hidden_states.device,
            dtype=torch.float16,
        )
        down_codes = torch.empty(
            assignment_count,
            intermediate_size,
            device=hidden_states.device,
            dtype=torch.int8,
        )
        if layer.swiglu_limit is None:
            raise ValueError("GGUF DSv4 requires a SwiGLU clamp limit")
        torch.ops._C.gguf_swiglu_weighted_q8_1(
            gate_up.gate,
            gate_up.up,
            topk_weights,
            down_scales,
            down_codes,
            float(layer.swiglu_limit),
        )
        down_output = torch.empty(
            token_count,
            topk,
            hidden_states.shape[1],
            device=hidden_states.device,
            dtype=torch.float32,
        )
        if gate_up.schedule is None:
            torch.ops._C.gguf_q2_k_q8_1_indexed_down(
                down_scales,
                down_codes,
                layer.down_raw,
                topk_ids,
                down_output,
            )
        else:
            torch.ops._C.gguf_q2_k_q8_1_grouped_down(
                down_scales,
                down_codes,
                layer.down_raw,
                *gate_up.schedule,
                down_output,
            )
        return down_output.sum(dim=1).to(hidden_states.dtype)

    def apply(
        self,
        layer: RoutedExperts,
        x: torch.Tensor,
        topk_weights: torch.Tensor,
        topk_ids: torch.Tensor,
        shared_experts: SharedExperts | None,
        shared_experts_input: torch.Tensor | None,
    ) -> torch.Tensor:
        del shared_experts, shared_experts_input
        if layer.use_ep:
            raise ValueError("GGUF DSv4 experts support tensor parallelism only")
        if layer.apply_router_weight_on_input:
            raise ValueError(
                "GGUF DSv4 folds router weights after gate/up, not on input"
            )
        original_shape = x.shape
        hidden_states = x.reshape(-1, x.shape[-1])
        gate_up = self._run_gate_up(layer, hidden_states, topk_ids)
        output = self._run_down(layer, hidden_states, topk_weights, topk_ids, gate_up)
        return output.reshape(*original_shape[:-1], output.shape[-1])
