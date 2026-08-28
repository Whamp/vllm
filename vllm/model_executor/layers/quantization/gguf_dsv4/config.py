# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Quantization ownership for native DeepSeek V4 GGUF weights."""

from dataclasses import dataclass
from typing import Any

import regex as re
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
from vllm.model_executor.model_loader.gguf_dsv4_index import (
    GGUF_TYPE_SPECS_BY_NAME,
    GGUFTypeSpec,
)
from vllm.model_executor.model_loader.gguf_dsv4_profile import (
    GGUFDSV4SourceProfile,
)
from vllm.model_executor.utils import set_weight_attrs

_GROUPED_EXPERT_MIN_TOKENS = 128
_GROUPED_LINEAR_MIN_TOKENS = 8
_Q8_BLOCK_ELEMENTS = 32
_Q8_BLOCK_BYTES = 34
_LAYER_PREFIX_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")


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

_PROFILED_LINEAR_SOURCE_SUFFIXES = {
    ".attn.fused_wqa_wkv": ("attn_q_a.weight", "attn_kv.weight"),
    ".attn.wq_b": ("attn_q_b.weight",),
    ".attn.wo_a": ("attn_output_a.weight",),
    ".attn.wo_b": ("attn_output_b.weight",),
    ".attn.indexer.wq_b": ("indexer.attn_q_b.weight",),
    ".attn.indexer.weights_proj": ("indexer.proj.weight",),
    ".attn.compressor.fused_wkv_wgate": (
        "attn_compressor_kv.weight",
        "attn_compressor_gate.weight",
    ),
    ".attn.indexer.compressor.fused_wkv_wgate": (
        "indexer_compressor_kv.weight",
        "indexer_compressor_gate.weight",
    ),
    ".ffn.shared_experts.gate_up_proj": (
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
    ),
    ".ffn.shared_experts.down_proj": ("ffn_down_shexp.weight",),
}


class GGUFDSV4QuantConfig(QuantizationConfig):
    """Select native GGUF methods from an identity-bound source-type profile."""

    def __init__(self, source_profile: GGUFDSV4SourceProfile | None = None) -> None:
        super().__init__()
        self.source_profile = source_profile

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
        if "source_quant_types" not in config:
            return cls()
        return cls(GGUFDSV4SourceProfile.from_config(config))

    @staticmethod
    def _layer_index(prefix: str) -> int:
        match = _LAYER_PREFIX_PATTERN.search(prefix)
        if match is None:
            raise ValueError(f"GGUF DSv4 layer prefix has no layer index: {prefix}")
        return int(match.group(1))

    def _routed_source_types(self, prefix: str) -> tuple[str, str]:
        if self.source_profile is None:
            return "IQ2_XXS", "Q2_K"
        layer_index = self._layer_index(prefix)
        source_prefix = f"blk.{layer_index}.ffn"
        gate_type = self.source_profile.require_type(
            f"{source_prefix}_gate_exps.weight"
        )
        up_type = self.source_profile.require_type(f"{source_prefix}_up_exps.weight")
        if up_type != gate_type:
            raise ValueError(
                f"GGUF DSv4 layer {layer_index} gate/up types differ: "
                f"{gate_type} != {up_type}"
            )
        down_type = self.source_profile.require_type(
            f"{source_prefix}_down_exps.weight"
        )
        return gate_type, down_type

    def _linear_source_types(self, prefix: str) -> tuple[str, ...] | None:
        if self.source_profile is None:
            return None
        if prefix == "model.embed_tokens":
            return (self.source_profile.require_type("token_embd.weight"),)
        if prefix == "lm_head":
            return (self.source_profile.require_type("output.weight"),)
        for runtime_suffix, source_suffixes in _PROFILED_LINEAR_SOURCE_SUFFIXES.items():
            if prefix.endswith(runtime_suffix):
                layer_index = self._layer_index(prefix)
                return tuple(
                    self.source_profile.require_type(
                        f"blk.{layer_index}.{source_suffix}"
                    )
                    for source_suffix in source_suffixes
                )
        return None

    def get_quant_method(
        self, layer: torch.nn.Module, prefix: str
    ) -> QuantizeMethodBase | None:
        from vllm.model_executor.layers.vocab_parallel_embedding import (
            ParallelLMHead,
            VocabParallelEmbedding,
        )

        if isinstance(layer, RoutedExperts):
            gate_type, down_type = self._routed_source_types(prefix)
            return GGUFDSV4MoEMethod(
                self,
                layer.moe_config,
                gate_type_name=gate_type,
                down_type_name=down_type,
            )
        if self.source_profile is not None and isinstance(
            layer, (LinearBase, VocabParallelEmbedding)
        ):
            source_types = self._linear_source_types(prefix)
            if source_types is not None:
                return GGUFDSV4LinearMethod(
                    source_type_names=source_types,
                    profiled_quantization=True,
                )
        if isinstance(layer, ParallelLMHead):
            return GGUFDSV4LinearMethod()
        if isinstance(layer, LinearBase):
            if prefix.endswith(_Q8_LINEAR_SUFFIXES):
                return GGUFDSV4LinearMethod()
            return UnquantizedLinearMethod()
        return None


class GGUFDSV4LinearMethod(LinearMethodBase):
    """Own exact profiled GGUF rows and dispatch their native linear kernels."""

    def __init__(
        self,
        source_type_names: tuple[str, ...] = ("Q8_0",),
        *,
        profiled_quantization: bool = False,
    ) -> None:
        self.source_type_specs = tuple(
            self._require_linear_type(type_name) for type_name in source_type_names
        )
        self.profiled_quantization = profiled_quantization

    @staticmethod
    def _require_linear_type(type_name: str) -> GGUFTypeSpec:
        try:
            return GGUF_TYPE_SPECS_BY_NAME[type_name]
        except KeyError as error:
            raise ValueError(
                f"GGUF DSv4 linear uses unsupported type {type_name}"
            ) from error

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
        if self.profiled_quantization:
            if len(self.source_type_specs) != len(output_partition_sizes):
                raise ValueError(
                    "GGUF DSv4 profiled linear source/partition count mismatch: "
                    f"{len(self.source_type_specs)} != {len(output_partition_sizes)}"
                )
        elif self.source_type_specs != (GGUF_TYPE_SPECS_BY_NAME["Q8_0"],):
            raise ValueError("GGUF DSv4 legacy linear must use Q8_0")
        output_size_per_partition = sum(output_partition_sizes)
        if self.profiled_quantization:
            for partition_index, (type_spec, output_rows) in enumerate(
                zip(self.source_type_specs, output_partition_sizes, strict=True)
            ):
                if input_size_per_partition % type_spec.block_elements:
                    raise ValueError(
                        f"GGUF {type_spec.name} linear input partition must be "
                        f"divisible by {type_spec.block_elements}"
                    )
                row_bytes = (
                    input_size_per_partition
                    // type_spec.block_elements
                    * type_spec.block_bytes
                )
                parameter_name = (
                    "weight_raw"
                    if len(self.source_type_specs) == 1
                    else f"weight_raw_{partition_index}"
                )
                weight_raw = torch.nn.Parameter(
                    torch.empty(output_rows, row_bytes, dtype=torch.uint8),
                    requires_grad=False,
                )
                layer.register_parameter(parameter_name, weight_raw)
                set_weight_attrs(weight_raw, extra_weight_attrs)
        else:
            if input_size_per_partition % _Q8_BLOCK_ELEMENTS:
                raise ValueError(
                    "GGUF Q8 linear input partition must be divisible by 32"
                )
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

    def _raw_parameter_name(self, partition_index: int) -> str:
        if self.profiled_quantization and len(self.source_type_specs) > 1:
            return f"weight_raw_{partition_index}"
        return "weight_raw"

    @staticmethod
    def _register_prepared_q8(
        layer: torch.nn.Module,
        prepared: GGUFQ8MarlinWeights,
        suffix: str,
    ) -> None:
        layer.register_parameter(
            f"weight{suffix}",
            torch.nn.Parameter(prepared.weight, requires_grad=False),
        )
        layer.register_parameter(
            f"weight_scale{suffix}",
            torch.nn.Parameter(prepared.scales, requires_grad=False),
        )
        layer.register_buffer(
            f"workspace{suffix}", prepared.workspace, persistent=False
        )
        layer.register_buffer(
            f"empty_indices{suffix}", prepared.empty_indices, persistent=False
        )

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if all(type_spec.name == "Q8_0" for type_spec in self.source_type_specs):
            if self.profiled_quantization and len(self.source_type_specs) > 1:
                raw_parameters = [
                    getattr(layer, self._raw_parameter_name(partition_index))
                    for partition_index in range(len(self.source_type_specs))
                ]
                weight_raw = torch.cat(raw_parameters, dim=0).contiguous()
                for partition_index, _ in enumerate(self.source_type_specs):
                    delattr(layer, self._raw_parameter_name(partition_index))
            else:
                weight_raw = layer.weight_raw
                delattr(layer, "weight_raw")
            prepared = prepare_gguf_q8_0_marlin(
                weight_raw,
                input_columns=layer.input_size_per_partition,
                scale_dtype=layer.params_dtype,
            )
            self._register_prepared_q8(layer, prepared, "")
            return
        for partition_index, type_spec in enumerate(self.source_type_specs):
            if type_spec.name != "Q8_0":
                continue
            raw_name = self._raw_parameter_name(partition_index)
            weight_raw = getattr(layer, raw_name)
            prepared = prepare_gguf_q8_0_marlin(
                weight_raw,
                input_columns=layer.input_size_per_partition,
                scale_dtype=layer.params_dtype,
            )
            delattr(layer, raw_name)
            self._register_prepared_q8(layer, prepared, f"_{partition_index}")

    @staticmethod
    def _apply_q8_partition(
        layer: torch.nn.Module,
        x: torch.Tensor,
        output_rows: int,
        suffix: str,
    ) -> torch.Tensor:
        prepared = GGUFQ8MarlinWeights(
            weight=getattr(layer, f"weight{suffix}"),
            scales=getattr(layer, f"weight_scale{suffix}"),
            workspace=getattr(layer, f"workspace{suffix}"),
            empty_indices=getattr(layer, f"empty_indices{suffix}"),
            input_columns=layer.input_size_per_partition,
            output_rows=output_rows,
        )
        return apply_gguf_q8_0_marlin(x, prepared)

    @staticmethod
    def _apply_raw_k_quant(
        type_name: str,
        activation_scales: torch.Tensor,
        activation_codes: torch.Tensor,
        weights: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        use_grouped = activation_codes.shape[0] >= _GROUPED_LINEAR_MIN_TOKENS
        if type_name == "Q4_K":
            if use_grouped:
                torch.ops._C.gguf_q4_k_q8_1_grouped_matmul(
                    activation_scales, activation_codes, weights, output
                )
            else:
                torch.ops._C.gguf_q4_k_q8_1_raw_matvec(
                    activation_scales, activation_codes, weights, output
                )
        elif type_name == "Q5_K":
            if use_grouped:
                torch.ops._C.gguf_q5_k_q8_1_grouped_matmul(
                    activation_scales, activation_codes, weights, output
                )
            else:
                torch.ops._C.gguf_q5_k_q8_1_raw_matvec(
                    activation_scales, activation_codes, weights, output
                )
        elif type_name == "Q6_K":
            if use_grouped:
                torch.ops._C.gguf_q6_k_q8_1_grouped_matmul(
                    activation_scales, activation_codes, weights, output
                )
            else:
                torch.ops._C.gguf_q6_k_q8_1_raw_matvec(
                    activation_scales, activation_codes, weights, output
                )
        else:
            raise RuntimeError(
                f"GGUF DSv4 linear execution is not registered for {type_name}"
            )

    def apply(
        self,
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if bias is not None:
            raise ValueError("GGUF DSv4 linear does not support bias")
        if all(type_spec.name == "Q8_0" for type_spec in self.source_type_specs):
            return self._apply_q8_partition(
                layer, x, layer.output_size_per_partition, ""
            )
        flat_input = x.reshape(-1, x.shape[-1])
        activation_scales = torch.empty(
            flat_input.shape[0],
            flat_input.shape[1] // _Q8_BLOCK_ELEMENTS,
            device=x.device,
            dtype=torch.float16,
        )
        activation_codes = torch.empty_like(flat_input, dtype=torch.int8)
        torch.ops._C.gguf_quantize_bf16_to_q8_1(
            flat_input, activation_scales, activation_codes
        )
        outputs: list[torch.Tensor] = []
        for partition_index, (type_spec, output_rows) in enumerate(
            zip(
                self.source_type_specs,
                layer.output_partition_sizes,
                strict=True,
            )
        ):
            if type_spec.name == "Q8_0":
                output = self._apply_q8_partition(
                    layer, flat_input, output_rows, f"_{partition_index}"
                )
            else:
                output_fp32 = torch.empty(
                    flat_input.shape[0],
                    output_rows,
                    device=x.device,
                    dtype=torch.float32,
                )
                self._apply_raw_k_quant(
                    type_spec.name,
                    activation_scales,
                    activation_codes,
                    getattr(layer, self._raw_parameter_name(partition_index)),
                    output_fp32,
                )
                output = output_fp32.to(x.dtype)
            outputs.append(output)
        combined = torch.cat(outputs, dim=-1)
        return combined.reshape(*x.shape[:-1], combined.shape[-1])

    def embedding(self, layer: torch.nn.Module, input_: torch.Tensor) -> torch.Tensor:
        if tuple(type_spec.name for type_spec in self.source_type_specs) != ("Q4_K",):
            type_names = ",".join(
                type_spec.name for type_spec in self.source_type_specs
            )
            raise RuntimeError(
                f"GGUF DSv4 embedding execution is not registered for {type_names}"
            )
        output = torch.empty(
            *input_.shape,
            layer.input_size_per_partition,
            device=input_.device,
            dtype=layer.params_dtype,
        )
        torch.ops._C.gguf_q4_k_embedding(input_, layer.weight_raw, output)
        return output


class GGUFDSV4MoEMethod(FusedMoEMethodBase):
    """Execute all 256 profiled GGUF experts with TP-sharded widths."""

    def __init__(
        self,
        quant_config: GGUFDSV4QuantConfig,
        moe: FusedMoEConfig,
        *,
        gate_type_name: str = "IQ2_XXS",
        down_type_name: str = "Q2_K",
    ) -> None:
        super().__init__(moe)
        self.quant_config = quant_config
        self.gate_type_spec = self._require_expert_type(gate_type_name)
        self.down_type_spec = self._require_expert_type(down_type_name)

    @staticmethod
    def _require_expert_type(type_name: str) -> GGUFTypeSpec:
        try:
            return GGUF_TYPE_SPECS_BY_NAME[type_name]
        except KeyError as error:
            raise ValueError(
                f"GGUF DSv4 experts use unsupported type {type_name}"
            ) from error

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
        if hidden_size % self.gate_type_spec.block_elements:
            raise ValueError(
                f"GGUF {self.gate_type_spec.name} hidden size must be divisible by "
                f"{self.gate_type_spec.block_elements}"
            )
        if intermediate_size_per_partition % self.down_type_spec.block_elements:
            raise ValueError(
                f"GGUF {self.down_type_spec.name} intermediate partition must be "
                f"divisible by {self.down_type_spec.block_elements}"
            )
        gate_row_bytes = (
            hidden_size
            // self.gate_type_spec.block_elements
            * self.gate_type_spec.block_bytes
        )
        down_row_bytes = (
            intermediate_size_per_partition
            // self.down_type_spec.block_elements
            * self.down_type_spec.block_bytes
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
        if self.gate_type_spec.name == "IQ1_S":
            if token_count < _GROUPED_EXPERT_MIN_TOKENS:
                torch.ops._C.gguf_iq1_s_q8_1_indexed_gate_up(
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
            torch.ops._C.gguf_iq1_s_q8_1_grouped_gate_up(
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
        if self.gate_type_spec.name == "IQ1_M":
            if token_count < _GROUPED_EXPERT_MIN_TOKENS:
                torch.ops._C.gguf_iq1_m_q8_1_indexed_gate_up(
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
            torch.ops._C.gguf_iq1_m_q8_1_grouped_gate_up(
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
        if self.gate_type_spec.name != "IQ2_XXS":
            raise RuntimeError(
                f"GGUF DSv4 {self.gate_type_spec.name} gate/up execution is not "
                "registered"
            )
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
        if self.down_type_spec.name == "IQ3_XXS":
            if gate_up.schedule is None:
                torch.ops._C.gguf_iq3_xxs_q8_1_indexed_down(
                    down_scales,
                    down_codes,
                    layer.down_raw,
                    topk_ids,
                    down_output,
                )
            else:
                torch.ops._C.gguf_iq3_xxs_q8_1_grouped_down(
                    down_scales,
                    down_codes,
                    layer.down_raw,
                    *gate_up.schedule,
                    down_output,
                    topk,
                )
        elif self.down_type_spec.name == "MXFP4":
            if gate_up.schedule is None:
                torch.ops._C.gguf_mxfp4_q8_1_indexed_down(
                    down_scales,
                    down_codes,
                    layer.down_raw,
                    topk_ids,
                    down_output,
                )
            else:
                torch.ops._C.gguf_mxfp4_q8_1_grouped_down(
                    down_scales,
                    down_codes,
                    layer.down_raw,
                    *gate_up.schedule,
                    down_output,
                    topk,
                )
        elif self.down_type_spec.name != "Q2_K":
            raise RuntimeError(
                f"GGUF DSv4 {self.down_type_spec.name} down execution is not registered"
            )
        elif gate_up.schedule is None:
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
