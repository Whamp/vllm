# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.gguf_dsv4 import (
    GGUFDSV4LinearMethod,
    GGUFDSV4MoEMethod,
    GGUFDSV4QuantConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead


def _empty_module(module_type):
    module = object.__new__(module_type)
    torch.nn.Module.__init__(module)
    return module


def test_gguf_dsv4_quant_config_selects_only_native_surfaces() -> None:
    assert get_quantization_config("gguf_dsv4") is GGUFDSV4QuantConfig
    config = GGUFDSV4QuantConfig()
    linear = _empty_module(LinearBase)
    lm_head = _empty_module(ParallelLMHead)
    experts = _empty_module(RoutedExperts)
    experts.moe_config = object()

    assert isinstance(
        config.get_quant_method(linear, "model.layers.0.attn.fused_wqa_wkv"),
        GGUFDSV4LinearMethod,
    )
    assert isinstance(
        config.get_quant_method(linear, "model.layers.0.ffn.gate"),
        UnquantizedLinearMethod,
    )
    assert isinstance(config.get_quant_method(lm_head, "lm_head"), GGUFDSV4LinearMethod)
    assert isinstance(
        config.get_quant_method(experts, "model.layers.0.ffn.experts"),
        GGUFDSV4MoEMethod,
    )


def test_gguf_dsv4_linear_allocates_exact_q8_rows() -> None:
    method = GGUFDSV4LinearMethod()
    layer = torch.nn.Module()

    method.create_weights(
        layer,
        input_size_per_partition=4096,
        output_partition_sizes=[1024, 512],
        input_size=4096,
        output_size=1536,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.weight_raw.dtype == torch.uint8
    assert layer.weight_raw.shape == (1536, 4096 // 32 * 34)
    assert layer.weight_raw.numel() == 1536 * 4352


def test_gguf_dsv4_moe_allocates_all_experts_with_tp_intermediate_shard() -> None:
    method = GGUFDSV4MoEMethod(quant_config=GGUFDSV4QuantConfig(), moe=object())
    layer = torch.nn.Module()

    method.create_weights(
        layer,
        num_experts=64,
        global_num_experts=256,
        hidden_size=4096,
        intermediate_size_per_partition=512,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert layer.gate_raw.shape == (256, 512, 4096 // 256 * 66)
    assert layer.up_raw.shape == layer.gate_raw.shape
    assert layer.down_raw.shape == (256, 4096, 512 // 256 * 84)
    assert sum(parameter.numel() for parameter in layer.parameters()) == (
        2 * 256 * 512 * 1056 + 256 * 4096 * 168
    )
