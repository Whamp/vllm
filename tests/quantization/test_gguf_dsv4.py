# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

import pytest
import torch

from vllm.model_executor.layers.fused_moe import RoutedExperts
from vllm.model_executor.layers.linear import LinearBase, UnquantizedLinearMethod
from vllm.model_executor.layers.quantization import get_quantization_config
from vllm.model_executor.layers.quantization.gguf_dsv4 import (
    GGUFDSV4LinearMethod,
    GGUFDSV4MoEMethod,
    GGUFDSV4QuantConfig,
)
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)


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


def test_gguf_dsv4_profile_allocates_mixed_linear_and_q4_embedding() -> None:
    source_types = {
        "blk.0.attn_q_a.weight": "Q5_K",
        "blk.0.attn_kv.weight": "Q8_0",
        "token_embd.weight": "Q4_K",
    }
    profile_bytes = json.dumps(
        source_types, sort_keys=True, separators=(",", ":")
    ).encode()
    config = GGUFDSV4QuantConfig.from_config(
        {
            "source_quant_types": source_types,
            "source_quant_types_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        }
    )
    linear = _empty_module(LinearBase)
    linear_method = config.get_quant_method(linear, "model.layers.0.attn.fused_wqa_wkv")
    assert isinstance(linear_method, GGUFDSV4LinearMethod)
    linear_layer = torch.nn.Module()

    linear_method.create_weights(
        linear_layer,
        input_size_per_partition=4096,
        output_partition_sizes=[1024, 512],
        input_size=4096,
        output_size=1536,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert linear_layer.weight_raw_0.shape == (1024, 4096 // 256 * 176)
    assert linear_layer.weight_raw_1.shape == (512, 4096 // 32 * 34)

    embedding = _empty_module(VocabParallelEmbedding)
    embedding_method = config.get_quant_method(embedding, "model.embed_tokens")
    assert isinstance(embedding_method, GGUFDSV4LinearMethod)
    embedding_layer = torch.nn.Module()
    embedding_method.create_weights(
        embedding_layer,
        input_size_per_partition=4096,
        output_partition_sizes=[32320],
        input_size=4096,
        output_size=129280,
        params_dtype=torch.bfloat16,
        weight_loader=lambda *args, **kwargs: None,
    )

    assert embedding_layer.weight_raw.shape == (32320, 4096 // 256 * 144)


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


@pytest.mark.parametrize(
    ("gate_type", "gate_block_bytes", "down_type", "down_block_bytes"),
    [
        ("IQ1_S", 50, "IQ3_XXS", 98),
        ("IQ1_M", 56, "MXFP4", 17),
    ],
)
def test_gguf_dsv4_moe_allocates_profiled_compressed_types(
    gate_type: str,
    gate_block_bytes: int,
    down_type: str,
    down_block_bytes: int,
) -> None:
    source_types = {
        "blk.0.ffn_gate_exps.weight": gate_type,
        "blk.0.ffn_up_exps.weight": gate_type,
        "blk.0.ffn_down_exps.weight": down_type,
    }
    profile_bytes = json.dumps(
        source_types, sort_keys=True, separators=(",", ":")
    ).encode()
    config = GGUFDSV4QuantConfig.from_config(
        {
            "source_quant_types": source_types,
            "source_quant_types_sha256": hashlib.sha256(profile_bytes).hexdigest(),
        }
    )
    experts = _empty_module(RoutedExperts)
    experts.moe_config = object()
    method = config.get_quant_method(experts, "model.layers.0.ffn.experts")
    assert isinstance(method, GGUFDSV4MoEMethod)
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

    assert layer.gate_raw.shape == (256, 512, 4096 // 256 * gate_block_bytes)
    assert layer.up_raw.shape == layer.gate_raw.shape
    down_block_elements = 32 if down_type == "MXFP4" else 256
    assert layer.down_raw.shape == (
        256,
        4096,
        512 // down_block_elements * down_block_bytes,
    )


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
