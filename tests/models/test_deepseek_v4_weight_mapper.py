# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v4.nvidia.model import _make_deepseek_v4_weights_mapper
from vllm.models.deepseek_v4.quant_config import DeepseekV4FP8Config


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def _hybrid_quantization_config() -> dict:
    return {
        "quant_method": "compressed-tensors",
        "base_quant_method": "deepseek_v4_fp8",
        "format": "pack-quantized",
        "config_groups": {
            "experts": {
                "format": "pack-quantized",
                "targets": ["model.layers.0.ffn.experts.0.gate_proj"],
                "weights": {
                    "num_bits": 2,
                    "type": "int",
                    "strategy": "group",
                    "group_size": 128,
                    "symmetric": True,
                    "dynamic": False,
                },
                "input_activations": None,
                "output_activations": None,
            }
        },
    }


def test_hybrid_config_overrides_to_deepseek_native_fp8() -> None:
    assert (
        DeepseekV4FP8Config.override_quantization_method(
            _hybrid_quantization_config(),
            None,
            SimpleNamespace(model_type="deepseek_v4"),
        )
        == "deepseek_v4_fp8"
    )


def test_hybrid_config_delegates_only_routed_experts(monkeypatch) -> None:
    from vllm.models.deepseek_v4 import quant_config as quant_config_module

    class FakeRoutedExperts:
        pass

    monkeypatch.setattr(quant_config_module, "RoutedExperts", FakeRoutedExperts)
    config = DeepseekV4FP8Config.from_config(_hybrid_quantization_config())
    delegated_method = object()
    config._compressed_tensors_config.get_quant_method = lambda layer, prefix: (
        delegated_method
    )

    method = config.get_quant_method(FakeRoutedExperts(), "model.layers.0.ffn.experts")
    assert method is delegated_method


def test_hybrid_config_resolves_projection_specific_group_sizes(monkeypatch) -> None:
    from vllm.models.deepseek_v4 import quant_config as quant_config_module

    class FakeRoutedExperts:
        moe_config = object()

    monkeypatch.setattr(quant_config_module, "RoutedExperts", FakeRoutedExperts)
    config_payload = _hybrid_quantization_config()
    gate_group = config_payload["config_groups"]["experts"]
    gate_group["targets"] = [
        "model.layers.0.ffn.experts.0.gate_proj",
        "model.layers.0.ffn.experts.0.up_proj",
    ]
    gate_group["weights"]["group_size"] = 512
    config_payload["config_groups"]["down"] = {
        **gate_group,
        "targets": ["model.layers.0.ffn.experts.0.down_proj"],
        "weights": {**gate_group["weights"], "group_size": 128},
    }

    config = DeepseekV4FP8Config.from_config(config_payload)
    compressed_config = config._compressed_tensors_config
    assert compressed_config is not None
    gate = compressed_config.get_scheme_dict(
        FakeRoutedExperts(),
        "model.layers.0.ffn.experts.0.gate_proj",
    )
    down = compressed_config.get_scheme_dict(
        FakeRoutedExperts(),
        "model.layers.0.ffn.experts.0.down_proj",
    )

    assert gate is not None and down is not None
    assert gate["weights"].group_size == 512
    assert down["weights"].group_size == 128


def test_hybrid_mapper_keeps_native_fp8_linear_scale_name() -> None:
    mapper = _make_deepseek_v4_weights_mapper("fp4")

    assert mapper._map_name("layers.0.attn.wq_a.scale") == (
        "model.layers.0.attn.wq_a.weight_scale_inv"
    )
    assert mapper._map_name("layers.0.ffn.experts.0.w1.weight_scale") == (
        "model.layers.0.ffn.experts.0.w1.weight_scale"
    )
