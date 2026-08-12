# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.models.deepseek_v4.nvidia.model import (
    _make_deepseek_v4_weights_mapper,
    _uses_compressed_tensors_fp8_linears,
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def test_compressed_tensors_fp8_linear_gate_requires_explicit_fallback() -> None:
    config = SimpleNamespace(
        quantization_config={
            "quant_method": "compressed-tensors",
            "config_groups": {
                "experts": {"format": "pack-quantized", "targets": ["RoutedExperts"]}
            },
        }
    )
    assert not _uses_compressed_tensors_fp8_linears(config)

    config.quantization_config["config_groups"]["linears"] = {
        "format": "float-quantized",
        "targets": ["Linear"],
    }
    assert _uses_compressed_tensors_fp8_linears(config)


def test_compressed_tensors_mapper_keeps_fp8_linear_scale_name() -> None:
    mapper = _make_deepseek_v4_weights_mapper("fp4", compressed_tensors_hybrid=True)

    assert mapper._map_name("layers.0.attn.wq_a.scale") == (
        "model.layers.0.attn.wq_a.weight_scale"
    )
    assert mapper._map_name("layers.0.ffn.experts.0.w1.weight_scale") == (
        "model.layers.0.ffn.experts.0.w1.weight_scale"
    )
