# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

import vllm.model_executor.custom_op as custom_op
from vllm.model_executor.layers.rotary_embedding.deepseek_scaling_rope import (
    DeepseekV4ScalingRotaryEmbedding,
)
from vllm.models.deepseek_v4.common.rope import build_deepseek_v4_rope


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def _deepseek_v4_rope_config() -> SimpleNamespace:
    return SimpleNamespace(
        rope_parameters={
            "rope_type": "yarn",
            "factor": 4,
            "original_max_position_embeddings": 16,
            "beta_fast": 32,
            "beta_slow": 1,
        },
        rope_theta=10_000,
        compress_rope_theta=160_000,
    )


@pytest.fixture(autouse=True)
def disable_optional_rope_backend_probe(monkeypatch: pytest.MonkeyPatch) -> None:
    compilation_config = SimpleNamespace(
        custom_ops=["none"],
        enabled_custom_ops=set(),
        disabled_custom_ops=set(),
    )
    monkeypatch.setattr(
        custom_op,
        "get_cached_compilation_config",
        lambda: compilation_config,
    )
    monkeypatch.setattr(
        DeepseekV4ScalingRotaryEmbedding,
        "enabled",
        classmethod(lambda cls: False),
    )


def test_deepseek_v4_rope_cache_is_bounded_by_runtime_context() -> None:
    bounded = build_deepseek_v4_rope(
        _deepseek_v4_rope_config(),
        head_dim=8,
        rope_head_dim=4,
        max_position_embeddings=64,
        max_model_len=20,
        compress_ratio=1,
    )
    full = build_deepseek_v4_rope(
        _deepseek_v4_rope_config(),
        head_dim=8,
        rope_head_dim=4,
        max_position_embeddings=64,
        max_model_len=80,
        compress_ratio=1,
    )

    assert bounded.cos_sin_cache.shape == (20, 4)
    assert full.cos_sin_cache.shape == (64, 4)
    torch.testing.assert_close(
        bounded.cos_sin_cache,
        full.cos_sin_cache[:20],
        rtol=0,
        atol=0,
    )
    assert bounded is not full


def test_deepseek_v4_rope_cache_reuses_matching_runtime_context() -> None:
    first = build_deepseek_v4_rope(
        _deepseek_v4_rope_config(),
        head_dim=8,
        rope_head_dim=4,
        max_position_embeddings=64,
        max_model_len=20,
        compress_ratio=4,
    )
    second = build_deepseek_v4_rope(
        _deepseek_v4_rope_config(),
        head_dim=8,
        rope_head_dim=4,
        max_position_embeddings=64,
        max_model_len=20,
        compress_ratio=4,
    )

    assert first is second
