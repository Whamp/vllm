# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.nvidia import qsa as qsa_backend


def test_bound_qwen4_exp_rope_cache_reclaims_storage_idempotently() -> None:
    original = torch.arange(128 * 16, dtype=torch.bfloat16).reshape(128, 16)
    rope = SimpleNamespace(cos_sin_cache=original)

    qsa_backend.bound_qwen4_exp_rope_cache(rope, 32)

    assert rope.cos_sin_cache.shape == (32, 16)
    assert rope.cos_sin_cache.untyped_storage().nbytes() == 32 * 16 * 2
    torch.testing.assert_close(rope.cos_sin_cache, original[:32])
    data_ptr = rope.cos_sin_cache.data_ptr()
    qsa_backend.bound_qwen4_exp_rope_cache(rope, 32)
    assert rope.cos_sin_cache.data_ptr() == data_ptr


@pytest.mark.parametrize("bound", [0, -1])
def test_bound_qwen4_exp_rope_cache_rejects_nonpositive_bound(bound: int) -> None:
    rope = SimpleNamespace(cos_sin_cache=torch.zeros(32, 16))

    with pytest.raises(ValueError, match="RoPE cache bound must be positive"):
        qsa_backend.bound_qwen4_exp_rope_cache(rope, bound)


def test_bound_qwen4_exp_rope_cache_rejects_too_few_source_rows() -> None:
    rope = SimpleNamespace(cos_sin_cache=torch.zeros(31, 16))

    with pytest.raises(ValueError, match="Qwen4Exp RoPE cache has 31 rows"):
        qsa_backend.bound_qwen4_exp_rope_cache(rope, 32)
