# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import weakref

import pytest
import torch

from vllm.model_executor.models.qwen2_5_vl import Qwen2_5_VisionTransformer
from vllm.model_executor.models.qwen3_vl import Qwen3_VisionTransformer
from vllm.utils.cache import LRUCache

pytestmark = [pytest.mark.cpu_test, pytest.mark.skip_global_cleanup]


def test_qwen3_vision_rope_cache_is_owned_by_each_model() -> None:
    first_model = object.__new__(Qwen3_VisionTransformer)
    second_model = object.__new__(Qwen3_VisionTransformer)
    first_model._rot_pos_ids_cache = LRUCache(capacity=1024)
    second_model._rot_pos_ids_cache = LRUCache(capacity=1024)

    first_result = first_model.rot_pos_ids(2, 2, 1)
    repeated_result = first_model.rot_pos_ids(2, 2, 1)
    second_result = second_model.rot_pos_ids(2, 2, 1)

    expected = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]])
    torch.testing.assert_close(first_result, expected)
    assert repeated_result is first_result
    assert second_result is not first_result
    torch.testing.assert_close(second_result, expected)


def test_qwen2_5_vision_rope_cache_does_not_retain_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    legacy_cache_clear = getattr(
        Qwen2_5_VisionTransformer.get_rope_by_thw,
        "cache_clear",
        None,
    )
    if legacy_cache_clear is not None:
        legacy_cache_clear()

    monkeypatch.setattr(
        Qwen2_5_VisionTransformer,
        "get_window_index_thw",
        lambda self, t, h, w: (
            torch.tensor([0]),
            torch.tensor([1], dtype=torch.int32),
        ),
    )
    monkeypatch.setattr(
        Qwen2_5_VisionTransformer,
        "rotary_pos_emb_thw",
        lambda self, t, h, w: (
            torch.zeros((1, 1, 1)),
            torch.zeros((1, 1, 1)),
        ),
    )

    model = object.__new__(Qwen2_5_VisionTransformer)
    model._rope_by_thw_cache = LRUCache(capacity=1024)
    first_result = model.get_rope_by_thw(1, 1, 1)
    repeated_result = model.get_rope_by_thw(1, 1, 1)

    assert repeated_result is first_result
    torch.testing.assert_close(first_result[0], torch.zeros((1, 1)))
    torch.testing.assert_close(first_result[1], torch.zeros((1, 1)))
    torch.testing.assert_close(first_result[2], torch.tensor([0]))
    torch.testing.assert_close(first_result[3], torch.tensor([1], dtype=torch.int32))
    torch.testing.assert_close(first_result[4], torch.tensor([1], dtype=torch.int32))

    model_ref = weakref.ref(model)
    del model
    gc.collect()

    try:
        assert model_ref() is None
    finally:
        if legacy_cache_clear is not None:
            legacy_cache_clear()
