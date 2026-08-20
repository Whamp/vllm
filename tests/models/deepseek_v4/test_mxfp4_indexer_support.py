# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest

from vllm.v1.attention.ops import mqa_logits_triton


@pytest.mark.parametrize(
    ("is_cuda", "capability", "is_sm100_family", "expected"),
    [
        (True, (8, 6), False, True),
        (True, (10, 0), True, True),
        (True, (8, 0), False, False),
        (True, (8, 9), False, False),
        (True, (9, 0), False, False),
        (True, (12, 0), False, False),
        (False, (8, 6), False, False),
    ],
)
def test_supports_mxfp4_indexer_cache(
    monkeypatch,
    is_cuda: bool,
    capability: tuple[int, int],
    is_sm100_family: bool,
    expected: bool,
) -> None:
    platform = SimpleNamespace(
        is_cuda=lambda: is_cuda,
        is_device_capability_family=lambda family: family == 100 and is_sm100_family,
        get_device_capability=lambda: capability,
    )
    monkeypatch.setattr(mqa_logits_triton, "current_platform", platform)

    assert mqa_logits_triton.supports_mxfp4_indexer_cache() is expected
