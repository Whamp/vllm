# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def test_cuda_split_k_decode_requires_sufficient_shared_memory() -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mla

    assert not sparse_mla.cuda_split_k_decode_supported(101_376)
    assert sparse_mla.cuda_split_k_decode_supported(166_912)


def test_cuda_split_k_decode_dispatches_by_shared_memory(monkeypatch) -> None:
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_mla

    monkeypatch.setattr(sparse_mla.current_platform, "is_cuda", lambda: True)
    sparse_mla._use_split_k_decode.cache_clear()

    monkeypatch.setattr(sparse_mla, "get_max_shared_memory_bytes", lambda: 101_376)
    assert not sparse_mla._use_split_k_decode()

    sparse_mla._use_split_k_decode.cache_clear()
    monkeypatch.setattr(sparse_mla, "get_max_shared_memory_bytes", lambda: 166_912)
    assert sparse_mla._use_split_k_decode()

    sparse_mla._use_split_k_decode.cache_clear()
