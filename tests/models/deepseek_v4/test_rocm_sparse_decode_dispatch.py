# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest


@pytest.mark.skip_global_cleanup
def test_sparse_decode_partial_kernel_selection_preserves_gfx950_and_grouped_paths(
    monkeypatch,
):
    from vllm.v1.attention.ops import rocm_aiter_mla_sparse as sparse_decode

    monkeypatch.setattr(sparse_decode, "_ON_GFX950", True)
    assert sparse_decode._select_sparse_decode_partial_kernel(0) is (
        sparse_decode._sparse_attn_decode_gfx950_partial_kernel
    )
    assert sparse_decode._select_sparse_decode_partial_kernel(4) is (
        sparse_decode._sparse_attn_decode_partial_blocked_kernel
    )

    monkeypatch.setattr(sparse_decode, "_ON_GFX950", False)
    assert sparse_decode._select_sparse_decode_partial_kernel(0) is (
        sparse_decode._sparse_attn_decode_partial_kernel
    )
