# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.models.deepseek_v4.common.ops import fused_compress_quant_cache


class _RecordingKernel:
    def __init__(self) -> None:
        self.kwargs: dict[str, object] | None = None

    def __getitem__(self, grid):
        assert grid == (1,)

        def launch(*args, **kwargs) -> None:
            self.kwargs = kwargs

        return launch


def _launch_compressor(
    monkeypatch: pytest.MonkeyPatch, head_dim: int, use_fp4_cache: bool
) -> _RecordingKernel:
    kernel = _RecordingKernel()
    if head_dim == 512:
        kernel_name = "_fused_kv_compress_norm_rope_insert_sparse_attn"
    elif use_fp4_cache:
        kernel_name = "_fused_kv_compress_norm_rope_insert_indexer_mxfp4_attn"
    else:
        kernel_name = "_fused_kv_compress_norm_rope_insert_indexer_attn"
    monkeypatch.setattr(fused_compress_quant_cache, kernel_name, kernel)
    monkeypatch.setattr(
        fused_compress_quant_cache.triton,
        "next_power_of_2",
        lambda value: value,
        raising=False,
    )

    fused_compress_quant_cache.compress_norm_rope_store_triton(
        state_cache=torch.empty((1, 1, head_dim * 2)),
        num_actual=1,
        token_to_req_indices=torch.zeros(1, dtype=torch.int32),
        positions=torch.zeros(1, dtype=torch.int64),
        slot_mapping=torch.zeros(1, dtype=torch.int64),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=1,
        state_width=head_dim,
        cos_sin_cache=torch.empty((1, 64)),
        kv_cache=torch.empty((1, 1, 368 if head_dim == 512 else head_dim)),
        k_cache_metadata=SimpleNamespace(
            slot_mapping=torch.zeros(1, dtype=torch.int64)
        ),
        pdl_kwargs={},
        head_dim=head_dim,
        rope_head_dim=64,
        compress_ratio=4,
        overlap=True,
        use_fp4_cache=use_fp4_cache,
        rms_norm_weight=torch.empty(head_dim),
        rms_norm_eps=1e-6,
        quant_block=32 if use_fp4_cache else 64,
        token_stride=352 if head_dim == 512 and use_fp4_cache else head_dim,
        scale_dim=16 if head_dim == 512 and use_fp4_cache else 4,
    )
    assert kernel.kwargs is not None
    return kernel


def test_fp4_ds_mla_compressor_passes_fp4_constexpr(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kernel = _launch_compressor(monkeypatch, head_dim=512, use_fp4_cache=True)
    assert kernel.kwargs is not None
    assert kernel.kwargs["USE_FP4_CACHE"] is True


@pytest.mark.parametrize("use_fp4_cache", [False, True])
def test_indexer_compressor_does_not_receive_ds_mla_fp4_constexpr(
    monkeypatch: pytest.MonkeyPatch,
    use_fp4_cache: bool,
) -> None:
    kernel = _launch_compressor(monkeypatch, head_dim=128, use_fp4_cache=use_fp4_cache)
    assert kernel.kwargs is not None
    assert "USE_FP4_CACHE" not in kernel.kwargs
