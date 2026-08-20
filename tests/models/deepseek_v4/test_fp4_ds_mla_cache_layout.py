# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

from vllm.config import CacheConfig
from vllm.models.deepseek_v4.attention import _resolve_dsv4_kv_cache_dtype
from vllm.models.deepseek_v4.cache_layout import (
    FP4_DS_MLA_CACHE_LAYOUT,
    FP8_DS_MLA_CACHE_LAYOUT,
    get_deepseek_v4_cache_layout,
)
from vllm.models.deepseek_v4.sparse_mla import DeepseekV4FlashMLABackend
from vllm.utils.torch_utils import STR_DTYPE_TO_TORCH_DTYPE
from vllm.v1.attention.backends.mla.sparse_swa import DeepseekSparseSWABackend
from vllm.v1.kv_cache_interface import MLAAttentionSpec, SlidingWindowMLASpec


def test_fp4_ds_mla_layout_contract() -> None:
    layout = FP4_DS_MLA_CACHE_LAYOUT

    assert layout.cache_dtype == "fp4_ds_mla"
    assert layout.nope_dim == 448
    assert layout.rope_dim == 64
    assert layout.quant_group_size == 32
    assert layout.nope_data_bytes == 224
    assert layout.rope_data_offset == 224
    assert layout.rope_data_bytes == 128
    assert layout.token_data_bytes == 352
    assert layout.num_scale_groups == 14
    assert layout.scale_bytes == 16
    assert layout.row_bytes == 368
    assert layout.block_alignment == 352


def test_fp8_ds_mla_layout_contract_is_unchanged() -> None:
    layout = FP8_DS_MLA_CACHE_LAYOUT

    assert layout.cache_dtype == "fp8_ds_mla"
    assert layout.nope_dim == 448
    assert layout.rope_dim == 64
    assert layout.quant_group_size == 64
    assert layout.nope_data_bytes == 448
    assert layout.rope_data_offset == 448
    assert layout.rope_data_bytes == 128
    assert layout.token_data_bytes == 576
    assert layout.num_scale_groups == 7
    assert layout.scale_bytes == 8
    assert layout.row_bytes == 584
    assert layout.block_alignment == 576


def test_cache_layout_lookup_fails_closed() -> None:
    assert get_deepseek_v4_cache_layout("fp4_ds_mla") is FP4_DS_MLA_CACHE_LAYOUT
    assert get_deepseek_v4_cache_layout("fp8_ds_mla") is FP8_DS_MLA_CACHE_LAYOUT

    try:
        get_deepseek_v4_cache_layout("fp8")
    except ValueError as error:
        assert "Unsupported DeepSeek V4 cache layout" in str(error)
    else:
        raise AssertionError("non-canonical cache dtype must fail closed")


def test_fp4_ds_mla_cache_dtype_is_accepted_by_config() -> None:
    config = CacheConfig(cache_dtype="fp4_ds_mla")
    assert config.cache_dtype == "fp4_ds_mla"


def test_fp4_ds_mla_global_storage_dtype_is_uint8() -> None:
    assert STR_DTYPE_TO_TORCH_DTYPE["fp4_ds_mla"] == torch.uint8


def test_fp4_ds_mla_dtype_resolves_to_uint8_without_rewrite() -> None:
    config = CacheConfig(cache_dtype="fp4_ds_mla")

    cache_dtype, torch_dtype = _resolve_dsv4_kv_cache_dtype(
        True, config.cache_dtype, config
    )

    assert cache_dtype == "fp4_ds_mla"
    assert torch_dtype == torch.uint8
    assert config.cache_dtype == "fp4_ds_mla"


def test_fp4_ds_mla_backend_shape_uses_physical_row_bytes() -> None:
    assert DeepseekV4FlashMLABackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp4_ds_mla",
    ) == (3, 64, 368)
    assert DeepseekV4FlashMLABackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp8_ds_mla",
    ) == (3, 64, 584)


def test_fp4_ds_mla_swa_backend_shape_uses_physical_row_bytes() -> None:
    assert DeepseekSparseSWABackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp4_ds_mla",
    ) == (3, 64, 368)
    assert DeepseekSparseSWABackend.get_kv_cache_shape(
        num_blocks=3,
        block_size=64,
        num_kv_heads=1,
        head_size=512,
        cache_dtype_str="fp8_ds_mla",
    ) == (3, 64, 584)


def test_fp4_ds_mla_spec_accounts_for_compressed_storage_rows() -> None:
    spec = MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        compress_ratio=4,
        cache_dtype_str="fp4_ds_mla",
        alignment=FP4_DS_MLA_CACHE_LAYOUT.block_alignment,
        model_version="deepseek_v4",
        physical_row_bytes=FP4_DS_MLA_CACHE_LAYOUT.row_bytes,
    )

    assert spec.storage_block_size == 64
    assert spec.real_page_size_bytes == 64 * 368
    assert spec.page_size_bytes == 67 * 352


def test_fp4_ds_mla_sliding_window_spec_accounts_for_physical_rows() -> None:
    spec = SlidingWindowMLASpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        sliding_window=128,
        compress_ratio=1,
        cache_dtype_str="fp4_ds_mla",
        alignment=FP4_DS_MLA_CACHE_LAYOUT.block_alignment,
        model_version="deepseek_v4",
        physical_row_bytes=FP4_DS_MLA_CACHE_LAYOUT.row_bytes,
    )

    assert spec.storage_block_size == 256
    assert spec.real_page_size_bytes == 256 * 368
    assert spec.page_size_bytes == 268 * 352
    assert SlidingWindowMLASpec.merge([spec, spec]).physical_row_bytes == 368
