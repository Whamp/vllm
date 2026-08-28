# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import torch

from vllm.v1.core.kv_cache_utils import (
    is_dcp_exempt_spec,
    resolve_kv_cache_block_sizes,
)
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    KVCacheGroupSpec,
    MLAAttentionSpec,
    SlidingWindowMLASpec,
)


def _compressed_mla_spec() -> MLAAttentionSpec:
    return MLAAttentionSpec(
        block_size=256,
        num_kv_heads=1,
        head_size=512,
        dtype=torch.uint8,
        cache_dtype_str="fp8_ds_mla",
        model_version="deepseek_v4",
        compress_ratio=4,
    )


def _compressor_state_spec() -> SlidingWindowMLASpec:
    return SlidingWindowMLASpec(
        block_size=4,
        num_kv_heads=1,
        head_size=2048,
        dtype=torch.float32,
        sliding_window=8,
    )


def _vllm_config(dcp_world_size: int, enable_prefix_caching: bool):
    return SimpleNamespace(
        cache_config=SimpleNamespace(
            block_size=256,
            enable_prefix_caching=enable_prefix_caching,
            prefix_match_unit=None,
        ),
        parallel_config=SimpleNamespace(
            decode_context_parallel_size=dcp_world_size
        ),
        kv_transfer_config=None,
    )


def test_sm86_dcp_exempts_only_replicated_sliding_window_groups(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    assert not is_dcp_exempt_spec(_compressed_mla_spec())
    assert is_dcp_exempt_spec(_compressor_state_spec())


def test_sm86_dcp_exemption_is_disabled_by_default(monkeypatch) -> None:
    monkeypatch.delenv("VLLM_SM86_DCP", raising=False)
    assert not is_dcp_exempt_spec(_compressor_state_spec())


def test_sm86_dcp_block_geometry_matches_worker_ownership(monkeypatch) -> None:
    monkeypatch.setenv("VLLM_SM86_DCP", "1")
    compressed = _compressed_mla_spec()
    state = _compressor_state_spec()
    groups = [
        KVCacheGroupSpec(["layer.compressed"], compressed),
        KVCacheGroupSpec(["layer.state"], state),
    ]
    kv_cache_config = KVCacheConfig(
        num_blocks=1024,
        kv_cache_tensors=[],
        kv_cache_groups=groups,
    )
    vllm_config = _vllm_config(4, enable_prefix_caching=True)

    scheduler_block_size, hash_block_size = resolve_kv_cache_block_sizes(
        kv_cache_config, vllm_config
    )

    assert scheduler_block_size == 1024
    assert hash_block_size == 4
    assert compressed.max_num_blocks_per_req(vllm_config, 4096) == 4
    assert state.max_num_blocks_per_req(vllm_config, 4096) == 1024
