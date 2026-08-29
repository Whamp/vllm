# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from hypothesis import event, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

from vllm.models.qwen4_exp.nvidia import (
    model as _qwen4_exp_model,  # noqa: F401
)
from vllm.models.qwen4_exp.nvidia import qsa as qsa_backend
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON
from vllm.v1.attention.ops.int4_per_token_head import single_rht
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVQuantMode


@settings(max_examples=150, deadline=None)
@given(
    num_blocks=st.integers(min_value=1, max_value=4),
    num_kv_heads=st.integers(min_value=1, max_value=4),
    block_size=st.sampled_from([16, 32, 64]),
    head_size=st.sampled_from([16, 32, 64, 128, 256]),
    physical_layout=st.sampled_from(["HND", "NHD"]),
    scale_offset=st.integers(min_value=-31, max_value=31),
)
def test_qsa_q8k_q4v_views_follow_declared_cache_layout(
    num_blocks: int,
    num_kv_heads: int,
    block_size: int,
    head_size: int,
    physical_layout: str,
    scale_offset: int,
) -> None:
    event(f"layout={physical_layout}")
    event(f"head_size={head_size}")

    key_bytes = head_size
    value_bytes = head_size // 2
    expected_content_bytes = key_bytes + 4 + value_bytes + 4
    spec = FullAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_size,
        head_size_v=head_size,
        dtype=torch.uint8,
        kv_quant_mode=KVQuantMode.QSA_Q8K_Q4V,
    )
    customized = qsa_backend.Qwen4ExpQSAFlashAttentionBackend.customize_spec(spec)

    assert "qsa_q8k_q4v" in (
        qsa_backend.Qwen4ExpQSAFlashAttentionBackend.supported_kv_cache_dtypes
    )
    assert customized.state_content_bytes == expected_content_bytes
    logical_shape = qsa_backend.Qwen4ExpQSAFlashAttentionBackend.get_kv_cache_shape(
        num_blocks,
        block_size,
        num_kv_heads,
        head_size,
        "qsa_q8k_q4v",
    )
    assert logical_shape == (
        num_blocks,
        num_kv_heads,
        block_size,
        expected_content_bytes,
    )

    if physical_layout == "HND":
        raw_cache = torch.zeros(logical_shape, dtype=torch.uint8)
    else:
        physical = torch.zeros(
            num_blocks,
            block_size,
            num_kv_heads,
            expected_content_bytes,
            dtype=torch.uint8,
        )
        raw_cache = physical.permute(0, 2, 1, 3)

    impl = qsa_backend.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_backend.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.head_size = head_size
    impl._kv_quant_mode = KVQuantMode.QSA_Q8K_Q4V
    key_data, key_scales, value_data, value_scales = impl.q8k_q4v_cache_views(raw_cache)
    canonical_shape = (num_blocks, block_size, num_kv_heads)
    assert key_data.shape == (*canonical_shape, head_size)
    assert key_data.dtype == torch.int8
    assert key_scales.shape == canonical_shape
    assert key_scales.dtype == torch.float32
    assert value_data.shape == (*canonical_shape, value_bytes)
    assert value_data.dtype == torch.uint8
    assert value_scales.shape == canonical_shape
    assert value_scales.dtype == torch.float32

    scale_values = (
        torch.arange(
            num_blocks * block_size * num_kv_heads,
            dtype=torch.float32,
        ).reshape(canonical_shape)
        + scale_offset
        + 0.25
    )
    key_data.fill_(-3)
    value_data.fill_(0xA5)
    key_scales.copy_(scale_values)
    value_scales.copy_(-scale_values)

    assert torch.all(raw_cache[..., :key_bytes] == 0xFD)
    key_scale_raw = raw_cache[..., key_bytes : key_bytes + 4].view(torch.float32)
    value_start = key_bytes + 4
    assert torch.all(raw_cache[..., value_start : value_start + value_bytes] == 0xA5)
    value_scale_raw = raw_cache[
        ..., value_start + value_bytes : expected_content_bytes
    ].view(torch.float32)
    torch.testing.assert_close(key_scale_raw.squeeze(-1), scale_values.permute(0, 2, 1))
    torch.testing.assert_close(
        value_scale_raw.squeeze(-1), -scale_values.permute(0, 2, 1)
    )


def test_qsa_q8k_q4v_constructor_selects_mixed_quant_mode(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def initialize_flash_impl(instance, *args, **kwargs) -> None:
        del kwargs
        instance.kv_cache_dtype = args[6]
        instance.dcp_world_size = 1

    monkeypatch.setattr(
        qsa_backend.FlashAttentionImpl,
        "__init__",
        initialize_flash_impl,
    )
    monkeypatch.setattr(
        qsa_backend,
        "is_flash_attn_varlen_func_available",
        lambda: True,
    )

    implementation = qsa_backend.Qwen4ExpQSAFlashAttentionImpl(
        6,
        256,
        256**-0.5,
        1,
        None,
        None,
        "qsa_q8k_q4v",
        None,
        qsa_backend.AttentionType.DECODER,
        None,
    )

    assert implementation.kv_cache_dtype == "qsa_q8k_q4v"
    assert implementation._kv_quant_mode == KVQuantMode.QSA_Q8K_Q4V


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA Q8-K/Q4-V writer requires CUDA and Triton",
)
def test_qsa_q8k_q4v_writer_reconstructs_and_replays_on_caller_stream() -> None:
    from vllm.models.qwen4_exp.nvidia.ops.qsa_q8k_q4v import (
        reshape_and_cache_q8k_q4v,
    )

    torch.manual_seed(20260829)
    token_count = 5
    head_size = 256
    block_size = 16
    key = torch.randn((token_count, 1, head_size), dtype=torch.bfloat16, device="cuda")
    value = torch.randn_like(key)
    slot_mapping = torch.tensor([0, 17, -1, 3, 31], dtype=torch.int64, device="cuda")
    raw_cache = torch.full(
        (2, 1, block_size, head_size + 4 + head_size // 2 + 4),
        0xCC,
        dtype=torch.uint8,
        device="cuda",
    )
    impl = qsa_backend.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_backend.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.head_size = head_size
    impl._kv_quant_mode = KVQuantMode.QSA_Q8K_Q4V
    key_data, key_scales, value_data, value_scales = impl.q8k_q4v_cache_views(raw_cache)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        reshape_and_cache_q8k_q4v(
            key,
            value,
            key_data,
            key_scales,
            value_data,
            value_scales,
            slot_mapping,
        )
    stream.synchronize()

    transformed_key = single_rht(key.float()).to(key.dtype).float()
    transformed_value = single_rht(value.float()).to(value.dtype).float()
    for token_index, slot in enumerate(slot_mapping.cpu().tolist()):
        if slot < 0:
            continue
        block_index, slot_index = divmod(slot, block_size)
        key_reconstructed = (
            key_data[block_index, slot_index, 0].float()
            * key_scales[block_index, slot_index, 0]
        )
        key_error = torch.linalg.vector_norm(
            key_reconstructed - transformed_key[token_index, 0]
        ) / torch.linalg.vector_norm(transformed_key[token_index, 0])
        assert key_error < 0.02

        scale_raw = value_scales[block_index, slot_index, 0]
        scale_bits = scale_raw.view(torch.int32)
        value_zero_point = (scale_bits & 0xF).float()
        value_scale = (scale_bits & -16).view(torch.float32)
        packed = value_data[block_index, slot_index, 0]
        value_codes = torch.stack((packed & 0xF, (packed >> 4) & 0xF), dim=1)
        value_reconstructed = (
            value_codes.flatten().float() - value_zero_point
        ) * value_scale
        value_error = torch.linalg.vector_norm(
            value_reconstructed - transformed_value[token_index, 0]
        ) / torch.linalg.vector_norm(transformed_value[token_index, 0])
        assert value_error < 0.20

    untouched = raw_cache[0, 0, 1]
    assert torch.all(untouched == 0xCC)

    for _ in range(3):
        reshape_and_cache_q8k_q4v(
            key,
            value,
            key_data,
            key_scales,
            value_data,
            value_scales,
            slot_mapping,
        )
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        reshape_and_cache_q8k_q4v(
            key,
            value,
            key_data,
            key_scales,
            value_data,
            value_scales,
            slot_mapping,
        )
    graph.replay()
    first = raw_cache.clone()
    graph.replay()
    second = raw_cache.clone()
    torch.accelerator.synchronize()
    assert torch.equal(first, second)


@composite
def _qsa_mixed_sparse_cases(draw) -> tuple[int, int, list[list[int]]]:
    query_rows = draw(st.integers(min_value=1, max_value=4))
    cache_tokens = draw(st.integers(min_value=16, max_value=64))
    selection_width = draw(st.integers(min_value=16, max_value=64))
    selections = []
    for _ in range(query_rows):
        first = draw(st.integers(min_value=0, max_value=cache_tokens - 1))
        rest = draw(
            st.lists(
                st.integers(min_value=-1, max_value=cache_tokens - 1),
                min_size=selection_width - 1,
                max_size=selection_width - 1,
            )
        )
        selections.append([first, *rest])
    return query_rows, cache_tokens, selections


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason="QSA Q8-K/Q4-V reader requires CUDA and Triton",
)
@given(case=_qsa_mixed_sparse_cases())
@settings(max_examples=50, deadline=None)
def test_qsa_q8k_q4v_sparse_attention_tracks_bf16_reference(
    case: tuple[int, int, list[list[int]]],
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.qsa_q8k_q4v import (
        qsa_sparse_paged_attention_q8k_q4v,
        reshape_and_cache_q8k_q4v,
    )

    query_rows, cache_tokens, selections = case
    head_size = 256
    query_heads = 6
    block_size = 16
    num_blocks = (cache_tokens + block_size - 1) // block_size
    generator = torch.Generator(device="cuda").manual_seed(
        query_rows * 1000003 + cache_tokens * 1009 + len(selections[0])
    )
    query = torch.randn(
        (query_rows, query_heads, head_size),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    key = torch.randn(
        (cache_tokens, 1, head_size),
        generator=generator,
        dtype=torch.bfloat16,
        device="cuda",
    )
    value = torch.randn_like(key)
    raw_cache = torch.zeros(
        (num_blocks, 1, block_size, head_size + 4 + head_size // 2 + 4),
        dtype=torch.uint8,
        device="cuda",
    )
    impl = qsa_backend.Qwen4ExpQSAFlashAttentionImpl.__new__(
        qsa_backend.Qwen4ExpQSAFlashAttentionImpl
    )
    impl.head_size = head_size
    impl._kv_quant_mode = KVQuantMode.QSA_Q8K_Q4V
    key_data, key_scales, value_data, value_scales = impl.q8k_q4v_cache_views(raw_cache)
    slot_mapping = torch.arange(cache_tokens, dtype=torch.int64, device="cuda")
    reshape_and_cache_q8k_q4v(
        key,
        value,
        key_data,
        key_scales,
        value_data,
        value_scales,
        slot_mapping,
    )
    logical_indices = torch.tensor(selections, dtype=torch.int32, device="cuda")
    block_table = torch.arange(num_blocks, dtype=torch.int32, device="cuda")[None]
    token_to_request = torch.zeros(query_rows, dtype=torch.int32, device="cuda")

    actual = qsa_sparse_paged_attention_q8k_q4v(
        query,
        key_data,
        key_scales,
        value_data,
        value_scales,
        logical_indices,
        block_table,
        token_to_request,
    )
    expected = torch.empty_like(actual)
    for row_index, row_selection in enumerate(selections):
        valid = [index for index in row_selection if index >= 0]
        selected_key = key[valid, 0].float()
        selected_value = value[valid, 0].float()
        scores = query[row_index].float() @ selected_key.T / head_size**0.5
        expected[row_index] = (torch.softmax(scores, dim=-1) @ selected_value).to(
            expected.dtype
        )

    error = actual.float() - expected.float()
    normalized_rmse = torch.linalg.vector_norm(error) / torch.linalg.vector_norm(
        expected.float()
    )
    cosine = torch.nn.functional.cosine_similarity(
        actual.float().reshape(1, -1),
        expected.float().reshape(1, -1),
    )
    assert normalized_rmse <= 0.17
    assert cosine >= 0.985
