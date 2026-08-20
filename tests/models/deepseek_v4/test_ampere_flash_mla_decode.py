# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import importlib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from vllm.models.deepseek_v4.ampere import ampere_sparse


def test_flash_mla_decode_dependency_fails_closed(monkeypatch) -> None:
    def missing_flash_mla(name: str):
        assert name == "flash_mla"
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", missing_flash_mla)

    with pytest.raises(RuntimeError, match="FlashMLA decode was enabled"):
        ampere_sparse.load_ampere_flash_mla_decode("fp8_ds_mla")


def test_flash_mla_decode_dependency_requires_fp8_operator(monkeypatch) -> None:
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(),
    )

    with pytest.raises(RuntimeError, match="sparse_mla_decode_fp8"):
        ampere_sparse.load_ampere_flash_mla_decode("fp8_ds_mla")


def test_flash_mla_decode_dependency_returns_operator(monkeypatch) -> None:
    expected = Mock()
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(sparse_mla_decode_fp8=expected),
    )

    assert ampere_sparse.load_ampere_flash_mla_decode("fp8_ds_mla") is expected


def test_flash_mla_decode_loads_fp4_operator(monkeypatch) -> None:
    expected = Mock()
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(sparse_mla_decode_fp4=expected),
    )

    assert ampere_sparse.load_ampere_flash_mla_decode("fp4_ds_mla") is expected


def test_flash_mla_prefill_loads_fp4_operator(monkeypatch) -> None:
    expected = Mock()
    monkeypatch.setattr(
        importlib,
        "import_module",
        lambda name: SimpleNamespace(sparse_mla_prefill_fp4=expected),
    )

    assert ampere_sparse.load_ampere_flash_mla_prefill("fp4_ds_mla") is expected


def test_flash_mla_prefill_rejects_fp8_cache() -> None:
    with pytest.raises(ValueError, match="supports fp4_ds_mla"):
        ampere_sparse.load_ampere_flash_mla_prefill("fp8_ds_mla")


def test_fp4_cache_requires_native_flash_mla_decode(monkeypatch) -> None:
    monkeypatch.setattr(ampere_sparse.envs, "VLLM_DSV4_FLASH_MLA_DECODE", False)
    layer = object.__new__(ampere_sparse.DeepseekV4AmpereMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_dtype = "fp4_ds_mla"
    monkeypatch.setattr(
        ampere_sparse.DeepseekV4ROCMAiterMLAAttention,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    with pytest.raises(ValueError, match="requires native AppMana FlashMLA"):
        ampere_sparse.DeepseekV4AmpereMLAAttention.__init__(layer)


def test_flash_mla_decode_is_opt_in(monkeypatch) -> None:
    monkeypatch.setattr(ampere_sparse.envs, "VLLM_DSV4_FLASH_MLA_DECODE", False)
    load_decode = Mock()
    monkeypatch.setattr(ampere_sparse, "load_ampere_flash_mla_decode", load_decode)

    layer = object.__new__(ampere_sparse.DeepseekV4AmpereMLAAttention)
    torch.nn.Module.__init__(layer)
    layer.kv_cache_dtype = "fp8_ds_mla"
    monkeypatch.setattr(
        ampere_sparse.DeepseekV4ROCMAiterMLAAttention,
        "__init__",
        lambda self, *args, **kwargs: None,
    )

    ampere_sparse.DeepseekV4AmpereMLAAttention.__init__(layer)

    assert layer._flash_mla_decode is None
    load_decode.assert_not_called()


def test_flash_mla_decode_uses_existing_triton_path_when_disabled(monkeypatch) -> None:
    layer = object.__new__(ampere_sparse.DeepseekV4AmpereMLAAttention)
    torch.nn.Module.__init__(layer)
    layer._flash_mla_decode = None
    fallback = Mock()
    monkeypatch.setattr(
        ampere_sparse.DeepseekV4ROCMAiterMLAAttention,
        "_forward_decode",
        fallback,
    )
    arguments = {
        "q": Mock(),
        "kv_cache": Mock(),
        "swa_metadata": Mock(),
        "attn_metadata": Mock(),
        "swa_only": False,
        "output": Mock(),
    }

    layer._forward_decode(**arguments)

    fallback.assert_called_once_with(**arguments)


def test_flash_mla_decode_dispatches_c128a_batch() -> None:
    flash_decode = Mock(return_value=torch.full((2, 64, 512), 3.0))
    layer = object.__new__(ampere_sparse.DeepseekV4AmpereMLAAttention)
    torch.nn.Module.__init__(layer)
    layer._flash_mla_decode = flash_decode
    layer.kv_cache_dtype = "fp8_ds_mla"
    layer.compress_ratio = 128
    layer.scale = 0.125
    layer.attn_sink = None
    layer.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros((4, 64, 584), dtype=torch.uint8)
    )

    q = torch.zeros((2, 64, 512), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    compressed_cache = torch.zeros((3, 64, 584), dtype=torch.uint8)
    extra_indices = torch.tensor([[[1, 2]], [[3, -1]]], dtype=torch.int32)
    extra_lens = torch.tensor([2, 1], dtype=torch.int32)
    attn_metadata = SimpleNamespace(
        c128a_global_decode_topk_indices=extra_indices,
        c128a_decode_topk_lens=extra_lens,
    )
    swa_metadata = SimpleNamespace(
        num_decodes=2,
        num_decode_tokens=2,
        decode_swa_indices=torch.tensor([[4, 5], [6, -1]], dtype=torch.int32),
        decode_swa_lens=torch.tensor([2, 1], dtype=torch.int32),
    )

    layer._forward_decode(
        q=q,
        kv_cache=compressed_cache,
        swa_metadata=swa_metadata,
        attn_metadata=attn_metadata,
        swa_only=False,
        output=output,
    )

    torch.testing.assert_close(output, torch.full_like(output, 3.0))
    call = flash_decode.call_args.kwargs
    assert call["q"] is q
    assert call["swa_cache"] is layer.swa_cache_layer.kv_cache
    assert call["extra_cache"] is compressed_cache
    torch.testing.assert_close(call["extra_indices"], extra_indices.reshape(2, -1))
    torch.testing.assert_close(call["extra_lens"], extra_lens)


def test_flash_mla_decode_dispatches_c4a_global_indices(monkeypatch) -> None:
    flash_decode = Mock(return_value=torch.full((1, 64, 512), 2.0))
    layer = object.__new__(ampere_sparse.DeepseekV4AmpereMLAAttention)
    torch.nn.Module.__init__(layer)
    layer._flash_mla_decode = flash_decode
    layer.kv_cache_dtype = "fp8_ds_mla"
    layer.compress_ratio = 4
    layer.scale = 0.125
    layer.attn_sink = None
    layer.swa_cache_layer = SimpleNamespace(
        kv_cache=torch.zeros((4, 64, 584), dtype=torch.uint8)
    )
    layer.topk_indices_buffer = torch.tensor([[0, 1]], dtype=torch.int32)
    output_buffers = (torch.empty_like(layer.topk_indices_buffer), torch.empty(1))
    monkeypatch.setattr(
        layer,
        "_global_topk_output_buffers",
        lambda source: output_buffers,
    )
    global_indices = torch.tensor([[8, 9]], dtype=torch.int32)
    global_lens = torch.tensor([2], dtype=torch.int32)
    compute_indices = Mock(return_value=(global_indices, global_lens))
    monkeypatch.setattr(
        ampere_sparse,
        "compute_global_topk_indices_and_lens",
        compute_indices,
    )

    q = torch.zeros((1, 64, 512), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    compressed_cache = torch.zeros((3, 64, 584), dtype=torch.uint8)
    attn_metadata = SimpleNamespace(
        block_size=256,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32),
    )
    swa_metadata = SimpleNamespace(
        num_decodes=1,
        num_decode_tokens=1,
        is_valid_token=torch.tensor([True]),
        token_to_req_indices=torch.tensor([0], dtype=torch.int32),
        decode_swa_indices=torch.tensor([[4, 5]], dtype=torch.int32),
        decode_swa_lens=torch.tensor([2], dtype=torch.int32),
    )

    layer._forward_decode(
        q=q,
        kv_cache=compressed_cache,
        swa_metadata=swa_metadata,
        attn_metadata=attn_metadata,
        swa_only=False,
        output=output,
    )

    compute_indices.assert_called_once()
    assert compute_indices.call_args.kwargs["output_buffers"] is output_buffers
    torch.testing.assert_close(
        flash_decode.call_args.kwargs["extra_indices"], global_indices
    )
    torch.testing.assert_close(flash_decode.call_args.kwargs["extra_lens"], global_lens)
    torch.testing.assert_close(output, torch.full_like(output, 2.0))


def test_flash_mla_prefill_dispatches_c4a_global_indices(monkeypatch) -> None:
    flash_prefill = Mock(return_value=torch.full((2, 64, 512), 4.0))
    layer = object.__new__(ampere_sparse.DeepseekV4AmpereMLAAttention)
    torch.nn.Module.__init__(layer)
    layer._flash_mla_prefill = flash_prefill
    layer.kv_cache_dtype = "fp4_ds_mla"
    layer.compress_ratio = 4
    layer.scale = 0.125
    layer.attn_sink = None
    layer.eager_scratch_pool = None
    layer.topk_indices_buffer = torch.tensor(
        [[99, 99], [1, 2], [3, -1]], dtype=torch.int32
    )

    q = torch.zeros((2, 64, 512), dtype=torch.bfloat16)
    output = torch.empty_like(q)
    swa_cache = torch.zeros((4, 64, 368), dtype=torch.uint8)
    compressed_cache = torch.zeros((3, 64, 368), dtype=torch.uint8)
    swa_indices = torch.tensor([[[4, 5]], [[6, -1]]], dtype=torch.int32)
    swa_lens = torch.tensor([2, 1], dtype=torch.int32)
    swa_metadata = SimpleNamespace(
        num_prefill_tokens=2,
        num_decode_tokens=1,
        prefill_swa_indices=swa_indices,
        prefill_swa_lens=swa_lens,
        token_to_req_indices=torch.tensor([0, 1, 1], dtype=torch.int32),
        is_valid_token=torch.tensor([True, True, True]),
    )
    attn_metadata = SimpleNamespace(
        block_size=256,
        block_table=torch.tensor([[0, 1], [2, 3]], dtype=torch.int32),
    )
    global_indices = torch.tensor([[8, 9], [10, -1]], dtype=torch.int32)
    global_lens = torch.tensor([2, 1], dtype=torch.int32)
    map_topk = Mock(return_value=(global_indices, global_lens))
    monkeypatch.setattr(
        ampere_sparse,
        "compute_global_topk_indices_and_lens",
        map_topk,
    )

    layer._forward_prefill(
        q=q,
        positions=torch.tensor([10, 11]),
        compressed_k_cache=compressed_cache,
        swa_k_cache=swa_cache,
        output=output,
        attn_metadata=attn_metadata,
        swa_metadata=swa_metadata,
    )

    torch.testing.assert_close(output, torch.full_like(output, 4.0))
    map_call = map_topk.call_args.args
    torch.testing.assert_close(map_call[0], layer.topk_indices_buffer[1:3])
    torch.testing.assert_close(map_call[1], torch.tensor([1, 1], dtype=torch.int32))
    assert map_call[2] is attn_metadata.block_table
    assert map_call[3] == 64
    call = flash_prefill.call_args.kwargs
    assert call["q"] is q
    assert call["swa_cache"] is swa_cache
    assert call["extra_cache"] is compressed_cache
    torch.testing.assert_close(call["swa_indices"], swa_indices)
    torch.testing.assert_close(call["extra_indices"], global_indices)
    torch.testing.assert_close(call["extra_lens"], global_lens)


def test_flash_mla_decode_rejects_unsupported_cache() -> None:
    layer = object.__new__(ampere_sparse.DeepseekV4AmpereMLAAttention)
    torch.nn.Module.__init__(layer)
    layer._flash_mla_decode = Mock()
    layer.kv_cache_dtype = "bfloat16"

    with pytest.raises(ValueError, match="requires fp8_ds_mla or fp4_ds_mla"):
        layer._forward_decode(
            q=Mock(),
            kv_cache=Mock(),
            swa_metadata=Mock(),
            attn_metadata=Mock(),
            swa_only=False,
            output=Mock(),
        )
