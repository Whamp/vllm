# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from torch import nn

import vllm.envs as envs
import vllm.models.qwen4_exp.nvidia.ple_layer as ple_layer_module
from vllm.config import VllmConfig
from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpBf16PleMmapConfig,
    Qwen4ExpNGramEmbedding,
    Qwen4ExpPLELayer,
)
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.fixture(autouse=True)
def clear_vllm_env_cache() -> Iterator[None]:
    cache_clear = getattr(envs.__getattr__, "cache_clear", None)
    if cache_clear is not None:
        cache_clear()
    yield
    if cache_clear is not None:
        cache_clear()


def _ngram_config() -> Qwen4ExpTextConfig:
    return cast(
        Qwen4ExpTextConfig,
        SimpleNamespace(
            ngram_size=3,
            heads_per_ngram=1,
            eos_token_id=2,
            vocab_size=32,
            split_ngram_parts=2,
            ngram_vocab_size_base=4,
            make_ngram_vocab_size_divisible_by=1,
            seed=1234,
        ),
    )


def _bf16_mmap_config() -> Qwen4ExpBf16PleMmapConfig:
    return Qwen4ExpBf16PleMmapConfig(
        checkpoint_path="/ple/table",
        expected_sha256="a" * 64,
        native_library_path="/opt/vllm/libvllm_ple_gather.so",
        tensor_prefix=(
            "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
        ),
    )


def _ple_layer_inputs() -> tuple[Qwen4ExpTextConfig, VllmConfig]:
    config = cast(
        Qwen4ExpTextConfig,
        SimpleNamespace(
            hidden_size=16,
            ple_embed_dim=32,
            ngram_size=3,
            heads_per_ngram=1,
            hc_count=2,
            ple_conv_kernel_size=2,
            rms_norm_eps=1e-6,
        ),
    )
    vllm_config = cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(dtype=torch.bfloat16),
            cache_config=SimpleNamespace(),
            quant_config=None,
            num_speculative_tokens=0,
            scheduler_config=SimpleNamespace(
                max_num_batched_tokens=8,
                max_num_seqs=2,
            ),
        ),
    )
    return config, vllm_config


class _FakeBf16PleMmapGather:
    instances: list["_FakeBf16PleMmapGather"] = []

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        expected_sha256: str,
        native_library_path: str | Path,
        tensor_prefix: str,
        total_rows: int,
        width: int,
    ) -> None:
        self.checkpoint_path = Path(checkpoint_path)
        self.expected_sha256 = expected_sha256
        self.native_library_path = Path(native_library_path)
        self.tensor_prefix = tensor_prefix
        self.total_rows = total_rows
        self.width = width
        self.gathered_row_ids: torch.Tensor | None = None
        self.instances.append(self)

    def gather_into(self, row_ids: torch.Tensor, output: torch.Tensor) -> None:
        self.gathered_row_ids = row_ids.clone()
        output.copy_(row_ids.reshape(-1, 1).expand(-1, self.width))


def _patch_embedding_dependencies(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(ple_layer_module, "is_offload_process", lambda: True)
    monkeypatch.setattr(ple_layer_module, "_nth_prime_after", lambda *_: 4)


def _set_bf16_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(envs, "VLLM_PLE_CPU_OFFLOAD", True, raising=False)
    monkeypatch.setattr(envs, "VLLM_PLE_BF16_MMAP_FILE", "/ple/table", raising=False)
    monkeypatch.setattr(envs, "VLLM_PLE_BF16_MMAP_SHA256", "a" * 64, raising=False)
    monkeypatch.setattr(
        envs,
        "VLLM_PLE_BF16_MMAP_LIBRARY",
        "/opt/vllm/libvllm_ple_gather.so",
        raising=False,
    )
    monkeypatch.setattr(envs, "VLLM_PLE_NVFP4_SIDECAR_DIR", None, raising=False)
    monkeypatch.setattr(
        envs,
        "VLLM_PLE_NVFP4_SIDECAR_META_SHA256",
        None,
        raising=False,
    )


def test_bf16_sidecar_avoids_resident_embedding_and_gathers_rows(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _FakeBf16PleMmapGather.instances.clear()
    _patch_embedding_dependencies(monkeypatch)
    monkeypatch.setattr(
        ple_layer_module,
        "Bf16PleMmapGather",
        _FakeBf16PleMmapGather,
    )
    monkeypatch.setattr(
        ple_layer_module,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: pytest.fail(
            "BF16 sidecar construction allocated the resident PLE embedding"
        ),
    )

    embedding = Qwen4ExpNGramEmbedding(
        _ngram_config(),
        embedding_dim=32,
        ple_dense_layer_id=0,
        max_total_tokens=8,
        max_num_reqs=2,
        prefix="language_model.model.layers.1.ple.ple_embedding",
        params_dtype=torch.bfloat16,
        bf16_mmap_config=_bf16_mmap_config(),
    )

    assert not list(embedding.named_parameters())
    assert embedding.get_offload_output_dtype(torch.float16) == torch.bfloat16
    assert embedding.get_offload_output_dim(32) == 32
    sidecar = _FakeBf16PleMmapGather.instances[0]
    assert sidecar.tensor_prefix == _bf16_mmap_config().tensor_prefix
    assert sidecar.total_rows == 8
    assert sidecar.width == 16
    assert (
        embedding.load_weights(
            [("ngram_embedding.shard_0.weight", torch.empty((4, 16)))]
        )
        == set()
    )

    output_buffer = torch.empty((2, 32), dtype=torch.bfloat16)
    output = embedding.forward_impl(
        torch.empty((2, 0)),
        torch.tensor([3, 5]),
        torch.tensor([0, 2]),
        torch.tensor([[2, 2]]),
        output_buffer=output_buffer,
    )

    assert output.data_ptr() == output_buffer.data_ptr()
    assert sidecar.gathered_row_ids is not None
    expected = sidecar.gathered_row_ids.reshape(2, 2, 1).expand(-1, -1, 16)
    assert torch.equal(output.reshape(2, 2, 16), expected)


def test_bf16_mmap_environment_is_part_of_the_vllm_config(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PLE_BF16_MMAP_FILE", "/ple/table")
    monkeypatch.setenv("VLLM_PLE_BF16_MMAP_SHA256", "a" * 64)
    monkeypatch.setenv(
        "VLLM_PLE_BF16_MMAP_LIBRARY",
        "/opt/vllm/libvllm_ple_gather.so",
    )

    assert envs.VLLM_PLE_BF16_MMAP_FILE == "/ple/table"
    assert envs.VLLM_PLE_BF16_MMAP_SHA256 == "a" * 64
    assert envs.VLLM_PLE_BF16_MMAP_LIBRARY == "/opt/vllm/libvllm_ple_gather.so"


def test_ple_layer_passes_bf16_mmap_environment_to_embedding(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    def make_embedding(*args: object, **kwargs: object) -> nn.Module:
        captured.update(kwargs)
        return nn.Identity()

    _set_bf16_environment(monkeypatch)
    monkeypatch.setattr(ple_layer_module, "is_offload_process", lambda: True)
    monkeypatch.setattr(ple_layer_module, "Qwen4ExpNGramEmbedding", make_embedding)
    monkeypatch.setattr(
        ple_layer_module, "ReplicatedLinear", lambda *args, **kwargs: nn.Identity()
    )
    monkeypatch.setattr(
        ple_layer_module,
        "Qwen4ExpPLEGroupedNorm",
        lambda *args, **kwargs: nn.Identity(),
    )
    monkeypatch.setattr(
        ple_layer_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
    )
    config, vllm_config = _ple_layer_inputs()

    Qwen4ExpPLELayer(
        config,
        vllm_config,
        layer_idx=1,
        ple_dense_layer_id=0,
        prefix="language_model.model.layers.1.ple",
    )

    mmap_config = captured["bf16_mmap_config"]
    assert isinstance(mmap_config, Qwen4ExpBf16PleMmapConfig)
    assert mmap_config == _bf16_mmap_config()


def test_bf16_sidecar_requires_complete_environment(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_bf16_environment(monkeypatch)
    monkeypatch.setattr(envs, "VLLM_PLE_BF16_MMAP_SHA256", None)
    config, vllm_config = _ple_layer_inputs()

    with pytest.raises(ValueError, match="file, SHA-256, and native library"):
        Qwen4ExpPLELayer(config, vllm_config)


def test_bf16_sidecar_rejects_non_bf16_model_dtype(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_embedding_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="requires a BF16 model dtype"):
        Qwen4ExpNGramEmbedding(
            _ngram_config(),
            embedding_dim=32,
            ple_dense_layer_id=0,
            max_total_tokens=8,
            max_num_reqs=2,
            prefix="language_model.model.layers.1.ple.ple_embedding",
            params_dtype=torch.float16,
            bf16_mmap_config=_bf16_mmap_config(),
        )


def test_bf16_and_nvfp4_sidecars_are_mutually_exclusive(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _patch_embedding_dependencies(monkeypatch)

    with pytest.raises(ValueError, match="mutually exclusive"):
        Qwen4ExpNGramEmbedding(
            _ngram_config(),
            embedding_dim=32,
            ple_dense_layer_id=0,
            max_total_tokens=8,
            max_num_reqs=2,
            prefix="language_model.model.layers.1.ple.ple_embedding",
            params_dtype=torch.bfloat16,
            nvfp4_sidecar_dir="/ple/nvfp4",
            nvfp4_sidecar_manifest_sha256="b" * 64,
            bf16_mmap_config=_bf16_mmap_config(),
        )
