# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import struct
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.gguf_dsv4 import GGUFDSV4ModelLoader


def _source_profile_sha256(source_types: dict[str, str]) -> str:
    payload = json.dumps(source_types, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def _profile_model_config(source_profile_sha256: str) -> SimpleNamespace:
    return SimpleNamespace(
        model_arch_config=SimpleNamespace(
            quantization_config={"source_quant_types_sha256": source_profile_sha256}
        )
    )


def _string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _write_loader_fixture(path: Path) -> tuple[bytes, torch.Tensor]:
    tensors = [
        ("blk.0.attn_q_a.weight", (32, 1), 8, 0),
        ("blk.0.ffn_norm.weight", (4,), 0, 64),
    ]
    header = bytearray(b"GGUF") + struct.pack("<IQQ", 3, len(tensors), 1)
    header += _string("general.alignment") + struct.pack("<II", 4, 32)
    for name, dims, type_id, offset in tensors:
        header += _string(name) + struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", dim) for dim in dims)
        header += struct.pack("<IQ", type_id, offset)
    data_start = (len(header) + 31) & ~31
    quantized = bytes(range(34))
    norm = torch.tensor([1.25, -2.5, 3.75, -4.0], dtype=torch.float32)
    payload = bytearray(80)
    payload[:34] = quantized
    payload[64:80] = norm.numpy().tobytes()
    path.write_bytes(bytes(header) + bytes(data_start - len(header)) + payload)
    return quantized, norm


def _write_split_loader_fixture(
    path: Path,
    *,
    split_number: int,
    tensor: tuple[str, tuple[int, ...], int, bytes],
) -> None:
    name, dims, type_id, payload = tensor
    metadata = [
        ("general.alignment", 4, struct.pack("<I", 32)),
        ("split.count", 4, struct.pack("<I", 2)),
        ("split.no", 4, struct.pack("<I", split_number)),
        ("split.tensors.count", 10, struct.pack("<Q", 2)),
    ]
    header = bytearray(b"GGUF") + struct.pack("<IQQ", 3, 1, len(metadata))
    for key, value_type, value in metadata:
        header += _string(key) + struct.pack("<I", value_type) + value
    header += _string(name) + struct.pack("<I", len(dims))
    header += b"".join(struct.pack("<Q", dim) for dim in dims)
    header += struct.pack("<IQ", type_id, 0)
    data_start = (len(header) + 31) & ~31
    path.write_bytes(bytes(header) + bytes(data_start - len(header)) + payload)


class _LoaderFixtureModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = torch.nn.Module()
        self.model.layers = torch.nn.ModuleList([torch.nn.Module()])
        layer = self.model.layers[0]
        layer.attn = torch.nn.Module()
        layer.attn.fused_wqa_wkv = torch.nn.Module()
        layer.attn.fused_wqa_wkv.weight_raw = torch.nn.Parameter(
            torch.full((34,), 255, dtype=torch.uint8), requires_grad=False
        )
        layer.attn.fused_wqa_wkv.weight_raw_0 = torch.nn.Parameter(
            torch.full((34,), 255, dtype=torch.uint8), requires_grad=False
        )
        layer.ffn_norm = torch.nn.Module()
        layer.ffn_norm.weight = torch.nn.Parameter(
            torch.empty(4, dtype=torch.bfloat16), requires_grad=False
        )


def test_gguf_dsv4_loader_verifies_and_loads_exact_targets(
    tmp_path: Path, monkeypatch
) -> None:
    path = tmp_path / "model.gguf"
    quantized, norm = _write_loader_fixture(path)
    config = LoadConfig(
        load_format="gguf_dsv4",
        model_loader_extra_config={
            "gguf_path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "file_size": path.stat().st_size,
            "tensor_count": 2,
            "max_source_chunk_bytes": 9,
        },
    )
    loader = get_model_loader(config)
    assert isinstance(loader, GGUFDSV4ModelLoader)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.gguf_dsv4.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.gguf_dsv4."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    model = _LoaderFixtureModel()
    import vllm.model_executor.model_loader.gguf_dsv4 as loader_module

    verify_calls = 0
    real_verify = loader_module.verify_gguf_sha256

    def counting_verify(*args, **kwargs):
        nonlocal verify_calls
        verify_calls += 1
        return real_verify(*args, **kwargs)

    monkeypatch.setattr(loader_module, "verify_gguf_sha256", counting_verify)
    loader.download_model(model_config=None)
    loader.load_weights(model, model_config=None)

    assert verify_calls == 1
    assert bytes(model.model.layers[0].attn.fused_wqa_wkv.weight_raw.tolist()) == (
        quantized
    )
    torch.testing.assert_close(
        model.model.layers[0].ffn_norm.weight.float(),
        norm.bfloat16().float(),
        rtol=0,
        atol=0,
    )


def test_gguf_dsv4_loader_verifies_and_loads_ordered_split_artifact(
    tmp_path: Path, monkeypatch
) -> None:
    quantized = bytes(range(34))
    norm = torch.tensor([1.25, -2.5, 3.75, -4.0], dtype=torch.float32)
    shards = [
        tmp_path / "model-00001-of-00002.gguf",
        tmp_path / "model-00002-of-00002.gguf",
    ]
    _write_split_loader_fixture(
        shards[0],
        split_number=0,
        tensor=("blk.0.attn_q_a.weight", (32, 1), 8, quantized),
    )
    _write_split_loader_fixture(
        shards[1],
        split_number=1,
        tensor=("blk.0.ffn_norm.weight", (4,), 0, norm.numpy().tobytes()),
    )
    source_profile_sha256 = _source_profile_sha256({"blk.0.attn_q_a.weight": "Q8_0"})
    config = LoadConfig(
        load_format="gguf_dsv4",
        model_loader_extra_config={
            "gguf_shards": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "file_size": path.stat().st_size,
                    "tensor_count": 1,
                }
                for path in shards
            ],
            "tensor_count": 2,
            "source_quant_types_sha256": source_profile_sha256,
            "max_source_chunk_bytes": 9,
        },
    )
    loader = get_model_loader(config)
    assert isinstance(loader, GGUFDSV4ModelLoader)
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.gguf_dsv4.get_tensor_model_parallel_rank",
        lambda: 0,
    )
    monkeypatch.setattr(
        "vllm.model_executor.model_loader.gguf_dsv4."
        "get_tensor_model_parallel_world_size",
        lambda: 1,
    )
    model = _LoaderFixtureModel()
    model_config = _profile_model_config(source_profile_sha256)

    loader.download_model(model_config=model_config)
    loader.load_weights(model, model_config=model_config)

    assert bytes(model.model.layers[0].attn.fused_wqa_wkv.weight_raw_0.tolist()) == (
        quantized
    )
    torch.testing.assert_close(
        model.model.layers[0].ffn_norm.weight.float(),
        norm.bfloat16().float(),
        rtol=0,
        atol=0,
    )


def test_gguf_dsv4_loader_rejects_out_of_order_split_artifact(
    tmp_path: Path,
) -> None:
    payload = bytes(range(34))
    shards = [
        tmp_path / "model-00001-of-00002.gguf",
        tmp_path / "model-00002-of-00002.gguf",
    ]
    source_names = ("blk.0.attn_q_a.weight", "blk.0.attn_kv.weight")
    for path, split_number, source_name in zip(
        shards, (1, 0), source_names, strict=True
    ):
        _write_split_loader_fixture(
            path,
            split_number=split_number,
            tensor=(source_name, (32, 1), 8, payload),
        )
    source_profile_sha256 = _source_profile_sha256(
        {
            "blk.0.attn_q_a.weight": "Q8_0",
            "blk.0.attn_kv.weight": "Q8_0",
        }
    )
    config = LoadConfig(
        load_format="gguf_dsv4",
        model_loader_extra_config={
            "gguf_shards": [
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "file_size": path.stat().st_size,
                    "tensor_count": 1,
                }
                for path in shards
            ],
            "tensor_count": 2,
            "source_quant_types_sha256": source_profile_sha256,
        },
    )
    loader = get_model_loader(config)

    with pytest.raises(ValueError, match="split.no"):
        loader.download_model(model_config=_profile_model_config(source_profile_sha256))
