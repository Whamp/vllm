# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import struct
from pathlib import Path

import torch

from vllm.config.load import LoadConfig
from vllm.model_executor.model_loader import get_model_loader
from vllm.model_executor.model_loader.gguf_dsv4 import GGUFDSV4ModelLoader


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
