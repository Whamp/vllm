# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest
import torch
from safetensors.torch import save_file


@pytest.fixture(autouse=True)
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.fixture(scope="module")
def production_worker_overlay(
    tmp_path_factory: pytest.TempPathFactory,
) -> ModuleType:
    base_worker = os.environ.get("VLLM_QWEN38_PRODUCTION_PLE_WORKER")
    if base_worker is None:
        pytest.skip("production Qwen3.8 PLE worker source was not provided")
    assert base_worker is not None
    root = Path(__file__).parents[3]
    output_dir = tmp_path_factory.mktemp("qwen38_production_overlay")
    script = root / "benchmarks/qwen38_ple_runtime/build_native_gather_overlay.py"
    subprocess.run(
        [sys.executable, script, base_worker, output_dir],
        check=True,
    )
    worker_path = output_dir / "worker_image_quant.py"
    spec = importlib.util.spec_from_file_location(
        "qwen38_production_worker_overlay",
        worker_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_overlay_routes_supported_gathers_to_native(
    production_worker_overlay: ModuleType,
) -> None:
    calls = []

    class NativeGather:
        def gather_into(self, row_ids: torch.Tensor, output: torch.Tensor) -> bool:
            calls.append(row_ids.clone())
            output.fill_(17)
            return True

    table = production_worker_overlay._PleQuantTable.__new__(
        production_worker_overlay._PleQuantTable
    )
    table._native_gather = NativeGather()
    output = torch.empty((3, 16), dtype=torch.bfloat16)
    row_ids = torch.tensor([7, 0, 7])

    table.gather_into(row_ids, output)

    assert len(calls) == 1
    assert torch.equal(calls[0], row_ids)
    assert torch.equal(output, torch.full_like(output, 17))


def test_production_overlay_ple_storage_modes_fail_closed(
    production_worker_overlay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for name in (
        "VLLM_PLE_QUANT_DIR",
        "VLLM_PLE_DISK_OFFLOAD_DIR",
        "VLLM_PLE_BF16_MMAP_FILE",
        "VLLM_PLE_BF16_MMAP_SHA256",
        "VLLM_PLE_BF16_MMAP_LIBRARY",
    ):
        monkeypatch.delenv(name, raising=False)
    assert production_worker_overlay._ple_storage_config() == (None, None, None)

    monkeypatch.setenv("VLLM_PLE_QUANT_DIR", "/quant")
    monkeypatch.setenv("VLLM_PLE_DISK_OFFLOAD_DIR", "/disk")
    assert production_worker_overlay._ple_storage_config() == (
        "/quant",
        None,
        None,
    )
    monkeypatch.delenv("VLLM_PLE_QUANT_DIR")
    monkeypatch.delenv("VLLM_PLE_DISK_OFFLOAD_DIR")
    monkeypatch.setenv("VLLM_PLE_BF16_MMAP_FILE", "/table")
    with pytest.raises(RuntimeError, match="requires VLLM_PLE_BF16_MMAP_SHA256"):
        production_worker_overlay._ple_storage_config()

    monkeypatch.setenv("VLLM_PLE_BF16_MMAP_SHA256", "a" * 64)
    monkeypatch.setenv("VLLM_PLE_BF16_MMAP_LIBRARY", "/library")
    monkeypatch.setenv("VLLM_PLE_QUANT_DIR", "/quant")
    with pytest.raises(RuntimeError, match="mutually exclusive"):
        production_worker_overlay._ple_storage_config()


def test_production_overlay_attaches_direct_bf16_table(
    production_worker_overlay: ModuleType,
    tmp_path: Path,
) -> None:
    prefix = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
    source_rows = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    temporary = tmp_path / "ple.safetensors"
    save_file(
        {
            f"{prefix}.shard_0.weight": source_rows[:3],
            f"{prefix}.shard_1.weight": source_rows[3:],
        },
        temporary,
    )
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    checkpoint_path = temporary.with_name(digest)
    temporary.rename(checkpoint_path)
    assert production_worker_overlay.__file__ is not None
    output_dir = Path(production_worker_overlay.__file__).parent
    assert (output_dir / "bf16_ple_mmap_gather.py").is_file()
    assert "bf16_ple_mmap_gather.py" in (output_dir / "SHA256SUMS").read_text()
    layer = torch.nn.Module()
    layer.ple_embedding = torch.nn.Module()
    layer.ple_embedding.ngram_embedding = torch.nn.Embedding(6, 4, dtype=torch.bfloat16)

    parameter_name = production_worker_overlay._ple_bf16_mmap_attach(
        "language_model.model.layers.1.ple",
        layer,
        str(checkpoint_path),
        digest,
        str(output_dir / "libvllm_ple_nvfp4_gather.so"),
    )

    assert parameter_name == "ple_embedding.ngram_embedding.weight"
    output = torch.empty((3, 4), dtype=torch.bfloat16)
    layer.ple_embedding.ngram_embedding._ple_quant.gather_into(  # type: ignore[attr-defined]
        torch.tensor([5, 0, 3]), output
    )
    assert torch.equal(
        output.view(torch.uint16),
        source_rows[torch.tensor([5, 0, 3])].view(torch.uint16),
    )
    layer.ple_embedding.ngram_embedding._ple_quant.close()  # type: ignore[attr-defined]


def test_production_overlay_keeps_python_fallback(
    production_worker_overlay: ModuleType,
) -> None:
    class UnsupportedNativeGather:
        def gather_into(self, row_ids: torch.Tensor, output: torch.Tensor) -> bool:
            del row_ids, output
            return False

    table = production_worker_overlay._PleQuantTable.__new__(
        production_worker_overlay._PleQuantTable
    )
    table._native_gather = UnsupportedNativeGather()
    table.layout = "group16_e2m1_e4m3scale_lownibblefirst"
    table.width = 16
    table._lut = None
    table._q = [torch.full((1, 8), 0x21, dtype=torch.uint8)]
    table._s = [torch.full((1, 1), 0.5, dtype=torch.float8_e4m3fn)]
    table._s2 = [2.0]
    table.ROWS_PER_SHARD = 1
    output = torch.empty((1, 16), dtype=torch.bfloat16)

    table.gather_into(torch.tensor([0]), output)

    expected = torch.tensor([0.5, 1.0] * 8, dtype=torch.bfloat16).reshape(1, 16)
    assert torch.equal(output, expected)
