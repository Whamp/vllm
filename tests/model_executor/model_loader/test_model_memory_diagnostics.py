# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch

import vllm.envs as envs
from vllm.model_executor.model_loader.model_memory_diagnostics import (
    capture_device_memory_report,
    capture_model_memory_report,
    capture_model_memory_report_if_enabled,
    collect_registered_tensor_storage_inventory,
    reset_model_memory_peak_if_enabled,
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


class ModelWithUninitializedParameter(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lazy_weight = torch.nn.parameter.UninitializedParameter(
            requires_grad=False
        )


class AliasedModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(
            torch.arange(6, dtype=torch.float32).reshape(2, 3)
        )
        self.weight_alias = self.weight
        self.register_buffer(
            "weight_view",
            self.weight.detach().view(-1),
            persistent=False,
        )
        self.register_buffer(
            "independent_buffer",
            torch.ones(2, dtype=torch.float16),
            persistent=False,
        )


def test_registered_tensor_inventory_deduplicates_aliased_storage() -> None:
    inventory = collect_registered_tensor_storage_inventory(AliasedModel())

    assert inventory["unique_storage_bytes"] == 28
    assert inventory["unique_storage_count"] == 2
    assert inventory["parameter_names"] == 2
    assert inventory["buffer_names"] == 2

    shared_storage = next(
        storage for storage in inventory["storages"] if storage["nbytes"] == 24
    )
    assert [tensor["name"] for tensor in shared_storage["tensors"]] == [
        "weight",
        "weight_alias",
        "weight_view",
    ]
    assert [tensor["kind"] for tensor in shared_storage["tensors"]] == [
        "parameter",
        "parameter",
        "buffer",
    ]


def test_registered_tensor_inventory_reports_uninitialized_parameter() -> None:
    inventory = collect_registered_tensor_storage_inventory(
        ModelWithUninitializedParameter()
    )

    assert inventory["unique_storage_bytes"] == 0
    assert inventory["unbacked_tensors"] == [
        {
            "name": "lazy_weight",
            "kind": "parameter",
            "materialized": False,
        }
    ]


def test_model_memory_report_dir_is_not_compile_factor(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MODEL_MEMORY_REPORT_DIR", "/tmp/vllm-memory-reports")

    assert envs.VLLM_MODEL_MEMORY_REPORT_DIR == "/tmp/vllm-memory-reports"
    assert "VLLM_MODEL_MEMORY_REPORT_DIR" not in envs.compile_factors()


def test_capture_model_memory_report_is_disabled_by_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("VLLM_MODEL_MEMORY_REPORT_DIR", raising=False)

    assert (
        capture_model_memory_report_if_enabled(
            AliasedModel(),
            stage="disabled",
        )
        is None
    )


def test_reset_model_memory_peak_skips_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_MODEL_MEMORY_REPORT_DIR", "/tmp/vllm-memory-reports")

    reset_model_memory_peak_if_enabled(torch.device("cpu"))


def test_capture_device_memory_report_writes_counters_without_model(tmp_path) -> None:
    report_path = capture_device_memory_report(
        stage="distributed initialized",
        report_dir=tmp_path,
        device=torch.device("cpu"),
        memory_counters={
            "torch_allocated_bytes": 4,
            "torch_reserved_bytes": 8,
            "device_used_bytes": 12,
        },
    )

    report = json.loads(report_path.read_text())
    assert report["stage"] == "distributed initialized"
    assert report["report_kind"] == "device"
    assert report["memory_counters"]["allocator_cache_bytes"] == 4
    assert report["memory_counters"]["non_torch_device_bytes"] == 4
    assert "registered_tensors" not in report


def test_capture_model_memory_report_writes_rank_local_json(tmp_path) -> None:
    report_path = capture_model_memory_report(
        AliasedModel(),
        stage="after weights",
        report_dir=tmp_path,
        memory_counters={
            "torch_allocated_bytes": 40,
            "torch_reserved_bytes": 48,
            "device_used_bytes": 60,
        },
    )

    assert report_path is not None
    assert report_path.name.startswith("after-weights-device-cpu-pid-")
    report = json.loads(report_path.read_text())
    assert report["stage"] == "after weights"
    assert report["registered_tensors"]["unique_storage_bytes"] == 28
    assert report["memory_counters"]["unregistered_torch_allocated_bytes"] == 12
    assert report["memory_counters"]["allocator_cache_bytes"] == 8
    assert report["memory_counters"]["non_torch_device_bytes"] == 12
