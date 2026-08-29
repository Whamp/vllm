# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

import vllm.envs as envs
from vllm.model_executor.model_loader.model_memory_diagnostics import (
    capture_device_memory_report,
    capture_model_memory_report,
    capture_model_memory_report_if_enabled,
    collect_registered_tensor_storage_inventory,
    reset_model_memory_peak_if_enabled,
    start_allocator_memory_history_if_enabled,
    stop_allocator_memory_history_if_enabled,
    summarize_allocator_memory_snapshot,
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
        named_allocations={"workspace_manager_bytes": 16},
        allocator_snapshot={
            "segments": [
                {
                    "address": 100,
                    "total_size": 48,
                    "allocated_size": 40,
                    "active_size": 40,
                    "segment_pool_id": (0, 0),
                    "blocks": [
                        {
                            "address": 100,
                            "size": 40,
                            "requested_size": 28,
                            "state": "active_allocated",
                            "frames": [
                                {
                                    "filename": "/workspace/vllm/models/example.py",
                                    "line": 7,
                                    "name": "allocate_example",
                                }
                            ],
                        },
                        {
                            "address": 140,
                            "size": 8,
                            "requested_size": 0,
                            "state": "inactive",
                            "frames": [],
                        },
                    ],
                }
            ],
            "device_traces": [],
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
    assert report["named_allocations"] == {"workspace_manager_bytes": 16}
    assert report["allocator_snapshot"]["block_bytes_by_state"] == {
        "active_allocated": 40,
        "inactive": 8,
    }
    assert report["allocator_snapshot"]["active_allocations_by_frame"] == [
        {
            "bytes": 40,
            "count": 1,
            "frame": "vllm/models/example.py:7:allocate_example",
            "requested_bytes": 28,
        }
    ]
    assert report_path.stat().st_mode & 0o777 == 0o644


@st.composite
def _allocator_blocks(draw):
    count = draw(st.integers(min_value=0, max_value=40))
    states = draw(
        st.lists(
            st.sampled_from(["active_allocated", "active_awaiting_free", "inactive"]),
            min_size=count,
            max_size=count,
        )
    )
    sizes = draw(
        st.lists(
            st.integers(min_value=1, max_value=1 << 20),
            min_size=count,
            max_size=count,
        )
    )
    owners = draw(
        st.lists(
            st.sampled_from(["qsa", "ple", "workspace", "unattributed"]),
            min_size=count,
            max_size=count,
        )
    )
    blocks = []
    address = 4096
    for state, size, owner in zip(states, sizes, owners, strict=True):
        frames = (
            []
            if owner == "unattributed"
            else [
                {
                    "filename": f"/workspace/vllm/{owner}.py",
                    "line": 10,
                    "name": f"allocate_{owner}",
                }
            ]
        )
        blocks.append(
            {
                "address": address,
                "size": size,
                "requested_size": size,
                "state": state,
                "frames": frames,
            }
        )
        address += size
    return blocks


@given(_allocator_blocks())
def test_allocator_snapshot_summary_preserves_state_and_owner_bytes(blocks) -> None:
    total_size = sum(block["size"] for block in blocks)
    snapshot = {
        "segments": [
            {
                "address": 4096,
                "total_size": total_size,
                "allocated_size": sum(
                    block["size"]
                    for block in blocks
                    if block["state"] == "active_allocated"
                ),
                "active_size": sum(
                    block["size"] for block in blocks if block["state"] != "inactive"
                ),
                "segment_pool_id": (0, 0),
                "blocks": blocks,
            }
        ],
        "device_traces": [],
    }

    summary = summarize_allocator_memory_snapshot(snapshot)

    expected_state_bytes: dict[str, int] = {}
    for block in blocks:
        state = block["state"]
        expected_state_bytes[state] = expected_state_bytes.get(state, 0) + block["size"]
    assert summary["block_bytes_by_state"] == expected_state_bytes
    assert sum(
        owner["bytes"] for owner in summary["active_allocations_by_frame"]
    ) == expected_state_bytes.get("active_allocated", 0)
    assert summary["segment_total_bytes"] == total_size


def test_allocator_memory_history_start_and_stop(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setenv("VLLM_MODEL_MEMORY_REPORT_DIR", str(tmp_path))
    monkeypatch.setattr(
        torch.cuda.memory,
        "_record_memory_history",
        lambda **kwargs: calls.append(kwargs),
    )

    start_allocator_memory_history_if_enabled(torch.device("cuda", 7))
    start_allocator_memory_history_if_enabled(torch.device("cuda", 7))
    stop_allocator_memory_history_if_enabled(torch.device("cuda", 7))

    assert calls == [
        {
            "enabled": "state",
            "context": "all",
            "stacks": "python",
            "max_entries": 100_000,
            "device": torch.device("cuda", 7),
            "clear_history": True,
        },
        {"enabled": None, "device": torch.device("cuda", 7)},
    ]


def test_allocator_memory_history_is_opt_in(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[object] = []
    monkeypatch.delenv("VLLM_MODEL_MEMORY_REPORT_DIR", raising=False)
    monkeypatch.setattr(torch.cuda.memory, "_record_memory_history", calls.append)

    start_allocator_memory_history_if_enabled(torch.device("cuda", 0))
    stop_allocator_memory_history_if_enabled(torch.device("cuda", 0))

    assert not calls
