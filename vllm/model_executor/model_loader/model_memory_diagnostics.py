# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in model memory diagnostics for checkpoint loading."""

import json
import os
import tempfile
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import regex as re
import torch
from torch import nn
from torch.nn.parameter import UninitializedBuffer, UninitializedParameter

import vllm.envs as envs

_REPORT_SCHEMA_VERSION = 2
_STAGE_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")
_ALLOCATOR_HISTORY_DEVICES: set[str] = set()


def collect_registered_tensor_storage_inventory(model: nn.Module) -> dict[str, Any]:
    """Inventory registered model tensors without double-counting shared storage."""
    storages: dict[tuple[str, int, int, int], dict[str, Any]] = {}
    unbacked_tensors: list[dict[str, Any]] = []
    parameter_names = 0
    buffer_names = 0

    registered_tensors = (
        (
            "parameter",
            model.named_parameters(recurse=True, remove_duplicate=False),
        ),
        (
            "buffer",
            model.named_buffers(recurse=True, remove_duplicate=False),
        ),
    )
    for kind, tensors in registered_tensors:
        for name, tensor in tensors:
            if kind == "parameter":
                parameter_names += 1
            else:
                buffer_names += 1
            if isinstance(tensor, (UninitializedParameter, UninitializedBuffer)):
                unbacked_tensors.append(
                    {
                        "name": name,
                        "kind": kind,
                        "materialized": False,
                    }
                )
                continue
            tensor_record = _describe_registered_tensor(name, kind, tensor)
            try:
                storage = tensor.untyped_storage()
            except RuntimeError:
                unbacked_tensors.append(tensor_record)
                continue

            storage_nbytes = storage.nbytes()
            storage_data_ptr = storage.data_ptr()
            storage_identity = int(getattr(storage, "_cdata", 0))
            storage_key = (
                str(tensor.device),
                storage_data_ptr,
                storage_nbytes,
                storage_identity if storage_data_ptr == 0 else 0,
            )
            storage_record = storages.setdefault(
                storage_key,
                {
                    "device": str(tensor.device),
                    "data_ptr": storage_data_ptr,
                    "nbytes": storage_nbytes,
                    "tensors": [],
                },
            )
            storage_record["tensors"].append(tensor_record)

    sorted_storages = sorted(
        storages.values(),
        key=lambda storage: (
            storage["device"],
            storage["tensors"][0]["name"],
        ),
    )
    for storage in sorted_storages:
        storage["tensors"].sort(key=lambda tensor: tensor["name"])

    bytes_by_device: dict[str, int] = {}
    for storage in sorted_storages:
        device = storage["device"]
        bytes_by_device[device] = bytes_by_device.get(device, 0) + storage["nbytes"]

    return {
        "parameter_names": parameter_names,
        "buffer_names": buffer_names,
        "unique_storage_count": len(sorted_storages),
        "unique_storage_bytes": sum(storage["nbytes"] for storage in sorted_storages),
        "unique_storage_bytes_by_device": bytes_by_device,
        "unbacked_tensors": sorted(
            unbacked_tensors,
            key=lambda tensor: (tensor["kind"], tensor["name"]),
        ),
        "storages": sorted_storages,
    }


def capture_device_memory_report(
    *,
    stage: str,
    report_dir: str | Path,
    device: torch.device | str,
    memory_counters: Mapping[str, int] | None = None,
    named_allocations: Mapping[str, int] | None = None,
    allocator_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    """Write one rank-local device memory report without traversing a model."""
    report_directory = Path(report_dir)
    report_directory.mkdir(parents=True, exist_ok=True)
    selected_device = torch.device(device)
    counters = dict(
        memory_counters
        if memory_counters is not None
        else _collect_device_memory_counters(selected_device)
    )
    _add_device_memory_residuals(counters)
    report_path = _make_report_path(report_directory, stage, selected_device)
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_kind": "device",
        "stage": stage,
        "timestamp_unix_seconds": time.time(),
        "pid": os.getpid(),
        "rank": _get_distributed_rank(),
        "device": str(selected_device),
        "memory_counters": counters,
    }
    _add_named_memory_details(
        report,
        selected_device,
        named_allocations=named_allocations,
        allocator_snapshot=allocator_snapshot,
    )
    _write_atomic_json(report_path, report)
    return report_path


def capture_device_memory_report_if_enabled(
    *,
    stage: str,
    device: torch.device | str,
    reset_peak_after_capture: bool = False,
    named_allocations: Mapping[str, int] | None = None,
) -> Path | None:
    """Write a device-only report when the diagnostics directory is configured."""
    report_dir = envs.VLLM_MODEL_MEMORY_REPORT_DIR
    if report_dir is None:
        return None
    report_path = capture_device_memory_report(
        stage=stage,
        report_dir=report_dir,
        device=device,
        named_allocations=named_allocations,
    )
    if reset_peak_after_capture:
        reset_model_memory_peak_if_enabled(device)
    return report_path


def capture_model_memory_report(
    model: nn.Module,
    *,
    stage: str,
    report_dir: str | Path,
    device: torch.device | str | None = None,
    memory_counters: Mapping[str, int] | None = None,
    include_storage_details: bool = True,
    named_allocations: Mapping[str, int] | None = None,
    allocator_snapshot: Mapping[str, Any] | None = None,
) -> Path:
    """Write one rank-local model memory report as an atomic JSON file."""
    report_directory = Path(report_dir)
    report_directory.mkdir(parents=True, exist_ok=True)
    inventory = collect_registered_tensor_storage_inventory(model)
    selected_device = _select_report_device(inventory, device)
    counters = dict(
        memory_counters
        if memory_counters is not None
        else _collect_device_memory_counters(selected_device)
    )
    _add_memory_accounting_residuals(counters, inventory, selected_device)
    if not include_storage_details:
        inventory.pop("storages")

    rank = _get_distributed_rank()
    report_path = _make_report_path(report_directory, stage, selected_device)
    report = {
        "schema_version": _REPORT_SCHEMA_VERSION,
        "report_kind": "model",
        "stage": stage,
        "timestamp_unix_seconds": time.time(),
        "pid": os.getpid(),
        "rank": rank,
        "device": str(selected_device),
        "model_class": f"{type(model).__module__}.{type(model).__qualname__}",
        "storage_details_included": include_storage_details,
        "registered_tensors": inventory,
        "memory_counters": counters,
    }
    _add_named_memory_details(
        report,
        selected_device,
        named_allocations=named_allocations,
        allocator_snapshot=allocator_snapshot,
    )
    _write_atomic_json(report_path, report)
    return report_path


def capture_model_memory_report_if_enabled(
    model: nn.Module,
    *,
    stage: str,
    device: torch.device | str | None = None,
    include_storage_details: bool = True,
    reset_peak_after_capture: bool = False,
    named_allocations: Mapping[str, int] | None = None,
) -> Path | None:
    """Write a model memory report only when its report directory is configured."""
    report_dir = envs.VLLM_MODEL_MEMORY_REPORT_DIR
    if report_dir is None:
        return None
    report_path = capture_model_memory_report(
        model,
        stage=stage,
        report_dir=report_dir,
        device=device,
        include_storage_details=include_storage_details,
        named_allocations=named_allocations,
    )
    if reset_peak_after_capture and device is not None:
        reset_model_memory_peak_if_enabled(device)
    return report_path


def reset_model_memory_peak_if_enabled(device: torch.device | str) -> None:
    """Reset allocator peaks before post-load conversion when reporting is enabled."""
    selected_device = torch.device(device)
    if envs.VLLM_MODEL_MEMORY_REPORT_DIR is None or selected_device.type == "cpu":
        return
    torch.accelerator.reset_peak_memory_stats(selected_device)


def start_allocator_memory_history_if_enabled(
    device: torch.device | str,
) -> None:
    """Record live CUDA allocator stacks only during opt-in memory diagnostics."""
    selected_device = torch.device(device)
    device_key = str(selected_device)
    if (
        envs.VLLM_MODEL_MEMORY_REPORT_DIR is None
        or selected_device.type != "cuda"
        or device_key in _ALLOCATOR_HISTORY_DEVICES
    ):
        return
    torch.cuda.memory._record_memory_history(
        enabled="state",
        context="all",
        stacks="python",
        max_entries=100_000,
        device=selected_device,
        clear_history=True,
    )
    _ALLOCATOR_HISTORY_DEVICES.add(device_key)


def stop_allocator_memory_history_if_enabled(
    device: torch.device | str,
) -> None:
    """Stop allocator stack recording after the final diagnostic report."""
    selected_device = torch.device(device)
    device_key = str(selected_device)
    if device_key not in _ALLOCATOR_HISTORY_DEVICES:
        return
    torch.cuda.memory._record_memory_history(enabled=None, device=selected_device)
    _ALLOCATOR_HISTORY_DEVICES.remove(device_key)


def summarize_allocator_memory_snapshot(
    snapshot: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize CUDA allocator blocks by state, pool, and first vLLM frame."""
    block_bytes_by_state: dict[str, int] = {}
    segment_bytes_by_pool: dict[str, int] = {}
    active_by_frame: dict[str, dict[str, int]] = {}
    segment_total_bytes = 0
    segment_allocated_bytes = 0
    segment_active_bytes = 0
    active_internal_fragmentation_bytes = 0

    for segment in snapshot.get("segments", []):
        total_size = int(segment["total_size"])
        segment_total_bytes += total_size
        segment_allocated_bytes += int(segment.get("allocated_size", 0))
        segment_active_bytes += int(segment.get("active_size", 0))
        pool_key = str(tuple(segment.get("segment_pool_id", ())))
        segment_bytes_by_pool[pool_key] = (
            segment_bytes_by_pool.get(pool_key, 0) + total_size
        )
        for block in segment.get("blocks", []):
            state = str(block["state"])
            block_size = int(block["size"])
            requested_size = int(block.get("requested_size", block_size))
            block_bytes_by_state[state] = (
                block_bytes_by_state.get(state, 0) + block_size
            )
            if state != "active_allocated":
                continue
            active_internal_fragmentation_bytes += block_size - requested_size
            frame = _select_allocator_owner_frame(block.get("frames", []))
            owner = active_by_frame.setdefault(
                frame,
                {"bytes": 0, "requested_bytes": 0, "count": 0},
            )
            owner["bytes"] += block_size
            owner["requested_bytes"] += requested_size
            owner["count"] += 1

    active_allocations_by_frame = [
        {"frame": frame, **values}
        for frame, values in sorted(
            active_by_frame.items(),
            key=lambda item: (-item[1]["bytes"], item[0]),
        )
    ]
    return {
        "segment_total_bytes": segment_total_bytes,
        "segment_allocated_bytes": segment_allocated_bytes,
        "segment_active_bytes": segment_active_bytes,
        "segment_bytes_by_pool": dict(sorted(segment_bytes_by_pool.items())),
        "block_bytes_by_state": dict(sorted(block_bytes_by_state.items())),
        "active_internal_fragmentation_bytes": (active_internal_fragmentation_bytes),
        "active_allocations_by_frame": active_allocations_by_frame,
    }


def _select_allocator_owner_frame(frames: list[Mapping[str, Any]]) -> str:
    for frame in frames:
        filename = str(frame.get("filename", ""))
        marker = "/vllm/"
        if marker not in filename:
            continue
        relative_filename = f"vllm/{filename.split(marker, 1)[1]}"
        return (
            f"{relative_filename}:{int(frame.get('line', 0))}:"
            f"{frame.get('name', '<unknown>')}"
        )
    return "<unattributed>"


def _add_named_memory_details(
    report: dict[str, Any],
    device: torch.device,
    *,
    named_allocations: Mapping[str, int] | None,
    allocator_snapshot: Mapping[str, Any] | None,
) -> None:
    if named_allocations is not None:
        normalized_allocations = {
            str(name): int(value) for name, value in named_allocations.items()
        }
        if any(value < 0 for value in normalized_allocations.values()):
            raise ValueError("Named memory allocations must be non-negative")
        report["named_allocations"] = dict(sorted(normalized_allocations.items()))
    snapshot = allocator_snapshot
    if snapshot is None and str(device) in _ALLOCATOR_HISTORY_DEVICES:
        snapshot = torch.cuda.memory._snapshot(device=device)
    if snapshot is not None:
        report["allocator_snapshot"] = summarize_allocator_memory_snapshot(snapshot)


def _describe_registered_tensor(
    name: str,
    kind: str,
    tensor: torch.Tensor,
) -> dict[str, Any]:
    return {
        "name": name,
        "kind": kind,
        "device": str(tensor.device),
        "dtype": str(tensor.dtype),
        "shape": list(tensor.shape),
        "stride": list(tensor.stride()),
        "storage_offset": tensor.storage_offset(),
        "logical_nbytes": tensor.numel() * tensor.element_size(),
    }


def _select_report_device(
    inventory: Mapping[str, Any], device: torch.device | str | None
) -> torch.device:
    devices = list(inventory["unique_storage_bytes_by_device"])
    if device is not None:
        selected_device = torch.device(device)
        if selected_device.index is None:
            matching_devices = [
                registered_device
                for registered_device in devices
                if torch.device(registered_device).type == selected_device.type
            ]
            if len(matching_devices) == 1:
                return torch.device(matching_devices[0])
        return selected_device
    if len(devices) == 1:
        return torch.device(devices[0])
    accelerator_devices = [
        registered_device
        for registered_device in devices
        if not registered_device.startswith(("cpu", "meta"))
    ]
    if len(accelerator_devices) == 1:
        return torch.device(accelerator_devices[0])
    return torch.device("cpu")


def _collect_device_memory_counters(device: torch.device) -> dict[str, int]:
    if device.type == "cpu":
        return {}
    torch.accelerator.synchronize(device)
    free_bytes, total_bytes = torch.accelerator.get_memory_info(device)
    return {
        "torch_allocated_bytes": torch.accelerator.memory_allocated(device),
        "torch_reserved_bytes": torch.accelerator.memory_reserved(device),
        "torch_peak_allocated_bytes": torch.accelerator.max_memory_allocated(device),
        "torch_peak_reserved_bytes": torch.accelerator.max_memory_reserved(device),
        "device_free_bytes": free_bytes,
        "device_total_bytes": total_bytes,
        "device_used_bytes": total_bytes - free_bytes,
    }


def _add_device_memory_residuals(counters: dict[str, int]) -> None:
    torch_allocated = counters.get("torch_allocated_bytes")
    torch_reserved = counters.get("torch_reserved_bytes")
    device_used = counters.get("device_used_bytes")
    if torch_allocated is not None and torch_reserved is not None:
        counters["allocator_cache_bytes"] = torch_reserved - torch_allocated
    if torch_reserved is not None and device_used is not None:
        counters["non_torch_device_bytes"] = device_used - torch_reserved
    peak_allocated = counters.get("torch_peak_allocated_bytes")
    if peak_allocated is not None and torch_allocated is not None:
        counters["transient_torch_peak_bytes"] = peak_allocated - torch_allocated


def _add_memory_accounting_residuals(
    counters: dict[str, int],
    inventory: Mapping[str, Any],
    device: torch.device,
) -> None:
    bytes_by_device = inventory["unique_storage_bytes_by_device"]
    registered_storage_bytes = bytes_by_device.get(str(device), 0)
    counters["registered_storage_bytes"] = registered_storage_bytes

    torch_allocated = counters.get("torch_allocated_bytes")
    if torch_allocated is not None:
        counters["unregistered_torch_allocated_bytes"] = (
            torch_allocated - registered_storage_bytes
        )
    _add_device_memory_residuals(counters)


def _get_distributed_rank() -> int | None:
    if not torch.distributed.is_available() or not torch.distributed.is_initialized():
        return None
    return torch.distributed.get_rank()


def _make_report_path(
    report_directory: Path,
    stage: str,
    device: torch.device,
) -> Path:
    stage_slug = _slugify_stage(stage)
    device_slug = str(device).replace(":", "-")
    rank = _get_distributed_rank()
    rank_component = "" if rank is None else f"-rank-{rank}"
    filename = (
        f"{stage_slug}{rank_component}-device-{device_slug}-pid-{os.getpid()}.json"
    )
    return report_directory / filename


def _slugify_stage(stage: str) -> str:
    slug = _STAGE_SLUG_PATTERN.sub("-", stage.lower()).strip("-")
    if not slug:
        raise ValueError("Model memory report stage must contain a letter or digit")
    return slug


def _write_atomic_json(path: Path, report: Mapping[str, Any]) -> None:
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        suffix=".tmp",
        delete=False,
    ) as temporary_file:
        temporary_path = Path(temporary_file.name)
        json.dump(report, temporary_file, sort_keys=True, separators=(",", ":"))
        temporary_file.write("\n")
        temporary_file.flush()
        os.fchmod(temporary_file.fileno(), 0o644)
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)
