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

_REPORT_SCHEMA_VERSION = 1
_STAGE_SLUG_PATTERN = re.compile(r"[^a-z0-9]+")


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
    _write_atomic_json(report_path, report)
    return report_path


def capture_device_memory_report_if_enabled(
    *,
    stage: str,
    device: torch.device | str,
    reset_peak_after_capture: bool = False,
) -> Path | None:
    """Write a device-only report when the diagnostics directory is configured."""
    report_dir = envs.VLLM_MODEL_MEMORY_REPORT_DIR
    if report_dir is None:
        return None
    report_path = capture_device_memory_report(
        stage=stage,
        report_dir=report_dir,
        device=device,
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
    _write_atomic_json(report_path, report)
    return report_path


def capture_model_memory_report_if_enabled(
    model: nn.Module,
    *,
    stage: str,
    device: torch.device | str | None = None,
    include_storage_details: bool = True,
    reset_peak_after_capture: bool = False,
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
        os.fsync(temporary_file.fileno())
    os.replace(temporary_path, path)
