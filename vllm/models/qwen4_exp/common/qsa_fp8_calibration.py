# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Opt-in calibration recorder for QSA E4M3 main-cache scales."""

from __future__ import annotations

import atexit
import json
import os
from pathlib import Path
from typing import TypedDict

import torch

QSA_FP8_CALIBRATION_PATH_ENV = "VLLM_QSA_FP8_CALIBRATION_PATH"
_FLUSH_INTERVAL = 10
_CALIBRATION_PATH = os.environ.get(QSA_FP8_CALIBRATION_PATH_ENV)


class _LayerCalibrationState(TypedDict):
    k_absmax: torch.Tensor
    v_absmax: torch.Tensor
    calls: int


_CALIBRATION_STATE: dict[str, _LayerCalibrationState] = {}


def _tensor_finite_absmax(tensor: torch.Tensor) -> torch.Tensor:
    maximum = tensor.detach().float().abs().amax()
    return torch.nan_to_num(maximum, nan=0.0, posinf=0.0, neginf=0.0)


def record_qsa_fp8_calibration(
    layer_name: str,
    key: torch.Tensor,
    value: torch.Tensor,
) -> None:
    """Record K/V absolute maxima only when QSA calibration is enabled."""

    if _CALIBRATION_PATH is None:
        return
    state = _CALIBRATION_STATE.get(layer_name)
    if state is None:
        state = {
            "k_absmax": torch.zeros((), dtype=torch.float32, device=key.device),
            "v_absmax": torch.zeros((), dtype=torch.float32, device=value.device),
            "calls": 0,
        }
        _CALIBRATION_STATE[layer_name] = state
    torch.maximum(
        state["k_absmax"],
        _tensor_finite_absmax(key),
        out=state["k_absmax"],
    )
    torch.maximum(
        state["v_absmax"],
        _tensor_finite_absmax(value),
        out=state["v_absmax"],
    )
    state["calls"] += 1
    if state["calls"] % _FLUSH_INTERVAL == 0:
        flush_qsa_fp8_calibration()


def _tensor_parallel_rank() -> int:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return get_tensor_model_parallel_rank()
    except (AssertionError, RuntimeError):
        return 0


def flush_qsa_fp8_calibration() -> None:
    """Atomically write this rank's current QSA calibration maxima."""

    if _CALIBRATION_PATH is None or not _CALIBRATION_STATE:
        return
    rank = _tensor_parallel_rank()
    output_path = Path(f"{_CALIBRATION_PATH}.rank{rank}.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    per_layer = {}
    for layer_name, state in sorted(_CALIBRATION_STATE.items()):
        per_layer[layer_name] = {
            "k_absmax": float(state["k_absmax"].item()),
            "v_absmax": float(state["v_absmax"].item()),
            "calls": state["calls"],
        }
    report = {
        "schema_version": 1,
        "rank": rank,
        "per_layer": per_layer,
    }
    temporary_path = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, output_path)


if _CALIBRATION_PATH is not None:
    atexit.register(flush_qsa_fp8_calibration)
