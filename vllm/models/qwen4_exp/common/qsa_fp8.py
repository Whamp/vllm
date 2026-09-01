# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Calibrated E4M3 scales for the QSA main K/V cache."""

from __future__ import annotations

import json
import math
from collections.abc import Iterable
from dataclasses import dataclass
from numbers import Real
from pathlib import Path
from typing import TypedDict


@dataclass(frozen=True, slots=True)
class QSAFP8LayerScales:
    """Per-layer E4M3 divisors for QSA key and value cache writes."""

    k_scale: float
    v_scale: float


class QSAFP8CalibrationLayerResult(TypedDict):
    """One merged layer's absolute maxima and calibrated E4M3 scales."""

    k_absmax: float
    v_absmax: float
    k_scale: float
    v_scale: float


class QSAFP8CalibrationMergeResult(TypedDict):
    """Deterministic output from merging all QSA rank calibration reports."""

    schema_version: int
    format: str
    safety_margin: float
    source_reports: list[str]
    per_layer: dict[str, QSAFP8CalibrationLayerResult]


def _parse_positive_qsa_fp8_scale(
    entry: dict[str, object],
    field: str,
    *,
    layer_name: str,
    scale_path: Path,
) -> float:
    value = entry.get(field)
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(
            f"QSA FP8 scale file {scale_path} has invalid {field} "
            f"for {layer_name}: {value!r}"
        )
    scale = float(value)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError(
            f"QSA FP8 scale file {scale_path} has invalid {field} "
            f"for {layer_name}: {value!r}"
        )
    return scale


def load_qsa_fp8_layer_scales(
    scale_path: str | Path,
    layer_name: str,
) -> QSAFP8LayerScales:
    """Load one exact calibrated QSA layer entry; defaults are forbidden."""

    path = Path(scale_path)
    try:
        data = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"QSA FP8 scale file {path} could not be loaded") from error

    if not isinstance(data, dict) or not isinstance(data.get("per_layer"), dict):
        raise ValueError(f"QSA FP8 scale file {path} requires a per_layer object")
    entry = data["per_layer"].get(layer_name)
    if not isinstance(entry, dict):
        raise ValueError(
            f"QSA FP8 scale file {path} is missing calibrated layer {layer_name}"
        )

    return QSAFP8LayerScales(
        k_scale=_parse_positive_qsa_fp8_scale(
            entry,
            "k_scale",
            layer_name=layer_name,
            scale_path=path,
        ),
        v_scale=_parse_positive_qsa_fp8_scale(
            entry,
            "v_scale",
            layer_name=layer_name,
            scale_path=path,
        ),
    )


def merge_qsa_fp8_calibration_reports(
    report_paths: Iterable[str | Path],
    *,
    safety_margin: float,
) -> QSAFP8CalibrationMergeResult:
    """Merge rank maxima into deterministic per-layer E4M3 cache scales."""

    if not math.isfinite(safety_margin) or safety_margin < 1.0:
        raise ValueError(f"QSA FP8 safety margin must be at least 1: {safety_margin}")
    paths = sorted(Path(path) for path in report_paths)
    if not paths:
        raise ValueError("QSA FP8 calibration merge requires at least one report")

    reports: list[tuple[Path, dict[str, object]]] = []
    expected_layers: set[str] | None = None
    for path in paths:
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as error:
            raise ValueError(
                f"QSA FP8 calibration report {path} could not be loaded"
            ) from error
        per_layer = data.get("per_layer") if isinstance(data, dict) else None
        if not isinstance(per_layer, dict):
            raise ValueError(
                f"QSA FP8 calibration report {path} requires a per_layer object"
            )
        layers = set(per_layer)
        if expected_layers is None:
            expected_layers = layers
        elif layers != expected_layers:
            raise ValueError(
                f"QSA FP8 calibration report {path} has a different layer set"
            )
        reports.append((path, per_layer))

    assert expected_layers is not None
    merged_layers: dict[str, QSAFP8CalibrationLayerResult] = {}
    for layer_name in sorted(expected_layers):
        maxima = {"k_absmax": 0.0, "v_absmax": 0.0}
        for path, per_layer in reports:
            entry = per_layer[layer_name]
            if not isinstance(entry, dict):
                raise ValueError(
                    f"QSA FP8 calibration report {path} has invalid layer {layer_name}"
                )
            for field in maxima:
                value = entry.get(field)
                if isinstance(value, bool) or not isinstance(value, Real):
                    raise ValueError(
                        f"QSA FP8 calibration report {path} has invalid {field} "
                        f"for {layer_name}: {value!r}"
                    )
                numeric_value = float(value)
                if not math.isfinite(numeric_value) or numeric_value <= 0:
                    raise ValueError(
                        f"QSA FP8 calibration report {path} has invalid {field} "
                        f"for {layer_name}: {value!r}"
                    )
                maxima[field] = max(maxima[field], numeric_value)
        merged_layers[layer_name] = QSAFP8CalibrationLayerResult(
            k_absmax=maxima["k_absmax"],
            v_absmax=maxima["v_absmax"],
            k_scale=maxima["k_absmax"] * safety_margin / 448.0,
            v_scale=maxima["v_absmax"] * safety_margin / 448.0,
        )

    return {
        "schema_version": 1,
        "format": "float8_e4m3fn",
        "safety_margin": safety_margin,
        "source_reports": [str(path) for path in paths],
        "per_layer": merged_layers,
    }
