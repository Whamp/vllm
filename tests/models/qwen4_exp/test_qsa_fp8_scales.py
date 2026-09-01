# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
import math
from pathlib import Path

import pytest
import torch

from vllm.models.qwen4_exp.common import qsa_fp8_calibration
from vllm.models.qwen4_exp.common.qsa_fp8 import (
    load_qsa_fp8_layer_scales,
    merge_qsa_fp8_calibration_reports,
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    """This pure scale-file test never initializes an accelerator."""
    return False


def _write_scale_file(path: Path, data: object) -> Path:
    path.write_text(json.dumps(data))
    return path


def test_qsa_fp8_calibration_flush_defers_during_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_base = tmp_path / "qsa-fp8"
    monkeypatch.setattr(qsa_fp8_calibration, "_CALIBRATION_PATH", str(output_base))
    monkeypatch.setattr(
        qsa_fp8_calibration,
        "_CALIBRATION_STATE",
        {
            "mtp.layers.48.self_attn.attn": {
                "k_absmax": torch.tensor(4.0),
                "v_absmax": torch.tensor(8.0),
                "calls": 10,
            }
        },
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: True)

    qsa_fp8_calibration.flush_qsa_fp8_calibration()

    assert not list(tmp_path.glob("qsa-fp8.rank*.json"))


def test_qsa_fp8_calibration_flush_writes_outside_cuda_graph_capture(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    output_base = tmp_path / "qsa-fp8"
    monkeypatch.setattr(qsa_fp8_calibration, "_CALIBRATION_PATH", str(output_base))
    monkeypatch.setattr(
        qsa_fp8_calibration,
        "_CALIBRATION_STATE",
        {
            "mtp.layers.48.self_attn.attn": {
                "k_absmax": torch.tensor(4.0),
                "v_absmax": torch.tensor(8.0),
                "calls": 10,
            }
        },
    )
    monkeypatch.setattr(torch.cuda, "is_current_stream_capturing", lambda: False)

    qsa_fp8_calibration.flush_qsa_fp8_calibration()

    report = json.loads((tmp_path / "qsa-fp8.rank0.json").read_text())
    assert report["per_layer"]["mtp.layers.48.self_attn.attn"] == {
        "k_absmax": 4.0,
        "v_absmax": 8.0,
        "calls": 10,
    }


def test_load_qsa_fp8_layer_scales_requires_exact_calibrated_layer(
    tmp_path: Path,
) -> None:
    layer_name = "language_model.model.layers.3.self_attn.attn"
    path = _write_scale_file(
        tmp_path / "qsa-fp8-scales.json",
        {
            "per_layer": {
                layer_name: {
                    "k_scale": 0.015,
                    "v_scale": 0.025,
                    "k_absmax": 6.72,
                    "v_absmax": 11.2,
                }
            }
        },
    )

    scales = load_qsa_fp8_layer_scales(path, layer_name)

    assert scales.k_scale == pytest.approx(0.015)
    assert scales.v_scale == pytest.approx(0.025)


def test_load_qsa_fp8_layer_scales_rejects_default_fallback(tmp_path: Path) -> None:
    path = _write_scale_file(
        tmp_path / "qsa-fp8-scales.json",
        {
            "per_layer": {
                "another.layer": {"k_scale": 0.01, "v_scale": 0.02},
            },
            "default": {"k_scale": 1.0, "v_scale": 1.0},
        },
    )

    with pytest.raises(ValueError, match="missing calibrated layer"):
        load_qsa_fp8_layer_scales(path, "requested.layer")


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("k_scale", 0.0),
        ("k_scale", -0.01),
        ("k_scale", math.inf),
        ("k_scale", math.nan),
        ("k_scale", True),
        ("v_scale", 0.0),
        ("v_scale", -0.01),
        ("v_scale", math.inf),
        ("v_scale", math.nan),
        ("v_scale", "0.01"),
    ],
)
def test_load_qsa_fp8_layer_scales_rejects_invalid_values(
    tmp_path: Path,
    field: str,
    value: object,
) -> None:
    entry: dict[str, object] = {"k_scale": 0.01, "v_scale": 0.02}
    entry[field] = value
    path = _write_scale_file(
        tmp_path / "qsa-fp8-scales.json",
        {"per_layer": {"layer": entry}},
    )

    with pytest.raises(ValueError, match=f"invalid {field}"):
        load_qsa_fp8_layer_scales(path, "layer")


@pytest.mark.parametrize("data", [None, [], {}, {"per_layer": []}])
def test_load_qsa_fp8_layer_scales_rejects_invalid_schema(
    tmp_path: Path,
    data: object,
) -> None:
    path = _write_scale_file(tmp_path / "qsa-fp8-scales.json", data)

    with pytest.raises(ValueError, match="QSA FP8 scale file"):
        load_qsa_fp8_layer_scales(path, "layer")


def test_merge_qsa_fp8_calibration_reports_uses_cross_rank_maxima(
    tmp_path: Path,
) -> None:
    rank0 = _write_scale_file(
        tmp_path / "calibration.rank0.json",
        {
            "per_layer": {
                "layer.3.attn": {"k_absmax": 4.0, "v_absmax": 12.0},
                "layer.7.attn": {"k_absmax": 8.0, "v_absmax": 3.0},
            }
        },
    )
    rank1 = _write_scale_file(
        tmp_path / "calibration.rank1.json",
        {
            "per_layer": {
                "layer.3.attn": {"k_absmax": 5.0, "v_absmax": 10.0},
                "layer.7.attn": {"k_absmax": 7.0, "v_absmax": 6.0},
            }
        },
    )

    merged = merge_qsa_fp8_calibration_reports([rank1, rank0], safety_margin=1.125)

    assert merged["source_reports"] == [str(rank0), str(rank1)]
    assert merged["per_layer"]["layer.3.attn"] == pytest.approx(
        {
            "k_absmax": 5.0,
            "v_absmax": 12.0,
            "k_scale": 5.0 * 1.125 / 448.0,
            "v_scale": 12.0 * 1.125 / 448.0,
        }
    )
    assert merged["per_layer"]["layer.7.attn"] == pytest.approx(
        {
            "k_absmax": 8.0,
            "v_absmax": 6.0,
            "k_scale": 8.0 * 1.125 / 448.0,
            "v_scale": 6.0 * 1.125 / 448.0,
        }
    )


def test_merge_qsa_fp8_calibration_reports_rejects_rank_layer_mismatch(
    tmp_path: Path,
) -> None:
    rank0 = _write_scale_file(
        tmp_path / "calibration.rank0.json",
        {"per_layer": {"layer.3.attn": {"k_absmax": 4.0, "v_absmax": 5.0}}},
    )
    rank1 = _write_scale_file(
        tmp_path / "calibration.rank1.json",
        {"per_layer": {"layer.7.attn": {"k_absmax": 4.0, "v_absmax": 5.0}}},
    )

    with pytest.raises(ValueError, match="different layer set"):
        merge_qsa_fp8_calibration_reports([rank0, rank1], safety_margin=1.125)
