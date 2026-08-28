#!/usr/bin/env python3
"""Compare GGUF-TP and llama.cpp DeepSeek V4 M6 layer-oracle dumps."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import statistics
import struct
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LAYER_COUNT = 43
LAYER_SHAPE = (4, 4096)
LOGIT_COUNT = 129280
LAYER_COSINE_MIN = 0.995
LAYER_NRMSE_MAX = 0.10
LAYER_NMAE_MAX = 0.10
MEDIAN_LAYER_COSINE_MIN = 0.999
MEDIAN_LAYER_NRMSE_MAX = 0.03
MEDIAN_LAYER_NMAE_MAX = 0.03
LOGIT_COSINE_MIN = 0.995
LOGIT_NRMSE_MAX = 0.10
LOGIT_NMAE_MAX = 0.10
LOGIT_TOP10_OVERLAP_MIN = 8


@dataclass(frozen=True)
class VectorMetrics:
    cosine: float
    normalized_rmse: float
    normalized_mae: float

    def as_dict(self) -> dict[str, float]:
        return {
            "cosine": self.cosine,
            "normalized_rmse": self.normalized_rmse,
            "normalized_mae": self.normalized_mae,
        }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        while chunk := source.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"M6 layer oracle invalid JSON: {path}") from error
    if not isinstance(value, dict):
        raise TypeError(f"M6 layer oracle JSON root must be an object: {path}")
    return value


def _verify_manifest_file(root: Path, entry: dict[str, Any]) -> Path:
    relative_path = entry.get("path")
    if not isinstance(relative_path, str) or Path(relative_path).is_absolute():
        raise ValueError("M6 layer oracle manifest path must be relative")
    path = root / relative_path
    expected_size = entry.get("size")
    expected_sha256 = entry.get("sha256")
    if path.stat().st_size != expected_size:
        raise ValueError(f"M6 layer oracle size mismatch: {path}")
    if _sha256(path) != expected_sha256:
        raise ValueError(f"M6 layer oracle SHA-256 mismatch: {path}")
    return path


def _load_vllm_tensor(path: Path) -> list[float]:
    try:
        import torch
    except ImportError as error:
        raise ValueError("M6 layer oracle comparator requires torch") from error
    tensor = torch.load(path, map_location="cpu", weights_only=True)
    if tensor.dtype != torch.float32 or not tensor.is_contiguous():
        raise ValueError(
            f"M6 layer oracle vLLM tensor must be contiguous float32: {path}"
        )
    values = tensor.reshape(-1).tolist()
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"M6 layer oracle vLLM tensor is non-finite: {path}")
    return values


def _load_llama_f32(path: Path, expected_count: int) -> list[float]:
    payload = path.read_bytes()
    if len(payload) != expected_count * 4:
        raise ValueError(f"M6 layer oracle llama.cpp float count mismatch: {path}")
    values = list(struct.unpack(f"<{expected_count}f", payload))
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"M6 layer oracle llama.cpp tensor is non-finite: {path}")
    return values


def _vector_metrics(candidate: list[float], reference: list[float]) -> VectorMetrics:
    if len(candidate) != len(reference) or not candidate:
        raise ValueError("M6 layer oracle vectors must be non-empty and equal length")
    squared_error = math.fsum(
        (candidate_value - reference_value) ** 2
        for candidate_value, reference_value in zip(candidate, reference, strict=True)
    )
    absolute_error = math.fsum(
        abs(candidate_value - reference_value)
        for candidate_value, reference_value in zip(candidate, reference, strict=True)
    )
    reference_squared = math.fsum(value * value for value in reference)
    reference_absolute = math.fsum(abs(value) for value in reference)
    candidate_squared = math.fsum(value * value for value in candidate)
    dot_product = math.fsum(
        candidate_value * reference_value
        for candidate_value, reference_value in zip(candidate, reference, strict=True)
    )
    if reference_squared == 0 or reference_absolute == 0 or candidate_squared == 0:
        raise ValueError("M6 layer oracle metric denominator must be nonzero")
    return VectorMetrics(
        cosine=dot_product / math.sqrt(candidate_squared * reference_squared),
        normalized_rmse=math.sqrt(squared_error / reference_squared),
        normalized_mae=absolute_error / reference_absolute,
    )


def _top_token_ids(values: list[float], count: int) -> list[int]:
    return sorted(range(len(values)), key=lambda index: (-values[index], index))[:count]


def compare_m6_layer_oracle(
    vllm_dir: Path,
    llama_dir: Path,
    token_ids_path: Path,
) -> dict[str, Any]:
    token_ids_sha256 = _sha256(token_ids_path)
    token_count = len(token_ids_path.read_text().split())
    vllm_manifest = _load_json(vllm_dir / "manifest.json")
    llama_manifest = _load_json(llama_dir / "manifest.json")
    if vllm_manifest.get("format") != "gguf-dsv4-layer-oracle-v1":
        raise ValueError("M6 layer oracle unexpected vLLM manifest format")
    if vllm_manifest.get("token_ids_sha256") != token_ids_sha256:
        raise ValueError("M6 layer oracle vLLM token identity mismatch")
    if llama_manifest != {
        "format": "llama-ds4-layer-oracle-v1",
        "token_count": token_count,
        "layer_count": LAYER_COUNT,
        "layer_value_count": math.prod(LAYER_SHAPE),
        "logit_count": LOGIT_COUNT,
    }:
        raise ValueError("M6 layer oracle llama.cpp manifest contract mismatch")

    layer_entries = vllm_manifest.get("layers")
    if not isinstance(layer_entries, list) or len(layer_entries) != LAYER_COUNT:
        raise ValueError("M6 layer oracle vLLM layer inventory mismatch")
    layer_reports: list[dict[str, Any]] = []
    llama_hashes: dict[str, str] = {}
    for expected_layer, entry in enumerate(layer_entries):
        if not isinstance(entry, dict) or entry.get("layer") != expected_layer:
            raise ValueError("M6 layer oracle vLLM layer ordering mismatch")
        if entry.get("shape") != list(LAYER_SHAPE):
            raise ValueError(
                f"M6 layer oracle vLLM layer shape mismatch: {expected_layer}"
            )
        vllm_path = _verify_manifest_file(vllm_dir, entry)
        llama_path = llama_dir / f"layer-{expected_layer:03d}.f32"
        candidate = _load_vllm_tensor(vllm_path)
        reference = _load_llama_f32(llama_path, math.prod(LAYER_SHAPE))
        metrics = _vector_metrics(candidate, reference)
        passed = (
            metrics.cosine >= LAYER_COSINE_MIN
            and metrics.normalized_rmse <= LAYER_NRMSE_MAX
            and metrics.normalized_mae <= LAYER_NMAE_MAX
        )
        layer_reports.append(
            {"layer": expected_layer, **metrics.as_dict(), "passed": passed}
        )
        llama_hashes[llama_path.name] = _sha256(llama_path)

    logits_entry = vllm_manifest.get("logits")
    if not isinstance(logits_entry, dict) or logits_entry.get("shape") != [LOGIT_COUNT]:
        raise ValueError("M6 layer oracle vLLM logits shape mismatch")
    vllm_logits_path = _verify_manifest_file(vllm_dir, logits_entry)
    llama_logits_path = llama_dir / "logits.f32"
    candidate_logits = _load_vllm_tensor(vllm_logits_path)
    reference_logits = _load_llama_f32(llama_logits_path, LOGIT_COUNT)
    logit_metrics = _vector_metrics(candidate_logits, reference_logits)
    candidate_top10 = _top_token_ids(candidate_logits, 10)
    reference_top10 = _top_token_ids(reference_logits, 10)
    top10_overlap = len(set(candidate_top10) & set(reference_top10))
    logits_passed = (
        logit_metrics.cosine >= LOGIT_COSINE_MIN
        and logit_metrics.normalized_rmse <= LOGIT_NRMSE_MAX
        and logit_metrics.normalized_mae <= LOGIT_NMAE_MAX
        and candidate_top10[0] == reference_top10[0]
        and top10_overlap >= LOGIT_TOP10_OVERLAP_MIN
    )
    llama_hashes[llama_logits_path.name] = _sha256(llama_logits_path)

    median_cosine = statistics.median(report["cosine"] for report in layer_reports)
    median_nrmse = statistics.median(
        report["normalized_rmse"] for report in layer_reports
    )
    median_nmae = statistics.median(
        report["normalized_mae"] for report in layer_reports
    )
    layer_aggregate_passed = (
        all(report["passed"] for report in layer_reports)
        and median_cosine >= MEDIAN_LAYER_COSINE_MIN
        and median_nrmse <= MEDIAN_LAYER_NRMSE_MAX
        and median_nmae <= MEDIAN_LAYER_NMAE_MAX
    )
    return {
        "format": "gguf-tp-m6-layer-comparison-v1",
        "passed": layer_aggregate_passed and logits_passed,
        "token_count": token_count,
        "token_ids_sha256": token_ids_sha256,
        "thresholds": {
            "layer_cosine_min": LAYER_COSINE_MIN,
            "layer_normalized_rmse_max": LAYER_NRMSE_MAX,
            "layer_normalized_mae_max": LAYER_NMAE_MAX,
            "median_layer_cosine_min": MEDIAN_LAYER_COSINE_MIN,
            "median_layer_normalized_rmse_max": MEDIAN_LAYER_NRMSE_MAX,
            "median_layer_normalized_mae_max": MEDIAN_LAYER_NMAE_MAX,
            "logit_cosine_min": LOGIT_COSINE_MIN,
            "logit_normalized_rmse_max": LOGIT_NRMSE_MAX,
            "logit_normalized_mae_max": LOGIT_NMAE_MAX,
            "logit_top10_overlap_min": LOGIT_TOP10_OVERLAP_MIN,
        },
        "layers": layer_reports,
        "layer_summary": {
            "passed": layer_aggregate_passed,
            "median_cosine": median_cosine,
            "median_normalized_rmse": median_nrmse,
            "median_normalized_mae": median_nmae,
        },
        "logits": {
            **logit_metrics.as_dict(),
            "candidate_top10": candidate_top10,
            "reference_top10": reference_top10,
            "top1_equal": candidate_top10[0] == reference_top10[0],
            "top10_overlap": top10_overlap,
            "passed": logits_passed,
        },
        "vllm_manifest_sha256": _sha256(vllm_dir / "manifest.json"),
        "llama_manifest_sha256": _sha256(llama_dir / "manifest.json"),
        "llama_file_sha256": llama_hashes,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--vllm-dir", type=Path, required=True)
    parser.add_argument("--llama-dir", type=Path, required=True)
    parser.add_argument("--token-ids", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    try:
        report = compare_m6_layer_oracle(args.vllm_dir, args.llama_dir, args.token_ids)
    except (OSError, TypeError, ValueError) as error:
        print(str(error), file=sys.stderr)
        return 2
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
