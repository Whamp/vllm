# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen W4A16 reconstruction error for all Qwen3.8 HC up projections.

This CPU-only stream reads one source shard at a time and applies the same
symmetric signed INT4 range used by Marlin's uint4b8 representation. It records
per-tensor and aggregate metrics for per-channel, group-64, and group-32 scales.
The screen does not establish task-level model quality.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open

INT4_MIN = -8
INT4_MAX = 7
SCALE_DTYPE = torch.float16
FP16_MIN_SUBNORMAL = 2**-24
UP_SUFFIX = ".input_mix_weight_up.weight"


def file_sha256(path: Path) -> str:
    """Return the hexadecimal SHA-256 of one model artifact."""

    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_symmetric_int4(weight: torch.Tensor, group_size: int) -> torch.Tensor:
    """Reconstruct signed INT4 weights with FP16 scales along K groups."""

    rows, columns = weight.shape
    effective_group = columns if group_size == -1 else group_size
    if columns % effective_group:
        raise ValueError(
            "Qwen hyperconnection W4 group size must divide input features"
        )
    blocks = weight.float().view(rows, columns // effective_group, effective_group)
    maximum = blocks.amax(dim=2, keepdim=True)
    minimum = blocks.amin(dim=2, keepdim=True)
    scales = torch.maximum(
        maximum.abs() / INT4_MAX,
        minimum.abs() / abs(INT4_MIN),
    )
    scales = scales.clamp_min(FP16_MIN_SUBNORMAL).to(SCALE_DTYPE).float()
    codes = (blocks / scales).round().clamp(INT4_MIN, INT4_MAX)
    return (codes * scales).view(rows, columns)


def tensor_error_metrics(
    reference: torch.Tensor,
    reconstructed: torch.Tensor,
) -> dict[str, float | int]:
    """Return additive error terms and normalized tensor metrics."""

    reference = reference.float().reshape(-1)
    reconstructed = reconstructed.float().reshape(-1)
    error = reconstructed - reference
    reference_squared = torch.dot(reference, reference).item()
    error_squared = torch.dot(error, error).item()
    dot = torch.dot(reference, reconstructed).item()
    reconstructed_squared = torch.dot(reconstructed, reconstructed).item()
    return {
        "element_count": reference.numel(),
        "reference_squared_sum": reference_squared,
        "error_squared_sum": error_squared,
        "dot_sum": dot,
        "reconstructed_squared_sum": reconstructed_squared,
        "normalized_rmse": math.sqrt(error_squared / reference_squared),
        "cosine_similarity": dot / math.sqrt(reference_squared * reconstructed_squared),
        "maximum_absolute_error": error.abs().amax().item(),
    }


def aggregate_error_metrics(rows: list[dict[str, Any]]) -> dict[str, float | int]:
    """Aggregate additive metrics across every screened up projection."""

    totals: defaultdict[str, float] = defaultdict(float)
    for row in rows:
        for key in (
            "element_count",
            "reference_squared_sum",
            "error_squared_sum",
            "dot_sum",
            "reconstructed_squared_sum",
        ):
            totals[key] += row[key]
    return {
        "element_count": int(totals["element_count"]),
        "normalized_rmse": math.sqrt(
            totals["error_squared_sum"] / totals["reference_squared_sum"]
        ),
        "cosine_similarity": totals["dot_sum"]
        / math.sqrt(
            totals["reference_squared_sum"] * totals["reconstructed_squared_sum"]
        ),
        "maximum_tensor_normalized_rmse": max(row["normalized_rmse"] for row in rows),
        "minimum_tensor_cosine_similarity": min(
            row["cosine_similarity"] for row in rows
        ),
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in rows),
        "passing_tensor_count": sum(
            row["normalized_rmse"] <= 0.02 and row["cosine_similarity"] >= 0.9999
            for row in rows
        ),
        "tensor_count": len(rows),
    }


def screen_hyperconnection_w4_up(
    model_directory: Path,
    *,
    thread_count: int,
) -> dict[str, Any]:
    """Stream and screen every indexed Qwen hyperconnection up tensor."""

    torch.set_num_threads(thread_count)
    index_path = model_directory / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    up_tensors = sorted(
        (name, shard)
        for name, shard in index["weight_map"].items()
        if name.endswith(UP_SUFFIX) and "hyper_connection" in name
    )
    if len(up_tensors) != 97:
        raise RuntimeError(
            "Qwen hyperconnection W4 screen expected 97 up tensors, "
            f"got {len(up_tensors)}"
        )

    schemes = {"per_channel": -1, "group_64": 64, "group_32": 32}
    rows: list[dict[str, Any]] = []
    for tensor_index, (name, shard) in enumerate(up_tensors, start=1):
        with safe_open(model_directory / shard, framework="pt", device="cpu") as source:
            weight = source.get_tensor(name)
        if weight.ndim != 2 or weight.shape[1] != 320:
            raise RuntimeError(
                f"Qwen hyperconnection W4 screen unexpected {name} shape {weight.shape}"
            )
        for scheme, group_size in schemes.items():
            reconstructed = quantize_symmetric_int4(weight, group_size)
            row = tensor_error_metrics(weight, reconstructed)
            row.update(
                {
                    "tensor_name": name,
                    "source_shard": shard,
                    "shape": list(weight.shape),
                    "scheme": scheme,
                    "group_size": group_size,
                }
            )
            rows.append(row)
            del reconstructed
        del weight
        if tensor_index % 10 == 0:
            print(f"processed_up_tensors={tensor_index}", flush=True)

    return {
        "schema_version": 1,
        "model_index_sha256": file_sha256(index_path),
        "model_index_tensor_count": len(index["weight_map"]),
        "up_tensor_count": len(up_tensors),
        "quantization": {
            "weight_codes": "symmetric signed INT4",
            "minimum_code": INT4_MIN,
            "maximum_code": INT4_MAX,
            "scale_dtype": "float16",
        },
        "decision_contract": {
            "maximum_normalized_rmse": 0.02,
            "minimum_cosine_similarity": 0.9999,
            "status": "model_reconstruction_screen_only",
        },
        "schemes": {
            scheme: aggregate_error_metrics(
                [row for row in rows if row["scheme"] == scheme]
            )
            for scheme in schemes
        },
        "tensor_metrics": rows,
    }


def parse_args() -> argparse.Namespace:
    """Parse the CPU-only W4 reconstruction-screen command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()
    if args.threads <= 0:
        parser.error("--threads must be positive")
    return args


def main() -> None:
    """Run the screen and atomically publish its JSON result."""

    args = parse_args()
    result = screen_hyperconnection_w4_up(args.model, thread_count=args.threads)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result["schemes"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
