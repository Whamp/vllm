# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
from pathlib import Path


def aggregate(rows: list[dict]) -> dict:
    reference_squared = sum(row["reference_squared_sum"] for row in rows)
    error_squared = sum(row["error_squared_sum"] for row in rows)
    dot = sum(row["dot_sum"] for row in rows)
    reconstructed_squared = sum(row["reconstructed_squared_sum"] for row in rows)
    return {
        "element_count": sum(row["element_count"] for row in rows),
        "normalized_rmse": math.sqrt(error_squared / reference_squared),
        "cosine_similarity": dot / math.sqrt(reference_squared * reconstructed_squared),
        "maximum_tensor_normalized_rmse": max(row["normalized_rmse"] for row in rows),
        "minimum_tensor_cosine_similarity": min(
            row["cosine_similarity"] for row in rows
        ),
        "maximum_absolute_error": max(row["maximum_absolute_error"] for row in rows),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("group_screen", type=Path)
    parser.add_argument("fp8_screen", type=Path)
    args = parser.parse_args()

    group_screen = json.loads(args.group_screen.read_text())
    fp8_screen = json.loads(args.fp8_screen.read_text())
    logical_k = {
        (row["group"], row["matrix_kind"]): row["logical_k"]
        for row in fp8_screen["marlin_storage"]
    }

    selected_rows = []
    per_row_rows = {}
    for row in group_screen["tensor_metrics"]:
        key = (row["group"], row["component"])
        if row["scheme"] == "per_row":
            per_row_rows[key] = row
        expected_scheme = "per_row" if row["component"] == "up" else "per_row_kgroup128"
        if row["scheme"] == expected_scheme:
            selected_rows.append(row)

    extra_weight_bytes = 0
    extra_scale_bytes = 0
    for key, row in per_row_rows.items():
        group, component = key
        if component == "up":
            continue
        columns = logical_k[(group, "merged_down")]
        rows = row["element_count"] // columns
        padded_columns = math.ceil(columns / 128) * 128
        extra_weight_bytes += rows * (padded_columns - columns)
        extra_scale_bytes += rows * ((padded_columns // 128) - 1) * 2

    per_row_storage = group_screen["storage"]["candidates"]["per_row"]
    candidate_registered_bytes = (
        per_row_storage["candidate_registered_bytes_per_rank"]
        + extra_weight_bytes
        + extra_scale_bytes
    )
    current_registered_bytes = group_screen["storage"][
        "current_registered_bytes_per_rank"
    ]
    saved_bytes = current_registered_bytes - candidate_registered_bytes

    result = {
        "schema_version": 1,
        "source_model_index_sha256": group_screen["model_index_sha256"],
        "component_policy": {
            "down": "symmetric INT8, FP16 scale per output row and K-group-128",
            "inject": "symmetric INT8, FP16 scale per output row and K-group-128",
            "up": "symmetric INT8, one FP16 scale per output row",
        },
        "separate_component_scales": True,
        "reconstruction": aggregate(selected_rows),
        "storage": {
            "current_registered_bytes_per_rank": current_registered_bytes,
            "candidate_registered_bytes_per_rank": candidate_registered_bytes,
            "registered_bytes_saved_per_rank": saved_bytes,
            "registered_mib_saved_per_rank": saved_bytes / 2**20,
            "candidate_registered_gib_per_rank": candidate_registered_bytes / 2**30,
            "extra_weight_bytes_over_per_row": extra_weight_bytes,
            "extra_scale_bytes_over_per_row": extra_scale_bytes,
        },
        "limitations": [
            "Weight reconstruction does not validate activation quantization.",
            "Storage assumes a custom exact-row layout without output-row padding.",
            "No SM86 kernel, CUDA Graph, sanitizer, or serving result is implied.",
        ],
    }
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
