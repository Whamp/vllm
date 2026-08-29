# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import torch
from safetensors import safe_open

MODEL = Path("/model")
INDEX = MODEL / "model.safetensors.index.json"
FP8_MAX = 448.0
BLOCK = 128
SM_COUNT = 82


torch.set_num_threads(4)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def quantize_per_tensor(weight):
    scale = weight.abs().amax().float() / FP8_MAX
    quantized = (weight.float() / scale).to(torch.float8_e4m3fn)
    return quantized.float() * scale


def quantize_per_block(weight):
    rows, columns = weight.shape
    padded_rows = math.ceil(rows / BLOCK) * BLOCK
    padded_columns = math.ceil(columns / BLOCK) * BLOCK
    padded = torch.zeros((padded_rows, padded_columns), dtype=weight.dtype)
    padded[:rows, :columns] = weight
    blocks = padded.view(
        padded_rows // BLOCK,
        BLOCK,
        padded_columns // BLOCK,
        BLOCK,
    )
    amax = blocks.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    scales = amax / FP8_MAX
    quantized = (blocks * (1.0 / scales)).to(torch.float8_e4m3fn)
    reconstructed = quantized.float() * scales
    return reconstructed.view(padded_rows, padded_columns)[:rows, :columns]


def metrics(reference, reconstructed):
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


def marlin_padded_nk(size_n, size_k, group_size=128):
    candidates = (
        (
            math.ceil(size_n / 64) * 64,
            math.ceil(size_k / math.lcm(128, group_size)) * math.lcm(128, group_size),
        ),
        (
            math.ceil(size_n / 128) * 128,
            math.ceil(size_k / math.lcm(64, group_size)) * math.lcm(64, group_size),
        ),
    )
    return min(candidates, key=lambda nk: (nk[0] * nk[1], nk[0] + nk[1]))


def marlin_storage(size_n, size_k):
    padded_n, padded_k = marlin_padded_nk(size_n, size_k)
    return {
        "logical_n": size_n,
        "logical_k": size_k,
        "padded_n": padded_n,
        "padded_k": padded_k,
        "weight_bytes": padded_n * padded_k,
        "scale_bytes": (padded_k // BLOCK) * padded_n * 2,
        "workspace_bytes": SM_COUNT * 4,
    }


def aggregate(metric_rows):
    totals = defaultdict(float)
    for row in metric_rows:
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
        "maximum_tensor_normalized_rmse": max(
            row["normalized_rmse"] for row in metric_rows
        ),
        "minimum_tensor_cosine_similarity": min(
            row["cosine_similarity"] for row in metric_rows
        ),
        "maximum_absolute_error": max(
            row["maximum_absolute_error"] for row in metric_rows
        ),
    }


index = json.loads(INDEX.read_text())
suffixes = {
    "input_mix_weight_down.weight": "down",
    "block_inject_weight.weight": "inject",
    "input_mix_weight_up.weight": "up",
}
groups = defaultdict(dict)
for name, shard in index["weight_map"].items():
    for suffix, kind in suffixes.items():
        marker = "." + suffix
        if name.endswith(marker) and "hyper_connection" in name:
            prefix = name[: -len(marker)]
            groups[prefix][kind] = (name, shard)
            break

if len(groups) != 97:
    raise RuntimeError(f"Expected 97 hyperconnection groups, found {len(groups)}")

results = []
storage_rows = []
current_bf16_matrix_bytes = 0
for group_index, (prefix, tensors) in enumerate(sorted(groups.items()), start=1):
    if set(tensors) not in ({"down", "up"}, {"down", "inject", "up"}):
        raise RuntimeError(f"Unexpected tensor set for {prefix}: {sorted(tensors)}")
    shards = {shard for _, shard in tensors.values()}
    if len(shards) != 1:
        raise RuntimeError(f"Hyperconnection group crosses shards: {prefix}")
    shard = next(iter(shards))
    with safe_open(MODEL / shard, framework="pt", device="cpu") as source:
        loaded = {kind: source.get_tensor(name) for kind, (name, _) in tensors.items()}

    down_parts = [loaded["down"]]
    logical_slices = [("down", 0, loaded["down"].shape[0])]
    if "inject" in loaded:
        inject_start = loaded["down"].shape[0]
        down_parts.append(loaded["inject"])
        logical_slices.append(
            ("inject", inject_start, inject_start + loaded["inject"].shape[0])
        )
        logical_rows = sum(part.shape[0] for part in down_parts)
        pad_rows = (-logical_rows) % 16
        if pad_rows:
            down_parts.append(
                torch.zeros(
                    (pad_rows, loaded["down"].shape[1]), dtype=loaded["down"].dtype
                )
            )
    merged_down = torch.cat(down_parts, dim=0)
    matrices = (
        ("merged_down", merged_down, logical_slices),
        ("up", loaded["up"], [("up", 0, loaded["up"].shape[0])]),
    )

    for matrix_kind, weight, slices in matrices:
        current_bf16_matrix_bytes += weight.numel() * weight.element_size()
        storage = marlin_storage(weight.shape[0], weight.shape[1])
        storage["group"] = prefix
        storage["matrix_kind"] = matrix_kind
        storage_rows.append(storage)
        reconstructed_by_scheme = {
            "per_tensor": quantize_per_tensor(weight),
            "block_128x128": quantize_per_block(weight),
        }
        for scheme, reconstructed in reconstructed_by_scheme.items():
            for logical_kind, start, stop in slices:
                row = metrics(weight[start:stop], reconstructed[start:stop])
                row.update(
                    {
                        "group": prefix,
                        "matrix_kind": matrix_kind,
                        "logical_kind": logical_kind,
                        "scheme": scheme,
                    }
                )
                results.append(row)
    del loaded, merged_down, reconstructed_by_scheme
    if group_index % 10 == 0:
        print(f"processed_groups={group_index}", file=sys.stderr, flush=True)

current_registered_bytes = 1_304_842_240
norm_bytes = current_registered_bytes - current_bf16_matrix_bytes
candidate_registered_bytes = norm_bytes + sum(
    row["weight_bytes"] + row["scale_bytes"] for row in storage_rows
)
candidate_workspace_bytes = sum(row["workspace_bytes"] for row in storage_rows)
summary = {
    "schema_version": 1,
    "model_index_sha256": sha256(INDEX),
    "model_index_tensor_count": len(index["weight_map"]),
    "hyperconnection_group_count": len(groups),
    "logical_matrix_component_count": len(results) // 2,
    "quantization": {
        "fp8_dtype": "E4M3FN",
        "fp8_max": FP8_MAX,
        "block_shape": [BLOCK, BLOCK],
        "scale_dtype_after_marlin": "BF16",
        "marlin_sm_count": SM_COUNT,
    },
    "schemes": {
        scheme: aggregate([row for row in results if row["scheme"] == scheme])
        for scheme in ("per_tensor", "block_128x128")
    },
    "storage": {
        "current_registered_bytes_per_rank": current_registered_bytes,
        "current_bf16_matrix_bytes_per_rank": current_bf16_matrix_bytes,
        "unchanged_norm_bytes_per_rank": norm_bytes,
        "candidate_registered_bytes_per_rank": candidate_registered_bytes,
        "candidate_workspace_bytes_per_rank": candidate_workspace_bytes,
        "registered_bytes_saved_per_rank": current_registered_bytes
        - candidate_registered_bytes,
    },
    "tensor_metrics": results,
    "marlin_storage": storage_rows,
}
json.dump(summary, sys.stdout, indent=2, sort_keys=True)
print()
