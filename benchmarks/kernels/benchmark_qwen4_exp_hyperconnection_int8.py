# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import hashlib
import json
import statistics
from pathlib import Path

import torch
import torch.nn.functional as F
from safetensors import safe_open

from vllm.models.qwen4_exp.nvidia.hyperconnection_int8 import (
    HyperconnectionInt8ScaleLayout,
    Qwen4ExpHyperconnectionInt8LinearMethod,
)
from vllm.v1.worker.workspace import (
    init_workspace_manager,
    reset_workspace_manager,
)

_MAXIMUM_NORMALIZED_RMSE = 0.02
_MINIMUM_COSINE_SIMILARITY = 0.9999
_MINIMUM_SPEED_RATIO = 0.90


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_layer_zero_weights(model: Path) -> dict[str, torch.Tensor]:
    index_path = model / "model.safetensors.index.json"
    index = json.loads(index_path.read_text())
    prefix = "model.language_model.layers.0.attn_hyper_connection."
    names = {
        "down": prefix + "input_mix_weight_down.weight",
        "inject": prefix + "block_inject_weight.weight",
        "up": prefix + "input_mix_weight_up.weight",
    }
    shards = {index["weight_map"][name] for name in names.values()}
    if len(shards) != 1:
        raise RuntimeError("Layer-0 hyperconnection weights cross source shards")
    with safe_open(model / next(iter(shards)), framework="pt", device="cpu") as source:
        loaded = {kind: source.get_tensor(name) for kind, name in names.items()}
    valid_output_rows = loaded["down"].shape[0] + loaded["inject"].shape[0]
    padding_rows = (-valid_output_rows) % 16
    merged_down = torch.cat(
        (
            loaded["down"],
            loaded["inject"],
            torch.zeros(
                (padding_rows, loaded["down"].shape[1]),
                dtype=loaded["down"].dtype,
            ),
        )
    )
    return {
        "merged_down": merged_down,
        "up": loaded["up"],
    }


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_flat = reference.float().reshape(-1)
    candidate_flat = candidate.float().reshape(-1)
    error = candidate_flat - reference_flat
    return {
        "normalized_rmse": float(
            torch.linalg.vector_norm(error) / torch.linalg.vector_norm(reference_flat)
        ),
        "cosine_similarity": float(
            F.cosine_similarity(reference_flat[None], candidate_flat[None])
        ),
        "maximum_absolute_error": float(error.abs().amax()),
    }


def _time_operation(operation, iterations: int, repeats: int = 5) -> dict:
    for _ in range(20):
        operation()
    torch.accelerator.synchronize()
    samples = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iterations):
            operation()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end) * 1000 / iterations)
    return {
        "median_microseconds": statistics.median(samples),
        "samples_microseconds": samples,
    }


def _benchmark_matrix(
    name: str,
    source_weight: torch.Tensor,
) -> dict:
    source_weight = source_weight.cuda()
    valid_output_rows = 324 if name == "merged_down" else None
    layout = (
        HyperconnectionInt8ScaleLayout.K_GROUP_128
        if name == "merged_down"
        else HyperconnectionInt8ScaleLayout.PER_ROW
    )
    method = Qwen4ExpHyperconnectionInt8LinearMethod(
        scale_layout=layout,
        valid_output_rows=valid_output_rows,
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(source_weight.clone(), requires_grad=False)
    )
    method.process_weights_after_loading(layer)

    result = {
        "shape": list(source_weight.shape),
        "layout": layout.value,
        "bf16_weight_bytes": source_weight.numel() * source_weight.element_size(),
        "int8_weight_bytes": layer.weight.numel() * layer.weight.element_size(),
        "scale_bytes": layer.weight_scale.numel() * layer.weight_scale.element_size(),
        "token_counts": {},
    }
    for token_count in (1, 2, 256):
        inputs = torch.randn(
            (token_count, source_weight.shape[1]),
            dtype=torch.bfloat16,
            device="cuda",
        )
        reference = F.linear(inputs, source_weight)
        candidate = method.apply(layer, inputs)
        token_result = _metrics(reference, candidate)
        iterations = 1000 if token_count <= 2 else 100
        bf16_timing = _time_operation(
            lambda inputs=inputs: F.linear(inputs, source_weight), iterations
        )
        int8_timing = _time_operation(
            lambda inputs=inputs: method.apply(layer, inputs), iterations
        )
        token_result.update(
            {
                "bf16_timing": bf16_timing,
                "candidate_timing": int8_timing,
                "speed_ratio": bf16_timing["median_microseconds"]
                / int8_timing["median_microseconds"],
            }
        )
        result["token_counts"][str(token_count)] = token_result
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("Qwen hyperconnection INT8 benchmark requires CUDA")
    if torch.cuda.get_device_capability() != (8, 6):
        raise RuntimeError("Qwen hyperconnection INT8 benchmark requires SM86")
    torch.manual_seed(20260829)
    reset_workspace_manager()
    init_workspace_manager(torch.device("cuda"), num_ubatches=1)

    matrices = {
        name: _benchmark_matrix(name, weight)
        for name, weight in _load_layer_zero_weights(args.model).items()
    }
    token_results = [
        token_result
        for matrix in matrices.values()
        for token_result in matrix["token_counts"].values()
    ]
    result = {
        "schema_version": 1,
        "model_index_sha256": _sha256(args.model / "model.safetensors.index.json"),
        "device_name": torch.cuda.get_device_name(),
        "compute_capability": list(torch.cuda.get_device_capability()),
        "acceptance": {
            "maximum_normalized_rmse": _MAXIMUM_NORMALIZED_RMSE,
            "minimum_cosine_similarity": _MINIMUM_COSINE_SIMILARITY,
            "minimum_candidate_to_bf16_speed_ratio": _MINIMUM_SPEED_RATIO,
        },
        "matrices": matrices,
        "numerical_gate_passed": all(
            item["normalized_rmse"] <= _MAXIMUM_NORMALIZED_RMSE
            and item["cosine_similarity"] >= _MINIMUM_COSINE_SIMILARITY
            for item in token_results
        ),
        "performance_gate_passed": all(
            item["speed_ratio"] >= _MINIMUM_SPEED_RATIO for item in token_results
        ),
    }
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "numerical_gate_passed": result["numerical_gate_passed"],
                "performance_gate_passed": result["performance_gate_passed"],
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    if not result["numerical_gate_passed"] or not result["performance_gate_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
