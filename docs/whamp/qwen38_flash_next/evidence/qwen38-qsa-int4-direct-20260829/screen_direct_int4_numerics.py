# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import argparse
import json
import math
from pathlib import Path

import torch
import torch.nn.functional as F

HEAD_DIM = 256
QUERY_HEADS = 6
SELECTED_TOKENS = 2051
NUM_SPLITS = 32
MAXIMUM_NRMSE = 0.17
MINIMUM_COSINE = 0.985
MAXIMUM_NRMSE_DELTA = 0.02
MAXIMUM_COSINE_LOSS = 0.002


def hadamard_matrix(order: int) -> torch.Tensor:
    matrix = torch.ones((1, 1), dtype=torch.float32)
    while matrix.shape[0] < order:
        matrix = torch.cat(
            (
                torch.cat((matrix, matrix), dim=1),
                torch.cat((matrix, -matrix), dim=1),
            ),
            dim=0,
        )
    if matrix.shape != (order, order):
        raise ValueError(f"Hadamard order must be a power of two, got {order}")
    return matrix


def asymmetric_int4(
    values: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    minimum = values.amin(dim=-1)
    maximum = values.amax(dim=-1)
    scale = ((maximum - minimum) / 15).clamp_min(1e-8)
    zero_point = (-minimum / scale).round().clamp(0, 15)
    codes = (values / scale[..., None] + zero_point[..., None]).round().clamp(0, 15)
    return codes, scale, zero_point


def symmetric_int8(values: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scale = (values.abs().amax(dim=-1) / 127).clamp_min(1e-8)
    codes = (values / scale[..., None]).round().clamp(-127, 127)
    return codes, scale


def output_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_flat = reference.reshape(-1)
    candidate_flat = candidate.reshape(-1)
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


def simulate(seed: int) -> dict:
    generator = torch.Generator().manual_seed(seed)
    query = torch.randn((QUERY_HEADS, HEAD_DIM), generator=generator)
    key = torch.randn((SELECTED_TOKENS, HEAD_DIM), generator=generator)
    value = torch.randn((SELECTED_TOKENS, HEAD_DIM), generator=generator)

    signs = torch.randint(
        0,
        2,
        (HEAD_DIM,),
        generator=generator,
        dtype=torch.int64,
    ).float()
    signs = signs.mul_(2).sub_(1)
    hadamard = hadamard_matrix(HEAD_DIM)
    forward = signs[:, None] * hadamard
    inverse = forward.T / HEAD_DIM

    reference_scores = query @ key.T / math.sqrt(HEAD_DIM)
    probabilities = torch.softmax(reference_scores, dim=-1)
    reference_output = probabilities @ value

    transformed_query = query @ forward
    transformed_key = key @ forward
    transformed_value = value @ forward
    key_codes, key_scale, key_zero_point = asymmetric_int4(transformed_key)
    value_codes, value_scale, value_zero_point = asymmetric_int4(transformed_value)

    key_reconstructed = (key_codes - key_zero_point[:, None]) * key_scale[:, None]
    value_reconstructed = (value_codes - value_zero_point[:, None]) * value_scale[
        :, None
    ]
    float_q4_scores = (
        transformed_query @ key_reconstructed.T / HEAD_DIM / math.sqrt(HEAD_DIM)
    )
    float_q4_probability = torch.softmax(float_q4_scores, dim=-1)
    float_q4_transformed_output = float_q4_probability @ value_reconstructed
    float_q4_output = float_q4_transformed_output @ inverse

    query_codes, query_scale = symmetric_int8(transformed_query)
    integer_scores = query_codes @ key_codes.T
    integer_scores -= query_codes.sum(dim=-1, keepdim=True) * key_zero_point[None, :]
    direct_scores = (
        integer_scores
        * query_scale[:, None]
        * key_scale[None, :]
        / HEAD_DIM
        / math.sqrt(HEAD_DIM)
    )
    direct_probability = torch.softmax(direct_scores, dim=-1)

    transformed_output = torch.zeros((QUERY_HEADS, HEAD_DIM), dtype=torch.float32)
    for split_index in range(NUM_SPLITS):
        start = split_index * SELECTED_TOKENS // NUM_SPLITS
        stop = (split_index + 1) * SELECTED_TOKENS // NUM_SPLITS
        weighted_probability = (
            direct_probability[:, start:stop] * value_scale[None, start:stop]
        )
        probability_codes, probability_scale = symmetric_int8(weighted_probability)
        probability_codes.clamp_(min=0)
        split_values = value_codes[start:stop]
        split_zero_points = value_zero_point[start:stop]
        integer_output = probability_codes @ split_values
        zero_point_correction = (probability_codes * split_zero_points[None, :]).sum(
            dim=1, keepdim=True
        )
        transformed_output += (
            integer_output - zero_point_correction
        ) * probability_scale[:, None]
    direct_output = transformed_output @ inverse

    float_metrics = output_metrics(reference_output, float_q4_output)
    direct_metrics = output_metrics(reference_output, direct_output)
    return {
        "seed": seed,
        "float_unpack_q4": float_metrics,
        "direct_integer_q4": direct_metrics,
        "direct_minus_float_nrmse": direct_metrics["normalized_rmse"]
        - float_metrics["normalized_rmse"],
        "float_minus_direct_cosine": float_metrics["cosine_similarity"]
        - direct_metrics["cosine_similarity"],
    }


def aggregate(cases: list[dict], key: str) -> dict:
    return {
        "mean_normalized_rmse": sum(case[key]["normalized_rmse"] for case in cases)
        / len(cases),
        "maximum_normalized_rmse": max(case[key]["normalized_rmse"] for case in cases),
        "mean_cosine_similarity": sum(case[key]["cosine_similarity"] for case in cases)
        / len(cases),
        "minimum_cosine_similarity": min(
            case[key]["cosine_similarity"] for case in cases
        ),
        "maximum_absolute_error": max(
            case[key]["maximum_absolute_error"] for case in cases
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.seeds <= 0:
        raise ValueError("seeds must be positive")

    torch.set_num_threads(4)
    cases = [simulate(20260829 + offset) for offset in range(args.seeds)]
    direct_summary = aggregate(cases, "direct_integer_q4")
    result = {
        "schema_version": 1,
        "geometry": {
            "head_dim": HEAD_DIM,
            "query_heads": QUERY_HEADS,
            "selected_tokens": SELECTED_TOKENS,
            "num_splits": NUM_SPLITS,
            "seeds": args.seeds,
        },
        "acceptance": {
            "maximum_normalized_rmse": MAXIMUM_NRMSE,
            "minimum_cosine_similarity": MINIMUM_COSINE,
            "maximum_nrmse_delta_from_float_q4": MAXIMUM_NRMSE_DELTA,
            "maximum_cosine_loss_from_float_q4": MAXIMUM_COSINE_LOSS,
        },
        "float_unpack_q4": aggregate(cases, "float_unpack_q4"),
        "direct_integer_q4": direct_summary,
        "maximum_nrmse_delta_from_float_q4": max(
            case["direct_minus_float_nrmse"] for case in cases
        ),
        "maximum_cosine_loss_from_float_q4": max(
            case["float_minus_direct_cosine"] for case in cases
        ),
        "cases": cases,
    }
    result["passed"] = (
        direct_summary["maximum_normalized_rmse"] <= MAXIMUM_NRMSE
        and direct_summary["minimum_cosine_similarity"] >= MINIMUM_COSINE
        and result["maximum_nrmse_delta_from_float_q4"] <= MAXIMUM_NRMSE_DELTA
        and result["maximum_cosine_loss_from_float_q4"] <= MAXIMUM_COSINE_LOSS
    )
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"output": str(args.output), "passed": result["passed"]}))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
