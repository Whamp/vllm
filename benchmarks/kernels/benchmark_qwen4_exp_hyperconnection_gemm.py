# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark exact Qwen3.8 hyperconnection BF16 GEMMs on SM86.

The benchmark compares Torch/CUTLASS with vLLM's shape-dynamic CuTe skinny
GEMM for the two production projection shapes. It rotates through independent
weights and captures all calls in one CUDA Graph so warm L2 reuse cannot stand
in for the 1.2 GiB model-wide hyperconnection weight stream.

This is an experiment gate. It does not change Qwen production dispatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import statistics
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any


@dataclasses.dataclass(frozen=True, slots=True)
class ProjectionCase:
    """One Qwen3.8 hyperconnection projection and its calls per decode step."""

    name: str
    output_features: int
    input_features: int
    production_calls: int

    @property
    def weight_bytes(self) -> int:
        return self.output_features * self.input_features * 2


PROJECTION_CASES = {
    "down": ProjectionCase(
        name="merged_down_injection",
        output_features=336,
        input_features=10240,
        production_calls=96,
    ),
    "up": ProjectionCase(
        name="up",
        output_features=10240,
        input_features=320,
        production_calls=97,
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class CandidateConfig:
    """CuTe skinny-GEMM launch configuration for one exact token count."""

    block_size: int
    outputs_per_block: int
    k_unroll: int
    vector_width: int
    static_k: int | None

    def validate(self, case: ProjectionCase, tokens: int) -> None:
        if not 1 <= tokens <= 16:
            raise ValueError("Qwen hyperconnection benchmark requires 1 <= M <= 16")
        if self.block_size % 32:
            raise ValueError("Qwen hyperconnection block size must be warp aligned")
        if case.output_features % self.outputs_per_block:
            raise ValueError(
                "Qwen hyperconnection output width must divide outputs_per_block"
            )
        tile_k = self.block_size * self.vector_width
        if case.input_features % tile_k:
            raise ValueError(
                "Qwen hyperconnection K must be divisible by block_size * vector_width"
            )
        if self.static_k is not None and self.static_k != case.input_features:
            raise ValueError("Qwen hyperconnection static K must match the projection")
        if self.static_k is not None and self.static_k < 2 * tile_k:
            raise ValueError("Qwen hyperconnection static K requires two tiles")


def parse_candidate_config(value: str) -> CandidateConfig:
    """Parse BLOCK,OUTPUTS,K_UNROLL,VECTOR[,STATIC_K] from the CLI."""

    try:
        fields = [int(field) for field in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "config must be BLOCK,OUTPUTS,K_UNROLL,VECTOR[,STATIC_K]"
        ) from error
    if len(fields) == 4:
        return CandidateConfig(*fields, static_k=None)
    if len(fields) == 5:
        return CandidateConfig(*fields)
    raise argparse.ArgumentTypeError(
        "config must be BLOCK,OUTPUTS,K_UNROLL,VECTOR[,STATIC_K]"
    )


def percentile(samples: Sequence[float], fraction: float) -> float:
    """Return a linearly interpolated percentile from microsecond samples."""

    ordered = sorted(samples)
    position = fraction * (len(ordered) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def summarize_samples(samples: Sequence[float]) -> dict[str, Any]:
    """Summarize per-call latency samples in microseconds."""

    mean = statistics.mean(samples)
    return {
        "median_us": statistics.median(samples),
        "p10_us": percentile(samples, 0.1),
        "p90_us": percentile(samples, 0.9),
        "mean_us": mean,
        "cv_pct": statistics.pstdev(samples) / mean * 100.0,
        "samples_us": list(samples),
    }


def output_error_metrics(actual, reference) -> dict[str, float]:
    """Compare a BF16 GEMM output with an independent FP32 reference."""

    import torch

    actual_fp32 = actual.float()
    reference_fp32 = reference.float()
    error = actual_fp32 - reference_fp32
    reference_rms = reference_fp32.square().mean().sqrt().clamp_min(1e-12)
    nrmse = error.square().mean().sqrt() / reference_rms
    cosine = torch.nn.functional.cosine_similarity(
        actual_fp32.flatten(), reference_fp32.flatten(), dim=0
    )
    return {
        "max_abs_error": error.abs().max().item(),
        "normalized_rmse": nrmse.item(),
        "cosine": cosine.item(),
    }


def capture_graph(launch: Callable[[], list[Any]]):
    """Capture one pointer-distinct projection sequence in a CUDA Graph."""

    import torch

    outputs = launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = launch()
    return graph, outputs


def time_graphs_alternating(
    graphs: dict[str, Any],
    *,
    calls: int,
    warmups: int,
    repeats: int,
    replays: int,
) -> dict[str, list[float]]:
    """Time baseline and candidate graphs in alternating sample order."""

    import torch

    for graph in graphs.values():
        for _ in range(warmups):
            graph.replay()
    torch.accelerator.synchronize()

    samples = {name: [] for name in graphs}
    names = list(graphs)
    for repeat in range(repeats):
        ordered_names = names if repeat % 2 == 0 else list(reversed(names))
        for name in ordered_names:
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(replays):
                graphs[name].replay()
            end.record()
            end.synchronize()
            per_call_us = start.elapsed_time(end) * 1000.0 / (replays * calls)
            samples[name].append(per_call_us)
    return samples


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the exact-shape Torch versus CuTe comparison on one SM86 GPU."""

    import torch

    from vllm.model_executor.kernels.linear.cute_dsl.skinny_gemm import (
        SkinnyGemmConfig,
        shape_dynamic_skinny_gemm,
    )

    case = PROJECTION_CASES[args.projection]
    config: CandidateConfig = args.config
    config.validate(case, args.tokens)

    torch.accelerator.set_device_index(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability != (8, 6):
        raise RuntimeError(
            f"Qwen hyperconnection benchmark requires SM86, got {capability}"
        )
    torch.manual_seed(args.seed)

    calls = args.calls if args.calls is not None else case.production_calls
    activation = torch.randn(
        (args.tokens, case.input_features),
        dtype=torch.bfloat16,
        device="cuda",
    )
    weights = [
        torch.randn(
            (case.output_features, case.input_features),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(calls)
    ]
    cute_config = SkinnyGemmConfig(
        num_rows=args.tokens,
        block_size=config.block_size,
        outputs_per_block=config.outputs_per_block,
        k_unroll=config.k_unroll,
        vector_width=config.vector_width,
        static_k=config.static_k,
    )

    def launch_torch() -> list[torch.Tensor]:
        return [torch.nn.functional.linear(activation, weight) for weight in weights]

    def launch_cute() -> list[torch.Tensor]:
        return [
            shape_dynamic_skinny_gemm(activation, weight, cute_config)
            for weight in weights
        ]

    # Compile the CuTe kernel before correctness or timing.
    candidate_outputs = launch_cute()
    baseline_outputs = launch_torch()
    torch.accelerator.synchronize()

    reference = activation.float() @ weights[0].float().t()
    baseline_error = output_error_metrics(baseline_outputs[0], reference)
    candidate_error = output_error_metrics(candidate_outputs[0], reference)
    candidate_vs_baseline = output_error_metrics(
        candidate_outputs[0], baseline_outputs[0].float()
    )

    baseline_graph, baseline_graph_outputs = capture_graph(launch_torch)
    candidate_graph, candidate_graph_outputs = capture_graph(launch_cute)
    candidate_graph.replay()
    torch.accelerator.synchronize()
    first_replay = candidate_graph_outputs[0].clone()
    candidate_graph.replay()
    torch.accelerator.synchronize()
    graph_bitwise_deterministic = torch.equal(first_replay, candidate_graph_outputs[0])

    samples = time_graphs_alternating(
        {"torch_cutlass": baseline_graph, "cute_skinny": candidate_graph},
        calls=calls,
        warmups=args.warmups,
        repeats=args.repeats,
        replays=args.replays,
    )
    summaries = {
        name: summarize_samples(backend_samples)
        for name, backend_samples in samples.items()
    }
    for summary in summaries.values():
        summary["effective_weight_gb_per_s"] = (
            case.weight_bytes / summary["median_us"] / 1000.0
        )
    speedup = (
        summaries["torch_cutlass"]["median_us"] / summaries["cute_skinny"]["median_us"]
    )

    return {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(args.device),
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "projection": dataclasses.asdict(case),
        "tokens": args.tokens,
        "calls": calls,
        "candidate_config": dataclasses.asdict(config),
        "baseline_error": baseline_error,
        "candidate_error": candidate_error,
        "candidate_vs_baseline": candidate_vs_baseline,
        "candidate_graph_bitwise_deterministic": graph_bitwise_deterministic,
        "timing": summaries,
        "candidate_speedup": speedup,
        "decision_contract": {
            "status": "screening_only",
            "required_trace_weighted_savings_ms_per_token": 0.8,
            "note": (
                "Combine down and up results using the production trace mix; "
                "one projection cannot pass the service-level gate alone."
            ),
        },
        "arguments": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "replays": args.replays,
            "seed": args.seed,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", choices=sorted(PROJECTION_CASES), required=True)
    parser.add_argument("--tokens", type=int, choices=(1, 2), required=True)
    parser.add_argument(
        "--config",
        type=parse_candidate_config,
        required=True,
        help="BLOCK,OUTPUTS,K_UNROLL,VECTOR[,STATIC_K]",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calls", type=int)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8421)
    args = parser.parse_args()
    if args.calls is not None and args.calls <= 0:
        parser.error("--calls must be positive")
    for name in ("warmups", "repeats", "replays"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    return args


def main() -> None:
    args = parse_args()
    result = run_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
