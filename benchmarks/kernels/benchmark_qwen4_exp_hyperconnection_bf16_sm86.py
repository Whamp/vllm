# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark a native BF16 Qwen3.8 hyperconnection skinny GEMM on SM86.

The CUDA extension assigns complete output rows to blocks, loads two BF16 K
values per lane, reuses each weight across M=2, accumulates in FP32, and writes
BF16 output. Pointer-distinct weights and CUDA Graph replay model the production
working set without changing production dispatch.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path
from typing import Any

from benchmark_qwen4_exp_hyperconnection_gemm import (
    PROJECTION_CASES,
    capture_graph,
    output_error_metrics,
    summarize_samples,
    time_graphs_alternating,
)


@dataclasses.dataclass(frozen=True, slots=True)
class Sm86Bf16KernelPlan:
    """Native SM86 skinny-GEMM launch plan for one exact projection."""

    block_threads: int
    outputs_per_block: int

    def validate(self, projection_name: str) -> None:
        """Reject unsupported launch geometry before extension compilation."""

        projection = PROJECTION_CASES[projection_name]
        if self.block_threads not in (32, 64, 128, 256):
            raise ValueError(
                "Qwen hyperconnection SM86 block threads must be 32/64/128/256"
            )
        if self.outputs_per_block not in (1, 4, 8):
            raise ValueError(
                "Qwen hyperconnection SM86 outputs per block must be 1/4/8"
            )
        if projection.output_features % self.outputs_per_block:
            raise ValueError(
                "Qwen hyperconnection output features must divide outputs per block"
            )
        if projection_name == "down" and self.outputs_per_block != 1:
            raise ValueError(
                "Qwen hyperconnection merged-down plans require one row per block"
            )


def load_sm86_bf16_extension(source: Path, build_directory: Path):
    """Compile the benchmark-only native SM86 BF16 CUDA extension."""

    import os

    from torch.utils.cpp_extension import load

    os.environ["TORCH_CUDA_ARCH_LIST"] = "8.6"
    build_directory.mkdir(parents=True, exist_ok=True)
    return load(
        name="qwen4_exp_hyperconnection_bf16_sm86",
        sources=[str(source)],
        build_directory=str(build_directory),
        extra_cflags=["-O3"],
        extra_cuda_cflags=["-O3", "--use_fast_math", "-lineinfo"],
        verbose=True,
    )


def run_native_bf16_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Compare native SM86 BF16 execution with Torch/CUTLASS."""

    import torch

    projection = PROJECTION_CASES[args.projection]
    plan = Sm86Bf16KernelPlan(args.block_threads, args.outputs_per_block)
    plan.validate(args.projection)

    torch.accelerator.set_device_index(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability != (8, 6):
        raise RuntimeError(
            "Qwen hyperconnection native BF16 benchmark requires "
            f"SM86, got {capability}"
        )
    torch.manual_seed(args.seed)

    extension = load_sm86_bf16_extension(args.cuda_source, args.build_directory)
    activation = torch.randn(
        (args.tokens, projection.input_features),
        dtype=torch.bfloat16,
        device="cuda",
    )
    weights = [
        (
            torch.randn(
                (projection.output_features, projection.input_features),
                dtype=torch.bfloat16,
                device="cuda",
            )
            / projection.input_features**0.5
        )
        for _ in range(args.calls)
    ]
    candidate_outputs = [
        torch.empty(
            (args.tokens, projection.output_features),
            dtype=torch.bfloat16,
            device="cuda",
        )
        for _ in range(args.calls)
    ]

    def launch_bf16() -> list[torch.Tensor]:
        return [torch.nn.functional.linear(activation, weight) for weight in weights]

    def launch_candidate() -> list[torch.Tensor]:
        for weight, output in zip(weights, candidate_outputs, strict=True):
            extension.run(
                activation,
                weight,
                output,
                plan.block_threads,
                plan.outputs_per_block,
            )
        return candidate_outputs

    baseline_outputs = launch_bf16()
    outputs = launch_candidate()
    torch.accelerator.synchronize()
    reference = activation.float() @ weights[0].float().t()
    baseline_error = output_error_metrics(baseline_outputs[0], reference)
    candidate_error = output_error_metrics(outputs[0], reference)
    candidate_vs_baseline = output_error_metrics(
        outputs[0], baseline_outputs[0].float()
    )

    baseline_graph, _ = capture_graph(launch_bf16)
    candidate_graph, graph_outputs = capture_graph(launch_candidate)
    candidate_graph.replay()
    torch.accelerator.synchronize()
    first_replay = graph_outputs[0].clone()
    candidate_graph.replay()
    torch.accelerator.synchronize()
    graph_bitwise_deterministic = torch.equal(first_replay, graph_outputs[0])

    samples = time_graphs_alternating(
        {"bf16_cutlass": baseline_graph, "bf16_sm86_native": candidate_graph},
        calls=args.calls,
        warmups=args.warmups,
        repeats=args.repeats,
        replays=args.replays,
    )
    summaries = {
        name: summarize_samples(backend_samples)
        for name, backend_samples in samples.items()
    }
    for summary in summaries.values():
        summary["logical_weight_gb_per_s"] = (
            projection.weight_bytes / summary["median_us"] / 1000.0
        )

    baseline_us = summaries["bf16_cutlass"]["median_us"]
    candidate_us = summaries["bf16_sm86_native"]["median_us"]
    projected_savings_ms = (
        (baseline_us - candidate_us) * projection.production_calls / 1000.0
    )
    properties = torch.cuda.get_device_properties(args.device)
    extension_path = Path(extension.__file__)
    return {
        "schema_version": 1,
        "gpu": properties.name,
        "compute_capability": list(capability),
        "multiprocessor_count": properties.multi_processor_count,
        "torch_version": torch.__version__,
        "projection": dataclasses.asdict(projection),
        "tokens": args.tokens,
        "calls": args.calls,
        "plan": dataclasses.asdict(plan),
        "baseline_error": baseline_error,
        "candidate_error": candidate_error,
        "candidate_vs_baseline": candidate_vs_baseline,
        "candidate_graph_bitwise_deterministic": graph_bitwise_deterministic,
        "timing": summaries,
        "candidate_speedup": baseline_us / candidate_us,
        "projected_projection_savings_ms_per_decode_step": projected_savings_ms,
        "extension": {
            "path": str(extension_path),
            "bytes": extension_path.stat().st_size,
        },
        "decision_contract": {
            "status": "screening_only",
            "required_combined_trace_weighted_savings_ms_per_generated_token": 0.8,
            "required_candidate_vs_baseline_cosine": 0.9999,
            "required_candidate_vs_baseline_normalized_rmse": 0.01,
        },
        "arguments": {
            "warmups": args.warmups,
            "repeats": args.repeats,
            "replays": args.replays,
            "seed": args.seed,
        },
    }


def parse_args() -> argparse.Namespace:
    """Parse the native SM86 BF16 benchmark command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--projection", choices=sorted(PROJECTION_CASES), required=True)
    parser.add_argument("--tokens", type=int, choices=(1, 2), required=True)
    parser.add_argument("--block-threads", type=int, required=True)
    parser.add_argument("--outputs-per-block", type=int, required=True)
    parser.add_argument("--cuda-source", type=Path, required=True)
    parser.add_argument("--build-directory", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--calls", type=int, default=16)
    parser.add_argument("--warmups", type=int, default=20)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--replays", type=int, default=20)
    parser.add_argument("--seed", type=int, default=8421)
    args = parser.parse_args()
    if args.calls <= 0:
        parser.error("--calls must be positive")
    for name in ("warmups", "repeats", "replays"):
        if getattr(args, name) <= 0:
            parser.error(f"--{name} must be positive")
    Sm86Bf16KernelPlan(args.block_threads, args.outputs_per_block).validate(
        args.projection
    )
    return args


def main() -> None:
    """Run the benchmark and atomically publish its JSON result."""

    args = parse_args()
    result = run_native_bf16_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
