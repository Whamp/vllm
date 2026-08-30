# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark exact Qwen3.8 hyperconnection W8A16/W4A16 Marlin GEMMs.

The benchmark compares BF16 Torch/CUTLASS with symmetric weight-only Marlin on
SM86 for the two production hyperconnection shapes. It rotates pointer-distinct
weights beyond L2 and captures each backend in a CUDA Graph. Weight preparation,
quantization, packing, allocation, and random input creation remain outside the
timed region.

This is an experiment gate. It does not change Qwen production dispatch, and
random-weight reconstruction error is not a model-quality result.
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
class HyperconnectionProjection:
    """One Qwen3.8 hyperconnection projection and its calls per decode step."""

    name: str
    output_features: int
    input_features: int
    production_calls: int

    @property
    def bf16_weight_bytes(self) -> int:
        """Return logical BF16 weight bytes for one projection."""

        return self.output_features * self.input_features * 2


HYPERCONNECTION_PROJECTIONS = {
    "down": HyperconnectionProjection(
        name="merged_down_injection",
        output_features=336,
        input_features=10240,
        production_calls=96,
    ),
    "up": HyperconnectionProjection(
        name="up",
        output_features=10240,
        input_features=320,
        production_calls=97,
    ),
}


@dataclasses.dataclass(frozen=True, slots=True)
class MarlinExperimentPlan:
    """Weight-only Marlin format and scale grouping for one projection."""

    bits: int
    group_size: int

    def validate(self, projection: HyperconnectionProjection) -> None:
        """Reject unsupported bit widths or quantization groups before CUDA."""

        if self.bits not in (4, 8):
            raise ValueError("Qwen hyperconnection Marlin bits must be 4 or 8")
        if self.group_size not in (-1, 32, 64, 128):
            raise ValueError(
                "Qwen hyperconnection Marlin group size must be -1, 32, 64, or 128"
            )
        if self.group_size > 0 and projection.input_features % self.group_size:
            raise ValueError(
                "Qwen hyperconnection Marlin group size must divide input features"
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


def summarize_latency_samples(samples: Sequence[float]) -> dict[str, Any]:
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
    """Compare a candidate output with an independent FP32 reference."""

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


def capture_cuda_graph(launch: Callable[[], list[Any]]):
    """Capture one pointer-distinct projection sequence in a CUDA Graph."""

    import torch

    outputs = launch()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        outputs = launch()
    return graph, outputs


def time_cuda_graphs_alternating(
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


def prepare_marlin_weight(
    weight_kn,
    *,
    bits: int,
    group_size: int,
):
    """Quantize and repack one KxN weight through production Marlin helpers."""

    import torch

    from vllm import _custom_ops as ops
    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        marlin_make_empty_g_idx,
        marlin_pad_qweight,
        marlin_pad_scales,
        marlin_padded_nk,
        marlin_permute_scales,
    )
    from vllm.model_executor.layers.quantization.utils.quant_utils import (
        gptq_pack,
        gptq_quantize_weights,
    )
    from vllm.scalar_type import scalar_types

    quant_type = scalar_types.uint4b8 if bits == 4 else scalar_types.uint8b128
    size_k, size_n = weight_kn.shape
    dequantized_weight, quantized_weight, scales, _, _ = gptq_quantize_weights(
        weight_kn,
        quant_type,
        group_size,
        act_order=False,
    )
    packed_weight = gptq_pack(
        quantized_weight,
        quant_type.size_bits,
        size_k,
        size_n,
    )
    padded_n, padded_k = marlin_padded_nk(size_n, size_k, group_size)
    packed_weight = marlin_pad_qweight(
        packed_weight,
        size_n,
        size_k,
        padded_n,
        padded_k,
    )
    repacked_weight = ops.gptq_marlin_repack(
        b_q_weight=packed_weight,
        perm=torch.empty(0, dtype=torch.int, device=weight_kn.device),
        size_k=padded_k,
        size_n=padded_n,
        num_bits=quant_type.size_bits,
    )
    scales = marlin_pad_scales(
        scales,
        size_n,
        size_k,
        padded_n,
        padded_k,
        group_size,
    )
    permuted_scales = marlin_permute_scales(
        scales,
        size_k=padded_k,
        size_n=padded_n,
        group_size=group_size,
    )
    empty = marlin_make_empty_g_idx(weight_kn.device)
    return {
        "quant_type": quant_type,
        "dequantized_weight": dequantized_weight,
        "repacked_weight": repacked_weight,
        "permuted_scales": permuted_scales,
        "empty_g_idx": empty,
        "padded_n": padded_n,
        "padded_k": padded_k,
    }


def run_marlin_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    """Run the exact-shape BF16 versus weight-only Marlin comparison."""

    import torch

    from vllm.model_executor.layers.quantization.utils.marlin_utils import (
        apply_gptq_marlin_linear,
        marlin_make_workspace_new,
    )

    projection = HYPERCONNECTION_PROJECTIONS[args.projection]
    plan = MarlinExperimentPlan(bits=args.bits, group_size=args.group_size)
    plan.validate(projection)

    torch.accelerator.set_device_index(args.device)
    capability = torch.cuda.get_device_capability(args.device)
    if capability != (8, 6):
        raise RuntimeError(
            f"Qwen hyperconnection Marlin benchmark requires SM86, got {capability}"
        )
    torch.manual_seed(args.seed)

    activation = torch.randn(
        (args.tokens, projection.input_features),
        dtype=torch.bfloat16,
        device="cuda",
    )
    original_weight_kn = (
        torch.randn(
            (projection.input_features, projection.output_features),
            dtype=torch.bfloat16,
            device="cuda",
        )
        / projection.input_features**0.5
    )
    prepared = prepare_marlin_weight(
        original_weight_kn,
        bits=plan.bits,
        group_size=plan.group_size,
    )

    # Clone pointer-distinct weights after one preparation. Identical values are
    # intentional: this isolates execution while making the working set exceed L2.
    bf16_weights = [
        original_weight_kn.t().contiguous().clone() for _ in range(args.calls)
    ]
    marlin_weights = [prepared["repacked_weight"].clone() for _ in range(args.calls)]
    marlin_scales = [prepared["permuted_scales"].clone() for _ in range(args.calls)]
    workspace = marlin_make_workspace_new(activation.device)

    def launch_bf16() -> list[torch.Tensor]:
        return [
            torch.nn.functional.linear(activation, weight) for weight in bf16_weights
        ]

    def launch_marlin() -> list[torch.Tensor]:
        return [
            apply_gptq_marlin_linear(
                input=activation,
                weight=weight,
                weight_scale=scale,
                weight_zp=prepared["empty_g_idx"],
                g_idx=prepared["empty_g_idx"],
                g_idx_sort_indices=prepared["empty_g_idx"],
                workspace=workspace,
                wtype=prepared["quant_type"],
                output_size_per_partition=projection.output_features,
                input_size_per_partition=projection.input_features,
                is_k_full=True,
            )
            for weight, scale in zip(marlin_weights, marlin_scales, strict=True)
        ]

    bf16_outputs = launch_bf16()
    marlin_outputs = launch_marlin()
    torch.accelerator.synchronize()

    original_reference = activation.float() @ original_weight_kn.float()
    quantized_reference = activation.float() @ prepared["dequantized_weight"].float()
    bf16_error = output_error_metrics(bf16_outputs[0], original_reference)
    marlin_quantization_error = output_error_metrics(
        marlin_outputs[0], original_reference
    )
    marlin_execution_error = output_error_metrics(
        marlin_outputs[0], quantized_reference
    )

    bf16_graph, _ = capture_cuda_graph(launch_bf16)
    marlin_graph, marlin_graph_outputs = capture_cuda_graph(launch_marlin)
    marlin_graph.replay()
    torch.accelerator.synchronize()
    first_replay = marlin_graph_outputs[0].clone()
    marlin_graph.replay()
    torch.accelerator.synchronize()
    graph_bitwise_deterministic = torch.equal(first_replay, marlin_graph_outputs[0])

    samples = time_cuda_graphs_alternating(
        {"bf16_cutlass": bf16_graph, f"w{plan.bits}a16_marlin": marlin_graph},
        calls=args.calls,
        warmups=args.warmups,
        repeats=args.repeats,
        replays=args.replays,
    )
    summaries = {
        name: summarize_latency_samples(backend_samples)
        for name, backend_samples in samples.items()
    }
    candidate_name = f"w{plan.bits}a16_marlin"
    summaries["bf16_cutlass"]["logical_weight_gb_per_s"] = (
        projection.bf16_weight_bytes / summaries["bf16_cutlass"]["median_us"] / 1000.0
    )
    physical_marlin_bytes = (
        prepared["repacked_weight"].nbytes + prepared["permuted_scales"].nbytes
    )
    summaries[candidate_name]["physical_weight_gb_per_s"] = (
        physical_marlin_bytes / summaries[candidate_name]["median_us"] / 1000.0
    )

    baseline_us = summaries["bf16_cutlass"]["median_us"]
    candidate_us = summaries[candidate_name]["median_us"]
    per_step_savings_ms = (
        (baseline_us - candidate_us) * projection.production_calls / 1000.0
    )
    return {
        "schema_version": 1,
        "gpu": torch.cuda.get_device_name(args.device),
        "compute_capability": list(capability),
        "torch_version": torch.__version__,
        "projection": dataclasses.asdict(projection),
        "tokens": args.tokens,
        "calls": args.calls,
        "plan": dataclasses.asdict(plan),
        "padded_shape": {
            "output_features": prepared["padded_n"],
            "input_features": prepared["padded_k"],
        },
        "weight_storage": {
            "bf16_logical_bytes": projection.bf16_weight_bytes,
            "marlin_physical_bytes": physical_marlin_bytes,
            "compression_ratio": projection.bf16_weight_bytes / physical_marlin_bytes,
        },
        "bf16_error": bf16_error,
        "marlin_quantization_error": marlin_quantization_error,
        "marlin_execution_error": marlin_execution_error,
        "candidate_graph_bitwise_deterministic": graph_bitwise_deterministic,
        "timing": summaries,
        "candidate_speedup": baseline_us / candidate_us,
        "projected_projection_savings_ms_per_decode_step": per_step_savings_ms,
        "decision_contract": {
            "status": "screening_only",
            "required_combined_trace_weighted_savings_ms_per_generated_token": 0.8,
            "note": (
                "Random-weight numerical error is not a model-quality result. "
                "A candidate must also pass the complete real-weight screen."
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
    """Parse the exact-shape Marlin benchmark command line."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--projection", choices=sorted(HYPERCONNECTION_PROJECTIONS), required=True
    )
    parser.add_argument("--tokens", type=int, choices=(1, 2), required=True)
    parser.add_argument("--bits", type=int, choices=(4, 8), required=True)
    parser.add_argument(
        "--group-size", type=int, choices=(-1, 32, 64, 128), required=True
    )
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
    MarlinExperimentPlan(args.bits, args.group_size).validate(
        HYPERCONNECTION_PROJECTIONS[args.projection]
    )
    return args


def main() -> None:
    """Run the benchmark and atomically publish its JSON result."""

    args = parse_args()
    result = run_marlin_benchmark(args)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
