# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark TP4 DeepSeek V4 Q8_0 Marlin-diagonal wo_a execution."""

import argparse
import json
from functools import partial

import torch

from vllm.model_executor.layers.quantization.gguf_dsv4.q8_0_marlin import (
    apply_gguf_q8_0_marlin,
    prepare_gguf_q8_0_marlin,
)
from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
    _apply_dsv4_wo_a_marlin_diagonal,
)


def make_q8_0_weights(output_rows: int, input_columns: int) -> torch.Tensor:
    """Create deterministic finite Q8_0 blocks for a timing-only fixture."""
    generator = torch.Generator().manual_seed(20260817)
    scales = (
        torch.rand(
            output_rows,
            input_columns // 32,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
        + 0.001
    ).to(torch.float16)
    codes = torch.randint(
        -127,
        128,
        (output_rows, input_columns),
        generator=generator,
        dtype=torch.int8,
    )
    blocks = torch.empty(output_rows, input_columns // 32, 34, dtype=torch.uint8)
    blocks[:, :, :2] = (
        scales.contiguous()
        .view(torch.uint8)
        .reshape(output_rows, input_columns // 32, 2)
    )
    blocks[:, :, 2:] = codes.view(output_rows, input_columns // 32, 32).view(
        torch.uint8
    )
    return blocks.reshape(output_rows, -1)


def time_operation(operation, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=2500)
    parser.add_argument("--token-counts", default="1,2,4")
    parser.add_argument("--output")
    args = parser.parse_args()
    token_counts = tuple(int(value) for value in args.token_counts.split(","))
    if not token_counts or any(value <= 0 for value in token_counts):
        raise ValueError("Q8 wo_a token counts must be positive integers")

    local_groups, output_rank = 2, 1024
    input_columns = 4096
    output_rows = local_groups * output_rank
    raw = make_q8_0_weights(output_rows, input_columns)
    prepared = prepare_gguf_q8_0_marlin(
        raw.cuda(), input_columns=input_columns, scale_dtype=torch.bfloat16
    )

    class PreparedWoA(torch.nn.Module):
        def forward(self, inputs):
            return apply_gguf_q8_0_marlin(inputs, prepared)

    wo_a = PreparedWoA()
    results = []
    for token_count in token_counts:
        inputs = torch.randn(
            token_count,
            local_groups,
            input_columns,
            device="cuda",
            dtype=torch.bfloat16,
        )

        eager_operation = partial(
            _apply_dsv4_wo_a_marlin_diagonal,
            inputs,
            wo_a,
            n_local_groups=local_groups,
            o_lora_rank=output_rank,
        )

        graph_output = torch.empty(
            token_count,
            local_groups,
            output_rank,
            device="cuda",
            dtype=torch.bfloat16,
        )
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output.copy_(
                _apply_dsv4_wo_a_marlin_diagonal(
                    inputs,
                    wo_a,
                    n_local_groups=local_groups,
                    o_lora_rank=output_rank,
                )
            )
        eager_ms = time_operation(eager_operation, args.iterations, args.warmup)
        graph_ms = time_operation(graph.replay, args.iterations, args.warmup)
        results.append(
            {
                "tokens": token_count,
                "eager_ms": eager_ms,
                "graph_ms": graph_ms,
            }
        )

    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {
            "local_groups": local_groups,
            "output_rank": output_rank,
            "K": input_columns,
            "N": output_rows,
        },
        "iterations": args.iterations,
        "warmup": args.warmup,
        "token_counts": token_counts,
        "raw_weight_bytes": raw.numel(),
        "prepared_weight_scale_bytes": prepared.weight.nbytes + prepared.scales.nbytes,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
