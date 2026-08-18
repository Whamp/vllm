# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark rank-local DeepSeek V4 dense Q8_0 Marlin projection shapes."""

import argparse
import gc
import json
from dataclasses import asdict, dataclass

import torch

from vllm.model_executor.layers.quantization.gguf_dsv4.q8_0_marlin import (
    apply_gguf_q8_0_marlin,
    prepare_gguf_q8_0_marlin,
)


@dataclass(frozen=True)
class DenseQ8Shape:
    name: str
    output_rows: int
    input_columns: int
    executions_per_layer: int


DECODE_DENSE_Q8_SHAPES = (
    DenseQ8Shape("fused_wqa_wkv", 1536, 4096, 1),
    DenseQ8Shape("wq_b", 8192, 1024, 1),
    DenseQ8Shape("wo_b", 4096, 2048, 1),
    DenseQ8Shape("shared_gate_up", 1024, 4096, 1),
    DenseQ8Shape("shared_down", 4096, 512, 1),
    DenseQ8Shape("lm_head", 32320, 4096, 0),
)


def make_q8_0_weights(shape: DenseQ8Shape) -> torch.Tensor:
    """Create deterministic finite Q8_0 blocks for timing only."""
    generator = torch.Generator().manual_seed(20260817 + shape.output_rows)
    block_count = shape.input_columns // 32
    scales = (
        torch.rand(
            shape.output_rows,
            block_count,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
        + 0.001
    ).to(torch.float16)
    codes = torch.randint(
        -127,
        128,
        (shape.output_rows, shape.input_columns),
        generator=generator,
        dtype=torch.int8,
    )
    blocks = torch.empty(shape.output_rows, block_count, 34, dtype=torch.uint8)
    blocks[:, :, :2] = (
        scales.contiguous().view(torch.uint8).reshape(shape.output_rows, block_count, 2)
    )
    blocks[:, :, 2:] = codes.view(shape.output_rows, block_count, 32).view(torch.uint8)
    return blocks.reshape(shape.output_rows, -1)


def time_graph_replay(
    graph: torch.cuda.CUDAGraph, iterations: int, warmup: int
) -> float:
    for _ in range(warmup):
        graph.replay()
    torch.accelerator.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iterations):
        graph.replay()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=5000)
    parser.add_argument("--warmup", type=int, default=2500)
    parser.add_argument("--output")
    args = parser.parse_args()

    results = []
    for shape in DECODE_DENSE_Q8_SHAPES:
        raw = make_q8_0_weights(shape)
        prepared = prepare_gguf_q8_0_marlin(
            raw.cuda(),
            input_columns=shape.input_columns,
            scale_dtype=torch.bfloat16,
        )
        token_results = []
        for token_count in (1, 2, 4):
            inputs = torch.randn(
                token_count,
                shape.input_columns,
                device="cuda",
                dtype=torch.bfloat16,
            )
            output = torch.empty(
                token_count,
                shape.output_rows,
                device="cuda",
                dtype=torch.bfloat16,
            )
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                output.copy_(apply_gguf_q8_0_marlin(inputs, prepared))
            graph_ms = time_graph_replay(graph, args.iterations, args.warmup)
            token_results.append({"tokens": token_count, "graph_ms": graph_ms})
        results.append(
            {
                "shape": asdict(shape),
                "raw_weight_bytes": raw.numel(),
                "prepared_weight_scale_bytes": prepared.weight.nbytes
                + prepared.scales.nbytes,
                "tokens": token_results,
            }
        )
        del graph, inputs, output, prepared, raw
        gc.collect()

    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": results,
    }
    rendered = json.dumps(report, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
