# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark indexed GGUF Q2_K down projection at the DeepSeek V4 TP4 shape."""

import argparse
import json

import numpy as np
import torch
import vllm._C_stable_libtorch  # noqa: F401


def make_q2_k_weights(experts: int, rows: int, columns: int) -> np.ndarray:
    """Build finite synthetic Q2_K bytes without duplicating decode logic."""
    rng = np.random.default_rng(20260820)
    block_count = columns // 256
    packed = rng.integers(0, 256, (experts, rows, block_count * 84), dtype=np.uint8)
    scale = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
    min_scale = np.frombuffer(np.float16(0.005).tobytes(), dtype=np.uint8)
    for block in range(block_count):
        packed[:, :, block * 84 + 80 : block * 84 + 82] = scale
        packed[:, :, block * 84 + 82 : block * 84 + 84] = min_scale
    return packed


def time_operation(operation, iterations: int, warmup: int) -> float:
    for _ in range(warmup):
        operation()
    torch.accelerator.synchronize()
    start, end = (
        torch.cuda.Event(enable_timing=True),
        torch.cuda.Event(enable_timing=True),
    )
    start.record()
    for _ in range(iterations):
        operation()
    end.record()
    torch.accelerator.synchronize()
    return start.elapsed_time(end) / iterations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=10000)
    parser.add_argument("--warmup", type=int, default=5000)
    parser.add_argument("--output")
    args = parser.parse_args()
    token_count, topk, experts, rows, columns = 1, 6, 8, 4096, 512
    weights = torch.from_numpy(make_q2_k_weights(experts, rows, columns)).cuda()
    activations = torch.randn((topk, columns), device="cuda", dtype=torch.bfloat16)
    scales = torch.empty((topk, columns // 32), device="cuda", dtype=torch.float16)
    codes = torch.empty_like(activations, dtype=torch.int8)
    topk_ids = torch.arange(topk, device="cuda", dtype=torch.int32).reshape(1, topk)
    output = torch.empty((token_count, topk, rows), device="cuda", dtype=torch.float32)

    def quantize() -> None:
        torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)

    def down() -> None:
        torch.ops._C.gguf_q2_k_q8_1_indexed_down(
            scales, codes, weights, topk_ids, output
        )

    quantize_ms = time_operation(quantize, args.iterations, args.warmup)
    quantize()
    down_ms = time_operation(down, args.iterations, args.warmup)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        quantize()
        down()
    graph_ms = time_operation(graph.replay, args.iterations, args.warmup)
    logical_bytes = topk * rows * (columns // 256) * 84
    result = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "shape": {"tokens": token_count, "topk": topk, "K": columns, "N": rows},
        "iterations": args.iterations,
        "warmup": args.warmup,
        "logical_bytes": logical_bytes,
        "down_ms": down_ms,
        "logical_gbps": logical_bytes / (down_ms * 1e6),
        "quantize_ms": quantize_ms,
        "pipeline_graph_ms": graph_ms,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
