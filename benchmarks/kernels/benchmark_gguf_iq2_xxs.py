# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark GGUF IQ2_XXS raw vs aligned matvec at DeepSeek V4 shapes."""

import argparse
import json

import numpy as np
import torch
import vllm._C_stable_libtorch  # noqa: F401  # Registers torch.ops._C GGUF ops.


def make_packed_iq2_xxs(output_rows: int, input_columns: int) -> np.ndarray:
    """Build finite synthetic IQ2_XXS bytes without duplicating decode logic."""
    block_count = input_columns // 256
    rng = np.random.default_rng(20260817)
    packed = rng.integers(0, 256, size=(output_rows, block_count * 66), dtype=np.uint8)
    scale_bytes = np.frombuffer(np.float16(0.01).tobytes(), dtype=np.uint8)
    for block_index in range(block_count):
        packed[:, block_index * 66 : block_index * 66 + 2] = scale_bytes
    return packed


def repack_iq2_xxs_aligned(
    packed: np.ndarray, input_columns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Split raw blocks into the byte-neutral scale, grid, and sign streams."""
    output_rows = packed.shape[0]
    block_count = input_columns // 256
    scales = np.empty((output_rows, block_count, 2), dtype=np.uint8)
    grid_bytes = np.empty((output_rows, block_count, 8, 4), dtype=np.uint8)
    scale_sign_bytes = np.empty_like(grid_bytes)
    for block_index in range(block_count):
        block_offset = block_index * 66
        scales[:, block_index] = packed[:, block_offset : block_offset + 2]
        groups = packed[:, block_offset + 2 : block_offset + 66].reshape(
            output_rows, 8, 8
        )
        grid_bytes[:, block_index] = groups[:, :, :4]
        scale_sign_bytes[:, block_index] = groups[:, :, 4:]
    return scales, grid_bytes, scale_sign_bytes


def time_cuda_operation(operation, iterations: int, warmup: int) -> float:
    """Return mean CUDA-event milliseconds for repeated operation calls."""
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


def capture_cuda_operation(operation) -> torch.cuda.CUDAGraph:
    """Capture one operation after its allocation/JIT state is warm."""
    operation()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    return graph


def benchmark_token_count(
    token_count: int,
    input_columns: int,
    output_rows: int,
    packed_weights: torch.Tensor,
    aligned_streams: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
    iterations: int,
    warmup: int,
) -> dict[str, float | int]:
    """Benchmark raw and aligned kernels for one token count."""
    activations = torch.randn(
        (token_count, input_columns), device="cuda", dtype=torch.bfloat16
    )
    raw_output = torch.empty(
        (token_count, output_rows), device="cuda", dtype=torch.float32
    )
    aligned_output = torch.empty_like(raw_output)

    def raw_operation() -> None:
        torch.ops._C.gguf_iq2_xxs_raw_matvec(activations, packed_weights, raw_output)

    def aligned_operation() -> None:
        torch.ops._C.gguf_iq2_xxs_aligned_matvec(
            activations, *aligned_streams, aligned_output
        )

    raw_ms = time_cuda_operation(raw_operation, iterations, warmup)
    aligned_ms = time_cuda_operation(aligned_operation, iterations, warmup)
    raw_graph = capture_cuda_operation(raw_operation)
    aligned_graph = capture_cuda_operation(aligned_operation)
    raw_graph_ms = time_cuda_operation(raw_graph.replay, iterations, warmup)
    aligned_graph_ms = time_cuda_operation(aligned_graph.replay, iterations, warmup)
    weight_bytes = output_rows * (input_columns // 256) * 66
    return {
        "token_count": token_count,
        "weight_bytes": weight_bytes,
        "raw_ms": raw_ms,
        "aligned_ms": aligned_ms,
        "aligned_over_raw": aligned_ms / raw_ms,
        "raw_graph_ms": raw_graph_ms,
        "aligned_graph_ms": aligned_graph_ms,
        "aligned_graph_over_raw_graph": aligned_graph_ms / raw_graph_ms,
        "raw_gbps": weight_bytes / (raw_ms * 1e6),
        "aligned_gbps": weight_bytes / (aligned_ms * 1e6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--output")
    args = parser.parse_args()

    input_columns, output_rows = 4096, 2048
    packed = make_packed_iq2_xxs(output_rows, input_columns)
    aligned = repack_iq2_xxs_aligned(packed, input_columns)
    packed_gpu = torch.from_numpy(packed).cuda()
    aligned_gpu = tuple(torch.from_numpy(stream).cuda() for stream in aligned)

    results = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "input_columns": input_columns,
        "output_rows": output_rows,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": [
            benchmark_token_count(
                token_count,
                input_columns,
                output_rows,
                packed_gpu,
                aligned_gpu,
                args.iterations,
                args.warmup,
            )
            for token_count in (1, 2, 4)
        ],
    }
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
