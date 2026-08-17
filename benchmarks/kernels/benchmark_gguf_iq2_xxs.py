# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark GGUF IQ2_XXS raw vs aligned matvec at DeepSeek V4 shapes."""

import argparse
import json

import numpy as np
import torch
import vllm._C_stable_libtorch  # noqa: F401  # Registers torch.ops._C GGUF ops.


def make_packed_iq2_xxs(output_rows: int, input_columns: int) -> np.ndarray:
    """Build deterministic finite IQ2_XXS bytes for one benchmark matrix."""
    return make_seeded_packed_iq2_xxs(output_rows, input_columns, 20260817)


def make_seeded_packed_iq2_xxs(
    output_rows: int, input_columns: int, seed: int
) -> np.ndarray:
    """Build finite synthetic IQ2_XXS bytes with a named random seed."""
    block_count = input_columns // 256
    rng = np.random.default_rng(seed)
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
    q8_scales = torch.empty(
        (token_count, input_columns // 32), device="cuda", dtype=torch.float16
    )
    q8_codes = torch.empty_like(activations, dtype=torch.int8)
    q8_raw_output = torch.empty_like(raw_output)
    q8_aligned_output = torch.empty_like(raw_output)

    def raw_operation() -> None:
        torch.ops._C.gguf_iq2_xxs_raw_matvec(activations, packed_weights, raw_output)

    def aligned_operation() -> None:
        torch.ops._C.gguf_iq2_xxs_aligned_matvec(
            activations, *aligned_streams, aligned_output
        )

    def quantize_operation() -> None:
        torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, q8_scales, q8_codes)

    def q8_raw_operation() -> None:
        torch.ops._C.gguf_iq2_xxs_q8_1_raw_matvec(
            q8_scales, q8_codes, packed_weights, q8_raw_output
        )

    def q8_aligned_operation() -> None:
        torch.ops._C.gguf_iq2_xxs_q8_1_aligned_matvec(
            q8_scales, q8_codes, *aligned_streams, q8_aligned_output
        )

    raw_ms = time_cuda_operation(raw_operation, iterations, warmup)
    aligned_ms = time_cuda_operation(aligned_operation, iterations, warmup)
    quantize_ms = time_cuda_operation(quantize_operation, iterations, warmup)
    quantize_operation()
    q8_raw_ms = time_cuda_operation(q8_raw_operation, iterations, warmup)
    q8_aligned_ms = time_cuda_operation(q8_aligned_operation, iterations, warmup)
    raw_graph = capture_cuda_operation(raw_operation)
    aligned_graph = capture_cuda_operation(aligned_operation)
    q8_pipeline_graph = capture_cuda_operation(
        lambda: (quantize_operation(), q8_aligned_operation())
    )
    raw_graph_ms = time_cuda_operation(raw_graph.replay, iterations, warmup)
    aligned_graph_ms = time_cuda_operation(aligned_graph.replay, iterations, warmup)
    q8_pipeline_graph_ms = time_cuda_operation(
        q8_pipeline_graph.replay, iterations, warmup
    )
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
        "q8_quantize_ms": quantize_ms,
        "q8_raw_ms": q8_raw_ms,
        "q8_aligned_ms": q8_aligned_ms,
        "q8_aligned_over_raw": q8_aligned_ms / q8_raw_ms,
        "q8_raw_gbps": weight_bytes / (q8_raw_ms * 1e6),
        "q8_aligned_gbps": weight_bytes / (q8_aligned_ms * 1e6),
        "q8_pipeline_graph_ms": q8_pipeline_graph_ms,
        "q8_gate_up_top6_effective_ms": quantize_ms + 12 * q8_aligned_ms,
    }


def benchmark_indexed_gate_up(
    token_count: int,
    input_columns: int,
    output_rows: int,
    gate_weights: torch.Tensor,
    up_weights: torch.Tensor,
    iterations: int,
    warmup: int,
) -> dict[str, float | int]:
    """Benchmark one top-6 gate+up launch over indexed resident experts."""
    topk = 6
    activations = torch.randn(
        (token_count, input_columns), device="cuda", dtype=torch.bfloat16
    )
    q8_scales = torch.empty(
        (token_count, input_columns // 32), device="cuda", dtype=torch.float16
    )
    q8_codes = torch.empty_like(activations, dtype=torch.int8)
    topk_ids = (
        torch.arange(token_count * topk, device="cuda", dtype=torch.int32)
        .remainder(gate_weights.shape[0])
        .reshape(token_count, topk)
    )
    gate_output = torch.empty(
        (token_count, topk, output_rows), device="cuda", dtype=torch.float32
    )
    up_output = torch.empty_like(gate_output)

    def quantize_operation() -> None:
        torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, q8_scales, q8_codes)

    def indexed_operation() -> None:
        torch.ops._C.gguf_iq2_xxs_q8_1_indexed_gate_up(
            q8_scales,
            q8_codes,
            gate_weights,
            up_weights,
            topk_ids,
            gate_output,
            up_output,
        )

    quantize_ms = time_cuda_operation(quantize_operation, iterations, warmup)
    quantize_operation()
    indexed_ms = time_cuda_operation(indexed_operation, iterations, warmup)
    pipeline_graph = capture_cuda_operation(
        lambda: (quantize_operation(), indexed_operation())
    )
    graph_ms = time_cuda_operation(pipeline_graph.replay, iterations, warmup)
    one_matrix_bytes = output_rows * (input_columns // 256) * 66
    logical_bytes = 2 * topk * one_matrix_bytes
    return {
        "indexed_gate_up_ms": indexed_ms,
        "indexed_gate_up_logical_bytes": logical_bytes,
        "indexed_gate_up_logical_gbps": logical_bytes / (indexed_ms * 1e6),
        "indexed_quantize_ms": quantize_ms,
        "indexed_pipeline_graph_ms": graph_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--input-columns", type=int, default=4096)
    parser.add_argument("--output-rows", type=int, default=2048)
    parser.add_argument("--output")
    args = parser.parse_args()

    input_columns, output_rows = args.input_columns, args.output_rows
    packed = make_packed_iq2_xxs(output_rows, input_columns)
    aligned = repack_iq2_xxs_aligned(packed, input_columns)
    packed_gpu = torch.from_numpy(packed).cuda()
    aligned_gpu = tuple(torch.from_numpy(stream).cuda() for stream in aligned)
    expert_count = 8
    row_bytes = packed.shape[1]
    gate = make_seeded_packed_iq2_xxs(
        expert_count * output_rows, input_columns, 20260818
    ).reshape(expert_count, output_rows, row_bytes)
    up = make_seeded_packed_iq2_xxs(
        expert_count * output_rows, input_columns, 20260819
    ).reshape(expert_count, output_rows, row_bytes)
    gate_gpu, up_gpu = torch.from_numpy(gate).cuda(), torch.from_numpy(up).cuda()

    rows = []
    for token_count in (1, 2, 4):
        row = benchmark_token_count(
            token_count,
            input_columns,
            output_rows,
            packed_gpu,
            aligned_gpu,
            args.iterations,
            args.warmup,
        )
        row.update(
            benchmark_indexed_gate_up(
                token_count,
                input_columns,
                output_rows,
                gate_gpu,
                up_gpu,
                args.iterations,
                args.warmup,
            )
        )
        rows.append(row)

    results = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "input_columns": input_columns,
        "output_rows": output_rows,
        "iterations": args.iterations,
        "warmup": args.warmup,
        "results": rows,
    }
    rendered = json.dumps(results, indent=2)
    print(rendered)
    if args.output:
        with open(args.output, "w", encoding="utf-8") as output_file:
            output_file.write(rendered + "\n")


if __name__ == "__main__":
    main()
