# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark GGUF IQ1/IQ3/K-quant ops at DeepSeek V4 TP=4 serving shapes.

Timing-only harness: weights are deterministic random bytes with benign
scale words, matching the convention of benchmark_gguf_iq2_xxs.py. All
shapes mirror the production Unsloth UD-IQ1_* TP=4 geometry:

- routed gate/up per rank: K=4096 -> N=512 over 256 resident experts
- routed down per rank:    K=512  -> N=4096 over 256 resident experts
- shared gate/up:          K=4096 -> N=512   (Q5_K)
- shared down:             K=512  -> N=2048  (Q6_K)
"""

import argparse
import json

import numpy as np
import torch
import vllm._C_stable_libtorch  # noqa: F401  # Registers torch.ops._C GGUF ops.


def make_random_blocks(
    output_rows: int,
    input_columns: int,
    block_bytes: int,
    seed: int,
    scale_words: dict[int, bytes] | None = None,
) -> np.ndarray:
    """Deterministic random block bytes with fixed fp16 scale words."""
    block_count = input_columns // 256
    rng = np.random.default_rng(seed)
    packed = rng.integers(
        0, 256, size=(output_rows, block_count * block_bytes), dtype=np.uint8
    )
    if scale_words:
        for offset, pattern in scale_words.items():
            packed[:, offset : offset + len(pattern)] = np.frombuffer(
                pattern, dtype=np.uint8
            )
    return packed


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


def quantize_activations(activations: torch.Tensor) -> tuple[torch.Tensor, ...]:
    scales = torch.empty(
        activations.shape[0],
        activations.shape[1] // 32,
        device="cuda",
        dtype=torch.float16,
    )
    codes = torch.empty_like(activations, dtype=torch.int8)
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    return scales, codes


def bench_raw_matvec(
    format_name: str,
    weight_bytes_per_row: int,
    input_columns: int,
    output_rows: int,
    token_count: int,
    seed: int,
    iterations: int,
    warmup: int,
    scale_words: dict[int, bytes] | None = None,
) -> dict[str, float | int]:
    raw = make_random_blocks(
        output_rows, input_columns, weight_bytes_per_row, seed, scale_words
    )
    weights = torch.from_numpy(raw).cuda()
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = quantize_activations(activations)
    output = torch.empty(token_count, output_rows, device="cuda", dtype=torch.float32)
    op = getattr(torch.ops._C, f"gguf_{format_name}_q8_1_raw_matvec")
    operation = lambda: op(scales, codes, weights, output)  # noqa: E731
    milliseconds = time_cuda_operation(operation, iterations, warmup)
    graph = capture_cuda_operation(operation)
    graph_ms = time_cuda_operation(graph.replay, iterations, warmup)
    weight_bytes = output_rows * weight_bytes_per_row * (input_columns // 256)
    return {
        "op": f"{format_name}_raw_matvec",
        "token_count": token_count,
        "input_columns": input_columns,
        "output_rows": output_rows,
        "ms": milliseconds,
        "graph_ms": graph_ms,
        "weight_bytes": weight_bytes,
        "gbps": weight_bytes / (milliseconds * 1e6),
    }


def bench_indexed_gate_up(
    format_name: str,
    weight_bytes_per_row: int,
    input_columns: int,
    output_rows: int,
    expert_count: int,
    topk: int,
    token_count: int,
    seed: int,
    iterations: int,
    warmup: int,
    scale_words: dict[int, bytes] | None = None,
) -> dict[str, float | int]:
    gate = (
        torch.from_numpy(
            make_random_blocks(
                expert_count * output_rows,
                input_columns,
                weight_bytes_per_row,
                seed,
                scale_words,
            )
        )
        .reshape(expert_count, output_rows, -1)
        .cuda()
    )
    up = (
        torch.from_numpy(
            make_random_blocks(
                expert_count * output_rows,
                input_columns,
                weight_bytes_per_row,
                seed + 1,
                scale_words,
            )
        )
        .reshape(expert_count, output_rows, -1)
        .cuda()
    )
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = quantize_activations(activations)
    topk_ids = (
        torch.arange(token_count * topk, device="cuda", dtype=torch.int32)
        .remainder(expert_count)
        .reshape(token_count, topk)
    )
    gate_output = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    up_output = torch.empty_like(gate_output)
    op = getattr(torch.ops._C, f"gguf_{format_name}_q8_1_indexed_gate_up")
    operation = lambda: op(  # noqa: E731
        scales, codes, gate, up, topk_ids, gate_output, up_output
    )
    milliseconds = time_cuda_operation(operation, iterations, warmup)
    graph = capture_cuda_operation(operation)
    graph_ms = time_cuda_operation(graph.replay, iterations, warmup)
    one_matrix = (
        expert_count * output_rows * weight_bytes_per_row * (input_columns // 256)
    )
    logical_bytes = 2 * topk * one_matrix
    return {
        "op": f"{format_name}_indexed_gate_up",
        "token_count": token_count,
        "expert_count": expert_count,
        "topk": topk,
        "ms": milliseconds,
        "graph_ms": graph_ms,
        "logical_weight_bytes": logical_bytes,
        "gbps_logical": logical_bytes / (milliseconds * 1e6),
    }


def bench_grouped_gate_up(
    format_name: str,
    weight_bytes_per_row: int,
    input_columns: int,
    output_rows: int,
    expert_count: int,
    topk: int,
    token_count: int,
    seed: int,
    iterations: int,
    warmup: int,
    scale_words: dict[int, bytes] | None = None,
) -> dict[str, float | int]:
    import vllm._moe_C_stable_libtorch  # noqa: F401

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    gate = (
        torch.from_numpy(
            make_random_blocks(
                expert_count * output_rows,
                input_columns,
                weight_bytes_per_row,
                seed,
                scale_words,
            )
        )
        .reshape(expert_count, output_rows, -1)
        .cuda()
    )
    up = (
        torch.from_numpy(
            make_random_blocks(
                expert_count * output_rows,
                input_columns,
                weight_bytes_per_row,
                seed + 1,
                scale_words,
            )
        )
        .reshape(expert_count, output_rows, -1)
        .cuda()
    )
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = quantize_activations(activations)
    # Uniform routing: every expert receives token_count*topk/expert_count
    # assignments, the realistic serving case measured in prior M2 screens.
    assignments = token_count * topk
    topk_ids = (
        torch.arange(assignments, device="cuda", dtype=torch.int32)
        .remainder(expert_count)
        .reshape(token_count, topk)
    )
    schedule = moe_align_block_size(
        topk_ids=topk_ids, block_size=8, num_experts=expert_count
    )
    gate_output = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    up_output = torch.empty_like(gate_output)
    op = getattr(torch.ops._C, f"gguf_{format_name}_q8_1_grouped_gate_up")
    operation = lambda: op(  # noqa: E731
        scales, codes, gate, up, *schedule, gate_output, up_output, topk
    )
    milliseconds = time_cuda_operation(operation, iterations, warmup)
    graph = capture_cuda_operation(operation)
    graph_ms = time_cuda_operation(graph.replay, iterations, warmup)
    # Uniform routing touches every expert when assignments >= expert_count.
    touched_experts = min(expert_count, token_count * topk)
    logical_bytes = (
        touched_experts
        * 2
        * output_rows
        * weight_bytes_per_row
        * (input_columns // 256)
    )
    return {
        "op": f"{format_name}_grouped_gate_up",
        "token_count": token_count,
        "expert_count": expert_count,
        "topk": topk,
        "ms": milliseconds,
        "graph_ms": graph_ms,
        "touched_weight_bytes_uniform": logical_bytes,
        "gbps_touched": logical_bytes / (milliseconds * 1e6),
    }


def bench_indexed_down(
    input_columns: int,
    output_rows: int,
    expert_count: int,
    topk: int,
    token_count: int,
    seed: int,
    iterations: int,
    warmup: int,
) -> dict[str, float | int]:
    weights = (
        torch.from_numpy(
            make_random_blocks(expert_count * output_rows, input_columns, 98, seed)
        )
        .reshape(expert_count, output_rows, -1)
        .cuda()
    )
    activations = torch.randn(
        token_count * topk, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = quantize_activations(activations)
    topk_ids = (
        torch.arange(token_count * topk, device="cuda", dtype=torch.int32)
        .remainder(expert_count)
        .reshape(token_count, topk)
    )
    output = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    op = torch.ops._C.gguf_iq3_xxs_q8_1_indexed_down
    operation = lambda: op(scales, codes, weights, topk_ids, output)  # noqa: E731
    milliseconds = time_cuda_operation(operation, iterations, warmup)
    graph = capture_cuda_operation(operation)
    graph_ms = time_cuda_operation(graph.replay, iterations, warmup)
    one_matrix = expert_count * output_rows * 98 * (input_columns // 256)
    logical_bytes = topk * one_matrix
    return {
        "op": "iq3_xxs_indexed_down",
        "token_count": token_count,
        "expert_count": expert_count,
        "topk": topk,
        "ms": milliseconds,
        "graph_ms": graph_ms,
        "logical_weight_bytes": logical_bytes,
        "gbps_logical": logical_bytes / (milliseconds * 1e6),
    }


def bench_grouped_matmul(
    format_name: str,
    weight_bytes_per_row: int,
    input_columns: int,
    output_rows: int,
    token_count: int,
    seed: int,
    iterations: int,
    warmup: int,
    scale_words: dict[int, bytes] | None = None,
) -> dict[str, float | int]:
    weights = torch.from_numpy(
        make_random_blocks(
            output_rows, input_columns, weight_bytes_per_row, seed, scale_words
        )
    ).cuda()
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = quantize_activations(activations)
    output = torch.empty(token_count, output_rows, device="cuda", dtype=torch.float32)
    op = getattr(torch.ops._C, f"gguf_{format_name}_q8_1_grouped_matmul")
    operation = lambda: op(scales, codes, weights, output)  # noqa: E731
    milliseconds = time_cuda_operation(operation, iterations, warmup)
    graph = capture_cuda_operation(operation)
    graph_ms = time_cuda_operation(graph.replay, iterations, warmup)
    weight_bytes = output_rows * weight_bytes_per_row * (input_columns // 256)
    return {
        "op": f"{format_name}_grouped_matmul",
        "token_count": token_count,
        "ms": milliseconds,
        "graph_ms": graph_ms,
        "weight_bytes": weight_bytes,
        "gbps": weight_bytes / (milliseconds * 1e6),
    }


IQ1_S_SCALE_WORDS = {0: np.float16(0.01).tobytes()}
IQ1_M_SCALE_WORDS = {offset: np.float16(0.01).tobytes() for offset in range(48, 56, 2)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--output")
    args = parser.parse_args()

    rows: list[dict[str, float | int]] = []

    # Decode (M=1): routed gate/up raw + indexed, routed down indexed,
    # shared-expert and attention K-quant raw matvecs.
    for format_index, (format_name, block_bytes, scale_words) in enumerate(
        (
            ("iq1_s", 50, IQ1_S_SCALE_WORDS),
            ("iq1_m", 56, IQ1_M_SCALE_WORDS),
        )
    ):
        seed = 910000 + format_index * 1000
        rows.append(
            bench_raw_matvec(
                format_name,
                block_bytes,
                4096,
                512,
                1,
                seed,
                args.iterations,
                args.warmup,
                scale_words,
            )
        )
        rows.append(
            bench_indexed_gate_up(
                format_name,
                block_bytes,
                4096,
                512,
                256,
                6,
                1,
                seed + 10,
                args.iterations,
                args.warmup,
                scale_words,
            )
        )

    rows.append(
        bench_indexed_down(512, 4096, 256, 6, 1, 424242, args.iterations, args.warmup)
    )
    for format_index, (format_name, block_bytes) in enumerate(
        (("q4_k", 144), ("q5_k", 176))
    ):
        rows.append(
            bench_raw_matvec(
                format_name,
                block_bytes,
                4096,
                512,
                1,
                920000 + format_index,
                args.iterations,
                args.warmup,
            )
        )
    rows.append(
        bench_raw_matvec(
            "q6_k", 210, 512, 2048, 1, 4242424, args.iterations, args.warmup
        )
    )

    # Prefill screen (M=256 uniform routing): grouped variants.
    for format_index, (format_name, block_bytes, scale_words) in enumerate(
        (
            ("iq1_s", 50, IQ1_S_SCALE_WORDS),
            ("iq1_m", 56, IQ1_M_SCALE_WORDS),
        )
    ):
        seed = 930000 + format_index * 1000
        rows.append(
            bench_grouped_gate_up(
                format_name,
                block_bytes,
                4096,
                512,
                256,
                6,
                256,
                seed,
                args.iterations,
                args.warmup,
                scale_words,
            )
        )
    for format_index, (format_name, block_bytes) in enumerate(
        (("q4_k", 144), ("q5_k", 176), ("q6_k", 210))
    ):
        rows.append(
            bench_grouped_matmul(
                format_name,
                block_bytes,
                4096,
                512,
                256,
                940000 + format_index,
                args.iterations,
                args.warmup,
            )
        )

    results = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
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
