# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Screen native indexed GGUF experts across DeepSeek V4 prefill token counts."""

import argparse
import json
from functools import partial

import torch
import vllm._C_stable_libtorch  # noqa: F401

from benchmarks.kernels.benchmark_gguf_iq2_xxs import make_seeded_packed_iq2_xxs
from benchmarks.kernels.benchmark_gguf_q2_k import make_q2_k_weights

PREFILL_TOKEN_COUNTS = (16, 32, 64, 128, 256)
EXPERT_COUNT = 256
TOPK = 6


def make_topk_ids(token_count: int, routing: str) -> torch.Tensor:
    """Return deterministic valid expert IDs for one routing-reuse boundary."""
    if routing == "uniform":
        ids = torch.arange(token_count * TOPK, dtype=torch.int32).remainder(
            EXPERT_COUNT
        )
    elif routing == "concentrated":
        ids = torch.arange(TOPK, dtype=torch.int32).repeat(token_count)
    else:
        raise ValueError(f"Unknown GGUF prefill routing pattern: {routing}")
    return ids.reshape(token_count, TOPK).cuda()


def run_indexed_gate_up_pipeline(
    activations,
    scales,
    codes,
    gate_weights,
    up_weights,
    topk_ids,
    gate_output,
    up_output,
) -> None:
    """Quantize one token batch and run indexed IQ2_XXS gate plus up."""
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    torch.ops._C.gguf_iq2_xxs_q8_1_indexed_gate_up(
        scales,
        codes,
        gate_weights,
        up_weights,
        topk_ids,
        gate_output,
        up_output,
    )


def run_grouped_gate_up_pipeline(
    activations,
    scales,
    codes,
    gate_weights,
    up_weights,
    sorted_ids,
    expert_ids,
    num_tokens_padded,
    gate_output,
    up_output,
    topk,
) -> None:
    """Quantize one token batch and run block-8 grouped IQ2 gate plus up."""
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    torch.ops._C.gguf_iq2_xxs_q8_1_grouped_gate_up(
        scales,
        codes,
        gate_weights,
        up_weights,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        gate_output,
        up_output,
        topk,
    )


def run_grouped_down_pipeline(
    activations,
    scales,
    codes,
    down_weights,
    sorted_ids,
    expert_ids,
    num_tokens_padded,
    output,
) -> None:
    """Quantize routed activations and run block-8 grouped Q2_K down."""
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    torch.ops._C.gguf_q2_k_q8_1_grouped_down(
        scales,
        codes,
        down_weights,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        output,
    )


def run_indexed_down_pipeline(
    activations, scales, codes, down_weights, topk_ids, output
) -> None:
    """Quantize routed activations and run indexed Q2_K down."""
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    torch.ops._C.gguf_q2_k_q8_1_indexed_down(
        scales, codes, down_weights, topk_ids, output
    )


def capture_pipeline(operation) -> torch.cuda.CUDAGraph:
    operation()
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        operation()
    return graph


def time_cuda_operation(operation, iterations: int, warmup: int) -> float:
    """Return mean CUDA-event time for a possibly allocating operation."""
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


def time_graph(graph: torch.cuda.CUDAGraph, iterations: int, warmup: int) -> float:
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
    import vllm._moe_C_stable_libtorch  # noqa: F401

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    parser = argparse.ArgumentParser()
    parser.add_argument("--iterations", type=int, default=500)
    parser.add_argument("--warmup", type=int, default=250)
    parser.add_argument("--output")
    args = parser.parse_args()

    gate_rows, gate_columns = 512, 4096
    gate_row_bytes = gate_columns // 256 * 66
    gate = make_seeded_packed_iq2_xxs(
        EXPERT_COUNT * gate_rows, gate_columns, 20260821
    ).reshape(EXPERT_COUNT, gate_rows, gate_row_bytes)
    up = make_seeded_packed_iq2_xxs(
        EXPERT_COUNT * gate_rows, gate_columns, 20260822
    ).reshape(EXPERT_COUNT, gate_rows, gate_row_bytes)
    gate_weights = torch.from_numpy(gate).cuda()
    up_weights = torch.from_numpy(up).cuda()

    down_rows, down_columns = 4096, 512
    down_weights = torch.from_numpy(
        make_q2_k_weights(EXPERT_COUNT, down_rows, down_columns)
    ).cuda()

    results = []
    for routing in ("uniform", "concentrated"):
        for token_count in PREFILL_TOKEN_COUNTS:
            topk_ids = make_topk_ids(token_count, routing)
            gate_activations = torch.randn(
                token_count,
                gate_columns,
                device="cuda",
                dtype=torch.bfloat16,
            )
            gate_scales = torch.empty(
                token_count,
                gate_columns // 32,
                device="cuda",
                dtype=torch.float16,
            )
            gate_codes = torch.empty_like(gate_activations, dtype=torch.int8)
            gate_output = torch.empty(
                token_count,
                TOPK,
                gate_rows,
                device="cuda",
                dtype=torch.float32,
            )
            up_output = torch.empty_like(gate_output)

            down_activations = torch.randn(
                token_count * TOPK,
                down_columns,
                device="cuda",
                dtype=torch.bfloat16,
            )
            down_scales = torch.empty(
                token_count * TOPK,
                down_columns // 32,
                device="cuda",
                dtype=torch.float16,
            )
            down_codes = torch.empty_like(down_activations, dtype=torch.int8)
            down_output = torch.empty(
                token_count,
                TOPK,
                down_rows,
                device="cuda",
                dtype=torch.float32,
            )

            gate_up_pipeline = partial(
                run_indexed_gate_up_pipeline,
                gate_activations,
                gate_scales,
                gate_codes,
                gate_weights,
                up_weights,
                topk_ids,
                gate_output,
                up_output,
            )
            down_pipeline = partial(
                run_indexed_down_pipeline,
                down_activations,
                down_scales,
                down_codes,
                down_weights,
                topk_ids,
                down_output,
            )

            sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
                topk_ids=topk_ids,
                block_size=8,
                num_experts=EXPERT_COUNT,
            )
            grouped_gate_output = torch.empty_like(gate_output)
            grouped_up_output = torch.empty_like(up_output)
            grouped_down_output = torch.empty_like(down_output)

            alignment_operation = partial(
                moe_align_block_size,
                topk_ids=topk_ids,
                block_size=8,
                num_experts=EXPERT_COUNT,
            )

            grouped_gate_up_pipeline = partial(
                run_grouped_gate_up_pipeline,
                gate_activations,
                gate_scales,
                gate_codes,
                gate_weights,
                up_weights,
                sorted_ids,
                expert_ids,
                num_tokens_padded,
                grouped_gate_output,
                grouped_up_output,
                TOPK,
            )
            grouped_down_pipeline = partial(
                run_grouped_down_pipeline,
                down_activations,
                down_scales,
                down_codes,
                down_weights,
                sorted_ids,
                expert_ids,
                num_tokens_padded,
                grouped_down_output,
            )
            gate_graph = capture_pipeline(gate_up_pipeline)
            grouped_gate_graph = capture_pipeline(grouped_gate_up_pipeline)
            down_graph = capture_pipeline(down_pipeline)
            grouped_down_graph = capture_pipeline(grouped_down_pipeline)
            alignment_ms = time_cuda_operation(
                alignment_operation, args.iterations, args.warmup
            )
            gate_ms = time_graph(gate_graph, args.iterations, args.warmup)
            grouped_gate_ms = time_graph(
                grouped_gate_graph, args.iterations, args.warmup
            )
            down_ms = time_graph(down_graph, args.iterations, args.warmup)
            grouped_down_ms = time_graph(
                grouped_down_graph, args.iterations, args.warmup
            )
            results.append(
                {
                    "routing": routing,
                    "tokens": token_count,
                    "alignment_ms": alignment_ms,
                    "gate_up_graph_ms": gate_ms,
                    "grouped_gate_up_graph_ms": grouped_gate_ms,
                    "grouped_gate_up_speedup": gate_ms / grouped_gate_ms,
                    "down_graph_ms": down_ms,
                    "grouped_down_graph_ms": grouped_down_ms,
                    "grouped_down_speedup": down_ms / grouped_down_ms,
                    "expert_graph_ms": gate_ms + down_ms,
                    "expert_ms_per_token": (gate_ms + down_ms) / token_count,
                    "grouped_expert_graph_ms": grouped_gate_ms + grouped_down_ms,
                    "grouped_expert_with_alignment_ms": grouped_gate_ms
                    + grouped_down_ms
                    + alignment_ms,
                    "grouped_expert_ms_per_token": (
                        grouped_gate_ms + grouped_down_ms + alignment_ms
                    )
                    / token_count,
                }
            )

    report = {
        "schema_version": 1,
        "device": torch.cuda.get_device_name(),
        "capability": list(torch.cuda.get_device_capability()),
        "expert_count": EXPERT_COUNT,
        "topk": TOPK,
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
