# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Benchmark the native NVFP4 PLE gather against the production Torch path."""

import argparse
import json
import statistics
import time
from collections.abc import Callable
from pathlib import Path

import torch

from vllm.v1.ple_offload.nvfp4_native_gather import NvFp4PleNativeGather


def _measure(operation: Callable[[], None], iterations: int) -> dict[str, float]:
    samples = []
    for _ in range(iterations):
        started = time.perf_counter_ns()
        operation()
        samples.append((time.perf_counter_ns() - started) / 1_000_000)
    ordered = sorted(samples)
    return {
        "mean_ms": statistics.mean(samples),
        "p50_ms": statistics.median(samples),
        "p95_ms": ordered[int(0.95 * len(ordered)) - 1],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--row-count", type=int, default=16)
    parser.add_argument("--iterations", type=int, default=1000)
    args = parser.parse_args()
    if args.row_count <= 0 or args.iterations <= 0:
        parser.error("row count and iterations must be positive")

    torch.set_num_threads(1)
    generator = torch.Generator().manual_seed(20260830)
    shard_count = 128
    rows_per_shard = 64
    width = 160
    code_shards = tuple(
        torch.randint(
            0,
            256,
            (rows_per_shard, width // 2),
            dtype=torch.uint8,
            generator=generator,
        )
        for _ in range(shard_count)
    )
    scale_shards = tuple(
        torch.randn(
            (rows_per_shard, width // 16),
            generator=generator,
        ).to(torch.float8_e4m3fn)
        for _ in range(shard_count)
    )
    outer_scales = tuple(
        float(torch.rand((), generator=generator) * 0.75 + 0.125)
        for _ in range(shard_count)
    )
    row_ids = torch.randint(
        0,
        shard_count * rows_per_shard,
        (args.row_count,),
        generator=generator,
    )
    output = torch.empty((args.row_count, width), dtype=torch.bfloat16)
    magnitudes = [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
    dequant_lut = torch.tensor(magnitudes + [-value for value in magnitudes])

    def torch_gather() -> None:
        shard_ids = row_ids // rows_per_shard
        local_ids = row_ids - shard_ids * rows_per_shard
        order = torch.argsort(shard_ids)
        sorted_shards = shard_ids[order]
        sorted_local_ids = local_ids[order]
        unique_shards, counts = torch.unique_consecutive(
            sorted_shards,
            return_counts=True,
        )
        position = 0
        for shard_index, count in zip(
            unique_shards.tolist(),
            counts.tolist(),
        ):
            selected = sorted_local_ids[position : position + count]
            packed = code_shards[shard_index].index_select(0, selected)
            low = (packed & 0xF).long()
            high = (packed >> 4).long()
            nibbles = torch.stack((low, high), dim=-1).reshape(count, width)
            scales = scale_shards[shard_index].index_select(0, selected).float()
            rows = (
                dequant_lut[nibbles]
                * scales.repeat_interleave(16, dim=1)
                * outer_scales[shard_index]
            )
            output[order[position : position + count]] = rows.to(output.dtype)
            position += count

    native = NvFp4PleNativeGather(
        library_path=args.library,
        code_shards=code_shards,
        scale_shards=scale_shards,
        outer_scales=outer_scales,
        rows_per_shard=rows_per_shard,
        width=width,
    )

    def native_gather() -> None:
        if not native.gather_into(row_ids, output):
            raise RuntimeError("native gather rejected the production BF16 shape")

    torch_gather()
    expected = output.clone()
    native_gather()
    if not torch.equal(output, expected):
        raise RuntimeError("native gather does not match the Torch reference")
    for _ in range(50):
        torch_gather()
        native_gather()

    torch_result = _measure(torch_gather, args.iterations)
    native_result = _measure(native_gather, args.iterations)
    print(
        json.dumps(
            {
                "shard_count": shard_count,
                "rows_per_shard": rows_per_shard,
                "width": width,
                "row_count": args.row_count,
                "iterations": args.iterations,
                "torch": torch_result,
                "native": native_result,
                "mean_speedup": torch_result["mean_ms"] / native_result["mean_ms"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
