# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measure direct BF16 SSD-backed PLE gathers without starting a GPU model."""

import argparse
import json
import os
import resource
import time
from pathlib import Path

import torch

from vllm.v1.ple_offload.bf16_ple_mmap_gather import Bf16PleMmapGather

_QWEN38_PLE_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"
_QWEN38_PLE_ROWS = 320_001_536
_QWEN38_PLE_WIDTH = 160
_QWEN38_PLE_HEADS = 16
_UINT64_MASK = (1 << 64) - 1


def _splitmix64(value: int) -> int:
    value = (value + 0x9E3779B97F4A7C15) & _UINT64_MASK
    value = ((value ^ (value >> 30)) * 0xBF58476D1CE4E5B9) & _UINT64_MASK
    value = ((value ^ (value >> 27)) * 0x94D049BB133111EB) & _UINT64_MASK
    return value ^ (value >> 31)


def _process_read_bytes() -> int:
    for line in Path("/proc/self/io").read_text().splitlines():
        if line.startswith("read_bytes:"):
            return int(line.split()[1])
    raise RuntimeError("process I/O counters do not contain read_bytes")


def _evict_checkpoint_pages(checkpoint_path: Path) -> None:
    if not hasattr(os, "posix_fadvise") or not hasattr(os, "POSIX_FADV_DONTNEED"):
        raise RuntimeError("cold BF16 PLE benchmark requires POSIX_FADV_DONTNEED")
    descriptor = os.open(checkpoint_path, os.O_RDONLY)
    try:
        os.posix_fadvise(descriptor, 0, 0, os.POSIX_FADV_DONTNEED)
    finally:
        os.close(descriptor)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--expected-sha256", required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--steps", type=int, default=64)
    parser.add_argument("--seed", type=int, default=20260831)
    parser.add_argument("--drop-file-cache", action="store_true")
    args = parser.parse_args()
    if args.steps <= 0:
        parser.error("--steps must be positive")
    if args.drop_file_cache:
        _evict_checkpoint_pages(args.checkpoint)

    rows_per_head = _QWEN38_PLE_ROWS // _QWEN38_PLE_HEADS
    row_batches = tuple(
        torch.tensor(
            [
                head * rows_per_head
                + _splitmix64(args.seed + step * _QWEN38_PLE_HEADS + head)
                % rows_per_head
                for head in range(_QWEN38_PLE_HEADS)
            ],
            dtype=torch.int64,
        )
        for step in range(args.steps)
    )
    output = torch.empty((_QWEN38_PLE_HEADS, _QWEN38_PLE_WIDTH), dtype=torch.bfloat16)
    table = Bf16PleMmapGather(
        checkpoint_path=args.checkpoint,
        expected_sha256=args.expected_sha256,
        native_library_path=args.library,
        tensor_prefix=_QWEN38_PLE_PREFIX,
        total_rows=_QWEN38_PLE_ROWS,
        width=_QWEN38_PLE_WIDTH,
    )
    try:
        usage_before = resource.getrusage(resource.RUSAGE_SELF)
        reads_before = _process_read_bytes()
        started_ns = time.perf_counter_ns()
        checksum = 0
        for row_ids in row_batches:
            table.gather_into(row_ids, output)
            output_bits = output.view(torch.uint16)
            checksum += int(output_bits[0, 0]) + int(output_bits[-1, -1])
        elapsed_ms = (time.perf_counter_ns() - started_ns) / 1_000_000
        reads_after = _process_read_bytes()
        usage_after = resource.getrusage(resource.RUSAGE_SELF)
    finally:
        table.close()

    read_bytes = reads_after - reads_before
    print(
        json.dumps(
            {
                "access_advice": "MADV_RANDOM",
                "checksum": checksum,
                "cold_file_cache_requested": args.drop_file_cache,
                "elapsed_ms": elapsed_ms,
                "major_faults": usage_after.ru_majflt - usage_before.ru_majflt,
                "minor_faults": usage_after.ru_minflt - usage_before.ru_minflt,
                "ms_per_token": elapsed_ms / args.steps,
                "read_bytes": read_bytes,
                "read_bytes_per_token": read_bytes / args.steps,
                "seed": args.seed,
                "steps": args.steps,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
