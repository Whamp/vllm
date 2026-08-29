#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize matched Qwen3.8 llama-benchy c=1 and c=2 results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def _phase(result: dict[str, Any], *, cold: bool) -> dict[str, Any]:
    matches = [
        benchmark
        for benchmark in result["benchmarks"]
        if benchmark["is_context_prefill_phase"] is cold
    ]
    if len(matches) != 1:
        raise ValueError(f"Expected one cold={cold} benchmark, got {len(matches)}")
    return matches[0]


def summarize_llama_benchy(
    c1_path: Path,
    c2_path: Path,
    *,
    alesha_decode_tokens_per_second: float,
) -> dict[str, Any]:
    """Return phase-correct decode and mixed-workload measurements."""

    c1 = json.loads(c1_path.read_text())
    c2 = json.loads(c2_path.read_text())
    for name, result, concurrency in (("c1", c1, 1), ("c2", c2, 2)):
        if result["max_concurrency"] != concurrency:
            raise ValueError(
                f"{name} result declares max_concurrency={result['max_concurrency']}"
            )
        if result["model"] != c1["model"]:
            raise ValueError(f"{name} model does not match c1")

    c1_cached = _phase(c1, cold=False)
    c2_cold = _phase(c2, cold=True)
    c2_cached = _phase(c2, cold=False)
    c1_decode = float(c1_cached["tg_throughput"]["mean"])
    c2_decode = float(c2_cached["tg_throughput"]["mean"])
    c1_ms_per_token = 1000.0 / c1_decode
    alesha_ms_per_token = 1000.0 / alesha_decode_tokens_per_second

    return {
        "schema_version": 1,
        "model": c1["model"],
        "llama_benchy_version": c1["version"],
        "cold_c2_mixed_prefill_decode": {
            "aggregate_tokens_per_second": c2_cold["tg_throughput"]["mean"],
            "per_request_tokens_per_second": c2_cold["tg_req_throughput"]["values"],
            "time_to_first_response_ms": c2_cold["ttfr"]["values"],
            "interpretation": (
                "Mixed concurrent-prefill and decode interval; "
                "not steady decode scaling"
            ),
        },
        "cached_decode_scaling": {
            "c1_aggregate_tokens_per_second": c1_decode,
            "c2_aggregate_tokens_per_second": c2_decode,
            "c2_mean_per_request_tokens_per_second": c2_cached["tg_req_throughput"][
                "mean"
            ],
            "aggregate_speedup": c2_decode / c1_decode,
            "parallel_efficiency": c2_decode / (2.0 * c1_decode),
        },
        "decode_causal_budget": {
            "local_tokens_per_second": c1_decode,
            "local_ms_per_token": c1_ms_per_token,
            "alesha_tokens_per_second": alesha_decode_tokens_per_second,
            "alesha_ms_per_token": alesha_ms_per_token,
            "gap_ms_per_token": c1_ms_per_token - alesha_ms_per_token,
        },
        "sources": {
            "c1": c1_path.name,
            "c2": c2_path.name,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--c1", type=Path, required=True)
    parser.add_argument("--c2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--alesha-decode-tps",
        type=float,
        default=64.31,
        help="Pinned near-maximum Alesha comparison result",
    )
    args = parser.parse_args()
    summary = summarize_llama_benchy(
        args.c1,
        args.c2,
        alesha_decode_tokens_per_second=args.alesha_decode_tps,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
