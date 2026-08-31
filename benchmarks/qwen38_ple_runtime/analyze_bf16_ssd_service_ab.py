#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Analyze the order-balanced BF16 SSD PLE service A/B."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any


def _load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _summary(values: list[float]) -> dict[str, float]:
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "minimum": min(values),
        "maximum": max(values),
        "population_cv_percent": statistics.pstdev(values)
        / statistics.mean(values)
        * 100,
    }


def _telemetry_delta(before_path: Path, after_path: Path) -> dict[str, float | int]:
    before = _load(before_path)
    after = _load(after_path)
    before_worker = before["ple_worker"]
    after_worker = after["ple_worker"]
    return {
        "wall_seconds": (
            after["timestamp_monotonic_ns"] - before["timestamp_monotonic_ns"]
        )
        / 1e9,
        "ple_cpu_seconds": (
            after_worker["user_cpu_seconds"]
            + after_worker["system_cpu_seconds"]
            - before_worker["user_cpu_seconds"]
            - before_worker["system_cpu_seconds"]
        ),
        "ple_major_faults": (
            after_worker["major_faults"] - before_worker["major_faults"]
        ),
        "ple_minor_faults": (
            after_worker["minor_faults"] - before_worker["minor_faults"]
        ),
        "ple_process_read_bytes": (
            after_worker["io"]["read_bytes"] - before_worker["io"]["read_bytes"]
        ),
        "host_nvme_read_bytes": (
            after["nvme0n1"]["read_bytes"] - before["nvme0n1"]["read_bytes"]
        ),
        "host_nvme_read_time_ms": (
            after["nvme0n1"]["read_time_ms"] - before["nvme0n1"]["read_time_ms"]
        ),
    }


def _percent_change(candidate: float, control: float) -> float:
    return (candidate / control - 1) * 100


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    arms = {
        label: _load(args.root / label / "results.json")
        for label in ("candidate-a", "control-a", "control-b", "candidate-b")
    }
    families = {
        "candidate": ("candidate-a", "candidate-b"),
        "control": ("control-a", "control-b"),
    }
    comparison: dict[str, Any] = {}
    for concurrency in ("1", "2"):
        pooled: dict[str, Any] = {}
        for family, labels in families.items():
            decode = [
                run["aggregate_decode_tokens_per_second"]
                for label in labels
                for run in arms[label]["concurrency"][concurrency]["decode"]["measured"]
            ]
            prefill = [
                run["aggregate_prompt_tokens_per_second"]
                for label in labels
                for run in arms[label]["concurrency"][concurrency]["prefill"][
                    "measured"
                ]
            ]
            pooled[family] = {
                "decode_tokens_per_second": _summary(decode),
                "prefill_tokens_per_second": _summary(prefill),
                "decode_runs": len(decode),
                "prefill_runs": len(prefill),
            }
        candidate_decode = pooled["candidate"]["decode_tokens_per_second"]["mean"]
        control_decode = pooled["control"]["decode_tokens_per_second"]["mean"]
        candidate_prefill = pooled["candidate"]["prefill_tokens_per_second"]["mean"]
        control_prefill = pooled["control"]["prefill_tokens_per_second"]["mean"]
        comparison[concurrency] = {
            **pooled,
            "decode_percent_change": _percent_change(candidate_decode, control_decode),
            "prefill_percent_change": _percent_change(
                candidate_prefill, control_prefill
            ),
        }

    telemetry = {
        label: _telemetry_delta(
            args.root / label / "telemetry-before.json",
            args.root / label / "telemetry-after.json",
        )
        for label in arms
    }
    limits = {
        "concurrency_1_decode_minimum_percent_change": -10.0,
        "concurrency_2_decode_minimum_percent_change": -15.0,
        "prefill_minimum_percent_change": -5.0,
    }
    checks = {
        "concurrency_1_decode": comparison["1"]["decode_percent_change"]
        >= limits["concurrency_1_decode_minimum_percent_change"],
        "concurrency_2_decode": comparison["2"]["decode_percent_change"]
        >= limits["concurrency_2_decode_minimum_percent_change"],
        "concurrency_1_prefill": comparison["1"]["prefill_percent_change"]
        >= limits["prefill_minimum_percent_change"],
        "concurrency_2_prefill": comparison["2"]["prefill_percent_change"]
        >= limits["prefill_minimum_percent_change"],
    }
    result = {
        "schema_version": 1,
        "arm_order": ["candidate-a", "control-a", "control-b", "candidate-b"],
        "comparison": comparison,
        "telemetry": telemetry,
        "limits": limits,
        "checks": checks,
        "performance_gate_passed": all(checks.values()),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
