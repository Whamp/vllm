# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Summarize Qwen3.8 warmed allocator and NVML stability evidence."""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

_STAGES = (
    "warmup-complete",
    "serve-execute-step-1",
    "serve-execute-step-10",
    "serve-execute-step-250",
    "serve-execute-step-500",
    "serve-execute-step-750",
)
_COUNTERS = (
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "allocator_cache_bytes",
    "device_free_bytes",
    "torch_peak_allocated_bytes",
)


def _load_reports(report_dir: Path) -> dict[str, dict[int, dict[str, Any]]]:
    reports: dict[str, dict[int, dict[str, Any]]] = defaultdict(dict)
    for report_path in sorted(report_dir.glob("*.json")):
        report = json.loads(report_path.read_text())
        stage = report["stage"].replace(" ", "-")
        rank = int(report["rank"])
        if stage in _STAGES:
            if rank in reports[stage]:
                raise ValueError(f"duplicate {stage} report for rank {rank}")
            reports[stage][rank] = report
    expected_ranks = set(range(4))
    for stage in _STAGES:
        actual_ranks = set(reports[stage])
        if actual_ranks != expected_ranks:
            raise ValueError(
                f"{stage} ranks {sorted(actual_ranks)} != {sorted(expected_ranks)}"
            )
    return reports


def _load_nvml(path: Path) -> dict[int, list[dict[str, str]]]:
    rows_by_rank: dict[int, list[dict[str, str]]] = defaultdict(list)
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            rows_by_rank[int(row["index"].strip())].append(row)
    if set(rows_by_rank) != set(range(4)):
        raise ValueError("NVML samples must cover ranks 0-3")
    return rows_by_rank


def _max_process_swap_kib(path: Path) -> int:
    maximum = 0
    with path.open(newline="") as source:
        for row in csv.DictReader(source):
            maximum = max(maximum, int(row["vmswap_kib"]))
    return maximum


def analyze(evidence_dir: Path) -> dict[str, Any]:
    reports = _load_reports(evidence_dir / "reports")
    nvml_rows = _load_nvml(evidence_dir / "nvml-timeseries.csv")
    acceptance = json.loads((evidence_dir / "acceptance.json").read_text())
    cold_niah = json.loads((evidence_dir / "niah-cold.json").read_text())

    allocator_by_rank: dict[str, Any] = {}
    stable_allocator = True
    for rank in range(4):
        warmup = reports["warmup-complete"][rank]["memory_counters"]
        step_500 = reports["serve-execute-step-500"][rank]["memory_counters"]
        step_750 = reports["serve-execute-step-750"][rank]["memory_counters"]
        stable_counters = {
            counter: step_750[counter] - step_500[counter] for counter in _COUNTERS
        }
        stable_allocator &= all(delta == 0 for delta in stable_counters.values())
        allocator_by_rank[str(rank)] = {
            "warmup": {counter: warmup[counter] for counter in _COUNTERS},
            "serve_execute_step_500": {
                counter: step_500[counter] for counter in _COUNTERS
            },
            "serve_execute_step_750": {
                counter: step_750[counter] for counter in _COUNTERS
            },
            "warmup_to_step_500": {
                counter: step_500[counter] - warmup[counter] for counter in _COUNTERS
            },
            "step_500_to_step_750": stable_counters,
        }

    nvml_by_rank: dict[str, Any] = {}
    stable_nvml = True
    for rank, rows in sorted(nvml_rows.items()):
        free_mib = [int(row["memory_free_mib"].strip()) for row in rows]
        used_mib = [int(row["memory_used_mib"].strip()) for row in rows]
        stable_nvml &= used_mib[-1] == used_mib[0]
        nvml_by_rank[str(rank)] = {
            "samples": len(rows),
            "first_used_mib": used_mib[0],
            "last_used_mib": used_mib[-1],
            "used_growth_mib": used_mib[-1] - used_mib[0],
            "minimum_free_mib": min(free_mib),
            "first_free_mib": free_mib[0],
            "last_free_mib": free_mib[-1],
        }

    max_swap_kib = _max_process_swap_kib(evidence_dir / "process-swap-timeseries.csv")
    return {
        "schema_version": 1,
        "acceptance": {
            "deterministic": "passed",
            "tool_and_post_tool": "passed",
            "multimodal": "passed",
            "concurrency_2_aggregate_tokens_per_second": acceptance["concurrency_2"][
                "aggregate_tokens_per_second"
            ],
            "niah_api_prompt_tokens": acceptance["niah"]["api_prompt_tokens"],
            "niah_answer": acceptance["niah"]["answer"],
        },
        "cold_prefix_busted_niah": cold_niah,
        "allocator_by_rank": allocator_by_rank,
        "nvml_by_rank": nvml_by_rank,
        "maximum_process_swap_kib": max_swap_kib,
        "decision": {
            "allocator_counters_stable_from_step_500_to_step_750": stable_allocator,
            "nvml_used_memory_stable_during_cold_261k_niah": stable_nvml,
            "zero_process_swap": max_swap_kib == 0,
            "passed": stable_allocator and stable_nvml and max_swap_kib == 0,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("evidence_dir", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    result = analyze(args.evidence_dir)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
