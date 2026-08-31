# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Compare CUDA Graph replay spans and stream overlap in matched Nsight traces."""

from __future__ import annotations

import argparse
import json
import sqlite3
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any

GRAPH_PHASES = {8: "c1", 5: "c2"}


def _active_times(rows: list[tuple[int, int, int]]) -> tuple[int, int, int]:
    """Return busy union, two-stream overlap, and maximum active streams."""
    by_stream: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for stream, start, end in rows:
        by_stream[stream].append((start, end))

    endpoints: list[tuple[int, int, int]] = []
    for stream, intervals in by_stream.items():
        merged: list[list[int]] = []
        for start, end in sorted(intervals):
            if not merged or start > merged[-1][1]:
                merged.append([start, end])
            elif end > merged[-1][1]:
                merged[-1][1] = end
        for start, end in merged:
            endpoints.append((start, 1, stream))
            endpoints.append((end, -1, stream))

    endpoints.sort(key=lambda endpoint: (endpoint[0], endpoint[1]))
    active: set[int] = set()
    previous: int | None = None
    busy_union = 0
    overlap_two_plus = 0
    maximum_active = 0
    for timestamp, delta, stream in endpoints:
        if previous is not None and timestamp > previous:
            duration = timestamp - previous
            if active:
                busy_union += duration
            if len(active) >= 2:
                overlap_two_plus += duration
        if delta < 0:
            active.discard(stream)
        else:
            active.add(stream)
        maximum_active = max(maximum_active, len(active))
        previous = timestamp
    return busy_union, overlap_two_plus, maximum_active


def _summarize_trace(path: Path) -> dict[str, Any]:
    connection = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    result: dict[str, Any] = {}
    try:
        for device in range(4):
            grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
            graph_ids: dict[int, int] = {}
            query = """
                SELECT correlationId, graphId, streamId, start, end
                FROM CUPTI_ACTIVITY_KIND_KERNEL
                WHERE deviceId = ? AND graphId IS NOT NULL
                ORDER BY correlationId, start
            """
            for correlation_id, graph_id, stream_id, start, end in connection.execute(
                query, (device,)
            ):
                grouped[correlation_id].append((stream_id, start, end))
                graph_ids[correlation_id] = graph_id

            replays: dict[int, list[dict[str, float | int]]] = defaultdict(list)
            for correlation_id, rows in grouped.items():
                start = min(row[1] for row in rows)
                end = max(row[2] for row in rows)
                busy_union, overlap_two_plus, maximum_active = _active_times(rows)
                replays[graph_ids[correlation_id]].append(
                    {
                        "span_us": (end - start) / 1e3,
                        "busy_union_us": busy_union / 1e3,
                        "overlap_2plus_us": overlap_two_plus / 1e3,
                        "overlap_fraction": (
                            overlap_two_plus / busy_union if busy_union else 0.0
                        ),
                        "kernel_sum_us": sum(
                            row_end - row_start for _, row_start, row_end in rows
                        )
                        / 1e3,
                        "streams": len({row[0] for row in rows}),
                        "max_active_streams": maximum_active,
                        "kernel_count": len(rows),
                    }
                )

            device_result: dict[str, Any] = {}
            for graph_id, values in replays.items():
                if graph_id not in GRAPH_PHASES:
                    continue
                summary: dict[str, float | int | str] = {
                    "phase": GRAPH_PHASES[graph_id],
                    "replay_count": len(values),
                }
                for key in values[0]:
                    summary[f"{key}_median"] = statistics.median(
                        float(value[key]) for value in values
                    )
                    summary[f"{key}_mean"] = statistics.mean(
                        float(value[key]) for value in values
                    )
                device_result[str(graph_id)] = summary
            result[str(device)] = device_result
    finally:
        connection.close()
    return result


def _add_deltas(control: dict[str, Any], candidate: dict[str, Any]) -> dict[str, Any]:
    deltas: dict[str, Any] = {}
    for device in range(4):
        device_result: dict[str, Any] = {}
        for graph_id, phase in GRAPH_PHASES.items():
            control_summary = control[str(device)][str(graph_id)]
            candidate_summary = candidate[str(device)][str(graph_id)]
            phase_result: dict[str, Any] = {"phase": phase}
            for metric in ("span_us", "busy_union_us", "overlap_2plus_us"):
                key = f"{metric}_median"
                control_value = float(control_summary[key])
                candidate_value = float(candidate_summary[key])
                phase_result[f"{metric}_control"] = control_value
                phase_result[f"{metric}_candidate"] = candidate_value
                phase_result[f"{metric}_delta_percent"] = (
                    candidate_value / control_value - 1.0
                ) * 100.0
            phase_result["overlap_fraction_control"] = control_summary[
                "overlap_fraction_median"
            ]
            phase_result["overlap_fraction_candidate"] = candidate_summary[
                "overlap_fraction_median"
            ]
            phase_result["streams_control"] = control_summary["streams_median"]
            phase_result["streams_candidate"] = candidate_summary["streams_median"]
            device_result[str(graph_id)] = phase_result
        deltas[str(device)] = device_result
    return deltas


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--control", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    control = _summarize_trace(args.control)
    candidate = _summarize_trace(args.candidate)
    result = {
        "schema_version": 1,
        "graph_phases": {str(key): value for key, value in GRAPH_PHASES.items()},
        "control": control,
        "candidate": candidate,
        "deltas": _add_deltas(control, candidate),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    temporary.replace(args.output)


if __name__ == "__main__":
    main()
