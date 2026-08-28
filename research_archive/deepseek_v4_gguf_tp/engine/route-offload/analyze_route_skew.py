#!/usr/bin/env python3
"""Analyze DeepSeek V4 routed-expert skew for the GGUF-TP offload gate.

Each input is a compact workload summary with per-layer expert visit counts and,
when available, the ordered top-k expert IDs for consecutive tokens. The report
contains full coverage-vs-hot-count curves, cross-session hot-set transfer, and
consecutive-token/LRU proxies. The preregistered decision is conservative across
all layers and workloads: GO only when every H99 is at most 224; NO-GO when any
H99 is at least 248; otherwise INCONCLUSIVE.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import OrderedDict
from itertools import pairwise
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
EXPECTED_LAYERS = 43
EXPECTED_EXPERTS = 256
EXPECTED_TOP_K = 6
GO_MAX_H99 = 224
NO_GO_MIN_H99 = 248
REPORTED_HOT_COUNTS = (192, 208, 224, 240, 248, 256)

JsonObject = dict[str, Any]


def load_workload_summary(path: Path) -> JsonObject:
    """Load and validate one route-skew workload summary."""
    with path.open(encoding="utf-8") as stream:
        workload = json.load(stream)
    if workload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: unsupported schema_version")
    if not workload.get("workload_id"):
        raise ValueError(f"{path}: missing workload_id")
    if workload.get("n_experts") != EXPECTED_EXPERTS:
        raise ValueError(f"{path}: expected {EXPECTED_EXPERTS} experts")
    if workload.get("top_k") != EXPECTED_TOP_K:
        raise ValueError(f"{path}: expected top_k={EXPECTED_TOP_K}")
    layers = workload.get("layers")
    if not isinstance(layers, list) or not layers:
        raise ValueError(f"{path}: layers must be a nonempty list")
    seen: set[int] = set()
    for layer in layers:
        layer_index = layer.get("layer")
        if not isinstance(layer_index, int) or layer_index < 0:
            raise ValueError(f"{path}: invalid layer index {layer_index!r}")
        if layer_index in seen:
            raise ValueError(f"{path}: duplicate layer {layer_index}")
        seen.add(layer_index)
        counts = layer.get("counts")
        if not isinstance(counts, list) or len(counts) != EXPECTED_EXPERTS:
            raise ValueError(f"{path}: layer {layer_index} has invalid counts")
        if any(not isinstance(count, int) or count < 0 for count in counts):
            raise ValueError(f"{path}: layer {layer_index} has negative/non-int counts")
        if sum(counts) <= 0:
            raise ValueError(f"{path}: layer {layer_index} has zero visits")
        routes = layer.get("routes")
        reuse = layer.get("reuse")
        if routes is not None and reuse is not None:
            raise ValueError(
                f"{path}: layer {layer_index} cannot contain routes and reuse"
            )
        if routes is not None:
            if not isinstance(routes, list):
                raise ValueError(f"{path}: layer {layer_index} routes must be a list")
            for token_index, experts in enumerate(routes):
                if not isinstance(experts, list) or len(experts) != EXPECTED_TOP_K:
                    raise ValueError(
                        f"{path}: layer {layer_index} token {token_index} has invalid top-k"
                    )
                if len(set(experts)) != EXPECTED_TOP_K:
                    raise ValueError(
                        f"{path}: layer {layer_index} token {token_index} repeats experts"
                    )
                if any(
                    not isinstance(expert, int)
                    or expert < 0
                    or expert >= EXPECTED_EXPERTS
                    for expert in experts
                ):
                    raise ValueError(
                        f"{path}: layer {layer_index} token {token_index} has invalid expert"
                    )
        if reuse is not None:
            validate_precomputed_reuse(reuse, path=path, layer_index=layer_index)
    workload["_path"] = str(path)
    workload["_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return workload


def validate_precomputed_reuse(reuse: Any, *, path: Path, layer_index: int) -> None:
    """Validate compact consecutive-route statistics from a trusted summarizer."""
    if not isinstance(reuse, dict):
        raise TypeError(f"{path}: layer {layer_index} reuse must be an object")
    token_count = reuse.get("token_count")
    transition_count = reuse.get("transition_count")
    if not isinstance(token_count, int) or token_count <= 0:
        raise ValueError(f"{path}: layer {layer_index} reuse has invalid token_count")
    if not isinstance(transition_count, int) or not 0 <= transition_count < token_count:
        raise ValueError(
            f"{path}: layer {layer_index} reuse has invalid transition_count"
        )
    for key in ("mean_consecutive_overlap", "exact_set_repeat_rate"):
        value = reuse.get(key)
        if not isinstance(value, (int, float)) or not 0 <= value <= 1:
            raise ValueError(f"{path}: layer {layer_index} reuse has invalid {key}")
    lru = reuse.get("lru_hit_rate")
    if not isinstance(lru, dict) or set(lru) != {"224", "248"}:
        raise ValueError(f"{path}: layer {layer_index} reuse has invalid LRU keys")
    if any(
        not isinstance(value, (int, float)) or not 0 <= value <= 1
        for value in lru.values()
    ):
        raise ValueError(f"{path}: layer {layer_index} reuse has invalid LRU rate")


def expert_coverage_curve(counts: list[int]) -> list[float]:
    """Return cumulative visit coverage for hot-set sizes H=1..len(counts)."""
    total = sum(counts)
    ordered = sorted(counts, reverse=True)
    cumulative = 0
    result: list[float] = []
    for count in ordered:
        cumulative += count
        result.append(cumulative / total)
    return result


def hot_count_for_coverage(
    counts: list[int], *, numerator: int = 99, denominator: int = 100
) -> int:
    """Return the smallest H reaching an exact rational coverage threshold."""
    if numerator <= 0 or denominator <= 0 or numerator > denominator:
        raise ValueError("Coverage threshold must satisfy 0 < numerator <= denominator")
    total = sum(counts)
    cumulative = 0
    for index, count in enumerate(sorted(counts, reverse=True), 1):
        cumulative += count
        if cumulative * denominator >= total * numerator:
            return index
    raise ValueError("Coverage counts never reach the requested threshold")


def top_expert_set(counts: list[int], hot_count: int) -> set[int]:
    """Return a deterministic top-H expert set, breaking ties by expert ID."""
    ordered = sorted(range(len(counts)), key=lambda expert: (-counts[expert], expert))
    return set(ordered[:hot_count])


def coverage_by_expert_set(counts: list[int], experts: set[int]) -> float:
    """Return visit coverage obtained by one explicit expert set."""
    return sum(counts[expert] for expert in experts) / sum(counts)


def consecutive_route_metrics(routes: list[list[int]]) -> JsonObject:
    """Measure consecutive-set reuse and LRU hit rates for one layer route sequence."""
    if not routes:
        return {
            "token_count": 0,
            "transition_count": 0,
            "mean_consecutive_overlap": None,
            "exact_set_repeat_rate": None,
            "lru_hit_rate": {},
        }
    overlap_sum = 0
    exact_repeats = 0
    for previous, current in pairwise(routes):
        overlap = len(set(previous) & set(current))
        overlap_sum += overlap
        exact_repeats += set(previous) == set(current)
    transitions = max(0, len(routes) - 1)
    lru_rates = {
        str(hot_count): simulate_route_lru_hit_rate(routes, hot_count)
        for hot_count in (GO_MAX_H99, NO_GO_MIN_H99)
    }
    return {
        "token_count": len(routes),
        "transition_count": transitions,
        "mean_consecutive_overlap": (
            overlap_sum / (transitions * EXPECTED_TOP_K) if transitions else None
        ),
        "exact_set_repeat_rate": exact_repeats / transitions if transitions else None,
        "lru_hit_rate": lru_rates,
    }


def simulate_route_lru_hit_rate(routes: list[list[int]], capacity: int) -> float:
    """Simulate per-layer LRU; token peers test hits before their shared update."""
    cache: OrderedDict[int, None] = OrderedDict()
    hits = 0
    accesses = 0
    for experts in routes:
        resident_before = set(cache)
        hits += sum(expert in resident_before for expert in experts)
        accesses += len(experts)
        for expert in experts:
            cache.pop(expert, None)
            cache[expert] = None
        while len(cache) > capacity:
            cache.popitem(last=False)
    return hits / accesses if accesses else 0.0


def analyze_route_workloads(workloads: list[JsonObject]) -> JsonObject:
    """Build coverage, stability, reuse, and preregistered decision evidence."""
    if len(workloads) < 2:
        raise ValueError("Route-skew analysis requires at least two workloads")
    layer_sets = [
        {layer["layer"] for layer in workload["layers"]} for workload in workloads
    ]
    if any(layer_set != layer_sets[0] for layer_set in layer_sets[1:]):
        raise ValueError("Route-skew workloads do not contain identical layer sets")

    workload_results: list[JsonObject] = []
    layer_lookup_by_workload: list[dict[int, JsonObject]] = []
    for workload in workloads:
        layer_lookup = {layer["layer"]: layer for layer in workload["layers"]}
        layer_lookup_by_workload.append(layer_lookup)
        layer_results: list[JsonObject] = []
        h99_values: list[int] = []
        for layer_index in sorted(layer_lookup):
            layer = layer_lookup[layer_index]
            curve = expert_coverage_curve(layer["counts"])
            h99 = hot_count_for_coverage(layer["counts"])
            h99_values.append(h99)
            layer_results.append(
                {
                    "layer": layer_index,
                    "h99": h99,
                    "coverage_curve": curve,
                    "reported_coverage": {
                        str(hot_count): curve[hot_count - 1]
                        for hot_count in REPORTED_HOT_COUNTS
                    },
                    "reuse": (
                        consecutive_route_metrics(layer["routes"])
                        if "routes" in layer
                        else layer.get("reuse")
                    ),
                }
            )
        workload_results.append(
            {
                "workload_id": workload["workload_id"],
                "source_path": workload["_path"],
                "source_sha256": workload["_sha256"],
                "layers": layer_results,
                "max_h99": max(h99_values),
                "median_h99": sorted(h99_values)[len(h99_values) // 2],
            }
        )

    stability = []
    for left_index in range(len(workloads)):
        for right_index in range(left_index + 1, len(workloads)):
            left = workloads[left_index]
            right = workloads[right_index]
            left_layers = layer_lookup_by_workload[left_index]
            right_layers = layer_lookup_by_workload[right_index]
            per_layer = []
            for layer_index in sorted(layer_sets[0]):
                layer_result: JsonObject = {"layer": layer_index}
                for hot_count in (GO_MAX_H99, NO_GO_MIN_H99):
                    left_set = top_expert_set(
                        left_layers[layer_index]["counts"], hot_count
                    )
                    right_set = top_expert_set(
                        right_layers[layer_index]["counts"], hot_count
                    )
                    layer_result[str(hot_count)] = {
                        "jaccard": len(left_set & right_set)
                        / len(left_set | right_set),
                        "left_hot_set_coverage_on_right": coverage_by_expert_set(
                            right_layers[layer_index]["counts"], left_set
                        ),
                        "right_hot_set_coverage_on_left": coverage_by_expert_set(
                            left_layers[layer_index]["counts"], right_set
                        ),
                    }
                per_layer.append(layer_result)
            stability.append(
                {
                    "left_workload_id": left["workload_id"],
                    "right_workload_id": right["workload_id"],
                    "layers": per_layer,
                }
            )

    maximum_h99 = max(result["max_h99"] for result in workload_results)
    observed_layer_count = len(layer_sets[0])
    if maximum_h99 >= NO_GO_MIN_H99:
        decision = "NO-GO"
        rationale = f"At least one observed layer requires H>={NO_GO_MIN_H99} for 99% visit coverage."
    elif observed_layer_count < EXPECTED_LAYERS:
        decision = "INCONCLUSIVE"
        rationale = (
            f"Only {observed_layer_count}/{EXPECTED_LAYERS} layers are present; partial "
            "passing evidence cannot satisfy the GO gate."
        )
    elif maximum_h99 <= GO_MAX_H99:
        decision = "GO"
        rationale = f"Every layer in every workload reaches 99% visit coverage at H<={GO_MAX_H99}."
    else:
        decision = "INCONCLUSIVE"
        rationale = f"Worst-layer H99={maximum_h99}, between the preregistered GO and NO-GO gates."
    reuse_available = all(
        layer["reuse"] is not None
        for workload in workload_results
        for layer in workload["layers"]
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "decision": decision,
        "decision_rationale": rationale,
        "maximum_h99": maximum_h99,
        "observed_layer_count": observed_layer_count,
        "expected_layer_count": EXPECTED_LAYERS,
        "reuse_available_for_all_layers": reuse_available,
        "workloads": workload_results,
        "cross_workload_stability": stability,
    }


def render_route_report_markdown(analysis: JsonObject) -> str:
    """Render the route-skew result as a concise cold-reader Markdown report."""
    lines = [
        "# DeepSeek V4 GGUF-TP cold-expert offload route-skew report",
        "",
        f"**Decision: {analysis['decision']}** — {analysis['decision_rationale']}",
        "",
        (
            "The preregistered gate is GO only when every observed layer reaches 99% "
            f"coverage at H≤{GO_MAX_H99}; NO-GO when any requires H≥{NO_GO_MIN_H99}."
        ),
        "",
        "## Workloads",
        "",
        "| Workload | Median H99 | Worst H99 | Full consecutive reuse data |",
        "| --- | ---: | ---: | --- |",
    ]
    for workload in analysis["workloads"]:
        reuse = all(layer["reuse"] is not None for layer in workload["layers"])
        lines.append(
            f"| `{workload['workload_id']}` | {workload['median_h99']} | "
            f"{workload['max_h99']} | {'yes' if reuse else 'no'} |"
        )
    lines.extend(
        [
            "",
            "## Per-layer coverage",
            "",
            "Full H=1..256 curves are in the adjacent JSON report. Selected points:",
            "",
        ]
    )
    header = ["Layer"]
    for workload in analysis["workloads"]:
        header.extend(
            [
                f"{workload['workload_id']} H99",
                f"{workload['workload_id']} cov@224",
                f"{workload['workload_id']} cov@248",
            ]
        )
    lines.append("| " + " | ".join(header) + " |")
    lines.append("| " + " | ".join(["---:"] * len(header)) + " |")
    layer_count = len(analysis["workloads"][0]["layers"])
    for layer_position in range(layer_count):
        row = [str(analysis["workloads"][0]["layers"][layer_position]["layer"])]
        for workload in analysis["workloads"]:
            layer = workload["layers"][layer_position]
            row.extend(
                [
                    str(layer["h99"]),
                    f"{layer['reported_coverage']['224']:.6f}",
                    f"{layer['reported_coverage']['248']:.6f}",
                ]
            )
        lines.append("| " + " | ".join(row) + " |")
    lines.extend(
        [
            "",
            "## Interpretation constraints",
            "",
            "- Coverage is per-layer routed-expert visit coverage, not a performance claim.",
            (
                "- Cross-session transfer and consecutive/LRU evidence must be inspected before "
                "designing a cache even when the H99 gate says GO."
            ),
            (
                "- Missing consecutive route sequences leave the LRU proxy unverified; histogram "
                "coverage alone cannot establish temporal locality."
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("workloads", type=Path, nargs="+")
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-markdown", type=Path, required=True)
    args = parser.parse_args()
    workloads = [load_workload_summary(path) for path in args.workloads]
    analysis = analyze_route_workloads(workloads)
    write_json_atomic(args.output_json, analysis)
    args.output_markdown.write_text(
        render_route_report_markdown(analysis), encoding="utf-8"
    )
    print(
        json.dumps(
            {"decision": analysis["decision"], "maximum_h99": analysis["maximum_h99"]}
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
