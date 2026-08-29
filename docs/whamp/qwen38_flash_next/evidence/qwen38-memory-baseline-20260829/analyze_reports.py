#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Summarize checksum-bound staged Qwen3.8 GPU memory reports."""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

EXPECTED_STAGES = (
    "distributed initialized",
    "model runner initialized",
    "initialized",
    "weights loaded",
    "postprocessed",
    "PLE offload initialized",
    "worker model loaded",
    "profile complete",
    "kv cache allocated",
    "warmup complete",
)
EXPECTED_RANKS = frozenset(range(4))
MEMORY_COUNTERS = (
    "torch_allocated_bytes",
    "torch_reserved_bytes",
    "device_used_bytes",
    "unregistered_torch_allocated_bytes",
    "allocator_cache_bytes",
    "non_torch_device_bytes",
)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_raw_report_manifest(raw_directory: Path) -> str:
    manifest_path = raw_directory / "SHA256SUMS"
    manifest_lines = manifest_path.read_text().splitlines()
    manifest_files: set[str] = set()
    for line in manifest_lines:
        expected_sha256, relative_name = line.split(maxsplit=1)
        report_name = relative_name.removeprefix("./")
        report_path = raw_directory / report_name
        if not report_path.is_file():
            raise ValueError(f"Missing raw memory report: {report_name}")
        actual_sha256 = file_sha256(report_path)
        if actual_sha256 != expected_sha256:
            raise ValueError(
                f"Raw memory report checksum mismatch for {report_name}: "
                f"expected {expected_sha256}, got {actual_sha256}"
            )
        manifest_files.add(report_name)

    actual_files = {path.name for path in raw_directory.glob("*.json")}
    if actual_files != manifest_files:
        raise ValueError(
            "Raw memory report manifest inventory mismatch: "
            f"missing={sorted(actual_files - manifest_files)}, "
            f"unexpected={sorted(manifest_files - actual_files)}"
        )
    return file_sha256(manifest_path)


def load_ranked_stage_reports(raw_directory: Path) -> dict[str, list[dict[str, Any]]]:
    reports_by_stage: dict[str, list[dict[str, Any]]] = defaultdict(list)
    seen_stage_ranks: set[tuple[str, int]] = set()
    for report_path in sorted(raw_directory.glob("*.json")):
        report = json.loads(report_path.read_text())
        if report.get("schema_version") != 1:
            raise ValueError(f"Unsupported report schema in {report_path.name}")
        stage = report["stage"]
        rank = report["rank"]
        stage_rank = (stage, rank)
        if stage_rank in seen_stage_ranks:
            raise ValueError(
                f"Duplicate memory report for stage={stage!r}, rank={rank}"
            )
        seen_stage_ranks.add(stage_rank)
        reports_by_stage[stage].append(report)

    if set(reports_by_stage) != set(EXPECTED_STAGES):
        raise ValueError(
            "Memory report stage inventory mismatch: "
            f"expected={sorted(EXPECTED_STAGES)}, actual={sorted(reports_by_stage)}"
        )
    for stage, reports in reports_by_stage.items():
        ranks = {report["rank"] for report in reports}
        if ranks != EXPECTED_RANKS:
            raise ValueError(
                f"Memory report rank inventory mismatch for {stage!r}: "
                f"expected={sorted(EXPECTED_RANKS)}, actual={sorted(ranks)}"
            )
        reports.sort(key=lambda report: report["rank"])
    return reports_by_stage


def summarize_integer_values(values: list[int]) -> dict[str, int | float]:
    return {
        "minimum": min(values),
        "maximum": max(values),
        "mean": sum(values) / len(values),
    }


def summarize_stage(reports: list[dict[str, Any]]) -> dict[str, Any]:
    registered_values: list[int] = []
    for report in reports:
        if report["report_kind"] == "device":
            registered_values.append(0)
        elif report["report_kind"] == "model":
            registered_values.append(
                report["registered_tensors"]["unique_storage_bytes"]
            )
        else:
            raise ValueError(f"Unsupported memory report kind: {report['report_kind']}")
    counter_names = sorted(
        set(MEMORY_COUNTERS)
        & set.intersection(*(set(report["memory_counters"]) for report in reports))
    )
    return {
        "rank_count": len(reports),
        "registered_storage_bytes": summarize_integer_values(registered_values),
        "memory_counters": {
            counter_name: summarize_integer_values(
                [report["memory_counters"][counter_name] for report in reports]
            )
            for counter_name in counter_names
        },
    }


def classify_registered_storage(tensor_names: list[str]) -> str:
    joined_names = "|".join(tensor_names)
    if ".mlp.experts.routed_experts." in joined_names:
        return "routed_experts"
    if "hyper_connection" in joined_names:
        return "hyperconnections"
    if ".linear_attn." in joined_names:
        return "linear_attention"
    if "embed_tokens" in joined_names or "lm_head" in joined_names:
        return "embedding_and_head"
    if "topk_indices_buffer" in joined_names:
        return "qsa_topk_buffers"
    if "rotary_emb.cos_sin_cache" in joined_names:
        return "rope_cache"
    if ".self_attn.indexer." in joined_names:
        return "qsa_indexer"
    if ".self_attn." in joined_names:
        return "qsa_attention"
    if joined_names.startswith("visual.") or "|visual." in joined_names:
        return "vision"
    if ".mlp.gate." in joined_names:
        return "moe_router"
    if (
        ".mlp.experts._shared_experts." in joined_names
        or ".mlp.shared_expert." in joined_names
    ):
        return "shared_experts"
    if ".ple." in joined_names:
        return "ple_projections_and_buffers"
    raise ValueError(f"Unclassified registered storage: {tensor_names}")


def summarize_registered_storage_categories(
    reports: list[dict[str, Any]],
) -> dict[str, Any]:
    categories_by_rank: list[dict[str, int]] = []
    for report in reports:
        storages = report["registered_tensors"].get("storages")
        if storages is None:
            raise ValueError("Postprocessed report does not include storage details")
        category_bytes: dict[str, int] = defaultdict(int)
        for storage in storages:
            tensor_names = [tensor["name"] for tensor in storage["tensors"]]
            category_bytes[classify_registered_storage(tensor_names)] += storage[
                "nbytes"
            ]
        if (
            sum(category_bytes.values())
            != report["registered_tensors"]["unique_storage_bytes"]
        ):
            raise ValueError("Registered storage categories do not cover every byte")
        categories_by_rank.append(dict(category_bytes))

    category_names = set(categories_by_rank[0])
    if any(
        set(category_bytes) != category_names for category_bytes in categories_by_rank
    ):
        raise ValueError("Registered storage categories differ across ranks")
    return {
        category: summarize_integer_values(
            [rank_categories[category] for rank_categories in categories_by_rank]
        )
        for category in sorted(category_names)
    }


def build_memory_baseline_summary(raw_directory: Path) -> dict[str, Any]:
    manifest_sha256 = verify_raw_report_manifest(raw_directory)
    reports_by_stage = load_ranked_stage_reports(raw_directory)
    stage_summaries = {
        stage: summarize_stage(reports_by_stage[stage]) for stage in EXPECTED_STAGES
    }
    baseline_device_bytes = stage_summaries["distributed initialized"][
        "memory_counters"
    ]["device_used_bytes"]["mean"]
    profile_summary = stage_summaries["profile complete"]
    profile_registered_bytes = profile_summary["registered_storage_bytes"]["mean"]
    profile_counters = profile_summary["memory_counters"]
    profile_growth_beyond_registered = (
        profile_counters["device_used_bytes"]["mean"]
        - baseline_device_bytes
        - profile_registered_bytes
    )
    return {
        "schema_version": 1,
        "raw_manifest_sha256": manifest_sha256,
        "rank_count": len(EXPECTED_RANKS),
        "stage_order": list(EXPECTED_STAGES),
        "stages": stage_summaries,
        "postprocessed_registered_storage_categories": (
            summarize_registered_storage_categories(reports_by_stage["postprocessed"])
        ),
        "derived": {
            "baseline_device_bytes_per_rank": baseline_device_bytes,
            "profile_growth_beyond_registered_bytes_per_rank": (
                profile_growth_beyond_registered
            ),
            "weights_loaded_to_postprocessed_registered_delta_bytes_per_rank": (
                stage_summaries["postprocessed"]["registered_storage_bytes"]["mean"]
                - stage_summaries["weights loaded"]["registered_storage_bytes"]["mean"]
            ),
            "ple_initialization_torch_allocated_delta_bytes_per_rank": (
                stage_summaries["PLE offload initialized"]["memory_counters"][
                    "torch_allocated_bytes"
                ]["mean"]
                - stage_summaries["postprocessed"]["memory_counters"][
                    "torch_allocated_bytes"
                ]["mean"]
            ),
            "kv_allocation_torch_allocated_delta_bytes_per_rank": (
                stage_summaries["kv cache allocated"]["memory_counters"][
                    "torch_allocated_bytes"
                ]["mean"]
                - stage_summaries["profile complete"]["memory_counters"][
                    "torch_allocated_bytes"
                ]["mean"]
            ),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("raw_directory", type=Path)
    parser.add_argument("output", type=Path)
    arguments = parser.parse_args()
    summary = build_memory_baseline_summary(arguments.raw_directory)
    arguments.output.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    main()
