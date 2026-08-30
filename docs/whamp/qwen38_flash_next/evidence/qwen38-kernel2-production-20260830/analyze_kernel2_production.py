#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Recompute the accepted Qwen3.8 Kernel2 production result."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

EVIDENCE_DIR = Path(__file__).resolve().parent


def load_evidence_json(filename: str) -> dict[str, Any]:
    """Load one Kernel2 evidence JSON object."""
    value = json.loads((EVIDENCE_DIR / filename).read_text())
    if not isinstance(value, dict):
        raise ValueError(f"Kernel2 evidence must be a JSON object: {filename}")
    return value


def read_matrix_mean(
    matrix: dict[str, Any], concurrency: int, phase: str, metric: str
) -> float:
    """Read one aggregate mean from a service benchmark matrix."""
    return float(matrix["concurrency"][str(concurrency)][phase][metric]["mean"])


def percent_change(candidate: float, baseline: float) -> float:
    """Return the candidate percentage change from its baseline."""
    return (candidate / baseline - 1.0) * 100.0


def build_kernel2_production_summary() -> dict[str, Any]:
    """Build the deterministic Kernel2 production acceptance summary."""
    prior_image = load_evidence_json("baseline-matrix.json")
    kernel2_enabled = load_evidence_json("candidate-matrix.json")
    same_image_disabled = load_evidence_json("ablation-matrix.json")
    final_benchmark = load_evidence_json("final-production-benchmark.json")
    benchlocal = load_evidence_json("benchlocal-quick.json")

    metrics: dict[str, dict[str, float]] = {}
    for concurrency in (1, 2):
        for phase, metric in (
            ("decode", "aggregate_decode_tokens_per_second"),
            ("prefill", "aggregate_prompt_tokens_per_second"),
        ):
            key = f"c{concurrency}_{phase}"
            enabled = read_matrix_mean(kernel2_enabled, concurrency, phase, metric)
            disabled = read_matrix_mean(same_image_disabled, concurrency, phase, metric)
            prior = read_matrix_mean(prior_image, concurrency, phase, metric)
            metrics[key] = {
                "kernel2_enabled": enabled,
                "same_image_disabled": disabled,
                "same_image_change_percent": percent_change(enabled, disabled),
                "prior_image": prior,
                "prior_image_change_percent": percent_change(enabled, prior),
            }

    return {
        "schema_version": 1,
        "decision": "PROMOTE",
        "source_commit": "42b918e36fa3bdd04e3d7bd7ad4a9c7695b9624f",
        "production_image": (
            "sha256:acff9d8e08096a2265b23e50f5ff0d52a3f1e95ffa91e2fb099346e274a9b735"
        ),
        "stable_extension_sha256": (
            "91118abf4f8b94e1b41dc4226dfd1ef9cf32bd69a610156543c649b57e523381"
        ),
        "metrics": metrics,
        "exact_final": {
            "decode_tokens_per_second": float(
                final_benchmark["decode"]["tokens_per_second"]["mean"]
            ),
            "prefill_tokens_per_second": float(
                final_benchmark["prefill"]["tokens_per_second"]["mean"]
            ),
        },
        "benchlocal_quick": {
            "passed": int(benchlocal["totals"]["passed"]),
            "total": int(benchlocal["totals"]["total"]),
        },
    }


if __name__ == "__main__":
    print(json.dumps(build_kernel2_production_summary(), indent=2, sort_keys=True))
