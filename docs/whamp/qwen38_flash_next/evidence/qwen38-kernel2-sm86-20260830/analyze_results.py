# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Reproduce the Qwen3.8 SM86 Kernel2 experiment decision summary."""

from __future__ import annotations

import csv
import gzip
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
REQUIRED_SAVINGS_MS = 0.8


def load_json(path: Path) -> dict[str, Any]:
    """Load one experiment JSON object."""

    value = json.loads(path.read_text())
    if not isinstance(value, dict):
        raise TypeError(f"Kernel2 evidence must contain a JSON object: {path}")
    return value


def summarize_marlin() -> dict[str, Any]:
    """Summarize the weight-only Marlin experiment matrix."""

    rows = []
    for path in sorted((ROOT / "marlin").glob("*.json")):
        result = load_json(path)
        candidate_name = f"w{result['plan']['bits']}a16_marlin"
        rows.append(
            {
                "case": path.stem,
                "projection": result["projection"]["name"],
                "tokens": result["tokens"],
                "bits": result["plan"]["bits"],
                "group_size": result["plan"]["group_size"],
                "baseline_us": result["timing"]["bf16_cutlass"]["median_us"],
                "candidate_us": result["timing"][candidate_name]["median_us"],
                "speedup": result["candidate_speedup"],
                "projected_savings_ms": result[
                    "projected_projection_savings_ms_per_decode_step"
                ],
                "random_weight_nrmse": result["marlin_quantization_error"][
                    "normalized_rmse"
                ],
                "random_weight_cosine": result["marlin_quantization_error"]["cosine"],
                "execution_nrmse": result["marlin_execution_error"]["normalized_rmse"],
                "graph_deterministic": result["candidate_graph_bitwise_deterministic"],
            }
        )

    def find(
        *, projection: str, tokens: int, bits: int, group_size: int
    ) -> dict[str, Any]:
        matches = [
            row
            for row in rows
            if row["projection"] == projection
            and row["tokens"] == tokens
            and row["bits"] == bits
            and row["group_size"] == group_size
        ]
        if len(matches) != 1:
            raise RuntimeError(f"Kernel2 Marlin evidence match count: {len(matches)}")
        return matches[0]

    combinations = {}
    for tokens in (1, 2):
        w8_down = find(
            projection="merged_down_injection",
            tokens=tokens,
            bits=8,
            group_size=128,
        )
        w8_up = find(projection="up", tokens=tokens, bits=8, group_size=-1)
        w4_up = find(projection="up", tokens=tokens, bits=4, group_size=32)
        combinations[f"m{tokens}_w8_both_savings_ms"] = (
            w8_down["projected_savings_ms"] + w8_up["projected_savings_ms"]
        )
        combinations[f"m{tokens}_bf16_down_w4_up_savings_ms"] = w4_up[
            "projected_savings_ms"
        ]
    return {"rows": rows, "combinations": combinations}


def summarize_native_bf16() -> dict[str, Any]:
    """Select the fastest valid native BF16 plan for each projection and M."""

    rows = []
    for path in sorted((ROOT / "native-bf16").glob("*.json")):
        result = load_json(path)
        rows.append(
            {
                "case": path.stem,
                "projection": result["projection"]["name"],
                "tokens": result["tokens"],
                "block_threads": result["plan"]["block_threads"],
                "outputs_per_block": result["plan"]["outputs_per_block"],
                "baseline_us": result["timing"]["bf16_cutlass"]["median_us"],
                "candidate_us": result["timing"]["bf16_sm86_native"]["median_us"],
                "speedup": result["candidate_speedup"],
                "projected_savings_ms": result[
                    "projected_projection_savings_ms_per_decode_step"
                ],
                "candidate_gb_per_s": result["timing"]["bf16_sm86_native"][
                    "logical_weight_gb_per_s"
                ],
                "nrmse": result["candidate_vs_baseline"]["normalized_rmse"],
                "cosine": result["candidate_vs_baseline"]["cosine"],
                "graph_deterministic": result["candidate_graph_bitwise_deterministic"],
            }
        )

    best_rows = []
    combined = {}
    for tokens in (1, 2):
        selected = []
        for projection in ("merged_down_injection", "up"):
            candidates = [
                row
                for row in rows
                if row["tokens"] == tokens and row["projection"] == projection
            ]
            if not candidates:
                raise RuntimeError(
                    f"Kernel2 native BF16 evidence missing M={tokens} {projection}"
                )
            best = min(candidates, key=lambda row: row["candidate_us"])
            best_rows.append(best)
            selected.append(best)
        combined[f"m{tokens}_savings_ms"] = sum(
            row["projected_savings_ms"] for row in selected
        )
    return {"rows": rows, "best_rows": best_rows, "combined": combined}


def summarize_gpu_telemetry(path: Path) -> dict[str, float | int]:
    """Summarize GPU-0 telemetry for one benchmark family."""

    rows = []
    with gzip.open(path, "rt") as source:
        for row in csv.DictReader(source):
            if row["index"].strip() == "0":
                rows.append(row)
    return {
        "samples": len(rows),
        "maximum_sm_clock_mhz": max(float(row["sm_clock_mhz"]) for row in rows),
        "maximum_power_w": max(float(row["power_w"]) for row in rows),
        "maximum_temperature_c": max(float(row["temp_c"]) for row in rows),
    }


def main() -> None:
    """Recompute and atomically write the complete decision summary."""

    marlin = summarize_marlin()
    native = summarize_native_bf16()
    w4_real = load_json(ROOT / "w4-up-real-screen.json")
    with gzip.open(ROOT / "cute-sm86-compile-failure.log.gz", "rt") as source:
        cute_failure = source.read()
    summary = {
        "schema_version": 1,
        "required_combined_savings_ms_per_generated_token": REQUIRED_SAVINGS_MS,
        "cute_dsl": {
            "sm86_supported": False,
            "compile_failure_confirmed": "CONFIG_UNSUPPORTED_ARCH" in cute_failure,
        },
        "marlin": marlin,
        "native_bf16": native,
        "w4_real_weight_screen": w4_real["schemes"],
        "telemetry": {
            "marlin": summarize_gpu_telemetry(ROOT / "marlin" / "telemetry.csv.gz"),
            "native_bf16": summarize_gpu_telemetry(
                ROOT / "native-bf16" / "telemetry.csv.gz"
            ),
        },
        "decision": {
            "status": "NO_GO",
            "production_dispatch_changed": False,
            "reasons": [
                "CuTe skinny GEMM does not compile for sm_86",
                "W8A16 combined projection timing is a net loss",
                "all 97 W4 up tensors fail the real-weight numerical gate",
                "native BF16 saves less than 0.8 ms at M=1 and M=2",
            ],
        },
    }
    output = ROOT / "summary.json"
    temporary = output.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)


if __name__ == "__main__":
    main()
