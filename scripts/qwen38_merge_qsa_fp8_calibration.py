#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Merge Qwen3.8 QSA FP8 calibration reports into a scale file."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from vllm.models.qwen4_exp.common.qsa_fp8 import (
    merge_qsa_fp8_calibration_reports,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reports", nargs="+", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--safety-margin", required=True, type=float)
    parser.add_argument("--expected-ranks", required=True, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.expected_ranks <= 0:
        raise ValueError("QSA FP8 expected rank count must be positive")
    if len(args.reports) != args.expected_ranks:
        raise ValueError(
            f"QSA FP8 calibration expected {args.expected_ranks} reports, "
            f"received {len(args.reports)}"
        )
    observed_ranks = set()
    for report_path in args.reports:
        report = json.loads(report_path.read_text())
        rank = report.get("rank") if isinstance(report, dict) else None
        if not isinstance(rank, int) or isinstance(rank, bool):
            raise ValueError(
                f"QSA FP8 calibration report {report_path} has invalid rank"
            )
        observed_ranks.add(rank)
    expected_ranks = set(range(args.expected_ranks))
    if observed_ranks != expected_ranks:
        raise ValueError(
            "QSA FP8 calibration reports have ranks "
            f"{sorted(observed_ranks)}, expected {sorted(expected_ranks)}"
        )

    merged = merge_qsa_fp8_calibration_reports(
        args.reports,
        safety_margin=args.safety_margin,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = args.output.with_suffix(args.output.suffix + ".tmp")
    temporary_path.write_text(json.dumps(merged, indent=2, sort_keys=True) + "\n")
    os.replace(temporary_path, args.output)


if __name__ == "__main__":
    main()
