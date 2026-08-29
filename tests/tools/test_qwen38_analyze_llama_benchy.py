# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest

from scripts.qwen38_analyze_llama_benchy import summarize_llama_benchy


def _result(concurrency: int, cold_tps: float, cached_tps: float) -> dict:
    def phase(cold: bool, aggregate: float) -> dict:
        return {
            "is_context_prefill_phase": cold,
            "tg_throughput": {"mean": aggregate},
            "tg_req_throughput": {
                "mean": aggregate / concurrency,
                "values": [aggregate / concurrency] * concurrency,
            },
            "ttfr": {"values": [21_000.0, 40_000.0][:concurrency]},
        }

    return {
        "version": "test",
        "model": "qwen-test",
        "max_concurrency": concurrency,
        "benchmarks": [phase(True, cold_tps), phase(False, cached_tps)],
    }


def _write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data))


def test_summarize_llama_benchy_separates_cold_and_cached_decode(
    tmp_path: Path,
) -> None:
    c1_path = tmp_path / "c1.json"
    c2_path = tmp_path / "c2.json"
    _write_json(c1_path, _result(1, 48.0, 48.0))
    _write_json(c2_path, _result(2, 11.0, 66.0))

    summary = summarize_llama_benchy(
        c1_path,
        c2_path,
        alesha_decode_tokens_per_second=64.0,
    )

    mixed = summary["cold_c2_mixed_prefill_decode"]
    assert mixed["aggregate_tokens_per_second"] == 11.0
    assert "not steady decode" in mixed["interpretation"]
    cached = summary["cached_decode_scaling"]
    assert cached["aggregate_speedup"] == 66.0 / 48.0
    assert cached["parallel_efficiency"] == 66.0 / 96.0
    budget = summary["decode_causal_budget"]
    assert budget["local_ms_per_token"] == 1000.0 / 48.0
    assert budget["gap_ms_per_token"] == 1000.0 / 48.0 - 1000.0 / 64.0


def test_summarize_llama_benchy_rejects_mismatched_model(tmp_path: Path) -> None:
    c1_path = tmp_path / "c1.json"
    c2_path = tmp_path / "c2.json"
    _write_json(c1_path, _result(1, 48.0, 48.0))
    c2 = _result(2, 11.0, 66.0)
    c2["model"] = "other-model"
    _write_json(c2_path, c2)

    with pytest.raises(ValueError, match="model does not match"):
        summarize_llama_benchy(
            c1_path,
            c2_path,
            alesha_decode_tokens_per_second=64.0,
        )
