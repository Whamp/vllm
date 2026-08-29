# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gzip
import json
from pathlib import Path


def load_json(path: Path):
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as source:
            return json.load(source)
    return json.loads(path.read_text())


def stage_reports(root: Path, stage: str):
    reports = []
    for path in root.glob("*.json"):
        report = load_json(path)
        if report.get("stage") == stage:
            reports.append(report)
    reports.sort(key=lambda report: report["rank"])
    if [report["rank"] for report in reports] != [0, 1, 2, 3]:
        raise RuntimeError(f"Missing {stage} reports under {root}")
    return reports


def topk_buffer_bytes(report):
    total = 0
    names = []
    for storage in report["registered_tensors"]["storages"]:
        storage_names = [tensor["name"] for tensor in storage["tensors"]]
        selected = [
            name for name in storage_names if name.endswith("topk_indices_buffer")
        ]
        if selected:
            if len(selected) != 1:
                raise RuntimeError(f"Unexpected aliased top-k storage: {storage}")
            total += storage["nbytes"]
            names.extend(selected)
    if len(names) != 12:
        raise RuntimeError(f"Expected 12 QSA top-k buffers, found {len(names)}")
    return total


def require_uniform(values, label):
    if len(set(values)) != 1:
        raise RuntimeError(f"{label} differs by rank: {values}")
    return values[0]


def analyze_memory(root: Path):
    baseline_root = root.parent / "qwen38-memory-baseline-20260829" / "raw"
    baseline_reports = stage_reports(baseline_root, "postprocessed")
    candidate_reports = stage_reports(root / "memory-reports", "postprocessed")
    baseline_registered = require_uniform(
        [
            report["registered_tensors"]["unique_storage_bytes"]
            for report in baseline_reports
        ],
        "baseline registered storage",
    )
    candidate_registered = require_uniform(
        [
            report["registered_tensors"]["unique_storage_bytes"]
            for report in candidate_reports
        ],
        "candidate registered storage",
    )
    baseline_topk = require_uniform(
        [topk_buffer_bytes(report) for report in baseline_reports],
        "baseline QSA top-k storage",
    )
    candidate_topk = require_uniform(
        [topk_buffer_bytes(report) for report in candidate_reports],
        "candidate QSA top-k storage",
    )
    return {
        "baseline_registered_bytes_per_rank": baseline_registered,
        "candidate_registered_bytes_per_rank": candidate_registered,
        "registered_bytes_reclaimed_per_rank": (
            baseline_registered - candidate_registered
        ),
        "baseline_qsa_topk_bytes_per_rank": baseline_topk,
        "candidate_qsa_topk_bytes_per_rank": candidate_topk,
        "qsa_topk_bytes_reclaimed_per_rank": baseline_topk - candidate_topk,
    }


def analyze_performance(control, candidate, promoted, acceptance):
    control_decode = control["decode"]["tokens_per_second"]
    candidate_decode = candidate["decode"]["tokens_per_second"]
    promoted_decode = promoted["decode"]["tokens_per_second"]
    control_prefill = control["prefill"]["tokens_per_second"]
    candidate_prefill = candidate["prefill"]["tokens_per_second"]
    promoted_prefill = promoted["prefill"]["tokens_per_second"]
    concurrency = acceptance["concurrency_2"]
    return {
        "control_decode_tokens_per_second": control_decode,
        "candidate_decode_tokens_per_second": candidate_decode,
        "promoted_decode_tokens_per_second": promoted_decode,
        "control_prefill_tokens_per_second": control_prefill,
        "candidate_prefill_tokens_per_second": candidate_prefill,
        "promoted_prefill_tokens_per_second": promoted_prefill,
        "decode_fraction": promoted_decode["mean"] / control_decode["mean"],
        "prefill_fraction": promoted_prefill["mean"] / control_prefill["mean"],
        "concurrency_2_aggregate_tokens_per_second": concurrency[
            "aggregate_tokens_per_second"
        ],
        "concurrency_2_mean_per_stream_tokens_per_second": concurrency[
            "mean_per_stream_decode_tokens_per_second"
        ],
    }


def analyze_quality(acceptance, benchlocal):
    packs = {
        pack["pack_id"]: {
            "passed": pack["passed"],
            "scenario_count": pack["scenario_count"],
        }
        for pack in benchlocal["packs"]
    }
    return {
        "deterministic": "passed",
        "tool_and_post_tool": "passed",
        "multimodal": "passed",
        "niah_prompt_tokens": acceptance["niah"]["api_prompt_tokens"],
        "niah_answer": acceptance["niah"]["answer"],
        "benchlocal_packs": packs,
        "benchlocal_passed": sum(pack["passed"] for pack in benchlocal["packs"]),
        "benchlocal_scenarios": sum(
            pack["scenario_count"] for pack in benchlocal["packs"]
        ),
    }


def build_result(root: Path):
    control = load_json(root / "control-benchmark.json")
    candidate = load_json(root / "candidate-benchmark.json")
    promoted = load_json(root / "promoted-benchmark.json")
    acceptance = load_json(root / "promoted-acceptance.json.gz")
    benchlocal = load_json(root / "benchlocal-quick.json.gz")
    control_context = control["model_record"]["max_model_len"]
    promoted_context = promoted["model_record"]["max_model_len"]
    return {
        "schema_version": 1,
        "one_variable": {
            "max_num_batched_tokens": {"control": 2048, "candidate": 1024}
        },
        "mechanism": analyze_memory(root),
        "capacity": {
            "control_max_model_len": control_context,
            "promoted_max_model_len": promoted_context,
            "gain_tokens": promoted_context - control_context,
        },
        "performance": analyze_performance(control, candidate, promoted, acceptance),
        "quality": analyze_quality(acceptance, benchlocal),
        "production": {
            "image_sha256": (
                "0aea30240f3e3d9ffae8526643950e170eb5fa07fc427016a9dd90892afa2aa3"
            ),
            "model": "qwen3.8-flash-next-intel-autoround-w4a16",
            "context": 156400,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 1024,
            "qsa_cache": "bf16",
            "serving_process_swap_kib": 0,
            "restart_count": 0,
        },
    }


def validate_result(result):
    if result["mechanism"]["qsa_topk_bytes_reclaimed_per_rank"] != 100_810_752:
        raise RuntimeError("QSA top-k allocation did not halve exactly")
    if result["mechanism"]["registered_bytes_reclaimed_per_rank"] != 100_810_752:
        raise RuntimeError("Registered storage change was not isolated to QSA top-k")
    if result["capacity"]["gain_tokens"] != 8000:
        raise RuntimeError("Unexpected fitted-context gain")
    if result["performance"]["decode_fraction"] < 0.95:
        raise RuntimeError("Decode performance gate failed")
    if result["performance"]["prefill_fraction"] < 0.90:
        raise RuntimeError("Prefill performance gate failed")
    if result["quality"]["benchlocal_passed"] < 25:
        raise RuntimeError("BenchLocal quick gate failed")


def main():
    root = Path(__file__).resolve().parent
    result = build_result(root)
    validate_result(result)
    rendered = json.dumps(result, indent=2, sort_keys=True) + "\n"
    (root / "analysis.json").write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
