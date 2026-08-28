#!/usr/bin/env python3
"""Tests for the GGUF-TP cold-expert route-skew decision engine."""

from __future__ import annotations

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("analyze_route_skew.py")
SPEC = importlib.util.spec_from_file_location("analyze_route_skew", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
route_skew = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(route_skew)


def counts_reaching_h99_at(hot_count: int) -> list[int]:
    """Construct counts whose smallest 99%-coverage set is hot_count."""
    if hot_count < 2 or hot_count > 248:
        raise ValueError("hot_count must be in [2, 248] for this fixture")
    counts = [0] * 256
    # Keep every cold expert colder than every hot expert so sorting does not
    # change the intended boundary. The top H then total exactly 99%.
    hot_base, hot_remainder = divmod(99_000, hot_count)
    for expert in range(hot_count):
        counts[expert] = hot_base + (expert < hot_remainder)
    cold_count = 256 - hot_count
    cold_base, cold_remainder = divmod(1_000, cold_count)
    for offset in range(cold_count):
        counts[hot_count + offset] = cold_base + (offset < cold_remainder)
    return counts


def make_workload(workload_id: str, h99: int, routes=None):
    """Build one 43-layer synthetic workload."""
    return {
        "schema_version": 1,
        "workload_id": workload_id,
        "n_experts": 256,
        "top_k": 6,
        "layers": [
            {
                "layer": layer,
                "counts": counts_reaching_h99_at(h99),
                **({"routes": routes} if routes is not None else {}),
            }
            for layer in range(43)
        ],
        "_path": f"{workload_id}.json",
        "_sha256": "0" * 64,
    }


class RouteSkewAnalysisTests(unittest.TestCase):
    def test_go_requires_all_workloads_at_or_below_224(self):
        analysis = route_skew.analyze_route_workloads(
            [make_workload("pilot", 220), make_workload("coding", 224)]
        )
        self.assertEqual(analysis["decision"], "GO")
        self.assertEqual(analysis["maximum_h99"], 224)

    def test_no_go_when_any_layer_requires_248(self):
        pilot = make_workload("pilot", 224)
        coding = make_workload("coding", 224)
        coding["layers"][20]["counts"] = counts_reaching_h99_at(248)
        analysis = route_skew.analyze_route_workloads([pilot, coding])
        self.assertEqual(analysis["decision"], "NO-GO")
        self.assertEqual(analysis["maximum_h99"], 248)

    def test_intermediate_gate_is_inconclusive(self):
        analysis = route_skew.analyze_route_workloads(
            [make_workload("pilot", 224), make_workload("coding", 240)]
        )
        self.assertEqual(analysis["decision"], "INCONCLUSIVE")
        self.assertEqual(analysis["maximum_h99"], 240)

    def test_partial_passing_layers_cannot_produce_go(self):
        pilot = make_workload("pilot", 224)
        coding = make_workload("coding", 224)
        pilot["layers"] = pilot["layers"][:3]
        coding["layers"] = coding["layers"][:3]
        analysis = route_skew.analyze_route_workloads([pilot, coding])
        self.assertEqual(analysis["decision"], "INCONCLUSIVE")
        self.assertEqual(analysis["observed_layer_count"], 3)

    def test_partial_failing_layer_can_produce_no_go(self):
        pilot = make_workload("pilot", 248)
        coding = make_workload("coding", 224)
        pilot["layers"] = pilot["layers"][:3]
        coding["layers"] = coding["layers"][:3]
        analysis = route_skew.analyze_route_workloads([pilot, coding])
        self.assertEqual(analysis["decision"], "NO-GO")
        self.assertEqual(analysis["maximum_h99"], 248)

    def test_consecutive_reuse_and_lru_use_pre_token_cache_state(self):
        routes = [
            [0, 1, 2, 3, 4, 5],
            [0, 1, 2, 3, 4, 6],
            [10, 11, 12, 13, 14, 15],
        ]
        metrics = route_skew.consecutive_route_metrics(routes)
        self.assertEqual(metrics["transition_count"], 2)
        self.assertAlmostEqual(metrics["mean_consecutive_overlap"], 5 / 12)
        self.assertEqual(metrics["exact_set_repeat_rate"], 0)
        # A large cache has five hits on the second token and none on the third.
        self.assertAlmostEqual(metrics["lru_hit_rate"]["224"], 5 / 18)

    def test_cross_workload_hot_set_transfer_is_directional(self):
        left = make_workload("left", 224)
        right = make_workload("right", 224)
        # Shift every right-workload hot set by 16 expert IDs.
        shifted = left["layers"][0]["counts"][-16:] + left["layers"][0]["counts"][:-16]
        for layer in right["layers"]:
            layer["counts"] = shifted[:]
        analysis = route_skew.analyze_route_workloads([left, right])
        stability = analysis["cross_workload_stability"][0]["layers"][0]["224"]
        self.assertLess(stability["jaccard"], 1)
        self.assertLess(stability["left_hot_set_coverage_on_right"], 0.99)
        self.assertLess(stability["right_hot_set_coverage_on_left"], 0.99)

    def test_loader_rejects_repeated_expert_in_one_token(self):
        workload = make_workload("bad", 224, routes=[[0, 0, 1, 2, 3, 4]])
        workload.pop("_path")
        workload.pop("_sha256")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bad.json"
            path.write_text(json.dumps(workload), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "repeats experts"):
                route_skew.load_workload_summary(path)


if __name__ == "__main__":
    unittest.main()
