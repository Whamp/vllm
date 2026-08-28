#!/usr/bin/env python3
"""Tests for exact static tid2eid route summary construction."""

from __future__ import annotations

import importlib.util
import unittest
from array import array
from pathlib import Path

MODULE_PATH = Path(__file__).with_name("build_static_route_workload.py")
SPEC = importlib.util.spec_from_file_location(
    "build_static_route_workload", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
static_routes = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(static_routes)


class StaticRouteWorkloadTests(unittest.TestCase):
    def test_session_boundaries_reset_consecutive_and_lru_state(self):
        table = array("i", [0] * (static_routes.VOCAB_SIZE * static_routes.TOP_K))
        table[0:6] = array("i", [0, 1, 2, 3, 4, 5])
        table[6:12] = array("i", [0, 1, 2, 3, 4, 6])
        table[12:18] = array("i", [10, 11, 12, 13, 14, 15])
        counts, reuse = static_routes.summarize_static_layer_routes(
            table, [[0, 1], [2]]
        )
        self.assertEqual(sum(counts), 18)
        self.assertEqual(reuse["token_count"], 3)
        self.assertEqual(reuse["transition_count"], 1)
        self.assertAlmostEqual(reuse["mean_consecutive_overlap"], 5 / 6)
        self.assertEqual(reuse["exact_set_repeat_rate"], 0)
        # The second session starts cold, so token 2 cannot hit token 1's routes.
        self.assertAlmostEqual(reuse["lru_hit_rate"]["224"], 5 / 18)
        self.assertAlmostEqual(reuse["lru_hit_rate"]["248"], 5 / 18)


if __name__ == "__main__":
    unittest.main()
