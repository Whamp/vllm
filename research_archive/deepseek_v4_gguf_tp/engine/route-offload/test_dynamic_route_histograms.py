#!/usr/bin/env python3
"""Tests for rank-safe dynamic route histogram extraction."""

from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import torch  # ty: ignore[unresolved-import]

MODULE_PATH = Path(__file__).with_name("extract_dynamic_route_histograms.py")
SPEC = importlib.util.spec_from_file_location(
    "extract_dynamic_route_histograms", MODULE_PATH
)
assert SPEC is not None and SPEC.loader is not None
dynamic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(dynamic)


class DynamicRouteHistogramTests(unittest.TestCase):
    def test_merge_boundary_and_subtract_complete_workload(self):
        startup = torch.tensor([[10, 20], [30, 40], [50, 60]])
        start_a = startup.clone()
        start_a[1:] -= 1
        start_b = startup.clone()
        start_b[0] += 1
        workload = torch.tensor([[3, 3], [3, 3], [3, 3]])
        end_expected = startup + workload
        end_a = end_expected.clone()
        end_a[1:] = startup[1:]
        end_b = end_expected.clone()
        end_b[0] += 1

        start = dynamic.merge_route_histogram_boundary(start_a, start_b)
        end = dynamic.merge_route_histogram_boundary(end_a, end_b)
        observed = dynamic.subtract_route_workload(start, end, top_k=2)

        self.assertTrue(torch.equal(start, startup))
        self.assertTrue(torch.equal(observed, workload))
        self.assertEqual(observed.sum(dim=1).tolist(), [6, 6, 6])

    def test_subtraction_rejects_incomplete_layer_rows(self):
        start = torch.zeros((2, 4), dtype=torch.int64)
        end = start.clone()
        end[0, :2] = 1
        with self.assertRaisesRegex(ValueError, "different token-row totals"):
            dynamic.subtract_route_workload(start, end, top_k=2)


if __name__ == "__main__":
    unittest.main()
