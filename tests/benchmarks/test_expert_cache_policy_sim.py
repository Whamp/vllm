# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Tests for the offline expert-cache policy simulator.

The simulator replays captured routed-expert traces against different
cache policies (LRU, FIFO, LFU, LFRU, Belady) to compare hit rates before
any GPU work is attempted. All logic is pure Python/numpy.
"""

from pathlib import Path

import numpy as np
import pytest

from vllm.benchmarks.expert_cache_sim import (
    POLICY_NAMES,
    PolicySimResult,
    simulate_expert_cache,
    write_routing_trace,
)


def _sequential_trace(num_steps: int, num_layers: int = 2, top_k: int = 1):
    """Each step touches exactly one expert; steps walk forward through
    the expert space so LRU, FIFO, and Belady are all exactly determined."""
    routing = np.arange(num_steps, dtype=np.int32)[:, None, None].repeat(
        num_layers, axis=1
    )
    return routing


class TestSimulateExpertCache:
    def test_unknown_policy_rejected(self):
        with pytest.raises(ValueError, match="unknown expert-cache policy"):
            simulate_expert_cache(np.zeros((4, 1, 1), dtype=np.int32), 4, "mru")

    def test_result_shape_and_counts(self):
        result = simulate_expert_cache(_sequential_trace(8), num_slots=4, policy="lru")
        assert isinstance(result, PolicySimResult)
        assert result.policy == "lru"
        assert result.num_slots == 4
        # 2 layers x 8 steps = 16 touches; every miss is distinct at first,
        # then capacity-4 thrashing on a sequential scan yields zero reuse.
        assert result.touches == 16
        assert result.hits == 0
        assert result.hit_rate == pytest.approx(0.0)

    def test_working_set_within_capacity_is_all_hits_after_warmup(self):
        # 4 experts revisited cyclically fit entirely in an 8-slot cache.
        routing = (
            np.arange(4, dtype=np.int32)[None, :, None]
            .repeat(12, axis=0)
            .reshape(12, 4, 1)
        )
        result = simulate_expert_cache(routing, num_slots=8, policy="lru")
        # Step 0 warms 4 misses; every later touch hits.
        assert result.misses == 4
        assert result.hits == 44
        assert result.hit_rate == pytest.approx(44 / 48)

    def test_lru_beats_fifo_when_rehits_protect_recency(self):
        # Cache 2 over the access sequence A B A C A: the rehit on A makes
        # it most-recently used, so LRU evicts B at C and hits A again;
        # FIFO's insertion order evicts A instead and loses that hit.
        ids = {"A": 0, "B": 1, "C": 2}
        sequence = ["A", "B", "A", "C", "A"]
        routing = np.array([[[ids[t]]] for t in sequence], dtype=np.int32)
        lru = simulate_expert_cache(routing, num_slots=2, policy="lru")
        fifo = simulate_expert_cache(routing, num_slots=2, policy="fifo")
        assert lru.hits == 2
        assert fifo.hits == 1

    def test_belady_is_upper_bound(self):
        rng = np.random.default_rng(7)
        routing = rng.integers(0, 16, size=(64, 4, 2)).astype(np.int32)
        results = {
            name: simulate_expert_cache(routing, 16, name) for name in POLICY_NAMES
        }
        belady = results["belady"]
        for name, result in results.items():
            assert result.hits <= belady.hits, f"{name} beat the oracle"

    def test_oversubscribed_step_streams_without_asserting(self):
        # Five distinct experts against a 2-slot cache: the step is served
        # streaming (lookups counted, nothing inserted) and later steps on
        # a small working set still behave normally.
        routing = np.array(
            [[[0], [1], [2], [3], [4]], [[0], [0], [0], [0], [0]]],
            dtype=np.int32,
        )
        result = simulate_expert_cache(routing, num_slots=2, policy="lru")
        assert result.touches == 10
        # Step 1 touches expert 0 five times; it was streamed away, so all
        # five lookups miss.
        assert result.hits == 0

    def test_lfru_protects_hot_experts(self):
        # Cache 3 with a hot expert used every fourth step and three fresh
        # cold experts between uses. Recency alone fails: by the hot
        # expert's next arrival LRU has flooded it out with cold traffic,
        # while LFRU's hit-count protection retains it.
        steps: list[list[int]] = []
        cold_id = 1
        for _ in range(8):
            steps.append([0])
            for _ in range(3):
                steps.append([cold_id])
                cold_id += 1
        routing = np.array([[[e]] for e in steps], dtype=np.int32)
        lfru = simulate_expert_cache(routing, num_slots=3, policy="lfru")
        lru = simulate_expert_cache(routing, num_slots=3, policy="lru")
        assert lfru.hits > lru.hits

    def test_trace_round_trip(self, tmp_path):
        routing = _sequential_trace(6)
        path = tmp_path / "trace.npz"
        write_routing_trace(path, routing)
        from vllm.benchmarks.expert_cache_sim import load_routing_trace

        loaded = load_routing_trace(path)
        assert np.array_equal(loaded.routing_data, routing)

    def test_load_rejects_missing_key(self, tmp_path):
        path = tmp_path / "bad.npz"
        np.savez(path, unrelated=np.zeros(3))
        from vllm.benchmarks.expert_cache_sim import load_routing_trace

        with pytest.raises(ValueError, match="missing 'routing_data'"):
            load_routing_trace(path)

    @pytest.mark.parametrize(
        "routing",
        [np.zeros((4, 8), dtype=np.int32), np.zeros((2, 3, 4, 5), np.int32)],
    )
    def test_load_rejects_wrong_rank(self, tmp_path, routing):
        path = tmp_path / "badrank.npz"
        np.savez(path, routing_data=routing)
        from vllm.benchmarks.expert_cache_sim import load_routing_trace

        with pytest.raises(ValueError, match=r"expected rank 3.*got rank"):
            load_routing_trace(path)

    def test_load_rejects_float_dtype(self, tmp_path):
        path = tmp_path / "float.npz"
        np.savez(path, routing_data=np.zeros((2, 3, 4), dtype=np.float32))
        from vllm.benchmarks.expert_cache_sim import load_routing_trace

        with pytest.raises(ValueError, match="integer expert ids"):
            load_routing_trace(path)


class TestPolicySimCli:
    def test_synthetic_trace_is_seeded_and_in_range(self):
        from vllm.benchmarks.expert_cache_sim import synth_trace

        a = synth_trace(50, 4, 32, 2, 0.3, seed=11)
        b = synth_trace(50, 4, 32, 2, 0.3, seed=11)
        assert np.array_equal(a, b)
        assert a.shape == (50, 4, 2)
        assert a.min() >= 0 and a.max() < 32

    def test_results_table_orders_by_hits(self):
        from vllm.benchmarks.expert_cache_sim import format_results_table

        results = [
            PolicySimResult("lru", 8, 100, 40),
            PolicySimResult("belady", 8, 100, 60),
            PolicySimResult("fifo", 8, 100, 20),
        ]
        table = format_results_table(results, num_slots=8)
        lines = table.splitlines()
        assert lines[0].startswith("| policy | slots | touches | hits |")
        assert "belady" in lines[2]
        assert "fifo" in lines[-1]

    def test_cli_runs_end_to_end(self, tmp_path):
        import os
        import subprocess
        import sys

        repo_root = Path(__file__).resolve().parents[2]
        env = {
            **os.environ,
            "PYTHONPATH": str(repo_root)
            + os.pathsep
            + os.environ.get("PYTHONPATH", ""),
        }
        result = subprocess.run(
            [
                sys.executable,
                str(repo_root / "benchmarks" / "expert_cache_policy_sim.py"),
                "--slots",
                "64",
                "--num-steps",
                "40",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env=env,
        )
        assert result.returncode == 0, result.stderr
        assert "| policy | slots | touches | hits | misses | hit rate |" in (
            result.stdout
        )
