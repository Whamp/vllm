# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for bandwidth-profile-driven expert/KV budget planning.

These planners decide how scarce GPU memory is divided between an expert
cache and the KV pool, and how a decode-step cache miss is divided between
H2D fetches and direct host-side expert execution. All logic is pure
integer/float arithmetic and runs without a GPU.
"""

import json

import pytest

from vllm.model_executor.offloader.bandwidth_profile import (
    HybridBandwidthProfile,
    load_bandwidth_profile,
    profile_matches_hardware,
    save_bandwidth_profile,
)
from vllm.model_executor.offloader.hybrid_budget import (
    balanced_miss_split,
    plan_expert_kv_budget,
)


def _rtx3090_epyc_profile() -> HybridBandwidthProfile:
    return HybridBandwidthProfile(
        gpu_name="NVIDIA GeForce RTX 3090",
        cpu_model="AMD EPYC 7302P",
        interconnect="PCIe-Gen4-x16",
        quant_format="gguf_iq2_xxs",
        host_moe_gbps=90.0,
        pcie_h2d_gbps=26.0,
    )


class TestBandwidthProfileArtifact:
    def test_json_round_trip(self, tmp_path):
        path = tmp_path / "bw.json"
        save_bandwidth_profile(_rtx3090_epyc_profile(), path)
        loaded = load_bandwidth_profile(path)
        assert loaded == _rtx3090_epyc_profile()

    def test_rejects_schema_mismatch(self, tmp_path):
        path = tmp_path / "bw.json"
        raw = json.loads(_rtx3090_epyc_profile().model_dump_json())
        raw["schema_version"] = 999
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="unsupported bandwidth profile"):
            load_bandwidth_profile(path)

    def test_rejects_missing_field(self, tmp_path):
        path = tmp_path / "bw.json"
        raw = json.loads(_rtx3090_epyc_profile().model_dump_json())
        del raw["pcie_h2d_gbps"]
        path.write_text(json.dumps(raw))
        with pytest.raises(ValueError, match="invalid bandwidth profile"):
            load_bandwidth_profile(path)

    def test_rejects_nonpositive_bandwidth(self):
        with pytest.raises(ValueError, match="greater than 0"):
            HybridBandwidthProfile(
                gpu_name="g",
                cpu_model="c",
                interconnect="i",
                quant_format="q",
                host_moe_gbps=0.0,
                pcie_h2d_gbps=26.0,
            )

    def test_match_requires_every_supplied_identity_field(self):
        profile = _rtx3090_epyc_profile()
        assert profile_matches_hardware(
            profile,
            gpu_name=profile.gpu_name,
            cpu_model=profile.cpu_model,
            interconnect=profile.interconnect,
            quant_format=profile.quant_format,
        )
        assert not profile_matches_hardware(
            profile,
            gpu_name="NVIDIA GeForce RTX 4090",
            cpu_model=profile.cpu_model,
        )
        # A weaker key (GPU name only) must not silently pass on a
        # different machine: every field the caller supplies participates.
        assert not profile_matches_hardware(
            profile,
            gpu_name=profile.gpu_name,
            cpu_model="Intel Xeon Platinum 8358",
        )
        # Supplying nothing matches nothing.
        assert not profile_matches_hardware(profile)

    def test_match_can_ignore_optional_fields(self):
        profile = _rtx3090_epyc_profile()
        assert profile_matches_hardware(
            profile,
            gpu_name=profile.gpu_name,
            cpu_model=profile.cpu_model,
            interconnect=None,
            quant_format=None,
        )


class TestBalancedMissSplit:
    def test_equal_bandwidth_splits_evenly(self):
        profile = HybridBandwidthProfile(
            gpu_name="g",
            cpu_model="c",
            interconnect="i",
            quant_format="q",
            host_moe_gbps=100.0,
            pcie_h2d_gbps=100.0,
        )
        assert balanced_miss_split(8, 1024, profile) == (4, 4)

    def test_fast_pcie_shifts_work_toward_fetch(self):
        profile = HybridBandwidthProfile(
            gpu_name="g",
            cpu_model="c",
            interconnect="i",
            quant_format="q",
            host_moe_gbps=75.0,
            pcie_h2d_gbps=25.0,
        )
        # Balanced point is 25/(25+75)=0.25 of misses fetched.
        fetched, hosted = balanced_miss_split(100, 1024, profile)
        assert fetched == 25
        assert hosted == 75

    def test_split_partitions_all_misses(self):
        profile = _rtx3090_epyc_profile()
        for n in range(0, 64):
            fetched, hosted = balanced_miss_split(n, 512, profile)
            assert fetched + hosted == n
            assert fetched >= 0 and hosted >= 0

    def test_zero_misses_is_noop(self):
        assert balanced_miss_split(0, 512, _rtx3090_epyc_profile()) == (0, 0)

    def test_negative_misses_rejected(self):
        with pytest.raises(ValueError, match="non-negative"):
            balanced_miss_split(-1, 512, _rtx3090_epyc_profile())


class TestPlanExpertKvBudget:
    def test_kv_floor_first_then_experts(self):
        kv_page_bytes = 4096
        per_expert_bytes = 1 << 20
        budget = 64 << 20
        slots, pages = plan_expert_kv_budget(
            budget_bytes=budget,
            per_expert_bytes=per_expert_bytes,
            kv_page_bytes=kv_page_bytes,
            num_experts=32,
            kv_floor_pages=2048,
        )
        # The 8 MiB KV floor is honored before any expert spend.
        assert pages >= 2048
        assert slots * per_expert_bytes + pages * kv_page_bytes <= budget
        # Leftover after capping experts at num_experts flows back to KV.
        expected_slots = (budget - 2048 * kv_page_bytes) // per_expert_bytes
        assert slots == min(expected_slots, 32)

    def test_zero_expert_slots_still_serves_context(self):
        slots, pages = plan_expert_kv_budget(
            budget_bytes=1 << 20,
            per_expert_bytes=1 << 20,
            kv_page_bytes=4096,
            num_experts=8,
            kv_floor_pages=128,
        )
        assert slots == 0
        assert pages >= 128

    def test_budget_below_kv_floor_raises(self):
        with pytest.raises(ValueError, match="KV context floor"):
            plan_expert_kv_budget(
                budget_bytes=100,
                per_expert_bytes=1024,
                kv_page_bytes=4096,
                num_experts=8,
                kv_floor_pages=128,
            )

    def test_min_expert_slots_enforced_when_affordable(self):
        slots, pages = plan_expert_kv_budget(
            budget_bytes=(256 << 20),
            per_expert_bytes=(1 << 20),
            kv_page_bytes=4096,
            num_experts=64,
            kv_floor_pages=2048,
            min_expert_slots=16,
        )
        assert slots >= 16

    def test_min_expert_slots_relaxed_when_not_affordable(self):
        # Budget fits only the KV floor plus 4 expert slots; demanding 16
        # would evict the context floor, so the floor wins and the demand
        # is relaxed instead of violating capacity.
        slots, pages = plan_expert_kv_budget(
            budget_bytes=(8 << 20) + 4 * (1 << 20),
            per_expert_bytes=(1 << 20),
            kv_page_bytes=4096,
            num_experts=64,
            kv_floor_pages=2048,
            min_expert_slots=16,
        )
        assert slots == 4
        assert pages >= 2048
