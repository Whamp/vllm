# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "kernels"
    / "benchmark_qwen4_exp_hyperconnection_gemm.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "benchmark_qwen4_exp_hyperconnection_gemm", _BENCHMARK_PATH
)
assert _SPEC is not None and _SPEC.loader is not None
benchmark = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = benchmark
_SPEC.loader.exec_module(benchmark)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize("tokens", [1, 2])
def test_qwen_hyperconnection_known_candidate_tiles(tokens: int) -> None:
    down = benchmark.PROJECTION_CASES["down"]
    up = benchmark.PROJECTION_CASES["up"]

    benchmark.CandidateConfig(128, 1, 1, 8, 10240).validate(down, tokens)
    benchmark.CandidateConfig(64, 4, 1, 1, 320).validate(up, tokens)
    benchmark.CandidateConfig(32, 4, 1, 2, 320).validate(up, tokens)


@given(
    projection_name=st.sampled_from(tuple(benchmark.PROJECTION_CASES)),
    tokens=st.integers(min_value=0, max_value=17),
    block_size=st.integers(min_value=1, max_value=256),
    outputs_per_block=st.integers(min_value=1, max_value=16),
    k_unroll=st.integers(min_value=1, max_value=8),
    vector_width=st.integers(min_value=1, max_value=8),
    static_k=st.one_of(
        st.none(),
        st.sampled_from([64, 128, 320, 1024, 10240, 20480]),
    ),
)
@settings(max_examples=300, deadline=None)
def test_qwen_hyperconnection_candidate_guard_matches_tile_contract(
    projection_name: str,
    tokens: int,
    block_size: int,
    outputs_per_block: int,
    k_unroll: int,
    vector_width: int,
    static_k: int | None,
) -> None:
    case = benchmark.PROJECTION_CASES[projection_name]
    tile_k = block_size * vector_width
    should_accept = (
        1 <= tokens <= 16
        and block_size % 32 == 0
        and case.output_features % outputs_per_block == 0
        and case.input_features % tile_k == 0
        and (static_k is None or static_k == case.input_features)
        and (static_k is None or static_k >= 2 * tile_k)
    )
    config = benchmark.CandidateConfig(
        block_size,
        outputs_per_block,
        k_unroll,
        vector_width,
        static_k,
    )

    if should_accept:
        config.validate(case, tokens)
    else:
        with pytest.raises(ValueError):
            config.validate(case, tokens)


def test_qwen_hyperconnection_projection_pool_bytes() -> None:
    total_bytes = sum(
        case.weight_bytes * case.production_calls
        for case in benchmark.PROJECTION_CASES.values()
    )
    assert total_bytes == 1_296_302_080
