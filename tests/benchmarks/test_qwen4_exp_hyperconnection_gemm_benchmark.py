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

_MARLIN_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "kernels"
    / "benchmark_qwen4_exp_hyperconnection_marlin.py"
)
_MARLIN_SPEC = importlib.util.spec_from_file_location(
    "benchmark_qwen4_exp_hyperconnection_marlin", _MARLIN_BENCHMARK_PATH
)
assert _MARLIN_SPEC is not None and _MARLIN_SPEC.loader is not None
marlin_benchmark = importlib.util.module_from_spec(_MARLIN_SPEC)
sys.modules[_MARLIN_SPEC.name] = marlin_benchmark
_MARLIN_SPEC.loader.exec_module(marlin_benchmark)

_NATIVE_BF16_BENCHMARK_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "kernels"
    / "benchmark_qwen4_exp_hyperconnection_bf16_sm86.py"
)
_NATIVE_BF16_SPEC = importlib.util.spec_from_file_location(
    "benchmark_qwen4_exp_hyperconnection_bf16_sm86",
    _NATIVE_BF16_BENCHMARK_PATH,
)
assert _NATIVE_BF16_SPEC is not None and _NATIVE_BF16_SPEC.loader is not None
native_bf16_benchmark = importlib.util.module_from_spec(_NATIVE_BF16_SPEC)
sys.modules[_NATIVE_BF16_SPEC.name] = native_bf16_benchmark
_NATIVE_BF16_SPEC.loader.exec_module(native_bf16_benchmark)

_W4_SCREEN_PATH = (
    Path(__file__).resolve().parents[2]
    / "benchmarks"
    / "kernels"
    / "screen_qwen4_exp_hyperconnection_w4_up.py"
)
_W4_SCREEN_SPEC = importlib.util.spec_from_file_location(
    "screen_qwen4_exp_hyperconnection_w4_up",
    _W4_SCREEN_PATH,
)
assert _W4_SCREEN_SPEC is not None and _W4_SCREEN_SPEC.loader is not None
w4_screen = importlib.util.module_from_spec(_W4_SCREEN_SPEC)
sys.modules[_W4_SCREEN_SPEC.name] = w4_screen
_W4_SCREEN_SPEC.loader.exec_module(w4_screen)


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


def test_qwen_hyperconnection_known_marlin_plans() -> None:
    down = marlin_benchmark.HYPERCONNECTION_PROJECTIONS["down"]
    up = marlin_benchmark.HYPERCONNECTION_PROJECTIONS["up"]

    marlin_benchmark.MarlinExperimentPlan(bits=8, group_size=128).validate(down)
    marlin_benchmark.MarlinExperimentPlan(bits=8, group_size=-1).validate(up)
    marlin_benchmark.MarlinExperimentPlan(bits=4, group_size=64).validate(up)


@given(
    projection_name=st.sampled_from(
        tuple(marlin_benchmark.HYPERCONNECTION_PROJECTIONS)
    ),
    bits=st.integers(min_value=1, max_value=9),
    group_size=st.integers(min_value=-2, max_value=160),
)
@settings(max_examples=300, deadline=None)
def test_qwen_hyperconnection_marlin_guard_matches_group_contract(
    projection_name: str,
    bits: int,
    group_size: int,
) -> None:
    projection = marlin_benchmark.HYPERCONNECTION_PROJECTIONS[projection_name]
    should_accept = (
        bits in (4, 8)
        and group_size in (-1, 32, 64, 128)
        and (group_size == -1 or projection.input_features % group_size == 0)
    )
    plan = marlin_benchmark.MarlinExperimentPlan(bits, group_size)

    if should_accept:
        plan.validate(projection)
    else:
        with pytest.raises(ValueError):
            plan.validate(projection)


def test_qwen_hyperconnection_marlin_projection_pool_bytes() -> None:
    total_bytes = sum(
        projection.bf16_weight_bytes * projection.production_calls
        for projection in marlin_benchmark.HYPERCONNECTION_PROJECTIONS.values()
    )
    assert total_bytes == 1_296_302_080


def test_qwen_hyperconnection_known_native_bf16_plans() -> None:
    native_bf16_benchmark.Sm86Bf16KernelPlan(128, 1).validate("down")
    native_bf16_benchmark.Sm86Bf16KernelPlan(32, 4).validate("up")
    native_bf16_benchmark.Sm86Bf16KernelPlan(64, 8).validate("up")


@given(
    projection_name=st.sampled_from(tuple(benchmark.PROJECTION_CASES)),
    block_threads=st.integers(min_value=1, max_value=300),
    outputs_per_block=st.integers(min_value=1, max_value=10),
)
@settings(max_examples=300, deadline=None)
def test_qwen_hyperconnection_native_bf16_guard_matches_launch_contract(
    projection_name: str,
    block_threads: int,
    outputs_per_block: int,
) -> None:
    projection = benchmark.PROJECTION_CASES[projection_name]
    should_accept = (
        block_threads in (32, 64, 128, 256)
        and outputs_per_block in (1, 4, 8)
        and projection.output_features % outputs_per_block == 0
        and (projection_name != "down" or outputs_per_block == 1)
    )
    plan = native_bf16_benchmark.Sm86Bf16KernelPlan(
        block_threads,
        outputs_per_block,
    )

    if should_accept:
        plan.validate(projection_name)
    else:
        with pytest.raises(ValueError):
            plan.validate(projection_name)


@given(
    rows=st.integers(min_value=1, max_value=4),
    group_size=st.sampled_from((-1, 32, 64)),
    values=st.lists(
        st.floats(
            min_value=-8.0,
            max_value=8.0,
            allow_nan=False,
            allow_infinity=False,
            width=32,
        ),
        min_size=256,
        max_size=256,
    ),
)
@settings(max_examples=100, deadline=None)
def test_qwen_hyperconnection_w4_reconstruction_respects_signed_code_range(
    rows: int,
    group_size: int,
    values: list[float],
) -> None:
    import torch

    weight = torch.tensor(values, dtype=torch.float32).repeat(rows, 1)
    reconstructed = w4_screen.quantize_symmetric_int4(weight, group_size)
    effective_group = weight.shape[1] if group_size == -1 else group_size
    blocks = weight.view(rows, weight.shape[1] // effective_group, effective_group)
    reconstructed_blocks = reconstructed.view_as(blocks)
    maximum = blocks.amax(dim=2, keepdim=True)
    minimum = blocks.amin(dim=2, keepdim=True)
    scales = torch.maximum(maximum.abs() / 7, minimum.abs() / 8)
    scales = scales.clamp_min(w4_screen.FP16_MIN_SUBNORMAL).to(torch.float16).float()
    recovered_codes = reconstructed_blocks / scales

    assert reconstructed.shape == weight.shape
    assert torch.isfinite(reconstructed).all()
    torch.testing.assert_close(recovered_codes, recovered_codes.round())
    assert recovered_codes.min() >= -8
    assert recovered_codes.max() <= 7


def test_qwen_hyperconnection_w4_reconstruction_rejects_partial_group() -> None:
    import torch

    with pytest.raises(ValueError, match="group size must divide"):
        w4_screen.quantize_symmetric_int4(torch.ones((2, 320)), 128)
