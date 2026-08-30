# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from __future__ import annotations

from types import SimpleNamespace

import torch
from hypothesis import given
from hypothesis import strategies as st
from torch import nn

from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.qwen4_exp.nvidia import low_latency_gemm


class _FakeLinear(nn.Module):
    def __init__(self, output_features: int, input_features: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(output_features, input_features, dtype=torch.bfloat16)
        )
        self.quant_method = UnquantizedLinearMethod()


class _FakeQwenModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.down = _FakeLinear(336, 10240)
        self.final_down = _FakeLinear(320, 10240)
        self.up = _FakeLinear(10240, 320)
        self.unrelated = _FakeLinear(640, 2560)


def test_sm86_hyperconnection_launch_plans_are_exact() -> None:
    assert low_latency_gemm.qwen4_exp_sm86_hyperconnection_plan(1, 336, 10240) == (
        256,
        1,
    )
    assert low_latency_gemm.qwen4_exp_sm86_hyperconnection_plan(2, 336, 10240) == (
        256,
        1,
    )
    assert low_latency_gemm.qwen4_exp_sm86_hyperconnection_plan(1, 10240, 320) == (
        32,
        4,
    )
    assert low_latency_gemm.qwen4_exp_sm86_hyperconnection_plan(2, 10240, 320) is None
    assert low_latency_gemm.qwen4_exp_sm86_hyperconnection_plan(1, 320, 10240) is None
    assert low_latency_gemm.qwen4_exp_sm86_hyperconnection_plan(4, 336, 10240) is None


@given(
    tokens=st.integers(min_value=1, max_value=8),
    output_features=st.sampled_from([24, 320, 336, 640, 10240]),
    input_features=st.sampled_from([320, 1536, 2560, 10240]),
)
def test_sm86_hyperconnection_plan_rejects_every_other_geometry(
    tokens: int,
    output_features: int,
    input_features: int,
) -> None:
    plan = low_latency_gemm.qwen4_exp_sm86_hyperconnection_plan(
        tokens,
        output_features,
        input_features,
    )
    expected = {
        (1, 336, 10240): (256, 1),
        (2, 336, 10240): (256, 1),
        (1, 10240, 320): (32, 4),
    }.get((tokens, output_features, input_features))
    assert plan == expected


def test_sm86_dispatch_uses_native_op_only_for_registered_shapes(monkeypatch) -> None:
    native_result = object()
    fallback_result = object()
    native_calls: list[tuple[object, object]] = []
    monkeypatch.setattr(low_latency_gemm, "_is_sm86", lambda: True)
    monkeypatch.setattr(low_latency_gemm, "_runtime_ok", lambda x, weight: True)
    monkeypatch.setattr(
        low_latency_gemm.envs,
        "VLLM_QWEN4_EXP_HYPERCONNECTION_BF16_SM86",
        True,
        raising=False,
    )

    def record_native_call(x: object, weight: object) -> object:
        native_calls.append((x, weight))
        return native_result

    monkeypatch.setattr(
        low_latency_gemm,
        "_run_qwen4_exp_sm86_hyperconnection",
        record_native_call,
    )
    monkeypatch.setattr(
        torch.nn.functional,
        "linear",
        lambda x, weight: fallback_result,
    )

    native_shapes = (
        ((1, 10240), (336, 10240)),
        ((2, 10240), (336, 10240)),
        ((1, 320), (10240, 320)),
    )
    for activation_shape, weight_shape in native_shapes:
        result = low_latency_gemm._qwen4_exp_low_latency_gemm(
            SimpleNamespace(shape=activation_shape),
            SimpleNamespace(shape=weight_shape),
        )
        assert result is native_result

    result = low_latency_gemm._qwen4_exp_low_latency_gemm(
        SimpleNamespace(shape=(2, 320)),
        SimpleNamespace(shape=(10240, 320)),
    )
    assert result is fallback_result
    assert len(native_calls) == len(native_shapes)


def test_sm86_selector_replaces_only_hyperconnection_linears(monkeypatch) -> None:
    model = _FakeQwenModel()
    monkeypatch.setattr(low_latency_gemm, "LinearBase", _FakeLinear)
    monkeypatch.setattr(low_latency_gemm, "_is_sm86", lambda: True)
    monkeypatch.setattr(
        low_latency_gemm,
        "_sm86_hyperconnection_op_available",
        lambda: True,
    )
    monkeypatch.setattr(
        low_latency_gemm.envs,
        "VLLM_QWEN4_EXP_HYPERCONNECTION_BF16_SM86",
        True,
        raising=False,
    )

    low_latency_gemm.enable_qwen4_exp_low_latency_gemm(model, torch.bfloat16)

    assert isinstance(
        model.down.quant_method,
        low_latency_gemm.Qwen4ExpSm86HyperconnectionLinearMethod,
    )
    assert isinstance(
        model.up.quant_method,
        low_latency_gemm.Qwen4ExpSm86HyperconnectionLinearMethod,
    )
    assert type(model.final_down.quant_method) is UnquantizedLinearMethod
    assert type(model.unrelated.quant_method) is UnquantizedLinearMethod


def test_sm86_selector_is_default_off(monkeypatch) -> None:
    model = _FakeQwenModel()
    monkeypatch.setattr(low_latency_gemm, "LinearBase", _FakeLinear)
    monkeypatch.setattr(low_latency_gemm, "_is_sm86", lambda: True)
    monkeypatch.setattr(
        low_latency_gemm.envs,
        "VLLM_QWEN4_EXP_HYPERCONNECTION_BF16_SM86",
        False,
        raising=False,
    )

    low_latency_gemm.enable_qwen4_exp_low_latency_gemm(model, torch.bfloat16)

    assert all(
        type(child.quant_method) is UnquantizedLinearMethod
        for child in (model.down, model.final_down, model.up, model.unrelated)
    )


def test_sm86_selector_fails_when_native_op_is_missing(monkeypatch) -> None:
    model = _FakeQwenModel()
    monkeypatch.setattr(low_latency_gemm, "_is_sm86", lambda: True)
    monkeypatch.setattr(
        low_latency_gemm,
        "_sm86_hyperconnection_op_available",
        lambda: False,
    )
    monkeypatch.setattr(
        low_latency_gemm.envs,
        "VLLM_QWEN4_EXP_HYPERCONNECTION_BF16_SM86",
        True,
        raising=False,
    )

    try:
        low_latency_gemm.enable_qwen4_exp_low_latency_gemm(model, torch.bfloat16)
    except RuntimeError as error:
        assert str(error).startswith("Qwen4Exp SM86 hyperconnection op unavailable")
    else:
        raise AssertionError("missing native SM86 op must fail closed")
