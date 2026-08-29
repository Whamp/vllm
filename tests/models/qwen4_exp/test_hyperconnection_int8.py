# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite

import vllm.envs as envs
import vllm.model_executor.layers.linear as linear_module
import vllm.model_executor.parameter as parameter_module
from vllm.model_executor.layers.linear import UnquantizedLinearMethod
from vllm.models.qwen4_exp.common.hyperconnection import HyperConnectionConfig
from vllm.models.qwen4_exp.nvidia.hyperconnection import GatedResidual
from vllm.models.qwen4_exp.nvidia.hyperconnection_int8 import (
    HyperconnectionInt8ScaleLayout,
    Qwen4ExpHyperconnectionInt8LinearMethod,
)
from vllm.platforms import current_platform
from vllm.triton_utils import HAS_TRITON


def test_hyperconnection_int8_quantizes_k_groups_and_zeroes_padding() -> None:
    weight = torch.stack(
        (
            torch.linspace(-2.0, 1.5, 256, dtype=torch.bfloat16),
            torch.linspace(0.125, 4.0, 256, dtype=torch.bfloat16),
            torch.full((256,), 9.0, dtype=torch.bfloat16),
            torch.full((256,), -7.0, dtype=torch.bfloat16),
        )
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
    )

    method = Qwen4ExpHyperconnectionInt8LinearMethod(
        scale_layout=HyperconnectionInt8ScaleLayout.K_GROUP_128,
        valid_output_rows=2,
    )
    method.process_weights_after_loading(layer)

    assert layer.weight.dtype == torch.int8
    assert layer.weight_scale.dtype == torch.float16
    assert layer.weight_scale.shape == (4, 2)
    assert torch.count_nonzero(layer.weight[2:]) == 0
    assert torch.equal(layer.weight_scale[2:], torch.ones((2, 2), dtype=torch.float16))

    reconstructed = (
        layer.weight[:2]
        .reshape(2, 2, 128)
        .float()
        .mul(layer.weight_scale[:2, :, None].float())
        .reshape(2, 256)
    )
    error = (reconstructed - weight[:2].float()).abs()
    per_value_bound = layer.weight_scale[:2].float().repeat_interleave(128, dim=1)
    assert torch.all(error <= per_value_bound * 0.51 + 1e-6)


def _reference_quantize(
    weight: torch.Tensor,
    layout: HyperconnectionInt8ScaleLayout,
) -> tuple[torch.Tensor, torch.Tensor]:
    group_size = (
        128 if layout is HyperconnectionInt8ScaleLayout.K_GROUP_128 else weight.shape[1]
    )
    group_count = weight.shape[1] // group_size
    scales = torch.empty((weight.shape[0], group_count), dtype=torch.float16)
    codes = torch.empty_like(weight, dtype=torch.int8)
    for row_index in range(weight.shape[0]):
        for group_index in range(group_count):
            start = group_index * group_size
            stop = start + group_size
            values = weight[row_index, start:stop].float()
            scale = values.abs().amax() / 127
            if scale == 0:
                scale = torch.ones((), dtype=torch.float32)
            scale = scale.to(torch.float16)
            scales[row_index, group_index] = scale
            codes[row_index, start:stop] = (
                (values / scale.float()).round().clamp(-127, 127).to(torch.int8)
            )
    return codes, scales


@composite
def _finite_bf16_matrices(draw) -> torch.Tensor:
    rows = draw(st.integers(min_value=1, max_value=6))
    groups = draw(st.integers(min_value=1, max_value=3))
    element_count = rows * groups * 128
    values = draw(
        st.lists(
            st.floats(
                min_value=-8,
                max_value=8,
                allow_nan=False,
                allow_infinity=False,
                width=32,
            ),
            min_size=element_count,
            max_size=element_count,
        )
    )
    return torch.tensor(values, dtype=torch.bfloat16).reshape(rows, groups * 128)


@pytest.mark.parametrize(
    "layout",
    (
        HyperconnectionInt8ScaleLayout.PER_ROW,
        HyperconnectionInt8ScaleLayout.K_GROUP_128,
    ),
)
@given(weight=_finite_bf16_matrices())
@settings(max_examples=100, deadline=None)
def test_hyperconnection_int8_matches_independent_quantization(
    layout: HyperconnectionInt8ScaleLayout,
    weight: torch.Tensor,
) -> None:
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(weight.clone(), requires_grad=False)
    )

    method = Qwen4ExpHyperconnectionInt8LinearMethod(scale_layout=layout)
    method.process_weights_after_loading(layer)

    expected_codes, expected_scales = _reference_quantize(weight, layout)
    assert torch.equal(layer.weight, expected_codes)
    assert torch.equal(layer.weight_scale, expected_scales)


def test_hyperconnection_int8_rejects_partial_k_group() -> None:
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight",
        torch.nn.Parameter(torch.zeros((2, 129), dtype=torch.bfloat16), False),
    )
    method = Qwen4ExpHyperconnectionInt8LinearMethod(
        scale_layout=HyperconnectionInt8ScaleLayout.K_GROUP_128
    )

    with pytest.raises(ValueError, match="divisible by 128"):
        method.process_weights_after_loading(layer)


@pytest.mark.skipif(current_platform.is_cuda(), reason="CPU fail-closed contract")
def test_hyperconnection_int8_execution_fails_closed_outside_sm86() -> None:
    method = Qwen4ExpHyperconnectionInt8LinearMethod(
        scale_layout=HyperconnectionInt8ScaleLayout.PER_ROW
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(torch.zeros((2, 2), dtype=torch.int8), False)
    )

    with pytest.raises(RuntimeError, match="requires CUDA SM86"):
        method.apply(layer, torch.zeros((1, 2), dtype=torch.bfloat16))


def test_qwen_hyperconnection_int8_flag_selects_projection_scale_layouts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config = HyperConnectionConfig(
        hc_count=4,
        hidden_size=32,
        hc_lowrank=8,
        params_dtype=torch.bfloat16,
    )

    monkeypatch.setattr(parameter_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        parameter_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(linear_module, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(
        linear_module, "get_tensor_model_parallel_world_size", lambda: 1
    )
    monkeypatch.setattr(envs, "VLLM_QWEN4_EXP_HYPERCONNECTION_INT8", True)
    quantized = GatedResidual(config, use_combine=True)

    down_method = quantized.input_mix_weight_down_block_inject.quant_method
    assert isinstance(down_method, Qwen4ExpHyperconnectionInt8LinearMethod)
    assert down_method.scale_layout is HyperconnectionInt8ScaleLayout.K_GROUP_128
    assert down_method.valid_output_rows == config.hc_lowrank + config.hc_count
    up_method = quantized.input_mix_weight_up.quant_method
    assert isinstance(up_method, Qwen4ExpHyperconnectionInt8LinearMethod)
    assert up_method.scale_layout is HyperconnectionInt8ScaleLayout.PER_ROW
    assert up_method.valid_output_rows is None

    monkeypatch.setattr(envs, "VLLM_QWEN4_EXP_HYPERCONNECTION_INT8", False)
    control = GatedResidual(config, use_combine=True)
    assert isinstance(
        control.input_mix_weight_down_block_inject.quant_method,
        UnquantizedLinearMethod,
    )
    assert isinstance(control.input_mix_weight_up.quant_method, UnquantizedLinearMethod)


def _quantized_linear_reference(
    inputs: torch.Tensor,
    weight: torch.Tensor,
    weight_scale: torch.Tensor,
    group_size: int,
) -> torch.Tensor:
    token_count, input_columns = inputs.shape
    group_count = input_columns // group_size
    grouped_inputs = inputs.float().reshape(token_count, group_count, group_size)
    input_scale = grouped_inputs.abs().amax(dim=2) / 127
    input_scale = torch.where(
        input_scale == 0,
        torch.ones_like(input_scale),
        input_scale,
    )
    input_codes = (grouped_inputs / input_scale[:, :, None]).round().clamp(-127, 127)
    grouped_weight = weight.float().reshape(weight.shape[0], group_count, group_size)
    output = torch.zeros(
        (token_count, weight.shape[0]),
        dtype=torch.float32,
        device=inputs.device,
    )
    for group_index in range(group_count):
        integer_dot = input_codes[:, group_index] @ grouped_weight[:, group_index].T
        output += (
            integer_dot
            * input_scale[:, group_index, None]
            * weight_scale[None, :, group_index].float()
        )
    return output.to(inputs.dtype)


_GPU_TEST_REASON = "Qwen hyperconnection INT8 kernels require CUDA and Triton"


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason=_GPU_TEST_REASON,
)
@pytest.mark.parametrize(
    "token_count,output_rows,input_columns,layout",
    (
        (1, 336, 10240, HyperconnectionInt8ScaleLayout.K_GROUP_128),
        (2, 336, 10240, HyperconnectionInt8ScaleLayout.K_GROUP_128),
        (1, 10240, 320, HyperconnectionInt8ScaleLayout.PER_ROW),
        (2, 10240, 320, HyperconnectionInt8ScaleLayout.PER_ROW),
    ),
)
def test_hyperconnection_int8_decode_matches_grouped_reference_and_cuda_graph(
    token_count: int,
    output_rows: int,
    input_columns: int,
    layout: HyperconnectionInt8ScaleLayout,
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.hc_int8 import (
        hyperconnection_int8_decode,
    )

    torch.manual_seed(20260829)
    source_weight = torch.randn(
        (output_rows, input_columns), dtype=torch.bfloat16, device="cuda"
    )
    inputs = torch.randn(
        (token_count, input_columns), dtype=torch.bfloat16, device="cuda"
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(source_weight.clone(), requires_grad=False)
    )
    method = Qwen4ExpHyperconnectionInt8LinearMethod(scale_layout=layout)
    method.process_weights_after_loading(layer)
    group_size = (
        128 if layout is HyperconnectionInt8ScaleLayout.K_GROUP_128 else input_columns
    )

    expected = _quantized_linear_reference(
        inputs,
        layer.weight,
        layer.weight_scale,
        group_size,
    )
    actual = hyperconnection_int8_decode(
        inputs,
        layer.weight,
        layer.weight_scale,
        group_size,
    )
    torch.testing.assert_close(actual, expected, rtol=5e-3, atol=5e-2)

    for _ in range(3):
        hyperconnection_int8_decode(
            inputs,
            layer.weight,
            layer.weight_scale,
            group_size,
        )
    torch.accelerator.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        graph_output = hyperconnection_int8_decode(
            inputs,
            layer.weight,
            layer.weight_scale,
            group_size,
        )
    graph.replay()
    first = graph_output.clone()
    graph.replay()
    second = graph_output.clone()
    torch.accelerator.synchronize()
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()


@pytest.mark.skipif(
    not current_platform.is_cuda() or not HAS_TRITON,
    reason=_GPU_TEST_REASON,
)
@pytest.mark.parametrize(
    "output_rows,input_columns,layout",
    (
        (336, 10240, HyperconnectionInt8ScaleLayout.K_GROUP_128),
        (10240, 320, HyperconnectionInt8ScaleLayout.PER_ROW),
    ),
)
def test_hyperconnection_int8_dequantizes_on_caller_stream(
    output_rows: int,
    input_columns: int,
    layout: HyperconnectionInt8ScaleLayout,
) -> None:
    from vllm.models.qwen4_exp.nvidia.ops.hc_int8 import (
        dequantize_hyperconnection_int8_weight,
    )

    torch.manual_seed(20260829)
    source_weight = torch.randn(
        (output_rows, input_columns), dtype=torch.bfloat16, device="cuda"
    )
    layer = torch.nn.Module()
    layer.register_parameter(
        "weight", torch.nn.Parameter(source_weight.clone(), requires_grad=False)
    )
    method = Qwen4ExpHyperconnectionInt8LinearMethod(scale_layout=layout)
    method.process_weights_after_loading(layer)
    group_size = (
        128 if layout is HyperconnectionInt8ScaleLayout.K_GROUP_128 else input_columns
    )
    group_count = input_columns // group_size
    expected = (
        (
            layer.weight.float().reshape(output_rows, group_count, group_size)
            * layer.weight_scale.float()[:, :, None]
        )
        .reshape(output_rows, input_columns)
        .to(torch.bfloat16)
    )
    actual = torch.empty_like(expected)

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        dequantize_hyperconnection_int8_weight(
            layer.weight,
            layer.weight_scale,
            actual,
            group_size,
        )
    stream.synchronize()
    assert torch.equal(actual, expected)
