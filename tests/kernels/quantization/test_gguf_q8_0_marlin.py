# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.model_executor.layers.quantization.gguf_dsv4.q8_0_marlin import (
    apply_gguf_q8_0_marlin,
    prepare_gguf_q8_0_marlin,
    unpack_gguf_q8_0_to_gptq,
)
from vllm.platforms import current_platform


def test_unpack_gguf_q8_0_to_gptq_preserves_codes_and_scale_orientation():
    codes = np.array(
        [
            -128,
            -127,
            -1,
            0,
            1,
            2,
            3,
            4,
            5,
            6,
            7,
            8,
            9,
            10,
            11,
            12,
            13,
            14,
            15,
            16,
            17,
            18,
            19,
            20,
            21,
            22,
            23,
            24,
            25,
            26,
            126,
            127,
        ],
        dtype=np.int8,
    )
    scale_bytes = np.frombuffer(np.float16(0.125).tobytes(), dtype=np.uint8)
    raw = np.concatenate((scale_bytes, codes.view(np.uint8))).reshape(1, 34)

    packed, scales = unpack_gguf_q8_0_to_gptq(
        torch.from_numpy(raw), input_columns=32, scale_dtype=torch.bfloat16
    )

    unsigned = codes.astype(np.int16) + 128
    expected_packed = np.array(
        [
            sum(int(unsigned[4 * word + byte]) << (8 * byte) for byte in range(4))
            for word in range(8)
        ],
        dtype=np.uint32,
    ).view(np.int32)
    torch.testing.assert_close(packed[:, 0], torch.from_numpy(expected_packed))
    assert packed.shape == (8, 1)
    assert scales.shape == (1, 1)
    assert scales.dtype == torch.bfloat16
    assert scales.item() == 0.125


@pytest.mark.parametrize(
    "raw_shape,input_columns,error",
    [
        ((2, 34), 64, "row byte count"),
        ((2, 68), 48, "multiple of 32"),
    ],
)
def test_unpack_gguf_q8_0_to_gptq_rejects_invalid_geometry(
    raw_shape: tuple[int, int], input_columns: int, error: str
):
    with pytest.raises(ValueError, match=error):
        unpack_gguf_q8_0_to_gptq(
            torch.zeros(raw_shape, dtype=torch.uint8),
            input_columns=input_columns,
            scale_dtype=torch.bfloat16,
        )


def _make_q8_0_weights(output_rows: int, input_columns: int):
    generator = torch.Generator().manual_seed(20260817)
    scales = (
        torch.rand(
            output_rows,
            input_columns // 32,
            generator=generator,
            dtype=torch.float32,
        )
        * 0.02
        + 0.001
    ).to(torch.float16)
    codes = torch.randint(
        -127,
        128,
        (output_rows, input_columns),
        generator=generator,
        dtype=torch.int8,
    )
    blocks = torch.empty(output_rows, input_columns // 32, 34, dtype=torch.uint8)
    blocks[:, :, :2] = (
        scales.contiguous()
        .view(torch.uint8)
        .reshape(output_rows, input_columns // 32, 2)
    )
    blocks[:, :, 2:] = codes.view(output_rows, input_columns // 32, 32).view(
        torch.uint8
    )
    grouped_codes = codes.to(torch.float32).reshape(
        output_rows, input_columns // 32, 32
    )
    dequantized = (grouped_codes * scales.to(torch.float32)[:, :, None]).reshape(
        output_rows, input_columns
    )
    bf16_scale_dequantized = (
        grouped_codes * scales.to(torch.bfloat16).to(torch.float32)[:, :, None]
    ).reshape(output_rows, input_columns)
    return blocks.reshape(output_rows, -1), dequantized, bf16_scale_dequantized


def _normalized_output_errors(output: torch.Tensor, reference: torch.Tensor):
    output = output.float().flatten()
    reference = reference.float().flatten()
    error = output - reference
    return {
        "nrmse": error.square().mean().sqrt() / reference.square().mean().sqrt(),
        "normalized_mae": error.abs().mean() / reference.abs().mean(),
        "max_ratio": error.abs().max() / reference.abs().max(),
        "cosine": torch.nn.functional.cosine_similarity(output, reference, dim=0),
    }


def _assert_gguf_q8_0_marlin_matches_reference(token_count: int, input_columns: int):
    output_rows = 256
    raw, dequantized, bf16_scale_dequantized = _make_q8_0_weights(
        output_rows, input_columns
    )
    prepared = prepare_gguf_q8_0_marlin(
        raw.cuda(), input_columns=input_columns, scale_dtype=torch.bfloat16
    )
    inputs = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )

    output = apply_gguf_q8_0_marlin(inputs, prepared)
    transformed_reference = inputs.float() @ bf16_scale_dequantized.cuda().T
    transformed_errors = _normalized_output_errors(output, transformed_reference)
    assert transformed_errors["nrmse"] <= 0.005
    assert transformed_errors["normalized_mae"] <= 0.005
    assert transformed_errors["max_ratio"] <= 0.01

    original_reference = inputs.float() @ dequantized.cuda().T
    original_errors = _normalized_output_errors(output, original_reference)
    assert original_errors["nrmse"] <= 0.01
    assert original_errors["normalized_mae"] <= 0.01
    assert original_errors["max_ratio"] <= 0.025
    assert original_errors["cosine"] >= 0.9999


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Marlin requires CUDA")
@pytest.mark.parametrize("token_count", [1, 4])
def test_gguf_q8_0_marlin_matches_dequantized_reference(token_count: int):
    _assert_gguf_q8_0_marlin_matches_reference(token_count, input_columns=256)


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Marlin requires CUDA")
@pytest.mark.parametrize("input_columns", [512, 1024, 2048, 4096])
def test_gguf_q8_0_marlin_covers_deepseek_dense_input_widths(input_columns: int):
    _assert_gguf_q8_0_marlin_matches_reference(
        token_count=1, input_columns=input_columns
    )


@pytest.mark.skipif(not current_platform.is_cuda(), reason="Marlin requires CUDA")
def test_gguf_q8_0_marlin_is_graph_safe_and_storage_byte_neutral():
    from vllm.v1.attention.ops.rocm_aiter_mla_sparse import (
        _apply_dsv4_wo_a_marlin_diagonal,
    )

    output_rows, input_columns = 2048, 4096
    local_groups, output_rank = 2, 1024
    raw, _, bf16_scale_dequantized = _make_q8_0_weights(output_rows, input_columns)
    prepared = prepare_gguf_q8_0_marlin(
        raw.cuda(), input_columns=input_columns, scale_dtype=torch.bfloat16
    )
    assert prepared.weight.nbytes + prepared.scales.nbytes == raw.numel()

    class PreparedWoA(torch.nn.Module):
        def forward(self, inputs):
            return apply_gguf_q8_0_marlin(inputs, prepared)

    wo_a = PreparedWoA()
    grouped_weight = bf16_scale_dequantized.cuda().view(
        local_groups, output_rank, input_columns
    )
    for token_count in (1, 2, 4):
        inputs = torch.randn(
            token_count,
            local_groups,
            input_columns,
            device="cuda",
            dtype=torch.bfloat16,
        )
        output = _apply_dsv4_wo_a_marlin_diagonal(
            inputs,
            wo_a,
            n_local_groups=local_groups,
            o_lora_rank=output_rank,
        )
        reference = torch.einsum("tgd,grd->tgr", inputs.float(), grouped_weight)
        errors = _normalized_output_errors(output, reference)
        assert errors["nrmse"] <= 0.005
        assert errors["normalized_mae"] <= 0.005
        assert errors["max_ratio"] <= 0.01

        graph_output = torch.empty_like(output)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            graph_output.copy_(
                _apply_dsv4_wo_a_marlin_diagonal(
                    inputs,
                    wo_a,
                    n_local_groups=local_groups,
                    o_lora_rank=output_rank,
                )
            )
        graph.replay()
        torch.accelerator.synchronize()
        torch.testing.assert_close(graph_output, output, rtol=0, atol=0)
