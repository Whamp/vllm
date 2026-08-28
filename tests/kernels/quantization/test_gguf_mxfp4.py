# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GGUF MXFP4 kernels require CUDA"
)

E2M1_DOUBLED = np.array(
    [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12],
    dtype=np.float32,
)


def _decode_e8m0_half(exponent: int) -> np.float32:
    if exponent == 0:
        bits = np.uint32(0x00200000)
    elif exponent == 1:
        bits = np.uint32(0x00400000)
    else:
        bits = np.uint32(exponent - 1) << np.uint32(23)
    return np.frombuffer(bits.tobytes(), dtype=np.float32)[0]


def _make_mxfp4_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks_per_row = input_columns // 32
    raw = np.empty((output_rows, blocks_per_row * 17), dtype=np.uint8)
    decoded = np.empty((output_rows, input_columns), dtype=np.float32)
    exponent_choices = np.array([0, 1, 120, 124, 127, 130], dtype=np.uint8)
    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            offset = block_index * 17
            exponent = int(rng.choice(exponent_choices))
            codes = rng.integers(0, 16, size=32, dtype=np.uint8)
            raw[row, offset] = exponent
            raw[row, offset + 1 : offset + 17] = codes[:16] | (codes[16:] << 4)
            decoded[row, block_index * 32 : block_index * 32 + 32] = E2M1_DOUBLED[
                codes
            ] * _decode_e8m0_half(exponent)
    return raw, decoded


def _quantize_q8_1(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.empty(
        activations.shape[0],
        activations.shape[1] // 32,
        device=activations.device,
        dtype=torch.float16,
    )
    codes = torch.empty_like(activations, dtype=torch.int8)
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    return scales, codes


def _q8_1_reference(
    decoded_weights: np.ndarray,
    scales: torch.Tensor,
    codes: torch.Tensor,
) -> torch.Tensor:
    scale_values = scales.float().cpu().numpy()
    code_values = codes.cpu().numpy().astype(np.float32)
    groups = code_values.reshape(code_values.shape[0], -1, 32)
    dequantized = (groups * scale_values[:, :, None]).reshape(code_values.shape)
    return torch.from_numpy(dequantized @ decoded_weights.T).to(codes.device)


@pytest.mark.parametrize("token_count", [1, 3])
@pytest.mark.parametrize("input_columns", [32, 512])
def test_gguf_mxfp4_raw_matvec_matches_independent_reference(
    token_count: int, input_columns: int
) -> None:
    output_rows = 9
    raw, decoded = _make_mxfp4_weights(
        output_rows, input_columns, token_count + input_columns
    )
    torch.manual_seed(8421)
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = _quantize_q8_1(activations)
    output = torch.empty(token_count, output_rows, device="cuda", dtype=torch.float32)

    torch.ops._C.gguf_mxfp4_q8_1_raw_matvec(
        scales, codes, torch.from_numpy(raw).cuda(), output
    )

    expected = _q8_1_reference(decoded, scales, codes)
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


def test_gguf_mxfp4_indexed_down_matches_independent_reference() -> None:
    token_count, topk, expert_count = 2, 2, 3
    input_columns, output_rows = 512, 7
    raw_weights = []
    decoded_weights = []
    for expert in range(expert_count):
        raw, decoded = _make_mxfp4_weights(output_rows, input_columns, 100 + expert)
        raw_weights.append(raw)
        decoded_weights.append(decoded)
    torch.manual_seed(8421)
    activations = torch.randn(
        token_count * topk, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = _quantize_q8_1(activations)
    topk_ids = torch.tensor([[0, 2], [1, 0]], device="cuda", dtype=torch.int32)
    output = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )

    torch.ops._C.gguf_mxfp4_q8_1_indexed_down(
        scales,
        codes,
        torch.from_numpy(np.stack(raw_weights)).cuda(),
        topk_ids,
        output,
    )

    for token in range(token_count):
        for slot in range(topk):
            assignment = token * topk + slot
            expert = int(topk_ids[token, slot])
            expected = _q8_1_reference(
                decoded_weights[expert],
                scales[assignment : assignment + 1],
                codes[assignment : assignment + 1],
            )[0]
            torch.testing.assert_close(
                output[token, slot], expected, rtol=2e-3, atol=2e-3
            )


def test_gguf_mxfp4_grouped_down_matches_indexed_and_replays() -> None:
    import vllm._moe_C_stable_libtorch  # noqa: F401

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    token_count, topk, expert_count = 16, 2, 4
    input_columns, output_rows = 512, 16
    raw_weights = np.stack(
        [
            _make_mxfp4_weights(output_rows, input_columns, 8000 + expert)[0]
            for expert in range(expert_count)
        ]
    )
    topk_ids = torch.tensor(
        [
            [token % expert_count, (token + 1) % expert_count]
            for token in range(token_count)
        ],
        device="cuda",
        dtype=torch.int32,
    )
    activations = torch.randn(
        token_count * topk,
        input_columns,
        device="cuda",
        dtype=torch.bfloat16,
    )
    scales, codes = _quantize_q8_1(activations)
    weights = torch.from_numpy(raw_weights).cuda()
    indexed = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    grouped = torch.empty_like(indexed)
    torch.ops._C.gguf_mxfp4_q8_1_indexed_down(scales, codes, weights, topk_ids, indexed)
    schedule = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=8,
        num_experts=expert_count,
    )
    grouped_op = torch.ops._C.gguf_mxfp4_q8_1_grouped_down
    grouped_op(scales, codes, weights, *schedule, grouped, topk)
    torch.accelerator.synchronize()
    error = (grouped - indexed).flatten()
    reference = indexed.flatten()
    assert error.square().mean().sqrt() / reference.square().mean().sqrt() < 0.01
    assert error.abs().mean() / reference.abs().mean() < 0.01
    assert error.abs().max() / reference.abs().max() < 0.025
    assert (
        torch.nn.functional.cosine_similarity(grouped.flatten(), reference, dim=0)
        > 0.9999
    )
    before = grouped.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_op(scales, codes, weights, *schedule, grouped, topk)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(grouped, before, rtol=0, atol=0)


def test_gguf_mxfp4_raw_matvec_replays_in_cuda_graph() -> None:
    raw, _ = _make_mxfp4_weights(8, 512, 8421)
    activations = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    scales, codes = _quantize_q8_1(activations)
    weights = torch.from_numpy(raw).cuda()
    output = torch.empty(1, 8, device="cuda", dtype=torch.float32)
    graph = torch.cuda.CUDAGraph()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_mxfp4_q8_1_raw_matvec(scales, codes, weights, output)
    graph.replay()
    first = output.clone()
    output.fill_(float("nan"))
    graph.replay()

    torch.testing.assert_close(output, first, rtol=0, atol=0)
