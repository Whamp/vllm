# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GGUF Q4/Q5/Q6 kernels require CUDA"
)


def _decode_scale_min(scales: np.ndarray, group_index: int) -> tuple[int, int]:
    if group_index < 4:
        return int(scales[group_index] & 63), int(scales[group_index + 4] & 63)
    scale = int(scales[group_index + 4] & 15)
    scale |= int(scales[group_index - 4] >> 6) << 4
    minimum = int(scales[group_index + 4] >> 4)
    minimum |= int(scales[group_index] >> 6) << 4
    return scale, minimum


def _make_q45_weights(
    output_rows: int,
    input_columns: int,
    seed: int,
    *,
    use_q5: bool,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    block_bytes = 176 if use_q5 else 144
    blocks_per_row = input_columns // 256
    raw = np.empty((output_rows, blocks_per_row * block_bytes), dtype=np.uint8)
    decoded = np.empty((output_rows, input_columns), dtype=np.float32)
    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            offset = block_index * block_bytes
            scale = np.float16(rng.uniform(0.001, 0.1))
            minimum = np.float16(rng.uniform(0.001, 0.1))
            raw[row, offset : offset + 2] = np.frombuffer(
                scale.tobytes(), dtype=np.uint8
            )
            raw[row, offset + 2 : offset + 4] = np.frombuffer(
                minimum.tobytes(), dtype=np.uint8
            )
            packed_scales = rng.integers(0, 256, size=12, dtype=np.uint8)
            raw[row, offset + 4 : offset + 16] = packed_scales
            high_bits = rng.integers(0, 256, size=32, dtype=np.uint8)
            quant_offset = offset + 16
            if use_q5:
                raw[row, quant_offset : quant_offset + 32] = high_bits
                quant_offset += 32
            quants = rng.integers(0, 256, size=128, dtype=np.uint8)
            raw[row, quant_offset : quant_offset + 128] = quants
            for group_index in range(8):
                group_scale, group_minimum = _decode_scale_min(
                    packed_scales, group_index
                )
                segment = group_index // 2
                high_nibble = group_index % 2
                packed = quants[segment * 32 : segment * 32 + 32]
                values = (packed >> 4 if high_nibble else packed & np.uint8(15)).astype(
                    np.int32
                )
                if use_q5:
                    values |= ((high_bits & np.uint8(1 << group_index)) != 0).astype(
                        np.int32
                    ) << 4
                start = block_index * 256 + group_index * 32
                decoded[row, start : start + 32] = (
                    np.float32(scale) * group_scale * values
                    - np.float32(minimum) * group_minimum
                )
    return raw, decoded


def _make_q4_k_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    return _make_q45_weights(output_rows, input_columns, seed, use_q5=False)


def _make_q5_k_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    return _make_q45_weights(output_rows, input_columns, seed, use_q5=True)


def _make_q6_k_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks_per_row = input_columns // 256
    raw = np.empty((output_rows, blocks_per_row * 210), dtype=np.uint8)
    decoded = np.empty((output_rows, input_columns), dtype=np.float32)
    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            offset = block_index * 210
            low = rng.integers(0, 256, size=128, dtype=np.uint8)
            high = rng.integers(0, 256, size=64, dtype=np.uint8)
            scales = rng.integers(-127, 128, size=16, dtype=np.int8)
            block_scale = np.float16(rng.uniform(0.001, 0.1))
            raw[row, offset : offset + 128] = low
            raw[row, offset + 128 : offset + 192] = high
            raw[row, offset + 192 : offset + 208] = scales.view(np.uint8)
            raw[row, offset + 208 : offset + 210] = np.frombuffer(
                block_scale.tobytes(), dtype=np.uint8
            )
            block_output = decoded[row, block_index * 256 : block_index * 256 + 256]
            for half in range(2):
                for element in range(32):
                    scale_index = element // 16
                    low_base = half * 64
                    high_base = half * 32
                    quant_values = (
                        int(low[low_base + element] & 15)
                        | (int((high[high_base + element] >> 0) & 3) << 4),
                        int(low[low_base + 32 + element] & 15)
                        | (int((high[high_base + element] >> 2) & 3) << 4),
                        int(low[low_base + element] >> 4)
                        | (int((high[high_base + element] >> 4) & 3) << 4),
                        int(low[low_base + 32 + element] >> 4)
                        | (int((high[high_base + element] >> 6) & 3) << 4),
                    )
                    for quadrant, quant_value in enumerate(quant_values):
                        output_index = half * 128 + quadrant * 32 + element
                        block_output[output_index] = (
                            np.float32(block_scale)
                            * int(scales[half * 8 + 2 * quadrant + scale_index])
                            * (quant_value - 32)
                        )
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


_FORMAT_CASES: list[
    tuple[
        str,
        int,
        Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
    ]
] = [
    ("q4_k", 144, _make_q4_k_weights),
    ("q5_k", 176, _make_q5_k_weights),
    ("q6_k", 210, _make_q6_k_weights),
]


@pytest.mark.parametrize(("format_name", "block_bytes", "make_weights"), _FORMAT_CASES)
@pytest.mark.parametrize("token_count", [1, 3])
@pytest.mark.parametrize("input_columns", [256, 512])
def test_gguf_q456_raw_matvec_matches_independent_reference(
    format_name: str,
    block_bytes: int,
    make_weights: Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
    token_count: int,
    input_columns: int,
) -> None:
    output_rows = 9
    raw, decoded = make_weights(output_rows, input_columns, token_count + input_columns)
    assert raw.shape == (output_rows, input_columns // 256 * block_bytes)
    torch.manual_seed(8421)
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = _quantize_q8_1(activations)
    output = torch.empty(token_count, output_rows, device="cuda", dtype=torch.float32)

    getattr(torch.ops._C, f"gguf_{format_name}_q8_1_raw_matvec")(
        scales, codes, torch.from_numpy(raw).cuda(), output
    )

    expected = _q8_1_reference(decoded, scales, codes)
    torch.testing.assert_close(output, expected, rtol=2e-3, atol=2e-3)


@pytest.mark.parametrize(("format_name", "_", "make_weights"), _FORMAT_CASES)
def test_gguf_q456_grouped_matmul_matches_raw_and_replays(
    format_name: str,
    _: int,
    make_weights: Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    token_count, output_rows, input_columns = 16, 16, 256
    raw, _ = make_weights(output_rows, input_columns, 8421)
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = _quantize_q8_1(activations)
    weights = torch.from_numpy(raw).cuda()
    raw_output = torch.empty(
        token_count, output_rows, device="cuda", dtype=torch.float32
    )
    grouped_output = torch.empty_like(raw_output)
    getattr(torch.ops._C, f"gguf_{format_name}_q8_1_raw_matvec")(
        scales, codes, weights, raw_output
    )
    grouped_op = getattr(torch.ops._C, f"gguf_{format_name}_q8_1_grouped_matmul")
    grouped_op(scales, codes, weights, grouped_output)
    torch.accelerator.synchronize()
    torch.testing.assert_close(grouped_output, raw_output, rtol=2e-3, atol=2e-3)
    before = grouped_output.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_op(scales, codes, weights, grouped_output)
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(grouped_output, before, rtol=0, atol=0)


def test_gguf_q4_k_embedding_decodes_only_selected_rows() -> None:
    raw, decoded = _make_q4_k_weights(6, 512, 8421)
    input_ids = torch.tensor([[0, 5], [2, 1]], device="cuda", dtype=torch.int64)
    output = torch.empty(2, 2, 512, device="cuda", dtype=torch.bfloat16)

    torch.ops._C.gguf_q4_k_embedding(input_ids, torch.from_numpy(raw).cuda(), output)

    expected = torch.from_numpy(decoded[[0, 5, 2, 1]]).reshape(2, 2, 512)
    torch.testing.assert_close(
        output.float().cpu(), expected.bfloat16().float(), rtol=0, atol=0
    )


@pytest.mark.parametrize(("format_name", "_", "make_weights"), _FORMAT_CASES)
def test_gguf_q456_raw_matvec_replays_in_cuda_graph(
    format_name: str,
    _: int,
    make_weights: Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    raw, _ = make_weights(8, 512, 8421)
    activations = torch.randn(1, 512, device="cuda", dtype=torch.bfloat16)
    scales, codes = _quantize_q8_1(activations)
    weights = torch.from_numpy(raw).cuda()
    output = torch.empty(1, 8, device="cuda", dtype=torch.float32)
    graph = torch.cuda.CUDAGraph()
    torch.accelerator.synchronize()
    with torch.cuda.graph(graph):
        getattr(torch.ops._C, f"gguf_{format_name}_q8_1_raw_matvec")(
            scales, codes, weights, output
        )
    graph.replay()
    first = output.clone()
    output.fill_(float("nan"))
    graph.replay()

    torch.testing.assert_close(output, first, rtol=0, atol=0)
