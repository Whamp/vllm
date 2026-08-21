# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from collections.abc import Callable

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GGUF IQ1/IQ3 kernels require CUDA"
)

IQ1_GRID_WORDS = {
    0: 0x00000000,
    1: 0x00000002,
    2: 0x00000101,
    17: 0x02000002,
    63: 0x02000022,
    127: 0x01021021,
    255: 0x01110222,
    1023: 0x11101210,
    2047: 0x22222222,
}
IQ3_GRID_WORDS = {
    0: 0x04040404,
    1: 0x04040414,
    2: 0x04040424,
    17: 0x040C142C,
    63: 0x0C04141C,
    127: 0x141C0C24,
    255: 0x3E341C04,
}


def _decode_iq1_grid(table_index: int) -> np.ndarray:
    word = IQ1_GRID_WORDS[table_index]
    packed = [(word >> (4 * index)) & 3 for index in range(8)]
    return np.array(packed[0::2] + packed[1::2])


def _sign_byte(selector: int) -> int:
    return selector | ((selector.bit_count() & 1) << 7)


def _make_iq1_s_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks_per_row = input_columns // 256
    raw = np.zeros((output_rows, blocks_per_row * 50), dtype=np.uint8)
    decoded = np.empty((output_rows, input_columns), dtype=np.float32)
    table_indices = np.array(tuple(IQ1_GRID_WORDS), dtype=np.int32)
    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            block_offset = block_index * 50
            block_scale = np.float16(rng.uniform(0.001, 0.1))
            raw[row, block_offset : block_offset + 2] = np.frombuffer(
                block_scale.tobytes(), dtype=np.uint8
            )
            for group_index in range(8):
                indices = rng.choice(table_indices, size=4, replace=True)
                scale = int(rng.integers(0, 8))
                negative_delta = bool(rng.integers(0, 2))
                high = scale << 12
                if negative_delta:
                    high |= 0x8000
                for part, table_index in enumerate(indices):
                    raw[row, block_offset + 2 + group_index * 4 + part] = (
                        table_index & 0xFF
                    )
                    high |= ((table_index >> 8) & 7) << (3 * part)
                    delta = -1.125 if negative_delta else -0.875
                    values = _decode_iq1_grid(int(table_index)) + delta
                    start = block_index * 256 + group_index * 32 + part * 8
                    decoded[row, start : start + 8] = (
                        np.float32(block_scale) * (2 * scale + 1) * values
                    )
                high_offset = block_offset + 34 + group_index * 2
                raw[row, high_offset : high_offset + 2] = np.frombuffer(
                    np.uint16(high).tobytes(), dtype=np.uint8
                )
    return raw, decoded


def _make_iq1_m_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks_per_row = input_columns // 256
    raw = np.zeros((output_rows, blocks_per_row * 56), dtype=np.uint8)
    decoded = np.empty((output_rows, input_columns), dtype=np.float32)
    table_indices = np.array(tuple(IQ1_GRID_WORDS), dtype=np.int32)
    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            block_offset = block_index * 56
            block_scale = np.float16(rng.uniform(0.001, 0.1))
            scale_bits = int(np.frombuffer(block_scale.tobytes(), dtype=np.uint16)[0])
            scale_words = np.empty(4, dtype=np.uint16)
            group_scale_codes = rng.integers(0, 8, size=(8, 2), dtype=np.uint16)
            for word_index in range(4):
                first_group = 2 * word_index
                packed = int(group_scale_codes[first_group, 0])
                packed |= int(group_scale_codes[first_group, 1]) << 3
                packed |= int(group_scale_codes[first_group + 1, 0]) << 6
                packed |= int(group_scale_codes[first_group + 1, 1]) << 9
                packed |= ((scale_bits >> (4 * word_index)) & 0xF) << 12
                scale_words[word_index] = packed
            raw[row, block_offset + 48 : block_offset + 56] = np.frombuffer(
                scale_words.tobytes(), dtype=np.uint8
            )
            for group_index in range(8):
                indices = rng.choice(table_indices, size=4, replace=True)
                high_bytes = [0, 0]
                for part, table_index in enumerate(indices):
                    negative_delta = bool(rng.integers(0, 2))
                    high_nibble = (table_index >> 8) & 7
                    if negative_delta:
                        high_nibble |= 8
                    high_bytes[part // 2] |= high_nibble << (4 * (part % 2))
                    raw[row, block_offset + group_index * 4 + part] = table_index & 0xFF
                    delta = -1.125 if negative_delta else -0.875
                    half = part // 2
                    scale = 2 * int(group_scale_codes[group_index, half]) + 1
                    values = _decode_iq1_grid(int(table_index)) + delta
                    start = block_index * 256 + group_index * 32 + part * 8
                    decoded[row, start : start + 8] = (
                        np.float32(block_scale) * scale * values
                    )
                raw[
                    row,
                    block_offset + 32 + group_index * 2 : block_offset
                    + 34
                    + group_index * 2,
                ] = high_bytes
    return raw, decoded


def _make_iq3_xxs_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    blocks_per_row = input_columns // 256
    raw = np.zeros((output_rows, blocks_per_row * 98), dtype=np.uint8)
    decoded = np.empty((output_rows, input_columns), dtype=np.float32)
    table_indices = np.array(tuple(IQ3_GRID_WORDS), dtype=np.uint8)
    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            block_offset = block_index * 98
            block_scale = np.float16(rng.uniform(0.001, 0.1))
            raw[row, block_offset : block_offset + 2] = np.frombuffer(
                block_scale.tobytes(), dtype=np.uint8
            )
            for group_index in range(8):
                indices = rng.choice(table_indices, size=8, replace=True)
                selectors = rng.integers(0, 128, size=4, dtype=np.uint32)
                scale = int(rng.integers(0, 16))
                raw[
                    row,
                    block_offset + 2 + group_index * 8 : block_offset
                    + 10
                    + group_index * 8,
                ] = indices
                scale_signs = np.uint32(scale << 28)
                for part, selector in enumerate(selectors):
                    scale_signs |= selector << np.uint32(7 * part)
                    signs = _sign_byte(int(selector))
                    values: list[int] = []
                    for table_index in indices[2 * part : 2 * part + 2]:
                        word = IQ3_GRID_WORDS[int(table_index)]
                        values.extend(
                            (word >> (8 * index)) & 0xFF for index in range(4)
                        )
                    signed = np.array(
                        [
                            -value if signs & (1 << index) else value
                            for index, value in enumerate(values)
                        ],
                        dtype=np.float32,
                    )
                    start = block_index * 256 + group_index * 32 + part * 8
                    decoded[row, start : start + 8] = (
                        np.float32(block_scale) * (0.5 + scale) * 0.5 * signed
                    )
                aux_offset = block_offset + 2 + 64 + group_index * 4
                raw[row, aux_offset : aux_offset + 4] = np.frombuffer(
                    scale_signs.tobytes(), dtype=np.uint8
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
    expected = dequantized @ decoded_weights.T
    return torch.from_numpy(expected).to(device=codes.device)


_FORMAT_CASES: list[
    tuple[
        str,
        int,
        Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
    ]
] = [
    ("iq1_s", 50, _make_iq1_s_weights),
    ("iq1_m", 56, _make_iq1_m_weights),
    ("iq3_xxs", 98, _make_iq3_xxs_weights),
]


@pytest.mark.parametrize(("format_name", "block_bytes", "make_weights"), _FORMAT_CASES)
@pytest.mark.parametrize("token_count", [1, 3])
@pytest.mark.parametrize("input_columns", [256, 512])
def test_gguf_iq1_iq3_raw_matvec_matches_independent_reference(
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


@pytest.mark.parametrize(
    ("format_name", "make_weights"),
    [("iq1_s", _make_iq1_s_weights), ("iq1_m", _make_iq1_m_weights)],
)
def test_gguf_iq1_indexed_gate_up_matches_independent_reference(
    format_name: str,
    make_weights: Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    token_count, topk, expert_count = 2, 2, 3
    input_columns, output_rows = 256, 7
    gate_raw = []
    up_raw = []
    gate_decoded = []
    up_decoded = []
    for expert in range(expert_count):
        raw, decoded = make_weights(output_rows, input_columns, 100 + expert)
        gate_raw.append(raw)
        gate_decoded.append(decoded)
        raw, decoded = make_weights(output_rows, input_columns, 200 + expert)
        up_raw.append(raw)
        up_decoded.append(decoded)
    torch.manual_seed(8421)
    activations = torch.randn(
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = _quantize_q8_1(activations)
    topk_ids = torch.tensor([[0, 2], [1, 0]], device="cuda", dtype=torch.int32)
    gate_output = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    up_output = torch.empty_like(gate_output)

    getattr(torch.ops._C, f"gguf_{format_name}_q8_1_indexed_gate_up")(
        scales,
        codes,
        torch.from_numpy(np.stack(gate_raw)).cuda(),
        torch.from_numpy(np.stack(up_raw)).cuda(),
        topk_ids,
        gate_output,
        up_output,
    )

    for token in range(token_count):
        token_scales = scales[token : token + 1]
        token_codes = codes[token : token + 1]
        for slot in range(topk):
            expert = int(topk_ids[token, slot])
            expected_gate = _q8_1_reference(
                gate_decoded[expert], token_scales, token_codes
            )[0]
            expected_up = _q8_1_reference(
                up_decoded[expert], token_scales, token_codes
            )[0]
            torch.testing.assert_close(
                gate_output[token, slot], expected_gate, rtol=2e-3, atol=2e-3
            )
            torch.testing.assert_close(
                up_output[token, slot], expected_up, rtol=2e-3, atol=2e-3
            )


@pytest.mark.parametrize(
    ("format_name", "make_weights"),
    [("iq1_s", _make_iq1_s_weights), ("iq1_m", _make_iq1_m_weights)],
)
def test_gguf_iq1_grouped_gate_up_matches_indexed_and_replays(
    format_name: str,
    make_weights: Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    import vllm._moe_C_stable_libtorch  # noqa: F401

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    token_count, topk = 16, 2
    expert_count, output_rows, input_columns = 4, 16, 256
    gate = np.stack(
        [
            make_weights(output_rows, input_columns, 5000 + expert)[0]
            for expert in range(expert_count)
        ]
    )
    up = np.stack(
        [
            make_weights(output_rows, input_columns, 6000 + expert)[0]
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
        token_count, input_columns, device="cuda", dtype=torch.bfloat16
    )
    scales, codes = _quantize_q8_1(activations)
    gate_weights = torch.from_numpy(gate).cuda()
    up_weights = torch.from_numpy(up).cuda()
    indexed_gate = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    indexed_up = torch.empty_like(indexed_gate)
    getattr(torch.ops._C, f"gguf_{format_name}_q8_1_indexed_gate_up")(
        scales,
        codes,
        gate_weights,
        up_weights,
        topk_ids,
        indexed_gate,
        indexed_up,
    )
    schedule = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=8,
        num_experts=expert_count,
    )
    grouped_gate = torch.empty_like(indexed_gate)
    grouped_up = torch.empty_like(indexed_up)
    grouped_op = getattr(torch.ops._C, f"gguf_{format_name}_q8_1_grouped_gate_up")
    grouped_op(
        scales,
        codes,
        gate_weights,
        up_weights,
        *schedule,
        grouped_gate,
        grouped_up,
        topk,
    )
    torch.accelerator.synchronize()
    for grouped, indexed in (
        (grouped_gate, indexed_gate),
        (grouped_up, indexed_up),
    ):
        error = (grouped - indexed).flatten()
        reference = indexed.flatten()
        assert error.square().mean().sqrt() / reference.square().mean().sqrt() < 0.01
        assert error.abs().mean() / reference.abs().mean() < 0.01
        assert error.abs().max() / reference.abs().max() < 0.025
        assert (
            torch.nn.functional.cosine_similarity(grouped.flatten(), reference, dim=0)
            > 0.9999
        )
    before_gate = grouped_gate.clone()
    before_up = grouped_up.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        grouped_op(
            scales,
            codes,
            gate_weights,
            up_weights,
            *schedule,
            grouped_gate,
            grouped_up,
            topk,
        )
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(grouped_gate, before_gate, rtol=0, atol=0)
    torch.testing.assert_close(grouped_up, before_up, rtol=0, atol=0)


def test_gguf_iq3_xxs_indexed_down_matches_independent_reference() -> None:
    token_count, topk, expert_count = 2, 2, 3
    input_columns, output_rows = 256, 7
    raw_weights = []
    decoded_weights = []
    for expert in range(expert_count):
        raw, decoded = _make_iq3_xxs_weights(output_rows, input_columns, 300 + expert)
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

    torch.ops._C.gguf_iq3_xxs_q8_1_indexed_down(
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


def test_gguf_iq3_xxs_grouped_down_matches_indexed_and_replays() -> None:
    import vllm._moe_C_stable_libtorch  # noqa: F401

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    token_count, topk, expert_count = 16, 2, 4
    input_columns, output_rows = 256, 16
    raw_weights = np.stack(
        [
            _make_iq3_xxs_weights(output_rows, input_columns, 7000 + expert)[0]
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
    torch.ops._C.gguf_iq3_xxs_q8_1_indexed_down(
        scales, codes, weights, topk_ids, indexed
    )
    schedule = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=8,
        num_experts=expert_count,
    )
    grouped_op = torch.ops._C.gguf_iq3_xxs_q8_1_grouped_down
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


@pytest.mark.parametrize(("format_name", "_", "make_weights"), _FORMAT_CASES)
def test_gguf_iq1_iq3_raw_matvec_replays_in_cuda_graph(
    format_name: str,
    _: int,
    make_weights: Callable[[int, int, int], tuple[np.ndarray, np.ndarray]],
) -> None:
    raw, _ = make_weights(8, 256, 8421)
    activations = torch.randn(1, 256, device="cuda", dtype=torch.bfloat16)
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
