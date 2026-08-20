# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GGUF IQ2_XXS kernels require CUDA"
)

GRID_WORDS = {
    0: 0x0808080808080808,
    1: 0x080808080808082B,
    2: 0x0808080808081919,
    255: 0x2B2B2B1908081908,
}
GRID_INDICES = np.array(tuple(GRID_WORDS), dtype=np.uint8)


def _even_parity_sign_byte(selector: int) -> int:
    return selector | ((selector.bit_count() & 1) << 7)


def _make_iq2_xxs_weights(
    output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    assert input_columns % 256 == 0
    rng = np.random.default_rng(seed)
    blocks_per_row = input_columns // 256
    raw = np.zeros((output_rows, blocks_per_row * 66), dtype=np.uint8)
    decoded = np.empty((output_rows, input_columns), dtype=np.float32)

    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            block_offset = block_index * 66
            scale = np.float16(rng.uniform(0.001, 0.1))
            raw[row, block_offset : block_offset + 2] = np.frombuffer(
                scale.tobytes(), dtype=np.uint8
            )
            for group_index in range(8):
                group_offset = block_offset + 2 + group_index * 8
                indices = rng.choice(GRID_INDICES, size=4, replace=True)
                selectors = rng.integers(0, 128, size=4, dtype=np.uint32)
                subscale = int(rng.integers(0, 16))
                raw[row, group_offset : group_offset + 4] = indices
                sign_word = np.uint32(subscale << 28)
                for grid_part in range(4):
                    sign_word |= selectors[grid_part] << np.uint32(7 * grid_part)
                raw[row, group_offset + 4 : group_offset + 8] = np.frombuffer(
                    sign_word.tobytes(), dtype=np.uint8
                )

                group_scale = (
                    np.float32(scale) * np.float32(0.5 + subscale) * np.float32(0.25)
                )
                output_offset = block_index * 256 + group_index * 32
                for grid_part, grid_index in enumerate(indices):
                    grid_values = np.frombuffer(
                        GRID_WORDS[int(grid_index)].to_bytes(8, "little"),
                        dtype=np.uint8,
                    )
                    sign_byte = _even_parity_sign_byte(int(selectors[grid_part]))
                    signs = np.array(
                        [-1.0 if sign_byte & (1 << bit) else 1.0 for bit in range(8)],
                        dtype=np.float32,
                    )
                    first = output_offset + grid_part * 8
                    decoded[row, first : first + 8] = (
                        group_scale * grid_values.astype(np.float32) * signs
                    )
    return raw, decoded


def _repack_iq2_xxs_aligned(
    raw: np.ndarray, input_columns: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    output_rows = raw.shape[0]
    blocks_per_row = input_columns // 256
    scales = np.empty((output_rows, blocks_per_row, 2), dtype=np.uint8)
    grid_bytes = np.empty((output_rows, blocks_per_row, 8, 4), dtype=np.uint8)
    scale_sign_bytes = np.empty_like(grid_bytes)
    for row in range(output_rows):
        for block_index in range(blocks_per_row):
            block_offset = block_index * 66
            scales[row, block_index] = raw[row, block_offset : block_offset + 2]
            groups = raw[row, block_offset + 2 : block_offset + 66].reshape(8, 8)
            grid_bytes[row, block_index] = groups[:, :4]
            scale_sign_bytes[row, block_index] = groups[:, 4:]
    return scales, grid_bytes, scale_sign_bytes


def _quantize_q8_1_reference(
    activations: torch.Tensor,
) -> tuple[np.ndarray, np.ndarray]:
    values = activations.float().cpu().numpy()
    token_count, input_columns = values.shape
    groups = values.reshape(token_count, input_columns // 32, 32)
    absolute_max = np.max(np.abs(groups), axis=2)
    float_scale = absolute_max / np.float32(127.0)
    safe_scale = np.where(absolute_max == 0, np.float32(1.0), float_scale)
    unrounded = groups / safe_scale[:, :, None]
    rounded = np.where(
        unrounded >= 0, np.floor(unrounded + 0.5), np.ceil(unrounded - 0.5)
    )
    codes = np.where(absolute_max[:, :, None] == 0, 0, rounded).astype(np.int8)
    return float_scale.astype(np.float16), codes.reshape(token_count, input_columns)


def _iq2_xxs_q8_1_reference(
    raw: np.ndarray,
    activation_scales: np.ndarray,
    activation_codes: np.ndarray,
    input_columns: int,
) -> np.ndarray:
    token_count = activation_codes.shape[0]
    output_rows = raw.shape[0]
    blocks_per_row = input_columns // 256
    output = np.zeros((token_count, output_rows), dtype=np.float32)
    for token_index in range(token_count):
        for output_row in range(output_rows):
            partial = np.float32(0.0)
            for block_index in range(blocks_per_row):
                block_offset = block_index * 66
                weight_scale = np.frombuffer(
                    raw[output_row, block_offset : block_offset + 2].tobytes(),
                    dtype=np.float16,
                ).astype(np.float32)[0]
                for group_index in range(8):
                    group_offset = block_offset + 2 + group_index * 8
                    grid_indices = raw[output_row, group_offset : group_offset + 4]
                    sign_word = int.from_bytes(
                        raw[output_row, group_offset + 4 : group_offset + 8].tobytes(),
                        "little",
                    )
                    signed_weights: list[int] = []
                    for grid_part, grid_index in enumerate(grid_indices):
                        grid = np.frombuffer(
                            GRID_WORDS[int(grid_index)].to_bytes(8, "little"),
                            dtype=np.uint8,
                        )
                        signs = _even_parity_sign_byte(
                            (sign_word >> (7 * grid_part)) & 127
                        )
                        signed_weights.extend(
                            -int(value) if signs & (1 << bit) else int(value)
                            for bit, value in enumerate(grid)
                        )
                    code_group = block_index * 8 + group_index
                    codes = activation_codes[
                        token_index, code_group * 32 : code_group * 32 + 32
                    ].astype(np.int32)
                    integer_sum = int(
                        np.dot(np.array(signed_weights, dtype=np.int32), codes)
                    )
                    integer_sum = int(integer_sum * ((sign_word >> 27) | 1) / 8)
                    activation_scale = np.float32(
                        activation_scales[token_index, code_group]
                    )
                    partial = np.float32(
                        partial
                        + np.float32(weight_scale * activation_scale)
                        * np.float32(integer_sum)
                    )
            output[token_index, output_row] = partial
    return output


def _run_raw_iq2_xxs(
    activations: torch.Tensor, packed_weights: torch.Tensor
) -> torch.Tensor:
    output = torch.empty(
        (activations.shape[0], packed_weights.shape[0]),
        device=activations.device,
        dtype=torch.float32,
    )
    torch.ops._C.gguf_iq2_xxs_raw_matvec(activations, packed_weights, output)
    return output


def _run_aligned_iq2_xxs(
    activations: torch.Tensor,
    scales: torch.Tensor,
    grid_bytes: torch.Tensor,
    scale_sign_bytes: torch.Tensor,
) -> torch.Tensor:
    output = torch.empty(
        (activations.shape[0], scales.shape[0]),
        device=activations.device,
        dtype=torch.float32,
    )
    torch.ops._C.gguf_iq2_xxs_aligned_matvec(
        activations, scales, grid_bytes, scale_sign_bytes, output
    )
    return output


def _quantize_q8_1(activations: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.empty(
        (activations.shape[0], activations.shape[1] // 32),
        device=activations.device,
        dtype=torch.float16,
    )
    codes = torch.empty_like(activations, dtype=torch.int8)
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    return scales, codes


def _run_q8_1_iq2_xxs(
    activation_scales: torch.Tensor,
    activation_codes: torch.Tensor,
    packed_weights: torch.Tensor,
    aligned_streams: tuple[torch.Tensor, torch.Tensor, torch.Tensor],
) -> tuple[torch.Tensor, torch.Tensor]:
    shape = (activation_codes.shape[0], packed_weights.shape[0])
    raw_output = torch.empty(shape, device=activation_codes.device, dtype=torch.float32)
    aligned_output = torch.empty_like(raw_output)
    torch.ops._C.gguf_iq2_xxs_q8_1_raw_matvec(
        activation_scales, activation_codes, packed_weights, raw_output
    )
    torch.ops._C.gguf_iq2_xxs_q8_1_aligned_matvec(
        activation_scales, activation_codes, *aligned_streams, aligned_output
    )
    return raw_output, aligned_output


@pytest.mark.parametrize("token_count", [1, 2, 4])
@pytest.mark.parametrize("input_columns", [256, 512])
def test_gguf_iq2_xxs_raw_and_aligned_match_reference(
    token_count: int, input_columns: int
) -> None:
    output_rows = 8
    raw, decoded_weights = _make_iq2_xxs_weights(
        output_rows, input_columns, seed=token_count * 1000 + input_columns
    )
    scales, grid_bytes, scale_sign_bytes = _repack_iq2_xxs_aligned(raw, input_columns)
    torch.manual_seed(1234)
    activations = torch.randn(
        (token_count, input_columns), device="cuda", dtype=torch.bfloat16
    )
    expected = activations.float() @ torch.from_numpy(decoded_weights).cuda().T

    raw_output = _run_raw_iq2_xxs(
        activations, torch.from_numpy(raw).cuda(non_blocking=True)
    )
    aligned_output = _run_aligned_iq2_xxs(
        activations,
        torch.from_numpy(scales).cuda(non_blocking=True),
        torch.from_numpy(grid_bytes).cuda(non_blocking=True),
        torch.from_numpy(scale_sign_bytes).cuda(non_blocking=True),
    )

    torch.testing.assert_close(raw_output, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(aligned_output, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(aligned_output, raw_output, rtol=0, atol=0)


@pytest.mark.parametrize("token_count", [1, 2, 4])
@pytest.mark.parametrize("input_columns", [256, 512])
def test_gguf_iq2_xxs_q8_1_dp4a_matches_integer_reference(
    token_count: int, input_columns: int
) -> None:
    output_rows = 8
    raw, _ = _make_iq2_xxs_weights(
        output_rows, input_columns, seed=2000 + token_count + input_columns
    )
    aligned = _repack_iq2_xxs_aligned(raw, input_columns)
    torch.manual_seed(5678)
    activations = torch.randn(
        (token_count, input_columns), device="cuda", dtype=torch.bfloat16
    )
    expected_scales, expected_codes = _quantize_q8_1_reference(activations)
    expected_output = _iq2_xxs_q8_1_reference(
        raw, expected_scales, expected_codes, input_columns
    )

    activation_scales, activation_codes = _quantize_q8_1(activations)
    raw_output, aligned_output = _run_q8_1_iq2_xxs(
        activation_scales,
        activation_codes,
        torch.from_numpy(raw).cuda(),
        tuple(torch.from_numpy(stream).cuda() for stream in aligned),
    )
    torch.accelerator.synchronize()

    np.testing.assert_array_equal(activation_scales.cpu().numpy(), expected_scales)
    np.testing.assert_array_equal(activation_codes.cpu().numpy(), expected_codes)
    expected = torch.from_numpy(expected_output).cuda()
    torch.testing.assert_close(raw_output, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(aligned_output, expected, rtol=2e-3, atol=2e-3)
    torch.testing.assert_close(aligned_output, raw_output, rtol=0, atol=0)


@pytest.mark.parametrize("token_count", [1, 2])
def test_gguf_iq2_xxs_indexed_gate_up_matches_selected_experts(
    token_count: int,
) -> None:
    input_columns, output_rows, expert_count, topk = 256, 8, 4, 2
    gate_raw = []
    up_raw = []
    for expert in range(expert_count):
        gate_raw.append(
            _make_iq2_xxs_weights(output_rows, input_columns, 3000 + expert)[0]
        )
        up_raw.append(
            _make_iq2_xxs_weights(output_rows, input_columns, 4000 + expert)[0]
        )
    gate = np.stack(gate_raw)
    up = np.stack(up_raw)
    topk_ids = np.array([[0, 3], [2, 1]][:token_count], dtype=np.int32)
    activations = torch.randn(
        (token_count, input_columns), device="cuda", dtype=torch.bfloat16
    )
    expected_scales, expected_codes = _quantize_q8_1_reference(activations)
    expected_gate = np.empty((token_count, topk, output_rows), dtype=np.float32)
    expected_up = np.empty_like(expected_gate)
    for token_index in range(token_count):
        for slot in range(topk):
            expert = int(topk_ids[token_index, slot])
            expected_gate[token_index, slot] = _iq2_xxs_q8_1_reference(
                gate[expert],
                expected_scales[token_index : token_index + 1],
                expected_codes[token_index : token_index + 1],
                input_columns,
            )[0]
            expected_up[token_index, slot] = _iq2_xxs_q8_1_reference(
                up[expert],
                expected_scales[token_index : token_index + 1],
                expected_codes[token_index : token_index + 1],
                input_columns,
            )[0]

    activation_scales, activation_codes = _quantize_q8_1(activations)
    gate_output = torch.empty(
        (token_count, topk, output_rows), device="cuda", dtype=torch.float32
    )
    up_output = torch.empty_like(gate_output)
    torch.ops._C.gguf_iq2_xxs_q8_1_indexed_gate_up(
        activation_scales,
        activation_codes,
        torch.from_numpy(gate).cuda(),
        torch.from_numpy(up).cuda(),
        torch.from_numpy(topk_ids).cuda(),
        gate_output,
        up_output,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        gate_output, torch.from_numpy(expected_gate).cuda(), rtol=2e-3, atol=2e-3
    )
    torch.testing.assert_close(
        up_output, torch.from_numpy(expected_up).cuda(), rtol=2e-3, atol=2e-3
    )


def test_gguf_iq2_xxs_ops_replay_in_cuda_graph() -> None:
    input_columns, output_rows = 512, 8
    raw, _ = _make_iq2_xxs_weights(output_rows, input_columns, seed=42)
    scales, grid_bytes, scale_sign_bytes = _repack_iq2_xxs_aligned(raw, input_columns)
    activations = torch.randn((2, input_columns), device="cuda", dtype=torch.bfloat16)
    raw_weights = torch.from_numpy(raw).cuda()
    aligned_tensors = tuple(
        torch.from_numpy(array).cuda()
        for array in (scales, grid_bytes, scale_sign_bytes)
    )
    raw_output = torch.empty((2, output_rows), device="cuda", dtype=torch.float32)
    aligned_output = torch.empty_like(raw_output)
    q8_scales = torch.empty(
        (2, input_columns // 32), device="cuda", dtype=torch.float16
    )
    q8_codes = torch.empty_like(activations, dtype=torch.int8)
    q8_raw_output = torch.empty_like(raw_output)
    q8_aligned_output = torch.empty_like(raw_output)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_iq2_xxs_raw_matvec(activations, raw_weights, raw_output)
        torch.ops._C.gguf_iq2_xxs_aligned_matvec(
            activations, *aligned_tensors, aligned_output
        )
        torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, q8_scales, q8_codes)
        torch.ops._C.gguf_iq2_xxs_q8_1_raw_matvec(
            q8_scales, q8_codes, raw_weights, q8_raw_output
        )
        torch.ops._C.gguf_iq2_xxs_q8_1_aligned_matvec(
            q8_scales, q8_codes, *aligned_tensors, q8_aligned_output
        )
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(aligned_output, raw_output, rtol=0, atol=0)
    torch.testing.assert_close(q8_aligned_output, q8_raw_output, rtol=0, atol=0)


def test_gguf_iq2_xxs_grouped_gate_up_matches_indexed_and_replays() -> None:
    import vllm._moe_C_stable_libtorch  # noqa: F401

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    token_count, topk = 16, 2
    expert_count, output_rows, input_columns = 4, 16, 256
    gate_pairs = [
        _make_iq2_xxs_weights(output_rows, input_columns, seed=5000 + expert)
        for expert in range(expert_count)
    ]
    up_pairs = [
        _make_iq2_xxs_weights(output_rows, input_columns, seed=6000 + expert)
        for expert in range(expert_count)
    ]
    gate = np.stack([pair[0] for pair in gate_pairs])
    up = np.stack([pair[0] for pair in up_pairs])
    gate_dequantized = np.stack([pair[1] for pair in gate_pairs])
    up_dequantized = np.stack([pair[1] for pair in up_pairs])
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
    expected_scales, expected_codes = _quantize_q8_1_reference(activations)
    scales, codes = _quantize_q8_1(activations)
    gate_weights = torch.from_numpy(gate).cuda()
    up_weights = torch.from_numpy(up).cuda()
    indexed_gate = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    indexed_up = torch.empty_like(indexed_gate)
    torch.ops._C.gguf_iq2_xxs_q8_1_indexed_gate_up(
        scales,
        codes,
        gate_weights,
        up_weights,
        topk_ids,
        indexed_gate,
        indexed_up,
    )
    sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=8,
        num_experts=expert_count,
    )
    grouped_gate = torch.empty_like(indexed_gate)
    grouped_up = torch.empty_like(indexed_up)
    torch.ops._C.gguf_iq2_xxs_q8_1_grouped_gate_up(
        scales,
        codes,
        gate_weights,
        up_weights,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        grouped_gate,
        grouped_up,
        topk,
    )
    torch.accelerator.synchronize()
    dequantized_activations = (
        expected_codes.reshape(token_count, input_columns // 32, 32).astype(np.float32)
        * expected_scales[:, :, None].astype(np.float32)
    ).reshape(token_count, input_columns)
    expected_gate = np.empty((token_count, topk, output_rows), dtype=np.float32)
    expected_up = np.empty_like(expected_gate)
    topk_ids_cpu = topk_ids.cpu().numpy()
    for token in range(token_count):
        for slot in range(topk):
            expert = int(topk_ids_cpu[token, slot])
            expected_gate[token, slot] = (
                dequantized_activations[token] @ gate_dequantized[expert].T
            )
            expected_up[token, slot] = (
                dequantized_activations[token] @ up_dequantized[expert].T
            )
    torch.testing.assert_close(
        grouped_gate,
        torch.from_numpy(expected_gate).cuda(),
        rtol=2e-3,
        atol=2e-3,
    )
    torch.testing.assert_close(
        grouped_up,
        torch.from_numpy(expected_up).cuda(),
        rtol=2e-3,
        atol=2e-3,
    )
    for grouped, indexed in ((grouped_gate, indexed_gate), (grouped_up, indexed_up)):
        error = (grouped - indexed).flatten()
        reference = indexed.flatten()
        assert error.square().mean().sqrt() / reference.square().mean().sqrt() < 0.01
        assert error.abs().mean() / reference.abs().mean() < 0.01
        assert error.abs().max() / reference.abs().max() < 0.025
        assert (
            torch.nn.functional.cosine_similarity(grouped.flatten(), reference, dim=0)
            > 0.9999
        )

    grouped_gate_before_replay = grouped_gate.clone()
    grouped_up_before_replay = grouped_up.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_iq2_xxs_q8_1_grouped_gate_up(
            scales,
            codes,
            gate_weights,
            up_weights,
            sorted_ids,
            expert_ids,
            num_tokens_padded,
            grouped_gate,
            grouped_up,
            topk,
        )
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(grouped_gate, grouped_gate_before_replay, rtol=0, atol=0)
    torch.testing.assert_close(grouped_up, grouped_up_before_replay, rtol=0, atol=0)
