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

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_iq2_xxs_raw_matvec(activations, raw_weights, raw_output)
        torch.ops._C.gguf_iq2_xxs_aligned_matvec(
            activations, *aligned_tensors, aligned_output
        )
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(aligned_output, raw_output, rtol=0, atol=0)
