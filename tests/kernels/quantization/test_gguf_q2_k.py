# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import numpy as np
import pytest
import torch

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GGUF Q2_K kernels require CUDA"
)


def _make_q2_k_weights(
    expert_count: int, output_rows: int, input_columns: int, seed: int
) -> tuple[np.ndarray, np.ndarray]:
    assert input_columns % 256 == 0
    rng = np.random.default_rng(seed)
    blocks_per_row = input_columns // 256
    raw = rng.integers(
        0,
        256,
        size=(expert_count, output_rows, blocks_per_row * 84),
        dtype=np.uint8,
    )
    decoded = np.empty((expert_count, output_rows, input_columns), dtype=np.float32)
    for expert in range(expert_count):
        for row in range(output_rows):
            for block_index in range(blocks_per_row):
                block_offset = block_index * 84
                scale = np.float16(rng.uniform(0.001, 0.05))
                min_scale = np.float16(rng.uniform(0.001, 0.05))
                raw[expert, row, block_offset + 80 : block_offset + 82] = np.frombuffer(
                    scale.tobytes(), dtype=np.uint8
                )
                raw[expert, row, block_offset + 82 : block_offset + 84] = np.frombuffer(
                    min_scale.tobytes(), dtype=np.uint8
                )
                scales = raw[expert, row, block_offset : block_offset + 16]
                codes = raw[expert, row, block_offset + 16 : block_offset + 80]
                output_offset = block_index * 256
                scale_index = 0
                code_base = 0
                for chunk in range(2):
                    for shift_stage in range(4):
                        shift = 2 * shift_stage
                        for half in range(2):
                            packed_scale = int(scales[scale_index])
                            decoded_scale = np.float32(scale) * np.float32(
                                packed_scale & 0xF
                            )
                            decoded_min = np.float32(min_scale) * np.float32(
                                packed_scale >> 4
                            )
                            first = code_base + 16 * half
                            q2 = ((codes[first : first + 16] >> shift) & 3).astype(
                                np.float32
                            )
                            weight_offset = (
                                output_offset
                                + 128 * chunk
                                + 32 * shift_stage
                                + 16 * half
                            )
                            decoded[expert, row, weight_offset : weight_offset + 16] = (
                                decoded_scale * q2 - decoded_min
                            )
                            scale_index += 1
                    code_base += 32
    return raw, decoded


def _quantize_q8_on_gpu(
    activations: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    scales = torch.empty(
        (activations.shape[0], activations.shape[1] // 32),
        device=activations.device,
        dtype=torch.float16,
    )
    codes = torch.empty_like(activations, dtype=torch.int8)
    torch.ops._C.gguf_quantize_bf16_to_q8_1(activations, scales, codes)
    return scales, codes


@pytest.mark.parametrize("token_count", [1, 2])
@pytest.mark.parametrize("input_columns", [256, 512])
def test_gguf_q2_k_indexed_down_matches_dequantized_reference(
    token_count: int, input_columns: int
) -> None:
    expert_count, output_rows, topk = 4, 8, 2
    raw, decoded_weights = _make_q2_k_weights(
        expert_count, output_rows, input_columns, seed=5000 + input_columns
    )
    topk_ids = np.array([[0, 3], [2, 1]][:token_count], dtype=np.int32)
    activations = torch.randn(
        (token_count * topk, input_columns),
        device="cuda",
        dtype=torch.bfloat16,
    )
    activation_scales, activation_codes = _quantize_q8_on_gpu(activations)
    torch.accelerator.synchronize()
    scales_cpu = activation_scales.cpu().numpy().astype(np.float32)
    codes_cpu = activation_codes.cpu().numpy().astype(np.float32)
    dequantized_activations = (
        codes_cpu.reshape(token_count * topk, input_columns // 32, 32)
        * scales_cpu[:, :, None]
    ).reshape(token_count * topk, input_columns)
    expected = np.empty((token_count, topk, output_rows), dtype=np.float32)
    for token_index in range(token_count):
        for slot in range(topk):
            expert = int(topk_ids[token_index, slot])
            activation_row = token_index * topk + slot
            expected[token_index, slot] = (
                dequantized_activations[activation_row] @ decoded_weights[expert].T
            )

    output = torch.empty(
        (token_count, topk, output_rows), device="cuda", dtype=torch.float32
    )
    torch.ops._C.gguf_q2_k_q8_1_indexed_down(
        activation_scales,
        activation_codes,
        torch.from_numpy(raw).cuda(),
        torch.from_numpy(topk_ids).cuda(),
        output,
    )
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        output, torch.from_numpy(expected).cuda(), rtol=2e-3, atol=2e-3
    )


def test_gguf_q2_k_indexed_down_replays_in_cuda_graph() -> None:
    expert_count, output_rows, input_columns, topk = 4, 8, 512, 2
    raw, _ = _make_q2_k_weights(expert_count, output_rows, input_columns, 42)
    activations = torch.randn(
        (topk, input_columns), device="cuda", dtype=torch.bfloat16
    )
    scales, codes = _quantize_q8_on_gpu(activations)
    topk_ids = torch.tensor([[0, 3]], device="cuda", dtype=torch.int32)
    output = torch.empty((1, topk, output_rows), device="cuda", dtype=torch.float32)
    weights = torch.from_numpy(raw).cuda()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_q2_k_q8_1_indexed_down(
            scales, codes, weights, topk_ids, output
        )
    graph.replay()
    torch.accelerator.synchronize()
    assert torch.isfinite(output).all()


def test_gguf_q2_k_grouped_down_matches_reference_and_replays() -> None:
    import vllm._moe_C_stable_libtorch  # noqa: F401

    from vllm.model_executor.layers.fused_moe.moe_align_block_size import (
        moe_align_block_size,
    )

    token_count, topk = 16, 2
    expert_count, output_rows, input_columns = 4, 16, 512
    raw, decoded_weights = _make_q2_k_weights(
        expert_count, output_rows, input_columns, seed=7000
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
    scales, codes = _quantize_q8_on_gpu(activations)
    weights = torch.from_numpy(raw).cuda()
    indexed = torch.empty(
        token_count, topk, output_rows, device="cuda", dtype=torch.float32
    )
    torch.ops._C.gguf_q2_k_q8_1_indexed_down(scales, codes, weights, topk_ids, indexed)
    sorted_ids, expert_ids, num_tokens_padded = moe_align_block_size(
        topk_ids=topk_ids,
        block_size=8,
        num_experts=expert_count,
    )
    grouped = torch.empty_like(indexed)
    torch.ops._C.gguf_q2_k_q8_1_grouped_down(
        scales,
        codes,
        weights,
        sorted_ids,
        expert_ids,
        num_tokens_padded,
        grouped,
    )
    torch.accelerator.synchronize()

    scales_cpu = scales.cpu().numpy().astype(np.float32)
    codes_cpu = codes.cpu().numpy().astype(np.float32)
    dequantized_activations = (
        codes_cpu.reshape(token_count * topk, input_columns // 32, 32)
        * scales_cpu[:, :, None]
    ).reshape(token_count * topk, input_columns)
    expected = np.empty((token_count, topk, output_rows), dtype=np.float32)
    topk_ids_cpu = topk_ids.cpu().numpy()
    for token in range(token_count):
        for slot in range(topk):
            assignment = token * topk + slot
            expert = int(topk_ids_cpu[token, slot])
            expected[token, slot] = (
                dequantized_activations[assignment] @ decoded_weights[expert].T
            )
    expected_gpu = torch.from_numpy(expected).cuda()
    torch.testing.assert_close(grouped, expected_gpu, rtol=2e-3, atol=2e-3)

    for actual in (grouped, indexed):
        error = (actual - expected_gpu).flatten()
        reference = expected_gpu.flatten()
        assert error.square().mean().sqrt() / reference.square().mean().sqrt() < 0.01
        assert error.abs().mean() / reference.abs().mean() < 0.01
        assert error.abs().max() / reference.abs().max() < 0.025
        assert (
            torch.nn.functional.cosine_similarity(actual.flatten(), reference, dim=0)
            > 0.9999
        )

    grouped_before_replay = grouped.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_q2_k_q8_1_grouped_down(
            scales,
            codes,
            weights,
            sorted_ids,
            expert_ids,
            num_tokens_padded,
            grouped,
        )
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(grouped, grouped_before_replay, rtol=0, atol=0)
