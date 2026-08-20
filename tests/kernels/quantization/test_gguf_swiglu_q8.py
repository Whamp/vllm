# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch
import torch.nn.functional as F

from vllm.platforms import current_platform

pytestmark = pytest.mark.skipif(
    not current_platform.is_cuda(), reason="GGUF SwiGLU Q8 kernel requires CUDA"
)


@pytest.mark.parametrize("token_count", [1, 4])
def test_gguf_swiglu_weighted_q8_matches_reference_and_replays(
    token_count: int,
) -> None:
    topk, intermediate_size, clamp_limit = 2, 64, 10.0
    torch.manual_seed(20260818)
    gate = (
        torch.randn(
            token_count,
            topk,
            intermediate_size,
            device="cuda",
            dtype=torch.float32,
        )
        * 6.0
    )
    up = torch.randn_like(gate) * 6.0
    router_weights = torch.rand(token_count, topk, device="cuda", dtype=torch.float32)
    scales = torch.empty(
        token_count * topk,
        intermediate_size // 32,
        device="cuda",
        dtype=torch.float16,
    )
    codes = torch.empty(
        token_count * topk,
        intermediate_size,
        device="cuda",
        dtype=torch.int8,
    )

    torch.ops._C.gguf_swiglu_weighted_q8_1(
        gate,
        up,
        router_weights,
        scales,
        codes,
        clamp_limit,
    )
    torch.accelerator.synchronize()
    reference = (
        F.silu(torch.clamp(gate, max=clamp_limit))
        * torch.clamp(up, min=-clamp_limit, max=clamp_limit)
        * router_weights[..., None]
    ).reshape(token_count * topk, intermediate_size)
    reconstructed = (
        codes.float().reshape(token_count * topk, intermediate_size // 32, 32)
        * scales.float()[:, :, None]
    ).reshape_as(reference)
    error = (reconstructed - reference).flatten()
    reference_flat = reference.flatten()
    nrmse = error.square().mean().sqrt() / reference_flat.square().mean().sqrt()
    normalized_mae = error.abs().mean() / reference_flat.abs().mean()
    max_ratio = error.abs().max() / reference_flat.abs().max()
    cosine = torch.nn.functional.cosine_similarity(
        reconstructed.flatten(), reference_flat, dim=0
    )
    assert nrmse < 0.01
    assert normalized_mae < 0.0125
    assert max_ratio < 0.025
    assert cosine > 0.9999

    generic_scales = torch.empty_like(scales)
    generic_codes = torch.empty_like(codes)
    torch.ops._C.gguf_quantize_bf16_to_q8_1(
        reference.to(torch.bfloat16), generic_scales, generic_codes
    )
    generic_reconstructed = (
        generic_codes.float().reshape(token_count * topk, intermediate_size // 32, 32)
        * generic_scales.float()[:, :, None]
    ).reshape_as(reference)
    generic_error = (generic_reconstructed - reference).flatten()
    assert nrmse <= (
        generic_error.square().mean().sqrt() / reference_flat.square().mean().sqrt()
    )
    assert normalized_mae <= generic_error.abs().mean() / reference_flat.abs().mean()
    assert max_ratio <= generic_error.abs().max() / reference_flat.abs().max()

    scales_before_replay = scales.clone()
    codes_before_replay = codes.clone()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        torch.ops._C.gguf_swiglu_weighted_q8_1(
            gate,
            up,
            router_weights,
            scales,
            codes,
            clamp_limit,
        )
    graph.replay()
    torch.accelerator.synchronize()
    torch.testing.assert_close(scales, scales_before_replay, rtol=0, atol=0)
    torch.testing.assert_close(codes, codes_before_replay, rtol=0, atol=0)
