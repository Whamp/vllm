# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
import torch

from tests.kernels.test_fused_deepseek_v4_qnorm_rope_kv_insert import (
    apply_rope_gptj_last_k,
    make_cos_sin_cache,
)
from vllm.models.deepseek_v4.cache_layout import FP4_DS_MLA_CACHE_LAYOUT
from vllm.models.deepseek_v4.common.ops import (
    dequantize_and_gather_k_cache,
    quantize_and_insert_k_cache,
)

HEAD_DIM = 512
NOPE_DIM = 448
ROPE_DIM = 64
GROUP_SIZE = 32
BLOCK_SIZE = 64

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")


def _encode_reference(rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    nope = rows[:, :NOPE_DIM].float().reshape(-1, NOPE_DIM // GROUP_SIZE, GROUP_SIZE)
    amax = nope.abs().amax(dim=-1, keepdim=True).clamp_min(6.0 * (2.0**-126))
    exponent = torch.ceil(torch.log2(amax / 6.0)).clamp(-127.0, 127.0)
    scaled = (nope * torch.exp2(-exponent)).clamp(-6.0, 6.0)
    magnitude = scaled.abs()
    code = torch.zeros_like(magnitude, dtype=torch.int32)
    code = torch.where(magnitude > 0.25, 1, code)
    code = torch.where(magnitude >= 0.75, 2, code)
    code = torch.where(magnitude > 1.25, 3, code)
    code = torch.where(magnitude >= 1.75, 4, code)
    code = torch.where(magnitude > 2.5, 5, code)
    code = torch.where(magnitude >= 3.5, 6, code)
    code = torch.where(magnitude > 5.0, 7, code)
    nibble = code.to(torch.uint8) | (torch.signbit(scaled).to(torch.uint8) << 3)
    nibble = nibble.reshape(-1, NOPE_DIM)
    packed = nibble[:, 0::2] | (nibble[:, 1::2] << 4)
    rope = rows[:, NOPE_DIM:].contiguous().view(torch.uint8)
    token_data = torch.cat((packed, rope), dim=-1)
    scales = torch.zeros(
        rows.shape[0],
        FP4_DS_MLA_CACHE_LAYOUT.scale_bytes,
        dtype=torch.uint8,
        device=rows.device,
    )
    scales[:, : NOPE_DIM // GROUP_SIZE] = (exponent.squeeze(-1) + 127).to(torch.uint8)
    return token_data, scales


def _decode_reference(token_data: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    packed = token_data[:, : NOPE_DIM // 2]
    code = torch.empty(
        token_data.shape[0], NOPE_DIM, dtype=torch.uint8, device=token_data.device
    )
    code[:, 0::2] = packed & 0x0F
    code[:, 1::2] = packed >> 4
    lut = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
        device=token_data.device,
    )
    value = lut[(code & 0x07).long()]
    value = torch.where((code & 0x08) != 0, -value, value)
    scale = torch.exp2(
        scales[:, : NOPE_DIM // GROUP_SIZE].float() - 127.0
    ).repeat_interleave(GROUP_SIZE, dim=-1)
    nope = (value * scale).to(torch.bfloat16)
    rope = token_data[:, NOPE_DIM // 2 :].contiguous().view(torch.bfloat16)
    return torch.cat((nope, rope), dim=-1)


def _extract_rows(
    cache: torch.Tensor, slots: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    flat = cache.view(cache.shape[0], -1)
    token_data = []
    scales = []
    for slot in slots.tolist():
        block = slot // BLOCK_SIZE
        position = slot % BLOCK_SIZE
        data_start = position * FP4_DS_MLA_CACHE_LAYOUT.token_data_bytes
        scale_start = (
            BLOCK_SIZE * FP4_DS_MLA_CACHE_LAYOUT.token_data_bytes
            + position * FP4_DS_MLA_CACHE_LAYOUT.scale_bytes
        )
        token_data.append(
            flat[
                block,
                data_start : data_start + FP4_DS_MLA_CACHE_LAYOUT.token_data_bytes,
            ]
        )
        scales.append(
            flat[block, scale_start : scale_start + FP4_DS_MLA_CACHE_LAYOUT.scale_bytes]
        )
    return torch.stack(token_data), torch.stack(scales)


def test_fp4_triton_writer_is_byte_exact_across_cache_blocks() -> None:
    torch.manual_seed(211)
    device = "cuda"
    slots = torch.tensor([0, 1, 63, 64, 69], dtype=torch.int64, device=device)
    rows = torch.randn(slots.numel(), HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.full(
        (2, BLOCK_SIZE, FP4_DS_MLA_CACHE_LAYOUT.row_bytes),
        0xA5,
        dtype=torch.uint8,
        device=device,
    )

    quantize_and_insert_k_cache(
        rows,
        cache,
        slots,
        block_size=BLOCK_SIZE,
        cache_dtype="fp4_ds_mla",
    )
    actual_data, actual_scales = _extract_rows(cache, slots)
    expected_data, expected_scales = _encode_reference(rows)

    torch.testing.assert_close(actual_data, expected_data, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
    assert torch.all(actual_scales[:, 14:] == 0)
    # Unwritten token 2 remains untouched in both segregated sections.
    untouched_data, untouched_scales = _extract_rows(
        cache, torch.tensor([2], dtype=torch.int64)
    )
    assert torch.all(untouched_data == 0xA5)
    assert torch.all(untouched_scales == 0xA5)


def test_fp4_gather_matches_independent_decoder() -> None:
    torch.manual_seed(223)
    device = "cuda"
    num_tokens = 79
    rows = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(
        (2, BLOCK_SIZE, FP4_DS_MLA_CACHE_LAYOUT.row_bytes),
        dtype=torch.uint8,
        device=device,
    )
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)
    quantize_and_insert_k_cache(
        rows, cache, slots, block_size=BLOCK_SIZE, cache_dtype="fp4_ds_mla"
    )
    token_data, scales = _extract_rows(cache, slots.cpu())
    expected = _decode_reference(token_data, scales)

    out = torch.empty(1, num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    dequantize_and_gather_k_cache(
        out,
        cache,
        seq_lens=torch.tensor([num_tokens], dtype=torch.int32, device=device),
        gather_lens=None,
        block_table=torch.tensor([[0, 1]], dtype=torch.int32, device=device),
        block_size=BLOCK_SIZE,
        offset=0,
        cache_dtype="fp4_ds_mla",
    )

    torch.testing.assert_close(out[0], expected, rtol=0, atol=0)


def test_fused_fp4_writer_matches_independent_reference() -> None:
    if not hasattr(
        torch.ops._C,
        "fused_deepseek_v4_qnorm_rope_kv_rope_fp4_quant_insert",
    ):
        pytest.skip("fused FP4 DeepSeek V4 writer is not built")

    torch.manual_seed(227)
    device = "cuda"
    num_tokens, num_heads = 17, 8
    positions = torch.arange(num_tokens, dtype=torch.int64, device=device)
    q = torch.randn(
        num_tokens, num_heads, HEAD_DIM, dtype=torch.bfloat16, device=device
    )
    kv = torch.randn(num_tokens, HEAD_DIM, dtype=torch.bfloat16, device=device)
    cache = torch.zeros(
        (1, BLOCK_SIZE, FP4_DS_MLA_CACHE_LAYOUT.row_bytes),
        dtype=torch.uint8,
        device=device,
    )
    cos_sin = make_cos_sin_cache(128, ROPE_DIM, torch.float32, device)
    slots = torch.arange(num_tokens, dtype=torch.int64, device=device)

    torch.ops._C.fused_deepseek_v4_qnorm_rope_kv_rope_fp4_quant_insert(
        q,
        kv,
        cache.view(1, -1),
        slots,
        positions,
        cos_sin,
        num_heads,
        1e-6,
        BLOCK_SIZE,
    )

    rotated = apply_rope_gptj_last_k(kv, positions, cos_sin).to(torch.bfloat16)
    expected_data, expected_scales = _encode_reference(rotated)
    actual_data, actual_scales = _extract_rows(cache, slots.cpu())
    torch.testing.assert_close(actual_data, expected_data, rtol=0, atol=0)
    torch.testing.assert_close(actual_scales, expected_scales, rtol=0, atol=0)
