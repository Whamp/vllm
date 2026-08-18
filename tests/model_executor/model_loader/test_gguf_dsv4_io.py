# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from pathlib import Path

import torch

from vllm.model_executor.model_loader.gguf_dsv4_index import GGUFIndex
from vllm.model_executor.model_loader.gguf_dsv4_io import (
    load_gguf_plan_into_parameter,
    verify_gguf_sha256,
)
from vllm.model_executor.model_loader.gguf_dsv4_plan import (
    GGUFByteSpan,
    GGUFStridedSpan,
    GGUFTensorLoadPlan,
)


def test_load_gguf_plan_streams_strided_rows_and_casts(tmp_path: Path) -> None:
    path = tmp_path / "weights.gguf"
    prefix = bytes(range(32))
    rows = b"".join(bytes([row] * 8) for row in range(4))
    path.write_bytes(prefix + rows)
    index = GGUFIndex(path, 3, path.stat().st_size, 32, {}, ())
    plan = GGUFTensorLoadPlan(
        source_name="rows",
        target_name="target",
        source_type="F16",
        source_dims=(4, 4),
        spans=(
            GGUFStridedSpan(
                source_offset=2,
                target_offset=0,
                nbytes=4,
                count=4,
                source_stride=8,
                target_stride=4,
            ),
        ),
        target_nbytes=16,
    )
    target = torch.nn.Parameter(torch.empty(8, dtype=torch.float32), False)

    load_gguf_plan_into_parameter(index, plan, target, max_source_chunk_bytes=9)

    expected_half_bits = torch.tensor(
        [0x0000, 0x0000, 0x0101, 0x0101, 0x0202, 0x0202, 0x0303, 0x0303],
        dtype=torch.uint16,
    )
    expected = expected_half_bits.view(torch.float16).float()
    torch.testing.assert_close(target, expected, rtol=0, atol=0)


def test_load_gguf_plan_preserves_quantized_bytes_and_target_offset(
    tmp_path: Path,
) -> None:
    path = tmp_path / "quant.gguf"
    payload = bytes(range(64))
    path.write_bytes(bytes(32) + payload)
    index = GGUFIndex(path, 3, path.stat().st_size, 32, {}, ())
    plan = GGUFTensorLoadPlan(
        source_name="quant",
        target_name="target",
        source_type="Q8_0",
        source_dims=(32, 1),
        spans=(GGUFByteSpan(8, 10, 34),),
        target_nbytes=34,
    )
    target = torch.nn.Parameter(torch.full((64,), 255, dtype=torch.uint8), False)

    load_gguf_plan_into_parameter(index, plan, target)

    assert target[:10].tolist() == [255] * 10
    assert bytes(target[10:44].tolist()) == payload[8:42]
    assert target[44:].tolist() == [255] * 20


def test_verify_gguf_sha256_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "hash.gguf"
    path.write_bytes(b"verified bytes")

    digest = verify_gguf_sha256(path, expected_sha256=None)
    assert len(digest) == 64

    try:
        verify_gguf_sha256(path, expected_sha256="0" * 64)
    except ValueError as error:
        assert "GGUF SHA-256 mismatch" in str(error)
    else:
        raise AssertionError("Expected hash mismatch")
