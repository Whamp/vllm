# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded file IO and tensor copying for DeepSeek V4 GGUF load plans."""

from __future__ import annotations

import hashlib
import os
from collections.abc import Iterator
from pathlib import Path

import numpy as np
import torch

from vllm.model_executor.model_loader.gguf_dsv4_index import GGUFIndex
from vllm.model_executor.model_loader.gguf_dsv4_plan import (
    GGUFByteSpan,
    GGUFSpan,
    GGUFStridedSpan,
    GGUFTensorLoadPlan,
)

_QUANTIZED_GGUF_TYPES = {"Q8_0", "Q2_K", "IQ2_XXS"}
_SOURCE_TORCH_DTYPES = {
    "F32": torch.float32,
    "F16": torch.float16,
    "I8": torch.int8,
    "I32": torch.int32,
    "I64": torch.int64,
    "F64": torch.float64,
    "BF16": torch.bfloat16,
}


def verify_gguf_sha256(
    path: str | Path,
    *,
    expected_sha256: str | None,
    chunk_bytes: int = 64 * 1024 * 1024,
) -> str:
    """Stream and optionally verify the complete GGUF content SHA-256."""
    if chunk_bytes <= 0:
        raise ValueError("GGUF hash chunk size must be positive")
    digest = hashlib.sha256()
    with Path(path).open("rb") as gguf_file:
        while chunk := gguf_file.read(chunk_bytes):
            digest.update(chunk)
    actual = digest.hexdigest()
    if expected_sha256 is not None and actual != expected_sha256.lower():
        raise ValueError(
            f"GGUF SHA-256 mismatch: expected {expected_sha256.lower()}, got {actual}"
        )
    return actual


def _pread_exact(file_descriptor: int, length: int, offset: int) -> bytes:
    data = os.pread(file_descriptor, length, offset)
    if len(data) != length:
        raise ValueError(
            f"Short GGUF pread at offset {offset}: expected {length}, got {len(data)}"
        )
    return data


def _iter_byte_span_chunks(
    file_descriptor: int,
    data_start: int,
    span: GGUFByteSpan,
    max_source_chunk_bytes: int,
) -> Iterator[tuple[int, bytes]]:
    consumed = 0
    while consumed < span.nbytes:
        length = min(max_source_chunk_bytes, span.nbytes - consumed)
        yield (
            span.target_offset + consumed,
            _pread_exact(
                file_descriptor,
                length,
                data_start + span.source_offset + consumed,
            ),
        )
        consumed += length


def _iter_strided_span_chunks(
    file_descriptor: int,
    data_start: int,
    span: GGUFStridedSpan,
    max_source_chunk_bytes: int,
) -> Iterator[tuple[int, bytes]]:
    if span.count <= 0 or span.nbytes <= 0:
        raise ValueError("GGUF strided span count and width must be positive")
    if span.source_stride < span.nbytes or span.target_stride < span.nbytes:
        raise ValueError("GGUF strided span stride is smaller than its width")
    rows_per_chunk = max(1, max_source_chunk_bytes // span.source_stride)
    for first_row in range(0, span.count, rows_per_chunk):
        row_count = min(rows_per_chunk, span.count - first_row)
        source_offset = span.source_offset + first_row * span.source_stride
        read_length = (row_count - 1) * span.source_stride + span.nbytes
        source = _pread_exact(file_descriptor, read_length, data_start + source_offset)
        view = np.ndarray(
            shape=(row_count, span.nbytes),
            dtype=np.uint8,
            buffer=source,
            strides=(span.source_stride, 1),
        )
        contiguous = view.copy().tobytes()
        yield span.target_offset + first_row * span.target_stride, contiguous


def iter_gguf_span_chunks(
    file_descriptor: int,
    data_start: int,
    span: GGUFSpan,
    *,
    max_source_chunk_bytes: int,
) -> Iterator[tuple[int, bytes]]:
    """Yield contiguous target chunks from one contiguous or strided span."""
    if max_source_chunk_bytes <= 0:
        raise ValueError("GGUF source chunk size must be positive")
    if isinstance(span, GGUFByteSpan):
        yield from _iter_byte_span_chunks(
            file_descriptor, data_start, span, max_source_chunk_bytes
        )
    else:
        yield from _iter_strided_span_chunks(
            file_descriptor, data_start, span, max_source_chunk_bytes
        )


def _load_plan_contribution_bytes(
    index: GGUFIndex,
    plan: GGUFTensorLoadPlan,
    max_source_chunk_bytes: int,
) -> tuple[int, bytearray]:
    target_base = min(span.target_offset for span in plan.spans)
    contribution = bytearray(plan.target_nbytes)
    file_descriptor = os.open(index.path, os.O_RDONLY)
    try:
        for span in plan.spans:
            for target_offset, chunk in iter_gguf_span_chunks(
                file_descriptor,
                index.data_start,
                span,
                max_source_chunk_bytes=max_source_chunk_bytes,
            ):
                relative_offset = target_offset - target_base
                end = relative_offset + len(chunk)
                if relative_offset < 0 or end > len(contribution):
                    raise ValueError(
                        f"GGUF target span exceeds contribution for {plan.source_name}"
                    )
                contribution[relative_offset:end] = chunk
    finally:
        os.close(file_descriptor)
    return target_base, contribution


def load_gguf_plan_into_parameter(
    index: GGUFIndex,
    plan: GGUFTensorLoadPlan,
    parameter: torch.nn.Parameter,
    *,
    max_source_chunk_bytes: int = 64 * 1024 * 1024,
) -> None:
    """Load one planned source contribution into a flat runtime parameter."""
    target_base, contribution = _load_plan_contribution_bytes(
        index, plan, max_source_chunk_bytes
    )
    target = parameter.data.reshape(-1)
    if plan.source_type in _QUANTIZED_GGUF_TYPES:
        if target.dtype != torch.uint8:
            raise ValueError(
                f"Quantized GGUF target {plan.target_name} must be uint8, "
                f"got {target.dtype}"
            )
        start = target_base
        end = start + len(contribution)
        if end > target.numel():
            raise ValueError(f"GGUF target {plan.target_name} is too small")
        source = torch.frombuffer(contribution, dtype=torch.uint8)
        target[start:end].copy_(source.to(target.device))
        return

    try:
        source_dtype = _SOURCE_TORCH_DTYPES[plan.source_type]
    except KeyError as error:
        raise ValueError(
            f"Unsupported ordinary GGUF source type {plan.source_type}"
        ) from error
    source_element_size = torch.tensor([], dtype=source_dtype).element_size()
    if len(contribution) % source_element_size:
        raise ValueError(f"GGUF source {plan.source_name} has partial elements")
    target_element_size = target.element_size()
    if target_base % source_element_size:
        raise ValueError(f"GGUF target offset for {plan.source_name} is misaligned")
    source = torch.frombuffer(contribution, dtype=source_dtype)
    start = target_base // source_element_size
    if source_element_size != target_element_size and target_base != 0:
        raise ValueError(
            f"Stacked GGUF cast for {plan.source_name} changes element width"
        )
    end = start + source.numel()
    if end > target.numel():
        raise ValueError(f"GGUF target {plan.target_name} is too small")
    target[start:end].copy_(source.to(device=target.device, dtype=target.dtype))
