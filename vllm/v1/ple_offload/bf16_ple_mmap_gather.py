# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Direct BF16 row gathering from a content-addressed safetensors PLE table."""

import ctypes
import json
import mmap
import os
import struct
import threading
from pathlib import Path
from typing import BinaryIO

import numpy as np
import regex as re
import torch
from numpy.typing import NDArray

_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_MAX_SAFETENSORS_HEADER_BYTES = 16 << 20
_QWEN_PLE_LAYER_PATTERN = re.compile(r"(?:^|\.)layers\.(\d+)\.ple$")


class Bf16PleMmapGather:
    """Gather BF16 PLE rows from one content-addressed safetensors file.

    Construction validates the full tensor inventory and applies random-access
    advice. Payload integrity relies on the content-addressed download receipt;
    construction does not reread the 95 GiB production file to hash it.
    """

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        expected_sha256: str,
        native_library_path: str | Path,
        tensor_prefix: str,
        total_rows: int,
        width: int,
    ) -> None:
        self._lock = threading.Lock()
        self._mapping: mmap.mmap | None = None
        self._table: NDArray[np.uint16] | None = None
        if not _SHA256_PATTERN.fullmatch(expected_sha256):
            raise ValueError(
                "BF16 PLE expected SHA-256 must be 64 lowercase hex digits"
            )
        if total_rows <= 0 or width <= 0:
            raise ValueError("BF16 PLE geometry must have positive rows and width")

        resolved_path = Path(checkpoint_path).resolve(strict=True)
        if resolved_path.name != expected_sha256:
            raise ValueError(
                "BF16 PLE content-addressed filename mismatch: "
                f"expected {expected_sha256}, got {resolved_path.name}"
            )
        with resolved_path.open("rb") as checkpoint:
            data_offset = _validate_bf16_ple_safetensors(
                checkpoint,
                file_size=os.fstat(checkpoint.fileno()).st_size,
                tensor_prefix=tensor_prefix,
                total_rows=total_rows,
                width=width,
            )
            library = ctypes.CDLL(str(native_library_path))
            kernel = library.vllm_gather_bf16_ple_rows
            kernel.argtypes = [
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_int64,
                ctypes.c_void_p,
                ctypes.c_int64,
                ctypes.c_void_p,
            ]
            kernel.restype = ctypes.c_int
            mapping = mmap.mmap(checkpoint.fileno(), 0, access=mmap.ACCESS_READ)
            try:
                mapping.madvise(mmap.MADV_RANDOM)
                table = np.ndarray(
                    (total_rows, width),
                    dtype=np.uint16,
                    buffer=mapping,
                    offset=data_offset,
                )
            except Exception:
                mapping.close()
                raise

        self._mapping = mapping
        self._table = table
        self._total_rows = total_rows
        self._width = width
        self._library = library
        self._kernel = kernel

    def gather_into(self, row_ids: torch.Tensor, output: torch.Tensor) -> None:
        """Copy selected BF16 rows into a contiguous CPU output tensor.

        Raises:
            ValueError: Row IDs, output geometry, or row bounds are invalid.
            RuntimeError: The mapping is closed or the native gather fails.
        """
        if row_ids.device.type != "cpu":
            raise ValueError("BF16 PLE row IDs must be on CPU")
        if (
            output.device.type != "cpu"
            or output.dtype != torch.bfloat16
            or not output.is_contiguous()
            or output.shape != (row_ids.numel(), self._width)
        ):
            raise ValueError(
                "BF16 PLE output must be contiguous CPU BF16 with shape "
                f"{(row_ids.numel(), self._width)}"
            )
        with self._lock:
            table = self._table
            if table is None:
                raise RuntimeError("BF16 PLE gather is closed")
            row_ids = row_ids.reshape(-1).to(dtype=torch.int64).contiguous()
            if row_ids.numel() == 0:
                return
            result = self._kernel(
                table.ctypes.data,
                self._total_rows,
                self._width,
                row_ids.data_ptr(),
                row_ids.numel(),
                output.data_ptr(),
            )
        if result == -2:
            raise ValueError("row ID is outside the BF16 PLE table")
        if result != 0:
            raise RuntimeError(f"native BF16 PLE gather failed with status {result}")

    def close(self) -> None:
        """Release the direct BF16 PLE file mapping; safe to call repeatedly."""
        with self._lock:
            mapping = self._mapping
            if mapping is None:
                return
            self._table = None
            self._mapping = None
            mapping.close()

    def __del__(self) -> None:
        self.close()


def attach_bf16_ple_mmap_table(
    *,
    layer_name: str,
    layer: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str,
    native_library_path: str | Path,
) -> str:
    """Attach direct BF16 PLE lookup to one Qwen n-gram embedding parameter.

    Returns:
        The exact parameter name replaced with an empty BF16 stub.

    Raises:
        ValueError: The layer or table parameter does not match Qwen PLE.
        OSError: The checkpoint cannot be resolved or mapped.
        RuntimeError: The native library or attachment operation fails.
    """
    layer_match = _QWEN_PLE_LAYER_PATTERN.search(layer_name)
    if layer_match is None:
        raise ValueError(
            f"BF16 PLE layer name does not identify a Qwen layer: {layer_name}"
        )
    named_parameters = sorted(
        layer.named_parameters(), key=lambda item: item[1].numel(), reverse=True
    )
    if not named_parameters:
        raise ValueError(f"BF16 PLE layer has no table parameter: {layer_name}")
    parameter_name, parameter = named_parameters[0]
    if (
        parameter_name != "ple_embedding.ngram_embedding.weight"
        or parameter.device.type != "cpu"
        or parameter.dtype != torch.bfloat16
        or parameter.dim() != 2
    ):
        raise ValueError(
            "BF16 PLE table parameter must be "
            "ple_embedding.ngram_embedding.weight as a two-dimensional CPU BF16 tensor"
        )
    total_rows, width = parameter.shape
    tensor_prefix = (
        "model.language_model.layers."
        f"{layer_match.group(1)}.ple.ple_embedding.ngram_embedding"
    )
    table = Bf16PleMmapGather(
        checkpoint_path=checkpoint_path,
        expected_sha256=expected_sha256,
        native_library_path=native_library_path,
        tensor_prefix=tensor_prefix,
        total_rows=total_rows,
        width=width,
    )
    try:
        owner = layer
        parameter_parts = parameter_name.split(".")
        for part in parameter_parts[:-1]:
            owner = getattr(owner, part)
        owner._ple_quant = table  # type: ignore[attr-defined]
        getattr(owner, parameter_parts[-1]).data = parameter.new_empty((0, width))
    except Exception:
        table.close()
        raise
    return parameter_name


def _validate_bf16_ple_safetensors(
    checkpoint: BinaryIO,
    *,
    file_size: int,
    tensor_prefix: str,
    total_rows: int,
    width: int,
) -> int:
    raw_header_size = checkpoint.read(8)
    if len(raw_header_size) != 8:
        raise ValueError("BF16 PLE safetensors file is missing its header length")
    header_size = struct.unpack("<Q", raw_header_size)[0]
    if not 0 < header_size <= _MAX_SAFETENSORS_HEADER_BYTES:
        raise ValueError(f"BF16 PLE safetensors header size is invalid: {header_size}")
    raw_header = checkpoint.read(header_size)
    if len(raw_header) != header_size:
        raise ValueError("BF16 PLE safetensors header is truncated")
    try:
        header = json.loads(raw_header)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("BF16 PLE safetensors header is not valid JSON") from error

    if not isinstance(header, dict):
        raise ValueError("BF16 PLE safetensors header root must be an object")

    entries: list[tuple[int, int, int, int]] = []
    prefix = f"{tensor_prefix}.shard_"
    suffix = ".weight"
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(name, str) or not isinstance(metadata, dict):
            raise ValueError("BF16 PLE safetensors tensor metadata is invalid")
        if not name.startswith(prefix) or not name.endswith(suffix):
            raise ValueError(f"BF16 PLE safetensors contains unexpected tensor: {name}")
        shard_text = name[len(prefix) : -len(suffix)]
        if not shard_text.isdigit():
            raise ValueError(f"BF16 PLE shard name is invalid: {name}")
        shard_index = int(shard_text)
        if metadata.get("dtype") != "BF16":
            raise ValueError(f"BF16 PLE tensor has the wrong dtype: {name}")
        shape = metadata.get("shape")
        offsets = metadata.get("data_offsets")
        if (
            not isinstance(shape, list)
            or len(shape) != 2
            or not all(type(dimension) is int for dimension in shape)
            or shape[1] != width
            or shape[0] <= 0
            or not isinstance(offsets, list)
            or len(offsets) != 2
            or not all(type(offset) is int for offset in offsets)
        ):
            raise ValueError(f"BF16 PLE tensor geometry is invalid: {name}")
        start, end = offsets
        expected_bytes = shape[0] * width * 2
        if start < 0 or end - start != expected_bytes:
            raise ValueError(f"BF16 PLE tensor byte range is invalid: {name}")
        entries.append((shard_index, start, end, shape[0]))

    entries.sort()
    if not entries or [entry[0] for entry in entries] != list(range(len(entries))):
        raise ValueError("BF16 PLE shard indices must be contiguous from zero")
    if sum(entry[3] for entry in entries) != total_rows:
        raise ValueError("BF16 PLE shard rows do not match the configured table")
    if entries[0][1] != 0 or any(
        left[2] != right[1] for left, right in zip(entries, entries[1:])
    ):
        raise ValueError("BF16 PLE shard payloads must be contiguous in shard order")

    data_offset = 8 + header_size
    if file_size != data_offset + entries[-1][2]:
        raise ValueError("BF16 PLE safetensors file size does not match its payload")
    return data_offset
