# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fail-closed bounded GGUF v3 header and tensor-directory parser."""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class GGUFTypeSpec:
    name: str
    block_elements: int
    block_bytes: int


GGUF_TYPES: dict[int, GGUFTypeSpec] = {
    0: GGUFTypeSpec("F32", 1, 4),
    1: GGUFTypeSpec("F16", 1, 2),
    8: GGUFTypeSpec("Q8_0", 32, 34),
    9: GGUFTypeSpec("Q8_1", 32, 36),
    10: GGUFTypeSpec("Q2_K", 256, 84),
    12: GGUFTypeSpec("Q4_K", 256, 144),
    13: GGUFTypeSpec("Q5_K", 256, 176),
    14: GGUFTypeSpec("Q6_K", 256, 210),
    16: GGUFTypeSpec("IQ2_XXS", 256, 66),
    18: GGUFTypeSpec("IQ3_XXS", 256, 98),
    19: GGUFTypeSpec("IQ1_S", 256, 50),
    24: GGUFTypeSpec("I8", 1, 1),
    26: GGUFTypeSpec("I32", 1, 4),
    27: GGUFTypeSpec("I64", 1, 8),
    28: GGUFTypeSpec("F64", 1, 8),
    29: GGUFTypeSpec("IQ1_M", 256, 56),
    30: GGUFTypeSpec("BF16", 1, 2),
    39: GGUFTypeSpec("MXFP4", 32, 17),
}

GGUF_TYPE_SPECS_BY_NAME = {
    type_spec.name: type_spec for type_spec in GGUF_TYPES.values()
}
GGUF_QUANTIZED_TYPE_NAMES = frozenset(
    type_name
    for type_name, type_spec in GGUF_TYPE_SPECS_BY_NAME.items()
    if type_spec.block_elements > 1
)


@dataclass(frozen=True)
class GGUFTensorEntry:
    name: str
    type_id: int
    dims: tuple[int, ...]
    offset: int

    @property
    def type_spec(self) -> GGUFTypeSpec:
        try:
            return GGUF_TYPES[self.type_id]
        except KeyError as error:
            raise ValueError(
                f"Tensor {self.name} uses unknown GGUF tensor type {self.type_id}"
            ) from error

    @property
    def type_name(self) -> str:
        return self.type_spec.name

    @property
    def nbytes(self) -> int:
        return self.compute_nbytes(self.type_id, self.dims)

    @staticmethod
    def compute_nbytes(type_id: int, dims: tuple[int, ...]) -> int:
        try:
            spec = GGUF_TYPES[type_id]
        except KeyError as error:
            raise ValueError(f"unknown GGUF tensor type {type_id}") from error
        if not dims or any(dim <= 0 for dim in dims):
            raise ValueError(f"GGUF tensor dimensions must be positive: {dims}")
        first_dim_blocks = math.ceil(dims[0] / spec.block_elements)
        return first_dim_blocks * spec.block_bytes * math.prod(dims[1:])


@dataclass(frozen=True)
class GGUFIndex:
    path: Path
    version: int
    file_size: int
    data_start: int
    metadata: dict[str, Any]
    tensors: tuple[GGUFTensorEntry, ...]


class _GGUFHeaderReader:
    def __init__(self, data: bytes):
        self.data = data
        self.position = 0

    def read(self, length: int) -> bytes:
        end = self.position + length
        if length < 0 or end > len(self.data):
            raise ValueError(
                "GGUF header exceeds bounded read; increase the explicit header limit"
            )
        value = self.data[self.position : end]
        self.position = end
        return value

    def unpack(self, format_string: str):
        size = struct.calcsize(format_string)
        return struct.unpack(format_string, self.read(size))[0]

    def u32(self) -> int:
        return self.unpack("<I")

    def u64(self) -> int:
        return self.unpack("<Q")

    def string(self) -> str:
        length = self.u64()
        try:
            return self.read(length).decode("utf-8")
        except UnicodeDecodeError as error:
            raise ValueError("GGUF header contains invalid UTF-8") from error


_GGUF_SCALAR_FORMATS = {
    0: "<B",
    1: "<b",
    2: "<H",
    3: "<h",
    4: "<I",
    5: "<i",
    6: "<f",
    7: "<?",
    10: "<Q",
    11: "<q",
    12: "<d",
}


def _read_gguf_metadata_value(reader: _GGUFHeaderReader, value_type: int):
    if value_type == 8:
        return reader.string()
    if value_type == 9:
        element_type = reader.u32()
        count = reader.u64()
        if count > 2_000_000:
            raise ValueError(f"Implausible GGUF metadata array length {count}")
        return [_read_gguf_metadata_value(reader, element_type) for _ in range(count)]
    try:
        format_string = _GGUF_SCALAR_FORMATS[value_type]
    except KeyError as error:
        raise ValueError(f"unknown GGUF metadata value type {value_type}") from error
    return reader.unpack(format_string)


def parse_gguf_index(
    path: str | Path, *, header_limit_bytes: int = 16 * 1024 * 1024
) -> GGUFIndex:
    """Read a bounded GGUF v3 header and validate the complete tensor directory."""
    path = Path(path)
    file_size = path.stat().st_size
    with path.open("rb") as gguf_file:
        header = gguf_file.read(header_limit_bytes)
    reader = _GGUFHeaderReader(header)
    if reader.read(4) != b"GGUF":
        raise ValueError(f"Not a GGUF file: {path}")
    version = reader.u32()
    if version != 3:
        raise ValueError(f"Expected GGUF v3, got v{version}")
    tensor_count = reader.u64()
    metadata_count = reader.u64()
    if tensor_count > 100_000 or metadata_count > 100_000:
        raise ValueError(
            f"Implausible GGUF counts: {tensor_count} tensors, "
            f"{metadata_count} metadata entries"
        )

    metadata: dict[str, Any] = {}
    for _ in range(metadata_count):
        key = reader.string()
        if key in metadata:
            raise ValueError(f"Duplicate GGUF metadata key {key}")
        metadata[key] = _read_gguf_metadata_value(reader, reader.u32())

    tensors = []
    tensor_names = set()
    for _ in range(tensor_count):
        name = reader.string()
        if name in tensor_names:
            raise ValueError(f"Duplicate GGUF tensor name {name}")
        tensor_names.add(name)
        dimension_count = reader.u32()
        if not 1 <= dimension_count <= 8:
            raise ValueError(
                f"Tensor {name} has invalid dimension count {dimension_count}"
            )
        dims = tuple(reader.u64() for _ in range(dimension_count))
        type_id = reader.u32()
        offset = reader.u64()
        entry = GGUFTensorEntry(name, type_id, dims, offset)
        _ = entry.nbytes
        tensors.append(entry)

    alignment = metadata.get("general.alignment", 32)
    if not isinstance(alignment, int) or alignment <= 0 or alignment & (alignment - 1):
        raise ValueError(f"Invalid GGUF alignment {alignment!r}")
    data_start = (reader.position + alignment - 1) & -alignment
    payload_size = file_size - data_start
    if payload_size < 0:
        raise ValueError("GGUF data section starts beyond end of file")
    previous_end = 0
    for entry in sorted(tensors, key=lambda tensor: tensor.offset):
        if entry.offset < previous_end:
            raise ValueError(f"GGUF tensor overlap at {entry.name}")
        end = entry.offset + entry.nbytes
        if end > payload_size:
            raise ValueError(
                f"GGUF tensor {entry.name} exceeds data section: {end} > {payload_size}"
            )
        previous_end = end
    return GGUFIndex(
        path=path,
        version=version,
        file_size=file_size,
        data_start=data_start,
        metadata=metadata,
        tensors=tuple(tensors),
    )
