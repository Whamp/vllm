# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Packed NVFP4 PLE sidecar loading without a resident BF16 table."""

import hashlib
import json
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from safetensors import safe_open

NVFP4_PLE_SIDECAR_LAYOUT = "group16_e2m1_e4m3scale_lownibblefirst"
NVFP4_PLE_BLOCK_SIZE = 16


def _get_native_nvfp4_ple_gather() -> Callable[..., None] | None:
    """Return the native packed-row gather when the extension provides it."""
    return getattr(torch.ops._C, "gather_nvfp4_ple_rows", None)


@dataclass(frozen=True)
class NvFp4PleSidecarManifest:
    """Validated storage geometry for one packed NVFP4 PLE sidecar."""

    directory: Path
    shard_count: int
    total_rows: int
    rows_per_shard: int
    width: int
    manifest_sha256: str


@dataclass(frozen=True)
class NvFp4PleGatherPartition:
    """One sidecar shard's local rows and their caller output positions."""

    shard_index: int
    local_rows: torch.Tensor
    output_positions: torch.Tensor


def plan_nvfp4_ple_sidecar_gather(
    row_ids: torch.Tensor,
    *,
    shard_count: int,
    rows_per_shard: int,
) -> tuple[NvFp4PleGatherPartition, ...]:
    """Partition global PLE row IDs into shard-local, order-restorable groups."""
    if row_ids.device.type != "cpu" or row_ids.ndim != 1:
        raise ValueError("PLE sidecar row IDs must be a one-dimensional CPU tensor")
    if row_ids.dtype not in (torch.int32, torch.int64):
        raise ValueError("PLE sidecar row IDs must use an integer dtype")
    if shard_count <= 0 or rows_per_shard <= 0:
        raise ValueError("PLE sidecar shard geometry must be positive")
    if row_ids.numel() == 0:
        return ()

    normalized_ids = row_ids.to(torch.int64)
    total_rows = shard_count * rows_per_shard
    if bool(torch.any((normalized_ids < 0) | (normalized_ids >= total_rows))):
        raise ValueError(
            f"PLE sidecar row ID is outside the sidecar row range [0,{total_rows})"
        )

    shard_ids = torch.div(normalized_ids, rows_per_shard, rounding_mode="floor")
    local_rows = normalized_ids - shard_ids * rows_per_shard
    order = torch.argsort(shard_ids, stable=True)
    sorted_shards = shard_ids.index_select(0, order)
    sorted_rows = local_rows.index_select(0, order)
    unique_shards, counts = torch.unique_consecutive(sorted_shards, return_counts=True)

    partitions: list[NvFp4PleGatherPartition] = []
    start = 0
    for shard_tensor, count_tensor in zip(unique_shards, counts, strict=True):
        count = int(count_tensor)
        end = start + count
        partitions.append(
            NvFp4PleGatherPartition(
                shard_index=int(shard_tensor),
                local_rows=sorted_rows[start:end],
                output_positions=order[start:end],
            )
        )
        start = end
    return tuple(partitions)


def _load_nvfp4_ple_sidecar_manifest(
    directory: str | Path,
    *,
    expected_width: int,
    expected_manifest_sha256: str,
    expected_rows: int,
) -> NvFp4PleSidecarManifest:
    sidecar_dir = Path(directory)
    manifest_path = sidecar_dir / "META.json"
    try:
        manifest_bytes = manifest_path.read_bytes()
    except OSError as error:
        raise ValueError(
            f"Cannot read PLE sidecar manifest: {manifest_path}"
        ) from error

    actual_sha256 = hashlib.sha256(manifest_bytes).hexdigest()
    if actual_sha256 != expected_manifest_sha256:
        raise ValueError(
            "PLE sidecar manifest SHA-256 mismatch: "
            f"expected {expected_manifest_sha256}, got {actual_sha256}"
        )
    try:
        payload = json.loads(manifest_bytes)
    except json.JSONDecodeError as error:
        raise ValueError("PLE sidecar manifest is not valid JSON") from error

    required_fields = {"layout", "shards", "rows", "width"}
    if not isinstance(payload, dict) or not required_fields.issubset(payload):
        raise ValueError("PLE sidecar manifest is missing required fields")
    if payload["layout"] != NVFP4_PLE_SIDECAR_LAYOUT:
        raise ValueError(f"Unsupported PLE sidecar layout: {payload['layout']!r}")

    shard_count = int(payload["shards"])
    total_rows = int(payload["rows"])
    width = int(payload["width"])
    if shard_count <= 0 or total_rows <= 0 or total_rows % shard_count:
        raise ValueError("PLE sidecar rows must divide evenly across positive shards")
    if width != expected_width:
        raise ValueError(
            f"PLE sidecar width mismatch: expected {expected_width}, got {width}"
        )
    if width % NVFP4_PLE_BLOCK_SIZE:
        raise ValueError(
            f"PLE sidecar width must be divisible by {NVFP4_PLE_BLOCK_SIZE}"
        )
    if total_rows != expected_rows:
        raise ValueError(
            f"PLE sidecar row mismatch: expected {expected_rows}, got {total_rows}"
        )

    return NvFp4PleSidecarManifest(
        directory=sidecar_dir,
        shard_count=shard_count,
        total_rows=total_rows,
        rows_per_shard=total_rows // shard_count,
        width=width,
        manifest_sha256=actual_sha256,
    )


def _read_nvfp4_sidecar_shard(
    manifest: NvFp4PleSidecarManifest,
    shard_index: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    path = manifest.directory / f"shard_{shard_index}.safetensors"
    if not path.is_file():
        raise ValueError(f"PLE sidecar shard is missing: {path}")
    try:
        with safe_open(path, framework="pt", device="cpu") as shard:
            keys = set(shard.keys())
            required_keys = {"weight_e2m1", "weight_scale", "weight_scale_2"}
            if keys != required_keys:
                raise ValueError(
                    f"PLE sidecar shard {shard_index} has unexpected tensors: "
                    f"{sorted(keys)}"
                )
            codes = shard.get_tensor("weight_e2m1")
            scales = shard.get_tensor("weight_scale")
            outer_scale = shard.get_tensor("weight_scale_2")
    except (OSError, RuntimeError) as error:
        raise ValueError(f"Cannot read PLE sidecar shard: {path}") from error

    expected_codes_shape = (manifest.rows_per_shard, manifest.width // 2)
    expected_scales_shape = (
        manifest.rows_per_shard,
        manifest.width // NVFP4_PLE_BLOCK_SIZE,
    )
    if codes.dtype != torch.uint8 or tuple(codes.shape) != expected_codes_shape:
        raise ValueError(
            f"PLE sidecar shard {shard_index} code tensor mismatch: expected "
            f"uint8 {expected_codes_shape}, got {codes.dtype} {tuple(codes.shape)}"
        )
    if (
        scales.dtype != torch.float8_e4m3fn
        or tuple(scales.shape) != expected_scales_shape
    ):
        raise ValueError(
            f"PLE sidecar shard {shard_index} scale tensor mismatch: expected "
            f"float8_e4m3fn {expected_scales_shape}, got "
            f"{scales.dtype} {tuple(scales.shape)}"
        )
    if outer_scale.dtype != torch.float32 or outer_scale.numel() != 1:
        raise ValueError(
            f"PLE sidecar shard {shard_index} outer scale must be one float32 value"
        )
    outer_scale = outer_scale.reshape(())
    if not bool(torch.isfinite(outer_scale)) or outer_scale.item() <= 0:
        raise ValueError(
            f"PLE sidecar shard {shard_index} outer scale must be finite and positive"
        )
    return codes, scales, outer_scale


_NVFP4_MAGNITUDES = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


class NvFp4PleSidecar:
    """Memory-mapped NVFP4 PLE shards with bounded BF16 row gathers."""

    def __init__(
        self,
        manifest: NvFp4PleSidecarManifest,
        code_shards: tuple[torch.Tensor, ...],
        scale_shards: tuple[torch.Tensor, ...],
        outer_scales: tuple[torch.Tensor, ...],
    ) -> None:
        self.manifest = manifest
        self.code_shards = code_shards
        self.scale_shards = scale_shards
        self.outer_scales = outer_scales
        self._outer_scales_tensor = torch.stack(outer_scales).to(torch.float32)
        self._dequant_lut = torch.tensor(
            _NVFP4_MAGNITUDES + tuple(-value for value in _NVFP4_MAGNITUDES),
            dtype=torch.float32,
        )

    @classmethod
    def open(
        cls,
        directory: str | Path,
        *,
        expected_rows: int,
        expected_width: int,
        expected_manifest_sha256: str,
    ) -> "NvFp4PleSidecar":
        """Open and validate every packed PLE shard without BF16 residency."""
        manifest = _load_nvfp4_ple_sidecar_manifest(
            directory,
            expected_width=expected_width,
            expected_manifest_sha256=expected_manifest_sha256,
            expected_rows=expected_rows,
        )
        code_shards: list[torch.Tensor] = []
        scale_shards: list[torch.Tensor] = []
        outer_scales: list[torch.Tensor] = []
        for shard_index in range(manifest.shard_count):
            codes, scales, outer_scale = _read_nvfp4_sidecar_shard(
                manifest, shard_index
            )
            code_shards.append(codes)
            scale_shards.append(scales)
            outer_scales.append(outer_scale)
        return cls(
            manifest,
            tuple(code_shards),
            tuple(scale_shards),
            tuple(outer_scales),
        )

    def gather_dequantized_rows(
        self,
        row_ids: torch.Tensor,
        output: torch.Tensor,
    ) -> None:
        """Gather and dequantize rows into caller-owned CPU floating storage."""
        if (
            output.device.type != "cpu"
            or output.dtype not in (torch.bfloat16, torch.float16, torch.float32)
            or output.ndim != 2
            or not output.is_contiguous()
        ):
            raise ValueError(
                "PLE sidecar output must be contiguous 2D CPU floating storage"
            )
        expected_shape = (row_ids.numel(), self.manifest.width)
        if tuple(output.shape) != expected_shape:
            raise ValueError(
                f"PLE sidecar output shape mismatch: expected {expected_shape}, "
                f"got {tuple(output.shape)}"
            )
        native_gather = _get_native_nvfp4_ple_gather()
        if native_gather is not None:
            native_gather(
                self.code_shards,
                self.scale_shards,
                self._outer_scales_tensor,
                row_ids,
                output,
                self.manifest.rows_per_shard,
            )
            return

        plan = plan_nvfp4_ple_sidecar_gather(
            row_ids,
            shard_count=self.manifest.shard_count,
            rows_per_shard=self.manifest.rows_per_shard,
        )
        for partition in plan:
            codes = self.code_shards[partition.shard_index].index_select(
                0, partition.local_rows
            )
            low_nibbles = (codes & 0xF).to(torch.int64)
            high_nibbles = (codes >> 4).to(torch.int64)
            nibbles = torch.stack((low_nibbles, high_nibbles), dim=-1).reshape(
                codes.shape[0], self.manifest.width
            )
            scales = self.scale_shards[partition.shard_index].index_select(
                0, partition.local_rows
            )
            expanded_scales = scales.to(torch.float32).repeat_interleave(
                NVFP4_PLE_BLOCK_SIZE, dim=1
            )
            dequantized = (
                self._dequant_lut[nibbles]
                * expanded_scales
                * self.outer_scales[partition.shard_index]
            ).to(output.dtype)
            output.index_copy_(0, partition.output_positions, dequantized)
