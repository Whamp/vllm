#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Derive a Qwen3.8 MTP view with GPTQ-packed INT4 routed experts."""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

_INT4_VALUES_PER_INT32 = 8
_INT4_MASK = 0xF
_UINT32_MASK = 0xFFFFFFFF
_GPTQ_SYMMETRIC_QZERO_WORD = 0x77777777
_MINIMUM_FP16_SCALE = 1e-6
_MTP_EXPERT_EXCLUSION_PATTERN = re.compile(
    r"^mtp\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj$"
)
_MTP_EXPERT_WEIGHT_PATTERN = re.compile(
    r"^(mtp\.layers\.\d+\.mlp\.experts\.\d+\.(?:gate|up|down)_proj)\.weight$"
)
_MTP_SIDECAR_FILENAME = "model_extra_tensors.safetensors"
_CONFIG_FILENAME = "config.json"
_INDEX_FILENAME = "model.safetensors.index.json"
_PROVENANCE_FILENAMES = frozenset({"DERIVATION.txt", "SHA256SUMS", "SYMLINKS.tsv"})
_PRODUCTION_MTP_EXPERT_WEIGHT_COUNT = 1536


@dataclass(frozen=True)
class ConversionSummary:
    """Auditable result of one MTP expert checkpoint conversion."""

    source_view: str
    output_view: str
    quantized_weights: int
    removed_exclusions: int
    source_sidecar_bytes: int
    derived_sidecar_bytes: int
    bytes_saved: int
    source_sidecar_sha256: str
    derived_sidecar_sha256: str


def derive_int4_mtp_config(
    source_config: dict[str, Any],
    *,
    group_size: int = 128,
    runtime_layer_index: int = 48,
) -> tuple[dict[str, Any], int]:
    """Return a config selecting W4 only for the runtime MTP routed experts."""

    derived_config = copy.deepcopy(source_config)
    quantization_config = derived_config["quantization_config"]
    extra_config = quantization_config["extra_config"]
    expert_exclusions = [
        key for key in extra_config if _MTP_EXPERT_EXCLUSION_PATTERN.fullmatch(key)
    ]
    for key in expert_exclusions:
        del extra_config[key]
    runtime_expert_prefix = f"mtp.layers.{runtime_layer_index}.mlp.experts"
    extra_config[runtime_expert_prefix] = {
        "bits": 4,
        "group_size": group_size,
        "sym": True,
        "data_type": "int",
    }
    return derived_config, len(expert_exclusions)


def pack_gptq_int4_rows(unpacked: torch.Tensor) -> torch.Tensor:
    """Pack uint4 codes along the GPTQ input (row) dimension."""

    if unpacked.ndim != 2:
        raise ValueError(f"GPTQ INT4 input must be rank 2, got {unpacked.shape}")
    size_k, size_n = unpacked.shape
    if size_k % _INT4_VALUES_PER_INT32 != 0:
        raise ValueError(f"GPTQ INT4 input size {size_k} is not divisible by 8")
    if torch.any((unpacked < 0) | (unpacked > _INT4_MASK)):
        raise ValueError("GPTQ INT4 input codes must be between 0 and 15")

    groups = unpacked.to(torch.int64).reshape(
        size_k // _INT4_VALUES_PER_INT32,
        _INT4_VALUES_PER_INT32,
        size_n,
    )
    packed = torch.zeros(
        (size_k // _INT4_VALUES_PER_INT32, size_n),
        dtype=torch.int64,
        device=unpacked.device,
    )
    for nibble_index in range(_INT4_VALUES_PER_INT32):
        packed |= groups[:, nibble_index] << (4 * nibble_index)
    return packed.to(torch.int32).contiguous()


def unpack_gptq_int4_rows(packed: torch.Tensor, *, size_k: int) -> torch.Tensor:
    """Unpack GPTQ row-packed int32 values into uint4 codes."""

    if packed.ndim != 2:
        raise ValueError(f"GPTQ packed input must be rank 2, got {packed.shape}")
    if size_k != packed.shape[0] * _INT4_VALUES_PER_INT32:
        raise ValueError(
            f"GPTQ unpack size {size_k} does not match packed shape {packed.shape}"
        )

    unsigned = packed.to(torch.int64) & _UINT32_MASK
    unpacked = torch.empty(
        (size_k, packed.shape[1]),
        dtype=torch.int32,
        device=packed.device,
    )
    for nibble_index in range(_INT4_VALUES_PER_INT32):
        unpacked[nibble_index::_INT4_VALUES_PER_INT32] = (
            unsigned >> (4 * nibble_index)
        ) & _INT4_MASK
    return unpacked.contiguous()


def quantize_symmetric_int4_gptq(
    weight_out_in: torch.Tensor,
    *,
    group_size: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Quantize one linear weight into AutoRound-compatible GPTQ tensors."""

    if weight_out_in.ndim != 2:
        raise ValueError(
            f"GPTQ source weight must be rank 2, got {weight_out_in.shape}"
        )
    size_n, size_k = weight_out_in.shape
    if size_k % group_size != 0:
        raise ValueError(
            f"GPTQ input size {size_k} is not divisible by group size {group_size}"
        )
    if size_n % _INT4_VALUES_PER_INT32 != 0:
        raise ValueError(f"GPTQ output size {size_n} is not divisible by 8")
    if not torch.isfinite(weight_out_in).all():
        raise ValueError("GPTQ source weight contains non-finite values")

    weight_k_n = weight_out_in.float().t().contiguous()
    num_groups = size_k // group_size
    grouped_weight = weight_k_n.reshape(num_groups, group_size, size_n)
    scales = (grouped_weight.abs().amax(dim=1) / 7.0).clamp(min=_MINIMUM_FP16_SCALE)
    scales = scales.to(torch.float16)
    stored_scales = scales.float().unsqueeze(1)
    signed_codes = (grouped_weight / stored_scales).round().clamp(-8, 7)
    unsigned_codes = (signed_codes + 8).to(torch.int32).reshape(size_k, size_n)

    qweight = pack_gptq_int4_rows(unsigned_codes)
    qzeros = torch.full(
        (num_groups, size_n // _INT4_VALUES_PER_INT32),
        _GPTQ_SYMMETRIC_QZERO_WORD,
        dtype=torch.int32,
        device=weight_out_in.device,
    )
    return qweight, scales.contiguous(), qzeros.contiguous()


def quantize_mtp_expert_tensors(
    source_tensors: dict[str, torch.Tensor],
    *,
    group_size: int,
) -> tuple[dict[str, torch.Tensor], int]:
    """Replace routed-expert BF16 weights with GPTQ W4 tensor triplets."""

    derived_tensors: dict[str, torch.Tensor] = {}
    quantized_weights = 0
    for tensor_name, tensor in source_tensors.items():
        match = _MTP_EXPERT_WEIGHT_PATTERN.fullmatch(tensor_name)
        if match is None:
            derived_tensors[tensor_name] = tensor
            continue

        tensor_prefix = match.group(1)
        qweight, scales, qzeros = quantize_symmetric_int4_gptq(
            tensor,
            group_size=group_size,
        )
        derived_tensors[f"{tensor_prefix}.qweight"] = qweight
        derived_tensors[f"{tensor_prefix}.scales"] = scales
        derived_tensors[f"{tensor_prefix}.qzeros"] = qzeros
        quantized_weights += 1

    return derived_tensors, quantized_weights


def _tensor_bytes(tensors: dict[str, torch.Tensor]) -> int:
    return sum(tensor.numel() * tensor.element_size() for tensor in tensors.values())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _derive_weight_map(
    source_weight_map: dict[str, str],
) -> tuple[dict[str, str], int]:
    derived_weight_map: dict[str, str] = {}
    replaced_weights = 0
    for tensor_name, shard_name in source_weight_map.items():
        match = _MTP_EXPERT_WEIGHT_PATTERN.fullmatch(tensor_name)
        if match is None:
            derived_weight_map[tensor_name] = shard_name
            continue
        if shard_name != _MTP_SIDECAR_FILENAME:
            raise ValueError(
                f"MTP expert tensor {tensor_name} is outside {_MTP_SIDECAR_FILENAME}"
            )
        tensor_prefix = match.group(1)
        for suffix in ("qweight", "scales", "qzeros"):
            derived_weight_map[f"{tensor_prefix}.{suffix}"] = shard_name
        replaced_weights += 1
    return derived_weight_map, replaced_weights


def derive_int4_mtp_model_view(
    source_view: Path,
    output_view: Path,
    *,
    group_size: int = 128,
    runtime_layer_index: int = 48,
    expected_expert_weights: int = _PRODUCTION_MTP_EXPERT_WEIGHT_COUNT,
) -> ConversionSummary:
    """Atomically publish a derived view with only MTP routed experts in W4."""

    source_view = source_view.resolve(strict=True)
    output_view = output_view.absolute()
    if output_view.exists():
        raise FileExistsError(f"Output model view already exists: {output_view}")

    source_config_path = source_view / _CONFIG_FILENAME
    source_index_path = source_view / _INDEX_FILENAME
    source_sidecar_path = source_view / _MTP_SIDECAR_FILENAME
    for required_path in (
        source_config_path,
        source_index_path,
        source_sidecar_path,
    ):
        if not required_path.is_file():
            raise FileNotFoundError(
                f"Required source artifact is missing: {required_path}"
            )

    source_config = json.loads(source_config_path.read_text())
    derived_config, removed_exclusions = derive_int4_mtp_config(
        source_config,
        group_size=group_size,
        runtime_layer_index=runtime_layer_index,
    )
    if removed_exclusions != expected_expert_weights:
        raise ValueError(
            "MTP expert exclusion count mismatch: "
            f"expected {expected_expert_weights}, found {removed_exclusions}"
        )

    source_index = json.loads(source_index_path.read_text())
    derived_weight_map, indexed_expert_weights = _derive_weight_map(
        source_index["weight_map"]
    )
    if indexed_expert_weights != expected_expert_weights:
        raise ValueError(
            "Indexed MTP expert weight count mismatch: "
            f"expected {expected_expert_weights}, found {indexed_expert_weights}"
        )

    source_tensors = load_file(str(source_sidecar_path), device="cpu")
    source_sidecar_bytes = _tensor_bytes(source_tensors)
    derived_tensors, quantized_weights = quantize_mtp_expert_tensors(
        source_tensors,
        group_size=group_size,
    )
    if quantized_weights != expected_expert_weights:
        raise ValueError(
            "Sidecar MTP expert weight count mismatch: "
            f"expected {expected_expert_weights}, found {quantized_weights}"
        )
    if set(derived_weight_map).intersection(derived_tensors) != set(derived_tensors):
        missing = sorted(set(derived_tensors) - set(derived_weight_map))
        raise ValueError(
            f"Derived tensors are absent from the weight index: {missing[:3]}"
        )

    derived_sidecar_bytes = _tensor_bytes(derived_tensors)
    derived_index = copy.deepcopy(source_index)
    derived_index["weight_map"] = derived_weight_map
    derived_index["metadata"]["total_size"] = (
        int(source_index["metadata"]["total_size"])
        - source_sidecar_bytes
        + derived_sidecar_bytes
    )

    staging_view = output_view.with_name(f".{output_view.name}.building-{os.getpid()}")
    if staging_view.exists():
        raise FileExistsError(f"Staging model view already exists: {staging_view}")
    staging_view.mkdir(parents=True)
    try:
        reserved_names = {
            _CONFIG_FILENAME,
            _INDEX_FILENAME,
            _MTP_SIDECAR_FILENAME,
            *_PROVENANCE_FILENAMES,
        }
        symlink_lines = []
        for source_entry in sorted(source_view.iterdir()):
            if source_entry.name in reserved_names:
                continue
            target = source_entry.resolve(strict=True)
            (staging_view / source_entry.name).symlink_to(target)
            symlink_lines.append(f"{source_entry.name}\t{target}")

        derived_config_path = staging_view / _CONFIG_FILENAME
        derived_index_path = staging_view / _INDEX_FILENAME
        derived_sidecar_path = staging_view / _MTP_SIDECAR_FILENAME
        derived_config_path.write_text(json.dumps(derived_config, indent=2) + "\n")
        derived_index_path.write_text(json.dumps(derived_index, indent=2) + "\n")
        with safe_open(
            str(source_sidecar_path), framework="pt", device="cpu"
        ) as source_handle:
            sidecar_metadata = source_handle.metadata()
        save_file(
            derived_tensors,
            str(derived_sidecar_path),
            metadata=sidecar_metadata,
        )

        source_sidecar_sha256 = _sha256(source_sidecar_path)
        derived_sidecar_sha256 = _sha256(derived_sidecar_path)
        derivation_path = staging_view / "DERIVATION.txt"
        derivation_path.write_text(
            "purpose=qwen38-mtp-routed-experts-int4\n"
            f"source_view={source_view}\n"
            f"source_sidecar_sha256={source_sidecar_sha256}\n"
            f"derived_sidecar_sha256={derived_sidecar_sha256}\n"
            f"quantized_mtp_expert_weights={quantized_weights}\n"
            f"removed_mtp_expert_exclusions={removed_exclusions}\n"
            f"runtime_mtp_expert_prefix=mtp.layers.{runtime_layer_index}.mlp.experts\n"
            f"quantization=rtn_w4a16_sym_group{group_size}_gptq_layout\n"
            f"source_sidecar_bytes={source_sidecar_bytes}\n"
            f"derived_sidecar_bytes={derived_sidecar_bytes}\n"
            f"bytes_saved={source_sidecar_bytes - derived_sidecar_bytes}\n"
        )
        (staging_view / "SYMLINKS.tsv").write_text(
            "link\ttarget\n" + "\n".join(symlink_lines) + "\n"
        )
        hash_paths = (
            derived_config_path,
            derivation_path,
            derived_index_path,
            derived_sidecar_path,
        )
        (staging_view / "SHA256SUMS").write_text(
            "".join(f"{_sha256(path)}  {path.name}\n" for path in hash_paths)
        )

        for artifact in staging_view.iterdir():
            if not artifact.is_symlink():
                artifact.chmod(0o444)
        staging_view.chmod(0o555)
        staging_view.rename(output_view)
    except BaseException:
        shutil.rmtree(staging_view, ignore_errors=True)
        raise

    return ConversionSummary(
        source_view=str(source_view),
        output_view=str(output_view),
        quantized_weights=quantized_weights,
        removed_exclusions=removed_exclusions,
        source_sidecar_bytes=source_sidecar_bytes,
        derived_sidecar_bytes=derived_sidecar_bytes,
        bytes_saved=source_sidecar_bytes - derived_sidecar_bytes,
        source_sidecar_sha256=source_sidecar_sha256,
        derived_sidecar_sha256=derived_sidecar_sha256,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-view", type=Path, required=True)
    parser.add_argument("--output-view", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = derive_int4_mtp_model_view(args.source_view, args.output_view)
    print(json.dumps(asdict(summary), indent=2))


if __name__ == "__main__":
    main()
