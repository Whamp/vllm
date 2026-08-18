# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import struct
from pathlib import Path

import pytest

from vllm.model_executor.model_loader.gguf_dsv4 import (
    GGUFTensorEntry,
    build_gguf_dsv4_load_plan,
    classify_gguf_dsv4_tensor,
    parse_gguf_index,
)


def _gguf_string(value: str) -> bytes:
    encoded = value.encode()
    return struct.pack("<Q", len(encoded)) + encoded


def _write_synthetic_gguf(
    path: Path,
    tensors: list[tuple[str, list[int], int, int]],
) -> None:
    header = bytearray(b"GGUF")
    header += struct.pack("<IQQ", 3, len(tensors), 1)
    header += _gguf_string("general.alignment")
    header += struct.pack("<II", 4, 32)
    for name, dims, type_id, offset in tensors:
        header += _gguf_string(name)
        header += struct.pack("<I", len(dims))
        header += b"".join(struct.pack("<Q", dim) for dim in dims)
        header += struct.pack("<IQ", type_id, offset)
    data_start = (len(header) + 31) & ~31
    payload_end = max(
        offset
        + (
            GGUFTensorEntry.compute_nbytes(type_id=type_id, dims=tuple(dims))
            if type_id in (0, 1, 8, 9, 10, 16, 24, 26, 27, 28, 30)
            else 1
        )
        for _, dims, type_id, offset in tensors
    )
    path.write_bytes(bytes(header) + bytes(data_start - len(header) + payload_end))


def test_parse_gguf_index_reads_bounded_v3_directory(tmp_path: Path) -> None:
    path = tmp_path / "tiny.gguf"
    _write_synthetic_gguf(
        path,
        [
            ("blk.0.ffn_gate_exps.weight", [256, 8, 2], 16, 0),
            ("blk.0.ffn_down_exps.weight", [256, 8, 2], 10, 1056),
        ],
    )

    index = parse_gguf_index(path)

    assert index.version == 3
    assert index.metadata["general.alignment"] == 32
    assert index.data_start % 32 == 0
    assert index.tensors[0].type_name == "IQ2_XXS"
    assert index.tensors[0].nbytes == 8 * 2 * 66
    assert index.tensors[1].type_name == "Q2_K"
    assert index.file_size == path.stat().st_size


def test_parse_gguf_index_rejects_unknown_type_and_overlap(tmp_path: Path) -> None:
    unknown = tmp_path / "unknown.gguf"
    _write_synthetic_gguf(unknown, [("bad", [32, 1], 99, 0)])
    with pytest.raises(ValueError, match="unknown GGUF tensor type 99"):
        parse_gguf_index(unknown)

    overlap = tmp_path / "overlap.gguf"
    _write_synthetic_gguf(
        overlap,
        [("a", [32, 1], 8, 0), ("b", [32, 1], 8, 0)],
    )
    with pytest.raises(ValueError, match="overlap"):
        parse_gguf_index(overlap)


def test_gguf_dsv4_plan_maps_expert_coordinates_and_fused_slots() -> None:
    entries = (
        GGUFTensorEntry("blk.0.ffn_gate_exps.weight", 16, (4096, 2048, 256), 0),
        GGUFTensorEntry("blk.0.ffn_down_exps.weight", 10, (2048, 4096, 256), 1),
        GGUFTensorEntry("blk.0.attn_q_a.weight", 8, (4096, 1024), 2),
        GGUFTensorEntry("blk.0.attn_kv.weight", 8, (4096, 512), 3),
    )

    plan = build_gguf_dsv4_load_plan(entries, tp_rank=1, tp_size=4)
    by_source = {item.source_name: item for item in plan}

    gate = by_source["blk.0.ffn_gate_exps.weight"]
    assert gate.target_name == "model.layers.0.ffn.experts.gate_raw"
    assert len(gate.spans) == 1
    assert gate.spans[0].source_offset == 512 * 1056
    assert gate.spans[0].target_offset == 0
    assert gate.spans[0].count == 256
    assert gate.spans[0].source_stride == 2048 * 1056
    assert gate.spans[0].target_stride == 512 * 1056

    down = by_source["blk.0.ffn_down_exps.weight"]
    assert down.target_name == "model.layers.0.ffn.experts.down_raw"
    assert len(down.spans) == 1
    assert down.spans[0].source_offset == 1 + 2 * 84
    assert down.spans[0].nbytes == 2 * 84
    assert down.spans[0].count == 4096 * 256
    assert down.spans[0].source_stride == 8 * 84
    assert down.spans[0].target_stride == 2 * 84

    q_a = by_source["blk.0.attn_q_a.weight"]
    kv = by_source["blk.0.attn_kv.weight"]
    assert q_a.target_name == kv.target_name
    assert q_a.target_name == "model.layers.0.attn.fused_wqa_wkv.weight_raw"
    assert q_a.spans[0].target_offset == 0
    assert kv.spans[0].target_offset == q_a.target_nbytes


def _expected_dsv4_tensor_names() -> list[str]:
    names = ["token_embd.weight", "output.weight", "output_norm.weight"]
    names += [
        "output_hc_fn.weight",
        "output_hc_base.weight",
        "output_hc_scale.weight",
    ]
    every_layer = [
        "attn_q_a.weight",
        "attn_kv.weight",
        "attn_q_b.weight",
        "attn_output_a.weight",
        "attn_output_b.weight",
        "attn_q_a_norm.weight",
        "attn_kv_a_norm.weight",
        "attn_norm.weight",
        "attn_sinks.weight",
        "ffn_gate_exps.weight",
        "ffn_up_exps.weight",
        "ffn_down_exps.weight",
        "ffn_gate_shexp.weight",
        "ffn_up_shexp.weight",
        "ffn_down_shexp.weight",
        "ffn_gate_inp.weight",
        "ffn_norm.weight",
        "hc_attn_fn.weight",
        "hc_attn_base.weight",
        "hc_attn_scale.weight",
        "hc_ffn_fn.weight",
        "hc_ffn_base.weight",
        "hc_ffn_scale.weight",
    ]
    for layer in range(43):
        names.extend(f"blk.{layer}.{suffix}" for suffix in every_layer)
    for layer in range(2, 43):
        names.extend(
            f"blk.{layer}.{suffix}"
            for suffix in (
                "attn_compressor_kv.weight",
                "attn_compressor_gate.weight",
                "attn_compressor_ape.weight",
                "attn_compressor_norm.weight",
            )
        )
    for layer in range(2, 43, 2):
        names.extend(
            f"blk.{layer}.{suffix}"
            for suffix in (
                "indexer.attn_q_b.weight",
                "indexer.proj.weight",
                "indexer_compressor_kv.weight",
                "indexer_compressor_gate.weight",
                "indexer_compressor_ape.weight",
                "indexer_compressor_norm.weight",
            )
        )
    for layer in range(3):
        names.append(f"blk.{layer}.ffn_gate_tid2eid.weight")
    for layer in range(3, 43):
        names.append(f"blk.{layer}.exp_probs_b.bias")
    return names


def test_gguf_dsv4_classifier_covers_exact_1328_tensor_inventory() -> None:
    names = _expected_dsv4_tensor_names()
    assert len(names) == 1328

    classifications = [classify_gguf_dsv4_tensor(name) for name in names]

    assert all(item.target_name for item in classifications)
    assert len({item.source_name for item in classifications}) == 1328
    with pytest.raises(ValueError, match="unsupported DeepSeek V4 GGUF tensor"):
        classify_gguf_dsv4_tensor("blk.0.unexpected.weight")
