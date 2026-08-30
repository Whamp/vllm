# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st
from safetensors.torch import save_file

import vllm.models.qwen4_exp.nvidia.ple_layer as ple_layer_module
from vllm.models.qwen4_exp.common.ple_sidecar import (
    NvFp4PleSidecar,
    plan_nvfp4_ple_sidecar_gather,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


def _write_nvfp4_ple_sidecar(
    directory: Path,
    *,
    shards: int = 2,
    rows_per_shard: int = 4,
    width: int = 16,
    outer_scales: tuple[float, ...] | None = None,
) -> str:
    directory.mkdir()
    manifest = {
        "layout": "group16_e2m1_e4m3scale_lownibblefirst",
        "shards": shards,
        "rows": shards * rows_per_shard,
        "width": width,
        "worst_shard_rel_err": 0.125,
    }
    manifest_bytes = json.dumps(manifest, indent=1).encode()
    (directory / "META.json").write_bytes(manifest_bytes)
    if outer_scales is None:
        outer_scales = tuple(0.25 * (shard + 1) for shard in range(shards))
    if len(outer_scales) != shards:
        raise ValueError("outer_scales must contain one value per shard")
    for shard in range(shards):
        row_start = shard * rows_per_shard
        codes = torch.arange(
            row_start * (width // 2),
            (row_start + rows_per_shard) * (width // 2),
            dtype=torch.uint8,
        ).reshape(rows_per_shard, width // 2)
        scales = (
            torch.arange(
                row_start + 1,
                row_start + rows_per_shard + 1,
                dtype=torch.float32,
            )
            .reshape(rows_per_shard, 1)
            .to(torch.float8_e4m3fn)
        )
        save_file(
            {
                "weight_e2m1": codes,
                "weight_scale": scales,
                "weight_scale_2": torch.tensor(outer_scales[shard]),
            },
            directory / f"shard_{shard}.safetensors",
        )
    return hashlib.sha256(manifest_bytes).hexdigest()


@st.composite
def _sidecar_geometry_and_ids(draw):
    shard_count = draw(st.integers(min_value=1, max_value=8))
    rows_per_shard = draw(st.integers(min_value=1, max_value=32))
    total_rows = shard_count * rows_per_shard
    ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=total_rows - 1),
            min_size=0,
            max_size=80,
        )
    )
    return shard_count, rows_per_shard, ids


@given(_sidecar_geometry_and_ids())
def test_nvfp4_ple_sidecar_gather_plan_preserves_global_rows(case) -> None:
    shard_count, rows_per_shard, ids = case
    plan = plan_nvfp4_ple_sidecar_gather(
        torch.tensor(ids, dtype=torch.int64),
        shard_count=shard_count,
        rows_per_shard=rows_per_shard,
    )

    observed: list[tuple[int, int] | None] = [None] * len(ids)
    for partition in plan:
        for position, local_row in zip(
            partition.output_positions.tolist(),
            partition.local_rows.tolist(),
            strict=True,
        ):
            assert observed[position] is None
            observed[position] = (partition.shard_index, local_row)

    expected = [divmod(row, rows_per_shard) for row in ids]
    assert observed == expected


def test_nvfp4_ple_sidecar_gather_plan_rejects_invalid_rows() -> None:
    with pytest.raises(ValueError, match="outside the sidecar row range"):
        plan_nvfp4_ple_sidecar_gather(
            torch.tensor([-1, 8]),
            shard_count=2,
            rows_per_shard=4,
        )


def test_nvfp4_ple_sidecar_boundary_kills_wrong_shard_width() -> None:
    ids = torch.tensor([3, 4, 7])
    correct = plan_nvfp4_ple_sidecar_gather(
        ids,
        shard_count=2,
        rows_per_shard=4,
    )
    counterfeit = plan_nvfp4_ple_sidecar_gather(
        ids,
        shard_count=2,
        rows_per_shard=5,
    )

    correct_rows = [
        (part.shard_index, row) for part in correct for row in part.local_rows.tolist()
    ]
    counterfeit_rows = [
        (part.shard_index, row)
        for part in counterfeit
        for row in part.local_rows.tolist()
    ]
    assert correct_rows != counterfeit_rows


def test_nvfp4_ple_sidecar_gathers_dequantized_rows(tmp_path: Path) -> None:
    sidecar_dir = tmp_path / "ple"
    manifest_sha256 = _write_nvfp4_ple_sidecar(sidecar_dir)
    sidecar = NvFp4PleSidecar.open(
        sidecar_dir,
        expected_rows=8,
        expected_width=16,
        expected_manifest_sha256=manifest_sha256,
    )
    ids = torch.tensor([7, 0, 4, 3, 4])
    output = torch.empty((ids.numel(), 16), dtype=torch.bfloat16)

    sidecar.gather_dequantized_rows(ids, output)

    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0], dtype=torch.float32
    )
    expected_rows = []
    for row in ids.tolist():
        shard, local_row = divmod(row, 4)
        codes = sidecar.code_shards[shard][local_row]
        nibbles = (
            torch.stack((codes & 0xF, codes >> 4), dim=-1).reshape(16).to(torch.int64)
        )
        signed = magnitudes[nibbles & 0x7] * (1 - 2 * (nibbles >> 3).float())
        scales = sidecar.scale_shards[shard][local_row].float().repeat_interleave(16)
        expected_rows.append(
            (signed * scales * sidecar.outer_scales[shard]).to(torch.bfloat16)
        )
    assert torch.equal(output, torch.stack(expected_rows))
    assert [scale.item() for scale in sidecar.outer_scales] == [0.25, 0.5]


def test_nvfp4_ple_sidecar_rejects_manifest_hash_mismatch(tmp_path: Path) -> None:
    sidecar_dir = tmp_path / "ple"
    _write_nvfp4_ple_sidecar(sidecar_dir)

    with pytest.raises(ValueError, match="manifest SHA-256 mismatch"):
        NvFp4PleSidecar.open(
            sidecar_dir,
            expected_rows=8,
            expected_width=16,
            expected_manifest_sha256="0" * 64,
        )


def test_ngram_sidecar_avoids_resident_embedding_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sidecar_dir = tmp_path / "ple"
    manifest_sha256 = _write_nvfp4_ple_sidecar(sidecar_dir)
    config = SimpleNamespace(
        ngram_size=3,
        heads_per_ngram=1,
        eos_token_id=2,
        vocab_size=32,
        split_ngram_parts=2,
        ngram_vocab_size_base=4,
        make_ngram_vocab_size_divisible_by=1,
        seed=1234,
    )
    monkeypatch.setattr(ple_layer_module, "is_offload_process", lambda: True)
    monkeypatch.setattr(ple_layer_module, "_nth_prime_after", lambda *_: 4)
    monkeypatch.setattr(
        ple_layer_module,
        "VocabParallelEmbedding",
        lambda *args, **kwargs: pytest.fail(
            "sidecar construction allocated the resident PLE embedding"
        ),
    )

    embedding = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=32,
        ple_dense_layer_id=0,
        max_total_tokens=8,
        max_num_reqs=2,
        prefix="ple",
        nvfp4_sidecar_dir=sidecar_dir,
        nvfp4_sidecar_manifest_sha256=manifest_sha256,
        params_dtype=torch.bfloat16,
    )

    assert not list(embedding.named_parameters())
    assert embedding.get_offload_output_dtype(torch.bfloat16) == torch.bfloat16
    assert embedding.get_offload_output_dim(32) == 32
