# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
import torch
import torch.nn as nn
from hypothesis import example, given
from hypothesis import strategies as st
from safetensors.torch import save_file

import vllm.models.qwen4_exp.common.ple_sidecar as ple_sidecar_module
import vllm.models.qwen4_exp.nvidia.ple_layer as ple_layer_module
from vllm.models.qwen4_exp.common.ple_sidecar import (
    NvFp4PleSidecar,
    plan_nvfp4_ple_sidecar_gather,
)
from vllm.models.qwen4_exp.nvidia.ple_layer import Qwen4ExpNGramEmbedding
from vllm.transformers_utils.configs.qwen4_exp import Qwen4ExpTextConfig


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


def _test_ngram_sidecar_config() -> Qwen4ExpTextConfig:
    return cast(
        Qwen4ExpTextConfig,
        SimpleNamespace(
            ngram_size=3,
            heads_per_ngram=1,
            eos_token_id=2,
            vocab_size=32,
            split_ngram_parts=2,
            ngram_vocab_size_base=4,
            make_ngram_vocab_size_divisible_by=1,
            seed=1234,
        ),
    )


@st.composite
def _decode_token_batches(draw):
    num_reqs = draw(st.integers(min_value=1, max_value=8))
    input_ids = draw(
        st.lists(
            st.integers(min_value=0, max_value=31), min_size=num_reqs, max_size=num_reqs
        )
    )
    ngram_context = draw(
        st.lists(
            st.lists(st.integers(min_value=0, max_value=31), min_size=2, max_size=2),
            min_size=num_reqs,
            max_size=num_reqs,
        )
    )
    return input_ids, ngram_context


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


@given(_decode_token_batches())
@example(([5, 6], [[3, 4], [9, 2]]))
def test_decode_ngram_ids_match_generic_eos_reference(case) -> None:
    input_id_values, context_values = case
    input_ids = torch.tensor(input_id_values, dtype=torch.int64)
    ngram_context = torch.tensor(context_values, dtype=torch.int64)
    embedding = Qwen4ExpNGramEmbedding.__new__(Qwen4ExpNGramEmbedding)
    nn.Module.__init__(embedding)
    embedding.ngram_size = 3
    embedding.heads_per_ngram = 8
    embedding.eos_token_id = 2
    multipliers = torch.tensor([1_000_003, 2_000_003, 3_000_017], dtype=torch.int64)
    vocab_sizes = torch.tensor([20_000_003 + 2 * index for index in range(16)])
    offsets = torch.arange(16, dtype=torch.int64) * 20_000_100
    embedding.register_buffer("layer_multipliers", multipliers)
    embedding.register_buffer("ngram_heads_vocab_sizes", vocab_sizes)
    embedding.register_buffer("ngram_heads_offsets", offsets)

    observed = embedding._compute_decode_ngram_ids(input_ids, ngram_context)

    context = torch.cat((ngram_context, input_ids[:, None]), dim=1)
    positions, positions_in_segment = embedding._shift_precompute(
        context, embedding.eos_token_id
    )
    shifted = [context]
    for shift in range(1, embedding.ngram_size):
        shifted.append(
            embedding._shift_apply(
                context,
                positions,
                positions_in_segment,
                shift,
                embedding.eos_token_id,
            )
        )
    expected_blocks = []
    for ngram in range(2, embedding.ngram_size + 1):
        start = (ngram - 2) * embedding.heads_per_ngram
        end = start + embedding.heads_per_ngram
        mixed = shifted[0] * multipliers[0]
        for index in range(1, ngram):
            mixed = torch.bitwise_xor(mixed, shifted[index] * multipliers[index])
        expected_blocks.append(
            torch.remainder(mixed[:, -1, None], vocab_sizes[start:end])
            + offsets[start:end]
        )
    expected = torch.cat(expected_blocks, dim=1)
    assert torch.equal(observed, expected)


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


def test_nvfp4_ple_sidecar_python_fallback_gathers_dequantized_rows(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(
        ple_sidecar_module,
        "_get_native_nvfp4_ple_gather",
        lambda: None,
    )
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


def test_nvfp4_ple_sidecar_uses_native_gather_when_available(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
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
    calls = []

    def native_gather(
        code_shards: tuple[torch.Tensor, ...],
        scale_shards: tuple[torch.Tensor, ...],
        outer_scales: torch.Tensor,
        row_ids: torch.Tensor,
        destination: torch.Tensor,
        rows_per_shard: int,
    ) -> None:
        calls.append(
            (
                code_shards,
                scale_shards,
                outer_scales,
                row_ids,
                destination,
                rows_per_shard,
            )
        )
        destination.fill_(17)

    monkeypatch.setattr(
        ple_sidecar_module,
        "_get_native_nvfp4_ple_gather",
        lambda: native_gather,
        raising=False,
    )

    sidecar.gather_dequantized_rows(ids, output)

    assert len(calls) == 1
    assert calls[0][0] is sidecar.code_shards
    assert calls[0][1] is sidecar.scale_shards
    torch.testing.assert_close(calls[0][2], torch.tensor([0.25, 0.5]))
    assert calls[0][3] is ids
    assert calls[0][4] is output
    assert calls[0][5] == 4
    assert torch.equal(output, torch.full_like(output, 17))


@pytest.mark.parametrize("output_dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_native_nvfp4_ple_gather_matches_python_fallback(
    output_dtype: torch.dtype,
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    native_gather = ple_sidecar_module._get_native_nvfp4_ple_gather()
    if native_gather is None:
        pytest.skip("native NVFP4 PLE gather extension is unavailable")
    sidecar_dir = tmp_path / "ple"
    manifest_sha256 = _write_nvfp4_ple_sidecar(sidecar_dir)
    sidecar = NvFp4PleSidecar.open(
        sidecar_dir,
        expected_rows=8,
        expected_width=16,
        expected_manifest_sha256=manifest_sha256,
    )
    ids = torch.tensor([7, 0, 4, 3, 4])
    expected = torch.empty((ids.numel(), 16), dtype=output_dtype)
    actual = torch.empty_like(expected)
    monkeypatch.setattr(
        ple_sidecar_module,
        "_get_native_nvfp4_ple_gather",
        lambda: None,
    )
    sidecar.gather_dequantized_rows(ids, expected)

    native_gather(
        sidecar.code_shards,
        sidecar.scale_shards,
        sidecar._outer_scales_tensor,
        ids,
        actual,
        sidecar.manifest.rows_per_shard,
    )

    assert torch.equal(actual, expected)


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


def test_ngram_sidecar_decode_uses_exact_direct_row_ids(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sidecar_dir = tmp_path / "ple"
    manifest_sha256 = _write_nvfp4_ple_sidecar(sidecar_dir)
    config = _test_ngram_sidecar_config()
    monkeypatch.setattr(ple_layer_module, "is_offload_process", lambda: True)
    monkeypatch.setattr(ple_layer_module, "_nth_prime_after", lambda *_: 4)
    embedding = Qwen4ExpNGramEmbedding(
        config,
        embedding_dim=32,
        ple_dense_layer_id=0,
        max_total_tokens=8,
        max_num_reqs=2,
        prefix="ple",
        nvfp4_sidecar_dir=str(sidecar_dir),
        nvfp4_sidecar_manifest_sha256=manifest_sha256,
        params_dtype=torch.bfloat16,
    )
    captured_row_ids = []

    def native_gather(
        _code_shards: tuple[torch.Tensor, ...],
        _scale_shards: tuple[torch.Tensor, ...],
        _outer_scales: torch.Tensor,
        row_ids: torch.Tensor,
        output: torch.Tensor,
        _rows_per_shard: int,
    ) -> None:
        captured_row_ids.append(row_ids.clone())
        output.zero_()

    monkeypatch.setattr(
        ple_sidecar_module,
        "_get_native_nvfp4_ple_gather",
        lambda: native_gather,
    )
    monkeypatch.setattr(
        embedding,
        "_shift_precompute",
        lambda *_: pytest.fail("decode used the generic PLE pack-and-shift path"),
    )
    input_ids = torch.tensor([5, 6])
    ngram_context = torch.tensor([[3, 4], [9, 2]])

    output = embedding.forward_impl(
        torch.empty(2, 32),
        input_ids,
        torch.tensor([0, 1, 2]),
        ngram_context,
        output_buffer=torch.empty(8, 32, dtype=torch.bfloat16),
    )

    multipliers = cast(torch.Tensor, embedding.layer_multipliers)
    vocab_sizes = cast(torch.Tensor, embedding.ngram_heads_vocab_sizes)
    offsets = cast(torch.Tensor, embedding.ngram_heads_offsets)
    multiplier_0, multiplier_1, multiplier_2 = multipliers
    previous = torch.tensor([4, 2])
    previous_previous = torch.tensor([3, 2])
    bigram = input_ids * multiplier_0
    bigram = torch.bitwise_xor(bigram, previous * multiplier_1)
    trigram = torch.bitwise_xor(bigram, previous_previous * multiplier_2)
    expected = torch.stack(
        (
            torch.remainder(bigram, vocab_sizes[0]) + offsets[0],
            torch.remainder(trigram, vocab_sizes[1]) + offsets[1],
        ),
        dim=1,
    )
    assert len(captured_row_ids) == 1
    assert torch.equal(captured_row_ids[0].reshape(2, 2), expected)
    assert output.shape == (2, 32)


def test_ngram_sidecar_avoids_resident_embedding_allocation(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    sidecar_dir = tmp_path / "ple"
    manifest_sha256 = _write_nvfp4_ple_sidecar(sidecar_dir)
    config = _test_ngram_sidecar_config()
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
        nvfp4_sidecar_dir=str(sidecar_dir),
        nvfp4_sidecar_manifest_sha256=manifest_sha256,
        params_dtype=torch.bfloat16,
    )

    assert not list(embedding.named_parameters())
    assert embedding.get_offload_output_dtype(torch.bfloat16) == torch.bfloat16
    assert embedding.get_offload_output_dim(32) == 32
