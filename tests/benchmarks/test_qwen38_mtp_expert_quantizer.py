# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file, save_file

from benchmarks.qwen38_ple_runtime.quantize_qwen38_mtp_experts import (
    derive_int4_mtp_config,
    derive_int4_mtp_model_view,
    pack_gptq_int4_rows,
    quantize_mtp_expert_tensors,
    quantize_symmetric_int4_gptq,
    unpack_gptq_int4_rows,
)


@pytest.mark.parametrize("size_k", [8, 16, 32])
@pytest.mark.parametrize("size_n", [1, 7, 16, 33])
def test_gptq_int4_row_packing_round_trips_all_nibbles(
    size_k: int,
    size_n: int,
) -> None:
    """GPTQ row packing preserves every uint4 code exactly."""

    generator = torch.Generator().manual_seed(size_k * 1000 + size_n)
    random_values = torch.randint(
        0,
        16,
        (size_k, size_n),
        dtype=torch.int32,
        generator=generator,
    )
    all_nibbles = (
        torch.arange(size_k * size_n, dtype=torch.int32).reshape(size_k, size_n) % 16
    )

    for unpacked in (random_values, all_nibbles):
        packed = pack_gptq_int4_rows(unpacked)

        assert packed.dtype == torch.int32
        assert packed.shape == (size_k // 8, size_n)
        assert torch.equal(unpack_gptq_int4_rows(packed, size_k=size_k), unpacked)


def test_symmetric_int4_quantization_matches_worked_gptq_literal() -> None:
    """A known K-axis code sequence packs to the documented GPTQ word."""

    weight_out_in = torch.arange(-7, 1, dtype=torch.float32).repeat(8, 1)

    qweight, scales, qzeros = quantize_symmetric_int4_gptq(
        weight_out_in,
        group_size=8,
    )

    signed_0x87654321 = 0x87654321 - (1 << 32)
    assert torch.equal(
        qweight,
        torch.full((1, 8), signed_0x87654321, dtype=torch.int32),
    )
    assert torch.equal(scales, torch.ones((1, 8), dtype=torch.float16))
    assert torch.equal(
        qzeros,
        torch.full((1, 1), 0x77777777, dtype=torch.int32),
    )


@pytest.mark.parametrize("size_k", [128, 256])
@pytest.mark.parametrize("size_n", [8, 16])
def test_symmetric_int4_quantization_reconstructs_within_half_scale(
    size_k: int,
    size_n: int,
) -> None:
    """Stored codes and FP16 scales bound round-to-nearest reconstruction error."""

    generator = torch.Generator().manual_seed(size_k * 1000 + size_n)
    weight_out_in = torch.randn(
        (size_n, size_k),
        dtype=torch.float32,
        generator=generator,
    )

    qweight, scales, _ = quantize_symmetric_int4_gptq(
        weight_out_in,
        group_size=128,
    )
    codes = unpack_gptq_int4_rows(qweight, size_k=size_k)
    grouped_codes = codes.reshape(size_k // 128, 128, size_n)
    reconstructed = (
        (grouped_codes - 8).to(torch.float32) * scales.float().unsqueeze(1)
    ).reshape(size_k, size_n)
    error = (reconstructed - weight_out_in.t()).abs()
    bound = (
        scales.float().unsqueeze(1).expand_as(grouped_codes).reshape(size_k, size_n) / 2
    )

    assert torch.all(error <= bound + 1e-6)


def test_derive_int4_mtp_config_quantizes_only_routed_experts() -> None:
    """The derived config leaves every non-routed MTP override unchanged."""

    source_config = {
        "model_type": "qwen4_exp",
        "quantization_config": {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "data_type": "int",
            "packing_format": "auto_round:auto_gptq",
            "extra_config": {
                ".*mtp.*": {"bits": 16, "data_type": "float"},
                "mtp.fc_embedding": {"bits": 16, "data_type": "fp"},
                "mtp.layers.0.mlp.experts.0.gate_proj": {
                    "bits": 16,
                    "data_type": "fp",
                },
                "mtp.layers.0.mlp.experts.0.up_proj": {
                    "bits": 16,
                    "data_type": "fp",
                },
                "mtp.layers.0.mlp.experts.0.down_proj": {
                    "bits": 16,
                    "data_type": "fp",
                },
                "model.language_model.layers.0.self_attn.q_proj": {
                    "bits": 16,
                    "data_type": "fp",
                },
            },
        },
    }

    derived_config, removed_exclusions = derive_int4_mtp_config(source_config)

    assert removed_exclusions == 3
    assert source_config["quantization_config"]["extra_config"].keys() == {
        ".*mtp.*",
        "mtp.fc_embedding",
        "mtp.layers.0.mlp.experts.0.gate_proj",
        "mtp.layers.0.mlp.experts.0.up_proj",
        "mtp.layers.0.mlp.experts.0.down_proj",
        "model.language_model.layers.0.self_attn.q_proj",
    }
    assert derived_config["quantization_config"]["extra_config"] == {
        ".*mtp.*": {"bits": 16, "data_type": "float"},
        "mtp.fc_embedding": {"bits": 16, "data_type": "fp"},
        "model.language_model.layers.0.self_attn.q_proj": {
            "bits": 16,
            "data_type": "fp",
        },
        "mtp.layers.48.mlp.experts": {
            "bits": 4,
            "group_size": 128,
            "sym": True,
            "data_type": "int",
        },
    }


def test_quantize_mtp_expert_tensors_preserves_non_expert_tensors() -> None:
    """Only routed expert weights change representation in the sidecar."""

    generator = torch.Generator().manual_seed(20260901)
    gate_weight = torch.randn((8, 16), generator=generator, dtype=torch.float32)
    down_weight = torch.randn((16, 8), generator=generator, dtype=torch.float32)
    attention_weight = torch.randn((4, 4), generator=generator).to(torch.bfloat16)
    source_tensors = {
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": gate_weight,
        "mtp.layers.0.mlp.experts.0.down_proj.weight": down_weight,
        "mtp.layers.0.self_attn.q_proj.weight": attention_weight,
    }

    derived_tensors, quantized_weights = quantize_mtp_expert_tensors(
        source_tensors,
        group_size=8,
    )

    assert quantized_weights == 2
    assert set(derived_tensors) == {
        "mtp.layers.0.mlp.experts.0.gate_proj.qweight",
        "mtp.layers.0.mlp.experts.0.gate_proj.scales",
        "mtp.layers.0.mlp.experts.0.gate_proj.qzeros",
        "mtp.layers.0.mlp.experts.0.down_proj.qweight",
        "mtp.layers.0.mlp.experts.0.down_proj.scales",
        "mtp.layers.0.mlp.experts.0.down_proj.qzeros",
        "mtp.layers.0.self_attn.q_proj.weight",
    }
    assert torch.equal(
        derived_tensors["mtp.layers.0.self_attn.q_proj.weight"],
        attention_weight,
    )

    for projection, source_weight in (
        ("gate_proj", gate_weight),
        ("down_proj", down_weight),
    ):
        prefix = f"mtp.layers.0.mlp.experts.0.{projection}"
        qweight = derived_tensors[f"{prefix}.qweight"]
        scales = derived_tensors[f"{prefix}.scales"]
        codes = unpack_gptq_int4_rows(qweight, size_k=source_weight.shape[1])
        grouped_codes = codes.reshape(
            source_weight.shape[1] // 8,
            8,
            source_weight.shape[0],
        )
        reconstructed = (
            (grouped_codes - 8).float() * scales.float().unsqueeze(1)
        ).reshape(source_weight.shape[1], source_weight.shape[0])
        error = (reconstructed - source_weight.t()).abs()
        bound = (
            scales.float()
            .unsqueeze(1)
            .expand_as(grouped_codes)
            .reshape_as(reconstructed)
            / 2
        )
        assert torch.all(error <= bound + 1e-6)


def test_derive_int4_mtp_model_view_writes_coherent_checkpoint(
    tmp_path: Path,
) -> None:
    """The derived view has matching tensors, index, config, and provenance."""

    source_view = tmp_path / "source-view"
    source_view.mkdir()
    output_view = tmp_path / "output-view"
    source_sidecar = source_view / "model_extra_tensors.safetensors"
    generator = torch.Generator().manual_seed(42)
    source_tensors = {
        "mtp.layers.0.mlp.experts.0.gate_proj.weight": torch.randn(
            (8, 16), generator=generator
        ),
        "mtp.layers.0.mlp.experts.0.down_proj.weight": torch.randn(
            (16, 8), generator=generator
        ),
        "mtp.layers.0.self_attn.q_proj.weight": torch.randn(
            (4, 4), generator=generator
        ).to(torch.bfloat16),
    }
    save_file(source_tensors, source_sidecar)
    source_config = {
        "model_type": "qwen4_exp",
        "quantization_config": {
            "bits": 4,
            "group_size": 8,
            "sym": True,
            "data_type": "int",
            "packing_format": "auto_round:auto_gptq",
            "extra_config": {
                ".*mtp.*": {"bits": 16, "data_type": "float"},
                "mtp.layers.0.mlp.experts.0.gate_proj": {
                    "bits": 16,
                    "data_type": "fp",
                },
                "mtp.layers.0.mlp.experts.0.down_proj": {
                    "bits": 16,
                    "data_type": "fp",
                },
            },
        },
    }
    (source_view / "config.json").write_text(json.dumps(source_config))
    source_tensor_bytes = sum(
        tensor.numel() * tensor.element_size() for tensor in source_tensors.values()
    )
    source_index = {
        "metadata": {
            "format": "safetensors",
            "total_parameters": sum(
                tensor.numel() for tensor in source_tensors.values()
            ),
            "total_shards": 1,
            "total_size": source_tensor_bytes,
        },
        "weight_map": {name: source_sidecar.name for name in source_tensors},
    }
    (source_view / "model.safetensors.index.json").write_text(json.dumps(source_index))
    (source_view / "tokenizer.json").write_text("synthetic tokenizer")

    summary = derive_int4_mtp_model_view(
        source_view,
        output_view,
        group_size=8,
        runtime_layer_index=48,
        expected_expert_weights=2,
    )

    assert summary.quantized_weights == 2
    assert summary.removed_exclusions == 2
    derived_tensors = load_file(output_view / source_sidecar.name)
    assert len(derived_tensors) == 7
    assert "mtp.layers.0.mlp.experts.0.gate_proj.weight" not in derived_tensors
    assert "mtp.layers.0.mlp.experts.0.gate_proj.qweight" in derived_tensors
    assert torch.equal(
        derived_tensors["mtp.layers.0.self_attn.q_proj.weight"],
        source_tensors["mtp.layers.0.self_attn.q_proj.weight"],
    )

    derived_index = json.loads(
        (output_view / "model.safetensors.index.json").read_text()
    )
    assert set(derived_index["weight_map"]) == set(derived_tensors)
    assert derived_index["metadata"]["total_size"] == sum(
        tensor.numel() * tensor.element_size() for tensor in derived_tensors.values()
    )
    derived_config = json.loads((output_view / "config.json").read_text())
    assert (
        derived_config["quantization_config"]["extra_config"][
            "mtp.layers.48.mlp.experts"
        ]["bits"]
        == 4
    )
    assert (output_view / "tokenizer.json").is_symlink()
    assert (output_view / "tokenizer.json").read_text() == "synthetic tokenizer"
    assert (
        "quantized_mtp_expert_weights=2" in (output_view / "DERIVATION.txt").read_text()
    )
    assert "model_extra_tensors.safetensors" in (output_view / "SHA256SUMS").read_text()
