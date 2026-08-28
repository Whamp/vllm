# SPDX-License-Identifier: MIT

import hashlib
import json
import struct

import pytest

torch = pytest.importorskip("torch")

import compare_m6_layer_oracle as comparison


def _write_synthetic_dumps(tmp_path):
    comparison.LAYER_COUNT = 2
    comparison.LAYER_SHAPE = (2, 2)
    comparison.LOGIT_COUNT = 5
    comparison.LOGIT_TOP10_OVERLAP_MIN = 5
    vllm_dir = tmp_path / "vllm"
    llama_dir = tmp_path / "llama"
    vllm_dir.mkdir()
    llama_dir.mkdir()
    token_path = tmp_path / "tokens.txt"
    token_path.write_text("1\n2\n3\n")

    entries = []
    for layer in range(comparison.LAYER_COUNT):
        tensor = torch.ones(comparison.LAYER_SHAPE, dtype=torch.float32)
        path = vllm_dir / f"layer-{layer:03d}.pt"
        torch.save(tensor, path)
        payload = path.read_bytes()
        entries.append(
            {
                "layer": layer,
                "path": path.name,
                "shape": list(comparison.LAYER_SHAPE),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        )
        (llama_dir / f"layer-{layer:03d}.f32").write_bytes(
            struct.pack("<4f", 1.0, 1.0, 1.0, 1.0)
        )

    logits = torch.arange(comparison.LOGIT_COUNT, dtype=torch.float32)
    logits_path = vllm_dir / "logits.pt"
    torch.save(logits, logits_path)
    logits_payload = logits_path.read_bytes()
    (vllm_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "gguf-dsv4-layer-oracle-v1",
                "token_ids": [1, 2, 3],
                "token_ids_sha256": hashlib.sha256(token_path.read_bytes()).hexdigest(),
                "layers": entries,
                "logits": {
                    "path": logits_path.name,
                    "shape": [comparison.LOGIT_COUNT],
                    "size": len(logits_payload),
                    "sha256": hashlib.sha256(logits_payload).hexdigest(),
                },
            }
        )
        + "\n"
    )
    (llama_dir / "logits.f32").write_bytes(struct.pack("<5f", 0.0, 1.0, 2.0, 3.0, 4.0))
    (llama_dir / "manifest.json").write_text(
        json.dumps(
            {
                "format": "llama-ds4-layer-oracle-v1",
                "token_count": 3,
                "layer_count": comparison.LAYER_COUNT,
                "layer_value_count": 4,
                "logit_count": comparison.LOGIT_COUNT,
            }
        )
        + "\n"
    )
    return vllm_dir, llama_dir, token_path


def test_identical_layer_oracle_dumps_pass(tmp_path):
    vllm_dir, llama_dir, token_path = _write_synthetic_dumps(tmp_path)

    report = comparison.compare_m6_layer_oracle(vllm_dir, llama_dir, token_path)

    assert report["passed"]
    assert report["layer_summary"]["median_normalized_rmse"] == 0
    assert report["logits"]["top10_overlap"] == comparison.LOGIT_COUNT


def test_checksum_valid_numerical_drift_fails(tmp_path):
    vllm_dir, llama_dir, token_path = _write_synthetic_dumps(tmp_path)
    layer_path = vllm_dir / "layer-000.pt"
    torch.save(torch.full(comparison.LAYER_SHAPE, 2.0), layer_path)
    manifest_path = vllm_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    payload = layer_path.read_bytes()
    manifest["layers"][0]["size"] = len(payload)
    manifest["layers"][0]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest_path.write_text(json.dumps(manifest) + "\n")

    report = comparison.compare_m6_layer_oracle(vllm_dir, llama_dir, token_path)

    assert not report["passed"]
    assert report["layers"][0]["normalized_rmse"] == 1.0


def test_corrupt_vllm_payload_is_rejected(tmp_path):
    vllm_dir, llama_dir, token_path = _write_synthetic_dumps(tmp_path)
    with (vllm_dir / "layer-000.pt").open("ab") as output:
        output.write(b"corrupt")

    with pytest.raises(ValueError, match="size mismatch"):
        comparison.compare_m6_layer_oracle(vllm_dir, llama_dir, token_path)
