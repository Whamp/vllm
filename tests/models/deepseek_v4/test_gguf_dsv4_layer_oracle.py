# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import hashlib
import json

import pytest
import torch

import vllm.models.deepseek_v4.nvidia.model as deepseek_v4_model
from vllm.models.deepseek_v4.gguf_dsv4_layer_oracle import (
    GGUF_DSV4_LAYER_ORACLE_DIR,
    GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE,
    build_gguf_dsv4_layer_oracle_recorder,
)


def _write_token_ids(path, token_ids=(128000, 123, 456)):
    path.write_text("\n".join(str(token_id) for token_id in token_ids) + "\n")


def test_layer_oracle_is_disabled_without_output_directory(monkeypatch):
    monkeypatch.delenv(GGUF_DSV4_LAYER_ORACLE_DIR, raising=False)
    monkeypatch.delenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE, raising=False)

    assert (
        build_gguf_dsv4_layer_oracle_recorder(
            quantization_method="gguf_dsv4",
            enforce_eager=True,
            tensor_parallel_rank=0,
            expected_layer_count=2,
        )
        is None
    )


def test_layer_oracle_requires_gguf_eager_rank_zero(tmp_path, monkeypatch):
    token_path = tmp_path / "tokens.txt"
    _write_token_ids(token_path)
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_DIR, str(tmp_path / "out"))
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE, str(token_path))

    with pytest.raises(ValueError, match="requires quantization_method=gguf_dsv4"):
        build_gguf_dsv4_layer_oracle_recorder(
            quantization_method="fp8",
            enforce_eager=True,
            tensor_parallel_rank=0,
            expected_layer_count=2,
        )
    with pytest.raises(ValueError, match="requires enforce_eager=True"):
        build_gguf_dsv4_layer_oracle_recorder(
            quantization_method="gguf_dsv4",
            enforce_eager=False,
            tensor_parallel_rank=0,
            expected_layer_count=2,
        )
    assert (
        build_gguf_dsv4_layer_oracle_recorder(
            quantization_method="gguf_dsv4",
            enforce_eager=True,
            tensor_parallel_rank=1,
            expected_layer_count=2,
        )
        is None
    )


@pytest.mark.parametrize("contents", ["", "12\nnot-an-id\n", "-1\n", "129280\n"])
def test_layer_oracle_rejects_invalid_token_files(contents, tmp_path, monkeypatch):
    token_path = tmp_path / "tokens.txt"
    token_path.write_text(contents)
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_DIR, str(tmp_path / "out"))
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE, str(token_path))

    with pytest.raises(ValueError, match="GGUF DSV4 layer oracle token IDs"):
        build_gguf_dsv4_layer_oracle_recorder(
            quantization_method="gguf_dsv4",
            enforce_eager=True,
            tensor_parallel_rank=0,
            expected_layer_count=2,
            vocab_size=129280,
        )


def test_layer_oracle_records_exact_trigger_once(tmp_path, monkeypatch):
    token_ids = (128000, 123, 456)
    token_path = tmp_path / "tokens.txt"
    _write_token_ids(token_path, token_ids)
    output_dir = tmp_path / "out"
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_DIR, str(output_dir))
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE, str(token_path))

    recorder = build_gguf_dsv4_layer_oracle_recorder(
        quantization_method="gguf_dsv4",
        enforce_eager=True,
        tensor_parallel_rank=0,
        expected_layer_count=2,
        vocab_size=129280,
    )
    assert recorder is not None
    assert not recorder.matches_forward(
        torch.tensor([128000, 123]), torch.tensor([0, 1])
    )
    assert not recorder.matches_forward(
        torch.tensor(token_ids), torch.tensor([1, 2, 3])
    )
    assert recorder.matches_forward(torch.tensor(token_ids), torch.tensor([0, 1, 2]))

    recorder.record_attention_layer(0, torch.tensor([[0.5, 1.0], [1.5, 2.0]]))
    recorder.record_layer(0, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
    recorder.record_attention_layer(1, torch.tensor([[2.5, 3.0], [3.5, 4.0]]))
    recorder.record_layer(1, torch.tensor([[5.0, 6.0], [7.0, 8.0]]))
    recorder.record_logits(torch.tensor([[-1.0, 0.0], [1.0, 2.0]]))
    recorder.finish()

    manifest = json.loads((output_dir / "manifest.json").read_text())
    assert manifest["format"] == "gguf-dsv4-layer-oracle-v1"
    assert manifest["token_ids"] == list(token_ids)
    assert (
        manifest["token_ids_sha256"]
        == hashlib.sha256(token_path.read_bytes()).hexdigest()
    )
    assert [entry["layer"] for entry in manifest["attention_layers"]] == [0, 1]
    assert [entry["layer"] for entry in manifest["layers"]] == [0, 1]
    logits_entry = manifest["logits"]
    logits_path = output_dir / logits_entry["path"]
    assert torch.equal(
        torch.load(logits_path, weights_only=True), torch.tensor([1.0, 2.0])
    )
    for entry in manifest["layers"]:
        layer_path = output_dir / entry["path"]
        assert layer_path.stat().st_size == entry["size"]
        assert hashlib.sha256(layer_path.read_bytes()).hexdigest() == entry["sha256"]
        saved = torch.load(layer_path, weights_only=True)
        assert saved.dtype == torch.float32
        assert tuple(saved.shape) == (2, 2)

    assert not recorder.matches_forward(
        torch.tensor(token_ids), torch.tensor([0, 1, 2])
    )


def test_layer_oracle_rejects_incomplete_or_duplicate_layers(tmp_path, monkeypatch):
    token_path = tmp_path / "tokens.txt"
    _write_token_ids(token_path)
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_DIR, str(tmp_path / "out"))
    monkeypatch.setenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE, str(token_path))
    recorder = build_gguf_dsv4_layer_oracle_recorder(
        quantization_method="gguf_dsv4",
        enforce_eager=True,
        tensor_parallel_rank=0,
        expected_layer_count=2,
    )
    assert recorder is not None
    assert recorder.matches_forward(
        torch.tensor([128000, 123, 456]), torch.tensor([0, 1, 2])
    )
    recorder.record_attention_layer(0, torch.zeros(4))
    recorder.record_layer(0, torch.zeros(4))
    with pytest.raises(ValueError, match="duplicate post-FFN layer 0"):
        recorder.record_layer(0, torch.zeros(4))
    with pytest.raises(ValueError, match="expected 2 recorded"):
        recorder.finish()


def test_deepseek_v4_forward_records_reconstructed_layer_boundaries(monkeypatch):
    class FakePipelineGroup:
        is_first_rank = True
        is_last_rank = True

    class FakeLayer:
        def __init__(self, increment):
            self.increment = increment

        def __call__(self, hidden_states, *args):
            return hidden_states + self.increment, None, None, None, None

    class FakeRecorder:
        def __init__(self):
            self.layers = []
            self.finished = False

        def matches_forward(self, input_ids, positions):
            return True

        def record_layer(self, layer_index, final_token_state):
            self.layers.append((layer_index, final_token_state.clone()))

        def finish(self):
            self.finished = True

    recorder = FakeRecorder()
    fake_model = type("FakeDeepseekV4Model", (), {})()
    fake_model.use_mega_moe = False
    fake_model.embed_input_ids = lambda input_ids: torch.zeros(3, 4, 2)
    fake_model.layers = [FakeLayer(1.0), FakeLayer(2.0)]
    fake_model.start_layer = 0
    fake_model.end_layer = 2
    fake_model.aux_hidden_state_layers = set()
    fake_model._mtp_hidden_buffer = None
    fake_model.hc_head_fn = torch.empty(0)
    fake_model.hc_head_scale = torch.empty(0)
    fake_model.hc_head_base = torch.empty(0)
    fake_model.rms_norm_eps = 1e-6
    fake_model.hc_eps = 1e-6
    fake_model.norm = lambda hidden_states: hidden_states
    fake_model.layer_oracle_recorder = recorder

    monkeypatch.setattr(deepseek_v4_model, "get_pp_group", FakePipelineGroup)
    monkeypatch.setattr(
        deepseek_v4_model,
        "mhc_post_tilelang",
        lambda hidden_states, *args, **kwargs: hidden_states,
    )
    monkeypatch.setattr(
        deepseek_v4_model,
        "hc_head_fused_kernel_tilelang",
        lambda hidden_states, *args, **kwargs: hidden_states.mean(dim=1),
    )

    output = deepseek_v4_model.DeepseekV4Model.forward(
        fake_model,
        torch.tensor([128000, 123, 456]),
        torch.tensor([0, 1, 2]),
        None,
    )

    assert output.shape == (3, 2)
    assert not recorder.finished
    assert [layer_index for layer_index, _ in recorder.layers] == [0, 1]
    assert torch.equal(recorder.layers[0][1], torch.ones(4, 2))
    assert torch.equal(recorder.layers[1][1], torch.full((4, 2), 3.0))


def test_compute_logits_finishes_layer_oracle_capture():
    class FakeRecorder:
        def __init__(self):
            self.logits = None
            self.finished = False
            self.is_capturing = True

        def record_logits(self, logits):
            self.logits = logits.clone()

        def finish(self):
            self.finished = True

    recorder = FakeRecorder()
    fake_causal_lm = type("FakeDeepseekV4ForCausalLM", (), {})()
    fake_causal_lm.model = type("FakeModel", (), {"layer_oracle_recorder": recorder})()
    fake_causal_lm.lm_head = object()
    fake_causal_lm.logits_processor = lambda lm_head, hidden_states: hidden_states + 10

    logits = deepseek_v4_model.DeepseekV4ForCausalLM.compute_logits(
        fake_causal_lm, torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    )

    assert torch.equal(logits, torch.tensor([[11.0, 12.0], [13.0, 14.0]]))
    assert torch.equal(recorder.logits, logits)
    assert recorder.finished
