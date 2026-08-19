# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import os

import pytest
import torch

import vllm.model_executor.layers.quantization.gguf_dsv4.route_stats as route_stats


class _FakeLayer:
    def __init__(self, layer_name: str, num_experts: int = 256):
        self.layer_name = layer_name
        self.num_experts = num_experts


@pytest.fixture(autouse=True)
def _reset_route_stats(monkeypatch, tmp_path):
    monkeypatch.setattr(route_stats, "_STATS_DIR", None)
    route_stats.reset_for_tests()
    yield
    route_stats.reset_for_tests()


def _enable(monkeypatch, tmp_path):
    monkeypatch.setattr(route_stats, "_STATS_DIR", str(tmp_path))
    route_stats.reset_for_tests()


def test_disabled_by_default_is_noop(tmp_path):
    layer = _FakeLayer("model.layers.0.mlp.experts.routed_experts")
    route_stats.record_routes(layer, torch.zeros((1, 6), dtype=torch.int32))
    assert route_stats._STATE is None
    assert route_stats.maybe_flush(force=True) == []
    assert list(tmp_path.iterdir()) == []


def test_layer_index_parsing_and_histogram(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    layer0 = _FakeLayer("model.layers.0.mlp.experts.routed_experts")
    layer7 = _FakeLayer("model.layers.7.mlp.experts.routed_experts")
    route_stats.record_routes(layer0, torch.tensor([[1, 2, 3, 4, 5, 6]]))
    route_stats.record_routes(layer0, torch.tensor([[1, 2, 3, 4, 5, 6]]))
    route_stats.record_routes(layer7, torch.arange(42, dtype=torch.int32).reshape(7, 6))
    written = route_stats.maybe_flush(force=True)
    assert len(written) == 2  # forced flush writes hist and ring
    hist_path = next(p for p in written if p.name.startswith("hist-"))
    payload = torch.load(hist_path, weights_only=True)
    assert payload["layers"] == [0, 7]
    assert payload["n_experts"] == 256
    hist = payload["hist"]
    assert hist.shape == (2, 256)
    assert hist[0, 1].item() == 2
    assert hist[0, 6].item() == 2
    assert hist[1, 41].item() == 1
    assert hist[1].sum().item() == 42


def test_ring_records_only_decode_scale_forwards(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    layer = _FakeLayer("model.layers.3.mlp.experts.routed_experts")
    route_stats.record_routes(layer, torch.tensor([[10, 11, 12, 13, 14, 15]]))
    route_stats.record_routes(
        layer, torch.tensor([[20, 21, 22, 23, 24, 25], [30, 31, 32, 33, 34, 35]])
    )
    # Prefill-scale forward: histogram grows, ring does not.
    route_stats.record_routes(layer, torch.zeros((64, 6), dtype=torch.int32))
    written = route_stats.maybe_flush(force=True)
    ring_path = next(p for p in written if p.name.startswith("ring-"))
    payload = torch.load(ring_path, weights_only=True)
    ring = payload["ring"][0]
    pos = payload["ring_pos"][0].item()
    assert pos == 2
    assert ring[1, 0].tolist() == [10, 11, 12, 13, 14, 15]
    assert ring[2, 0].tolist() == [20, 21, 22, 23, 24, 25]
    assert ring[2, 1].tolist() == [30, 31, 32, 33, 34, 35]
    assert ring[2, 2].tolist() == [-1] * 6
    assert ring[0].eq(-1).all()
    assert ring[3].eq(-1).all()


def test_flush_throttle(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    layer = _FakeLayer("model.layers.0.mlp.experts.routed_experts")
    route_stats.record_routes(layer, torch.tensor([[1, 2, 3, 4, 5, 6]]))
    first = route_stats.maybe_flush()
    assert first == []  # record happened < interval ago
    state = route_stats._STATE
    assert state is not None
    state.last_hist_flush -= route_stats._HIST_FLUSH_INTERVAL_S
    state.last_ring_flush -= route_stats._RING_FLUSH_INTERVAL_S
    second = route_stats.maybe_flush()
    assert len(second) == 2
    third = route_stats.maybe_flush()
    assert third == []
    # Atomic writes leave no temporaries behind.
    assert not list(tmp_path.glob("*.tmp"))


def test_unparseable_layer_name_fails_loud(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    layer = _FakeLayer("model.blocks.0.mlp.experts")
    with pytest.raises(ValueError, match="cannot parse a layer index"):
        route_stats.record_routes(layer, torch.tensor([[1, 2, 3, 4, 5, 6]]))


def test_snapshot_payload_metadata(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    layer = _FakeLayer("model.layers.42.mlp.experts.routed_experts")
    route_stats.record_routes(layer, torch.tensor([[0, 1, 2, 3, 4, 5]]))
    written = route_stats.maybe_flush(force=True)
    for path in written:
        payload = torch.load(path, weights_only=True)
        assert payload["top_k"] == 6
        assert payload["pid"] == os.getpid()
        assert payload["layers"] == [42]
        assert path.name.endswith(".pt")
