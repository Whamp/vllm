# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.quantization import auto_gptq_expert_vmm
from vllm.model_executor.layers.quantization.auto_gptq_expert_vmm import (
    AutoGPTQExpertVMM,
)


def _make_layer() -> torch.nn.Module:
    layer = torch.nn.Module()
    layer.register_parameter(
        "w13_qweight",
        torch.nn.Parameter(
            torch.tensor([[10, 11], [20, 21], [30, 31]], dtype=torch.int32),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w2_qweight",
        torch.nn.Parameter(
            torch.tensor([[100], [200], [300]], dtype=torch.int32),
            requires_grad=False,
        ),
    )
    layer.register_parameter(
        "w13_scales",
        torch.nn.Parameter(
            torch.tensor([[1.0], [2.0], [3.0]]),
            requires_grad=False,
        ),
    )
    layer.w13_weight = layer.w13_qweight
    layer.w2_weight = layer.w2_qweight
    layer.global_num_experts = 5
    layer.expert_map = torch.tensor([-1, 2, 0, -1, 1], dtype=torch.int32)
    layer.expert_map_manager = SimpleNamespace(_expert_map=layer.expert_map)
    return layer


def test_auto_gptq_expert_vmm_places_only_exact_ranked_layer(tmp_path, monkeypatch):
    target_layer = "language_model.model.layers.0.mlp.experts"
    rankings_path = tmp_path / "rankings.json"
    rankings_path.write_text(f'{{"{target_layer}": [4, 1]}}')
    manager = AutoGPTQExpertVMM(hot_experts=2, rankings_path=str(rankings_path))
    layer = _make_layer()

    def fake_allocate(source, new_to_old, hot_experts):
        destination = torch.index_select(source, 0, new_to_old)
        allocation = SimpleNamespace(device_bytes=16, host_bytes=8)
        return destination, allocation

    monkeypatch.setattr(
        auto_gptq_expert_vmm,
        "_allocate_mixed_vmm_tensor",
        fake_allocate,
    )
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: None)

    placement = manager.place_experts(layer, target_layer)

    assert placement is not None
    assert placement.hot_experts == 2
    assert placement.device_bytes == 32
    assert placement.host_bytes == 16
    assert torch.equal(
        layer.w13_qweight,
        torch.tensor([[20, 21], [30, 31], [10, 11]], dtype=torch.int32),
    )
    assert torch.equal(layer.w2_qweight, torch.tensor([[200], [300], [100]]))
    assert torch.equal(layer.w13_scales, torch.tensor([[2.0], [3.0], [1.0]]))
    assert torch.equal(layer._expert_map, torch.tensor([-1, 1, 2, -1, 0]))
    assert layer.expert_map_manager._expert_map is layer._expert_map


def test_auto_gptq_expert_vmm_rejects_missing_layer_prefix(tmp_path):
    rankings_path = tmp_path / "rankings.json"
    rankings_path.write_text('{"language_model.model.layers.0.mlp.experts": [4, 1]}')
    manager = AutoGPTQExpertVMM(hot_experts=2, rankings_path=str(rankings_path))

    with pytest.raises(RuntimeError, match="requires the routed-expert layer prefix"):
        manager.place_experts(_make_layer(), "")


def test_auto_gptq_expert_vmm_does_not_touch_unlisted_mtp_layer(tmp_path):
    rankings_path = tmp_path / "rankings.json"
    rankings_path.write_text('{"language_model.model.layers.0.mlp.experts": [4, 1]}')
    manager = AutoGPTQExpertVMM(hot_experts=2, rankings_path=str(rankings_path))
    layer = _make_layer()
    original_w13 = layer.w13_qweight

    placement = manager.place_experts(
        layer,
        "language_model.mtp.layers.0.mlp.experts",
    )

    assert placement is None
    assert layer.w13_qweight is original_w13
