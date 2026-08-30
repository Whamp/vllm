# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import cast
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.config import VllmConfig
from vllm.model_executor.layers import ple_offload_layer
from vllm.model_executor.layers.fused_moe.config import FusedMoEConfig
from vllm.model_executor.layers.fused_moe.runner import shared_experts as shared_module
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts
from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn as gdn_module
from vllm.model_executor.layers.mamba.gdn.base import GatedDeltaNetAttention
from vllm.model_executor.layers.mamba.gdn.qwen_gdn_linear_attn import (
    QwenGatedDeltaNetAttention,
)
from vllm.transformers_utils.configs.qwen3_next import Qwen3NextConfig


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.mark.parametrize(
    ("ple_offload_enabled", "offload_process_flag", "expected_offload_process"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_ple_meta_construction_skips_shared_experts_cuda_stream(
    monkeypatch: pytest.MonkeyPatch,
    ple_offload_enabled: bool,
    offload_process_flag: bool,
    expected_offload_process: bool,
) -> None:
    stream = object()
    create_aux_stream = Mock(return_value=stream)
    is_offload_process = Mock(return_value=offload_process_flag)
    monkeypatch.setattr(
        ple_offload_layer,
        "is_offload_process",
        is_offload_process,
    )
    monkeypatch.setattr(shared_module, "aux_stream", create_aux_stream)
    monkeypatch.setattr(
        shared_module.envs,
        "VLLM_DISABLE_SHARED_EXPERTS_STREAM",
        False,
    )
    monkeypatch.setattr(
        shared_module.envs,
        "VLLM_PLE_CPU_OFFLOAD",
        ple_offload_enabled,
    )

    shared_experts = SharedExperts(
        layer=nn.Identity(),
        moe_config=cast(FusedMoEConfig, object()),
        enable_dbo=False,
        mk_can_overlap_shared_experts=lambda: False,
    )

    if expected_offload_process:
        assert shared_experts._stream is None
        create_aux_stream.assert_not_called()
    else:
        assert shared_experts._stream is stream
        create_aux_stream.assert_called_once_with()
    assert is_offload_process.call_count == int(ple_offload_enabled)


class _TestLinear(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(1))


def _initialize_test_gdn_base(
    self: GatedDeltaNetAttention,
    config: object,
    vllm_config: VllmConfig,
    prefix: str,
) -> None:
    del config, vllm_config
    nn.Module.__init__(self)
    self.prefix = prefix
    self.tp_size = 1
    self.tp_rank = 0
    self.hidden_size = 8
    self.layer_norm_epsilon = 1e-6
    self.quant_config = None
    self.speculative_config = None
    self.num_spec = 0


@pytest.mark.parametrize(
    ("ple_offload_enabled", "offload_process_flag", "expected_offload_process"),
    [
        (False, True, False),
        (True, False, False),
        (True, True, True),
    ],
)
def test_ple_meta_construction_keeps_gdn_norm_on_meta(
    monkeypatch: pytest.MonkeyPatch,
    ple_offload_enabled: bool,
    offload_process_flag: bool,
    expected_offload_process: bool,
) -> None:
    cuda_device = torch.device("cuda:0")
    current_device = Mock(return_value=cuda_device)
    norm_devices: list[torch.device] = []
    is_offload_process = Mock(return_value=offload_process_flag)

    monkeypatch.setattr(
        ple_offload_layer,
        "is_offload_process",
        is_offload_process,
    )
    monkeypatch.setattr(
        gdn_module.envs,
        "VLLM_PLE_CPU_OFFLOAD",
        ple_offload_enabled,
    )
    monkeypatch.setattr(
        GatedDeltaNetAttention,
        "__init__",
        _initialize_test_gdn_base,
    )
    monkeypatch.setattr(gdn_module.current_platform, "is_xpu", lambda: False)
    monkeypatch.setattr(gdn_module.current_platform, "is_cpu", lambda: False)
    monkeypatch.setattr(gdn_module.current_platform, "is_rocm", lambda: False)
    monkeypatch.setattr(gdn_module.current_platform, "current_device", current_device)
    monkeypatch.setattr(
        gdn_module, "ColumnParallelLinear", lambda *_args, **_kwargs: _TestLinear()
    )
    monkeypatch.setattr(
        gdn_module, "RowParallelLinear", lambda *_args, **_kwargs: _TestLinear()
    )
    monkeypatch.setattr(
        QwenGatedDeltaNetAttention,
        "create_qkvz_proj",
        lambda *_args, **_kwargs: _TestLinear(),
    )
    monkeypatch.setattr(
        QwenGatedDeltaNetAttention,
        "create_ba_proj",
        lambda *_args, **_kwargs: _TestLinear(),
    )
    monkeypatch.setattr(
        QwenGatedDeltaNetAttention,
        "maybe_disable_tp",
        lambda *_args, **_kwargs: False,
    )
    monkeypatch.setattr(gdn_module, "mamba_v2_sharded_weight_loader", lambda *_: None)
    monkeypatch.setattr(gdn_module, "set_weight_attrs", lambda *_args, **_kwargs: None)

    def make_norm(*_args: object, device: torch.device, **_kwargs: object) -> object:
        norm_devices.append(device)
        return SimpleNamespace(activation="silu")

    monkeypatch.setattr(gdn_module, "RMSNormGated", make_norm)
    monkeypatch.setattr(
        gdn_module,
        "ChunkGatedDeltaRule",
        lambda: SimpleNamespace(gdn_prefill_backend="triton"),
    )
    monkeypatch.setattr(gdn_module.envs, "VLLM_GDN_DECODE_KERNEL", "triton")
    monkeypatch.setattr(
        gdn_module.envs,
        "VLLM_ENABLE_FLA_PACKED_RECURRENT_DECODE",
        False,
    )
    monkeypatch.setattr(
        gdn_module,
        "get_current_vllm_config",
        lambda: SimpleNamespace(
            compilation_config=SimpleNamespace(static_forward_context={})
        ),
    )

    config = SimpleNamespace(
        linear_num_key_heads=1,
        linear_num_value_heads=1,
        linear_key_head_dim=2,
        linear_value_head_dim=2,
        linear_conv_kernel_dim=4,
        output_gate_type="silu",
    )
    QwenGatedDeltaNetAttention(
        config=cast(Qwen3NextConfig, config),
        vllm_config=cast(VllmConfig, object()),
        prefix="model.layers.0.linear_attn",
    )

    if expected_offload_process:
        assert norm_devices == [torch.device("meta")]
        current_device.assert_not_called()
    else:
        assert norm_devices == [cuda_device]
        current_device.assert_called_once_with()
    assert is_offload_process.call_count == int(ple_offload_enabled)
