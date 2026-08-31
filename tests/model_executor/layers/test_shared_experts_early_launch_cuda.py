# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""RTX 3090 acceptance tests for CUDA shared-expert early launch."""

from types import SimpleNamespace

import pytest
import torch

from vllm.model_executor.layers.fused_moe.runner import (
    shared_experts as shared_experts_module,
)
from vllm.model_executor.layers.fused_moe.runner.shared_experts import SharedExperts

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="CUDA shared-expert acceptance requires a GPU",
)

_HIDDEN_SIZE = 64


def _new_linear() -> torch.nn.Linear:
    torch.manual_seed(8421)
    return torch.nn.Linear(
        _HIDDEN_SIZE,
        _HIDDEN_SIZE,
        bias=False,
        device="cuda",
        dtype=torch.bfloat16,
    )


def _new_shared_experts(
    layer: torch.nn.Module,
    *,
    early_launch: bool,
    enable_dbo: bool = False,
) -> SharedExperts:
    """Construct the production lifecycle around a small real CUDA layer."""
    shared_experts = SharedExperts.__new__(SharedExperts)
    torch.nn.Module.__init__(shared_experts)
    shared_experts.enable_dbo = enable_dbo
    shared_experts._output = [None, None]
    shared_experts._async_in_flight = [False, False]
    shared_experts._layer = layer
    shared_experts._moe_config = SimpleNamespace(
        moe_parallel_config=SimpleNamespace(
            enable_eplb=False,
            all2all_backend=None,
            use_fi_nvl_two_sided_kernels=False,
        )
    )
    shared_experts._mk_can_overlap_shared_experts = lambda: False
    shared_experts._cuda_early_launch = early_launch
    shared_experts._stream = torch.cuda.Stream()
    if early_launch:
        shared_experts._input_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
        shared_experts._output_ready_event = [torch.cuda.Event(), torch.cuda.Event()]
    else:
        shared_experts._input_ready_event = []
        shared_experts._output_ready_event = []
    return shared_experts


def _run_early_launch(
    shared_experts: SharedExperts,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    assert shared_experts.maybe_forward_async(hidden_states)
    shared_experts.wait()
    return shared_experts.output


def _run_legacy_launch(
    shared_experts: SharedExperts,
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    order = shared_experts._determine_shared_experts_order(hidden_states)
    shared_experts.maybe_sync_shared_experts_stream(hidden_states)
    shared_experts(hidden_states, order)
    return shared_experts.output


@pytest.mark.parametrize("num_tokens", [1, 2, 256])
def test_cuda_shared_expert_early_launch_matches_legacy(num_tokens: int) -> None:
    layer = _new_linear()
    hidden_states = torch.randn(
        num_tokens,
        _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    early = _new_shared_experts(layer, early_launch=True)
    legacy = _new_shared_experts(layer, early_launch=False)

    expected = _run_legacy_launch(legacy, hidden_states)
    actual = _run_early_launch(early, hidden_states)
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cuda_shared_expert_early_launch_preserves_threshold_fallback() -> None:
    layer = _new_linear()
    hidden_states = torch.randn(
        257,
        _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    shared_experts = _new_shared_experts(layer, early_launch=True)

    assert not shared_experts.maybe_forward_async(hidden_states)
    actual = _run_legacy_launch(shared_experts, hidden_states)
    expected = layer(hidden_states)
    torch.accelerator.synchronize()

    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_cuda_shared_expert_early_launch_isolates_dbo_slots(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    layer = _new_linear()
    shared_experts = _new_shared_experts(layer, early_launch=True, enable_dbo=True)
    current_slot = 0
    monkeypatch.setattr(
        shared_experts_module,
        "dbo_current_ubatch_id",
        lambda: current_slot,
    )
    inputs = [
        torch.randn(2, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16),
        torch.randn(2, _HIDDEN_SIZE, device="cuda", dtype=torch.bfloat16),
    ]

    current_slot = 0
    assert shared_experts.maybe_forward_async(inputs[0])
    current_slot = 1
    assert shared_experts.maybe_forward_async(inputs[1])

    outputs = []
    for current_slot in (0, 1):
        shared_experts.wait()
        outputs.append(shared_experts.output)
    torch.accelerator.synchronize()

    for actual, hidden_states in zip(outputs, inputs, strict=True):
        torch.testing.assert_close(actual, layer(hidden_states), rtol=0, atol=0)


class _RaiseAfterCudaSubmission(torch.nn.Module):
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        torch.sin(hidden_states)
        raise ValueError("intentional CUDA shared-expert failure")


def test_cuda_shared_expert_partial_submission_can_retry() -> None:
    shared_experts = _new_shared_experts(
        _RaiseAfterCudaSubmission(),
        early_launch=True,
    )
    hidden_states = torch.randn(
        2,
        _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )

    with pytest.raises(ValueError, match="intentional CUDA shared-expert failure"):
        shared_experts.maybe_forward_async(hidden_states)
    assert shared_experts._output == [None, None]
    assert shared_experts._async_in_flight == [False, False]

    shared_experts._layer = _new_linear()
    actual = _run_early_launch(shared_experts, hidden_states)
    torch.accelerator.synchronize()
    torch.testing.assert_close(
        actual,
        shared_experts._layer(hidden_states),
        rtol=0,
        atol=0,
    )


@pytest.mark.parametrize("num_tokens", [1, 2])
def test_cuda_shared_expert_early_launch_graph_replay(num_tokens: int) -> None:
    layer = _new_linear()
    shared_experts = _new_shared_experts(layer, early_launch=True)
    static_input = torch.randn(
        num_tokens,
        _HIDDEN_SIZE,
        device="cuda",
        dtype=torch.bfloat16,
    )
    static_output = torch.empty_like(static_input)

    for _ in range(3):
        static_output.copy_(_run_early_launch(shared_experts, static_input))
    torch.accelerator.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        launched = shared_experts.maybe_forward_async(static_input)
        assert launched
        shared_experts.wait()
        static_output.copy_(shared_experts.output)

    replay_outputs = []
    for seed in (11, 29, 47):
        torch.manual_seed(seed)
        replay_input = torch.randn_like(static_input)
        static_input.copy_(replay_input)
        graph.replay()
        torch.accelerator.synchronize()
        replay_outputs.append(static_output.clone())
        torch.testing.assert_close(
            static_output,
            layer(replay_input),
            rtol=0,
            atol=0,
        )

    assert any(
        not torch.equal(replay_outputs[0], replay_output)
        for replay_output in replay_outputs[1:]
    )
