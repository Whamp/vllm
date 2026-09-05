# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bounded PLE prefill output, history, graph replay, and allocation contracts."""

import json
import statistics
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F
from torch import nn

from vllm.models.qwen4_exp.nvidia.ple_layer import (
    Qwen4ExpPLEGroupedNorm,
    Qwen4ExpPLELayer,
)

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.fixture(autouse=True)
def enable_bounded_ple_prefill(monkeypatch):
    monkeypatch.setenv("VLLM_QWEN4_EXP_PLE_PREFILL_TRITON", "1")
    monkeypatch.setenv("VLLM_QWEN4_EXP_PLE_GROUPED_NORM_TRITON", "1")
    previous_tf32 = torch.backends.cudnn.allow_tf32
    torch.backends.cudnn.allow_tf32 = False
    yield
    torch.backends.cudnn.allow_tf32 = previous_tf32


def test_ragged_lengths_and_metadata_offsets_reuse_compiled_kernels(monkeypatch):
    from triton import knobs

    from vllm.models.qwen4_exp.nvidia.ops.ple_prefill import ple_prefill_convolution

    state = torch.zeros(6, 64, 11, device="cuda", dtype=torch.bfloat16)
    weights = torch.randn(64, 4, device="cuda", dtype=torch.bfloat16)

    def run_shape(tokens, requests, offset):
        x = torch.randn(tokens, 64, device="cuda", dtype=torch.bfloat16)
        starts_storage = torch.zeros(
            offset + requests + 1, device="cuda", dtype=torch.int32
        )
        starts = starts_storage[offset:]
        starts.copy_(
            torch.tensor(
                [tokens * i // requests for i in range(requests + 1)],
                device="cuda",
                dtype=torch.int32,
            )
        )
        index_storage = torch.zeros(offset + requests, device="cuda", dtype=torch.int32)
        indices = index_storage[offset:]
        indices.copy_(torch.arange(1, requests + 1, device="cuda", dtype=torch.int32))
        initial = torch.ones(offset + requests, device="cuda", dtype=torch.bool)[
            offset:
        ]
        ple_prefill_convolution(
            x, state, weights, starts, indices, initial, 3, torch.empty_like(x)
        )
        torch.cuda.synchronize()

    run_shape(17, 1, 0)
    compiled = []
    previous_hook = knobs.runtime.jit_post_compile_hook

    def record_compilation(**kwargs):
        name = getattr(kwargs.get("fn"), "name", "")
        if name.startswith("_ple_prefill_"):
            compiled.append(name)
        if previous_hook is not None:
            return previous_hook(**kwargs)
        return None

    monkeypatch.setattr(knobs.runtime, "jit_post_compile_hook", record_compilation)
    for shape in ((1, 1, 1), (16, 2, 2), (31, 3, 3), (64, 4, 0), (73, 2, 1)):
        run_shape(*shape)
    assert compiled == [], (
        f"Ragged PLE inputs created new JIT specializations: {compiled}"
    )


def make_inputs(lengths, hidden, dilation, dtype, strided=False, kernel=4):
    torch.manual_seed(9325)
    state_len = dilation * (kernel - 1)
    requests = len(lengths)
    x = torch.randn(sum(lengths), hidden * 2, device="cuda", dtype=dtype)[:, ::2]
    weights = (torch.randn(hidden, kernel * 2, device="cuda", dtype=dtype) / 4)[:, ::2]
    # Extra leading and trailing storage catches damage to unrelated state.
    backing = torch.randn(
        requests + 2, state_len + 5, hidden, device="cuda", dtype=dtype
    )
    state = backing.transpose(1, 2)[..., 2:]
    if not strided:
        x, weights, state = x.contiguous(), weights.contiguous(), state.contiguous()
    indices = torch.arange(1, requests + 1, device="cuda", dtype=torch.int32)
    if requests > 2:
        indices[1] = 0  # Padding block: output zero and state unchanged.
    initial = torch.tensor([index % 2 == 0 for index in range(requests)], device="cuda")
    offsets = torch.tensor(
        [0, *torch.tensor(lengths).cumsum(0).tolist()], device="cuda", dtype=torch.int32
    )
    metadata = SimpleNamespace(
        non_spec_query_start_loc=offsets,
        has_initial_states_p=initial,
        max_prefill_query_len=max(lengths, default=0),
    )
    layer = Qwen4ExpPLELayer.__new__(Qwen4ExpPLELayer)
    nn.Module.__init__(layer)
    layer.conv_state_len = state_len
    layer.short_conv_dilation = dilation
    return layer, x, metadata, state, weights, indices


def prefill(layer, x, metadata, state, weights, indices):
    return layer._short_conv_dilated_prefill_batched(
        x, metadata, state, weights, indices, len(indices), 0, len(x)
    )


def reference_prefill(x, state, weights, indices, initial, lengths, dilation):
    """Independent per-request convolution, without production's batch padding."""
    state_len = dilation * (weights.shape[1] - 1)
    output = torch.zeros_like(x)
    cursor = 0
    for request, length in enumerate(lengths):
        index = int(indices[request])
        if length and index:
            past = state[index, :, :state_len].clone()
            if not bool(initial[request]):
                past.zero_()
            history = torch.cat((past, x[cursor : cursor + length].T), dim=1)
            conv = F.conv1d(
                history[None],
                weights[:, None].contiguous(),
                groups=x.shape[1],
                dilation=dilation,
            )
            output[cursor : cursor + length] = F.silu(conv)[0].T
            if state_len:
                state[index, :, :state_len] = history[:, -state_len:]
        cursor += length
    return output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize(
    "lengths,hidden,dilation,strided,kernel",
    [
        ([1], 1, 3, False, 4),
        ([2, 19], 17, 3, True, 4),
        ([0, 6, 1, 33, 0], 95, 3, False, 4),
        ([3, 7, 2], 257, 1, True, 4),
        ([4, 2], 33, 3, True, 1),
        ([0, 0], 17, 3, False, 4),
    ],
)
def test_ple_prefill_matches_per_request_convolution(
    dtype, lengths, hidden, dilation, strided, kernel
):
    layer, x, metadata, state, weights, indices = make_inputs(
        lengths, hidden, dilation, dtype, strided, kernel
    )
    expected_state = state.clone()
    backing_prefix = state._base[:, :2].clone() if state._base is not None else None
    expected = reference_prefill(
        x,
        expected_state,
        weights,
        indices,
        metadata.has_initial_states_p,
        lengths,
        dilation,
    )
    output = prefill(layer, x, metadata, state, weights, indices)
    torch.testing.assert_close(
        output,
        expected,
        rtol=0.016
        if dtype == torch.bfloat16
        else 0.002
        if dtype == torch.float16
        else 2e-5,
        atol=2e-5 if dtype != torch.float32 else 2e-6,
    )
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)
    if backing_prefix is not None:
        torch.testing.assert_close(state._base[:, :2], backing_prefix, rtol=0, atol=0)


def test_ple_prefill_chunk_boundary_and_graph_replay_preserve_state():
    lengths = [3, 8, 1]
    layer, x, metadata, state, weights, indices = make_inputs(
        lengths, 257, 3, torch.bfloat16, True
    )
    expected_state = state.clone()
    prefill(layer, x, metadata, state, weights, indices)  # Compile before capture.
    state.copy_(expected_state)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = prefill(layer, x, metadata, state, weights, indices)
    state.copy_(expected_state)
    for step in range(3):
        x.mul_(0.5)
        if step:
            metadata.has_initial_states_p.fill_(True)
        expected = reference_prefill(
            x,
            expected_state,
            weights,
            indices,
            metadata.has_initial_states_p,
            lengths,
            3,
        )
        graph.replay()
        torch.testing.assert_close(output, expected, rtol=0.016, atol=2e-5)
        torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_ple_prefill_workspace_is_bounded_by_real_output_size():
    # Four requests, only one long. Padding this batch multiplies workspace.
    layer, x, metadata, state, weights, indices = make_inputs(
        [4090, 2, 2, 2], 10240, 3, torch.bfloat16
    )
    warmup = prefill(layer, x, metadata, state, weights, indices)
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = prefill(layer, x, metadata, state, weights, indices)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - baseline
    allowed = output.numel() * output.element_size() + 1024**2
    assert peak_extra <= allowed, (
        f"PLE prefill workspace {peak_extra} exceeds output plus 1 MiB {allowed}"
    )


def test_ple_prefill_subtracts_decode_token_metadata_offset():
    lengths = [2, 19]
    layer, x, metadata, state, weights, indices = make_inputs(
        lengths, 95, 3, torch.bfloat16
    )
    expected_state = state.clone()
    expected = reference_prefill(
        x, expected_state, weights, indices, metadata.has_initial_states_p, lengths, 3
    )
    metadata.non_spec_query_start_loc += 7
    output = layer._short_conv_dilated_prefill_batched(
        x, metadata, state, weights, indices, len(indices), 7, len(x)
    )
    torch.testing.assert_close(output, expected, rtol=0.016, atol=2e-5)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


def test_ple_prefill_disabled_kernel_uses_torch_fallback(monkeypatch):
    monkeypatch.setenv("VLLM_QWEN4_EXP_PLE_PREFILL_TRITON", "0")

    def forbidden_kernel(*args, **kwargs):
        raise AssertionError("disabled PLE prefill kernel was called")

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.ple_layer.ple_prefill_convolution",
        forbidden_kernel,
    )
    layer, x, metadata, state, weights, indices = make_inputs(
        [3], 17, 3, torch.bfloat16
    )
    expected_state = state.clone()
    expected = reference_prefill(
        x, expected_state, weights, indices, metadata.has_initial_states_p, [3], 3
    )
    output = prefill(layer, x, metadata, state, weights, indices)
    torch.testing.assert_close(output, expected, rtol=0.016, atol=2e-5)
    torch.testing.assert_close(state, expected_state, rtol=0, atol=0)


@pytest.mark.parametrize("lengths", [[2048], [2042, 2, 2, 2], [4096], [4090, 2, 2, 2]])
def test_ple_prefill_reports_matched_kernel_cost(monkeypatch, lengths):
    layer, x, metadata, state, weights, indices = make_inputs(
        lengths, 10240, 3, torch.bfloat16
    )
    expected = None
    for backend, enabled in (("torch", "0"), ("triton", "1")):
        monkeypatch.setenv("VLLM_QWEN4_EXP_PLE_PREFILL_TRITON", enabled)
        for _ in range(5):
            output = prefill(layer, x, metadata, state, weights, indices)
            del output
        torch.cuda.synchronize()
        baseline = torch.cuda.memory_allocated()
        torch.cuda.reset_peak_memory_stats()
        elapsed = []
        for _ in range(9):
            start, end = (
                torch.cuda.Event(enable_timing=True),
                torch.cuda.Event(enable_timing=True),
            )
            start.record()
            output = prefill(layer, x, metadata, state, weights, indices)
            end.record()
            end.synchronize()
            elapsed.append(start.elapsed_time(end))
            del output
        peak_extra = torch.cuda.max_memory_allocated() - baseline
        result = {
            "backend": backend,
            "lengths": lengths,
            "hidden": 10240,
            "median_gpu_ms": statistics.median(elapsed),
            "gpu_ms": elapsed,
            "peak_extra_bytes": peak_extra,
        }
        print("PLE_PREFILL_BENCHMARK=" + json.dumps(result), flush=True)
        output = prefill(layer, x, metadata, state, weights, indices)
        if expected is None:
            expected = output.clone()
        else:
            torch.testing.assert_close(output, expected, rtol=0.016, atol=2e-5)
        del output


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
@pytest.mark.parametrize("hidden,group_size", [(17, None), (256, 16), (10240, 2560)])
@torch.inference_mode()
def test_ple_grouped_norm_matches_independent_rmsnorm(dtype, hidden, group_size):
    torch.manual_seed(142)
    module = Qwen4ExpPLEGroupedNorm(hidden, 1e-6, group_size, dtype).cuda()
    module.weight.normal_(std=0.25)
    inputs = torch.randn(9, hidden, device="cuda", dtype=dtype)
    width = group_size or hidden
    groups = hidden // width
    normalized = F.rms_norm(
        inputs.float().reshape(9, groups, width), (width,), eps=module.eps
    ).reshape_as(inputs)
    expected = (normalized * (1.0 + module.weight.float())).to(dtype)
    actual = module(inputs)
    tolerance = {torch.bfloat16: 0.016, torch.float16: 0.002, torch.float32: 2e-5}
    torch.testing.assert_close(actual, expected, rtol=tolerance[dtype], atol=2e-6)


@torch.inference_mode()
def test_ple_grouped_norm_workspace_is_bounded_by_output_size():
    module = Qwen4ExpPLEGroupedNorm(10240, 1e-6, 2560, torch.bfloat16).cuda()
    inputs = torch.randn(4096, 10240, device="cuda", dtype=torch.bfloat16)
    warmup = module(inputs)
    torch.cuda.synchronize()
    del warmup
    torch.cuda.empty_cache()
    baseline = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    output = module(inputs)
    torch.cuda.synchronize()
    peak_extra = torch.cuda.max_memory_allocated() - baseline
    allowed = output.numel() * output.element_size() + 1024**2
    print(f"PLE_GROUPED_NORM_PEAK_EXTRA_BYTES={peak_extra}", flush=True)
    assert peak_extra <= allowed, (
        f"PLE grouped norm workspace {peak_extra} exceeds {allowed}"
    )


@pytest.mark.parametrize("layout", ["disabled", "strided", "rank3", "empty", "cpu"])
@torch.inference_mode()
def test_ple_grouped_norm_retains_fallback(monkeypatch, layout):
    device = "cpu" if layout == "cpu" else "cuda"
    module = Qwen4ExpPLEGroupedNorm(17, 1e-6, None, torch.bfloat16).to(device)
    inputs = torch.randn(6, 17, device=device, dtype=torch.bfloat16)
    if layout == "disabled":
        monkeypatch.setenv("VLLM_QWEN4_EXP_PLE_GROUPED_NORM_TRITON", "0")
    elif layout == "strided":
        inputs = torch.randn(6, 34, device=device, dtype=torch.bfloat16)[:, ::2]
    elif layout == "rank3":
        inputs = inputs.reshape(2, 3, 17)
    elif layout == "empty":
        inputs = inputs[:0]

    def forbidden_kernel(*args, **kwargs):
        raise AssertionError("unsupported PLE norm input reached the fused kernel")

    monkeypatch.setattr(
        "vllm.models.qwen4_exp.nvidia.ple_layer.grouped_gemma_rmsnorm", forbidden_kernel
    )
    expected = F.rms_norm(inputs.float(), (17,), eps=module.eps).to(inputs.dtype)
    torch.testing.assert_close(module(inputs), expected, rtol=0.016, atol=2e-6)


@torch.inference_mode()
def test_ple_grouped_norm_graph_replay_reads_new_input_and_weight():
    module = Qwen4ExpPLEGroupedNorm(256, 1e-6, 64, torch.bfloat16).cuda()
    inputs = torch.randn(9, 256, device="cuda", dtype=torch.bfloat16)
    module(inputs)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        output = module(inputs)
    for scale in (0.25, 0.5, -0.5):
        inputs.mul_(scale)
        module.weight.add_(0.125)
        graph.replay()
        norm = F.rms_norm(
            inputs.float().reshape(9, 4, 64), (64,), eps=module.eps
        ).reshape_as(inputs)
        expected = (norm * (1.0 + module.weight.float())).to(inputs.dtype)
        torch.testing.assert_close(output, expected, rtol=0.016, atol=2e-6)
