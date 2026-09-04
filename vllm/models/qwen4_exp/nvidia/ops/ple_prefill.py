# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Dilated PLE prefill without request padding or full-size intermediates."""

import torch

from vllm.triton_utils import tl, triton


@triton.jit
def _ple_prefill_convolution_kernel(
    X,
    STATE,
    WEIGHTS,
    STARTS,
    INDICES,
    INITIAL,
    OUTPUT,
    X_TOKEN_STRIDE: tl.constexpr,
    X_CHANNEL_STRIDE: tl.constexpr,
    STATE_REQUEST_STRIDE: tl.constexpr,
    STATE_CHANNEL_STRIDE: tl.constexpr,
    STATE_TIME_STRIDE: tl.constexpr,
    WEIGHT_CHANNEL_STRIDE: tl.constexpr,
    WEIGHT_TIME_STRIDE: tl.constexpr,
    START_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    INITIAL_STRIDE: tl.constexpr,
    OUTPUT_TOKEN_STRIDE: tl.constexpr,
    OUTPUT_CHANNEL_STRIDE: tl.constexpr,
    TOKENS: tl.constexpr,
    CHANNELS: tl.constexpr,
    REQUESTS: tl.constexpr,
    STATE_ROWS: tl.constexpr,
    HISTORY: tl.constexpr,
    KERNEL_SIZE: tl.constexpr,
    DILATION: tl.constexpr,
    BLOCK_TOKENS: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
):
    tokens = tl.program_id(0) * BLOCK_TOKENS + tl.arange(0, BLOCK_TOKENS)
    channels = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    request = tl.full((BLOCK_TOKENS,), -1, tl.int32)
    sequence_start = tl.full((BLOCK_TOKENS,), 0, tl.int32)
    # Serving batches contain few requests. Resolve ownership once per token
    # tile rather than materializing a padded [requests, max_length, channels].
    for row in range(REQUESTS):
        start = tl.load(STARTS + row * START_STRIDE).to(tl.int32)
        end = tl.load(STARTS + (row + 1) * START_STRIDE).to(tl.int32)
        owns_token = (tokens >= start) & (tokens < end)
        request = tl.where(owns_token, row, request)
        sequence_start = tl.where(owns_token, start, sequence_start)
    token_valid = (tokens < TOKENS) & (request >= 0)
    state_index = tl.load(INDICES + request * INDEX_STRIDE, token_valid, other=0)
    initial = tl.load(INITIAL + request * INITIAL_STRIDE, token_valid, other=0)
    valid = token_valid[:, None] & (channels[None, :] < CHANNELS)
    accumulator = tl.full((BLOCK_TOKENS, BLOCK_CHANNELS), 0, tl.float32)
    for tap in tl.static_range(KERNEL_SIZE):
        relative = tokens - sequence_start - HISTORY + tap * DILATION
        current = tl.load(
            X
            + (sequence_start + relative).to(tl.int64)[:, None] * X_TOKEN_STRIDE
            + channels[None, :] * X_CHANNEL_STRIDE,
            valid & (relative[:, None] >= 0),
            other=0,
        ).to(tl.float32)
        previous = (
            tl.load(
                STATE
                + state_index.to(tl.int64)[:, None] * STATE_REQUEST_STRIDE
                + channels[None, :] * STATE_CHANNEL_STRIDE
                + (relative[:, None] + HISTORY) * STATE_TIME_STRIDE,
                valid
                & (relative[:, None] < 0)
                & (state_index[:, None] != 0)
                & initial[:, None]
                & (STATE_ROWS > 0),
                other=0,
            )
            .to(X.dtype.element_ty)
            .to(tl.float32)
        )
        value = tl.where(relative[:, None] >= 0, current, previous)
        weight = tl.load(
            WEIGHTS + channels * WEIGHT_CHANNEL_STRIDE + tap * WEIGHT_TIME_STRIDE,
            channels < CHANNELS,
            other=0,
        ).to(tl.float32)
        accumulator += value * weight[None, :]
    # Match torch conv1d -> dtype rounding -> SiLU, not one FP32 fused expression.
    rounded = accumulator.to(X.dtype.element_ty).to(tl.float32)
    result = rounded / (1.0 + tl.exp(-rounded))
    result = tl.where(state_index[:, None] != 0, result, 0.0)
    tl.store(
        OUTPUT
        + tokens.to(tl.int64)[:, None] * OUTPUT_TOKEN_STRIDE
        + channels[None, :] * OUTPUT_CHANNEL_STRIDE,
        result,
        valid,
    )


@triton.jit
def _ple_prefill_update_history_kernel(
    X,
    STATE,
    STARTS,
    INDICES,
    INITIAL,
    X_TOKEN_STRIDE: tl.constexpr,
    X_CHANNEL_STRIDE: tl.constexpr,
    STATE_REQUEST_STRIDE: tl.constexpr,
    STATE_CHANNEL_STRIDE: tl.constexpr,
    STATE_TIME_STRIDE: tl.constexpr,
    START_STRIDE: tl.constexpr,
    INDEX_STRIDE: tl.constexpr,
    INITIAL_STRIDE: tl.constexpr,
    CHANNELS: tl.constexpr,
    HISTORY: tl.constexpr,
    BLOCK_CHANNELS: tl.constexpr,
    BLOCK_HISTORY: tl.constexpr,
):
    request = tl.program_id(0)
    channels = tl.program_id(1) * BLOCK_CHANNELS + tl.arange(0, BLOCK_CHANNELS)
    positions = tl.arange(0, BLOCK_HISTORY)
    start = tl.load(STARTS + request * START_STRIDE)
    end = tl.load(STARTS + (request + 1) * START_STRIDE)
    length = end - start
    state_index = tl.load(INDICES + request * INDEX_STRIDE)
    initial = tl.load(INITIAL + request * INITIAL_STRIDE)
    relative = length - HISTORY + positions
    valid = (
        (channels[:, None] < CHANNELS)
        & (positions[None, :] < HISTORY)
        & (state_index != 0)
        & (length > 0)
    )
    current = tl.load(
        X
        + (start + relative).to(tl.int64)[None, :] * X_TOKEN_STRIDE
        + channels[:, None] * X_CHANNEL_STRIDE,
        valid & (relative[None, :] >= 0),
        other=0,
    )
    previous = tl.load(
        STATE
        + state_index.to(tl.int64) * STATE_REQUEST_STRIDE
        + channels[:, None] * STATE_CHANNEL_STRIDE
        + (relative[None, :] + HISTORY) * STATE_TIME_STRIDE,
        valid & (relative[None, :] < 0) & initial,
        other=0,
    ).to(X.dtype.element_ty)
    updated = tl.where(relative[None, :] >= 0, current, previous)
    # Each program owns all history positions for its disjoint channel tile.
    # Load the complete old tile before storing; short chunks cannot race a
    # different program reading the same history. Speculative tail is untouched.
    tl.store(
        STATE
        + state_index.to(tl.int64) * STATE_REQUEST_STRIDE
        + channels[:, None] * STATE_CHANNEL_STRIDE
        + positions[None, :] * STATE_TIME_STRIDE,
        updated,
        valid,
    )


def ple_prefill_convolution(
    x: torch.Tensor,
    conv_state: torch.Tensor,
    conv_weights: torch.Tensor,
    query_start_loc: torch.Tensor,
    state_indices: torch.Tensor,
    has_initial_states: torch.Tensor,
    dilation: int,
    output: torch.Tensor,
) -> torch.Tensor:
    """Write ragged prefill output, then update history on the caller's stream.

    Metadata uses zero-based offsets covering exactly x; nonzero state indices
    are valid and unique. Block zero is padding. Shapes/strides are host metadata:
    the hot path never reads tensor values onto the CPU. Caller owns output and
    caches; output must not alias x/state. No allocation scales with padded size.
    """
    requests = query_start_loc.numel() - 1
    tokens, channels = x.shape
    kernel_size = conv_weights.shape[1]
    history = dilation * (kernel_size - 1)
    if tokens == 0 or requests == 0:
        return output
    _ple_prefill_convolution_kernel[
        (triton.cdiv(tokens, 8), triton.cdiv(channels, 128))
    ](
        x,
        conv_state,
        conv_weights,
        query_start_loc,
        state_indices,
        has_initial_states,
        output,
        *x.stride(),
        *conv_state.stride(),
        *conv_weights.stride(),
        query_start_loc.stride(0),
        state_indices.stride(0),
        has_initial_states.stride(0),
        *output.stride(),
        tokens,
        channels,
        requests,
        conv_state.shape[0],
        history,
        kernel_size,
        dilation,
        BLOCK_TOKENS=8,
        BLOCK_CHANNELS=128,
        num_warps=4,
    )
    if history > 0 and conv_state.shape[0] > 0:
        _ple_prefill_update_history_kernel[(requests, triton.cdiv(channels, 128))](
            x,
            conv_state,
            query_start_loc,
            state_indices,
            has_initial_states,
            *x.stride(),
            *conv_state.stride(),
            query_start_loc.stride(0),
            state_indices.stride(0),
            has_initial_states.stride(0),
            channels,
            history,
            BLOCK_CHANNELS=128,
            BLOCK_HISTORY=triton.next_power_of_2(history),
            num_warps=4,
        )
    return output
