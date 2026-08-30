# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

ARG BASE_IMAGE
FROM ${BASE_IMAGE}

ARG BASE_IMAGE_ID
ARG ENV_INPUT_SHA256
ARG ENV_OUTPUT_SHA256
ARG MOE_RUNNER_INPUT_SHA256
ARG MOE_RUNNER_OUTPUT_SHA256
ARG SHARED_EXPERTS_INPUT_SHA256
ARG SHARED_EXPERTS_OUTPUT_SHA256

COPY runtime/vllm/envs.py \
    /usr/local/lib/python3.12/dist-packages/vllm/envs.py
COPY runtime/vllm/model_executor/layers/fused_moe/runner/moe_runner.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/runner/moe_runner.py
COPY runtime/vllm/model_executor/layers/fused_moe/runner/shared_experts.py \
    /usr/local/lib/python3.12/dist-packages/vllm/model_executor/layers/fused_moe/runner/shared_experts.py

LABEL io.whamp.qwen38.shared-expert-early-launch="experimental-default-off" \
      io.whamp.qwen38.shared-expert-base-image="${BASE_IMAGE_ID}" \
      io.whamp.qwen38.shared-expert-env-input-sha256="${ENV_INPUT_SHA256}" \
      io.whamp.qwen38.shared-expert-env-output-sha256="${ENV_OUTPUT_SHA256}" \
      io.whamp.qwen38.shared-expert-moe-runner-input-sha256="${MOE_RUNNER_INPUT_SHA256}" \
      io.whamp.qwen38.shared-expert-moe-runner-output-sha256="${MOE_RUNNER_OUTPUT_SHA256}" \
      io.whamp.qwen38.shared-expert-runtime-input-sha256="${SHARED_EXPERTS_INPUT_SHA256}" \
      io.whamp.qwen38.shared-expert-runtime-output-sha256="${SHARED_EXPERTS_OUTPUT_SHA256}"
