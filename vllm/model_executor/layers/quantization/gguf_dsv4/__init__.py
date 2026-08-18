# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Native quantization adapters for the DeepSeek V4 GGUF runtime."""

from vllm.model_executor.layers.quantization.gguf_dsv4.config import (
    GGUFDSV4LinearMethod,
    GGUFDSV4MoEMethod,
    GGUFDSV4QuantConfig,
)

__all__ = [
    "GGUFDSV4LinearMethod",
    "GGUFDSV4MoEMethod",
    "GGUFDSV4QuantConfig",
]
