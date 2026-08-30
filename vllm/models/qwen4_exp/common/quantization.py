# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Shared Qwen4Exp quantization selection rules."""

from vllm.model_executor.layers.quantization import QuantizationConfig


def without_qwen4_exp_modelopt_fp4(
    quant_config: QuantizationConfig | None,
) -> QuantizationConfig | None:
    """Disable ModelOpt FP4 for Qwen4Exp weights outside its declared scope."""
    if quant_config is not None and quant_config.get_name() == "modelopt_fp4":
        return None
    return quant_config
