# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp MTP's exact local-argmax adapter uses the collapsed LM-head input."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch import nn

from vllm.models.qwen4_exp.nvidia.mtp import Qwen4ExpMTP
from vllm.v1.worker.gpu.spec_decode.speculator import DraftModelSpeculator


def make_mtp_with_readout():
    model = Qwen4ExpMTP.__new__(Qwen4ExpMTP)
    nn.Module.__init__(model)
    model.lm_head = nn.Linear(3, 8, bias=False)
    model.logits_processor = Mock()
    return model


def test_qwen4_mtp_local_argmax_delegates_head_and_collapsed_hidden():
    model = make_mtp_with_readout()
    hidden = torch.randn(3, 3)
    expected = torch.tensor([1, 4, 7], dtype=torch.int64)
    model.logits_processor.get_top_tokens.return_value = expected
    actual = model.get_top_tokens(hidden)
    model.logits_processor.get_top_tokens.assert_called_once_with(model.lm_head, hidden)
    assert actual is expected


@pytest.mark.parametrize("local_argmax", [False, True])
def test_qwen4_mtp_greedy_selection_uses_only_selected_readout(local_argmax):
    model = make_mtp_with_readout()
    hidden = torch.randn(2, 3)
    logits = torch.tensor([[0.0, 9.0, 9.0], [3.0, 2.0, 1.0]])
    expected = logits.argmax(dim=-1)
    model.logits_processor.get_top_tokens.return_value = expected
    model.compute_logits = Mock(return_value=logits)
    speculator = SimpleNamespace(use_local_argmax_reduction=local_argmax, model=model)
    actual = DraftModelSpeculator._greedy_sample_draft(speculator, hidden)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    if local_argmax:
        model.compute_logits.assert_not_called()
        model.logits_processor.get_top_tokens.assert_called_once_with(
            model.lm_head, hidden
        )
    else:
        model.compute_logits.assert_called_once_with(hidden)
        model.logits_processor.get_top_tokens.assert_not_called()
