# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Qwen4Exp checkpoint filtering for the current automatic weight loader."""

from __future__ import annotations

from torch import nn

from vllm.model_executor.models.utils import AutoWeightsLoader


class Qwen4ExpWeightsLoader(AutoWeightsLoader):
    """Filter Qwen4Exp mapped weight names without changing the global loader."""

    def __init__(
        self,
        module: nn.Module,
        *,
        skip_prefixes: list[str] | None = None,
        skip_substrings: list[str] | None = None,
        ignore_unexpected_prefixes: list[str] | None = None,
        ignore_unexpected_suffixes: list[str] | None = None,
    ) -> None:
        super().__init__(
            module,
            ignore_unexpected_prefixes=ignore_unexpected_prefixes,
            ignore_unexpected_suffixes=ignore_unexpected_suffixes,
        )
        self.skip_prefixes = tuple(skip_prefixes or ())
        self.skip_substrings = tuple(skip_substrings or ())

    def _can_skip(self, qualname: str) -> bool:
        return (
            super()._can_skip(qualname)
            or qualname.startswith(self.skip_prefixes)
            or any(substring in qualname for substring in self.skip_substrings)
        )
