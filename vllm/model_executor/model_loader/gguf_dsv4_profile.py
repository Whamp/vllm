# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Identity-bound quantized source-type profiles for DeepSeek V4 GGUF."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

from vllm.model_executor.model_loader.gguf_dsv4_index import (
    GGUF_QUANTIZED_TYPE_NAMES,
    GGUFTensorEntry,
)
from vllm.model_executor.model_loader.gguf_dsv4_plan import (
    classify_gguf_dsv4_tensor,
)


def _canonical_source_types_bytes(source_types: Mapping[str, str]) -> bytes:
    return json.dumps(
        source_types,
        sort_keys=True,
        separators=(",", ":"),
    ).encode()


@dataclass(frozen=True)
class GGUFDSV4SourceProfile:
    """Bind every quantized GGUF source tensor to its exact storage type."""

    source_types: Mapping[str, str]
    sha256: str

    @classmethod
    def from_entries(cls, entries: Sequence[GGUFTensorEntry]) -> GGUFDSV4SourceProfile:
        source_types = {
            entry.name: entry.type_name
            for entry in entries
            if entry.type_name in GGUF_QUANTIZED_TYPE_NAMES
        }
        return cls._from_source_types(source_types)

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> GGUFDSV4SourceProfile:
        raw_source_types = config.get("source_quant_types")
        if not isinstance(raw_source_types, dict) or not raw_source_types:
            raise ValueError(
                "GGUF DSv4 source_quant_types must be a non-empty dictionary"
            )
        source_types = {}
        for source_name, type_name in raw_source_types.items():
            if not isinstance(source_name, str) or not isinstance(type_name, str):
                raise ValueError(
                    "GGUF DSv4 source quant profile entries must be strings"
                )
            source_types[source_name] = type_name
        profile = cls._from_source_types(source_types)
        expected_sha256 = config.get("source_quant_types_sha256")
        if expected_sha256 != profile.sha256:
            raise ValueError(
                "GGUF DSv4 source quant profile SHA-256 mismatch: "
                f"expected {expected_sha256!r}, computed {profile.sha256}"
            )
        return profile

    @classmethod
    def _from_source_types(
        cls, source_types: Mapping[str, str]
    ) -> GGUFDSV4SourceProfile:
        if not source_types:
            raise ValueError("GGUF DSv4 source quant profile cannot be empty")
        normalized = dict(sorted(source_types.items()))
        for source_name, type_name in normalized.items():
            classify_gguf_dsv4_tensor(source_name)
            if type_name not in GGUF_QUANTIZED_TYPE_NAMES:
                raise ValueError(
                    f"GGUF DSv4 source {source_name} has non-quantized type {type_name}"
                )
        digest = hashlib.sha256(_canonical_source_types_bytes(normalized)).hexdigest()
        return cls(MappingProxyType(normalized), digest)

    def require_type(self, source_name: str) -> str:
        """Return one source tensor type or fail on an incomplete profile."""
        try:
            return self.source_types[source_name]
        except KeyError as error:
            raise ValueError(
                f"GGUF DSv4 source quant profile is missing {source_name}"
            ) from error
