# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Diagnostic DeepSeek V4 layer-state capture for the native GGUF oracle.

This module is inert unless both documented environment variables are set.
It exists only to compare the GGUF-TP runtime with the pinned llama.cpp model
at decoder-layer boundaries; it is not a serving or observability feature.
"""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import torch

GGUF_DSV4_LAYER_ORACLE_DIR: Final = "VLLM_GGUF_DSV4_LAYER_ORACLE_DIR"
GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE: Final = (
    "VLLM_GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE"
)
_GGUF_DSV4_LAYER_ORACLE_FORMAT: Final = "gguf-dsv4-layer-oracle-v1"


@dataclass(frozen=True)
class _LayerBoundaryCapture:
    boundary_name: str
    layer_index: int
    final_token_state: torch.Tensor


def _read_gguf_dsv4_layer_oracle_token_ids(
    token_ids_path: Path,
    vocab_size: int,
) -> tuple[tuple[int, ...], str]:
    try:
        token_bytes = token_ids_path.read_bytes()
        fields = token_bytes.decode("ascii").split()
        token_ids = tuple(int(field, 10) for field in fields)
    except (OSError, UnicodeDecodeError, ValueError) as error:
        raise ValueError(
            "GGUF DSV4 layer oracle token IDs must be readable ASCII integers"
        ) from error
    if not token_ids or any(
        token_id < 0 or token_id >= vocab_size for token_id in token_ids
    ):
        raise ValueError(
            "GGUF DSV4 layer oracle token IDs must be non-empty and within "
            f"[0, {vocab_size})"
        )
    return token_ids, hashlib.sha256(token_bytes).hexdigest()


class GgufDsv4LayerOracleRecorder:
    """Atomically records one exact GGUF DeepSeek V4 decoder pass on TP rank 0."""

    def __init__(
        self,
        *,
        output_dir: Path,
        token_ids: tuple[int, ...],
        token_ids_sha256: str,
        expected_layer_count: int,
    ) -> None:
        if expected_layer_count <= 0:
            raise ValueError(
                "GGUF DSV4 layer oracle expected layer count must be positive"
            )
        if output_dir.exists() and any(output_dir.iterdir()):
            raise ValueError(
                f"GGUF DSV4 layer oracle output directory is not empty: {output_dir}"
            )
        output_dir.mkdir(parents=True, exist_ok=True)
        self.output_dir = output_dir
        self.token_ids = token_ids
        self.token_ids_sha256 = token_ids_sha256
        self.expected_layer_count = expected_layer_count
        self._capturing = False
        self._finished = False
        self._attention_layer_entries: dict[int, dict[str, object]] = {}
        self._layer_entries: dict[int, dict[str, object]] = {}
        self._logits_entry: dict[str, object] | None = None

    @property
    def is_capturing(self) -> bool:
        """Report whether the exact trigger forward awaits layer/logit completion."""
        return self._capturing and not self._finished

    def matches_forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
    ) -> bool:
        """Claim the first exact-token forward beginning at position zero."""
        if self._capturing or self._finished:
            return False
        if input_ids.numel() != len(self.token_ids) or positions.numel() != len(
            self.token_ids
        ):
            return False
        expected_tokens = torch.tensor(
            self.token_ids,
            dtype=input_ids.dtype,
            device=input_ids.device,
        )
        expected_positions = torch.arange(
            len(self.token_ids),
            dtype=positions.dtype,
            device=positions.device,
        )
        if not torch.equal(input_ids.reshape(-1), expected_tokens):
            return False
        if not torch.equal(positions.reshape(-1), expected_positions):
            return False
        self._capturing = True
        return True

    def _record_layer_boundary(self, capture: _LayerBoundaryCapture) -> None:
        boundary_name = capture.boundary_name
        layer_index = capture.layer_index
        final_token_state = capture.final_token_state
        if boundary_name == "attention":
            entries = self._attention_layer_entries
            filename_prefix = "attention-layer"
        elif boundary_name == "post-FFN":
            entries = self._layer_entries
            filename_prefix = "layer"
        else:
            raise ValueError(
                f"GGUF DSV4 layer oracle unknown boundary: {boundary_name}"
            )
        if not self.is_capturing:
            raise ValueError("GGUF DSV4 layer oracle is not capturing a forward")
        if layer_index in entries:
            raise ValueError(
                f"GGUF DSV4 layer oracle duplicate {boundary_name} layer {layer_index}"
            )
        if not 0 <= layer_index < self.expected_layer_count:
            raise ValueError(
                f"GGUF DSV4 layer oracle {boundary_name} layer index out of range: "
                f"{layer_index}"
            )

        saved_state = final_token_state.detach().to(device="cpu", dtype=torch.float32)
        saved_state = saved_state.contiguous()
        if not torch.isfinite(saved_state).all():
            raise ValueError(
                f"GGUF DSV4 layer oracle non-finite {boundary_name} layer {layer_index}"
            )

        relative_path = f"{filename_prefix}-{layer_index:03d}.pt"
        destination = self.output_dir / relative_path
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(saved_state, temporary)
        payload = temporary.read_bytes()
        os.replace(temporary, destination)
        entries[layer_index] = {
            "layer": layer_index,
            "path": relative_path,
            "shape": list(saved_state.shape),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def record_attention_layer(
        self, layer_index: int, final_token_state: torch.Tensor
    ) -> None:
        """Record reconstructed HC state after attention and before the FFN."""
        self._record_layer_boundary(
            _LayerBoundaryCapture("attention", layer_index, final_token_state)
        )

    def record_layer(self, layer_index: int, final_token_state: torch.Tensor) -> None:
        """Record reconstructed HC state after the layer FFN."""
        self._record_layer_boundary(
            _LayerBoundaryCapture("post-FFN", layer_index, final_token_state)
        )

    def record_logits(self, logits: torch.Tensor) -> None:
        """Record the final-token vocabulary logits after output projection."""
        if not self.is_capturing:
            raise ValueError("GGUF DSV4 layer oracle is not capturing a forward")
        if self._logits_entry is not None:
            raise ValueError("GGUF DSV4 layer oracle duplicate logits")
        if logits.ndim < 1 or logits.shape[-1] <= 0:
            raise ValueError("GGUF DSV4 layer oracle logits shape is invalid")
        saved_logits = logits.detach().reshape(-1, logits.shape[-1])[-1]
        saved_logits = saved_logits.to(device="cpu", dtype=torch.float32).contiguous()
        if not torch.isfinite(saved_logits).all():
            raise ValueError("GGUF DSV4 layer oracle non-finite logits")

        relative_path = "logits.pt"
        destination = self.output_dir / relative_path
        temporary = destination.with_suffix(".pt.tmp")
        torch.save(saved_logits, temporary)
        payload = temporary.read_bytes()
        os.replace(temporary, destination)
        self._logits_entry = {
            "path": relative_path,
            "shape": list(saved_logits.shape),
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }

    def finish(self) -> None:
        """Publish a manifest only after every expected layer has been recorded."""
        expected_layers = set(range(self.expected_layer_count))
        actual_attention_layers = set(self._attention_layer_entries)
        actual_layers = set(self._layer_entries)
        if actual_attention_layers != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded attention layers, got "
                f"{len(actual_attention_layers)}"
            )
        if actual_layers != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded layers, got {len(actual_layers)}"
            )
        if self._logits_entry is None:
            raise ValueError("GGUF DSV4 layer oracle expected recorded logits")
        manifest = {
            "format": _GGUF_DSV4_LAYER_ORACLE_FORMAT,
            "token_ids": list(self.token_ids),
            "token_ids_sha256": self.token_ids_sha256,
            "attention_layers": [
                self._attention_layer_entries[index]
                for index in sorted(actual_attention_layers)
            ],
            "layers": [self._layer_entries[index] for index in sorted(actual_layers)],
            "logits": self._logits_entry,
        }
        destination = self.output_dir / "manifest.json"
        temporary = destination.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
        os.replace(temporary, destination)
        self._finished = True
        self._capturing = False


def build_gguf_dsv4_layer_oracle_recorder(
    *,
    quantization_method: str | None,
    enforce_eager: bool,
    tensor_parallel_rank: int,
    expected_layer_count: int,
    vocab_size: int = 129280,
) -> GgufDsv4LayerOracleRecorder | None:
    """Build the opt-in rank-zero recorder or fail closed on an invalid runtime."""
    output_dir_value = os.getenv(GGUF_DSV4_LAYER_ORACLE_DIR)
    token_ids_path_value = os.getenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE)
    if output_dir_value is None and token_ids_path_value is None:
        return None
    if not output_dir_value or not token_ids_path_value:
        raise ValueError(
            "GGUF DSV4 layer oracle requires both "
            f"{GGUF_DSV4_LAYER_ORACLE_DIR} and "
            f"{GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE}"
        )
    if quantization_method != "gguf_dsv4":
        raise ValueError(
            "GGUF DSV4 layer oracle requires quantization_method=gguf_dsv4"
        )
    if not enforce_eager:
        raise ValueError("GGUF DSV4 layer oracle requires enforce_eager=True")
    if tensor_parallel_rank != 0:
        return None

    token_ids, token_ids_sha256 = _read_gguf_dsv4_layer_oracle_token_ids(
        Path(token_ids_path_value), vocab_size
    )
    return GgufDsv4LayerOracleRecorder(
        output_dir=Path(output_dir_value),
        token_ids=token_ids,
        token_ids_sha256=token_ids_sha256,
        expected_layer_count=expected_layer_count,
    )
