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
import re
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from typing import Final

import torch

GGUF_DSV4_LAYER_ORACLE_DIR: Final = "VLLM_GGUF_DSV4_LAYER_ORACLE_DIR"
GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE: Final = (
    "VLLM_GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE"
)
_GGUF_DSV4_LAYER_ORACLE_FORMAT: Final = "gguf-dsv4-layer-oracle-v1"
_GGUF_DSV4_LAYER_PATTERN: Final = re.compile(r"(?:^|\.)layers\.(\d+)(?:\.|$)")
_ACTIVE_GGUF_DSV4_LAYER_ORACLE_RECORDER: GgufDsv4LayerOracleRecorder | None = None


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
        self._ffn_input_entries: dict[int, dict[str, object]] = {}
        self._ffn_route_entries: dict[int, dict[str, object]] = {}
        self._ffn_route_weight_entries: dict[int, dict[str, object]] = {}
        self._ffn_gate_layer_entries: dict[int, dict[str, object]] = {}
        self._ffn_up_layer_entries: dict[int, dict[str, object]] = {}
        self._ffn_routed_layer_entries: dict[int, dict[str, object]] = {}
        self._ffn_shared_layer_entries: dict[int, dict[str, object]] = {}
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
        elif boundary_name == "FFN input":
            entries = self._ffn_input_entries
            filename_prefix = "ffn-input-layer"
        elif boundary_name == "FFN routed":
            entries = self._ffn_routed_layer_entries
            filename_prefix = "ffn-routed-layer"
        elif boundary_name == "FFN shared":
            entries = self._ffn_shared_layer_entries
            filename_prefix = "ffn-shared-layer"
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

    def record_ffn_input(
        self, layer_index: int, final_token_state: torch.Tensor
    ) -> None:
        """Record the normalized final-token input entering the routed/shared FFN."""
        self._record_layer_boundary(
            _LayerBoundaryCapture("FFN input", layer_index, final_token_state)
        )

    def record_ffn_routing(
        self,
        layer_index: int,
        topk_ids: torch.Tensor,
        topk_weights: torch.Tensor,
    ) -> None:
        """Record the final token's ordered routed expert IDs and weights."""
        if not self.is_capturing:
            raise ValueError("GGUF DSV4 layer oracle is not capturing a forward")
        if layer_index in self._ffn_route_entries:
            raise ValueError(
                f"GGUF DSV4 layer oracle duplicate FFN routes layer {layer_index}"
            )
        if not 0 <= layer_index < self.expected_layer_count:
            raise ValueError(
                f"GGUF DSV4 layer oracle FFN routes layer index out of range: "
                f"{layer_index}"
            )
        saved_ids = topk_ids.detach().to(device="cpu", dtype=torch.int32).contiguous()
        saved_weights = (
            topk_weights.detach().to(device="cpu", dtype=torch.float32).contiguous()
        )
        if saved_ids.ndim != 1 or saved_ids.shape != saved_weights.shape:
            raise ValueError("GGUF DSV4 layer oracle FFN routes shape mismatch")
        if not torch.isfinite(saved_weights).all():
            raise ValueError("GGUF DSV4 layer oracle non-finite FFN route weights")
        for kind, tensor, entries in (
            ("routes", saved_ids, self._ffn_route_entries),
            ("route-weights", saved_weights, self._ffn_route_weight_entries),
        ):
            relative_path = f"ffn-{kind}-layer-{layer_index:03d}.pt"
            destination = self.output_dir / relative_path
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(tensor, temporary)
            payload = temporary.read_bytes()
            os.replace(temporary, destination)
            entries[layer_index] = {
                "layer": layer_index,
                "path": relative_path,
                "shape": list(tensor.shape),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

    def record_ffn_gate_up(
        self,
        layer_index: int,
        gate: torch.Tensor,
        up: torch.Tensor,
    ) -> None:
        """Record all-gathered final-token gate and up expert outputs."""
        if not self.is_capturing:
            raise ValueError("GGUF DSV4 layer oracle is not capturing a forward")
        if layer_index in self._ffn_gate_layer_entries:
            raise ValueError(
                f"GGUF DSV4 layer oracle duplicate FFN gate/up layer {layer_index}"
            )
        if gate.shape != up.shape:
            raise ValueError("GGUF DSV4 layer oracle FFN gate/up shape mismatch")
        for kind, tensor, entries in (
            ("gate", gate, self._ffn_gate_layer_entries),
            ("up", up, self._ffn_up_layer_entries),
        ):
            saved = tensor.detach().to(device="cpu", dtype=torch.float32).contiguous()
            if not torch.isfinite(saved).all():
                raise ValueError(
                    f"GGUF DSV4 layer oracle non-finite FFN {kind} layer {layer_index}"
                )
            relative_path = f"ffn-{kind}-layer-{layer_index:03d}.pt"
            destination = self.output_dir / relative_path
            temporary = destination.with_suffix(".pt.tmp")
            torch.save(saved, temporary)
            payload = temporary.read_bytes()
            os.replace(temporary, destination)
            entries[layer_index] = {
                "layer": layer_index,
                "path": relative_path,
                "shape": list(saved.shape),
                "size": len(payload),
                "sha256": hashlib.sha256(payload).hexdigest(),
            }

    def record_ffn_component(
        self,
        layer_index: int,
        component_name: str,
        final_token_state: torch.Tensor,
    ) -> None:
        """Record one separately reduced routed or shared FFN output."""
        if component_name not in ("routed", "shared"):
            raise ValueError(
                "GGUF DSV4 layer oracle FFN component must be routed or shared"
            )
        self._record_layer_boundary(
            _LayerBoundaryCapture(
                f"FFN {component_name}", layer_index, final_token_state
            )
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
        actual_ffn_inputs = set(self._ffn_input_entries)
        actual_ffn_routes = set(self._ffn_route_entries)
        actual_ffn_route_weights = set(self._ffn_route_weight_entries)
        actual_ffn_gate_layers = set(self._ffn_gate_layer_entries)
        actual_ffn_up_layers = set(self._ffn_up_layer_entries)
        actual_ffn_routed_layers = set(self._ffn_routed_layer_entries)
        actual_ffn_shared_layers = set(self._ffn_shared_layer_entries)
        actual_layers = set(self._layer_entries)
        if actual_attention_layers != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded attention layers, got "
                f"{len(actual_attention_layers)}"
            )
        if actual_ffn_inputs != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded FFN input layers, got "
                f"{len(actual_ffn_inputs)}"
            )
        if actual_ffn_routes != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded FFN route layers, got "
                f"{len(actual_ffn_routes)}"
            )
        if actual_ffn_route_weights != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded FFN route-weight layers, got "
                f"{len(actual_ffn_route_weights)}"
            )
        if actual_ffn_gate_layers != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded FFN gate layers, got "
                f"{len(actual_ffn_gate_layers)}"
            )
        if actual_ffn_up_layers != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded FFN up layers, got "
                f"{len(actual_ffn_up_layers)}"
            )
        if actual_ffn_routed_layers != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded routed FFN layers, got "
                f"{len(actual_ffn_routed_layers)}"
            )
        if actual_ffn_shared_layers != expected_layers:
            raise ValueError(
                "GGUF DSV4 layer oracle expected "
                f"{self.expected_layer_count} recorded shared FFN layers, got "
                f"{len(actual_ffn_shared_layers)}"
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
            "ffn_inputs": [
                self._ffn_input_entries[index] for index in sorted(actual_ffn_inputs)
            ],
            "ffn_routes": [
                self._ffn_route_entries[index] for index in sorted(actual_ffn_routes)
            ],
            "ffn_route_weights": [
                self._ffn_route_weight_entries[index]
                for index in sorted(actual_ffn_route_weights)
            ],
            "ffn_gate_layers": [
                self._ffn_gate_layer_entries[index]
                for index in sorted(actual_ffn_gate_layers)
            ],
            "ffn_up_layers": [
                self._ffn_up_layer_entries[index]
                for index in sorted(actual_ffn_up_layers)
            ],
            "ffn_routed_layers": [
                self._ffn_routed_layer_entries[index]
                for index in sorted(actual_ffn_routed_layers)
            ],
            "ffn_shared_layers": [
                self._ffn_shared_layer_entries[index]
                for index in sorted(actual_ffn_shared_layers)
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


@cache
def _read_gguf_dsv4_layer_oracle_runtime_token_ids(
    token_ids_path_value: str,
) -> tuple[int, ...]:
    token_ids, _ = _read_gguf_dsv4_layer_oracle_token_ids(
        Path(token_ids_path_value), 129280
    )
    return token_ids


def maybe_record_gguf_dsv4_ffn_routing(
    *,
    layer_name: str,
    topk_ids: torch.Tensor,
    topk_weights: torch.Tensor,
) -> None:
    """Record the rank-zero ordered FFN routes during the exact oracle pass."""
    recorder = _ACTIVE_GGUF_DSV4_LAYER_ORACLE_RECORDER
    if recorder is None or not recorder.is_capturing:
        return
    match = _GGUF_DSV4_LAYER_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(
            f"GGUF DSV4 layer oracle cannot parse FFN layer name: {layer_name}"
        )
    recorder.record_ffn_routing(
        int(match.group(1)),
        topk_ids.reshape(-1, topk_ids.shape[-1])[-1],
        topk_weights.reshape(-1, topk_weights.shape[-1])[-1],
    )


def maybe_record_gguf_dsv4_ffn_gate_up(
    *,
    layer_name: str,
    gate: torch.Tensor,
    up: torch.Tensor,
) -> None:
    """All-gather and capture final-token gate/up outputs for diagnosis."""
    if os.getenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE) is None:
        return
    from vllm.distributed import tensor_model_parallel_all_gather

    gathered_gate = tensor_model_parallel_all_gather(gate[-1], dim=-1)
    gathered_up = tensor_model_parallel_all_gather(up[-1], dim=-1)
    recorder = _ACTIVE_GGUF_DSV4_LAYER_ORACLE_RECORDER
    if recorder is None or not recorder.is_capturing:
        return
    match = _GGUF_DSV4_LAYER_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(
            f"GGUF DSV4 layer oracle cannot parse FFN layer name: {layer_name}"
        )
    recorder.record_ffn_gate_up(int(match.group(1)), gathered_gate, gathered_up)


def maybe_record_gguf_dsv4_ffn_components(
    *,
    layer_name: str,
    input_ids: torch.Tensor | None,
    routed_output: torch.Tensor,
    shared_output: torch.Tensor | None,
    outputs_reduced: bool,
) -> None:
    """Capture globally reduced FFN branches for the exact oracle request."""
    token_ids_path_value = os.getenv(GGUF_DSV4_LAYER_ORACLE_TOKEN_IDS_FILE)
    if token_ids_path_value is None or input_ids is None:
        return
    token_ids = _read_gguf_dsv4_layer_oracle_runtime_token_ids(token_ids_path_value)
    if input_ids.numel() != len(token_ids):
        return
    expected_tokens = torch.tensor(
        token_ids,
        device=input_ids.device,
        dtype=input_ids.dtype,
    )
    if not torch.equal(input_ids.reshape(-1), expected_tokens):
        return
    if shared_output is None:
        raise ValueError("GGUF DSV4 layer oracle requires a shared FFN output")
    match = _GGUF_DSV4_LAYER_PATTERN.search(layer_name)
    if match is None:
        raise ValueError(
            f"GGUF DSV4 layer oracle cannot parse FFN layer name: {layer_name}"
        )

    routed_capture = routed_output.detach().clone()
    shared_capture = shared_output.detach().clone()
    if not outputs_reduced:
        from vllm.distributed import tensor_model_parallel_all_reduce

        routed_capture = tensor_model_parallel_all_reduce(routed_capture)
        shared_capture = tensor_model_parallel_all_reduce(shared_capture)

    recorder = _ACTIVE_GGUF_DSV4_LAYER_ORACLE_RECORDER
    if recorder is None or not recorder.is_capturing:
        return
    layer_index = int(match.group(1))
    recorder.record_ffn_component(layer_index, "routed", routed_capture[-1])
    recorder.record_ffn_component(layer_index, "shared", shared_capture[-1])


def build_gguf_dsv4_layer_oracle_recorder(
    *,
    quantization_method: str | None,
    enforce_eager: bool,
    tensor_parallel_rank: int,
    expected_layer_count: int,
    vocab_size: int = 129280,
) -> GgufDsv4LayerOracleRecorder | None:
    """Build the opt-in rank-zero recorder or fail closed on an invalid runtime."""
    global _ACTIVE_GGUF_DSV4_LAYER_ORACLE_RECORDER
    _ACTIVE_GGUF_DSV4_LAYER_ORACLE_RECORDER = None
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
    recorder = GgufDsv4LayerOracleRecorder(
        output_dir=Path(output_dir_value),
        token_ids=token_ids,
        token_ids_sha256=token_ids_sha256,
        expected_layer_count=expected_layer_count,
    )
    _ACTIVE_GGUF_DSV4_LAYER_ORACLE_RECORDER = recorder
    return recorder
