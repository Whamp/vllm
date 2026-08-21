# SPDX-License-Identifier: Apache-2.0
"""Qwen3.5 with replayed (relayered) middle-layer spans for inference.

Re-executes contiguous spans of decoder layers extra times, matching the
recurrent-depth topology used by the rys experiments. Every replay occurrence
is a distinct module instance with its own ``layer_idx``, so vLLM allocates
separate KV-cache and GDN-state slots for it — the same logical-cache contract
the training probe enforces (original layers keep slot ids ``0..L-1``; each
replay occurrence occupies the next contiguous block of slot ids).

Execution order for spans [[12, 20), [20, 28)] replayed once each:

    L0 ... L11, L12 ... L19, R12 ... R19, L20 ... L27, R20 ... R27, ...

where ``R(i)`` is the replay-slot instance whose weights are tied to original
layer ``i``.

Activate with:

.. code-block:: sh

    vllm serve <checkpoint> \\
        --hf-overrides '{"architectures": ["Qwen3_5ReplayForCausalLM"], \\
                         "replay_spans": [[12, 44]]}'

Scope notes:

- Raw replay only. Repair adapters (bridges, gates, norm deltas) are not yet
  applied at inference; this serves untrained repeated-model baseline arms.
- Text-only ``ForCausalLM`` path. Multimodal conditional-generation wrappers
  are not yet wired.
- Pipeline parallelism and speculative decoding are rejected at init;
  tensor parallelism follows the base implementation.
"""

import re
from collections.abc import Iterable

import torch
import torch.nn as nn

from vllm.config import VllmConfig
from vllm.distributed import get_pp_group
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.vocab_parallel_embedding import (
    VocabParallelEmbedding,
)
from vllm.model_executor.models.qwen3_5 import (
    Qwen3_5DecoderLayer,
    Qwen3_5ForCausalLMBase,
    Qwen3_5Model,
)
from vllm.model_executor.models.qwen3_next import (
    _all_gather_hidden_and_residual,
)
from vllm.model_executor.models.utils import (
    AutoWeightsLoader,
    extract_layer_index,
    make_empty_intermediate_tensors_factory,
    make_layers,
)

# Matches "...layers.<idx>.<rest>" with any or no leading namespace, so both
# "model.layers.12.attn.w" and "layers.12.attn.w" duplicate correctly. The
# lookbehinds keep the leading dot out of the match so reconstruction can use
# name[:match.start()] verbatim.
_LAYER_NAME_RE = re.compile(r"(?:^|(?<=\.))layers\.(\d+)\.(.+)")


def normalize_replay_spans(
    replay_spans: list[list[int]],
    num_hidden_layers: int,
) -> list[tuple[int, int]]:
    """Validate and normalize raw span lists into sorted tuples."""
    spans: list[tuple[int, int]] = []
    for raw in replay_spans:
        if len(raw) != 2:
            raise ValueError(f"Each replay span must be [start, end); got {raw!r}")
        start, end = int(raw[0]), int(raw[1])
        if not 0 <= start < end <= num_hidden_layers:
            raise ValueError(
                f"Replay span [{start}, {end}) outside layer range "
                f"[0, {num_hidden_layers})"
            )
        spans.append((start, end))
    spans.sort()
    for (_, e0), (s1, _) in zip(spans, spans[1:]):
        if s1 < e0:
            raise ValueError(
                f"Overlapping replay spans: [{s1}, ...) starts before {e0}."
            )
    return spans


class ReplayLayout:
    """Slot layout and execution schedule for a replayed Qwen3.5 model."""

    def __init__(
        self,
        num_original_layers: int,
        spans: list[tuple[int, int]],
    ) -> None:
        self.num_original_layers = num_original_layers
        self.spans = spans

        self.slot_to_source: dict[int, int] = {}
        next_slot = num_original_layers
        # Insert each span's replay occurrence immediately after the last
        # original layer of that span executes.
        occurrence_by_trigger: dict[int, list[int]] = {}
        for start, end in spans:
            occurrence = []
            for src in range(start, end):
                self.slot_to_source[next_slot] = src
                occurrence.append(next_slot)
                next_slot += 1
            occurrence_by_trigger.setdefault(end - 1, []).extend(occurrence)

        schedule: list[int] = []
        for layer_idx in range(num_original_layers):
            schedule.append(layer_idx)
            schedule.extend(occurrence_by_trigger.get(layer_idx, []))
        self.execution_schedule = schedule

    @property
    def num_slots(self) -> int:
        return len(self.execution_schedule)


def _duplicate_replay_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    slot_to_source: dict[int, int],
) -> Iterable[tuple[str, torch.Tensor]]:
    """Yield every checkpoint tensor once per replay slot it feeds."""
    slots_by_source: dict[int, list[int]] = {}
    for slot, source in slot_to_source.items():
        slots_by_source.setdefault(source, []).append(slot)

    for name, tensor in weights:
        yield name, tensor
        match = _LAYER_NAME_RE.search(name)
        if match is None:
            continue
        source = int(match.group(1))
        targets = slots_by_source.get(source)
        if not targets:
            continue
        head = name[: match.start()]
        rest = match.group(2)
        for slot in targets:
            yield f"{head}layers.{slot}.{rest}", tensor


class Qwen3_5ReplayModel(Qwen3_5Model):
    """Qwen3.5 model body executing configured spans extra times."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_text_config

        replay_spans_raw = getattr(config, "replay_spans", None)
        if not replay_spans_raw:
            raise ValueError(
                "Qwen3_5ReplayModel requires 'replay_spans' in the model "
                "config; pass them via --hf-overrides."
            )
        spans = normalize_replay_spans(replay_spans_raw,
                                       config.num_hidden_layers)
        self.layout = ReplayLayout(config.num_hidden_layers, spans)

        if get_pp_group().world_size > 1:
            raise NotImplementedError(
                "Qwen3_5ReplayModel does not support pipeline parallelism.")
        if vllm_config.speculative_config is not None:
            raise NotImplementedError(
                "Qwen3_5ReplayModel does not support speculative decoding.")

        # NOTE: intentionally skipping Qwen3_5Model.__init__: the parent
        # constructs exactly config.num_hidden_layers layer slots, while we
        # need one module per logical cache slot (originals + replays).
        nn.Module.__init__(self)

        self.config = config
        self.quant_config = vllm_config.quant_config
        self.vocab_size = config.vocab_size
        parallel_config = vllm_config.parallel_config
        self.num_redundant_experts = parallel_config.eplb_config.num_redundant_experts

        self.embed_tokens = VocabParallelEmbedding(
            self.vocab_size,
            config.hidden_size,
        )

        def get_layer(prefix: str):
            slot = extract_layer_index(prefix)
            source = self.layout.slot_to_source.get(slot, slot)
            return Qwen3_5DecoderLayer(
                vllm_config,
                layer_type=config.layer_types[source],
                prefix=prefix,
            )

        self.start_layer, self.end_layer, self.layers = make_layers(
            self.layout.num_slots, get_layer, prefix=f"{prefix}.layers"
        )
        self.make_empty_intermediate_tensors = (
            make_empty_intermediate_tensors_factory(
                ["hidden_states", "residual"], config.hidden_size
            )
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        self.aux_hidden_state_layers: tuple[int, ...] = ()

    @property
    def tie_weights_enabled(self) -> bool:
        return bool(getattr(self.config, "replay_tie_weights", True))

    @torch.no_grad()
    def tie_replay_slot_weights(self) -> None:
        """Point replay-slot parameters at the source layer's tensors.

        Slots already loaded identical values; aliasing removes the duplicate
        memory footprint so a relayered model costs originals-plus-activations
        rather than originals plus duplicated middle weights.
        """
        if not self.tie_weights_enabled:
            return
        for slot, source in self.layout.slot_to_source.items():
            src_params = dict(
                self.layers[source].named_parameters(remove_duplicate=False)
            )
            dst_params = dict(
                self.layers[slot].named_parameters(remove_duplicate=False)
            )
            if set(src_params) != set(dst_params):
                missing = set(src_params) - set(dst_params)
                extra = set(dst_params) - set(src_params)
                raise RuntimeError(
                    f"Replay slot {slot} parameters do not match source "
                    f"layer {source}; cannot tie weights "
                    f"(missing={sorted(missing)}, extra={sorted(extra)})."
                )
            for name, dst_param in dst_params.items():
                dst_param.data = src_params[name].data

    def load_weights(
        self, weights: Iterable[tuple[str, torch.Tensor]]
    ) -> set[str]:
        loader = AutoWeightsLoader(self)
        loaded = loader.load_weights(
            _duplicate_replay_weights(weights, self.layout.slot_to_source),
            mapper=self.hf_to_vllm_mapper,
        )
        self.tie_replay_slot_weights()
        return loaded

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
    ):
        hidden_states = self.embed_input_ids(input_ids)
        residual = None

        full_num_tokens = positions.shape[-1]
        aux_hidden_states: list[torch.Tensor] = []

        for pos, slot in enumerate(self.layout.execution_schedule):
            layer = self.layers[slot]
            gather_needed = (
                hidden_states.shape[0] != full_num_tokens
                and not getattr(layer, "use_attn_reduce_scatter_for_moe", False)
            )
            if gather_needed:
                hidden_states, residual = _all_gather_hidden_and_residual(
                    hidden_states,
                    residual,
                    full_num_tokens,
                    self.config.hidden_size,
                )
            hidden_states, residual = layer(
                positions=positions,
                hidden_states=hidden_states,
                residual=residual,
            )
            if (pos + 1) in self.aux_hidden_state_layers:
                if hidden_states.shape[0] != full_num_tokens:
                    hidden_states, residual = _all_gather_hidden_and_residual(
                        hidden_states,
                        residual,
                        full_num_tokens,
                        self.config.hidden_size,
                    )
                aux_hidden_states.append(hidden_states)

        if hidden_states.shape[0] != full_num_tokens:
            hidden_states, residual = _all_gather_hidden_and_residual(
                hidden_states,
                residual,
                full_num_tokens,
                self.config.hidden_size,
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states


class Qwen3_5ReplayForCausalLM(Qwen3_5ForCausalLMBase):
    """Text-causal LM wrapper serving relayered Qwen3.5 checkpoints."""

    model_cls = Qwen3_5ReplayModel
