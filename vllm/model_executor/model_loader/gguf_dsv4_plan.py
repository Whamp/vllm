# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 GGUF tensor classification and TP byte-span planning."""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from vllm.model_executor.model_loader.gguf_dsv4_index import (
    GGUF_QUANTIZED_TYPE_NAMES,
    GGUFTensorEntry,
)


class GGUFShardKind(str, Enum):
    REPLICATE = "replicate"
    OUTPUT_ROWS = "output_rows"
    INPUT_BLOCKS = "input_blocks"
    VECTOR = "vector"


@dataclass(frozen=True)
class GGUFTensorClassification:
    source_name: str
    target_name: str
    shard_kind: GGUFShardKind
    stack_after_source: str | None = None


@dataclass(frozen=True)
class GGUFByteSpan:
    source_offset: int
    target_offset: int
    nbytes: int


@dataclass(frozen=True)
class GGUFStridedSpan:
    source_offset: int
    target_offset: int
    nbytes: int
    count: int
    source_stride: int
    target_stride: int


GGUFSpan = GGUFByteSpan | GGUFStridedSpan


@dataclass(frozen=True)
class GGUFTensorLoadPlan:
    source_name: str
    target_name: str
    source_type: str
    source_dims: tuple[int, ...]
    spans: tuple[GGUFSpan, ...]
    target_nbytes: int


_ROOT_RULES = {
    "token_embd.weight": ("model.embed_tokens.weight", GGUFShardKind.OUTPUT_ROWS),
    "output.weight": ("lm_head.weight_raw", GGUFShardKind.OUTPUT_ROWS),
    "output_norm.weight": ("model.norm.weight", GGUFShardKind.REPLICATE),
    "output_hc_fn.weight": ("model.hc_head_fn", GGUFShardKind.REPLICATE),
    "output_hc_base.weight": ("model.hc_head_base", GGUFShardKind.REPLICATE),
    "output_hc_scale.weight": ("model.hc_head_scale", GGUFShardKind.REPLICATE),
}


RuleSpec = tuple[str, GGUFShardKind, str | None]


def _parse_layer_tensor_name(name: str) -> tuple[str, str]:
    parts = name.split(".", 2)
    if len(parts) != 3 or parts[0] != "blk" or not parts[1].isdigit():
        raise ValueError(f"unsupported DeepSeek V4 GGUF tensor {name}")
    return parts[1], parts[2]


def _attention_rules(prefix: str, layer: str) -> dict[str, RuleSpec]:
    fused = f"{prefix}.attn.fused_wqa_wkv.weight_raw"
    return {
        "attn_q_a.weight": (fused, GGUFShardKind.REPLICATE, None),
        "attn_kv.weight": (
            fused,
            GGUFShardKind.REPLICATE,
            f"blk.{layer}.attn_q_a.weight",
        ),
        "attn_q_b.weight": (
            f"{prefix}.attn.wq_b.weight_raw",
            GGUFShardKind.OUTPUT_ROWS,
            None,
        ),
        "attn_output_a.weight": (
            f"{prefix}.attn.wo_a.weight_raw",
            GGUFShardKind.OUTPUT_ROWS,
            None,
        ),
        "attn_output_b.weight": (
            f"{prefix}.attn.wo_b.weight_raw",
            GGUFShardKind.INPUT_BLOCKS,
            None,
        ),
        "attn_q_a_norm.weight": (
            f"{prefix}.attn.q_norm.weight",
            GGUFShardKind.REPLICATE,
            None,
        ),
        "attn_kv_a_norm.weight": (
            f"{prefix}.attn.kv_norm.weight",
            GGUFShardKind.REPLICATE,
            None,
        ),
        "attn_norm.weight": (
            f"{prefix}.attn_norm.weight",
            GGUFShardKind.REPLICATE,
            None,
        ),
        "attn_sinks.weight": (
            f"{prefix}.attn.attn_sink",
            GGUFShardKind.VECTOR,
            None,
        ),
        "indexer.attn_q_b.weight": (
            f"{prefix}.attn.indexer.wq_b.weight",
            GGUFShardKind.REPLICATE,
            None,
        ),
        "indexer.proj.weight": (
            f"{prefix}.attn.indexer.weights_proj.weight",
            GGUFShardKind.REPLICATE,
            None,
        ),
    }


def _ffn_rules(prefix: str, layer: str) -> dict[str, RuleSpec]:
    shared_gate_up = f"{prefix}.ffn.shared_experts.gate_up_proj.weight_raw"
    rules = {
        "ffn_gate_exps.weight": (
            f"{prefix}.ffn.experts.routed_experts.gate_raw",
            GGUFShardKind.OUTPUT_ROWS,
            None,
        ),
        "ffn_up_exps.weight": (
            f"{prefix}.ffn.experts.routed_experts.up_raw",
            GGUFShardKind.OUTPUT_ROWS,
            None,
        ),
        "ffn_down_exps.weight": (
            f"{prefix}.ffn.experts.routed_experts.down_raw",
            GGUFShardKind.INPUT_BLOCKS,
            None,
        ),
        "ffn_gate_shexp.weight": (
            shared_gate_up,
            GGUFShardKind.OUTPUT_ROWS,
            None,
        ),
        "ffn_up_shexp.weight": (
            shared_gate_up,
            GGUFShardKind.OUTPUT_ROWS,
            f"blk.{layer}.ffn_gate_shexp.weight",
        ),
        "ffn_down_shexp.weight": (
            f"{prefix}.ffn.shared_experts.down_proj.weight_raw",
            GGUFShardKind.INPUT_BLOCKS,
            None,
        ),
        "ffn_gate_inp.weight": (
            f"{prefix}.ffn.gate.weight",
            GGUFShardKind.REPLICATE,
            None,
        ),
        "ffn_gate_tid2eid.weight": (
            f"{prefix}.ffn.gate.tid2eid",
            GGUFShardKind.REPLICATE,
            None,
        ),
        "exp_probs_b.bias": (
            f"{prefix}.ffn.gate.e_score_correction_bias",
            GGUFShardKind.REPLICATE,
            None,
        ),
        "ffn_norm.weight": (
            f"{prefix}.ffn_norm.weight",
            GGUFShardKind.REPLICATE,
            None,
        ),
    }
    for component in ("attn", "ffn"):
        for field in ("fn", "base", "scale"):
            rules[f"hc_{component}_{field}.weight"] = (
                f"{prefix}.hc_{component}_{field}",
                GGUFShardKind.REPLICATE,
                None,
            )
    return rules


def _compressor_rules(prefix: str, layer: str) -> dict[str, RuleSpec]:
    rules: dict[str, RuleSpec] = {}
    for source_prefix, target_prefix in {
        "attn_compressor": f"{prefix}.attn.compressor",
        "indexer_compressor": f"{prefix}.attn.indexer.compressor",
    }.items():
        kv_source = f"blk.{layer}.{source_prefix}_kv.weight"
        fused = f"{target_prefix}.fused_wkv_wgate.weight"
        rules[f"{source_prefix}_kv.weight"] = (
            fused,
            GGUFShardKind.REPLICATE,
            None,
        )
        rules[f"{source_prefix}_gate.weight"] = (
            fused,
            GGUFShardKind.REPLICATE,
            kv_source,
        )
        rules[f"{source_prefix}_ape.weight"] = (
            f"{target_prefix}.ape",
            GGUFShardKind.REPLICATE,
            None,
        )
        rules[f"{source_prefix}_norm.weight"] = (
            f"{target_prefix}.norm.weight",
            GGUFShardKind.REPLICATE,
            None,
        )
    return rules


def classify_gguf_dsv4_tensor(name: str) -> GGUFTensorClassification:
    """Map one supported DeepSeek V4 GGUF tensor to its runtime parameter."""
    root_rule = _ROOT_RULES.get(name)
    if root_rule is not None:
        target, shard = root_rule
        return GGUFTensorClassification(name, target, shard)

    layer, suffix = _parse_layer_tensor_name(name)
    prefix = f"model.layers.{layer}"
    rules = _attention_rules(prefix, layer)
    rules.update(_ffn_rules(prefix, layer))
    rules.update(_compressor_rules(prefix, layer))
    try:
        target, shard, stack_after = rules[suffix]
    except KeyError as error:
        raise ValueError(f"unsupported DeepSeek V4 GGUF tensor {name}") from error
    return GGUFTensorClassification(name, target, shard, stack_after)


def classify_profiled_gguf_dsv4_tensor(
    entry: GGUFTensorEntry,
) -> GGUFTensorClassification:
    """Map a profiled quantized source to its format-specific raw parameter."""
    classification = classify_gguf_dsv4_tensor(entry.name)
    if entry.type_name not in GGUF_QUANTIZED_TYPE_NAMES:
        return classification
    source_name = entry.name
    target_name = classification.target_name
    if source_name == "token_embd.weight":
        target_name = "model.embed_tokens.weight_raw"
    elif source_name.endswith("attn_q_a.weight"):
        target_name = f"{target_name}_0"
    elif source_name.endswith("attn_kv.weight"):
        target_name = f"{target_name}_1"
    elif source_name.endswith("ffn_gate_shexp.weight"):
        target_name = f"{target_name}_0"
    elif source_name.endswith("ffn_up_shexp.weight"):
        target_name = f"{target_name}_1"
    elif source_name.endswith(("_compressor_kv.weight", "_compressor_gate.weight")):
        partition = 0 if source_name.endswith("_kv.weight") else 1
        target_name = f"{target_name.removesuffix('.weight')}.weight_raw_{partition}"
    elif target_name.endswith(".weight"):
        target_name = f"{target_name.removesuffix('.weight')}.weight_raw"
    return GGUFTensorClassification(
        source_name=source_name,
        target_name=target_name,
        shard_kind=classification.shard_kind,
    )


def _matrix_row_bytes(entry: GGUFTensorEntry) -> int:
    spec = entry.type_spec
    return math.ceil(entry.dims[0] / spec.block_elements) * spec.block_bytes


def _plan_entry_spans(
    entry: GGUFTensorEntry, shard_kind: GGUFShardKind, tp_rank: int, tp_size: int
) -> tuple[tuple[GGUFSpan, ...], int]:
    if not 0 <= tp_rank < tp_size:
        raise ValueError(f"Invalid TP rank {tp_rank} for size {tp_size}")
    if shard_kind == GGUFShardKind.REPLICATE:
        return (GGUFByteSpan(entry.offset, 0, entry.nbytes),), entry.nbytes
    if shard_kind == GGUFShardKind.VECTOR:
        if len(entry.dims) != 1 or entry.dims[0] % tp_size:
            raise ValueError(f"Cannot TP-shard GGUF vector {entry.name}: {entry.dims}")
        shard_elements = entry.dims[0] // tp_size
        element_bytes = entry.type_spec.block_bytes
        shard_bytes = shard_elements * element_bytes
        return (
            GGUFByteSpan(entry.offset + tp_rank * shard_bytes, 0, shard_bytes),
        ), shard_bytes
    if len(entry.dims) < 2:
        raise ValueError(f"GGUF matrix shard requires rank >=2: {entry.name}")
    row_bytes = _matrix_row_bytes(entry)
    output_rows = entry.dims[1]
    outer_count = math.prod(entry.dims[2:])
    if shard_kind == GGUFShardKind.OUTPUT_ROWS:
        if output_rows % tp_size:
            raise ValueError(
                f"Output rows for {entry.name} are not divisible by TP={tp_size}"
            )
        rows_per_rank = output_rows // tp_size
        span_bytes = rows_per_rank * row_bytes
        source_offset = entry.offset + tp_rank * rows_per_rank * row_bytes
        if outer_count == 1:
            spans: tuple[GGUFSpan, ...] = (GGUFByteSpan(source_offset, 0, span_bytes),)
        else:
            spans = (
                GGUFStridedSpan(
                    source_offset=source_offset,
                    target_offset=0,
                    nbytes=span_bytes,
                    count=outer_count,
                    source_stride=output_rows * row_bytes,
                    target_stride=span_bytes,
                ),
            )
        return spans, outer_count * span_bytes
    if shard_kind == GGUFShardKind.INPUT_BLOCKS:
        spec = entry.type_spec
        if entry.dims[0] % tp_size or (entry.dims[0] // tp_size) % spec.block_elements:
            raise ValueError(f"Input blocks for {entry.name} are not TP/block aligned")
        blocks_per_row = entry.dims[0] // spec.block_elements
        blocks_per_rank = blocks_per_row // tp_size
        span_bytes = blocks_per_rank * spec.block_bytes
        row_count = output_rows * outer_count
        spans = (
            GGUFStridedSpan(
                source_offset=entry.offset + tp_rank * span_bytes,
                target_offset=0,
                nbytes=span_bytes,
                count=row_count,
                source_stride=row_bytes,
                target_stride=span_bytes,
            ),
        )
        return spans, row_count * span_bytes
    raise AssertionError(f"Unhandled GGUF shard kind {shard_kind}")


def build_gguf_dsv4_load_plan(
    entries: tuple[GGUFTensorEntry, ...] | list[GGUFTensorEntry],
    *,
    tp_rank: int,
    tp_size: int,
    profiled_quantization: bool = False,
) -> tuple[GGUFTensorLoadPlan, ...]:
    """Build source/target byte spans for every supported tensor on one TP rank."""
    entries_by_name = {entry.name: entry for entry in entries}
    if len(entries_by_name) != len(entries):
        raise ValueError("GGUF load plan received duplicate tensor names")
    partial: dict[str, tuple[GGUFTensorClassification, tuple[GGUFSpan, ...], int]] = {}
    for entry in entries:
        classification = (
            classify_profiled_gguf_dsv4_tensor(entry)
            if profiled_quantization
            else classify_gguf_dsv4_tensor(entry.name)
        )
        spans, target_nbytes = _plan_entry_spans(
            entry, classification.shard_kind, tp_rank, tp_size
        )
        partial[entry.name] = (classification, spans, target_nbytes)

    plan = []
    for entry in entries:
        classification, spans, target_nbytes = partial[entry.name]
        target_base = 0
        if classification.stack_after_source is not None:
            try:
                previous = partial[classification.stack_after_source]
            except KeyError as error:
                raise ValueError(
                    f"GGUF stacked tensor {entry.name} is missing predecessor "
                    f"{classification.stack_after_source}"
                ) from error
            previous_classification, _, previous_nbytes = previous
            if previous_classification.target_name != classification.target_name:
                raise ValueError(f"GGUF stack target mismatch for {entry.name}")
            target_base = previous_nbytes
        adjusted_spans = tuple(
            GGUFByteSpan(
                span.source_offset, span.target_offset + target_base, span.nbytes
            )
            if isinstance(span, GGUFByteSpan)
            else GGUFStridedSpan(
                source_offset=span.source_offset,
                target_offset=span.target_offset + target_base,
                nbytes=span.nbytes,
                count=span.count,
                source_stride=span.source_stride,
                target_stride=span.target_stride,
            )
            for span in spans
        )
        plan.append(
            GGUFTensorLoadPlan(
                source_name=entry.name,
                target_name=classification.target_name,
                source_type=entry.type_name,
                source_dims=entry.dims,
                spans=adjusted_spans,
                target_nbytes=target_nbytes,
            )
        )
    return tuple(plan)
