"""Property-based laws for the DeepSeek V4 GGUF TP load planner.

Generated inventories of supported tensor names search the planner for
defects that a single pinned-artifact example cannot: out-of-bounds
reads, overlapping source spans, byte loss or duplication across TP
ranks, unaligned block/row shards, colliding stacked targets, and
missing fail-closed rejection. The shard-kind expectation table below is
transcribed independently from the TP mapping spec
(.research/gguf-tp-engine/TP-MAPPING.md in club-3090), not from the
production rule tables.
"""

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from vllm.model_executor.model_loader.gguf_dsv4_index import GGUFTensorEntry
from vllm.model_executor.model_loader.gguf_dsv4_plan import (
    GGUFByteSpan,
    GGUFShardKind,
    GGUFStridedSpan,
    build_gguf_dsv4_load_plan,
    classify_gguf_dsv4_tensor,
)

# --- Independent expectation table (spec-transcribed) --------------------

_ROOT_NAMES = {
    "token_embd.weight": GGUFShardKind.OUTPUT_ROWS,
    "output.weight": GGUFShardKind.OUTPUT_ROWS,
    "output_norm.weight": GGUFShardKind.REPLICATE,
}

_OUTPUT_ROWS_SUFFIXES = (
    "attn_q_b.weight",
    "attn_output_a.weight",
    "ffn_gate_exps.weight",
    "ffn_up_exps.weight",
    "ffn_gate_shexp.weight",
    "ffn_up_shexp.weight",
)
_INPUT_BLOCKS_SUFFIXES = (
    "attn_output_b.weight",
    "ffn_down_exps.weight",
    "ffn_down_shexp.weight",
)
_VECTOR_SUFFIXES = ("attn_sinks.weight",)
_REPLICATE_SUFFIXES = (
    "attn_q_a.weight",
    "attn_kv.weight",
    "attn_q_a_norm.weight",
    "attn_kv_a_norm.weight",
    "attn_norm.weight",
    "indexer.attn_q_b.weight",
    "indexer.proj.weight",
    "ffn_gate_inp.weight",
    "ffn_gate_tid2eid.weight",
    "exp_probs_b.bias",
    "ffn_norm.weight",
    "hc_attn_fn.weight",
    "hc_ffn_scale.weight",
    "attn_compressor_kv.weight",
    "attn_compressor_gate.weight",
    "attn_compressor_norm.weight",
)

# Tensors fused into one runtime parameter must appear with their stack
# predecessor or the planner fails closed; generate them as units.
_STACK_UNITS = (
    ("attn_q_a.weight", "attn_kv.weight"),
    ("ffn_gate_shexp.weight", "ffn_up_shexp.weight"),
    ("attn_compressor_kv.weight", "attn_compressor_gate.weight"),
)

# Quantized type_id -> block_elements; block_elements == 1 types cannot
# produce unshardable shapes, so rejection draws use this table only.
_QUANT_TYPE_IDS = (8, 10, 12, 13, 14, 19, 29, 39)
_BLOCK_ELEMENTS = {8: 32, 10: 256, 12: 256, 13: 256, 14: 256, 19: 256, 29: 256, 39: 32}


def _suffix_of(name: str) -> str:
    return name.split(".", 2)[-1]


def _expected_kind(name: str) -> GGUFShardKind:
    if name in _ROOT_NAMES:
        return _ROOT_NAMES[name]
    suffix = _suffix_of(name)
    if suffix in _OUTPUT_ROWS_SUFFIXES:
        return GGUFShardKind.OUTPUT_ROWS
    if suffix in _INPUT_BLOCKS_SUFFIXES:
        return GGUFShardKind.INPUT_BLOCKS
    if suffix in _VECTOR_SUFFIXES:
        return GGUFShardKind.VECTOR
    assert suffix in _REPLICATE_SUFFIXES, name
    return GGUFShardKind.REPLICATE


# --- Generators -----------------------------------------------------------

_TP_SIZES = st.integers(min_value=1, max_value=8)


@st.composite
def _inventories(draw):
    tp_size = draw(_TP_SIZES)
    layer = draw(st.integers(min_value=0, max_value=47))
    prefix = f"blk.{layer}."
    names: list[str] = []

    # One stack unit exercises fused-target handling.
    unit = draw(st.sampled_from(_STACK_UNITS))
    names.extend(f"{prefix}{part}" for part in unit)
    # A routed-expert pair, a vector, and a norm keep multiple kinds
    # co-present in every inventory.
    names.append(f"{prefix}ffn_gate_exps.weight")
    names.append(f"{prefix}ffn_down_exps.weight")
    names.append(f"{prefix}attn_sinks.weight")
    names.append(f"{prefix}ffn_norm.weight")
    if draw(st.booleans()):
        names.append(draw(st.sampled_from(tuple(_ROOT_NAMES))))

    entries = []
    # Real GGUF tensor data regions are disjoint; lay entries out
    # sequentially with random alignment padding so generated inventories
    # preserve that precondition.
    cursor = 512 * draw(st.integers(0, 1_000))
    for name in names:
        kind = _expected_kind(name)
        if kind == GGUFShardKind.VECTOR:
            # attn_sinks ships as F32; one full vector per TP rank.
            vector_entry = GGUFTensorEntry(
                name=name, type_id=0, dims=(tp_size,), offset=cursor
            )
            entries.append(vector_entry)
            cursor += vector_entry.nbytes + 512 * draw(st.integers(0, 4))
            continue
        type_id = draw(
            st.sampled_from(
                _QUANT_TYPE_IDS
                if kind == GGUFShardKind.INPUT_BLOCKS
                else (1,) + _QUANT_TYPE_IDS
            )
        )
        if kind == GGUFShardKind.OUTPUT_ROWS:
            dims = (
                _BLOCK_ELEMENTS.get(type_id, 1),
                tp_size * draw(st.integers(1, 4)),
            )
        elif kind == GGUFShardKind.INPUT_BLOCKS:
            dims = (_BLOCK_ELEMENTS[type_id] * tp_size, draw(st.integers(1, 4)))
        else:
            dims = (draw(st.integers(1, 64)),)
        entry = GGUFTensorEntry(name=name, type_id=type_id, dims=dims, offset=cursor)
        entries.append(entry)
        cursor += entry.nbytes + 512 * draw(st.integers(0, 4))
    return entries, tp_size


# --- Span expansion helpers (test-local, independent of production) -------


def _covered_intervals(plan_entry):
    for span in plan_entry.spans:
        if isinstance(span, GGUFByteSpan):
            yield span.source_offset, span.nbytes
        else:
            for repetition in range(span.count):
                yield (
                    span.source_offset + repetition * span.source_stride,
                    span.nbytes,
                )


def _merge(intervals):
    merged = []
    for start, length in sorted(intervals):
        if merged and start <= merged[-1][0] + merged[-1][1]:
            end = max(merged[-1][0] + merged[-1][1], start + length)
            merged[-1] = (merged[-1][0], end - merged[-1][0])
        else:
            merged.append((start, length))
    return merged


# --- Laws -----------------------------------------------------------------


@settings(max_examples=50, deadline=None)
@given(data=_inventories(), rank_draw=st.integers(0, 7))
def test_plan_partitions_every_tensor_exactly_once(data, rank_draw):
    entries, tp_size = data
    plans_by_rank = [
        build_gguf_dsv4_load_plan(entries, tp_rank=rank, tp_size=tp_size)
        for rank in range(tp_size)
    ]
    plan = plans_by_rank[rank_draw % tp_size]
    assert len(plan) == len(entries)

    for position, entry in enumerate(entries):
        extent_start = entry.offset
        extent_end = entry.offset + entry.nbytes

        for rank_plans in plans_by_rank:
            mine = rank_plans[position]
            assert mine.source_name == entry.name
            for start, length in _covered_intervals(mine):
                assert extent_start <= start
                assert start + length <= extent_end

        covered = []
        for rank_plans in plans_by_rank:
            covered.extend(_covered_intervals(rank_plans[position]))
        relative = sorted((start - extent_start, length) for start, length in covered)
        merged = _merge(relative)
        # Sharded kinds partition the tensor exactly once across ranks;
        # replicated kinds read the full tensor on every rank. Both laws
        # reduce to one merged interval covering [0, nbytes).
        assert merged == [(0, entry.nbytes)]


@settings(max_examples=50, deadline=None)
@given(data=_inventories())
def test_plan_spans_never_overlap_within_a_rank(data):
    entries, tp_size = data
    for tp_rank in range(tp_size):
        plan = build_gguf_dsv4_load_plan(entries, tp_rank=tp_rank, tp_size=tp_size)
        covered = []
        for planned in plan:
            covered.extend(_covered_intervals(planned))
        merged = _merge(covered)
        # Merging must not shrink total coverage; any overlap would.
        assert sum(length for _, length in merged) == sum(
            length for _, length in covered
        )


def _target_range(plan_entry):
    lows, highs = [], []
    for span in plan_entry.spans:
        lows.append(span.target_offset)
        if isinstance(span, GGUFByteSpan):
            highs.append(span.target_offset + span.nbytes)
        else:
            highs.append(
                span.target_offset + (span.count - 1) * span.target_stride + span.nbytes
            )
    return min(lows), max(highs)


@settings(max_examples=50, deadline=None)
@given(data=_inventories())
def test_stacked_targets_are_contiguous_and_disjoint(data):
    entries, tp_size = data
    plan = build_gguf_dsv4_load_plan(entries, tp_rank=0, tp_size=tp_size)
    by_target = {}
    for planned in plan:
        by_target.setdefault(planned.target_name, []).append(planned)
    for grouped in by_target.values():
        ranges = sorted(_target_range(p) for p in grouped)
        cursor = 0
        for start, end in ranges:
            assert start == cursor
            cursor = end


@settings(max_examples=50, deadline=None)
@given(data=_inventories())
def test_block_and_row_shards_are_aligned(data):
    entries, tp_size = data
    plan = build_gguf_dsv4_load_plan(entries, tp_rank=1 % tp_size, tp_size=tp_size)
    for entry, planned in zip(entries, plan):
        kind = _expected_kind(entry.name)
        if kind == GGUFShardKind.INPUT_BLOCKS:
            block_bytes = entry.type_spec.block_bytes
            for span in planned.spans:
                assert isinstance(span, GGUFStridedSpan)
                assert span.nbytes % block_bytes == 0
                assert (span.source_offset - entry.offset) % block_bytes == 0
        if kind == GGUFShardKind.OUTPUT_ROWS:
            blocks_per_row = math.ceil(entry.dims[0] / entry.type_spec.block_elements)
            row_bytes = blocks_per_row * entry.type_spec.block_bytes
            for span in planned.spans:
                assert (span.source_offset - entry.offset) % row_bytes == 0


@settings(max_examples=25, deadline=None)
@given(
    suffix=st.sampled_from(_OUTPUT_ROWS_SUFFIXES + _INPUT_BLOCKS_SUFFIXES),
    type_id=st.sampled_from(_QUANT_TYPE_IDS),
    # Divisibility constraints only bind for TP >= 2; every shape shards
    # trivially at TP=1.
    tp_size=st.integers(min_value=2, max_value=8),
    extra_rows=st.integers(1, 7),
)
def test_plan_rejects_unshardable_shapes(suffix, type_id, tp_size, extra_rows):
    block_elements = _BLOCK_ELEMENTS[type_id]
    if suffix in _OUTPUT_ROWS_SUFFIXES:
        dims = (block_elements, tp_size * extra_rows + 1)
    else:
        dims = (block_elements * tp_size + 1, 2)
    entry = GGUFTensorEntry(
        name=f"blk.0.{suffix}", type_id=type_id, dims=dims, offset=0
    )
    with pytest.raises(ValueError):
        build_gguf_dsv4_load_plan([entry], tp_rank=0, tp_size=tp_size)


@settings(max_examples=25, deadline=None)
@given(layer=st.integers(0, 47), noise=st.integers(0, 5))
def test_classifier_fails_closed_on_unsupported_names(layer, noise):
    bogus = f"blk.{layer}.definitely_not_a_tensor_{noise}.weight"
    with pytest.raises(ValueError):
        classify_gguf_dsv4_tensor(bogus)
