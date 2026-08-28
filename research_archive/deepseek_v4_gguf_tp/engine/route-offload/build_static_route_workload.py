#!/usr/bin/env python3
"""Build exact route-skew summaries for GGUF DeepSeek V4 static-routing layers.

The pinned Antirez GGUF stores layers 0–2 as little-endian I32 `tid2eid`
tables with shape [6, 129280]. Given exact rendered token IDs, this tool derives
per-layer expert counts and temporal reuse without model execution. It resets
LRU state at each input-session boundary and emits compact aggregate statistics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from array import array
from collections import OrderedDict
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 1
N_EXPERTS = 256
TOP_K = 6
VOCAB_SIZE = 129_280
LRU_CAPACITIES = (224, 248)


def load_token_ids(path: Path) -> list[int]:
    """Load one rendered token-ID sequence with strict vocabulary validation."""
    token_ids = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(token_ids, list) or not token_ids:
        raise ValueError(f"{path}: token IDs must be a nonempty list")
    if any(
        not isinstance(token_id, int) or token_id < 0 or token_id >= VOCAB_SIZE
        for token_id in token_ids
    ):
        raise ValueError(f"{path}: token ID outside [0, {VOCAB_SIZE})")
    return token_ids


def load_tid2eid_table(path: Path) -> array:
    """Load one little-endian [vocab, top_k] I32 static routing table."""
    raw = path.read_bytes()
    expected_bytes = VOCAB_SIZE * TOP_K * 4
    if len(raw) != expected_bytes:
        raise ValueError(f"{path}: expected {expected_bytes} bytes, got {len(raw)}")
    values = array("i")
    values.frombytes(raw)
    if values.itemsize != 4:
        raise ValueError(f"{path}: host I32 array item size is {values.itemsize}")
    if sys.byteorder != "little":
        values.byteswap()
    if min(values) < 0 or max(values) >= N_EXPERTS:
        raise ValueError(f"{path}: expert ID outside [0, {N_EXPERTS})")
    for token_id in range(VOCAB_SIZE):
        begin = token_id * TOP_K
        if len(set(values[begin : begin + TOP_K])) != TOP_K:
            raise ValueError(f"{path}: token {token_id} repeats a routed expert")
    return values


def summarize_static_layer_routes(
    table: array, token_sessions: list[list[int]]
) -> tuple[list[int], dict[str, Any]]:
    """Summarize one layer, resetting temporal state at each session boundary."""
    counts = [0] * N_EXPERTS
    token_count = 0
    transition_count = 0
    overlap_sum = 0
    exact_repeat_count = 0
    lru_hits = {capacity: 0 for capacity in LRU_CAPACITIES}
    lru_accesses = 0

    for token_ids in token_sessions:
        previous: set[int] | None = None
        lru = {capacity: OrderedDict[int, None]() for capacity in LRU_CAPACITIES}
        for token_id in token_ids:
            begin = token_id * TOP_K
            experts = list(table[begin : begin + TOP_K])
            current = set(experts)
            for expert in experts:
                counts[expert] += 1
            if previous is not None:
                transition_count += 1
                overlap_sum += len(previous & current)
                exact_repeat_count += previous == current
            previous = current
            for capacity, cache in lru.items():
                resident_before = set(cache)
                lru_hits[capacity] += sum(
                    expert in resident_before for expert in experts
                )
                for expert in experts:
                    cache.pop(expert, None)
                    cache[expert] = None
                while len(cache) > capacity:
                    cache.popitem(last=False)
            token_count += 1
            lru_accesses += TOP_K

    if token_count == 0 or transition_count == 0:
        raise ValueError("Static route workload needs at least one multi-token session")
    return counts, {
        "token_count": token_count,
        "transition_count": transition_count,
        "mean_consecutive_overlap": overlap_sum / (transition_count * TOP_K),
        "exact_set_repeat_rate": exact_repeat_count / transition_count,
        "lru_hit_rate": {
            str(capacity): lru_hits[capacity] / lru_accesses
            for capacity in LRU_CAPACITIES
        },
    }


def write_json_atomic(path: Path, value: Any) -> None:
    """Atomically write deterministic UTF-8 JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload-id", required=True)
    parser.add_argument("--token-ids", type=Path, action="append", required=True)
    parser.add_argument("--tid2eid", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if len(args.tid2eid) != 3:
        raise ValueError("Expected exactly three static-routing tid2eid tables")

    token_sessions = [load_token_ids(path) for path in args.token_ids]
    source_sessions = [
        {
            "path": str(path),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "token_count": len(token_ids),
        }
        for path, token_ids in zip(args.token_ids, token_sessions, strict=True)
    ]
    layers = []
    tables = []
    for layer_index, table_path in enumerate(args.tid2eid):
        table = load_tid2eid_table(table_path)
        counts, reuse = summarize_static_layer_routes(table, token_sessions)
        layers.append({"layer": layer_index, "counts": counts, "reuse": reuse})
        tables.append(
            {
                "layer": layer_index,
                "path": str(table_path),
                "sha256": hashlib.sha256(table_path.read_bytes()).hexdigest(),
            }
        )
    workload = {
        "schema_version": SCHEMA_VERSION,
        "workload_id": args.workload_id,
        "n_experts": N_EXPERTS,
        "top_k": TOP_K,
        "route_source": "pinned GGUF tid2eid exact lookup",
        "source_sessions": source_sessions,
        "tid2eid_tables": tables,
        "layers": layers,
    }
    write_json_atomic(args.output, workload)
    print(
        json.dumps(
            {
                "workload_id": args.workload_id,
                "sessions": len(token_sessions),
                "tokens": sum(len(session) for session in token_sessions),
                "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
