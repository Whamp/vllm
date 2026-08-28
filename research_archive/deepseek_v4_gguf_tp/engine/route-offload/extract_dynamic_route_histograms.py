#!/usr/bin/env python3
"""Extract rank-consistent workload histograms from GGUF route snapshots."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
from typing import Any

import torch  # ty: ignore[unresolved-import]

JsonObject = dict[str, Any]


def merge_route_histogram_boundary(
    first_trigger_snapshot: torch.Tensor,
    second_trigger_snapshot: torch.Tensor,
) -> torch.Tensor:
    """Merge two layer-0-triggered snapshots into one complete boundary."""
    if first_trigger_snapshot.shape != second_trigger_snapshot.shape:
        raise ValueError("GGUF route boundary snapshot shapes differ")
    if first_trigger_snapshot.ndim != 2 or first_trigger_snapshot.shape[0] < 2:
        raise ValueError("GGUF route boundary snapshots must be [layers, experts]")
    if torch.any(second_trigger_snapshot < first_trigger_snapshot):
        raise ValueError("GGUF route boundary counters moved backwards")
    boundary = first_trigger_snapshot.clone()
    boundary[1:] = second_trigger_snapshot[1:]
    return boundary


def subtract_route_workload(
    start_boundary: torch.Tensor,
    end_boundary: torch.Tensor,
    *,
    top_k: int,
) -> torch.Tensor:
    """Return one complete workload delta with equal token rows per layer."""
    if start_boundary.shape != end_boundary.shape:
        raise ValueError("GGUF route workload boundary shapes differ")
    workload = end_boundary - start_boundary
    if torch.any(workload < 0):
        raise ValueError("GGUF route workload counters moved backwards")
    visits = workload.sum(dim=1)
    if torch.any(visits.remainder(top_k) != 0):
        raise ValueError("GGUF route workload has partial top-k rows")
    token_rows = visits // top_k
    if not torch.all(token_rows == token_rows[0]):
        raise ValueError("GGUF route workload has different token-row totals by layer")
    return workload


def load_rank_route_snapshot(
    snapshot_dir: Path, flush_index: str
) -> tuple[torch.Tensor, JsonObject]:
    """Load one flush index and require byte-semantic agreement across TP ranks."""
    files = sorted(snapshot_dir.glob(f"hist-{flush_index}-*.pt"))
    if len(files) != 4:
        raise ValueError(
            f"GGUF route snapshot {flush_index} expected 4 ranks, found {len(files)}"
        )
    payloads = [
        torch.load(path, weights_only=True, map_location="cpu") for path in files
    ]
    first = payloads[0]
    histogram = first["hist"]
    for path, payload in zip(files[1:], payloads[1:], strict=True):
        if payload["layers"] != first["layers"]:
            raise ValueError(f"GGUF route snapshot layer mismatch in {path.name}")
        if payload["n_experts"] != first["n_experts"]:
            raise ValueError(f"GGUF route snapshot expert mismatch in {path.name}")
        if payload["top_k"] != first["top_k"]:
            raise ValueError(f"GGUF route snapshot top-k mismatch in {path.name}")
        if not torch.equal(histogram, payload["hist"]):
            raise ValueError(f"GGUF route snapshot rank mismatch in {path.name}")
    provenance = {
        "flush_index": flush_index,
        "files": [
            {
                "name": path.name,
                "bytes": path.stat().st_size,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for path in files
        ],
    }
    return histogram, {
        "layers": list(first["layers"]),
        "n_experts": int(first["n_experts"]),
        "top_k": int(first["top_k"]),
        "provenance": provenance,
    }


def load_route_boundary(
    snapshot_dir: Path, first_index: str, second_index: str
) -> tuple[torch.Tensor, JsonObject]:
    """Load and merge one two-trigger workload boundary."""
    first, first_meta = load_rank_route_snapshot(snapshot_dir, first_index)
    second, second_meta = load_rank_route_snapshot(snapshot_dir, second_index)
    for key in ("layers", "n_experts", "top_k"):
        if first_meta[key] != second_meta[key]:
            raise ValueError(f"GGUF route boundary metadata differs for {key}")
    return merge_route_histogram_boundary(first, second), {
        "first": first_meta["provenance"],
        "second": second_meta["provenance"],
        "layers": first_meta["layers"],
        "n_experts": first_meta["n_experts"],
        "top_k": first_meta["top_k"],
    }


def write_route_workload(
    output_path: Path,
    *,
    workload_id: str,
    sessions: int,
    histogram: torch.Tensor,
    metadata: JsonObject,
) -> None:
    """Atomically write one dynamic capture in the route analyzer schema."""
    top_k = int(metadata["top_k"])
    visits = histogram.sum(dim=1)
    token_rows = visits // top_k
    payload = {
        "schema_version": 1,
        "workload_id": workload_id,
        "sessions": sessions,
        "n_experts": histogram.shape[1],
        "top_k": top_k,
        "source": "server60 GGUF-TP activation routing histogram",
        "capture": metadata,
        "layers": [
            {
                "layer": layer,
                "counts": [int(value) for value in histogram[layer].tolist()],
                "token_count": int(token_rows[layer]),
                "transition_count": 0,
            }
            for layer in range(histogram.shape[0])
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_suffix(output_path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n",
        encoding="utf-8",
    )
    os.replace(temporary, output_path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-dir", type=Path, required=True)
    parser.add_argument("--baseline", nargs=2, required=True)
    parser.add_argument("--pilot", nargs=2, required=True)
    parser.add_argument("--corpus", nargs=2, required=True)
    parser.add_argument("--pilot-output", type=Path, required=True)
    parser.add_argument("--corpus-output", type=Path, required=True)
    args = parser.parse_args()

    baseline, baseline_meta = load_route_boundary(args.snapshot_dir, *args.baseline)
    pilot_end, pilot_meta = load_route_boundary(args.snapshot_dir, *args.pilot)
    corpus_end, corpus_meta = load_route_boundary(args.snapshot_dir, *args.corpus)
    if (
        baseline_meta["top_k"] != pilot_meta["top_k"]
        or pilot_meta["top_k"] != corpus_meta["top_k"]
    ):
        raise ValueError("GGUF route workload boundaries have different top-k values")
    top_k = int(baseline_meta["top_k"])
    pilot = subtract_route_workload(baseline, pilot_end, top_k=top_k)
    corpus = subtract_route_workload(pilot_end, corpus_end, top_k=top_k)
    write_route_workload(
        args.pilot_output,
        workload_id="deepswe-pilot-dynamic-routes",
        sessions=1,
        histogram=pilot,
        metadata={
            "top_k": top_k,
            "baseline_boundary": baseline_meta,
            "end_boundary": pilot_meta,
        },
    )
    write_route_workload(
        args.corpus_output,
        workload_id="deepswe-12task-dynamic-routes",
        sessions=12,
        histogram=corpus,
        metadata={
            "top_k": top_k,
            "baseline_boundary": pilot_meta,
            "end_boundary": corpus_meta,
        },
    )
    print(
        json.dumps(
            {
                "pilot_token_rows": int(pilot.sum(dim=1)[0] // top_k),
                "corpus_token_rows": int(corpus.sum(dim=1)[0] // top_k),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
