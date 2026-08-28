# SPDX-License-Identifier: Apache-2.0
"""Aggregate routed-expert statistics for the GGUF DSv4 path (diagnostic only).

Enabled by setting ``VLLM_GGUF_DSV4_ROUTE_STATS_DIR`` to a writable directory.
When the variable is unset every entry point returns immediately and no
buffers are allocated, so production serving is bit-identical and overhead
free.

When enabled, each routed-expert forward records, using CUDA-graph-safe ops
on persistent device buffers:

* a per-layer visit histogram (one ``[n_experts]`` int64 tensor per layer),
  and, only when ``VLLM_GGUF_DSV4_ROUTE_STATS_RING=1`` is also set,
* a per-layer decode ring (``[RING_SIZE, 4, top_k]`` int32, initialized to
  -1) holding the raw ``topk_ids`` rows of decode-scale forwards
  (``M <= _MAX_DECODE_ROWS``) in arrival order.

Python does not run during CUDA-graph replay, which is why the accumulators
are updated by captured device ops instead of host-side bookkeeping. Flushes
(host copy + atomic snapshot write) only ever happen from eager forward
passes: the histogram every ``_HIST_FLUSH_INTERVAL_S`` seconds and the ring
every ``_RING_FLUSH_INTERVAL_S`` seconds, plus a best-effort final snapshot
at interpreter exit.

The recorded data never feeds back into model math, so numerics and output
determinism are unchanged. Decode overhead when enabled is ~43 tiny scatter
ops plus ~43 ring writes per token (~1% of decode time at 76 tok/s), which is
why this is a time-boxed diagnostic build, not a permanent fixture.
"""

from __future__ import annotations

import atexit
import math
import os
import re
import time
from contextlib import suppress
from pathlib import Path

import torch

_STATS_DIR = os.environ.get("VLLM_GGUF_DSV4_ROUTE_STATS_DIR")
_RING_ENABLED = os.environ.get("VLLM_GGUF_DSV4_ROUTE_STATS_RING") == "1"

RING_SIZE = 8192
"""Decode-ring slots per layer; ~108 s of decode at 76 tok/s per layer."""

_MAX_DECODE_ROWS = 4
"""Forwards with more token rows than this are prefill chunks; they update the
histogram but not the decode ring, keeping the ring a decode-scale sequence.
Production decode forwards have at most max_num_seqs=2 rows."""


def _read_route_stats_interval_seconds(name: str, default: float) -> float:
    """Read one positive finite route-stats flush interval in seconds."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        value = float(raw)
    except ValueError as error:
        raise ValueError(
            f"GGUF DSv4 route stats invalid interval {name}={raw!r}: "
            "expected positive finite seconds"
        ) from error
    if not math.isfinite(value) or value <= 0:
        raise ValueError(
            f"GGUF DSv4 route stats invalid interval {name}={raw!r}: "
            "expected positive finite seconds"
        )
    return value


_HIST_FLUSH_INTERVAL_S = _read_route_stats_interval_seconds(
    "VLLM_GGUF_DSV4_ROUTE_STATS_HIST_FLUSH_SECONDS", 300.0
)
_RING_FLUSH_INTERVAL_S = 1800.0

_LAYER_RE = re.compile(r"layers\.(\d+)\.")


class _LayerStats:
    """Per-layer persistent device state (allocated pre-capture)."""

    __slots__ = ("hist", "ring", "pos")

    def __init__(self, device: torch.device, n_experts: int, top_k: int, ring: bool):
        self.hist = torch.zeros(n_experts, dtype=torch.int64, device=device)
        if ring:
            self.ring = torch.full(
                (RING_SIZE, _MAX_DECODE_ROWS, top_k),
                -1,
                dtype=torch.int32,
                device=device,
            )
            self.pos = torch.zeros((), dtype=torch.int64, device=device)
        else:
            self.ring = None
            self.pos = None


class _State:
    __slots__ = (
        "device",
        "n_experts",
        "top_k",
        "ones",
        "stage",
        "layers",
        "layer_indices",
        "last_hist_flush",
        "last_ring_flush",
        "hist_flush_count",
        "ring_flush_count",
    )

    def __init__(self, device: torch.device, n_experts: int, top_k: int):
        self.device = device
        self.n_experts = n_experts
        self.top_k = top_k
        self.ones = torch.ones(4096, dtype=torch.int64, device=device)
        self.stage = torch.full(
            (1, _MAX_DECODE_ROWS, top_k), -1, dtype=torch.int32, device=device
        )
        self.layers: dict[int, _LayerStats] = {}
        self.layer_indices: dict[int, int] = {}
        now = time.monotonic()
        self.last_hist_flush = now
        self.last_ring_flush = now
        self.hist_flush_count = 0
        self.ring_flush_count = 0


_STATE: _State | None = None


def _layer_index(layer: object, layer_indices: dict[int, int]) -> int:
    key = id(layer)
    index = layer_indices.get(key)
    if index is None:
        layer_name = getattr(layer, "layer_name", None)
        match = _LAYER_RE.search(layer_name) if isinstance(layer_name, str) else None
        if match is None:
            raise ValueError(
                "GGUF DSv4 route stats cannot parse a layer index from "
                f"layer_name={layer_name!r}"
            )
        index = int(match.group(1))
        layer_indices[key] = index
    return index


def _n_experts_of(layer: object) -> int:
    for attr in ("global_num_experts", "num_experts", "n_routed_experts"):
        value = getattr(layer, attr, None)
        if isinstance(value, int) and value > 0:
            return value
    moe_config = getattr(layer, "moe_config", None)
    value = getattr(moe_config, "num_experts", None)
    if isinstance(value, int) and value > 0:
        return value
    raise ValueError(
        "GGUF DSv4 route stats cannot determine the expert count from "
        f"{type(layer).__name__}"
    )


def record_routes(layer: object, topk_ids: torch.Tensor) -> None:
    """Record one routed-expert selection; no-op unless stats are enabled.

    Only issues device ops on persistent buffers (plus throttled eager-pass
    flushes), so it is safe under CUDA-graph capture and replay.
    """
    global _STATE
    if _STATS_DIR is None:
        return
    if topk_ids.numel() == 0:
        return
    ids = topk_ids.reshape(-1, topk_ids.shape[-1])
    rows, top_k = ids.shape
    capturing = torch.cuda.is_current_stream_capturing() if ids.is_cuda else False
    if _STATE is None:
        if capturing:
            raise RuntimeError(
                "GGUF DSv4 route stats first record happened during CUDA-graph "
                "capture; buffers must be allocated during the eager profile run"
            )
        _STATE = _State(ids.device, _n_experts_of(layer), top_k)
    state = _STATE
    index = _layer_index(layer, state.layer_indices)
    stats = state.layers.get(index)
    if stats is None:
        if capturing:
            raise RuntimeError(
                f"GGUF DSv4 route stats first saw layer {index} during "
                "CUDA-graph capture; buffers must be allocated eagerly"
            )
        stats = _LayerStats(state.device, state.n_experts, state.top_k, _RING_ENABLED)
        state.layers[index] = stats
    flat = ids.reshape(-1).to(torch.int64)
    ones = (
        state.ones[: flat.numel()]
        if flat.numel() <= state.ones.numel()
        else torch.ones(flat.numel(), dtype=torch.int64, device=state.device)
    )
    stats.hist.index_add_(0, flat, ones)
    if stats.ring is not None and rows <= _MAX_DECODE_ROWS and top_k == state.top_k:
        state.stage.fill_(-1)
        state.stage[0, :rows].copy_(ids)
        stats.pos.add_(1)
        slot = (stats.pos % RING_SIZE).reshape(1)
        stats.ring.index_copy_(0, slot, state.stage)
    if not capturing:
        maybe_flush()


def _rank_tag() -> str:
    try:
        from vllm.distributed import get_tensor_model_parallel_rank

        return f"tp{get_tensor_model_parallel_rank()}"
    except Exception:
        return os.environ.get("RANK") or os.environ.get("LOCAL_RANK") or "tp?"


def _write_snapshot(state: _State, payload: dict[str, object], stem: str) -> Path:
    out_dir = Path(_STATS_DIR or "")
    out_dir.mkdir(parents=True, exist_ok=True)
    destination = out_dir / f"{stem}-{_rank_tag()}-{os.getpid()}.pt"
    temporary = destination.with_suffix(".pt.tmp")
    torch.save(payload, temporary)
    os.replace(temporary, destination)
    return destination


def maybe_flush(force: bool = False) -> list[Path]:
    """Write due snapshots; returns the paths written (possibly empty)."""
    state = _STATE
    if _STATS_DIR is None or state is None:
        return []
    if state.device.type == "cuda" and torch.cuda.is_current_stream_capturing():
        return []
    now = time.monotonic()
    written: list[Path] = []
    ordered = sorted(state.layers)
    base = {
        "wall_time": time.time(),
        "pid": os.getpid(),
        "n_experts": state.n_experts,
        "top_k": state.top_k,
        "layers": ordered,
    }
    if force or now - state.last_hist_flush >= _HIST_FLUSH_INTERVAL_S:
        state.last_hist_flush = now
        state.hist_flush_count += 1
        payload = {
            **base,
            "hist_flush_index": state.hist_flush_count,
            "hist": torch.stack([state.layers[i].hist.detach().cpu() for i in ordered]),
        }
        written.append(
            _write_snapshot(state, payload, f"hist-{state.hist_flush_count:05d}")
        )
    ring_layers = [i for i in ordered if state.layers[i].ring is not None]
    if ring_layers and (force or now - state.last_ring_flush >= _RING_FLUSH_INTERVAL_S):
        state.last_ring_flush = now
        state.ring_flush_count += 1
        payload = {
            **base,
            "ring_flush_index": state.ring_flush_count,
            "ring": torch.stack(
                [state.layers[i].ring.detach().cpu() for i in ring_layers]
            ),
            "ring_pos": torch.stack(
                [state.layers[i].pos.detach().cpu() for i in ring_layers]
            ),
            "ring_layers": ring_layers,
        }
        written.append(
            _write_snapshot(state, payload, f"ring-{state.ring_flush_count:05d}")
        )
    return written


def _flush_at_exit() -> None:
    with suppress(Exception):
        maybe_flush(force=True)


if _STATS_DIR is not None:
    atexit.register(_flush_at_exit)


def reset_for_tests() -> None:
    """Drop all accumulated state (test isolation only)."""
    global _STATE
    _STATE = None
