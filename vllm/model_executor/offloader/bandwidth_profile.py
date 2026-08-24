# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Measured-bandwidth profiles for hybrid MoE execution planning.

A profile records the two measurements a hybrid expert-cache decision
rides on — host-side expert execution throughput for one expert weight
format, and pinned-host H2D gather bandwidth over one interconnect —
together with the full hardware identity they were measured on.

Identity matching is deliberately strict: a profile is trusted only when
every hardware field the caller supplies matches the measurement
identity. Matching on GPU name alone would silently apply an EPYC box's
CPU-execution bandwidth to a desktop Ryzen and invert the fetch/host
tradeoff, so :func:`profile_matches_hardware` requires all supplied
fields to agree.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, field_validator

PROFILE_SCHEMA_VERSION = 1


class HybridBandwidthProfile(BaseModel):
    """Measured bandwidths backing hybrid expert-cache planning decisions.

    ``host_moe_gbps`` is CPU expert-execution throughput (weight bytes
    consumed per second while running the expert math on the host) for
    ``quant_format``; ``pcie_h2d_gbps`` is pinned-memory device-to-host
    gather bandwidth. Both are positive.
    """

    schema_version: int = PROFILE_SCHEMA_VERSION
    gpu_name: str
    cpu_model: str
    interconnect: str
    quant_format: str
    host_moe_gbps: float = Field(gt=0)
    pcie_h2d_gbps: float = Field(gt=0)

    @field_validator("gpu_name", "cpu_model", "interconnect", "quant_format")
    @classmethod
    def _nonempty_identity(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("profile identity fields must be non-empty")
        return v


def save_bandwidth_profile(profile: HybridBandwidthProfile, path: Path) -> None:
    """Write the JSON profile artifact to ``path``."""
    path.write_text(json.dumps(profile.model_dump(), indent=2) + "\n")


def load_bandwidth_profile(path: Path) -> HybridBandwidthProfile:
    """Read and validate a profile artifact written by
    :func:`save_bandwidth_profile`; raises ``ValueError`` on unknown
    schema versions or malformed fields."""
    try:
        raw = json.loads(path.read_text())
    except (OSError, json.JSONDecodeError) as e:
        raise ValueError(f"unreadable bandwidth profile {path}: {e}") from e
    if raw.get("schema_version") != PROFILE_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported bandwidth profile schema_version "
            f"{raw.get('schema_version')!r} in {path}; expected "
            f"{PROFILE_SCHEMA_VERSION}"
        )
    try:
        return HybridBandwidthProfile.model_validate(raw)
    except Exception as e:
        raise ValueError(f"invalid bandwidth profile in {path}: {e}") from e


def profile_matches_hardware(
    profile: HybridBandwidthProfile,
    *,
    gpu_name: str | None = None,
    cpu_model: str | None = None,
    interconnect: str | None = None,
    quant_format: str | None = None,
) -> bool:
    """Whether the profile was measured on this machine.

    Every supplied field must match the measurement identity exactly;
    ``None`` fields are skipped. Calling with no supplied fields returns
    ``False`` so an empty identity never silently trusts a profile.
    """
    checks = (
        (gpu_name, profile.gpu_name),
        (cpu_model, profile.cpu_model),
        (interconnect, profile.interconnect),
        (quant_format, profile.quant_format),
    )
    supplied = [actual for actual, expected in checks if actual is not None]
    if not supplied:
        return False
    return all(actual == expected for actual, expected in checks if actual is not None)
