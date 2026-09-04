# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import pytest
from pydantic import ValidationError

from vllm.config.offload import (
    ExpertVMMOffloadConfig,
    OffloadConfig,
    PrefetchOffloadConfig,
    UVAOffloadConfig,
)


def test_expert_vmm_requires_rankings_path():
    with pytest.raises(ValidationError, match="rankings_path is required"):
        OffloadConfig(
            expert_vmm=ExpertVMMOffloadConfig(hot_experts=110),
        )


@pytest.mark.parametrize(
    ("uva", "prefetch"),
    [
        (UVAOffloadConfig(cpu_offload_gb=1), PrefetchOffloadConfig()),
        (
            UVAOffloadConfig(),
            PrefetchOffloadConfig(offload_group_size=2, offload_num_in_group=1),
        ),
    ],
)
def test_expert_vmm_rejects_other_weight_offload_backends(uva, prefetch):
    with pytest.raises(ValidationError, match="cannot be combined"):
        OffloadConfig(
            uva=uva,
            prefetch=prefetch,
            expert_vmm=ExpertVMMOffloadConfig(
                hot_experts=110,
                rankings_path="/rankings.json",
            ),
        )


def test_expert_vmm_configuration_changes_compilation_hash():
    baseline = OffloadConfig()
    configured = OffloadConfig(
        expert_vmm=ExpertVMMOffloadConfig(
            hot_experts=110,
            rankings_path="/rankings.json",
        )
    )

    assert configured.compute_hash() != baseline.compute_hash()
