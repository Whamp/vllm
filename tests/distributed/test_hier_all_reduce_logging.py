# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from unittest.mock import Mock

from vllm.distributed.device_communicators import cuda_communicator


def test_hierarchical_all_reduce_is_reported_in_dispatch_order(monkeypatch) -> None:
    communicator = object.__new__(cuda_communicator.CudaCommunicator)
    communicator.world_size = 4
    communicator.unique_name = "tp"
    communicator.pynccl_comm = None
    communicator.qr_comm = None
    communicator.fi_ar_comm = None
    communicator.aiter_ar_comm = None
    communicator.hier_ar_comm = object()
    communicator.ca_comm = None
    communicator.symm_mem_comm = None
    logger = Mock()
    monkeypatch.setattr(cuda_communicator, "logger", logger)
    monkeypatch.setattr(
        cuda_communicator,
        "is_symmetric_memory_enabled",
        lambda: False,
    )

    communicator._log_all_reduce_backend_selection()

    enabled_backends = logger.info_once.call_args.args[1]
    potential_backends = logger.info_once.call_args.args[3]
    assert "'HIERARCHICAL'" in enabled_backends
    assert potential_backends.index("'HIERARCHICAL'") < potential_backends.index(
        "'CUSTOM'"
    )
