# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch
from hypothesis import given
from hypothesis import strategies as st

from vllm.v1.ple_offload import cuda_ipc as ple_cuda_ipc
from vllm.v1.ple_offload.cuda_ipc import (
    PleCudaIpcMapping,
    PleCudaIpcTensor,
    export_ple_cuda_ipc_tensor,
    validate_cuda_ipc_tensor_region,
)


@pytest.fixture
def should_do_global_cleanup_after_test() -> bool:
    return False


@given(
    allocation_base=st.integers(min_value=1, max_value=1 << 40),
    allocation_size=st.integers(min_value=1, max_value=1 << 24),
    data=st.data(),
)
def test_cuda_ipc_tensor_region_accepts_every_in_bounds_slice(
    allocation_base: int,
    allocation_size: int,
    data,
) -> None:
    offset = data.draw(st.integers(min_value=0, max_value=allocation_size - 1))
    tensor_nbytes = data.draw(
        st.integers(min_value=1, max_value=allocation_size - offset)
    )

    observed_offset = validate_cuda_ipc_tensor_region(
        allocation_base=allocation_base,
        allocation_size=allocation_size,
        tensor_pointer=allocation_base + offset,
        tensor_nbytes=tensor_nbytes,
    )

    assert observed_offset == offset


@pytest.mark.parametrize(
    ("tensor_pointer", "tensor_nbytes"),
    [(999, 1), (1000, 0), (1064, 1), (1063, 2)],
)
def test_cuda_ipc_tensor_region_rejects_out_of_bounds_slices(
    tensor_pointer: int,
    tensor_nbytes: int,
) -> None:
    with pytest.raises(ValueError, match="outside its CUDA allocation"):
        validate_cuda_ipc_tensor_region(
            allocation_base=1000,
            allocation_size=64,
            tensor_pointer=tensor_pointer,
            tensor_nbytes=tensor_nbytes,
        )


def test_export_cuda_ipc_tensor_records_allocation_offset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = SimpleNamespace(value=0)
    tensor = SimpleNamespace(
        is_cuda=True,
        is_contiguous=lambda: True,
        device=torch.device("cuda", 2),
        data_ptr=lambda: 0x1020,
        nbytes=32,
        shape=(4, 2),
        dtype=torch.float32,
        element_size=lambda: 4,
    )
    handle = SimpleNamespace(reserved=b"h" * 64)
    monkeypatch.setattr(
        ple_cuda_ipc.torch.accelerator,
        "device_index",
        lambda _: nullcontext(),
    )
    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "cuMemGetAddressRange",
        lambda _: (error, 0x1000, 0x100),
    )
    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "cuIpcGetMemHandle",
        lambda _: (error, handle),
    )

    descriptor = export_ple_cuda_ipc_tensor(tensor)

    assert descriptor.device_index == 2
    assert descriptor.allocation_handle == b"h" * 64
    assert descriptor.allocation_size == 0x100
    assert descriptor.tensor_offset == 0x20
    assert descriptor.tensor_nbytes == 32
    assert descriptor.shape == (4, 2)
    assert descriptor.dtype == "torch.float32"
    assert descriptor.element_size == 4


def test_cuda_ipc_mapping_copies_signals_and_closes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    error = SimpleNamespace(value=0)
    opened_handle = SimpleNamespace(reserved=None)
    events: list[tuple] = []
    monkeypatch.setattr(
        ple_cuda_ipc.torch.accelerator,
        "device_index",
        lambda _: nullcontext(),
    )
    initialized_devices: list[int] = []
    monkeypatch.setattr(
        ple_cuda_ipc,
        "_ensure_cuda_context",
        initialized_devices.append,
    )
    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "CUipcMemHandle",
        lambda: opened_handle,
    )
    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "cuIpcOpenMemHandle",
        lambda handle, flags: (error, 0x2000),
    )

    def record_copy(dst, src, size, stream):
        events.append(("copy", int(dst), src, size, int(stream)))
        return (error,)

    def record_wait(stream, ptr, value, flags):
        events.append(("wait", int(stream), int(ptr), value, flags))
        return (error,)

    def record_write(stream, ptr, value, flags):
        events.append(("write", int(stream), int(ptr), value, flags))
        return (error,)

    def record_close(ptr):
        events.append(("close", int(ptr)))
        return (error,)

    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "cuMemcpyHtoDAsync",
        record_copy,
    )
    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "cuStreamWaitValue32",
        record_wait,
    )
    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "cuStreamWriteValue32",
        record_write,
    )
    monkeypatch.setattr(
        ple_cuda_ipc.cuda_driver,
        "cuIpcCloseMemHandle",
        record_close,
    )
    descriptor = PleCudaIpcTensor(
        device_index=0,
        allocation_handle=b"i" * 64,
        allocation_size=256,
        tensor_offset=32,
        tensor_nbytes=64,
        shape=(8, 2),
        dtype="torch.float32",
        element_size=4,
    )
    mapping = PleCudaIpcMapping.open(descriptor)
    stream = SimpleNamespace(cuda_stream=77)
    source = SimpleNamespace(
        device=torch.device("cpu"),
        is_contiguous=lambda: True,
        is_pinned=lambda: True,
        dtype=torch.float32,
        ndim=2,
        shape=(4, 2),
        nbytes=32,
        data_ptr=lambda: 0x3000,
    )

    mapping.copy_from_host(source, stream)
    mapping.wait_value32(0, stream)
    mapping.write_value32(1, stream)
    mapping.close()
    mapping.close()

    assert initialized_devices == [0]
    assert opened_handle.reserved == b"i" * 64
    assert events[0][:2] == ("copy", 0x2020)
    assert events[0][3:] == (source.nbytes, 77)
    assert events[1][0:4] == ("wait", 77, 0x2020, 0)
    assert events[2][0:4] == ("write", 77, 0x2020, 1)
    assert events[3] == ("close", 0x2000)
