# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Privilege-free raw CUDA IPC mappings for PLE output transport."""

from dataclasses import dataclass
from typing import Any

import torch
from cuda.bindings import driver as cuda_driver
from cuda.bindings.driver import CUipcMem_flags, CUstreamWaitValue_flags


def _cuda_check(result: Any, operation: str) -> Any:
    """Validate a cuda-python driver result tuple and return it unchanged."""
    error = result[0] if isinstance(result, tuple) else result
    if error.value != 0:
        raise RuntimeError(f"{operation} failed: {error}")
    return result


def validate_cuda_ipc_tensor_region(
    *,
    allocation_base: int,
    allocation_size: int,
    tensor_pointer: int,
    tensor_nbytes: int,
) -> int:
    """Return a tensor's byte offset after validating its allocation bounds."""
    allocation_end = allocation_base + allocation_size
    tensor_end = tensor_pointer + tensor_nbytes
    if (
        allocation_base <= 0
        or allocation_size <= 0
        or tensor_nbytes <= 0
        or tensor_pointer < allocation_base
        or tensor_end > allocation_end
    ):
        raise ValueError("PLE tensor is outside its CUDA allocation")
    return tensor_pointer - allocation_base


@dataclass(frozen=True)
class PleCudaIpcTensor:
    """Serializable CUDA allocation handle and tensor-view metadata."""

    device_index: int
    allocation_handle: bytes
    allocation_size: int
    tensor_offset: int
    tensor_nbytes: int
    shape: tuple[int, ...]
    dtype: str
    element_size: int


def export_ple_cuda_ipc_tensor(tensor: torch.Tensor) -> PleCudaIpcTensor:
    """Export a long-lived contiguous CUDA tensor without PyTorch IPC FDs."""
    if not tensor.is_cuda or not tensor.is_contiguous():
        raise ValueError("PLE CUDA IPC export requires a contiguous CUDA tensor")
    device_index = tensor.device.index
    if device_index is None:
        raise ValueError("PLE CUDA IPC export requires an explicit device index")
    tensor_pointer = tensor.data_ptr()
    with torch.accelerator.device_index(device_index):
        _, allocation_base, allocation_size = _cuda_check(
            cuda_driver.cuMemGetAddressRange(cuda_driver.CUdeviceptr(tensor_pointer)),
            "cuMemGetAddressRange(PLE tensor)",
        )
        tensor_offset = validate_cuda_ipc_tensor_region(
            allocation_base=int(allocation_base),
            allocation_size=int(allocation_size),
            tensor_pointer=tensor_pointer,
            tensor_nbytes=tensor.nbytes,
        )
        _, handle = _cuda_check(
            cuda_driver.cuIpcGetMemHandle(allocation_base),
            "cuIpcGetMemHandle(PLE tensor)",
        )
    return PleCudaIpcTensor(
        device_index=device_index,
        allocation_handle=bytes(handle.reserved),
        allocation_size=int(allocation_size),
        tensor_offset=tensor_offset,
        tensor_nbytes=tensor.nbytes,
        shape=tuple(tensor.shape),
        dtype=str(tensor.dtype),
        element_size=tensor.element_size(),
    )


class PleCudaIpcMapping:
    """Imported CUDA allocation view owned by the PLE offload process."""

    def __init__(
        self,
        descriptor: PleCudaIpcTensor,
        allocation_pointer: int,
    ) -> None:
        self.descriptor = descriptor
        self._allocation_pointer: int | None = allocation_pointer
        self.tensor_pointer = allocation_pointer + descriptor.tensor_offset

    @classmethod
    def open(cls, descriptor: PleCudaIpcTensor) -> "PleCudaIpcMapping":
        """Open a raw CUDA IPC allocation in the descriptor's device context."""
        handle = cuda_driver.CUipcMemHandle()
        handle.reserved = descriptor.allocation_handle
        with torch.accelerator.device_index(descriptor.device_index):
            _, allocation_pointer = _cuda_check(
                cuda_driver.cuIpcOpenMemHandle(
                    handle,
                    CUipcMem_flags.CU_IPC_MEM_LAZY_ENABLE_PEER_ACCESS.value,
                ),
                "cuIpcOpenMemHandle(PLE tensor)",
            )
        return cls(descriptor, int(allocation_pointer))

    def copy_from_host(self, source: torch.Tensor, stream: Any) -> None:
        """Copy a leading tensor slice from pinned CPU storage asynchronously."""
        descriptor = self.descriptor
        if (
            source.device.type != "cpu"
            or not source.is_contiguous()
            or not source.is_pinned()
        ):
            raise ValueError("PLE CUDA IPC copies require contiguous pinned CPU input")
        if str(source.dtype) != descriptor.dtype:
            raise ValueError(
                f"PLE CUDA IPC dtype mismatch: {source.dtype} != {descriptor.dtype}"
            )
        if (
            source.ndim != len(descriptor.shape)
            or tuple(source.shape[1:]) != descriptor.shape[1:]
            or source.shape[0] > descriptor.shape[0]
            or source.nbytes > descriptor.tensor_nbytes
        ):
            raise ValueError(
                f"PLE CUDA IPC shape mismatch: {tuple(source.shape)} does not fit "
                f"{descriptor.shape}"
            )
        _cuda_check(
            cuda_driver.cuMemcpyHtoDAsync(
                cuda_driver.CUdeviceptr(self.tensor_pointer),
                source.data_ptr(),
                source.nbytes,
                cuda_driver.CUstream(stream.cuda_stream),
            ),
            "cuMemcpyHtoDAsync(PLE output)",
        )

    def wait_value32(self, value: int, stream: Any) -> None:
        """Enqueue a stream wait on this mapping's first int32 value."""
        _cuda_check(
            cuda_driver.cuStreamWaitValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self.tensor_pointer),
                value,
                CUstreamWaitValue_flags.CU_STREAM_WAIT_VALUE_EQ.value,
            ),
            "cuStreamWaitValue32(PLE semaphore)",
        )

    def write_value32(self, value: int, stream: Any) -> None:
        """Enqueue a stream write to this mapping's first int32 value."""
        _cuda_check(
            cuda_driver.cuStreamWriteValue32(
                cuda_driver.CUstream(stream.cuda_stream),
                cuda_driver.CUdeviceptr(self.tensor_pointer),
                value,
                0,
            ),
            "cuStreamWriteValue32(PLE semaphore)",
        )

    def close(self) -> None:
        """Close the imported allocation mapping; safe to call more than once."""
        allocation_pointer = self._allocation_pointer
        if allocation_pointer is None:
            return
        with torch.accelerator.device_index(self.descriptor.device_index):
            _cuda_check(
                cuda_driver.cuIpcCloseMemHandle(
                    cuda_driver.CUdeviceptr(allocation_pointer)
                ),
                "cuIpcCloseMemHandle(PLE tensor)",
            )
        self._allocation_pointer = None
