# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Mixed GPU/host VMM placement for AutoGPTQ routed-expert weights."""

from dataclasses import dataclass
from typing import Any

import torch
import triton
import triton.language as tl

from vllm.logger import init_logger
from vllm.model_executor.layers.quantization.utils import replace_parameter
from vllm.utils.expert_vmm import (
    load_expert_rankings,
    plan_expert_permutation,
    plan_vmm_tier_bytes,
)

logger = init_logger(__name__)

_LARGE_EXPERT_WEIGHT_NAMES = ("w13_qweight", "w2_qweight")
_EXPERT_METADATA_NAMES = (
    "w13_scales",
    "w2_scales",
    "w13_qzeros",
    "w2_qzeros",
    "w13_g_idx",
    "w2_g_idx",
    "w13_g_idx_sort_indices",
    "w2_g_idx_sort_indices",
    "w13_bias",
    "w2_bias",
)


@triton.jit
def _permute_expert_rows_kernel(
    source,
    destination,
    new_to_old,
    row_size: tl.constexpr,
    block_size: tl.constexpr,
):
    new_expert_id = tl.program_id(0)
    block_id = tl.program_id(1)
    offsets = block_id * block_size + tl.arange(0, block_size)
    mask = offsets < row_size
    old_expert_id = tl.load(new_to_old + new_expert_id)
    values = tl.load(source + old_expert_id * row_size + offsets, mask=mask)
    tl.store(destination + new_expert_id * row_size + offsets, values, mask=mask)


@dataclass(frozen=True)
class ExpertVMMPlacement:
    """Physical tier accounting for one routed-expert layer."""

    layer_name: str
    hot_experts: int
    device_bytes: int
    host_bytes: int
    metadata_device_bytes: int


@dataclass
class _MixedVMMAllocation:
    """Keep CUDA driver mappings and their PyTorch storage alive."""

    storage: Any
    address: Any
    handles: tuple[Any, ...]
    mapped_bytes: int
    device_bytes: int
    host_bytes: int


_MIXED_VMM_ALLOCATIONS: list[_MixedVMMAllocation] = []


def _cuda_driver_check(result: tuple[Any, ...], operation: str) -> Any:
    from cuda.bindings import driver

    error, *values = result
    if error != driver.CUresult.CUDA_SUCCESS:
        raise RuntimeError(f"{operation} failed: {error}")
    if not values:
        return None
    return values[0] if len(values) == 1 else tuple(values)


def _allocation_property(location_type: Any, location_id: int) -> Any:
    from cuda.bindings import driver

    prop = driver.CUmemAllocationProp()
    prop.type = driver.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    prop.location.type = location_type
    prop.location.id = location_id
    prop.requestedHandleTypes = driver.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_NONE
    return prop


def _allocate_mixed_vmm_tensor(
    source: torch.Tensor,
    new_to_old: torch.Tensor,
    hot_experts: int,
) -> tuple[torch.Tensor, _MixedVMMAllocation]:
    """Copy one expert-major tensor into contiguous device and host mappings."""
    from cuda.bindings import driver

    if not source.is_cuda or not source.is_contiguous() or source.ndim == 0:
        raise ValueError(
            "AutoGPTQ expert VMM requires a contiguous CUDA tensor; "
            f"got device={source.device}, shape={tuple(source.shape)}"
        )

    device_index = source.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _cuda_driver_check(driver.cuInit(0), "cuInit")
    device = _cuda_driver_check(driver.cuDeviceGet(device_index), "cuDeviceGet")
    numa_id = _cuda_driver_check(
        driver.cuDeviceGetAttribute(
            driver.CUdevice_attribute.CU_DEVICE_ATTRIBUTE_HOST_NUMA_ID,
            device,
        ),
        "cuDeviceGetAttribute(HOST_NUMA_ID)",
    )

    device_prop = _allocation_property(
        driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE,
        device_index,
    )
    host_prop = _allocation_property(
        driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA,
        numa_id,
    )
    minimum = driver.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM
    device_granularity = _cuda_driver_check(
        driver.cuMemGetAllocationGranularity(device_prop, minimum),
        "cuMemGetAllocationGranularity(device)",
    )
    host_granularity = _cuda_driver_check(
        driver.cuMemGetAllocationGranularity(host_prop, minimum),
        "cuMemGetAllocationGranularity(host)",
    )
    granularity = max(device_granularity, host_granularity)
    total_bytes = source.numel() * source.element_size()
    row_bytes = source[0].numel() * source.element_size()
    tier_bytes = plan_vmm_tier_bytes(
        total_bytes=total_bytes,
        row_bytes=row_bytes,
        hot_experts=hot_experts,
        granularity=granularity,
    )

    address = _cuda_driver_check(
        driver.cuMemAddressReserve(
            tier_bytes.mapped_bytes,
            granularity,
            0,
            0,
        ),
        "cuMemAddressReserve",
    )
    handles: list[Any] = []
    if tier_bytes.device_bytes:
        device_handle = _cuda_driver_check(
            driver.cuMemCreate(tier_bytes.device_bytes, device_prop, 0),
            "cuMemCreate(device)",
        )
        handles.append(device_handle)
        _cuda_driver_check(
            driver.cuMemMap(
                address,
                tier_bytes.device_bytes,
                0,
                device_handle,
                0,
            ),
            "cuMemMap(device)",
        )
    if tier_bytes.host_bytes:
        host_handle = _cuda_driver_check(
            driver.cuMemCreate(tier_bytes.host_bytes, host_prop, 0),
            "cuMemCreate(host)",
        )
        handles.append(host_handle)
        _cuda_driver_check(
            driver.cuMemMap(
                int(address) + tier_bytes.device_bytes,
                tier_bytes.host_bytes,
                0,
                host_handle,
                0,
            ),
            "cuMemMap(host)",
        )

    access = driver.CUmemAccessDesc()
    access.location.type = driver.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
    access.location.id = device_index
    access.flags = driver.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
    _cuda_driver_check(
        driver.cuMemSetAccess(
            address,
            tier_bytes.mapped_bytes,
            [access],
            1,
        ),
        "cuMemSetAccess",
    )

    storage = torch._C._construct_storage_from_data_pointer(
        int(address), source.device, tier_bytes.mapped_bytes
    )
    metadata = {
        "nbytes": tier_bytes.mapped_bytes,
        "data_ptr": int(address),
        "size": tuple(source.shape),
        "stride": tuple(source.stride()),
        "dtype": source.dtype,
        "device": source.device,
        "storage_offset": 0,
    }
    destination = torch._C._construct_CUDA_Tensor_From_Storage_And_Metadata(
        metadata, storage
    )
    row_size = source[0].numel()
    block_size = 1024
    grid = (source.shape[0], triton.cdiv(row_size, block_size))
    _permute_expert_rows_kernel[grid](
        source,
        destination,
        new_to_old,
        row_size=row_size,
        block_size=block_size,
        num_warps=8,
    )
    torch.cuda.synchronize(source.device)

    allocation = _MixedVMMAllocation(
        storage=storage,
        address=address,
        handles=tuple(handles),
        mapped_bytes=tier_bytes.mapped_bytes,
        device_bytes=tier_bytes.device_bytes,
        host_bytes=tier_bytes.host_bytes,
    )
    _MIXED_VMM_ALLOCATIONS.append(allocation)
    return destination, allocation


class AutoGPTQExpertVMM:
    """Place ranked AutoGPTQ experts in GPU pages and the remainder in RAM."""

    def __init__(self, *, hot_experts: int, rankings_path: str):
        if hot_experts < 0:
            raise ValueError("hot_experts must be non-negative")
        self.hot_experts = hot_experts
        self.rankings = load_expert_rankings(rankings_path)

    def place_experts(
        self,
        layer: torch.nn.Module,
        layer_name: str,
    ) -> ExpertVMMPlacement | None:
        """Apply placement when the exact layer prefix exists in the rankings."""
        if not layer_name:
            raise RuntimeError(
                "AutoGPTQ expert VMM requires the routed-expert layer prefix"
            )
        ranked_global_ids = self.rankings.get(layer_name)
        if ranked_global_ids is None:
            return None

        w13 = layer.w13_qweight
        w2 = layer.w2_qweight
        if w13.shape[0] != w2.shape[0]:
            raise ValueError(
                f"AutoGPTQ expert tensors disagree on expert count for {layer_name}"
            )
        local_experts = w13.shape[0]
        if self.hot_experts >= local_experts:
            logger.info(
                "AutoGPTQ expert VMM skipped for %s: hot_experts=%d, local_experts=%d",
                layer_name,
                self.hot_experts,
                local_experts,
            )
            return None

        original_map = layer.expert_map
        if original_map is None:
            original_map = torch.arange(
                layer.global_num_experts,
                dtype=torch.int32,
                device=w13.device,
            )
        original_map_cpu = tuple(int(value) for value in original_map.cpu().tolist())
        permutation = plan_expert_permutation(
            ranked_global_ids,
            original_map_cpu,
            self.hot_experts,
        )
        new_to_old = torch.tensor(
            permutation.new_to_old,
            dtype=torch.long,
            device=w13.device,
        )

        device_bytes = 0
        host_bytes = 0
        metadata_device_bytes = 0
        with torch.no_grad():
            for name in _LARGE_EXPERT_WEIGHT_NAMES:
                source = getattr(layer, name)
                destination, allocation = _allocate_mixed_vmm_tensor(
                    source,
                    new_to_old,
                    len(permutation.hot_local_ids),
                )
                parameter = torch.nn.Parameter(destination, requires_grad=False)
                # replace_parameter intentionally reuses equal-sized storage,
                # which would copy this mixed VMM tensor back into the original
                # all-device allocation and defeat the placement.
                layer.register_parameter(name, parameter)
                if name == "w13_qweight":
                    layer.w13_weight = parameter
                else:
                    layer.w2_weight = parameter
                device_bytes += allocation.device_bytes
                host_bytes += allocation.host_bytes

            for name in _EXPERT_METADATA_NAMES:
                source = getattr(layer, name, None)
                if (
                    source is None
                    or source.ndim == 0
                    or source.shape[0] != local_experts
                ):
                    continue
                destination = torch.index_select(source, 0, new_to_old).contiguous()
                parameter = torch.nn.Parameter(destination, requires_grad=False)
                replace_parameter(layer, name, parameter)
                metadata_device_bytes += (
                    destination.numel() * destination.element_size()
                )

            new_expert_map = torch.tensor(
                permutation.expert_map,
                dtype=original_map.dtype,
                device=original_map.device,
            )
            layer._expert_map = new_expert_map
            layer.expert_map_manager._expert_map = new_expert_map

        del w13, w2
        torch.cuda.empty_cache()
        placement = ExpertVMMPlacement(
            layer_name=layer_name,
            hot_experts=len(permutation.hot_local_ids),
            device_bytes=device_bytes,
            host_bytes=host_bytes,
            metadata_device_bytes=metadata_device_bytes,
        )
        logger.info(
            "AutoGPTQ expert VMM: layer=%s hot=%d device=%.2f MiB "
            "host=%.2f MiB metadata_device=%.2f MiB",
            placement.layer_name,
            placement.hot_experts,
            placement.device_bytes / 1024**2,
            placement.host_bytes / 1024**2,
            placement.metadata_device_bytes / 1024**2,
        )
        return placement
