# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Native CPU row gathering for mmap-backed NVFP4 PLE tables."""

import ctypes
from collections.abc import Sequence
from pathlib import Path

import torch


class NvFp4PleNativeGather:
    """Bind immutable NVFP4 shard storage to one raw CPU gather kernel."""

    def __init__(
        self,
        *,
        library_path: str | Path,
        code_shards: Sequence[torch.Tensor],
        scale_shards: Sequence[torch.Tensor],
        outer_scales: Sequence[float],
        rows_per_shard: int,
        width: int,
    ) -> None:
        if not code_shards or len(code_shards) != len(scale_shards):
            raise ValueError("NVFP4 PLE code and scale shard counts must match")
        if len(code_shards) != len(outer_scales):
            raise ValueError("NVFP4 PLE outer scales must match the shard count")
        if rows_per_shard <= 0 or width <= 0 or width % 16:
            raise ValueError("NVFP4 PLE geometry requires positive 16-wide blocks")

        code_shape = (rows_per_shard, width // 2)
        scale_shape = (rows_per_shard, width // 16)
        for shard in code_shards:
            self._validate_shard(shard, code_shape, torch.uint8, "code")
        for shard in scale_shards:
            self._validate_shard(
                shard,
                scale_shape,
                torch.float8_e4m3fn,
                "scale",
            )

        self._code_shards = tuple(code_shards)
        self._scale_shards = tuple(scale_shards)
        self._outer_scales = torch.tensor(outer_scales, dtype=torch.float32)
        magnitudes = torch.tensor(
            [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
            dtype=torch.float32,
        )
        self._nvfp4_lut = torch.cat((magnitudes, -magnitudes))
        fp8_bit_patterns = torch.arange(256, dtype=torch.uint8)
        self._fp8_lut = fp8_bit_patterns.view(torch.float8_e4m3fn).to(torch.float32)
        self._rows_per_shard = rows_per_shard
        self._width = width

        pointer_array_type = ctypes.c_void_p * len(code_shards)
        self._code_pointers = pointer_array_type(
            *(shard.data_ptr() for shard in self._code_shards)
        )
        self._scale_pointers = pointer_array_type(
            *(shard.data_ptr() for shard in self._scale_shards)
        )
        self._library = ctypes.CDLL(str(library_path))
        self._kernel = self._library.vllm_gather_nvfp4_ple_rows
        self._kernel.argtypes = [
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_int64,
            ctypes.c_void_p,
            ctypes.c_int64,
            ctypes.c_void_p,
        ]
        self._kernel.restype = ctypes.c_int

    @staticmethod
    def _validate_shard(
        shard: torch.Tensor,
        shape: tuple[int, int],
        dtype: torch.dtype,
        kind: str,
    ) -> None:
        if (
            shard.device.type != "cpu"
            or not shard.is_contiguous()
            or shard.dtype != dtype
            or tuple(shard.shape) != shape
        ):
            raise ValueError(
                f"NVFP4 PLE {kind} shard must be contiguous CPU {dtype} {shape}"
            )

    def gather_into(self, row_ids: torch.Tensor, output: torch.Tensor) -> bool:
        """Gather into BF16 output, or return false for the caller's fallback."""
        if output.dtype != torch.bfloat16:
            return False
        if (
            output.device.type != "cpu"
            or not output.is_contiguous()
            or output.dim() != 2
            or output.shape != (row_ids.numel(), self._width)
        ):
            raise ValueError(
                "native NVFP4 PLE output must be contiguous CPU BF16 with "
                f"shape {(row_ids.numel(), self._width)}"
            )
        row_ids = row_ids.reshape(-1).to(dtype=torch.int64).contiguous()
        if row_ids.numel() == 0:
            return True

        result = self._kernel(
            self._code_pointers,
            self._scale_pointers,
            self._outer_scales.data_ptr(),
            self._nvfp4_lut.data_ptr(),
            self._fp8_lut.data_ptr(),
            len(self._code_shards),
            self._rows_per_shard,
            self._width,
            row_ids.data_ptr(),
            row_ids.numel(),
            output.data_ptr(),
        )
        if result == -2:
            raise ValueError("row ID is outside the NVFP4 PLE table")
        if result != 0:
            raise RuntimeError(f"native NVFP4 PLE gather failed with status {result}")
        return True
