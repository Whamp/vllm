# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import shutil
import subprocess
from pathlib import Path

import pytest
import torch

from vllm.v1.ple_offload.nvfp4_native_gather import NvFp4PleNativeGather


@pytest.fixture(autouse=True)
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native NVFP4 PLE gather test")
    assert compiler is not None
    source = Path(__file__).parents[3] / "csrc/cpu/ple_nvfp4_gather.cpp"
    library = tmp_path_factory.mktemp("nvfp4_native") / "libvllm_ple_nvfp4_gather.so"
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-fPIC",
            "-shared",
            str(source),
            "-o",
            str(library),
        ],
        check=True,
    )
    return library


def _python_nvfp4_gather(
    code_shards: tuple[torch.Tensor, ...],
    scale_shards: tuple[torch.Tensor, ...],
    outer_scales: tuple[float, ...],
    row_ids: torch.Tensor,
    rows_per_shard: int,
    width: int,
) -> torch.Tensor:
    magnitudes = torch.tensor(
        [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0],
        dtype=torch.float32,
    )
    lut = torch.cat((magnitudes, -magnitudes))
    output = torch.empty((row_ids.numel(), width), dtype=torch.bfloat16)
    for output_index, global_row in enumerate(row_ids.tolist()):
        shard_index, local_row = divmod(global_row, rows_per_shard)
        packed = code_shards[shard_index][local_row]
        nibbles = torch.stack((packed & 0xF, packed >> 4), dim=-1).reshape(width)
        scales = scale_shards[shard_index][local_row].to(torch.float32)
        output[output_index] = (
            lut[nibbles.long()]
            * scales.repeat_interleave(16)
            * outer_scales[shard_index]
        ).to(torch.bfloat16)
    return output


def _random_nvfp4_shards(
    *,
    generator: torch.Generator,
    shard_count: int,
    rows_per_shard: int,
    width: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[torch.Tensor, ...]]:
    code_shards = tuple(
        torch.randint(
            0,
            256,
            (rows_per_shard, width // 2),
            dtype=torch.uint8,
            generator=generator,
        )
        for _ in range(shard_count)
    )
    scale_shards = tuple(
        torch.randn(
            (rows_per_shard, width // 16),
            dtype=torch.float32,
            generator=generator,
        ).to(torch.float8_e4m3fn)
        for _ in range(shard_count)
    )
    return code_shards, scale_shards


def test_native_nvfp4_ple_gather_preserves_boundaries_and_duplicates(
    native_library: Path,
) -> None:
    generator = torch.Generator().manual_seed(1234)
    shard_count = 128
    rows_per_shard = 7
    width = 160
    code_shards, scale_shards = _random_nvfp4_shards(
        generator=generator,
        shard_count=shard_count,
        rows_per_shard=rows_per_shard,
        width=width,
    )
    outer_scales = tuple(0.125 + shard_index / 256 for shard_index in range(128))
    row_ids = torch.tensor(
        [
            0,
            rows_per_shard - 1,
            rows_per_shard,
            63 * rows_per_shard + 3,
            127 * rows_per_shard,
            shard_count * rows_per_shard - 1,
            rows_per_shard,
            0,
        ],
        dtype=torch.int64,
    )
    expected = _python_nvfp4_gather(
        code_shards,
        scale_shards,
        outer_scales,
        row_ids,
        rows_per_shard,
        width,
    )
    actual = torch.empty_like(expected)
    native = NvFp4PleNativeGather(
        library_path=native_library,
        code_shards=code_shards,
        scale_shards=scale_shards,
        outer_scales=outer_scales,
        rows_per_shard=rows_per_shard,
        width=width,
    )

    assert native.gather_into(row_ids, actual)
    assert torch.equal(actual, expected)


def test_native_nvfp4_ple_gather_covers_all_finite_scale_bits_and_nibbles(
    native_library: Path,
) -> None:
    width = 16
    scale_bits = torch.tensor(
        [bit_pattern for bit_pattern in range(256) if bit_pattern not in (127, 255)],
        dtype=torch.uint8,
    )
    scale_shards = (scale_bits.reshape(-1, 1).view(torch.float8_e4m3fn),)
    packed_nibbles = torch.tensor(
        [0x10, 0x32, 0x54, 0x76, 0x98, 0xBA, 0xDC, 0xFE],
        dtype=torch.uint8,
    )
    code_shards = (packed_nibbles.repeat(scale_bits.numel(), 1),)
    row_ids = torch.arange(scale_bits.numel(), dtype=torch.int64)
    expected = _python_nvfp4_gather(
        code_shards,
        scale_shards,
        (1.0,),
        row_ids,
        scale_bits.numel(),
        width,
    )
    actual = torch.empty_like(expected)
    native = NvFp4PleNativeGather(
        library_path=native_library,
        code_shards=code_shards,
        scale_shards=scale_shards,
        outer_scales=(1.0,),
        rows_per_shard=scale_bits.numel(),
        width=width,
    )

    assert native.gather_into(row_ids, actual)
    assert torch.equal(actual, expected)


@pytest.mark.parametrize(
    ("shard_count", "rows_per_shard", "width", "row_count"),
    [
        (1, 1, 16, 0),
        (2, 3, 32, 1),
        (7, 5, 160, 16),
        (128, 3, 160, 32),
    ],
)
def test_native_nvfp4_ple_gather_matches_production_shapes(
    native_library: Path,
    shard_count: int,
    rows_per_shard: int,
    width: int,
    row_count: int,
) -> None:
    generator = torch.Generator().manual_seed(
        shard_count * 1000 + rows_per_shard * 100 + width + row_count
    )
    code_shards, scale_shards = _random_nvfp4_shards(
        generator=generator,
        shard_count=shard_count,
        rows_per_shard=rows_per_shard,
        width=width,
    )
    outer_scales = tuple(
        0.125 + shard_index / (2 * shard_count) for shard_index in range(shard_count)
    )
    row_ids = torch.randint(
        0,
        shard_count * rows_per_shard,
        (row_count,),
        dtype=torch.int64,
        generator=generator,
    )
    expected = _python_nvfp4_gather(
        code_shards,
        scale_shards,
        outer_scales,
        row_ids,
        rows_per_shard,
        width,
    )
    actual = torch.empty_like(expected)
    native = NvFp4PleNativeGather(
        library_path=native_library,
        code_shards=code_shards,
        scale_shards=scale_shards,
        outer_scales=outer_scales,
        rows_per_shard=rows_per_shard,
        width=width,
    )

    assert native.gather_into(row_ids, actual)
    assert torch.equal(actual, expected)


def test_native_nvfp4_ple_gather_leaves_unsupported_output_to_fallback(
    native_library: Path,
) -> None:
    code_shards = (torch.zeros((1, 8), dtype=torch.uint8),)
    scale_shards = (torch.ones((1, 1), dtype=torch.float8_e4m3fn),)
    native = NvFp4PleNativeGather(
        library_path=native_library,
        code_shards=code_shards,
        scale_shards=scale_shards,
        outer_scales=(1.0,),
        rows_per_shard=1,
        width=16,
    )
    output = torch.full((1, 16), 9.0, dtype=torch.float32)

    assert not native.gather_into(torch.tensor([0]), output)
    assert torch.equal(output, torch.full_like(output, 9.0))


def test_native_nvfp4_ple_gather_rejects_out_of_range_rows(
    native_library: Path,
) -> None:
    native = NvFp4PleNativeGather(
        library_path=native_library,
        code_shards=(torch.zeros((1, 8), dtype=torch.uint8),),
        scale_shards=(torch.ones((1, 1), dtype=torch.float8_e4m3fn),),
        outer_scales=(1.0,),
        rows_per_shard=1,
        width=16,
    )

    with pytest.raises(ValueError, match="outside the NVFP4 PLE table"):
        native.gather_into(
            torch.tensor([1]),
            torch.empty((1, 16), dtype=torch.bfloat16),
        )
