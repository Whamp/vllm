# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import gc
import hashlib
import shutil
import struct
import subprocess
import threading
from pathlib import Path

import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from safetensors.torch import save_file

from vllm.v1.ple_offload.bf16_ple_mmap_gather import (
    Bf16PleMmapGather,
    attach_bf16_ple_mmap_table,
)

_PLE_TENSOR_PREFIX = "model.language_model.layers.1.ple.ple_embedding.ngram_embedding"


@pytest.fixture(autouse=True)
def should_do_global_cleanup_after_test() -> bool:
    return False


@pytest.fixture(scope="module")
def native_library(tmp_path_factory: pytest.TempPathFactory) -> Path:
    compiler = shutil.which("g++")
    if compiler is None:
        pytest.skip("g++ is required for the native BF16 PLE gather test")
    assert compiler is not None
    source = Path(__file__).parents[3] / "csrc/cpu/ple_nvfp4_gather.cpp"
    library = tmp_path_factory.mktemp("bf16_ple_native") / "libvllm_ple_gather.so"
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


def _write_content_addressed_ple(
    root: Path,
    shards: tuple[torch.Tensor, ...],
) -> tuple[Path, str]:
    temporary = root / "ple.safetensors"
    save_file(
        {
            f"{_PLE_TENSOR_PREFIX}.shard_{index}.weight": shard
            for index, shard in enumerate(shards)
        },
        temporary,
    )
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    content_addressed = root / digest
    if content_addressed.exists():
        temporary.unlink()
    else:
        temporary.rename(content_addressed)
    return content_addressed, digest


def test_bf16_ple_mmap_gather_preserves_boundaries_and_duplicates(
    native_library: Path,
    tmp_path: Path,
) -> None:
    shards = tuple(
        (torch.arange(12, dtype=torch.float32).reshape(3, 4) + 100 * index).to(
            torch.bfloat16
        )
        for index in range(3)
    )
    checkpoint_path, digest = _write_content_addressed_ple(tmp_path, shards)
    row_ids = torch.tensor([0, 2, 3, 8, 3, 0], dtype=torch.int64)
    expected = torch.cat(shards)[row_ids]
    output = torch.empty_like(expected)

    table = Bf16PleMmapGather(
        checkpoint_path=checkpoint_path,
        expected_sha256=digest,
        native_library_path=native_library,
        tensor_prefix=_PLE_TENSOR_PREFIX,
        total_rows=9,
        width=4,
    )
    try:
        table.gather_into(row_ids, output)
    finally:
        table.close()

    assert torch.equal(output.view(torch.uint16), expected.view(torch.uint16))


def test_attach_bf16_ple_mmap_table_preserves_worker_gather_seam(
    native_library: Path,
    tmp_path: Path,
) -> None:
    source_rows = torch.arange(24, dtype=torch.float32).reshape(6, 4).to(torch.bfloat16)
    checkpoint_path, digest = _write_content_addressed_ple(
        tmp_path,
        tuple(source_rows.chunk(2)),
    )
    layer = torch.nn.Module()
    layer.ple_embedding = torch.nn.Module()
    layer.ple_embedding.ngram_embedding = torch.nn.Embedding(6, 4, dtype=torch.bfloat16)

    parameter_name = attach_bf16_ple_mmap_table(
        layer_name="language_model.model.layers.1.ple.ple_embedding",
        layer=layer,
        checkpoint_path=checkpoint_path,
        expected_sha256=digest,
        native_library_path=native_library,
    )

    assert parameter_name == "ple_embedding.ngram_embedding.weight"
    assert layer.ple_embedding.ngram_embedding.weight.shape == (0, 4)
    output = torch.empty((3, 4), dtype=torch.bfloat16)
    layer.ple_embedding.ngram_embedding._ple_quant.gather_into(  # type: ignore[attr-defined]
        torch.tensor([5, 0, 3]), output
    )
    assert torch.equal(
        output.view(torch.uint16),
        source_rows[torch.tensor([5, 0, 3])].view(torch.uint16),
    )
    layer.ple_embedding.ngram_embedding._ple_quant.close()  # type: ignore[attr-defined]


def test_bf16_ple_mmap_gather_rejects_wrong_content_identity(
    native_library: Path,
    tmp_path: Path,
) -> None:
    checkpoint_path, _digest = _write_content_addressed_ple(
        tmp_path,
        (torch.zeros((2, 4), dtype=torch.bfloat16),),
    )

    with pytest.raises(ValueError, match="content-addressed filename mismatch"):
        Bf16PleMmapGather(
            checkpoint_path=checkpoint_path,
            expected_sha256="0" * 64,
            native_library_path=native_library,
            tensor_prefix=_PLE_TENSOR_PREFIX,
            total_rows=2,
            width=4,
        )
    gc.collect()


def test_bf16_ple_mmap_gather_rejects_non_bf16_payload(
    native_library: Path,
    tmp_path: Path,
) -> None:
    checkpoint_path, digest = _write_content_addressed_ple(
        tmp_path,
        (torch.zeros((2, 4), dtype=torch.float16),),
    )

    with pytest.raises(ValueError, match="wrong dtype"):
        Bf16PleMmapGather(
            checkpoint_path=checkpoint_path,
            expected_sha256=digest,
            native_library_path=native_library,
            tensor_prefix=_PLE_TENSOR_PREFIX,
            total_rows=2,
            width=4,
        )


def test_bf16_ple_mmap_gather_rejects_missing_shard_index(
    native_library: Path,
    tmp_path: Path,
) -> None:
    temporary = tmp_path / "missing-shard.safetensors"
    save_file(
        {
            f"{_PLE_TENSOR_PREFIX}.shard_0.weight": torch.zeros(
                (2, 4), dtype=torch.bfloat16
            ),
            f"{_PLE_TENSOR_PREFIX}.shard_2.weight": torch.ones(
                (2, 4), dtype=torch.bfloat16
            ),
        },
        temporary,
    )
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    checkpoint_path = temporary.with_name(digest)
    temporary.rename(checkpoint_path)

    with pytest.raises(ValueError, match="contiguous from zero"):
        Bf16PleMmapGather(
            checkpoint_path=checkpoint_path,
            expected_sha256=digest,
            native_library_path=native_library,
            tensor_prefix=_PLE_TENSOR_PREFIX,
            total_rows=4,
            width=4,
        )


def test_bf16_ple_mmap_gather_rejects_trailing_file_bytes(
    native_library: Path,
    tmp_path: Path,
) -> None:
    checkpoint_path, _digest = _write_content_addressed_ple(
        tmp_path,
        (torch.zeros((2, 4), dtype=torch.bfloat16),),
    )
    with checkpoint_path.open("ab") as checkpoint:
        checkpoint.write(b"unexpected")
    digest = hashlib.sha256(checkpoint_path.read_bytes()).hexdigest()
    renamed_path = checkpoint_path.with_name(digest)
    checkpoint_path.rename(renamed_path)

    with pytest.raises(ValueError, match="file size does not match"):
        Bf16PleMmapGather(
            checkpoint_path=renamed_path,
            expected_sha256=digest,
            native_library_path=native_library,
            tensor_prefix=_PLE_TENSOR_PREFIX,
            total_rows=2,
            width=4,
        )


def test_bf16_ple_mmap_gather_rejects_non_object_header(
    native_library: Path,
    tmp_path: Path,
) -> None:
    raw_header = b"[]"
    temporary = tmp_path / "invalid-header.safetensors"
    temporary.write_bytes(struct.pack("<Q", len(raw_header)) + raw_header)
    digest = hashlib.sha256(temporary.read_bytes()).hexdigest()
    checkpoint_path = temporary.with_name(digest)
    temporary.rename(checkpoint_path)

    with pytest.raises(ValueError, match="header root must be an object"):
        Bf16PleMmapGather(
            checkpoint_path=checkpoint_path,
            expected_sha256=digest,
            native_library_path=native_library,
            tensor_prefix=_PLE_TENSOR_PREFIX,
            total_rows=2,
            width=4,
        )


def test_bf16_ple_mmap_gather_rejects_out_of_range_rows_without_partial_output(
    native_library: Path,
    tmp_path: Path,
) -> None:
    checkpoint_path, digest = _write_content_addressed_ple(
        tmp_path,
        (torch.zeros((2, 4), dtype=torch.bfloat16),),
    )
    table = Bf16PleMmapGather(
        checkpoint_path=checkpoint_path,
        expected_sha256=digest,
        native_library_path=native_library,
        tensor_prefix=_PLE_TENSOR_PREFIX,
        total_rows=2,
        width=4,
    )
    output = torch.full((2, 4), 9.0, dtype=torch.bfloat16)

    try:
        with pytest.raises(ValueError, match="outside the BF16 PLE table"):
            table.gather_into(torch.tensor([0, 2]), output)
    finally:
        table.close()

    assert torch.equal(output, torch.full_like(output, 9.0))


def test_bf16_ple_mmap_gather_close_waits_for_inflight_gather(
    native_library: Path,
    tmp_path: Path,
) -> None:
    checkpoint_path, digest = _write_content_addressed_ple(
        tmp_path,
        (torch.zeros((2, 4), dtype=torch.bfloat16),),
    )
    table = Bf16PleMmapGather(
        checkpoint_path=checkpoint_path,
        expected_sha256=digest,
        native_library_path=native_library,
        tensor_prefix=_PLE_TENSOR_PREFIX,
        total_rows=2,
        width=4,
    )
    gather_started = threading.Event()
    release_gather = threading.Event()
    close_finished = threading.Event()
    errors: list[Exception] = []

    def blocking_kernel(*_args: object) -> int:
        gather_started.set()
        if not release_gather.wait(timeout=5):
            raise TimeoutError("test did not release the native gather")
        return 0

    def run_gather() -> None:
        try:
            table.gather_into(
                torch.tensor([0]), torch.empty((1, 4), dtype=torch.bfloat16)
            )
        except Exception as error:
            errors.append(error)

    def run_close() -> None:
        try:
            table.close()
        except Exception as error:
            errors.append(error)
        finally:
            close_finished.set()

    table._kernel = blocking_kernel
    gather_thread = threading.Thread(target=run_gather)
    close_thread = threading.Thread(target=run_close)
    gather_thread.start()
    assert gather_started.wait(timeout=5)
    close_thread.start()
    try:
        assert not close_finished.wait(timeout=0.05)
    finally:
        release_gather.set()
        gather_thread.join(timeout=5)
        close_thread.join(timeout=5)
        table.close()

    assert not gather_thread.is_alive()
    assert not close_thread.is_alive()
    assert errors == []


def test_bf16_ple_mmap_gather_rejects_use_after_close(
    native_library: Path,
    tmp_path: Path,
) -> None:
    checkpoint_path, digest = _write_content_addressed_ple(
        tmp_path,
        (torch.zeros((2, 4), dtype=torch.bfloat16),),
    )
    table = Bf16PleMmapGather(
        checkpoint_path=checkpoint_path,
        expected_sha256=digest,
        native_library_path=native_library,
        tensor_prefix=_PLE_TENSOR_PREFIX,
        total_rows=2,
        width=4,
    )
    output = torch.empty((1, 4), dtype=torch.bfloat16)

    table.close()
    table.close()
    with pytest.raises(RuntimeError, match="BF16 PLE gather is closed"):
        table.gather_into(torch.tensor([0]), output)


@settings(
    max_examples=40,
    deadline=None,
    suppress_health_check=[HealthCheck.function_scoped_fixture],
)
@given(
    shard_count=st.integers(min_value=1, max_value=5),
    rows_per_shard=st.integers(min_value=1, max_value=8),
    width=st.sampled_from([2, 4, 8, 16]),
    values=st.data(),
    row_ids_data=st.data(),
)
def test_bf16_ple_mmap_gather_matches_generated_tables(
    native_library: Path,
    tmp_path: Path,
    shard_count: int,
    rows_per_shard: int,
    width: int,
    values: st.DataObject,
    row_ids_data: st.DataObject,
) -> None:
    element_count = shard_count * rows_per_shard * width
    raw_values = values.draw(
        st.lists(
            st.integers(min_value=-4096, max_value=4096),
            min_size=element_count,
            max_size=element_count,
        )
    )
    all_rows = (
        torch.tensor(raw_values, dtype=torch.float32)
        .reshape(shard_count * rows_per_shard, width)
        .to(torch.bfloat16)
    )
    shards = tuple(all_rows.chunk(shard_count))
    row_ids = torch.tensor(
        row_ids_data.draw(
            st.lists(
                st.integers(
                    min_value=0,
                    max_value=shard_count * rows_per_shard - 1,
                ),
                min_size=0,
                max_size=24,
            )
        ),
        dtype=torch.int64,
    )
    checkpoint_path, digest = _write_content_addressed_ple(tmp_path, shards)
    expected = all_rows[row_ids]
    output = torch.empty_like(expected)

    table = Bf16PleMmapGather(
        checkpoint_path=checkpoint_path,
        expected_sha256=digest,
        native_library_path=native_library,
        tensor_prefix=_PLE_TENSOR_PREFIX,
        total_rows=shard_count * rows_per_shard,
        width=width,
    )
    try:
        table.gather_into(row_ids, output)
    finally:
        table.close()

    assert torch.equal(output.view(torch.uint16), expected.view(torch.uint16))
