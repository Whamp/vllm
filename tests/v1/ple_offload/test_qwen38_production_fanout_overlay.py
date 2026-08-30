# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import contextlib
import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any, cast

import pytest
import torch


@pytest.fixture(scope="module")
def production_fanout_overlay(
    tmp_path_factory: pytest.TempPathFactory,
) -> ModuleType:
    base_worker = os.environ.get("VLLM_QWEN38_PRODUCTION_PLE_WORKER")
    if base_worker is None:
        pytest.skip("production Qwen3.8 PLE worker source was not provided")
    assert base_worker is not None
    root = Path(__file__).parents[3]
    native_dir = tmp_path_factory.mktemp("qwen38_native_worker")
    native_builder = (
        root / "benchmarks/qwen38_ple_runtime/build_native_gather_overlay.py"
    )
    subprocess.run(
        [sys.executable, native_builder, base_worker, native_dir],
        check=True,
    )
    fanout_dir = tmp_path_factory.mktemp("qwen38_fanout_worker")
    fanout_builder = (
        root / "benchmarks/qwen38_ple_runtime/build_output_fanout_overlay.py"
    )
    subprocess.run(
        [
            sys.executable,
            fanout_builder,
            native_dir / "worker_image_quant.py",
            fanout_dir,
        ],
        check=True,
    )
    worker_path = fanout_dir / "worker_image_quant.py"
    spec = importlib.util.spec_from_file_location(
        "qwen38_production_fanout_overlay",
        worker_path,
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_production_fanout_submits_all_targets_before_waiting(
    production_fanout_overlay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    events: list[tuple[str, int]] = []
    device_contexts: list[int] = []

    def target_device(index: int) -> contextlib.AbstractContextManager[None]:
        device_contexts.append(index)
        return contextlib.nullcontext()

    monkeypatch.setattr(
        production_fanout_overlay.torch.accelerator,
        "device_index",
        target_device,
    )

    class DeferredFuture:
        def __init__(self, operation: Any, target: Any) -> None:
            self._operation = operation
            self._target = target

        def result(self) -> None:
            events.append(("wait", self._target.tp_rank))
            self._operation(self._target)

    class RecordingExecutor:
        def submit(self, operation: Any, target: Any) -> DeferredFuture:
            events.append(("submit", target.tp_rank))
            return DeferredFuture(operation, target)

    runner = production_fanout_overlay.PleOffloadRunner.__new__(
        production_fanout_overlay.PleOffloadRunner
    )
    runner._output_target_executor = cast(Any, RecordingExecutor())
    targets = cast(
        list[Any],
        [
            SimpleNamespace(
                tp_rank=rank,
                gpu_output_buffer=SimpleNamespace(device=SimpleNamespace(index=rank)),
            )
            for rank in range(4)
        ],
    )

    runner._run_output_target_operations(
        targets,
        lambda target: events.append(("operation", target.tp_rank)),
    )

    assert events == [
        ("submit", 0),
        ("submit", 1),
        ("submit", 2),
        ("submit", 3),
        ("wait", 0),
        ("operation", 0),
        ("wait", 1),
        ("operation", 1),
        ("wait", 2),
        ("operation", 2),
        ("wait", 3),
        ("operation", 3),
    ]
    assert device_contexts == [0, 1, 2, 3]


def test_production_fanout_keeps_tp1_serial(
    production_fanout_overlay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device_contexts: list[int] = []
    monkeypatch.setattr(
        production_fanout_overlay.torch.accelerator,
        "device_index",
        lambda index: device_contexts.append(index),
    )
    runner = production_fanout_overlay.PleOffloadRunner.__new__(
        production_fanout_overlay.PleOffloadRunner
    )
    runner._output_target_executor = None
    target = cast(Any, SimpleNamespace(tp_rank=0))
    operations: list[int] = []

    runner._run_output_target_operations(
        [target],
        lambda item: operations.append(item.tp_rank),
    )

    assert operations == [0]
    assert device_contexts == []


@pytest.mark.parametrize(
    ("fanout_mode", "tp_size", "expects_executor"),
    [("0", 4, False), ("1", 1, False), ("1", 4, True)],
)
def test_production_fanout_executor_is_explicit_and_tp_aware(
    production_fanout_overlay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
    fanout_mode: str,
    tp_size: int,
    expects_executor: bool,
) -> None:
    monkeypatch.setenv("VLLM_PLE_CONCURRENT_FANOUT", fanout_mode)
    monkeypatch.setattr(
        production_fanout_overlay.PleOffloadRunner,
        "_load_weights",
        lambda _self: None,
    )
    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=tp_size),
    )

    runner = production_fanout_overlay.PleOffloadRunner(config)

    assert (runner._output_target_executor is not None) is expects_executor
    runner.close()
    assert runner._output_target_executor is None


def test_production_fanout_rejects_invalid_mode(
    production_fanout_overlay: ModuleType,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_PLE_CONCURRENT_FANOUT", "yes")
    config = SimpleNamespace(
        speculative_config=None,
        parallel_config=SimpleNamespace(tensor_parallel_size=4),
    )

    with pytest.raises(
        ValueError,
        match="VLLM_PLE_CONCURRENT_FANOUT must be 0 or 1",
    ):
        production_fanout_overlay.PleOffloadRunner(config)


def test_production_fanout_routes_readiness_and_delivery_through_helper(
    production_fanout_overlay: ModuleType,
) -> None:
    class FakeLayer:
        def forward_impl(
            self,
            _first_ids: torch.Tensor,
            second_ids: torch.Tensor,
            _query_start_loc: torch.Tensor,
            _ngram_context: torch.Tensor,
            *,
            output_buffer: torch.Tensor,
        ) -> torch.Tensor:
            output = output_buffer[: second_ids.numel()]
            output.fill_(23)
            return output

    runner = production_fanout_overlay.PleOffloadRunner.__new__(
        production_fanout_overlay.PleOffloadRunner
    )
    runner._clamp_input_ids = False
    runner._layers = {"ple": FakeLayer()}
    targets = cast(
        list[Any],
        [SimpleNamespace(tp_rank=rank) for rank in range(4)],
    )
    runner._worker_targets = {0: {"ple": targets}}
    runner._input_bufs = {
        0: SimpleNamespace(
            input_ids_buf=torch.tensor([3, 4], dtype=torch.int32),
            query_start_loc_buf=torch.tensor([0, 2], dtype=torch.int32),
            ngram_context_buf=torch.tensor([[1, 2]], dtype=torch.int32),
        )
    }
    runner._pinned_bufs = {0: {"ple": torch.empty((2, 3))}}
    phases: list[tuple[str, int]] = []

    def record_operations(target_list: list[Any], operation: Any) -> None:
        name = (
            operation.__name__
            if hasattr(operation, "__name__")
            else operation.func.__name__
        )
        phases.append((name, len(target_list)))

    runner._run_output_target_operations = record_operations
    runner._handle_requests([SimpleNamespace(dp_rank=0, num_tokens=2, num_reqs=1)])

    assert phases == [
        ("_prepare_output_target_for_write", 4),
        ("_copy_result_to_output_target", 4),
    ]
    assert torch.equal(
        runner._pinned_bufs[0]["ple"],
        torch.full((2, 3), 23.0),
    )


def test_production_fanout_builder_rejects_another_worker(
    tmp_path: Path,
) -> None:
    root = Path(__file__).parents[3]
    wrong_worker = tmp_path / "worker.py"
    wrong_worker.write_text("# another worker\n")
    result = subprocess.run(
        [
            sys.executable,
            root / "benchmarks/qwen38_ple_runtime/build_output_fanout_overlay.py",
            wrong_worker,
            tmp_path / "output",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    assert "promoted PLE worker hash mismatch" in result.stderr
