# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build a concurrent output fan-out overlay for the promoted Qwen3.8 PLE worker."""

import argparse
import hashlib
import py_compile
from pathlib import Path

EXPECTED_NATIVE_WORKER_SHA256 = (
    "8d6d71de8b56f32e27850ba24dcd194482808e6b7f0dc27f9312d39eb3c0fa32"
)
_OS_IMPORT_OLD = "import multiprocessing.process\n"
_OS_IMPORT_NEW = _OS_IMPORT_OLD + "import os\n"
_COLLECTION_IMPORTS_OLD = """from collections.abc import Iterable
from dataclasses import dataclass
"""
_COLLECTION_IMPORTS_NEW = """from collections.abc import Callable, Iterable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
"""
_RUNNER_LOCAL_OLD = """        zmq_context: zmq.Context | None = None
        pull_socket: zmq.Socket | None = None
        try:
"""
_RUNNER_LOCAL_NEW = """        zmq_context: zmq.Context | None = None
        pull_socket: zmq.Socket | None = None
        runner: PleOffloadRunner | None = None
        try:
"""
_RUNNER_CLOSE_OLD = """        finally:
            if pull_socket is not None:
"""
_RUNNER_CLOSE_NEW = """        finally:
            if runner is not None:
                runner.close()
            if pull_socket is not None:
"""
_RUNNER_INIT_PREFIX = (
    "        # Shared-memory inputs are registered once per DP rank by TP "
    "rank zero.\n"
    "        self._input_bufs: dict[int, PleOffloadInputBuffers] = {}\n"
)
_RUNNER_INIT_OLD = _RUNNER_INIT_PREFIX + "        self._load_weights()\n"
_RUNNER_INIT_NEW = (
    _RUNNER_INIT_PREFIX
    + """        tp_size = self.vllm_config.parallel_config.tensor_parallel_size
        fanout_mode = os.environ.get("VLLM_PLE_CONCURRENT_FANOUT", "0")
        if fanout_mode not in {"0", "1"}:
            raise ValueError(
                "VLLM_PLE_CONCURRENT_FANOUT must be 0 or 1, got "
                f"{fanout_mode!r}"
            )
        self._output_target_executor = (
            ThreadPoolExecutor(
                max_workers=tp_size,
                thread_name_prefix="PleOffloadOutput",
            )
            if fanout_mode == "1" and tp_size > 1
            else None
        )
        if self._output_target_executor is not None:
            logger.info(
                "Concurrent PLE output fan-out enabled for TP=%d.",
                tp_size,
            )
        self._load_weights()
"""
)
_METHODS_ANCHOR = (
    "    def _handle_requests(self, requests: list[PleOffloadRequest]) -> None:\n"
)
_METHODS_NEW = (
    """    def _run_output_target_operations(
        self,
        targets: list[PleOffloadOutputTarget],
        operation: Callable[[PleOffloadOutputTarget], None],
    ) -> None:
        \"\"\"Run one operation per TP target, concurrently when enabled.\"\"\"
        executor = self._output_target_executor
        if executor is None:
            for target in targets:
                operation(target)
            return

        def run_on_target_device(target: PleOffloadOutputTarget) -> None:
            device_index = target.gpu_output_buffer.device.index
            if device_index is None:
                raise RuntimeError("PLE output target has no CUDA device index")
            with torch.accelerator.device_index(device_index):
                operation(target)

        futures = [executor.submit(run_on_target_device, target) for target in targets]
        for future in futures:
            future.result()

    @staticmethod
    def _prepare_output_target_for_write(target: PleOffloadOutputTarget) -> None:
        \"\"\"Wait until a target's prior transfer and GPU use are ordered.\"\"\"
        target.copy_stream.synchronize()
        target.sem.wait_reset(target.copy_stream)

    @staticmethod
    def _copy_result_to_output_target(
        result: torch.Tensor,
        target: PleOffloadOutputTarget,
    ) -> None:
        \"\"\"Enqueue one result copy and signal its target stream.\"\"\"
        slices = tuple(slice(0, size) for size in result.shape)
        with torch.cuda.stream(target.copy_stream):
            target.gpu_output_buffer[slices].copy_(
                result[slices],
                non_blocking=True,
            )
            target.sem.signal(target.copy_stream)

"""
    + _METHODS_ANCHOR
)
_READY_OLD = """                for target in targets:
                    target.copy_stream.synchronize()
                    target.sem.wait_reset(target.copy_stream)
"""
_READY_NEW = """                self._run_output_target_operations(
                    targets,
                    self._prepare_output_target_for_write,
                )
"""
_COPY_OLD = """                slices = tuple(slice(0, size) for size in result.shape)
                for target in targets:
                    with torch.cuda.stream(target.copy_stream):
                        target.gpu_output_buffer[slices].copy_(
                            result[slices], non_blocking=True
                        )
                        target.sem.signal(target.copy_stream)
"""
_COPY_NEW = """                self._run_output_target_operations(
                    targets,
                    partial(self._copy_result_to_output_target, result),
                )

    def close(self) -> None:
        \"\"\"Stop output threads before the offload runner releases resources.\"\"\"
        executor = self._output_target_executor
        if executor is not None:
            executor.shutdown(wait=True)
            self._output_target_executor = None
"""


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _replace_once(source: str, old: str, new: str) -> str:
    if source.count(old) != 1:
        raise RuntimeError("promoted PLE worker patch anchor is not unique")
    return source.replace(old, new)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("native_worker", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    actual_worker_sha = _sha256(args.native_worker)
    if actual_worker_sha != EXPECTED_NATIVE_WORKER_SHA256:
        parser.error(f"promoted PLE worker hash mismatch: {actual_worker_sha}")

    worker_source = args.native_worker.read_text()
    for old, new in (
        (_OS_IMPORT_OLD, _OS_IMPORT_NEW),
        (_COLLECTION_IMPORTS_OLD, _COLLECTION_IMPORTS_NEW),
        (_RUNNER_LOCAL_OLD, _RUNNER_LOCAL_NEW),
        (_RUNNER_CLOSE_OLD, _RUNNER_CLOSE_NEW),
        (_RUNNER_INIT_OLD, _RUNNER_INIT_NEW),
        (_METHODS_ANCHOR, _METHODS_NEW),
        (_READY_OLD, _READY_NEW),
        (_COPY_OLD, _COPY_NEW),
    ):
        worker_source = _replace_once(worker_source, old, new)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_output = args.output_dir / "worker_image_quant.py"
    worker_output.write_text(worker_source)
    py_compile.compile(worker_output, doraise=True)
    (args.output_dir / "SHA256SUMS").write_text(
        f"{_sha256(worker_output)}  {worker_output.name}\n"
    )


if __name__ == "__main__":
    main()
