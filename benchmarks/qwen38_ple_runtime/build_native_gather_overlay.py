# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Build the native gather overlay for the pinned Qwen3.8 PLE worker."""

import argparse
import hashlib
import os
import py_compile
import shutil
import subprocess
from pathlib import Path

EXPECTED_WORKER_SHA256 = (
    "e85a2d599a422b0b2451f7ad74e408688f53e59e3a77ba17afb4fdffd0bcebad"
)
_WORKER_INIT_OLD = """        self.width = width
        self._lut = None
        logger.info("PLE quant table: %s, %d shards mmapped from %s",
                    self.layout, n_shards, quant_dir)
"""
_WORKER_INIT_NEW = """        self.width = width
        self._lut = None
        self._native_gather = None
        native_library = os.environ.get("VLLM_PLE_NVFP4_GATHER_LIBRARY")
        if native_library and "e2m1" in self.layout:
            from vllm.v1.ple_offload.nvfp4_native_gather import (
                NvFp4PleNativeGather,
            )

            self._native_gather = NvFp4PleNativeGather(
                library_path=native_library,
                code_shards=self._q,
                scale_shards=self._s,
                outer_scales=self._s2,
                rows_per_shard=self.ROWS_PER_SHARD,
                width=self.width,
            )
            logger.info("PLE quant table native NVFP4 gather enabled: %s",
                        native_library)
        logger.info("PLE quant table: %s, %d shards mmapped from %s",
                    self.layout, n_shards, quant_dir)
"""
_WORKER_GATHER_OLD = (
    "    def gather_into(self, ids: torch.Tensor, out: torch.Tensor)"
    " -> None:\n"
    "        ids = ids.long()\n"
    "        shard = ids // self.ROWS_PER_SHARD\n"
)
_WORKER_GATHER_NEW = (
    "    def gather_into(self, ids: torch.Tensor, out: torch.Tensor)"
    " -> None:\n"
    "        ids = ids.long()\n"
    "        if (self._native_gather is not None\n"
    "                and self._native_gather.gather_into(ids, out)):\n"
    "            return\n"
    "        shard = ids // self.ROWS_PER_SHARD\n"
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("worker_image_quant", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    script_dir = Path(__file__).resolve().parent
    repo_root = script_dir.parents[1]
    actual_worker_sha = _sha256(args.worker_image_quant)
    if actual_worker_sha != EXPECTED_WORKER_SHA256:
        parser.error(f"production PLE worker hash mismatch: {actual_worker_sha}")

    compiler = shutil.which(os.environ.get("CXX", "g++"))
    if compiler is None:
        parser.error("C++ compiler not found")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    worker_output = args.output_dir / "worker_image_quant.py"
    helper_output = args.output_dir / "nvfp4_native_gather.py"
    library_output = args.output_dir / "libvllm_ple_nvfp4_gather.so"
    worker_source = args.worker_image_quant.read_text()
    for old, new in (
        (_WORKER_INIT_OLD, _WORKER_INIT_NEW),
        (_WORKER_GATHER_OLD, _WORKER_GATHER_NEW),
    ):
        if worker_source.count(old) != 1:
            raise RuntimeError("production PLE worker patch anchor is not unique")
        worker_source = worker_source.replace(old, new)
    worker_output.write_text(worker_source)
    shutil.copyfile(
        repo_root / "vllm/v1/ple_offload/nvfp4_native_gather.py",
        helper_output,
    )
    subprocess.run(
        [
            compiler,
            "-std=c++17",
            "-O3",
            "-DNDEBUG",
            "-fPIC",
            "-shared",
            repo_root / "csrc/cpu/ple_nvfp4_gather.cpp",
            "-o",
            library_output,
        ],
        check=True,
    )
    py_compile.compile(worker_output, doraise=True)
    py_compile.compile(helper_output, doraise=True)

    artifacts = (worker_output, helper_output, library_output)
    manifest = "".join(f"{_sha256(path)}  {path.name}\n" for path in artifacts)
    (args.output_dir / "SHA256SUMS").write_text(manifest)


if __name__ == "__main__":
    main()
