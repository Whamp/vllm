# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 GGUF loader boundary."""

from pathlib import Path
from typing import Any

import torch.distributed as dist
from torch import nn

from vllm.config import ModelConfig
from vllm.config.load import LoadConfig
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.gguf_dsv4_index import (
    GGUFIndex,
    GGUFTensorEntry,
    parse_gguf_index,
)
from vllm.model_executor.model_loader.gguf_dsv4_io import (
    load_gguf_plan_into_parameter,
    verify_gguf_sha256,
)
from vllm.model_executor.model_loader.gguf_dsv4_plan import (
    GGUFByteSpan,
    GGUFStridedSpan,
    GGUFTensorClassification,
    GGUFTensorLoadPlan,
    build_gguf_dsv4_load_plan,
    classify_gguf_dsv4_tensor,
)

__all__ = [
    "GGUFByteSpan",
    "GGUFIndex",
    "GGUFStridedSpan",
    "GGUFTensorClassification",
    "GGUFTensorEntry",
    "GGUFTensorLoadPlan",
    "build_gguf_dsv4_load_plan",
    "classify_gguf_dsv4_tensor",
    "parse_gguf_index",
    "GGUFDSV4ModelLoader",
]


class GGUFDSV4ModelLoader(BaseModelLoader):
    """Load the one pinned DeepSeek V4 GGUF through exact TP byte plans."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config
        if not isinstance(extra, dict):
            raise ValueError("GGUF DSv4 loader extra config must be a dictionary")
        self.gguf_path = Path(self._required(extra, "gguf_path"))
        self.expected_sha256 = str(self._required(extra, "sha256")).lower()
        if len(self.expected_sha256) != 64 or any(
            character not in "0123456789abcdef" for character in self.expected_sha256
        ):
            raise ValueError("GGUF DSv4 sha256 must be 64 lowercase hex characters")
        self.expected_file_size = int(self._required(extra, "file_size"))
        self.expected_tensor_count = int(self._required(extra, "tensor_count"))
        self.max_source_chunk_bytes = int(
            extra.get("max_source_chunk_bytes", 64 * 1024 * 1024)
        )
        if self.max_source_chunk_bytes <= 0:
            raise ValueError("GGUF DSv4 max_source_chunk_bytes must be positive")
        self._index: GGUFIndex | None = None

    @staticmethod
    def _required(config: dict[str, Any], key: str):
        try:
            return config[key]
        except KeyError as error:
            raise ValueError(f"GGUF DSv4 loader requires extra config {key}") from error

    def _read_and_validate_index(self) -> GGUFIndex:
        if self._index is None:
            index = parse_gguf_index(self.gguf_path)
            if index.file_size != self.expected_file_size:
                raise ValueError(
                    "GGUF file size mismatch: "
                    f"expected {self.expected_file_size}, got {index.file_size}"
                )
            if len(index.tensors) != self.expected_tensor_count:
                raise ValueError(
                    "GGUF tensor count mismatch: "
                    f"expected {self.expected_tensor_count}, got {len(index.tensors)}"
                )
            self._index = index
        return self._index

    def _verify_hash_once(self) -> str:
        if not dist.is_initialized():
            return verify_gguf_sha256(
                self.gguf_path, expected_sha256=self.expected_sha256
            )
        rank = dist.get_rank()
        result: list[tuple[str, str] | None] = [None]
        if rank == 0:
            try:
                digest = verify_gguf_sha256(
                    self.gguf_path, expected_sha256=self.expected_sha256
                )
                result[0] = ("ok", digest)
            except Exception as error:  # propagate exact rank-0 failure
                result[0] = ("error", str(error))
        dist.broadcast_object_list(result, src=0)
        assert result[0] is not None
        status, value = result[0]
        if status != "ok":
            raise ValueError(f"GGUF rank-0 identity verification failed: {value}")
        return value

    def download_model(self, model_config: ModelConfig) -> None:
        del model_config
        self._read_and_validate_index()
        self._verify_hash_once()

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        del model_config
        index = self._read_and_validate_index()
        self._verify_hash_once()
        tp_rank = get_tensor_model_parallel_rank()
        tp_size = get_tensor_model_parallel_world_size()
        plan = build_gguf_dsv4_load_plan(
            index.tensors, tp_rank=tp_rank, tp_size=tp_size
        )
        parameters = dict(model.named_parameters())
        loaded_sources = set()
        for item in plan:
            try:
                parameter = parameters[item.target_name]
            except KeyError as error:
                raise ValueError(
                    f"GGUF target parameter not found: {item.target_name} "
                    f"for source {item.source_name}"
                ) from error
            load_gguf_plan_into_parameter(
                index,
                item,
                parameter,
                max_source_chunk_bytes=self.max_source_chunk_bytes,
            )
            loaded_sources.add(item.source_name)
        expected_sources = {entry.name for entry in index.tensors}
        if loaded_sources != expected_sources:
            missing = sorted(expected_sources - loaded_sources)
            unexpected = sorted(loaded_sources - expected_sources)
            raise ValueError(
                f"GGUF source coverage mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
