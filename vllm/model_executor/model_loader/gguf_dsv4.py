# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 GGUF loader boundary."""

from dataclasses import dataclass
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
from vllm.model_executor.model_loader.gguf_dsv4_profile import (
    GGUFDSV4SourceProfile,
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


@dataclass(frozen=True)
class _GGUFDSV4ShardSpec:
    path: Path
    sha256: str
    file_size: int
    tensor_count: int


class GGUFDSV4ModelLoader(BaseModelLoader):
    """Load an identity-bound DeepSeek V4 GGUF artifact through exact TP plans."""

    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        extra = load_config.model_loader_extra_config
        if not isinstance(extra, dict):
            raise ValueError("GGUF DSv4 loader extra config must be a dictionary")
        self.expected_tensor_count = int(self._required(extra, "tensor_count"))
        if self.expected_tensor_count <= 0:
            raise ValueError("GGUF DSv4 aggregate tensor_count must be positive")
        self.shard_specs = self._parse_shard_specs(extra)
        raw_profile_sha256 = extra.get("source_quant_types_sha256")
        if len(self.shard_specs) > 1 and raw_profile_sha256 is None:
            raise ValueError(
                "GGUF DSv4 split config requires source_quant_types_sha256"
            )
        self.expected_source_profile_sha256 = (
            self._validate_sha256(raw_profile_sha256)
            if raw_profile_sha256 is not None
            else None
        )
        self.max_source_chunk_bytes = int(
            extra.get("max_source_chunk_bytes", 64 * 1024 * 1024)
        )
        if self.max_source_chunk_bytes <= 0:
            raise ValueError("GGUF DSv4 max_source_chunk_bytes must be positive")
        self._indexes: tuple[GGUFIndex, ...] | None = None
        self._verified_sha256: tuple[str, ...] | None = None

    @staticmethod
    def _required(config: dict[str, Any], key: str) -> Any:
        try:
            return config[key]
        except KeyError as error:
            raise ValueError(f"GGUF DSv4 loader requires extra config {key}") from error

    @staticmethod
    def _validate_sha256(value: Any) -> str:
        digest = str(value).lower()
        if len(digest) != 64 or any(
            character not in "0123456789abcdef" for character in digest
        ):
            raise ValueError("GGUF DSv4 sha256 must be 64 lowercase hex characters")
        return digest

    @classmethod
    def _parse_shard_specs(
        cls, extra: dict[str, Any]
    ) -> tuple[_GGUFDSV4ShardSpec, ...]:
        raw_shards = extra.get("gguf_shards")
        if raw_shards is None:
            return (
                _GGUFDSV4ShardSpec(
                    path=Path(cls._required(extra, "gguf_path")),
                    sha256=cls._validate_sha256(cls._required(extra, "sha256")),
                    file_size=int(cls._required(extra, "file_size")),
                    tensor_count=int(cls._required(extra, "tensor_count")),
                ),
            )
        if any(key in extra for key in ("gguf_path", "sha256", "file_size")):
            raise ValueError(
                "GGUF DSv4 split config cannot mix gguf_shards with single-file keys"
            )
        if not isinstance(raw_shards, list) or not raw_shards:
            raise ValueError("GGUF DSv4 gguf_shards must be a non-empty list")
        specs = []
        for shard_number, raw_spec in enumerate(raw_shards):
            if not isinstance(raw_spec, dict):
                raise ValueError(
                    f"GGUF DSv4 shard {shard_number} spec must be a dictionary"
                )
            spec = _GGUFDSV4ShardSpec(
                path=Path(cls._required(raw_spec, "path")),
                sha256=cls._validate_sha256(cls._required(raw_spec, "sha256")),
                file_size=int(cls._required(raw_spec, "file_size")),
                tensor_count=int(cls._required(raw_spec, "tensor_count")),
            )
            if spec.file_size <= 0 or spec.tensor_count < 0:
                raise ValueError(
                    f"GGUF DSv4 shard {shard_number} size/count must be non-negative"
                )
            specs.append(spec)
        paths = {spec.path for spec in specs}
        if len(paths) != len(specs):
            raise ValueError("GGUF DSv4 split config contains duplicate shard paths")
        return tuple(specs)

    def _read_and_validate_indexes(self) -> tuple[GGUFIndex, ...]:
        if self._indexes is not None:
            return self._indexes
        indexes = []
        for shard_number, spec in enumerate(self.shard_specs):
            index = parse_gguf_index(spec.path)
            if index.file_size != spec.file_size:
                raise ValueError(
                    f"GGUF shard {shard_number} file size mismatch: "
                    f"expected {spec.file_size}, got {index.file_size}"
                )
            if len(index.tensors) != spec.tensor_count:
                raise ValueError(
                    f"GGUF shard {shard_number} tensor count mismatch: "
                    f"expected {spec.tensor_count}, got {len(index.tensors)}"
                )
            indexes.append(index)
        self._validate_split_indexes(tuple(indexes))
        entries = tuple(entry for index in indexes for entry in index.tensors)
        if self.expected_source_profile_sha256 is not None:
            profile = GGUFDSV4SourceProfile.from_entries(entries)
            if profile.sha256 != self.expected_source_profile_sha256:
                raise ValueError(
                    "GGUF source quant profile SHA-256 mismatch: "
                    f"expected {self.expected_source_profile_sha256}, "
                    f"computed {profile.sha256}"
                )
        self._indexes = tuple(indexes)
        return self._indexes

    def _validate_split_indexes(self, indexes: tuple[GGUFIndex, ...]) -> None:
        all_names = [entry.name for index in indexes for entry in index.tensors]
        if len(all_names) != self.expected_tensor_count:
            raise ValueError(
                "GGUF aggregate tensor count mismatch: "
                f"expected {self.expected_tensor_count}, got {len(all_names)}"
            )
        if len(set(all_names)) != len(all_names):
            raise ValueError("GGUF split artifact contains duplicate tensor names")
        if len(indexes) == 1:
            return
        for split_number, index in enumerate(indexes):
            metadata = index.metadata
            if metadata.get("split.count") != len(indexes):
                raise ValueError(
                    f"GGUF shard {split_number} split.count does not match shard list"
                )
            if metadata.get("split.no") != split_number:
                raise ValueError(
                    f"GGUF shard {split_number} split.no is not in configured order"
                )
            if metadata.get("split.tensors.count") != self.expected_tensor_count:
                raise ValueError(
                    f"GGUF shard {split_number} split.tensors.count does not match "
                    "the aggregate tensor count"
                )

    def _validate_model_source_profile(self, model_config: ModelConfig) -> None:
        if self.expected_source_profile_sha256 is None:
            return
        model_arch_config = getattr(model_config, "model_arch_config", None)
        quantization_config = getattr(model_arch_config, "quantization_config", None)
        if not isinstance(quantization_config, dict):
            raise ValueError(
                "GGUF DSv4 split artifact requires a model quantization profile"
            )
        model_profile_sha256 = quantization_config.get("source_quant_types_sha256")
        if model_profile_sha256 != self.expected_source_profile_sha256:
            raise ValueError(
                "GGUF DSv4 model/artifact source profile mismatch: "
                f"model={model_profile_sha256!r}, "
                f"artifact={self.expected_source_profile_sha256}"
            )

    def _verify_hashes_once(self) -> tuple[str, ...]:
        if self._verified_sha256 is not None:
            return self._verified_sha256

        def verify_all() -> tuple[str, ...]:
            return tuple(
                verify_gguf_sha256(spec.path, expected_sha256=spec.sha256)
                for spec in self.shard_specs
            )

        if not dist.is_initialized():
            self._verified_sha256 = verify_all()
            return self._verified_sha256
        result: list[dict[str, Any] | None] = [None]
        if dist.get_rank() == 0:
            try:
                result[0] = {"status": "ok", "digests": verify_all()}
            except Exception as error:  # propagate exact rank-0 failure
                result[0] = {"status": "error", "message": str(error)}
        dist.broadcast_object_list(result, src=0)
        assert result[0] is not None
        if result[0].get("status") != "ok":
            raise ValueError(
                "GGUF rank-0 identity verification failed: "
                f"{result[0].get('message', 'unknown error')}"
            )
        digests = result[0].get("digests")
        if not isinstance(digests, tuple) or not all(
            isinstance(digest, str) for digest in digests
        ):
            raise ValueError(
                "GGUF rank-0 identity verification returned invalid digests"
            )
        self._verified_sha256 = digests
        return digests

    def download_model(self, model_config: ModelConfig) -> None:
        self._validate_model_source_profile(model_config)
        self._read_and_validate_indexes()
        self._verify_hashes_once()

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        self._validate_model_source_profile(model_config)
        indexes = self._read_and_validate_indexes()
        self._verify_hashes_once()
        entries = tuple(entry for index in indexes for entry in index.tensors)
        source_indexes = {
            entry.name: index for index in indexes for entry in index.tensors
        }
        plan = build_gguf_dsv4_load_plan(
            entries,
            tp_rank=get_tensor_model_parallel_rank(),
            tp_size=get_tensor_model_parallel_world_size(),
            profiled_quantization=self.expected_source_profile_sha256 is not None,
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
                source_indexes[item.source_name],
                item,
                parameter,
                max_source_chunk_bytes=self.max_source_chunk_bytes,
            )
            loaded_sources.add(item.source_name)
        expected_sources = set(source_indexes)
        if loaded_sources != expected_sources:
            missing = sorted(expected_sources - loaded_sources)
            unexpected = sorted(loaded_sources - expected_sources)
            raise ValueError(
                f"GGUF source coverage mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
