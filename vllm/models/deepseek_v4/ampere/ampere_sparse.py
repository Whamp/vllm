# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""DeepSeek V4 sparse MLA attention for SM8x (Ampere: A100/A800).

Reuses the ROCm Triton sparse-MLA implementation wholesale: its kernels,
ragged metadata builders, and bf16 o_proj reference path are plain
Triton/torch (the aiter-only preshuffle GEMMs self-disable off ROCm), and
``vllm.v1.attention.ops.fp8_sm80`` supplies e4m3 encode/decode below SM89
where Triton refuses native fp8 converts.
"""

import importlib
from collections.abc import Callable
from typing import ClassVar

import torch

import vllm.envs as envs
from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV4ROCMAiterMLASparseMetadata,
    DeepseekV4ROCMAiterSparseSWAMetadata,
)
from vllm.models.deepseek_v4.cache_layout import get_deepseek_v4_cache_layout
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)
from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)

FlashMLADecode = Callable[..., torch.Tensor]
FlashMLAPrefill = Callable[..., torch.Tensor]


def load_ampere_flash_mla_decode(cache_dtype: str) -> FlashMLADecode:
    """Load the native operator for one DS-MLA cache format or fail closed."""
    try:
        flash_mla = importlib.import_module("flash_mla")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DeepSeek V4 FlashMLA decode was enabled, but flash_mla is not installed"
        ) from error
    operator_name = {
        "fp8_ds_mla": "sparse_mla_decode_fp8",
        "fp4_ds_mla": "sparse_mla_decode_fp4",
    }.get(cache_dtype)
    if operator_name is None:
        raise ValueError(
            "DeepSeek V4 native FlashMLA decode supports fp8_ds_mla or "
            f"fp4_ds_mla, got {cache_dtype}"
        )
    decode = getattr(flash_mla, operator_name, None)
    if decode is None:
        raise RuntimeError(
            "DeepSeek V4 FlashMLA decode was enabled, but flash_mla does not "
            f"export {operator_name}"
        )
    return decode


def load_ampere_flash_mla_prefill(cache_dtype: str) -> FlashMLAPrefill:
    """Load native FP4 DS-MLA prefill or fail closed."""
    if cache_dtype != "fp4_ds_mla":
        raise ValueError(
            "DeepSeek V4 native FlashMLA prefill supports fp4_ds_mla, got "
            f"{cache_dtype}"
        )
    try:
        flash_mla = importlib.import_module("flash_mla")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DeepSeek V4 FlashMLA prefill was enabled, but flash_mla is not installed"
        ) from error
    prefill = getattr(flash_mla, "sparse_mla_prefill_fp4", None)
    if prefill is None:
        raise RuntimeError(
            "DeepSeek V4 FlashMLA prefill was enabled, but flash_mla does not "
            "export sparse_mla_prefill_fp4"
        )
    return prefill


class DeepseekV4AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = [
        "auto",
        "fp8_ds_mla",
        "fp4_ds_mla",
        "fp8",
    ]

    @staticmethod
    def get_name() -> str:
        return "TRITON_MLA_SPARSE_DSV4"

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major == 8


class DeepseekV4AmpereMLAAttention(DeepseekV4ROCMAiterMLAAttention):
    """SM8x DeepSeek V4 attention with optional native FlashMLA decode."""

    backend_cls = DeepseekV4AmpereMLASparseBackend

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.kv_cache_dtype == "fp4_ds_mla" and not envs.VLLM_DSV4_FLASH_MLA_DECODE:
            raise ValueError(
                "fp4_ds_mla requires native AppMana FlashMLA decode; set "
                "VLLM_DSV4_FLASH_MLA_DECODE=1"
            )
        self._flash_mla_decode = (
            load_ampere_flash_mla_decode(self.kv_cache_dtype)
            if envs.VLLM_DSV4_FLASH_MLA_DECODE
            else None
        )
        self._flash_mla_prefill = (
            load_ampere_flash_mla_prefill(self.kv_cache_dtype)
            if self.kv_cache_dtype == "fp4_ds_mla" and envs.VLLM_DSV4_FLASH_MLA_DECODE
            else None
        )
        if self._flash_mla_decode is not None:
            logger.info_once(
                "Using native Ampere FlashMLA sparse decode for %s%s.",
                self.kv_cache_dtype,
                " and native sparse prefill" if self._flash_mla_prefill else "",
            )

    def _uses_native_sparse_prefill(self) -> bool:
        return self._flash_mla_prefill is not None

    def _forward_prefill(
        self,
        q: torch.Tensor,
        positions: torch.Tensor,
        compressed_k_cache: torch.Tensor | None,
        swa_k_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
    ) -> None:
        flash_mla_prefill = self._flash_mla_prefill
        if flash_mla_prefill is None:
            super()._forward_prefill(
                q=q,
                positions=positions,
                compressed_k_cache=compressed_k_cache,
                swa_k_cache=swa_k_cache,
                output=output,
                attn_metadata=attn_metadata,
                swa_metadata=swa_metadata,
            )
            return

        assert self.kv_cache_dtype == "fp4_ds_mla"
        assert swa_metadata.prefill_swa_indices is not None
        assert swa_metadata.prefill_swa_lens is not None
        num_prefill_tokens = swa_metadata.num_prefill_tokens
        num_decode_tokens = swa_metadata.num_decode_tokens
        swa_indices = swa_metadata.prefill_swa_indices[:num_prefill_tokens]
        swa_lens = swa_metadata.prefill_swa_lens[:num_prefill_tokens]

        extra_indices = None
        extra_lens = None
        if attn_metadata is not None:
            if compressed_k_cache is None:
                raise RuntimeError("compressed FP4 cache is missing for sparse prefill")
            if self.compress_ratio == 4:
                assert self.topk_indices_buffer is not None
                local_topk = self.topk_indices_buffer[
                    num_decode_tokens : num_decode_tokens + num_prefill_tokens
                ]
            else:
                local_topk = attn_metadata.c128a_prefill_topk_indices
            if local_topk is None:
                raise RuntimeError("compressed FP4 prefill indices are missing")
            assert swa_metadata.is_valid_token is not None
            token_to_req_indices = swa_metadata.token_to_req_indices
            if token_to_req_indices is None:
                raise RuntimeError("FP4 prefill request indices are missing")
            token_slice = slice(
                num_decode_tokens, num_decode_tokens + num_prefill_tokens
            )
            compressed_block_size = attn_metadata.block_size // self.compress_ratio
            extra_indices, extra_lens = compute_global_topk_indices_and_lens(
                local_topk,
                token_to_req_indices[token_slice],
                attn_metadata.block_table,
                compressed_block_size,
                swa_metadata.is_valid_token[token_slice],
                output_buffers=self._global_topk_output_buffers(local_topk),
            )

        result = flash_mla_prefill(
            q=q,
            swa_cache=swa_k_cache,
            swa_indices=swa_indices,
            swa_lens=swa_lens,
            scale=self.scale,
            attn_sink=self.attn_sink,
            extra_cache=compressed_k_cache,
            extra_indices=extra_indices,
            extra_lens=extra_lens,
        )
        output.copy_(result)

    def _forward_decode(
        self,
        q: torch.Tensor,
        kv_cache: torch.Tensor | None,
        swa_metadata: DeepseekV4ROCMAiterSparseSWAMetadata,
        attn_metadata: DeepseekV4ROCMAiterMLASparseMetadata | None,
        swa_only: bool,
        output: torch.Tensor,
    ) -> None:
        flash_mla_decode = self._flash_mla_decode
        if flash_mla_decode is None:
            super()._forward_decode(
                q=q,
                kv_cache=kv_cache,
                swa_metadata=swa_metadata,
                attn_metadata=attn_metadata,
                swa_only=swa_only,
                output=output,
            )
            return
        if self.kv_cache_dtype not in ("fp8_ds_mla", "fp4_ds_mla"):
            raise ValueError(
                "DeepSeek V4 native FlashMLA decode requires fp8_ds_mla or "
                f"fp4_ds_mla, got {self.kv_cache_dtype}"
            )

        num_decodes = swa_metadata.num_decodes
        num_decode_tokens = swa_metadata.num_decode_tokens
        topk_indices = None
        topk_lens = None
        if not swa_only:
            assert attn_metadata is not None
            if self.compress_ratio == 4:
                assert swa_metadata.is_valid_token is not None
                assert self.topk_indices_buffer is not None
                block_size = attn_metadata.block_size // self.compress_ratio
                source_indices = self.topk_indices_buffer[:num_decode_tokens]
                global_indices, topk_lens = compute_global_topk_indices_and_lens(
                    source_indices,
                    swa_metadata.token_to_req_indices,
                    attn_metadata.block_table[:num_decodes],
                    block_size,
                    swa_metadata.is_valid_token[:num_decode_tokens],
                    output_buffers=self._global_topk_output_buffers(source_indices),
                )
                topk_indices = global_indices.view(num_decode_tokens, 1, -1)
            else:
                topk_indices = attn_metadata.c128a_global_decode_topk_indices
                topk_lens = attn_metadata.c128a_decode_topk_lens

        extra_indices = None
        if topk_indices is not None:
            extra_indices = topk_indices.reshape(num_decode_tokens, -1).contiguous()

        expected_row_bytes = get_deepseek_v4_cache_layout(self.kv_cache_dtype).row_bytes
        cache_shapes = {"swa_cache": self.swa_cache_layer.kv_cache}
        if not swa_only:
            cache_shapes["extra_cache"] = kv_cache
        for cache_name, cache in cache_shapes.items():
            if (
                cache is None
                or cache.ndim != 3
                or cache.shape[-1] != expected_row_bytes
            ):
                actual_shape = None if cache is None else tuple(cache.shape)
                raise RuntimeError(
                    f"{self.kv_cache_dtype} {cache_name} must have shape "
                    f"[num_blocks, block_size, {expected_row_bytes}], got "
                    f"{actual_shape}"
                )

        decode_swa_indices = swa_metadata.decode_swa_indices
        decode_swa_lens = swa_metadata.decode_swa_lens
        if decode_swa_indices is None or decode_swa_lens is None:
            raise RuntimeError("sparse decode SWA metadata is missing")
        result = flash_mla_decode(
            q=q,
            swa_cache=self.swa_cache_layer.kv_cache,
            swa_indices=decode_swa_indices[:num_decode_tokens],
            swa_lens=decode_swa_lens[:num_decode_tokens],
            scale=self.scale,
            attn_sink=self.attn_sink,
            extra_cache=None if swa_only else kv_cache,
            extra_indices=extra_indices,
            extra_lens=topk_lens,
        )
        output.copy_(result)
