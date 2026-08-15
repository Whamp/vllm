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

import torch

import vllm.envs as envs
from vllm.logger import init_logger
from vllm.models.deepseek_v4.amd.rocm import (
    DeepseekV4ROCMAiterMLAAttention,
    DeepseekV4ROCMAiterMLASparseBackend,
    DeepseekV4ROCMAiterMLASparseMetadata,
    DeepseekV4ROCMAiterSparseSWAMetadata,
)
from vllm.models.deepseek_v4.common.ops.cache_utils import (
    compute_global_topk_indices_and_lens,
)
from vllm.platforms.interface import DeviceCapability

logger = init_logger(__name__)

FlashMLADecode = Callable[..., torch.Tensor]


def load_ampere_flash_mla_decode() -> FlashMLADecode:
    """Load the optional Ampere sparse MLA decode operator or fail closed."""
    try:
        flash_mla = importlib.import_module("flash_mla")
    except ModuleNotFoundError as error:
        raise RuntimeError(
            "DeepSeek V4 FlashMLA decode was enabled, but flash_mla is not installed"
        ) from error
    decode = getattr(flash_mla, "sparse_mla_decode_fp8", None)
    if decode is None:
        raise RuntimeError(
            "DeepSeek V4 FlashMLA decode was enabled, but flash_mla does not "
            "export sparse_mla_decode_fp8"
        )
    return decode


class DeepseekV4AmpereMLASparseBackend(DeepseekV4ROCMAiterMLASparseBackend):
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
        self._flash_mla_decode = (
            load_ampere_flash_mla_decode() if envs.VLLM_DSV4_FLASH_MLA_DECODE else None
        )
        if self._flash_mla_decode is not None:
            if self.kv_cache_dtype != "fp8_ds_mla":
                raise ValueError(
                    "DeepSeek V4 FlashMLA decode requires fp8_ds_mla KV cache, "
                    f"got {self.kv_cache_dtype}"
                )
            logger.info_once(
                "Using native Ampere FlashMLA sparse decode; prefill remains "
                "on the Triton sparse MLA path."
            )

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
        if self.kv_cache_dtype != "fp8_ds_mla":
            raise ValueError(
                "DeepSeek V4 FlashMLA decode requires fp8_ds_mla KV cache, "
                f"got {self.kv_cache_dtype}"
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
        result = flash_mla_decode(
            q=q,
            swa_cache=self.swa_cache_layer.kv_cache,
            swa_indices=swa_metadata.decode_swa_indices[:num_decode_tokens],
            swa_lens=swa_metadata.decode_swa_lens[:num_decode_tokens],
            scale=self.scale,
            attn_sink=self.attn_sink,
            extra_cache=None if swa_only else kv_cache,
            extra_indices=extra_indices,
            extra_lens=topk_lens,
        )
        output.copy_(result)
