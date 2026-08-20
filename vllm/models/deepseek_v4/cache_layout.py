# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Physical cache layouts for DeepSeek V4 absorbed MLA attention."""

from dataclasses import dataclass


@dataclass(frozen=True)
class DeepseekV4CacheLayout:
    cache_dtype: str
    nope_dim: int
    rope_dim: int
    quant_group_size: int
    nope_data_bytes: int
    scale_bytes: int
    block_alignment: int

    @property
    def rope_data_offset(self) -> int:
        return self.nope_data_bytes

    @property
    def rope_data_bytes(self) -> int:
        return self.rope_dim * 2

    @property
    def token_data_bytes(self) -> int:
        return self.nope_data_bytes + self.rope_data_bytes

    @property
    def num_scale_groups(self) -> int:
        return self.nope_dim // self.quant_group_size

    @property
    def row_bytes(self) -> int:
        return self.token_data_bytes + self.scale_bytes


FP8_DS_MLA_CACHE_LAYOUT = DeepseekV4CacheLayout(
    cache_dtype="fp8_ds_mla",
    nope_dim=448,
    rope_dim=64,
    quant_group_size=64,
    nope_data_bytes=448,
    scale_bytes=8,
    block_alignment=576,
)

FP4_DS_MLA_CACHE_LAYOUT = DeepseekV4CacheLayout(
    cache_dtype="fp4_ds_mla",
    nope_dim=448,
    rope_dim=64,
    quant_group_size=32,
    nope_data_bytes=224,
    scale_bytes=16,
    block_alignment=352,
)

_DEEPSEEK_V4_CACHE_LAYOUTS = {
    layout.cache_dtype: layout
    for layout in (FP8_DS_MLA_CACHE_LAYOUT, FP4_DS_MLA_CACHE_LAYOUT)
}


def get_deepseek_v4_cache_layout(cache_dtype: str) -> DeepseekV4CacheLayout:
    try:
        return _DEEPSEEK_V4_CACHE_LAYOUTS[cache_dtype]
    except KeyError as error:
        raise ValueError(
            f"Unsupported DeepSeek V4 cache layout: {cache_dtype}"
        ) from error
