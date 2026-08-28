"""Independent NumPy decoders written only from FORMAT-CONTRACT.md.

These are reference B of the GGUF format L0 oracle. They intentionally do
not share implementation code with the pinned llama.cpp reference A.
"""

import numpy as np

FLOAT32 = np.float32
QK_K = 256
QK8_0 = 32

GGUF_BLOCK_SIZE = {"q8_0": 34, "q2_K": 84, "iq2_xxs": 66}
GGUF_ELEMENTS_PER_BLOCK = {"q8_0": QK8_0, "q2_K": QK_K, "iq2_xxs": QK_K}


def dequantize_gguf_q8_0(raw: np.ndarray, _tables: dict[str, np.ndarray]) -> np.ndarray:
    """Decode Q8_0 blocks: fp16 d at byte 0 and 32 int8 codes at byte 2."""
    block_count = raw.shape[0]
    scale = raw[:, 0:2].copy().view(np.float16).astype(FLOAT32).reshape(block_count)
    codes = (
        raw[:, 2:34].copy().view(np.int8).astype(FLOAT32).reshape(block_count, QK8_0)
    )
    return (scale[:, None] * codes).reshape(-1)


def dequantize_gguf_q2_k(raw: np.ndarray, _tables: dict[str, np.ndarray]) -> np.ndarray:
    """Decode Q2_K blocks with the contract's two 128-weight chunk schedule."""
    block_count = raw.shape[0]
    scale = raw[:, 80:82].copy().view(np.float16).astype(FLOAT32).reshape(block_count)
    min_scale = (
        raw[:, 82:84].copy().view(np.float16).astype(FLOAT32).reshape(block_count)
    )
    packed_scales = raw[:, 0:16]
    packed_codes = raw[:, 16:80]

    output = np.empty((block_count, QK_K), dtype=FLOAT32)
    scale_index, code_base = 0, 0
    for chunk_index in range(2):
        for shift_stage in range(4):
            shift = 2 * shift_stage
            for half_index in range(2):
                packed_scale = packed_scales[:, scale_index]
                decoded_scale = scale * (packed_scale & 0xF).astype(FLOAT32)
                decoded_min = min_scale * (packed_scale >> 4).astype(FLOAT32)
                code_offset = code_base + 16 * half_index
                codes = (
                    (packed_codes[:, code_offset : code_offset + 16] >> shift) & 3
                ).astype(FLOAT32)
                weight_offset = 128 * chunk_index + 32 * shift_stage + 16 * half_index
                output[:, weight_offset : weight_offset + 16] = (
                    decoded_scale[:, None] * codes
                ) - decoded_min[:, None]
                scale_index += 1
        code_base += 32
    return output.reshape(-1)


def _little_endian_u32(group_bytes: np.ndarray, first_column: int) -> np.ndarray:
    """Decode one little-endian uint32 lane from IQ2_XXS group bytes."""
    return (
        group_bytes[:, :, first_column].astype(np.uint32)
        | (group_bytes[:, :, first_column + 1].astype(np.uint32) << np.uint32(8))
        | (group_bytes[:, :, first_column + 2].astype(np.uint32) << np.uint32(16))
        | (group_bytes[:, :, first_column + 3].astype(np.uint32) << np.uint32(24))
    )


def dequantize_gguf_iq2_xxs(
    raw: np.ndarray, tables: dict[str, np.ndarray]
) -> np.ndarray:
    """Decode IQ2_XXS blocks using the contract's grid and sign selectors."""
    grid = tables["iq2xxs_grid"]
    sign_table = tables["ksigns_iq2xs"]
    sign_mask = tables["kmask_iq2xs"]
    block_count = raw.shape[0]
    scale = raw[:, 0:2].copy().view(np.float16).astype(FLOAT32).reshape(block_count)
    group_bytes = raw[:, 2:66].reshape(block_count, 8, 8)
    scale_sign_word = _little_endian_u32(group_bytes, 4)

    group_scale = (
        scale[:, None, None]
        * (FLOAT32(0.5) + (scale_sign_word >> np.uint32(28)).astype(FLOAT32))[..., None]
    ) * FLOAT32(0.25)

    output = np.empty((block_count, 8, 32), dtype=FLOAT32)
    for grid_index_column in range(4):
        grid_values = grid[group_bytes[:, :, grid_index_column].astype(np.int32)]
        selector = (
            (scale_sign_word >> np.uint32(7 * grid_index_column)) & np.uint32(127)
        ).astype(np.int32)
        signs = sign_table[selector]
        negative = (signs[:, :, None] & sign_mask[None, None, :]) != 0
        product = group_scale * grid_values.astype(FLOAT32)
        output[:, :, 8 * grid_index_column : 8 * grid_index_column + 8] = np.where(
            negative, product * FLOAT32(-1.0), product * FLOAT32(1.0)
        )
    return output.reshape(-1)


GGUF_FORMAT_DECODERS = {
    "q8_0": dequantize_gguf_q8_0,
    "q2_K": dequantize_gguf_q2_k,
    "iq2_xxs": dequantize_gguf_iq2_xxs,
}
