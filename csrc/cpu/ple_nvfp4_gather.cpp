// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <cstdint>
#include <cstring>

namespace {

uint16_t float_to_bfloat16(float value) {
  uint32_t bits;
  std::memcpy(&bits, &value, sizeof(bits));
  const uint32_t magnitude = bits & 0x7fffffffU;
  if (magnitude > 0x7f800000U) {
    return static_cast<uint16_t>((bits >> 16) | 0x0040U);
  }
  const uint32_t rounding_bias = 0x7fffU + ((bits >> 16) & 1U);
  return static_cast<uint16_t>((bits + rounding_bias) >> 16);
}

}  // namespace

extern "C" int vllm_gather_nvfp4_ple_rows(
    const void* const* code_shards, const void* const* scale_shards,
    const float* outer_scales, const float* nvfp4_lut, const float* fp8_lut,
    int64_t shard_count, int64_t rows_per_shard, int64_t width,
    const int64_t* row_ids, int64_t row_count, void* output) {
  if (code_shards == nullptr || scale_shards == nullptr ||
      outer_scales == nullptr || nvfp4_lut == nullptr || fp8_lut == nullptr ||
      row_ids == nullptr || output == nullptr || shard_count <= 0 ||
      rows_per_shard <= 0 || width <= 0 || width % 16 != 0 || row_count < 0) {
    return -1;
  }

  const int64_t packed_width = width / 2;
  const int64_t scale_width = width / 16;
  auto* output_rows = static_cast<uint16_t*>(output);
  for (int64_t output_row = 0; output_row < row_count; ++output_row) {
    const int64_t global_row = row_ids[output_row];
    if (global_row < 0 || global_row / rows_per_shard >= shard_count) {
      return -2;
    }
    const int64_t shard_index = global_row / rows_per_shard;
    const int64_t local_row = global_row % rows_per_shard;
    const auto* code = static_cast<const uint8_t*>(code_shards[shard_index]) +
                       local_row * packed_width;
    const auto* scales =
        static_cast<const uint8_t*>(scale_shards[shard_index]) +
        local_row * scale_width;
    uint16_t* destination = output_rows + output_row * width;
    const float outer_scale = outer_scales[shard_index];

    for (int64_t packed_column = 0; packed_column < packed_width;
         ++packed_column) {
      const uint8_t packed = code[packed_column];
      const int64_t first_column = packed_column * 2;
      const float first = nvfp4_lut[packed & 0x0fU] *
                          fp8_lut[scales[first_column / 16]] * outer_scale;
      const float second = nvfp4_lut[packed >> 4] *
                           fp8_lut[scales[(first_column + 1) / 16]] *
                           outer_scale;
      destination[first_column] = float_to_bfloat16(first);
      destination[first_column + 1] = float_to_bfloat16(second);
    }
  }
  return 0;
}
