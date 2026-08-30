#include "core/registration.h"

#include <c10/util/BFloat16.h>
#include <c10/util/Float8_e4m3fn.h>
#include <c10/util/Half.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <array>
#include <cstdint>
#include <vector>

namespace {

using torch::headeronly::ScalarType;
using torch::stable::Tensor;

constexpr int64_t kNvFp4BlockSize = 16;
constexpr std::array<float, 16> kNvFp4Values = {
    0.0f,  0.5f,  1.0f,  1.5f,  2.0f,  3.0f,  4.0f,  6.0f,
    -0.0f, -0.5f, -1.0f, -1.5f, -2.0f, -3.0f, -4.0f, -6.0f,
};

void validate_nvfp4_ple_shards(const std::vector<Tensor>& code_shards,
                               const std::vector<Tensor>& scale_shards,
                               const Tensor& outer_scales,
                               int64_t rows_per_shard, int64_t output_width) {
  STD_TORCH_CHECK(!code_shards.empty(),
                  "PLE NVFP4 gather requires at least one shard");
  STD_TORCH_CHECK(code_shards.size() == scale_shards.size(),
                  "PLE NVFP4 code and scale shard counts must match");
  STD_TORCH_CHECK(rows_per_shard > 0,
                  "PLE NVFP4 rows per shard must be positive");
  STD_TORCH_CHECK(output_width > 0 && output_width % kNvFp4BlockSize == 0,
                  "PLE NVFP4 output width must be a positive multiple of 16");
  STD_TORCH_CHECK(
      outer_scales.device().is_cpu() && outer_scales.is_contiguous() &&
          outer_scales.scalar_type() == ScalarType::Float &&
          outer_scales.dim() == 1 &&
          outer_scales.size(0) == static_cast<int64_t>(code_shards.size()),
      "PLE NVFP4 outer scales must be one contiguous CPU float32 value per "
      "shard");

  const int64_t packed_width = output_width / 2;
  const int64_t scale_width = output_width / kNvFp4BlockSize;
  for (size_t shard_index = 0; shard_index < code_shards.size();
       ++shard_index) {
    const Tensor& codes = code_shards[shard_index];
    const Tensor& scales = scale_shards[shard_index];
    STD_TORCH_CHECK(codes.device().is_cpu() && codes.is_contiguous() &&
                        codes.scalar_type() == ScalarType::Byte &&
                        codes.dim() == 2 && codes.size(0) == rows_per_shard &&
                        codes.size(1) == packed_width,
                    "PLE NVFP4 code shard has incompatible storage geometry");
    STD_TORCH_CHECK(scales.device().is_cpu() && scales.is_contiguous() &&
                        scales.scalar_type() == ScalarType::Float8_e4m3fn &&
                        scales.dim() == 2 && scales.size(0) == rows_per_shard &&
                        scales.size(1) == scale_width,
                    "PLE NVFP4 scale shard has incompatible storage geometry");
  }
}

template <typename OutputType>
void gather_nvfp4_ple_rows_typed(const std::vector<Tensor>& code_shards,
                                 const std::vector<Tensor>& scale_shards,
                                 const Tensor& outer_scales,
                                 const Tensor& row_ids, Tensor& output,
                                 int64_t rows_per_shard) {
  const int64_t output_width = output.size(1);
  const int64_t packed_width = output_width / 2;
  const int64_t scale_width = output_width / kNvFp4BlockSize;
  const auto* ids = row_ids.const_data_ptr<int64_t>();
  const auto* outer = outer_scales.const_data_ptr<float>();
  auto* output_rows = output.mutable_data_ptr<OutputType>();

  for (int64_t output_row = 0; output_row < row_ids.size(0); ++output_row) {
    const int64_t global_row = ids[output_row];
    const int64_t shard_index = global_row / rows_per_shard;
    STD_TORCH_CHECK(global_row >= 0 &&
                        shard_index < static_cast<int64_t>(code_shards.size()),
                    "PLE NVFP4 row ID is outside the sidecar row range");
    const int64_t local_row = global_row - shard_index * rows_per_shard;
    const auto* codes = code_shards[shard_index].const_data_ptr<uint8_t>() +
                        local_row * packed_width;
    const auto* scales =
        scale_shards[shard_index].const_data_ptr<c10::Float8_e4m3fn>() +
        local_row * scale_width;
    auto* output_values = output_rows + output_row * output_width;

    for (int64_t block = 0; block < scale_width; ++block) {
      const float block_scale = static_cast<float>(scales[block]);
      const int64_t packed_block_start = block * (kNvFp4BlockSize / 2);
      const int64_t value_block_start = block * kNvFp4BlockSize;
      for (int64_t packed_index = 0; packed_index < kNvFp4BlockSize / 2;
           ++packed_index) {
        const uint8_t code = codes[packed_block_start + packed_index];
        float low_value = kNvFp4Values[code & 0x0f] * block_scale;
        low_value *= outer[shard_index];
        float high_value = kNvFp4Values[code >> 4] * block_scale;
        high_value *= outer[shard_index];
        output_values[value_block_start + 2 * packed_index] =
            static_cast<OutputType>(low_value);
        output_values[value_block_start + 2 * packed_index + 1] =
            static_cast<OutputType>(high_value);
      }
    }
  }
}

void gather_nvfp4_ple_rows(const std::vector<Tensor>& code_shards,
                           const std::vector<Tensor>& scale_shards,
                           const Tensor& outer_scales, const Tensor& row_ids,
                           Tensor& output, int64_t rows_per_shard) {
  STD_TORCH_CHECK(
      row_ids.device().is_cpu() && row_ids.is_contiguous() &&
          row_ids.scalar_type() == ScalarType::Long && row_ids.dim() == 1,
      "PLE NVFP4 row IDs must be a contiguous one-dimensional CPU int64 "
      "tensor");
  STD_TORCH_CHECK(output.device().is_cpu() && output.is_contiguous() &&
                      output.dim() == 2 && output.size(0) == row_ids.size(0),
                  "PLE NVFP4 output must be contiguous two-dimensional CPU "
                  "storage with one row per ID");
  validate_nvfp4_ple_shards(code_shards, scale_shards, outer_scales,
                            rows_per_shard, output.size(1));

  switch (output.scalar_type()) {
    case ScalarType::BFloat16:
      gather_nvfp4_ple_rows_typed<c10::BFloat16>(code_shards, scale_shards,
                                                 outer_scales, row_ids, output,
                                                 rows_per_shard);
      return;
    case ScalarType::Half:
      gather_nvfp4_ple_rows_typed<c10::Half>(code_shards, scale_shards,
                                             outer_scales, row_ids, output,
                                             rows_per_shard);
      return;
    case ScalarType::Float:
      gather_nvfp4_ple_rows_typed<float>(code_shards, scale_shards,
                                         outer_scales, row_ids, output,
                                         rows_per_shard);
      return;
    default:
      STD_TORCH_CHECK(
          false, "PLE NVFP4 output must use bfloat16, float16, or float32");
  }
}

}  // namespace

STABLE_TORCH_LIBRARY_IMPL(_C, CPU, ops) {
  ops.impl("gather_nvfp4_ple_rows", TORCH_BOX(&gather_nvfp4_ple_rows));
}
