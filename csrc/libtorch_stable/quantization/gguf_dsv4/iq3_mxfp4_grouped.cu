// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// IQ3_XXS and MXFP4 arithmetic follows the MIT-licensed
// Whamp/llama.cpp@0379cf4 GGML format definitions. The block-8 expert-major
// SM86 IMMA schedule and caller-owned Q8_1 contract are vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"
#include "int8_mma.cuh"
#include "iq1_iq3_tables.cuh"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kBlockElements = 256;
constexpr int kGroupElements = 32;
constexpr int kAssignmentTile = 8;
constexpr int kOutputTile = 16;
constexpr int kThreads = 32;

enum class DownFormat { kIq3XXS, kMXFP4 };

template <DownFormat kFormat>
__host__ __device__ constexpr int block_bytes() {
  if constexpr (kFormat == DownFormat::kIq3XXS) {
    return 98;
  } else {
    return 17;
  }
}

template <DownFormat kFormat>
__host__ __device__ constexpr int block_elements() {
  if constexpr (kFormat == DownFormat::kIq3XXS) {
    return 256;
  } else {
    return 32;
  }
}

struct GroupedDownSharedStorage {
  alignas(16) int8_t activation_codes[kAssignmentTile * kGroupElements];
  int32_t weight_fragments[kOutputTile * 8];
  float weight_scales[kOutputTile];
  float activation_scales[kAssignmentTile];
  int32_t assignments[kAssignmentTile];
};

__device__ __forceinline__ uint32_t load_u32(const uint8_t* address) {
  uint32_t value;
  memcpy(&value, address, sizeof(value));
  return value;
}

__device__ __forceinline__ int signed_iq3_grid_pack(uint32_t grid,
                                                    uint8_t signs,
                                                    bool high_half) {
  const uint32_t replicated = static_cast<uint32_t>(signs) * 0x01010101U;
  const uint32_t bit_selector = high_half ? 0x80402010U : 0x08040201U;
  const int sign_mask = __vcmpne4(replicated & bit_selector, 0);
  return __vsub4(static_cast<int>(grid) ^ sign_mask, sign_mask);
}

__device__ __forceinline__ int8_t decode_e2m1_times_two(uint8_t code) {
  const int magnitude_code = code & 7;
  const int magnitude = magnitude_code <= 4   ? magnitude_code
                        : magnitude_code == 5 ? 6
                        : magnitude_code == 6 ? 8
                                              : 12;
  return static_cast<int8_t>((code & 8) != 0 ? -magnitude : magnitude);
}

__device__ __forceinline__ float decode_e8m0_half(uint8_t exponent) {
  const uint32_t bits = exponent < 2
                            ? 0x00200000U << exponent
                            : static_cast<uint32_t>(exponent - 1) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ uint32_t pack_bytes(const int8_t* values) {
  uint32_t packed = 0;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(values[index]))
              << (8 * index);
  }
  return packed;
}

template <DownFormat kFormat>
__device__ void load_weight_tile(const uint8_t* __restrict__ weights,
                                 int expert, int output_tile_start,
                                 int expert_count, int output_rows,
                                 int input_columns, int group_index,
                                 GroupedDownSharedStorage& shared) {
  const int lane = threadIdx.x;
  if (lane < kOutputTile) {
    const int output_row = output_tile_start + lane;
    if (expert >= expert_count || output_row >= output_rows) {
#pragma unroll
      for (int pack = 0; pack < 8; ++pack) {
        shared.weight_fragments[lane * 8 + pack] = 0;
      }
      shared.weight_scales[lane] = 0.0f;
    } else {
      constexpr int elements_per_block = block_elements<kFormat>();
      const int blocks_per_row = input_columns / elements_per_block;
      const int block_index =
          group_index / (elements_per_block / kGroupElements);
      const int group_in_block =
          group_index % (elements_per_block / kGroupElements);
      const int raw_row_bytes = blocks_per_row * block_bytes<kFormat>();
      const uint8_t* block =
          weights + (expert * output_rows + output_row) * raw_row_bytes +
          block_index * block_bytes<kFormat>();
      if constexpr (kFormat == DownFormat::kIq3XXS) {
        const float block_scale =
            __half2float(*reinterpret_cast<const __half*>(block));
        const uint8_t* indices = block + 2 + group_in_block * 8;
        const uint32_t scale_signs =
            load_u32(block + 2 + 64 + group_in_block * 4);
        shared.weight_scales[lane] =
            block_scale * (0.5f + static_cast<float>(scale_signs >> 28)) * 0.5f;
#pragma unroll
        for (int pack = 0; pack < 8; ++pack) {
          const int part = pack / 2;
          const uint8_t signs = kIqSigns[(scale_signs >> (7 * part)) & 127];
          shared.weight_fragments[lane * 8 + pack] = signed_iq3_grid_pack(
              kIq3XXSGrid[indices[pack]], signs, (pack & 1) != 0);
        }
      } else {
        shared.weight_scales[lane] = decode_e8m0_half(block[0]);
        const uint8_t* quants = block + 1;
#pragma unroll
        for (int pack = 0; pack < 8; ++pack) {
          int8_t values[4];
#pragma unroll
          for (int element = 0; element < 4; ++element) {
            const int byte_index = (pack % 4) * 4 + element;
            const uint8_t packed = quants[byte_index];
            const uint8_t code = pack < 4 ? packed & 15 : packed >> 4;
            values[element] = decode_e2m1_times_two(code);
          }
          shared.weight_fragments[lane * 8 + pack] =
              static_cast<int>(pack_bytes(values));
        }
      }
    }
  }
  __syncwarp();
}

template <DownFormat kFormat>
__global__ void grouped_down_kernel(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_padded, float* __restrict__ output,
    int token_count, int topk, int expert_count, int output_rows,
    int input_columns) {
  const int assignment_start = blockIdx.y * kAssignmentTile;
  if (assignment_start >= *num_tokens_padded) {
    return;
  }
  const int expert = expert_ids[blockIdx.y];
  const int output_tile_start = blockIdx.x * kOutputTile;
  const int total_assignments = token_count * topk;
  __shared__ GroupedDownSharedStorage shared;
  const int lane = threadIdx.x;
  if (lane < kAssignmentTile) {
    shared.assignments[lane] = sorted_token_ids[assignment_start + lane];
  }
  __syncwarp();

  float sums[4] = {};
  const int group_count = input_columns / kGroupElements;
  for (int group_index = 0; group_index < group_count; ++group_index) {
    for (int index = lane; index < kAssignmentTile * kGroupElements;
         index += kThreads) {
      const int assignment_slot = index / kGroupElements;
      const int element = index % kGroupElements;
      const int assignment = shared.assignments[assignment_slot];
      shared.activation_codes[index] =
          assignment < total_assignments
              ? activation_codes[assignment * input_columns +
                                 group_index * kGroupElements + element]
              : 0;
    }
    if (lane < kAssignmentTile) {
      const int assignment = shared.assignments[lane];
      shared.activation_scales[lane] =
          assignment < total_assignments
              ? __half2float(
                    activation_scales[assignment * group_count + group_index])
              : 0.0f;
    }
    __syncwarp();
    load_weight_tile<kFormat>(weights, expert, output_tile_start, expert_count,
                              output_rows, input_columns, group_index, shared);
    const int row0 = lane / 4;
    const int row1 = row0 + 8;
    const int chunk = lane & 3;
    const int weight_fragment[4] = {
        shared.weight_fragments[row0 * 8 + chunk],
        shared.weight_fragments[row1 * 8 + chunk],
        shared.weight_fragments[row0 * 8 + chunk + 4],
        shared.weight_fragments[row1 * 8 + chunk + 4],
    };
    int accumulator[4] = {};
    mma_int8_m16n8k32_row_col(weight_fragment, shared.activation_codes, lane,
                              accumulator);
#pragma unroll
    for (int local = 0; local < 4; ++local) {
      const int output_row = (local / 2) * 8 + lane / 4;
      const int assignment = (lane % 4) * 2 + local % 2;
      sums[local] = fmaf(shared.weight_scales[output_row] *
                             shared.activation_scales[assignment],
                         static_cast<float>(accumulator[local]), sums[local]);
    }
    __syncwarp();
  }

#pragma unroll
  for (int local = 0; local < 4; ++local) {
    const int output_row_in_tile = (local / 2) * 8 + lane / 4;
    const int assignment_slot = (lane % 4) * 2 + local % 2;
    const int output_row = output_tile_start + output_row_in_tile;
    const int assignment = shared.assignments[assignment_slot];
    if (assignment < total_assignments && output_row < output_rows) {
      output[assignment * output_rows + output_row] = sums[local];
    }
  }
}

template <DownFormat kFormat>
void launch_grouped_down(const torch::stable::Tensor& activation_scales,
                         const torch::stable::Tensor& activation_codes,
                         const torch::stable::Tensor& weights,
                         const torch::stable::Tensor& sorted_token_ids,
                         const torch::stable::Tensor& expert_ids,
                         const torch::stable::Tensor& num_tokens_padded,
                         torch::stable::Tensor& output, int64_t topk) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &activation_scales, &activation_codes,  &weights, &sorted_token_ids,
      &expert_ids,        &num_tokens_padded, &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(
        tensor->device().is_cuda() && tensor->is_contiguous(),
        "GGUF grouped IQ3/MXFP4 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(
        tensor->get_device_index() == output.get_device_index(),
        "GGUF grouped IQ3/MXFP4 tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(activation_scales.scalar_type() == ScalarType::Half &&
                      activation_codes.scalar_type() == ScalarType::Char &&
                      weights.scalar_type() == ScalarType::Byte &&
                      sorted_token_ids.scalar_type() == ScalarType::Int &&
                      expert_ids.scalar_type() == ScalarType::Int &&
                      num_tokens_padded.scalar_type() == ScalarType::Int &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF grouped IQ3/MXFP4 dtype contract mismatch");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      weights.dim() == 3 && sorted_token_ids.dim() == 1 &&
                      expert_ids.dim() == 1 && num_tokens_padded.numel() == 1 &&
                      output.dim() == 3,
                  "GGUF grouped IQ3/MXFP4 tensor rank mismatch");
  const int token_count = output.size(0);
  const int expert_count = weights.size(0);
  const int output_rows = weights.size(1);
  const int input_columns = activation_codes.size(1);
  constexpr int elements_per_block = block_elements<kFormat>();
  STD_TORCH_CHECK(
      topk > 0 && output.size(1) == topk && output.size(2) == output_rows &&
          activation_codes.size(0) == token_count * topk &&
          activation_scales.size(0) == token_count * topk &&
          input_columns % elements_per_block == 0 &&
          activation_scales.size(1) == input_columns / kGroupElements &&
          weights.size(2) ==
              input_columns / elements_per_block * block_bytes<kFormat>(),
      "GGUF grouped IQ3/MXFP4 shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const dim3 grid((output_rows + kOutputTile - 1) / kOutputTile,
                  expert_ids.size(0));
  grouped_down_kernel<kFormat><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
      reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
      weights.const_data_ptr<uint8_t>(), sorted_token_ids.const_data_ptr<int>(),
      expert_ids.const_data_ptr<int>(), num_tokens_padded.const_data_ptr<int>(),
      output.mutable_data_ptr<float>(), token_count, static_cast<int>(topk),
      expert_count, output_rows, input_columns);
}

}  // namespace

void gguf_iq3_xxs_q8_1_grouped_down(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& sorted_token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_tokens_padded,
    torch::stable::Tensor& output, int64_t topk) {
  launch_grouped_down<DownFormat::kIq3XXS>(
      activation_scales, activation_codes, weights, sorted_token_ids,
      expert_ids, num_tokens_padded, output, topk);
}

void gguf_mxfp4_q8_1_grouped_down(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& weights,
    const torch::stable::Tensor& sorted_token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_tokens_padded,
    torch::stable::Tensor& output, int64_t topk) {
  launch_grouped_down<DownFormat::kMXFP4>(activation_scales, activation_codes,
                                          weights, sorted_token_ids, expert_ids,
                                          num_tokens_padded, output, topk);
}

}  // namespace vllm::gguf_dsv4
