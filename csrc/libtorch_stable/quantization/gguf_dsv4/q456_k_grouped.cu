// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Q4_K, Q5_K, and Q6_K arithmetic follows the MIT-licensed
// Whamp/llama.cpp@0379cf4 GGML format definition. The 8x16 SM86 IMMA
// schedule and caller-owned Q8_1 contract are vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"
#include "int8_mma.cuh"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kBlockElements = 256;
constexpr int kGroupElements = 32;
constexpr int kTokenTile = 8;
constexpr int kOutputTile = 16;
constexpr int kThreads = 32;

enum class KQuantFormat { kQ4, kQ5, kQ6 };

template <KQuantFormat kFormat>
constexpr int block_bytes() {
  if constexpr (kFormat == KQuantFormat::kQ4) {
    return 144;
  } else if constexpr (kFormat == KQuantFormat::kQ5) {
    return 176;
  } else {
    return 210;
  }
}

struct GroupedKQuantSharedStorage {
  alignas(16) int8_t activation_codes[kTokenTile * kGroupElements];
  int32_t weight_fragments[kOutputTile * 8];
  float weight_scales[kOutputTile];
  float weight_mins[kOutputTile];
  float activation_scales[kTokenTile];
  int32_t code_sums[kTokenTile];
};

__device__ __forceinline__ uint32_t pack_q456_values(const int8_t* values) {
  uint32_t packed = 0;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(values[index]))
              << (8 * index);
  }
  return packed;
}

__device__ __forceinline__ void decode_q45_scale_min(const uint8_t* scales,
                                                     int group_index,
                                                     int& scale, int& minimum) {
  if (group_index < 4) {
    scale = scales[group_index] & 63;
    minimum = scales[group_index + 4] & 63;
  } else {
    scale =
        (scales[group_index + 4] & 15) | ((scales[group_index - 4] >> 6) << 4);
    minimum =
        (scales[group_index + 4] >> 4) | ((scales[group_index] >> 6) << 4);
  }
}

template <KQuantFormat kFormat>
__device__ void load_q45_weight_tile(const uint8_t* __restrict__ weights,
                                     int output_tile_start, int output_rows,
                                     int input_columns, int group_index,
                                     GroupedKQuantSharedStorage& shared) {
  const int lane = threadIdx.x;
  if (lane < kOutputTile) {
    const int output_row = output_tile_start + lane;
    if (output_row >= output_rows) {
#pragma unroll
      for (int pack = 0; pack < 8; ++pack) {
        shared.weight_fragments[lane * 8 + pack] = 0;
      }
      shared.weight_scales[lane] = 0.0f;
      shared.weight_mins[lane] = 0.0f;
    } else {
      const int blocks_per_row = input_columns / kBlockElements;
      const int block_index = group_index / 8;
      const int group_in_block = group_index % 8;
      const uint8_t* block =
          weights + output_row * blocks_per_row * block_bytes<kFormat>() +
          block_index * block_bytes<kFormat>();
      const float2 decoded_scales =
          __half22float2(*reinterpret_cast<const __half2*>(block));
      int scale;
      int minimum;
      decode_q45_scale_min(block + 4, group_in_block, scale, minimum);
      shared.weight_scales[lane] = decoded_scales.x * scale;
      shared.weight_mins[lane] = decoded_scales.y * minimum;
      const uint8_t* high_bits =
          kFormat == KQuantFormat::kQ5 ? block + 16 : nullptr;
      const uint8_t* quants =
          kFormat == KQuantFormat::kQ5 ? block + 48 : block + 16;
      const int segment = group_in_block / 2;
      const bool high_nibble = (group_in_block & 1) != 0;
      const uint8_t* segment_quants = quants + segment * 32;
#pragma unroll
      for (int pack = 0; pack < 8; ++pack) {
        int8_t values[4];
#pragma unroll
        for (int byte_index = 0; byte_index < 4; ++byte_index) {
          const int element = 4 * pack + byte_index;
          const uint8_t packed = segment_quants[element];
          uint8_t value = high_nibble ? packed >> 4 : packed & 15;
          if constexpr (kFormat == KQuantFormat::kQ5) {
            if ((high_bits[element] & (1U << group_in_block)) != 0) {
              value |= 16;
            }
          }
          values[byte_index] = static_cast<int8_t>(value);
        }
        shared.weight_fragments[lane * 8 + pack] =
            static_cast<int>(pack_q456_values(values));
      }
    }
  }
  __syncwarp();
}

__device__ void load_q6_weight_tile(const uint8_t* __restrict__ weights,
                                    int output_tile_start, int output_rows,
                                    int input_columns, int group_index,
                                    bool second_half,
                                    GroupedKQuantSharedStorage& shared) {
  const int lane = threadIdx.x;
  if (lane < kOutputTile) {
    const int output_row = output_tile_start + lane;
    if (output_row >= output_rows) {
#pragma unroll
      for (int pack = 0; pack < 8; ++pack) {
        shared.weight_fragments[lane * 8 + pack] = 0;
      }
      shared.weight_scales[lane] = 0.0f;
      shared.weight_mins[lane] = 0.0f;
    } else {
      const int blocks_per_row = input_columns / kBlockElements;
      const int block_index = group_index / 8;
      const int group_in_block = group_index % 8;
      const uint8_t* block =
          weights +
          output_row * blocks_per_row * block_bytes<KQuantFormat::kQ6>() +
          block_index * block_bytes<KQuantFormat::kQ6>();
      const uint8_t* low = block;
      const uint8_t* high = block + 128;
      const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
      const int half = group_in_block / 4;
      const int quadrant = group_in_block % 4;
      const int low_base = half * 64 + ((quadrant & 1) != 0 ? 32 : 0);
      const int high_base = half * 32;
      const int nibble_shift = quadrant >= 2 ? 4 : 0;
      const int high_shift = 2 * quadrant;
      const int scale_base = half * 8 + 2 * quadrant;
      shared.weight_scales[lane] =
          __half2float(*reinterpret_cast<const __half*>(block + 208)) *
          scales[scale_base + static_cast<int>(second_half)];
      shared.weight_mins[lane] = 0.0f;
#pragma unroll
      for (int pack = 0; pack < 8; ++pack) {
        int8_t values[4] = {};
        if ((pack >= 4) == second_half) {
#pragma unroll
          for (int byte_index = 0; byte_index < 4; ++byte_index) {
            const int element = 4 * pack + byte_index;
            const int low_value =
                (low[low_base + element] >> nibble_shift) & 15;
            const int high_value =
                (high[high_base + element] >> high_shift) & 3;
            values[byte_index] =
                static_cast<int8_t>(low_value | (high_value << 4)) - 32;
          }
        }
        shared.weight_fragments[lane * 8 + pack] =
            static_cast<int>(pack_q456_values(values));
      }
    }
  }
  __syncwarp();
}

__device__ void run_k_quant_mma(GroupedKQuantSharedStorage& shared,
                                int (&accumulator)[4]) {
  const int lane = threadIdx.x;
  const int row0 = lane / 4;
  const int row1 = row0 + 8;
  const int chunk = lane & 3;
  const int weight_fragment[4] = {
      shared.weight_fragments[row0 * 8 + chunk],
      shared.weight_fragments[row1 * 8 + chunk],
      shared.weight_fragments[row0 * 8 + chunk + 4],
      shared.weight_fragments[row1 * 8 + chunk + 4],
  };
  mma_int8_m16n8k32_row_col(weight_fragment, shared.activation_codes, lane,
                            accumulator);
  __syncwarp();
}

template <KQuantFormat kFormat>
__global__ void k_quant_q8_1_grouped_matmul_kernel(
    const __half* __restrict__ token_scales,
    const int8_t* __restrict__ token_codes, const uint8_t* __restrict__ weights,
    float* __restrict__ output, int token_count, int output_rows,
    int input_columns, int output_row_stride) {
  const int token_tile_start = blockIdx.y * kTokenTile;
  const int output_tile_start = blockIdx.x * kOutputTile;
  __shared__ GroupedKQuantSharedStorage shared;
  const int lane = threadIdx.x;
  float sums[4] = {};
  const int group_count = input_columns / kGroupElements;
  for (int group_index = 0; group_index < group_count; ++group_index) {
    for (int index = lane; index < kTokenTile * kGroupElements;
         index += kThreads) {
      const int token_in_tile = index / kGroupElements;
      const int element = index % kGroupElements;
      const int token = token_tile_start + token_in_tile;
      shared.activation_codes[index] =
          token < token_count
              ? token_codes[token * input_columns +
                            group_index * kGroupElements + element]
              : 0;
    }
    if (lane < kTokenTile) {
      const int token = token_tile_start + lane;
      shared.activation_scales[lane] =
          token < token_count
              ? __half2float(token_scales[token * group_count + group_index])
              : 0.0f;
    }
    __syncwarp();
    if (lane < kTokenTile) {
      const int token = token_tile_start + lane;
      int code_sum = 0;
      if (token < token_count) {
        const int8_t* codes = shared.activation_codes + lane * kGroupElements;
#pragma unroll
        for (int element = 0; element < kGroupElements; ++element) {
          code_sum += codes[element];
        }
      }
      shared.code_sums[lane] = code_sum;
    }
    __syncwarp();
    int first_accumulator[4] = {};
    if constexpr (kFormat == KQuantFormat::kQ6) {
      load_q6_weight_tile(weights, output_tile_start, output_rows,
                          input_columns, group_index, false, shared);
      run_k_quant_mma(shared, first_accumulator);
      const float first_scales[4] = {
          shared.weight_scales[lane / 4],
          shared.weight_scales[lane / 4],
          shared.weight_scales[lane / 4 + 8],
          shared.weight_scales[lane / 4 + 8],
      };
      __syncwarp();
      int second_accumulator[4] = {};
      load_q6_weight_tile(weights, output_tile_start, output_rows,
                          input_columns, group_index, true, shared);
      run_k_quant_mma(shared, second_accumulator);
#pragma unroll
      for (int local = 0; local < 4; ++local) {
        const int token_in_tile = (lane % 4) * 2 + local % 2;
        const int output_row = (local / 2) * 8 + lane / 4;
        sums[local] = fmaf(
            shared.activation_scales[token_in_tile],
            first_scales[local] * first_accumulator[local] +
                shared.weight_scales[output_row] * second_accumulator[local],
            sums[local]);
      }
    } else {
      load_q45_weight_tile<kFormat>(weights, output_tile_start, output_rows,
                                    input_columns, group_index, shared);
      run_k_quant_mma(shared, first_accumulator);
#pragma unroll
      for (int local = 0; local < 4; ++local) {
        const int token_in_tile = (lane % 4) * 2 + local % 2;
        const int output_row = (local / 2) * 8 + lane / 4;
        const float corrected =
            shared.weight_scales[output_row] * first_accumulator[local] -
            shared.weight_mins[output_row] * shared.code_sums[token_in_tile];
        sums[local] = fmaf(shared.activation_scales[token_in_tile], corrected,
                           sums[local]);
      }
    }
    __syncwarp();
  }
#pragma unroll
  for (int local = 0; local < 4; ++local) {
    const int token = token_tile_start + (lane % 4) * 2 + local % 2;
    const int output_row = output_tile_start + (local / 2) * 8 + lane / 4;
    if (token < token_count && output_row < output_rows) {
      output[token * output_row_stride + output_row] = sums[local];
    }
  }
}

template <KQuantFormat kFormat>
void launch_k_quant_grouped_matmul(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& weights, torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(activation_scales.device().is_cuda() &&
                      activation_scales.is_contiguous() &&
                      activation_codes.device().is_cuda() &&
                      activation_codes.is_contiguous() &&
                      weights.device().is_cuda() && weights.is_contiguous(),
                  "GGUF grouped Q4/Q5/Q6 inputs must be contiguous CUDA "
                  "tensors");
  STD_TORCH_CHECK(output.device().is_cuda() && output.dim() == 2 &&
                      output.stride(1) == 1,
                  "GGUF grouped Q4/Q5/Q6 output must be a 2D CUDA tensor "
                  "with unit-stride columns; row stride may come from a "
                  "wider combined buffer");
  STD_TORCH_CHECK(activation_scales.get_device_index() ==
                      output.get_device_index(),
                  "GGUF grouped Q4/Q5/Q6 tensors must share one CUDA device");
  STD_TORCH_CHECK(activation_scales.scalar_type() == ScalarType::Half &&
                      activation_codes.scalar_type() == ScalarType::Char &&
                      weights.scalar_type() == ScalarType::Byte &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF grouped Q4/Q5/Q6 dtype contract mismatch");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      weights.dim() == 2 && output.dim() == 2,
                  "GGUF grouped Q4/Q5/Q6 tensor rank mismatch");
  const int token_count = activation_codes.size(0);
  const int input_columns = activation_codes.size(1);
  const int output_rows = weights.size(0);
  STD_TORCH_CHECK(
      input_columns % kBlockElements == 0 &&
          activation_scales.size(0) == token_count &&
          activation_scales.size(1) == input_columns / kGroupElements &&
          weights.size(1) ==
              input_columns / kBlockElements * block_bytes<kFormat>() &&
          output.size(0) == token_count && output.size(1) == output_rows,
      "GGUF grouped Q4/Q5/Q6 shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const dim3 grid((output_rows + kOutputTile - 1) / kOutputTile,
                  (token_count + kTokenTile - 1) / kTokenTile);
  k_quant_q8_1_grouped_matmul_kernel<kFormat><<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
      reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
      weights.const_data_ptr<uint8_t>(), output.mutable_data_ptr<float>(),
      token_count, output_rows, input_columns,
      static_cast<int>(output.stride(0)));
}

}  // namespace

void gguf_q4_k_q8_1_grouped_matmul(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& weights, torch::stable::Tensor& output) {
  launch_k_quant_grouped_matmul<KQuantFormat::kQ4>(
      activation_scales, activation_codes, weights, output);
}

void gguf_q5_k_q8_1_grouped_matmul(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& weights, torch::stable::Tensor& output) {
  launch_k_quant_grouped_matmul<KQuantFormat::kQ5>(
      activation_scales, activation_codes, weights, output);
}

void gguf_q6_k_q8_1_grouped_matmul(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& weights, torch::stable::Tensor& output) {
  launch_k_quant_grouped_matmul<KQuantFormat::kQ6>(
      activation_scales, activation_codes, weights, output);
}

}  // namespace vllm::gguf_dsv4
