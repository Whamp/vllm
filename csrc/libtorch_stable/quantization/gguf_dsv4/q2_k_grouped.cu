// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Q2_K format math follows the MIT-licensed llama.cpp/GGML implementation
// and antirez/ds4@84cc882 cuda/mmq/ds4_mmq_d2r.cu. The stable ABI and
// block-8 grouped operator are original vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"
#include "int8_mma.cuh"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kQ2BlockElements = 256;
constexpr int kQ2BlockBytes = 84;
constexpr int kGroupElements = 32;
constexpr int kAssignmentTile = 8;
constexpr int kOutputTile = 16;
constexpr int kThreads = 32;

struct GroupedQ2SharedStorage {
  alignas(16) int8_t activation_codes[kAssignmentTile * kGroupElements];
  float activation_scales[kAssignmentTile];
  int activation_sum_low[kAssignmentTile];
  int activation_sum_high[kAssignmentTile];
  float weight_scales[kOutputTile];
  float min_scales_low[kOutputTile];
  float min_scales_high[kOutputTile];
  int32_t assignments[kAssignmentTile];
};

__device__ __forceinline__ int sum_int8_four(int packed) {
  return static_cast<int>(static_cast<int8_t>(packed)) +
         static_cast<int>(static_cast<int8_t>(packed >> 8)) +
         static_cast<int>(static_cast<int8_t>(packed >> 16)) +
         static_cast<int>(static_cast<int8_t>(packed >> 24));
}

__device__ __forceinline__ const uint8_t* q2_block_address(
    const uint8_t* weights, int expert, int output_row, int output_rows,
    int input_columns, int group_index) {
  const int blocks_per_row = input_columns / kQ2BlockElements;
  const int row_bytes = blocks_per_row * kQ2BlockBytes;
  return weights + (expert * output_rows + output_row) * row_bytes +
         (group_index / 8) * kQ2BlockBytes;
}

__device__ void load_q2_weight_metadata(const uint8_t* __restrict__ weights,
                                        int expert, int output_tile_start,
                                        int output_rows, int input_columns,
                                        int group_index,
                                        GroupedQ2SharedStorage& shared) {
  const int lane = threadIdx.x;
  if (lane < kOutputTile) {
    const int output_row = output_tile_start + lane;
    if (output_row < output_rows) {
      const uint8_t* block = q2_block_address(
          weights, expert, output_row, output_rows, input_columns, group_index);
      const int group_in_block = group_index % 8;
      const int scale_low = block[2 * group_in_block];
      const int scale_high = block[2 * group_in_block + 1];
      shared.weight_scales[lane] =
          __half2float(*reinterpret_cast<const __half*>(block + 80));
      const float min_scale =
          __half2float(*reinterpret_cast<const __half*>(block + 82));
      shared.min_scales_low[lane] = min_scale * (scale_low >> 4);
      shared.min_scales_high[lane] = min_scale * (scale_high >> 4);
    } else {
      shared.weight_scales[lane] = 0.0f;
      shared.min_scales_low[lane] = 0.0f;
      shared.min_scales_high[lane] = 0.0f;
    }
  }
  __syncwarp();
}

__device__ __forceinline__ int decode_q2_fragment_register(
    const uint8_t* weights, int expert, int output_row, int output_rows,
    int input_columns, int group_index, int chunk) {
  if (output_row >= output_rows) {
    return 0;
  }
  const uint8_t* block = q2_block_address(
      weights, expert, output_row, output_rows, input_columns, group_index);
  const int group_in_block = group_index % 8;
  const int q_byte_base = (group_in_block / 4) * 32;
  const int half_offset = chunk < 4 ? 0 : 16;
  const int chunk_in_half = chunk & 3;
  const int packed = *reinterpret_cast<const int*>(
      block + 16 + q_byte_base + half_offset + chunk_in_half * 4);
  const int q2 = (packed >> (2 * (group_in_block % 4))) & 0x03030303;
  const int scale = block[2 * group_in_block + (chunk >= 4 ? 1 : 0)] & 0x0F;
  return q2 * scale;
}

__device__ void run_q2_mma_group(const uint8_t* __restrict__ weights,
                                 int expert, int output_tile_start,
                                 int output_rows, int input_columns,
                                 int group_index,
                                 GroupedQ2SharedStorage& shared,
                                 float (&sums)[4]) {
  load_q2_weight_metadata(weights, expert, output_tile_start, output_rows,
                          input_columns, group_index, shared);
  const int lane = threadIdx.x;
  const int row0 = output_tile_start + lane / 4;
  const int row1 = row0 + 8;
  const int chunk = lane & 3;
  int weight_fragment[4] = {
      decode_q2_fragment_register(weights, expert, row0, output_rows,
                                  input_columns, group_index, chunk),
      decode_q2_fragment_register(weights, expert, row1, output_rows,
                                  input_columns, group_index, chunk),
      decode_q2_fragment_register(weights, expert, row0, output_rows,
                                  input_columns, group_index, chunk + 4),
      decode_q2_fragment_register(weights, expert, row1, output_rows,
                                  input_columns, group_index, chunk + 4),
  };
  int accumulator[4] = {};
  mma_int8_m16n8k32_row_col(weight_fragment, shared.activation_codes, lane,
                            accumulator);

#pragma unroll
  for (int local = 0; local < 4; ++local) {
    const int output_row = (local / 2) * 8 + lane / 4;
    const int assignment = (lane % 4) * 2 + local % 2;
    const float activation_scale = shared.activation_scales[assignment];
    const float min_correction = shared.min_scales_low[output_row] *
                                     shared.activation_sum_low[assignment] +
                                 shared.min_scales_high[output_row] *
                                     shared.activation_sum_high[assignment];
    sums[local] = fmaf(static_cast<float>(accumulator[local]),
                       shared.weight_scales[output_row] * activation_scale,
                       sums[local] - activation_scale * min_correction);
  }
  __syncwarp();
}

__global__ void q2_k_q8_1_grouped_down_kernel(
    const __half* __restrict__ assignment_scales,
    const int8_t* __restrict__ assignment_codes,
    const uint8_t* __restrict__ down_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_padded, float* __restrict__ output,
    int assignment_count, int expert_count, int output_rows,
    int input_columns) {
  const int assignment_block = blockIdx.y;
  const int assignment_start = assignment_block * kAssignmentTile;
  if (assignment_start >= *num_tokens_padded) {
    return;
  }
  const int expert = expert_ids[assignment_block];
  if (expert < 0 || expert >= expert_count) {
    return;
  }
  const int output_tile_start = blockIdx.x * kOutputTile;
  __shared__ GroupedQ2SharedStorage shared;
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
          assignment < assignment_count
              ? assignment_codes[assignment * input_columns +
                                 group_index * kGroupElements + element]
              : 0;
    }
    __syncwarp();
    if (lane < kAssignmentTile) {
      const int assignment = shared.assignments[lane];
      if (assignment < assignment_count) {
        shared.activation_scales[lane] = __half2float(
            assignment_scales[assignment * group_count + group_index]);
        const int* codes = reinterpret_cast<const int*>(
            shared.activation_codes + lane * kGroupElements);
        shared.activation_sum_low[lane] =
            sum_int8_four(codes[0]) + sum_int8_four(codes[1]) +
            sum_int8_four(codes[2]) + sum_int8_four(codes[3]);
        shared.activation_sum_high[lane] =
            sum_int8_four(codes[4]) + sum_int8_four(codes[5]) +
            sum_int8_four(codes[6]) + sum_int8_four(codes[7]);
      } else {
        shared.activation_scales[lane] = 0.0f;
        shared.activation_sum_low[lane] = 0;
        shared.activation_sum_high[lane] = 0;
      }
    }
    __syncwarp();
    run_q2_mma_group(down_weights, expert, output_tile_start, output_rows,
                     input_columns, group_index, shared, sums);
  }

#pragma unroll
  for (int local = 0; local < 4; ++local) {
    const int output_row_in_tile = (local / 2) * 8 + lane / 4;
    const int assignment_slot = (lane % 4) * 2 + local % 2;
    const int output_row = output_tile_start + output_row_in_tile;
    const int assignment = shared.assignments[assignment_slot];
    if (assignment < assignment_count && output_row < output_rows) {
      output[assignment * output_rows + output_row] = sums[local];
    }
  }
}

}  // namespace

void gguf_q2_k_q8_1_grouped_down(const torch::stable::Tensor& assignment_scales,
                                 const torch::stable::Tensor& assignment_codes,
                                 const torch::stable::Tensor& down_weights,
                                 const torch::stable::Tensor& sorted_token_ids,
                                 const torch::stable::Tensor& expert_ids,
                                 const torch::stable::Tensor& num_tokens_padded,
                                 torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &assignment_scales, &assignment_codes,  &down_weights, &sorted_token_ids,
      &expert_ids,        &num_tokens_padded, &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF grouped Q2 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(tensor->get_device_index() == output.get_device_index(),
                    "GGUF grouped Q2 tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(assignment_scales.scalar_type() == ScalarType::Half &&
                      assignment_codes.scalar_type() == ScalarType::Char &&
                      down_weights.scalar_type() == ScalarType::Byte &&
                      sorted_token_ids.scalar_type() == ScalarType::Int &&
                      expert_ids.scalar_type() == ScalarType::Int &&
                      num_tokens_padded.scalar_type() == ScalarType::Int &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF grouped Q2 dtype contract mismatch");
  STD_TORCH_CHECK(assignment_codes.dim() == 2 && assignment_scales.dim() == 2 &&
                      down_weights.dim() == 3 && sorted_token_ids.dim() == 1 &&
                      expert_ids.dim() == 1 && num_tokens_padded.numel() == 1 &&
                      output.dim() == 3,
                  "GGUF grouped Q2 tensor rank mismatch");
  const int assignment_count = assignment_codes.size(0);
  const int input_columns = assignment_codes.size(1);
  const int expert_count = down_weights.size(0);
  const int output_rows = down_weights.size(1);
  STD_TORCH_CHECK(output.size(0) * output.size(1) == assignment_count &&
                      output.size(2) == output_rows,
                  "GGUF grouped Q2 output shape mismatch");
  STD_TORCH_CHECK(
      input_columns % kQ2BlockElements == 0 &&
          down_weights.size(2) ==
              input_columns / kQ2BlockElements * kQ2BlockBytes &&
          assignment_scales.size(0) == assignment_count &&
          assignment_scales.size(1) == input_columns / kGroupElements,
      "GGUF grouped Q2 weight/activation shape mismatch");

  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const dim3 grid((output_rows + kOutputTile - 1) / kOutputTile,
                  expert_ids.size(0));
  q2_k_q8_1_grouped_down_kernel<<<grid, kThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(assignment_scales.const_data_ptr()),
      assignment_codes.const_data_ptr<int8_t>(),
      down_weights.const_data_ptr<uint8_t>(),
      sorted_token_ids.const_data_ptr<int32_t>(),
      expert_ids.const_data_ptr<int32_t>(),
      num_tokens_padded.const_data_ptr<int32_t>(),
      output.mutable_data_ptr<float>(), assignment_count, expert_count,
      output_rows, input_columns);
}

}  // namespace vllm::gguf_dsv4
