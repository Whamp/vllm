// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// IQ2_XXS format math follows the MIT-licensed llama.cpp/GGML tables and
// antirez/ds4@84cc882 cuda/mmq load_tiles_iq2_xxs. The block-16 WMMA
// decomposition, stable ABI, and caller-owned scheduling contract are original
// vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"
#include "int8_mma.cuh"
#include "iq2_xxs_tables.cuh"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kIq2BlockElements = 256;
constexpr int kIq2BlockBytes = 66;
constexpr int kGroupElements = 32;
constexpr int kAssignmentTile = 8;
constexpr int kOutputTile = 16;
constexpr int kTopLevelThreads = 32;

struct GroupedIq2SharedStorage {
  alignas(16) int8_t activation_codes[kAssignmentTile * kGroupElements];
  uint32_t grid_words[kOutputTile];
  uint32_t scale_sign_words[kOutputTile];
  float weight_scales[kOutputTile];
  float activation_scales[kAssignmentTile];
  int32_t assignments[kAssignmentTile];
};

__device__ __forceinline__ uint32_t load_iq2_split_u32(const uint8_t* address) {
  const auto* halfwords = reinterpret_cast<const uint16_t*>(address);
  return static_cast<uint32_t>(halfwords[0]) |
         (static_cast<uint32_t>(halfwords[1]) << 16);
}

__device__ __forceinline__ int decode_iq2_xxs_fragment_register(
    uint32_t grid_word, uint32_t scale_sign_word, int chunk) {
  const int grid_part = chunk >> 1;
  const bool high_half = (chunk & 1) != 0;
  const uint8_t grid_index = static_cast<uint8_t>(grid_word >> (8 * grid_part));
  const uint64_t grid_values = iq2xxs_grid[grid_index];
  const uint32_t grid_half = high_half
                                 ? static_cast<uint32_t>(grid_values >> 32)
                                 : static_cast<uint32_t>(grid_values);
  const uint8_t sign_selector =
      static_cast<uint8_t>((scale_sign_word >> (7 * grid_part)) & 127);
  const uint32_t replicated_signs =
      static_cast<uint32_t>(ksigns_iq2xs[sign_selector]) * 0x01010101u;
  const uint32_t sign_mask = high_half ? 0x80402010u : 0x08040201u;
  const int sign_bytes = __vcmpne4(replicated_signs & sign_mask, 0);
  return __vsub4(static_cast<int>(grid_half) ^ sign_bytes, sign_bytes);
}

__device__ void load_iq2_xxs_weight_tile(const uint8_t* __restrict__ weights,
                                         int expert, int output_tile_start,
                                         int output_rows, int input_columns,
                                         int group_index,
                                         GroupedIq2SharedStorage& shared) {
  const int lane = threadIdx.x;
  const int blocks_per_row = input_columns / kIq2BlockElements;
  const int raw_row_bytes = blocks_per_row * kIq2BlockBytes;
  const int block_index = group_index / 8;
  const int group_in_block = group_index % 8;
  if (lane < kOutputTile) {
    const int output_row = output_tile_start + lane;
    if (output_row < output_rows) {
      const uint8_t* block =
          weights + (expert * output_rows + output_row) * raw_row_bytes +
          block_index * kIq2BlockBytes;
      const uint8_t* group = block + 2 + group_in_block * 8;
      shared.grid_words[lane] = load_iq2_split_u32(group);
      shared.scale_sign_words[lane] = load_iq2_split_u32(group + 4);
      const float block_scale =
          __half2float(*reinterpret_cast<const __half*>(block));
      shared.weight_scales[lane] =
          block_scale *
          static_cast<float>((shared.scale_sign_words[lane] >> 27) | 1) / 8.0f;
    } else {
      shared.grid_words[lane] = 0;
      shared.scale_sign_words[lane] = 0;
      shared.weight_scales[lane] = 0.0f;
    }
  }
  __syncwarp();
}

__device__ void run_iq2_xxs_mma_group(const uint8_t* __restrict__ weights,
                                      int expert, int output_tile_start,
                                      int output_rows, int input_columns,
                                      int group_index,
                                      GroupedIq2SharedStorage& shared,
                                      float (&sums)[4]) {
  load_iq2_xxs_weight_tile(weights, expert, output_tile_start, output_rows,
                           input_columns, group_index, shared);

  const int lane = threadIdx.x;
  const int row0 = lane / 4;
  const int row1 = row0 + 8;
  const int chunk = lane & 3;
  int weight_fragment[4] = {
      decode_iq2_xxs_fragment_register(shared.grid_words[row0],
                                       shared.scale_sign_words[row0], chunk),
      decode_iq2_xxs_fragment_register(shared.grid_words[row1],
                                       shared.scale_sign_words[row1], chunk),
      decode_iq2_xxs_fragment_register(
          shared.grid_words[row0], shared.scale_sign_words[row0], chunk + 4),
      decode_iq2_xxs_fragment_register(
          shared.grid_words[row1], shared.scale_sign_words[row1], chunk + 4),
  };
  int accumulator[4] = {};
  mma_int8_m16n8k32_row_col(weight_fragment, shared.activation_codes, lane,
                            accumulator);

#pragma unroll
  for (int local = 0; local < 4; ++local) {
    const int output_row = (local / 2) * 8 + lane / 4;
    const int assignment = (lane % 4) * 2 + local % 2;
    sums[local] = fmaf(
        static_cast<float>(accumulator[local]),
        shared.weight_scales[output_row] * shared.activation_scales[assignment],
        sums[local]);
  }
  __syncwarp();
}

__global__ void iq2_xxs_q8_1_grouped_gate_up_kernel(
    const __half* __restrict__ token_scales,
    const int8_t* __restrict__ token_codes,
    const uint8_t* __restrict__ gate_weights,
    const uint8_t* __restrict__ up_weights,
    const int32_t* __restrict__ sorted_token_ids,
    const int32_t* __restrict__ expert_ids,
    const int32_t* __restrict__ num_tokens_padded,
    float* __restrict__ gate_output, float* __restrict__ up_output,
    int token_count, int topk, int expert_count, int output_rows,
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
  const int total_assignments = token_count * topk;
  __shared__ GroupedIq2SharedStorage shared;
  const int lane = threadIdx.x;
  if (lane < kAssignmentTile) {
    const int assignment = sorted_token_ids[assignment_start + lane];
    shared.assignments[lane] = assignment;
  }
  __syncwarp();

  float gate_sums[4] = {};
  float up_sums[4] = {};
  const int group_count = input_columns / kGroupElements;
  for (int group_index = 0; group_index < group_count; ++group_index) {
    for (int index = lane; index < kAssignmentTile * kGroupElements;
         index += kTopLevelThreads) {
      const int assignment_slot = index / kGroupElements;
      const int element = index % kGroupElements;
      const int assignment = shared.assignments[assignment_slot];
      if (assignment < total_assignments) {
        const int token_index = assignment / topk;
        shared.activation_codes[index] =
            token_codes[token_index * input_columns +
                        group_index * kGroupElements + element];
      } else {
        shared.activation_codes[index] = 0;
      }
    }
    if (lane < kAssignmentTile) {
      const int assignment = shared.assignments[lane];
      shared.activation_scales[lane] =
          assignment < total_assignments
              ? __half2float(token_scales[(assignment / topk) * group_count +
                                          group_index])
              : 0.0f;
    }
    __syncwarp();
    run_iq2_xxs_mma_group(gate_weights, expert, output_tile_start, output_rows,
                          input_columns, group_index, shared, gate_sums);
    run_iq2_xxs_mma_group(up_weights, expert, output_tile_start, output_rows,
                          input_columns, group_index, shared, up_sums);
  }

#pragma unroll
  for (int local = 0; local < 4; ++local) {
    const int output_row_in_tile = (local / 2) * 8 + lane / 4;
    const int assignment_slot = (lane % 4) * 2 + local % 2;
    const int output_row = output_tile_start + output_row_in_tile;
    const int assignment = shared.assignments[assignment_slot];
    if (assignment < total_assignments && output_row < output_rows) {
      gate_output[assignment * output_rows + output_row] = gate_sums[local];
      up_output[assignment * output_rows + output_row] = up_sums[local];
    }
  }
}

}  // namespace

void gguf_iq2_xxs_q8_1_grouped_gate_up(
    const torch::stable::Tensor& token_scales,
    const torch::stable::Tensor& token_codes,
    const torch::stable::Tensor& gate_weights,
    const torch::stable::Tensor& up_weights,
    const torch::stable::Tensor& sorted_token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_tokens_padded,
    torch::stable::Tensor& gate_output, torch::stable::Tensor& up_output,
    int64_t topk) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &token_scales,      &token_codes,      &gate_weights,
      &up_weights,        &sorted_token_ids, &expert_ids,
      &num_tokens_padded, &gate_output,      &up_output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF grouped IQ2 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(
        tensor->get_device_index() == gate_output.get_device_index(),
        "GGUF grouped IQ2 tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(token_scales.scalar_type() == ScalarType::Half &&
                      token_codes.scalar_type() == ScalarType::Char &&
                      gate_weights.scalar_type() == ScalarType::Byte &&
                      up_weights.scalar_type() == ScalarType::Byte &&
                      sorted_token_ids.scalar_type() == ScalarType::Int &&
                      expert_ids.scalar_type() == ScalarType::Int &&
                      num_tokens_padded.scalar_type() == ScalarType::Int &&
                      gate_output.scalar_type() == ScalarType::Float &&
                      up_output.scalar_type() == ScalarType::Float,
                  "GGUF grouped IQ2 dtype contract mismatch");
  STD_TORCH_CHECK(token_codes.dim() == 2 && token_scales.dim() == 2 &&
                      gate_weights.dim() == 3 && up_weights.dim() == 3 &&
                      sorted_token_ids.dim() == 1 && expert_ids.dim() == 1 &&
                      num_tokens_padded.numel() == 1 &&
                      gate_output.dim() == 3 && up_output.dim() == 3,
                  "GGUF grouped IQ2 tensor rank mismatch");
  const int token_count = token_codes.size(0);
  const int input_columns = token_codes.size(1);
  const int expert_count = gate_weights.size(0);
  const int output_rows = gate_weights.size(1);
  STD_TORCH_CHECK(topk > 0 && gate_output.size(0) == token_count &&
                      gate_output.size(1) == topk &&
                      gate_output.size(2) == output_rows &&
                      up_output.sizes().equals(gate_output.sizes()),
                  "GGUF grouped IQ2 output/topk shape mismatch");
  STD_TORCH_CHECK(input_columns % kIq2BlockElements == 0 &&
                      gate_weights.size(2) ==
                          input_columns / kIq2BlockElements * kIq2BlockBytes &&
                      up_weights.sizes().equals(gate_weights.sizes()) &&
                      token_scales.size(0) == token_count &&
                      token_scales.size(1) == input_columns / kGroupElements,
                  "GGUF grouped IQ2 weight/activation shape mismatch");

  const int device_index = gate_output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const dim3 grid((output_rows + kOutputTile - 1) / kOutputTile,
                  expert_ids.size(0));
  iq2_xxs_q8_1_grouped_gate_up_kernel<<<grid, kTopLevelThreads, 0, stream>>>(
      reinterpret_cast<const __half*>(token_scales.const_data_ptr()),
      token_codes.const_data_ptr<int8_t>(),
      gate_weights.const_data_ptr<uint8_t>(),
      up_weights.const_data_ptr<uint8_t>(),
      sorted_token_ids.const_data_ptr<int32_t>(),
      expert_ids.const_data_ptr<int32_t>(),
      num_tokens_padded.const_data_ptr<int32_t>(),
      gate_output.mutable_data_ptr<float>(),
      up_output.mutable_data_ptr<float>(), token_count, static_cast<int>(topk),
      expert_count, output_rows, input_columns);
}

}  // namespace vllm::gguf_dsv4
