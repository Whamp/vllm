// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// IQ1 format arithmetic follows the MIT-licensed Whamp/llama.cpp@0379cf4
// tables. The block-8 SM86 IMMA schedule, stable ABI, and caller-owned
// alignment contract are vLLM code.

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

enum class Iq1Format { kIq1S, kIq1M };

template <Iq1Format kFormat>
constexpr int block_bytes() {
  return kFormat == Iq1Format::kIq1S ? 50 : 56;
}

struct GroupedIq1SharedStorage {
  alignas(16) int8_t activation_codes[kAssignmentTile * kGroupElements];
  // Pre-split parity words looked up from kIq1SGridEven/kIq1SGridOdd so the
  // per-fragment nibble extraction never runs inside the MMA loop.
  uint32_t grid_words_even[kOutputTile * 4];
  uint32_t grid_words_odd[kOutputTile * 4];
  int8_t scale_codes[kOutputTile * 4];
  float delta_corrections[kOutputTile * 4];
  float block_scales[kOutputTile];
  float activation_scales[kAssignmentTile];
  int32_t code_sums[kAssignmentTile * 4];
  int32_t assignments[kAssignmentTile];
};

__device__ __forceinline__ uint16_t load_iq1_u16(const uint8_t* address) {
  uint16_t value;
  memcpy(&value, address, sizeof(value));
  return value;
}

__device__ __forceinline__ float iq1_half_bits_to_float(uint16_t bits) {
  __half value;
  memcpy(&value, &bits, sizeof(value));
  return __half2float(value);
}

__device__ __forceinline__ int multiply_packed_bytes(int packed, int scale) {
  uint32_t result = 0;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const int value = (packed >> (8 * index)) & 15;
    result |= static_cast<uint32_t>(value * scale) << (8 * index);
  }
  return static_cast<int>(result);
}

__device__ __forceinline__ int decode_iq1_fragment_register(
    const GroupedIq1SharedStorage& shared, int output_row, int chunk) {
  const int part = chunk >> 1;
  const int parity = chunk & 1;
  const uint32_t word = parity != 0
                            ? shared.grid_words_odd[output_row * 4 + part]
                            : shared.grid_words_even[output_row * 4 + part];
  return multiply_packed_bytes(static_cast<int>(word),
                               shared.scale_codes[output_row * 4 + part]);
}

template <Iq1Format kFormat>
__device__ void load_iq1_weight_tile(const uint8_t* __restrict__ weights,
                                     int expert, int output_tile_start,
                                     int output_rows, int input_columns,
                                     int group_index,
                                     GroupedIq1SharedStorage& shared) {
  const int lane = threadIdx.x;
  if (lane < kOutputTile) {
    const int output_row = output_tile_start + lane;
    if (output_row >= output_rows) {
#pragma unroll
      for (int part = 0; part < 4; ++part) {
        shared.grid_words_even[lane * 4 + part] = 0;
        shared.grid_words_odd[lane * 4 + part] = 0;
        shared.scale_codes[lane * 4 + part] = 0;
        shared.delta_corrections[lane * 4 + part] = 0.0f;
      }
      shared.block_scales[lane] = 0.0f;
    } else {
      const int blocks_per_row = input_columns / kBlockElements;
      const int raw_row_bytes = blocks_per_row * block_bytes<kFormat>();
      const int block_index = group_index / 8;
      const int group_in_block = group_index % 8;
      const uint8_t* block =
          weights + (expert * output_rows + output_row) * raw_row_bytes +
          block_index * block_bytes<kFormat>();
      if constexpr (kFormat == Iq1Format::kIq1S) {
        const uint8_t* indices = block + 2 + group_in_block * 4;
        const uint16_t high = load_iq1_u16(block + 34 + group_in_block * 2);
        const int scale = 2 * ((high >> 12) & 7) + 1;
        const float delta = (high & 0x8000) != 0 ? -1.125f : -0.875f;
        shared.block_scales[lane] =
            __half2float(*reinterpret_cast<const __half*>(block));
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          const int table_index =
              indices[part] | (((high >> (3 * part)) & 7) << 8);
          shared.grid_words_even[lane * 4 + part] = kIq1SGridEven[table_index];
          shared.grid_words_odd[lane * 4 + part] = kIq1SGridOdd[table_index];
          shared.scale_codes[lane * 4 + part] = scale;
          shared.delta_corrections[lane * 4 + part] = delta * scale;
        }
      } else {
        const uint8_t* indices = block + group_in_block * 4;
        const uint8_t* high_bytes = block + 32 + group_in_block * 2;
        const uint8_t* scales = block + 48;
        const uint16_t scale_words[4] = {
            load_iq1_u16(scales), load_iq1_u16(scales + 2),
            load_iq1_u16(scales + 4), load_iq1_u16(scales + 6)};
        const uint16_t global_scale_bits =
            (scale_words[0] >> 12) | ((scale_words[1] >> 8) & 0x00f0) |
            ((scale_words[2] >> 4) & 0x0f00) | (scale_words[3] & 0xf000);
        const int packed_scales =
            scale_words[group_in_block / 2] >> (6 * (group_in_block % 2));
        const int half_scales[2] = {2 * (packed_scales & 7) + 1,
                                    2 * ((packed_scales >> 3) & 7) + 1};
        shared.block_scales[lane] = iq1_half_bits_to_float(global_scale_bits);
#pragma unroll
        for (int part = 0; part < 4; ++part) {
          const uint8_t high = high_bytes[part / 2] >> (4 * (part % 2));
          const int table_index = indices[part] | ((high & 7) << 8);
          const int scale = half_scales[part / 2];
          const float delta = (high & 8) != 0 ? -1.125f : -0.875f;
          shared.grid_words_even[lane * 4 + part] = kIq1SGridEven[table_index];
          shared.grid_words_odd[lane * 4 + part] = kIq1SGridOdd[table_index];
          shared.scale_codes[lane * 4 + part] = scale;
          shared.delta_corrections[lane * 4 + part] = delta * scale;
        }
      }
    }
  }
  __syncwarp();
}

template <Iq1Format kFormat>
__device__ void run_iq1_mma_group(const uint8_t* __restrict__ weights,
                                  int expert, int output_tile_start,
                                  int output_rows, int input_columns,
                                  int group_index,
                                  GroupedIq1SharedStorage& shared,
                                  float (&sums)[4]) {
  load_iq1_weight_tile<kFormat>(weights, expert, output_tile_start, output_rows,
                                input_columns, group_index, shared);
  const int lane = threadIdx.x;
  const int row0 = lane / 4;
  const int row1 = row0 + 8;
  const int chunk = lane & 3;
  int weight_fragment[4] = {
      decode_iq1_fragment_register(shared, row0, chunk),
      decode_iq1_fragment_register(shared, row1, chunk),
      decode_iq1_fragment_register(shared, row0, chunk + 4),
      decode_iq1_fragment_register(shared, row1, chunk + 4),
  };
  int accumulator[4] = {};
  mma_int8_m16n8k32_row_col(weight_fragment, shared.activation_codes, lane,
                            accumulator);
#pragma unroll
  for (int local = 0; local < 4; ++local) {
    const int output_row = (local / 2) * 8 + lane / 4;
    const int assignment = (lane % 4) * 2 + local % 2;
    float correction = 0.0f;
#pragma unroll
    for (int part = 0; part < 4; ++part) {
      correction += shared.delta_corrections[output_row * 4 + part] *
                    shared.code_sums[assignment * 4 + part];
    }
    sums[local] = fmaf(
        static_cast<float>(accumulator[local]) + correction,
        shared.block_scales[output_row] * shared.activation_scales[assignment],
        sums[local]);
  }
  __syncwarp();
}

template <Iq1Format kFormat>
__global__ void iq1_q8_1_grouped_gate_up_kernel(
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
  const int assignment_start = blockIdx.y * kAssignmentTile;
  if (assignment_start >= *num_tokens_padded) {
    return;
  }
  const int expert = expert_ids[blockIdx.y];
  if (expert < 0 || expert >= expert_count) {
    return;
  }
  const int output_tile_start = blockIdx.x * kOutputTile;
  const int total_assignments = token_count * topk;
  __shared__ GroupedIq1SharedStorage shared;
  const int lane = threadIdx.x;
  if (lane < kAssignmentTile) {
    shared.assignments[lane] = sorted_token_ids[assignment_start + lane];
  }
  __syncwarp();

  float gate_sums[4] = {};
  float up_sums[4] = {};
  const int group_count = input_columns / kGroupElements;
  for (int group_index = 0; group_index < group_count; ++group_index) {
    for (int index = lane; index < kAssignmentTile * kGroupElements;
         index += kThreads) {
      const int assignment_slot = index / kGroupElements;
      const int element = index % kGroupElements;
      const int assignment = shared.assignments[assignment_slot];
      shared.activation_codes[index] =
          assignment < total_assignments
              ? token_codes[(assignment / topk) * input_columns +
                            group_index * kGroupElements + element]
              : 0;
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
    const int sum_assignment = lane / 4;
    const int sum_part = lane % 4;
    const int assignment = shared.assignments[sum_assignment];
    int code_sum = 0;
    if (assignment < total_assignments) {
      const int8_t* codes = shared.activation_codes +
                            sum_assignment * kGroupElements + sum_part * 8;
#pragma unroll
      for (int index = 0; index < 8; ++index) {
        code_sum += codes[index];
      }
    }
    shared.code_sums[sum_assignment * 4 + sum_part] = code_sum;
    __syncwarp();
    run_iq1_mma_group<kFormat>(gate_weights, expert, output_tile_start,
                               output_rows, input_columns, group_index, shared,
                               gate_sums);
    run_iq1_mma_group<kFormat>(up_weights, expert, output_tile_start,
                               output_rows, input_columns, group_index, shared,
                               up_sums);
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

template <Iq1Format kFormat>
void launch_iq1_grouped_gate_up(const torch::stable::Tensor& token_scales,
                                const torch::stable::Tensor& token_codes,
                                const torch::stable::Tensor& gate_weights,
                                const torch::stable::Tensor& up_weights,
                                const torch::stable::Tensor& sorted_token_ids,
                                const torch::stable::Tensor& expert_ids,
                                const torch::stable::Tensor& num_tokens_padded,
                                torch::stable::Tensor& gate_output,
                                torch::stable::Tensor& up_output,
                                int64_t topk) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &token_scales,      &token_codes,      &gate_weights,
      &up_weights,        &sorted_token_ids, &expert_ids,
      &num_tokens_padded, &gate_output,      &up_output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF grouped IQ1 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(
        tensor->get_device_index() == gate_output.get_device_index(),
        "GGUF grouped IQ1 tensors must share one CUDA device");
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
                  "GGUF grouped IQ1 dtype contract mismatch");
  STD_TORCH_CHECK(token_codes.dim() == 2 && token_scales.dim() == 2 &&
                      gate_weights.dim() == 3 && up_weights.dim() == 3 &&
                      sorted_token_ids.dim() == 1 && expert_ids.dim() == 1 &&
                      num_tokens_padded.numel() == 1 &&
                      gate_output.dim() == 3 && up_output.dim() == 3,
                  "GGUF grouped IQ1 tensor rank mismatch");
  const int token_count = token_codes.size(0);
  const int input_columns = token_codes.size(1);
  const int expert_count = gate_weights.size(0);
  const int output_rows = gate_weights.size(1);
  STD_TORCH_CHECK(topk > 0 && gate_output.size(0) == token_count &&
                      gate_output.size(1) == topk &&
                      gate_output.size(2) == output_rows &&
                      up_output.sizes().equals(gate_output.sizes()),
                  "GGUF grouped IQ1 output/topk shape mismatch");
  STD_TORCH_CHECK(input_columns % kBlockElements == 0 &&
                      gate_weights.size(2) == input_columns / kBlockElements *
                                                  block_bytes<kFormat>() &&
                      up_weights.sizes().equals(gate_weights.sizes()) &&
                      token_scales.size(0) == token_count &&
                      token_scales.size(1) == input_columns / kGroupElements,
                  "GGUF grouped IQ1 weight/activation shape mismatch");
  const int device_index = gate_output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const dim3 grid((output_rows + kOutputTile - 1) / kOutputTile,
                  expert_ids.size(0));
  iq1_q8_1_grouped_gate_up_kernel<kFormat><<<grid, kThreads, 0, stream>>>(
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

}  // namespace

void gguf_iq1_s_q8_1_grouped_gate_up(
    const torch::stable::Tensor& token_scales,
    const torch::stable::Tensor& token_codes,
    const torch::stable::Tensor& gate_weights,
    const torch::stable::Tensor& up_weights,
    const torch::stable::Tensor& sorted_token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_tokens_padded,
    torch::stable::Tensor& gate_output, torch::stable::Tensor& up_output,
    int64_t topk) {
  launch_iq1_grouped_gate_up<Iq1Format::kIq1S>(
      token_scales, token_codes, gate_weights, up_weights, sorted_token_ids,
      expert_ids, num_tokens_padded, gate_output, up_output, topk);
}

void gguf_iq1_m_q8_1_grouped_gate_up(
    const torch::stable::Tensor& token_scales,
    const torch::stable::Tensor& token_codes,
    const torch::stable::Tensor& gate_weights,
    const torch::stable::Tensor& up_weights,
    const torch::stable::Tensor& sorted_token_ids,
    const torch::stable::Tensor& expert_ids,
    const torch::stable::Tensor& num_tokens_padded,
    torch::stable::Tensor& gate_output, torch::stable::Tensor& up_output,
    int64_t topk) {
  launch_iq1_grouped_gate_up<Iq1Format::kIq1M>(
      token_scales, token_codes, gate_weights, up_weights, sorted_token_ids,
      expert_ids, num_tokens_padded, gate_output, up_output, topk);
}

}  // namespace vllm::gguf_dsv4
