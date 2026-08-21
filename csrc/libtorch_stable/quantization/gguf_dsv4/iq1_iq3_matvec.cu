// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// IQ1_S, IQ1_M, and IQ3_XXS arithmetic follows the MIT-licensed
// Whamp/llama.cpp@0379cf4 GGML format definitions. The stable ABI,
// caller-owned Q8_1 contract, and indexed expert schedule are vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"
#include "iq1_iq3_tables.cuh"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kBlockElements = 256;
constexpr int kGroupsPerBlock = 8;
constexpr int kGroupElements = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;

enum class IqFormat { kIq1S, kIq1M, kIq3XXS };

template <IqFormat kFormat>
constexpr int block_bytes() {
  if constexpr (kFormat == IqFormat::kIq1S) {
    return 50;
  } else if constexpr (kFormat == IqFormat::kIq1M) {
    return 56;
  } else {
    return 98;
  }
}

__device__ __forceinline__ uint16_t load_u16(const uint8_t* address) {
  uint16_t value;
  memcpy(&value, address, sizeof(value));
  return value;
}

__device__ __forceinline__ uint32_t load_u32(const uint8_t* address) {
  uint32_t value;
  memcpy(&value, address, sizeof(value));
  return value;
}

__device__ __forceinline__ float half_bits_to_float(uint16_t bits) {
  __half value;
  memcpy(&value, &bits, sizeof(value));
  return __half2float(value);
}

__device__ __forceinline__ float warp_sum_iq(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ int sum_q8_codes(const int* code_packs) {
  int sum = 0;
#pragma unroll
  for (int index = 0; index < 8; ++index) {
    sum = __dp4a(code_packs[index], 0x01010101, sum);
  }
  return sum;
}

// Grid words are consumed through the pre-split parity tables
// (kIq1SGridEven/kIq1SGridOdd) so the per-group nibble extraction is done
// once at table-generation time instead of per dot product.
__device__ __forceinline__ int iq1_grid_dot(uint32_t table_index,
                                            const int* code_packs,
                                            int first_pack) {
  int sum = __dp4a(static_cast<int>(kIq1SGridEven[table_index]),
                   code_packs[first_pack], 0);
  return __dp4a(static_cast<int>(kIq1SGridOdd[table_index]),
                code_packs[first_pack + 1], sum);
}

__device__ __forceinline__ float iq1_s_group_dot(const uint8_t* block,
                                                 int group_index,
                                                 const int* code_packs) {
  const float block_scale =
      __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* grid_indices = block + 2 + group_index * 4;
  const uint16_t high = load_u16(block + 34 + group_index * 2);
  int integer_sum = 0;
#pragma unroll
  for (int grid_index = 0; grid_index < 4; ++grid_index) {
    const int table_index =
        grid_indices[grid_index] | (((high >> (3 * grid_index)) & 7) << 8);
    integer_sum += iq1_grid_dot(table_index, code_packs, 2 * grid_index);
  }
  const int scale = 2 * ((high >> 12) & 7) + 1;
  const float delta = (high & 0x8000) != 0 ? -1.125f : -0.875f;
  return block_scale * static_cast<float>(scale) *
         (static_cast<float>(integer_sum) +
          delta * static_cast<float>(sum_q8_codes(code_packs)));
}

__device__ __forceinline__ float iq1_m_group_dot(const uint8_t* block,
                                                 int group_index,
                                                 const int* code_packs) {
  const uint8_t* grid_indices = block + group_index * 4;
  const uint8_t* high_bytes = block + 32 + group_index * 2;
  const uint8_t* scales = block + 48;
  int integer_sums[2] = {0, 0};
  float deltas[4];
#pragma unroll
  for (int grid_index = 0; grid_index < 4; ++grid_index) {
    const int high_shift = 4 * ((grid_index) % 2);
    const uint8_t high = high_bytes[grid_index / 2] >> high_shift;
    const int table_index = grid_indices[grid_index] | ((high & 7) << 8);
    const int half_index = grid_index / 2;
    integer_sums[half_index] +=
        iq1_grid_dot(table_index, code_packs, 2 * grid_index);
    deltas[grid_index] = (high & 8) != 0 ? -1.125f : -0.875f;
  }
  const uint16_t scale_words[4] = {load_u16(scales), load_u16(scales + 2),
                                   load_u16(scales + 4), load_u16(scales + 6)};
  const uint16_t global_scale_bits =
      (scale_words[0] >> 12) | ((scale_words[1] >> 8) & 0x00f0) |
      ((scale_words[2] >> 4) & 0x0f00) | (scale_words[3] & 0xf000);
  const int packed_scales =
      scale_words[group_index / 2] >> (6 * (group_index % 2));
  const int first_scale = 2 * (packed_scales & 7) + 1;
  const int second_scale = 2 * ((packed_scales >> 3) & 7) + 1;
  const float first_sum =
      static_cast<float>(integer_sums[0]) +
      deltas[0] * static_cast<float>(__dp4a(code_packs[0], 0x01010101, 0) +
                                     __dp4a(code_packs[1], 0x01010101, 0)) +
      deltas[1] * static_cast<float>(__dp4a(code_packs[2], 0x01010101, 0) +
                                     __dp4a(code_packs[3], 0x01010101, 0));
  const float second_sum =
      static_cast<float>(integer_sums[1]) +
      deltas[2] * static_cast<float>(__dp4a(code_packs[4], 0x01010101, 0) +
                                     __dp4a(code_packs[5], 0x01010101, 0)) +
      deltas[3] * static_cast<float>(__dp4a(code_packs[6], 0x01010101, 0) +
                                     __dp4a(code_packs[7], 0x01010101, 0));
  return half_bits_to_float(global_scale_bits) *
         (first_scale * first_sum + second_scale * second_sum);
}

__device__ __forceinline__ int signed_grid_pack(uint32_t grid, uint8_t signs,
                                                bool high_half) {
  const uint32_t replicated = static_cast<uint32_t>(signs) * 0x01010101U;
  const uint32_t bit_selector = high_half ? 0x80402010U : 0x08040201U;
  const int sign_mask = __vcmpne4(replicated & bit_selector, 0);
  return __vsub4(static_cast<int>(grid) ^ sign_mask, sign_mask);
}

__device__ __forceinline__ float iq3_xxs_group_dot(const uint8_t* block,
                                                   int group_index,
                                                   const int* code_packs) {
  const float block_scale =
      __half2float(*reinterpret_cast<const __half*>(block));
  const uint8_t* indices = block + 2 + group_index * 8;
  uint32_t scale_signs = load_u32(block + 2 + 64 + group_index * 4);
  int integer_sum = 0;
#pragma unroll
  for (int part = 0; part < 4; ++part) {
    const uint8_t signs = kIqSigns[(scale_signs >> (7 * part)) & 127];
    const int first_grid =
        signed_grid_pack(kIq3XXSGrid[indices[2 * part]], signs, false);
    const int second_grid =
        signed_grid_pack(kIq3XXSGrid[indices[2 * part + 1]], signs, true);
    integer_sum = __dp4a(first_grid, code_packs[2 * part], integer_sum);
    integer_sum = __dp4a(second_grid, code_packs[2 * part + 1], integer_sum);
  }
  const int scale = static_cast<int>(scale_signs >> 28);
  return block_scale * (0.5f + static_cast<float>(scale)) * 0.5f *
         static_cast<float>(integer_sum);
}

template <IqFormat kFormat>
__device__ __forceinline__ float iq_q8_1_dot(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ weight_row, int activation_row,
    int input_columns) {
  const int lane = threadIdx.x & 31;
  const int group_count = input_columns / kGroupElements;
  float partial = 0.0f;
  for (int activation_group = lane; activation_group < group_count;
       activation_group += 32) {
    const int block_index = activation_group / kGroupsPerBlock;
    const int group_index = activation_group % kGroupsPerBlock;
    const uint8_t* block = weight_row + block_index * block_bytes<kFormat>();
    const int* code_packs = reinterpret_cast<const int*>(
        activation_codes + activation_row * input_columns +
        activation_group * kGroupElements);
    float group_sum;
    if constexpr (kFormat == IqFormat::kIq1S) {
      group_sum = iq1_s_group_dot(block, group_index, code_packs);
    } else if constexpr (kFormat == IqFormat::kIq1M) {
      group_sum = iq1_m_group_dot(block, group_index, code_packs);
    } else {
      group_sum = iq3_xxs_group_dot(block, group_index, code_packs);
    }
    const float activation_scale = __half2float(
        activation_scales[activation_row * group_count + activation_group]);
    partial = fmaf(activation_scale, group_sum, partial);
  }
  return warp_sum_iq(partial);
}

template <IqFormat kFormat>
__global__ void iq_q8_1_matvec_kernel(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ weights, float* __restrict__ output,
    int token_count, int output_rows, int input_columns, int raw_row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int output_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (output_index >= token_count * output_rows) {
    return;
  }
  const int token_index = output_index / output_rows;
  const int output_row = output_index % output_rows;
  const float sum = iq_q8_1_dot<kFormat>(activation_scales, activation_codes,
                                         weights + output_row * raw_row_bytes,
                                         token_index, input_columns);
  if (lane == 0) {
    output[output_index] = sum;
  }
}

template <IqFormat kFormat>
__global__ void iq_q8_1_indexed_gate_up_kernel(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ gate_weights,
    const uint8_t* __restrict__ up_weights,
    const int32_t* __restrict__ topk_ids, float* __restrict__ gate_output,
    float* __restrict__ up_output, int token_count, int topk, int expert_count,
    int output_rows, int input_columns, int raw_row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  int work_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  const int total_work = token_count * topk * 2 * output_rows;
  if (work_index >= total_work) {
    return;
  }
  const int output_row = work_index % output_rows;
  work_index /= output_rows;
  const int projection = work_index & 1;
  work_index >>= 1;
  const int slot = work_index % topk;
  const int token_index = work_index / topk;
  const int expert = topk_ids[token_index * topk + slot];
  if (expert < 0 || expert >= expert_count) {
    return;
  }
  const uint8_t* projection_weights =
      projection == 0 ? gate_weights : up_weights;
  const uint8_t* weight_row =
      projection_weights + (expert * output_rows + output_row) * raw_row_bytes;
  const float sum =
      iq_q8_1_dot<kFormat>(activation_scales, activation_codes, weight_row,
                           token_index, input_columns);
  if (lane == 0) {
    const int output_index =
        (token_index * topk + slot) * output_rows + output_row;
    if (projection == 0) {
      gate_output[output_index] = sum;
    } else {
      up_output[output_index] = sum;
    }
  }
}

template <IqFormat kFormat>
__global__ void iq_q8_1_indexed_down_kernel(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ down_weights,
    const int32_t* __restrict__ topk_ids, float* __restrict__ output,
    int token_count, int topk, int expert_count, int output_rows,
    int input_columns, int raw_row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  int output_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (output_index >= token_count * topk * output_rows) {
    return;
  }
  const int output_row = output_index % output_rows;
  output_index /= output_rows;
  const int slot = output_index % topk;
  const int token_index = output_index / topk;
  const int expert = topk_ids[token_index * topk + slot];
  if (expert < 0 || expert >= expert_count) {
    return;
  }
  const uint8_t* weight_row =
      down_weights + (expert * output_rows + output_row) * raw_row_bytes;
  const int activation_row = token_index * topk + slot;
  const float sum =
      iq_q8_1_dot<kFormat>(activation_scales, activation_codes, weight_row,
                           activation_row, input_columns);
  if (lane == 0) {
    output[(token_index * topk + slot) * output_rows + output_row] = sum;
  }
}

template <IqFormat kFormat>
void check_q8_1_inputs(const torch::stable::Tensor& activation_scales,
                       const torch::stable::Tensor& activation_codes,
                       const torch::stable::Tensor& weights,
                       const torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &activation_scales, &activation_codes, &weights, &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF IQ1/IQ3 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(tensor->get_device_index() == output.get_device_index(),
                    "GGUF IQ1/IQ3 tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(activation_scales.scalar_type() == ScalarType::Half &&
                      activation_codes.scalar_type() == ScalarType::Char &&
                      weights.scalar_type() == ScalarType::Byte &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF IQ1/IQ3 dtype contract mismatch");
  const int input_columns = activation_codes.size(-1);
  STD_TORCH_CHECK(input_columns % kBlockElements == 0,
                  "GGUF IQ1/IQ3 input columns must be divisible by 256");
  STD_TORCH_CHECK(activation_scales.size(-1) == input_columns / kGroupElements,
                  "GGUF IQ1/IQ3 Q8_1 scale shape mismatch");
  STD_TORCH_CHECK(weights.size(-1) ==
                      input_columns / kBlockElements * block_bytes<kFormat>(),
                  "GGUF IQ1/IQ3 raw weight row size mismatch");
}

template <IqFormat kFormat>
void launch_raw_matvec(const torch::stable::Tensor& activation_scales,
                       const torch::stable::Tensor& activation_codes,
                       const torch::stable::Tensor& weights,
                       torch::stable::Tensor& output) {
  check_q8_1_inputs<kFormat>(activation_scales, activation_codes, weights,
                             output);
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      weights.dim() == 2 && output.dim() == 2,
                  "GGUF IQ1/IQ3 raw matvec tensor rank mismatch");
  const int token_count = activation_codes.size(0);
  const int input_columns = activation_codes.size(1);
  const int output_rows = weights.size(0);
  STD_TORCH_CHECK(activation_scales.size(0) == token_count &&
                      output.size(0) == token_count &&
                      output.size(1) == output_rows,
                  "GGUF IQ1/IQ3 raw matvec output shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  iq_q8_1_matvec_kernel<kFormat><<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
      reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
      weights.const_data_ptr<uint8_t>(), output.mutable_data_ptr<float>(),
      token_count, output_rows, input_columns, weights.size(1));
}

template <IqFormat kFormat>
void launch_indexed_gate_up(const torch::stable::Tensor& activation_scales,
                            const torch::stable::Tensor& activation_codes,
                            const torch::stable::Tensor& gate_weights,
                            const torch::stable::Tensor& up_weights,
                            const torch::stable::Tensor& topk_ids,
                            torch::stable::Tensor& gate_output,
                            torch::stable::Tensor& up_output) {
  using torch::headeronly::ScalarType;
  check_q8_1_inputs<kFormat>(activation_scales, activation_codes, gate_weights,
                             gate_output);
  STD_TORCH_CHECK(up_weights.device().is_cuda() && up_weights.is_contiguous() &&
                      topk_ids.device().is_cuda() && topk_ids.is_contiguous() &&
                      up_output.device().is_cuda() && up_output.is_contiguous(),
                  "GGUF IQ1 gate/up tensors must be contiguous CUDA tensors");
  STD_TORCH_CHECK(up_weights.scalar_type() == ScalarType::Byte &&
                      topk_ids.scalar_type() == ScalarType::Int &&
                      up_output.scalar_type() == ScalarType::Float,
                  "GGUF IQ1 gate/up dtype contract mismatch");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      gate_weights.dim() == 3 && up_weights.dim() == 3 &&
                      topk_ids.dim() == 2 && gate_output.dim() == 3 &&
                      up_output.dim() == 3,
                  "GGUF IQ1 gate/up tensor rank mismatch");
  STD_TORCH_CHECK(gate_weights.sizes().equals(up_weights.sizes()) &&
                      gate_output.sizes().equals(up_output.sizes()),
                  "GGUF IQ1 gate/up paired shapes mismatch");
  const int token_count = topk_ids.size(0);
  const int topk = topk_ids.size(1);
  const int expert_count = gate_weights.size(0);
  const int output_rows = gate_weights.size(1);
  const int input_columns = activation_codes.size(1);
  STD_TORCH_CHECK(activation_codes.size(0) == token_count &&
                      activation_scales.size(0) == token_count &&
                      gate_output.size(0) == token_count &&
                      gate_output.size(1) == topk &&
                      gate_output.size(2) == output_rows,
                  "GGUF IQ1 gate/up activation/output shape mismatch");
  const int device_index = gate_output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * topk * 2 * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  iq_q8_1_indexed_gate_up_kernel<kFormat>
      <<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
          reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
          gate_weights.const_data_ptr<uint8_t>(),
          up_weights.const_data_ptr<uint8_t>(),
          topk_ids.const_data_ptr<int32_t>(),
          gate_output.mutable_data_ptr<float>(),
          up_output.mutable_data_ptr<float>(), token_count, topk, expert_count,
          output_rows, input_columns, gate_weights.size(2));
}

template <IqFormat kFormat>
void launch_indexed_down(const torch::stable::Tensor& activation_scales,
                         const torch::stable::Tensor& activation_codes,
                         const torch::stable::Tensor& down_weights,
                         const torch::stable::Tensor& topk_ids,
                         torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  check_q8_1_inputs<kFormat>(activation_scales, activation_codes, down_weights,
                             output);
  STD_TORCH_CHECK(topk_ids.device().is_cuda() && topk_ids.is_contiguous(),
                  "GGUF IQ3 top-k IDs must be a contiguous CUDA tensor");
  STD_TORCH_CHECK(topk_ids.scalar_type() == ScalarType::Int,
                  "GGUF IQ3 top-k IDs must be int32");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      down_weights.dim() == 3 && topk_ids.dim() == 2 &&
                      output.dim() == 3,
                  "GGUF IQ3 indexed down tensor rank mismatch");
  const int token_count = topk_ids.size(0);
  const int topk = topk_ids.size(1);
  const int expert_count = down_weights.size(0);
  const int output_rows = down_weights.size(1);
  const int input_columns = activation_codes.size(1);
  STD_TORCH_CHECK(activation_codes.size(0) == token_count * topk &&
                      activation_scales.size(0) == token_count * topk &&
                      output.size(0) == token_count && output.size(1) == topk &&
                      output.size(2) == output_rows,
                  "GGUF IQ3 indexed down activation/output shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * topk * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  iq_q8_1_indexed_down_kernel<kFormat>
      <<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
          reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
          down_weights.const_data_ptr<uint8_t>(),
          topk_ids.const_data_ptr<int32_t>(), output.mutable_data_ptr<float>(),
          token_count, topk, expert_count, output_rows, input_columns,
          down_weights.size(2));
}

}  // namespace

void gguf_iq1_s_q8_1_raw_matvec(const torch::stable::Tensor& activation_scales,
                                const torch::stable::Tensor& activation_codes,
                                const torch::stable::Tensor& weights,
                                torch::stable::Tensor& output) {
  launch_raw_matvec<IqFormat::kIq1S>(activation_scales, activation_codes,
                                     weights, output);
}

void gguf_iq1_m_q8_1_raw_matvec(const torch::stable::Tensor& activation_scales,
                                const torch::stable::Tensor& activation_codes,
                                const torch::stable::Tensor& weights,
                                torch::stable::Tensor& output) {
  launch_raw_matvec<IqFormat::kIq1M>(activation_scales, activation_codes,
                                     weights, output);
}

void gguf_iq3_xxs_q8_1_raw_matvec(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& weights, torch::stable::Tensor& output) {
  launch_raw_matvec<IqFormat::kIq3XXS>(activation_scales, activation_codes,
                                       weights, output);
}

void gguf_iq1_s_q8_1_indexed_gate_up(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& gate_weights,
    const torch::stable::Tensor& up_weights,
    const torch::stable::Tensor& topk_ids, torch::stable::Tensor& gate_output,
    torch::stable::Tensor& up_output) {
  launch_indexed_gate_up<IqFormat::kIq1S>(activation_scales, activation_codes,
                                          gate_weights, up_weights, topk_ids,
                                          gate_output, up_output);
}

void gguf_iq1_m_q8_1_indexed_gate_up(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& gate_weights,
    const torch::stable::Tensor& up_weights,
    const torch::stable::Tensor& topk_ids, torch::stable::Tensor& gate_output,
    torch::stable::Tensor& up_output) {
  launch_indexed_gate_up<IqFormat::kIq1M>(activation_scales, activation_codes,
                                          gate_weights, up_weights, topk_ids,
                                          gate_output, up_output);
}

void gguf_iq3_xxs_q8_1_indexed_down(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& down_weights,
    const torch::stable::Tensor& topk_ids, torch::stable::Tensor& output) {
  launch_indexed_down<IqFormat::kIq3XXS>(activation_scales, activation_codes,
                                         down_weights, topk_ids, output);
}

}  // namespace vllm::gguf_dsv4
