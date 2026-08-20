// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Q2_K format math follows the MIT-licensed llama.cpp/GGML implementation
// and antirez/ds4@84cc882 cuda/mmq/test/proto_m2_q2k.cu. Kernel integration
// and stable-ABI contracts are original vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kQ2BlockElements = 256;
constexpr int kQ2BlockBytes = 84;
constexpr int kQ8GroupElements = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;

__device__ __forceinline__ float warp_sum_q2(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ float q2_k_q8_1_dot(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ weight_row, int activation_row,
    int input_columns) {
  const int lane = threadIdx.x & 31;
  const int blocks_per_row = input_columns / kQ2BlockElements;
  const int block_index = lane >> 4;
  const int iqs = lane & 15;
  float partial = 0.0f;
  if (block_index < blocks_per_row) {
    const uint8_t* block = weight_row + block_index * kQ2BlockBytes;
    const int packed_q2 = *reinterpret_cast<const int*>(block + 16 + iqs * 4);
    const int q8_block_base = block_index * 8 + 4 * (iqs / 8);
    const int q8_pack_index = iqs & 7;
    const int scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
    float scaled_sum = 0.0f;
    float min_sum = 0.0f;
#pragma unroll
    for (int q8_block = 0; q8_block < 4; ++q8_block) {
      const int group_index = q8_block_base + q8_block;
      const int* code_packs = reinterpret_cast<const int*>(
          activation_codes + activation_row * input_columns +
          group_index * kQ8GroupElements);
      const int codes = code_packs[q8_pack_index];
      const float activation_scale =
          __half2float(activation_scales[activation_row * (input_columns / 32) +
                                         group_index]);
      const int packed_2bit = (packed_q2 >> (2 * q8_block)) & 0x03030303;
      const int packed_scale = block[scale_offset + 2 * q8_block];
      scaled_sum +=
          activation_scale * static_cast<float>(__dp4a(packed_2bit, codes, 0) *
                                                (packed_scale & 0xF));
      int packed_min = packed_scale >> 4;
      packed_min |= packed_min << 8;
      packed_min |= packed_min << 16;
      min_sum +=
          activation_scale * static_cast<float>(__dp4a(packed_min, codes, 0));
    }
    const __half2 weight_scales = *reinterpret_cast<const __half2*>(block + 80);
    const float2 decoded_scales = __half22float2(weight_scales);
    partial = decoded_scales.x * scaled_sum - decoded_scales.y * min_sum;
  }
  return warp_sum_q2(partial);
}

__global__ void q2_k_q8_1_indexed_down_kernel(
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
  const float sum = q2_k_q8_1_dot(activation_scales, activation_codes,
                                  weight_row, activation_row, input_columns);
  if (lane == 0) {
    output[(token_index * topk + slot) * output_rows + output_row] = sum;
  }
}

}  // namespace

void gguf_q2_k_q8_1_indexed_down(const torch::stable::Tensor& activation_scales,
                                 const torch::stable::Tensor& activation_codes,
                                 const torch::stable::Tensor& down_weights,
                                 const torch::stable::Tensor& topk_ids,
                                 torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &activation_scales, &activation_codes, &down_weights, &topk_ids, &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF Q2_K tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(tensor->get_device_index() == output.get_device_index(),
                    "GGUF Q2_K tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(activation_scales.scalar_type() == ScalarType::Half &&
                      activation_codes.scalar_type() == ScalarType::Char &&
                      down_weights.scalar_type() == ScalarType::Byte &&
                      topk_ids.scalar_type() == ScalarType::Int &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF Q2_K dtype contract mismatch");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      down_weights.dim() == 3 && topk_ids.dim() == 2 &&
                      output.dim() == 3,
                  "GGUF Q2_K tensor rank mismatch");
  const int token_count = topk_ids.size(0);
  const int topk = topk_ids.size(1);
  const int expert_count = down_weights.size(0);
  const int output_rows = down_weights.size(1);
  const int raw_row_bytes = down_weights.size(2);
  const int input_columns = activation_codes.size(1);
  STD_TORCH_CHECK(
      input_columns % kQ2BlockElements == 0 &&
          raw_row_bytes == input_columns / kQ2BlockElements * kQ2BlockBytes,
      "GGUF Q2_K raw weight row size mismatch");
  STD_TORCH_CHECK(activation_codes.size(0) == token_count * topk &&
                      activation_scales.size(0) == token_count * topk &&
                      activation_scales.size(1) == input_columns / 32 &&
                      output.size(0) == token_count && output.size(1) == topk &&
                      output.size(2) == output_rows,
                  "GGUF Q2_K activation/output shape mismatch");

  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * topk * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  q2_k_q8_1_indexed_down_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
      reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
      down_weights.const_data_ptr<uint8_t>(),
      topk_ids.const_data_ptr<int32_t>(), output.mutable_data_ptr<float>(),
      token_count, topk, expert_count, output_rows, input_columns,
      raw_row_bytes);
}

}  // namespace vllm::gguf_dsv4
