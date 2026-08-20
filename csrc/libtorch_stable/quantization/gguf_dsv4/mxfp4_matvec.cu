// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// MXFP4 E8M0/E2M1 arithmetic follows the MIT-licensed
// Whamp/llama.cpp@0379cf4 GGML format definition. The stable ABI,
// caller-owned Q8_1 contract, and indexed expert schedule are vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kBlockElements = 32;
constexpr int kBlockBytes = 17;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;

__device__ __constant__ int8_t kMxfp4Values[16] = {
    0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12};

__device__ __forceinline__ float warp_sum_mxfp4(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ int8_t decode_e2m1_doubled(uint8_t code) {
  return kMxfp4Values[code & 15];
}

__device__ __forceinline__ int pack_four_e2m1(const uint8_t* packed,
                                              int first_index,
                                              bool high_nibble) {
  uint32_t result = 0;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    const uint8_t byte = packed[first_index + index];
    const uint8_t code = high_nibble ? byte >> 4 : byte & 15;
    const uint8_t value = static_cast<uint8_t>(decode_e2m1_doubled(code));
    result |= static_cast<uint32_t>(value) << (8 * index);
  }
  return static_cast<int>(result);
}

__device__ __forceinline__ float decode_e8m0_half(uint8_t exponent) {
  const uint32_t bits = exponent < 2
                            ? 0x00200000U << exponent
                            : static_cast<uint32_t>(exponent - 1) << 23;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float mxfp4_q8_1_dot(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ weight_row, int activation_row,
    int input_columns) {
  const int lane = threadIdx.x & 31;
  const int block_count = input_columns / kBlockElements;
  float partial = 0.0f;
  for (int block_index = lane; block_index < block_count; block_index += 32) {
    const uint8_t* block = weight_row + block_index * kBlockBytes;
    const uint8_t* packed = block + 1;
    const int* codes = reinterpret_cast<const int*>(
        activation_codes + activation_row * input_columns +
        block_index * kBlockElements);
    int integer_sum = 0;
#pragma unroll
    for (int pack_index = 0; pack_index < 4; ++pack_index) {
      const int first_index = 4 * pack_index;
      integer_sum = __dp4a(pack_four_e2m1(packed, first_index, false),
                           codes[pack_index], integer_sum);
      integer_sum = __dp4a(pack_four_e2m1(packed, first_index, true),
                           codes[pack_index + 4], integer_sum);
    }
    const float activation_scale = __half2float(
        activation_scales[activation_row * block_count + block_index]);
    partial = fmaf(decode_e8m0_half(block[0]) * activation_scale,
                   static_cast<float>(integer_sum), partial);
  }
  return warp_sum_mxfp4(partial);
}

__global__ void mxfp4_q8_1_matvec_kernel(
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
  const float sum = mxfp4_q8_1_dot(activation_scales, activation_codes,
                                   weights + output_row * raw_row_bytes,
                                   token_index, input_columns);
  if (lane == 0) {
    output[output_index] = sum;
  }
}

__global__ void mxfp4_q8_1_indexed_down_kernel(
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
  const float sum = mxfp4_q8_1_dot(activation_scales, activation_codes,
                                   weight_row, activation_row, input_columns);
  if (lane == 0) {
    output[(token_index * topk + slot) * output_rows + output_row] = sum;
  }
}

void check_mxfp4_inputs(const torch::stable::Tensor& activation_scales,
                        const torch::stable::Tensor& activation_codes,
                        const torch::stable::Tensor& weights,
                        const torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &activation_scales, &activation_codes, &weights, &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF MXFP4 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(tensor->get_device_index() == output.get_device_index(),
                    "GGUF MXFP4 tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(activation_scales.scalar_type() == ScalarType::Half &&
                      activation_codes.scalar_type() == ScalarType::Char &&
                      weights.scalar_type() == ScalarType::Byte &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF MXFP4 dtype contract mismatch");
  const int input_columns = activation_codes.size(-1);
  STD_TORCH_CHECK(input_columns % kBlockElements == 0,
                  "GGUF MXFP4 input columns must be divisible by 32");
  STD_TORCH_CHECK(activation_scales.size(-1) == input_columns / kBlockElements,
                  "GGUF MXFP4 Q8_1 scale shape mismatch");
  STD_TORCH_CHECK(
      weights.size(-1) == input_columns / kBlockElements * kBlockBytes,
      "GGUF MXFP4 raw weight row size mismatch");
}

}  // namespace

void gguf_mxfp4_q8_1_raw_matvec(const torch::stable::Tensor& activation_scales,
                                const torch::stable::Tensor& activation_codes,
                                const torch::stable::Tensor& weights,
                                torch::stable::Tensor& output) {
  check_mxfp4_inputs(activation_scales, activation_codes, weights, output);
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      weights.dim() == 2 && output.dim() == 2,
                  "GGUF MXFP4 raw matvec tensor rank mismatch");
  const int token_count = activation_codes.size(0);
  const int input_columns = activation_codes.size(1);
  const int output_rows = weights.size(0);
  STD_TORCH_CHECK(activation_scales.size(0) == token_count &&
                      output.size(0) == token_count &&
                      output.size(1) == output_rows,
                  "GGUF MXFP4 raw matvec output shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  mxfp4_q8_1_matvec_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
      reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
      weights.const_data_ptr<uint8_t>(), output.mutable_data_ptr<float>(),
      token_count, output_rows, input_columns, weights.size(1));
}

void gguf_mxfp4_q8_1_indexed_down(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& down_weights,
    const torch::stable::Tensor& topk_ids, torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  check_mxfp4_inputs(activation_scales, activation_codes, down_weights, output);
  STD_TORCH_CHECK(topk_ids.device().is_cuda() && topk_ids.is_contiguous(),
                  "GGUF MXFP4 top-k IDs must be a contiguous CUDA tensor");
  STD_TORCH_CHECK(topk_ids.scalar_type() == ScalarType::Int,
                  "GGUF MXFP4 top-k IDs must be int32");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      down_weights.dim() == 3 && topk_ids.dim() == 2 &&
                      output.dim() == 3,
                  "GGUF MXFP4 indexed down tensor rank mismatch");
  const int token_count = topk_ids.size(0);
  const int topk = topk_ids.size(1);
  const int expert_count = down_weights.size(0);
  const int output_rows = down_weights.size(1);
  const int input_columns = activation_codes.size(1);
  STD_TORCH_CHECK(activation_codes.size(0) == token_count * topk &&
                      activation_scales.size(0) == token_count * topk &&
                      output.size(0) == token_count && output.size(1) == topk &&
                      output.size(2) == output_rows,
                  "GGUF MXFP4 indexed down activation/output shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * topk * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  mxfp4_q8_1_indexed_down_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
      reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
      down_weights.const_data_ptr<uint8_t>(),
      topk_ids.const_data_ptr<int32_t>(), output.mutable_data_ptr<float>(),
      token_count, topk, expert_count, output_rows, input_columns,
      down_weights.size(2));
}

}  // namespace vllm::gguf_dsv4
