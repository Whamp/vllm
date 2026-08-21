// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Q4_K, Q5_K, and Q6_K arithmetic follows the MIT-licensed
// Whamp/llama.cpp@0379cf4 GGML format definition. The stable ABI and
// caller-owned Q8_1 execution contract are vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kBlockElements = 256;
constexpr int kGroupElements = 32;
constexpr int kGroupsPerBlock = 8;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;

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

__device__ __forceinline__ float warp_sum_k_quant(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

__device__ __forceinline__ uint32_t pack_four_bytes(const int8_t* values) {
  uint32_t packed = 0;
#pragma unroll
  for (int index = 0; index < 4; ++index) {
    packed |= static_cast<uint32_t>(static_cast<uint8_t>(values[index]))
              << (8 * index);
  }
  return packed;
}

__device__ __forceinline__ void decode_scale_min(const uint8_t* scales,
                                                 int group_index, int& scale,
                                                 int& minimum) {
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

__device__ __forceinline__ uint32_t load_u32(const uint8_t* address) {
  uint32_t value;
  memcpy(&value, address, sizeof(value));
  return value;
}

// 16-byte vector load. Callers must guarantee 16-byte alignment: Q4_K/Q5_K
// blocks are 144/176 bytes and rows are block-strided, so every quant segment
// sits at a multiple of 16 from the tensor base.
__device__ __forceinline__ uint4 load_u16x(const uint8_t* address) {
  return *reinterpret_cast<const uint4*>(address);
}

template <KQuantFormat kFormat>
__device__ __forceinline__ float q45_group_dot(const uint8_t* block,
                                               int group_index,
                                               const int* activation_packs) {
  const __half2 scales = *reinterpret_cast<const __half2*>(block);
  const float2 decoded_scales = __half22float2(scales);
  const uint8_t* packed_scales = block + 4;
  const uint8_t* quants =
      kFormat == KQuantFormat::kQ5 ? block + 48 : block + 16;
  const int segment = group_index / 2;
  const bool high_nibble = (group_index & 1) != 0;
  // Each group consumes a full 32-byte quant window (one byte per element);
  // adjacent groups share the window and select low/high nibbles.
  const uint8_t* segment_quants = quants + segment * 32;
  const uint4 quant_words[2] = {load_u16x(segment_quants),
                                load_u16x(segment_quants + 8)};
  const uint32_t nibble_mask = high_nibble ? 0xf0f0f0f0U : 0x0f0f0f0fU;
  const uint32_t nibble_shift = high_nibble ? 4 : 0;
  const uint32_t words[8] = {
      quant_words[0].x, quant_words[0].y, quant_words[0].z, quant_words[0].w,
      quant_words[1].x, quant_words[1].y, quant_words[1].z, quant_words[1].w};
  int dot = 0;
  int code_sum = 0;
#pragma unroll
  for (int pack_index = 0; pack_index < 8; ++pack_index) {
    uint32_t packed = (words[pack_index] >> nibble_shift) & 0x0f0f0f0fU;
    if constexpr (kFormat == KQuantFormat::kQ5) {
      // High bits are bit-plane-per-position: byte element at block+16,
      // bit group_index. One aligned 32-bit load covers this pack's four
      // elements; shift the plane bit into each byte's value bit 4.
      const uint32_t plane =
          load_u32(block + 16 + 4 * pack_index) >> group_index;
      packed |= (plane & 0x01010101U) << 4;
    }
    dot = __dp4a(static_cast<int>(packed), activation_packs[pack_index], dot);
    code_sum = __dp4a(0x01010101, activation_packs[pack_index], code_sum);
  }
  int scale;
  int minimum;
  decode_scale_min(packed_scales, group_index, scale, minimum);
  return decoded_scales.x * static_cast<float>(scale * dot) -
         decoded_scales.y * static_cast<float>(minimum * code_sum);
}

__device__ __forceinline__ float q6_group_dot(const uint8_t* block,
                                              int group_index,
                                              const int* activation_packs) {
  const uint8_t* low = block;
  const uint8_t* high = block + 128;
  const int8_t* scales = reinterpret_cast<const int8_t*>(block + 192);
  const int half = group_index / 4;
  const int quadrant = group_index % 4;
  const int low_base = half * 64 + ((quadrant & 1) != 0 ? 32 : 0);
  const int high_base = half * 32;
  const int nibble_shift = quadrant >= 2 ? 4 : 0;
  const int high_shift = 2 * quadrant;
  const int scale_base = half * 8 + 2 * quadrant;
  int first_dot = 0;
  int second_dot = 0;
#pragma unroll
  for (int pack_index = 0; pack_index < 8; ++pack_index) {
    int8_t values[4];
#pragma unroll
    for (int byte_index = 0; byte_index < 4; ++byte_index) {
      const int element = 4 * pack_index + byte_index;
      const int low_value = (low[low_base + element] >> nibble_shift) & 15;
      const int high_value = (high[high_base + element] >> high_shift) & 3;
      values[byte_index] =
          static_cast<int8_t>(low_value | (high_value << 4)) - 32;
    }
    if (pack_index < 4) {
      first_dot = __dp4a(static_cast<int>(pack_four_bytes(values)),
                         activation_packs[pack_index], first_dot);
    } else {
      second_dot = __dp4a(static_cast<int>(pack_four_bytes(values)),
                          activation_packs[pack_index], second_dot);
    }
  }
  const float block_scale =
      __half2float(*reinterpret_cast<const __half*>(block + 208));
  return block_scale *
         (static_cast<float>(scales[scale_base] * first_dot) +
          static_cast<float>(scales[scale_base + 1] * second_dot));
}

template <KQuantFormat kFormat>
__device__ __forceinline__ float k_quant_q8_1_dot(
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
    const int* activation_packs = reinterpret_cast<const int*>(
        activation_codes + activation_row * input_columns +
        activation_group * kGroupElements);
    float group_sum;
    if constexpr (kFormat == KQuantFormat::kQ6) {
      group_sum = q6_group_dot(block, group_index, activation_packs);
    } else {
      group_sum = q45_group_dot<kFormat>(block, group_index, activation_packs);
    }
    const float activation_scale = __half2float(
        activation_scales[activation_row * group_count + activation_group]);
    partial = fmaf(activation_scale, group_sum, partial);
  }
  return warp_sum_k_quant(partial);
}

template <KQuantFormat kFormat>
__global__ void k_quant_q8_1_matvec_kernel(
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
  const float sum = k_quant_q8_1_dot<kFormat>(
      activation_scales, activation_codes, weights + output_row * raw_row_bytes,
      token_index, input_columns);
  if (lane == 0) {
    output[output_index] = sum;
  }
}

__global__ void q4_k_embedding_kernel(const int64_t* __restrict__ input_ids,
                                      const uint8_t* __restrict__ weights,
                                      __nv_bfloat16* __restrict__ output,
                                      int token_count, int vocab_rows,
                                      int hidden_size, int raw_row_bytes) {
  const int token_index = blockIdx.x;
  if (token_index >= token_count) {
    return;
  }
  const int64_t row_index = input_ids[token_index];
  if (row_index < 0 || row_index >= vocab_rows) {
    return;
  }
  const uint8_t* weight_row = weights + row_index * raw_row_bytes;
  for (int column = threadIdx.x; column < hidden_size; column += blockDim.x) {
    const int block_index = column / kBlockElements;
    const int within_block = column % kBlockElements;
    const int group_index = within_block / kGroupElements;
    const int element = within_block % kGroupElements;
    const uint8_t* block =
        weight_row + block_index * block_bytes<KQuantFormat::kQ4>();
    const float2 decoded_scales =
        __half22float2(*reinterpret_cast<const __half2*>(block));
    int scale;
    int minimum;
    decode_scale_min(block + 4, group_index, scale, minimum);
    const uint8_t packed = block[16 + (group_index / 2) * 32 + element];
    const int value = (group_index & 1) != 0 ? packed >> 4 : packed & 15;
    const float decoded =
        decoded_scales.x * scale * value - decoded_scales.y * minimum;
    output[token_index * hidden_size + column] = __float2bfloat16(decoded);
  }
}

template <KQuantFormat kFormat>
void launch_k_quant_raw_matvec(const torch::stable::Tensor& activation_scales,
                               const torch::stable::Tensor& activation_codes,
                               const torch::stable::Tensor& weights,
                               torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {
      &activation_scales, &activation_codes, &weights, &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF Q4/Q5/Q6 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(tensor->get_device_index() == output.get_device_index(),
                    "GGUF Q4/Q5/Q6 tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(activation_scales.scalar_type() == ScalarType::Half &&
                      activation_codes.scalar_type() == ScalarType::Char &&
                      weights.scalar_type() == ScalarType::Byte &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF Q4/Q5/Q6 dtype contract mismatch");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2 &&
                      weights.dim() == 2 && output.dim() == 2,
                  "GGUF Q4/Q5/Q6 raw matvec tensor rank mismatch");
  const int token_count = activation_codes.size(0);
  const int input_columns = activation_codes.size(1);
  const int output_rows = weights.size(0);
  STD_TORCH_CHECK(input_columns % kBlockElements == 0,
                  "GGUF Q4/Q5/Q6 input columns must be divisible by 256");
  STD_TORCH_CHECK(
      activation_scales.size(0) == token_count &&
          activation_scales.size(1) == input_columns / kGroupElements &&
          weights.size(1) ==
              input_columns / kBlockElements * block_bytes<kFormat>() &&
          output.size(0) == token_count && output.size(1) == output_rows,
      "GGUF Q4/Q5/Q6 raw matvec shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  k_quant_q8_1_matvec_kernel<kFormat>
      <<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
          reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
          reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
          weights.const_data_ptr<uint8_t>(), output.mutable_data_ptr<float>(),
          token_count, output_rows, input_columns, weights.size(1));
}

}  // namespace

void gguf_q4_k_q8_1_raw_matvec(const torch::stable::Tensor& activation_scales,
                               const torch::stable::Tensor& activation_codes,
                               const torch::stable::Tensor& weights,
                               torch::stable::Tensor& output) {
  launch_k_quant_raw_matvec<KQuantFormat::kQ4>(
      activation_scales, activation_codes, weights, output);
}

void gguf_q5_k_q8_1_raw_matvec(const torch::stable::Tensor& activation_scales,
                               const torch::stable::Tensor& activation_codes,
                               const torch::stable::Tensor& weights,
                               torch::stable::Tensor& output) {
  launch_k_quant_raw_matvec<KQuantFormat::kQ5>(
      activation_scales, activation_codes, weights, output);
}

void gguf_q6_k_q8_1_raw_matvec(const torch::stable::Tensor& activation_scales,
                               const torch::stable::Tensor& activation_codes,
                               const torch::stable::Tensor& weights,
                               torch::stable::Tensor& output) {
  launch_k_quant_raw_matvec<KQuantFormat::kQ6>(
      activation_scales, activation_codes, weights, output);
}

void gguf_q4_k_embedding(const torch::stable::Tensor& input_ids,
                         const torch::stable::Tensor& weights,
                         torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {&input_ids, &weights, &output};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(
        tensor->device().is_cuda() && tensor->is_contiguous(),
        "GGUF Q4_K embedding tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(tensor->get_device_index() == output.get_device_index(),
                    "GGUF Q4_K embedding tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(input_ids.scalar_type() == ScalarType::Long &&
                      weights.scalar_type() == ScalarType::Byte &&
                      output.scalar_type() == ScalarType::BFloat16,
                  "GGUF Q4_K embedding dtype contract mismatch");
  STD_TORCH_CHECK(weights.dim() == 2 && output.dim() == input_ids.dim() + 1,
                  "GGUF Q4_K embedding tensor rank mismatch");
  const int token_count = input_ids.numel();
  const int vocab_rows = weights.size(0);
  const int hidden_size = output.size(-1);
  STD_TORCH_CHECK(hidden_size % kBlockElements == 0 &&
                      weights.size(1) == hidden_size / kBlockElements *
                                             block_bytes<KQuantFormat::kQ4>() &&
                      output.numel() == token_count * hidden_size,
                  "GGUF Q4_K embedding shape mismatch");
  const int device_index = output.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  q4_k_embedding_kernel<<<token_count, 256, 0, stream>>>(
      input_ids.const_data_ptr<int64_t>(), weights.const_data_ptr<uint8_t>(),
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr()), token_count,
      vocab_rows, hidden_size, weights.size(1));
}

}  // namespace vllm::gguf_dsv4
