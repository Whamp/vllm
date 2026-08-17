// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// IQ2_XXS format math follows the MIT-licensed llama.cpp/GGML format tables
// adapted through antirez/ds4@84cc882. Kernel structure is original vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "../../torch_utils.h"
#include "iq2_xxs_tables.cuh"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kIq2BlockElements = 256;
constexpr int kIq2BlockBytes = 66;
constexpr int kIq2GroupsPerBlock = 8;
constexpr int kIq2GroupElements = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;

__device__ __forceinline__ uint32_t load_split_u32(const uint8_t* address) {
  const auto* halfwords = reinterpret_cast<const uint16_t*>(address);
  return static_cast<uint32_t>(halfwords[0]) |
         (static_cast<uint32_t>(halfwords[1]) << 16);
}

__device__ __forceinline__ float warp_sum(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value += __shfl_down_sync(0xffffffff, value, offset);
  }
  return value;
}

template <bool kAligned>
__global__ void iq2_xxs_matvec_kernel(
    const __nv_bfloat16* __restrict__ input,
    const uint8_t* __restrict__ raw_weights,
    const uint8_t* __restrict__ aligned_scales,
    const uint8_t* __restrict__ aligned_grid_bytes,
    const uint8_t* __restrict__ aligned_scale_sign_bytes,
    float* __restrict__ output, int token_count, int output_rows,
    int input_columns, int blocks_per_row, int raw_row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int output_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  const int total_outputs = token_count * output_rows;
  if (output_index >= total_outputs) {
    return;
  }

  const int token_index = output_index / output_rows;
  const int output_row = output_index % output_rows;
  float partial = 0.0f;

  for (int block_index = 0; block_index < blocks_per_row; ++block_index) {
    float scale = 0.0f;
    if (lane == 0) {
      const uint8_t* scale_address;
      if constexpr (kAligned) {
        scale_address =
            aligned_scales + (output_row * blocks_per_row + block_index) * 2;
      } else {
        scale_address = raw_weights + output_row * raw_row_bytes +
                        block_index * kIq2BlockBytes;
      }
      scale = __half2float(*reinterpret_cast<const __half*>(scale_address));
    }
    scale = __shfl_sync(0xffffffff, scale, 0);

#pragma unroll
    for (int group_index = 0; group_index < kIq2GroupsPerBlock; ++group_index) {
      uint32_t grid_word = 0;
      uint32_t scale_sign_word = 0;
      if (lane == 0) {
        if constexpr (kAligned) {
          const int group_offset =
              ((output_row * blocks_per_row + block_index) *
                   kIq2GroupsPerBlock +
               group_index) *
              4;
          grid_word = *reinterpret_cast<const uint32_t*>(aligned_grid_bytes +
                                                         group_offset);
          scale_sign_word = *reinterpret_cast<const uint32_t*>(
              aligned_scale_sign_bytes + group_offset);
        } else {
          const uint8_t* group_address =
              raw_weights + output_row * raw_row_bytes +
              block_index * kIq2BlockBytes + 2 + group_index * 8;
          grid_word = load_split_u32(group_address);
          scale_sign_word = load_split_u32(group_address + 4);
        }
      }
      grid_word = __shfl_sync(0xffffffff, grid_word, 0);
      scale_sign_word = __shfl_sync(0xffffffff, scale_sign_word, 0);

      const int grid_part = lane >> 3;
      const int grid_element = lane & 7;
      const uint8_t grid_index =
          static_cast<uint8_t>(grid_word >> (8 * grid_part));
      const uint64_t grid_values = iq2xxs_grid[grid_index];
      const float grid_value =
          static_cast<float>((grid_values >> (8 * grid_element)) & 0xff);
      const uint8_t sign_selector =
          static_cast<uint8_t>((scale_sign_word >> (7 * grid_part)) & 127);
      const bool negative =
          (ksigns_iq2xs[sign_selector] & kmask_iq2xs[grid_element]) != 0;
      const float group_scale =
          scale * (0.5f + static_cast<float>(scale_sign_word >> 28)) * 0.25f;
      const float weight =
          negative ? -group_scale * grid_value : group_scale * grid_value;
      const int input_column = block_index * kIq2BlockElements +
                               group_index * kIq2GroupElements + lane;
      const float activation =
          __bfloat162float(input[token_index * input_columns + input_column]);
      partial = fmaf(activation, weight, partial);
    }
  }

  const float sum = warp_sum(partial);
  if (lane == 0) {
    output[output_index] = sum;
  }
}

void check_common_matvec_tensors(const torch::stable::Tensor& input,
                                 torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.device().is_cuda() && output.device().is_cuda(),
                  "GGUF IQ2_XXS matvec tensors must be CUDA tensors");
  STD_TORCH_CHECK(input.get_device_index() == output.get_device_index(),
                  "GGUF IQ2_XXS matvec tensors must share one CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16,
                  "GGUF IQ2_XXS matvec input must be bfloat16");
  STD_TORCH_CHECK(output.scalar_type() == ScalarType::Float,
                  "GGUF IQ2_XXS matvec output must be float32");
  STD_TORCH_CHECK(input.is_contiguous() && output.is_contiguous(),
                  "GGUF IQ2_XXS matvec tensors must be contiguous");
  STD_TORCH_CHECK(input.dim() == 2 && output.dim() == 2,
                  "GGUF IQ2_XXS matvec input/output must be rank 2");
  STD_TORCH_CHECK(input.size(1) % kIq2BlockElements == 0,
                  "GGUF IQ2_XXS matvec K must be divisible by 256");
}

void launch_iq2_xxs_matvec(const torch::stable::Tensor& input,
                           const uint8_t* raw_weights,
                           const uint8_t* aligned_scales,
                           const uint8_t* aligned_grid_bytes,
                           const uint8_t* aligned_scale_sign_bytes,
                           torch::stable::Tensor& output, int output_rows,
                           int raw_row_bytes, bool aligned) {
  const int token_count = input.size(0);
  const int input_columns = input.size(1);
  const int blocks_per_row = input_columns / kIq2BlockElements;
  STD_TORCH_CHECK(
      output.size(0) == token_count && output.size(1) == output_rows,
      "GGUF IQ2_XXS matvec output shape mismatch");

  const int device_index = input.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;

  if (aligned) {
    iq2_xxs_matvec_kernel<true><<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()), nullptr,
        aligned_scales, aligned_grid_bytes, aligned_scale_sign_bytes,
        output.mutable_data_ptr<float>(), token_count, output_rows,
        input_columns, blocks_per_row, 0);
  } else {
    iq2_xxs_matvec_kernel<false><<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),
        raw_weights, nullptr, nullptr, nullptr,
        output.mutable_data_ptr<float>(), token_count, output_rows,
        input_columns, blocks_per_row, raw_row_bytes);
  }
}

}  // namespace

void gguf_iq2_xxs_raw_matvec(const torch::stable::Tensor& input,
                             const torch::stable::Tensor& packed_weights,
                             torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  check_common_matvec_tensors(input, output);
  STD_TORCH_CHECK(
      packed_weights.device().is_cuda() &&
          packed_weights.get_device_index() == input.get_device_index(),
      "GGUF IQ2_XXS raw weights must share the input CUDA device");
  STD_TORCH_CHECK(packed_weights.scalar_type() == ScalarType::Byte &&
                      packed_weights.is_contiguous() &&
                      packed_weights.dim() == 2,
                  "GGUF IQ2_XXS raw weights must be contiguous uint8 rank 2");
  const int blocks_per_row = input.size(1) / kIq2BlockElements;
  STD_TORCH_CHECK(packed_weights.size(1) == blocks_per_row * kIq2BlockBytes,
                  "GGUF IQ2_XXS raw weight row size mismatch");
  launch_iq2_xxs_matvec(input, packed_weights.const_data_ptr<uint8_t>(),
                        nullptr, nullptr, nullptr, output,
                        packed_weights.size(0), packed_weights.size(1), false);
}

void gguf_iq2_xxs_aligned_matvec(
    const torch::stable::Tensor& input,
    const torch::stable::Tensor& aligned_scales,
    const torch::stable::Tensor& aligned_grid_bytes,
    const torch::stable::Tensor& aligned_scale_sign_bytes,
    torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  check_common_matvec_tensors(input, output);
  for (const auto* tensor :
       {&aligned_scales, &aligned_grid_bytes, &aligned_scale_sign_bytes}) {
    STD_TORCH_CHECK(
        tensor->device().is_cuda() &&
            tensor->get_device_index() == input.get_device_index() &&
            tensor->scalar_type() == ScalarType::Byte &&
            tensor->is_contiguous(),
        "GGUF IQ2_XXS aligned streams must be contiguous uint8 "
        "tensors on the input CUDA device");
  }
  const int output_rows = aligned_scales.size(0);
  const int blocks_per_row = input.size(1) / kIq2BlockElements;
  STD_TORCH_CHECK(aligned_scales.dim() == 3 &&
                      aligned_scales.size(1) == blocks_per_row &&
                      aligned_scales.size(2) == 2,
                  "GGUF IQ2_XXS aligned scale shape mismatch");
  for (const auto* stream : {&aligned_grid_bytes, &aligned_scale_sign_bytes}) {
    STD_TORCH_CHECK(stream->dim() == 4 && stream->size(0) == output_rows &&
                        stream->size(1) == blocks_per_row &&
                        stream->size(2) == kIq2GroupsPerBlock &&
                        stream->size(3) == 4,
                    "GGUF IQ2_XXS aligned byte-stream shape mismatch");
  }
  launch_iq2_xxs_matvec(input, nullptr,
                        aligned_scales.const_data_ptr<uint8_t>(),
                        aligned_grid_bytes.const_data_ptr<uint8_t>(),
                        aligned_scale_sign_bytes.const_data_ptr<uint8_t>(),
                        output, output_rows, 0, true);
}

}  // namespace vllm::gguf_dsv4
