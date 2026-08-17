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
#include <cuda_runtime.h>

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

__device__ __forceinline__ float warp_max(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return value;
}

__global__ void quantize_bf16_to_q8_1_kernel(
    const __nv_bfloat16* __restrict__ input, __half* __restrict__ output_scales,
    int8_t* __restrict__ output_codes, int group_count) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int group_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (group_index >= group_count) {
    return;
  }

  const int element_index = group_index * kIq2GroupElements + lane;
  const float value = __bfloat162float(input[element_index]);
  const float absolute_max = __shfl_sync(0xffffffff, warp_max(fabsf(value)), 0);
  const float scale = absolute_max / 127.0f;
  const int quantized =
      absolute_max == 0.0f ? 0 : static_cast<int>(roundf(value / scale));
  output_codes[element_index] = static_cast<int8_t>(quantized);
  if (lane == 0) {
    output_scales[group_index] = __float2half_rn(scale);
  }
}

template <bool kAligned>
__device__ float iq2_xxs_q8_1_dot(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ raw_weights,
    const uint8_t* __restrict__ aligned_scales,
    const uint8_t* __restrict__ aligned_grid_bytes,
    const uint8_t* __restrict__ aligned_scale_sign_bytes, int token_index,
    int output_row, int input_columns, int blocks_per_row, int raw_row_bytes) {
  const int lane = threadIdx.x & 31;
  float partial = 0.0f;
  for (int first_block = 0; first_block < blocks_per_row; first_block += 4) {
    const int block_index = first_block + (lane >> 3);
    const int group_index = lane & 7;
    if (block_index >= blocks_per_row) {
      continue;
    }

    float weight_scale;
    uint32_t grid_word;
    uint32_t scale_sign_word;
    if constexpr (kAligned) {
      const int block_offset = output_row * blocks_per_row + block_index;
      weight_scale = __half2float(
          *reinterpret_cast<const __half*>(aligned_scales + block_offset * 2));
      const int group_offset =
          (block_offset * kIq2GroupsPerBlock + group_index) * 4;
      grid_word =
          *reinterpret_cast<const uint32_t*>(aligned_grid_bytes + group_offset);
      scale_sign_word = *reinterpret_cast<const uint32_t*>(
          aligned_scale_sign_bytes + group_offset);
    } else {
      const uint8_t* block_address = raw_weights + output_row * raw_row_bytes +
                                     block_index * kIq2BlockBytes;
      weight_scale =
          __half2float(*reinterpret_cast<const __half*>(block_address));
      const uint8_t* group_address = block_address + 2 + group_index * 8;
      grid_word = load_split_u32(group_address);
      scale_sign_word = load_split_u32(group_address + 4);
    }

    const int activation_group = block_index * kIq2GroupsPerBlock + group_index;
    const int* activation_packs = reinterpret_cast<const int*>(
        activation_codes + token_index * input_columns +
        activation_group * kIq2GroupElements);
    int integer_sum = 0;
#pragma unroll
    for (int grid_part = 0; grid_part < 4; ++grid_part) {
      const uint8_t grid_index =
          static_cast<uint8_t>(grid_word >> (8 * grid_part));
      const uint64_t grid_values = iq2xxs_grid[grid_index];
      const int low_grid_pack = static_cast<int>(grid_values);
      const int high_grid_pack = static_cast<int>(grid_values >> 32);
      const uint8_t sign_selector =
          static_cast<uint8_t>((scale_sign_word >> (7 * grid_part)) & 127);
      const uint32_t replicated_signs =
          static_cast<uint32_t>(ksigns_iq2xs[sign_selector]) * 0x01010101u;
      const int low_sign_bytes = __vcmpne4(replicated_signs & 0x08040201u, 0);
      const int high_sign_bytes = __vcmpne4(replicated_signs & 0x80402010u, 0);
      const int low_signed_grid =
          __vsub4(low_grid_pack ^ low_sign_bytes, low_sign_bytes);
      const int high_signed_grid =
          __vsub4(high_grid_pack ^ high_sign_bytes, high_sign_bytes);
      integer_sum =
          __dp4a(low_signed_grid, activation_packs[2 * grid_part], integer_sum);
      integer_sum = __dp4a(high_signed_grid,
                           activation_packs[2 * grid_part + 1], integer_sum);
    }
    const int integer_scale = (scale_sign_word >> 27) | 1;
    integer_sum = integer_sum * integer_scale / 8;
    const float activation_scale =
        __half2float(activation_scales[token_index * (input_columns / 32) +
                                       activation_group]);
    partial = fmaf(weight_scale * activation_scale,
                   static_cast<float>(integer_sum), partial);
  }
  return warp_sum(partial);
}

template <bool kAligned>
__global__ void iq2_xxs_q8_1_matvec_kernel(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ raw_weights,
    const uint8_t* __restrict__ aligned_scales,
    const uint8_t* __restrict__ aligned_grid_bytes,
    const uint8_t* __restrict__ aligned_scale_sign_bytes,
    float* __restrict__ output, int token_count, int output_rows,
    int input_columns, int blocks_per_row, int raw_row_bytes) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int output_index = blockIdx.x * kWarpsPerBlock + warp_in_block;
  if (output_index >= token_count * output_rows) {
    return;
  }
  const int token_index = output_index / output_rows;
  const int output_row = output_index % output_rows;
  const float sum = iq2_xxs_q8_1_dot<kAligned>(
      activation_scales, activation_codes, raw_weights, aligned_scales,
      aligned_grid_bytes, aligned_scale_sign_bytes, token_index, output_row,
      input_columns, blocks_per_row, raw_row_bytes);
  if (lane == 0) {
    output[output_index] = sum;
  }
}

__global__ void iq2_xxs_q8_1_indexed_gate_up_kernel(
    const __half* __restrict__ activation_scales,
    const int8_t* __restrict__ activation_codes,
    const uint8_t* __restrict__ gate_weights,
    const uint8_t* __restrict__ up_weights,
    const int32_t* __restrict__ topk_ids, float* __restrict__ gate_output,
    float* __restrict__ up_output, int token_count, int topk, int expert_count,
    int output_rows, int input_columns, int blocks_per_row, int raw_row_bytes) {
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
  const uint8_t* expert_weights =
      projection_weights + expert * output_rows * raw_row_bytes;
  const float sum = iq2_xxs_q8_1_dot<false>(
      activation_scales, activation_codes, expert_weights, nullptr, nullptr,
      nullptr, token_index, output_row, input_columns, blocks_per_row,
      raw_row_bytes);
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

void check_q8_1_tensors(const torch::stable::Tensor& activation_scales,
                        const torch::stable::Tensor& activation_codes,
                        torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(activation_scales.device().is_cuda() &&
                      activation_codes.device().is_cuda() &&
                      output.device().is_cuda(),
                  "GGUF Q8_1 matvec tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      activation_scales.get_device_index() == output.get_device_index() &&
          activation_codes.get_device_index() == output.get_device_index(),
      "GGUF Q8_1 matvec tensors must share one CUDA device");
  STD_TORCH_CHECK(activation_scales.scalar_type() == ScalarType::Half &&
                      activation_codes.scalar_type() == ScalarType::Char &&
                      output.scalar_type() == ScalarType::Float,
                  "GGUF Q8_1 matvec requires fp16 scales, int8 codes, and "
                  "float32 output");
  STD_TORCH_CHECK(activation_scales.is_contiguous() &&
                      activation_codes.is_contiguous() &&
                      output.is_contiguous(),
                  "GGUF Q8_1 matvec tensors must be contiguous");
  STD_TORCH_CHECK(activation_scales.dim() == 2 && activation_codes.dim() == 2,
                  "GGUF Q8_1 activation scales/codes must be rank 2");
  STD_TORCH_CHECK(activation_codes.size(1) % kIq2BlockElements == 0 &&
                      activation_scales.size(0) == activation_codes.size(0) &&
                      activation_scales.size(1) ==
                          activation_codes.size(1) / kIq2GroupElements,
                  "GGUF Q8_1 scale/code shape mismatch");
}

void launch_iq2_xxs_q8_1_matvec(const torch::stable::Tensor& activation_scales,
                                const torch::stable::Tensor& activation_codes,
                                const uint8_t* raw_weights,
                                const uint8_t* aligned_scales,
                                const uint8_t* aligned_grid_bytes,
                                const uint8_t* aligned_scale_sign_bytes,
                                torch::stable::Tensor& output, int output_rows,
                                int raw_row_bytes, bool aligned) {
  const int token_count = activation_codes.size(0);
  const int input_columns = activation_codes.size(1);
  const int blocks_per_row = input_columns / kIq2BlockElements;
  STD_TORCH_CHECK(output.dim() == 2 && output.size(0) == token_count &&
                      output.size(1) == output_rows,
                  "GGUF IQ2_XXS Q8_1 matvec output shape mismatch");

  const int device_index = activation_codes.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  if (aligned) {
    iq2_xxs_q8_1_matvec_kernel<true>
        <<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
            reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
            nullptr, aligned_scales, aligned_grid_bytes,
            aligned_scale_sign_bytes, output.mutable_data_ptr<float>(),
            token_count, output_rows, input_columns, blocks_per_row, 0);
  } else {
    iq2_xxs_q8_1_matvec_kernel<false>
        <<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
            reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
            reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
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

void gguf_quantize_bf16_to_q8_1(const torch::stable::Tensor& input,
                                torch::stable::Tensor& output_scales,
                                torch::stable::Tensor& output_codes) {
  using torch::headeronly::ScalarType;
  STD_TORCH_CHECK(input.device().is_cuda() &&
                      output_scales.device().is_cuda() &&
                      output_codes.device().is_cuda(),
                  "GGUF Q8_1 quantization tensors must be CUDA tensors");
  STD_TORCH_CHECK(
      input.get_device_index() == output_scales.get_device_index() &&
          input.get_device_index() == output_codes.get_device_index(),
      "GGUF Q8_1 quantization tensors must share one CUDA device");
  STD_TORCH_CHECK(input.scalar_type() == ScalarType::BFloat16 &&
                      output_scales.scalar_type() == ScalarType::Half &&
                      output_codes.scalar_type() == ScalarType::Char,
                  "GGUF Q8_1 quantization requires bf16 input, fp16 scales, "
                  "and int8 codes");
  STD_TORCH_CHECK(input.is_contiguous() && output_scales.is_contiguous() &&
                      output_codes.is_contiguous() && input.dim() == 2 &&
                      output_scales.dim() == 2 && output_codes.dim() == 2,
                  "GGUF Q8_1 quantization tensors must be contiguous rank 2");
  STD_TORCH_CHECK(
      input.size(1) % kIq2GroupElements == 0 &&
          output_codes.size(0) == input.size(0) &&
          output_codes.size(1) == input.size(1) &&
          output_scales.size(0) == input.size(0) &&
          output_scales.size(1) == input.size(1) / kIq2GroupElements,
      "GGUF Q8_1 quantization output shape mismatch");

  const int device_index = input.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int group_count = input.numel() / kIq2GroupElements;
  const int grid_blocks = (group_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  quantize_bf16_to_q8_1_kernel<<<grid_blocks, kThreadsPerBlock, 0, stream>>>(
      reinterpret_cast<const __nv_bfloat16*>(input.const_data_ptr()),
      reinterpret_cast<__half*>(output_scales.mutable_data_ptr()),
      reinterpret_cast<int8_t*>(output_codes.mutable_data_ptr()), group_count);
}

void gguf_iq2_xxs_q8_1_raw_matvec(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& packed_weights,
    torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  check_q8_1_tensors(activation_scales, activation_codes, output);
  STD_TORCH_CHECK(
      packed_weights.device().is_cuda() &&
          packed_weights.get_device_index() == output.get_device_index() &&
          packed_weights.scalar_type() == ScalarType::Byte &&
          packed_weights.is_contiguous() && packed_weights.dim() == 2,
      "GGUF IQ2_XXS raw weights must be contiguous uint8 rank 2 on the "
      "output CUDA device");
  const int blocks_per_row = activation_codes.size(1) / kIq2BlockElements;
  STD_TORCH_CHECK(packed_weights.size(1) == blocks_per_row * kIq2BlockBytes,
                  "GGUF IQ2_XXS raw weight row size mismatch");
  launch_iq2_xxs_q8_1_matvec(activation_scales, activation_codes,
                             packed_weights.const_data_ptr<uint8_t>(), nullptr,
                             nullptr, nullptr, output, packed_weights.size(0),
                             packed_weights.size(1), false);
}

void gguf_iq2_xxs_q8_1_aligned_matvec(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& aligned_weight_scales,
    const torch::stable::Tensor& aligned_grid_bytes,
    const torch::stable::Tensor& aligned_scale_sign_bytes,
    torch::stable::Tensor& output) {
  using torch::headeronly::ScalarType;
  check_q8_1_tensors(activation_scales, activation_codes, output);
  for (const auto* tensor : {&aligned_weight_scales, &aligned_grid_bytes,
                             &aligned_scale_sign_bytes}) {
    STD_TORCH_CHECK(
        tensor->device().is_cuda() &&
            tensor->get_device_index() == output.get_device_index() &&
            tensor->scalar_type() == ScalarType::Byte &&
            tensor->is_contiguous(),
        "GGUF IQ2_XXS aligned streams must be contiguous uint8 tensors on "
        "the output CUDA device");
  }
  const int output_rows = aligned_weight_scales.size(0);
  const int blocks_per_row = activation_codes.size(1) / kIq2BlockElements;
  STD_TORCH_CHECK(aligned_weight_scales.dim() == 3 &&
                      aligned_weight_scales.size(1) == blocks_per_row &&
                      aligned_weight_scales.size(2) == 2,
                  "GGUF IQ2_XXS aligned scale shape mismatch");
  for (const auto* stream : {&aligned_grid_bytes, &aligned_scale_sign_bytes}) {
    STD_TORCH_CHECK(stream->dim() == 4 && stream->size(0) == output_rows &&
                        stream->size(1) == blocks_per_row &&
                        stream->size(2) == kIq2GroupsPerBlock &&
                        stream->size(3) == 4,
                    "GGUF IQ2_XXS aligned byte-stream shape mismatch");
  }
  launch_iq2_xxs_q8_1_matvec(activation_scales, activation_codes, nullptr,
                             aligned_weight_scales.const_data_ptr<uint8_t>(),
                             aligned_grid_bytes.const_data_ptr<uint8_t>(),
                             aligned_scale_sign_bytes.const_data_ptr<uint8_t>(),
                             output, output_rows, 0, true);
}

void gguf_iq2_xxs_q8_1_indexed_gate_up(
    const torch::stable::Tensor& activation_scales,
    const torch::stable::Tensor& activation_codes,
    const torch::stable::Tensor& gate_weights,
    const torch::stable::Tensor& up_weights,
    const torch::stable::Tensor& topk_ids, torch::stable::Tensor& gate_output,
    torch::stable::Tensor& up_output) {
  using torch::headeronly::ScalarType;
  check_q8_1_tensors(activation_scales, activation_codes, gate_output);
  STD_TORCH_CHECK(
      up_output.device().is_cuda() &&
          up_output.get_device_index() == gate_output.get_device_index() &&
          up_output.scalar_type() == ScalarType::Float &&
          up_output.is_contiguous() && up_output.dim() == 3,
      "GGUF indexed up output must be contiguous float32 rank 3");
  STD_TORCH_CHECK(gate_output.dim() == 3,
                  "GGUF indexed gate output must be rank 3");
  for (const auto* weights : {&gate_weights, &up_weights}) {
    STD_TORCH_CHECK(
        weights->device().is_cuda() &&
            weights->get_device_index() == gate_output.get_device_index() &&
            weights->scalar_type() == ScalarType::Byte &&
            weights->is_contiguous() && weights->dim() == 3,
        "GGUF indexed weights must be contiguous uint8 rank 3 on the output "
        "CUDA device");
  }
  STD_TORCH_CHECK(
      topk_ids.device().is_cuda() &&
          topk_ids.get_device_index() == gate_output.get_device_index() &&
          topk_ids.scalar_type() == ScalarType::Int &&
          topk_ids.is_contiguous() && topk_ids.dim() == 2,
      "GGUF indexed top-k ids must be contiguous int32 rank 2");
  STD_TORCH_CHECK(gate_weights.size(0) == up_weights.size(0) &&
                      gate_weights.size(1) == up_weights.size(1) &&
                      gate_weights.size(2) == up_weights.size(2),
                  "GGUF indexed gate/up weight shapes must match");
  const int token_count = activation_codes.size(0);
  const int input_columns = activation_codes.size(1);
  const int expert_count = gate_weights.size(0);
  const int output_rows = gate_weights.size(1);
  const int raw_row_bytes = gate_weights.size(2);
  const int topk = topk_ids.size(1);
  const int blocks_per_row = input_columns / kIq2BlockElements;
  STD_TORCH_CHECK(raw_row_bytes == blocks_per_row * kIq2BlockBytes,
                  "GGUF indexed raw weight row size mismatch");
  STD_TORCH_CHECK(
      topk_ids.size(0) == token_count && gate_output.size(0) == token_count &&
          gate_output.size(1) == topk && gate_output.size(2) == output_rows &&
          up_output.size(0) == gate_output.size(0) &&
          up_output.size(1) == gate_output.size(1) &&
          up_output.size(2) == gate_output.size(2),
      "GGUF indexed top-k/output shape mismatch");

  const int device_index = activation_codes.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int total_warps = token_count * topk * 2 * output_rows;
  const int grid_blocks = (total_warps + kWarpsPerBlock - 1) / kWarpsPerBlock;
  iq2_xxs_q8_1_indexed_gate_up_kernel<<<grid_blocks, kThreadsPerBlock, 0,
                                        stream>>>(
      reinterpret_cast<const __half*>(activation_scales.const_data_ptr()),
      reinterpret_cast<const int8_t*>(activation_codes.const_data_ptr()),
      gate_weights.const_data_ptr<uint8_t>(),
      up_weights.const_data_ptr<uint8_t>(), topk_ids.const_data_ptr<int32_t>(),
      gate_output.mutable_data_ptr<float>(),
      up_output.mutable_data_ptr<float>(), token_count, topk, expert_count,
      output_rows, input_columns, blocks_per_row, raw_row_bytes);
}

}  // namespace vllm::gguf_dsv4
