// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include "torch_utils.h"

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/library.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>
#include <torch/headeronly/util/Exception.h>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int kDownInputFeatures = 10240;
constexpr int kDownOutputFeatures = 336;
constexpr int kUpInputFeatures = 320;
constexpr int kUpOutputFeatures = 10240;

template <int Tokens, int OutputsPerBlock, int BlockThreads>
__global__ void qwen4_exp_hyperconnection_bf16_kernel(
    const __nv_bfloat16* __restrict__ activations,
    const __nv_bfloat16* __restrict__ weights,
    __nv_bfloat16* __restrict__ output, int output_features,
    int input_features) {
  constexpr int kAccumCount = Tokens * OutputsPerBlock;
  constexpr int kWarpCount = BlockThreads / 32;
  __shared__ float warp_partials[8][16];

  const int first_output = blockIdx.x * OutputsPerBlock;
  const int lane = threadIdx.x & 31;
  const int warp = threadIdx.x >> 5;
  float accumulators[kAccumCount] = {};

  const auto* activation_pairs =
      reinterpret_cast<const __nv_bfloat162*>(activations);
  const auto* weight_pairs = reinterpret_cast<const __nv_bfloat162*>(weights);
  const int input_pairs = input_features / 2;

  for (int pair = threadIdx.x; pair < input_pairs; pair += BlockThreads) {
    float2 activation_values[Tokens];
#pragma unroll
    for (int token = 0; token < Tokens; ++token) {
      activation_values[token] =
          __bfloat1622float2(activation_pairs[token * input_pairs + pair]);
    }
#pragma unroll
    for (int output_index = 0; output_index < OutputsPerBlock; ++output_index) {
      const int row = first_output + output_index;
      const float2 weight_values =
          __bfloat1622float2(weight_pairs[row * input_pairs + pair]);
#pragma unroll
      for (int token = 0; token < Tokens; ++token) {
        float sum = accumulators[token * OutputsPerBlock + output_index];
        sum = fmaf(activation_values[token].x, weight_values.x, sum);
        sum = fmaf(activation_values[token].y, weight_values.y, sum);
        accumulators[token * OutputsPerBlock + output_index] = sum;
      }
    }
  }

#pragma unroll
  for (int value = 0; value < kAccumCount; ++value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
      accumulators[value] +=
          __shfl_down_sync(0xffffffff, accumulators[value], offset);
    }
    if (lane == 0) {
      warp_partials[warp][value] = accumulators[value];
    }
  }
  __syncthreads();

  if (warp == 0) {
#pragma unroll
    for (int value = 0; value < kAccumCount; ++value) {
      float sum = lane < kWarpCount ? warp_partials[lane][value] : 0.0f;
#pragma unroll
      for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_down_sync(0xffffffff, sum, offset);
      }
      if (lane == 0) {
        const int token = value / OutputsPerBlock;
        const int output_index = value % OutputsPerBlock;
        output[token * output_features + first_output + output_index] =
            __float2bfloat16_rn(sum);
      }
    }
  }
}

template <int Tokens, int OutputsPerBlock, int BlockThreads>
void launch_qwen4_exp_hyperconnection_bf16(const __nv_bfloat16* activations,
                                           const __nv_bfloat16* weights,
                                           __nv_bfloat16* output,
                                           int output_features,
                                           int input_features,
                                           cudaStream_t stream) {
  const int blocks = output_features / OutputsPerBlock;
  qwen4_exp_hyperconnection_bf16_kernel<Tokens, OutputsPerBlock, BlockThreads>
      <<<blocks, BlockThreads, 0, stream>>>(activations, weights, output,
                                            output_features, input_features);
}

}  // namespace

torch::stable::Tensor qwen4_exp_hyperconnection_bf16_sm86(
    const torch::stable::Tensor& activations,
    const torch::stable::Tensor& weights) {
  using torch::headeronly::ScalarType;

  STD_TORCH_CHECK(activations.device().is_cuda(),
                  "Qwen4Exp SM86 hyperconnection activations must be CUDA");
  STD_TORCH_CHECK(weights.device().is_cuda(),
                  "Qwen4Exp SM86 hyperconnection weights must be CUDA");
  STD_TORCH_CHECK(activations.device() == weights.device(),
                  "Qwen4Exp SM86 hyperconnection tensors must share a device");
  STD_TORCH_CHECK(activations.scalar_type() == ScalarType::BFloat16 &&
                      weights.scalar_type() == ScalarType::BFloat16,
                  "Qwen4Exp SM86 hyperconnection tensors must be BF16");
  STD_TORCH_CHECK(activations.is_contiguous() && weights.is_contiguous(),
                  "Qwen4Exp SM86 hyperconnection tensors must be contiguous");
  STD_TORCH_CHECK(activations.dim() == 2 && weights.dim() == 2,
                  "Qwen4Exp SM86 hyperconnection tensors must be 2-D");
  STD_TORCH_CHECK(activations.size(0) == 1 || activations.size(0) == 2,
                  "Qwen4Exp SM86 hyperconnection supports M=1 or M=2");
  STD_TORCH_CHECK(activations.size(1) == weights.size(1),
                  "Qwen4Exp SM86 hyperconnection K dimensions must match");

  const auto* properties = get_device_prop();
  STD_TORCH_CHECK(properties->major == 8 && properties->minor == 6,
                  "Qwen4Exp SM86 hyperconnection requires compute capability "
                  "8.6");

  const int tokens = static_cast<int>(activations.size(0));
  const int output_features = static_cast<int>(weights.size(0));
  const int input_features = static_cast<int>(weights.size(1));
  const bool is_down = output_features == kDownOutputFeatures &&
                       input_features == kDownInputFeatures;
  const bool is_up = output_features == kUpOutputFeatures &&
                     input_features == kUpInputFeatures;
  STD_TORCH_CHECK(is_down || (is_up && tokens == 1),
                  "Qwen4Exp SM86 hyperconnection received unsupported "
                  "M/N/K geometry");

  const torch::stable::accelerator::DeviceGuard device_guard(
      activations.get_device_index());
  auto output =
      torch::stable::empty({tokens, output_features}, ScalarType::BFloat16,
                           std::nullopt, activations.device());
  const auto* activation_pointer =
      reinterpret_cast<const __nv_bfloat16*>(activations.const_data_ptr());
  const auto* weight_pointer =
      reinterpret_cast<const __nv_bfloat16*>(weights.const_data_ptr());
  auto* output_pointer =
      reinterpret_cast<__nv_bfloat16*>(output.mutable_data_ptr());
  const cudaStream_t stream =
      get_current_cuda_stream(activations.get_device_index());

  if (is_down && tokens == 1) {
    launch_qwen4_exp_hyperconnection_bf16<1, 1, 256>(
        activation_pointer, weight_pointer, output_pointer, output_features,
        input_features, stream);
  } else if (is_down) {
    launch_qwen4_exp_hyperconnection_bf16<2, 1, 256>(
        activation_pointer, weight_pointer, output_pointer, output_features,
        input_features, stream);
  } else {
    launch_qwen4_exp_hyperconnection_bf16<1, 4, 32>(
        activation_pointer, weight_pointer, output_pointer, output_features,
        input_features, stream);
  }

  const cudaError_t error = cudaGetLastError();
  STD_TORCH_CHECK(error == cudaSuccess,
                  "Qwen4Exp SM86 hyperconnection launch failed: ",
                  cudaGetErrorString(error));
  return output;
}

STABLE_TORCH_LIBRARY_FRAGMENT(_C, qwen4_exp_hyperconnection_sm86_ops) {
  qwen4_exp_hyperconnection_sm86_ops.def(
      "qwen4_exp_hyperconnection_bf16_sm86(Tensor activations, Tensor weights) "
      "-> Tensor");
}

STABLE_TORCH_LIBRARY_IMPL(_C, CUDA, qwen4_exp_hyperconnection_sm86_ops) {
  qwen4_exp_hyperconnection_sm86_ops.impl(
      "qwen4_exp_hyperconnection_bf16_sm86",
      TORCH_BOX(&qwen4_exp_hyperconnection_bf16_sm86));
}
