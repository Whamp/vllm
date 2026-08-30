// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#include <c10/cuda/CUDAException.h>
#include <c10/cuda/CUDAStream.h>
#include <cuda_bf16.h>
#include <torch/extension.h>

namespace {

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

template <int Tokens, int OutputsPerBlock>
void launch_for_block_size(int block_threads, const __nv_bfloat16* activations,
                           const __nv_bfloat16* weights, __nv_bfloat16* output,
                           int output_features, int input_features,
                           cudaStream_t stream) {
  const int blocks = output_features / OutputsPerBlock;
  switch (block_threads) {
    case 32:
      qwen4_exp_hyperconnection_bf16_kernel<Tokens, OutputsPerBlock, 32>
          <<<blocks, 32, 0, stream>>>(activations, weights, output,
                                      output_features, input_features);
      break;
    case 64:
      qwen4_exp_hyperconnection_bf16_kernel<Tokens, OutputsPerBlock, 64>
          <<<blocks, 64, 0, stream>>>(activations, weights, output,
                                      output_features, input_features);
      break;
    case 128:
      qwen4_exp_hyperconnection_bf16_kernel<Tokens, OutputsPerBlock, 128>
          <<<blocks, 128, 0, stream>>>(activations, weights, output,
                                       output_features, input_features);
      break;
    case 256:
      qwen4_exp_hyperconnection_bf16_kernel<Tokens, OutputsPerBlock, 256>
          <<<blocks, 256, 0, stream>>>(activations, weights, output,
                                       output_features, input_features);
      break;
    default:
      TORCH_CHECK(false,
                  "Qwen hyperconnection SM86 block size must be 32/64/128/256");
  }
}

template <int Tokens>
void launch_for_outputs_per_block(int outputs_per_block, int block_threads,
                                  const __nv_bfloat16* activations,
                                  const __nv_bfloat16* weights,
                                  __nv_bfloat16* output, int output_features,
                                  int input_features, cudaStream_t stream) {
  switch (outputs_per_block) {
    case 1:
      launch_for_block_size<Tokens, 1>(block_threads, activations, weights,
                                       output, output_features, input_features,
                                       stream);
      break;
    case 4:
      launch_for_block_size<Tokens, 4>(block_threads, activations, weights,
                                       output, output_features, input_features,
                                       stream);
      break;
    case 8:
      launch_for_block_size<Tokens, 8>(block_threads, activations, weights,
                                       output, output_features, input_features,
                                       stream);
      break;
    default:
      TORCH_CHECK(false,
                  "Qwen hyperconnection SM86 outputs per block must be 1/4/8");
  }
}

}  // namespace

void qwen4_exp_hyperconnection_bf16_sm86(torch::Tensor activations,
                                         torch::Tensor weights,
                                         torch::Tensor output,
                                         int64_t block_threads,
                                         int64_t outputs_per_block) {
  TORCH_CHECK(activations.is_cuda(),
              "Qwen hyperconnection activations must be CUDA");
  TORCH_CHECK(weights.is_cuda(), "Qwen hyperconnection weights must be CUDA");
  TORCH_CHECK(output.is_cuda(), "Qwen hyperconnection output must be CUDA");
  TORCH_CHECK(activations.scalar_type() == at::ScalarType::BFloat16,
              "Qwen hyperconnection activations must be BF16");
  TORCH_CHECK(weights.scalar_type() == at::ScalarType::BFloat16,
              "Qwen hyperconnection weights must be BF16");
  TORCH_CHECK(output.scalar_type() == at::ScalarType::BFloat16,
              "Qwen hyperconnection output must be BF16");
  TORCH_CHECK(activations.is_contiguous(),
              "Qwen hyperconnection activations must be contiguous");
  TORCH_CHECK(weights.is_contiguous(),
              "Qwen hyperconnection weights must be contiguous");
  TORCH_CHECK(output.is_contiguous(),
              "Qwen hyperconnection output must be contiguous");
  TORCH_CHECK(activations.dim() == 2,
              "Qwen hyperconnection activations must be 2-D");
  TORCH_CHECK(weights.dim() == 2, "Qwen hyperconnection weights must be 2-D");
  TORCH_CHECK(output.dim() == 2, "Qwen hyperconnection output must be 2-D");
  TORCH_CHECK(activations.size(0) == 1 || activations.size(0) == 2,
              "Qwen hyperconnection SM86 kernel supports M=1 or M=2");
  TORCH_CHECK(activations.size(1) == weights.size(1),
              "Qwen hyperconnection activation and weight K must match");
  TORCH_CHECK(output.size(0) == activations.size(0) &&
                  output.size(1) == weights.size(0),
              "Qwen hyperconnection output shape mismatch");
  TORCH_CHECK(activations.size(1) % 2 == 0,
              "Qwen hyperconnection input features must be BF16-pair aligned");
  TORCH_CHECK(
      weights.size(0) % outputs_per_block == 0,
      "Qwen hyperconnection output features must divide outputs per block");

  const auto* activation_pointer = reinterpret_cast<const __nv_bfloat16*>(
      activations.data_ptr<at::BFloat16>());
  const auto* weight_pointer =
      reinterpret_cast<const __nv_bfloat16*>(weights.data_ptr<at::BFloat16>());
  auto* output_pointer =
      reinterpret_cast<__nv_bfloat16*>(output.data_ptr<at::BFloat16>());
  cudaStream_t stream = c10::cuda::getCurrentCUDAStream().stream();

  if (activations.size(0) == 1) {
    launch_for_outputs_per_block<1>(
        outputs_per_block, block_threads, activation_pointer, weight_pointer,
        output_pointer, weights.size(0), weights.size(1), stream);
  } else {
    launch_for_outputs_per_block<2>(
        outputs_per_block, block_threads, activation_pointer, weight_pointer,
        output_pointer, weights.size(0), weights.size(1), stream);
  }
  C10_CUDA_KERNEL_LAUNCH_CHECK();
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
  module.def("run", &qwen4_exp_hyperconnection_bf16_sm86,
             "Qwen3.8 hyperconnection BF16 skinny GEMM for SM86");
}
