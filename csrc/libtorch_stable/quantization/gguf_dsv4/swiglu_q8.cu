// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project
//
// Fusing clamped SwiGLU, router weighting, and Q8_1 quantization follows the
// causal design of antirez/ds4@84cc882 ds4_swiglu_weighted_f32 plus its direct
// Q8 path. The stable ABI and kernel implementation are original vLLM code.

#include <torch/csrc/stable/accelerator.h>
#include <torch/csrc/stable/tensor.h>
#include <torch/headeronly/core/ScalarType.h>

#include <cuda_fp16.h>
#include <cuda_runtime.h>

#include "../../torch_utils.h"
#include "q8_1_utils.cuh"

namespace vllm::gguf_dsv4 {
namespace {

constexpr int kQ8GroupElements = 32;
constexpr int kWarpsPerBlock = 8;
constexpr int kThreadsPerBlock = 32 * kWarpsPerBlock;

__global__ void swiglu_weighted_q8_1_kernel(
    const float* __restrict__ gate, const float* __restrict__ up,
    const float* __restrict__ router_weights, __half* __restrict__ scales,
    int8_t* __restrict__ codes, int assignment_count, int intermediate_size,
    float clamp_limit) {
  const int lane = threadIdx.x & 31;
  const int warp_in_block = threadIdx.x >> 5;
  const int group = blockIdx.x * kWarpsPerBlock + warp_in_block;
  const int groups_per_assignment = intermediate_size / kQ8GroupElements;
  if (group >= assignment_count * groups_per_assignment) {
    return;
  }
  const int assignment = group / groups_per_assignment;
  const int group_in_assignment = group % groups_per_assignment;
  const int element = group_in_assignment * kQ8GroupElements + lane;
  const int offset = assignment * intermediate_size + element;
  const float gate_value = fminf(gate[offset], clamp_limit);
  const float up_value = fminf(fmaxf(up[offset], -clamp_limit), clamp_limit);
  const float silu = gate_value / (1.0f + expf(-gate_value));
  const float value = silu * up_value * router_weights[assignment];
  const float absolute_max = warp_max_q8_1(fabsf(value));
  codes[offset] = quantize_q8_1_code(value, absolute_max);
  if (lane == 0) {
    scales[group] = __float2half_rn(absolute_max / 127.0f);
  }
}

}  // namespace

void gguf_swiglu_weighted_q8_1(const torch::stable::Tensor& gate,
                               const torch::stable::Tensor& up,
                               const torch::stable::Tensor& router_weights,
                               torch::stable::Tensor& output_scales,
                               torch::stable::Tensor& output_codes,
                               double clamp_limit) {
  using torch::headeronly::ScalarType;
  const torch::stable::Tensor* tensors[] = {&gate, &up, &router_weights,
                                            &output_scales, &output_codes};
  for (const auto* tensor : tensors) {
    STD_TORCH_CHECK(tensor->device().is_cuda() && tensor->is_contiguous(),
                    "GGUF SwiGLU Q8 tensors must be contiguous CUDA tensors");
    STD_TORCH_CHECK(tensor->get_device_index() == gate.get_device_index(),
                    "GGUF SwiGLU Q8 tensors must share one CUDA device");
  }
  STD_TORCH_CHECK(gate.scalar_type() == ScalarType::Float &&
                      up.scalar_type() == ScalarType::Float &&
                      router_weights.scalar_type() == ScalarType::Float &&
                      output_scales.scalar_type() == ScalarType::Half &&
                      output_codes.scalar_type() == ScalarType::Char,
                  "GGUF SwiGLU Q8 dtype contract mismatch");
  STD_TORCH_CHECK(gate.dim() == 3 && up.sizes().equals(gate.sizes()) &&
                      router_weights.dim() == 2 && output_scales.dim() == 2 &&
                      output_codes.dim() == 2,
                  "GGUF SwiGLU Q8 tensor rank mismatch");
  const int token_count = gate.size(0);
  const int topk = gate.size(1);
  const int assignment_count = token_count * topk;
  const int intermediate_size = gate.size(2);
  STD_TORCH_CHECK(
      intermediate_size % kQ8GroupElements == 0 &&
          router_weights.size(0) == token_count &&
          router_weights.size(1) == topk &&
          output_codes.size(0) == assignment_count &&
          output_codes.size(1) == intermediate_size &&
          output_scales.size(0) == assignment_count &&
          output_scales.size(1) == intermediate_size / kQ8GroupElements,
      "GGUF SwiGLU Q8 shape mismatch");
  STD_TORCH_CHECK(clamp_limit > 0.0,
                  "GGUF SwiGLU Q8 clamp limit must be positive");

  const int device_index = gate.get_device_index();
  const torch::stable::accelerator::DeviceGuard device_guard(device_index);
  const auto stream = get_current_cuda_stream(device_index);
  const int group_count =
      assignment_count * intermediate_size / kQ8GroupElements;
  const int blocks = (group_count + kWarpsPerBlock - 1) / kWarpsPerBlock;
  swiglu_weighted_q8_1_kernel<<<blocks, kThreadsPerBlock, 0, stream>>>(
      gate.const_data_ptr<float>(), up.const_data_ptr<float>(),
      router_weights.const_data_ptr<float>(),
      reinterpret_cast<__half*>(output_scales.mutable_data_ptr()),
      output_codes.mutable_data_ptr<int8_t>(), assignment_count,
      intermediate_size, static_cast<float>(clamp_limit));
}

}  // namespace vllm::gguf_dsv4
