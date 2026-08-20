// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright contributors to the vLLM project

#pragma once

#include <cuda_runtime.h>

namespace vllm::gguf_dsv4 {

__device__ __forceinline__ float warp_max_q8_1(float value) {
#pragma unroll
  for (int offset = 16; offset > 0; offset >>= 1) {
    value = fmaxf(value, __shfl_down_sync(0xffffffff, value, offset));
  }
  return __shfl_sync(0xffffffff, value, 0);
}

__device__ __forceinline__ int8_t quantize_q8_1_code(float value,
                                                     float absolute_max) {
  if (absolute_max == 0.0f) {
    return 0;
  }
  const float scale = absolute_max / 127.0f;
  return static_cast<int8_t>(roundf(value / scale));
}

}  // namespace vllm::gguf_dsv4
