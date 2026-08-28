// Host-side differential proof for the GPU-free IQ1/Q456 kernel rewrite.
//
// Compiles the REAL modified headers (iq1_iq3_tables.cuh) as plain C++ with
// CUDA qualifiers shimmed away, and compares against verbatim copies of the
// pre-rewrite arithmetic extracted from git. Three parts:
//
//   A. Exhaustive proof (all 2048 entries) that kIq1SGridEven/Odd equal
//      pack_iq1_grid_parity(kIq1SGrid[i], 0/1).
//   B. Byte-identity of untouched tables vs the git parent revision.
//   C. Differential test of q45_group_dot old-vs-new over randomized and
//      adversarial blocks, Q4_K and Q5_K, bit-exact float equality.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

// ---- CUDA shims -----------------------------------------------------------

#define __device__
#define __forceinline__ inline

static inline int __dp4a(int a, int b, int c) {
  // PTX dp4a.s32.s32: four signed byte products, saturating-free accumulate.
  auto byte = [](int word, int index) -> int {
    return static_cast<int8_t>((word >> (8 * index)) & 0xFF);
  };
  int sum = c;
  for (int index = 0; index < 4; ++index) {
    sum += byte(a, index) * byte(b, index);
  }
  return sum;
}

struct Uint4Shim {
  uint32_t x, y, z, w;
};
using uint4_shim = Uint4Shim;

struct alignas(4) Half2Shim {
  uint16_t x, y;
};

static inline float half_bits_to_float(uint16_t bits) {
  const uint32_t sign = static_cast<uint32_t>(bits >> 15) & 1;
  const uint32_t exponent = static_cast<uint32_t>(bits >> 10) & 0x1F;
  const uint32_t mantissa = static_cast<uint32_t>(bits) & 0x3FF;
  uint32_t result;
  if (exponent == 0) {
    if (mantissa == 0) {
      result = sign << 31;
    } else {
      // Subnormal: normalize.
      uint32_t e = 127 - 15 + 1;
      uint32_t m = mantissa;
      while ((m & 0x400) == 0) {
        m <<= 1;
        --e;
      }
      m &= 0x3FF;
      result = (sign << 31) | (e << 23) | (m << 13);
    }
  } else if (exponent == 31) {
    result = (sign << 31) | (0xFF << 23) | (mantissa << 13);
  } else {
    result = (sign << 31) | ((exponent - 15 + 127) << 23) | (mantissa << 13);
  }
  float out;
  std::memcpy(&out, &result, sizeof(out));
  return out;
}

#define __half2 Half2Shim
static inline Half2Shim __half22float2_shim_identity(Half2Shim value) {
  return value;
}
struct Float2Shim {
  float x, y;
};
static inline Float2Shim __half22float2(Half2Shim value) {
  return {half_bits_to_float(value.x), half_bits_to_float(value.y)};
}
using float2 = Float2Shim;

// The kernels use __half2float on raw __half; not needed in q45 path beyond
// the half2 decode above.

// ---- Real modified header under test --------------------------------------

#include "iq1_iq3_tables.cuh"

namespace vllm::gguf_dsv4 {

// Shared helpers, copied verbatim from the current source (unchanged by the
// rewrite; re-verified against git during authoring of this test).

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

enum class KQuantFormat { kQ4, kQ5 };

__device__ __forceinline__ uint32_t load_u32(const uint8_t* address) {
  uint32_t value;
  memcpy(&value, address, sizeof(value));
  return value;
}

__device__ __forceinline__ uint4_shim load_u16x(const uint8_t* address) {
  return *reinterpret_cast<const uint4_shim*>(address);
}

__device__ __forceinline__ int pack_four_bytes(const int8_t* values) {
  return static_cast<uint32_t>(static_cast<uint8_t>(values[0])) |
         (static_cast<uint32_t>(static_cast<uint8_t>(values[1])) << 8) |
         (static_cast<uint32_t>(static_cast<uint8_t>(values[2])) << 16) |
         (static_cast<uint32_t>(static_cast<uint8_t>(values[3])) << 24);
}

// NEW implementation, transcribed from current q456_k_matvec.cu.
template <KQuantFormat kFormat>
__device__ __forceinline__ float new_q45_group_dot(const uint8_t* block,
                                                   int group_index,
                                                   const int* activation_packs) {
  const __half2 scales = *reinterpret_cast<const __half2*>(block);
  const float2 decoded_scales = __half22float2(scales);
  const uint8_t* packed_scales = block + 4;
  const uint8_t* quants =
      kFormat == KQuantFormat::kQ5 ? block + 48 : block + 16;
  const int segment = group_index / 2;
  const bool high_nibble = (group_index & 1) != 0;
  const uint8_t* segment_quants = quants + segment * 32;
  const uint4_shim quant_words[2] = {load_u16x(segment_quants),
                                     load_u16x(segment_quants + 16)};
  const uint32_t nibble_shift = high_nibble ? 4 : 0;
  const uint32_t words[8] = {quant_words[0].x, quant_words[0].y,
                             quant_words[0].z, quant_words[0].w,
                             quant_words[1].x, quant_words[1].y,
                             quant_words[1].z, quant_words[1].w};
  int dot = 0;
  int code_sum = 0;
  for (int pack_index = 0; pack_index < 8; ++pack_index) {
    uint32_t packed = (words[pack_index] >> nibble_shift) & 0x0f0f0f0fU;
    if constexpr (kFormat == KQuantFormat::kQ5) {
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

// OLD implementation, verbatim from git parent e7982b484~1.
template <KQuantFormat kFormat>
__device__ __forceinline__ float old_q45_group_dot(const uint8_t* block,
                                                   int group_index,
                                                   const int* activation_packs) {
  const __half2 scales = *reinterpret_cast<const __half2*>(block);
  const float2 decoded_scales = __half22float2(scales);
  const uint8_t* packed_scales = block + 4;
  const uint8_t* high_bits =
      kFormat == KQuantFormat::kQ5 ? block + 16 : nullptr;
  const uint8_t* quants =
      kFormat == KQuantFormat::kQ5 ? block + 48 : block + 16;
  const int segment = group_index / 2;
  const bool high_nibble = (group_index & 1) != 0;
  const uint8_t* segment_quants = quants + segment * 32;
  int dot = 0;
  int code_sum = 0;
  for (int pack_index = 0; pack_index < 8; ++pack_index) {
    int8_t values[4];
    for (int byte_index = 0; byte_index < 4; ++byte_index) {
      const int element = 4 * pack_index + byte_index;
      const uint8_t packed = segment_quants[element];
      uint8_t value = high_nibble ? packed >> 4 : packed & 15;
      if constexpr (kFormat == KQuantFormat::kQ5) {
        if ((high_bits[element] & (1U << group_index)) != 0) {
          value |= 16;
        }
      }
      values[byte_index] = static_cast<int8_t>(value);
    }
    dot = __dp4a(static_cast<int>(pack_four_bytes(values)),
                 activation_packs[pack_index], dot);
    code_sum = __dp4a(0x01010101, activation_packs[pack_index], code_sum);
  }
  int scale;
  int minimum;
  decode_scale_min(packed_scales, group_index, scale, minimum);
  return decoded_scales.x * static_cast<float>(scale * dot) -
         decoded_scales.y * static_cast<float>(minimum * code_sum);
}

}  // namespace vllm::gguf_dsv4

// ---- Old tables for part B (extracted from git parent revision) -----------
#include "iq1_iq3_tables_old.inc"

// ---- Test driver ----------------------------------------------------------

using namespace vllm::gguf_dsv4;

static int failures = 0;

static void check(bool condition, const char* message) {
  if (!condition) {
    ++failures;
    std::printf("FAIL: %s\n", message);
  }
}

int main() {
  // Part A: exhaustive table-split proof.
  for (int index = 0; index < 2048; ++index) {
    const uint32_t grid = kIq1SGrid[index];
    uint32_t packed_even = 0;
    uint32_t packed_odd = 0;
    for (int i = 0; i < 4; ++i) {
      packed_even |= ((grid >> (4 * (2 * i))) & 15U) << (8 * i);
      packed_odd |= ((grid >> (4 * (2 * i + 1))) & 15U) << (8 * i);
    }
    check(kIq1SGridEven[index] == packed_even,
          "even parity split matches pack_iq1_grid_parity(grid, 0)");
    check(kIq1SGridOdd[index] == packed_odd,
          "odd parity split matches pack_iq1_grid_parity(grid, 1)");
  }
  std::printf("part A: 2048-entry table-split proof done\n");

  // Part B: untouched tables byte-identical to git parent.
  check(std::memcmp(kIqSigns, kIqSignsOld, sizeof(kIqSigns)) == 0,
        "kIqSigns unchanged");
  check(std::memcmp(kIq3XXSGrid, kIq3XXSGridOld, sizeof(kIq3XXSGrid)) == 0,
        "kIq3XXSGrid unchanged");
  check(std::memcmp(kIq1SGrid, kIq1SGridOld, sizeof(kIq1SGrid)) == 0,
        "kIq1SGrid unchanged");
  std::printf("part B: untouched-table identity done\n");

  // Part C: differential q45_group_dot.
  std::mt19937 rng(20260821);
  constexpr int kBlockBytesQ5 = 176;
  constexpr int kBlockBytesQ4 = 144;

  auto run_case = [&](bool is_q5, const std::vector<uint8_t>& block_bytes,
                      const std::vector<uint8_t>& activation_bytes,
                      int group_index) {
    alignas(16) uint8_t block[256];
    std::memset(block, 0, sizeof(block));
    const size_t span = is_q5 ? kBlockBytesQ5 : kBlockBytesQ4;
    std::memcpy(block, block_bytes.data(), span);
    // Force benign finite fp16 scales in d/dmin words.
    const uint16_t one = 0x3C00;
    std::memcpy(block, &one, 2);
    std::memcpy(block + 2, &one, 2);
    int activation_packs[8];
    std::memcpy(activation_packs, activation_bytes.data(), 32);
    if (is_q5) {
      const float a = old_q45_group_dot<KQuantFormat::kQ5>(
          block, group_index, activation_packs);
      const float b = new_q45_group_dot<KQuantFormat::kQ5>(
          block, group_index, activation_packs);
      check(std::memcmp(&a, &b, sizeof(float)) == 0,
            "q5 differential mismatch");
    } else {
      const float a = old_q45_group_dot<KQuantFormat::kQ4>(
          block, group_index, activation_packs);
      const float b = new_q45_group_dot<KQuantFormat::kQ4>(
          block, group_index, activation_packs);
      check(std::memcmp(&a, &b, sizeof(float)) == 0,
            "q4 differential mismatch");
    }
  };

  // Randomized cases.
  for (int iteration = 0; iteration < 200000; ++iteration) {
    for (int is_q5 = 0; is_q5 <= 1; ++is_q5) {
      const size_t span = is_q5 ? kBlockBytesQ5 : kBlockBytesQ4;
      std::vector<uint8_t> block(span);
      for (auto& byte : block) {
        byte = static_cast<uint8_t>(rng());
      }
      // Scale region bytes are masked by decode_scale_min anyway.
      std::vector<uint8_t> activations(32);
      for (auto& byte : activations) {
        byte = static_cast<uint8_t>(rng());
      }
      run_case(is_q5, block, activations, static_cast<int>(rng() % 8));
    }
  }
  std::printf("part C: randomized differential (%d iterations) done\n",
              200000);

  // Adversarial patterns: extremes, single-bit planes, per-group sweep.
  std::vector<uint8_t> zero(kBlockBytesQ5, 0x00);
  std::vector<uint8_t> ones(kBlockBytesQ5, 0xFF);
  std::vector<uint8_t> fives(kBlockBytesQ5, 0x55);
  std::vector<uint8_t> threes(kBlockBytesQ5, 0x33);
  std::vector<std::vector<uint8_t>> patterns = {zero, ones, fives, threes};
  for (const auto& pattern : patterns) {
    for (int group_index = 0; group_index < 8; ++group_index) {
      run_case(true, pattern, ones, group_index);
      run_case(true, pattern, zero, group_index);
      run_case(false, pattern, ones, group_index);
      // Single-bit activation planes.
      for (int bit = 0; bit < 8; ++bit) {
        std::vector<uint8_t> plane(32, static_cast<uint8_t>(1u << bit));
        run_case(true, pattern, plane, group_index);
        run_case(false, pattern, plane, group_index);
      }
      // Single-bit weight planes.
      for (int bit = 0; bit < 8; ++bit) {
        std::vector<uint8_t> plane(pattern);
        const uint8_t mask = static_cast<uint8_t>(1u << bit);
        for (auto& byte : plane) {
          byte = mask;
        }
        run_case(true, plane, ones, group_index);
        run_case(false, plane, ones, group_index);
      }
    }
  }
  // Q5 high-bit-plane sweep: each group's plane bit set one at a time.
  for (int group_index = 0; group_index < 8; ++group_index) {
    for (int element = 0; element < 32; ++element) {
      std::vector<uint8_t> block(kBlockBytesQ5, 0x00);
      block[16 + element] = static_cast<uint8_t>(1u << group_index);
      for (int i = 0; i < 128; ++i) {
        block[48 + i] = static_cast<uint8_t>(0x11 + i);
      }
      std::vector<uint8_t> acts(32);
      for (auto& byte : acts) {
        byte = static_cast<uint8_t>(rng());
      }
      run_case(true, block, acts, group_index);
    }
  }
  std::printf("part C: adversarial differential done\n");

  if (failures == 0) {
    std::printf("ALL HOST DIFFERENTIAL CHECKS PASSED\n");
    return 0;
  }
  std::printf("%d FAILURES\n", failures);
  return 1;
}
