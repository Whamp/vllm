/* ref_a.c — L0 oracle reference A (pinned implementation).
 *
 * Verbatim extraction from Whamp/llama.cpp @ 0379cf4bf889f3d28038a005210c4bc193fc8ba1
 * (local study checkout /home/will/projects/llama.cpp-ds4-study):
 *   - block layouts and lookup tables: ggml/src/ggml-common.h (host path:
 *     static const tables; includes iq2xxs_grid, ksigns_iq2xs, kmask_iq2xs)
 *   - dequantize_row_q8_0:   ggml/src/ggml-quants.c:491
 *   - dequantize_row_q2_K:   ggml/src/ggml-quants.c:899
 *   - dequantize_row_iq2_xxs: ggml/src/ggml-quants.c:2412
 *   - fp16->fp32 conversion: ggml/src/ggml-impl.h:385 ggml_compute_fp16_to_fp32
 *     (bit-exact IEEE conversion; replicated verbatim below)
 *
 * This file exists only to gate the GGUF-TP format contract (class-A oracle).
 * It is not linked into any runtime. Any edit other than re-extraction from
 * the pinned source invalidates the oracle.
 *
 * Build: cc -O2 -shared -fPIC -I<study>/ggml/src ref_a.c -o ref_a.so
 */

#include <stdint.h>
#include <stddef.h>
#include <string.h>
#include <assert.h>

#define GGML_COMMON_DECL_C   /* host-C struct branch of ggml-common.h */
#define GGML_COMMON_IMPL_C   /* host-C table branch (iq2xxs_grid, ksigns, kmask) */
#define GGML_RESTRICT restrict

#include "ggml-common.h" /* block structs + all tables, pinned revision */

/* --- fp16 -> fp32, verbatim from ggml-impl.h:365-411 --- */
static inline float ref_fp32_from_bits(uint32_t w) {
    union { uint32_t b; float f; } v = { w };
    return v.f;
}
static inline uint32_t ref_fp32_to_bits(float f) {
    union { float f; uint32_t b; } v = { f };
    return v.b;
}
static inline float ggml_compute_fp16_to_fp32(ggml_half h) {
    const uint32_t w = (uint32_t) h << 16;
    const uint32_t sign = w & UINT32_C(0x80000000);
    const uint32_t two_w = w + w;

    const uint32_t exp_offset = UINT32_C(0xE0) << 23;
    const float exp_scale = 0x1.0p-112f;
    const float normalized_value = ref_fp32_from_bits((two_w >> 4) + exp_offset) * exp_scale;

    const uint32_t magic_mask = UINT32_C(126) << 23;
    const float magic_bias = 0.5f;
    const float denormalized_value = ref_fp32_from_bits((two_w >> 17) | magic_mask) - magic_bias;

    const uint32_t denormalized_cutoff = UINT32_C(1) << 27;
    const uint32_t result = sign |
        (two_w < denormalized_cutoff ? ref_fp32_to_bits(denormalized_value) : ref_fp32_to_bits(normalized_value));
    return ref_fp32_from_bits(result);
}
#define GGML_FP16_TO_FP32(x) ggml_compute_fp16_to_fp32(x)

/* --- dequantize_row_q8_0, verbatim body from ggml-quants.c:491 --- */
void ref_a_q8_0(const block_q8_0 * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    static const int qk = QK8_0;

    assert(k % qk == 0);

    const int nb = k / qk;

    for (int i = 0; i < nb; i++) {
        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int j = 0; j < qk; ++j) {
            y[i*qk + j] = x[i].qs[j]*d;
        }
    }
}

/* --- dequantize_row_q2_K, verbatim body from ggml-quants.c:899 --- */
void ref_a_q2_K(const block_q2_K * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int nb = k / QK_K;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);
        const float min = GGML_FP16_TO_FP32(x[i].dmin);

        const uint8_t * q = x[i].qs;

        int is = 0;
        float dl, ml;
        for (int n = 0; n < QK_K; n += 128) {
            int shift = 0;
            for (int j = 0; j < 4; ++j) {

                uint8_t sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l] >> shift) & 3)) - ml;

                sc = x[i].scales[is++];
                dl = d * (sc & 0xF); ml = min * (sc >> 4);
                for (int l = 0; l < 16; ++l) *y++ = dl * ((int8_t)((q[l+16] >> shift) & 3)) - ml;

                shift += 2;
            }
            q += 32;
        }
    }
}

/* --- dequantize_row_iq2_xxs, verbatim body from ggml-quants.c:2412 --- */
void ref_a_iq2_xxs(const block_iq2_xxs * GGML_RESTRICT x, float * GGML_RESTRICT y, int64_t k) {
    assert(k % QK_K == 0);
    const int64_t nb = k / QK_K;

    uint32_t aux32[2];
    const uint8_t * aux8 = (const uint8_t *)aux32;

    for (int i = 0; i < nb; i++) {

        const float d = GGML_FP16_TO_FP32(x[i].d);

        for (int ib32 = 0; ib32 < QK_K/32; ++ib32) {
            memcpy(aux32, x[i].qs + 4*ib32, 2*sizeof(uint32_t));
            const float db = d * (0.5f + (aux32[1] >> 28)) * 0.25f;
            for (int l = 0; l < 4; ++l) {
                const uint8_t * grid = (const uint8_t *)(iq2xxs_grid + aux8[l]);
                const uint8_t  signs = ksigns_iq2xs[(aux32[1] >> 7*l) & 127];
                for (int j = 0; j < 8; ++j) {
                    y[j] = db * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
                y += 8;
            }
        }
    }
}

/* --- layout facts + table exposure for the evidence report --- */
static size_t ref_a_sizes_v[3]  = { sizeof(block_q8_0), sizeof(block_q2_K), sizeof(block_iq2_xxs) };
static size_t ref_a_off_qs_v[3] = { offsetof(block_q8_0, qs), offsetof(block_q2_K, qs),
                                    offsetof(block_iq2_xxs, qs) };

const size_t * ref_a_sizes(void)  { return ref_a_sizes_v; }
const size_t * ref_a_off_qs(void) { return ref_a_off_qs_v; }

const void * ref_a_table(const char * name, size_t * bytes) {
    if (!strcmp(name, "iq2xxs_grid"))  { *bytes = sizeof(iq2xxs_grid);  return iq2xxs_grid; }
    if (!strcmp(name, "ksigns_iq2xs")) { *bytes = sizeof(ksigns_iq2xs); return ksigns_iq2xs; }
    if (!strcmp(name, "kmask_iq2xs"))  { *bytes = sizeof(kmask_iq2xs);  return kmask_iq2xs; }
    *bytes = 0;
    return NULL;
}
