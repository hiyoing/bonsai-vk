// f16_rne_host.h - the host side of the fp16 packer, extracted verbatim from attn_vk.h so that
// bonsai_forward.h can reuse it WITHOUT dragging in the whole CPU mirror (attn_forward() and
// attn_forward_ideal() are plain `static`, so including that header into the engine translation
// unit would raise two -Wunused-function warnings, and §6.2 of HANDOVER.md makes "0 warning under
// -Wall" part of the regression gate).
//
// This is the SAME arithmetic that align_attn.exe stage F closed over 4096 floats and that
// shaders/f16_rne.glslh mirrors on the GPU: the reference engine converts float->half with
// round-to-nearest-even on the CPU, while THIS driver's packHalf2x16() lowers to V_CVT_PKRTZ_F16_F32
// (toward zero) and disagreed on 62.16% of ordinary words. One packer, two spellings, zero drift.
#ifndef BONSAI_F16_RNE_HOST_H
#define BONSAI_F16_RNE_HOST_H

#include <cmath>
#include <cstdint>
#include <cstring>

#include "ptq_decode.h"          // f16_to_f32(), whose whole 65536-pattern domain align_f16 closed

// ---------------------------------------------------------------- fp16 round-trip, host side
// f32 -> f16 with round-to-nearest-even: the direction the REFERENCE engine converts in
// (ggml_fp32_to_fp16_row / float->half on the CPU), so it is the semantics our arena must have.
// It is NOT what this card's packHalf2x16() does - align_attn stage F measured the instruction to
// truncate toward zero on 62.16% of 2048 ordinary words, which is why the shaders carry
// shaders/f16_rne.glslh instead of calling it. Stage F compares all four columns:
// host RNE vs GLSL RNE (want 0 mismatches) and host RTZ vs the raw instruction (want 0 too).
static inline uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  e = int32_t((x >> 23) & 0xFFu);          // f32 biased exponent
    const uint32_t m = x & 0x7FFFFFu;

    if (e == 0xFF) return uint16_t(sign | 0x7C00u | (m ? 0x0200u : 0u));   // Inf, or quiet-NaN
    if (e >= 113 && e <= 142) {                            // half NORMAL: half_exp = e - 112
        uint16_t h = uint16_t(sign | (uint32_t(e - 112) << 10) | (m >> 13));
        const uint32_t rest = m & 0x1FFFu;                 // the 13 bits being dropped
        if (rest > 0x1000u || (rest == 0x1000u && (h & 1u))) ++h;  // RNE; carry may hit the exponent
        return h;
    }
    if (e > 142) return uint16_t(sign | 0x7C00u);          // overflow -> +-Inf
    if (e == 0)  return uint16_t(sign);                    // f32 subnormal (<2^-126) -> signed zero
    // half SUBNORMAL. value = (2^23 + m) * 2^(e-150) and one half subnormal step is 2^-24, so the
    // integer significand is (2^23+m) >> (126-e), again with RNE.  e <= 112 is all that is left.
    const uint32_t mant  = m | 0x800000u;
    const uint32_t shift = uint32_t(126 - e);
    if (shift >= 25) return uint16_t(sign);                // below halfway to the smallest subnormal
    const uint32_t half = 1u << (shift - 1u);
    const uint32_t rest = mant & ((1u << shift) - 1u);
    uint32_t k = mant >> shift;
    if (rest > half || (rest == half && (k & 1u))) ++k;    // k == 1024 correctly becomes min normal
    return uint16_t(sign | k);
}

// The same conversion with the rounding increments removed, i.e. round TOWARD ZERO. Not used for
// anything in the engine; it exists so align_attn stage F can NAME the instruction this card's
// packHalf2x16() lowers to, with a bit-pattern count, instead of just observing a disagreement.
static inline uint16_t f32_to_f16_bits_rtz(float f) {
    uint32_t x;
    std::memcpy(&x, &f, sizeof x);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t  e = int32_t((x >> 23) & 0xFFu);
    const uint32_t m = x & 0x7FFFFFu;
    if (e == 0xFF) return uint16_t(sign | 0x7C00u | (m ? 0x0200u : 0u));
    if (e >= 113 && e <= 142) return uint16_t(sign | (uint32_t(e - 112) << 10) | (m >> 13));
    if (e > 142) return uint16_t(sign | 0x7C00u);
    if (e == 0)  return uint16_t(sign);
    const uint32_t shift = uint32_t(126 - e);
    if (shift >= 25) return uint16_t(sign);
    return uint16_t(sign | ((m | 0x800000u) >> shift));
}

static inline float  f32_to_f16_as_f32(float f) { return f16_to_f32(f32_to_f16_bits(f)); }
static inline double round16d(double x)         { return double(f32_to_f16_as_f32(float(x))); }

// The cache word layout the shaders use: word w holds elements 2w (low half) and 2w+1 (high half).
static inline uint32_t pack2(float lo, float hi) {
    return uint32_t(f32_to_f16_bits(lo)) | (uint32_t(f32_to_f16_bits(hi)) << 16);
}
// the toward-zero twin, only used by align_attn stage F to name the card's packHalf2x16() behaviour
static inline uint32_t pack2_rtz(float lo, float hi) {
    return uint32_t(f32_to_f16_bits_rtz(lo)) | (uint32_t(f32_to_f16_bits_rtz(hi)) << 16);
}
static inline float word_lo(uint32_t w) { return f16_to_f32(uint16_t(w & 0xFFFFu)); }
static inline float word_hi(uint32_t w) { return f16_to_f32(uint16_t(w >> 16)); }

#endif  // BONSAI_F16_RNE_HOST_H
