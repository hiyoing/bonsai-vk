// ptq_decode.h - the single home of PTQ1_0 ternary decoding for this runtime.
//
// WHY A HEADER AND NOT CODE IN bonsai_vk.cpp: the faithful-reproduction test (align_dequant.cpp)
// has to measure the arithmetic the ENGINE actually uses. A second copy of the decoder inside the
// test would prove nothing about bonsai_vk.exe, so both include this file and both call the same
// functions. The test's third party is the official shader, run on the card.
//
// FORMAT (measured against models/Ternary-Bonsai-2-9B-PTQ1_0.gguf by tools/gguf_dump.py, and
// read off the fork's own kernels - _mul_mat_vec_ptq1_0.comp / _dequant_ptq1_0.comp):
//   one block = 128 weights = 7 x 32-bit words = 28 bytes = 1.75 bpw
//     qs[0..15]   words 0..3 : byte j, trit t -> element 16*t + j
//     qs[16..23]  words 4..5 : byte j, trit t -> element 80 + 8*t + (j-16)
//     qh[0..1]    word 6     : byte h, trit t -> element 120 + 2*t + h
//     d           bytes 26..27 (upper half of word 6): fp16 block scale
//   weights are signed ternary {-1, 0, +1} x d, so the stored symbol must be biased by -1.
//
// The trit extraction is the fork's base-3 fixed-point recurrence, NOT a bitfield:
//   v = (b * 3^t) & 0xFF ; raw = (v * 3) >> 8 ; weight = (raw - 1) * d
// which is exact for the 5 trits that fit in a byte (3^5 = 243 <= 256).
#ifndef BONSAI_PTQ_DECODE_H
#define BONSAI_PTQ_DECODE_H

#include <cmath>
#include <cstdint>
#include <cstring>

static const uint32_t PTQ_BLOCK_BYTES = 28;   // 128 weights + f16 scale = 1.75 bpw
static const uint32_t PTQ_BLOCK_W = 7;        // 28 B / 4
static const int      PTQ_BLOCK_W_ELEMS = 128;

// raw ternary symbol of byte b after t steps of the *3 mod 256 recurrence, in {0,1,2}.
// Iterating the multiply is the same number as (b * 3^t) & 0xFF because every step reduces mod 256;
// align_dequant measures that identity against the official shader's explicit pow3[] multiply.
static inline int trit_of_byte(uint8_t b, int t) {
    uint32_t v = b;
    for (int i = 0; i < t; ++i) v = (v * 3u) & 0xFFu;
    return int((v * 3u) >> 8u);
}

// element e of a 128-weight block -> (byte index within the 28-byte block, trit index within it)
static inline void elem_to_byte_trit(int e, int &byte, int &trit) {
    if (e < 80)        { byte = e % 16;             trit = e / 16; }
    else if (e < 120)  { byte = 16 + (e - 80) % 8;  trit = (e - 80) / 8; }
    else               { byte = 24 + (e - 120) % 2; trit = (e - 120) / 2; }
}

static inline float f16_to_f32(uint16_t h) {
    uint32_t sign = uint32_t(h & 0x8000u) << 16;
    uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    const float sgn = sign ? -1.0f : 1.0f;
    if (exp == 0) {
        // A zero exponent means the value is man * 2^-24 - there is no implicit leading 1 and the
        // exponent field is not usable, so the ONLY correct conversion is the direct scaled integer.
        // Measured lesson (runs/dq_synth.txt): a normalise-then-rebase loop here - shifting man up
        // until bit 10 is set, clearing that bit and adjusting the exponent - silently returns 0.0
        // for 0x0001 (true 5.96046448e-08) and 8.34465027e-06 for 0x0123 (true 1.73449516e-05),
        // because clearing the bit it just promoted throws away the integer part of the significand.
        // The official shader disagreed with us on exactly those blocks and was the correct party:
        // float(float16_t) on the RX 550X matched IEEE-754 binary16 for all 65536 patterns.
        if (man == 0) return std::copysign(0.0f, sgn);
        return std::copysign(std::ldexp(float(man), -24), sgn);
    }
    if (exp == 0x1F) {
        if (man == 0) return std::copysign(INFINITY, sgn);
        // NaN: keep the payload, do NOT substitute a canonical quiet NaN. Measured over all 65536
        // patterns (runs/f16_map.txt): the card's float(float16_t) applies the same exponent-bias and
        // mantissa-shift formula it uses for normals, so f16 0x7C10 becomes f32 0x7F820000 - the
        // payload moved, not silenced. Returning NAN here differed from the shader on 2044 of the
        // 65536 patterns (all NaN, all value-equal, but not bit-equal), which is exactly the class
        // of drift this project is not supposed to introduce.
        uint32_t f = sign | 0x7F800000u | (man << 13);
        float nan_out; memcpy(&nan_out, &f, 4); return nan_out;
    }
    uint32_t f = sign | ((exp + 112u) << 23) | (man << 13);
    float out; memcpy(&out, &f, 4); return out;
}

// the block scale, exactly where the format puts it
static inline float ptq_scale(const uint8_t *pb) {
    return f16_to_f32(uint16_t(pb[26]) | (uint16_t(pb[27]) << 8));
}

// ---- OUR path: the dequantised weights of one block, from the two primitives above.
// This is the same arithmetic bonsai_vk.cpp's GEMV reference performs as d*(cs - xs); the test
// checks the two forms agree element for element as well as against the official shader.
static inline void ptq_dequant_ours(const uint8_t *pb, float *w /*128*/) {
    const float d = ptq_scale(pb);
    for (int e = 0; e < PTQ_BLOCK_W_ELEMS; ++e) {
        int by, tr; elem_to_byte_trit(e, by, tr);
        w[e] = (float(trit_of_byte(pb[by], tr)) - 1.0f) * d;
    }
}

// ---- OFFICIAL path, host side: a line-by-line transcription of the arithmetic in
// _dequant_ptq1_0.comp (pow3[] multiply, (v*3)>>8 - 1, the il/ir/l element numbering that shader
// uses). Written independently of elem_to_byte_trit() on purpose: if our mapping were wrong the
// two would disagree and max|err| would be non-zero.
static inline void ptq_dequant_official_cpu(const uint8_t *pb, float *w /*128*/) {
    static const uint32_t pow3[5] = {1u, 3u, 9u, 27u, 81u};     // 3^n, as the shader's const uint[5]
    const float d = ptq_scale(pb);
    // il = 0..15 is the shader's 8-element lane, e = 8*il + l
    for (uint32_t il = 0; il < 16; ++il) {
        for (uint32_t l = 0; l < 8; ++l) {
            uint32_t b; uint32_t p_val;
            if (il < 10u) {                    // e in [0..79]  : qs[0..15], half = il & 1
                const uint32_t n = il >> 1u;
                const uint32_t qs_offset = (il & 1u) != 0u ? 8u : 0u;
                b = uint32_t(pb[qs_offset + l]); p_val = pow3[n];
            } else if (il < 15u) {             // e in [80..119]: qs[16..23]
                const uint32_t n = il - 10u;
                b = uint32_t(pb[16u + l]); p_val = pow3[n];
            } else {                           // il == 15, e in [120..127]: qh[0],qh[1]
                const uint32_t n = l >> 1u;
                b = uint32_t((l & 1u) != 0u ? pb[25] : pb[24]); p_val = pow3[n];
            }
            const uint32_t v = (b * p_val) & 0xFFu;
            const float trit = float(int((v * 3u) >> 8u) - 1);
            w[8 * il + l] = trit * d;
        }
    }
}

#endif
