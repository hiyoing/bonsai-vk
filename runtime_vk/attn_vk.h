// attn_vk.h - the attention core of the runtime, written from the arithmetic we MEASURED out of the
// official PrismML/llama.cpp Vulkan graph (layer 3, -fa off), not copied from its C++.
//
// WHY A HEADER: same reason as ptq_decode.h. align_attn.cpp has to test the arithmetic the engine
// uses, so the CPU reference and (later) bonsai_vk.cpp include this file and call the same
// functions, while the third party is the GPU kernel running on the RX 550X. A private copy of the
// reference inside the test would prove nothing about the runtime.
//
// ------------------------------------------------------------------ every constant is measured
// Sources: runs/tensor_table.tsv and gguf_meta.txt (tools/gguf_dump.py), the -fa off dumps in
// runs/dbg_attn_oracle.log, the 8-value --rope-freq-base sweep runs/dbg_rope_b*.log analysed by
// tools/rope_fit.py, and the n_rot=4 metadata probe (a COPY of the model, rope_probe_nrot4.gguf).
//
//   head_dim 256, n_head 16, n_head_kv 4 (GQA group 4), 32 layers, full_attention_interval 4
//     -> softmax attention only in layers 3,7,...,31; layers 0..2 are GDN linear attention.
//   RMS norm: qwen35.attention.layer_norm_rms_epsilon = 9.999999974752427e-07 == float(1e-6f).
//     Applied PER HEAD over the 256 dims (RMS_NORM node), then a SEPARATE MUL applies
//     blk.N.attn_q_norm / attn_k_norm. V gets neither.
//   RoPE: half-split (NeoX) pairs (p, p + n_rot/2) inside the FIRST n_rot = 64 dims,
//     theta_p = freq_base^(-2p/n_rot) = 1e7^(-p/32), angle = position * theta_p,
//     out[p]      = x[p]*cos(angle) - x[p+n_rot/2]*sin(angle)
//     out[p+half] = x[p+half]*cos(angle) + x[p]*sin(angle)
//     dims >= n_rot pass through (0 mismatches measured); position is the ABSOLUTE sequence index
//     (Model add_bos measured false); mrope dimension_sections [11,11,10,0] sums to 32 = n_rot/2,
//     i.e. it splits the PAIRS, and is numerically inert for plain text because all four per-token
//     fields carry the same value. Interleaved (GPT-J style) pairing was REJECTED by measurement:
//     worst |predicted - printed| = 7.6814 with zero free parameters.
//   Scores: kq = K^T * Q with NO scale in the matmul; the scale lives in SOFT_MAX and measured
//     1/sqrt(head_dim) = 0.0625 exactly (free bisection fit 0.062501; the wrong candidates
//     1/sqrt(128), 1/sqrt(64), 0.05, 0.08, 1.0 land 5.8e-2..5.2e-1 away in probability space).
//   Mask: additive, broadcast over heads, -INF in the slots the causal (and padding) rule kills;
//     all 30 causally-masked probabilities the official engine printed are exactly 0.0000.
//   Storage: the cache SET_ROWS writes is fp16 ({1024, cap, n_layer} = 4 kv heads x 256 dims).
//     ggml's mixed-type MUL_MAT casts the f32 operand to the f16 operand's type, so Q and the
//     softmax probabilities enter their matmuls as halves with an f32 accumulate. We mirror that ON
//     PURPOSE (the fp32 leg) and charge each rounding separately against float64 (the ideal legs).
//
// ------------------------------------------------------------------ determinism, not speed
// Every reduction runs in ASCENDING INDEX ORDER in a single lane here, and the three shaders do the
// same reduction in their thread 0 (attn_norm_rope.comp's ss, attn_qk_softmax.comp's max and sum)
// or keep a whole accumulation inside one lane (the per-j dot in attn_qk_softmax.comp, the per-d sum
// over j in attn_pv.comp). That is what turns "GPU kernel vs CPU reference" from a tolerance claim
// into a bit-exact statement. The price is a serial accumulator per row - a throughput question for
// a later pass, not a correctness one. Order IS part of the arithmetic: changing it must change the
// numbers, and the ideal legs below are what measure that.
#ifndef BONSAI_ATTN_VK_H
#define BONSAI_ATTN_VK_H

#include <cmath>
#include <cstdint>
#include <cstring>
#include <vector>

#include "ptq_decode.h"          // f16_to_f32(), whose whole 65536-pattern domain align_f16 closed

// ---------------------------------------------------------------- fp16 round-trip, host side
// MOVED VERBATIM to f16_rne_host.h (same names, same bodies, same comments) so the engine can
// reuse the packer without inheriting attn_forward()/attn_forward_ideal()'s -Wunused-function.
#include "f16_rne_host.h"

static const uint32_t NEG_INF_BITS = 0xFF800000u;
static inline float neg_inf() { float f; std::memcpy(&f, &NEG_INF_BITS, 4); return f; }

// ---------------------------------------------------------------- configuration
struct AttnCfg {
    int   hd   = 256;           // head_dim: qwen35.attention.key_length == value_length
    int   nh   = 16;            // q heads
    int   nkv  = 4;             // kv heads
    int   nrot = 64;            // qwen35.rope.dimension_count
    int   seq  = 0;             // tokens in this pass
    int   cap  = 0;             // KV arena capacity in positions (<= 256: one workgroup row)
    float eps  = 1.0e-6f;       // read from the GGUF, not typed in
    float scale = 0.0625f;      // 1/sqrt(head_dim) - MEASURED, see the header
    float freq_base = 10000000.0f;
    int   gqa() const { return nh / nkv; }
    int   hw()  const { return hd / 2; }                     // cache words per head row
};

// The single place the rope frequency is written. The fork builds it in two steps
// (theta_scale = powf(base, -2/n_dims), then powf(theta_scale, p)); the one-shot powf below is the
// same function algebraically, and attn_freq_spread() MEASURES the difference instead of assuming 0.
static inline float rope_theta(const AttnCfg &c, int p) {
    return std::pow(c.freq_base, -2.0f * float(p) / float(c.nrot));
}
static inline float rope_theta_fork(const AttnCfg &c, int p) {
    const float ts = std::pow(c.freq_base, -2.0f / float(c.nrot));
    return std::pow(ts, float(p));
}
static inline double attn_freq_spread(const AttnCfg &c, int *at_p) {
    double worst = 0.0; int at = 0;
    for (int p = 0; p < c.nrot / 2; ++p) {
        const double d = std::fabs(double(rope_theta(c, p)) - double(rope_theta_fork(c, p)));
        if (d > worst) { worst = d; at = p; }
    }
    if (at_p) *at_p = at;
    return worst;
}

// ---------------------------------------------------------------- stage 1: per-head RMS norm + rope
// ggml's RMS_NORM: ss = sum(x*x) ascending; inv = 1/sqrt(ss/n + eps); y = x*inv. The weight MUL is a
// separate node, so the multiply order is (x*inv)*w - the shader does exactly that, in that order.
static inline void attn_rms_norm(const AttnCfg &c, const float *x, const float *w, float *y) {
    float ss = 0.0f;
    for (int d = 0; d < c.hd; ++d) ss = ss + x[d] * x[d];
    const float inv = 1.0f / std::sqrt(ss / float(c.hd) + c.eps);
    for (int d = 0; d < c.hd; ++d) y[d] = (x[d] * inv) * w[d];
}

// ALIAS-SAFE, and that is the whole story of leg A's 13.48 score error. The rotation writes BOTH
// elements of a pair, so the two partner values have to be latched before either store: with x == y,
// `y[p+half] = x[p+half]*cs + x[p]*sn` reads the ALREADY ROTATED y[p] instead of the input x[p]. That
// is exactly what this function used to do, and what attn_forward() fed it in place - so the mirror,
// not the card, was wrong. Evidence, from runs/attn_run5.txt: leg A saw q16 word 17 (dims 34,35) as
// (1.226562,-1.886719) on the CPU vs (0.282715,2.226562) on the GPU while leg R - same inputs, but
// called as attn_rope(cfg, y, position, z) with y != z - reported max|err| 1.19e-06 for the same row.
// It is invisible at position 0 (sn = 0) and gross everywhere else, which is why only q16 and the
// non-zero-position kc rows disagreed and vc (no rope) was bit-exact.
// The non-aliased form is also the faithful one: ggml's ROPE node writes dst while reading src as two
// separate tensors, and attn_norm_rope.comp latches the partner from sh[] (the normed input), never
// from its own output. The term order stays lo*cs -/+ hi*sn so the fp32 operation sequence still
// matches the shader lane for lane.
static inline void attn_rope(const AttnCfg &c, const float *x, int pos, float *y) {
    const int half = c.nrot / 2;
    for (int p = 0; p < half; ++p) {
        const float a  = float(pos) * rope_theta(c, p);
        const float cs = std::cos(a), sn = std::sin(a);
        const float lo = x[p], hi = x[p + half];
        y[p]        = lo * cs - hi * sn;
        y[p + half] = hi * cs + lo * sn;
    }
    for (int d = c.nrot; d < c.hd; ++d) y[d] = x[d];         // pass-through above n_rot
}

// the same two formulas in double, for the ideal legs
static inline void attn_rms_norm_d(const AttnCfg &c, const double *x, const double *w, double *y) {
    double ss = 0.0;
    for (int d = 0; d < c.hd; ++d) ss = ss + x[d] * x[d];
    const double inv = 1.0 / std::sqrt(ss / double(c.hd) + double(c.eps));
    for (int d = 0; d < c.hd; ++d) y[d] = (x[d] * inv) * w[d];
}
static inline void attn_rope_d(const AttnCfg &c, const double *x, int pos, double *y) {
    const int half = c.nrot / 2;
    for (int p = 0; p < half; ++p) {
        const double a  = double(pos) * double(rope_theta(c, p));
        const double cs = std::cos(a), sn = std::sin(a);
        const double lo = x[p], hi = x[p + half];      // latched: this one is called in place too
        y[p]        = lo * cs - hi * sn;
        y[p + half] = hi * cs + lo * sn;
    }
    for (int d = c.nrot; d < c.hd; ++d) y[d] = x[d];
}

// ---------------------------------------------------------------- stage 2: scores, mask, softmax
// One query row against the first `nfill` cached positions. q16/kc are PACKED HALVES (the arena is
// fp16 and Q enters the matmul as fp16); the products are f32 and the accumulation ascends over d.
static inline void attn_score_row(const AttnCfg &c, const uint32_t *q16,
                                  const uint32_t *kc, int kc_stride_words, int kv_head,
                                  int nfill, float *score) {
    const int hw = c.hw();
    for (int j = 0; j < nfill; ++j) {
        const uint32_t *krow = kc + size_t(j) * kc_stride_words + size_t(kv_head) * hw;
        float acc = 0.0f;
        for (int d = 0; d < hw; ++d) {
            const uint32_t qw = q16[d], kw = krow[d];
            acc = acc + word_lo(qw) * word_lo(kw);
            acc = acc + word_hi(qw) * word_hi(kw);
        }
        score[j] = acc;
    }
}

// Softmax with the measured semantics: additive mask (-INF in the dead slots, broadcast over heads),
// max over the SCALED live scores, exp(scaled - max), ascending sum, divide. A masked entry is
// exp(-inf - max) = 0 EXACTLY, the property the official print confirmed 30 times.
static inline void attn_softmax_row(const AttnCfg &c, const float *score, const float *mask,
                                    int n, float *p) {
    float mx = neg_inf();
    for (int j = 0; j < n; ++j) {
        const float v = score[j] * c.scale + mask[j];
        if (v > mx) mx = v;
    }
    float sum = 0.0f;
    for (int j = 0; j < n; ++j) {
        const float v = score[j] * c.scale + mask[j];
        p[j] = std::exp(v - mx);
        sum  = sum + p[j];
    }
    for (int j = 0; j < n; ++j) p[j] = p[j] / sum;
}

// ---------------------------------------------------------------- stage 3: PV, then the gate
// p_as_f32 are the probabilities AS THE MATMUL SEES THEM (already rounded to half under the mirror);
// vc is the fp16 V arena. The sum over j ascends inside one lane, on both sides.
static inline void attn_pv_row(const AttnCfg &c, const float *p_as_f32, const uint32_t *vc,
                               int vc_stride_words, int kv_head, int nfill, float *out) {
    const int hw = c.hw();
    for (int d = 0; d < c.hd; ++d) {
        float acc = 0.0f;
        for (int j = 0; j < nfill; ++j) {
            const uint32_t w = vc[size_t(j) * vc_stride_words + size_t(kv_head) * hw + d / 2];
            const float v = (d & 1) ? word_hi(w) : word_lo(w);
            acc = acc + p_as_f32[j] * v;
        }
        out[d] = acc;
    }
}

// The gate is the SECOND half of attn_q's 8192 outputs, SIGMOIDed and MULed into the result.
// ggml's sigmoid is 1/(1+exp(-x)); the shader writes it the same way, so it is not "close enough".
static inline void attn_gate_row(const float *pregate, const float *gate, int hd, float *out) {
    for (int d = 0; d < hd; ++d) out[d] = pregate[d] * (1.0f / (1.0f + std::exp(-gate[d])));
}

// ---------------------------------------------------------------- the buffers, OUR layout
// Deliberately NOT ggml's permuted {hd, cap, nkv} view: we own the arena, so the cache is position
// major and contiguous, which is what a wave64 GEMV wants later. Layout (all row-major):
//   q_in  [seq][nh ][hd]   k_in [seq][nkv][hd]   v_in [seq][nkv][hd]   gate_in [seq][nh][hd]
//   qn_w  [hd], kn_w [hd]   the REAL blk.3.attn_q_norm / attn_k_norm weights read from the GGUF
//   pos   [seq] uint32     absolute position per token - an integer, and the shaders read it as one
//                          (buffer POS { uint pos[] }). pos2tok[cap] maps a position back to the row
//                          of k_in/v_in that filled it, so the ideal legs work for decode too.
//                          Lesson from the first run: this field used to be float while the shader
//                          declared uint, so the card read the BIT PATTERN of 7.0f (0x40E00000) as a
//                          position, looped a billion times and took the device down with
//                          VK_ERROR_DEVICE_LOST. One type, stated here.
//   q16   [seq][nh ][hw]   packed halves, hw = hd/2
//   kc    [cap][nkv][hw]   the K arena, packed halves   } these two ARE the Vulkan KV buffer
//   vc    [cap][nkv][hw]   the V arena, packed halves   }
//   score [seq][nh][cap]   prob [seq][nh][cap]   pregate/gated [seq][nh][hd]   mask [seq][cap]
struct AttnBuffers {
    std::vector<float>    q_in, k_in, v_in, gate_in, qn_w, kn_w, mask;
    std::vector<float>    score, prob, pregate, gated;
    std::vector<uint32_t> q16, kc, vc, p16, pos;
    std::vector<int>      pos2tok;

    void alloc(const AttnCfg &c) {
        q_in.assign(size_t(c.seq) * c.nh  * c.hd, 0.0f);
        gate_in.assign(size_t(c.seq) * c.nh * c.hd, 0.0f);
        k_in.assign(size_t(c.seq) * c.nkv * c.hd, 0.0f);
        v_in.assign(size_t(c.seq) * c.nkv * c.hd, 0.0f);
        qn_w.assign(c.hd, 1.0f); kn_w.assign(c.hd, 1.0f);
        pos.assign(c.seq, 0u);
        mask.assign(size_t(c.seq) * c.cap, 0.0f);
        const int hw = c.hw();
        q16.assign(size_t(c.seq) * c.nh * hw, 0u);
        kc.assign(size_t(c.cap) * c.nkv * hw, 0u);
        vc.assign(size_t(c.cap) * c.nkv * hw, 0u);
        score.assign(size_t(c.seq) * c.nh * c.cap, 0.0f);
        prob .assign(size_t(c.seq) * c.nh * c.cap, 0.0f);
        p16  .assign(size_t(c.seq) * c.nh * (c.cap / 2), 0u);
        pregate.assign(size_t(c.seq) * c.nh * c.hd, 0.0f);
        gated  .assign(size_t(c.seq) * c.nh * c.hd, 0.0f);
        pos2tok.assign(c.cap, -1);
    }
};

// ---------------------------------------------------------------- the fp32 mirror (GPU's twin)
// The arena is fp16 because that is what the engine stores, so cache rounding is NOT a knob here.
// kRoundProb switches only the probability rounding on the way into PV, which lets the test charge
// the official mixed-type cast separately from the cache.
template <bool kRoundProb>
static void attn_forward(const AttnCfg &c, AttnBuffers &b) {
    const int hw = c.hw(), gqa = c.gqa();
    std::vector<float> y(c.hd), s(size_t(c.cap)), p(size_t(c.cap)), pr(size_t(c.cap)), o(c.hd);

    for (int t = 0; t < c.seq; ++t) {
        const int pos = int(b.pos[t]);
        b.pos2tok[pos] = t;
        for (int h = 0; h < c.nh; ++h) {                    // Q: norm, weight, rope, -> half operand
            attn_rms_norm(c, &b.q_in[(size_t(t) * c.nh + h) * c.hd], b.qn_w.data(), y.data());
            attn_rope(c, y.data(), pos, y.data());
            uint32_t *dst = &b.q16[(size_t(t) * c.nh + h) * hw];
            for (int d = 0; d < hw; ++d) dst[d] = pack2(y[2 * d], y[2 * d + 1]);
        }
        for (int h = 0; h < c.nkv; ++h) {                    // K: norm, weight, rope, -> arena
            attn_rms_norm(c, &b.k_in[(size_t(t) * c.nkv + h) * c.hd], b.kn_w.data(), y.data());
            attn_rope(c, y.data(), pos, y.data());
            uint32_t *dst = &b.kc[(size_t(pos) * c.nkv + h) * hw];
            for (int d = 0; d < hw; ++d) dst[d] = pack2(y[2 * d], y[2 * d + 1]);
            const float *v = &b.v_in[(size_t(t) * c.nkv + h) * c.hd];   // V: no norm, no rope
            uint32_t *vd = &b.vc[(size_t(pos) * c.nkv + h) * hw];
            for (int d = 0; d < hw; ++d) vd[d] = pack2(v[2 * d], v[2 * d + 1]);
        }
    }

    for (int t = 0; t < c.seq; ++t) {
        const int pos = int(b.pos[t]), nfill = pos + 1;      // causal AND filled: j in [0, pos]
        for (int h = 0; h < c.nh; ++h) {
            const int kv = h / gqa;
            attn_score_row(c, &b.q16[(size_t(t) * c.nh + h) * hw], b.kc.data(), c.nkv * hw,
                           kv, nfill, s.data());
            float *sc = &b.score[(size_t(t) * c.nh + h) * c.cap];
            for (int j = 0; j < nfill; ++j) sc[j] = s[j];
            attn_softmax_row(c, s.data(), &b.mask[size_t(t) * c.cap], nfill, p.data());
            float *pb = &b.prob[(size_t(t) * c.nh + h) * c.cap];
            uint32_t *pw = &b.p16[(size_t(t) * c.nh + h) * (size_t(c.cap) / 2)];
            for (int j = 0; j < nfill; ++j) {
                pb[j] = p[j];
                pr[j] = kRoundProb ? f32_to_f16_as_f32(p[j]) : p[j];
            }
            for (int j = nfill; j < c.cap; ++j) pr[j] = 0.0f;      // the shader writes 0 there too
            for (int j = 0; j < c.cap / 2; ++j)
                pw[j] = pack2(pr[2 * j], pr[2 * j + 1]);
            attn_pv_row(c, pr.data(), b.vc.data(), c.nkv * hw, kv, nfill, o.data());
            float *pg = &b.pregate[(size_t(t) * c.nh + h) * c.hd];
            for (int d = 0; d < c.hd; ++d) pg[d] = o[d];
            attn_gate_row(pg, &b.gate_in[(size_t(t) * c.nh + h) * c.hd], c.hd,
                          &b.gated[(size_t(t) * c.nh + h) * c.hd]);
        }
    }
}

// ---------------------------------------------------------------- the ideal legs (float64)
// round_bits says WHERE the official loses bits, so the loss is attributed instead of argued:
//   bit 0 = the fp16 cache (K and V stored as halves - MEASURED, the SET_ROWS target is fp16)
//   bit 1 = the fp16 probability operand (ggml's mixed-type cast - INFERRED from the type contract)
//   bit 2 = the fp16 Q operand, which the cache bit also covers; kept separate so the test can name
//           the two matmul operands individually.
// mode 0 is therefore "no fp16 anywhere": the arithmetic floor of this layer.
static void attn_forward_ideal(const AttnCfg &c, const AttnBuffers &b, std::vector<double> &gated,
                               int round_bits) {
    const int gqa = c.gqa();
    const bool r_cache = round_bits & 1;
    const bool r_q     = round_bits & 2;
    const bool r_prob  = round_bits & 4;
    std::vector<double> qc(size_t(c.seq) * c.nh * c.hd), kc_d(size_t(c.cap) * c.nkv * c.hd);
    std::vector<double> y(c.hd), x(c.hd), w(c.hd), s(size_t(c.cap)), p(size_t(c.cap)), o(c.hd);

    for (int t = 0; t < c.seq; ++t) {
        const int pos = int(b.pos[t]);
        for (int h = 0; h < c.nh; ++h) {
            for (int d = 0; d < c.hd; ++d) { x[d] = b.q_in[(size_t(t) * c.nh + h) * c.hd + d];
                                             w[d] = b.qn_w[d]; }
            attn_rms_norm_d(c, x.data(), w.data(), y.data());
            attn_rope_d(c, y.data(), pos, y.data());
            double *dst = &qc[(size_t(t) * c.nh + h) * c.hd];
            for (int d = 0; d < c.hd; ++d) dst[d] = r_q ? round16d(y[d]) : y[d];
        }
        for (int h = 0; h < c.nkv; ++h) {
            for (int d = 0; d < c.hd; ++d) { x[d] = b.k_in[(size_t(t) * c.nkv + h) * c.hd + d];
                                             w[d] = b.kn_w[d]; }
            attn_rms_norm_d(c, x.data(), w.data(), y.data());
            attn_rope_d(c, y.data(), pos, y.data());
            double *dst = &kc_d[(size_t(pos) * c.nkv + h) * c.hd];
            for (int d = 0; d < c.hd; ++d) dst[d] = r_cache ? round16d(y[d]) : y[d];
        }
    }

    for (int t = 0; t < c.seq; ++t) {
        const int pos = int(b.pos[t]), nfill = pos + 1;
        for (int h = 0; h < c.nh; ++h) {
            const int kv = h / gqa;
            for (int j = 0; j < nfill; ++j) {
                const double *qrow = &qc[(size_t(t) * c.nh + h) * c.hd];
                const double *krow = &kc_d[(size_t(j) * c.nkv + kv) * c.hd];
                double acc = 0.0;
                for (int d = 0; d < c.hd; ++d) acc = acc + qrow[d] * krow[d];
                s[j] = acc;
            }
            double mx = -INFINITY;
            for (int j = 0; j < nfill; ++j) {
                const double v = s[j] * double(c.scale) + double(b.mask[size_t(t) * c.cap + j]);
                if (v > mx) mx = v;
            }
            double sum = 0.0;
            for (int j = 0; j < nfill; ++j) {
                const double v = s[j] * double(c.scale) + double(b.mask[size_t(t) * c.cap + j]);
                p[j] = std::exp(v - mx); sum = sum + p[j];
            }
            for (int j = 0; j < nfill; ++j) p[j] = p[j] / sum;
            for (int d = 0; d < c.hd; ++d) {
                double acc = 0.0;
                for (int j = 0; j < nfill; ++j) {
                    const int tk = b.pos2tok[j];
                    const double pv = r_prob ? round16d(p[j]) : p[j];
                    const double raw = double(b.v_in[(size_t(tk) * c.nkv + kv) * c.hd + d]);
                    acc = acc + pv * (r_cache ? round16d(raw) : raw);
                }
                o[d] = acc;
            }
            for (int d = 0; d < c.hd; ++d) {
                const double g = double(b.gate_in[(size_t(t) * c.nh + h) * c.hd + d]);
                gated[(size_t(t) * c.nh + h) * c.hd + d] = o[d] * (1.0 / (1.0 + std::exp(-g)));
            }
        }
    }
}

#endif // BONSAI_ATTN_VK_H
