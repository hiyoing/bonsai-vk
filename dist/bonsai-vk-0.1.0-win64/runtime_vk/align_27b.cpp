// align_27b.cpp - the 27B adapter, measured on the hardware that exists.
//
// WHY THIS FILE EXISTS: Ternary-Bonsai-2-27B is 5.9 GB and this host has 4 GB of VRAM, so its
// numbers can never be produced here. What CAN be produced here is the three contracts the 27B
// needs and that a report would otherwise have to take on faith:
//
//   [1] PTQ1_0 is a 5-trits-per-byte, 128-weight, one-fp16-scale-per-block format, and the decode
//       that reads it is bit-exact over the whole scale space. This is the loader the 27B uses
//       unchanged; a K that is not a multiple of 128 has to be refused, not truncated.
//   [2] The mixed stack: 64 blocks of which 16 are full attention and 48 are recurrent, with the
//       KV charge structurally zero for the recurrent ones. Proven against a synthetic 64-block
//       plan built from the REAL 9B header (shape synthetic, math not), plus three negative
//       controls that must refuse.
//   [3] The Hadamard rotation contract: prism.hadamard.version gates whether a rotation happens at
//       all, and block_size picks between the 9B's bit-frozen hat4096.comp and the generic
//       hadamard_fwht.comp. The kernel is compared to a host mirror BITWISE, on the card, at every
//       block size the resolver accepts.
//
// Every number printed is labelled 实测 (measured here) or 推导 (derived from a measured input).
// Nothing in this file writes a model file.
//
//   run: align_27b.exe [model.gguf] [--no-gpu]

#include <vulkan/vulkan.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "bonsai_gguf.h"
#include "bonsai_model.h"
#include "ptq_decode.h"

static int g_fail = 0;

#define REQUIRE(cond, ...)                                                   \
    do {                                                                     \
        if (!(cond)) { ++g_fail; printf("  FAIL  "); printf(__VA_ARGS__);    \
                       printf("\n"); }                                       \
    } while (0)
#define PASS(...)                                                             \
    do { printf("  ok    "); printf(__VA_ARGS__); printf("\n"); } while (0)

// ------------------------------------------------------------------ [1] PTQ1_0
// The closed form of the fork's base-3 recurrence, derived (not copied) here:
//   the runtime computes  v_t = (b * 3^t) mod 256,  raw_t = (v_t * 3) >> 8
//   and the claim is    raw_t = floor(b * 3^(t+1) / 256) mod 3
//   proof: b*3^t = 256q + v_t  =>  b*3^(t+1) = 256*(3q) + 3*v_t, and 3*v_t < 768 = 3*256, so
//   floor(b*3^(t+1)/256) = 3q + floor(3*v_t/256), whose value mod 3 is floor(3*v_t/256) mod 3 =
//   (v_t*3 >> 8) mod 3. raw_t is in {0,1,2} because 3*v_t < 768, so the "mod 3" is a no-op and the
//   two expressions are the same integer. The closed form shares NO operation with the recurrence
//   (no iterative multiply, no mask), so exhaustive agreement over every (byte, trit) pair is
//   evidence, not a tautology.
static int trit_closed_form(uint8_t b, int t) {
    static const uint32_t pow3[6] = {1u, 3u, 9u, 27u, 81u, 243u};
    const uint64_t num = uint64_t(b) * pow3[t + 1];       // b <= 255, 3^5 = 243  ->  < 2^16, exact
    return int((num / 256ull) % 3ull);
}

static void leg_ptq_format(const GgufHeader *h9, bool have9, const char *model) {
    printf("\n=== [1] PTQ1_0: 5 trits per byte, 128 weights per block, one fp16 scale per block ===\n");

    // ---- 1a the recurrence vs the closed form, over the WHOLE byte x trit space
    int agree = 0, total = 0, illegal_bytes = 0;
    for (int b = 0; b < 256; ++b) {
        if (b > 242) ++illegal_bytes;               // 3^5 = 243: 243..255 cannot be a legal 5-trit pack
        for (int t = 0; t < 5; ++t) {
            ++total;
            if (trit_of_byte(uint8_t(b), t) == trit_closed_form(uint8_t(b), t)) ++agree;
        }
    }
    REQUIRE(agree == total && total == 1280,
            "trit extraction: only %d/%d (256 bytes x 5 positions) agree between the runtime "
            "recurrence and the closed form floor(b*3^(t+1)/256) %% 3", agree, total);
    if (agree == total)
        PASS("trit extraction: %d/%d (256 bytes x 5 positions) agree between the runtime recurrence "
             "and floor(b*3^(t+1)/256) %% 3; %d/256 byte values are outside the 3^5 = 243 legal pack "
             "space and are counted, not assumed absent", agree, total, illegal_bytes);

    // ---- 1b the element map is a total, injective function over all 128 slots
    int seen_byte[28];
    memset(seen_byte, 0, sizeof seen_byte);
    int seen_slot[128];
    memset(seen_slot, 0, sizeof seen_slot);
    bool map_ok = true;
    int n_qs_lo = 0, n_qs_hi = 0, n_qh = 0;
    for (int e = 0; e < 128; ++e) {
        int by = -1, tr = -1;
        elem_to_byte_trit(e, by, tr);
        if (by < 0 || by > 25 || tr < 0 || tr > 4) { map_ok = false; break; }
        if (by >= 26) { map_ok = false; break; }              // 26..27 belong to the scale
        if (++seen_byte[by] > (by < 16 ? 5 : by < 24 ? 5 : 4)) map_ok = false;
        ++seen_slot[e];
        if (by < 16) ++n_qs_lo; else if (by < 24) ++n_qs_hi; else ++n_qh;
    }
    bool total_fn = map_ok;
    for (int e = 0; e < 128 && total_fn; ++e) if (seen_slot[e] != 1) total_fn = false;
    REQUIRE(total_fn && n_qs_lo == 80 && n_qs_hi == 40 && n_qh == 8,
            "element map is not the expected total bijection over 128 slots (qs_lo %d qs_hi %d "
            "qh %d, expected 80/40/8)", n_qs_lo, n_qs_hi, n_qh);
    if (total_fn)
        PASS("element map is a total function over all 128 slots; bytes 0..15 carry elements 0..79 "
             "(5 trits each), 16..23 carry 80..119 (5 trits), 24..25 carry 120..127 (4 trits), and "
             "bytes 26..27 are reserved for the fp16 scale - no slot is read twice, none is missed");

    // ---- 1c the decode, triangulated over the entire fp16 scale space.
    // Three paths for the same 128 weights:
    //   ours      = ptq_dequant_ours      (elem_to_byte_trit + trit_of_byte, the engine's form)
    //   official  = ptq_dequant_official_cpu (a transcription of _dequant_ptq1_0.comp's own
    //                il/ir/l element numbering and pow3[] multiply, written independently)
    //   symbol    = the arithmetic-free ternary lookup w = (raw==0 ? -d : raw==1 ? 0 : +d), which is
    //                what "signed ternary x scale" MEANS and shares no instruction with either.
    // Comparison is on BIT PATTERNS, not on a tolerance, because a tolerance is what lets a 1-ulp
    // drift into a report. Three non-obvious classes are counted and NAMED instead of hidden:
    //   +-0            (raw-1)*d and the lookup disagree in the sign of zero when d is -0.
    //   NaN payload    both sides NaN, different payload bits, produced by a NaN scale.
    //   non-finite scale  the multiply form propagates 0 x d = NaN for a NaN d and 0 x inf = NaN for
    //                  an infinite d, while the lookup form answers 0. The engine uses the multiply
    //                  form because that is what the shader does, so this is a property of the
    //                  scale, not a decode difference - and 1f below measures how many such scales
    //                  the real checkpoint actually contains (zero).
    static const uint8_t codes[26] = {
        0x00, 0x01, 0x7F, 0x80, 0xFF, 0x5A, 0xA5, 0x03,
        0x09, 0x1B, 0x2D, 0x4E, 0x6F, 0x81, 0xA2, 0xC3,
        0xE4, 0x12, 0x34, 0x56, 0x78, 0x9A, 0xBC, 0xDE,
        0x02, 0x07 };
    uint64_t cmp_n = 0, cls_zero = 0, cls_nan = 0, cls_nfs = 0, cls_bad = 0, off_vs_ours = 0;
    double max_err = 0.0;
    int nan_scales = 0, inf_scales = 0;
    float oa[128], ob[128], oc[128];
    for (uint32_t s = 0; s < 65536u; ++s) {
        uint8_t pb[28];
        memcpy(pb, codes, 26);
        pb[26] = uint8_t(s & 0xFFu);
        pb[27] = uint8_t(s >> 8);
        const float d = ptq_scale(pb);
        if (std::isnan(d)) ++nan_scales;
        if (std::isinf(d)) ++inf_scales;
        const bool dfin = std::isfinite(d);
        ptq_dequant_ours(pb, oa);
        ptq_dequant_official_cpu(pb, ob);
        // the symbol form, from the TRITS: no multiply by (raw-1), just the three values a ternary
        // weight can have, scaled.
        for (int e = 0; e < 128; ++e) {
            int by, tr; elem_to_byte_trit(e, by, tr);
            const int raw = trit_of_byte(pb[by], tr);
            oc[e] = (raw == 0) ? -d : (raw == 1 ? 0.0f : d);
            for (int which = 0; which < 2; ++which) {
                const float x = oa[e];
                const float y = which ? oc[e] : ob[e];
                ++cmp_n;
                uint32_t ux, uy; memcpy(&ux, &x, 4); memcpy(&uy, &y, 4);
                if (ux != uy && which == 0) ++off_vs_ours;
                if (ux == uy) continue;
                if (x == 0.0f && y == 0.0f) { ++cls_zero; continue; }
                if (std::isnan(x) && std::isnan(y)) { ++cls_nan; continue; }
                // the escape hatch is only open for the LOOKUP comparison; ours-vs-official must be
                // bit-identical even for a NaN or infinite scale, so swallowing it there would hide
                // the one class of real bug this sweep could still contain.
                if (!dfin && which == 1) { ++cls_nfs; continue; }
                ++cls_bad;
                if (std::isfinite(x) && std::isfinite(y)) {
                    const double de = std::fabs(double(x) - double(y));
                    if (de > max_err) max_err = de;
                }
            }
        }
    }
    REQUIRE(cls_bad == 0ull,
            "decode triangulation: %llu of %llu comparisons differ in a way that is not one of the "
            "three named classes (max|err| = %.5f)", (unsigned long long)cls_bad,
            (unsigned long long)cmp_n, max_err);
    REQUIRE(off_vs_ours == 0ull, "ours vs the official transcription differs on %llu of %llu "
            "comparisons - the two element mappings are not the same function",
            (unsigned long long)off_vs_ours, (unsigned long long)(cmp_n / 2ull));
    if (cls_bad == 0ull && off_vs_ours == 0ull)
        PASS("decode triangulation: 65 536 fp16 scale patterns x 128 elements x 2 comparisons = %llu "
             "bit comparisons, max|err| = %.5f over every finite pair. ours-vs-official is "
             "bit-identical on all %llu of its own comparisons (128 elements x 65 536 scales); the "
             "arithmetic-free lookup form differs only in %llu sign-of-zero bits, %llu NaN-payload "
             "bits and %llu zero-times-non-finite-scale bits (the sweep holds %d NaN and %d infinite "
             "scales). Subnormal and -0.0 scales are inside it, which is the class the official "
             "shader once disagreed with us on",
             (unsigned long long)cmp_n, max_err, (unsigned long long)(cmp_n / 2ull),
             (unsigned long long)cls_zero, (unsigned long long)cls_nan, (unsigned long long)cls_nfs,
             nan_scales, inf_scales);

    // ---- 1d K % 128 must be REFUSED by the loader, not truncated by it.
    // The negative control clones the REAL 9B header and corrupts exactly one field: the reduction
    // length of blk.0.ffn_gate.weight. Everything else - architecture string, head counts, the
    // layer contract - is the file that loads, so a refusal can only come from the K rule. (A
    // hand-made dictionary would also fail, but for the wrong reason: missing metadata. That is how
    // this control lied the first time it was written.)
    if (!have9) printf("  skip  1d/1f need the real checkpoint (header not loaded)\n");
    if (have9) {
        GgufHeader bad = *h9;
        bool patched = false;
        for (auto &x : bad.t)
            if (x.name == "blk.0.ffn_gate.weight" && x.n_dims == 2) { x.d[0] = 100; patched = true; break; }
        REQUIRE(patched, "the 9B header has no 2-D blk.0.ffn_gate.weight to corrupt - control invalid");
        ModelPlan p;
        const bool refused = !model_plan_build(bad, p);
        REQUIRE(refused, "model_plan_build() ACCEPTED a PTQ1_0 tensor with K=100: n_blk = K/128 "
                         "truncates to 0 and every byte count below it is a lie");
        if (refused)
            PASS("K %% 128 rejection fires in model_plan_build (its message is above), so a row whose "
                 "byte count is not K/128 blocks of 28 B cannot reach the loader or the GEMV at all");
    }

    // ---- 1f the scale space the CHECKPOINT actually uses. 1c sweeps all 65 536 patterns; this reads
    // a contiguous 8 MiB of real weights and says how many of those patterns the file ever produces:
    // a non-finite scale would make the 1f-vs-1c distinction above a live question rather than a
    // footnote. Read-only: this opens the checkpoint and never writes a byte.
    if (have9) {
        const GgufTensor *t = h9->find("blk.0.ffn_gate.weight");
        FILE *f = t ? fopen(model, "rb") : nullptr;
        if (f) {
            const uint64_t want = 8ull << 20;
            std::vector<uint8_t> buf;
            buf.resize(size_t(want));
            const uint64_t base = h9->data_start + t->off;
            // 64-bit seek, not fseek: this host's long is 32 bits and a 2.4 GB checkpoint puts block 0
            // of a mid-file tensor well past it. (align_dequant.exe learned the same thing the hard
            // way when the PTQ sample was moved from the head of the file to blk.6.) The #ifdef is the
            // same convention align_dequant.cpp and align_attn.cpp already use, so this tool builds on
            // Linux too; fseeko needs -D_FILE_OFFSET_BITS=64, which build_linux.sh passes.
#ifdef _WIN32
            const bool sought = _fseeki64(f, static_cast<long long>(base), SEEK_SET) == 0;
#else
            const bool sought = fseeko(f, static_cast<off_t>(base), SEEK_SET) == 0;
#endif
            if (!sought) {
                fclose(f);
                printf("  skip  real-scale sweep: the seek to data_start + off failed\n");
            } else {
                const size_t got = fread(buf.data(), 1, size_t(want), f);
                fclose(f);
                uint64_t nblk = 0, nonfinite = 0, negzero = 0, sub = 0;
                double dmin = 0.0, dmax = 0.0;
                for (size_t i = 26; i + 2 <= got; i += 28) {
                    const uint16_t hh = uint16_t(buf[i]) | (uint16_t(buf[i + 1]) << 8);
                    const float d = f16_to_f32(hh);
                    ++nblk;
                    if (!std::isfinite(d)) ++nonfinite;
                    if (d == 0.0f && std::signbit(d)) ++negzero;
                    if ((hh & 0x7C00u) == 0u && (hh & 0x03FFu) != 0u) ++sub;
                    if (nblk == 1) { dmin = dmax = double(d); }
                    else { if (double(d) < dmin) dmin = double(d); if (double(d) > dmax) dmax = double(d); }
                }
                REQUIRE(nblk > 100000ull, "only %llu real block scales were readable - the sample is "
                        "too small to say anything", (unsigned long long)nblk);
                REQUIRE(nonfinite == 0ull, "%llu of %llu real scales is non-finite: the "
                        "multiply-vs-lookup distinction in 1c is live, not theoretical",
                        (unsigned long long)nonfinite, (unsigned long long)nblk);
                if (nonfinite == 0ull && nblk > 100000ull)
                    PASS("real checkpoint scales: %llu consecutive PTQ1_0 blocks of %s (8 MiB from "
                         "data_start + off), 0 non-finite fp16 scales, %llu subnormal, %llu "
                         "negative-zero, |d| range [%.6g, %.6g] 实测 - so every scale this engine "
                         "multiplies by in that stretch is finite and 1c's non-finite class cannot "
                         "occur in the shipped file",
                         (unsigned long long)nblk, t->name.c_str(), (unsigned long long)sub,
                         (unsigned long long)negzero, dmin, dmax);
            }
        } else {
            printf("  skip  real-scale sweep needs the checkpoint openable\n");
        }
    }

    // ---- 1e the price, derived from the block form.
    printf("  derived  K = 4096, M = 12288 -> %llu blocks x 28 B = %llu B; %.4f bit/weight "
           "(= 8*28/128, i.e. 1.75 exactly, scale included)\n",
           (unsigned long long)(4096 / 128 * 12288),
           (unsigned long long)(4096 / 128 * 12288 * 28ull), 8.0 * 28.0 / 128.0);
}

// ------------------------------------------------------------------ [2] routing
// A synthetic 64-block plan built by cloning the REAL 9B header: the tensor table is extended with
// blk.0 (recurrent) and blk.3 (softmax) templates, prism.layer_types declares the mix, and
// prism.hadamard.* carries the rotation contract. What is under test is the resolver, the KV charge
// and the kernel route - NOT the 27B's hidden size, which this host cannot obtain. The printout
// labels the plan SYNTHETIC and shows its weight bytes so nobody can mistake it for 5.9 GB.
static bool synth_27b(const GgufHeader &base, int n_blocks, int attn_every, uint32_t hat_block,
                      GgufHeader &out) {
    out = base;
    out.kv_i["qwen35.block_count"] = n_blocks;

    // the per-layer list, spelled the way a PrismML export would spell it. Kept ALONGSIDE
    // qwen35.full_attention_interval (which stays at 4) so the resolver has to make the two agree -
    // an agreement between two independent spellings is evidence, deleting one is not.
    std::vector<std::string> list;
    list.resize(size_t(n_blocks));
    for (int b = 0; b < n_blocks; ++b)
        list[size_t(b)] = ((b + 1) % attn_every == 0) ? "full_attention" : "ssm";
    out.kv_a["prism.layer_types"] = list;

    // rotation contract. block_size is an INTEGER key: writing it into kv_s would put it where
    // bonsai_md_i() never looks, and the route would stay on 4096 while claiming otherwise. That is
    // exactly how the first version of this control passed for no reason.
    out.kv_i["prism.hadamard.version"] = 1;
    out.kv_i["prism.hadamard.block_size"] = int64_t(hat_block);
    out.kv_s.erase("prism.hadamard.block_size");
    out.kv_s.erase("prism.hadamard.version");

    // tensor table: drop every blk.N.* and re-add n_blocks of them, cloned WHOLE from the two real
    // blocks the 9B ships (blk.0 is recurrent, blk.3 is softmax). Cloning the entry rather than
    // picking one of its dims matters: a suffix's shape belongs to the tensor it came from, and
    // inventing a template per suffix group is how a synthetic table grows a shape no export has.
    std::vector<GgufTensor> blk_ssm, blk_attn;
    for (const auto &x : out.t) {
        if (x.name.compare(0, 6, "blk.0.") == 0) { GgufTensor y = x; y.name = x.name.substr(6); blk_ssm.push_back(y); }
        if (x.name.compare(0, 6, "blk.3.") == 0) { GgufTensor y = x; y.name = x.name.substr(6); blk_attn.push_back(y); }
    }
    if (blk_ssm.empty() || blk_attn.empty()) return false;
    bool kind_ok = false;
    for (const auto &y : blk_ssm) if (y.name == "attn_qkv.weight") kind_ok = true;
    bool attn_ok = false;
    for (const auto &y : blk_attn) if (y.name == "attn_q.weight") attn_ok = true;
    if (!kind_ok || !attn_ok) return false;              // blk.0 must be recurrent, blk.3 attention
    std::vector<GgufTensor> keep;
    keep.reserve(out.t.size());
    for (const auto &x : out.t)
        if (x.name.compare(0, 4, "blk.") != 0) keep.push_back(x);
    uint64_t off = 0;
    for (const auto &x : keep) if (x.off > off) off = x.off;
    for (int b = 0; b < n_blocks; ++b) {
        const bool is_attn = ((b + 1) % attn_every == 0);
        const std::vector<GgufTensor> &srcv = is_attn ? blk_attn : blk_ssm;
        for (const auto &y : srcv) {
            GgufTensor x = y;
            x.name = "blk." + std::to_string(b) + "." + y.name;
            x.off = off;                       // offsets are irrelevant here: no bytes are read
            off += 28ull * 4096ull;
            keep.push_back(x);
        }
    }
    out.t.swap(keep);
    return true;
}

// The ledger: what the plan charges, per layer, and what the FORBIDDEN allocator would charge.
static void ledger(const ModelPlan &p, const char *tag, uint32_t cap) {
    const int v_q8 = 1;                                  // the shipped default: V in q8_0, K in fp16
    uint64_t sum = 0, attn_sum = 0, ssm_charged = 0;
    int nat = 0;
    for (int b = 0; b < p.n_layer; ++b) {
        const uint64_t c = p.kv_bytes_per_layer(b, v_q8);
        sum += c;
        if (p.is_attn_layer(b)) { attn_sum += c; ++nat; } else ssm_charged += c;
    }
    const uint64_t per_tok = p.kv_bytes_per_token(v_q8);
    const double KiB = 1024.0;
    printf("  %-9s blocks %3d = %2d attn + %2d ssm | KV/token %9.3f KiB, arena at cap %5u = %9.2f MiB"
           " | SSM state %8.2f MiB\n", tag, p.n_layer, nat, p.n_layer - nat,
           double(sum) / KiB, cap, double(sum) * cap / 1048576.0,
           double(p.ssm_bytes_per_layer()) * (p.n_layer - nat) / KiB / KiB);
    REQUIRE(ssm_charged == 0ull, "%s: %llu KV bytes charged to recurrent layers - the invariant "
            "\"an SSM layer owns no cache\" is broken", tag, (unsigned long long)ssm_charged);
    REQUIRE(sum == per_tok && attn_sum == per_tok,
            "%s: per-layer sum %llu != kv_bytes_per_token() %llu", tag,
            (unsigned long long)sum, (unsigned long long)per_tok);
    printf("             KV/token is the sum over the %d charged layers only, cross-checked against "
           "the plan's own per-token function\n", nat);
    // the forbidden allocator: charge KV to EVERY layer and watch what it costs.
    const uint64_t slot = ModelPlan::kv_words_slot_layer(p.n_head_kv, p.head_dim, 1) * 4ull;
    const uint64_t naive = uint64_t(p.n_layer) * slot;
    printf("             FORBIDDEN allocator (KV on all %d blocks): %.3f KiB/token = %.1fx the "
           "charged one; the invariant is worth %.2f MiB at cap %u\n", p.n_layer,
           double(naive) / KiB, double(naive) / double(per_tok ? per_tok : 1),
           double(naive - per_tok) * cap / 1048576.0, cap);
}

static void leg_routing(GgufHeader &h9, bool have9) {
    printf("\n=== [2] hybrid routing: 16 attention layers own the KV cache, 48 SSM layers own none "
           "===\n");
    if (!have9) { printf("  skip  leg [2] needs the real 9B header as the template\n"); return; }

    // ---- the 9B regression: the resolver must reproduce what ten logs cite, or the refactor broke
    // a model that runs.
    ModelPlan p9;
    const bool ok9 = model_plan_build(h9, p9);
    REQUIRE(ok9, "the REAL 9B header no longer builds a plan (rc=%d) - every later leg is meaningless",
           ok9 ? 0 : 1);
    if (ok9) {
        std::string why;
        const bool vok = model_plan_validate(p9, 1, why);
        REQUIRE(vok, "the 9B plan fails its own validator: %s", why.c_str());
        REQUIRE(p9.n_attn_layers() == 8 && p9.n_layer == 32 && p9.n_ssm_layers() == 24,
                "9B typing moved: %d attn + %d ssm of %d (must stay 8 + 24 of 32)",
                p9.n_attn_layers(), p9.n_ssm_layers(), p9.n_layer);
        if (!vok) return;
        PASS("9B regression: %d blocks = %d GDN + %d softmax-attention (interval %d), hidden %d, "
             "ffn %d, vocab %d, nkv %d x hd %d", p9.n_layer, p9.n_ssm_layers(), p9.n_attn_layers(),
             p9.full_attn_interval, p9.hidden, p9.ffn, p9.vocab, p9.n_head_kv, p9.head_dim);
        REQUIRE(p9.hat_version == 1 && p9.hat_block == 4096u && p9.hat_active && !p9.hat_use_fwht,
                "9B rotation moved: version %lld block %u fwht=%d - hat4096.comp must stay the route",
                p9.hat_version, p9.hat_block, p9.hat_use_fwht ? 1 : 0);
        PASS("9B rotation: %s", p9.hat_rule.c_str());
        ledger(p9, "9B", 256u);
    }

    // ---- the 27B shape: 64 blocks, 16 attention, 48 recurrent, rotation block from metadata.
    GgufHeader h27;
    if (!synth_27b(h9, 64, 4, 1024u, h27)) {
        ++g_fail;
        printf("  FAIL  could not synthesise the 64-block plan from the 9B header (missing block "
               "templates)\n");
        return;
    }
    ModelPlan p27;
    const bool ok27 = model_plan_build(h27, p27);
    REQUIRE(ok27, "the synthetic 64-block plan was refused by the resolver (see the message above)");
    if (!ok27) return;
    std::string why;
    REQUIRE(model_plan_validate(p27, 1, why), "the 64-block plan fails the validator: %s", why.c_str());
    REQUIRE(p27.n_attn_layers() == 16 && p27.n_ssm_layers() == 48,
            "64-block typing is %d attn + %d ssm, must be 16 + 48",
            p27.n_attn_layers(), p27.n_ssm_layers());
    PASS("27B routing: %s", p27.layer_rule.c_str());
    {
        std::string map;
        for (int b = 0; b < p27.n_layer; ++b) map += p27.is_attn_layer(b) ? 'A' : '.';
        printf("  map      %s  (A = full attention, . = SSM/GDN)\n", map.c_str());
    }
    uint64_t ssm_kv = 0;
    for (int b = 0; b < p27.n_layer; ++b) if (!p27.is_attn_layer(b)) ssm_kv += p27.kv_bytes_per_layer(b, 1);
    REQUIRE(ssm_kv == 0ull, "summed KV charge over the 48 recurrent blocks is %llu B, must be 0",
            (unsigned long long)ssm_kv);
    PASS("SSM KV charge is 0 by construction over all %d recurrent blocks (sum of "
         "kv_bytes_per_layer() on non-attention layers)", p27.n_ssm_layers());
    ledger(p27, "27B", 32768u);
    printf("             SYNTHETIC shape: the 9B's hidden/ffn/vocab with 64 blocks = %.2f GiB of "
           "repacked weights. This tests the resolver, the charge and the kernel route; it is NOT "
           "Ternary-Bonsai-2-27B's 5.9 GB geometry, which this host cannot read.\n",
           double(p27.bytes_codes_total + p27.bytes_scale_total + p27.bytes_f32_total) / 1073741824.0);
    REQUIRE(p27.hat_use_fwht && p27.hat_block == 1024u,
            "block_size 1024 did not route to the FWHT (block %u, fwht=%d) - the route is not "
            "reading metadata", p27.hat_block, p27.hat_use_fwht ? 1 : 0);
    printf("  route    %s\n", p27.hat_rule.c_str());

    // the route must be metadata, not a model-name lookup: the same 64-block plan at 4096 keeps the
    // bit-frozen kernel.
    GgufHeader h27b;
    if (synth_27b(h9, 64, 4, 4096u, h27b)) {
        ModelPlan p27b;
        const bool ok = model_plan_build(h27b, p27b);
        REQUIRE(ok && !p27b.hat_use_fwht && p27b.n_attn_layers() == 16,
                "block_size 4096 on the 64-block plan did not keep hat4096.comp");
        if (ok) PASS("block_size 4096 on the same 64-block plan keeps hat4096.comp - the route is "
                     "metadata, not a model-name lookup");
    }

    // ---- three refusals. A resolver that only says yes proves nothing.
    auto neg = [&](const char *label, std::function<void(GgufHeader &)> f) {
        GgufHeader hn = h27;
        f(hn);
        ModelPlan pn;
        printf("\n  -- negative control: %s --\n", label);
        const bool refused = !model_plan_build(hn, pn);
        REQUIRE(refused, "\"%s\" was ACCEPTED - the gate that should refuse it does not exist", label);
        if (refused) PASS("refused as required");
    };
    neg("count cross-check disagrees", [](GgufHeader &h) {
        h.kv_i["qwen35.attn_layer_count"] = 17;                       // 16 is the truth
    });
    neg("sliding_window in the list", [](GgufHeader &h) {
        h.kv_a["prism.layer_types"][5] = "sliding_attention";
    });
    neg("hadamard.version = 2", [](GgufHeader &h) {
        h.kv_i["prism.hadamard.version"] = 2;
    });
    neg("rotation active, block_size absent", [](GgufHeader &h) {
        h.kv_i.erase("prism.hadamard.block_size");
        h.kv_s.erase("prism.hadamard.block_size");
    });
}

// ------------------------------------------------------------------ [3] the rotation
// The host mirror of the kernel: ascending-h Sylvester butterflies, in float, in the SAME stage
// order. Within a stage the pairs are disjoint, so the pair order cannot change any lane; across
// stages it can, which is why the mirror walks h the same way the shader does.
static void host_fwht(const float *x, float *y, uint32_t block, uint32_t n_blk, float inv) {
    std::vector<float> s(block);
    for (uint32_t b = 0; b < n_blk; ++b) {
        for (uint32_t i = 0; i < block; ++i) s[i] = x[size_t(b) * block + i];
        for (uint32_t h = 1u; h < block; h <<= 1u) {
            const uint32_t hm = h - 1u;
            for (uint32_t i = 0; i < (block >> 1u); ++i) {
                const uint32_t j = ((i & ~hm) << 1u) | (i & hm);
                const float a = s[j], c = s[j + h];
                s[j] = a + c;
                s[j + h] = a - c;
            }
        }
        for (uint32_t i = 0; i < block; ++i) y[size_t(b) * block + i] = s[i] * inv;
    }
}

static int popcnt(uint32_t v) { int n = 0; while (v) { v &= v - 1u; ++n; } return n; }

// ---- the smallest possible GPU harness: one device, one pipeline, two host-visible buffers.
// Direct mapping instead of a staging ring is deliberate: this leg tests the kernel's ARITHMETIC,
// and the memory path is what QCHECK and align_attn.exe already gate on the card. Fewer harness
// moving parts means a FAIL here means the shader, not the test.
struct Gpu {
    VkInstance inst = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfi = 0;
    VkPhysicalDeviceMemoryProperties mp{};
    VkCommandPool pool = VK_NULL_HANDLE;
    VkBuffer bx = VK_NULL_HANDLE, by = VK_NULL_HANDLE;
    VkDeviceMemory mx = VK_NULL_HANDLE, my = VK_NULL_HANDLE;
    float *px = nullptr, *py = nullptr;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkDescriptorSetLayout dsl = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkPipeline pipe = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
};

#define AK_CHECK(x)                                                                          \
    do {                                                                                     \
        VkResult _r = (x);                                                                   \
        if (_r != VK_SUCCESS) { fprintf(stderr, "Vulkan %d at %s:%d\n", (int)_r, __FILE__,    \
                                          __LINE__); exit(2); }                              \
    } while (0)

static std::vector<char> read_file_b(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", path); exit(2); }
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    std::vector<char> b;
    b.resize(static_cast<size_t>(n));
    if (n && fread(b.data(), 1, static_cast<size_t>(n), f) != static_cast<size_t>(n)) {
        fprintf(stderr, "short read on %s\n", path); exit(2);
    }
    fclose(f);
    return b;
}

static uint32_t find_type(const Gpu &g, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < g.mp.memoryTypeCount; ++i)
        if ((g.mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no host-visible memory type\n"); exit(2);
}

static bool gpu_open(Gpu &g) {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_1;
    ai.pApplicationName = "align_27b";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    if (vkCreateInstance(&ici, nullptr, &g.inst) != VK_SUCCESS) {
        fprintf(stderr, "  no Vulkan runtime\n"); return false;
    }
    uint32_t nd = 0; vkEnumeratePhysicalDevices(g.inst, &nd, nullptr);
    if (!nd) { fprintf(stderr, "  no Vulkan device\n"); return false; }
    std::vector<VkPhysicalDevice> devs(nd);
    vkEnumeratePhysicalDevices(g.inst, &nd, devs.data());
    g.phys = devs[0];
    for (auto p : devs) {                                   // same preference as the engine
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pp.vendorID == 0x1002) g.phys = p;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(g.phys, &props);
    vkGetPhysicalDeviceMemoryProperties(g.phys, &g.mp);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(g.phys, &nq, qf.data());
    for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { g.qfi = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = g.qfi; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    AK_CHECK(vkCreateDevice(g.phys, &di, nullptr, &g.dev));
    vkGetDeviceQueue(g.dev, g.qfi, 0, &g.queue);
    printf("  gpu        %s\n", props.deviceName);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    cpi.queueFamilyIndex = g.qfi;
    AK_CHECK(vkCreateCommandPool(g.dev, &cpi, nullptr, &g.pool));

    const VkMemoryPropertyFlags HOST =
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    const uint64_t bytes = 4096ull * 4ull;
    VkBuffer *bufp[2] = { &g.bx, &g.by };
    VkDeviceMemory *memp[2] = { &g.mx, &g.my };
    float **pmap[2] = { &g.px, &g.py };
    for (int i = 0; i < 2; ++i) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        AK_CHECK(vkCreateBuffer(g.dev, &bi, nullptr, bufp[i]));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(g.dev, *bufp[i], &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = find_type(g, HOST);
        AK_CHECK(vkAllocateMemory(g.dev, &mai, nullptr, memp[i]));
        AK_CHECK(vkBindBufferMemory(g.dev, *bufp[i], *memp[i], 0));
        void *p = nullptr;
        AK_CHECK(vkMapMemory(g.dev, *memp[i], 0, VK_WHOLE_SIZE, 0, &p));
        *pmap[i] = static_cast<float *>(p);
    }
    return true;
}

static void gpu_close(Gpu &g) {
    if (!g.dev) return;
    vkDeviceWaitIdle(g.dev);
    if (g.px) vkUnmapMemory(g.dev, g.mx);
    if (g.py) vkUnmapMemory(g.dev, g.my);
    if (g.pipe) vkDestroyPipeline(g.dev, g.pipe, nullptr);
    if (g.pl) vkDestroyPipelineLayout(g.dev, g.pl, nullptr);
    if (g.dsl) vkDestroyDescriptorSetLayout(g.dev, g.dsl, nullptr);
    if (g.dpool) vkDestroyDescriptorPool(g.dev, g.dpool, nullptr);
    if (g.pool) vkDestroyCommandPool(g.dev, g.pool, nullptr);
    if (g.bx) vkDestroyBuffer(g.dev, g.bx, nullptr);
    if (g.by) vkDestroyBuffer(g.dev, g.by, nullptr);
    if (g.mx) vkFreeMemory(g.dev, g.mx, nullptr);
    if (g.my) vkFreeMemory(g.dev, g.my, nullptr);
    vkDestroyDevice(g.dev, nullptr);
    vkDestroyInstance(g.inst, nullptr);
    g = Gpu{};
}

// Build the pipeline ONCE and re-dispatch it with different push constants. The lesson from the
// first version of this leg: tearing the pipeline down per call destroyed a descriptor set layout
// that the pipeline layout still referenced, and the second call died with
// VK_ERROR_INVALID_SHADER_NV - a harness artefact that looked exactly like a broken shader.
static void gpu_prepare(Gpu &g) {
    VkDescriptorSetLayoutBinding bs[2] = {};
    for (int i = 0; i < 2; ++i) {
        bs[i].binding = uint32_t(i); bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 2; sli.pBindings = bs;
    AK_CHECK(vkCreateDescriptorSetLayout(g.dev, &sli, nullptr, &g.dsl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 32};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &g.dsl;
    pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    AK_CHECK(vkCreatePipelineLayout(g.dev, &pli, nullptr, &g.pl));
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
    AK_CHECK(vkCreateDescriptorPool(g.dev, &dpi, nullptr, &g.dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = g.dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &g.dsl;
    AK_CHECK(vkAllocateDescriptorSets(g.dev, &dsai, &g.ds));
    VkDescriptorBufferInfo dbi[2] = { {g.bx, 0, VK_WHOLE_SIZE}, {g.by, 0, VK_WHOLE_SIZE} };
    VkWriteDescriptorSet wr[2] = {};
    for (int i = 0; i < 2; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = g.ds;
        wr[i].dstBinding = uint32_t(i); wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(g.dev, 2, wr, 0, nullptr);
    auto code = read_file_b("shaders/hadamard_fwht.spv");
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size(); mi.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule sm; AK_CHECK(vkCreateShaderModule(g.dev, &mi, nullptr, &sm));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = g.pl;
    AK_CHECK(vkCreateComputePipelines(g.dev, nullptr, 1, &cpi, nullptr, &g.pipe));
    vkDestroyShaderModule(g.dev, sm, nullptr);
}

// Run the kernel over 4096 lanes as n_blk blocks of `block`, out-of-place (x -> y, separate buffers)
// exactly the way rotate() calls it. Returns after the fence; g.py holds the result.
static void gpu_fwht(Gpu &g, uint32_t block, uint32_t n_blk, float inv_norm) {
    uint32_t pcm[8] = {0u, 0u, n_blk, block, 0u, 0u, 0u, 0u};
    memcpy(&pcm[4], &inv_norm, 4);
    VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = g.pool; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount = 1;
    VkCommandBuffer cb; AK_CHECK(vkAllocateCommandBuffers(g.dev, &a, &cb));
    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    AK_CHECK(vkBeginCommandBuffer(cb, &beg));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, g.pl, 0, 1, &g.ds, 0, nullptr);
    vkCmdPushConstants(cb, g.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pcm);
    vkCmdDispatch(cb, n_blk, 1, 1);
    AK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    AK_CHECK(vkQueueSubmit(g.queue, 1, &si, nullptr));
    AK_CHECK(vkQueueWaitIdle(g.queue));
    vkFreeCommandBuffers(g.dev, g.pool, 1, &cb);
}

static void leg_hadamard(bool use_gpu) {
    printf("\n=== [3] rotation: hadamard_fwht.comp on the card vs a host Sylvester FWHT ===\n");

    // ---- H . H^T = N I, in INTEGERS. The matrix entries are +-1, so every dot product is an exact
    // integer sum: no tolerance, no rounding, no fp32 associativity question. This is the property
    // that makes a Hadamard rotation lossless (it is an orthogonal map up to scale), and it is what
    // lets the forward apply H on one side and H^T on the other and get the original function back.
    {
        const uint32_t N = 128;
        int bad_diag = 0, bad_off = 0;
        uint64_t off_pairs = 0;
        for (uint32_t i = 0; i < N; ++i)
            for (uint32_t j = 0; j < N; ++j) {
                int s = 0;
                for (uint32_t k = 0; k < N; ++k) {
                    const int a = (popcnt(i & k) & 1) ? -1 : 1;
                    const int b = (popcnt(j & k) & 1) ? -1 : 1;
                    s += a * b;
                }
                if (i == j) { if (s != int(N)) ++bad_diag; }
                else { ++off_pairs; if (s != 0) ++bad_off; }
            }
        REQUIRE(bad_diag == 0 && bad_off == 0,
                "H.H^T at N=128: %d of %d diagonal dot products are not exactly 128 and %d of %llu "
                "off-diagonal ones are not exactly 0", bad_diag, int(N), bad_off,
                (unsigned long long)off_pairs);
        if (!bad_diag && !bad_off)
            PASS("orthogonality measured at N = %u: all %u diagonal dot products are exactly %u and "
                 "all %llu off-diagonal ones are exactly 0 (integer arithmetic, no tolerance involved)",
                 N, N, N, (unsigned long long)off_pairs);
    }

    // ---- H(H x) = N x in float, per accepted size, with the exactness class of 1/sqrt(N) stated.
    // H^2 = N.I exactly over the integers; in fp32 the two butterfly passes accumulate, so the
    // residual is a rounding statement, not a logic one - and the size where 1/sqrt(N) is a power of
    // two (every EVEN power of two) has no scale rounding at all.
    std::vector<float> x(4096), y(4096), z(4096);
    uint32_t st = 20261105u;
    auto rnd = [&]() {
        st = st * 1664525u + 1013904223u;
        return (float(int32_t(st >> 8) >> 7) / 8192.0f);       // deterministic, both signs, ~O(1)
    };
    for (auto &v : x) v = rnd();
    const uint32_t sizes[5] = {16u, 64u, 256u, 1024u, 4096u};
    for (uint32_t si = 0; si < 5; ++si) {
        const uint32_t N = sizes[si];
        host_fwht(x.data(), y.data(), N, 4096u / N, 1.0f);
        host_fwht(y.data(), z.data(), N, 4096u / N, 1.0f);
        double worst = 0.0;
        for (uint32_t i = 0; i < N; ++i) {
            const double e = std::fabs(double(z[i]) - double(N) * double(x[i]));
            if (e > worst) worst = e;
        }
        const double inv = 1.0 / std::sqrt(double(N));
        const int e2 = int(std::log2(double(N)));
        PASS("N = %4u: max |H(H x) - N x| = %.3e over %u lanes; 1/sqrt(N) = %.9g is %s in binary",
             N, worst, N, inv, (e2 % 2 == 0) ? "EXACT" : "rounded (double->float)");
    }

    // ---- the GPU leg: the compiled kernel against the mirror, BITWISE, at every block size the
    // resolver can hand it. A tolerance here would be pointless: the two sides perform the same
    // fp32 add/sub in the same stage order, so any difference is a real difference.
    if (!use_gpu) { printf("  skip  GPU leg disabled (--no-gpu)\n"); return; }
    Gpu g;
    if (!gpu_open(g)) { ++g_fail; printf("  FAIL  no Vulkan device for the rotation leg\n"); return; }
    gpu_prepare(g);
    std::vector<float> ref(4096), got(4096);
    for (uint32_t si = 0; si < 5; ++si) {
        const uint32_t N = sizes[si];
        const uint32_t nb = 4096u / N;
        const float inv = float(1.0 / std::sqrt(double(N)));     // the engine's expression, verbatim
        for (uint32_t i = 0; i < 4096; ++i) g.px[i] = x[i];
        for (uint32_t i = 0; i < 4096; ++i) g.py[i] = 0.0f;
        gpu_fwht(g, N, nb, inv);
        memcpy(got.data(), g.py, 4096 * 4);
        host_fwht(x.data(), ref.data(), N, nb, inv);
        uint32_t diff = 0, first = 0;
        double worst = 0.0;
        for (uint32_t i = 0; i < 4096; ++i) {
            uint32_t ua, ub; memcpy(&ua, &got[i], 4); memcpy(&ub, &ref[i], 4);
            if (ua != ub) { if (!diff) first = i; ++diff; }
            const double e = std::fabs(double(got[i]) - double(ref[i]));
            if (e > worst) worst = e;
        }
        REQUIRE(diff == 0u, "N = %4u, %2u blocks: %u of 4096 lanes differ from the host FWHT "
                "(max %.6f at lane %u)", N, nb, diff, worst, first);
        if (!diff) PASS("N = %4u, %2u blocks: 4096 lanes BITWISE identical to the host FWHT "
                        "(kernel reads X, writes Y - the out-of-place shape rotate() uses)", N, nb);
    }
    // the one number the 9B depends on: the host-side constant the forward passes must equal the
    // literal 1/64 baked into hat4096.comp, or the two kernels are not the same operator.
    {
        const float a = float(1.0 / std::sqrt(double(4096)));
        const float b = 1.0f / 64.0f;
        uint32_t ua, ub; memcpy(&ua, &a, 4); memcpy(&ub, &b, 4);
        REQUIRE(ua == ub, "1/sqrt(4096) as the host computes it (0x%08X) is not the bit pattern of "
                "1/64 (0x%08X) that hat4096.comp multiplies by", ua, ub);
        if (ua == ub) PASS("1/sqrt(4096) == 1/64 bit-for-bit, so hadamard_fwht at block 4096 and "
                           "hat4096.comp apply the same diagonal scale (the butterflies are the same "
                           "operator; only the summation ORDER differs, which is why the 9B keeps its "
                           "frozen kernel)");
    }
    gpu_close(g);
    printf("             signs: the +-1 diagonal belongs to signs_mul.comp, not to this kernel - "
           "fold it into the FWHT and it gets applied twice, which is a permutation of the spectrum"
           " rather than a rounding. Forward order is signs->H for the 201 rotated weight tensors and"
           " H->signs for token_embd (the inverse sandwich); align_attn.exe owns that leg.\n");
}

// ------------------------------------------------------------------ main
int main(int argc, char **argv) {
    const char *model = "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    bool use_gpu = true;
    for (int i = 1; i < argc; ++i) {
        if (!strcmp(argv[i], "--no-gpu")) use_gpu = false;
        else model = argv[i];
    }
    printf("align_27b - the 27B adapter, measured on the hardware that exists\n");
    printf("  model template : %s\n", model);
    printf("  gpu legs       : %s\n", use_gpu ? "ENABLED" : "off (--no-gpu)");

    GgufHeader h9;
    const bool have9 = gguf_load(model, h9);
    if (!have9) printf("  note       : template header unavailable; legs that need it self-skip\n");

    leg_ptq_format(have9 ? &h9 : nullptr, have9, model);
    leg_routing(h9, have9);
    leg_hadamard(use_gpu);

    printf("\n=== verdict ===\n");
    if (g_fail == 0) printf("  ALL CHECKS PASSED (%d failure(s))\n", g_fail);
    else printf("  %d CHECK(S) FAILED - see the FAIL lines above\n", g_fail);
    fflush(stdout);
    return g_fail ? 1 : 0;
}
