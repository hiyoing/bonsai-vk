// bonsai_model.h - the weight-side plan for the WHOLE checkpoint, decided on the host, before a
// single byte is uploaded. Pure C++ and GPU-free on purpose: the numbers it prints are the VRAM
// budget the 4 GB card lives or dies by, and they must be reviewable without a driver in the loop.
//
// WHAT IT DOES
//   1. walks the GGUF tensor table (bonsai_gguf.h, our own parser - no ggml, no llama.cpp) and
//      splits the 427 tensors into the two things this checkpoint actually contains: PTQ1_0
//      quantized weights (dtype 143, 202 tensors, 8.946e9 of the 8.946e9+ weights) and f32
//      tensors (dtype 0, 225 tensors: norms, biases, ssm_a, ssm_conv1d, ssm_alpha/beta).
//   2. assigns every tensor its offsets in the two big device-local buffers the measured GEMV
//      reads (codes [K/64][M] of uvec4 = 2 bits/weight, scales [K/128][M] of uint with the f16 in
//      the low half) and in one f32 blob for the non-quantized tensors.
//   3. prints the price of that layout against the checkpoint's own 1.75 bpw block form, because
//      the difference is the KV cache we could otherwise afford. This is the single most
//      consequential number in the whole engine and it is measured, not styled.
//
// SHAPE CONVENTION (measured from this file, runs/tensor_table.tsv): a 2-D tensor is stored
// d0 x d1 = K x M, i.e. d0 is the REDUCTION length (what the GEMV contracts over) and d1 is the
// row count. blk.0.ffn_down.weight is 12288 x 4096 -> K=12288, M=4096; blk.0.ffn_gate.weight is
// 4096 x 12288 -> K=4096, M=12288. Every PTQ1_0 tensor in the file has K in {4096, 12288}, which
// is exactly what prism.hadamard.sign_widths = [4096, 12288] says the activation widths are.
//
// ---------------------------------------------------------------- 2026-11 ADDENDUM: the mixed-stack plan
// The 9B is one model shape and the code below used to encode that as facts: an interval of 4
// hardcoded into is_attn_layer(), "qwen35." hardcoded into every metadata key, a Hadamard block of
// 4096 hardcoded into rotate(), and a KV cost of 2*nkv*hd*2*nat typed into the report. All four were
// true of the 9B and all four are the wrong thing to do for a 64-layer 16-attn/48-ssm sibling: a
// wrong layer type does not crash, it charges (or forgets to charge) a cache slot, and a forgotten
// slot is an overwrite of whatever region the arena put next. So:
//   - layer typing is now RESOLVED from metadata (three accepted spellings + one structural
//     cross-check against the tensor table), with the interval rule kept as one of the spellings so
//     the 9B's answer is bit-identical and comes from the same expression it did before;
//   - hyper-parameters are read through a prefix list (general.architecture, then prism., then
//     general., then the bare key) that REJECTS conflicting spellings instead of preferring one;
//   - the rotation contract has a version gate and a block size that comes from the file;
//   - the KV charge is a per-layer function whose value for an SSM layer is structurally 0, and the
//     engine asserts the arena it built equals that sum.
// Nothing here changed the 9B's resolved plan; align_27b.exe #R re-derives it from the real
// checkpoint and compares against the values the ten measurement logs cite.
#ifndef BONSAI_MODEL_H
#define BONSAI_MODEL_H

#include <cctype>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bonsai_gguf.h"
#include "ptq_decode.h"

// repack_ptq1_0.comp / gemv_ternary.comp addressing units
static const uint32_t CODES_PER_VEC4 = 64u;                     // 16 B word = 64 x 2-bit codes
static const uint32_t SCALE_PER_BLOCK = 1u;                     // 1 uint per 128-weight block

// A layer is either a softmax-attention block (it owns KV slots) or a fixed-size recurrent block
// (it owns a state tensor and NO cache). There is no third kind in this engine: sliding-window
// attention is a THIRD kind and the QK kernels have no window term, so the resolver below rejects
// it outright rather than mapping it onto one of these two.
enum BonsaiLayerKind : uint8_t { BONSAI_LK_SSM = 0, BONSAI_LK_ATTN = 1 };

struct WeightPlan {
    std::string name;
    uint32_t K = 0, M = 0;             // reduction length, output rows
    uint32_t dtype = 0;                // 143 PTQ1_0, 0 f32
    uint64_t file_off = 0;             // byte offset inside the data section
    uint64_t n_blk = 0;                // K / 128
    // arena placement, filled by model_plan_build()
    uint64_t ptq_word0 = 0;            // uint (4 B) units into the upload/source blob
    uint64_t codes_word0 = 0;          // uvec4 units into CODES
    uint64_t scale_word0 = 0;          // uint units into SCALES
    uint64_t f32_off = 0;              // byte offset into the f32 blob (dtype 0 only)
    uint64_t bytes_src = 0, bytes_codes = 0, bytes_scale = 0, bytes_f32 = 0;
    bool quantized() const { return dtype == 143; }
};

struct ModelPlan {
    std::vector<WeightPlan> w;
    std::vector<std::string> order;                    // tensor table order
    std::vector<uint64_t> f32_bytes_at;                 // parallel to w
    uint64_t data_start = 0;
    // totals
    uint64_t bytes_src_total = 0, bytes_codes_total = 0, bytes_scale_total = 0, bytes_f32_total = 0;
    uint64_t n_weights = 0, n_ptq_tensors = 0, n_f32_tensors = 0;
    // ---- hyper-parameters, every one read from the file ----
    int n_layer = 32, hidden = 4096, ffn = 12288, vocab = 248320;
    int n_head = 16, n_head_kv = 4, head_dim = 256, n_rot = 64;
    int full_attn_interval = 4;
    int conv_kernel = 4, state_size = 128, group_count = 16, ts_rank = 32, inner = 4096;
    // The ROW COUNT of blk.0.attn_qkv.weight - the channel count the short causal conv and its shift
    // register actually address. Read from the tensor table by model_plan_build() (0 when the file has
    // no recurrent block at all). It is a separate field because the forward sizes CH from the TENSOR
    // while the mirror formula here sizes it from inner, and those two agree only while the export
    // puts q and k side by side at `inner` each: 8192 == 2*4096 on the 9B. model_plan_validate() is
    // where that equality stops being an assumption.
    int gdn_channels = 0;
    float rms_eps = 1e-6f, rope_base = 1e7f;
    bool gdn_v_grouped = true;
    // ---- what the resolver actually did, kept so the report can show it instead of assert it ----
    std::string arch;                    // general.architecture as read from the file
    std::string meta_prefixes;           // the prefix list, ";"-joined, in lookup order
    std::string layer_rule;              // the ONE sentence that says where the layer types came from
    std::string hat_rule;                // the same for the rotation contract
    int ssm_count_declared = -1;         // a CROSS-CHECK only; never a source of positions
    int attn_count_declared = -1;        // idem
    // ---- layer typing. EMPTY means "the interval rule", which is the 9B's spelling; a populated
    // vector means an explicit list (or the tensor-table derivation) answered and it has been
    // cross-checked against the interval/table before getting here.
    std::vector<uint8_t> layer_kind;
    // ---- prism.hadamard.* version gate (0 = no rotation at all, 1 = the 9B's block transform)
    long long hat_version = 0;
    uint32_t hat_block = 0;              // prism.hadamard.block_size, NEVER a literal
    std::string hat_transform;           // prism.hadamard.transform
    std::string hat_sign_mode;           // prism.hadamard.sign_mode
    std::string hat_axis;                // prism.hadamard.axis
    std::vector<uint32_t> sign_widths;   // prism.hadamard.sign_widths
    uint64_t n_sign_values = 0;          // length of prism.hadamard.sign_values
    std::string sign_values_key;         // the KEY that answered, prefix and all - the forward reads
                                         // the array back out of the header through this instead of
                                         // retyping "prism.hadamard.sign_values"
    bool hat_active = false;             // version >= 1: rotate before the ternary matmul
    bool hat_use_fwht = false;           // false == hat4096.comp (the 9B's bit-frozen path)
    bool hat_explicit_signs = false;     // sign_mode == "explicit": the +-1 diagonal is a vector

    const WeightPlan *find(const char *name) const {
        for (const auto &x : w) if (x.name == name) return &x;
        return nullptr;
    }
    // The interval rule, exactly as it was before this file grew a resolver: block b is a softmax
    // layer when (b+1) % interval == 0, i.e. for the 9B's interval 4 the attention blocks are
    // 3, 7, ... 31. An interval <= 0 would be a modulo-by-zero, and a file that omits the key
    // entirely is a plain transformer (every layer attention), so that is what it now means - the
    // resolver only reaches this expression when the file carries no list.
    bool interval_rule(int b) const {
        return full_attn_interval > 0 ? ((b + 1) % full_attn_interval == 0) : true;
    }
    bool is_attn_layer(int b) const {
        if (!layer_kind.empty())
            return b >= 0 && b < (int)layer_kind.size() && layer_kind[b] != BONSAI_LK_SSM;
        return interval_rule(b);
    }
    int n_attn_layers() const {
        int n = 0;
        for (int b = 0; b < n_layer; ++b) if (is_attn_layer(b)) ++n;
        return n;
    }
    int n_ssm_layers() const { return n_layer - n_attn_layers(); }
    const char *layer_kind_name(int b) const { return is_attn_layer(b) ? "attn" : "ssm "; }

    // ------------------------------------------------------------------ the KV charge
    // One (layer, position) slot, in uint (4 B) words, DERIVED from the layout the KV kernels write:
    //   K   fp16, always                          nkv*hd/2 words
    //   V   fp16                                  nkv*hd/2 words          (BONSAI_KV_V_QUANT=none)
    //   V   q8_0 data int8   + fp16 block scales   nkv*hd/4 + nkv*(hd/32)/2 (the default)
    // This is a SECOND derivation of the number KvGeom::compute() (kv_guard.h) produces for the
    // arena. It is duplicated on purpose - two independent formulas that disagree is a bug report,
    // while one formula used twice can only ever agree with itself - and fwd_init() asserts the two
    // agree before it allocates anything, so the duplication is checked, not trusted. kv_guard.h
    // pulls in windows.h/PDH and this header must stay includable by a CPU-only test, which is the
    // mechanical reason the mirror lives here and not in a shared header.
    static inline uint64_t kv_words_slot_layer(int nkv, int hd, int v_q8) {
        const uint64_t w_k  = uint64_t(nkv) * uint64_t(hd) / 2ull;
        const uint64_t w_vd = v_q8 ? uint64_t(nkv) * uint64_t(hd) / 4ull
                                   : uint64_t(nkv) * uint64_t(hd) / 2ull;
        const uint64_t w_vs = v_q8 ? uint64_t(nkv) * (uint64_t(hd) / 32ull) / 2ull : 0ull;
        return w_k + w_vd + w_vs;
    }
    // THE INVARIANT: an SSM/GDN layer charges zero cache bytes. Not "we hope the arena is sized
    // right" - the function returns 0 by construction for every layer the resolver did not type as
    // attention, and model_plan_validate() plus fwd_init() compare the sum against the arena.
    uint64_t kv_bytes_per_layer(int b, int v_q8) const {
        if (!is_attn_layer(b)) return 0ull;
        return kv_words_slot_layer(n_head_kv, head_dim, v_q8) * 4ull;
    }
    uint64_t kv_bytes_per_token(int v_q8) const {
        uint64_t t = 0ull;
        for (int b = 0; b < n_layer; ++b) t += kv_bytes_per_layer(b, v_q8);
        return t;
    }
    // The context-INDEPENDENT part one recurrent layer owns: the [dim][dim][nh] state plus the
    // (conv_kernel-1) x channels shift register, both f32 (measured layout, see gdn_delta.comp).
    // The channel term is spelled `2 * inner` and NOT `attn_qkv.M` on purpose: the forward sizes its
    // arena from the TENSOR (blk.0.attn_qkv.weight's row count), so reading the same tensor here
    // would make the two agree by construction. As two independent derivations they only agree while
    // the export really does store q and k adjacent at `inner` rows each - which model_plan_validate()
    // checks against gdn_channels and refuses the file when it stops being true.
    uint64_t ssm_bytes_per_layer() const {
        const uint64_t st = uint64_t(state_size) * state_size *
                            uint64_t(inner / (state_size ? state_size : 1)) * 4ull;
        const uint64_t ch = uint64_t(conv_kernel - 1) * uint64_t(inner) * 2ull * 4ull;
        return st + ch;
    }

    // The sign diagonal, sized from the file rather than from the 9B's literal 16384: the array is
    // the concatenation of one +-1 vector per activation width (prism.hadamard.sign_widths), and
    // signs_mul.comp addresses the second vector at the first vector's WIDTH in ivec4 units. Both
    // numbers used to be typed into the forward.
    uint64_t signs_total() const {
        if (!hat_active || !hat_explicit_signs) return 0ull;
        if (n_sign_values) return n_sign_values;
        uint64_t t = 0;
        for (auto x : sign_widths) t += x;
        return t;
    }
    uint32_t sign_width0() const {
        if (!sign_widths.empty()) return sign_widths[0];
        return uint32_t(hidden);            // one vector per width: the second starts at hidden
    }
};

static inline uint64_t align_up_b(uint64_t a, uint64_t b) { return (a + b - 1) / b * b; }

// ------------------------------------------------------------------ metadata spelling resolution
// WHY NOT A LITERAL KEY LIST: the 9B spells everything "qwen35.*", llama.cpp's generic readers
// spell the same fields "<arch>.attention.head_count" with arch from general.architecture, and a
// PrismML export has been observed to carry "prism.hadamard.*" next to the arch block. Hardcoding
// one spelling means the 27B either fails to load or - worse - loads with a DEFAULT and prints a
// plan nobody checked. So the rule is:
//   prefixes = [general.architecture] + ["prism", "general", ""] (deduplicated, in that order)
// and a logical field is looked up under every prefix for every accepted suffix. Values found in
// more than one place must be EQUAL: a conflict is a hard error naming both keys and both values,
// because "the first one wins" is exactly how a wrong head count reaches a GEMV.
struct BonsaiMeta {
    std::vector<std::string> pre;   // lookup prefixes; "" == the bare key
    std::string arch;               // general.architecture ("" if the file has none)
    std::string note;               // how the prefix list was chosen - printed once
};

static inline std::string bonsai_md_key(const std::string &pre, const char *suf) {
    return pre.empty() ? std::string(suf) : (pre + "." + suf);
}

static inline bool bonsai_md_is_pow2(uint64_t v) { return v != 0ull && (v & (v - 1ull)) == 0ull; }

// Build the prefix list. The architecture string is the normal source; a file without
// general.architecture still resolves if EXACTLY one prefix carries a block_count, which is the
// 9B-compatible fallback. Two different prefixes both carrying block_count and no architecture
// string is ambiguous, and guessing here would silently mix two models' hyper-parameters.
static inline bool bonsai_md_init(const GgufHeader &h, BonsaiMeta &m, std::string &why) {
    auto a = h.kv_s.find("general.architecture");
    if (a != h.kv_s.end()) m.arch = a->second;
    std::vector<std::string> with_bc;
    for (const auto &kv : h.kv_i) {
        const std::string &k = kv.first;
        const std::string suf = ".block_count";
        if (k.size() <= suf.size()) continue;
        if (k.compare(k.size() - suf.size(), suf.size(), suf) != 0) continue;
        with_bc.push_back(k.substr(0, k.size() - suf.size()));
    }
    auto push = [&](const std::string &p) {
        for (const auto &x : m.pre) if (x == p) return;
        m.pre.push_back(p);
    };
    if (!m.arch.empty()) {
        push(m.arch);
        m.note = "general.architecture = " + m.arch;
    } else if (with_bc.size() == 1ull) {
        push(with_bc[0]);
        m.arch = with_bc[0];
        m.note = "general.architecture absent; prefix taken from the single *.block_count key (" +
                 with_bc[0] + ".block_count)";
    } else if (with_bc.empty()) {
        m.note = "no general.architecture and no *.block_count key";
    } else {
        std::string l;
        for (const auto &x : with_bc) { if (!l.empty()) l += ", "; l += x; }
        why = "general.architecture is absent and " + std::to_string(with_bc.size()) +
              " prefixes each declare a block_count (" + l + ") - refusing to guess which model"
              " the hyper-parameters belong to";
        return false;
    }
    push("prism"); push("general"); push("");
    std::string pj;
    for (const auto &p : m.pre) {
        if (!pj.empty()) pj += ",";
        pj += p.empty() ? "<bare>" : p;
    }
    m.note += " -> prefixes [" + pj + "]";
    return true;
}

// One logical integer field. `sufs` is the accepted spelling list for it; the first slot that has
// ANY hit decides, and all hits must agree. Returns false when nothing matched (the caller decides
// whether that is optional or fatal) or when two spellings disagree.
static inline bool bonsai_md_i(const GgufHeader &h, const BonsaiMeta &m,
                               const char *const *sufs, int nsuf, long long &out,
                               std::string &used, std::string &why, bool allow_float = false) {
    bool have = false;
    std::string first_key;
    long long first = 0;
    for (int s = 0; s < nsuf; ++s) {
        for (const auto &p : m.pre) {
            const std::string k = bonsai_md_key(p, sufs[s]);
            long long v = 0;
            auto it = h.kv_i.find(k);
            if (it != h.kv_i.end()) v = it->second;
            else if (allow_float) {
                // A float-valued spelling of an integer field (an export that wrote 4096.0 for a
                // width) is only usable if it is EXACTLY integral; 4096.5 is a corrupt width, and
                // rounding it into a GEMV stride is the silent-wrong-answer this file exists to stop.
                auto ft = h.kv_f.find(k);
                if (ft == h.kv_f.end()) continue;
                const long long rounded = static_cast<long long>(ft->second);
                if (ft->second != static_cast<double>(rounded)) continue;
                v = rounded;
            } else continue;
            if (!have) { have = true; first_key = k; first = v; continue; }
            // Only compare within the SAME logical field, and only across suffixes of the same
            // meaning: the suffix loop above already stopped at the first suffix that matched, so
            // reaching here means two prefixes spelled the same suffix differently.
            if (v != first) {
                why = "conflicting spellings for the same field: " + first_key + " = " +
                      std::to_string(first) + " but " + k + " = " + std::to_string(v) +
                      " - this runtime picks neither";
                return false;
            }
        }
        if (have) break;                       // the first suffix that answers wins
    }
    if (!have) return false;
    out = first; used = first_key;
    return true;
}

static inline bool bonsai_md_f(const GgufHeader &h, const BonsaiMeta &m,
                               const char *const *sufs, int nsuf, double &out,
                               std::string &used, std::string &why) {
    bool have = false;
    std::string first_key;
    double first = 0;
    for (int s = 0; s < nsuf; ++s) {
        for (const auto &p : m.pre) {
            const std::string k = bonsai_md_key(p, sufs[s]);
            auto it = h.kv_f.find(k);
            if (it == h.kv_f.end()) continue;
            if (!have) { have = true; first_key = k; first = it->second; continue; }
            if (it->second != first) {
                why = "conflicting spellings: " + first_key + " = " + std::to_string(first) +
                      " but " + k + " = " + std::to_string(it->second);
                return false;
            }
        }
        if (have) break;
    }
    if (!have) return false;
    out = first; used = first_key;
    return true;
}

// The string/array reader the layer-type list and the transform-name checks use. A GGUF string
// lands in kv_s; a string ARRAY and a numeric ARRAY both land in kv_a (bonsai_gguf.h stringifies
// numeric arrays, which is why a "0,1,0,..." layer mask can arrive as an array of numbers and still
// parse here without a second pass over the file).
//
// Same agreement rule as bonsai_md_i(): the FIRST suffix that answers decides, but every PREFIX hit
// for that suffix must carry the identical sequence. Two prefixes shipping two different 64-token
// layer masks is a corrupt export, and "first prefix wins" would pick one at random (map order).
static inline bool bonsai_md_arr(const GgufHeader &h, const BonsaiMeta &m,
                                 const char *const *sufs, int nsuf,
                                 std::vector<std::string> &out, std::string &used,
                                 std::string *why = nullptr) {
    for (int s = 0; s < nsuf; ++s) {
        bool have = false;
        std::vector<std::string> first;
        std::string first_key;
        for (const auto &p : m.pre) {
            const std::string k = bonsai_md_key(p, sufs[s]);
            std::vector<std::string> v;
            auto a = h.kv_a.find(k);
            if (a != h.kv_a.end()) v = a->second;
            else {
                auto st = h.kv_s.find(k);
                if (st == h.kv_s.end()) continue;
                v.push_back(st->second);
            }
            if (!have) { have = true; first = v; first_key = k; continue; }
            if (v != first && why) {
                *why = "conflicting spellings for the same array field: " + first_key + " has " +
                       std::to_string(first.size()) + " entries, " + k + " has " +
                       std::to_string(v.size()) + " (or different values) - this runtime picks neither";
                return false;
            }
        }
        if (have) { out = first; used = first_key; return true; }
    }
    return false;
}

static inline std::string bonsai_tolower(std::string s) {
    for (auto &c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

// Split a comma/semicolon/whitespace-separated list. Written by hand because this project has no
// <sstream>-heavy style in the hot headers and because the accepted separators are part of the
// published contract below (",", ";", space, tab, CR, LF).
static inline std::vector<std::string> bonsai_split(const std::string &s) {
    std::vector<std::string> v;
    std::string cur;
    for (char c : s) {
        if (c == ',' || c == ';' || c == ' ' || c == '\t' || c == '\r' || c == '\n') {
            if (!cur.empty()) { v.push_back(cur); cur.clear(); }
        } else cur.push_back(c);
    }
    if (!cur.empty()) v.push_back(cur);
    return v;
}

// One token of a layer-type list -> its kind. Returns -1 unknown, -2 KNOWN-BUT-UNSUPPORTED.
// The second value exists because the dangerous bug is a silent mapping: "sliding_attention" is an
// attention layer with a window, and treating it as full attention costs the same KV bytes but reads
// keys the model was never allowed to see, so the output would be plausible and wrong.
static inline int bonsai_layer_token(const std::string &raw, std::string &canon) {
    const std::string t = bonsai_tolower(raw);
    canon = t;
    if (t == "full_attention" || t == "attention" || t == "attn" || t == "full" ||
        t == "softmax" || t == "self" || t == "1") return int(BONSAI_LK_ATTN);
    if (t == "ssm" || t == "gdn" || t == "gated_delta" || t == "gated_delta_net" ||
        t == "linear_attention" || t == "linear" || t == "recurrent" || t == "mamba" ||
        t == "none" || t == "0") return int(BONSAI_LK_SSM);
    if (t == "sliding_attention" || t == "sliding_window" || t == "window" || t == "windowed" ||
        t == "chunked" || t == "moe" || t == "hybrid") return -2;
    return -1;
}

// ------------------------------------------------------------------ layer typing
// ACCEPTED SPELLINGS - this is the contract, and it is what a report cites, so it is written as
// data rather than as prose:
//
//  S1 EXPLICIT PER-LAYER LIST. Key = "<prefix>.attention.layer_types" | "<prefix>.layer_types" |
//     "<prefix>.layers.types" for prefix in the BonsaiMeta list, so "prism.layer_types" and
//     "qwen35.attention.layer_types" both work. Payload may be
//       - a GGUF string ARRAY of exactly block_count tokens, or
//       - a GGUF numeric ARRAY of exactly block_count 0/1 entries (they arrive stringified), or
//       - a single GGUF STRING containing block_count tokens separated by ", ", "; ", or space.
//     Tokens, case-insensitive:
//       ATTENTION: full_attention, attention, attn, full, softmax, self, "1"
//       RECURRENT: ssm, gdn, gated_delta, gated_delta_net, linear_attention, linear, recurrent,
//                  mamba, none, "0"
//     REJECTED with a loud error, never remapped: sliding_attention, sliding_window, window,
//     windowed, chunked, moe, hybrid (the QK kernels have no window term; see the token function).
//     A list whose length is not block_count, or whose entries are two different lengths of the
//     same key, is an error.
//
//  S2 INTERVAL. Key = "<prefix>.full_attention_interval" (integer >= 1) -> layer b is attention
//     iff (b + 1) % interval == 0. This is the spelling the 9B uses and its answer is unchanged.
//     interval == 0 or negative is an ERROR here even though is_attn_layer() has a safe fallback,
//     because a file that means "every layer is attention" should say so with S1/S4, not with zero.
//
//  S3 COUNTS. Keys = "<prefix>.ssm_layer_count" | ".linear_attention_layer_count" |
//     ".recurrent_layer_count" | ".gdn_layer_count", and "<prefix>.attn_layer_count" |
//     ".full_attention_layer_count". These are CROSS-CHECKS ONLY. A count cannot say WHERE the
//     layers are, so S3 with neither S1 nor S2 is an error, not a guess.
//
//  S4 TENSOR TABLE (always computed, never the sole authority unless it is the only evidence).
//     Block b is attention iff blk.b.attn_q.weight or blk.b.attn_k.weight or blk.b.attn_v.weight
//     exists; it is recurrent iff blk.b.attn_qkv.weight exists. Both is an error. Neither is an
//     error UNLESS a resolved S1/S2 list says the layer is one of the two kinds AND the layer's
//     other tensors agree with that (see the "table contradicts" branch).
//     NOTE (measured, gguf_meta.txt:56 vs :213): the 9B ships in two exports whose qwen35.block_count
//     is 33 (nextn_predict_layers = 1, i.e. the last block is a draft/MTP layer this engine has no
//     path for) and 32. This gate takes block_count at face value and lets S4 catch the rest: block
//     32 of the 33-block export has neither attn_q/k/v nor attn_qkv, so it lands in tbl_unknown and
//     the load is REFUSED. That refusal is the intended behaviour of a strict superset - we do not
//     silently drop a block we cannot type. Running the 33-block file needs an MTP path, not a
//     smaller n_layer.
//
// AGREEMENT RULE: every spelling that answers must give the SAME 64-bit mask. Two that answer
// differently is a hard error - the failure mode being paid for is a KV arena sized for 16 layers
// while the forward touches 17.
static inline bool model_plan_layer_types(const GgufHeader &h, const BonsaiMeta &m, ModelPlan &p,
                                          std::string &why) {
    if (p.n_layer <= 0) { why = "block_count is not positive"; return false; }

    // ---- S1
    static const char *kListSufs[] = {"attention.layer_types", "layer_types", "layers.types"};
    std::vector<std::string> used_arr;
    std::vector<uint8_t> kind_list;
    std::string list_key;
    why.clear();
    if (bonsai_md_arr(h, m, kListSufs, 3, used_arr, list_key, &why)) {
        std::vector<std::string> toks;
        if (used_arr.size() == 1ull) toks = bonsai_split(used_arr[0]);   // the single-string form
        else toks = used_arr;
        if ((int)toks.size() != p.n_layer) {
            why = list_key + " has " + std::to_string(toks.size()) + " entries but block_count is " +
                  std::to_string(p.n_layer) + " - a per-layer list that does not cover every layer"
                  " would leave the tail undefined";
            return false;
        }
        kind_list.resize(toks.size());
        for (size_t i = 0; i < toks.size(); ++i) {
            std::string canon;
            int k = bonsai_layer_token(toks[i], canon);
            if (k == -2) {
                why = list_key + "[" + std::to_string(i) + "] = \"" + toks[i] + "\": this runtime has"
                      " no windowed/MoE layer path. Mapping it onto full attention would charge the"
                      " right number of bytes and read keys outside the window, i.e. plausible"
                      " garbage; mapping it onto the recurrent state would be worse. Refusing.";
                return false;
            }
            if (k == -1) {
                why = list_key + "[" + std::to_string(i) + "] = \"" + toks[i] + "\" is not a token"
                      " this runtime recognises (accepted: full_attention/attention/attn/full/"
                      "softmax/self/1 and ssm/gdn/gated_delta/linear_attention/linear/recurrent/"
                      "mamba/none/0)";
                return false;
            }
            kind_list[i] = uint8_t(k);
        }
    } else if (!why.empty()) {
        return false;                                 // two prefixes, two lists: see bonsai_md_arr
    }

    // ---- S2
    long long iv = 0; std::string iv_key;
    static const char *kIvSufs[] = {"full_attention_interval"};
    why.clear();
    bool have_iv = bonsai_md_i(h, m, kIvSufs, 1, iv, iv_key, why);
    if (why.size()) return false;                    // a conflict on the interval is fatal
    if (have_iv && iv <= 0) {
        why = iv_key + " = " + std::to_string(iv) + ": an interval of 0 or less is not a layer"
              " pattern, and the engine will not read it as \"all layers attention\" (say that with"
              " a per-layer list instead)";
        return false;
    }
    std::vector<uint8_t> kind_iv;
    if (have_iv) {
        p.full_attn_interval = int(iv);
        kind_iv.resize(p.n_layer);
        for (int b = 0; b < p.n_layer; ++b) kind_iv[b] = uint8_t(p.interval_rule(b) ? BONSAI_LK_ATTN : BONSAI_LK_SSM);
    }

    // ---- S3 (cross-check values, remembered for the report)
    long long t = 0; std::string kk;
    static const char *kSsmSufs[] = {"ssm_layer_count", "linear_attention_layer_count",
                                     "recurrent_layer_count", "gdn_layer_count"};
    why.clear();
    if (bonsai_md_i(h, m, kSsmSufs, 4, t, kk, why)) p.ssm_count_declared = int(t);
    static const char *kAttnSufs[] = {"attn_layer_count", "full_attention_layer_count"};
    why.clear();
    if (bonsai_md_i(h, m, kAttnSufs, 2, t, kk, why)) p.attn_count_declared = int(t);

    // ---- S4, the tensor table. Independent evidence, so build it whenever there IS a table: if
    // model_plan_build() has not run yet, p.w is empty and every block would look like "neither",
    // which is not evidence of anything. In that case the table simply does not answer (0xFF = no
    // opinion) and layer_rule says so, rather than manufacturing a contradiction.
    const bool have_table = !p.w.empty();
    std::vector<uint8_t> kind_tbl(p.n_layer, 0xFF);
    std::string tbl_unknown = "";
    for (int b = 0; have_table && b < p.n_layer; ++b) {
        const std::string base = "blk." + std::to_string(b) + ".";
        const bool q  = p.find((base + "attn_q.weight").c_str()) != nullptr;
        const bool kx = p.find((base + "attn_k.weight").c_str()) != nullptr;
        const bool vx = p.find((base + "attn_v.weight").c_str()) != nullptr;
        const bool qkv = p.find((base + "attn_qkv.weight").c_str()) != nullptr;
        const bool att = q || kx || vx;
        if (att && qkv) {
            why = base + " has BOTH attn_q/k/v (softmax) and attn_qkv (recurrent) - the table"
                          " contradicts itself about this layer";
            return false;
        }
        if (att) kind_tbl[b] = BONSAI_LK_ATTN;
        else if (qkv) kind_tbl[b] = BONSAI_LK_SSM;
        else tbl_unknown += (tbl_unknown.empty() ? "" : ",") + std::to_string(b);
    }

    // ---- agreement
    auto cmp = [&](const std::vector<uint8_t> &x, const std::vector<uint8_t> &y,
                   const char *nx, const char *ny) {
        for (size_t b = 0; b < x.size(); ++b) {
            if (y[b] == 0xFF) continue;                       // the table had no answer for b
            if (x[b] != y[b]) {
                why = std::string(nx) + " says block " + std::to_string(b) + " is " +
                      (x[b] ? "attention" : "recurrent") + " but " + ny + " says the opposite."
                      " A KV arena built from one and a forward built from the other overwrites"
                      " live state, so neither wins.";
                return false;
            }
        }
        return true;
    };
    std::vector<uint8_t> resolved;
    if (!kind_list.empty() && !kind_iv.empty()) {
        if (!cmp(kind_list, kind_iv, list_key.c_str(), iv_key.c_str())) return false;
        resolved = kind_list;
        p.layer_rule = "explicit list " + list_key + " (== " + iv_key + " agreement)";
    } else if (!kind_list.empty()) {
        resolved = kind_list;
        p.layer_rule = "explicit list " + list_key;
    } else if (!kind_iv.empty()) {
        resolved = kind_iv;
        p.layer_rule = iv_key + " = " + std::to_string(iv) + " -> (b+1) % interval == 0"
                       "  [the 9B's spelling]";
    } else if (!have_table) {
        why = "no layer-type list and no full_attention_interval in the metadata, AND the tensor table"
              " was not available to this call (model_plan_build() had not filled it) - typing every"
              " layer as attention by default is exactly the silent 2x-KV-overcharge this gate exists"
              " to prevent, so this is a refusal, not a fallback";
        return false;
    } else if (!tbl_unknown.empty()) {
        why = "no layer-type list, no full_attention_interval, and blocks [" + tbl_unknown +
              "] have neither attn_q/k/v nor attn_qkv in the tensor table - the engine cannot type"
              " them, and guessing the position of the attention layers is precisely the class of"
              " bug this file was written to stop";
        return false;
    } else {
        resolved = kind_tbl;
        p.layer_rule = "DERIVED FROM THE TENSOR TABLE (no layer-type key and no interval in the file)"
                       "  [UNVERIFIED against metadata - check this line before trusting a run]";
    }
    // The table is a cross-check whenever it answered at all.
    if (!cmp(resolved, kind_tbl, "layer typing", "the tensor table")) return false;

    // S3 last: it can only confirm.
    int nat = 0;
    for (auto v : resolved) if (v == BONSAI_LK_ATTN) ++nat;
    if (p.ssm_count_declared >= 0 && p.ssm_count_declared != p.n_layer - nat) {
        why = "the declared recurrent layer count is " + std::to_string(p.ssm_count_declared) +
              " but the resolved typing puts " + std::to_string(p.n_layer - nat) + " recurrent"
              " layers among " + std::to_string(p.n_layer) + " - a count is not a position, so this"
              " is not reconciled by rounding";
        return false;
    }
    if (p.attn_count_declared >= 0 && p.attn_count_declared != nat) {
        why = "the declared attention layer count is " + std::to_string(p.attn_count_declared) +
              " but the resolved typing gives " + std::to_string(nat);
        return false;
    }
    p.layer_kind = resolved;
    // An all-attention model is the plain-transformer case and it is legal; keep the interval field
    // meaningful for the report when the list and the interval both said the same thing.
    return true;
}

// ------------------------------------------------------------------ hyper-parameters
// Pulled out of the metadata. Anything missing that the FORWARD needs is a hard error rather than a
// default: a silently wrong head count is exactly the class of bug this project has already been
// bitten by once (the n_rot probe, FAITHFUL_REPRO.md §3).
static bool model_plan_hparams(const GgufHeader &h, ModelPlan &p) {
    BonsaiMeta m;
    std::string why;
    if (!bonsai_md_init(h, m, why)) { fprintf(stderr, "model: %s\n", why.c_str()); return false; }
    p.arch = m.arch;
    p.meta_prefixes = m.note;

    // The two readers every field below goes through. `fatal` says what a missing key means: the
    // 9B's own fields are all present, so for them missing == a corrupt/unknown file and the engine
    // stops; the ssm.* group is only fatal once the tensor table shows a recurrent layer, because a
    // pure-transformer checkpoint legitimately has no ssm.conv_kernel to read. Both clear `dst`
    // first: the caller reuses one scratch variable, and "absent, non-fatal" must leave 0 (which the
    // validator rejects for a recurrent model) rather than the previous field's value.
    auto req_i = [&](const char *const *sufs, int ns, long long &dst, const char *label, bool fatal) {
        std::string key; why.clear(); dst = 0;
        if (!bonsai_md_i(h, m, sufs, ns, dst, key, why)) {
            if (!why.empty()) { fprintf(stderr, "model: %s\n", why.c_str()); return false; }
            if (fatal) {
                fprintf(stderr, "model: metadata field %s missing - accepted spelling \"%s.<suffix>\""
                                " (prefixes tried: %s)\n", label, sufs[0], m.note.c_str());
                return false;
            }
            return true;
        }
        if (!why.empty()) { fprintf(stderr, "model: %s\n", why.c_str()); return false; }
        return true;
    };
    auto req_f = [&](const char *const *sufs, int ns, double &dst, const char *label, bool fatal) {
        std::string key; why.clear();
        if (!bonsai_md_f(h, m, sufs, ns, dst, key, why)) {
            if (!why.empty()) { fprintf(stderr, "model: %s\n", why.c_str()); return false; }
            if (fatal) { fprintf(stderr, "model: metadata key %s missing\n", label); return false; }
            return true;
        }
        if (!why.empty()) { fprintf(stderr, "model: %s\n", why.c_str()); return false; }
        return true;
    };

    static const char *s_nlayer[]  = {"block_count"};
    static const char *s_hidden[]  = {"embedding_length", "n_embd"};
    static const char *s_ffn[]     = {"feed_forward_length", "ffn_dim"};
    static const char *s_nhead[]   = {"attention.head_count"};
    static const char *s_nkv[]     = {"attention.head_count_kv"};
    static const char *s_hdim[]    = {"attention.key_data_shape", "attention.key_length",
                                      "attention.head_dim"};
    static const char *s_nrot[]    = {"rope.dimension_count"};
    static const char *s_conv[]    = {"ssm.conv_kernel", "ssm.conv_kernel_size"};
    static const char *s_state[]   = {"ssm.state_size"};
    static const char *s_group[]   = {"ssm.group_count"};
    static const char *s_tsrank[]  = {"ssm.time_step_rank"};
    static const char *s_inner[]   = {"ssm.inner_size"};

    long long tmp = 0;
    // One call per field: req_i() already zeroes its scratch, so an absent non-fatal field lands as
    // 0 in the target (which model_plan_validate() rejects for a recurrent model) instead of keeping
    // the previous field's value.
    auto get_i = [&](const char *const *sufs, int ns, const char *label, bool fatal, int &dst) {
        if (!req_i(sufs, ns, tmp, label, fatal)) return false;
        dst = int(tmp);
        return true;
    };
    if (!get_i(s_nlayer, 1, "block_count", true, p.n_layer)) return false;
    if (!get_i(s_hidden, 2, "embedding_length", true, p.hidden)) return false;
    if (!get_i(s_ffn, 2, "feed_forward_length", true, p.ffn)) return false;
    if (!get_i(s_nhead, 1, "attention.head_count", true, p.n_head)) return false;
    if (!get_i(s_nkv, 1, "attention.head_count_kv", true, p.n_head_kv)) return false;
    if (!get_i(s_hdim, 3, "attention.head_dim", true, p.head_dim)) return false;
    // value_length is a SEPARATE key this engine cannot honour: every attention kernel is written
    // for one head_dim used by BOTH K and V. Read it as a cross-check, not as a suggestion.
    {
        std::string key; long long vl = p.head_dim; why.clear();
        static const char *s_vlen[] = {"attention.value_length", "attention.value_data_shape"};
        const bool ok = bonsai_md_i(h, m, s_vlen, 2, vl, key, why);
        if (!ok && !why.empty()) { fprintf(stderr, "model: %s\n", why.c_str()); return false; }
        if (ok && vl != p.head_dim) {
            fprintf(stderr, "model: %s = %lld but head_dim = %d; the attention kernels use ONE width"
                            " for K and V, so an asymmetric head is not supported here.\n",
                    key.c_str(), vl, p.head_dim);
            return false;
        }
    }
    if (!get_i(s_nrot, 1, "rope.dimension_count", true, p.n_rot)) return false;

    // The recurrent group is fatal only if the tensor table shows an attn_qkv somewhere. Doing that
    // test on the TABLE rather than on the resolved typing keeps this function order-free: the
    // typing resolver runs later (it needs these widths to size anything), and a file whose layer
    // types are unresolved still cannot quietly skip the ssm.* checks.
    bool any_ssm = false;
    for (const auto &t : h.t)
        if (t.name.size() > 4 && t.name.compare(0, 4, "blk.") == 0 &&
            t.name.find(".attn_qkv.weight") != std::string::npos) { any_ssm = true; break; }
    if (!get_i(s_conv, 2, "ssm.conv_kernel", any_ssm, p.conv_kernel)) return false;
    if (!get_i(s_state, 1, "ssm.state_size", any_ssm, p.state_size)) return false;
    if (!get_i(s_group, 1, "ssm.group_count", any_ssm, p.group_count)) return false;
    if (!get_i(s_tsrank, 1, "ssm.time_step_rank", any_ssm, p.ts_rank)) return false;
    if (!get_i(s_inner, 1, "ssm.inner_size", any_ssm, p.inner)) return false;

    double dv = 0;
    static const char *s_eps[] = {"attention.layer_norm_rms_epsilon"};
    if (!req_f(s_eps, 1, dv, "attention.layer_norm_rms_epsilon", true)) return false;
    p.rms_eps = float(dv);
    static const char *s_ropeb[] = {"rope.freq_base"};
    dv = 0;
    if (!req_f(s_ropeb, 1, dv, "rope.freq_base", true)) return false;
    p.rope_base = float(dv);

    // An optional flag: absent == the 9B's own default (grouped V de-interleave, which is what
    // permute_vg.comp implements and what the 9B logs were produced with). Present-but-conflicting
    // is an error, not a shrug. The 9B spells it "prism.hadamard.gdn_v_grouped".
    static const char *s_vgrp[] = {"hadamard.gdn_v_grouped", "gdn_v_grouped"};
    std::string used;
    const bool vgrp_found = bonsai_md_i(h, m, s_vgrp, 2, tmp, used, why);
    if (!vgrp_found && !why.empty()) { fprintf(stderr, "model: %s\n", why.c_str()); return false; }
    if (vgrp_found) p.gdn_v_grouped = (tmp != 0);

    auto tv = h.kv_a.find("tokenizer.ggml.tokens");
    if (tv != h.kv_a.end()) p.vocab = int(tv->second.size());
    return true;
}

// ------------------------------------------------------------------ the resolved contract, as text
// Printed once at load, because every claim in this file's docstring is something a report has to
// be able to point at in a log rather than re-derive from source.
//
// CALL ORDER (deliberate): model_plan_build() does NOT call this. The layout walk it does is the
// thing plan_probe.cpp and every existing budget run already depend on, and a checkpoint that
// carries tensors but no layer-type key would then stop producing a layout those tools can still
// print. So the contract is a SEPARATE gate, called by fwd_init() before it allocates anything (a
// refusal there means no VRAM is touched) and by the CPU tests directly. Consequence: this reads the
// tensor table through ModelPlan::find(), so it must run AFTER model_plan_build() has filled p.w -
// the S4 branch below says out loud when it had no table to consult instead of inventing agreement.
static bool model_plan_contract(const GgufHeader &h, ModelPlan &p, std::string &why) {
    BonsaiMeta m;
    if (!bonsai_md_init(h, m, why)) return false;
    if (!model_plan_layer_types(h, m, p, why)) return false;

    // ---- the rotation gate (inline: the same key machinery, but its decision needs the widths
    // resolved above, so it lives here rather than in a separate public entry point).
    static const char *kVer[] = {"hadamard.version"};
    static const char *kBk[]  = {"hadamard.block_size"};
    static const char *kTx[]  = {"hadamard.transform"};
    static const char *kSm[]  = {"hadamard.sign_mode"};
    static const char *kAx[]  = {"hadamard.axis"};
    static const char *kSw[]  = {"hadamard.sign_widths"};
    static const char *kSv[]  = {"hadamard.sign_values"};
    std::string used;
    long long t = 0;
    why.clear();
    if (bonsai_md_i(h, m, kVer, 1, t, used, why)) {
        if (!why.empty()) return false;
        p.hat_version = t;
    } else p.hat_version = 0;                      // absent == 0, which == NO rotation
    if (p.hat_version < 0) { why = "hadamard.version is negative"; return false; }
    if (p.hat_version > 1) {
        why = "hadamard.version = " + std::to_string(p.hat_version) + ": this runtime implements"
              " version 1 - the normalized Sylvester-Walsh block transform - and NOTHING else."
              " A new rotation contract is a new kernel plus a new alignment run, not a fallback";
        return false;
    }
    p.hat_active = p.hat_version >= 1;

    auto str_field = [&](const char *const *sufs, int ns, std::string &dst) -> bool {
        std::vector<std::string> a; std::string k;
        why.clear();
        if (!bonsai_md_arr(h, m, sufs, ns, a, k, &why)) return why.empty();
        if (!a.empty()) dst = a[0];
        return true;
    };
    if (!str_field(kTx, 1, p.hat_transform)) return false;
    if (!str_field(kSm, 1, p.hat_sign_mode)) return false;
    if (!str_field(kAx, 1, p.hat_axis)) return false;

    if (!p.hat_active) {
        p.hat_rule = "version " + std::to_string(p.hat_version) + " -> NO ROTATION: the normed"
                     " vector goes into every ternary GEMV as-is; the signs kernel and the"
                     " Hadamard kernel are not dispatched";
        // A file with no rotation still has no sign vectors to read; the forward must not require
        // them, but if they ARE present that is a contradiction worth an error, not a shrug.
        auto sv = h.kv_a.find("prism.hadamard.sign_values");
        if (sv != h.kv_a.end() && !sv->second.empty()) {
            why = "hadamard.version says there is no rotation but prism.hadamard.sign_values is"
                  " present and non-empty - a rotated export read without its rotation is the worst"
                  " kind of silent wrong answer, so this is refused rather than ignored";
            return false;
        }
        return true;
    }

    if (!p.hat_transform.empty() && p.hat_transform != "normalized-sylvester-walsh-hadamard") {
        why = "prism.hadamard.transform = \"" + p.hat_transform + "\" is not the one transform"
              " implemented (normalized-sylvester-walsh-hadamard); the engine will not apply a"
              " rotation it has not verified";
        return false;
    }
    if (p.hat_sign_mode == "folded") p.hat_explicit_signs = false;
    else p.hat_explicit_signs = true;               // "explicit" or absent: the 9B's behaviour

    // The block size. Required for an active rotation - 4096 is NOT a default here, because the
    // FWHT of the wrong block size is an orthogonal matrix that just is not the model's.
    bool have_blk = false;
    why.clear();
    if (bonsai_md_i(h, m, kBk, 1, t, used, why)) {
        if (!why.empty()) return false;
        p.hat_block = uint32_t(t); have_blk = true;
    }
    if (!have_blk) {
        why = "prism.hadamard.block_size is absent while the rotation is active: the transform"
              " width cannot be defaulted to 4096, that is the 9B's answer not this file's";
        return false;
    }
    if (!bonsai_md_is_pow2(p.hat_block) || p.hat_block < 16u || p.hat_block > 4096u) {
        why = "prism.hadamard.block_size = " + std::to_string(p.hat_block) + ": the FWHT kernel"
              " covers powers of two from 16 to 4096 inclusive; outside that range it is not a"
              " bug to work around, it is a different transform";
        return false;
    }
    if (p.hidden % int(p.hat_block) != 0) {
        why = "embedding_length " + std::to_string(p.hidden) + " is not a multiple of the Hadamard"
              " block " + std::to_string(p.hat_block) + ": the transform is defined per block, so a"
              " partial block would be left unrotated while the weights were rotated with it";
        return false;
    }
    // hat4096.comp stays the path when the metadata says 4096 AND the runtime's own materialised-H
    // equivalent is present (the .spv). It is bit-frozen because ten logs quote the numbers it
    // produced; a 4096-wide FWHT computes the same operator with a different summation order and
    // fp32 addition is not associative, so "equivalent" is not "identical" and the 9B keeps its
    // kernel.
    p.hat_use_fwht = p.hat_block != 4096u;

    // sign widths + value count, read rather than assumed (the 9B: [4096, 12288] and 16384 values).
    std::vector<std::string> sw;
    std::string sw_key;
    why.clear();
    if (bonsai_md_arr(h, m, kSw, 1, sw, sw_key, &why)) {
        p.sign_widths.clear();
        for (auto &s : sw) p.sign_widths.push_back(uint32_t(strtoul(s.c_str(), nullptr, 10)));
    } else if (!why.empty()) return false;
    std::string sv_key;
    const std::vector<std::string> *sv = nullptr;
    for (const auto &pre : m.pre) {
        const std::string k = bonsai_md_key(pre, kSv[0]);
        auto it = h.kv_a.find(k);
        if (it != h.kv_a.end()) { sv = &it->second; sv_key = k; break; }
    }
    if (sv) p.n_sign_values = sv->size();
    if (p.hat_explicit_signs) {
        uint64_t want = 0;
        for (auto x : p.sign_widths) want += x;
        if (!p.sign_widths.empty() && want != p.n_sign_values && p.n_sign_values != 0) {
            why = sv_key + " has " + std::to_string(p.n_sign_values) + " entries but the declared"
                  " sign_widths sum to " + std::to_string(want);
            return false;
        }
        // Every declared width must be a width this engine actually rotates, i.e. one of the two
        // activation widths. A third one means the export rotates something the forward does not -
        // which is not a bug to paper over, it is a different graph.
        for (auto x : p.sign_widths)
            if (int(x) != p.hidden && int(x) != p.ffn) {
                why = "sign_widths contains " + std::to_string(x) + ", which is neither"
                      " embedding_length " + std::to_string(p.hidden) + " nor feed_forward_length " +
                      std::to_string(p.ffn) + " - the engine has no activation of that width to"
                      " multiply signs into";
                return false;
            }
    }
    p.hat_rule = "version " + std::to_string(p.hat_version) + ", block " + std::to_string(p.hat_block) +
                 ", kernel " + (p.hat_use_fwht ? "hadamard_fwht.comp (FWHT, 1/sqrt(block))"
                                               : "hat4096.comp (the 9B's bit-frozen path)") +
                 ", signs " + (p.hat_explicit_signs ? "explicit" : "folded");
    return true;
}

// ------------------------------------------------------------------ structural validation
// Everything the arena math and the forward assume, checked once before a byte is uploaded, and
// reported as the per-layer byte table rather than as prose. Returns false with a reason.
inline bool model_plan_validate(const ModelPlan &p, int v_q8, std::string &why) {
    char b[256];
    if (p.n_layer <= 0)   { snprintf(b, sizeof b, "block_count %d is not positive", p.n_layer); why = b; return false; }
    if (p.hidden <= 0 || p.hidden % 4) { snprintf(b, sizeof b, "embedding_length %d: the vec4 kernels need a multiple of 4", p.hidden); why = b; return false; }
    if (p.ffn <= 0 || p.ffn % 4)       { snprintf(b, sizeof b, "feed_forward_length %d: same vec4 rule", p.ffn); why = b; return false; }
    if (p.n_head <= 0 || p.n_head_kv <= 0) { snprintf(b, sizeof b, "head counts %d/%d", p.n_head, p.n_head_kv); why = b; return false; }
    if (p.head_dim <= 0 || p.head_dim % 4)  { snprintf(b, sizeof b, "head_dim %d: fp16 K packs two per uint", p.head_dim); why = b; return false; }
    if (p.head_dim % 32 || (p.head_dim / 32) % 2) {
        snprintf(b, sizeof b, "head_dim %d: a q8_0 V block (32) must fit a head row and pair its scales", p.head_dim);
        why = b; return false;
    }
    if (int(p.layer_kind.size()) != p.n_layer) { snprintf(b, sizeof b, "layer_kind has %zu entries for %d layers", p.layer_kind.size(), p.n_layer); why = b; return false; }
    if (p.hat_active) {
        if (p.hat_block == 0 || !bonsai_md_is_pow2(p.hat_block)) {
            snprintf(b, sizeof b, "rotation active with block size %u", p.hat_block); why = b; return false;
        }
        if (p.hidden % int(p.hat_block)) {
            snprintf(b, sizeof b, "hidden %d not a multiple of Hadamard block %u", p.hidden, p.hat_block);
            why = b; return false;
        }
        if (p.hat_explicit_signs && p.n_sign_values == 0) {
            snprintf(b, sizeof b, "sign_mode explicit but prism.hadamard.sign_values is absent");
            why = b; return false;
        }
    }
    if (p.n_ssm_layers() > 0) {
        if (p.state_size != 128) {
            snprintf(b, sizeof b, "ssm.state_size %d: gdn_delta.comp is written with local_size_x = 128"
                                  " (one lane per state column element), so a different dim is a new"
                                  " kernel, not a push constant", p.state_size);
            why = b; return false;
        }
        if (p.conv_kernel != 4) {
            snprintf(b, sizeof b, "ssm.conv_kernel %d: conv_silu.comp hardcodes 4 taps and a 3-deep"
                                  " history (c*4 / c*3); a different kernel length is a new shader", p.conv_kernel);
            why = b; return false;
        }
        if (p.state_size && p.inner % p.state_size) {
            snprintf(b, sizeof b, "ssm.inner_size %d is not a multiple of state_size %d - the v-head"
                                  " count inner/dim is what sizes the state", p.inner, p.state_size);
            why = b; return false;
        }
        // The three numbers the GDN dispatches still carry as PUSH-CONSTANT LITERALS (bonsai_forward.h
        // gdn_block: {32,128,16,...} and permute_vg {128,32,16,1}). They are checked against the file
        // rather than read from it, which is the honest way to keep a differently-shaped recurrent
        // block from running: it REFUSES at load. Reading them from the plan too would make the check
        // tautological and the shader would silently receive values it was never written for.
        if (p.ts_rank != 32 || p.group_count != 16) {
            snprintf(b, sizeof b, "ssm.time_step_rank %d / ssm.group_count %d: the GDN and permute"
                                  " dispatches pass 32 and 16 as literals (gdn_block in"
                                  " bonsai_forward.h), so this file's recurrent graph is not the one"
                                  " those kernels implement", p.ts_rank, p.group_count);
            why = b; return false;
        }
        if (p.gdn_channels != 2 * p.inner) {
            snprintf(b, sizeof b, "blk.0.attn_qkv.weight has %d rows but ssm.inner_size is %d:"
                                  " the forward sizes the conv history from the TENSOR and the"
                                  " budget mirror here sizes it as 2*inner (q and k adjacent, inner"
                                  " rows each). They disagree, so one of the two would scribble over"
                                  " the next layer's history", p.gdn_channels, p.inner);
            why = b; return false;
        }
    }
    // The rotation walks whole activation vectors, so every width it rotates has to divide into
    // blocks: hidden for the projections and the FFN's gate/up input, ffn for the down input (3
    // blocks at 12288/4096 on the 9B). A remainder would rotate part of a vector.
    if (p.hat_active && p.hat_block) {
        if (p.ffn % int(p.hat_block)) {
            snprintf(b, sizeof b, "feed_forward_length %d is not a multiple of the Hadamard block %u"
                                  " - the FFN-down input would be rotated only in part",
                     p.ffn, p.hat_block);
            why = b; return false;
        }
    }
    // attn_q's output is the rotated vector the attention kernels slice into heads, so the head
    // geometry and the activation width have to close over each other.
    if (p.n_head > 0 && p.head_dim > 0 && p.n_head * p.head_dim != p.hidden) {
        snprintf(b, sizeof b, "attention.head_count %d x head_dim %d = %d != embedding_length %d:"
                              " the forward rotates %d floats and the normrope kernel slices %d,"
                              " and the two only agree on this file's 9B-shaped case",
                 p.n_head, p.head_dim, p.n_head * p.head_dim, p.hidden, p.hidden, p.n_head * p.head_dim);
        why = b; return false;
    }
    // The one that matters most: an SSM layer charged for KV.
    for (int i = 0; i < p.n_layer; ++i) {
        if (!p.is_attn_layer(i) && p.kv_bytes_per_layer(i, v_q8) != 0ull) {
            snprintf(b, sizeof b, "block %d is recurrent but is charged %llu B/token of KV",
                     i, (unsigned long long)p.kv_bytes_per_layer(i, v_q8));
            why = b; return false;
        }
    }
    why.clear();
    return true;
}

// ------------------------------------------------------------------ build
static bool model_plan_build(const GgufHeader &h, ModelPlan &p) {
    if (!model_plan_hparams(h, p)) return false;
    std::string why;
    if (!model_plan_contract(h, p, why)) {
        fprintf(stderr, "model: layer/rotation contract rejected: %s\n", why.c_str());
        return false;
    }
    p.data_start = h.data_start;
    uint64_t c_src = 0, c_code = 0, c_scl = 0, c_f32 = 0;
    for (const auto &t : h.t) {
        WeightPlan x;
        x.name = t.name;
        x.dtype = t.dtype;
        x.file_off = t.off;
        if (t.n_dims != 1 && t.n_dims != 2) {
            fprintf(stderr, "model: %s has %u dims, expected 1 or 2\n", t.name.c_str(), t.n_dims);
            return false;
        }
        x.K = uint32_t(t.d[0]);
        x.M = uint32_t(t.n_dims == 2 ? t.d[1] : 1);
        uint64_t elems = uint64_t(x.K) * x.M;
        p.n_weights += elems;
        if (t.dtype == 143) {
            // THE K RULE. PTQ1_0 is a BLOCK format: a row is exactly K/128 blocks of 28 B, so a K
            // that is not a multiple of 128 has no legal byte count. Rejecting here is the only
            // place a rejection can still be heard: without it, n_blk = K/128 truncates, every
            // byte count below is short by (K%128)/128 of a row, the loader reads the wrong stride
            // (repack_ptq1_0.comp indexes src by (row*n_layer_blk + blk)*7 with n_layer_blk from
            // this very division) and the GEMV reads codes past the end of the tensor - which on
            // WDDM is not an error, it is the next tensor's bytes.
            if (x.K % PTQ_BLOCK_W_ELEMS) {
                fprintf(stderr, "model: %s K=%u is not a multiple of 128 - PTQ1_0 cannot tile it\n",
                        t.name.c_str(), x.K);
                return false;
            }
            x.n_blk = x.K / PTQ_BLOCK_W_ELEMS;
            x.bytes_src = x.M * x.n_blk * PTQ_BLOCK_BYTES;
            x.bytes_codes = uint64_t(x.M) * (x.K / CODES_PER_VEC4) * 16ull;
            x.bytes_scale = x.M * x.n_blk * 4ull * SCALE_PER_BLOCK;
            x.ptq_word0 = c_src / 4;
            x.codes_word0 = c_code / 16;
            x.scale_word0 = c_scl / 4;
            c_src += align_up_b(x.bytes_src, 16);
            c_code += align_up_b(x.bytes_codes, 16);
            c_scl += align_up_b(x.bytes_scale, 16);
            p.n_ptq_tensors++;
        } else if (t.dtype == 0) {
            x.bytes_f32 = elems * 4ull;
            x.f32_off = c_f32;
            c_f32 += align_up_b(x.bytes_f32, 16);
            p.n_f32_tensors++;
        } else {
            fprintf(stderr, "model: %s dtype %u unsupported\n", t.name.c_str(), t.dtype);
            return false;
        }
        p.w.push_back(x);
    }
    p.bytes_src_total = c_src;
    p.bytes_codes_total = c_code;
    p.bytes_scale_total = c_scl;
    p.bytes_f32_total = c_f32;
    // The recurrent channel count, taken from the tensor the forward's conv arena is sized from
    // (0 when the file has no attn_qkv at all, i.e. a plain transformer). See gdn_channels.
    if (const WeightPlan *qkv = p.find("blk.0.attn_qkv.weight")) p.gdn_channels = int(qkv->M);
    return true;
}

// The budget print. Every line is a number a report cites, so it is emitted by the engine itself
// rather than transcribed by hand.
inline void model_plan_report(const ModelPlan &p, uint64_t device_local_mib) {
    const uint64_t MiB = 1048576ull, GiB = 1073741824ull;
    const uint64_t repacked = p.bytes_codes_total + p.bytes_scale_total;
    const int nat = p.n_attn_layers(), ngdn = p.n_layer - nat;
    // DERIVED, not typed: the per-token KV price is the sum of the per-layer charge, which is 0 for
    // every layer the resolver did not type as attention. The old expression
    // (2 * n_head_kv * head_dim * 2 * n_attn) is reproduced exactly by that sum for an fp16 V - the
    // report prints both and flags a disagreement, so the two derivations cross-check instead of one
    // silently replacing the other.
    const uint64_t kv_per_token = p.kv_bytes_per_token(0);
    const uint64_t kv_per_token_q8 = p.kv_bytes_per_token(1);
    const uint64_t kv_closed_form = 2ull * p.n_head_kv * p.head_dim * 2ull * uint64_t(nat);
    const uint64_t gdn_state = uint64_t(p.state_size) * p.state_size *
                               uint64_t(p.inner / (p.state_size ? p.state_size : 1)) * 4ull;
    const uint64_t gdn_conv = uint64_t(p.conv_kernel - 1) * uint64_t(p.inner) * 2ull * 4ull;
    const uint64_t recurrent = (gdn_state + gdn_conv) * uint64_t(ngdn);

    printf("\n=== weight plan, read from the tensor table (实测, not estimated) ===\n");
    printf("  tensors %zu = %zu PTQ1_0 + %zu f32   weights %llu elems = %.3f B\n",
           p.w.size(), size_t(p.n_ptq_tensors), size_t(p.n_f32_tensors),
           (unsigned long long)p.n_weights, p.n_weights / 1e9);
    printf("  per-tensor shapes are K x M; the activation widths this file declares"
           " (prism.hadamard.sign_widths) are [");
    for (size_t i = 0; i < p.sign_widths.size(); ++i) printf("%s%u", i ? ", " : "", p.sign_widths[i]);
    printf("], which is what the rotation has to cover.\n");
    printf("  checkpoint's own block form : %9llu B = %.3f GiB  (1.75 bpw)\n",
           (unsigned long long)p.bytes_src_total, double(p.bytes_src_total) / GiB);
    printf("  our 2-bit random-access form: %9llu B = %.3f GiB  (codes %.3f GiB + scales %.3f GiB, 2.125 bpw)\n",
           (unsigned long long)repacked, double(repacked) / GiB,
           double(p.bytes_codes_total) / GiB, double(p.bytes_scale_total) / GiB);
    printf("  f32 blob                    : %9llu B = %.3f MiB\n",
           (unsigned long long)p.bytes_f32_total, double(p.bytes_f32_total) / MiB);
    // A model with NO softmax layer has no KV cache at all, so this ratio is not a number - say so
    // instead of printing "inf tokens". (The 27B question is exactly this: how much of the 4 GB the
    // layout costs buys nothing, because the layers that would have used it are recurrent.)
    if (kv_per_token)
        printf("  LAYOUT COST = %.3f GiB extra = %.0f tokens of KV at %llu B/token\n",
               double(repacked - p.bytes_src_total) / GiB,
               double(repacked - p.bytes_src_total) / double(kv_per_token),
               (unsigned long long)kv_per_token);
    else
        printf("  LAYOUT COST = %.3f GiB extra, and this checkpoint has no attention layer, so the"
               " extra buys 0 tokens of KV (the whole budget is recurrent state + logits)\n",
               double(repacked - p.bytes_src_total) / GiB);
    printf("\n=== the rest of the footprint ===\n");
    printf("  metadata prefixes : %s\n",
           p.meta_prefixes.empty() ? "NOT RESOLVED (model_plan_contract() has not run;"
                                   " this is the budget-only path)" : p.meta_prefixes.c_str());
    printf("  layer typing    : %s\n",
           p.layer_rule.empty() ? "NOT RESOLVED - see model_plan_contract()" : p.layer_rule.c_str());
    printf("  rotation        : %s\n",
           p.hat_rule.empty() ? "NOT RESOLVED - see model_plan_contract()" : p.hat_rule.c_str());
    printf("  blocks: %d recurrent + %d softmax-attention of %d (interval key = %d,%s)\n", ngdn, nat,
           p.n_layer, p.full_attn_interval,
           p.layer_kind.empty() ? " interval rule in force" : " explicit typing in force");
    printf("  recurrent state, context-INDEPENDENT: %llu B/layer state + %llu B/layer conv = %.1f MiB over %d layers\n",
           (unsigned long long)gdn_state, (unsigned long long)gdn_conv, double(recurrent) / MiB, ngdn);
    printf("  KV cache: %llu B/token (%.1f MiB per 1k tokens), fp16, %d layers | q8_0 V: %llu B/token\n",
           (unsigned long long)kv_per_token, double(kv_per_token) * 1024 / MiB, nat,
           (unsigned long long)kv_per_token_q8);
    printf("  device-local heap: %llu MiB\n", (unsigned long long)device_local_mib);
    if (kv_per_token != kv_closed_form)
        printf("  [!] per-layer sum %llu != closed form %llu for fp16 KV - the typing and the"
               " formula disagree, report this\n",
               (unsigned long long)kv_per_token, (unsigned long long)kv_closed_form);

    // ---- THE KV CHARGE TABLE. Printed per layer because "an SSM layer gets no cache" is the
    // invariant the whole mixed-stack design hangs on, and a log line that shows 0 B next to every
    // recurrent layer is evidence in a way that a comment is not.
    printf("\n=== per-layer cache charge (structural: recurrent layers own state, never KV) ===\n");
    uint64_t sum_fp16 = 0, sum_q8 = 0, sum_state = 0;
    for (int b = 0; b < p.n_layer; ++b) {
        const bool att = p.is_attn_layer(b);
        const uint64_t k16 = p.kv_bytes_per_layer(b, 0);
        const uint64_t kq8 = p.kv_bytes_per_layer(b, 1);
        const uint64_t st  = att ? 0ull : p.ssm_bytes_per_layer();
        sum_fp16 += k16; sum_q8 += kq8; sum_state += st;
        if (b % 4 == 0 || nat <= 16) {
            printf("  b%-2d %s  kv fp16 %8llu B/token   kv q8_0-V %8llu B/token   state %10llu B\n",
                   b, att ? "attn" : "ssm ", (unsigned long long)k16, (unsigned long long)kq8,
                   (unsigned long long)st);
        }
    }
    printf("  TOTAL over %d layers: %d attention -> %llu B/token fp16-V | %llu B/token q8_0-V;"
           " %d recurrent -> %llu B of context-independent state\n",
           p.n_layer, nat, (unsigned long long)sum_fp16, (unsigned long long)sum_q8, ngdn,
           (unsigned long long)sum_state);
    // The sum over here and kv_bytes_per_token() above are the SAME function applied to the same
    // typing, so this check is not new evidence - it is the guard against this file ever growing a
    // second path to the number. The real cross-check (the arena the forward builds vs. this sum) is
    // the assert in fwd_init(), which the CPU mirror in align_27b.exe reproduces.
    if (sum_fp16 != kv_per_token || sum_q8 != kv_per_token_q8)
        printf("  [!] the per-layer charge disagrees with kv_bytes_per_token() - report this line\n");
}

#endif  // BONSAI_MODEL_H
