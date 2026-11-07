// bonsai_gguf.h - our own minimal GGUF header reader. No llama.cpp, no ggml.
//
// Scope: everything the alignment tests and (later) the engine need from the file header:
//   - the metadata KV table (arrays of strings included: vocab + merges)
//   - the tensor table (name, dims, dtype, offset within the data section)
//   - data_start, so tensor bytes can be read with a single pread-style seek.
// It reads ~11 MB of header out of a 2 GB file and nothing else, so it is fast.
//
// Type ids are GGUF's own (0 U8 1 I8 2 U16 3 I16 4 U32 5 I32 6 F32 7 BOOL 8 STR 9 ARRAY
// 10 U64 11 I64 12 F64). Tensor dtype ids are ggml's, and this file uses the private 143
// (PTQ1_0) for quantized weights plus 0 (f32) for norms/biases/ssm parameters.

#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

struct GgufTensor {
    std::string name;
    uint32_t    n_dims = 0;
    uint64_t    d[4] = {0, 0, 0, 0};
    uint32_t    dtype = 0;
    uint64_t    off = 0;          // relative to data_start
};

struct GgufHeader {
    std::vector<GgufTensor> t;
    std::map<std::string, std::string> kv_s;
    std::map<std::string, std::vector<std::string>> kv_a;
    std::map<std::string, int64_t> kv_i;
    std::map<std::string, double>  kv_f;
    uint64_t version = 0, n_tensors = 0, data_start = 0;
    uint32_t alignment = 32;

    const GgufTensor* find(const std::string& name) const {
        for (const auto& x : t) if (x.name == name) return &x;
        return nullptr;
    }
};

class Rdr {
    FILE* f_;
public:
    explicit Rdr(FILE* f) : f_(f) {}
    void skip(size_t n) { fseek(f_, (long)n, SEEK_CUR); }
    std::vector<uint8_t> raw(size_t n) {
        std::vector<uint8_t> b(n);
        if (n && fread(b.data(), 1, n, f_) != n) { bad(); }
        return b;
    }
    uint64_t u64() { uint64_t v; if (fread(&v, 8, 1, f_) != 1) bad(); return v; }
    uint32_t u32() { uint32_t v; if (fread(&v, 4, 1, f_) != 1) bad(); return v; }
    uint16_t u16() { uint16_t v; if (fread(&v, 2, 1, f_) != 1) bad(); return v; }
    int16_t  i16() { int16_t  v; if (fread(&v, 2, 1, f_) != 1) bad(); return v; }
    int32_t  i32() { int32_t  v; if (fread(&v, 4, 1, f_) != 1) bad(); return v; }
    float    f32() { float    v; if (fread(&v, 4, 1, f_) != 1) bad(); return v; }
    int64_t  i64() { int64_t  v; if (fread(&v, 8, 1, f_) != 1) bad(); return v; }
    double   f64() { double   v; if (fread(&v, 8, 1, f_) != 1) bad(); return v; }
    uint8_t  u8()  { uint8_t  v; if (fread(&v, 1, 1, f_) != 1) bad(); return v; }
    std::string str() {
        size_t n = (size_t)u64();
        std::vector<uint8_t> b = raw(n);
        return std::string(b.begin(), b.end());
    }
    void bad() { fprintf(stderr, "bonsai_gguf: short read\n"); exit(1); }
};

inline bool gguf_load(const char* path, GgufHeader& h) {
    FILE* f = fopen(path, "rb");
    if (!f) { fprintf(stderr, "bonsai_gguf: cannot open %s\n", path); return false; }
    Rdr r(f);
    if (r.raw(4) != std::vector<uint8_t>{'G', 'G', 'U', 'F'}) {
        fprintf(stderr, "bonsai_gguf: %s is not GGUF\n", path); fclose(f); return false;
    }
    h.version = r.u32();
    h.n_tensors = r.u64();
    uint64_t n_kv = r.u64();

    // one scalar/array value. arrays of arrays are not needed by this model, so a nested array
    // is reported and skipped rather than silently mis-read.
    struct Val {
        uint32_t elem;                 // for arrays: element type
        std::vector<std::string>* sa;  // string array payload
    };
    auto read_scalar = [&](uint32_t ty, std::string* out_s, int64_t* out_i, double* out_f) {
        int64_t iv = 0; double fv = 0; std::string sv;
        switch (ty) {
            case 0: iv = r.u8(); break;
            case 1: iv = (int8_t)r.u8(); break;
            case 2: iv = r.u16(); break;
            case 3: iv = r.i16(); break;
            case 4: iv = r.u32(); break;
            case 5: iv = r.i32(); break;
            case 6: { float x = r.f32(); iv = 0; fv = (double)x; } break;
            case 7: iv = r.u8(); break;
            case 8: sv = r.str(); break;
            case 10: iv = (int64_t)r.u64(); break;
            case 11: iv = r.i64(); break;
            case 12: fv = r.f64(); break;
            default: fprintf(stderr, "bonsai_gguf: bad scalar type %u\n", ty); exit(1);
        }
        if (out_s) *out_s = std::move(sv);
        if (out_i) *out_i = iv;
        if (out_f) *out_f = fv;
    };

    for (uint64_t i = 0; i < n_kv; ++i) {
        std::string key = r.str();
        uint32_t ty = r.u32();
        if (ty == 9) {                       // ARRAY
            uint32_t et = r.u32();
            uint64_t n = r.u64();
            if (et == 8) {                   // array of strings - vocab, merges
                auto& v = h.kv_a[key];
                v.resize(n);
                for (uint64_t k = 0; k < n; ++k) v[k] = r.str();
            } else {                         // numeric array: kept as decimal strings so the
                auto& v = h.kv_a[key];       // tokenizer can read token_type without a second pass
                v.resize(n);
                for (uint64_t k = 0; k < n; ++k) {
                    std::string s; int64_t iv = 0; double fv = 0;
                    read_scalar(et, &s, &iv, &fv);
                    v[k] = (et == 6 || et == 12) ? std::to_string(fv) : std::to_string(iv);
                }
                h.kv_i[key + ".n"] = (int64_t)n;
            }
        } else {
            std::string s; int64_t iv = 0; double fv = 0;
            read_scalar(ty, &s, &iv, &fv);
            if (ty == 8) h.kv_s[key] = s;
            else if (ty == 6 || ty == 12) h.kv_f[key] = fv;
            else h.kv_i[key] = iv;
        }
    }

    for (uint64_t i = 0; i < h.n_tensors; ++i) {
        GgufTensor x;
        x.name = r.str();
        x.n_dims = r.u32();
        for (uint32_t k = 0; k < x.n_dims; ++k) x.d[k] = r.u64();
        x.dtype = r.u32();
        x.off = r.u64();
        h.t.push_back(std::move(x));
    }
    auto it = h.kv_i.find("general.alignment");
    if (it != h.kv_i.end() && it->second > 0) h.alignment = (uint32_t)it->second;
    uint64_t pos = (uint64_t)ftell(f);
    h.data_start = (pos + h.alignment - 1) / h.alignment * h.alignment;
    fclose(f);
    return true;
}
