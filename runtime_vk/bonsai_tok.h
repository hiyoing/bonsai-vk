// bonsai_tok.h - our own byte-level BPE for this checkpoint. Written from the format, not from
// llama.cpp: the only thing it takes from the file is the vocab + merges arrays, which are data.
//
// What the file actually says (measured, runs/tok_meta.txt):
//   tokenizer.ggml.model = "gpt2"   -> byte-level BPE: every token is stored as the GPT-2
//                                      bytes_to_unicode image of its raw bytes (' ' -> "Ä "),
//                                      so a token's bytes are recovered by inverting that table
//   tokenizer.ggml.pre   = "qwen35" -> pre-tokenizer splits text into pieces before merging.
//                                      Implemented as the Qwen2 alternative set, in the regex's own
//                                      priority order, by a hand-written scanner.
//   token_type census: 1 NORMAL 248044 / 3 CONTROL 27 / 4 USER_DEFINED 6 / 5 BYTE 243 ("<0xNN>")
//   vocab 248320, merges 247587, bos 248044, eos 248046, add_bos 0
//
// Correctness target: the same id sequence as the reference tokenizer. align_tokenizer.cpp is what
// measures that, against llama-tokenize.exe from the reference build. Every probe that disagrees is
// a bug in this file, not in the data.

#pragma once

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

#include "bonsai_gguf.h"

// ------------------------------------------------------------------ utf-8

inline bool utf8_next(const std::string& s, size_t& i, uint32_t& cp) {
    size_t n = s.size();
    if (i >= n) return false;
    unsigned char c0 = (unsigned char)s[i];
    size_t len = c0 < 0x80 ? 1 : (c0 >= 0xF0 ? 4 : c0 >= 0xE0 ? 3 : c0 >= 0xC0 ? 2 : 1);
    if (i + len > n) len = 1;
    uint32_t v = c0 & (len == 1 ? 0x7Fu : 0x3Fu);
    for (size_t k = 1; k < len; ++k) v = (v << 6) | ((unsigned char)s[i + k] & 0x3Fu);
    cp = v; i += len; return true;
}
inline void utf8_put(std::string& out, uint32_t cp) {
    if (cp < 0x80) out += (char)cp;
    else if (cp < 0x800) { out += (char)(0xC0 | (cp >> 6)); out += (char)(0x80 | (cp & 0x3F)); }
    else if (cp < 0x10000) { out += (char)(0xE0 | (cp >> 12)); out += (char)(0x80 | ((cp >> 6) & 0x3F));
                             out += (char)(0x80 | (cp & 0x3F)); }
    else { out += (char)(0xF0 | (cp >> 18)); out += (char)(0x80 | ((cp >> 12) & 0x3F));
           out += (char)(0x80 | ((cp >> 6) & 0x3F)); out += (char)(0x80 | (cp & 0x3F)); }
}

// ------------------------------------------------------------------ gpt2 byte <-> codepoint table

struct ByteMap {
    uint32_t fwd[256];
    int back[0x11000];
    ByteMap() {
        std::vector<int> bs;
        for (int b = 33; b <= 126; ++b) bs.push_back(b);
        for (int b = 161; b <= 172; ++b) bs.push_back(b);
        for (int b = 174; b <= 255; ++b) bs.push_back(b);
        std::vector<int> cs = bs;
        int n = 0;
        for (int b = 0; b < 256; ++b) {
            if (std::find(bs.begin(), bs.end(), b) == bs.end()) { bs.push_back(b); cs.push_back(256 + n); ++n; }
        }
        // bs is NOT in byte order: it starts at 33. The pairing is positional, so index into bs to
        // get the byte and cs to get its codepoint - getting this wrong silently corrupts every
        // multi-byte vocab lookup, which is exactly what the first alignment run measured.
        std::fill(back, back + 0x11000, -1);
        for (int i = 0; i < 256; ++i) {
            int b = bs[i], c = cs[i];
            fwd[b] = (uint32_t)c;
            if (c >= 0 && c < 0x11000) back[c] = b;
        }
    }
    // "Ä it" -> {' ','i','t'}; falls back to the stored bytes when a codepoint is not in the table
    bool text_to_bytes(const std::string& text, std::string& out) const {
        out.clear();
        size_t i = 0; uint32_t cp;
        while (i < text.size()) {
            if (!utf8_next(text, i, cp)) break;
            if (cp < 0x11000 && back[(int)cp] >= 0) { out += (char)back[(int)cp]; continue; }
            if (cp < 0x800000) utf8_put(out, cp);      // not a mapped byte: keep the utf-8 bytes
        }
        return true;
    }
    std::string bytes_to_text(const std::string& bytes) const {
        std::string out;
        for (unsigned char c : bytes) utf8_put(out, fwd[c]);
        return out;
    }
};

// ------------------------------------------------------------------ unicode classes for the splitter
// \p{L} / \p{N} by script for the ranges this distribution uses, minus the punctuation blocks that
// must NOT count as letters (ideographic full stop U+3002, CJK brackets, fullwidth forms).

inline bool cp_letter(uint32_t cp) {
    if (cp < 128) return (cp >= 'A' && cp <= 'Z') || (cp >= 'a' && cp <= 'z');
    if (cp >= 0x4E00 && cp <= 0x9FFF) return true;
    if (cp >= 0x3400 && cp <= 0x4DBF) return true;
    if (cp >= 0xF900 && cp <= 0xFAFF) return true;
    if (cp >= 0x20000 && cp <= 0x2A6DF) return true;
    if (cp >= 0xAC00 && cp <= 0xD7A3) return true;
    if (cp >= 0x1100 && cp <= 0x11FF) return true;
    if (cp >= 0x3040 && cp <= 0x30FF) return true;
    if (cp >= 0x31F0 && cp <= 0x31FF) return true;
    if (cp >= 0x0400 && cp <= 0x04FF) return true;
    if (cp >= 0x0370 && cp <= 0x03FF) return true;
    if (cp >= 0x0600 && cp <= 0x06FF) return true;
    if (cp >= 0x0900 && cp <= 0x097F) return true;
    if (cp >= 0x0E00 && cp <= 0x0E7F) return true;
    if (cp >= 0x1000 && cp <= 0x109A) return true;
    if (cp >= 0x0530 && cp <= 0x058F) return true;
    if (cp >= 0x10A0 && cp <= 0x10FF) return true;
    if (cp >= 0x00C0 && cp <= 0x02AF) {
        if (cp == 0x00D7 || cp == 0x00F7) return false;
        if (cp >= 0x02C6 && cp <= 0x02CF) return false;
        if (cp >= 0x02D8 && cp <= 0x02DD) return false;
        return true;
    }
    if (cp >= 0x1E00 && cp <= 0x1EFF) return true;
    if (cp >= 0x2C60 && cp <= 0x2C7F) return true;
    if (cp >= 0xA000 && cp <= 0xA4CF) return true;
    if (cp >= 0xFB00 && cp <= 0xFB17) return true;
    return false;
}
inline bool cp_number(uint32_t cp) {
    if (cp >= '0' && cp <= '9') return true;
    if (cp >= 0x0660 && cp <= 0x0669) return true;
    if (cp >= 0x06F0 && cp <= 0x06F9) return true;
    if (cp >= 0x0966 && cp <= 0x096F) return true;
    if (cp == 0x3007) return true;
    if (cp >= 0xFF10 && cp <= 0xFF19) return true;
    return false;
}
inline bool cp_space(uint32_t cp) {
    return cp == ' ' || cp == '\t' || cp == '\n' || cp == '\r' || cp == '\v' || cp == '\f' ||
           cp == 0x00A0 || cp == 0x1680 || (cp >= 0x2000 && cp <= 0x200A) ||
           cp == 0x2028 || cp == 0x2029 || cp == 0x202F || cp == 0x205F || cp == 0x3000;
}

// ------------------------------------------------------------------ the tokenizer itself

struct Bpe {
    ByteMap bm;
    std::vector<std::string> tok_bytes;              // id -> raw bytes
    std::unordered_map<std::string, int> id_of;      // raw bytes -> id (lowest id wins)
    std::unordered_map<uint64_t, int> rank;          // (a<<32|b) -> merge rank (lower = earlier)
    std::vector<std::string> specials;               // literal text, longest-first, for parse_special
    std::unordered_map<std::string, int> special_id;
    int bos = -1, eos = -1, unk = -1;
    bool add_bos = false;
    size_t n_vocab = 0;

    static uint64_t pk(int a, int b) { return (uint64_t)(uint32_t)a << 32 | (uint32_t)(uint32_t)b; }

    bool load(const GgufHeader& h) {
        auto vt = h.kv_a.find("tokenizer.ggml.tokens");
        if (vt == h.kv_a.end()) { fprintf(stderr, "bonsai_tok: header has no vocab\n"); return false; }
        const std::vector<std::string>& toks = vt->second;
        n_vocab = toks.size();
        tok_bytes.assign(n_vocab, std::string());
        for (size_t i = 0; i < n_vocab; ++i) {
            const std::string& t = toks[i];
            if (t.size() == 6 && t.compare(0, 3, "<0x") == 0 && t[5] == '>') {
                int v = 0;
                for (int k = 3; k < 5; ++k) { char c = t[k]; v = v * 16 + (c <= '9' ? c - '0' : ((c | 32) - 'a' + 10)); }
                tok_bytes[i].assign(1, (char)v);
            } else {
                bm.text_to_bytes(t, tok_bytes[i]);
            }
            if (!id_of.count(tok_bytes[i])) id_of[tok_bytes[i]] = (int)i;   // lowest id wins
        }
        auto mt = h.kv_a.find("tokenizer.ggml.merges");
        if (mt != h.kv_a.end()) {
            const std::vector<std::string>& mg = mt->second;
            rank.reserve(mg.size() * 2);
            for (size_t i = 0; i < mg.size(); ++i) {
                std::string a, b; bool have_b = false; size_t pos = 0;
                while (pos < mg[i].size()) {
                    uint32_t cp; size_t save = pos;
                    if (!utf8_next(mg[i], pos, cp)) break;
                    if (cp == ' ' && !have_b) { have_b = true; continue; }   // first raw space splits
                    std::string piece = mg[i].substr(save, pos - save);
                    if (have_b) b += piece; else a += piece;
                }
                // the merge file stores the GPT-2 image of the bytes ("Ä "), the vocab lookup is
                // keyed on raw bytes, so both halves have to go through the same inversion as the
                // vocab itself.
                std::string ab, bb;
                bm.text_to_bytes(a, ab);
                bm.text_to_bytes(b, bb);
                auto ia = id_of.find(ab), ib = id_of.find(bb);
                if (ia == id_of.end() || ib == id_of.end()) continue;      // pair we cannot form
                // A merge whose result is not in the vocab is dead: llama.cpp's own table would
                // never select it, and selecting it here would stop the whole loop early and leave
                // raw bytes behind. Filter it at load time, not at encode time.
                auto im = id_of.find(ab + bb);
                if (im == id_of.end()) continue;
                uint64_t k = pk(ia->second, ib->second);
                if (!rank.count(k)) rank[k] = (int)i;
            }
        }
        auto it = h.kv_i.find("tokenizer.ggml.bos_token_id"); if (it != h.kv_i.end()) bos = (int)it->second;
        it = h.kv_i.find("tokenizer.ggml.eos_token_id");      if (it != h.kv_i.end()) eos = (int)it->second;
        it = h.kv_i.find("tokenizer.ggml.unknown_token_id");  if (it != h.kv_i.end()) unk = (int)it->second;
        add_bos = h.kv_i.count("tokenizer.ggml.add_bos_token") ? h.kv_i.at("tokenizer.ggml.add_bos_token") != 0 : true;
        if (unk < 0) { auto u = id_of.find(std::string("!")); if (u != id_of.end()) unk = u->second; }

        // special tokens: token_type 3 (CONTROL) and 4 (USER_DEFINED) are matched literally and
        // are never split. They come as "<|...|>" style text in the vocab.
        auto tt = h.kv_a.find("tokenizer.ggml.token_type");
        if (tt != h.kv_a.end()) {
            const std::vector<std::string>& types = tt->second;
            for (size_t i = 0; i < n_vocab && i < types.size(); ++i) {
                int ty = atoi(types[i].c_str());
                if ((ty == 3 || ty == 4) && toks[i].size() > 4 && toks[i][0] == '<') {
                    special_id[toks[i]] = (int)i;
                    specials.push_back(toks[i]);
                }
            }
            std::sort(specials.begin(), specials.end(),
                      [](const std::string& a, const std::string& b) { return a.size() > b.size(); });
        }
        return true;
    }

    // ---------------- pre-tokenizer (qwen2 alternative set, in the regex's own priority order)
    // (?i:'s|'t|'re|'ve|'m|'ll|'d) | [^\r\n\p{L}\p{N}]?\p{L}+ | \p{N}{1}
    // | ?[^\s\p{L}\p{N}]+[\r\n]* | \s*[\r\n]+ | \s+(?!\S) | \s+
    // At every position the alternatives are tried left to right and the first that matches wins -
    // that is exactly what an ordered regex alternation does, backtracking included for \s+(?!\S).

    struct Cp { uint32_t cp; size_t off, len; };
    static std::vector<Cp> cps(const std::string& s) {
        std::vector<Cp> v; size_t i = 0;
        while (i < s.size()) {
            Cp c; c.off = i; uint32_t cp;
            size_t n = i;
            if (!utf8_next(s, n, cp)) break;
            c.cp = cp; c.len = n - i; i = n; v.push_back(c);
        }
        return v;
    }
    static bool is_newline(uint32_t cp) { return cp == '\n' || cp == '\r'; }

    // returns the end index (exclusive, in Cp units) of the piece starting at i
    size_t piece_end(const std::vector<Cp>& v, size_t i) const {
        uint32_t c0 = v[i].cp;
        auto letters = [&](size_t k) { while (k < v.size() && cp_letter(v[k].cp)) ++k; return k; };
        auto punct = [&](size_t k) {
            return k < v.size() && !cp_space(v[k].cp) && !cp_letter(v[k].cp) && !cp_number(v[k].cp);
        };
        // alt 1: (?i:'s|'t|'re|'ve|'m|'ll|'d)
        if (c0 == '\'') {
            static const char cont[][3] = {{'s',0,0},{'t',0,0},{'r','e',0},{'v','e',0},
                                           {'m',0,0},{'l','l',0},{'d',0,0}};
            static const int clen[7] = {1, 1, 2, 2, 1, 2, 1};
            for (int q = 0; q < 7; ++q) {
                bool ok = true;
                for (int k = 0; k < clen[q]; ++k) {
                    if (i + 1 + k >= v.size()) { ok = false; break; }
                    uint32_t c = v[i + 1 + k].cp;
                    char lc = (char)((c >= 'A' && c <= 'Z') ? c + 32 : c);
                    if (lc != cont[q][k]) { ok = false; break; }
                }
                if (ok) return i + 1 + (size_t)clen[q];
            }
        }
        // alt 2: [^\r\n\pL\pN]? \pL+   (the optional char may be a space - that is what makes
        //        " the" one piece instead of " " + "the")
        if (cp_letter(c0)) return letters(i);
        if (!is_newline(c0) && !cp_letter(c0) && !cp_number(c0) && i + 1 < v.size() &&
            cp_letter(v[i + 1].cp)) return letters(i + 1);
        // alt 3: \pN{1}
        if (cp_number(c0)) return i + 1;
        // alt 4: ?[^\s\pL\pN]+[\r\n]*
        if (punct(i)) {
            size_t k = i;
            while (k < v.size() && punct(k)) ++k;
            while (k < v.size() && is_newline(v[k].cp)) ++k;
            return k;
        }
        if (cp_space(c0)) {
            if (punct(i + 1)) {                                  // " 。" : the optional space of alt 4
                size_t k = i + 1;
                while (k < v.size() && punct(k)) ++k;
                while (k < v.size() && is_newline(v[k].cp)) ++k;
                return k;
            }
            size_t run = i; while (run < v.size() && cp_space(v[run].cp)) ++run;
            // alt 5: \s*[\r\n]+ -> up to and including the newline block after the first newline
            for (size_t k = i; k < run; ++k)
                if (is_newline(v[k].cp)) { size_t e = k; while (e < run && is_newline(v[e].cp)) ++e; return e; }
            // alt 6: \s+(?!\S) -> all but the last char of the run, unless the run ends the text
            if (run == v.size()) return run;
            return run - 1;
        }
        return i + 1;
    }

    std::vector<std::string> pre_tokenize(const std::string& text) const {
        std::vector<std::string> out;
        std::vector<Cp> v = cps(text);
        size_t i = 0;
        while (i < v.size()) {
            size_t j = piece_end(v, i);
            if (j <= i) j = i + 1;
            std::string piece;
            for (size_t k = i; k < j; ++k) piece += text.substr(v[k].off, v[k].len);
            out.push_back(piece);
            i = j;
        }
        return out;
    }

    // ---------------- one piece: the reference is BYTE-initial. Each raw utf-8 byte becomes its own
    // token, then merges fire purely by rank - so a merge can (and does) fire across a utf-8
    // code-point boundary. Code-point-initial looks tidier but is wrong: measured on the official
    // oracle it gives 31/32 on runs/tok_probes.txt and only 1/7 on the CJK/latin set, where every
    // failure is "our standalone space 220 vs the reference's merged id" (e.g. " 한국어" = 190925).
    // Byte-initial gives 32/32 and 7/7. BONSAI_TOK_INIT=cp keeps the losing mode for reproduction.
    std::vector<int> encode_piece(const std::string& piece) const {
        std::vector<int> ids;
        const char* init = getenv("BONSAI_TOK_INIT");        // "byte" (default) | "cp"
        bool by_byte = !(init && strcmp(init, "cp") == 0);
        size_t i = 0;
        while (i < piece.size()) {
            size_t n = i; uint32_t cp;
            if (!utf8_next(piece, n, cp)) break;
            std::string cpb = piece.substr(i, n - i);
            auto it = by_byte ? id_of.end() : id_of.find(cpb);
            if (it != id_of.end()) ids.push_back(it->second);
            else for (size_t k = i; k < n; ++k) {
                auto b = id_of.find(std::string(1, piece[k]));
                ids.push_back(b != id_of.end() ? b->second : unk);
            }
            i = n;
        }
        while (ids.size() > 1) {
            int best = -1; uint64_t bk = 0;
            for (size_t i = 0; i + 1 < ids.size(); ++i) {
                uint64_t k = pk(ids[i], ids[i + 1]);
                auto it = rank.find(k);
                if (it != rank.end() && (best < 0 || it->second < best)) { best = it->second; bk = k; }
            }
            if (best < 0) break;
            int a = (int)(bk >> 32), b = (int)(bk & 0xFFFFFFFFu);
            std::string merged = tok_bytes[a] + tok_bytes[b];
            auto idm = id_of.find(merged);
            if (idm == id_of.end()) break;                      // rank without a vocab entry: stop
            int m = idm->second;
            std::vector<int> out;
            out.reserve(ids.size());
            for (size_t i = 0; i < ids.size();) {
                if (i + 1 < ids.size() && ids[i] == a && ids[i + 1] == b) { out.push_back(m); i += 2; }
                else out.push_back(ids[i++]);
            }
            if (out.size() == ids.size()) break;                 // nothing merged: cannot progress
            ids.swap(out);
        }
        return ids;
    }

    std::vector<int> encode(const std::string& text, bool parse_special = true, bool with_bos = false) const {
        std::vector<int> ids;
        if (with_bos && add_bos && bos >= 0) ids.push_back(bos);
        size_t i = 0;
        while (i < text.size()) {
            bool hit = false;
            if (parse_special) {
                for (const std::string& s : specials) {         // specials are never split
                    if (text.compare(i, s.size(), s) == 0) {
                        ids.push_back(special_id.at(s)); i += s.size(); hit = true; break;
                    }
                }
            }
            if (hit) continue;
            size_t next = i;                                     // extend to the next special
            for (const std::string& s : specials) {
                size_t p = text.find(s, i + 1);
                if (p != std::string::npos && (next == i || p < next)) next = p;
            }
            if (next <= i) next = text.size();
            std::string run = text.substr(i, next - i);
            for (const std::string& piece : pre_tokenize(run)) {
                std::vector<int> e = encode_piece(piece);
                ids.insert(ids.end(), e.begin(), e.end());
            }
            i = next;
        }
        return ids;
    }

    std::string decode_ids(const std::vector<int>& ids) const {
        std::string out;
        for (int i : ids) if (i >= 0 && (size_t)i < tok_bytes.size()) out += tok_bytes[i];
        return out;
    }
};
