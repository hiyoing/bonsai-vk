// bonsai_chat.h - the model's own chat template, transcribed into C++.
//
// WHY THIS FILE EXISTS. Without it the engine feeds the base model a bare question, and a base model
// answers a bare question by continuing the surface form: runs/ignition_v2.log showed the old run
// inventing its own task ("(3-2-1) 在机器学习中的意义") and then writing pseudo-scaffolding. That is
// not a broken model, it is an un-wrapped prompt.
//
// The template is tokenizer.chat_template of THIS checkpoint: 7756 chars, dumped verbatim into
// runs/chat_tpl_probe.log. The markers it emits are CONTROL tokens of this vocab, so they are looked
// up here BY ID and their literal text never appears in this source. Two reasons: the id table is the
// truth (a vocab edit cannot be missed here), and control-token literals in an agent/tool input stream
// have been observed to truncate edits in this repo (HANDOVER.md §5.6 records that).
//   248045 im_start   248046 im_end   248068 think-open   248069 think-close
//
// The branch implemented is the template's own "no tools" path, transcribed literally:
//   system            -> [im_start] "system\n"    trim(content) [im_end] "\n"
//   user              -> [im_start] "user\n"      trim(content) [im_end] "\n"
//   add_generation_prompt:
//     enable_thinking false -> [im_start] "assistant\n" [think-open] "\n\n" [think-close] "\n\n"
//     otherwise             -> [im_start] "assistant\n" [think-open] "\n"
// It is NOT hand-written id sequences: the rendered STRING goes through Bpe::encode, which is the
// path that was aligned to the reference tokenizer 39/39 (FAITHFUL_REPRO.md §1). The id-built twin
// below (render_ids) exists only as a self-check: fwd_run prints whether the two paths are IDENTICAL
// (runs/chat_think_v1.log:90, chat_nothink_v1.log:90, chat_en_v1.log:90 all report IDENTICAL), and
// ids_are_singles() refuses to render at all if a control id does not round-trip as one token.
// NOT DONE: comparing the rendered id sequence against the oracle's own `llama-tokenize
// -chat-template` output id-by-id (HANDOVER §7.1 step D4). There is no verify_chat_ids.py in tools/;
// the template itself was dumped by tools/gguf_dump.py -> runs/chat_template.txt.

#pragma once

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "bonsai_tok.h"

namespace chat {

inline const int kImStart   = 248045;
inline const int kImEnd     = 248046;
inline const int kThinkOpen = 248068;
inline const int kThinkClose = 248069;

// the literal bytes of one CONTROL id, straight from the vocab
inline std::string ctl(const Bpe &tok, int id) {
    if (id < 0 || size_t(id) >= tok.tok_bytes.size()) return std::string();
    return tok.tok_bytes[size_t(id)];
}

// the template runs |trim over every message body
inline std::string trim(const std::string &s) {
    size_t a = 0, b = s.size();
    auto ws = [](char c) { return c == ' ' || c == '\t' || c == '\n' || c == '\r' ||
                                c == '\v' || c == '\f'; };
    while (a < b && ws(s[a])) ++a;
    while (b > a && ws(s[b - 1])) --b;
    return s.substr(a, b - a);
}

inline std::string render(const Bpe &tok, const std::string &system,
                          const std::string &user, bool thinking) {
    const std::string S = ctl(tok, kImStart), E = ctl(tok, kImEnd);
    const std::string T = ctl(tok, kThinkOpen), X = ctl(tok, kThinkClose);
    std::string o;
    if (!system.empty()) o += S + "system\n" + trim(system) + E + "\n";
    o += S + "user\n" + trim(user) + E + "\n";
    o += S + "assistant\n";
    o += thinking ? (T + "\n") : (T + "\n\n" + X + "\n\n");
    return o;
}

// The same string as ids, assembled from its parts: encode() is not allowed to grade its own homework,
// so the two paths must agree element by element before either is trusted.
inline std::vector<int> render_ids(const Bpe &tok, const std::string &system,
                                   const std::string &user, bool thinking) {
    std::vector<int> v;
    auto put = [&](int id) { v.push_back(id); };
    auto text = [&](const std::string &s) {
        for (int i : tok.encode(s, false, false)) v.push_back(i);
    };
    if (!system.empty()) { put(kImStart); text("system\n"); text(trim(system));
                           put(kImEnd); text("\n"); }
    put(kImStart); text("user\n"); text(trim(user)); put(kImEnd); text("\n");
    put(kImStart); text("assistant\n");
    if (thinking) { put(kThinkOpen); text("\n"); }
    else          { put(kThinkOpen); text("\n\n"); put(kThinkClose); text("\n\n"); }
    return v;
}

// The four markers have to be SINGLE ids in our vocab, or the whole exercise is a no-op: an
// 8-character split of im_start into normal tokens makes the model see prose, not a delimiter.
inline bool ids_are_singles(const Bpe &tok) {
    const int ids[4] = {kImStart, kImEnd, kThinkOpen, kThinkClose};
    for (int q = 0; q < 4; ++q) {
        const std::string lit = ctl(tok, ids[q]);
        if (lit.empty()) return false;
        std::vector<int> e = tok.encode(lit, true, false);
        if (e.size() != 1 || e[0] != ids[q]) return false;
    }
    return true;
}

}  // namespace chat
