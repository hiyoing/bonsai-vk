// bonsai_http.h - the local HTTP shell: an OpenAI-compatible endpoint and the WebUI, on the OS socket.
//
// WHY THIS FILE IS SMALL. The engine is 539 KiB of C++17 with no third-party dependency, and a GUI
// shell is not a licence to add one. Everything here is the operating system's own surface, reached
// through bonsai_platform.h: Winsock + shell32 + kernel32 on Windows, POSIX sockets + pthread on
// Linux/macOS. No Qt, no ImGui, no cpp-httplib, no JSON library, no thread library - <thread> and
// <mutex> are deliberately absent because they depend on the compiler's thread model, and
// CreateThread/pthread_create do not. The JSON parser below is 200 lines because the request body is
// the only structured input this program has ever accepted.
//
// PLATFORM STATUS. There is no #ifdef in this file: every OS call goes through a bonsai_* helper, so
// the HTTP semantics are one code path on both stacks. The Windows path is the one that has been run;
// the POSIX path is written against the documented API and is 待实机验证 (see bonsai_platform.h).
//
// WHAT IT DOES NOT DO. It does not touch the forward path. fwd_main brings the engine up exactly as
// it does for `run` - same env knobs, same rotation gate, same cap guard, same printed budget - and
// hands the live Engine/Fwd to serve_loop() through g_serve_hook. Each request then calls the SAME
// fwd_run() the CLI calls, with three additions that are dead by default: a pre-rendered transcript
// (Fwd::pre_ids), a per-token emission hook (g_emit_token) and a RunStats out-param. The validated
// kernels, the dequant path and the per-token timing are character-identical.
//
// CONCURRENCY. One GPU process, one queue, one Fwd. The engine is not thread-safe and never was, so
// generation is serialised by a try-lock that refuses instead of queueing on: a second concurrent
// completion gets 429 and a JSON error, which is what an OpenAI client already knows how to retry.
// Everything that does not touch the GPU - the static file, /v1/models, /v1/status - answers while a
// reply is streaming, because the status badge has to stay live.
//
// EXPOSURE. Binds 127.0.0.1 only, sends no Access-Control-Allow-Origin, and checks Host against the
// loopback names. A page on the open internet therefore cannot read a response from this server
// (no CORS) and cannot reach it through a rebinding hostname either (Host check). The request body
// is still untrusted input: it is length-capped, depth-capped, and its roles are validated.

#pragma once

// The whole OS surface. Already included by bonsai_weights.h earlier in the translation unit (which
// is what guarantees winsock2.h precedes windows.h on Windows); #pragma once makes this a no-op there
// and the real include for anyone who reads this header first.
#include "bonsai_platform.h"

#include <atomic>
#include <ctime>

// ------------------------------------------------------------------ JSON, in one struct
namespace hj {

struct Val;
using Arr = std::vector<Val>;
using Obj = std::vector<std::pair<std::string, Val>>;

struct Val {
    enum T { NUL, BOOL, NUM, STR, ARR, OBJ } t = NUL;
    bool b = false;
    double n = 0.0;
    std::string s;
    Arr a;
    Obj o;
};

inline const Val *get(const Val &o, const char *k) {
    if (o.t != Val::OBJ) return nullptr;
    for (const auto &kv : o.o) if (kv.first == k) return &kv.second;
    return nullptr;
}
inline std::string gs(const Val &o, const char *k, const std::string &d = std::string()) {
    const Val *v = get(o, k);
    return v && v->t == Val::STR ? v->s : d;
}
inline double gd(const Val &o, const char *k, double d) {
    const Val *v = get(o, k);
    if (!v) return d;
    if (v->t == Val::NUM) return v->n;
    if (v->t == Val::BOOL) return v->b ? 1.0 : 0.0;
    return d;
}
inline bool gb(const Val &o, const char *k, bool d) {
    const Val *v = get(o, k);
    if (!v) return d;
    if (v->t == Val::BOOL) return v->b;
    if (v->t == Val::NUM) return v->n != 0.0;
    return d;
}

inline void put_utf8(std::string &o, uint32_t cp) {
    if (cp < 0x80u) o += char(cp);
    else if (cp < 0x800u) { o += char(0xC0u | (cp >> 6)); o += char(0x80u | (cp & 0x3Fu)); }
    else if (cp < 0x10000u) {
        o += char(0xE0u | (cp >> 12)); o += char(0x80u | ((cp >> 6) & 0x3Fu));
        o += char(0x80u | (cp & 0x3Fu));
    } else {
        o += char(0xF0u | (cp >> 18)); o += char(0x80u | ((cp >> 12) & 0x3Fu));
        o += char(0x80u | ((cp >> 6) & 0x3Fu)); o += char(0x80u | (cp & 0x3Fu));
    }
}

struct Parser {
    const std::string &s;
    size_t i = 0;
    bool ok = true;

    explicit Parser(const std::string &src) : s(src) {}
    void ws() { while (i < s.size() && (s[i]==' '||s[i]=='\t'||s[i]=='\n'||s[i]=='\r')) ++i; }
    bool hex4(uint32_t &v) {
        if (i + 4 > s.size()) return false;
        v = 0;
        for (int k = 0; k < 4; ++k) {
            const char c = s[i++];
            v <<= 4;
            if (c >= '0' && c <= '9') v |= uint32_t(c - '0');
            else if (c >= 'a' && c <= 'f') v |= uint32_t(c - 'a' + 10);
            else if (c >= 'A' && c <= 'F') v |= uint32_t(c - 'A' + 10);
            else return false;
        }
        return true;
    }
    std::string str() {
        std::string o;
        if (i >= s.size() || s[i] != '"') { ok = false; return o; }
        ++i;
        while (i < s.size()) {
            const char c = s[i++];
            if (c == '"') return o;
            if (c != '\\') { o += c; continue; }
            if (i >= s.size()) break;
            const char e = s[i++];
            switch (e) {
                case '"': o += '"'; break;
                case '\\': o += '\\'; break;
                case '/': o += '/'; break;
                case 'b': o += '\b'; break;
                case 'f': o += '\f'; break;
                case 'n': o += '\n'; break;
                case 'r': o += '\r'; break;
                case 't': o += '\t'; break;
                case 'u': {
                    uint32_t cp = 0;
                    if (!hex4(cp)) { ok = false; return o; }
                    if (cp >= 0xD800u && cp <= 0xDBFFu && i + 1 < s.size() &&
                        s[i] == '\\' && s[i+1] == 'u') {
                        i += 2;
                        uint32_t lo = 0;
                        if (!hex4(lo)) { ok = false; return o; }
                        if (lo < 0xDC00u || lo > 0xDFFFu) { ok = false; return o; }
                        cp = 0x10000u + ((cp - 0xD800u) << 10) + (lo - 0xDC00u);
                    } else if (cp >= 0xD800u && cp <= 0xDFFFu) { ok = false; return o; }
                    put_utf8(o, cp);
                    break;
                }
                default: ok = false; return o;
            }
        }
        ok = false;
        return o;
    }
    Val parse(int depth) {
        Val v;
        if (depth > 24 || !ok) { ok = false; return v; }
        ws();
        if (i >= s.size()) { ok = false; return v; }
        const char c = s[i];
        if (c == '{') {
            ++i; v.t = Val::OBJ; ws();
            if (i < s.size() && s[i] == '}') { ++i; return v; }
            for (;;) {
                ws();
                std::string k = str();
                if (!ok) return v;
                ws();
                if (i >= s.size() || s[i] != ':') { ok = false; return v; }
                ++i;
                Val v2 = parse(depth + 1);
                if (!ok) return v;
                v.o.emplace_back(k, std::move(v2));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == '}') { ++i; return v; }
                ok = false; return v;
            }
        }
        if (c == '[') {
            ++i; v.t = Val::ARR; ws();
            if (i < s.size() && s[i] == ']') { ++i; return v; }
            for (;;) {
                Val v2 = parse(depth + 1);
                if (!ok) return v;
                v.a.push_back(std::move(v2));
                ws();
                if (i < s.size() && s[i] == ',') { ++i; continue; }
                if (i < s.size() && s[i] == ']') { ++i; return v; }
                ok = false; return v;
            }
        }
        if (c == '"') { v.t = Val::STR; v.s = str(); return v; }
        if (s.compare(i, 4, "true") == 0)  { i += 4; v.t = Val::BOOL; v.b = true;  return v; }
        if (s.compare(i, 5, "false") == 0) { i += 5; v.t = Val::BOOL; v.b = false; return v; }
        if (s.compare(i, 4, "null") == 0)  { i += 4; v.t = Val::NUL; return v; }
        if (c == '-' || (c >= '0' && c <= '9')) {
            const size_t a = i;
            if (s[i] == '-') ++i;
            while (i < s.size() && ((s[i] >= '0' && s[i] <= '9') || s[i] == '.' ||
                   s[i] == 'e' || s[i] == 'E' || s[i] == '+' || s[i] == '-')) ++i;
            v.t = Val::NUM;
            v.n = atof(s.substr(a, i - a).c_str());
            if (i == a || (s[a] == '-' && i == a + 1)) ok = false;
            return v;
        }
        ok = false;
        return v;
    }
};

inline bool parse(const std::string &s, Val &out) {
    Parser p(s);
    out = p.parse(0);
    if (!p.ok) return false;
    p.ws();
    return p.i == s.size();
}

inline std::string esc(const std::string &s) {
    std::string o;
    o.reserve(s.size() + 8);
    for (unsigned char c : s) {
        switch (c) {
            case '"':  o += "\\\""; break;
            case '\\': o += "\\\\"; break;
            case '\n': o += "\\n";  break;
            case '\r': o += "\\r";  break;
            case '\t': o += "\\t";  break;
            case '\b': o += "\\b";  break;
            case '\f': o += "\\f";  break;
            default:
                if (c < 0x20u) { char b[8]; snprintf(b, sizeof b, "\\u%04x", unsigned(c)); o += b; }
                else o += char(c);
        }
    }
    return o;
}
inline std::string num(double v, int prec = 2) {
    char b[64];
    snprintf(b, sizeof b, "%.*f", prec, v);
    return b;
}
inline std::string i64(long long v) {
    char b[32];
    snprintf(b, sizeof b, "%lld", v);
    return b;
}

}  // namespace hj

// ------------------------------------------------------------------ UTF-8 boundary
// A BPE token can end in the middle of a multi-byte codepoint (Chinese answers do this constantly),
// and decode_ids() hands back the raw bytes. Sending half a codepoint as JSON would make the browser
// render U+FFFD and then the next chunk would render the rest as a second one. So the stream holds
// back the incomplete tail until the token that completes it arrives.
static size_t utf8_complete(const std::string &s) {
    size_t n = s.size();
    if (!n) return 0;
    size_t back = 0;
    while (back < 4 && back < n && (uint8_t(s[n - 1 - back]) & 0xC0u) == 0x80u) ++back;
    if (back >= n) return 0;
    const uint8_t lead = uint8_t(s[n - 1 - back]);
    size_t want = 0;
    if (lead < 0x80u) want = 1;
    else if ((lead & 0xE0u) == 0xC0u) want = 2;
    else if ((lead & 0xF0u) == 0xE0u) want = 3;
    else if ((lead & 0xF8u) == 0xF0u) want = 4;
    else return n;                       // not a lead byte: pass it through, the vocab owns it
    return (back + 1 == want) ? n : n - 1 - back;
}

// ------------------------------------------------------------------ the transcript
// chat::render() in bonsai_chat.h is the template's single-user-turn branch and it stays exactly as
// it was - every fidelity log in this repo was produced through it. An OpenAI messages[] array is N
// turns, so the loop below is that same branch iterated: same control ids looked up from the same
// vocab, same |trim over each body, same generation prompt. The id-assembled twin exists for the
// same reason chat::render_ids does: encode() is not allowed to grade its own homework.
struct Msg { std::string role, content; };

static std::string render_transcript(const Bpe &tok, const std::vector<Msg> &m, bool thinking) {
    const std::string S = chat::ctl(tok, chat::kImStart), E = chat::ctl(tok, chat::kImEnd);
    const std::string T = chat::ctl(tok, chat::kThinkOpen), X = chat::ctl(tok, chat::kThinkClose);
    std::string sys;
    for (const auto &q : m)
        if (q.role == "system") { if (!sys.empty()) sys += "\n"; sys += chat::trim(q.content); }
    std::string o;
    if (!sys.empty()) o += S + "system\n" + sys + E + "\n";
    for (const auto &q : m) {
        if (q.role == "system") continue;
        o += S + q.role + "\n" + chat::trim(q.content) + E + "\n";
    }
    o += S + "assistant\n";
    o += thinking ? (T + "\n") : (T + "\n\n" + X + "\n\n");
    return o;
}

static std::vector<int> transcript_ids(const Bpe &tok, const std::vector<Msg> &m, bool thinking) {
    std::vector<int> v;
    auto text = [&](const std::string &s) { for (int i : tok.encode(s, false, false)) v.push_back(i); };
    std::string sys;
    for (const auto &q : m)
        if (q.role == "system") { if (!sys.empty()) sys += "\n"; sys += chat::trim(q.content); }
    if (!sys.empty()) { v.push_back(chat::kImStart); text("system\n"); text(sys);
                        v.push_back(chat::kImEnd); text("\n"); }
    for (const auto &q : m) {
        if (q.role == "system") continue;
        v.push_back(chat::kImStart); text(q.role + "\n"); text(chat::trim(q.content));
        v.push_back(chat::kImEnd); text("\n");
    }
    v.push_back(chat::kImStart); text("assistant\n");
    if (thinking) { v.push_back(chat::kThinkOpen); text("\n"); }
    else { v.push_back(chat::kThinkOpen); text("\n\n"); v.push_back(chat::kThinkClose); text("\n\n"); }
    return v;
}

// ------------------------------------------------------------------ socket plumbing
static bool send_all(bonsai_sock_t s, const char *p, size_t n) {
    size_t done = 0;
    while (done < n) {
        // 1 MiB per call: Winsock's send() takes an int, and a whole index.html or a long transcript
        // must not be the reason that cast overflows.
        const size_t chunk = (n - done > (1u << 20)) ? (1u << 20) : (n - done);
        const long k = bonsai_send(s, p + done, chunk);
        if (k <= 0) return false;
        done += size_t(k);
    }
    return true;
}
static bool send_str(bonsai_sock_t s, const std::string &x) { return send_all(s, x.data(), x.size()); }

struct Request {
    std::string method, target, body;
    std::vector<std::pair<std::string, std::string>> hdr;
    std::string header(const char *k) const {
        for (const auto &kv : hdr) if (kv.first == k) return kv.second;
        return std::string();
    }
};

static std::string lower(std::string s) {
    for (auto &c : s) if (c >= 'A' && c <= 'Z') c = char(c - 'A' + 'a');
    return s;
}

static bool read_request(bonsai_sock_t s, Request &r) {
    std::string buf;
    char tmp[4096];
    size_t he = std::string::npos;
    while (buf.size() < 65536u) {
        const long n = bonsai_recv(s, tmp, sizeof tmp);
        if (n <= 0) return false;
        buf.append(tmp, size_t(n));
        he = buf.find("\r\n\r\n");
        if (he != std::string::npos) break;
    }
    if (he == std::string::npos) return false;
    size_t a = 0;
    bool first = true;
    while (a < he) {
        size_t b = buf.find("\r\n", a);
        if (b == std::string::npos || b > he) b = he;
        const std::string line = buf.substr(a, b - a);
        a = b + 2;
        if (first) {
            first = false;
            const size_t s1 = line.find(' ');
            if (s1 == std::string::npos) return false;
            const size_t s2 = line.find(' ', s1 + 1);
            if (s2 == std::string::npos) return false;
            r.method = line.substr(0, s1);
            r.target = line.substr(s1 + 1, s2 - s1 - 1);
            continue;
        }
        const size_t c = line.find(':');
        if (c == std::string::npos) continue;
        std::string k = lower(line.substr(0, c)), v = line.substr(c + 1);
        while (!v.empty() && (v[0] == ' ' || v[0] == '\t')) v.erase(0, 1);
        while (!v.empty() && (v[v.size()-1] == ' ' || v[v.size()-1] == '\r')) v.erase(v.size()-1);
        r.hdr.emplace_back(k, v);
    }
    const std::string cl = r.header("content-length");
    size_t want = 0;
    if (!cl.empty()) {
        const long long n = atoll(cl.c_str());
        if (n < 0 || n > 4ll << 20) return false;      // 4 MiB of transcript is already a long talk
        want = size_t(n);
    }
    r.body = buf.substr(he + 4);
    if (r.body.size() > want) r.body.resize(want);
    while (r.body.size() < want) {
        const size_t left = want - r.body.size();
        const long n = bonsai_recv(s, tmp, left < sizeof tmp ? left : sizeof tmp);
        if (n <= 0) return false;
        r.body.append(tmp, size_t(n));
    }
    return true;
}

static const char *status_text(int code) {
    switch (code) {
        case 200: return "OK";
        case 204: return "No Content";
        case 400: return "Bad Request";
        case 404: return "Not Found";
        case 405: return "Method Not Allowed";
        case 413: return "Payload Too Large";
        case 429: return "Too Many Requests";
        case 500: return "Internal Server Error";
        case 503: return "Service Unavailable";
        default:  return "Error";
    }
}

static std::string http_head(int code, const char *ctype, size_t len, bool sse) {
    std::string h = "HTTP/1.1 " + hj::i64(code) + " " + status_text(code) + "\r\n";
    h += "Content-Type: "; h += ctype; h += "\r\n";
    if (sse) h += "Cache-Control: no-cache\r\nX-Accel-Buffering: no\r\n";
    else { h += "Content-Length: " + hj::i64((long long)len) + "\r\n"; }
    h += "Connection: close\r\n\r\n";
    return h;
}

static void respond(bonsai_sock_t s, int code, const char *ctype, const std::string &body) {
    send_str(s, http_head(code, ctype, body.size(), false));
    send_str(s, body);
}

static void api_error(bonsai_sock_t s, int code, const char *type, const std::string &msg) {
    const std::string b = "{\"error\":{\"message\":\"" + hj::esc(msg) + "\",\"type\":\"" + type +
                          "\",\"code\":" + hj::i64(code) + "}}";
    respond(s, code, "application/json; charset=utf-8", b);
}

// ------------------------------------------------------------------ server state
struct HttpStats {
    std::atomic<long long> requests{0}, completions{0}, tokens{0};
    std::atomic<long long> last_tps_bits{0};       // double, bit-cast: no lock on the status path
    std::atomic<long long> last_ms_bits{0};
    std::atomic<int> busy{0};
    void set_rate(double tps, double ms) {
        long long tb = 0, mb = 0;
        memcpy(&tb, &tps, 8); memcpy(&mb, &ms, 8);
        last_tps_bits.store(tb); last_ms_bits.store(mb);
    }
    double tps() const { long long b = last_tps_bits.load(); double d = 0; memcpy(&d, &b, 8); return d; }
    double ms() const  { long long b = last_ms_bits.load();  double d = 0; memcpy(&d, &b, 8); return d; }
};

struct Serve {
    Engine *e = nullptr;
    Fwd *f = nullptr;
    Bpe tok;
    std::string model, device, index_path;
    uint64_t exe_bytes = 0;
    uint16_t port = 8080;
    bool open_browser = true;
    bonsai_sock_t ls = BONSAI_BAD_SOCK;
    bonsai_mutex gpu;                  // CRITICAL_SECTION / pthread_mutex_t: the one-GPU serialiser
    HttpStats st;
    std::atomic<long long> seq{0};
};

static std::string file_text(const std::string &p) {
    FILE *fp = fopen(p.c_str(), "rb");
    if (!fp) return std::string();
    fseek(fp, 0, SEEK_END);
    const long n = ftell(fp);
    fseek(fp, 0, SEEK_SET);
    if (n < 0) { fclose(fp); return std::string(); }
    std::string s(size_t(n), '\0');
    const size_t got = fread(&s[0], 1, size_t(n), fp);
    fclose(fp);
    s.resize(got);
    return s;
}

// ------------------------------------------------------------------ one completion
struct GenCtx {
    bonsai_sock_t s = BONSAI_BAD_SOCK;
    bool stream = false;
    std::string id, model, pending, content, reasoning;
    bool role_sent = false;
    int last_reasoning = 0;
    long long created = 0;

    bool chunk(const std::string &delta, const char *finish) {
        std::string j = "{\"id\":\"" + id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                        hj::i64(created) + ",\"model\":\"" + hj::esc(model) +
                        "\",\"choices\":[{\"index\":0,\"delta\":" + delta + ",\"finish_reason\":";
        j += finish ? ("\"" + std::string(finish) + "\"") : std::string("null");
        j += "}]}";
        return send_str(s, "data: " + j + "\n\n");
    }
};

static bool emit_cb(void *p, const char *utf8, int reasoning) {
    GenCtx &g = *static_cast<GenCtx *>(p);
    g.last_reasoning = reasoning;
    g.pending += utf8;
    const size_t cut = utf8_complete(g.pending);
    if (!cut) return true;                       // a whole token of continuation bytes: keep waiting
    const std::string piece = g.pending.substr(0, cut);
    g.pending.erase(0, cut);
    if (reasoning) g.reasoning += piece; else g.content += piece;
    if (!g.stream) return true;
    if (!g.role_sent) {
        g.role_sent = true;
        if (!g.chunk("{\"role\":\"assistant\"}", nullptr)) return false;
    }
    const std::string key = reasoning ? "reasoning_content" : "content";
    return g.chunk("{\"" + key + "\":\"" + hj::esc(piece) + "\"}", nullptr);
}

static void run_completion(Serve &sv, bonsai_sock_t s, const hj::Val &req) {
    sv.st.requests.fetch_add(1);
    std::vector<Msg> msgs;
    const hj::Val *mv = hj::get(req, "messages");
    if (!mv || mv->t != hj::Val::ARR || mv->a.empty()) {
        api_error(s, 400, "invalid_request_error", "messages must be a non-empty array");
        return;
    }
    for (const auto &m : mv->a) {
        std::string role = hj::gs(m, "role");
        if (role == "developer") role = "system";
        if (role != "system" && role != "user" && role != "assistant") {
            api_error(s, 400, "invalid_request_error",
                      "unsupported role '" + role + "': this template's no-tools path has system, user and assistant");
            return;
        }
        const hj::Val *c = hj::get(m, "content");
        std::string text;
        if (c && c->t == hj::Val::STR) text = c->s;
        else if (c && c->t == hj::Val::ARR) {
            for (const auto &part : c->a)
                if (hj::gs(part, "type") == "text") text += hj::gs(part, "text");
        } else if (c && c->t != hj::Val::NUL) {
            api_error(s, 400, "invalid_request_error", "content must be a string or an array of text parts");
            return;
        }
        msgs.push_back({role, text});
    }

    const bool stream = hj::gb(req, "stream", false);
    int n_max = int(hj::gd(req, "max_tokens", 1024.0));
    if (n_max < 1) n_max = 1;
    if (n_max > 65536) n_max = 65536;
    const std::string model = hj::gs(req, "model", sv.model);

    // The GPU is one queue and one Fwd. Refuse rather than queue: a waiting socket would hold a
    // thread and a browser tab open behind a reply that may take a minute, and 429 is a status every
    // OpenAI client already retries.
    if (!bonsai_mutex_try(sv.gpu)) {
        api_error(s, 429, "rate_limit_error",
                  "the engine is generating another reply; this build runs one conversation at a time");
        return;
    }
    sv.st.busy.store(1);

    GenCtx g;
    g.s = s; g.stream = stream; g.model = model;
    g.created = (long long)time(nullptr);
    g.id = "chatcmpl-bonsai-" + hj::i64(sv.seq.fetch_add(1) + 1);

    const bool thinking = sv.f->thinking != 0u;
    const std::string rendered = render_transcript(sv.tok, msgs, thinking);
    std::vector<int> ids = sv.tok.encode(rendered, true, false);
    const std::vector<int> assembled = transcript_ids(sv.tok, msgs, thinking);
    size_t d = 0;
    while (d < ids.size() && d < assembled.size() && ids[d] == assembled[d]) ++d;
    const bool same = (d == ids.size() && d == assembled.size());
    printf("\n[serve] %s %s | %zu messages -> %zu prompt tokens, id paths %s\n",
           g.id.c_str(), stream ? "stream" : "unary", msgs.size(), ids.size(),
           same ? "IDENTICAL" : "DIVERGE");
    if (!same) {
        // The CLI prints DIVERGE and continues because a human is watching the log. An API cannot
        // hand back text whose prompt two paths disagree about.
        bonsai_mutex_leave(sv.gpu);
        sv.st.busy.store(0);
        api_error(s, 500, "server_error", "the rendered transcript and its id-assembled twin diverge");
        return;
    }
    if (ids.size() > size_t(sv.f->cap)) {
        bonsai_mutex_leave(sv.gpu);
        sv.st.busy.store(0);
        api_error(s, 400, "invalid_request_error",
                  "the conversation renders to " + hj::i64((long long)ids.size()) +
                  " tokens but the KV arena has " + hj::i64(sv.f->cap) +
                  " slots; start a new conversation or raise BONSAI_CAP");
        return;
    }

    fwd_reset_state(*sv.f);
    sv.f->pre_ids = ids;
    g_req = ReqOverride();
    const double temp = hj::gd(req, "temperature", -1.0);
    const double topp = hj::gd(req, "top_p", -1.0);
    const double topk = hj::gd(req, "top_k", -1.0);
    const double seed = hj::gd(req, "seed", 0.0);
    const double rep  = hj::gd(req, "repetition_penalty", -1.0);
    const double pres = hj::gd(req, "presence_penalty", -1.0);
    const double freq = hj::gd(req, "frequency_penalty", -1.0);
    if (temp >= 0.0) g_req.temp = float(temp);
    if (topp >= 0.0) g_req.top_p = float(topp);
    if (topk >= 0.0) g_req.top_k = int(topk);
    if (seed > 0.0)  g_req.seed = uint64_t(seed);
    if (rep  >= 0.0) g_req.rep_penalty = float(rep);
    if (pres >= 0.0) g_req.presence = float(pres);
    if (freq >= 0.0) g_req.freq = float(freq);

    if (stream) {
        if (!send_str(s, http_head(200, "text/event-stream; charset=utf-8", 0, true))) {
            bonsai_mutex_leave(sv.gpu); sv.st.busy.store(0); return;
        }
    }
    g_emit_ctx = &g;
    g_emit_token = &emit_cb;
    RunStats rs;
    const int rc = fwd_run(*sv.e, *sv.f, "", n_max, &rs);
    g_emit_token = nullptr;
    g_emit_ctx = nullptr;
    sv.f->pre_ids.clear();
    g_req = ReqOverride();
    bonsai_mutex_leave(sv.gpu);
    sv.st.busy.store(0);

    if (rc) {
        if (!stream) api_error(s, 500, "server_error", "the forward pass returned " + hj::i64(rc));
        else { g.chunk("{}", "stop"); send_str(s, "data: [DONE]\n\n"); }
        return;
    }
    // Whatever is still held back at the end of the stream was never completed by a later token.
    if (!g.pending.empty()) {
        if (g.last_reasoning) g.reasoning += g.pending; else g.content += g.pending;
        g.pending.clear();
    }
    sv.st.completions.fetch_add(1);
    sv.st.tokens.fetch_add(rs.n_gen);
    sv.st.set_rate(rs.tok_s(), rs.ms_gen);
    printf("[serve] %s done: %d prompt + %d generated (%d reasoning) in %.1f ms -> %.2f tok/s, %s\n",
           g.id.c_str(), rs.n_prompt, rs.n_gen, rs.n_think, rs.ms_gen, rs.tok_s(), rs.finish);

    const std::string usage = "{\"prompt_tokens\":" + hj::i64(rs.n_prompt) +
                              ",\"completion_tokens\":" + hj::i64(rs.n_gen) +
                              ",\"total_tokens\":" + hj::i64(rs.n_prompt + rs.n_gen) + "}";
    if (stream) {
        if (!g.role_sent) g.chunk("{\"role\":\"assistant\"}", nullptr);
        std::string fin = "{\"id\":\"" + g.id + "\",\"object\":\"chat.completion.chunk\",\"created\":" +
                          hj::i64(g.created) + ",\"model\":\"" + hj::esc(g.model) +
                          "\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"" +
                          rs.finish + "\"}],\"usage\":" + usage + "}";
        send_str(s, "data: " + fin + "\n\n");
        send_str(s, "data: [DONE]\n\n");
        return;
    }
    std::string msg = "{\"role\":\"assistant\",\"content\":\"" + hj::esc(g.content) + "\"";
    if (!g.reasoning.empty()) msg += ",\"reasoning_content\":\"" + hj::esc(g.reasoning) + "\"";
    msg += "}";
    const std::string out = "{\"id\":\"" + g.id + "\",\"object\":\"chat.completion\",\"created\":" +
                            hj::i64(g.created) + ",\"model\":\"" + hj::esc(g.model) +
                            "\",\"choices\":[{\"index\":0,\"message\":" + msg +
                            ",\"finish_reason\":\"" + rs.finish + "\"}],\"usage\":" + usage + "}";
    respond(s, 200, "application/json; charset=utf-8", out);
}

// ------------------------------------------------------------------ the read-only routes
static std::string status_json(Serve &sv) {
    const double tps = sv.st.tps();
    std::string j = "{\"engine\":\"Project Bonsai\",\"api\":\"openai-compatible\",\"device\":\"" +
        hj::esc(sv.device) + "\",\"model\":\"" + hj::esc(sv.model) + "\",\"engine_bytes\":" +
        hj::i64((long long)sv.exe_bytes) + ",\"engine_kib\":" + hj::i64((long long)(sv.exe_bytes / 1024u)) +
        ",\"cap\":" + hj::i64(sv.f->cap) + ",\"kv_kib_per_token\":" +
        hj::num(double(sv.f->kvgeom.bytes_slot) / 1024.0, 2) + ",\"v_quant\":\"" +
        (sv.f->v_q8 ? "q8_0" : "fp16") + "\",\"thinking\":" + (sv.f->thinking ? "true" : "false") +
        ",\"weights_mib\":" + hj::num(double(sv.e->arena.size) / 1048576.0, 1) +
        ",\"activations_mib\":" + hj::num(double(sv.f->bytes) / 1048576.0, 1) +
        ",\"requests\":" + hj::i64(sv.st.requests.load()) +
        ",\"completions\":" + hj::i64(sv.st.completions.load()) +
        ",\"tokens_generated\":" + hj::i64(sv.st.tokens.load()) +
        ",\"busy\":" + (sv.st.busy.load() ? "true" : "false") +
        ",\"last_tok_s\":" + (tps > 0.0 ? hj::num(tps, 2) : std::string("null")) +
        ",\"last_ms\":" + (sv.st.ms() > 0.0 ? hj::num(sv.st.ms(), 1) : std::string("null")) +
        // The badge shows a measured number before the first reply exists. It is the repo's own
        // short-context regression, named here so the UI can cite it instead of implying it was
        // measured in this session; the first completion replaces it with this process's rate.
        ",\"reference_tok_s\":14.16,\"reference_source\":\"runs/regression_v14.log:95\"" +
        std::string("}");
    return j;
}

static void handle(Serve &sv, bonsai_sock_t s) {
    Request r;
    if (!read_request(s, r)) { bonsai_sock_close(s); return; }
    // DNS-rebinding guard: a hostname that resolves to 127.0.0.1 still carries its own Host header.
    std::string host = lower(r.header("host"));
    const size_t colon = host.find(':');
    if (colon != std::string::npos) host = host.substr(0, colon);
    if (!host.empty() && host != "localhost" && host != "127.0.0.1" && host != "[::1]") {
        api_error(s, 400, "invalid_request_error", "this server only answers loopback Host names");
        bonsai_sock_close(s); return;
    }
    std::string path = r.target;
    const size_t q = path.find('?');
    if (q != std::string::npos) path = path.substr(0, q);

    if (r.method == "GET" && (path == "/" || path == "/index.html")) {
        const std::string html = file_text(sv.index_path);
        if (html.empty())
            respond(s, 500, "text/plain; charset=utf-8",
                    "index.html was not found next to the executable or in the working directory.\n"
                    "The release package ships it in the same folder as bonsai_vk.exe.\n");
        else respond(s, 200, "text/html; charset=utf-8", html);
    } else if (r.method == "GET" && path == "/favicon.ico") {
        send_str(s, "HTTP/1.1 204 No Content\r\nConnection: close\r\n\r\n");
    } else if (r.method == "GET" && path == "/v1/models") {
        const std::string j = "{\"object\":\"list\",\"data\":[{\"id\":\"" + hj::esc(sv.model) +
            "\",\"object\":\"model\",\"created\":" + hj::i64((long long)time(nullptr)) +
            ",\"owned_by\":\"project-bonsai\"}]}";
        respond(s, 200, "application/json; charset=utf-8", j);
    } else if (r.method == "GET" && (path == "/v1/status" || path == "/health")) {
        respond(s, 200, "application/json; charset=utf-8", status_json(sv));
    } else if (path == "/v1/chat/completions") {
        if (r.method != "POST") {
            api_error(s, 405, "invalid_request_error", "use POST for /v1/chat/completions");
        } else {
            hj::Val req;
            if (!hj::parse(r.body, req) || req.t != hj::Val::OBJ)
                api_error(s, 400, "invalid_request_error", "the body is not a JSON object");
            else run_completion(sv, s, req);
        }
    } else if (r.method == "OPTIONS") {
        // No CORS headers on purpose: a cross-origin page must not be able to drive this engine.
        api_error(s, 405, "invalid_request_error", "cross-origin requests are not accepted");
    } else {
        api_error(s, 404, "invalid_request_error", "unknown path " + path);
    }
    bonsai_sock_close(s);
}

struct Conn { Serve *sv; bonsai_sock_t s; std::atomic<int> *live; };

// The entry point has one signature on both platforms because bonsai_thread_spawn supplies the
// trampoline: DWORD WINAPI on Windows, void *(*)(void *) on POSIX. The Conn is owned by this thread
// and so is the connection slot, which is released before the delete - a status request arriving in
// that window must not see a slot that is already free.
static void conn_thread(void *p) {
    Conn *c = static_cast<Conn *>(p);
    handle(*c->sv, c->s);
    c->live->fetch_sub(1);
    delete c;
}

// ------------------------------------------------------------------ entry
static int serve_loop(Engine &e, Fwd &f) {
    Serve sv;
    sv.e = &e; sv.f = &f;
    if (const char *v = getenv("BONSAI_PORT")) sv.port = uint16_t(atoi(v));
    if (const char *v = getenv("BONSAI_NO_BROWSER")) sv.open_browser = atoi(v) == 0;
    if (!sv.tok.load(e.hdr)) { fprintf(stderr, "tokenizer load failed\n"); return 5; }
    if (!chat::ids_are_singles(sv.tok)) {
        fprintf(stderr, "serve: a chat control id does not round-trip as one token\n"); return 6;
    }
    {
        const std::string p = e.path;
        const size_t k = p.find_last_of("\\/");
        std::string stem = k == std::string::npos ? p : p.substr(k + 1);
        const size_t dot = stem.rfind('.');
        if (dot != std::string::npos) stem = stem.substr(0, dot);
        sv.model = stem.empty() ? "project-bonsai" : stem;
    }
    {
        VkPhysicalDeviceProperties pp{};
        vkGetPhysicalDeviceProperties(e.c.phys, &pp);
        sv.device = pp.deviceName;
    }
    {
        const std::string exe = bonsai_exe_path();
        const std::string d = bonsai_dir_of(exe);
        if (!exe.empty()) {
            FILE *fp = fopen(exe.c_str(), "rb");
            if (fp) { fseek(fp, 0, SEEK_END); sv.exe_bytes = uint64_t(ftell(fp)); fclose(fp); }
        }
        sv.index_path = "index.html";
        FILE *fp = fopen(sv.index_path.c_str(), "rb");
        if (fp) fclose(fp);
        else if (!d.empty()) sv.index_path = d + "/index.html";
    }

    bonsai_mutex_init(sv.gpu);
    g_watch_persist = 1;                 // one spill query for the life of the process, not per reply

    if (!bonsai_net_init()) { fprintf(stderr, "socket layer init failed\n"); return 9; }
    int lerr = 0;
    // Loopback only, on every platform: this is a local shell for the person at the machine, never a
    // service to expose. bonsai_listen closes the socket itself on failure, so there is nothing to
    // clean up here beyond the socket layer.
    sv.ls = bonsai_listen(sv.port, 8, lerr);
    if (bonsai_sock_bad(sv.ls)) {
        fprintf(stderr, "bind/listen 127.0.0.1:%u failed (OS error %d) - another Bonsai is probably"
                        " already serving; set BONSAI_PORT to use a different port\n",
                unsigned(sv.port), lerr);
        bonsai_net_cleanup(); return 9;
    }

    printf("\n=== Bonsai HTTP shell ===\n");
    printf("  listening   http://localhost:%u  (127.0.0.1 only, no CORS)\n", unsigned(sv.port));
    printf("  model       %s\n", sv.model.c_str());
    printf("  device      %s\n", sv.device.c_str());
    printf("  engine      %llu B = %.0f KiB (this executable, measured now)\n",
           (unsigned long long)sv.exe_bytes, double(sv.exe_bytes) / 1024.0);
    printf("  WebUI       %s\n", sv.index_path.c_str());
    printf("  context     cap %u, KV %.2f KiB/token, V = %s, thinking %s\n",
           f.cap, double(f.kvgeom.bytes_slot) / 1024.0, f.v_q8 ? "q8_0" : "fp16",
           f.thinking ? "on" : "off");
    printf("  endpoints   GET /  |  GET /v1/models  |  GET /v1/status  |  POST /v1/chat/completions\n");
    printf("  one reply at a time: a second concurrent completion gets HTTP 429, the status route"
           " stays live while the GPU is busy\n");
    fflush(stdout);

    if (sv.open_browser) {
        const std::string url = "http://localhost:" + hj::i64(sv.port);
        if (!bonsai_open_url(url))
            printf("  could not open a browser - open %s yourself\n", url.c_str());
        else printf("  opened %s in the default browser\n", url.c_str());
    }

    std::atomic<int> live{0};
    for (;;) {
        bonsai_sock_t s = accept(sv.ls, nullptr, nullptr);
        if (bonsai_sock_bad(s)) {
            const int err = bonsai_sock_err();
            if (bonsai_accept_retry(err)) continue;
            fprintf(stderr, "accept failed (OS error %d)\n", err);
            break;
        }
        if (live.load() >= 16) {
            api_error(s, 503, "server_error", "too many open connections");
            bonsai_sock_close(s);
            continue;
        }
        // 30 s on both directions: a browser that stops reading an SSE stream must not park a thread
        // and a connection slot forever, and a request that never finishes its headers must not either.
        bonsai_sock_timeout(s, 30000);
        Conn *c = new Conn{&sv, s, &live};
        live.fetch_add(1);
        if (!bonsai_thread_spawn(conn_thread, c)) {
            delete c; bonsai_sock_close(s); live.fetch_sub(1); continue;
        }
    }
    bonsai_sock_close(sv.ls);
    bonsai_net_cleanup();
    bonsai_mutex_destroy(sv.gpu);
    return 0;
}