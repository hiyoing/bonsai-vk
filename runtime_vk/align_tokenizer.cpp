// align_tokenizer.cpp - test 1 of the faithful-reproduction programme: our byte-level BPE must
// produce EXACTLY the same token id sequence as the reference tokenizer in the PrismML build.
//
//   g++ -std=c++17 -O2 align_tokenizer.cpp -o align_tokenizer.exe
//   align_tokenizer.exe <model.gguf> <probes.txt> <llama-tokenize.exe>
//
// probes.txt is UTF-8, one probe per line. Each probe is written out as raw bytes to a temp file
// (so no shell/argv encoding can touch it), fed to the oracle with --ids --no-bos, and the printed
// list is parsed back. Our side calls Bpe::encode on the same bytes.
//
// The oracle is the reference implementation's own tokenizer binary - nothing from its source is
// compiled in here. bonsai_tok.h is 100% our code.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bonsai_gguf.h"
#include "bonsai_tok.h"

static std::string slurp(const char* p) {
    FILE* f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s\n", p); exit(1); }
    std::string s;
    char buf[65536]; size_t n;
    while ((n = fread(buf, 1, sizeof buf, f)) > 0) s.append(buf, n);
    fclose(f);
    return s;
}

// the reference id lists come from runs/tok_ref.jsonl, produced by tools/oracle_tokenize.py, which
// runs llama-tokenize.exe from the reference build over the same bytes. Parsing is deliberately
// dumb: each line is one JSON object, we only need "i" and the integer array after "ids".
static std::map<int, std::vector<int>> load_ref(const std::string& path) {
    std::map<int, std::vector<int>> out;
    std::string all = slurp(path.c_str());
    size_t i = 0;
    while (i < all.size()) {
        size_t nl = all.find('\n', i);
        std::string line = all.substr(i, nl == std::string::npos ? std::string::npos : nl - i);
        i = (nl == std::string::npos) ? all.size() : nl + 1;
        if (line.empty()) continue;
        size_t ip = line.find("\"i\":");
        size_t kp = line.find("\"ids\"");
        if (ip == std::string::npos || kp == std::string::npos) continue;
        int idx = (int)strtol(line.c_str() + ip + 4, nullptr, 10);
        size_t lb = line.find('[', kp), rb = line.find(']', lb);
        std::vector<int> ids;
        // the array text has to live in a named object: taking .c_str() of the substr() temporary
        // left `c` pointing into freed memory, the scan loop read whatever the allocator had put
        // there, and strtol then pushed garbage - which is how this tool produced a std::bad_alloc
        // on some runs and a plausible answer on others.
        std::string arr = line.substr(lb + 1, rb - lb - 1);
        const char *c = arr.c_str();
        while (*c) { while (*c == ' ' || *c == ',') ++c; if (!*c) break; ids.push_back((int)strtol(c, (char**)&c, 10)); }
        out[idx] = ids;
    }
    return out;
}

static std::string fmt(const std::vector<int>& v) {
    std::string s = "[";
    for (size_t i = 0; i < v.size(); ++i) { s += std::to_string(v[i]); if (i + 1 < v.size()) s += ", "; }
    s += "]";
    return s;
}

static std::string unescape(const std::string& s) {
    std::string o;
    for (size_t i = 0; i < s.size(); ++i) {
        if (s[i] == '\\' && i + 1 < s.size()) {
            char c = s[++i];
            if (c == 'n') { o += '\n'; continue; }
            if (c == 't') { o += '\t'; continue; }
            if (c == '\\') { o += '\\'; continue; }
        }
        o += s[i];
    }
    return o;
}

int main(int argc, char** argv) {
    if (argc < 4) {
        fprintf(stderr, "usage: %s <model.gguf> <probes.txt> <tok_ref.jsonl>\n", argv[0]);
        return 2;
    }
    // line-buffered: when something goes wrong the lines already produced must survive a redirect
    // to a file, otherwise the tail of the log points at the wrong probe.
    setvbuf(stdout, nullptr, _IOLBF, 0);
    std::string model = argv[1], probes = argv[2];

    GgufHeader h;
    if (!gguf_load(model.c_str(), h)) return 1;
    printf("[hdr] GGUF v%llu  tensors=%llu  data_start=%llu  vocab=%zu  merges=%zu\n",
           (unsigned long long)h.version, (unsigned long long)h.n_tensors,
           (unsigned long long)h.data_start, h.kv_a["tokenizer.ggml.tokens"].size(),
           h.kv_a["tokenizer.ggml.merges"].size());
    printf("[hdr] model=%s  pre=%s  add_bos=%lld  bos=%lld  eos=%lld\n",
           h.kv_s["tokenizer.ggml.model"].c_str(), h.kv_s["tokenizer.ggml.pre"].c_str(),
           (long long)h.kv_i["tokenizer.ggml.add_bos_token"],
           (long long)h.kv_i["tokenizer.ggml.bos_token_id"],
           (long long)h.kv_i["tokenizer.ggml.eos_token_id"]);

    Bpe bpe;
    if (!bpe.load(h)) return 1;
    printf("[bpe] vocab=%zu  ranks=%zu  specials=%zu  unk=%d  eos=%d\n",
           bpe.n_vocab, bpe.rank.size(), bpe.specials.size(), bpe.unk, bpe.eos);

    std::string all = slurp(probes.c_str());
    std::vector<std::string> lines;
    size_t i = 0;
    while (i <= all.size()) {
        size_t nl = all.find('\n', i);
        std::string ln = all.substr(i, nl == std::string::npos ? std::string::npos : nl - i);
        if (!ln.empty() && ln[ln.size() - 1] == '\r') ln.erase(ln.size() - 1);
        if (!ln.empty()) lines.push_back(unescape(ln));
        if (nl == std::string::npos) break;
        i = nl + 1;
    }

    int pass = 0, seen = 0;
    auto refs = load_ref(argv[3]);
    printf("[ref] %zu reference probes loaded from %s\n", refs.size(), argv[3]);
    for (size_t pi = 0; pi < lines.size(); ++pi) {
        const std::string& probe = lines[pi];
        fprintf(stderr, "[probe %zu/%zu] %zu bytes\n", pi + 1, lines.size(), probe.size());
        std::vector<int> ours = bpe.encode(probe, true, false);
        auto rit = refs.find((int)pi + 1);
        if (rit == refs.end()) { printf("\nprobe %2zu  no reference line - skipped\n", pi + 1); continue; }
        std::vector<int> ref = rit->second;
        ++seen;
        bool same = ours == ref;
        if (same) ++pass;
        printf("\nprobe %2zu  bytes=%zu  text=%s\n", pi + 1, probe.size(),
               probe.substr(0, 70).c_str());
        printf("  OURS (%2zu): %s\n", ours.size(), fmt(ours).c_str());
        printf("  REF  (%2zu): %s\n", ref.size(), fmt(ref).c_str());
        if (!same) {
            size_t k = 0;
            while (k < ours.size() && k < ref.size() && ours[k] == ref[k]) ++k;
            printf("  FIRST DIFF at index %zu: ours=%s ref=%s\n", k,
                   k < ours.size() ? std::to_string(ours[k]).c_str() : "<none>",
                   k < ref.size() ? std::to_string(ref[k]).c_str() : "<none>");
        } else {
            printf("  MATCH 1:1\n");
        }
    }
    printf("\n==== tokenizer alignment: %d/%d probes byte-identical ====\n", pass, seen);
    return pass == seen && seen > 0 ? 0 : 1;
}
