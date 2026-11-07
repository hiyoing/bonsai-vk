// plan_probe.cpp - host-only rehearsal of the engine's weight layout. No Vulkan, no driver: this
// prints the VRAM budget and the per-token GEMV inventory BEFORE we boot a device, because on a 4 GB
// card the layout decision is the context ceiling and it has to be reviewable as text.
//
//   g++ -std=c++17 -O2 -m64 -Wall plan_probe.cpp -o plan_probe.exe   (no vulkan link needed)
//   plan_probe.exe models/Ternary-Bonsai-2-9B-PTQ1_0.gguf
#include <cstdio>
#include <cstdint>
#include <string>
#include <vector>

#include "bonsai_model.h"

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char *path = argc > 1 ? argv[1] : "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    GgufHeader h;
    if (!gguf_load(path, h)) return 1;
    ModelPlan p;
    if (!model_plan_build(h, p)) return 2;
    model_plan_report(p, 3840);   // measured device-local heap of the RX 550X (runs/build_full_3.txt)

    // ---- the per-token GEMV inventory, built from the tensor table, not from the graph ----
    // 201 is the number the checkpoint itself asserts (prism.hadamard.weight_names has 201
    // entries) and the number STAGE5 measured on the official engine's command graph. Recomputing
    // it here is what lets the forward's op count be checked against the file rather than against
    // somebody's memory of a graph.
    auto name = [&](const std::string &s) { return s; };
    uint64_t gemv = 0, macs = 0, bytes = 0;
    printf("\n=== per-token GEMV inventory (实测, from the tensor table) ===\n");
    for (int b = 0; b < p.n_layer; ++b) {
        char buf[96];
        std::vector<std::string> want;
        if (p.is_attn_layer(b)) {
            snprintf(buf, sizeof buf, "blk.%d.attn_q.weight", b); want.push_back(buf);
            snprintf(buf, sizeof buf, "blk.%d.attn_k.weight", b); want.push_back(buf);
            snprintf(buf, sizeof buf, "blk.%d.attn_v.weight", b); want.push_back(buf);
            snprintf(buf, sizeof buf, "blk.%d.attn_output.weight", b); want.push_back(buf);
        } else {
            snprintf(buf, sizeof buf, "blk.%d.attn_qkv.weight", b); want.push_back(buf);
            snprintf(buf, sizeof buf, "blk.%d.attn_gate.weight", b); want.push_back(buf);
            snprintf(buf, sizeof buf, "blk.%d.ssm_out.weight", b); want.push_back(buf);
        }
        snprintf(buf, sizeof buf, "blk.%d.ffn_gate.weight", b); want.push_back(buf);
        snprintf(buf, sizeof buf, "blk.%d.ffn_up.weight", b); want.push_back(buf);
        snprintf(buf, sizeof buf, "blk.%d.ffn_down.weight", b); want.push_back(buf);
        int cnt = 0;
        for (auto &n : want) {
            const WeightPlan *w = p.find(name(n).c_str());
            if (!w) { fprintf(stderr, "MISSING %s\n", n.c_str()); continue; }
            ++cnt; ++gemv; macs += uint64_t(w->M) * w->K; bytes += w->bytes_codes + w->bytes_scale;
        }
        if (b < 4 || b == p.n_layer - 1)
            printf("  block %-2d (%s): %d GEMV\n", b, p.is_attn_layer(b) ? "attention" : "GDN", cnt);
    }
    {
        const WeightPlan *o = p.find("output.weight");
        if (o) { ++gemv; macs += uint64_t(o->M) * o->K; bytes += o->bytes_codes + o->bytes_scale;
                 printf("  output.weight: M=%u K=%u -> %llu logits\n", o->M, o->K, (unsigned long long)o->M); }
    }
    printf("  TOTAL per token: %llu GEMV, %.3f GMAC, weights read %.3f GiB/token (%.2f bits/weight)\n",
           (unsigned long long)gemv, macs / 1e9, double(bytes) / (1ull << 30),
           8.0 * double(bytes) / double(macs));
    printf("  at the measured 176.3 Gop/s (runs/build_full_3.txt) the projections alone cost %.1f ms "
           "-> %.1f tok/s, and the weight stream is %.1f GB/s\n",
           macs / 1e9 / 176.3 * 1000.0, 1000.0 / (macs / 1e9 / 176.3 * 1000.0),
           double(bytes) / (macs / 1e9 / 176.3) / 1e9);
    printf("  (that tok/s is the PROJECTION ceiling only: glue ops, barriers and the per-token fence "
           "come off it. STAGE5 measured the official engine's real per-token structure at 201 ops, "
           "so this is the same op count we are about to pay for.)\n");

    // ---- what the two candidate layouts do to the context window ----
    const uint64_t MiB = 1048576ull, GiB = 1073741824ull;
    uint64_t kv_tok = 2ull * p.n_head_kv * p.head_dim * 2ull * p.n_attn_layers();
    uint64_t repacked = p.bytes_codes_total + p.bytes_scale_total;
    uint64_t heap = 3840 * MiB;
    uint64_t driver_reserve = 634 * MiB;    // 实测 unaccounted device memory, STAGE5 (607-634 MiB band)
    printf("\n=== context ceiling per layout (推算 from the measured bytes above) ===\n");
    for (int pass = 0; pass < 2; ++pass) {
        uint64_t wbytes = pass ? p.bytes_src_total : repacked;
        uint64_t free_b = heap - wbytes - p.bytes_f32_total - driver_reserve;
        int ngdn = p.n_layer - p.n_attn_layers();
        free_b -= uint64_t(ngdn) * (uint64_t(p.state_size) * p.state_size * (p.inner / p.state_size) * 4ull +
                                    uint64_t(p.conv_kernel - 1) * p.inner * 2ull * 4ull);
        printf("  [%s layout] weights %.3f GiB -> %.0f MiB left for KV -> %llu tokens\n",
               pass ? "native 1.75 bpw block" : "our 2.125 bpw random-access",
               double(wbytes) / GiB, double(free_b) / MiB, (unsigned long long)(free_b / kv_tok));
    }
    printf("  (the second number is what the context ceiling actually costs: our fast layout buys\n"
           "   speed with the same gigabytes that would have been tokens. Decide it with a measurement,\n"
           "   not with a preference - a native-block GEMV kernel does not exist yet.)\n");
    return 0;
}
