// bonsai_fleet.cpp - the task-3 rig for the two capabilities gpu_fleet.h implements.
//
//   bonsai_fleet probe                     topology: every device, every queue family, every
//                                          memory type, and an explicit verdict on (A) multi-card
//                                          layer pipelining and (B) iGPU zero-copy for THIS host.
//                                          Enumerate + classify only - it creates NO logical
//                                          device, so it is the cheapest GPU-touching command here.
//   bonsai_fleet ring [depth] [slot_MiB] [micro]
//                                          the pinned ring, measured. Allocates a depth-D pinned
//                                          ring of slot_MiB slots, runs `micro` producer/consumer
//                                          round trips through the pipeline executor (fill -> D2H
//                                          -> publish -> H2D -> next stage), checks the bytes that
//                                          arrived, then measures each copy leg on its own, fenced
//                                          and batched. Everything that ONE card can exercise is
//                                          exercised; the cross-card PCIe hop is NOT EXERCISABLE
//                                          HERE and the tool says so instead of printing a
//                                          number it did not take.
//   bonsai_fleet split <layers> <cards> [budget_MiB] [--model PATH] [--fallback]
//                                          host-side scheduling arithmetic only - no Vulkan at all.
//                                          Per-card layer list, per-card DEVICE-resident bytes (the
//                                          GGUF tensor table through the engine's own formulas),
//                                          the FILE bytes next to them, and a refusal of any split
//                                          that does not fit the smallest card's budget.
//   bonsai_fleet selftest                  deterministic PASS/FAIL over the offline parts: the
//                                          ring's index algebra, the scheduler's arithmetic and its
//                                          refusals, the zero-copy memory-type decision on
//                                          synthetic tables, and the model's byte sums. Never
//                                          touches the GPU, so it is safe to run while a benchmark
//                                          ladder owns the card.
//
// WHY THIS EXISTS AND NOT SOMETHING ELSE
//   The runtime's numbers were made on one RX 550X with a 3840 MiB device-local heap, no iGPU in
//   the CPU, under a loader that may hand out only 1.1 entry points. Most of what this task is about
//   cannot be run here. The honest response is: build it so it compiles, test its arithmetic on a
//   table the host cannot fake, and measure the one piece that IS reachable - a pinned ring, an
//   async DMA leg on the transfer-only queue, a fence per in-flight stage. `probe` prints which is
//   which and the report quotes `probe`, not this comment.
//
//   NULL-FUNCTION-POINTER RULE: enforced in gpu_fleet.h, where every vkGetInstanceProcAddr /
//   vkGetDeviceProcAddr result is checked before being called. The recorded amdvlk SIGSEGV came from
//   jumping a NULL vkResetQueryPool resolved under a 1.1 instance (STAGE3_RUNTIME §7,
//   END_TO_END_MEASURED §2.3). This file resolves nothing beyond the header's own queries.
//
// build: the bonsai_fleet line in build_mingw.bat, same flags as bonsai_vk.exe.

#include "gpu_fleet.h"

#include "bonsai_gguf.h"
#include "bonsai_model.h"

#include <cctype>
#include <utility>

static const uint64_t MiB = 1048576ull;

// ------------------------------------------------------------------ instance
// Same shape as bonsai_vk.cpp:204-219. 1.1 by default: apiVersion is what decides whether
// vkGetDeviceProcAddr returns NULL for a 1.2 entry point, and the whole fleet design is deliberately
// 1.1-safe, so the default must be 1.1 and any 1.2 ask must be explicit (BONSAI_API12=1).
static VkInstance make_instance(const char *app) {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = getenv("BONSAI_API12") ? VK_API_VERSION_1_2 : VK_API_VERSION_1_1;
    ai.pApplicationName = app;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    const char *val = "VK_LAYER_KHRONOS_validation";
    bool want_val = getenv("VK_VALIDATION") != nullptr;
    if (want_val) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &val; }
    VkInstance inst = VK_NULL_HANDLE;
    VkResult r = vkCreateInstance(&ici, nullptr, &inst);
    if (r != VK_SUCCESS && want_val) {
        printf("validation layer unavailable, continuing without it\n");
        ici.enabledLayerCount = 0;
        r = vkCreateInstance(&ici, nullptr, &inst);
    }
    if (r != VK_SUCCESS) {
        fprintf(stderr, "bonsai_fleet: no Vulkan runtime (%d). `split` and `selftest` need no "
                        "driver and still work.\n", (int)r);
        return VK_NULL_HANDLE;
    }
    return inst;
}

// Which device the rig opens: BONSAI_GPU_INDEX by enumeration index, BONSAI_GPU_TYPE by class,
// default = the first COMPUTE-capable device (which on this host is the same card bonsai_vk.cpp
// picks). When the requested class does not exist the tool REFUSES and prints what it saw: a
// "zero-copy run" that quietly opened the discrete card instead would produce a log that describes
// a machine that is not here.
static bool pick_device(VkInstance inst, std::vector<FleetDevice> &ds, FleetDevice &out) {
    if (fleet_enumerate(inst, ds) != VK_SUCCESS || ds.empty()) {
        fprintf(stderr, "bonsai_fleet: the loader reported no devices\n");
        return false;
    }
    const char *wi = getenv("BONSAI_GPU_INDEX");
    const char *wt = getenv("BONSAI_GPU_TYPE");
    const int idx = wi ? atoi(wi) : -1;
    const std::string want = wt ? wt : std::string("compute");
    for (auto &d : ds) {
        const bool by_index = (idx >= 0) ? (int(d.index) == idx) : true;
        bool by_type = true;
        if (want == "integrated") by_type = d.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU;
        else if (want == "discrete") by_type = d.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU;
        else if (want == "zero-copy") by_type = d.has_zero_copy_type;
        if (by_index && by_type && d.compute_family != 0xFFFFFFFFu) { out = d; return true; }
    }
    fprintf(stderr, "bonsai_fleet: REFUSE - no device matches BONSAI_GPU_INDEX=%s "
                    "BONSAI_GPU_TYPE=%s among the %zu device(s) the loader reported:\n",
            wi ? wi : "-", want.c_str(), ds.size());
    for (auto &d : ds)
        fprintf(stderr, "    [%u] %s = %s, zero-copy type %s, COMPUTE family %s\n", d.index, d.name,
                fleet_type_name(d.type), d.has_zero_copy_type ? "present" : "absent",
                d.compute_family == 0xFFFFFFFFu ? "none" : "present");
    return false;
}

// ------------------------------------------------------------------ byte tables
static bool layer_index(const std::string &name, int &idx) {
    const size_t p = name.find("blk.");
    if (p == std::string::npos) return false;
    size_t s = p + 4, e = s;
    while (e < name.size() && isdigit((unsigned char)name[e])) ++e;
    if (e == s || e >= name.size() || name[e] != '.') return false;
    idx = atoi(name.substr(s, e - s).c_str());
    return true;
}

// The two byte counts a layer has, and confusing them is a 28% error in a budget:
//   DEV  = the repacked form the arena really holds: codes (M*(K/64)*16) + scales (M*(K/128)*4) for
//          a PTQ1_0 tensor, elems*4 for an f32 one, each padded to the 16 B the arena uses. These are
//          the formulas bonsai_model.h:1050-1062 uses to lay the arena out, reused rather than
//          restated. THIS is what a card's budget must cover.
//   FILE = what crosses the link on upload: bytes_src (28 B per 128-weight block, the 1.75 bpw on-disk
//          form) for a quantized tensor. THIS is what the 7.12 GB/s cold-boot model divides.
static uint64_t dev_bytes_of(const WeightPlan &x) {
    if (x.quantized()) return align_up_b(x.bytes_codes, 16) + align_up_b(x.bytes_scale, 16);
    return align_up_b(x.bytes_f32, 16);
}
static uint64_t file_bytes_of(const WeightPlan &x) {
    if (x.quantized()) return align_up_b(x.bytes_src, 16);
    return align_up_b(x.bytes_f32, 16);
}

struct ByteTable {
    int layers = 0;
    std::vector<uint64_t> dev, file;      // per layer
    std::vector<uint8_t> attn;            // per layer, 1 = softmax attention (owns KV)
    uint64_t head_dev = 0, tail_dev = 0;  // embedding / output(+norm), DEVICE form
    uint64_t head_file = 0, tail_file = 0;
    uint64_t other_dev = 0;               // non-layer tensors that are neither head nor tail
    std::string source;
};

static bool bytes_from_model(const char *path, ByteTable &t, std::string &why) {
    GgufHeader h;
    if (!gguf_load(path, h)) { why = "gguf_load failed (cannot open / not GGUF)"; return false; }
    ModelPlan p;
    if (!model_plan_build(h, p)) { why = "model_plan_build rejected the file"; return false; }
    if (p.n_layer <= 0) { why = "block_count is not positive"; return false; }
    t.layers = p.n_layer;
    t.dev.assign(size_t(t.layers), 0);
    t.file.assign(size_t(t.layers), 0);
    t.attn.assign(size_t(t.layers), 0);
    for (int b = 0; b < t.layers; ++b) t.attn[size_t(b)] = p.is_attn_layer(b) ? 1u : 0u;
    bool has_emb = false, has_out = false;
    std::string unlayered;
    for (const auto &x : p.w) {
        int b = -1;
        if (layer_index(x.name, b) && b >= 0 && b < t.layers) {
            t.dev[size_t(b)] += dev_bytes_of(x);
            t.file[size_t(b)] += file_bytes_of(x);
            continue;
        }
        if (x.name.find("token_embd") != std::string::npos) {
            t.head_dev += dev_bytes_of(x); t.head_file += file_bytes_of(x); has_emb = true;
            continue;
        }
        if (x.name.find("output") != std::string::npos) {
            t.tail_dev += dev_bytes_of(x); t.tail_file += file_bytes_of(x); has_out = true;
            continue;
        }
        // Anything that is not inside a layer still occupies a card. Charging it to card 0 keeps the
        // identity `sum of ranges == total_bytes`, which fleet_split_print checks; dropping it would
        // make every budget in the print a little below the truth, and "a little below" is how a
        // split that OOMs gets scheduled.
        t.other_dev += dev_bytes_of(x); t.head_dev += dev_bytes_of(x);
        t.head_file += file_bytes_of(x);
        unlayered += " " + x.name;
    }
    if (!has_emb || !has_out) { why = "no token_embd / no output tensor in the file"; return false; }
    t.source = std::string("GGUF tensor table of ") + path + " - " + std::to_string(p.w.size()) +
               " tensors, " + std::to_string(t.layers) + " layers, hidden " +
               std::to_string(p.hidden) + ", ffn " + std::to_string(p.ffn) +
               ", full_attention_interval " + std::to_string(p.full_attn_interval);
    if (!unlayered.empty()) t.source += "; charged to card 0:" + unlayered;
    return true;
}

// The documented fallback, used ONLY when the file is not readable. DERIVED on 2026-11-03 by reading
// the 9B's tensor table and applying the formulas above:
//   GDN block      = attn_gate/attn_qkv/out_proj + the gdn_* f32 tensors + ffn_gate/up/down
//                  = 62,554,880 B DEVICE form (24 layers)        | 48,923,392 B FILE form
//   softmax block  = the same projections minus the gdn tensors  = 59,017,216 B DEVICE  (8 layers)
//                                                                 | 45,910,016 B FILE
//   24*62,554,880 + 8*59,017,216 = 1,973,454,848 B = 1.838 GiB of layer weights
//   head = token_embd.weight (K 4096, M 248320): 248320*(4096/64)*16 + 248320*(4096/128)*4
//        = 286,064,640 B DEVICE | 222,494,720 B FILE
//   tail = output.weight (same shape, same 286,064,640 B) + output_norm.weight (4096 f32 = 16,384 B)
//        = 286,081,024 B DEVICE | 222,511,104 B FILE
// `selftest` re-derives every one of these from the file when the file is present, so a fallback
// constant that went stale is caught rather than shipped.
static void bytes_fallback(ByteTable &t, int layers) {
    static const uint64_t kGdnDev = 62554880ull, kAttnDev = 59017216ull;
    static const uint64_t kGdnFile = 48923392ull, kAttnFile = 45910016ull;
    static const uint64_t kHeadDev = 286064640ull, kTailDev = 286081024ull;
    static const uint64_t kHeadFile = 222494720ull, kTailFile = 222511104ull;
    t.layers = layers;
    t.dev.assign(size_t(layers), 0); t.file.assign(size_t(layers), 0);
    t.attn.assign(size_t(layers), 0);
    for (int b = 0; b < layers; ++b) {
        // The 9B's interval rule (qwen35.full_attention_interval = 4): block b is softmax attention
        // when (b+1) % 4 == 0 - the same expression ModelPlan::interval_rule() uses, so the fallback
        // disagrees with the file only if the file stops matching its own metadata.
        const bool att = ((b + 1) % 4) == 0;
        t.attn[size_t(b)] = att ? 1u : 0u;
        t.dev[size_t(b)] = att ? kAttnDev : kGdnDev;
        t.file[size_t(b)] = att ? kAttnFile : kGdnFile;
    }
    t.head_dev = kHeadDev; t.tail_dev = kTailDev;
    t.head_file = kHeadFile; t.tail_file = kTailFile;
    t.source = "DERIVED per-layer constants (the GGUF was not read this run): 62,554,880 B per GDN "
               "block, 59,017,216 B per softmax block, head 286,064,640 B, tail 286,081,024 B - "
               "arithmetic in the comment above bytes_fallback()";
}

static void print_byte_table(const ByteTable &t) {
    uint64_t ld = 0, lf = 0;
    for (size_t i = 0; i < t.dev.size(); ++i) { ld += t.dev[i]; lf += t.file[i]; }
    printf("  byte source: %s\n", t.source.c_str());
    // Distinct per-layer sums, not an average: a byte-balanced cut is decided by the asymmetry an
    // average hides, and the card that overflows is the only card that matters.
    std::vector<uint64_t> uniq;
    for (auto v : t.dev) if (std::find(uniq.begin(), uniq.end(), v) == uniq.end()) uniq.push_back(v);
    std::sort(uniq.begin(), uniq.end());
    printf("  per-layer DEVICE bytes:");
    for (auto v : uniq) {
        int n = 0, na = 0;
        for (size_t i = 0; i < t.dev.size(); ++i) if (t.dev[i] == v) { ++n; if (t.attn[i]) ++na; }
        printf("  %llu B x %d layer(s) [%d softmax, %d recurrent]", (unsigned long long)v, n, na,
               n - na);
    }
    printf("\n  layers total  : device %llu B (%.3f GiB) | file %llu B (%.3f GiB)",
           (unsigned long long)ld, double(ld) / 1073741824.0, (unsigned long long)lf,
           double(lf) / 1073741824.0);
    printf("\n  head (embedding%s): device %llu B | file %llu B",
           t.other_dev ? " + unlayered" : "", (unsigned long long)t.head_dev,
           (unsigned long long)t.head_file);
    if (t.other_dev)
        printf("   [unlayered charged to card 0: %llu B]", (unsigned long long)t.other_dev);
    printf("\n  tail (output + output_norm): device %llu B | file %llu B\n",
           (unsigned long long)t.tail_dev, (unsigned long long)t.tail_file);
    printf("  whole model   : device %llu B (%.3f GiB) | file %llu B (%.3f GiB)\n",
           (unsigned long long)(ld + t.head_dev + t.tail_dev),
           double(ld + t.head_dev + t.tail_dev) / 1073741824.0,
           (unsigned long long)(lf + t.head_file + t.tail_file),
           double(lf + t.head_file + t.tail_file) / 1073741824.0);
}

// ------------------------------------------------------------------ split
static int cmd_split(int argc, char **argv) {
    int layers = 32, cards = 2;
    uint64_t budget_mib = FLEET_MEASURED_DEVICE_LOCAL_MIB;
    bool forced_fallback = false;
    const char *model = getenv("BONSAI_MODEL");
    static const char *kDefaultModel = "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    std::vector<char *> pos;
    for (int i = 2; i < argc; ++i) {
        if (!strcmp(argv[i], "--fallback")) { forced_fallback = true; continue; }
        if (!strcmp(argv[i], "--model") && i + 1 < argc) { model = argv[++i]; continue; }
        pos.push_back(argv[i]);
    }
    if (pos.size() >= 1) layers = atoi(pos[0]);
    if (pos.size() >= 2) cards = atoi(pos[1]);
    if (pos.size() >= 3) budget_mib = strtoull(pos[2], nullptr, 10);
    if (pos.size() > 3) {
        printf("usage: bonsai_fleet split <layers> <cards> [budget_MiB] [--model PATH] [--fallback]\n");
        return 2;
    }

    printf("\n== fleet split: %d layers over %d card(s), budget %llu MiB/card ==\n", layers, cards,
           (unsigned long long)budget_mib);
    printf("  budget basis: MEASURED device-local heap of THIS host's 4 GiB card = %llu MiB "
           "(DERIVED driver reservation = 4096 - 3840 = %llu MiB, STAGE3_RUNTIME §7). On another "
           "card that is an ASSUMPTION - pass its real heap as budget_MiB.\n",
           (unsigned long long)FLEET_MEASURED_DEVICE_LOCAL_MIB,
           (unsigned long long)FLEET_DRIVER_RESERVATION_MIB);

    ByteTable t;
    std::string why;
    const char *path = model ? model : kDefaultModel;
    bool got = false;
    if (!forced_fallback) {
        FILE *f = fopen(path, "rb");
        if (f) { fclose(f); got = bytes_from_model(path, t, why); }
        if (!got) printf("  (model bytes not read from %s: %s - using the DERIVED fallback)\n",
                         path, why.empty() ? "file not present" : why.c_str());
    }
    if (!got) bytes_fallback(t, layers);
    if (t.layers != layers) {
        // The file's own block_count beats the CLI number: a schedule of "40" against a 32-layer
        // model is arithmetic about layers that do not exist.
        printf("  NOTE: asked for %d layers, the file has %d - the file's count is used.\n",
               layers, t.layers);
        layers = t.layers;
    }
    print_byte_table(t);

    uint64_t lf = 0;
    for (auto v : t.file) lf += v;
    const double gbps = FLEET_MEASURED_SDMA_GBPS_H2D;
    FleetSplit s = fleet_split_schedule(uint32_t(layers), uint32_t(cards), t.dev, t.head_dev,
                                        t.tail_dev, budget_mib * MiB);
    fleet_split_print(s, got ? "the GGUF tensor table, DEVICE form (codes+scales+f32)"
                             : "the DERIVED per-layer constants");
    if (!s.ok) return 1;

    // What the split does to the one cost the single-card build already pays, and the number the
    // zero-copy branch removes. Weights are uploaded once at start-up, not per token, so this is a
    // cold-boot column - which is exactly why a layer split does not make tokens faster.
    printf("  cold-boot upload per card at the MEASURED %.2f GB/s SDMA H2D:\n", gbps);
    const uint32_t nr = uint32_t(s.ranges.size());
    for (uint32_t k = 0; k < nr; ++k) {
        uint64_t fb = (k == 0 ? t.head_file : 0) + (k + 1 == nr ? t.tail_file : 0);
        for (uint32_t j = s.ranges[k].layer_lo; j < s.ranges[k].layer_hi && j < t.file.size(); ++j)
            fb += t.file[j];
        printf("    card %u: %llu B to upload = %.0f ms of SDMA | %llu B resident = %.1f%% of its "
               "%llu MiB budget\n", k, (unsigned long long)fb,
               double(fb) / (gbps * 1e9) * 1000.0, (unsigned long long)s.ranges[k].bytes,
               100.0 * double(s.ranges[k].bytes) / double(s.ranges[k].budget),
               (unsigned long long)(s.ranges[k].budget / MiB));
    }
    printf("  ZERO-COPY (B) removes the %.0f ms/card above only as an SDMA COPY: on an iGPU the "
           "stores that fill the arena ARE the upload, so the copy and the staging allocation both "
           "go away. It removes none of the per-token cost - 1682 MiB/token stays where it is, which "
           "is why the MEASURED h >= 78.3%% device-residency line is still the judge.\n",
           double(t.head_file + lf + t.tail_file) / (gbps * 1e9) * 1000.0 / double(nr));
    // The state that travels with a split but never crosses the ring: per-sequence GDN state lives
    // on the card that owns the layer (MEASURED 50.25 MiB/sequence for the 32 layers) and KV is
    // 32 KiB/token on the softmax layers only (MEASURED). A budget that counts only weights would
    // pass configurations that cannot run, so the headroom line subtracts them.
    const double gdn_all = 50.25, kv_per_tok_kib = 32.0 / 1024.0;   // MEASURED, MiB
    printf("  state on top of the weights (DERIVED from the MEASURED totals; 1 sequence, 4 K ctx):\n");
    const uint32_t nl = uint32_t(t.dev.size());
    for (const auto &r : s.ranges) {
        uint32_t ng = 0, na = 0;
        for (uint32_t j = r.layer_lo; j < r.layer_hi && j < nl; ++j) {
            if (t.attn[j]) ++na; else ++ng;
        }
        const double g = gdn_all * double(ng) / double(nl ? nl : 1);
        const double kv = kv_per_tok_kib * 4096.0 * double(na) / double(nl ? nl : 1);
        const double free_mib = double(int64_t(r.budget) - int64_t(r.bytes)) / double(MiB) - g - kv;
        printf("    card %u: %u recurrent + %u attention -> GDN state %.2f MiB + KV %.2f MiB = "
               "%.2f MiB on top of %.1f MiB of weights -> headroom %s%.1f MiB\n", r.card, ng, na,
               g, kv, g + kv, double(r.bytes) / double(MiB), free_mib < 0 ? "NEGATIVE " : "",
               free_mib);
        if (free_mib < 0)
            printf("    [!] card %u fits its WEIGHTS and still has no room for its state: the split "
                   "is schedulable and not runnable. Raise the budget or add a card.\n", r.card);
    }
    return 0;
}

// ------------------------------------------------------------------ ring rig
// A VkBuffer carved out of the ctx's arena, created CONCURRENT over {compute, transfer} whenever the
// card has a transfer family - the reason bonsai_vk.cpp:293-299 gives, repeated by the ring: an
// EXCLUSIVE buffer forces a queue-ownership barrier at every hop, which is the cost being measured.
struct ArenaBuf {
    VkBuffer b = VK_NULL_HANDLE;
    uint64_t off = 0, size = 0;
};

static ArenaBuf make_arena_buf(FleetCtx &c, uint64_t off, uint64_t size) {
    ArenaBuf a; a.off = off; a.size = size;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = size;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    // Declared OUTSIDE the if so it is still alive at vkCreateBuffer: pQueueFamilyIndices is read
    // during the call and copied nowhere. With the array inside the block this exact line was the
    // -O2 segfault inside amdvlk64.dll (gdb bt: make_arena_buf bonsai_fleet.cpp:392 -> vulkan-1 ->
    // amdvlk64), while -O0 passed - the classic signature of a dead stack frame being read.
    uint32_t fams[2] = {c.qfi, c.tfi};
    if (c.has_tx) {
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fams;
    } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    FK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &a.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, a.b, &mr);
    if (!((mr.memoryTypeBits >> c.arena_type) & 1u)) {
        fprintf(stderr, "bonsai_fleet ring REFUSE: the arena memory type %u cannot back a "
                        "TRANSFER-capable buffer (its memoryTypeBits are 0x%X). Continuing would "
                        "print a hop that cannot be built.\n", c.arena_type, mr.memoryTypeBits);
        exit(1);
    }
    if (off % mr.alignment) {
        fprintf(stderr, "bonsai_fleet ring REFUSE: arena offset %llu is not a multiple of this "
                        "buffer's alignment %llu.\n", (unsigned long long)off,
                (unsigned long long)mr.alignment);
        exit(1);
    }
    FK_CHECK(vkBindBufferMemory(c.dev, a.b, c.arena, off));
    return a;
}

static void destroy_arena_bufs(FleetCtx &c, std::vector<ArenaBuf> &v) {
    for (auto &a : v) if (a.b) vkDestroyBuffer(c.dev, a.b, nullptr);
    v.clear();
}

// The rig's stand-in for "this range's dispatches": fill the window with a word that encodes the
// sequence number, so a slot that arrives with the wrong content is a visible bug and not a
// statistical one. vkCmdFillBuffer wants a size that is a multiple of 4; slot_bytes is whole MiB.
static uint32_t ring_pattern(uint64_t seq) { return 0xB05A0000u | uint32_t(seq & 0xFFFFu); }

static int cmd_ring(int argc, char **argv) {
    const auto cmd_t0 = std::chrono::steady_clock::now();
    uint32_t depth = 4, micro = 4;
    uint64_t slot_mib = 1;
    std::vector<char *> pos;
    for (int i = 2; i < argc; ++i) pos.push_back(argv[i]);
    if (pos.size() >= 1) depth = uint32_t(atoi(pos[0]));
    if (pos.size() >= 2) slot_mib = strtoull(pos[1], nullptr, 10);
    if (pos.size() >= 3) micro = uint32_t(atoi(pos[2]));
    if (pos.size() > 3) {
        printf("usage: bonsai_fleet ring [depth] [slot_MiB] [microbatches]\n");
        return 2;
    }
    if (slot_mib == 0) { fprintf(stderr, "bonsai_fleet ring: slot_MiB must be >= 1\n"); return 2; }
    const uint64_t slot_bytes = slot_mib * MiB;

    VkInstance inst = make_instance("bonsai_fleet_ring");
    if (!inst) return 1;
    std::vector<FleetDevice> ds;
    FleetDevice d;
    if (!pick_device(inst, ds, d)) { vkDestroyInstance(inst, nullptr); return 1; }
    printf("\n== fleet ring rig: depth %u x %llu MiB slots, %u microbatch(es) ==\n", depth,
           (unsigned long long)slot_mib, micro);
    printf("  card [%u] %s (%s) | compute family %u | transfer-only family %s\n", d.index, d.name,
           fleet_type_name(d.type), d.compute_family,
           d.transfer_only_family == 0xFFFFFFFFu ? "NONE"
                                                 : std::to_string(d.transfer_only_family).c_str());
    if (d.transfer_only_family == 0xFFFFFFFFu)
        printf("  NO transfer-only family: the hop's copy queues behind compute. MEASURED cost of "
               "that on this driver is D2H 7.15 -> 3.89 GB/s (DEEPINFER_EVAL §0), so every number "
               "below is a serialised ring, not a pipelined one.\n");

    // One ctx per card. std::vector rather than a bare FleetCtx because FleetPipeline holds a
    // vector<FleetCtx>*, and a pipeline whose ctx array was copied into existence would be a
    // pipeline of dangling handles the moment the temporary died.
    std::vector<FleetCtx> cards(1);
    FleetCtx &c = cards[0];
    if (!fleet_ctx_init(c, inst, d, false)) { vkDestroyInstance(inst, nullptr); return 1; }
    // Three windows: the producer's output, the consumer's input, and range 1's own output. They
    // stand in for "two cards' sub-arenas" on a one-card machine, which is what keeps the hop a
    // real device -> host -> device round trip instead of an alias.
    if (!fleet_arena_alloc(c, fleet_align_up(slot_bytes * 3, 256), FLEET_ARENA_DEVICE_LOCAL,
                           VK_BUFFER_USAGE_TRANSFER_SRC_BIT)) {
        fleet_ctx_release(c); vkDestroyInstance(inst, nullptr); return 1;
    }
    std::vector<ArenaBuf> bufs;
    bufs.push_back(make_arena_buf(c, 0, slot_bytes));                              // PROD
    bufs.push_back(make_arena_buf(c, fleet_align_up(slot_bytes, 256), slot_bytes));      // CONS
    bufs.push_back(make_arena_buf(c, fleet_align_up(slot_bytes * 2, 256), slot_bytes));  // WORK

    // The schedule the rig runs: two ranges, both on this one card, so the hop path is exercised
    // end to end. This is NOT a claim that one card is two cards - see the line printed below.
    FleetSplit split;
    split.layers = 2; split.cards = 1;
    split.total_bytes = 0; split.smallest_budget = c.arena_bytes;
    for (uint32_t r = 0; r < 2; ++r) {
        LayerRange lr; lr.card = 0; lr.layer_lo = r; lr.layer_hi = r + 1;
        lr.bytes = slot_bytes; lr.budget = c.arena_bytes;
        split.ranges.push_back(lr);
    }
    split.ok = true;

    FleetPipeline p;
    if (!fleet_pipeline_init(p, cards, split, depth, slot_bytes)) {
        fprintf(stderr, "bonsai_fleet ring: fleet_pipeline_init failed\n");
        destroy_arena_bufs(c, bufs); fleet_ctx_release(c); vkDestroyInstance(inst, nullptr);
        return 1;
    }
    printf("  ring: %u slot(s) x %llu B = %llu B, memory type %d (%s)\n", p.ring.depth,
           (unsigned long long)p.ring.slot_bytes, (unsigned long long)p.ring.total_bytes,
           p.ring.mem_type,
           fleet_heap_is_unified(c.mp, c.mp.memoryTypes[p.ring.mem_type].heapIndex)
               ? "on a unified heap" : "host-visible, not on a unified heap");
    printf("  pipelined run below moves each %llu B slot D2H out of PROD and H2D into CONS, so the\n"
           "  bytes cross the link twice per hop. %s\n", (unsigned long long)slot_bytes,
           p.ring.shared_with
               ? "shared_with is TRUE: a second ctx imported the ring's host pointer, so this host\n"
                 "  does have two devices that see one heap - the copy was skipped and the hop is a\n"
                 "  store/load."
               : "NOT EXERCISABLE HERE: only one compute-capable device was opened, so the "
                 "second card's H2D->compute overlap and the PCIe-to-PCIe path a two-card machine "
                 "takes are untested, and the D2H->host->H2D round trip is what IS tested.");

    p.input_ref = [&bufs, slot_bytes](FleetCtx &, uint32_t, uint32_t) {
        HiddenStateRef h;                       // the consumer of the hop reads CONS
        h.buf = bufs[1].b; h.off = 0; h.bytes = slot_bytes;
        return h;
    };
    const FleetRecordFn record = [&bufs, slot_bytes](FleetCtx &, const FleetStage &st,
                                                     VkCommandBuffer cb, HiddenStateRef *out) {
        // Even sequence: range 0 produces into PROD. Odd: range 1 consumes CONS and produces into
        // WORK, so the bytes a hop delivered are never overwritten by the stage that received them.
        const ArenaBuf &w = (st.range == 0) ? bufs[0] : bufs[2];
        vkCmdFillBuffer(cb, w.b, 0, slot_bytes, ring_pattern(st.seq));
        if (out) { out->buf = w.b; out->off = 0; out->bytes = slot_bytes; }
        return true;
    };

    const auto t0 = std::chrono::steady_clock::now();
    const bool ran = fleet_pipeline_run(p, micro, record);
    const double run_ms = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - t0).count() / 1000.0;
    if (!ran) {
        fprintf(stderr, "bonsai_fleet ring: fleet_pipeline_run failed\n");
        fleet_pipeline_destroy(p); destroy_arena_bufs(c, bufs); fleet_ctx_release(c);
        vkDestroyInstance(inst, nullptr); return 1;
    }
    printf("\n  pipelined run: %llu stage(s), %llu hop(s) in %.3f ms\n",
           (unsigned long long)p.stages_run, (unsigned long long)p.hops, run_ms);
    if (p.hops) {
        printf("  per hop: wall %.1f us | host-visible waits: D2H fence %.1f us, doorbell stamp "
               "%.1f us, back-pressure %.1f us total over %llu wait(s), command-buffer recycle "
               "%.1f us total over %llu wait(s)\n",
               run_ms * 1000.0 / double(p.hops), p.us_wait_out / double(p.hops),
               p.us_wait_bell / double(p.hops), p.us_wait_in,
               (unsigned long long)p.waits_in, p.us_wait_cb, (unsigned long long)p.waits_cb);
        // The two completion columns side by side IS the finding. While the pipeline's fences were
        // created SIGNALED they disagreed by ~1000x (fleet_ring_v3.log: fence 2.3 us/hop, stamp
        // 2445.4 us/hop) and every rate the rig printed before the doorbell existed was measured
        // against that early return. Now the fences are created with flags = 0, so agreement is what
        // this line should report - and if it does not, the created-SIGNALED control further down is
        // the first suspect. Printed rather than narrated.
        printf("  completion signal audit: fence column %.1f us/hop vs doorbell column %.1f us/hop, "
               "%llu stamped / %llu timeout(s) -> %s\n",
               p.us_wait_out / double(p.hops), p.us_wait_bell / double(p.hops),
               (unsigned long long)p.bell_hits, (unsigned long long)p.bell_timeouts,
               p.ring.bell_map
                   ? (p.us_wait_bell > 4.0 * p.us_wait_out
                          ? "DISAGREE: the stamp needed >4x the fence column, so a hop published on "
                            "that fence alone would hand the consumer bytes that have not landed"
                          : "fence and stamp agree (within 4x): with flags = 0 the fence armed by its "
                            "own submit is a sound completion signal, and the stamp is kept as the "
                            "assumption-free proof")
                   : "no doorbell available - see the init line; the fence column is unverified");
        printf("  bytes across the ring: %llu B total, %.2f MiB per hop; at the MEASURED 7.12 GB/s "
               "a %llu B slot is %.2f us of engine time per leg. Compare that with the doorbell "
               "column above: the hop is DMA-bound, not SYNC-bound - the earlier \"sync-dominated\" "
               "reading of this line was an artefact of timing the created-SIGNALED fence, which "
               "returns before the copy retires (MEASURED submit+fence round trip %.1f us)\n",
               (unsigned long long)p.hop_bytes,
               double(p.hop_bytes) / double(p.hops) / double(MiB),
               (unsigned long long)slot_bytes,
               double(slot_bytes) / (FLEET_MEASURED_SDMA_GBPS_H2D * 1e9) * 1e6,
               FLEET_MEASURED_SUBMIT_FENCE_US);
        printf("  back-pressure events: %llu at depth %u (a depth-1 ring would take one wait per "
               "hop; D>1 is the mechanism, not a tuning knob)\n",
               (unsigned long long)p.backpressure, p.ring.depth);
    }

    // ---- did the RIGHT bytes arrive? That is the claim worth making; bandwidth without it is noise.
    const uint64_t produced = p.ring.produced.load(std::memory_order_acquire);
    const uint64_t consumed = p.ring.consumed.load(std::memory_order_acquire);
    printf("  ring cursors: produced %llu, consumed %llu %s\n", (unsigned long long)produced,
           (unsigned long long)consumed,
           produced == consumed ? "(equal, as they must be after the drain)"
                                : "(UNEQUAL - the drain did not collect every hop)");
    // The cursor is a PAYLOAD count, so it must equal the hop count exactly. If it does not, the
    // ring and the scheduler disagree about what a cursor means - the disagreement that made this
    // rig print 6 phantom mismatches (the cursor counted stages, only every nb-th stage is a hop, so
    // slots that never received a copy were expected to hold bytes).
    printf("  cursor cross-check: produced %llu vs hops %llu -> %s\n",
           (unsigned long long)produced, (unsigned long long)p.hops,
           produced == p.hops ? "consistent" : "INCONSISTENT (the ring cursor is not the hop count)");
    // What each slot SHOULD hold comes from the pipeline's own record: the stage whose bytes landed
    // there last. A slot nothing ever wrote is reported as untouched, not silently expected to be
    // zero (and not counted as a mismatch, which is what the stage-vs-payload conflation produced).
    std::vector<uint64_t> src(depth, ~0ull);
    for (uint32_t s = 0; s < depth && s < p.slot_src_seq.size(); ++s) src[s] = p.slot_src_seq[s];
    uint64_t bad = 0, checked = 0, untouched = 0;
    const size_t words_per_slot = slot_bytes / 4;
    for (uint32_t s = 0; s < depth; ++s) {
        if (src[s] == ~0ull) { ++untouched; continue; }
        const uint32_t *w = (const uint32_t *)p.ring.map + size_t(s) * words_per_slot;
        const uint32_t want = ring_pattern(src[s]);
        // first word, middle word, last word: "the copy stopped early" is the failure mode, and a
        // one-word check would pass it.
        const size_t probe[3] = {0, words_per_slot / 2, words_per_slot - 1};
        for (int i = 0; i < 3; ++i) {
            ++checked;
            if (w[probe[i]] != want) {
                ++bad;
                if (bad == 1ull)
                    printf("    slot %u: expected 0x%08X (stage %llu), found 0x%08X at word %llu\n",
                           s, want, (unsigned long long)src[s], w[probe[i]],
                           (unsigned long long)probe[i]);
            }
        }
    }
    printf("  D2H arrival: %llu word(s) read through the ring mapping over %u written slot(s) "
           "(%u untouched), %llu mismatch(es) -> %s\n",
           (unsigned long long)checked, unsigned(depth - untouched), unsigned(untouched),
           (unsigned long long)bad,
           bad ? "FAIL" : "PASS - the DMA put into the slot exactly what the producer's dispatch "
                         "wrote into its window, and the publish happened after the fence");

    // ---- the H2D leg, verified with no readback buffer: slot -> CONS, then CONS -> a different
    // slot, then read that slot through the mapping. If either leg lost bytes the comparison fails.
    if (produced && depth >= 2 && !p.ring.shared_with) {
        const uint64_t src_seq = produced - 1;
        const uint32_t s_src = uint32_t(src_seq & p.ring.mask);
        const uint32_t s_dst = (s_src + 1) & p.ring.mask;
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        fci.flags = 0;                 // unsignaled; see gpu_fleet.h fleet_pipeline_init for why
        VkFence f = VK_NULL_HANDLE;
        FK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &f));
        // Each leg is timed twice - fence wait, then vkQueueWaitIdle - because the first run of this
        // block printed an H2D leg of 1.2 us for an 8 MiB PCIe copy, which is 6000x the MEASURED
        // 7.12 GB/s. Either the fence returned before the DMA retired or the copy is not PCIe-bound;
        // the idle time after the wait tells which, and until it is known no rate from this rig is
        // quotable. Reporting both is the only honest option for a number this surprising.
        fleet_dma_async(c, p.dma_cb[0][0], p.ring.view, uint64_t(s_src) * slot_bytes, bufs[1].b, 0,
                        slot_bytes, f, VK_NULL_HANDLE, VK_NULL_HANDLE);
        const double t_h2d = fleet_fence_wait_arm(c, f);
        const auto h0 = std::chrono::steady_clock::now();
        FK_CHECK(vkQueueWaitIdle(c.has_tx ? c.qtx : c.queue));
        const double h2d_idle = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - h0).count();
        fleet_dma_async(c, p.dma_cb[0][0], bufs[1].b, 0, p.ring.view, uint64_t(s_dst) * slot_bytes,
                        slot_bytes, f, VK_NULL_HANDLE, VK_NULL_HANDLE);
        const double t_d2h = fleet_fence_wait_arm(c, f);
        const auto h1 = std::chrono::steady_clock::now();
        FK_CHECK(vkQueueWaitIdle(c.has_tx ? c.qtx : c.queue));
        const double d2h_idle = std::chrono::duration<double, std::micro>(
            std::chrono::steady_clock::now() - h1).count();
        const uint32_t *w = (const uint32_t *)p.ring.map + size_t(s_dst) * words_per_slot;
        // The pattern is keyed on the PRODUCING STAGE, and s_src is the last payload's slot, so the
        // value to expect is the one the pipeline recorded for that slot - not ring_pattern of the
        // payload index (the two sequences differ whenever a microbatch has more than one range).
        const uint32_t want = ring_pattern(p.slot_src_seq[s_src]);
        const bool ok = w[0] == want && w[words_per_slot / 2] == want && w[words_per_slot - 1] == want;
        printf("  round trip slot%u -> CONS -> slot%u: %s (fence %.1f / %.1f us, + queue-idle %.1f / "
               "%.1f us; each leg is copy + one submit+fence round trip, MEASURED %.1f us)\n", s_src,
               s_dst, ok ? "bytes intact" : "BYTES WRONG", t_h2d, t_d2h, h2d_idle, d2h_idle,
               FLEET_MEASURED_SUBMIT_FENCE_US);
        if (!ok) {
            printf("    slot%u[0] = 0x%08X, [%llu] = 0x%08X, [%llu] = 0x%08X, expected 0x%08X "
                   "(payload %llu came from stage %llu)\n", s_dst, w[0],
                   (unsigned long long)(words_per_slot / 2), w[words_per_slot / 2],
                   (unsigned long long)(words_per_slot - 1), w[words_per_slot - 1], want,
                   (unsigned long long)src_seq, (unsigned long long)p.slot_src_seq[s_src]);
            const uint32_t *ws = (const uint32_t *)p.ring.map + size_t(s_src) * words_per_slot;
            printf("    source slot%u[0] = 0x%08X (CONS carried whatever the source held at the "
                   "moment the SECOND copy read it)\n", s_src, ws[0]);
            ++bad;
        }
        vkDestroyFence(c.dev, f, nullptr);
    }

    // ---- per-leg bandwidth, fenced (what one hop pays) and batched (what the engine can do when
    // the sync is amortised, which is what a depth-D pipeline is chasing).
    int bell_untrusted = 0;      // bumped if the created-SIGNALED control reproduces an early return
    uint64_t iters = (32ull * MiB) / slot_bytes;
    if (iters < 8) iters = 8;
    if (iters > 64) iters = 64;
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    fci.flags = 0;                 // unsignaled, so the only thing that can signal it is its own
                                   // submit; the created-SIGNALED pattern this rig first shipped
                                   // with returns from every wait before the copy retires
                                   // (fleet_ring_v3.log, control below)
    VkFence f = VK_NULL_HANDLE;
    FK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &f));
    double acc = 0, tail = 0;
    for (uint64_t i = 0; i < iters; ++i) {
        fleet_dma_async(c, p.dma_cb[0][uint32_t(i % depth)], bufs[0].b, 0, p.ring.view,
                        uint64_t(i % depth) * slot_bytes, slot_bytes, f, VK_NULL_HANDLE,
                        VK_NULL_HANDLE);
        acc += fleet_fence_wait_arm(c, f);
        // The same fence-vs-queue-idle audit as the batched line below, run every iteration so the
        // two independent measurements of "when did the DMA really retire" are averaged over the
        // whole sweep instead of spot-checked once.
        const auto q0 = std::chrono::steady_clock::now();
        FK_CHECK(vkQueueWaitIdle(c.has_tx ? c.qtx : c.queue));
        tail += std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now() - q0)
                    .count();
    }
    const double per_hop_us = (acc + tail) / double(iters);
    printf("  fenced D2H audit: the fence alone reported %.1f us per hop, the queue-idle tail after "
           "it added %.1f us (a tail near zero means the fence IS the completion signal here)\n",
           acc / double(iters), tail / double(iters));
    const double gobs = double(slot_bytes) / (per_hop_us * 1e-6) / 1e9;
    printf("  fenced D2H  %llu x %llu B on family %u: %.1f us per hop -> %.2f GB/s wall = %.0f%% of "
           "the MEASURED %.2f GB/s ceiling, submit, fence AND the queue-drain tail included\n",
           (unsigned long long)iters, (unsigned long long)slot_bytes, c.has_tx ? c.tfi : c.qfi,
           per_hop_us, gobs, 100.0 * gobs / FLEET_MEASURED_SDMA_GBPS_D2H,
           FLEET_MEASURED_SDMA_GBPS_D2H);
    // ---- the trap, demonstrated on demand rather than archived in a comment.
    // The loop above now uses UNSIGNALED fences (created with flags = 0, submitted, waited, reset,
    // resubmitted), and its fence column should be the copy's real time. This control repeats the
    // same leg with a fence created VK_FENCE_CREATE_SIGNALED_BIT - the pattern the first three
    // versions of this rig used - so every run re-measures the failure instead of asking the reader
    // to trust runs/fleet_ring_v3.log. Measured there: created-SIGNALED 0.7-2.3 us with a 1202 us
    // queue tail, flags = 0 1263.6 us with a 0.8 us tail, both for the same 8 MiB D2H on the
    // transfer-only queue. The mechanism is not a driver bug: a fence is a binary state, so waiting
    // one that is ALREADY signalled returns without reference to the submit it should witness. The
    // control's queue-idle runs BEFORE vkDestroyFence on purpose: destroying a fence whose signal
    // operation is still pending is undefined, and the early wait is exactly what leaves it pending.
    {
        VkFenceCreateInfo bad{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        bad.flags = VK_FENCE_CREATE_SIGNALED_BIT;
        double acc_b = 0, tail_b = 0;
        const uint64_t b_iters = iters < 4ull ? iters : 4ull;
        for (uint64_t i = 0; i < b_iters; ++i) {
            VkFence bf = VK_NULL_HANDLE;
            FK_CHECK(vkCreateFence(c.dev, &bad, nullptr, &bf));
            fleet_dma_async(c, p.dma_cb[0][uint32_t(i % depth)], bufs[0].b, 0, p.ring.view,
                            uint64_t(i % depth) * slot_bytes, slot_bytes, bf, VK_NULL_HANDLE,
                            VK_NULL_HANDLE);
            acc_b += fleet_fence_wait_arm(c, bf);
            const auto q0 = std::chrono::steady_clock::now();
            FK_CHECK(vkQueueWaitIdle(c.has_tx ? c.qtx : c.queue));
            tail_b += std::chrono::duration<double, std::micro>(
                std::chrono::steady_clock::now() - q0).count();
            vkDestroyFence(c.dev, bf, nullptr);
        }
        const bool lies = (acc_b / double(b_iters)) < (tail_b / double(b_iters));
        printf("  created-SIGNALED control %llu x %llu B: fence %.1f us, queue-idle tail %.1f us -> "
               "%s\n", (unsigned long long)b_iters, (unsigned long long)slot_bytes,
               acc_b / double(b_iters), tail_b / double(b_iters),
               lies
                   ? "REPRODUCED: the wait returned while the queue still had the copy in it. A fence "
                     "created SIGNALED is already signalled when the wait runs, so it witnesses "
                     "nothing; every fence in this repo is therefore created with flags = 0 and is "
                     "armed only by its own submit."
                   : "the pattern behaved here (fence >= tail); the v3 log's early return did not "
                     "reproduce, so treat created-SIGNALED as unproven rather than proven-safe");
        if (lies) ++bell_untrusted;
    }

    const uint64_t batch = std::min<uint64_t>(iters, 16);
    fleet_begin_cb(c, p.dma_cb[0][0]);
    for (uint64_t i = 0; i < batch; ++i) {
        VkBufferCopy cp{0, uint64_t(i % depth) * slot_bytes, slot_bytes};
        vkCmdCopyBuffer(p.dma_cb[0][0], bufs[0].b, p.ring.view, 1, &cp);
    }
    FK_CHECK(vkEndCommandBuffer(p.dma_cb[0][0]));
    fleet_submit(c, c.has_tx ? c.qtx : c.queue, p.dma_cb[0][0], f, VK_NULL_HANDLE, VK_NULL_HANDLE,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
    const double t_batch_fence = fleet_fence_wait_arm(c, f);
    const auto b1 = std::chrono::steady_clock::now();
    FK_CHECK(vkQueueWaitIdle(c.has_tx ? c.qtx : c.queue));
    const double t_batch_idle = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - b1).count();
    const double bus = t_batch_fence + t_batch_idle;
    // Both numbers are printed because on the first run they disagreed by orders of magnitude and a
    // reader could not tell which one the rig believes: with the created-SIGNALED fence of that first
    // version, `submit -> fence` claimed 148 us per 8 MiB copy (56.67 GB/s on a PCIe link whose
    // MEASURED ceiling is 7.15 GB/s). The queue-idle tail is the independent check. The rate quoted
    // uses the SUM, which is what the host actually waited for whether or not the fence is sound.
    printf("  batched D2H %llu x %llu B in ONE submit: fence %.1f us total, queue-idle tail %.1f us "
           "total -> %.1f us per copy, %.2f GB/s end-to-end (MEASURED single-copy ceiling %.2f GB/s; "
           "if the fence column alone exceeds it, the fence is not the completion signal and the SUM "
           "is the number to quote)\n",
           (unsigned long long)batch, (unsigned long long)slot_bytes, t_batch_fence, t_batch_idle,
           bus / double(batch), double(slot_bytes * batch) / (bus * 1e-6) / 1e9,
           FLEET_MEASURED_SDMA_GBPS_D2H);

    vkDestroyFence(c.dev, f, nullptr);
    // Printed BEFORE the teardown: fleet_ring_free nulls bell_map, and reading it afterwards would
    // report "no doorbell" for a run that had one.
    printf("\n  publish condition in force: %s; every fence here is created with flags = 0 and armed "
           "only by its own submit (a fence created SIGNALED is %s on this driver, see the control "
           "above)\n",
           p.ring.bell_map ? "the DOORBELL stamp, written in the same CB as the copy, plus the fence"
                           : "the FENCE ONLY - no doorbell could be allocated, so every hop here is "
                             "unverified and the rates are upper bounds",
           bell_untrusted ? "REPRODUCIBLY SATISFIED BEFORE ITS SUBMIT RETIRES"
                          : "not observed returning early on this run");
    fleet_pipeline_destroy(p);
    destroy_arena_bufs(c, bufs);
    fleet_ctx_release(c);
    vkDestroyInstance(inst, nullptr);
    const double total_ms = std::chrono::duration<double, std::micro>(
        std::chrono::steady_clock::now() - cmd_t0).count() / 1000.0;
    printf("  device released. Whole command took %.0f ms of GPU-touching time.\n", total_ms);
    return bad ? 1 : 0;
}

// ------------------------------------------------------------------ probe
static int cmd_probe(int argc, char **argv) {
    (void)argc; (void)argv;
    VkInstance inst = make_instance("bonsai_fleet_probe");
    if (!inst) return 1;
    fleet_probe(inst);
    vkDestroyInstance(inst, nullptr);
    printf("\n  instance destroyed. `probe` never created a logical device, never allocated, never\n"
           "  submitted: it is the only command here that reads topology without touching the card.\n");
    return 0;
}

// ------------------------------------------------------------------ selftest
// Offline and deterministic. The point is that the algebra and the DECISIONS can be wrong in a way
// a single-card host is still able to catch - so they are tested against synthetic tables the host
// cannot influence, rather than against the one memory-type list this driver happens to print.
static int g_checks = 0, g_failed = 0;
static void st_expect(bool cond, const char *what) {
    ++g_checks;
    if (cond) { printf("  ok  : %s\n", what); return; }
    ++g_failed; printf("  FAIL: %s\n", what);
}

// SYNTHETIC memory-type tables: {propertyFlags, heapIndex} pairs plus the heap sizes. Real
// VkMemoryPropertyFlags values, hand-built heaps - which is the only way to ask this host's allocator
// "what would you do on an APU" when there is no APU.
struct TypeSpec { VkMemoryPropertyFlags f; uint32_t heap; };
static VkPhysicalDeviceMemoryProperties mk_mp(const std::vector<TypeSpec> &types,
                                              const std::vector<uint64_t> &heaps) {
    VkPhysicalDeviceMemoryProperties mp{};
    const size_t nh = std::min<size_t>(heaps.size(), 16);
    for (size_t h = 0; h < nh; ++h) {
        mp.memoryHeaps[h].size = heaps[h];
        mp.memoryHeaps[h].flags = (heaps[h] >= (1ull << 30)) ? VK_MEMORY_HEAP_DEVICE_LOCAL_BIT : 0u;
    }
    const size_t nt = std::min<size_t>(types.size(), 16);
    for (size_t i = 0; i < nt; ++i) {
        mp.memoryTypes[i].propertyFlags = types[i].f;
        mp.memoryTypes[i].heapIndex = std::min<uint32_t>(types[i].heap, uint32_t(nh ? nh - 1 : 0));
    }
    mp.memoryTypeCount = uint32_t(nt);
    mp.memoryHeapCount = uint32_t(nh);
    return mp;
}

static void st_ring_algebra() {
    printf("\n  -- ring algebra (no driver) --\n");
    PinnedRing r;
    st_expect(fleet_ring_alloc_host(r, 4, FLEET_HIDDEN_STATE_BYTES),
              "depth 4 x 16 KiB host ring allocates");
    st_expect(r.depth == 4 && r.mask == 3, "mask = depth-1, so the slot select is an AND and never "
                                          "a division");
    st_expect(r.total_bytes == 4ull * FLEET_HIDDEN_STATE_BYTES, "total = depth x slot_bytes");
    PinnedRing b1, b2, b3;
    st_expect(!fleet_ring_alloc_host(b1, 3, 1024), "depth 3 refused - a non-power-of-two ring needs "
                                                  "modulo in the per-boundary path");
    st_expect(!fleet_ring_alloc_host(b2, 1, 1024), "depth 1 refused - it cannot hold a payload while "
                                                   "the next one is produced");
    st_expect(!fleet_ring_alloc_host(b3, 2, 0), "slot_bytes 0 refused");
    void *s0 = fleet_ring_slot(r, 0), *s1 = fleet_ring_slot(r, 1), *s4 = fleet_ring_slot(r, 4);
    st_expect(s0 != s1, "consecutive sequences use different slots");
    st_expect(s0 == s4, "sequence 4 wraps onto sequence 0's slot - the exact reason reserve() must "
                        "refuse rather than evict");
    st_expect(fleet_ring_reserve(r, 0), "an empty ring accepts seq 0");
    for (uint64_t seq = 0; seq < 4; ++seq) fleet_ring_publish(r, seq);
    st_expect(r.produced.load(std::memory_order_acquire) == 4, "publish is monotone: 4 publishes "
                                                              "=> produced 4");
    st_expect(fleet_ring_ready(r, 3) && !fleet_ring_ready(r, 4), "readiness follows the cursor, not "
                                                                 "the wall clock");
    st_expect(!fleet_ring_reserve(r, 4), "a FULL ring refuses seq 4 instead of overwriting slot 0");
    fleet_ring_consume(r, 0);
    st_expect(fleet_ring_reserve(r, 4), "after one consume the wrapped sequence is admitted");
    uint32_t *w = (uint32_t *)fleet_ring_slot(r, 2);
    w[0] = 0xDEADBEEFu;
    w[FLEET_HIDDEN_STATE_BYTES / 4 - 1] = 0x12345678u;
    st_expect(((uint32_t *)fleet_ring_slot(r, 6))[0] == 0xDEADBEEFu &&
              ((uint32_t *)fleet_ring_slot(r, 2))[FLEET_HIDDEN_STATE_BYTES / 4 - 1] == 0x12345678u,
              "a slot's first and last word are addressable through the mapping at the wrapped index "
              "too");
    free(r.map); r.map = nullptr;
}

static void st_split_arithmetic() {
    printf("\n  -- split arithmetic (no driver) --\n");
    const std::vector<uint64_t> per(8, 1ull * MiB);
    FleetSplit s = fleet_split_schedule(8, 2, per, 1ull * MiB, 1ull * MiB, 8ull * MiB);
    st_expect(s.ok, "8 uniform layers over 2 cards inside an 8 MiB budget");
    st_expect(s.ranges.size() == 2, "one contiguous range per card");
    st_expect(s.ranges[0].layer_lo == 0 && s.ranges[1].layer_hi == 8 &&
              s.ranges[0].layer_hi == s.ranges[1].layer_lo,
              "the ranges tile [0,8) with no gap and no overlap - the contiguity the GDN-state-never-"
              "crosses argument depends on");
    uint64_t sum = 0; for (auto &r : s.ranges) sum += r.bytes;
    st_expect(sum == s.total_bytes && s.total_bytes == 10ull * MiB,
              "byte conservation: sum(ranges) == layers + head + tail");
    st_expect(s.ranges[0].bytes >= 1ull * MiB && s.ranges[1].bytes >= 1ull * MiB,
              "head charged to card 0, tail to the last card");
    st_expect(s.hops_per_token() == 1, "2 ranges => 1 boundary crossing per token");
    FleetSplit one = fleet_split_schedule(8, 1, per, 1ull * MiB, 1ull * MiB, 16ull * MiB);
    st_expect(one.ok && one.hops_per_token() == 0 && one.ranges[0].bytes == 10ull * MiB,
              "1 card: everything on it, zero hops - the schedule degenerates to today's build");
    FleetSplit many = fleet_split_schedule(8, 9, per, 1ull * MiB, 1ull * MiB, 8ull * MiB);
    st_expect(!many.ok && many.refuse.find("zero layers") != std::string::npos,
              "cards > layers is refused instead of building an empty range");
    FleetSplit tight = fleet_split_schedule(8, 2, per, 1ull * MiB, 1ull * MiB, 4ull * MiB);
    st_expect(!tight.ok, "a budget below (layers+head+tail)/cards is refused");
    FleetSplit headheavy = fleet_split_schedule(8, 2, per, 9ull * MiB, 1ull * MiB, 8ull * MiB);
    st_expect(!headheavy.ok && headheavy.refuse.find("already over budget") != std::string::npos,
              "a head tensor larger than one card's budget is refused BEFORE any layer is assigned");
    const std::vector<uint64_t> short_table(7, 1ull * MiB);
    st_expect(!fleet_split_schedule(8, 2, short_table, 0, 0, 8ull * MiB).ok,
              "a byte table that does not cover every layer is refused");
    // Unequal layers: a byte-balanced cut must not put every heavy block on one card.
    std::vector<uint64_t> uneven(4, 4ull * MiB); uneven[0] = 1ull * MiB; uneven[3] = 1ull * MiB;
    FleetSplit u = fleet_split_schedule(4, 2, uneven, 0, 0, 8ull * MiB);
    st_expect(u.ok && u.ranges[0].bytes <= 8ull * MiB && u.ranges[1].bytes <= 8ull * MiB,
              "uneven layer bytes still fit both budgets");
}

static void st_zero_copy_decision() {
    printf("\n  -- arena memory-type decisions (SYNTHETIC tables, no driver) --\n");
    const uint32_t D = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, V = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT,
                   C = VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, K = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    // (1) THIS host's discrete shape, transcribed from what amdvlk reports: two host-visible heap-0
    // types (one cached, one not) and one DEVICE_LOCAL type on a heap that has no host type at all.
    VkPhysicalDeviceMemoryProperties disc = mk_mp(
        {{V | C | K, 0}, {D, 1}, {V | C, 0}}, {8ull << 30, FLEET_MEASURED_DEVICE_LOCAL_MIB * MiB});
    st_expect(fleet_arena_decide(disc, FLEET_ARENA_ZERO_COPY).type < 0,
              "zero-copy REFUSES this table (no type is DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT at "
              "once) - the branch cannot fall back and the refusal is the answer, not a failure");
    FleetArenaPick pl = fleet_arena_decide(disc, FLEET_ARENA_DEVICE_LOCAL);
    st_expect(pl.type == 1, "discrete branch picks the FIRST type carrying DEVICE_LOCAL and no other "
                            "requirement - that is bonsai_vk.cpp:331-335's predicate, and ~30 "
                            "existing logs are computed against exactly this index");
    st_expect(!fleet_heap_is_unified(disc, 1), "a discrete card's DEVICE_LOCAL heap is not reported "
                                               "as unified, so the ring import must not be skipped "
                                               "on its account");
    // (2) an APU shape: ONE heap that carries plain host-visible types and a
    // DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type.
    VkPhysicalDeviceMemoryProperties apu = mk_mp(
        {{V | C | K, 0}, {D | V | C | K, 0}, {D | V | C, 0}, {D, 0}}, {16ull << 30, 1ull << 30});
    FleetArenaPick pz = fleet_arena_decide(apu, FLEET_ARENA_ZERO_COPY);
    st_expect(pz.type == 1, "zero-copy picks a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type");
    st_expect(pz.cached, "and prefers the HOST_CACHED one - MEASURED, WC vs cached mapping is "
                         "24.8 ms against 6.28 ms over the same bytes (END_TO_END_MEASURED §2.2)");
    st_expect(pz.shared_heap, "and reports the heap unified, which is what makes the ring alias and "
                              "the hop a store instead of a copy");
    st_expect(fleet_arena_decide(apu, FLEET_ARENA_DEVICE_LOCAL).type == 1,
              "on the same table the discrete branch still returns the first DEVICE_LOCAL type: the "
              "MODE flag, not the device class, decides which branch runs");
    // (3) DEVICE_LOCAL|HOST_VISIBLE without HOST_COHERENT: refuse rather than guess about flushes.
    VkPhysicalDeviceMemoryProperties noco =
        mk_mp({{V | C | K, 0}, {D | V, 0}}, {4ull << 30, 4ull << 30});
    st_expect(fleet_arena_decide(noco, FLEET_ARENA_ZERO_COPY).type < 0,
              "DEVICE_LOCAL|HOST_VISIBLE without HOST_COHERENT is refused - a non-coherent mapping "
              "needs flushes the arena writer does not issue, so it would corrupt weights silently");
    // (4) PCIe BAR memory: a valid single-allocation arena, but NOT a unified heap.
    VkPhysicalDeviceMemoryProperties bar =
        mk_mp({{V | C | K, 0}, {D | V | C, 1}}, {8ull << 30, 256ull << 20});
    st_expect(fleet_arena_decide(bar, FLEET_ARENA_ZERO_COPY).type == 1,
              "a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT type on its own heap is still accepted "
              "(BAR-streaming weights are CPU-writable)");
    st_expect(!fleet_heap_is_unified(bar, 1),
              "but it is NOT called unified - so the ring-alias shortcut is not taken on a machine "
              "that only has BAR memory");
    // (5) the unified-heap test needs BOTH halves: a heap with only DEVICE_LOCAL|HOST_VISIBLE and no
    // plain host type is not proven shared.
    VkPhysicalDeviceMemoryProperties half =
        mk_mp({{D | V | C, 0}}, {4ull << 30});
    st_expect(!fleet_heap_is_unified(half, 0), "a single-type heap is not enough to call it unified");
}

static void st_model_bytes() {
    printf("\n  -- model byte sums --\n");
    static const char *kDefaultModel = "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    const char *path = getenv("BONSAI_MODEL"); if (!path) path = kDefaultModel;
    FILE *f = fopen(path, "rb");
    if (!f) {
        printf("  SKIP: %s is not readable here, so the DERIVED fallback constants are the only "
               "source and their arithmetic is NOT re-verified by this run.\n", path);
        return;
    }
    fclose(f);
    ByteTable t; std::string why;
    if (!bytes_from_model(path, t, why)) {
        ++g_checks; ++g_failed;
        printf("  FAIL: bytes_from_model(%s): %s\n", path, why.c_str());
        return;
    }
    uint64_t tot = 0, gdn = 0, att = 0, tf = 0;
    for (size_t i = 0; i < t.dev.size(); ++i) {
        tot += t.dev[i]; tf += t.file[i];
        if (t.attn[i]) att = t.dev[i]; else gdn = t.dev[i];
    }
    st_expect(t.layers == 32, "the 9B's block_count reads as 32");
    st_expect(gdn == 62554880ull, "a GDN block's DEVICE-resident bytes are 62,554,880");
    st_expect(att == 59017216ull, "a softmax block's are 59,017,216");
    st_expect(tot == 1973454848ull, "24 GDN + 8 softmax layer bytes sum to 1,973,454,848 = 1.838 GiB");
    st_expect(tf == 1541441536ull, "the FILE (PTQ1_0, 1.75 bpw) layer bytes sum to 1,541,441,536 - "
                                   "the upload number, not the budget number");
    st_expect(t.head_dev == 286064640ull, "token_embd in the device form is 286,064,640 B");
    st_expect(t.tail_dev == 286064640ull + 16384ull, "output.weight + output_norm = 286,081,024 B");
    st_expect(t.head_file == 222494720ull, "token_embd's file form is 222,494,720 B");
    st_expect(t.other_dev == 0, "no non-layer tensor other than the head/tail pair went unaccounted "
                               "for (a nonzero value would be charged to card 0 and printed)");
    // The splits a fleet would actually run.
    FleetSplit s2 = fleet_split_schedule(32, 2, t.dev, t.head_dev, t.tail_dev, 3840ull * MiB);
    st_expect(s2.ok && s2.hops_per_token() == 1, "32 layers / 2 cards at 3840 MiB/card fits, 1 "
                                                 "boundary per token");
    FleetSplit s2tight = fleet_split_schedule(32, 2, t.dev, t.head_dev, t.tail_dev, 1200ull * MiB);
    st_expect(!s2tight.ok, "the same split at 1200 MiB/card is REFUSED - the weights alone need "
                           "2.37 GiB / 2 = 1213.9 MiB per card");
    FleetSplit s4 = fleet_split_schedule(32, 4, t.dev, t.head_dev, t.tail_dev, 1024ull * MiB);
    st_expect(s4.ok && s4.hops_per_token() == 3, "4 cards at 1 GiB each fits and costs 3 hops/token");
    FleetSplit s8 = fleet_split_schedule(32, 8, t.dev, t.head_dev, t.tail_dev, 512ull * MiB);
    st_expect(!s8.ok, "8 cards at 512 MiB each is refused: head 273 MiB + tail 273 MiB on the two "
                      "end cards plus a quarter-GiB of layers per card does not fit, which is the "
                      "answer a budget check exists to give");
    // The fallback table must agree with the file, or the fallback is a stale number being shipped.
    ByteTable fb; bytes_fallback(fb, 32);
    st_expect(fb.dev == t.dev && fb.file == t.file && fb.head_dev == t.head_dev &&
              fb.tail_dev == t.tail_dev, "the DERIVED fallback constants match the file exactly "
                                        "for this model - if they ever stop matching, this check "
                                        "is the one that says so");
}

static void st_pipeline_contracts() {
    printf("\n  -- declared status and the numbers a report cites --\n");
    // The two unimplemented policies exist so that no reader mistakes the host-fence policy for the
    // only one possible; they must not be silently used by anything in this rig.
    st_expect(FLEET_SYNC_HOST_FENCE == 0 && FLEET_SYNC_EXTERNAL_SEMAPHORE == 1 &&
              FLEET_SYNC_SLOT_FLAG_SPIN == 2, "policy enum order = documented status: implemented, "
                                              "declared, declared");
    st_expect(FLEET_HIDDEN_STATE_BYTES == 4096ull * 4ull,
              "the hop payload is one hidden vector = 4096 floats = 16 KiB");
    st_expect(FLEET_MEASURED_DEVICE_LOCAL_MIB + FLEET_DRIVER_RESERVATION_MIB ==
              FLEET_MEASURED_CARD_NOMINAL_MIB, "the driver-reservation constant is arithmetic on the "
                                               "two MEASURED numbers (4096 - 3840), not a guess");
    // The pipeline refuses a range that names a card the fleet does not have. It is the only
    // connection between a schedule built by fleet_split_schedule and a vector of contexts built by
    // a caller, so it is checked here with the cheapest possible stand-in (no device created).
    FleetPipeline p;
    FleetSplit bad_split; bad_split.layers = 2; bad_split.cards = 2; bad_split.ok = true;
    for (uint32_t r = 0; r < 2; ++r) {
        LayerRange lr; lr.card = r; lr.layer_lo = r; lr.layer_hi = r + 1;
        bad_split.ranges.push_back(lr);
    }
    std::vector<FleetCtx> no_cards(1);   // dev == VK_NULL_HANDLE: init must not get as far as using it
    // fflush first: the refusal prints on stderr and stdout is block-buffered when piped, so without
    // this the two streams interleave mid-line and the PASS log becomes unreadable.
    fflush(stdout);
    st_expect(!fleet_pipeline_init(p, no_cards, bad_split, 2, 1024),
              "fleet_pipeline_init refuses a schedule whose ranges name cards that do not exist, "
              "instead of indexing the command-buffer table out of bounds");
}

static int cmd_selftest(int argc, char **argv) {
    (void)argc; (void)argv;
    printf("== bonsai_fleet selftest (offline; no Vulkan object is created by any path below) ==\n");
    st_ring_algebra();
    st_split_arithmetic();
    st_zero_copy_decision();
    st_model_bytes();
    st_pipeline_contracts();
    printf("\nselftest: %d checks, %d failed -> %s\n", g_checks, g_failed,
           g_failed ? "FAIL" : "PASS");
    return g_failed ? 1 : 0;
}

// ------------------------------------------------------------------ main
int main(int argc, char **argv) {
    if (argc < 2) {
        printf("bonsai_fleet - the multi-GPU / iGPU machinery for the ternary runtime (task 3)\n"
               "  probe                           topology + (A)/(B) availability on this host\n"
               "  ring [depth] [slot_MiB] [micro] the pinned ring, measured on the queues that exist\n"
               "                                  (defaults: depth 4, 1 MiB slots, 4 microbatches)\n"
               "  split <layers> <cards> [budget_MiB] [--model PATH] [--fallback]\n"
               "                                  scheduling arithmetic only; never opens a device\n"
               "  selftest                        deterministic PASS/FAIL, no GPU, no driver\n"
               "\n"
               "  env: BONSAI_GPU_INDEX=<i>  BONSAI_GPU_TYPE=integrated|discrete|zero-copy\n"
               "       BONSAI_MODEL=<path>   BONSAI_API12=1   VK_VALIDATION=1\n");
        return 2;
    }
    const std::string cmd = argv[1];
    if (cmd == "probe")    return cmd_probe(argc, argv);
    if (cmd == "ring")     return cmd_ring(argc, argv);
    if (cmd == "split")    return cmd_split(argc, argv);
    if (cmd == "selftest") return cmd_selftest(argc, argv);
    fprintf(stderr, "unknown subcommand \"%s\" - run bonsai_fleet with no arguments for usage\n",
            cmd.c_str());
    return 2;
}
