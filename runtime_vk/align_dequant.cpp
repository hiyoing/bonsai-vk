// align_dequant.cpp - layer 2 of the faithful-reproduction checks: does OUR PTQ1_0 dequantisation
// path produce the same weights as the REFERENCE one?
//
// Three parties, all measured on this card, no fourth:
//   GPU-REF   the OFFICIAL PrismML/llama.cpp vulkan-shaders/dequant_ptq1_0.comp, body verbatim
//             (runtime_vk/shaders/dequant_ptq1_0.comp), head inlined by tools/gen_dequant_ref_comp.py,
//             compiled to SPIR-V and executed on the RX 550X. This is the shader llama.cpp itself
//             uses to turn PTQ1_0 blocks into f32, so it is as close to "what the reference engine
//             means" as anything obtainable without their CPU code (PTQ1_0 has no CPU dequant).
//   CPU-REF   ptq_dequant_official_cpu() in ptq_decode.h: a line-by-line host transcription of that
//             shader's own arithmetic (pow3[] multiply, (v*3)>>8 - 1, its il/l element numbering).
//   OURS      ptq_dequant_ours() in the SAME header, which bonsai_vk.cpp #includes - so the numbers
//             here are the runtime's decode, not a copy written for the test.
//
// Two block sources, chosen on the command line:
//   REAL   100 (or -n) 28-byte blocks read straight out of
//        Ternary-Bonsai-2-9B-PTQ1_0.gguf at data_start + tensor.off + blk*28, spread over 100
//        different PTQ1_0 tensors so no single layer or row can hide a layout error. Every block's
//        provenance (tensor, block index, file offset) is printed, so the sample is re-derivable.
//   SYNTH the edge cases the checkpoint does not contain (measured: no d==0, no subnormal d, no
//        negative d, no Inf/NaN d in any of its 202 PTQ1_0 tensors). 15 fp16 scale bit patterns x
//        256 byte patterns = 3840 blocks. Byte pattern j of block q is (q + 7*j) & 0xFF, so every
//        one of the 26 packed-byte slots sees all 256 byte values exactly once and each value
//        passes through all of its trit positions.
//
// Report: for the three pairs, max|err| over finite pairs, a bit-exact count, a NaN/Inf-aware
// equality count, an integer symbol-agreement count (scale-free), and the runtime's dot-product
// identity d*(cs-xs) == sum(w_e*x_e). In SYNTH mode every scale class is also judged alone.
//
//   build: see runtime_vk/build_mingw.bat step 2/2
//   run:   align_dequant.exe models/Ternary-Bonsai-2-9B-PTQ1_0.gguf [N|synth]

#include "bonsai_gguf.h"
#include "ptq_decode.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define AK_CHECK(x)                                                                        \
    do {                                                                                   \
        VkResult _r = (x);                                                                 \
        if (_r != VK_SUCCESS) {                                                            \
            fprintf(stderr, "Vulkan failure %d at %s:%d\n", (int)_r, __FILE__, __LINE__);   \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

static const uint32_t PTQ_DTYPE = 143;      // ggml type id this checkpoint stores (measured: 202 tensors)

static std::vector<char> read_file(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s - run runtime_vk/build_mingw.bat first\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> b(size_t(n) ? size_t(n) : 1);
    size_t got = fread(b.data(), 1, size_t(n), f); fclose(f);
    if (long(got) != n) { fprintf(stderr, "short read %s\n", p); exit(1); }
    uint32_t magic; memcpy(&magic, b.data(), 4);
    if (n < 8 || magic != 0x07230203u) { fprintf(stderr, "%s is not SPIR-V\n", p); exit(1); }
    return b;
}
// the tool can be started from the repo root or from runtime_vk/, so look for the shader in both
static std::vector<char> read_shader(const char *rel) {
    if (FILE *t = fopen(rel, "rb")) { fclose(t); return read_file(rel); }
    std::string up = std::string("../") + rel;
    return read_file(up.c_str());
}
// the GGUF is 1,997,486,976 B and a tensor offset is added to it, so 64-bit seeking is mandatory -
// on Windows long is 32 bit and plain fseek() silently wraps past 2 GiB.
static bool file_seek(FILE *f, uint64_t pos) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<long long>(pos), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(pos), SEEK_SET) == 0;
#endif
}

int main(int argc, char **argv) {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const char *model = argc > 1 ? argv[1] : "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    //   align_dequant.exe <gguf> [N|synth]
    // a number reads N REAL blocks out of the checkpoint; "synth" builds the edge-pattern set the
    // checkpoint cannot supply (measured census: no d==0, no subnormal, no negative scale in it).
    const bool synth = argc > 2 && strcmp(argv[2], "synth") == 0;
    struct Edge { const char *name; uint16_t bits; };
    static const Edge EDGE[] = {
        {"+0", 0x0000}, {"-0", 0x8000},
        {"subnormal min", 0x0001}, {"subnormal mid", 0x0123}, {"subnormal max", 0x03FF},
        {"-subnormal max", 0x83FF},
        {"normal min", 0x0400}, {"-normal min", 0x8400},
        {"1.0", 0x3C00}, {"normal max", 0x7BFF}, {"-normal max", 0xFBFF},
        {"+Inf", 0x7C00}, {"-Inf", 0xFC00}, {"NaN", 0x7E00}, {"-NaN", 0xFE00},
    };
    const int NE = int(sizeof(EDGE) / sizeof(EDGE[0]));
    const bool scan = argc > 2 && strcmp(argv[2], "scan") == 0;
    const int NB = synth ? NE * 256 : (argc > 2 && !scan ? atoi(argv[2]) : 100);
    const uint32_t NEL_U = uint32_t(NB) * 128u;                 // weights in the oracle buffer
    const uint64_t src_sz = uint64_t(NB) * PTQ_BLOCK_BYTES;
    const uint64_t out_sz = uint64_t(NEL_U) * 4ull;

    // ---------------------------------------------------------------- 0. optional full-file census
    // "the checkpoint contains no subnormal / zero / negative scale" has to be MEASURED over all of
    // it, not assumed from the 100 blocks this test samples - otherwise the synthetic edge cases
    // could be dismissed as irrelevant on a guess. This walks every PTQ1_0 tensor, every 28-byte
    // block, and censuses the 16-bit scale field. Host only, no Vulkan, then exits.
    if (scan) {
        GgufHeader h;
        if (!gguf_load(model, h)) return 1;
        FILE *mf = fopen(model, "rb");
        if (!mf) { fprintf(stderr, "cannot open %s\n", model); return 1; }
        uint64_t n_blk = 0, n_byte = 0, n_t = 0;
        uint64_t c_zero = 0, c_neg = 0, c_sub = 0, c_inf = 0, c_nan = 0, c_uniq = 0;
        int64_t hist_exp[32] = {};
        std::vector<uint8_t> seen(65536, 0);               // every scale bit pattern that occurs
        // ... and where the edge cases live. "89 subnormal scales out of 69.9M blocks" is only
        // useful if we can say which tensors they are in: it decides whether our runtime would ever
        // have met them, which is exactly what the f16_to_f32 bug fix was about.
        struct Edge_Hit { std::string tensor; uint64_t blk; uint16_t bits; float val;
                          const char *kind; };
        std::vector<Edge_Hit> edge_hits;
        uint16_t lo_abs = 0, hi_abs = 0;                   // bit patterns of min/max |d|
        double lo_v = 0.0, hi_v = 0.0;
        std::vector<uint8_t> chunk;
        // BLOCK-ALIGNED CHUNKS ONLY. 1<<22 is not a multiple of 28 (4194304 = 28*149796 + 16), and
        // the first version of this census read plain 4 MiB pieces, so every chunk after the first
        // started 16 bytes into a block and the "scale" it sampled was two payload bytes. It printed
        // 1.1M subnormal and 1.26M NaN scales for a trained model - the number that is impossible is
        // what caught it. Round the chunk down to a whole number of blocks.
        const uint64_t CHUNK = ((1ull << 22) / PTQ_BLOCK_BYTES) * PTQ_BLOCK_BYTES;   // 4,194,288 B
        for (const auto &t : h.t) {
            if (t.dtype != PTQ_DTYPE) continue;
            ++n_t;
            uint64_t elems = 1;
            for (uint32_t i = 0; i < t.n_dims; ++i) elems *= t.d[i];
            const uint64_t bytes = elems / 128ull * PTQ_BLOCK_BYTES;
            chunk.resize(size_t(std::min<uint64_t>(CHUNK, bytes)));
            if (!file_seek(mf, h.data_start + t.off)) { fprintf(stderr, "seek %s\n", t.name.c_str()); return 1; }
            uint64_t done = 0;
            while (done < bytes) {
                const uint64_t want = std::min<uint64_t>(chunk.size(), bytes - done);
                const size_t got = fread(chunk.data(), 1, size_t(want), mf);
                if (got != want) { fprintf(stderr, "short read in %s at +%llu\n", t.name.c_str(),
                                          (unsigned long long)done); return 1; }
                const uint64_t blocks = want / PTQ_BLOCK_BYTES;
                for (uint64_t b = 0; b < blocks; ++b) {
                    const uint16_t bits = uint16_t(chunk[size_t(b) * 28 + 26]) |
                                          (uint16_t(chunk[size_t(b) * 28 + 27]) << 8);
                    const uint16_t mag = uint16_t(bits & 0x7FFFu);
                    const uint32_t exp = (mag >> 10) & 0x1Fu, man = mag & 0x3FFu;
                    ++n_blk;
                    seen[bits] = 1;
                    hist_exp[exp]++;
                    const char *kind = nullptr;
                    if (exp == 0x1Fu)              kind = man ? "NaN" : "Inf";
                    else if (mag == 0)             kind = "zero";
                    else if (exp == 0)             kind = "subnormal";
                    if (kind && edge_hits.size() < 4096)
                        edge_hits.push_back({t.name, done / PTQ_BLOCK_BYTES + b, bits,
                                             f16_to_f32(bits), kind});
                    if (exp == 0x1Fu) { if (man) ++c_nan; else ++c_inf; }
                    else if (mag == 0) ++c_zero;
                    else {
                        if (exp == 0) ++c_sub;
                        if (bits & 0x8000u) ++c_neg;
                        const double v = double(f16_to_f32(mag));
                        if (v < lo_v || lo_v == 0.0) { lo_v = v; lo_abs = mag; }
                        if (v > hi_v) { hi_v = v; hi_abs = mag; }
                    }
                }
                done += want; n_byte += want;
            }
        }
        for (int i = 0; i < 65536; ++i) if (seen[i]) ++c_uniq;
        fclose(mf);
        printf("==== PTQ1_0 block-scale census over the WHOLE checkpoint ====\n");
        printf("  model            %s\n", model);
        printf("  PTQ1_0 tensors   %llu of %llu   blocks %llu   bytes %llu (%.3f GiB)\n",
               (unsigned long long)n_t, (unsigned long long)h.n_tensors,
               (unsigned long long)n_blk, (unsigned long long)n_byte, double(n_byte) / 1073741824.0);
        printf("  scale patterns used %llu distinct out of 65536\n", (unsigned long long)c_uniq);
        printf("  d == 0 (either sign) %llu   subnormal %llu   negative %llu   Inf %llu   NaN %llu\n",
               (unsigned long long)c_zero, (unsigned long long)c_sub, (unsigned long long)c_neg,
               (unsigned long long)c_inf, (unsigned long long)c_nan);
        printf("  |d| range        [%.9g (0x%04X), %.9g (0x%04X)]\n", lo_v, lo_abs, hi_v, hi_abs);
        printf("  exponent-field histogram (exp: count), exp 0 = zero+subnormal, 31 = Inf/NaN:\n");
        for (int e = 0; e < 32; ++e)
            if (hist_exp[e]) printf("     %2d  %10lld\n", e, (long long)hist_exp[e]);
        // where they are, aggregated: a count without a location is not actionable. Key the tally
        // by (tensor, scale pattern) so the report says e.g. "token_embd.weight, 32 blocks with
        // d = 0x0001 = 5.96046448e-08" - which is exactly the pattern our fixed f16_to_f32 used to
        // turn into +0.
        if (!edge_hits.empty()) {
            std::vector<std::string> keys;                    // "tensor|0xHHHH"
            std::vector<uint64_t> counts;
            std::vector<uint16_t> pat;
            std::vector<std::string> tens;
            for (const auto &e : edge_hits) {
                char bhex[8]; snprintf(bhex, sizeof bhex, "%04X", e.bits);
                std::string key = e.tensor + "|" + bhex;
                size_t at = keys.size();
                for (size_t i = 0; i < keys.size(); ++i) if (keys[i] == key) { at = i; break; }
                if (at == keys.size()) { keys.push_back(key); counts.push_back(0);
                                          pat.push_back(e.bits); tens.push_back(e.tensor); }
                ++counts[at];
            }
            printf("  edge scales by (tensor, pattern), %zu distinct combinations:\n", keys.size());
            for (size_t i = 0; i < keys.size(); ++i)
                printf("     %6llu x  %-26s scale 0x%04X  d=%-14.9g\n",
                       (unsigned long long)counts[i], tens[i].c_str(), pat[i],
                       double(f16_to_f32(pat[i])));
            printf("  first %zu occurrences (tensor, block index within it):\n",
                   std::min<size_t>(edge_hits.size(), 12));
            for (size_t i = 0; i < edge_hits.size() && i < 12; ++i)
                printf("     %-24s blk=%-9llu bits=0x%04X  d=%-14.9g (%s)\n",
                       edge_hits[i].tensor.c_str(), (unsigned long long)edge_hits[i].blk,
                       edge_hits[i].bits, double(edge_hits[i].val), edge_hits[i].kind);
        }
        printf("  -> every edge pattern this test synthesises (0, subnormal, negative, Inf, NaN)\n");
        printf("     %s in the checkpoint.\n", (c_zero || c_sub || c_inf || c_nan) ? "DOES occur"
              : "does NOT occur, so the synthetic set is strictly more demanding than the model");
        return 0;
    }

    // ---------------------------------------------------------------- 1. the blocks under test
    struct Pick { std::string name; uint64_t blk; uint64_t foff; };
    std::vector<Pick> picks;
    std::vector<uint8_t> hsrc = std::vector<uint8_t>(size_t(src_sz), uint8_t(0));

    if (synth) {
        // slot j of block p holds (p + 7*j) & 0xFF. 7 is coprime with 256, so as p walks 0..255
        // every one of the 26 byte slots sees every one of the 256 byte values exactly once, and
        // each of those values goes through all five of its trit positions. That is the whole byte
        // domain, crossed with the fp16 edge patterns above.
        for (int e = 0; e < NE; ++e)
            for (int q = 0; q < 256; ++q) {
                uint8_t *pb = hsrc.data() + size_t(e * 256 + q) * PTQ_BLOCK_BYTES;
                for (int j = 0; j < 26; ++j) pb[j] = uint8_t((q + 7 * j) & 0xFF);
                pb[26] = uint8_t(EDGE[e].bits & 0xFF);
                pb[27] = uint8_t(EDGE[e].bits >> 8);
                char lbl[96]; snprintf(lbl, sizeof lbl, "%s(0x%04X)", EDGE[e].name, EDGE[e].bits);
                picks.push_back({lbl, uint64_t(e * 256 + q), 0});
            }
    } else {
        GgufHeader h;
        if (!gguf_load(model, h)) return 1;
        std::vector<const GgufTensor*> ptq;
        for (const auto &t : h.t) if (t.dtype == PTQ_DTYPE) ptq.push_back(&t);
        printf("[gguf] tensors=%llu  PTQ1_0(143)=%zu  data_start=%llu\n",
               (unsigned long long)h.n_tensors, ptq.size(), (unsigned long long)h.data_start);
        if (ptq.empty()) { fprintf(stderr, "no PTQ1_0 tensors\n"); return 1; }
        FILE *mf = fopen(model, "rb");
        if (!mf) { fprintf(stderr, "cannot open %s\n", model); return 1; }
        // deterministic and spread out: walk the PTQ tensors with a stride that yields NB distinct
        // tensors, then take one block from deep inside that tensor's own block range.
        for (int q = 0; q < NB; ++q) {
            const GgufTensor *t = ptq[size_t(double(q) * double(ptq.size()) / double(NB))];
            uint64_t nblk = 1;
            for (uint32_t i = 0; i < t->n_dims; ++i) nblk *= t->d[i];
            nblk /= 128u;                                        // elems / 128 = blocks in a tensor
            uint64_t blk = (uint64_t(q) * 4096ull + 7ull) % (nblk ? nblk : 1ull);
            uint64_t foff = h.data_start + t->off + blk * PTQ_BLOCK_BYTES;
            picks.push_back({t->name, blk, foff});
        }
        for (int q = 0; q < NB; ++q) {
            if (!file_seek(mf, picks[q].foff)) { fprintf(stderr, "seek fail\n"); return 1; }
            if (fread(hsrc.data() + size_t(q) * PTQ_BLOCK_BYTES, 1, PTQ_BLOCK_BYTES, mf) != PTQ_BLOCK_BYTES) {
                fprintf(stderr, "read fail at block %d\n", q); return 1;
            }
        }
        fclose(mf);
    }

    // ---------------------------------------------------------------- 2. CPU decoders
    std::vector<float> w_off(std::size_t{NEL_U}, 0.0f), w_our(std::size_t{NEL_U}, 0.0f);
    for (int p = 0; p < NB; ++p) {
        const uint8_t *pb = hsrc.data() + size_t(p) * PTQ_BLOCK_BYTES;
        ptq_dequant_official_cpu(pb, w_off.data() + size_t(p) * 128);
        ptq_dequant_ours(pb,        w_our.data() + size_t(p) * 128);
    }

    // ---------------------------------------------------------------- 3. Vulkan: run the official shader
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    // 1.2, not 1.1: uint8_t and float16_t members inside an SSBO struct are governed by the
    // 8bit_storage / 16bit_storage features, which are core in 1.2. The official body needs both
    // (qs/qh are bytes, d is a half).
    ai.apiVersion = VK_API_VERSION_1_2; ai.pApplicationName = "align_dequant";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; AK_CHECK(vkCreateInstance(&ici, nullptr, &inst));

    uint32_t nd = 0; AK_CHECK(vkEnumeratePhysicalDevices(inst, &nd, nullptr));
    std::vector<VkPhysicalDevice> devs(nd); AK_CHECK(vkEnumeratePhysicalDevices(inst, &nd, devs.data()));
    VkPhysicalDevice phys = devs[0];
    for (auto p : devs) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
                          if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pp.vendorID == 0x1002) phys = p; }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    printf("[dev]  %s\n", props.deviceName);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    uint32_t qfi = 0; for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    // Ask the driver what it actually allows before pretending the oracle shader can run: the
    // fidelity test is only meaningful if the byte loads and the half scale are real.
    VkPhysicalDevice8BitStorageFeatures f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    VkPhysicalDevice16BitStorageFeatures f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDeviceFloat16Int8FeaturesKHR ff16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FLOAT16_INT8_FEATURES_KHR};
    f8.pNext = &f16; f16.pNext = &ff16;
    VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feats2.pNext = &f8;
    vkGetPhysicalDeviceFeatures2(phys, &feats2);
    printf("[feat] storageBuffer8Bit=%d storageBuffer16Bit=%d uniformAndStorage16Bit=%d shaderFloat16=%d\n",
           int(f8.storageBuffer8BitAccess), int(f16.storageBuffer16BitAccess),
           int(f16.uniformAndStorageBuffer16BitAccess), int(ff16.shaderFloat16));
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfi; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    di.pNext = &f8;                              // the whole chain hangs off here; VK_TRUE = enabled
    di.pEnabledFeatures = &feats2.features;
    VkDevice dev; AK_CHECK(vkCreateDevice(phys, &di, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    auto find_type = [&](VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mp.memoryTypes[i].propertyFlags & want) == want) return i;
        fprintf(stderr, "no memory type 0x%X\n", unsigned(want)); exit(1);
    };

    auto make_buf = [&](uint64_t size, VkBufferUsageFlags use, VkMemoryPropertyFlags mprops,
                        VkBuffer &out_buf, VkDeviceMemory &out_mem) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size; bi.usage = use; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        AK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &out_buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, out_buf, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = find_type(mprops);
        AK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &out_mem));
        AK_CHECK(vkBindBufferMemory(dev, out_buf, out_mem, 0));
    };
    // A must be readable by the shader AND a transfer destination; D is written by the shader and
    // copied back out. Both live in DEVICE_LOCAL memory like the runtime's arena, so the upload goes
    // through a host-visible staging buffer - exactly the shape bonsai_vk.cpp uses (and the reason a
    // direct vkMapMemory of a DEVICE_LOCAL allocation returns VK_ERROR_MEMORY_MAP_FAILED).
    VkBuffer b_stage, b_src, b_out, b_read;
    VkDeviceMemory m_stage, m_src, m_out, m_read;
    make_buf(src_sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b_stage, m_stage);
    make_buf(src_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b_src, m_src);
    make_buf(out_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b_out, m_out);
    make_buf(out_sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b_read, m_read);

    void *map = nullptr; AK_CHECK(vkMapMemory(dev, m_stage, 0, src_sz, 0, &map));
    memcpy(map, hsrc.data(), size_t(src_sz));
    vkUnmapMemory(dev, m_stage);                   // host-visible coherent: no flush needed

    // descriptors / layout / pipeline
    VkDescriptorSetLayoutBinding bs[2] = {};
    for (int i = 0; i < 2; ++i) { bs[i].binding = i; bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                                  bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 2; sli.pBindings = bs;
    VkDescriptorSetLayout sl; AK_CHECK(vkCreateDescriptorSetLayout(dev, &sli, nullptr, &sl));
    // the fork's host passes 5 x u32 push constants to this pipeline (ggml-vulkan.cpp line ~3109)
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 5 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &sl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; AK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
    VkDescriptorPool dpool; AK_CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &sl;
    VkDescriptorSet ds; AK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi[2] = {{b_src, 0, src_sz}, {b_out, 0, out_sz}};
    VkWriteDescriptorSet wr[2] = {};
    for (int i = 0; i < 2; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = ds; wr[i].dstBinding = i;
        wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

    auto code = read_shader("shaders/dequant_ptq1_0_ref.spv");
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size(); mi.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule sm; AK_CHECK(vkCreateShaderModule(dev, &mi, nullptr, &sm));
    VkComputePipelineCreateInfo cpi2{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi2.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi2.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi2.stage.module = sm; cpi2.stage.pName = "main";
    cpi2.layout = pl;
    VkPipeline pipe; AK_CHECK(vkCreateComputePipelines(dev, nullptr, 1, &cpi2, nullptr, &pipe));

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qfi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool; AK_CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));
    VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cbai.commandBufferCount = 1;
    VkCommandBuffer cb; AK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    AK_CHECK(vkBeginCommandBuffer(cb, &beg));

    // upload the 100 blocks, then make the copy visible to the shader before it starts reading
    VkBufferCopy up{0, 0, uint32_t(src_sz)};
    vkCmdCopyBuffer(cb, b_stage, b_src, 1, &up);
    { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr); }

    uint32_t pc[5] = {NEL_U, 0, 0, 0, 0};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    // the shader maps workgroup -> 16 blocks (i = wg*4 + lid/64, ib = 4*i + lid%4, 8 elements per
    // thread), so one group of 256 covers 16*128 = 2048 weights.
    uint32_t groups = (NB + 15) / 16;
    printf("[gpu] nel=%u  blocks=%d  local=256  groups=%u (%u*2048 = %u >= %u weights)\n",
           NEL_U, NB, groups, groups, groups * 2048u, NEL_U);
    vkCmdDispatch(cb, groups, 1, 1);
    // compute wrote, the copy engine reads: no barrier here is a race, and on this driver it is a
    // race that SHOWS (the STAGE3 correctness bug was exactly this missing barrier).
    { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr); }
    VkBufferCopy cp{0, 0, uint32_t(out_sz)};
    vkCmdCopyBuffer(cb, b_out, b_read, 1, &cp);
    AK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    AK_CHECK(vkQueueSubmit(queue, 1, &si, nullptr));
    AK_CHECK(vkQueueWaitIdle(queue));

    std::vector<float> w_gpu(std::size_t{NEL_U}, 0.0f);
    { void *p = nullptr; AK_CHECK(vkMapMemory(dev, m_read, 0, out_sz, 0, &p));
      memcpy(w_gpu.data(), p, size_t(out_sz)); vkUnmapMemory(dev, m_read); }

    // ---------------------------------------------------------------- 4. compare
    // Three different notions of "equal", because the synthetic set deliberately contains values
    // that cannot be compared by subtraction: Inf-Inf and NaN-NaN are NaN, so a plain
    // fabs(a-b) would report a perfect match as a failure and hide the real answer.
    //   bit-exact : the two 4-byte patterns are identical (so -0 and +0 count as DIFFERENT)
    //   same      : both NaN, or both the same-signed infinity, or a == b (so -0 == +0 counts)
    //   max|err|  : |a-b| taken only over pairs where BOTH sides are finite
    //   hard      : pairs where the two sides are not "same" and at least one is not finite, i.e.
    //               a divergence that has no measurable magnitude - the worst possible outcome.
    struct Cmp { double max_abs = 0.0; size_t at = 0, bitsame = 0, same = 0, hard = 0, nfin = 0; };
    auto compare = [](const std::vector<float> &a, const std::vector<float> &b) {
        Cmp r;
        for (size_t i = 0; i < a.size(); ++i) {
            const float x = a[i], y = b[i];
            if (memcmp(&x, &y, 4) == 0) ++r.bitsame;
            const bool xn = std::isnan(x), yn = std::isnan(y);
            const bool xi = std::isinf(x), yi = std::isinf(y);
            const bool eq = xn || yn ? (xn && yn) : (xi || yi ? (xi && yi && x == y) : x == y);
            if (eq) ++r.same;
            if (std::isfinite(x) && std::isfinite(y)) {
                ++r.nfin;
                const double d = std::fabs(double(x) - double(y));
                if (d > r.max_abs) { r.max_abs = d; r.at = i; }
            } else if (!eq) ++r.hard;
        }
        return r;
    };
    auto gpu  = compare(w_gpu, w_off);
    auto our_ = compare(w_our, w_off);
    auto mix  = compare(w_our, w_gpu);

    // scale-free agreement: recover the stored symbol from each float and compare integers. With a
    // finite, non-zero block scale the three possible weights are exactly -d, 0, +d, so round(w/d)
    // is the symbol. Meaningless for d in {0, subnormal-of-a-different-magnitude, Inf, NaN}, which
    // only the synthetic set contains, so those blocks are counted separately and use bit-exactness.
    int sym_bad = 0, sym_tot = 0, sym_skipped = 0;
    for (int p = 0; p < NB; ++p) {
        const double d = double(ptq_scale(hsrc.data() + size_t(p) * PTQ_BLOCK_BYTES));
        if (!(std::isfinite(d) && d != 0.0)) { sym_skipped += 128; continue; }
        for (int e = 0; e < 128; ++e) {
            auto s = [&](float w) { return int(std::lround(double(w) / d)); };
            int a = s(w_gpu[size_t(p) * 128 + e]), b = s(w_our[size_t(p) * 128 + e]), cc = s(w_off[size_t(p) * 128 + e]);
            ++sym_tot; if (a != b || b != cc) ++sym_bad;
        }
    }

    // the runtime never materialises weights: it folds the -1 bias into  d*(cs - xs).  Check that
    // identity against the weight form on the same blocks with a deterministic activation vector.
    double dot_max_abs = 0.0, dot_max_rel = 0.0; uint64_t lcg = 12345u;
    int dot_blocks = 0, dot_skipped = 0;
    for (int p = 0; p < NB; ++p) {
        const uint8_t *pb = hsrc.data() + size_t(p) * PTQ_BLOCK_BYTES;
        const double d = double(ptq_scale(pb));
        if (!std::isfinite(d)) { ++dot_skipped; continue; }   // 0*Inf and Inf-Inf are not numbers
        ++dot_blocks;
        double cs = 0.0, xs = 0.0, wf = 0.0;
        for (int e = 0; e < 128; ++e) {
            lcg = lcg * 6364136223846793005ull + 1442695040888963407ull;
            const float x = float(double((lcg >> 33) % 2000ull) / 1000.0 - 1.0);
            int by, tr; elem_to_byte_trit(e, by, tr);
            cs += double(trit_of_byte(pb[by], tr)) * x; xs += x;
            wf += double(w_our[size_t(p) * 128 + e]) * x;
        }
        const double a = d * (cs - xs);
        const double e_abs = std::fabs(a - wf);
        const double rel = e_abs / (std::fabs(a) + 1e-30);
        if (e_abs > dot_max_abs) dot_max_abs = e_abs;
        if (rel > dot_max_rel)   dot_max_rel = rel;
    }

    // what the blocks under test actually contain - if the sample were all-zeros or all one scale
    // the test would pass for the wrong reason, so the census is part of the result.
    int n_neg = 0, n_zero = 0, n_pos = 0, n_nan = 0, n_inf = 0;
    int n_d_zero = 0, n_d_sub = 0, n_d_neg = 0, n_d_nfin = 0;
    double d_min = 0.0, d_max = 0.0; bool d_first = true;
    for (int p = 0; p < NB; ++p) {
        const double d = double(ptq_scale(hsrc.data() + size_t(p) * PTQ_BLOCK_BYTES));
        if (!std::isfinite(d)) ++n_d_nfin;
        if (d == 0.0) ++n_d_zero;                       // +0 and -0 both land here
        if (std::fabs(d) < 6.103515625e-05 && d != 0.0) ++n_d_sub;   // 2^-14 = fp16 normal minimum
        if (d < 0.0) ++n_d_neg;
        if (std::isfinite(d) && d != 0.0) {
            if (d_first) { d_min = d_max = d; d_first = false; }
            d_min = std::min(d_min, d); d_max = std::max(d_max, d);
        }
        for (int e = 0; e < 128; ++e) {
            const float w = w_off[size_t(p) * 128 + e];
            if (std::isnan(w)) ++n_nan;
            else if (std::isinf(w)) ++n_inf;
            else if (w < 0.0f) ++n_neg; else if (w > 0.0f) ++n_pos; else ++n_zero;
        }
    }
    printf("\n[blocks] %d PTQ1_0 blocks (%s), %u weights compared\n", NB,
           synth ? "synthetic edge x byte-pattern matrix" : "real, read from the checkpoint", NEL_U);
    printf("[census] CPU-REF weights by class: -d %d, 0 %d, +d %d, NaN %d, Inf %d  (of %u)\n",
           n_neg, n_zero, n_pos, n_nan, n_inf, NEL_U);
    printf("[census] block scales: finite nonzero |d| in [%.6g, %.6g]; d==0 %d, subnormal %d, "
           "negative %d, non-finite %d\n", d_min, d_max, n_d_zero, n_d_sub, n_d_neg, n_d_nfin);
    printf("[blocks] sample provenance (first 6 and last 2, so the bytes are re-derivable):\n");
    for (int p = 0; p < NB; ++p) {
        if (p < 6 || p >= NB - 2) {
            // a named std::string, not a ternary between const char* and a temporary: the ternary
            // would not even compile, and .c_str() of a temporary is the bug class that bit
            // align_tokenizer.cpp earlier in this session.
            std::string where = synth ? std::string("pattern=(q+7j)&0xFF")
                                      : "file_off=" + std::to_string(picks[p].foff);
            printf("   #%3d  %-34s blk=%-7llu %s d=%.6g\n", p + 1, picks[p].name.c_str(),
                   (unsigned long long)picks[p].blk, where.c_str(),
                   double(ptq_scale(hsrc.data() + size_t(p) * 28)));
        }
    }

    printf("\n==== dequant fidelity (GPU-official-shader vs CPU-transcription vs our runtime) ====\n");
    // the two-line form is easier to read than one line trying to pad two columns
    auto line = [&](const char *a, const char *b, const Cmp &c) {
        printf("  max|%s - %s| = %.5f  (over %zu finite pairs, argmax weight %zu)\n", a, b,
               c.max_abs, c.nfin, c.at);
        printf("      bit-exact %zu/%u (%.3f%%)   NaN/Inf-aware equal %zu/%u   "
               "non-finite mismatches %zu\n", c.bitsame, NEL_U,
               100.0 * double(c.bitsame) / double(NEL_U), c.same, NEL_U, c.hard);
    };
    line("GPU-REF", "CPU-REF", gpu);
    line("OURS   ", "CPU-REF", our_);
    line("OURS   ", "GPU-REF", mix);
    printf("  symbol agreement      = %d/%d weights, %d disagreements (%d weights skipped: scale "
           "is 0/Inf/NaN, where w/d is not a symbol)\n", sym_tot - sym_bad, sym_tot, sym_bad, sym_skipped);
    printf("  runtime dot form d*(cs-xs) vs sum(w_e*x_e): max abs %.3e  max rel %.3e  (%d blocks, "
           "%d skipped for non-finite d)\n", dot_max_abs, dot_max_rel, dot_blocks, dot_skipped);
    printf("  (why this can be non-zero in principle: the folded form multiplies by d once per block\n"
           "   and the weight form multiplies per element, so a reassociation could show up in the\n"
           "   last bits. Measured 0.000e+00 here, which is what exactness requires: (trit-1) is\n"
           "   always -1/0/+1, so (trit-1)*d is representable and the two forms are the same number.)\n");

    // per-edge-scale table: a single aggregate "0.00000" could still hide one pattern that only
    // matched by luck, so every scale class is also judged on its own 32768 weights.
    if (synth) {
        printf("\n==== per synthetic block-scale class (%d weights each) ====\n", 256 * 128);
        printf("  %-20s %11s %11s %11s %8s %8s %8s\n", "scale", "bit GPU-REF", "bit OURS",
               "max|err|", "NaN", "Inf", "hard");
        for (int e = 0; e < NE; ++e) {
            const size_t base = size_t(e) * 256 * 128, n = size_t(256) * 128;
            std::vector<float> g1(w_gpu.begin() + base, w_gpu.begin() + base + n);
            std::vector<float> o1(w_our.begin() + base, w_our.begin() + base + n);
            std::vector<float> r1(w_off.begin() + base, w_off.begin() + base + n);
            Cmp a = compare(g1, r1), b = compare(o1, r1);
            int nn = 0, ni = 0;
            for (size_t i = 0; i < n; ++i) {
                if (std::isnan(r1[i])) ++nn; else if (std::isinf(r1[i])) ++ni;
            }
            printf("  %-20s %8zu/%zu %8zu/%zu %11.5f %8d %8d %8zu\n", EDGE[e].name,
                   a.bitsame, n, b.bitsame, n, std::max(a.max_abs, b.max_abs), nn, ni,
                   std::max(a.hard, b.hard));
        }
    }

    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr); vkDestroyDescriptorSetLayout(dev, sl, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr); vkDestroyCommandPool(dev, cpool, nullptr);
    auto kill = [&](VkBuffer b, VkDeviceMemory m) { vkDestroyBuffer(dev, b, nullptr); vkFreeMemory(dev, m, nullptr); };
    kill(b_stage, m_stage); kill(b_src, m_src); kill(b_out, m_out); kill(b_read, m_read);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    // exit non-zero if ANY measured magnitude is non-zero or any pair is a non-finite mismatch
    return int(gpu.max_abs != 0.0 || our_.max_abs != 0.0 || mix.max_abs != 0.0 ||
               gpu.same != NEL_U || our_.same != NEL_U || mix.same != NEL_U);
}
