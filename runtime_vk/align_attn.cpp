// align_attn.cpp - layer 3 of the faithful-reproduction checks: does OUR attention core produce the
// numbers the official engine's arithmetic means?
//
// Three legs, all measured, and they answer different questions:
//
//   MIRROR  attn_forward<true>() from attn_vk.h - fp32, with the fp16 roundings the official graph
//           provably has (the K/V cache is fp16; ggml's mixed-type MUL_MAT makes Q and the
//           probabilities fp16 operands too). The GPU kernels run the same formulas in the same
//           order. This is the alignment the boss asked for: kernel vs reference, printed as
//           max|err| per stage plus a bit-exact count.
//   IDEAL   attn_forward_ideal() in float64 with a rounding bitmask, so each fp16 loss can be
//           CHARGED separately instead of argued. mode 0 is the arithmetic floor of this layer.
//   OFFICIAL runs/dbg_*.log: the numbers the fork itself printed. Those checks live in
//           tools/rope_fit.py and tools/attn_softmax_check.py because they need the dumps.
//
// Shapes and constants are the real layer-3 ones, read out of the checkpoint by bonsai_gguf.h:
// head_dim 256, 16 q heads, 4 kv heads (GQA 4), n_rot 64, rms eps float(1e-6f), freq_base 1e7, and
// the two real per-head norm vectors blk.3.attn_q_norm.weight / blk.3.attn_k_norm.weight. The
// projected Q/K/V inputs are synthetic (a deterministic xorshift): the official printer elides any
// axis longer than 6, so nobody can hand us the real 256-wide Q rows - what is NOT synthetic here is
// the norm weights, the mask semantics, the scale, and the cache dtype.
//
// What the test actually asserts (each one maps to a way this could be wrong):
//   A  kernel vs mirror, per stage: q16/kc/vc as BIT patterns, score/prob/pregate/gated as max|err|
//      + bit-exact fraction. The fp16 arena is the interesting one: a packing difference would show
//      up here before it poisoned the scores.
//   B  causal fill: every probability outside the live prefix is exactly 0.0, and a -INF placed
//      INSIDE the live prefix must also give exactly 0.0 with the row renormalised.
//   C  rope structure: the position-0 row must come out of the card equal to the norm-only value
//      (cos 0 = 1, sin 0 = 0) - an independent sign/angle tripwire; dims >= n_rot pass through;
//      and the one-shot powf() this project uses is compared against the fork's two-step
//      theta_scale, so "we rewrote the frequency law" is a measured difference, not a claim.
//   D  GQA: with every q head given the same input, heads 0..3 must agree with each other, heads
//      4..7 must agree with each other, and the two groups must differ (they read kv heads 0 and 1).
//   E  prefill vs decode: fill the arena one token at a time and recompute the last token - a
//      bit-identical result is what proves the cache round-trip and the arena indexing.
//   F  fp16 packer: packHalf2x16() on this card vs our host f32_to_f16_bits(), over normals,
//      subnormals, exact rounding ties and signed zeros.
//
//   build: runtime_vk/build_mingw.bat
//   run:   align_attn.exe [seq] [cap]        (default 8 256; cap <= 256 and head_dim <= 256 are
//          required by the one-workgroup-per-row layout, and the program checks both)

#include "attn_vk.h"
#include "bonsai_gguf.h"

#include <vulkan/vulkan.h>

#include <chrono>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define AK_CHECK(x)                                                                          \
    do {                                                                                     \
        VkResult _r = (x);                                                                   \
        if (_r != VK_SUCCESS) {                                                              \
            fprintf(stderr, "Vulkan failure %d at %s:%d\n", (int)_r, __FILE__, __LINE__);     \
            exit(1);                                                                         \
        }                                                                                    \
    } while (0)

// ------------------------------------------------------------------ tiny helpers
static std::vector<char> read_shader(const char *rel) {
    const std::vector<std::string> tries = { rel, std::string("../") + rel };
    for (const std::string &s : tries) {
        FILE *f = fopen(s.c_str(), "rb");
        if (!f) continue;
        fseek(f, 0, SEEK_END); const long n = ftell(f); fseek(f, 0, SEEK_SET);
        std::vector<char> b; b.resize(size_t(n));      // NOT "vector<char> b(size_t(n))": that is the
                                                       // most vexing parse and declares a function
        if (n > 0 && fread(b.data(), 1, size_t(n), f) != size_t(n)) { fclose(f); return b; }
        fclose(f);
        return b;
    }
    fprintf(stderr, "cannot read %s - run build_mingw.bat first\n", rel);
    exit(1);
}

static void file_seek(FILE *f, uint64_t off) {
#ifdef _WIN32
    if (_fseeki64(f, (__int64)off, SEEK_SET) != 0) { fprintf(stderr, "seek\n"); exit(1); }
#else
    if (fseeko(f, (off_t)off, SEEK_SET) != 0) { fprintf(stderr, "seek\n"); exit(1); }
#endif
}

// deterministic, seed-fixed: every number in this report has to be reproducible from the command line
struct Rng {
    uint64_t s;
    explicit Rng(uint64_t seed) : s(seed ? seed : 0x9E3779B97F4A7C15ull) {}
    uint64_t next() { s ^= s << 13; s ^= s >> 7; s ^= s << 17; return s; }
    float unit() { return float(next() >> 11) * (1.0f / 9007199254740992.0f); }   // [0,1)
    // roughly N(0, sigma): the sum of three uniforms is a triangular-ish bell and, unlike a
    // Box-Muller, cannot produce an inf when the draw hits exactly 0.
    float gauss(float sigma) { return sigma * ((unit() + unit() + unit()) * 1.1547f - 1.7320508f); }
};

struct Cmp {
    double max_abs = 0.0; size_t at = 0, bitsame = 0, n = 0, nfin = 0, hard = 0;
    double max_rel = 0.0;
    // the reference tensor's own largest magnitude, so every error can also be read as a fraction of
    // full scale. max rel alone is misleading: one flipped fp16 bit in an operand changes a score that
    // happens to sit near zero by 1e-2 RELATIVE and by 3e-5 of the largest score in the run, and only
    // the second number is comparable with what fp16 storage itself costs (leg G).
    double ref_max = 0.0;
};
// NaN/Inf-aware, same three notions of "equal" as align_dequant: subtracting two NaNs is NaN, which
// would score a perfect match as a failure.
static Cmp compare_f(const std::vector<float> &a, const std::vector<float> &b) {
    Cmp r; r.n = a.size();
    for (size_t i = 0; i < a.size(); ++i) {
        const float x = a[i], y = b[i];
        const bool bx = std::isnan(x) || std::isinf(x), by = std::isnan(y) || std::isinf(y);
        uint32_t ux, uy; memcpy(&ux, &x, 4); memcpy(&uy, &y, 4);
        if (ux == uy) ++r.bitsame;
        if (!bx && !by) {
            ++r.nfin;
            const double e = std::fabs(double(x) - double(y));
            if (e > r.max_abs) { r.max_abs = e; r.at = i; }
            r.ref_max = std::fmax(r.ref_max, std::fabs(double(x)));
            const double den = std::fmax(std::fabs(double(x)), 1e-30);
            r.max_rel = std::fmax(r.max_rel, e / den);
        } else if (!(std::isnan(x) && std::isnan(y)) && !(x == y)) {
            ++r.hard;                                   // one side finite, or two different NaN/Inf
        }
    }
    return r;
}
static size_t compare_bits(const std::vector<uint32_t> &a, const std::vector<uint32_t> &b,
                           size_t &mismatch) {
    mismatch = 0;
    for (size_t i = 0; i < a.size(); ++i) if (a[i] != b[i]) ++mismatch;
    return a.size() - mismatch;
}

static void show(const char *name, const Cmp &c, double scale_ref) {
    printf("  %-22s n=%-9zu max|err|=%.8f  at=%-8zu  bit-exact=%7.4f%%  max rel=%.3e  hard=%zu",
           name, c.n, c.max_abs, c.at, 100.0 * double(c.bitsame) / double(std::max<size_t>(c.n, 1)),
           c.max_rel, c.hard);
    if (c.ref_max > 0.0)
        printf("  =%.3e of full scale (%.3f)", c.max_abs / c.ref_max, c.ref_max);
    if (scale_ref > 0.0) printf("   (rms scale of the data: %.3f)", scale_ref);
    printf("\n");
}

// ------------------------------------------------------------------ Vulkan scaffolding
struct Buf {
    VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; uint64_t bytes = 0;
};
struct Ctx {
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    VkPhysicalDeviceMemoryProperties mp{};
    uint32_t qfi = 0;
    Buf stage, read;                                  // one of each, reused
    std::vector<Buf> all;
};
static uint32_t find_type(const Ctx &c, VkMemoryPropertyFlags want) {
    for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i)
        if ((c.mp.memoryTypes[i].propertyFlags & want) == want) return i;
    fprintf(stderr, "no memory type 0x%X\n", unsigned(want)); exit(1);
}
static Buf make_buf(Ctx &c, uint64_t size, VkBufferUsageFlags use, VkMemoryPropertyFlags prop) {
    Buf o; o.bytes = size ? size : 16;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = o.bytes; bi.usage = use; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    AK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &o.b));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, o.b, &mr);
    VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    mai.allocationSize = mr.size; mai.memoryTypeIndex = find_type(c, prop);
    AK_CHECK(vkAllocateMemory(c.dev, &mai, nullptr, &o.m));
    AK_CHECK(vkBindBufferMemory(c.dev, o.b, o.m, 0));
    c.all.push_back(o);
    return o;
}
static const VkMemoryPropertyFlags HOST =
    VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
static const VkMemoryPropertyFlags DEV = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
static const VkBufferUsageFlags SS =
    VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;

static VkCommandPool g_pool = VK_NULL_HANDLE;   // created once in main, used by every helper below

// host -> device through a staging buffer, exactly like the runtime's arena upload
static void upload(Ctx &c, const Buf &dst, const void *src, uint64_t bytes) {
    // The ring in Ctx is sized from the attention tensors, but leg T's math probe uploads the whole
    // exponent set in one go, which is bigger at long seq. The old guard printed "stage too small" and
    // called exit(1): a harness that dies when the shape grows is a harness that hides bugs, so an
    // oversized payload now gets its own one-shot staging buffer. It stays in c.all and dies with the
    // device - deliberate, because this is a test harness that runs once and exits.
    const Buf *stg = &c.stage;
    Buf one{};
    if (bytes > c.stage.bytes) { one = make_buf(c, bytes, SS, HOST); stg = &one; }
    void *p = nullptr; AK_CHECK(vkMapMemory(c.dev, stg->m, 0, bytes, 0, &p));
    // `stg->m`, not `c.stage.m`: the oversized path above can leave `stg` pointing at the one-shot
    // buffer, and unmapping the ring instead of the buffer we actually wrote is undefined behaviour
    // that this harness paid for. Measured symptom, before the fix: at seq=64 the probe's 1.19 MB
    // upload took the one-shot path and leg T then reported the card returning 1/sqrt(0.755595) =
    // 76.59 (host 1.15) and 1/0.755595 = -5865.75 (host 1.32) - a DRIVERS-ARE-BROKEN conclusion that
    // was really the harness handing the shader a stale tail. seq=8/32 never hit the path, which is
    // why the same probe looked sane there. The round-trip check right below is the tripwire.
    memcpy(p, src, size_t(bytes)); vkUnmapMemory(c.dev, stg->m);
    VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = g_pool; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount = 1;
    VkCommandBuffer cb; AK_CHECK(vkAllocateCommandBuffers(c.dev, &a, &cb));
    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    AK_CHECK(vkBeginCommandBuffer(cb, &beg));
    VkBufferCopy cp{0, 0, uint32_t(bytes)};
    vkCmdCopyBuffer(cb, stg->b, dst.b, 1, &cp);
    { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr); }
    AK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    AK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr));
    AK_CHECK(vkQueueWaitIdle(c.queue));
    vkFreeCommandBuffers(c.dev, g_pool, 1, &cb);
}
static void readback(Ctx &c, const Buf &src, void *dst, uint64_t bytes) {
    // The same one-shot rule upload() learned the hard way, and the read side had NO guard at all:
    // leg T's seq=64 probe reads back 1.19 MB per tensor through a 1.05 MB ring, so vkCmdCopyBuffer
    // wrote past the end of the destination and the memcpy read past the end of the mapping. That is
    // where the "card returns 1/0.755595 = -5865.75" readings came from - a harness overrun being
    // reported as a driver defect. Any buffer bigger than the ring now gets its own allocation.
    const Buf *rd = &c.read;
    Buf one{};
    if (bytes > c.read.bytes) {
        one = make_buf(c, bytes, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
        rd = &one;
    }
    VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = g_pool; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount = 1;
    VkCommandBuffer cb; AK_CHECK(vkAllocateCommandBuffers(c.dev, &a, &cb));
    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    AK_CHECK(vkBeginCommandBuffer(cb, &beg));
    { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr); }
    VkBufferCopy cp{0, 0, uint32_t(bytes)};
    vkCmdCopyBuffer(cb, src.b, rd->b, 1, &cp);
    AK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    AK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr));
    AK_CHECK(vkQueueWaitIdle(c.queue));
    void *p = nullptr; AK_CHECK(vkMapMemory(c.dev, rd->m, 0, bytes, 0, &p));
    memcpy(dst, p, size_t(bytes)); vkUnmapMemory(c.dev, rd->m);
    vkFreeCommandBuffers(c.dev, g_pool, 1, &cb);
}
// One compute pipeline plus the descriptor set that points at its buffers.
struct Pipe {
    VkPipeline pipe = VK_NULL_HANDLE;
    VkPipelineLayout pl = VK_NULL_HANDLE;
    VkDescriptorSet ds = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    uint32_t push_bytes = 0;
};
static Pipe make_pipe(Ctx &c, const char *spv_path, const std::vector<Buf *> &bufs,
                      uint32_t push_bytes, uint32_t *out_local_size) {
    Pipe P; P.push_bytes = push_bytes;
    std::vector<VkDescriptorSetLayoutBinding> bs(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        bs[i].binding = uint32_t(i); bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = uint32_t(bs.size()); sli.pBindings = bs.data();
    VkDescriptorSetLayout sl; AK_CHECK(vkCreateDescriptorSetLayout(c.dev, &sli, nullptr, &sl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, push_bytes};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &sl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    AK_CHECK(vkCreatePipelineLayout(c.dev, &pli, nullptr, &P.pl));
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, uint32_t(bufs.size())};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
    AK_CHECK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &P.pool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = P.pool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &sl;
    AK_CHECK(vkAllocateDescriptorSets(c.dev, &dsai, &P.ds));
    std::vector<VkDescriptorBufferInfo> dbi(bufs.size());
    std::vector<VkWriteDescriptorSet> wr(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        dbi[i] = { bufs[i]->b, 0, bufs[i]->bytes };
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = P.ds;
        wr[i].dstBinding = uint32_t(i); wr[i].descriptorCount = 1;
        wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(c.dev, uint32_t(wr.size()), wr.data(), 0, nullptr);
    auto code = read_shader(spv_path);
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size(); mi.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule sm; AK_CHECK(vkCreateShaderModule(c.dev, &mi, nullptr, &sm));
    VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
    cpi.layout = P.pl;
    AK_CHECK(vkCreateComputePipelines(c.dev, nullptr, 1, &cpi, nullptr, &P.pipe));
    vkDestroyShaderModule(c.dev, sm, nullptr);
    vkDestroyDescriptorSetLayout(c.dev, sl, nullptr);
    if (out_local_size) *out_local_size = 256;
    return P;
}
// dispatch, wait, and report the wall time it took (submit + kernel + fence, this driver's shape)
static double run(Ctx &c, Pipe &P, const void *pc, uint32_t groups) {
    VkCommandBufferAllocateInfo a{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    a.commandPool = g_pool; a.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; a.commandBufferCount = 1;
    VkCommandBuffer cb; AK_CHECK(vkAllocateCommandBuffers(c.dev, &a, &cb));
    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    AK_CHECK(vkBeginCommandBuffer(cb, &beg));
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, P.pl, 0, 1, &P.ds, 0, nullptr);
    if (P.push_bytes) vkCmdPushConstants(cb, P.pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, P.push_bytes, pc);
    vkCmdDispatch(cb, groups, 1, 1);
    { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
      mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT | VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr); }
    AK_CHECK(vkEndCommandBuffer(cb));
    auto t0 = std::chrono::steady_clock::now();
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    AK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr));
    AK_CHECK(vkQueueWaitIdle(c.queue));
    auto t1 = std::chrono::steady_clock::now();
    vkFreeCommandBuffers(c.dev, g_pool, 1, &cb);
    return std::chrono::duration<double, std::milli>(t1 - t0).count();
}

// ------------------------------------------------------------------ the push-constant blocks
struct NRPush { uint32_t seq, nh, nkv, hd, nrot, cap; float eps, freq_base; };      // 32 B
struct QSPush { uint32_t seq, nh, nkv, hd, cap, round_probs; float scale, pad0; };   // 32 B
struct PVPush { uint32_t seq, nh, nkv, hd, cap, round_probs; float pad0, pad1; };    // 32 B
struct PackPush { uint32_t n, pad0, pad1, pad2; };                                   // 16 B

// ------------------------------------------------------------------ one attention pass on the card
struct Gpu {
    Ctx c;
    Buf q_in, k_in, v_in, gate, qn_w, kn_w, pos, mask;
    Buf q16, kc, vc, score, prob, p16, pregate, gated;
    Buf qf, kf;                       // the fp32 debug stream of attn_norm_rope.comp (stage R)
    Pipe nr, qs, pv;
};

static void dispatch_all(Gpu &G, const AttnCfg &cfg, const AttnBuffers &b, bool round_probs,
                         double *t_nr, double *t_qs, double *t_pv) {
    NRPush p1{ uint32_t(cfg.seq), uint32_t(cfg.nh), uint32_t(cfg.nkv), uint32_t(cfg.hd),
              uint32_t(cfg.nrot), uint32_t(cfg.cap), cfg.eps, cfg.freq_base };
    upload(G.c, G.q_in,  b.q_in.data(),  b.q_in.size() * 4);
    upload(G.c, G.k_in,  b.k_in.data(),  b.k_in.size() * 4);
    upload(G.c, G.v_in,  b.v_in.data(),  b.v_in.size() * 4);
    upload(G.c, G.gate,  b.gate_in.data(), b.gate_in.size() * 4);
    upload(G.c, G.pos,   b.pos.data(),   b.pos.size() * 4);
    upload(G.c, G.mask,  b.mask.data(),  b.mask.size() * 4);
    // The timing outputs are OPTIONAL; the dispatches are not. Writing this as "if (t_nr) *t_nr = run()"
    // looked harmless and was not: every leg that passed null timings silently ran against the PREVIOUS
    // leg's readback, which is how the GQA test came out "all 192 same-kv-head pairs disagree".
    double a;
    a = run(G.c, G.nr, &p1, uint32_t(cfg.seq * (cfg.nh + cfg.nkv))); if (t_nr) *t_nr = a;

    QSPush p2{ uint32_t(cfg.seq), uint32_t(cfg.nh), uint32_t(cfg.nkv), uint32_t(cfg.hd),
               uint32_t(cfg.cap), round_probs ? 1u : 0u, cfg.scale, 0.0f };
    a = run(G.c, G.qs, &p2, uint32_t(cfg.seq * cfg.nh)); if (t_qs) *t_qs = a;

    PVPush p3{ uint32_t(cfg.seq), uint32_t(cfg.nh), uint32_t(cfg.nkv), uint32_t(cfg.hd),
               uint32_t(cfg.cap), round_probs ? 1u : 0u, 0.0f, 0.0f };
    a = run(G.c, G.pv, &p3, uint32_t(cfg.seq * cfg.nh)); if (t_pv) *t_pv = a;
}

static void fetch_all(Gpu &G, const AttnCfg &cfg, AttnBuffers &out) {
    const size_t hw = size_t(cfg.seq) * cfg.nh * (cfg.hd / 2);
    const size_t cw = size_t(cfg.cap) * cfg.nkv * (cfg.hd / 2);
    out.q16.assign(hw, 0u); out.kc.assign(cw, 0u); out.vc.assign(cw, 0u);
    readback(G.c, G.q16, out.q16.data(), hw * 4);
    readback(G.c, G.kc,  out.kc.data(),  cw * 4);
    readback(G.c, G.vc,  out.vc.data(),  cw * 4);
    const size_t sw = size_t(cfg.seq) * cfg.nh * cfg.cap;
    const size_t pw = size_t(cfg.seq) * cfg.nh * (cfg.cap / 2);
    const size_t ow = size_t(cfg.seq) * cfg.nh * cfg.hd;
    out.score.assign(sw, 0.f); out.prob.assign(sw, 0.f); out.p16.assign(pw, 0u);
    out.pregate.assign(ow, 0.f); out.gated.assign(ow, 0.f);
    readback(G.c, G.score,   out.score.data(),   sw * 4);
    readback(G.c, G.prob,    out.prob.data(),    sw * 4);
    readback(G.c, G.p16,     out.p16.data(),     pw * 4);
    readback(G.c, G.pregate, out.pregate.data(), ow * 4);
    readback(G.c, G.gated,   out.gated.data(),   ow * 4);
}

// ------------------------------------------------------------------ main
int main(int argc, char **argv) {
    const int SEQ = argc > 1 ? atoi(argv[1]) : 8;
    const int CAP = argc > 2 ? atoi(argv[2]) : 256;
    const char *MODEL = getenv("BONSAI_MODEL") ? getenv("BONSAI_MODEL")
                                               : "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    printf("[cfg]  seq=%d cap=%d  model=%s\n", SEQ, CAP, MODEL);

    // ---- the real metadata and the real per-head norm vectors, out of the checkpoint
    GgufHeader gh; bool have_model = gguf_load(MODEL, gh);
    AttnCfg cfg; cfg.seq = SEQ; cfg.cap = CAP;
    if (have_model) {
        auto get = [&](const char *k, double dflt) {
            auto it = gh.kv_f.find(k); if (it != gh.kv_f.end()) return it->second;
            auto iit = gh.kv_i.find(k); if (iit != gh.kv_i.end()) return double(iit->second);
            return dflt;
        };
        cfg.hd        = int(get("qwen35.attention.key_length",        cfg.hd));
        cfg.nh        = int(get("qwen35.attention.head_count",        cfg.nh));
        cfg.nkv       = int(get("qwen35.attention.head_count_kv",     cfg.nkv));
        cfg.nrot      = int(get("qwen35.rope.dimension_count",        cfg.nrot));
        cfg.eps       = float(get("qwen35.attention.layer_norm_rms_epsilon", cfg.eps));
        cfg.freq_base = float(get("qwen35.rope.freq_base",            cfg.freq_base));
        cfg.scale     = 1.0f / std::sqrt(float(cfg.hd));      // MEASURED to be the value in SOFT_MAX
        printf("[gguf] head_dim=%d n_head=%d n_head_kv=%d n_rot=%d eps=%.9g freq_base=%g  "
               "scale=1/sqrt(hd)=%.6f\n", cfg.hd, cfg.nh, cfg.nkv, cfg.nrot, double(cfg.eps),
               double(cfg.freq_base), double(cfg.scale));
    } else {
        printf("[gguf] %s not readable - using the same constants typed from gguf_meta.txt\n", MODEL);
    }
    if (cfg.hd > 256 || cfg.cap > 256) {
        fprintf(stderr, "one workgroup row holds hd<=256 lanes and cap<=256 positions; got hd=%d cap=%d\n",
                cfg.hd, cfg.cap);
        return 1;
    }
    if (cfg.seq > cfg.cap) { fprintf(stderr, "seq > cap\n"); return 1; }

    AttnBuffers base; base.alloc(cfg);
    {
        // real norm weights when the file is there, ones when it is not (the algebra does not change)
        const GgufTensor *tq = have_model ? gh.find("blk.3.attn_q_norm.weight") : nullptr;
        const GgufTensor *tk = have_model ? gh.find("blk.3.attn_k_norm.weight") : nullptr;
        if (tq && tk && tq->dtype == 0 && tk->dtype == 0 && int(tq->d[0]) == cfg.hd) {
            FILE *f = fopen(MODEL, "rb");
            std::vector<uint8_t> bq(size_t(cfg.hd) * 4), bk(size_t(cfg.hd) * 4);
            file_seek(f, gh.data_start + tq->off);
            if (fread(bq.data(), 1, bq.size(), f) != bq.size()) { fprintf(stderr, "short read\n"); return 1; }
            file_seek(f, gh.data_start + tk->off);
            if (fread(bk.data(), 1, bk.size(), f) != bk.size()) { fprintf(stderr, "short read\n"); return 1; }
            fclose(f);
            memcpy(base.qn_w.data(), bq.data(), bq.size());
            memcpy(base.kn_w.data(), bk.data(), bk.size());
            double lo = 1e30, hi = -1e30;
            for (int d = 0; d < cfg.hd; ++d) { lo = std::fmin(lo, base.qn_w[d]);
                                               hi = std::fmax(hi, base.qn_w[d]); }
            printf("[w]    blk.3.attn_q_norm/attn_k_norm loaded from the file: qn range %.4f..%.4f, "
                   "qn[0]=%.6f kn[0]=%.6f\n", lo, hi, base.qn_w[0], base.kn_w[0]);
        } else {
            printf("[w]    norm weights = 1.0 (tensors unavailable)\n");
        }
    }
    // deterministic synthetic projections. sigma 1.6 on Q/K puts the raw dot products of 256 terms
    // in the same decade as the printed official kq (~1e2), so the exp() range the kernel sees is
    // realistic rather than a toy.
    {
        Rng r(0xB0A5u);
        for (auto &x : base.q_in)   x = r.gauss(1.6f);
        for (auto &x : base.k_in)   x = r.gauss(1.6f);
        for (auto &x : base.v_in)   x = r.gauss(1.1f);
        for (auto &x : base.gate_in) x = r.gauss(0.9f);
        for (int t = 0; t < cfg.seq; ++t) { base.pos[t] = uint32_t(t); base.pos2tok[t] = t; }
        for (auto &m : base.mask) m = 0.0f;          // nothing masked but causality
    }

    // ---- Vulkan up
    Gpu G;
    {
        VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
        ai.apiVersion = VK_API_VERSION_1_2;
        VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
        ici.pApplicationInfo = &ai;
        const char *ext[] = {"VK_KHR_get_physical_device_properties2"};
        ici.enabledExtensionCount = 1; ici.ppEnabledExtensionNames = ext;
        VkInstance inst; AK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
        uint32_t nd = 0; vkEnumeratePhysicalDevices(inst, &nd, nullptr);
        std::vector<VkPhysicalDevice> ps(nd); vkEnumeratePhysicalDevices(inst, &nd, ps.data());
        VkPhysicalDevice phys = VK_NULL_HANDLE;
        for (auto p : ps) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
                            if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pp.vendorID == 0x1002)
                                phys = p; }
        if (phys == VK_NULL_HANDLE) phys = ps[0];
        VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
        vkGetPhysicalDeviceMemoryProperties(phys, &G.c.mp);
        printf("[dev]  %s\n", props.deviceName);
        uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
        std::vector<VkQueueFamilyProperties> qf(nq);
        vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
        uint32_t qfi = 0;
        for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
        G.c.qfi = qfi;
        float prio = 1.0f;
        VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
        qi.queueFamilyIndex = qfi; qi.queueCount = 1; qi.pQueuePriorities = &prio;
        VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
        di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
        AK_CHECK(vkCreateDevice(phys, &di, nullptr, &G.c.dev));
        vkGetDeviceQueue(G.c.dev, qfi, 0, &G.c.queue);
        VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpci.queueFamilyIndex = qfi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        AK_CHECK(vkCreateCommandPool(G.c.dev, &cpci, nullptr, &g_pool));
    }

    const size_t hw = size_t(cfg.seq) * cfg.nh * (cfg.hd / 2);
    const size_t cw = size_t(cfg.cap) * cfg.nkv * (cfg.hd / 2);
    const size_t sw = size_t(cfg.seq) * cfg.nh * cfg.cap;
    const size_t pw = size_t(cfg.seq) * cfg.nh * (cfg.cap / 2);
    const size_t ow = size_t(cfg.seq) * cfg.nh * cfg.hd;
    G.q_in   = make_buf(G.c, base.q_in.size() * 4,   SS, DEV);
    G.k_in   = make_buf(G.c, base.k_in.size() * 4,   SS, DEV);
    G.v_in   = make_buf(G.c, base.v_in.size() * 4,   SS, DEV);
    G.gate   = make_buf(G.c, base.gate_in.size() * 4, SS, DEV);
    G.qn_w   = make_buf(G.c, cfg.hd * 4, SS, DEV);
    G.kn_w   = make_buf(G.c, cfg.hd * 4, SS, DEV);
    G.pos    = make_buf(G.c, std::max<size_t>(cfg.cap * 4, 64), SS, DEV);
    G.mask   = make_buf(G.c, base.mask.size() * 4, SS, DEV);
    G.q16    = make_buf(G.c, hw * 4, SS, DEV);
    G.kc     = make_buf(G.c, cw * 4, SS, DEV);
    G.vc     = make_buf(G.c, cw * 4, SS, DEV);
    G.score  = make_buf(G.c, sw * 4, SS, DEV);
    G.prob   = make_buf(G.c, sw * 4, SS, DEV);
    G.p16    = make_buf(G.c, pw * 4, SS, DEV);
    G.pregate= make_buf(G.c, ow * 4, SS, DEV);
    G.gated  = make_buf(G.c, ow * 4, SS, DEV);
    const size_t kfw = size_t(cfg.cap) * cfg.nkv * cfg.hd;      // fp32 debug arena
    G.qf     = make_buf(G.c, ow * 4, SS, DEV);
    G.kf     = make_buf(G.c, kfw * 4, SS, DEV);
    // the staging ring has to hold the BIGGEST thing any leg moves, and that is the fp16 arena itself
    // (cap*nkv*hw words), not the projections: leg E uploads a zeroed arena of exactly that size.
    const size_t stage_w = std::max(std::max(std::max(std::max(base.q_in.size(), ow), sw), cw), kfw);
    G.c.stage  = make_buf(G.c, stage_w * 4, VK_BUFFER_USAGE_TRANSFER_SRC_BIT, HOST);
    G.c.read   = make_buf(G.c, stage_w * 4, VK_BUFFER_USAGE_TRANSFER_DST_BIT, HOST);
    upload(G.c, G.qn_w, base.qn_w.data(), cfg.hd * 4);
    upload(G.c, G.kn_w, base.kn_w.data(), cfg.hd * 4);

    std::vector<Buf *> nr_b{ &G.q_in, &G.k_in, &G.v_in, &G.qn_w, &G.kn_w, &G.pos,
                             &G.kc, &G.vc, &G.q16, &G.qf, &G.kf };
    std::vector<Buf *> qs_b{ &G.q16, &G.kc, &G.mask, &G.pos, &G.score, &G.prob, &G.p16 };
    std::vector<Buf *> pv_b{ &G.prob, &G.p16, &G.vc, &G.pos, &G.gate, &G.pregate, &G.gated };
    G.nr = make_pipe(G.c, "shaders/attn_norm_rope.spv",  nr_b, sizeof(NRPush), nullptr);
    G.qs = make_pipe(G.c, "shaders/attn_qk_softmax.spv", qs_b, sizeof(QSPush), nullptr);
    G.pv = make_pipe(G.c, "shaders/attn_pv.spv",         pv_b, sizeof(PVPush), nullptr);

    // ================================================================= F: the fp16 packer
    printf("\n== F. the fp16 packer: the card's packHalf2x16() vs RNE, four columns ==\n");
    {
        const int NP = 4096;                       // 2048 arena words
        std::vector<float> f(NP);
        Rng r(0xF116u);
        for (int i = 0; i < NP; ++i) {
            switch (i % 5) {
                case 0: f[i] = r.gauss(1.0f); break;                        // ordinary normals
                case 1: f[i] = r.gauss(1.0f) * 1e-5f; break;                // subnormal territory
                case 2: {                                                 // exact half-ulp TIES:
                    uint32_t u; float v = 1.0f + float(i) * 1.0e-3f;        //   dropped bits = 1000..0
                    memcpy(&u, &v, 4); u |= 0x00001000u; u &= ~0x0FFFu;     //   -> RNE must look at
                    memcpy(&f[i], &u, 4);                                   //   the mantissa parity
                } break;
                case 3: f[i] = std::copysign(std::ldexp(1.0f, -24) * float(i % 7 + 1),
                                             (i & 4) ? -1.0f : 1.0f);       // subnormals, both signs
                        break;
                default: f[i] = r.gauss(300.0f); break;                     // values near the arena's
            }                                                                 // real magnitudes
        }
        Buf in = make_buf(G.c, NP * 4, SS, DEV);
        Buf raw = make_buf(G.c, (NP / 2) * 4, SS, DEV);
        Buf rne = make_buf(G.c, (NP / 2) * 4, SS, DEV);
        Buf bk  = make_buf(G.c, NP * 4, SS, DEV);
        upload(G.c, in, f.data(), NP * 4);
        Pipe probe = [&] {
            std::vector<Buf *> bs{ &in, &raw, &rne, &bk };
            return make_pipe(G.c, "shaders/f16_pack_probe.spv", bs, sizeof(PackPush), nullptr);
        }();
        PackPush pp{ uint32_t(NP), 0, 0, 0 };
        double ms = run(G.c, probe, &pp, uint32_t(NP / 2 / 256 + 1));
        std::vector<uint32_t> graw(NP / 2), grne(NP / 2); std::vector<float> gback(NP);
        readback(G.c, raw, graw.data(), (NP / 2) * 4);
        readback(G.c, rne, grne.data(), (NP / 2) * 4);
        readback(G.c, bk,  gback.data(), NP * 4);

        size_t h_rne_mism = 0, instr_vs_rne = 0, instr_vs_rtz = 0, first = 0;
        size_t trunc_dir = 0, up_dir = 0;               // does the instruction ever go UP in magnitude
        for (int i = 0; i < NP / 2; ++i) {
            const uint32_t hr = pack2(f[2 * i], f[2 * i + 1]);
            const uint32_t ht = pack2_rtz(f[2 * i], f[2 * i + 1]);
            if (hr != grne[i]) ++h_rne_mism;
            if (hr != graw[i]) {
                ++instr_vs_rne;
                if (!first) first = size_t(i);
                // magnitude comparison of the two halves, RNE vs the instruction
                const double a = std::fabs(double(f16_to_f32(uint16_t(hr & 0xFFFFu)))) +
                                 std::fabs(double(f16_to_f32(uint16_t(hr >> 16))));
                const double b = std::fabs(double(f16_to_f32(uint16_t(graw[i] & 0xFFFFu)))) +
                                 std::fabs(double(f16_to_f32(uint16_t(graw[i] >> 16))));
                if (b <= a) ++trunc_dir; else ++up_dir;
            }
            if (ht != graw[i]) ++instr_vs_rtz;
        }
        printf("  %d floats -> %d words  [%.3f ms]\n", NP, NP / 2, ms);
        printf("  host f32_to_f16_bits() vs GLSL pack2_rne():     %zu/%zu mismatching words (want 0)\n",
               h_rne_mism, size_t(NP / 2));
        printf("  host f32_to_f16_bits() vs packHalf2x16():       %zu/%zu (%.4f%%)\n",
               instr_vs_rne, size_t(NP / 2), 100.0 * instr_vs_rne / (NP / 2.0));
        printf("  host f32_to_f16_bits_rtz() vs packHalf2x16():   %zu/%zu (want 0 - it truncates)\n",
               instr_vs_rtz, size_t(NP / 2));
        printf("  direction of every disagreement: %zu never increase |value|, %zu do  "
               "(RTZ means the second number is 0)\n", trunc_dir, up_dir);
        if (instr_vs_rne) {
            for (size_t k = first; k < first + 8 && k < NP / 2; ++k) {
                const uint32_t hr = pack2(f[2 * k], f[2 * k + 1]);
                if (hr == graw[k]) continue;
                uint32_t u; memcpy(&u, &f[2 * k], 4);
                printf("    word %zu: host RNE %08X  card %08X  f[2k]=%.9g (f32 bits %08X, dropped "
                       "13 bits of mantissa = %u)\n", k, hr, graw[k], f[2 * k], u, u & 0x1FFFu);
                break;
            }
        }
        // and the OTHER direction: the card's unpackHalf2x16() against our f16_to_f32(), on the words
        // the card itself produced. align_f16 closed that conversion over all 65536 patterns; this is
        // the same statement made inside the attention harness, on the values attention really stores.
        size_t unpack_mism = 0; double rt = 0.0; size_t rt_at = 0;
        size_t unpack_shown = 0;
        for (int i = 0; i < NP / 2; ++i) {
            const float lo = word_lo(graw[i]), hi = word_hi(graw[i]);
            if (std::memcmp(&lo, &gback[2 * i], 4) != 0) {
                ++unpack_mism;
                // print the pattern, not just the count: whether the card and we disagree on a
                // NORMAL, a SUBNORMAL or a NaN tells us which branch of f16_to_f32 is in question.
                if (unpack_shown < 4) {
                    ++unpack_shown;
                    const uint16_t hb = uint16_t(graw[i] & 0xFFFFu);
                    uint32_t cb, hbb; memcpy(&cb, &gback[2 * i], 4); memcpy(&hbb, &lo, 4);
                    printf("    half %04X: card %08X (%.9g)  ours %08X (%.9g)  [exp=%u man=%u]\n",
                           hb, cb, gback[2 * i], hbb, lo, (hb >> 10) & 0x1Fu, unsigned(hb & 0x3FFu));
                }
            }
            if (std::memcmp(&hi, &gback[2 * i + 1], 4) != 0) ++unpack_mism;
        }
        for (int i = 0; i < NP; ++i) {
            const double e = std::fabs(double(gback[i]) - double(f32_to_f16_as_f32(f[i])));
            if (e > rt) { rt = e; rt_at = size_t(i); }
        }
        printf("  the card's unpackHalf2x16() vs our f16_to_f32(): %zu/%d halves differ (want 0); "
               "RNE round-trip error vs the instruction's: max|err|=%.8f at i=%zu\n",
               unpack_mism, 2 * (NP / 2), rt, rt_at);
    }

    // ================================================================= A: kernel vs mirror
    printf("\n== A. GPU kernels vs the fp32 mirror (attn_forward<true>), real layer-3 shapes ==\n");
    AttnBuffers ref = base, got = base;
    attn_forward<true>(cfg, ref);
    double t_nr = 0, t_qs = 0, t_pv = 0;
    dispatch_all(G, cfg, base, /*round_probs=*/true, &t_nr, &t_qs, &t_pv);
    fetch_all(G, cfg, got);

    size_t bm, mism;
    bm = compare_bits(ref.q16, got.q16, mism);
    printf("  q16  (fp16 Q operand)   words=%-8zu bit-exact=%zu  mismatches=%zu\n", ref.q16.size(), bm, mism);
    bm = compare_bits(ref.kc, got.kc, mism);
    printf("  kc   (the fp16 K arena) words=%-8zu bit-exact=%zu  mismatches=%zu\n", ref.kc.size(), bm, mism);
    bm = compare_bits(ref.vc, got.vc, mism);
    printf("  vc   (the fp16 V arena) words=%-8zu bit-exact=%zu  mismatches=%zu\n", ref.vc.size(), bm, mism);
    bm = compare_bits(ref.p16, got.p16, mism);
    printf("  p16  (fp16 probs)       words=%-8zu bit-exact=%zu  mismatches=%zu\n", ref.p16.size(), bm, mism);
    Cmp cs = compare_f(ref.score,  got.score);
    show("score (raw kq)",      cs,  0.0);
    show("prob (f32)",          compare_f(ref.prob,   got.prob),   0.0);
    show("pregate (context)",   compare_f(ref.pregate, got.pregate), 0.0);
    show("gated (layer out)",   compare_f(ref.gated,  got.gated),  0.0);
    {
        // Decompose the worst score into its 128 arena words. The point is the SIZE of the operand
        // differences: one flipped fp16 bit at |x|~4 is 0.0078, so a dot product of 512 such values
        // cannot move by 13 unless something far bigger than a rounding flip is happening. This prints
        // what actually moved, word by word, for the worst row.
        const size_t i = cs.at;
        const int hw = cfg.hw();
        const int t = int(i / (size_t(cfg.nh) * cfg.cap));
        const int h = int((i / size_t(cfg.cap)) % size_t(cfg.nh));
        const int j = int(i % size_t(cfg.cap));
        const int kv = h / cfg.gqa();
        struct Term { double err; int d; uint32_t qa, qb, ka, kb; };
        std::vector<Term> terms; terms.reserve(size_t(hw));
        size_t dq = 0, dk = 0; double sum_abs = 0.0;
        for (int d = 0; d < hw; ++d) {
            const size_t qi = (size_t(t) * cfg.nh + h) * hw + size_t(d);
            const size_t ki = (size_t(j) * cfg.nkv + kv) * hw + size_t(d);
            if (ref.q16[qi] != got.q16[qi]) ++dq;
            if (ref.kc[ki]  != got.kc[ki])  ++dk;
            const double ta = double(word_lo(ref.q16[qi])) * double(word_lo(ref.kc[ki]))
                            + double(word_hi(ref.q16[qi])) * double(word_hi(ref.kc[ki]));
            const double tb = double(word_lo(got.q16[qi])) * double(word_lo(got.kc[ki]))
                            + double(word_hi(got.q16[qi])) * double(word_hi(got.kc[ki]));
            const double e = std::fabs(ta - tb);
            sum_abs += e;
            terms.push_back(Term{ e, d, ref.q16[qi], got.q16[qi], ref.kc[ki], got.kc[ki] });
        }
        std::sort(terms.begin(), terms.end(), [](const Term &x, const Term &y) { return x.err > y.err; });
        printf("  worst score[t=%d][h=%d][j=%d] (kv head %d): mirror=%.6f card=%.6f  "
               "differing q words=%zu k words=%zu  sum|term err|=%.6f\n",
               t, h, j, kv, double(ref.score[i]), double(got.score[i]), dq, dk, sum_abs);
        for (int k = 0; k < 3 && k < int(terms.size()); ++k) {
            const Term &x = terms[k];
            if (!(x.err > 0.0)) break;
            printf("    word %3d: q %08X (%.6f,%.6f) -> %08X (%.6f,%.6f)   k %08X (%.6f,%.6f) -> "
                   "%08X (%.6f,%.6f)   term err %.6f\n", x.d,
                   x.qa, word_lo(x.qa), word_hi(x.qa), x.qb, word_lo(x.qb), word_hi(x.qb),
                   x.ka, word_lo(x.ka), word_hi(x.ka), x.kb, word_lo(x.kb), word_hi(x.kb), x.err);
        }
    }
    printf("  [time] norm_rope %.3f ms   qk_softmax %.3f ms   pv+gate %.3f ms  "
           "(submit + kernel + fence, seq=%d)\n", t_nr, t_qs, t_pv, cfg.seq);

    // magnitudes, so max|err| can be read as "how much of the signal"
    {
        double s_max = 0, o_max = 0, g_max = 0, p_sum_dev = 0;
        for (size_t i = 0; i < ref.score.size(); ++i) s_max = std::fmax(s_max, std::fabs(ref.score[i]));
        for (size_t i = 0; i < ref.pregate.size(); ++i) o_max = std::fmax(o_max, std::fabs(ref.pregate[i]));
        for (size_t i = 0; i < ref.gated.size(); ++i) g_max = std::fmax(g_max, std::fabs(ref.gated[i]));
        for (int t = 0; t < cfg.seq; ++t) for (int h = 0; h < cfg.nh; ++h) {
            double s = 0; const int nf = int(ref.pos[t]) + 1;
            for (int j = 0; j < cfg.cap; ++j) {
                if (j < nf) s += ref.prob[(size_t(t) * cfg.nh + h) * cfg.cap + j];
                else if (ref.prob[(size_t(t) * cfg.nh + h) * cfg.cap + j] != 0.0f) s += 1e9;
            }
            p_sum_dev = std::fmax(p_sum_dev, std::fabs(s - 1.0));
        }
        printf("  magnitudes: |kq|max=%.3f  |context|max=%.3f  |gated|max=%.3f   "
               "max|sum(p)-1|=%.3e (and every dead slot is exactly 0.0)\n",
               s_max, o_max, g_max, p_sum_dev);
    }

    // ================================================================= A2: stages 2+3 alone
    printf("\n== A2. stages 2+3 fed the MIRROR's own fp16 arena (stage 1 taken out of the loop) ==\n");
    {
        // If the fp16 operands are BYTWISE the mirror's, everything left in the comparison is the dot
        // product, the softmax and PV: their accumulation order, and whether this driver contracts
        // acc = acc + p*v into an FMA. A non-zero error here is a stage 2/3 statement, not a stage 1 one.
        upload(G.c, G.q16,  ref.q16.data(),  ref.q16.size() * 4);
        upload(G.c, G.kc,   ref.kc.data(),   ref.kc.size() * 4);
        upload(G.c, G.vc,   ref.vc.data(),   ref.vc.size() * 4);
        upload(G.c, G.pos,  base.pos.data(), base.pos.size() * 4);
        upload(G.c, G.mask, base.mask.data(), base.mask.size() * 4);
        upload(G.c, G.gate, base.gate_in.data(), base.gate_in.size() * 4);
        QSPush p2{ uint32_t(cfg.seq), uint32_t(cfg.nh), uint32_t(cfg.nkv), uint32_t(cfg.hd),
                   uint32_t(cfg.cap), 1u, cfg.scale, 0.0f };
        PVPush p3{ uint32_t(cfg.seq), uint32_t(cfg.nh), uint32_t(cfg.nkv), uint32_t(cfg.hd),
                   uint32_t(cfg.cap), 1u, 0.0f, 0.0f };
        double t2 = run(G.c, G.qs, &p2, uint32_t(cfg.seq * cfg.nh));
        double t3 = run(G.c, G.pv, &p3, uint32_t(cfg.seq * cfg.nh));
        AttnBuffers s2 = base; fetch_all(G, cfg, s2);
        size_t sm = 0, pm = 0;
        compare_bits(ref.p16, s2.p16, pm);
        printf("  p16 (fp16 probs)   words=%-8zu bit-exact=%-8zu mismatches=%zu\n",
               ref.p16.size(), ref.p16.size() - pm, pm);
        show("score (fed arena)", compare_f(ref.score, s2.score), 0.0);
        show("prob  (fed arena)", compare_f(ref.prob,  s2.prob),  0.0);
        show("pregate(fed arena)", compare_f(ref.pregate, s2.pregate), 0.0);
        show("gated (fed arena)", compare_f(ref.gated, s2.gated), 0.0);
        (void)sm;
        printf("  [time] qk_softmax %.3f ms   pv+gate %.3f ms, operands identical by construction\n",
               t2, t3);
    }

    // ================================================================= T: built-in transcendentals
    printf("\n== T. the card's pow/cos/sin/exp vs the C library's, on the arguments attention uses ==\n");
    {
        const int NT = cfg.nrot / 2;                       // one theta per rotated PAIR
        // four blocks of indices, one argument set each:
        //   [0, NT)                    theta_p = pow(freq_base, -2p/nrot)
        //   [NT, NT+seq*NT)            the rope angles pos*theta, for cos and sin
        //   then the softmax exponents score*scale + mask - rowmax (the live ones)
        //   then the sigmoid's -gate over every output
        const size_t nang = size_t(cfg.seq) * NT;
        std::vector<float> soft;
        for (int t = 0; t < cfg.seq; ++t) {
            const int nf = int(ref.pos[t]) + 1;
            for (int h = 0; h < cfg.nh; ++h) {
                const float *sc = &ref.score[(size_t(t) * cfg.nh + h) * cfg.cap];
                const float *mk = &base.mask[size_t(t) * cfg.cap];
                float mx = -INFINITY;
                for (int j = 0; j < nf; ++j) mx = std::fmax(mx, sc[j] * cfg.scale + mk[j]);
                for (int j = 0; j < nf; ++j) soft.push_back(sc[j] * cfg.scale + mk[j] - mx);
            }
        }
        const size_t n_pow = NT, n_ang = nang, n_soft = soft.size(), n_gate = ow;
        const size_t N = n_pow + n_ang + n_soft + n_gate;
        std::vector<float> A(N, cfg.freq_base), B(N, 0.0f), C(N, 0.0f);
        for (size_t i = 0; i < N; ++i) {
            if (i < n_pow) { B[i] = -2.0f * float(i) / float(cfg.nrot); C[i] = 1.0f; }
            else if (i < n_pow + n_ang) {
                const size_t k = i - n_pow;
                const float th = rope_theta(cfg, int(k % size_t(NT)));
                C[i] = float(int(k / size_t(NT))) * th;                // exactly the shader's product
                B[i] = -2.0f * float(k % size_t(NT)) / float(cfg.nrot);
            } else if (i < n_pow + n_ang + n_soft) {
                C[i] = soft[i - n_pow - n_ang];
            } else {
                C[i] = -base.gate_in[i - n_pow - n_ang - n_soft];       // sigmoid's exp(-x)
            }
        }
        Buf a = make_buf(G.c, N * 4, SS, DEV), b = make_buf(G.c, N * 4, SS, DEV),
            cc = make_buf(G.c, N * 4, SS, DEV), o1 = make_buf(G.c, N * 4, SS, DEV),
            o2 = make_buf(G.c, N * 4, SS, DEV), o3 = make_buf(G.c, N * 4, SS, DEV),
            o4 = make_buf(G.c, N * 4, SS, DEV), o5 = make_buf(G.c, N * 4, SS, DEV),
            o6 = make_buf(G.c, N * 4, SS, DEV), o7 = make_buf(G.c, N * 4, SS, DEV);
        upload(G.c, a, A.data(), N * 4); upload(G.c, b, B.data(), N * 4);
        upload(G.c, cc, C.data(), N * 4);
        // Tripwire, and the one this whole leg needed: before any of the numbers below can be blamed
        // on the card's math, the card's copy of the ARGUMENTS has to be proven bit-identical to the
        // host array. Leg T exists to attribute a difference to the driver; an attribute is worthless
        // when the inputs silently differ. Written because this probe really was reported as "the
        // driver's division is off by 4433x" for two runs, and the cause was the harness.
        std::vector<float> cb(N, 0.0f);
        readback(G.c, cc, cb.data(), N * 4);
        size_t cbad = 0;
        for (size_t i = 0; i < N; ++i) {
            uint32_t x, y; memcpy(&x, &cb[i], 4); memcpy(&y, &C[i], 4);
            if (x != y) ++cbad;
        }
        printf("  [round-trip] the card's copy of the argument buffer differs from the host array in "
               "%zu/%zu words (want 0; anything else makes every number below a statement about the "
               "harness, not about the card)\n", cbad, N);
        std::vector<Buf *> bs{ &a, &b, &cc, &o1, &o2, &o3, &o4, &o5, &o6, &o7 };
        Pipe mp = make_pipe(G.c, "shaders/math_probe.spv", bs, sizeof(PackPush), nullptr);
        PackPush pp{ uint32_t(N), 0, 0, 0 };
        double ms = run(G.c, mp, &pp, uint32_t((N + 255) / 256));
        std::vector<float> g1(N), g2(N), g3(N), g4v(N), g5(N), g6(N), g7(N);
        readback(G.c, o1, g1.data(), N * 4); readback(G.c, o2, g2.data(), N * 4);
        readback(G.c, o3, g3.data(), N * 4); readback(G.c, o4, g4v.data(), N * 4);
        readback(G.c, o5, g5.data(), N * 4); readback(G.c, o6, g6.data(), N * 4);
        readback(G.c, o7, g7.data(), N * 4);
        // a bit-exact count plus the largest relative error, per function, over its own block
        struct Acc { size_t n = 0, diff = 0; double mx_abs = 0.0, mx_rel = 0.0; size_t at = 0;
                     // The operating band is stated on BOTH ends - the argument and the result - because
                     // a relative error is meaningless at the edges in either direction: sqrt(1e-12) is a
                     // fine number computed from an argument whose only purpose is to test the driver's
                     // subnormal path, and 1/1e-40 has no fp32 answer at all. Outside the band the number
                     // is reported but never blended into the band figure, and the worst in-band sample
                     // is named (argument, host answer, card answer) so a huge max can be checked instead
                     // of waved away.
                     size_t n_band = 0, out_band = 0; double mx_rel_band = 0.0, mx_rel_out = 0.0;
                     size_t band_at = 0; float band_arg = 0.f, band_want = 0.f, band_got = 0.f; };
        auto judge = [&](Acc &k, size_t i, float got, float want, double arg) {
            ++k.n;
            uint32_t gb, wb; memcpy(&gb, &got, 4); memcpy(&wb, &want, 4);
            if (gb != wb) ++k.diff;
            const double e = std::fabs(double(got) - double(want));
            const double r = e / std::fmax(std::fabs(double(want)), 1e-30);
            if (e > k.mx_abs) { k.mx_abs = e; k.at = i; }
            if (r > k.mx_rel) k.mx_rel = r;
            const double aw = std::fabs(double(want));
            const bool in = std::fabs(arg) >= 1e-6 && std::fabs(arg) <= 1e6 && aw >= 1e-6 && aw <= 1e6;
            if (in) {
                ++k.n_band;
                if (r > k.mx_rel_band) { k.mx_rel_band = r; k.band_at = i;
                                         k.band_arg = float(arg); k.band_want = want; k.band_got = got; }
            } else { ++k.out_band; if (r > k.mx_rel_out) k.mx_rel_out = r; }
        };
        Acc apow, cos_, sin_, exp1, exp2, sq_, invsq_, div_;
        for (size_t i = 0; i < N; ++i) {
            if (i < n_pow) judge(apow, i, g1[i], std::pow(A[i], B[i]), B[i]);
            else if (i < n_pow + n_ang) {
                judge(cos_,  i, g2[i], std::cos(C[i]), C[i]);
                judge(sin_,  i, g3[i], std::sin(C[i]), C[i]);
            }
            else if (i < n_pow + n_ang + n_soft) judge(exp1, i, g4v[i], std::exp(C[i]), C[i]);
            else judge(exp2, i, g4v[i], std::exp(C[i]), C[i]);
            // the RMS-norm path, judged on every index: this is the only way the PASS-THROUGH rope
            // dims (which contain no cos/sin/pow at all) can differ from the mirror
            judge(sq_,    i, g5[i], std::sqrt(std::fabs(C[i])), std::fabs(C[i]));
            judge(invsq_, i, g6[i], 1.0f / std::sqrt(std::fabs(C[i])), std::fabs(C[i]));
            judge(div_,   i, g7[i], 1.0f / C[i], C[i]);
        }
        auto row = [](const char *name, const Acc &k) {
            printf("  %-14s n=%-6zu bit-differ=%-6zu (%.3f%%)  max|abs|=%.3e  max|rel|=%.3e\n",
                   name, k.n, k.diff, 100.0 * double(k.diff) / double(std::max<size_t>(k.n, 1)),
                   k.mx_abs, k.mx_rel);
        };
        row("pow (theta)", apow); row("cos (angle)", cos_); row("sin (angle)", sin_);
        row("exp (softmax)", exp1); row("exp (sigmoid)", exp2);
        row("sqrt", sq_); row("1/sqrt (norm inv)", invsq_); row("1/x (div)", div_);
        // Stated in fp32 ulp, the only unit that makes sense here: one ulp of a float at |x|~1 is
        // 2^-23 = 1.192e-07. The previous version of this line divided a RELATIVE pow difference by an
        // ABSOLUTE fp16 step and printed "0.00 ulp" - meaningless in one direction, misleading in the
        // other, because it made the gap look smaller than it is. What matters for stage 1 is the
        // ABSOLUTE size of the angle error, since the arena grid is absolute too.
        const double ulp1 = 1.1920928955078125e-07;      // 2^-23, one fp32 ulp at |x| in [1,2)
        double mxang = 0.0;
        for (size_t i = n_pow; i < n_pow + n_ang; ++i)
            mxang = std::fmax(mxang, std::fabs(double(C[i])));
        printf("  [%d indices, %.3f ms] in fp32 ulp at |x|~1: pow max|abs| = %.2f ulp (relative %.2f "
               "ulp - theta spans 1e-07..1, so relative is the honest unit there), cos %.2f ulp, sin "
               "%.2f ulp (ABSOLUTE, because |cos|,|sin| <= 1; the largest angle this run feeds in is "
               "%.1f rad, and argument reduction at that size is what puts the gap at tens of ulp, not "
               "a few), exp(softmax) %.2f ulp absolute / %.2f ulp relative. The arena rounds at 2^-11, "
               "so a flip needs an fp32 value that lands within that band of a halfway point: leg R "
               "measures the band, leg A counts the actual flips.\n",
               int(N), ms, apow.mx_abs / ulp1, apow.mx_rel / ulp1, cos_.mx_abs / ulp1,
               sin_.mx_abs / ulp1, mxang, exp1.mx_abs / ulp1, exp1.mx_rel_band / ulp1);
        // Each of the three is printed with its worst in-band sample, so a large number is checkable:
        // "max|rel| = 3e8 ulp" is an accusation, "worst x=1.000001e-06: host 1.0000005e-03 card
        // 3.6e-04" is a measurement that can be re-run by hand.
        auto band_row = [&](const char *name, const Acc &k) {
            printf("     %-17s band: max|rel| = %8.2f fp32 ulp over %6zu samples | worst x=%.6g: host "
                   "%.8g card %.8g | %zu samples outside the band (edge arguments), max|rel| there %.3e\n",
                   name, k.mx_rel_band / ulp1, k.n_band, double(k.band_arg), double(k.band_want),
                   double(k.band_got), k.out_band, k.mx_rel_out);
        };
        printf("  and the RMS-norm path, which has NO transcendental in it at all: sqrt and a division are "
               "the only arithmetic in it, so a difference there reaches EVERY dim of the row, the "
               "pass-through ones included. Judged on the arguments this run actually contains:\n");
        band_row("sqrt(x)", sq_);
        band_row("1/sqrt(x)=inv", invsq_);
        band_row("1/x", div_);
        printf("     leg R2 takes the next step and tests it per row: it moves the mirror's inv by k ulps "
               "for k in [-4,+4] and asks whether the card's pass-through dims come out bit-identical.\n");
    }

    // ================================================================= R: stage 1 in fp32
    printf("\n== R. stage 1 BEFORE the pack: the fp32 gap behind leg A's flipped bits ==\n");
    {
        // The question leg A cannot answer on its own: when a packed half disagrees, is it a ONE ULP
        // fp16 decision made by an fp32 value that sits within a few fp32 ulp of the halfway point
        // (i.e. the transcendental band of leg T), or is it an algebra bug? So every dim is looked at
        // three ways: the mirror's fp32 value, the card's fp32 value, and both fp16 patterns.
        auto half_of = [](uint32_t w, int d) -> uint32_t {
            return (d & 1) ? (w >> 16) : (w & 0xFFFFu);
        };
        // An fp32 value v against the fp16 grid it is about to be rounded onto: the fp16 step in v's
        // binade is 2^(e-10), the tie point below v is lo + step/2, and `side` says which side of it v is
        // on. A flip is a rounding statement only if the two fp32 values are on OPPOSITE sides.
        auto tie = [](float v, double *dist_step, int *side) -> bool {
            const int e = std::ilogb(v);
            if (e < -14 || e > 15) return false;                       // subnormal / overflow: not here
            const double step = std::ldexp(1.0, e - 10);              // fp16 spacing in this binade
            const double lo = std::floor(double(v) / step) * step;
            const double bdry = lo + step / 2.0;                      // the RNE tie point below v
            // measured in fp16 CELLS, not fp32 ulp: one cell is exactly 2^13 fp32 ulp in the same binade,
            // so a cell-fraction is the number that decides the rounding (0.5 = the tie).
            *dist_step = std::fabs(double(v) - bdry) / step;
            *side = double(v) < bdry ? -1 : (double(v) > bdry ? 1 : 0);
            return true;
        };
        struct Flip { int t, h, d, side_m, side_c; uint32_t rh, gh; float mv, cv;
                      double dm, dc, gap_abs, gap_cells, ulp16; };
        std::vector<Flip> flips;
        size_t qflips = 0, kflips = 0, expchange = 0, opposite = 0, exact_tie = 0;
        double worst_abs = 0.0, worst_ulp16 = 0.0;
        std::vector<float> qf(ow), kf(kfw);
        readback(G.c, G.qf, qf.data(), ow * 4);
        readback(G.c, G.kf, kf.data(), kfw * 4);
        std::vector<float> y(cfg.hd), z(cfg.hd);
        size_t qd = 0, kd = 0, qrot = 0, qpas = 0, krot = 0, kpas = 0;
        double qmax = 0.0, kmax = 0.0; size_t qat = 0, kat = 0;
        std::vector<size_t> byp(8, 0); size_t byp_high = 0;      // mismatches per rotated pair index
        const int nhalf = cfg.nrot / 2, hw16 = cfg.hw();
        // one flipped arena half, recorded with everything that decides whether it is a rounding event:
        // the two fp32 values, their distances to the tie point, and which side of it each one is on.
        auto note = [&](int t, int h, int d, float mv, float cv, uint32_t rh, uint32_t gh) {
            double dm = 0.0, dc = 0.0; int sm = 0, sc = 0;
            const bool ok = tie(mv, &dm, &sm) && tie(cv, &dc, &sc);
            if (ok && sm * sc < 0) ++opposite;
            if (ok && (sm == 0 || sc == 0)) ++exact_tie;
            const int e = std::ilogb(mv);
            const double step = (e >= -14 && e <= 15) ? std::ldexp(1.0, e - 10) : 0.0;
            const double ga = std::fabs(double(mv) - double(cv));
            // fp16 patterns are monotone inside a sign, so a pattern difference IS an ulp count
            const double u16 = std::fabs(double(int(rh) - int(gh)));
            if (ga > worst_abs) worst_abs = ga;
            if (u16 > worst_ulp16) worst_ulp16 = u16;
            flips.push_back(Flip{ t, h, d, sm, sc, rh, gh, mv, cv, dm, dc, ga,
                                  step > 0.0 ? ga / step : 0.0, u16 });
        };
        for (int t = 0; t < cfg.seq; ++t) {
            const int position = int(base.pos[t]);
            for (int h = 0; h < cfg.nh; ++h) {
                attn_rms_norm(cfg, &base.q_in[(size_t(t) * cfg.nh + h) * cfg.hd],
                              base.qn_w.data(), y.data());
                attn_rope(cfg, y.data(), position, z.data());
                const size_t o = (size_t(t) * cfg.nh + h) * cfg.hd;
                const size_t wrow = (size_t(t) * cfg.nh + h) * size_t(hw16);
                for (int d = 0; d < cfg.hd; ++d) {
                    // did this dim's fp32 gap actually change the fp16 pattern, and by how much?
                    const uint32_t rh = half_of(ref.q16[wrow + size_t(d) / 2], d);
                    const uint32_t gh = half_of(got.q16[wrow + size_t(d) / 2], d);
                    if (rh != gh) {
                        ++qflips;
                        if (((rh >> 10) & 0x1Fu) != ((gh >> 10) & 0x1Fu)) ++expchange;
                        note(t, h, d, z[d], qf[o + size_t(d)], rh, gh);
                    }
                    uint32_t a, b; memcpy(&a, &z[d], 4); memcpy(&b, &qf[o + d], 4);
                    if (a == b) continue;
                    ++qd;
                    if (d < cfg.nrot) {
                        ++qrot;
                        const int p = (d < nhalf) ? d : d - nhalf;
                        if (p < 8) ++byp[p]; else ++byp_high;
                    } else ++qpas;
                    const double e = std::fabs(double(z[d]) - double(qf[o + d]));
                    if (e > qmax) { qmax = e; qat = o + size_t(d); }
                }
            }
            for (int h = 0; h < cfg.nkv; ++h) {
                attn_rms_norm(cfg, &base.k_in[(size_t(t) * cfg.nkv + h) * cfg.hd],
                              base.kn_w.data(), y.data());
                attn_rope(cfg, y.data(), position, z.data());
                const size_t o = (size_t(position) * cfg.nkv + h) * cfg.hd;
                // the arena is POSITION major (that is the whole point of owning it), so the K row of
                // the arena is indexed by `position`, exactly as attn_forward and the shader write it
                const size_t wrow = (size_t(position) * cfg.nkv + h) * size_t(hw16);
                for (int d = 0; d < cfg.hd; ++d) {
                    const uint32_t rh = half_of(ref.kc[wrow + size_t(d) / 2], d);
                    const uint32_t gh = half_of(got.kc[wrow + size_t(d) / 2], d);
                    if (rh != gh) {
                        ++kflips;
                        if (((rh >> 10) & 0x1Fu) != ((gh >> 10) & 0x1Fu)) ++expchange;
                        note(t, h, d, z[d], kf[o + size_t(d)], rh, gh);
                    }
                    uint32_t a, b; memcpy(&a, &z[d], 4); memcpy(&b, &kf[o + d], 4);
                    if (a == b) continue;
                    ++kd;
                    if (d < cfg.nrot) ++krot; else ++kpas;
                    const double e = std::fabs(double(z[d]) - double(kf[o + d]));
                    if (e > kmax) { kmax = e; kat = o + size_t(d); }
                }
            }
        }
        printf("  Q: %zu/%zu fp32 values differ bit for bit (%zu inside dims<n_rot, %zu pass-through), "
               "max|err|=%.3e at %zu\n", qd, ow, qrot, qpas, qmax, qat);
        printf("  K: %zu/%zu fp32 values differ (%zu rotated, %zu pass-through), max|err|=%.3e at %zu\n",
               kd, kfw, krot, kpas, kmax, kat);
        printf("  rotated Q mismatches by pair index p: ");
        for (int p = 0; p < 8; ++p) printf("p%d=%zu ", p, byp[p]);
        printf(" p8..31=%zu\n", byp_high);
        // the verdict leg A cannot give by itself
        printf("  arena flips caused by that fp32 band: Q %zu/%zu halves, K %zu/%zu halves; the largest "
               "pattern delta is %.0f fp16 ulp, and %zu flips land on a binade edge where the fp16 "
               "exponent field itself changes (that is legal rounding: 0.9995 becomes 1.000)\n",
               qflips, ow, kflips, kfw, worst_ulp16, expchange);
        if (!flips.empty()) {
            std::sort(flips.begin(), flips.end(), [](const Flip &a, const Flip &b) {
                return a.gap_cells > b.gap_cells; });
            for (size_t i = 0; i < flips.size() && i < 8; ++i) {
                const Flip &x = flips[i];
                printf("    t=%-2d h=%-2d dim=%-3d mirror %.6g vs card %.6g: gap %.3e = %.3f of one fp16 "
                       "cell; tie distance %.4f (side %+d) vs %.4f (side %+d) -> half %04X becomes %04X "
                       "= %.0f fp16 ulp\n",
                       x.t, x.h, x.d, double(x.mv), double(x.cv), x.gap_abs, x.gap_cells,
                       x.dm, x.side_m, x.dc, x.side_c, unsigned(x.rh), unsigned(x.gh), x.ulp16);
            }
            printf("  verdict: of the %zu flips, %zu have the two fp32 values on strictly OPPOSITE sides "
                   "of the cell boundary and %zu have one side sitting EXACTLY on it (a true RNE tie, "
                   "decided by the even-mantissa rule). No flip disagrees by more than %.0f fp16 ulp for a "
                   "reason that is not rounding: the ABSOLUTE disagreement peaks at %.3e, inside leg R's "
                   "own fp32 band of %.3e, and the widest ulp counts belong to dims where the rotation "
                   "nearly cancelled (small value, small cell). %zu of %zu halves = %.4f%%.\n",
                   qflips + kflips, opposite, exact_tie, worst_ulp16, worst_abs, std::fmax(qmax, kmax),
                   qflips + kflips, ow + kfw, 100.0 * double(qflips + kflips) / double(ow + kfw));
        }
        printf("  the angle at pair p is position * 1e7^(-p/32): p0=%g p1=%g p2=%g p3=%g and by p8 it "
               "is %.1e, i.e. cos=1.000000 and sin=angle exactly in fp32 for every position we run. So "
               "a mismatch at large p is an algebra bug, and one confined to small p is the "
               "transcendental gap of leg T.\n",
               double(rope_theta(cfg, 0)), double(rope_theta(cfg, 1)), double(rope_theta(cfg, 2)),
               double(rope_theta(cfg, 3)), double(rope_theta(cfg, 8)));
    }

    // ================================================= R2: is the residual the norm constant, or FMA?
    printf("\n== R2. one scalar per row: can an ulp shift of the norm constant explain the card? ==\n");
    {
        // Leg R found that dims >= n_rot differ too, and those dims contain NO transcendental at all:
        // the card writes exactly (x[d]*inv)*w[d], the same expression the mirror writes, and the only
        // value that can differ between them is inv = 1/sqrt(sum(x*x)/hd + eps). Two candidate causes:
        //   (1) the driver's sqrt / division, measured by leg T at ~1 fp32 ulp;
        //   (2) an FMA-contracted sum of squares (ss = fma(x,x,ss) is a DIFFERENT number than
        //       ss = ss + x*x because the product enters unrounded) -> this run tested it by
        //       materialising x*x in shared memory before the reduction, and leg R's counts came out
        //       BIT-IDENTICAL to the fused version (15213/131072 Q, 4492 pass-through K), so whatever
        //       this is, it is not the compiler contracting the multiply into the add.
        // So this leg stops guessing at the mechanism and does arithmetic instead: for each row, take
        // the host's inv, move it by k ulps for k in [-4..4], and ask whether ANY of those reproduces
        // the card's pass-through dims bit for bit. If yes, the row is one bad sqrt away and the
        // residual is the driver, not our algebra. If no k works, the row is a bug and gets named.
        auto inv_of = [&](const float *x) -> float {
            float ss = 0.0f;
            for (int d = 0; d < cfg.hd; ++d) ss = ss + x[d] * x[d];
            return 1.0f / std::sqrt(ss / float(cfg.hd) + cfg.eps);   // the same two lines as attn_rms_norm
        };
        std::vector<float> qf(ow), kf(kfw);
        readback(G.c, G.qf, qf.data(), ow * 4);
        readback(G.c, G.kf, kf.data(), kfw * 4);
        const int hd = cfg.hd, nrot = cfg.nrot, npas = hd - nrot;
        std::vector<float> y(hd), z(hd);
        struct Acc2 { size_t rows = 0, exact = 0, shifted = 0, unexplained = 0;
                      size_t mis0 = 0, mis_expl = 0, left_rot = 0, left_pas = 0, rot_only = 0;
                      int hist[9] = {0,0,0,0,0,0,0,0,0};        // best k = -4..+4, index k+4
                      size_t worst_row = 0, worst_mis = 0; int worst_k = 0; };
        // the card row for a given trial inv, counted the same way leg R counts it
        auto count_row = [&](const float *x, const float *w, const float *card, int position,
                             float inv, size_t *rot_mis, size_t *pas_mis) {
            for (int d = 0; d < hd; ++d) y[d] = (x[d] * inv) * w[d];
            attn_rope(cfg, y.data(), position, z.data());
            *rot_mis = *pas_mis = 0;
            for (int d = 0; d < hd; ++d) {
                uint32_t ab, bb; memcpy(&ab, &z[d], 4); memcpy(&bb, &card[d], 4);
                if (ab == bb) continue;
                if (d < nrot) ++*rot_mis; else ++*pas_mis;
            }
        };
        auto ulp_of = [](float v, int k) {
            for (int i = 0; i < (k > 0 ? k : -k); ++i)
                v = std::nextafter(v, k > 0 ? INFINITY : -INFINITY);
            return v;
        };
        auto probe_row = [&](const float *x, const float *w, const float *card, int position, Acc2 &a) {
            const float ih = inv_of(x);
            ++a.rows;
            size_t r0 = 0, p0 = 0;
            count_row(x, w, card, position, ih, &r0, &p0);
            a.mis0 += p0;
            int best_k = 0; size_t best_p = p0, best_r = r0;
            for (int k = -4; k <= 4; ++k) {
                if (k == 0) continue;
                size_t r = 0, p = 0;
                count_row(x, w, card, position, ulp_of(ih, k), &r, &p);
                // fewer pass-through mismatches wins; ties keep the smaller |k|, so a row that is
                // already clean at the host's inv never gets attributed to a phantom shift
                if (p < best_p || (p == best_p && (k < 0 ? -k : k) < (best_k < 0 ? -best_k : best_k))) {
                    best_p = p; best_r = r; best_k = k;
                }
            }
            if (p0 == 0) { ++a.exact; if (r0 > 0) ++a.rot_only; }
            if (best_p == 0) {
                if (best_k != 0) { ++a.shifted; ++a.hist[best_k + 4]; }
                a.mis_expl += p0;
            } else {
                ++a.unexplained;
                if (best_p > a.worst_mis) { a.worst_mis = best_p; a.worst_row = a.rows; a.worst_k = best_k; }
            }
            a.left_rot += best_r; a.left_pas += best_p;
        };
        auto report = [&](const char *tag, const Acc2 &a, size_t total_vals) {
            printf("  %s: %zu rows - %zu have bit-clean pass-through dims at the host's inv, %zu become "
                   "clean only when inv moves by 1..4 ulp, %zu are NOT reproducible by ANY inv in "
                   "[host-4ulp, host+4ulp]\n", tag, a.rows, a.exact, a.shifted, a.unexplained);
            printf("     best-shift histogram: ");
            for (int k = -4; k <= 4; ++k) if (k) printf("k%+d=%d ", k, a.hist[k + 4]);
            printf("\n     pass-through mismatches at the host inv: %zu/%zu (%.3f%%), of which %zu (%.1f%%) "
                   "sit in rows a single scalar shift reproduces EXACTLY\n",
                   a.mis0, total_vals, 100.0 * double(a.mis0) / double(total_vals), a.mis_expl,
                   a.mis0 ? 100.0 * double(a.mis_expl) / double(a.mis0) : 100.0);
            printf("     after that shift the residual is %zu mismatches: %zu in dims<n_rot (cos/sin, leg "
                   "T's band) + %zu in pass-through dims (want 0). %zu rows are already clean on "
                   "pass-through and differ ONLY in the rotation.\n",
                   a.left_rot + a.left_pas, a.left_rot, a.left_pas, a.rot_only);
            if (a.unexplained)
                printf("     WORST unexplained row: #%zu of %zu, %zu pass-through dims still differ at its "
                       "best k=%+d\n", a.worst_row, a.rows, a.worst_mis, a.worst_k);
        };
        Acc2 aq, ak;
        for (int t = 0; t < cfg.seq; ++t) {
            const int position = int(base.pos[t]);
            for (int h = 0; h < cfg.nh; ++h) {
                const size_t o = (size_t(t) * cfg.nh + h) * cfg.hd;
                probe_row(&base.q_in[o], base.qn_w.data(), &qf[o], position, aq);
            }
            for (int h = 0; h < cfg.nkv; ++h) {
                const size_t i = (size_t(t) * cfg.nkv + h) * cfg.hd;
                const size_t o = (size_t(position) * cfg.nkv + h) * cfg.hd;
                probe_row(&base.k_in[i], base.kn_w.data(), &kf[o], position, ak);
            }
        }
        report("Q", aq, ow * size_t(npas));
        report("K", ak, kfw * size_t(npas));
    }

    // ================================================================= B: causal and -INF fill
    printf("\n== B. causal prefix and the -INF fill value ==\n");
    {
        size_t dead_nonzero = 0, live = 0;
        for (int t = 0; t < cfg.seq; ++t) {
            const int nf = int(base.pos[t]) + 1;
            for (int h = 0; h < cfg.nh; ++h)
                for (int j = nf; j < cfg.cap; ++j)
                    if (got.prob[(size_t(t) * cfg.nh + h) * cfg.cap + j] != 0.0f) ++dead_nonzero;
            live += size_t(nf) * cfg.nh;
        }
        printf("  live slots=%zu  dead slots that are NOT exactly 0.00000000: %zu\n", live, dead_nonzero);
        double dev = 0.0;
        for (size_t i = 0; i < got.prob.size(); ++i)
            dev = std::fmax(dev, std::fabs(double(got.prob[i]) - double(ref.prob[i])));
        printf("  GPU dead-slot zeros agree with the mirror: max|err| over the whole prob tensor = %.8f\n", dev);
    }
    {
        // a -INF INSIDE the live prefix: the mask must kill that column exactly and renormalise the
        // rest. This is the "fill 值处理正确" case the brief asked for, and it is not something the
        // causal test can catch (an additive -0.0 or a large finite sentinel passes causal).
        AttnBuffers m2 = base;
        for (int t = 0; t < cfg.seq; ++t) if (t >= 1) m2.mask[size_t(t) * cfg.cap + 0] = neg_inf();
        dispatch_all(G, cfg, m2, true, nullptr, nullptr, nullptr);
        AttnBuffers g2 = base; fetch_all(G, cfg, g2);
        attn_forward<true>(cfg, m2);
        size_t nonzero_killed = 0, killed = 0;
        for (int t = 1; t < cfg.seq; ++t) {
            ++killed;
            if (g2.prob[(size_t(t) * cfg.nh + 0) * cfg.cap + 0] != 0.0f) ++nonzero_killed;
        }
        const Cmp cm = compare_f(m2.prob, g2.prob);
        printf("  -INF in the live prefix: %zu rows tested, GPU wrote a non-zero probability in %zu of "
               "them\n", killed, nonzero_killed);
        printf("  masked-row probabilities vs the mirror: max|err|=%.8f at %zu, bit-exact=%.4f%%\n",
               cm.max_abs, cm.at, 100.0 * cm.bitsame / double(cm.n));
        double sum_dev = 0.0;
        for (int t = 1; t < cfg.seq; ++t) {
            double s = 0; for (int j = 0; j <= t; ++j) s += g2.prob[(size_t(t) * cfg.nh + 3) * cfg.cap + j];
            sum_dev = std::fmax(sum_dev, std::fabs(s - 1.0));
        }
        printf("  after the kill, the row still sums to 1: max|sum-1| = %.3e\n", sum_dev);
    }

    // ================================================================= C: rope structure
    printf("\n== C. rope: position 0, pass-through dims, and the frequency expression ==\n");
    {
        const int hw = cfg.hd / 2;
        // (1) the position-0 row must come out of the card equal to the NORM-ONLY value: cos(0)=1 and
        // sin(0)=0 make the rotation contribute nothing, so any sign flip or an angle that is not
        // position*theta collapses this check. Built here from the inputs, without attn_rope().
        std::vector<float> normed(cfg.hd);
        size_t p0_mism = 0;
        attn_rms_norm(cfg, base.k_in.data(), base.kn_w.data(), normed.data());     // token 0, kv 0
        for (int d = 0; d < hw; ++d)
            if (got.kc[size_t(d)] != pack2(normed[2 * d], normed[2 * d + 1])) ++p0_mism;
        printf("  position 0, kv head 0: the cached row differs from the norm-only value in %d/%d "
               "words (want 0)\n", int(p0_mism), hw);

        // (2) dims >= n_rot are pass-through for EVERY token and kv head, checked against the
        // normed value only - so a rotation that leaked past n_rot would show up here.
        size_t pt_words = 0, pt_mism = 0;
        for (int t = 0; t < cfg.seq; ++t)
            for (int h = 0; h < cfg.nkv; ++h) {
                attn_rms_norm(cfg, &base.k_in[(size_t(t) * cfg.nkv + h) * cfg.hd], base.kn_w.data(),
                              normed.data());
                const size_t base_i = (size_t(t) * cfg.nkv + h) * hw;   // position == token here
                for (int d = cfg.nrot; d < cfg.hd; d += 2) {
                    const int wi = d / 2;
                    ++pt_words;
                    if (got.kc[base_i + wi] != pack2(normed[d], normed[d + 1])) ++pt_mism;
                }
            }
        printf("  dims %d..%d pass through: %zu/%zu fp16 words differ (want 0)\n",
               cfg.nrot, cfg.hd - 1, pt_mism, pt_words);

        // (3) the frequency EXPRESSION. The fork builds theta in two powf() calls, we build it in
        // one; the difference is real, so it is measured rather than waved away.
        int at_p = 0;
        const double spread = attn_freq_spread(cfg, &at_p);
        double rel = 0.0;
        for (int p = 0; p < cfg.nrot / 2; ++p) {
            const double a = rope_theta(cfg, p), b = rope_theta_fork(cfg, p);
            rel = std::fmax(rel, std::fabs(a - b) / a);
        }
        printf("  powf(base,-2p/nrot) vs the fork's powf(powf(base,-2/nrot),p): max|d theta| = "
               "%.3e at p=%d (%.3e relative)\n", spread, at_p, rel);
        const float ts = std::pow(cfg.freq_base, -2.0f / float(cfg.nrot));
        double worst = 0.0; size_t w_at = 0;
        for (int t = 0; t < cfg.seq; ++t) {
            std::vector<float> xn(cfg.hd), fork_y(cfg.hd), ours(cfg.hd);
            attn_rms_norm(cfg, &base.q_in[size_t(t) * cfg.nh * cfg.hd], base.qn_w.data(), xn.data());
            const int half = cfg.nrot / 2;
            for (int p = 0; p < half; ++p) {
                const float a = float(t) * std::pow(ts, float(p));
                const float c = std::cos(a), s = std::sin(a);
                fork_y[p]        = xn[p] * c - xn[p + half] * s;
                fork_y[p + half] = xn[p + half] * c + xn[p] * s;
            }
            for (int d = cfg.nrot; d < cfg.hd; ++d) fork_y[d] = xn[d];
            attn_rope(cfg, xn.data(), t, ours.data());
            for (int d = 0; d < cfg.nrot; ++d) {
                const double e = std::fabs(double(fork_y[d]) - double(ours[d]));
                if (e > worst) { worst = e; w_at = size_t(t) * cfg.hd + d; }
            }
        }
        printf("  propagated into a rotated row: max|err| = %.3e at token*hd+dim = %zu, i.e. %.3e of "
               "the value - the fp16 operand quantum here is ~6e-4, so the two forms cannot be told "
               "apart once cached\n", worst, w_at, worst);
    }

    // ================================================================= D: GQA broadcast
    printf("\n== D. GQA: 16 q heads reading 4 kv heads ==\n");
    {
        // Give EVERY q head the same input row. Then heads that share a kv head must produce
        // bit-identical scores, and heads on different kv heads must not - which is the only way to
        // see that the shader indexes kv head h/(nh/nkv) rather than h.
        AttnBuffers g3 = base;
        Rng r(0x6CA4u);
        std::vector<float> oneq(cfg.hd);
        for (int d = 0; d < cfg.hd; ++d) oneq[d] = r.gauss(1.6f);
        for (int t = 0; t < cfg.seq; ++t)
            for (int h = 0; h < cfg.nh; ++h)
                for (int d = 0; d < cfg.hd; ++d)
                    g3.q_in[(size_t(t) * cfg.nh + h) * cfg.hd + d] = oneq[d];
        dispatch_all(G, cfg, g3, true, nullptr, nullptr, nullptr);
        AttnBuffers g4 = base; fetch_all(G, cfg, g4);
        size_t same_group_diff = 0, diff_group_same = 0, pairs_same = 0, pairs_diff = 0;
        for (int t = 0; t < cfg.seq; ++t)
            for (int h = 0; h < cfg.nh; ++h)
                for (int h2 = h + 1; h2 < cfg.nh; ++h2) {
                    bool eq = true;
                    for (int j = 0; j <= int(g3.pos[t]); ++j)
                        if (g4.score[(size_t(t) * cfg.nh + h) * cfg.cap + j] !=
                            g4.score[(size_t(t) * cfg.nh + h2) * cfg.cap + j]) { eq = false; break; }
                    if (h / cfg.gqa() == h2 / cfg.gqa()) { ++pairs_same; if (!eq) ++same_group_diff; }
                    else { ++pairs_diff; if (eq) ++diff_group_same; }
                }
        printf("  identical Q in all 16 heads: %zu head pairs share a kv head, %zu do not\n",
               pairs_same, pairs_diff);
        printf("  pairs that SHARE a kv head and disagree: %zu (want 0)\n", same_group_diff);
        printf("  pairs on DIFFERENT kv heads that agree:  %zu (want 0 - they must read different K)\n",
               diff_group_same);
    }

    // ================================================================= E: prefill vs decode
    printf("\n== E. the KV arena: one prefill pass vs a token-at-a-time decode, bit for bit ==\n");
    {
        // This is the test that the arena we own is indexed the way we think: fill it position by
        // position with the same stage-1 kernel, recompute only the newest token, and demand that the
        // row be BIT-IDENTICAL to the prefill row. A tolerance would hide a stale-slot read.
        dispatch_all(G, cfg, base, true, nullptr, nullptr, nullptr);
        AttnBuffers pre = base; fetch_all(G, cfg, pre);

        AttnCfg c1 = cfg; c1.seq = 1;
        AttnBuffers one; one.alloc(c1);
        std::vector<uint32_t> zk(cw, 0u), zv(cw, 0u);
        upload(G.c, G.kc, zk.data(), zk.size() * 4);      // an EMPTY arena: nothing to inherit
        upload(G.c, G.vc, zv.data(), zv.size() * 4);

        const size_t row_f = size_t(cfg.nh) * cfg.hd;      // gated floats per token
        size_t rows = 0, bad_rows = 0, prob_bad = 0;
        double worst = 0.0; size_t worst_at = 0;
        for (int t = 0; t < cfg.seq; ++t) {
            one.alloc(c1);                                 // clears q16/kc/vc views of this run
            for (int h = 0; h < cfg.nh; ++h)
                for (int d = 0; d < cfg.hd; ++d) {
                    one.q_in[size_t(h) * cfg.hd + d]    = base.q_in[(size_t(t) * cfg.nh + h) * cfg.hd + d];
                    one.gate_in[size_t(h) * cfg.hd + d] = base.gate_in[(size_t(t) * cfg.nh + h) * cfg.hd + d];
                }
            for (int h = 0; h < cfg.nkv; ++h)
                for (int d = 0; d < cfg.hd; ++d) {
                    one.k_in[size_t(h) * cfg.hd + d] = base.k_in[(size_t(t) * cfg.nkv + h) * cfg.hd + d];
                    one.v_in[size_t(h) * cfg.hd + d] = base.v_in[(size_t(t) * cfg.nkv + h) * cfg.hd + d];
                }
            one.pos[0] = uint32_t(t);
            for (int j = 0; j < cfg.cap; ++j) one.mask[size_t(j)] = 0.0f;
            // stage 1 writes position t into the arena that stage 2/3 then read for tokens 0..t
            dispatch_all(G, c1, one, true, nullptr, nullptr, nullptr);
            std::vector<float> dg(row_f), dp(size_t(cfg.nh) * cfg.cap);
            readback(G.c, G.gated, dg.data(), row_f * 4);
            readback(G.c, G.prob,  dp.data(), dp.size() * 4);
            ++rows;
            for (int h = 0; h < cfg.nh; ++h) {
                for (int d = 0; d < cfg.hd; ++d) {
                    const float a = dg[size_t(h) * cfg.hd + d];
                    const float b = pre.gated[(size_t(t) * cfg.nh + h) * cfg.hd + d];
                    if (a != b) { ++bad_rows; break; }
                }
                for (int j = 0; j <= t; ++j) {
                    const float a = dp[size_t(h) * cfg.cap + j];
                    const float b = pre.prob[(size_t(t) * cfg.nh + h) * cfg.cap + j];
                    const double e = std::fabs(double(a) - double(b));
                    if (e > worst) { worst = e; worst_at = size_t(t); }
                    if (a != b) ++prob_bad;
                }
            }
        }
        printf("  %zu decode steps against the same prefill: %zu head-rows differ in the gated output, "
               "%zu probability slots differ (want 0 and 0)\n", rows, bad_rows, prob_bad);
        printf("  largest prefill/decode disagreement: %.8f (token %zu)\n", worst, worst_at);
    }

    // ================================================================= G: rounding attribution
    printf("\n== G. what each fp16 rounding costs (float64 ideal legs, same ascending order) ==\n");
    {
        const int RM_CACHE = 1, RM_Q = 2, RM_PROBS = 4;
        std::vector<double> i0(ow), i1(ow), i3(ow), i7(ow);
        attn_forward_ideal(cfg, base, i0, 0);
        attn_forward_ideal(cfg, base, i1, RM_CACHE);
        attn_forward_ideal(cfg, base, i3, RM_CACHE | RM_Q);
        attn_forward_ideal(cfg, base, i7, RM_CACHE | RM_Q | RM_PROBS);
        std::vector<double> mirror_f(ow);
        for (size_t i = 0; i < ow; ++i) mirror_f[i] = double(ref.gated[i]);
        auto diff = [](const std::vector<double> &a, const std::vector<double> &b, double &mx,
                       size_t &at) {
            mx = 0.0; at = 0;
            for (size_t i = 0; i < a.size(); ++i) {
                const double e = std::fabs(a[i] - b[i]);
                if (e > mx) { mx = e; at = i; }
            }
        };
        double m1, m2, m3, m4; size_t a1, a2, a3, a4;
        diff(i0, i1, m1, a1);                    // fp16 K/V arena  - MEASURED property of the engine
        diff(i1, i3, m2, a2);                    // fp16 Q operand  - same contract
        diff(i3, i7, m3, a3);                    // fp16 probs      - INFERRED from ggml's mixed type
        diff(i7, mirror_f, m4, a4);              // fp32 accumulation vs fp64
        printf("  ideal(fp64, nothing rounded) -> +fp16 arena : max|err|=%.4e at %zu\n", m1, a1);
        printf("                               -> +fp16 Q     : max|err|=%.4e at %zu\n", m2, a2);
        printf("                               -> +fp16 probs : max|err|=%.4e at %zu\n", m3, a3);
        printf("  the fp32 mirror vs the all-fp16 ideal       : max|err|=%.4e at %zu  "
               "(float32 accumulation, not a layout error)\n", m4, a4);
        double gmax = 0.0, asum = 0.0;
        for (size_t i = 0; i < ref.gated.size(); ++i) gmax = std::fmax(gmax, std::fabs(ref.gated[i]));
        asum = m1 + m2 + m3 + m4;
        printf("  |gated|max = %.3f, so the whole fp16+fp32 chain costs at most %.3e absolute "
               "(%.3e relative)\n", gmax, asum, asum / std::fmax(gmax, 1e-30));

        // ---------------------------------------------------------------- the layered verdict
        // The brief's question is not "is the card bit-identical to a CPU" - it is "how large is our
        // divergence, measured against the error the official data format already imposes". Both numbers
        // are on the table by now, so put them side by side instead of quoting max|err| bare: a bare
        // 2.3e-04 on the context (runs/attn_s32.txt, seq=32) looks alarming and is 1.3e-04 of full
        // scale, i.e. BELOW the 5.1e-04 that storing K/V in fp16 costs the reference engine itself.
        const Cmp v_sc = compare_f(ref.score, got.score), v_pb = compare_f(ref.prob, got.prob);
        const Cmp v_pg = compare_f(ref.pregate, got.pregate), v_ga = compare_f(ref.gated, got.gated);
        const double our = std::fmax(v_pg.max_abs, v_ga.max_abs);
        printf("\n  V. verdict - card vs mirror against the format's own floor:\n");
        printf("     card vs mirror   score=%.3e (=%.2e of |kq|max)  prob=%.3e  context=%.3e  "
               "layer-out=%.3e (=%.2e of |gated|max)\n",
               v_sc.max_abs, v_sc.max_abs / std::fmax(v_sc.ref_max, 1e-30), v_pb.max_abs, v_pg.max_abs,
               v_ga.max_abs, v_ga.max_abs / std::fmax(v_ga.ref_max, 1e-30));
        printf("     fp16 layout cost  arena=%.3e  Q=%.3e  probs=%.3e  sum=%.3e  (= the SAME tensor "
               "space the engine cannot opt out of)\n", m1, m2, m3, asum);
        printf("     ratio our/format = %.3f -> %s\n", our / std::fmax(asum, 1e-30),
               our <= asum ? "our divergence is SMALLER than the error the official fp16 cache imposes; "
                             "the remaining card/host difference is the last fp32 bits of a transcendental, "
                             "not a term the official engine gets right and we get wrong"
                           : "our divergence exceeds the fp16 floor - leg A/R still has work");
    }

    printf("\n[done]\n");
    return 0;
}
