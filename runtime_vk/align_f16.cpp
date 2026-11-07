// align_f16.cpp - exhaustive closure of the fp16 question that 层2 left open.
//
// align_dequant.cpp compares OUR PTQ1_0 decode against the OFFICIAL shader over 3840 synthetic
// blocks whose block scales are 15 hand-picked fp16 patterns (0, -0, subnormal min/mid/max, normal
// min/max, +-Inf, NaN, ...). That found a real bug in our own f16_to_f32() subnormal branch, but it
// is still a 15-point SAMPLE of a 65536-point domain. This tool makes it a closed statement:
//
//   GPU-CONV    shaders/f16_map.comp -> float(float16_t), i.e. the one expression
//               `const float d = float(data_a[ib].d)` in the official dequant_ptq1_0.comp uses for
//               the block scale. Executed on the RX 550X.
//   OURS        f16_to_f32() from ptq_decode.h - the function bonsai_vk.cpp actually calls.
//   IEEE-REF    an independent host reference computed in double from the definition
//               (subnormal: man*2^-24, normal: (1024+man)*2^(exp-25), non-finite: +-Inf/NaN).
//               binary16 -> binary32 is exact, so a double is enough to be the truth, and nothing
//               in this reference shares code with the bit-twiddling under test.
//
// All 65536 patterns are fed through all three, and every disagreement is printed as
// pattern / expected / got / ulp-distance, classified into the fp16 region it lives in.
//
//   build: runtime_vk/build_mingw.bat step 2/2
//   run:   align_f16.exe

#include "ptq_decode.h"

#include <vulkan/vulkan.h>

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
            fprintf(stderr, "Vulkan failure %d at %s:%d\n", (int)_r, __FILE__, __LINE__);  \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

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
static std::vector<char> read_shader(const char *rel) {
    if (FILE *t = fopen(rel, "rb")) { fclose(t); return read_file(rel); }
    std::string up = std::string("../") + rel;
    return read_file(up.c_str());
}

// ---- the truth, from the definition of binary16, in a double (binary16 -> double is exact) -------
static double f16_exact(uint16_t h) {
    const uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    const double s = (h & 0x8000u) ? -1.0 : 1.0;
    if (exp == 0u)  return s * std::ldexp(double(man), -24);                    // 0 and subnormals
    if (exp == 0x1Fu) return man ? std::nan("") : s * INFINITY;
    return s * std::ldexp(double(man) + 1024.0, int(exp) - 25);                 // normals
}
static const char *region(uint16_t h) {
    const uint32_t exp = (h >> 10) & 0x1Fu, man = h & 0x3FFu;
    if (exp == 0u)  return man == 0u ? "zero" : "subnormal";
    if (exp == 0x1Fu) return man ? "NaN" : "Inf";
    if (h == 0x3C00u) return "one";
    return "normal";
}
// distance in f32 ulps between two halves-converted-to-floats; 0 means the same number.
static long ulps(float a, float b) {
    if (a == b) return 0;
    auto bias = [](float x) { uint32_t u; memcpy(&u, &x, 4);
                              return (u & 0x80000000u) ? int64_t(0x80000000) - int64_t(u) : int64_t(u); };
    const int64_t d = bias(a) - bias(b);
    return long(d < 0 ? -d : d);
}

// "is the NUMBER right": NaN == NaN is false in IEEE but two NaN results are the same outcome, and
// +-0 are the same number, so value comparison is done by class, not by subtraction.
static bool same_num(float x, float y) {
    const bool xn = std::isnan(x), yn = std::isnan(y);
    if (xn || yn) return xn && yn;
    return x == y;
}

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);
    const uint32_t N = 65536u;
    std::vector<uint16_t> bits(N);
    for (uint32_t i = 0; i < N; ++i) bits[i] = uint16_t(i);

    // -------------------------------------------------- CPU: OUR conversion vs the IEEE reference
    std::vector<float> ours(N), ref(N);
    long ours_vs_ref = 0, ref_ulp_max = 0, ours_bit_vs_ref = 0;
    for (uint32_t i = 0; i < N; ++i) {
        ours[i] = f16_to_f32(uint16_t(i));
        ref[i]  = float(f16_exact(uint16_t(i)));
        if (!same_num(ours[i], ref[i])) ++ours_vs_ref;
        if (memcmp(&ours[i], &ref[i], 4) != 0) ++ours_bit_vs_ref;
        if (std::isfinite(ours[i]) && std::isfinite(ref[i]))
            ref_ulp_max = std::max(ref_ulp_max, ulps(ours[i], ref[i]));
    }
    printf("[cpu] f16_to_f32 vs IEEE-754 definition: %ld/%u wrong numbers, %ld/%u different bit "
           "patterns (+-0 and NaN payloads are counted here, not as errors), max %ld ulp\n",
           ours_vs_ref, N, ours_bit_vs_ref, N, ref_ulp_max);

    // -------------------------------------------------- GPU: run the official conversion expression
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_2; ai.pApplicationName = "align_f16";   // 1.2 => 16-bit storage
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

    VkPhysicalDevice16BitStorageFeatures f16{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_16BIT_STORAGE_FEATURES};
    VkPhysicalDevice8BitStorageFeatures f8{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_8BIT_STORAGE_FEATURES};
    f8.pNext = &f16;
    VkPhysicalDeviceFeatures2 feats2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_FEATURES_2};
    feats2.pNext = &f8;
    vkGetPhysicalDeviceFeatures2(phys, &feats2);
    printf("[feat] storageBuffer16Bit=%d uniformAndStorage16Bit=%d\n",
           int(f16.storageBuffer16BitAccess), int(f16.uniformAndStorageBuffer16BitAccess));

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    uint32_t qfi = 0; for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfi; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi; di.pNext = &f8; di.pEnabledFeatures = &feats2.features;
    VkDevice dev; AK_CHECK(vkCreateDevice(phys, &di, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    auto find_type = [&](VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mp.memoryTypes[i].propertyFlags & want) == want) return i;
        fprintf(stderr, "no memory type 0x%X\n", unsigned(want)); exit(1);
    };
    auto make_buf = [&](uint64_t size, VkBufferUsageFlags use, VkMemoryPropertyFlags mprops,
                        VkBuffer &ob, VkDeviceMemory &om) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = size; bi.usage = use;
        AK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &ob));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, ob, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size; mai.memoryTypeIndex = find_type(mprops);
        AK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &om));
        AK_CHECK(vkBindBufferMemory(dev, ob, om, 0));
    };
    const uint64_t h_sz = uint64_t(N) * 2, v_sz = uint64_t(N) * 4;
    VkBuffer b_hst, b_h, b_v, b_vst, b_wst, b_w, b_u, b_ust;
    VkDeviceMemory m_hst, m_h, m_v, m_vst, m_wst, m_w, m_u, m_ust;
    make_buf(h_sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b_hst, m_hst);
    make_buf(h_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b_h, m_h);
    make_buf(v_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b_v, m_v);
    make_buf(v_sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b_vst, m_vst);
    // the u32 the runtime's repack kernel actually writes: f16 bits in the LOW half, because
    // gemv_ternary.comp reads `.x` of unpackHalf2x16(). The high half is filled with the same bits
    // so a swapped .x/.y bug would also show up as a value difference rather than silently passing.
    make_buf(v_sz, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b_wst, m_wst);
    make_buf(v_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b_w, m_w);
    make_buf(v_sz, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
             VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, b_u, m_u);
    make_buf(v_sz, VK_BUFFER_USAGE_TRANSFER_DST_BIT,
             VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT, b_ust, m_ust);
    { void *p = nullptr; AK_CHECK(vkMapMemory(dev, m_hst, 0, h_sz, 0, &p));
      memcpy(p, bits.data(), size_t(h_sz)); vkUnmapMemory(dev, m_hst); }
    { void *p = nullptr; AK_CHECK(vkMapMemory(dev, m_wst, 0, v_sz, 0, &p));
      uint32_t *w = reinterpret_cast<uint32_t *>(p);
      for (uint32_t i = 0; i < N; ++i) w[i] = uint32_t(bits[i]) | (uint32_t(bits[i]) << 16);
      vkUnmapMemory(dev, m_wst); }

    VkDescriptorSetLayoutBinding bs[4] = {};
    for (int i = 0; i < 4; ++i) { bs[i].binding = i; bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                                  bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
    VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    sli.bindingCount = 4; sli.pBindings = bs;
    VkDescriptorSetLayout sl; AK_CHECK(vkCreateDescriptorSetLayout(dev, &sli, nullptr, &sl));
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 5 * sizeof(uint32_t)};
    VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    pli.setLayoutCount = 1; pli.pSetLayouts = &sl; pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; AK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
    VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 4};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
    VkDescriptorPool dpool; AK_CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool));
    VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
    dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &sl;
    VkDescriptorSet ds; AK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
    VkDescriptorBufferInfo dbi[4] = {{b_h, 0, h_sz}, {b_v, 0, v_sz}, {b_w, 0, v_sz}, {b_u, 0, v_sz}};
    VkWriteDescriptorSet wr[4] = {};
    for (int i = 0; i < 4; ++i) {
        wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET; wr[i].dstSet = ds; wr[i].dstBinding = i;
        wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        wr[i].pBufferInfo = &dbi[i];
    }
    vkUpdateDescriptorSets(dev, 4, wr, 0, nullptr);
    auto code = read_shader("shaders/f16_map.spv");
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
    VkBufferCopy up{0, 0, uint32_t(h_sz)};
    vkCmdCopyBuffer(cb, b_hst, b_h, 1, &up);
    VkBufferCopy up2{0, 0, uint32_t(v_sz)};
    vkCmdCopyBuffer(cb, b_wst, b_w, 1, &up2);
    { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr); }
    uint32_t pc[5] = {N, 0, 0, 0, 0};
    vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
    vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
    vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
    vkCmdDispatch(cb, (N + 255) / 256, 1, 1);
    { VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
      mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT; mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
      vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                           0, 1, &mb, 0, nullptr, 0, nullptr); }
    VkBufferCopy cp{0, 0, uint32_t(v_sz)};
    vkCmdCopyBuffer(cb, b_v, b_vst, 1, &cp);
    vkCmdCopyBuffer(cb, b_u, b_ust, 1, &cp);
    AK_CHECK(vkEndCommandBuffer(cb));
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    AK_CHECK(vkQueueSubmit(queue, 1, &si, nullptr));
    AK_CHECK(vkQueueWaitIdle(queue));
    std::vector<float> gpu(N), gpu2(N);
    { void *p = nullptr; AK_CHECK(vkMapMemory(dev, m_vst, 0, v_sz, 0, &p));
      memcpy(gpu.data(), p, size_t(v_sz)); vkUnmapMemory(dev, m_vst); }
    { void *p = nullptr; AK_CHECK(vkMapMemory(dev, m_ust, 0, v_sz, 0, &p));
      memcpy(gpu2.data(), p, size_t(v_sz)); vkUnmapMemory(dev, m_ust); }

    // -------------------------------------------------- verdict
    // Two different questions, deliberately measured with two different yardsticks:
    //   "is the NUMBER right"  - NaN == NaN is false in IEEE, so a NaN result is compared by class,
    //                            and +-0 is compared by value (they are the same number).
    //   "is the BIT PATTERN right" - strict memcmp, which is what the dequant fidelity test uses.
    struct Tally { long num = 0, bits = 0, payload = 0, ulp = 0; };
    auto tally = [&](const std::vector<float> &a, const std::vector<float> &b) {
        Tally t;
        for (uint32_t i = 0; i < N; ++i) {
            if (!same_num(a[i], b[i])) {
                ++t.num;
                if (std::isfinite(a[i]) && std::isfinite(b[i])) t.ulp = std::max(t.ulp, ulps(a[i], b[i]));
            }
            if (memcmp(&a[i], &b[i], 4) != 0) { ++t.bits; if (same_num(a[i], b[i])) ++t.payload; }
        }
        return t;
    };
    Tally g_ref  = tally(gpu,  ref);      // float(float16_t)          vs IEEE
    Tally g2_ref = tally(gpu2, ref);      // float(unpackHalf2x16().x) vs IEEE
    Tally g_our  = tally(gpu,  ours);     // the official dequant shader's expression vs our host code
    Tally g2_our = tally(gpu2, ours);     // OUR GEMV's expression vs our host code
    Tally g_g2   = tally(gpu,  gpu2);     // the two hardware conversions against each other
    auto show = [&](const char *a, const char *b, const Tally &t) {
        printf("  %-26s vs %-26s : %4ld/%u wrong numbers, %4ld/%u different bit patterns "
               "(%ld of them payload-only), max %ld ulp\n", a, b, t.num, N, t.bits, N, t.payload, t.ulp);
    };

    printf("\n==== all 65536 patterns, five ways ====\n");
    show("GPU float(f16)",        "IEEE-754 definition",  g_ref);
    show("GPU unpackHalf2x16(.x)", "IEEE-754 definition",  g2_ref);
    show("GPU float(f16)",        "OURS f16_to_f32",      g_our);
    show("GPU unpackHalf2x16(.x)", "OURS f16_to_f32",      g2_our);
    show("GPU float(f16)",        "GPU unpackHalf2x16",   g_g2);
    int printed = 0;
    for (uint32_t i = 0; i < N && printed < 12; ++i) {
        if (same_num(gpu[i], ref[i]) && same_num(gpu2[i], ref[i]) &&
            memcmp(&gpu[i], &ours[i], 4) == 0 && memcmp(&gpu2[i], &ours[i], 4) == 0) continue;
        uint32_t gb, ob, g2b; memcpy(&gb, &gpu[i], 4); memcpy(&ob, &ours[i], 4); memcpy(&g2b, &gpu2[i], 4);
        printf("   f16 0x%04X %-9s expect %-13.9g  float(f16) 0x%08X  unpackHalf 0x%08X  ours 0x%08X\n",
               i, region(uint16_t(i)), double(ref[i]), gb, g2b, ob);
        ++printed;
    }
    if (!printed) printf("   (every one of the 65536 patterns agrees everywhere)\n");
    printf("\n==== verdict ====\n");
    // The pass condition is about NUMBERS, plus an explicit allowance: two IEEE NaNs with different
    // payloads are both NaN, and the dequant multiply (trit * d) canonicalises them anyway - which
    // align_dequant's synth run measures directly (the NaN scale classes came out bit-exact on the
    // weights, 32768/32768 GPU vs OURS). A bit-pattern difference that cannot survive one multiply
    // is not a fidelity gap; a wrong number would be.
    const bool ok = ours_vs_ref == 0 && g_ref.num == 0 && g2_ref.num == 0 &&
                    g_our.bits == 0 && g2_our.bits == 0;
    printf("  our f16_to_f32 is exact over the whole binary16 domain:  %s (%ld wrong numbers)\n",
           ours_vs_ref ? "NO" : "YES", ours_vs_ref);
    printf("  card's float(float16_t)  is IEEE-exact:                   %s (%ld wrong, %ld bit-diff)\n",
           g_ref.num ? "NO" : "YES", g_ref.num, g_ref.bits);
    printf("  card's float(unpackHalf2x16(x).x) is IEEE-exact:          %s (%ld wrong, %ld bit-diff)\n",
           g2_ref.num ? "NO" : "YES", g2_ref.num, g2_ref.bits);
    printf("  BOTH shader expressions == our host conversion, bit for bit:  %s (%ld / %ld bit-diff)\n",
           (g_our.bits == 0 && g2_our.bits == 0) ? "YES" : "NO", g_our.bits, g2_our.bits);

    vkDestroyPipeline(dev, pipe, nullptr); vkDestroyShaderModule(dev, sm, nullptr);
    vkDestroyPipelineLayout(dev, pl, nullptr); vkDestroyDescriptorSetLayout(dev, sl, nullptr);
    vkDestroyDescriptorPool(dev, dpool, nullptr); vkDestroyCommandPool(dev, cpool, nullptr);
    auto kill = [&](VkBuffer b, VkDeviceMemory m) { vkDestroyBuffer(dev, b, nullptr); vkFreeMemory(dev, m, nullptr); };
    kill(b_hst, m_hst); kill(b_h, m_h); kill(b_v, m_v); kill(b_vst, m_vst);
    kill(b_wst, m_wst); kill(b_w, m_w); kill(b_u, m_u); kill(b_ust, m_ust);
    vkDestroyDevice(dev, nullptr); vkDestroyInstance(inst, nullptr);
    return ok ? 0 : 1;
}
