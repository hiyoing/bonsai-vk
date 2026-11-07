// shared_probe.cpp - does the RX 550X / AMD Windows driver actually accept and correctly RUN a
// compute pipeline whose static shared allocation equals maxComputeSharedMemorySize?
//
// The question is load-bearing for one line of the delivery report. attn_qk_softmax_wide.comp keeps
// a row's scores and probabilities in two `shared float[MAXCAP]` arrays, so with MAXCAP 2048 it uses
// 16 KiB of a 32,768 B ceiling and 4 K context (which needs MAXCAP 4096) would use the whole ceiling
// with zero headroom. Until this tool existed, "4 K needs a real two-pass restructure" was a
// DERIVATION from that arithmetic. It is allowed to be a derivation; it is not allowed to be reported
// as if it were a measurement, so here is the measurement.
//
// Each PROBE_N lives in its own SPIR-V module (shaders/probe_shm_nNNNN.comp -> shared_mem_probe.glslh)
// precisely because `shared` arrays are file-scope declarations: putting several sizes in one module
// would let the driver see a module-wide sum and make the per-case numbers meaningless. One module,
// one size, one pipeline, one dispatch, then every element is read back and compared.
//
// Pipeline creation success alone is NOT the verdict: a driver that accepted an oversized declaration
// and silently clipped the allocation would pass that test. So the second half of every case checks
// that all 2*n values round-tripped through shared memory are the values that were written.
//
// The ladder (bytes = 2 * n * 4):
//   n=3072 -> 24576 B   far below the limit        : control that must succeed
//   n=4000 -> 32000 B     768 B below              : still below, with headroom
//   n=4090 -> 32720 B     48 B below               : the last legal size that leaves ANY headroom
//   n=4096 -> 32768 B     EXACTLY the limit        : the case the 4 K question turns on
//   n=4224 -> 33792 B     1024 B OVER              : over-limit control
//   n=4608 -> 36864 B     4096 B OVER              : over-limit control, 1 LDS page
//
// NOTE ON THE CONTROLS. maxComputeSharedMemorySize is a software limit; gfx803 has 64 KiB of LDS per
// CU, so an over-limit module may well be accepted and may well run. That is a finding, not a bug in
// the harness - the controls exist so that "case X returned OK" can be told apart from "the harness
// cannot see a difference between cases". Both are reported, neither is assumed.
//
//   build: runtime_vk/build_mingw.bat (last step)
//   run:   shared_probe.exe            (from runtime_vk\; clear_field.ps1 first, one GPU process)

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define AK_CHECK(x)                                                                         \
    do {                                                                                    \
        VkResult _r = (x);                                                                  \
        if (_r != VK_SUCCESS) {                                                             \
            fprintf(stderr, "Vulkan failure %d at %s:%d\n", (int)_r, __FILE__, __LINE__);   \
            exit(1);                                                                        \
        }                                                                                   \
    } while (0)

// "2.13x" style ratio, printed in the occupancy table's "vs 2048" column.
static const char *fmt_ratio(double ms, double base) {
    static char buf[32];
    snprintf(buf, sizeof buf, "%.2fx", base > 0 ? ms / base : 0.0);
    return buf;
}

static const char *vk_name(VkResult r) {
    switch (r) {
        case VK_SUCCESS:                        return "VK_SUCCESS";
        case VK_INCOMPLETE:                     return "VK_INCOMPLETE";
        case VK_ERROR_OUT_OF_HOST_MEMORY:       return "VK_ERROR_OUT_OF_HOST_MEMORY";
        case VK_ERROR_OUT_OF_DEVICE_MEMORY:     return "VK_ERROR_OUT_OF_DEVICE_MEMORY";
        case VK_ERROR_INITIALIZATION_FAILED:    return "VK_ERROR_INITIALIZATION_FAILED";
        case VK_ERROR_INVALID_SHADER_NV:        return "VK_ERROR_INVALID_SHADER_NV";
        default:                                return "VK_ERROR_(other)";
    }
}

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

struct Case {
    uint32_t n;
    const char *note;
};

int main() {
    setvbuf(stdout, nullptr, _IONBF, 0);

    const Case cases[] = {
        {3072u, "24576 B, far below the limit - must-succeed control"},
        {4000u, "32000 B, 768 B below the limit"},
        {4090u, "32720 B, 48 B below the limit - the last size with any headroom"},
        {4096u, "32768 B, EXACTLY the limit - the 4 K question"},
        {4224u, "33792 B, 1024 B over the limit - over-limit control"},
        {4608u, "36864 B, 4096 B over the limit - over-limit control"},
    };
    const int ncase = int(sizeof cases / sizeof cases[0]);

    // ------------------------------------------------------------------ device
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    ai.apiVersion = VK_API_VERSION_1_2; ai.pApplicationName = "shared_probe";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO}; ici.pApplicationInfo = &ai;
    VkInstance inst; AK_CHECK(vkCreateInstance(&ici, nullptr, &inst));
    uint32_t nd = 0; AK_CHECK(vkEnumeratePhysicalDevices(inst, &nd, nullptr));
    std::vector<VkPhysicalDevice> devs(nd);
    AK_CHECK(vkEnumeratePhysicalDevices(inst, &nd, devs.data()));
    VkPhysicalDevice phys = devs[0];
    for (auto p : devs) { VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
                          if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pp.vendorID == 0x1002) phys = p; }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(phys, &props);
    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq); vkGetPhysicalDeviceQueueFamilyProperties(phys, &nq, qf.data());
    uint32_t qfi = 0; for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { qfi = i; break; }

    const uint32_t LIMIT = props.limits.maxComputeSharedMemorySize;
    printf("==== device ====\n");
    printf("  %s (vendor 0x%04X, device 0x%04X, driver %u)\n", props.deviceName, props.vendorID,
           props.deviceID, props.driverVersion);
    printf("  maxComputeSharedMemorySize      = %u B\n", LIMIT);
    printf("  maxComputeWorkGroupInvocations  = %u\n", props.limits.maxComputeWorkGroupInvocations);
    printf("  maxComputeWorkGroupSize         = %u x %u x %u\n",
           props.limits.maxComputeWorkGroupSize[0], props.limits.maxComputeWorkGroupSize[1],
           props.limits.maxComputeWorkGroupSize[2]);
    printf("  api %u.%u\n", VK_VERSION_MAJOR(props.apiVersion), VK_VERSION_MINOR(props.apiVersion));

    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi{VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO};
    qi.queueFamilyIndex = qfi; qi.queueCount = 1; qi.pQueuePriorities = &prio;
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = 1; di.pQueueCreateInfos = &qi;
    VkDevice dev; AK_CHECK(vkCreateDevice(phys, &di, nullptr, &dev));
    VkQueue queue; vkGetDeviceQueue(dev, qfi, 0, &queue);

    VkPhysicalDeviceMemoryProperties mp; vkGetPhysicalDeviceMemoryProperties(phys, &mp);
    auto find_type = [&](VkMemoryPropertyFlags want) {
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if ((mp.memoryTypes[i].propertyFlags & want) == want) return i;
        fprintf(stderr, "no memory type 0x%X\n", unsigned(want)); exit(1);
    };

    VkCommandPoolCreateInfo cpci{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpci.queueFamilyIndex = qfi; cpci.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VkCommandPool cpool; AK_CHECK(vkCreateCommandPool(dev, &cpci, nullptr, &cpool));

    // The output buffer is sized for the largest case and reused; it is host-visible so the read-back
    // is a plain mapping, and binding it at 2*4608 floats is smaller than any limit that matters.
    uint32_t maxn = 0; for (int i = 0; i < ncase; ++i) maxn = std::max(maxn, cases[i].n);
    const uint64_t out_sz = uint64_t(maxn) * 2 * sizeof(float);
    VkBuffer b_out; VkDeviceMemory m_out;
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = out_sz; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        AK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b_out));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b_out, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_type(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        AK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &m_out));
        AK_CHECK(vkBindBufferMemory(dev, b_out, m_out, 0));
    }

    printf("\n==== per case ====\n");
    printf("  %6s %9s  %-9s %-8s %-11s %s\n", "n", "shared B", "vs limit", "pipe", "round-trip", "note");
    int over_created = 0, over_ran = 0, n_over = 0, exact_ok = -1;
    for (int i = 0; i < ncase; ++i)
        if (uint64_t(cases[i].n) * 8 > uint64_t(LIMIT)) ++n_over;

    for (int c = 0; c < ncase; ++c) {
        const uint32_t n = cases[c].n;
        const uint64_t bytes = uint64_t(n) * 2 * 4;

        std::string path = "shaders/probe_shm_n" + std::to_string(n) + ".spv";
        auto code = read_shader(path.c_str());

        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = code.size(); mi.pCode = reinterpret_cast<const uint32_t *>(code.data());
        VkShaderModule sm; AK_CHECK(vkCreateShaderModule(dev, &mi, nullptr, &sm));

        VkDescriptorSetLayoutBinding bs[2] = {};
        for (int i = 0; i < 2; ++i) { bs[i].binding = i; bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
                                      bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT; }
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        sli.bindingCount = 2; sli.pBindings = bs;
        VkDescriptorSetLayout sl; AK_CHECK(vkCreateDescriptorSetLayout(dev, &sli, nullptr, &sl));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &sl;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        VkPipelineLayout pl; AK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 2};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
        VkDescriptorPool dpool; AK_CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &sl;
        VkDescriptorSet ds; AK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        const VkDescriptorBufferInfo dbi[2] = {{b_out, 0, out_sz},
                                               {b_out, uint64_t(n) * 4, out_sz - uint64_t(n) * 4}};
        // sd_out occupies [0, n*4) and sp_out occupies [n*4, 2*n*4) of the SAME buffer, so the
        // read-back is one mapping of 2n floats.
        VkWriteDescriptorSet wr[2] = {};
        for (int i = 0; i < 2; ++i) {
            wr[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
            wr[i].dstSet = ds; wr[i].dstBinding = uint32_t(i);
            wr[i].descriptorCount = 1; wr[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
            wr[i].pBufferInfo = &dbi[i];
        }
        vkUpdateDescriptorSets(dev, 2, wr, 0, nullptr);

        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
        cpi.layout = pl;
        VkPipeline pipe = nullptr;
        const VkResult cr = vkCreateComputePipelines(dev, nullptr, 1, &cpi, nullptr, &pipe);

        const char *vs = bytes == uint64_t(LIMIT) ? "EQUALS"
                       : bytes < uint64_t(LIMIT) ? "under" : "OVER";

        std::string rt = "not tried";
        bool ran_ok = false;
        if (cr == VK_SUCCESS) {
            // one workgroup of 256 lanes, write shared, barrier, read back to global
            VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
            cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
            cbai.commandBufferCount = 1;
            VkCommandBuffer cb; AK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            AK_CHECK(vkBeginCommandBuffer(cb, &beg));
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
            const uint32_t pc[4] = {n, 0, 0, 0};
            vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
            vkCmdDispatch(cb, 1, 1, 1);
            AK_CHECK(vkEndCommandBuffer(cb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
            si.commandBufferCount = 1; si.pCommandBuffers = &cb;
            const VkResult sr = vkQueueSubmit(queue, 1, &si, nullptr);
            const VkResult wr2 = (sr == VK_SUCCESS) ? vkQueueWaitIdle(queue) : sr;
            if (sr != VK_SUCCESS || wr2 != VK_SUCCESS) {
                rt = std::string("submit/wait -> ") + vk_name(sr) + "/" + vk_name(wr2);
            } else {
                std::vector<float> got(size_t(n) * 2, 0.0f);
                void *p = nullptr;
                AK_CHECK(vkMapMemory(dev, m_out, 0, out_sz, 0, &p));
                memcpy(got.data(), p, size_t(n) * 2 * 4);
                vkUnmapMemory(dev, m_out);
                long bad_sd = 0, bad_sp = 0; uint32_t first_bad = 0xFFFFFFFFu;
                for (uint32_t k = 0; k < n; ++k) {
                    if (got[size_t(k)] != float(k))                       { if (!bad_sd && first_bad > k) first_bad = k; ++bad_sd; }
                    if (got[size_t(n) + k] != 1000000.0f + float(k))      { if (!bad_sp && first_bad > k) first_bad = k; ++bad_sp; }
                }
                ran_ok = (bad_sd == 0 && bad_sp == 0);
                char buf[128];
                if (ran_ok) snprintf(buf, sizeof buf, "all %u/%u OK", n, n);
                else snprintf(buf, sizeof buf, "BAD sd=%ld sp=%ld first@%u", bad_sd, bad_sp,
                              first_bad == 0xFFFFFFFFu ? 0u : first_bad);
                rt = buf;
            }
            vkDestroyPipeline(dev, pipe, nullptr);
            vkFreeCommandBuffers(dev, cpool, 1, &cb);
            if (bytes > uint64_t(LIMIT)) { ++over_created; if (ran_ok) ++over_ran; }
            if (bytes == uint64_t(LIMIT)) exact_ok = ran_ok ? 1 : 0;
        } else {
            rt = std::string("pipeline creation skipped (") + vk_name(cr) + ")";
            if (bytes == uint64_t(LIMIT)) exact_ok = 0;
        }

        printf("  %6u %9llu  %-9s %-8s %-11s %s\n", n, (unsigned long long)bytes, vs,
               vk_name(cr), rt.c_str(), cases[c].note);
        fflush(stdout);

        vkDestroyDescriptorPool(dev, dpool, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyDescriptorSetLayout(dev, sl, nullptr);
        vkDestroyShaderModule(dev, sm, nullptr);
    }

    // ================= stage 2: the occupancy question =================
    // Stage 1 says a 32,768 B shared allocation is created AND runs correctly, so MAXCAP 4096 is not
    // blocked by the driver. What raising it would still cost is residence: gfx803 has 64 KiB of LDS
    // per compute unit, so a 256-lane workgroup asking for 32 KiB cannot share a CU the way a 16 KiB
    // one can. bench_shm_nNNNN.comp keeps the INSTRUCTION STREAM identical (the loop bound is a push
    // constant, nothing in the body depends on PROBE_N except the array declaration) and only varies
    // that declaration, so a duration difference between sizes is an occupancy difference.
    //
    // Reported per size: min wall ms over the reps, and the ratio to the engine's CURRENT MAXCAP
    // (2048). Min, not mean, because Windows adds scheduler noise to individual submits; the min is
    // the GPU-bound floor and it is the same estimator for every size.
    printf("\n==== stage 2: same work, different shared size (occupancy) ====\n");
    const uint32_t grid = 64u, iters = 20000u;
    const uint32_t reps = 6u;
    const uint32_t bsizes[] = {512u, 1024u, 2048u, 4096u};
    const int nb = int(sizeof bsizes / sizeof bsizes[0]);
    double ms_at_2048 = -1.0;
    printf("  grid=%u workgroups x 256 lanes, iters=%u, min of %u reps\n", grid, iters, reps);
    printf("  %6s %9s  %-9s %-9s %s\n", "n", "shared B", "min ms", "vs 2048", "checksum");

    VkBuffer b_res; VkDeviceMemory m_res;
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = 4096; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        AK_CHECK(vkCreateBuffer(dev, &bi, nullptr, &b_res));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(dev, b_res, &mr);
        VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        mai.allocationSize = mr.size;
        mai.memoryTypeIndex = find_type(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                        VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);
        AK_CHECK(vkAllocateMemory(dev, &mai, nullptr, &m_res));
        AK_CHECK(vkBindBufferMemory(dev, b_res, m_res, 0));
    }
    for (int s = 0; s < nb; ++s) {
        const uint32_t n = bsizes[s];
        const uint64_t bytes = uint64_t(n) * 8;
        std::string path = "shaders/bench_shm_n" + std::to_string(n) + ".spv";
        auto code = read_shader(path.c_str());
        VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
        mi.codeSize = code.size(); mi.pCode = reinterpret_cast<const uint32_t *>(code.data());
        VkShaderModule sm; AK_CHECK(vkCreateShaderModule(dev, &mi, nullptr, &sm));
        VkDescriptorSetLayoutBinding bs{0, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1,
                                         VK_SHADER_STAGE_COMPUTE_BIT, nullptr};
        VkDescriptorSetLayoutCreateInfo sli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        sli.bindingCount = 1; sli.pBindings = &bs;
        VkDescriptorSetLayout sl; AK_CHECK(vkCreateDescriptorSetLayout(dev, &sli, nullptr, &sl));
        VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, 4 * sizeof(uint32_t)};
        VkPipelineLayoutCreateInfo pli{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        pli.setLayoutCount = 1; pli.pSetLayouts = &sl;
        pli.pushConstantRangeCount = 1; pli.pPushConstantRanges = &pcr;
        VkPipelineLayout pl; AK_CHECK(vkCreatePipelineLayout(dev, &pli, nullptr, &pl));
        VkDescriptorPoolSize dps{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1};
        VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        dpi.maxSets = 1; dpi.poolSizeCount = 1; dpi.pPoolSizes = &dps;
        VkDescriptorPool dpool; AK_CHECK(vkCreateDescriptorPool(dev, &dpi, nullptr, &dpool));
        VkDescriptorSetAllocateInfo dsai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        dsai.descriptorPool = dpool; dsai.descriptorSetCount = 1; dsai.pSetLayouts = &sl;
        VkDescriptorSet ds; AK_CHECK(vkAllocateDescriptorSets(dev, &dsai, &ds));
        const VkDescriptorBufferInfo dbi{b_res, 0, 4096};
        VkWriteDescriptorSet wrd{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET};
        wrd.dstSet = ds; wrd.dstBinding = 0; wrd.descriptorCount = 1;
        wrd.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; wrd.pBufferInfo = &dbi;
        vkUpdateDescriptorSets(dev, 1, &wrd, 0, nullptr);
        VkComputePipelineCreateInfo cpi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
        cpi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        cpi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; cpi.stage.module = sm; cpi.stage.pName = "main";
        cpi.layout = pl;
        VkPipeline pipe = nullptr;
        const VkResult cr = vkCreateComputePipelines(dev, nullptr, 1, &cpi, nullptr, &pipe);
        if (cr != VK_SUCCESS) {
            printf("  %6u %9llu  %-9s %-9s PIPELINE CREATION FAILED (%s)\n", n,
                   (unsigned long long)bytes, "-", "-", vk_name(cr));
            vkDestroyDescriptorPool(dev, dpool, nullptr); vkDestroyPipelineLayout(dev, pl, nullptr);
            vkDestroyDescriptorSetLayout(dev, sl, nullptr); vkDestroyShaderModule(dev, sm, nullptr);
            continue;
        }
        VkCommandBufferAllocateInfo cbai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cbai.commandPool = cpool; cbai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cbai.commandBufferCount = 1;
        VkCommandBuffer cb; AK_CHECK(vkAllocateCommandBuffers(dev, &cbai, &cb));
        VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        AK_CHECK(vkBeginCommandBuffer(cb, &beg));
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pipe);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl, 0, 1, &ds, 0, nullptr);
        const uint32_t pc[4] = {iters, n, 0u, 0u};
        vkCmdPushConstants(cb, pl, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof pc, pc);
        vkCmdDispatch(cb, grid, 1, 1);
        AK_CHECK(vkEndCommandBuffer(cb));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &cb;

        double best = 1e30;
        for (uint32_t r = 0; r < reps + 2u; ++r) {                 // 2 untimed warm-up reps
            const auto t0 = std::chrono::steady_clock::now();
            AK_CHECK(vkQueueSubmit(queue, 1, &si, nullptr));
            AK_CHECK(vkQueueWaitIdle(queue));
            const auto t1 = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
            if (r >= 2u) best = std::min(best, ms);
        }
        // checksum: every workgroup wrote a finite number, i.e. the timed dispatch really ran the loop
        long bad = 0;
        { void *p = nullptr; AK_CHECK(vkMapMemory(dev, m_res, 0, 4096, 0, &p));
          const float *v = reinterpret_cast<const float *>(p);
          for (uint32_t g = 0; g < grid; ++g) if (!(v[g] == v[g] && v[g] < 1e30f && v[g] > -1e30f)) ++bad;
          vkUnmapMemory(dev, m_res); }
        if (n == 2048u) ms_at_2048 = best;
        printf("  %6u %9llu  %9.3f  %-9s %s\n", n, (unsigned long long)bytes, best,
               n == 2048u ? "1.00x (base)" : (ms_at_2048 > 0 ? fmt_ratio(best, ms_at_2048) : "-"),
               bad ? "NON-FINITE OUTPUT" : "all finite");
        fflush(stdout);
        vkDestroyPipeline(dev, pipe, nullptr);
        vkFreeCommandBuffers(dev, cpool, 1, &cb);
        vkDestroyDescriptorPool(dev, dpool, nullptr);
        vkDestroyPipelineLayout(dev, pl, nullptr);
        vkDestroyDescriptorSetLayout(dev, sl, nullptr);
        vkDestroyShaderModule(dev, sm, nullptr);
    }

    printf("\n==== what this decides ====\n");
    printf("  MAXCAP 4096 (32768 B, the size 4 K context needs) : %s\n",
           exact_ok == 1 ? "ACCEPTED AND RUN CORRECTLY - 4 K is blocked by something else"
           : exact_ok == 0 ? "REJECTED OR WRONG - 4 K needs the two-pass restructure, measured"
                           : "NOT MEASURED");
    printf("  over-limit cases: %d declared above the limit, %d had a pipeline created, %d also round-"
           "tripped every element correctly\n", n_over, over_created, over_ran);
    printf("    (gfx803 has 64 KiB of LDS per CU, so an over-limit declaration CAN execute: the Vulkan\n"
           "     limit is a software rule, not the silicon. That is why the engine may not rely on it,\n"
           "     and why this is reported as the measured behaviour of THIS driver, not as a licence.)\n");
    printf("  the engine keeps MAXCAP 2048 either way; the numbers above are what this machine did.\n");

    vkDestroyBuffer(dev, b_out, nullptr);
    vkFreeMemory(dev, m_out, nullptr);
    vkDestroyCommandPool(dev, cpool, nullptr);
    vkDestroyDevice(dev, nullptr);
    vkDestroyInstance(inst, nullptr);
    return 0;
}
