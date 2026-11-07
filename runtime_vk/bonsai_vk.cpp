// bonsai_vk.cpp - minimal Vulkan host for the custom ternary GEMV path on the RX 550X.
//
// It does exactly four things, all of them load-bearing for the three questions in
// STAGE2_CUSTOM.md:
//   1. ONE device-local vkAllocateMemory arena for every tensor, sub-bound by offset, and it
//      PRINTS the alignment the driver demanded plus the padding it cost. That number is the
//      empirical answer to "does extreme alignment buy VRAM for longer context".
//   2. A one-time in-VRAM repack of PTQ1_0 blocks -> the random-access 2-bit layout
//      (repack_ptq1_0.comp). No CPU in the loop beyond the initial upload.
//   3. A correctness check against a CPU reference that decodes with the SAME trit recurrence
//      the PrismML fork's own shader uses, so a mismatch means a bug in repack/GEMV, not in
//      our reading of the format.
//   4. GPU-timestamp timing -> Gop/s and GB/s, comparable to the WGSL numbers in probes 11/13
//      and to llama.cpp's mul_mat_vec_ptq1_0.comp on the same tensor.
//
// Deliberately absent: GGUF parsing, tokenizer, chat template, attention, sampler. Those are
// most of llama.cpp and none of them are on the path we are measuring.
//
// VERIFICATION STATUS: built and RUN on the target card (RX 550X / gfx803) with MinGW-w64 against
// the driver's own vulkan-1.dll - see STAGE3_RUNTIME.md for the toolchain paths, the 24-point
// correctness grid and the measured rates. The kernel numerics were first measured in WGSL on this
// same card (probes 11/13, 143-155 Gop/s) and the native path now beats that by 1.08x weighted.
// BONSAI_PARTIAL / BONSAI_DIAG / BONSAI_RANDSCALE / BONSAI_XFER / BONSAI_SYNC are the
// instrumentation modes that produced every number quoted in the reports.

// First include in the translation unit on purpose: on Windows it pulls winsock2.h in before any
// windows.h (kv_guard.h needs windows.h for PDH/DXGI, and the two Winsock generations disagree about
// every type). On POSIX it is the socket/pthread layer instead. See bonsai_platform.h.
#include "bonsai_platform.h"

#include <vulkan/vulkan.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define VK_CHECK(x)                                                                       \
    do {                                                                                  \
        VkResult _r = (x);                                                                \
        if (_r != VK_SUCCESS) {                                                           \
            fprintf(stderr, "Vulkan failure %d at %s:%d\n", (int)_r, __FILE__, __LINE__);\
            exit(1);                                                                      \
        }                                                                                 \
    } while (0)

// ------------------------------------------------------------------ PTQ1_0 ground truth
// Lives in ptq_decode.h so that align_dequant.exe measures the arithmetic THIS engine runs instead
// of a private copy inside the test. Same functions, same bytes, same card - see FAITHFUL_REPRO.md.
#include "ptq_decode.h"

// ------------------------------------------------------------------ small helpers

static std::vector<char> read_file(const char *p) {
    FILE *f = fopen(p, "rb");
    if (!f) { fprintf(stderr, "cannot open %s - compile shaders first (build.bat)\n", p); exit(1); }
    fseek(f, 0, SEEK_END); long n = ftell(f); fseek(f, 0, SEEK_SET);
    std::vector<char> b(size_t(n) ? size_t(n) : 1);
    size_t got = fread(b.data(), 1, size_t(n), f); fclose(f);
    if (long(got) != n) { fprintf(stderr, "short read %s\n", p); exit(1); }
    uint32_t magic; memcpy(&magic, b.data(), 4);
    if (n < 8 || magic != 0x07230203u) { fprintf(stderr, "%s is not SPIR-V\n", p); exit(1); }
    return b;
}
static uint64_t align_up_u64(uint64_t a, uint64_t b) { return (a + b - 1) / b * b; }

struct Buf { VkBuffer buf = VK_NULL_HANDLE; uint64_t offset = 0, size = 0, align = 0; };

struct Ctx {
    VkInstance inst = VK_NULL_HANDLE;
    VkPhysicalDevice phys = VK_NULL_HANDLE;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;
    uint32_t qfi = 0;
    // the transfer-only family (0xFFFFFFFF if the device has none) and its queue: everything the
    // BONSAI_OVERLAP probe needs to know whether a copy can run while the GEMV is running.
    VkQueue qtx = VK_NULL_HANDLE;
    uint32_t tfi = 0xFFFFFFFFu;
    VkPhysicalDeviceLimits limits{};
    VkPhysicalDeviceMemoryProperties mp{};
    VkDeviceMemory arena = VK_NULL_HANDLE, host_mem = VK_NULL_HANDLE;
    uint32_t arena_type = 0, host_type = 0;
    VkCommandPool cpool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
};

static VkPipeline make_pipeline(Ctx &c, const std::string &path, VkPipelineLayout pl) {
    auto code = read_file(path.c_str());
    VkShaderModuleCreateInfo mi{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
    mi.codeSize = code.size(); mi.pCode = reinterpret_cast<const uint32_t *>(code.data());
    VkShaderModule sm; VK_CHECK(vkCreateShaderModule(c.dev, &mi, nullptr, &sm));
    VkComputePipelineCreateInfo pi{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
    pi.stage.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
    pi.stage.stage = VK_SHADER_STAGE_COMPUTE_BIT; pi.stage.module = sm; pi.stage.pName = "main";
    pi.layout = pl;
    VkPipeline p; VK_CHECK(vkCreateComputePipelines(c.dev, nullptr, 1, &pi, nullptr, &p));
    vkDestroyShaderModule(c.dev, sm, nullptr);
    return p;
}

static VkDescriptorSetLayout make_set_layout(Ctx &c, uint32_t n_bindings) {
    std::vector<VkDescriptorSetLayoutBinding> bs(n_bindings);
    for (uint32_t i = 0; i < n_bindings; ++i) {
        bs[i] = {}; bs[i].binding = i; bs[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        bs[i].descriptorCount = 1; bs[i].stageFlags = VK_SHADER_STAGE_COMPUTE_BIT;
    }
    VkDescriptorSetLayoutCreateInfo li{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
    li.bindingCount = n_bindings; li.pBindings = bs.data();
    VkDescriptorSetLayout sl; VK_CHECK(vkCreateDescriptorSetLayout(c.dev, &li, nullptr, &sl));
    return sl;
}

static VkPipelineLayout make_pipe_layout(Ctx &c, VkDescriptorSetLayout sl, uint32_t pc_bytes) {
    VkPushConstantRange pcr{VK_SHADER_STAGE_COMPUTE_BIT, 0, pc_bytes};
    VkPipelineLayoutCreateInfo li{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
    li.setLayoutCount = 1; li.pSetLayouts = &sl; li.pushConstantRangeCount = 1; li.pPushConstantRanges = &pcr;
    VkPipelineLayout pl; VK_CHECK(vkCreatePipelineLayout(c.dev, &li, nullptr, &pl));
    return pl;
}

static void bind_set(Ctx &c, VkDescriptorSet ds, const std::vector<Buf> &bufs) {
    std::vector<VkDescriptorBufferInfo> infos(bufs.size());
    std::vector<VkWriteDescriptorSet> writes(bufs.size());
    for (size_t i = 0; i < bufs.size(); ++i) {
        // offset is INTO the VkBuffer, and the buffer is already bound at bufs[i].offset inside
        // the arena. Passing the arena offset here applied it twice, so every shader read past the
        // end of its own buffer: x came back as 0, and the repack's stores landed nowhere.
        // This is exactly the bug the "not compiled yet" caveat in STAGE2 5. was for.
        infos[i] = {bufs[i].buf, 0, bufs[i].size};
        writes[i] = {}; writes[i].sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
        writes[i].dstSet = ds; writes[i].dstBinding = uint32_t(i); writes[i].descriptorCount = 1;
        writes[i].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[i].pBufferInfo = &infos[i];
    }
    vkUpdateDescriptorSets(c.dev, uint32_t(writes.size()), writes.data(), 0, nullptr);
}

// Two dispatches recorded in one command buffer have NO memory dependency unless the host puts one
// there. Without these the driver is free to run xsum and gemv concurrently (and on Polaris it
// does), so gemv reads a xsum buffer that is still being written - the symptom is a diff that is
// row-independent within a run but DIFFERENT between runs of the same binary and the same args.
static void bar_compute(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

// Kept because the moment an upload and a dispatch share one command buffer it is required.
[[maybe_unused]] static void bar_transfer_to_compute(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_TRANSFER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_SHADER_READ_BIT | VK_ACCESS_SHADER_WRITE_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_TRANSFER_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

// compute wrote, the copy engine reads: without this the readback can race the GEMV and the
// "logits" the CPU samples from are whatever the previous token left in the buffer.
[[maybe_unused]] static void bar_compute_to_transfer(VkCommandBuffer cb) {
    VkMemoryBarrier mb{VK_STRUCTURE_TYPE_MEMORY_BARRIER};
    mb.srcAccessMask = VK_ACCESS_SHADER_WRITE_BIT;
    mb.dstAccessMask = VK_ACCESS_TRANSFER_READ_BIT;
    vkCmdPipelineBarrier(cb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_TRANSFER_BIT,
                         0, 1, &mb, 0, nullptr, 0, nullptr);
}

static std::string fmt1(double v) {
    char b[32]; snprintf(b, sizeof b, "%.1f", v); return std::string(b);
}

// The engine half of bonsai_vk.exe. Everything below this include is the kernel laboratory; this
// header is the 4 GB weight path (place -> SDMA -> repack -> verify) that the forward rides on.
// Included here, after Ctx/Buf/make_*/bar_* exist, because it is built out of exactly those.
#include "bonsai_weights.h"
// The whole-token forward rides on the same arena discipline the loader just built: one more
// device-local allocation for activations and state, offsets in push constants, one submit per token.
#include "bonsai_forward.h"
// The GUI shell. Last, because it is built out of Engine/Fwd/fwd_run and the chat template above, and
// because it registers itself through g_serve_hook rather than reaching into any of them.
#include "bonsai_http.h"

int main(int argc, char **argv) {    // unbuffered: every number in this tool is printed next to a loop that can crash the process,
    // and a block-buffered stdout loses the whole transcript on a segfault (which is exactly how
    // the first BONSAI_SYNC run presented as "no output at all").
    setvbuf(stdout, nullptr, _IONBF, 0);
    // "gen" is not a number, so the benchmark's M/K parsing below cannot eat it: the engine mode
    // takes over before any benchmark state exists.
    if (argc > 1 && !strcmp(argv[1], "gen")) return weights_main(argc, argv);
    // "run" is the heart: the same loaded weights, driven token by token by our own forward.
    if (argc > 1 && !strcmp(argv[1], "run")) return fwd_main(argc, argv);
    // "serve" is the same bring-up with an HTTP loop instead of one CLI reply. No arguments at all is
    // the double-click case, and a browser is the only interface a double-click can offer, so it takes
    // the same path. The benchmark modes below stay reachable by passing M explicitly.
    if (argc == 1 || !strcmp(argv[1], "serve")) {
        // A chat needs more room than the benchmark's 256-slot default: 256 positions is a prompt and
        // one short answer. 2048 costs 101.339 MiB of activations (measured, KV_P0P1P2_MEASURED.md),
        // which fits beside the 2467.7 MiB of weights on a 4 GB card. It is a DEFAULT, not a limit:
        // BONSAI_CAP still wins, and the init OOM fallback still shrinks it if the driver says no.
        // The `run` regression is untouched by this - it never enters this branch.
        if (!getenv("BONSAI_CAP")) {
            bonsai_setenv("BONSAI_CAP", "2048");
            printf("  [serve] BONSAI_CAP unset: defaulting to 2048 slots for a conversation"
                   " (set BONSAI_CAP to override)\n");
        }
        g_serve_hook = &serve_loop;
        // Re-shape argv into what fwd_main already parses, so the engine init, every env knob, every
        // gate and every printed budget line are the SAME code the CLI runs. Nothing is duplicated
        // here, including the default model path: a missing argument stays missing and fwd_main
        // supplies its own default.
        const int base = argc == 1 ? 1 : 2;      // first user argument after the mode word
        char *av[4];
        int ac = 2;
        av[0] = argv[0];
        av[1] = const_cast<char *>("run");
        if (argc > base)     { av[2] = argv[base];         ac = 3; }
        if (argc > base + 1) { av[3] = argv[base + 1];     ac = 4; }
        return fwd_main(ac, av);
    }
    uint32_t M = argc > 1 ? uint32_t(atoi(argv[1])) : 12288u;
    uint32_t K = argc > 2 ? uint32_t(atoi(argv[2])) : 4096u;
    uint32_t S = argc > 3 ? uint32_t(atoi(argv[3])) : 4u;
    uint32_t ITER = argc > 4 ? uint32_t(atoi(argv[4])) : 100u;
    if (K % 128 || (K / 128) % S) { fprintf(stderr, "need K%%128==0 and (K/128)%%%u==0\n", S); return 1; }
    // BONSAI_PARTIAL=1 is the diagnosis mode: force one quant block per split-K slot and skip the
    // reduce, so out[s*M+row] IS block s's contribution. That turns "rel=5.8e-01, diff=+lo-half"
    // into a per-block table and says WHICH block's term is wrong instead of implying one.
    const bool partial = getenv("BONSAI_PARTIAL") != nullptr;
    if (partial) S = K / 128u;

    Ctx c;
    // ---- instance / device ---------------------------------------------------------
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    // BONSAI_API12=1 asks for Vulkan 1.2 instead of 1.1. That is not cosmetic: vkResetQueryPool is
    // core in 1.2, and with a 1.1 instance vkGetDeviceProcAddr() hands back NULL for it, which is
    // the most likely explanation for the PC=0 crash seen through the MinGW import-lib stub.
    ai.apiVersion = getenv("BONSAI_API12") ? VK_API_VERSION_1_2 : VK_API_VERSION_1_1;
    ai.pApplicationName = "bonsai_vk";
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    const char *val = "VK_LAYER_KHRONOS_validation";
    bool want_val = getenv("VK_VALIDATION") != nullptr;
    if (want_val) { ici.enabledLayerCount = 1; ici.ppEnabledLayerNames = &val; }
    VkResult r = vkCreateInstance(&ici, nullptr, &c.inst);
    if (r != VK_SUCCESS && want_val) {
        printf("validation layer unavailable, continuing without it\n");
        ici.enabledLayerCount = 0; VK_CHECK(vkCreateInstance(&ici, nullptr, &c.inst));
    } else if (r != VK_SUCCESS) { fprintf(stderr, "no Vulkan runtime: %d\n", (int)r); return 1; }

    uint32_t ndev = 0; VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &ndev, nullptr));
    std::vector<VkPhysicalDevice> devs(ndev); VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &ndev, devs.data()));
    c.phys = devs[0];
    for (auto p : devs) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pp.vendorID == 0x1002) c.phys = p;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.phys, &props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mp);
    c.limits = props.limits;
    // device-local heap, read the honest way (heap of the first DEVICE_LOCAL memory type) -
    // the old printf printed maxMemoryAllocationCount under a "MiB" label, which is a bug in
    // the reporting, not a measurement.
    uint64_t ldd_heap = 0;
    for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i)
        if (c.mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            ldd_heap = c.mp.memoryHeaps[c.mp.memoryTypes[i].heapIndex].size; break;
        }
    printf("device: %s | driver %u.%u.%u | device-local heap %u MiB | timestampPeriod %.3f ns\n",
           props.deviceName,
           VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion),
           VK_VERSION_PATCH(props.driverVersion), unsigned(ldd_heap / 1048576), c.limits.timestampPeriod);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, qf.data());
    // Print every family: whether a copy can overlap a compute depends on there being a second,
    // TRANSFER-capable family to put it on. This is the fact that decides the tiering/expert-prefetch
    // designs in the DeepInfer evaluation, so it is measured, not assumed.
    for (uint32_t i = 0; i < nq; ++i)
        printf("  queue family %u: flags=0x%X %s%s%s%s%s  tiler=%u  contexts=%u\n", i, qf[i].queueFlags,
               (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) ? "COMPUTE " : "",
               (qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) ? "TRANSFER " : "",
               (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) ? "GRAPHICS " : "",
               (qf[i].queueFlags & VK_QUEUE_SPARSE_BINDING_BIT) ? "SPARSE " : "",
               (qf[i].queueFlags & VK_QUEUE_PROTECTED_BIT) ? "PROTECTED " : "",
               qf[i].timestampValidBits, qf[i].minImageTransferGranularity.width);
    for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qfi = i; break; }
    // a TRANSFER-only family (AMD's SDMA rings, 8 contexts on this card) is what makes an async
    // copy able to run *under* compute instead of queueing behind it. Look for it explicitly.
    for (uint32_t i = 0; i < nq; ++i)
        if ((qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) && !(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) &&
            qf[i].queueCount > 0) { c.tfi = i; break; }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi[2] = {};
    qi[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi[0].queueFamilyIndex = c.qfi; qi[0].queueCount = 1; qi[0].pQueuePriorities = &prio;
    uint32_t nqi = 1;
    if (c.tfi != 0xFFFFFFFFu) {
        qi[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi[1].queueFamilyIndex = c.tfi; qi[1].queueCount = 1; qi[1].pQueuePriorities = &prio;
        nqi = 2;
    }
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = nqi; di.pQueueCreateInfos = qi;
    VK_CHECK(vkCreateDevice(c.phys, &di, nullptr, &c.dev));
    vkGetDeviceQueue(c.dev, c.qfi, 0, &c.queue);
    if (c.tfi != 0xFFFFFFFFu) vkGetDeviceQueue(c.dev, c.tfi, 0, &c.qtx);
    printf("  compute queue = family %u, transfer-only queue = family %u%s\n", c.qfi, c.tfi,
           c.tfi == 0xFFFFFFFFu ? " (NONE - copies would serialise behind compute)" : "");

    // ---- arena: one allocation, many buffers ----------------------------------------
    // xsum is K/128 floats; out is M*S floats; ref is read through the staging allocation.
    const uint64_t src_sz  = uint64_t(M) * (K / 128u) * PTQ_BLOCK_BYTES;
    const uint64_t code_sz = uint64_t(M) * (K / 64u) * 16u;
    const uint64_t scl_sz  = uint64_t(M) * (K / 128u) * 4u;
    const uint64_t x_sz    = uint64_t(K) * 4u;
    const uint64_t xs_sz   = uint64_t(K / 128u) * 4u;
    const uint64_t out_sz  = uint64_t(M) * S * 4u;
    Buf bufs[6] = {};
    const uint64_t sizes[6] = {src_sz, code_sz, scl_sz, x_sz, xs_sz, out_sz};
    const char *names[6] = {"ptq_src", "codes", "scales", "x", "xsum", "out"};
    // OPTIMISATION 1 (ROI #1 of DEEPINFER_EVAL §6): every buffer that the copy engine touches from
    // now on is VK_SHARING_MODE_CONCURRENT over {compute family, transfer-only family}, because a
    // buffer can only be read/written by two different queues if it was created that way. EXCLUSIVE
    // would force a queue-ownership transfer barrier on every handoff, and on this driver that is
    // the thing we are trying not to pay for. When the device has no transfer-only family we keep
    // EXCLUSIVE and every copy stays on the compute queue (which §0 measured at 4.93/3.89 GB/s
    // instead of 7.12/7.15).
    const bool has_tx = (c.tfi != 0xFFFFFFFFu);
    uint32_t fam2[2] = {c.qfi, c.tfi};
    uint64_t cursor = 0, pad_total = 0;
    printf("\narena (single vkAllocateMemory, DEVICE_LOCAL):\n");
    for (int i = 0; i < 6; ++i) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = sizes[i];
        bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                   VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        if (has_tx) {
            bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fam2;
        } else {
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        }
        VK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &bufs[i].buf));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, bufs[i].buf, &mr);
        uint64_t aligned = align_up_u64(cursor, mr.alignment);
        pad_total += aligned - cursor;
        bufs[i].offset = aligned; bufs[i].size = sizes[i]; bufs[i].align = mr.alignment;
        cursor = aligned + mr.size;
        printf("  %-9s size=%10llu B  alignment=%llu  -> offset %llu\n", names[i],
               (unsigned long long)sizes[i], (unsigned long long)mr.alignment,
               (unsigned long long)aligned);
    }
    uint64_t arena_sz = cursor;
    printf("  tensors=%.1f MiB  alignment padding=%llu B (%.4f%%)  arena=%.1f MiB\n",
           double(arena_sz - pad_total) / 1048576.0, (unsigned long long)pad_total,
           100.0 * double(pad_total) / double(arena_sz), double(arena_sz) / 1048576.0);

    bool ok = false;
    for (uint32_t i = 0; i < c.mp.memoryTypeCount && !ok; ++i)
        if (c.mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            c.arena_type = i; ok = true;
        }
    if (!ok) { fprintf(stderr, "no DEVICE_LOCAL memory type\n"); return 1; }
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = arena_sz; ma.memoryTypeIndex = c.arena_type;
    VK_CHECK(vkAllocateMemory(c.dev, &ma, nullptr, &c.arena));
    for (int i = 0; i < 6; ++i) VK_CHECK(vkBindBufferMemory(c.dev, bufs[i].buf, c.arena, bufs[i].offset));

    // host-visible staging: upload blob + readback window + stage-diagnostic window, one allocation
    const uint64_t diag_base = src_sz + x_sz + out_sz;
    const uint64_t diag_sz = uint64_t(M) * 40u + uint64_t(K / 128u) * 4u + 1024u;
    const uint64_t host_sz = diag_base + diag_sz;
    ok = false;
    for (uint32_t i = 0; i < c.mp.memoryTypeCount && !ok; ++i) {
        VkMemoryPropertyFlags w = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((c.mp.memoryTypes[i].propertyFlags & w) == w) { c.host_type = i; ok = true; }
    }
    if (!ok) { fprintf(stderr, "no host-visible coherent memory type\n"); return 1; }
    VkMemoryAllocateInfo hma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    hma.allocationSize = host_sz; hma.memoryTypeIndex = c.host_type;
    VK_CHECK(vkAllocateMemory(c.dev, &hma, nullptr, &c.host_mem));

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = c.qfi;
    VK_CHECK(vkCreateCommandPool(c.dev, &cpi, nullptr, &c.cpool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = c.cpool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai, &c.cmd));

    // Created BEFORE the upload, and the order is not cosmetic: the upload brackets itself with
    // timestamps, and calling vkCmdResetQueryPool() with a still-NULL c.qp takes down amdvlk64.dll
    // with a SIGSEGV (backtrace ends at the loader thunk inside main()). Observed by bisecting a
    // crash that first looked like a queue-capability problem - it was a null handle, not a barrier.
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    // 512, not 2: BONSAI_SYNC brackets every token in its own pair of slots so two command
    // buffers can be in flight without trampling each other's timestamps, and BONSAI_PIPE brackets
    // all 201 per-token ops in their own pairs (402 slots) to get the true per-op GPU time
    // distribution instead of multiplying one op by 201.
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 512;
    VK_CHECK(vkCreateQueryPool(c.dev, &qpi, nullptr, &c.qp));

    // ---- synthetic data + upload -----------------------------------------------------
    const size_t n_src = size_t(src_sz);          // named, not parenthesised: (type(x)) would be
    std::vector<uint8_t> hsrc(n_src);             // the most vexing parse and declare a function
    std::vector<float> hx(K);
    uint32_t st = 0x2545F491u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };
    for (auto &v : hsrc) v = uint8_t(rnd());
    // every block's fp16 scale (bytes 26..27) = 1.0 so the reference and the GPU agree exactly.
    // BONSAI_RANDSCALE=1 replaces that with scales spread over [0.25,1) - exponent fields 14/15,
    // random mantissas - which is the only way to exercise the unpackHalf2x16 path in the GEMV:
    // with d == 1.0 everywhere a wrong d index or a wrong half-select is invisible.
    if (getenv("BONSAI_RANDSCALE")) {
        for (uint64_t o = 26; o + 1 < src_sz; o += PTQ_BLOCK_BYTES) {
            uint16_t h = uint16_t(((rnd() & 1u) ? 0x3800u : 0x3000u) | (rnd() & 0x3FFu));
            hsrc[o] = uint8_t(h & 0xFF); hsrc[o + 1] = uint8_t(h >> 8);
        }
    } else {
        for (uint64_t o = 26; o + 1 < src_sz; o += PTQ_BLOCK_BYTES) { hsrc[o] = 0x00; hsrc[o + 1] = 0x3C; }
    }
    for (auto &v : hx) v = (float(rnd() & 0xFFFF) / 32768.0f) - 1.0f;

    // The upload semaphore: recorded/created here, waited by the first compute submit (the repack
    // block below). This is the H2D half of optimisation 1 - the 24 MB weight blob now crosses
    // PCIe on the SDMA ring instead of behind a compute-queue copy.
    VkSemaphore up_sig = VK_NULL_HANDLE;
    VkFence up_fin = VK_NULL_HANDLE;
    {
        void *p = nullptr; VK_CHECK(vkMapMemory(c.dev, c.host_mem, 0, host_sz, 0, &p));
        memcpy(p, hsrc.data(), size_t(src_sz));
        memcpy((char *)p + src_sz, hx.data(), size_t(x_sz));
        vkUnmapMemory(c.dev, c.host_mem);
        VkCommandPoolCreateInfo cpi2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
        cpi2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
        cpi2.queueFamilyIndex = has_tx ? c.tfi : c.qfi;
        VkCommandPool upool; VK_CHECK(vkCreateCommandPool(c.dev, &cpi2, nullptr, &upool));
        VkCommandBufferAllocateInfo cai2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai2.commandPool = upool; cai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai2.commandBufferCount = 1;
        VkCommandBuffer ucb; VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, &ucb));
        VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(ucb, &beg));
        // dstOffset is relative to the destination BUFFER, not to the arena: the buffers are
        // already bound at their arena offsets. bufs[0].offset happened to be 0 so the weight
        // upload looked fine; the x copy targeted offset 25165824 inside a 16 KB buffer and the
        // driver dropped it. Same class of mistake as the descriptor offset above.
        VkBufferCopy c1{0, 0, src_sz}, c2{src_sz, 0, x_sz};
        // the staging memory is already visible as buffers bufs[] in the arena; to copy out of
        // host memory we need a buffer bound to host_mem. It is CONCURRENT for the same reason
        // the arena is: the queue that reads it is no longer the queue that computes.
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = host_sz; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
        if (has_tx) {
            bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fam2;
        } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer stage; VK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &stage));
        VK_CHECK(vkBindBufferMemory(c.dev, stage, c.host_mem, 0));
        vkCmdResetQueryPool(ucb, c.qp, 96, 2);
        vkCmdWriteTimestamp(ucb, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, 96);
        vkCmdCopyBuffer(ucb, stage, bufs[0].buf, 1, &c1);
        vkCmdCopyBuffer(ucb, stage, bufs[3].buf, 1, &c2);
        vkCmdWriteTimestamp(ucb, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, 97);
        // No queue-ownership or TRANSFER->COMPUTE barrier is recorded here. The handoff is already
        // safe: this submit is fence-waited before the compute submit is even recorded, and it also
        // signals the semaphore the compute side waits on. Note for the record what the crash of
        // this block actually was, because the first diagnosis was wrong: a barrier recorded on the
        // transfer-only queue looked guilty, but bisecting with checkpoints showed the fault landed
        // on vkCmdResetQueryPool() with a still-NULL c.qp - the upload ran before the query pool
        // existed. The query pool is now created before this block, and whether a lone barrier on
        // family 2 is survivable is decided by the BONSAI_TXBARRIER probe at the end of BONSAI_PIPE
        // rather than asserted here.
        VK_CHECK(vkEndCommandBuffer(ucb));
        VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
        VK_CHECK(vkCreateSemaphore(c.dev, &sci, nullptr, &up_sig));
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        VK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &up_fin));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
        si.commandBufferCount = 1; si.pCommandBuffers = &ucb;
        si.signalSemaphoreCount = 1; si.pSignalSemaphores = &up_sig;
        auto up0 = std::chrono::steady_clock::now();
        VK_CHECK(vkQueueSubmit(has_tx ? c.qtx : c.queue, 1, &si, up_fin));
        VK_CHECK(vkWaitForFences(c.dev, 1, &up_fin, VK_TRUE, ~0ull));
        auto up1 = std::chrono::steady_clock::now();
        uint64_t uts[2];
        VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, 96, 2, sizeof(uts), uts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double eng_us = double(uts[1] - uts[0]) * c.limits.timestampPeriod / 1000.0;
        double wall_us = std::chrono::duration<double, std::micro>(up1 - up0).count();
        printf("  upload %s (family %u): %.1f MiB, engine %.1f us = %.2f GB/s, wall %.1f us\n",
               has_tx ? "on the transfer-only queue" : "on the compute queue",
               has_tx ? c.tfi : c.qfi,
               double(src_sz + x_sz) / 1048576.0, eng_us,
               double(src_sz + x_sz) / (eng_us * 1e-6) / 1e9, wall_us);
        vkDestroyFence(c.dev, up_fin, nullptr); up_fin = VK_NULL_HANDLE;
        vkDestroyBuffer(c.dev, stage, nullptr);
        vkDestroyCommandPool(c.dev, upool, nullptr);
    }

    // ---- pipelines -------------------------------------------------------------------
    // Persistent host-visible window for the M result floats (49 KB at M=12288). The BONSAI_SYNC
    // probe below needs the readback INSIDE the timed loop - that copy is the thing a real server
    // waits for between the last projection and the sampler - so it cannot be created and
    // destroyed per iteration the way the one-shot diag/readback blocks do.
    VkBuffer readback = VK_NULL_HANDLE;
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = uint64_t(M) * 8u;         // two slots, so a 2-deep pipeline never reads a torn row
        bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
        if (has_tx) {
            bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fam2;
        } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &readback));
        VK_CHECK(vkBindBufferMemory(c.dev, readback, c.host_mem, diag_base));
    }

    VkDescriptorSetLayout sl[4];
    sl[0] = make_set_layout(c, 3);   // repack : src, codes, scales
    sl[1] = make_set_layout(c, 2);   // xsum   : x, xsum
    sl[2] = make_set_layout(c, 5);   // gemv   : codes, x, scales, xsum, out
    sl[3] = make_set_layout(c, 1);   // reduce : out
    VkPipelineLayout pl[4];
    for (int i = 0; i < 4; ++i) pl[i] = make_pipe_layout(c, sl[i], 32);
    VkPipeline pp[4];
    pp[0] = make_pipeline(c, "shaders/repack_ptq1_0.spv", pl[0]);
    pp[1] = make_pipeline(c, "shaders/xsum.spv", pl[1]);
    pp[2] = make_pipeline(c, "shaders/gemv_ternary.spv", pl[2]);
    pp[3] = make_pipeline(c, "shaders/reduce.spv", pl[3]);

    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 8; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    VK_CHECK(vkCreateDescriptorPool(c.dev, &dpi, nullptr, &c.dpool));
    VkDescriptorSet ds[4];
    for (int i = 0; i < 4; ++i) {
        VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sai.descriptorPool = c.dpool; sai.descriptorSetCount = 1; sai.pSetLayouts = &sl[i];
        VK_CHECK(vkAllocateDescriptorSets(c.dev, &sai, &ds[i]));
    }
    bind_set(c, ds[0], {bufs[0], bufs[1], bufs[2]});
    bind_set(c, ds[1], {bufs[3], bufs[4]});
    bind_set(c, ds[2], {bufs[1], bufs[3], bufs[2], bufs[4], bufs[5]});
    bind_set(c, ds[3], {bufs[5]});

    uint32_t pc[8];
    auto push = [&](VkCommandBuffer cb, int which, int bytes) {
        vkCmdPushConstants(cb, pl[which], VK_SHADER_STAGE_COMPUTE_BIT, 0, bytes, pc);
    };
    auto bind_pipe = [&](VkCommandBuffer cb, int which) {
        vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pp[which]);
        vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, pl[which], 0, 1, &ds[which], 0, nullptr);
    };
    // the per-token sequence: xsum -> gemv ->(reduce). repack runs once, up front.
    // bar_compute() between every pair: within one command buffer a dispatch does not wait for the
    // previous dispatch's writes without an explicit dependency.
    auto dispatch_token = [&](VkCommandBuffer cb) {
        memset(pc, 0, sizeof(pc)); pc[0] = 0; pc[1] = 0; pc[2] = K / 128u;      // xsum PC
        bind_pipe(cb, 1); push(cb, 1, 32);
        vkCmdDispatch(cb, K / 128u, 1, 1);
        bar_compute(cb);

        memset(pc, 0, sizeof(pc)); pc[0] = M; pc[1] = K; pc[2] = S;             // gemv PC
        bind_pipe(cb, 2); push(cb, 2, 32);
        vkCmdDispatch(cb, (M + 63) / 64u, S, 1);
        bar_compute(cb);

        if (S > 1 && !partial) {                                          // reduce PC
            memset(pc, 0, sizeof(pc)); pc[0] = M; pc[1] = S;
            bind_pipe(cb, 3); push(cb, 3, 32);
            vkCmdDispatch(cb, (M + 63) / 64u, 1, 1);
            bar_compute(cb);
        }
    };
    // record one token into cb and close it, for the per-token-submit measurements below.
    // rb_slot >= 0 additionally copies the M result floats out to mapped host memory at slot
    // rb_slot - that copy is what a real server waits for between the last projection and the
    // sampler, so it has to be inside the timed loop, not outside it.
    auto record_one = [&](VkCommandBuffer cb_, uint32_t q0, uint32_t q1, int rb_slot) {
        VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(cb_, &beg));
        vkCmdResetQueryPool(cb_, c.qp, q0, 1);
        vkCmdResetQueryPool(cb_, c.qp, q1, 1);
        vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, q0);
        dispatch_token(cb_);
        if (rb_slot >= 0) {
            bar_compute_to_transfer(cb_);
            VkBufferCopy cp{0, uint64_t(rb_slot) * M * 4u, uint64_t(M) * 4u};
            vkCmdCopyBuffer(cb_, bufs[5].buf, readback, 1, &cp);
        }
        vkCmdWriteTimestamp(cb_, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, q1);
        VK_CHECK(vkEndCommandBuffer(cb_));
    };

    // repack once, bracketed by timestamps so the one-time cost is measured too
    {
        VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cmd, &beg));
        vkCmdResetQueryPool(c.cmd, c.qp, 0, 2);
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 0);
        memset(pc, 0, sizeof(pc)); pc[0] = M; pc[1] = K; pc[2] = K / 128u; pc[7] = M;   // repack PC
        // pc[6]=row0 stays 0 and pc[7]=row_end is M: repack_ptq1_0.comp now takes an explicit row
        // range so the full-model loader can stream big tensors through a 32 MiB scratch. The
        // one-tensor benchmark is the whole tensor, so the range is [0, M).
        bind_pipe(c.cmd, 0); push(c.cmd, 0, 32);
        vkCmdDispatch(c.cmd, K / 128u, (M + 63) / 64u, 1);
        bar_compute(c.cmd);        // the token dispatch below reads codes/scales the repack just wrote
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 1);
        dispatch_token(c.cmd);
        VK_CHECK(vkEndCommandBuffer(c.cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
        // wait the upload: it ran on the other queue, so without this the repack could read the
        // weights before the SDMA wrote them. This is the handoff the CONCURRENT sharing mode buys.
        VkPipelineStageFlags up_stg = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
        if (up_sig != VK_NULL_HANDLE) {
            si.waitSemaphoreCount = 1; si.pWaitSemaphores = &up_sig; si.pWaitDstStageMask = &up_stg;
        }
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr)); VK_CHECK(vkQueueWaitIdle(c.queue));
        vkDestroySemaphore(c.dev, up_sig, nullptr); up_sig = VK_NULL_HANDLE;
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, 0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        double ms = double(ts[1] - ts[0]) * c.limits.timestampPeriod / 1e6;
        printf("repack: %.1f MB -> %.1f MB in %.2f ms (%.1f GB/s effective, one-time cost)\n",
               src_sz / 1e6, code_sz / 1e6, ms, double(src_sz + code_sz) / (ms * 1e-3) / 1e9);
    }

    // ---- stage diagnostics ---------------------------------------------------------------
    // Written because "correctness rel=5e16" does not say WHICH kernel is wrong. This reads the
    // arena back and checks xsum, the scales and the first 64 codes of block 0 for every row,
    // element by element, against the same CPU decode the reference uses. Enable with
    // BONSAI_DIAG=1.
    if (getenv("BONSAI_DIAG")) {
        const uint64_t d_codes = uint64_t(M) * 32ull;   // words 0 and 1 = codes 0..127 of block 0
        const uint64_t d_scl   = uint64_t(M) * 8ull;    // scales of blocks 0 and 1, every row
        const uint64_t d_xs    = uint64_t(K / 128u) * 4ull;
        const uint64_t o_src   = d_codes + d_scl + d_xs;   // first 512 B of the uploaded PTQ source
        const uint64_t d_src   = 512;
        const uint64_t o_x     = o_src + d_src;            // first 64 floats of the uploaded x
        const uint64_t d_all   = o_x + 256;
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = d_all; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT;
        bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer db; VK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &db));
        VK_CHECK(vkBindBufferMemory(c.dev, db, c.host_mem, diag_base));
        VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cmd, &beg));
        VkBufferCopy cc{0, 0, d_codes};            vkCmdCopyBuffer(c.cmd, bufs[1].buf, db, 1, &cc);
        cc = {0, d_codes, d_scl};                  vkCmdCopyBuffer(c.cmd, bufs[2].buf, db, 1, &cc);
        cc = {0, d_codes + d_scl, d_xs};           vkCmdCopyBuffer(c.cmd, bufs[4].buf, db, 1, &cc);
        cc = {0, o_src, d_src};                    vkCmdCopyBuffer(c.cmd, bufs[0].buf, db, 1, &cc);
        cc = {0, o_x, 256};                        vkCmdCopyBuffer(c.cmd, bufs[3].buf, db, 1, &cc);
        VK_CHECK(vkEndCommandBuffer(c.cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr)); VK_CHECK(vkQueueWaitIdle(c.queue));
        std::vector<uint8_t> raw; raw.resize(size_t(d_all));
        void *p = nullptr; VK_CHECK(vkMapMemory(c.dev, c.host_mem, diag_base, d_all, 0, &p));
        memcpy(raw.data(), p, size_t(d_all)); vkUnmapMemory(c.dev, c.host_mem);
        vkDestroyBuffer(c.dev, db, nullptr);

        const uint32_t *gc = reinterpret_cast<const uint32_t *>(raw.data());
        const uint32_t *gs = reinterpret_cast<const uint32_t *>(raw.data() + d_codes);
        const float    *gx = reinterpret_cast<const float *>(raw.data() + d_codes + d_scl);

        // (0) did the upload land at all? this is the check that separates "kernel bug" from
        // "the arena never received the data" - the two look identical from the outside.
        const uint8_t *gsrc = raw.data() + o_src;
        int src_bad = -1;
        for (uint32_t i = 0; i < d_src && src_bad < 0; ++i) if (gsrc[i] != hsrc[i]) src_bad = int(i);
        printf("[diag] upload: %s", src_bad < 0 ? "ptq_src bytes 0..511 match host exactly\n"
                                                : "MISMATCH\n");
        if (src_bad >= 0)
            printf("         byte %d gpu=0x%02X cpu=0x%02X | gpu[0..7]=%02X %02X %02X %02X %02X %02X %02X %02X\n",
                   src_bad, gsrc[src_bad], hsrc[src_bad], gsrc[0], gsrc[1], gsrc[2], gsrc[3],
                   gsrc[4], gsrc[5], gsrc[6], gsrc[7]);
        const float *gxv = reinterpret_cast<const float *>(raw.data() + o_x);
        int xv_bad = -1;
        for (uint32_t i = 0; i < 64 && xv_bad < 0; ++i) if (std::fabs(gxv[i] - hx[i]) > 1e-9f) xv_bad = int(i);
        printf("[diag] x     : %s", xv_bad < 0 ? "floats 0..63 match host exactly\n" : "MISMATCH\n");
        if (xv_bad >= 0) printf("         float %d gpu=%.6f cpu=%.6f\n", xv_bad, gxv[xv_bad], hx[xv_bad]);

        // (a) xsum
        int xb_bad = -1; double xb_gpu = 0, xb_cpu = 0;
        for (uint32_t b = 0; b < K / 128u && xb_bad < 0; ++b) {
            double s = 0.0; for (uint32_t e = 0; e < 128; ++e) s += double(hx[b * 128u + e]);
            if (std::fabs(s - double(gx[b])) > 1e-3) { xb_bad = int(b); xb_gpu = gx[b]; xb_cpu = s; }
        }
        printf("[diag] xsum  : %s%s\n", xb_bad < 0 ? "all K/128 blocks match CPU" : "MISMATCH",
               xb_bad < 0 ? "" : "");
        if (xb_bad >= 0) printf("         block %d gpu=%.6f cpu=%.6f\n", xb_bad, xb_gpu, xb_cpu);

        // (b) scales: blocks 0 and 1 were synthesised with f16 1.0 = 0x3C00, and the layout is
        // [blk * M + row], so block 1 is the second M-sized group of the window we grabbed.
        int sb_bad = -1;
        uint32_t sb_gpu = 0, sb_want = 0;
        for (uint64_t i = 0; i < 2ull * M && sb_bad < 0; ++i) {
            const uint64_t blk = i / M, row = i % M;                        // layout is [blk * M + row]
            const uint8_t *pb = hsrc.data() + (row * (K / 128u) + blk) * PTQ_BLOCK_BYTES;
            const uint32_t want = uint32_t(pb[26]) | (uint32_t(pb[27]) << 8);   // f16 bits, low half
            if ((gs[i] & 0xFFFFu) != want) { sb_bad = int(i); sb_gpu = gs[i] & 0xFFFFu; sb_want = want; }
        }
        printf("[diag] scale : %s\n", sb_bad < 0 ? "blocks 0-1, every row carry the source f16 bits"
                                                 : "MISMATCH");
        if (sb_bad >= 0)
            printf("         blk=%d row=%d gpu=0x%04X want 0x%04X\n", sb_bad / int(M), sb_bad % int(M),
                   sb_gpu, sb_want);

        // (c) all 128 codes of block 0, every row: rebuild the 8 expected u32 from the source
        // bytes. Words 0 and 1 of the destination are M uvec4 apart, so index them separately -
        // this is the path that covers elements 80..119 (qs[16..23]) and 120..127 (qh).
        auto gpu_word = [&](uint32_t r, int q) -> uint32_t {
            return (q < 4) ? gc[r * 4u + q] : gc[(M + r) * 4u + (q - 4)];
        };
        int cb_bad = -1, cb_word = 0, cb_row = 0;
        uint32_t cb_gpu = 0, cb_cpu = 0;
        for (uint32_t r = 0; r < M && cb_bad < 0; ++r) {
            const uint8_t *pb = hsrc.data() + size_t(r) * (K / 128u) * PTQ_BLOCK_BYTES;   // block 0
            uint32_t exp[8] = {};
            for (int e = 0; e < 128; ++e) {
                int by, tr; elem_to_byte_trit(e, by, tr);
                exp[e >> 4] |= uint32_t(trit_of_byte(pb[by], tr)) << uint32_t((e & 15) * 2);
            }
            for (int q = 0; q < 8; ++q)
                if (gpu_word(r, q) != exp[q]) { cb_bad = 1; cb_row = int(r); cb_word = q;
                                                cb_gpu = gpu_word(r, q); cb_cpu = exp[q]; break; }
        }
        printf("[diag] codes : %s\n", cb_bad < 0 ? "block 0 words 0..7 match CPU exactly for every row"
                                                 : "MISMATCH");
        if (cb_bad >= 0) {
            printf("         row %d word %d (codes %d..%d) gpu=0x%08X cpu=0x%08X\n", cb_row, cb_word,
                   cb_word * 16, cb_word * 16 + 15, cb_gpu, cb_cpu);
            printf("         gpu trits:");
            for (int t = 0; t < 16; ++t) printf(" %u", (cb_gpu >> (2 * t)) & 3u);
            printf("\n         cpu trits:");
            for (int t = 0; t < 16; ++t) printf(" %u", (cb_cpu >> (2 * t)) & 3u);
            printf("\n");
        }
    }

    // ---- correctness ------------------------------------------------------------------
    std::vector<float> ref(M);
    // cref[blk * M + row] only filled in partial mode; it is what the GPU's split-blk slot must equal.
    std::vector<float> cref(size_t(partial ? M * (K / 128u) : 0));
    for (uint32_t row = 0; row < M; ++row) {
        double acc = 0.0;
        for (uint32_t blk = 0; blk < K / 128u; ++blk) {
            const uint8_t *pb = hsrc.data() + size_t(row) * (K / 128u) * PTQ_BLOCK_BYTES + size_t(blk) * PTQ_BLOCK_BYTES;
            float d = f16_to_f32(uint16_t(pb[26]) | (uint16_t(pb[27]) << 8));
            double cs = 0.0, xs = 0.0;
            for (int e = 0; e < 128; ++e) {
                int by, tr; elem_to_byte_trit(e, by, tr);
                float x = hx[blk * 128u + e];
                cs += double(trit_of_byte(pb[by], tr)) * x;
                xs += x;
            }
            double term = double(d) * (cs - xs);
            if (partial) cref[size_t(blk) * M + row] = float(term);
            acc += term;
        }
        ref[row] = float(acc);
    }
    const uint64_t read_n = partial ? uint64_t(M) * S : uint64_t(M);
    const size_t n_got = size_t(read_n);          // named: got(size_t(read_n)) is the most vexing
    std::vector<float> got(n_got);                // parse again and declares a function
    {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = out_sz; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT; bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer rb; VK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &rb));
        VK_CHECK(vkBindBufferMemory(c.dev, rb, c.host_mem, 0));
        VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cmd, &beg));
        VkBufferCopy cp{0, 0, uint32_t(read_n * 4u)};   // srcOffset 0: the out buffer starts at its own 0
        vkCmdCopyBuffer(c.cmd, bufs[5].buf, rb, 1, &cp);
        VK_CHECK(vkEndCommandBuffer(c.cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr)); VK_CHECK(vkQueueWaitIdle(c.queue));
        void *p = nullptr; VK_CHECK(vkMapMemory(c.dev, c.host_mem, 0, host_sz, 0, &p));
        memcpy(got.data(), p, size_t(read_n) * 4u);
        vkUnmapMemory(c.dev, c.host_mem);
        vkDestroyBuffer(c.dev, rb, nullptr);
    }
    if (partial) {
        // One line per quant block: what the GPU put in split slot b vs what block b alone is worth.
        // xsum is row independent, so an xsum-side bug shows as a constant err within a row of this
        // table; a codes/scales-side bug shows as a per-row spread. The two are not confusable here.
        printf("[part]  blk   xs(blk)   gpu(row0)   ref(row0)   err(row0)   max|err|   rows!=row0\n");
        for (uint32_t b = 0; b < S; ++b) {
            double xs = 0.0; for (uint32_t e = 0; e < 128; ++e) xs += double(hx[b * 128u + e]);
            double e0 = double(got[b * M]) - double(cref[b * M]);
            double mx = 0.0; int nd = 0;
            for (uint32_t r = 0; r < M; ++r) {
                double d = double(got[size_t(b) * M + r]) - double(cref[size_t(b) * M + r]);
                if (std::fabs(d) > mx) mx = std::fabs(d);
                if (std::fabs(d - e0) > 1e-3) ++nd;
            }
            printf("[part]  %3u  %9.4f  %11.4f  %10.4f  %9.4f  %8.4f  %6d\n", b, xs,
                   double(got[b * M]), double(cref[b * M]), e0, mx, nd);
        }
        double sum_bad = 0.0; int nb = 0;
        double total0 = 0.0;
        for (uint32_t b = 0; b < S; ++b) {
            double e0 = double(got[b * M]) - double(cref[b * M]);
            if (std::fabs(e0) > 1e-3) { sum_bad += e0; ++nb; }
            total0 += double(got[b * M]);
        }
        printf("[part]  blocks with a row0 error: %d/%u, sum of their errors = %.4f | sum(gpu)=%.4f ref=%.4f\n",
               nb, S, sum_bad, total0, double(ref[0]));
    }
    // The checks below are written against the final per-row result. In partial mode the buffer
    // holds un-reduced per-block slots, so sum them here - that way both modes compare the same
    // quantity and a partial-mode run still says whether the total is right.
    std::vector<float> fin(M, 0.0f);
    if (partial) {
        for (uint32_t b = 0; b < S; ++b)
            for (uint32_t r = 0; r < M; ++r) fin[r] = float(double(fin[r]) + double(got[size_t(b) * M + r]));
    } else {
        for (uint32_t r = 0; r < M; ++r) fin[r] = got[r];
    }
    double maxerr = 0.0, maxmag = 0.0;
    for (uint32_t i = 0; i < M; ++i) {
        maxerr = std::max(maxerr, double(std::fabs(fin[i] - ref[i])));
        maxmag = std::max(maxmag, double(std::fabs(ref[i])));
    }
    printf("correctness: max|err|=%.5f  max|ref|=%.1f  rel=%.2e  (expect ~1e-6 rel in f32 over K=%u)\n",
           maxerr, maxmag, maxerr / (maxmag + 1e-9), K);
    if (getenv("BONSAI_DIAG")) {
        // Is the error row-independent (an additive term) or row-dependent (a data/layout bug)?
        // min == max <=> every row is off by the same number, which only a term that does not
        // involve the weights can produce.
        double xsum_all = 0.0, ax = 0.0;
        for (float v : hx) { xsum_all += double(v); ax += std::fabs(double(v)); }
        double dmin = 1e30, dmax = -1e30, dsum = 0.0; int ndiff = 0;
        for (uint32_t i = 0; i < M; ++i) {
            double d = double(fin[i]) - double(ref[i]);
            if (d < dmin) dmin = d;
            if (d > dmax) dmax = d;
            dsum += d;
            if (std::fabs(d - double(fin[0] - ref[0])) > 1e-3) ++ndiff;
        }
        printf("[diag] sum(x)=%.4f sum|x|=%.4f | diff min=%.4f max=%.4f mean=%.4f rows!=row0:%d/%u\n",
               xsum_all, ax, dmin, dmax, dsum / double(M), ndiff, M);
        // Which formula did the GPU actually evaluate? Reconstruct the plausible candidates for
        // row 0 and print them next to what came back.
        auto row_ref = [&](bool subtract, int nblk) {
            double acc = 0.0;
            for (int blk = 0; blk < nblk; ++blk) {
                const uint8_t *pb = hsrc.data() + size_t(blk) * PTQ_BLOCK_BYTES;
                float d = f16_to_f32(uint16_t(pb[26]) | (uint16_t(pb[27]) << 8));
                double cs = 0.0, xs = 0.0;
                for (int e = 0; e < 128; ++e) {
                    int by, tr; elem_to_byte_trit(e, by, tr);
                    float x = hx[blk * 128u + e];
                    cs += double(trit_of_byte(pb[by], tr)) * x; xs += x;
                }
                acc += double(d) * (cs - (subtract ? xs : 0.0));
            }
            return acc;
        };
        const int NB = int(K / 128u);
        // The GEMV evaluated  sum(c*x) - (something). got[0] tells us that something was
        // 34.3320 - 21.7847 = 12.5473 here, so print the block-sum subsets and find which one
        // that is: it identifies the index the shader really used for xsum.
        double s_even = 0, s_odd = 0, s_lo = 0, s_hi = 0, s_q[4] = {0, 0, 0, 0};
        for (int b = 0; b < NB; ++b) {
            double xs = 0; for (int e = 0; e < 128; ++e) xs += double(hx[b * 128u + e]);
            if (b & 1) s_odd += xs; else s_even += xs;
            if (b < NB / 2) s_lo += xs; else s_hi += xs;
            s_q[b % 4] += xs;
        }
        printf("[diag] xsum subsets: even=%.4f odd=%.4f lo-half=%.4f hi-half=%.4f mod4=%.4f/%.4f/%.4f/%.4f\n",
               s_even, s_odd, s_lo, s_hi, s_q[0], s_q[1], s_q[2], s_q[3]);
        printf("[diag] row0  gpu=%.4f | full=%.4f  no-minus=%.4f  half-blocks=%.4f  implied-subtract=%.4f\n",
               fin[0], row_ref(true, NB), row_ref(false, NB), row_ref(true, NB / 2),
               row_ref(false, NB) - double(fin[0]));
        for (uint32_t i = 0; i < 6; ++i)
            printf("         %2u  got=%10.4f ref=%10.4f d=%10.4f\n", i, fin[i], ref[i],
                   double(fin[i]) - double(ref[i]));
    }

    // ---- timing -----------------------------------------------------------------------
    double best = 1e30, best_wall = 1e30;
    for (int round = 0; round < 3; ++round) {
        VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
        beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
        VK_CHECK(vkBeginCommandBuffer(c.cmd, &beg));
        vkCmdResetQueryPool(c.cmd, c.qp, 0, 2);
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 0);
        for (uint32_t i = 0; i < ITER; ++i) dispatch_token(c.cmd);
        vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 1);
        VK_CHECK(vkEndCommandBuffer(c.cmd));
        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
        auto w0 = std::chrono::steady_clock::now();
        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr));
        VK_CHECK(vkQueueWaitIdle(c.queue));
        auto w1 = std::chrono::steady_clock::now();
        best_wall = std::min(best_wall,
                             std::chrono::duration<double, std::milli>(w1 - w0).count() / double(ITER));
        uint64_t ts[2];
        VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, 0, 2, sizeof(ts), ts, 8,
                                       VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
        best = std::min(best, double(ts[1] - ts[0]) * c.limits.timestampPeriod / 1e6 / double(ITER));
    }
    printf("M=%u K=%u splits=%u : %.3f ms/token-op  %.1f Gop/s  %.1f GB/s (packed weights)\n",
           M, K, S, best, double(M) * K / (best * 1e-3) / 1e9, code_sz / (best * 1e-3) / 1e9);
    // A GPU-bound batch is the calibration: there is nothing for the GPU to wait for, so wall clock
    // and the timestamp delta have to agree. They do, to ~0.3%, which is what licenses every
    // timestamp-derived number in this file - including the Gop/s table in STAGE3_RUNTIME.md.
    printf("  cross-check: wall %.3f ms vs timestamps %.3f ms = %.1f%% of wall -> limits.timestampPeriod "
           "%.1f ns is trustworthy here\n",
           best_wall, best, 100.0 * best / best_wall, c.limits.timestampPeriod);
    printf("WGSL reference on this card (probe 13): 143-155 Gop/s for these shapes.\n");
    printf("per-token projection cost for a 9B dense model: 9e9 MACs / rate; see report.\n");

    // ---- BONSAI_XFER=1: the measured link, not the bus spec --------------------------------
    // Questions A (tier KV / recurrent state into host RAM) and C (stream experts in) are both
    // "how fast do bytes cross the link" questions. STAGE0 quoted 7.88 GB/s, which is the PCIe 3.0
    // x8 *spec*; what a runtime actually gets on Windows/WDDM is lower and, more importantly, is
    // dominated by a per-submission cost at exactly the sizes KV paging moves. So measure the
    // shape: rate vs size, cached vs write-combined host memory, and the cost per scattered copy.
    if (getenv("BONSAI_XFER")) {
        printf("\n== host<->device transfer (BONSAI_XFER) ==\n");
        printf("  memory types:\n");
        for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i) {
            VkMemoryPropertyFlags f = c.mp.memoryTypes[i].propertyFlags;
            printf("    %u heap%u  %s%s%s%s%s  (%llu MiB heap)\n", i, c.mp.memoryTypes[i].heapIndex,
                   (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
                   (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
                   (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT) ? "HOST_COHERENT " : "",
                   (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) ? "HOST_CACHED " : "",
                   (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT) ? "LAZILY_ALLOCATED " : "",
                   (unsigned long long)(c.mp.memoryHeaps[c.mp.memoryTypes[i].heapIndex].size / 1048576));
        }
        auto find_type = [&](uint32_t need, uint32_t forbid) -> int {
            for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i) {
                VkMemoryPropertyFlags f = c.mp.memoryTypes[i].propertyFlags;
                if ((f & need) == need && !(f & forbid)) return int(i);
            }
            return -1;
        };
        const int t_dev  = find_type(VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
        const int t_wc   = find_type(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
        const int t_wb   = find_type(VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                     VK_MEMORY_PROPERTY_HOST_CACHED_BIT, 0);
        printf("  chosen: device-local=%d  host uncached(WC)=%d  host cached(WB)=%d\n", t_dev, t_wc, t_wb);

        const uint64_t CAP = 128ull << 20;
        struct Mem { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; };
        auto mk = [&](Mem &mm, uint64_t size, int type) -> bool {
            if (type < 0) return false;
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = size;
            bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
            bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
            if (vkCreateBuffer(c.dev, &bi, nullptr, &mm.b) != VK_SUCCESS) return false;
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, mm.b, &mr);
            if (mr.memoryTypeBits & (1u << type)) {
                VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                ma.allocationSize = std::max(mr.size, size); ma.memoryTypeIndex = uint32_t(type);
                if (vkAllocateMemory(c.dev, &ma, nullptr, &mm.m) == VK_SUCCESS &&
                    vkBindBufferMemory(c.dev, mm.b, mm.m, 0) == VK_SUCCESS) return true;
            }
            vkDestroyBuffer(c.dev, mm.b, nullptr); mm.b = VK_NULL_HANDLE;
            return false;
        };
        Mem dwe, dwl, hwc, hwb;
        bool have_dev = mk(dwe, CAP, t_dev) && mk(dwl, CAP, t_dev);
        bool have_wc  = mk(hwc, CAP, t_wc);
        bool have_wb  = mk(hwb, CAP, t_wb);
        // the host side the bulk tests use: cached (WB) if the driver exposes it, because that is
        // what a KV/state tier would actually live in; write-combined (WC) only as a comparison.
        Mem  *hs    = have_wb ? &hwb : (have_wc ? &hwc : nullptr);
        const char *hs_name = have_wb ? "WB(cached)" : (have_wc ? "WC(uncached)" : "none");
        if (!have_dev) printf("  no %llu MiB device-local pair available - skipping bulk sizes\n",
                              (unsigned long long)(CAP >> 20));

        VkCommandBuffer xcb = VK_NULL_HANDLE;
        VkCommandBufferAllocateInfo xai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        xai.commandPool = c.cpool; xai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; xai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(c.dev, &xai, &xcb));

        // one submit + waitIdle carrying `nreg` identical copies of `reg` bytes: returns GB/s and
        // the wall-clock per submission. Recording is outside the timer on purpose - the point is
        // the link and the WDDM round trip, and the record cost is measured by BONSAI_SYNC.
        auto bench = [&](VkBuffer src, VkBuffer dst, uint32_t nreg, uint64_t reg, double &ms_iter) -> double {
            // nreg, not size_t(nreg): the cast makes this look like a declaration of a function
            // named rc taking a parameter nreg of type size_t - the most vexing parse, third time.
            std::vector<VkBufferCopy> rc(nreg);
            for (uint32_t i = 0; i < nreg; ++i) {
                rc[i].srcOffset = uint64_t(i) * reg; rc[i].dstOffset = uint64_t(i) * reg; rc[i].size = reg;
            }
            const uint64_t moved = uint64_t(nreg) * reg;
            uint32_t iters = uint32_t(std::min<uint64_t>(2000, std::max<uint64_t>(3, (100ull << 20) / std::max<uint64_t>(1ull, moved))));
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(xcb, &beg));
            vkCmdCopyBuffer(xcb, src, dst, nreg, rc.data());
            VK_CHECK(vkEndCommandBuffer(xcb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &xcb;
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                for (uint32_t it = 0; it < iters; ++it) {
                    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr));
                    VK_CHECK(vkQueueWaitIdle(c.queue));
                }
                auto t1 = std::chrono::steady_clock::now();
                double ms = std::chrono::duration<double, std::milli>(t1 - t0).count() / double(iters);
                best = std::min(best, ms);
            }
            ms_iter = best;
            return double(moved) / (best * 1e-3) / 1e9;
        };

        if (have_dev && hs) {
            const uint64_t sizes[] = {4096, 65536, 262144, 1u << 20, 16u << 20, 64u << 20, 128u << 20};
            printf("  bulk single-copy (host %s <-> device-local, one submit + waitIdle each):\n", hs_name);
            printf("     %10s %10s %10s   %10s %10s\n", "size", "H2D GB/s", "H2D us", "D2H GB/s", "D2H us");
            for (uint64_t s : sizes) {
                double ms_h2d = 0, ms_d2h = 0;
                double g_h2d = bench(hs->b, dwe.b, 1, s, ms_h2d);
                double g_d2h = bench(dwl.b, hs->b, 1, s, ms_d2h);
                printf("     %9.1fKB %10.2f %10.1f   %10.2f %10.1f\n", double(s) / 1024.0,
                       g_h2d, ms_h2d * 1000.0, g_d2h, ms_d2h * 1000.0);
            }
            if (have_wc && have_wb) {
                double ms1 = 0, ms2 = 0;
                double g1 = bench(hwc.b, dwe.b, 1, 16u << 20, ms1);
                double g2 = bench(dwl.b, hwc.b, 1, 16u << 20, ms2);
                printf("  write-combined host memory @16 MiB: H2D %.2f GB/s (%.1f us), D2H %.2f GB/s (%.1f us)\n",
                       g1, ms1 * 1000.0, g2, ms2 * 1000.0);
            }
            printf("  scattered copies (what KV/state paging actually looks like, host %s):\n", hs_name);
            printf("     %10s %7s %10s %10s %12s\n", "region", "n", "GB/s", "us/submit", "us/region");
            const uint64_t regs[] = {4096, 8192, 36864, 65536, 512 * 1024};
            for (uint64_t r : regs) {
                double ms = 0;
                double g = bench(hs->b, dwe.b, 100, r, ms);
                printf("     %9.1fKB %7u %10.2f %10.1f %12.2f\n", double(r) / 1024.0, 100u, g,
                       ms * 1000.0, ms * 1000.0 / 100.0);
            }
        }
        {   // the fixed cost of touching the queue at all, with nothing in the buffer
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(xcb, &beg));
            vkCmdWriteTimestamp(xcb, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 60);
            VK_CHECK(vkEndCommandBuffer(xcb));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &xcb;
            double best = 1e30;
            for (int rep = 0; rep < 3; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                for (int it = 0; it < 2000; ++it) {
                    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr));
                    VK_CHECK(vkQueueWaitIdle(c.queue));
                }
                auto t1 = std::chrono::steady_clock::now();
                best = std::min(best, std::chrono::duration<double, std::micro>(t1 - t0).count() / 2000.0);
            }
            printf("  empty submit + waitIdle: %.1f us  (this is the floor under every paging op)\n", best);
        }
        for (auto *mm : {&dwe, &dwl, &hwc, &hwb}) {
            if (mm->b) vkDestroyBuffer(c.dev, mm->b, nullptr);
            if (mm->m) vkFreeMemory(c.dev, mm->m, nullptr);
        }
        vkFreeCommandBuffers(c.dev, c.cpool, 1, &xcb);
    }

    // ---- BONSAI_SYNC=1: who is waiting on whom -------------------------------------------
    // The benchmark above already answers the first half of question B by construction: every
    // token in the timing loop is a *recorded* dispatch inside ONE command buffer submitted once,
    // so the CPU is not in the per-token path at all there. A server cannot do that: between
    // tokens it must read the logits back and pick one. This measures the three shapes and prices
    // each of them, per token, with GPU timestamps next to the wall clock so the gap is visible.
    if (getenv("BONSAI_SYNC")) {
        printf("\n== CPU/GPU overlap (BONSAI_SYNC) ==\n");
        const uint32_t VOCAB = 248320;              // real, from tokenizer.ggml.tokens in the GGUF
        const uint32_t NQ = 24;                     // tokens per shape; 24*1.3 ms keeps the run short
        const double TOKEN_MS = 61.1;               // measured: calc_weighted_native.py, whole-token projection
        std::vector<VkCommandBuffer> cb(NQ);
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = c.cpool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = NQ;
        VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai, cb.data()));
        std::vector<VkFence> fe(NQ);
        VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
        for (auto &f : fe) VK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &f));
        auto wait_f = [&](uint32_t i) {
            VkFence f = fe[i];
            while (vkWaitForFences(c.dev, 1, &f, VK_TRUE, 1000000000ull) == VK_TIMEOUT) {}
            VK_CHECK(vkResetFences(c.dev, 1, &f));
        };
        auto sub = [&](VkCommandBuffer b, VkFence f) {
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &b;
            VK_CHECK(vkQueueSubmit(c.queue, 1, &si, f));
        };
        // GPU busy for one token, straight from the timestamps that shape recorded. Reading it
        // inside the same rep loop matters: every shape reuses query slots, so a read after the
        // *next* shape starts reports that shape's number, not this one's.
        auto gpu_busy = [&](uint32_t i) -> double {
            uint64_t t2[2];
            VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, 2 * i, 2, sizeof(t2), t2, 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            return double(t2[1] - t2[0]) * c.limits.timestampPeriod / 1000.0;   // microseconds
        };
        void *rb_ptr = nullptr;
        VK_CHECK(vkMapMemory(c.dev, c.host_mem, diag_base, M * 8u, 0, &rb_ptr));
        const float *rb[2] = {reinterpret_cast<const float *>(rb_ptr),
                              reinterpret_cast<const float *>(rb_ptr) + size_t(M)};

        // The host half of a decode step, as an actual greedy sampler does it: take the row the
        // GPU produced, then scan the whole 248320-entry vocabulary. Cost is measured, not assumed.
        std::vector<float> hlog(VOCAB);
        for (uint32_t i = 0; i < VOCAB; ++i) hlog[i] = float((int(rnd() % 2000) - 1000)) * 0.001f;
        auto host_step = [&](const float *lg) -> uint32_t {
            memcpy(hlog.data(), lg, size_t(M) * 4u);
            uint32_t bi = 0; float bv = hlog[0];
            for (uint32_t v = 1; v < VOCAB; ++v) if (hlog[v] > bv) { bv = hlog[v]; bi = v; }
            return bi;
        };
        {
            uint32_t sink = 0;
            auto t0 = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < NQ; ++i) sink = host_step(rb[i & 1]);
            auto t1 = std::chrono::steady_clock::now();
            printf("  host step (copy %u logits out of mapped memory + greedy argmax over %u): %.1f us/token"
                   "  (picked token %u)\n",
                   M, VOCAB, std::chrono::duration<double, std::micro>(t1 - t0).count() / double(NQ), sink);
        }

        // Keep the fastest rep together with the timestamp reading *of that same rep*. Taking
        // min-of-3 for the wall clock and the last rep for the timestamps made the GPU-bound shape
        // report a negative bubble, because this card's clocks wander ~20% between reps.
        auto keep = [](double w_r, double g_r, double &w, double &g) { if (w_r < w) { w = w_r; g = g_r; } };

        struct Shape { const char *name; double wall_us, gpu_us, host_us; };
        std::vector<Shape> shapes;

        // (1) batched: the shape the existing benchmark uses - NQ tokens in one buffer, one submit
        double wall = 1e30, gpu = 0.0;
        // NOTE (measured, not theoretical): the host-side entry point vkResetQueryPool() jumps to
        // address 0 and kills the process on this driver (AMD driver 2.0.279 + the vulkan-1.dll the
        // MinGW import lib was generated from), even though the instance asks for VK_API 1.1. gdb
        // shows PC=0x0 inside the loader thunk. vkCmdResetQueryPool() - resetting from the command
        // buffer - works fine, so every reset below is recorded, not called. Any future device-level
        // entry point (vkResetQueryPool, vkCmdWriteTimestamp2, ...) needs vkGetDeviceProcAddr.
        for (int rep = 0; rep < 3; ++rep) {
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(c.cmd, &beg));
            vkCmdResetQueryPool(c.cmd, c.qp, 60, 2);
            vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 60);
            for (uint32_t i = 0; i < NQ; ++i) dispatch_token(c.cmd);
            vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 61);
            VK_CHECK(vkEndCommandBuffer(c.cmd));
            auto t0 = std::chrono::steady_clock::now();
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &c.cmd;
            VK_CHECK(vkQueueSubmit(c.queue, 1, &si, nullptr));
            VK_CHECK(vkQueueWaitIdle(c.queue));
            auto t1 = std::chrono::steady_clock::now();
            uint64_t t2[2];
            VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, 60, 2, sizeof(t2), t2, 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            keep(std::chrono::duration<double, std::micro>(t1 - t0).count() / double(NQ),
                 double(t2[1] - t2[0]) * c.limits.timestampPeriod / 1000.0 / double(NQ), wall, gpu);
        }
        shapes.push_back({"batched NQ tokens / 1 submit (no host work)", wall, gpu, 0.0});

        // (2) serialized, GPU only: one submit + fence per token, no readback, no host work.
        wall = 1e30; gpu = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
            for (uint32_t i = 0; i < NQ; ++i) record_one(cb[i], 2 * i, 2 * i + 1, -1);
            auto t0 = std::chrono::steady_clock::now();
            for (uint32_t i = 0; i < NQ; ++i) { sub(cb[i], fe[i]); wait_f(i); }
            auto t1 = std::chrono::steady_clock::now();
            keep(std::chrono::duration<double, std::micro>(t1 - t0).count() / double(NQ), gpu_busy(5),
                 wall, gpu);
        }
        shapes.push_back({"serialized: 1 submit + fence /token", wall, gpu, 0.0});

        // (3) serialized WITH readback + host sample: what a straightforward server actually does.
        wall = 1e30; gpu = 0.0;
        double host_us = 1e30;
        for (int rep = 0; rep < 3; ++rep) {
            for (uint32_t i = 0; i < NQ; ++i) record_one(cb[i], 2 * i, 2 * i + 1, int(i & 1));
            auto t0 = std::chrono::steady_clock::now();
            uint32_t sink = 0;
            for (uint32_t i = 0; i < NQ; ++i) {
                sub(cb[i], fe[i]); wait_f(i);
                sink = host_step(rb[i & 1]);
            }
            auto t1 = std::chrono::steady_clock::now();
            (void)sink;
            keep(std::chrono::duration<double, std::micro>(t1 - t0).count() / double(NQ), gpu_busy(5),
                 wall, gpu);
        }
        for (int rep = 0; rep < 3; ++rep) {   // the host-only part of shape (3), timed on its own
            auto t0 = std::chrono::steady_clock::now();
            uint32_t sink = 0;
            for (uint32_t i = 0; i < NQ; ++i) sink = host_step(rb[i & 1]);
            auto t1 = std::chrono::steady_clock::now();
            (void)sink;
            host_us = std::min(host_us, std::chrono::duration<double, std::micro>(t1 - t0).count() / double(NQ));
        }
        shapes.push_back({"serialized + readback + sample (status quo)", wall, gpu, host_us});

        // (4) depth-2 pipeline: submit token i, then sample token i-1 while i runs on the GPU.
        // This is the whole of DeepInfer's Zero-Overhead idea, in one place, on our card.
        wall = 1e30; gpu = 0.0;
        for (int rep = 0; rep < 3; ++rep) {
            for (uint32_t i = 0; i < NQ; ++i) record_one(cb[i], 2 * i, 2 * i + 1, int(i & 1));
            auto t0 = std::chrono::steady_clock::now();
            uint32_t sink = 0;
            sub(cb[0], fe[0]);
            for (uint32_t i = 1; i < NQ; ++i) {
                sub(cb[i], fe[i]);                 // GPU starts i ...
                wait_f(i - 1);                     // ... only then wait for i-1
                sink = host_step(rb[(i - 1) & 1]); // ... and sample i-1 while i is still running
            }
            wait_f(NQ - 1);
            sink = host_step(rb[(NQ - 1) & 1]);
            auto t1 = std::chrono::steady_clock::now();
            (void)sink;
            keep(std::chrono::duration<double, std::micro>(t1 - t0).count() / double(NQ), gpu_busy(5),
                 wall, gpu);
        }
        shapes.push_back({"depth-2 pipeline (sample N-1 under GPU N)", wall, gpu, host_us});
        printf("  %-46s %11s %11s %11s %11s\n", "shape", "wall us/tok", "gpu us/tok", "host us/tok",
               "bubble us");
        for (auto &s : shapes)
            printf("  %-46s %11.1f %11.1f %11.1f %11.1f\n", s.name, s.wall_us, s.gpu_us, s.host_us,
                   s.wall_us - s.gpu_us);
        fflush(stdout);
        double base = shapes[2].wall_us, pipe = shapes[3].wall_us;
        double submit_cost = shapes[1].wall_us - shapes[1].gpu_us;   // pure submit+fence round trip
        // 201 GEMV-shaped ops per token for THIS model, from the tensor table: 24 GDN blocks x
        // (attn_qkv, attn_gate, ssm_out, ffn_gate, ffn_up, ffn_down) = 144, 8 full-attention blocks
        // x (attn_q, attn_k, attn_v, attn_output, ffn_gate, ffn_up, ffn_down) = 56, + output.weight
        // = 201. A runtime that submitted each one separately pays submit_cost for every one of them.
        const double NOPS = 201.0;
        printf("  recoverable by overlapping the host step: %.1f us/token = %.3f%% of the %.1f ms/token "
               "projection cost (measured, BONSAI_SYNC runs one GEMV, not a whole token)\n",
               base - pipe, 100.0 * (base - pipe) / (TOKEN_MS * 1000.0), TOKEN_MS);
        printf("  one submit+fence round trip: %.1f us -> submitting every op separately (%.0f ops/token) "
               "would pay %.1f ms/token = %.1f%% of %.1f ms; one submit per block (%.0f) pays %.1f ms = %.1f%%; "
               "one submit per token costs %.1f us = %.3f%%\n",
               submit_cost, NOPS, submit_cost * NOPS / 1000.0, 100.0 * submit_cost * NOPS / (TOKEN_MS * 1000.0),
               TOKEN_MS, 32.0, submit_cost * 32.0 / 1000.0, 100.0 * submit_cost * 32.0 / (TOKEN_MS * 1000.0),
               shapes[0].wall_us - shapes[0].gpu_us,
               100.0 * (shapes[0].wall_us - shapes[0].gpu_us) / (TOKEN_MS * 1000.0));
        vkUnmapMemory(c.dev, c.host_mem);
        for (auto &f : fe) vkDestroyFence(c.dev, f, nullptr);
        vkFreeCommandBuffers(c.dev, c.cpool, NQ, cb.data());
    }

    // ---- BONSAI_OVERLAP=1: can a copy run UNDER compute? ------------------------------------
    // Every tiering design in the DeepInfer evaluation - KV offload, expert prefetch, bounded state
    // replay - is only cheap if bytes move while the shader is busy. Whether that is true on a
    // Polaris card under WDDM is a property of the driver and the rings, not of the spec, so
    // measure it: the same token batch and the same 16 MiB copy, alone and together.
    if (getenv("BONSAI_OVERLAP")) {
        printf("\n== compute/copy concurrency (BONSAI_OVERLAP) ==\n");
        if (c.tfi == 0xFFFFFFFFu) {
            printf("  no transfer-only queue family - a copy would queue behind the GEMV, so no\n"
                   "  design that depends on overlapping PCIe traffic with compute can work here.\n");
        } else {
            const uint64_t SZ = 64ull << 20;      // big enough for the largest row of the sweep below
            const uint32_t NT = 10;               // 10 token-ops ~ 3.3 ms, matched to a 16-24 MiB copy
            uint32_t fams[2] = {c.qfi, c.tfi};
            struct CBuf { VkBuffer b = VK_NULL_HANDLE; VkDeviceMemory m = VK_NULL_HANDLE; };
            auto mkc = [&](CBuf &o, uint64_t size, uint32_t need) -> bool {
                VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
                bi.size = size;
                bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
                           VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
                bi.sharingMode = VK_SHARING_MODE_CONCURRENT;      // touched by two queues
                bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fams;
                if (vkCreateBuffer(c.dev, &bi, nullptr, &o.b) != VK_SUCCESS) return false;
                VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, o.b, &mr);
                for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i) {
                    if (!((mr.memoryTypeBits >> i) & 1u)) continue;
                    if ((c.mp.memoryTypes[i].propertyFlags & need) != need) continue;
                    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
                    ma.allocationSize = mr.size; ma.memoryTypeIndex = i;
                    if (vkAllocateMemory(c.dev, &ma, nullptr, &o.m) != VK_SUCCESS) continue;
                    if (vkBindBufferMemory(c.dev, o.b, o.m, 0) == VK_SUCCESS) return true;
                    vkFreeMemory(c.dev, o.m, nullptr); o.m = VK_NULL_HANDLE;
                }
                vkDestroyBuffer(c.dev, o.b, nullptr); o.b = VK_NULL_HANDLE;
                return false;
            };
            CBuf devb, hob;
            bool got = mkc(devb, SZ, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) &&
                       mkc(hob, SZ, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                    VK_MEMORY_PROPERTY_HOST_COHERENT_BIT |
                                    VK_MEMORY_PROPERTY_HOST_CACHED_BIT);
            if (!got) { printf("  concurrent buffers unavailable\n"); }
            else {
                VkCommandPool tcp = VK_NULL_HANDLE;
                VkCommandPoolCreateInfo cpc{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
                cpc.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpc.queueFamilyIndex = c.tfi;
                VK_CHECK(vkCreateCommandPool(c.dev, &cpc, nullptr, &tcp));
                VkCommandBuffer tcb = VK_NULL_HANDLE;
                VkCommandBufferAllocateInfo ca{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
                ca.commandPool = tcp; ca.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; ca.commandBufferCount = 1;
                VK_CHECK(vkAllocateCommandBuffers(c.dev, &ca, &tcb));
                VkFence f0 = VK_NULL_HANDLE, f1 = VK_NULL_HANDLE;
                VkFenceCreateInfo fc{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
                VK_CHECK(vkCreateFence(c.dev, &fc, nullptr, &f0));
                VK_CHECK(vkCreateFence(c.dev, &fc, nullptr, &f1));
                auto us = [](std::chrono::steady_clock::time_point a,
                             std::chrono::steady_clock::time_point b) {
                    return std::chrono::duration<double, std::micro>(b - a).count();
                };
                auto rec_tokens = [&]() {
                    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    VK_CHECK(vkBeginCommandBuffer(c.cmd, &beg));
                    vkCmdResetQueryPool(c.cmd, c.qp, 122, 2);
                    vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 122);
                    for (uint32_t i = 0; i < NT; ++i) dispatch_token(c.cmd);
                    vkCmdWriteTimestamp(c.cmd, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 123);
                    VK_CHECK(vkEndCommandBuffer(c.cmd));
                };
                auto rec_copy = [&](uint64_t sz, bool d2h) {
                    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
                    VK_CHECK(vkBeginCommandBuffer(tcb, &beg));
                    vkCmdResetQueryPool(tcb, c.qp, 120, 2);
                    vkCmdWriteTimestamp(tcb, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, 120);
                    VkBufferCopy cp{0, 0, sz};
                    // H2D is the weight/expert paging direction; D2H is the KV/state eviction
                    // direction. They are different engines' workloads and different measured rates,
                    // so both have to be on the table before a tiering design can be costed.
                    vkCmdCopyBuffer(tcb, d2h ? devb.b : hob.b, d2h ? hob.b : devb.b, 1, &cp);
                    vkCmdWriteTimestamp(tcb, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, 121);
                    VK_CHECK(vkEndCommandBuffer(tcb));
                };
                auto wait_f = [&](VkFence f) {
                    while (vkWaitForFences(c.dev, 1, &f, VK_TRUE, 1000000000ull) == VK_TIMEOUT) {}
                    vkResetFences(c.dev, 1, &f);
                };
                // duration on the engine, not time-until-the-CPU-noticed: the fences only say that
                // a job finished somewhere in the stream, which made a 32 KiB copy look like it took
                // the whole 2.9 ms compute window. These are the engine's own numbers.
                auto ticks = [&](uint32_t a, uint32_t b) -> double {
                    uint64_t t2[2];
                    VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, a, 2, sizeof(t2), t2, 8,
                                                   VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                    return double(t2[1] - t2[0]) * c.limits.timestampPeriod / 1000.0;   // us
                };
                // the compute baseline first: 10 token-ops on the compute queue, nothing else running
                double t_tok = 1e30, ts_tok = 1e30;
                for (int rep = 0; rep < 5; ++rep) {
                    rec_tokens();
                    VkSubmitInfo A{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    A.commandBufferCount = 1; A.pCommandBuffers = &c.cmd;
                    auto s0 = std::chrono::steady_clock::now();
                    VK_CHECK(vkQueueSubmit(c.queue, 1, &A, f0)); wait_f(f0);
                    auto s1 = std::chrono::steady_clock::now();
                    t_tok = std::min(t_tok, us(s0, s1));
                    ts_tok = std::min(ts_tok, ticks(122, 123));
                }
                // sweep the page sizes the designs actually move: one KV token group (32 KiB), one
                // expert (1 MiB), a block of experts / a state checkpoint (16 and 64 MiB)
                printf("  compute queue family %u, transfer queue family %u, compute = %u token-ops "
                       "(wall %.0f us, engine %.0f us)\n", c.qfi, c.tfi, NT, t_tok, ts_tok);
                printf("     %10s %11s %7s %11s %7s %11s %9s %11s\n", "copy size", "alone eng",
                       "GB/s", "conc eng", "GB/s", "tok conc", "tok slow", "both wall");
                double conc16 = 0.0;
                for (int dir = 0; dir < 2; ++dir) {          // 0 = H2D, 1 = D2H
                const bool d2h = dir == 1;
                printf("     %s\n", d2h ? "  --- device -> host (evict) ---"
                                        : "  --- host -> device (load) ---");
                for (uint64_t sz : {uint64_t(32768), uint64_t(1) << 20, uint64_t(16) << 20,
                                    uint64_t(64) << 20}) {
                    double a_alone = 1e30, cp_alone = 1e30, cp_conc = 1e30, tok_conc = 1e30, both = 1e30;
                    for (int rep = 0; rep < 5; ++rep) {
                        rec_tokens();
                        VkSubmitInfo A{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        A.commandBufferCount = 1; A.pCommandBuffers = &c.cmd;
                        VkSubmitInfo B{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        B.commandBufferCount = 1; B.pCommandBuffers = &tcb;
                        // the copy by itself: wall says what the CPU paid, the engine timestamp says
                        // what the SDMA ring was busy for. Both matter - the first is the cost of a
                        // sync point, the second is the bandwidth the tiering design actually gets.
                        rec_copy(sz, d2h);
                        auto s0 = std::chrono::steady_clock::now();
                        VK_CHECK(vkQueueSubmit(c.qtx, 1, &B, f1)); wait_f(f1);
                        auto s1 = std::chrono::steady_clock::now();
                        a_alone = std::min(a_alone, us(s0, s1));
                        cp_alone = std::min(cp_alone, ticks(120, 121));
                        // the same copy against the same compute, on two queues, no barrier between.
                        // both command buffers are ONE_TIME_SUBMIT and were already consumed, so
                        // re-record rather than re-submit.
                        rec_tokens(); rec_copy(sz, d2h);
                        auto e0 = std::chrono::steady_clock::now();
                        VK_CHECK(vkQueueSubmit(c.queue, 1, &A, f0));
                        VK_CHECK(vkQueueSubmit(c.qtx, 1, &B, f1));
                        wait_f(f0); wait_f(f1);
                        auto e1 = std::chrono::steady_clock::now();
                        cp_conc = std::min(cp_conc, ticks(120, 121));
                        tok_conc = std::min(tok_conc, ticks(122, 123));
                        both = std::min(both, us(e0, e1));
                    }
                    if (sz == (uint64_t(16) << 20) && !d2h) conc16 = cp_conc;
                    printf("     %9.1fKB %11.1f %7.2f   %11.1f %7.2f %11.1f %+8.1f%% %11.1f  (alone wall %.0f)\n",
                           double(sz) / 1024.0, cp_alone, double(sz) / (cp_alone * 1e-6) / 1e9,
                           cp_conc, double(sz) / (cp_conc * 1e-6) / 1e9, tok_conc,
                           100.0 * (tok_conc - ts_tok) / ts_tok, both, a_alone);
                }
                }
                const double us_per_mib = conc16 / 16.0;
                printf("  -> copy engine while compute runs (16 MiB row): %.1f us per MiB (%.2f GB/s). "
                       "A 1 MiB expert page is %.0f us = %.1f%% of the 61.1 ms/token projection cost "
                       "measured in STAGE3, and it is only visible at all if the transfer queue is "
                       "already busy.\n",
                       us_per_mib, double(16u << 20) / (conc16 * 1e-6) / 1e9, us_per_mib,
                       100.0 * us_per_mib / 61100.0);
                vkDestroyFence(c.dev, f0, nullptr); vkDestroyFence(c.dev, f1, nullptr);
                vkDestroyCommandPool(c.dev, tcp, nullptr);
                vkDestroyBuffer(c.dev, devb.b, nullptr); vkFreeMemory(c.dev, devb.m, nullptr);
                vkDestroyBuffer(c.dev, hob.b, nullptr);  vkFreeMemory(c.dev, hob.m, nullptr);
            }
        }
    }

    // ---- BONSAI_PIPE=1: the submission layer, four ways, measured --------------------------
    // DEEPINFER_EVAL §3 quoted "逐 op 提交 = 201 × 99.4 µs = 20.0 ms/token = 37.1% 纯损失" as a
    // PROJECTION made of two measured constants (the 99.4 µs round trip and the 201-op count read
    // off the GGUF tensor table). Projections are not measurements, so this block runs the same
    // 201 ops with the same GPU work through four different submission structures and reports the
    // wall clock of each. Shapes:
    //   A  per-op submit + per-op fence            <- the anti-pattern, this is what 37% claims about
    //   B  one command buffer per token, D2H on the compute queue, sync  <- batched, naive copy path
    //   C  B, but the D2H runs on the transfer-only queue (family 2) under a semaphore  <- opt 1
    //   D  C + depth-2: token N's logits are sampled by the CPU while token N+1 is on the GPU
    // Every number printed here is a clock reading or an engine timestamp; none of them multiply.
    if (getenv("BONSAI_PIPE")) {
        printf("\n== submission pipeline (BONSAI_PIPE) ==\n");
        const uint32_t OPS = 201;                 // per-token GEMV count, from the GGUF tensor table
        const uint32_t VOCAB = 248320;            // output.weight rows = the real per-token logits
        const uint64_t LOG_B = uint64_t(VOCAB) * 4u;
        const uint32_t NTOK = 6;                  // tokens per rep; 6 x ~60 ms keeps a rep ~0.4 s
        auto us_of = [](std::chrono::steady_clock::time_point a,
                        std::chrono::steady_clock::time_point b) {
            return std::chrono::duration<double, std::micro>(b - a).count(); };
        auto wait_f = [](VkDevice d, VkFence f) {
            while (vkWaitForFences(d, 1, &f, VK_TRUE, 1000000000ull) == VK_TIMEOUT) {}
            VK_CHECK(vkResetFences(d, 1, &f));
        };
        // the host half of a decode step, at the real vocabulary: greedy argmax over 248320 floats.
        auto argmax = [](const float *p, uint32_t n) -> uint32_t {
            uint32_t b = 0; float v = p[0];
            for (uint32_t i = 1; i < n; ++i) if (p[i] > v) { v = p[i]; b = i; }
            return b;
        };

        // ---- pipeline-owned buffers -------------------------------------------------------
        // logd stands in for the logits row a real token produces, so the D2H carries the REAL
        // 993,280 bytes per token. Its content is filled once over PCIe from host memory and never
        // updated: this block measures submission, synchronisation and link cost, not numerics, and
        // the semaphore chain below still models the compute -> copy dependency the real engine has.
        VkDeviceMemory p_dev = VK_NULL_HANDLE, p_host = VK_NULL_HANDLE;
        VkBuffer logd = VK_NULL_HANDLE, logh = VK_NULL_HANDLE;
        void *logh_ptr = nullptr;
        {
            VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
            bi.size = LOG_B; bi.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            if (has_tx) { bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
                          bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fam2; }
            VK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &logd));
            VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c.dev, logd, &mr);
            VkMemoryAllocateInfo mai{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
            mai.allocationSize = mr.size; mai.memoryTypeIndex = c.arena_type;
            VK_CHECK(vkAllocateMemory(c.dev, &mai, nullptr, &p_dev));
            VK_CHECK(vkBindBufferMemory(c.dev, logd, p_dev, 0));

            bi.size = LOG_B * 2; bi.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT;
            VK_CHECK(vkCreateBuffer(c.dev, &bi, nullptr, &logh));
            vkGetBufferMemoryRequirements(c.dev, logh, &mr);
            mai.allocationSize = mr.size; mai.memoryTypeIndex = c.host_type;
            VK_CHECK(vkAllocateMemory(c.dev, &mai, nullptr, &p_host));
            VK_CHECK(vkBindBufferMemory(c.dev, logh, p_host, 0));
            VK_CHECK(vkMapMemory(c.dev, p_host, 0, LOG_B * 2, 0, &logh_ptr));
            // any old numbers, but a real distribution so argmax is not a degenerate first-element scan
            float *lf = reinterpret_cast<float *>(logh_ptr);
            for (uint64_t i = 0; i < LOG_B / 4u; ++i) lf[i] = float(int(rnd() % 20000) - 10000) * 1e-4f;
        }
        // command pool for the transfer family (the compute pool c.cpool cannot make its CBs)
        VkCommandPool tpool = VK_NULL_HANDLE;
        if (has_tx) {
            VkCommandPoolCreateInfo pi2{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
            pi2.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pi2.queueFamilyIndex = c.tfi;
            VK_CHECK(vkCreateCommandPool(c.dev, &pi2, nullptr, &tpool));
        }
        VkCommandBufferAllocateInfo cai2{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai2.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        std::vector<VkCommandBuffer> opcb(OPS);
        cai2.commandPool = c.cpool; cai2.commandBufferCount = OPS;
        VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, opcb.data()));
        VkCommandBuffer cbc[2] = {}, xcb[2] = {}, bbc = VK_NULL_HANDLE;
        cai2.commandBufferCount = 2;
        VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, cbc));
        if (has_tx) { cai2.commandPool = tpool; VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, xcb)); }
        VkFence ff[2] = {}, fc[2] = {}, fd[2] = {};
        VkSemaphore ss[2] = {}, sd[2] = {};
        {
            VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
            VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
            for (int i = 0; i < 2; ++i) {
                VK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &ff[i]));
                VK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &fc[i]));
                VK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &fd[i]));
                VK_CHECK(vkCreateSemaphore(c.dev, &sci, nullptr, &ss[i]));
                VK_CHECK(vkCreateSemaphore(c.dev, &sci, nullptr, &sd[i]));
            }
        }
        VkFence fone = VK_NULL_HANDLE;
        { VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO}; VK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &fone)); }

        // one op, one timestamp pair -> per-op GPU time is read directly, not multiplied by 201
        auto rec_op = [&](VkCommandBuffer b, uint32_t q0) {
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};   // flags 0: re-submittable
            VK_CHECK(vkBeginCommandBuffer(b, &beg));
            vkCmdResetQueryPool(b, c.qp, q0, 2);
            vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, q0);
            dispatch_token(b);
            vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, q0 + 1);
            VK_CHECK(vkEndCommandBuffer(b));
        };
        // a whole token = OPS ops in ONE command buffer. wait_sd is submitted as a wait, not
        // recorded: it is the write-after-read guard that stops token N+1 from clobbering the
        // logits row while token N's copy is still reading it.
        auto rec_token = [&](VkCommandBuffer b, uint32_t q0) {
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHECK(vkBeginCommandBuffer(b, &beg));
            vkCmdResetQueryPool(b, c.qp, q0, 2);
            vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, q0);
            for (uint32_t i = 0; i < OPS; ++i) dispatch_token(b);
            vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, q0 + 1);
            VK_CHECK(vkEndCommandBuffer(b));
        };
        // the copy is also recorded inline in shape B (see rec_token_xfer_inline's replacement
        // below), so it needs no separate recorder here.
        auto rec_copy = [&](VkCommandBuffer b, uint32_t q0, uint32_t slot) {
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHECK(vkBeginCommandBuffer(b, &beg));
            vkCmdResetQueryPool(b, c.qp, q0, 2);
            vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, q0);
            VkBufferCopy cp{0, uint64_t(slot) * LOG_B, LOG_B};
            vkCmdCopyBuffer(b, logd, logh, 1, &cp);
            vkCmdWriteTimestamp(b, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, q0 + 1);
            VK_CHECK(vkEndCommandBuffer(b));
        };
        auto gpu_us = [&](uint32_t q0) -> double {
            uint64_t t2[2];
            VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, q0, 2, sizeof(t2), t2, 8,
                                           VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
            return double(t2[1] - t2[0]) * c.limits.timestampPeriod / 1000.0;
        };

        // fill logd over the link once, on the queue we are about to benchmark with
        {
            VkBufferCopy cp{0, 0, LOG_B};
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VkCommandBuffer u = has_tx ? xcb[0] : c.cmd;
            VK_CHECK(vkBeginCommandBuffer(u, &beg));
            vkCmdCopyBuffer(u, logh, logd, 1, &cp);
            VK_CHECK(vkEndCommandBuffer(u));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &u;
            VK_CHECK(vkQueueSubmit(has_tx ? c.qtx : c.queue, 1, &si, fone));
            auto e0 = std::chrono::steady_clock::now();
            wait_f(c.dev, fone);
            auto e1 = std::chrono::steady_clock::now();
            printf("  H2D %llu B (logits stand-in) on %s: wall %.1f us\n",
                   (unsigned long long)LOG_B, has_tx ? "transfer-only queue" : "compute queue",
                   us_of(e0, e1));
        }
        // The host half of a decode step, done the way it has to be done rather than the way that
        // looks shortest. The logits window is HOST_COHERENT, which on this driver is uncached /
        // write-combining: scanning 993,280 bytes in place through the mapping measured ~25 ms per
        // token (printed below as the trap), while copying the row into normal memory and scanning
        // there is what an engine can actually afford. Every shape from here on runs the same
        // host_step, so wall/token stays comparable across shapes.
        std::vector<float> hlog(VOCAB);
        double memcpy_tot = 0, scan_tot = 0;
        double nsteps = 0;
        uint32_t picked = 0;
        auto host_step = [&](const float *mapped) -> double {
            auto m0 = std::chrono::steady_clock::now();
            memcpy(hlog.data(), mapped, size_t(LOG_B));
            auto m1 = std::chrono::steady_clock::now();
            picked = argmax(hlog.data(), VOCAB);
            auto m2 = std::chrono::steady_clock::now();
            memcpy_tot += us_of(m0, m1); scan_tot += us_of(m1, m2); nsteps += 1.0;
            return us_of(m0, m2);
        };
        double uncached_us = 0.0;
        {
            auto u0 = std::chrono::steady_clock::now();
            volatile uint32_t v = argmax(reinterpret_cast<const float *>(logh_ptr), VOCAB);
            uncached_us = us_of(u0, std::chrono::steady_clock::now()); (void)v;
        }

        struct Pipe { const char *name; double wall, gpu, host, copy; };
        std::vector<Pipe> res;

        // ---- A: per-op submit + per-op fence ------------------------------------------------
        // Same token shape as B (201 ops, then one logits D2H on the compute queue, then the same
        // host_step) so the ONLY difference between A and B is how many times the work is submitted.
        {
            for (uint32_t i = 0; i < OPS; ++i) rec_op(opcb[i], 2 * i);   // slots 0..401
            // the tail: one op-sized CB holding just the logits copy, re-submitted once per token
            VkCommandBuffer acb = VK_NULL_HANDLE;
            cai2.commandPool = c.cpool; cai2.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, &acb));
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHECK(vkBeginCommandBuffer(acb, &beg));
            VkBufferCopy cp{0, 0, LOG_B};
            vkCmdCopyBuffer(acb, logd, logh, 1, &cp);
            VK_CHECK(vkEndCommandBuffer(acb));
            double best_w = 1e30, best_g = 0;
            double hm0 = memcpy_tot + scan_tot;   // host cost is reported as the mean over the 3 reps
            for (int rep = 0; rep < 3; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                for (uint32_t t = 0; t < NTOK; ++t) {
                    for (uint32_t i = 0; i < OPS; ++i) {
                        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        si.commandBufferCount = 1; si.pCommandBuffers = &opcb[i];
                        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, ff[0]));
                        wait_f(c.dev, ff[0]);
                    }
                    VkSubmitInfo s2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    s2.commandBufferCount = 1; s2.pCommandBuffers = &acb;
                    VK_CHECK(vkQueueSubmit(c.queue, 1, &s2, ff[0]));
                    wait_f(c.dev, ff[0]);
                    host_step(reinterpret_cast<const float *>(logh_ptr));
                }
                auto t1 = std::chrono::steady_clock::now();
                // one batched read of all 402 timestamps
                std::vector<uint64_t> tq(size_t(2) * OPS);
                VK_CHECK(vkGetQueryPoolResults(c.dev, c.qp, 0, 2 * OPS, tq.size() * 8, tq.data(), 8,
                                               VK_QUERY_RESULT_64_BIT | VK_QUERY_RESULT_WAIT_BIT));
                double gsum = 0;
                for (uint32_t i = 0; i < OPS; ++i)
                    gsum += double(tq[2 * i + 1] - tq[2 * i]) * c.limits.timestampPeriod / 1000.0;
                double w = us_of(t0, t1) / double(NTOK);
                if (w < best_w) { best_w = w; best_g = gsum / double(NTOK); }
            }
            res.push_back({"A per-op submit + per-op fence (201x)", best_w, best_g,
                           (memcpy_tot + scan_tot - hm0) / double(NTOK * 3), 0.0});
            // A's per-op engine timestamps are NOT trustworthy, and the reason is measurable rather
            // than hand-waved: 201 fenced ops report 48 us each while the identical op inside one
            // command buffer (shape B) reports 289 us, and B's number agrees with the standalone
            // benchmark to within 1%. The timestamp period therefore tracks a clock domain that
            // drops when the queue goes idle between submits. Only wall is used for A below.
            printf("  A per-op engine timestamps: %.1f us/op vs B's %.1f us/op -> engine timestamps are "
                   "valid only while the queue stays continuously busy\n",
                   best_g / double(OPS), res.back().gpu / double(OPS));
        }

        // ---- A2: per-op submit, ONE fence at the end -----------------------------------------
        // Splits A's loss into "submitting 201 times" and "waiting 201 times", which is the thing
        // the 37% projection never separated.
        {
            double best_w = 1e30;
            double hm0 = memcpy_tot + scan_tot;
            VkCommandBuffer acb2 = VK_NULL_HANDLE;
            cai2.commandPool = c.cpool; cai2.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, &acb2));
            {
                VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
                VK_CHECK(vkBeginCommandBuffer(acb2, &beg));
                VkBufferCopy cp{0, 0, LOG_B};
                vkCmdCopyBuffer(acb2, logd, logh, 1, &cp);
                VK_CHECK(vkEndCommandBuffer(acb2));
            }
            for (int rep = 0; rep < 3; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                for (uint32_t t = 0; t < NTOK; ++t) {
                    for (uint32_t i = 0; i < OPS; ++i) {
                        VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        si.commandBufferCount = 1; si.pCommandBuffers = &opcb[i];
                        // no fence per op: the queue is allowed to run ahead. The single fence of
                        // the token goes on its LAST piece of work (the logits copy below), so the
                        // CPU learns about the token exactly once.
                        VK_CHECK(vkQueueSubmit(c.queue, 1, &si, VkFence(VK_NULL_HANDLE)));
                    }
                    VkSubmitInfo s2{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    s2.commandBufferCount = 1; s2.pCommandBuffers = &acb2;
                    VK_CHECK(vkQueueSubmit(c.queue, 1, &s2, ff[0]));
                    wait_f(c.dev, ff[0]);
                    host_step(reinterpret_cast<const float *>(logh_ptr));
                }
                auto t1 = std::chrono::steady_clock::now();
                double w = us_of(t0, t1) / double(NTOK);
                if (w < best_w) best_w = w;
            }
            res.push_back({"A2 per-op submit, 1 fence/token", best_w, 0.0,
                           (memcpy_tot + scan_tot - hm0) / double(NTOK * 3), 0.0});
        }

        // ---- B: one command buffer per token, copy inline on the compute queue ---------------
        {
            cai2.commandPool = c.cpool; cai2.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, &bbc));
            // two timestamp pairs inside one CB: ops at 420/421, the copy at 422/423. Summing the
            // pair durations avoids asking a COMPUTE-stage timestamp to account for a transfer op.
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            VK_CHECK(vkBeginCommandBuffer(bbc, &beg));
            vkCmdResetQueryPool(bbc, c.qp, 420, 4);
            vkCmdWriteTimestamp(bbc, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 420);
            for (uint32_t i = 0; i < OPS; ++i) dispatch_token(bbc);
            vkCmdWriteTimestamp(bbc, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, c.qp, 421);
            bar_compute_to_transfer(bbc);
            vkCmdWriteTimestamp(bbc, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, 422);
            VkBufferCopy cp{0, 0, LOG_B};
            vkCmdCopyBuffer(bbc, logd, logh, 1, &cp);
            vkCmdWriteTimestamp(bbc, VK_PIPELINE_STAGE_TRANSFER_BIT, c.qp, 423);
            VK_CHECK(vkEndCommandBuffer(bbc));
            double best_w = 1e30, best_g = 0;
            double hm0 = memcpy_tot + scan_tot;
            for (int rep = 0; rep < 3; ++rep) {
                auto t0 = std::chrono::steady_clock::now();
                for (uint32_t t = 0; t < NTOK; ++t) {
                    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                    si.commandBufferCount = 1; si.pCommandBuffers = &bbc;
                    VK_CHECK(vkQueueSubmit(c.queue, 1, &si, ff[1]));
                    wait_f(c.dev, ff[1]);
                    host_step(reinterpret_cast<const float *>(logh_ptr));
                }
                auto t1 = std::chrono::steady_clock::now();
                double g = gpu_us(420) + gpu_us(422);
                double w = us_of(t0, t1) / double(NTOK);
                if (w < best_w) { best_w = w; best_g = g; }
            }
            res.push_back({"B 1 CB/token, D2H inline on compute queue", best_w, best_g,
                           (memcpy_tot + scan_tot - hm0) / double(NTOK * 3), gpu_us(422)});
        }

        // ---- C and D: the D2H moves to the transfer-only family -----------------------------
        if (has_tx) {
            rec_token(cbc[0], 410); rec_token(cbc[1], 412);
            rec_copy(xcb[0], 414, 0); rec_copy(xcb[1], 416, 1);
            const VkPipelineStageFlags stg_compute = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
            const VkPipelineStageFlags stg_transfer = VK_PIPELINE_STAGE_TRANSFER_BIT;

            // C: batched compute + SDMA copy, still one token at a time (host waits, then samples)
            {
                double best_w = 1e30, best_g = 0, best_c = 0;
                double hm0 = memcpy_tot + scan_tot;
                for (int rep = 0; rep < 3; ++rep) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (uint32_t t = 0; t < NTOK; ++t) {
                        VkSubmitInfo a{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        a.commandBufferCount = 1; a.pCommandBuffers = &cbc[0];
                        a.signalSemaphoreCount = 1; a.pSignalSemaphores = &ss[0];
                        VK_CHECK(vkQueueSubmit(c.queue, 1, &a, fc[0]));
                        VkSubmitInfo b{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        b.waitSemaphoreCount = 1; b.pWaitSemaphores = &ss[0]; b.pWaitDstStageMask = &stg_transfer;
                        b.commandBufferCount = 1; b.pCommandBuffers = &xcb[0];
                        b.signalSemaphoreCount = 1; b.pSignalSemaphores = &sd[0];
                        VK_CHECK(vkQueueSubmit(c.qtx, 1, &b, fd[0]));
                        wait_f(c.dev, fd[0]);
                        host_step(reinterpret_cast<const float *>(logh_ptr));
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double w = us_of(t0, t1) / double(NTOK);
                    if (w < best_w) { best_w = w; best_g = gpu_us(410); best_c = gpu_us(414); }
                }
                res.push_back({"C 1 CB/token + SDMA D2H (family 2)", best_w, best_g,
                           (memcpy_tot + scan_tot - hm0) / double(NTOK * 3), best_c});
            }

            // D: depth-2. Token N's logits are sampled by the CPU while token N+1 already runs on
            // the GPU, so the sampler and the copy engine are both under the compute time.
            {
                double best_w = 1e30, best_g = 0, best_c = 0;
                double hm0 = memcpy_tot + scan_tot;
                const float *slot[2] = {reinterpret_cast<const float *>(logh_ptr),
                                        reinterpret_cast<const float *>(logh_ptr) + LOG_B / 4u};
                for (int rep = 0; rep < 3; ++rep) {
                    auto t0 = std::chrono::steady_clock::now();
                    for (uint32_t t = 0; t < NTOK; ++t) {
                        uint32_t s = t & 1u, o = s ^ 1u;
                        VkSubmitInfo a{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        // from t>=2 the copy of token t-2 has to have finished reading the logits
                        // row before this token's GEMVs overwrite it. With a static stand-in buffer
                        // that read is harmless, but a real engine needs the edge, so it is here.
                        if (t >= 2) { a.waitSemaphoreCount = 1; a.pWaitSemaphores = &sd[o];
                                      a.pWaitDstStageMask = &stg_compute; }
                        a.commandBufferCount = 1; a.pCommandBuffers = &cbc[s];
                        a.signalSemaphoreCount = 1; a.pSignalSemaphores = &ss[s];
                        VK_CHECK(vkQueueSubmit(c.queue, 1, &a, fc[s]));
                        VkSubmitInfo b{VK_STRUCTURE_TYPE_SUBMIT_INFO};
                        b.waitSemaphoreCount = 1; b.pWaitSemaphores = &ss[s]; b.pWaitDstStageMask = &stg_transfer;
                        b.commandBufferCount = 1; b.pCommandBuffers = &xcb[s];
                        b.signalSemaphoreCount = 1; b.pSignalSemaphores = &sd[s];
                        VK_CHECK(vkQueueSubmit(c.qtx, 1, &b, fd[s]));
                        if (t > 0) {   // sample the PREVIOUS token: this is the overlap being bought
                            wait_f(c.dev, fd[o]);
                            host_step(slot[o]);
                        }
                    }
                    wait_f(c.dev, fd[(NTOK - 1) & 1u]);
                    {
                        host_step(slot[(NTOK - 1) & 1u]);
                    }
                    auto t1 = std::chrono::steady_clock::now();
                    double w = us_of(t0, t1) / double(NTOK);
                    if (w < best_w) {
                        best_w = w; best_g = (gpu_us(410) + gpu_us(412)) / 2.0;
                        best_c = (gpu_us(414) + gpu_us(416)) / 2.0;
                    }
                }
                res.push_back({"D depth-2 pipeline (sample N while GPU runs N+1)", best_w, best_g,
                           (memcpy_tot + scan_tot - hm0) / double(NTOK * 3), best_c});
            }
        } else {
            printf("  no transfer-only family on this device: shapes C/D skipped\n");
        }

        printf("  shape                                      wall/token     GPU busy    copy/sdma     host step    bubble\n");
        for (auto &p : res) {
            double bub = p.wall - (p.gpu > 0 ? p.gpu : p.wall);
            printf("  %-44s %10.1f %11s %11.1f %11.1f %10.1f us\n",
                   p.name, p.wall, p.gpu > 0 ? fmt1(p.gpu).c_str() : "untrust", p.copy, p.host, bub);
        }
        printf("  host step = memcpy 993,280 B out of the HOST_COHERENT mapping + greedy argmax over %u; "
               "scanning the mapping IN PLACE instead costs %.1f us/token (%.1f ms!): %.1f us memcpy, "
               "%.1f us scan. Read this as a design rule, not a footnote: never sample through the "
               "coherent pointer.\n",
               VOCAB, uncached_us, uncached_us / 1000.0,
               memcpy_tot / nsteps, scan_tot / nsteps);
        if (res.size() == 5) {
            const double a = res[0].wall, a2 = res[1].wall, b = res[2].wall;
            printf("  -> submission granularity, measured: A (201 submits + 201 fences) %.1f ms, "
                   "A2 (201 submits + 1 fence) %.1f ms, B (1 submit + 1 fence) %.1f ms per token.\n",
                   a / 1000.0, a2 / 1000.0, b / 1000.0);
            printf("     splitting the loss: waiting costs %.1f ms/token (%.1f%% of A), submitting "
                   "separately costs %.1f ms/token (%.1f%% of A). DEEPINFER_EVAL §3 projected the "
                   "total as 20.0 ms = 37.1%%.\n",
                   (a - a2) / 1000.0, 100.0 * (a - a2) / a, (a2 - b) / 1000.0, 100.0 * (a2 - b) / a);
            printf("  -> D2H of %llu B on the compute queue holds the GPU %.1f us/token; on the "
                   "transfer-only queue the same bytes take %.1f us = %.2f GB/s and the compute queue "
                   "is free for the next token\n",
                   (unsigned long long)LOG_B, res[2].copy, res[3].copy,
                   double(LOG_B) / (res[3].copy * 1e-6) / 1e9);
            printf("  -> depth-2 (D) is the only shape that hides the sampler: wall %.1f ms/token, "
                   "GPU busy %.1f ms, bubble %.1f ms = %.2f%% of the token\n",
                   res[4].wall / 1000.0, res[4].gpu / 1000.0,
                   (res[4].wall - res[4].gpu) / 1000.0,
                   100.0 * (res[4].wall - res[4].gpu) / res[4].wall);
            printf("  -> end-to-end decode rate implied by D alone: %.2f tokens/s at this op shape "
                   "(%.1f ms/token); that is THIS harness, not the real graph\n",
                   1000.0 / (res[4].wall / 1000.0), res[4].wall / 1000.0);
        }

        // ---- host-side vkResetQueryPool, the driver trap and its fix -------------------------
        // Done LAST on purpose: on this driver the instance-level entry point vkResetQueryPool()
        // jumps to address 0 and kills the process (AMD 2.0.279 + the MinGW import lib generated
        // from C:\Windows\System32\vulkan-1.dll), so if the workaround below ever fails again the
        // measurements above are already printed.
        {
            using PFN_resetQP = void (VKAPI_PTR *)(VkDevice, VkQueryPool, uint32_t, uint32_t);
            PFN_resetQP rp = reinterpret_cast<PFN_resetQP>(vkGetDeviceProcAddr(c.dev, "vkResetQueryPool"));
            PFN_resetQP ip = reinterpret_cast<PFN_resetQP>(vkGetInstanceProcAddr(c.inst, "vkResetQueryPool"));
            printf("  vkResetQueryPool: vkGetDeviceProcAddr -> %s, vkGetInstanceProcAddr -> %s "
                   "(instance asked for API %u.%u; BONSAI_API12=1 asks for 1.2)\n",
                   rp ? "resolved" : "NULL", ip ? "resolved" : "NULL",
                   VK_VERSION_MAJOR(ai.apiVersion), VK_VERSION_MINOR(ai.apiVersion));
            if (rp) {
                rp(c.dev, c.qp, 420, 2);
                printf("  device entry point called: returned, no crash.\n");
            } else if (ip) {
                ip(c.dev, c.qp, 420, 2);
                printf("  instance entry point called with the device handle: returned, no crash.\n");
            } else {
                printf("  neither resolves: this build has no host-side vkResetQueryPool at all, so "
                       "the recorded vkCmdResetQueryPool() is not a workaround, it is the only path. "
                       "Calling the import-lib stub (what the earlier crash did) is undefined.\n");
            }
        }
        // Cause isolation, deliberately LAST because it can kill the process. The upload path used
        // to record a TRANSFER->COMPUTE pipeline barrier inside a command buffer submitted to the
        // transfer-only family, and amdvlk64.dll segfaulted on it (backtrace: amdvlk64!... <-
        // vulkan-1 loader thunk <- main, with the printf above already on screen). Submitting a
        // command buffer that contains nothing but that barrier decides whether the barrier or
        // something else is the cause.
        if (has_tx && getenv("BONSAI_TXBARRIER")) {
            VkCommandBuffer bc = VK_NULL_HANDLE;
            cai2.commandPool = tpool; cai2.commandBufferCount = 1;
            VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai2, &bc));
            VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
            beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
            VK_CHECK(vkBeginCommandBuffer(bc, &beg));
            bar_transfer_to_compute(bc);
            VK_CHECK(vkEndCommandBuffer(bc));
            VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO}; si.commandBufferCount = 1; si.pCommandBuffers = &bc;
            VK_CHECK(vkQueueSubmit(c.qtx, 1, &si, fone));
            wait_f(c.dev, fone);
            printf("  probe BONSAI_TXBARRIER: a pipeline barrier alone on the transfer-only queue "
                   "SURVIVED -> the crash had a different cause\n");
            vkFreeCommandBuffers(c.dev, tpool, 1, &bc);
        }
        VK_CHECK(vkQueueWaitIdle(c.queue));
        if (has_tx) VK_CHECK(vkQueueWaitIdle(c.qtx));   // before the pools and CBs go away
        for (uint32_t i = 0; i < OPS; ++i) vkResetCommandBuffer(opcb[i], 0);
        vkDestroyFence(c.dev, fone, nullptr);
        for (int i = 0; i < 2; ++i) {
            vkDestroyFence(c.dev, ff[i], nullptr); vkDestroyFence(c.dev, fc[i], nullptr);
            vkDestroyFence(c.dev, fd[i], nullptr); vkDestroySemaphore(c.dev, ss[i], nullptr);
            vkDestroySemaphore(c.dev, sd[i], nullptr);
        }
        if (tpool) vkDestroyCommandPool(c.dev, tpool, nullptr);
        vkUnmapMemory(c.dev, p_host);
        vkDestroyBuffer(c.dev, logd, nullptr); vkDestroyBuffer(c.dev, logh, nullptr);
        vkFreeMemory(c.dev, p_dev, nullptr); vkFreeMemory(c.dev, p_host, nullptr);
    }

    for (auto &bb : bufs) vkDestroyBuffer(c.dev, bb.buf, nullptr);
    vkFreeMemory(c.dev, c.arena, nullptr);
    vkFreeMemory(c.dev, c.host_mem, nullptr);
    vkDestroyDevice(c.dev, nullptr);
    vkDestroyInstance(c.inst, nullptr);
    return 0;
}
