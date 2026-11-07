// bonsai_weights.h - the full-checkpoint weight path: place every tensor in one device-local
// arena, stream the 1.823 GiB of PTQ1_0 blocks over PCIe in row chunks through a small scratch,
// repack them on the GPU into the 2-bit random-access layout the measured ternary GEMV reads, and
// then PROVE the bytes landed where the plan says they should by running the engine's own GEMV
// against a CPU reference that dequantizes with ptq_decode.h.
//
// WHY THIS IS ITS OWN INCLUDE
//   The benchmark in bonsai_vk.cpp uploads ONE synthetic tensor. The engine needs 427 resident at
//   once, and on a 4 GB card that is a VRAM budget problem before it is a code problem: 2.083 GiB
//   of codes + 0.260 GiB of scales + 28 MiB of f32 + 50 MiB of recurrent state + the KV cache.
//   Every byte allocated here is a byte the context window cannot have, so the staging scratch is
//   small and row-chunked (a scratch big enough for output.weight is 212 MiB = 6664 tokens of KV
//   thrown away) and the offsets come from the same ModelPlan that plan_probe.exe prints, never
//   from a second copy of the arithmetic.
//
// THE HANDOFF DISCIPLINE (all of it measured - STAGE3, STAGE5, END_TO_END_MEASURED.md)
//   - copies run on the TRANSFER-only family (SDMA, 7.11 GB/s), not behind compute (4.93 GB/s),
//     which is only legal because every buffer a copy touches is VK_SHARING_MODE_CONCURRENT over
//     both families;
//   - the copy submit signals a semaphore the repack submit waits on, with the wait stage at
//     COMPUTE_SHADER_BIT. A host fence alone is not the documented dependency mechanism;
//   - two dispatches in one command buffer have no memory dependency unless the host records a
//     barrier, so every repack is followed by bar_compute(). Missing that barrier is STAGE3's
//     entire "correctness rel=5e16" story;
//   - vkCmdResetQueryPool must never see a NULL query pool: that is an amdvlk64.dll SIGSEGV.
#ifndef BONSAI_WEIGHTS_H
#define BONSAI_WEIGHTS_H

#include <chrono>
#include <cmath>
#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "bonsai_model.h"
#include "bonsai_platform.h"   // the OS layer. Included HERE, from the earliest engine header, so
                               // winsock2.h is seen before any windows.h in the translation unit.

// ------------------------------------------------------------------ device setup
// The same selection logic the benchmark uses, hoisted into a function so the engine and the
// benchmark cannot drift apart on which card they are talking about.
static bool vk_ctx_init(Ctx &c, const char *app) {
    VkApplicationInfo ai{VK_STRUCTURE_TYPE_APPLICATION_INFO};
    // 1.2, not 1.1: vkResetQueryPool is core in 1.2, and on a 1.1 instance vkGetDeviceProcAddr
    // returns NULL for it - jumping into NULL is what took down amdvlk64.dll last time.
    ai.apiVersion = VK_API_VERSION_1_2;
    ai.pApplicationName = app;
    VkInstanceCreateInfo ici{VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO};
    ici.pApplicationInfo = &ai;
    VkResult r = vkCreateInstance(&ici, nullptr, &c.inst);
    if (r != VK_SUCCESS) { fprintf(stderr, "no Vulkan runtime: %d\n", (int)r); return false; }

    uint32_t ndev = 0; VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &ndev, nullptr));
    std::vector<VkPhysicalDevice> devs(ndev);
    VK_CHECK(vkEnumeratePhysicalDevices(c.inst, &ndev, devs.data()));
    c.phys = devs[0];
    for (auto p : devs) {
        VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(p, &pp);
        if (pp.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && pp.vendorID == 0x1002) c.phys = p;
    }
    VkPhysicalDeviceProperties props; vkGetPhysicalDeviceProperties(c.phys, &props);
    vkGetPhysicalDeviceMemoryProperties(c.phys, &c.mp);
    c.limits = props.limits;
    uint64_t ldd_heap = 0;
    for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i)
        if (c.mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
            ldd_heap = c.mp.memoryHeaps[c.mp.memoryTypes[i].heapIndex].size; break;
        }
    printf("device: %s | driver %u.%u.%u | device-local heap %u MiB | timestampPeriod %.3f ns\n",
           props.deviceName,
           VK_VERSION_MAJOR(props.driverVersion), VK_VERSION_MINOR(props.driverVersion),
           VK_VERSION_PATCH(props.driverVersion), unsigned(ldd_heap / 1048576), c.limits.timestampPeriod);
    printf("  limits: maxComputeWorkGroupSize[0]=%u maxComputeWorkGroupCount[0]=%u "
           "bufferImageGranularity=%llu\n",
           c.limits.maxComputeWorkGroupSize[0], c.limits.maxComputeWorkGroupCount[0],
           (unsigned long long)c.limits.bufferImageGranularity);
    // Queried, not assumed. ggml-vulkan prints 32768 B for this card in its own device banner, but
    // widening attn_qk_softmax's shared arrays is exactly the kind of change that has to be sized
    // against the driver's real answer - two float[MAXCAP] arrays cost 8*MAXCAP bytes.
    printf("  limits: maxComputeSharedMemorySize=%u B (= %u KiB) maxComputeWorkGroupInvocations=%u\n",
           c.limits.maxComputeSharedMemorySize, c.limits.maxComputeSharedMemorySize / 1024u,
           c.limits.maxComputeWorkGroupInvocations);

    uint32_t nq = 0; vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(c.phys, &nq, qf.data());
    for (uint32_t i = 0; i < nq; ++i) if (qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) { c.qfi = i; break; }
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
           c.tfi == 0xFFFFFFFFu ? " (NONE - copies serialise behind compute)" : "");
    return true;
}

// ------------------------------------------------------------------ one arena, many buffers
// The device-local heap, read the honest way: the heap of the first memory type that carries
// DEVICE_LOCAL. Two call sites need it - the plan report and the OOM budget table in
// bonsai_forward.h - and the plan report used to be handed a literal 0, which printed
// "device-local heap: 0 MiB" into every bring-up log while this card reports 3840 MiB.
static uint64_t device_local_heap(const Ctx &c) {
    for (uint32_t i = 0; i < c.mp.memoryTypeCount; ++i)
        if (c.mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)
            return c.mp.memoryHeaps[c.mp.memoryTypes[i].heapIndex].size;
    return 0;
}

struct Arena {
    Ctx *c = nullptr;
    VkDeviceMemory mem = VK_NULL_HANDLE;
    uint64_t cursor = 0, size = 0, pad = 0;
    uint32_t mem_type = 0;
    bool has_tx = false;
    uint32_t fam2[2] = {0, 0};

    void init(Ctx &cc) {
        c = &cc; has_tx = (cc.tfi != 0xFFFFFFFFu); fam2[0] = cc.qfi; fam2[1] = cc.tfi;
        for (uint32_t i = 0; i < cc.mp.memoryTypeCount; ++i)
            if (cc.mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                mem_type = i; break;
            }
    }
    // Pass one: create the buffer only to ASK the driver what alignment it wants, then record the
    // offset. Alignment is never assumed here - assuming it is how the first arena build read
    // zeros out of a buffer that was not where the descriptor said it was.
    uint64_t reserve(VkBufferUsageFlags usage, uint64_t bytes, const char *name) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes; bi.usage = usage;
        if (has_tx) {
            bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fam2;
        } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        VkBuffer b; VK_CHECK(vkCreateBuffer(c->dev, &bi, nullptr, &b));
        VkMemoryRequirements mr; vkGetBufferMemoryRequirements(c->dev, b, &mr);
        uint64_t a = align_up_u64(cursor, mr.alignment);
        pad += a - cursor; cursor = a + mr.size;
        vkDestroyBuffer(c->dev, b, nullptr);
        printf("  %-12s %13llu B  align %-4llu -> offset %llu\n", name,
               (unsigned long long)bytes, (unsigned long long)mr.alignment, (unsigned long long)a);
        return a;
    }
    Buf create_at(VkBufferUsageFlags usage, uint64_t bytes, uint64_t off, const char *name) {
        VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
        bi.size = bytes; bi.usage = usage;
        if (has_tx) {
            bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
            bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fam2;
        } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        Buf r; r.buf = VK_NULL_HANDLE;
        VK_CHECK(vkCreateBuffer(c->dev, &bi, nullptr, &r.buf));
        VK_CHECK(vkBindBufferMemory(c->dev, r.buf, mem, off));
        r.offset = off; r.size = bytes;
        (void)name;
        return r;
    }
    bool commit() {
        VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        ma.allocationSize = cursor; ma.memoryTypeIndex = mem_type;
        size = cursor;
        VkResult r = vkAllocateMemory(c->dev, &ma, nullptr, &mem);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "arena: vkAllocateMemory(%llu MiB) FAILED (%d). This is the 4 GB wall:\n"
                            "  the card's device-local heap minus the driver's reservation is the\n"
                            "  whole budget for weights + state + activations + KV.\n",
                    (unsigned long long)(cursor >> 20), (int)r);
            return false;
        }
        return true;
    }
};

// ------------------------------------------------------------------ engine state
struct Engine {
    Ctx c;
    GgufHeader hdr;
    ModelPlan plan;
    std::string path;
    FILE *f = nullptr;

    Arena arena;
    Buf codes, scales, f32blob, scratch;         // device-local, the whole weight set
    Buf x, xsum, out;                             // the small working set, used by verification now
    uint64_t chunk_bytes = 32ull << 20;           // the single biggest VRAM knob

    VkDeviceMemory host_mem = VK_NULL_HANDLE;
    VkBuffer host_buf = VK_NULL_HANDLE;           // TRANSFER_SRC view of the staging allocation
    void *host_ptr = nullptr;
    uint32_t host_type = 0;

    VkCommandPool cpool = VK_NULL_HANDLE, upool = VK_NULL_HANDLE;
    VkCommandBuffer cmd = VK_NULL_HANDLE;
    VkQueryPool qp = VK_NULL_HANDLE;
    VkDescriptorPool dpool = VK_NULL_HANDLE;
    VkSemaphore up_sig = VK_NULL_HANDLE;
    VkFence up_fence = VK_NULL_HANDLE;

    // repack
    VkPipeline p_repack = VK_NULL_HANDLE;
    VkPipelineLayout pl_repack = VK_NULL_HANDLE;
    VkDescriptorSetLayout sl_repack = VK_NULL_HANDLE;
    VkDescriptorSet ds_repack = VK_NULL_HANDLE;
    // verify: xsum -> gemv -> reduce, the engine's real projection path
    VkPipeline pp3[3] = {};
    VkPipelineLayout pl3[3] = {};
    VkDescriptorSetLayout sl3[3] = {};
    VkDescriptorSet ds3[3] = {};

    double ms_sdma = 0, ms_repack = 0;
    uint64_t bytes_ptq = 0, bytes_f32 = 0, n_batches = 0, n_repack_dispatch = 0;
};

// ------------------------------------------------------------------ small plumbing
// Every dispatch group in this engine records into a freshly allocated primary command buffer, which
// is the normal one-submit-per-token idiom. BONSAI_CB_REUSE=1 makes the forward loop share ONE buffer
// and reset it instead: vkAllocateCommandBuffers from a pool that is never reset returns nothing to
// the driver, and on this AMD driver a command buffer costs GPU-side tracking memory, so the P2 watch
// can climb with the number of submits rather than the number of tokens. The knob exists to tell those
// two explanations apart with a measurement instead of with an argument. It is set only from the
// forward phase (fwd_main, after the weights are resident and waited), so a weight-upload command
// buffer that was submitted across queues is never reset under it.
//
// MEASURED (this card, RX 550X 4GB, 23-token prompt + 60 generated tokens, PDH Shared Usage sampled
// every token): one buffer per dispatch group -> 53.8 MiB climbing to 79.3 MiB while Dedicated Usage
// stays flat; one reused buffer -> flat 40.93 MiB at the same 14.22 tok/s. So the climb is driver-side
// command-buffer tracking, not KV and not a VidMm eviction of our arena. The reuse is therefore the
// default now; runs/p2_rate_q8_v1.log and runs/p2_rate_q8_cbreuse_v1.log are the two curves.
// g_cb_in_forward makes the confinement explicit rather than a comment: before fwd_main calls
// cb_enter_forward() the allocator path is used even with reuse on, so an upload buffer recorded on a
// queue that is synchronised by a semaphore alone is never reset under anyone's feet.
static uint32_t        g_cb_reuse = 1;      // 1 = reuse one buffer in the forward phase; 0 = allocate every time
static uint32_t        g_cb_in_forward = 0; // set once by cb_enter_forward(); never cleared
static uint64_t        g_cb_alloc = 0, g_cb_reset = 0;
// Phase split, so "MiB per command buffer" can be measured instead of argued: everything allocated
// before cb_enter_forward() belongs to weight loading and is not part of the per-token story.
static uint64_t        g_cb_alloc_load = 0, g_cb_begin_load = 0;
static VkCommandBuffer g_cb_shared = nullptr;

// End of the load phase: snapshot the counters and switch the reuse on. Idempotent.
static void cb_enter_forward() {
    g_cb_alloc_load = g_cb_alloc;
    g_cb_begin_load = g_cb_alloc + g_cb_reset;
    g_cb_in_forward = 1;
}

static VkCommandBuffer begin_cmd(Ctx &c, VkCommandPool pool) {
    VkCommandBuffer cb = nullptr;
    if (g_cb_reuse && g_cb_in_forward && g_cb_shared) {
        // Legal only because of the property above: each of these buffers is submitted and fence-waited
        // before the next begin_cmd() runs, so a reset cannot touch a submit that is still in flight.
        VK_CHECK(vkResetCommandBuffer(g_cb_shared, 0));
        ++g_cb_reset;
        cb = g_cb_shared;
    } else {
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = pool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
        VK_CHECK(vkAllocateCommandBuffers(c.dev, &cai, &cb));
        ++g_cb_alloc;
        if (g_cb_reuse && g_cb_in_forward) g_cb_shared = cb;
    }
    VkCommandBufferBeginInfo beg{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    beg.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(cb, &beg));
    return cb;
}

// One submit, with the two directions of the cross-queue handoff spelled out: this submit can wait
// on a semaphore (recorded at COMPUTE_SHADER_BIT, the stage that reads what the copy wrote) and can
// signal one for the next queue to wait on. A fence is only used when the host wants the result.
static void submit_wait_sig(Ctx &c, VkQueue q, VkCommandBuffer cb, VkFence fence,
                            VkSemaphore wait, VkSemaphore signal) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    VkPipelineStageFlags stg = VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT;
    if (wait) { si.waitSemaphoreCount = 1; si.pWaitSemaphores = &wait; si.pWaitDstStageMask = &stg; }
    if (signal) { si.signalSemaphoreCount = 1; si.pSignalSemaphores = &signal; }
    VK_CHECK(vkQueueSubmit(q, 1, &si, fence));
    if (fence) {
        VK_CHECK(vkWaitForFences(c.dev, 1, &fence, VK_TRUE, ~0ull));
        // Reset, or the fence stays signalled and every later "wait" returns immediately. The first
        // load run measured 1866 MiB of SDMA in 12.5 ms = 156 GB/s, which is not a bandwidth, it is
        // a fence that was never armed again. PCIe on this box cannot do more than ~7 GB/s.
        VK_CHECK(vkResetFences(c.dev, 1, &fence));
    }
}

static void stage_alloc(Engine &e, uint64_t bytes) {
    bool ok = false;
    for (uint32_t i = 0; i < e.c.mp.memoryTypeCount && !ok; ++i) {
        VkMemoryPropertyFlags w = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((e.c.mp.memoryTypes[i].propertyFlags & w) == w) { e.host_type = i; ok = true; }
    }
    if (!ok) { fprintf(stderr, "no host-visible coherent memory type\n"); exit(1); }
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = bytes; ma.memoryTypeIndex = e.host_type;
    VK_CHECK(vkAllocateMemory(e.c.dev, &ma, nullptr, &e.host_mem));
    void *p = nullptr; VK_CHECK(vkMapMemory(e.c.dev, e.host_mem, 0, bytes, 0, &p));
    e.host_ptr = p;
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = bytes; bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    // The array lives in the SAME scope as the vkCreateBuffer below, not in the if-block that
    // decides the sharing mode. Declaring it inside the block was a dangling pQueueFamilyIndices:
    // the block's stack frame is dead by the time the driver reads it, so the driver received two
    // garbage family indices, indexed its own per-family tables out of bounds and SIGSEGVed inside
    // amdvlk64.dll. -O0 hid it (the frame still held the values); -O2 exposed it. VkBufferCreateInfo
    // copies nothing - the pointer is read during the call, so its lifetime is the call's.
    uint32_t fam2[2] = {e.c.qfi, e.c.tfi};
    if (e.c.tfi != 0xFFFFFFFFu) {
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2;
        bi.pQueueFamilyIndices = fam2;
    } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VK_CHECK(vkCreateBuffer(e.c.dev, &bi, nullptr, &e.host_buf));
    VK_CHECK(vkBindBufferMemory(e.c.dev, e.host_buf, e.host_mem, 0));
}

// ------------------------------------------------------------------ bring-up
static bool engine_init(Engine &e, const char *gguf_path, uint64_t chunk_bytes) {
    e.path = gguf_path;
    e.chunk_bytes = chunk_bytes;
    if (!vk_ctx_init(e.c, "bonsai_gen")) return false;
    if (!gguf_load(gguf_path, e.hdr)) return false;
    if (!model_plan_build(e.hdr, e.plan)) return false;
    // THE CONTRACT GATE, and it runs HERE because this is the last line before the first byte of
    // device-local memory is reserved (the arena is created below). model_plan_build() deliberately
    // does not call it - plan_probe.cpp and every budget run must still be able to print a layout for
    // a file the engine refuses to load - so the refusal has exactly one home, and it is this one.
    // Layer typing, the prism.hadamard.version gate and the structural invariants (no KV slot for a
    // recurrent layer, a state_size the GDN kernel can actually address, a Hadamard block that tiles
    // the activation width) are all decided before any upload, because a wrong answer downstream is
    // not a crash but a plausible token stream.
    std::string cwhy;
    if (!model_plan_contract(e.hdr, e.plan, cwhy)) {
        fprintf(stderr, "\n[!] checkpoint REJECTED, nothing allocated:\n    %s\n", cwhy.c_str());
        return false;
    }
    // Both V modes are validated because the engine picks one at RUN time from BONSAI_KV_V; the plan
    // must be sound for whichever the user asks for.
    for (int vq = 0; vq <= 1; ++vq) {
        std::string vwhy;
        if (!model_plan_validate(e.plan, vq, vwhy)) {
            fprintf(stderr, "\n[!] checkpoint REJECTED (V storage = %s), nothing allocated:\n    %s\n",
                    vq ? "q8_0" : "fp16", vwhy.c_str());
            return false;
        }
    }
    model_plan_report(e.plan, device_local_heap(e.c) / 1048576ull);

    const uint64_t U = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                       VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    printf("\narena (single DEVICE_LOCAL allocation; sizes are the plan's, offsets are the driver's):\n");
    e.arena.init(e.c);
    uint64_t sz_codes = e.plan.bytes_codes_total, sz_scale = e.plan.bytes_scale_total;
    uint64_t sz_f32 = e.plan.bytes_f32_total;
    e.codes.offset   = e.arena.reserve(U, sz_codes, "codes");
    e.scales.offset  = e.arena.reserve(U, sz_scale, "scales");
    e.f32blob.offset = e.arena.reserve(U, sz_f32, "f32blob");
    e.scratch.offset = e.arena.reserve(U, chunk_bytes, "ptq_scratch");
    e.x.offset    = e.arena.reserve(U, 65536, "x");   // K is up to 12288 floats = 48 KiB
    e.xsum.offset = e.arena.reserve(U, 4096, "xsum");
    e.out.offset  = e.arena.reserve(U, 8u << 20, "out");
    if (!e.arena.commit()) return false;
    e.codes      = e.arena.create_at(U, sz_codes, e.codes.offset, "codes");
    e.scales     = e.arena.create_at(U, sz_scale, e.scales.offset, "scales");
    e.f32blob    = e.arena.create_at(U, sz_f32, e.f32blob.offset, "f32blob");
    e.scratch    = e.arena.create_at(U, chunk_bytes, e.scratch.offset, "ptq_scratch");
    e.x          = e.arena.create_at(U, 65536, e.x.offset, "x");
    e.xsum       = e.arena.create_at(U, 4096, e.xsum.offset, "xsum");
    e.out        = e.arena.create_at(U, 8u << 20, e.out.offset, "out");
    printf("  arena committed %.3f GiB, alignment padding %llu B\n",
           double(e.arena.size) / 1073741824.0, (unsigned long long)e.arena.pad);

    stage_alloc(e, chunk_bytes);

    VkCommandPoolCreateInfo cpi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    cpi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; cpi.queueFamilyIndex = e.c.qfi;
    VK_CHECK(vkCreateCommandPool(e.c.dev, &cpi, nullptr, &e.cpool));
    cpi.queueFamilyIndex = (e.c.tfi != 0xFFFFFFFFu) ? e.c.tfi : e.c.qfi;
    VK_CHECK(vkCreateCommandPool(e.c.dev, &cpi, nullptr, &e.upool));
    VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
    cai.commandPool = e.cpool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY; cai.commandBufferCount = 1;
    VK_CHECK(vkAllocateCommandBuffers(e.c.dev, &cai, &e.cmd));
    VkQueryPoolCreateInfo qpi{VK_STRUCTURE_TYPE_QUERY_POOL_CREATE_INFO};
    qpi.queryType = VK_QUERY_TYPE_TIMESTAMP; qpi.queryCount = 512;
    VK_CHECK(vkCreateQueryPool(e.c.dev, &qpi, nullptr, &e.qp));
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    VK_CHECK(vkCreateSemaphore(e.c.dev, &sci, nullptr, &e.up_sig));
    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    VK_CHECK(vkCreateFence(e.c.dev, &fci, nullptr, &e.up_fence));

    e.sl_repack = make_set_layout(e.c, 3);
    e.pl_repack = make_pipe_layout(e.c, e.sl_repack, 32);
    e.p_repack = make_pipeline(e.c, "shaders/repack_ptq1_0.spv", e.pl_repack);
    // gemv needs CODES, XVEC, SCALE, XSUM, OUT; xsum needs XVEC, XSUM; reduce needs OUT.
    e.sl3[0] = make_set_layout(e.c, 2); e.pl3[0] = make_pipe_layout(e.c, e.sl3[0], 32);
    e.sl3[1] = make_set_layout(e.c, 5); e.pl3[1] = make_pipe_layout(e.c, e.sl3[1], 32);
    e.sl3[2] = make_set_layout(e.c, 1); e.pl3[2] = make_pipe_layout(e.c, e.sl3[2], 32);
    e.pp3[0] = make_pipeline(e.c, "shaders/xsum.spv", e.pl3[0]);
    e.pp3[1] = make_pipeline(e.c, "shaders/gemv_ternary.spv", e.pl3[1]);
    e.pp3[2] = make_pipeline(e.c, "shaders/reduce.spv", e.pl3[2]);
    VkDescriptorPoolSize psz{VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 32};
    VkDescriptorPoolCreateInfo dpi{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
    dpi.maxSets = 8; dpi.poolSizeCount = 1; dpi.pPoolSizes = &psz;
    VK_CHECK(vkCreateDescriptorPool(e.c.dev, &dpi, nullptr, &e.dpool));
    {
        std::vector<VkDescriptorSetLayout> lays = {e.sl_repack, e.sl3[0], e.sl3[1], e.sl3[2]};
        std::vector<VkDescriptorSet> sets(4);
        VkDescriptorSetAllocateInfo sai{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        sai.descriptorPool = e.dpool; sai.descriptorSetCount = 4; sai.pSetLayouts = lays.data();
        VK_CHECK(vkAllocateDescriptorSets(e.c.dev, &sai, sets.data()));
        e.ds_repack = sets[0]; e.ds3[0] = sets[1]; e.ds3[1] = sets[2]; e.ds3[2] = sets[3];
    }
    bind_set(e.c, e.ds_repack, {e.scratch, e.codes, e.scales});
    bind_set(e.c, e.ds3[0], {e.x, e.xsum});
    bind_set(e.c, e.ds3[1], {e.codes, e.x, e.scales, e.xsum, e.out});
    bind_set(e.c, e.ds3[2], {e.out});

    e.f = fopen(gguf_path, "rb");
    if (!e.f) { fprintf(stderr, "cannot reopen %s\n", gguf_path); return false; }
    return true;
}

// ------------------------------------------------------------------ the loader
// A batch is "as many consecutive PTQ tensors as fit in the scratch", or one row chunk of a tensor
// bigger than the scratch. Both cases cost one SDMA submit plus one repack submit, so the sync
// count is ~(1.823 GiB / chunk) rather than one pair per tensor.
struct BatchItem { uint64_t stage_off; const WeightPlan *w; uint32_t row0, nrows; };

static void flush_batch(Engine &e, std::vector<BatchItem> &batch) {
    if (batch.empty()) return;
    auto t0 = std::chrono::steady_clock::now();
    // 1) the copy: staging -> scratch. dstOffset is relative to the SCRATCH buffer (it is already
    //    bound at e.scratch.offset in the arena) - passing the arena offset here applies it twice.
    {
        VkCommandBuffer ucb = begin_cmd(e.c, e.upool);
        std::vector<VkBufferCopy> cps;
        cps.reserve(batch.size());
        for (auto &it : batch)
            cps.push_back({it.stage_off, it.stage_off,
                           uint64_t(it.nrows) * it.w->n_blk * PTQ_BLOCK_BYTES});
        vkCmdCopyBuffer(ucb, e.host_buf, e.scratch.buf, uint32_t(cps.size()), cps.data());
        VK_CHECK(vkEndCommandBuffer(ucb));
        submit_wait_sig(e.c, (e.c.tfi != 0xFFFFFFFFu) ? e.c.qtx : e.c.queue, ucb,
                        e.up_fence, VK_NULL_HANDLE, e.up_sig);
    }
    auto t1 = std::chrono::steady_clock::now();
    // 2) the repack, on the compute queue, waiting on the semaphore the SDMA just signalled.
    {
        VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
        for (auto &it : batch) {
            uint32_t pc[8] = {};
            pc[0] = it.w->M; pc[1] = it.w->K; pc[2] = uint32_t(it.w->n_blk);
            pc[3] = uint32_t(it.stage_off / 4u);      // SRC is uint, so words
            pc[4] = uint32_t(it.w->codes_word0);      // DST is uvec4, so 16 B units
            pc[5] = uint32_t(it.w->scale_word0);      // SCL is uint
            pc[6] = it.row0; pc[7] = it.row0 + it.nrows;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.p_repack);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.pl_repack, 0, 1,
                                    &e.ds_repack, 0, nullptr);
            vkCmdPushConstants(cb, e.pl_repack, VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
            vkCmdDispatch(cb, uint32_t(it.w->n_blk), (it.nrows + 63u) / 64u, 1);
            bar_compute(cb);
            e.n_repack_dispatch++;
        }
        VK_CHECK(vkEndCommandBuffer(cb));
        submit_wait_sig(e.c, e.c.queue, cb, VK_NULL_HANDLE, e.up_sig, VK_NULL_HANDLE);
        VK_CHECK(vkQueueWaitIdle(e.c.queue));
    }
    auto t2 = std::chrono::steady_clock::now();
    e.ms_sdma += std::chrono::duration<double, std::milli>(t1 - t0).count();
    e.ms_repack += std::chrono::duration<double, std::milli>(t2 - t1).count();
    e.n_batches++;
    batch.clear();
}

static bool read_at(Engine &e, uint64_t file_pos, uint64_t stage_pos, uint64_t nbytes, const char *who) {
    if (!bonsai_seek(e.f, file_pos)) { fprintf(stderr, "%s: seek\n", who); return false; }
    if (fread((char *)e.host_ptr + stage_pos, 1, size_t(nbytes), e.f) != nbytes) {
        fprintf(stderr, "%s: short read at %llu\n", who, (unsigned long long)file_pos);
        return false;
    }
    return true;
}

static bool load_ptq_weights(Engine &e) {
    std::vector<BatchItem> batch;
    uint64_t stage_used = 0;
    for (const auto &w : e.plan.w) {
        if (!w.quantized()) continue;
        const uint64_t row_bytes = w.n_blk * PTQ_BLOCK_BYTES;
        if (row_bytes > e.chunk_bytes) {
            fprintf(stderr, "ptq: %s is %llu B per row, scratch is %llu\n", w.name.c_str(),
                    (unsigned long long)row_bytes, (unsigned long long)e.chunk_bytes);
            return false;
        }
        uint32_t done = 0;
        while (done < w.M) {
            uint32_t fit = uint32_t((e.chunk_bytes - stage_used) / row_bytes);
            if (fit == 0) { flush_batch(e, batch); stage_used = 0; fit = uint32_t(e.chunk_bytes / row_bytes); }
            uint32_t nrows = std::min(fit, w.M - done);
            const uint64_t file_pos = e.plan.data_start + w.file_off + uint64_t(done) * row_bytes;
            const uint64_t nbytes = uint64_t(nrows) * row_bytes;
            if (!read_at(e, file_pos, stage_used, nbytes, w.name.c_str())) return false;
            batch.push_back({stage_used, &w, done, nrows});
            stage_used += align_up_b(nbytes, 16);
            done += nrows;
            e.bytes_ptq += nbytes;
        }
    }
    flush_batch(e, batch);
    return true;
}

// The 225 f32 tensors (norms, biases, ssm_a/dt/conv1d/norm, alpha, beta) are 28 MiB total and are
// NOT repacked - the forward reads them as plain floats. Their arena offsets came from ModelPlan.
static bool load_f32_weights(Engine &e) {
    uint64_t stage_used = 0;
    std::vector<VkBufferCopy> cps;
    auto flush = [&]() {
        if (cps.empty()) return;
        VkCommandBuffer ucb = begin_cmd(e.c, e.upool);
        vkCmdCopyBuffer(ucb, e.host_buf, e.f32blob.buf, uint32_t(cps.size()), cps.data());
        VK_CHECK(vkEndCommandBuffer(ucb));
        submit_wait_sig(e.c, (e.c.tfi != 0xFFFFFFFFu) ? e.c.qtx : e.c.queue, ucb,
                        e.up_fence, VK_NULL_HANDLE, e.up_sig);
        VK_CHECK(vkQueueWaitIdle(e.c.queue));
        cps.clear(); stage_used = 0;
    };
    for (const auto &w : e.plan.w) {
        if (w.quantized()) continue;
        const uint64_t nbytes = w.bytes_f32;
        if (stage_used + nbytes > e.chunk_bytes) flush();
        if (nbytes > e.chunk_bytes) { fprintf(stderr, "f32: %s too big\n", w.name.c_str()); return false; }
        if (!read_at(e, e.plan.data_start + w.file_off, stage_used, nbytes, w.name.c_str())) return false;
        cps.push_back({stage_used, w.f32_off, nbytes});
        stage_used += align_up_b(nbytes, 16);
        e.bytes_f32 += nbytes;
    }
    flush();
    return true;
}

// ------------------------------------------------------------------ the loader report
static void load_report(Engine &e) {
    const double s_sd = e.ms_sdma / 1000.0, s_rp = e.ms_repack / 1000.0;
    printf("\n=== weight load, measured on this card ===\n");
    printf("  SDMA  %8.1f MiB in %7.1f ms = %5.2f GB/s (%s)\n",
           e.bytes_ptq / 1048576.0, e.ms_sdma, e.bytes_ptq / (s_sd ? s_sd : 1e-9) / 1e9,
           e.c.tfi != 0xFFFFFFFFu ? "transfer-only queue" : "compute queue");
    printf("  repack %8.1f MiB -> %.1f MiB in %7.1f ms = %5.2f GB/s effective, %llu dispatches in %llu batches\n",
           e.bytes_ptq / 1048576.0, (e.plan.bytes_codes_total + e.plan.bytes_scale_total) / 1048576.0,
           e.ms_repack,
           (e.bytes_ptq + e.plan.bytes_codes_total + e.plan.bytes_scale_total) /
               (s_rp ? s_rp : 1e-9) / 1e9,
           (unsigned long long)e.n_repack_dispatch, (unsigned long long)e.n_batches);
    printf("  f32blob %8.1f MiB   TOTAL %.1f ms\n", e.bytes_f32 / 1048576.0, e.ms_sdma + e.ms_repack);
}

// ------------------------------------------------------------------ verification
// "The loader finished" is not "the weights are right". This runs the engine's real projection path
// (xsum -> gemv -> reduce) on real tensors and compares against a CPU reference that dequantizes
// with the bit-validated ptq_decode_ours() and accumulates in float64. A wrong codes_word0, a wrong
// scale stride, or a mis-tiled chunk shows up here as a mismatch, which is the only reason to trust
// the rest of the forward.
static uint32_t pick_splits(uint32_t M, uint32_t nblk) {
    // Same rule the probe measured: fill the 40 SIMDs (64-wide workgroups => ~640 groups), and the
    // split count must divide the block count (gemv's contract, host-enforced).
    uint32_t s = 1;
    while (s < 16u && uint64_t(M / 64u + (M % 64u != 0)) * s < 640ull && (nblk % (s * 2u) == 0)) s *= 2u;
    return s;
}

// ------------------------------------------------------------------ byte-level layout proof
// The GEMV comparison above says "the number is wrong"; it cannot say WHICH of the two halves of
// the repack (codes, scale) or which row range is wrong. This reads the actual 32 code bytes + the
// scale word back out of the arena and compares them to the packing computed on the host from the
// file's own block bytes with ptq_decode.h's element map. It is independent of the GEMV, so a
// mismatch localises to the loader, and a match plus a GEMV mismatch localises to the kernel.
static int verify_block_layout(Engine &e, const WeightPlan &w, uint32_t row, uint32_t blk) {
    if (!w.quantized()) { printf("  %-24s not quantized - no codes/scales to check\n", w.name.c_str()); return 0; }
    std::vector<uint8_t> pb(PTQ_BLOCK_BYTES);
    const uint64_t pos = e.plan.data_start + w.file_off + (uint64_t(row) * w.n_blk + blk) * PTQ_BLOCK_BYTES;
    if (!bonsai_seek(e.f, pos) ||
        fread(pb.data(), 1, PTQ_BLOCK_BYTES, e.f) != PTQ_BLOCK_BYTES) {
        fprintf(stderr, "layout: cannot read %s row %u blk %u\n", w.name.c_str(), row, blk);
        return -1;
    }
    uint32_t exp[9] = {};                       // 8 u32 of codes (two uvec4 words) + the scale word
    for (int el = 0; el < 128; ++el) {
        int by, tr; elem_to_byte_trit(el, by, tr);
        const uint32_t t = uint32_t(trit_of_byte(pb[by], tr));
        exp[el >> 4] |= t << ((el & 15) * 2);   // exactly repack_ptq1_0.comp's packing rule
    }
    exp[8] = uint32_t(pb[26]) | (uint32_t(pb[27]) << 8);       // the raw f16 bits, not the float

    // A block is TWO uvec4 words that live M uvec4 apart, not 16 B apart: codes[co + 2*blk*M + row]
    // and codes[co + (2*blk+1)*M + row]. The first version of this checker asked for 32 CONTIGUOUS
    // bytes, so words 4..7 came out of the neighbouring row and were reported as mismatches while
    // the real stores were bit-exact. Three separate copies, one per thing being read.
    const uint64_t code_byte = (w.codes_word0 + uint64_t(2u * blk) * w.M + row) * 16ull;
    const uint64_t code_byte1 = code_byte + uint64_t(w.M) * 16ull;
    const uint64_t scl_byte  = (w.scale_word0 + uint64_t(blk) * w.M + row) * 4ull;
    {
        VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
        VkBufferCopy cp[3] = {{code_byte, 0, 16}, {code_byte1, 16, 16}, {scl_byte, 32, 4}};
        vkCmdCopyBuffer(cb, e.codes.buf, e.host_buf, 2, &cp[0]);
        vkCmdCopyBuffer(cb, e.scales.buf, e.host_buf, 1, &cp[2]);
        VK_CHECK(vkEndCommandBuffer(cb));
        submit_wait_sig(e.c, e.c.queue, cb, e.up_fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
    }
    uint32_t got[9] = {};
    memcpy(got, e.host_ptr, 36);
    int bad = 0;
    for (int i = 0; i < 9; ++i) bad += (got[i] != exp[i]) ? 1 : 0;
    printf("  %-24s row %-7d blk %-3d %s  scale %04x\n",
           w.name.c_str(), row, blk, bad ? "MISMATCH" : "bit-exact", uint16_t(got[8]));
    printf("      gpu       w0 %08x %08x %08x %08x  w1 %08x %08x %08x %08x\n",
           got[0], got[1], got[2], got[3], got[4], got[5], got[6], got[7]);
    if (bad) printf("      expected  w0 %08x %08x %08x %08x  w1 %08x %08x %08x %08x   scale %04x\n",
                    exp[0], exp[1], exp[2], exp[3], exp[4], exp[5], exp[6], exp[7], uint16_t(exp[8]));
    return bad;
}

// The other 225 tensors are f32 and take a plain memcpy path, so their proof is a byte compare.
// Head + tail of each, because a wrong f32_off shows up as "the tail is the next tensor" and a
// short copy shows up as "the tail is stale scratch".
static int verify_f32_blob(Engine &e, const char *const *names, int n) {
    int bad = 0;
    for (int i = 0; i < n; ++i) {
        const WeightPlan *w = e.plan.find(names[i]);
        if (!w || w->quantized()) { printf("  %-24s not an f32 tensor\n", names[i]); continue; }
        const uint64_t nb = w->bytes_f32;
        const uint64_t take = nb < 4096ull ? nb : 4096ull;
        const size_t tb = size_t(take);           // separate name: vector<uint8_t>(size_t(take)) is
                                                  // the most-vexing-parse again and gcc says so
        std::vector<uint8_t> ref(tb);
        if (!bonsai_seek(e.f, e.plan.data_start + w->file_off) ||
            fread(ref.data(), 1, tb, e.f) != take) {
            fprintf(stderr, "f32: cannot read %s\n", w->name.c_str()); bad++; continue;
        }
        std::vector<uint8_t> tail_ref(tb);
        if (!bonsai_seek(e.f, e.plan.data_start + w->file_off + nb - take) ||
            fread(tail_ref.data(), 1, tb, e.f) != take) {
            fprintf(stderr, "f32: cannot read tail of %s\n", w->name.c_str()); bad++; continue;
        }
        std::vector<uint8_t> got(2 * tb);
        VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
        VkBufferCopy cp[2] = {{w->f32_off, 0, take}, {w->f32_off + nb - take, take, take}};
        vkCmdCopyBuffer(cb, e.f32blob.buf, e.host_buf, 2, cp);
        VK_CHECK(vkEndCommandBuffer(cb));
        submit_wait_sig(e.c, e.c.queue, cb, e.up_fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
        memcpy(got.data(), e.host_ptr, 2 * tb);
        const int h = memcmp(got.data(), ref.data(), tb) != 0;
        const int t = memcmp(got.data() + take, tail_ref.data(), tb) != 0;
        printf("  %-24s %-10u x %-7u  %llu B  head %s  tail %s\n", w->name.c_str(), w->K, w->M,
               (unsigned long long)nb, h ? "MISMATCH" : "exact", t ? "MISMATCH" : "exact");
        bad += h + t;
    }
    return bad;
}

static bool verify_weights(Engine &e, int n_tensors, int n_rows) {
    std::vector<const WeightPlan *> cand;
    // prefer a spread of shapes: K=4096 with M=8192/12288/4096 and K=12288 with M=4096, plus the
    // tiny-M ssm projections whose split-K path is a different code shape inside the same kernel.
    const char *want[] = {"blk.0.attn_qkv.weight", "blk.0.ffn_gate.weight", "blk.0.ffn_down.weight",
                          "blk.0.ssm_out.weight", "blk.31.attn_output.weight", "blk.7.attn_k.weight",
                          "blk.0.attn_gate.weight", "blk.7.ffn_up.weight"};
    for (auto &n : want) if (const WeightPlan *p = e.plan.find(n)) cand.push_back(p);
    if (int(cand.size()) > n_tensors) cand.resize(size_t(n_tensors));
    if (cand.empty()) { fprintf(stderr, "verify: no candidate tensors found\n"); return false; }

    // the f32 row is read back once per tensor, so the GPU result is compared row by row
    std::vector<float> hx(12288), href(12288);
    uint32_t st = 0x9E3779B9u;
    auto rnd = [&]() { st ^= st << 13; st ^= st >> 17; st ^= st << 5; return st; };

    double worst = 0; int worst_row = 0; std::string worst_name;
    uint64_t compared = 0;
    for (const WeightPlan *w : cand) {
        const uint32_t K = w->K, M = w->M;
        const uint32_t nblk = K / 128u, S = pick_splits(M, nblk);
        for (uint32_t i = 0; i < K; ++i) hx[i] = (float(rnd() & 0xFFFF) / 32768.0f) - 1.0f;
        // upload x (vec4 view: x.offset is in the X buffer, the shader indexes xb[i] as vec4)
        memcpy(e.host_ptr, hx.data(), K * 4ull);
        {
            VkCommandBuffer ucb = begin_cmd(e.c, e.upool);
            VkBufferCopy cp{0, 0, K * 4ull};
            vkCmdCopyBuffer(ucb, e.host_buf, e.x.buf, 1, &cp);
            VK_CHECK(vkEndCommandBuffer(ucb));
            submit_wait_sig(e.c, (e.c.tfi != 0xFFFFFFFFu) ? e.c.qtx : e.c.queue, ucb,
                            e.up_fence, VK_NULL_HANDLE, e.up_sig);
        }
        // xsum -> gemv -> reduce, one command buffer, barriers between every pair
        {
            VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
            uint32_t pc[8];
            memset(pc, 0, sizeof pc); pc[0] = 0; pc[1] = 0; pc[2] = nblk;         // xsum
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.pp3[0]);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.pl3[0], 0, 1, &e.ds3[0], 0, nullptr);
            vkCmdPushConstants(cb, e.pl3[0], VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
            vkCmdDispatch(cb, nblk, 1, 1);
            bar_compute(cb);
            memset(pc, 0, sizeof pc); pc[0] = M; pc[1] = K; pc[2] = S;            // gemv
            pc[3] = uint32_t(w->codes_word0); pc[4] = uint32_t(w->scale_word0); pc[5] = 0;
            vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.pp3[1]);
            vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.pl3[1], 0, 1, &e.ds3[1], 0, nullptr);
            vkCmdPushConstants(cb, e.pl3[1], VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
            vkCmdDispatch(cb, (M + 63u) / 64u, S, 1);
            bar_compute(cb);
            if (S > 1) {
                memset(pc, 0, sizeof pc); pc[0] = M; pc[1] = S; pc[2] = 0;        // reduce
                vkCmdBindPipeline(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.pp3[2]);
                vkCmdBindDescriptorSets(cb, VK_PIPELINE_BIND_POINT_COMPUTE, e.pl3[2], 0, 1, &e.ds3[2], 0, nullptr);
                vkCmdPushConstants(cb, e.pl3[2], VK_SHADER_STAGE_COMPUTE_BIT, 0, 32, pc);
                vkCmdDispatch(cb, (M + 63u) / 64u, 1, 1);
                bar_compute(cb);
            }
            VK_CHECK(vkEndCommandBuffer(cb));
            submit_wait_sig(e.c, e.c.queue, cb, VK_NULL_HANDLE, e.up_sig, VK_NULL_HANDLE);
            VK_CHECK(vkQueueWaitIdle(e.c.queue));
        }
        // read the whole output back (it is at most 248320 floats, and only for the biggest M)
        std::vector<float> hout(M);   // parens, not braces, not (size_t(M)) - the vexing parse
        {
            VkCommandBuffer cb = begin_cmd(e.c, e.cpool);
            VkBufferCopy cp{0, 0, uint64_t(M) * 4u};
            vkCmdCopyBuffer(cb, e.out.buf, e.host_buf, 1, &cp);
            VK_CHECK(vkEndCommandBuffer(cb));
            submit_wait_sig(e.c, e.c.queue, cb, e.up_fence, VK_NULL_HANDLE, VK_NULL_HANDLE);
        }
        memcpy(hout.data(), e.host_ptr, size_t(M) * 4u);

        // CPU reference for a handful of rows: dequantize this tensor's blocks from the FILE with
        // the same function the engine's own repack is validated against, dot in float64.
        std::vector<uint8_t> blk(PTQ_BLOCK_BYTES);
        float wf[128];
        for (int t = 0; t < n_rows; ++t) {
            uint32_t row = uint32_t((uint64_t(rnd()) * M) >> 32);
            double acc = 0;
            for (uint32_t b = 0; b < nblk; ++b) {
                const uint64_t pos = e.plan.data_start + w->file_off + (uint64_t(row) * nblk + b) * PTQ_BLOCK_BYTES;
                if (!bonsai_seek(e.f, pos) ||
                    fread(blk.data(), 1, PTQ_BLOCK_BYTES, e.f) != PTQ_BLOCK_BYTES) {
                    fprintf(stderr, "verify: cannot read %s block %u\n", w->name.c_str(), b);
                    return false;
                }
                ptq_dequant_ours(blk.data(), wf);
                for (uint32_t k = 0; k < 128; ++k) acc += double(wf[k]) * double(hx[b * 128u + k]);
            }
            href[row] = float(acc);
            const double a = double(hout[row]), r = acc;
            const double adiff = std::fabs(a - r);
            const double rel =   adiff / (std::fabs(r) > 1e-30 ? std::fabs(r) : 1.0);
            if (rel > worst) { worst = rel; worst_row = int(row); worst_name = w->name; }
            compared++;
            printf("  %-26s row %-7d gpu=%14.6f ref=%14.6f  abs=%.2e rel=%.2e\n",
                   w->name.c_str(), row, a, r, adiff, rel);
            (void)rel;
        }
        printf("  %-26s M=%u K=%u splits=%u\n", w->name.c_str(), M, K, S);
    }
    printf("\nverify: %llu rows compared, worst rel = %.3e (%s row %d)\n",
           (unsigned long long)compared, worst, worst_name.c_str(), worst_row);
    return worst < 1e-5;
}

// ------------------------------------------------------------------ the engine entry point
// bonsai_vk.exe gen <gguf> [chunk_MiB] [noverify]
// This is stage one of the heart: the whole 8.95 B-parameter checkpoint resident in 4 GB of
// device-local memory, in the layout the measured GEMV reads, verified row by row against the CPU
// dequant. The forward passes ride on top of it.
static int weights_main(int argc, char **argv) {
    const char *path = argc > 2 ? argv[2] : "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf";
    const uint64_t chunk_mib = argc > 3 ? uint64_t(atoll(argv[3])) : 32u;
    const bool verify = !(argc > 4 && !strcmp(argv[4], "noverify"));
    if (chunk_mib < 1 || chunk_mib > 512) { fprintf(stderr, "chunk must be 1..512 MiB\n"); return 1; }

    Engine e;
    auto t0 = std::chrono::steady_clock::now();
    if (!engine_init(e, path, chunk_mib << 20)) return 2;
    auto t1 = std::chrono::steady_clock::now();
    if (!load_ptq_weights(e)) return 3;
    if (!load_f32_weights(e)) return 3;
    auto t2 = std::chrono::steady_clock::now();
    load_report(e);
    double ms_setup = std::chrono::duration<double, std::milli>(t1 - t0).count();
    double ms_load  = std::chrono::duration<double, std::milli>(t2 - t1).count();
    printf("\n  plan+arena+pipelines %.1f ms   weight load+repack %.1f ms   TOTAL %.1f ms\n",
           ms_setup, ms_load, ms_setup + ms_load);
    printf("  the official fork took 16050 ms cold / 3680 ms warm for the same file; this is the\n"
           "  pure-blood engine's own cold path, disk -> SDMA -> repack in VRAM, measured above.\n");
    printf("  device-local footprint of the weights alone: %.3f GiB\n",
           double(e.arena.size) / 1073741824.0);

    bool ok = true;
    printf("\n=== byte-level layout proof (codes + scale read back from the arena) ===\n");
    int layout_bad = 0;
    // rows 0 / M/2 / M-1: the middle one is the row range that only exists because a big tensor is
    // split across scratch batches, which is exactly what the first real load got wrong.
    const char *lay[] = {"output.weight", "token_embd.weight", "blk.0.attn_qkv.weight",
                         "blk.0.ffn_gate.weight", "blk.31.ffn_down.weight", "blk.7.attn_k.weight"};
    for (auto n : lay)
        if (const WeightPlan *p = e.plan.find(n)) {
            layout_bad += verify_block_layout(e, *p, 0, 0) > 0;
            layout_bad += verify_block_layout(e, *p, p->M / 2u, 3u) > 0;
            layout_bad += verify_block_layout(e, *p, p->M - 1u, uint32_t(p->n_blk) - 1u) > 0;
        }
    printf("  layout mismatches: %d\n", layout_bad);
    printf("\n=== f32 blob byte proof (head + tail of five representative tensors) ===\n");
    const char *f32n[] = {"output_norm.weight", "blk.0.attn_norm.weight", "blk.0.ssm_alpha.weight",
                          "blk.0.ssm_conv1d.weight", "blk.0.ssm_dt.bias", "blk.31.attn_q_norm.weight"};
    int f32_bad = verify_f32_blob(e, f32n, int(sizeof f32n / sizeof f32n[0]));
    printf("  f32 mismatches: %d\n", f32_bad);
    if (verify) {
        printf("\n=== engine GEMV on REAL weights vs CPU reference (ptq_decode, float64) ===\n");
        ok = verify_weights(e, 8, 2);
    }
    return (ok && layout_bad == 0 && f32_bad == 0) ? 0 : 4;
}

#endif  // BONSAI_WEIGHTS_H
