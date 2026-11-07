// gpu_fleet.h - the multi-GPU / iGPU machinery for the ternary runtime (task 3, ternary-only).
//
// WHAT THIS IS
//   Header-only, dependency-free C++17 + raw Vulkan. It is the reusable half of two architectural
//   capabilities that this host CANNOT exercise:
//     (A) LAYER-PIPELINED MULTI-GPU ("--split-mode layer"): a 2..N card array of e-waste cards,
//         hidden state crossing between cards through a PINNED host ring with async DMA on each
//         card's transfer-only queue family.
//     (B) IGPU ZERO-COPY: when the chosen device is VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU, the
//         weight arena itself is forced onto a heap that is HOST_VISIBLE | DEVICE_LOCAL at the same
//         time, mapped once, and the CPU writes the weights straight into device-visible memory.
//         No staging buffer, no per-upload H2D copy. If no such memory type exists the allocator
//         REFUSES - prints the memory-type table it saw and exits nonzero - because a silent
//         fallback to "allocate device-local, copy it in" is exactly the thing that would make an
//         iGPU build look like it worked while paying the discrete path's PCIe cost twice.
//
// WHAT IS AND IS NOT A MEASUREMENT HERE (read this before citing anything below in a report)
//   Every constant in this file is tagged MEASURED (produced by a run in ../runs or in a *_measured
//   doc, on THIS card) or DERIVED (arithmetic on a MEASURED number, with the derivation written
//   next to it). Nothing else. Two examples of the discipline:
//     MEASURED  device-local heap 3840 MiB on a 4096 MiB RX 550X (STAGE3_RUNTIME §7) -> the AMD
//               driver's reservation is 256 MiB; applied to another card that is an ASSUMPTION and
//               is printed as one.
//     MEASURED  7.12 GB/s SDMA H2D, 7.15-7.18 GB/s D2H, 6.2 us engine for a 32 KiB copy,
//               81.9 us for an empty submit+waitIdle, 99.4 us for a submit+fence round trip
//               (STAGE3_RUNTIME §7, DEEPINFER_EVAL §0). So a 16 KiB hidden-state hop is
//     DERIVED   6.2 us x 16384/32768 = 3.1 us engine + ~99 us fence round trip: the handoff is
//               dominated by the SYNC, not by the bytes. That single fact shapes the whole design
//               and it is why the ring carries a back-pressure depth instead of a per-hop flag.
//
// THE ONE STRUCTURAL WARNING, stated here because it is easy to lose: a transformer's layer chain is
//   SEQUENTIAL for one token. Splitting layers across cards therefore buys CAPACITY (a 9B that does
//   not fit one card fits two) and it does NOT, by itself, buy speed. The depth-D ring earns its
//   keep only when there are >=2 microbatches in flight - then "card k computes microbatch i+1's
//   range" genuinely overlaps "DMA microbatch i's boundary hidden state to card k+1", which is the
//   bubble the BONSAI_PIPE measurements found on one card (6.2-6.7 ms -> 1.13 ms at depth 2,
//   END_TO_END_MEASURED §2.2). With one microbatch the handoff is pure added cost. The scheduler
//   below takes `n_micro` and prints the bubble ratio for the value you gave it; it does not let a
//   caller claim an overlap it did not schedule.
//
// SYNCHRONISATION: timeline-free, on purpose. VkTimelineSemaphore is 1.2; this host driver may only
//   hand out 1.1 entry points, and there is a RECORDED amdvlk SIGSEGV from jumping a NULL
//   vkResetQueryPool that was obtained through vkGetDeviceProcAddr under a 1.1 instance
//   (STAGE3_RUNTIME §7 "two driver traps", END_TO_END_MEASURED §2.3 - measured: the device entry
//   point is NULL with a 1.1 instance, resolved with 1.2). RULE ENFORCED THROUGHOUT THIS FILE:
//   every function pointer that comes from vkGetDeviceProcAddr or vkGetInstanceProcAddr is
//   NULL-CHECKED before it is called, and the fallback path is printed rather than assumed.
//   Cross-CARD ordering is host-mediated (VkFence per in-flight stage + an acquire/release cursor on
//   the ring), because cross-device semaphore signalling needs VK_KHR_external_semaphore and a
//   handle type both drivers agree on; the policy enum says which options exist and which are
//   declared but NOT IMPLEMENTED.
//
// LINKAGE: everything is `static inline` rather than plain `static` on purpose - a header full of
//   plain static functions that some TU does not call produces a -Wunused-function wall, and
//   -Wall is on deliberately in this project (build_mingw.bat: "the two most vexing parse warnings
//   in this file were real bugs").

#ifndef BONSAI_GPU_FLEET_H
#define BONSAI_GPU_FLEET_H

#include <vulkan/vulkan.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#define FK_CHECK(x)                                                                        \
    do {                                                                                   \
        VkResult _r = (x);                                                                 \
        if (_r != VK_SUCCESS) {                                                            \
            fprintf(stderr, "fleet Vulkan failure %d at %s:%d\n", (int)_r, __FILE__,       \
                    __LINE__);                                                             \
            exit(1);                                                                       \
        }                                                                                  \
    } while (0)

// ------------------------------------------------------------------ ground-truth constants
// MEASURED on this host unless tagged otherwise. Do not "round" these: they are the denominators
// the cost model in the report is built from.
static const uint64_t FLEET_MEASURED_CARD_NOMINAL_MIB   = 4096ull;  // RX 550X, nominal
static const uint64_t FLEET_MEASURED_DEVICE_LOCAL_MIB   = 3840ull;  // the heap the driver reports
static const uint64_t FLEET_DRIVER_RESERVATION_MIB      = 256ull;   // DERIVED: 4096 - 3840
static const double   FLEET_MEASURED_SDMA_GBPS_H2D      = 7.12;     // MEASURED, >=1 MiB
static const double   FLEET_MEASURED_SDMA_GBPS_D2H      = 7.15;     // MEASURED, >=1 MiB
static const double   FLEET_MEASURED_COPY32K_ENGINE_US  = 6.2;      // MEASURED, 32 KiB, engine
static const double   FLEET_MEASURED_SUBMIT_FENCE_US    = 99.4;     // MEASURED round trip
// The payload one block boundary produces: hidden 4096 floats, RES region, 4096*4 B.
// MEASURED from the checkpoint's own metadata (qwen35.embedding_length = 4096) and from
// bonsai_forward.h Regions::RES = R.take(4096).
static const uint64_t FLEET_HIDDEN_STATE_BYTES          = 16384ull;

// ------------------------------------------------------------------ tiny helpers
static inline uint64_t fleet_align_up(uint64_t a, uint64_t b) { return (a + b - 1) / b * b; }

static inline const char *fleet_type_name(VkPhysicalDeviceType t) {
    switch (t) {
        case VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU: return "INTEGRATED_GPU";
        case VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU:   return "DISCRETE_GPU";
        case VK_PHYSICAL_DEVICE_TYPE_VIRTUAL_GPU:    return "VIRTUAL_GPU";
        case VK_PHYSICAL_DEVICE_TYPE_CPU:            return "CPU";
        case VK_PHYSICAL_DEVICE_TYPE_OTHER:          return "OTHER";
        default:                                     return "?";
    }
}

// The property string in the same order the existing prints use, so a fleet log and a bonsai_vk log
// can be read side by side. Every bit named here exists in THIS header set (vulkan_core.h:2559-2570
// lists exactly DEVICE_LOCAL, HOST_VISIBLE, HOST_COHERENT, HOST_CACHED, LAZILY_ALLOCATED, PROTECTED,
// DEVICE_COHERENT_AMD, DEVICE_UNCACHED_AMD, RDMA_CAPABLE_NV).
// The task asks for VK_MEMORY_PROPERTY_ACCESS_INTEGRATED_MEMORY_BIT; that name is NOT in Vulkan
// headers 1.4.309 (`grep -n INTEGRATED vulkan_core.h` returns only the physical-device TYPE enum),
// so it is not used and not faked. Its meaning - "this allocation is the same DRAM the integrated
// engine sees" - is recoverable from data that does exist, and fleet_heap_is_unified() below is that
// recovery. Nothing in this file invents a flag value.
static inline void fleet_flags_str(VkMemoryPropertyFlags f, char *out, size_t n) {
    snprintf(out, n, "%s%s%s%s%s%s%s%s%s",
             (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)       ? "DEVICE_LOCAL " : "",
             (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)       ? "HOST_VISIBLE " : "",
             (f & VK_MEMORY_PROPERTY_HOST_COHERENT_BIT)      ? "HOST_COHERENT " : "",
             (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)        ? "HOST_CACHED " : "",
             (f & VK_MEMORY_PROPERTY_LAZILY_ALLOCATED_BIT)   ? "LAZILY_ALLOCATED " : "",
             (f & VK_MEMORY_PROPERTY_PROTECTED_BIT)          ? "PROTECTED " : "",
             (f & VK_MEMORY_PROPERTY_DEVICE_COHERENT_BIT_AMD)? "DEVICE_COHERENT_AMD " : "",
             (f & VK_MEMORY_PROPERTY_DEVICE_UNCACHED_BIT_AMD)? "DEVICE_UNCACHED_AMD " : "",
             (f & VK_MEMORY_PROPERTY_RDMA_CAPABLE_BIT_NV)    ? "RDMA_CAPABLE " : "");
}

// "Is this heap one DRAM that both the CPU and the device address?" The observable signature, using
// only VkPhysicalDeviceMemoryProperties: the heap carries a type that is DEVICE_LOCAL *and*
// HOST_VISIBLE at once, and it is also the heap the plain host-visible types live on. On this host's
// RX 550X that is false by construction - its DEVICE_LOCAL heap has no host-visible type at all,
// which is exactly why the zero-copy branch must refuse here rather than "fall back" (MEASURED:
// device-local heap 3840 MiB on a dedicated GDDR5 frame buffer). On an APU with one shared heap both
// halves hold, and that is what makes the arena map writable and the ring alias free.
static inline bool fleet_heap_is_unified(const VkPhysicalDeviceMemoryProperties &mp, uint32_t hi) {
    bool dev_and_host = false, host_only = false;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        if (mp.memoryTypes[i].heapIndex != hi) continue;
        const VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        const bool dl = (f & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) != 0;
        const bool hv = (f & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        if (dl && hv) dev_and_host = true;
        if (hv && !dl) host_only = true;
    }
    return dev_and_host && host_only;
}

static inline uint64_t fleet_max_unified_heap(const VkPhysicalDeviceMemoryProperties &mp) {
    uint64_t best = 0;
    for (uint32_t h = 0; h < mp.memoryHeapCount; ++h)
        if (fleet_heap_is_unified(mp, h)) best = std::max(best, mp.memoryHeaps[h].size);
    return best;
}

static inline void fleet_queue_flags_str(VkQueueFlags f, char *out, size_t n) {
    snprintf(out, n, "%s%s%s%s%s",
             (f & VK_QUEUE_COMPUTE_BIT)         ? "COMPUTE " : "",
             (f & VK_QUEUE_TRANSFER_BIT)        ? "TRANSFER " : "",
             (f & VK_QUEUE_GRAPHICS_BIT)        ? "GRAPHICS " : "",
             (f & VK_QUEUE_SPARSE_BINDING_BIT)  ? "SPARSE " : "",
             (f & VK_QUEUE_PROTECTED_BIT)       ? "PROTECTED " : "");
}

// ------------------------------------------------------------------ memory-type decisions
// The single authority for "which memory type does the arena get". Two answers, and the discrete
// one is worded so it reproduces bonsai_vk.cpp's ORIGINAL search exactly (first type that merely
// HAS the DEVICE_LOCAL bit, no other requirement) - the discrete path must not change, because
// ~30 measurement logs in ../runs are computed against it.
enum FleetArenaMode {
    FLEET_ARENA_DEVICE_LOCAL = 0,   // what every existing log ran on
    FLEET_ARENA_ZERO_COPY    = 1    // iGPU: HOST_VISIBLE | DEVICE_LOCAL | HOST_COHERENT at once
};

struct FleetArenaPick {
    int  type = -1;                 // -1 = refused
    uint32_t need = 0, prefer = 0;
    bool shared_heap = false;       // the chosen type sits on a heap the CPU also addresses
    bool cached = false;
    const char *why = "";
};

static inline void fleet_print_memory_types(const VkPhysicalDeviceMemoryProperties &mp,
                                            const char *heading) {
    printf("  %s\n", heading);
    char fs[160];
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        fleet_flags_str(f, fs, sizeof fs);
        printf("    %2u heap%-2u %-64s (%llu MiB heap)\n", i, mp.memoryTypes[i].heapIndex, fs,
               (unsigned long long)(mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size / 1048576ull));
    }
}

// mode == FLEET_ARENA_DEVICE_LOCAL: identical predicate to bonsai_vk.cpp:331-335 (first type with
// DEVICE_LOCAL). mode == ZERO_COPY: the mask must be satisfied SIMULTANEOUSLY, which is only
// possible on a heap that is both host- and device-accessible - an iGPU's shared DRAM, or a
// driver that exposes PCIe BAR memory. Among the candidates the score prefers, in order,
// (a) a unified heap (fleet_heap_is_unified: the CPU and the display engine address the same DRAM),
// (b) HOST_CACHED, because a cached mapping is the difference between a 6.28 ms and a 24.8 ms pass
// over the same bytes (MEASURED: scanning logits through a HOST_COHERENT WC mapping cost
// 24.8 ms/token vs 6.28 + 0.14 ms for a memcpy into normal memory first, END_TO_END_MEASURED §2.2).
static inline FleetArenaPick fleet_arena_decide(const VkPhysicalDeviceMemoryProperties &mp,
                                                FleetArenaMode mode) {
    FleetArenaPick p;
    if (mode == FLEET_ARENA_DEVICE_LOCAL) {
        p.need  = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        p.why   = "discrete branch: first type carrying DEVICE_LOCAL, exactly the predicate every "
                  "existing runs/ log was produced with";
        for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
            if (mp.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
                p.type = int(i);
                p.shared_heap = fleet_heap_is_unified(mp, mp.memoryTypes[i].heapIndex);
                p.cached = (mp.memoryTypes[i].propertyFlags &
                            VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
                return p;
            }
        return p;
    }
    p.need = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT | VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
             VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    p.prefer = VK_MEMORY_PROPERTY_HOST_CACHED_BIT;
    p.why = "zero-copy branch: the arena must be host-writeable AND device-local in one allocation, "
            "so the weights cross the link once as stores instead of once as stores plus once as an "
            "SDMA copy";
    int best = -1; uint32_t best_score = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = mp.memoryTypes[i].propertyFlags;
        if ((f & p.need) != p.need) continue;
        uint32_t score = 0;
        if (fleet_heap_is_unified(mp, mp.memoryTypes[i].heapIndex)) score += 4u;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)                 score += 2u;
        if (best < 0 || score > best_score) { best = int(i); best_score = score; }
    }
    p.type = best;
    if (best >= 0) {
        p.shared_heap = fleet_heap_is_unified(mp, mp.memoryTypes[best].heapIndex);
        p.cached = (mp.memoryTypes[best].propertyFlags & VK_MEMORY_PROPERTY_HOST_CACHED_BIT) != 0;
    }
    return p;
}

// The refusal, in one place, so the iGPU path cannot degrade into a silent fallback on any machine.
static inline void fleet_arena_refuse(const VkPhysicalDeviceMemoryProperties &mp,
                                     FleetArenaMode mode, const char *device_name) {
    fleet_print_memory_types(mp, "memory types this driver reported:");
    fprintf(stderr,
            "FLEET REFUSE (%s branch) on \"%s\": no memory type satisfies the mask this branch "
            "needs.\n"
            "  zero-copy requires DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT simultaneously.\n"
            "  Falling back to device-local + an H2D copy would keep the discrete path's PCIe cost\n"
            "  and would make an iGPU run look correct while measuring the wrong thing, so the only\n"
            "  honest answer is to stop here with the table above.\n",
            mode == FLEET_ARENA_ZERO_COPY ? "zero-copy" : "device-local", device_name);
}

// ------------------------------------------------------------------ device classification
struct FleetQueue {
    uint32_t index = 0;
    VkQueueFlags flags = 0;
    uint32_t queue_count = 0;
    uint32_t timestamp_valid_bits = 0;
    VkExtent3D tiler{};
    bool transfer_only = false;      // TRANSFER and not COMPUTE: AMD's SDMA rings
};

struct FleetDevice {
    uint32_t index = 0;
    VkPhysicalDevice h = VK_NULL_HANDLE;
    char name[256] = {0};
    VkPhysicalDeviceType type = VK_PHYSICAL_DEVICE_TYPE_OTHER;
    uint32_t vendor_id = 0, device_id = 0;
    uint32_t api_version = 0, driver_version = 0;

    uint64_t device_local_heap = 0;      // biggest heap with a DEVICE_LOCAL memory type on it
    uint64_t host_visible_heap = 0;
    uint64_t shared_heap = 0;            // biggest heap that is device- AND host-addressable at once
    bool has_zero_copy_type = false;     // DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT at once
    int  zero_copy_type = -1;
    bool has_shared_heap = false;

    std::vector<FleetQueue> queues;
    uint32_t compute_family = 0xFFFFFFFFu;
    uint32_t transfer_only_family = 0xFFFFFFFFu;

    bool ext_external_memory_host = false;   // the only portable way to alias one host pointer
    bool ext_external_memory_win32 = false;  // across two devices on Vulkan 1.1
    bool ext_external_memory_fd = false;
    double timestamp_period_ns = 0.0;
    uint32_t max_mem_allocs = 0;
    // VkPhysicalDeviceLimits has no maxBufferSize in this header set (the only buffer-shaped limits
    // are maxTexelBufferElements / maxUniformBufferRange / maxStorageBufferRange), so what a fleet
    // actually needs instead is recorded: nonCoherentAtomSize, which is the granularity a
    // HOST_NON_COHERENT mapping would have to respect, and the VK_EXT_external_memory_host import
    // alignment, which is the granularity a ring alias has to respect. Both are queried, neither is
    // assumed.
    uint64_t noncoherent_atom_size = 0;
    uint64_t min_imported_host_align = 0;

    // The same hard requirement bonsai_vk.cpp:226 applies, stated as data so `probe` can say why a
    // card was or was not eligible for the existing single-GPU path.
    bool discrete_amd = false;

    VkPhysicalDeviceMemoryProperties mp{};
};

static inline uint64_t fleet_max_heap_with(const VkPhysicalDeviceMemoryProperties &mp,
                                           VkMemoryPropertyFlags f) {
    uint64_t best = 0;
    for (uint32_t i = 0; i < mp.memoryTypeCount; ++i)
        if (mp.memoryTypes[i].propertyFlags & f)
            best = std::max(best, mp.memoryHeaps[mp.memoryTypes[i].heapIndex].size);
    return best;
}

static inline void fleet_classify(VkInstance inst, uint32_t index, VkPhysicalDevice p,
                                  FleetDevice &d) {
    d.index = index;
    d.h = p;
    VkPhysicalDeviceProperties pp;
    vkGetPhysicalDeviceProperties(p, &pp);
    snprintf(d.name, sizeof d.name, "%s", pp.deviceName);
    d.type = pp.deviceType;
    d.vendor_id = pp.vendorID; d.device_id = pp.deviceID;
    d.api_version = pp.apiVersion; d.driver_version = pp.driverVersion;
    d.timestamp_period_ns = pp.limits.timestampPeriod;
    d.max_mem_allocs = pp.limits.maxMemoryAllocationCount;
    d.noncoherent_atom_size = pp.limits.nonCoherentAtomSize;
    vkGetPhysicalDeviceMemoryProperties(p, &d.mp);
    d.device_local_heap = fleet_max_heap_with(d.mp, VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT);
    d.host_visible_heap = fleet_max_heap_with(d.mp, VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT);
    d.shared_heap       = fleet_max_unified_heap(d.mp);
    d.has_shared_heap   = d.shared_heap != 0;
    FleetArenaPick z = fleet_arena_decide(d.mp, FLEET_ARENA_ZERO_COPY);
    d.has_zero_copy_type = z.type >= 0;
    d.zero_copy_type = z.type;
    d.discrete_amd = (d.type == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU && d.vendor_id == 0x1002);

    uint32_t nq = 0;
    vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, nullptr);
    std::vector<VkQueueFamilyProperties> qf(nq);
    vkGetPhysicalDeviceQueueFamilyProperties(p, &nq, qf.data());
    d.queues.clear();
    for (uint32_t i = 0; i < nq; ++i) {
        FleetQueue q;
        q.index = i; q.flags = qf[i].queueFlags; q.queue_count = qf[i].queueCount;
        q.timestamp_valid_bits = qf[i].timestampValidBits;
        q.tiler = qf[i].minImageTransferGranularity;
        q.transfer_only = (qf[i].queueFlags & VK_QUEUE_TRANSFER_BIT) &&
                          !(qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && qf[i].queueCount > 0;
        d.queues.push_back(q);
        if ((qf[i].queueFlags & VK_QUEUE_COMPUTE_BIT) && d.compute_family == 0xFFFFFFFFu)
            d.compute_family = i;
        if (q.transfer_only && d.transfer_only_family == 0xFFFFFFFFu) d.transfer_only_family = i;
    }
    uint32_t ne = 0;
    vkEnumerateDeviceExtensionProperties(p, nullptr, &ne, nullptr);
    std::vector<VkExtensionProperties> ex(ne);
    if (ne) vkEnumerateDeviceExtensionProperties(p, nullptr, &ne, ex.data());
    for (auto &e : ex) {
        // Names spelled as literals where the header guards the macro: VK_KHR_external_memory_win32's
        // EXTENSION_NAME define lives in vulkan_win32.h behind VK_USE_PLATFORM_WIN32_KHR, and this
        // file does not define platform macros it does not need. The strings are the registry's, not
        // invented.
        if (!strcmp(e.extensionName, VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME))
            d.ext_external_memory_host = true;
        if (!strcmp(e.extensionName, "VK_KHR_external_memory_win32"))
            d.ext_external_memory_win32 = true;
        if (!strcmp(e.extensionName, VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME))
            d.ext_external_memory_fd = true;
    }
    // The host-pointer import granularity, if the extension is there to answer. Queried through
    // vkGetPhysicalDeviceProperties2 fetched from the INSTANCE (a 1.1 core entry point) and
    // NULL-CHECKED: this is the exact call pattern behind the recorded amdvlk crash - a function
    // pointer resolved to NULL and jumped to anyway (STAGE3_RUNTIME §7) - so the check is not
    // decoration, and `min_imported_host_align == 0` afterwards means "never asked / not offered",
    // which the caller must treat as "do not assume an import will work".
    if (d.ext_external_memory_host) {
        auto pfn_p2 = reinterpret_cast<PFN_vkGetPhysicalDeviceProperties2>(
            vkGetInstanceProcAddr(inst, "vkGetPhysicalDeviceProperties2"));
        if (pfn_p2) {
            VkPhysicalDeviceExternalMemoryHostPropertiesEXT hp{
                VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_EXTERNAL_MEMORY_HOST_PROPERTIES_EXT};
            VkPhysicalDeviceProperties2 pp2{VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_PROPERTIES_2};
            pp2.pNext = &hp;
            pfn_p2(p, &pp2);
            d.min_imported_host_align = hp.minImportedHostPointerAlignment;
        }
    }
}

static inline VkResult fleet_enumerate(VkInstance inst, std::vector<FleetDevice> &out) {
    uint32_t n = 0;
    VkResult r = vkEnumeratePhysicalDevices(inst, &n, nullptr);
    if (r != VK_SUCCESS && r != VK_INCOMPLETE) return r;
    std::vector<VkPhysicalDevice> ps(n);
    if (n) vkEnumeratePhysicalDevices(inst, &n, ps.data());
    out.clear();
    for (uint32_t i = 0; i < n; ++i) {
        FleetDevice d;
        fleet_classify(inst, i, ps[i], d);
        out.push_back(d);
    }
    return VK_SUCCESS;
}

// ------------------------------------------------------------------ per-device context
// One VkDevice per card. bonsai_vk.cpp's Ctx stays as it is (the discrete path must not move); this
// struct exists for fleet drivers, where N of them are alive at once and each owns its arena, its
// two queues and its command pools.
struct FleetCtx {
    VkInstance inst = VK_NULL_HANDLE;
    FleetDevice d;
    VkDevice dev = VK_NULL_HANDLE;
    VkQueue queue = VK_NULL_HANDLE;      // compute
    VkQueue qtx = VK_NULL_HANDLE;        // transfer-only, VK_NULL_HANDLE when there is none
    uint32_t qfi = 0, tfi = 0xFFFFFFFFu;
    bool has_tx = false;
    VkPhysicalDeviceMemoryProperties mp{};
    VkPhysicalDeviceLimits limits{};

    VkCommandPool cpool = VK_NULL_HANDLE;   // compute family, RESET_COMMAND_BUFFER_BIT
    VkCommandPool upool = VK_NULL_HANDLE;   // the queue DMA runs on (tfi if present, else qfi)

    // arena, zero-copy aware
    VkDeviceMemory arena = VK_NULL_HANDLE;
    void *arena_map = nullptr;             // non-NULL only on the zero-copy branch
    uint32_t arena_type = 0;
    bool zero_copy = false;
    uint64_t arena_bytes = 0;
};

static inline void fleet_ctx_release(FleetCtx &c) {
    if (!c.dev) return;
    if (c.arena_map && c.arena) vkUnmapMemory(c.dev, c.arena);
    c.arena_map = nullptr;
    if (c.upool) vkDestroyCommandPool(c.dev, c.upool, nullptr);
    if (c.cpool) vkDestroyCommandPool(c.dev, c.cpool, nullptr);
    if (c.arena) vkFreeMemory(c.dev, c.arena, nullptr);
    c.arena = VK_NULL_HANDLE;
    vkDeviceWaitIdle(c.dev);
    vkDestroyDevice(c.dev, nullptr);
    c.dev = VK_NULL_HANDLE;
}

// Creates the logical device over {compute family, transfer-only family}, the same two-family
// shape bonsai_vk.cpp:264-278 uses. `zero_copy` selects the arena memory type only - the queue
// choice is identical either way, because a fleet DMA needs the SDMA ring on an iGPU too.
static inline bool fleet_ctx_init(FleetCtx &c, VkInstance inst, const FleetDevice &d,
                                  bool zero_copy) {
    c.inst = inst; c.d = d; c.zero_copy = zero_copy;
    vkGetPhysicalDeviceMemoryProperties(d.h, &c.mp);
    VkPhysicalDeviceProperties pp; vkGetPhysicalDeviceProperties(d.h, &pp);
    c.limits = pp.limits;
    c.qfi = d.compute_family; c.tfi = d.transfer_only_family;
    c.has_tx = (c.tfi != 0xFFFFFFFFu);
    if (c.qfi == 0xFFFFFFFFu) {
        fprintf(stderr, "fleet: \"%s\" has no COMPUTE-capable queue family - nothing can run.\n",
                d.name);
        return false;
    }
    float prio = 1.0f;
    VkDeviceQueueCreateInfo qi[2] = {};
    qi[0].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
    qi[0].queueFamilyIndex = c.qfi; qi[0].queueCount = 1; qi[0].pQueuePriorities = &prio;
    uint32_t nqi = 1;
    std::vector<const char *> en;
    if (c.has_tx) {
        qi[1].sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qi[1].queueFamilyIndex = c.tfi; qi[1].queueCount = 1; qi[1].pQueuePriorities = &prio;
        nqi = 2;
    }
    // VK_EXT_external_memory_host is only enabled when it is actually advertised: asking for a
    // missing extension is a vkCreateDevice failure, and on a fleet tool that failure would look
    // like "the second card is broken" rather than "this driver does not alias host pointers".
    if (d.ext_external_memory_host) {
        en.push_back(VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME);
    }
    VkDeviceCreateInfo di{VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO};
    di.queueCreateInfoCount = nqi; di.pQueueCreateInfos = qi;
    di.enabledExtensionCount = uint32_t(en.size());
    di.ppEnabledExtensionNames = en.empty() ? nullptr : en.data();
    VkResult r = vkCreateDevice(d.h, &di, nullptr, &c.dev);
    if (r != VK_SUCCESS && !en.empty()) {
        printf("  vkCreateDevice with %s failed (%d); retrying without it - the ring then has to "
               "hand bytes over by a host memcpy instead of an import\n",
               VK_EXT_EXTERNAL_MEMORY_HOST_EXTENSION_NAME, (int)r);
        di.enabledExtensionCount = 0; di.ppEnabledExtensionNames = nullptr;
        FK_CHECK(vkCreateDevice(d.h, &di, nullptr, &c.dev));
    } else if (r != VK_SUCCESS) {
        fprintf(stderr, "fleet: vkCreateDevice on \"%s\" failed (%d)\n", d.name, (int)r);
        return false;
    }
    vkGetDeviceQueue(c.dev, c.qfi, 0, &c.queue);
    if (c.has_tx) vkGetDeviceQueue(c.dev, c.tfi, 0, &c.qtx);

    VkCommandPoolCreateInfo pi{VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO};
    pi.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT; pi.queueFamilyIndex = c.qfi;
    FK_CHECK(vkCreateCommandPool(c.dev, &pi, nullptr, &c.cpool));
    pi.queueFamilyIndex = c.has_tx ? c.tfi : c.qfi;
    FK_CHECK(vkCreateCommandPool(c.dev, &pi, nullptr, &c.upool));
    return true;
}

// The arena allocation. In zero-copy mode this REFUSES when fleet_arena_decide() has no answer -
// prints the table first - and never quietly falls back.
static inline bool fleet_arena_alloc(FleetCtx &c, uint64_t bytes, FleetArenaMode mode,
                                     VkBufferUsageFlags usage) {
    FleetArenaPick p = fleet_arena_decide(c.mp, mode);
    if (p.type < 0) {
        fleet_arena_refuse(c.mp, mode, c.d.name);
        return false;
    }
    c.zero_copy = (mode == FLEET_ARENA_ZERO_COPY);
    c.arena_type = uint32_t(p.type);
    c.arena_bytes = bytes;
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = bytes; ma.memoryTypeIndex = c.arena_type;
    VkResult r = vkAllocateMemory(c.dev, &ma, nullptr, &c.arena);
    if (r != VK_SUCCESS) {
        fprintf(stderr, "fleet: vkAllocateMemory(%llu MiB, type %d) on \"%s\" failed (%d)\n",
                (unsigned long long)(bytes >> 20), p.type, c.d.name, (int)r);
        return false;
    }
    printf("  arena type %d  %s%s%s%s  %.1f MiB  [shared-heap=%s cached=%s]  <- %s\n", p.type,
           (p.need & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) ? "DEVICE_LOCAL " : "",
           (p.need & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) ? "HOST_VISIBLE " : "",
           p.shared_heap ? "UNIFIED_HEAP " : "", p.cached ? "HOST_CACHED " : "",
           double(bytes) / 1048576.0, p.shared_heap ? "yes" : "no", p.cached ? "yes" : "no", p.why);
    (void)usage;
    if (c.zero_copy) {
        // One mapping, kept for the life of the arena: the whole point of this branch is that the
        // CPU stores the weights once and no copy engine is involved afterwards.
        void *p_map = nullptr;
        r = vkMapMemory(c.dev, c.arena, 0, bytes, 0, &p_map);
        if (r != VK_SUCCESS) {
            fprintf(stderr, "fleet: vkMapMemory on the zero-copy arena failed (%d) - the type said "
                            "HOST_VISIBLE, so this is a driver bug and the run must not continue "
                            "with a staged upload pretending to be zero-copy.\n", (int)r);
            return false;
        }
        c.arena_map = p_map;
    }
    return true;
}

// ------------------------------------------------------------------ the pinned ring
// Depth D, all slots the same size, one producer (the scheduler thread that submits card k's work)
// and one consumer (the same thread after card k+1's fence). The cursors are HOST-ONLY integers in
// this struct, NOT bytes inside the mapped ring: a GPU never reads them, and putting them in the
// mapping would make every per-boundary check a read through WC memory, which this project has
// already measured at 24.8 ms/token for a scan of 993280 B (END_TO_END_MEASURED §2.2).
//
// The index algebra is deliberately a monotone sequence + mask rather than head/tail modulo, so the
// fullness test is `produced - consumed < D` with no wrap special case, and D must be a power of
// two so the slot select is a shift-free AND. A ring that silently wrapped instead of refusing is
// the same class of bug as the missing bar_compute() in bonsai_vk.cpp:139-148 - the symptom is a
// result that differs between runs of the same binary, because a slot gets overwritten while a
// flight is still reading it. So `ring_reserve()` returns false; it does not evict.
struct PinnedRing {
    uint32_t depth = 0;
    uint32_t mask = 0;
    uint64_t slot_bytes = 0;
    uint64_t total_bytes = 0;

    // storage
    bool host_only = false;            // the offline algebra harness: malloc, no VkDeviceMemory
    VkDeviceMemory mem = VK_NULL_HANDLE;
    VkBuffer view = VK_NULL_HANDLE;    // TRANSFER_SRC|DST view of the SAME allocation
    void *map = nullptr;
    int mem_type = -1;
    bool shared_heap = false;          // that type's heap is one DRAM the CPU also addresses
    bool shared_with = false;          // a second FleetCtx imported this same host pointer
    uint32_t owner = 0xFFFFFFFFu;      // index into the FleetCtx array

    // SPSC cursors. release on publish, acquire on wait: the payload bytes were written by the DMA
    // engine before its completion proof returned - a fence armed by its own submit (flags = 0, see
    // fleet_pipeline_init) or the doorbell word below - and that wait is the host-visible ordering
    // point; the atomics order the scheduler's own view of slot ownership, nothing more.
    std::atomic<uint64_t> produced{0};
    std::atomic<uint64_t> consumed{0};

    // ---- doorbell: a completion proof that does not depend on fence semantics ----
    // Why it is here, with the attribution corrected by the control run. The first version of this
    // rig created its fences VK_FENCE_CREATE_SIGNALED_BIT and reused them as submit -> wait -> reset
    // -> submit. Measured (runs/fleet_ring_v2.log): such a leg on the TRANSFER-ONLY queue returned
    // from vkWaitForFences in 1.0-1.7 us while vkQueueWaitIdle behind it still needed 1199-1255 us.
    // The v3 control (runs/fleet_ring_v3.log) separates the two candidate causes: a FRESH fence,
    // created for that submit, blocks correctly - 1263.6 us of fence with a 0.8 us queue-idle tail
    // behind it. So the driver's SDMA fence is NOT lying; OUR reuse pattern was. Pipeline fences are
    // now created with flags = 0 for that reason, and the created-SIGNALED control still runs in the
    // rig so the claim stays re-measurable instead of becoming folklore.
    // The doorbell is kept anyway, as the proof that needs no assumption about fences at all: the
    // payload copy AND a 4 B vkCmdFillBuffer of the payload number are recorded in the SAME command
    // buffer on the SAME queue, so queue order alone guarantees "bell[slot] == V => payload V landed".
    // The host polls ONE word through the coherent mapping - a scan of a WC mapping was measured at
    // 24.8 ms/token for 993280 B (END_TO_END_MEASURED 2.2), a single-word poll is not that - and the
    // fence is still waited and still timed beside it, so the fence-vs-true-completion gap stays a
    // number in the log rather than a footnote.
    // Stride is 128 B = this card's MEASURED nonCoherentAtomSize, so the line never shares a slot
    // with a neighbour even on a non-coherent type.
    static constexpr uint64_t BELL_STRIDE = 128ull;
    VkBuffer bell = VK_NULL_HANDLE;
    VkDeviceMemory bell_mem = VK_NULL_HANDLE;
    volatile uint32_t *bell_map = nullptr;
    uint64_t bell_bytes = 0;

    // NO synchronisation objects live here. A VkFence or VkSemaphore belongs to the ONE VkDevice it
    // was created on, and arming it from another device's vkQueueSubmit is undefined - the first
    // draft of this struct kept a single set of fences in the ring and used f_in on the consumer
    // card, which would have passed on a shared-heap APU only by luck and failed on two discrete
    // cards. FleetPipeline owns them, indexed [card][slot].

    // A second card's view of the same bytes, when the import succeeded. Kept so that they are
    // destroyed with the device that created them: leaving a VkDeviceMemory alive across
    // vkDestroyDevice is the kind of "it worked on my machine" that only shows up as a leak.
    struct Attach {
        VkDevice dev = VK_NULL_HANDLE;
        VkBuffer view = VK_NULL_HANDLE;
        VkDeviceMemory mem = VK_NULL_HANDLE;
    };
    std::vector<Attach> attach;

    ~PinnedRing() {}
};

static inline bool fleet_ring_init_geom(PinnedRing &r, uint32_t depth, uint64_t slot_bytes) {
    if (depth < 2 || (depth & (depth - 1))) {
        fprintf(stderr, "fleet ring: depth must be a power of two and >= 2 (got %u). The slot index "
                        "is seq & mask so a non-power-of-two depth would need a division in the "
                        "per-token path.\n", depth);
        return false;
    }
    if (slot_bytes == 0) { fprintf(stderr, "fleet ring: slot_bytes must be > 0\n"); return false; }
    r.depth = depth; r.mask = depth - 1;
    r.slot_bytes = slot_bytes;
    r.total_bytes = uint64_t(depth) * slot_bytes;
    r.produced = 0; r.consumed = 0;
    return true;
}

// The offline form: the algebra, the refusal-on-full and the wrap are all testable with no driver.
static inline bool fleet_ring_alloc_host(PinnedRing &r, uint32_t depth, uint64_t slot_bytes) {
    if (!fleet_ring_init_geom(r, depth, slot_bytes)) return false;
    r.host_only = true;
    r.map = calloc(1, size_t(r.total_bytes) + 1);
    return r.map != nullptr;
}

static inline void fleet_ring_free(VkDevice dev, PinnedRing &r) {
    if (r.bell_map && dev && r.bell_mem) { vkUnmapMemory(dev, r.bell_mem); r.bell_map = nullptr; }
    if (r.bell && dev) vkDestroyBuffer(dev, r.bell, nullptr);
    r.bell = VK_NULL_HANDLE;
    if (r.bell_mem && dev) vkFreeMemory(dev, r.bell_mem, nullptr);
    r.bell_mem = VK_NULL_HANDLE; r.bell_bytes = 0;
    if (r.view && dev) vkDestroyBuffer(dev, r.view, nullptr);
    r.view = VK_NULL_HANDLE;
    for (auto &a : r.attach) {
        if (a.view) vkDestroyBuffer(a.dev, a.view, nullptr);
        if (a.mem) vkFreeMemory(a.dev, a.mem, nullptr);
    }
    r.attach.clear();
    if (r.host_only) { free(r.map); r.map = nullptr; return; }
    if (r.mem && dev) vkFreeMemory(dev, r.mem, nullptr);
    r.mem = VK_NULL_HANDLE; r.map = nullptr;
}

// The pinned allocation. Preferred memory type: HOST_VISIBLE | HOST_COHERENT, and among those the one
// on a UNIFIED heap (fleet_heap_is_unified), because when both cards' host-visible types are the same
// physical DRAM the cross-card hop needs no PCIe round trip at all. The buffer view is CONCURRENT
// over {compute, transfer} - the same reason bonsai_vk.cpp:293-299 gives: EXCLUSIVE would demand a
// queue-ownership barrier at every hop.
static inline bool fleet_ring_alloc_device(PinnedRing &r, FleetCtx &owner, uint32_t depth,
                                           uint64_t slot_bytes) {
    if (!fleet_ring_init_geom(r, depth, slot_bytes)) return false;
    const uint32_t need = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    int best = -1; uint32_t best_score = 0;
    for (uint32_t i = 0; i < owner.mp.memoryTypeCount; ++i) {
        VkMemoryPropertyFlags f = owner.mp.memoryTypes[i].propertyFlags;
        if ((f & need) != need) continue;
        uint32_t score = 0;
        if (fleet_heap_is_unified(owner.mp, owner.mp.memoryTypes[i].heapIndex)) score += 4u;
        if (f & VK_MEMORY_PROPERTY_HOST_CACHED_BIT)                            score += 2u;
        if (best < 0 || score > best_score) { best = int(i); best_score = score; }
    }
    if (best < 0) {
        fleet_print_memory_types(owner.mp, "memory types this driver reported:");
        fprintf(stderr, "fleet ring REFUSE: \"%s\" exposes no HOST_VISIBLE|HOST_COHERENT memory "
                        "type, so there is nothing to pin.\n", owner.d.name);
        return false;
    }
    r.mem_type = best; r.owner = 0;
    r.shared_heap = fleet_heap_is_unified(owner.mp, owner.mp.memoryTypes[best].heapIndex);
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = r.total_bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    // Lifetime rule, applied at every vkCreateBuffer in this file: pQueueFamilyIndices points INTO
    // the caller's memory for the duration of the call, so the array must outlive the block that
    // decides the sharing mode. An in-block array is what segfaulted amdvlk64.dll under -O2 only
    // (see bonsai_fleet.cpp make_arena_buf).
    uint32_t fams[2] = {owner.qfi, owner.tfi};
    if (owner.has_tx) {
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fams;
    } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    FK_CHECK(vkCreateBuffer(owner.dev, &bi, nullptr, &r.view));
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(owner.dev, r.view, &mr);
    if (!((mr.memoryTypeBits >> best) & 1u)) {
        vkDestroyBuffer(owner.dev, r.view, nullptr); r.view = VK_NULL_HANDLE;
        fleet_print_memory_types(owner.mp, "memory types this driver reported:");
        fprintf(stderr, "fleet ring REFUSE: the pinned type %d is not in the ring buffer's "
                        "memoryTypeBits 0x%X.\n", best, mr.memoryTypeBits);
        return false;
    }
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = std::max(mr.size, r.total_bytes); ma.memoryTypeIndex = uint32_t(best);
    FK_CHECK(vkAllocateMemory(owner.dev, &ma, nullptr, &r.mem));
    FK_CHECK(vkBindBufferMemory(owner.dev, r.view, r.mem, 0));
    FK_CHECK(vkMapMemory(owner.dev, r.mem, 0, r.total_bytes, 0, &r.map));
    r.host_only = false;

    // ---- the doorbell (see the PinnedRing comment: the trap was our created-SIGNALED fence reuse,
    // proven by the v3 fresh-fence control; this is the fence-free proof). Same host-visible coherent
    // type as the ring itself; TRANSFER_DST because vkCmdFillBuffer
    // requires a transfer flag, STORAGE so a real producer shader could also write it.
    r.bell_bytes = uint64_t(depth) * PinnedRing::BELL_STRIDE;
    VkBufferCreateInfo bb{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bb.size = r.bell_bytes;
    bb.usage = VK_BUFFER_USAGE_TRANSFER_DST_BIT | VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    bb.sharingMode = VK_SHARING_MODE_CONCURRENT;
    uint32_t bfam[2] = {owner.qfi, owner.tfi};
    if (owner.has_tx) {
        bb.queueFamilyIndexCount = 2; bb.pQueueFamilyIndices = bfam;
    } else {
        bb.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
        bb.queueFamilyIndexCount = 0; bb.pQueueFamilyIndices = nullptr;
    }
    VkResult br = vkCreateBuffer(owner.dev, &bb, nullptr, &r.bell);
    VkMemoryRequirements bmr;
    if (br == VK_SUCCESS) vkGetBufferMemoryRequirements(owner.dev, r.bell, &bmr);
    int btype = -1;
    if (br == VK_SUCCESS && ((bmr.memoryTypeBits >> best) & 1u)) btype = best;
    if (btype < 0 && br == VK_SUCCESS) {
        for (uint32_t i = 0; i < owner.mp.memoryTypeCount; ++i) {
            if (!((bmr.memoryTypeBits >> i) & 1u)) continue;
            VkMemoryPropertyFlags f = owner.mp.memoryTypes[i].propertyFlags;
            if ((f & need) != need) continue;
            btype = int(i); break;
        }
    }
    if (btype >= 0) {
        VkMemoryAllocateInfo bma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
        bma.allocationSize = std::max(bmr.size, r.bell_bytes);
        bma.memoryTypeIndex = uint32_t(btype);
        if (vkAllocateMemory(owner.dev, &bma, nullptr, &r.bell_mem) == VK_SUCCESS &&
            vkBindBufferMemory(owner.dev, r.bell, r.bell_mem, 0) == VK_SUCCESS &&
            vkMapMemory(owner.dev, r.bell_mem, 0, r.bell_bytes, 0,
                        (void **)&r.bell_map) == VK_SUCCESS) {
            for (uint32_t s = 0; s < depth; ++s)
                r.bell_map[s * (PinnedRing::BELL_STRIDE / 4)] = 0u;   // 0 = no payload landed yet
        } else {
            r.bell_map = nullptr;
        }
    }
    if (!r.bell_map) {
        if (r.bell) { vkDestroyBuffer(owner.dev, r.bell, nullptr); r.bell = VK_NULL_HANDLE; }
        if (r.bell_mem) { vkFreeMemory(owner.dev, r.bell_mem, nullptr); r.bell_mem = VK_NULL_HANDLE; }
        r.bell_bytes = 0;
        printf("  doorbell: NOT AVAILABLE (create/bind/map failed or no host-visible type backs it)."
               " The rig falls back to fence + vkQueueWaitIdle and says so at every hop - which on"
               " this driver means paying a full queue drain per hop instead of a word poll.\n");
    }
    return true;
}

// The second card's view of the SAME bytes. On Vulkan 1.1 the only portable way is
// VK_EXT_external_memory_host: hand the driver the host pointer we already have.
// NULL-CHECK DISCIPLINE: vkGetMemoryHostPointerPropertiesEXT is fetched with
// vkGetInstanceProcAddr and may legitimately be NULL (the extension is not core, and under a 1.1
// instance vkGetDeviceProcAddr has been MEASURED to return NULL for functions that do exist - that
// is the exact pointer whose jump-to-0 took down amdvlk64.dll, STAGE3_RUNTIME §7). Calling it
// without the check would reproduce that crash on purpose.
static inline bool fleet_ring_attach(PinnedRing &r, FleetCtx &other,
                                     VkInstance inst, uint32_t other_index) {
    if (!other.d.ext_external_memory_host || !r.map) {
        printf("  card %u (%s): cannot import the ring's host pointer - %s. The hop then costs one "
               "host memcpy per boundary instead of an alias.\n", other_index, other.d.name,
               other.d.ext_external_memory_host ? "no mapped host pointer"
                                                : "no VK_EXT_external_memory_host");
        return false;
    }
    auto pfn_props = reinterpret_cast<PFN_vkGetMemoryHostPointerPropertiesEXT>(
        vkGetInstanceProcAddr(inst, "vkGetMemoryHostPointerPropertiesEXT"));
    if (!pfn_props) {   // advertised but unresolved: the NULL-check that the amdvlk crash demands
        printf("  card %u: vkGetInstanceProcAddr(\"vkGetMemoryHostPointerPropertiesEXT\") -> NULL; "
               "not called. Import skipped.\n", other_index);
        return false;
    }
    VkMemoryHostPointerPropertiesEXT hp{VK_STRUCTURE_TYPE_MEMORY_HOST_POINTER_PROPERTIES_EXT};
    if (pfn_props(other.dev, VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT, r.map, &hp)
        != VK_SUCCESS) {
        printf("  card %u: host pointer not importable (granularity query failed).\n", other_index);
        return false;
    }
    // The alignment the driver demands is not negotiable, and vkMapMemory does not promise any
    // particular one, so this is checked rather than hoped for: an unaligned host pointer makes
    // vkAllocateMemory return an error that reads like "the second card is broken".
    // `minImportedHostPointerAlignment` is a property of the PHYSICAL device
    // (VkPhysicalDeviceExternalMemoryHostPropertiesEXT, chained into
    // vkGetPhysicalDeviceProperties2) - it is NOT a field of VkMemoryHostPointerPropertiesEXT, which
    // only answers memoryTypeBits. The value was queried once at classify time and NULL-checks
    // behind it are documented there; 0 here means the query never ran, which is treated as
    // "unknown, do not import" instead of "no requirement".
    const uint64_t align = other.d.min_imported_host_align;
    if (align == 0) {
        printf("  card %u: minImportedHostPointerAlignment unknown (properties2 query did not "
               "run) - import skipped rather than guessed.\n", other_index);
        return false;
    }
    if (align > 1 && (reinterpret_cast<uintptr_t>(r.map) % uintptr_t(align))) {
        printf("  card %u: ring mapping %p is not a multiple of the driver's min import alignment "
               "%llu B - import skipped, the hop falls back to a host memcpy.\n", other_index,
               r.map, (unsigned long long)align);
        return false;
    }
    VkImportMemoryHostPointerInfoEXT imp{VK_STRUCTURE_TYPE_IMPORT_MEMORY_HOST_POINTER_INFO_EXT};
    imp.handleType = VK_EXTERNAL_MEMORY_HANDLE_TYPE_HOST_ALLOCATION_BIT_EXT;
    imp.pHostPointer = r.map;
    // hp.memoryTypeBits is the driver's own answer to "which of MY types may hold this host
    // pointer"; using it is what makes the import correct rather than merely plausible, and it also
    // kills the wrong shortcut of "pick any HOST_VISIBLE type on the second card", which can name a
    // type the import does not support.
    int type = -1;
    for (uint32_t i = 0; i < other.mp.memoryTypeCount; ++i) {
        if (!((hp.memoryTypeBits >> i) & 1u)) continue;
        const uint32_t need = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                              VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
        if ((other.mp.memoryTypes[i].propertyFlags & need) != need) continue;
        // If the producer's ring lives on a unified heap, the consumer must import it onto the same
        // kind of heap or the "alias" is two different memories wearing one address - the copy is
        // skipped and the data is wrong, which is the worst possible failure mode because every
        // number still prints plausibly.
        if (r.shared_heap && !fleet_heap_is_unified(other.mp, other.mp.memoryTypes[i].heapIndex))
            continue;
        type = int(i); break;
    }
    if (type < 0) {
        printf("  card %u: no type in the import's memoryTypeBits 0x%X is HOST_VISIBLE|"
               "HOST_COHERENT%s - import skipped.\n", other_index, hp.memoryTypeBits,
               r.shared_heap ? " and on a unified heap" : "");
        return false;
    }
    VkBufferCreateInfo bi{VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO};
    bi.size = r.total_bytes;
    bi.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT |
               VK_BUFFER_USAGE_STORAGE_BUFFER_BIT;
    uint32_t fams[2] = {other.qfi, other.tfi};   // same lifetime rule as the ring view above
    if (other.has_tx) {
        bi.sharingMode = VK_SHARING_MODE_CONCURRENT;
        bi.queueFamilyIndexCount = 2; bi.pQueueFamilyIndices = fams;
    } else bi.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
    VkBuffer v;
    if (vkCreateBuffer(other.dev, &bi, nullptr, &v) != VK_SUCCESS) {
        printf("  card %u: ring buffer creation with export/import usage refused.\n", other_index);
        return false;
    }
    VkMemoryRequirements mr; vkGetBufferMemoryRequirements(other.dev, v, &mr);
    VkMemoryAllocateInfo ma{VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO};
    ma.allocationSize = mr.size; ma.memoryTypeIndex = type; ma.pNext = &imp;
    VkDeviceMemory m = VK_NULL_HANDLE;
    VkResult r2 = vkAllocateMemory(other.dev, &ma, nullptr, &m);
    if (r2 == VK_SUCCESS) r2 = vkBindBufferMemory(other.dev, v, m, 0);
    if (r2 != VK_SUCCESS) {
        printf("  card %u: importing the ring failed (%d) - falling back to the host-memcpy hop.\n",
               other_index, (int)r2);
        vkDestroyBuffer(other.dev, v, nullptr);
        if (m) vkFreeMemory(other.dev, m, nullptr);
        return false;
    }
    r.shared_with = true;
    PinnedRing::Attach a;
    a.dev = other.dev; a.view = v; a.mem = m;
    r.attach.push_back(a);            // so fleet_ring_free can put them back on the right device
    printf("  card %u: imported the ring's host pointer at granularity %llu B - the two cards now "
           "address the same pages, so a hop is a store and a load, not a copy.\n", other_index,
           (unsigned long long)align);
    return true;
}

static inline void *fleet_ring_slot(PinnedRing &r, uint64_t seq) {
    return (char *)r.map + (seq & uint64_t(r.mask)) * r.slot_bytes;
}

// Back-pressure test. False means "D payloads are still in flight; do not overwrite slot seq%D".
static inline bool fleet_ring_reserve(const PinnedRing &r, uint64_t seq) {
    return r.produced.load(std::memory_order_relaxed) -
           r.consumed.load(std::memory_order_acquire) < r.depth;
}

static inline void fleet_ring_publish(PinnedRing &r, uint64_t seq) {
    // release: everything the DMA engine wrote into slot seq%D has to be visible to whoever next
    // acquires produced. The DMA itself was fence-waited before this call, so this store is the
    // last link in the chain, not the only one.
    uint64_t want = seq + 1;
    uint64_t cur = r.produced.load(std::memory_order_relaxed);
    while (want > cur) {
        if (!r.produced.compare_exchange_weak(cur, want, std::memory_order_release,
                                             std::memory_order_relaxed)) break;
    }
}

// The consumer's view: has payload `seq` landed? `consumed` only ever moves in order, because the
// scheduler is one thread and the pipeline drains the ring in submission order.
static inline bool fleet_ring_ready(const PinnedRing &r, uint64_t seq) {
    return r.produced.load(std::memory_order_acquire) > seq;
}

// Poll the doorbell for payload `hseq` of `depth` payloads. True = the copy in front of the stamp has
// retired, so slot bytes are readable. The spin is one volatile word per iteration and the deadline
// is generous on purpose (a 8 MiB leg measures ~1.2 ms here): on timeout the CALLER does the
// vkQueueWaitIdle and the rig counts the timeout, so a driver that never writes the bell shows up as
// a number rather than as a hang or as silence. With no bell (host-only algebra, or the allocation
// refused) this returns true immediately - there is nothing to poll and no GPU to wait for.
static inline bool fleet_ring_wait_bell(const PinnedRing &r, uint64_t hseq, double timeout_ms) {
    if (!r.bell_map) return true;
    volatile uint32_t *w = const_cast<volatile uint32_t *>(r.bell_map) +
                           size_t(hseq & uint64_t(r.mask)) * (PinnedRing::BELL_STRIDE / 4);
    const uint32_t want = uint32_t(hseq & 0xFFFFFFFFull) + 1u;   // 0 = untouched slot
    const auto t0 = std::chrono::steady_clock::now();
    for (;;) {
        if (*w == want) return true;
        const double ms = std::chrono::duration<double, std::micro>(
                              std::chrono::steady_clock::now() - t0).count() / 1000.0;
        if (ms > timeout_ms) return false;
    }
}

static inline void fleet_ring_consume(PinnedRing &r, uint64_t seq) {
    r.consumed.store(seq + 1, std::memory_order_release);
}

// ------------------------------------------------------------------ the layer split
// Contiguous ranges, one per card, in execution order. Contiguous and not round-robin because the
// GDN recurrent state (MEASURED: 50.25 MiB per sequence, DEEPINFER_EVAL §0/§5) lives on the card
// that runs its layer and never crosses; an interleaved split would drag 50 MiB of state over the
// link every token.
struct LayerRange {
    uint32_t card = 0;
    uint32_t layer_lo = 0, layer_hi = 0;    // [lo, hi)
    uint64_t bytes = 0;                      // device-resident weight bytes for the range
    uint64_t budget = 0;                     // what this card may hold
};

struct FleetSplit {
    uint32_t layers = 0, cards = 0;
    uint64_t head_bytes = 0, tail_bytes = 0;   // token_embd / output.weight + output_norm
    std::vector<LayerRange> ranges;
    uint64_t total_bytes = 0, smallest_budget = 0;
    bool ok = false;
    std::string refuse;
    // hops per token = one per CARD TRANSITION, i.e. ranges-1 (a contiguous cut has exactly one
    // boundary between adjacent ranges). This is the number the handoff cost model multiplies, so it
    // is computed here rather than by callers. A rig that puts two ranges on one card (what
    // `bonsai_fleet ring` does to exercise the mechanism on this host) gets the same arithmetic with
    // the card transition simulated instead of real.
    uint32_t hops_per_token() const { return ranges.empty() ? 0u : uint32_t(ranges.size() - 1); }
};

// Balanced by BYTES, then cut at layer boundaries. Contiguous and byte-weighted on purpose: with 24
// GDN and 8 softmax blocks the per-layer counts are NOT equal, so a count-balanced split quietly
// loads the GDN-heavy card more and the card that overflows is the one that decides whether the
// model runs at all.
// WHICH BYTE COUNT (this is where a budget can be wrong by 28%):
//   the caller must pass DEVICE-RESIDENT bytes - the repacked codes+scales (+f32) the arena really
//   holds, which is what the card's budget has to cover. On the 9B, read out of the tensor table with
//   the same formulas bonsai_model.h:1050-1051 uses (M*(K/64)*16 + M*(K/128)*4 per quantized tensor,
//   elems*4 per f32 tensor): 62,554,880 B for a GDN block and 59,017,216 B for a softmax block, and
//   24*62554880 + 8*59017216 = 1,973,454,848 B = 1.838 GiB across the 32 layers.
//   The FILE (upload/staging) bytes are smaller - 48,923,392 / 45,910,016 and 1,541,441,536 total,
//   because PTQ1_0 is 1.75 bpw on disk and 2.25 bpw after repack - and those are the numbers the
//   7.12 GB/s upload cost model uses, not the budget.
// `bonsai_fleet split` prints both columns so a reader can tell which one a refusal was about.
static inline FleetSplit fleet_split_schedule(uint32_t layers, uint32_t cards,
                                              const std::vector<uint64_t> &bytes_per_layer,
                                              uint64_t head_bytes, uint64_t tail_bytes,
                                              uint64_t budget_per_card) {
    FleetSplit s; s.layers = layers; s.cards = cards;
    s.head_bytes = head_bytes; s.tail_bytes = tail_bytes;
    auto fail = [&](const std::string &m) { s.refuse = m; s.ok = false; return s; };
    if (layers == 0 || cards == 0) return fail("layers and cards must both be >= 1");
    if (cards > layers) {
        char b[128]; snprintf(b, sizeof b, "%u cards for %u layers: a card would own zero layers "
                              "and every hop is pure cost", cards, layers);
        return fail(b);
    }
    if (bytes_per_layer.size() != layers) {
        char b[160]; snprintf(b, sizeof b, "byte table has %zu entries for %u layers",
                              bytes_per_layer.size(), layers);
        return fail(b);
    }
    uint64_t layer_total = 0;
    for (auto v : bytes_per_layer) layer_total += v;
    s.total_bytes = layer_total + head_bytes + tail_bytes;
    s.smallest_budget = budget_per_card;

    // card 0 also carries the embedding, the last card also carries the output projection: in the
    // DEVICE form those two are 286,064,640 B each (2.25 bpw repacked: 4096*248320 weights ->
    // 248320*(4096/64)*16 codes + 248320*(4096/128)*4 scales), which is why a budget that ignored
    // them is a lie - one head tensor alone is 273 MiB, 7% of a 3840 MiB heap.
    uint64_t share = (layer_total + cards - 1) / cards;
    std::vector<uint64_t> cap(cards, budget_per_card);
    std::vector<uint64_t> used(cards, 0);
    used[0] = head_bytes; used[cards - 1] += tail_bytes;
    for (uint32_t k = 0; k < cards; ++k) {
        if (used[k] >= cap[k]) {
            char b[192];
            snprintf(b, sizeof b, "card %u is already over budget (%llu B of fixed head/tail weight "
                                  "against a %llu B budget) before any layer is assigned",
                     k, (unsigned long long)used[k], (unsigned long long)cap[k]);
            return fail(b);
        }
    }
    std::vector<uint32_t> cut(cards + 1, 0);   // cut[k] = first layer on card k
    cut[0] = 0; cut[cards] = layers;
    uint32_t l = 0;
    for (uint32_t k = 0; k + 1 < cards; ++k) {
        uint64_t want_load = share;
        while (l < layers - (cards - 1 - k) && used[k] + bytes_per_layer[l] <=
               std::min(cap[k], used[k] + want_load)) {
            used[k] += bytes_per_layer[l];
            ++l;
            if (used[k] >= share) break;
        }
        cut[k + 1] = l;
    }
    for (uint32_t k = cards - 1; k < cards; ++k) {
        while (l < layers) { used[k] += bytes_per_layer[l]; ++l; }
    }
    for (uint32_t k = 0; k < cards; ++k) {
        if (k + 1 < cards && cut[k + 1] <= cut[k]) {
            char b[192];
            snprintf(b, sizeof b, "card %u got an empty or inverted range - the byte balance "
                                  "cannot fit %u layers into %u cards at %llu B/card",
                     k, layers, cards, (unsigned long long)budget_per_card);
            return fail(b);
        }
        LayerRange r;
        r.card = k; r.layer_lo = cut[k];
        r.layer_hi = (k + 1 < cards) ? cut[k + 1] : layers;
        uint64_t by = (k == 0 ? head_bytes : 0) + (k + 1 == cards ? tail_bytes : 0);
        for (uint32_t j = r.layer_lo; j < r.layer_hi; ++j) by += bytes_per_layer[j];
        r.bytes = by; r.budget = cap[k];
        s.ranges.push_back(r);
    }
    for (uint32_t k = 0; k < cards; ++k) {
        if (s.ranges[k].bytes > cap[k]) {
            char b[400];   // sized for the longest of these sentences; a truncated refusal is the
                           // worst possible output because the reader cannot tell which constraint fired.
            snprintf(b, sizeof b,
                     "card %u would need %llu B against a %llu B budget (smallest card). "
                     "Raise the budget, add a card, or tier the FFN weights - the 78.3%% "
                     "device-residency judge line (DERIVED: 1682 MiB/token at 7.12 GB/s against a "
                     "53.8 ms/token budget, DEEPINFER_EVAL 0) says a card that spills to host "
                     "memory is slower than the single-card build, not faster.",
                     k, (unsigned long long)s.ranges[k].bytes, (unsigned long long)cap[k]);
            return fail(b);
        }
    }
    s.ok = true;
    return s;
}

static inline void fleet_split_print(const FleetSplit &s, const char *byte_source) {
    if (!s.ok) {
        printf("  SPLIT REFUSED: %s\n", s.refuse.c_str());
        return;
    }
    printf("  per-card weight bytes from %s\n", byte_source);
    printf("  %4s %-14s %14s %14s %9s\n", "card", "layers", "bytes needed", "bytes budget",
           "headroom");
    for (const auto &r : s.ranges) {
        printf("  %4u [%3u,%3u)    %14llu %14llu %8.1f MiB\n", r.card, r.layer_lo, r.layer_hi,
               (unsigned long long)r.bytes, (unsigned long long)r.budget,
               double(int64_t(r.budget) - int64_t(r.bytes)) / 1048576.0);
    }
    uint64_t sum = 0; for (const auto &r : s.ranges) sum += r.bytes;
    printf("  layers %u over %u card(s): %u hop(s)/token (contiguous split => one boundary per "
           "card transition), %llu B/token across the ring at %llu B each, payload total %llu B, "
           "sum of ranges %llu B (conservation %s)\n",
           s.layers, s.cards, s.hops_per_token(),
           (unsigned long long)(FLEET_HIDDEN_STATE_BYTES * s.hops_per_token()),
           (unsigned long long)FLEET_HIDDEN_STATE_BYTES,
           (unsigned long long)s.total_bytes, (unsigned long long)sum,
           sum == s.total_bytes ? "exact" : "BROKEN");
    // The claim the report is allowed to make, and the one it is not.
    printf("  DERIVED hop cost per token: %u x (engine %.1f us + fence round trip %.1f us) = "
           "%.2f ms  [engine scaled from the MEASURED 6.2 us at 32 KiB; round trip MEASURED 99.4 "
           "us]\n", s.hops_per_token(),
           FLEET_MEASURED_COPY32K_ENGINE_US * double(FLEET_HIDDEN_STATE_BYTES) / 32768.0,
           FLEET_MEASURED_SUBMIT_FENCE_US,
           s.hops_per_token() * (FLEET_MEASURED_COPY32K_ENGINE_US *
                                 double(FLEET_HIDDEN_STATE_BYTES) / 32768.0 +
                                 FLEET_MEASURED_SUBMIT_FENCE_US) / 1000.0);
    printf("  CAPACITY, NOT SPEED: one token's layer chain is sequential, so a 2-card split halves\n"
           "  the bytes per card and leaves the token's critical path alone. The depth-D ring only\n"
           "  recovers time when >= 2 microbatches (tokens) are in flight; then card k computes\n"
           "  microbatch i+1's range while microbatch i's boundary bytes are still moving.\n");
}

// ------------------------------------------------------------------ the pipelined fleet runner
// The policy actually implemented, and the two that are only declared. This is deliberately
// explicit so that nobody reads an enum value as a claim that the machine was tested.
enum FleetSyncPolicy {
    // Implemented. Cross-card order = host waits the producer's D2H fence, then submits the
    // consumer's H2D. Works with 1.0 entry points only. Cost: one host wait per card transition
    // per microbatch, MEASURED round trip 99.4 us each.
    FLEET_SYNC_HOST_FENCE = 0,
    // Declared, NOT IMPLEMENTED: would let the consumer's transfer queue wait the producer's fence
    // without a host in the loop. Needs VK_KHR_external_semaphore + a handle type both drivers
    // advertise; fleet_probe() reports whether that is even on the table.
    FLEET_SYNC_EXTERNAL_SEMAPHORE = 1,
    // Declared, NOT IMPLEMENTED: the producer's last dispatch stores a sequence word into the
    // ring's host-visible flag line and the consumer runs a bounded spin kernel before its copy.
    // Needs a shader this repo does not have; it is the only shape with zero host waits, so it is
    // the right thing to build next, not the right thing to claim now.
    FLEET_SYNC_SLOT_FLAG_SPIN = 2
};

struct HiddenStateRef {
    VkBuffer buf = VK_NULL_HANDLE;
    uint64_t off = 0;
    uint64_t bytes = FLEET_HIDDEN_STATE_BYTES;
};

// One unit of scheduled work: a contiguous layer range on one card, for one microbatch.
struct FleetStage {
    uint32_t card = 0;
    uint32_t range = 0;      // index into FleetSplit::ranges
    uint32_t micro = 0;      // token / microbatch index
    uint64_t seq = 0;        // position in the pipeline: micro * n_ranges + range
    VkCommandBuffer cb = VK_NULL_HANDLE;
    HiddenStateRef hs;       // filled by the record callback: where the boundary hidden state is
};

// The record callback is the ONLY thing a fleet driver has to supply to drive a real model: it
// records the range's dispatches into cb on ctx, and reports where the hidden state ended up. The
// hook in bonsai_forward.h (fwd_note_boundary) is what hands exactly that triple to it, so the
// forward graph does not get restructured to fit this file.
using FleetRecordFn = std::function<bool (FleetCtx &ctx, const FleetStage &st, VkCommandBuffer cb,
                                          HiddenStateRef *out_hs)>;

// Where the NEXT range's hidden state lives, on the consumer card. This exists because `out_hs`
// above belongs to the PRODUCER: on a real fleet each card owns its own sub-arena, so the H2D leg
// must not write the producer's buffer - it writes the consumer's input window for range r+1. With
// no callback set the pipeline falls back to hs.buf, which is correct only for the single-window
// rig `bonsai_fleet ring` runs, and that limitation is printed rather than hidden.
using FleetInputRefFn = std::function<HiddenStateRef (FleetCtx &ctx, uint32_t card, uint32_t range)>;

// Sync objects are indexed [card][slot], because a fence created on card A cannot be armed by card
// B's submit. Three DISTINCT fences per (card, slot):
//   f_comp - that card's compute for the slot's stage finished. Two users: the D2H leg (replaced by
//            s_comp when the copy is on the same card, which is the normal case) and the recycling of
//            p.cb[card][slot], which may not be re-recorded until its previous submit has retired.
//   f_out  - that card's D2H into the ring slot finished. This is the ONLY host wait the
//            FLEET_SYNC_HOST_FENCE policy takes per hop, and it is what makes the publish safe.
//   f_in   - the consumer card's H2D out of the slot finished. Waited lazily, only when the ring is
//            about to reuse the slot, so depth D is what keeps this wait off the critical path.
// s_comp/s_in are same-device binary semaphores: compute -> D2H and H2D -> next compute. They cost
// nothing in host time, which is the entire point of using them instead of a fence.
// Fences are created UNSIGNALED (flags = 0). That costs the "wait before anything was submitted
// returns immediately" convenience the first draft of this struct relied on, and buys the property
// that matters: a fence only reads signalled because its own submit retired it. See
// fleet_pipeline_init for the measurement. Every host wait here carries a timeout, so a fence that
// was never armed shows up as a counted timeout plus a vkQueueWaitIdle in the log rather than a hang.
struct FleetPipeline {
    std::vector<FleetCtx> *cards = nullptr;
    FleetSplit split;
    PinnedRing ring;
    FleetSyncPolicy policy = FLEET_SYNC_HOST_FENCE;
    uint32_t slots = 0;
    std::vector<std::vector<VkCommandBuffer>> cb;   // [card][slot] compute recording
    // [card][slot] copy-leg CBs. NOT one per card: a card can be the consumer of hop i (its H2D
    // leg) and the producer of hop i+1 in the same pipeline turn, and re-recording one VkCommandBuffer
    // while an earlier submit of it is still running is undefined - it is the same rule that makes
    // bonsai_weights.h wait a fence before the next begin_cmd(). Per-slot CBs are safe precisely
    // because a slot recurs D sequences later, by which time both its f_out (waited at step 5) and
    // its f_in (waited by the back-pressure loop) have been retired and re-armed.
    std::vector<std::vector<VkCommandBuffer>> dma_cb;
    std::vector<std::vector<VkFence>> f_comp, f_out, f_in;
    std::vector<std::vector<VkSemaphore>> s_comp, s_in;
    // the hop whose H2D is in flight: the next stage on `card` waits s_in before its compute submit,
    // so the consumer never reads an input buffer the copy has not finished writing.
    uint32_t pending_in_card = 0xFFFFFFFFu;
    uint64_t pending_in_slot = 0;
    FleetInputRefFn input_ref;
    // Which card armed f_in[slot], i.e. whose fence it is: the H2D runs on the CONSUMER, while the
    // scheduler's idea of "sequence seq lives on range seq%nb" names the PRODUCER's card. Indexing
    // f_in by the producer waits a fence on a device that never submitted anything - and with fences
    // now created UNSIGNALED that wait does not return instantly, it burns its whole timeout and
    // then forces the vkQueueWaitIdle escape hatch, so the bug stops being silent and starts being
    // slow-and-loud. Either way the owner table is what keeps the overwrite guarded: wait semantics
    // only protect you if the fence you wait on is the one your submit arms. Wrong-card fence
    // indexing is the single most dangerous bug class in this file, hence the explicit owner table
    // instead of a derivation.
    std::vector<uint32_t> slot_card;
    // slot -> the stage number whose bytes are in that ring slot NOW (UINT64_MAX = nothing has ever
    // landed there). The cursor counts payloads, the producer's pattern is keyed on stages, and a
    // verification that conflates the two is exactly what made the rig report 6 phantom mismatches.
    std::vector<uint64_t> slot_src_seq;
    // Doorbell accounting, so the fence-vs-true-completion gap is a number in the log rather than a
    // claim in a comment. us_wait_out stays the FENCE column (what the design used to trust);
    // us_wait_bell is the stamp column (what the bytes actually needed).
    double us_wait_bell = 0.0;
    uint64_t bell_hits = 0, bell_timeouts = 0;
    uint64_t stages_run = 0, hops = 0, waits_out = 0, waits_in = 0, backpressure = 0, waits_cb = 0;
    uint64_t hop_bytes = 0;
    double us_wait_out = 0.0, us_wait_in = 0.0, us_wait_cb = 0.0;
    bool ready = false;
};

// One fenced submit, with the reset-after-wait rule bonsai_weights.h:295-298 states: a fence that
// is never re-armed reports success for work that has not started, and that is how 1866 MiB of
// "SDMA in 12.5 ms = 156 GB/s" got printed once already.
static inline double fleet_fence_wait_arm(FleetCtx &c, VkFence f) {
    const auto a0 = std::chrono::steady_clock::now();
    VkResult r = vkWaitForFences(c.dev, 1, &f, VK_TRUE, ~0ull);
    if (r != VK_SUCCESS) { fprintf(stderr, "fleet: fence wait failed (%d)\n", (int)r); exit(1); }
    const auto a1 = std::chrono::steady_clock::now();
    FK_CHECK(vkResetFences(c.dev, 1, &f));
    return std::chrono::duration<double, std::micro>(a1 - a0).count();
}

static inline void fleet_submit(FleetCtx &c, VkQueue q, VkCommandBuffer cb, VkFence f,
                                VkSemaphore wait, VkSemaphore signal,
                                VkPipelineStageFlags wait_stage) {
    VkSubmitInfo si{VK_STRUCTURE_TYPE_SUBMIT_INFO};
    si.commandBufferCount = 1; si.pCommandBuffers = &cb;
    if (wait) { si.waitSemaphoreCount = 1; si.pWaitSemaphores = &wait;
                si.pWaitDstStageMask = &wait_stage; }
    if (signal) { si.signalSemaphoreCount = 1; si.pSignalSemaphores = &signal; }
    FK_CHECK(vkQueueSubmit(q, 1, &si, f));
}

static inline void fleet_begin_cb(FleetCtx &c, VkCommandBuffer cb) {
    VkCommandBufferBeginInfo bi{VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO};
    bi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    FK_CHECK(vkBeginCommandBuffer(cb, &bi));
    (void)c;
}

// Records one vkCmdCopyBuffer on the card's OWN transfer-only queue when it has one and submits it
// asynchronously - no fence wait inside, because a synchronous "DMA leg" would serialise the very
// overlap this design exists to create. When the card has no transfer-only family the copy goes on
// the compute queue and the overlap is gone, which is not a detail: MEASURED on this card D2H is
// 3.89 GB/s on the compute queue against 7.15 GB/s on the SDMA ring, and the concurrency sweep
// shows a copy costs the compute queue +0.0..+0.5% only when the two really are separate rings
// (DEEPINFER_EVAL §0/§3).
// The doorbell word a D2H leg should stamp into the ring's bell buffer, right after the payload copy
// and in the SAME command buffer. Queue order is the whole guarantee: the fill cannot retire before
// the copy in front of it, so the host reading `value` through the coherent mapping means the payload
// bytes are in the slot - without trusting a fence that this driver measures as lying.
struct FleetStamp {
    VkBuffer buf = VK_NULL_HANDLE;
    uint64_t off = 0;
    uint32_t value = 0;
};

static inline void fleet_dma_async(FleetCtx &c, VkCommandBuffer cb, VkBuffer src, uint64_t s_off,
                                   VkBuffer dst, uint64_t d_off, uint64_t bytes, VkFence fence,
                                   VkSemaphore wait, VkSemaphore signal,
                                   const FleetStamp *stamp = nullptr) {
    fleet_begin_cb(c, cb);
    // bytes == 0 (or a NULL buffer) records nothing at all. vkCmdCopyBuffer with size 0 is not
    // something to build a synchronisation scheme on, and the degenerate "the hop needs no copy"
    // case - the ring is imported, so both cards already see the same pages - wants an empty CB, not
    // a zero-length copy.
    if (bytes && src && dst) {
        VkBufferCopy cp{s_off, d_off, bytes};
        vkCmdCopyBuffer(cb, src, dst, 1, &cp);
    }
    // vkCmdFillBuffer needs TRANSFER_SRC or TRANSFER_DST on the target; the bell is created with
    // TRANSFER_DST for exactly this line.
    if (stamp && stamp->buf) vkCmdFillBuffer(cb, stamp->buf, stamp->off, 4, stamp->value);
    FK_CHECK(vkEndCommandBuffer(cb));
    // wait-stage: the semaphore is being waited by the TRANSFER stage here (a copy reads what the
    // previous copy or shader wrote), never by COMPUTE_SHADER - naming the wrong stage is how a
    // barrier "works" in a test and races in production.
    fleet_submit(c, c.has_tx ? c.qtx : c.queue, cb, fence, wait, signal,
                 VK_PIPELINE_STAGE_TRANSFER_BIT);
}

// The rig's stand-in for "card k computes a layer range" is supplied by the caller (bonsai_fleet's
// `ring` uses a vkCmdFillBuffer of the hidden-state window): it costs no .spv, no descriptor set and
// no pipeline, and for measuring the ring mechanism all that matters is that the engine writes those
// bytes and the D2H cannot start before they are written. The real driver replaces the callback with
// the per-block recording bonsai_forward.h exposes.
static inline bool fleet_pipeline_init(FleetPipeline &p, std::vector<FleetCtx> &cards,
                                       const FleetSplit &split, uint32_t depth,
                                       uint64_t slot_bytes) {
    if (!split.ok) { p.split = split; p.ready = false; return false; }
    if (depth < 1) return false;
    // A range naming a card that is not in `cards` would index p.cb[card][...] past the end on the
    // first stage. The scheduler and the fleet are built by different callers (fleet_split_schedule
    // vs. the rig's hand-made split), so the link between them is checked rather than assumed.
    for (const auto &r : split.ranges)
        if (r.card >= cards.size()) {
            fprintf(stderr, "fleet pipeline REFUSE: range [%u,%u) names card %u but only %zu card "
                            "context(s) exist.\n", r.layer_lo, r.layer_hi, r.card, cards.size());
            return false;
        }
    if (split.ranges.empty()) { fprintf(stderr, "fleet pipeline REFUSE: empty schedule\n"); return false; }
    p.cards = &cards; p.split = split; p.slots = depth;
    if (!fleet_ring_alloc_device(p.ring, cards[0], depth, slot_bytes)) return false;
    // The attach attempt is the only honest probe of "do the two cards see one heap". On a machine
    // with one card there is nothing to attach, and fleet_probe() says so instead of printing a
    // success this host cannot produce.
    for (uint32_t k = 1; k < cards.size(); ++k)
        fleet_ring_attach(p.ring, cards[k], cards[0].inst, k);

    VkFenceCreateInfo fci{VK_STRUCTURE_TYPE_FENCE_CREATE_INFO};
    // Created UNSIGNALED, and the reason is a measured misread rather than a style preference. The
    // first three versions of this rig created fences VK_FENCE_CREATE_SIGNALED_BIT so that "wait
    // before anything was submitted" would return immediately. It does - that is the problem, not
    // the feature: a fence is a binary state, so a wait on a fence that is ALREADY signalled returns
    // without reference to the submit it was supposed to witness. runs/fleet_ring_v3.log times such
    // an 8 MiB D2H leg at 0.7-2.3 us while vkQueueWaitIdle behind it still needs 1202 us; the flags =
    // 0 fence in the same log blocks for 1263.6 us and leaves a 0.8 us tail. Nothing vendor-specific
    // is being blamed - the driver follows the spec, and the engine already knew this rule (the
    // reset-after-wait note in bonsai_weights.h's submit path quotes the same trap as a 156 GB/s
    // "bandwidth" that could not be one). The only VK_FENCE_CREATE_SIGNALED_BIT left in the tree is
    // the rig's control, which re-measures the effect every run instead of asking the reader to
    // trust an old log. What flags = 0 costs is that an early wait cannot return immediately, so
    // every host wait in this file carries a timeout and a counted escape hatch.
    fci.flags = 0;
    VkSemaphoreCreateInfo sci{VK_STRUCTURE_TYPE_SEMAPHORE_CREATE_INFO};
    p.f_comp.resize(cards.size()); p.f_out.resize(cards.size()); p.f_in.resize(cards.size());
    p.s_comp.resize(cards.size()); p.s_in.resize(cards.size());
    p.cb.resize(cards.size());
    p.dma_cb.resize(cards.size());
    p.slot_card.assign(depth, 0xFFFFFFFFu);
    p.slot_src_seq.assign(depth, ~uint64_t(0));
    for (uint32_t k = 0; k < cards.size(); ++k) {
        FleetCtx &c = cards[k];
        p.f_comp[k].assign(depth, VK_NULL_HANDLE);
        p.f_out[k].assign(depth, VK_NULL_HANDLE);
        p.f_in[k].assign(depth, VK_NULL_HANDLE);
        p.s_comp[k].assign(depth, VK_NULL_HANDLE);
        p.s_in[k].assign(depth, VK_NULL_HANDLE);
        for (uint32_t s = 0; s < depth; ++s) {
            FK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &p.f_comp[k][s]));
            FK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &p.f_out[k][s]));
            FK_CHECK(vkCreateFence(c.dev, &fci, nullptr, &p.f_in[k][s]));
            FK_CHECK(vkCreateSemaphore(c.dev, &sci, nullptr, &p.s_comp[k][s]));
            FK_CHECK(vkCreateSemaphore(c.dev, &sci, nullptr, &p.s_in[k][s]));
        }
        VkCommandBufferAllocateInfo cai{VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO};
        cai.commandPool = c.cpool; cai.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
        cai.commandBufferCount = depth;
        p.cb[k].resize(depth);
        FK_CHECK(vkAllocateCommandBuffers(c.dev, &cai, p.cb[k].data()));
        // one copy-leg CB per slot, for the reuse reason in the struct comment. They come from
        // c.upool, NOT c.cpool: a command buffer allocated from a pool bound to family A may only be
        // submitted to a queue of family A, and fleet_dma_async submits these to the transfer-only
        // queue when the card has one. upool's queueFamilyIndex is exactly that queue's family, so
        // the copy runs on a pool-family match instead of relying on the driver not minding.
        p.dma_cb[k].assign(depth, VK_NULL_HANDLE);
        cai.commandPool = c.upool;
        FK_CHECK(vkAllocateCommandBuffers(c.dev, &cai, p.dma_cb[k].data()));
    }
    p.ready = true;
    return true;
}

static inline void fleet_pipeline_destroy(FleetPipeline &p) {
    if (!p.ready && !p.cards) return;
    if (p.cards) {
        for (uint32_t k = 0; k < p.cards->size(); ++k) {
            FleetCtx &c = (*p.cards)[k];
            if (!c.dev) continue;
            vkDeviceWaitIdle(c.dev);
            for (uint32_t s = 0; s < p.slots; ++s) {
                if (!p.f_comp[k].empty() && p.f_comp[k][s]) vkDestroyFence(c.dev, p.f_comp[k][s], nullptr);
                if (!p.f_out[k].empty() && p.f_out[k][s]) vkDestroyFence(c.dev, p.f_out[k][s], nullptr);
                if (!p.f_in[k].empty() && p.f_in[k][s]) vkDestroyFence(c.dev, p.f_in[k][s], nullptr);
                if (!p.s_comp[k].empty() && p.s_comp[k][s]) vkDestroySemaphore(c.dev, p.s_comp[k][s], nullptr);
                if (!p.s_in[k].empty() && p.s_in[k][s]) vkDestroySemaphore(c.dev, p.s_in[k][s], nullptr);
            }
            if (!p.cb[k].empty())
                vkFreeCommandBuffers(c.dev, c.cpool, uint32_t(p.cb[k].size()), p.cb[k].data());
            if (!p.dma_cb[k].empty())
                vkFreeCommandBuffers(c.dev, c.upool, uint32_t(p.dma_cb[k].size()), p.dma_cb[k].data());
        }
        // The ring's own allocation belongs to card 0 - freeing it once per card would be a
        // double-free, which is the least recoverable kind of bug a teardown can have.
        if (!p.cards->empty()) fleet_ring_free((*p.cards)[0].dev, p.ring);
    }
    p.f_comp.clear(); p.f_out.clear(); p.f_in.clear(); p.s_comp.clear(); p.s_in.clear();
    p.cb.clear(); p.dma_cb.clear();
    p.ready = false;
}

// Runs the schedule: `n_micro` microbatches through the card chain, ring depth D as back-pressure.
// With one card and one microbatch this degenerates into the sequence a single-GPU build performs,
// which is what makes `bonsai_fleet ring` a measurement of the mechanism rather than a simulation of
// one. With >= 2 microbatches it is the shape the multi-card win actually needs: the hop of
// microbatch i overlaps the compute of microbatch i+1 on the producer card.
static inline bool fleet_pipeline_run(FleetPipeline &p, uint32_t n_micro,
                                      const FleetRecordFn &record) {
    if (!p.ready || !record) return false;
    const uint32_t nb = uint32_t(p.split.ranges.size());
    const uint32_t mask = p.ring.mask;
    for (uint32_t m = 0; m < n_micro; ++m) {
        for (uint32_t r = 0; r < nb; ++r) {
            const uint64_t seq = uint64_t(m) * nb + r;
            const uint32_t slot = uint32_t(seq & uint64_t(mask));
            const uint32_t card = p.split.ranges[r].card;
            FleetCtx &cc = (*p.cards)[card];

            // (1) Back-pressure lives in the crossing branch below, keyed on the PAYLOAD number,
            //     not here. It was keyed on the stage number `seq`, which is a different sequence:
            //     a payload exists only at a card transition (r -> r+1), so with nb ranges per
            //     microbatch the ring sees every nb-th stage, and `fleet_ring_publish(seq)` made
            //     `produced` a stage cursor while only nb-of-nb slots ever held bytes. The rig's
            //     arrival check then expected a payload in slots that were never written and printed
            //     FAIL against a copy that had in fact delivered everything it was asked to. One
            //     cursor, one meaning: produced/consumed/slot are now hops, and the producer's own
            //     stage number travels with the payload as `slot_src_seq` for the verification.

            // (1b) Command-buffer recycling. p.cb[card][slot] was last submitted D stages ago for a
            // different microbatch and its compute may still be running; begin_cmd on a CB whose
            // previous submit has not retired is undefined. bonsai_weights.h records the same rule for
            // its own buffers - "legal only because each of these buffers is submitted and
            // fence-waited before the next begin_cmd()" - and it is applied here rather than assumed.
            // This wait is counted separately (waits_cb): on a single-card rig with depth D it is the
            // cost of the rig's own finite command-buffer pool, not of any real hop.
            if (seq >= uint64_t(p.slots)) {
                const auto w0 = std::chrono::steady_clock::now();
                fleet_fence_wait_arm(cc, p.f_comp[card][slot]);
                p.us_wait_cb += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - w0).count();
                ++p.waits_cb;
            }

            // (2) The consumer's side of the previous hop, device-side: this stage's compute waits
            // the H2D that filled its input buffer, on the SAME card, so no host wait is needed
            // here at all.
            VkSemaphore in_sem = VK_NULL_HANDLE;
            if (p.pending_in_card == card) {
                in_sem = p.s_in[card][uint32_t(p.pending_in_slot & uint64_t(mask))];
                p.pending_in_card = 0xFFFFFFFFu;
            }

            VkCommandBuffer cb = p.cb[card][slot];
            FleetStage st;
            st.card = card; st.range = r; st.micro = m; st.seq = seq; st.cb = cb;
            fleet_begin_cb(cc, cb);
            HiddenStateRef hs;
            if (!record(cc, st, cb, &hs)) {
                fprintf(stderr, "fleet: record callback failed on card %u range %u\n", card, r);
                return false;
            }
            FK_CHECK(vkEndCommandBuffer(cb));
            fleet_submit(cc, cc.queue, cb, p.f_comp[card][slot], in_sem,
                         p.s_comp[card][slot], VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT);
            p.stages_run++;
            if (in_sem != VK_NULL_HANDLE) {
                // The semaphore was consumed by this submit; the fence behind it is now the thing to
                // collect, and it is collected at (1) rather than here so the host does not stall.
                p.pending_in_card = 0xFFFFFFFFu;
            }

            // (3) The boundary payload only exists at a card transition; inside one card's range the
            // hidden state never leaves the device, which is the whole reason for a contiguous cut
            // (and why the 50.25 MiB/sequence GDN state, MEASURED, does not move either).
            const bool crosses = (r + 1 < nb);
            if (!crosses) continue;
            const uint32_t ncard = p.split.ranges[r + 1].card;
            FleetCtx &nc = (*p.cards)[ncard];

            // (1) Back-pressure, AT THE HOP, keyed on the payload number hseq = p.hops. The slot may
            // not be re-written while a flight still reads it: that is the guard against the
            // silent-overwrite failure mode, and the only place the ring's depth shows up. Depth 1
            // degenerates to a wait every hop, so D>1 is not tuning, it is the mechanism.
            const uint64_t hseq = p.hops;
            const uint32_t hslot = uint32_t(hseq & uint64_t(mask));
            while (hseq >= uint64_t(p.ring.depth) +
                   p.ring.consumed.load(std::memory_order_acquire)) {
                const uint64_t old = p.ring.consumed.load(std::memory_order_acquire);
                const uint32_t oslot = uint32_t(old & uint64_t(mask));
                // The fence belongs to the card that ran the H2D (the consumer of hop `old`), not to
                // the card that produced it. 0xFFFFFFFF means nothing ever armed this slot's f_in -
                // waiting an unarmed, created-signaled fence would "succeed" instantly and teach the
                // reader that the mechanism is sound when it is not - so the owner table is checked.
                const uint32_t osc = p.slot_card[oslot];
                if (osc != 0xFFFFFFFFu) {
                    const auto w0 = std::chrono::steady_clock::now();
                    fleet_fence_wait_arm((*p.cards)[osc], p.f_in[osc][oslot]);
                    p.us_wait_in += std::chrono::duration<double, std::micro>(
                        std::chrono::steady_clock::now() - w0).count();
                    ++p.waits_in;
                    p.slot_card[oslot] = 0xFFFFFFFFu;
                }
                ++p.backpressure;
                p.ring.consumed.store(old + 1, std::memory_order_release);
            }

            const uint64_t roff = uint64_t(hslot) * p.ring.slot_bytes;
            const uint64_t n = std::min<uint64_t>(hs.bytes, p.ring.slot_bytes);
            // The stamp value is payload-number + 1 because 0 is the "nothing has landed" pattern the
            // bell was zeroed to. It is a uint32 while the cursor is uint64: after 2^32 hops the
            // stamp would alias, which a long-running server must handle by resetting the bell with
            // the ring (documented, not silently ignored - an alias would read as "landed" early).
            FleetStamp stamp;
            if (p.ring.bell_map) {
                stamp.buf = p.ring.bell;
                stamp.off = uint64_t(hslot) * PinnedRing::BELL_STRIDE;
                stamp.value = uint32_t(hseq & 0xFFFFFFFFull) + 1u;
            }

            // (4) D2H out of the producer, gated on the producer's own compute by a same-device
            // semaphore. Async: the host does NOT wait here.
            // The transfer objects are indexed by hslot (the payload's slot), while the compute
            // semaphore that gates this copy is stage-indexed, because that is the submit whose
            // completion it waits: s_comp[card][slot] was signalled by THIS stage's compute.
            fleet_dma_async(cc, p.dma_cb[card][hslot], hs.buf, hs.off, p.ring.view, roff, n,
                            p.f_out[card][hslot], p.s_comp[card][slot], VK_NULL_HANDLE,
                            p.ring.bell_map && n ? &stamp : nullptr);

            // (5) The ONE unavoidable host wait of FLEET_SYNC_HOST_FENCE: cross-device ordering has
            // no portable primitive below VK_KHR_external_semaphore, so the scheduler must observe
            // "the bytes are in the slot" before letting the consumer read it. Everything the GPU
            // could do without being told (the producer's next range, the consumer's H2D queueing)
            // has already been submitted by the time we get here, so this wait is where the bubble
            // lives - and it is counted, not hidden.
            // MEASURED, and the reason there are two columns. In runs/fleet_ring_v3.log, when the
            // pipeline's fences were still created SIGNALED, the fence column read 2.3 us per hop
            // while the stamp column behind it needed 2445.4 us for the same hops - a factor of ~1000
            // - because a wait on an already-signalled fence proves nothing (see fleet_pipeline_init).
            // With flags = 0 the columns are consistent in the way a sound fence should be: in
            // runs/fleet_ring_v4.log the fence took 2706.7 us/hop and the stamp then found the
            // payload already there in 0.3 us, which is the expected shape because the fence waits the
            // same queue the stamp is written on. Read the pair, not the second number: the stamp
            // alone is the column that would show the copy's true cost if the fence were ever removed,
            // and it stays the publish condition because it is queue order, needing no assumption
            // about fence semantics at all.
            {
                const auto w0 = std::chrono::steady_clock::now();
                fleet_fence_wait_arm(cc, p.f_out[card][hslot]);
                p.us_wait_out += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - w0).count();
                ++p.waits_out;
            }
            {
                const auto w0 = std::chrono::steady_clock::now();
                const bool landed = fleet_ring_wait_bell(p.ring, hseq, 250.0);
                p.us_wait_bell += std::chrono::duration<double, std::micro>(
                    std::chrono::steady_clock::now() - w0).count();
                if (landed) {
                    ++p.bell_hits;
                } else {
                    // Never publish a slot the stamp did not confirm. Drain the queue, count it, and
                    // let the rig print the timeout: silently falling back would be the same mistake
                    // as trusting the fence.
                    ++p.bell_timeouts;
                    vkQueueWaitIdle(cc.has_tx ? cc.qtx : cc.queue);
                }
            }
            fleet_ring_publish(p.ring, hseq);

            // (6) H2D into the consumer's OWN input window. hs.buf is the producer's window and must
            // not be the destination: with a real fleet the two cards do not share a VkBuffer, and
            // even on this rig's single-card stand-in writing the producer's buffer would let the
            // producer clobber what the consumer has not read yet. input_ref() says where range r+1
            // reads its hidden state from; if the caller did not supply one the hop lands in hs.buf
            // and `ring` prints that it is running the shared-window variant.
            // When the ring was IMPORTED by that card the slot and the consumer's view are the same
            // pages; the copy is then redundant, and skipping it is the entire iGPU/APU payoff. The
            // attach print at init says which case this run was.
            HiddenStateRef in = p.input_ref ? p.input_ref(nc, ncard, r + 1) : hs;
            if (!p.ring.shared_with) {
                fleet_dma_async(nc, p.dma_cb[ncard][hslot], p.ring.view, roff, in.buf, in.off, n,
                                p.f_in[ncard][hslot], VK_NULL_HANDLE, p.s_in[ncard][hslot]);
            } else {
                // Zero bytes, so the CB is empty and the submit only exists to arm f_in and release
                // s_in. Without arming it, the back-pressure wait at (1) would block on a fence that
                // nothing ever signals (and now that fences are created with flags = 0, that is a
                // timeout + vkQueueWaitIdle, not an instant return); arming a fence created SIGNALED
                // instead would "satisfy" the wait without witnessing anything and the slot would be
                // overwritten unguarded. Both are failure modes; the empty submit is the one that
                // costs nothing and is defined.
                fleet_dma_async(nc, p.dma_cb[ncard][hslot], VK_NULL_HANDLE, 0, VK_NULL_HANDLE, 0, 0,
                                p.f_in[ncard][hslot], VK_NULL_HANDLE, p.s_in[ncard][hslot]);
            }
            p.pending_in_card = ncard;
            p.pending_in_slot = hseq;
            p.slot_card[hslot] = ncard;
            // Which producer stage's bytes are in this slot NOW - the rig's arrival check needs it,
            // because the fill pattern is written by the stage and the cursor is counted in hops.
            p.slot_src_seq[hslot] = seq;
            p.hops++; p.hop_bytes += n;
        }
    }
    // Drain: every outstanding H2D must be collected before the ring cursors are believed, and
    // before the command pools that hold these CBs can be reused by the caller.
    while (p.ring.consumed.load(std::memory_order_acquire) <
           p.ring.produced.load(std::memory_order_acquire)) {
        const uint64_t old = p.ring.consumed.load(std::memory_order_acquire);
        const uint32_t oslot = uint32_t(old & uint64_t(mask));
        const uint32_t osc = p.slot_card[oslot];
        if (osc != 0xFFFFFFFFu) {
            fleet_fence_wait_arm((*p.cards)[osc], p.f_in[osc][oslot]);
            p.slot_card[oslot] = 0xFFFFFFFFu;
        }
        p.ring.consumed.store(old + 1, std::memory_order_release);
    }
    for (auto &c : *p.cards) if (c.has_tx) vkQueueWaitIdle(c.qtx);
    for (auto &c : *p.cards) vkQueueWaitIdle(c.queue);
    return true;
}

// ------------------------------------------------------------------ topology print
// The verdicts this host can honestly give. `probe` prints these; it does not infer them from the
// device count, because "there are two cards" and "two cards can share a heap" are different facts.
struct FleetCapability {
    bool multi_card_layer_split = false;
    bool igpu_zero_copy = false;
    bool transfer_only_per_card = false;
    bool shared_host_heap = false;
    bool external_semaphore = false;
    // The two counts the (A) verdict is derived from. They are published rather than left inside
    // the predicate because the single-card case is NOT a "NO": with one compute device the
    // fleet-wide property cannot be evaluated at all, while this very card's per-device property
    // is right above the verdict in the enumeration table (family 2, TRANSFER-ONLY). Printing a bare
    // "NO" there contradicted the table three lines earlier - the exact class of internal
    // contradiction the audit rule in this repo's docs forbids, so the verdict prints three-valued.
    uint32_t n_compute_devices = 0;
    uint32_t n_devices_with_transfer_only = 0;
    std::string why_multi, why_zero;
};

static inline FleetCapability fleet_capabilities(const std::vector<FleetDevice> &ds) {
    FleetCapability cap;
    uint32_t n_comp = 0;
    uint32_t n_tonly = 0;
    for (auto &d : ds) {
        if (d.compute_family == 0xFFFFFFFFu) continue;
        ++n_comp;
        if (d.transfer_only_family != 0xFFFFFFFFu) ++n_tonly;
        if (d.has_zero_copy_type) cap.igpu_zero_copy = true;
        if (d.type == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU && d.has_zero_copy_type)
            cap.igpu_zero_copy = true;
    }
    cap.n_compute_devices = n_comp;
    cap.n_devices_with_transfer_only = n_tonly;
    cap.multi_card_layer_split = n_comp >= 2;
    cap.transfer_only_per_card = n_comp >= 2 && n_tonly == n_comp;
    bool all_shared = n_comp > 0;
    for (auto &d : ds) if (d.compute_family != 0xFFFFFFFFu && !d.has_shared_heap)
        all_shared = false;
    cap.shared_host_heap = all_shared && n_comp >= 2 &&
        std::all_of(ds.begin(), ds.end(), [](const FleetDevice &d) {
            return d.ext_external_memory_host;
        });
    if (!cap.multi_card_layer_split) {
        // Say what was counted, not what the machine is: the count is the measurement, the
        // explanation is ours. ds.size() and n_comp are printed so a reader can check the sentence
        // against the enumeration table above it.
        cap.why_multi = std::to_string(n_comp) + " COMPUTE-capable device(s) of " +
                        std::to_string(ds.size()) + " physical device(s) visible to this loader - " +
                        "NOT EXERCISABLE HERE: a layer pipeline needs two independent command " +
                        "streams, so the schedule, the ring and the fences are compiled and offline-" +
                        "tested but no PCIe-to-PCIe hop has been taken on this host";
    }
    if (!cap.igpu_zero_copy)
        cap.why_zero = "no device reports a DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT memory type, "
                       "and no device is INTEGRATED_GPU - the zero-copy branch would refuse";
    return cap;
}

static inline void fleet_probe(VkInstance inst) {
    std::vector<FleetDevice> ds;
    if (fleet_enumerate(inst, ds) != VK_SUCCESS) {
        fprintf(stderr, "fleet probe: enumeration failed\n");
        return;
    }
    printf("\n== gpu fleet topology (no logical device created - enumerate + classify only) ==\n");
    printf("  %zu physical device(s)\n", ds.size());
    char qs[80];
    for (auto &d : ds) {
        printf("\n  [%u] %s | %s | vendor 0x%X device 0x%X | api %u.%u.%u | driver %u.%u.%u\n",
               d.index, d.name, fleet_type_name(d.type), d.vendor_id, d.device_id,
               VK_VERSION_MAJOR(d.api_version), VK_VERSION_MINOR(d.api_version),
               VK_VERSION_PATCH(d.api_version),
               VK_VERSION_MAJOR(d.driver_version), VK_VERSION_MINOR(d.driver_version),
               VK_VERSION_PATCH(d.driver_version));
        printf("      device-local heap %.1f MiB | host-visible heap %.1f MiB | unified (both at "
               "once) heap %.1f MiB | timestampPeriod %.3f ns | maxMemoryAllocationCount %u | "
               "nonCoherentAtomSize %llu B | host-import alignment %llu B\n",
               double(d.device_local_heap) / 1048576.0, double(d.host_visible_heap) / 1048576.0,
               double(d.shared_heap) / 1048576.0, d.timestamp_period_ns, d.max_mem_allocs,
               (unsigned long long)d.noncoherent_atom_size,
               (unsigned long long)d.min_imported_host_align);
        std::string zc = d.has_zero_copy_type
                             ? ("yes, type " + std::to_string(d.zero_copy_type))
                             : std::string("NO - zero-copy arena would REFUSE here");
        printf("      zero-copy (DEVICE_LOCAL|HOST_VISIBLE|HOST_COHERENT at once) type = %s | "
               "ext: %s %s %s\n",
               zc.c_str(),
               d.ext_external_memory_host ? "external_memory_host" : "-",
               d.ext_external_memory_win32 ? "external_memory_win32" : "-",
               d.ext_external_memory_fd ? "external_memory_fd" : "-");
        for (auto &q : d.queues) {
            fleet_queue_flags_str(q.flags, qs, sizeof qs);
            printf("      queue family %u: flags=0x%X %s tiler=%u,%u,%u contexts=%u  %s\n",
                   q.index, q.flags, qs, q.tiler.width, q.tiler.height, q.tiler.depth,
                   q.queue_count,
                   q.transfer_only ? "<- TRANSFER-ONLY (SDMA): async hops possible" : "");
        }
        fleet_print_memory_types(d.mp, "memory types:");
    }

    // Physical device groups: the only query that says whether the driver thinks two handles are
    // linked (NVLink / a coherent mesh). Fetched dynamically and NULL-checked - the rule this whole
    // file follows because a NULL from a 1.1 instance is a recorded amdvlk SIGSEGV, not a curiosity.
    auto pfn_groups = reinterpret_cast<PFN_vkEnumeratePhysicalDeviceGroups>(
        vkGetInstanceProcAddr(inst, "vkEnumeratePhysicalDeviceGroups"));
    if (!pfn_groups) {
        printf("\n  vkEnumeratePhysicalDeviceGroups -> NULL: not called. Group/link information is\n"
               "  unavailable, so \"the two cards share a heap\" could not be confirmed by this\n"
               "  query even on a machine that had two cards.\n");
    } else {
        uint32_t ng = 0;
        if (pfn_groups(inst, &ng, nullptr) == VK_SUCCESS && ng) {
            std::vector<VkPhysicalDeviceGroupProperties> gp(ng);
            for (auto &g : gp) g.sType = VK_STRUCTURE_TYPE_PHYSICAL_DEVICE_GROUP_PROPERTIES;
            pfn_groups(inst, &ng, gp.data());
            printf("\n  %u physical device group(s)\n", ng);
            for (uint32_t g = 0; g < ng; ++g) {
                // `subsetAllocation` (not "allocationControl"): the driver's answer to "may a
                // suballocation be spread across the group". A group of 1 is what a machine without
                // any link reports, and printing the count is what makes that visible instead of
                // assuming it.
                printf("    group %u: %u device(s), subsetAllocation=%d\n", g,
                       gp[g].physicalDeviceCount, int(gp[g].subsetAllocation));
            }
        } else {
            printf("\n  no device groups reported\n");
        }
    }

    FleetCapability cap = fleet_capabilities(ds);
    printf("\n== capability verdicts for THIS host ==\n");
    printf("  (A) layer-pipelined multi-GPU : %s\n", cap.multi_card_layer_split ? "AVAILABLE"
                                                                                : "NOT AVAILABLE");
    printf("      %s\n", cap.multi_card_layer_split
        ? "more than one COMPUTE device; the ring, the fences and the schedule are all compiled, but "
          "no measurement of a real PCIe-to-PCIe hop has been taken on this host"
        : cap.why_multi.c_str());
    // Three-valued on purpose. "every compute device has its own transfer-only queue" is a FLEET
    // property; on a one-device host it is unevaluable, and the fleet answer must not be allowed to
    // read as a claim about this card - the table above shows this card's family 2 is exactly that
    // queue. The measured cost of NOT using it is single-card evidence and stays quoted as such:
    // DEEPINFER_EVAL.md / HANDOVER.md 418 (BONSAI_XFER) - D2H 7.15 -> 3.89 GB/s, H2D 7.12 -> 4.93.
    std::string tof;
    if (cap.n_compute_devices >= 2 && cap.transfer_only_per_card) {
        tof = "yes (" + std::to_string(cap.n_devices_with_transfer_only) + "/" +
              std::to_string(cap.n_compute_devices) + ") - every hop can go on its own queue";
    } else if (cap.n_compute_devices >= 2) {
        tof = "NO (" + std::to_string(cap.n_devices_with_transfer_only) + "/" +
              std::to_string(cap.n_compute_devices) + "): a hop on those cards would serialise "
              "behind compute, MEASURED on this driver as D2H 7.15 -> 3.89 GB/s and H2D 7.12 -> "
              "4.93 when the same copy rides the compute queue";
    } else {
        tof = "UNEVALUABLE for a fleet (needs >= 2 compute devices; here " +
              std::to_string(cap.n_compute_devices) + "). Per-device on THIS card: " +
              std::to_string(cap.n_devices_with_transfer_only) + "/" +
              std::to_string(cap.n_compute_devices) + " compute device(s) expose a transfer-only "
              "family - see the queue table above. The queue choice itself is measured, not "
              "assumed: a copy on the compute queue runs D2H 7.15 -> 3.89 GB/s, H2D 7.12 -> 4.93.";
    }
    printf("      transfer-only queue on every compute device: %s\n", tof.c_str());
    printf("      shared unified host heap across cards  : %s\n",
           cap.shared_host_heap ? "candidate (needs the import to succeed at runtime)"
                                : cap.n_compute_devices >= 2
                                    ? "no - hops cost a real copy or a host memcpy each"
                                    : "unevaluable with one compute device; per-device the unified "
                                      "(DEVICE_LOCAL|HOST_VISIBLE) heap is printed above");
    printf("  (B) iGPU zero-copy arena        : %s\n", cap.igpu_zero_copy ? "AVAILABLE"
                                                                           : "NOT AVAILABLE");
    printf("      %s\n", cap.igpu_zero_copy
        ? "a device offers a heap that is host- and device-accessible at once; the arena can be "
          "mapped and written directly"
        : cap.why_zero.c_str());
}

#endif  // BONSAI_GPU_FLEET_H
