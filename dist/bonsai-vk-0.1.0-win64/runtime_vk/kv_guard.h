// kv_guard.h - P1's q8_0 capacity math and P2's WDDM spill defence.
//
// WHAT IS VERIFIED HERE AND WHAT IS NOT, stated up front because the whole point of this file is
// that the two must never be confused in a report:
//   * the byte arithmetic is DERIVED from the layout the shaders actually write (see KvGeom) and is
//     cross-checked against the arena printf in bonsai_forward.h at every init;
//   * the PDH counter is a WINDOWS API contract, and the numbers it returns are measured on this
//     machine at run time, not asserted here;
//   * the eviction/refusal policy is a real code path with real counters, and this file says in plain
//     words what it can and cannot buy on this card.
//
// ---------------------------------------------------------------- why Shared Usage and not Dedicated
// Under WDDM the process-visible pool is not the physical VRAM. A device-local allocation the driver
// cannot keep resident is *evicted* by VidMm - it is moved to system memory and the process's
// "Shared Usage" (the part of its GPU-memory footprint that is not exclusively resident) grows while
// "Dedicated Usage" stays flat and while every VkMemoryAllocate success is still VK_SUCCESS. That is
// the exact shape of the failure this project has already been bitten by once (see HANDOVER: the
// 4 GB wall that first showed up as a compute-buffer failure, not as an OOM). So:
//   Shared Usage   = the spill judge. Monotonic growth across decode boundaries = eviction in flight.
//   Dedicated Usage= informational. It is the exclusive-residency slice and it is the counter a
//                    budget-based heuristic would wrongly use. Using it as a spill judge is banned in
//                    the spec and this file does not offer it as one - it is sampled only to be
//                    printed next to the number that decides.
//   IDXGIAdapter3::QueryVideoMemoryInfo = a BUDGET query. The measured struct (mingw-w64
//                    dxgi1_4.h) is {Budget, CurrentUsage, AvailableForReservation,
//                    CurrentReservation} - there is NO "available for commit" field, so the honest
//                    reading is "what the driver will commit and how much it will let us reserve".
//                    It is useful for choosing cap at init. It says nothing about
//                    eviction of already-committed pages, so it is NOT a spill judge and it is only
//                    ever used here as the auxiliary init-time reading.
//
// ---------------------------------------------------------------- the same three roles on Linux
// There is no PDH and no DXGI on Linux, but the ROLES above are not Windows-specific. amdgpu's TTM
// moves a BO it cannot keep resident in VRAM into GTT (the system-memory aperture), which is exactly
// the destination WDDM's "Shared Usage" counts. So the substitution is:
//   mem_info_gtt_used   = the spill judge   (the analogue of Shared Usage)
//   mem_info_vram_used  = informational     (the analogue of Dedicated Usage: residency, NOT a judge)
//   mem_info_vram_total = the budget        (the analogue of DXGI's LOCAL Budget)
// vram_used/vram_total are what a VRAM meter actually wants and both are read and printed here. But
// vram_used must never become the judge, for the same reason Dedicated Usage must not: it counts what
// is resident, and residency staying flat while the process pages is precisely the failure watched for.
// ONE HONEST DIFFERENCE. sysfs is DEVICE-WIDE, not per-process. PDH's counter is filtered by pid;
// mem_info_gtt_used is the whole card's GTT footprint, so a second GPU process on the same card can
// move it. The project's standing rule is one GPU process at a time, which is what makes the reading
// usable, and the printed line names the card so the reader can see what was measured. BONSAI_DRM_CARD
// overrides the discovery; otherwise /sys/class/drm is scanned for a cardN that exposes the counters.
// macOS has neither sysfs nor these counters: open() says why and the watch is inert, which is the same
// graceful degradation the Windows branch already has when the driver has no PDH instance.
// STATUS: the POSIX branch is written against the documented sysfs contract and HAS NOT BEEN COMPILED
// OR RUN - the development machine is Windows and its WSL has no distribution installed. The claim is
// "Linux 编译支持已就绪，待实机验证", never "measured".

// ---------------------------------------------------------------- what the block hook can and cannot do
// The arenas are indexed by ABSOLUTE position (norm_rope writes at `position`, qk/pv read 0..pos), so
// a KV block can be taken out of the *model* - its scores go to -INF, its probabilities go to 0, and
// it can never be read again - but taking it out does not return its bytes to the allocator. True
// reclaim needs the ring-buffer arena rewrite (positions would become slots mod cap, with the K/V
// kernels re-indexed and the whole BONSAI_QCHECK bitwise contract re-measured), which is out of scope
// for this change and is NOT pretending to happen here. What the hook therefore does:
//   REFUSE  - stop accepting new positions: generation halts at the watermark with an explicit
//             message and a recommended BONSAI_CAP computed from the same formula as the init OOM
//             branch. No dispatch is recorded past it, so no half-written cache state exists.
//   EVICT   - -INF-mask the farthest still-live block (default OFF: it changes what the model can
//             attend to, so it is opt-in via BONSAI_KV_EVICT=1 and it never evicts the last block).
//   FALLBACK- at init, before anything is dispatched, the geometry chooses the largest cap that fits
//             the measured budget instead of failing (the pre-existing OOM branch, now mode-aware).
// The honest summary: on this card the hook's value is (a) a hard stop before the driver starts
// paging, (b) an instrumented signal that says WHY it stopped, and (c) the API shape a ring-buffer
// arena can plug into later.

#ifndef BONSAI_KV_GUARD_H
#define BONSAI_KV_GUARD_H

#include "bonsai_platform.h"   // FIRST, and deliberately: on Windows it pulls winsock2.h in before
                               // windows.h, and the windows.h below must not be the first one seen.
#ifdef _WIN32
#include <pdh.h>
#include <pdhmsg.h>   // PDH_CSTATUS_VALID_DATA lives in pdhmsg.h, not pdh.h (measured: without this
                      // include MinGW says "was not declared in this scope")
#include <dxgi1_4.h>    // in mingw-w64 IDXGIAdapter3 is declared here, not in dxgi1_3.h
                        // (dxgi1_3.h ends at IDXGIOutput3); __CRT_UUID_DECL(IDXGIAdapter3) is at
                        // dxgi1_4.h:1148, so __uuidof works without hardcoding the GUID.
#endif                  // POSIX needs nothing beyond what bonsai_platform.h already included

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

// ---------------------------------------------------------------- the geometry
// One attention layer, one token slot, in BYTES - computed from the layout the two kernels write, not
// typed in. q8_0 means 32 elements per block with one fp16 scale per block, which is ggml's QK8_0.
struct KvGeom {
    uint32_t nkv = 4, hd = 256, n_attn = 8;
    uint32_t blk = 32;              // elements per q8_0 block (the format's own number)
    uint32_t v_q8 = 1;              // 0 = V stays fp16, 1 = V is a q8_0 block array
    // per layer, per token slot, in uint (4-byte) words - what Regions::take() counts in
    uint64_t w_k = 0, w_vd = 0, w_vs = 0, w_layer = 0;
    uint64_t bytes_slot = 0;        // x n_attn: the real per-token KV cost
    uint32_t nblk_head = 0;         // blocks (and scales) per head row
    uint32_t evict_blk_slots = 64;  // the granularity the refuse/evict hook works in

    // The three divisibility rules the shaders depend on. Each one is a silent-corruption class if it
    // ever stops holding: an odd number of scale words per head row would make one fp16 scale straddle
    // two heads (the packer writes 2 per uint), a hd not divisible by 4 would split an element across
    // two data words, and a hd not divisible by 32 would leave a block straddling a head boundary so
    // the butterfly reduction and the scale index disagree about what a block is.
    bool check(std::string &why) const {
        char m[192];
        if (hd % 4u)  { snprintf(m, sizeof m, "hd %% 4 == %u: a q8_0 data word would straddle elements", hd % 4u);  why = m; return false; }
        if (hd % blk) { snprintf(m, sizeof m, "hd %% blk == %u: a block would straddle head rows", hd % blk); why = m; return false; }
        if ((hd / blk) % 2u) { snprintf(m, sizeof m, "hd/blk == %u is odd: two-per-word scale packing would straddle a head", hd / blk); why = m; return false; }
        if (blk != 32u)      { snprintf(m, sizeof m, "block size %u is not ggml's QK8_0 = 32", blk); why = m; return false; }
        why.clear(); return true;
    }

    void compute() {
        nblk_head = hd / blk;
        w_k  = uint64_t(nkv) * hd / 2u;               // fp16: 2 halves per uint
        if (v_q8) {
            w_vd = uint64_t(nkv) * hd / 4u;           // int8: 4 bytes per uint
            w_vs = uint64_t(nkv) * nblk_head / 2u;    // fp16 scales, 2 per uint
            w_layer = w_k + w_vd + w_vs;
        } else {
            w_vd = uint64_t(nkv) * hd / 2u;           // fp16 V arena (the pre-P1 layout)
            w_vs = 0;
            w_layer = w_k + w_vd;
        }
        bytes_slot = w_layer * 4ull * n_attn;
    }

    void report() const {
        printf("  KV geometry (derived from the kernel layout, not typed):\n");
        printf("    K  fp16          %6llu words/slot/layer = %6llu B   (never compressed)\n",
               (unsigned long long)w_k, (unsigned long long)(w_k * 4));
        if (v_q8)
            printf("    V q8_0 data    %6llu words/slot/layer = %6llu B   (%u blocks of %u per head row)\n"
                   "    V q8_0 scales  %6llu words/slot/layer = %6llu B   (1 fp16 per %u-element block)\n",
                   (unsigned long long)w_vd, (unsigned long long)(w_vd * 4), nblk_head, blk,
                   (unsigned long long)w_vs, (unsigned long long)(w_vs * 4), blk);
        else
            printf("    V fp16         %6llu words/slot/layer = %6llu B   (BONSAI_KV_V_QUANT=none)\n",
                   (unsigned long long)w_vd, (unsigned long long)(w_vd * 4));
        printf("    per layer-slot %8llu B -> x%u attention layers = %llu B/token = %.2f KiB/token\n",
               (unsigned long long)(w_layer * 4), n_attn, (unsigned long long)bytes_slot,
               double(bytes_slot) / 1024.0);
    }
};

// ---------------------------------------------------------------- the watcher
// One policy, two instruments. Everything from `soft_mib` down to `curve_at` - the thresholds, the
// reference sample, the monotonic-run state, the curve - is PLATFORM-NEUTRAL and is the same code on
// both. Only the thing that produces a number differs: PDH on Windows, amdgpu sysfs on Linux. The two
// readers are exposed under the same names (read_judge / read_info) so sample() has no #ifdef in it.
struct SpillWatch {
#ifdef _WIN32
    PDH_HQUERY  q = nullptr;
    PDH_HCOUNTER c_shared = nullptr, c_dedicated = nullptr;
    std::vector<char> arr;        // reusable PdhGetFormattedCounterArrayA buffer (no per-sample malloc)
#else
    std::string sys_dir;          // /sys/class/drm/cardN/device - chosen once at open()
    uint64_t vram_total = 0;      // the budget, read once: it does not change while we run
#endif
    bool opened = false, counter_ok = false;
    bool wildcard = false;        // Windows: the counter had to be added as "GPU Process Memory(*)\..."
    uint32_t pid = 0;
    std::string err;

    // The field names in the printed sample line. On Windows these are "shared"/"dedicated" and the
    // line is byte-identical to the one every log in runs/ quotes; on Linux they name the sysfs file
    // that produced the number, because printing "shared" next to a GTT reading would be a lie.
#ifdef _WIN32
    static const char *judge_label() { return "shared"; }
    static const char *info_label()  { return "dedicated"; }
    static const char *abs_label()   { return "ABSOLUTE Shared Usage (BONSAI_SPILL_ABS=1)"; }
    static const char *judge_api()   { return "PDH"; }
    static const char *judge_source(){ return "PDH Shared Usage"; }
    static const char *judge_scope() { return "for this pid"; }
#else
    static const char *judge_label() { return "gtt_used"; }
    static const char *info_label()  { return "vram_used"; }
    static const char *abs_label()   { return "ABSOLUTE GTT usage (BONSAI_SPILL_ABS=1)"; }
    static const char *judge_api()   { return "amdgpu sysfs"; }
    static const char *judge_source(){ return "amdgpu mem_info_gtt_used"; }
    static const char *judge_scope() { return "for this card (sysfs is device-wide)"; }
#endif

    // policy, all of it env-overridable so the thresholds are testable without spilling a 4 GB card
    double soft_mib = 32.0;       // "start looking" - logs, does not act
    double hard_mib = 64.0;       // "this is eviction" - acts
    uint32_t every = 32;          // sample every N decode tokens AND at every phase boundary
    uint32_t mono_need = 3;       // consecutive increases that count as monotonic growth
    // WHAT the thresholds are measured against. The spec's own words are "monotonic GROWTH past a
    // 32-64 MB threshold", and on this card the absolute number is useless: a clean q8_0 run sits at a
    // flat 40.93 MiB of Shared Usage from the moment the weights are resident (measured, runs/
    // p2_rate_q8_reuse_v2.log), so an absolute 32 MiB threshold flags every single token of a
    // perfectly healthy run - that is exactly what "soft-watched 48" in runs/short_ctx_q8_0_v3.log is.
    // So the default reference is the FIRST decode sample and the thresholds judge growth over it,
    // which is the quantity VidMm eviction actually shows up in. BONSAI_SPILL_ABS=1 restores the
    // absolute reading for reproducing the earlier A/B logs.
    uint32_t use_absolute = 0;
    double ref_shared = -1.0;     // the zero of the growth axis (first sample that returned data)

    // state
    double last_shared = -1.0, peak_shared = 0.0;
    double last_growth = 0.0, peak_growth = 0.0;   // what the policy actually judges
    uint32_t run_up = 0;          // consecutive increases
    uint32_t n_sample = 0, n_soft = 0, n_hard = 0, n_grow = 0;
    uint32_t n_attempt = 0;       // samples tried, including the ones the counter had no data for
    std::vector<double> curve;    // shared MiB per sample, for the printed curve
    std::vector<uint32_t> curve_at;   // token position of each sample

#ifdef _WIN32
    // One counter path, with the status it returned. 0 == accepted. PdhAddEnglishCounter* on purpose:
    // the OS is zh-CN and PdhAddCounter would have to match the LOCALISED name; the English API
    // resolves the English path through the driver's counter metadata, which is what makes the same
    // string work on an English and a Chinese machine.
    DWORD add_(const char *path, PDH_HCOUNTER &out) {
        out = nullptr;
        return PdhAddEnglishCounterA(q, path, 0, &out);
    }
    static const char *tail(const char *p) {
        const char *s = strrchr(p, '\\');            // the LAST one: the path now starts with "\"
        return s ? s : p;
    }
    // THE PATH FORM IS NOT COSMETIC. Measured on this machine with tools/pdh_probe.cpp
    // (runs/pdh_probe_self_v1.log):
    //   "\GPU Process Memory(*)\Shared Usage"                     add -> OK,  collect -> OK,
    //                                                                 20 instances listed
    //   "GPU Process Memory(*)\Shared Usage"        (no backslash) add -> 0xC0000BC0
    //   "GPU Process Memory(pid_<self>_*)\Shared Usage"            add -> 0xC0000BC0
    //   "\GPU Process Memory\Shared Usage"          (no instance)  add -> 0xC0000BC0
    // 0xC0000BC0 is PDH_CSTATUS_BAD_COUNTERNAME, i.e. the NAME was rejected - which is why retrying
    // the read at sample time could never have helped. The instance part is mandatory, and the
    // leading backslash is mandatory, and PdhExpandWildCardPathA without it returns 0xC0000BC4
    // (PDH_INVALID_PATH). Get-Counter has always written the backslash; that is the only reason
    // tools/vram_sample.ps1 works and this file did not.
    // Star first, pid-scoped second: the pid form resolves to PDH_CSTATUS_NO_DATA (0x800007D5) for as
    // long as the process has no GPU instance, and the star form answers the whole list from the first
    // collect, so read_mib() filters by pid instead of asking the driver to have registered us yet.
    // Measured AMD instance names are
    //   pid_<pid>_luid_0x00000000_0x0001108F_phys_0   (and some pids also have a 0x00012652 LUID)
    // so ONE PROCESS CAN HAVE MORE THAN ONE INSTANCE - which is why read_mib() sums them instead of
    // picking the first hit, and why the sum, not any single instance, is the process's footprint.
    bool add_counter(const char *counter, PDH_HCOUNTER &out, std::string &why) {
        char path[160];
        snprintf(path, sizeof path, "\\GPU Process Memory(*)\\%s", counter);
        const DWORD s2 = add_(path, out);
        if (s2 == 0) { wildcard = true; return true; }
        snprintf(path, sizeof path, "\\GPU Process Memory(pid_%u_*)\\%s", pid, counter);
        DWORD s1 = add_(path, out);
        if (s1 == 0) { wildcard = false; return true; }
        char b[128];
        snprintf(b, sizeof b, "%s: star-path 0x%08lX, pid-path 0x%08lX", tail(path),
                 (unsigned long)s2, (unsigned long)s1);
        why += b;
        return false;
    }

    bool open(uint32_t pid_) {
        pid = pid_;
        err.clear();
        if (PdhOpenQuery(nullptr, 0, &q) != 0) { err = "PdhOpenQuery failed"; return false; }
        opened = true;
        std::string why;
        if (!add_counter("Shared Usage", c_shared, why)) { err = why; return false; }
        // Informational. A failure to add it must not disable the watcher, and a success must not be
        // used for any decision (see the header note).
        PDH_HCOUNTER d = nullptr;
        std::string w2;
        if (add_counter("Dedicated Usage", d, w2)) c_dedicated = d;
        counter_ok = true;
        printf("  [spill] PDH Shared Usage counter added as %s (sum of every instance whose name"
               " contains \"pid_%u_\")\n",
               wildcard ? "\\GPU Process Memory(*)\\Shared Usage"
                        : "\\GPU Process Memory(pid_*_*)\\Shared Usage", pid);
        return true;
    }

    // Sum of THIS process's instances of one counter, in MiB.
    // The array form is mandatory, not an optimisation: a wildcard counter path resolves to N
    // instances, and even the pid-scoped one can name more than one (the AMD instance name carries a
    // LUID and a process can own two LUIDs - measured above), while PdhGetFormattedCounterValue only
    // ever answers for a single instance and would silently report one slice of the footprint.
    // PDH_FMT_LARGE because these are byte counters.
    bool read_mib(PDH_HCOUNTER c, double &mib) {
        mib = 0.0;
        if (!c || !q) return false;
        if (PdhCollectQueryData(q) != 0) return false;
        char key[24];
        snprintf(key, sizeof key, "pid_%u_", pid);
        for (int pass = 0; pass < 3; ++pass) {
            DWORD bytes = arr.empty() ? 4096u : DWORD(arr.size()), items = 0;
            if (bytes > 8u << 20) return false;                 // pathological instance count: give up
            arr.resize(bytes);
            const LONG r = PdhGetFormattedCounterArrayA(c, PDH_FMT_LARGE, &bytes, &items,
                                    (PPDH_FMT_COUNTERVALUE_ITEM_A)arr.data());
            // (LONG) cast: PDH_MORE_DATA expands through __MSABI_LONG to an unsigned-typed literal on
            // this toolchain, and comparing it to the LONG the API returns is a -Wsign-compare warning.
            if (r == (LONG)PDH_MORE_DATA) { arr.resize(bytes + 1024u); continue; }  // retry at the size asked for
            if (r != 0) return false;
            double sum = 0.0;
            uint32_t hit = 0;
            const PPDH_FMT_COUNTERVALUE_ITEM_A it = (const PPDH_FMT_COUNTERVALUE_ITEM_A)arr.data();
            for (DWORD i = 0; i < items; ++i) {
                // Always pid-filtered, whichever path the counter was added on: the filter is a no-op
                // for a pid-scoped path (every instance it returns is ours) and is the whole point of
                // the "*" fallback. It also makes the per-counter "wildcard" mode irrelevant to the
                // arithmetic, so shared and dedicated can fall back independently.
                if (!strstr(it[i].szName, key)) continue;
                if (it[i].FmtValue.CStatus != PDH_CSTATUS_VALID_DATA) continue;
                sum += double(it[i].FmtValue.largeValue);
                ++hit;
            }
            if (!hit) return false;          // no data yet: the instance appears once the GPU is used
            mib = sum / 1048576.0;
            return true;
        }
        return false;
    }
    // The two roles, named the same on both platforms so sample() stays free of #ifdef.
    bool read_judge(double &mib) { return read_mib(c_shared, mib); }
    bool read_info(double &mib)  { return read_mib(c_dedicated, mib); }

#else  // ------------------------------------------------------------ amdgpu sysfs (Linux)
    // One sysfs attribute, in bytes. These files are a single ASCII integer followed by a newline and
    // are read by the kernel on demand, so fopen/fscanf per sample is the contract, not a shortcut;
    // there is no descriptor to keep open and no buffer to size. `false` means the attribute is not
    // there (a driver without TTM accounting, an iGPU without a VRAM aperture, or macOS), and the
    // caller turns that into an inert watch rather than a zero - a zero would read as "no spill".
    bool read_attr(const char *name, uint64_t &bytes) const {
        bytes = 0;
        if (sys_dir.empty()) return false;
        const std::string p = sys_dir + "/" + name;
        FILE *fp = fopen(p.c_str(), "rb");
        if (!fp) return false;
        unsigned long long v = 0;
        const int n = fscanf(fp, "%llu", &v);
        fclose(fp);
        if (n != 1) return false;
        bytes = uint64_t(v);
        return true;
    }
    static bool has_attr(const std::string &dir, const char *name) {
        const std::string p = dir + "/" + name;
        FILE *fp = fopen(p.c_str(), "rb");
        if (!fp) return false;
        fclose(fp);
        return true;
    }
    // Card discovery. /sys/class/drm holds both "card0" (the device) and "card0-DP-1" (a connector),
    // and only the device directory has a `device/` child with the memory attributes - so a name with
    // a '-' in it is skipped rather than probed. BONSAI_DRM_CARD pins the choice, because on a box with
    // an iGPU plus a discrete card "card0" is not necessarily the one Vulkan ended up on, and a
    // silently wrong card would produce a plausible-looking curve for the wrong device.
    static bool pick_dir(std::string &dir, std::string &why) {
        const char *want = getenv("BONSAI_DRM_CARD");
        if (want && *want) {
            const std::string d = std::string("/sys/class/drm/") + want + "/device";
            if (has_attr(d, "mem_info_gtt_used")) { dir = d; return true; }
            why = std::string("BONSAI_DRM_CARD=") + want + " has no mem_info_gtt_used";
            return false;
        }
        DIR *dp = opendir("/sys/class/drm");
        if (!dp) { why = "/sys/class/drm is absent (not Linux, or no DRM driver)"; return false; }
        std::string first;
        while (const dirent *de = readdir(dp)) {
            const std::string n = de->d_name;
            if (n.compare(0, 4, "card") != 0 || n.find('-') != std::string::npos) continue;
            const std::string d = "/sys/class/drm/" + n + "/device";
            if (!has_attr(d, "mem_info_gtt_used")) continue;
            if (first.empty()) first = d;
        }
        closedir(dp);
        if (first.empty()) { why = "no cardN/device/mem_info_gtt_used under /sys/class/drm (amdgpu "
                                   "sysfs accounting absent on this driver)"; return false; }
        dir = first;
        return true;
    }

    bool open(uint32_t pid_) {
        pid = pid_;
        err.clear();
        std::string why;
        if (!pick_dir(sys_dir, why)) { err = why; sys_dir.clear(); return false; }
        opened = true;
        uint64_t g = 0;
        if (!read_attr("mem_info_gtt_used", g)) {
            err = sys_dir + "/mem_info_gtt_used is unreadable";
            counter_ok = false;
            return false;
        }
        uint64_t t = 0;
        read_attr("mem_info_vram_total", t);       // informational; a missing total is not fatal
        vram_total = t;
        counter_ok = true;
        printf("  [spill] amdgpu sysfs judge = %s/mem_info_gtt_used (DEVICE-WIDE, not pid-filtered;"
               " informational vram_used, budget vram_total %.2f MiB)\n",
               sys_dir.c_str(), double(vram_total) / 1048576.0);
        return true;
    }
    bool read_judge(double &mib) {
        uint64_t v = 0;
        if (!read_attr("mem_info_gtt_used", v)) return false;
        mib = double(v) / 1048576.0;
        return true;
    }
    bool read_info(double &mib) {
        uint64_t v = 0;
        if (!read_attr("mem_info_vram_used", v)) return false;
        mib = double(v) / 1048576.0;
        return true;
    }
#endif

    // One sample. Returns 0 = fine, 1 = soft (growing, past the look-at threshold), 2 = hard breach.
    // `pos` only labels the sample. The decision is taken on `last_growth` unless BONSAI_SPILL_ABS=1.
    int sample(uint32_t pos) {
        double s = 0.0;
        ++n_attempt;
        if (!counter_ok || !read_judge(s)) return 0;
        double d = 0.0; read_info(d);
        ++n_sample;
        if (ref_shared < 0.0) ref_shared = s;                 // the first sample defines the zero
        last_shared = s;
        last_growth = use_absolute ? s : s - ref_shared;
        if (last_growth > peak_growth) peak_growth = last_growth;
        curve.push_back(s); curve_at.push_back(pos);
        if (n_sample > 1) {
            if (s > curve[curve.size() - 2] + 1e-9) { if (++run_up >= mono_need) ++n_grow; }
            else run_up = 0;
        }
        if (s > peak_shared) peak_shared = s;
        int level = 0;
        if (last_growth > soft_mib) { ++n_soft; level = 1; }
        if (last_growth > hard_mib && run_up >= mono_need) { ++n_hard; level = 2; }
        // The field names come from the instrument, not from a literal: on Windows they substitute to
        // "shared"/"dedicated" and this line is byte-identical to the one runs/*.log quotes.
        printf("    [spill] tok %-6u %s %8.2f MiB (growth %+7.2f over the %s %8.2f,"
               " %s %8.2f MiB, growth run %u)%s\n",
               pos, judge_label(), s, last_growth,
               use_absolute ? "absolute zero 0.00" : "reference", ref_shared, info_label(), d, run_up,
               level == 2 ? "  <- HARD: monotonic growth past the eviction threshold"
                          : level == 1 ? "  <- soft threshold" : "");
        return level;
    }

    // The monotonic-run verdict the policy needs alongside the threshold: "has the counter gone up
    // mono_need times in a row". Same definition used for the n_hard sample counter, so the guard and
    // the curve can never disagree about whether a growth run exists.
    bool growing() const { return run_up >= mono_need; }

    void print_summary() const {
#ifdef _WIN32
        printf("\n=== PDH 'GPU Process Memory\\Shared Usage' for this pid (the spill judge) ===\n");
#else
        printf("\n=== amdgpu sysfs 'mem_info_gtt_used' for %s (the spill judge) ===\n",
               sys_dir.empty() ? "no card" : sys_dir.c_str());
#endif
        if (!counter_ok) { printf("  counter unavailable (%s); the watch was inert this run\n",
                                 err.empty() ? "not opened" : err.c_str()); return; }
        printf("  thresholds judge %s; reference sample %.2f MiB, soft > %.0f MiB, hard > %.0f MiB"
               " with a growth run of >= %u\n",
               use_absolute ? abs_label() : "GROWTH over the first decode sample",
               ref_shared, soft_mib, hard_mib, mono_need);
        printf("  samples %u of %u tried, peak absolute %.2f MiB, peak growth %+.2f MiB,"
               " soft %u, hard %u, monotonic-run(>=%u increases) %u\n",
               n_sample, n_attempt, peak_shared, peak_growth, n_soft, n_hard, mono_need, n_grow);
        if (!n_sample)
#ifdef _WIN32
            printf("  no sample ever returned data: the driver had no 'GPU Process Memory' instance"
                   " for this pid during the run (it appears only once the GPU context is registered),"
                   " so NO spill verdict can be made from this run\n");
#else
            printf("  no sample ever returned data: %s/mem_info_gtt_used could not be read during the"
                   " run, so NO spill verdict can be made from this run\n",
                   sys_dir.empty() ? "/sys/class/drm/cardN/device" : sys_dir.c_str());
#endif
        const size_t n = curve.size();
        if (!n) return;
        printf("  curve (every sample, MiB):");
        for (size_t i = 0; i < n; ++i) printf(" %u:%.1f", curve_at[i], curve[i]);
        printf("\n");
    }

#ifdef _WIN32
    void close() { if (opened && q) { PdhCloseQuery(q); q = nullptr; opened = false; } }
#else
    void close() { sys_dir.clear(); opened = false; counter_ok = false; }
#endif
};

// ---------------------------------------------------------------- the auxiliary budget query
// NOT a spill judge - see the header. Used once at init to print what the driver will commit, which
// is the number the OOM fallback's formula should agree with. Purely informational on both platforms:
// no field of it is read by the policy, so the two implementations only have to agree on
// open()/report()/close().
#ifdef _WIN32
struct DxgiBudget {
    IDXGIFactory1 *fac = nullptr;
    IDXGIAdapter3 *ad3 = nullptr;
    bool ok = false;

    void open() {
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void **)&fac))) return;
        IDXGIAdapter *ad = nullptr;
        if (FAILED(fac->EnumAdapters(0, &ad))) { fac->Release(); fac = nullptr; return; }
        const HRESULT hr = ad->QueryInterface(__uuidof(IDXGIAdapter3), (void **)&ad3);
        ad->Release();
        if (FAILED(hr) || !ad3) { fac->Release(); fac = nullptr; return; }
        ok = true;
    }
    // LOCAL = the video-memory segment, NON_LOCAL = the system-memory side of the same budget.
    void report() const {
        if (!ok) { printf("  dxgi: IDXGIAdapter3 unavailable; budget line skipped\n"); return; }
        static const struct { DXGI_MEMORY_SEGMENT_GROUP g; const char *n; } sg[2] = {
            {DXGI_MEMORY_SEGMENT_GROUP_LOCAL, "LOCAL (VRAM)"},
            {DXGI_MEMORY_SEGMENT_GROUP_NON_LOCAL, "NON_LOCAL (system)"}
        };
        for (const auto &x : sg) {
            DXGI_QUERY_VIDEO_MEMORY_INFO i{};
            if (SUCCEEDED(ad3->QueryVideoMemoryInfo(0, x.g, &i)))
                printf("  dxgi budget %-18s: budget %.1f MiB, current usage %.1f MiB,"
                       " available for reservation %.1f MiB, current reservation %.1f MiB"
                       "   (AUXILIARY: a reservation budget, not an eviction signal)\n",
                       x.n, double(i.Budget) / 1048576.0, double(i.CurrentUsage) / 1048576.0,
                       double(i.AvailableForReservation) / 1048576.0,
                       double(i.CurrentReservation) / 1048576.0);
        }
    }
    void close() { if (ad3) ad3->Release(); if (fac) fac->Release(); ad3 = nullptr; fac = nullptr; }
};
typedef DxgiBudget VramBudget;
#else
// The amdgpu analogue. sysfs has no "budget" and no "reservation" concept - it reports the size of the
// two apertures and how much of each is in use - so this prints exactly that and does NOT invent a
// reservation column to make the line look like the DXGI one. `gtt_total` is the system-memory
// aperture the judge counts into, which is the number that says how much room eviction has.
struct SysfsBudget {
    std::string dir;
    bool ok = false;
    uint64_t vram_total = 0, vram_used = 0, gtt_total = 0, gtt_used = 0;

    static bool rd(const std::string &d, const char *name, uint64_t &v) {
        v = 0;
        FILE *fp = fopen((d + "/" + name).c_str(), "rb");
        if (!fp) return false;
        unsigned long long x = 0;
        const int n = fscanf(fp, "%llu", &x);
        fclose(fp);
        if (n != 1) return false;
        v = uint64_t(x);
        return true;
    }
    void open() {
        std::string why;
        if (!SpillWatch::pick_dir(dir, why)) { dir.clear(); return; }
        ok = rd(dir, "mem_info_vram_total", vram_total);
        rd(dir, "mem_info_vram_used", vram_used);
        rd(dir, "mem_info_gtt_total", gtt_total);
        rd(dir, "mem_info_gtt_used", gtt_used);
    }
    void report() const {
        if (!ok) { printf("  sysfs: amdgpu memory attributes unavailable; budget line skipped\n"); return; }
        printf("  sysfs budget %-18s: total %.1f MiB, current usage %.1f MiB"
               "   (AUXILIARY: an aperture size, not an eviction signal)\n",
               "VRAM", double(vram_total) / 1048576.0, double(vram_used) / 1048576.0);
        printf("  sysfs budget %-18s: total %.1f MiB, current usage %.1f MiB"
               "   (AUXILIARY: the aperture TTM evicts into)\n",
               "GTT (system)", double(gtt_total) / 1048576.0, double(gtt_used) / 1048576.0);
    }
    void close() { dir.clear(); ok = false; }
};
typedef SysfsBudget VramBudget;
#endif

// ---------------------------------------------------------------- the block-level policy
enum KvAction { KV_OK = 0, KV_WARN = 1, KV_REFUSE = 2, KV_EVICT = 3 };

struct KvGuard {
    KvGeom g;
    uint32_t cap = 0;               // slots in the arena
    uint32_t blk_slots = 64;        // slots per evictable/refusable block
    uint32_t n_blk = 0;
    uint64_t bytes_per_blk = 0;     // one block of positions, across every attention layer
    uint64_t ceiling_bytes = 0;     // what the KV part may reach, from the init budget
    int allow_evict = 0;            // BONSAI_KV_EVICT=1
    // state
    uint32_t live_low = 0;          // lowest still-attended position (eviction moves it up)
    // One counter per ACTION the guard took. These are deliberately separate from SpillWatch's
    // n_soft/n_hard: those count SAMPLES that crossed a threshold, these count the policy decisions
    // taken at a block boundary. A run can take a refusal without a single hard sample (the geometry
    // ceiling path, or a hard sample that lacked the monotonic run), and printing only the second
    // number is how "refused but hard=0" got reported as a contradiction.
    uint32_t n_refuse = 0, n_evict = 0, n_warn = 0;
    uint64_t bytes_at_refuse = 0;

    void init(const KvGeom &geom, uint32_t cap_, uint64_t kv_ceiling_bytes, int evict) {
        g = geom; cap = cap_; allow_evict = evict;
        n_blk = cap / blk_slots;
        bytes_per_blk = uint64_t(blk_slots) * g.bytes_slot;
        ceiling_bytes = kv_ceiling_bytes;
    }

    // thresholds live here so decide() has them without threading the whole SpillWatch in; they are
    // copied from SpillWatch at init, and both sides are env-overridable.
    double g_spill_soft = 32.0, g_spill_hard = 64.0;

    // The one decision point, called at every block boundary with the current spill-watch metric
    // (`value` = growth over the reference sample by default, absolute Shared Usage under
    // BONSAI_SPILL_ABS=1; the watch and the guard must judge the same quantity or the printed curve
    // and the action taken are two different stories). `have_counter` is explicit instead of encoded
    // as a negative number: growth is legitimately negative when the driver releases tracking memory,
    // and a policy value must never double as its own "unknown" sentinel.
    // `growing` is the watch's monotonic-run verdict (run_up >= mono_need). The spec's own sentence is
    // "monotonic growth past a 32-64 MB threshold", so BOTH halves of it have to be true before the
    // guard acts - and they have to be checked the same way in the sample line and in the action line,
    // or a log prints "hard 0" next to "[KV REFUSED]" (the first refusal test did exactly that: two
    // samples, both increasing, growth 2.12 > hard 2, so the guard acted while the sample counter,
    // which also demands a run of 3, still said 0 - runs/p2_refuse_v1.log). A single spike is noise;
    // growth that keeps growing is what VidMm eviction looks like.
    KvAction decide(uint32_t next_pos, double value, bool have_counter, bool growing,
                    std::string &why) {
        const uint32_t b = next_pos / blk_slots;
        why.clear();
        // (1) geometry ceiling: the arena is fixed at cap slots; positions past it are not "spill",
        //     they are out of bounds, and this is the path that keeps the guard honest.
        if (next_pos >= cap) { why = "KV arena full: the position reaches cap"; return KV_REFUSE; }
        if (!have_counter) return KV_OK;                       // no counter -> policy cannot judge
        if (value > g_spill_hard && growing) {
            if (allow_evict && b >= 1 && n_blk >= 2) { why = "spill: evict the farthest KV block"; return KV_EVICT; }
            why = "spill: refuse new KV blocks"; return KV_REFUSE;
        }
        if (value > g_spill_soft) { why = "spill: past the soft threshold, watching"; return KV_WARN; }
        return KV_OK;
    }

    void report() const {
        printf("  KV block hook: %u slots = %u blocks of %u positions, %.2f MiB per block"
               " (%.2f KiB/slot x %u layers x %u), ceiling %.2f MiB, evict %s\n",
               cap, n_blk, blk_slots, double(bytes_per_blk) / 1048576.0,
               double(g.bytes_slot) / 1024.0, g.n_attn, blk_slots,
               double(ceiling_bytes) / 1048576.0, allow_evict ? "ENABLED (BONSAI_KV_EVICT=1)" : "off");
    }
};

#endif // BONSAI_KV_GUARD_H
