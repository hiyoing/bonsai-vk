// pdh_probe.cpp - which exact PDH path string this Windows + AMD driver will ACCEPT for
// "GPU Process Memory\Shared Usage", and what instances it then answers for.
//
// Why this exists: inside bonsai_vk.exe, PdhAddEnglishCounterA returned 0xC0000BC0 for both the
// pid-scoped and the (*) form. 0xC0000BC0 is PDH_CSTATUS_BAD_COUNTERNAME (pdhmsg.h line 37) - the
// NAME was rejected, which is a completely different problem from "the instance does not exist yet",
// and no amount of retrying at sample time fixes a rejected name. Meanwhile tools/vram_sample.ps1
// has been reading this counter on this machine for weeks through Get-Counter, whose paths all start
// with a BACKSLASH. So the leading backslash, the English-vs-localised API, and the wildcard form are
// each tried separately here instead of being guessed at inside the engine.
//
// Build:  g++ -std=c++17 -O2 -o ../runs/pdh_probe.exe pdh_probe.cpp -lpdh
// Run:    ../runs/pdh_probe.exe [pid-to-watch]        (self pid when no argument)
#define WIN32_LEAN_AND_MEAN
#define NOMINMAX
#include <windows.h>
#include <pdh.h>
#include <pdhmsg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cstdint>
#include <string>
#include <vector>

static const char *st(DWORD s) {
    // if-chain, not switch: MinGW's PDH status macros are unsigned (they came from a C#-style
    // __MSABI_LONG(0x...) definition), and a switch over a LONG has to narrow them.
    const LONG v = LONG(s);
    if (v == 0) return "OK";
    if (v == (LONG)PDH_CSTATUS_BAD_COUNTERNAME) return "BAD_COUNTERNAME (0xC0000BC0)";
    if (v == (LONG)PDH_INVALID_PATH)            return "INVALID_PATH (0xC0000BC4)";
    if (v == (LONG)PDH_INVALID_INSTANCE)        return "INVALID_INSTANCE (0xC0000BC5)";
    if (v == (LONG)PDH_MORE_DATA)               return "MORE_DATA (0x800007D2)";
    if (v == (LONG)PDH_CSTATUS_NO_INSTANCE)     return "NO_INSTANCE (0x800007D1)";
    if (v == (LONG)PDH_CSTATUS_NO_MACHINE)      return "NO_MACHINE (0x800007D0)";
    if (v == (LONG)PDH_CSTATUS_NO_OBJECT)       return "NO_OBJECT (0x800007CE)";
    return "other";
}

// One path through the English or the localised add API, then collect and read every instance it
// answers for. Values are printed in MiB; instances whose name does not contain this pid are marked
// so the sum-vs-filter decision can be made from the output instead of from a guess.
static void try_path(PDH_HQUERY q, const char *path, bool english, uint32_t pid) {
    PDH_HCOUNTER c = nullptr;
    const DWORD s = english ? PdhAddEnglishCounterA(q, path, 0, &c) : PdhAddCounterA(q, path, 0, &c);
    printf("  %-56s %-8s -> add: %s\n", path, english ? "ENGLISH" : "LOCALISED", st(s));
    if (s != 0) return;
    const DWORD cs = PdhCollectQueryData(q);
    printf("      collect: %s\n", cs == 0 ? "OK" : st(cs));
    if (cs != 0) return;
    DWORD bytes = 0, items = 0;
    LONG r = PdhGetFormattedCounterArrayA(c, PDH_FMT_LARGE, &bytes, &items, nullptr);
    if (r != (LONG)PDH_MORE_DATA && r != 0) {
        printf("      array size probe: 0x%08lX (bytes %u items %u)\n", (unsigned long)r,
               (unsigned)bytes, (unsigned)items);
        return;
    }
    std::vector<char> buf(bytes + 4096u);
    bytes = DWORD(buf.size()); items = 0;
    r = PdhGetFormattedCounterArrayA(c, PDH_FMT_LARGE, &bytes, &items,
                                     (PDH_FMT_COUNTERVALUE_ITEM_A *)buf.data());
    if (r != 0) {
        printf("      array read: 0x%08lX (bytes %u)\n", (unsigned long)r, (unsigned)bytes);
        return;
    }
    const std::string want = "pid_" + std::to_string(pid) + "_";   // the trailing underscore is what
                                                                    // keeps pid 123 from matching
                                                                    // the instance of pid 1234
    printf("      %u instance(s) (looking for \"%s\"):\n", (unsigned)items, want.c_str());
    const PDH_FMT_COUNTERVALUE_ITEM_A *it = (const PDH_FMT_COUNTERVALUE_ITEM_A *)buf.data();
    double mine = 0.0; int n_mine = 0;
    for (DWORD i = 0; i < items && i < 14; ++i) {
        const bool is_mine = strstr(it[i].szName, want.c_str()) != nullptr;
        if (it[i].FmtValue.CStatus != PDH_CSTATUS_VALID_DATA &&
            it[i].FmtValue.CStatus != PDH_CSTATUS_NEW_DATA) {
            printf("        %-64s status 0x%08lX\n", it[i].szName,
                   (unsigned long)it[i].FmtValue.CStatus);
            continue;
        }
        const double v = double(it[i].FmtValue.largeValue) / 1048576.0;
        printf("        %-64s %10.2f MiB  %s\n", it[i].szName, v, is_mine ? "MINE" : "other pid");
        if (is_mine) { mine += v; ++n_mine; }
    }
    if (items > 14) printf("        ... %u more\n", (unsigned)(items - 14));
    printf("      SUM OVER THIS PID: %d instance(s), %.2f MiB\n", n_mine, mine);
}

int main(int argc, char **argv) {
    const uint32_t pid = argc > 1 ? uint32_t(atoi(argv[1])) : uint32_t(GetCurrentProcessId());
    printf("=== PDH path acceptance probe (watching pid %u) ===\n", pid);
    char star[256], star_ns[256], pidp[256], pidp_ns[256], bare[256];
    snprintf(star,    sizeof star,    "\\GPU Process Memory(*)\\Shared Usage");
    snprintf(star_ns, sizeof star_ns, "GPU Process Memory(*)\\Shared Usage");
    snprintf(pidp,    sizeof pidp,    "\\GPU Process Memory(pid_%u_*)\\Shared Usage", pid);
    snprintf(pidp_ns, sizeof pidp_ns, "GPU Process Memory(pid_%u_*)\\Shared Usage", pid);
    snprintf(bare,    sizeof bare,    "\\GPU Process Memory\\Shared Usage");

    PDH_HQUERY q;
    if (PdhOpenQuery(nullptr, 0, &q)) { printf("  PdhOpenQuery failed\n"); return 1; }
    printf("\n  A) the star form WITH a leading backslash (the form Get-Counter always uses)\n");
    try_path(q, star, true, pid);
    printf("\n  B) the same two forms WITHOUT the leading backslash (what the engine passes today)\n");
    try_path(q, star_ns, true, pid);
    try_path(q, pidp_ns, true, pid);
    printf("\n  C) the pid-scoped form WITH the backslash\n");
    try_path(q, pidp, true, pid);
    printf("\n  D) the LOCALISED API on the star path: the test for a driver that registered only a\n"
           "     zh-CN name, in which case PdhAddCounterA matches and PdhAddEnglishCounterA does not\n");
    try_path(q, star, false, pid);
    printf("\n  E) object-only path, no instance part at all\n");
    try_path(q, bare, true, pid);
    PdhCloseQuery(q);

    // ExpandWildCard answers "which instances exist right now" without adding a counter, so the
    // engine can discover the driver's exact spelling instead of assuming pid_<pid>_*.
    printf("\n  F) PdhExpandWildCardPath(\"GPU Process Memory(*)\\\\Shared Usage\")\n");
    DWORD n = 0;
    LONG r = PdhExpandWildCardPathA(nullptr, "GPU Process Memory(*)\\Shared Usage", nullptr, &n, 0);
    printf("      first call: 0x%08lX, %u characters\n", (unsigned long)r, (unsigned)n);
    if (r == 0 && n) {
        std::vector<char> b(n + 16, 0);
        DWORD n2 = DWORD(b.size());
        r = PdhExpandWildCardPathA(nullptr, "GPU Process Memory(*)\\Shared Usage", b.data(), &n2, 0);
        printf("      second call: 0x%08lX, %u characters\n", (unsigned long)r, (unsigned)n2);
        const char *p = b.data();
        int i = 0;
        while (*p && i < 14) { printf("      [%d] %s\n", i++, p); p += strlen(p) + 1; }
        if (i == 0) printf("      (the multi-string is empty: no instance exists yet)\n");
    }
    return 0;
}
