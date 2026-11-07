// bonsai_platform.h - every OS call the engine makes, in one place, behind #ifdef _WIN32.
//
// WHY ONE HEADER. The engine was Windows-only for a reason: PDH's "GPU Process Memory\Shared Usage"
// is the WDDM eviction signal this project's spill defence was measured against, and DXGI's
// QueryVideoMemoryInfo is the budget the OOM fallback reads. Neither exists on Linux. But neither is
// on the generation path either - they are instruments. So the port is not a rewrite of the engine,
// it is a substitution of instruments plus a socket layer, and putting all of it here means the
// forward path, the shaders and the dequant arithmetic contain zero platform branches.
//
// WHAT IS VERIFIED. The Windows branch is the code that produced every log in runs/. The POSIX branch
// is written against the documented API and sysfs contracts and HAS NOT BEEN COMPILED OR RUN: the
// development machine is Windows and its WSL has no distribution installed. README says so in those
// words. Nothing here claims a Linux measurement.
//
// INCLUDE ORDER IS LOAD-BEARING ON WINDOWS. winsock2.h must precede windows.h, because windows.h
// otherwise pulls in the Winsock 1.1 header and the two disagree about every type. kv_guard.h includes
// windows.h for PDH/DXGI, so this header - which includes winsock2.h first - has to be the first of
// the two to be seen. Including it from bonsai_weights.h (the earliest engine header) guarantees that
// for the whole translation unit, and #pragma once makes the later includes no-ops.
//
// Every function is `static inline`: an unused one is neither emitted nor warned about, so a tool that
// includes this header for bonsai_seek() alone does not acquire a link dependency on ws2_32.

#pragma once

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#ifndef NOMINMAX
#define NOMINMAX        // windows.h defines min/max as macros; std::min in the budget code needs them gone
#endif
#include <winsock2.h>   // MUST precede windows.h - see the header note
#include <ws2tcpip.h>
#include <windows.h>
#include <shellapi.h>   // ShellExecuteA: "open the default browser"
#else
#include <sys/types.h>
#include <sys/socket.h>
#include <sys/time.h>
#include <netinet/in.h>
#include <netinet/tcp.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <dirent.h>
#include <pthread.h>
#include <spawn.h>
#include <cerrno>
#include <csignal>
#ifdef __APPLE__
#include <mach-o/dyld.h>  // _NSGetExecutablePath
#endif
#endif

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>

// ------------------------------------------------------------------ identity and files
static inline uint32_t bonsai_pid() {
#ifdef _WIN32
    return uint32_t(GetCurrentProcessId());
#else
    return uint32_t(getpid());
#endif
}

// 64-bit seek is mandatory, not an optimisation: the checkpoint is ~2 GB and a tensor offset is added
// to it, so on a host where long is 32 bits plain fseek() wraps silently past 2 GiB. fseeko needs
// off_t to be 64 bits, which the Linux build script guarantees with -D_FILE_OFFSET_BITS=64.
static inline bool bonsai_seek(FILE *f, uint64_t pos) {
#ifdef _WIN32
    return _fseeki64(f, static_cast<long long>(pos), SEEK_SET) == 0;
#else
    return fseeko(f, static_cast<off_t>(pos), SEEK_SET) == 0;
#endif
}

static inline std::string bonsai_exe_path() {
    char b[4096];
#ifdef _WIN32
    const DWORD n = GetModuleFileNameA(nullptr, b, DWORD(sizeof b));
    if (!n || n >= sizeof b) return std::string();
    return std::string(b, n);
#elif defined(__APPLE__)
    uint32_t n = uint32_t(sizeof b);
    if (_NSGetExecutablePath(b, &n) != 0) return std::string();
    return std::string(b);
#else
    const ssize_t n = readlink("/proc/self/exe", b, sizeof b - 1);
    if (n <= 0) return std::string();
    b[n] = '\0';
    return std::string(b, size_t(n));
#endif
}

static inline std::string bonsai_dir_of(const std::string &p) {
    const size_t k = p.find_last_of("\\/");
    return k == std::string::npos ? std::string() : p.substr(0, k);
}

// Only ever used to supply a DEFAULT the caller can still override: the serve path sets BONSAI_CAP
// when the user has not, and getenv() in fwd_main then reads it like any other knob.
static inline bool bonsai_setenv(const char *k, const char *v) {
#ifdef _WIN32
    return _putenv_s(k, v) == 0;
#else
    return setenv(k, v, 1) == 0;
#endif
}

// Open the default browser. POSIX uses posix_spawn rather than system(): no shell is involved, so
// there is no quoting question even though the URL here is built only from an integer port. The
// child is not waited for - the browser is meant to outlive the launcher that opened it.
static inline bool bonsai_open_url(const std::string &url) {
#ifdef _WIN32
    const HINSTANCE h = ShellExecuteA(nullptr, "open", url.c_str(), nullptr, nullptr, SW_SHOWNORMAL);
    return reinterpret_cast<intptr_t>(h) > 32;
#else
#ifdef __APPLE__
    const char *const cmds[2] = {"/usr/bin/open", "open"};
#else
    const char *const cmds[2] = {"/usr/bin/xdg-open", "xdg-open"};
#endif
    extern char **environ;
    for (const char *cmd : cmds) {
        char *argv[3] = {const_cast<char *>(cmd), const_cast<char *>(url.c_str()), nullptr};
        pid_t child = 0;
        if (posix_spawn(&child, cmd, nullptr, nullptr, argv, environ) == 0) return true;
    }
    return false;
#endif
}

// ------------------------------------------------------------------ the GPU serialisation lock
// One queue, one Fwd, one not-thread-safe engine. A TRY lock on purpose: a second concurrent request
// is refused with 429 rather than parked on a socket behind a reply that can take a minute.
struct bonsai_mutex {
#ifdef _WIN32
    CRITICAL_SECTION cs;
#else
    pthread_mutex_t m;
#endif
};

static inline void bonsai_mutex_init(bonsai_mutex &x) {
#ifdef _WIN32
    InitializeCriticalSection(&x.cs);
#else
    pthread_mutex_init(&x.m, nullptr);
#endif
}
static inline bool bonsai_mutex_try(bonsai_mutex &x) {
#ifdef _WIN32
    return TryEnterCriticalSection(&x.cs) != 0;
#else
    return pthread_mutex_trylock(&x.m) == 0;
#endif
}
static inline void bonsai_mutex_leave(bonsai_mutex &x) {
#ifdef _WIN32
    LeaveCriticalSection(&x.cs);
#else
    pthread_mutex_unlock(&x.m);
#endif
}
static inline void bonsai_mutex_destroy(bonsai_mutex &x) {
#ifdef _WIN32
    DeleteCriticalSection(&x.cs);
#else
    pthread_mutex_destroy(&x.m);
#endif
}

// ------------------------------------------------------------------ detached worker threads
// pthread_create and CreateThread disagree about the entry point's return type and calling
// convention, so both get a trampoline over one shared argument block. No reinterpret_cast between
// function pointer types, which would be undefined behaviour however often it works.
struct BonsaiThreadArg { void (*fn)(void *); void *arg; };

#ifdef _WIN32
static DWORD WINAPI bonsai_thread_win(LPVOID p) {
    BonsaiThreadArg *a = static_cast<BonsaiThreadArg *>(p);
    void (*fn)(void *) = a->fn;
    void *arg = a->arg;
    delete a;
    fn(arg);
    return 0;
}
#else
static void *bonsai_thread_posix(void *p) {
    BonsaiThreadArg *a = static_cast<BonsaiThreadArg *>(p);
    void (*fn)(void *) = a->fn;
    void *arg = a->arg;
    delete a;
    fn(arg);
    return nullptr;
}
#endif

static inline bool bonsai_thread_spawn(void (*fn)(void *), void *arg) {
    BonsaiThreadArg *a = new BonsaiThreadArg{fn, arg};
#ifdef _WIN32
    const HANDLE h = CreateThread(nullptr, 0, bonsai_thread_win, a, 0, nullptr);
    if (!h) { delete a; return false; }
    CloseHandle(h);                    // detached: the thread owns its argument block
    return true;
#else
    pthread_t t;
    if (pthread_create(&t, nullptr, bonsai_thread_posix, a) != 0) { delete a; return false; }
    pthread_detach(t);
    return true;
#endif
}

// ------------------------------------------------------------------ sockets
#ifdef _WIN32
typedef SOCKET bonsai_sock_t;
#define BONSAI_BAD_SOCK INVALID_SOCKET
#define BONSAI_SOCK_EINTR WSAEINTR
#else
typedef int bonsai_sock_t;
#define BONSAI_BAD_SOCK (-1)
#define BONSAI_SOCK_EINTR EINTR
#endif

static inline bool bonsai_sock_bad(bonsai_sock_t s) { return s == BONSAI_BAD_SOCK; }
static inline void bonsai_sock_close(bonsai_sock_t s) {
#ifdef _WIN32
    closesocket(s);
#else
    close(s);
#endif
}
static inline int bonsai_sock_err() {
#ifdef _WIN32
    return WSAGetLastError();
#else
    return errno;
#endif
}

// accept() failures that mean "try again", not "the listener is broken". ECONNABORTED is the POSIX
// one: a client that finished the handshake and vanished before accept() ran. Windows reports the
// same situation as WSAECONNRESET on a listening socket. Treating either as fatal would let one
// closed browser tab stop the server.
static inline bool bonsai_accept_retry(int err) {
#ifdef _WIN32
    return err == WSAEINTR || err == WSAECONNRESET;
#else
    return err == EINTR || err == ECONNABORTED;
#endif
}

static inline bool bonsai_net_init() {
#ifdef _WIN32
    WSADATA wd;
    return WSAStartup(MAKEWORD(2, 2), &wd) == 0;
#else
    // A streaming server's client can vanish mid-reply. Without this, the next send() raises SIGPIPE
    // and the default disposition kills the whole engine - one closed browser tab would take the
    // GPU process with it. Linux also gets MSG_NOSIGNAL per call; macOS has no such flag, so this is
    // the half that covers it.
    signal(SIGPIPE, SIG_IGN);
    return true;
#endif
}
static inline void bonsai_net_cleanup() {
#ifdef _WIN32
    WSACleanup();
#endif
}

// EINTR is retried here rather than in the HTTP layer: a signal that interrupts a send mid-SSE must
// not be reported to the caller as "the client went away", which is what a bare `k <= 0` would do.
static inline long bonsai_send(bonsai_sock_t s, const char *p, size_t n) {
#ifdef _WIN32
    const int flags = 0;
#elif defined(MSG_NOSIGNAL)
    const int flags = MSG_NOSIGNAL;   // Linux: per-call, on top of the process-wide SIG_IGN
#else
    const int flags = 0;              // macOS has no MSG_NOSIGNAL; SO_NOSIGPIPE + SIG_IGN cover it
#endif
    for (;;) {
#ifdef _WIN32
        const long k = long(send(s, p, int(n), flags));
        if (k < 0 && WSAGetLastError() == WSAEINTR) continue;
#else
        const long k = long(send(s, p, n, flags));
        if (k < 0 && errno == EINTR) continue;
#endif
        return k;
    }
}
static inline long bonsai_recv(bonsai_sock_t s, char *p, size_t n) {
    for (;;) {
#ifdef _WIN32
        const long k = long(recv(s, p, int(n), 0));
        if (k < 0 && WSAGetLastError() == WSAEINTR) continue;
#else
        const long k = long(recv(s, p, n, 0));
        if (k < 0 && errno == EINTR) continue;
#endif
        return k;
    }
}

static inline void bonsai_sock_timeout(bonsai_sock_t s, uint32_t ms) {
#ifdef _WIN32
    const DWORD t = ms;
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, reinterpret_cast<const char *>(&t), sizeof t);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, reinterpret_cast<const char *>(&t), sizeof t);
#else
    timeval tv{};
    tv.tv_sec = long(ms / 1000u);
    tv.tv_usec = long((ms % 1000u) * 1000u);
    setsockopt(s, SOL_SOCKET, SO_RCVTIMEO, &tv, sizeof tv);
    setsockopt(s, SOL_SOCKET, SO_SNDTIMEO, &tv, sizeof tv);
#ifdef SO_NOSIGPIPE
    int yes = 1;
    setsockopt(s, SOL_SOCKET, SO_NOSIGPIPE, &yes, sizeof yes);
#endif
#endif
}

// socket + SO_REUSEADDR + bind(127.0.0.1) + listen, or BAD_SOCK with the OS error in `err`.
// Loopback only, on every platform: this is a local shell for the person sitting at the machine,
// not a service to expose. bind/listen return nonzero on failure on both stacks, so the checks below
// are the same code; only the setsockopt argument type and the error accessor differ.
static inline bonsai_sock_t bonsai_listen(uint16_t port, int backlog, int &err) {
    err = 0;
    bonsai_sock_t s = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
    if (bonsai_sock_bad(s)) { err = bonsai_sock_err(); return BONSAI_BAD_SOCK; }
    int yes = 1;
#ifdef _WIN32
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<const char *>(&yes), sizeof yes);
#else
    setsockopt(s, SOL_SOCKET, SO_REUSEADDR, &yes, sizeof yes);
#endif
    sockaddr_in sa{};
    sa.sin_family = AF_INET;
    sa.sin_port = htons(port);
    sa.sin_addr.s_addr = htonl(INADDR_LOOPBACK);
    if (bind(s, reinterpret_cast<const sockaddr *>(&sa), sizeof sa) != 0) {
        err = bonsai_sock_err(); bonsai_sock_close(s); return BONSAI_BAD_SOCK;
    }
    if (listen(s, backlog) != 0) {
        err = bonsai_sock_err(); bonsai_sock_close(s); return BONSAI_BAD_SOCK;
    }
    return s;
}