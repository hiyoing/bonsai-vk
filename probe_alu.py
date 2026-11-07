"""Probe 12: the ALU roof. Which instruction does GCN4 actually charge for, and how many
lane-ops per cycle can this card issue?

This decides whether the ternary GEMV is memory-bound or instruction-bound, i.e. whether a
custom runtime can beat llama.cpp at all. Each kernel is a register-only throughput test:
a long independent chain of the op under test, no memory traffic except one sink write.

Reported as G lane-ops/s (one lane-op = one op on one of the 640 streams).
Occupancy is swept because a single wavefront per SIMD cannot hide the ~4-cycle VALU latency.
"""
import time
import numpy as np
import wgpu

HDR = """struct P { n: u32, reps: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read_write> out: array<f32>;
@group(0) @binding(1) var<uniform> p: P;
"""


def kernel(ops_per_iter: int, body: str, decl: str, sink: str) -> str:
    return HDR + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    if (g.x >= p.n) { return; }
%s
    for (var r: u32 = 0u; r < p.reps; r = r + 1u) {
%s    }
%s
}
""" % (decl, body, sink)


def make_fma(nchain=8):
    decl = "".join("    var a%d: f32 = f32(g.x + %du) * 0.001 + 1.0;\n" % (k, k) for k in range(nchain))
    decl += "    let b: f32 = 1.0000001;\n    let c: f32 = 0.0000001;\n"
    body = "".join("        a%d = a%d * b + c;\n" % (k, k) for k in range(nchain))
    sink = "    out[g.x] = %s;\n" % " + ".join("a%d" % k for k in range(nchain))
    return kernel(nchain, body, decl, sink), nchain


def make_int(kind, nchain=8):
    decl = "".join("    var a%d: u32 = g.x * %du + 0x12345678u;\n" % (k, k) for k in range(nchain))
    decl += "    let m: u32 = 7u;\n"
    if kind == "bfe":
        body = "".join("        a%d = a%d + extractBits(a%d, %du, 2u);\n" % (k, k, k, k % 30) for k in range(nchain))
    elif kind == "shr_and":
        body = "".join("        a%d = a%d + ((a%d >> %du) & 3u);\n" % (k, k, k, k % 30) for k in range(nchain))
    elif kind == "iadd":
        body = "".join("        a%d = a%d + m;\n" % (k, k) for k in range(nchain))
    elif kind == "imul":
        body = "".join("        a%d = a%d * m;\n" % (k, k) for k in range(nchain))
    sink = "    out[g.x] = f32(%s);\n" % " + ".join("a%d" % k for k in range(nchain))
    return kernel(nchain, body, decl, sink), nchain


def make_cvt(nchain=8):
    decl = "".join("    var a%d: u32 = g.x * %du + 0x12345678u;\n" % (k, k) for k in range(nchain))
    decl += "".join("    var f%d: f32 = 0.0;\n" % k for k in range(nchain))
    body = "".join("        f%d = f32(a%d); a%d = a%d + 1u;\n" % (k, k, k, k) for k in range(nchain))
    sink = "    out[g.x] = %s;\n" % " + ".join("f%d" % k for k in range(nchain))
    return kernel(nchain * 2, body, decl, sink), nchain


def make_bfe_cvt_fma(nchain=4):
    """The actual ternary inner loop, register-only: bfe + cvt + fma per MAC."""
    decl = "".join("    var a%d: f32 = 0.0;\n" % k for k in range(nchain))
    decl += "    var w: u32 = g.x * 2654435761u;\n    let b: f32 = 1.0000001;\n"
    inner = []
    for k in range(nchain):
        inner.append("        a%d = fma(f32(extractBits(w, %du, 2u)), b, a%d);\n" % (k, 2 * k, k))
        inner.append("        w = w + 0x00010001u;\n")
    body = "".join("    " + s + "\n" for s in inner)
    sink = "    out[g.x] = %s + f32(w);\n" % " + ".join("a%d" % k for k in range(nchain))
    ops = nchain * 3 + nchain  # 3 per MAC + one extra u32 add per lane to keep w live
    return kernel(ops, body, decl, sink), ops


def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()
    print("adapter:", dict(ad.info).get("device"), "|", dict(ad.info).get("backend_type"), "\n")
    bgl = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])

    tests = [("fma f32 n8    ", make_fma(8)), ("fma f32 n16   ", make_fma(16)),
             ("u32 iadd      ", make_int("iadd", 8)),
             ("u32 imul      ", make_int("imul", 8)),
             ("bfe+add       ", make_int("bfe", 8)),
             ("shr+and+add   ", make_int("shr_and", 8)),
             ("cvt u32->f32  ", make_cvt(8)),
             ("bfe+cvt+fma/3 ", make_bfe_cvt_fma(4))]

    for N, REPS in [(2560, 2000), (20480, 2000), (65536, 2000)]:
        out = dev.create_buffer(size=N * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(
            data=np.array([N, REPS, 0, 0], dtype=np.uint32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": out, "offset": 0, "size": N * 4}},
            {"binding": 1, "resource": {"buffer": par, "offset": 0, "size": 16}},
        ])
        print(f"--- {N} threads = {N // 64} wf = {max(N // 64 // 40, 1)} wf/SIMD (40 SIMDs), reps={REPS} ---")
        for label, (src, ops) in tests:
            m = dev.create_shader_module(code=src)
            pipe = dev.create_compute_pipeline(layout=pl, compute={"module": m, "entry_point": "main"})
            enc = dev.create_command_encoder()
            c = enc.begin_compute_pass(); c.set_pipeline(pipe); c.set_bind_group(0, bg)
            c.dispatch_workgroups(N // 64, 1, 1); c.end()
            dev.queue.submit([enc.finish()])
            dev.queue.read_buffer(out, 0, 4)
            best = 1e9
            for _ in range(3):
                enc = dev.create_command_encoder()
                c = enc.begin_compute_pass()
                for _ in range(5):
                    c.set_pipeline(pipe); c.set_bind_group(0, bg); c.dispatch_workgroups(N // 64, 1, 1)
                c.end(); dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(out, 0, 4)
                t0 = time.perf_counter()
                enc = dev.create_command_encoder()
                c = enc.begin_compute_pass()
                for _ in range(10):
                    c.set_pipeline(pipe); c.set_bind_group(0, bg); c.dispatch_workgroups(N // 64, 1, 1)
                c.end(); dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(out, 0, 4)
                best = min(best, (time.perf_counter() - t0) / 10)
            lane_ops = N * REPS * ops / best / 1e9
            print(f"{label:14s} {lane_ops:9.1f}G lane-op/s   ({N * REPS * ops:.3e} ops / {best*1e3:.2f} ms)")
        print()

    print("theoretical: 640 lanes x 1.287 GHz = 824 G lane-op/s if every op issued every cycle.")
    print("GCN4 VALU reality: a wavefront64 f32 op occupies one 16-wide SIMD for 4 cycles,")
    print("so 40 SIMDs x 16 lanes x clock = same 824 G only at perfect issue.")


if __name__ == "__main__":
    main()
