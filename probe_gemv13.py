"""Probe 13: the last two levers, measured together.

probe 10: weights stream at 69-79 GB/s (no MACs); GEMV only 21-32 GB/s -> not DRAM-bound.
probe 11: serving x as vec4<f32> (1 load per 4 MACs) lifts 22 -> 29 GB/s and 32 -> 38 GB/s,
          but only for M=12288. rows>1 (2/4 rows per thread) always LOSES (register pressure).
probe 12: pure register fma tops out at ~220-250 G lane-ops/s on this card, so ~1.5 ops/MAC
          is already what the 153 Gop/s kernel spends. The ALU, not PCIe/VRAM, is the wall.

What is left: M=4096 shapes dispatch only 64 wavefronts (1.6 per SIMD) -> latency-bound, they
measured 117 Gop/s vs 153 for M=12288. Fix = split-K (more wavefronts, atomic-free 2-pass
reduce). Sweep splits for both 4096-row shapes on top of the vec4-x kernel from probe 11.
"""
import time
import numpy as np
import wgpu
from probe_gemv10 import HDR, pack2_vec4

HDR4 = HDR.replace("var<storage, read> x: array<f32>;", "var<storage, read> x: array<vec4<f32>>;")

REDUCE = HDR + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < p.rows; i = i + 1u) {
        s = s + out[i * p.m + row];
    }
    out[row] = s + p.corr;
}
"""


def build_xvec(K, splits=1, nacc=4, unroll=2, wg=64):
    """One output row per thread, x as vec4<f32>, optional split-K over g.y."""
    body = []
    for v in range(unroll):
        body.append("        let w%d = wv[(i + %du) * p.m + row];\n" % (v, v))
    for v in range(unroll):
        for u in range(4):
            for t in range(4):
                body.append("        let xv%d_%d_%d = x[(i + %du) * 16u + %du + %du];\n"
                            % (v, u, t, v, u * 4, t))
                for c in range(4):
                    j = t * 4 + c
                    body.append("        a%d = a%d + xv%d_%d_%d.%s * f32(extractBits(w%d.%s, %du, 2u));\n"
                                % ((u * 16 + j) % nacc, (u * 16 + j) % nacc, v, u, t,
                                   "xyzw"[c], v, "xyzw"[u], 2 * j))
    accs = ["a%d" % k for k in range(nacc)]
    decl = "".join("    var %s: f32 = 0.0;\n" % a for a in accs)
    if splits == 1:
        head = ("    let nvec = p.k / 64u;\n    let lo: u32 = 0u;\n    let hi: u32 = nvec;\n")
        tail = "    out[row] = (%s) + p.corr;\n" % " + ".join(accs)
    else:
        head = ("    let nvec = p.k / 64u;\n    let per = nvec / p.rows;\n"
                "    let lo = g.y * per;\n    let hi = lo + per;\n")
        tail = "    out[g.y * p.m + row] = (%s);\n" % " + ".join(accs)
    return HDR4 + """
@compute @workgroup_size(%du)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
%s%s    for (var i: u32 = lo; i < hi; i = i + %du) {
%s    }
%s}
""" % (wg, decl, head, unroll, "".join(body), tail)


def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()
    print("adapter:", dict(ad.info).get("device"), "|", dict(ad.info).get("backend_type"), "\n")
    bgl = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    rm = dev.create_shader_module(code=REDUCE)
    rpipe = dev.create_compute_pipeline(layout=pl, compute={"module": rm, "entry_point": "main"})
    rng = np.random.default_rng(5)

    cfgs = [("XVEC s1 n4 u2", dict(splits=1, nacc=4, unroll=2)),
            ("XVEC s2 n4 u2", dict(splits=2, nacc=4, unroll=2)),
            ("XVEC s4 n4 u2", dict(splits=4, nacc=4, unroll=2)),
            ("XVEC s8 n4 u2", dict(splits=8, nacc=4, unroll=2)),
            ("XVEC s16 n4 u2", dict(splits=16, nacc=4, unroll=2)),
            ("XVEC s4 n4 u1", dict(splits=4, nacc=4, unroll=1)),
            ("XVEC s4 n8 u4", dict(splits=4, nacc=8, unroll=4)),
            ("XVEC s8 n4 u4", dict(splits=8, nacc=4, unroll=4))]

    for M, K, tag in [(4096, 4096, "attn q/o"), (4096, 12288, "down"), (12288, 4096, "gate+up"),
                      (1024, 4096, "narrow M=1024")]:
        codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
        x = rng.standard_normal(K).astype(np.float32)
        ref = (codes.astype(np.float32) - 1.0) @ x
        p2 = pack2_vec4(codes)
        xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        wb = dev.create_buffer_with_data(data=p2.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        built = []
        for label, kw in cfgs:
            sp = kw["splits"]
            src = build_xvec(K, **kw)
            m = dev.create_shader_module(code=src)
            pipe = dev.create_compute_pipeline(layout=pl, compute={"module": m, "entry_point": "main"})
            n = M * sp * 4
            ob = dev.create_buffer(size=n, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            par = dev.create_buffer_with_data(
                data=np.array([M, K, K // 16, sp], dtype=np.uint32).tobytes()
                + np.float32(-x.sum()).tobytes() + b"\0" * 12, usage=wgpu.BufferUsage.UNIFORM)
            bg = dev.create_bind_group(layout=bgl, entries=[
                {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
                {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
                {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": n}},
                {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 32}},
            ])

            def run(c, pipe=pipe, bg=bg, sp=sp):
                c.set_pipeline(pipe); c.set_bind_group(0, bg)
                c.dispatch_workgroups((M + 63) // 64, sp, 1)
                if sp > 1:
                    c.set_pipeline(rpipe); c.set_bind_group(0, bg)
                    c.dispatch_workgroups((M + 63) // 64, 1, 1)
            enc = dev.create_command_encoder()
            cc = enc.begin_compute_pass(); run(cc); cc.end()
            dev.queue.submit([enc.finish()])
            got = np.frombuffer(dev.queue.read_buffer(ob, 0, M * 4), dtype=np.float32)
            built.append((label, run, float(np.abs(got - ref).max()), ob))

        res = {b[0]: [] for b in built}
        for _ in range(3):
            for label, run, _e, ob in built:
                enc = dev.create_command_encoder()
                cc = enc.begin_compute_pass()
                for _ in range(3):
                    run(cc)
                cc.end(); dev.queue.submit([enc.finish()])
                t0 = time.perf_counter()
                enc = dev.create_command_encoder()
                cc = enc.begin_compute_pass()
                for _ in range(100):
                    run(cc)
                cc.end(); dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(ob, 0, 4)
                res[label].append((time.perf_counter() - t0) / 100)

        print("=== %s M=%d K=%d (%.1f MB weights) ===" % (tag, M, K, p2.nbytes / 1e6))
        print(f"{'variant':16s} {'med ms':>8s} {'best':>7s} {'Gop/s':>8s} {'GB/s':>7s}  err")
        for label, run, err, ob in built:
            med, best = float(np.median(res[label])), min(res[label])
            print(f"{label:16s} {med*1e3:8.3f} {best*1e3:7.3f} {M*K/med/1e9:8.1f} "
                  f"{p2.nbytes/med/1e9:7.1f}  {err:.1e}")
        print()


if __name__ == "__main__":
    main()
