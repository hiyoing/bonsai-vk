"""Probe 11: fix the real bottleneck found in probe 10.

probe 10 verdict: the packed weights stream at 69-79 GB/s (READ) but the ternary GEMV only
reaches 22-32 GB/s (BASE). The gap is not DRAM, it is the 64 separate scalar x-loads per
vec4<u32> of weights: 1 load instruction + 4 bytes of L1 traffic per MAC, a 16:1 read
amplification against the weight stream. Staging x in workgroup memory (LDS) made things
2.3-8x WORSE because Polaris' 64KB LDS caps resident workgroups per CU.

Fix under test: load x as vec4<f32> (one instruction feeds 4 MACs) -> 4x fewer x loads and
4x less L1 traffic, and/or give each thread more output rows so a loaded x value is reused.
"""
import time
import numpy as np
import wgpu
from probe_gemv10 import HDR, pack2_vec4

HDR4 = HDR.replace("var<storage, read> x: array<f32>;", "var<storage, read> x: array<vec4<f32>>;")


def build_xvec(K, rows=1, nacc=4, unroll=2, wg=64):
    """x served as vec4<f32>: 1 load per 4 MACs. rows = output rows owned per thread."""
    body = []
    for v in range(unroll):
        for rr in range(rows):
            body.append("        let w%d_%d = wv[(i + %du) * p.m + r%d];\n" % (v, rr, v, rr))
    for v in range(unroll):
        for u in range(4):
            for t in range(4):
                body.append("        let xv%d_%d_%d = x[(i + %du) * 16u + %du + %du];\n"
                            % (v, u, t, v, u * 4, t))
                for c in range(4):
                    j = t * 4 + c
                    for rr in range(rows):
                        a = "a%d_%d" % ((u * 16 + j) % nacc, rr)
                        body.append("        %s = %s + xv%d_%d_%d.%s * f32(extractBits(w%d_%d.%s, %du, 2u));\n"
                                    % (a, a, v, u, t, "xyzw"[c], v, rr, "xyzw"[u], 2 * j))
    accs = "".join("    var a%d_%d: f32 = 0.0;\n" % (k, rr)
                   for rr in range(rows) for k in range(nacc))
    rowsdecl = "".join("    let r%d = r0 + %du;\n" % (rr, rr) for rr in range(1, rows))
    tail = "".join("    if (r%d < p.m) { out[r%d] = (%s) + p.corr; }\n"
                   % (rr, rr, " + ".join("a%d_%d" % (k, rr) for k in range(nacc)))
                   for rr in range(rows))
    return HDR4 + """
@compute @workgroup_size(%du)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let r0 = g.x * %du;
    if (r0 >= p.m) { return; }
%s%s    let nvec = p.k / 64u;
    for (var i: u32 = 0u; i < nvec; i = i + %du) {
%s    }
%s}
""" % (wg, rows, rowsdecl, accs, unroll, "".join(body), tail)


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
    rng = np.random.default_rng(5)

    cfgs = [("XVEC r1 n4 u2", dict(rows=1, nacc=4, unroll=2)),
            ("XVEC r1 n8 u4", dict(rows=1, nacc=8, unroll=4)),
            ("XVEC r2 n4 u2", dict(rows=2, nacc=4, unroll=2)),
            ("XVEC r2 n8 u1", dict(rows=2, nacc=8, unroll=1)),
            ("XVEC r4 n4 u1", dict(rows=4, nacc=4, unroll=1)),
            ("XVEC r2 n4 u4", dict(rows=2, nacc=4, unroll=4))]

    for M, K, tag in [(4096, 4096, "attn q/o"), (12288, 4096, "gate+up"), (4096, 12288, "down")]:
        codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
        x = rng.standard_normal(K).astype(np.float32)
        ref = (codes.astype(np.float32) - 1.0) @ x
        p2 = pack2_vec4(codes)
        xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        wb = dev.create_buffer_with_data(data=p2.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        ob = dev.create_buffer(size=M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(
            data=np.array([M, K, K // 16, 1], dtype=np.uint32).tobytes()
            + np.float32(-x.sum()).tobytes() + b"\0" * 12, usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 32}},
        ])

        built = []
        for label, kw in cfgs:
            src = build_xvec(K, **kw)
            m = dev.create_shader_module(code=src)
            pipe = dev.create_compute_pipeline(layout=pl, compute={"module": m, "entry_point": "main"})
            rows = kw["rows"]
            grid = ((M + rows - 1) // rows + 63) // 64

            def run(c, pipe=pipe, grid=grid):
                c.set_pipeline(pipe); c.set_bind_group(0, bg); c.dispatch_workgroups(grid, 1, 1)
            enc = dev.create_command_encoder()
            cc = enc.begin_compute_pass(); run(cc); cc.end()
            dev.queue.submit([enc.finish()])
            got = np.frombuffer(dev.queue.read_buffer(ob, 0, M * 4), dtype=np.float32)
            built.append((label, run, float(np.abs(got - ref).max())))

        res = {b[0]: [] for b in built}
        for _ in range(3):
            for label, run, _e in built:
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
        for label, run, err in built:
            med, best = float(np.median(res[label])), min(res[label])
            print(f"{label:16s} {med*1e3:8.3f} {best*1e3:7.3f} {M*K/med/1e9:8.1f} "
                  f"{p2.nbytes/med/1e9:7.1f}  {err:.1e}")
        print()


if __name__ == "__main__":
    main()
