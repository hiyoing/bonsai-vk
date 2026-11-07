"""Probe 10: where does the ternary GEMV actually lose? Read-bandwidth ceiling vs decode cost.

Variants, all on the SAME packed layout ([K/64][M] vec4<u32>):
  READ   - stream the weights, do no MACs at all  -> the layout's achievable GB/s
  XREAD  - stream x only, no weights              -> the cost of the x re-read storm
  BASE   - current best (bfefold, unroll 4, 8 accumulators, 1 row/thread)
  LDS    - BASE + x staged into workgroup memory  -> kills M*K lanes of x traffic
  ROW2   - BASE but each thread owns 2 output rows (x loaded once, reused)
  ROW2L  - ROW2 + LDS

Shapes are the real dense-projection shapes of a 9B-class model (hidden 4096, ffn 12288).
"""
import time
import numpy as np
import wgpu

HDR = """struct P { m: u32, k: u32, kw: u32, rows: u32, corr: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
"""

READ = HDR + """
@compute @workgroup_size(%du)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var a: vec4<u32> = vec4<u32>(0u);
    let nvec = p.k / 64u;
    for (var i: u32 = 0u; i < nvec; i = i + 4u) {
        a = a + wv[i * p.m + row] + wv[(i + 1u) * p.m + row]
            + wv[(i + 2u) * p.m + row] + wv[(i + 3u) * p.m + row];
    }
    if (((a.x ^ a.y) ^ (a.z ^ a.w)) == 0xdeadbeefu) { out[row] = f32(a.x); }
}
""" % 64

XREAD = HDR + """
@compute @workgroup_size(%du)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var a: f32 = 0.0;
    for (var i: u32 = 0u; i < p.k; i = i + 1u) {
        a = a + x[i];
    }
    if (a == 12345.678) { out[row] = a; }
}
""" % 64


def build(kind, K, unroll=4, nacc=8, wg=64, rows=1):
    lds = kind.endswith("L") or kind == "LDS"
    xs = "xs[%s]" if lds else "x[%s]"
    body = []
    for v in range(unroll):
        for rr in range(rows):
            body.append("        let w%d_%d = wv[(i + %du) * p.m + r%d];\n" % (v, rr, v, rr))
    for v in range(unroll):
        for u in range(4):
            for j in range(16):
                col = "i + %du) * 64u + %du" % (v, u * 16 + j)
                for rr in range(rows):
                    a = "a%d_%d" % ((u * 16 + j) % nacc, rr)
                    body.append("        %s = %s + %s * f32(extractBits(w%d_%d.%s, %du, 2u));\n"
                                % (a, a, xs % ("(" + col), v, rr, "xyzw"[u], 2 * j))
    tail = "".join("    out[r%d] = (%s) + p.corr;\n"
                   % (rr, " + ".join("a%d_%d" % (k, rr) for k in range(nacc)))
                   for rr in range(rows))
    accs = "".join("    var a%d_%d: f32 = 0.0;\n" % (k, rr)
                   for rr in range(rows) for k in range(nacc))
    rowsdecl = "".join("    let r%d = r0 + %du;\n" % (rr, rr) for rr in range(1, rows))
    fill = ""
    decl = "var<workgroup> xs: array<f32, %d>;\n" % K if lds else ""
    if lds:
        fill = ("    for (var t: u32 = l.x; t < p.k; t = t + %du) { xs[t] = x[t]; }\n"
                "    workgroupBarrier();\n" % wg)
    return HDR + decl + """
@compute @workgroup_size(%du)
fn main(@builtin(global_invocation_id) g: vec3<u32>, @builtin(local_invocation_id) l: vec3<u32>) {
    let r0 = g.x * %du;
    if (r0 >= p.m) { return; }
%s%s%s    let nvec = p.k / 64u;
    for (var i: u32 = 0u; i < nvec; i = i + %du) {
%s    }
%s}
""" % (wg, rows, fill, rowsdecl, accs, unroll, "".join(body), tail)


def pack2_vec4(codes):
    M, K = codes.shape
    nvec = K // 64
    out = np.zeros((nvec, M, 4), dtype=np.uint32)
    for i in range(nvec):
        for u in range(4):
            w = np.zeros(M, dtype=np.uint32)
            for j in range(16):
                w |= codes[:, i * 64 + u * 16 + j].astype(np.uint32) << np.uint32(2 * j)
            out[i, :, u] = w
    return out


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
            + np.float32(-x.sum()).tobytes() + b"\0" * 12,
            usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 32}},
        ])

        cfgs = [("READ", READ, 64, 1), ("XREAD", XREAD, 64, 1),
                ("BASE", build("BASE", K), 64, 1),
                ("BASE/128", build("BASE", K, wg=128), 128, 1),
                ("LDS", build("LDS", K), 64, 1),
                ("ROW2", build("ROW2", K, rows=2), 64, 2),
                ("ROW2L", build("ROW2L", K, rows=2), 64, 2)]
        built = []
        for label, src, wg, rows in cfgs:
            m = dev.create_shader_module(code=src)
            pipe = dev.create_compute_pipeline(layout=pl, compute={"module": m, "entry_point": "main"})
            n = (M + rows - 1) // rows
            grid = (n + wg - 1) // wg

            def run(c, pipe=pipe, grid=grid):
                c.set_pipeline(pipe); c.set_bind_group(0, bg); c.dispatch_workgroups(grid, 1, 1)
            enc = dev.create_command_encoder()
            c = enc.begin_compute_pass(); run(c); c.end()
            dev.queue.submit([enc.finish()])
            err = float("nan")
            if label not in ("READ", "XREAD"):
                got = np.frombuffer(dev.queue.read_buffer(ob, 0, M * 4), dtype=np.float32)
                err = float(np.abs(got - ref).max())
            built.append((label, run, err, rows))

        res = {b[0]: [] for b in built}
        for _ in range(3):
            for label, run, _e, rows in built:
                rep = 60 if K == 12288 and rows == 1 else 100
                enc = dev.create_command_encoder()
                c = enc.begin_compute_pass()
                for _ in range(3):
                    run(c)
                c.end()
                dev.queue.submit([enc.finish()])
                t0 = time.perf_counter()
                enc = dev.create_command_encoder()
                c = enc.begin_compute_pass()
                for _ in range(rep):
                    run(c)
                c.end()
                dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(ob, 0, 4)
                res[label].append((time.perf_counter() - t0) / rep)

        print("=== %s  M=%d K=%d  weights=%.2f MB ===" % (tag, M, K, p2.nbytes / 1e6))
        print(f"{'variant':10s} {'med ms':>8s} {'best ms':>8s} {'Gop/s':>8s} {'GB/s':>7s}  err")
        for label, run, err, rows in built:
            med, best = float(np.median(res[label])), min(res[label])
            gb = p2.nbytes / med / 1e9
            print(f"{label:10s} {med*1e3:8.3f} {best*1e3:8.3f} {M*K/med/1e9:8.1f} {gb:7.1f}  {err:.1e}")
        print()


if __name__ == "__main__":
    main()
