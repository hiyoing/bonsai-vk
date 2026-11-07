"""Probe 10: stable multi-round measurement of the best ternary kernels + wg/occupancy sweep."""
import time
import numpy as np
import wgpu

HDR = """struct P { m: u32, k: u32, kw: u32, splits: u32, corr: f32, pad0: u32, pad1: u32, pad2: u32 };
@group(0) @binding(0) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
"""

REDUCE = HDR + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < p.splits; i = i + 1u) {
        s = s + out[i * p.m + row];
    }
    out[g.y * p.m + p.m * p.splits + row] = s + p.corr;
}
"""


def term(kind, word, shift, idx, acc):
    if kind == "sel":
        c = "((%s >> %du) & 3u)" % (word, shift)
        return ("        %s = %s + x[%s] * select(select(1.0, 0.0, %s == 1u), -1.0, %s == 0u);\n"
                % (acc, acc, idx, c, c))
    if kind == "subfold":
        return "        %s = %s + x[%s] * f32((%s >> %du) & 3u);\n" % (acc, acc, idx, word, shift)
    if kind == "bfefold":
        return "        %s = %s + x[%s] * f32(extractBits(%s, %du, 2u));\n" % (acc, acc, idx, word, shift)
    raise ValueError(kind)


def make_shader(kind, unroll_vec=4, splits=1, nacc=8, wg=64, inc=False):
    fold = kind.endswith("fold")
    accs = ["a%d" % k for k in range(nacc)]
    body = []
    for v in range(unroll_vec):
        idx = "base + %du * p.m" % v if inc else "(i + %du) * p.m + row" % v
        body.append("        let w%d = wv[%s];\n" % (v, idx))
    for v in range(unroll_vec):
        for u in range(4):
            for j in range(16):
                k = (u * 16 + j) % nacc
                xi = "(i + %du) * 64u + %du" % (v, u * 16 + j)
                body.append(term(kind, "w%d.%s" % (v, "xyzw"[u]), 2 * j, xi, accs[k]))
    if inc:
        body.append("        base = base + %du * p.m;\n" % unroll_vec)

    pre = "    var base: u32 = lo * p.m + row;\n" if inc else ""
    if splits == 1:
        head = "    let nvec = p.k / 64u;\n    let lo: u32 = 0u;\n    let hi: u32 = nvec;\n"
        tail = "    out[row] = (%s) + p.corr;\n" % " + ".join(accs) if fold else \
               "    out[row] = %s;\n" % " + ".join(accs)
    else:
        head = ("    let nvec = p.k / 64u;\n    let per = nvec / p.splits;\n"
                "    let lo: u32 = g.y * per;\n    let hi: u32 = lo + per;\n")
        tail = "    out[g.y * p.m + row] = %s;\n" % " + ".join(accs)

    decl = "".join("    var %s: f32 = 0.0;\n" % a for a in accs)
    return HDR + """
@compute @workgroup_size(%du)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
%s%s%s
    for (var i: u32 = lo; i < hi; i = i + %du) {
%s    }
%s}
""" % (wg, decl, head, pre, unroll_vec, "".join(body), tail)


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

    e = [
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ]
    bgl = dev.create_bind_group_layout(entries=e)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])

    rng = np.random.default_rng(5)
    M, K = 4096, 2048
    codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
    x = rng.standard_normal(K).astype(np.float32)
    ref = (codes.astype(np.float32) - 1.0) @ x
    p2 = pack2_vec4(codes)
    xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    wb = dev.create_buffer_with_data(data=p2.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    corr = float(-x.sum())

    cfgs = []
    for label, kind, uv, sp, na, wg, inc, folded in [
        ("sel      K1/64",  "sel",     4, 1, 8,  64, False, False),
        ("subfold  K4/64",  "subfold", 4, 4, 8,  64, False, True),
        ("bfefold  K1/64",  "bfefold", 4, 1, 8,  64, False, True),
        ("bfefold  K4/64",  "bfefold", 4, 4, 8,  64, False, True),
        ("bfefold  K4/128", "bfefold", 4, 4, 8, 128, False, True),
        ("bfefold  K4/256", "bfefold", 4, 4, 8, 256, False, True),
        ("bfefold  K4/64acc4", "bfefold", 4, 4, 4, 64, False, True),
        ("bfefold  K8/128", "bfefold", 4, 8, 8, 128, False, True),
        ("bfefold  K4/64inc", "bfefold", 4, 4, 8, 64, True, True),
        ("bfefold  K8/64inc", "bfefold", 4, 8, 8, 64, True, True),
    ]:
        cfgs.append((label, make_shader(kind, uv, sp, na, wg, inc), sp, na, wg, folded))

    built = []
    for label, shader, sp, na, wg, folded in cfgs:
        m = dev.create_shader_module(code=shader)
        p = dev.create_compute_pipeline(layout=pl, compute={"module": m, "entry_point": "main"})
        nbuf = M * sp * 4
        ob = dev.create_buffer(size=nbuf + M * 4,
                               usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(
            data=np.array([M, K, K // 16, sp], dtype=np.uint32).tobytes()
            + np.float32(corr if folded else 0.0).tobytes() + b"\0" * 12,
            usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": nbuf + M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 32}},
        ])
        rp = None
        if sp > 1:
            rm = dev.create_shader_module(code=REDUCE)
            rp = dev.create_compute_pipeline(layout=pl, compute={"module": rm, "entry_point": "main"})

        def passes(c, p=p, bg=bg, rp=rp, sp=sp, wg=wg):
            c.set_pipeline(p); c.set_bind_group(0, bg)
            c.dispatch_workgroups((M + wg - 1) // wg, sp, 1)
            if rp:
                c.set_pipeline(rp); c.set_bind_group(0, bg)
                c.dispatch_workgroups((M + 63) // 64, 1, 1)

        enc = dev.create_command_encoder()
        c = enc.begin_compute_pass(); passes(c); c.end()
        dev.queue.submit([enc.finish()])
        off = nbuf if sp > 1 else 0
        got = np.frombuffer(dev.queue.read_buffer(ob, off, M * 4), dtype=np.float32)
        err = float(np.abs(got - ref).max())
        built.append((label, passes, ob, off, err))

    results = {b[0]: [] for b in built}
    for _ in range(3):
        for label, passes, ob, off, _ in built:
            enc = dev.create_command_encoder()
            c = enc.begin_compute_pass()
            for _ in range(5):
                passes(c)
            c.end()
            dev.queue.submit([enc.finish()])
            dev.queue.read_buffer(ob, 0, 4)

            t0 = time.perf_counter()
            enc = dev.create_command_encoder()
            c = enc.begin_compute_pass()
            for _ in range(100):
                passes(c)
            c.end()
            dev.queue.submit([enc.finish()])
            dev.queue.read_buffer(ob, off, 4)
            results[label].append((time.perf_counter() - t0) / 100)

    print(f"{'kernel':22s} {'best ms':>8s} {'Gop/s':>8s} {'GB/s':>7s} {'median':>8s}  err")
    for label, passes, ob, off, err in built:
        dt = min(results[label])
        med = float(np.median(results[label]))
        print(f"{label:22s} {dt*1e3:8.3f} {M*K/dt/1e9:8.1f} {p2.nbytes/dt/1e9:7.1f} "
              f"{M*K/med/1e9:8.1f}  {err:.2e}")


if __name__ == "__main__":
    main()
