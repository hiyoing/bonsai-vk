"""Probe 9: fold the -1 offset out of the inner loop + single-instruction bitfield extract."""
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
    if kind == "sub":
        return ("        %s = %s + x[%s] * (f32((%s >> %du) & 3u) - 1.0);\n"
                % (acc, acc, idx, word, shift))
    if kind == "subfold":
        return ("        %s = %s + x[%s] * f32((%s >> %du) & 3u);\n"
                % (acc, acc, idx, word, shift))
    if kind == "bfe":
        return ("        %s = %s + x[%s] * (f32(extractBits(%s, %du, 2u)) - 1.0);\n"
                % (acc, acc, idx, word, shift))
    if kind == "bfefold":
        return ("        %s = %s + x[%s] * f32(extractBits(%s, %du, 2u));\n"
                % (acc, acc, idx, word, shift))
    raise ValueError(kind)


def make_shader(kind, unroll_vec=4, splits=1, nacc=8):
    fold = kind.endswith("fold")
    body = []
    accs = ["a%d" % k for k in range(nacc)]
    for v in range(unroll_vec):
        body.append("        let w%d = wv[(i + %du) * p.m + row];\n" % (v, v))
    for v in range(unroll_vec):
        for u in range(4):
            for j in range(16):
                k = (u * 16 + j) % nacc
                idx = "(i + %du) * 64u + %du" % (v, u * 16 + j)
                body.append(term(kind, "w%d.%s" % (v, "xyzw"[u]), 2 * j, idx, accs[k]))

    if splits == 1:
        head = "    let nvec = p.k / 64u;\n    let lo: u32 = 0u;\n    let hi: u32 = nvec;\n"
        target = "    out[row] = %s;\n" % " + ".join(accs)
    else:
        head = ("    let nvec = p.k / 64u;\n    let per = nvec / p.splits;\n"
                "    let lo: u32 = g.y * per;\n    let hi: u32 = lo + per;\n")
        target = "    out[g.y * p.m + row] = %s;\n" % " + ".join(accs)

    if fold:
        if splits == 1:
            target = "    out[row] = (%s) + p.corr;\n" % " + ".join(accs)
        else:
            target = "    out[g.y * p.m + row] = %s;\n" % " + ".join(accs)

    decl = "".join("    var %s: f32 = 0.0;\n" % a for a in accs)
    return HDR + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
%s%s
    for (var i: u32 = lo; i < hi; i = i + %du) {
%s    }
%s}
""" % (decl, head, unroll_vec, "".join(body), target)


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

    def bench(label, shader, splits=1, repeat=200, folded=False):
        m = dev.create_shader_module(code=shader)
        p = dev.create_compute_pipeline(layout=pl, compute={"module": m, "entry_point": "main"})
        nbuf = M * splits * 4
        ob = dev.create_buffer(size=nbuf + M * 4,
                               usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(
            data=np.array([M, K, K // 16, splits], dtype=np.uint32).tobytes()
            + np.float32(corr if folded else 0.0).tobytes() + b"\0" * 12,
            usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": nbuf + M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 32}},
        ])
        rp = None
        if splits > 1:
            rm = dev.create_shader_module(code=REDUCE)
            rp = dev.create_compute_pipeline(layout=pl, compute={"module": rm, "entry_point": "main"})

        def passes(c):
            c.set_pipeline(p); c.set_bind_group(0, bg)
            c.dispatch_workgroups((M + 63) // 64, splits, 1)
            if rp:
                c.set_pipeline(rp); c.set_bind_group(0, bg)
                c.dispatch_workgroups((M + 63) // 64, 1, 1)

        enc = dev.create_command_encoder()
        c = enc.begin_compute_pass(); passes(c); c.end()
        dev.queue.submit([enc.finish()])
        off = nbuf if splits > 1 else 0
        got = np.frombuffer(dev.queue.read_buffer(ob, off, M * 4), dtype=np.float32)
        err = float(np.abs(got - ref).max())

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
        for _ in range(repeat):
            passes(c)
        c.end()
        dev.queue.submit([enc.finish()])
        dev.queue.read_buffer(ob, off, 4)
        dt = (time.perf_counter() - t0) / repeat
        print(f"{label:22s} {dt*1e3:7.3f} ms  {M*K/dt/1e9:7.1f} Gop/s  "
              f"{p2.nbytes/dt/1e9:6.1f} GB/s  err={err:.2e}")

    bench("sel      (ref)", make_shader("sel", 4))
    bench("sub", make_shader("sub", 4))
    bench("sub+fold", make_shader("subfold", 4), folded=True)
    bench("bfe      (4 ops)", make_shader("bfe", 4))
    bench("bfe+fold (3 ops)", make_shader("bfefold", 4), folded=True)
    bench("bfe+fold vec8", make_shader("bfefold", 8), folded=True)
    bench("bfe+fold K2", make_shader("bfefold", 4, splits=2), 2, folded=True)
    bench("bfe+fold K4", make_shader("bfefold", 4, splits=4), 4, folded=True)
    bench("bfe+fold K8", make_shader("bfefold", 4, splits=8), 8, folded=True)
    bench("bfe+fold K4/acc16", make_shader("bfefold", 4, splits=4, nacc=16), 4, folded=True)


if __name__ == "__main__":
    main()
