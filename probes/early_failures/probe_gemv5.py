"""Probe 8: cheaper ternary decode (code-1) + split-K occupancy + LUT decode."""
import time
import numpy as np
import wgpu

HDR = """struct P { m: u32, k: u32, kw: u32, splits: u32 };
@group(0) @binding(0) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
"""

DECL = """    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var a4: f32 = 0.0; var a5: f32 = 0.0; var a6: f32 = 0.0; var a7: f32 = 0.0;
"""


def _sel_term(word, shift, idx, acc):
    c = "((%s >> %du) & 3u)" % (word, shift)
    return "        %s = %s + x[%s] * select(select(1.0, 0.0, %s == 1u), -1.0, %s == 0u);\n" % (
        acc, acc, idx, c, c)


def _sub_term(word, shift, idx, acc):
    return "        %s = %s + x[%s] * (f32((%s >> %du) & 3u) - 1.0);\n" % (
        acc, acc, idx, word, shift)


def make_shader(kind, unroll_vec=4, splits=1, nacc=8):
    """kind: 'sel' | 'sub' | 'lut'.  splits=1 writes y directly."""
    body = []
    accs = ["a%d" % k for k in range(nacc)]

    if kind == "lut":
        for v in range(unroll_vec):
            body.append("        let w%d = wv[(i + %du) * p.m + row];\n" % (v, v))
            for u in range(4):
                for b in range(4):
                    body.append("        let v%d_%d = lut[(w%d.%s >> %du) & 255u];\n"
                                % (v, u * 4 + b, v, "xyzw"[u], 8 * b))
                    for t in range(4):
                        k = (u * 16 + b * 4 + t) % nacc
                        idx = "(i + %du) * 64u + %du" % (v, u * 16 + b * 4 + t)
                        body.append("        %s = %s + x[%s] * v%d_%d.%s;\n"
                                    % (accs[k], accs[k], idx, v, u * 4 + b, "xyzw"[t]))
    else:
        term = _sub_term if kind == "sub" else _sel_term
        for v in range(unroll_vec):
            body.append("        let w%d = wv[(i + %du) * p.m + row];\n" % (v, v))
        for v in range(unroll_vec):
            for u in range(4):
                for j in range(16):
                    k = (u * 16 + j) % nacc
                    idx = "(i + %du) * 64u + %du" % (v, u * 16 + j)
                    body.append(term("w%d.%s" % (v, "xyzw"[u]), 2 * j, idx, accs[k]))

    if splits == 1:
        head = "    let nvec = p.k / 64u;\n    let lo: u32 = 0u;\n    let hi: u32 = nvec;\n"
        store = "    y_dummy(row);\n"
        store = "    out[row] = %s;\n" % " + ".join(accs)
    else:
        head = ("    let nvec = p.k / 64u;\n    let per = nvec / p.splits;\n"
                "    let lo: u32 = g.y * per;\n    let hi: u32 = lo + per;\n")
        store = "    out[g.y * p.m + row] = %s;\n" % " + ".join(accs)

    lut_decl = ("@group(0) @binding(4) var<storage, read> lut: array<vec4<f32>>;\n"
                if kind == "lut" else "")

    decl = "".join("    var %s: f32 = 0.0;\n" % a for a in accs)

    return HDR + lut_decl + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
%s%s
    for (var i: u32 = lo; i < hi; i = i + %du) {
%s    }
%s}
""" % (decl, head, unroll_vec, "".join(body), store)


REDUCE = HDR + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var s: f32 = 0.0;
    for (var i: u32 = 0u; i < p.splits; i = i + 1u) {
        s = s + out[i * p.m + row];
    }
    out[g.y * p.m + p.m * p.splits + row] = s;
}
"""


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


def make_lut():
    lut = np.zeros((256, 4), dtype=np.float32)
    for b in range(256):
        for t in range(4):
            lut[b, t] = float((b >> (2 * t)) & 3) - 1.0
    return lut


def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()
    info = dict(ad.info)
    print("adapter:", info.get("device"), "|", info.get("backend_type"), "\n")

    def bgl(n):
        e = [
            {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
            {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
            {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
        ]
        if n == 5:
            e.append({"binding": 4, "visibility": wgpu.ShaderStage.COMPUTE,
                      "buffer": {"type": wgpu.BufferBindingType.read_only_storage}})
        return dev.create_bind_group_layout(entries=e)

    layouts = {4: bgl(4), 5: bgl(5)}
    rng = np.random.default_rng(5)
    M, K = 4096, 2048
    codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
    x = rng.standard_normal(K).astype(np.float32)
    ref = (codes.astype(np.float32) - 1.0) @ x
    p2 = pack2_vec4(codes)
    lut = make_lut()
    xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    lb = dev.create_buffer_with_data(data=lut.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    wb = dev.create_buffer_with_data(data=p2.tobytes(), usage=wgpu.BufferUsage.STORAGE)

    def pipe(code, nbind):
        m = dev.create_shader_module(code=code)
        return dev.create_compute_pipeline(
            layout=dev.create_pipeline_layout(bind_group_layouts=[layouts[nbind]]),
            compute={"module": m, "entry_point": "main"})

    def bench(label, shader, splits=1, repeat=200):
        p = pipe(shader, 5 if "lut" in shader else 4)
        nbuf = M * splits * 4
        ob = dev.create_buffer(size=nbuf + M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(
            data=np.array([M, K, K // 16, splits], dtype=np.uint32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
        ents = [
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": nbuf + M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 16}},
        ]
        if "lut" in shader:
            ents.append({"binding": 4, "resource": {"buffer": lb, "offset": 0, "size": lut.nbytes}})
        bg = dev.create_bind_group(layout=layouts[5 if "lut" in shader else 4], entries=ents)

        red = None
        if splits > 1:
            rp = pipe(REDUCE, 4)
            rpar = dev.create_buffer_with_data(
                data=np.array([M, K, K // 16, splits], dtype=np.uint32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
            rbg = dev.create_bind_group(layout=layouts[4], entries=[
                {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
                {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
                {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": nbuf + M * 4}},
                {"binding": 3, "resource": {"buffer": rpar, "offset": 0, "size": 16}},
            ])
            # reduce writes to ob[M*splits + row]; y_dummy signature kept simple: read that region
            red = (rp, rbg)

        def enc_once():
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(p); cp.set_bind_group(0, bg)
            cp.dispatch_workgroups((M + 63) // 64, splits, 1)
            cp.end()
            encs = [enc]
            if red:
                e2 = dev.create_command_encoder()
                c2 = e2.begin_compute_pass()
                c2.set_pipeline(red[0]); c2.set_bind_group(0, red[1])
                c2.dispatch_workgroups((M + 63) // 64, 1, 1)
                c2.end()
                encs.append(e2)
            return encs

        for e in enc_once():
            dev.queue.submit([e.finish()])
        off = M * splits * 4 if splits > 1 else 0
        got = np.frombuffer(dev.queue.read_buffer(ob, off, M * 4), dtype=np.float32)
        err = float(np.abs(got - ref).max())

        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass()
        cp.set_pipeline(p); cp.set_bind_group(0, bg)
        for _ in range(5):
            cp.dispatch_workgroups((M + 63) // 64, splits, 1)
        cp.end()
        dev.queue.submit([enc.finish()])
        dev.queue.read_buffer(ob, 0, 4)

        def run(n):
            e = dev.create_command_encoder()
            c = e.begin_compute_pass()
            for _ in range(n):
                c.set_pipeline(p); c.set_bind_group(0, bg)
                c.dispatch_workgroups((M + 63) // 64, splits, 1)
                if red:
                    c.set_pipeline(red[0]); c.set_bind_group(0, red[1])
                    c.dispatch_workgroups((M + 63) // 64, 1, 1)
            c.end()
            dev.queue.submit([e.finish()])
        t0 = time.perf_counter(); run(repeat)
        dev.queue.read_buffer(ob, off, 4); t1 = time.perf_counter()
        dt = (t1 - t0) / repeat
        print(f"{label:20s} {dt*1e3:7.3f} ms  {M*K/dt/1e9:7.1f} Gop/s  "
              f"{p2.nbytes/dt/1e9:6.1f} GB/s  err={err:.2e}")

    bench("sel  (baseline)", make_shader("sel", 4), 1)
    bench("sub  vec4", make_shader("sub", 4), 1)
    bench("sub  vec8/acc16", make_shader("sub", 8, nacc=16), 1)
    bench("sub  split-K2", make_shader("sub", 4, splits=2), 2)
    bench("sub  split-K4", make_shader("sub", 4, splits=4), 4)
    bench("sub  split-K8", make_shader("sub", 4, splits=8), 8)
    bench("sub  K4+acc16", make_shader("sub", 4, splits=4, nacc=16), 4)
    bench("lut  vec4", make_shader("lut", 4), 1)
    bench("lut  split-K4", make_shader("lut", 2, splits=4), 4)


if __name__ == "__main__":
    main()
