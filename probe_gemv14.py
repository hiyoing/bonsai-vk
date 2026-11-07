"""Probe 14: the production kernel shape, measured.

probes 11/13 established 145-155 Gop/s for a bare ternary dot. The real Bonsai tensor has one
fp16 scale per 128-weight block, so the accumulator must be segmented at block boundaries:

    for b in blocks:  cp = sum of 2 words of c*x ;  acc = fma(d[b][row], cp - xsum[b], acc)

This is exactly gemv_ternary.comp. The question the report depends on: does the per-block scale
cost anything? It adds 1 scale load + 1 xsum load + 1 cvt + 1 sub + 1 fma + the per-word
accumulator resets and combines, ~12 lane-ops per 128 codes = 3% of the inner loop.
Predicted: 145 -> ~140 Gop/s. Measured below.

Also verified here, because it must not be shipped broken: numerical agreement with a numpy
reference built from the SAME per-block scaled sum (probes 11/13 used a single global
-correction shortcut, which is not what PTQ1_0 needs).
"""
import time
import numpy as np
import wgpu

HDR = """struct P { m: u32, k: u32, splits: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> sc: array<f32>;
@group(0) @binding(3) var<storage, read> xsum: array<f32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
"""

REDUCE = HDR + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var s: f32 = out[row];
    for (var i: u32 = 1u; i < p.splits; i = i + 1u) { s = s + out[i * p.m + row]; }
    out[row] = s;
}
"""


def word_mac(v, out_lines):
    """16 x-loads (vec4) + 64 extractBits/MAC for one uvec4 word, 4 fresh accumulators."""
    out_lines.append("        let w%d = wv[(i0 + %du) * p.m + row];\n" % (v, v))
    for k in range(4):
        out_lines.append("        var a%d_%d: f32 = 0.0;\n" % (v, k))
    for u in range(4):
        for t in range(4):
            out_lines.append("        let xv%d_%d_%d = x[(i0 + %du) * 16u + %du + %du];\n"
                             % (v, u, t, v, u * 4, t))
            for cc in range(4):
                j = t * 4 + cc
                out_lines.append(
                    "        a%d_%d = a%d_%d + xv%d_%d_%d.%s * f32(extractBits(w%d.%s, %du, 2u));\n"
                    % (v, cc, v, cc, v, u, t, "xyzw"[cc], v, "xyzw"[u], 2 * j))
    out_lines.append("        let cp%d = (a%d_0 + a%d_1) + (a%d_2 + a%d_3);\n" % (v, v, v, v, v))


def build():
    lines = []
    word_mac(0, lines)
    word_mac(1, lines)
    lines.append("        acc = acc + sc[b * p.m + row] * ((cp0 + cp1) - xsum[b]);\n")
    loop = "\n        for (var b: u32 = b_lo; b < b_hi; b = b + 1u) {\n            let i0 = b * 2u;\n%s        }\n" % "".join(lines)
    return HDR + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    let nblk = p.k / 128u;
    let per = nblk / p.splits;
    let b_lo = g.y * per;
    let b_hi = b_lo + per;
    var acc: f32 = 0.0;
%s
    if (p.splits == 1u) { out[row] = acc; } else { out[g.y * p.m + row] = acc; }
}
""" % loop


def pack_blocks(codes):
    """row-major codes[M,K] in {0,1,2} -> [K/64][M] vec4<u32>, 2 bits per code."""
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
    entries = []
    for i in range(6):
        t = (wgpu.BufferBindingType.storage if i == 4 else
             wgpu.BufferBindingType.uniform if i == 5 else wgpu.BufferBindingType.read_only_storage)
        entries.append({"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}})
    bgl = dev.create_bind_group_layout(entries=entries)
    layout = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    rng = np.random.default_rng(11)

    src = build()
    mpipe = dev.create_compute_pipeline(
        layout=layout, compute={"module": dev.create_shader_module(code=src), "entry_point": "main"})
    rpipe = dev.create_compute_pipeline(
        layout=layout, compute={"module": dev.create_shader_module(code=REDUCE), "entry_point": "main"})

    for M, K, tag in [(4096, 4096, "attn q/o"), (12288, 4096, "gate+up"), (4096, 12288, "down")]:
        codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
        x = rng.standard_normal(K).astype(np.float32)
        w = codes.astype(np.float32) - 1.0
        nb = K // 128
        sc = (1.0 + 0.1 * rng.standard_normal((nb, M))).astype(np.float32)
        ref = np.zeros(M, dtype=np.float64)
        for b in range(nb):
            xb = x[b * 128:(b + 1) * 128]
            ref += sc[b].astype(np.float64) * (w[:, b * 128:(b + 1) * 128] @ xb)
        p2 = pack_blocks(codes)
        xsum = x.reshape(nb, 128).sum(axis=1).astype(np.float32)

        bufs = {
            0: dev.create_buffer_with_data(data=p2.tobytes(), usage=wgpu.BufferUsage.STORAGE),
            1: dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE),
            2: dev.create_buffer_with_data(data=sc.reshape(-1).tobytes(), usage=wgpu.BufferUsage.STORAGE),
            3: dev.create_buffer_with_data(data=xsum.tobytes(), usage=wgpu.BufferUsage.STORAGE),
        }
        print("=== %s M=%d K=%d  packed %.1f MB ===" % (tag, M, K, p2.nbytes / 1e6))
        for splits in [1, 2, 4, 8]:
            if nb % splits:
                continue
            out = dev.create_buffer(size=M * splits * 4,
                                    usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
            par = dev.create_buffer_with_data(data=np.array([M, K, splits, 0], dtype=np.uint32).tobytes(),
                                              usage=wgpu.BufferUsage.UNIFORM)
            bg = dev.create_bind_group(layout=bgl, entries=[
                {"binding": 0, "resource": {"buffer": bufs[0], "offset": 0, "size": p2.nbytes}},
                {"binding": 1, "resource": {"buffer": bufs[1], "offset": 0, "size": x.nbytes}},
                {"binding": 2, "resource": {"buffer": bufs[2], "offset": 0, "size": sc.nbytes}},
                {"binding": 3, "resource": {"buffer": bufs[3], "offset": 0, "size": xsum.nbytes}},
                {"binding": 4, "resource": {"buffer": out, "offset": 0, "size": M * splits * 4}},
                {"binding": 5, "resource": {"buffer": par, "offset": 0, "size": 16}},
            ])

            def passes(c):
                c.set_pipeline(mpipe); c.set_bind_group(0, bg)
                c.dispatch_workgroups((M + 63) // 64, splits, 1)
                if splits > 1:
                    c.set_pipeline(rpipe); c.set_bind_group(0, bg)
                    c.dispatch_workgroups((M + 63) // 64, 1, 1)
            enc = dev.create_command_encoder()
            cc = enc.begin_compute_pass(); passes(cc); cc.end()
            dev.queue.submit([enc.finish()])
            got = np.frombuffer(dev.queue.read_buffer(out, 0, M * 4), dtype=np.float32)
            rel = float(np.abs(got - ref).max() / np.abs(ref).max())

            res = []
            for _ in range(3):
                enc = dev.create_command_encoder()
                cc = enc.begin_compute_pass()
                for _ in range(5):
                    passes(cc)
                cc.end(); dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(out, 0, 4)
                t0 = time.perf_counter()
                enc = dev.create_command_encoder()
                cc = enc.begin_compute_pass()
                for _ in range(100):
                    passes(cc)
                cc.end(); dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(out, 0, 4)
                res.append((time.perf_counter() - t0) / 100)
            med, best = float(np.median(res)), min(res)
            print("  splits=%-2d med %.3f ms  best %.3f  %6.1f Gop/s  %5.1f GB/s  relerr %.1e"
                  % (splits, med * 1e3, best * 1e3, M * K / med / 1e9, p2.nbytes / med / 1e9, rel))
    print("\nshader source: %d lines, %d bytes" % (src.count("\n"), len(src)))


if __name__ == "__main__":
    main()
