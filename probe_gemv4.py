"""Probe 7: corrected + vectorized (vec4<u32> loads) ternary GEMV, GPU-only timing."""
import time
import numpy as np
import wgpu


def term(word, shift, x_index, acc):
    c = "((%s >> %du) & 3u)" % (word, shift)
    return "        %s = %s + x[%s] * select(select(1.0, 0.0, %s == 1u), -1.0, %s == 0u);\n" % (
        acc, acc, x_index, c, c)


def make_2bit(unroll_vec=1):
    body = []
    for v in range(unroll_vec):
        body.append("        let w%d = wv[(i + %du) * p.m + row];\n" % (v, v))
    accs = ["a%d" % k for k in range(8)]
    for v in range(unroll_vec):
        for u in range(4):
            for j in range(16):
                k = (u * 16 + j) % 8
                idx = "(i + %du) * 64u + %du" % (v, u * 16 + j)
                body.append(term("w%d.%s" % (v, "xyzw"[u]), 2 * j, idx, accs[k]))
    return """
struct P { m: u32, k: u32, kw: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var a4: f32 = 0.0; var a5: f32 = 0.0; var a6: f32 = 0.0; var a7: f32 = 0.0;
    let nvec = p.k / 64u;
    for (var i: u32 = 0u; i < nvec; i = i + %du) {
%s    }
    y[row] = %s;
}
""" % (unroll_vec, "".join(body), " + ".join(accs))


def make_i8():
    body = []
    accs = ["a%d" % k for k in range(8)]
    for u in range(4):
        for j in range(4):
            k = (u * 4 + j) % 8
            b = "b_%d_%d" % (u, j)
            body.append("        let %s = bitcast<i32>(wv[i * p.m + row].%s << %du) >> 24;\n" % (b, "xyzw"[u], 8 * j))
            body.append("        %s = %s + x[i * 16u + %du] * f32(%s);\n" % (accs[k], accs[k], u * 4 + j, b))
    return """
struct P { m: u32, k: u32, kw: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> wv: array<vec4<u32>>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    var a4: f32 = 0.0; var a5: f32 = 0.0; var a6: f32 = 0.0; var a7: f32 = 0.0;
    let nvec = p.k / 16u;
    for (var i: u32 = 0u; i < nvec; i = i + 1u) {
%s    }
    y[row] = %s;
}
""" % ("".join(body), " + ".join(accs))


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


def pack_i8_vec4(codes):
    M, K = codes.shape
    nvec = K // 16
    vals = (codes.astype(np.int8) - 1).astype(np.uint32)
    out = np.zeros((nvec, M, 4), dtype=np.uint32)
    for i in range(nvec):
        for u in range(4):
            w = np.zeros(M, dtype=np.uint32)
            for j in range(4):
                w |= (vals[:, i * 16 + u * 4 + j] & 0xFF) << np.uint32(8 * j)
            out[i, :, u] = w
    return out


def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()
    bgl = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    rng = np.random.default_rng(5)
    M, K = 4096, 2048
    codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
    w = codes.astype(np.float32) - 1.0
    x = rng.standard_normal(K).astype(np.float32)
    ref = w @ x
    p2 = pack2_vec4(codes)
    pi8 = pack_i8_vec4(codes)
    xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)

    def bench(label, code, wdata, kw, repeat=200):
        shader = dev.create_shader_module(code=code)
        pipe = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[bgl]),
                                           compute={"module": shader, "entry_point": "main"})
        wb = dev.create_buffer_with_data(data=wdata.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        yb = dev.create_buffer(size=M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(data=np.array([M, K, kw, 1], dtype=np.uint32).tobytes(),
                                          usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": wdata.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": yb, "offset": 0, "size": M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 16}},
        ])
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((M + 63) // 64, 1, 1); cp.end()
        dev.queue.submit([enc.finish()])
        got = np.frombuffer(dev.queue.read_buffer(yb, 0, M * 4), dtype=np.float32)
        err = np.abs(got - ref).max()

        def run(n):
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
            for _ in range(n):
                cp.dispatch_workgroups((M + 63) // 64, 1, 1)
            cp.end()
            dev.queue.submit([enc.finish()])
        run(3); dev.queue.read_buffer(yb, 0, 4)
        t0 = time.perf_counter(); run(repeat); dev.queue.read_buffer(yb, 0, 4); t1 = time.perf_counter()
        dt = (t1 - t0) / repeat
        print(f"{label:16s} {dt*1e3:7.3f} ms  {M*K/dt/1e9:7.1f} Gop/s  {wdata.nbytes/dt/1e9:6.1f} GB/s  err={err:.1e}")

    bench("2bit vec1", make_2bit(1), p2, K // 16)
    bench("2bit vec2", make_2bit(2), p2, K // 16)
    bench("2bit vec4", make_2bit(4), p2, K // 16)
    bench("i8 vec", make_i8(), pi8, K // 16)


if __name__ == "__main__":
    main()
