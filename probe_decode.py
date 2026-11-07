"""Probe 4: decode-strategy shootout for ternary GEMV + overhead baseline.
All kernels: one workgroup (64 lanes) per output row, split-K, tree reduce.
"""
import time
import numpy as np
import wgpu

COMMON = """
struct P { m: u32, k: u32, kw: u32, pad: u32 };
@group(0) @binding(3) var<uniform> p: P;
var<workgroup> red: array<f32, 64>;
"""

REDUCE = """
    red[lid.x] = acc;
    workgroupBarrier();
    var s: u32 = 32u;
    loop { if (s == 0u) { break; }
        if (lid.x < s) { red[lid.x] = red[lid.x] + red[lid.x + s]; }
        workgroupBarrier(); s = s / 2u; }
    if (lid.x == 0u) { y[wid.x] = red[0]; }
"""

SH_F32 = COMMON + """
@group(0) @binding(0) var<storage, read> w: array<f32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x; if (row >= p.m) { return; }
    let base = row * p.k;
    var acc: f32 = 0.0;
    var t: u32 = lid.x;
    loop { if (t >= p.k) { break; }
        acc = acc + w[base + t] * x[t];
        t = t + 64u; }
""" + REDUCE + """
}
"""

SH_2BIT_SEL = COMMON + """
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x; if (row >= p.m) { return; }
    let base = row * p.kw;
    var acc: f32 = 0.0;
    var t: u32 = lid.x;
    loop { if (t >= p.kw) { break; }
        let word = w[base + t];
        let kk = t * 16u;
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let c = (word >> (2u * j)) & 3u;
            let s = select(select(1.0, 0.0, c == 1u), -1.0, c == 0u);
            acc = acc + x[kk + j] * s;
        }
        t = t + 64u; }
""" + REDUCE + """
}
"""

SH_2BIT_TBL = COMMON + """
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
var<workgroup> tbl: array<f32, 4>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x; if (row >= p.m) { return; }
    if (lid.x < 4u) { tbl[lid.x] = f32(i32(lid.x)) - 1.0; }
    workgroupBarrier();
    let base = row * p.kw;
    var acc: f32 = 0.0;
    var t: u32 = lid.x;
    loop { if (t >= p.kw) { break; }
        let word = w[base + t];
        let kk = t * 16u;
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let c = (word >> (2u * j)) & 3u;
            acc = acc + x[kk + j] * tbl[c];
        }
        t = t + 64u; }
""" + REDUCE + """
}
"""

SH_I8 = COMMON + """
@group(0) @binding(0) var<storage, read> w: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x; if (row >= p.m) { return; }
    let base = row * p.kw;     // kw = k/4 bytes packed as u32
    var acc: f32 = 0.0;
    var t: u32 = lid.x;
    loop { if (t >= p.kw) { break; }
        let word = w[base + t];
        let kk = t * 4u;
        for (var j: u32 = 0u; j < 4u; j = j + 1u) {
            let c = (word >> (8u * j)) & 255u;
            acc = acc + x[kk + j] * f32(i32(c)) - x[kk + j];
        }
        t = t + 64u; }
""" + REDUCE + """
}
"""


def pack2(codes):
    M, K = codes.shape
    out = np.zeros((M, K // 16), dtype=np.uint32)
    for t in range(K // 16):
        w = np.zeros(M, dtype=np.uint32)
        for j in range(16):
            w |= codes[:, t * 16 + j].astype(np.uint32) << np.uint32(2 * j)
        out[:, t] = w
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
    rng = np.random.default_rng(2)
    M, K = 2048, 2048
    codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
    wf = (codes.astype(np.float32) - 1.0)
    x = rng.standard_normal(K).astype(np.float32)
    ref = wf @ x
    p2 = pack2(codes)
    i8 = (codes.astype(np.uint32).reshape(M, K // 4, 4) << np.arange(0, 32, 8, dtype=np.uint32)).sum(axis=2).astype(np.uint32)
    xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    yb = dev.create_buffer(size=M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    def bench(name, code, wdata, kw, words_per_row_bytes):
        shader = dev.create_shader_module(code=code)
        pipe = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[bgl]),
                                           compute={"module": shader, "entry_point": "main"})
        wb = dev.create_buffer_with_data(data=wdata.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        par = dev.create_buffer_with_data(data=np.array([M, K, kw, 0], dtype=np.uint32).tobytes(),
                                          usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": wdata.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": yb, "offset": 0, "size": M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 16}},
        ])
        def run(n):
            enc = dev.create_command_encoder()
            for _ in range(n):
                cp = enc.begin_compute_pass()
                cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
                cp.dispatch_workgroups(M, 1, 1)
                cp.end()
            dev.queue.submit([enc.finish()])
        run(1)
        got = np.frombuffer(dev.queue.read_buffer(yb, 0, M * 4), dtype=np.float32)
        err = np.abs(got - ref).max()
        n = 100
        run(3); dev.queue.read_buffer(yb, 0, 4)
        t0 = time.perf_counter(); run(n); dev.queue.read_buffer(yb, 0, 4); t1 = time.perf_counter()
        dt = (t1 - t0) / n
        print(f"{name:10s} {dt*1e3:7.3f} ms/dispatch  {M*K/dt/1e9:7.1f} Gop/s  err={err:.1e}  weight_bytes={wdata.nbytes/1e6:.1f}MB")
        return dt

    bench("f32", SH_F32, wf, K, 4 * K)
    bench("2bit_sel", SH_2BIT_SEL, p2, K // 16, K // 4)
    bench("2bit_tbl", SH_2BIT_TBL, p2, K // 16, K // 4)
    bench("i8", SH_I8, i8, K // 4, K)

    # empty dispatch baseline
    print("(wall clock includes submit+sync; baseline below)")
    enc = dev.create_command_encoder()
    t0 = time.perf_counter()
    for _ in range(100):
        cp = enc.begin_compute_pass(); cp.end()
    dev.queue.submit([enc.finish()])
    dev.queue.read_buffer(yb, 0, 4)
    print(f"empty x100: {(time.perf_counter()-t0)*1e3:.2f} ms total")

if __name__ == "__main__":
    main()
