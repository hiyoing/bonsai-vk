"""Probe 6: GPU-only GEMV timing (single pass, N dispatches), format + workgroup sweep."""
import time
import numpy as np
import wgpu

def wgsl_2bit(mult=1):
    return f"""
struct P {{ m: u32, k: u32, kw: u32, batch: u32 }};
@group(0) @binding(0) var<storage, read> w: array<u32>;   // [kw][m]
@group(0) @binding(1) var<storage, read> x: array<f32>;   // [k]
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // [m]
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size({mult * 64})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {{
    let rows_per = 1u;
    let row = g.x;
    if (row >= p.m) {{ return; }}
    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    for (var t: u32 = 0u; t < p.kw; t = t + 1u) {{
        let word = w[t * p.m + row];
        let kk = t * 16u;
        for (var j: u32 = 0u; j < 16u; j = j + 2u) {{
            let c0 = (word >> (2u * j)) & 3u;
            let c1 = (word >> (2u * (j + 1u))) & 3u;
            let s0 = select(select(1.0, 0.0, c0 == 1u), -1.0, c0 == 0u);
            let s1 = select(select(1.0, 0.0, c1 == 1u), -1.0, c1 == 0u);
            a0 = a0 + x[kk + j] * s0;
            a1 = a1 + x[kk + j + 1u] * s1;
        }}
        a2 = a2 + a0; a3 = a3 + a1;
    }}
    y[row] = (a2 + a3);
}}
"""

def wgsl_i8():
    return """
struct P { m: u32, k: u32, kw: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;   // [k/4][m], signed bytes
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    for (var t: u32 = 0u; t < p.kw; t = t + 1u) {
        let word = w[t * p.m + row];
        let kk = t * 4u;
        let b0 = (word << 24u) >> 24u;
        let b1 = (word << 16u) >> 24u;
        let b2 = (word << 8u) >> 24u;
        let b3 = word >> 24u;
        a0 = a0 + x[kk + 0u] * f32(bitcast<i32>(b0));
        a1 = a1 + x[kk + 1u] * f32(bitcast<i32>(b1));
        a2 = a2 + x[kk + 2u] * f32(bitcast<i32>(b2));
        a3 = a3 + x[kk + 3u] * f32(bitcast<i32>(b3));
    }
    y[row] = (a0 + a1) + (a2 + a3);
}
"""

def wgsl_f32():
    return """
struct P { m: u32, k: u32, kw: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> w: array<f32>;   // [k][m]
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var a0: f32 = 0.0; var a1: f32 = 0.0; var a2: f32 = 0.0; var a3: f32 = 0.0;
    for (var t: u32 = 0u; t < p.kw; t = t + 1u) {
        let base = t * p.m + row;
        a0 = a0 + w[base] * x[t];
        a1 = a1 + w[base + 1u * p.m] * x[t + 1u];
        a2 = a2 + w[base + 2u * p.m] * x[t + 2u];
        a3 = a3 + w[base + 3u * p.m] * x[t + 3u];
        t = t + 3u;
    }
    y[row] = (a0 + a1) + (a2 + a3);
}
"""

def pack2_t(codes):
    M, K = codes.shape
    out = np.zeros((K // 16, M), dtype=np.uint32)
    for t in range(K // 16):
        w = np.zeros(M, dtype=np.uint32)
        for j in range(16):
            w |= codes[:, t * 16 + j].astype(np.uint32) << np.uint32(2 * j)
        out[t] = w
    return out

def pack_i8_t(codes):
    M, K = codes.shape
    vals = codes.astype(np.int8) - 1  # -1,0,1
    out = np.zeros((K // 4, M), dtype=np.uint32)
    for t in range(K // 4):
        w = np.zeros(M, dtype=np.uint32)
        for j in range(4):
            w |= vals[:, t * 4 + j].astype(np.uint32) << np.uint32(8 * j)
        out[t] = w
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
    rng = np.random.default_rng(4)
    M, K = 4096, 2048
    codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
    w = codes.astype(np.float32) - 1.0
    x = rng.standard_normal(K).astype(np.float32)
    ref = w @ x
    p2 = pack2_t(codes)
    pi8 = pack_i8_t(codes)
    pf = np.ascontiguousarray(w.T)  # [K][M]
    xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)

    def bench(label, code, wdata, kw, wg, repeat=200):
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
        # correctness
        enc = dev.create_command_encoder()
        cp = enc.begin_compute_pass(); cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
        cp.dispatch_workgroups((M + wg - 1) // wg, 1, 1); cp.end()
        dev.queue.submit([enc.finish()])
        got = np.frombuffer(dev.queue.read_buffer(yb, 0, M * 4), dtype=np.float32)
        err = np.abs(got - ref).max()
        # GPU-only: all dispatches in one pass
        def run(n):
            enc = dev.create_command_encoder()
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
            for _ in range(n):
                cp.dispatch_workgroups((M + wg - 1) // wg, 1, 1)
            cp.end()
            dev.queue.submit([enc.finish()])
        run(3); dev.queue.read_buffer(yb, 0, 4)
        t0 = time.perf_counter(); run(repeat); dev.queue.read_buffer(yb, 0, 4); t1 = time.perf_counter()
        dt = (t1 - t0) / repeat
        print(f"{label:14s} wg={wg:3d}  {dt*1e3:7.3f} ms  {M*K/dt/1e9:7.1f} Gop/s  {wdata.nbytes/dt/1e9:6.1f} GB/s  err={err:.1e}")

    bench("2bit wg64", wgsl_2bit(1), p2, K // 16, 64)
    bench("2bit wg128", wgsl_2bit(2), p2, K // 16, 128)
    bench("2bit wg256", wgsl_2bit(4), p2, K // 16, 256)
    bench("i8 wg64", wgsl_i8(), pi8, K // 4, 64)
    bench("f32 wg64", wgsl_f32(), pf, K, 64)

if __name__ == "__main__":
    main()
