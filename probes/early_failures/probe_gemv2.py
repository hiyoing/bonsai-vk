"""Probe 5: barrier-free GEMV with transposed 2-bit weights + row-parallel lanes.
Layout: packed weights stored as [K/16][M] u32 (t-major), so a wave reads 64 consecutive rows.
"""
import time
import numpy as np
import wgpu

WGSL = """
struct P { m: u32, k: u32, kw: u32, batch: u32 };
@group(0) @binding(0) var<storage, read> w: array<u32>;   // [kw][m]
@group(0) @binding(1) var<storage, read> x: array<f32>;   // [batch][k]
@group(0) @binding(2) var<storage, read_write> y: array<f32>; // [batch][m]
@group(0) @binding(3) var<uniform> p: P;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var acc0: f32 = 0.0;
    var acc1: f32 = 0.0;
    var acc2: f32 = 0.0;
    var acc3: f32 = 0.0;
    for (var t: u32 = 0u; t < p.kw; t = t + 1u) {
        let word = w[t * p.m + row];
        let kk = t * 16u;
        // two halves, each 8 unrolled codes, alternate accumulators for ILP
        acc0 = acc0 + x[kk + 0u] * select(select(1.0, 0.0, ((word >> 0u) & 3u) == 1u), -1.0, ((word >> 0u) & 3u) == 0u);
        acc1 = acc1 + x[kk + 1u] * select(select(1.0, 0.0, ((word >> 2u) & 3u) == 1u), -1.0, ((word >> 2u) & 3u) == 0u);
        acc2 = acc2 + x[kk + 2u] * select(select(1.0, 0.0, ((word >> 4u) & 3u) == 1u), -1.0, ((word >> 4u) & 3u) == 0u);
        acc3 = acc3 + x[kk + 3u] * select(select(1.0, 0.0, ((word >> 6u) & 3u) == 1u), -1.0, ((word >> 6u) & 3u) == 0u);
        acc0 = acc0 + x[kk + 4u] * select(select(1.0, 0.0, ((word >> 8u) & 3u) == 1u), -1.0, ((word >> 8u) & 3u) == 0u);
        acc1 = acc1 + x[kk + 5u] * select(select(1.0, 0.0, ((word >> 10u) & 3u) == 1u), -1.0, ((word >> 10u) & 3u) == 0u);
        acc2 = acc2 + x[kk + 6u] * select(select(1.0, 0.0, ((word >> 12u) & 3u) == 1u), -1.0, ((word >> 12u) & 3u) == 0u);
        acc3 = acc3 + x[kk + 7u] * select(select(1.0, 0.0, ((word >> 14u) & 3u) == 1u), -1.0, ((word >> 14u) & 3u) == 0u);
        acc0 = acc0 + x[kk + 8u] * select(select(1.0, 0.0, ((word >> 16u) & 3u) == 1u), -1.0, ((word >> 16u) & 3u) == 0u);
        acc1 = acc1 + x[kk + 9u] * select(select(1.0, 0.0, ((word >> 18u) & 3u) == 1u), -1.0, ((word >> 18u) & 3u) == 0u);
        acc2 = acc2 + x[kk + 10u] * select(select(1.0, 0.0, ((word >> 20u) & 3u) == 1u), -1.0, ((word >> 20u) & 3u) == 0u);
        acc3 = acc3 + x[kk + 11u] * select(select(1.0, 0.0, ((word >> 22u) & 3u) == 1u), -1.0, ((word >> 22u) & 3u) == 0u);
        acc0 = acc0 + x[kk + 12u] * select(select(1.0, 0.0, ((word >> 24u) & 3u) == 1u), -1.0, ((word >> 24u) & 3u) == 0u);
        acc1 = acc1 + x[kk + 13u] * select(select(1.0, 0.0, ((word >> 26u) & 3u) == 1u), -1.0, ((word >> 26u) & 3u) == 0u);
        acc2 = acc2 + x[kk + 14u] * select(select(1.0, 0.0, ((word >> 28u) & 3u) == 1u), -1.0, ((word >> 28u) & 3u) == 0u);
        acc3 = acc3 + x[kk + 15u] * select(select(1.0, 0.0, ((word >> 30u) & 3u) == 1u), -1.0, ((word >> 30u) & 3u) == 0u);
    }
    y[row] = (acc0 + acc1) + (acc2 + acc3);
}
"""


def pack_t(codes):
    M, K = codes.shape
    out = np.zeros((K // 16, M), dtype=np.uint32)
    for t in range(K // 16):
        w = np.zeros(M, dtype=np.uint32)
        for j in range(16):
            w |= codes[:, t * 16 + j].astype(np.uint32) << np.uint32(2 * j)
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
    shader = dev.create_shader_module(code=WGSL)
    pipe = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[bgl]),
                                       compute={"module": shader, "entry_point": "main"})
    rng = np.random.default_rng(3)

    for (M, K) in [(2048, 2048), (4096, 2048), (2048, 4096), (1024, 2048)]:
        codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
        w = (codes.astype(np.float32) - 1.0)
        x = rng.standard_normal(K).astype(np.float32)
        ref = w @ x
        wt = pack_t(codes)
        wb = dev.create_buffer_with_data(data=wt.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        yb = dev.create_buffer(size=M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(data=np.array([M, K, K // 16, 1], dtype=np.uint32).tobytes(),
                                          usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": wt.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": yb, "offset": 0, "size": M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 16}},
        ])
        def run(n):
            enc = dev.create_command_encoder()
            for _ in range(n):
                cp = enc.begin_compute_pass()
                cp.set_pipeline(pipe); cp.set_bind_group(0, bg)
                cp.dispatch_workgroups((M + 63) // 64, 1, 1)
                cp.end()
            dev.queue.submit([enc.finish()])
        run(1)
        got = np.frombuffer(dev.queue.read_buffer(yb, 0, M * 4), dtype=np.float32)
        err = np.abs(got - ref).max()
        n = 100
        run(3); dev.queue.read_buffer(yb, 0, 4)
        t0 = time.perf_counter(); run(n); dev.queue.read_buffer(yb, 0, 4); t1 = time.perf_counter()
        dt = (t1 - t0) / n
        print(f"M={M:5d} K={K:5d}  {dt*1e3:7.3f} ms  {M*K/dt/1e9:7.1f} Gop/s  {wt.nbytes/dt/1e9:6.1f} GB/s  err={err:.1e}")

if __name__ == "__main__":
    main()
