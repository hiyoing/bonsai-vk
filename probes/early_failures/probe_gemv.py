"""Probe 2: 2-bit ternary GEMV with split-K cooperative reduction (Vulkan via wgpu).
Measures the core primitive of the runtime under different shapes.
"""
import os, time
import numpy as np
import wgpu

WGSL = """
struct Params { m: u32, k: u32, kwords: u32, pad: u32 };
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> p: Params;

var<workgroup> red: array<f32, 64>;

@compute @workgroup_size(64)
fn main(@builtin(local_invocation_id) lid: vec3<u32>,
        @builtin(workgroup_id) wid: vec3<u32>) {
    let row = wid.x;
    if (row >= p.m) { return; }
    let words = p.kwords;
    let base = row * words;
    var acc: f32 = 0.0;
    // each of 64 lanes handles a contiguous chunk of words
    var t: u32 = lid.x;
    loop {
        if (t >= words) { break; }
        let w = packed[base + t];
        let kk = t * 16u;
        // unroll 16 codes
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let c = (w >> (2u * j)) & 3u;
            let s = select(select(1.0, 0.0, c == 1u), -1.0, c == 0u);
            acc = acc + x[kk + j] * s;
        }
        t = t + 64u;
    }
    red[lid.x] = acc;
    workgroupBarrier();
    var stride: u32 = 32u;
    loop {
        if (stride == 0u) { break; }
        if (lid.x < stride) { red[lid.x] = red[lid.x] + red[lid.x + stride]; }
        workgroupBarrier();
        stride = stride / 2u;
    }
    if (lid.x == 0u) { y[row] = red[0]; }
}
"""

def pack_ternary(codes):
    M, K = codes.shape
    assert K % 16 == 0
    packed = np.zeros((M, K // 16), dtype=np.uint32)
    shifts = (np.arange(16, dtype=np.uint32) * np.uint32(2))
    for j in range(16):
        packed |= codes[:, j::16][:, :].astype(np.uint32) << np.uint32(2 * j) if False else np.uint32(0)
    # simple loop (probe only)
    for t in range(K // 16):
        word = np.zeros(M, dtype=np.uint32)
        for j in range(16):
            word |= codes[:, t * 16 + j].astype(np.uint32) << np.uint32(2 * j)
        packed[:, t] = word
    return packed

def main():
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = adapter.request_device_sync()
    bgl = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    shader = dev.create_shader_module(code=WGSL)
    pipe = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[bgl]),
                                       compute={"module": shader, "entry_point": "main"})
    rng = np.random.default_rng(1)

    for (M, K) in [(2048, 2048), (2048, 4096), (1024, 2048), (4096, 4096), (2048, 2048)]:
        codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
        packed = pack_ternary(codes)
        x = rng.standard_normal(K).astype(np.float32)
        w = codes.astype(np.float32) - 1.0
        ref = w @ x
        pb = dev.create_buffer_with_data(data=packed.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        yb = dev.create_buffer(size=M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(data=np.array([M, K, K // 16, 0], dtype=np.uint32).tobytes(),
                                          usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": pb, "offset": 0, "size": packed.nbytes}},
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
        n = 20
        run(2); dev.queue.read_buffer(yb, 0, 4)
        t0 = time.perf_counter(); run(n); dev.queue.read_buffer(yb, 0, 4); t1 = time.perf_counter()
        dt = (t1 - t0) / n
        gops = M * K / dt / 1e9
        gb = packed.nbytes / dt / 1e9
        print(f"M={M:5d} K={K:5d} err={err:.2e}  {dt*1e3:7.2f} ms  {gops:7.1f} Gop/s  {gb:6.1f} GB/s")

if __name__ == "__main__":
    main()
