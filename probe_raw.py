"""Probe 3: isolate raw bandwidth / compute throughput of the Vulkan path."""
import time
import numpy as np
import wgpu

CPY = """
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    b[i] = a[i] * 2.0 + 1.0;
}
"""
FMA = """
struct P { n: u32, iters: u32 };
@group(0) @binding(0) var<storage, read> a: array<f32>;
@group(0) @binding(1) var<storage, read_write> b: array<f32>;
@group(0) @binding(2) var<uniform> p: P;
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let i = g.x;
    if (i >= p.n) { return; }
    var x = a[i];
    var y = x + 1.0;
    var z = x + 2.0;
    var w = x + 3.0;
    for (var k: u32 = 0u; k < p.iters; k = k + 1u) {
        x = x * 1.000001 + 0.5;
        y = y * 1.000001 + 0.5;
        z = z * 1.000001 + 0.5;
        w = w * 1.000001 + 0.5;
    }
    b[i] = x + y + z + w;
}
"""

def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()

    def mk(code):
        m = dev.create_shader_module(code=code)
        return m

    N = 32 * 1024 * 1024 // 4  # 32M floats = 128MB
    a = np.random.rand(N).astype(np.float32)
    ab = dev.create_buffer_with_data(data=a.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    bb = dev.create_buffer(size=N * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)

    bgl1 = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
    ])
    pipe_cpy = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[bgl1]),
                                           compute={"module": mk(CPY), "entry_point": "main"})
    bg1 = dev.create_bind_group(layout=bgl1, entries=[
        {"binding": 0, "resource": {"buffer": ab, "offset": 0, "size": N * 4}},
        {"binding": 1, "resource": {"buffer": bb, "offset": 0, "size": N * 4}},
    ])
    def run_cpy(n):
        enc = dev.create_command_encoder()
        for _ in range(n):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe_cpy); cp.set_bind_group(0, bg1)
            cp.dispatch_workgroups((N + 255) // 256, 1, 1)
            cp.end()
        dev.queue.submit([enc.finish()])
    run_cpy(1); dev.queue.read_buffer(bb, 0, 4)
    t0 = time.perf_counter(); run_cpy(5); dev.queue.read_buffer(bb, 0, 4); t1 = time.perf_counter()
    dt = (t1 - t0) / 5
    print(f"copy+FMA 256MB traffic: {dt*1e3:.2f} ms -> {N*4*2/dt/1e9:.1f} GB/s")

    p = dev.create_buffer_with_data(data=np.array([N // 8, 4000], dtype=np.uint32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    bgl2 = dev.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    pipe_fma = dev.create_compute_pipeline(layout=dev.create_pipeline_layout(bind_group_layouts=[bgl2]),
                                           compute={"module": mk(FMA), "entry_point": "main"})
    bg2 = dev.create_bind_group(layout=bgl2, entries=[
        {"binding": 0, "resource": {"buffer": ab, "offset": 0, "size": N * 4}},
        {"binding": 1, "resource": {"buffer": bb, "offset": 0, "size": N * 4}},
        {"binding": 2, "resource": {"buffer": p, "offset": 0, "size": 8}},
    ])
    def run_fma(n):
        enc = dev.create_command_encoder()
        for _ in range(n):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe_fma); cp.set_bind_group(0, bg2)
            cp.dispatch_workgroups(((N // 8) + 255) // 256, 1, 1)
            cp.end()
        dev.queue.submit([enc.finish()])
    run_fma(1); dev.queue.read_buffer(bb, 0, 4)
    t0 = time.perf_counter(); run_fma(3); dev.queue.read_buffer(bb, 0, 4); t1 = time.perf_counter()
    dt = (t1 - t0) / 3
    flops = (N // 8) * 4000 * 4 * 2  # 4 fma chains, 2 flops each
    print(f"fma: {dt*1e3:.2f} ms -> {flops/dt/1e9:.1f} GFLOP/s")

if __name__ == "__main__":
    main()
