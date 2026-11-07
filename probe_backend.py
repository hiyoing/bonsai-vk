"""Stage-0 backend probe: enumerate wgpu adapters, report Vulkan device limits,
then microbenchmark a 2-bit ternary packed matvec kernel (the core primitive).
"""
import os, sys, time, struct
import numpy as np
import wgpu

def dump_adapters():
    print("WGPU_BACKEND env =", os.environ.get("WGPU_BACKEND"))
    for a in wgpu.gpu.enumerate_adapters_sync():
        i = dict(a.info)
        print(f"- {i.get('device')!r} | vendor={i.get('vendor')!r} | backend={i.get('backend_type')!r} | "
              f"type={i.get('adapter_type')!r} | desc={i.get('description')!r} | subgroups={i.get('subgroup_min_size')}-{i.get('subgroup_max_size')}")
    print()

def main():
    dump_adapters()
    adapter = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    info = dict(adapter.info)
    print("selected:", info.get("device"), "|", info.get("backend_type"), "|", info.get("description"))
    lim = adapter.limits
    for k in ["max-storage-buffer-binding-size", "max-buffer-size", "max-compute-workgroup-size-x",
              "max-compute-invocations-per-workgroup", "max-storage-buffers-per-shader-stage",
              "max-compute-workgroup-storage-size", "max-bind-groups"]:
        try:
            print(f"  {k} = {lim[k]}")
        except Exception as e:
            print("  limit missing", k, e)
    feats = set(adapter.features)
    for f in ["timestamp-query", "timestamp-query-inside-passes", "shader-f16", "subgroups",
              "pipeline-statistics-query"]:
        print(f"  feature {f}: {f in feats}")
    device = adapter.request_device_sync()

    # ---- 2-bit ternary packed matvec: y[M] = sum_k x[k] * w[m,k], w in {-1,0,+1}
    M, K = 4096, 4096
    rng = np.random.default_rng(0)
    codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8).astype(np.uint8)  # 0,1,2 -> -1,0,+1
    packed = np.zeros((M, K // 16), dtype=np.uint32)
    for t in range(K // 16):
        word = np.zeros(M, dtype=np.uint32)
        for j in range(16):
            word |= codes[:, t * 16 + j].astype(np.uint32) << np.uint32(2 * j)
        packed[:, t] = word
    x = rng.standard_normal(K).astype(np.float32)
    w = codes.astype(np.float32) - 1.0
    ref = w @ x

    WGSL = """
struct Meta { m: u32, k: u32, pad0: u32, pad1: u32 };
@group(0) @binding(0) var<storage, read> packed: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read_write> y: array<f32>;
@group(0) @binding(3) var<uniform> params: Meta;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.m) { return; }
    let words = params.k / 16u;
    let base = row * words;
    var acc: f32 = 0.0;
    for (var t: u32 = 0u; t < words; t = t + 1u) {
        let w = packed[base + t];
        let kk = t * 16u;
        for (var j: u32 = 0u; j < 16u; j = j + 1u) {
            let c = (w >> (2u * j)) & 3u;
            let s = select(select(1.0, 0.0, c == 1u), -1.0, c == 0u);
            acc = acc + x[kk + j] * s;
        }
    }
    y[row] = acc;
}
"""

    shader = device.create_shader_module(code=WGSL)
    packed_buf = device.create_buffer_with_data(data=packed.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    x_buf = device.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    y_buf = device.create_buffer(size=M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    meta = np.array([M, K, 0, 0], dtype=np.uint32)
    meta_buf = device.create_buffer_with_data(data=meta.tobytes(), usage=wgpu.BufferUsage.UNIFORM)

    bgl = device.create_bind_group_layout(entries=[
        {"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
        {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
        {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}},
    ])
    bg = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": packed_buf, "offset": 0, "size": packed.nbytes}},
        {"binding": 1, "resource": {"buffer": x_buf, "offset": 0, "size": x.nbytes}},
        {"binding": 2, "resource": {"buffer": y_buf, "offset": 0, "size": M * 4}},
        {"binding": 3, "resource": {"buffer": meta_buf, "offset": 0, "size": 16}},
    ])
    layout = device.create_pipeline_layout(bind_group_layouts=[bgl])
    pipe = device.create_compute_pipeline(layout=layout, compute={"module": shader, "entry_point": "main"})

    def run(n):
        enc = device.create_command_encoder()
        for _ in range(n):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe)
            cp.set_bind_group(0, bg)
            cp.dispatch_workgroups((M + 63) // 64, 1, 1)
            cp.end()
        device.queue.submit([enc.finish()])

    # correctness
    run(1)
    out = device.queue.read_buffer(y_buf, 0, M * 4)
    got = np.frombuffer(out, dtype=np.float32)
    err = np.abs(got - ref).max()
    print(f"correctness max_abs_err = {err:.4e}  (ref[0]={ref[0]:.4f} got[0]={got[0]:.4f})")

    # timing
    run(3); device.queue.read_buffer(y_buf, 0, 4)
    n = 30
    t0 = time.perf_counter(); run(n); device.queue.read_buffer(y_buf, 0, 4); t1 = time.perf_counter()
    dt = (t1 - t0) / n
    ops = M * K
    print(f"matvec: M={M} K={K}  {dt*1e3:.2f} ms/dispatch  -> {ops/dt/1e9:.1f} G weight-ops/s")
    # bigger to amortize overhead
    M2 = 16384
    packed2 = np.tile(packed, (4, 1))
    y2 = device.create_buffer(size=M2 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    p2 = device.create_buffer_with_data(data=packed2.tobytes(), usage=wgpu.BufferUsage.STORAGE)
    bg2 = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": p2, "offset": 0, "size": packed2.nbytes}},
        {"binding": 1, "resource": {"buffer": x_buf, "offset": 0, "size": x.nbytes}},
        {"binding": 2, "resource": {"buffer": y2, "offset": 0, "size": M2 * 4}},
        {"binding": 3, "resource": {"buffer": meta_buf, "offset": 0, "size": 16}},
    ])
    meta2 = device.create_buffer_with_data(data=np.array([M2, K, 0, 0], dtype=np.uint32).tobytes(), usage=wgpu.BufferUsage.UNIFORM)
    bg3 = device.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": p2, "offset": 0, "size": packed2.nbytes}},
        {"binding": 1, "resource": {"buffer": x_buf, "offset": 0, "size": x.nbytes}},
        {"binding": 2, "resource": {"buffer": y2, "offset": 0, "size": M2 * 4}},
        {"binding": 3, "resource": {"buffer": meta2, "offset": 0, "size": 16}},
    ])
    def run2(nn):
        enc = device.create_command_encoder()
        for _ in range(nn):
            cp = enc.begin_compute_pass()
            cp.set_pipeline(pipe); cp.set_bind_group(0, bg3)
            cp.dispatch_workgroups((M2 + 63) // 64, 1, 1)
            cp.end()
        device.queue.submit([enc.finish()])
    run2(1); device.queue.read_buffer(y2, 0, 4)
    t0 = time.perf_counter(); run2(10); device.queue.read_buffer(y2, 0, 4); t1 = time.perf_counter()
    dt = (t1 - t0) / 10
    print(f"matvec: M={M2} K={K}  {dt*1e3:.2f} ms/dispatch  -> {M2*K/dt/1e9:.1f} G weight-ops/s")

if __name__ == "__main__":
    main()
