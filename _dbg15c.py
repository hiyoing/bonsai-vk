"""Bisect probe 15: VEC/VECQ were right at M=8,K=128 (relerr 3.8e-07) but ~1.0 at full shapes.
Run both ports at increasing K with a small M to find which factor breaks them."""
import numpy as np, wgpu, importlib
p = importlib.import_module("probe_gemv15")

ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
dev = ad.request_device_sync()
ents = []
for i in range(9):
    t = (wgpu.BufferBindingType.storage if i == 4 else
         wgpu.BufferBindingType.uniform if i == 5 else wgpu.BufferBindingType.read_only_storage)
    ents.append({"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}})
bgl = dev.create_bind_group_layout(entries=ents)
pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])


def buf(arr, usage=wgpu.BufferUsage.STORAGE):
    return dev.create_buffer_with_data(data=arr.tobytes(), usage=usage)


def run(src, M, K, R, grid):
    nb = K // 128
    rng = np.random.default_rng(3)
    words, tr, d = p.make_blocks(M, K, nb, rng)
    s_val = np.float32(1.0 / 255.0)
    a = rng.integers(0, 256, size=K, dtype=np.uint8)
    x = (a.astype(np.float32) * s_val).astype(np.float32)
    w1 = (tr.astype(np.float32) - 1.0).reshape(M, nb, 128)
    xs = x.reshape(nb, 128)
    ref = np.zeros(M, np.float64)
    for b in range(nb):
        ref += d[b].astype(np.float64) * (w1[:, b, :] @ xs[b].astype(np.float64))
    bs = np.zeros((nb, 4), np.float32)
    for b in range(nb):
        for lane in range(2):          # h (VEC) == sub (VECQ), same element split
            idx = [p.elem_of(3 * lane + kk, t, bb) for kk in range(3) for t in range(5) for bb in range(4)]
            if lane == (0 if src is p.vec_src else 1):
                idx += list(range(120, 128))
            bs[b, lane if src is p.vec_src else 2 + lane] = xs[b][idx].sum()
    ypad = np.zeros((K // 4 + 3) // 4 * 4, np.uint32)
    ypad[:K // 4] = a.view(np.uint32)
    bo = dev.create_buffer(size=M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
    bg = dev.create_bind_group(layout=bgl, entries=[
        {"binding": 0, "resource": {"buffer": buf(words), "offset": 0, "size": words.nbytes}},
        {"binding": 1, "resource": {"buffer": buf(x), "offset": 0, "size": x.nbytes}},
        {"binding": 2, "resource": {"buffer": buf(d.reshape(-1)), "offset": 0, "size": d.nbytes}},
        {"binding": 3, "resource": {"buffer": buf(bs.reshape(-1)), "offset": 0, "size": bs.nbytes}},
        {"binding": 4, "resource": {"buffer": bo, "offset": 0, "size": M * 4}},
        {"binding": 5, "resource": {"buffer": buf(np.array([M, K, nb, 1], np.uint32),
                                                   wgpu.BufferUsage.UNIFORM), "offset": 0, "size": 16}},
        {"binding": 6, "resource": {"buffer": buf(ypad), "offset": 0, "size": ypad.nbytes}},
        {"binding": 7, "resource": {"buffer": buf(ypad), "offset": 0, "size": ypad.nbytes}},
        {"binding": 8, "resource": {"buffer": buf(np.full(nb, s_val, np.float32)), "offset": 0, "size": nb * 4}}])
    pipe = dev.create_compute_pipeline(layout=pl, compute={"module": dev.create_shader_module(code=src(R)),
                                                           "entry_point": "main"})
    e = dev.create_command_encoder()
    c = e.begin_compute_pass(); c.set_pipeline(pipe); c.set_bind_group(0, bg)
    c.dispatch_workgroups(*grid); c.end()
    dev.queue.submit([e.finish()])
    got = np.frombuffer(dev.queue.read_buffer(bo, 0, M * 4), np.float32)
    rel = float(np.abs(got - ref).max() / np.abs(ref).max())
    print("  M=%-4d K=%-6d nb=%-3d R=%d grid=%-12s relerr %.2e   first3 %s / ref %s"
          % (M, K, nb, R, grid, rel, np.round(got[:3], 3), np.round(ref[:3], 3)))


for name, src in [("VEC", p.vec_src), ("VECQ", p.vecq_src)]:
    print("===", name)
    for K in (128, 256, 512, 1024, 4096):
        run(src, 8, K, 1, (8, 1, 1))
    run(src, 8, 4096, 4, (2, 1, 1))
    run(src, 4096, 4096, 1, (4096, 1, 1))
