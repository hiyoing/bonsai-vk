"""Probe 12: per-layer expert-compute cost vs group size (overlap granularity)."""
import time
import numpy as np
import wgpu
from probe_gemv7 import make_shader, pack2_vec4, REDUCE


def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()
    ent = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
           {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
           {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
           {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}}]
    bgl = dev.create_bind_group_layout(entries=ent)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    rng = np.random.default_rng(11)

    def bench(M, K, splits, uv, rounds=4, reps=200):
        nvec = K // 64
        per = nvec // splits
        codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
        x = rng.standard_normal(K).astype(np.float32)
        p2 = pack2_vec4(codes)
        corr = float(-x.sum())
        xb = dev.create_buffer_with_data(data=x.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        wb = dev.create_buffer_with_data(data=p2.tobytes(), usage=wgpu.BufferUsage.STORAGE)
        nbuf = M * splits * 4
        ob = dev.create_buffer(size=nbuf + M * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        par = dev.create_buffer_with_data(
            data=np.array([M, K, K // 16, splits], dtype=np.uint32).tobytes()
            + np.float32(corr).tobytes() + b"\0" * 12, usage=wgpu.BufferUsage.UNIFORM)
        bg = dev.create_bind_group(layout=bgl, entries=[
            {"binding": 0, "resource": {"buffer": wb, "offset": 0, "size": p2.nbytes}},
            {"binding": 1, "resource": {"buffer": xb, "offset": 0, "size": x.nbytes}},
            {"binding": 2, "resource": {"buffer": ob, "offset": 0, "size": nbuf + M * 4}},
            {"binding": 3, "resource": {"buffer": par, "offset": 0, "size": 32}}])
        p = dev.create_compute_pipeline(layout=pl, compute={
            "module": dev.create_shader_module(code=make_shader("bfefold", uv, splits)),
            "entry_point": "main"})
        rp = dev.create_compute_pipeline(layout=pl, compute={
            "module": dev.create_shader_module(code=REDUCE), "entry_point": "main"})

        def passes(c):
            c.set_pipeline(p); c.set_bind_group(0, bg)
            c.dispatch_workgroups((M + 63) // 64, splits, 1)
            c.set_pipeline(rp); c.set_bind_group(0, bg)
            c.dispatch_workgroups((M + 63) // 64, 1, 1)

        ts = []
        for _ in range(rounds):
            enc = dev.create_command_encoder(); c = enc.begin_compute_pass()
            for _ in range(5):
                passes(c)
            c.end(); dev.queue.submit([enc.finish()])
            dev.queue.read_buffer(ob, 0, 4)
            t0 = time.perf_counter()
            enc = dev.create_command_encoder(); c = enc.begin_compute_pass()
            for _ in range(reps):
                passes(c)
            c.end(); dev.queue.submit([enc.finish()])
            dev.queue.read_buffer(ob, nbuf, 4)
            ts.append((time.perf_counter() - t0) / reps)
        return float(np.median(ts))

    print(f"{'n/group':>7s} {'gate+up ms':>11s} {'down ms':>8s} {'layer ms':>9s} "
          f"{'Gop/s':>7s} {'groups':>7s}")
    print("n=8 (fully fused) is one group; n=1 means 8 separate dispatches\n")
    for n in (1, 2, 4, 8):
        up = bench(n * 1536, 2048, 4, 4)
        dn = bench(n * 2048, 768, 2, 3)
        layer = (up + dn) * (8 // n)
        gops = 8 * (1536 * 2048 + 2048 * 768) / ((up + dn) * 1e-3) / 1e9 * (1 if n else 1)
        gops = (8 // n) * n * (1536 * 2048 + 2048 * 768) / (layer * 1e-3) / 1e9
        print(f"{n:7d} {up*1e3:11.3f} {dn*1e3:8.3f} {layer*1e3:9.3f} {gops:7.1f} {8//n:7d}")


if __name__ == "__main__":
    main()
