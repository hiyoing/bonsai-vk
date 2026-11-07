"""Probe 11: realistic MoE shapes (M=768/2048, K=2048/768) with expert fusion."""
import time
import numpy as np
import wgpu
from probe_gemv7 import make_shader, pack2_vec4, REDUCE


def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()
    e = [{"binding": 0, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
         {"binding": 1, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.read_only_storage}},
         {"binding": 2, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.storage}},
         {"binding": 3, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": wgpu.BufferBindingType.uniform}}]
    bgl = dev.create_bind_group_layout(entries=e)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    rng = np.random.default_rng(7)

    def bench(label, M, K, splits, uv, kind="bfefold", nacc=8, wg=64, rounds=3, reps=150):
        nvec = K // 64
        per = nvec // splits
        assert nvec % splits == 0 and per % uv == 0, (nvec, splits, uv)
        codes = rng.integers(0, 3, size=(M, K), dtype=np.uint8)
        x = rng.standard_normal(K).astype(np.float32)
        ref = (codes.astype(np.float32) - 1.0) @ x
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
            "module": dev.create_shader_module(code=make_shader(kind, uv, splits, nacc, wg)),
            "entry_point": "main"})
        rp = None
        if splits > 1:
            rp = dev.create_compute_pipeline(layout=pl, compute={
                "module": dev.create_shader_module(code=REDUCE), "entry_point": "main"})

        def passes(c):
            c.set_pipeline(p); c.set_bind_group(0, bg)
            c.dispatch_workgroups((M + wg - 1) // wg, splits, 1)
            if rp:
                c.set_pipeline(rp); c.set_bind_group(0, bg)
                c.dispatch_workgroups((M + 63) // 64, 1, 1)

        enc = dev.create_command_encoder(); c = enc.begin_compute_pass(); passes(c); c.end()
        dev.queue.submit([enc.finish()])
        off = nbuf if splits > 1 else 0
        got = np.frombuffer(dev.queue.read_buffer(ob, off, M * 4), dtype=np.float32)
        err = float(np.abs(got - ref).max())

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
            dev.queue.read_buffer(ob, off, 4)
            ts.append((time.perf_counter() - t0) / reps)
        dt = float(np.median(ts))
        print(f"{label:34s} M={M:5d} K={K:4d} sp={splits} uv={uv} "
              f"{dt*1e3:7.3f} ms  {M*K/dt/1e9:7.1f} Gop/s  err={err:.1e}")
        return dt

    print("--- layer cost model: 8 experts fused, Qwen3-30B-A3B (hidden 2048, moe_inter 768) ---")
    t_up = bench("gate+up  8 experts fused", 8 * 1536, 2048, 4, 4)
    t_dn = bench("down     8 experts fused", 8 * 2048, 768, 2, 3)
    print(f"\nper-layer expert time = {t_up*1e3:.2f} + {t_dn*1e3:.2f} = {(t_up+t_dn)*1e3:.2f} ms")
    print(f"48 layers            = {(t_up+t_dn)*48*1e3:.1f} ms  ->  {1/((t_up+t_dn)*48):.2f} tok/s (compute only)")

    print("\n--- same work, one expert per dispatch (what a naive runtime does) ---")
    a = bench("gate+up  1 expert", 1536, 2048, 4, 4)
    b = bench("down     1 expert", 2048, 768, 2, 3)
    print(f"per layer x8 experts = {(a+b)*8*1e3:.2f} ms  ->  {1/((a+b)*8*48):.2f} tok/s")


if __name__ == "__main__":
    main()
