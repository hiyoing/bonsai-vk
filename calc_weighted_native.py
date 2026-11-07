"""Weighted per-layer rate for the NATIVE GLSL runtime (runtime_vk/), same weighting as
calc_weighted.py, so the two are directly comparable.

Rates are 实测, median-of-3 rounds x 100 token-ops each, taken on the RX 550X through
runtime_vk/bonsai_vk.exe with BONSAI_RANDSCALE=1 (random fp16 per-block scales, so the
unpackHalf2x16 path is exercised, not just d == 1.0). Each timed token-op is the whole per-token
sequence - xsum dispatch + gemv dispatch + (reduce dispatch) with the pipeline barriers between
them - i.e. the number includes the overhead the WGSL prototype paid implicitly.

The splits column is the one that measured best for that shape in the sweep below; it is not a
guess, it is argmax over {1,2,4,8,16,32}:
  4096x4096   s1 119.1  s2 131.6  s4 131.4  s8 131.2  s16 123.5 s32 114.7
  1024x4096   s1  56.5  s2  91.4  s4 108.6  s8 114.7  s16 105.0 s32  87.3
  12288x4096  s1 148.6  s2 152.1  s4 154.4  s8 149.7  s16 141.4 s32 126.5
  4096x12288  s1 119.3  s2 138.5  s4 144.0  s8 152.5  s16 152.4 s32 139.3

MAC counts and the 9e9 MAC/token for a 9B dense model are the same 估算 as STAGE2 2.1.
"""
import numpy as np

# (name, M, K, native Gop/s, splits, WGSL-proto Gop/s, llama.cpp VEC Gop/s)
LAYER = [
    ("q",    4096,  4096, 131.6, 2,  124.1, 44.8),
    ("o",    4096,  4096, 131.6, 2,  124.1, 44.8),
    ("k",    1024,  4096, 114.7, 8,   83.4, 38.1),
    ("v",    1024,  4096, 114.7, 8,   83.4, 38.1),
    ("gate", 12288, 4096, 154.4, 4,  144.3, 43.7),
    ("up",   12288, 4096, 154.4, 4,  144.3, 43.7),
    ("down", 4096,  12288, 152.5, 8,  142.8, 42.5),
]
MAC = np.array([m * k for _n, m, k, _a, _s, _b, _c in LAYER], dtype=float)
TOT = MAC.sum()
print("per-layer MAC: %.2f M  (FFN share %.1f%%)" % (TOT / 1e6, 100 * MAC[4:].sum() / TOT))
for i, (n, m, k, a, s, b, c) in enumerate(LAYER):
    print("  %-5s %6dx%-6d %8.3f M  splits %2d   native %7.1f us (%6.1f G)   WGSL proto %7.1f us"
          " (%5.1f G)   llama.cpp %7.1f us (%5.1f G)"
          % (n, m, k, MAC[i] / 1e6, s, MAC[i] / a * 1e-3, a, MAC[i] / b * 1e-3, b,
             MAC[i] / c * 1e-3, c))

for idx, label in [(3, "native GLSL runtime"), (5, "WGSL prototype (probe 14)"), (6, "llama.cpp VEC rows4 port")]:
    rate = np.array([x[idx] for x in LAYER], dtype=float)
    t = (MAC / rate).sum() * 1e-9
    w = TOT / t / 1e9
    print("%-28s layer %7.1f us  weighted %5.1f Gop/s  9e9 MAC/token -> %6.1f ms  %5.2f tok/s"
          % (label, t * 1e6, w, 9.0 / w * 1e3, w / 9.0))

t_nat = (MAC / np.array([x[3] for x in LAYER])).sum() * 1e-9
t_wgsl = (MAC / np.array([x[5] for x in LAYER])).sum() * 1e-9
t_ll = (MAC / np.array([x[6] for x in LAYER])).sum() * 1e-9
print("layer-time ratio  llama.cpp/native = %.2fx   WGSL/native = %.2fx" % (t_ll / t_nat, t_wgsl / t_nat))
for name, t in [("native", t_nat), ("WGSL proto", t_wgsl), ("llama.cpp", t_ll)]:
    print("%-11s: 40 dense layers -> %6.2f ms/token of GEMV alone" % (name, t * 40 * 1e3))
