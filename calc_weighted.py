"""Weight the measured per-shape Gop/s into one per-layer number, for two kernels.

Rates are 实测 (median of 3x50 dispatches) on this RX 550X today:
  llama.cpp  = probe_gemv15.py "VEC rows4" (mul_mat_vec_ptq1_0.comp port, BLOCK_SIZE = 64 = the
               card's wave64 subgroup, NUM_ROWS = 2*rm_stdq = 4 on GCN). This is the kernel the
               fork actually selects here: device->integer_dot_product is false (vulkaninfo: all
               4x8BitPacked*Accelerated = false), so quantize_y is false and the q8_1 pipeline is
               never created.
  ours       = probe_gemv14.py, the production 2-bit bfe kernel with per-128-block scale, at the
               splits that measured best per shape.
MAC counts and the 9e9 MAC/token are the 估算 from STAGE2 2.1 (135.8 Gop/s -> 15.1 tok/s).
"""
import numpy as np

# (name, M, K, ours Gop/s, llamapp Gop/s)   -- shapes of one Bonsai-2-9B dense projection set
LAYER = [
    ("q",    4096,  4096, 124.1, 44.8),
    ("o",    4096,  4096, 124.1, 44.8),
    ("k",    1024,  4096,  83.4, 38.1),
    ("v",    1024,  4096,  83.4, 38.1),
    ("gate", 12288, 4096, 144.3, 43.7),
    ("up",   12288, 4096, 144.3, 43.7),
    ("down", 4096,  12288, 142.8, 42.5),
]
MAC = np.array([m * k for _n, m, k, _a, _b in LAYER], dtype=float)
TOT = MAC.sum()
print("per-layer MAC: %.2f M  (FFN share %.1f%%)" % (TOT / 1e6, 100 * MAC[4:].sum() / TOT))
for i, (n, m, k, a, b) in enumerate(LAYER):
    print("  %-5s %6dx%-6d %8.3f M   ours %7.1f us (%5.1f G)   llama.cpp %7.1f us (%5.1f G)"
          % (n, m, k, MAC[i] / 1e6, MAC[i] / a * 1e-3, a, MAC[i] / b * 1e-3, b))

for idx, label in [(0, "ours"), (1, "llama.cpp (measured VEC rows4 port)")]:
    rate = np.array([x[3 + idx] for x in LAYER], dtype=float)
    t = (MAC / rate).sum() * 1e-9          # seconds per layer (rate is in Gop/s)
    w = TOT / t / 1e9
    print("%-34s layer %7.1f us  weighted %5.1f Gop/s  9e9 MAC/token -> %6.1f ms  %.2f tok/s"
          % (label, t * 1e6, w, 9.0 / w * 1e3, w / 9.0))

t_ours = (MAC / np.array([x[3] for x in LAYER])).sum() * 1e-9
t_ll = (MAC / np.array([x[4] for x in LAYER])).sum() * 1e-9
print("ratio layer-time llama.cpp / ours = %.2fx" % (t_ll / t_ours))

# dense-layer count is 估算 (layer count not read from the checkpoint yet): illustrative only
for name, t in [("ours", t_ours), ("llama.cpp", t_ll)]:
    print("%-9s: 40 dense layers -> %.2f ms/token of GEMV alone" % (name, t * 40 * 1e3))
