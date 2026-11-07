import numpy as np, wgpu, importlib
p = importlib.import_module("probe_gemv15")
M, K = 4, 128
nb = 1
rng = np.random.default_rng(3)
words, tr, d = p.make_blocks(M, K, nb, rng)
a = rng.integers(0, 256, size=K, dtype=np.uint8)
x = a.astype(np.float32) / 255.0
w1 = (tr.astype(np.float32) - 1.0).reshape(M, nb, 128)
ref = np.zeros(M, np.float64)
for b in range(nb):
    ref += d[b].astype(np.float64) * (w1[:, b, :] @ x.astype(np.float64))
print("ref", ref)
# hand-simulate the VEC kernel on the cpu
def elem_of(k, t, bb): return (4*k + 16*t if k < 4 else 80 + 4*(k-4) + 8*t) + bb
for row in range(M):
    tot = 0.0
    for h in range(2):
        acc = 0.0; ys = 0.0
        for kk in range(3):
            k = 3*h + kk
            w = words[(row*nb + 0)*7 + k]
            ev = w & 0x00FF00FF; od = (w >> 8) & 0x00FF00FF
            for t in range(5):
                me, mo = ev*3, od*3
                te, to = (me >> 8) & 0x00030003, (mo >> 8) & 0x00030003
                ev, od = me & 0x00FF00FF, mo & 0x00FF00FF
                q = [te & 3, to & 3, te >> 16, to >> 16]
                wi = k + 4*t if k < 4 else 20 + (k-4) + 2*t
                yv = x[4*wi:4*wi+4]
                ys += yv.sum(); acc += float(np.dot(q, yv))
        if h == 0:
            w6 = words[(row*nb+0)*7 + 6]
            xq = (w6 & 0xFF) | ((w6 >> 8) & 0xFF) << 16
            qq = []
            for t in range(4):
                m = xq*3; tq = (m >> 8) & 0x00030003; xq = m & 0x00FF00FF
                qq += [tq & 3, tq >> 16]
            yq = x[120:128]
            ys += yq.sum(); acc += float(np.dot(qq, yq))
        print("  row", row, "h", h, "acc", acc, "ys", ys)
        tot += d[0][row] * (acc - ys)
    print("sim", row, tot)
