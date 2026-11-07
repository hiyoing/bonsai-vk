#!/usr/bin/env python3
"""qkv_arith_probe.py - 任务三's decisive experiment: WHICH ARITHMETIC does the official engine use
for a PTQ1_0 GEMV, and is our 0.41 / 1.3e-3 gap an accumulation-ORDER artifact or a precision artifact?

Why this exists
---------------
runs/ladder_v6_s1.log vs _s4.log already proved that reordering OUR split-K accumulation
(BONSAI_SPLIT=1 -> 4) moves every block sum by <= 1.2e-3 absolute, i.e. <= 2.0e-07 of the block's own
Sum|x|, while the gap to the oracle is <= 1.9e-04 of Sum|x| - four orders of magnitude more. So order
is out. What is left is the FORM of the arithmetic inside the kernel.

The primary source is the fork's own shader, saved locally as _mul_mat_vec_ptq1_0.comp. Its line 24-25
states the contract outright:

    Weight = (trit - 1) * d, so sum_e w_e*y_e = d * (sum_e trit_e*y_e - sum_e y_e)

and everything it accumulates - ysum, acc, temp - is declared FLOAT_TYPE, a macro the backend injects
in its shader prelude. GL_EXT_shader_16bit_storage is required by that prelude file
(_mul_mat_vec_base.glsl:2), which is how the fp16-accumulator hypothesis got its legs. FLOAT_TYPE is
NOT visible in the shipped DLL (the shaders are precompiled SPIR-V; a byte-scan for "FLOAT_TYPE" and
"float16_t" over prism-vk/prism/ggml-vulkan.dll returns 0 hits), so it cannot be read off -
it has to be MEASURED.

Method
------
Rebuild block 0's input vector (HR = H(signs * (attn_norm * RMS_NORM(H(signs * H(emb[365])))))) on the
CPU straight from the GGUF, validate it against the oracle's own printed anchors (head3 + tail3 + sum,
which is also what our GPU reproduces), then evaluate SIX candidate arithmetics for the six channels
the oracle prints (0,1,2,8189,8190,8191) and ask which ones land inside the oracle's printed interval.

The oracle prints 4 decimals, so each want value v is the interval [v - 5e-5, v + 5e-5]. The candidates
differ from each other by 1e-5 (all fp32) to 1e-2 (fp16 accumulator), so a 1e-4-wide window separates
them cleanly. Nothing is fitted: a candidate either falls in all six windows or it does not.

CPU only. Reads the GGUF (read-only, never written) and runs nothing on the card.
"""
import math
import os
import re
import struct
import sys

GGUF = os.environ.get("GGUF", r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf")
TOK = 365                      # the token the oracle's 1-token pass ran on (runs/fwd_align_v10.log:88)
ROWS = [0, 1, 2, 8189, 8190, 8191]
# node_21 = MUL_MAT(blk.0.attn_qkv.weight, (reshaped)) printed in runs/dbg_full_real.log pass 2:
WANT = {0: 0.5366, 1: -0.8338, 2: 0.3179, 8189: 1.3872, 8190: 1.1099, 8191: -4.4314}
# our GPU's values, transcribed from runs/fwd_align_v10.log line "op  9  reduce -> QKV"
OURS_GPU = {0: 0.53794, 1: -0.82876, 2: 0.31063}
HALF_WIN = 5e-5                # the oracle's own print resolution (4 decimals)
CACHE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "qkv_probe_hr.bin")
CACHE64 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "qkv_probe_hr64.bin")

PTQ_BLOCK_BYTES, PTQ_NW = 28, 128
GG_T = {0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4), 5: ("<i", 4),
        6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8), 12: ("<d", 8)}


# ------------------------------------------------------------------ minimal GGUF header reader
class GReader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        assert len(b) == n, (n, len(b))
        return b

    def one(self, t):
        if t == 8:
            return self.s()
        fmt, sz = GG_T[t]
        return struct.unpack(fmt, self.raw(sz))[0]

    def s(self):
        n = struct.unpack("<Q", self.raw(8))[0]
        return self.raw(n).decode("utf-8", "replace")

    def arr(self):
        et = struct.unpack("<I", self.raw(4))[0]
        n = struct.unpack("<Q", self.raw(8))[0]
        if et == 8:
            return et, [self.s() for _ in range(n)]
        fmt, sz = GG_T[et]
        buf = self.raw(n * sz)
        return et, [struct.unpack_from(fmt, buf, i * sz)[0] for i in range(n)]


def read_header(path):
    f = open(path, "rb")
    r = GReader(f)
    assert r.raw(4) == b"GGUF"
    ver = struct.unpack("<I", r.raw(4))[0]
    n_tensors = struct.unpack("<Q", r.raw(8))[0]
    n_kv = struct.unpack("<Q", r.raw(8))[0]
    md = {}
    for _ in range(n_kv):
        k = r.s()
        t = struct.unpack("<I", r.raw(4))[0]
        if t == 9:
            md[k] = r.arr()
        else:
            md[k] = (t, [r.one(t)])
    infos = []
    for _ in range(n_tensors):
        name = r.s()
        nd = struct.unpack("<I", r.raw(4))[0]
        dims = [struct.unpack("<Q", r.raw(8))[0] for _ in range(nd)]
        dt = struct.unpack("<I", r.raw(4))[0]
        off = struct.unpack("<Q", r.raw(8))[0]
        infos.append((name, dims, dt, off))
    al = md.get("general.alignment", (5, [32]))[1][0]
    data_start = (f.tell() + al - 1) // al * al
    return f, {(i[0]): i for i in infos}, md, data_start


# ------------------------------------------------------------------ PTQ1_0 decode (ptq_decode.h twin)
def trit_of_byte(b, t):
    v = b
    for _ in range(t):
        v = (v * 3) & 0xFF
    return (v * 3) >> 8


def byte_trit_of(e):
    if e < 80:
        return e % 16, e // 16
    if e < 120:
        return 16 + (e - 80) % 8, (e - 80) // 8
    return 24 + (e - 120) % 2, (e - 120) // 2


def f16(bits):
    return struct.unpack("<e", struct.pack("<H", bits))[0]


def ptq_row(buf, base, row, K):
    """the exact dequantized f32 values of PTQ1_0 matrix row `row` (K elements, K/128 blocks)."""
    nblk = K // PTQ_NW
    rb = row * nblk * PTQ_BLOCK_BYTES
    out = []
    for b in range(nblk):
        p = base + rb + b * PTQ_BLOCK_BYTES
        blk = buf[p:p + PTQ_BLOCK_BYTES]
        d = f16(blk[26] | (blk[27] << 8))
        for e in range(PTQ_NW):
            by, tr = byte_trit_of(e)
            out.append((trit_of_byte(blk[by], tr) - 1) * d)
    return out


# ------------------------------------------------------------------ the transform, hat4096.comp twin
def fwht(x, n=4096):
    """normalized Sylvester-Walsh H: the same ascending-h butterfly order as the kernel, /sqrt(N)."""
    s = list(x)
    h = 1
    while h < n:
        for i in range(n // 2):
            j = ((i & ~(h - 1)) << 1) | (i & (h - 1))
            a, c = s[j], s[j + h]
            s[j] = a + c
            s[j + h] = a - c
        h <<= 1
    sc = 1.0 / (n ** 0.5)
    return [v * sc for v in s]


def f32(v):
    return struct.unpack("<f", struct.pack("<f", v))[0]


def bf16(v):
    """round-to-nearest-even bfloat16 truncation of an f32 - the 8-bit-mantissa cast."""
    b = struct.pack("<I", struct.unpack("<I", struct.pack("<f", v))[0])
    u = struct.unpack("<I", b)[0]
    lsb = (u >> 16) & 1
    u = (u + 0x7FFF + lsb) & 0xFFFF0000
    return struct.unpack("<f", struct.pack("<I", u))[0]


def q8_0_dequant(x, blk=32, ties="even"):
    """ggml's q8_0 re-encoding of the activation column, dequantised again: per `blk` values
    d = absmax/127 (guarded), q = nearbyint(clamp(x/d, -127, 127)), x_hat = d*q.
    This is what ggml-cpu puts in front of a vec_dot against a packed weight type.
    ties="even" is C's nearbyint/lrintf under the default rounding mode; ties="away" is the
    roundf(x) spelling that some ggml builds use. They can only differ on exact halves."""
    out = []
    for i in range(0, len(x), blk):
        seg = x[i:i + blk]
        amax = max(abs(v) for v in seg)
        d = amax / 127.0 if amax > 1e-35 else 1e-35
        for v in seg:
            u = v / d
            if ties == "away":
                q = math.floor(abs(u) + 0.5) * (1.0 if u >= 0.0 else -1.0)
            else:
                q = float(round(u))
            out.append(f32(d * max(-127.0, min(127.0, q))))
    return out


def q8_0_recip(x, blk=32, f32grid=True):
    """The reciprocal-grid q8_0 quantiser. ggml's SIMD quantise kernels do NOT divide by the scale:
    they form scale = absmax/127, then scale_inv = 1/scale, and code q = round(x * scale_inv). Because
    1/scale is itself rounded, x*(1/scale) can land on the other side of a .5 tie from x/scale, which
    flips single int8 codes. One flipped code moves a 4096-wide ternary dot by about
    d_block * |w| ~ 1e-4..3e-4 - which is precisely the size of the residual V8 leaves behind.
    f32grid=True rounds the scale and the reciprocal to f32, as C does."""
    out = []
    for i in range(0, len(x), blk):
        seg = x[i:i + blk]
        amax = max(abs(v) for v in seg)
        if f32grid:
            s = f32(amax / 127.0) if amax > 1e-35 else 1e-35
            inv = f32(1.0 / s)
        else:
            s = amax / 127.0 if amax > 1e-35 else 1e-35
            inv = 1.0 / s
        for v in seg:
            q = max(-127.0, min(127.0, float(round(v * inv))))
            out.append(f32(s * q) if f32grid else s * q)
    return out


def q8_1_dequant(x, blk=32):
    """ggml's q8_1: asymmetric per-block, d = (max-min)/127, q = round((x-min)/d), x_hat = d*q + min."""
    out = []
    for i in range(0, len(x), blk):
        seg = x[i:i + blk]
        mn, mx = min(seg), max(seg)
        d = (mx - mn) / 127.0
        for v in seg:
            out.append(f32(f32(d * max(0.0, min(127.0, float(round((v - mn) / d))))) + mn))
    return out


def main():
    f, ten, md, ds = read_header(GGUF)
    print(f"[gguf] {GGUF}\n[gguf] tensors={len(ten)} data_start={ds} "
          f"sign_values key present={'prism.hadamard.sign_values' in md}")
    sv_et, sv = md["prism.hadamard.sign_values"][0], md["prism.hadamard.sign_values"][1]
    signs4096 = [f32(float(s)) for s in sv[:4096]]
    print(f"[meta] sign_values elem-type={sv_et} n={len(sv)} sum(0..4095)={sum(map(float, sv[:4096])):.0f}"
          f"  (the engine prints 60 for the same slice - runs/qcheck_256.log)")
    eps_key = [k for k in md if k.endswith("layer_norm_rms_epsilon")]
    eps = md[eps_key[0]][1][0] if eps_key else 1e-6
    print(f"[meta] rms_epsilon = {eps!r} from {eps_key or 'default'}")

    def tensor_bytes(name, nbytes, at=0):
        off = ten[name][3]
        f.seek(ds + off + at)
        return f.read(nbytes)

    # ---- 1. rebuild HR on the CPU, exactly the op sequence the align harness replays (ops 0..6)
    K = ten["token_embd.weight"][1][0]
    row = ptq_row(tensor_bytes("token_embd.weight", (K // 128) * 28, TOK * (K // 128) * 28), 0, 0, K)
    rot1 = [f32(v) for v in fwht(row)]                       # op 1: H
    res = [f32(a * b) for a, b in zip(rot1, signs4096)]      # op 2: MUL signs  (= model.input_embed)
    ms = f32(sum(f32(v * v) for v in res) / len(res))
    inv = f32(1.0 / (ms ** 0.5 + 0.0))                       # RMS_NORM, eps below
    inv = 1.0 / ((ms + eps) ** 0.5)
    nrm = [f32(v * inv) for v in res]                        # op 3
    anw = struct.unpack("<4096f", tensor_bytes("blk.0.attn_norm.weight", 16384))
    an = [f32(a * b) for a, b in zip(nrm, anw)]              # op 4
    rot2 = [f32(a * b) for a, b in zip(an, signs4096)]       # op 5

    # The same chain kept in double, i.e. with every f32 clamp removed. This is NOT a competing
    # hypothesis about the engine - it is the MEASURED size of my own reconstruction uncertainty:
    # two legitimate ways to rebuild the operand that differ only in where f32 rounding is inserted.
    # If the q8_0 residual is smaller than the spread this produces, the residual cannot discriminate.
    r1 = fwht(row)
    r2 = [a * b for a, b in zip(r1, signs4096)]
    r3 = [v / ((sum(v * v for v in r2) / len(r2) + eps) ** 0.5) for v in r2]
    r4 = [a * b for a, b in zip(r3, anw)]
    r5 = [a * b for a, b in zip(r4, signs4096)]

    def load_or_compute(path, fn, label):
        if os.path.exists(path) and os.environ.get("QKV_PROBE_CACHE") == "1":
            v = list(struct.unpack("<4096f", open(path, "rb").read()))
            print(f"[cache] {label} loaded from {path} (QKV_PROBE_CACHE=1)")
            return v
        v = fn()
        open(path, "wb").write(struct.pack("<4096f", *v))
        print(f"[cache] {label} written to {path}")
        return v

    hr = load_or_compute(CACHE, lambda: [f32(v) for v in fwht(rot2)], "HR (f32 chain)")
    hr64 = load_or_compute(CACHE64, lambda: fwht(r5), "HR (f64 chain)")
    print("\n=== CPU-rebuilt block-0 operand chain vs the oracle's own print (pass 2) ===")
    # the RES head is the RES anchor's own head (bonsai_forward.h:822); its sum is -0.464154.
    anc = [("RES  input_embed", res, (0.0002, 0.0116, 0.0131), -0.464154),
           ("NRM  norm-0",      nrm, (0.0143, 1.0184, 1.1478), -40.661980),
           ("AN   attn_norm-0", an,  (0.0134, 1.1238, 1.1596), -45.631546),
           ("ROT  node_17",     rot2, (-0.0134, -1.1238, -1.1596), -89.238937),
           ("HR   node_19",     hr,  (-1.3944, 0.0263, -1.4444), -0.860536)]
    for nm, v, h3, ssum in anc:
        print(f"  {nm:<16} head [{v[0]:9.4f} {v[1]:9.4f} {v[2]:9.4f}] want [{h3[0]:9.4f} {h3[1]:9.4f}"
              f" {h3[2]:9.4f}]   sum {sum(v):12.6f} want {ssum:12.6f}")
    rms_spread = (sum((a - b) ** 2 for a, b in zip(hr, hr64)) / len(hr)) ** 0.5
    print(f"  my two reconstructions of the SAME operand differ by rms {rms_spread:.3e} per element"
          f" (f32 chain vs f64 chain)")

    # ---- 2. the six candidate arithmetics for the QKV GEMV over HR
    qkv = ten["blk.0.attn_qkv.weight"]
    Kb = qkv[1][0] // 128
    nblk = Kb
    print(f"\n[weights] blk.0.attn_qkv.weight dims={qkv[1]} type={qkv[2]} blocks/row={nblk}")

    def h(v):                       # round a float to fp16 and back (the FLOAT_TYPE=half hypothesis)
        return struct.unpack("<e", struct.pack("<e", v))[0]

    def variants(r):
        w = ptq_row(tensor_bytes("blk.0.attn_qkv.weight", nblk * 28, r * nblk * 28), 0, 0, nblk * 128)
        w16 = [h(x) for x in w]
        x16 = [h(v) for v in hr]
        # the row's own block scales and trits, straight out of the 28-byte blocks
        f.seek(ds + qkv[3] + r * nblk * 28)
        raw = f.read(nblk * 28)
        ds_l = [f16(raw[i * 28 + 26] | (raw[i * 28 + 27] << 8)) for i in range(nblk)]
        trits = [[trit_of_byte(raw[i * 28 + byte_trit_of(e)[0]], byte_trit_of(e)[1])
                  for e in range(128)] for i in range(nblk)]

        def ternary_form(round_fn, xv=None):
            """the FORM both kernels use: out = sum_b d_b * (sum_{k in b} c_k*x_k - sum_{k in b} x_k).
            round_fn is the accumulator's rounding: identity for FLOAT_TYPE=float, h for half."""
            xv = hr if xv is None else xv
            t = 0.0
            for i in range(nblk):
                s = 0.0
                ysum = 0.0
                for e in range(128):
                    a = xv[i * 128 + e]
                    s = round_fn(s + round_fn(trits[i][e] * a))
                    ysum = round_fn(ysum + a)
                t = round_fn(t + round_fn(ds_l[i] * round_fn(s - ysum)))
            return t

        def direct(round_fn, wr=None, xr=None):
            """the mathematically equivalent form with no cancellation: w = (c-1)*d per element."""
            wr = w if wr is None else wr
            xr = hr if xr is None else xr
            t = 0.0
            for e in range(nblk * 128):
                t = round_fn(t + round_fn(wr[e] * xr[e]))
            return t

        idn = lambda v: v
        out = {}
        out["V0 exact, f64"] = sum(a * b for a, b in zip(w, hr))
        out["V1 ternary form, f32 acc"] = ternary_form(idn)
        out["V2 ternary form, f16 acc"] = ternary_form(h)
        out["V3 direct dot, f32 acc"] = direct(idn)
        out["V4 direct dot, f16 acc"] = direct(h)
        out["V5 direct dot, x rounded f16"] = direct(idn, xr=x16)
        out["V6 direct dot, w rounded f16"] = direct(idn, wr=w16)
        out["V7 direct dot, both rounded f16"] = direct(idn, wr=w16, xr=x16)
        # --- the ggml-cpu "quantized src1" family: before a dot against a packed weight type,
        # ggml converts the f32 activation column into q8_0 blocks (32 values, per-block absmax/127
        # scale, nearbyint codes). That is a LOSSY re-encoding of x, and it is the one candidate
        # whose error scale (absmax/254 per element) is already ~1e-3 of a typical activation value.
        xq = q8_0_dequant(hr)
        out["V8 direct dot, x -> q8_0"] = direct(idn, xr=xq)
        out["V9 ternary form, x -> q8_0"] = ternary_form(idn, xv=xq)
        out["V10 x -> q8_0, then f16 acc"] = direct(h, xr=xq)
        out["V11 x -> q8_0, w rounded f16"] = direct(idn, wr=w16, xr=xq)
        # --- q8_0 emulation details, swept so that a residual here cannot be blamed on my scale
        #     rule: ggml spells the rounding both ways in different builds, the block size is 32 in
        #     q8_0 but 64/128 in relatives, and the quantiser may see an already-f16 src1.
        out["V13 q8_0 blk32, ties away from 0"] = direct(idn, xr=q8_0_dequant(hr, ties="away"))
        out["V14 q8_0 blk64"] = direct(idn, xr=q8_0_dequant(hr, 64))
        out["V15 q8_0, x from the f64 chain"] = direct(idn, xr=q8_0_dequant(hr64))
        out["V16 x -> f16 -> q8_0"] = direct(idn, xr=q8_0_dequant([h(v) for v in hr]))
        # --- and the grid itself: divide-by-scale vs multiply-by-reciprocal-of-scale. The residual
        #     V8 leaves is 2.6e-5..2.8e-4, i.e. ONE flipped code, so this is the discriminating test.
        out["V17 q8_0 reciprocal grid, f32"] = direct(idn, xr=q8_0_recip(hr))
        out["V18 q8_0 reciprocal grid, f64"] = direct(idn, xr=q8_0_recip(hr, f32grid=False))
        out["V19 recip grid + ternary form"] = ternary_form(idn, xv=q8_0_recip(hr))
        # --- the last two ways the same "lossy int8 activation" idea can be spelled: ggml's
        #     asymmetric q8_1, and a row-wide scale (tests whether the block really is 32).
        out["V20 x -> q8_1 (asym, blk32)"] = direct(idn, xr=q8_1_dequant(hr))
        out["V21 x -> q8_0, one scale/row"] = direct(idn, xr=q8_0_dequant(hr, 4096))
        # --- the bf16 activation cast, for completeness (8-bit mantissa, 1e-2 relative per element)
        xb = [bf16(v) for v in hr]
        out["V12 direct dot, x rounded bf16"] = direct(idn, xr=xb)
        return out, w, xq

    print(f"\n=== the six channels the oracle prints: candidate arithmetics vs the 4-decimal window")
    names = None
    hits, worst = {}, {}
    per = {}
    for r in ROWS:
        vv, w, xq = variants(r)
        if names is None:
            names = list(vv)
            print("\n  (a candidate is CONSISTENT only if EVERY channel it predicts lands inside"
                  " want +/- 5e-5)")
        print(f"\n  channel {r}: want {WANT[r]:9.4f}  window [{WANT[r]-HALF_WIN:.5f},"
              f" {WANT[r]+HALF_WIN:.5f}]"
              + (f"   our GPU {OURS_GPU[r]:.5f}" if r in OURS_GPU else ""))
        for nm in names:
            v = vv[nm]
            d = v - WANT[r]
            inw = abs(d) <= HALF_WIN
            hits[nm] = hits.get(nm, 0) + (1 if inw else 0)
            worst[nm] = max(worst.get(nm, 0.0), abs(d))
            print(f"     {nm:<32} {v:12.6f}   {'IN  ' if inw else 'out'}"
                  f"  d(want)={d:+9.5f}")
        # row-scale diagnostics: how much of the exact-vs-oracle gap is cancellation, and how big an
        # activation error each residual corresponds to under the two standard noise models.
        l2 = math.sqrt(sum(a * a for a in w))
        l1 = sum(abs(a) for a in w)
        per[r] = dict(v0=vv["V0 exact, f64"], v8=vv["V8 direct dot, x -> q8_0"],
                      v15=vv["V15 q8_0, x from the f64 chain"],
                      abs_sum=sum(abs(a * b) for a, b in zip(w, xq)), l2=l2, l1=l1)
    print("\n=== verdict table: how many of the oracle's 6 printed channels each arithmetic hits ===")
    print(f"  {'arithmetic':<34} {'in window':>10} {'worst |d(want)|':>17}")
    for nm in names:
        print(f"  {nm:<34} {hits[nm]:>7}/6   {worst[nm]:>15.5f}")

    # ---------------------------------------------------------------- cancellation and reconstruction noise
    print("\n=== is what is LEFT of the q8_0 residual bigger than my uncertainty about the operand? ===")
    print("  row   |V0 exact|   Sum|w*x|  cancel   ||w||2    d(want) V8   recon spread V8-V15"
          "   dx implied (rms / coh)     rms|x|")
    for r in ROWS:
        p = per[r]
        d8 = p["v8"] - WANT[r]
        spread = abs(p["v8"] - p["v15"])
        dx_rms = abs(d8) / p["l2"]
        dx_coh = abs(d8) / p["l1"]
        rms_x = (sum(v * v for v in hr) / len(hr)) ** 0.5
        print(f"  {r:>5} {abs(p['v0']):11.5f} {p['abs_sum']:11.3f} {p['abs_sum']/abs(p['v0']):8.1f}"
              f" {p['l2']:8.4f} {d8:+15.6f} {spread:14.3e}   {dx_rms:10.3e} {dx_coh:10.3e}"
              f" {rms_x:10.3e}")
    print(f"  cancel      = Sum|w_k x_k| / |result|: how many digits the 4096-term dot throws away.\n"
          f"  d(want) V8  = the q8_0 hypothesis' residual against the oracle's 4-decimal print; exact\n"
          f"                arithmetic's residual on the same channel is "
          f"{abs(per[ROWS[0]]['v0'] - WANT[ROWS[0]]):.5f} (ch 0) to "
          f"{abs(per[2]['v0'] - WANT[2]):.5f} (ch 2).\n"
          f"  recon spread= the SAME arithmetic on the f64-chain operand - the floor below which a\n"
          f"                channel-by-channel comparison cannot discriminate anything.\n"
          f"  dx implied  = the per-element activation error the residual would require under\n"
          f"                random-sign noise (/||w||2) and fully coherent noise (/||w||1), shown\n"
          f"                against the operand's own rms.")
    print("\n  A candidate that hits 6/6 identifies the oracle's arithmetic. The reading of this table"
          "\n  is in runs/traitor_forensics_v1.md; nothing here is fitted - the windows are the"
          "\n  oracle's own printed resolution and the candidates are fixed before looking at them.")
    print("[note] d(want) is measured against the oracle's PRINTED value, whose own quantisation is"
          " +/-5e-5.\n[note] our GPU's channel 0/1/2 are transcribed from runs/fwd_align_v10.log.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
