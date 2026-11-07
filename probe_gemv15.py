"""Probe 15: measure what llama.cpp's OWN PTQ1_0 Vulkan kernels cost on this card.

STAGE2 section 4 has so far been an instruction-count estimate (~52-60 Gop/s for llama.cpp).
This probe ports the two real kernels from PrismML-Eng/llama.cpp (branch `prism`, fetched today,
repo pushed 2026-10-31) into WGSL and runs them next to our kernel on the same shapes/harness:

  VEC   <- mul_mat_vec_ptq1_0.comp: decode_word() = 2 base-3 muls + shifts/masks per 4 codes,
           f32 activations, dot(vec4), d*(acc - ysum) folded per half-block item, and their
           `if (h == 0u)` qh branch kept as-is (that branch is maximal divergence at tid parity).
  VECQ  <- mul_mat_vecq_ptq1_0.comp, nc==1 branch: 16-bit-lane base-3 recurrence + DOT4, where on
           gfx803 GL_EXT_integer_dot_product is absent so their dot4_emulated() (4 bfe + 4 imul +
           3 add per 2 useful products) is what runs. Counterfactual here: this card never reaches
           this kernel (see the gate below), but it is the shape the same fork uses on DP4A hardware.
  OURS  <- probe 14's production 2-bit bfe kernel.

Blocks are synthesized: random bytes -> trits through the same (v*3)>>8 recurrence in numpy, so
shader and reference agree by construction. Element map is the documented one (qs[j<16] -> 16t+j,
qs[16+j'] -> 80+8t+j', qh[h] -> 120+2t+h, word6 = qh0 | qh1<<8 | d<<16). Activations live on a
u8 grid x = s*a, so VEC (f32) and VECQ (int8) must reproduce the same reference: the numeric
check doubles as the element-map consistency test.

Fidelity caveats, all of which only flatter the ports:
 - unsigned bytes, so extractBits is 1 op; their signed bitfieldExtract costs the same or more.
 - the fp16 block scale d is read from an array instead of unpacked out of word 6 (~1 op less).
 - the quantize-x pass that VECQ needs is not timed here (it is a separate kernel in llama.cpp).

Dispatch constants are the ones this card actually gets - read from ggml-vulkan.cpp and from
vulkaninfo run on this machine today:
 - ggml-vulkan.cpp:4333: device->integer_dot_product = extension present && integerDotProduct4x8BitPackedSignedAccelerated.
   vulkaninfo here: all three 4x8BitPacked flags are FALSE -> quantize_y is false (line 6830) and the
   q8_1 pipeline is never created (line 3004). So a single-token PTQ1_0 decode on THIS card runs
   mul_mat_vec_ptq1_0.comp with f32 activations = the VEC port, not VECQ.
 - GCN sets rm_stdq = 2, rm_kq = 4, rm_stdq_int = 4 while rm_kq_int stays 1 (lines 2877-2887).
   The f32 PTQ1_0 vec pipeline is created with {2*rm_stdq} = 4 rows (line 2945); the q8_1 one gets
   {4*rm_kq_int} = 4 rows, CG = 3, and NUM_COLS = i+1 = 1 for a single token, i.e. process_group's
   nc == 1 branch (line 3010).
 - w = DMMV_WG_SIZE_SUBGROUP for AMD (the LARGE override is NVIDIA/Intel only, lines 5667-5685) ->
   BLOCK_SIZE = device->subgroup_size, and vulkaninfo reports subgroupSize = 64 because GCN is
   wave64 (RDNA reports 32). So both ports are built at 64 threads = exactly one wavefront; the
   32-lane variant is timed as a sensitivity check.
rows1 is kept as the "what a single-row kernel costs" reference point.
"""
import time
import numpy as np
import wgpu

LANE = "0x00FF00FFu"          # bytes 0 and 2
YMASK = "0xFF00FF00u"         # bytes 1 and 3
TRIM = "0x00030003u"

HDR = """struct P { m: u32, k: u32, nb: u32, splits: u32 };
@group(0) @binding(0) var<storage, read> ab: array<u32>;          // 7 words / 128-weight block
@group(0) @binding(1) var<storage, read> xv: array<vec4<f32>>;    // f32 activations
@group(0) @binding(2) var<storage, read> sc: array<f32>;          // block scale d, [nb][m]
@group(0) @binding(3) var<storage, read> bs: array<f32>;          // sum of x per (block, lane)
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> p: P;
@group(0) @binding(6) var<storage, read> ya4: array<vec4<u32>>;   // u8x4 activations, 16B view
@group(0) @binding(7) var<storage, read> ya1: array<u32>;         // same buffer, dword view
@group(0) @binding(8) var<storage, read> sq: array<f32>;          // per-block u8 scale s
"""
HDR_OURS = HDR.replace("var<storage, read> ab: array<u32>;", "var<storage, read> ab: array<vec4<u32>>;")


def digits(b, n):
    """byte -> the n trits the shader's recurrence yields: m = v*3, d = m>>8, v = m & 0xFF."""
    u = b.astype(np.uint16)
    d = np.zeros(b.shape + (n,), dtype=np.uint8)
    for t in range(n):
        m = u * 3
        d[..., t] = ((m >> 8) & 3).astype(np.uint8)
        u = (m & 0xFF).astype(np.uint16)
    return d


def make_blocks(M, K, nb, rng):
    """random weight bytes -> (flat u32 words [M,nb,7], trits [M,nb,128], d [nb,M])."""
    qb = rng.integers(0, 256, size=(M, nb, 24), dtype=np.uint8)
    hb = rng.integers(0, 256, size=(M, nb, 2), dtype=np.uint8)
    qd, hd = digits(qb, 5), digits(hb, 4)
    tr = np.zeros((M, nb, 128), dtype=np.uint8)
    for t in range(5):
        tr[:, :, 16 * t + np.arange(16)] = qd[:, :, :16, t]
        tr[:, :, 80 + 8 * t + np.arange(8)] = qd[:, :, 16:, t]
    for t in range(4):
        tr[:, :, 120 + 2 * t + np.arange(2)] = hd[:, :, :, t]
    d = (1.0 + 0.1 * rng.standard_normal((nb, M))).astype(np.float32)
    w = np.zeros((M, nb, 7), dtype=np.uint32)
    for k in range(6):
        for b in range(4):
            w[:, :, k] |= qb[:, :, 4 * k + b].astype(np.uint32) << np.uint32(8 * b)
    w[:, :, 6] = (hb[:, :, 0].astype(np.uint32) | (hb[:, :, 1].astype(np.uint32) << np.uint32(8))
                  | (d.T.astype(np.float16).view(np.uint16).astype(np.uint32) << np.uint32(16)))
    return w.reshape(-1).copy(), tr, d


def elem_of(k, t, b):
    """element index in the block of byte b of qs dword k at trit level t."""
    return (4 * k + 16 * t if k < 4 else 80 + 4 * (k - 4) + 8 * t) + b


# Both llama.cpp kernels end in reduce_result() (mul_mat_vec_base.glsl:138-241): per-thread partials
# either go through subgroupAdd (SHADER_REDUCTION_MODE_SUBGROUP, what this card gets because
# use_subgroups = device->subgroup_arithmetic is true on the AMD proprietary driver) or through the
# shared-memory halving tree below. Measured here is the SHMEM tree, because naga/WGSL has no
# subgroupAdd on this build; it costs 6 extra barriers per row-group, i.e. it is the slower of the
# two, which only understates the baseline.
#
# WG = BLOCK_SIZE = spec constant 0 = wg_size_subgroup = device->subgroup_size. vulkaninfo on this
# machine reports subgroupSize = 64: GCN runs wave64 (RDNA would say 32), so llama.cpp's workgroup
# here is one 64-lane wavefront. 32 is measured too, to show what a wave32 card gets.
DEFAULT_WG = 64


def reduce_snippet(R, WG):
    st = ["".join("    sh[%du * %du + li.x] = tot%d;\n" % (n, WG, n) for n in range(R)),
          "    workgroupBarrier();"]
    s = WG // 2
    while s > 0:
        st.append("    if (li.x < %du) {\n%s    }\n    workgroupBarrier();\n"
                  % (s, "".join("        sh[%du * %du + li.x] += sh[%du * %du + li.x + %du];\n"
                                % (n, WG, n, WG, s) for n in range(R))))
        s //= 2
    st.append("    if (li.x == 0u) {\n%s    }\n"
              % "".join("        if (first_row + %du < p.m) { out[first_row + %du] = sh[%du * %du]; }\n"
                        % (n, n, n, WG) for n in range(R)))
    return "".join(st)


# ------------------------------------------------------------------ VEC: mul_mat_vec_ptq1_0
def vec_src(R, WG=DEFAULT_WG):
    body = []
    for n in range(R):
        body.append("        var acc%d: f32 = 0.0;\n        var ys%d: f32 = 0.0;\n" % (n, n))
    for kk in range(3):
        for n in range(R):
            body.append("        let wbase%d_%d = (r%d * p.nb + ib) * 7u + 3u * h + %du;\n" % (n, kk, n, kk))
            body.append("        var ev%d_%d = ab[wbase%d_%d] & %s;\n" % (n, kk, n, kk, LANE))
            body.append("        var od%d_%d = (ab[wbase%d_%d] >> 8u) & %s;\n" % (n, kk, n, kk, LANE))
        for t in range(5):
            for n in range(R):
                body.append("        let me%d_%d_%d = ev%d_%d * 3u;\n" % (kk, t, n, n, kk))
                body.append("        let mo%d_%d_%d = od%d_%d * 3u;\n" % (kk, t, n, n, kk))
                body.append("        let te%d_%d_%d = (me%d_%d_%d >> 8u) & %s;\n" % (kk, t, n, kk, t, n, TRIM))
                body.append("        let to%d_%d_%d = (mo%d_%d_%d >> 8u) & %s;\n" % (kk, t, n, kk, t, n, TRIM))
                body.append("        ev%d_%d = me%d_%d_%d & %s;\n" % (n, kk, kk, t, n, LANE))
                body.append("        od%d_%d = mo%d_%d_%d & %s;\n" % (n, kk, kk, t, n, LANE))
                body.append("        let q%d_%d_%d = vec4<f32>(f32(te%d_%d_%d & 3u), f32(to%d_%d_%d & 3u), "
                            "f32(te%d_%d_%d >> 16u), f32(to%d_%d_%d >> 16u));\n"
                            % (kk, t, n, kk, t, n, kk, t, n, kk, t, n, kk, t, n))
            body.append("        let wi%d_%d = select(3u * h + %du + 4u * %du, "
                        "20u + (3u * h + %du - 4u) + 2u * %du, (3u * h + %du) >= 4u);\n"
                        % (kk, t, kk, t, kk, t, kk))
            body.append("        let yv%d_%d = xv[ib * 32u + wi%d_%d];\n" % (kk, t, kk, t))
            body.append("        let y4%d_%d = (yv%d_%d.x + yv%d_%d.y) + (yv%d_%d.z + yv%d_%d.w);\n"
                        % (kk, t, kk, t, kk, t, kk, t, kk, t))
            for n in range(R):
                body.append("        ys%d = ys%d + y4%d_%d;\n" % (n, n, kk, t))
                body.append("        acc%d = acc%d + dot(q%d_%d_%d, yv%d_%d);\n" % (n, n, kk, t, n, kk, t))
    for n in range(R):
        body.append("        if (h == 0u) {\n")
        body.append("            let w6%d = ab[(r%d * p.nb + ib) * 7u + 6u];\n" % (n, n))
        body.append("            var xq%d = (w6%d & 0xFFu) | ((w6%d >> 8u) & 0xFFu) << 16u;\n" % (n, n, n))
        for t in range(4):
            body.append("            let mq%d_%d = xq%d * 3u;\n" % (n, t, n))
            body.append("            let tq%d_%d = (mq%d_%d >> 8u) & %s;\n" % (n, t, n, t, TRIM))
            body.append("            xq%d = mq%d_%d & %s;\n" % (n, n, t, LANE))
        body.append("            let qa%d = vec4<f32>(f32(tq%d_0 & 3u), f32(tq%d_0 >> 16u), "
                    "f32(tq%d_1 & 3u), f32(tq%d_1 >> 16u));\n" % (n, n, n, n, n))
        body.append("            let qb%d = vec4<f32>(f32(tq%d_2 & 3u), f32(tq%d_2 >> 16u), "
                    "f32(tq%d_3 & 3u), f32(tq%d_3 >> 16u));\n" % (n, n, n, n, n))
        body.append("            let yq0%d = xv[ib * 32u + 30u];\n" % n)
        body.append("            let yq1%d = xv[ib * 32u + 31u];\n" % n)
        body.append("            ys%d = ys%d + ((yq0%d.x + yq0%d.y) + (yq0%d.z + yq0%d.w))"
                    " + ((yq1%d.x + yq1%d.y) + (yq1%d.z + yq1%d.w));\n" % (n, n, n, n, n, n, n, n, n, n))
        body.append("            acc%d = acc%d + dot(qa%d, yq0%d) + dot(qb%d, yq1%d);\n" % (n, n, n, n, n, n))
        body.append("            tot%d = tot%d + sc[ib * p.m + r%d] * (acc%d - bs[ib * 4u + 0u + h]);\n" % (n, n, n, n))
        body.append("        } else {\n")
        body.append("            tot%d = tot%d + sc[ib * p.m + r%d] * (acc%d - bs[ib * 4u + 0u + h]);\n" % (n, n, n, n))
        body.append("        }\n")
    pre = "".join("    let r%d = min(first_row + %du, p.m - 1u);\n" % (n, n) for n in range(R))
    acc = "".join("    var tot%d: f32 = 0.0;\n" % n for n in range(R))
    return HDR + """
var<workgroup> sh: array<f32, %du>;

@compute @workgroup_size(%du)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let first_row = wg.x * %du;
    if (first_row >= p.m) { return; }
%s%s    for (var item: u32 = li.x; item < 2u * p.nb; item = item + %du) {
        let ib = item >> 1u;
        let h = item & 1u;
%s    }
%s}
""" % (R * WG, WG, R, pre, acc, WG, "".join(body), reduce_snippet(R, WG))


# ----------------------------------------------------------------- VECQ: mul_mat_vecq_ptq1_0
# their accumulator mapping for the nc==1 path (lines 156-174): (level, dword) -> accumulator
VQ_ACC = {(0, 0): "A0", (0, 1): "B0", (0, 2): "B0",
          (1, 0): "A0", (1, 1): "B0", (1, 2): "B0",
          (2, 0): "A1", (2, 1): "B1", (2, 2): "B1",
          (3, 0): "A1", (3, 1): "B1", (3, 2): "B1",
          (4, 0): "A2", (4, 1): "B2", (4, 2): "B2"}


def vecq_src(R, WG=DEFAULT_WG):
    b = []
    # per-block activation prep, outside the row loop (their lines 117-135)
    b.append("        let s = sq[ib];\n")
    b.append("        let bias = bs[ib * 4u + 2u + sub];\n")
    b.append("        let qh0 = ya1[ib * 32u + 30u];\n")
    b.append("        let qh1 = ya1[ib * 32u + 31u];\n")
    for t in range(5):
        b.append("        let q4%d = ya4[ib * 8u + %du];\n" % (t, t))
        b.append("        let qa%d = ya1[ib * 32u + 20u + %du];\n" % (t, 2 * t))
        b.append("        let qbb%d = ya1[ib * 32u + 21u + %du];\n" % (t, 2 * t))
        for kk in range(3):
            sel = ["q4%d.x", "q4%d.y", "q4%d.z"][kk] % t if kk < 3 else ""
            alt = ["q4%d.w", "qa%d", "qbb%d"][kk] % t
            b.append("        let yy%d_%d = select(%s, %s, sub != 0u);\n" % (kk, t, sel, alt))
            b.append("        let yh%d_%d = yy%d_%d & %s;\n" % (kk, t, kk, t, YMASK))
            b.append("        let yl%d_%d = (yy%d_%d << 8u) & %s;\n" % (kk, t, kk, t, YMASK))
    for n in range(R):
        b.append("        let abase%d = (r%d * p.nb + ib) * 7u;\n" % (n, n))
        b.append("        let w0%d = ab[abase%d + 3u * sub];\n" % (n, n))
        b.append("        let w1%d = ab[abase%d + 3u * sub + 1u];\n" % (n, n))
        b.append("        let w2%d = ab[abase%d + 3u * sub + 2u];\n" % (n, n))
        b.append("        let w6%d = ab[abase%d + 6u];\n" % (n, n))
        for kk, wn in enumerate(["w0", "w1", "w2"]):
            b.append("        var vl%d_%d = %s%d & %s;\n" % (n, kk, wn, n, LANE))
            b.append("        var vh%d_%d = (%s%d >> 8u) & %s;\n" % (n, kk, wn, n, LANE))
        b.append("        var aA0%d: i32 = 0;\n        var aA1%d: i32 = 0;\n        var aA2%d: i32 = 0;\n" % (n, n, n))
        b.append("        var aB0%d: i32 = 0;\n        var aB1%d: i32 = 0;\n        var aB2%d: i32 = 0;\n" % (n, n, n))
        for t in range(5):
            for kk in range(3):
                acc = VQ_ACC[(t, kk)]
                last = (t == 4)
                if not last:
                    b.append("        let wl%d_%d_%d = (vl%d_%d << 1u) + vl%d_%d;\n" % (kk, t, n, n, kk, n, kk))
                    b.append("        let wh%d_%d_%d = (vh%d_%d << 1u) + vh%d_%d;\n" % (kk, t, n, n, kk, n, kk))
                    b.append("        vl%d_%d = wl%d_%d_%d & %s;\n" % (n, kk, kk, t, n, LANE))
                    b.append("        vh%d_%d = wh%d_%d_%d & %s;\n" % (n, kk, kk, t, n, LANE))
                else:
                    b.append("        let wl%d_%d_%d = (vl%d_%d << 1u) + vl%d_%d;\n" % (kk, t, n, n, kk, n, kk))
                    b.append("        let wh%d_%d_%d = (vh%d_%d << 1u) + vh%d_%d;\n" % (kk, t, n, n, kk, n, kk))
                b.append("        a%s%d = dot4(wl%d_%d_%d, yl%d_%d, a%s%d);\n" % (acc, n, kk, t, n, kk, t, acc, n))
                b.append("        a%s%d = dot4(wh%d_%d_%d, yh%d_%d, a%s%d);\n" % (acc, n, kk, t, n, kk, t, acc, n))
        # qh: 4 levels of 2 bytes, only on lane sub!=0 (lane 0 feeds zeros) - their lines 293-308
        b.append("        var vq%d = select(0u, (w6%d & 0xFFu) | ((w6%d >> 8u) & 0xFFu) << 16u, sub != 0u);\n" % (n, n, n))
        b.append("        var q01%d: i32 = 0;\n        var q23%d: i32 = 0;\n" % (n, n))
        for t in range(4):
            b.append("        let mw%d_%d = (vq%d << 1u) + vq%d;\n" % (n, t, n, n))
            if t < 3:
                b.append("        vq%d = mw%d_%d & %s;\n" % (n, n, t, LANE))
            dst = "q01" if t < 2 else "q23"
            if t % 2 == 0:
                b.append("        %s%d = i32(((mw%d_%d >> 8u) & 3u) | ((mw%d_%d >> 16u) & 0x300u));\n"
                         % (dst, n, n, t, n, t))
            else:
                b.append("        %s%d = %s%d | i32(((mw%d_%d << 8u) & 0x30000u) | (mw%d_%d & 0x03000000u));\n"
                         % (dst, n, dst, n, n, t, n, t))
        b.append("        aB2%d = dot4(bitcast<u32>(q01%d), qh0, aB2%d);\n" % (n, n, n))
        b.append("        aB2%d = dot4(bitcast<u32>(q23%d), qh1, aB2%d);\n" % (n, n, n))
        b.append("        let sm%d = fma(s, f32(aA0%d), fma(s, f32(aA1%d), fma(s, f32(aA2%d),\n"
                 "                   fma(s, f32(aB0%d), fma(s, f32(aB1%d), fma(s, f32(aB2%d), -bias))))));\n"
                 % (n, n, n, n, n, n, n))
        b.append("        tot%d = tot%d + sc[ib * p.m + r%d] * sm%d;\n" % (n, n, n, n))
    pre = "".join("    let r%d = min(first_row + %du, p.m - 1u);\n" % (n, n) for n in range(R))
    acc0 = "".join("    var tot%d: f32 = 0.0;\n" % n for n in range(R))
    return HDR + """
fn dot4(a: u32, b: u32, acc: i32) -> i32 {
    return acc + i32(extractBits(a, 0u, 8u)) * i32(extractBits(b, 0u, 8u))
        + i32(extractBits(a, 8u, 8u)) * i32(extractBits(b, 8u, 8u))
        + i32(extractBits(a, 16u, 8u)) * i32(extractBits(b, 16u, 8u))
        + i32(extractBits(a, 24u, 8u)) * i32(extractBits(b, 24u, 8u));
}

var<workgroup> sh: array<f32, %du>;

@compute @workgroup_size(%du)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
    let first_row = wg.x * %du;
    if (first_row >= p.m) { return; }
    let sub = li.x & 1u;
%s%s    for (var ib: u32 = li.x >> 1u; ib < p.nb; ib = ib + %du) {
%s    }
%s}
""" % (R * WG, WG, R, pre, acc0, WG // 2, "".join(b), reduce_snippet(R, WG))


# ------------------------------------------------------------------------- OURS (2-bit bfe)
def ours_src():
    lines = []
    for v in range(2):
        lines.append("        let w%d = ab[(i0 + %du) * p.m + row];\n" % (v, v))
        for k in range(4):
            lines.append("        var a%d_%d: f32 = 0.0;\n" % (v, k))
    for v in range(2):
        for u in range(4):
            for t in range(4):
                lines.append("        let xv%d_%d_%d = xv[(i0 + %du) * 16u + %du + %du];\n" % (v, u, t, v, u * 4, t))
                for cc in range(4):
                    j = t * 4 + cc
                    lines.append("        a%d_%d = a%d_%d + xv%d_%d_%d.%s * f32(extractBits(w%d.%s, %du, 2u));\n"
                                 % (v, cc, v, cc, v, u, t, "xyzw"[cc], v, "xyzw"[u], 2 * j))
        lines.append("        let cp%d = (a%d_0 + a%d_1) + (a%d_2 + a%d_3);\n" % (v, v, v, v, v))
    lines.append("        acc = acc + sc[b * p.m + row] * ((cp0 + cp1) - bs[b]);\n")
    return HDR_OURS + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    let per = p.nb / p.splits;
    let b_lo = g.y * per;
    let b_hi = b_lo + per;
    var acc: f32 = 0.0;
    for (var b: u32 = b_lo; b < b_hi; b = b + 1u) {
        let i0 = b * 2u;
%s    }
    if (p.splits == 1u) { out[row] = acc; } else { out[g.y * p.m + row] = acc; }
}
""" % "".join(lines)


REDUCE = HDR_OURS + """
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
    let row = g.x;
    if (row >= p.m) { return; }
    var s: f32 = out[row];
    for (var i: u32 = 1u; i < p.splits; i = i + 1u) { s = s + out[i * p.m + row]; }
    out[row] = s;
}
"""


def pack2(tr, M, K, nb):
    """trits [M,nb,128] -> [K/64][M] vec4<u32>, 2 bits per code, element order = row-major."""
    c = tr.reshape(M, K)
    out = np.zeros((K // 64, M, 4), dtype=np.uint32)
    for i in range(K // 64):
        for u in range(4):
            w = np.zeros(M, dtype=np.uint32)
            for j in range(16):
                w |= c[:, i * 64 + u * 16 + j].astype(np.uint32) << np.uint32(2 * j)
            out[i, :, u] = w
    return out


def main():
    ad = wgpu.gpu.request_adapter_sync(power_preference="high-performance")
    dev = ad.request_device_sync()
    print("adapter:", dict(ad.info).get("device"), "|", dict(ad.info).get("backend_type"), "\n")
    entries = []
    for i in range(9):
        t = (wgpu.BufferBindingType.storage if i == 4 else
             wgpu.BufferBindingType.uniform if i == 5 else wgpu.BufferBindingType.read_only_storage)
        entries.append({"binding": i, "visibility": wgpu.ShaderStage.COMPUTE, "buffer": {"type": t}})
    bgl = dev.create_bind_group_layout(entries=entries)
    pl = dev.create_pipeline_layout(bind_group_layouts=[bgl])
    rng = np.random.default_rng(15)

    for M, K, tag in [(4096, 4096, "attn q/o"), (12288, 4096, "gate+up"),
                      (4096, 12288, "down"), (1024, 4096, "k/v narrow")]:
        nb = K // 128
        words, tr, d = make_blocks(M, K, nb, rng)
        s_val = np.float32(1.0 / 255.0)
        a = rng.integers(0, 256, size=K, dtype=np.uint8)
        s_blk = np.full(nb, s_val, dtype=np.float32)
        x = (a.astype(np.float32) * s_val).astype(np.float32)
        w1 = (tr.astype(np.float32) - 1.0).reshape(M, nb, 128)
        ref = np.zeros(M, dtype=np.float64)
        for b in range(nb):
            ref += d[b].astype(np.float64) * (w1[:, b, :] @ x[b * 128:(b + 1) * 128].astype(np.float64))

        # per-(block,lane) activation sums: lanes 0,1 = VEC's h halves, 2,3 = VECQ's sub halves.
        # Both splits are the same set (item lane decides which qs dwords), and the lane that owns
        # the qh word must count elements 120..127 too - that was the bug that made both ports read
        # ~1.0 relative error before this line was fixed.
        xs = x.reshape(nb, 128)
        bs = np.zeros((nb, 4), dtype=np.float32)
        qh = [120 + i for i in range(8)]
        for b in range(nb):
            for lane in range(2):
                idx = [elem_of(3 * lane + kk, t, bb) for kk in range(3) for t in range(5) for bb in range(4)]
                bs[b, lane] = xs[b][idx + (qh if lane == 0 else [])].sum()          # VEC: h == 0 owns qh
                bs[b, 2 + lane] = xs[b][idx + (qh if lane == 1 else [])].sum()      # VECQ: sub == 1 owns qh

        ya1 = a.view(np.uint32) if K % 4 == 0 else None
        ypad = np.zeros((ya1.size + 3) // 4 * 4, dtype=np.uint32)
        ypad[:ya1.size] = ya1
        p2 = pack2(tr, M, K, nb)
        xsum = x.reshape(nb, 128).sum(axis=1).astype(np.float32)

        def buf(arr):
            return dev.create_buffer_with_data(data=arr.tobytes(), usage=wgpu.BufferUsage.STORAGE)

        dummy = buf(np.zeros(16, dtype=np.uint32))
        b_ab, b_xv, b_sc, b_bs = buf(words), buf(x), buf(d.reshape(-1)), buf(bs.reshape(-1))
        b_out = dev.create_buffer(size=M * 8 * 4, usage=wgpu.BufferUsage.STORAGE | wgpu.BufferUsage.COPY_SRC)
        b_ya4, b_ya1 = buf(ypad), buf(ypad)
        b_sq = buf(s_blk)
        b_wv = buf(p2.reshape(-1))
        b_xs = buf(xsum)

        built = []

        def add(label, src, grid, rows, splits=1, need_reduce=False):
            par = np.array([M, K, nb, splits], dtype=np.uint32)
            b_p = dev.create_buffer_with_data(data=par.tobytes(), usage=wgpu.BufferUsage.UNIFORM)
            bg = dev.create_bind_group(layout=bgl, entries=[
                {"binding": 0, "resource": {"buffer": b_wv if label.startswith("OURS") else b_ab,
                                            "offset": 0, "size": (p2.nbytes if label.startswith("OURS") else words.nbytes)}},
                {"binding": 1, "resource": {"buffer": b_xv, "offset": 0, "size": x.nbytes}},
                {"binding": 2, "resource": {"buffer": b_sc, "offset": 0, "size": d.nbytes}},
                {"binding": 3, "resource": {"buffer": (b_xs if label.startswith("OURS") else b_bs),
                                            "offset": 0, "size": (xsum.nbytes if label.startswith("OURS") else bs.nbytes)}},
                {"binding": 4, "resource": {"buffer": b_out, "offset": 0, "size": M * 8 * 4}},
                {"binding": 5, "resource": {"buffer": b_p, "offset": 0, "size": 16}},
                {"binding": 6, "resource": {"buffer": b_ya4, "offset": 0, "size": ypad.nbytes}},
                {"binding": 7, "resource": {"buffer": b_ya1, "offset": 0, "size": ypad.nbytes}},
                {"binding": 8, "resource": {"buffer": b_sq, "offset": 0, "size": s_blk.nbytes}},
            ])
            pipe = dev.create_compute_pipeline(layout=pl, compute={"module": dev.create_shader_module(code=src),
                                                                   "entry_point": "main"})
            rp = (dev.create_compute_pipeline(layout=pl, compute={"module": dev.create_shader_module(code=REDUCE),
                                                                  "entry_point": "main"}) if need_reduce else None)
            par_buf = b_p

            def run(c):
                c.set_pipeline(pipe); c.set_bind_group(0, bg); c.dispatch_workgroups(*grid)
                if rp is not None:
                    c.set_pipeline(rp); c.set_bind_group(0, bg); c.dispatch_workgroups((M + 63) // 64, 1, 1)
            enc = dev.create_command_encoder()
            cc = enc.begin_compute_pass(); run(cc); cc.end()
            dev.queue.submit([enc.finish()])
            got = np.frombuffer(dev.queue.read_buffer(b_out, 0, M * 4), dtype=np.float32)
            rel = float(np.abs(got - ref).max() / np.abs(ref).max())
            built.append((label, run, rel, rows, splits))

        def grid(R):
            return ((M + R - 1) // R, 1, 1)

        add("VEC rows1", vec_src(1), grid(1), 1)
        add("VEC rows4", vec_src(4), grid(4), 4)
        add("VEC rows4 wg32", vec_src(4, 32), grid(4), 4)
        add("VECQ rows1", vecq_src(1), grid(1), 1)
        add("VECQ rows4", vecq_src(4), grid(4), 4)
        add("VECQ rows4 wg32", vecq_src(4, 32), grid(4), 4)
        for sp in (1, 4):
            add("OURS splits%d" % sp, ours_src(), ((M + 63) // 64, sp, 1), 1, sp, sp > 1)

        res = {b[0]: [] for b in built}
        for _ in range(3):
            for label, run, _r, _w, _s in built:
                enc = dev.create_command_encoder()
                cc = enc.begin_compute_pass()
                for _ in range(5):
                    run(cc)
                cc.end(); dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(b_out, 0, 4)
                t0 = time.perf_counter()
                enc = dev.create_command_encoder()
                cc = enc.begin_compute_pass()
                for _ in range(50):
                    run(cc)
                cc.end(); dev.queue.submit([enc.finish()])
                dev.queue.read_buffer(b_out, 0, 4)
                res[label].append((time.perf_counter() - t0) / 50)

        ptq_mb = words.nbytes / 1e6
        print("=== %s M=%d K=%d  PTQ1_0 %.2f MB (%.2f bpw) ===" % (tag, M, K, ptq_mb, ptq_mb * 8e6 / (M * K)))
        print("%-14s %8s %8s %9s %8s  %s" % ("variant", "med ms", "best", "Gop/s", "GB/s", "relerr"))
        for label, run, rel, rows, splits in built:
            med, best = float(np.median(res[label])), min(res[label])
            print("%-14s %8.3f %8.3f %9.1f %8.1f  %.1e" % (label, med * 1e3, best * 1e3,
                                                           M * K / med / 1e9, ptq_mb / 1e3 / med, rel))
        print()


if __name__ == "__main__":
    main()
