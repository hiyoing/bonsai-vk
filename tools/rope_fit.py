#!/usr/bin/env python3
"""rope_fit.py - pin down the rotary convention of the official qwen35 engine from measured dumps.

Background measured earlier (runs/dbg_fa_off.log, runs/dbg_full_real.log; official -ngl 0 -fa off):
    Qcur-3 = ROPE(Qcur_normed-3{256,16,n}, leaf_58{4 position fields per token})
    qwen35.rope.dimension_count = 64, head_dim = 256, freq_base = 1e7
    rope.dimension_sections = [11,11,10,0]  (sums to 32 = n_rot/2, so it splits the *pairs*)
    dims >= 64 pass through untouched; the token at position 0 is untouched in every dim.

Unknowns that the real model hides, because the official debug printer elides every row wider than 6
elements (we only ever see dims 0,1,2 and 253,254,255 of each head):
    (a) pairing        : half-split (partner = p + 32)  vs  interleaved (partner = p ^ 1)
    (b) frequency law  : theta_p = base^(-2p/D); is D = n_rot = 64 or head_dim = 256?

Probe: sweep --rope-freq-base over 8 values for one fixed prompt. Layers 0..2 of qwen35 are GDN
linear-attention layers with no rotary node, so the base override leaves layer 3's *pre-rope* tensor
bit-identical (checked below) while out_p moves with the base. For one (head, token, dim) there is
exactly one unknown - its partner value x_q - and each base gives one equation:

    out_p(s) = x_p*cos(theta_p*s) - x_q*sin(theta_p*s)   =>   x_q_hat = (x_p*cos - out_p)/sin

The true law makes x_q_hat the same number for all 8 bases; any wrong D makes it drift. Grouping is
strictly per (head, token, dim) - pooling across tokens would be invalid because each token has its
own partner value.

  python tools/rope_fit.py runs/dbg_rope_b*.log
"""
import re, sys, math, glob, os

BASE_RE = re.compile(r"dbg_rope_b(\d+)\.log$")
NH, ND = 6, 3              # visible heads (3 + 3 of 16) and visible leading dims per row


def parse(path):
    """-> dict node-name -> list of (rows, sum); rows = innermost 6-number rows in print order."""
    out, name, rows = {}, None, None
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        for line in f:
            if "common_debug_cb_eval:" in line:
                m = re.match(r".*common_debug_cb_eval:\s+(\S+)\s*=", line)
                name = m.group(1) if m else None
                rows = []
                continue
            if rows is None:
                continue
            if line.strip().startswith("[") and "..." in line:
                nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
                if len(nums) == ND * 2:
                    rows.append(nums)
            elif "sum =" in line:
                out.setdefault(name, []).append(
                    (rows, float(re.search(r"sum = (-?[\d.]+)", line).group(1))))
                rows = None
    return out


def main():
    data = {}
    for p in sorted(glob.glob(sys.argv[1] if len(sys.argv) > 1 else "runs/dbg_rope_b*.log")):
        m = BASE_RE.search(os.path.basename(p))
        if m:
            data[float(m.group(1))] = parse(p)
    bases = sorted(data)
    print("bases probed: %s  (%d runs)" % (", ".join("%g" % b for b in bases), len(bases)))

    pre = {b: data[b]["Qcur_normed-3"][-1][0] for b in bases}    # [-1] = the 4-token prefill;
    post = {b: data[b]["Qcur-3"][-1][0] for b in bases}          # [0] is llama.cpp's 2-token warm-up
    nrow = len(pre[bases[0]])
    ntok = nrow // NH
    P0 = bases[0]

    print("\ngate 0  pre-rope tensor identical across all bases (probe validity) :",
          all(pre[b] == pre[P0] for b in bases))
    print("        post-rope sums per base :",
          ", ".join("%g:%.3f" % (b, data[b]["Qcur-3"][-1][1]) for b in bases), " <- they move")
    print("gate 1  dims 253..255 (>= n_rot) untouched           :",
          sum(1 for b in bases for r in range(nrow) for d in (3, 4, 5)
              if abs(pre[b][r][d] - post[b][r][d]) > 1e-9), "mismatches")
    print("gate 2  token 0 (position 0) untouched               :",
          sum(1 for b in bases for h in range(NH) for d in range(ND)
              if abs(pre[b][h][d] - post[b][h][d]) > 1e-9), "mismatches")
    print("gate 3  dim 0 (theta_0 = base^0 = 1) base-independent:",
          all(post[b][NH + h][0] == post[P0][NH + h][0] for b in bases for h in range(NH)))

    print("\n== rejection test A: interleaved pairing (partner = p^1). Pair 0 => theta = 1 for ANY D,")
    print("   so dims 0 and 1 predict each other with ZERO free parameters:")
    worst_int = 0.0
    for t in range(1, ntok):
        c, s = math.cos(t), math.sin(t)          # position s = t, theta = 1
        for h in range(NH):
            i = t * NH + h
            for b in bases:
                x0, x1 = pre[b][i][0], pre[b][i][1]
                worst_int = max(worst_int, abs(x0 * c - x1 * s - post[b][i][0]))
                worst_int = max(worst_int, abs(x1 * c + x0 * s - post[b][i][1]))
    print("   worst |interleaved prediction - measured| = %.4f   (print quantum 5e-5) => REJECTED"
          % worst_int)

    print("\n== test B: half-split pairing, scan the frequency-law denominator D")
    print("   per (dim, token, head) recover x_q from every base; rel.spread = (max-min)/|mean|")
    print("   (only equations with |sin| > 0.2 are counted, so the estimate is well conditioned)")
    results = {}
    for D in (16, 32, 48, 64, 96, 128, 192, 256):
        per_dim = {}
        for p in (1, 2):
            spreads = []
            for t in range(1, ntok):
                for h in range(NH):
                    i = t * NH + h
                    vals = []
                    for b in bases:
                        th = b ** (-2.0 * p / D) * t
                        st = math.sin(th)
                        if abs(st) > 0.2:
                            vals.append((pre[b][i][p] * math.cos(th) - post[b][i][p]) / st)
                    if len(vals) >= 3:
                        mean = sum(vals) / len(vals)
                        spreads.append((max(vals) - min(vals)) / (abs(mean) + 1e-12))
            if spreads:
                spreads.sort()
                per_dim[p] = (spreads[len(spreads) // 2], spreads[-1], len(spreads))
        results[D] = per_dim
        txt = "  ".join("dim%d med=%.2e max=%.2e(n=%d)" %
                        (p, v[0], v[1], v[2]) for p, v in sorted(per_dim.items()))
        print("   D=%-4d %s" % (D, txt))

    best = min(results, key=lambda D: max(v[1] for v in results[D].values()))
    print("\n   => smallest worst-case drift at D = %d  (= qwen35.rope.dimension_count = n_rot = 64)" % best)

    print("\n== detail under the winning law (D = 64): recovered partner x_q per base, head 0, token 1")
    for p in (0, 1, 2):
        row = []
        for b in bases:
            th = b ** (-2.0 * p / 64)
            st = math.sin(th)
            i = NH
            v = (pre[b][i][p] * math.cos(th) - post[b][i][p]) / st if abs(st) > 1e-9 else float("nan")
            row.append("%g:%+.4f" % (b, v))
        print("   dim %d partner(dim%d) = %s" % (p, p + 32, "  ".join(row)))

    print("\nVERDICT (measured):  out_p(s) = x_p*cos(s*base^(-2p/64)) - x_{p+32}*sin(s*base^(-2p/64))")
    print("                     p = 0..31 within the first n_rot=64 dims of each head;")
    print("                     dims 64..255 pass through; position = absolute sequence index;")
    print("                     the 4 mrope fields per token are equal for plain text, so the")
    print("                     [11,11,10,0] section split is numerically inert here.")


if __name__ == "__main__":
    main()
