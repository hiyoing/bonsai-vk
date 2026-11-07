#!/usr/bin/env python3
"""attn_softmax_check.py - verify the attention score->weight semantics against the official engine.

Oracle: runs/dbg_attn_oracle.log, produced by
    llama-debug.exe -m Ternary-Bonsai-2-9B-PTQ1_0.gguf -p "one two three four" -n 0 -c 32 -ngl 0 -fa off
                    --tensor-filter 'kq-3|kq_soft_max-3|attn_inp'
The two nodes are the pre-Flash-Attention path the fork takes with -fa off:
    kq-3          = MUL_MAT(cache_k_l3 (f16, permuted){256, cap, 4}, Qcur-3 (f32, permuted){256, n, 16})
    kq_soft_max-3 = SOFT_MAX(kq-3, attn_inp_kq_mask{cap, n, 1, 1})          (mask broadcast over heads)

The printer elides rows wider than 6, so for tensor {cap=256, n_tokens, 16 heads} we get, per head and
per query token, exactly the scores/probabilities at key positions 0,1,2 and 253,254,255. That is
enough, because the causal mask makes query token t attend only to positions 0..t:
    t = 1 -> unmasked set {0,1}          fully visible  -> probabilities are computable exactly
    t = 2 -> unmasked set {0,1,2}        fully visible  -> ditto, and two independent ratios
    t = 3 -> unmasked set {0,1,2,3}      position 3 hidden -> skipped, not guessed
Everything masked must come out as exactly 0.0000, which is the "fill value" assertion.

The scan is over the scale that SOFT_MAX applies (llama.cpp passes 1/sqrt(head_dim) as an op attr, and
head_dim = 256 here): 0.0625 vs 1/sqrt(64) vs 1/sqrt(128) vs 1.0 vs a free fit. Print quantum is 5e-5,
so a wrong scale shows up as O(1) error and the right one as error at the quantum.
"""
import re, sys, math

LOG = sys.argv[1] if len(sys.argv) > 1 else "runs/dbg_attn_oracle.log"


def parse(path):
    out, name, rows = {}, None, None
    for line in open(path, "r", encoding="utf-8", errors="replace"):
        if "common_debug_cb_eval:" in line:
            m = re.match(r".*common_debug_cb_eval:\s+(\S+)\s*=", line)
            name = m.group(1) if m else None
            rows = []
            continue
        if rows is None:
            continue
        if line.strip().startswith("[") and "..." in line:
            nums = [float(x) for x in re.findall(r"-?\d+\.\d+", line)]
            if len(nums) == 6:
                rows.append(nums)
        elif "sum =" in line:
            out.setdefault(name, []).append((rows, float(re.search(r"sum = (-?[\d.]+)", line).group(1))))
            rows = None
    return out


def softmax(scale, s):
    mx = max(s)
    e = [math.exp(scale * (x - mx)) for x in s]
    z = sum(e)
    return [v / z for v in e]


def main():
    d = parse(LOG)
    # the 4-token prefill is the dump whose rows = 6 head-groups x 4 tokens = 24 rows
    key_q = [x for x in d["kq-3"] if len(x[0]) == 24]
    key_p = [x for x in d["kq_soft_max-3"] if len(x[0]) == 24]
    assert key_q and key_p, "no 4-token prefill dumps in %s" % LOG
    q, pq = key_q[-1][0], key_p[-1][0]
    # kq is {cap=256, n_tokens, n_head}: the printer's outer axis is n_head (16 -> 3+3 shown) and the
    # rows inside one group are the 4 query tokens, so the flattened row index is group*NTOK + token.
    NH, NTOK = 6, 4
    print("layer-3 prefill: %d rows (=%d head-groups x 4 tokens), kq sum=%.6f, p sum=%.6f"
          % (len(q), NH, key_q[-1][1], key_p[-1][1]))

    print("\n== assertion: every causally-masked slot is exactly 0 ==")
    bad = []
    for g in range(NH):
        for t in range(NTOK):
            p = pq[g * NTOK + t]
            for pos in (253, 254, 255):
                if p[3] != 0.0 or p[4] != 0.0 or p[5] != 0.0:
                    bad.append((g, t))
            if t < 2 and p[2] != 0.0:          # position 2 is future for tokens 0 and 1
                bad.append((g, t))
    print("   violations: %d" % len(set(bad)))
    print("   exact-zero check on all printed masked slots: %s"
          % all(pq[g * NTOK + t][k] == 0.0 for g in range(NH) for t in range(NTOK)
                for k in (3, 4, 5)))

    print("\n== row normalisation as printed (each unmasked row carries total mass 1) ==")
    tot = sum(sum(r) for r in pq)
    print("   sum of all printed probabilities = %.6f   (only the 0/1/2 columns can be non-zero,"
          " so this is a lower bound on the true row sums)" % tot)

    print("\n== candidate scales: our softmax(kq) vs the official probabilities ==")
    HD = 256
    cands = {"1/sqrt(head_dim=256)": 1.0 / math.sqrt(256), "1/sqrt(128)": 1.0 / math.sqrt(128),
             "1/sqrt(64)": 1.0 / math.sqrt(64), "1/8": 0.125, "1.0": 1.0, "0.05": 0.05, "0.08": 0.08}
    def err(scale):
        e, n = 0.0, 0
        for g in range(NH):
            for t in (1, 2):
                i = g * NTOK + t
                vis = list(range(t + 1))
                p = softmax(scale, [q[i][j] for j in vis])
                for k, j in enumerate(vis):
                    e = max(e, abs(p[k] - pq[i][j]))
                    n += 1
        return e, n
    for name, sc in sorted(cands.items(), key=lambda kv: err(kv[1])[0]):
        e, n = err(sc)
        print("   %-20s scale=%.6f  max|err| = %.2e  (%d comparisons)" % (name, sc, e, n))

    print("\n== free fit of the scale (bisection on the max-error over the same comparisons) ==")
    lo, hi = 0.01, 0.30
    for _ in range(80):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        if err(m1)[0] <= err(m2)[0]:
            hi = m2
        else:
            lo = m1
    best = (lo + hi) / 2.0
    print("   fitted scale = %.6f  (max|err| %.2e)   1/sqrt(256) = %.6f  ->  delta %.2e"
          % (best, err(best)[0], 1.0 / 16.0, abs(best - 1.0 / 16.0)))

    print("\n== worked example, head group 0, query token 2 ==")
    i = 0 * NTOK + 2
    print("   kq[0..2]  = %s" % q[i][:3])
    print("   official p= %s" % pq[i][:3])
    for sc, nm in ((1.0 / math.sqrt(256), "1/sqrt(256)"), (1.0 / math.sqrt(64), "1/sqrt(64)")):
        print("   our p (%-10s) = %s" % (nm, ["%.4f" % v for v in softmax(sc, q[i][:3])]))


if __name__ == "__main__":
    main()
