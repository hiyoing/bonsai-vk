#!/usr/bin/env python3
"""Self-consistency of the ORACLE's own printed sums.

The ladder compares our per-block sums against runs/dbg_full_real.log's `sum = ...` lines. Before
blaming anybody's arithmetic, measure the reference side: the graph says
      l_out-N        = ADD(ffn_out-N, attn_residual-N)
so the PRINTED sums must satisfy sum(l_out-N) = sum(ffn_out-N) + sum(attn_residual-N) to the
precision of whatever accumulator the printer uses. Any gap is the oracle's own summation error,
and that gap is the floor under every rho/delta in the ladder.

Read-only over the log. Prints a table plus the statistics that decide the question.
"""
import re, sys

LOG = sys.argv[1] if len(sys.argv) > 1 else "runs/dbg_full_real.log"
START = int(sys.argv[2]) if len(sys.argv) > 2 else 32000     # the cold 1-token decode pass

node_re = re.compile(r"common_debug_cb_eval:\s+([A-Za-z0-9_\-.]+) = \(f32\)\s+([A-Z_]+)\(")
sum_re = re.compile(r"^\s*sum = (-?[\d.]+)")

# name -> list of (op, sum) in evaluation order; the same node can be evaluated more than once.
seen = {}
cur = None
with open(LOG, "r", errors="replace") as f:
    for ln, line in enumerate(f, 1):
        if ln < START:
            continue
        m = node_re.search(line)
        if m:
            cur = (m.group(1), m.group(2))
            continue
        s = sum_re.match(line)
        if s and cur:
            seen.setdefault(cur[0], []).append((cur[1], float(s.group(1))))
            cur = None

def last(name, occ):
    v = seen.get(name)
    if not v or occ >= len(v):
        return None
    return v[occ]

print("occurrences of l_out-0 in the window: %d" % len(seen.get("l_out-0", [])))
print("\n=== the reference's own ADD identity: sum(l_out-N) vs sum(ffn_out-N)+sum(attn_residual-N)")
print("  N   ours(ladder table)      gap = printed_sum - (a + b)   |gap|/Sum|x|-proxy")
worst = 0.0
import math
gaps = []
for occ in range(len(seen.get("l_out-0", []))):
    tot = 0.0
    rows = 0
    for n in range(32):
        lo = last("l_out-%d" % n, occ)
        ff = last("ffn_out-%d" % n, occ)
        ar = last("attn_residual-%d" % n, occ)
        if not (lo and ff and ar):
            continue
        gap = lo[1] - (ff[1] + ar[1])
        gaps.append((n, occ, lo[1], ff[1], ar[1], gap))
        tot += abs(gap)
        worst = max(worst, abs(gap))
        rows += 1
    if rows:
        print("  pass %d: %2d/32 blocks, mean |gap| = %.5f, max |gap| = %.5f"
              % (occ, rows, tot / rows, worst))

print("\n=== every block, pass 0 ===")
print("   blk   l_out        ffn_out       attn_res      gap        |gap|/|l_out|")
for (n, occ, lo, ff, ar, gap) in gaps:
    if occ != 0:
        continue
    rel = abs(gap) / abs(lo) if lo else float("inf")
    print("   %2d  %11.6f %12.6f %12.6f  %+10.6f  %.2e" % (n, lo, ff, ar, gap, rel))

g = [abs(x[5]) for x in gaps if x[1] == 0]
if g:
    mean = sum(g) / len(g)
    print("\n  pass-0 gap: mean %.6f  median %.6f  max %.6f over %d blocks" %
          (mean, sorted(g)[len(g) // 2], max(g), len(g)))
    print("  compare with the ladder's own per-block delta (0.02..1.25): if the two are the same")
    print("  order, the discrepancy lives in how a 4096-term sum is accumulated, not in the engine.")
