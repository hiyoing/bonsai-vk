#!/usr/bin/env python3
"""
order_sensitivity.py - 任务三 forensics: is the残差 ladder an accumulation-ORDER artifact or a
logic bug?

Method (the discriminating experiment is BONSAI_SPLIT=1 vs BONSAI_SPLIT=4, which changes the split-K
accumulation order of every PTQ1_0 GEMV without touching a single line of the shader logic):

    d_order = |ours(SPLIT=1) - ours(SPLIT=4)|     how much the order can move our number
    gap     = |ours(SPLIT=1) - official|          how far our number is from the oracle
    ratio   = gap / d_order                       >> 1  => the order is NOT the cause

Reads two ladder logs, prints per-block and per-anchor tables. Written for
runs/ladder_v6_s1.log and runs/ladder_v6_s4.log.

Line formats it parses (from runtime_vk/bonsai_forward.h's ladder printer):

  block lines:
    "  7       attn      br   -27.88589/-27.99823   b-FAIL  ffn     4.65047/4.67318     f-FAIL  res    40.80972/40.74003    r-FAIL  [ ... ] vs [ ... ] h-FAIL"
  block-3 anchor lines:
    "   QKV total    (=Qcur_full)              sum   -16926.21503 vs   -16964.92969  FAIL   [ ... ]"
"""
import re
import sys

BLOCK_RE = re.compile(
    r"^\s+(\d+)\s+(gdn|attn)\s+"
    r"br\s+(-?[\d.]+)/(-?[\d.]+)\s+b-(OK|FAIL)\s+"
    r"ffn\s+(-?[\d.]+)/(-?[\d.]+)\s+f-(OK|FAIL)\s+"
    r"res\s+(-?[\d.]+)/(-?[\d.]+)\s+r-(OK|FAIL)")

ANCHOR_RE = re.compile(
    r"^\s+([A-Z][A-Za-z0-9 ]*?)\s+\([^)]*\)\s+sum\s+(-?[\d.]+)\s+vs\s+(-?[\d.]+)\s+(OK|FAIL)")

# "  blk  0 delta     0.02288  Sum|x|       172.8  rho  1.32e-04  cancel  0.0212   <- outside the band"
RHO_RE = re.compile(
    r"^\s+blk\s+(\d+)\s+delta\s+(-?[\d.eE+-]+)\s+Sum\|x\|\s+([\d.]+)\s+rho\s+([\d.eE+-]+)"
    r"\s+cancel\s+([\d.]+)(.*)$")


def parse(path):
    blocks, anchors, rho = {}, {}, {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            m = BLOCK_RE.match(line)
            if m:
                i = int(m.group(1))
                g = m.groups()
                blocks[i] = {"type": g[1],
                             "br": (float(g[2]), float(g[3]), g[4]),
                             "ffn": (float(g[5]), float(g[6]), g[7]),
                             "res": (float(g[8]), float(g[9]), g[10])}
                continue
            m = ANCHOR_RE.match(line)
            if m:
                name = " ".join(m.group(1).split())
                anchors[name] = (float(m.group(2)), float(m.group(3)), m.group(4))
                continue
            m = RHO_RE.match(line)
            if m:
                rho[int(m.group(1))] = float(m.group(3))
    return blocks, anchors, rho


def col(v):
    return f"{v:>16.6f}"


def table(blocks_a, blocks_b, which, rho):
    print(f"\n=== per-block {which}: order sensitivity vs oracle gap "
          f"(normalized by each block's own Sum|x|) ===")
    print(f"  {'blk':>3} {'ty':<5} {'ours s=1':>14} {'ours s=4':>14} {'official':>14}"
          f" {'d_order':>11} {'gap':>11} {'gap/d_order':>11} {'d_order/S':>11} {'gap/S':>11}"
          f"  verdict")
    for i in sorted(set(blocks_a) & set(blocks_b)):
        a_o, off, _ = blocks_a[i][which]
        b_o = blocks_b[i][which][0]
        d_order = abs(a_o - b_o)
        gap = abs(a_o - off)
        ratio = gap / d_order if d_order > 0 else float("inf")
        s = rho.get(i, 0.0)
        ds = f"{d_order / s:.2e}" if s else "n/a"
        gs = f"{gap / s:.2e}" if s else "n/a"
        verdict = "order-dominated" if ratio < 3 else "NOT order"
        print(f"  {i:>3} {blocks_a[i]['type']:<5} {a_o:>14.5f} {b_o:>14.5f} {off:>14.5f}"
              f" {d_order:>11.2e} {gap:>11.2e} {ratio:>11.1f} {ds:>11} {gs:>11}  {verdict}")


def main():
    p1, p4 = sys.argv[1], sys.argv[2]
    b1, an1, rho1 = parse(p1)
    b4, an4, rho4 = parse(p4)
    print(f"parsed {p1}: {len(b1)} blocks, {len(an1)} anchors, {len(rho1)} Sum|x| rows")
    print(f"parsed {p4}: {len(b4)} blocks, {len(an4)} anchors, {len(rho4)} Sum|x| rows")
    if not b1 or not an1:
        print("PARSE FAILURE - nothing matched. Check the log format against BLOCK_RE/ANCHOR_RE.")
        return 1

    for which in ("br", "ffn", "res"):
        table(b1, b4, which, rho1)

    print("\n=== block-3 internal anchors: order sensitivity vs oracle gap ===")
    print(f"  {'anchor':<22} {'ours s=1':>14} {'ours s=4':>14} {'official':>14}"
          f" {'d_order':>11} {'gap':>11} {'gap/d_order':>11}")
    for name in an1:
        if name not in an4:
            continue
        a_o, off, v = an1[name]
        b_o = an4[name][0]
        d_order = abs(a_o - b_o)
        gap = abs(a_o - off)
        ratio = gap / d_order if d_order > 0 else float("inf")
        print(f"  {name:<22} {a_o:>14.5f} {b_o:>14.5f} {off:>14.5f}"
              f" {d_order:>11.2e} {gap:>11.2e} {ratio:>11.1f}")

    # ---- the one-number summary the report quotes, in the cancellation-aware units the project
    # already uses: |x| / Sum|x|, i.e. the share of the block's own magnitude scale.
    rel_order, rel_gap, min_ratio = [], [], float("inf")
    for i in sorted(set(b1) & set(b4)):
        s = rho1.get(i, 0.0)
        if not s:
            continue
        for which in ("br", "ffn", "res"):
            a_o, off, _ = b1[i][which]
            b_o = b4[i][which][0]
            d_order = abs(a_o - b_o)
            gap = abs(a_o - off)
            if d_order > 0:
                min_ratio = min(min_ratio, gap / d_order)
            rel_order.append(d_order / s)
            rel_gap.append(gap / s)
    print("\n=== summary (S = the block's own Sum|x|, from the log's rho table) ===")
    print(f"  blocks with a Sum|x| row         : {len(rho1)}")
    print(f"  d_order/S   max / median         : {max(rel_order):.3e} / "
          f"{sorted(rel_order)[len(rel_order)//2]:.3e}")
    print(f"  gap-to-S    max / median         : {max(rel_gap):.3e} / "
          f"{sorted(rel_gap)[len(rel_gap)//2]:.3e}")
    print(f"  smallest per-row gap/d_order     : {min_ratio:.1f}x")
    print(f"  anchors: QKV d_order             : "
          f"{abs(an1['QKV total'][0] - an4['QKV total'][0]):.2e}  "
          f"gap {abs(an1['QKV total'][0] - an1['QKV total'][1]):.2e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
