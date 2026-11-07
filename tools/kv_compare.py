#!/usr/bin/env python3
"""Compare the two KV arena dumps produced by BONSAI_KVCHECK and report what compressing V costs.

The two dumps come from two runs of the SAME binary with the SAME prompt, differing only in
BONSAI_KV_V_QUANT=none|q8_0, so the question this answers is exactly "how much does q8_0-V move the
cache the model reads", with the fp16 V arena as the reference.

Two separate jobs, because they mean different things:

  1. COMPARABILITY. K is stored as fp16 words in BOTH modes (never compressed), so the K rows of the
     two dumps have to be bit-for-bit equal at position 0 - same activations, no cache read has
     influenced them yet. If position 0's K differs, the runs are not comparable and every V number
     below is meaningless. Positions >= 1 are allowed to differ: the compressed V changes the
     activations upstream of them, which is the drift itself, not a bug - so those are reported but
     never used as a gate.

  2. THE ERROR, in two bands reported apart:
       band A  position 0    pure quantisation error (identical inputs in both runs)
       band B  positions >=1 mixed band: quantisation + the drift the compressed cache causes
     Reporting them merged would overstate the first and understate the second.

Usage:
    python tools/kv_compare.py runs/kv_none_v3.tsv runs/kv_q8_0_v3.tsv [--out runs/kv_err_v3.tsv]

Exit code 0 = comparable and the band-A error is inside the printed gate; 1 = not comparable; 2 = bad args.
"""
import argparse
import math
import sys
from collections import defaultdict


def load(path):
    """-> (header dict, {(kind,pos,head,dim): float}, per-position max-abs)"""
    hdr = {}
    vals = {}
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.rstrip("\n")
            if line.startswith("#"):
                # "# layer 0  positions 23  nkv 4  hd 256  v_q8 0  kk 512 kvd 512 kvs 0  per-layer 262144 words"
                # label/value pairs, read by the label: the printf writes "  " between groups and a
                # single space inside one, so positional indexing is fragile and key lookup is not.
                parts = line[1:].split()
                labels = ("layer", "positions", "nkv", "hd", "v_q8", "kk", "kvd", "kvs", "per-layer")
                for i, tok in enumerate(parts):
                    if tok in labels and i + 1 < len(parts):
                        try:
                            hdr[tok.replace("-", "_")] = int(parts[i + 1])
                        except ValueError:
                            pass
                hdr["path"] = path
                continue
            f = line.split()
            if len(f) != 5:
                continue
            kind, p, h, d, v = f[0], int(f[1]), int(f[2]), int(f[3]), float(f[4])
            vals[(kind, p, h, d)] = v
    if not hdr:
        raise SystemExit("%s: no '# layer' header line - not a BONSAI_KVCHECK dump" % path)
    return hdr, vals


def stats(nums):
    if not nums:
        return dict(n=0, max=0.0, mean=0.0, rms=0.0)
    return dict(n=len(nums), max=max(nums), mean=sum(nums) / len(nums),
                rms=math.sqrt(sum(x * x for x in nums) / len(nums)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ref", help="the BONSAI_KV_V_QUANT=none dump (fp16 V, the reference)")
    ap.add_argument("q8", help="the BONSAI_KV_V_QUANT=q8_0 dump (q8_0 V, reconstructed to float)")
    ap.add_argument("--out", help="write the per-position error table here (TSV)")
    ap.add_argument("--gate", type=float, default=0.02,
                      help="band-A max|err| gate as a fraction of band-A max|ref| (default 0.02)")
    a = ap.parse_args()

    h1, v1 = load(a.ref)
    h2, v2 = load(a.q8)
    if h1["positions"] != h2["positions"] or h1["hd"] != h2["hd"] or h1["nkv"] != h2["nkv"]:
        raise SystemExit("the two dumps do not describe the same geometry: %s vs %s"
                         % ({k: h1[k] for k in ("layer", "positions", "nkv", "hd")},
                            {k: h2[k] for k in ("layer", "positions", "nkv", "hd")}))
    npos, nkv, hd, hw = h1["positions"], h1["nkv"], h1["hd"], h1["hd"] // 2

    print("=== KV fidelity: q8_0-V against the fp16-V arena (same binary, same prompt) ===")
    print("  layer %d   positions %d   nkv %d   head_dim %d   kk %d  kvd %d  kvs %d"
          % (h1["layer"], npos, nkv, hd, h1["kk"], h1["kvd"], h1["kvs"]))
    print("  reference : %s (V stored fp16)" % a.ref)
    print("  candidate : %s (V stored q8_0, block 32: 32 int8 + 1 fp16 scale, decoded back to float)"
          % a.q8)

    # ---- 1. comparability: the K arena, which neither mode compresses
    kp = [k for k in v1 if k[0] == "K"]
    if set(k for k in v2 if k[0] == "K") != set(kp):
        print("  K: the two dumps do not cover the same K elements - NOT COMPARABLE")
        return 1
    diff = [(k, v1[k], v2[k]) for k in kp if v1[k] != v2[k]]
    p0 = [d for d in diff if d[0][1] == 0]
    rel = [abs(b - c) for (_, b, c) in diff]
    print("  K arena  : %d elements, %d differ%s" % (len(kp), len(diff),
            "" if diff else "  ->  BITWISE IDENTICAL, the two runs are comparable"))
    if p0:
        print("  K arena  : %d of those differences are at POSITION 0, where the inputs are identical"
              " by construction -> the dumps are NOT comparable (max |dK| = %.6g)"
              % (len(p0), max(abs(b - c) for (_, b, c) in p0)))
        return 1
    if diff:
        print("             all of them are at positions >= 1 (max |dK| %.6g, mean %.6g): that is the"
              " drift the compressed V causes upstream, reported separately below"
              % (max(rel), sum(rel) / len(rel)))

    # ---- 2. the V error, band by band
    vp = [k for k in v1 if k[0] == "V"]
    if not vp:
        print("  the reference dump has no V rows (expected V = fp16)")
        return 1
    missing = [k for k in vp if k not in v2]
    if missing:
        print("  V: %d reference elements have no candidate -> NOT COMPARABLE" % len(missing))
        return 1

    per_pos = defaultdict(lambda: dict(err=[], abs_ref=[], rel=[]))
    worst = dict(max=0.0, key=None)
    for k in vp:
        r, q = v1[k], v2[k]
        p = k[1]
        e = abs(q - r)
        d = per_pos[p]
        d["err"].append(e)
        d["abs_ref"].append(abs(r))
        d["rel"].append(e / abs(r) if abs(r) > 1e-30 else 0.0)
        if e > worst["max"]:
            worst = dict(max=e, key=k)

    band_a, band_b = [], []
    for p in range(npos):
        (band_a if p == 0 else band_b).append(p)

    def agg(idxs, field):
        out = []
        for p in idxs:
            out.extend(per_pos[p][field])
        return out

    def signal_rms(idxs):
        xs = agg(idxs, "abs_ref")
        return math.sqrt(sum(x * x for x in xs) / len(xs)) if xs else 0.0

    a_err, b_err = agg(band_a, "err"), agg(band_b, "err")
    a_rel, b_rel = agg(band_a, "rel"), agg(band_b, "rel")
    sa, sb = stats(a_err), stats(b_err)
    print("\n  band A  position 0        (pure quantisation: identical inputs in both runs)")
    print("          %6d elements   max|err| %.6g   mean %.6g   rms %.6g   max|ref| %.6g"
          % (sa["n"], sa["max"], sa["mean"], sa["rms"], max(agg(band_a, "abs_ref"))))
    print("          signal rms %.6g  ->  err/signal %.3e   (%.2f dB)"
          % (signal_rms(band_a), sa["rms"] / signal_rms(band_a) if signal_rms(band_a) else 0.0,
             20.0 * math.log10(signal_rms(band_a) / sa["rms"]) if sa["rms"] > 0 and signal_rms(band_a) > 0 else float("inf")))
    print("          mean of the per-element relative error %.3e  max %.3e"
          % (sum(a_rel) / len(a_rel), max(a_rel)))
    if band_b:
        print("  band B  positions 1..%d   (quantisation + the drift that quantisation causes upstream)"
              % (npos - 1))
        print("          %6d elements   max|err| %.6g   mean %.6g   rms %.6g"
              % (sb["n"], sb["max"], sb["mean"], sb["rms"]))
        print("          signal rms %.6g  ->  err/signal %.3e"
              % (signal_rms(band_b), sb["rms"] / signal_rms(band_b) if signal_rms(band_b) else 0.0))
    print("          worst single element overall: |err| %.6g at V pos %d head %d dim %d"
          % (worst["max"], worst["key"][1], worst["key"][2], worst["key"][3]) if worst["key"] else "")

    # ---- 3. the scale array, as a sanity check on the block quantiser itself
    sp = [k for k in v2 if k[0] == "S"]
    if sp:
        sv = [v2[k] for k in sp]
        nz = [x for x in sv if x != 0.0]
        print("\n  V scales: %d block scales, %d nonzero, max %.6g, min nonzero %.6g"
              % (len(sv), len(nz), max(sv) if sv else 0.0, min(nz) if nz else 0.0))
        # an all-zero scale block means the block was constant; expected only for padding positions
        if len(nz) < len(sv):
            print("          the %d zero-scale blocks are all-zero V rows (a block whose absmax is 0"
                  " stores d = 0 by ggml's rule)" % (len(sv) - len(nz)))

    gate_abs = a.gate * max(agg(band_a, "abs_ref"))
    verdict = "WITHIN" if sa["max"] <= gate_abs else "ABOVE"
    print("\n  gate: band A max|err| %.6g %s %.0f%% of band A max|ref| (%.6g)"
          % (sa["max"], verdict, 100.0 * a.gate, gate_abs))

    if a.out:
        with open(a.out, "w", encoding="utf-8") as fh:
            fh.write("pos\tn\tmax_abs_err\tmean_abs_err\trms_abs_err\tmax_abs_ref"
                     "\tmean_rel\trms_rel\trms_rel_over_rms_signal\tdrift_only\n")
            for p in range(npos):
                d = per_pos[p]
                s = stats(d["err"])
                sr = math.sqrt(sum(x * x for x in d["abs_ref"]) / len(d["abs_ref"]))
                rr = math.sqrt(sum(x * x for x in d["rel"]) / len(d["rel"]))
                fh.write("%d\t%d\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%.9g\t%s\n"
                         % (p, s["n"], s["max"], s["mean"], s["rms"],
                            max(d["abs_ref"]) if d["abs_ref"] else 0.0,
                            sum(d["rel"]) / len(d["rel"]) if d["rel"] else 0.0, rr,
                            s["rms"] / sr if sr else 0.0,
                            "quantisation only" if p == 0 else "mixed"))
        print("  per-position table written: %s" % a.out)
    return 0 if sa["max"] <= gate_abs else 3


if __name__ == "__main__":
    sys.exit(main())
