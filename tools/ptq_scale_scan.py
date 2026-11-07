#!/usr/bin/env python3
"""ptq_scale_scan.py - independent census of every PTQ1_0 block scale in the checkpoint.

Why a second reader: align_dequant.cpp's `scan` mode first reported 1.1M subnormal and 1.26M NaN
block scales, which is impossible for a trained model - the cause was a bug in ITS OWN loop (a 4 MiB
read chunk is not a multiple of the 28-byte block size, so every chunk after the first started 16
bytes into a block and the "scale" it sampled was two payload bytes). A census is only evidence if
the reader is right, so this file re-derives the same numbers with a completely different method:
strided byte slicing plus a Counter over the at-most-65536 distinct patterns, and Python's own
struct '<e' IEEE half decoder - nothing in here shares code with the C++ f16_to_f32() under test.

  run: python tools/ptq_scale_scan.py [model.gguf]
"""
import os
import struct
import sys
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PATH = sys.argv[1] if len(sys.argv) > 1 else r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf"
TSV = os.path.join(ROOT, "runs", "tensor_table.tsv")
PTQ, BLK = 143, 28


def read_val(f, t):
    """Skip one GGUF value of type t, returning the value only for the scalar types we care about."""
    if t == 0: return struct.unpack("<f", f.read(4))[0]
    if t == 1: return struct.unpack("<B", f.read(1))[0]
    if t == 2: return struct.unpack("<b", f.read(1))[0]
    if t == 3: return struct.unpack("<H", f.read(2))[0]
    if t == 4: return struct.unpack("<h", f.read(2))[0]
    if t == 5: return struct.unpack("<I", f.read(4))[0]
    if t == 6: return struct.unpack("<i", f.read(4))[0]
    if t == 7: return struct.unpack("<Q", f.read(8))[0]
    if t == 8: return struct.unpack("<q", f.read(8))[0]
    if t == 9: return struct.unpack("<d", f.read(8))[0]
    if t == 10:                                       # string
        n = struct.unpack("<Q", f.read(8))[0]
        return f.read(n).decode("utf-8", "replace")
    if t == 11:                                       # array
        it, ct = struct.unpack("<IQ", f.read(12))
        return [read_val(f, it) for _ in range(ct)]
    raise ValueError("unknown GGUF type %d" % t)


# ---- data_start: the measured value, VALIDATED here rather than trusted.
# The file's tensor regions must tile the tail of the file exactly, so
#     max(offset + tensor_bytes) == filesize - data_start
# is an arithmetic proof that the base is right (and that our per-dtype block math is right).
DATA_START = 11_039_616            # measured by tools/gguf_dump.py and by bonsai_gguf.h (agree)
with open(PATH, "rb") as fh:
    magic, ver = struct.unpack("<II", fh.read(8))
    n_tensors, n_kv = struct.unpack("<QQ", fh.read(16))
print(f"[hdr] GGUF v{ver} n_tensors={n_tensors} n_kv={n_kv}")

rows = [l.rstrip("\n").split("\t") for l in open(TSV, encoding="utf-8")][1:]
ptq = [(r[0], int(r[7]), int(r[8])) for r in rows if int(r[6]) == PTQ]   # name, ne, offset
print(f"[tsv] {len(ptq)} PTQ1_0 tensors of {len(rows)}   model={PATH}")

# ---- validate data_start and the block math over ALL tensors, not just PTQ ones
end_max = 0
for r in rows:
    name, dt, ne, off = r[0], int(r[6]), int(r[7]), int(r[8])
    if dt == 0:
        nbytes = ne * 4
    elif dt == PTQ:
        assert ne % 128 == 0, f"{name}: {ne} elems is not a whole number of 128-weight blocks"
        nbytes = ne // 128 * BLK
    else:
        print(f"  [!] unhandled dtype {dt} in {name}"); sys.exit(1)
    end_max = max(end_max, off + nbytes)
file_size = os.path.getsize(PATH)
print(f"[chk] filesize={file_size:,}  data_start={DATA_START:,}  "
      f"tail={file_size - DATA_START:,}  last tensor ends at +{end_max:,}  "
      f"-> {'CONSISTENT' if end_max == file_size - DATA_START else 'MISMATCH'}")
assert end_max == file_size - DATA_START, "data_start or the block math is wrong"

cnt = Counter()
per_tensor_sub = {}
n_blk = n_bytes = 0
CH = 1 << 24                                          # 16 MiB; slicing below is stride-safe
with open(PATH, "rb") as fh:
    for name, ne, off in ptq:
        nb = ne // 128
        if ne % 128:
            print(f"  [!] {name}: {ne} elems is not a whole number of 128-weight blocks")
        fh.seek(DATA_START + off)
        done = 0
        while done < nb:
            take = min(CH // BLK, nb - done)
            buf = fh.read(take * BLK)
            if len(buf) != take * BLK:
                print(f"  [!] short read in {name} at block {done}")
                sys.exit(1)
            # byte 26 and byte 27 of every 28-byte block, as two strided slices, interleaved back
            # into little-endian uint16. Pure C-speed slicing: no Python loop over 69.9M blocks.
            sw = bytearray(2 * take)
            sw[0::2], sw[1::2] = buf[26::BLK], buf[27::BLK]
            chunk_cnt = Counter(struct.unpack(f"<{take}H", bytes(sw)))
            cnt.update(chunk_cnt)                        # C-level merge, not a Python per-block loop
            per_tensor_sub[name] = per_tensor_sub.get(name, 0) + sum(
                k for b, k in chunk_cnt.items()
                if (b & 0x7FFF) != 0 and ((b >> 10) & 0x1F) == 0)
            done += take
            n_blk += take
            n_bytes += take * BLK

print(f"\n==== PTQ1_0 block-scale census over the WHOLE checkpoint (python reader) ====")
print(f"  blocks {n_blk:,}   bytes {n_bytes:,} = {n_bytes/2**30:.3f} GiB   "
      f"(file {os.path.getsize(PATH):,} B)")
print(f"  distinct scale bit patterns: {len(cnt):,} of 65536")

tot = dict(zero_pos=0, zero_neg=0, sub=0, sub_neg=0, neg=0, neg_finite=0, inf=0, inf_neg=0,
           nan=0, nan_neg=0)
exp_hist = Counter()
vals = []
for bits, k in cnt.items():
    mag, sgn = bits & 0x7FFF, bits & 0x8000
    exp, man = (mag >> 10) & 0x1F, mag & 0x3FF
    exp_hist[exp] += k
    if sgn:
        tot["neg"] += k
    if exp == 0:
        if man == 0:
            tot["zero_pos" if not sgn else "zero_neg"] += k
        else:
            tot["sub"] += k
            if sgn: tot["sub_neg"] += k
    elif exp == 31:
        if man:
            tot["nan"] += k
            if sgn: tot["nan_neg"] += k
        else:
            tot["inf"] += k
            if sgn: tot["inf_neg"] += k
    # every real (non-Inf/NaN) scale, decoded by Python's own binary16 converter
    if exp != 31:
        v = struct.unpack("<e", struct.pack("<H", bits))[0]
        if v != 0.0:
            vals.append((abs(v), bits, k))
        if v < 0.0:
            tot["neg_finite"] += k

print(f"  +0 {tot['zero_pos']:,}   -0 {tot['zero_neg']:,}   subnormal {tot['sub']:,} "
      f"(negative {tot['sub_neg']:,})   negative finite {tot['neg_finite']:,}   "
      f"Inf {tot['inf']:,} (-Inf {tot['inf_neg']:,})   NaN {tot['nan']:,} (-NaN {tot['nan_neg']:,})")
vals.sort()
print("  smallest nonzero |d|: " + ", ".join(f"{v:.9g}(0x{b:04X})x{k}" for v, b, k in vals[:6]))
print("  largest  nonzero |d|: " + ", ".join(f"{v:.9g}(0x{b:04X})x{k}" for v, b, k in vals[-6:]))
print("  exponent-field histogram (0 = zero/subnormal, 31 = Inf/NaN):")
for e in range(32):
    if exp_hist[e]:
        print(f"     {e:2d}  {exp_hist[e]:>12,}")
if per_tensor_sub:
    print(f"  the {tot['sub']} subnormal scales live in {len(per_tensor_sub)} tensor(s):")
    for name, k in sorted(per_tensor_sub.items(), key=lambda x: -x[1])[:20]:
        print(f"     {k:>8,}  {name}")
