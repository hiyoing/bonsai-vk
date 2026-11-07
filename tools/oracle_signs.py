#!/usr/bin/env python3
"""oracle_signs.py - recover the incoherence sign arrays from the GGUF and FINGERPRINT them
against the official engine's own per-node dump (runs/dbg_full_real.log).

Why this file exists
--------------------
The PrismML fork's graph multiplies activations by `prism.hadamard.signs.4096` /
`.signs.12288`. Those are NOT tensors in the GGUF tensor table (runs/tensor_table.tsv has 427
rows, none named prism.*): they live in the *metadata* as one flat i32 array
`prism.hadamard.sign_values` of 16384 entries, split by `prism.hadamard.sign_widths = [4096,
12288]`. Which half comes first, and what the values are, is decided here by reading the file and
then cross-checking against numbers the official engine printed.

Cross-check method (no GPU, no model run)
-----------------------------------------
A MUL(x, signs) node prints x and y = x * signs with 3 head + 3 tail values at 4 decimals.
The per-position ratio y/x IS the sign. Three independent fingerprints are used:

  L18 node_2 (Hadamard out)  vs  L34 model.input_embed (= that * signs.4096)   -> signs4096[0..2]
  L50 attn_norm-0            vs  L131 node_17 (= attn_norm-0 * signs.4096)     -> signs4096[0..2]
                                                                                + signs4096[4093..4095]
  L1074 ffn_swiglu-0         vs  L1082 node_83 (= that * signs.12288)          -> signs12288[0..2]
                                                                                + signs12288[12285..12287]
  L1090 RESHAPE {12288,2}->{4096,6} gives the flat addresses of the 12288 array: column c of the
  reshaped tensor is flat [c*4096, c*4096+4096) -> it also tells us the chunking convention.

Outputs: runs/signs_4096.txt, runs/signs_12288.txt (one +-1 per line), plus a printed report.
"""
import array
import os
import re
import struct
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL = os.environ.get("GGUF", r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf")
RUNS = os.path.join(ROOT, "runs")
LOG = os.path.join(RUNS, "dbg_full_real.log")

# ---------------------------------------------------------------- GGUF v3 metadata reader
# magic "GGUF", u32 ver, u64 n_tensors, u64 n_kv, then n_kv records of
# (u64 keylen, key, u32 type, value); type 5 = i32, type 9 = array{u32 elem_type, u64 count, elems}
# Tensor dtypes use GGML ids (143 = PTQ1_0), metadata types use GGUF ids. Do not mix them up.
class Reader:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        assert len(b) == n, ("short read", n)
        return b

    def u32(self):  return struct.unpack("<I", self.raw(4))[0]
    def i32(self):  return struct.unpack("<i", self.raw(4))[0]
    def u64(self):  return struct.unpack("<Q", self.raw(8))[0]
    def i64(self):  return struct.unpack("<q", self.raw(8))[0]
    def f32(self):  return struct.unpack("<f", self.raw(4))[0]
    def f64(self):  return struct.unpack("<d", self.raw(8))[0]
    def u8(self):   return self.raw(1)[0]
    def string(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

    def value(self, t):
        # scalar GGUF types: 0 u8 1 i8 2 u16 3 i16 4 u32 5 i32 6 f32 7 bool 8 str 10 u64 11 i64 12 f64
        if t == 0: return self.u8()
        if t == 1: return struct.unpack("<b", self.raw(1))[0]
        if t == 2: return struct.unpack("<H", self.raw(2))[0]
        if t == 3: return struct.unpack("<h", self.raw(2))[0]
        if t == 4: return self.u32()
        if t == 5: return self.i32()
        if t == 6: return self.f32()
        if t == 7: return bool(self.u8())
        if t == 8: return self.string()
        if t == 10: return self.u64()
        if t == 11: return self.i64()
        if t == 12: return self.f64()
        raise KeyError("metadata type %d" % t)


def read_gguf_metadata(path):
    """Returns (kv dict name->value, tensor list, header_end, data_start)."""
    kv = {}
    with open(path, "rb") as f:
        assert f.read(4) == b"GGUF", "not GGUF"
        r = Reader(f)
        ver = r.u32()
        n_tensors = r.u64()
        n_kv = r.u64()
        for _ in range(n_kv):
            key = r.string()
            t = r.u32()
            if t == 9:                                     # ARRAY
                et = r.u32()
                n = r.u64()
                if et == 9:
                    raise KeyError("nested array at %s - not present in this model" % key)
                # read the whole array as one packed buffer when the element is fixed size,
                # otherwise element by element (strings).
                if et == 8:
                    kv[key] = [r.string() for _ in range(n)]
                else:
                    buf = array.array("B", r.raw(_size(et) * n))
                    kv[key] = _unpack(et, buf)
            else:
                kv[key] = r.value(t)
        tensors = []
        for _ in range(n_tensors):
            name = r.string()
            nd = r.u32()
            dims = [r.u64() for _ in range(nd)]
            dtype = r.u32()
            off = r.u64()
            tensors.append((name, dims, dtype, off))
        header_end = f.tell()
    align = kv.get("general.alignment", 32) or 32
    data_start = (header_end + align - 1) // align * align
    return kv, tensors, header_end, data_start, ver


def _size(t):
    return {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}[t]


def _unpack(t, buf):
    fmt = {0: "B", 1: "b", 2: "H", 3: "h", 4: "I", 5: "i", 6: "f", 7: "?",
           10: "Q", 11: "q", 12: "d"}[t]
    a = array.array(fmt)
    a.frombytes(buf.tobytes())
    return list(a)


# ---------------------------------------------------------------- dump printer parser
NUM = r"[-+]?\d+\.\d+(?:e[-+]?\d+)?"


def parse_node_block(log_lines, start_line):
    """Parse one debug-callback node starting at 1-based line `start_line`.

    Returns dict with: text (the op line), out_shape (tuple), rows (list of (head, tail) float
    lists), sum (float or None).  A printed row looks like
        [      0.0062,      -0.0066,       0.0471,    ...,       0.0113,       0.0033,      -0.0150  ],
    rows that are elided show up as a line containing only '..., '.
    """
    text = log_lines[start_line - 1]
    m = re.search(r"=\s*\((\w+)\)\s*([A-Z_0-9]+)\((.*)\)\s*=\s*\{([0-9,\s]+)\}", text)
    out_shape = tuple(int(x) for x in m.group(4).replace(" ", "").split(",")) if m else None
    rows, sums = [], None
    i = start_line
    depth_seen = False
    while i < len(log_lines):
        line = log_lines[i]
        if re.match(r"^\s*sum = ", line):
            sums = float(re.search(NUM, line).group(0))
            break
        stripped = line.strip()
        if stripped.startswith("[") and re.search(NUM, stripped):
            dots = stripped.find("...")
            head_txt = stripped if dots < 0 else stripped[:dots]
            tail_txt = "" if dots < 0 else stripped[dots + 3:]
            head = [float(x) for x in re.findall(NUM, head_txt)]
            tail = [float(x) for x in re.findall(NUM, tail_txt)]
            rows.append((head, tail))
        i += 1
    return {"text": text.strip(), "shape": out_shape, "rows": rows, "sum": sums}


def signs_from_pair(a, b, n):
    """a = printed x rows, b = printed y=x*s rows -> dict position -> +-1 (None when unusable)."""
    out = {}
    for (head_a, tail_a), (head_b, tail_b) in zip(a["rows"], b["rows"]):
        for off, xa, xb in [(0, head_a, head_b), (n - 3, tail_a, tail_b)]:
            for k, (va, vb) in enumerate(zip(xa, xb)):
                pos = off + k
                if abs(va) < 1e-3:
                    out[pos] = None                  # ratio unstable at 4 decimals
                    continue
                r = vb / va
                out[pos] = 1 if r > 0 else -1
    return out


def main():
    kv, tensors, hend, dstart, ver = read_gguf_metadata(MODEL)
    print("GGUF version %d, %d tensors, metadata ends at %d, data at %d" % (ver, len(tensors), hend, dstart))
    for key in ("prism.hadamard.transform", "prism.hadamard.block_size", "prism.hadamard.axis",
                "prism.hadamard.sign_mode", "prism.hadamard.version", "prism.hadamard.gdn_v_grouped"):
        print("  %-34s = %s" % (key, kv.get(key)))
    widths = kv["prism.hadamard.sign_widths"]
    values = kv["prism.hadamard.sign_values"]
    print("  sign_widths = %s ; sign_values n = %d ; value set = %s"
          % (widths, len(values), sorted(set(values))))
    assert sum(widths) == len(values), (widths, len(values))
    for v in values:
        assert v in (-1, 1), v

    # split, widths order as written in the file
    parts, off = [], 0
    for w in widths:
        parts.append(values[off:off + w])
        off += w
    signs_4096, signs_12288 = parts[0], parts[1]
    os.makedirs(RUNS, exist_ok=True)
    for name, arr in (("signs_4096.txt", signs_4096), ("signs_12288.txt", signs_12288)):
        with open(os.path.join(RUNS, name), "w") as o:
            o.write("\n".join(str(x) for x in arr) + "\n")
        print("wrote runs/%s (%d entries, sum=%d)" % (name, len(arr), sum(arr)))

    # ---------------------------------------------------------------- fingerprint vs the dump
    with open(LOG, "r", encoding="utf-8", errors="replace") as f:
        lines = f.read().split("\n")

    node2 = parse_node_block(lines, 18)          # MUL_MAT(hadamard, emb)
    iembed = parse_node_block(lines, 34)         # MUL(node2_reshaped, signs.4096)
    anorm = parse_node_block(lines, 50)          # MUL(RMS_NORM(...), attn_norm.weight)
    node17 = parse_node_block(lines, 131)        # MUL(attn_norm-0, signs.4096)
    swiglu = parse_node_block(lines, 1074)       # SWIGLU(ffn_gate, ffn_up)  {12288,2}
    node83 = parse_node_block(lines, 1082)       # MUL(ffn_swiglu-0, signs.12288)

    fp4 = {}
    fp4.update(signs_from_pair(node2, iembed, 4096))
    fp4.update(signs_from_pair(anorm, node17, 4096))
    fp12 = signs_from_pair(swiglu, node83, 12288)

    print("\n== fingerprint: signs.4096 recovered from the dump vs the file ==")
    ok4 = bad4 = 0
    for pos in sorted(fp4):
        got = fp4[pos]
        if got is None:
            print("  pos %5d : dump unusable (|x|<1e-3), file says %d" % (pos, signs_4096[pos]))
            continue
        want = signs_4096[pos]
        agree = got == want
        ok4 += agree
        bad4 += not agree
        print("  pos %5d : dump %+d  file %+d  %s" % (pos, got, want, "MATCH" if agree else "*** MISMATCH ***"))

    print("\n== fingerprint: signs.12288 recovered from the dump vs the file ==")
    ok12 = bad12 = 0
    for pos in sorted(fp12):
        got = fp12[pos]
        want = signs_12288[pos]
        if got is None:
            print("  pos %5d : dump unusable, file says %+d" % (pos, want))
            continue
        agree = got == want
        ok12 += agree
        bad12 += not agree
        print("  pos %5d : dump %+d  file %+d  %s" % (pos, got, want, "MATCH" if agree else "*** MISMATCH ***"))

    print("\n== split-order test ==")
    # the alternative reading would be [12288 first, 4096 second]; check its head against the dump
    alt_12288 = values[:12288]
    alt_4096 = values[12288:16384]
    for pos in sorted(fp4):
        if fp4[pos] is None:
            continue
        print("  signs.4096[%d]: dump %+d ; widths order (4096 first) %+d ; 12288-first reading %+d"
              % (pos, fp4[pos], signs_4096[pos], alt_4096[pos]))
        break
    for pos in sorted(fp12):
        if fp12[pos] is None:
            continue
        print("  signs.12288[%d]: dump %+d ; widths order (12288 second) %+d ; 12288-first reading %+d"
              % (pos, fp12[pos], signs_12288[pos], alt_12288[pos]))
        break

    print("\nRESULT: signs.4096 matches %d/%d printed positions (%d mismatch), "
          "signs.12288 matches %d/%d (%d mismatch)"
          % (ok4, ok4 + bad4, bad4, ok12, ok12 + bad12, bad12))
    print("signs_4096[0..2] = %s   (task predicted -1,-1,-1)" % signs_4096[0:3])
    print("signs_4096[4093..4095] = %s" % signs_4096[-3:])
    print("signs_12288[0..2] = %s ; signs_12288[12285..12287] = %s" % (signs_12288[0:3], signs_12288[-3:]))
    return 0 if (bad4 == 0 and bad12 == 0) else 1


if __name__ == "__main__":
    sys.exit(main())
