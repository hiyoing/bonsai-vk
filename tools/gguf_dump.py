#!/usr/bin/env python3
"""gguf_dump.py - our own GGUF reader, used to produce the ground truth the C++ engine is built on.

Emits three files next to runs/:
  runs/tensor_table.tsv  name \t n_dims \t d0..d3 \t ggml_type \t elems \t row_bytes \t total_bytes \t offset
  runs/tok_meta.txt      tokenizer ground truth: pre-config, byte tokens, sample merges, specials
  runs/layer_shapes.txt  per-tensor shapes for block.0 / block.1 / norms / head, i.e. what the
                         forward pass actually has to multiply

GGUF type ids are GGUF's own (0..12, ARRAY=9), NOT ggml's. Confirmed by the fact that reading
ARRAY as 8 raised KeyError on the first string key.
"""
import struct, sys, os, json

PATH = sys.argv[1] if len(sys.argv) > 1 else r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs")
os.makedirs(OUT, exist_ok=True)

# GGUF type id -> (name, size_t, elems_per_block, bytes_per_block)
TYPES = {
    0: ("f32", 4, 1, 4), 1: ("f16", 2, 1, 2), 2: ("bf16", 2, 1, 2),
    8: ("q8_0", 1, 32, 34), 10: ("q4_0", 1, 32, 18), 12: ("q5_0", 1, 32, 22),
}
# PTQ1_0 is a PrismML private type. general.file_type=143; the tensor-level id must be discovered.
PTQ_BLOCK_BYTES = 28   # 128 trits + f16 scale -> 1.75 bpw
PTQ_BLOCK_W = 7


class R:
    def __init__(self, f):
        self.f = f

    def raw(self, n):
        b = self.f.read(n)
        assert len(b) == n, (n, len(b))
        return b

    def u8(self):   return struct.unpack("<B", self.raw(1))[0]
    def i32(self):  return struct.unpack("<i", self.raw(4))[0]
    def u32(self):  return struct.unpack("<I", self.raw(4))[0]
    def i64(self):  return struct.unpack("<q", self.raw(8))[0]
    def f32(self):  return struct.unpack("<f", self.raw(4))[0]
    def s(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")

    def u64(self):  return struct.unpack("<Q", self.raw(8))[0]

    def arr(self):
        t = self.u32()
        n = self.u64()
        return t, n, [self.one(t) for _ in range(n)] if n <= 400000 else None

    def one(self, t):
        # GGUF's own type ids - NOT ggml's. 0 U8 1 I8 2 U16 3 I16 4 U32 5 I32 6 F32 7 BOOL
        # 8 STRING 9 ARRAY 10 U64 11 I64 12 F64
        if t == 0: return self.u8()
        if t == 1: return self.i8() & 0xFF
        if t == 2: return self.u16()
        if t == 3: return self.i16()
        if t == 4: return self.u32()
        if t == 5: return self.i32()
        if t == 6: return self.f32()
        if t == 7: return bool(self.u8())
        if t == 8: return self.s()
        if t == 9:
            et, n, vals = self.arr()
            return vals
        if t == 10: return self.u64()
        if t == 11: return self.i64()
        if t == 12: return self.f64()
        raise KeyError(t)

    def i8(self):  return struct.unpack("<b", self.raw(1))[0]
    def u16(self): return struct.unpack("<H", self.raw(2))[0]
    def i16(self): return struct.unpack("<h", self.raw(2))[0]
    def f64(self): return struct.unpack("<d", self.raw(8))[0]


def main():
    f = open(PATH, "rb")
    r = R(f)
    assert r.raw(4) == b"GGUF", "not GGUF"
    ver = r.u32()
    n_tensors = r.u64()
    n_kv = r.u64()
    md = {}
    for _ in range(n_kv):
        k = r.s()
        t = r.u32()
        if t == 9:                                   # ARRAY
            et, n, vals = r.arr()
            md[k] = ("<array %d elems, elem-type %d>" % (n, et), et, n, vals)
        else:
            md[k] = (r.one(t), t, 1, None)
    header_end = f.tell()

    infos = []
    for _ in range(n_tensors):
        name = r.s()
        nd = r.u32()
        dims = [r.u64() for _ in range(nd)]
        dt = r.u32()
        ne = 1
        for d in dims:
            ne *= d
        off = r.u64()
        infos.append(dict(name=name, nd=nd, dims=dims, dtype=dt, ne=ne, off=off))
    data_start = align(r.u64(), 32) if False else f.tell()
    # alignment: GGUF puts an i64 alignment value here
    # re-read: after tensor infos comes padding to `general.alignment` (default 32)
    f.seek(f.tell())
    al = md.get("general.alignment", (32, 4, 1, None))[0]
    data_start = (f.tell() + al - 1) // al * al

    # ---- infer the private dtype id + per-element bytes from PTQ block math
    by_type = {}
    for i in infos:
        by_type.setdefault(i["dtype"], []).append(i)
    lines = []
    lines.append("GGUF v%d  n_tensors=%d  n_kv=%d  header_end=%d  alignment=%d  data_start=%d"
                 % (ver, n_tensors, n_kv, header_end, al, data_start))
    lines.append("")
    lines.append("=== dtype census (which tensor-level ggml type ids does this file actually use?) ===")
    for dt, lst in sorted(by_type.items()):
        # smallest tensor of this dtype tells us the block math with almost no ambiguity
        lst2 = sorted(lst, key=lambda x: x["ne"])
        c = lst2[0]
        nbytes = None
        if len(lst2) > 1:
            nbytes = lst2[1]["off"] - c["off"]
        lines.append("  type %-4d count %-5d smallest=%dx%d next-tensor-gap=%s elems/gap=%.4f"
                     % (dt, len(lst), c["dims"][0], c["dims"][1] if len(c["dims"]) > 1 else 0,
                        nbytes, (c["ne"] / nbytes) if nbytes else float("nan")))
        lines.append("      smallest tensor: %s" % c["name"])
    # ---- tensor table
    with open(os.path.join(OUT, "tensor_table.tsv"), "w", encoding="utf-8") as o:
        o.write("name\tn_dims\td0\td1\td2\td3\tdtype\tne\toffset\n")
        for i in infos:
            d = i["dims"] + [0] * (4 - len(i["dims"]))
            o.write("%s\t%d\t%d\t%d\t%d\t%d\t%d\t%d\t%d\n" %
                    (i["name"], i["nd"], d[0], d[1], d[2], d[3], i["dtype"], i["ne"], i["off"]))

    # ---- per-layer shapes
    def uniq_names(suffix_filter):
        s = {}
        for i in infos:
            if i["name"].startswith(suffix_filter):
                s[i["name"]] = i
        return s
    with open(os.path.join(OUT, "layer_shapes.txt"), "w", encoding="utf-8") as o:
        o.write("=== non-block tensors (embeddings, norms, head, mtp) ===\n")
        for i in infos:
            if ".blocks." not in i["name"]:
                o.write("%-46s dims=%-24s dtype=%d\n" % (i["name"], "x".join(str(d) for d in i["dims"]), i["dtype"]))
        for b in (0, 1, 2, 3, 4):
            o.write("\n=== block.%d ===\n" % b)
            for i in infos:
                if (".blocks.%d." % b) in i["name"]:
                    o.write("%-52s dims=%-20s dtype=%-3s off=%d\n" %
                            (i["name"], "x".join(str(d) for d in i["dims"]), i["dtype"], i["off"]))
        # name template census: strip block index
        pats = {}
        for i in infos:
            p = i["name"]
            import re
            p = re.sub(r"\.blocks\.\d+\.", ".blocks.N.", p)
            pats.setdefault(p, []).append(i["dtype"])
        o.write("\n=== name patterns (%d distinct) ===\n" % len(pats))
        for p in sorted(pats):
            o.write("%-60s dtype(s)=%s\n" % (p, sorted(set(pats[p]))))

    # ---- tokenizer ground truth
    with open(os.path.join(OUT, "tok_meta.txt"), "w", encoding="utf-8") as o:
        for k in sorted(md):
            if k.startswith("tokenizer") or k.startswith("general") or k.startswith("vocab"):
                v = md[k]
                if isinstance(v[0], str) and len(v[0]) > 60:
                    o.write("%-42s = <%s chars>\n" % (k, len(v[0])))
                else:
                    o.write("%-42s = %r\n" % (k, v[0] if not isinstance(v[0], tuple) else v[0]))
        toks = md["tokenizer.ggml.tokens"][3]
        tt = md["tokenizer.ggml.token_type"][3]
        merges = md["tokenizer.ggml.merges"][3]
        pre = md.get("tokenizer.ggml.pre")
        o.write("\nvocab = %d   token_type = %s   merges = %s\n" %
                (len(toks), "None" if tt is None else len(tt), "None" if merges is None else len(merges)))
        if pre:
            o.write("pre = %r\n" % (pre[3],))
        o.write("\n--- first 20 tokens (repr, utf-8 decoded as stored) ---\n")
        for i in range(20):
            o.write("  %6d %r type=%s\n" % (i, toks[i], tt[i] if tt else "?"))
        o.write("\n--- tokens 151600..151700 (the qwen special neighbourhood) ---\n")
        for i in range(151600, min(151700, len(toks))):
            o.write("  %6d %r type=%s\n" % (i, toks[i], tt[i] if tt else "?"))
        o.write("\n--- last 300 tokens (byte-fallback <0xNN>?) ---\n")
        for i in range(max(0, len(toks) - 300), len(toks)):
            o.write("  %6d %r type=%s\n" % (i, toks[i], tt[i] if tt else "?"))
        o.write("\n--- token_type census ---\n")
        if tt:
            c = {}
            for t in tt:
                c[t] = c.get(t, 0) + 1
            o.write("  %r\n" % c)
        o.write("\n--- merges: first 10, last 10 ---\n")
        if merges:
            for m in merges[:10]:
                o.write("  %r\n" % m)
            o.write("  ...\n")
            for m in merges[-10:]:
                o.write("  %r\n" % m)
        o.write("\n--- chat_template (whole thing) ---\n")
        o.write(md["tokenizer.chat_template"][0] if md.get("tokenizer.chat_template") else "none")
        # every token containing '<' or a non-printable
        o.write("\n--- control-type tokens (token_type==3) ---\n")
        if tt:
            for i, t in enumerate(tt):
                if t == 3:
                    o.write("  %6d %r\n" % (i, toks[i]))
        o.write("\n--- tokens that are pure non-ascii sample (10) ---\n")
        n = 0
        for i, t in enumerate(toks):
            if any(ord(ch) > 127 for ch in t) and "<0x" not in t:
                o.write("  %6d %r\n" % (i, t))
                n += 1
                if n >= 10:
                    break
    # ---- chat template saved verbatim for the C++ side to compare against
    if md.get("tokenizer.chat_template"):
        with open(os.path.join(OUT, "chat_template.txt"), "w", encoding="utf-8") as o:
            o.write(md["tokenizer.chat_template"][0])

    print("wrote tensor_table.tsv / layer_shapes.txt / tok_meta.txt / chat_template.txt to", OUT)
    print(lines[0])
    for l in lines[2:]:
        print(l)


if __name__ == "__main__":
    main()
