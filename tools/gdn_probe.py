#!/usr/bin/env python3
"""
gdn_probe.py -- extraction helper for the Gated DeltaNet forward spec.

CPU only. Reads:
  runs/dbg_full_real.log   (per-node dump of the official engine)  -> 实测 source
  the GGUF metadata + f32 tensors of blk.0                        -> 实测 source

Writes:
  runs/gdn_nodes_b0.tsv   ordered node list of block 0 (line, node, op, operands, shape, sum)
  runs/gdn_sums.tsv       all "node -> sum" pairs of the whole log (oracle for any later diff)
  runs/gdn_blk0_consts.txt  the f32 constants of blk.0 (ssm_a, ssm_dt.bias, ssm_norm, conv1d)

Usage:
  PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe tools/gdn_probe.py [--gguf PATH]
"""
import os
import re
import struct
import sys
import json

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG = os.path.join(ROOT, "runs", "dbg_full_real.log")
DEFAULT_GGUF = r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf"

NODE_RE = re.compile(r"^common_debug_cb_eval:\s+(.*?)\s*=\s*\((\w+)\)\s+(\w+)\((.*)\)\s*=\s*(\{.*\})\s*$")
SUM_RE = re.compile(r"^\s+sum\s*=\s*(-?[\d.]+)")


# ---------------------------------------------------------------- log parsing
def parse_log(path=LOG):
    """Return list of dicts: line, name, dtype, op, args, shape, sum."""
    out = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        pending = None
        for i, raw in enumerate(f, 1):
            m = NODE_RE.match(raw.rstrip("\n"))
            if m:
                pending = {"line": i, "name": m.group(1).strip(), "dtype": m.group(2),
                           "op": m.group(3), "args": m.group(4), "shape": m.group(5), "sum": None}
                out.append(pending)
                continue
            s = SUM_RE.match(raw)
            if s and pending is not None and pending["sum"] is None:
                pending["sum"] = float(s.group(1))
    return out


def shape_tuple(sh):
    return tuple(int(x) for x in re.findall(r"\d+", sh))


def main_log(nodes):
    # block 0 = first occurrence of each "-0" suffix node + the unnumbered prologue
    b0 = [n for n in nodes if n["name"].endswith("-0") or n["name"] in
          ("node_0", "(reshaped)", "node_2", "model.input_embed")]
    with open(os.path.join(ROOT, "runs", "gdn_nodes_b0.tsv"), "w", encoding="utf-8") as f:
        f.write("line\tname\top\tshape\tsum\targs\n")
        for n in b0:
            f.write(f"{n['line']}\t{n['name']}\t{n['op']}\t{n['shape']}\t{n['sum']}\t{n['args']}\n")

    with open(os.path.join(ROOT, "runs", "gdn_sums.tsv"), "w", encoding="utf-8") as f:
        f.write("line\tname\top\tshape\tsum\n")
        for n in nodes:
            f.write(f"{n['line']}\t{n['name']}\t{n['op']}\t{n['shape']}\t{n['sum']}\n")

    print(f"[gdn_probe] parsed {len(nodes)} nodes, block-0 region -> {len(b0)} nodes")
    return nodes


# ------------------------------------------------------------------ gguf part
GGUF_MAGIC = b"GGUF"
T_UINT8, T_INT8, T_UINT16, T_INT16, T_UINT32, T_INT32, T_FLOAT32, T_BOOL = 1, 2, 3, 4, 5, 6, 7, 8
T_STRING, T_ARRAY, T_UINT64, T_INT64, T_FLOAT64, T_FLOAT16, T_BFLOAT16 = 9, 10, 11, 12, 13, 14, 15

TYPE_SIZE = {0: 2, 1: 1, 2: 1, 3: 2, 4: 2, 5: 4, 6: 4, 7: 4, 8: 1, 9: 0, 10: 0,
             11: 8, 12: 8, 13: 8, 14: 2, 15: 2}


class Reader:
    def __init__(self, data):
        self.d = data
        self.p = 0

    def raw(self, n):
        r = self.d[self.p:self.p + n]
        self.p += n
        return r

    def u32(self):
        return struct.unpack("<I", self.raw(4))[0]

    def i64(self):
        return struct.unpack("<q", self.raw(8))[0]

    def u64(self):
        return struct.unpack("<Q", self.raw(8))[0]

    def f32(self):
        return struct.unpack("<f", self.raw(4))[0]

    def string(self):
        n = self.u64()
        return self.raw(n).decode("utf-8", "replace")


SCALAR_FMT = {T_UINT8: "B", T_INT8: "b", T_UINT16: "H", T_INT16: "h", T_UINT32: "I",
              T_INT32: "i", T_FLOAT32: "f", T_BOOL: "?", T_UINT64: "Q", T_INT64: "q",
              T_FLOAT64: "d", T_FLOAT16: "e"}


def read_gguf(path):
    with open(path, "rb") as f:
        head = f.read(4)
        assert head == GGUF_MAGIC, head
        version = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        blob = f.read()
    r = Reader(blob)
    kv = {}
    for _ in range(n_kv):
        k = r.string()
        t = r.u32()
        if t == T_ARRAY:
            at = r.u32()
            n = r.u64()
            if at == T_STRING:
                kv[k] = [r.string() for _ in range(n)]
            else:
                fmt = SCALAR_FMT[at]
                sz = struct.calcsize("<" + fmt)
                kv[k] = list(struct.unpack(f"<{n}{fmt}", r.raw(sz * n)))
                if at == T_BOOL:
                    kv[k] = [bool(x) for x in kv[k]]
        elif t == T_STRING:
            kv[k] = r.string()
        else:
            fmt = SCALAR_FMT[t]
            kv[k] = struct.unpack("<" + fmt, r.raw(struct.calcsize("<" + fmt)))[0]
    tensors = []
    for _ in range(n_tensors):
        name = r.string()
        n_dims = r.u32()
        dims = [r.u64() for _ in range(n_dims)]
        dtype = r.u32()
        offset = r.u64()
        tensors.append({"name": name, "dims": dims, "dtype": dtype, "offset": offset})
    alignment = int(kv.get("general.alignment", 32))
    info_end_abs = 24 + r.p
    data_start_abs = (info_end_abs + alignment - 1) // alignment * alignment
    return {"version": version, "kv": kv, "tensors": tensors,
            "alignment": alignment, "data_start_abs": data_start_abs}


def read_tensor_raw(path, gg, name, count=None):
    """Seek and read a tensor's raw bytes (dtype taken from the table)."""
    for t in gg["tensors"]:
        if t["name"] != name:
            continue
        n = 1
        for d in t["dims"]:
            n *= d
        if count is not None:
            n = min(n, count)
        item = TYPE_SIZE[t["dtype"]]
        off = gg["data_start_abs"] + t["offset"]
        with open(path, "rb") as f:
            f.seek(off)
            data = f.read(item * n)
        return t, data
    return None, None


def main_gguf(path):
    gg = read_gguf(path)
    lines = []
    lines.append(f"gguf: {path}")
    lines.append(f"version={gg['version']} n_tensors={len(gg['tensors'])} n_kv={len(gg['kv'])} "
                 f"data_start_abs={gg['data_start_abs']}")
    for k in sorted(gg["kv"]):
        if k.startswith("ssm") or k.startswith("gdn") or "hadamard" in k or "layer_norm" in k \
           or k.startswith("general.architecture") or k.endswith("block_count") \
           or k.endswith("embedding_length"):
            v = gg["kv"][k]
            if isinstance(v, list) and len(v) > 12:
                v = str(v[:12]) + f"... (len {len(v)})"
            lines.append(f"KV {k} = {v}")
    lines.append("")
    lines.append("blk.0 tensor table:")
    for t in gg["tensors"]:
        if t["name"].startswith("blk.0.") or t["name"].startswith("prism.hadamard"):
            lines.append(f"  {t['name']}  dims={t['dims']} dtype={t['dtype']} offset={t['offset']}")
    lines.append("")
    for nm in ("blk.0.ssm_a", "blk.0.ssm_dt.bias", "blk.0.ssm_norm.weight"):
        t, data = read_tensor_raw(path, gg, nm)
        if t is None:
            lines.append(f"{nm}: NOT FOUND")
            continue
        if t["dtype"] != T_FLOAT32:
            lines.append(f"{nm}: dtype={t['dtype']} (not f32)")
            continue
        vals = list(struct.unpack(f"<{len(data)//4}f", data))
        lines.append(f"{nm} dims={t['dims']} n={len(vals)}  (f32, 实测 from file)")
        lines.append("  all: " + ", ".join(f"{v:.8f}" for v in vals))
        lines.append("  min=%.8f max=%.8f all_negative=%s" %
                     (min(vals), max(vals), all(v < 0 for v in vals)))
    for nm in ("blk.0.ssm_conv1d.weight",):
        t, data = read_tensor_raw(path, gg, nm)
        if t is None:
            lines.append(f"{nm}: NOT FOUND")
            continue
        lines.append(f"{nm} dims={t['dims']} dtype={t['dtype']} bytes={len(data)}")
        if t["dtype"] == T_FLOAT32:
            vals = list(struct.unpack(f"<{len(data)//4}f", data))
            n_c = t["dims"][1]
            lines.append(f"  channel0 kernel (4): {vals[:4]}")
            lines.append(f"  channel1 kernel (4): {vals[4:8]}")
            lines.append(f"  last  channel kernel: {vals[4*(n_c-1):4*n_c]}")
            with open(os.path.join(ROOT, "runs", "gdn_conv1d_b0.tsv"), "w", encoding="utf-8") as fh:
                fh.write("channel\tk0\tk1\tk2\tk3\n")
                for c in range(n_c):
                    fh.write("%d\t%.9g\t%.9g\t%.9g\t%.9g\n" % tuple([c] + vals[4 * c:4 * c + 4]))
    txt = "\n".join(lines)
    with open(os.path.join(ROOT, "runs", "gdn_blk0_consts.txt"), "w", encoding="utf-8") as f:
        f.write(txt + "\n")
    print(txt[:6000])
    return gg


if __name__ == "__main__":
    gguf_path = DEFAULT_GGUF
    if "--gguf" in sys.argv:
        gguf_path = sys.argv[sys.argv.index("--gguf") + 1]
    nodes = main_log(parse_log())
    if os.path.exists(gguf_path):
        main_gguf(gguf_path)
    else:
        print(f"[gdn_probe] gguf not found: {gguf_path}")
