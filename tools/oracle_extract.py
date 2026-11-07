#!/usr/bin/env python3
"""oracle_extract.py - turn the official engine's per-node debug dumps into machine-readable oracles.

The dumps (runs/dbg_*.log) were produced by the PrismML fork's `llama-debug.exe` with ggml's
print_debug callback enabled: every graph node prints one line

    common_debug_cb_eval: <name> = (<dtype>) <OP>(<src0>{n0, n1, n2, n3}, <src1>{...}}) = {o0, o1, o2, o3}
        [ ... first/last 3 of first/last 3 rows, 4 decimals ... ]
        sum = -0.996582                            <- full precision, the only usable invariant

The element prints are truncated, so this tool keeps the *structure* (name, op, operand names, all
shapes) plus the `sum =` values, in file order, tagged with the pass (graph evaluation) they belong
to.  One node can print several arrays and therefore several sums (observed only for the GDN
`final_output-N (reshaped)/(permuted)/(cont)` chain); the FIRST sum is taken as the node's sum and
the rest are kept in `all_sums` so nothing is silently dropped.

Outputs (all under runs/, TSV with a header line):
  runs/oracle_sums.tsv        one row per node, in file order
  runs/oracle_named.tsv       rows for nodes that carry a graph label or mention a model tensor
  runs/oracle_gemv_map.tsv    one row per MUL_MAT, with the GEMV ordinal over quantized weights
  runs/oracle_hadamard_set.tsv per-weight-name membership table (dump operand set vs the GGUF's
                              prism.hadamard.weight_names array)  [only when the GGUF is readable]

Re-runnable, CPU only, reads nothing but the logs (and optionally the GGUF header).

    python tools/oracle_extract.py                     # default dumps
    python tools/oracle_extract.py runs/a.log runs/b.log
"""
import os
import re
import sys
import struct
from collections import Counter, OrderedDict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
GGUF = os.environ.get("GGUF", r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf")

DEFAULT_LOGS = ["dbg_full_real.log", "dbg_fa_off.log", "dbg_names.log",
                "dbg_probe2.log", "dbg_attn_oracle.log", "dbg_probe1.log",
                "dbg_rope_probe.log", "dbg_tiny.log"]

PREFIX = "common_debug_cb_eval:"
NODE_RE = re.compile(
    r"^common_debug_cb_eval:\s?(.*?) = \((\w+)\)\s+([A-Z_0-9]+)\((.*)\)\s*=\s*\{([^}]*)\}\s*$")
SUM_RE = re.compile(r"^\s+sum = (-?[\d.eE+]+)\s*$")
ARG_RE = re.compile(r"\{([0-9, ]*)\}")
AUTONAME_RE = re.compile(r"^node_(\d+)$")
PARENONLY_RE = re.compile(r"^\(.*\)$")
LEAF_RE = re.compile(r"^(leaf_\d+|\d*)$")

# ------------------------------------------------------------------ GGUF metadata (read-only)
GGUF_TYPE_NAMES = ["u8", "i8", "u16", "i16", "u32", "i32", "f32", "bool",
                   "string", "array", "u64", "i64", "f64"]


def read_str_array(path, key):
    """Return metadata array `key` (array of str) or None.  Reads only the KV section."""
    if not os.path.exists(path):
        return None
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            return None
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]

        def rd(fmt):
            return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]

        def rstr():
            n = rd("Q")
            return f.read(n).decode("utf-8", "replace")

        def skip(t, depth=0):
            if t in (0, 7):
                f.read(1)
            elif t in (1, 2, 4, 5, 10, 11):
                f.read(struct.calcsize("<" + {1: "b", 2: "h", 4: "I", 5: "i", 10: "Q", 11: "q"}[t]))
            elif t == 3:
                f.read(2)
            elif t in (6, 12):
                f.read(4)
            elif t == 13:
                f.read(8)
            elif t == 8:
                n = rd("Q"); f.read(n)
            elif t == 9:
                et = rd("I"); n = rd("Q")
                for _ in range(n):
                    skip(et, depth + 1)
            else:
                raise ValueError("gguf type %d" % t)

        found = None
        for _ in range(n_kv):
            k = rstr()
            t = rd("I")
            if k == key and t == 9:
                et = rd("I"); n = rd("Q")
                assert et == 8, "expected array<string>, got array of type %d" % et
                found = [rstr() for _ in range(n)]
            elif k == key:
                found = ("<not-an-array type %d>" % t)
            else:
                skip(t)
        del n_tensors, ver
    return found


# ------------------------------------------------------------------ dump parsing
def split_args(s):
    """`'a{1, 2, 3, 4}, b{5, 6, 7, 8}}'` -> [('a', '1,2,3,4'), ('b', '5,6,7,8')]."""
    out, prev = [], 0
    first = True
    for m in ARG_RE.finditer(s):
        name = s[prev:m.start()]
        name = name.strip()
        if name.endswith(","):
            name = name[:-1].strip()
        if first and name == "":
            pass
        out.append((name, m.group(1).strip().replace(" ", "")))
        prev = m.end()
        first = False
        s2 = s[prev:]
        if s2.startswith(", "):
            prev += 2
        elif s2.startswith(","):
            prev += 1
    return out


def parse_log(path):
    """-> list of node dicts, in file order, tagged with pass index (0-based)."""
    nodes = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        lines = fh.read().split("\n")
    cur_pass = 0
    max_auto = -1
    i = 0
    while i < len(lines):
        l = lines[i]
        i += 1
        if not l.startswith(PREFIX):
            continue
        m = NODE_RE.match(l.rstrip("\r"))
        if not m:
            sys.stderr.write("PARSE FAIL %s:%d: %s\n" % (path, i, l[:160]))
            continue
        name, dtype, op, args, shape = m.groups()
        auto = AUTONAME_RE.match(name)
        # new pass = ggml node counter restarts (auto node index goes backwards)
        if auto:
            idx = int(auto.group(1))
            if nodes and idx < max_auto and idx <= 1:
                cur_pass += 1
                max_auto = -1
            max_auto = max(max_auto, idx)
        sums = []
        while i < len(lines):
            s = lines[i]
            if s.startswith(PREFIX):
                break
            sm = SUM_RE.match(s)
            if sm:
                sums.append(float(sm.group(1)))
            elif not re.match(r"^(\[\]?$|[...,]+$|\[|\])", s.strip()):
                if s.strip():
                    sys.stderr.write("STRAY %s:%d: %s\n" % (path, i + 1, s[:120]))
            i += 1
        nodes.append(dict(line=i, end_line=i, pass_idx=cur_pass, name=name, dtype=dtype,
                          op=op, args=split_args(args), shape=shape.replace(" ", ""),
                          sums=sums))
    return nodes


def is_model_tensor(nm):
    """operand looks like a GGUF tensor (not an intermediate / leaf / device-copy tag)"""
    if nm.endswith(".weight") or nm.endswith(".bias"):
        return True
    return nm.startswith("prism.")


def is_hadamard_matrix(nm):
    return nm.startswith("prism.hadamard.")


def plain_label(nm):
    """graph label we can trust (not an auto node id, not a bare transform tag, not a leaf)"""
    if nm == "":
        return False
    if AUTONAME_RE.match(nm) or PARENONLY_RE.match(nm):
        return False
    if LEAF_RE.match(nm):
        return False
    return True


def strip_dev_tags(nm):
    """'Vulkan0#norm-31#0' / 'CPU#Qcur-3 (view)#0' -> 'norm-31' (the underlying graph label)"""
    m = re.match(r"^(?:Vulkan\d+|CPU)#(.*)#\d+$", nm)
    return m.group(1) if m else nm


def fmt_args(args):
    return "|".join(n for n, _ in args)


def fmt_shapes(args):
    return "|".join(s if s else "-" for _, s in args)


# ------------------------------------------------------------------ emit
def write_tsv(path, header, rows):
    with open(path, "w", encoding="utf-8", newline="") as o:
        o.write("\t".join(header) + "\n")
        for r in rows:
            o.write("\t".join("" if c is None else str(c) for c in r) + "\n")
    return path


def main():
    logs = sys.argv[1:]
    if not logs:
        logs = [os.path.join(RUNS, n) for n in DEFAULT_LOGS
                if os.path.exists(os.path.join(RUNS, n))]
    per_file = OrderedDict()
    for p in logs:
        p = p if os.path.isabs(p) else os.path.join(ROOT, p)
        if not os.path.exists(p):
            print("skip (missing): %s" % p)
            continue
        per_file[os.path.basename(p)] = parse_log(p)

    # ---------------- oracle_sums.tsv
    rows = []
    for fn, nodes in per_file.items():
        for nd in nodes:
            rows.append([fn, nd["line"], nd["pass_idx"], nd["name"],
                         (AUTONAME_RE.match(nd["name"]).group(1)
                          if AUTONAME_RE.match(nd["name"]) else ""),
                         nd["op"], nd["dtype"], nd["shape"],
                         ("%.6f" % nd["sums"][0]) if nd["sums"] else "NO_SUM",
                         len(nd["sums"]),
                         ";".join("%.6f" % x for x in nd["sums"][1:]) if len(nd["sums"]) > 1 else "",
                         fmt_args(nd["args"]), fmt_shapes(nd["args"])])
    write_tsv(os.path.join(RUNS, "oracle_sums.tsv"),
              ["file", "line_number", "pass", "node_index_or_name", "auto_node_index", "op",
               "dtype", "out_shape", "sum", "sum_count", "extra_sums", "operand_tensor_names",
               "operand_shapes"], rows)

    # ---------------- oracle_named.tsv
    rows = []
    for fn, nodes in per_file.items():
        for nd in nodes:
            ops_model = [strip_dev_tags(n) for n, _ in nd["args"] if is_model_tensor(strip_dev_tags(n))]
            lbl = plain_label(nd["name"])
            if not lbl and not ops_model:
                continue
            src = ("node_name" if lbl else "") + ("+operand" if ops_model else "")
            rows.append([nd["name"] if lbl else "(unnamed)", nd["op"], nd["dtype"], nd["shape"],
                         ("%.6f" % nd["sums"][0]) if nd["sums"] else "NO_SUM",
                         fn, nd["line"], nd["pass_idx"], src,
                         fmt_args(nd["args"]), fmt_shapes(nd["args"])])
    write_tsv(os.path.join(RUNS, "oracle_named.tsv"),
              ["name", "op", "dtype", "out_shape", "sum", "file", "line", "pass",
               "label_source", "operand_tensor_names", "operand_shapes"], rows)

    # ---------------- oracle_gemv_map.tsv
    had = read_str_array(GGUF, "prism.hadamard.weight_names")
    had_set = set(had) if isinstance(had, list) else None
    gemv_rows = []
    for fn, nodes in per_file.items():
        gemv_i = 0          # ordinal over quantized-weight GEMVs, graph order, per pass
        seen_weight = Counter()
        for nd in nodes:
            if nd["op"] != "MUL_MAT":
                continue
            if len(nd["args"]) < 2:
                continue
            w_name, w_shape = nd["args"][0]
            a_name, a_shape = nd["args"][1]
            w_name = strip_dev_tags(w_name)
            ham = is_hadamard_matrix(w_name)
            if not ham:
                gemv_i_this = gemv_i
                gemv_i += 1
            else:
                gemv_i_this = ""
            if not ham:
                seen_weight[w_name] += 1
            gemv_rows.append([fn, nd["line"], nd["pass_idx"], nd["name"], gemv_i_this,
                              w_name, w_shape, ham, a_name, a_shape, nd["shape"],
                              ("%.6f" % nd["sums"][0]) if nd["sums"] else "NO_SUM",
                              (("" if had_set is None else
                                ("yes" if w_name in had_set else "no")) if not ham else "n/a")])
    write_tsv(os.path.join(RUNS, "oracle_gemv_map.tsv"),
              ["file", "line", "pass", "node_index_or_name", "gemv_ordinal", "weight_tensor_name",
               "weight_shape", "is_hadamard_matrix", "activation_tensor_name", "activation_shape",
               "out_shape", "sum_of_output", "in_prism_hadamard_weight_names"], gemv_rows)

    # ---------------- hadamard set diff
    diff_path = None
    if had_set is not None:
        names = sorted(set([r[5] for r in gemv_rows if r[7] is False]))
        hset_rows = []
        for nm in sorted(had_set | set(names)):
            hset_rows.append([nm, "yes" if nm in had_set else "no",
                              "yes" if nm in names else "no",
                              "OK" if (nm in had_set and nm in names) else
                              ("IN_META_NOT_IN_DUMP" if nm in had_set else "IN_DUMP_NOT_IN_META")])
        diff_path = write_tsv(os.path.join(RUNS, "oracle_hadamard_set.tsv"),
                              ["tensor_name", "in_prism_hadamard_weight_names",
                               "is_MUL_MAT_operand", "verdict"], hset_rows)
        meta_only = [r[0] for r in hset_rows if r[3] == "IN_META_NOT_IN_DUMP"]
        dump_only = [r[0] for r in hset_rows if r[3] == "IN_DUMP_NOT_IN_META"]
    else:
        meta_only = dump_only = None

    # ---------------- coverage summary
    print("=" * 78)
    print("oracle_extract.py coverage summary")
    print("=" * 78)
    for fn, nodes in per_file.items():
        npass = nodes[-1]["pass_idx"] + 1 if nodes else 0
        withsum = sum(1 for n in nodes if n["sums"])
        multi = sum(1 for n in nodes if len(n["sums"]) > 1)
        print("\n%s : %d nodes in %d pass(es); %d with sum (%d nodes print >1 sum)"
              % (fn, len(nodes), npass, withsum, multi))
        for p in range(npass):
            sub = [n for n in nodes if n["pass_idx"] == p]
            tok = [n for n in sub if n["name"] == "node_0" and n["op"] == "GET_ROWS"]
            ntok = ""
            if tok:
                sh = tok[0]["args"][1][1]
                ntok = " n_tokens=%s (inp_tokens shape %s), out %s" % (
                    sh.split(",")[0], sh, tok[0]["shape"])
            outs = [n for n in sub if n["name"] == "result_output"]
            osum = ("%.6f" % outs[0]["sums"][0]) if (outs and outs[0]["sums"]) else "MISSING"
            print("  pass %d: %d nodes, lines %d-%d, %s  result_output sum=%s"
                  % (p, len(sub), sub[0]["line"], sub[-1]["end_line"], ntok.strip(), osum))
            # per-block named coverage
            blocks = Counter()
            for n in sub:
                m = re.search(r"-(\d+)$", n["name"])
                if m:
                    blocks[int(m.group(1))] += 1
            named = len(blocks)
            miss = [b for b in range(32) if b not in blocks]
            print("           named '-B' nodes: %d distinct blocks covered, %d nodes total"
                  % (named, sum(blocks.values())))
            if miss:
                print("           blocks with NO named node: %s" % (miss,))
            ops_c = Counter(n["op"] for n in sub)
            print("           ops: %s" % ", ".join("%s=%d" % kv for kv in sorted(ops_c.items())))
    if had_set is not None:
        print("\nprism.hadamard.weight_names: %d entries; MUL_MAT non-hadamard operands: %d distinct"
              % (len(had_set), len(set(r[5] for r in gemv_rows if r[7] is False))))
        print("  in metadata but never a MUL_MAT operand (%d): %s" % (len(meta_only), meta_only))
        print("  MUL_MAT operand but not in metadata (%d): %s" % (len(dump_only), dump_only))
        print("  ->", diff_path)
    else:
        print("\nWARNING: GGUF %s not readable -> prism.hadamard.weight_names NOT RECOVERABLE here"
              % GGUF)
    print("\nwrote:", os.path.join(RUNS, "oracle_sums.tsv"))
    print("wrote:", os.path.join(RUNS, "oracle_named.tsv"))
    print("wrote:", os.path.join(RUNS, "oracle_gemv_map.tsv"))


if __name__ == "__main__":
    main()
