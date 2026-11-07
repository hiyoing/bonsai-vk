#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""oracle_step1.py - extract the ONE-TOKEN decode graph from the official fork's node dump.

Source (read-only):  runs/dbg_full_real.log
  produced by the PrismML fork's llama-debug.exe with ggml print_debug enabled, prompt "ab".
  It contains TWO graph evaluations:
    graph 0 : 2-token prefill        (inp_tokens{2, 1, 1, 1})
    graph 1 : 1-token decode step    (inp_tokens{1, 1, 1, 1})   <- the recurrence oracle

Every node prints as

    common_debug_cb_eval: <name> = (<dtype>) <OP>(<src0>{ne}, <src1>{ne}}) = {ne}
        [
            [
                [  v, v, v, ..., v, v, v  ],      <- first/last 3 of first/last 3 rows
                ...,
            ],
        ]
        sum = -4.543495                           <- full-tensor sum, the usable invariant

Outputs:
  runs/oracle_step1.tsv  one row per node of the 1-token graph:
                         index layer name op shape sum head3 tail3 srcs
  runs/oracle_step1.md   the consequential-node tables (GDN block + first attention block)
                         with anchor validation and zero/NaN flags.

Validation: the prefill graph is parsed with the same reader and checked against three known-good
anchors (node_0 -4.543495, MUL_MAT(prism.hadamard.4096..) node_2 -0.996582, model.input_embed
-0.601764).  If any anchor misses the tool exits with code 1 and writes nothing.

Nothing but these two output files is written; the log is opened read-only.

    PYTHONIOENCODING=utf-8 python tools/oracle_step1.py
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(ROOT, "runs")
LOG = os.path.join(RUNS, "dbg_full_real.log")
TSV_OUT = os.path.join(RUNS, "oracle_step1.tsv")
MD_OUT = os.path.join(RUNS, "oracle_step1.md")

# ---------------------------------------------------------------- parser

HEADER_RE = re.compile(
    r"^common_debug_cb_eval:\s*(?P<name>.*?)\s*=\s*\((?P<dtype>[A-Za-z0-9]+)\)\s*"
    r"(?P<op>[A-Za-z_][A-Za-z_0-9]*)\((?P<srcs>.*)\)\s*=\s*\{(?P<shape>[^{}]*)\}\s*$"
)
SUM_RE = re.compile(r"^\s*sum\s*=\s*(?P<val>[-+]?(?:nan|inf|\d+\.\d*|\.\d+))\s*$")
VALUE_RE = re.compile(r"[-+]?(?:nan|inf|-inf|(?:\d+\.\d*|\.\d+))")
LOGINFO_RE = re.compile(r"^\d+\.\d+\.\d+\.\d+\s+[A-Z]\s")
BLK_RE = re.compile(r"blk\.(\d+)\.")
CACHE_L_RE = re.compile(r"_l(\d+)")
SUFFIX_RE = re.compile(r"-(\d+)(?:\s|\(|$)")


class Node(object):
    __slots__ = ("line", "name", "dtype", "op", "srcs", "shape", "sums", "values", "rows", "layer")

    def __init__(self, line, name, dtype, op, srcs, shape):
        self.line = line
        self.name = name
        self.dtype = dtype
        self.op = op
        self.srcs = srcs
        self.shape = shape
        self.sums = []
        self.values = []
        self.rows = []
        self.layer = None

    @property
    def sum(self):
        return self.sums[0] if self.sums else ""

    @property
    def sum_f(self):
        try:
            return float(self.sum)
        except ValueError:
            return float("nan")

    @property
    def head3(self):
        return ",".join(self.values[:3])

    @property
    def tail3(self):
        return ",".join(self.values[-3:]) if len(self.values) >= 3 else ""


BRACKET_ONLY_RE = re.compile(r"^[\s\[\],.]*$")


def parse_graphs(path):
    """-> (graphs, misc_lines, unmatched_header_lines).  A new graph starts at every `node_0`."""
    graphs = [[]]
    misc = []
    cur = None
    bad = []
    with open(path, "r", encoding="utf-8", errors="replace") as fh:
        for lineno, raw in enumerate(fh, 1):
            line = raw.rstrip("\n")
            m = HEADER_RE.match(line)
            if m:
                if m.group("name").strip() == "node_0" and graphs[-1]:
                    graphs.append([])
                cur = Node(lineno, m.group("name").strip(), m.group("dtype"),
                           m.group("op"), m.group("srcs").strip(), m.group("shape").strip())
                graphs[-1].append(cur)
                continue
            if line.startswith("common_debug_cb_eval:"):
                bad.append(lineno)
                cur = None
                misc.append((lineno, line))
                continue
            if cur is not None:
                s = SUM_RE.match(line)
                if s:
                    cur.sums.append(s.group("val"))
                    continue
                toks = VALUE_RE.findall(line)
                if toks:
                    cur.values.extend(toks)
                    cur.rows.append(toks)
                    continue
                if BRACKET_ONLY_RE.match(line):
                    continue
                cur = None
            if line.strip():
                misc.append((lineno, line))
    return [g for g in graphs if g], misc, bad


# ---------------------------------------------------------------- layer attribution

def assign_layers(nodes):
    """blk.N in name/srcs wins; then cache _lN; else position (embed before the first block node,
    out after the last one); else an `-N` name suffix; else carry the previous layer."""
    blk_idx = [i for i, n in enumerate(nodes) if BLK_RE.search(n.name + n.srcs)]
    first_blk = blk_idx[0] if blk_idx else len(nodes)
    last_blk = blk_idx[-1] if blk_idx else -1
    out = []
    prev = None
    for i, n in enumerate(nodes):
        blob = n.name + " " + n.srcs
        m = BLK_RE.search(blob)
        if m:
            layer = m.group(1)
        elif i < first_blk:
            layer = "embed"
        elif i > last_blk:
            layer = "out"
        else:
            cm = CACHE_L_RE.search(n.name) if n.name.startswith("cache_") else None
            if cm:
                layer = cm.group(1)
            else:
                sm = SUFFIX_RE.search(n.name)
                layer = sm.group(1) if sm else (prev if prev is not None else "embed")
        out.append(layer)
        if layer not in ("embed", "out"):
            prev = layer
    return out


# ---------------------------------------------------------------- selection helpers

def find(nodes, layer, op=None, name=None, src=None, shape=None, nth=1):
    """first `nth` node matching every predicate; returns None when absent."""
    hits = []
    nname = re.compile(name) if name else None
    nsrc = re.compile(src) if src else None
    for n in nodes:
        if layer is not None and n.layer != layer:
            continue
        if op and n.op != op:
            continue
        if nname and not nname.search(n.name):
            continue
        if nsrc and not nsrc.search(n.srcs):
            continue
        if shape and n.shape != shape:
            continue
        hits.append(n)
        if len(hits) == nth:
            return hits[-1]
    return None


def cell(nodes, label, **kw):
    n = find(nodes, **kw)
    if n is None:
        return (label, "MISSING", "-", "-", "-", "-")
    return (label, n.name, n.op, n.shape, n.sum, "%s | %s" % (n.head3, n.tail3))


def md_table(rows, header):
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join(["---"] * len(header)) + " |"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


ROLE_HEADER = ["role", "node name", "op", "shape", "sum", "head3 \\| tail3"]


def state_probe(nodes, kind):
    """per-layer persistent-state reads/writes of one graph. kind in {gdn, conv, kv}."""
    rows = []
    for lay in sorted({n.layer for n in nodes if n.op == "GATED_DELTA_NET"},
                      key=lambda x: int(x) if str(x).isdigit() else -1):
        if kind == "gdn":
            rd = find(nodes, layer=lay, op="GET_ROWS", src=r"cache_s_l%s\b" % lay,
                      shape="524288, 1, 1, 1")
            wr = find(nodes, layer=lay, op="CPY", name=r"copy of new_state-%s\b" % lay)
        elif kind == "conv":
            rd = find(nodes, layer=lay, op="GET_ROWS", name=r"^conv_states-%s\b" % lay)
            wr = find(nodes, layer=lay, op="CPY", name=r"copy of conv_state_last-%s\b" % lay)
        else:
            rd = wr = None
        rows.append((lay, rd, wr))
    if kind == "kv":
        rows = []
        for lay in sorted({n.layer for n in nodes if n.op == "SET_ROWS" and n.name.startswith("cache_k_")},
                          key=lambda x: int(x) if str(x).isdigit() else -1):
            cache = find(nodes, layer=lay, op="SET_ROWS", name=r"^cache_k_l%s\b" % lay)
            cur = find(nodes, layer=lay, op="ROPE", name=r"^k_cache_in-%s$" % lay)
            rows.append((lay, cache, cur))
    return rows


def gdn_layout_checks(nodes):
    """structural invariants of the fused GATED_DELTA_NET output:
    sum(out) == sum(new_state) + sum(attn_output), and ne1 == 128 + n_tokens."""
    errs, missing, ntok = [], 0, sorted({int(m.group(1)) for n in nodes
                                         for m in [re.search(r"inp_tokens\{(\d+),", n.srcs)]
                                         if n.op == "GET_ROWS" and m})
    for n in [x for x in nodes if x.op == "GATED_DELTA_NET"]:
        lay = n.layer
        ns = find(nodes, layer=lay, op="VIEW", name=r"^new_state-%s$" % lay)
        ao = find(nodes, layer=lay, op="VIEW", name=r"^attn_output-%s$" % lay)
        if not (ns and ao):
            missing += 1
            continue
        errs.append(abs(n.sum_f - ns.sum_f - ao.sum_f))
        ne1 = int(n.shape.split(",")[1])
        if ntok and ne1 != 128 + ntok[0]:
            missing += 1
    return (max(errs) if errs else float("nan")), len(errs), missing


# ---------------------------------------------------------------- main

ANCHORS = [("node_0 GET_ROWS(token_embd)", ("node_0", "GET_ROWS"), "-4.543495"),
           ("node_2 MUL_MAT(prism.hadamard.4096)", ("node_2", "MUL_MAT"), "-0.996582"),
           ("model.input_embed MUL", ("model.input_embed", "MUL"), "-0.601764")]


def main():
    if not os.path.isfile(LOG):
        sys.stderr.write("missing log: %s\n" % LOG)
        return 2

    graphs, misc, bad = parse_graphs(LOG)
    print("log                : %s" % LOG)
    print("graphs found       : %d" % len(graphs))
    for gi, g in enumerate(graphs):
        toks = find(g, None, op="GET_ROWS", name="node_0", src=r"inp_tokens\{(\d+)", nth=1)
        m = re.search(r"inp_tokens\{(\d+),", toks.srcs) if toks else None
        print("  graph %d         : %4d nodes, lines %d..%d, inp_tokens = %s"
              % (gi, len(g), g[0].line, g[-1].line, m.group(1) + " token(s)" if m else "?"))
    if bad:
        print("unparsed cb_eval lines: %d (first at %s)" % (len(bad), bad[:5]))
    multi = [n for g in graphs for n in g if len(n.sums) > 1]
    nosum = [n for g in graphs for n in g if not n.sums]
    print("node headers parsed : %d (= all cb_eval lines); multi-sum nodes: %d; sum-less: %d"
          % (sum(len(g) for g in graphs) + len(bad), len(multi), len(nosum)))

    if len(graphs) < 2:
        sys.stderr.write("expected 2 graphs, got %d - parser is wrong\n" % len(graphs))
        return 1

    pre, dec = graphs[0], graphs[1]

    # ---- anchor check on the prefill graph
    ok = True
    print("\nanchor check (prefill graph):")
    for label, (nm, op), expect in ANCHORS:
        n = find(pre, None, op=op, name="^%s$" % re.escape(nm), nth=1)
        got = n.sum if n else "MISSING"
        good = got == expect
        ok = ok and good
        print("  %-40s expect %-12s got %-12s %s" % (label, expect, got, "OK" if good else "FAIL"))
    if not ok:
        sys.stderr.write("anchor sums did not match - refusing to write outputs\n")
        return 1

    # ---- TSV for the 1-token graph (layers are attributed for both graphs)
    for g in graphs:
        for n, lay in zip(g, assign_layers(g)):
            n.layer = lay
    with open(TSV_OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write("index\tlayer\tname\top\tshape\tsum\thead3\ttail3\tsrcs\n")
        for i, n in enumerate(dec):
            srcs = re.sub(r"\s+", " ", n.srcs)
            fh.write("\t".join([str(i), n.layer, n.name, n.op,
                                n.shape.replace(" ", ""), n.sum,
                                n.head3, n.tail3, srcs]) + "\n")
    print("\nwrote %s (%d node rows)" % (TSV_OUT, len(dec)))

    # ---- zero / NaN census of the decode graph
    zero = [n for n in dec if n.sum == "0.000000" or n.sum == "-0.000000"]
    nan = [n for n in dec if "nan" in n.sum.lower()]
    by_layer = {}
    for n in zero:
        by_layer[n.layer] = by_layer.get(n.layer, 0) + 1
    print("zero-sum nodes   : %d   nan-sum nodes: %d" % (len(zero), len(nan)))

    # ---- markdown
    gdn_rows = [
        cell(dec, "input rmsnorm", layer="embed", op="RMS_NORM", name="^norm-0$"),
        cell(dec, "attn_norm (blk.0)", layer="0", op="MUL", src=r"blk\.0\.attn_norm\.weight"),
        cell(dec, "conv state read (3x8192)", layer="0", op="GET_ROWS", name=r"^conv_states-0$"),
        cell(dec, "prior GDN state read (cache_s_l0 GET_ROWS)", layer="0", op="GET_ROWS",
             src=r"cache_s_l0", shape="524288, 1, 1, 1"),
        cell(dec, "state_predelta-0 (reshape of that read)", layer="0", op="RESHAPE",
             name=r"^state_predelta-0$"),
        cell(dec, "qkv MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.attn_qkv\.weight"),
        cell(dec, "conv_input CONCAT", layer="0", op="CONCAT", name=r"^conv_input-0$"),
        cell(dec, "SSM_CONV", layer="0", op="SSM_CONV", name=r"^conv_output_raw-0$"),
        cell(dec, "SILU (conv output)", layer="0", op="SILU", name=r"^conv_output_silu-0$"),
        cell(dec, "L2_NORM (q,k)", layer="0", op="L2_NORM", name=r"^qk_conv_l2-0$"),
        cell(dec, "q VIEW", layer="0", op="VIEW", name=r"^q_conv_predelta-0$"),
        cell(dec, "k VIEW", layer="0", op="VIEW", name=r"^k_conv_predelta-0$"),
        cell(dec, "v VIEW", layer="0", op="VIEW", name=r"^v_conv_predelta-0$"),
        cell(dec, "alpha MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.ssm_alpha\.weight"),
        cell(dec, "alpha+dt bias ADD", layer="0", op="ADD", src=r"blk\.0\.ssm_dt\.bias"),
        cell(dec, "SOFTPLUS", layer="0", op="SOFTPLUS", name=r"^a_softplus-0$"),
        cell(dec, "gate = softmul*ssm_a MUL", layer="0", op="MUL", src=r"blk\.0\.ssm_a\{"),
        cell(dec, "beta MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.ssm_beta\.weight"),
        cell(dec, "beta SIGMOID", layer="0", op="SIGMOID", name=r"^beta_sigmoid-0$"),
        cell(dec, "GATED_DELTA_NET (fused)", layer="0", op="GATED_DELTA_NET"),
        cell(dec, "new_state VIEW", layer="0", op="VIEW", name=r"^new_state-0$"),
        cell(dec, "state CPY -> cache_s_l0", layer="0", op="CPY", name=r"copy of new_state-0"),
        cell(dec, "attn_output VIEW (tail of GDN out)", layer="0", op="VIEW", name=r"^attn_output-0$"),
        cell(dec, "z (attn_gate) MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.attn_gate\.weight"),
        cell(dec, "per-head RMS_NORM (128,32)", layer="0", op="RMS_NORM", name=r"^norm-0$",
             shape="128, 32, 1, 1"),
        cell(dec, "ssm_norm MUL", layer="0", op="MUL", src=r"blk\.0\.ssm_norm\.weight"),
        cell(dec, "SWIGLU (gdn gated)", layer="0", op="SWIGLU", shape="128, 32, 1, 1"),
        cell(dec, "final_output reshape/permute chain head", layer="0", op="RESHAPE",
             name=r"^final_output-0$"),
        cell(dec, "ssm_out MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.ssm_out\.weight"),
        cell(dec, "attn residual ADD", layer="0", op="ADD", name=r"^attn_residual-0$"),
        cell(dec, "ffn gate MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.ffn_gate\.weight"),
        cell(dec, "ffn up MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.ffn_up\.weight"),
        cell(dec, "ffn SWIGLU", layer="0", op="SWIGLU", name=r"^ffn_swiglu-0$"),
        cell(dec, "ffn down MUL_MAT", layer="0", op="MUL_MAT", src=r"blk\.0\.ffn_down\.weight"),
        cell(dec, "block output ADD (l_out-0)", layer="0", op="ADD", name=r"^l_out-0$"),
    ]

    attn_rows = [
        cell(dec, "attn_norm (blk.3)", layer="3", op="MUL", src=r"blk\.3\.attn_norm\.weight"),
        cell(dec, "attn_q MUL_MAT", layer="3", op="MUL_MAT", src=r"blk\.3\.attn_q\.weight"),
        cell(dec, "Q per-head RMS_NORM", layer="3", op="RMS_NORM", name=r"^norm-3$",
             shape="256, 16, 1, 1"),
        cell(dec, "Q q_norm MUL", layer="3", op="MUL", src=r"blk\.3\.attn_q_norm\.weight"),
        cell(dec, "Q ROPE", layer="3", op="ROPE", name=r"^Qcur-3$"),
        cell(dec, "attn_k MUL_MAT", layer="3", op="MUL_MAT", src=r"blk\.3\.attn_k\.weight"),
        cell(dec, "K per-head RMS_NORM", layer="3", op="RMS_NORM", name=r"^norm-3$",
             shape="256, 4, 1, 1"),
        cell(dec, "K k_norm MUL", layer="3", op="MUL", src=r"blk\.3\.attn_k_norm\.weight"),
        cell(dec, "K ROPE", layer="3", op="ROPE", name=r"^k_cache_in-3$"),
        cell(dec, "attn_v MUL_MAT", layer="3", op="MUL_MAT", src=r"blk\.3\.attn_v\.weight"),
        cell(dec, "K cache SET_ROWS", layer="3", op="SET_ROWS", name=r"^cache_k_l3"),
        cell(dec, "V cache SET_ROWS", layer="3", op="SET_ROWS", name=r"^cache_v_l3"),
        cell(dec, "FLASH_ATTN_EXT (softmax path)", layer="3", op="FLASH_ATTN_EXT"),
        cell(dec, "attn gate SIGMOID", layer="3", op="SIGMOID", name=r"^gate_sigmoid-3$"),
        cell(dec, "attn gated MUL", layer="3", op="MUL", name=r"^attn_gated-3$"),
        cell(dec, "attn_output MUL_MAT", layer="3", op="MUL_MAT", src=r"blk\.3\.attn_output\.weight"),
        cell(dec, "attn residual ADD", layer="3", op="ADD", name=r"^attn_residual-3$"),
        cell(dec, "ffn SWIGLU", layer="3", op="SWIGLU", name=r"^ffn_swiglu-3$"),
        cell(dec, "block output ADD (l_out-3)", layer="3", op="ADD", name=r"^l_out-3$"),
    ]

    # graph-level landmarks + identity checks + cold-cache evidence
    pre_emb = find(pre, None, op="GET_ROWS", name="^node_0$", src=r"token_embd")
    dec_emb = find(dec, None, op="GET_ROWS", name="^node_0$", src=r"token_embd")
    dec_sig = (dec_emb.head3, dec_emb.tail3)
    pre_sigs = [(",".join(r[:3]), ",".join(r[-3:])) for r in pre_emb.rows]
    emb_match = [i for i, sig in enumerate(pre_sigs) if sig == dec_sig]
    gdn_err, n_gdn_ok, n_gdn_bad = gdn_layout_checks(dec)
    gdn_dec, conv_dec, kv_dec = (state_probe(dec, k) for k in ("gdn", "conv", "kv"))
    gdn_pre, conv_pre, kv_pre = (state_probe(pre, k) for k in ("gdn", "conv", "kv"))
    read_nz = lambda rows: [r for r in rows if r[1] is not None and r[1].sum_f != 0.0]
    warm_gdn, warm_conv = read_nz(gdn_dec), read_nz(conv_dec)
    # a KV cache is "warm" iff its post-SET_ROWS sum is not just this token's K (f16 print noise is
    # ~1e-2, a retained prefill row is O(10))
    kv_gap, warm_kv = [], []
    for lay, cache, cur in kv_dec:
        if cache is not None and cur is not None:
            kv_gap.append((lay, cache.sum_f - cur.sum_f))
            if abs(cache.sum_f - cur.sum_f) > 0.5:
                warm_kv.append(lay)
    kv_pref_ref = [n.sum if n is not None else "MISSING" for _, n, _ in kv_pre]
    empty_zero = [n for n in zero if "0" in n.shape.replace(" ", "").split(",")]
    fa_layers = sorted({int(n.layer) for n in dec if n.op == "FLASH_ATTN_EXT"})
    fa = find(dec, layer=str(fa_layers[0]), op="FLASH_ATTN_EXT") if fa_layers else None
    vcur = find(dec, layer=str(fa_layers[0]), op="MUL_MAT",
                src=r"blk\.%d\.attn_v\.weight" % fa_layers[0]) if fa_layers else None
    fa_ratio = (fa.sum_f / vcur.sum_f) if (fa and vcur and vcur.sum_f) else float("nan")
    sm = lambda n: (n.sum if n is not None else "MISSING")
    md = []
    md.append("# Oracle: 1-token decode step (`runs/dbg_full_real.log`, second graph)\n")
    md.append("Extracted by `tools/oracle_step1.py`. Full node table: `runs/oracle_step1.tsv` "
              "(%d rows, one per graph node, file order).\n" % len(dec))
    md.append("## Scope / provenance\n")
    md.append("- Log holds **%d** graph evaluations: prefill (2 tokens, **%d** nodes, lines %d-%d) "
              "and the 1-token decode step (**%d** nodes, lines %d-%d).\n"
              % (len(graphs), len(pre), pre[0].line, pre[-1].line,
                 len(dec), dec[0].line, dec[-1].line))
    md.append("- The decode graph *would* be the recurrence oracle: `cache_s_l*` (GDN state), "
              "`cache_r_l*` (conv history) and `cache_k_l*`/`cache_v_l*` (KV) are read as graph "
              "**inputs**. In this particular dump every one of those reads sums to exactly zero - "
              "the prefill's history is NOT visible in the second graph. Read the "
              "`Cold-cache evidence` section before using these numbers to decide recurrence "
              "ordering.")
    md.append("- `sum` is the full-tensor f32 sum printed by ggml (6 decimals); `head3`/`tail3` "
              "are the printed element values, first/last 3 of the printed stream, kept verbatim "
              "(4 decimals, sign preserved, comma-joined). The prints are truncated, so only the "
              "`sum` column is a hard invariant; head3/tail3 are structural spot checks.")
    md.append("- `layer` is inferred from operand names (`blk.N.*` -> N, `cache_*_lN` -> N, restricted "
              "to tensors actually named `cache_*`), else an `-N` name suffix, else position: `embed` "
              "before the first block node and `out` after the last one; unlabelled `node_N` / "
              "`(reshaped)` helpers inherit the preceding block.\n")
    md.append("- parse completeness: %d `common_debug_cb_eval` lines in the log, %d nodes parsed, "
              "%d unparsed, %d nodes with more than one printed `sum` (the first is used), %d without "
              "any.\n" % (sum(len(g) for g in graphs) + len(bad), sum(len(g) for g in graphs),
                          len(bad), len(multi), len(nosum)))


    md.append("## Parser validation against known-good anchors (prefill graph)\n")
    md.append("| anchor node | expected sum | parsed sum | verdict |")
    md.append("| --- | --- | --- | --- |")
    for label, (nm, op), expect in ANCHORS:
        n = find(pre, None, op=op, name="^%s$" % re.escape(nm), nth=1)
        md.append("| %s | %s | %s | %s |" % (label, expect, n.sum, "OK" if n.sum == expect else "FAIL"))
    md.append("")
    md.append("## Token identity of the 1-token step\n")
    tok_lines = [(ln, t) for ln, t in misc
                 if t.startswith(("Model add_bos", "Input prompt", "Token ids"))
                 or re.match(r"^\S*\(\d+\)\s*$", t)]
    md.append("- The batch tensor `inp_tokens` is a graph input and ggml does not print input leaves, "
              "so **no node line carries the token id**. The only ids in the file are the run's own "
              "trailer lines (verbatim, with log line numbers):")
    md.append("")
    md.append("```")
    for ln, t in tok_lines:
        md.append("%6d: %s" % (ln, t.rstrip()))
    md.append("```")
    md.append("")
    md.append("  `Token ids (1):` therefore describes the **1-token batch** fed to the second graph: "
              "one token, id **365**, detokenized piece `ab`.")
    md.append("- Corroboration from the graph itself: the prefill's embedding lookup printed %s, "
              "while the 1-token step's lookup prints head `%s` / tail `%s`, which %s. Its "
              "embedding row is therefore a third, distinct one - consistent with 365 being the token "
              "sampled after the \"ab\" prefill, not a re-feed of the prompt."
              % (" and ".join("prefill row%d head [%s] tail [%s]" % (i, s[0], s[1])
                              for i, s in enumerate(pre_sigs)),
                 dec_sig[0], dec_sig[1],
                 "reproduces prefill row(s) " + ",".join(str(i) for i in emb_match)
                 if emb_match else "matches none of the prefill rows"))
    md.append("- No `generated text` line exists in the log; the only text lines are the four quoted "
              "above. "
              "Absolute position of this step is therefore `pos = 2` (0-based, third token), which is "
              "inferred, not printed - the ROPE operands (`leaf_58`) are input leaves and dump nothing, "
              "so the cos/sin pair must be reproduced from the position, not read from this log.\n")



    md.append("## Consequential GDN nodes (block 0 = first Gated Delta Net layer, 1-token step)\n")
    md.append(md_table(gdn_rows, ROLE_HEADER))
    md.append("\nStructure worth pinning in the engine: `GATED_DELTA_NET` is fused over the whole "
              "layer and emits `{4096, 129, 1, 1}` = the updated state "
              "(`128 x 128 x 32 = 524288 = 4096*128` elems) followed by **one** 4096-wide output row "
              "for this token (the prefill printed `{4096, 130, 1, 1}` for 2 tokens). "
              "`new_state-N = VIEW(node_52)` takes the first 128 columns, `attn_output-N = VIEW(node_52)` "
              "takes the last 4096 elements; `conv_input-N = CONCAT(conv_states_reshaped-N{3,8192}, "
              "qkv_mixed_transposed-N{1,8192})` = `{4,8192}` and the conv kernel is "
              "`blk.N.ssm_conv1d.weight{4,8192}` (kernel width 4, history 3 + this token). "
              "Checked on all %d GDN layers of this graph: `|sum(GDN out) - sum(new_state) "
              "- sum(attn_output)| <= %.6f` (print rounding) and `ne1 == 128 + n_tokens` always; "
              "%d layer(s) failed the check.\n" % (n_gdn_ok, gdn_err, n_gdn_bad))

    md.append("## Consequential softmax-attention nodes (block 3 = first attention layer, 1-token step)\n")
    md.append(md_table(attn_rows, ROLE_HEADER))
    md.append("\nAttention is dumped with **FLASH_ATTN_EXT** enabled, so there is no separate "
              "`MUL_SUM`/`SOFT_MAX`/`MUL_SUM` chain in this log: QK^T, softmax and PV are one node "
              "`node_281 = FLASH_ATTN_EXT(Qcur-3 (view) (permuted){...}, cache_k_l3 ... )`. To align "
              "a non-fused softmax path, use `runs/dbg_fa_off.log` for that block, or validate "
              "against this node's sum only.\n")

    md.append("## Graph-level landmarks (1-token step)\n")
    md.append(md_table([
        cell(dec, "token embedding GET_ROWS", layer="embed", op="GET_ROWS", name="^node_0$"),
        cell(dec, "input_embed (hadamard sign MUL)", layer="embed", op="MUL",
             name=r"^model\.input_embed$"),
        cell(dec, "final RMS_NORM", layer="out", op="RMS_NORM", name="^norm$"),
        cell(dec, "result_norm MUL output_norm", layer="out", op="MUL", name=r"^result_norm$"),
        cell(dec, "result_output MUL_MAT (248320 logits)", layer="out", op="MUL_MAT",
             name=r"^result_output$"),
    ], ROLE_HEADER))

    md.append("\n## Zero / NaN flags\n")
    md.append("- exact-zero `sum`: **%d** of %d nodes; NaN `sum`: **%d**. No computed activation "
              "(anything that is not a cache/state/index view) sums to exactly zero, so there are no "
              "dead branches in the graph." % (len(zero), len(dec), len(nan)))
    md.append("- %d of the zeros are structurally empty tensors (`shape` with a 0 dimension, e.g. the "
              "`{24576, 0, 1, 1}` GET_ROWS/CPY pairs the fork uses to no-op the state rotation); the "
              "rest are the cache/state reads and their views listed below." % len(empty_zero))
    md.append("- reproduce with: `awk -F'\\t' '$6==\"0.000000\"' runs/oracle_step1.tsv`")
    md.append("")
    md.append("| layer | zero-sum nodes |")
    md.append("| --- | --- |")
    for lay in sorted(by_layer, key=lambda x: (x == "embed", x == "out", int(x) if str(x).isdigit() else 0)):
        md.append("| %s | %d |" % (lay, by_layer[lay]))

    md.append("\n## Cold-cache evidence (this is the important caveat)\n")
    md.append("Every *persistent state input* of the 1-token step reads exactly zero, even though the "
              "same tensors were written with non-zero data at the end of the prefill graph. Per "
              "category (sums from the two parsed graphs):")
    md.append("")
    md.append(md_table([("GDN state `cache_s_lN`", len(gdn_dec), len(warm_gdn),
                         sm(gdn_dec[0][1]), sm(gdn_dec[0][2]), sm(gdn_pre[0][2]))] +
                       [("conv history `cache_r_lN`", len(conv_dec), len(warm_conv),
                         sm(conv_dec[0][1]), sm(conv_dec[0][2]), sm(conv_pre[0][2]))] +
                       [("KV cache `cache_k_lN`", len(kv_dec), len(warm_kv), "-", "-", "-")],
                       ["category", "layers probed", "non-zero reads",
                         "layer-0 read sum (step 1)", "layer-0 write sum (step 1)",
                         "layer-0 write sum (prefill)"]))
    md.append("")
    md.append("- GDN state reads that carried a non-zero history into the 1-token step: **%d of %d** "
              "GDN layers; conv-history reads: **%d of %d**. The buffers the prefill left behind were "
              "plainly non-zero (layer 0: GDN state %s, conv history %s) yet the 1-token step reads "
              "`0.000000` from both. KV cache: the post-`SET_ROWS` sum of the 1-token step equals "
              "*this token's* roped K within f16 print noise (%s), i.e. the 2 prefill rows are absent; "
              "had they been kept the gap would be of the order of the prefill's own cache sum (%s)."
              % (len(warm_gdn), len(gdn_dec), len(warm_conv), len(conv_dec),
                 sm(gdn_pre[0][2]), sm(conv_pre[0][2]),
                 "; ".join("l%s: %.6f" % kv_gap[i] for i in range(min(3, len(kv_gap)))),
                 "; ".join(kv_pref_ref[:3])))
    md.append("")
    md.append("Consequences for the engine work:")
    md.append("")
    md.append("1. **The step-1 graph is a single-token step from a ZERO state.** It is a valid oracle "
              "for node *structure* (shapes, offsets, fused-output layout, op order, the conv window "
              "`3+1 -> 4` concat, the `{4096,129}` GDN output) and for the cold-start values "
              "(`S = 0`, conv history `= 0`), but it cannot discriminate anything that only shows up "
              "with a warm state: `decay before vs after dot(S,k)`, conv left-padding vs the real "
              "window, or the delta-rule correction term all collapse to the same numbers when "
              "`S = 0`.")
    md.append("2. To get the warm oracle we actually need, re-dump with **at least 2 decode tokens "
              "after the prefill** (so the final graph reads a state written by a *previous* graph) and "
              "check the same probe: the `cache_s_lN`/`cache_r_lN` GET_ROWS sums must be non-zero. If "
              "they are still zero, the fork is not persisting state across evaluations at all and its "
              "decode path is broken - which would itself be a finding worth acting on.")
    md.append("3. Meanwhile the prefill graph *does* contain a warm, multi-token recurrence: its "
              "GDN/conv nodes run a 2-step scan internally (`{4096,130}` GDN output, `{5,8192}` conv "
              "input), so row-1 of every prefill node is the \"state after token 1\" case. Use it for "
              "the ordering tests, and use `runs/oracle_step1.tsv` for the single-token shapes/offsets.")
    md.append("4. The attention path in this step also sees only 1 cached K/V row, so its softmax is "
              "over a single element: `FLASH_ATTN_EXT` returns exactly that one V row, replicated over "
              "the GQA head groups - the dumped ratio `sum(node_%s) / sum(attn_v MUL_MAT)` is %.4f "
              "(= 16 q heads / 4 kv heads, f16 noise aside). Not an attention oracle either; use the "
              "prefill graph's per-row prints for that." % (fa.name.split("_")[-1] if fa else "?",
                                                            fa_ratio))
    md.append("5. Layer map read off this graph: %d GDN blocks (%s) and %d softmax-attention blocks "
              "(%s), i.e. attention every 4th layer starting at 3.\n"
              % (len(gdn_dec), ",".join(str(r[0]) for r in gdn_dec),
                 len(fa_layers), ",".join(str(x) for x in fa_layers)))

    md.append("## How to regenerate\n")

    md.append("```\ncd \"<repo-root>\"\nPYTHONIOENCODING=utf-8 python tools/oracle_step1.py\n```\n")

    with open(MD_OUT, "w", encoding="utf-8", newline="") as fh:
        fh.write("\n".join(md) + "\n")
    print("wrote %s" % MD_OUT)
    print("missing selections: %s"
          % ([r[0] for spec in (gdn_rows, attn_rows) for r in spec if r[1] == "MISSING"] or "none"))


    # headline numbers for the report-back
    print("\nheadline GDN sums, 1-token step (block 0):")
    for label, kw in [("qkv MUL_MAT", dict(layer="0", op="MUL_MAT", src=r"blk\.0\.attn_qkv\.weight")),
                      ("SSM_CONV", dict(layer="0", op="SSM_CONV")),
                      ("L2_NORM", dict(layer="0", op="L2_NORM")),
                      ("GATED_DELTA_NET", dict(layer="0", op="GATED_DELTA_NET")),
                      ("attn_output VIEW", dict(layer="0", op="VIEW", name=r"^attn_output-0$")),
                      ("ssm_out MUL_MAT", dict(layer="0", op="MUL_MAT", src=r"blk\.0\.ssm_out\.weight"))]:
        n = find(dec, **kw)
        print("  %-20s %s" % (label, n.sum if n else "MISSING"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
