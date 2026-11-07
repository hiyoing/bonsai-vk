#!/usr/bin/env python3
"""make_tiny_qwen35.py - build a SYNTHETIC qwen35 checkpoint whose every attention tensor is small
enough for the official engine's debug printer to dump element by element.

WHY THIS EXISTS.  runs/dbg_full_real.log (produced by llama-debug.exe --tensor-filter '.') gave us the
official layer-3 attention graph verbatim - op order, shapes, dtypes.  That settles the STRUCTURE.  It
did not settle the ARITHMETIC, because common_debug_cb_eval prints each tensor row as "first 3 values,
..., last 3 values" and only 4 decimals.  With head_dim = 256 the middle of every row is invisible, so
the two questions that actually decide whether our RoPE is right or merely plausible stay open:

  1. which dims are paired by the rotary transform - NeoX/half-split (e, e+n_rot/2) or interleaved
     (2e, 2e+1)?  both live inside dims [0, n_rot), and we only ever see dims 0,1,2,253,254,255.
  2. how rope.dimension_sections = [11, 11, 10, 0] maps the 32 frequency pairs onto the 4 position
     fields, and what those fields hold for text.

The answer is a checkpoint small enough that nothing is elided: rows printed in full.  The printer
elides a row only when it is longer than 3 + 3 = 6 elements, so head_dim = 4 is dumpable elementwise
while still exercising partial rotary (dimension_count = 2 < head_dim), GQA (4 q heads / 2 kv heads),
the gated-attention half of attn_q, per-head QK RMSNorm, and an f16 KV cache.

Everything is F32 and deterministic, so an audit can re-derive any single number from this file.
"""
import struct, sys, os, math, random

REAL = r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "runs", "tiny_qwen35.gguf")

# ggml type ids used in the tensor table (0 = f32, 1 = f16, 143 = PTQ1_0)
F32, F16 = 0, 1

# ---------------------------------------------------------------- GGUF reading (metadata only)
GU8, GI8, GU16, GI16, GU32, GI32, GF32, GBOOL, GSTR, GARR, GU64, GI64, GF64 = range(13)


class R:
    def __init__(self, f): self.f = f
    def raw(self, n):
        b = self.f.read(n); assert len(b) == n, (n, len(b)); return b
    def u8(self):  return struct.unpack("<B", self.raw(1))[0]
    def i8(self):  return struct.unpack("<b", self.raw(1))[0]
    def u16(self): return struct.unpack("<H", self.raw(2))[0]
    def i16(self): return struct.unpack("<h", self.raw(2))[0]
    def u32(self): return struct.unpack("<I", self.raw(4))[0]
    def i32(self): return struct.unpack("<i", self.raw(4))[0]
    def f32(self): return struct.unpack("<f", self.raw(4))[0]
    def u64(self): return struct.unpack("<Q", self.raw(8))[0]
    def i64(self): return struct.unpack("<q", self.raw(8))[0]
    def f64(self): return struct.unpack("<d", self.raw(8))[0]
    def s(self):   return self.raw(self.u64()).decode("utf-8", "replace")

    def one(self, t):
        return {GU8: self.u8, GI8: self.i8, GU16: self.u16, GI16: self.i16, GU32: self.u32,
                GI32: self.i32, GF32: self.f32, GBOOL: lambda: bool(self.u8()), GSTR: self.s,
                GU64: self.u64, GI64: self.i64, GF64: self.f64}[t]()

    def arr(self):
        et = self.u32(); n = self.u64()
        return et, [self.one(et) for _ in range(n)]


def read_kv(path):
    """returns (version, n_tensors, [(key, vtype, value_or_(elem_type, list))], alignment)"""
    out = []
    with open(path, "rb") as fh:
        r = R(fh)
        assert r.raw(4) == b"GGUF", "not a gguf"
        ver = r.u32(); nt = r.u64(); nkv = r.u64()
        for _ in range(nkv):
            k = r.s(); t = r.u32()
            v = r.arr() if t == GARR else r.one(t)
            out.append((k, t, v))
        al = 32
        for k, t, v in out:
            if k == "general.alignment": al = v
        # we stop before the tensor info: the tiny file replaces every tensor
    return ver, nt, out, al


# ---------------------------------------------------------------- GGUF writing
def w_str(o, s):
    b = s.encode("utf-8"); o.write(struct.pack("<Q", len(b)) + b)

def w_val(o, t, v):
    o.write({GU8: lambda: struct.pack("<B", v), GI8: lambda: struct.pack("<b", v),
             GU16: lambda: struct.pack("<H", v), GI16: lambda: struct.pack("<h", v),
             GU32: lambda: struct.pack("<I", v), GI32: lambda: struct.pack("<i", v),
             GF32: lambda: struct.pack("<f", v), GBOOL: lambda: struct.pack("<B", 1 if v else 0),
             GSTR: lambda: (w_str(o, v), b"")[1],
             GU64: lambda: struct.pack("<Q", v), GI64: lambda: struct.pack("<q", v),
             GF64: lambda: struct.pack("<d", v)}[t]())

def w_kv(o, k, t, v):
    w_str(o, k); o.write(struct.pack("<I", t))
    if t == GARR:
        et, items = v
        o.write(struct.pack("<I", et) + struct.pack("<Q", len(items)))
        for it in items: w_val(o, et, it)
    else:
        w_val(o, t, v)

def pad(o, al):
    z = (-o.tell()) % al
    o.write(b"\0" * z)


# ---------------------------------------------------------------- geometry (see module docstring)
N_EMBD   = 16     # hidden
N_HEAD   = 4      # query heads
N_HEADKV = 2      # kv heads -> n_rep = 2, so GQA broadcasting is exercised
HEAD_DIM = 4      # == N_EMBD / N_HEAD, and <= 6 so the official printer dumps every element
N_ROT    = 2      # partial rotary: dims [0,2) rotate, dims [2,4) pass through untouched
N_LAYER  = 2
FFN      = 32
N_VOCAB  = 248320                 # the real vocab, copied verbatim, so tokenisation is real too
SECTIONS = [1, 0, 0, 0]           # N_ROT/2 = 1 frequency pair, all in section 0
FREQ_BASE = 10000000.0            # same as the real checkpoint, so the table is the real one


def rnd(rng, n):
    """small, distinct, symmetric-about-zero values; distinct indices must map to distinct numbers so
    that a transposed or off-by-one index cannot hide."""
    return [rng.uniform(-0.9, 0.9) for _ in range(n)]


def main():
    ver, nt_real, kv, al = read_kv(REAL)
    print("read %d kv pairs from the real checkpoint (gguf v%d, %d tensors, alignment %d)"
          % (len(kv), ver, nt_real, al))

    arch = "qwen35"
    ov = {                       # key -> (gguf vtype, value)
        "general.architecture":            (GSTR, arch),
        "general.name":                    (GSTR, "tiny_qwen35_align"),
        "general.file_type":               (GU32, 1),          # 1 = F32, no quantisation anywhere
        "general.quantization_version":    (GU32, 3),
        arch + ".embedding_length":        (GU32, N_EMBD),
        arch + ".block_count":             (GU32, N_LAYER),
        arch + ".context_length":          (GU32, 4096),
        arch + ".feed_forward_length":     (GU32, FFN),
        arch + ".attention.head_count":    (GU32, N_HEAD),
        arch + ".attention.head_count_kv": (GU32, N_HEADKV),
        arch + ".attention.key_length":    (GU32, HEAD_DIM),
        arch + ".attention.value_length":  (GU32, HEAD_DIM),
        arch + ".attention.layer_norm_rms_epsilon": (GF32, 9.999999974752427e-07),
        arch + ".rope.dimension_count":    (GU32, N_ROT),
        arch + ".rope.dimension_sections": (GARR, (GI32, list(SECTIONS))),
        arch + ".rope.freq_base":          (GF32, FREQ_BASE),
        arch + ".full_attention_interval": (GU32, 1),          # every layer = full attention
        arch + ".vocab_size":              (GU32, N_VOCAB),
    }
    # ssm/linear-attention geometry: shrink it too, in case the arch config reads it regardless
    ov[arch + ".ssm.group_count"]    = (GU32, 1)
    ov[arch + ".ssm.inner_size"]     = (GU32, 8)
    ov[arch + ".ssm.state_size"]     = (GU32, 4)
    ov[arch + ".ssm.conv_kernel"]    = (GU32, 4)
    ov[arch + ".ssm.time_step_rank"] = (GU32, 2)
    ov[arch + ".nextn_predict_layers"] = (GU32, 0)

    # ---- the PrismML Incoherence/Hadamard side of the checkpoint, measured from the real file:
    #   prism.hadamard.transform = 'normalized-sylvester-walsh-hadamard', block_size = 4096,
    #   sign_widths = [4096, 12288], sign_values = 16384 x +-1, weight_names = the 201 tensors that
    #   were quantised AFTER being rotated, and inverse_weight_names = [token_embd.weight].
    #   The engine therefore synthesises H at run time and multiplies x by diag(signs) and by H before
    #   those matmuls.  That is a property of the FFN / GDN / LM-head path, NOT of full attention (no
    #   attn_q/attn_k/attn_v/attn_output name is in weight_names), which is exactly why attention can
    #   be aligned without reproducing H.  Here we keep the mechanism alive but make it the identity:
    #   every sign is +1, so `MUL(x, signs)` cannot hide an attention bug.
    present = ["output.weight"] + ["blk.%d.%s.weight" % (il, w) for il in range(N_LAYER)
                                   for w in ("ffn_gate", "ffn_up", "ffn_down")]
    ov["prism.hadamard.block_size"]    = (GU32, N_EMBD)
    ov["prism.hadamard.sign_widths"]   = (GARR, (GI32, [N_EMBD, FFN]))
    ov["prism.hadamard.sign_values"]   = (GARR, (GI32, [1] * (N_EMBD + FFN)))
    ov["prism.hadamard.weight_names"]  = (GARR, (GSTR, present))

    merged = []
    replaced = set()
    for k, t, v in kv:
        if k in ov:
            merged.append((k, ov[k][0], ov[k][1])); replaced.add(k)
        else:
            merged.append((k, t, v))
    for k in ov:
        if k not in replaced:
            merged.append((k, ov[k][0], ov[k][1]))
    # general.alignment must exist and stay what we write
    merged = [(k, t, v) for k, t, v in merged if k != "general.alignment"]
    merged.append(("general.alignment", GU32, al))

    # ---- tensors, in the dim order the real file uses: (d0, d1) = (input, output) for weights
    rng = random.Random(20261101)
    T = []   # (name, dims, dtype, bytes)

    def add(name, dims, dtype, gen):
        n = 1
        for d in dims: n *= d
        vals = gen(n)
        T.append((name, dims, dtype, struct.pack("<%df" % n, *vals)))

    add("token_embd.weight", [N_EMBD, N_VOCAB], F32,
        lambda n: [((i * 2654435761 >> 13) % 1000) / 4995.0 - 0.1 for i in range(n)])
    add("output_norm.weight", [N_EMBD], F32, lambda n: [1.0 + 0.05 * i for i in range(n)])
    add("output.weight", [N_EMBD, N_VOCAB], F32,
        lambda n: [((i * 40503 + 7) >> 5) % 997 / 4985.0 - 0.1 for i in range(n)])

    nbox = [0]
    def norm_vec(n):     # deliberately != 1 everywhere: catches an unapplied or mis-sliced norm.
        nbox[0] += 1     # and different per tensor, so swapping attn_q_norm / attn_k_norm is visible
        return [0.8 + 0.031 * ((i + 5 * nbox[0]) % 17) for i in range(n)]
    seed_box = [0]                    # per-tensor seed, so no two tensors share a value pattern
    def mat(n):          # deterministic LCG in (-1.0, 1.0); NOT restarted, so attn_q != attn_k != ...
        seed_box[0] += 1
        s = (12345 + 7919 * seed_box[0]) & 0xFFFFFFFF
        out = []
        for _ in range(n):
            s = (s * 1664525 + 1013904223) & 0xFFFFFFFF
            out.append(((s >> 8) & 0xFFFFFF) / 8388608.0 - 1.0)
        return out

    for il in range(N_LAYER):
        p = "blk.%d." % il
        add(p + "attn_norm.weight",          [N_EMBD],        F32, norm_vec)
        # attn_q is DOUBLED width: query heads in the first n_embd, the attention gate in the second
        # half. Shape asserted by the loader itself: "check_tensor_dims: tensor 'blk.0.attn_q.weight'
        # has wrong shape; expected 16, 32, got 16, 16" - matching the real file's 4096 x 8192.
        add(p + "attn_q.weight",             [N_EMBD, 2 * N_EMBD], F32, mat)
        add(p + "attn_k.weight",             [N_EMBD, N_HEADKV * HEAD_DIM], F32, mat)
        add(p + "attn_v.weight",             [N_EMBD, N_HEADKV * HEAD_DIM], F32, mat)
        add(p + "attn_q_norm.weight",        [HEAD_DIM],      F32, norm_vec)
        add(p + "attn_k_norm.weight",        [HEAD_DIM],      F32, norm_vec)
        add(p + "attn_output.weight",        [N_EMBD, N_EMBD], F32, mat)
        add(p + "post_attention_norm.weight", [N_EMBD],       F32, norm_vec)
        add(p + "ffn_gate.weight",           [N_EMBD, FFN],   F32, mat)
        add(p + "ffn_up.weight",             [N_EMBD, FFN],   F32, mat)
        add(p + "ffn_down.weight",           [FFN, N_EMBD],   F32, mat)

    # ---- write
    with open(OUT, "wb") as o:
        o.write(b"GGUF" + struct.pack("<I", 3))
        o.write(struct.pack("<Q", len(T)) + struct.pack("<Q", len(merged)))
        for k, t, v in merged: w_kv(o, k, t, v)
        # tensor info: name, n_dims, dims[], type, offset (offsets are relative to the data section)
        # GGUF insists every tensor starts on a multiple of `alignment`, and checks it:
        #   "tensor 'blk.0.attn_k_norm.weight' has offset 31787152, expected 31787168"
        # (a 4-float tensor is 16 B, so the next one must be padded up to 32). Measured, not guessed.
        isz = 0
        for name, dims, dtype, data in T:
            isz += 8 + len(name.encode("utf-8")) + 4 + 8 * len(dims) + 4 + 8
        base = o.tell() + isz
        base += (-base) % al
        off = base
        pads = []
        for name, dims, dtype, data in T:
            pads.append(off - base)
            off += len(data)
            off += (-off) % al
        for (name, dims, dtype, data), poff in zip(T, pads):
            w_str(o, name)
            o.write(struct.pack("<I", len(dims)))
            for d in dims: o.write(struct.pack("<Q", d))
            o.write(struct.pack("<I", dtype))
            o.write(struct.pack("<Q", poff))
        pad(o, al)
        assert o.tell() == base, (o.tell(), base)
        last = 0
        for (name, dims, dtype, data), poff in zip(T, pads):
            assert poff >= last
            o.write(b"\0" * (poff - last))     # alignment padding before this tensor
            o.write(data)
            # `last` is the ACTUAL end, not the aligned end: rounding it here made the next gap
            # evaluate to zero and silently dropped the padding bytes (measured: "tensor
            # blk.1.ffn_down.weight data is not within the file bounds", file 64 B short).
            last = poff + len(data)
        assert o.tell() == base + last, (o.tell(), base + last)
        o.write(b"\0" * ((-o.tell()) % al))
        total = o.tell()
        exp = base + last
        assert total >= exp, ("file is %d bytes short of the declared tensor range" % (exp - total))
        # self-check: the reader computes data_start as align_up(end-of-info-block) and then requires
        # data_start + every (offset + size) <= filesize. Re-derive it here or the loader fails with
        # "tensor ... data is not within the file bounds".
        exp = base + last
        assert total == exp, ("file is %d bytes short of the declared tensor range" % (exp - total))

    print("wrote %s  (%.2f MiB, %d tensors)" % (OUT, total / 1048576.0, len(T)))
    for name, dims, dtype, data in T[:4]:
        print("   %-28s dims=%s dtype=%d bytes=%d" % (name, dims, dtype, len(data)))
    print("   ...")
    print("geometry: n_embd=%d n_head=%d n_head_kv=%d head_dim=%d n_rot=%d sections=%s "
          "n_layer=%d ffn=%d base=%g" %
          (N_EMBD, N_HEAD, N_HEADKV, HEAD_DIM, N_ROT, SECTIONS, N_LAYER, FFN, FREQ_BASE))


if __name__ == "__main__":
    main()
