#!/usr/bin/env python3
"""patch_gguf_meta.py - copy a GGUF and rewrite scalar/array metadata values IN PLACE (same byte size).

Used for the rotary sign probe: qwen35 has head_dim 256 and n_rot 64, so a rotated pair is
(p, p+32) and the official debug printer - which elides every row wider than 6 elements - only ever
shows dims 0,1,2,253,254,255. Neither member of a pair is ever visible together with the other, so
the *sign* of the cross term (does dim p get -x_q*sin and dim p+32 get +x_p*sin, or the reverse?)
cannot be measured from the unmodified checkpoint.

Shrinking qwen35.rope.dimension_count to 4 makes the pairs (0,2) and (1,3): both members of pair 0
now live inside the first 3 printed columns, and theta_0 = base^(-2*0/4) = 1, so the pair (0,2)
rotates by exactly the position value. The sign then falls out of two printed numbers.

Only the value payload is rewritten, so every byte offset after the KV section is unchanged and the
tensor data is copied verbatim. Sizes must match: the tool refuses if they would not.

  python tools/patch_gguf_meta.py <in.gguf> <out.gguf> key=value[:type] ...
  value list syntax:  [1,1,0,0]  -> ARRAY whose element type is taken from the original
"""
import struct, sys, shutil, os

GU8, GI8, GU16, GI16, GU32, GI32, GF32, GBOOL, GSTR, GARR, GU64, GI64, GF64 = range(13)
SSZ = {GU8: 1, GI8: 1, GU16: 2, GI16: 2, GU32: 4, GI32: 4, GF32: 4, GBOOL: 1,
       GU64: 8, GI64: 8, GF64: 8}
FMT = {GU8: "<B", GI8: "<b", GU16: "<H", GI16: "<h", GU32: "<I", GI32: "<i",
       GF32: "<f", GBOOL: "<B", GU64: "<Q", GI64: "<q", GF64: "<d"}


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
    def u64(self):  return struct.unpack("<Q", self.raw(8))[0]
    def s(self):    return self.raw(self.u64()).decode("utf-8", "replace")

    def one(self, t):
        if t in (GSTR,):
            return self.s()
        if t == GARR:
            et = self.u32()
            n = self.u64()
            off = self.f.tell()
            vals = [self.one(et) for _ in range(n)]
            return (et, n, off, vals)
        return struct.unpack(FMT[t], self.raw(SSZ[t]))[0]


def main():
    src, dst = sys.argv[1], sys.argv[2]
    edits = {}
    for a in sys.argv[3:]:
        k, v = a.split("=", 1)
        edits[k] = v

    if not os.path.exists(dst):
        print("copying %s -> %s" % (src, dst))
        shutil.copyfile(src, dst)

    with open(dst, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF"
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        print("GGUF v%d, %d tensors, %d kv" % (ver, n_tensors, n_kv))
        found = {}
        for _ in range(n_kv):
            key = R(f).s()
            r = R(f)
            t = r.u32()
            off = f.tell()
            val = r.one(t)
            if key in edits:
                found[key] = (t, off, val)
                print("  %s: type=%d offset=%d value=%s" % (
                    key, t, off, val[3] if t == GARR else val))

    for key, raw in edits.items():
        if key not in found:
            sys.exit("key not present in the source GGUF: %s" % key)
        t, off, val = found[key]
        if t == GARR:
            et, n, voff, vals = val
            items = [x for x in raw.strip("[]").split(",") if x != ""]
            assert len(items) == n, "array length must stay %d, got %d" % (n, len(items))
            payload = b"".join(struct.pack(FMT[et], int(float(x))) for x in items)
        elif t in (GSTR,):
            sys.exit("refusing to resize a string value: %s" % key)
        else:
            num = float(raw)
            payload = struct.pack(FMT[t], int(num) if t != GF32 else num) \
                if t != GBOOL else struct.pack("<B", 1 if num else 0)
        with open(dst, "r+b") as f:
            base = off if t != GARR else voff
            f.seek(base)
            f.write(payload)
            print("patched %-40s -> %s (%d bytes at %d)" % (key, payload.hex(), len(payload), base))
    print("done:", dst)


if __name__ == "__main__":
    main()
