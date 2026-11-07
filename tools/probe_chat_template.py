#!/usr/bin/env python3
"""Read-only probe: pull the exact chat template and the special-token table out of the GGUF
header, so the engine implements what the checkpoint ships instead of what Qwen is believed to do.
The model file is opened 'rb' only; nothing is ever written back."""
import struct, sys, json

# This box's console codepage is cp936. Without this, printing the template's U+0120 raises
# UnicodeEncodeError at the LAST print statement and the redirect file comes back as 5 bytes
# ('ab' from the previous run's name, i.e. nothing) instead of the 19 KB dump we need.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")
except AttributeError:
    pass

PATH = "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf"

# gguf type enum (llama.cpp ggufElementType), verified against the header parse below
U8, I8, U16, I16, U32, I32, F32, BOOL, STR, ARR, U64, I64, F64, F16, BF16 = range(15)
FIXED = {U8: "<B", I8: "<b", U16: "<H", I16: "<h", U32: "<I", I32: "<i",
         F32: "<f", F64: "<d", F16: "<e", BF16: "<e", BOOL: "<B"}


class R:
    def __init__(self, f):
        self.f = f

    def u32(self):
        return struct.unpack("<I", self.f.read(4))[0]

    def i64(self):
        return struct.unpack("<q", self.f.read(8))[0]

    def s(self):
        # GGUF v2/v3 string length is uint64, matching our own working reader bonsai_gguf.h:65
        return self.f.read(self.u64())

    def u64(self):
        return struct.unpack("<Q", self.f.read(8))[0]

    def val(self, t):
        if t == STR:
            return self.s().decode("utf-8", "replace")
        if t == ARR:
            et = self.u32(); n = self.u64()
            return [self.val(et) for _ in range(n)]
        if t in FIXED:
            fmt = FIXED[t]
            return struct.unpack(fmt, self.f.read(struct.calcsize(fmt)))[0]
        if t == U64:
            return struct.unpack("<Q", self.f.read(8))[0]
        if t == I64:
            return struct.unpack("<q", self.f.read(8))[0]
        raise ValueError("unknown gguf type %d at offset %d" % (t, self.f.tell()))


def main():
    with open(PATH, "rb") as f:
        assert f.read(4) == b"GGUF", "not a GGUF"
        ver = struct.unpack("<I", f.read(4))[0]
        n_tensors = struct.unpack("<Q", f.read(8))[0]
        n_kv = struct.unpack("<Q", f.read(8))[0]
        r = R(f)
        md = {}
        for _ in range(n_kv):
            k = r.s().decode("utf-8", "replace")
            md[k] = r.val(r.u32())
        print("version %d tensors %d kv %d" % (ver, n_tensors, n_kv))

        ids = md.get("tokenizer.ggml.tokens")
        tok2id = {t: i for i, t in enumerate(ids)} if isinstance(ids, list) else {}

        print("\n===== keys that smell like chat / tokens / thinking =====")
        for k in sorted(md):
            low = k.lower()
            if any(w in low for w in ("chat", "special", "added", "token", ".pre", "eos", "bos",
                                       "utf8", "byte", "template", "sep", "eot", "think")):
                v = md[k]
                if isinstance(v, str) and len(v) > 300:
                    print("%-44s = <string len %d, printed in full below>" % (k, len(v)))
                elif isinstance(v, list) and len(v) > 30:
                    print("%-44s = list[%d]" % (k, len(v)))
                else:
                    print("%-44s = %r" % (k, v))

        tpl = md.get("tokenizer.chat_template")
        print("\n===== tokenizer.chat_template :: repr =====")
        print(repr(tpl))
        print("\n===== tokenizer.chat_template :: raw =====")
        print(tpl)

        print("\n===== added_tokens (name -> id) =====")
        for t in (md.get("tokenizer.ggml.added_tokens") or []):
            print("  %-30r -> %s" % (t, tok2id.get(t, "NOT IN VOCAB")))
        print("\n===== tokenizer.ggml.token_type list (1=CONTROL etc) =====")
        tt = md.get("tokenizer.ggml.token_type")
        if isinstance(tt, list):
            from collections import Counter
            print("  size %d, type histogram %s" % (len(tt), Counter(tt)))
            for i, ty in enumerate(tt):
                if ty == 3:  # CONTROL in llama.cpp
                    print("  control: %-30r -> id %d" % (ids[i], i))

        print("\nvocab=%s eos=%r bos=%r add_bos=%r byte_fallback=%r pre=%r" % (
            len(ids) if isinstance(ids, list) else "?",
            md.get("tokenizer.ggml.eos_token_id"), md.get("tokenizer.ggml.bos_token_id"),
            md.get("tokenizer.ggml.add_bos_token"), md.get("tokenizer.ggml.byte_fallback"),
            md.get("tokenizer.ggml.pre")))
        print("arch=%r name=%r" % (md.get("general.architecture"), md.get("general.name")))

        print("\n===== any value mentioning 'think' =====")
        for k, v in md.items():
            s = str(v)
            if "think" in s.lower():
                print("  %s = %.300s" % (k, s))

        out = {k: (v if not isinstance(v, list) or len(v) < 64 else "<list %d>" % len(v))
               for k, v in md.items()}
        json.dump(out, open("runs/gguf_metadata_dump.json", "w"), ensure_ascii=False, indent=1,
                  default=str)
        print("\nfull metadata -> runs/gguf_metadata_dump.json")


main()
