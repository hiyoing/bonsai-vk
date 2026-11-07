"""Read the FULL GGUF header (metadata + tensor info table) of the Bonsai 9B checkpoints.

Why: the first pass (1 MB window) proved the model is architecture=qwen35 with
full_attention_interval=4, i.e. a HYBRID linear-attention model, not the dense Qwen3 shape
STAGE2 assumed from tensor naming. It also cut off before block_count, which is the single
number every KV-cache / SSM-state budget in the DeepInfer comparison depends on.

The header layout is: magic, version, n_tensors, n_kv, [n_kv key/value pairs], then
n_tensors records of (name, n_dims, dims[], dtype, offset). Parsing the tensor table gives the
layer count, the per-layer shapes and the quantization actually used, straight from the file.
"""
import json
import struct
import sys
import urllib.parse
import urllib.request

try:                                            # the Windows console here is cp936; metadata strings
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")   # are UTF-8 and crash print()
except Exception:                               # noqa: BLE001
    pass

MIRROR = "https://hf-mirror.com"
TYPES = {0: ("uint8", "<B"), 1: ("int8", "<b"), 2: ("uint16", "<H"), 3: ("int16", "<h"),
         4: ("uint32", "<I"), 5: ("int32", "<i"), 6: ("float32", "<f"), 7: ("bool", "<B"),
         8: ("string", None), 9: ("array", None), 10: ("uint64", "<Q"), 11: ("int64", "<q"),
         12: ("float64", "<d")}


def get(url, rng=None, timeout=120):
    h = {"User-Agent": "bonsai-probe/1.0", "Accept-Encoding": "identity"}
    if rng:
        h["Range"] = rng
    req = urllib.request.Request(url, headers=h)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.read(), dict(r.headers)


def api(path):
    body, _ = get(MIRROR + path)
    return json.loads(body)


class R:
    def __init__(self, b):
        self.b, self.i = b, 0

    def raw(self, n):
        if self.i + n > len(self.b):
            raise EOFError("need %d more bytes at %d" % (n, self.i))
        v = self.b[self.i:self.i + n]
        self.i += n
        return v

    def val(self, t):
        name, fmt = TYPES[t]
        if name == "string":
            n = struct.unpack("<Q", self.raw(8))[0]
            return self.raw(n).decode("utf-8", "replace")
        if name == "array":
            et = struct.unpack("<I", self.raw(4))[0]
            n = struct.unpack("<Q", self.raw(8))[0]
            if TYPES[et][1] is None:
                return [self.val(et) for _ in range(n)]
            sz = struct.calcsize(TYPES[et][1])
            return [struct.unpack(TYPES[et][1], self.raw(sz))[0] for _ in range(n)]
        return struct.unpack(fmt, self.raw(struct.calcsize(fmt)))[0]


def parse(buf):
    assert buf[:4] == b"GGUF", "not a GGUF: %r" % buf[:4]
    ver, n_tensors, n_kv = struct.unpack("<IQQ", buf[4:24])
    r = R(buf[24:])
    meta = {}
    for _ in range(n_kv):
        k = r.val(8)
        t = struct.unpack("<I", r.raw(4))[0]
        meta[k] = r.val(t)
    tensors = []
    for _ in range(n_tensors):
        nm = r.val(8)
        nd = struct.unpack("<I", r.raw(4))[0]
        dims = [struct.unpack("<Q", r.raw(8))[0] for _ in range(nd)]
        dt, off = struct.unpack("<I", r.raw(4))[0], struct.unpack("<Q", r.raw(8))[0]
        tensors.append((nm, dims, dt, off))
    return ver, meta, tensors, r.i


def arch_prefix(meta):
    return meta.get("general.architecture", "?")


def report(url, label):
    print("\n======== %s ========" % label)
    print("  url: %s" % url)
    try:
        first, hdr = get(url, "bytes=0-15")
    except Exception as e:
        print("  !! range probe failed: %s" % e)
        return
    total = None
    cr = hdr.get("Content-Range", "")
    if "/" in cr:
        total = int(cr.split("/")[-1])
    print("  file size: %s MB" % ("%.1f" % (total / 1e6) if total else "?"))
    if first[:4] != b"GGUF":
        print("  !! not a GGUF: %r" % first[:64])
        return
    # 32 MB is enough for a header carrying a 65536-entry Hadamard sign array
    buf, _ = get(url, "bytes=0-%d" % (32 * 1024 * 1024 - 1))
    ver, meta, tensors, consumed = parse(buf)
    p = arch_prefix(meta)
    print("  GGUF v%d, %d tensors, %d metadata keys, header = %.2f MB" % (ver, len(tensors), len(meta), (24 + consumed) / 1e6))
    keep = [k for k in meta if not k.startswith("prism.hadamard.sign_values")]
    SKIP = ("tokenizer.ggml.chat_template", "tokenizer.ggml.pre", "tokenizer.ggml.token_type",
            "tokenizer.ggml.merges", "tokenizer.ggml.added_tokens", "tokenizer.ggml.tokens")
    for k in sorted(keep):
        if k in SKIP:
            print("    %-52s = <%d entries, skipped>" % (k, len(meta[k])))
            continue
        v = meta[k]
        if isinstance(v, str) and len(v) > 90:
            v = "%s...<%d chars>" % (v[:60].replace("\n", "\\n"), len(v))
        if isinstance(v, list) and len(v) > 10:
            v = "%s... (+%d)" % (str(v[:10])[:-1], len(v) - 10)
        print("    %-52s = %s" % (k, v))
    print("  prism.hadamard.sign_values: %d entries (skipped)" % len(meta.get("prism.hadamard.sign_values", [])))

    # layer count straight from the tensor names, not from a key we hope exists
    import re
    layers = set()
    for nm, dims, dt, off in tensors:
        m = re.match(r"blk\.(\d+)\.", nm)
        if m:
            layers.add(int(m.group(1)))
    print("  == tensor table: %d layers (blk.0..blk.%d) ==" % (len(layers), max(layers) if layers else -1))
    per_kind = {}
    for nm, dims, dt, off in tensors:
        m = re.match(r"blk\.\d+\.(.+)$", nm)
        key = m.group(1) if m else nm
        per_kind.setdefault(key, []).append((dims, dt))
    for key in sorted(per_kind):
        ex = per_kind[key]
        print("    %-34s x%-3d shape %-28s dtype=%d" % (key, len(ex), str(ex[0][0]), ex[0][1]))
    # the state/KV question answered structurally: which per-layer tensors are NOT weights
    ssm = [k for k in per_kind if "ssm" in k or "conv" in k or "gate" in k or "delta" in k]
    print("  == recurrent/linear-attention tensors per layer: %s ==" % ssm)
    return meta, tensors


def find():
    seen = []
    for q in ("Ternary-Bonsai-2-9B-PTQ1_0", "Ternary-Bonsai-2-9B", "Bonsai-2-9B", "qwen35 ternary"):
        try:
            d = api("/api/models?search=" + urllib.parse.quote(q, safe=""))
        except Exception as e:
            print("search %r failed: %s" % (q, e))
            continue
        for m in (d if isinstance(d, list) else d.get("models", [])):
            rid = m.get("id")
            if rid and rid not in seen:
                seen.append(rid)
                print("  hit: %-58s downloads=%s" % (rid, m.get("downloads")))
    return seen


def main():
    # python probe_gguf_meta.py --only <repo-id>/<file.gguf>   -> skip discovery, probe one file.
    # repo ids stay un-percent-encoded (only the filename is quoted); without --only behave as before.
    argv = sys.argv[1:]
    if argv[:1] == ["--only"]:
        if len(argv) < 2 or "/" not in argv[1]:
            sys.exit("usage: probe_gguf_meta.py --only <repo-id>/<file.gguf>")
        rid, _, gf = argv[1].rpartition("/")
        url = "%s/%s/resolve/main/%s" % (MIRROR, rid, urllib.parse.quote(gf, safe="/"))
        report(url, "%s / %s" % (rid, gf))
        return
    repos = find()
    if not repos:
        sys.exit("nothing found")
    done = 0
    for rid in repos:
        try:
            d = api("/api/models/" + rid)
        except Exception as e:
            print("  %s: %s" % (rid, e))
            continue
        ggs = [f["rfilename"] for f in d.get("siblings", []) if f["rfilename"].endswith(".gguf")]
        print("\n%s -> gguf files: %s" % (rid, ggs))
        for gf in ggs:
            if "Q8_0" in gf or "PTQ1_0" in gf or len(ggs) == 1:
                url = "%s/%s/resolve/main/%s" % (MIRROR, rid, urllib.parse.quote(gf, safe="/"))
                report(url, "%s / %s" % (rid, gf))
                done += 1
                if done >= 2:
                    break
        if done >= 2:
            break


if __name__ == "__main__":
    main()
