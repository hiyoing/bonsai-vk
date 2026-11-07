#!/usr/bin/env python3
"""oracle_tokenize.py - run the REFERENCE tokenizer (llama-tokenize.exe from the PrismML build) over
runs/tok_probes.txt and write runs/tok_ref.jsonl, one line per probe:

    {"i": 1, "bytes": 33, "text": "...", "ids": [95826, 110827, ...]}

The probe text never touches a shell: it is written to a temp file as raw bytes and passed with -f,
which is exactly what our C++ side is fed too.
Override the input/output pair with the env vars PROBES (file name inside runs/) and REFOUT.
"""
import json, os, pathlib, re, subprocess, sys

ROOT = pathlib.Path(__file__).resolve().parent.parent
EXE = pathlib.Path(r"prism-vk/prism/llama-tokenize.exe")
MODEL = pathlib.Path(r"models/Ternary-Bonsai-2-9B-PTQ1_0.gguf")
PROBES = ROOT / "runs" / os.environ.get("PROBES", "tok_probes.txt")
OUT = ROOT / "runs" / os.environ.get("REFOUT", "tok_ref.jsonl")
EXTRA = [a for a in sys.argv[1:]]        # extra probes given on the command line


def unescape(s):
    out, i = [], 0
    while i < len(s):
        if s[i] == "\\" and i + 1 < len(s):
            c = s[i + 1]
            if c == "n":
                out.append("\n"); i += 2; continue
            if c == "t":
                out.append("\t"); i += 2; continue
            if c == "\\":
                out.append("\\"); i += 2; continue
        out.append(s[i]); i += 1
    return "".join(out)


def run_oracle(data: bytes, tmp):
    tmp.write_bytes(data)
    p = subprocess.run([str(EXE), "-m", str(MODEL), "-f", str(tmp), "--ids", "--no-bos"],
                       capture_output=True, text=True, encoding="utf-8", errors="replace")
    m = re.search(r"\[([0-9,\s]*)\]", p.stdout)
    if not m:
        return None, p.stdout + p.stderr
    return [int(x) for x in m.group(1).replace(" ", "").split(",") if x], ""


def main():
    probes = [unescape(l) for l in PROBES.read_text(encoding="utf-8").split("\n") if l != ""]
    probes += [unescape(a) for a in EXTRA]
    tmp = pathlib.Path(ROOT / "runs" / "_tok_probe.bin")
    rows = []
    with OUT.open("w", encoding="utf-8") as o:
        for i, pr in enumerate(probes, 1):
            b = pr.encode("utf-8")
            ids, err = run_oracle(b, tmp)
            if ids is None:
                print("probe %d FAILED: %s" % (i, err.strip()[:300]))
                continue
            rec = {"i": i, "bytes": len(b), "text": pr, "ids": ids}
            o.write(json.dumps(rec, ensure_ascii=False) + "\n")
            rows.append(rec)
            print("%3d n=%-3d %s" % (i, len(ids), ids))
    print("wrote %d reference lines -> %s" % (len(rows), OUT))


if __name__ == "__main__":
    main()
