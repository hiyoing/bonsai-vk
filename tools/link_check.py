#!/usr/bin/env python3
"""tools/link_check.py - relative links must resolve in EVERY context a document ships in.

WHY THIS EXISTS: the release package at dist/bonsai-vk-0.1.0-win64/ carries copies of README.md and
README_zh.md. Those READMEs link to source paths such as `runtime_vk/gpu_fleet.h`. In the repository
root every one of those links resolves; inside the package - which is a binary package - it did not:
5 broken links for whoever unpacks it. Nobody noticed until this script was written, because nobody
opens a package and clicks a link in a README.

WHAT IT DOES: for each (document, directory) pair it strips fenced code blocks (so a `a | b` example
inside backticks is never mistaken for a table cell or a link), collects every relative Markdown
link target, drops http(s) ones, and checks os.path.exists against that directory. Anchors and query
strings are split off first.

Exit code: 0 clean, 1 if any target is missing. Deterministic - same tree in, same list out.
"""
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"```.*?```", re.S)
LINK = re.compile(r"\]\(([^)\s]+)")

# (directory the links are resolved against, documents that live there)
CONTEXTS = [
    (ROOT, ["README.md", "README_zh.md", "HANDOVER.md", "BENCHMARK.md", "ROADMAP.md",
            "PROJECT_BONSAI_FINAL_REPORT.md", "PROJECT_BONSAI_FULL_DOSSIER.md", "FAITHFUL_REPRO.md",
            "KV_P0P1P2_MEASURED.md",
            "ATTENTION_ALIGN_FINAL.md", "END_TO_END_MEASURED.md", "DEEPINFER_EVAL.md",
            "PROJECT_BONSAI_WEBUI_REPORT.md"]),
    (ROOT / "dist" / "bonsai-vk-0.1.0-win64", ["README.md", "README_zh.md", "HANDOVER.md",
                                               "BENCHMARK.md", "ROADMAP.md",
                                               "PROJECT_BONSAI_FINAL_REPORT.md",
                                               "PROJECT_BONSAI_FULL_DOSSIER.md",
                                               "PROJECT_BONSAI_WEBUI_REPORT.md"]),
]


def missing_for(base: Path, name: str):
    text = (base / name).read_text(encoding="utf-8")
    text = FENCE.sub("", text)
    out = []
    for target in LINK.findall(text):
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        path = target.split("#", 1)[0].split("?", 1)[0]
        if not path:
            continue
        if not os.path.exists(base / path):
            out.append(path)
    return sorted(set(out))


def main() -> int:
    bad = 0
    for base, docs in CONTEXTS:
        if not base.is_dir():
            print(f"{base}: ABSENT (skipped - package not assembled yet)")
            continue
        for doc in docs:
            if not (base / doc).is_file():
                print(f"{base / doc}: MISSING DOCUMENT")
                bad += 1
                continue
            miss = missing_for(base, doc)
            rel = str(base.relative_to(ROOT)) if base != ROOT else "."
            label = f"{rel}/{doc}"
            if miss:
                bad += len(miss)
                print(f"{label:58s} BROKEN {len(miss)}: " + ", ".join(miss))
            else:
                print(f"{label:58s} ok")
    print(f"TOTAL BROKEN {bad}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
