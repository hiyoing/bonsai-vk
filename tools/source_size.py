#!/usr/bin/env python3
"""tools/source_size.py - the number behind every "how big is the engine" claim.

WHY THIS EXISTS: README.md, README_zh.md, BENCHMARK.md and HANDOVER.md all quote a source size, and
an honest one has to be recomputed whenever a header lands. Three different values were sitting in
three documents last release. This script is now the single producer of those numbers; the docs quote
its output and name it as the method.

THREE DEFINITIONS, printed for every group, because "374 KiB" meant different things in different
files:
  raw                  bytes on disk, as written (comments included)
  de-commented         // line comments and /* */ block comments removed, whitespace KEPT - this is
                       the figure a reader scrolling the source sees
  whitespace-stripped  comments AND every whitespace run removed - the code volume itself, the
                       number to quote if you want "how much did a human type"

GROUPS, reported separately because they are different promises:
  host     bonsai_vk.cpp + the 11 headers it includes       (the C++17 half)
  shaders  shaders/*.comp + shaders/*.glslh                 (the Vulkan GLSL half)
  tools    align_*.cpp, bonsai_fleet.cpp, gpu_fleet.h, shared_probe.cpp, plan_probe.cpp, attn_vk.h
           (the measurement drivers that produce runs/*.log - not needed to run the model)

Deterministic: same tree in, same bytes out. Exit 1 if the declared file set moved.
"""
import glob
import os
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RK = ROOT / "runtime_vk"

BLOCK = re.compile(r"/\*.*?\*/", re.S)
LINE = re.compile(r"//[^\n]*")

HOST = ["bonsai_vk.cpp", "bonsai_platform.h", "bonsai_model.h", "bonsai_weights.h", "bonsai_forward.h",
        "bonsai_gguf.h", "bonsai_tok.h", "bonsai_chat.h", "kv_guard.h", "f16_rne_host.h", "ptq_decode.h",
        "bonsai_http.h"]
TOOLS = ["align_tokenizer.cpp", "align_dequant.cpp", "align_f16.cpp", "align_attn.cpp",
         "align_27b.cpp", "bonsai_fleet.cpp", "gpu_fleet.h", "shared_probe.cpp", "plan_probe.cpp",
         "attn_vk.h"]


def measure(path):
    raw = path.read_bytes()
    txt = raw.decode("utf-8", "replace")
    docom = LINE.sub("", BLOCK.sub("", txt))
    nows = re.sub(r"\s+", "", docom)
    return (len(raw), len(docom.encode("utf-8")), len(nows.encode("utf-8")))


def group(paths, label):
    tot = [0, 0, 0]
    for p in paths:
        m = measure(Path(p))
        for i in range(3):
            tot[i] += m[i]
    print(f"  {label:<10} {len(paths):>3} files | raw {tot[0]:>9,} B = {tot[0]/1024:5.0f} KiB"
          f" | de-commented {tot[1]:>9,} B = {tot[1]/1024:5.0f} KiB"
          f" | whitespace-stripped {tot[2]:>9,} B = {tot[2]/1024:5.0f} KiB")
    return tot


def main():
    missing = [str(RK / f) for f in HOST + TOOLS if not (RK / f).exists()]
    if missing:
        print("MISSING FILES - the file set moved. Fix this script; do not hand-edit the README:")
        for m in missing:
            print("   ", m)
        return 1
    comp = sorted(glob.glob(str(RK / "shaders/*.comp")))
    glslh = sorted(glob.glob(str(RK / "shaders/*.glslh")))
    print("GROUPS (all paths relative to runtime_vk/)")
    h = group([str(RK / f) for f in HOST], "host")
    s = group(comp + glslh, "shaders")
    t = group([str(RK / f) for f in TOOLS], "tools")
    eng = [h[i] + s[i] for i in range(3)]
    whole = [eng[i] + t[i] for i in range(3)]
    print("SUMS")
    for name, v in (("engine = host + shaders", eng), ("whole tree = engine + tools", whole)):
        print(f"  {name:<28} raw {v[0]:>9,} B = {v[0]/1024:5.0f} KiB | de-commented "
              f"{v[1]:>9,} B = {v[1]/1024:5.0f} KiB | whitespace-stripped {v[2]:>9,} B = "
              f"{v[2]/1024:5.0f} KiB")
    spv = sorted(glob.glob(str(RK / "shaders/*.spv")))
    exe = RK / "bonsai_vk.exe"
    n = len(spv)
    tot_spv = sum(os.path.getsize(p) for p in spv)
    print("SHIPPED ARTIFACTS")
    print(f"  bonsai_vk.exe                {os.path.getsize(exe):,} B = "
          f"{os.path.getsize(exe)/1024:.1f} KiB")
    print(f"  shaders/*.spv                {n} modules, {tot_spv:,} B = {tot_spv/1024:.0f} KiB")
    ui = RK / "index.html"
    if ui.exists():
        print(f"  index.html (WebUI)           {os.path.getsize(ui):,} B = "
              f"{os.path.getsize(ui)/1024:.1f} KiB (single file, zero dependencies; built by webui/build.mjs)")
    print(f"  third-party dependencies     0 (Windows link flags -lvulkan-1 -lpdh -ldxgi -lws2_32 "
          "-lshell32: OS loader + OS import libs; Linux needs -lvulkan -lpthread only)")
    print(f"  shader sources               {len(comp)} .comp + {len(glslh)} .glslh")
    return 0


if __name__ == "__main__":
    sys.exit(main())
