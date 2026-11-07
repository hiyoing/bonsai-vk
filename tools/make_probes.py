#!/usr/bin/env python3
"""make_probes.py - writes runs/tok_probes.txt, one probe per line, UTF-8, no CR.

Backslash escapes (\n, \t, \\) are written as the two-character sequences and expanded by the C++
driver, so a probe can contain a real newline without breaking the line-based probe file.
"""
import pathlib
BS = chr(92)
NL = BS + "n"
TAB = BS + "t"

probes = [
    "用一句话解释三值模型。",
    "用一句话解释什么是三值模型",
    "Hello world, this is a test of the 2026 emergency number 123.",
    "三值模型 = ternary model (-1, 0, +1)",
    "它是指除了真和假之外，还包含第三种状态（如不确定）的逻辑判断模型。",
    "I don't know - 3.14 x 2 = 6.28",
    "中文English混排123abc",
    "Multiple   spaces" + TAB + "and" + TAB + "tabbed" + TAB,
    "Line1" + NL + "Line2" + NL + NL + "trailing" + NL,
    "请解释 KV cache 的显存占用：4096 tokens x 32 KiB = 128 MiB。",
    "The quick brown fox jumps over the lazy dog.",
    "100% sure, 50/50 maybe; 3.14, 2026.",
    "。！？，、；：（）《》【】",
    "三值模型是允许变量取 -1/0/+1 三种状态的量化方案。",
    "aaa   bbb ccc   ddd",
    "权重 1.75 bpw 三值(-1/0/+1)，group 128。",
    'He said: "use the runtime", then left.',
    "  leading and trailing spaces  ",
    "日本語のテキスト과 한국어 텍스트",
    "Price is $1,000.00 (USD) in 2026.",
    "Thinking Process:" + NL + NL + "1.  **Analyze the Request:**",
    "端到端 decode tok/s = 4.417 t/s（50 token 真跑）",
    "用",
    "一句话",
    "用一句话",
    "模型",
    "三值模型",
    "。用",
    " 用",
    "1234567890",
    "０１２３",
    "R5 3600 + RX 550X = gfx803",
]

out = pathlib.Path(__file__).resolve().parent.parent / "runs" / "tok_probes.txt"
out.write_text("\n".join(probes) + "\n", encoding="utf-8", newline="")
print("wrote %d probes, %d bytes -> %s" % (len(probes), out.stat().st_size, out))
