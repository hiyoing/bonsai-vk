# Project Bonsai — announcement drafts

Every number below is the measured one from [`BENCHMARK.md`](BENCHMARK.md), and each has a log in `runs/`.
Anything not measured on this RX 550X is not claimed. Keep it that way when editing: if a figure changes,
it changes here too, from `tools/source_size.py` or a new log — never from memory.

---

## X / Twitter — single post (the one to send if you send only one; 280-char limit respected)

> 416 KB of hand-written C++17 + raw Vulkan, built for one thing: 1.58-bit ternary models.
> A 9B ternary checkpoint on a 4 GB RX 550X (2017 e-waste) does 14.16 tok/s; the reference engine on
> the same card does 4.417. 3.21x, 0 dependencies. #LocalLLaMA #Vulkan #AMD

## X / Twitter — thread (every tweet is <= 280 characters; the checker is at the bottom of this file)

**1/**
I wrote a 416 KB inference engine in raw C++17 + Vulkan, for one thing only: 1.58-bit ternary models.
No framework, no kernel lib, no SDK, no admin rights.
It runs a 9B ternary checkpoint on a 4 GB AMD RX 550X — 2017 laptop e-waste — at 14.16 tok/s.
#LocalLLaMA #Vulkan #AMD

**2/**
Same checkpoint, same card, reference engine: 4.417 t/s. Bonsai: 14.16 t/s = 3.21x.
That did not come from a better GPU. It came from refusing to be general: no compute graph, no op
registry, no tensor library. The engine is the shape of this checkpoint family.

**3/**
What 1.58-bit-only buys: PTQ1_0 packs 5 trits per byte, one fp16 scale per 128 weights.
Dequant = 5 bitfieldExtracts + a sign table, so it fits in registers and is INLINED into the GEMV
instead of being a pass over memory. 177.0 Gop/s on a card with no DP4A and no matrix cores.

**4/**
Speed that changes the numbers is a bug, not an optimization.
Tokenizer: 39/39 IDs identical to the official tokenizer, CJK included.
Dequant: max|err| = 0.00000 vs the official kernel running on this card, and bitwise identical over
491,520 synthetic weights.

**5/**
The attention row kernels come in 4 lineages: narrow, wide, tiled, GQA-merged. Each is compared
BIT BY BIT against the kernel it replaces, on the same KV bytes, all heads, all slots.
One differing bit = exit 6 = the engine refuses to generate. Bitwise different = a second model.

**6/**
Ternary checkpoints ship their own transform contract in metadata: prism.hadamard.* —
normalized-sylvester-walsh-hadamard, block 4096, 2 sign vectors, 201 rotated tensors, token_embd in
the INVERSE basis. Skip it and the output is garbage. First thing a ternary port gets wrong.

**7/**
Long context, measured on this card (fp16 K + q8_0 V = 24.50 KiB/token):
4K 9.41, 8K 6.93, 16K 4.57, 32K 2.67 tok/s.
Peak 3324 MiB device-local, WDDM Shared-Usage growth +0.00: no spill propping it up.
Arena ceiling ~39.2K tokens, printed before the allocation.

**8/**
One honest number: ms/token = 69.97 + 0.01408 x positions. The constant — 70 ms, 7/8 of a short
token — belongs to the ternary GEMV and the 24 recurrent layers. No attention trick touches it.
Long context is a slope problem; short context is a constant problem.

**9/**
The GQA merge (4 query heads share K, 16 workgroups to 4) measured +19% at 4K, +45% at 32K, 0% at
short. The V-side merge I built, measured and switched OFF: -11% at 4K, -25% at 32K,
+2.5% at short. Bytes were never the cost; the gather shape and workgroup count were.

**10/**
Vulkan lesson 1: VkBufferCreateInfo::pQueueFamilyIndices is read during vkCreateBuffer and copied
nowhere. Declare the array inside the if that sets sharing mode and the driver points at a dead stack
frame: survives -O0, SIGSEGVs in amdvlk64.dll at -O2, looks like a driver bug.

**11/**
Vulkan lesson 2: a fence created with VK_FENCE_CREATE_SIGNALED_BIT is already signalled, so the wait
witnesses nothing. Same 8 MiB copy: 0.8 us "fence" with a 1195.2 us queue drain behind it; with
flags = 0: 1204.1 us + 0.4 us tail. Spec behaviour, not a driver bug.

**12/**
A 5.9 GB ternary 27B will never fit in 4 GB, so: layer-pipelined multi-GPU, hidden state over a
pinned ring + async DMA on the transfer-only queue. Measured here: an 8 MiB leg is 6.96 GB/s = 97%
of this driver's ceiling, batched 7.13, 0 byte mismatches.

**13/**
Honest boundary on that rig: probe sees 1 compute-capable device of 1, so no card-to-card hop was
taken — the tested leg is D2H to host to H2D. Second path: iGPU zero-copy, arena on a
HOST_VISIBLE|DEVICE_LOCAL heap, staging buffer deleted. No pool to exhaust.

**14/**
And the 27B shape is already gated: 64 blocks, only 16 own a KV cache, 49.000 KiB/token charged,
while the forbidden "KV on every block" allocator prices at 4.0x. Enforced in code, printed as
bytes, and four negative controls that must refuse and do.

**15/**
The repo is a pile of markdown and logs on purpose: every claim has a filename behind it, and every
byte count comes from one script — 416 KB de-commented source, 539 KB exe, 52 SPIR-V modules, 0 deps.

**16/**
Details in BENCHMARK.md, FAITHFUL_REPRO.md, KV_P0P1P2_MEASURED.md. Built on a $50 GPU from 2017 with
a free toolchain, no admin rights, and nothing written to C:. #LocalLLaMA #Vulkan #AMD

**Length check** (run this before posting; it re-measures the file rather than trusting it):

```bash
python - <<'PY'
import re, sys
t = open('ANNOUNCEMENTS.md', encoding='utf-8').read()
th = t.split('## X / Twitter — thread')[1].split('**Length check**')[0]
bad = 0
for n, body in re.findall(r'\*\*(\d+)/\*\*\n(.*?)(?=\n\n\*\*|\Z)', th, re.S):
    s = ' '.join(body.split())
    if len(s) > 280:
        bad += 1; print('OVER', n, len(s))
print('tweets over the 280-char limit:', bad)
PY
```

---

## Reddit (r/LocalLLaMA, long post)

**Title:** 14.16 tok/s on a 9B ternary model with a 4 GB RX 550X — a 416 KB dependency-free Vulkan engine,
and what "1.58-bit only" actually buys you

I spent a while writing a from-scratch inference runtime for one specific kind of model: the ternary
PTQ1_0 checkpoints (1.58-bit, 5 trits per byte, one fp16 scale per 128 weights). The interesting
constraint was the hardware: an AMD RX 550X with 4 GB of VRAM, which is a 2017 laptop GPU, plus a
zero-budget rule (MinGW-w64, Vulkan-Headers, the driver's own `vulkan-1.dll`; no SDK installer, no
Visual Studio, no admin rights, nothing written to `C:`).

The result: **14.16 tok/s** decoding a 9B ternary model end to end on that card
(`runs/regression_v14.log`: 48 tokens in 3389.5 ms, 70.6 ms/token steady state), versus the **4.417 t/s**
the reference engine produces for the same checkpoint on the same card — **3.21×**. The whole engine is
**416 KiB of source with comments stripped** (696 KiB as written, because the comments record *why*),
building to a **539 KiB exe + 52 SPIR-V modules (300 KiB)** with three link flags, all supplied by the
operating system. `tools/source_size.py` is the only producer of those numbers. Here is what I learned,
including the parts that came out negative.

**1. Specialising on the format is the whole win.** Because a PTQ1_0 block dequantises into registers
(5 `bitfieldExtract`s and a sign table), the ternary operand is decoded *inside* the GEMV loop rather
than in a separate pass. There is no fp16 arithmetic available at all — this driver reports
`shaderFloat16 = 0` — so half-precision conversion is done with bit tricks in a header, and every MAC is
scalar fp32. gfx803 has no DP4A and no matrix cores. Measured on the projections: up to 177.0 Gop/s for
M=12288/K=4096, 44.2 GB/s of packed weight bytes, and a weighted model-level mix of 147.3 Gop/s.

**2. Fidelity gates found every real bug, and they were all plumbing bugs.** I compared against the
reference layer by layer instead of eyeballing the output: the tokenizer matches ID-by-ID on 39 probes
(including CJK); the dequant path matches the official `dequant_ptq1_0` kernel *run on the same card*
with `max|err| = 0.00000` over 128,000 real weights and is bitwise identical over 491,520 synthetic
weights covering zero/subnormal/negative-scale/non-finite; the attention chain matches an fp32 mirror
to within 0.26× the rounding cost of the reference's own fp16 KV cache. Two of my root causes were
humiliating: a missing `vkCmdPipelineBarrier`, and reading `attn_q.weight`'s 8192 outputs as two halves
when they are actually interleaved 512-wide groups of `[q | gate]` per head.

**3. "Fast but different numbers" is not an optimisation.** The attention row kernels now exist in four
lineages (a lane-per-key narrow kernel, a strided wide kernel, a tiled kernel with no compile-time row
limit, and a GQA-merged kernel). All of them are compared **bitwise** against the one they replace, on
the same KV bytes, over every head and every slot of the first and last attention layer. One differing
bit ⇒ exit code 6 ⇒ the engine refuses to generate. That is why I was willing to touch a validated
kernel at all.

**4. The GQA result, with the negative half included and corrected.** Grouped-query attention means 4
query heads share a K vector, so the K gather is fetched four times per position. Merging heads into one
workgroup (16 workgroups → 4) cuts the gather transactions and the K unpack count 4×. Paired against the
incumbent kernel in the *same binary*, inside one thermal window (`runs/gqa_pair_v1_summary.txt`):

- 4 K tokens: 126.9 → **106.3 ms/token** (7.88 → **9.41 tok/s**, +19.4 %)
- 8 K: 187.4 → 144.3 (5.34 → 6.93) · 16 K: 303.6 → 218.6 (3.29 → 4.57) · 32 K: 542.5 → **374.8** (1.84 → **2.67**, +45.1 %)
- short context (cap 256): 73.4 → 72.7 ms/token, and the same control re-measured last read 72.3 — so the
  short-context "gain" is inside the 1.5 % drift band. The honest statement is "free, not faster": the
  default build measures 14.16 tok/s against a 14.05–14.20 frozen baseline, i.e. 0 % regression.

The merge sits behind a runtime switch (`BONSAI_GQA_OPT=0` restores the exact reference path), and behind
a *second* switch (`BONSAI_GQA_PART`) whose only purpose was attribution. That switch is how I found the
V-side merge was not worth shipping — and my first read of it was too strong. The unpaired pass said "PV
merge is a regression" (184.2 vs 128.7 ms/token when run alone); the paired pass says it is
**regime-dependent**: adding PV merge on top of the QK merge costs 11 % at 4 K (106.3 → 118.1) and 25 %
at 32 K (374.8 → 470.4), but *saves* 2.5 % at cap 256 (72.7 → 70.9). The shape of the explanation: the K
operand is a genuine gather with ~8× sector waste, so merging removes transactions; the V operand is read
contiguously and already shared inside a quad, so merging only removes ALU work while cutting resident
workgroups 16 → 4 — on a 12-CU card that trade turns positive at short context and negative at long. So
the shipped default merges the QK half only. I would have sworn the opposite before measuring it.

**5. The honest ceiling.** A least-squares fit over measured points gives `ms/token = 69.97 + 0.01408 ×
positions` for the tiled kernel, and `67.17 + 0.00936 × positions` with the QK merge on — the same floor,
a slope cut by 33.5 %. The constant, 70 ms, i.e. 7/8 of a short token, belongs to the ternary GEMV and
the 24 recurrent layers, and no amount of attention cleverness touches it. Long context is a slope
problem; short context is a constant problem; these need different surgeries and I wasted effort assuming
they were the same one.

**6. Memory behaviour on WDDM, measured rather than assumed.** 8 attention layers with fp16 K and q8_0 V
cost 24.50 KiB/token; the safe ceiling on 4 GB is ≈ 39.2 K tokens, and 40 K is already out of band. At a
32 K arena the peak is 3324 MiB device-local with WDDM `Shared Usage` growth of **+0.00** across the whole
run, so the long-context numbers above are not being propped up by a spill. A related finding: my first
spill detector keyed off `VK_MEMORY_PROPERTY_DEDICATED_BIT` usage, which is the wrong counter under WDDM;
and the biggest per-token memory growth was not the KV cache at all, it was allocating a new command
buffer per token (0.37 MiB each, and `vkFreeCommandBuffers` returns 0.00 of it). Also, do not confuse the
two "4 GB walls" that circulate in my repo: `-c 49152` passing / `53248` failing is the *reference
llama.cpp fork's* allocator budget, while ≈ 39.2 K is *my* arena arithmetic. Different engines, different
numbers, and neither has been driven past 32 K live.

**7. Two Vulkan lessons that cost real hours.** (a) `VkBufferCreateInfo::pQueueFamilyIndices` is read by
`vkCreateBuffer` and copied nowhere, so declaring the family array inside the `if` that sets the sharing
mode leaves the driver pointing at a dead stack frame — it survives `-O0`, SIGSEGVs inside
`amdvlk64.dll` at `-O2`, and the backtrace looks exactly like a driver bug. Four call sites, one of them
in my own staging-buffer path. (b) A fence created with `VK_FENCE_CREATE_SIGNALED_BIT` is already
signalled when the wait runs, so `vkWaitForFences` returns without witnessing the submit that should have
armed it. Same 8 MiB copy, same queue: created-SIGNALED = fence 0.8 µs with a 1195.2 µs queue drain
behind it; `flags = 0` = fence 1204.1 µs with a 0.4 µs tail. Before I wrote that control I believed two
false things: an impossible 56.67 GB/s "batched" rate, and that the hop was sync-dominated when it is
DMA-bound. Spec behaviour, not a driver bug — but it produced two wrong conclusions in my own logs.

**8. What is next: a 27B without buying hardware.** A 5.9 GB ternary 27B will never fit in 4 GB, so the
two paths are about not needing it to. (a) Layer-split pipeline parallelism across cheap cards, with the
hidden state moving through a **pinned ring buffer** and async DMA on the **transfer-only queue family**.
Measured on this host (`runs/fleet_ring_v4.log`): one fenced 8 MiB leg = 1204.6 µs = **6.96 GB/s**, 97 %
of the measured 7.15 GB/s ceiling for that queue; eight legs in one submit = 1176.4 µs each = **7.13 GB/s**;
0 byte mismatches on arrival; 2 back-pressure events at depth 4. The honest boundary: `probe` sees **1
compute-capable device of 1**, so the card-to-card hop is not exercised here — the tested path is D2H →
host → H2D. Two older, independent numbers make the design worth building anyway: SDMA runs at 7.12 GB/s
and does **not** discount concurrent compute, and the per-token host bubble on one card is 16.8 ms
(11.8 fence + 5.0 submit), which is the pipeline's free window. (b) An iGPU zero-copy path that puts the
weight arena on a `HOST_VISIBLE | DEVICE_LOCAL` heap and deletes the staging buffer entirely — an iGPU
has no dedicated pool to exhaust. (This discrete card exposes such heaps too, as a 256 MiB BAR window,
which is exactly why the branch is gated on `deviceType == INTEGRATED_GPU` and never taken here.) The 27B
is 64 blocks but only **16 own a KV cache**: `align_27b.exe` measures 49.000 KiB/token charged, 1568 MiB
of arena at cap 32 768, and prices the forbidden "KV on every block" allocator at **4.0×** — plus four
negative controls that must refuse and do. What is *not* proven is running the real checkpoint: this host
cannot load 5.9 GB, so the gates run against a synthetic 64-block plan built from the 9B's real header.

**Caveats, so nobody is surprised:** sampling is not yet aligned against the official sampler (text
quality is validated by the fidelity ladder and by eye, not by a sampler diff); prefill is still
quadratic — a 32 K prompt costs ≈ 2 hours on this card, integrated from the measured fit, and it
is the next surgery; 40 K/48 K contexts are computed from the budget formula, not run; and the engine
only loads this checkpoint family.

If anyone has run PTQ1_0 or QAT ternary weights on Polaris-era GCN, I would love to hear what I left on
the table — particularly anything about the LDS/SGPR budget on gfx803, since one of my "impossible"
assumptions (that a 32 KiB static shared allocation would not even build) turned out to be a measurement
away from a 19.6 % occupancy cost instead of a hard wall.
