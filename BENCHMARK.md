# Project Bonsai — BENCHMARK

Every number in this file is either **MEASURED** (a log in `runs/` on the machine named below) or
**DERIVED** (arithmetic from measured numbers, with the arithmetic shown). Nothing is a target, an
estimate from another card, or a vendor figure. The audit standard is that a reader can find the line
that produced the number.

```
Test machine   : AMD RX 550X 4 GB (gfx803 / Polaris 12 / wave64), Ryzen 5 3600, 16 GB DDR4,
                 Windows 10/11 WDDM, Vulkan 1.3 loader over the driver's vulkan-1.dll
Toolchain      : MinGW-w64 g++ 16.2 (UCRT) + Vulkan-Headers 1.4.309 + glslang 16.6.0, no SDK,
                 no Visual Studio, no administrator rights, nothing written to C: (build_mingw.bat)
Model          : Ternary-Bonsai-2-9B-PTQ1_0.gguf  (1.75 bpw ternary, group 128, ~2.41 GB on disk)
Rule           : one GPU process at a time; tools/clear_field.ps1 before every timing run
```

---

## 1. Engine size and shape (MEASURED, recomputed at every release)

| Quantity | Value | How it was obtained |
|---|---|---|
| Hand-written source, comments and blanks stripped | **416 KiB** = 425 944 B | `python tools/source_size.py` over `bonsai_vk.cpp` + 10 host headers + 52 `shaders/*.comp` + 5 `shaders/*.glslh`, after stripping `//` and `/* */` |
| Same source as written (the value that makes the code reviewable) | **696 KiB** = 712 716 B | raw size of the same file set |
| Same source with whitespace also removed | **279 KiB** = 285 377 B | third metric of the same script |
| — split: C++ host / GLSL | **352 KiB / 64 KiB** de-commented | the script prints the three groups separately (host 10 files, shaders 57 files, measurement tools 10 files) |
| Measurement drivers, *not* needed to run a model | **310 KiB** de-commented | the `tools` group: `align_*.cpp`, `bonsai_fleet.cpp`, `gpu_fleet.h`, probe harnesses |
| Single binary | **539 KiB** = 552 178 B | `bonsai_vk.exe` on disk |
| SPIR-V modules shipped | **300 KiB / 52 modules** = 307 432 B | sum of `shaders/*.spv` |
| Third-party dependencies | **0** | the only link flags are `-lvulkan-1 -lpdh -ldxgi`; all three are the OS loader / OS import libraries |

`tools/source_size.py` is the **only** producer of every byte count in this repository — README,
README_zh and this file all quote it rather than a hand-copied number. It lists its own file set and
exits non-zero if a file moved, so a stale size claim fails loudly instead of shipping.

The engine is one translation unit plus headers, one `VkBuffer` arena, one submit per token, and no
allocator other than the region table in `bonsai_forward.h`.

## 2. Kernel rate on the ternary path (MEASURED)

| Shape (hidden 4096 / ffn 12288) | Rate | Packed-weight bytes/s | Log |
|---|---|---|---|
| `M=12288 K=4096 splits=4` (gate+up projection set) | **173.6–177.0 Gop/s** (12 consecutive build self-tests, `runs/build_v31.log` … `build_v42.log`; the newest three are 177.0 / 176.9 / 176.9 ⇒ **176.9 Gop/s** is today's number, 0.285 ms/token-op) | 43.4–44.2 GB/s of packed weights | each build log's own self-test line, so the spread is run-to-run thermal, not code drift |
| Weighted kernel mix over the whole model | **147.3 Gop/s** | — | STAGE3 report |
| WGSL reference kernels for the same shapes, same card | 143–155 Gop/s | — | `probe 13`, STAGE1/STAGE3 |

No DP4A and no matrix cores exist on gfx803, so every one of those operations is a scalar fp32
multiply-accumulate over an inline dequantised ternary operand. `shaderFloat16 = 0` on this driver,
which is why `f16_rne.glslh` exists instead of `float16_t`.

## 3. End-to-end generation, 9B on 4 GB (MEASURED)

| Item | Value | Note / log |
|---|---|---|
| Steady-state decode, today's default build | **70.6 ms/token = 14.16 tok/s** (48 tokens / 3389.5 ms, first token 70.2 ms, prefill 1154.5 ms for 16 prompt tokens) | `runs/regression_v14.log`, run with `BONSAI_QCHECK=1` so the same log carries the bitwise gate: `gate: PASS (bitmask 0 = A ok A2 ok A3 ok A4 ok B ok C ok D ok E ok)` |
| Short-context band across all releases | **14.05–14.27 tok/s** (70.2–70.7 ms/token) | `runs/regr_short_default_v1.log`, §8 of `KV_P0P1P2_MEASURED.md`; V=fp16 14.27 vs V=q8_0 14.26 = 0.01 tok/s = noise; the default GQA config sits inside this band ⇒ **0 % regression** |
| Prefill, real prompt, current binary | **1310 prompt tokens in 103 766.4 ms = 12.62 tok/s = 79.2 ms/position** (that figure is the mean over the ramp — position *t* scans *t* keys, so it is not comparable to the cap-256 decode number) | `runs/gqa_real_c2048_merged_v1.log:113`; integrated forward with the §6 fit it gives **≈ 5.9 min at 4 K** and **≈ 120 min at 32 K**, both DERIVED (see §10 item 4, which also records the ≈ 55 min figure this replaced) |
| First measured pure-blood run | **13.99 tok/s** (48 tokens / 3143.1... 3431.1 ms, 71.5 ms/token) | STAGE8, before the KV work |
| Shipped package, run from the release directory | **14.31 tok/s** (8 tokens / 559.1 ms, steady 69.9 ms/token, first token 69.9 ms) with `gate: PASS` | `runs/package_smoke_v2.log` - the run against the **as-shipped 90-file tree**, from `dist/bonsai-vk-0.1.0-win64/`, not `runtime_vk/`, so these are the bytes a user actually gets (`package_smoke_v1.log` is the same exe against the earlier 86-file tree: identical verdicts, identical 559.1 ms, only the wall-clock load line differs). n = 8 tokens, which is why it is NOT folded into the 14.05-14.27 band above (that band is n = 48 runs); a 2 % spread on an 8-token sample is arithmetic, not signal |
| Weight load, *this engine* | **14.05–14.44 s cold**, **1051.9–1201.0 ms warm** (in-process timer; forward bring-up 44.9–716.7 ms) | cold: `runs/kv_smoke_v1.log:104`, `fwd_stop0.log:39`, `ladder_v6_rho.log:86`, `limits_probe.log:87` — four independent first-reads of the checkpoint. warm: the `kvfx_*` / `c*_*` ladder logs, e.g. `runs/kvfx_c32768_q2048_v1.log:112` (1076.8 ms), plus the two package runs, both taken immediately after `clear_field.ps1`: `runs/package_smoke_v1.log:147` (1201.0 ms load, 716.7 ms bring-up) and `runs/package_smoke_v2.log:147` (1099.9 ms, 421.2 ms) — same exe, same cap, same prompt, **17 % apart**, so the earlier note that "clear_field had just killed 7 processes, so the cache was only partially warm" cannot be a cause: the second run was cleared the same way and came in lower. The page-cache state is *not* controlled or measured by the harness, which is why both regimes are listed as ranges instead of a single number, and why the warm figure carries no causal story |
| Device-local footprint, *this engine* | weights committed **2467.7 MiB**; activations **58.2 MiB** at cap 256 (now MEASURED - the engine prints the decomposition itself: 52.0 cap-independent + 256 slots x 24.66 KiB), **101.339 MiB** at cap 2048 (measured), **841.261 MiB** at cap 32768 (measured, = state 48.0 + KV 784.0 + logits 0.95). PDH dedicated peak **2539.23 MiB** at cap 256 (measured, = 2467.7 + 58.2 + **13.3 MiB** the driver charges that our budget table does not model) and **3323.84 MiB** at 32 K; `Shared Usage` growth **+0.00** at both | `runs/package_smoke_v2.log:86,87` (weights, activations) and `:176,196` (the `[spill] tok 16/24` dedicated readings), `runs/gqa_real_c2048_merged_v1.log:45,48`, `runs/kvfx_c32768_q2048_v1.log:46-48,108,124` |
| Peak device-local, *reference engine* | **1985.3 MiB** WDDM per-process (engine's own account 2024 MiB = 1746 model + 178 ctx + 100 compute) | `END_TO_END_MEASURED.md:20,147`. **It is smaller than our footprint because that run keeps only 1746 MiB of model on the card** (§ row "GPU offload +9.2 %"), so it is not comparable to ours and must never be quoted as ours |
| Reference engine, same checkpoint, same card | **4.417 t/s** (PrismML build 10777) | ⇒ Bonsai is **3.16×** at the first ignition (13.99 / 4.417) and **3.21×** today (14.16 / 4.417) |
| GPU offload vs CPU-only on the reference engine | +9.2 % | `END_TO_END_MEASURED.md` — the reason a hand-written kernel path was necessary |
| 4 GB wall, *reference engine* | passes at `-c 49152`, fails at `53248` (compute buffer first) | measured, not inferred. Not the same measurement as our arena ceiling in §5 — see the note there |
| Per-token host bubble | **16.8 ms** = fence wait 11.8 + submit 5.0 | STAGE5; the "37 % submission loss" claim, measured |
| Whole-tree build warnings | **0** | `runs/build_v40.log` / `v41` / `v42`, each `rc=0`, 52 shaders + engine + every tool |

## 4. Numerical fidelity (MEASURED, bitwise wherever the contract says bitwise)

| Layer | Reference | Result |
|---|---|---|
| Tokenizer | official `llama-tokenize.exe` | **39/39 probes identical ID-by-ID** (CJK, Latin, Korean, digits, whitespace, newline) |
| PTQ1_0 dequant | official `dequant_ptq1_0.comp` running on this card | real 1000 blocks / 128 000 weights **max\|err\| = 0.00000**; synthetic 3840 blocks / 491 520 weights **bitwise identical**, including d=0, subnormal, negative scale, ±Inf, ±NaN |
| Attention scores | fp32 CPU mirror + official `kq` dumps | layer 3 full chain max\|err\| 6.7e-04 / 2.9e-03 / 3.6e-03 at seq 8/32/64 = **0.26× the rounding cost of the official fp16 KV cache itself**; residual fully attributed to ±1 fp32 ulp in the per-row norm constant |
| KV layout | own position-major fp16 arena | one-shot prefill vs token-by-token, 8/32/64 steps: **0 bit differences**; dead slots are exactly +0.0 |
| Row kernels (4 lineages) | `BONSAI_QCHECK=1` gate, rc 6 on a single differing bit | narrow ≡ wide ≡ tiled ≡ **GQA-merged**; 0 bit-diff over all 16 heads × cap slots and all packed pairs (A/A2/A3/A4 + B/C/D/E), `runs/gqa_gate_c256_v1.log`, `runs/gqa_real_c2048_merged_v1.log` |
| Independent oracle | double-precision recompute of 512 dots from the read-back fp16 arena | **worst \|dot − SCORE\| = 0.000e+00** |

Sampling (the official sampler) is **not** aligned yet — see §10, open items.

## 5. Long-context ladder (MEASURED with the synthetic KV fixture, cross-validated against real prefill)

Fixture: `BONSAI_KVFILL=N` fills the arena and pins every token at position `N-1`. Valid for ms/token,
tok/s, kernel mix and traffic; **invalid** for token identities, text and any fidelity statement.

| Positions | source of the merge-off row | ms/token | tok/s |
|---|---|---|---|
| 4 096 | `runs/kvfx_c4096_q2048_v1.log` | 127.4 | **7.85** |
| 8 192 | `runs/kvfx_c8192_q2048_v1.log` | 185.8 | **5.39** |
| 16 384 | `runs/kvfx_c16384_q2048_v1.log` | 301.4 | **3.32** |
| 32 768 | `runs/kvfx_c32768_q2048_v1.log` | 531.1 | **1.88** |
| the same four points, **default config** (merged QK on) | batch H paired, `runs/gqa_pair_v1_summary.txt` | 106.3 / 144.3 / 218.6 / 374.8 | **9.41 / 6.93 / 4.57 / 2.67** |

Cross-validation of the fixture against real prefill, same binary: 1 335 positions · wide 90.4 real vs
90.4 fixture (**0.0 %**); 1 335 · tile q2048 88.8 vs 88.8 (**0.0 %**); 1 816 · wide across binaries
96.9 vs 97.3 (**+0.4 %**).

Least-squares fits over the measured points (positions = live keys per head per layer):

```
wide   : ms/token = 71.12 + 0.01443 x positions      max residual 0.03 ms
tiled  : ms/token = 69.97 + 0.01408 x positions      max residual 0.67 ms (0.5 %)
tiled + merged QK (default, batch H, DERIVED from the four measured on1 points):
       : ms/token = 67.17 + 0.00936 x positions      max residual 1.99 ms (0.9 %)
       slope 0.01408 -> 0.00936 = -33.5 %; the constant is unchanged within the drift band
```

**The constant is the floor.** 69.97 ms of a token is the ternary GEMV path plus the 24 recurrent-state
layers; the attention term at 32 K is 461 ms of the same token. Any lever that only touches attention
cannot move a short context, and cannot double a long one — it moves the slope. That sentence is the
reason §6's result is what it is, and the third fit row is that prediction being checked: the merge
moves the slope by −33.5 % and leaves the floor alone.

Memory at the widest point (DERIVED from the arena geometry, MEASURED as device state): q8_0-V + fp16-K
arena = **24.50 KiB/token** over 8 attention layers (24.66 KiB/slot once the fp32 side is added), safe
ceiling ≈ **39.2 K** tokens (39 237 slots; see the correction note in
`KV_P0P1P2_MEASURED.md` §「重算的安全 / 极限上下文」); 40 K is already
out of band and 48 K exceeds the budget by 239 MiB. At 32 K: dedicated 3323.84 MiB, PDH `Shared Usage`
growth **+0.00** over the whole run, i.e. no WDDM spill.

**Do not confuse the two "4 GB walls" in this file.** The ≈ 39.2 K-token figure above is *our* arena
arithmetic, and every constant in it is a line the engine prints itself: weights committed 2467.7 MiB,
cap-independent activations 52.0 MiB, 24.66 KiB/slot, DXGI LOCAL budget 3464.6 MiB ⇒
`(3464.6 − 2467.7 − 52.0) × 1024 ÷ 24.66 = 39 237`. Against the raw 3840 MiB device-local heap the same
formula gives 54 825 slots, and the engine's own cap-256 line agrees: `headroom 1314.1 MiB (= 54558 more
slots)` ⇒ 32 768 + 22 046 = 54 814 at cap 32 K (`runs/gqa_gate_c256_v1.log:49`,
`runs/kvfx_c32768_q2048_v1.log:48-49`). The `-c 49152` pass / `53248` fail in §3 is the *reference
llama.cpp fork's* allocator budget on the same checkpoint (`END_TO_END_MEASURED.md`). Two different
engines, two different numbers, and neither has been driven past 32 K in a live run here.

## 6. GQA 4× redundancy removal (this release)

Shape: one workgroup per (layer, kv-head) serving `HPK = n_head / n_head_kv = 4` query heads, so the K
gather instruction count and the K half-unpack count drop 4× and the grid drops 16 → 4 workgroups.
Operand per position per layer, from the same geometry table: **98.0 KiB → 24.5 KiB** for a full merge
(K 64.0 → 16.0, q8_0 V 34.0 → 8.5); **→ 64.0 KiB** for the QK half alone.

Paired measurement (batch H, `runtime_vk/batch_h.sh v1`; each configuration measured next to its own
control inside one thermal window, the short-context pair taken first on a cold card, the control
repeated last as the drift estimate). All rows are ms/token from `runs/gqa_pair_v1_summary.txt`:

| Positions | control = tiled q2048 + frozen PV (`GQA_OPT=0`) | merged QK only (`on1`, the default) | merged QK+PV (`on3`) | on1 latency Δ | on1 tok/s gain |
|---|---|---|---|---|---|
| cap 256 (short, real prefill) | 73.4 ⇒ 13.63 | 72.7 ⇒ 13.75 | **70.9 ⇒ 14.10** | −1.0 % | +0.9 % |
| 4 096 | 126.9 ⇒ 7.88 | **106.3 ⇒ 9.41** | 118.1 ⇒ 8.47 | −16.2 % | **+19.4 %** |
| 8 192 | 187.4 ⇒ 5.34 | **144.3 ⇒ 6.93** | 169.5 ⇒ 5.90 | −23.0 % | +29.8 % |
| 16 384 | 303.6 ⇒ 3.29 | **218.6 ⇒ 4.57** | 271.5 ⇒ 3.68 | −28.0 % | +38.9 % |
| 32 768 | 542.5 ⇒ 1.84 | **374.8 ⇒ 2.67** | 470.4 ⇒ 2.13 | **−31.0 %** | **+45.1 %** |
| control repeated last | 72.3 ⇒ 13.83 | — | — | — | — |

The last row is the drift estimate for the whole table: the *identical* control configuration measured
again at the end of the session read 73.4 → 72.3 ms/token, i.e. a **1.5 % band**. Every long-context gain
above (16–31 %) clears that band by more than 10×; the short-context row (0.9–1.5 %) does not clear it at
all, which is exactly why the honest statement about cap 256 is "free, not faster". Today's default build
measures **70.6 ms/token = 14.16 tok/s** end to end with the gate on (`runs/regression_v14.log`), inside
the 14.05–14.20 frozen baseline ⇒ 0 % regression.

First-pass (batch G, unpaired — kept because it is the attribution instrument, not the headline; the
absolute values run ~1 % slower than batch H because the card was not paired):

| Positions | off | merged both halves | delta |
|---|---|---|---|
| 4 096 | 128.7 | 118.8 | **+7.7 %** |
| 8 192 | 186.7 | 171.4 | **+8.2 %** |
| 16 384 | 315.0 | 264.9 | **+15.9 %** |
| 32 768 | 545.4 | 461.9 | **+15.3 %** |

attribution at 4 096 (batch G): QK-half-only **109.8 ms/token**, PV-half-only **184.2 ms/token**, control
128.7. That pass produced the sentence "the PV half is a measured regression", and the paired pass
corrected it: **the PV merge is regime-dependent, not uniformly negative** — measured in `on3` against
`on1` it *costs* 11.1 % at 4 K (106.3 → 118.1) and 25.5 % at 32 K (374.8 → 470.4), but *saves* 2.5 % at
cap 256 (72.7 → 70.9). Bit 1 of `BONSAI_GQA_PART` is off by default because the switch exists for long
context. The explanation that fits both regimes: the QK operand is a *gather* (lane-to-lane stride 2 KiB,
so a 64-lane instruction touches 4 of 32 bytes per sector, ~8× waste) and merging removes transactions;
the PV operand is read contiguously and already shared within a quad, so merging removes only ALU work
while cutting resident workgroups 16 → 4 — at cap 256 the grid is already small, so the ALU saving is all
that is left and it wins, while at 32 K the lost occupancy dominates. Bytes were never the cost; the
gather shape and the resident-workgroup count were.

Prediction check that ties the two regimes together (DERIVED): a 15 % cut of the 0.01408 ms/key slope
predicts −2.8 ms at 1 310 real positions, i.e. 88.8 → ~86.0 ms/token; the measured real 1 310-token run
with the merge enabled was **87.4 ms/token** (`runs/gqa_real_c2048_merged_v1.log`) — the same-binary
control for that point is the tiled 88.8, so the real-context gain is ~1.6 % at 1.3 K, ~8 % at 4 K and
~15 % at 32 K, exactly the slope-only signature the model predicts.

## 7. The 27B architecture gates (MEASURED on this host, `runs/regression_v14.log`)

The real Ternary-Bonsai-2-27B file (≈ 5.9 GB) does not fit in 4 GB and is not on this machine, so what is
measured here is the **architecture contract**, exercised on a synthetic 64-block plan built from the 9B's
real GGUF header. `runtime_vk/align_27b.cpp` is the gate; `runtime_vk/bonsai_model.h` is the plan.

| Gate | Measured result |
|---|---|
| PTQ1_0 decode on the 64-block plan | real 1000 blocks / 128 000 weights: `max\|GPU-REF − CPU-REF\| = 0.00000`, `max\|OURS − GPU-REF\| = 0.00000`, **bit-exact 128 000/128 000**, non-finite mismatches 0; the runtime dot form `d*(cs−xs)` vs `Σ w_e·x_e` reports `max abs 0.000e+00` |
| Hybrid routing | 27B plan = 64 blocks = **16 attn + 48 ssm**; charged KV **49.000 KiB/token**, arena at cap 32 768 = **1568.00 MiB**, recurrent state **100.50 MiB**, i.e. the SSM layers' KV charge is exactly 0 |
| The forbidden allocator | "KV on all 64 blocks" priced at **196.000 KiB/token = 4.0×** the charged one, 4704.00 MiB at cap 32 768 — the invariant is printed as a number, not asserted in prose |
| Rotation route is metadata | `version 1, block 1024 → hadamard_fwht.comp`; `block_size 4096` on the same 64-block plan keeps `hat4096.comp`. No model-name lookup anywhere in the route |
| Negative controls | four of them, each **refused as required**: `hadamard.version = 2`, rotation active with `block_size` absent (message: "the transform width cannot be defaulted to 4096, that is the 9B's answer not this file's"), and the two format refusals |
| FWHT kernel vs host Sylvester FWHT | orthogonal at N = 128 with **integer-exact** evidence (128 diagonal dots exactly 128, 16 256 off-diagonal dots exactly 0); involution `H(H x) − N x` = 0.000e+00 at N ≤ 256 and 9.766e-04 / 9.766e-03 at N = 1024 / 4096; **BITWISE identical to the host FWHT** at N = 16/64/256/1024/4096 over 4096 lanes; `1/sqrt(4096) == 1/64` bit-for-bit, which is why the 9B keeps its frozen kernel |
| Verdict | `ALL CHECKS PASSED (0 failure(s))` |

Scope statement, so this section is not over-read: the resolver, the charge invariant, the rotation route
and the kernel equivalence **are** proven on hardware. Running the actual 27B checkpoint on this card is
**not** proven and cannot be — it needs 5.9 GB of weight storage, which is what §8's fleet machinery and
the iGPU zero-copy branch exist to make unnecessary rather than to make fit.

## 8. Multi-card pinned ring and iGPU zero-copy (MEASURED, `runs/fleet_ring_v4.log`)

`bonsai_fleet.exe ring 4 8 6` = depth 4, 8 MiB slots, 6 hops, on the transfer-only queue family (family 2
on this driver). Code: `runtime_vk/gpu_fleet.h`; rig: `runtime_vk/bonsai_fleet.cpp`.

| Quantity | Value | Note |
|---|---|---|
| Host-side self-test | **55 checks, 0 failed → PASS** | `bonsai_fleet.exe selftest`, creates no Vulkan object |
| Ring integrity | D2H arrival **0 mismatch(es)** over 12 words read through the ring mapping / 4 written slots (0 untouched) → **PASS** | cursor cross-check `produced 6 vs hops 6 → consistent` |
| One fenced 8 MiB D2H leg | **1204.6 µs → 6.96 GB/s = 97 % of the MEASURED 7.15 GB/s ceiling** | submit, fence **and** the queue-drain tail included |
| Fence alone vs tail | fence 1204.1 µs, queue-idle tail after it **0.4 µs** | a tail near zero is what a correct fence looks like |
| Eight 8 MiB legs in **one** submit | fence 9410.9 µs + tail 0.6 µs → **1176.4 µs per copy = 7.13 GB/s** | ≤ the single-copy ceiling, which is the check that the batching is real |
| Round trip slot1 → consumer → slot2 | bytes intact, fences 1207.5 / 1349.6 µs + 0.3 / 0.4 µs tails | each leg = copy + one submit+fence round trip (MEASURED 99.4 µs) |
| Per hop on this host | wall **3171.3 µs** = D2H fence 2706.7 + doorbell 0.3 + back-pressure 3.2 over 2 waits + CB-recycle 17.8 over 8 waits | the host-visible waits are counted, not hidden |
| Back-pressure | **2 events at depth 4** | a depth-1 ring takes one wait per hop ⇒ depth is the mechanism |
| Topology verdict | `probe`: **1 COMPUTE-capable device of 1 physical device visible to the loader** | (A) layer-pipelined multi-GPU is **NOT EXERCISABLE HERE**: schedule, ring and fences are compiled and driven on the paths this host can reach; **no PCIe-to-PCIe hop has been taken**, the tested leg is D2H → host → H2D |
| iGPU zero-copy verdict | `probe`: memory types **2 and 6** are `DEVICE_LOCAL \| HOST_VISIBLE \| HOST_COHERENT` simultaneously → capability **AVAILABLE** | on this *discrete* card that heap is a **256 MiB BAR window**, not a 2.34 GiB weight arena — which is precisely why the branch is gated on `deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` and is never taken here. The discrete path is unchanged when the branch is not taken. |

Two bug classes were found while producing that table, and both are now guarded in code:

**`VkBufferCreateInfo::pQueueFamilyIndices` is read during `vkCreateBuffer` and copied nowhere.** Four
call sites declared the family array inside the `if` that set the sharing mode, so the driver pointed at a
dead stack frame: survived `-O0`, SIGSEGV inside `amdvlk64.dll` at `-O2`, backtrace indistinguishable from
a driver bug. All four are hoisted, one of them in the engine's own staging path
(`bonsai_weights.h::stage_alloc`) — which is why §3's fidelity re-run exists: after that edit the bitwise
gates still pass (bitmask 0, A/A2/A3/A4/B/C/D/E) and the rate is 14.16 tok/s, so the fix costs nothing.

**A fence created with `VK_FENCE_CREATE_SIGNALED_BIT` is already signalled when the wait runs.** The rig
measures the effect and the correction in the same log line pair: created-SIGNALED control = fence **0.8 µs**
with a **1195.2 µs** queue drain behind it (REPRODUCED — the wait returned while the queue still held the
copy); `flags = 0` = fence **1204.1 µs** with a **0.4 µs** tail. That is spec behaviour, not a driver fault,
and before the control existed it produced two wrong conclusions: an impossible **56.67 GB/s** "batched"
rate and a hop that looked sync-dominated when it is DMA-bound. Both claims are withdrawn here. Every fence
in the repo is now created with `flags = 0` and armed only by its own submit; the ring additionally
publishes on a doorbell word written in the **same** command buffer as the payload (queue order, no fence
assumptions at all), and the log's completion-signal audit prints `fence and stamp agree (within 4x)`.

## 9. Reproducing any row above

```
runtime_vk\build_mingw.bat                      :: 52 shaders + engine + all align_*/fleet tools -> 0 warnings
runtime_vk\align_tokenizer.exe <gguf> runs/tok_probes.txt runs/tok_ref_32.jsonl
runtime_vk\align_f16.exe
runtime_vk\align_dequant.exe <gguf> 1000        :: and: synth
runtime_vk\align_attn.exe 64 256
runtime_vk\align_27b.exe <gguf>                 :: the 27B gates of section 7 (--no-gpu drops the on-card FWHT leg)
runtime_vk\bonsai_vk.exe run <gguf> 32 48       :: end to end; BONSAI_QCHECK=1 to add the kernel gate
runtime_vk\bonsai_fleet.exe selftest            :: 55 host-side checks
runtime_vk\bonsai_fleet.exe probe               :: topology verdicts, no logical device created
runtime_vk\bonsai_fleet.exe ring 4 8 6          :: the pinned-ring measurement of section 8
runtime_vk\bonsai_fleet.exe split 32 2          :: layer-range split over 2 cards (refuses, honestly, on 1)
runtime_vk\batch_b.sh v1                        :: tiled-vs-wide fixture ladder -> runs/kvfx_*_v1.log
runtime_vk\batch_g.sh v1                        :: GQA first pass         -> runs/gqa_ladder_v1_summary.txt
runtime_vk\batch_h.sh v1                        :: GQA paired A/B         -> runs/gqa_pair_v1_summary.txt
python tools\source_size.py                     :: the only producer of every byte count in section 1
```

## 10. Open items, stated plainly

1. Sampling is not aligned with the official sampler; text quality is validated by the layer-1..4
   fidelity ladder and by eye, not by a sampler diff.
2. The chat template + layer-5 sampling hypothesis from STAGE8 is still untested end to end.
3. 40 K/48 K contexts remain **DERIVED-out-of-band** (the budget formula exceeds the card), not measured.
   Neither our arena ceiling (≈ 39.2 K) nor the reference fork's `-c 53248` failure has been driven live.
4. Prefill wall time is still quadratic and un-optimized, and the honest extrapolation is **worse than
   the one this file used to print**. DERIVED from the §6 fit `ms/token = 67.17 + 0.00936 × positions`,
   integrating the key-count term over the ramp (position *t* scans *t* keys, so the average live-key
   count over an *N*-token prompt is *N*/2): a 32 K prompt costs **≈ 120 min**, not the ≈ 55 min an
   earlier draft of this list claimed by multiplying *N* by a flat 100 ms — that flat constant was the
   1 310-position cost, which is the *cheapest* part of the ramp. The method is checked against the one
   measured prefill we have: at 1 310 prompt tokens it predicts 73.3 ms/position and 1.6 min while the
   real run measured **79.2 ms/position and 103.8 s** (`runs/gqa_real_c2048_merged_v1.log:113`,
   12.62 tok/s) — **+8 %**, so the integration is conservative in the right direction. By the same
   arithmetic a 4 K prompt costs **≈ 5.9 min**. The fixture measures decode only, so every prefill number
   above 1 310 positions remains DERIVED, not measured.
5. The PV half of the GQA merge is off by measurement, not by design — and §6 now says where it pays
   (cap 256, +2.5 %) and where it costs (32 K, −25.5 %). A half-merge (2 heads per workgroup, 8
   workgroups) is the obvious next experiment and is **not** implemented.
6. The long-context ladder in §5 was paired for the merge A/B, but the *default* configuration has not
   been re-measured across the whole ladder in one fresh session since; the on1 column of §6 is the
   current number for it.
7. Multi-card layer pipelining is compiled, self-tested and driven over a real pinned ring, but never
   over two compute-capable devices — this laptop has one. The claim boundary is in §8, row "topology
   verdict".
8. The 27B numbers in §7 are a synthetic 64-block plan built from the 9B's real header, because the real
   5.9 GB checkpoint cannot be loaded here. Proven: routing, charge invariant, rotation contract. Not
   proven: 27B text quality.

