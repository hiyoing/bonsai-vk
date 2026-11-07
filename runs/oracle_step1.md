# Oracle: 1-token decode step (`runs/dbg_full_real.log`, second graph)

Extracted by `tools/oracle_step1.py`. Full node table: `runs/oracle_step1.tsv` (2424 rows, one per graph node, file order).

## Scope / provenance

- Log holds **2** graph evaluations: prefill (2 tokens, **2424** nodes, lines 2-32036) and the 1-token decode step (**2424** nodes, lines 32046-57247).

- The decode graph *would* be the recurrence oracle: `cache_s_l*` (GDN state), `cache_r_l*` (conv history) and `cache_k_l*`/`cache_v_l*` (KV) are read as graph **inputs**. In this particular dump every one of those reads sums to exactly zero - the prefill's history is NOT visible in the second graph. Read the `Cold-cache evidence` section before using these numbers to decide recurrence ordering.
- `sum` is the full-tensor f32 sum printed by ggml (6 decimals); `head3`/`tail3` are the printed element values, first/last 3 of the printed stream, kept verbatim (4 decimals, sign preserved, comma-joined). The prints are truncated, so only the `sum` column is a hard invariant; head3/tail3 are structural spot checks.
- `layer` is inferred from operand names (`blk.N.*` -> N, `cache_*_lN` -> N, restricted to tensors actually named `cache_*`), else an `-N` name suffix, else position: `embed` before the first block node and `out` after the last one; unlabelled `node_N` / `(reshaped)` helpers inherit the preceding block.

- parse completeness: 4848 `common_debug_cb_eval` lines in the log, 4848 nodes parsed, 0 unparsed, 72 nodes with more than one printed `sum` (the first is used), 0 without any.

## Parser validation against known-good anchors (prefill graph)

| anchor node | expected sum | parsed sum | verdict |
| --- | --- | --- | --- |
| node_0 GET_ROWS(token_embd) | -4.543495 | -4.543495 | OK |
| node_2 MUL_MAT(prism.hadamard.4096) | -0.996582 | -0.996582 | OK |
| model.input_embed MUL | -0.601764 | -0.601764 | OK |

## Token identity of the 1-token step

- The batch tensor `inp_tokens` is a graph input and ggml does not print input leaves, so **no node line carries the token id**. The only ids in the file are the run's own trailer lines (verbatim, with log line numbers):

```
 57254: Model add_bos: false
 57255: Input prompt: "ab"
 57256: Token ids (1):
 57257: ab(365)
```

  `Token ids (1):` therefore describes the **1-token batch** fed to the second graph: one token, id **365**, detokenized piece `ab`.
- Corroboration from the graph itself: the prefill's embedding lookup printed prefill row0 head [0.0000,0.0166,-0.0166] tail [0.0000,0.0168,-0.0168] and prefill row1 head [-0.0156,0.0156,-0.0156] tail [0.0139,0.0000,-0.0139], while the 1-token step's lookup prints head `-0.0153,0.0000,-0.0153` / tail `0.0000,0.0138,0.0138`, which matches none of the prefill rows. Its embedding row is therefore a third, distinct one - consistent with 365 being the token sampled after the "ab" prefill, not a re-feed of the prompt.
- No `generated text` line exists in the log; the only text lines are the four quoted above. Absolute position of this step is therefore `pos = 2` (0-based, third token), which is inferred, not printed - the ROPE operands (`leaf_58`) are input leaves and dump nothing, so the cos/sin pair must be reproduced from the position, not read from this log.

## Consequential GDN nodes (block 0 = first Gated Delta Net layer, 1-token step)

| role | node name | op | shape | sum | head3 \| tail3 |
| --- | --- | --- | --- | --- | --- |
| input rmsnorm | norm-0 | RMS_NORM | 4096, 1, 1, 1 | -40.661980 | 0.0143,1.0184,1.1478 | 0.5768,-0.2608,-0.2352 |
| attn_norm (blk.0) | attn_norm-0 | MUL | 4096, 1, 1, 1 | -45.631546 | 0.0134,1.1238,1.1596 | 0.6142,-0.2627,-0.2431 |
| conv state read (3x8192) | conv_states-0 | GET_ROWS | 24576, 1, 1, 1 | 0.000000 | 0.0000,0.0000,0.0000 | 0.0000,0.0000,0.0000 |
| prior GDN state read (cache_s_l0 GET_ROWS) | node_31 | GET_ROWS | 524288, 1, 1, 1 | 0.000000 | 0.0000,0.0000,0.0000 | 0.0000,0.0000,0.0000 |
| state_predelta-0 (reshape of that read) | state_predelta-0 | RESHAPE | 128, 128, 32, 1 | 0.000000 | 0.0000,0.0000,0.0000 | 0.0000,0.0000,0.0000 |
| qkv MUL_MAT | node_21 | MUL_MAT | 8192, 1, 1, 1 | -324.560303 | 0.5366,-0.8338,0.3179 | 1.3872,1.1099,-4.4314 |
| conv_input CONCAT | conv_input-0 | CONCAT | 4, 8192, 1, 1 | -324.560303 | 0.0000,0.0000,0.0000 | 0.0000,0.0000,-4.4314 |
| SSM_CONV | conv_output_raw-0 | SSM_CONV | 8192, 1, 1, 1 | 67.495415 | -0.0590,-0.0892,0.0300 | 0.0010,0.0016,-0.0181 |
| SILU (conv output) | conv_output_silu-0 | SILU | 8192, 1, 1, 1 | 168.968338 | -0.0286,-0.0426,0.0152 | 0.0005,0.0008,-0.0090 |
| L2_NORM (q,k) | qk_conv_l2-0 | L2_NORM | 128, 32, 1, 1 | 21.658962 | -0.0117,-0.0174,0.0062 | -0.0197,0.0400,-0.0001 |
| q VIEW | q_conv_predelta-0 | VIEW | 128, 16, 1, 1 | 10.530460 | -0.0117,-0.0174,0.0062 | -0.0004,0.0035,-0.0060 |
| k VIEW | k_conv_predelta-0 | VIEW | 128, 16, 1, 1 | 11.128533 | 0.0024,-0.0059,0.0040 | -0.0197,0.0400,-0.0001 |
| v VIEW | v_conv_predelta-0 | VIEW | 128, 32, 1, 1 | 124.935616 | -0.0023,0.0291,0.0161 | 0.0005,0.0008,-0.0090 |
| alpha MUL_MAT | node_42 | MUL_MAT | 32, 1, 1, 1 | -37.683086 | -0.0428,0.0203,3.2403 | -2.0902,-2.0179,2.7920 |
| alpha+dt bias ADD | node_44 | ADD | 32, 1, 1, 1 | 38.082535 | -3.5740,-1.4250,21.7403 | -4.6058,-5.9866,14.9170 |
| SOFTPLUS | a_softplus-0 | SOFTPLUS | 32, 1, 1, 1 | 146.662094 | 0.0277,0.2155,21.7403 | 0.0099,0.0025,14.9170 |
| gate = softmul*ssm_a MUL | gate-0 | MUL | 32, 1, 1, 1 | -14.298706 | -0.0019,-0.0019,-1.2265 | -0.0002,-0.0001,-1.7000 |
| beta MUL_MAT | node_48 | MUL_MAT | 32, 1, 1, 1 | 34.892235 | -3.0942,1.4797,3.6127 | -0.7704,-0.7278,3.2281 |
| beta SIGMOID | beta_sigmoid-0 | SIGMOID | 1, 32, 1, 1 | 19.822166 | 0.0433,0.8145,0.9737 | 0.3164,0.3257,0.9619 |
| GATED_DELTA_NET (fused) | node_52 | GATED_DELTA_NET | 4096, 129, 1, 1 | 65.066391 | -0.0000,0.0000,0.0000 | 0.0002,-0.0003,0.0000 |
| new_state VIEW | new_state-0 | VIEW | 128, 128, 32, 1 | 63.934986 | -0.0000,0.0000,-0.0000 | 0.0002,-0.0003,0.0000 |
| state CPY -> cache_s_l0 | cache_s_l0 (view) (copy of new_state-0) | CPY | 524288, 1, 1, 1 | 63.934986 | -0.0000,0.0000,-0.0000 | 0.0002,-0.0003,0.0000 |
| attn_output VIEW (tail of GDN out) | attn_output-0 | VIEW | 128, 32, 1, 1 | 1.131346 | -0.0000,0.0000,0.0000 | 0.0000,0.0000,-0.0000 |
| z (attn_gate) MUL_MAT | z-0 | MUL_MAT | 4096, 1, 1, 1 | -2659.278564 | 1.9385,-2.0294,-1.2083 | -0.6352,-0.4104,1.2977 |
| per-head RMS_NORM (128,32) | norm-0 | RMS_NORM | 128, 32, 1, 1 | 267.544434 | -0.0006,0.0070,0.0039 | 0.0007,0.0011,-0.0121 |
| ssm_norm MUL | node_60 | MUL | 128, 32, 1, 1 | 203.370972 | -0.0005,0.0066,0.0035 | 0.0006,0.0010,-0.0108 |
| SWIGLU (gdn gated) | node_61 | SWIGLU | 128, 32, 1, 1 | -9.603460 | -0.0008,-0.0016,-0.0010 | -0.0001,-0.0002,-0.0110 |
| final_output reshape/permute chain head | final_output-0 | RESHAPE | 4096, 1, 1, 1 | -9.603460 | -0.0008,-0.0016,-0.0010 | -0.0001,-0.0002,-0.0110 |
| ssm_out MUL_MAT | linear_attn_out-0 | MUL_MAT | 4096, 1, 1, 1 | 0.883923 | -0.0183,0.0056,-0.0036 | 0.0076,-0.0033,0.0109 |
| attn residual ADD | attn_residual-0 | ADD | 4096, 1, 1, 1 | 0.419769 | -0.0181,0.0172,0.0095 | 0.0141,-0.0063,0.0082 |
| ffn gate MUL_MAT | ffn_gate-0 | MUL_MAT | 12288, 1, 1, 1 | 281.086334 | 0.0743,-0.1599,0.0074 | 0.1678,0.1662,0.1384 |
| ffn up MUL_MAT | ffn_up-0 | MUL_MAT | 12288, 1, 1, 1 | 11.754902 | 0.2557,-0.4531,-0.0126 | -0.0533,0.1394,-0.0008 |
| ffn SWIGLU | ffn_swiglu-0 | SWIGLU | 12288, 1, 1, 1 | 6.937200 | 0.0099,0.0333,-0.0000 | -0.0048,0.0125,-0.0001 |
| ffn down MUL_MAT | ffn_out-0 | MUL_MAT | 4096, 1, 1, 1 | 3.236981 | -0.0664,0.1302,-0.0362 | 0.0488,0.0301,0.0270 |
| block output ADD (l_out-0) | l_out-0 | ADD | 4096, 1, 1, 1 | 3.656747 | -0.0845,0.1474,-0.0267 | 0.0630,0.0238,0.0353 |

Structure worth pinning in the engine: `GATED_DELTA_NET` is fused over the whole layer and emits `{4096, 129, 1, 1}` = the updated state (`128 x 128 x 32 = 524288 = 4096*128` elems) followed by **one** 4096-wide output row for this token (the prefill printed `{4096, 130, 1, 1}` for 2 tokens). `new_state-N = VIEW(node_52)` takes the first 128 columns, `attn_output-N = VIEW(node_52)` takes the last 4096 elements; `conv_input-N = CONCAT(conv_states_reshaped-N{3,8192}, qkv_mixed_transposed-N{1,8192})` = `{4,8192}` and the conv kernel is `blk.N.ssm_conv1d.weight{4,8192}` (kernel width 4, history 3 + this token). Checked on all 24 GDN layers of this graph: `|sum(GDN out) - sum(new_state) - sum(attn_output)| <= 0.000065` (print rounding) and `ne1 == 128 + n_tokens` always; 0 layer(s) failed the check.

## Consequential softmax-attention nodes (block 3 = first attention layer, 1-token step)

| role | node name | op | shape | sum | head3 \| tail3 |
| --- | --- | --- | --- | --- | --- |
| attn_norm (blk.3) | attn_norm-3 | MUL | 4096, 1, 1, 1 | -24.155247 | -1.4550,2.2676,0.2996 | 0.4211,-0.3421,0.1607 |
| attn_q MUL_MAT | Qcur_full-3 | MUL_MAT | 8192, 1, 1, 1 | -16964.929688 | -0.5844,0.3708,-0.6714 | -4.8532,-5.0353,-5.1983 |
| Q per-head RMS_NORM | norm-3 | RMS_NORM | 256, 16, 1, 1 | 102.466438 | -0.5562,0.3529,-0.6391 | 0.0945,0.6886,-0.9071 |
| Q q_norm MUL | Qcur_normed-3 | MUL | 256, 16, 1, 1 | 102.664566 | -0.7387,0.4956,-0.8825 | 0.1315,0.9643,-1.1516 |
| Q ROPE | Qcur-3 | ROPE | 256, 16, 1, 1 | 102.664566 | -0.7387,0.4956,-0.8825 | 0.1315,0.9643,-1.1516 |
| attn_k MUL_MAT | Kcur-3 | MUL_MAT | 1024, 1, 1, 1 | -62.642960 | 1.6857,1.0914,-0.6894 | -0.8354,-0.2739,-1.5043 |
| K per-head RMS_NORM | norm-3 | RMS_NORM | 256, 4, 1, 1 | -29.754698 | 0.7995,0.5177,-0.3270 | -0.3668,-0.1203,-0.6605 |
| K k_norm MUL | Kcur_normed-3 | MUL | 256, 4, 1, 1 | -42.430355 | 1.1166,0.7411,-0.4637 | -0.5101,-0.1685,-0.8386 |
| K ROPE | k_cache_in-3 | ROPE | 256, 4, 1, 1 | -42.430355 | 1.1166,0.7411,-0.4637 | -0.5101,-0.1685,-0.8386 |
| attn_v MUL_MAT | Vcur-3 | MUL_MAT | 1024, 1, 1, 1 | 46.230995 | -0.0227,-0.0059,-0.2507 | -0.2829,1.3612,-0.9048 |
| K cache SET_ROWS | cache_k_l3 (view) | SET_ROWS | 1024, 256, 1, 1 | -42.443554 | 1.1162,0.7412,-0.4636 | 0.0000,0.0000,0.0000 |
| V cache SET_ROWS | cache_v_l3 (view) | SET_ROWS | 1024, 256, 1, 1 | 46.232056 | -0.0227,-0.0059,-0.2507 | 0.0000,0.0000,0.0000 |
| FLASH_ATTN_EXT (softmax path) | node_281 | FLASH_ATTN_EXT | 256, 16, 1, 1 | 184.928223 | -0.0227,-0.0059,-0.2507 | -0.2830,1.3613,-0.9048 |
| attn gate SIGMOID | gate_sigmoid-3 | SIGMOID | 4096, 1, 1, 1 | 120.543068 | 0.0250,0.0179,0.0105 | 0.0077,0.0065,0.0055 |
| attn gated MUL | attn_gated-3 | MUL | 4096, 1, 1, 1 | -0.834408 | -0.0006,-0.0001,-0.0026 | -0.0022,0.0088,-0.0050 |
| attn_output MUL_MAT | attn_output-3 | MUL_MAT | 4096, 1, 1, 1 | 4.167934 | 0.0025,0.0078,0.0126 | 0.1053,0.0039,0.0391 |
| attn residual ADD | attn_residual-3 | ADD | 4096, 1, 1, 1 | 2.255422 | -0.2579,0.3772,0.0663 | 0.1805,-0.0550,0.0670 |
| ffn SWIGLU | ffn_swiglu-3 | SWIGLU | 12288, 1, 1, 1 | -4.810397 | 0.0386,-0.0028,0.0009 | 0.0488,0.0035,0.0899 |
| block output ADD (l_out-3) | l_out-3 | ADD | 4096, 1, 1, 1 | 0.738097 | -0.3339,0.5213,0.0561 | 0.1576,-0.0235,-0.0068 |

Attention is dumped with **FLASH_ATTN_EXT** enabled, so there is no separate `MUL_SUM`/`SOFT_MAX`/`MUL_SUM` chain in this log: QK^T, softmax and PV are one node `node_281 = FLASH_ATTN_EXT(Qcur-3 (view) (permuted){...}, cache_k_l3 ... )`. To align a non-fused softmax path, use `runs/dbg_fa_off.log` for that block, or validate against this node's sum only.

## Graph-level landmarks (1-token step)

| role | node name | op | shape | sum | head3 \| tail3 |
| --- | --- | --- | --- | --- | --- |
| token embedding GET_ROWS | node_0 | GET_ROWS | 4096, 1, 1, 1 | -0.010437 | -0.0153,0.0000,-0.0153 | 0.0000,0.0138,0.0138 |
| input_embed (hadamard sign MUL) | model.input_embed | MUL | 4096, 1, 1, 1 | -0.464154 | 0.0002,0.0116,0.0131 | 0.0066,-0.0030,-0.0027 |
| final RMS_NORM | norm | RMS_NORM | 4096, 1, 1, 1 | -50.572254 | -0.3923,1.2082,0.6322 | 0.6327,-0.2332,0.4010 |
| result_norm MUL output_norm | result_norm | MUL | 4096, 1, 1, 1 | -46.624306 | -0.8796,2.5297,1.4026 | 1.3791,-0.4992,0.8771 |
| result_output MUL_MAT (248320 logits) | result_output | MUL_MAT | 248320, 1, 1, 1 | 43374.937500 | -1.8606,-0.4644,-0.0287 | 2.0350,2.1095,1.3751 |

## Zero / NaN flags

- exact-zero `sum`: **434** of 2424 nodes; NaN `sum`: **0**. No computed activation (anything that is not a cache/state/index view) sums to exactly zero, so there are no dead branches in the graph.
- 145 of the zeros are structurally empty tensors (`shape` with a 0 dimension, e.g. the `{24576, 0, 1, 1}` GET_ROWS/CPY pairs the fork uses to no-op the state rotation); the rest are the cache/state reads and their views listed below.
- reproduce with: `awk -F'\t' '$6=="0.000000"' runs/oracle_step1.tsv`

| layer | zero-sum nodes |
| --- | --- |
| 0 | 20 |
| 1 | 18 |
| 2 | 18 |
| 4 | 18 |
| 5 | 18 |
| 6 | 18 |
| 8 | 18 |
| 9 | 18 |
| 10 | 18 |
| 12 | 18 |
| 13 | 18 |
| 14 | 18 |
| 16 | 18 |
| 17 | 18 |
| 18 | 18 |
| 20 | 18 |
| 21 | 18 |
| 22 | 18 |
| 24 | 18 |
| 25 | 18 |
| 26 | 18 |
| 28 | 18 |
| 29 | 18 |
| 30 | 18 |

## Cold-cache evidence (this is the important caveat)

Every *persistent state input* of the 1-token step reads exactly zero, even though the same tensors were written with non-zero data at the end of the prefill graph. Per category (sums from the two parsed graphs):

| category | layers probed | non-zero reads | layer-0 read sum (step 1) | layer-0 write sum (step 1) | layer-0 write sum (prefill) |
| --- | --- | --- | --- | --- | --- |
| GDN state `cache_s_lN` | 24 | 0 | 0.000000 | 63.934986 | 97.733658 |
| conv history `cache_r_lN` | 24 | 0 | 0.000000 | -324.560303 | -272.361328 |
| KV cache `cache_k_lN` | 8 | 0 | - | - | - |

- GDN state reads that carried a non-zero history into the 1-token step: **0 of 24** GDN layers; conv-history reads: **0 of 24**. The buffers the prefill left behind were plainly non-zero (layer 0: GDN state 97.733658, conv history -272.361328) yet the 1-token step reads `0.000000` from both. KV cache: the post-`SET_ROWS` sum of the 1-token step equals *this token's* roped K within f16 print noise (l3: -0.013199; l7: 0.002899; l11: -0.006954), i.e. the 2 prefill rows are absent; had they been kept the gap would be of the order of the prefill's own cache sum (-69.628746; 91.477669; -40.311989).

Consequences for the engine work:

1. **The step-1 graph is a single-token step from a ZERO state.** It is a valid oracle for node *structure* (shapes, offsets, fused-output layout, op order, the conv window `3+1 -> 4` concat, the `{4096,129}` GDN output) and for the cold-start values (`S = 0`, conv history `= 0`), but it cannot discriminate anything that only shows up with a warm state: `decay before vs after dot(S,k)`, conv left-padding vs the real window, or the delta-rule correction term all collapse to the same numbers when `S = 0`.
2. To get the warm oracle we actually need, re-dump with **at least 2 decode tokens after the prefill** (so the final graph reads a state written by a *previous* graph) and check the same probe: the `cache_s_lN`/`cache_r_lN` GET_ROWS sums must be non-zero. If they are still zero, the fork is not persisting state across evaluations at all and its decode path is broken - which would itself be a finding worth acting on.
3. Meanwhile the prefill graph *does* contain a warm, multi-token recurrence: its GDN/conv nodes run a 2-step scan internally (`{4096,130}` GDN output, `{5,8192}` conv input), so row-1 of every prefill node is the "state after token 1" case. Use it for the ordering tests, and use `runs/oracle_step1.tsv` for the single-token shapes/offsets.
4. The attention path in this step also sees only 1 cached K/V row, so its softmax is over a single element: `FLASH_ATTN_EXT` returns exactly that one V row, replicated over the GQA head groups - the dumped ratio `sum(node_281) / sum(attn_v MUL_MAT)` is 4.0001 (= 16 q heads / 4 kv heads, f16 noise aside). Not an attention oracle either; use the prefill graph's per-row prints for that.
5. Layer map read off this graph: 24 GDN blocks (0,1,2,4,5,6,8,9,10,12,13,14,16,17,18,20,21,22,24,25,26,28,29,30) and 8 softmax-attention blocks (3,7,11,15,19,23,27,31), i.e. attention every 4th layer starting at 3.

## How to regenerate

```
cd "<repo-root>"
PYTHONIOENCODING=utf-8 python tools/oracle_step1.py
```

