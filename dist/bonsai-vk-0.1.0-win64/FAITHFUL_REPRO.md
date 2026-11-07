# 忠实复现（Faithful Reproduction）五层对齐验证

底线：绝不引入额外偏差，让模型输出尽可能接近它在训练/参考实现下的行为。本文件只记实测数字与方法，不记推测。
参考实现 = PrismML 的 llama.cpp 构建（引擎 b10777-e3de71522，Vulkan 后端）；被测对象 = 我们的 `runtime_vk/`。

| 层 | 内容 | 状态 |
| --- | --- | --- |
| 1 | Tokenizer 逐 ID | 已完成：39/39 探针 1:1（32 通用 + 7 日韩） |
| 2 | PTQ1_0 反量化路径 | 已完成：真实 1000 块 128000 权重 `max\|err\| = 0.00000`；合成 3840 块 491520 权重（含 `d==0`／次正规／负 scale／±Inf／±NaN）同样全 0 |
| 3 | 注意力 | 已完成：层 3 全链路（norm→rope→kq→softmax→PV→gate）`max\|err\|` 在 8/32/64 三档 = `6.7e-04 / 2.9e-03 / 3.6e-03`，占满量程 `6.4e-06 / 2.7e-05 / 3.2e-05`，且小于 fp16 缓存格式自身的舍入代价（比值 0.000/0.141/0.262） |
| 4 | KV Cache 布局 | 已完成：position-major 自有 arena，prefill 一把灌 vs 逐 token 灌 **64 步 0 位差异**；死槽严格 0.0 |
| 5 | 采样 | 未开始（用户明确后置） |

---

## 1. Tokenizer 逐 ID 对齐

### 方法（两端都真跑，不比对代码）

- 参考侧：`prism-vk/prism/llama-tokenize.exe -m <gguf> -f <probe-file> --ids --no-bos`（实测单次 ~0.8 s，只装词表不装张量）。
  探针字节先写成临时文件再 `-f` 喂进去，中间不过任何 shell/argv 编码，`tools/oracle_tokenize.py` → `runs/tok_ref.jsonl`。
- 我们侧：`runtime_vk/bonsai_tok.h` + `runtime_vk/bonsai_gguf.h`（自己重写的 byte-level BPE 与 GGUF 读取，不含官方一行 C++）。
- 比对：`runtime_vk/align_tokenizer.exe` 逐个探针打印两侧 id 列表，只有**完全相等**才算过。

### 结果（实测）

```
[bpe] vocab=248320  ranks=247587  specials=33  unk=0  eos=248046
==== tokenizer alignment: 32/32 probes byte-identical ====     (runs/tok_probes.txt)
==== tokenizer alignment:  7/7 probes byte-identical ====      (runs/tok_probes_kr.txt，日/韩/混排专门集)
```

用户指定的那句：

```
probe 1  bytes=33  text=用一句话解释三值模型。
  OURS (7): [95826, 110827, 98682, 95986, 96326, 103725, 1710]
  REF  (7): [95826, 110827, 98682, 95986, 96326, 103725, 1710]
  MATCH 1:1
```

其余 38 条的完整 id 列表在 `runs/tok_align.txt`（含 OURS/REF 两行与 `MATCH 1:1`）。抽几条有代表性的：

| 探针 | 文本 | id 列表（两侧一致） |
| --- | --- | --- |
| 3 | `Hello world, this is a test of the 2026 emergency number 123.` | `[9419, 1814, 11, 411, 369, 264, 1228, 314, 279, 220, 17, 15, 17, 21, 12483, 1324, 220, 16, 17, 18, 13]` |
| 8 | `Multiple   spaces<tab>and<tab>tabbed<tab>` | `[31032, 256, 12258, 50711, 56196, 2634, 197]` |
| 9 | `Line1\nLine2\n\ntrailing\n` | `[2380, 16, 198, 2380, 17, 271, 371, 13868, 198]` |
| 13 | `。！？，、；：（）《》【】` | `[1710, 190500, 3709, 5205, 24178, 4960, 9616, 7313, 26076, 24273, 10586, 10638]` |
| 19 | `日本語のテキスト과 한국어 텍스트` | `[247359, 15303, 210342, 51879, 190925, 225287]` |
| 21 | 官方 fork 自己日志里的 `Thinking Process:` 段 | `[90700, 8340, 25, 271, 16, 13, 220, 2972, 2014, 53983, 279, 5952, 64700]` |
| 30 | `1234567890` | `[16, 17, 18, 19, 20, 21, 22, 23, 24, 15]` |
| 31 | `０１２３`（全角） | `[25191, 19496, 24128, 32405]` |
| kr5 | `のテキスト과 한국어` | `[15303, 210342, 51879, 190925]` |

### 两条决定性的格式事实（都是量出来的，不是抄来的）

1. **初始切分粒度必须是字节，不是码点。** 先按 UTF-8 码点起手的实现只能过 31/32 和 1/7，每个失败点都是「我们给一个独立空格 220，参考给合并后的 id」——例如 ` 한국어` = 190925、` 국` = 149289、` 한국` = 159140。改成字节起手（每个字节先成 token，再纯按 rank 合并）后 32/32 + 7/7。开关留在 `bonsai_tok.h`：默认 byte，`BONSAI_TOK_INIT=cp` 可复现输的那版。
2. **GPT-2 bytes_to_unicode 的配对是「位置对位置」，不是 `fwd[byte]=cp`。** 候选字节表 `bs` 从 33 开始（33..126, 161..172, 174..255，然后把不存在的字节按递增补进去、码点从 256 起）。早期按 `bs[i]` 与 `i` 配对写过一次，结果所有多字节词表查找静默错位——这也是第一次对齐跑出来的数对不上却看不出报错的原因。

另外两个当场抓出来的 Bug（都是 UB/格式类，不是调参）：

- `align_tokenizer.cpp` 的 `load_ref()` 取了 `line.substr(...).c_str()` 这个**临时对象的指针**，扫描循环在已释放内存上跑，`strtol` 读出垃圾 → 某些运行 `std::bad_alloc` 退出 3，某些运行看起来正常。改成先绑到具名 `std::string` 后，32 探针集才在重定向到文件时稳定跑完（先前那次「32/32」是 UB 恰好没炸，结果仍然重跑验证过）。
- 合并表必须在装载时过滤：拼接结果不在词表里的 merge 不能进 rank 表，否则贪心循环提前 break 并把原始字节倒出来。

---

## 2. PTQ1_0 反量化路径保真度

### 方法（三方对比，参考侧是真在卡上跑的官方着色器）

| 方 | 是什么 |
| --- | --- |
| GPU-REF | 官方 `vulkan-shaders/dequant_ptq1_0.comp` **本体逐字未改**（仓库根 `_dequant_ptq1_0.comp`），由 `tools/gen_dequant_ref_comp.py` 把我们写的 `dequant_head.glsl` 内联进去（glslang 不认官方构建期的 `#include`），glslang 编成 SPIR-V 后在 RX 550X 上执行 |
| CPU-REF | `ptq_decode.h::ptq_dequant_official_cpu()`：把那支着色器的算术（`pow3[]` 乘、`(v*3)>>8 - 1`、它自己的 `il/l` 元素编号）逐行转写成 C++，**故意不复用**我们的映射函数 |
| OURS | `ptq_decode.h::ptq_dequant_ours()`：我们的运行时用的解码，`bonsai_vk.cpp` 现在直接 `#include "ptq_decode.h"`，所以测的就是引擎本体而不是测试里的副本 |

块是**真的**：100 个 28 字节块从 `data_start + tensor.off + blk*28` 直接读出，落在 100 个不同张量上（`output.weight`、`blk.0.attn_gate` … `blk.31.ffn_down`），每块出处都打印，可复算。

### 结果（实测）

```
[feat] storageBuffer8Bit=1 storageBuffer16Bit=1 uniformAndStorage16Bit=1 shaderFloat16=0
[gpu] nel=12800  blocks=100  local=256  groups=7
[census] weights by sign: -d 4224, 0 4194, +d 4382  (of 12800)
[census] block scales: |d| in [0.0091095, 0.0519104], d==0 0, subnormal-range 0, negative 0

  max|GPU-REF - CPU-REF| = 0.00000   (12800/12800 bit-exact)
  max|OURS   - CPU-REF| = 0.00000   (12800/12800 bit-exact)
  max|OURS   - GPU-REF| = 0.00000   (12800/12800 bit-exact)
  symbol agreement      = 12800/12800 weights, 0 disagreements
  runtime dot form d*(cs-xs) vs sum(w_e*x_e): max abs 0.000e+00  max rel 0.000e+00
```

扩测到 1000 块（覆盖全部 202 个 PTQ1_0 张量、128000 权重）：三个 `max|err|` 与符号一致性同样全为 0。

### 合成块把 f16 边缘分支封死（原文这一节曾写「本测试不覆盖」，现已被实测推翻并补做）

真实 100 块的 census 里 `d==0`、次正规、负 scale 一个都没有（`|d|` 全落在 0.0091..0.0519），所以那三条 f16 分支当时确实**没有真权重依据**。本轮按承诺用合成块补上：`align_dequant.exe <gguf> synth` 造 3840 个块、491520 个权重，scale 覆盖 15 个类别（`+0/-0`、次正规 min/mid/max、`-次正规 max`、正规 min/max、`-正规 max`、`+Inf/-Inf`、`+NaN/-NaN`）× 每类 32768 权重，权重字节走 `(q+7j)&0xFF` 的确定性图案，出处逐块打印。

```
[census] block scales: finite nonzero |d| in [-65504, 65504]; d==0 512, subnormal 1024, negative 1024, non-finite 1024
  scale                bit GPU-REF    bit OURS    max|err|      NaN      Inf     hard
  +0 / -0                 32768/32768    32768/32768     0.00000        0        0        0
  subnormal min/mid/max   32768/32768    32768/32768     0.00000        0        0        0
  -subnormal max          32768/32768    32768/32768     0.00000        0        0        0
  normal min / -normal min / 1.0 / normal max / -normal max
                          32768/32768    32768/32768     0.00000        0        0        0
  +Inf                    32768/32768    32768/32768     0.00000    10880    21888        0
  -Inf                    32768/32768    32768/32768     0.00000    10880    21888        0
  NaN / -NaN              32768/32768    32768/32768     0.00000    32768        0        0
      bit-exact 491520/491520 (100.000%)   NaN/Inf-aware equal 491520/491520   非有限不匹配 0
```

`NaN` / `Inf` 两列数的是 **CPU-REF（官方算术的 C 转写）在该类里真产出的非有限值个数**（例如 `d = ±Inf` 且 `trit = 0` 时 `d*(cs-xs)` 按 IEEE 产出 NaN，一类里 10880 个 NaN + 21888 个 Inf；`d = NaN` 类则 32768 个全是 NaN）——它们是「这个类会走到非有限路径」的说明，不是误差。真正的判定量是另外三列：GPU-REF 与 OURS **各自**对 CPU-REF 都 `32768/32768` 逐位一致、`max|err| = 0.00000`、`hard = 0`（`hard` = 「至少一侧非有限且两侧不相等」的对数，即无法测出幅度的分歧，最坏情形）。程序退出码本身就是断言：任何一个 `max|err| ≠ 0` 或任何一个 `hard` 都会让它非 0 —— 实测 `exit 0`。

### 一条被实测纠正的前提

`block_ptq1_0.d` 必须是 **`float16_t`（半浮点值）**，不是 `uint16_t`（位模式）。用 `uint16_t` 时官方本体里的 `const float d = float(data_a[ib].d)` 把位模式当成了数值：首轮实测 `max|GPU-REF - CPU-REF| = 10916.94809`，恰好只有 4194 个零权重相等（零不受 scale 影响），而真实块 scale 全在 0.0091..0.0519。改成 `float16_t` 后为 0。旁证：`_ggml-vulkan.cpp:1483` 自己写着 "FLOAT_TYPE in the shader is float16_t with fp16 support, otherwise float"。

同理，这张卡（Polaris/gfx803）**能**用 8/16 位 SSBO 存储（`storageBuffer8Bit/16Bit` 均为 1），但 `shaderFloat16=0`：着色器只做 `float(d)` 提升到 f32 再算，不需要 f16 ALU，与此一致。

### 运行时回归

重构（把 PTQ 解码搬进 `ptq_decode.h`）后 `bonsai_vk.exe` 重新编译无警告，原有自检与正确性不变。最后一次全量构建 + 引擎自检用的是**与本文档所述完全相同的源**（`runs/build_full_3.txt`，实测；每次现编 12 支 SPIR-V + 5 个 exe）：

```
[diag] upload / x / xsum / scale / codes : 五项全 match CPU exactly
correctness: max|err|=0.00001  max|ref|=72.2  rel=1.06e-07  (K=4096 f32 累加，预期 ~1e-6 rel)
arena 24.2 MiB 单分配 | transfer-only 队列上传 10.5 MiB / 1701.8 us = 6.48 GB/s
repack 11.0 MB -> 12.6 MB / 0.73 ms | M=12288 K=4096 splits=4: 0.285 ms/token-op  176.3 Gop/s  44.1 GB/s
cross-check: wall 0.287 ms vs timestamps 0.285 ms = 99.5%
```

上一次的同类构建（`runs/build_full_2.txt`）给出 `0.287 ms / 175.1 Gop/s / 43.8 GB/s`、上传 `1550.2 us = 7.11 GB/s`。两份的**正确性行逐字相同**，差别只在墙钟：GEMV 吞吐的 run-to-run 散布约 **0.7%**（175.1–176.3 Gop/s），上传带宽散布约 **9%**（6.48–7.11 GB/s，受同时占用 PCIe 的其它活动影响）。因此引用这两个数时按区间看，别当成三位有效数字的定值。

这里的 `max|err|=0.00001` 是 **GEMV 累加顺序**带来的 f32 残差（GPU 分 4 路 split-K 累加 vs CPU 按 double 顺序累加），不是反量化差异——反量化差异由上表的 0.00000 单独证明。

---

---

## 3. 注意力（层 3）

### 3.1 逻辑分成两半，各自有各自的证据

| 证据 | 工具 | 定的是什么 |
| --- | --- | --- |
| 官方 fork 自己打印的中间张量 | `tools/attn_softmax_check.py`、`tools/rope_fit.py`（oracle = `runs/dbg_attn_oracle.log`、`runs/dbg_rope_b*.log`，由 `llama-debug.exe -p "one two three four" -n 0 -c 32 -ngl 0 -fa off --tensor-filter 'kq-3\|kq_soft_max-3\|attn_inp'` 产出） | **规格**：哪一步有 scale、scale 是多少、rope 的配对与底数、mask 的填值语义 |
| 我们卡上的着色器 vs 我们的 fp32 镜像 | `runtime_vk/align_attn.exe`（`attn_norm_rope.comp` 133 行 + `attn_qk_softmax.comp` 99 行 + `attn_pv.comp` 57 行 vs `attn_vk.h` 451 行，全部自写，官方 C++ 一行未抄） | **数值**：GPU 实现与规定语义逐位差多少 |

规格侧的硬数字（实测，打印量子 5e-5）：scale 在 30 组比较上自由拟合 = **0.062501**，`1/sqrt(256)` = 0.062500，差 `7.13e-07`；`1/sqrt(64)`、`1/8`、`1.0`、`1/sqrt(128)` 全被否掉（`max|err|` 0.069~0.523）。rope 定律拟合出的最小漂移解 `D = 64` = `qwen35.rope.dimension_count`，形式为
`out_p = x_p*cos(s·base^(-2p/64)) − x_{p+32}*sin(s·base^(-2p/64))`，`p = 0..31`，`dims 64..255` 直通，`s` = 绝对位置。
worked example（head group 0，query token 2）：官方 `p = [0.5506, 0.1369, 0.3126]`，我们按 `1/sqrt(256)` 算出 `['0.5506','0.1369','0.3126']`；按 `1/sqrt(64)` 会得 `['0.7225','0.0447','0.2329']`。

### 3.2 结果（实测，`align_attn.exe <seq> 256`，真实层 3 形状 head_dim=256 / 16q·4kv / n_rot=64）

| 档 | q16 不匹配 | kc | vc | p16 | score `max\|err\|`（占 \|kq\|max） | prob | context | layer-out（占 \|gated\|max） |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| seq=8 | 2/16384 | **0**/131072 | **0**/131072 | **0**/16384 | 6.70e-04（6.40e-06） | 2.21e-06 | **0.000e+00（100% 逐位）** | 1.19e-07（1.01e-07） |
| seq=32 | 24/65536 | 3/131072 | 0 | 28/65536 | 2.94e-03（2.72e-05） | 2.33e-05 | 2.28e-04 | 1.45e-04（1.16e-04） |
| seq=64 | 68/131072 | 17/131072 | 0 | 154/131072 | 3.57e-03（3.16e-05） | 2.49e-05 | 4.10e-04 | 2.56e-04（1.98e-04） |

三档 `hard` 列（非有限值不匹配）全 0，`max|sum(p)−1|` ≤ `3.5e-07`。把阶段 1 从环里拿掉（A2：直接把镜像的 fp16 arena 喂给阶段 2/3）后 **score 逐位 100% 一致、`max|err| = 0.00000000`**，这就把「matmul/softmax/PV/gate 的算术」与「rope+norm 的超越函数」分开了：前者与镜像同序同形即逐位相同，后者才是残差来源。

与格式自身代价的比值（G/V 腿，float64 理想链）：

```
seq=8   our 6.695e-04  vs fp16 布局代价 1.889e-03  ->  ratio 0.000
seq=32  our 2.945e-03  vs 1.621e-03                ->  ratio 0.141
seq=64  our 3.567e-03  vs 1.564e-03                ->  ratio 0.262
```

即：**我们的偏差比官方 fp16 缓存格式强制引入的舍入误差还小一个量级到四倍区间**，不是「官方算对而算错」的项。

### 3.3 结构断言（要求是 0，不是误差）

```
B  因果前缀：live 槽 576/8448/33280，死槽里非 0.00000000 的个数 = 0（三档）
   活前缀里塞 -INF：7/31/63 行，卡上写出非 0 概率的行数 = 0；杀完仍归一 max|sum-1| = 9.5e-08/1.6e-07/2.6e-07
C  position=0 的缓存行 vs 只 norm 的值：0/128 词差异；dims 64..255 直通：0 / 1 / 3 个 fp16 词（8/32/64）
   theta 的两种写法 powf(base,-2p/nrot) vs powf(powf(base,-2/nrot),p)：max|d theta| = 1.49e-08 @ p=3（相对 2.36e-07），
   传进旋转后的行 = 1.25e-06（32 档），远小于此处 fp16 操作数量子 ~6e-4 —— 缓存之后两种写法分不出来
D  GQA：16 个 q 头喂同一个 Q，同 kv 头的 192/768/1536 对全一致（0 对不等），不同 kv 头 0 对「碰巧相等」
E  KV arena：同一份内容，prefill 一把灌 vs 逐 token 灌 8/32/64 步 -> gated 行差异 0，概率槽差异 0，
   最大 prefill/decode 分歧 0.00000000
F  fp16 打包器：host RNE vs 我们的 pack2_rne() 0/2048；host RNE vs 卡的 packHalf2x16() 1366/2048（66.70%）；
   host RTZ vs 卡的 packHalf2x16() 0/2048，且 1366 个分歧「全部是 |值| 不变大」—— 这支指令是**向零舍入**，
   所以引擎自己手写了 RNE（shaders/f16_rne.glslh，55 行）。卡的 unpackHalf2x16() 与我们的 f16_to_f32() 0/4096 逐位同
```

### 3.4 Debug 循环里真正挖出来的五条（每条都是先有错判、后有实测）

1. **13.48 的 score 误差不是卡造成的。** 首轮 leg A 报 `score max|err| = 13.4824`、q16 有 661 个词不对，卡在 `word 17`（dim 34/35）给 `(0.282715, 2.226562)` 而 CPU 给 `(1.226562, -1.886719)`。真因是**我们镜像里 `attn_rope(cfg, y, pos, y)` 原地调用**：循环写 `y[p]` 之后，后面的配对还要读 `y[p+half]` 的原值，被覆盖的是自己的输入。同一段代数改成 `const float lo = x[p], hi = x[p+half];` 再写两处输出，误差从 13.48 掉到 **6.70e-04**（2 万倍），q16 不匹配 661→2、kc 171→0、pregate 到 100% 逐位。项的顺序保持 `hi*cs + lo*sn`，与着色器的 lane 一致。position 0 时 `sin = 0`，所以这个 Bug 在最短探针上完全隐形。
2. **卡的 `packHalf2x16()` 是向零舍入**（leg F，1366/2048 vs RNE，0/2048 vs host RTZ，且分歧方向 100% 是「不放大绝对值」）。官方缓存是 RNE 建的，所以 GLSL 原生打包函数不能用，`f16_rne.glslh` 是必须的而不是优化。
3. **FMA 合并假设被否证。** 直通维（`d ≥ n_rot`，里面根本没有超越函数）也在差，第一嫌疑人是 `ss = ss + x*x` 被编译成 `fma`（未舍入的乘积直接进加法，是另一个数）。于是把 `x*x` 先写进 shared 数组 `sq[]`、barrier 后再由 lane 0 归约（编译器无法再折叠回去），重编重跑：**leg A 与 leg R 的计数逐位不变**（32 档 `Q: 31008/131072，15213 pass-through` 前后一模一样）。假设排除，改动保留（它与镜像同形，零代价）。
4. **残差是一个标量：每行的 norm 常数 `inv`。** 新增 leg R2：对每一行取镜像的 `inv = 1/sqrt(ss/hd+eps)`，把它按 fp32 ulp 平移 `k ∈ [-4,+4]`，用平移后的 `inv` 重算整行并与卡上读回的 fp32（`qf`/`kf` 调试流）逐位比。

   ```
   seq=8   Q 128 行：108 行直通维本来就逐位一致，20 行把 inv 移 ±1 ulp 后全对（-1:16, +1:4），0 行不可解释；
           3604 个直通差异 100% 属于「一个标量平移就能复现」的行；平移后直通残差 0，残差 2420 全在 dims<n_rot
   seq=32  Q 512 行：426 干净，86 需 ±1 ulp（-1:58, +1:28），0 不可解释；15213 个直通差异 100% 被解释，残差直通 = 0
           K 128 行：103 干净，25 需 ±1 ulp，0 不可解释；4492 个 100% 被解释
   seq=64  Q 1024 行：846 干净，178 需 ±1 ulp（-1:123, +1:55），0 不可解释；31515 个 100% 被解释
   ```
   结论：卡与镜像的 stage-1 分歧 = **每行一个 fp32 ulp 的 `inv` 差 + `cos/sin` 的最后几位**，没有第三项，也没有「代数写错」的样本。上游证据（leg T，卡 vs libm）：工作带内 `sqrt` 最大相对误差 **1.00 fp32 ulp**（bit-differ 15.21%）、`1/sqrt` **1.16 ulp**（28.59%）、`1/x` **1.00 ulp**（10.69%）；`cos/sin` 在 63 rad 的最大角度下绝对差 **39.12 / 38.50 ulp**；`exp` 相对 **4.62 ulp**。这些都是驱动/硬件的正确舍入级差异，在着色器里除了自己实现正确舍入的 `sqrt`（牛顿迭代）无法消除，本轮**没有**为它改引擎——按「先证明、再动手」的规矩，它记为已知下限而不是隐藏。
5. **harness 自己越界，差点被当成驱动的缺陷。** seq=64 时 leg T 一次上传/回读 1.19 MB，而 staging 环只有 1.05 MB：`readback()` **根本没有尺寸保护**（`vkCmdCopyBuffer` 写过目标末尾 + `memcpy` 读过映射末尾），`upload()` 的 one-shot 分支又 `vkUnmapMemory(c.dev, c.stage.m)` 解错了句柄（解的是环，不是它写的那个 buffer）。当时的「实测」输出是 `1/sqrt(0.755595) = 76.59`（libm 1.15）、`1/0.755595 = −5865.75`（libm 1.32）、`exp(sigmoid)` bit-differ 24.9% / `max|rel| = 9.037`——一个「这卡的除法错 4433 倍」的结论。两处修好并重跑：`1/x` 回到 **1.00 ulp**、`exp(sigmoid)` bit-differ 13.19% / `max|rel| = 1.30e-07`；`sqrt / 1/sqrt / 1/x` 的工作带最大相对误差在 8/32/64 三档分别是 `1.00/1.16/1.00`、`1.00/1.09/1.16`、`1.00/1.00/1.00` fp32 ulp（8/32 两档从来没走过 one-shot 路径，所以它们本来就是干净的参照），`exp(sigmoid)` 的 bit-differ 三档 `13.19% / 13.01% / 13.19%`。同时加了永久哨兵：leg T 现在先把上传后的参数缓冲回读，与宿主数组逐位比，三档都是 **0/33632、0/140576、0/297504**。这条写进文档是因为它是本轮最贵的一课：**归因给硬件之前，先证明输入输出通道自己是干净的。**

### 3.5 本轮层 3 的最大误差（回答「Attention 的最大误差是多少」）

| 指标 | seq=8 | seq=32 | seq=64 | 单位 |
| --- | --- | --- | --- | --- |
| 绝对最大（score，未归一） | 6.70e-04 | 2.94e-03 | **3.57e-03** | 与 \|kq\|max≈113 同量纲 |
| 归一后（占满量程） | 6.40e-06 | 2.72e-05 | 3.16e-05 | 无量纲 |
| 层输出（gated） | 1.19e-07 | 1.45e-04 | 2.56e-04 | \|gated\|max≈1.29 → 相对 1.0e-07 / 1.2e-04 / 2.0e-04 |

授权里给的判据是「降到纯浮点舍入级别（如 < 1e-5）」：按**归一化**口径，seq=8 是 `6.4e-06` 已过；seq=32/64 是 `2.7e-05 / 3.2e-05`，绝对值随上下文长度增长而增长是 fp16 格点与超越函数尾数共同决定的，其上界就是 §3.2 的格式代价（比值 0.141 / 0.262，仍小于格式强制误差）。逐位口径的另一半：**V 竞技场（无 norm、无 rope）三档 0 不匹配**，`p16` 在 seq=8 也是 0 不匹配。

---

## 4. KV Cache 布局（层 4）

我们的 arena 是**自持的 Vulkan buffer**，position-major、连续、`uint` 存打包半浮点（word `w` 装元素 `2w`/`2w+1`），刻意不抄 ggml 的 `{hd, cap, nkv}` permuted 视图。实测证据：

- leg E：同一份 K/V 内容，prefill 一次灌 vs 逐 token 灌，`8/32/64` 步下 gated 行差异 `0`、概率槽差异 `0`、最大分歧 `0.00000000`——布局不引入任何数值差。
- leg A：`vc`（V 竞技场）三档 `0/131072` 不匹配；`kc`（K 竞技场，要过 norm+rope）三档 `0/3/17`。
- 死槽（causal 未覆盖的位置）在 `prob` 里严格 `0.00000000`，三档 0 例外，与官方打印的 `exp(-inf - max) = 0` 一致（`attn_softmax_check.py` 的 30 组比较）。
- 容量核算（实测推导，非估算）：单个注意力层的 K+V arena = `2 × cap × nkv × hd × 2 B`；`cap=256` 时 = `2×256×4×256×2 = 1048576 B = 1.000 MiB`，每 token 每层 `4096 B`，8 个注意力层 → **32768 B/token = 32 KiB/token**（与 STAGE4 记的数一致），`cap=256` 全 8 层 = 8.00 MiB。
- **注意（未闭合，不糊）**：这三支注意力着色器和 `attn_vk.h` 目前只在 `align_attn.exe` 里真跑，`bonsai_vk.cpp` 还没有 `#include "attn_vk.h"`，端到端 forward 仍走官方 fork 的图。也就是说层 3/4 的**语义与数值已经对齐并封死，工程接入还没做**。

## 复现命令

```
runtime_vk\build_mingw.bat                        :: 生成 oracle 源 + 编 12 支 SPIR-V + 编 5 个 exe（一次跑完）
python tools/oracle_tokenize.py                   :: 官方参考 id -> runs/tok_ref.jsonl
runtime_vk\align_tokenizer.exe <gguf> runs/tok_probes.txt   runs/tok_ref_32.jsonl   :: 32/32
runtime_vk\align_tokenizer.exe <gguf> runs/tok_probes_kr.txt runs/tok_ref_kr.jsonl  ::  7/7
runtime_vk\align_f16.exe                          :: 半浮点转换全域 65536/65536
cd runtime_vk && align_dequant.exe <gguf> 1000    :: 真实块；也跑 1000 扩测
cd runtime_vk && align_dequant.exe <gguf> synth   :: 合成边缘块 491520 权重
cd runtime_vk && align_attn.exe 8 256             :: 层 3 全链路，也跑 32 / 64
python tools/attn_softmax_check.py                :: scale 与 -INF 语义（对官方 dump）
python tools/rope_fit.py                          :: rope 配对/底数/n_rot（对官方 dump）
```

本轮归档（原始输出，未剪辑）：`runs/build_full_3.txt`（最后一次源码改动后的全量构建，即「表里的数与盘上的源同一份」）、`runs/build_full_2.txt`（上一次同类构建）、`runs/reg_tok.txt`、`runs/reg_tok_kr.txt`、`runs/reg_f16.txt`、`runs/reg_dq1000.txt`、`runs/reg_dq_synth.txt`、`runs/reg_attn8.txt`、`runs/reg_attn32.txt`、`runs/reg_attn64.txt`、`runs/rebuild_8.txt`/`rebuild_32.txt`/`rebuild_64.txt`（重编后的同一 exe 复跑：抹掉 `NNN.NNN ms` 墙钟字段后与 `reg_attn*.txt` 逐字相同）、`runs/reg_spec_softmax.txt`、`runs/reg_spec_rope.txt`；debug 过程的对照档 `runs/attn_s32.txt`（合并 FMA 假设**改前**）与 `runs/attn_fma_a32.txt`（**改后**）——两份的 leg A/leg R 数字逐个相同，是 §3.4 第 3 条的证据本体。`runs/build_full_3.txt` 之后又用最终链接出的 5 个 exe 把/token、f16、dequant 全部重跑一遍，归档为 `runs/reg2_{tok,tok_kr,f16,dq100,dq1000,dq_synth}.txt`，与对应的 `reg_*` **逐字节相同**（`diff -q` 无输出），即本文所有数字都由「盘上这份源」产生（注意力那三支的复跑对照见上一段的 `rebuild_*`）。

未覆盖 / 待做：**采样对齐（层 5）**；**把三支注意力着色器与 `attn_vk.h` 接进 `bonsai_vk.exe` 的端到端 forward**（现在只有 harness 在跑）；chat template 与特殊 token 端到端；`rope_probe_nrot4.gguf`（约 2 GiB 的探针副本，用来把 `n_rot` 从模型元数据里「拟合」出来的，原模型从未改动）用完该删。
