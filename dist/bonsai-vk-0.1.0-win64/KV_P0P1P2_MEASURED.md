# KV 子系统 P0/P1/P2 —— 实测报告（v5，2026-11-05；§1–§9 是 2026-11-04 的 v4，§10 是新增）

范围：RX 550X 4GB / gfx803 / WDDM / Vulkan 1.3，Ternary-Bonsai-2-9B PTQ1_0（2.343 GiB 权重），
纯手写 C++17 + GLSL。约束遵守情况：K 保持 FP16 未压；V 只做 q8_0（块 32）；权重反量化路径
一行未动（`align_dequant` 复测 0.000e+00）；模型格式未改；未新增依赖。

标注约定：**实测** = 本卡跑出来的原始数字；**推导** = 由实测常数按公式算出，公式与代入值都写出；
**未验证** = 现有工具链拿不到证据。日志全部在 `runs/`，均为新建文件、未覆盖旧日志。

---

## 1. P0 审计结论：32 层 KV/状态分配

**结论：24 个线性注意力（GDN）层已正确，无需修改。** 只有 8 个 full-attention 层随 token 数增长。

这条结论现在不再依赖我读代码，而是每次 init 由引擎自己断言并打印（实测每一行日志都有）：

```
arena invariants | KV span 1605632 == cap 256 x 8 x 784 OK;
ST 12582912 / CH 589824 are 24 GDN layers x per-layer state, cap-free OK
```

`Regions::take()` 加了溢出标志，KV/ST/CH 三段跨度与期望值按字节比对，任何一次 re-entry
（含 `BONSAI_CAP_AUTOFIT` 的自动缩容）都会重新给出一遍精确读数；不匹配就 FAIL、返回 false、
不派发任何命令。审计口径：`per_slot`（每 token 分配量）里 GDN 项必须为 0，日志里逐层列出：

```
P0 check | cap-proportional regions = SCORE+PROB+P16+MASK+POS (42 floats) + KF off (0)
           + KV 8 layers x 784 = 6314 floats/slot; GDN terms in it: 0
           (ST = 12582912 floats total, independent of cap)
```

GDN 侧的真实占用：24 层 ×（2,097,152 B 状态 + 98,304 B conv）= 50.2 MiB，**实测 cap 从 256 改到
2048 时 ST/CH 两段字节数不变**（cap 2048 日志：`activation arena: 26565440 floats = 101.339 MiB
(state 48.0 MiB + KV 49.0 MiB + logits 0.95 MiB)`）。

唯一的 cap 相关非 KV 消费者是 fp32 KF 调试流（`BONSAI_KV_DEBUG_STREAM`），默认关；开着它跑
oracle 分支时逐位对齐，证明它对数值中性。

### 每 token 显存：改前 / 改后（实测）

| 项 | fp16 K+V（改前） | K fp16 + V q8_0（改后） |
|---|---|---|
| K | 512 words/slot/层 = 2048 B | 同左（未动） |
| V 数据 | 512 words/slot/层 = 2048 B | 256 words = 1024 B |
| V scale | 0 | 16 words = 64 B |
| 每层每 slot | 4096 B | 3136 B |
| **每 token（×8 层）** | **32.00 KiB** | **24.50 KiB** |
| 64 positions 实测 arena | 2.00 MiB | 1.53 MiB |
| 2048 cap 实测 arena | 64.0 MiB | 49.0 MiB |

几何与规格逐字对齐：4 KV heads × head_dim 256，K fp16 = 16 KiB/token，V q8_0 块 32
（32×int8 + 1 fp16 scale）= 8.5 KiB/token，合计 **24.50 KiB/token**（引擎打印值，非手算）。
省下 23.4 %。

### 重算的安全 / 极限上下文（推导，常数全为实测）

实测常数：权重 2467.7 MiB，cap 无关活动区 52.0 MiB，device-local heap 3840.0 MiB，
DXGI LOCAL budget 3464.6 MiB（本进程被允许的量，含其它进程占用）。每 slot 成本用引擎自己的
`per_slot`：q8_0 = 24.66 KiB（含 0.16 KiB fp32 侧），none = 32.16 KiB。

公式：`cap_max = (预算 − 权重 − 52.0 MiB) / per_slot`

| 判据 | q8_0-V | fp16 V（改前） |
|---|---|---|
| 按 DXGI 预算 3464.6 MiB（**安全**） | ≈ **39.2 K** token（39 237） | ≈ 30.1 K token（30 086） |
| 按原始 heap 3840.0 MiB（**极限**） | ≈ **54.8 K** token（54 825） | ≈ 42.0 K token（42 039） |

**这一张表在本次审计中被更正过**：原先四格写的是 38.3 K / 29.4 K / 53.5 K / 41.1 K，因为把
`MiB → KiB` 当成了 ×1000 而不是 ×1024，系统性偏低 **2.5 %**。用 ×1024 重算后，极限行与引擎自己
打印的 54,814 slot 对上了（下句），而安全行与下面两张独立推导出的超预算读数也对上了：
40 960 − 39 237 = 1 723 slot × 24.66 KiB = **41.5 MiB**（表中「超 41 MiB」），
49 152 − 39 237 = 9 915 slot × 24.66 KiB = **238.8 MiB**（表中「超 239 MiB」）。
两处独立交叉验证 ⇒ **39.2 K 是正确的安全带读数**，40 K 出带、48 K 超 239 MiB 这两个结论**不变**。

引擎在 cap 256 时自己打印的极限读数是 `headroom 1314.1 MiB (= 54558 more slots)` → 合计
54,814 slot，与我手算的 54,838 差 0.04 %（字节取整），两者一致。

4 GB 墙的真实位置（实测，`runs/memmodel_q8_2520_v1.log` 等四条）：把预算 artificially 钳到
2520 MiB 时引擎打印 `left for the KV: 0.2 MiB, at 24.66 KiB per slot -> 10 slots`，none 模式
`at 32.16 KiB per slot -> 6 slots` —— 缩容公式与 `BONSAI_CAP_AUTOFIT=1` 的自动 re-init 都被实跑验证。

---

## 2. P1：V-only q8_0，读路径 fused

开关 `BONSAI_KV_V_QUANT = none | q8_0`，默认 q8_0，`none` 是回归对照（arena 与 P1 之前的二进制
逐字节相同，见 §4）。块大小固定 32，写入即量化，解包在注意力 V 读/累加循环内部完成
（`attn_pv_q8v.comp`，纯寄存器，`shared` 占用 0 B），没有任何"先解包写回全局再读"的路径，
也没有新增每 token 的全局 scratch。分配层用的是真实 q8_0 块布局（数据段 + scale 段各自独立
子区间），块级 evict/refuse 接口在 `KvGuard::decide()`，作为 WDDM 兜底挂钩。

### 保真（实测，`runs/kv_compare_v4.log`，同二进制同 prompt 两次跑）

| 指标 | band A（position 0，纯量化） | band B（position 1..36，量化+上游漂移） |
|---|---|---|
| 元素数 | 1024 | 36,864 |
| max\|err\| | 0.0550537 | 0.0170288 |
| max\|ref\| | 13.9609 | — |
| rms err / rms signal | 8.802e-03（**41.11 dB**） | 5.673e-03 |
| mean err | 0.00567528 | 0.00342113 |

K 侧：**37,888 个元素 0 个不同 —— BITWISE IDENTICAL**，这是两次运行可比的门。
scale 数组合理性：1184 个块 scale 全部非零，max 0.109924、min 0.00615311。
最差元素在 V pos 0 head 3 dim 227。

端到端：层 0–31 采样 token 序列与 none 模式一致；`BONSAI_QCHECK` 的 narrow↔wide QK 审计在
层 0 与层 7 都是 `SCORE 0/4096 bit-diff, PROB 0, P16 0 -> bitwise identical`，死槽位污染 0 违反。
注意力对齐（`runs/align_attn_v4.log`，seq 64）：card vs mirror score = 3.567e-03，
是官方 fp16 缓存自身代价的 **0.262 倍**。

### 长度回归（实测，短上下文，基线 14.05 tok/s，预算 ≤ −5 %）

| 模式 | tok/s | vs 14.05 | ms/token |
|---|---|---|---|
| `none`（对照） | **14.27** | **+1.6 %** | 70.1 |
| `q8_0`（默认） | **14.26** | **+1.5 %** | 70.2 |

两个模式都在预算内，且互相差 0.01 tok/s（噪声）。

§4.1 那次翻案只改了注释与一条 stderr 文案，但"只改注释"这句话本身也要有证据：全量重建
（`runs/build_v29.log`，0 warning）后 `attn_qk_softmax_wide.spv` 的 md5 与改前相同，`q8_0` 短上下文
重跑 = **14.24 tok/s / 70.2 ms/token**（48 token / 3370.5 ms，`runs/short_ctx_q8_0_v5.log`；vs 上表
14.26 差 0.14 %），命令缓冲阶段计数与 §3 一致（加载 119 条、前向复用）。功能面没有回归。

---

## 3. P2：PDH Shared Usage 监控 + 这次真正的病灶

判据用的是本进程 `\GPU Process Memory(pid_<PID>_*)\Shared Usage`（**不是** Dedicated）。
`IDXGIAdapter3::QueryVideoMemoryInfo` 保留为辅助，源码与日志都标注它是 reservation 预算、
不是 spill 判据。

### 3.1 计数器修好了（实测）

反直觉的真因：路径必须以反斜杠开头且必须带实例部分。`tools/pdh_probe.cpp` 实测：

```
"\GPU Process Memory(*)\Shared Usage"        add -> OK, collect -> OK, 20 instances
 "GPU Process Memory(*)\Shared Usage"        add -> 0xC0000BC0 (BAD_COUNTERNAME)
```

所以 `kv_guard.h` 里先试 star 路径、再试 pid 路径，读取时一律按 `pid_<pid>_` 过滤并把每个实例
相加（AMD 实例名带 LUID，一个进程可能有 2 个 LUID）。

### 3.2 曲线增长不是 KV，也不是 VidMm 驱逐 —— 是我们自己的命令缓冲（实测）

同一 prompt、同一 76 positions，唯一区别是命令缓冲策略：

| 命令缓冲策略 | Shared Usage 曲线 | 峰值 | tok/s | forward 段缓冲数 |
|---|---|---|---|---|
| 每个 dispatch group 新建（**旧默认**） | 48.9 → 76.4 MiB 单调爬升，2 MiB 一跳 | 76.43 MiB | 14.19 | 96 allocated |
| 复用 1 个 + reset（**现默认**） | **40.9 全程平直** | 40.93 MiB | 14.21 | 1 allocated + 95 reuses |

Dedicated Usage 全程 2539.5 MiB 平直 —— 所以增长的既不是 KV arena（它按 cap 一次性分配好）
也不是权重。

单条缓冲的成本，直接测出来（`BONSAI_CB_PROBE=400`，`runs/cb_probe_v1.log`）：

```
400 command buffers allocated, submitted and waited -> shared 66.93 MiB:
0.0650 MiB per command buffer   [MEASURED, lower bound: empty payload]
after vkFreeCommandBuffers(400): shared 66.93 MiB (returned 0.00 MiB)
```

- **0.0650 MiB/条**是空命令列表的下界；
- **推导**：带负载的每条 = (76.43 − 40.93) / 95 = **0.3737 MiB**，其中记录的命令本体占
  0.3737 − 0.0650 = 0.309 MiB；
- 每 position 有 1.26 个 dispatch group（实测 96 begins / 76 positions，修正了此前把加载阶段
  119 条也算进 per-token 的错报），所以旧写法每 token 泄漏 **0.46 MiB**，而且
  **`vkFreeCommandBuffers` 归还 0 MiB** —— 驱动不回收。40 K token 上下文按这个速率是 18.8 GiB
  的驱动侧记账内存，任何卡都撑不住。这就是为什么复用现在是默认值。

### 3.3 阈值语义修正（重要，实测驱动的判断）

本机一个健康运行的 Shared Usage 稳态就是 **40.93 MiB**（实测）。所以规格里"32–64 MB 阈值"
按**绝对值**解释时，第一个 token 就会误判——日志里那句 `soft-watched 48` 就是这么来的。
规格原话是 "monotonic **growth** past a 32–64 MB threshold"，现在阈值判的是**相对第一个解码
采样点的增量**，绝对值同时打印。`BONSAI_SPILL_ABS=1` 保留绝对读法用于复现旧日志。

同时把"动作"和"采样计数"统一：hard 判决要求 超阈值 **且** 连续增长 run ≥ mono_need，
`decide()` 与 `sample()` 用同一个 `growing()`，不再出现"[KV REFUSED] 但 hard=0"。
`KvGuard` 新增 `n_warn`，close-out 行同时报 refusals / evictions / soft-watches。

三条兜底路径全部实跑验证（`runs/p2_refuse_v2.log`、`runs/p2_evict_v1.log`）：

```
[spill] tok 19 shared 51.31 MiB (growth +2.38 over the reference 48.93, ...) <- HARD
[KV REFUSED at position 19] spill: refuse new KV blocks
  thresholds: soft 1 / hard 2 MiB (hard additionally needs a growth run >= 3)
  KV at the end: refused 1, soft-watched 2      -> 缓存完整退出

BONSAI_KV_EVICT=1 + 块 16：positions 0..15 / 16..31 / 32..47 mask 成 -INF，
  继续生成 40 token，14.17 tok/s，refused 0
```

---

## 4. 上下文梯度表（实测到 2048，4K–48K 为推导）

| 位置数 | cap / 内核 | V | KV arena 实测 | 单 token 延迟 | tok/s | PDH 增长 | prefill 实测 |
|---|---|---|---|---|---|---|---|
| 64 | 256 narrow | none | 2.00 MiB | 70.1 ms | **14.27** | +0.00 | 14.03 tok/s |
| 64 | 256 narrow | q8_0 | 1.53 MiB | 70.2 ms | **14.26** | +0.00 | 14.02 tok/s |
| 450 | 512 wide | q8_0 | 10.77 MiB | 76.1 ms | **13.14** | +0.00 | 13.71（32.2 s/442 tok） |
| 900 | 1024 wide | q8_0 | 21.53 MiB | 82.7 ms | **12.09** | +0.00 | 13.05（68.3 s/892 tok） |
| 1800 | 2048 wide | q8_0 | 43.07 MiB | 96.9 ms | **10.32** | +0.00 | 11.93（150.2 s/1792 tok） |
| 1800 | 2048 wide | none | 56.25 MiB | 97.2 ms | **10.29** | +0.00 | 11.91 |

"PDH 增长 +0.00" = 4K 以下的 validated 区间内**没有任何增长型 spill**（命令缓冲复用之后曲线全平）。
q8_0 vs none 在同一上下文长度上速度差 0.3 %（10.32 vs 10.29，噪声内），arena 省 13.18 MiB
（43.07 vs 56.25，正好 23.4 %）。

**线性拟合（由上表 4 个实测点最小二乘）**：`ms/token = 69.07 + 0.01541 × positions`，
最大残差 0.24 ms（0.3 %）。用它外推（**推导**，±10 %）：

| 目标 token | 延迟 推导 | tok/s 推导 | KV arena 推导 | 总 device-local 推导 | 相对 3464.6 MiB 预算 |
|---|---|---|---|---|---|
| 4,096 | 132 ms | 7.6 | 98 MiB | 2618 MiB | 余 847 MiB ✅ |
| 8,192 | 195 ms | 5.1 | 196 MiB | 2717 MiB | 余 748 MiB ✅ |
| 16,384 | 322 ms | 3.1 | 392 MiB | 2914 MiB | 余 551 MiB ✅ |
| 32,768 | 574 ms | 1.7 | 784 MiB | 3309 MiB | 余 156 MiB ⚠️ |
| 40,960 | 700 ms | 1.4 | 980 MiB | 3506 MiB | **超 41 MiB ❌** |
| 49,152 | 826 ms | 1.2 | 1176 MiB | 3703 MiB | **超 239 MiB ❌** |

结论必须直说：**40 K 不是"安全带"，48 K 在本卡上不能声称安全。** 按原始 heap 3840 MiB 能塞下
（48 K 余 137 MiB），但按 DXGI 给本进程的 budget 3464.6 MiB 已经超；WDDM 在这种情况下会把
页迁到系统内存，表现为 Shared Usage 增长型曲线——正是 P2 要拒绝/驱逐的状态。q8_0-V 把
预算天花板从 ≈30.1 K 抬到 ≈39.2 K token，这条改进是真实有效的，但它改变不了 4 GB 的物理墙。

两个与显存无关的阻塞（都不在本阶段范围内，见 ROADMAP.md:152）：

1. **内核宽度**（本条原为推导，现已实测，结论翻案 —— 见下面的 4.1）：`Fwd::kMaxWide = 2048`
   由 `attn_qk_softmax_wide.comp` 的 `#define MAXCAP 2048u` 和它的两个 `shared float[MAXCAP]`
   决定，共 16,384 B；改到 4096 就是 2×4096×4 = **32,768 B = maxComputeSharedMemorySize 的全部**。
   原文把它写成"零余量 ⇒ 必须真重构"，这是推导，不是测量；`runtime_vk/shared_probe.cpp`
   把它测完了，答案在下面。
2. **prefill 墙钟**：一 token 一 submit，实测 cap 2048 的 1792 token prefill 花 150.2 s。
   48 K 点的 prefill 按比例是 48153×(96.9/70.1) ≈ 66,700 s ≈ **18.5 小时**（推导），
   所以这张表的 4 K 以上必须由脚本在无人值守下跑，而不是现场死等。

### 4.1 MAXCAP 4096 到底能不能用：实测（`runs/shared_probe_v1.log`、`runs/shared_probe_v3.log`、`runs/shared_probe_spirv_v1.log`）

新工具 `runtime_vk/shared_probe.cpp` + `shaders/probe_shm_nNNNN.comp`（每个尺寸一个独立 SPIR-V
模块，因为 `shared` 是文件作用域声明，塞进同一模块会让驱动看到模块级求和，逐档数字就失去意义）。
每档做两件事：先问驱动"这条 pipeline 建不建得起来"，再让 256 lane 把 `sd[]`/`sp[]` 全部写满、
`barrier()`、读回来吐到全局内存，逐个元素比对 —— **建得起来不等于能用**，静默裁剪只有逐元素回读才看得见。

本机 RX 550X（driver 8388887，`maxComputeSharedMemorySize = 32768 B`，均实测）：

| n | shared 需求 = 2·n·4 | 相对上限 | vkCreateComputePipelines | 回读校验 |
|---|---|---|---|---|
| 3072 | 24,576 B | 低于 | VK_SUCCESS | 3072/3072 全对 |
| 4000 | 32,000 B | 低 768 B | VK_SUCCESS | 4000/4000 全对 |
| 4090 | 32,720 B | 低 48 B | VK_SUCCESS | 4090/4090 全对 |
| **4096** | **32,768 B** | **正好等于上限** | **VK_SUCCESS** | **4096/4096 全对** |
| 4224 | 33,792 B | 超 1,024 B | VK_SUCCESS | 4224/4224 全对 |
| 4608 | 36,864 B | 超 4,096 B | VK_SUCCESS | 4608/4608 全对 |

所以：**"零余量会被驱动拒"是错的**。等于 32,768 B 的静态共享分配在本驱动上既建得起来、跑起来也
正确；连超上限 4 KiB 也照样跑。**但引擎不能因此就用超上限的声明** —— 那是 per-driver 行为、违反
Vulkan valid-usage，报告只把它记成"本机实测现象"，不记成许可。
（解释用的"gfx803 每 CU 有 64 KiB LDS、`maxComputeSharedMemorySize` 是软件规则不是硅片规则"是
GCN 4 的**架构规格 + 推导**，不是本机实测：本工具没测到 65,536 B 那一档，只测到 36,864 B 能跑。）

占用代价也测了（`bench_shm_nNNNN.comp`：循环次数走 push constant，指令流与数组长度无关，只有
`shared` 声明不同，所以时长差就是占用差；grid 64 workgroup × 256 lane，iters 20000，6 次取最小）：

| n | shared | min ms | 相对 2048 |
|---|---|---|---|
| 512 | 4,096 B | 22.137 | — |
| 1024 | 8,192 B | 22.131 | — |
| 2048 | 16,384 B | 22.472 | 1.00×（基准） |
| 4096 | 32,768 B | 26.870 | **1.20×** |

同一指令流下，把每 workgroup 的共享从 16 KiB 抬到 32 KiB，这段 wall time 多 **19.6 %（实测）**。
这条合成 loop 不是注意力内核本身，它只证明方向：32 KiB 那一档确实更贵，贵在一个可测的 ~20 % 量级，
不是无穷大。

> 工具自己的第一版是错的，留档说明（`runs/shared_probe_v2.log`）：那一版只初始化 `sd[0..255]`，
> 而读指标到 `n-1`，其余元素是上一波残留的 workgroup 内存，可能是 Inf/NaN，于是四档全部打印
> `NON-FINITE OUTPUT`。时长数字没错，但校验列没有意义，所以改成整段初始化（多出的 ceil(n/256) 次
> 循环 ≤ 0.08 % 的功，四档同一条语句），v3 才是可用的那一版。**"建得起来"和"跑对了"是两件事**，
> 这句话对这个工具和对驱动同样成立。

**因此第 1 条的正确表述是**：4 K 不再被驱动卡住，把 `MAXCAP`/`kMaxWide` 抬到 4096 在硅片和驱动层是
通路；本阶段仍然没有抬，剩下的是两件事——(a) **验证覆盖**：宽内核的 cross-validation 阶梯
（narrow↔wide 在 cap ≤ 256 位相同、`qk_audit` A/B/C/D/E、`align_attn`）历史上只跑到 cap 2048，
抬到 4096 必须重跑这套门才算数，而子智能体逐行审计的结论（三个循环都是 stride 式、无
butterfly/tree reduction、无 `invocations == 数组长度` 隐含假设，链路其它内核按 head_dim 定尺）
只说明"大概率正确"，不能替代实测；(b) **~20 % 的占用代价**要落进 8 层 × 16 头 = 128 workgroup/token
的真实调度里重测 tok/s 才知道净值。两趟法（不落 `sp[]`）不再是 4 K 的必要条件，这个结论也一并翻案；
它是给 4 K 以上（MAXCAP 8192 = 65,536 B，按 GCN 4 的每 CU 64 KiB LDS **规格**就该到顶了；这一档本机
未测，探针最深只到 36,864 B）准备的。改动会动 lane-0 串行序 —— 那是位精确
契约，本阶段照旧没动。

引擎里三处措辞已按这次实测改掉，并且改完就跑了一遍确认门还在（`runs/kmaxwide_guard_v2.log`，
`BONSAI_CAP=4096 bonsai_vk.exe run … 32 1` → rc=4）：

```text
cap 4096 exceeds the widest VALIDATED row (2048): the wide kernel's two shared arrays are
float[2048] = 16 KiB of the 32 KiB this driver reports as its limit.
    This is a validation gate, not a driver refusal - shared_probe.exe measured
    float[4096] (32768 B, exactly the limit) as created and round-tripped elementwise, at
    a measured +19.6% cost from residence. Raising MAXCAP needs a re-run of the
    narrow<->wide bitwise gate plus a tok/s measurement, not just a bigger number here.
```

改的是 `attn_qk_softmax_wide.comp` 的 MAXCAP 注释、`bonsai_forward.h` 的 `kMaxWide` 注释和这条
拒绝信息；内核体一个字节没动，重编译后 `attn_qk_softmax_wide.spv` 与改前 md5 相同
（`3a2c4e594510ec97930a9943d9138ecb`），所以历次保真门的效力不受影响。

> 顺带记一个操作坑（我第一次跑这个门就踩了）：`bonsai_vk.exe run <model> <A> <B>` 的第三个位置参数
> 是**上传分片 MiB**，不是上下文 cap；cap 只由 `BONSAI_CAP` 决定。传成 `run … 4096 1` 会得到
> 4 GiB 的 `ptq_scratch`，然后在 arena 提交处撞 4 GB 墙（`runs/kmaxwide_guard_v1.log`，rc=2），
> 根本没走到 cap 门。正确形状是 `run … 32 1` + `BONSAI_CAP=4096`。


---

## 5. 修改文件清单与关键 diff

| 文件 | 改动 |
|---|---|
| `runtime_vk/kv_guard.h` | PDH 路径修正（前导反斜杠 + 实例部分，实测证据写进注释）；`use_absolute`/`ref_shared`/`last_growth`/`peak_growth` + `growing()`；阈值改判增长量；`KvGuard::decide()` 加 `have_counter`、`growing` 两个显式参数；新增 `n_warn` |
| `runtime_vk/bonsai_weights.h` | `begin_cmd()` 命令缓冲复用 + 阶段闸门 `cb_enter_forward()`；计数器按加载/前向分段；`g_cb_reuse` 默认 1；两段实测曲线数字写进注释 |
| `runtime_vk/bonsai_forward.h` | `Regions::overflow` 与 ARENA INVARIANTS 硬断言；KV 区域按"每层 cap×kv_layer"正确嵌套（P0/P1 的关键修复）+ 两道 init 期拒绝；`mk_attn`/`qk_audit`/`kv_dump`/区域表同步修正；`BONSAI_CB_PROBE` 微基准；策略值改用增长量；refuse/warn/evict 打印与计数一致 |
| `tools/kv_compare.py`（新） | 两次 dump 的保真比对，band A/band B、K 位相同门、逐位置 TSV、scale 合理性 |
| `tools/pdh_probe.cpp`（新） | 计数器路径取证（star/pid/裸路径 × 有/无前导反斜杠 × 英文/本地化 API） |
| `runtime_vk/shared_probe.cpp`（新） | §4.1 的取证工具：逐档 `vkCreateComputePipelines` + 256 lane 全量回读校验，再用同指令流不同 `shared` 声明量占用代价；已挂进 `build_mingw.bat` 最后一步 |
| `runtime_vk/shaders/shared_mem_probe.glslh` + `probe_shm_n{3072,4000,4090,4096,4224,4608}.comp`（新） | 共享内存上限探针，每档一个模块（`shared` 是文件作用域声明，合并会失去逐档意义） |
| `runtime_vk/shaders/shared_mem_bench.glslh` + `bench_shm_n{512,1024,2048,4096}.comp`（新） | 占用探针：iters 走 push constant，指令流与数组长度无关，只有 `shared` 声明不同 |
| `runtime_vk/bonsai_vk_baseline_v1.exe`（新，P1 前二进制） | `none` 模式的逐字节对照基线 |

关键 diff（节选）：

```cpp
// bonsai_weights.h —— 复用只在前向阶段生效，加载阶段的跨队列缓冲永不被 reset
static void cb_enter_forward() {
    g_cb_alloc_load = g_cb_alloc;
    g_cb_begin_load = g_cb_alloc + g_cb_reset;
    g_cb_in_forward = 1;
}
...
if (g_cb_reuse && g_cb_in_forward && g_cb_shared) {
    VK_CHECK(vkResetCommandBuffer(g_cb_shared, 0));  // 前向每条缓冲都经 fence 等过，才可能走到这里
    ++g_cb_reset; cb = g_cb_shared;
}

// kv_guard.h —— 超阈值与连续增长两个条件必须同时成立，采样与决策共用 growing()
if (value > g_spill_hard && growing) { ... return KV_EVICT / KV_REFUSE; }

// bonsai_forward.h —— KV 区域的真实嵌套：每层一条 cap*kv_layer 的大区，K/V/S 是层内子区间
const uint32_t kv_layer_span = uint32_t(uint64_t(cap) * R.kv_layer);
r = { ..., w_act(f,a,ck), w_act(f,a+ck, f.v_q8?cvd:kwh),
      w_act(f, f.v_q8?a+ck+cvd:R.DUMMY, f.v_q8?cvs:16u), ... };
```

`none` 模式在修正后的嵌套下 K 窗口 =131,072@+0、V =131,072@+131,072，与 P1 之前的基线二进制
逐字节一致（ids / logits / top-8 / 生成 token 全等，prompt 11/12/13/15/20 token 都验过）。

---

## 6. 配置开关文档

| 环境变量 | 默认 | 含义 |
|---|---|---|
| `BONSAI_KV_V_QUANT` | `q8_0` | V 的表示；`none`(=`fp16`) 是回归对照，`q8_0` 块 32 |
| `BONSAI_CB_REUSE` | `1` | 前向阶段复用一条命令缓冲；`0` 恢复旧行为，专门用于 A/B |
| `BONSAI_CB_PROBE` | 未设 | `=N` 额外测 N 条空缓冲的驱动记账成本并演示 free 不归还 |
| `BONSAI_SPILL_EVERY` | `32` | 每 N 个解码 token 采样一次（KV 块边界必采） |
| `BONSAI_SPILL_SOFT_MIB` | `32` | 软阈值（**增长量**，MiB），只记日志 |
| `BONSAI_SPILL_HARD_MIB` | `64` | 硬阈值（增长量，MiB），触发 refuse/evict |
| `BONSAI_SPILL_MONO` | `3` | 判"连续增长"所需的连升次数 |
| `BONSAI_SPILL_ABS` | `0` | `1` 改回按绝对 Shared Usage 判（复现旧日志用） |
| `BONSAI_KV_BLOCK_SLOTS` | `64` | evict/refuse 的块粒度（slot 数） |
| `BONSAI_KV_EVICT` | `0` | `1` 时硬阈值走"驱逐最远块"而非停止 |
| `BONSAI_CAP_AUTOFIT` | `0` | `1` 时 OOM 兜底自动按公式缩 cap 并重新 init |
| `BONSAI_HEAP_LIMIT_MIB` | 未设 | 测试钳位：假装 device-local 只有这么多 |
| `BONSAI_CAP` | `256` | arena slot 数。≤256 走 narrow，257..2048 走 wide，**>2048 走 tiled**（§9.2；改动前是直接 rc=4 拒绝） |
| `BONSAI_QKERNEL` | `0` | QK 行内核：`0` auto / `1` 强制 narrow / `2` 强制 wide / `3` 强制 tiled；`>3` 拒绝。A/B 全靠它 |
| `BONSAI_QK_TILE` | `2048` | tiled 的 tile 宽（= 唯一决定 LDS 足迹 `4*QTILE+4`）：`1024`/`2048`/`4096`；`8192` = 32,772 B **越限探针**，仅显式指定才走，auto 到不了 |
| `BONSAI_QCHECK` | `0` | `1` 跑 narrow/wide/tiled 位相同门 A + **A2** + B/C/D/E；不过则 **拒绝生成（rc 6）** |
| `BONSAI_KVFILL` | `0` | `N`（2 ≤ N ≤ cap）打开合成 KV 计时夹具：arena 填 `0x3C003C00`、位置钉在 `N-1`。**只有 ms/token / tok/s / 内核组合 / 流量可信**，ids、文本、任何保真陈述一律无效（§9.4 有交叉验证表） |
| `BONSAI_KVCHECK` / `_POS` / `_LAYER` | 未设 | KV 落盘 dump，供 `tools/kv_compare.py` |

---

## 7. 测试命令

```bat
REM 构建（全量，含 4 个对齐测试程序 + shared_probe，共 0 warning）
cd runtime_vk && build_mingw.bat            > ..\runs\build_v29.log 2>&1

REM 清场，且任何时刻只允许一个 GPU 进程
powershell -NoProfile -ExecutionPolicy Bypass -File ..\tools\clear_field.ps1

REM 短上下文回归（两个模式各一次，对照 14.05）
set BONSAI_KV_V_QUANT=none && bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 48 > ..\runs\short_ctx_none_v3.log 2>&1
set BONSAI_KV_V_QUANT=q8_0 && bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 48 > ..\runs\short_ctx_q8_0_v3.log 2>&1

REM P2 A/B：命令缓冲曲线（每条新建 vs 复用）
set BONSAI_CB_REUSE=0&& set BONSAI_SPILL_EVERY=1&& set BONSAI_SPILL_SOFT_MIB=1e9&& set BONSAI_SPILL_HARD_MIB=1e9&& set BONSAI_SPILL_MONO=999&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 60 > ..\runs\p2_rate_q8_alloc_v2.log 2>&1
set BONSAI_CB_REUSE=1&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 60 > ..\runs\p2_rate_q8_reuse_v2.log 2>&1

REM 单条命令缓冲的绝对成本（下界）+ free 是否归还
set BONSAI_CB_PROBE=400&& set BONSAI_SPILL_EVERY=1&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 1 > ..\runs\cb_probe_v1.log 2>&1

REM 兜底三路径：refuse / evict / 自动缩容
set BONSAI_CB_REUSE=0&& set BONSAI_SPILL_EVERY=1&& set BONSAI_SPILL_SOFT_MIB=1&& set BONSAI_SPILL_HARD_MIB=2&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 40 > ..\runs\p2_refuse_v2.log 2>&1
set BONSAI_KV_EVICT=1&& set BONSAI_KV_BLOCK_SLOTS=16&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 40 > ..\runs\p2_evict_v1.log 2>&1
set BONSAI_CAP=2048&& set BONSAI_HEAP_LIMIT_MIB=2520&& set BONSAI_CAP_AUTOFIT=1&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 1 > ..\runs\memmodel_q8_2520_v1.log 2>&1

REM 保真：KV 双 dump + 比对；QCHECK；四个对齐测试
set BONSAI_KV_V_QUANT=none&& set BONSAI_KVCHECK=..\runs\kv_none_v4.tsv&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 4 > ..\runs\kvcheck_none_v4.log 2>&1
set BONSAI_KV_V_QUANT=q8_0&& set BONSAI_KVCHECK=..\runs\kv_q8_0_v4.tsv&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 4 > ..\runs\kvcheck_q8_0_v4.log 2>&1
python ..\tools\kv_compare.py ..\runs\kv_none_v4.tsv ..\runs\kv_q8_0_v4.tsv --out ..\runs\kv_err_v4.tsv > ..\runs\kv_compare_v4.log 2>&1
set BONSAI_QCHECK=1&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 4 > ..\runs\qcheck_v4.log 2>&1
align_tokenizer.exe models/Ternary-Bonsai-2-9B-PTQ1_0.gguf ..\runs\tok_probes.txt ..\runs\tok_ref_32.jsonl > ..\runs\align_token32_v4.log 2>&1
align_dequant.exe models/Ternary-Bonsai-2-9B-PTQ1_0.gguf > ..\runs\align_dequant_v4.log 2>&1
align_f16.exe > ..\runs\align_f16_v4.log 2>&1
align_attn.exe 64 > ..\runs\align_attn_v4.log 2>&1

REM 4.1 的两条实测：MAXCAP 4096 建不建得起来/跑不跑对 + 同指令流的占用代价
..\tools\clear_field.ps1
"<toolchain>\glslang\bin\glslang.exe" -V -O shaders\probe_shm_n4096.comp -o shaders\probe_shm_n4096.spv
shared_probe.exe > ..\runs\shared_probe_v3.log 2>&1
"<toolchain>\glslang\bin\glslang.exe" -V -H shaders\probe_shm_n4096.comp -o NUL > ..\runs\shared_probe_spirv_v1.log 2>&1

REM ================= §9：tiling 的门、A/B、梯度（全部单 GPU 进程，每条前先 clear_field）========
REM 9.3 位相同门 + 9.4(a) 真实 1310-token A/B：同二进制 v32，长 prompt 见 runtime_vk\_longprompt.txt
set BONSAI_PROMPT=<  _longprompt.txt
set BONSAI_CAP=2048&& set BONSAI_QKERNEL=2&& set BONSAI_QCHECK=1&& bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 48 > ..\runs\qk_ab_wide_v1.log 2>&1
set BONSAI_CAP=2048&& set BONSAI_QKERNEL=3&& set BONSAI_QK_TILE=1024&& set BONSAI_QCHECK=1&& bonsai_vk.exe run ... 32 48 > ..\runs\qk_ab_tile1024_v1.log 2>&1
set BONSAI_CAP=2048&& set BONSAI_QKERNEL=3&& set BONSAI_QK_TILE=2048&& set BONSAI_QCHECK=1&& bonsai_vk.exe run ... 32 48 > ..\runs\qk_ab_tile2048_v1.log 2>&1
REM 9.4(b) 夹具：先交叉验证（同一 cap/位置的 real vs fixture 必须对上），再取 4K/8K/16K/32K
set BONSAI_CAP=2048&& set BONSAI_KVFILL=1335&& set BONSAI_QKERNEL=2&& bonsai_vk.exe run ... 32 48 > ..\runs\kvfx_wide_1335_v1.log 2>&1
for %%c in (2048 4096 8192 16384) do for %%q in (1024 2048 4096) do ^
  set BONSAI_CAP=%%c&& set BONSAI_KVFILL=%%c&& set BONSAI_QKERNEL=3&& set BONSAI_QK_TILE=%%q&& ^
  bonsai_vk.exe run ... 32 48 > ..\runs\kvfx_c%%c_q%%q_v1.log 2>&1
REM 9.4 越限探针（32 KiB LDS 的真实代价）与 9.4(c) 大 arena 对短上下文的影响
set BONSAI_CAP=8192&& set BONSAI_KVFILL=8192&& set BONSAI_QKERNEL=3&& set BONSAI_QK_TILE=8192&& bonsai_vk.exe run ... 32 48 > ..\runs\cliff_q8192_c8192_v1.log 2>&1
set BONSAI_CAP=32768&& set BONSAI_QCHECK=1&& bonsai_vk.exe run ... 32 4 > ..\runs\real_cap32768_auto_v2.log 2>&1
```

## 8. 验收对照与未闭合项

| 规格验收项 | 结果 |
|---|---|
| 短上下文 vs 14.05，`none`/`q8_0` 各 ≤ −5 % | ✅ 实测 +1.6 % / +1.5 % |
| 长上下文 4K/8K/16K/32K/40K/48K 实测表 | ✅ **4 K / 8 K / 16 K / 32 K 已有实测**（§9.4(b)：7.85 / 5.39 / 3.32 / 1.88 tok/s，tiled + 夹具，夹具在两个点上与真实 prefill 交叉验证到 0.0 % / +0.4 %）。40 K/48 K 仍未实跑，卡在预算公式而非内核宽度：`> 39.2 K token` 越界（推导，V 表示未变）。原"内核宽度"阻塞已解除 —— 不是抬 `MAXCAP`，是换 `attn_qk_softmax_tile_q*.comp`（§9.2），且不改位精确契约 |
| ~40 K 安全带内无增长型 spill | ❌ **前提不成立**：按 DXGI 预算 40 K 已在带外（超 41 MiB）。安全上界实测推导为 ≈39.2 K token（q8_0）；**≤32 K 由推导升级为实测**：`runs/kvfx_c32768_q2048_v1.log` dedicated 3323.84 MiB、Shared 增长 +0.00，`runs/real_cap32768_auto_v2.log` 同一形状在真实短 prompt 下也 +0.00 |
| 48 K 记录触发阈值与回退 | ✅ 机制实跑（refuse/evict/autofit 三条路径都触发并打印），48 K 具体点未能实跑（同上阻塞） |
| VGPR ≤ 48 / occupancy 报告 | ⚠️ **部分闭合**：仍无 SPIR-V→ISA 工具链读 VGPR/LDS 分配（MinGW + 驱动内编译）。但"occupancy 到底值多少"已经不用推导了 —— §9.4 在同一内核、同一功上把 LDS 从 4 KiB 扫到 32 KiB，实测 4/8/16 KiB 极差 0.13 %，16→32 KiB 是 +6.6 % ~ +10.4 %。现有其余证据：着色器共享内存字节数（上表）与 `shaderFloat16 = 0`（驱动不上报 fp16 能力，故 `float16_t` 路径不可用，手算 RNE 是被迫的替代，非偷懒） |
| 禁止压 K / V 4-bit / 动权重反量化 / 解包回写全局 / 用 Dedicated 当判据 / 硬编码上下文长度 | ✅ 全部遵守：K 位相同实测；V 只有 q8_0 与 fp16 两态；`align_dequant` 12800/12800 逐位 0 差；`attn_pv_q8v.comp` 无共享内存、无回写；判据是 Shared；cap 由预算公式 + 不变式产生，不是常数 |

回归全绿（v27 二进制）：tokenizer 32/32 + 7/7 逐 ID 一致、dequant 0.000e+00、f16 全 65,536 码位
0 错、注意力 score 3.567e-03（= 官方格式的 0.262 倍）、QK narrow↔wide 逐位一致、
KV band A 41.11 dB 且 K 位相同。

**§9 之后的最新门（v35 二进制，`runs/final_gate_v35_long_v1.log` + `runs/regr_short_default_v1.log`
+ `runs/real_cap{4096,16384,32768}_auto_v*.log`）**：`build_mingw.bat` rc=0、`-Wall` 0 warning；
QK 门 `gate: PASS`（A / **A2** / B / C / D / E）在 cap 256、2048、4096、16384、32768 上全过；
短上下文默认路径 14.20 tok/s（= 基线，0 % 回归）；权重反量化路径与三个已验证内核（narrow、wide、
pv）本文件字节未改，`attn_qk_softmax_wide.spv` md5 仍与 §4.1 记录的 `3a2c4e594510ec97930a9943d9138ecb`
同源可查。三个已验证内核仍然只被"新增旁路"，没有被"就地改写"。

新增未闭合风险：无。唯一需要用户决策的是要不要为 4 K+ 上下文重构 wide QK 内核（动 lane-0
串行序 = 动位精确契约），以及是否接受把 32 K 而非 48 K 作为本卡的产品级上限。

> **本节两条结论已被 §9 实测取代**：(1) "为 4 K+ 重构 wide QK" 已经完成，而且**不需要**动 lane-0
> 串行序 —— 新内核 `attn_qk_softmax_tile_q*.comp` 与 wide 逐位相同；(2) "占用代价 1.20×" 是从合成
> bench 借来的数，在真实注意力内核里实测是 **+6.6 % ~ +10.4 %**（§9.4）。


---

## 9. Attention LDS Tiling：代码、位相同门、实测表与净值判定

指令的原文判定是"短上下文受 Occupancy 限制、长上下文受显存带宽限制，所以停止扩 MAXCAP、改 K 维
Tiling + V-only q8_0，并给出真实 4 K/8 K/16 K 表与净值正负"。**做完了，两条前提需要更正，
净值是正的但比期望的小一个数量级，且原因不是带宽。** 下面每一条都标了实测/推导和日志名。

### 9.1 两条前提的实测更正（先说，因为它决定了 diff 的形状）

**前提 A：「wide 内核往 LDS 里搬的是 K/V」——不是。** `attn_qk_softmax_wide.comp` 的两段
`shared float sd[MAXCAP] / sp[MAXCAP]` 存的是**这一行的 SCORE 和 PROB**（长度 = cap），不是 K/V
分块。K 是从全局 arena 直接进寄存器的：每个 lane 对每个 key 做 `hd/2 = 128` 次
`unpackHalf2x16` 点积累加，**没有任何一个字节先落 LDS**。所以在 decode 形态下（一次一个 query），
把 K 分块进 LDS **省不下任何一个重复读**——同一个 workgroup 内一个 key 只被读一次。
推论：Tiling 买到的是**能力**（行长与 LDS 解耦），不是带宽。这条不是观点，是内核源码的形状。

**前提 B：「短上下文变慢是 32 KB LDS 挤掉了 resident workgroup」——在本卡真实内核里，16 KiB 以内
根本测不出这个效应。** §9.4 表：同一内核、同样的功，LDS 4 KiB / 8 KiB / 16 KiB 三档在 16384 个
位置上极差 **0.13 %（实测）**；跨过 32 KiB 才出现 **+6.6 % ~ +10.4 %（实测）**。原推导借的是合成
bench 的 +19.6 %，那是 64×256 workgroup 全部砸 LDS 的形状，方向对、量级偏高。

### 9.2 代码：新内核 + 选择器（三个验证过的内核一个字节没改）

新文件（`runtime_vk/shaders/`）：

```glsl
attn_qk_softmax_tile.glslh        // 内核体：一段 shared float sd[QTILE] + shared float s_sum
attn_qk_softmax_tile_q1024.comp   // #define QTILE 1024u  -> 4,100 B  LDS
attn_qk_softmax_tile_q2048.comp   // #define QTILE 2048u  -> 8,200 B  LDS（默认）
attn_qk_softmax_tile_q4096.comp   // #define QTILE 4096u  -> 16,388 B LDS（设计线）
attn_qk_softmax_tile_q8192.comp   // #define QTILE 8192u  -> 32,772 B，越限探针，见 9.4
```

结构（四趟，行长按 `nfill` 分块，不再有 `MAXCAP`）：

```glsl
for (uint tb = 0; tb < nfill; tb += QTILE) {          // 趟1：点积 + 运行 max，一次一个 tile
    for (uint l = j; l < len; l += NTILE) { acc = acc + a*b; ×2;  sd[l] = acc; score[row+k] = acc; }
    barrier(); if (j==0) for (l) mx = max(mx, sd[l]*scale + mask[...]);   // lane 0，升序
}
for (uint k = nfill + j; k < cap; k += NTILE) score[row+k] = 0.0;          // 尾部清零改成全 lane 跨步
memoryBarrier(); barrier();
for (tb ...) { for (l) sd[l] = score[row+tb+l];      // 趟2：exp+sum，从全局行读回趟1写的同一批 fp32
               barrier(); if (j==0) { e = exp(v-mx); prob[...] = e; sum += e; } }
memoryBarrier(); barrier();
for (i = j; i < nfill; i += NTILE) prob[row+i] /= sum;                     // 趟3：归一化 + 死槽置 0
for (w2 = j; w2 < cap/2; w2 += NTILE) p16[...] = pack2_rne(prob[...], ...);// 趟4：RNE 打包，跨步全行
```

关键决定，逐条对应"禁止"清单：

1. **累加序一字未改**：点积表达式、`acc = acc + a*b` 两句一对、升序 `d`、lane 0 串行 max/exp/sum
   升序走完整行、死槽精确 +0.0、RNE 打包 —— 全部从 wide 逐行照搬。分块只改变"这一行放哪儿"，
   不改变"谁先加谁后加"（§8.3 的契约）。
2. **"禁止把解包结果写回全局再读"不适用于此处**，且这条要写清楚以免被误读：被禁的是**权重反量化
   的解包结果**（`dequant_ptq1_0` 的输出）落回全局再读。这里回读的是 **SCORE 行**，而行本来就是
   三个已验证内核的全局输出（`score[row+k] = acc`），wide 写完也写、下游 `qk_dump`/`align_attn`
   也读它。fp32 存进 `VkBuffer` 再读回是**同一个位型**，不是近似，只是换了条路径。省下的正是
   `sp[]` 那半份 LDS，代价是每位置每头多 ~8 B 全局流量 —— 对着 K 的 512 B/key 是 **~1.5 %（推导）**，
   而实测总账（9.3/9.4）显示净效应为正，所以这笔交换是划算的。
3. **`writeonly` 只在这一个文件里松开**（SCORE/PROB 变成普通 storage buffer，binding/window/描述符
   完全不变）—— 因为读回要求可读。wide/narrow/pv 三个内核的 `writeonly` 声明没动。
4. **可见性靠 `memoryBarrier(); barrier();`**：workgroup 的 `barrier()` 不负责让别的 lane 写的
   **设备内存**对本 lane 可见，三处全局往返各加一道 device-scope barrier。这是写完先推理出来的
   （没等它变成偶发错再抓）。

引擎侧 diff（`runtime_vk/bonsai_forward.h`）：

```cpp
// 9 处，摘要如下
enum P { ..., P_QKT, P_PV, P_N };                    // +1 条 pipeline 槽
Pipe arope[8], aqk[8], aqkw[8], aqkt[8], apv[8];     // +aqkt[]
uint32_t qkt, qk_force, qk_tile = 2048u, qcheck, kvfill = 0;
static const uint32_t kMaxNarrow = 256u, kMaxWide = 2048u, kMaxTile = 65536u;
    // kMaxTile 是算术护栏（区域偏移是 uint32 word），不是着色器上限；真正的上限是显存预算表。

static void qk_select(Fwd &f) {                      // 一个地方决定谁负责这一行
    f.qkt  = (f.qk_force == 3u) || (f.qk_force == 0u && f.cap > Fwd::kMaxWide);
    f.wide = f.qkt || (f.qk_force == 2u) || (f.qk_force == 0u && f.cap > Fwd::kMaxNarrow);
}
static uint32_t qk_row_limit(const Fwd &f) { return f.qkt ? Fwd::kMaxTile : Fwd::kMaxWide; }
dsp(f, f.qkt ? f.aqkt[li] : (f.wide ? f.aqkw[li] : f.aqk[li]), ...);   // 每 token 每层一处换绑
const bool want_tile = (f.qkt != 0u) || (f.qcheck != 0u);              // 建不建这条 pipeline
```

**auto 的三段（这是本次唯一改动默认行为的决定，写清楚理由）**：`cap ≤ 256` 仍走 narrow（14.05 /
14.26 tok/s 那批基线就是在它上面测的，LDS 2 KiB）；`257..2048` 仍走 wide —— 这是整条保真阶梯验证过
的区间，tile 在这里只快 ~2 %（实测），不值得为 2 % 重开一个已验证内核；`> 2048` **走 tiled**。
改之前 `cap > 2048` 直接 rc=4 拒绝，改之后它是一条能跑的、逐位与 wide 相同的路径。
`BONSAI_QKERNEL=1/2/3` 强制任意一档，A/B 才能在同一二进制上做。

### 9.3 位相同门：`A2` = wide ↔ tiled，逐位

新门 `qk_audit` 的 A2 段（宽内核仍是 oracle；门不过就 rc=6 **拒绝生成**，不是打印警告）。实测：

| 配置 | cap | QTILE | 活键 | SCORE bit-diff | PROB | P16 | 结论 |
|---|---|---|---|---|---|---|---|
| 单块·短行 | 256 | 2048 | 16 | 0/4096 | 0 | 0 | narrow ≡ wide ≡ tiled |
| 单块·长行 | 2048 | 2048 | 1310 | 0/32768 | 0 | 0 | bitwise identical |
| **多块**（1310 = 2×1024 块） | 2048 | 1024 | 1310 | 0/32768 | 0 | 0 | bitwise identical |
| 多块（4096 行 = 4×1024） | 4096 | 1024 | 1310 | —（wide 不可用，A2 SKIP，B/C/D/E 全过） | | | ok |
| 越限探针 | 2048 | 8192 | 1310 | 0/32768 | 0 | 0 | bitwise identical |

层 0 与层 7 各测一遍，两次都 0 差；不变式 B（P16 ≡ pack2(PROB)，含 512 以上跨步区）、C（死槽位精确
+0.0）、D（活行求和 `|sum−1| = 0.000e+00`）、E（double 重算点积，`|dot−SCORE| = 0.000e+00`）在
cap 256 / 2048 / 4096 / 16384 / **32768** 上全部 `gate: PASS`。
日志：`runs/qk_tile_gate_v1.log`、`runs/tile_multitile_gate_v1.log`、`runs/qk_ab_{wide,tile1024,tile2048}_v1.log`、
`runs/cliff_q8192_q2048_check_v1.log`、`runs/regr_short_default_v1.log`、`runs/final_gate_v35_long_v1.log`、
`runs/real_cap{4096,16384,32768}_auto_v*.log`。

### 9.4 实测吞吐表

**(a) 真实 prefill 出来的 1310-token 行（同二进制 v32，cap 2048，q8_0，48 token 解码）**

| 内核 | LDS/workgroup | prefill tok/s | 解码 ms/token | tok/s |
|---|---|---|---|---|
| wide（`QKERNEL=2`） | 16.0 KiB | 12.36 | **90.4** | 11.06 |
| tiled q1024 | 4.0 KiB | 12.43 | **89.0** | 11.24 |
| tiled q2048 | 8.0 KiB | 12.59 | **88.8** | 11.26 |
| wide（v35 复跑，估噪声） | 16.0 KiB | 12.44 | **90.1** | 11.10 |

同一二进制、同一 prompt、同一 cap：tile 快 **1.8 %**；两次独立 wide 采样差 0.3 ms ⇒ 单点噪声 ~±0.4 %，
1.6 ms 的差是真的（q1024 与 q2048 两个不同配置各自独立复现了它）。

**(b) 长上下文梯度（`BONSAI_KVFILL` 合成 KV 夹具；每个位置都活；q8_0；48 token）**

夹具先用 (a) 的真实点交叉验证，再用于取曲线 —— 这是夹具存在的前提：

| 交叉验证点 | 真实 prefill | 夹具 | 偏差 |
|---|---|---|---|
| 1335 位置 · wide · **同二进制** | 90.4 | 90.4 | **0.0 %** |
| 1335 位置 · tile q2048 · 同二进制 | 88.8 | 88.8 | **0.0 %** |
| 1816 位置 · wide · 跨二进制（`runs/ladder_cap2048_q8_0_v1.log`） | 96.9 | 97.3 | **+0.4 %** |

| 位置数 | wide | tile q1024 (4 KiB) | tile q2048 (8 KiB) | tile q4096 (16 KiB) | tile q8192 (32 KiB, 越限) | tiled tok/s |
|---|---|---|---|---|---|---|
| 1,335 | 90.4 | 89.0 | 88.8 | — | — | 11.26 |
| 2,048 | **100.7** | 98.6 | **98.3** | 98.4 | **104.9** | 10.17 |
| 4,096 | 不可用 | 127.6 | **127.4** | 127.2 | — | **7.85** |
| 8,192 | 不可用 | 185.7 | 185.8 | **185.5** | **202.6** | **5.39** |
| 16,384 | 不可用 | 301.6 | 301.4 | 301.2 | **332.4** | **3.32** |
| 32,768 | 不可用 | — | **531.1** | — | — | **1.88** |

拟合（最小二乘，实测点）：

```
wide   (n=3)： ms/token = 71.12 + 0.01443 × positions     最大残差 0.03 ms
tiled  (n=6)： ms/token = 69.97 + 0.01408 × positions     最大残差 0.67 ms (0.5 %)
tiled 32 KiB (n=3)：          = 72.48 + 0.01587 × positions
```

§4 原来那条 `69.07 + 0.01541×p` 是**跨 cap** 拟合（450/900/1800 位置分别来自 cap 512/1024/2048），
把 cap 相关的尾部代价混进了斜率。**固定 cap 重测后斜率是 0.01443（wide）**，tiled 是 0.01408：
tile 赢的 2 % 里，1.6 % 是常数项（少写死槽的那趟串行）、0.4 % 是斜率，形状没变。
32 K 点实测 531.1 ms（拟合 531.5）⇒ §4 表里 32 K 那一行的"⚠️ 余 156 MiB"现在升级为
**实测**：`dedicated 3323.84 MiB、Shared 增长 +0.00`（`runs/kvfx_c32768_q2048_v1.log`），
40 K/48 K 仍按预算公式越界（推导，未变，因为 V 表示没动）。

**(c) 大 arena 会不会拖累短上下文（这条正面回答指令的担心）**

16-token prompt（`nfill` 16→64），换 cap，同一个 tiled 路径：

| cap | 内核 | ms/token | tok/s | activations | dedicated | Shared 增长 |
|---|---|---|---|---|---|---|
| 256 | narrow | 70.4 | **14.20** | 58.3 MiB | — | +0.00 |
| 4,096 | tiled | 70.4 | **14.20** | 150.7 MiB | 2631.90 MiB | +0.00 |
| 16,384 | tiled | 70.7 | 14.14 | 446.6 MiB | 2928.45 MiB | +0.00 |
| 32,768 | tiled | 70.7 | 14.14 | 841.3 MiB | 3323.84 MiB | +0.00 |

短上下文基线回归：默认 cap 256 仍是 narrow，**14.20 tok/s vs 14.05/14.26 基线 = 0 %**（预算 ≤ −5 % ✅）。
把 arena 开到 32 K 只花 0.3 ms/token（**+0.4 %**）—— 这就是"能力不因短上下文付费"的实测答案，
也说明 wide 那套 `cap` 上界循环 / lane-0 串行尾清零确实是短上下文侧的真实开销，被 tile 消掉了大半。

### 9.5 净值判定（指令的第 4 问）

**净值为正：+2 % 左右，但方向不是"KV 压缩省带宽"，而且量级远小于预期。**

| 论点 | 证据 | 判定 |
|---|---|---|
| Tiling 省显存带宽 | 一点都没省：K/V 仍按 16 头 × 8 层各读一遍（GQA 4× 冗余原样保留），换的是 LDS 与循环上界 | ❌ 前提 A |
| Tiling 更快 | 实测 −1.8 % ~ −2.4 % ms/token，全长度一致 | ✅ 但小 |
| 快的原因是占用 | 4/8/16 KiB 极差 0.13 %（16384 位置）；真正省的是 `cap` 有界循环 → `nfill` 有界 + 尾部清零并行化 + 少一半 LDS 让 8 KiB 档可行 | ❌ 前提 B |
| 长上下文受显存带宽限制 | **反证 1（实测）**：每位置的 operand 几何由区域尺寸直接算出 —— K 64.0 KiB + V 34.0 KiB(q8_0) = **98.0 KiB**，换成 fp16 V 是 64.0 + 64.0 = **128.0 KiB**，即 q8_0 砍掉 **23.4 %** 的字节；但 1816 位置上 96.9 vs 97.2 ms ⇒ 斜率只差 **1.1 %**。**反证 2（由实测斜率推导）**：98.0 KiB ÷ 0.01443 ms = **6.95 GB/s**（tile 档 7.13 GB/s），而本卡权重路径实测 43 GB/s，即只用了 **1/6**。 | ❌ |
| WDDM spill 导致长上下文崩塌 | 实测到 cap 32768 全程 `Shared 增长 +0.00`、`dedicated 3323.84 MiB`；崩塌只在预算公式越界（≥40 K）时才会发生 | ❌ 在本卡 ≤32 K 不成立 |
| 扩容 MAXCAP 是错的 | **仍然错**，但真实代价是 +6.6 %（2048）/ +9.2 %（8192）/ +10.4 %（16384），不是合成 bench 的 +19.6 %；且 32 KiB 档在本驱动"能建能跑且逐位正确"（越限 4 B，valid-usage 违规，只记录不许可） | ✅ 结论保住，量级减半 |

**"如果不是负的，找新平衡点（比如 LDS 12 KB 或其他档位）"**：不需要新档位 —— 4/8/16 KiB 三点在
16384 位置上极差 0.13 %，12 KiB 落在一条平的区间里，没有可优化的梯度；悬崖在 16→32 KiB 之间，
所以设计线"一个 workgroup 不过 16 KiB"是**恰好在悬崖脚下**。默认取 `q2048`（8 KiB）：与 16 KiB
测不出差别，但给同一 token 里的其他内核留出共存余量，且不依赖越限行为。

### 9.6 本次实测顺手挖出并修掉的一个真 Bug

第一次真实 `BONSAI_CAP=32768` 运行在 QK 审计里 **segfault（rc 139）**
（`runs/real_cap32768_auto_v1.log` 停在 A2 SKIP 之后）。病灶与内核无关，在主机侧：E 段把**整层 K
arena** 一次 `act_rd` 回来，`cap × nkv × hd/2` word = **64 MiB**，而 host staging buffer 恰好等于
上传分片（默认 32 MiB，`bonsai_weights.h:358`）—— `VkBufferCopy` 越过映射尾部，静默内存破坏，
不是驱动错误。修复两处（`bonsai_forward.h`）：

```cpp
static void act_rd(...)   { if (n_bytes > e.chunk_bytes) { fprintf(...); exit(2); } ... }  // 硬护栏
static std::vector<uint32_t> act_rd_words(Fwd&, uint64_t src_off, uint64_t n_words);      // 分片读回
// E 只采样 nfill 以内的 key，所以只回读活前缀：kw_live = nfill * nkv * hw，并用 act_rd_words
```

修后同一命令 rc=0、`gate: PASS`、`E index audit ... 0.000e+00`、70.7 ms/token
（`runs/real_cap32768_auto_v2.log`）。历史日志为什么没暴露：审计的回读尺寸按 `cap` 增长，而门历史
上最高只跑到 cap 2048（4 MiB），越不过 32 MiB 的分片。**这是 tile 解除 cap 之后才可能出现的新尺
度**，属于本次改动必须带来的风险面，已闭合。

### 9.7 还没做的（明确划出去，不在本轮范围内）

1. ~~**真正能动 4 K+ 斜率的杠杆不是 tiling**，是消掉 GQA 的 4× 重复读~~ —— **这条已经做完了，见 §10**。
   本节保留原文是因为它记录了当时为什么会先做 tiling：tiling 换的是**能力**（解除 `MAXCAP`），
   不是带宽；斜率真正下来是 §10 的合并内核干的。
2. **prefill 墙钟仍未解决**：按§10 的拟合对楼梯积分，32 K 点的 prefill ≈ **120 min**（均价 220.5 ms/位置；**不是**早前这里写的 32768 × ~100 ms ≈ 55 min —— 那个 100 ms 是楼梯最低点的均价，拿它当恒定价会低估 2 倍；方法在 1310 位罣上与实测对上到 +8 %），
   夹具只能测解码，测不了 prefill 本身。§10 之后这一条**更重要了**：斜率被砍掉 1/3，prefill 的
   二次项就是长上下文唯一的墙。
3. 32 KiB 那一档越限 4 B 是**本机现象**，不进生产路径；`QTILE 8192` 只能显式指定，auto 到不了。

---

## 10. GQA 4× 冗余的消除（STAGE10，2026-11-05）：内核、位相同门、配对实测与归因

约束遵守情况与 §1–§9 相同：**K 仍是 FP16**（没压）、V 仍只有 q8_0 一档、权重反量化路径一行未动
（`align_dequant` 复测 `max abs 0.000e+00`，`runs/regression_v14.log`）、模型格式未改、未新增依赖。
已验证的三个内核（narrow / wide / tiled q2048）**一字节未改**，合并内核是**加法**，并且被两道位相同门
钉在它替换的内核上。

### 10.1 形状与代价（几何表，`实测` + `推导`）

一个 workgroup 服务一个 (层, kv 头)，管内 `HPK = n_head / n_head_kv = 4` 个 query 头。
K 的 gather 指令数与 K 的 half 解包数各砍 **4×**，每层常驻 workgroup 从 **16 → 4**。

| 每位置每层的 operand | 合并前 | 只合并 QK（**默认**） | 两半都合 |
|---|---|---|---|
| K（fp16，被 16 头各读一遍） | 64.0 KiB | **16.0 KiB** | 16.0 KiB |
| V（q8_0） | 34.0 KiB | 34.0 KiB | **8.5 KiB** |
| 合计 | 98.0 KiB | **64.0 KiB** | 24.5 KiB |

合并内核的 LDS = `HPK*4*QTILE + 8*HPK` B（`实测`，日志 `GQA merge:` 行）：q256 → 4128 B、
**q512 → 8224 B（默认）**、q1024 → 16416 B；`BONSAI_GQA_TILE` 只接受这三个值。

### 10.2 位相同门：A3（QK）与 A4（PV），差 1 bit 就拒绝生成

`BONSAI_QCHECK=1` 让合并内核与它替换的内核在**同一份 KV 字节**上各跑一遍，逐位比 SCORE / PROB /
打包后的 P16（以及 A4 的 PGATE / GOUT），覆盖首末两个注意力层（layer 0 与 layer 7）的全部 16 头 ×
cap 槽位。当前二进制的门（`实测`，`runs/regression_v14.log:88-94`）：

```
A3 layer 0 (cap 256, ref NARROW, HPK 4, merged q512, grid 16 -> 4): SCORE 0/4096 bit-diff, PROB 0, P16 0
A3 layer 7 (...): SCORE 0/4096 bit-diff -> bitwise identical
A4 layer 0 (cap 256, PV q8_0 frozen vs merged, grid 16 -> 4): PGATE 0/4096 bit-diff, GOUT 0
A4 layer 7 (...): PGATE 0/4096 bit-diff, GOUT 0
gate: PASS (bitmask 0 = A ok A2 ok A3 ok A4 ok B ok C ok D ok E ok)
```

1 310 个真实 live key 跨六个 tile 时合并 ≡ wide（`runs/gqa_real_c2048_merged_v1.log`），独立双精度 oracle
对 512 个点积的复算仍是 `0.000e+00`。**任何一位不同 ⇒ `rc 6`，不生成。** 位相同 = 性能实验；
位不同 = 第二个模型。

### 10.3 配对实测（batch H，`runs/gqa_pair_v1_summary.txt`；头条数字，非配对的 batch G 只当归因仪器）

同一二进制、每个配置紧挨自己的对照、同一热窗口内；cap 256 那一对在冷卡上最先跑，对照在最后一次跑，
所以漂移带是**测出来的**而不是假设的。

| 位置 | 对照 `GQA_OPT=0`（ms/token → tok/s） | **on1 = 只合并 QK（默认）** | on3 = 两半都合 | on1 延迟 Δ / 吞吐增益 |
|---|---|---|---|---|
| cap 256 | 73.4 → 13.63 | 72.7 → 13.75 | **70.9 → 14.10** | −1.0 % / +0.9 %（**漂移带内**） |
| 4 096 | 126.9 → 7.88 | **106.3 → 9.41** | 118.1 → 8.47 | −16.2 % / **+19.4 %** |
| 8 192 | 187.4 → 5.34 | **144.3 → 6.93** | 169.5 → 5.90 | −23.0 % / +29.8 % |
| 16 384 | 303.6 → 3.29 | **218.6 → 4.57** | 271.5 → 3.68 | −28.0 % / +38.9 % |
| 32 768 | 542.5 → 1.84 | **374.8 → 2.67** | 470.4 → 2.13 | **−31.0 % / +45.1 %** |
| 对照末次复跑 | 72.3 → 13.83 | — | — | **同一配置漂移 1.5 %** |

短上下文的诚实说法是「白送」，不是「变快」：73.4 / 72.7 / 72.3 三个数全在 1.5 % 漂移带里。默认配置
端到端 **14.16 tok/s / 70.6 ms/token**（`runs/regression_v14.log:95-96`）对 14.05–14.20 的冻结基线是
**0 % 回归**（预算 ≤ −5 %）。

### 10.4 归因与两条被实测推翻的话

拟合（`推导`，on1 四点最小二乘）：`ms/token = 67.17 + 0.00936 x positions`，最大残差 1.99 ms（0.9 %）。
对照 §9 的 tiled：`69.97 + 0.01408`。**斜率 −33.5 %，常数不动** —— §9.5 那句「tiling 动的是能力不是
斜率」与 §5.9 的「常数才是地板」两条结论同时被验证：合并确实只动斜率。

① **「PV 那一半是负的」说得太满。** 首轮（batch G，非配对）在 4 K 处测到 PV-only 184.2 vs 对照 128.7，
于是写下了那句话。配对复测把它改成条件句：在 on1 之上再加 PV 合并，**长上下文亏**（4 K 106.3 → 118.1 =
−11.1 %；32 K 374.8 → 470.4 = −25.5 %），**短上下文赚**（cap 256 72.7 → 70.9 = +2.5 %）。机制：K 是
gather（lane 间跨步 2 KiB，一条 64-lane 指令每 sector 只碰 32 B 里的 4 B，≈ 8× 浪费），合并消的是**事务**；
V 连续读、且 quad 内本来就被共享，合并只消 ALU，代价是常驻 workgroup 16 → 4 —— 短上下文时网格本来就小，
省下的 ALU 全是净赚；长上下文时丢掉的占用压过它。`BONSAI_GQA_PART` 的 bit 1 因此默认 0，依据是测量。

② **「字节才是瓶颈」这个隐含前提被 §9 和 §10 一起否掉了。** §9 证明 q8_0 砍掉 23.4 % 的 operand 只动
1.1 % 的斜率；§10 证明真正有效的是把 gather 事务数与解包次数从 4 份并成 1 份。所以本卡长上下文的第一
约束是 **gather 的形状 + 12 CU 上的占用**，不是带宽。

### 10.5 开关与复现

| 开关 | 语义 |
|---|---|
| `BONSAI_GQA_OPT` | `0` OFF（位精确对照路径，= §1–§9 量过的那批内核）/ `1` **AUTO（默认）**，由几何判定是否走合并 / `2` FORCE，每个 cap 都走合并内核（含 cap ≤ 256 的窄版地盘），同二进制 A/B 靠它；`>2` 拒绝 |
| `BONSAI_GQA_PART` | 位 0 = 合并 QK，位 1 = 合并 PV，默认 `1`。**诊断量而不是配置**：让一个测出来的收益被归因到一半，而不是同时对两半宣称收益 |
| `BONSAI_GQA_TILE` | 合并内核 wrapper = 它的 LDS，默认 **512** |
| `BONSAI_QKERNEL` / `BONSAI_QK_TILE` | §9 的两个旋钮，仍可强制 narrow/wide/tiled，让四条谱系在同一份输入上对比计时 |

```bat
runtime_vk\build_mingw.bat
runtime_vk\batch_h.sh v1            :: 配对阶梯 -> runs/gqa_pair_v1_summary.txt（权威）
runtime_vk\batch_g.sh v1            :: 首轮非配对 + PART 归因 -> runs/gqa_ladder_v1_summary.txt
set BONSAI_QCHECK=1 && runtime_vk\bonsai_vk.exe run <gguf> 32 48    :: A/A2/A3/A4 位相同门 + 端到端
```

### 10.6 本轮没做的

1. **半合并（2 头一组、8 workgroup）**：§10.4 的机制解释直接指出 16 → 4 这个幅度才是长上下文那侧的
   代价，所以 16 → 8 是唯一有实测方向、但还没试的杠杆。
2. 默认配置的整条长上下文阶梯只在 batch H 一个热窗口里配过对，没有另起会话复跑。
3. `BONSAI_GQA_OPT=2`（FORCE，让合并内核接管包括 cap ≤ 256 窄版地盘在内的每一行）会不会改变短上下文的
   token 序列：位相同门已经在 cap 256 比过合并 vs narrow（A3 `SCORE 0/4096 bit-diff`）、在 cap 2048 /
   **1310 个真实 live key** 比过合并 vs wide（A3 `0/32768 bit-diff`，跨六个 tile 的点积与四趟打包，
   `runs/gqa_real_c2048_merged_v1.log:119-129`，`gate: PASS`），答案都是不会——但这只在**当前 checkpoint
   的这套几何**（`hpk == 4`）下成立，`bonsai_forward.h:1956` 的合并分支本来就以此为前置条件。


