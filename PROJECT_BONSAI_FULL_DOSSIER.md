# Project Bonsai 终极全景档案

**一份可以从第一行复算到最后一行的项目底稿**
生成时间：2026-11-05 12:40（本机时钟）· 仓库 tip `d73cdec` · 引擎 `bonsai_vk.exe` build_v42
标注口径贯穿全文：**〔实测〕** = 某份 `runs/*.log` 里引擎或工具自己打印的数；**〔推导〕** = 由已打印常数算出、过程写在旁边；**〔未验证〕** = 本机做不了，直说做不了。

---

## 0. 写给审计者的三条前置更正（先把底稿自己的地基弄对）

这份档案要发到 GitHub 和 X 上，所以第一件事是把**三个会被逐行审计抓住的说法**改掉，包括我自己在上一轮指令里用过的两个。

| 流传的说法 | 实际 | 证据 |
|---|---|---|
| 「最终体积仅 419 KB」 | 发行的二进制是 **552 178 B = 539 KiB**。419 KB 是历史上四个互相矛盾的数之一；**416 KiB 是"去注释后的源码"**，不是产物体积 | `tools/source_size.py`：engine 源码 raw 696 KiB / de-commented 416 KiB / whitespace-stripped 279 KiB；`ls -l` = 552 178 B |
| 「Vulkan 1.3 手搓」 | 引擎的 instance 默认建在 **Vulkan 1.1**；`BONSAI_API12=1` 才请求 **1.2**（只为 `vkResetQueryPool`）。**没有任何 1.3 依赖，没有 SDK，没有校验层要求** | `bonsai_vk.cpp:208` `ai.apiVersion = getenv("BONSAI_API12") ? VK_API_VERSION_1_2 : VK_API_VERSION_1_1;` |
| 「结合截图里的内存分配日志」 | 本轮对话**没有收到附件截图**。显存分配证据改用引擎自己打印的 region 表与预算行，逐字引用 `runs/package_smoke_v2.log` | 见 §2.4 与 §2.3 的引用块 |

**为什么这三条值得单列一节**：这个项目对外唯一的资本就是"数字可以复算"。一个错误的体积、一个错误的 API 版本，就足以让整篇长文在 HN 上被拆掉。所以宁可先自己拆。

---

## 1. 项目序言与赛博朋克背景

### 1.1 机器：一块 2017 年的入门级移动独显

这不是一台"性能受限的开发机"，这是一台按 2026 年的标准应当被扔进回收站的机器。〔实测〕全部来自它。

| 项 | 值 | 备注 |
|---|---|---|
| GPU | **Radeon RX 550X 4 GB**，gfx803 / POLARIS_12 | 2017 年 GCN4 架构 |
| 矩阵核 / DP4A | **无 / 无** | 所有张量运算只能靠标量 ALU 与 LDS |
| 每 CU 共享内存（LDS） | 64 KiB | 4 KiB 静态共享分配实测建得起来，见 `KV_P0P1P2_MEASURED.md` §8 翻案 |
| CPU / 内存 | Ryzen 5 3600 / 16 GB | |
| 系统 / 驱动 | Win10 + **WDDM** + amdvlk64 2.0.279 | WDDM 是本项目最难的对手机制，见 §2.3 |
| 预算 / 权限 / C 盘依赖 | **0 元 / 0 次 UAC / 0** | 工具链全部装在 D 盘（MinGW-w64 + Vulkan-Headers + glslang） |
| 被加载的权重 | `Ternary-Bonsai-2-9B-PTQ1_0.gguf`，1 997 486 976 B | **从未被写过**，只读加载 |

### 1.2 软件极限：0 第三方依赖的纯 C++17 + 手写 SPIR-V

| 层 | 数量〔实测〕 | 说明 |
|---|---|---|
| 引擎产物 | `bonsai_vk.exe` **552 178 B = 539 KiB** | `-Wall` **0 warning** |
| 着色器 | **52 支 SPIR-V，307 432 B = 300 KiB**（52 `.comp` + 5 `.glslh`） | 手写 GLSL compute，无图框架、无 codegen |
| 链接依赖 | **0 个第三方库** — `-lvulkan-1 -lpdh -ldxgi` | 前一个是 OS 自带的 Vulkan loader，后两个是 Windows 自带的导入库 |
| 源码体积 | engine（host+shaders）去注释 **416 KiB**，去空白 **279 KiB**；含工具全树去注释 726 KiB | 唯一生产者：`tools/source_size.py` |
| 发行包 | 91 个文件 ≈ 6.5 MiB | 精确值不钉死，包根一条命令现算 |

**没有做的事同样重要**：没有 PyTorch、没有 ONNX Runtime、没有 VulkanMA（显存编排是自己算 offset 的，见 §2.4）、没有 tensor core intrinsic（硬件没有）、没有 DP4A（硬件没有）、没有厂商数学库。

### 1.3 头条数字：在这张卡上，比官方参考引擎快 3.21 倍

| 引擎 | 同一张卡 / 同一份权重 | tok/s | 出处 |
|---|---|---|---|
| **Project Bonsai（本仓库，纯手写）** | 48 token / 3389.5 ms，稳态 70.6 ms/token | **14.16** | `runs/regression_v14.log:95`〔实测〕，同一趟 `gate: PASS`（:94） |
| 官方 llama.cpp（上游） | **打不开这个 checkpoint** | — | 拒绝 `type 143`（PTQ1_0）；这是 STAGE5 的实测结论 |
| PrismML llama.cpp fork b10777（参考实现） | 33/33 层跑通 | **4.417** | `runs/`（STAGE5 实测） |
| 比值 | | **3.21×** | 14.16 / 4.417〔推导〕 |

而且这 **不是靠放弃保真换来的**。同一趟运行里，位相同门全绿：

```
gate: PASS (bitmask 0 = A ok A2 ok A3 ok A4 ok B ok C ok D ok E ok)
```

任何一位不同 ⇒ 引擎返回 `rc 6` 并**拒绝生成**。这不是测试报告的语气，是代码里的控制流。

### 1.4 我们离这张卡自己的物理上限还有 16%——而且那 16% 不是偷懒

一个能被逐行审计的长文必须敢说"还剩多少不是我的功劳"。两条独立的上界〔推导，常数全为实测〕：

| 角度 | 推导 | 结论 |
|---|---|---|
| **权重带宽** | 每个 token 要流过的**权重区** = codes 2 516 189 184 + f32blob 29 411 328 = **2 545 600 512 B = 2545.6 MB**（§2.4 的 arena 里另有 42.0 MB 是 scratch/输出，不计入流量；前向路径没有专家路由分支，所以是"全部"）；实测 token 70.6 ms ⇒ 需要 **36.06 GB/s**；本卡实测峰值 **43.4–44.2 GB/s** | 已用到峰值的 **82–83 %**；纯带宽上限 ≈ 2545.6 / 43.4 = 58.7 ms = **17.0 tok/s**，即 14.16 是它的 **83 %** |
| **算力（GEMV）** | 本卡加权 176.9 Gop/s〔实测〕；每 token 7.929 GMAC〔实测〕 ⇒ 下限 44.8 ms/token | GEMV 理论地板占实际 70.6 ms 的 **63 %** |

也就是说：**14.16 tok/s 已经是"这块 4 GB 矿卡能做到的事情的八成分"**，剩下的两成里大部分是注意力与非 GEMV 算子的固定开销（每 token 1 次 submit，`dispatches 35760, barriers 35736`〔实测〕），不是"还没优化"。

**这就是赛博朋克的部分**：不是我们用魔法让 2017 年的硬件超过了物理，而是我们把一块被判定为"电子垃圾"的硬件逼到了它自己带宽曲线上的 **83 %**，并且在这个过程中每一位输出都可以被逐位比对。

---

## 2. 四大底层极限手术

每一台"能跑的引擎"都会说自己快。这四节讲的是**为了在这张卡上跑起来，必须动刀的四个地方**，以及每一刀换来多少、代价是什么。

### 2.1 手术一：把 GQA 的 4× 冗余从注意力内核里切掉

**问题形状**〔实测自 GGUF header〕：每个注意力层 **16 个 query 头 / 4 个 KV 头** ⇒ 每个 KV 头服务 `HPK = 4` 个 query 头。原实现按 query 头开 workgroup，于是**同一份 K 在显存里被读了 4 遍**。在 head_dim 256、上下文 32 K 的位置上，这就是纯粹的浪费。

**做法**：新着色器 `attn_qk_softmax_gqa_h4_q{256,512,1024}.comp` —— 一个 workgroup 负责一个 KV 头，**一次 K gather 服务 4 个 query 头**，grid 从 **16 → 4 workgroups/层**。LDS 8224 B/workgroup（`HPK*4*QTILE + 8*HPK`，QTILE = 512）。

**收益（配对阶梯，同一支二进制、同一次清场、只有开关不同）**〔实测，`runs/gqa_pair_v1_summary.txt`〕：

| 上下文 | GQA off | GQA on（只合 QK） | 增益 |
|---|---|---|---|
| 4 K | 7.88 | **9.41** | **+19.4 %** |
| 8 K | 5.34 | **6.93** | **+29.8 %** |
| 16 K | 3.29 | **4.57** | **+38.9 %** |
| 32 K | 1.84 | **2.67** | **+45.1 %** |
| cap 256（短上下文） | 13.63 | 13.75 → 合 PV 后 14.10 | 见下 |

**买到的是冗余，不是调度**〔推导自实测拟合，ms/token = 常数 + 斜率 × 位置〕：

```
wide   （按 query 头，未 tile）  71.12 + 0.01443 x
tiled  （已消共享内存上限）      69.97 + 0.01408 x     斜率 −2.5 %
merged （GQA 合并 QK）           67.17 + 0.00936 x     斜率 −33.5 %，常数几乎不动
```

常数项基本不动、斜率砍掉三分之一 ⇒ 减少的确实是**每位置的工作量**，而不是每 token 的启动开销。这正是 GQA 合并应有的签名；如果常数也大幅下降，反而要怀疑是不是量错了。

**代价与另一半真话**：

- **保真为零代价**：门 **A3** 直接证明合并内核与参考内核**逐位相同**〔实测，`runs/regression_v14.log`〕：
  `A3 layer 0 (cap 256, ref NARROW, HPK 4, merged q512, grid 16 -> 4): SCORE 0/4096 bit-diff, PROB 0, P16 0 -> bitwise identical`（层 7 同）。覆盖 8192 个 SCORE/PROB 元素 + 4096 个 P16 字，含越过 512 字边界的打包对。
- **PV 合并不是免费午餐，它是双相的**〔实测〕：`BONSAI_GQA_PART` bit1 打开后，cap 256 **快 2.5 %**（72.7 → 70.9 ms/token），但 32 K **慢 25.5 %**（374.8 → 470.4 ms）。所以默认是 `gqa_part = 1u` —— **只合 QK**。写"全部打开更快"是错的，写进文档就是骗人。
- **短上下文 0 回归**〔实测〕：默认配置仍在 **14.05–14.27 tok/s** 历史带内（14.16），预算 ≤ −5 % 没碰到。

### 2.2 手术二：只压 V、绝不压 K 的 q8_0 KV Cache —— 把 4 GB 的算术做穿

这是全项目最需要"讲清为什么"的一处，因为**直觉会说反**（大家都想压 K，因为 K 先被读）。

**第一步：几何。**〔实测，引擎自己从内核布局推导并打印，不是手填的常数〕

```
K  fp16             512 words/slot/layer =   2048 B   (never compressed)
V q8_0 data       256 words/slot/layer =   1024 B   (8 blocks of 32 per head row)
V q8_0 scales      16 words/slot/layer =     64 B   (1 fp16 per 32-element block)
per layer-slot     3136 B -> x8 attention layers = 25088 B/token = 24.50 KiB/token
```

模型只有 **8 层**产生 KV Cache，另外 **24 层是循环状态（GDN）**、与上下文长度无关（52 690 944 B = 50.25 MiB 的固定量，实测 cap 256→2048 时字节数不变）。4 KV heads × head_dim 256 = 1024 个元素/层/slot：fp16 K = 2048 B，V 走 q8_0（32 元素 + 1 个 fp16 scale = 每 32 元素 33 B）= 1024 + 64 = 1088 B。

**第二步：为什么 K 不能压。** K 进的是 QK 点积——它的误差**乘在每一个 score 上**，然后被 softmax 放大成整行概率的形状变化；而 V 进的是 softmax **之后**的凸组合，它的误差被概率权重在位置上平均掉。这不是修辞，是可测的〔实测，`runs/kv_compare_v4.log`〕：

| 指标 | K 侧 | V 侧（q8_0） |
|---|---|---|
| 逐位比较 | **37 888 个元素，0 个不同 —— BITWISE IDENTICAL** | —— |
| 误差幅度 | 0（未压） | position 0（纯量化）：max\|err\| 0.0550537 对 max\|ref\| 13.9609；rms err / rms signal **8.802e-03 = 41.11 dB** |
| 端到端 | 层 0–31 采样 token 序列与 `none` 模式**一致** | 同左 |

作为对照，注意力整体对齐（`runs/align_attn_v4.log`，seq 64）card vs mirror score = **3.567e-03**，是**官方引擎自己的 fp16 缓存代价的 0.262 倍**〔实测〕。

**第三步：为什么压 V 对 4 GB 卡是"必须"。** 把预算算穿〔推导，常数全为实测：权重 2467.7 MiB、cap 无关活动区 52.0 MiB、DXGI LOCAL 预算 3464.6 MiB、原始 device-local heap 3840.0 MiB；每 slot 用引擎自己的 `per_slot`，q8_0 = 24.66 KiB、fp16-V = 32.16 KiB，差额是 0.16 KiB 的 fp32 侧〕：

```
cap_max = (预算 − 权重 2467.7 − 52.0 MiB) / per_slot
        = 944.9 MiB × 1024 / per_slot
```

| 判据 | V = q8_0 | V = fp16（改前） | 差 |
|---|---|---|---|
| 按 DXGI 预算 3464.6 MiB（**安全线**） | ≈ **39.2 K** token（39 237） | ≈ 30.1 K token（30 086） | **+9 151 个位置 = +30.4 %** |
| 按原始 heap 3840.0 MiB（**极限**） | ≈ **54.8 K**（54 825） | ≈ 42.0 K（42 039） | +12.8 K |

引擎自己打印的极限读数与手算对上了：`headroom 1314.1 MiB (= 54558 more slots ...)` ⇒ 256 + 54 558 = 54 814 vs 手算 54 825，差 **0.04 %**。
**这张表本身被更正过一次**：原先四格用 `MiB→KiB ×1000` 算，系统性偏低 2.5 %，重算后与两处独立回算交叉验证（40 960 − 39 237 = 1 723 slot × 24.66 KiB = 41.5 MiB；49 152 − 39 237 = 9 915 × 24.66 = 238.8 MiB）。

**第四步：诚实的另一半——压 V 买的是容量，不是速度。** 三条实测否掉了"带宽论"：

- q8_0 把每位置的操作数量砍 **23.4 %**（K 64.0 + V 34.0 = 98.0 KiB → V 段 34.0 → 8.5 KiB），但 4 K–32 K 的**斜率只动了 1.1 %**；反推 KV 有效带宽只有 **6.95 GB/s**，是权重路径 43 GB/s 的 **1/6** ⇒ 长上下文**不是**被 KV 带宽支配的。
- 短上下文长度回归：`none` **14.27** vs `q8_0` **14.26** tok/s = 0.01 tok/s = 噪声〔实测〕。
- ≤32 K 全程 PDH `Shared Usage` 增长 **+0.00**〔实测〕⇒ 也不是被 spill 支配的。

**实现纪律**：写入即量化，解包在 `attn_pv_q8v.comp` 的 V 读/累加循环**内部完成**（纯寄存器，`shared` 占用 0 B）。**没有**"先解包写回全局再读"的路径，**没有**新增每 token 的全局 scratch，**没有** 4-bit V，**没有**碰 K。开关 `BONSAI_KV_V_QUANT = none | q8_0`，默认 q8_0，`none` 永久保留为回归对照。

### 2.3 手术三：WDDM 显存溢出拦截——硬挡，而不是等系统悄悄把页搬走

Windows 上 4 GB 显存的真正对手不是 OOM，是 **WDDM 静默 spill**：超了不报错，把页迁到系统内存，于是基准测试从"GPU 速度"变成"PCIe 往返速度"，而你的曲线看上去还是"能跑"。

**判据选得对不对，是这个模块的全部价值。**〔实测 + 用户明令禁止〕

| 计数器 | 用途 | 结论 |
|---|---|---|
| `\GPU Process Memory(*)\Shared Usage`（本 pid 全部实例求和） | **唯一的 spill 判据** | ✅ 这才是 WDDM 把页扔进系统内存时会长的那个数 |
| `\GPU Process Memory\Private/Dedicated Usage` | 辅助读数 | ❌ **禁止当 spill 判据**（Dedicated 会随分配涨而不代表外迁；用它 = 自欺） |
| `IDXGIAdapter3::QueryVideoMemoryInfo`（LOCAL / NON_LOCAL 预算） | 辅助 | 引擎自己标注 `AUXILIARY: a reservation budget, not an eviction signal` |

**两个坑是拿命换的**〔实测，`tools/pdh_probe.cpp`〕：计数器路径**必须以反斜杠开头且必须带实例部分**——`"GPU Process Memory(*)\Shared Usage"`（少头一个 `\`）返回 `0xC0000BC0 (BAD_COUNTERNAME)`。第二个坑更阴险：**`VK_FENCE_CREATE_SIGNALED_BIT` 创建的 fence 在等待时已经是 signaled 的**，`vkWaitForFences` 立刻返回、与 arm 它的那次 submit 无关。这个错误先前制造了两个假结论——物理上不可能的 **56.67 GB/s**，以及"hop 被同步开销支配"（真实是 DMA 支配）。本仓库现在的纪律是**每支 fence 一律 `flags = 0`**，完成状态用**门铃 word + 同一条 CB 里的 4 B fill** 发布。

**真凶不是分配，是命令缓冲**〔实测〕：每 token 新建一条命令缓冲的驱动代价是 **0.3737 MiB**（空 CB 0.0650 MiB），而 `vkFreeCommandBuffers` 归还 **0.00** ⇒ 曲线每 token 稳定爬升。`BONSAI_CB_REUSE=1` 之后 Shared Usage 全程走平。阈值语义也因此改判：判的是**相对首个 decode 采样的增长量**（soft > 32 MiB、hard > 64 MiB、增长连续 ≥ 3 次），不是绝对值；`BONSAI_SPILL_ABS=1` 保留绝对读数。

**现在的实测长这样**〔实测，`runs/package_smoke_v2.log:176,:196`，发行包开箱那一趟〕：

```
[spill] tok 16   shared  40.93 MiB (growth  +0.00 over the reference  40.93, dedicated 2539.23 MiB, growth run 0)
[spill] tok 24   shared  40.93 MiB (growth  +0.00 over the reference  40.93, dedicated 2539.23 MiB, growth run 0)
samples 2 of 2 tried, peak absolute 40.93 MiB, peak growth +0.00 MiB, soft 0, hard 0, monotonic-run(>=3 increases) 0
```

**顺带把"这张卡到底吃了多少"钉死，并纠正一处误引。**〔实测〕`dedicated 2539.23 MiB` = 权重 **2467.7** + 活动区 **58.2** + **13.3 MiB 驱动收走但预算表没建模的开销**。这 13.3 MiB 是真实缺口，写出来而不是抹平。
另外：**过去长期被引用的 `1985.3 MiB 峰值 / 16.05 s 冷加载` 是官方 fork 的账，不是本引擎的**。本引擎实测：权重装载**冷 14.05–14.44 s / 热 1051.9–1201.0 ms**；活动区 **58.2 MiB @cap256（实测）/ 101.339 @2048 / 841.261 MiB @32K**。

### 2.4 手术四：零误差反量化 + 0 Padding 显存编排

**"零误差"是三份独立比对撑起来的，不是形容词**〔实测〕：

| 门 | 覆盖 | 结果 |
|---|---|---|
| PTQ1_0 三方反量化 | 1 000 个真实块 | `max\|err\| = 0.00000` |
| PTQ1_0 合成边缘块 | 491 520 个权重（含 0 / 次正规 / 负 scale） | **逐位 0 差** |
| BPE 分词器 | 39 条探针（32 probes + 7 个 kr 集） | **逐 ID 与官方 1:1**（决定性事实：初始切分必须字节起手） |
| fp16↔fp32 | 全域 + 两条着色器表达式 | 逐位 |

（一处真因纠正值得留着：PTQ1_0 块的 `d` 字段是 **`float16_t` 而不是 `uint16_t`**——把它当整数读是这个"0 误差"曾经达不到的原因。）

**0 Padding：不是运气，是布局的定义。**〔实测，引擎逐字打印〕

```
arena (single DEVICE_LOCAL allocation; sizes are the plan's, offsets are the driver's):
  codes           2236612608 B  align 4    -> offset 0
  scales           279576576 B  align 4    -> offset 2236612608
  f32blob           29411328 B  align 4    -> offset 2516189184
  ptq_scratch       33554432 B  align 4    -> offset 2545600512
  x                    65536 B  align 4    -> offset 2579154944
  xsum                  4096 B  align 4    -> offset 2579220480
  out                8388608 B  align 4    -> offset 2579224576
  arena committed 2.410 GiB, alignment padding 0 B
```

七段的 offset 精确等于尺寸的前缀和 ⇒ **对齐插入字节 = 0**。之所以能这样，是因为整块 arena 以 fp32 word 为单位寻址、每段尺寸都是 4 的整数倍，而本驱动 `minStorageBufferOffsetAlignment = 4 B`〔实测〕。总量也对得上：末段尾 `2 587 613 184 B` = 2.4099 GiB = **2467.75 MiB**，与 §2.3 预算行里引擎自己印的"weights 2467.7 MiB"是同一笔钱（引擎向下截断到一位小数）——**这张表的两个数字是同一笔钱的两本账**。

活动区同理，由几何而非手填：

```
KV geometry (derived from the kernel layout, not typed)
arena invariants | KV span 1605632 == cap 256 x 8 x 784 OK; ST / CH are 24 GDN layers x per-layer state, cap-free OK
mixed-stack invariants | plan charges 1605632 KV words == arena 1605632 (+0 B/term of offset alignment);
                         24 recurrent layers charged 0 B of KV
activation arena: 15250752 floats = 58.177 MiB (state 48.0 MiB + KV 6.1 MiB + logits 0.95 MiB)
```

**"节俭"的证据不是形容词，是这些恒等式在每次运行里被重新检查**：`+0 B/term`、`cap-free OK`、`charged 0 B of KV`（这是不变量，见 §3.1）。

---

## 3. 跨硬件与未来布局

这一节把"已经证明能移植的东西"和"还没在这台机器上演练的东西"分开写。

### 3.1 27B 架构闸门：先把形状与计费锁死，再谈跑不跑得动

〔实测，`align_27b.exe` → `ALL CHECKS PASSED`〕

| 闸门 | 断言 | 读数 |
|---|---|---|
| PTQ1_0 解码 | 27B 形状下逐位复现反量化 | PASS |
| **64-block 混合路由** | **16 层全注意力 / 48 层循环状态** | PASS |
| **SSM 层 KV 计费 = 0** | **不变量（严禁给循环状态层分配 KV Cache）** | PASS：`48 SSM layers own none` / `SSM KV charge is 0 by construction over all 48 recurrent blocks` |
| 每 token 定价 | 16 计费层 = **49.000 KiB/token**；被禁止的"每层都挂 KV"定价 = **196.000 KiB/token** | **4.0×**，cap 32 768 时 arena **1568.00 MiB** |
| 旋转路由 | 由 `prism.hadamard.version` 决定是否转、`block_size` 在 `hat4096.comp`（9B 位冻结路径）与 `hadamard_fwht.comp` 之间路由 | 通用支与主机 Sylvester FWHT **逐位相同** |

**边界要说满**〔未验证〕：**本机跑不动真实的 5.9 GB checkpoint**（4 GB 显存 + 16 GB 内存）。被证明的是 **resolver、计费不变量、内核路由**，用的是从 9B 真 header 搭出的合成形状。这句话如果省掉，这份档案就是吹牛。

### 3.2 Pinned Memory 环形队列、iGPU 零拷贝与多卡流水线

**已测量的部分**〔实测，`bonsai_fleet.exe`，`runs/fleet_ring_v4.log`〕：

| 项 | 读数 |
|---|---|
| 主机侧 selftest | **55/55 PASS，0 字节不符** |
| 单腿 8 MiB fenced D2H | **1204.6 µs = 6.96 GB/s = 本机 7.15 GB/s 上限的 97 %** |
| 8 腿一批 | **1176.4 µs/腿 = 7.13 GB/s**（ batching 有效） |
| 深度 4 的环 | 背压 **2 次**（说明深度 4 在 6 hop 尺度上确实会撞） |
| 每 hop 墙钟 | 3171.3 µs |

**测不到的部分同样写清楚**〔未验证〕：`probe` 只报 **1 个 COMPUTE-capable device** ⇒ **卡对卡那一跳在本机不可演练**；上面测的是 **D2H → host → H2D** 这条真实存在的腿。iGPU 零拷贝按 `deviceType` 门控，**256 MiB BAR 是否真的可用是未验证前提**。`split 32 2`（把 32 层切成 2 卡的区间）在只有 1 卡时**直接拒绝**——这是刻意的：不给没条件跑的代码制造假绿。

**为什么这套东西对"580 矿卡阵列"这个社区有吸引力**〔推导，架构事实〕：RX 550X 与 RX 580 **同代**——GCN4 / Polaris（POLARIS_12 vs POLARIS_20），**同样每 CU 64 KiB LDS、同样没有矩阵核、同样没有 DP4A**。本引擎的每一个关键决策都不依赖厂商扩展：手写标量 SPIR-V、word 对齐的自管理 arena（0 padding）、按 `deviceType` 门控的多卡腿、每 token 1 次 submit 的命令缓冲复用。⇒ **移植的障碍是验证，不是架构**。〔未验证：本机没有 RX 580，这句话是推理，不是测量。〕
另一条已经落地的可移植性证据：**驱动栈无关的入口已经备好**——实例默认 1.1、`BONSAI_API12=1` 才升 1.2、`VK_VALIDATION=1` 可挂校验层〔实测代码路径〕；在第二驱动栈（RADV）上跑全套 `align_*` 五件套仍然**未做**，见 §5。

### 3.3 长上下文与 prefill 的真实定价：它是平方级的

如果只宣传 39.2 K 的安全上限而不说 prefill 的价格，那是误导。〔推导自实测拟合 `67.17 + 0.00936 × 位置`，方法校验见下〕

| 上下文 | 该位置的 decode 代价 | **prefill 整段（平方级积分）** |
|---|---|---|
| 4 K | ≈106 ms/token | **≈5.9 min** |
| 32 K | ≈375 ms/token | **≈120 min** |

旧结论"32 K ≈ 55 min"低估了 2 倍，因为**把阶梯的最低点当成了恒定价**；正确的积分是第 t 个位置要扫 t 个 key ⇒ N 个 token 的均价落在 N/2。方法本身有校验：同一式对 1 310 个位置预测 **73.3 ms/位**，实测 **79.2 ms/位**（+8 %，偏保守）〔实测，1310 token / 103 766.4 ms = 12.62 tok/s〕。
**结论**：这张卡适合"长上下文能存、能读、但一次性灌进去很贵"的用法；`KV block` 的 evict/refuse 钩子（`KvGuard::decide()`，64 位置一块）就是为这个真实约束留的兜底接口。

---

## 4. 终极宣发弹药（中英双语，且每句都能被复算）

### 4.1 English — X thread / Reddit `[H]` post

**X（7 条一串，可直发）**

```
1/ I wrote a ternary-LLM inference engine from scratch for a 2017 e-waste GPU.
   Radeon RX 550X 4GB. No tensor cores. No DP4A. No vendor math lib. No CUDA.
   Result: 14.16 tok/s on a 9B ternary model. 3.21x the official reference
   engine on the same card and the same weights.

2/ The reference engine can't even open the checkpoint. Upstream llama.cpp
   rejects PTQ1_0 (type 143). So this isn't "faster than llama.cpp",
   it's "runs a model class the reference runtime refuses to load".

3/ The whole engine is one 539 KiB executable + 52 hand-written SPIR-V
   compute shaders. Zero third-party dependencies. The only link flags are
   -lvulkan-1 -lpdh -ldxgi: the OS loader, and two Windows import libs.

4/ 3.21x was bought with surgery, not luck:
   - GQA: one K gather now serves 4 query heads. Grid 16 -> 4 per layer.
     32K context: 1.84 -> 2.67 tok/s (+45%). And bit-identical to the
     reference kernel. Not "close". Zero differing bits.

5/ KV cache: I compress V to q8_0 and NEVER touch K.
   K feeds the QK dot product (error multiplies into every score, then
   softmax amplifies it). V feeds a convex combination after softmax
   (probability weights average the error out). 24.50 KiB/token instead of
   32.00 => context ceiling 30.1K -> 39.2K positions on a 4GB card.

6/ Windows WDDM doesn't OOM, it silently spills to system RAM and your
   benchmark becomes a PCIe latency test that still looks like it's working.
   So the spill judge is PDH \GPU Process Memory\Shared Usage growth over
   the first decode sample. The culprit was a fresh command buffer per token
   (0.3737 MiB, and vkFreeCommandBuffers returns 0.00). Reuse it: flat.

7/ Honesty, because this will be audited: 14.16 tok/s is already ~83% of
   this card's measured weight-bandwidth ceiling (17.0 tok/s at 43.4 GB/s,
   from 2545.6 MB of weight bytes streamed per token).
   Attention is NOT bit-exact vs the reference: 3.57e-3, which is 0.26x the
   cost of the reference engine's own fp16 cache. Prefill is quadratic:
   32K takes ~120 min. Every number links to a log line.
```

**Reddit 标题 + 正文首段**

```
[H] Pure C++17 + hand-written Vulkan compute: 14.16 tok/s on a 4GB 2017
Radeon (no tensor cores, no DP4A), 3.21x the reference llama.cpp fork on the
same card and the same weights — and upstream llama.cpp refuses to open the
checkpoint at all.

Body: 539 KiB binary, 52 hand-written SPIR-V modules, 0 third-party deps.
Everything ships behind a bitwise gate (A/A2/A3/A4/B/C/D/E): any differing
bit => the engine returns rc 6 and refuses to generate. PTQ1_0 dequant is
bitwise identical to the reference on 1000 real blocks plus 491520 synthetic
edge weights; the BPE tokenizer matches official per-ID on 39/39 probes.
Attention is the honest exception: 3.57e-3 max score error, 0.26x the
reference's own fp16 cache cost. All numbers, gates, and failure notes are
in BENCHMARK.md / KV_P0P1P2_MEASURED.md / runs/*.log.
```

### 4.2 中文 — B站 / 知乎

**标题**：《一块 2017 年被判定为电子垃圾的 4GB 独显，跑 9B 三值大模型：14.16 token/秒》

**正文骨架（每句后面都有日志）**

```
先说清楚这块卡有多老：Radeon RX 550X 4GB，2017 年的 GCN4，没有矩阵核，
没有 DP4A，没有 CUDA，驱动是 AMD 的 Vulkan 栈。CPU 是 R5 3600，内存 16GB。
一分钱预算，没点过一次"以管理员身份运行"，C 盘一个文件都没落。

再说结果：端到端 14.16 token/秒。同一张卡、同一份权重上，官方参考引擎
PrismML llama.cpp fork 是 4.417 token/秒——3.21 倍。
而 upstream llama.cpp 更尴尬：它连这个模型都打不开，直接拒绝 type 143。
所以这不是"我比 llama.cpp 快"，这是"这类模型别人加载不了，我们能加载"。

整个引擎是：一个 539 KB 的 exe + 52 支手写 SPIR-V compute 着色器，
第三方依赖 0 个。链接参数只有 -lvulkan-1 -lpdh -ldxgi——前一个是系统自带的
Vulkan 加载器，后两个是 Windows 自带的导入库。没有 PyTorch，没有 ONNX，
没有 VulkanMemoryAllocator（显存 offset 是自己按前缀和排的，对齐插入 0 字节）。

四刀，每一刀都有数字：
1) GQA 的 4 倍冗余：一个 K 读取服务 4 个 query 头，每层 workgroup 16 → 4。
   32K 上下文 1.84 → 2.67 token/秒（+45%），而且和参考内核【逐位相同】。
2) 只压 KV 的 V、绝不碰 K：V 用 q8_0（32 元素 + 1 个 fp16 scale），
   每 token 从 32.00 KiB 降到 24.50 KiB，4GB 卡的安全上下文上限
   从 30.1K 抬到 39.2K 个位置（+30.4%）。K 为什么不能压？因为 K 进点积，
   误差乘在每一个 score 上再被 softmax 放大；V 在 softmax 之后，
   误差被概率权重平均掉。K 侧 37888 个元素逐位不变，就是这个选择的验收。
3) 硬挡 Windows 的显存静默溢出：WDDM 超存不报错，把页迁到内存里，
   你的跑分就变成 PCIe 往返，但曲线看着还是"能跑"。判据用 PDH 的
   Shared Usage 相对首个 decode 采样的增长量；真凶是每 token 新建一条
   命令缓冲（0.3737 MiB，而 vkFreeCommandBuffers 归还 0.00）。复用之后全平。
4) 零误差反量化：1000 个真实块 max|err| = 0.00000，491520 个合成边缘权重
   逐位 0 差，分词器 39/39 逐 ID 和官方一致。

也说三件不好听的：注意力不是逐位对齐的（最大误差 3.57e-3，但这只有官方
自己 fp16 缓存代价的 0.26 倍）；14.16 已经到这张卡带宽上限（≈17.0 tok/s）的 83%，
剩下的不是"还没优化"；prefill 是平方级的，32K 要 ≈120 分钟。
另外，那台万元机我没借到，所以"暴打"这个词我不写——我只写能测的：
官方运行时打不开的模型，我这台 2017 年的机器打开了，还跑到 14 token/秒。

图吧精神的定义从来不是"花钱买最强的"，是"把手上这块板子的最后一档性能
榨出来，并且能证明榨的是哪一档"。
```

### 4.3 口径红线（外发前必须过的四条，防止被拆台）

| 不许说 | 为什么 | 可以这么说 |
|---|---|---|
| 「体积 419 KB」 | 产物实测 552 178 B = **539 KiB**；416 KiB 是**去注释源码** | "539 KiB 的可执行文件 + 300 KiB 的手写 SPIR-V" |
| 「Vulkan 1.3 手搓」 | 实例默认 **1.1**，`BONSAI_API12=1` 才 1.2，没有 1.3 依赖 | "只要 Vulkan 1.1（1.2 可选），不要 SDK" |
| 「完全逐位精确 / bit-exact 无保留」 | 只有反量化、分词、内核等价门是逐位；注意力 score 有 3.57e-3 | "bit-exact where the gate says so; attention error is 0.26× the reference's own fp16 cost" |
| 「暴打万元神机 / 比 4090 快」 | **本机没有测量过任何万元级机器或 4090**，也没有 CPU 对照（STAGE4 那条 9.2 % 是 fork 的路径，不是本引擎） | "官方运行时加载不了的模型，2017 年 4GB 卡跑到了 14 tok/s" |

---

## 5. 未闭合清单（这份档案不自称完工）

| # | 缺口 | 状态 |
|---|---|---|
| 1 | **采样器与官方对齐**（temp/top_k/top_p 与官方后处理逐条比） | 头号口子，未做 |
| 2 | chat template 之后逐 ID 与官方 `llama-tokenize -chat-template` 比对 | 未做 |
| 3 | cap 1024 / 2048 两档的显存实测表 | 只有 cap 256 与 32K 有实测 |
| 4 | 第二驱动栈（RADV）上的 `align_*` 五件套 | 未做 |
| 5 | 卡对卡那一跳 | 本机只有 1 张 COMPUTE 卡，不可演练 |
| 6 | 真实 27B checkpoint 上卡 | 本机 4 GB + 16 GB 装不下 5.9 GB |
| 7 | P0-1 submit 侧 −12.5 % | 未做 |
| 8 | GQA 半合并 HPK=2、>32 K 的门覆盖 | 未做 |

（本表 **8 行**，`PROJECT_BONSAI_FINAL_REPORT.md` 记的是 **9 条**，两条对不上的地方都在下面，不是两份文档打架：那份的**第 7 条**（位相同门在 >65 536 行之外的覆盖）被我并进本表第 8 行；那份的**第 9 条**（`BENCHMARK.md` 行号引用没有自动校验工具）属于工具链缺口、没进这张引擎表。反过来，本表第 8 行的「GQA 半合并 HPK=2」是 `ROADMAP.md` §4 的候选杠杆，不在那份的 9 条里。**同一批缺口，两种切法，逐条可对上。**）

---

## 6. 怎么自己复算（不要求你信任何一句话）

```bat
cd /d runtime_vk
build_mingw.bat                      :: 52 支 SPIR-V + 引擎 + 全部 align_* 工具，0 warning
set BONSAI_QCHECK=1
bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 48
::                                              -> gate: PASS + "48 generated tokens in 3389.5 ms -> 14.16 tok/s"

align_dequant.exe <gguf> 1000        :: max|err| = 0.00000
align_tokenizer.exe <gguf> ..\runs\tok_probes.txt ..\runs\tok_ref_32.jsonl   :: 39/39 逐 ID
align_attn.exe 64 256                :: score 最大误差 3.57e-03
align_27b.exe <gguf>                 :: ALL CHECKS PASSED（16/48 路由、SSM 计费 = 0）
bonsai_fleet.exe selftest            :: 55/55
cd .. && python tools\source_size.py :: 每一个字节数的唯一生产者
```

**发行包验货**（包根目录，`shaders\` 与 exe 同级；着色器按**当前目录**加载，所以必须先 `cd`）：

```sh
sha256sum -c MANIFEST.sha256         # 本轮装配后全 OK
python -c "import glob,os;fs=[p for p in glob.glob('**/*',recursive=True) if os.path.isfile(p)];print(len(fs),sum(os.path.getsize(p) for p in fs))"
```

**硬规矩**（全是踩坑换来的，已写进代码）：同一时刻只允许一个 GPU 进程；任何计时之前先 `tools/clear_field.ps1`；结果一律落 `runs/<名字>_v<N>.log`，**旧日志永不覆盖**；不能用批处理脚本复跑的基准测试等于没发生过；任何写模型文件的操作都要先问所有者（权重在本项目里**从未被写过**）。

---

> **We didn't wait for better hardware. We wrote the kernel.**
> **电子垃圾的尽头不是回收站，是还没被手写出来的那一支 kernel。**
