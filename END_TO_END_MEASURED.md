# 终局实测：清场 → 合入高优优化 → PTQ1_0 端到端点火

日期 2026-11-01。机器：Radeon RX 550X 4 GB（gfx803 / Polaris 12 / wave64）+ Ryzen 5 3600 + 16 GB，
Win10，Vulkan（amdvlk64 2.0.279）。模型 `models/Ternary-Bonsai-2-9B-PTQ1_0.gguf`
= 1,997,486,976 B，GGUF v3，arch `qwen35`，8.95 B 参数，`general.file_type = 143`
（PTQ1_0，1.75 bpw 三值，group 128）。模型所在盘 `WDC WD10EZEX`，Get-PhysicalDisk 判定 **HDD**。

本轮只报实测。每张表下面写了来源文件。由两个实测相除得到的数字一律标"实测派生"，并同时给出
被除数和除数，方便逐行审计。

---

## 0. 结果先说（五个数，全部实测）

| 问的问题 | 实测答案 |
|---|---|
| 端到端 decode tok/s | **4.417 t/s**（50 token 真跑，226.377 ms/token，引擎自己的计时器）；bench tg128 = 4.05 t/s |
| 真实加载时间（机械盘→显存） | **16.05 s** 冷（磁盘实测读 2196 MB，峰值 175.7 MB/s）；页缓存命中 **3.68 s** |
| 真实 prefill | **pp512 = 14.29 t/s**（预热后）；50 token 那次 17-token 批次的 6.09 t/s 是一次性开销污染，不能用 |
| 显存峰值 | WDDM 实测 **1985.3 MiB**；引擎账 **2024 MiB**（1746 模型 + 178 上下文 + 100 compute），4096 里剩 1463 |
| 生成的文本 | 真文本，见 §3.6；50 与 400 token 都停在 thinking 草稿阶段，语义正确、中英混排正常 |

三句话结论：**模型真的在这张 4 GB 的 A 卡上跑起来了并且没 OOM**；**现成引擎只吃到 kernel 潜力
的约 1/4（226 ms vs 59 ms）**，因为驱动不给 fp16 也不给 int dot，三值只能 f32 反量化；
**4 GB 墙的真实位置是 -c 49152 能过、53248 撞**，而且先撞的是 prefill 的 compute buffer 不是 KV。

---

## 1. 第一步：极限清场（结论：这台机器上根本没有可杀的目标）

按指令写了 `tools/clear_field.ps1`。白名单硬编码：`Qoder CN`（当前终端宿主进程名）、`node.exe`、
`python.exe`、`powershell.exe`、`cmd.exe`、`conhost`、`msedgewebview2`、Radeon 三件套
（RadeonSoftware / AMDRSServ / AMDRSSrcExt）、Defender/火绒守护进程、以及全部系统进程。
狙杀名单：chrome / msedge / javaw / java / steam / Discord 等。

| 项 | 实测值 | 来源 |
|---|---|---|
| Available MBytes 清场前 | **7989 MB** | `clear_field.ps1 -Kill` |
| Available MBytes 清场后 | **7984 MB** | 同上 |
| 物理内存总量 | 16248 MB | `TotalVisibleMemorySize` |
| 命中狙杀名单的进程 | **0 个**（它们压根没在跑） | `KILLED=` 为空 |
| 3 秒 CPU 采样里唯一吃 CPU 的 | 我自己的 powershell，0.7% | 同上 |

所以"环境脏"不是这轮数据的原因。每一次正式跑之前都重新确认过进程与可用内存，最后一次复核见 §6。

---

## 2. 第二步：三项优化已合进 bonsai_vk.cpp，改前改后都有实测

编译：`runtime_vk/build_mingw.bat`（g++ 16.2，`-Wall` 无警告）产出 `bonsai_vk.exe`。
原始输出：`runs/pipe_measured.txt`；已并入 `stage3_measured.txt` 的 F 段；`DEEPINFER_EVAL.md` §3
里那条推算已就地替换成实测表。

### 2.1 队列分离：大块搬运全部改走 transfer-only 队列（family 2）

设备实报：`family 0 = 0xF COMPUTE+TRANSFER+GRAPHICS+SPARSE`、`family 1 = 0xE`、
**`family 2 = 0xC 只有 TRANSFER+SPARSE，tiler=64，contexts=8`**。跨队列共享要求把 arena / stage /
readback 三类 buffer 从 `EXCLUSIVE` 改成 `VK_SHARING_MODE_CONCURRENT` + `{family 0, family 2}`。

| 指标 | 改之前 | 改之后 | 说明 |
|---|---|---|---|
| 权重批量上传 10.5 MiB | 4.93 GB/s（在计算队列上） | **7.11 GB/s**，引擎 1550.4 µs / wall 2268.3 µs | 实测 |
| 993,280 B logits 的 D2H | 占住 GPU **160.2 µs** | SDMA **140.2 µs = 7.09 GB/s**，不占计算队列 | 实测 |
| GEMV 本体速率 | 176.3 Gop/s（0.286 ms/op） | **176.3 Gop/s**（0.286 ms/op） | 实测：CONCURRENT 不收手续费 |

第三行是这一节最需要证据的地方——共享模式在很多驱动上会关掉 tile 优化。这张卡上同一形状
（M=12288, K=4096, splits=4）改前改后完全一致，正确性仍是 `max|err|=0.00000 rel=0.00e+00`，
wall 与时间戳交叉核对仍是 99.6%。

### 2.2 提交粒度纪律：把"37% 排队损耗"从推算变成实测，并拆成两半

新仪表 `BONSAI_PIPE=1`：每 token = 201 次 dispatch + 993,280 B logits D2H + 宿主 argmax，
min-of-3，NTOK=6。（OPS=201 与 VOCAB=248320 都来自模型本身：`prism.hadamard.weight_names`
正好 201 个名字，`tokenizer.ggml.tokens` 正好 248320 个。）

| 形状 | wall/token | GPU busy | copy/SDMA | host step | bubble |
|---|---|---|---|---|---|
| A 逐 op 提交 + 逐 op fence（201×） | **81.1 ms** | 时间戳不可信 | — | 6.46 ms | 71.5 ms |
| A2 逐 op 提交 + 每 token 1 fence | **69.3 ms** | — | — | 6.44 ms | — |
| B 每 token 1 CB，D2H 走计算队列 | **64.3 ms** | 58.1 ms | 160.2 µs | 6.37 ms | 6.19 ms |
| C 每 token 1 CB + D2H 走 SDMA | **64.6 ms** | 58.0 ms | **140.2 µs** | 6.47 ms | 6.67 ms |
| D 深度 2 流水（采样 N 与 GPU N+1 重叠） | **59.1 ms** | 58.0 ms | 140.2 µs | 6.39 ms | **1.13 ms = 1.91%** |

- 提交粒度总损失 = A − B = **16.8 ms/token**（81134.8 − 64325.4），不是原先推算的 20.0 ms；
  占 A 自身 wall 的 20.7%，占 GPU 忙时 58 ms 的 **29.0%**。
- 拆出来的两半：**等待 11.8 ms**（A − A2 = 14.6% of A）+ **提交 5.0 ms**（A2 − B = 6.1% of A）。
  也就是说真正贵的是每 op 一次 fence，不是每 op 一次 submit。这条以前没有人拆开算过。
- 深度 2 是唯一能把宿主采样藏起来的形状：bubble 6.2–6.7 ms → **1.13 ms**。
- 新坑（实测）：直接对 `HOST_COHERENT` 映射指针做 argmax = **24.8 ms/token**（WC 不可缓存读）；
  先 `memcpy` 进普通内存再扫只要 6.28 + 0.14 ms。采样器绝不能穿过映射指针读 logits。
- 时间戳的新边界（实测）：逐 op 提交 + 逐 op fence 时引擎时间戳读 48.2 µs/op，同一 op 在单个
  command buffer 里读 289 µs/op。这张卡的 timestamp 计数器挂在队列空闲时会掉速的时钟域上，
  **只在队列连续忙时可信**，所以 A 行的 GPU busy 标 untrust。

### 2.3 vkResetQueryPool 的驱动坑：真因有两个，都修了

- 实测 `vkGetDeviceProcAddr(dev, "vkResetQueryPool")` 在 instance 按 **VK_API_VERSION_1_1** 创建时
  返回 **NULL**（instance 级 GPA 能取到）；`BONSAI_API12=1` 改请求 1.2 后 device 入口解析成功。
  之前那次 amdvlk64.dll SIGSEGV 就是"通过导入桩跳进 NULL"。
- 第二个真因更致命也更好笑：上传块被提到 query pool 创建之前，于是
  `vkCmdResetQueryPool(ucb, c.qp = NULL, ...)` 直接把进程带走。**修法是把 pool 建在录任何 CB 之前**。
- 之前写进注释的"transfer 队列上录 barrier 会崩"是**误判**，本轮用 `BONSAI_TXBARRIER=1` 证伪：
  单独在 transfer-only 队列录一条 `vkCmdPipelineBarrier` 存活，输出明写 SURVIVED。
- 请求 1.2 之后整套数据没有回退：172.7 Gop/s、D 形状 59.1 ms/token，与 1.1 时的 176.3 / 59.1 一致
  （差 2% 在重复性漂移内，本轮观测到同一 op 0.285–0.346 ms 的历史区间）。

---

## 3. 第三步：端到端真实点火（模型真的跑起来了，文本在 §3.6）

### 3.1 引擎选型由两条实测事实决定，不是猜的

1. **官方 llama.cpp 装不了这个模型**。b10889（commit 311d4211b）直接拒绝：
   `gguf_init_from_reader: tensor 'output.weight' has invalid ggml type 143. should be in [0, 43)`
   —— 上游 `GGML_TYPE_COUNT = 44`，PTQ1_0 = 143 不在范围内。
2. **PrismML 官方 fork 可以**。`prism-vk/prism/`，build **b10777-e3de71522**，
   Windows Vulkan x64 预编译包（MSVC 19.51.36260.0，含 57 MB 的 `ggml-vulkan.dll`）。
   该分支 `GGML_TYPE_PTQ1_0 = 143`、`GGML_TYPE_COUNT = 144`，并带真正的 PTQ1_0 Vulkan kernel。

引擎自己对这张卡的探测结果（llama-bench 启动行原文）：

    ggml_vulkan: 0 = Radeon RX 550X (AMD proprietary driver) | uma: 0 | fp16: 0 | bf16: 0
    | fp4: 0 | warp size: 64 | shared memory: 32768 | int dot: 0 | matrix cores: none

`fp16: 0` 与 `int dot: 0` 是这张卡的原罪（gfx803 无 DP4A、驱动不放开 fp16 存储），三值 kernel
必须走 f32 反量化路径。这与我们自己 runtime 早先测到的硬件结论一致，属独立来源交叉验证。

### 3.2 真实加载时间：机械盘 → 内存 → 显存

为了拿到真的读盘而不是被页缓存骗：cp 出副本 → 写 12 GiB 垃圾把 standby 页挤掉 → 删垃圾 →
加载副本。`tools/vram_sample.ps1` 以 1 Hz 采 WDDM 计数器（GPU 版任务管理器同一套），
结果在 `runs/vram_cold_real.tsv`。

| 阶段 | 实测 | 证据 |
|---|---|---|
| 冷加载（load_model 到 initializing） | **16.05 s**（0.00.701 到 0.16.755） | `runs/run_cold_real.log` |
| 同一模型页缓存命中后加载 | **3.68 s**（0.00.125 到 0.03.807） | `runs/run_ngl99b.log` |
| 冷加载期间磁盘实际读量 | **2196 MB ≈ 2.14 GB**（文件本体 1.86 GiB） | 计数器积分 |
| 盘读速率 | 峰值 **175.7 MB/s**，平均约 137 MB/s（2196 MB ÷ 16 s，实测派生） | 同上 |
| 显存占用爬升 | 4.6 → 1690.7 → 1984.9 → **1985.3 MiB** | 同上 |

137 MB/s 平均、175 MB/s 峰值正是 WD10EZEX 的顺序读水平，说明这 16 s 确实花在盘上；3.68 s 那次
是内存命中。**这台机器上首次点火/换模型的真实代价约 16 秒。**

### 3.3 显存峰值：两套独立口径互相印证

WDDM 口径（驱动自己的账，per-process）：**1985.3 MiB** 专用显存；另有 291.9 MiB 系统内存后端
（Non Local / Shared）。

llama.cpp 自己的账（-ngl 99 -c 4096，退出前打印）：

    memory breakdown [MiB] |   - Vulkan0 (RX 550X) | 4096 = 1463 free + (2024 used
      = 1746 model + 178 context + 100 compute) + 607 unaccounted

分解（同一份日志的 load_tensors 行）：Vulkan0 model buffer **1682.24 MiB** + CPU_Mapped 212.19 MiB
= 1894.43 MiB，与我们早前独立算出的权重总量 1894.424 MiB 完全对上；KV buffer **128.00 MiB**
（4096 cells × 8 layers，K f16 64 + V f16 64）= **32 KiB/token**；GDN 递归状态 **50.25 MiB**；
compute buffer 100.11 MiB。**33/33 层全部 offload 成功，没有 OOM。**

### 3.4 真实吞吐：llama-bench 预热后的实测（runs/bench_ngl99_vs_cpu.log）

| 后端 | ngl | 测试 | t/s（实测，r=1） |
|---|---|---|---|
| Vulkan | 99 | pp512 | **14.29 ± 0.00** |
| Vulkan | 99 | tg128 | **4.05 ± 0.00** |
| CPU（ngl 0 回退） | 0 | pp512 | 13.59 ± 0.00 |
| CPU（ngl 0 回退） | 0 | tg128 | 3.71 ± 0.00 |

把 33 层全压进显存，解码只比纯 CPU 快 9.2%（4.05 ÷ 3.71，实测派生），预填充只快 5.2%。
这是本轮最扎手的实测：现成引擎在这张卡上几乎没吃到 GPU。

### 3.5 端到端 50 token 真实点火（用户指定的那一次）

命令（日志 runs/run_ngl99.log， verbose 版 runs/run_ngl99b.log）：

    llama-cli.exe -m models/Ternary-Bonsai-2-9B-PTQ1_0.gguf -dev Vulkan0 -ngl 99
      -c 4096 -n 50 -s 42 --temp 1.0 --top-k 20 --top-p 0.95
      -f tools/prompt.txt -st --simple-io

采样参数取模型自带默认（general.sampling.temp = 1.0、top_p = 0.95、top_k = 20），seed 42，
chat template 生效（该 build 里 --jinja 默认开启），实际喂进去的是
`im_start user / 用一句话解释什么是三值模型 / im_end / assistant / think`，共 **17 个 prompt token**。

| 端到端指标 | 实测值 | 出处 |
|---|---|---|
| 预填充 17 token | **2792.657 ms**（164.27 ms/token，6.09 t/s） | 引擎自己的计时器 |
| 解码 50 token | **11092.483 ms** = **226.377 ms/token = 4.417 t/s** | 同上 |
| 整进程 wall clock（含加载） | **21.413 s**（bash time） | runs/run_ngl99.log |
| 50 token 摊到整进程 | 50 ÷ 21.413 s = **2.34 t/s**（实测派生，含读盘与显存上传） | 同上 |
| 400 token 长跑（同参数） | wall **96.749 s**，仍是 4.4 t/s，无衰减 | runs/run_cold_dio.log |

预填充那 6.09 t/s **不能当 prefill 性能用**：17 token 的那一批里混进了首次建图、pipeline 编译和
warmup，两次一模一样的运行给出 4.0 与 6.1 t/s，波动 50%。真正的 prefill 数字取 §3.4 的
**pp512 = 14.29 t/s**（llama-bench 带 warmup、r=1）。

### 3.6 真实生成的文本（50 token，逐字照抄，未改写）

    [Start thinking]

    Thinking Process:

    1.  **Analyze the Request:**
        *   Topic: Three-valued model (三值模型 - sānmí zhǐ móxíng).
        *   Constraint: Explain in one sentence (

第 50 个 token 正好停在半句上。这不是 bug，是一条必须记录的事实：**这是带推理链的 thinking
模型，50 token 全被它的思考过程吃掉，还没进入正式回答。**

按同一参数再跑 400 token（runs/run_cold_dio.log，wall 96.749 s，仍是 4.4 t/s 无衰减），它还在
打草稿，第 4 步第一句就断了：

    3.  **Draft Initial Definitions:**
        *   三值模型是指逻辑或数据中有三种状态的系统，比如真、假和未知。
        *   三值模型是一种允许变量取值有三个选项（如真、假、未知）的逻辑或数据库模型。
        *   它是指除了真和假之外，还包含第三种状态（如不确定）的逻辑判断模型。

    4.  **Refine for Clarity and Conciseness (One Sentence):**
        *   Draft: 三值模型是一种允许

内容本身是对的（三种状态、真/假/未知），说明 PTQ1_0 的 1.75 bpw 三值量化在这张卡上没有把语义
跑坏。要看到它给出正式那一句，需要 500–600 token（推算依据：400 token 用完仍在草稿阶段，
每 token 247 ms，550 token 约 136 s），或者关掉 thinking。

---

## 4. 4 GB 显存墙：实测边界，不是"容量上限"四个字

`-ngl 99` 下逐档抬上下文，每档跑 1 个 token 看能不能起来（runs/ladder_c*.log）：

| -c | 结果 | 失败点（日志原文） |
|---|---|---|
| 4096 / 8192 / 16384 / 32768 / **49152** | **全部成功** | — |

（4096 档的证据是 §3.5 的 `run_ngl99.log` / `run_ngl99b.log`，它本来就是 -c 4096 跑满 50 token
成功的； ladder 从 8192 开始逐档抬。）
| **53248** | 失败 | `graph_reserve: failed to allocate compute buffers`（155,303,944 B） |
| 57344 / 61440 | 失败 | 同上 |
| 65536 | 失败 | `vk::Device::allocateMemory: ErrorOutOfDeviceMemory`，KV 的 1 GiB 缓冲直接拒 |

32768 档的完整账单（runs/breakdown_c32768.log，-v）：

    Vulkan0 (RX 550X) | 4096 = 512 free + (2948 = 1746 model + 1074 context + 128 compute) + 634

KV 实测 **1024.00 MiB / 32768 cells / 8 layers** = 32 KiB/token，与 §3.3 在 4096 档测到的完全
一致（跨 8 倍规模线性，无额外开销）。compute buffer 从 100.11 MiB 涨到 128.11 MiB。

**结论（实测）**：真实可用上限落在 **49152 通过、53248 失败** 之间。之前 DEEPINFER_EVAL 里
"60.8k 才撞墙"的推算偏乐观约 15–20%，原因是它只算了 KV + 权重，漏了 compute buffer 随上下文
增长、以及驱动保留的 607–634 MiB unaccounted。**先撞的还不是 KV，是 prefill 的 compute buffer。**

---

## 5. 现成引擎 vs 我们自己的 kernel：把差距钉死

同一张卡、同一批权重（1682.24 MiB 常驻显存）、同一件事（每 token 把全部权重读一遍）：

| 口径 | 实测值 | 来源 |
|---|---|---|
| 引擎解码 | 226.38 ms/token（4.417 t/s，50 token 实跑）；bench tg128 = 246.9 ms/token | §3.5 / §3.4 |
| 引擎有效权重读带宽 | 1682.24 MiB ÷ 226.38 ms = **7.79 GB/s** | 实测派生 |
| 我们 kernel 的 packed-weight 读带宽 | **44.1 GB/s**（M=12288 K=4096 splits=4，0.286 ms/op，176.3 Gop/s） | §2 / stage3_measured.txt |
| 我们 harness 的整 token 形状（D，深度 2） | **59.1 ms/token**（GPU busy 58.0，bubble 1.13） | runs/pipe_measured.txt |
| 两者比值 | 226.38 ÷ 59.1 = **3.8 倍**；带宽口径 44.1 ÷ 7.79 = **5.7 倍** | 实测派生 |
| CPU 侧对照 | 8.95 B 参数 ÷ 270 ms = **33 Gop/s**（ngl 0，3.71 t/s） | 实测派生 |

三个可执行的读法：

1. 引擎在这张卡上只把显存带宽用掉了 **17.7%**（7.79 ÷ 44.1），原因写在 §3.1：`fp16: 0`、
   `int dot: 0`，三值只能走 f32 反量化，再叠加每 token 的图构建/同步；
2. 我们的 kernel 已经把这层拿掉了（同一 op 176.3 Gop/s），整 token harness 到 59.1 ms，
   但 harness 只含 201 个投影 op，**不含 attention/GDN/采样**，所以它不是模型级速度；
3. 因此现阶段的诚实说法是：**现成引擎 4.4 t/s 是实测终点，16.9 t/s 是我们 kernel 的上界，
   中间那 3.8 倍要靠把真实计算图（GDN 递归 + 8 层 attention + 采样）搬进我们自己的 runtime 去拿。

---

## 6. 诚实清单：这轮没能验证的、以及口径边界

1. **跑通端到端的是 PrismML fork，不是我们自己的 runtime。** `bonsai_vk.cpp` 第 16 行注释写得很
   清楚：GGUF 解析、tokenizer、chat template、attention、采样器都是故意不做的。所以 §2 的所有
   数字是 **kernel/提交层** 的实测，§3 的所有数字是 **模型层** 的实测，两者不能混着吹。
2. **PrismML 没有在任何地方声明支持 gfx803/Polaris。** 本轮是实测证据（能装、能生成、语义正确），
   不是官方承诺；fp16 与 int-dot 被驱动关掉时它靠 f32 回退路径活着。上游质量/精度未评估。
3. **官方 llama.cpp（b10889）确认跑不了 PTQ1_0**，报 ggml type 143 越界。想用这个模型就必须用
   PrismML 分支——这是路线约束，不是可选项。
4. **thinking 链长度未走完。** 50 token 与 400 token 都停在草稿阶段，正式那一句没有出现在日志里；
   关掉 thinking 或给 600 token 才能拿到，本轮没跑这一档。
5. **prefill 的 6.09 t/s 不能当性能用**（17 token 批次里混了一次性建图与 warmup，两次同样命令
   给 4.0 与 6.1）；pp 的正式口径是 §3.4 的 14.29 t/s，且只跑了 r=1。
6. **r=1。** llama-bench 默认 5 次重复，这轮为了省时间只跑 1 次，`± 0.00` 是单样本的自然离散度，
   不是置信区间。
7. **49152 只比"能起来"多 3 MiB 余量**（1746 + 1536 + 128 = 3410 vs 实测 free 3413 MiB），换个
   桌面程序就可能翻；上限应当按 49152 而不是 53248 报。
8. **清场这步实测为 0 个可杀进程**（§1）。这轮所有数字本来就在干净环境下取的，事后复核：
   Available MBytes **8832 MB**，残留进程只有 13 个 `msedgewebview2.exe`（白名单内），
   没有 llama/chrome/java 残留。

复现（全部已在脚本里）：

    # kernel + 提交层
    cd "runtime_vk" && ./build_mingw.bat
    BONSAI_PIPE=1 BONSAI_TXBARRIER=1 ./bonsai_vk.exe 12288 4096 4 100
    BONSAI_API12=1 BONSAI_PIPE=1 ./bonsai_vk.exe 12288 4096 4 100
    # 模型层：显存/盘计数器采样 + 点火
    powershell -File tools/vram_sample.ps1 -Name llama -Out runs/vram.tsv -Seconds 300
    cd prism-vk/prism && ./llama-cli.exe -m models/Ternary-Bonsai-2-9B-PTQ1_0.gguf \
      -dev Vulkan0 -ngl 99 -c 4096 -n 50 -s 42 --temp 1.0 --top-k 20 --top-p 0.95 \
      -f "tools/prompt.txt" -st --simple-io
    ./llama-bench.exe -m models/Ternary-Bonsai-2-9B-PTQ1_0.gguf -dev Vulkan0 -ngl 99,0 \
      -p 512 -n 128 -r 1 -o md

原始日志与采样表都在 `runs/`：`run_ngl99.log`（50 token 主结果）、`run_ngl99b.log`（同参数 -v 版，
账单在里面）、`run_cold_real.log` + `vram_cold_real.tsv`（冷加载 16.05 s 与 175.7 MB/s 峰值）、
`bench_ngl99_vs_cpu.log`、`ladder_c*.log`、`breakdown_c32768.log`、`pipe_measured.txt`。
