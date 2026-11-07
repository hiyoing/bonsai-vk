# 交接文档 — 三值 MoE / Bonsai-2-9B 原生 Vulkan 运行时

> 读者：接手这个项目的人（或下一次会话里的 agent）。
> 目的：**让你在不开机瞎摸的前提下做出正确判断**。
> 每个数字后面都标来源：`实测` = 从本仓库某次运行的日志直接抄；`推导` = 由实测值算出并给算式；`未验证` = 猜测，必须先测再用。
> 项目根：`<repo-root>`　最后更新：2026-11-05（STAGE10：GQA 4× 冗余合并 + 多卡流水线取证 + 两个 Bug 类 + 27B 架构适配器；上一个里程碑 STAGE9：chat template + 宽内核 + 时间拆解 + 叛徒定性 + OOM 兜底 + Roadmap）
>
> **读之前先看这条**：§1、§3、§4、§5、§6、§7 已被 STAGE9 更新，**§15 是 STAGE10（2026-11-05 凌晨）的新事实**
> （多卡 rig 的实测链、两个 Bug 类、batch H 的权威 GQA 阶梯、27B 适配器）。
> §7.1–§7.4 原来写的四项任务**现在都结了**，
> 里面那些「接下来的三步」「要做的只有一行」是历史，别照做——做之前先读 §15、§10 和 `ROADMAP.md`。
> **特别一条**：§3.1/§4.3 表里的**源码行号是 STAGE9 那一版量出来的**，GQA 合并与 fleet 代码进来之后已经整体下移
> （`bonsai_forward.h` 1990 → **3268 行**，`bonsai_weights.h` 764 → **837 行**，实测 `grep -c ""`）。
> 现在有了 git，**行号一律用 `git diff d5ea17a -- runtime_vk/<文件>` 复核，别再抄本文件里的旧行号**。

---

## 0. 三十秒上手（如果你只做一件事）

```bat
cd /d "runtime_vk"
build_mingw.bat
bonsai_vk.exe run "models/Ternary-Bonsai-2-9B-PTQ1_0.gguf" 32 48
```

预期：权重装载 ~1090 ms（实测，`runs/ignition_v3.log`），`=== pure-blood token rate ===` 报 **14.2 tok/s**
（`实测` STAGE10 默认配置 = 48 token / 3389.5 ms = **14.16 tok/s**，`runs/regression_v14.log:95`；
历史值 13.99 / 14.04 / 14.05 属于 STAGE8/9 的二进制，见 §6.1 与 §15.1）。默认 prompt 现在**会自动过 chat
template**（`BONSAI_CHAT` 默认 1，`bonsai_forward.h:201`），所以是 16 个 token 而不是 6 个，输出以 think
段开头。想同时验位相同门就 `set BONSAI_QCHECK=1`——差 1 bit 直接 `rc 6` 拒绝生成。
显存装不下时程序**不会崩**：它打一张预算表 + 一个可用的偶数 `BONSAI_CAP` 然后 `return 4`（§5.9）。
跑不出来先看 §12 的「构建失败三种可能」。

**本仓库现在是 git 仓库**（2026-11-05 更正；此前本节写的是「本仓库不是 git 仓库、没有版本回滚」，那句话对今天的树已经不成立）。
基线提交 `d5ea17a`「snapshot: 9B ternary engine at the GQA-merge point (pre-cleanup baseline, .bak safety nets included)」，
2026-11-05 00:31:25 +0800（`实测`：`git log --format=%H %ad %s`）。`.git/` 与 `.gitignore` 都在仓库根。
**构建产物（`*.exe` / `*.spv`）是被有意跟踪的**，不是疏忽：`.gitignore` 自己的注释写明「本项目的交付物就是一台没有包管理器的机器上的
一个二进制 + 它的 SPIR-V」，所以只有非源码、非测量、非产品的东西才进 ignore（`__pycache__/`、`*.pyc`、`.venv/`、`runs/*.gguf`、`runs/*.safetensors`、缩略图缓存）。
安全网因此换了性质：`.bak` 文件是**历史遗留**（基线提交里还有，工作区已把它们删掉 21 个文件未提交，`实测` `git status --porcelain`），
**回滚请用 `git diff d5ea17a` / `git checkout d5ea17a -- <文件>`，不要再靠复制 `.bak`**。

---

## 1. 一句话状态

自己写的原生 Vulkan 引擎在三值 30B-A3B（`Ternary-Bonsai-2-9B-PTQ1_0.gguf`）上**端到端跑通，实测 14.05 tok/s**（`runs/oom_fallback_v3.log` 分支 D，最终二进制），是官方 llama.cpp Vulkan fork（4.417 t/s，同卡同模型）的 **3.18×**，且每一步都有 oracle 逐张量对齐证据。
模型吃的是**它自己的 chat template**，上下文已经到 **cap = 2048**，单 token 时间有**逐项仪表**，装不下显存时**有预算表并干净退出**。

未闭合的仍然是「产品化外壳」，不是核心正确性，而且只剩三条：

1. **prefill 还是逐 token 一次 submit**（§8.1）——1579 token 的 prompt 实测 **131.99 s**（`runs/qcheck_2048_v2.log:93`，11.96 tok/s）。这是产品意义上唯一还大的体验洞。
2. **发布包没打**：`--help` / `--version` / 三个 MinGW dll 的 Release 目录，方案全在 `ROADMAP.md`，没执行。
3. **chat template 渲染后的整串 id 没有与官方 `llama-tokenize -chat-template` 逐 ID 对过**（§7.1 步骤 D4，仍是开的）。tokenizer 本身 39/39 对过，模板的两条 id 路径在引擎内互相 IDENTICAL，但**外部对照缺**。

已闭合、以前在这个列表里写着「未闭合」的：chat template（§5.6）、cap 256 的墙（§5.7）、时间拆解（§7.2）、
块 0 的 4 个锚点叛徒（§7.3，定性为 oracle 侧 q8_0 重量化，**不修**）、OOM 兜底（§5.9）。
采样器仍然只自测过，没跟官方 sampler 对过——这条从 STAGE8 起就一直开着，进 `ROADMAP.md` 的「不进 Release-0.1.0」。

### 里程碑一览

| 阶段 | 交付 | 关键实测数字 | 文档 |
|---|---|---|---|
| STAGE0–1 | 硬件/带宽/GEMV 内核摸底 | 加权 147.3 Gop/s（原生 kernel） | STAGE0_REPORT.md / STAGE1_KERNEL.md |
| STAGE2–3 | 免 UAC 工具链 + 原生 Vulkan 真机跑通 | 正确性真因是漏 `vkCmdPipelineBarrier` | STAGE2_CUSTOM.md / STAGE3_RUNTIME.md |
| STAGE4 | DeepInfer 四项机制可行性评估 | SDMA 7.12 GB/s 且与计算并发不打折；1 token = 7.929 GMAC | DEEPINFER_EVAL.md |
| STAGE5 | 端到端基线（官方 fork PrismML b10777） | 4.417 t/s、冷加载 16.05 s、显存峰值 1985.3 MiB、4GB 墙实测 `-c 49152` 过 / `53248` 撞（先撞 compute buffer）；GPU offload 仅比 CPU 快 9.2%（本机 `fp16:0 int dot:0`）；「37% 排队损耗」实测拆成 fence 等待 11.8 ms + submit 5.0 ms = 16.8 ms | END_TO_END_MEASURED.md |
| STAGE6–7 | 忠实复现三件套 | Tokenizer **39/39 逐 ID** 与官方 `llama-tokenize` 1:1；PTQ1_0 反量化三方比对 **max\|err\|=0.00000**（100 块 12800 权重，扩测 1000 块 128000 权重同样 0）；合成边缘块 491520 权重逐位 0 差；层 3 注意力 score 最大误差 6.69e-04 / 2.94e-03 / 3.57e-03（seq 8/32/64），残差被完全归因到每行 norm 常数的 ±1 fp32 ulp | FAITHFUL_REPRO.md / ATTENTION_ALIGN_FINAL.md |
| STAGE8 | **纯血端到端点火** | **13.99 tok/s**（48 token / 3431.1 ms，稳态 71.5 ms/token，首 token 70.9 ms）；决定性发现见 §5.3 | runs/ignition_v2.log、本文 §6 |
| STAGE9 | 产品外壳四件套：chat template、宽内核（cap 2048）、时间拆解、叛徒定性 + **OOM 兜底与 Roadmap** | 点火 **14.04 tok/s**；宽内核 vs 窄内核 `cap=256` 逐位 0 差（SCORE/PROB/P16 各 0/4096）；1579 live keys 下 P16 打包 16384 对 0 失配；拆解表 `T_gpu 63.648 / T_cpu 1.640 / T_memory 6.270 / bubble 8.490 ms/token`；4 个锚点 FAIL 归因到 oracle 的 q8_0 激活重量化（22 种算术里只有它进 ±2.8e-4） | `ROADMAP.md` / `runs/traitor_forensics_v1.md` / 本文 §10 |

---

## 2. 环境与工具链（全部免管理员、零预算）

### 2.1 硬件（实测）

| 项 | 值 | 证据 |
|---|---|---|
| GPU | AMD Radeon RX 550X 4 GB（gfx803 / Polaris / GCN4） | runs/ignition_v2.log:1 |
| 驱动 | AMD 专有，Vulkan `driver 2.0.279`（amdvlk64.dll） | 同上 |
| UMA | **0**（独显，不是 APU → 权重要真过 PCIe） | runs/bench_ngl99_vs_cpu.log:3 |
| deviceLocal heap | **3840 MiB** | runs/ignition_v2.log:1 |
| warp size | **64** | runs/bench_ngl99_vs_cpu.log:3 |
| 官方 fork 报的 shared memory | **32768 B** | runs/bench_ngl99_vs_cpu.log:3 |
| `minStorageBufferOffsetAlignment` | **4 B** | runs/ignition_v2.log「forward bring-up」段 |
| `maxStorageBufferRange` | 4096.000 MiB | 同上 |
| `maxComputeWorkGroupSize[0]` | 1024 | runs/ignition_v2.log:2 |
| `bufferImageGranularity` | 1 | 同上 |
| compute queue | family 0；transfer-only queue = family 2 | runs/ignition_v2.log:4 |
| `timestampPeriod` | 40.000 ns | runs/ignition_v2.log:1 |
| CPU / RAM | R5 3600 / 16 GB | 用户环境 |
| fp16 / int dot / matrix cores | **全 0 / none** | runs/bench_ngl99_vs_cpu.log:3 |

`未验证`：我们自己的引擎从未打印过 `VkPhysicalDeviceLimits::maxComputeSharedMemorySize`。官方 fork 报 32768 B，但那是 ggml 自己查询后的打印格式。**扩 cap 之前先在 bonsai_weights.h:74 那行 printf 里补上这个字段实测一次**（见 §7.1 步骤 A），不要拿 32768 当公理。

### 2.2 工具链（D 盘，免 UAC）

```
<toolchain>\mingw64\bin                     WinLibs GCC 16.2 UCRT（g++ 16.2 + cmake 4.4.4 + ninja + dlltool）
<toolchain>\vkheaders\Vulkan-Headers-1.4.309\include
<toolchain>\vklib\libvulkan-1.a             自己从 C:\Windows\System32\vulkan-1.dll 生成：
                                       gendef vulkan-1.dll > vulkan-1.def
                                       dlltool --input-def vulkan-1.def --output-lib libvulkan-1.a --dllname vulkan-1.dll
<toolchain>\glslang\bin\glslang.exe         glslang 16.6.0 —— 名字是 glslang.exe，不是 glslangValidator.exe
```

MinGW 不附带 libvulkan-1.a，驱动也不给；**这个 import lib 丢了这个文件就报 `cannot find -lvulkan-1`**，重建命令就是上面两行。

### 2.3 模型文件（只读，永远只读）

```
models\Ternary-Bonsai-2-9B-PTQ1_0.gguf
GGUF v3，tensors 427（202 PTQ1_0 + 225 f32），kv 49，vocab 248320，weights 8,953,803,264 elems
```

上一个会话曾造过一个临时探针文件 `rope_probe_nrot4.gguf` 想验证 RoPE 维度，**已删除，模型本体从未被写过**。任何写模型文件的操作都要先问用户。

### 2.4 官方对照引擎（oracle 的来源）

PrismML llama.cpp fork `b10777`（官方 llama.cpp 拒这个 checkpoint 的 type 143，PrismML 能跑通 33/33 层）。它的 `-fa off` dump、`llama-tokenize.exe`、`ggml_vulkan` 后端日志是本项目全部对齐测试的真值来源。产物在 `runs/`（`dbg_full_real.log`、`stop31_dump.log`、`run_ngl99b.log`、`bench_ngl99_vs_cpu.log` 等）。

---

## 3. 代码地图

### 3.1 主机侧（runtime_vk/）

> **行数怎么读的**：下表「行数」列 = 当前工作区文件的 `grep -c ""`（实测 2026-11-05），STAGE9 的数字保留在括号里。
> 表内的**行号引用**（:117、:1930 这类）仍是 STAGE9 那一版的实测位置，GQA 合并 + fleet 进来之后整体下移了，
> 用之前先 `git diff d5ea17a -- runtime_vk/<文件>` 或重新 `grep -n`。

| 文件 | 行数 | 职责 |
|---|---|---|
| `bonsai_vk.cpp` | 1906 | `main()` 分派 + 设备/arena 初始化 + 内核基准 + 各种 BONSAI_* 探针。`gen` → weights_main（:188），`run` → fwd_main（:190），其余位置参数 → GEMV 基准。查询池创建 :371-372（`queryCount = 512`，创建顺序是硬要求：早于 upload，否则 NULL handle 让 amdvlk64.dll SIGSEGV） |
| `bonsai_forward.h` | **3268**（STAGE9：1990） | **前向与生成全部在这里**。下列行号实测于 build_v42 的同源文件（`wc -l` = 3268），改动后请以符号名重找，不要信行号。`Regions` :56；`Fwd` 结构 :107（管线枚举含 `P_QKW` :119，`kMaxNarrow/kMaxWide` :191，`cap` 默认 256 :225，`chat` 默认 1 :201）；**GQA 合并的四个字段 `gqa_opt = 1, gqa_part = 1u, gqa_tile = 512u, hpk = 0u` 在 :156（`实测`，这是引擎默认值）**；`dsp()` 记录并派发 :346（每个 dispatch 后跟一个 `bar_compute`）；`gemv_ptq` :379；`attn_block` :551；内核选择 `qk_reference_selection` :2096-2101（按 `cap<=256 ? 窄 : 宽`，> `kMaxWide` 走 tiled），merged 分支另在 :289 与 :1956-1957；时间仪表 `g_prof` :625；`fwd_token` :634；**显存预算 + OOM 兜底 :1691-1766**（`per_slot` :1693、预算表 :1722-1724、headroom :1730-1732、两条 `NOTHING WAS DISPATCHED` 分支 :1737/:1757、autofit :1772）；描述符池 `maxSets = 128` :1871；`Sampler` :2493；`fwd_run` :2551（chat 渲染在 :2562-2577）；`fwd_main` :2947（cap 拒绝 `return 4` :3103/:3145/:3149，内核选择打印 :3151）。**当前 getenv 清点（`实测` `grep -o 'getenv("BONSAI_...")' runtime_vk/bonsai_forward.h \| sort -u`）= 45 个变量**，见 §4.3 |
| `bonsai_chat.h` | **105** | **STAGE9 新增**。把 GGUF 里的 chat template 逐字翻成 C++：`render()` 出字符串（走 `Bpe::encode`），`render_ids()` 出独立的 id 序列，两者必须 IDENTICAL；`ids_are_singles()` 是硬前置（CONTROL id 若在词表里被拆成多枚就拒绝渲染）。四个 control id 只以**数字常量**出现（:41-44），字面量不进源码，原因见 §5.6 开头的转义约定 |
| `bonsai_gguf.h` | — | GGUF 头解析。**字符串长度是 uint64**（:65）—— GGUF v3 的规则，踩过坑见 §8.8 |
| `bonsai_weights.h` | **837**（STAGE9：764） | 权重计划（tensor 表 → 每 tensor 的 K/M/量化类型/arena 偏移）。**`device_local_heap()` :117**（STAGE9 新增：从 `memoryHeaps[]` 取真实 device-local 堆大小，预算表和 limits 行都用它；以前 `model_plan_report(e.plan, 0)` 在每份 bring-up 日志里打「device-local heap: 0 MiB」，是个真 Bug）；:289 是它的调用点。**`stage_alloc`（引擎的 host staging buffer + memory 分配处）在 STAGE10 被改过**：`uint32_t fam2[2]` 从 `if` 块里提到 `vkCreateBuffer` 同一作用域，修的是悬垂 `pQueueFamilyIndices`（机制见 §15.4，注释在 :315-321）。**改这里 = 必须重跑保真五件套**，已经重跑了：`runs/regression_v14.log` |
| `bonsai_tok.h` | 402 | 自己写的 byte-level BPE。`struct Bpe` :146，`load()` :159-228，pre-tokenizer :230-317，`encode_piece` :325-365（**字节起手**），`encode` :367-395（special 切分），`decode_ids` :397 |
| `attn_vk.h` | — | CPU 镜像 / 理想腿（oracle 侧），三个 attn 内核以它为正确性合同 |
| **`gpu_fleet.h`** | **1803** | **STAGE10 新增（任务 3）**。多卡/iGPU 的机械层：设备枚举与分类（compute / transfer-only / zero-copy 内存类型判定）、**按层切分预算算术**（连续区间 + head/tail 计费，GDN state 不跨卡的依据写死在检查里）、**pinned ring**（depth 必须是 2 的幂，slot 选择用 `seq & mask` 所以永远不用除法）、doorbell 盖章（`vkCmdFillBuffer` 4 B，与拷贝同 CB 同队列，stride 128 B = 本卡实测 `nonCoherentAtomSize`）、`fleet_pipeline_init` 的越界拒绝。**文件顶部 704 行是「数组生命周期」纪律**（§15.4） |
| **`bonsai_fleet.cpp`** | **1093** | **STAGE10 新增**：`gpu_fleet.h` 的驱动台，四个子命令 `probe` / `ring [depth] [slot_MiB] [micro]` / `split <layers> <cards> [budget_MiB] [--model] [--fallback]` / `selftest`（usage 文本 :1072-1083）。**跑不出来的事它会明说，不会把成功印在墙上**（见 §15.2 的 NOT EXERCISABLE 文案）。环境面：`BONSAI_GPU_INDEX` / `BONSAI_GPU_TYPE=integrated\|discrete\|zero-copy` / `BONSAI_MODEL` / `BONSAI_API12` / `VK_VALIDATION` |
| **`align_27b.cpp`** | **828** | **STAGE10 新增**：27B 架构适配器（`align_27b.exe`）。门禁见 §15.5 |
| `shared_probe.cpp` | — | `KV_P0P1P2_MEASURED.md` §4.1 的共享内存上限/占用探针（已挂进构建最后一步） |
| `kv_guard.h` | — | PDH Shared Usage 采样 + `decide()` 的 refuse/evict 策略（同上 §3） |
| `batch_g.sh` / `batch_h.sh` | — | 无人值守批量脚本；**batch_h 是 GQA 合并的配对复测（权威，§15.3）**，batch_g 是首轮非配对阶梯（归因仪器） |

### 3.2 对齐测试（**4 个可执行 + 1 个阶梯 + STAGE10 新增 2 个**）

| 程序 | 源文件 | 测什么 | 当前状态 |
|---|---|---|---|
| `align_tokenizer.exe` | align_tokenizer.cpp | 我们的 BPE vs 官方 llama-tokenize 逐 ID | **32 + 7 = 39/39 通过**（两个探针文件，见 §4.5） |
| `align_dequant.exe` | align_dequant.cpp | 把官方 dequant 着色器编上卡做三方比对 | **max\|err\| = 0.00000** |
| `align_f16.exe` | align_f16.cpp | f16 RNE 打包（含次正规/0/负 scale 边缘） | 65536 个数 0 错、491520 权重逐位 0 差 |
| `align_attn.exe` | align_attn.cpp | 层 3 注意力：内核 vs CPU 镜像 vs 官方 dump | score 最大误差 6.695e-04（seq 8），占满量程 6.40e-06 |
| **`align_27b.exe`** | **align_27b.cpp（STAGE10 新增）** | 27B 架构适配器：PTQ1_0 解码、64-block 混合路由（SSM 层 KV charge = 0）、Hadamard version/block_size 路由 | `runs/regression_v14.log` 的 `--- align_27b` 段 + `runs/align_27b_v2.log`：**`ALL CHECKS PASSED (0 failure(s))`**（`实测`）。四条负向控制全部**按要求拒绝**。语义边界见 §15.5 |
| **`bonsai_fleet.exe`** | **bonsai_fleet.cpp + gpu_fleet.h（STAGE10 新增）** | 多卡/iGPU 机械层：`selftest`（纯主机代数 + 枚举，**不建任何 Vulkan 对象**）/ `probe`（拓扑，不建逻辑设备）/ `ring <depth> <slots> <mb>`（pinned ring 端到端 + 带宽）/ `split <layers> <cards> [mib]`（只算调度，不开设备） | `selftest: 55 checks, 0 failed -> PASS`（`runs/fleet_ring_v4.log:73`、`runs/fleet_offline_v1.log:72`，**实测**）。`probe` 在**本机**判 (A) 层流水线 **NOT EXERCISABLE**（只有 1 个 COMPUTE 设备）、(B) iGPU zero-copy **AVAILABLE**（`runs/fleet_probe_v2.log`，实测）。逐条数字见 §15.2/§15.3 |
| `BONSAI_ALIGN=1` | bonsai_forward.h | block-0 逐 op 锚点比对 | **4/32 失败**，已定性为 oracle 侧 q8_0 重量化，**不修**（§7.3） |
| `BONSAI_LADDER=1` | bonsai_forward.h | 每 block 一次 submit，读 RES，与 l_out-0..31 全 32 个 oracle 比 | 32/32 分支和 ≤5%，28/32 残差 ≤1% |
| `BONSAI_QCHECK=1` | bonsai_forward.h | QK 内核审计 **A/B/C/D/E 五项 + STAGE10 的 A2（wide↔tiled）与 A3/A4（GQA 合并↔被它替换的内核）** | `cap 256` 窄宽**逐位 0 差**（`runs/qcheck_256_v3.log:96-101`）；`cap 2048` / 1579 live keys：16384 对 P16 0 失配、死槽 0 违规、\|sum−1\| ≤ 1.472e-05、双精度点位差 0.000e+00（`runs/qcheck_2048_v2.log:95-101`）；**STAGE10 后 `gate: PASS (bitmask 0 = A ok A2 ok A3 ok A4 ok B ok C ok D ok E ok)`**（`runs/regression_v14.log:94`，实测） |

### 3.3 着色器（shaders/*.comp → 同名 .spv，构建自动生成）

`build_mingw.bat` 对 `shaders\*.comp` 逐个跑 `glslang -V -O`，所以**加一个 .comp 就自动产 .spv，不用改脚本**。

与注意力相关的四个（**项目自己立的规矩：窄版三件套是 0-error 的，扩功能时新增内核而不是改它们**，见 bonsai_forward.h 里 attn_block 上方的注释；STAGE9 正是照这条规矩新增宽内核，窄内核一行没动）：

| 内核 | local_size | 与 `cap` 的关系 | 备注 |
|---|---|---|---|
| `attn_norm_rope.comp` | 256（= head_dim） | **无关**：arena 按 `position` 索引（:117 `kc[(position*nkv+h)*hw+d]`），`cap` 只是 stride | 每头 RMSNorm → ×norm weight → 半分裂 RoPE（只转前 n_rot=64，配对 (p, p+32)）→ fp16 arena 写入；V 不 norm 不 rope 直通（实测） |
| `attn_qk_softmax.comp`（**窄**） | 256 | **lane 就是 key 位置**，所以 `cap <= 256`：`shared float sd[256], sp[256]` | stage 2：K^T·Q（无 scale）→ scale 放进 softmax（1/sqrt(head_dim) = 0.0625 实测）→ 加性 mask → 减 max → exp → 除 sum → f32 prob + fp16 prob。仍是 `cap <= 256` 的默认路径（`kMaxNarrow` :113），同时是宽内核的**交叉验证参考**：同 cap 下两者必须逐位相同 |
| `attn_qk_softmax_wide.comp`（**STAGE9 新增**） | `NTILE = 256`，`MAXCAP = 2048` | lane 跨步覆盖 `for (k = j; k < cap; k += NTILE)`（:59），打包同样跨步 `for (w2 = j; w2 < cap/2; w2 += NTILE)`（:106） | 2 × `float[2048]` = **16 KiB**，是本卡实测 `maxComputeSharedMemorySize = 32768 B`（`runs/limits_probe.log:3`）的一半；4096 会正好压满不留余量，**所以 2048 的墙是 shared memory，不是显存**。点积表达式逐字照抄窄内核（顺序即合同，§8.3） |
| `attn_pv.comp` | 256（= head_dim） | **无关**：每 lane 自己升序扫 `j < nfill`（:42） | stage 3：`out[d] = Σ_j p[j]·V[j][d]`，再 `gated = out × sigmoid(gate)`；gate 来自 `deint_qg.comp` 去交错后的连续窗口 |

其余 **52 支 `.comp` + 5 支 `.glslh`**（`实测`，2026-11-05 `python tools/source_size.py` 的「shader sources」行；它们编出 **52 个 SPIR-V 模块、合计 300 KiB**）是 GEMV / 逐元素 / FWHT（`hat4096`、`hadamard_fwht`）/ 量化重排 / `deint_qg`（§5.3 的修法）/ 共享内存探针（`probe_shm_n*`、`bench_shm_n*`）/ 逐元素探针。**STAGE9 之后新增的行（`实测` 目录列表 `ls runtime_vk/shaders`）**：

| 内核（新增） | 归属 | 与 `cap` / LDS 的关系 | 证据 |
|---|---|---|---|
| `attn_qk_softmax_tile_q{1024,2048,4096,8192}.comp` + `attn_qk_softmax_tile.glslh` | KV 报告 §9（tiling） | **无 `MAXCAP`**：行长与 LDS 解耦，LDS = `4*QTILE+4` B → 4/8/16/32 KiB；`q8192` 是越限探针，auto 到不了 | `KV_P0P1P2_MEASURED.md` §9.2/§9.4 |
| `attn_qk_softmax_gqa_h4_q{256,512,1024}.comp` + `attn_qk_softmax_gqa.glslh` | **STAGE10：GQA 4× 冗余合并** | 一个 workgroup 服务 `HPK = n_head/n_head_kv = 4` 个 query 头，**grid 16 → 4/层**；LDS = `HPK*4*QTILE + 8*HPK` B = q256 → 4128 B、q512 → 8224 B、q1024 → 16416 B（`实测`，日志行 `runs/c32768_on1_v1.log:33`、`runs/short_c256_on3_v1.log:33`） | §15.3 + `runs/gqa_pair_v1_summary.txt` |
| `attn_pv_q8v.comp`（frozen）/ `attn_pv_q8v_gqa.comp`（merged） | KV 报告 §2 + **STAGE10** | q8_0 V 的 fused 读路径；merged 版把每个 key 的 V word/scale 解包从 4 个 query 头共享。**两者被 A4 门禁逐位钉住**（`runs/regression_v14.log:90-91`：PGATE 0/4096 bit-diff、GOUT 0） | §15.3 |
| `attn_norm_rope2.comp` | KV 报告（norm+rope+pack 合一） | 与 cap 无关（按 `position` 索引） | `runs/short_c256_on3_v1.log:57` 的 pipeline 行 |

### 3.4 工具（tools/，Python，只读模型）

| 脚本 | 用途 |
|---|---|
| `probe_chat_template.py` | **本次会话新增**：只读拉出 chat template 全文 + special token 表 + 全部 metadata，落 `runs/gguf_metadata_dump.json` 与 `runs/chat_tpl_probe.log` |
| `oracle_extract.py` | 从官方 fork 的 dump 里抽中间张量做 oracle（block-0 前 6 stage / l_out-0..31） |
| `oracle_step1.py`（33 KB） | 锚点比对主体 |
| `oracle_signs.py` | Hadamard 符号向量核对（sum4096=60 / sum12288=-94） |
| `gdn_probe.py` | GDN 语义假设检验 |
| `rope_fit.py` | RoPE 维度/基拟合 |
| `attn_softmax_check.py` | softmax scale 的自由二分拟合（得出 0.062501） |
| `ptq_scale_scan.py` | PTQ scale 扫描 |
| `gguf_dump.py` | 通用 GGUF 头/张量表 dump |
| `make_tiny_qwen35.py` | 造 tiny 测试模型（runs/tiny_qwen35.gguf，42 MB） |
| `clear_field.ps1` | 跑基准前清场（按名单杀进程 + 验证可用内存） |
| `vram_sample.ps1` | 显存峰值采样（配合 STAGE5 的 1985.3 MiB 结论） |
| `gen_dequant_ref_comp.py` | 生成「官方 dequant 原样编上卡」的 oracle 着色器源码（构建第 0 步会调用） |
| `qkv_arith_probe.py` | **STAGE9 的决定性工具（§7.3）**：CPU 重建 block-0 的 GEMV 操作数并验证它 vs oracle 自己的锚点，再把 **22 种候选算术**逐个投进 oracle 的 4 位小数打印窗口（±5e-5），谁全中谁就是 oracle 的算术。结论：只有对称 q8_0 激活重量化族进得了 ±2.8e-4。纯 CPU，模型只读，`QKV_PROBE_CACHE=1` 复用 `runs/qkv_probe_hr.bin` |
| `order_sensitivity.py` | **§7.3 的证据 A**：从 `ladder_v6_{rho,_s1,_s4}.log` 算 `d_order/Sum\|x\|` vs `gap/Sum\|x\|`，把「累加顺序」否证到四个数量级 |
| `oracle_self_consistency.py` | **§8.7 说的那类审计**：先量 oracle 自己打印的聚合值之间的自相矛盾（`l_out = ADD(ffn, attn_residual)` ⇒ 三个 sum 必须自洽），那个差就是所有 rho/delta 的下界 |
| `make_probes.py` / `oracle_tokenize.py` | tokenizer 39 探针与官方 `llama-tokenize` 参考值的生成/比对（`runs/tok_probes*.txt` + `runs/tok_ref*.jsonl`） |
| `patch_gguf_meta.py` | **会写文件的工具**：`<in.gguf> <out.gguf> key=value` —— 复制一份再改 metadata 标量/数组（同字节长度，长度不符就拒绝），用于 RoPE 符号探针（把 `rope.dimension_count` 缩到 4 好让一对落在打印的前 3 列里）。**输入文件永远只读**，产物是另一个文件；这也正是用户铁律「任何写模型文件的操作都要先问用户」在本仓库的执行方式 |
| **`source_size.py`** | **STAGE10 新增，代码量与产物尺寸的唯一生产者**（`实测`，2026-11-05 的输出）：`python tools/source_size.py` 打印 host / shaders / tools 三组的 raw、去注释、去空白尺寸 + 产物。当前值：**engine = 696 KiB raw / 416 KiB 去注释 / 279 KiB 去空白**；tools 310 KiB 去注释；**whole tree 726 KiB 去注释**；`bonsai_vk.exe` **539 KiB**（552,178 B）；**52 支 SPIR-V 模块 = 300 KiB**；third-party dependencies **0**（链接参数只有 `-lvulkan-1 -lpdh -ldxgi`）。文档里任何「代码量 / 产物尺寸」的句子都应引用这个脚本，不要再手算 `wc -c`（§8.9） |
| `kv_compare.py` | KV 报告 §2 的保真比对：两次 dump 的 band A / band B、K 位相同门、逐位置 TSV、scale 合理性（`runs/kv_compare_v4.log`） |
| `pdh_probe.cpp` | PDH 计数器路径取证（star / pid / 裸路径 × 有/无前导反斜杠），KV 报告 §3.1 的证据来源 |

---

## 4. 构建与运行

### 4.1 构建

```bat
cd /d "runtime_vk"
build_mingw.bat
```

做四件事：① `python ..\tools\gen_dequant_ref_comp.py` 生成 oracle 着色器；② 全部 `.comp` → `.spv`（`glslang -V -O`，**52 支**，`实测` `tools/source_size.py`）；③ `g++ -std=c++17 -O2 -m64 -Wall bonsai_vk.cpp -I%HDR% -o bonsai_vk.exe -L%VKLIB% -lvulkan-1`；④ 其余可执行。**STAGE10 之后第 ④ 步是 7 个，总共 8 个 exe**（`实测` `runs/build_v40.log:109-116`，每个后面跟一行 `OK`，失败分支是 `exit /b 1`）：
`align_tokenizer / align_dequant / align_f16 / align_attn / shared_probe / align_27b / bonsai_fleet`。

**`-O2` 这件事现在有了新含义**：本项目已经实测过一次「`-O0` 过、`-O2` 段错误」，最后查明是悬垂的 `VkBufferCreateInfo::pQueueFamilyIndices`（§15.4）。
**从今天起，`-O0` 与 `-O2` 的行为分歧在本仓库是「有未定义行为」的证据，不是「优化器奇怪」**——先查数组/指针的生命周期，再考虑别的假设。

注意 `-Wall` 是故意开的（本文件里两个 most vexing parse 警告最后都是真 Bug）。构建脚本末尾还会顺手跑一次 `bonsai_vk.exe 12288 4096 4 100` 的内核基准，那不是回归测试，只是自检。

### 4.2 运行引擎

```
bonsai_vk.exe run [gguf] [chunk_MiB] [n_tokens] [-- "prompt"]
默认 gguf   = models/Ternary-Bonsai-2-9B-PTQ1_0.gguf
默认 chunk  = 32 MiB（权重上传分块）
默认 tokens = 24
默认 prompt = "用一句话解释三值模型"（UTF-8 以 `\xNN` 转义硬编码在 bonsai_forward.h:2951）
```

参数解析在 `fwd_main` :2947-2953（**行号 2026-11-05 在当前源码重测**；STAGE9 的 :1907-1913 是旧位置，GQA 合并 + fleet 进来后整体下移）。
**prompt 默认会被包进 chat template**（`f.chat = 1`，:201），
所以 6 字的默认 prompt 实测渲染成 **16 个 token**（`runs/ignition_v3.log:94` 打了完整 id 序列，
首尾就是 248045 / 248068 那对 control id）。`BONSAI_CHAT=0` 才回到裸 prompt——**裸 prompt 会让模型自己编任务**，
那是 §5.6 记录的行为，不是 Bug。

### 4.3 全部开关（`grep getenv` 实测清点）

前向 / 生成（**2026-11-05 在 `runtime_vk/bonsai_forward.h` 的当前源码上重测**：
`grep -o 'getenv("BONSAI_[A-Z_0-9]*")' | sort -u | wc -l` = **45 个变量**；STAGE9 记的是 24 个。
下表行号 = `grep -n getenv` 的当下读数）：

| 变量 | 作用 | 定义处（当前行号，实测） |
|---|---|---|
| `BONSAI_PROMPT` | 覆盖 prompt 字符串 | bonsai_forward.h:2953 |
| `BONSAI_STOP` | 图在第 N 个 block 切断（对齐用） | :2974 |
| `BONSAI_GDN_MODE` | 0=先衰减后 predict，1=反 | :2975 |
| `BONSAI_GDN_GMAP` | v-head h 用 q/k head `h%16` 还是 `h/2` | :2976 |
| `BONSAI_L2_MODE` | 0=`x/(‖x‖+eps)`，1=`x*rsqrt(ss+eps)` | :2977 |
| `BONSAI_CAP` | KV 槽位数（上下文上限）。**必须是偶数**（P16 成对打包），:3102-3105 直接拒绝奇数/0 | :2978，默认值 :225 |
| `BONSAI_SPLIT` | 强制 GEMV 的 split-K 数（累加顺序判别，§7.3 的证据 A） | :2979 → `g_split_force` |
| **`BONSAI_PROFILE`** | **STAGE9 新增**：启动那份一直是死代码的时间拆解仪表（§7.2） | :2982 → `g_prof`（声明已随行号漂移，用 `grep -n g_prof` 现取） |
| **`BONSAI_CHAT`** | 是否套 chat template，**默认 1** | :2983，默认值 :201 |
| **`BONSAI_THINKING`** | 模板的 `enable_thinking`，默认 1（与模板「未定义即为真」一致） | :2984 |
| **`BONSAI_QCHECK`** | QK 内核审计：A（窄↔宽）/ **A2（宽↔tiled，KV 报告 §9.3）** / **A3（GQA 合并 QK↔它的参考内核）/ A4（GQA 合并 PV↔frozen `attn_pv_q8v`）** / B / C / D / E；不过门就 **拒绝生成 rc 6** | :2985 |
| **`BONSAI_QKERNEL`** | 0=自动、1=强制窄、2=强制宽、**3=强制 tiled**，`>3` 拒绝（:2986 之后）—— 交叉验证用的就是 1/2/3 | :2986，消费处 `qk_select()` :293-296 |
| **`BONSAI_QK_TILE`** | tiled 内核的 tile 宽 = 唯一决定 LDS 足迹（`4*QTILE+4`）：1024 / 2048（默认）/ 4096 / **8192 = 越限探针**（32772 B > 本卡实测 32768 B，auto 路径到不了）；其它值 `return 4` | :2995 |
| **`BONSAI_GQA_OPT`** | **STAGE10 新增**：GQA 4× 冗余合并开关。`0` = OFF（位精确对照路径 = 阶梯量过的那批内核）、`1` = **AUTO = 默认**、`2` = FORCE（每个 cap 都走合并内核，包括 cap ≤ 256 的窄版地盘，同二进制 A/B 就靠它）。`>2` 拒绝 | :3022，默认值 :156 |
| **`BONSAI_GQA_PART`** | **诊断量，不是配置**：位 0 = 合并 QK，位 1 = 合并 PV，所以一个测出来的收益能被归因到其中一半而不是同时宣称两半。默认 `1`——**PV 那一半不是恒负的**：长上下文实测是负的（4 K −11 %、32 K −25 %），cap 256 实测是正的（+2.5 %）；关它因为开关的用途是长上下文（§15.3） | :3031，默认值 :156 |
| **`BONSAI_GQA_TILE`** | 合并内核的 wrapper = 它的 LDS：`HPK*4*QTILE + 8*HPK` B → 256 → 4128、**512 → 8224（默认）**、1024 → 16416；其它值拒绝 | :3039，默认值 :156 |
| **`BONSAI_KVFILL`** | `N`（2 ≤ N ≤ cap）打开**合成 KV 计时夹具**：只有 ms/token / tok/s / 内核组合可信，ids 与任何保真陈述无效（KV 报告 §9.4(b)） | :3049 |
| **`BONSAI_STOP_NL`** | 生成到第 N 个换行停 | :3050 |
| **`BONSAI_STRIP_THINK`** | 只打 think 段之后的正文 | :3051 |
| **`BONSAI_SYSTEM`** | 给模板一条 system 消息（空 = 不渲染 system 段） | :3052 |
| **`BONSAI_KV_V_QUANT`** | V 的表示：`q8_0`（默认，块 32）/ `none`(=fp16) 回归对照 | :3055 |
| **`BONSAI_KV_DEBUG_STREAM`** | fp32 KF 调试流，默认关（唯一的 cap 相关非 KV 消费者，KV 报告 §1） | :3061 |
| **`BONSAI_ROPE_ORACLE`** | oracle 分支：把管线钉回冻结的窄集合（与 GQA 合并互斥，见 :1956-1957 的 `want_gqa_qk/pv` 条件：合并内核只在 `qcheck` 或合并位打开时才建） | :3064 |
| **`BONSAI_CAP_AUTOFIT`** | `1` 时 OOM 兜底**自己按公式缩 cap 并重新 init**（STAGE9 时这是「未做」项，现已落地，KV 报告 §1/§6） | :3073，默认值 :226 |
| **`BONSAI_KV_EVICT`** / **`BONSAI_KV_BLOCK_SLOTS`** | spill 的驱逐动作 / 块粒度（默认 64 slot） | :3076 / :3077 |
| **`BONSAI_SPILL_EVERY` / `_SOFT_MIB` / `_HARD_MIB` / `_MONO` / `_ABS`** | PDH Shared Usage 采样节奏与阈值（阈值判**增长量**，`_ABS=1` 复现旧的绝对读法） | :3078-3082 |
| **`BONSAI_CB_REUSE`** | 前向阶段复用一条命令缓冲，**默认 1**；`0` 恢复旧的每 group 新建（KV 报告 §3.2 的病灶） | :2968 |
| **`BONSAI_CB_PROBE`** | `=N` 测 N 条空缓冲的驱动记账成本 | :3197 |
| **`BONSAI_CHAT_DUMP`** | `"<user_file> <ids_file>"`：把渲染前原文与渲染后 id 序列各写一份，**给步骤 D4 的外部比对用** | :2585 |
| **`BONSAI_HEAP_LIMIT_MIB`** | **测试旋钮，不是产品开关**：把预算表里的 heap 人为夹到 N MiB，用来在 4 GB 真卡上**真的走进** OOM 分支（§5.9）。它只夹预算判断，不改驱动看到的任何东西 | :1713 |
| `BONSAI_TEMP` / `BONSAI_TOPK` / `BONSAI_TOPP` | 采样器 | :2554-2556 |
| `BONSAI_KVCHECK` / `_POS` / `_LAYER` | KV 落盘 dump（供 `tools/kv_compare.py`） | :2655 / :2657 / :2658 |
| `BONSAI_ALIGN` | block-0 逐 op 锚点比对，不生成（`BONSAI_ALIGN_TOK` 改 token id：:3251、:3258） | :3244 |
| `BONSAI_LADDER` | 每 block 一次 submit 的残差阶梯 | :3254 |
| `BONSAI_TAIL` | 全网络尾段 vs 同一份 dump（prompt "ab"，冷缓存） | :2676 |

cap 的拒绝逻辑有两层，**都在任何 dispatch 之前**：奇数/0 → :3102-3105；超过「当前配置所允许的最宽行」→ :3131（`qk_row_limit()` :302-303：走 tiled/merged 时护栏是算术值 `kMaxTile = 65536`，:197；只走 wide 时仍是 `kMaxWide = 2048`，:191。`qk_select(f)` 在 :3130，**先选内核再夹 cap**，顺序是有注释的（:3105-3109：HPK 必须在夹 cap 之前算出来，否则会拒绝掉合并内核服务得了的行，实测 `:1732/:1755/:1767/:2613` 四处都用同一个 `qk_row_limit()`）。**注意措辞**：那条打印说的是「widest VALIDATED row」，tiled 之后**挡路的是显存预算而不是 shared memory**，见 KV 报告 §4.1/§9.2）。选到的内核打在 :3151（`attention QK kernel:`）与 :3162（`GQA merge:`），实测在 `runs/short_c256_on3_v1.log:32-33`。

多卡驱动台（**`bonsai_fleet.exe` 自带的环境面，STAGE10**）：`BONSAI_GPU_INDEX=<i>`（按物理索引选卡，读不到就 REFUSE 并打印 `bonsai_fleet.cpp:106`）、
`BONSAI_GPU_TYPE=integrated|discrete|zero-copy`（选卡类型过滤，默认 `compute`，:97）、`BONSAI_MODEL=<path>`（`split`/`probe` 读 GGUF 张量表用）、
`BONSAI_API12=1`、`VK_VALIDATION=1`。usage 文本在 `bonsai_fleet.cpp:1073-1083`（实测），子命令分发在 :1086-1089。

内核基准 / 探针（bonsai_vk.cpp，**行号未漂移**，该文件仍是 1906 行）：`BONSAI_DIAG`(607,802) `BONSAI_PARTIAL`(199) `BONSAI_RANDSCALE`(385) `BONSAI_XFER`(896) `BONSAI_SYNC`(1052) `BONSAI_OVERLAP`(1238) `BONSAI_PIPE`(1416) `BONSAI_TXBARRIER`(1869) `BONSAI_API12`(208)；tokenizer：`BONSAI_TOK_INIT=cp`（复现输掉的码点起手模式，`bonsai_tok.h:327`）。

### 4.4 跑基准的规矩

1. 先 `powershell -NoProfile -ExecutionPolicy Bypass -File ../tools/clear_field.ps1` 清场（16 GB 机器 + 独显，后台进程会偷 PCIe 带宽）。
   **`-ExecutionPolicy Bypass` 不能省**：这台机器的策略是 Restricted，不带它就是「脚本没跑成」而不是「环境干净」，
   而一次被别的进程干扰的基准比一次失败的基准危险得多。
2. **同一时刻只允许一个上卡的进程**。基准跑的时候不要再开第二次 `bonsai_vk.exe`，也不要边跑边 `align_*`。
3. 结果落 `runs/<名字>_v<N>.log`，**不要覆盖旧日志**——上一版和这一版的差就是回归证据。
   （这条本仓库**违约过一次**：`runs/ignition_v3.log` 里原来那份「同二进制复跑 13.77 tok/s、48/48 id 与 v2 逐位相同」
   的日志被 STAGE9 的新 run 覆盖了。旧内容只剩 §6.1 抄下来的数。见 §9.1 的标注。）
4. `bonsai_vk.exe run ...` 的 stdout 要整份留档，日志里的 `region table` 是诊断显存重叠的唯一手段。

### 4.5 已验证的复现序列（照抄即可，不要跳）

```bat
REM 0. 清场（必须带 ExecutionPolicy）
powershell -NoProfile -ExecutionPolicy Bypass -File ..\tools\clear_field.ps1

REM A. 回归：构建 + **五个** align_*（STAGE10 实测通过的那一份 = runs/regression_v14.log，
REM    它比 STAGE9 的 regression_v13.log 多一个 align_27b.exe，并且 QCHECK 门里多了 A2/A3/A4）
cd /d "runtime_vk" && build_mingw.bat
REM    tokenizer 对齐是**两个探针文件**，32 + 7 = 39。少跑第二个就不是 39/39。
set G=models/Ternary-Bonsai-2-9B-PTQ1_0.gguf
align_tokenizer.exe %G% ..\runs\tok_probes.txt     ..\runs\tok_ref_32.jsonl   REM 期望 32/32
align_tokenizer.exe %G% ..\runs\tok_probes_kr.txt  ..\runs\tok_ref_kr.jsonl   REM 期望 7/7
align_dequant.exe    REM 期望 max abs 0.000e+00。regression_v13 跑的是 100 块 / 12800 权重，
                     REM   regression_v14 跑的是 **1000 块 / 128000 权重**（日志第 8 行自己标了），
                     REM   两份都是 0.000e+00、bit-exact 128000/128000；引用时别说错块数。
align_f16.exe        REM 期望 0 wrong + bit for bit YES
align_attn.exe       REM 默认 seq=8 cap=256；期望 score=6.695e-04 (=6.40e-06 of |kq|max)，context 0.000e+00
                     REM   regression_v14 用的是显式 seq=64（`align_attn 64 256`）：score=3.567e-03、
                     REM   context=4.104e-04、ratio our/format = 0.262 —— 参数不同不是回归（§6.2）
align_27b.exe        REM STAGE10 新增，期望 ALL CHECKS PASSED (0 failure(s)) + 四条负向控制都拒绝

REM B. 对齐 + 阶梯 + 点火（输出名字按 §4.4 的规则自己递增 _vN，别写已存在的文件名）
set BONSAI_ALIGN=1   && bonsai_vk.exe run > ..\runs\fwd_align_v<N>.log 2>&1    REM 期望 4/32（§7.3：这 4 条不修；参照 fwd_align_v11.log）
set BONSAI_ALIGN=
set BONSAI_LADDER=1  && bonsai_vk.exe run > ..\runs\ladder_v<N>.log 2>&1        REM 参照 ladder_v6_rho.log
set BONSAI_LADDER=
bonsai_vk.exe run %G% 32 48 > ..\runs\ignition_v<N>.log 2>&1                    REM 参照 ignition_v3.log（14.04 tok/s）。
                     REM   STAGE10 之后**默认配置**（GQA 合并开、PART=1、TILE=512）的同一命令实测
                     REM   14.16 tok/s / 稳态 70.6 ms（runs/regression_v14.log:95-96），
                     REM   另一份默认门 70.1 ms / 首 71.0 ms（runs/qcheck_gqa_default_v2.log:232）

REM C. QK 内核审计（STAGE9 的验收面 + STAGE10 的 A2/A3/A4；2 个生成 token 就够，审计在 prefill 之后立刻跑）
set BONSAI_QCHECK=1  && bonsai_vk.exe run %G% 32 2 > ..\runs\qcheck_256_v<N>.log 2>&1    REM 期望 A/A2/A3/A4/B/C 全 0 差
set BONSAI_CAP=2048  && bonsai_vk.exe run %G% 32 2 > ..\runs\qcheck_2048_v<N>.log 2>&1   REM 期望 B/C/D/E 全 ok
set BONSAI_QCHECK=
set BONSAI_CAP=
REM    **默认配置的那一份门日志已经有了**：runs/qcheck_gqa_default_v2.log:154-173，
REM    `gate: PASS (bitmask 0 = A ok A2 ok A3 ok A4 ok B ok C ok D ok E ok)`，
REM    同一次运行还打印 `GQA merge: grid 16 -> 4 ... LDS 8224 B ... QTILE = 512`（:70）
REM    和 `V path: frozen attn_pv_q8v.comp（BONSAI_GQA_PART bit1 off...）`（:71）——
REM    **默认 `GQA_PART=1` 只合 QK，PV 走 frozen**；A4 项仍然逐位 0 差，是「合并 PV == frozen PV」的证据。

REM D. 时间拆解（§7.2 的那一行开关）
set BONSAI_PROFILE=1 && bonsai_vk.exe run %G% 32 32 > ..\runs\profile_v<N>.log 2>&1
set BONSAI_PROFILE=

REM E. OOM 兜底的三条分支（§5.9；BONSAI_HEAP_LIMIT_MIB 是测试旋钮，只有 B/C 用）
bonsai_vk.exe run %G% 32 48 > ..\runs\oom_v<N>.log 2>&1                        REM A：装得下 -> 打预算表 + headroom，exit 0
set BONSAI_CAP=2048
set BONSAI_HEAP_LIMIT_MIB=2580
bonsai_vk.exe run %G% 32 48 >> ..\runs\oom_v<N>.log 2>&1                       REM B：KV 不够 -> 建议一个偶数 cap，exit 4
set BONSAI_CAP=256
set BONSAI_HEAP_LIMIT_MIB=2500
bonsai_vk.exe run %G% 32 48 >> ..\runs\oom_v<N>.log 2>&1                       REM C：任何 cap 都不够 -> 报最低门槛，exit 4
set BONSAI_CAP=
set BONSAI_HEAP_LIMIT_MIB=
```

`align_tokenizer.exe` 的命令行是**三个位置参数** `<model.gguf> <probes.txt> <tok_ref.jsonl>`（`align_tokenizer.cpp:86` 打 usage），
少给一个就是 `cannot open ...` + `exit 2`——**`exit 2` 不是测试失败，是没用对**，别看错。
判 OOM 分支的成败要看 stderr 文本 + 退出码，**管道到 `tail` 之后 `$?` 是 tail 的**（§8.8）。

---

## 5. 已实测的决定性事实（当公理用，别再重新发现一遍）

这一节是本仓库最贵的资产，每条都付过 debug 成本。

### 5.1 权重与量化

* 427 tensor = **202 个 PTQ1_0（三值）+ 225 个 f32**；总元素 8,953,803,264 ≈ 8.954 B。`实测`
* **每个 PTQ tensor 的 K ∈ {4096, 12288}，只有两种宽度** → 激活侧 Hadamard 只需覆盖两个宽度。`实测`（runs/ignition_v2.log 权重计划段）
* 检查点自带紧凑块形 1,957,036,032 B = 1.823 GiB（**1.75 bpw**）；我们的**随机访问 2-bit 形式** = codes 2.083 GiB + scales 0.260 GiB = 2.343 GiB（**2.125 bpw**）。**布局代价 0.521 GiB**，等于 17,064 个 token 的 KV（按 32768 B/token）。`实测/推导`
* PTQ1_0 反量化真因纠正：`block_ptq1_0.d` **必须是 float16_t 而不是 uint16_t**；写成 uint16_t 的误差是 10916.9。`实测`（FAITHFUL_REPRO.md）
* fp16 打包必须走 `pack2_rne()`（RNE），不能用 `packHalf2x16()`：本驱动那条指令是 **round-toward-zero**，而参考引擎的 fp16 缓存是 RNE 建的。`实测`（stage F / stage R）

### 5.2 网络结构

* 32 个 block：**24 GDN（线性递归）+ 8 softmax-attention**，`full_attention_interval = 4` → 层 3,7,…,31 是注意力。`实测`
* 递归状态与上下文无关：每层 2,097,152 B state + 98,304 B conv = **50.2 MiB / 24 层**。`实测`
* KV 只属于那 8 层：fp16 K+V 时 **32,768 B/token**（1k token = 32.0 MiB）；**V 换成 q8_0（块 32）后是 25,088 B/token = 24.50 KiB/token**（`实测` 引擎自己打的 `| q8_0 V: 25088 B/token`，`runs/qcheck_gqa_default_v2.log:20`；预算行 `24.50 KiB` 在 `runs/short_c256_off_v1.log:46`）。逐层拆解与两种模式的对照在 `KV_P0P1P2_MEASURED.md` 的每-token 表（4096 → 3136 B/层/slot）。`实测`
* 注意力形状：`n_head=16`、`n_head_kv=4`（GQA，`kv = h/(nh/nkv) = h/4`）、`head_dim=256`、`n_rot=64`。`实测`
* RoPE 是**半分裂**：pair `(p, p+32)`，`theta_p = base^(-2p/64)`，只转前 64 维。`实测`
* GDN 的 conv 历史是 `(conv_kernel-1) × attn_qkv 行数(8192) = 24576 floats/层`，**不是** `inner=4096`。当年按 4096 分配，第 n 层覆写了第 n+1 层的 taps，静默串扰。`实测`（事故记录在 bonsai_forward.h:1091-1098）

### 5.3 `attn_q.weight` 的 8192 输出是「每头交错的 512 宽组 [q 256 | gate 256]」，**不是前后两半**

STAGE8 头号事实。误读成后半的后果（全部 `实测`，逐行证据在 runs/ladder_v4.log）：block 3 崩 → logits 和翻到 **−866431** → 输出退化成纯换行。

修法：新增 `deint_qg.comp` 把它拆成两个连续的 4096，并新增 `QD` / `GD` 区域（bonsai_forward.h:1080、:354-357、:377-378）。**任何后续改动都要保持这个解释：`attn_pv` 的 gate 窗口就是 `R.GD`。**

定位它的工具是 **oracle ladder**：逐 block 比「分支和 / FFN 和 / 残差和」三件事对官方 `l_out-0..31`，第一个 FAIL 的 block 就是病灶。

### 5.4 注意力语义（三个 GPU 内核的合同）

* `kq = K^T·Q` **不带 scale**；scale 在 softmax 里，且精确等于 `1/sqrt(head_dim) = 0.0625`。`实测`（tools/attn_softmax_check.py 自由二分拟合 0.062501；错误候选 1/sqrt(128)、1/sqrt(64)、0.05、0.08、1.0 分别差 5.8e-2…5.2e-1）
* mask 是**加性**、跨头广播、死槽位 −INF。官方引擎打印的每个被因果 mask 掉的概率都是 0.0000，只有 `exp(-inf-max)=0` 能复现 → **被 mask 的 lane 必须产出精确 0，不是「很小」**。`实测`
* softmax 先减去「已 scale 的活分数」的最大值再 exp。`实测`
* 我们的 `MASK` 区域**全是 0、主机从不写**：因果性完全由 `POS[t]` 推出的 `nfill = pos+1` 决定（prefill 是「一 token 一次 decode」的循环，`seq` 恒为 1）。**所以扩上下文不需要动 mask 语义。**`实测`
* `round_probs = 1`：PV 读 fp16 概率操作数（ggml 的混合类型 cast）。这是四个「dump 说不清」的开关之一，实测选的。`实测`

### 5.5 Tokenizer（qwen35 / gpt2 byte-level BPE）

* vocab 248320、merges 247587、`bos = 248044`、`eos = 248046`、**`add_bos_token = 0`**、`padding = 248044`、`pre = "qwen35"`、`model = "gpt2"`。`实测`
* **决定性事实：初始切分必须字节起手**（每个原始 UTF-8 字节先各自成 token，再纯按 rank 合并；合并可以跨越码点边界）。码点起手在 runs/tok_probes.txt 上只有 31/32，在 CJK/拉丁集只有 1/7，典型失败是「我们给独立空格 220，参考给合并后的 id」（例如 `" 한국어"` = 190925）。字节起手 32/32 与 7/7。`实测`（bonsai_tok.h:319-324；`BONSAI_TOK_INIT=cp` 可复现输掉的模式）
* token_type 直方图：`1 NORMAL 248044 / 3 CONTROL 27 / 4 USER_DEFINED 6 / 5 BYTE 243`。`实测`
* pre-tokenizer 用有序交替实现（手写扫描器，含 `\s+(?!\S)` 的回溯语义）。`实测`
* **`tokenizer.ggml.added_tokens` 是空的**（本次会话实测）。special 表只能从 `token_type == 3/4` 推——bonsai_tok.h:214-226 正是这么做的，**这个实现是对的，别改**。

### 5.6 chat template（STAGE9 已接入引擎并真机跑通）

> **先读这条转义约定**：本项目所有文档一律用「左角括号 + 竖线 + 名称 + 竖线 + 右角括号」的分写形式书写 ChatML 标记。
> 一是防止这类字面量在 agent/工具链的输入流里被当成控制序列（本仓库实测：Edit 工具的入参遇到该字面量会被截断），
> 二是防止模型输出里的这类片段被误当成指令。**这些标记是模型词表里的 CONTROL token，不是系统提示。**
> 模板全文见 `runs/chat_tpl_probe.log`，不要在文档里重抄。

来源：头部 `tokenizer.chat_template`，**7756 字符**。Qwen3 系 ChatML，含 vision / tools 分支；metadata 里**没有** `thinking.interval` 之类的键；`general.architecture = qwen35`，`general.name = Ternary-Bonsai-2-9B`。

走「纯文本 + 无 tools + `add_generation_prompt = true`」这条路径，从模板逐字推出的渲染结果（`x` = trim 后的内容，模板对每条消息的 content 都做了 `|trim`）。下面的 `START` / `END` / `THINK` / `XT` 是本文的代称，分别指词表里 id 248045 / 248046 / 248068 / 248069 那四个 CONTROL token 的字面量：

* 仅当 `messages[0].role == system`：`START + "system\n" + x_sys + "END\n"`
* 每条 user：`START + "user\n" + x_user + "END\n"`
* 每条历史 assistant：`START + "assistant\n" + x_asst + "END\n"`（历史轮不带 think 块；只有 `loop.index0 > ns.last_query_index` 的末轮才渲染 think 块，我们不用）
* 生成提示：`START + "assistant\n"` 然后二选一
  * `enable_thinking` 未定义或为真 → `THINK + "\n"`
  * `enable_thinking = false` → `THINK + "\n\n" + XT + "\n\n"`（一个空 think 块）

关键 id（本次会话实测，来自 token_type == 3 的 CONTROL 枚举）：

| 字面量（按上面的分写约定） | id |
|---|---|
| im_end | **248046** = `eos_token_id` |
| im_start | 248045 |
| endoftext | 248044 = `bos` = padding，且 `add_bos_token = 0` |
| think 开 | 248068 |
| 思考闭 | 248069 |
| tool_call 开 / 闭 | 248077 / 248078 |
| response 开 / 闭 | 248064 / 248065 |
| vision 一对 | 248053 / 248054 |

**已观测的行为（不是猜测）**：裸 prompt（不加模板）跑到第 13 个 token 时，模型**自己**吐出了 248068（think 开）并跟着换行，然后开始写英文的推理脚手架（`runs/ignition_v2.log` 的 gen 14–19）。也就是说加不加模板，模型都会进 think 模式；接上模板之后那段伪脚手架应当消失，而生成**必须以 EOS 停** —— 它的 id 是 248046，而 `sp.eos` 已经就是 248046（`bonsai_forward.h:1333`），**采样器不用改**。

**STAGE9 之后的状态**：模板已接进 `fwd_run`（`bonsai_chat.h` + `bonsai_forward.h:1659-1690`），`BONSAI_CHAT` 默认 1。
实测三条渲染路径都跑通并且**两条独立 id 路径互相 IDENTICAL**（`encode(渲染串)` vs `render_ids()`）：
`runs/chat_think_v1.log:90`（thinking on，30 字 → 16 token）、`chat_nothink_v1.log:90`（thinking off，48 字 → 24 token）、
`chat_en_v1.log:90`（英文 48 字 → 22 token）。裸 prompt 那次「自己编任务」的现象按预期消失，生成长段不再靠截断而靠 EOS/换行停
（`BONSAI_STOP_NL`）。硬前置 `ids_are_singles()`：若某个 control id 在词表里被拆成多枚就**拒绝渲染并 `return 6`**，
不给你一条「模型把分隔符当散文读」的静默错误。

**仍然没做完的（就这两条）**：
1. **多轮对话（历史 assistant 轮）没测过** —— `render()` 只走「单 system（可空）+ 单 user + 生成提示」这条路径，历史轮的渲染规则在 §5.6 上面写着，代码没实现。
2. **步骤 D4 的外部比对仍未跑**：我们组装的字符串过 `Bpe::encode` 的 id 序列，必须与官方 `llama-tokenize -chat-template` 逐 ID 相同。
   `BONSAI_CHAT_DUMP` 就是为这一步准备的（它把原文和 id 序列各写一份）。tokenizer 本身 39/39 对过 ≠ 模板渲染对过，别把这两件事混起来。

### 5.7 上下文长度限制的真实位置（STAGE9 到 2048 → STAGE10 到 32768，**墙从 shared memory 换成了显存预算**）

**STAGE8 时唯一的墙是 `shaders/attn_qk_softmax.comp`**：它把 key 位置 j 直接映射到 lane（`local_size_x = 256` + `shared float sd[256], sp[256]`）。
STAGE9 的解法**不是改那个内核**（它是 0-error 合同，§3.3），而是**新增** `attn_qk_softmax_wide.comp`：lane 跨步覆盖 + shared 数组按 `MAXCAP` 参数化，
主机侧 `attn_block`（`bonsai_forward.h:551`）按 `cap <= 256 ? 窄 : 宽` 二选一，`BONSAI_QKERNEL=1/2` 可以强制任一边走同一份输入做交叉验证；
STAGE10 又加了 tiled（`QKERNEL=3`）与 GQA 合并（`GQA_OPT`），三者由 `qk_select()`（:293-296）+ `qk_row_limit()`（:302-303）统一选。

**STAGE9 时墙确实是 shared memory**（下面这段保留，因为它记录了当时为什么只到 2048）：
`maxComputeSharedMemorySize = 32768 B`（本驱动实测，`runs/limits_probe.log:3`），
宽内核的两个 `float[MAXCAP]` 是 `8*MAXCAP` 字节，`MAXCAP = 2048` → 16 KiB（一半余量）；4096 → 正好 32768 B，不留驱动余量，所以不取。

**STAGE10 之后这句话过时了**：分块内核 `attn_qk_softmax_tile_q*.comp` 把 LDS 换成 `4*QTILE + 4`（QTILE 2048 → 8196 B = 8.0 KiB，`实测` `runs/c32768_on1_v1.log:35`），
行宽不再进 LDS，所以**「要上 4096 就得改内核结构」这一步已经做完了**（tile + GQA 合并两条路都做完了，见 §15.3）。
`cap > kMaxWide` 现在不再是拒绝条件——只要 `qk_select` 选了 tiled/merged，行上限就变成算术护栏 `kMaxTile = 65536`（`bonsai_forward.h:197`、`qk_row_limit()` :302-303）。
**现在挡路的是显存预算**，`实测`：cap 32768 + `BONSAI_GQA_TILE=1024`（LDS 16416 B 那档）在 `vkAllocateMemory` 上被拒——
`act arena did not fit: the driver refused 841.3 MiB although the heap ... still leaves 531.0 MiB free after the weights`（`runs/gqa_on_c32768_q1024_v1.log:59`，退出码 4），
而同一个 cap 在 **QTILE 512（LDS 8224 B）下跑通了**（`runs/gqa_on_c32768_q512_v1.log:179` = 稳态 461.9 ms/token；批次 H 的 32K 三点 `runs/gqa_pair_v1_summary.txt:13-15`）。
→ **32K 现在能跑，代价是 1.84–2.67 tok/s**，而且引用它之前要先读 KV 报告 §9.4(b)：那批是 `BONSAI_KVFILL` 的**合成 KV 计时点**，只有 ms/token 可信，token 流与保真陈述无效。
`TILE=1024` 被拒不是因为 LDS 超了（16416 < 32768），而是因为**这一档的 arena 请求落在了驱动自己保留的那段里**——同一条日志说的是 `vkAllocateMemory` 失败，不是 `maxComputeSharedMemorySize`。

| 位置 | 与 cap 的关系 |
|---|---|
| `attn_norm_rope.comp` | 无关。arena 按 `position` 索引（:117），`cap` 只当 stride 用 |
| `attn_pv.comp` | 无关。每 lane 自己升序循环 `j < nfill`（:42） |
| `MASK` / `POS` 区域 | 长度取 `cap`，且**从不主机写入**（§5.4） |
| KV arena `R.kv_layer` | **每层 784 floats/slot** = K fp16 512 + V q8_0 数据 256 + V scale 16（`实测` `runs/short_c256_off_v1.log:44` 的 `KV 8 layers x 784 = 6314 floats/slot`）；`BONSAI_KV_V_QUANT=none` 回到 fp16 V = 每层 1024 floats |
| `SCORE` / `PROB` / `P16` | `nh*cap` / `nh*cap` / `nh*cap/2`（合起来 42 floats/slot，`实测` 同一行） |
| 硬编码 256 的绑定宽度 | `attn_norm_rope` 与 `attn_pv` 是 `nh*hd = 4096`，**与 cap 无关，别误改**；GQA 合并版改的是**派发数**（一个 workgroup 服务 `HPK = 4` 个 query 头，grid 16 → 4），绑定宽度仍按头计，所以 A3/A4 门能用同一份 4096 元素缓冲区逐位比（`实测` `runs/qcheck_gqa_default_v2.log:163,166` 的 `SCORE 0/4096 bit-diff`） |
| 主机 guard | `:3102-3105` 奇数/0 拒绝；`:3130` 先 `qk_select()`、`:3131` 再按 `qk_row_limit()` 拒绝（tiled/merged → `kMaxTile = 65536`，wide-only → `kMaxWide = 2048`）；生成循环 `pos >= cap` 停 |
| **`p16` 打包循环** | **STAGE9 踩过的最深的坑**：窄内核 256 条 lane 一次只覆盖 `cap/2 = 512` 对，写成 `if (j < cap/2)` 就足够；宽内核必须改成跨步 `for (w2 = j; w2 < cap/2; w2 += NTILE)`（`attn_qk_softmax_wide.comp:106`）。**漏了它的表现是「超出 512 的整段上下文用上一 token 的残留」——不 NaN、不报错、只是输出慢慢漂移**，所以验收必须显式覆盖这条：`BONSAI_QCHECK` 的 B 项（`P16 == pack2(PROB)` 全 16384 对）就是为它写的 |

### 5.8 显存、带宽与 4GB 墙（全部实测）

| 项 | 值 |
|---|---|
| 权重 arena（DEVICE_LOCAL 单块） | committed **2.410 GiB**（codes 2,236,612,608 B + scales 279,576,576 B + f32blob 29,411,328 B + ptq_scratch 33,554,432 B），对齐 padding **0 B** |
| 激活 arena | `cap` 的函数，**逐项见 §5.9 的 `per_slot` 拆解**。当前默认（V q8_0）实测两档：cap 256 → **58.2 MiB**（`runs/short_c256_off_v1.log:46`），cap 4096 → **150.7 MiB**（`runs/c4096_on1_v1.log:49`）。STAGE9 的 fp16 K/V 那版是 cap 256 → 61.052 MiB、cap 2048 → 124.3 MiB（`runs/oom_fallback_v3.log:6,35`），**换 V 表示后每槽位从 36.16 KiB 降到 24.66 KiB**（§5.9） |
| 常驻总量 | 2.410 GiB + 激活，deviceLocal heap **3840 MiB**。当前默认实测 headroom：cap 256 → **1314.1 MiB**（`runs/short_c256_off_v1.log:47`，= 54558 个空槽位）、cap 4096 → **1221.6 MiB**（`runs/c4096_on1_v1.log:50`）；fp16 K/V 时代是 cap 2048 → 1247.9 MiB、cap 256 → 1311.2 MiB（`runs/oom_fallback_v3.log:7,48`） |
| **约束（重要）** | **至少 78.3% 的权重必须常驻显存**，DeepInfer 式流式换入换出在这张卡上恒亏（装载成本 > 计算收益，STAGE4 结论 D「不可行-装载恒赢」） |
| SDMA | **7.12 GB/s**，且与计算并发**不打折**；换队列做 D2H 会**翻倍**（STAGE4）。STAGE10 的 `probe` 把这组数**逐条重打**了一遍并写进判定文案：计算队列 vs 传输队列 = **D2H 7.15 → 3.89 GB/s，H2D 7.12 → 4.93 GB/s**（`实测` `runs/fleet_probe_v2.log:28`） |
| 单条 D2H（STAGE10 环测，**含 submit + fence + 队列尾**） | **8 MiB / 1204.6 us = 6.96 GB/s 墙钟 = 实测 7.15 GB/s 上限的 97 %**；8 条一批一次 submit = **9410.9 us / 8 = 1176.4 us/条 = 7.13 GB/s**（`实测` `runs/fleet_ring_v4.log:95,97`） |
| ~~批量 D2H 56.67 GB/s~~ | **已作废**（STAGE10）。那是 created-SIGNALED fence 提前返回测出来的「带宽」：fence 在拷贝退休前就 signal 了，148.0 us 里装不下 8 MiB。对照实验（同一批字节、flags=0 的 fence）落在 **1176 us/条 = 7.13 GB/s**。机制与撤回记录见 §15.4 |
| ~~「一跳是 SYNC-dominated」~~ | **已作废**（STAGE10）。`fleet_ring_v1/v2.log:19` 那句是从「fence 列 ≈ 2 us 而墙钟 3.5 ms」读出来的，而那个 fence 根本没等；`fleet_ring_v4.log:88` 用修正后的读数改判为 **DMA-bound**（引擎时间 1178.18 us/腿 vs 环上墙钟 3171.3 us/hop，其中 D2H fence 2706.7 us） |
| 跨层预取 | **不可行**（STAGE2/4）；专家融合是必需的 |
| 每 token 权重侧流量 | 7.929 GMAC（不是 8.954 GB —— A 卡是 compute-bound 在这一段，不是 bandwidth-bound） |

### 5.9 显存预算的解析式（STAGE9 落进代码，是 OOM 兜底的根据）

引擎现在在**任何 dispatch 之前**把这张表打出来（`per_slot` 在 `bonsai_forward.h:1693`，预算表 :1720-1732，缩容建议 :1753-1766，`BONSAI_CAP_AUTOFIT=1` 的自动 re-init :1778；STAGE9 记的 :1154-1190 是旧行号），并且装不下就干净退出。
**当前默认（V q8_0）的实测两档**（`实测` 逐字来自 `runs/short_c256_off_v1.log:45-47` 与 `runs/c4096_on1_v1.log:48-50`）：

```
memory budget | device-local heap 3840.0 MiB, weights 2467.7 MiB already committed
  activations at cap 256  :  58.2 MiB = cap-independent 52.0 MiB + 256 slots x 24.66 KiB/slot
                            (q8_0-V + fp16-K arena 24.50 KiB + fp32 side 0.16 KiB)
  activations at cap 4096 : 150.7 MiB = cap-independent 52.0 MiB + 4096 slots x 24.66 KiB/slot
  headroom after this allocation: 1221.6 MiB (= 50718 more slots of cap, but the widest
                                       VALIDATED row for the tiled QK kernel is 65536)
```

`per_slot` 不是拍的，是引擎自己从区域表算出来的（`nh=16, nkv=4, hd=256, n_attn=8`，`实测`，`runs/short_c256_off_v1.log:44`）：

    P0 check | cap-proportional regions = SCORE+PROB+P16+MASK+POS (42 floats)
               + KF off (0) + KV 8 layers x 784 = 6314 floats/slot; GDN terms in it: 0
    784 floats/层 = 3136 B = K fp16 2048 B + V q8_0 数据 1024 B + V scale 64 B（块 32：32×int8 + 1×fp16）
    6314 floats x 4 B = 25256 B = 24.66 KiB/slot   ← 引擎打印值，与手算一致（`推导`）
    cap-independent = R.total - per_slot*cap = 52.0 MiB（state 48.0 + logits 0.95 + scratch/固定区）

**历史值（STAGE9 的 fp16 K/V + KF 调试流开着那一版，保留以免被读成矛盾）**：
`32 + 8 + 2 + 1024(KF) + 8192(KV fp16 K+V) = 9258 floats = 36.16 KiB/slot`，
它对应 `runs/oom_fallback_v3.log:6` 的 `2048 slots x 36.16 KiB/slot`。**同一份代码里 `BONSAI_KV_V_QUANT=none` 现在打 32.16 KiB/slot**
（`runs/short_ctx_none_v3.log:44`，= 42 floats 侧车 + fp16 K/V 32.00 KiB），`36.16` 是**再加上 KF**（`runs/short_none_kfon_v1.log:43` 明写 `, + KF`）的读数。
→ 三个数不是互相推翻，是**同一式子的三种开关组合**：q8_0 V 默认 24.66 / fp16 V 32.16 / fp16 V + KF 流 36.16。
KV 报告 `KV_P0P1P2_MEASURED.md` 的「每 token 显存：改前 / 改后」表逐层拆了这件事，以它为准。

三个由实测夹出来的常数，**当公理用**：

| 常数 | 值 | 来源 |
|---|---|---|
| 模型最低门槛 | **2519.8 MiB** = weights 2467.7 + cap-independent 52.0 + 对齐（`推导`，且被分支 C 的 `实测` 拒绝点夹住：2500 MiB 连 cap=2 都装不下） | `runs/oom_fallback_v2.log` C |
| 本卡 device-local heap | **3840 MiB**（`device_local_heap()` 从 `memoryHeaps[]` 读，别再手写） | `bonsai_weights.h:117` |
| 每槽位 | **24.66 KiB**（默认 V q8_0；上面那条算式）。对照：fp16 V = 32.16 KiB，fp16 V + KF 流 = 36.16 KiB | 与日志里 `24.66 KiB/slot` 逐项一致（`runs/short_c256_off_v1.log:46`、`runs/c4096_on1_v1.log:49`、`runs/qcheck_gqa_default_v2.log:87`） |

**三条兜底分支（都真的在硬件上走过）**：`runs/oom_fallback_v2.log` 是第一次，
**`runs/oom_fallback_v3.log` 是在 STAGE9 发布二进制（build_v15）上重跑的**，三条都过、退出码与文案一致：
① 装得下 → 打预算表 + headroom，继续跑，exit 0（**那一份**：cap 2048 headroom **1247.9 MiB**；cap 256 headroom **1311.2 MiB**。当前默认二进制上的同一行是 cap 256 → **1314.1 MiB**、cap 4096 → **1221.6 MiB**，实测 `runs/short_c256_off_v1.log:47`、`runs/c4096_on1_v1.log:50`）；
② **只有 KV 不够** → 算 `fit_cap = (heap - weights - cap-independent) / per_slot`（当前默认 per_slot = 24.66 KiB，代码在 `bonsai_forward.h:1753`），夹到偶数、夹到 `qk_row_limit()`（tiled/merged 时是 65536 而不是旧的 `kMaxWide`），
打印「rerun with BONSAI_CAP=1704」，**exit 4，一条命令都没提交**；
③ **任何 cap 都不够** → 明说「this model needs at least 2519.8 MiB」，**绝不建议 `BONSAI_CAP=0`**，exit 4，
并且给出三条出路（释放显存 / CPU 回退 / 换更小的 checkpoint）。
分支 ②/③ 用的是 `BONSAI_HEAP_LIMIT_MIB` 这个测试旋钮在 4 GB 真卡上人造出来的小堆——**这是本仓库唯一能「真的」测 OOM 路径的办法**，
所以它在 §4.3 的表里被标成测试旋钮而不是产品开关。

**STAGE10 补的第四条：预算表会说谎，驱动不会**（`实测`，当前二进制，cap 32768）：
预算表算出还剩 **531.0 MiB** headroom（`runs/gqa_on_c32768_q1024_v1.log:50`），
于是引擎按表申请 841.3 MiB 的激活 arena（:49，= 52.0 + 32768 × 24.66 KiB），
`vkAllocateMemory` **返回 -2 = `VK_ERROR_OUT_OF_DEVICE_MEMORY`**（:57），
随后那条诊断自己写明原因：**「the driver keeps a reservation inside that heap, so the real ceiling for this cap is LOWER than the budget table above」**（:59），
退出码 **4**（`runs/gqa_ladder_v1_summary.txt:10` 的 `rc=4`）。
→ 用 §5.9 的式子推 `cap_max` 时，**heap 读数要再夹一层驱动保留**：DXGI 的 LOCAL budget（同一次日志 :54：budget 3464.6 MiB）是比 3840.0 MiB 更安全的那个数，
KV 报告 §「重算的安全 / 极限上下文」表正是按这两条分别给的（安全 ≈ 39.2 K token，极限 ≈ 54.8 K token；此前这行写的 38.3 K / 53.5 K 是 MiB→KiB 误用 ×1000 导致的偏低 2.5 %，2026-11-05 已更正）。
**同一份日志的对照组很关键**：cap 32768 + `QTILE=512` 在 batch G 里**装得下并且跑完整条解码**（`runs/gqa_on_c32768_q512_v1.log:179` 有稳态值），
它申请的是**同样 841.3 MiB** 的 arena（cap 决定的，与 QTILE 无关）。也就是说这次 -2 **不是预算表的必然结果**，
而是「驱动在同一堆里的保留量会随进程状态变化」——所以这条不是「32K 一定 OOM」，而是**「预算表不能当唯一判据，`vkAllocateMemory` 失败要有干净的退出路径」**，而它有。

**吞吐随的是 live keys，不是 cap**（这条容易被读错，`实测`）：

| 配置 | ms/token | 出处 |
|---|---|---|
| cap 256（窄内核），16-token prompt | **71.2** 稳态，首 token 70.8 → **14.05 tok/s** | `runs/oom_fallback_v3.log` 分支 D（最终二进制） |
| cap 2048（宽内核），同一 16-token prompt | 72.6 稳态，首 71.9 → 13.78 tok/s（cap 本身只值 **+2.0 %**） | `runs/oom_fallback_v3.log` 分支 A（与 D 背靠背、同二进制） |
| cap 2048，**1579 live keys** | **94.2** 稳态，首 token 94.3 | `runs/qcheck_2048_v2.log:109` |

也就是说「cap 2048 慢 32%」是**误读**：把 cap 从 256 提到 2048 只值 2%（71.2 → 72.6），
多出来的是 1579 个**真实 key 的计算**，不是分配出来的槽位。
prefill 1579 token 实测 **131.99 s = 11.96 tok/s**（`qcheck_2048_v2.log:93`），因为 §8.1 那条结构性原因还没解。

---

## 6. 数值基准（回归的尺子：跑出来的数偏离这张表就是回归）

### 6.1 点火基线（`bonsai_vk.exe run ... 32 48`，cap=256）

| 指标 | 值 | 来源 |
|---|---|---|
| 生成速率 | **13.99 tok/s**（48 token / 3431.1 ms） | runs/ignition_v2.log |
| 稳态每 token | **71.5 ms**（v2）/ 72.6 ms（v3，同一二进制复跑） | 同上 / ignition_v3.log |
| 首 token（prefill+1） | 70.9 ms | 同上 |
| prefill | 450.4 ms / 6 个 prompt token = **13.32 tok/s** | 同上 |
| prompt tokens | `95826 110827 98682 95986 96326 103725`（默认 prompt「用一句话解释三值模型」） | 同上 |
| 权重装载 | 1150.4 ms；前向 bring-up 43.9 ms | 同上 |
| 每条 CB 的 dispatch / barrier | **1,490 / 1,489**（一 token 一次 submit，所以「每条 CB」=「每个 forward」）。日志印的是**累计**值：`ignition_v2.log:151` 的 80,460 / 80,406 是整个 run 的总数，除以 54 个 forward（6 prefill + 48 decode）正好整除得 1,490 / 1,489。交叉验证：`profile_v2.log:151` 83,440÷56 = 1,490.0，`qcheck_2048_v2.log:110` 2,355,690÷1,581 = 1,490.0。**照字面把 80,460 读成每条 CB 会高估 54 倍** | 同上三份 |
| 采样器默认 | `temp=0.00 top_k=0 top_p=1.00 eos=248046 cap=256` → **贪心** | 同上 |
| 开关组合 | `gdn_mode=0 gdn_gmap=0 l2_mode=1 round_probs=1 cap=256` | 同上末行 |
| logits 形状 | max 13.7415 / min −13.2801 / mean −3.047024 / **nan 0 inf 0** | 同上 |
| prompt 后 top-1 | `9616 (13.74)`，8 个候选都在 10.9–13.7 量级 | 同上 |
| **复现性** | ignition_v3 与 v2 **48/48 token id 逐位相同**（同二进制、同 prompt、间隔约 4 分钟） | v2 vs v3 |
| 对齐回归 | `BONSAI_ALIGN=1` → **4/32 锚点比较失败**（v10 与 v9 同一模式，说明 §5.3 的修复没引入新问题） | runs/fwd_align_v10.log |
| 阶梯 | 32/32 分支和 ≤5%，28/32 残差和 ≤1%，logits 头 3 位与官方逐位接近 | runs/ladder_v5.log |

官方 fork 同卡同模型基线：**4.417 t/s**（`runs/bench_ngl99_vs_cpu.log` / END_TO_END_MEASURED.md）→ STAGE8 3.16×（13.99），STAGE9 **3.18×**（`14.04/4.417 = 3.179`；最终二进制 14.05 = 3.181）。

**STAGE9 之后同一命令的实测（`runs/ignition_v3.log`，cap 256 + chat template 默认开）**：

| 指标 | 值 | 与 STAGE8 的差 |
|---|---|---|
| 生成速率 | **14.04 tok/s**（48 token / 3419.9 ms） | 13.99 → 14.04（+0.36%，同一内核，属噪声级） |
| 稳态 / 首 token | 71.2 ms / 71.0 ms（:153） | 71.5 / 70.9 |
| prompt token 数 | **16**（默认 prompt 过模板；模板渲染两条 id 路径 IDENTICAL，:93） | 6 → 16 |
| prefill | 1143.7 ms / 16 token = 13.99 tok/s（:96） | 6 token / 450.4 ms |
| 权重装载 | 1089.1 ms；bring-up 46.1 ms | 1150.4 / 43.9 |
| logits after prompt | max 23.4437 min −12.8155 mean −1.276158，**nan 0 inf 0** | 模板后 top-1 变成 `90700 "Thinking"`（think 段合法开场，不再是自编任务） |
| 对齐回归 | `BONSAI_ALIGN=1` 仍 **4/32**，同一模式（`runs/fwd_align_v11.log`） | 宽内核 + 模板 + OOM 兜底**没动那 4 条**——这是 §7.3 定性的交叉证据 |
| 最终二进制的四件套 | `runs/regression_v13.log`（全文只有 16 行，逐行就是这四条）：`32/32` + `7/7` byte-identical、`max abs 0.000e+00`（100 blocks）、f16「exact over the whole binary16 domain: YES (0 wrong numbers)」+「BOTH shader expressions … bit for bit: YES」、score `6.695e-04` | 全过 |
| **最终二进制（build_v15）的点火 + OOM 三分支** | `runs/oom_fallback_v3.log`：D 分支 **14.05 tok/s**（cap 256 窄内核）、A 分支 13.78（cap 2048 宽）、B/C 都 `exit 4` 且 `NOTHING WAS DISPATCHED` | 发布件就是这一份，验收引用它而不是 v2 |
| **STAGE10 默认配置（GQA 合并 AUTO + `PART=1` + `TILE=512`）的同一命令** | `runs/regression_v14.log:95-96`：48 token / **3389.5 ms = 14.16 tok/s**，稳态 **70.6 ms**、首 token **70.2 ms**；同一次 prefill **1154.5 ms / 16 token = 13.86 tok/s**（:80）。另一份默认门：`runs/qcheck_gqa_default_v2.log:232` 稳态 **70.1 ms** / 首 **71.0 ms** | 14.05 → 14.16 只有 **+0.8 %**，与 batch H 内部**同配置复测的漂移**（`off` 73.4 → 72.3 ms = 1.5 %，`runs/gqa_pair_v1_summary.txt:1,16`）同量级 —— **别把它当提升引用**，要引用收益请用配对数（§15.3） |

### 6.2 忠实复现基线（改动前后都必须保持）

| 测试 | 必须的结果 |
|---|---|
| `align_tokenizer.exe <gguf> <probes> <ref>` | **32/32（`tok_probes.txt`）+ 7/7（`tok_probes_kr.txt`）= 39/39** 逐 ID 与官方 `llama-tokenize` 1:1。**只跑第一个文件得到的是 32/32，不是回归失败** |
| `align_dequant.exe` | `max\|err\| = 0.00000`（100 块 12800 权重，扩测 1000 块 128000 权重同样 0） |
| `align_f16.exe` | 合成边缘块 491520 权重逐位 0 差（含 0 / 次正规 / 负 scale）；65536 个数 0 wrong |
| `align_attn.exe [seq] [cap]` | 默认 `seq=8 cap=256` → score 最大误差 **6.695e-04**（= 6.40e-06 of \|kq\|max）、context 0.000e+00。`seq=32/64` 是**显式传参**跑的，历史值 2.94e-03 / 3.57e-03 记在 `ATTENTION_ALIGN_FINAL.md`，默认回归命令不会复现它们 |
| **`BONSAI_QCHECK=1`（STAGE9 新增的回归面）** | `cap<=256`：窄/宽内核 SCORE、PROB、**P16** 三项 **0 bit-diff**（逐位相同，不是「接近」）；死槽 bitwise +0.0 零违规。`cap>256`：`P16 == pack2(PROB)` 全对（含 512 以上的跨步段）、死槽零违规、行和 \|sum−1\| 有界、双精度重算点位差 0.000e+00 |
| 引擎 | 重编 `exit 0` 且 `-Wall` **0 warning**；复跑逐字可复现 |

**这张表是「必须的结果」，不是「最终二进制都重跑过」**。build_v15 那一次真正重跑的是 `regression_v13.log` 里的
四条 + `oom_fallback_v3.log` 的点火与 OOM 三分支。表里另外三个更大的数各自有出处，引用时要说清是哪一次：
12800 权重的逐块比对在 `regression_v12.log`（13:57，同一代码状态减两条注释；`regression_v13.log` 只截了汇总行，
`[gpu] nel=12800` 那两行没进日志）；扩测 1000 块 / 128000 权重与合成边缘块 491520 权重 0 差在 `runs/dq_synth.txt`
与 `runs/reg_dq1000.txt`（**2026-11-01** 那批）。dequant 与 f16 路径自那之后一行没改过，但这不等于在新二进制上验证过。

---

## 7. 四项任务：**全部结案**（下面写的是方案、证据与仍然开着的口子）

这一节原来是「四项未完成任务」。STAGE9 把四项都做完了，所以每一小节现在的最上面是一条**结案状态**，
后面那些「步骤 A/B/C」「接下来的三步」是**当时的方案原文**，保留是因为它们记录了为什么这么做；
**要动手前先读结案状态和 §10。**

### 7.1 任务一：接 chat template + 解除 256 上下文限制（`#24`）—— **结案：两项都做完并真机验证**

> **结案（2026-11-04，实测）**：
> * chat template 已接入（§5.6）：`bonsai_chat.h` + `bonsai_forward.h:1659-1690`，`BONSAI_CHAT` 默认 1，三条渲染路径的
>   两条 id 路径互相 IDENTICAL（`runs/chat_think_v1.log:90` / `chat_nothink_v1.log:90` / `chat_en_v1.log:90`）。
>   最长一次生成 **192 token / 13.73 tok/s**（`chat_think_v1.log:296`）——「跑完整对话」的实测证据。
> * 256 的墙已解除到 **cap = 2048**（§5.7）：新增宽内核，**窄内核一行没改**；
>   `cap=256` 窄 vs 宽 **SCORE/PROB/P16 各 0/4096 bit-diff**（逐位相同，`runs/qcheck_256_v3.log:96-101`），
>   `cap=2048` / 1579 live keys 五项审计全过（`runs/qcheck_2048_v2.log:95-101`）。
>   §7.1 原文里标为「最容易漏」的 **p16 跨步打包**确实是个真坑，宽内核 :106 已实现，B 项审计就是它的看门狗。
> * **仍然开着的两条**：① 步骤 D4（与官方 `llama-tokenize -chat-template` 逐 ID 比对）没跑；② 多轮对话（历史 assistant 轮）没实现。
>   步骤 A 的答案：`maxComputeSharedMemorySize = 32768 B`（实测，`runs/limits_probe.log:3`）——所以 `MAXCAP` 取 2048。

**当时写的已完成部分（本次会话）**：§5.6 的全部取证、§5.7 的容量限制定位、`BONSAI_CAP=300` 实跑确认拒绝点在
`bonsai_forward.h:1520`（**环境变量本身是生效的**，它读于 :1518、早于 :1521 的 `fwd_init`；只有那条 guard 挡着）。
小坑：`... | tail` 之后拿到的 EXIT 是 tail 的，程序返回 4 看不见——判成败看 stderr 文本，或读 `${PIPESTATUS[0]}`。

**步骤 A（先测再改，10 分钟）**：在 `bonsai_weights.h:74` 那行 limits 打印里补
`maxComputeSharedMemorySize`，跑一次拿到本卡实测值。**不要拿官方 fork 报的 32768 B 当公理**（§2.1）。
这一步决定 `MAXCAP` 能取 1024 还是 2048。

**步骤 B：新增 `shaders/attn_qk_softmax_wide.comp`（不改那三个 0-error 内核，遵守 :334-336 的项目规矩）**

改动只有两处形状：

1. `#define NTILE 256u`、`#define MAXCAP <A 步测出的上限允许值>`，shared 数组变 `sd[MAXCAP]` / `sp[MAXCAP]`；
2. lane 不再等于 key 位置，而是**跨步覆盖**：

    for (uint j2 = j; j2 < nfill; j2 += NTILE)   // 每 lane 负责 j, j+256, j+512, ...

   每个 key 的点积表达式**必须逐字照抄窄内核**（`acc = acc + unpack(q).x * unpack(k).x;` 再 `+ .y * .y`，
   升序 d）—— 这是位精确的来源，改写法就等于改数值（§8.3 FMA 教训）。
   `score[row + j]` 的两条分支（活槽 / 死槽写 0）同样要跟着跨步循环。

**必须一起改、最容易漏的一处**：末尾的 fp16 打包现在是

    if (j < p.cap / 2u)  p16[... + j] = pack2_rne(vec2(sp[2j], sp[2j+1]));

256 条 lane 最多只覆盖 `cap = 512`。**cap 取 1024 时，概率索引 512 以上永远不会被打包**，
而 `round_probs = 1` 时 PV 读的就是 `p16` → 超出 512 的整段上下文用的是上一 token 的残留，
表现为「长上下文输出开始漂移」，而且**不会 NaN、不会报错**。改成同样的跨步循环：

    for (uint w2 = j; w2 < p.cap / 2u; w2 += NTILE)

**判别这个漏没有漏的 cheapest test**：`cap = 1024`，喂 600 token 的 prompt，只跑一步 decode，
把 `P16` 区域读回主机并与 `PROB` 逐元素比对（前窄后宽都应为 0 差）。

**lane 0 的两遍串行扫描（max，然后 exp+sum，然后除）保持不变**：顺序即参考顺序，位精确靠它。
代价估算：`cap = 1024` 时单 lane 约 3×1024 次依赖操作；`cap = 2048` 时 6144 次。
按 40 ns/次量级（本卡 `timestampPeriod = 40 ns`）粗算每层 < 0.25 ms、8 层 < 2 ms —— `未验证`，
**扩完之后必须用 §7.2 的仪表实测，不要引用这个数**。

**步骤 C：主机侧**

| 改动 | 位置 |
|---|---|
| 放开 guard，改成「按 cap 选窄/宽内核」 | bonsai_forward.h:1520 |
| 新增 `Pipe aqkw[8]` 与 `P_QKW` 枚举 | :100-101、:95-99 |
| 建管线：绑 7 个窗口，**顺序必须和 `q` 那组逐字相同** | :1211-1215 抄，:1221-1223 旁边加 |
| `attn_block` 里按 `f.cap > 256` 二选一派发 | :369-371 |
| 描述符池 | :1165 `maxSets = 96`；cap>256 时要用 8×4 = 32 套，**96 够用，改之前先数清楚** |
| 生成循环停止条件 | :1440 的 `pos >= cap` 保持，另加 EOS/换行逻辑见下 |

**步骤 D：接 chat template**

1. 在 `fwd_run`（:1327）之前把 prompt 组装成 §5.6 的字符串；建议直接写死「无 tools、单 system + 单 user」这条路径，
   `enable_thinking` 用 `BONSAI_THINKING=0/1` 控制，默认 1（与模板的「未定义即为真」一致）。
2. 分词：`Bpe::encode(text, parse_special=true, with_bos=false)`（bonsai_tok.h:367-395）在结构上就是为这件事写的
   ——special 表从 `token_type == 3/4` 构建（:214-226），匹配按「最长优先、永不拆分」。
   **但请注意：这一步只做过代码审查，没做过实测。** 39/39 的 tokenizer 对齐用的是**纯文本探针**，不含 CONTROL token；
   `BONSAI_CAP=300` 那次跑在 `fwd_init` 之前就 return 4，**根本没走到 `Bpe::load`**。
   `未验证` ⇒ **步骤 D4 的断言就是它的验收测试，必须先跑再往下**。
   另外**不要自己手写 id 序列**，让模板字符串过 `encode` 才是与官方一致的路径。
   还有一个真实的坑：`load()` 里 special 的条件是 `toks[i][0] == '<'`，**如果某个 CONTROL token 的字面量不以 '<' 开头就会漏收**，
   本次实测的 27 个都以 '<' 开头（`runs/chat_tpl_probe.log` 的枚举），所以当前没问题——但扩词表时要记得。
3. 停止：`sp.eos = 248046` 已经是 im_end（:1333），模板渲染完的字符串里 im_end 是分隔符而非终止符，
   所以「生成到 im_end 停」是**语义正确的默认**；think 段要不要打印出来给用户看，是产品决定，先两种都留着
   （`BONSAI_STRIP_THINK=1` 只打 im_end 之后的正文）。
4. **验收断言**：同一句话，我们组装的字符串过 `Bpe::encode` 的 id 序列，必须与官方
   `llama-tokenize` 带 `-chat-template` 渲染后的 id 序列**逐 ID 相同**。这是 STAGE6 已经建立的方法（tools/oracle_tokenize.py），照抄它的比对形式。

**步骤 E：长度上限的产品决策（不要顺手设成 32k）**

`cap = 1024` 时激活 arena 从 61.052 MiB 涨到约 92 MiB（KV 8 MiB → 32 MiB，`推导`：KV = `cap*32768 B`，
见 §5.2/§6.1）；`cap = 2048` 约 123 MiB；`cap = 8192` 约 340 MiB。deviceLocal 余量约 1.5 GiB，
**装得下**。真正的问题是：`prefill = 一 token 一次 submit`（§8.1），所以 prefill 吞吐 ≈ decode 吞吐
（实测 6 token / 450.4 ms = 13.32 tok/s）。**4000 token 的 prompt 需要约 300 s**，
任何「长上下文」卖点在这条路径上都必须先做批量 prefill 才成立（§8.1）。

**风险**：宽内核会让 `align_attn.exe` 的窄配置失真吗？不会——只要窄内核继续用于 cap ≤ 256 的回归，且宽内核在
`cap = 256` 时与窄内核逐位相同（这是它的第一个测试：**同 cap 双内核交叉验证**，比任何外部 oracle 都便宜）。

### 7.2 任务二：单 token 延迟拆解（`#25`）—— **结案：仪表已接上开关，第一份真实拆解表已出**

> **结案（2026-11-04，实测，`runs/profile_v2.log:129-146`）**：那一行开关真的只有一行
> （`bonsai_forward.h:1930`，`g_prof` :427）。32 个 decode token、wall 2308.4 ms、72.138 ms/token（13.86 tok/s），
> 配置是 `BONSAI_CAP=1024`（`runs/profile_v2.log:92` 的 sampler 行），所以这张表读的是**中间档 cap**：
>
> | 项 | ms/token | 占 wall |
> |---|---|---|
> | `T_gpu`（CB 自身 span，设备时间戳） | **63.648** | 88.23% |
> | `T_sync`（`vkWaitForFences`，**与 T_gpu 按构造重叠**） | 63.802 | 88.44% |
> | `T_memory`（logits 回传 0.946 MiB） | **6.270** | 8.69% |
> | `T_cpu`（主机录 CB） | 1.640 | 2.27% |
> | `T_submit` / `T_reset` | 0.024 / 0.002 | 0.03% / 0.00% |
> | `T_other`（argmax / printf+detok / bookkeeping） | 0.159 / 0.240 / 0.001 | 0.22% / 0.33% / 0.00% |
> | **bubble = wall − T_gpu** | **8.490** | **11.77%** |
>
> 三条决定性结论：
> 1. **host-visible 分项之和 72.137 ms vs wall 72.138 ms** —— 拆解闭合到 1 µs，没有「不知去哪了的时间」。
> 2. **logits 回传是最大的一块可消除成本**：0.946 MiB / 6.268 ms = **150.9 MiB/s**（`推导` = 0.946×1048576÷0.006268），
>    而本卡 SDMA 实测 **7.12 GB/s ≈ 6790 MiB/s** → 慢 **45 倍**，因为它是 uncached HOST_COHERENT 的
>    **逐 32-bit 提交**，不是 DMA 批量。把它变成**在设备上做 argmax** = 6.268 ms → ~0，
>    即 `推导` 72.138 → 65.87 ms/token = **15.18 tok/s（+9.5%）**。这是 ROADMAP 的 P0-2。
> 3. **原 §7.2 猜的「37% 排队损耗」在这条路径上不成立**：submit 只有 0.024 ms/token，fence 等待就是 GPU 在算。
>    真正的 11.77% 是 **bubble**（派发延迟 + 主机段），要减它得减 barrier/依赖图（P1-1），**不是减 submit**。
>
> 下面保留原方案文本，因为它记录了仪表的槽位安全论证（查询池 512 vs 槽位 ≤499）——那段仍然有效。

这是本次会话最有价值的一条发现：**`g_prof` 从来没有被任何地方赋值。**

    bonsai_forward.h:397   static uint32_t g_prof = 0;          // 唯一出现处，没有 getenv
    grep getenv 全文       -> 没有 BONSAI_PROFILE

所以 :414（CB 内 `vkCmdResetQueryPool` + 起始时间戳）、:452（结束时间戳）、:457-486（submit / wait / reset /
readback 四段分别计时）、:1457-1486（**整张拆解表已经写好了**，8 行 + GPU span + 重叠说明）全部是**死代码**。
它同时解释了 §8.2 的教训：STAGE5 里「37% 排队损耗」当初只能靠差值猜，现在不用猜了。

**要做的只有一行**（放在 :1519 那组 getenv 旁边）：

    if (const char *v = getenv("BONSAI_PROFILE")) g_prof = uint32_t(atoi(v));

**跑法**：`BONSAI_PROFILE=1 bonsai_vk.exe run <gguf> 32 48 > ../runs/profile_v1.log 2>&1`

**已有的正确性前提（本次会话核实过，不用再查）**：

* 查询池 `queryCount = 512`（bonsai_vk.cpp:371），仪表用的槽位是
  `q0 = 200 + (pf_slot % 150) * 2`、`q1 = q0 + 1` → 最大 499 < 512，**不越界**（§8.7 的那类事故不会在这里发生）。
* 池在 upload 之前就创建（:362-372 的注释记录了原因：NULL handle 会让 amdvlk64.dll SIGSEGV）。
* `timestampPeriod = 40 ns`（实测）→ 分辨率足够，但**单次读数 ±40 ns，只有对 48 token 取平均才有意义**。
* 表里已经写明 `T_sync` 与 `T_gpu` **按构造重叠**，别把它们相加当作总时间；bubble = wall − T_gpu。

**验收**：8 行百分比之和 ≈ 100%（表里最后一行「host-visible phases sum」对 wall/token 的偏差就是残差），
并把结果与两个已知锚点交叉：STAGE5 实测 fence 等待 11.8 ms + submit 5.0 ms = 16.8 ms（官方 fork 路径，不是我们的，
只做量级参照），以及我们的 logits 回传量 = `248320 × 4 B = 0.95 MiB / token`（SDMA 7.12 GB/s → 约 0.13 ms，
`推导`；**这条是 `T_memory` 的先验下界**，如果实测远大于它，说明回传没走 DMA 或者被 barrier 卡住）。

**已知的下一个瓶颈（如果拆解显示 host 侧占大头，先做这两件）**：每个 forward **1,490 次 dispatch + 1,489 次 barrier**
（推导：三份日志的累计计数 ÷ forward 数，全部整除——80,460÷54、83,440÷56、2,355,690÷1,581 都等于 1,490；见 §6.1 的同一行。
`bonsai_forward.h:1896` 把**累计**值印成 "per command buffer"，照字面读会高估一个数量级，本项目早先就把它写成了每 CB 8 万次）。
barrier 是 `dsp()` 里无脑跟的保守同步（:194-197 的注释自己承认了：
「cost 是 measured 的，dependency graph 等这个数已知重要再上」）。**本节仪表的数出来之后，才有资格决定要不要做依赖图。**

### 7.3 任务三：4 个残差「叛徒」定性（`#26`）—— **结案：既不是累加顺序，也不是我们的 Bug；处置是「不修」**

> **结案（2026-11-04）**：完整报告 **`runs/traitor_forensics_v1.md`**，决定性数据在
> `runs/order_sensitivity_v1.txt`（证据 A）与 `runs/qkv_arith_probe_v5.txt`（证据 C，22 种算术的判定表）。
> 三条腿的论证，每条都可开关复现：
>
> 1. **累加顺序被四个数量级否证**（证据 A）：换 `BONSAI_SPLIT` 造成的逐行变化 `d_order/Sum|x|` 最大 **1.977e-07**（中位 2.297e-08），
>    而与 oracle 的差 `gap/Sum|x|` 最大 **1.903e-04**（中位 1.530e-05）；96/96 条 block 行的判决都是「不是顺序」。
>    block-3 的四个锚点更极端：QKV `gap/d_order` = **43993.9×**，GD 63020.8×，QD 1407.5×，PGATE 16050.4×。
> 2. **喂给 GEMV 的操作数是干净的**（证据 B）：CPU 从 GGUF 重建 RES/NRM/AN/ROT/HR 五级链，
>    与 oracle 自己打印的 head3+sum 对上（`runs/qkv_arith_probe_v5.txt` 开头那段），且 0/8192 个 conv 通道与 `silu(w3*x)` 矛盾。
>    CONV-sum 与 L2-sum 都在 QKV 下游 ⇒ **「4 个叛徒」= 1 个病灶（op 9 QKV）+ 3 个传播**。
> 3. **我们的 GPU 值 = 精确 f64 点积**（0.537943 / −0.828764 / 0.310630 vs 日志 0.53794 / −0.82876 / 0.31063），
>    而 22 种候选算术里**只有对称 q8_0 激活重量化族**（blk-32、absmax/127、就近取整）落进 oracle 的 ±2.8e-4，
>    比精确算术的 0.00727 **近 26 倍**；它对 tie 规则、倒数网格、操作数链精度（f32 vs f64）、点积写法（ternary vs direct）**全部免疫**。
>    残差换算成物理量 = **0.15–1.6 枚被翻掉的 int8 code**（`推导`：一枚 code 的扰动 ≈ `d_block × rms|w|` = 0.010869 × 0.0164 = 1.78e-04）。
>
> **处置：不修。** 要「对上 oracle」就得**故意把自己的激活重量化降级**——那是拿正确性去换一致性。
> 结论性推论（写进纪律）：`BONSAI_ALIGN` 的锚点带是**抓顺序/抓形状**的哨兵，**不是精度认证**；
> 这 4 条 FAIL 归档为 oracle 侧的已知偏离，不再计入回归红线。
>
> **仍然未验证的（诚实清单，别当已知）**：官方那台 SIMD 到底怎么把 q8_0 的 dot 排出来（**未验证**）；
> block-0 是否真的跑在 CPU 上（**未验证**）；其余 31 个 block 没逐个做同样的定性（**未做**，只有 block-3 的顺序判别）。
>
> 下面的原文是**上一个会话写的任务书**，其中「接下来的三步」已经被上面这条结案取代——
> 第 1 步（`BONSAI_ALIGN=1 BONSAI_SPLIT=1`）与第 2 步（三个 GDN 开关）**不必再跑**：
> 结论由证据 A/B/C 给出，且不依赖这两条路径。第 3 步 `BONSAI_PARTIAL=1` 的逐量化块误差表仍是好工具，留着。

#### 已做完的：累加顺序假设已被实验否证

上一个会话跑过 `BONSAI_SPLIT` 的三档（runs/ladder_v6_rho.log = 计划自选、_s1 = 强制 S=1、_s4 = 强制 S=4，
scratch 尺寸 65536 / 256 / 993280 floats 证明开关确实生效）。逐 block 的残差和偏差 `delta = ours - official`：

| blk | S=auto | S=1 | S=4 |
|---|---|---|---|
| 0 | 0.02287 | 0.02288 | 0.02290 |
| 2 | −0.07399 | −0.07400 | −0.07398 |
| 3 | 0.08175 | 0.08182 | 0.08193 |

**换累加顺序只动第 5 位有效数字，偏差本身一点没消失** → 阶梯上看到的残差**不是** split-K 累加顺序造成的。
（三次运行的 FAIL 条数都是 40，第一块出带的都是 `block 0 (gdn)`。）

同一次实验还留下一个**未解释的结构**：`rho = |delta| / Sum|x|` 的跨块离散度

    rho spread: min 1.72e-06 (blk 1)   max 1.55e-04 (blk 2)   ratio 89.7x      (S=auto)
                                                        94.3x                   (S=1)
                                                        85.2x                   (S=4)

日志自己写的判据是「若这个比值是 O(1)，则单一 fp32 累加噪声系数能解释所有块」。**85–94 倍不是 O(1)**，
所以「一切都由浮点噪声解释」这个结论**不成立**，剩下的必须是别的东西。三档实验里这个比值稳定在 85–94，
说明它是**系统性**的，不是随机噪声抖动。

#### 还没做的：block-0 四个锚点（BONSAI_ALIGN 路径）

`BONSAI_ALIGN=1` 的 32 个锚点比较里 4 个失败（runs/fwd_align_v10.log:108,122,124），**全在 block 0 的 GDN 路径，
且互为上下游**——所以这不是 4 个 Bug，很可能是 1 个：

| op | 锚点 | 我们 | 官方 | 差 | 判定 |
|---|---|---|---|---|---|
| 9 | QKV head3 | 0.53794 / −0.82876 / 0.31063 | 0.53660 / −0.83380 / 0.31790 | 最大 7.3e-03（约 2.3%） | **h-FAIL + s-FAIL** |
| 9 | QKV sum | −324.971145 | −324.560303 | 0.411 = **0.127%** | 同上 |
| 16 | CONV sum | 168.894486 | 168.968338 | 0.074 = **0.044%** | s-FAIL |
| 17 | L2 sum | 21.647462 | 21.658962 | 0.0115 = **0.053%** | s-FAIL |

同日志里主机侧复算的 L2 表：`1/‖x‖` 我们 0.40951 vs oracle 0.40848，**比值 1.003**（第 1、2 行 1.001）。
`ssm_conv` 吃 QKV，`L2_NORM q/k` 吃 conv 输出 → **CONV 与 L2 都在 QKV 下游**。
**先定性 QKV，另外两个大概率自己消失。**

#### 接下来的三步（顺序不要换）

1. **`BONSAI_ALIGN=1 BONSAI_SPLIT=1`**：这条路径**还没做过**（只做过 ladder 路径）。
   若 QKV 的 0.127% 不随顺序变（按上面证据大概率不变），**当场结案「非累加顺序」**，把这条结论写进
   FORWARD_ALIGN.md，然后进入第 2 步。若它变了，说明两条 harness 覆盖的 GEMV 集合不同——那本身就是线索。
2. **查 QKV 的「输出排布」而不是数值**：QKV 区域长 8192 = q(16×256) + k(4×256) + v(4×256)。
   block 0 是 **GDN** 层，它读 QKV 的是 `ssm_conv` / `L2_NORM q/k` 分支，
   **这里最可能错的是切片偏移或头配对**（`gdn_gmap` 的两个候选 `h%16` vs `h/2`，:109 的开关就是为它留的）。
   工具：`BONSAI_GDN_GMAP=1`、`BONSAI_GDN_MODE=1`、`BONSAI_L2_MODE=0` 三个各跑一次 `BONSAI_ALIGN=1`，
   看哪个 FAIL 集合发生变化——**变化即是归属**（这比继续猜数值便宜得多）。
3. **`BONSAI_PARTIAL=1`**（bonsai_vk.cpp:199，逐量化块误差表）：看误差是否随块号单调。
   单调 ⇒ scale/codes 偏移；乱序且集中在某些列 ⇒ 索引/交错问题（§5.3 那一类）。

**动手前先按 §8.7 审计 oracle**：官方那几个 sum 是 `%8.6f` 打印的聚合值还是被二次舍入过的？
`dbg_fa_off.log`（4.7 MB）与 `dbg_full_real.log`（3.2 MB）是原始出处，比对脚本是 `tools/oracle_step1.py`。
本项目有过一次「驱动除法异常」最后查明是 harness 越界伪造的（§8.6）。

**验收（本任务算完成的定义）**：能指着日志说「这 4 条 FAIL 的来源是 X」，
并且 X 有一个**可开关验证**的机制（换开关/换顺序会改变 FAIL 集合），而不是「误差很小可以接受」。

### 7.4 任务四：开源产品化（`#27`）—— **结案：OOM 兜底已落地并三分支实测；完整路线在 `ROADMAP.md`**

> **结案（2026-11-04）**：本小节的全部交付移到 **`ROADMAP.md`**（今日水位、Release-0.1.0 的目录与 CLI 方案、
> 按 ROI 排序的优化表 P0-1…P2-2、发布验收清单的**如实**通过/未通过状态、以及「不能修」的已知问题清单）。
> 这里只留两条**已经落地**的硬事实：
>
> 1. **OOM 兜底已实现并在硬件上走过三条分支**（代码 `bonsai_forward.h:1691-1766`（build_v15 时代是 :1150-1215，
>    行号随 STAGE9/10 的代码增加漂移，认符号 `per_slot` / `NOTHING WAS DISPATCHED` 比认行号可靠；
>    + `bonsai_weights.h:117` 的 `device_local_heap()`）；
>    日志 `runs/oom_fallback_v2.log`；预算解析式与三个常数见 **§5.9**）。
>    兜底顺序正是任务书要求的 ①降 cap ②拒绝对话而不是静默截断 ③打预算表，
>    而且 B/C 两条都**在第一条命令都没提交之前**退出（`NOTHING WAS DISPATCHED`），不留半初始化状态。
> 2. 顺带修掉一个**一直存在的数据 Bug**：`model_plan_report(e.plan, 0)` 让**每一份** bring-up 日志都打
>    「device-local heap: 0 MiB」。现在打 3840 MiB（实测），因为改用了 `device_local_heap()` 从 `memoryHeaps[]` 读。
>
> **原任务书里第 1 项（OOM 兜底）与第 2 项（chat template）现在都完成了；第 3–6 项没做，排序与理由在 `ROADMAP.md`。**
> 特别注意 ROADMAP 里的一条纪律：P1-1（减 barrier）**必须等 §7.2 的仪表出数**——数已经出来了（bubble 11.77%），
> 所以它现在是有根据的，不再是猜。

1. **OOM 兜底（最高优先，因为它是「用户第一次跑就撞上」的那堵墙）**
   现状：`Arena::init` 装不下时只有 `act arena did not fit in the 4 GB wall`（:1111）这一句 stderr，
   用户既不知道为什么，也不知道该调什么。
   实测的墙在**官方 fork** 上是 `-c 49152` 过 / `53248` 撞，**且先撞的是 compute buffer 不是 KV**（STAGE5）。
   兜底顺序应当是：① 降 `cap`；② 拒绝对话（不静默截断）；③ 打印一张「权重 2.410 GiB + 激活 X MiB + KV 32 KiB/token
   × cap」的预算表，让用户看到是哪个数超了。全部常数都是实测的，见 §6.1 与 §5.2。
2. **chat template + 停止条件**（= §7.1，产品意义上这才是「能对话」）
3. **prefill 批量化**（§8.1）。这是 4GB 卡上第一个真实的速度功能：长 prompt 时用户看到的是几十秒空白。
4. **依赖图 / 减 barrier**（§7.2 末尾）——**必须等仪表出数**，否则是在猜。
5. **量化档位 / 上下文档位**：`cap` 已经是 push constant + 区域参数，档位化成本很低（256 / 1024 / 2048）。
6. **Release 包**：`bonsai_vk.exe` 只依赖 `vulkan-1.dll`（系统自带）+ `libgcc/libstdc++/libwinpthread`（MinGW 运行时）。
   打包 = `windeploy` 式的三个 dll + 模型路径参数 + 一个 `--help`。**注意**：本卡驱动是 AMD 专有 amdvlk，
   别人的机器可能是 RADV/ANV —— 项目里已有两条驱动特定的硬事实（`packHalf2x16` 向零取整、NULL 查询池 SIGSEGV），
   所以 Release 前必须在第二块卡/第二个驱动上至少跑一次 `align_*` **五件套**（STAGE10 起多了 `align_27b`；四件套的旧提法在 §6.2/STAGE9 那些历史行里仍然指它们当时的那四条）。

**发布前的验收清单（写在这里，不要凭感觉）**：

    [ ] build_mingw.bat 干净机 exit 0 / -Wall 0 warning
    [x] §6.2 五件套全过（`runs/regression_v14.log`：含 align_27b 的 `ALL CHECKS PASSED`）
    [x] 位相同门 A/A2/A3/A4 全 0 bit-diff（同一日志 :88-94，`gate: PASS`）
    [ ] §6.1 点火基线 ≥ 13.8 tok/s（48 token）
    [ ] chat template 后的 id 序列 == 官方 llama-tokenize 逐 ID（§7.1 步骤 D4）
    [ ] cap 档位各自的 tok/s 与显存实测表（不许用推导值交付）
    [ ] OOM 路径：故意把 cap 设到装不下，确认预算表可读、程序不崩、不留半初始化状态
    [ ] 第二驱动栈上的 align_* **五件套**（§5.4/§8.3 的两条驱动特定事实 + §15.4 的两类 Bug 的可移植性）

---

## 8. 踩过的坑与必须遵守的纪律（**这一节的价值高于任何代码**）

### 8.1 结构性的：prefill 是「一 token 一次 submit」的 decode 循环

`fwd_run` 的两个逐 token 循环（prefill :1723、decode :1819）逐个调 `fwd_token`，每次一个 CB、一次 submit、一次 fence，`seq` 恒为 1。
好处：`Q16/PROB/GATE` 这些单份临时区永远只有一层活着，区域布局简单，注意力内核可以按 seq=1 写。
代价：`attn_qk_softmax` 每 (token, head) 一个 workgroup，且**每个位置都要付一整个 forward 的 1,490 次 dispatch**，
所以 prefill 的开销是**乘在 prompt 长度上**的（1,579 个位置 × 1,490 = 2,352,710 次；实测整 run 累计 2,355,690 = 1,581 个 forward × 1,490，差的 2 个 forward 就是那 2 个 decode token）。**任何「长上下文」「首 token 延迟」的讨论都必须从这里开始**。
改成批量 prefill 需要动：着色器（mask 的 stride 已经是 `t*cap`，OK）、
临时区（`Q16/SCORE/PROB/P16/PGATE/GOUT/QF/QD/GD` 现在都只按 `nh` 长度分配，要按 `seq*nh`）、
以及 KV 插入（`attn_norm_rope` 已按 `position` 写 arena，**这部分不用改**）。

### 8.2 「先测，再说是谁的锅」

* 「37% 排队损耗」这个说法最初是差值猜出来的，实测是 **16.8 ms = fence 等待 11.8 + submit 5.0**（STAGE5）。
* token 基准也更正过：7.929 GMAC / 53.8 ms / 18.58 tok/s 那一轮（STAGE4），不是最初估的。
* 结论：**报告里每个数都要有出处**，`实测 / 推导 / 未验证` 三态必须标（用户明确要求的纪律，见 §11）。

### 8.3 数值纪律：**累加顺序是合同，不是风格**

* CPU 镜像与 GPU 内核必须**以相同顺序加同一个数**，比较才是位精确的，不是「在容差内」。
* `x*x` 这类平方项**必须落到 shared memory 再取回来**，不能写成 `ss += x*x`：
  优化器会把加法与乘法合并成 FMA，**FMA 加的是未舍积，结果是另一个数**
  （`attn_norm_rope.comp:79-87` 整段注释就是这件事）。同理 `acc = acc + a*b` 的写法要逐字照抄，不要「美化」。
* fp16 打包用 `pack2_rne()`，因为本驱动的 `packHalf2x16` 是向零取整（§5.1）。
* 被 mask 的 lane 必须产出**精确 0**（`exp(-inf-max)`），不是「很小」。
* 驱动下限不是 Bug：本卡 `sqrt / 1/sqrt / 1/x` 实测 1.00 / 1.16 / 1.00 ulp，
  层 3 的残差被**完全归因**到每行 norm 常数的 ±1 fp32 ulp（0 条不可解释）。**这就是 §6.2 那些 1e-3 级误差的来源。**

### 8.4 Vulkan 三条硬教训

1. **正确性真因曾是一个漏掉的 `vkCmdPipelineBarrier`**（STAGE3）。`dsp()` 现在每个 dispatch 后跟一个
   `bar_compute`（:194-197，注释本身就说明这是保守同步、代价靠 `n_barrier` 实测），compute→transfer 另有
   `bar_compute_to_transfer`（:482-484 注释：少了它，
   CPU 采样到的是上一个 token 的 logits）。**加 dispatch 时必须同步加屏障，减屏障时必须给依赖证明。**
2. **`VkDescriptorBufferInfo::offset` 是「VkBuffer 之内」的偏移**，而我们的 VkBuffer 本身已经始于 arena 偏移，
   所以 `Win` 永远不带 arena 偏移，只带区域内的 region 偏移（`Win` 定义 :88，`bind_wins` :161，规则写在 :158-160 的注释）。
   当年 `bind_set` 与窗口两种绑定的混用就是 Bug 本体。**注意力三个内核从元素 0 开始索引缓冲区**，
   所以每层要单独一套描述符窗口（:1319-1348：每层 4 次 `mk_pipe`——norm/rope、窄 QK、宽 QK、PV；窄与宽传同一组 `q` 窗口，所以源码注释按 3 组窗口计），别改成整 arena 绑定。
3. **查询池必须先于 upload 创建**：NULL handle 传给 `vkCmdResetQueryPool` 直接把 amdvlk64.dll 打成 SIGSEGV，
   回栈只显示到 main 里的加载器 thunk（:362-365）。看着像队列能力问题，其实是空句柄。
   另外 `BONSAI_API12=1` 才拿得到真的 `vkResetQueryPool`（1.1 下返回 NULL）。

### 8.5 对齐/显存编排

* `minStorageBufferOffsetAlignment` 实测 **4 B**，但区域偏移仍统一按 16 的倍数（float 单位）走；
  代码里那句 `if (af < 4u) af = 4u; af = (af+15)/16*16;`（:1053-1054）是**查询后取严**，不是拍脑袋。
* 两个必须 64 字节对齐的内核（`attn_q_norm` / `attn_k_norm` 从元素 0 索引）被**从 f32 blob 里复制出来**
  放进 act（:1139-1158），因为文件只提供 16 字节张量对齐。**别把它们改回原地引用。**
* 区域重叠是显存问题里最常见的一种，**诊断工具就是打印出来的 region table**（:1230-1232，注释原话：
  「RES 为零有两种原因——没写过，或者两个 region 是同一批字节，只有偏移能区分」）。**保留这张表。**

### 8.6 归因硬件之前，先证明输入/输出通道干净

原话教训：**「驱动除法异常」是 harness 越界伪造出来的**（STAGE7）。
同类事故：conv 历史按 `inner=4096` 分配，导致第 n 层覆写第 n+1 层的 taps，静默串扰（§5.2）。
**先证明通道，再定罪硬件/驱动。**

### 8.7 oracle 侧也要审计

比对失败时，先确认：官方那行是怎么打印的（小数位、聚合是 mean 还是 sum、**fp16 还是 fp32**），
以及 oracle 抽取脚本读的范围对不对。**本项目吃过一次「我们没错，是 oracle 抄错了」的亏。**

### 8.8 环境坑（本次会话再撞一次）

* Windows 控制台默认 cp936：Python 打印含 U+0120（GPT-2 bytes_unicode 映射）的模板全文会在
  **最后一条 print** 上 `UnicodeEncodeError`；如果同时 `> file` 重定向，前面的 stdout 可能整份丢掉。
  修法是 `sys.stdout.reconfigure(encoding="utf-8", errors="backslashreplace")`（已加进
  `tools/probe_chat_template.py:12-16`）。**所有打印模型文本的新脚本都要带上这四行。**
* **GGUF v3 的字符串长度是 uint64**，不是 uint32（`bonsai_gguf.h:65` 一直是对的；本次会话手搓探针时用 uint32
  立刻炸在 `unknown gguf type 1701999988` —— 那串数字是 UTF-8 字节被当成 type 读了，**偏移错位的典型指纹**）。
* `bonsai_vk.exe ... | tail` 之后 `$?` 是 tail 的退出码；判 `return 4` 这类分支要看 stderr 文本或 `${PIPESTATUS[0]}`。
* **`shutdown /s /t N` 在这台机器上是真的会执行**（STAGE8 末尾那次留下 EventID 1074 @ 03:37:04，机器停了一整夜到手动开回）。
  所以：**关机只排在「所有产物已落盘、回归已跑完」之后**，并且先想想还有没有正在后台跑的基准——
  `run_in_background` 的任务不会因为你想「等一下再看」而等你。
* **`powershell -File xxx.ps1` 不带 `-ExecutionPolicy Bypass` 会静默失败**（策略 Restricted）。清场脚本没跑成 = 基准数据被污染，
  这比报错难查得多。§4.4 第 1 条已经把参数写全，照抄。
* **工具的 `exit 2` 往往是「用法不对」而不是「测试失败」**：`align_tokenizer.exe` 少给参数就 `cannot open ...` + `exit 2`，
  看着像回归。先读 usage 行（`align_tokenizer.cpp:86`）再下结论。
* **`dispatches ... per command buffer` 那行印的是累计值**（`bonsai_forward.h:1896`），标签是误导性的。
  真实数：**每 forward 1,490 dispatch / 1,489 barrier**——80,460÷54（`ignition_v2.log:151`）、83,440÷56
  （`profile_v2.log:151`）、2,355,690÷1,581（`qcheck_2048_v2.log:110`）三个独立 run 全部整除到同一个数。
  本项目的 ROADMAP 早先把它写成「每 token ~14,900」，**错了 10 倍**，现在 §4 P1-1 已更正并把错误留在记录里。
  教训：**任何「总数 ÷ 次数」的除法都要用两个独立 run 交叉，才能发现标签在撒谎。**
* **日志覆盖是不可逆的**（本仓库不是 git）。STAGE9 就覆盖过一次 `runs/ignition_v3.log`（见 §9.1 的标注）。
  写日志前先 `ls runs/<前缀>*` 看下一个空号，别用记忆里的版本号。

---

## 9. 产物索引（runs/ 里该看哪几份，别看错版本）

### 9.1 当前有效（= §6 基线的证据）

**STAGE10（2026-11-05）先读这四份**，它们是 §15 的原始出处；下面 STAGE8/9 的日志仍然是它们那一次的证据。

| 文件 | 内容 |
|---|---|
| `runs/regression_v14.log` | **当前二进制的全量保真复跑**：tokenizer 32/32 + 7/7、dequant `max abs 0.000e+00` / bit-exact 128000/128000、f16 两条着色器表达式与主机转换逐位、align_attn score 3.567e-03（seq 64，比值 0.262）、align_27b `ALL CHECKS PASSED`、QCHECK `gate: PASS (bitmask 0 = A A2 A3 A4 B C D E)`（:94）、端到端 **14.16 tok/s / 70.6 ms/token**（:95-96）。改 `bonsai_weights.h` 之后必须复跑的就是这一套 |
| `runs/gqa_pair_v1_summary.txt` | **GQA 阶梯的权威读数（batch H 配对）**：4 个长度 × {off, on1, on3} + cap 256 三行 + 末次对照复跑 ⇒ §15.3 的整张表与 1.5 % 漂移带 |
| `runs/fleet_ring_v4.log` | **多卡环的当前实测**：selftest 55/55 PASS、0 字节不符、6.96 GB/s 单腿 / 7.13 GB/s 批处理、背压 2 次、fence 2706.7 µs vs 门铃 0.3 µs、created-SIGNALED 对照 0.8 µs + 1195.2 µs 尾巴（REPRODUCED） |
| `dist/bonsai-vk-0.1.0-win64/` + `runs/package_smoke_v2.log` | **发行包本体（STAGE10 新加）**：91 个文件 ≈ 6.5 MiB（字节总数不钉死：它是文档之和，改一次 .md 就变；精确值用包根目录那条 `python -c` 递归求和现算，`MANIFEST.sha256` 90 条 `sha256sum -c` 实测 90/90 OK），含 `shaders/`（52 支 .spv，307 432 B，**必须随包**——着色器按当前目录加载，不内联进 exe）、`fidelity/`（5 支：4 支对齐工具 + shared_probe）、`runtime_vk/`（4 份源码，291 355 B，只为让 README 的相对链接在包内也成立；**本包仍是二进制包**，引擎源码与构建脚本不在内）、`LICENSES/README.md`（引擎许可仍 TBD）、`MANIFEST.sha256`。`package_smoke_v2.log` 是**从包根目录**实跑的最终形态开箱结果：rc=0、`gate: PASS`、14.31 tok/s（8 token，小样本）、cap 256 活动区 58.2 MiB 与 PDH dedicated 2539.23 MiB 首次成为实测；`package_smoke_v1.log` 是 86 文件形态那一次（同一支 exe，判定量全同，只有权重装载墙钟 1201.0 → 1099.9 ms），两份日志仓库都留着 |
| `runs/fleet_probe_v2.log` + `runs/fleet_offline_v1.log` | 拓扑判定：(A) 层流水线 **NOT EXERCISABLE**（1 COMPUTE-capable device of 1）、(B) zero-copy 能力 AVAILABLE（附 256 MiB BAR 的事实）；以及不建任何 Vulkan 对象的 selftest 独立留档 |
| `runs/build_v40.log` / `v41` / `v42` | 清理后的三次全量构建：`rc=0`、**0 warning**、52 支 SPIR-V + 8 个 exe |
| `runs/align_27b_v2.log` | 27B 适配器单独留档（同一套门禁的另一趟；§15.5） |

| 文件 | 内容 |
|---|---|
| `runs/ignition_v2.log` | **STAGE8 主基线**：13.99 tok/s、region table、显存计划、48 个 token 逐条打印 |
| `runs/ignition_v3.log` | **当前二进制的点火**：14.04 tok/s、chat template 开、16 prompt token、稳态 71.2 ms/token。**注意：这个文件名被覆盖过一次**——原来这里是「同二进制复跑、48/48 id 与 v2 逐位相同、13.77 tok/s」那份，它的内容只剩 §6.1 抄写的三个数。这是对本仓库 §4.4 第 3 条纪律的一次违约，记录在此而不是抹掉 |
| `runs/regression_v13.log` | **最终二进制（build_v15）的四件套**（16 行，逐行核对）：32/32 + 7/7 byte-identical、`max abs 0.000e+00`、f16「whole binary16 domain: YES (0 wrong numbers)」+ bit-for-bit YES、score 6.695e-04 |
| `runs/build_v15.log` | 最终构建：`exit 0`，`grep -icE "warning\|error"` = **0**（`ROADMAP.md` 的验收清单第 1 项以前引用的是一个没落盘的 build_v14，现在以这一份为准） |
| `runs/fwd_align_v11.log` | 最新 block-0 锚点：**4/32 FAIL**，与 v10 同一模式（宽内核 + 模板 + OOM 兜底没动它，§7.3 的交叉证据） |
| `runs/ladder_v5.log` | 残差阶梯（32 block × 分支/FFN/残差 三件事 vs `l_out-0..31`） |
| `runs/ladder_v6_rho.log` / `_s1.log` / `_s4.log` | **§7.3 的证据 A 原始日志**：三档 split-K 的 rho/cancel 表 |
| `runs/order_sensitivity_v1.txt` | §7.3 证据 A 的**计算结果**：`d_order/S` vs `gap/S` 逐行、96/96 判决、block-3 四锚点比值 |
| `runs/qkv_arith_probe_v5.txt` | §7.3 证据 C 的**判定表**（22 种算术 × 6 通道；v2–v4 是同一实验的迭代，留作审计链） |
| `runs/traitor_forensics_v1.md` | **§7.3 的结案报告**（中文，含推导、可复现块、诚实未验证清单、以及「为什么不修」） |
| `runs/profile_v2.log:129-146` | **§7.2 的拆解表**：T_gpu 63.648 / T_sync 63.802 / T_memory 6.270 / T_cpu 1.640 / bubble 8.490 ms/token，分项和 vs wall 闭合到 1 µs |
| `runs/oom_fallback_v2.log` | **§5.9 的三条兜底分支实测**（A 装得下 / B KV 不够 → 建议 BONSAI_CAP=1704 / C 任何 cap 都不够 → 报 2519.8 MiB 门槛），B/C 都 exit 4 且没有提交任何命令 |
| `runs/qcheck_256_v3.log` | **窄 vs 宽内核逐位交叉验证**（:96-101，SCORE/PROB/P16 各 0/4096 bit-diff；:103 死槽 0 违规） |
| `runs/qcheck_2048_v2.log` | 宽内核在 **1579 live keys** 下的 B/C/D/E 五项审计（:95-101）+ prefill 131.99 s（:93）+ 94.2 ms/token（:109） |
| `runs/chat_think_v1.log` / `chat_nothink_v1.log` / `chat_en_v1.log` | chat template 三条路径的完整对话实跑（192 / 96 / 96 token；两条 id 路径 IDENTICAL 在各自 :90） |
| `runs/limits_probe.log:3` | **`maxComputeSharedMemorySize = 32768 B` 的实测出处**——cap 2048 这条墙就钉在这里 |
| `runs/chat_tpl_probe.log` | **chat template 全文 + CONTROL token 表 + 全 metadata**（§5.6） |
| `runs/gguf_metadata_dump.json` | 同一份 metadata 的 JSON（不含大数组） |
| `runs/bench_ngl99_vs_cpu.log` | 官方 fork 的 GPU/CPU 对照，含 `uma/fp16/int dot/warp/shared memory` 那行 |
| `runs/tok_probes.txt` + `tok_ref_32.jsonl` / `tok_probes_kr.txt` + `tok_ref_kr.jsonl` | tokenizer 的两个探针集与官方参考值（32 + 7 = 39） |
| `runs/tensor_table.tsv` | 427 个 tensor 的 K/M/类型表（权重计划的原始输入） |

### 9.2 历史版本（别当基线，只做趋势对比）

`fwd_align_v1..v10.log`（v9/v10/v11 同一 4/32 失败模式 ⇒ §5.3 的修复与 STAGE9 的全部改动都没引入新的锚点差）、
`ladder_v1..v4.log`（v4 = 逐行证明 attn_q 交错布局的那份）、`qkv_arith_probe_v1..v4.txt`（v5 是判定版，前四份是迭代审计链）、
`oom_fallback_v1.log`（v2 是最终二进制的那份）、`regression_v1..v12.log`（v13 是最终二进制）、`qcheck_256.log` / `qcheck_256_v2.log`、
`tail_ab.log` / `tail_fixed.log`（网络尾段 vs dump）、
`stop31_dump.log`、`fwd_stop0*.log`、`run_ngl99*.log`、`run_cold_*.log` + `vram_cold_*.tsv`。
`ladder_c8192..c65536.log` 是官方 fork 的上下文墙扫描（STAGE5 的 `-c 49152 过 / 53248 撞`）。
`dbg_rope_*.log` **全是死路**：n_rot 已由 dump 钉死为 64，这些探针不再产生新信息，可删。

### 9.3 报告文档链

`README.md`（顶部有忠实复现声明）→ `STAGE0/1/2/3_REPORT|KERNEL|CUSTOM|RUNTIME` → `DEEPINFER_EVAL.md`
→ `END_TO_END_MEASURED.md` → `FAITHFUL_REPRO.md` → `ATTENTION_ALIGN_FINAL.md`
→ **`ROADMAP.md`（STAGE9：产品化路线 + 验收清单的真实状态）** → **`runs/traitor_forensics_v1.md`（STAGE9：§7.3 结案）**。

上一版这里写的「STAGE8 尚未单独成文，建议补 `FORWARD_ALIGN.md`」：STAGE8 的结论已经在 §6.1 成文，
`FORWARD_ALIGN.md` **没有建**，它原本要装的内容（4 个锚点定性）现在在 `runs/traitor_forensics_v1.md` 里，
**不要去开那个文件**。

---

## 10. STAGE9（2026-11-02 → 11-04 三个会话）改了什么、停在哪

> **STAGE10 已经走过了这一节的入口清单之外**：GQA 4× 合并、多卡机械层、27B 适配器、死代码清理与
> 0 warning 全仓构建都在 §15。**本节下面那四条「下一次会话的入口」仍然有效**（设备内 argmax、批量 prefill、
> D4 外部比对、Release 打包），但优先级请对照 §15.7 读。当前二进制尺寸是 **539 KiB / 552,178 B**
> （`实测` `python tools/source_size.py`）；本节写的 419,401 B 是 build_v15 那一次的，属于历史。

**四项任务（§7.1 / §7.2 / §7.3 / §7.4 + 任务一·甲）全部结案**。代码与文档的实际改动：

* **引擎**：新增 `bonsai_chat.h`（105 行，chat template）；新增 `shaders/attn_qk_softmax_wide.comp`
  （lane 跨步 + `MAXCAP 2048` + 跨步 fp16 打包，**窄内核未改一行**）；`bonsai_forward.h` 1552 → **1990**
  （cap 选内核、QCHECK 五项审计、`BONSAI_PROFILE` 那一行、chat 渲染、**显存预算 + OOM 三分支 :1150-1215**）；
  `bonsai_weights.h` 新增 `device_local_heap()` :117。当前二进制 `bonsai_vk.exe` **419,401 B，2026-11-04 14:05**（`runs/build_v15.log`，exit 0 / 0 warning）。
* **新开关**：`BONSAI_CHAT` / `THINKING` / `SYSTEM` / `CHAT_DUMP` / `STRIP_THINK` / `STOP_NL` /
  `QCHECK` / `QKERNEL` / `PROFILE` / `HEAP_LIMIT_MIB`（逐个的语义与行号见 §4.3）。
* **新工具**：`qkv_arith_probe.py`（22 种算术判定）、`order_sensitivity.py`（顺序否证）、`oracle_self_consistency.py`（oracle 自洽下界）。
* **文档**：新增 `ROADMAP.md`、`runs/traitor_forensics_v1.md`；本文件 §1/§3/§4/§5.6–5.9/§6/§7/§9 全部改写。

**停在哪（下一次会话的入口，按性价比排序）**：

1. **P0-2 设备内 argmax**：把 6.268 ms/token 的 logits 回传干掉，`推导` 到 **15.18 tok/s（+9.5%）**。
   这是唯一有实测根据、且不需要新算子语义的性能改动（依据 §7.2 的表）。
2. **P0-1 批量 prefill**：1579 token 现在要 131.99 s（实测），这是「用户第一次粘长文就劝退」的那堵墙（§8.1）。
3. **步骤 D4**：`BONSAI_CHAT_DUMP` + 官方 `llama-tokenize -chat-template` 逐 ID 比对。半小时的活，做完 §1 的第 3 条就能划掉。
4. **Release-0.1.0 打包**：`ROADMAP.md` §3 的目录 + `--help`/`--version`。**发布前必须先在第二个驱动栈上跑五件套**
   （§5.4/§8.3 那两条驱动特定事实的可移植性还没被验证过）。

**明确不要做的**：不要去「修」§7.3 的 4 条锚点 FAIL（结案结论是不修）；不要做专家流式换入换出
（STAGE4 结论 D：装载恒赢，≥78.3% 权重必须常驻）；不要在 `cap > 2048` 上想办法绕（那面墙是 32 KiB shared memory，
要动的是内核结构）；**永远不要写模型文件**（用户铁律）。

---

## 11. 文档与协作约定（**用户会逐行审计，照做**）

* 中文交付；**每个数字标 `实测 / 推导 / 未验证`**，推导要给算式。报告会被外援院士逐行审计。
* 不做口号式结论，**不许为了圆结论编常数**。
* **端到端真机跑才算完成**：静态读代码不算测过功能；基准必须在真卡上跑，日志留档。
* **全量跑一次、核心断言过了就归档**；边角 Bug 写进报告，不现场死磕。
* 改配置/删文件之前先证明（备份 + 回滚脚本 + 改动清单），**等用户说「动手」**；模型文件永远只读。
* 用户常给 2–3 个词的短指令：**先做可逆动作，再汇报**，少问多做。
* 文档里写 ChatML 类标记一律用「左角括号 + 竖线 + 名称 + 竖线 + 右角括号」的分写（原因见 §5.6 开头）。
* 项目记忆目录（正斜杠写法）：
  `C:/Users/<user>/.qoder-cn/projects/<project-slug>/memory/`，索引 `MEMORY.md`，正文
  `project-ternary-moe-runtime.md`。**记忆会过时**：它写的是「当时为真」，引用前先按 §6 的日志核一遍。

---

## 12. 构建/运行失败最常见的三种

| 症状 | 真因 | 出处 |
|---|---|---|
| `cannot find -lvulkan-1` | `<toolchain>/vklib/libvulkan-1.a` 丢了；MinGW 和驱动都不附带 | §2.2 的 `gendef` + `dlltool` 两行重建 |
| 启动即 SIGSEGV，回栈只到 main 里的加载器 thunk | **NULL 查询池**句柄传给了 `vkCmdResetQueryPool`（看着像队列能力问题） | bonsai_vk.cpp:362-365 |
| logits 是上一个 token 的残留 | 少了 compute→transfer 屏障 | bonsai_forward.h:445-447 |
| **`cap must be an even number of slots` / `cap N exceeds the widest validated row`** | 不是 Bug：前者是 P16 成对打包的硬要求，后者是 **32 KiB shared memory** 的墙（`runs/limits_probe.log:3`）。**改 cap 之前先读 §5.7** | bonsai_forward.h:1940 / :1944 |
| **`OOM FALLBACK ... NOTHING WAS DISPATCHED` + exit 4** | 显存预算不够，程序**按设计**拒绝启动并给你一个可用的偶数 `BONSAI_CAP`。看 §5.9 的三分支语义再决定调哪个数 | bonsai_forward.h:1150-1215 |
| **`chat: a control id does not round-trip as a SINGLE token` + exit 6** | 词表里某个分隔符被拆成多枚，模板会退化成散文。这是**故意拒绝**，不是失败 | bonsai_forward.h:1659-1663 |

另：`glslang.exe` 不叫 `glslangValidator.exe`；构建脚本第 0 步要 `python`（生成 oracle 着色器源码）。

---

## 13. 这份文档没覆盖的，去哪里找

* 本轮收口的完工记账：`PROJECT_BONSAI_FINAL_REPORT.md`（产物路径 + 精确字节数 + 四项任务的实测/推导/未验证判定 + 未闭合清单；
  与本文件冲突处以它的 §「未闭合」为准，因为它是最后写的）。
* 逐 op 的官方 dump 原文：`runs/dbg_fa_off.log`（4.7 MB，`-fa off` 的全部中间张量）、
  `runs/dbg_full_real.log`（3.2 MB，1-token decode 图，`result_output` 有 oracle）。
* 注意力的 CPU 镜像与「理想腿」：`runtime_vk/attn_vk.h`（内核注释里凡写「the GPU twin of `attn_*_row()`」都指它）。
* STAGE4 四项判定：`DEEPINFER_EVAL.md`（A 有条件 / B 可行但只 0.7% / C 不可行-无专家 / D 不可行-装载恒赢）。
* 显存编排的全部实测（PCIe、SDMA、提交形状）：`END_TO_END_MEASURED.md` + `stage3_measured.txt` + `runs/pipe_measured.txt`。

## 14. 一句话给接手的人

核心正确性已经用四类 oracle 钉死（token 逐 ID、反量化 0 差、f16 边缘 0 差、注意力 1e-3 级且有 ulp 级归因），
外加两条新的：**窄/宽 QK 内核在 cap 256 逐位相同**（§5.7）与 **§7.3 把剩下的 4 条锚点差归因到 oracle 自己的激活重量化**。
**外壳也基本齐了**：模板、长度到 2048、时间拆解、OOM 兜底、Roadmap。剩下的是 **§10 的四件事**：
设备内 argmax（实测依据最强、+9.5%）、批量 prefill（体验上最大的一块）、模板 id 的外部比对（D4）、Release 打包。
这张卡的真实瓶颈已被 §8.1 定性：**prefill 是逐 token 提交的**，1579 token 实测 131.99 s。
**读顺序建议**：§1（状态）→ **§15（STAGE10 的新事实，本文件最后更新的部分）** → §5.9 + §7.2（显存与时间的两个解析式）→ `ROADMAP.md`（下一步）→ §8（纪律，别跳）。

---

## 15. STAGE10（2026-11-05 凌晨）的新事实

这一节是本文件里**最新**的一层。前面 §5–§7 的所有数字都仍然是它自己那次的实测，没有被改写；
如果 §15 与更早的小节在同一个量上给出不同的值，**以 §15 为准，并在引用旧值时说明是哪一版二进制**。

### 15.1 现在的默认基线是什么（`实测`，`runs/regression_v14.log`）

| 量 | 值 | 出处 |
|---|---|---|
| 端到端解码（cap 256，默认配置 = GQA 合并 AUTO） | **14.16 tok/s**，48 token / 3389.5 ms，稳态 **70.6 ms/token**，首 token 70.2 ms | `runs/regression_v14.log:95-96` |
| prefill（chat template 默认开，16 prompt token） | 1154.5 ms = 13.86 tok/s | 同上 :80 |
| 位相同门 | `gate: PASS (bitmask 0 = A ok A2 ok A3 ok A4 ok B ok C ok D ok E ok)` | 同上 :94 |
| 对 4 GB 参考引擎 | 4.417 t/s ⇒ **3.21×**（STAGE8 是 3.16×，STAGE9 是 3.18×；三次都是同一张卡同一个 checkpoint） | `END_TO_END_MEASURED.md` + 上表 |
| 回归判定 | 14.16 落在 14.05–14.20 的冻结带内 ⇒ **0 % 回归**；预算是 ≤ −5 % | §6.1 + `KV_P0P1P2_MEASURED.md` §8 |
| 全仓编译 | **0 warning**，`rc=0`，8 个 exe（引擎 + 5 个 align/probe + align_27b + bonsai_fleet） | `runs/build_v40.log` / `v41` / `v42` |

**§7.3 那条「BONSAI_ALIGN=1 4/32 失败」的定性没有变**，STAGE10 没有触碰 GDN/L2 路径，所以那张表仍然按原文读。

### 15.2 多卡流水线：机械层已建，且本机能测的部分全测了（`实测`，`runs/fleet_ring_v4.log`）

命令 `bonsai_fleet.exe ring 4 8 6`（深度 4、槽 8 MiB、6 hop，传输专用队列族 = 本卡 family 2）：

| 量 | 值 |
|---|---|
| selftest | **55 checks, 0 failed → PASS**（纯主机，不建任何 Vulkan 对象） |
| 环的字节完整性 | D2H 落地 **0 mismatch**（读 12 word / 写 4 槽，0 未触碰）；游标交叉校验 `produced 6 vs hops 6 -> consistent` |
| 单条 8 MiB D2H（带 fence） | **1204.6 µs = 6.96 GB/s = 实测 7.15 GB/s 上限的 97 %**（含 submit + fence + 队列排空） |
| fence 与尾巴 | fence 1204.1 µs，之后 queue-idle 尾巴 **0.4 µs** ← 这才是「fence 就是完成信号」的样子 |
| 8 条 8 MiB 一次 submit | fence 9410.9 µs + 尾巴 0.6 µs ⇒ **每份 1176.4 µs = 7.13 GB/s**（≤ 单条上限，这个不等式就是批处理为真的自检） |
| 每 hop（本机一张卡扮演两张） | wall **3171.3 µs** = D2H fence 2706.7 + 门铃 0.3 + 背压 3.2（2 次等待）+ CB 回收 17.8（8 次等待） |
| 背压 | 深度 4 出现 **2 次**；深度 1 每 hop 一次主机等待 ⇒ 深度是机制不是旋钮 |
| 拓扑判定（`probe`） | **1 COMPUTE-capable device of 1** ⇒ (A) 层流水线在本机 **NOT EXERCISABLE**；(B) iGPU zero-copy 能力 **AVAILABLE** |

**边界必须照抄，不许省略**：环、DMA 腿、背压、门铃、带宽是实测；**第二张卡的计算流与卡间通道不是**，
本机测到的腿是 D2H → host → H2D。让这套设计值得建的两个数字是更早、独立的：SDMA **7.12 GB/s 且与计算
并发不打折**、单卡每 token 主机气泡 **16.8 ms**（`DEEPINFER_EVAL.md`、`END_TO_END_MEASURED.md`）。

**zero-copy 分支为什么在本机永不触发（`实测`）**：内存类型 2 和 6 确实同时是
`DEVICE_LOCAL | HOST_VISIBLE | HOST_COHERENT`，但在这张独显上那块堆是 **256 MiB 的 BAR 窗口**——放不下
2.343 GiB 权重 arena。所以分支的门槛是 `deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU`，不是
「有这种内存类型就走」。不触发时独显路径一字节未变，历史实测数据继续成立。

### 15.3 GQA 合并的最终数字以配对复测为准（`实测`，`runs/gqa_pair_v1_summary.txt`，batch H）

batch G（首轮、非配对）留在 `BENCHMARK.md` §6 里当**归因仪器**，头条是这一张：

| 位置 | 对照 `GQA_OPT=0` | **on1（默认）** | on3（两半都合） | on1 延迟 Δ |
|---|---|---|---|---|
| cap 256（真实 prefill） | 73.4 ms = 13.63 | 72.7 = 13.75 | **70.9 = 14.10** | −1.0 %（漂移带内） |
| 4 096 | 126.9 = 7.88 | **106.3 = 9.41** | 118.1 = 8.47 | −16.2 % |
| 8 192 | 187.4 = 5.34 | **144.3 = 6.93** | 169.5 = 5.90 | −23.0 % |
| 16 384 | 303.6 = 3.29 | **218.6 = 4.57** | 271.5 = 3.68 | −28.0 % |
| 32 768 | 542.5 = 1.84 | **374.8 = 2.67** | 470.4 = 2.13 | **−31.0 %** |
| 对照末次复跑 | 72.3 = 13.83 | — | — | **同一配置漂移 1.5 %** |

三条必须一起读的结论：

1. **长上下文是真收益**（+19.4 % → +45.1 % tok/s），且 16–31 % 的差距是漂移带（1.5 %）的 10 倍以上。
2. **短上下文是「白送」不是「变快」**：cap 256 的两行（73.4 / 72.7 / 72.3）全在漂移带里，所以说
   「0 % 回归」，不说「+1 %」。默认配置端到端 14.16 tok/s 就是这句话的证据。
3. **PV 那半边不是一个符号**。首轮（batch G）说过「PV 是负的」，配对复测把它改成条件句：在 on1 之上
   再加 PV 合并，4 K **−11.1 %**、32 K **−25.5 %**（亏），cap 256 **+2.5 %**（赚）。默认关 bit 1 是因为
   这个开关的存在理由就是长上下文。机制解释（K 是 gather、V 是连续且 quad 内已共享；12 CU 上常驻
   workgroup 从 16 掉到 4 的代价只在长上下文才压过省下的 ALU）见 `BENCHMARK.md` §6。

拟合层面（`推导`，由 on1 四点最小二乘）：`ms/token = 67.17 + 0.00936 x positions`（最大残差 1.99 ms）。
对照 tiled 的 `69.97 + 0.01408`：**斜率 −33.5 %，常数不变**——这正是 §5.9/`BENCHMARK.md` §5 那句
「注意力只能动斜率」被自己的模型预测、再被实测证实了一次。

### 15.4 两个 Bug 类：这次是真的贵（各吃掉一整天）

**① `VkBufferCreateInfo::pQueueFamilyIndices` 在 `vkCreateBuffer` 期间被读取，但不被拷贝到任何地方。**
数组声明在设置 sharing mode 的 `if` 里，驱动就指着已死的栈帧。表现：`-O0` 侥幸活着，`-O2` 在
`amdvlk64.dll` 内部 SIGSEGV，回溯完全像驱动 Bug。四处调用点全部上提，其中一处在引擎自己的 staging
路径（`bonsai_weights.h` 的 `stage_alloc`，注释在 :315-321）。**纪律**：`-O0`/`-O2` 行为分歧 = 有未定义
行为的证据，嫌疑人先指对象生命周期，不指优化器。**改 `bonsai_weights.h` 必须重跑保真五件套**——已重跑，
结果就是 §15.1 那一行（位相同门 PASS、14.16 tok/s）。

**② 用 `VK_FENCE_CREATE_SIGNALED_BIT` 创建的 fence，在等待执行时已经是 signaled 状态。**
`vkWaitForFences` 直接返回，与本该 arm 它的那次 submit 无关——这是规范行为，不是驱动 Bug。同一份
8 MiB 拷贝、同一队列并排实测：created-SIGNALED = fence **0.8 µs** + 队列排空 **1195.2 µs**；
`flags = 0` = fence **1204.1 µs** + 尾巴 **0.4 µs**（`runs/fleet_ring_v4.log:96` 与 `:94`）。
它在此之前制造了两条现在**公开撤回**的结论：物理上不可能的 **56.67 GB/s**「批处理」速率，以及
「这个 hop 被同步支配」（其实是 DMA 支配）。**纪律**：本仓库每一支 fence 都以 `flags = 0` 创建、只由
自己的 submit arm；环形缓冲另外用**门铃**发布完成——4 B 的 `vkCmdFillBuffer` 与载荷写在同一支命令缓冲、
同一队列，靠队列序保证，不依赖任何 fence 语义。rig 里保留那条 created-SIGNALED 对照，每跑一次就重测一次
这个效应，日志会把 fence 列与门铃列并排印出来（`completion signal audit`）。

### 15.5 27B 架构适配器：门禁已建并全过（`实测`，`runs/regression_v14.log` 的 align_27b 段）

`align_27b.exe <gguf> [--no-gpu]`（源码 `align_27b.cpp`，828 行；plan 与计费在 `bonsai_model.h`）：
64-block 合成 plan（由 9B 真实 header 搭出）= 16 attn + 48 ssm，计费 **49.000 KiB/token**、
cap 32768 时 arena **1568.00 MiB**、循环状态 **100.50 MiB**、SSM 层 KV charge **恰好 0**；被禁止的
「每层都挂 KV」定价 **196.000 KiB/token = 4.0×**（4704 MiB）。旋转路由由 metadata 决定：
`version 1 + block 1024 → hadamard_fwht.comp`，`block 4096` 在同一 plan 上仍走 `hat4096.comp`；
四条负向控制（version 2、block_size 缺失等）**全部按要求拒绝**。通用 FWHT 内核在 block
16/64/256/1024/4096 上与主机 Sylvester FWHT **逐位相同**（4096 lane），N=128 正交性用整数算术精确证明。
`verdict: ALL CHECKS PASSED (0 failure(s))`。

**语义边界**：被证明的是 resolver、计费不变量、旋转契约与内核路由；**没有被证明的是 27B 的文本质量**——
本机装不下 5.9 GB 权重，这正是 §15.2 那两条机制存在的理由（让它们变成「不需要装下」）。

### 15.6 代码量与产物尺寸：只有一个生产者

`python tools/source_size.py`（`实测`，2026-11-05）：
engine = host(10 文件) + shaders(52 `.comp` + 5 `.glslh`) → raw **696 KiB** / 去注释 **416 KiB** /
去空白 **279 KiB**（其中主机 352 KiB、GLSL 64 KiB）；测量工具组另有去注释 **310 KiB**（whole tree 726 KiB）；
`bonsai_vk.exe` **539 KiB**（552,178 B）；**52 支 SPIR-V = 300 KiB**；第三方依赖 **0**。
README、README_zh、BENCHMARK、本文件里的每一个字节数都必须是这个脚本的输出——**脚本会列出自己的文件集，
文件搬家就返回非 0**，所以过期尺寸会当场失败而不是悄悄发出去。历史上这里出现过 374/419/461/53 四组
互相矛盾的数，那就是没有唯一生产者的代价。

### 15.7 STAGE10 之后仍然开着的口子（`BENCHMARK.md` §10 是全集，这里是优先级）

1. **采样器仍未与官方对齐**——文本质量靠保真阶梯加肉眼，不是 sampler diff。这是当前最大的未闭合项。
2. **prefill 还是平方级**：32 K prompt 推导 **≈ 120 分钟**（`推导`：用 §15.3 的拟合 `67.17 + 0.00936 × positions` 对扫描阶梯积分——第 t 个位置要扫 t 个 key，所以 N token prompt 的平均 live keys = N/2，均价 220.5 ms/位置）。**此前这里写的 ≈ 55 分钟是错的**：它把 1310 位置那个阶梯最低点的均价当成了恒定价格，低估 2 倍。方法校验点：同一公式预测 1310 位置 = 73.3 ms/位置，实测 79.2 ms/位置（`runs/gqa_real_c2048_merged_v1.log:113`，+8 %，偏保守方向）。4 K prompt 同口径 ≈ 5.9 分钟。夹具只测 decode，所以 1310 位置以上全是推导。
   批量 prefill 依旧是体验上最大的一块。
3. **40 K / 48 K 是出带推导，没跑过**；两条「4 GB 墙」（我们的 39.2 K arena 安全上限——原先写 38.3 K，是 MiB→KiB 误用 ×1000，已更正——与参考 fork 的 `-c 49152`）
   谁都没被真机推过 32 K。
4. **PV 合并的中间形态**（2 头一组、8 workgroup）没实现——它是唯一有实测方向、但还没被试过的杠杆。
5. **第二张卡**：机械层建好、selftest 过、环实测过；缺的是硬件，不是代码。
6. 默认配置的长上下文阶梯只在 batch H 那一个热窗口里配过对，没有另起一场会话复跑。

### 15.8 收口现在有三条完整性检查：表格、链接、字节（后两条是收口时新增的）

`python tools/md_table_check.py` —— 表格列数（只数未转义竖线、跳过围栏代码块），当前全库 **TOTAL 0**。
`python tools/link_check.py` —— **相对链接是否指得到东西**，逐 (目录, 文档) 组合检查，仓库根与
`dist/bonsai-vk-0.1.0-win64/` 两个上下文都要过。当前 **19 份文档 TOTAL BROKEN 0、rc 0**。
它上线那一次跑出的真问题：包里的 README 副本把 4 份源码路径写成链接，而二进制包没有 `runtime_vk/`
⇒ 包侧 **6 处坏链**。修法是把那 4 份源码按同名路径放进包（不是删链接），代价写在 ROADMAP §3.1。
**给接手的人的教训**：文档被复制进另一个目录之后，"链接在仓库里能点"这个前提就不再成立了；
这类 Bug 人眼读文档永远看不见，只有把断言写成脚本才会当场失败。改完任何 .md：`md_table_check` →
`link_check` → 若动了包内文档就重跑 `MANIFEST.sha256`（`find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256`，
然后在包根 `sha256sum -c MANIFEST.sha256`）。

**第三条：字节也要能穿越 clone。** 本机 `core.autocrlf=true`，git 在 Windows 上 checkout 文本文件时把 LF
重写成 CRLF，所以"按工作树字节算出来的 sha256 清单"在别人 clone 出来的目录里会逐条对不上——这是**实测**，
在一次性空仓库里提交 `alpha\nbeta\n`、删掉工作文件、`git checkout --`，blob 与文件的 sha256 分成
`e49c81e2…` 与 `98ab4d3a…`。本包当时有 16 个文件处在会被改写的那一类。**不改 git 配置**（这条纪律高于方便），
改的是 `.gitattributes`：`dist/bonsai-vk-0.1.0-win64/** -text` ⇒ 加不规范化、出不转换，index == 工作树 == clone 产物。
装配一个包的完整可复现配方（本轮就是这么做的）：

```
cp -f <改动的 .md> dist/bonsai-vk-0.1.0-win64/          # 1. 同步文档副本
python -c "<把包内文本文件 CRLF 归一到 LF，含 NUL 的二进制跳过>"   # 2. 统一字节
cd dist/bonsai-vk-0.1.0-win64 && find . -type f ! -name MANIFEST.sha256 | sort | xargs sha256sum > MANIFEST.sha256
sha256sum -c MANIFEST.sha256                            # 3. 当场核对（本轮 90/90 OK）
git add .gitattributes && git add --renormalize dist/bonsai-vk-0.1.0-win64   # 4. 让 index 也变成 -text
git ls-files --eol dist/bonsai-vk-0.1.0-win64 | awk '{print $3}' | sort | uniq -c   # 5. 期望 91 attr/-text
git checkout-index -a --prefix=<空目录>/ && cd <空目录>/dist/bonsai-vk-0.1.0-win64 && sha256sum -c MANIFEST.sha256
```

最后一条是**唯一的真验收**：导出到空目录再核对，证明的不是"我这儿能过"，而是"clone 的人那儿能过"。
`-text` 之后 `git status` 里如果包内文件又变脏，说明有人往包里拷了新文件却没走上面第 2 步的字节归一。

