# 《Project Bonsai 终极完工报告》

生成时间：2026-11-05 03:30（本机时钟，收口时的最后一笔文档改动）
机器：ASUS 笔记本 / AMD Ryzen 5 3600 / **16 GB 内存** / **Radeon RX 550X 4 GB（gfx803，POLARIS_12）**，
Win10 + WDDM + amdvlk64 2.0.279。零预算、零管理员、零 C 盘依赖。
引擎：`bonsai_vk.exe` **552 178 B**（build_v42，`-Wall` 0 warning）+ `shaders/*.spv` **52 支 307 432 B**。
被加载的 checkpoint：`models/Ternary-Bonsai-2-9B-PTQ1_0.gguf`（1 997 486 976 B，**从未被写过**）。
git：`d5ea17a`（快照）→ `6c67d07`（STAGE10 主收口，160 files，+13 880 / −3 521）→ `05d994f`（发行包与链接
检查器）→ 本文件所在的那一笔是**最后一笔**（`git log -1 --oneline` 就是它；把哈希写进被它自己哈希过的
文件是个循环，所以这里只给取法，工作树应当是 clean 的）。身份一律用每次调用的 `-c user.name` / `-c user.email`，
**从未改动 git 配置**。

标注口径：**〔实测〕**= 某份 `runs/*.log` 里引擎/工具自己打印的数；**〔推导〕**= 由已打印常数算出，
过程写在后面；**〔未验证〕**= 本机做不了，直说做不了。所有数字可被 `tools/md_table_check.py`
+ 下面「复现」一节复算。

---

## 1. 一句话结果

**在一块 2017 年的 4 GB 矿卡上，纯手写的三值专用推理引擎端到端跑出 14.16 tok/s〔实测〕，
是官方参考引擎在同一张卡同一份权重上 4.417 t/s 的 3.21 倍，而且它现在是一个陌生人下载后
能直接跑通、能自己复算保真声明的发行包。**

---

## 2. 交付物清单（路径 = 可直接打开）

### 2.1 可执行产物

| 路径 | 体积〔实测〕 | 是什么 |
|---|---|---|
| `runtime_vk/bonsai_vk.exe` | 552 178 B | 引擎本体：`gen`（权重编排）/ `run`（端到端生成）/ GEMV 基准 / 设备与 arena 报告 |
| `runtime_vk/bonsai_fleet.exe` | 335 456 B | 多卡环测量台：`selftest` \| `probe` \| `ring <D> <slot_MiB> <hops>` \| `split` |
| `runtime_vk/align_tokenizer.exe` | 195 022 B | 逐 ID 对齐（32 probes + 7 个 kr 集） |
| `runtime_vk/align_dequant.exe` | 182 378 B | PTQ1_0 三方反量化比对（真实块 + 合成边缘块） |
| `runtime_vk/align_f16.exe` | 126 037 B | fp16↔fp32 全域与两条着色器表达式逐位 |
| `runtime_vk/align_attn.exe` | 260 683 B | 层 3 注意力 GPU 内核 vs 同序 fp32 镜像 |
| `runtime_vk/align_27b.exe` | 287 063 B | **27B 架构闸门**（PTQ1_0 解码、64-block 路由、SSM 计费 = 0、旋转路由、4 条负控） |
| `runtime_vk/shared_probe.exe` | 131 568 B | 共享内存上限翻案用（每档独立 SPIR-V） |
| **`dist/bonsai-vk-0.1.0-win64/`** | **91 文件 ≈ 6.5 MiB**（字节总数故意不钉死：它是文档之和，每改一次 .md 就变；精确值用包根目录的 `python -c "import glob,os;fs=[x for x in glob.glob('**/*',recursive=True) if os.path.isfile(x)];print(len(fs),sum(os.path.getsize(x) for x in fs))"` 现算） | **发行包**：3 支主 exe + `fidelity/` 5 支（4 支对齐工具 + shared_probe）+ `shaders/` 52 支 + 3 支 MinGW dll + 16 份文档 + `runtime_vk/` 4 份源码（291 355 B，只为把 README 的 6 处相对链接从坏链变成有效链接，见 ROADMAP §3.1）+ `LICENSES/README.md` + `MANIFEST.sha256`（90 条，不含自身，`sha256sum -c` 实测 90/90 OK）+ `tools/` + `runs/` 4 份日志（开箱实测发的是 `package_smoke_v2.log` = 当时 90 文件形态；第 91 个文件是文档，不为它重跑计时） |

源码体积（唯一生产者 `tools/source_size.py`）：**engine = host + shaders，去注释 425 944 B = 416 KiB**〔实测〕，
去空白 279 KiB；测量驱动另计 310 KiB，运行模型不需要。第三方依赖 **0**（链接 `-lvulkan-1 -lpdh -ldxgi`，三支都是系统自带）。

### 2.2 代码（纯 C++17 + 手写 Vulkan，不含任何框架一行）

`runtime_vk/bonsai_forward.h` **3268 行**（前向 + 生成 + 位相同门 + 显存预算 + OOM 兜底 + 27 个开关入口）
· `bonsai_weights.h` 837 行 · `bonsai_model.h`（plan/计费/Hadarmard 路由）· `bonsai_vk.cpp` 1906 行 ·
`bonsai_tok.h` / `bonsai_gguf.h` / `bonsai_chat.h` 105 行 · `gpu_fleet.h` + `bonsai_fleet.cpp` ·
`shaders/` **52 支 .spv ← 57 支源（52 `.comp` + 5 `.glslh`）**。

### 2.3 文档（每份都是可审计的，不是宣传）

`README.md`（英）· `README_zh.md`（中）· `BENCHMARK.md`（全部实测数字 + 日志行号）·
`ROADMAP.md`（今日水位 / 发行包**已装配**清单 / P0–P2 排序 / 验收清单如实状态 / 不能修清单）·
`HANDOVER.md`（14 节 + §15 STAGE10 新事实 + 产物索引）· `FAITHFUL_REPRO.md` ·
`ATTENTION_ALIGN_FINAL.md` · `KV_P0P1P2_MEASURED.md`（§9 = STAGE9 长上下文）·
`END_TO_END_MEASURED.md` · `DEEPINFER_EVAL.md` · `ANNOUNCEMENTS.md`（16 条推文 + 1 篇长文草稿）·
`STAGE0/1/2/3_*.md` · **本文件**。

### 2.4 关键日志（`runs/`，只新增、不覆盖）

| 文件 | 内容 |
|---|---|
| `runs/regression_v14.log` | 当前发布件全量保真：tokenizer 32/32 + 7/7、dequant `max\|err\| = 0.00000` + 128 000/128 000 逐位、f16 全域 0 wrong、attn score 3.567e-03、`align_27b ALL CHECKS PASSED`（:74）、**`gate: PASS`**（:94）、**14.16 tok/s**（:95） |
| `runs/gqa_pair_v1_summary.txt` | **GQA 配对阶梯的权威读数**（同二进制，4 长度 × {off,on1,on3} + cap 256 三行 + 复跑对照） |
| `runs/package_smoke_v2.log` | **从发行包根目录**实跑的最终 90 文件形态：rc=0、`gate: PASS`、14.31 tok/s（8 token / 559.1 ms）、cap 256 活动区 58.2 MiB、PDH dedicated 2539.23 MiB、权重装载 1099.9 ms |
| `runs/package_smoke_v1.log` | 同一支 exe 对上一形态（86 文件）的开箱跑：判定量与 v2 **全同**，只差墙钟（装载 1201.0 ms、bring-up 716.7 ms）。两份都留着的理由就是纪律本身：只新增、不覆盖 |
| `runs/fleet_ring_v4.log` | 环 55/55 selftest、单腿 6.96 GB/s、批处理 7.13 GB/s、背压 2 次、created-SIGNALED 对照 |
| `runs/oom_fallback_v3.log` | OOM 三分支（A exit 0 / B、C exit 4 + `NOTHING WAS DISPATCHED`） |
| `runs/build_v40.log` `…v41` `…v42` | 三次连编全部 exit 0、`grep -ciE "warning\|error"` = **0** |
| `runs/align_27b_v1.log` `…v2` | 27B 闸门首轮与负控复跑 |

### 2.5 新工具（这一轮写的，都还能用）

`tools/md_table_check.py` —— 只数**未转义**的 `|`、跳过围栏代码块，逐行比对表头列数；当前全库 **TOTAL 0**。
`tools/link_check.py` —— 本轮为发行包写的第二个完整性检查器：对每个 (目录, 文档) 组合解析相对 `]()` 链接，
仓库根与包根**两个上下文都要过**（先剥掉围栏代码块，所以示例/命令里的伪链接不会被误计）。当前 **19 份文档
TOTAL BROKEN 0、rc 0**；它上线的那一次跑出过包侧 6 处坏链 —— 见 §4 第 8 条。
`tools/source_size.py` —— 体积数字的唯一生产者，文件被动过就非零退出；README/BENCHMARK/ANNOUNCEMENTS 里
每一个「多少 KB」都必须在它输出里找得到。
`tools/clear_field.ps1` / `tools/vram_sample.ps1` —— 基准前清场、PDH 显存采样（最小间隔 1 s）。
`.gitattributes`（仓库根，本轮新增）—— 第三条完整性机制，也是唯一管**字节**的：
`dist/bonsai-vk-0.1.0-win64/** -text` + 对 `.spv/.exe/.dll/.gguf/.safetensors` 的显式 `-text`。
它让 `MANIFEST.sha256` 从"我这台机器能对上"升级成"clone 出来还能对上"，配方与验收写在 HANDOVER §15.8、
失败原因与实测哈希对写在 ROADMAP §3.1 与本报告 §4 第 9 条。**没有动 `core.autocrlf`**——规矩写在文件里，
不写在本地配置里，这样它跟着提交走而不是跟着这台机器走。

---

## 3. 关键数据（任务书四项逐条对账）

### 任务一：干掉 GQA 的 4× 冗余，同时 0 错误契约不动 —— **完成**

| 项 | 值 | 口径 |
|---|---|---|
| 默认点火 | **14.16 tok/s**（48 token / 3389.5 ms；稳态 70.6 ms、首 token 70.2 ms） | 〔实测〕`regression_v14.log:95` |
| 短上下文回归 | **0 %**（默认仍 narrow + 合并 QK；带内 14.05–14.27） | 〔实测〕 |
| 4 K / 8 K / 16 K / 32 K（合并 QK 开） | **9.41 / 6.93 / 4.57 / 2.67 tok/s** | 〔实测〕`gqa_pair_v1_summary.txt` |
| 同档关掉合并 | 7.88 / 5.34 / 3.29 / 1.84 tok/s ⇒ **+19.4 % / +29.8 % / +38.9 % / +45.1 %** | 〔实测〕 |
| 拟合 | wide `71.12 + 0.01443x` → tiled `69.97 + 0.01408x` → **合并 QK `67.17 + 0.00936x`** | 斜率 **−33.5 %**、常数几乎不动〔实测+推导：对各档 ms/token 最小二乘〕 |
| 为什么常数不动 | 合并消掉的是**每 kv 头被 4 个 q 头重复读 K** 的量（grid 16→4），不是启动开销 | 机理〔推导〕，与斜率单独下降一致 |
| **运行时开关** | `BONSAI_GQA_OPT` 0 off / **1 auto（默认）** / 2 force-at-every-cap；`BONSAI_GQA_PART` bit0=QK、bit1=PV（默认 **1**）；`BONSAI_GQA_TILE` 256/512/1024 | 〔实测：三档都在硬件上跑过〕 |
| 位相同门 | **A 窄↔宽、A2 宽↔tiled、A3 合并 QK↔被替换内核、A4 合并 PV↔冻结 q8v**，B/C/D/E 结构门；任一 bit 不同 ⇒ `rc 6` 不生成 | 〔实测〕`gate: PASS (bitmask 0 = A A2 A3 A4 B C D E)` |
| **PV 合并是双相的（重要取舍）** | cap 256：72.7 → 70.9 ms（**+2.5 %**）；cap 32768：374.8 → 470.4 ms（**−25.5 %**） | 〔实测〕⇒ **默认只合 QK**。合"全开"会把长上下文做慢 |

已验证的五条谱系（narrow / wide / tiled / 冻结 q8v PV / 冻结 FWHT 侧）**一字节未改**；
`attn_qk_softmax_wide.spv` 的 md5 仍是 `3a2c4e59…`。

### 任务二：27B 架构「绝对正确」的代码 + 产物 —— **代码与闸门完成；真 checkpoint 本机跑不动，直说**

| 项 | 值 | 口径 |
|---|---|---|
| 路由 | 64 block：**16 全注意力（计费 KV）+ 48 循环状态（KV 计费 = 0）** | 〔实测〕`align_27b`；SSM 挂 KV 是**被断言拒绝的不变量**，不是约定 |
| KV 定价 | **49.000 KiB/token**，cap 32 768 ⇒ arena **1568.00 MiB**；被禁止的"每层挂 KV" = **196.000 KiB/token = 4.0×** | 〔实测，引擎打印〕 |
| 循环状态 | 100.50 MiB / 序列（24 GDN 层的定尺状态，与 cap 无关） | 〔实测〕 |
| Hadarmard 契约 | `prism.hadamard.version` 决定要不要转；`block_size` 在 9B 位冻结 `hat4096.comp` 与通用 `hadamard_fwht.comp` 之间路由 | 〔实测〕`hadamard_fwht` 在 block 16/64/256/1024/4096 与主机 Sylvester FWHT **逐位相同**；4 条负控全部如期拒绝 |
| 严格性 | PTQ1_0 = **5 trits/byte、每 128 权重一个 FP16 scale**（28 B/128 = 1.75 bpw 打包） | 〔实测〕1000 真实块 / 128 000 权重三方 `max\|err\| = 0.00000` |
| **不能声称的** | 本机 4 GB 跑不动真实 5.9 GB 的 27B checkpoint | 〔未验证〕被证明的是 **resolver、计费不变量、内核路由**，形状来自一张由 9B 真 header 搭的合成 64-block |

### 任务三：多卡 layer 流水线 + iGPU 零拷贝 —— **骨架 + 测量台完成并实测；第二张卡本机不存在**

| 项 | 值 | 口径 |
|---|---|---|
| 环 selftest | **55 checks / 0 failed**，0 字节不符，不创建任何 Vulkan 对象 | 〔实测〕`fleet_ring_v4.log` |
| 单腿 8 MiB D2H（transfer-only family 2） | **1204.6 µs = 6.96 GB/s = 实测 7.15 GB/s 上限的 97 %**（含 submit、fence、排空尾巴） | 〔实测〕 |
| 8 腿一批 | **1176.4 µs/腿 = 7.13 GB/s** | 〔实测〕 |
| 每 hop 墙钟 | 3171.3 µs；D2H fence 2706.7 µs、门铃 0.3 µs、背压 3.2 µs / 2 次、CB 回收 17.8 µs / 8 次 | 〔实测〕 |
| 背压 | 深度 4 时 2 次事件（深度 1 = 每 hop 一次等待）⇒ **D>1 是机制不是调参** | 〔实测〕 |
| 零拷贝门控 | `VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU` ⇒ 强制 `HOST_VISIBLE\|DEVICE_LOCAL`、绕过 staging | 〔实测代码路径 + 判定〕，**〔未验证〕在真 iGPU 上的收益**：本机 `probe` 报 **1 个 COMPUTE-capable device / 共 1** |
| 卡对卡那一跳 | **本机不可演练**，实测的是 D2H→host→H2D 往返 | 〔未验证，已在包与文档里明写〕 |
| 单生产者 sizing | 每 hop 8 MiB 槽 ⇒ 环深度 × 2 leg 决定锁页量；416 KiB 是门铃+元数据侧的固定开销口径 | 〔推导，已在日志里逐项打印〕 |

### 任务四：删死代码 + 全库 0 warning + 双语 README + 对外文案 —— **完成**

- `build_mingw.bat` 连编三次（`build_v40/41/42.log`）**exit 0、0 warning**，产 52 SPIR-V + 8 exe。
- 死代码：GEMV 探针时代的 `probe_gemv*.py` 7 支移到 **`probes/early_failures/`**（保留可读的错误假说历史，
  不是删除）；`*.bak_*` 4 支、`build_fwd*.log` 6 支、`out_base.txt`、`shaders/comp.spv`、`shaders/fwht.*`、
  `bonsai_vk_dbg.exe` 已删。
- `README.md`（英）+ `README_zh.md`（中）：30 秒上手、实测表、两条驱动教训、位相同门为什么存在、
  **CWD 硬规矩**（见 §4 第 5 条）。口号按实测口径：「纯手搓 416 KiB 去注释源码，只为三值大模型而生；
  4 GB 矿卡跑 9B @ 14.16 tok/s」。
- `ANNOUNCEMENTS.md`：**16 条推文，逐条重新校验 ≤ 280 字符（当前 0 条超限）+ 1 篇长文草稿 259 字符**。**没有发布**
  —— 发推文/发 Reddit 需要用户点头（且本机没有授权过的账号通道）。

### 附加：显存与长度（这一轮把口径彻底钉死）

| 项 | 值 |
|---|---|
| 权重已提交 | **2467.7 MiB**（2.410 GiB；随机访问 2-bit 形态 2 516 189 184 B = codes 2.083 GiB + scales 0.260 GiB = **2.125 bpw**）〔实测〕 |
| 活动区 | **58.2 MiB** @ cap 256、**101.339** @ 2048、**841.261** @ 32 768〔实测，三档都由引擎自打印；256 那档 = 52.0 与 cap 无关 + 256 × 24.66 KiB〕 |
| PDH dedicated | **2539.23 MiB** @ cap 256（= 2467.7 + 58.2 + **13.3 MiB 驱动收走、预算表没建模**）、**3323.84 MiB** @ 32 K；两档 `Shared Usage` 增长 **+0.00**〔实测〕 |
| 每槽 / 固定 | **24.66 KiB/slot**（q8_0-V + fp16-K 的 arena 24.50 + fp32 侧 0.16）、**52.0 MiB** 与 cap 无关〔实测〕 |
| 上限 | 安全 **≈ 39.2 K**（`[(3464.6 − 2467.7 − 52.0) × 1024] ÷ 24.66 = 39 237`，DXGI LOCAL 预算）；极限 **≈ 54.8 K**（原始 3840 MiB heap = 54 825；引擎在 cap 256 就报 `headroom 1314.1 MiB = 54 558 more slots`）〔推导，全部用引擎打印常数〕 |
| 权重装载 | 冷 **14.05–14.44 s**（四次独立首读）/ 热 **1051.9–1201.0 ms**〔实测；页缓存状态夹具不控制，所以两个口径都列〕 |
| prefill | **1310 token / 103 766.4 ms = 12.62 tok/s = 79.2 ms/位**〔实测〕；积分到 4 K **≈ 5.9 min**、32 K **≈ 120 min**〔推导：第 t 位扫 t 个 key ⇒ N token 平均 live keys = N/2，均价 220.5 ms/位；同式预测 1310 位 = 73.3 vs 实测 79.2 ms，**+8 %、偏保守**〕 |
| 短上下文对大 arena 不敏感 | 16-token prompt 下 cap 256/4096/16384/32768 = 70.4/70.4/70.7/70.7 ms（**+0.4 %**）〔实测〕 |

---

## 4. 这一轮改错的 5 + 4 件事（每条都留了「这里以前是错的」注记，没有静悄悄改）

1. **`1985.3 MiB` / `16.05 s` 不是我们的数** —— 那是 STAGE5 官方 fork 的 WDDM 峰值与冷加载。文档曾把它们当本引擎的成就是引用。现已拆成三行：本引擎占用、本引擎装载、参考引擎的数并注明"永远不许当成我们的"。
2. **GEMV 带宽** —— 旧文档引单个历史点当事实；现在给带：**173.6–177.0 Gop/s**（`build_v31…v42` 十二次连编），最新三次 177.0/176.9/176.9 ⇒ 引 176.9 Gop/s、43.4–44.2 GB/s。
3. **显存上限算错 2.5 %** —— 旧表把 MiB→KiB 用了 `×1000`。正确 **≈ 39.2 K**（原写 38.3 K）。两条独立回算（1723 槽 → 41.5 MiB、9915 槽 → 238.8 MiB）与之吻合；**"40 K 出带 / 48 K 超 239 MiB"两个结论不受影响**。
4. **prefill 的 55 分钟是线性外推的错** —— 拿阶梯最低点当恒定价，低估 2 倍。积分口径 **≈ 120 min @ 32 K**，并留了被撤数字。
5. **「SPIR-V 内联进 exe」是假的** —— ROADMAP §3.1 上一版这么写。装配发行包时被打死：`make_pipeline` 读 `std::string("shaders/")+name+".spv"`（`bonsai_forward.h:339`、`bonsai_weights.h:408/413-415`），**且路径相对当前目录，不是 exe 目录**；缺 `shaders/` 时打印 `cannot open shaders/…` 并停下。⇒ shaders 必须随包；README/README_zh 的命令块已全部改成"先 cd"。
6. **「四件套」现在是五件套** —— `align_27b` 加入后，规范条款、第二驱动栈清单、`bonsai_weights.h` 改动纪律都改成五件套；历史行保留旧称并注明它们当时确实只有四条。
7. **行号引用会漂** —— `bonsai_forward.h` 从 1990 → 3268 行，让 HANDOVER §3.1 表里 8 处引用失效。已按符号重核（`Regions:56`、`Fwd:107`、GQA 默认 `:156`、`dsp:346`、`attn_block:551`、`qk_reference_selection:2096-2101`、预算表 `:1691-1766`、`Sampler:2493`、`fwd_run:2551`、`fwd_main:2947`、cap 拒绝 `:3103/3145/3149`、内核选择打印 `:3151`；`bonsai_vk.cpp` 查询池 `:371-372`；`bonsai_chat.h` control id `:41-44`），并在文档里写了「**认符号，别认行号**」。

8. **发行包里的 README 有 6 处坏链** —— 包是从仓库拷出的 README 副本，而它们把 `runtime_vk/gpu_fleet.h`、
   `bonsai_fleet.cpp`、`bonsai_model.h`、`align_27b.cpp` 写成**相对链接**；二进制包里没有 `runtime_vk/`，
   所以收包的人点进去全是 404。这条不是靠读文档发现的，是新工具 **`tools/link_check.py`** 逐 (目录, 文档)
   解析相对链接发现的：装配前跑 = 仓库侧 11 份文档 ok、包侧 6 处坏链（rc 1）；把 4 份源码按同名路径放进包后
   重跑 = **19 份文档 TOTAL BROKEN 0**（rc 0）。修法选了"补文件"而不是"删链接"，因为仓库上下文里那些链接有用。
   代价也直说：包从 86 个文件变成 90 个（+291 355 B 源码），而它**仍然是二进制包** —— 引擎自己的
   `bonsai_forward.h` / `bonsai_vk.cpp` / `shaders/*.comp` / 构建脚本没进来，那叫另一件事，没假装做了。

9. **`MANIFEST.sha256` 只在拷贝的那一侧成立** —— 本机 `core.autocrlf=true`，git 在 Windows 上 checkout
   文本时会把 LF 写成 CRLF。一次性空仓库实测：提交 `alpha\nbeta\n` → 删工作文件 → `git checkout --`
   ⇒ blob sha `e49c81e2…`、文件 sha `98ab4d3a…`、字节 `b'alpha\r\nbeta\r\n'`，**同一个文件两个哈希**。
   本包当时有 16 个文件正处在这一类（另有 9 份 .md 副本碰巧已是 CRLF，属于侥幸对上）。⇒ 一个 clone
   的人跑 `sha256sum -c` 会大面积失败，而这条"清单可核对"是发货声明的一部分。修法**不是改 git 配置**
   （那条纪律高于方便），是新增 **`.gitattributes`**：`dist/bonsai-vk-0.1.0-win64/** -text`
   （入不规范化、出不转换 ⇒ index == 工作树 == clone 产物），90 个文件全部 `attr/-text`，包内文本统一 LF，
   清单重算 89 条。**验收用的是导出而不是本机**：`git checkout-index -a` 到空目录 → 进导出包根 →
   `sha256sum -c` ⇒ **90 文件、89/89 OK、rc 0**（实测）。配方写在 HANDOVER §15.8。

**刻意不做的一件事**：没有为了"改一句 printf / 一行注释"重编译。那会让已签字的 552 178 B 二进制和它的全部引文作废。改文档不需要重编，改代码需要重跑五件套 —— 这条纪律写进了 ROADMAP §7。

---

## 5. 仍然开着的口子（不粉饰）

| # | 项 | 状态 |
|---|---|---|
| 1 | **采样器与官方对齐** | 未做。STAGE10 之后**最大**的未闭合项：top-k/top-p/温度/eos 的边界行为没有逐 ID 对照过官方。 |
| 2 | chat template 后的 **id 序列 == 官方逐 ID** | 引擎内两条渲染路径（`encode` vs `render_ids`）实测 IDENTICAL，但**与官方 `llama-tokenize -chat-template` 的逐 ID 比对未做**（HANDOVER §7.1 D4）。 |
| 3 | cap **1024 / 2048** 两档的显存实测表 + 「长 prompt × 三档」 | cap 256 与 32 K 已实测；中间档缺。（`runs/oom_fallback_v3.log` 里 36.16 KiB/槽 那一版是 build_v15 的读数，已作废。） |
| 4 | **第二驱动栈**（RADV / 第二块卡）上的 `align_*` 五件套 | 本机没有第二块卡 ⇒ 未做。`packHalf2x16` 向零取整 与 NULL 查询池 SIGSEGV 是**驱动特定**事实，发布前必须换栈复跑。 |
| 5 | 卡对卡那一跳、真 iGPU 上的零拷贝收益、256 MiB BAR 是否够 | 本机只有 1 个 compute-capable device ⇒ 未验证。 |
| 6 | 真实 27B / 30B checkpoint 端到端 | 未下载、本机也放不下；已证的是架构契约与计费不变量。 |
| 7 | 位相同门在 **>65 536 行**（cap > 32 K arena）之外 | 没测过，也没人该声称它能跑。 |
| 8 | P0-1（每 token 一次 submit 的 −12.5 %）、P0-2（70.6 → 64.3 ms ⇒ 15.55 tok/s，+10.1 % **推导**） | 排序与依据在 ROADMAP §4，**未实现**。 |
| 9 | `BENCHMARK.md` 里 `runs/*.log` 行号引用 | 抽查过多处，但没有一份"全库行号自动校验"工具 —— 现在的保证是"每条都被手工打开原文件对过"。 |

---

## 6. 复现（照抄即可）

```bat
:: 引擎（必须先 cd —— 着色器按当前目录加载）
cd /d runtime_vk
set BONSAI_QCHECK=1
bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 48

:: 发行包（包根即运行目录）
cd /d dist\bonsai-vk-0.1.0-win64
set BONSAI_QCHECK=1
bonsai_vk.exe run models/Ternary-Bonsai-2-9B-PTQ1_0.gguf 32 8

:: GQA 配对阶梯（一次批处理，同一二进制）
batch_g.sh v1 & batch_h.sh v1        :: 读 runs/gqa_pair_v1_summary.txt

:: 闸门与体积
python ..\tools\md_table_check.py    :: 期望 TOTAL 0
python ..\tools\source_size.py       :: 期望 552,178 B / 52 modules / 0 deps
```

纪律（已写进代码与文档）：同一时刻只允许 **1 个 GPU 进程**；每次计时前先 `tools/clear_field.ps1`；
日志只新增 `runs/<名字>_v<N>.log`，**不覆盖旧日志**；任何按 cap 增长的主机侧回读必须显式受 staging 尺寸约束。

---

## 7. 结论

四件事里，**任务一、任务四完全落地并实测**；**任务二落地到"闸门全过、真卡跑不动"这个诚实边界**；
**任务三落地到"骨架 + 测量台 + 单卡可测部分全过，双卡那一跳本机不可演练"**。
引擎的默认配置今天在一块 4 GB 的 2017 笔记本矿卡上是 **14.16 tok/s、0 bit-diff 门全过、≈ 6.5 MiB 的包开箱可跑**。

下一步真值得做的两件事，按 ROI：**① 采样器对齐**（它是"输出对不对"里唯一还没被逐 ID 钉死的一环）；
**② P0-1 的提交侧 −12.5 %**（分母已经量清了：70.6 ms 里 GPU 63.6、host bubble 8.49、内存 6.27）。
剩下的长上下文收益要靠 **半合并（HPK=2，8 workgroups）** 而不是新带宽 —— 32 K 的账已经证明它不是带宽问题。

（本报告生成后按任务书自动关机：`shutdown /s /t 120`，取消命令 `shutdown /a`。）
