# 任务四 — 开源产品化 Roadmap（Release 包 + OOM 兜底 + 按实测收益排序的优化清单）

日期 2026-11-05（STAGE10 状态刷新；§2、§4、§6、§7 的方法与推导自 2026-11-04 起未变）。
机器 RX 550X 4 GB (gfx803) + R5 3600，MinGW-only，零预算，无管理员。
标注约定：**实测** = 本机跑出来的；**推导** = 由实测常数按给出的公式算；**未验证** = 假说，附判定实验。

> **发布件 = `bonsai_vk.exe` 552,178 B（539 KiB）+ 52 支 SPIR-V（307,432 B = 300 KiB），2026-11-05 01:48**
> （`runs/build_v42.log`：exit 0、`-Wall` **0** warning；v40/v41 同口径，三次连编）。
> 体积的唯一出处是 `python tools/source_size.py`（去注释源码 416 KiB / 原样 696 KiB / 再去空白 279 KiB），
> 本文**不再**独立写任何字节数。
> 它自己重跑过的验收：`runs/regression_v14.log`（tokenizer 32/32 + 7/7 byte-identical、PTQ 反量化三方
> `max|err| = 0.00000` 且 128000/128000 逐位、f16 全 binary16 域 0 wrong、层 3 score 3.567e-03 @ seq 64、
> 27B 闸门 `ALL CHECKS PASSED`、`BONSAI_QCHECK` 位相同门 `gate: PASS`、**14.16 tok/s** / 稳态 70.6 ms/token）
> 与 `runs/oom_fallback_v3.log`（OOM 三分支，B/C 都 exit 4）。
> §2.2 的引文是**有省略的摘要**（driver 读数行、arena 行、options 行未抄），逐行原文以日志为准。
>
> **本文件此前那版引用的是 `build_v15.log` / `regression_v13.log` / 419,401 B / 14.05 tok/s，属于
> build_v15 时代的历史水位，不是当前发布件。** 保留它们只为说明跨二进制 provenance：`build_v15.log`
> 是「四件套首次在最终二进制上重跑」那一次，`regression_v13.log` 是它的验收；从 v15 到 v42 之间
> dequant / tokenizer / f16 三条路径**一字节未改**，所以 v13 的那些读数与 v14 的同项读数一致，不是被推翻。
> 早先草稿里的 `build_v14.log` **在磁盘上并不存在**（那次构建没落日志）——这一条更正仍然有效。
> `ignition_v3.log`（14.04）仍在 §1/§5 出现，标注为「同一代码状态的另一次」，不是成绩更正。

---

## 1. 今天就能发布的东西，和它此刻的真实水位

| 能力 | 状态 | 证据（实测） |
|---|---|---|
| 原生 Vulkan 单文件引擎，无 llama.cpp 依赖 | 可用 | `runtime_vk/bonsai_vk.exe`（539 KiB）+ 52 支 SPIR-V；只链 `vulkan-1` + `pdh` + `dxgi`，三个都是系统给的 |
| 忠实复现（token 级 1:1 语义，不是近似） | 已闭合 | **当前发布件实测**（`runs/regression_v14.log`）：tokenizer 32/32 + 7/7 = **39/39** byte-identical；PTQ 反量化 1000 个真实块三方 `max\|err\| = 0.00000` 且 **128000/128000 逐位**；f16 全 binary16 域 **0 wrong numbers**、两条着色器表达式逐位相同；层 3 注意力 score 最大误差 3.567e-03（seq 64，= 官方 fp16 缓存自身代价的 0.26 倍）。合成边缘块 491520 权重 0 差在 `runs/dq_synth.txt`（**2026-11-01 的实测**，此后 dequant 路径没动过，但**没在新二进制上重跑**——引用它时要说清是哪一次） |
| 完整对话（chat template + 思考块开关） | 可用 | `bonsai_chat.h`、`BONSAI_CHAT/THINKING/STRIP_THINK`，`runs/chat_en_v1.log` 等 |
| 上下文上限 | **验证到 32 768；算术上限 ≈ 39.2 K（推导，安全带）/ 54.8 K（极限，按原始 heap）** | 窄版 `kMaxNarrow = 256`、宽版 `kMaxWide = 2048`（= 着色器 `MAXCAP`，是**验证跑到的距离**，不是驱动许可，`bonsai_forward.h:184-191`）；`cap > 2048` 自动走 tiled，tiled 与合并内核**根本没有 MAXCAP**（`bonsai_forward.h:129/286`）。位相同门 A/A2/A3/A4 覆盖到 **1310 个真实 live key、跨六个 tile**（`runs/gqa_real_c2048_merged_v1.log:119-129`）。40 K / 48 K 是**出带推导**，没跑过 |
| 速度 | **14.16 tok/s**（cap 256，默认档 = GQA 合并开）/ 稳态 70.6 ms/token / 首 token 70.2 ms | `runs/regression_v14.log:95-96`。长上下文配对阶梯 4 K **9.41** / 8 K **6.93** / 16 K **4.57** / 32 K **2.67 tok/s**（`runs/gqa_pair_v1_summary.txt`） |
| 显存 | **本引擎**（不是参考引擎）：已提交权重 **2467.7 MiB** = 2.410 GiB（其内三值打包体 2.343 GiB = codes 2.083 + scales 0.260，2.125 bpw）+ 活动区 cap 2048 **101.339 MiB**（实测）/ cap 32768 **841.261 MiB**（实测）；32 K 时 PDH dedicated 峰值 **3323.84 MiB**，`Shared Usage` 增长 **+0.00**（无 WDDM spill）。参考 fork 的 1985.3 MiB 是**另一台账**（它只把 1746 MiB 模型放上卡），不要当我们的引用 | `runs/gqa_real_c2048_merged_v1.log:45,48`、`runs/kvfx_c32768_q2048_v1.log:46-48,108,124`、`END_TO_END_MEASURED.md:20`；每槽位现按 **24.66 KiB**（q8_0-V + fp16-K 24.50 + fp32 侧 0.16），见下面的口径警告 |
| OOM 兜底 | 可用（已真机三分支验证） | 见 §2；**注意 §2.2 那张预算表是 `build_v15` 时代、fp16 K/V 未压缩那一次的读数**（36.16 KiB/槽）。V 改 q8_0 后 per_slot 必然变小，**没有在 v42 上复跑过预算表**，所以 §2 的数字要按「该次运行的实测 + 代码路径仍然成立」来读 |
| 长 prompt 体验 | **最差的一块** | **同一份当前二进制**实测：1310-token prompt 的 prefill **103 766.4 ms = 12.62 tok/s = 79.2 ms/位置**（`runs/gqa_real_c2048_merged_v1.log:113`，推导 103766.4÷1310），而这 79.2 ms 本身是**楼梯的平均值**（第 *t* 个位置要扫 *t* 个 key，所以平均 live keys = N/2），不能直接和 cap 256 的 70.6 ms 比大小。按 §6 的拟合 `67.17 + 0.00936 × positions` 积分：**4 K prompt ≈ 5.9 min，32 K prompt ≈ 120 min**（均价 220.5 ms/位置）。**此前本文写的 ≈ 55 分钟是错的**（把楼梯最低点的均价当恒定价，低估 2 倍）；方法的校核点就是这一条实测：预测 1310 位罣 73.3 ms/位置 vs 实测 79.2 ms/位置（+8 %）。旧二进制上 1579 token = 131.99 s（`runs/qcheck_2048_v2.log:93`）同量级 |

一句话：**能跑、能对话、能到 32 K、比官方 fork 快 3.21 倍（14.16 vs 4.417 t/s，实测），但首 token 延迟是唯一会吓跑用户的地方。**

---

## 2. 已落地的 OOM 兜底（本次会话，真机三分支验证）

### 2.1 改了什么

`runtime_vk/bonsai_forward.h`（`fwd_init`，激活 arena 提交之前）新增 ~50 行：

1. 把 arena 的申请额**拆成两半并报出来**：与 cap 无关的固定部分（recurrent state + logits +
   split-K scratch + 固定区域）与 `cap × 每槽位字节数`。`per_slot` 是那些随 cap 变化的区域的**符号和**，
   `fixed = 实测总量 − per_slot×cap`，所以两半相加**恒等于**实际申请额（不是估算）。
2. `headroom`（成功后）与两条**拒绝路径**（失败后）：① KV 不够 → 直接给出可用的最大 cap；
   ② 连固定部分都放不下 → 给出「这个模型至少需要多少 MiB」。
3. 两条拒绝路径都在 `vkAllocateMemory` **之前**返回，日志明写 `NOTHING WAS DISPATCHED`，
   退出码 4，不留半初始化状态。
4. 顺手修了一个**已存在**的假数：`bonsai_weights.h` 以前把字面量 `0` 传给
   `model_plan_report`，于是每条 bring-up 日志都印 `device-local heap: 0 MiB`，而这张卡是 3840 MiB。
   现在走新的 `device_local_heap(Ctx&)`（`bonsai_weights.h:117`）。
5. 加了测试旋钮 `BONSAI_HEAP_LIMIT_MIB`（只在日志里以 `[TEST KNOB]` 声明它生效）。
   为什么必须有它：**这张卡上任何合法 cap 都不会溢出**（cap 2048 只需 124 MiB，堆还剩 1247.9 MiB），
   所以兜底路径本来**从没在真机上执行过一次**。钳住预算才谈得上断言。

### 2.2 实测输出（`runs/oom_fallback_v3.log` = 最终二进制 build_v15；`oom_fallback_v2.log` 是同一代码状态减两条注释的那一次）

```
--- A: cap 2048，驱动真实堆，48 token --- exit_A=0
  memory budget | device-local heap 3840.0 MiB, weights 2467.7 MiB already committed
    activations at cap 2048 : 124.3 MiB = cap-independent 52.0 MiB + 2048 slots x 36.16 KiB/slot
                              (fp16 K/V 32.00 + fp32 q/k+score/prob 4.16)
    headroom after this allocation: 1247.9 MiB (= 35335 more slots of cap, but the widest
                                     VALIDATED attention row is 2048)

--- B: cap 2048 + BONSAI_HEAP_LIMIT_MIB=2580 --- exit_B=4
  OOM FALLBACK - this cap does not fit, and NOTHING WAS DISPATCHED (no half-built state).
    short by 12.1 MiB / heap-weights 112.3 MiB / cap-independent 52.0 MiB
    left for the KV 60.2 MiB, at 36.16 KiB per slot -> 1704 slots
    so the usable cap here is 1704; rerun with BONSAI_CAP=1704 (even, or lower).

--- C: cap 256 + BONSAI_HEAP_LIMIT_MIB=2500 --- exit_C=4
  OOM FALLBACK - no cap fits at all, and NOTHING WAS DISPATCHED.
    this model therefore needs at least 2519.8 MiB of device-local memory; it cannot run in 2500.0 MiB.
```

**自洽校验**：A 的两个分量解出了此前独立测过的两个数——
cap 256 → `52.0 + 256×36.16 KiB = 61.05 MiB`（实测 61.052），
cap 2048 → `52.0 + 2048×36.16 KiB = 124.34 MiB`（实测 124.339）。
每槽位 36.16 KiB 里 fp16 K/V 占 32.00 KiB = `8 层 × 2 × 4 kv头 × 256 × 2 B`，
与 `bonsai_model.h` 独立打印的 `32768 B/token` 一致。**实测**。

### 2.3 兜底还差的两步（**未做**，按优先级）

1. **自动降档**（roadmap 原 §7.4 的 ①）：现在只**报出**可用 cap，不**改**它。
   原因是 `fwd_init` 的区域表是在一次直线代码里 `take()` 出来的，改 cap 意味着整段可重入；
   这需要把区域编排抽成一个函数，属于结构改动，不适合和预算表挤在同一次提交里。
   验收方式已经有了：B 分支的 `1704` 就是自动降档应当取的数。
2. **把别的进程的占用算进预算**：`device_local_heap` 是**堆的总大小**，不是**剩余大小**；
   桌面/别的 Vulkan 进程占用的部分只能靠 `vkAllocateMemory` 真失败才知道，
   所以现在「成功」的路径可能仍落到 `ar.commit()` 的第二句拒绝信息上（那条也在）。

---

## 3. Release 包（这是可以立刻做的，工作量 0.5 天）

### 3.1 目录与文件

**本节从「计划」改成「已装配」**：包体现在真在 `dist/bonsai-vk-0.1.0-win64/`，**91 个文件 ≈ 6.5 MiB**（第 91 个是 `PROJECT_BONSAI_FULL_DOSSIER.md`，纯文档）；
`MANIFEST.sha256` 有 90 条 = 除自身之外的每一个文件。字节总数**故意不钉死**——它是文档之和的一部分，
每改一次 .md 就变，留一个精确值只会制造下一份腐烂的数字。要精确值就在包根目录跑这一条（实测可用）：
`python -c "import glob,os;fs=[x for x in glob.glob('**/*',recursive=True) if os.path.isfile(x)];print(len(fs),sum(os.path.getsize(x) for x in fs))"`，
然后在包根 `sha256sum -c MANIFEST.sha256` 逐文件核对（本轮装配后实测 **90/90 全 OK**）。装配后从包根目录实跑端到端：`runs/package_smoke_v1.log` 是 86 文件形态那一次，`runs/package_smoke_v2.log` 是**当时那 90 文件形态**的复跑（同一支 exe，只是包里多了 4 份源码；之后加的第 91 个文件是一份文档，不可能改变 exe 的行为，所以**没有为它重跑计时**——这是刻意的例外）。两次的判定量**全部相同**：rc=0、`gate: PASS`
（A/A2/A3/A4/B/C/D/E 全 0 位差）、8 token 559.1 ms = **14.31 tok/s**、稳态 69.9 ms/token。两次的差只有墙钟：
权重装载 1201.0 ms → 1099.9 ms、forward bring-up 716.7 ms → 421.2 ms（页缓存状态没被仪表控制，所以这只说明
"两次都在带内"，不说明原因）。也就是说下面这份清单不是设计稿，是被验证过的发货内容。

```
bonsai-vk-0.1.0-win64/
  bonsai_vk.exe            539 KiB（实测 552,178 B，`tools/source_size.py`）；运行期只吃 4 支 dll：
                           系统的 vulkan-1.dll + 下面 3 支随包的 MinGW 运行时
  bonsai_fleet.exe         335456 B —— 多卡环/iGPU 零拷贝的测量台（selftest|probe|ring|split）
  align_27b.exe            287063 B —— 27B 架构闸门（PTQ1_0 解码、64-block 路由、SSM 电荷 = 0、旋转路由）
  vulkan-1.dll             由系统/驱动提供 —— 不打包（打包驱动 dll 是许可证与兼容性双重错误）
  libgcc_s_seh-1.dll       929969 B —— MinGW 运行时，随包（GCC runtime linking exception）
  libstdc++-6.dll          2597845 B
  libwinpthread-1.dll      72171 B
  shaders/                 **必须随包，52 支 .spv = 307432 B。〔更正〕上一版这里写的是「build 时内联进
                           exe，现状就是内联的字节数组」——那是错的**：`make_pipeline` 走的是
                           `std::string("shaders/") + name + ".spv"` 的文件读取（`bonsai_forward.h:339`、
                           `bonsai_weights.h:408/413-415`），仓库里没有任何把 SPIR-V 编成 C 数组的环节。
                           实测证据：把 exe 单独放进一个没有 shaders/ 的目录，它打印
                           `cannot open shaders/repack_ptq1_0.spv - compile shaders first` 并停下。
                           第二条连带事实：路径是**相对当前目录**而不是 exe 目录，所以包根目录就是运行目录。
  fidelity/                895688 B（5 支）：align_tokenizer 195022 / align_dequant 182378 /
                           align_f16 126037 / align_attn 260683 / shared_probe 131568。
                           〔偏离上一版〕上一版只列了 align_27b，但 README 的验收口径是**五件套**，
                           不带这四支的包 = 收到包的人无法复算任何保真声明，只能信日志。
  runtime_vk/              291355 B（4 份源码：gpu_fleet.h 107595 / bonsai_model.h 68776 /
                           bonsai_fleet.cpp 68655 / align_27b.cpp 46329）。〔本轮装配时新增，起因是一条
                           实测的坏链〕包内 README 的文档表把 `runtime_vk/gpu_fleet.h` 这类路径写成**相对链接**，
                           而二进制包里没有 `runtime_vk/` ⇒ 收包的人点进去 404。检查器已固化成
                           **`python tools/link_check.py`**（逐 (目录, 文档) 组合解析 `]()` 相对链接，
                           先剥围栏代码块）：装配前跑 = 仓库侧 11 份文档全 ok、包侧 **6 处坏链**（4 个不同的
                           源文件路径，`gpu_fleet.h` 被引了两次），rc 1。
                           修法不是删链接（仓库上下文里它们有用），是**把这 4 份按同名路径放进包**——于是
                           同一份 README 在两个上下文里都成立，重跑 `tools/link_check.py` = 19 份文档
                           **TOTAL BROKEN 0**、rc 0（实测）。
                           这是刻意的一半源码发布：**本包是二进制包**，引擎自己的 `bonsai_forward.h` /
                           `bonsai_vk.cpp` / `shaders/*.comp` / `build_mingw.bat` **不在包内**，
                           完整源码与构建路径见仓库与本文件 §1。发全量源码 = 另一件事（连 shader 源与构建脚本一起发
                           才叫源码发布），没做，也没假装做了。
  README.md / README_zh.md 30 秒上手 + 三条硬前提 + 上面的 CWD 规矩（体积一律引 `python tools/source_size.py`）
  *.md                     另外 14 份审计文档（PROJECT_BONSAI_FULL_DOSSIER/PROJECT_BONSAI_FINAL_REPORT/BENCHMARK/HANDOVER/FAITHFUL_REPRO/
                           KV_P0P1P2/ATTENTION_ALIGN_FINAL/END_TO_END/DEEPINFER_EVAL/ROADMAP/STAGE0-3）——README
                           里是指向它们的相对链接，不一起发就是坏链；这一行的字节数同样以本节命令为准
  LICENSES/README.md       2470 B —— 逐件来源与许可。**引擎自身许可仍是 TBD**（候选 Apache-2.0，
                           退 MIT），没替用户拍板；`dequant_ptq1_0_ref.spv` 是唯一一支官方参考 kernel，
                           只被 fidelity/align_dequant.exe 当量尺用，引擎前向路径从不加载它。
  MANIFEST.sha256          逐文件 sha256（`sha256sum`，90 条，不含自身）
  tools/clear_field.ps1    基准/清场脚本，用户不需要，但调试要
  tools/source_size.py     每一个字节数的唯一生产者；文件被动过就非零退出
                           〔不随包〕`tools/link_check.py` 留在仓库：它按仓库根 + `dist/…` 两个上下文解析
                           链接，在包内跑必然报自己找不到 `dist/`，把工具放进错误的上下文比没有工具更糟。
  runs/                    四份：regression_v14.log、gqa_pair_v1_summary.txt、oom_fallback_v3.log、
                           package_smoke_v2.log（**本包的开箱实测**；上一版发的 package_smoke_v1.log 已被它
                           取代——那是 86 文件形态，包内换成最终形态的复跑，仓库 `runs/` 两份都留着，
                           因为纪律是"只新增、不覆盖"）
```

**装配的第三条实测：`MANIFEST.sha256` 曾经只在拷贝的一侧成立**。本机 `core.autocrlf=true`（不改 git 配置，
规矩写进 `.gitattributes`），而 git 在 Windows 上 checkout 文本时会把 LF 重写成 CRLF。空仓库实测：
提交 `alpha\nbeta\n` → 删除工作文件 → `git checkout --` 之后，**blob 的 sha256 与文件的 sha256 不再是同一个**
（`e49c81e2…` vs `98ab4d3a…`，文件变成 `b'alpha\r\nbeta\r\n'`）。也就是说：按工作树字节算出来的清单，
在别人 `git clone` 出来的目录里逐条对不上。本包当时有 **16 个** 文本文件处于 `i/lf w/lf`（正是会被改写的那一类；
9 个 `.md` 副本碰巧已是 CRLF，算"侥幸对上"）。修法是 `dist/bonsai-vk-0.1.0-win64/** -text`
（加不规范化、出不转换 ⇒ index == 工作树 == clone 产物），把这 91 个文件全部标成 `attr/-text`，
包内文本文件统一落成 LF，清单重算。**验证方式也固化了**：`git checkout-index -a --prefix=<空目录>/`
导出 → 进导出包根 → `sha256sum -c MANIFEST.sha256` ⇒ **91 个文件、90/90 OK、rc 0**（实测）。

**没做的事，直说**：包没打 zip（目录即产物，压一遍只会让 `MANIFEST.sha256` 的核对多一步）；
`ANNOUNCEMENTS.md`（16 条推文 + 长文草稿）**不随包**——README 的文档表里没有指向它的链接，所以包内不存在坏链，
它是仓库里的对外文案工作底稿，不是运行或复算保真声明所需的东西；
`shaders/` 里 `bench_shm_n*` / `probe_shm_n*` / `math_probe` / `f16_pack_probe` 这 12 支是
`shared_probe` 与合成 bench 的模块，引擎不加载，随包发出是为了让 `shared_probe.exe` 开箱能复算 LDS
那两条翻案结论，不是为了体积好看。

### 3.2 必须写进 README 的三条硬前提（全部是实测踩出来的）

1. **模型文件要用户自己下载**，包内不带（体积 + 许可），命令行第一个参数就是路径。
2. **本卡驱动是 AMD 专有 amdvlk**；项目里有两条驱动特定的事实——
   `packHalf2x16` 向零取整、NULL 查询池 SIGSEGV。**发布前必须在第二块卡/第二个驱动（RADV）上
   至少跑一次 `align_*` 五件套**，否则「能跑」只对这一台机器成立。→ §5 清单第 7 项，当前**未过**。
3. 显存预算：**当前二进制**的口径是引擎自己打印的三个常数——
   `已提交权重 2467.7 MiB + 与 cap 无关 52.0 MiB + 24.66 KiB × cap`（q8_0-V + fp16-K 的 arena 24.50 KiB
   再加 0.16 KiB 的 fp32 侧，`runs/gqa_real_c2048_merged_v1.log:48`）。4 GB 卡的 arena 算术上限：
   按 **DXGI LOCAL 预算 3464.6 MiB** 是 `(3464.6 − 2467.7 − 52.0) × 1024 ÷ 24.66 = 39 237` 槽
   ≈ **39.2 K token**（**安全**），按原始 3840 MiB heap 是 54 825 ≈ **54.8 K**（**极限**；引擎在 cap 256
   就把这一档报出来了：`headroom 1314.1 MiB = 54558 more slots`，`runs/gqa_gate_c256_v1.log:49`）。
   安全上限之外没人真机跑过。
   `§2.2` 里那条 `52 MiB 固定 + 36.16 KiB/槽` 是 `build_v15` + fp16 K/V 未压缩那一次的读数，**两者不要混引**。
   2 GB 显存的卡**跑不了这个 checkpoint**（预算表 C 分支就是这个意思）。

### 3.3 CLI 面（现状 + 建议）

现状已经是 `bonsai_vk.exe run <gguf> <chunk_MiB> <tokens> [-- "prompt"]`，配置面是环境变量：
**引擎/生成侧 45 个 + `bonsai_vk.cpp` 的内核探针 9 个 + tokenizer 的 `BONSAI_TOK_INIT` 1 个**
（`实测` `grep -o 'getenv("BONSAI_[A-Z_0-9]*")' runtime_vk/bonsai_forward.h | sort -u | wc -l` = 45；
STAGE9 记的是 24 个，增量主要是 KV/spill/GQA 三族。逐个语义与行号见 `HANDOVER.md` §4.3，那张表就是 `--help` 的草稿）。
建议 0.1.0 只做三件事，**不改行为**：① `--help` 打印每个开关一句话；② `--version` 打印
构建时的日志戳与 §6.2 基线号（现在引用 `build_v42.log` / `regression_v14.log`）；③ 把 `BONSAI_HEAP_LIMIT_MIB` 明确标为「测试旋钮」并
从 `--help` 里注明「不是用户调的」。环境变量作为唯一配置面在 0.1.0 保留（命令行化是 0.2 的事）。

---

## 4. 优化清单（按**实测收益**排序，不按想法排序）

时间切片的底片（`runs/profile_v2.log`，cap 1024，32 decode token，wall 72.138 ms/token，13.86 tok/s）：

```
T_cpu    record               1.640 ms   2.27 %      T_gpu   CB span         63.648 ms  88.23 %
T_submit vkQueueSubmit        0.024 ms   0.03 %      bubble  wall − T_gpu     8.490 ms  11.77 %
T_sync   vkWaitForFences     63.802 ms  88.44 %        of which logits.assign  6.268 ms
T_reset  reset + query read   0.002 ms   0.00 %      T_memory readback        6.270 ms   8.69 %
T_other  sampler argmax       0.159 ms   0.22 %      T_other  printf+detok    0.240 ms   0.33 %
```

| # | 项 | 现在的代价（实测） | 收益（推导，给公式） | 改动点 | 工作量 | 验收断言 |
|---|---|---|---|---|---|---|
| **P0-1** | **批量 prefill**（seq>1） | **当前二进制实测**：1310-token prompt 的 prefill = **103 766.4 ms = 12.62 tok/s**，即每位置 **79.2 ms**（推导 = 103766.4 ÷ 1310，`runs/gqa_real_c2048_merged_v1.log:113`），而同一 run 的 decode 稳态是 70.6 ms ⇒ 位置之间**几乎没摊薄**（只省了 8.6 ms/位置，12 % 不到）：每个位置照付一整个 forward 的 1490 次 dispatch / 1489 次 barrier（推导 = 1310 × 1490 = 1.952 M；旧口径对照 `qcheck_2048_v2.log:110` 实测总数 2355690 = 1581 forward × 1490）。旧数据点 1579 token = **131.99 s**（`runs/qcheck_2048_v2.log:93`，旧二进制，量级一致） | prefill 的 1.640 ms 记录 + 8.490 ms bubble 是**每次 submit** 的代价；B 个位置合成一次 submit → 每个位置只付 1/B。B=32 时**推导**每位置 79.2 → 79.2−(10.13×31/32) ≈ **69.3 ms**（−12.5 %，比旧版那个 −27 % 保守，因为分子换成了当前实测的 79.2 ms），1310 token 的 prompt 从 103.8 s → **90.8 s**。**这条只覆盖 submit 侧**；平方级的那一半（每个新位置要扫已过的所有 key）不在这个数里，它才是 32 K prompt ≈ 120 min 推导的来源（`BENCHMARK.md` §10 第 4 条）。真正的红利在 attention：seq>1 让同一批 key 被一次载入服务多个 query（**假说，未验证**，不进上面那个 −12.5 %） | `fwd_run`（:2551）的两个逐 token 循环——prefill **:2630**（`fwd_token` 调用在 :2632）、decode **:2767**（旧版记的 :1723/:1819 已漂移）+ `attn_qk_softmax*` 的 mask stride（已经是 `t*cap`，OK）+ KV/conv/state 的推进必须按位置批写；:2603 那条 prefill cap 拒绝逻辑要跟着改 | **大**（唯一涉及结构的一条） | 同一 prompt 的 prefill 墙钟 ≤ 现在的 75 %，且 decode 出的 token 序列与逐 token 版**逐 ID 相同**（这条必须硬过，否则语义变了） |
| **P0-2** | **端侧 argmax / 不回读全 vocab** | `T_memory 6.270 ms`，其中 `logits.assign` **6.268 ms**（0.946 MiB → **150.9 MiB/s**（推导 = 0.946×1048576÷0.006268），而本卡 SDMA 实测 7.12 GB/s ≈ 6790 MiB/s → **差 45 倍**；瓶颈是 WC 主机内存的逐 32-bit assign，不是搬运。**注意口径**：这一行是 `profile_v2.log`（cap 1024、仪表打开、旧二进制）的读数，本代二进制稳态是 70.6 ms/token，**6.27 ms 需要在 v42 上复测一次再定档**） | 贪心解码（temp=0/top_k=0，就是当前基线）时 host 只需要**一个 token id**：端侧 argmax + 4 字节回读 → **省 ~6.2 ms**。按当前稳态推导：70.6 → 64.3 ms → **15.55 tok/s（+10.1 %，推导，假设可加）**；按当年那一次推导是 72.138 → 65.87 = 15.18（+9.5 %）。两个数都依赖上面那句"需要复测" | 新内核 `argmax.comp`（**不许改已验证的五条谱系**：narrow / wide / tile / merged-QK / merged-PV），LOGITS 区域留在 arena，只回读 (id, value) | 小（一个新内核 + 一条开关） | `BONSAI_PROFILE=1` 的 `T_memory` 掉到 < 0.5 ms，且与现在逐 token **同一串 id** |
| **P0-3** | **HOST_CACHED staging + flush/invalidate** | 同一份 0.946 MiB 走 write-combined 内存 | **未验证**：收益取决于缓存映射后的 assign 速度，可能是 P0-2 的一部分，也可能把 6.27 ms 直接压到 <1 ms 而不用新内核 | `bonsai_vk.cpp` 的 host 内存类型选择 + 用 `vkInvalidateMappedMemoryRanges` | 半小时 | 直接跑 `BONSAI_PROFILE=1`，读 `T_memory` 那一行；两个方案（P0-2/P0-3）**先做这个便宜的实验再决定** |
| **P1-1** | **减 barrier / 依赖图** | 每 forward **1490 次 dispatch、1489 次 barrier**（推导自三处实测总数，全部整除：`profile_v2.log:151` 83440÷56 forward = 1490.0、`qcheck_2048_v2.log:110` 2355690÷1581 = 1490.0、`ignition_v2.log:151` 80460÷54 = 1490.0；barrier 同口径 83384÷56 = 2354109÷1581 = 80406÷54 = 1489）。bubble 8.490 ms = 11.77 % | **未验证**，且必须先有仪表（`BONSAI_PROFILE` 的入口实测在 `bonsai_forward.h:1930`）。合并无依赖 barrier 的上界是 bubble 的一部分，不给数字 | `fwd_token` 的 barrier 放置；**顺带修一处会误导人的打印**：`bonsai_forward.h:1896` 把**累计**计数器标成 "per command buffer (one submit per token)"，谁照字面读就会把 1490 看成 8 万——本文早先那版就是这么错的（写成了 ~14900/forward） | 中 | bubble 下降**且** `BONSAI_QCHECK=1` 的 SCORE/PROB/P16 三个区域逐位不变 |
| **P1-2** | **上下文档位化**（256 → 32 768） | 已具备并且**已实测成阶梯**：`BONSAI_CAP` + 预算表 + auto 路由 | 短 prompt 下 **cap 本身几乎免费**（旧二进制实测：71.2 ms/token @256 vs 72.6 @2048 = **+2.0 %**，`oom_fallback_v3.log:53/:11`；新二进制再次确认：32 K arena 对 16-token prompt 只 +0.4 %）。贵的是**已写入的 key 数**——现在有一张**同一二进制、同一热窗口、配对**的实测阶梯：4 K **9.41** / 8 K **6.93** / 16 K **4.57** / 32 K **2.67 tok/s**（`runs/gqa_pair_v1_summary.txt`，合并 QK 开）对照 7.88 / 5.34 / 3.29 / 1.84（关）。最小二乘：`67.17 + 0.00936 × positions` ms/token，**常数没动、斜率 −33.5 %** ⇒ 档位应当按「live keys」宣传而不是「cap」 | 文档（已写：`BENCHMARK.md` §5/§6、`KV_P0P1P2_MEASURED.md` §10）+ `--help` 的档位建议 | 小（`--help`） | 三档 tok/s + 显存**实测表**：tok/s 侧已交付；显存侧的预算表**还没在 v42 上复跑**（见 §1 的口径警告），这是本条唯一没闭合的部分 |
| **P1-3** | ~~>2048 上下文~~ → **已交付（STAGE9）** | 原判「卡住的是共享内存不是显存」**部分成立、结论被推翻**：`shared_probe.exe` 实测驱动允许 32768 B LDS（`MAXCAP` 4096 建得起来且逐元素正确），而 tiled 内核的 LDS 是 `4×QTILE+4`，**与 cap 无关、根本没有 MAXCAP**（`bonsai_forward.h:129`） | 真实红利不是"能建 4096"，而是**解除 cap + 换斜率**：wide 的 LDS 装的是 SCORE 行不是 K/V，所以 tiling 省的是**能力**不是字节（前提更正①，`KV_P0P1P2_MEASURED.md` §9）。tile ≡ wide **逐位相同**（A2 门，含多块路径），三个已验证内核一字节未改；auto 改为 `cap > 2048` 走 tiled（以前是 `rc 4` 拒绝） | 已完成：`attn_qk_softmax_tile_q{1024,2048,4096,8192}.comp`；在此之上 STAGE10 又交付了 GQA 合并（`attn_qk_softmax_gqa_h4_q{256,512,1024}.comp` + `attn_pv_q8v_gqa.comp`，QK 那半边默认开） | — | 已验收：A2/A3 位相同 + 4K~32K 配对阶梯；剩下的只有 §10 的 40 K/48 K 出带推导 |
| **P1-4** | **半合并（2 头一组、8 workgroup）** | 现在只有 HPK = 4 的全合并与 HPK = 1 的窄/宽两条线；12 CU 的卡上「gather 事务 ÷4」和「常驻 workgroup 16→4」这两件事互相打架，中间点没测过 | **有实测方向但没试过**：`GQA_PART` 的归因数据显示 PV 那半边在 cap 256 是 **+2.5 %**、在 32 K 是 **−25.5 %**（§6/BENCHMARK），说明收益的符号由 occupancy 决定，而 2-head 合并正是改变 occupancy 的那颗旋钮 | 新着色器（不许改已验证的四条谱系）+ `BONSAI_GQA_HPK=2` 一类的开关 | 中 | 位相同门新增 A5：半合并 ≡ 全合并在同一份 KV 字节上逐位；过了才允许进默认档 |
| **P2-1** | **第二驱动栈验证** | 未做 | 决定「别人能不能跑」，不是速度。本仓库已有**两条驱动特定的事实**（`packHalf2x16` 向零取整、NULL 查询池 SIGSEGV）和**两条新踩的坑**（悬挂的 `pQueueFamilyIndices`、created-SIGNALED fence），它们都写在 README 里，但都只在 amdvlk 上验证过 | 无代码 | 小（需要别人的卡） | `align_*` **五件套**（tokenizer / dequant / f16 / attn / **27b**）+ `bonsai_vk.exe` 带 `BONSAI_QCHECK=1` 在 RADV 上同样过，且 `gate: PASS` |
| **P2-2** | 专家融合 / 流式权重 | STAGE4 已判：**≥78.3 % 权重必须常驻显存**，跨层预取不可行（SDMA 7.12 GB/s vs 计算不打折）；DeepInfer 四项评估里 C/D 两项**不可行** | 不排期，写清楚为什么不做 | — | — | — |

**为什么 P0-1 排在 P0-2 前面**：P0-2 是 +9.5 % 稳态速度，用户等的是**第一个 token**。
1310 token 的 prompt 现在要 **103.8 s**（实测，当前二进制），而稳态 decode 只差 ~10 %；
但 prefill 的均价是**随上下文爬的阶梯**（§1 那一行给了积分口径：4 K ≈ 5.9 min、32 K ≈ 120 min，推导），
所以首 token 延迟在高上下文里是数量级级别的体验差，而不是 10 %。

---

## 5. 发布验收清单（当前状态如实标注）

```
[x] build_mingw.bat 干净 exit 0 / -Wall 0 warning        —— build_v40.log / v41 / v42 三次连编：每个 exit 0，
                                                             grep -ciE "warning|error" = 0（52 支 SPIR-V + 8 个 exe）
[x] §6.2 五件套全过                                       —— regression_v14.log（当前发布件）：
                                                             tokenizer 32/32 + 7/7 byte-identical、
                                                             PTQ 反量化三方 max|err| = 0.00000 且 128000/128000 逐位、
                                                             f16 全域 0 wrong + bit-for-bit YES、
                                                             层 3 score 3.567e-03 @ seq 64（= 官方 fp16 缓存代价的 0.26 倍）、
                                                             align_27b 的 27B 闸门 ALL CHECKS PASSED（:74）
[x] 位相同门（新增，取代旧的「四件套即验收」）              —— 同一日志 :88-94：A 窄↔宽、A2 宽↔tiled、
                                                             A3 合并 QK↔被替换内核、A4 合并 PV↔冻结 q8v 全部
                                                             "0/4096 bit-diff"，gate: PASS（bitmask 0 = A A2 A3 A4 B C D E）；
                                                             真实 1310 live key / 六个 tile 的那一次在
                                                             gqa_real_c2048_merged_v1.log:119-129，同样 PASS
[x] §6.1 点火基线 ≥ 13.8 tok/s                            —— regression_v14.log:95：14.16 tok/s、稳态 70.6 ms/token、
                                                             首 token 70.2 ms（48 token 一趟，带 QCHECK 门）。
                                                             历史档：13.99（STAGE8 首燃）→ 14.05（v15 时代）→ 14.16（今天）
[ ] chat template 后的 id 序列 == 官方逐 ID               —— 任务一已对齐 tokenizer 本身；引擎内两条渲染路径
                                                             （encode vs render_ids）实测 IDENTICAL，但**与官方
                                                             llama-tokenize -chat-template 的逐 ID 比对未做**
                                                             （HANDOVER §7.1 步骤 D4，仍开着）
[ ] 采样器与官方对齐                                       —— **STAGE10 之后这是最大的未闭合项**（BENCHMARK §10 第 1 条）
[x] OOM 路径：预算表可读、不崩、不留半初始化状态           —— oom_fallback_v3.log 三分支（A exit 0；B/C exit 4）
                                                             + NOTHING WAS DISPATCHED + B 报出可用 cap 1704
[x] 长上下文阶梯（同一二进制、配对）                       —— gqa_pair_v1_summary.txt：4 K 9.41 / 8 K 6.93 /
                                                             16 K 4.57 / 32 K 2.67 tok/s，对照 7.88 / 5.34 / 3.29 / 1.84；
                                                             夹具与真实 prefill 在 1335 / 1816 两个位置交叉验证 0.0 % / +0.4 %
[ ] cap 档位的**显存**实测表（v42 上复跑预算表）           —— tok/s 侧已交付；显存侧**cap 256 已有 v42 实测**
                                                             （package_smoke_v2.log:86,87 + :176,196：引擎自报 58.2 MiB =
                                                             52.0 固定 + 256 × 24.66 KiB，同一次 PDH dedicated
                                                             2539.23 MiB、Shared 增长 +0.00）；build_v15 那个
                                                             36.16 KiB/槽 的读数已作废。仍缺 1024 / 2048 两档
                                                             和「长 prompt × 三档」的组合
[ ] 第二驱动栈上的 align_* 五件套                          —— 没有第二块卡，缺
[x] 发行包装配 + 开箱实测（**STAGE10 新加**）              —— `dist/bonsai-vk-0.1.0-win64/`：清单与体积口径
                                                             见 §3.1（那里给的是可复算的求和命令，不是
                                                             一个会自己过期的数）。从包根目录实跑
                                                             `runs/package_smoke_v2.log`（最终 90 文件形态）：
                                                             rc=0、gate: PASS（A A2 A3 A4 B C D E 全 0 位差）、
                                                             8 token 559.1 ms = **14.31 tok/s**、稳态 69.9 ms/token、
                                                             权重装载 1099.9 ms（页缓存热的，不是冷读；
                                                             上一版 86 文件形态的 package_smoke_v1.log 同一支
                                                             exe 打的是 1201.0 ms，判定量全同、只差墙钟）。
                                                             这次装配**抓出一个假前提**：§3.1 上一版写「SPIR-V
                                                             内联进 exe」是错的，实测加载器读 `shaders/` 相对路径
                                                             （`bonsai_forward.h:339`），所以 shaders/ 必须随包、
                                                             且必须在包根目录运行——README/README_zh 的
                                                             命令块已按这条重写
[x] 包内文档完整性（同一轮里新增的检查器，也抓了东西）    —— `tools/link_check.py` 在两个上下文（仓库根 + 包根）
                                                             解析相对链接：装配前包侧 **6 处坏链**（README 把 4 份
                                                             源码路径写成链接，而二进制包里没有 `runtime_vk/`）
                                                             ⇒ 把那 4 份按同名路径放进包（+291 355 B），重跑
                                                             19 份文档 **TOTAL BROKEN 0**、rc 0；
                                                             `md_table_check.py` 全库 TOTAL 0；
                                                             `MANIFEST.sha256` 90 条，包根 `sha256sum -c` = 90/90 OK
[x] 多卡环 / iGPU 零拷贝的「已测/未测」边界写清            —— fleet_ring_v4.log + BENCHMARK §8：环、DMA 腿、背压、
                                                             门铃、带宽都实测；第二张 compute 卡与卡对卡那一跳**本机不可演练**，
                                                             probe 报 1 COMPUTE-capable device of 1
```

**可以直接进 README 的新事实**（本轮新增，均已落日志）：预算表本身（52 MiB 固定、36.16 KiB/槽是
`build_v15` + fp16 V 那一次的读数；当前口径 24.66 KiB/槽）、最低门槛 2519.8 MiB、以及 §5 里那张
位相同门与长上下文阶梯。**体积数字一律引 `tools/source_size.py`，不在本文里手算。**

---

## 6. 已知「不能修」的清单（写进 README 的 Known Issues，别让人以为是 Bug）

1. **oracle 锚点的 4 条 FAIL**：官方把激活重量化成 q8_0 之后才算 PTQ GEMV，我们没这么做（我们是对的）。
   残差 0.15–1.6 个翻转的 int8 码。要「复现」必须故意降级自己的激活。
   详见 `runs/traitor_forensics_v1.md`。**不修。**
2. **注意力与官方 fp16 KV 缓存的固有误差**：seq=64 时层 3 score 最大误差 3.57e-03，
   是官方 fp16 缓存自身代价的 **0.26 倍**（比值 0.000/0.141/0.262 于 seq 8/32/64，
   `ATTENTION_ALIGN_FINAL.md:21,25`，原始读数 `runs/reg_attn{8,32,64}.txt`）。
   注意 **seq 32/64 不是默认档**：`align_attn` 的 seq/cap 是 argv（默认 8/256），所以
   `regression_v13.log` 里只有 6.695e-04 这一条，别拿它去对 3.57e-03。**格式决定的，不修。**
3. **每行 norm 常数 ±1 fp32 ulp**：本卡 `sqrt/1/sqrt/1/x` = 1.00/1.16/1.00 ulp，是下限不是 Bug。
4. **`device-local heap` 是总量不是剩余量**：桌面占用只有真分配失败才知道，所以预算表可能在
   显示「放得下」时被 `ar.commit()` 的第二条拒绝信息接住（那条消息也在包里，写清了原因）。

---

## 7. 建议的落地顺序（一次只动一个变量，每步都有回归）

```
第 0 步（10 分钟）两处措辞/引用修正（见下，**不改行为、不改数值**）+ 重编 + 重跑回归
第 1 步（0.5 天） P0-3 实验：HOST_CACHED 一行改 + BONSAI_PROFILE 读数    -> 决定 P0-2 值不值得做
第 2 步（1 天）   P0-2 argmax.comp（新内核，不碰已验证的五条谱系）        -> 稳态 14.16 -> ~15.2 tok/s
第 3 步（0.5 天） Release 0.1.0 打包 + --help/--version + README 三条硬前提
                 + **在 v42 上复跑一次预算表**（§5 唯一没闭合的实测行：36.16 → 24.66 KiB/槽）
第 4 步（3-5 天） P0-1 批量 prefill（唯一的结构改动，先写逐 ID 相同的断言再动手）
第 5 步（1-2 天） P1-4 半合并（2 头 / 8 workgroup）+ 新增 A5 位相同门 —— 唯一有实测方向、没试过的杠杆
第 6 步（1 天）   自动降档（fwd_init 的区域编排抽函数）
第 7 步（1 天）   采样器对齐官方（STAGE10 之后最大的未闭合项，BENCHMARK §10 第 1 条）
第 8 步（缺卡）   第二驱动栈五件套 —— 这一条不做，「开源」只对这台 RX 550X 成立
```

**第 0 步的具体内容**（本次审计中重新定位；**故意没有现在改**：改了就要重新构建、重新跑回归，
而 0.1.0 的发布件已经用 `build_v42` + `regression_v14` 验过一遍，不值得为一条 printf 文案把
552,178 B 那个已签发的二进制和它的全部引用作废）：

1. `bonsai_forward.h:2907`（v15 时代是 :1896）把 `n_dispatch / n_barrier` 的**累计值**印成
   "per command buffer (one submit per token)"。计数器在 :254 初始化为 0、在 `dsp()` :358 与 :684 递增、
   到 :2907 打印时**从未按 token 归零**（:2165/:2477 那对 save/restore 只保护审计段不污染计数）。
   实测每 forward 是 1,490 / 1,489（三份日志的总数除以 forward 数全部整除），照字面读会高估一个数量级。
   本项目早先在 ROADMAP 里就把它写成了「每 token ~14900」——错了 10 倍，**已在 §4 P1-1 更正并留痕**。
2. `bonsai_forward.h:643` 的注释说 `Engine::qp` 的创建处在 `bonsai_weights.h:316`，
   **实测现在在 `:400`**（声明 `:208`）；这一条比上一版记录的「漂移 11 行」更严重，是 84 行。
   建议注释改成引用**符号**（`Engine::qp` 的 `vkCreateQueryPool`）而不是行号，行号只在 HANDOVER 的表里维护。
3. ~~`bonsai_forward.h:1319` 的注释写 "8 layers x 3 attention descriptor sets"~~ —— **已修**：
   现在 `:1910` 写的是 "8 layers x 4 attention descriptor sets. The pipelines are shared; only the windows move."。
   （同类修正也在 `:2022` 的运行时打印里，它按 `%u layers x %u attention descriptor sets` 打印真实数字。）

预算：**零成本**（无云、无授权、无新硬件）。第 8 步需要别人的卡或别人的机器；
第 0/1/2/3/5/6/7 步全部可以在本机、当前工具链、当前这份 checkpoint 上完成，且每一步的回归都是
`regression_v14.log` 那一套（五件套 + `BONSAI_QCHECK=1` 位相同门 + 48 token 点火），不需要新夹具。
