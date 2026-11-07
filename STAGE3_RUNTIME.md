# STAGE3：原生 Vulkan 运行时第一次在这张卡上编译并跑通

日期：2026-11-01　目录：`runtime_vk`
前置：STAGE0（硬件）、STAGE1（WGSL 探针）、STAGE2（重排布局 + kernel 结构 + 加权推导）。
本文只写**在这台机器上真的执行过并读到输出**的东西；每条数字后面标 `实测` 或 `估算`，估算给推导。

## 1. 环境：没有 Visual Studio、没有 Vulkan SDK、没有管理员，也照样编译

`STAGE2 §5 未验证(1)` 现在关闭。落地位置全部在 D 盘，C 盘零占用增量。

| 件 | 位置 | 版本（`实测`，`--version` 原样输出） | 拿法 |
|---|---|---|---|
| C/C++ 编译器 | `<toolchain>\mingw64\bin\g++.exe` | MinGW-W64 x86_64-ucrt-posix-seh **16.2.0**（Brecht Sanders r3） | WinLibs 便携 zip 解包（`tar -xf`），winget 对 archive 包用 `--location` 会卡死 |
| CMake / Ninja | 同上目录 | cmake **4.4.4** / ninja **1.13.2** | 同一个 WinLibs 包自带，不用单独装 |
| GLSL→SPIR-V | `<toolchain>\glslang\bin\glslang.exe` | glslang **16.6.0** | 单独下的 release zip |
| Vulkan 头文件 | `<toolchain>\vkheaders\Vulkan-Headers-1.4.309\include` | 1.4.309 | GitHub codeload（实测 ~1.5 MB/s，比 lunarg 快 3 个数量级） |
| 链接用的 `libvulkan-1.a` | `<toolchain>\vklib\libvulkan-1.a` | — | **mingw-w64 不附带**，也不能从 SDK 拿：由驱动自带的 `C:\Windows\System32\vulkan-1.dll` 现场生成：`gendef vulkan-1.dll` → `dlltool --input-def vulkan-1.def --output-lib libvulkan-1.a --dllname vulkan-1.dll` |
| Vulkan SDK 1.4.363.0 | 未装（BITS 后台 43.0/289.0 MB，`实测`） | — | lunarg CDN 这条线路 BITS 实测 **~2 KB/s**（35 分钟 43 MB）。当前编译不需要它，缺的只是 validation layer / `vulkaninfo` / `spirv-val` 三样调试件 |

三个踩过的坑，写下来免得下次再踩：glslang ≥16 的可执行文件叫 **`glslang.exe`，不再是 `glslangValidator.exe`**；GLSL 里用 `[[unroll]]` 必须 `#extension GL_EXT_control_flow_attributes : require`；GNU tar（git 自带那个）会把 `D:\path` 当远程主机，解 zip 要用 `C:\Windows\System32\tar.exe`（bsdtar）。

入口两个，都是双击/一行：
- 桌面 `一键装环境.bat` → `<toolchain>\setup_env.ps1`：自检 5 项工具链 → 缺 `libvulkan-1.a` 就现场生成 → 用户 PATH 追加 MinGW（不碰系统变量，不需 UAC）→ SDK 下完且 SHA256 对上才弹一次 UAC 静默装 → 最后编译+跑分。
- `runtime_vk\build_mingw.bat`：编译 5 个着色器 + 宿主，然后直接跑一次基准。它不依赖 SDK、不依赖 VS、不需要管理员。

## 2. 正确性：五项逐字段落盘校验 + 一个真因

`bonsai_vk.cpp` 现在的调试面（都是本次新加的，靠 env 开关）：

- `BONSAI_DIAG=1`：把 arena 读回来，逐字段比对 ①上传的 PTQ1_0 源字节 0..511 ②x 向量 64 个 float ③全部 K/128 个 xsum ④块 0–1 每一行的 f16 scale 位 ⑤块 0 每一行全部 128 个 code（8 个 u32 全字）。当前 5 项全过：`upload/x/xsum/scale/codes` 都打印 match。
- `BONSAI_RANDSCALE=1`：把逐块 scale 从"全 1.0"换成 `[0.25,1)` 的随机 f16。**必需**——`d ≡ 1.0` 时 scale 取错半字、错索引都是不可见的。
- `BONSAI_PARTIAL=1`：强制 `splits = K/128` 且跳过 reduce，于是 `out[b*M+row]` 就是第 b 个量化块的贡献，逐块打出 `gpu(row0)/ref(row0)/err/最大err/不一致行数`。这是把"总误差是一个标量"拆成"哪个块错、错多少"的手段。

**真因（本次最重要的发现）**：宿主里 `vkCmdPipelineBarrier` 的调用次数是 **0**。同一个 command buffer 里连续录 `repack → xsum → gemv → reduce`，按 spec 相互之间没有内存依赖，驱动可以并发跑——Polaris 上确实并发了。症状是**同一二进制、同一命令行，两次运行的 diff 不一样**：`实测` splits=1 一次是 `diff min=max=10.4543, rows!=row0:0/12288`（全行同一个常数），下一次是 `min=8.4217 max=10.4543, rows!=row0:6144/12288`（一半行各自不同）。常数项来自 xsum 还没写就被读（行无关），行相关项来自 codes/scales 读到未落盘的重排结果（行相关）。这两种症状混在一起时，之前所有关于"element map 写错了 / 描述符偏移错了 / repack 布局错了"的推断都站不住——那些检查后来逐项证明是对的。

修法：`bar_compute()`（`SHADER_WRITE → SHADER_READ|SHADER_WRITE`，COMPUTE→COMPUTE）插在 repack 之后和 dispatch_token 内每次 dispatch 之后（含最后一发，使下一 token 的 xsum 与上一 token 的 gemv 也隔开）。加完以后，4 个形状 × splits{1,2,4,8,16,32} **全 24 个点**都过（`实测`，随机 f16 scale，rel = `max|err| / max|ref|`）：

| M×K | s1 | s2 | s4 | s8 | s16 | s32 |
|---|---|---|---|---|---|---|
| 4096×4096 | 2.14e-07 | 1.07e-07 | **6.69e-08** | 1.07e-07 | 1.60e-07 | 2.14e-07 |
| 1024×4096 | 2.00e-07 | 1.34e-07 | 1.00e-07 | **1.00e-07** | 1.34e-07 | 2.00e-07 |
| 12288×4096 | 2.11e-07 | 1.06e-07 | **1.06e-07** | 1.06e-07 | 1.59e-07 | 2.11e-07 |
| 4096×12288 | 3.83e-07 | 1.92e-07 | 1.28e-07 | **1.28e-07** | 1.28e-07 | 1.92e-07 |

这个"V 形"不是噪声，是两条串行链的和：GEMV 每线程串行累加 `per = NB/S` 个块，reduce 串行加 `S` 个偏量，误差 ≈ `c·√(NB/S) + c'·√(S)`（f32 舍入随机游走 `估算`），极小值在两者平衡处。数据自证：4096×4096（NB=32）s1 = 32 项串行 = 2.14e-07，s32 = 1 项串行 + 32 项 reduce = 2.14e-07，**两端完全对称**，中间 s4 = 8+4 项 = 6.69e-08 落到最低。也就是说：split-K 除了填 SIMD，还顺带把归约误差砍了 3 倍，这个在选 splits 时值得一起看。

`rel=0.00e+00` 出现在 scale≡1.0 那个模式：量级 ≤122、K=4096 的一次 f32 归约，逐行与 double 参照恰好相等是可能的（`max|err|` 同为 0.00000，不是"没测出来"）。随机 scale 下稳定在 1e-07 量级，与 STAGE2 里 WGSL 原型的 1.7e-07~3.9e-07 同一档次——即纯 f32 累加舍入，没有算法误差。

## 3. 性能：原生 GLSL 比 WGSL 原型还快 6–38%

每个数 = 3 轮 × 100 次 token-op 的最好一轮；一次 token-op 计 `xsum + gemv (+ reduce)` 全部 dispatch 和它们之间的 barrier，即**含**框架侧固定开销。参照 `实测`。

| 形状 M×K | splits=1 | 2 | 4 | 8 | 16 | 32 | 最好 | WGSL 原型 | 增益 |
|---|---|---|---|---|---|---|---|---|---|
| 4096×4096 | 119.1 | **131.6** | 131.4 | 131.2 | 123.5 | 114.7 | 131.6 (s2) | 124.1 | +6.0% |
| 1024×4096 | 56.5 | 91.4 | 108.6 | **114.7** | 105.0 | 87.3 | 114.7 (s8) | 83.4 | +37.5% |
| 12288×4096 | 148.6 | 152.1 | **154.4** | 149.7 | 141.4 | 126.5 | 154.4 (s4) | 144.3 | +7.0% |
| 4096×12288 | 119.3 | 138.5 | 144.0 | **152.5** | 152.4 | 139.3 | 152.5 (s8) | 142.8 | +6.8% |

单位 Gop/s（= GMAC/s）。重复性 `实测`：12288×4096 s4 三次 150.5/153.8/153.9，1024×4096 s8 三次 114.3/115.0/110.0 → ±3%。

按 STAGE2 §2.1 的同一套权重（q/o=4096×4096、k/v=1024×4096、gate/up=12288×4096、down=4096×12288，每层 192.94 MMAC，FFN 占 78.3%）重算，脚本 `calc_weighted_native.py`：

```
原生 GLSL        层 1310.1 us   加权 147.3 Gop/s   9e9 MAC/token -> 61.1 ms   16.36 tok/s
WGSL 原型 (14)   层 1421.0 us   加权 135.8 Gop/s   -> 66.3 ms                 15.09 tok/s
llama.cpp VEC    层 4456.9 us   加权  43.3 Gop/s   -> 207.9 ms                 4.81 tok/s
层时间比：llama.cpp / 原生 = 3.40x        WGSL / 原生 = 1.08x
```

`估算` 口径要说清楚：16.36 tok/s 只含 GEMV，且 9e9 MAC/token 是 hidden 4096 / ffn 12288 / 40 层的推算（`config.json` 仍未读，STAGE2 §5 未验证(4) 原样保留）。它不是端到端数字；端到端一定低于它，差额就是非 GEMV 算子 + 框架开销，也就是 STAGE2 §5 未验证(2) 那块，仍需在真 checkpoint 上量。

> **2026-11-01 更正（见 §7）**：`40 层` 这个前提已被 GGUF 头部推翻——实际 32 层，其中 24 层是 GDN
> 线性注意力（无 KV）、8 层全注意力。按实测张量形状逐层相加，真实算术量是 **7.929 GMAC/token**，
> 于是 147.3 GMAC/s 对应 **53.8 ms → 18.58 tok/s（GEMV-only）**。16.36 保留作历史口径。

一次性成本 `实测`：repack 11.0 MB→12.6 MB 用 0.73 ms（32.4 GB/s 有效），单张 24.2 MB 的测试张量；全模型 9 GB 权重的重排按同带宽 `估算` ≈ 0.6 s 量级（线性外推，未实测）。

## 4. 现在能跑什么

```
cd "runtime_vk"
build_mingw.bat                              REM 一键：编 5 个着色器 + 宿主 + 跑一次基准
bonsai_vk.exe 12288 4096 4 100               REM M K splits iterations
BONSAI_DIAG=1 bonsai_vk.exe 12288 4096 4 5   REM 五项逐字段校验 + 误差结构
BONSAI_PARTIAL=1 BONSAI_DIAG=1 bonsai_vk.exe 12288 4096 4 5   REM 逐块误差表
BONSAI_RANDSCALE=1 bonsai_vk.exe 4096 12288 8 100             REM 随机 f16 scale
```
数据仍是合成随机张量（不是 checkpoint），测的是 kernel 结构上限；这与 STAGE2 的探针口径一致。

## 5. 还欠着的（不变的两项）

1. **端到端真模型**：PrismML fork + `llama-bench -m Ternary-Bonsai-2-9B-PTQ1_0.gguf -ngl 99 -fa 0` 的实测 tok/s 还没有。上面 16.36 是 GEMV-only。
2. **格式一致性**：`repack_ptq1_0.comp` 与 fork 的 `ptq1_0_trit()` 在真实 GGUF 张量上的逐 element 对拍还没做（现在对的是我方 CPU 解码器，等于自证；参照物应是 fork 的 `tests/test-ptq1_0-element-map.cpp` 期望值）。
3. 附带：validation layer 到手后（SDK 下完）把 `VK_VALIDATION=1` 跑一遍——本次这个 barrier 问题正是它一眼就能报的 `SUBPASS-SELF`/`READ-WRITE` 类错误，我们是用逐字段落盘校验替代了它。

## 6. 三种仪表模式（本轮新增，全部真机跑过）

```
BONSAI_XFER=1    bonsai_vk.exe 12288 4096 4 20    REM 内存类型/批量与分散拷贝/compute 队列传输速率
BONSAI_SYNC=1    bonsai_vk.exe 12288 4096 4 20    REM 每 token CPU-GPU 泡：四形状 wall/gpu/host/bubble
BONSAI_OVERLAP=1 bonsai_vk.exe 12288 4096 4 20    REM transfer-only 队列 + 拷贝与计算并发（H2D/D2H）
```

原始输出归档在 `stage3_measured.txt`（A 校准 / B·D SYNC / C XFER / E OVERLAP）。

## 7. 本轮新测到的硬常数（DeepInfer 评估的输入，见 `DEEPINFER_EVAL.md`）

| 项 | 值 | 口径 |
|---|---|---|
| 设备本地堆 | 3840 MiB | 实测 |
| transfer-only 队列 | family 2，`contexts=8`（family 0 含 graphics，family 1 纯 compute） | 实测 |
| SDMA H2D / D2H（≥1 MiB） | 7.04–7.12 / 7.15–7.18 GB/s | 实测引擎时间戳 |
| 同一拷贝走 compute 队列 | 4.93 / 3.89 GB/s | 实测 → **换队列 D2H 翻倍** |
| 64 MiB | 7.12 GB/s ≈ PCIe 3.0 x8 规格 7.88 的 90% | 实测/规格 |
| 拷贝与计算并发 | 拷贝时长不变，计算 +0.0%~+0.5% | 实测（两队列、无 barrier） |
| 每次分页的固定开销 | 32 KiB：引擎 6.2 µs，wall 49–66 µs；空 submit+waitIdle 81.9 µs | 实测 → **小拷贝是开销主导，必须合并 region** |
| 提交+fence 往返（compute） | 99.4 µs → 201 op/token 若逐 op 提交 = 20.0 ms = 全 token 的 37% | 实测/推算 |
| 每 token GEMV 时间 | **53.8 ms（18.58 tok/s）** = 7.929 GMAC ÷ 147.3 GMAC/s | 推算自实测速率 + 实测张量表 |
| checkpoint 结构 | 32 层 = 24 GDN + 8 全注意力；KV 仅 8 层 → 32 KiB/token @f16；GDN 状态 50.25 MiB/序列 @f32；**无专家/路由张量、无 sliding_window、无 nextn** | 实测（GGUF 头部，字节和=文件尺寸） |

### 两个驱动级坑（会咬人，必须记住）

1. **宿主入口点 `vkResetQueryPool()` 在这套组合上跳地址 0 直接杀进程**（AMD 驱动 2.0.279 + 由
   `C:\Windows\System32ulkan-1.dll` 用 gendef/dlltool 生成的导入库；实例声明的是 VK_API 1.1
   也没用，gdb 显示 PC=0x0 落在 loader thunk 里）。`vkCmdResetQueryPool()`（录进 command buffer
   复位）完全正常，所以本轮所有复位都是录进去的。**推论**：今后任何设备级入口点
   （`vkResetQueryPool`、`vkCmdWriteTimestamp2`、`vkGetDeviceQueue2` …）都必须
   `vkGetDeviceProcAddr` 取，不能信导入库里的静态符号。
2. **stdout 是块缓冲的**：第一次 segfault 把整段输出吞了。`main()` 开头已加
   `setvbuf(stdout, nullptr, _IONBF, 0)`，别删。

### 计时纪律（本轮踩到并修好的两条）

- `wall = min-of-3` 必须与**同一次**的 gpu 时间戳配对；这张卡重复迭代之间频率漂移约 20%
  （同一 op 观测到 0.285–0.346 ms），否则会出现"gpu > wall"这种不可能的泡。修法：`keep()`
  在取 min 的同一支里记 gpu。
- 时长一律用 `limits.timestampPeriod`（实测 40.0 ns）换算，并**已用 wall 交叉核对：98.0–99.7%
  一致**，所以 µs 级归因可用。异步任务（拷贝）的时长要取**该引擎自己的**时间戳对
  （transfer 阶段，查询 120/121），不能用"fence 何时被 CPU 看到"——后者会把 32 KiB 拷贝记成
  整个 2.9 ms 计算窗口。
