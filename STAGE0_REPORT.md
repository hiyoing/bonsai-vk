# 阶段 0：硬件、后端与可行性实测报告

日期：2026-11-01　工作目录：`<repo-root>`

## 1. 硬件实测

| 项目 | 实测结果 | 备注 |
|---|---|---|
| GPU | **Radeon RX 550X**（PCI `1002:699F`，Polaris 12 / Lexa，**GCN 4 代**，gfx803） | OEM 卡；640 个流处理器 |
| 显存 | **4 GB GDDR5 / 128-bit**；Vulkan device-local heap **3.75 GiB** | 当前桌面+后台已占 **1.33 GiB**（java/Minecraft 0.64G、WeChat 0.19G、dwm 0.17G、Edge 0.14G） |
| 可用显存预算 | 约 **2.4 GiB**（关闭 Minecraft 后可到 ~3.1 GiB） | `VK_EXT_memory_budget` 报告 3.17 GiB budget |
| 显卡驱动 | AMD 26.5.2；Vulkan **1.3.260**；D3D12 驱动 31.0.21925.1001；OpenGL 4.6 | Vulkan 为 AMD 专有驱动 |
| PCIe | **Gen3 ×8**（当前=最大），理论 **7.88 GB/s**，实测预计 5–6 GB/s | 由 PnP 属性直接读取 |
| CPU | Ryzen 5 3600，6 核 12 线程，3.6 GHz | |
| 内存 | 16 GB DDR4（Vulkan host heap 7.68 GiB 可用） | 可容纳完整三值模型 + OS page cache |
| 系统盘/数据盘 | C: 28 GB 可用；**D: 395 GB 可用** | 模型放 D: |
| OS | Windows 10 家庭中文版 19045 | |
| 工具链 | Python 3.11.9 / 3.12、Node 24、.NET 8；**无 C/C++ 编译器、无 CMake** | 因此选择免编译路径 |
| 网络 | GitHub / PyPI(阿里源) / hf-mirror / ModelScope 可达；`huggingface.co:443` 不通 | hf-mirror 单流 ~4 MB/s，4 线程 ~12–20 MB/s |

关键结论：这是 **2017 年的入门级 Polaris 卡**，没有矩阵核心、没有 DP4A、不支持 FP16 加速指令（wgpu 未暴露 shader-f16）、wavefront 固定 64。

## 2. 后端可用性（已实跑，不是推测）

| 后端 | 状态 | 实测 |
|---|---|---|
| **ROCm / HIP** | **不可用** | 本机未安装；Polaris(gfx803) 自 ROCm 4.x 后移除支持，Windows HIP SDK 仅支持 RDNA3+；WSL GPU 也不支持 Polaris |
| **Vulkan (wgpu-native 0.32)** | **可用（主后端）** | 拷贝+FMA 内核 **69 GB/s**；纯 FMA **494 GFLOP/s**（≈40% 峰值） |
| D3D12 (wgpu) | 可用 | 同样内核拷贝 **43 GB/s**，慢于 Vulkan |
| OpenGL 4.6 (wgpu) | 可用 | 同样内核拷贝 **35 GB/s**，最慢 |
| CPU (numpy) | 可用 | 仅作正确性参考 |
| Vulkan 特性 | timestamp-query ✔、shader-f16 ✘、subgroups ✘、8-bit integer dot ✘ | 时间戳可用 → benchmark 可测 GPU 内核净时间 |

## 3. 核心内核实测（决定方案可行性）

测试形状 M=4096, K=2048（≈ 单个 7B MoE 层的一个专家矩阵量级）。

| 内核 | 结果 | 说明 |
|---|---|---|
| fp32 GEMV（带宽型） | **54 GB/s**，13.5 Gop/s | 4 字节/权重，完全受带宽限制 |
| 2-bit 三值 GEMV（转置布局 + vec4 加载 + 8 累加器） | **45 Gop/s**，11 GB/s | **ALU 受限**，不是带宽受限 |
| 2-bit 朴素版（每行一个 workgroup + 树归约） | 5 Gop/s | 低占用率 + 依赖链，已淘汰 |
| int8 解包版 | ~25 Gop/s（实现未完全正确，暂不作数） | 转换指令 1/4 吞吐，预计不如 2-bit |
| Python/wgpu 单 compute pass 开销 | **~66 µs** | 必须把每 token 的所有 dispatch 合并进少数 pass，否则 CPU 成为瓶颈 |

**推算（Granite 3.1 3B-A800M，活跃 800M 参数）**
- 每 token 专家权重 ≈ 32 层 × 8 专家 × 2.36M = **604M 三值权重**（151 MB）
- 专家矩阵乘上限 ≈ 604M / 45G = **13.4 ms/token（~74 tok/s）**
- 注意力等非专家部分 fp16 ≈ 0.45 GB/token → ~8.4 ms
- 合计理论 ~22 ms/token ≈ 45 tok/s；扣掉 KV/采样/调度开销，**现实目标 20–35 tok/s**
- 若专家 100% 冷启动、全靠 PCIe：151 MB / 5.5 GB/s ≈ **27 ms** → 上限 ~37 tok/s；70% 命中时约 8 ms

结论：**可行**，且还有明确的优化空间（查表法三值矩阵乘，可望把 45 Gop/s 提到 80–100 Gop/s 量级）。

## 4. 目标模型候选

| 模型 | 规模 | 下载量 | 三值权重 | 4 GB 显存适配 | 备注 |
|---|---|---|---|---|---|
| **Granite 3.1 3B-A800M Instruct**（推荐） | 3.4B 总 / 800M 活跃；32 层，40 专家，top-8，hidden 1536，inter 512，GQA 8 KV | ~6.9 GB | ~0.85 GB | **轻松**（全常驻+KV 约 1.6–2 GB） | 当前小 MoE 里最实用；支持 128k 位置（本机限制到 2–4k） |
| OLMoE-1B-7B-0924 | 7B 总 / 1.3B 活跃；64 专家，top-8，16 层 | ~13.8 GB | ~1.6 GB | 紧张（2.8 GB + KV） | 专家更多，缓存命中率更有戏剧性 |
| 自建 tiny MoE（~30–60M） | 合成/小训练 | <100 MB | <20 MB | 是 | 仅用于单测与内核回归，不作为交付主模型 |

**公开并不存在"三值 MoE"权重**（BitNet 系列都是稠密模型）。因此"三进制 MoE"只能由我们对真实 MoE 做 **PTQ 三值化** 得到：专家权重 → 2-bit 三值 + 分组 scale，注意力/路由/归一化/词嵌入保持 fp16 或 int8，必要时保留少量 fp16 离群权重。质量必须实测（PPL + 生成样例），若崩溃则按任务要求给出 4-bit 回退。

## 5. 需求边界与假设（待确认）

1. **运行时自研**：Python + wgpu(Vulkan) 写调度层与内核；不基于 llama.cpp，因为 llama.cpp 无法在推理中按专家热插拔显存（它只有 mmap/CPU 卸载）。可在最后附一个 llama.cpp Vulkan 对照基线（可选）。
2. **单机、Windows、单用户、batch=1 解码**为主；预填充按 batch 处理。上下文默认 2048（KV int8），可测到 4096。
3. **不训练模型、不写 HIP/ROCm 内核**；三值化是离线一次性转换。
4. **格式自定**：2-bit 打包 + 每 64/128 权重一个 fp16 scale + 可选离群保留；专家按层/专家分块存盘，支持 mmap 与异步预取。
5. **两级缓存**：SSD/RAM（冷）→ VRAM（热）；LRU/LFU/静态热度三种策略可切换；预取基于上一 token 路由的时序局部性 + 层内聚合传输。
6. **指标自采**：tok/s（prefill/decode）、峰值显存（Windows GPU 计数器 + 自记账）、专家命中率、PCIe 上传字节与实测带宽、内核 GPU 时间（timestamp query）。

## 6. 已产出的探针脚本

- `probe_backend.py`：适配器/特性/限额
- `probe_raw.py`：原始带宽、FMA 峰值
- `probe_gemv4.py`：最终三值 GEMV 内核（正确性 + 吞吐）
- 早期失败版本 park 在 `probes/early_failures/`（`probe_gemv.py`、`_gemv2/3/5/6/7/8`），被报告引用的迭代（`probe_gemv4/9/10/11/13/14/15`、`probe_decode.py`）仍在本目录根上。它们存在的意义是记录为什么朴素实现只有 5 Gop/s，不是引擎成绩——README/BENCHMARK 里没有任何一个合成 kernel 数字被当作端到端结果引用。
