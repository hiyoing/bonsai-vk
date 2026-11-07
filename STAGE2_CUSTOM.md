# Bonsai 2 9B + RX 550X 专属定制优化方案（STAGE2）

日期 2026-11-01。全部关键数字今天在这张卡上重新跑出来，标 **实测**；推算标 **估算** 并给推导；没验的一律写 **未验证**。
上下文里被反复引用的那份 stage-1 报告在 `STAGE1_KERNEL.md`，本文不覆盖它，只补一个更硬的结论：**上一版 kernel 慢在错误的地方**。

---

## 0. 先把前提钉死：30 tok/s 在你的卡上从来没存在过

PrismML 公布的 34 tok/s 是 **iPhone 18 Pro Max** 上的 9B 数字（官方 README，2026-10-15 那批）。RX 550X 上从来没有测过 Bonsai 2 9B，任何"从官方 30 再往上榨 10–20%"的说法都建立在假数字上。

真实的物理账（9B 稠密模型，hidden 4096，ffn 12288）：

| 量 | 值 | 来源 |
|---|---|---|
| 每 token MAC 数 | ≈ 9×10⁹ | 估算：稠密模型每权重每 token 用一次 |
| PTQ1_0 权重体积 | 9e9 × 1.75 bit = **1.97 GB** | 估算，与你说的 1.94 GB 一致（格式见 §1） |
| 显存读带宽 | **69–79 GB/s** | 实测（probe 10 READ，同一打包布局，零 MAC） |
| 纯带宽下界 | 1.97 GB ÷ 69.7 = **28.3 ms/token → 35.4 tok/s** | 估算，基于实测带宽 |
| 实测 kernel 下界 | 加权 **136 Gop/s** → **66 ms/token → 15.1 tok/s** | 逐形状实测（probe 14/13/10 今天复跑）+ 逐层加权，推导见 §2.1 |
| **llama.cpp 原生 kernel 实测** | 加权 **43.3 Gop/s** → **208 ms/token → 4.8 tok/s**（纯 GEMV） | 实测：probe 15 把 fork 的 `mul_mat_vec_ptq1_0.comp` 移植成 WGSL 校到 relerr 3–8e-07，配置按 `vulkaninfo` + 设备表取 4 行/wave64，见 §4 |
| 30 tok/s 需要什么 | 9e9 ÷ 30 tok/s = **270 Gop/s** = 实测加权值的 **2.0 倍** | 结论：不可达 |
| 寄存器-only FMA 峰值 | **220–252 G lane-op/s** | 实测（probe_alu，25 wavefront/SIMD） |

**这张卡的 DRAM 撑得起 30 tok/s，ALU 撑不起。** 三值的优势是体积，代价是每次 MAC 都要解码，而 gfx803 没有任何点积指令（无 DP4A、无 `v_dot4_i32_i8`，那是 gfx9+）。所以瓶颈从来不在 llama.cpp 的内存管理，而在每 code 的发射槽位。

---

## 1. 数据解包：正确做法不是"解压"，是"重排"

### 1.1 先把真实格式读准（不猜）

上游 llama.cpp **没有** PTQ1_0（`grep PTQ1_0 ggml/src/ggml-common.h` → 0 命中，已核实）。它在 PrismML fork 里，我把源码拉下来读了：

```
ggml/src/ggml-vulkan/vulkan-shaders/mul_mat_vec_ptq1_0.comp:
// Block layout ..., 128 weights, 7 words:                 <-- 28 B / 128 = 1.75 bpw
//   words 0..3 : qs[0..15]  byte j, trit t -> element 16*t + j
//   words 4..5 : qs[16..23] byte j, trit t -> element 80 + 8*t + (j - 16)
//   word  6    : qh[0],qh[1] (4 trits each, element 120 + 2*t + h), d (f16)
// Trits come out by the base-3 remainder recurrence t = (v*3)>>8, v = (v*3)&0xFF
```

要点：**一个 block = 128 个权重 = 28 字节**（24 B 的 qs 装 120 个 trit，2 B 的 qh 装 8 个，2 B 的 fp16 scale）。不是 g256，不是纯 2-bit。这也解释了为什么它叫 1.72/1.76 bpw：base-3 每 trit 只花 1.585 bit。

### 1.2 在显存里"解压成浮点"是死路，必须否决

- 解成 fp16：9e9 × 2 B = **18 GB**；解成 fp8 = 9 GB；解成 4-bit = 4.5 GB。4 GB 卡一个都放不下，连单层（12288×4096 = 50M 权重 = 100 MB fp16）都要来回搬。
- 更关键：**解包本身要把每个权重写一遍**，per-token 解包 = 每 token 多 18 GB 流量 = 260 ms。所以解包只能是**一次性**的。

### 1.3 真正要的一次性重排（GPU 上，零 CPU 参与）

把 PTQ1_0 → **随机访问 2-bit 布局 `[K/64][M]` 的 `vec4<u32>`**（每个 16 B 字装 64 个 code），scale 单独存 `[K/128][M]`。

为什么这个布局是对的，probe 10 实测：同一形状下它流式读能跑 **69.7 GB/s**（gate+up，零 MAC）。列主序保证 GEMV 里 64 个 lane 读 64 个连续字 = 完全合并访存。

代价与收益（全部有实测支撑）：

| | PTQ1_0 原样 | 重排 2-bit | Δ |
|---|---|---|---|
| 体积（9B） | 1.97 GB | 2.125 bpw → **2.39 GB** | **+420 MB** |
| 解码指令 | base-3 递推那一支：按源码估 ≈4.2 槽/code，**实测等效 ≈10.4 条源指令/code**（451.1 G lane-op/s ÷ 43.3 Gop/s，见 §4） | **bfe+cvt+fma：源里 3 条/code，实测折 1.65 槽** | 实测 **3.1×** |
| 实测吞吐 | **38.1–44.8 Gop/s**，加权 43.3（实测 p15：fork kernel 的 WGSL 移植，relerr 3–8e-07） | **124–154 Gop/s**，加权 136（实测 probe 10/13/14，见 §2、§2.1） | — |
| 一次性耗时 | — | ≈0.15–0.25 s（估算：4.4 GB 流量 ÷ 73 GB/s + 解码 ALU） | — |

结论：**用 +21% 体积换掉 base-3 解码**。这笔交易划算，因为带宽本来就有 2× 余量（生产 kernel 只用 31–38 GB/s，同布局实测能给 73–80），而 ALU 发射槽是稀缺资源。反过来说，如果 +420 MB 让上下文断崖式变短，就只重排 FFN（占 78% MAC 和 78% 体积，收益同样拿满，只花 ≈328 MB）。

重排 shader：`runtime_vk/shaders/repack_ptq1_0.comp`。trits 提取用与官方完全相同的 `(3v)>>8 / (3v)&0xFF` 递推，element 映射照抄上面的注释，所以它跟 CPU codec 的一致性由构造保证；最终仍需 §5 的 element-map 实测确认。

---

## 2. 算子融合：融合点不在解码，在 x 的加载方式和 scale 的分段

今天量出来的完整增益链（同一个 `[K/64][M]` 布局，同一形状，3 轮 × 100 次 dispatch 取中位；下表是**刚才一次性复跑**的值，与本文早先那次的差异 ≤6%，见下方"复现性"）：

| 步骤 | attn 4096×4096 | gate+up 12288×4096 | down 4096×12288 | 来源 |
|---|---|---|---|---|
| 纯读（零 MAC）= 天花板 | 79.7 GB/s / 318.9 Gop/s | 75.5 / 301.9 | 72.9 / 291.6 | 实测 p10 READ |
| BASE（scalar x，8 累加器） | 23.4 GB/s / 93.7 Gop/s | 30.8 / 123.0 | 21.5 / 85.9 | 实测 p10 |
| + x 改 `vec4<f32>`（1 次载 4 code） | 29.8 / 119.3 | 36.5 / 145.9 | 29.8 / 119.0 | 实测 p11 |
| + split-K（窄 M 填满 40 个 SIMD） | 31.7 / 126.9 | 38.0 / 152.1 | 38.6 / 154.4 | 实测 p13 |
| + 每 128 块 scale 段式累加（生产形态） | 31.0 / **124.1** | 36.1 / **144.3** | 35.7 / **142.8** | 实测 p14（splits 4/1/8） |

**复现性**：同一脚本两次独立运行的 BASE 档实测 87.9 → 93.7（attn）、127.3 → 123.0（gate+up）、84.9 → 85.9（down）Gop/s，即 ±6% 的运行间噪声；所以下面所有"×%"级别的结论都只在超过这个噪声带时才算成立。窄 M 一档也复现了：`M=1024 K=4096` 无 split-K 34.6 Gop/s → splits=8 **83.4** Gop/s（实测 p13）。

四条硬结论：

1. **不是显存瓶颈。** BASE 只有 21–31 GB/s，而同一布局纯读有 73–80 GB/s（同一个 kernel、同一个 bind group，只把 MAC 删掉）。差距的来源是 **每个 code 一次的 scalar x 加载**：一个 16 B 权重字装 64 个 code，scalar 布局要为它发 64 条加载指令，比 64 条 MAC 本身还多。把 x 改成 `vec4<f32>`（一条 16 B 加载喂 4 个 code）后，实测 attn 93.7→119.3（+27%）、gate+up 123.0→145.9（+19%）、down 85.9→119.0（+39%）Gop/s。
   - 注意 `XREAD` 档（0.765 ms，看起来比 BASE 还慢）不能拿来当"x 流量"的证据：它是一条 4096 长的串行加法链，测的是延迟不是带宽。这个坑我今天踩过，写下来避免误读。
2. **把 x 塞进 workgroup memory（LDS）是反效果**，慢 **2.6 / 3.2 / 8.1×**（down 层 LDS 4.744 ms vs 同基线 BASE 0.586 ms，实测 p10 复跑）。Polaris 每 CU 只有 64 KB LDS，一个 workgroup 占 16–48 KB 直接把驻留 wavefront 数打没。这条路已实测否决。
3. **每线程多扛几行输出（rows>1）也是反效果**：probe 11 里 rows 1→2 为 attn 119.3→67.0、gate+up 145.9→139.3、down 119.0→68.1 Gop/s，rows=4 更差（46.2 / 100.0 / 48.3）；probe 10 的 ROW2 同样 93.7→61.3（attn）、85.9→62.5（down）。寄存器压力换来的 x 复用不值。已实测否决。
4. **每 128 权重一个 fp16 scale 的段式累加几乎免费**：split-K 档 152.1 → 生产档 144.3（gate+up）、154.4 → 142.8（down），即 −5%～−7%；attn 126.9 → 124.1 只差 2%。这个量级仍然支持 §1.3 的重排不必为了"省掉 scale 分支"而牺牲 scale 精度，直接 `acc = fma(d[b], cp − xsum[b], acc)`。

生产 kernel（与 probe 14 完全同构，逐 code 源码 3 条指令 / 实测 1.65 个发射槽，无除法、无 base-3、无 select 链）：

```glsl
float block_part(uvec4 W, uint i) {            // 64 code / 16 B，4 个独立累加器盖住 v_fma 延迟
    float s0=0., s1=0., s2=0., s3=0.;
    for (u = 0..3) { uint wu = comp(W,u);
        for (t = 0..3) { vec4 xv = xb[i*16 + u*4 + t];      // 一次 16 B 载入喂 4 个 code
            uint sh = 8u*t;
            s0 = fma(float(bitfieldExtract(wu, sh+0, 2)), xv.x, s0);   // v_bfe_u32 + v_cvt + v_fma
            s1 = fma(float(bitfieldExtract(wu, sh+2, 2)), xv.y, s1);
            s2 = fma(float(bitfieldExtract(wu, sh+4, 2)), xv.z, s2);
            s3 = fma(float(bitfieldExtract(wu, sh+6, 2)), xv.w, s3); }}
    return (s0+s1)+(s2+s3);
}
// 主循环：一次迭代 = 一个 128 权重 block = 两个 uvec4
cp  = block_part(codes[2b*M + row], 2b) + block_part(codes[(2b+1)*M + row], 2b+1);
acc = fma(f16(scale[b*M + row]), cp - xsum[b], acc);
```

`xsum[b]` = 该 block 内 x 的和，由 `xsum.comp` 每 token 算一次（K/128 个数，几微秒）。这就是把三值的 "−1" 请出内层循环的办法：内层只剩 `c` 的加权和，`−1` 变成一个跟行无关的小向量。

**还没做完的融合**（有理由，无实测）：Bonsai 要求运行时对激活做 Hadamard（旋转烘进权重，激活侧必须同变换，官方说明）。`fwht.comp` 已写好，block-1024、shared memory、10 步蝶形，成本估算 ~0.0002 ms/向量。两个待核项：sign 对角阵是独立张量还是已烘进权重（决定 `USE_SIGNS`），以及它的 block 尺寸（我按 1024 写，需读 checkpoint 确认）。

诚实记录、并且今天已经解决的一个矛盾：按"每 code 3 条指令"和 fma 峰值推算，上限应是 83 Gop/s，但 GEMV 实测定量在 124–154 Gop/s。今天用 probe_alu 直接把这条链本身测了：`bfe + cvt + fma` 三元组在 25 wavefront/SIMD 下跑出 **451.1 G lane-op/s ÷ 3 = 150.4 G code/s**，也就是每 code 实占 **248.2 / 150.4 = 1.65 个 fma 等价发射槽**，不是 3（2-bit 的 `v_bfe` 结果和 cvt 被驱动折进了 fma 的操作数路径）。150.4 G code/s 与生产 kernel 实测的 144.3（gate+up）/ 142.8（down）几乎重合 ⇒ **GEMV 已经贴着这条指令链的 roof 在跑**，还能改的是"换更便宜的 code 表示"，不是继续调 unroll/splits。

因此 §1.3 表里的"3 槽/code"要按 **实测 1.65** 读；而 llama.cpp 的 base-3 解码没有这个折叠红利可指望（`v_mul`→移位→掩码是真实数据依赖链），按单指令速率累加估的是 **4.2 槽/code**（0.5×imul(1.77) + 1.25×shift/mask(1.0) + 1.0×cvt(1.05) + 1.0×fma(1.0)）。**这个估算今天被实测推翻了一半：probe 15 移植他们 kernel 跑出来 43.3 Gop/s（加权），按同一把尺子折 ≈10.4 条源指令/code**，也就是说光数源码行数会低估他们一半——分歧的 qh 分支、f32 `dot` 在 GCN 上展开成 4 mul+3 add、以及 4 行的寄存器压力，才是把 4.2 抬到 10.4 的东西（拆解见 §4）。结论方向不变，量级从"约 2.5×"变成"实测 3.1×"。

### 2.1 加权吞吐 136 Gop/s 的完整推导（§0 那个 15 tok/s 只有这一个依据）

一层的 7 个稠密投影（hidden 4096、ffn 12288；k/v 按 **GQA 假设** `n_kv_heads × head_dim = 1024`，**未从 checkpoint 核实**）：

| 矩阵 | M×K | MAC | 用哪档实测速率 | 单次 |
|---|---|---|---|---|
| q | 4096×4096 | 16.777 M | 124.1 Gop/s（p14 attn, splits=4） | 135.2 µs |
| o | 4096×4096 | 16.777 M | 124.1 | 135.2 µs |
| k | 1024×4096 | 4.194 M | 83.4（p13 窄 M, splits=8） | 50.3 µs |
| v | 1024×4096 | 4.194 M | 83.4 | 50.3 µs |
| gate | 12288×4096 | 50.332 M | 144.3（p14, splits=1） | 348.8 µs |
| up | 12288×4096 | 50.332 M | 144.3 | 348.8 µs |
| down | 4096×12288 | 50.332 M | 142.8（p14, splits=8） | 352.5 µs |
| **一层** | | **192.9 M** | | **1421 µs** |

加权 = 192.9e6 ÷ 1.421e-3 = **135.8 Gop/s**；每 token ≈ 9e9 MAC（估算，每权重每 token 用一次）→ **66.3 ms/token → 15.1 tok/s**。

顺带把两个"按比例"的结论算死：**FFN（gate+up+down）= 150.99 / 192.94 = 78% 的 MAC，也是 78% 的体积**，所以 §1.3 的"只重排 FFN"能拿 78% 的收益、只花 78% 的 +420 MB（≈328 MB）；注意力投影占 21.7%，且因为 M 只有 1024 而只有 83 Gop/s —— 这是 split-K 之后剩下的第二个短板（估算：把 k/v 合并成一个 2048×4096 dispatch 可以再拿一点，未实测）。

---

## 3. 显存压榨：`vkAllocateMemory` 的对齐几乎榨不出东西，真正的大头在别处

`bonsai_vk.cpp` 会把每个 buffer 的 `vkGetBufferMemoryRequirements().alignment` 和总 padding **打印出来**。按 AMD 驱动常见的 16–64 B 对齐、约 200 个权重张量推算：

> padding = 200 × ≤64 B ≈ **12.8 KB = 总量的 0.0006%**（估算，待主机实测确认）

**"最极端的对齐能省出更多上下文"这个假设是错的**，省下来的量级是 KB，不是 GB。真正吃显存的四件事，按能省的量排：

1. **别让原 PTQ1_0 和重排结果同时驻留**：一次性 +1.97 GB。重排在 VRAM 里跑完就把源 blob 释放（或干脆在加载时流式重排，源只走 staging）。这一条就值回整个方案。
2. **重排的代价**：+420 MB（§1.3）。这是唯一"花显存换速度"的开关，可按层选择性开。
3. **KV cache 精度与结构**：Bonsai 2 是混合注意力（约 75% 线性 / 25% 全注意力，官方说明），所以 KV 不是主导项——具体每 token 多少字节必须读 config 才知道，**不给假数**。公式：`2 × n_full_layers × n_kv_heads × head_dim × 2 B/token`。
4. **不加载 vision tower**（27B 版是独立 FP16 0.92 GB；9B 版待读 checkpoint）。文本推理一行都不用。
5. **arena 本身的价值是"零碎片 + 零重复 scratch"**，不是对齐。llama.cpp 会为最大算子单独留 compute buffer；手搓可以把 x/xsum/out/partials 全部塞进一个几百 KB 的固定环形区。

4 GB 卡的预算表（估算，Windows WDDM 预留 256–512 MB）：

| | 保守（不重排） | 激进（全重排） |
|---|---|---|
| 权重 | 1.97 GB | 2.39 GB |
| 0.0976% 高精度残差 | ~18 MB | ~18 MB |
| 运行时/帧缓存/驱动 | 0.3–0.5 GB | 0.3–0.5 GB |
| **能给 KV 的** | **~1.4 GB** | **~1.0 GB** |

---

## 4. 抛开 llama.cpp 自己造轮子，能不能多榨 10–20%？

**分两问回答，因为答案不一样。**

**（a）"造轮子"本身不带来速度。** tokenizer、chat template、混合线性注意力递推、sampler 都不在 hot path 上。把它们重写一遍，per-token 时间一毫秒都不会少，但会引入新的正确性风险——而这个模型恰好是**运行时必须做激活 Hadamard 变换**的那类，llama.cpp/PrismML 是目前唯一的正确性参照。为 0% 的速度丢掉参照物，是亏的。

**（b）换掉那一个 kernel 带来 **3.1×**（基线现在是实测，不再是估算）。** PrismML fork 确实有专门的 Vulkan 三值 GEMV（不是我原本以为的"通用 dequant 回退"，读源码纠正了），而且它已经用了 vec4 激活加载和 4 行复用——§2 的第 1、3 条增益他们已经吃到。差距在 **base-3 解码**：`decode_word()` 每 20 个权重要 10 次 `v_mul` + ~25 次移位/掩码 + 20 次 cvt + 20 次 fma。

今天把他们的 `mul_mat_vec_ptq1_0.comp` 逐行翻成 WGSL（`probe_gemv15.py`），在同一个 harness、同一批合成块上跟我们 §2 的生产 kernel 一起跑，并且用同一个 numpy 参照校到 **relerr 3–8e-07**（先要修好两个 bug：漏了 `reduce_result()` 的 workgroup 归约，以及 qh 那 8 个 element 的 ysum 归属——见脚本注释）。他们 kernel 在这张卡上的配置是**从源码 + `vulkaninfo` 实测出来的**，不是我挑的：

- `vulkaninfo`：`integerDotProduct4x8BitPackedSignedAccelerated = false`（三个 packed 位全 false）→ ggml-vulkan.cpp:4333 把 `device->integer_dot_product` 置 false → 6830 行 `quantize_y` 为 false → 3004 行的 q8_1 pipeline **根本不会创建**。所以这张卡走的是 f32 激活的那支 `mul_mat_vec_ptq1_0.comp`，不是 `vecq` 那支。
- GCN 分支 `rm_stdq = 2`（2884 行）→ f32 PTQ1_0 vec pipeline 的 `NUM_ROWS = 2*rm_stdq = 4`（2945 行）。
- `vulkaninfo`：`subgroupSize = 64`（GCN 是 wave64，RDNA 才报 32）→ AMD 走 `DMMV_WG_SIZE_SUBGROUP`（5667–5685 行的 LARGE 覆盖只对 NVIDIA/Intel 生效）→ **BLOCK_SIZE = 64 = 一个 wavefront**。

| 路径 | Gop/s（attn / gate+up / down / 窄 k-v） | 加权 | ms/token | tok/s | 依据 |
|---|---|---|---|---|---|
| llama.cpp 原生 kernel（实测移植，rows4/wave64） | 44.8 / 43.7 / 42.5 / 38.1 | **43.3** | 208 | **4.8** | 实测 p15 + §2.1 加权 |
| 同上但 wave32 配置 | 36.6 / 37.0 / 39.2 / 37.3 | ~37 | ~243 | ~4.1 | 实测 p15（说明 wave64 这个默认值对他们也是有效的） |
| 他们的整数支 `vecq`（这张卡取不到，反事实） | 50.5 / 47.6 / 54.4 / 46.2 | — | — | — | 实测 p15，且 `dot4` 用他们的 `dot4_emulated`（4 bfe+4 imul+3 add）；有 DP4A 的卡还会更快 |
| 只把 GEMV 换成 §2 的 kernel | 124.1 / 144.3 / 142.8 / 83.4 | **135.8** | **66** | **15.1** | 实测 p14 + §2.1 加权 + 估算 9e9 MAC/token |
| 纯带宽理想 | — | 321 | 28 | 35 | 实测带宽 + 估算 |

比之前那个 ±50% 的估算（52–60 Gop/s）低了一档：**实测 43 Gop/s，加速比 3.1×，不是 2.3–2.6×**。三个只对基线不利的近似还留着（移植里 d 从数组读、少了他们每行一次的 `unpackHalf2x16`；归约用 shared-memory 树代替他们的 `subgroupAdd`；没算他们节点级的图/队列开销），所以真实 llama.cpp 只会 ≤4.8 tok/s，不会更高。旁证：fork 自己在 ggml-vulkan.cpp:6747-6751 写着 Intel XE2 上打开 PTQ1_0 专用整数 kernel 的效果是 **Arc B390、Bonsai 2 27B、tg128：1.62 → 13.16 t/s**——那台 GPU 比 550X 快得多，走"非专用解码"那一支也只有 1.6 t/s。

**为什么实测比指令计数估算还低？** 用同一把尺子量：probe_alu 实测 `bfe+cvt+fma` 三条源指令折 1.65 个发射槽（=我们 kernel 的 3 条/code），把 451.1 G lane-op/s 的实测屋顶除以 43.3 Gop/s，llama.cpp 那支相当于**每个 code ≈10.4 条源指令**，比 §1.3 按源码行数估的 4.2 条多了一倍还多。多出来的三个来源都能在源码里点名：

1. `if (h == 0u)`：`h = item & 1u = tid & 1u`，所以 64 lane 的 wavefront 里恰好一半走 qh 解码、一半走 else —— **最大程度的分歧**，两个分支各自按半宽执行一遍（估算：这一支多付 ≈1–2 槽/code，未单独拆测）。
2. f32 的 `dot(vec4,vec4)`：GCN 没有点积指令，一个 dot = 4 mul + 3 add = 7 条，均摊 **1.75 条/code**，而整数支两条 `dot4` 摊 5.5 条/code 却整体更快（44.8 → 50.5），差别在 x 的载入宽度（f32 一支每 4 个 code 一次 vec4 载入，且要另算 ysum）。
3. `NUM_ROWS=4` 时每线程同时挂 4 份 `q[5]`（4×5 个 vec4 = 80 个 f32 值）。wave64 下一个 SIMD 16K VGPR，跑到 4 个 wavefront/SIMD 时每 lane 只有 64 个寄存器的配额，所以这支必然部分落栈——他们自己在 vecq 那支的注释里就写过"one loop over all columns makes it hoist every load (spills at NUM_COLS = 8)"（未拆测，属于估算）。

于是"能不能从 30 tok/s 再榨 10–20%"这个问题要倒过来问：**llama.cpp 在这张卡上的天花板本来就是单位数 tok/s**（4.8 是纯 GEMV 时间，还要加 attention/线性注意力递推/norm/embedding 和图开销）。换 kernel 是 3 倍级别的动作，10–20% 那种量级的东西（arena、push constants、去 compute buffer）排在它后面，不值得先做。

**结论：值得造，但要造在正确的地方。** 提速度 3× 的正确动作不是重写推理框架，而是：

1. 在 PrismML fork 的 `ggml-vulkan` 里**新增一个 quant type**（`GGML_TYPE_PTQ1_0_PACKED`，即 §1.3 的重排布局）+ 一个 mul_mat_vec kernel；加载时对每个张量做一次 `convert`（他们的 `copy_from_quant`/`dequant_tq1_0` 已经是现成挂载点）。
2. 保留他们的 tokenizer / 模板 / 混合注意力 / sampler。
3. 剩下 10% 再谈 `vkAllocateMemory` arena、去 compute buffer、push constants（这三条我都在 `bonsai_vk.cpp` 里实现了，可直接搬）。

如果一定要独立运行时，§5 的 host 就是骨架；但要清楚它现在只跑 GEMV，不跑模型。

---

## 5. 交付物与验证状态

```
ternary-moe-runtime/
  probe_gemv10.py  probe_gemv11.py  probe_gemv13.py  probe_gemv14.py  probe_alu.py   <- 逐档增益的实测来源
  probe_gemv15.py  llama.cpp 两个 PTQ1_0 kernel 的 WGSL 移植 + 同参照校验（§4 基线）
  calc_weighted.py 把逐形状速率加权成每层/每 token（§2.1、§4 的那两个数字只有这一个出处）
  runtime_vk/
    shaders/repack_ptq1_0.comp   PTQ1_0 -> 2-bit [K/64][M] 一次性重排（GPU）
    shaders/gemv_ternary.comp    生产 GEMV：bfe+cvt+fma、vec4 x、split-K、每块 scale
    shaders/xsum.comp            每 128 block 的 Σx（把 −1 请出内层循环）
    shaders/reduce.comp          split-K 偏量和
    shaders/fwht.comp            激活侧 block-1024 归一化 Hadamard
    bonsai_vk.cpp                单 arena vkAllocateMemory + 对齐/padding 打印 + 正确性校验 + timestamp 计时
    build.bat                    glslangValidator -V + cl/g++
  _ggml-vulkan.cpp  _mul_mat_vec_ptq1_0.comp  _mul_mat_vecq_ptq1_0.comp  _mul_mat_vec_base.glsl
                   _dequant_ptq1_0.comp        <- PrismML fork `prism` 分支今天拉下来的原件（带行号引用它们）
  _dbg15.py / _dbg15b.py / _dbg15c.py          <- probe 15 的三步定位：CPU 模拟、单块 GPU 复现、按 K 二分
```

**已验证（实测）**
- probe 10/11/13/14 在这张 RX 550X（Vulkan, wgpu）上跑通，数值与 numpy 参照一致：probe 14 的相对误差 **1.7e-07 ~ 3.9e-07**（纯 f32 累加舍入量级），参照里带了逐块 scale。
- **llama.cpp 基线已实测**（probe 15）：两个 fork kernel 的移植都校到 relerr **3–8e-07**，逐形状 **38.1–44.8 Gop/s**、加权 **43.3** → 208 ms → **4.8 tok/s**。移植的两个配置（4 行、wave64）不是挑的，是从 ggml-vulkan.cpp 的设备表 + 这台机器的 `vulkaninfo`（`integerDotProduct4x8BitPackedSignedAccelerated = false`、`subgroupSize = 64`）读出来的。
- 124–154 Gop/s 是**我们 kernel 结构**的实测值；69–79 GB/s 是**这个布局**的实测读上限。
- PTQ1_0 的 28 B/128 权重布局与 trit 递推：直接读 PrismML fork 源码得到，非推测；且 probe 15 的两个独立解码路径（f32 解码与 16-bit-lane 整数解码）对同一个参照都成立，等于交叉验证了 element map。

**未验证（必须你这边补的四项，全部免费）**
1. **C++/GLSL 没编译过**：这台机器没有 Vulkan SDK，也没有任何 C++ 编译器（已检查：无 g++/clang/cl，无 `$VULKAN_SDK`，无 `vulkan.h`）。装 SDK 后 `build.bat` 一次过不了很正常，把报错贴给我。着色器**数值逻辑**已由同构 WGSL 实测保证。
2. **模型本体从没在这张卡上跑过**：probe 15 测的是 kernel 形状与解码逻辑，不是 llama.cpp 的端到端。装 PrismML fork，跑 `llama-bench -m Ternary-Bonsai-2-9B-PTQ1_0.gguf -ngl 99 -fa 0`（Vulkan 后端）拿真实 tg tok/s。§4 的 4.8 是**纯 GEMV 时间**，端到端一定低于它；两者的差就是"框架开销 + 非 GEMV 算子"的实测值，也是我现在唯一补不上的那块。
3. **格式一致性**：先用 128×128 的小张量把 `repack_ptq1_0.comp` 的输出和 fork 的 `ptq1_0_trit()` 逐 element 对上。fork 仓库里确实有 `tests/test-ptq1_0-element-map.cpp`（今天列目录看到，另有 `tests/test-ptq1_0-cuda-dot.cpp`、`ggml/src/ggml-cuda/mmvq-ptq1_0.cuh`），直接借它的期望值。对不上就是映射写错了，别去调性能。
4. **config 里的两个假设**：`n_kv_heads × head_dim = 1024`（GQA）和 hidden 4096 / ffn 12288 是从 shape 命名推的，没读 `config.json`。它们只影响 §2.1 的加权比例，不影响任何单形状实测。

**一句话回答你的三个问题**：解包必须在显存里做、但做的是"重排成 2-bit 随机访问"而不是"解压成浮点"（+21% 体积换 −30% 解码指令）；算子融合的真正红利是 x 用 vec4 载入 + 逐块 scale + split-K，实测 93.7 → **124.1** Gop/s（attn）、85.9 → **142.8**（down）；`vkAllocateMemory` 的对齐只能省出 KB 级，省 GB 的是"别让两份权重同时驻留"和"不加载 vision tower"。抛开 llama.cpp 换不来 10–20%，但**把他们那一个 GEMV 换成 §2 的 kernel 是实测 3.1 倍**（43.3 → 135.8 Gop/s 加权），天花板 **≈15 tok/s**，不是 30——30 这个数在你这张卡上从来没存在过。
