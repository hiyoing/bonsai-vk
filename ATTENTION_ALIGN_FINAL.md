# 层 3 注意力对齐最终战报（STAGE6 · 阶段 1–4）

机型前提不变：RX 550X 4 GB（gfx803 / Polaris 12 / wave64，驱动报 `fp16:0`、`int dot:0`）+ R5 3600 + 16 GB，Win10，只用 Vulkan，零预算。
模型 `models/Ternary-Bonsai-2-9B-PTQ1_0.gguf`（1,997,486,976 B，arch `qwen35`，dtype 143 = PTQ1_0 1.75 bpw，group 128）。原模型文件**从未被修改**（`n_rot` 探针是在一份 ~2 GiB 副本上做的，见 §7）。
参考实现 = PrismML 的 llama.cpp 构建 b10777-e3de71522（Vulkan 后端）。所有数字标 **实测**（本轮真跑输出）或 **推算**（由实测数推出，给公式）。

---

## 1. 老板要的三个答案，先答

**Q1 合成块边缘情况是否完美通过？——是，0 反例。**
`align_dequant.exe <gguf> synth`：3840 个合成 PTQ1_0 块、491520 个权重，scale 覆盖 15 类（`+0 / −0`、次正规 min/mid/max、`−次正规 max`、正规 min/max、`−正规 max`、`1.0`、`+Inf / −Inf`、`+NaN / −NaN`），每类 32768 权重。逐类结果：`bit GPU-REF 32768/32768`、`bit OURS 32768/32768`、`max|err| 0.00000`、`hard 0`（`hard` = 至少一侧非有限且两侧不相等 = 测不出幅度的分歧，最坏情形）；`NaN / Inf` 两列是 **CPU-REF 在该类真产出的非有限值个数**（`±Inf` 类各 10880 个 NaN + 21888 个 Inf，`NaN` 类 32768 个全 NaN），不是误差项，我们侧逐位复现了它们。总计 **bit-exact 491520/491520（100.000%）**，非有限不匹配 0；退出码本身就是断言（任一 `max|err| ≠ 0` 或任一 `hard` 都会让它非 0），实测 `exit 0`。真实权重侧同步扩到 1000 块 / 128000 权重，三个 `max|err|` 与符号一致性仍全 0。**实测**（`runs/reg_dq_synth.txt`、`runs/reg_dq1000.txt`）。

**Q2 Attention 的最大误差是多少？**
层 3 全链路（per-head RMSNorm → ×norm 权重 → half-split RoPE → fp16 arena → kq → 加性 −INF mask → SOFT_MAX(scale) → PV → sigmoid gate 相乘），真实形状 `head_dim=256`、`16 q 头 / 4 kv 头`、`n_rot=64`、`eps=1e-6f`、`freq_base=1e7`：

| 档 | score `max\|err\|` | 占 \|kq\|max | prob | context(PV) | layer-out（gated） | 占 \|gated\|max |
| --- | --- | --- | --- | --- | --- | --- |
| seq=8 | 6.70e-04 | 6.40e-06 | 2.21e-06 | **0.000e+00（逐位 100%）** | 1.19e-07 | 1.01e-07 |
| seq=32 | 2.94e-03 | 2.72e-05 | 2.33e-05 | 2.28e-04 | 1.45e-04 | 1.16e-04 |
| seq=64 | **3.57e-03** | 3.16e-05 | 2.49e-05 | 4.10e-04 | 2.56e-04 | 1.98e-04 |

fp16 打包层（bit 口径）：`vc` 三档 **0** 不匹配；`kc` 0 / 3 / 17 词；`q16` 2 / 24 / 68 词；`p16` 0 / 28 / 154 词（分母 131072 或对应档的词数）。三档非有限值不匹配（`hard`）= 0，`max|sum(p)−1|` ≤ 3.5e-07。

对比基线（同一张卡、同一批输入、float64 理想链）：官方 fp16 缓存格式**自己**引入的代价是 `arena 4.6e-04 + Q 4.7e-04 + probs 3.3e-04 = 1.56e-03`；我们 card↔mirror 的偏差 / 该代价 = **0.000（seq=8）/ 0.141（seq=32）/ 0.262（seq=64)**。也就是说：**我们的分歧小于官方数据格式强制引入的舍入误差**（授权里的「纯浮点舍入级别」判据，归一化口径 6.4e-06 已达标；seq=32/64 的 2.7e-05 / 3.2e-05 仍在格式代价之内）。**实测**（`runs/reg_attn8.txt`、`runs/reg_attn32.txt`、`runs/reg_attn64.txt`）。

**Q3 引擎当前的总行数和编译状态？**

| 块 | 文件 | 行 |
| --- | --- | --- |
| 引擎本体 | `bonsai_vk.cpp` 1890 / `bonsai_gguf.h` 155 / `bonsai_tok.h` 402 / `ptq_decode.h` 123 | **2570** |
| 注意力（已验证，**尚未接入引擎**） | `attn_vk.h` | **451** |
| 对齐 harness | `align_attn.cpp` 1447 / `align_dequant.cpp` 612 / `align_f16.cpp` 334 / `align_tokenizer.cpp` 152 | **2545** |
| 着色器 | 12 支 `.comp` + `dequant_head.glsl` 50 + `f16_rne.glslh` 55；其中注意力三支 `attn_norm_rope` 133 + `attn_qk_softmax` 99 + `attn_pv` 57 = 289 | **995** |
| 合计 | | **6561** |

编译状态：`build_mingw.bat` 退出 0，**0 warning / 0 error**，12 支 SPIR-V 全部产出，5 个 exe（`bonsai_vk` + 4 个 align）全部链接成功；引擎自检五项（upload/x/xsum/scale/codes）与 CPU exactly 一致，`correctness max|err|=0.00001`（`rel=1.06e-07`，K=4096 的 f32 累加残差，非反量化差），`M=12288 K=4096 splits=4` 跑到 **0.285 ms/token-op、176.3 Gop/s、44.1 GB/s**，wall 与 timestamps 相符 99.5%。**实测**（`runs/build_full_3.txt` —— 这一支是在最后一次源码改动之后重编的，用于确认「表里的数与盘上的源是同一份」）。上一次同类构建 `runs/build_full_2.txt` 给 `0.287 ms / 175.1 Gop/s`：正确性行逐字相同，只有墙钟差，GEMV 吞吐散布 0.7%、上传带宽散布 9%（6.48–7.11 GB/s）。

同一份重建后的 `align_attn.exe` 在 seq=8/32/64 各复跑一次（`runs/rebuild_{8,32,64}.txt`）与 `runs/reg_attn{8,32,64}.txt` 逐字比对：**把 `NNN.NNN ms` 这类墙钟字段抹掉后三份完全相同**，即全部对齐断言、计数、ulp 数字都是可复现的确定量，只有耗时抖动。

---

## 2. 阶段完成度

| 阶段 | 内容 | 状态 |
| --- | --- | --- |
| 1 | f16 边缘情况补齐（0 / 次正规 / 负 scale 合成块） | 完成，491520/491520 逐位 |
| 2 | 手搓 Attention 核心（RoPE 频率基底匹配、Causal Mask 的 fill 值、自有 Vulkan KV 池读 K/V 加权） | 完成，三支着色器 + fp32 镜像，规格全部对官方 dump 拟合 |
| 3 | 注意力数值对齐（自 Debug → 重写 → 重编 → 重测 循环） | 完成，机制归因到「每行一个 fp32 ulp 的 norm 常数 + cos/sin 尾数」，无未知残差项 |
| 4 | 全量回归 + 文档 + 本战报 | 完成（本文 + `FAITHFUL_REPRO.md` §3/§4 重写） |

---

## 3. 规格是怎么钉死的（不是抄来的）

两条独立证据链，各自只能钉一部分：

- **`tools/rope_fit.py`（对官方 rope dump：`runs/dbg_rope_b10/b100/b1000.log`、`runs/dbg_rope_nrot4.log`）** —— 8 个候选底数 × 配对律 × 维度数的最小漂移拟合，赢者 `D = 64` = `qwen35.rope.dimension_count`，定律 `out_p = x_p·cos(s·base^(−2p/64)) − x_{p+32}·sin(s·base^(−2p/64))`，`p = 0..31`，`dims 64..255` 直通，`s` = 绝对序列位置；明文下 4 个 mrope 字段相等，`[11,11,10,0]` 分段在数值上是惰性的。
- **`tools/attn_softmax_check.py`（对官方 oracle：`runs/dbg_attn_oracle.log`，由 `llama-debug.exe -p "one two three four" -n 0 -c 32 -ngl 0 -fa off --tensor-filter 'kq-3|kq_soft_max-3|attn_inp'` 产出）** —— scale 在 30 组比较上自由拟合 = **0.062501**，`1/sqrt(256) = 0.062500`，差 `7.13e-07`；`1/sqrt(64)`、`1/8`、`1/sqrt(128)`、`1.0` 分别给出 `max|err| 0.213 / 0.069 / 0.099 / 0.523` 被否。worked example（head group 0，query token 2）：官方 `kq = [120.5268, 98.2563, 111.4694]`、官方 `p = [0.5506, 0.1369, 0.3126]`；我们按 `1/sqrt(256)` 算出 `[0.5506, 0.1369, 0.3126]`（在 5e-5 的打印量子里完全相同），按 `1/sqrt(64)` 会得到 `[0.7225, 0.0447, 0.2329]` —— 一个数量级的错，直接排除。`kq` 这支 `MUL_MAT` **不带 scale**，scale 在 `SOFT_MAX` 里面；mask 是加性 −INF 并按头广播；被 mask 的位置 `exp(−inf − max)` 必须是**精确 0.0**（官方打印 30 次确认）。
- GGUF 元数据（`gguf_meta.txt`）：`head_count=16`、`head_count_kv=4`、`key_length=value_length=256`、`layer_norm_rms_epsilon=9.999999974752427e-07`、`full_attention_interval=4` → 32 层里 **8 个全注意力层**（索引 3,7,…,31），其余是 recurrent/linear 分支。

## 4. 结构断言（要求是 0，不是误差）

```
B  因果前缀死槽非 0 的个数：0 / 0 / 0（live 槽 576 / 8448 / 33280）
   活前缀内塞 -INF：7 / 31 / 63 行，卡上写出非 0 概率的行数 0；杀完仍归一 max|sum-1| ≤ 2.6e-07
C  position=0 的缓存行 vs 只 norm 的值：0/128 词；dims 64..255 直通：0 / 1 / 3 个 fp16 词
   theta 两种写法 powf(base,-2p/nrot) 与 powf(powf(base,-2/nrot),p)：max|dtheta|=1.49e-08 @p=3
     传进行里 = 1.25e-06（seq=32），远小于此处 fp16 操作数量子 ~6e-4 → 缓存后分不出来
D  GQA：16 头喂同一个 Q，同 kv 头 192/768/1536 对全一致（0 对不等），不同 kv 头 0 对「碰巧相等」
E  KV arena：prefill 一把灌 vs 逐 token 灌，8/32/64 步 gated 行差异 0、概率槽差异 0、最大分歧 0.00000000
F  fp16 打包：host RNE vs pack2_rne() 0/2048；host RNE vs 卡 packHalf2x16() 1366/2048(66.70%)，
   host RTZ vs 卡 0/2048，且 1366 个分歧 100% 是「不放大绝对值」→ 这支指令是向零舍入，不能直接用
   卡的 unpackHalf2x16() vs 我们的 f16_to_f32()：0/4096 逐位同
```

**实测**，全部来自 `runs/reg_attn{8,32,64}.txt`。

## 5. 阶段 3 的 Debug 循环：五条真因（每条都先有错判）

1. **`score max|err| = 13.48` 与 661 个 q16 词不匹配，是 CPU 镜像的锅，不是卡的。** `attn_rope(cfg, y, pos, y)` 原地调用：循环先写 `y[p]`，后面的配对还要读 `y[p+half]` 的原值——自己的输入被覆盖。改成先取 `lo/hi` 再写两处输出（项序 `hi*cs + lo*sn` 保持不变，仍与着色器 lane 一致），误差 13.48 → **6.70e-04**（约 2 万倍），q16 661→2、kc 171→0、`pregate` 到 100% 逐位。`position=0` 时 `sin=0`，所以这个 bug 在最短探针上完全隐形——这是本轮第二个「先证明再动手」的教训。
2. **`packHalf2x16()` 是 RTZ**（leg F），而官方 fp16 缓存是 RNE 建的 → `shaders/f16_rne.glslh`（55 行手写 RNE）是必需项，不是优化。
3. **FMA 合并假设被否证。** 直通维（`d ≥ n_rot`，无超越函数）也在差，第一嫌疑人是 `ss = ss + x*x` 被编译成 `fma`（未舍入乘积直接进加法 = 另一个数）。做法：`x*x` 先写进 shared 数组 `sq[]`，barrier 后 lane 0 再归约（编译器无法折回去），重编重跑。**leg A 与 leg R 的数字逐个相同**（`Q 31008/131072`、`15213 pass-through`、`max|err| 6.557e-06 @ 同一索引 107040`；`q16 24`、`kc 3`）——见对照档 `runs/attn_s32.txt`（改前）与 `runs/attn_fma_a32.txt`（改后）。假设排除；改动保留（它与镜像同形，代价为零）。
4. **残差是一个标量：每行的 norm 常数 `inv`。** 新增 leg R2：把镜像的 `inv = 1/sqrt(ss/hd+eps)` 按 fp32 ulp 平移 `k ∈ [−4,+4]`，用平移后的 `inv` 重算整行，与卡上读回的 fp32 调试流（`qf`/`kf`）逐位比。

   ```
   seq=8   Q 128 行 = 108 干净 + 20 需 ±1 ulp（-1:16 / +1:4）+ 0 不可解释；3604 个直通差异 100% 被一个标量复现
           K  32 行 = 29 + 3 + 0；522 个 100% 被复现
   seq=32  Q 512 行 = 426 + 86（-1:58 / +1:28）+ 0；15213 个 100% 被复现；K 128 = 103 + 25 + 0，4492 个 100%
   seq=64  Q 1024 行 = 846 + 178（-1:123 / +1:55）+ 0；31515 个 100%；K 256 = 209 + 47 + 0；8388 个 100%
   平移后直通残差 = 0（三档、两侧都是 0），剩余失配全部落在 dims < n_rot，即 cos/sin 的最后几位
   ```
   每一行最多差 **1 个 ulp**（没有任何一行需要 |k|>1），0 行无法解释 → stage 1 的分歧被完全归因，没有「代数写错」的样本。
   上游（leg T，卡 vs libm，工作带内 = 参数与结果都落在 `[1e-6,1e6]`）：`sqrt` **1.00 fp32 ulp**（bit-differ 15.2%）、`1/sqrt` **1.16 ulp**（28.6%）、`1/x` **1.00 ulp**（10.7%）；`cos/sin` 绝对差在 seq=8/32/64 为 `1.19e-07 / 1.79e-06 / 4.66e-06`（63 rad 的角度，自变量归约吃掉几十 ulp）；`exp` 相对 4.62 ulp。这些是驱动/硬件的正确舍入级差异，**除非在着色器里自己实现正确舍入的 sqrt（牛顿迭代），卡上没有别的办法追上 libm**。本轮没为它动引擎：按「定罪前不改配置」，它记为已知下限。
5. **harness 自己越界，差点被写成「驱动除法错 4433 倍」。** seq=64 时 leg T 单次上传/回读 1.19 MB，而 staging 环只有 1.05 MB：`readback()` **完全没有尺寸保护**（`vkCmdCopyBuffer` 写过目标末尾、`memcpy` 读过映射末尾），`upload()` 的 one-shot 分支还 `vkUnmapMemory(c.dev, c.stage.m)` 解错了句柄（解环，不解自己写的那块）。旧输出：`1/sqrt(0.755595) = 76.59`（libm 1.15）、`1/0.755595 = −5865.75`（libm 1.32）、`exp(sigmoid)` bit-differ 24.9% / `max|rel| = 9.037`。两处修好重跑：`1/x` 回到 1.00 ulp、`exp(sigmoid)` 13.19% / `1.30e-07`；并加**永久哨兵**——leg T 现在先把上传后的参数缓冲回读、与宿主数组逐位比，三档 `0/33632`、`0/140576`、`0/297504`。这条必须留在文档里：归因给硬件之前，先证明输入输出通道自己是干净的。

## 6. 顺手量到的性能事实（不是本轮目标，但记下来）

三支注意力 kernel 一次 dispatch 做完 64 token × 1 层：**实测** `norm_rope 0.763 ms + qk_softmax 2.345 ms + pv+gate 0.550 ms = 3.658 ms`（含 submit 与 fence）。**推算**：`3.658 / 64 = 0.057 ms/token/层`，8 个全注意力层 = `0.457 ms/token` → 注意力对单 token 的上限约 `2188 tok/s`，远在 STAGE4 实测的 `18.58 tok/s` 主瓶颈（权重带宽 / 提交开销）之外。注意这是**故意不做优化**的写法（一个 workgroup 一行 256 词、lane 0 串行归约，为的是与镜像同序以保证逐位可比），真正的 wave64 GEMV 版是独立的一轮，不在对齐轮里混做。

## 7. 未闭合 / 下一步（不含糊）

1. **采样（层 5）未对齐** —— 用户明确后置。
2. **注意力尚未接入 `bonsai_vk.exe` 的端到端 forward**：`bonsai_vk.cpp` 目前不 `#include "attn_vk.h"`，端到端仍走官方 fork 的图。层 3/4 的语义与数值已封死，工程接入是下一件事。
3. **chat template 与特殊 token 的端到端**未测。
4. 探针副本 `models/rope_probe_nrot4.gguf`（~2 GiB，为把 `n_rot` 从元数据里「拟合」出来而拷贝的，原模型未动）—— 用完应删，等老板一句话我就删（`rm` 属不可逆动作，不擅动）。
5. `sqrt`/`1/sqrt` 的 1 ulp 下限：可选方案是着色器内牛顿精化或把 norm 常数挪到宿主算，两者都改变数值路径，必须先立新的对齐测试再动，不能顺手改。

## 8. 归档（原始输出，未剪辑）

`runs/build_full_3.txt`（最终源码的全量构建）、`runs/build_full_2.txt`（上一次同类构建）、`runs/reg_tok.txt`、`runs/reg_tok_kr.txt`、`runs/reg_f16.txt`、`runs/reg_dq1000.txt`、`runs/reg_dq_synth.txt`、`runs/reg_attn8.txt`、`runs/reg_attn32.txt`、`runs/reg_attn64.txt`、`runs/rebuild_{8,32,64}.txt`（最后一次源码改动后重编的同一 exe 复跑，用于证明可复现）、`runs/reg_spec_softmax.txt`、`runs/reg_spec_rope.txt`、`runs/attn_s32.txt`（FMA 改前）、`runs/attn_fma_a32.txt`（FMA 改后）、`runs/attn_inv{8,32,64}.txt`（R2 首跑）。
方法、证据与规格全文见 `FAITHFUL_REPRO.md` §3（注意力）与 §4（KV 布局）。
