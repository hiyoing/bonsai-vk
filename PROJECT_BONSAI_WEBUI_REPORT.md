# Project Bonsai — WebUI + 跨平台外壳 交付报告

> 口径纪律：每个数字标注 **实测**（带日志）或 **推导**（带算式）。未执行的检查写"未执行"，不写"通过"。
> 本报告覆盖 Phase 2（WebUI + HTTP API）与 Phase 3（跨平台 `#ifdef` 改造）。引擎内核与反量化路径一字节未动。

## 1. 交付物

| 产物 | 位置 | 体积 / 结果 | 口径 |
|---|---|---|---|
| 引擎二进制（含 HTTP 外壳） | `runtime_vk/bonsai_vk.exe` → 包根 | **650 996 B = 635.7 KiB** | 实测 `tools/source_size.py` |
| 单文件 WebUI | `runtime_vk/index.html` → 包根 | **101 578 B = 99.2 KiB**，5 模块内联，零依赖 | 实测 `webui/build.mjs` |
| 跨平台抽象 | `runtime_vk/bonsai_platform.h` | Win32 ↔ POSIX 条件编译 | 代码 |
| HTTP 服务 | `runtime_vk/bonsai_http.h` | Winsock ↔ POSIX sockets；OpenAI 兼容 | 代码 |
| 显存监控 | `runtime_vk/kv_guard.h` | PDH/DXGI ↔ amdgpu sysfs | 代码 |
| Linux 构建 | `runtime_vk/build_linux.sh` | `-lvulkan -lpthread` | **编译支持就绪，待实机验证** |
| 发行包 | `dist/bonsai-vk-0.1.0-win64/` | **119 文件，MANIFEST 118/118 OK** | 实测，原地 + clone 导出双核对 |

源码体量（唯一生产者 `tools/source_size.py`，`runs/source_size_webui_v1.log`）：
engine = host + shaders，**去注释 464 KiB / 原样 765 KiB / 去空白 313 KiB**（host 400 + GLSL 64）。
WebUI 之前是 416 KiB / 539 KiB 二进制；增量来自 HTTP 服务、`bonsai_platform.h` 与 WebUI 服务路径。
Windows 链接旗标 `-lvulkan-1 -lpdh -ldxgi -lws2_32 -lshell32`（全为系统导入库）；Linux 仅需 `-lvulkan -lpthread`。

## 2. 验证矩阵（全部已执行）

| 层 | 命令 / 日志 | 结果 |
|---|---|---|
| 全树编译 0 warning | `build_mingw.bat`（`-Wall`） | 0 warning（Windows） |
| 前端静态审计 | `runs/webui_audit_v3.log` | 17 pass / 0 fail / 1 warn（warn 为自测里 rgb 断言，人工复核仅用于允许语义） |
| 前端确定性 bot（A 数据/B 规则/C 模拟/D 真 DOM） | `runs/webui_bot_v3.log` | **154/154**（A30 B59 C28 D37），consoleErrors 0 |
| 前端真实输入驱动（CDP 真点击/键入，1440×900 + 420 窄屏） | `runs/webui_uidrive_v4.log` | **27/27**，exceptions 0，consoleErrors 0 |
| 截图验收 | `webui/shots/ui-0*.png` | 1440 与 420 全分辨率 Read 复核；窄屏标题/徽标重叠已修 |
| serve 连通性（thinking-on） | `runs/webui_api_v2.log` | **26/26** |
| serve 连通性（thinking-off，证真 content deltas） | `runs/webui_api_v2.log` | **26/26**，unary content="Hello"，SSE content 91 字符 |
| CLI 新旧二进制 A/B（同会话配对） | `runs/webui_ab_v1_summary.txt` | NEW≈OLD：14.03/14.00、14.02/14.03、14.01/13.99 ⇒ **0 回归** |
| 纯净点火（重构后，无 GameViewer） | `runs/webui_regression_v3.log` | **14.29 tok/s**，steady 70.0 ms，gate PASS，**无 OOM** |
| 包完整性 | 原地 + `git checkout-index -a` 导出 | **118/118 OK，rc 0**（两个上下文） |
| 文档表格 / 链接 | `md_table_check.py` / `link_check.py` | TOTAL 0 / TOTAL BROKEN 0 |

## 3. 性能归因：13.7 是环境，不是代码（纯净复跑已证实）

用户 2026-11-08 在管理员终端停掉 GameViewerService 后，复跑归因三腿 `sh batch_serve_attrib.sh v2`
（`runs/webui_attrib_v2.log`，每腿先 `clear_field`）。**这是权威结果**：

| 腿 | 配置 | 实测 tok/s | steady ms/token | gate | rc | dedicated | spill 增长 |
|---|---|---|---|---|---|---|---|
| A 对照 | CLI cap 256 | **14.22** | 70.3 | PASS | 0 | 2539.23 MiB | +0.00 |
| B arena | CLI cap 2048 | **14.20** | 70.4 | PASS | 0 | 2582.47 MiB | +0.00 |
| C 外壳 | serve cap 256 | **14.23**（引擎自报） | — | PASS | 0 | 2539.23 MiB | +0.00 |

三腿都 **≥ 14.1**、gate 全 PASS、rc 0、**无 OOM**、Shared Usage 增长全平 +0.00。腿 C 客户端墙钟
9.49 tok/s（含 prefill + loopback，见 §4 口径）。**A ≈ B ≈ C ⇒ HTTP 外壳与 2048 arena 都不拖累解码率**，
满足用户"网络层不拖累性能"的判据。

**诊断历史（v1，被环境污染，仅作对照保留）**：`runs/webui_attrib_v1.log` 当时 GameViewerServer 占
1590.9 MiB **活动**远程桌面表面（不可驱逐）、dwm 1639.4 MiB，总 dedicated ≈ 4154.6 MiB，三腿掉到
A 13.71 / B 13.68 / C 12.73。**连纯 CLI 对照都掉到 13.71**，证明 depression 来自显存抢占而非代码。
停服务后总 dedicated 降到 3237.3 MiB（−917 MiB，GameViewer 的活动表面消失），三腿全部回到 14.2+。
这印证 WDDM 纪律：空闲后台显存（dwm 残留）在引擎分配时可被驱逐，**活动**远程桌面表面不可——所以
GameViewer 才是真凶。该服务已加入 `tools/clear_field.ps1` 的 hard 名单（用户授权），今后 `-Kill` 清场自动处理。

## 4. serve 模式速率口径（不混报）

- CLI 基线（cap 256）：**14.16–14.29 tok/s**（`regression_v14` / `webui_regression_v3`）。
- serve 引擎自报（`/v1/status.last_tok_s`，不含 prefill）：cap 2048 干净时 **14.22**（`webui_api_v1`）、
  cap 256 干净时 **14.23**（`webui_attrib_v2` 腿 C）。
- 客户端墙钟（含 prefill + loopback）：≈ 9.1–9.8 tok/s —— 这是端到端体验值，**不是**引擎解码率，分开列。
- thinking-on 时小预算的可见 content 为空、文本在 `reasoning_content`，是契约不是 Bug；probe 按
  `/v1/status.thinking` 断言活跃通道，thinking-off 腿另证真 content。
- 探针日志 `GET / … bytes=100596` 的 `bytes` 实为 JS `text.length`（UTF-16 code units），**不是字节数**：
  index.html 真实体积 **101 578 B**（`source_size.py` 实测），差 982 来自中文等多字节 UTF-8 的额外字节
  （推导：101578 − 100596 = 982）。服务器原样吐 101 578 B，无截断；§1 的 101 578 B 才是字节口径。

## 5. 未闭合 / 未执行

- **Linux 实机编译与运行：未执行**（无 Linux 机器）。`#ifdef` 分支已写、Windows 0 warning；README 已注明"待实机验证"。
- 采样器↔官方对齐、chat-template 逐 ID（D4）、cap 1024/2048 显存表、第二驱动栈五件套、卡间 hop、真实 27B
  checkpoint、P0-1 submit −12.5%、GQA HPK=2 半合并 —— 沿用 `ROADMAP.md` 登记，不在本期范围。

## 6. 复跑入口

```
sh runtime_vk/batch_webui.sh v4        :: CLI 回归 + 纯净点火（先清场）
sh runtime_vk/batch_webui_serve.sh v3  :: serve 双腿连通性（thinking on/off）
sh runtime_vk/batch_serve_attrib.sh v2 :: 速率归因三腿
node webui/build.mjs                   :: 重建 index.html
```