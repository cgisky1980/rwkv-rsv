# rwkv-rsv（中文版）

[English primary doc](README.md) · [中文版](README_CN.md)

**RWKV-7 推理引擎（Rust + Vulkan/CUDA 计算着色器）**，支持 **fp16 / int8(8-bit)** 两路权重量化推理，按模型文件自动路由、互斥共存；附 **CPU fp32 参考实现**用于精度验证与内核调优。

> 主文档为英文（[README.md](README.md)），本文件为中文版。实现细节见 [参考/技术实现细节.md](参考/技术实现细节.md)。

---

## 目录

1. [作用](#1-作用)
2. [设计目标](#2-设计目标)
3. [构建](#3-构建)
4. [运行](#4-运行)
5. [量化工具链](#5-量化工具链)
6. [指标](#6-指标)
7. [目录结构](#7-目录结构)
8. [与信天翁 Albatross、cryscan/rosalia 的关系](#8-与信天翁-albatrosscryscanrosalia-的关系)
9. [已知限制与后续方向](#9-已知限制与后续方向)
10. [License](#10-license)

## 1. 作用

`rwkv-rsv` 是一个纯 Rust 实现的 **RWKV-7** 推理引擎，推理运行在 **Vulkan / CUDA 计算着色器**上，跨 GPU 厂商（NVIDIA / AMD / Intel / 移动端）与操作系统（Windows / Linux / macOS）。

核心目标有二：

1. **降低显存占用**：通过离线权重量化（int8 8-bit 省 41%）在不大幅损失精度的前提下，把 3B 模型权重从 5.49GB（fp16）压到 3.22GB。
2. **可复现的精度验证**：提供 CPU fp32 参考实现与 GPU 内核级单测，把「内核误差」与「量化误差」隔离，逐层逐 token 验证。

默认面向 **RWKV-7 Goosed g1h-3B** 模型，但模型结构按 safetensors 张量形状自适应，可加载同系其他规模模型。

## 2. 设计目标

- **可移植性优先，但不牺牲吞吐**：以 Vulkan 为主后端，换取跨厂商 / 跨平台可用性，且在本机上与 CUDA 后端**实测无差距**（见 §6.1）。
- **两路量化共存**：fp16（无损参考）/ int8（近无损）按模型文件自动路由，同一二进制全部兼容。
- **运行时自编译 shader**：`build.rs` 在构建期用 `glslangValidator` 把 `*.comp` 编译为 SPIR-V，`constant_id` 做跨硬件自适应。
- **研发向**：内嵌 CPU fp32 参考、DIAG 三方核对、logits 对比、teacher-forced Top-1 一致率等验证工具链。

## 3. 构建

需要 **Rust（edition 2024）** 与 **Vulkan SDK**（`glslangValidator`；`build.rs` 在构建时把 `assets/shaders/src/*.comp` 编译为 `assets/shaders/spv/*.spv`）。

```bash
cargo build --release
```

## 4. 运行

加载模型（默认 `c:\work\niceui\rwkv-g1h-3B.st`，可用 `MODEL_PATH` 切换；含 `.int8_idx` → int8，否则 fp16）：

```bash
cargo run --release
```

主要 env 变量（详见 [src/main.rs](src/main.rs)）：

| 变量 | 作用 |
|---|---|
| `MODEL_PATH` | 模型路径；含 `.int8_idx` → int8，否则 fp16 |
| `BACKEND` | `vulkan` / `cuda`，显式选择后端 |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | 多 prompt 的 teacher-forced Top-1 一致率验证 |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | 单 prompt 的 CPU fp32 参考一致率 |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | 单 prompt logits 的 RMSE / Top-10 对比 |
| `GEN_TOKENS` / `REPORT_EVERY` | 连续自回归生成（`memtest` 子流程） |
| `DIAG` | 三方核对（seq / tok / CPU）+ dequant 校验 |
| `PROF_GPU` / `PROF_HOST` | GPU / 主机端性能剖析 |
| `UNIFORM_POOL_MB` | Vulkan uniform 池容量（MB，默认 8；≈3 万次 dispatch/批） |
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | 覆盖跨硬件自适应参数 |

### 4.1 示例程序（web-rwkv 风格）

```bash
# 模型信息：加载模型、打印 ModelInfo、probe 前向
cargo run --release --example model_info
# 自回归生成：prefill + GPU self-loop（argmax 确定性 / sample 采样）
cargo run --release --example generate          # env: NTOKENS, TEMP, TOPK, TOPP, GEN_MODE, VOCAB_JSON
# 吞吐基准：infer_seq / infer_tokens / argmax_selfloop / sample_selfloop
cargo run --release --example benchmark
# 稳态 prefill 基准（同长度预热后测第二次起，排除一次性 pipeline 创建开销）
cargo run --release --example prof_prefill_steady   # env: PTOKENS
# State 序列化：前进→state_back→存盘→state_load→state_back 无损闭环
cargo run --release --example state_persist     # env: OUT=state.bin
```

### 4.2 库 API 与 GPU 采样

`rwkv-rsv` 同时以 **library** 形式导出，便于服务端（如 `ai00-server`）集成，对标 [web-rwkv](https://github.com/cryscan/web-rwkv)：

- **`ModelBuilder` + `Bundle`**：加载模型并绑定零初始化 `State`；`infer_tokens` / `infer_seq` / `infer` 推进会话并返回 logits。
- **`State`**：一等公民的会话状态——`state_back()` 下载为 CPU `Vec<f32>`，`state_load()` 回灌，`reset()` 清零，序列化逐位无损。
- **GPU 采样（`SamplerParams`）**：`infer_sample` / `infer_sample_selfloop` 全 GPU 端过滤 logits，只回传采样 token 索引。

## 5. 量化工具链

离线量化器（Python，`uv` 运行）：

```bash
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

校准 prompt 采集：[tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py)（aya_dataset 多语言分层抽样：中文 30% / 英文 30% / 其他 40%）。详细参数见 [参考/技术实现细节.md](参考/技术实现细节.md)。

## 6. 指标

硬件：**RTX 2080 Ti**。模型：RWKV-7 Goosed g1h-3B（weights fp16 5.49GB / int8 3.22GB，省 41%）。详细报告见 [参考/](参考/)。

**端到端精度（teacher-forced Top-1 一致率，对比 fp16 参考）：**

| 路径 | 一致率 | 说明 |
|---|---|---|
| int8 | **98.8%**（506/512） | 近无损；权重级 avg cos 0.999820 / rel 0.6135% |

**GPU decode self-loop 吞吐**（argmax self-loop，1000 tokens，GPU ~53°C 冷态起；`memtest` + `SELFLOOP_ONLY=1` / `SELFLOOP_N=1000`）：

| 权重 | Vulkan | CUDA |
|---|---|---|
| fp16 | 88.1 tok/s | 92.4 tok/s |
| int8 | 116.0 tok/s | 126.0 tok/s |

int8 较 fp16 约 +32%（Vulkan）/ +36%（CUDA）。

> **2026-10-04 同会话交错复测**（int8，同测法）：Vulkan **68.4** vs CUDA **68.3** tok/s —— 完全持平。
> 绝对值跨会话漂移很大（本次差 40% 以上），所以只有交错比值有意义；上面 2026-08 那张表早于
> Vulkan r/k/v 内核改造，且当时并非交错测得。详见 §6.1。

**GPU prefill 吞吐**（T=256 稳态，即同长度预热后的第二次起；`prof_prefill_steady`，Vulkan cooperative-matrix GEMM 路径）：

| 权重 | Vulkan 稳态 | 备注 |
|---|---|---|
| fp16 | **2677-2700 tok/s** | GEMM 有效算力 21-25 TFLOPS（f32 累加 tensor 峰值 75-89%） |
| int8 | 2263-2307 tok/s | 含每层 dequant 开销（见「已知限制」） |

这条路径上 Vulkan ⇄ CUDA 同样打平：同会话交错（2026-10-04，`PTOKENS=256`）**2455.6 vs 2433.0 tok/s**（1.01×）。

单次冷启动 prefill 会额外付出该长度桶（m_pad）首次的 pipeline 创建成本（~几十-上百 ms，一次性）；服务端可在加载后按常用长度桶各跑一次 dummy prefill 预热。详见 [参考/Vulkan-prefill单发与稳态差异排查记录.md](参考/Vulkan-prefill单发与稳态差异排查记录.md)。

### 6.1 批量解码吞吐总表：信天翁 / CUDA / Vulkan（int8，C=2560 / H=40 / N=64 / V=65536 / L=32）

| 并发 B | 信天翁 ¹ | rwkv-rsv CUDA ¹ | 提速 ¹ | rwkv-rsv Vulkan ² | Vulkan/CUDA ² |
|---:|---:|---:|---:|---:|---:|
| 1（单流） | 94.2 – 99.0 | 94.1 – 97.2 | **持平**（3 轮交错均值 1.012×） | 83.5 | 1.00 |
| 8 | 465.4 | **547.8** | +17.7% | 481.5 | 0.98 |
| 16 | 806.0 | **979.4** | +21.5% | 880.1 | 1.00 |
| 32 | 1294.6 | **1632.8** | +26.1% | 1471.7 | 0.99 |
| 64 | 2123.3 | **2354.7** | +10.9% | 2229.8 | 1.02 |
| 128 | 2704.6 | **3123.1** | +15.5% | 3454.1 | 1.01 |
| 256 | 3036.9 | **3467.7** | +14.2% | 3932.5 | 1.03 |

> ¹ **信天翁列与 CUDA 列取自同一次同会话交错 A/B**（2026-09-22；`batch_decode_bench` 对 `run_bench.bat`，
> 交替进行，我们跑完立刻跑上游）。**提速列只在这一对内部有效。**
>
> ² **Vulkan 列来自更晚的一次会话**（2026-10-04 交错，`SEED=42`，默认 `SEGS=4` / `PAD_TO=512`）。
> 该会话里配对的 CUDA 读数为 83.3 / 490.9 / 878.6 / 1487.0 / 2192.1 / 3427.8 / 3825.0 tok/s ——
> 所以 **Vulkan/CUDA 列是相对这些值算的，不是相对 ¹ 的 CUDA 列**。**Vulkan 列只能通过这个比值来读**：
> 它既不能与信天翁列比、也不能与 ¹ 的 CUDA 列比，因为**会话不同、基准参数也不同**。
>
> ⚠️ 该卡兼作桌面显示卡：**跨会话读数漂移可达 ±17%**，而基准参数（SEGS / PAD_TO / NTOK）本身也会移动绝对值。
> **只有同会话 + 同参数的交错 A/B 可信**，单次读数足以把结论反转 ±3%。信天翁现已不在这台机器上，其列无法刷新。

**正确性**：每一档批量的 decode token 指纹（sum / xor）在两后端间**逐位一致** ——
B=8 `0xd02e16`/`0x6d08`、B=16 `0x1a123ac`/`0x9de`、B=32 `0x345e198`/`0x64a2`、
B=64 `0x68f1ce4`/`0x6456`、B=128 `0xd21f638`/`0x7a5a`、B=256 `0x1a493fad`/`0x7e01`。
CUDA 后端保留了同一个 `gemv_int8_rkv_stage1_batch` 签名，但忽略新增的 fp16 激活入参、
仍走原 fp32 路径 —— 所以 CUDA 列是**真正未经改动的基准**。

**CUDA 路径反超信天翁的途径**：**int8 张量核**（`mma.m8n8k16.s8`；Turing 的 int8 张量核峰值是 fp16 档的 2×）跑在
**int8 常驻权重**上（每 token 读 2.68 GB，fp16 引擎要读 5.37 GB）、**split-K + 确定性归约内核**
（不增加总流量就能加块数）、**多链合并 launch**（r/k/v 三条 + 低秩四条）、**形状感知 tile**
（BM 按 batch 取、BN ≈ batch、块数凑满 68 个 SM）、**warp-per-row + `__shfl` 归约**
（替掉每行一次整块树归约）、**稀疏 FFN 无原子累加**，以及把采样器的每个扫描趟都展开以提
内存级并行。完整机理：[int8 IMMA 实施记录](参考/2026-09-21-int8-IMMA实施记录.md) ·
[跨架构基线表](参考/2026-09-21-跨架构基线表.md)。

**Vulkan 追平的途径**（全在 r/k/v 融合核 `gemv_int8_rkv_stage1_batch`）：

- **fp16 激活**：此前该核每槽每 k 读一个完整 `vec4`（16 B）的 fp32 激活，而权重已是 int8 紧凑格式 ——
  成为 L2 流量的第二个大头。现在改读 `f16vec2` 缓冲（8 B，**减半**），再展宽成 `vec4` 参与 fp32 累加。
- **权重 `vec4` 打包 + `dot`**：解量化权重按行打包成 `vec4`，与激活 `vec4` 走 `dot()` —— 4 条 FMA 并成 1 条。
- **x 分块常驻 shared memory**（`XTILE=256`）：原先 x 的每个 k 元素要被全部 `BGRP=8` 个槽 × `ROWS`
  行重复读；现改为每块只读一次。smem 占用从 80 KB（超 sm_75 的 48 KB 上限）降到 32 KB。
- **`BGRP=8` 槽分组**，跨槽复用权重。

细节与踩坑记录：[参考/2026-10-04-Vulkan追平CUDA实施记录.md](参考/2026-10-04-Vulkan追平CUDA实施记录.md)。

## 7. 目录结构

```
src/            Rust 源码（main / model / gpu_model / runtime / vulkan / backend）
assets/shaders/ Vulkan 计算着色器源码（*.comp，spv 为构建产物）
tools/          离线量化与校准工具（Python）
参考/          研发参考文档（量化报告、shader 记录、GPU 模型、技术实现细节等）
test/           开发用脚本
examples/       自包含示例程序
```

## 8. 与信天翁 Albatross、cryscan/rosalia 的关系

本项目最初参考了两份已有的 RWKV 推理实现：

- **信天翁 [Albatross](https://github.com/BlinkDL/Albatross)**（Apache-2.0）：RWKV-7 的 **CUDA** 高性能推理引擎（官方声称 7.2B fp16 单卡 5090 上 **15000+ tps decode**）。本仓库的 **kernel 融合、sequence-parallel 批量提交、argmax 采样（只回传 token 索引）、pipeline 编译一次复用**等设计均对标/借鉴其思路。
- **[cryscan/rosalia](https://github.com/cryscan/rosalia)**：**Rust + Vulkan 计算着色器**的 RWKV 推理引擎。本仓库在 **Vulkan 内核骨架、运行时结构与部分算子的初始设计**上受其启发。

| 维度 | Albatross | rwkv-rsv |
|---|---|---|
| 计算后端 | CUDA（单厂商） | Vulkan / CUDA（跨厂商/跨平台） |
| 路线 | 极致性能，硬件特定优化 | 可移植性优先，运行时自编译 shader |
| 权重精度 | fp16 | fp16 / **int8**（自动路由，显存下限 8GB） |
| 相对差距（CUDA 批量解码） | 基准 | **已反超** —— B=1 持平，B=8~256 快 **+11% ~ +26%**（见 §6） |
| 相对差距（Vulkan vs 自研 CUDA） | 不适用 | **持平** —— decode B=1~256 与稳态 prefill 全线 **0.98× ~ 1.03×**（见 §6.1） |

差距主要来自：① Albatross 更激进的 kernel 融合；② CUDA Graph 捕获减少启动开销；③ CUDA 生态成熟的硬件特定库。

在 CUDA 路径上，这个差距后来被**int8 张量核 + int8 常驻权重**（每 token 字节数减半、张量核峰值翻倍）、split-K、多链合并 launch 与形状感知 tile 追平并反超（见 §6）。在 **Vulkan** 路径上，同一水平的追平也已达成：r/k/v 融合核改读 **fp16 激活**（流量减半）、权重 `vec4` 打包 + `dot`、`x` 分块常驻 shared memory —— 最后约 40% 的差距由此关闭，Vulkan 现在相对 CUDA **吞吐中性**，同时保留跨厂商能力（见 §6.1）。

在以上两者的启发下，本仓库随后独立演进，新增了它们都没有的能力：**两路权重量化推理**（fp16 / int8 自动路由）、**CPU fp32 参考实现与 GPU 内核正确性单测**、**离线量化工具链**与可复现的精度验证工作流。

## 9. 已知限制与后续方向

- **prefill dequant 开销 ~10%**（T=512 时 int8 vs fp16）：彻底消除需 cooperative-matrix 融合反量化+GEMM（Marlin 式零副本），列为长期方向。
- **prefill 每 token 耗时随 T 近二次方增长**（T=256: 0.33ms → T=512: 0.81ms，WKV 并行形式的 chunk 内项），长 prompt 可关注 WKV 分块策略。
- **Vulkan dplr_seq 占用率偏低**（40 workgroup × 64 线程 + 每 token 2 次 barrier，稳态 prefill 中 ~10ms）：可对齐 CUDA 版结构（block 并行行 + shuffle 归约）。
- **新长度桶首次 prefill 付 pipeline 创建成本**（kernel cache 已按 (shader, spec) 共享 uniform 池 + DYNAMIC offset 解耦，条目 1722 → ~30；剩余为 spec 首现的固有创建费）：可用启动预热消除。
- **Vulkan uniform 池在长批量下需手工扩容**：`NTOK=32`、`batch ≥ 16` 时会超出默认 8MB，报 `uniform pool exhausted` 中止；目前需设 `UNIFORM_POOL_MB=64`。正确做法是在 `begin_batch` 内按 `batch × NTOK` 推导池容量。
- **Vulkan 张量核 r/k/v 路径尚不可用**（`VK_IMMA_OPS=rkv`）：inter-dispatch 停顿造成 3.6× 退化，需把 r/k/v 的 IMMA 核合并成更少的 dispatch。当前达到持平的是 SIMT 融合核路径。
- **精度增强后路**：校准加权 k-means、G=64 提高元数据密度。

## 10. License

MIT。依赖明细见 [Cargo.toml](Cargo.toml)。本仓库偏研究向，商用前请自行核对所用模型权重（RWKV-7 Goosed 权重许可）与第三方依赖的许可。