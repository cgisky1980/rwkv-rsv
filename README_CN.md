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

- **可移植性优先**：Vulkan 而非仅 CUDA，牺牲约 40% 峰值吞吐，换取跨厂商 / 跨平台可用性。
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
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | 覆盖跨硬件自适应参数 |

### 4.1 示例程序（web-rwkv 风格）

```bash
# 模型信息：加载模型、打印 ModelInfo、probe 前向
cargo run --release --example model_info
# 自回归生成：prefill + GPU self-loop（argmax 确定性 / sample 采样）
cargo run --release --example generate          # env: NTOKENS, TEMP, TOPK, TOPP, GEN_MODE, VOCAB_JSON
# 吞吐基准：infer_seq / infer_tokens / argmax_selfloop / sample_selfloop
cargo run --release --example benchmark
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

**GPU decode self-loop 吞吐**（argmax self-loop，1000 tokens，GPU 60°C 冷态；`memtest` + `SELFLOOP_ONLY=1` / `SELFLOOP_N=1000`）：

| 权重 | Vulkan | CUDA |
|---|---|---|
| fp16 | 80.0 tok/s | 85.7 tok/s |
| int8 | 110.6 tok/s | 110.8 tok/s |

int8 较 fp16 约 +38%（Vulkan）/ +29%（CUDA）。

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
| 相对差距 | 基准 | 约 **40%**（同为 GPU 推理路径时） |

差距主要来自：① Albatross 更激进的 kernel 融合；② CUDA Graph 捕获减少启动开销；③ CUDA 生态成熟的硬件特定库。这是**可移植性（Vulkan）换取峰值性能（CUDA）的取舍**，而非实现缺陷。

在以上两者的启发下，本仓库随后独立演进，新增了它们都没有的能力：**两路权重量化推理**（fp16 / int8 自动路由）、**CPU fp32 参考实现与 GPU 内核正确性单测**、**离线量化工具链**与可复现的精度验证工作流。

## 9. 已知限制与后续方向

- **prefill dequant 开销 ~10%**（T=512 时 int8 vs fp16）：彻底消除需 cooperative-matrix 融合反量化+GEMM（Marlin 式零副本），列为长期方向。
- **prefill 每 token 耗时随 T 近二次方增长**（T=256: 0.33ms → T=512: 0.81ms，WKV 并行形式的 chunk 内项），长 prompt 可关注 WKV 分块策略。
- **精度增强后路**：校准加权 k-means、G=64 提高元数据密度。

## 10. License

MIT。依赖明细见 [Cargo.toml](Cargo.toml)。本仓库偏研究向，商用前请自行核对所用模型权重（RWKV-7 Goosed 权重许可）与第三方依赖的许可。