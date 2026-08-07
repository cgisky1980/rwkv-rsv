# rwkv-rsv

[中文](#中文) · [English](#english)

**RWKV-7 推理引擎（Rust + Vulkan 计算着色器）**，无 CUDA 依赖，可跨 GPU 厂商与平台运行。支持 **fp16 / any4(4-bit) / int8(8-bit)** 三路权重量化推理，按模型文件自动路由、互斥共存；附 **CPU fp32 参考实现**用于精度验证与内核调优。

RWKV-7 inference engine in **Rust + Vulkan compute shaders** — no CUDA dependency, portable across GPU vendors and platforms. Supports **fp16 / any4(4-bit) / int8(8-bit)** weight-quantization paths auto-routed by model file, coexisting exclusively, plus a **CPU fp32 reference** for accuracy verification.

---

## 中文

### 目录

1. [概述](#1-概述)
2. [设计目标](#2-设计目标)
3. [模型参数（默认验证模型）](#3-模型参数默认验证模型)
4. [RWKV-7 架构要点](#4-rwkv-7-架构要点)
5. [三路量化推理路径](#5-三路量化推理路径)
6. [量化文件格式协议](#6-量化文件格式协议)
7. [跨硬件自适应](#7-跨硬件自适应)
8. [构建](#8-构建)
9. [运行](#9-运行)
10. [量化工具链](#10-量化工具链)
11. [精度与性能基准](#11-精度与性能基准)
12. [目录结构](#12-目录结构)
13. [与信天翁 Albatross、cryscan/rosalia 的关系](#13-与信天翁-albatrosscryscanrosalia-的关系)
14. [已知限制与后续方向](#14-已知限制与后续方向)
15. [License](#15-license)

### 1. 概述

`rwkv-rsv` 是一个纯 Rust 实现的 **RWKV-7** 推理引擎。推理全部运行在 **Vulkan 计算着色器**上，无 CUDA 依赖，因此天然跨 GPU 厂商（NVIDIA / AMD / Intel / 移动端）与操作系统（Windows / Linux / macOS）。

核心目标有二：

1. **降低显存占用**：通过离线权重量化（any4 4-bit 省 62%、int8 8-bit 省 41%）在不大幅损失精度的前提下，把 3B 模型的权重从 5.49GB（fp16）压到 2~3GB。
2. **可复现的精度验证**：提供 CPU fp32 参考实现与 GPU 内核级单测，可把「内核误差」与「量化误差」隔离，逐层逐 token 验证。

默认面向 **RWKV-7 Goosed g1h-3B** 模型，但模型结构按 safetensors 张量形状自适应（层数、隐藏维、头数、低秩中间维均从形状推导），可加载同系其他规模模型。

### 2. 设计目标

- **可移植性优先**：Vulkan 而非 CUDA，牺牲约 40% 峰值吞吐，换取跨厂商 / 跨平台可用性。
- **三路量化共存**：fp16（无损参考）/ int8（近无损）/ any4（高省存）按模型文件自动路由，同一二进制全部兼容。
- **运行时自编译 shader**：`build.rs` 在构建期用 `glslangValidator` 把 `*.comp` 编译为 SPIR-V，`constant_id` 做跨硬件自适应。
- **研发向**：内嵌 CPU fp32 参考、DIAG 三方核对、logits 对比、teacher-forced Top-1 一致率等验证工具链。

### 3. 模型参数（默认验证模型）

| 参数 | 值 |
|---|---|
| 模型 | RWKV-7 Goosed `g1h-3B` |
| 层数（n_layer） | 32 |
| 隐藏维（n_embd / C） | 2560 |
| 注意力头数（n_head） | 40 |
| 头维度（head_size / N） | 64 |
| FFN 隐藏维（ffn_hidden） | 10240 |
| 词表（vocab） | 65536 |
| 低秩中间维（w_mid / a_mid / v_mid / g_mid） | 从 safetensors 形状自动推导 |

> 所有超参均从 safetensors 张量形状推导（层数取 `blocks.*` 最大值 +1，头数/头维取 `r_k` 形状，低秩中间维取 `w1/a1/v1/g1` 中不等于 C 的那个维度），因此可加载同系其他规模模型而无需改代码。

**每层权重张量清单**（`blocks.{i}.` 前缀，`[M, K]` 为存储形状，行主序）：

| 类别 | 张量 | 形状 | 是否量化 |
|---|---|---|---|
| 注意力投影 | `att.receptance.weight` | 2560 × 2560 | ✅ 量化 |
| 注意力投影 | `att.key.weight` | 2560 × 2560 | ✅ 量化 |
| 注意力投影 | `att.value.weight` | 2560 × 2560 | ✅ 量化 |
| 注意力投影 | `att.output.weight` | 2560 × 2560 | ✅ 量化 |
| FFN 投影 | `ffn.key.weight` | 10240 × 2560 | ✅ 量化 |
| FFN 投影 | `ffn.value.weight` | 2560 × 10240 | ✅ 量化 |
| token-shift 系数 | `att.x_r / x_w / x_k / x_v / x_a / x_g` | [C] | ❌ 保留 |
| 注意力偏置 | `att.w0 / a0 / v0` | [C] | ❌ 保留 |
| 低秩门控 | `att.w1/w2, a1/a2, v1/v2, g1/g2` | [C,mid]/[mid,C] | ❌ 保留 |
| WKV 头参数 | `att.r_k` [H,N], `k_k`, `k_a` | [C] | ❌ 保留 |
| 归一化 | `ln1, ln2, ln_x, ln0, ln_out` weight/bias | [C] | ❌ 保留 |
| FFN shift | `ffn.x_k` | [C] | ❌ 保留 |

全模型量化矩阵共 **6 × 32 = 192** 个。不量化 head/emb（直控 logits，论文 any4 亦 skip lm_head）、LayerNorm、低秩小矩阵（仅占约 1.7% 流量）。

### 4. RWKV-7 架构要点

每层由 **Time Mixing + Channel Mixing** 组成，严格对齐 numpy baseline 实现（见 [src/model.rs](src/model.rs)）。

**Time Mixing（token 位移 + 线性注意力 / DPLR 状态）：**

```
ln1 = LayerNorm(x)
xr…xg = lerp(ln1, prev, x_*)              // token shift，6 路 lerp
r   = xr @ receptance_w                    // [C]
k   = xk @ key_w; v = xv @ value_w          // [C]
v   = lerp(v, v_first, sigmoid(v0 + xv @ v1 @ v2))   // v_first 门控回退
w   = exp( -sigmoid(w0 + tanh(xw @ w1) @ w2) / √e )  // 衰减因子 ∈ [0.545, 1.0]
a   = sigmoid(a0 + xa @ a1 @ a2)
kk  = L2_norm( k * k_k )                    // 按 head 分组 L2
k_mod = lerp(k, k * a, k_a)
S   = S * w + (S @ kk) * (-kk * a) + v ⊗ k_mod    // DPLR 状态更新
y   = S @ r + GroupNorm · ln_x
y  += Σ_h ( Σ_j r·k_mod·r_k ) · v          // sum_rk_rk 直连项
g   = sigmoid(xg @ g1) @ g2
x  += (y * g) @ output_w
```

**Channel Mixing：**

```
ln2 = LayerNorm(x)
xb  = lerp(ln2, prev, ffn_x_k)
x  += relu(xb @ ffn_key_w)² @ ffn_value_w
```

**RNN 状态**（每层）：`tmix_x` [C]、`tmix_rnn` [H, N, N]（DPLR 状态）、`cmix_x` [C]。推理为 O(1) 状态、逐 token 自回归。

### 5. 三路量化推理路径

权重按模型文件**自动路由**（优先级 int8 → any4 → fp16），互斥共存，同一二进制兼容三种模型：

| 路径 | 权重位宽 | 格式 | 3B 权重文件 | 省存 | 权重级精度 | 端到端 Top-1 一致率 |
|---|---|---|---|---|---|---|
| fp16 | 16-bit | 不量化 | 5.49 GB | — | 无损（参考） | 100% |
| int8 | 8-bit | 非对称 per-group(128)，scale/zero，无 LUT | 3.22 GB | 41% | avg cos **0.999820** / rel **0.6135%** | **98.8%**（506/512） |
| any4 | 4-bit | per-row k-means LUT(16) + per-group scale/zero，约 4.35 bit/权重 | 2.07 GB | 62% | avg cos 0.995718 / rel 9.18% | **94.5%**（nnq512） |

- **量化对象**：每层 6 大矩阵（att.receptance/key/value/output + ffn.key/value），全模型 192 个；head/emb、LayerNorm、低秩小矩阵不量化。
- **int8 精度优势**：相比 any4，int8 逐组均匀量化无 k-means 簇边界非线性，不出现 any4 在分布外的 logit 尺度压缩/排序翻转问题，是「近无损」档。
- **路由依据**：`MODEL_PATH` 指向的模型文件中是否存在 `.int8_idx` / `.any4_idx` 键。

### 6. 量化文件格式协议

矩阵存储 `[M, K]`，K 为收缩维，组大小 `group = 128`。反量化均在 GPU shader / CPU 参考中即时完成，**不常驻 fp16 副本**。

**int8（非对称 per-group 均匀量化，256 级，无 LUT）：**

| safetensors 键后缀 | 形状 | dtype | 说明 |
|---|---|---|---|
| `{name}.int8_idx` | [M, K] | U8（每 uint32 打包 4 字节，低位在前） | 每权重 1 字节，0..255 |
| `{name}.int8_sz` | [M, K/128] | U32 | 每元素 = (scale: fp16 低16位 \| zero: fp16 高16位) |

反量化：`w[m,k] = scale[m, k/128] * idx[m,k] + zero[m, k/128]`，其中 `scale = (max-min)/255`，`zero = min`。常数组（scale=0）精确重建为 zero。

**any4（per-row k-means 学习码本 + per-group scale/zero，arXiv:2507.04610）：**

| safetensors 键后缀 | 形状 | dtype | 说明 |
|---|---|---|---|
| `{name}.any4_idx` | [M, K/2] | U8 | 每字节 2 个 4-bit 索引（低 nibble=偶数 k，高 nibble=奇数 k） |
| `{name}.any4_lut` | [M, 16] | F16 | 每行 16 项学习码本（per-row k-means 质心，定义在组归一化域） |
| `{name}.any4_sz` | [M, K/128] | U32 | 每元素 = (scale: fp16 低16位 \| zero: fp16 高16位) |

反量化：`w[m,k] = scale[m, k/128] * lut[m, idx[m,k]] + zero[m, k/128]`，约 **4.35 bit/权重**（K=2560 时权重流量为 fp16 的 27.2%）。

> 格式协议与 GPU shader 解包逻辑由 [src/model.rs](src/model.rs) 的 CPU 单测（`dequant_any4` / `dequant_int8`）锁定，保证 Python 量化器打包 → Rust/GPU 解包逐位一致。

### 7. 跨硬件自适应

启动时自动检测 `vendor_id`、`subgroup_size`、`max_compute_shared_memory_size`、`max_compute_work_group_size`、`device_type`，据此选择最优参数（详见 [参考/GPU模型实现.md](参考/GPU模型实现.md)）：

**GEMM 瓦片尺寸自适应（`gemm_tile()`）：**

| 硬件类型 | 条件 | TILE_M | TILE_N | TILE_K |
|---|---|---|---|---|
| NVIDIA 高端 | shared ≥ 163840 | 128 | 128 | 32 |
| NVIDIA 中端 | shared ≥ 98304 | 64 | 64 | 64 |
| NVIDIA 低端 | 其他 | 64 | 64 | 32 |
| AMD 高端 | shared ≥ 65536 | 64 | 64 | 32 |
| AMD 低端 | 其他 | 32 | 32 | 32 |
| Intel | 所有 | 32 | 32 | 32 |
| 其他 | 所有 | 64 | 64 | 32 |

共享内存使用量超限时自动降级（先减 TILE_K，再减 TILE_M/TILE_N，最终回退最小配置）。所有使用 subgroup 运算的 shader 均参数化 `SUBGROUP_SIZE`（`constant_id`），保证 AMD（wave64）、Intel（可变）等硬件的正确性。

**可环境变量覆盖：** `GEMM_TILE_M/N/K`、`GEMV_BLOCK_SIZE`、`GEMV_ROWS`。

### 8. 构建

需要 **Rust（edition 2024）** 与 **Vulkan SDK**（`glslangValidator`；`build.rs` 在构建时把 `assets/shaders/src/*.comp` 编译为 `assets/shaders/spv/*.spv`）。

```bash
cargo build --release
```

### 9. 运行

加载模型（默认 `c:\work\niceui\rwkv-g1h-3B.st`，可用 `MODEL_PATH` 切换）：

```bash
cargo run --release
```

主要 env 变量（详见 [src/main.rs](src/main.rs)）：

| 变量 | 作用 |
|---|---|
| `MODEL_PATH` | 模型路径；含 `.int8_idx` → int8，含 `.any4_idx` → any4，否则 fp16 |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | 多 prompt 的 teacher-forced Top-1 一致率验证 |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | 单 prompt 的 CPU fp32 参考一致率 |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | 单 prompt logits 的 RMSE / Top-10 对比 |
| `GEN_TOKENS` / `REPORT_EVERY` | 连续自回归生成（`memtest` 子流程） |
| `DIAG` | 三方核对（seq / tok / CPU）+ dequant 校验 |
| `PROF_GPU` / `PROF_HOST` | GPU / 主机端性能剖析 |
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | 覆盖跨硬件自适应参数 |

### 10. 量化工具链

离线量化器（Python，`uv` 运行）：

```bash
# 4-bit any4
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.any4.st --bits 4
# 8-bit int8
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

主要参数：

| 参数 | 默认 | 说明 |
|---|---|---|
| `--bits` | 4 | 4=any4（LUT+k-means）；8=int8 非对称 per-group（近无损） |
| `--group` | 128 | 量化组大小 |
| `--iters` | 50 | k-means 迭代上限 |
| `--bias-pow` | 1.0 | 有符号幂失真：k-means 偏重重尾极值（官方 any4 同名参数） |
| `--keep-outliers` | off | 每行 LUT 极值项替换为实际极值权重，保证离群值精确重建 |
| `--calib` | — | 校准激活 npz，启用校准加权 k-means（`scale_sample_weight=True`） |
| `--nnq-calib` | — | nnq 输出域 LUT 优化校准（固定索引，最小化输出域 MSE，闭式解 ~100× 快于 Adam） |
| `--device` | auto | k-means 计算设备（有 CUDA 用 cuda，否则 cpu） |

校准 prompt 采集：[tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py)（aya_dataset 多语言分层抽样：中文 30% / 英文 30% / 其他 40%）。

权重级验收阈值（量化器内置 PASS/FAIL）：any4 要求 avg cos ≥ 0.995、avg rel ≤ 10.5% 且优于同组 int4 基线（≈11%）；int8 要求 avg cos ≥ 0.999、avg rel ≤ 1%。

### 11. 精度与性能基准

**硬件：RTX 2080 Ti。** 详细报告见 [参考/](参考/)（`int8量化报告.md`、`any4量化报告_nnq512.md`、`GPU模型实现.md`）。

**端到端精度（teacher-forced Top-1 一致率，对比 fp16 参考）：**

| 路径 | 一致率 | 说明 |
|---|---|---|
| int8 | **98.8%**（506/512） | 近无损，无分布外 logit 压缩/排序翻转 |
| any4 | **94.5%**（nnq512） | 真实多语言 92.6%；对比 AUXStar NVFP4 ~85%、FP8 93.75% |

**性能（fp16 路径）：**

- 单 token 推理：**69.3 tok/s**（14.44 ms/token）
- 序列推理（sequence-parallel）：**398.4 tok/s**（2.51 ms/token）
- argmax 推理：**68.4 tok/s**

**any4 量化收益（同机）：**

- decode self-loop：**114.7 tok/s**（相对核心里程碑 63.0，**1.82×**）
- seq prefill（T=512）：**1102.5 tok/s**（约 fp16 基线 1230.1 的 89.6%，差距为 dequant 开销）
- any4 decode 进程显存增量：**约 2.6GB**

> 显存口径说明：`memtest` 的 dedicated 读数为系统级（含桌面/浏览器底噪），须用「进程增量 = 读数 − 同时刻空闲值」。

### 12. 目录结构

```
src/            Rust 源码（main / model / gpu_model / runtime / vulkan）
assets/shaders/ Vulkan 计算着色器源码（*.comp，spv 为构建产物）
tools/          离线量化与校准工具（Python）
参考/           研发参考文档（量化报告、shader 记录、GPU 模型实现说明等）
test/           开发用脚本
```

### 13. 与信天翁 Albatross、cryscan/rosalia 的关系

本项目最初参考了两份已有的 RWKV 推理实现：

- **信天翁 [Albatross](https://github.com/BlinkDL/Albatross)**（Apache-2.0）：RWKV-7 的 **CUDA** 高性能推理引擎。官方仓库描述 7.2B fp16 在单卡 5090 上可达 **15000+ tps decode**。本仓库的 **kernel 融合（fuse_ka + dplr + group_norm + sum_rk_rk 单次 launch）、sequence-parallel 批量提交、argmax 采样（只回传 token 索引）、pipeline 编译一次复用**等设计均对标/借鉴其思路。
- **[cryscan/rosalia](https://github.com/cryscan/rosalia)**：**Rust + Vulkan 计算着色器**的 RWKV 推理引擎。本仓库在 **Vulkan 内核骨架（ROWS=4 / BLOCK_SIZE=128 / SUBGROUP_SIZE 的 `constant_id` 跨硬件自适应）、运行时结构与部分算子的初始设计**上受其启发。

#### 13.1 与 Albatross 的关键对比

| 维度 | Albatross | rwkv-rsv |
|---|---|---|
| 计算后端 | CUDA（单厂商） | Vulkan（跨厂商/跨平台） |
| 最大吞吐 | 单卡 5090 上 15000+ tps decode | 见下方本机基准（2080 Ti） |
| 路线 | 极致性能，硬件特定优化（cuBLAS/cuDNN、CUDA Graph） | 可移植性优先，运行时自编译 shader |
| 相对差距 | 基准 | 约 **40%**（同为 GPU 推理路径时） |

差距主要来自：① Albatross 更激进的 kernel 融合减少 dispatch；② CUDA Graph 捕获减少启动开销；③ CUDA 生态成熟的硬件特定库。这是**可移植性（Vulkan）换取峰值性能（CUDA）的取舍**，而非实现缺陷。

#### 13.2 独立演进

在以上两者的启发下，本仓库随后独立演进，新增了它们都没有的能力：

- **三路权重量化推理**（fp16 / any4 / int8）按模型文件自动路由、互斥共存；
- **CPU fp32 参考实现**（[src/model.rs](src/model.rs)）与 GPU 内核正确性单测，作为隔离内核误差与量化误差的基准；
- **离线量化工具链**（k-means any4 与 per-group int8）与可复现的精度验证工作流。

### 14. 已知限制与后续方向

- **any4 kernel 带宽利用率 58~67%**（fp16 版 82%）：字节数降后固定开销占比升高，可探索寄存器内联 LUT（tinygemm 式）或更大 ROWS。
- **prefill dequant 开销 ~10%**（T=512 时 1102.5 vs fp16 1230.1 tok/s）：彻底消除需 cooperative-matrix 融合反量化+GEMM（Marlin 式零副本），列为长期方向。
- **prefill 每 token 耗时随 T 近二次方增长**（T=256: 0.33ms → T=512: 0.81ms，WKV 并行形式的 chunk 内项），长 prompt 可关注 WKV 分块策略。
- **精度增强后路**：校准加权 k-means（论文 `sample_weight=calibrate`）、G=64 提高元数据密度。

### 15. License

MIT。依赖明细见 [Cargo.toml](Cargo.toml)。本仓库偏研究向，商用前请自行核对所用模型权重（RWKV-7 Goosed 权重许可）与第三方依赖的许可。

---

## English

### Contents

1. [Overview](#1-overview-1)
2. [Design Goals](#2-design-goals)
3. [Model Parameters](#3-model-parameters)
4. [RWKV-7 Architecture Notes](#4-rwkv-7-architecture-notes)
5. [Three Quantization Paths](#5-three-quantization-paths)
6. [Quantized File Format](#6-quantized-file-format)
7. [Cross-Hardware Adaptation](#7-cross-hardware-adaptation)
8. [Build](#8-build)
9. [Run](#9-run)
10. [Quantization Toolchain](#10-quantization-toolchain)
11. [Accuracy & Performance Benchmarks](#11-accuracy--performance-benchmarks)
12. [Layout](#12-layout)
13. [Relationship with Albatross and cryscan/rosalia](#13-relationship-with-albatross-and-cryscanrosalia)
14. [Known Limitations & Future Work](#14-known-limitations--future-work)
15. [License](#15-license-1)

### 1. Overview

`rwkv-rsv` is a pure-Rust **RWKV-7** inference engine that runs entirely on **Vulkan compute shaders** — no CUDA dependency, portable across GPU vendors (NVIDIA / AMD / Intel / mobile) and platforms (Windows / Linux / macOS).

Two core goals:

1. **Cut VRAM**: offline weight quantization (any4 4-bit saves 62%, int8 8-bit saves 41%) shrinks the 3B model's weights from 5.49GB (fp16) to 2–3GB with little accuracy loss.
2. **Reproducible verification**: a CPU fp32 reference plus GPU-kernel unit tests isolate kernel error from quantization error, verifiable layer-by-layer and token-by-token.

It is validated against **RWKV-7 Goosed g1h-3B**, but the model structure adapts to safetensors tensor shapes, so other models in the same family load without code changes.

### 2. Design Goals

- **Portability first**: Vulkan over CUDA, trading ~40% peak throughput for cross-vendor / cross-platform usability.
- **Three quantization paths coexist**: fp16 (lossless reference) / int8 (near-lossless) / any4 (max savings), auto-routed by model file, one binary for all.
- **Runtime-compiled shaders**: `build.rs` compiles `*.comp` → SPIR-V via `glslangValidator` at build time; `constant_id` enables cross-hardware adaptation.
- **Research-oriented**: CPU fp32 reference, DIAG three-way check, logits comparison, teacher-forced Top-1 agreement, and more.

### 3. Model Parameters

Default validation model **RWKV-7 Goosed `g1h-3B`**:

| Parameter | Value |
|---|---|
| Layers | 32 |
| n_embd (C) | 2560 |
| Heads | 40 |
| head_size (N) | 64 |
| ffn_hidden | 10240 |
| vocab | 65536 |
| Low-rank mid dims (w/a/v/g) | derived from safetensors shapes |

> All hyperparameters are derived from tensor shapes (layers = max `blocks.*` + 1; heads/head_size from `r_k`; low-rank mid dims from the non-C dimension of `w1/a1/v1/g1`).

**Per-layer weight tensors** (`blocks.{i}.`, stored `[M, K]`, row-major):

| Class | Tensor | Shape | Quantized |
|---|---|---|---|
| Attention | `att.receptance.weight` | 2560 × 2560 | ✅ |
| Attention | `att.key.weight` | 2560 × 2560 | ✅ |
| Attention | `att.value.weight` | 2560 × 2560 | ✅ |
| Attention | `att.output.weight` | 2560 × 2560 | ✅ |
| FFN | `ffn.key.weight` | 10240 × 2560 | ✅ |
| FFN | `ffn.value.weight` | 2560 × 10240 | ✅ |
| Token-shift | `att.x_r / x_w / x_k / x_v / x_a / x_g` | [C] | ❌ |
| Attn biases | `att.w0 / a0 / v0` | [C] | ❌ |
| Low-rank gates | `att.w1/w2, a1/a2, v1/v2, g1/g2` | [C,mid]/[mid,C] | ❌ |
| WKV params | `att.r_k` [H,N], `k_k`, `k_a` | [C] | ❌ |
| Norms | `ln1, ln2, ln_x, ln0, ln_out` | [C] | ❌ |
| FFN shift | `ffn.x_k` | [C] | ❌ |

Total quantized matrices: **6 × 32 = 192**. head/emb (directly control logits; any4 paper also skips lm_head), LayerNorm, and low-rank small matrices are kept as-is.

### 4. RWKV-7 Architecture Notes

Each layer = **Time Mixing + Channel Mixing**, strictly aligned with the numpy baseline (see [src/model.rs](src/model.rs)).

**Time Mixing (token shift + linear attention / DPLR state):**

```
ln1 = LayerNorm(x)
xr…xg = lerp(ln1, prev, x_*)              // 6-way token shift lerp
r   = xr @ receptance_w;  k = xk @ key_w;  v = xv @ value_w
v   = lerp(v, v_first, sigmoid(v0 + xv @ v1 @ v2))   // gated v_first rollback
w   = exp( -sigmoid(w0 + tanh(xw @ w1) @ w2) / √e )  // decay ∈ [0.545, 1.0]
a   = sigmoid(a0 + xa @ a1 @ a2)
kk  = L2_norm( k * k_k )                    // per-head L2
k_mod = lerp(k, k * a, k_a)
S   = S * w + (S @ kk) * (-kk * a) + v ⊗ k_mod    // DPLR state update
y   = S @ r + GroupNorm · ln_x
y  += Σ_h ( Σ_j r·k_mod·r_k ) · v          // sum_rk_rk term
g   = sigmoid(xg @ g1) @ g2
x  += (y * g) @ output_w
```

**Channel Mixing:**

```
ln2 = LayerNorm(x)
xb  = lerp(ln2, prev, ffn_x_k)
x  += relu(xb @ ffn_key_w)² @ ffn_value_w
```

**Per-layer RNN state**: `tmix_x` [C], `tmix_rnn` [H, N, N] (DPLR state), `cmix_x` [C]. O(1) state, per-token autoregressive.

### 5. Three Quantization Paths

Weights are **auto-routed by model file** (priority int8 → any4 → fp16), mutually exclusive, one binary for all three:

| Path | Weight bits | Format | 3B weight file | Saved | Weight-level | End-to-end Top-1 |
|---|---|---|---|---|---|---|
| fp16 | 16-bit | none | 5.49 GB | — | lossless (ref) | 100% |
| int8 | 8-bit | asymmetric per-group(128), scale/zero, no LUT | 3.22 GB | 41% | avg cos **0.999820** / rel **0.6135%** | **98.8%** (506/512) |
| any4 | 4-bit | per-row k-means LUT(16) + per-group scale/zero, ~4.35 bit/weight | 2.07 GB | 62% | avg cos 0.995718 / rel 9.18% | **94.5%** (nnq512) |

- **Quantized ops**: 6 matrices per layer (att.receptance/key/value/output + ffn.key/value), 192 total; head/emb, LayerNorm, low-rank kept as-is.
- **int8 advantage**: uniform per-group quantization has no k-means cluster-boundary nonlinearity, so it avoids any4's out-of-distribution logit-scale compression / ranking inversion — a near-lossless tier.
- **Routing**: presence of `.int8_idx` / `.any4_idx` keys in the model file at `MODEL_PATH`.

### 6. Quantized File Format

Matrices stored `[M, K]`, K contraction dim, `group = 128`. Dequantization happens on-the-fly in GPU shaders / CPU reference — **no permanent fp16 copy**.

**int8 (asymmetric per-group uniform, 256 levels, no LUT):**

| safetensors key suffix | Shape | dtype | Notes |
|---|---|---|---|
| `{name}.int8_idx` | [M, K] | U8 (4 bytes packed per uint32, little-endian) | 1 byte/weight, 0..255 |
| `{name}.int8_sz` | [M, K/128] | U32 | (scale: fp16 low16 \| zero: fp16 high16) |

Dequant: `w[m,k] = scale[m, k/128] * idx[m,k] + zero[m, k/128]`, with `scale = (max-min)/255`, `zero = min`. Constant groups (scale=0) rebuild exactly as zero.

**any4 (per-row k-means learned codebook + per-group scale/zero, arXiv:2507.04610):**

| safetensors key suffix | Shape | dtype | Notes |
|---|---|---|---|
| `{name}.any4_idx` | [M, K/2] | U8 | 2 × 4-bit indices/byte (low nibble=even k, high nibble=odd k) |
| `{name}.any4_lut` | [M, 16] | F16 | 16 learned centroids/row (per-row k-means, in group-normalized domain) |
| `{name}.any4_sz` | [M, K/128] | U32 | (scale: fp16 low16 \| zero: fp16 high16) |

Dequant: `w[m,k] = scale[m, k/128] * lut[m, idx[m,k]] + zero[m, k/128]`, ~**4.35 bit/weight** (27.2% of fp16 traffic at K=2560).

> The format contract is locked by CPU unit tests in [src/model.rs](src/model.rs) (`dequant_any4` / `dequant_int8`), guaranteeing bit-exact pack (Python) → unpack (Rust/GPU).

### 7. Cross-Hardware Adaptation

At startup we detect `vendor_id`, `subgroup_size`, `max_compute_shared_memory_size`, `max_compute_work_group_size`, `device_type`, and pick optimal parameters (see [参考/GPU模型实现.md](参考/GPU模型实现.md)):

**GEMM tile (`gemm_tile()`):**

| Hardware | Condition | TILE_M | TILE_N | TILE_K |
|---|---|---|---|---|
| NVIDIA high | shared ≥ 163840 | 128 | 128 | 32 |
| NVIDIA mid | shared ≥ 98304 | 64 | 64 | 64 |
| NVIDIA low | other | 64 | 64 | 32 |
| AMD high | shared ≥ 65536 | 64 | 64 | 32 |
| AMD low | other | 32 | 32 | 32 |
| Intel | all | 32 | 32 | 32 |
| Other | all | 64 | 64 | 32 |

Shared-memory overflow auto-degrades (reduce TILE_K first, then TILE_M/TILE_N, then fall back to minimal config). All subgroup-using shaders parameterize `SUBGROUP_SIZE` via `constant_id` for AMD (wave64) / Intel (variable) correctness.

**Overridable via env:** `GEMM_TILE_M/N/K`, `GEMV_BLOCK_SIZE`, `GEMV_ROWS`.

### 8. Build

Requires **Rust (edition 2024)** and the **Vulkan SDK** (`glslangValidator`; `build.rs` compiles `assets/shaders/src/*.comp` → `assets/shaders/spv/*.spv` at build time).

```bash
cargo build --release
```

### 9. Run

Load a model (default `c:\work\niceui\rwkv-g1h-3B.st`, switchable via `MODEL_PATH`):

```bash
cargo run --release
```

Key env vars (see [src/main.rs](src/main.rs)):

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Model path; `.int8_idx` → int8, `.any4_idx` → any4, otherwise fp16 |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | Multi-prompt teacher-forced Top-1 agreement |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | Single-prompt CPU fp32 reference agreement |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | Single-prompt logits RMSE / Top-10 comparison |
| `GEN_TOKENS` / `REPORT_EVERY` | Continuous autoregressive generation (`memtest` sub-flow) |
| `DIAG` | Three-way check (seq / tok / CPU) + dequant validation |
| `PROF_GPU` / `PROF_HOST` | GPU / host-side profiling |
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | Override cross-hardware adaptation |

### 10. Quantization Toolchain

Offline quantizer (Python, run via `uv`):

```bash
# 4-bit any4
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.any4.st --bits 4
# 8-bit int8
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

Key arguments:

| Arg | Default | Notes |
|---|---|---|
| `--bits` | 4 | 4=any4 (LUT+k-means); 8=int8 asymmetric per-group (near-lossless) |
| `--group` | 128 | quantization group size |
| `--iters` | 50 | k-means iteration cap |
| `--bias-pow` | 1.0 | signed power distortion: biases k-means toward heavy tails (any4 paper) |
| `--keep-outliers` | off | replace each row's LUT extremes with actual extremes for exact outlier rebuild |
| `--calib` | — | calibration activation npz → calibration-weighted k-means (`scale_sample_weight=True`) |
| `--nnq-calib` | — | nnq output-domain LUT optimization (fixed indices, min output MSE, closed form ~100× faster than Adam) |
| `--device` | auto | k-means device (cuda if available, else cpu) |

Calibration prompt collection: [tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py) (aya_dataset multilingual stratified sampling: 30% zh / 30% en / 40% other).

Built-in weight-level acceptance (PASS/FAIL): any4 requires avg cos ≥ 0.995, avg rel ≤ 10.5% and better than the int4 baseline (~11%); int8 requires avg cos ≥ 0.999, avg rel ≤ 1%.

### 11. Accuracy & Performance Benchmarks

**Hardware: RTX 2080 Ti.** Full reports under [参考/](参考/) (`int8量化报告.md`, `any4量化报告_nnq512.md`, `GPU模型实现.md`).

**End-to-end accuracy (teacher-forced Top-1 agreement vs fp16 reference):**

| Path | Agreement | Notes |
|---|---|---|
| int8 | **98.8%** (506/512) | near-lossless; no OOD logit compression / ranking inversion |
| any4 | **94.5%** (nnq512) | real multilingual 92.6%; vs AUXStar NVFP4 ~85%, FP8 93.75% |

**Performance (fp16 path):**

- Single-token inference: **69.3 tok/s** (14.44 ms/token)
- Sequence-parallel: **398.4 tok/s** (2.51 ms/token)
- Argmax inference: **68.4 tok/s**

**any4 quantization gains (same machine):**

- decode self-loop: **114.7 tok/s** (from core milestone 63.0, **1.82×**)
- seq prefill (T=512): **1102.5 tok/s** (~89.6% of fp16 baseline 1230.1; gap = dequant overhead)
- any4 decode process VRAM delta: **~2.6GB**

> VRAM note: `memtest`'s dedicated reading is system-level (includes desktop/browser noise); use "process delta = reading − idle value at the same time".

### 12. Layout

```
src/            Rust sources (main / model / gpu_model / runtime / vulkan)
assets/shaders/ Vulkan compute shader sources (*.comp; spv are build artifacts)
tools/          Offline quantization & calibration tools (Python)
参考/           Research reference docs (quantization reports, shader notes, GPU model notes)
test/           Development scripts
```

### 13. Relationship with Albatross and cryscan/rosalia

This project initially referenced two existing RWKV inference implementations:

- **信天翁 [Albatross](https://github.com/BlinkDL/Albatross)** (Apache-2.0): a high-performance **CUDA** inference engine for RWKV-7 (repo claims 15000+ tps decode for 7.2B fp16 on a single 5090). Our **kernel fusion (fuse_ka + dplr + group_norm + sum_rk_rk in one launch), sequence-parallel batched submission, argmax sampling (transferring only token indices), and compile-once/reuse pipeline** designs draw on / benchmark against its approach.
- **[cryscan/rosalia](https://github.com/cryscan/rosalia)**: an RWKV inference engine in **Rust + Vulkan compute shaders**. Our **Vulkan kernel skeleton (ROWS=4 / BLOCK_SIZE=128 / SUBGROUP_SIZE `constant_id` cross-hardware adaptation), runtime structure, and the initial design of some operators** were inspired by it.

#### 13.1 Key comparison vs Albatross

| Dimension | Albatross | rwkv-rsv |
|---|---|---|
| Compute backend | CUDA (single vendor) | Vulkan (cross-vendor / cross-platform) |
| Peak throughput | 15000+ tps decode on a single 5090 | see local benchmark below (2080 Ti) |
| Philosophy | peak performance, hardware-specific (cuBLAS/cuDNN, CUDA Graph) | portability first, runtime-compiled shaders |
| Relative gap | baseline | about **40%** (same GPU inference path) |

The gap mainly comes from: ① Albatross's more aggressive kernel fusion reducing dispatches; ② CUDA Graph capture cutting launch overhead; ③ CUDA's mature hardware-specific libraries. This is a **portability (Vulkan) vs peak performance (CUDA) trade-off**, not an implementation defect.

#### 13.2 Independent evolution

Inspired by both, this repository has since evolved independently, adding capabilities neither has:

- **Three weight-quantization paths** (fp16 / any4 / int8) auto-routed by model file;
- **CPU fp32 reference** ([src/model.rs](src/model.rs)) plus GPU-kernel correctness unit tests, as a baseline isolating kernel error from quantization error;
- **Offline quantization toolchain** (k-means any4 and per-group int8) with a reproducible accuracy-verification workflow.

### 14. Known Limitations & Future Work

- **any4 kernel bandwidth 58–67%** (fp16 82%): fixed overhead rises after byte count shrinks; explore register-resident LUT (tinygemm-style) or larger ROWS.
- **prefill dequant overhead ~10%** (1102.5 vs fp16 1230.1 tok/s at T=512): full removal needs cooperative-matrix fused dequant+GEMM (Marlin-style zero-copy), a long-term direction.
- **prefill time grows ~quadratically with T** (T=256: 0.33ms → T=512: 0.81ms, WKV parallel intra-chunk term); consider WKV chunking for long prompts.
- **Precision enhancements** (roadmap): calibration-weighted k-means (`sample_weight=calibrate`), G=64 for denser metadata.

### 15. License

MIT. See [Cargo.toml](Cargo.toml) for dependencies. Research-oriented; please verify model weights (RWKV-7 Goosed weight license) and third-party dependency licensing before commercial use.