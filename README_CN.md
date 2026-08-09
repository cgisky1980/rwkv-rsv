# rwkv-rsv（中文版）

[English primary doc](README.md) · [中文版](README_CN.md)

**RWKV-7 推理引擎（Rust + Vulkan 计算着色器）**，无 CUDA 依赖，可跨 GPU 厂商与平台运行。支持 **fp16 / int8(8-bit)** 两路权重量化推理，按模型文件自动路由、互斥共存；附 **CPU fp32 参考实现**用于精度验证与内核调优。

> 主文档为英文（[README.md](README.md)），本文件为中文版，内容与主文档一致。

---

## 中文

### 目录

1. [概述](#1-概述)
2. [设计目标](#2-设计目标)
3. [模型参数（默认验证模型）](#3-模型参数默认验证模型)
4. [RWKV-7 架构要点](#4-rwkv-7-架构要点)
5. [两路量化推理路径](#5-两路量化推理路径)
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

1. **降低显存占用**：通过离线权重量化（int8 8-bit 省 41%）在不大幅损失精度的前提下，把 3B 模型的权重从 5.49GB（fp16）压到 3.22GB。
2. **可复现的精度验证**：提供 CPU fp32 参考实现与 GPU 内核级单测，可把「内核误差」与「量化误差」隔离，逐层逐 token 验证。

默认面向 **RWKV-7 Goosed g1h-3B** 模型，但模型结构按 safetensors 张量形状自适应（层数、隐藏维、头数、低秩中间维均从形状推导），可加载同系其他规模模型。

### 2. 设计目标

- **可移植性优先**：Vulkan 而非 CUDA，牺牲约 40% 峰值吞吐，换取跨厂商 / 跨平台可用性。
- **两路量化共存**：fp16（无损参考）/ int8（近无损）按模型文件自动路由，同一二进制全部兼容。
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

全模型量化矩阵共 **6 × 32 = 192** 个。不量化 head/emb（直控 logits）、LayerNorm、低秩小矩阵（仅占约 1.7% 流量）。

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

### 5. 两路量化推理路径

权重按模型文件**自动路由**（优先级 int8 → fp16），互斥共存，同一二进制兼容两种模型：

| 路径 | 权重位宽 | 格式 | 3B 权重文件 | 省存 | 权重级精度 | 端到端 Top-1 一致率 |
|---|---|---|---|---|---|---|
| fp16 | 16-bit | 不量化 | 5.49 GB | — | 无损（参考） | 100% |
| int8 | 8-bit | 非对称 per-group(128)，scale/zero，无 LUT | 3.22 GB | 41% | avg cos **0.999820** / rel **0.6135%** | **98.8%**（506/512） |

- **量化对象**：每层 6 大矩阵（att.receptance/key/value/output + ffn.key/value），全模型 192 个；head/emb、LayerNorm、低秩小矩阵不量化。
- **int8 精度优势**：逐组均匀量化，无 k-means 簇边界非线性、无 LUT，是「近无损」档。
- **路由依据**：`MODEL_PATH` 指向的模型文件是否存在 `.int8_idx` 键。

### 6. 量化文件格式协议

矩阵存储 `[M, K]`，K 为收缩维，组大小 `group = 128`。反量化均在 GPU shader / CPU 参考中即时完成，**不常驻 fp16 副本**。

**int8（非对称 per-group 均匀量化，256 级，无 LUT）：**

| safetensors 键后缀 | 形状 | dtype | 说明 |
|---|---|---|---|
| `{name}.int8_idx` | [M, K] | U8（每 uint32 打包 4 字节，低位在前） | 每权重 1 字节，0..255 |
| `{name}.int8_sz` | [M, K/128] | U32 | 每元素 = (scale: fp16 低16位 \| zero: fp16 高16位) |

反量化：`w[m,k] = scale[m, k/128] * idx[m,k] + zero[m, k/128]`，其中 `scale = (max-min)/255`，`zero = min`。常数组（scale=0）精确重建为 zero。

> 格式协议与 GPU shader 解包逻辑由 [src/model.rs](src/model.rs) 的 CPU 单测（`dequant_int8`）锁定，保证 Python 量化器打包 → Rust/GPU 解包逐位一致。

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
| `MODEL_PATH` | 模型路径；含 `.int8_idx` → int8，否则 fp16 |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | 多 prompt 的 teacher-forced Top-1 一致率验证 |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | 单 prompt 的 CPU fp32 参考一致率 |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | 单 prompt logits 的 RMSE / Top-10 对比 |
| `GEN_TOKENS` / `REPORT_EVERY` | 连续自回归生成（`memtest` 子流程） |
| `DIAG` | 三方核对（seq / tok / CPU）+ dequant 校验 |
| `PROF_GPU` / `PROF_HOST` | GPU / 主机端性能剖析 |
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | 覆盖跨硬件自适应参数 |

#### 9.1 示例程序（web-rwkv 风格）

自包含可运行示例，位于 `examples/`：

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

#### 9.2 库 API（web-rwkv 风格）与 GPU 采样

`rwkv-rsv` 同时以 **library** 形式导出，便于服务端（如 `ai00-server`）集成，对标 [web-rwkv](https://github.com/cryscan/web-rwkv) 的抽象：

- **`ModelBuilder` + `Bundle`**：加载模型并绑定零初始化 `State`；`infer_tokens` / `infer_seq` / `infer` 推进会话并返回 logits。
- **`State`**：一等公民的会话状态——`state_back()` 下载为 CPU `Vec<f32>`，`state_load()` 回灌，`reset()` 清零。序列化逐位无损（往返 `max_diff == 0`）。
- **GPU 采样（`SamplerParams`）**：`infer_sample` / `infer_sample_selfloop` 全 GPU 端过滤 logits——`temperature`、`top-k`、`top-p`，以及兼容 OpenAI 的 `repetition_penalty` / `frequency_penalty` / `presence_penalty`——只回传采样 token 索引（不再逐 token 下载 logits）。

> 确定性说明：在相同 `State` 下 GPU 前向是确定性的。此前所谓的“非确定性”实为 `Bundle::reset()` 的 bug（重置了模型内部态而非会话态）；`reset()` 现已正确清零工作会话态。

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

权重级验收阈值（量化器内置 PASS/FAIL）：int8 要求 avg cos ≥ 0.999、avg rel ≤ 1%。

### 11. 精度与性能基准

**硬件：RTX 2080 Ti。** 详细报告见 [参考/](参考/)（`int8量化报告.md`、`GPU模型实现.md`）。

**端到端精度（teacher-forced Top-1 一致率，对比 fp16 参考）：**

| 路径 | 一致率 | 说明 |
|---|---|---|
| int8 | **98.8%**（506/512） | 近无损，无分布外 logit 压缩/排序翻转 |

**GPU decode self-loop 吞吐**（argmax self-loop，1000 tokens，GPU 60°C 冷态）：

| 权重 | 后端 | tok/s |
|---|---|---|
| fp16 | Vulkan | 80.0 |
| fp16 | CUDA | 85.7 |
| int8 | Vulkan | 110.6 |
| int8 | CUDA | 110.8 |

> 硬件：RTX 2080 Ti。通过 `cargo run --release -- memtest` 配以 `SELFLOOP_ONLY=1` / `SELFLOOP_N=1000` 测得（纯 GPU argmax self-loop，Batch=N，2026-08-09 GPU 60°C 冷态重测）。int8 较 fp16 提升约 +38%（Vulkan）/+29%（CUDA）。

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

- **两路权重量化推理**（fp16 / int8）按模型文件自动路由、互斥共存；
- **CPU fp32 参考实现**（[src/model.rs](src/model.rs)）与 GPU 内核正确性单测，作为隔离内核误差与量化误差的基准；
- **离线量化工具链**（per-group int8）与可复现的精度验证工作流。

### 14. 已知限制与后续方向

- **prefill dequant 开销 ~10%**（T=512 时 int8 vs fp16）：彻底消除需 cooperative-matrix 融合反量化+GEMM（Marlin 式零副本），列为长期方向。
- **prefill 每 token 耗时随 T 近二次方增长**（T=256: 0.33ms → T=512: 0.81ms，WKV 并行形式的 chunk 内项），长 prompt 可关注 WKV 分块策略。
- **精度增强后路**：校准加权 k-means（论文 `sample_weight=calibrate`）、G=64 提高元数据密度。

### 15. License

MIT。依赖明细见 [Cargo.toml](Cargo.toml)。本仓库偏研究向，商用前请自行核对所用模型权重（RWKV-7 Goosed 权重许可）与第三方依赖的许可。