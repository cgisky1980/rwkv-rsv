# GPU 模型实现与跨硬件自适应

> 日期：2026-08-05
> 目标：记录 RWKV-7 Vulkan 实现的跨硬件自适应机制与各硬件画像，确保在 NVIDIA / AMD / Intel / 移动端等各种 Vulkan 硬件上都能正确运行并发挥接近最优的效果。

## 1. 跨硬件自适应架构

### 1.1 硬件信息检测

系统在启动时自动检测以下硬件信息：

- **vendor_id**：厂商 ID（NVIDIA=0x10DE，AMD=0x1002，Intel=0x8086）
- **subgroup_size**：设备实际 subgroup 大小（NVIDIA=32，AMD=64，Intel 可变）
- **max_compute_shared_memory_size**：共享内存上限
- **max_compute_work_group_size**：最大工作组大小
- **device_type**：设备类型（DISCRETE_GPU / INTEGRATED_GPU）

### 1.2 参数自适应选择

基于检测到的硬件信息，系统自动选择最优参数：

#### GEMM 瓦片尺寸选择（gemm_tile()）

| 硬件类型 | 条件 | TILE_M | TILE_N | TILE_K | 说明 |
|---|---|---|---|---|---|
| NVIDIA 高端 | shared_memory ≥ 163840 | 128 | 128 | 32 | RTX 3080+ 等大共享内存卡 |
| NVIDIA 中端 | shared_memory ≥ 98304 | 64 | 64 | 64 | RTX 2080 Ti 等中端卡 |
| NVIDIA 低端 | 其他 | 64 | 64 | 32 | 低端 NVIDIA 卡 |
| AMD 高端 | shared_memory ≥ 65536 | 64 | 64 | 32 | 高端 AMD 卡 |
| AMD 低端 | 其他 | 32 | 32 | 32 | 低端 AMD 卡 |
| Intel | 所有 | 32 | 32 | 32 | Intel 集成显卡 |
| 其他 | 所有 | 64 | 64 | 32 | 保守默认值 |

#### 共享内存上限校验（gemm_tile()）

系统自动计算当前瓦片配置的共享内存使用量，并与硬件限制比较：

```
SH_A_BUF = TILE_M * (TILE_K + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4
SH_B_BUF = TILE_N * (TILE_K + NUM_ELEMENT_VEC4) / NUM_ELEMENT_VEC4
总共享内存 = (SH_A_BUF + SH_B_BUF) * 16 字节 (uvec4)
```

如果超过硬件限制，系统会自动调整瓦片尺寸：
1. 首先尝试减小 TILE_K（优先保持 M/N 维度）
2. 然后尝试减小 TILE_M 和 TILE_N（保持与 subgroup_size 兼容）
3. 最终回退到最小配置（subgroup_size × subgroup_size × 16）

### 1.3 SUBGROUP_SIZE 参数化

所有使用 subgroup 操作的着色器都已参数化 SUBGROUP_SIZE：

- **gemv_f32io.comp**：constant_id = 11
- **gemv_rkv_stage1.comp**：constant_id = 5
- **gemv_lowrank_chain4.comp**：constant_id = 5
- **norm.comp**：constant_id = 15
- **l2_norm.comp**：constant_id = 2
- **fuse_ka.comp**：constant_id = 2
- **sum_rk_rk.comp**：constant_id = 2

这确保了 NUM_SUBGROUPS = BLOCK_SIZE / SUBGROUP_SIZE 在所有硬件上都能正确计算。

## 2. 硬件画像

### 2.1 NVIDIA

#### RTX 2080 Ti (sm_75)
- **subgroup_size**: 32
- **shared_memory**: 64KB/SM
- **特点**: 当前基线，所有参数已针对此硬件优化
- **推荐参数**: GEMM_TILE=(64,64,64), GEMV_BLOCK_SIZE=128, GEMV_ROWS=4

#### RTX 3080+ (sm_80/90)
- **subgroup_size**: 32
- **shared_memory**: 164KB/SM
- **特点**: 更大的共享内存，支持更大的瓦片尺寸
- **推荐参数**: GEMM_TILE=(128,128,32), GEMV_BLOCK_SIZE=256, GEMV_ROWS=8

### 2.2 AMD

#### RDNA/GCN 架构
- **subgroup_size**: 64 (wave64)
- **shared_memory**: 64KB/SM
- **特点**: subgroup 大小为 64，BLOCK_SIZE 需要为 64 的倍数
- **推荐参数**: GEMM_TILE=(64,64,32), GEMV_BLOCK_SIZE=256, GEMV_ROWS=4

### 2.3 Intel

#### Arc/集成显卡
- **subgroup_size**: 8/16/32 可变
- **shared_memory**: 64-128KB
- **特点**: subgroup 大小不固定，需要动态适配
- **推荐参数**: GEMM_TILE=(32,32,32), GEMV_BLOCK_SIZE=64, GEMV_ROWS=2

### 2.4 移动端

#### Adreno/Mali
- **subgroup_size**: 可变
- **shared_memory**: 较小
- **特点**: 低功耗，对 occupancy 敏感
- **推荐参数**: 保守默认值，避免过大瓦片

## 3. 环境变量覆盖

所有自适应参数都可以通过环境变量覆盖，便于手动调优：

- **GEMM_TILE_M/N/K**：覆盖 GEMM 瓦片尺寸
- **GEMV_BLOCK_SIZE**：覆盖 GEMV 块大小
- **GEMV_ROWS**：覆盖 GEMV 行数

示例：
```bash
# Windows PowerShell
$env:GEMM_TILE_M=128; $env:GEMM_TILE_N=128; $env:GEMM_TILE_K=32; cargo run --release

# Linux/macOS
GEMM_TILE_M=128 GEMM_TILE_N=128 GEMM_TILE_K=32 cargo run --release
```

## 4. 性能优化历程

### 4.1 已实现的优化

1. **SUBGROUP_SIZE 参数化**：解决了 AMD/Intel 硬件上的正确性问题
2. **GEMM 瓦片自适应**：根据硬件能力选择最优瓦片尺寸
3. **共享内存上限校验**：防止在大瓦片配置上创建失败
4. **厂商特定优化**：针对 NVIDIA/AMD/Intel 的不同架构特点进行优化

### 4.2 性能基准

在 RTX 2080 Ti 上的基准测试结果：

- **单 token 推理**: 69.3 tokens/s (14.44 ms/token)
- **序列推理**: 398.4 tokens/s (2.51 ms/token)
- **argmax 推理**: 68.4 tokens/s (14.62 ms/token)

### 4.3 与 Albatross 的差距

当前实现与 Albatross (CUDA) 的差距约为 40%，主要原因：

1. **kernel 融合**: Albatross 通过 kernel 融合减少了 dispatch 次数
2. **CUDA Graph**: Albatross 使用 CUDA Graph 捕获减少了启动开销
3. **硬件特定优化**: CUDA 有更成熟的硬件特定优化库（cuBLAS/cuDNN）

## 5. any4 权重量化（decode 路径）

> 论文：any4: Learned 4-bit Numeric Representation for LLMs（Meta, ICML 2025, arXiv:2507.04610）
> 详细算法核对与验证记录见 `参考/any4论文要点.md`，权重级误差见 `参考/any4量化报告.md`。

### 5.1 格式与量化范围

对大权值矩阵 W[M, K]（行主序，K 为收缩维），group_size G=128：

| safetensors 键后缀 | 形状 | dtype | 说明 |
|---|---|---|---|
| `{name}.any4_idx` | [M, K/2] | U8 | 每字节 2 个 4-bit 索引（低 nibble=偶数 k） |
| `{name}.any4_lut` | [M, 16] | F16 | 每行 16 项学习码本（per-row k-means 质心） |
| `{name}.any4_sz` | [M, K/128] | U8→uint32 | scale fp16 低 16 位 \| zero fp16 高 16 位 |

反量化：`w[m,k] = scale[m,k/128] * lut[m, idx] + zero[m,k/128]`，约 **4.35 bit/权重**
（K=2560 时权重流量为 fp16 的 27.2%）。原 fp16 键被 any4 三键替换。

- **量化对象**（每层 6 矩阵，全模型 192 个）：att.receptance/key/value/output + ffn.key/value。
- **不量化**：head/emb（直控 logits，论文 skip lm_head）、低秩小矩阵（w/a/v/g 的 1/2 级，
  仅占 1.7% 流量）、LayerNorm、x_r/k_k/r_k 等向量参数。
- 离线量化器：`tools/quantize_any4.py`（uv 运行，per-row k-means 16 簇，分位数确定性初始化）。

### 5.2 GPU 实现

- **shader**（沿用 ROWS=4 / BLOCK_SIZE=128 / SUBGROUP_SIZE constant_id 跨硬件骨架）：
  - `gemv_any4.comp`（relu2 变体）→ ffn.key
  - `gemv_any4_add.comp`（MUL 0/1 变体）→ ffn.value（残差累加）/ att.output（y_g 门控折叠）
  - `gemv_any4_rkv_stage1.comp` → r/k/v 三投影 any4 化 + v1/w1/a1/g1 mid 投影（fp32）深度融合
  - 每 workgroup 把 ROWS 行的 LUT 与 scale/zero 协作加载到 shared memory，主循环每线程
    每次读 1 个 uint32（8 权重）解包 nibble 查 LUT，fma 累加。
- **加载**：`GpuLayer::load` 探测 `.any4_idx` 键自动切换（单二进制兼容 fp16/any4 模型）。
  any4 矩阵**不再常驻 fp16 副本**（旧方案两副本共存 ~6.4GB 已废除）。
- **prefill（方案A，2026-08-06 起）**：`dequant_any4_f16.comp` 把 any4 逐层反量化到**一块
  52.4MB 共享 fp16 scratch**（= 最大矩阵 ffn 10240×2560），复用 fp16 tensor-core GEMM。
  每矩阵 GEMM 前全量覆写 scratch，顺序复用由 record_kernel 读写 barrier 保证。
  旧标量 any4 GEMM prefill 保留为 `ANY4_GEMM_PREFILL=1` 可选路径（慢 ~5.7×，显存极限备用）。
- **emb_ln fp16 化**：GPU 端 embedding 表 [vocab, C] 存 fp16（671→335MB），gather 走
  `gather_row_f16.comp`；CPU 缓存存同一 f16 舍入值回读的 f32，保证 seq（CPU 上传）与
  tok（GPU gather）输入逐位一致。
- **CPU 参考**：model.rs `linear_to_f32` 对 any4 键透明反量化，DIAG 的 CPU 基准与被测
  GPU 权重同源，可隔离内核误差与量化误差。

### 5.3 验证结果（192 矩阵全模型，2026-08-06 深夜复测，均 PASS）

| 层级 | 指标 | 结果 |
|---|---|---|
| 权重级 | avg cos / avg rel | 0.995718 / 9.18%（优于 int4 基线 10.38%） |
| 内核正确性 | GPU any4 vs CPU dequant | 单 token diff 0.083（fp16 基线 0.092 同量级）；ARGMAX/SELFLOOP_VERIFY match |
| dequant kernel | GPU dequant vs CPU 参考 | max_diff=0.000488，bad(>1e-3)=0/6,553,600 |
| 端到端 | **teacher-forced Top-1 一致率**（256 token，nnq512） | **94.5%**（AUXStar NVFP4 ~85%、FP8 93.75%）；真实多语言 92.6% |
| seq-vs-tok | DIAG 单 token / 512 token | 0.19 / 7.25（fp16 基线 0.11 / 8.94，同量级，RNN 混沌累积） |
| 性能 | decode self-loop（冷态） | 63.0 → **114.7 tok/s（1.82×）** |
| 性能 | seq prefill（T=512） | **1102.5 tok/s = 同条件 fp16 基线 1230.1 的 89.6%**（差距=dequant 开销 ~48ms） |
| 显存 | decode 进程增量 | **~2.6GB**（any4 权重 1.37G + emb_ln/head fp16 0.67G + 低秩 0.18G + 状态/缓冲） |

验证工具：`DIAG=1`（seq/tok/CPU 三方 + dequant 校验）、`SAVE_LOGITS`/`COMPARE_LOGITS`（跨模型
logits）、`TOP1_REF_SAVE`/`TOP1_REF_COMPARE`（teacher-forced 一致率）、`PROF_GPU=1`（带宽）。

> 显存口径注意：memtest 的 dedicated 读数为**系统级**（含桌面/浏览器等底噪，空闲即可达
> ~3.7GB），跨时点绝对值不可直接比较；须用"进程增量 = 读数 - 同时刻空闲值"。

### 5.4 已知限制与后续增强

- any4 kernel 带宽利用率 58~67%（fp16 版 82%）：字节数降后固定开销占比升高，
  可探索寄存器内联 LUT（4 级 2:1 mux，tinygemm 式）或更大 ROWS。
- prefill dequant 开销 ~10%（T=512 时 1102.5 vs fp16 1230.1 tok/s）：彻底消除需
  cooperative-matrix 融合反量化+GEMM（方案C，Marlin 式零副本），列为长期方向。
- prefill 每 token 耗时随 T 近二次方增长（T=256: 0.33ms → T=512: 0.81ms，WKV 并行形式
  的 chunk 内项），长 prompt 场景可关注 WKV 分块策略。
- 精度增强后路：校准加权 k-means（论文 `sample_weight=calibrate`）、G=64 提高元数据密度。

## 6. 未来优化方向

### 6.1 短期优化

1. **GEMV 参数自适应**：目前 GEMV 参数仍固定，未来可实现类似 GEMM 的自适应选择
2. **自动调优**：实现启动时 micro-benchmark 自动选择最优参数
3. **更多硬件画像**：添加更多具体硬件型号的优化参数

### 6.2 长期优化

1. **kernel 融合**：减少 dispatch 次数，接近 Albatross 的性能
2. **异步双缓冲**：隐藏内存传输延迟
3. **多 GPU 支持**：利用多 GPU 并行计算

## 7. 故障排除

### 7.1 常见问题

1. **共享内存不足**：
   - 症状：启动时出现共享内存相关错误
   - 解决：系统会自动调整瓦片尺寸，或手动设置 GEMM_TILE_* 环境变量

2. **subgroup 相关错误**：
   - 症状：在非 NVIDIA 硬件上运行失败
   - 解决：确保已正确实现 SUBGROUP_SIZE 参数化

3. **性能不佳**：
   - 症状：推理速度明显低于预期
   - 解决：检查硬件画像是否正确识别，尝试手动调整参数

### 7.2 调试工具

1. **PROF_GPU=1**：启用 GPU 时间戳剖析，显示每个 kernel 的执行时间和带宽利用率
2. **PROF_HOST=1**：启用主机端性能剖析，显示 CPU 端耗时分布
3. **PEAK_GBS**：覆盖默认的峰值带宽值（默认 616 GB/s，RTX 2080 Ti）

## 8. 贡献指南

### 8.1 添加新硬件支持

1. 在 `gemm_tile()` 函数中添加新的硬件画像
2. 更新本文档中的硬件画像表格
3. 在目标硬件上验证正确性和性能

### 8.2 优化现有实现

1. 使用 PROF_GPU=1 分析性能瓶颈
2. 尝试调整参数并测量性能变化
3. 提交优化参数和性能数据

---

> 本文档将随着硬件支持的扩展和性能优化的进展持续更新。
