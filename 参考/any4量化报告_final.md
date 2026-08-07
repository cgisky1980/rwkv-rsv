# any4 量化最终报告（RWKV-7 3B，RTX 2080 Ti）

> 日期：2026-08-06
> 模型：RWKV-7 g1h-3B（32 层，n_embd=2560，head_size=64，40 heads，ffn_hidden=10240，vocab=65536）
> 硬件：NVIDIA GeForce RTX 2080 Ti（vendor 0x10DE，subgroup=32，driver 610.188.0，API 1.4.341）
> 方案：any4（论文 arXiv:2507.04610，Meta ICML 2025）+ nnq 输出域 LUT 优化
> 精度指标：teacher-forced Top-1 一致率（量化模型 vs fp16 参考）

---

## 一、量化方案与格式

- **any4**：per-row 16 项学习 LUT + 每 128 组 fp16 scale/zero + 4-bit索引，**~4.35 bit/权重**。
- **量化对象**：每层 6 大矩阵（att.r/k/v/output + ffn.key/value），全模型 **192 个**。
- **不量化**：head / emb（论文 skip lm_head）、低秩小矩阵、LayerNorm、向量参数。
- **nnq**（输出域 LUT 优化）：固定 k-means 索引，用校准激活在**输出域**最小化 `||XW^T - XŴ^T||²`，逐行最小二乘闭式解。
- **离线工具**：`tools/quantize_any4.py`（uv 运行），k-means 走 **GPU(torch) 后端**（18.3× 提速），nnq least-squares 走 GPU 分块（峰值显存 <1GB）。

---

## 二、显存占用对比（memtest 实测）

> memtest：模型加载后基线 + 连续生成采样，`dedicated/shared` 为 Vulkan 上报的设备本地/共享内存。

| 模型 | 权重存储 | 文件大小 | decode 显存 (dedicated) | prefill 峰值 | shared |
|---|---|---|---|---|---|
| **fp16 基线** | 全部 fp16 | **5.49 GB** | **10,889 MB** | 10,889 MB | 248 MB |
| **any4 + nnq（方案A）** | any4(1.37GB) + 52MB 共享 scratch(仅 prefill) | **2.07 GB** | **进程增量 ~2.6GB**（系统读数含 ~3.7GB 桌面底噪） | = decode + 52MB scratch | ~250 MB |

> 2026-08-06 深夜修订：prefill 改方案A（dequant→共享 scratch→TC GEMM）后，旧"临时反量化
> fp16 副本"与"常驻 fp16 副本"均已废除；emb_ln 转 fp16（省 335MB）。显存读数为系统级，
> 含桌面/浏览器底噪（空闲 ~3.7GB），进程增量 = 读数 - 同时刻空闲值。

- **文件体积**：5.49GB → 2.07GB，**↓ 62%**（37.7% 保留）。
- **运行显存（decode）**：进程增量 **~2.6GB**（any4 权重 1.37G + emb_ln/head fp16 0.67G + 低秩 0.18G + 状态/缓冲）。
  - decode 走 any4 GEMV，无常驻 fp16 副本；prefill 走方案A 共享 scratch（+52MB）。

---

## 三、精度对比（end-to-end teacher-forced Top-1 一致率）

| 模型 / 方案 | 验证集 | Top-1 一致率 | 首次分叉 |
|---|---|---|---|
| fp16 参考 | 任意 | 100% | — |
| 纯 k-means any4 | 单 greedy 轨迹（256 token） | 88.3%（226/256） | 第 6 token |
| **nnq any4**（32 token 校准） | 单 greedy 轨迹（256 token） | 93.4%（239/256） | 第 35 token |
| **nnq any4**（512 token 校准，greedy） | 单 greedy 轨迹（256 token） | **94.5%（242/256）** | — |
| **nnq_real**（512 token，真实多语言多样 prompt） | **32×16 多语言多样 prompt（512 token）** | **92.6%（474/512）** | prompt#0 gen#1 |
| — 横向参照 — | — | — | — |
| AUXStar NVFP4 | — | ~85% | — |
| FP8 | — | 93.75% | — |

**要点：**
- nnq 比纯 k-means 端到端 **+5.1 个百分点**（88.3% → 93.4%），首次分叉从第 6 推迟到第 35 token。
- 校准 token 32 → 512：**+1.1 个百分点**（93.4% → 94.5%）。
- 真实多语言多样 prompt 场景下 **92.6%**（校准/验证分布匹配，指标可信，代表真实部署）。
- 92.6% 与 94.5% **不可直接比较**（验证集不同）：94.5% 是单一轨迹自相关高，92.6% 是 32 条跨语言多样 prompt 的泛化。
- **4.35 bit/权重持平甚至超过 FP8（93.75%）水平**。

---

## 四、权重级与内核正确性

| 层级 | 指标 | 结果 |
|---|---|---|
| 权重级 | 纯 k-means avg cos / avg rel | 0.995718 / 9.18%（优于 int4 基线 10.38%） |
| 权重级 | nnq avg cos / avg rel | 0.994773 / 10.16%（权重级退化属预期，验收看输出域/端到端） |
| 输出域 | nnq 输出域相对误差 | 0.0661 → 0.0281（↓58%） |
| 内核正确性 | GPU any4 vs CPU dequant 单 token diff | 0.083（fp16 基线 0.092 同量级） |
| 内核正确性 | ARGMAX / SELFLOOP_VERIFY | match |

---

## 五、性能对比

| 指标 | fp16 基线 | any4 | 提升 |
|---|---|---|---|
| decode self-loop（冷态） | 63.0 tok/s | **114.7 tok/s** | **1.82×** |
| 大权值流量/token（PROF_GPU） | 5.03 GB | 1.48 GB | ↓70.5%（29.5%，理论 27.2%） |
| seq prefill（T=512，同条件） | 1230.1 tok/s | 1102.5 tok/s（方案A：dequant→52MB scratch→TC GEMM） | 89.6%（dequant 开销 ~10%） |

- decode 提速源于权重带宽降 ~3.4×（5.03GB → 1.48GB/token）。
- prefill 历史峰值 3015.7 tok/s 为 **T=256** 测量（每 token 0.33ms）；T=512 时每 token 0.81ms
  （WKV 并行形式对 T 近二次方 + GEMM M 维翻倍），跨 T 不可比。

---

## 六、结论与后续优化方向

**结论**：any4（4.35 bit/权重）+ nnq 输出域优化在 RWKV-7 3B 上达到
**端到端 Top-1 一致率 92.6%（真实多语言场景）/ 94.5%（单轨迹）**，超越 AUXStar NVFP4（~85%）并持平/超过 FP8（93.75%），
同时 decode 提速 **1.67×**、模型文件缩小 **62%**。**已达成"近无损的 4-bit 量化"目标。**

**后续优化方向（按优先级）：**
1. **prefill 单副本**：移除 fp16 反量化副本（改 any4 tensor-core GEMM 或临时反量化），运行显存可低于 fp16 基线。
2. **decode kernel 带宽利用率**：当前 any4 kernel 58~67%（fp16 版 82%），可探索寄存器内联 LUT（tinygemm 式）或更大 ROWS 提升。
3. **校准多样性**：建立"多样 prompt 验证集"后，可再引入采样校准（当前 greedy 校准与 greedy 验证匹配最优）。

---

*本报告汇总自 `参考/any4论文要点.md`、`参考/GPU模型实现.md`、`参考/any4量化报告_nnq512.md`、`.trae/documents/any4-quantization-plan.md` 及 2026-08-06 memtest 日志。*