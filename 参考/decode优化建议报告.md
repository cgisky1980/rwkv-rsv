# RWKV-7 3B decode 路径 kernel 优化建议报告

> 生成日期：2026-08-05
> 数据来源：`logs/rwkv-rsv_20260805_035529.log`（PROF_GPU 时间戳剖析）
> 状态：**P0 已完成（2026-08-05）**，**P1 已完成（2026-08-05）**

## 1. 基准信息

- **模型**：RWKV-7 3B（g1h-3B）
  - n_layer=32, C(n_embd)=2560, ffn_hidden=10240, vocab=65536
  - H(n_head)=40, N(head_size)=64
  - mid：vm=64, wm=96, am=96, gm=320
- **单 token decode 基线**：共 **17.7ms**（≈56.5 tok/s），**258 次 dispatch**（≈8 次/层 × 32 层）
- **GPU**：NVIDIA GeForce RTX 2080 Ti，峰值带宽 **616 GB/s**

## 2. 各 kernel 耗时与带宽利用率

| 优先级 | kernel | total | 占比 | avg | 估算读写 | 带宽 | 利用率 |
|---|---|---|---|---|---|---|---|
| **P0** | norm_lerp6 | 2.87ms | 16.2% | 89.8µs | 0.16MB | 1.8 GB/s | **0.3%** ⚠ |
| P1 | gemv_rkv_stage1 | 3.63ms | 20.5% | 113.5µs | 45.3MB | 399 GB/s | 64.8% |
| P2 | gemv_f32io_add (FFN value) | 3.31ms | 18.7% | 103.6µs | 52.5MB | 507 GB/s | 82.3% |
| P2 | gemv_f32io_relu2 (FFN key) | 3.27ms | 18.5% | 102.2µs | 52.5MB | 513 GB/s | 83.4% |
| P3 | gemv_lowrank_chain4 | 1.55ms | 8.7% | 48.3µs | ~23.6MB | ~489 GB/s | ~79% |
| P3 | gemv_f32io_add_mul | 1.12ms | 6.3% | 35.1µs | ~13MB | ~370 GB/s | ~60% |
| P4 | gemv_f32io（输出投影） | 0.61ms | 3.4% | 610µs | 335.8MB | 550 GB/s | **89.3%** |
| P4 | fuse_ka_dplr_norm | 0.92ms | 5.2% | 28.9µs | 0.12MB | 4.2 GB/s | 0.7% |
| P5 | cmix_norm_lerp | 0.41ms | 2.3% | 12.7µs | 小 | — | — |
| P5 | norm_f32_f32_affine | 0.015ms | 0.1% | 14.6µs | — | — | — |

合计 ≈ 17.7ms。

## 3. 优先级分析

### P0 —— 最高优先：norm_lerp6（2.87ms，16.2%）

**✅ 已完成（实现 + 验证）**

- **证据**：估算读写仅 0.16MB，却耗时 89.8µs/层，带宽利用率仅 **0.3%**。它根本不是内存受限，而是**延迟/占用率受限**。
- **根因**：`norm_lerp6` 的 dispatch 是 `(1,1,1)`（见 `src/runtime.rs`），即**单个 workgroup** 承担 C=2560 个元素的 layer_norm 归约 + 6 次 lerp。只有 1–8 个 warp，GPU 大部分单元空闲。
- **实现**：改为多 workgroup 网格 `(ceil(C/BLOCK_SIZE), 1, 1)`（BLOCK_SIZE=256）。每个 workgroup 独立对全 C 做**冗余归约**（x 仅 10KB，命中 L2 开销可忽略），再并行计算各自 C 片段的 apply。保持单 dispatch 融合不变。
  - 改动文件：
    - `assets/shaders/src/norm_lerp6.comp`：main() 用 `gl_WorkGroupID` 分片段，Phase1 归约不变，Phase2 按片段 apply（每线程 1 元素）。
    - `src/runtime.rs`：新增 `NORM_LERP6_BLOCK=256` 常量；dispatch 从 `(1,1,1)` 改为 `(c.div_ceil(NORM_LERP6_BLOCK) as u32, 1, 1)`。
- **结果**：
  - norm_lerp6：**89.8µs → ~8–12µs/层**（约 7–11× 提升），单 token 2.87ms → ~0.27–0.40ms
  - 解码速度：**56.5 → 66.5 tok/s（+17.7%）**，SUM 17.7ms → ~15.0ms/token
  - 正确性：DIAG 单 token top5 `[47,45,96,46,59]` 与 CPU/seq 完全一致，max_abs_diff 0.09–0.15（fp16 正常范围），逐层确定性 0.000000

### P1 —— gemv_rkv_stage1（3.63ms，20.5%，64.8% 带宽）

- 占比最高的单 kernel，但带宽利用率仅 **65%**，有明确余量（r/k/v 三个 f16 C×C + 4 个 mid 投影已融合为 1 次 dispatch）。
- **已实施（ROWS=4 多行复用）**：
  - `assets/shaders/src/gemv_rkv_stage1.comp`：每 workgroup 处理 `ROWS=4` 行 r/k/v，每线程一次加载 x（xr/xk/xv）后在内层 r 循环复用于全部 4 行，消除逐行的 x 重读（x 冗余 L2 流量降 4×）。
  - `src/runtime.rs`：dispatch 网格由 `(c+vm+wm+am+gm)` 改为 `(c/GEMV_ROWS + vm+wm+am+gm)`，并加 `c % GEMV_ROWS == 0` 断言。
- **结果**：
  - 单层耗时 113.5µs → **~85–95µs**（改善 ~16–25%），诊断模式 top5 `[47,45,96,46,59]` 与 CPU 完全一致、逐层确定性 0.000000。
  - 整体 tok/s 约中性（64.6 vs 66.5，属温度/噪声波动）：该核本质是**权重带宽受限**（3 个 f16 C×C=39MB 无可避免，ROWS 仅消除了小的 x 冗余读取），故收益有限。
  - 现瓶颈前移：`gemv_f32io_add`（0.12–0.15ms, 57–72%）、`gemv_f32io_relu2`（0.10ms, 83%）、`gemv_rkv_stage1`（0.085–0.113ms）。FFN 两个 gemv 已近带宽峰值，后续收益空间小。

### P2 —— FFN 两个 gemv（add 3.31ms + relu2 3.27ms，合计 37%）

- 带宽利用率 **82–83%**，已接近 fp16 实际可用峰值（~85–90%），**提升空间有限**。
- **建议**：可尝试 split-K 或更优的加载模式，但预期收益小，**不要过度投入**。A 矩阵读取是 f16 下的固有成本，难以再压缩。

### P3 —— gemv_lowrank_chain4（1.55ms，~79%）/ gemv_f32io_add_mul（1.12ms，~60%）

- 中等余量，可顺带调优向量加载与占用率。

### P4 —— 输出投影 gemv（0.61ms，89.3%）与 fuse_ka_dplr_norm（0.92ms）

- 输出投影已 **89% 靠近内存峰值**，单次 dispatch，属"已压到位"，优先级低。
- fuse_ka_dplr_norm 已优化过 9×，虽利用率低但体量已小，不必再动。

## 4. 横向参考：dispatch 数量

每 token **258 次 dispatch**。除 P0 融合 norm_lerp6 可省 1 次/层外，还可考虑把相邻小 kernel（to_f16、seq_shift、elementwise）进一步合批，降低启动开销。

**P3 可行性调查（2026-08-05）：decode 路径已无法再融合，不建议继续。**

- decode 每层 8 个 dispatch 全是重型核（norm_lerp6 / gemv_rkv_stage1 / gemv_lowrank_chain4 / fuse_ka_dplr_norm / output gemv / cmix_norm_lerp / ffn relu2 / ffn add），`to_f16`/`seq_shift`/`elementwise` 等小核都在 **seq(prefill)** 路径，不在 decode。
- 相邻核之间全部是「整向量生产者→消费者」全局依赖（如 ffn relu2 产出完整 r2、ffn add 需消费完整 r2；output gemv 写完整 x、cmix 需读完整 x）。单次 dispatch 内无法用 shared memory 跨 workgroup 传整向量，**必须 grid 级同步，Vulkan 无原生支持** → 融合不可行。
- 开销量化（PROF_HOST/PROF_GPU，稳态每 token）：GPU SUM **~12.3ms**、CPU 记录 **~1.24ms**（barrier 0.54 + dispatch 0.37）、wall **~15.5ms**。gap ~2ms 为 GPU 屏障停顿 + submit/fence 等待。
- 结论：dispatch **数量**与**单次开销**均已近最优，剩余 ~3.2ms 是串行链条强制同步所致，继续压 dispatch 收益低、风险高（易破坏正确性）。**不建议实施。**

## 4.1 备选方向（若继续提升 decode）

- 结构性：异步双缓冲提交（CPU 记录 token N+1 与 GPU 执行 token N 重叠）。但顺序 decode 状态原地依赖，无法真正重叠计算，仅能隐藏 ~1ms 记录开销，复杂度高。
- 反向：回到 GEMM/占用率（prefill 方向），或接受当前 ~64–66 tok/s。

## 5. 建议实施顺序

1. **P0 norm_lerp6 并行化**（一次性 +18% 收益，风险低）→ 先做
2. **P1 gemv_rkv_stage1 带宽调优** → 已完成（ROWS=4），收益有限（权重带宽受限）
3. P2/P3 微调（预期收益小）
4. 输出投影与 fuse_ka 维持现状