# head 权重量化速度复测记录（修复后基线）

日期：2026-08-09
前置：decode 两跳 dispatch 错位修复 + chain4 融合回退后的基线（见《decode两跳dispatch错位修复与chain4融合回退记录.md》）

## 目的

在修复后的基线上，单独量化 head 权重（fp16 → int8 per-group=128 min-max）对 decode 速度的实际收益，区分 **kernel 级** 与 **端到端** 两个层面。

## 测试配置

- 模型：RWKV-7 3B（n_layer=32, n_embd=2560, vocab=65536）
- 后端：Vulkan（`BACKEND=vulkan`），`PROF_GPU=1` 逐 kernel 时间戳采集
- 三种配置：
  - **A**：fp16 模型，`HEAD_QUANT=off`（fp16 head，走 `gemv_f32io.spv`）
  - **B**：fp16 模型，`HEAD_QUANT=on`（加载时合成 int8 head，走 `gemv_int8.spv`）
  - **C**：预量化 int8 模型（全层 int8，head 亦为预量化 int8）
- 两轮正反序（R1: A→B→C，R2: C→B→A）抵消 GPU 热漂移；每轮 decode 128 tok + argmax_selfloop 8 tok + sample_selfloop 128 tok
- 日志：r1a/r1b/r1c/r2a/r2b/r2c_*.log；kernel 统计脚本：analyze_head.ps1

## kernel 级结果（head GEMV，每 token 1 次，每轮 n=132 采样）

| 配置 | kernel | min | p25 | med | mean | max |
|---|---|---|---|---|---|---|
| A fp16 head | gemv_f32io | 0.565ms | 0.572 | 0.93~0.95 | 1.29~1.45 | 7.16 |
| B 合成 int8 head | gemv_int8 | 0.297ms | 0.300 | 0.30~0.34 | 0.70~0.90 | 3.36 |
| C int8 模型 | gemv_int8 | 0.298ms | 0.300 | 0.30~0.31 | 0.72~0.90 | 2.94 |

- 冷态下限（min/p25 最干净）：**fp16 head 0.57ms → int8 head 0.30ms，约 1.9×**
- 两者均在显存带宽 bound 下跑满：fp16 320.3MiB / 0.57ms ≈ 590 GB/s；int8 实际字节约 170MiB（idx 160MiB + sz 10MiB）/ 0.30ms ≈ 595 GB/s，均 ≈96% of 616 GB/s 峰值。int8 字节减半 → 时间减半，纯属带宽收益，无算力红利。
- mean/p75/max 受热节流与队列噪声污染（如 A 的 mean 1.29~1.45ms），比较时应以 min/p25 为准。
- **已知显示 artifact**：PROF_GPU 的 est_bytes 对 int8 仍按 m×k×2 估算（显示 320.3MiB），导致 int8 head 带宽显示 181.7% 超峰值；仅统计口径问题，不影响时间戳。

## 端到端结果（PROF_GPU=1 开销内含）

| 配置 | decode 128 tok | sample_selfloop 128 | argmax_selfloop 8 |
|---|---|---|---|
| A fp16 head off | 38.1 / 35.5 | 38.2 / 11.5* | 33.7 / 20.7* |
| B fp16 + int8 head | 37.2 / 40.8 | 38.8 / 34.2 | 34.4 / 35.0 |
| C int8 模型 | 48.1 / 53.6 | 48.2 / 54.0 | 43.7 / 44.9 |

（* r2a 为末位热态运行，sample_selfloop 11.5 tok/s、argmax 20.7 tok/s 明显热节流，剔除）

## 结论

1. **kernel 级**：head int8 量化使 head GEMV 从 0.57ms 降至 0.30ms（1.9×），B/C 两种 int8 head 路径耗时一致。
2. **端到端（fp16 模型只量化 head）无可见收益**：A vs B 两轮 decode 均值 36.8 vs 39.0 tok/s，差异在热噪声（±5%）内。原因：38 tok/s 时每 token 约 26.3ms，head 节省的 0.27ms 仅约 1%。
3. **全模型 int8（C）收益显著**：48~54 vs 35~38 tok/s（+34~42%），来自全部 32 层投影量化，head 只是其中一小部分。
4. 绝对吞吐低于无 PROF 基线记录（fp16 60.4 / int8 81.2 tok/s）：本轮 PROF_GPU=1 逐 kernel 时间戳有额外开销，且环境更热；同轮内相对比较有效，跨轮绝对值不可比。

## 建议

- fp16 模型单独量化 head 不划算（e2e ≈1%，还需承担加载时量化开销与精度损失）；head 量化应作为全 int8 模型的一部分使用。
- 后续 decode 优化应聚焦每层都执行的大 kernel（rkv stage1 / ffn relu2 / att output 等），head 每 token 仅 1 次、占比约 2%。
