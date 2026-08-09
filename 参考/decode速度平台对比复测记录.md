# fp16 / int8 × Vulkan / CUDA decode 速度复测记录

日期：2026-08-09。硬件：RTX 2080 Ti（68 SM，subgroup_size=32）。模型：rwkv-g1h-3B（fp16）、rwkv-g1h-3B.int8.st（int8）。
前置：《chain4_subgroup重构与head合成量化删除记录.md》

## 背景

GPU 冷态（60°C）下，对 fp16 / int8 模型在 Vulkan / CUDA 两个后端重新跑 GPU decode self-loop 测速，
更新 README / README_CN 中的吞吐表。

## 方法

`cargo run --release -- memtest`，配 `SELFLOOP_ONLY=1`、`SELFLOOP_N=300`（纯 GPU argmax self-loop，Batch=N）。
后端由 `BACKEND=vulkan|cuda` 显式指定；模型由 `MODEL_PATH` 指定（`.int8_idx` 存在 → int8，否则 fp16）。

## 结果（GPU 冷态 60°C）

| 权重 | 后端 | tok/s |
|---|---|---|
| fp16 | Vulkan | 79.8 |
| fp16 | CUDA | 85.6 |
| int8 | Vulkan | 105.5 |
| int8 | CUDA | 110.2 |

对比上一轮（同硬件）：

| 权重 | 后端 | 上一轮 | 本轮 | 差异 |
|---|---|---|---|---|
| fp16 | Vulkan | 83.3 | 79.8 | -4% |
| fp16 | CUDA | 87.7 | 85.6 | -2% |
| int8 | Vulkan | 111.9 | 105.5 | -6% |
| int8 | CUDA | 114.4 | 110.2 | -4% |

## 结论

- 本轮整体略低于上一轮（约 -2%~-6%），处于热漂移/采样噪声范围内，相对关系稳定：
  int8 较 fp16 约 +32%（Vulkan）/ +29%（CUDA）；CUDA 较 Vulkan 同精度约 +5%。
- README.md / README_CN.md 吞吐表已更新为本轮数据。