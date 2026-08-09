# fp16 / int8 × Vulkan / CUDA decode 速度复测记录

日期：2026-08-09。硬件：RTX 2080 Ti（68 SM，subgroup_size=32）。模型：rwkv-g1h-3B（fp16）、rwkv-g1h-3B.int8.st（int8）。
前置：《chain4_subgroup重构与head合成量化删除记录.md》

## 背景

GPU 冷态（60°C）下，对 fp16 / int8 模型在 Vulkan / CUDA 两个后端重新跑 GPU decode self-loop 测速，
更新 README / README_CN 中的吞吐表。

## 方法

`cargo run --release -- memtest`，配 `SELFLOOP_ONLY=1`、`SELFLOOP_N=1000`（纯 GPU argmax self-loop，Batch=N，取 1000 tok 使结果更稳定）。
后端由 `BACKEND=vulkan|cuda` 显式指定；模型由 `MODEL_PATH` 指定（`.int8_idx` 存在 → int8，否则 fp16）。

## 结果（GPU 冷态 60°C，1000 tokens）

| 权重 | 后端 | tok/s |
|---|---|---|
| fp16 | Vulkan | 80.0 |
| fp16 | CUDA | 85.7 |
| int8 | Vulkan | 110.6 |
| int8 | CUDA | 110.8 |

对比上一轮（300 tok，同硬件）：

| 权重 | 后端 | 300 tok | 1000 tok（本轮） | 差异 |
|---|---|---|---|---|
| fp16 | Vulkan | 79.8 | 80.0 | ~0% |
| fp16 | CUDA | 85.6 | 85.7 | ~0% |
| int8 | Vulkan | 105.5 | 110.6 | +5% |
| int8 | CUDA | 110.2 | 110.8 | ~0% |

## 结论

- 1000 tok 读数明显更稳定（int8-Vulkan 由 300 tok 的 105.5 校正到 110.6，其余几乎一致）。
- 相对关系稳定：int8 较 fp16 约 +38%（Vulkan）/ +29%（CUDA）；CUDA 与 Vulkan 同精度基本持平
  （fp16 略优 +7%，int8 持平）。
- README.md / README_CN.md 吞吐表已更新为本轮（1000 tokens）数据。