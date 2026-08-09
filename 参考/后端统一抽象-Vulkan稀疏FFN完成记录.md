# Vulkan 稀疏 FFN 完成记录（decode / self-loop 加速约 2.2~2.7×）

> 状态：完成。本轮在 Vulkan 后端引入稀疏 FFN value 投影 kernel（`ffn_value_sparse_add`），
> 对标 CUDA 的稀疏实现，把 decode 的 FFN value 投影从「全量读 50MiB 权重」降到「只读
> ~4% 非零列（0.8MiB）」，同时解除对稠密 value gemv 的带宽占用，使其余 kernel
> （key/relu2、rkv_stage1）带宽利用率从 36~47% 提升到 85~102%。
> 前置：S1~S10 已完成（含 CUDA int8 稀疏 FFN S9、prefill 稳态 S10）。

---

## 1. 目标

继续优化 Vulkan self-loop（decode 自回归）吞吐。前一轮修好 Vulkan self-loop 只生成
1 token 的 bug 后，真实吞吐约 **28~31 tok/s**。剖析（`PROF_GPU=1`）显示 FFN 两个
稠密 GEMV（`gemv_f32io_relu2` key 投影 + `gemv_f32io_add` value 投影）占 GPU 时间
大头：value gemv 每层 avg 0.22ms、读 50MiB、带宽利用率仅 38.8%。

关键机会：value 投影的输入 `r2` 是 relu² 输出，**约 96% 稀疏**，但稠密 kernel 仍
全量读 50MiB 权重矩阵。CUDA 侧已有 `ffn_value_sparse_add`（S9）只读非零列，Vulkan
此前回退稠密。把该稀疏 kernel 移植到 Vulkan 是最大杠杆。

---

## 2. 优化内容

### 2.1 Vulkan 稀疏 FFN value kernel（`assets/shaders/src/ffn_value_sparse_add.comp`）

- 每个 workgroup 处理一个 `(f_block, c_block)`：`f_block` 取 TILE=128 个 f（r2 一段），
  `c_block` 取 C_TILE=256 个 c（x 一段）。dispatch `(FH/128, C/256, 1)`。
- 流程：读 r2 片到 shared → 用 shared `atomicAdd` 统计非零并收集 `nnz_ids` →
  只遍历非零 f，读 `value_tiled` 对应列的 fp16 权重累加 → 跨 f_block 用 buffer 级
  fp32 原子（`VK_EXT_shader_atomic_float`）累加到 x。
- 依赖 `VK_EXT_shader_atomic_float` 的 `shader_buffer_float32_atomic_add`；
  设备不支持时 `supports_sparse_ffn()` 返回 false，自动回退稠密 gemv。

### 2.2 后端接线（`backend.rs` / `runtime.rs` / `gpu_model.rs`）

- `ComputeBackend::ffn_value_sparse_add` 默认回退稠密；Vulkan `Runtime` 覆写为
  录制稀疏 kernel，`supports_sparse_ffn()` 返回 `app.properties.atomic_float`。
- `gpu_model.rs` 解码路径：`supports_sparse_ffn() && !FFN_SPARSE_OFF` 时走稀疏
  （复用 S9 已加载的 `ffn_value_tiled`），否则按 int8 → any4 → fp16 回退稠密。

### 2.3 关键 bug 修复：参数顺序错位（`runtime.rs`）

移植后 `ERROR_DEVICE_LOST` / `STATUS_ACCESS_VIOLATION`。根因：`Runtime::ffn_value_sparse_add`
传入的 `params = [value_tiled, r2, x]` 与 shader `Params` 成员声明顺序
`BufR2 buf_r2; BufW buf_w; BufX buf_x`（即 r2, value_tiled, x）**不一致**。结果
r2 的 2048 字节小缓冲被当作权重矩阵以 `uint[]` 索引读到 65535，GPU 越界 → 设备丢失。
修复：参数顺序改为 `[r2, value_tiled, x]`。CUDA 参考顺序即 `(r2, value_tiled, x)`。

---

## 3. 测速结果

模型 `rwkv-g1h-3B`（3B, 纯 fp16），GPU **RTX 2080 Ti**，Vulkan 后端。

| 指标 | 稀疏关（稠密） | 稀疏开 | 提升 |
|------|------|------|------|
| argmax self-loop（8 tok） | 31.3 tok/s | **69.0 tok/s** | 2.2× |
| sample self-loop（128 tok）| 31.6 tok/s | **85.3 tok/s** | 2.7× |
| value gemv（ffn_value_sparse_add）| 0.22ms/层, 50MiB | 0.023ms/层, 0.8MiB | ~10× |
| self-loop GPU 总时间（8 tok）| 214ms / 2048 kernel | 79ms / 2048 kernel | 2.7× |

> 附：key/relu2（`gemv_f32io_relu2`）带宽利用率 36.5%→85.6%，rkv_stage1 44.8%→
> 102.6%，头投影 41.3%→96%。原因是移除稠密 value gemv 的全量读后，各 kernel
> 不再被带宽争抢，趋于各自峰值。

---

## 4. 正确性验证

- 新增回归测试 `tests/ffn_sparse.rs`：直接调用 `Runtime::ffn_value_sparse_add`，
  与 CPU 参考对比，`max_diff=5.2e-3`（<1e-2 阈值）。
- `cargo test --release`：**59 + 1（新稀疏测试）通过 / 0 失败**。
- `cargo fmt` / `cargo clippy --release --all-targets`：**零警告**。
- 临时诊断测试 `tests/ffn_sparse_probe.rs` 已删除（遵循「测试完清理」规则）。

---

## 5. 结论与下一步

Vulkan decode / self-loop 吞吐从 ~31 tok/s 提升到 **69~85 tok/s**（约 2.2~2.7×），
主要来自稀疏 FFN value 投影。剩余 decode 带宽大头是仍需全量读的 `gemv_f32io_relu2`
（key 投影，50MiB，已接近峰值带宽）与 `gemv_rkv_stage1`（43MiB，已接近峰值带宽）。

下一步可选方向：

1. **head 权重量化**：头投影 `gemv_f32io` 读 320MiB/层（7 次共 3.9~9.2ms），是
   decode 单次最大带宽开销；量化到 any4/int8 可显著降带宽。
2. **kernel 融合**：`gemv_lowrank_chain4`（0.8% BW）与 `fuse_ka_dplr_norm`（1.1% BW）
   虽小但 launch 密集，可再融合减少 260 kernel/token 的 dispatch 与 barrier 开销。
3. **key 投影稀疏化不可行**（输入稠密），但可评估 key 量化。