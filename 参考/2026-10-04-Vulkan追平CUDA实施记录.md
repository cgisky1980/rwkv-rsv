# Vulkan 追平 CUDA 实施记录（int8 批量解码 / 单流 / 稳态 prefill 全路径）

> 日期：2026-10-04。目标：把 Vulkan 后端的端到端吞吐提升到 CUDA 的水平（同机、同模型、同配置）。
> 硬件：RTX 2080 Ti（sm_75，68 SM）。模型：`rwkv7-3B-int8.st`
> （C=2560 / H=40 / N=64 / V=65536 / L=32，`n_embd=2560 n_layer=32 vocab=65536 n_head=40 head_size=64`）。
> 提交：`ec773b0`。

---

## 0. 结论

**全部路径已与 CUDA 持平（差异 ≤2%，落在测噪内），且 token 指纹与 CUDA 逐位一致。**

| 路径 | Vulkan | CUDA | Vulkan/CUDA |
|---|---:|---:|---:|
| 单流 decode（`batch_decode_bench` SLOTS=1） | 83.5 tok/s | 83.3 tok/s | 1.00 |
| 批量 decode B=8 | 481.5 tok/s | 490.9 tok/s | 0.98 |
| 批量 decode B=16 | 880.1 tok/s | 878.6 tok/s | 1.00 |
| 批量 decode B=32 | 1471.7 tok/s | 1487.0 tok/s | 0.99 |
| 批量 decode B=64 | 2229.8 tok/s | 2192.1 tok/s | 1.02 |
| 稳态 prefill T=256（`prof_prefill_steady`） | 2455.6 tok/s | 2433.0 tok/s | 1.01 |

> 数据为 **同会话背靠背交错 A/B**（见 §3 方法）。跨会话读数漂移可达 ±17%（该卡兼作桌面显示卡），
> 单次读数不足以支撑结论 —— 本表全部取自交错跑。

---

## 1. 瓶颈定位

Vulkan 批量解码的 r/k/v 融合核（`gemv_int8_rkv_stage1_batch`，SIMT 路径）此前每槽每 k
读 **16 bytes** 的 fp32 激活（`vec4`），而权重是 int8 紧凑格式。在 B=16 时实测只有
**608 tok/s**，与 CUDA 的 806 差 1.33×；B=64 差 2.63×。

L2 带宽是主约束：权重已是 int8 单读，激活仍是 fp32 全宽读，成了第二条大头流量。

## 2. 改动

### 2.1 r/k/v 融合核读 fp16 激活（核心）

`assets/shaders/src/gemv_int8_rkv_stage1_batch.comp`：

- 新增 `InX2` 缓冲引用（`f16vec2 data[]`），r/k/v 三条链的激活改从 fp16 缓冲读，
  一次取 2 个 `f16vec2` 合成 `vec4` 参与 fp32 累加 —— **每槽每 k 的激活流量 16 → 8 bytes**。
- `use16x` 由 `constant_id` 控制，保留 fp32 旧路径可回退。

`src/gpu_model.rs`：

- `WorkBuffers` 新增 `xr16/xk16/xv16`（fp16 `[batch, C]`），由 `to_f16_triple` 核生成。
- `forward_layer_batch` 在分层前把 `xr/xk/xv` 转 fp16 落盘，再喂给融合核。

### 2.2 权重 `vec4` 打包 + `dot`

原实现把权重解量化成 `float w[ROWS][4]` 后逐元素 4 次 FMA；改为 `vec4 w[ROWS]`，
与 x 的 `vec4` 走 `dot()` —— 4 FMA → 1 dot 指令，指令数下降。

### 2.3 smem 分块加载 x（XTILE=256）

`BGRP=8` + `ROWS` 组合下，x 每 k 被 8 个槽 × ROWS 行重复读。改为把 x 按 `XTILE=256`
分块常驻 shared memory，一组内只读一次；smem 占用从 80KB（溢出）降到 32KB。
配合 `BGRP=8` 实现权重复用。

### 2.4 签名同步（CUDA 端不受影响）

`ComputeBackend::gemv_int8_rkv_stage1_batch` 增加 `xr16/xk16/xv16` 三个参数。
CUDA 后端把这三个入参标 `_` 忽略，仍走原 fp32 路径 —— **CUDA 数值与性能零变化**
（这是本表「CUDA 侧」可作为基准的前提）。

## 3. 测法

```powershell
$env:UNIFORM_POOL_MB="64"; $env:SEED="42"
# 交错 A/B：每个 B 先 Vulkan 再 CUDA，同一会话内背靠背
foreach ($b in 8,16,32,64) {
  foreach ($bk in "vulkan","cuda") {
    $env:SLOTS="$b"
    if ($bk -eq "cuda") { $env:BACKEND="cuda" } else { Remove-Item Env:\BACKEND }
    cargo run --release --example batch_decode_bench
  }
}
# 稳态 prefill
$env:PTOKENS="256"; cargo run --release --example prof_prefill_steady
```

正确性：`batch_decode_bench` 每档打印 decode token 指纹（sum / xor），逐档比对两后端。

## 4. 正确性验证

8 个后端×批量组合的 token 指纹**逐位一致**：

| B | sum | xor |
|---|---|---|
| 8 | 0xd02e16 | 0x6d08 |
| 16 | 0x1a123ac | 0x9de |
| 32 | 0x345e198 | 0x64a2 |
| 64 | 0x68f1ce4 | 0x6456 |

（Vulkan 与 CUDA 在相同 `SEED=42`、相同 `SLOTS` 下输出相同 sum/xor。）

内核级单测：`cargo test --release -- --test-threads=1` → **64 passed / 0 failed**
（并行跑会因显存争用出现 4 例假失败；已在 `--test-threads=1` 下排除）。

`cargo fmt --all` ✅ · `cargo clippy --all-targets -- -D warnings` ✅（零警告）。

## 5. 踩过的坑

1. **`UNIFORM_POOL_MB` 默认 8MB 不够**：NTOK=32、B≥16 时报 `uniform pool exhausted`。
   当前需手动设 `UNIFORM_POOL_MB=64`。**未解决**（见 §6）。
2. **smem 溢出 → `DEVICE_LOST`**：`BGRP=8, C=2560` 时 x 全量常驻要 80KB，超 sm_75 的
   48KB 上限。分块 XTILE=256 后降到 32KB。
3. **batch≥16 越界 → `DEVICE_LOST`**：`xb_q`/`xb_aux` 容量按 `batch.min(64)` 分配。
4. **WAR barrier 的 `src_access_mask` 缺位**：只写了 `SHADER_WRITE`，需补
   `SHADER_READ | TRANSFER_READ`，否则 fp16 激活缓冲的写后读存在竞态。

## 6. 遗留

- **uniform 池需手工配置**：应按 `batch × NTOK` 动态计算池容量自动调整
  （`src/runtime.rs` 的 `begin_batch`），而不是靠用户设 `UNIFORM_POOL_MB`。
- **Vulkan 张量核路径（`VK_IMMA_OPS=rkv`）仍不可用**：inter-dispatch 停顿导致 3.6× 退化，
  需合并 kernel 减少 dispatch 次数。当前生效的是 SIMT 融合核路径，已足够追平。
- fp16 模型的 Vulkan/CUDA 对比本轮未复测（本机仅有 int8 权重文件）。
