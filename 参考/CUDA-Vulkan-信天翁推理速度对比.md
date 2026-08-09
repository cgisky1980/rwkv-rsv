# CUDA / Vulkan / 信天翁（Albatross）推理速度全面对比

> 日期：2026-08-08。目的：同硬件归一化对比三者的 **decode** 与 **prefill** 吞吐，
> 明确差距与优势。同卡 **RTX 2080 Ti**（fp16 理论 ~58 TFLOPS）。

---

## 0. 测试条件

| 项 | 本实现 | 信天翁（本地参考实现） |
|---|---|---|
| 模型 | `rwkv-g1h-3B`（3B，n_embd=2560，n_layer=32，vocab=65536） | `rwkv7-g1h_preview4533-2.9b`（2.9B，结构同） |
| 后端 | `BACKEND=cuda` / `BACKEND=vulkan` | `albatross_ref.rwkv7.RWKV_x070`（CUDA fp16） |
| 测法 | 稳态（预热后测，NTOKENS=256） | 同卡跑 `albatross_bench.py` / `albatross_seq_bench.py` |
| 量化 | fp16 / int8 / any4 | fp16 |

> 注：信天翁官方 README 的 15000+/17000+ tok/s 为 **RTX 5090 + 大批量**（B1024T1 /
> B1T1024）测得；本表本地实测为该参考实现在 RTX 2080 Ti 上的单流数值，与我们的实现同卡可比。

---

## 1. 汇总表（tok/s，越高越好）

### CUDA（本实现）

| 路径 | fp16 | int8 | any4 |
|------|------|------|------|
| prefill (infer_seq) | **3529** | 2891 | 2985 |
| decode (infer_tokens) | 89.9 | **115.1** | 113.3 |
| argmax_selfloop | 93.9 | **117.4** | 115.6 |
| sample_selfloop | 79.6 | **99.3** | 94.6 |

### Vulkan（本实现）

| 路径 | fp16 | int8 | any4 |
|------|------|------|------|
| prefill (infer_seq) | **2402** | 2164 | 2022 |
| decode (infer_tokens) | 67.0 | 96.0 | **101.6** |
| argmax_selfloop | ⚠️ 异常 | ⚠️ 异常 | ⚠️ 异常 |
| sample_selfloop | ⚠️ 异常 | ⚠️ 异常 | ⚠️ 异常 |

### 信天翁（本地参考，RTX 2080 Ti）

| 路径 | tok/s | 备注 |
|------|-------|------|
| decode (forward_one 逐 token) | **25.8 ~ 26.6** | 逐 token 串行，Python 逐 token 下载 |
| prefill (forward_seq 整段) | **3885.6** | 0.257 ms/token |

---

## 2. 逐项解读

### 2.1 decode：本实现大幅领先（~4×）

同卡上，信天翁参考实现逐 token decode 仅 **26.6 tok/s**；本实现 CUDA int8 **115 tok/s**、
CUDA fp16 **90 tok/s**、Vulkan int8 **96 tok/s**，均大幅反超。

原因：信天翁参考实现每 token 一次 Python 层调用 + CUDA 逐 kernel 启动（Python 解释器与
kernel launch 开销占主导）；本实现用 **CUDA graph 捕获/重放 decode 路径**（消除 258 次/层
的 cuLaunchKernel 启动开销）+ 融合 kernel + int8 稀疏 FFN，单流 decode 已远超参考实现。

信天翁官方 15000+ tok/s 是 **批量 B1024T1 在 RTX 5090** 上取得，属多用户并发吞吐，
非单流，二者不可直接比。

### 2.2 prefill：信天翁参考略快，本实现已接近

同卡上信天翁参考 `forward_seq` **3885 tok/s**（依赖其整段序列扫描 CUDA kernel）；
本实现 CUDA fp16 **3529 tok/s**（约 91%）、int8 2891、any4 2985。

差距主因：prefill 已转入 cuBLAS GEMM 与 `dplr_seq` 序列扫描，本实现 CUDA fp16 已基本
对齐；int8/any4 因反量化成 fp16 平铺权重再 GEMM，有 ~10% 反量化开销。Vulkan prefill
受限于自研 GEMM/序列扫描 shader，约低 30%。

### 2.3 Vulkan self-loop 异常（需要修复）

Vulkan 的 `begin_graph_capture`/`end_graph_capture`/`graph_replay` 均为 **no-op**
（`backend.rs` 默认实现），而 CUDA 用 CUDA graph 捕获+重放。结果 Vulkan self-loop 只把
**单步 kernel 记录进 batch 一次**，随后 `graph_replay()` 不执行任何东西，仅生成 **1 个
token 而非 n 个**，故测出 16000~21000 tok/s 的物理不可能数值、且序列为垃圾（与历史
"Vulkan all-0 token" 一致）。

**结论**：Vulkan 的 self-loop 路径是坏的，数值不可信。有效 decode 指标应看
`infer_tokens`（逐 token host 侧循环，真实）。

### 2.4 ✅ Vulkan self-loop 已修复（2026-08-08）

**根因**：`ComputeBackend::begin_graph_capture/end_graph_capture/graph_replay` 在
`backend.rs` 中为 no-op 默认实现，仅 CUDA 覆盖。原 self-loop 无条件走「捕获单步→
`graph_replay` n 次」，Vulkan 下 `graph_replay` 不执行任何指令，故只生成 1 个 token，
测出 16000~21000 tok/s 假吞吐且序列为垃圾。

**修法**：给 `ComputeBackend` 增加能力探测 `supports_graph_capture()`（默认 `false`，
`CudaBackend` 覆盖为 `true`）。`gpu_model.rs` 的 `forward_argmax_selfloop_with_state`
与 `forward_sample_selfloop_with_state` 按此分支：
- **CUDA**（`supports_graph_capture=true`）：保留原 CUDA graph 捕获+重放路径。
- **Vulkan**（`false`）：把 n 轮完整前向逐 token **记录进同一批次**（`selfloop_step` /
  `sample_selfloop_step` 循环 n 次，末尾一次 `end_batch` 提交）。`selfloop_step` 内部
  已重置 `v_first_set`，每轮重新快照 v_first，与 CUDA graph 捕获时的行为一致。

**验证**（`BACKEND=vulkan`，3B 模型）：
1. **正确性**：`ARGMAX_VERIFY=1` 下 self-loop 8 token
   `[47,11,46,20996,45175,4600,59,327]` 与逐 token 参考序列完全一致，`match=true`。
2. **吞吐回落到真实值**：`SELFLOOP_ONLY=1` 512 token 实测 **28.3 tok/s**（不再是
   16000~21000 假值）。该值偏低主因是单次 batch 记录 ~13 万 kernel 的 CPU 记录开销，
   属性能优化范畴，非正确性问题。
3. fmt/clippy 零警告，`cargo test --release` 59 通过 0 失败。

---

## 3. 结论

1. **decode**：本实现单流已是对标参考的大幅领先（CUDA int8 115 vs 26.6 tok/s，~4×）。
   与信天翁官方 5090 批量值不可同尺度比较。
2. **prefill**：本实现 CUDA fp16 已接近信天翁参考（3529 vs 3885，~91%）；剩余差距在
   int8/any4 反量化与 Vulkan shader GEMM。
3. **Vulkan self-loop 需修复**：为 Vulkan 补 CUDA-graph 等价物（持久 command buffer /
   二次提交循环），否则 Vulkan 自回归生成不可用。

## 4. 下一步建议

1. **修 Vulkan self-loop**：`graph_replay` 需在 Vulkan 上真正重放 n 步（把单步 kernel
   序列记录的 command buffer 循环提交 n 次，或录制后循环 vkCmdDispatch）。
2. **prefill 反量化消除**：int8/any4 prefill 直接走量化 GEMM（含反量化），省去
   "反量化→fp16→GEMM" 的中间张量与额外 kernel，预期 int8 prefill 追平 fp16。
3. **prefill M 维度**：T≥1024 时 GEMM 利用率与 `dplr_seq` T 循环是瓶颈，可再剖析。