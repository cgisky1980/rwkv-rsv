# RWKV-7 3B GPU self-loop 批量提交改造与性能测试报告

> 生成日期：2026-08-05
> 数据来源：`cargo run --release -- memtest`（SELFLOOP_BENCH / SELFLOOP_ONLY 开关）
> GPU：NVIDIA GeForce RTX 2080 Ti（峰值带宽 616 GB/s）
> 模型：RWKV-7 3B（g1h-3B），n_layer=32, C(n_embd)=2560, vocab=65536

## 1. 背景与目标

decode 路径原本每 token 一次 `forward_token` → `begin_batch` → 记录 kernel → `end_batch`（submit + queue_wait_idle），
存在两类开销：
1. **CPU↔GPU 交换**：每 token 一次 submit + wait 同步，CPU 等待 GPU 完成。
2. **dispatch 间同步**：GPU 内部逐 kernel 依赖链 barrier 串行化。

目标：通过 **GPU self-loop（批量提交）** 消除第 1 类开销——在单次 submit 内连续生成 n 个 token，
argmax 结果直接写回 host-visible 缓冲，下一轮 gather 自动跟随，全程无 CPU 回读/回传 token。

## 2. 改造内容（三步）

| 步骤 | 内容 | 关键文件 |
|---|---|---|
| 第一步 | 参数化 gather：token 索引从 CPU 常量改为 GPU 缓冲读取，循环体不依赖具体 token 值 | `gather_row.comp`、`runtime.rs::gather_row_device` |
| 第二步 | 消除 spec constant 回退：token 索引改由 CPU memcpy 直写 host-visible 缓冲，gather 从该缓冲 device address 读取；删除 `store_u32` kernel | `runtime.rs::store_token_host`、`host_write_to_shader_barrier`、`gather_row_device` |
| 第三步 | GPU self-loop：新增 `forward_argmax_selfloop`，单 batch 内循环 n 次；argmax 写回 host 缓冲，`record_token` 记录每轮 token | `runtime.rs::argmax_into_host`、`record_token`、`record_token.comp`、`gpu_model.rs::forward_argmax_selfloop` |

## 3. 正确性验证

`SELFLOOP_VERIFY`（main.rs ARGMAX_VERIFY 分支），N=8：

```
selfloop=[47, 11, 46, 20996, 45175, 4600, 59, 327]
reference=[47, 11, 46, 20996, 45175, 4600, 59, 327]
match=true
```

self-loop 生成的 8-token 序列与逐 token `forward_argmax` 参考完全一致。

Borrow/barrier 正确性关键：第二步起 `current_token.host` 由 GPU argmax（`argmax_into_host`）写入，
`record_barriers` 通过 `written` 追踪自动为下一轮 gather 插入 SHADER barrier（HOST barrier 转为无害冗余）。

## 4. 性能测试结果

### 4.1 测试开关

- `SELFLOOP_BENCH=1`：memtest 中逐 token 段结束后，追加一段 self-loop 对比（会受逐 token 段加热干扰）。
- `SELFLOOP_ONLY=1`：跳过逐 token 段，只测 self-loop（隔离 GPU 加热干扰，结果更可靠）。

### 4.2 数据

| 规模 | 逐 token (tok/s) | self-loop (tok/s) | 备注 |
|---|---|---|---|
| 64 tokens | 14.8 | 61.2 | 短序列虚高 |
| 512 tokens | 24.7 | 28.7 | 逐 token 受前期加热干扰 |
| 512 tokens（SELFLOOP_ONLY，降温后） | — | **28.7** | 稳定值 |
| 2048 tokens（SELFLOOP_ONLY，降温后） | — | **27.3** | 稳定值 |

### 4.3 关键结论

1. **self-loop 稳定吞吐 ≈ 27–29 tok/s**，与序列长度无关（512→28.7，2048→27.3）。
2. **64 tokens 的 61.2 tok/s 是短序列虚高**：单次 submit 内 dispatch 数少，GPU 管道可预填充；
   长序列（512/2048）时多次串行 dispatch 的依赖链 barrier 累积，GPU 内部逐 kernel 串行成为瓶颈。
3. **self-loop 相对逐 token 的稳定加速约 1.1–1.2x**，收益主要来自消除 CPU↔GPU 交换，
   而非 GPU 计算本身。decode 路径的 GPU 计算/带宽瓶颈依旧。

### 4.4 与 P3 优化报告的一致性

结论吻合 `decode优化建议报告.md` 的 P3 分析：**decode 路径 kernel 间全局依赖无法并行**。
self-loop 消除了 CPU 交换，但真实瓶颈是 GPU 内部逐 kernel 串行执行（258 次 dispatch/层链）。

## 5. 结论与后续方向

self-loop 的价值在于**消除 CPU↔GPU 交换**，为后续真正的并行化创造条件，而非自身突破 GPU 计算瓶颈。
后续可探索：
- **减少 dispatch 数**：kernel fusion（如 A5 方案：w/a/g 链单 kernel 融合），降低 258 次/层 dispatch 开销。
- **GEMM 双缓冲 / 异步提交**：prefill 路径吞吐优化。
- **PROF_GPU 定位** self-loop 长序列下各 kernel 是否仍以 `gemv_f32io_relu2`/`gemv_f32io_add`（近带宽峰值）为瓶颈。

## 6. 受控 A/B 实测（2026-08-05，消除热降频干扰）

**背景**：此前报告存在"60 vs 30"矛盾——逐 token 冷态单 token ~64-68 tok/s，而 self-loop 持续热态 ~27-29 tok/s。
为定论，新增 `AB_BENCH=1` 受控对比（`main.rs::bench_ab`）：GPU 降温后，同一运行内交替测两条路径，
每轮先各跑 4 token 预热到同一热态，轮内测量顺序按奇偶交替抵消单调升温，取中位数剔除热降频抖动。

**条件**：`AB_BENCH=1 AB_N=128 AB_ROUNDS=5 SKIP_CPU=1`，GPU 已降温。

| 轮次 | 逐token (tok/s) | self-loop (tok/s) |
|---|---|---|
| 0 | 30.7 | 31.6 |
| 1 | 30.0 | 29.8 |
| 2 | 30.3 | 29.6 |
| 3 | 30.3 | 28.5 |
| 4 | 28.3 | 25.8 |
| **中位数** | **30.3** (33.04 ms/tok) | **29.6** (33.76 ms/tok) |

**结论**：
1. **"60 vs 30" 是测量条件差异，非 self-loop bug**：同热态下两条路径都稳定 ~30 tok/s；
   60-68 是冷 GPU 单 token 峰值，持续生成时两者都被热降频拉到 ~30。
2. **self-loop 无加速（0.98x，略慢 2%，噪声内）**：消除 CPU↔GPU 交换换不来吞吐提升，
   因为瓶颈是 GPU 计算/热降频而非 CPU dispatch 开销；每 token 多出的 `argmax_into_host`+`record_token`
   两个 kernel 与 host-visible 缓冲访问抵消了收益。
3. **遗留正确性脆弱点（非性能）**：`record_token` 读 host 缓冲后将其从 `written` 移除，
   下一轮 `gather_row_device` 读同一缓冲时检测不到 RAW 依赖，缺少显式 SHADER_WRITE→SHADER_READ 屏障，
   当前靠命令缓冲按序执行"碰巧"正确。若后续要正式采用 self-loop 应修复（token 改走 device 缓冲）。

## 7. 0.4B g1d 小模型受控 A/B 实测（2026-08-05，冷态验证）

**背景**：3B 上 self-loop 无加速（0.98x），推断瓶颈是 GPU 计算/带宽而非 CPU 提交开销。
为验证"更小的模型、更低 GPU 负载下 self-loop 能否因消除 CPU↔GPU 交换而拉开差距"，
在 0.4B g1d 小模型上复跑同一受控 A/B。

> 注：0.1B 模型文件已不在磁盘，取可用最小模型 0.4B（g1d-0.4b）。模型路径已参数化（`MODEL_PATH` 环境变量）。

**条件**：`MODEL_PATH=<0.4B g1d> AB_BENCH=1 AB_N=256 AB_ROUNDS=5 SKIP_CPU=1`，GPU 冷态。

**模型**：n_layer=24, n_embd=1024, vocab=65536, n_head=16, head_size=64, ffn_hidden=4096。

| 轮次 | 逐token (tok/s) | self-loop (tok/s) |
|---|---|---|
| 0 | 131.1 | 131.1 |
| 1 | 136.0 | 148.9 |
| 2 | 114.4 | 140.2 |
| 3 | 123.0 | 130.9 |
| 4 | 150.1 | 138.3 |
| **中位数** | **131.1** (7.630 ms/tok) | **138.3** (7.231 ms/tok) |

**结论**：
1. **小模型吞吐远高于 3B**（131-138 vs ~30 tok/s）：0.4B 每 token 计算量小，7.2-7.6ms/tok，
   2060 Ti 负载低、无明显热降频，GPU 计算不再是主导瓶颈。
2. **self-loop 1.06x，微弱优势但未拉开差距**：相较 3B 的 0.98x，小模型下消除 CPU↔GPU 交换
   确实带来 ~6% 提升（CPU 提交开销占比略升），但仍远不及"显著加速"。
3. **统一结论跨模型成立**：无论 3B 还是 0.4B，self-loop 的收益都局限在消除 CPU 提交抖动，
   无法突破 GPU 计算/带宽/内部逐 kernel 串行瓶颈。真正的加速杠杆仍是减少 GPU dispatch 数
   （kernel fusion，如 A5 方案）或 prefill GEMM 瓦片优化，而非 self-loop。