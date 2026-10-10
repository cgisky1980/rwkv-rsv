# `imma_partial` 超配 64 倍：self-loop 阶段 5.3 GB 的归因与收回（2026-10-11）

## 结论摘要

1. 上一条记录（`2026-10-10-Vulkan采样mask位图化与显存记账-实施记录.md`）留下一个没交代的数：
   `log_vram` 从「prefill 后」到「self-loop 图捕获后」涨了 **+5471 MiB（B=256）**，约**每槽 21.4 MiB**，
   与整份 RNN 状态同量级。本次把它归因到底：**不是状态，是 `imma_partial` 分配大了 64 倍**。
2. 修正后 B=256：登记张量 **17 807 → 12 747 MiB**，外部 `nvidia-smi` 峰值 **20 585 → 15 528 MiB**
   （−5057 MiB）。**逐位等价**（B=192/256 交替两轮解码指纹全同），**吞吐持平**（±1%）。
3. 附带否证了一条旧叙事：在**旧尺寸**下、本机当前环境里，§6.2③ 的「B≥192 采样核 57 倍变慢」
   **根本复现不出来**（1465~1505 tok/s，随批宽单调）。所以那条断崖要么是当时共存显存压力的产物，
   要么是没有暖机的一次性开销；无论如何，容量这一解释项现在已经不成立了——B=256 留出 ~7 GB 空闲。
4. **这不是 Vulkan 专属问题**：`WorkBuffers::new` 与后端无关，CUDA 同样在分配这块它那侧根本用不到的
   5.1 GB（CUDA 的 split-K 部分和另有其缓冲）⇒ 历史 CUDA 大批量读数都含着这笔开销。

## 一、症状与归因方法（十行代码，值得留在工具箱里）

现象：三个阶段的登记张量里，只有 self-loop 那一档对不上任何已知结构——
`SelfloopBufs`（temp/counter/mask/sampler/token_seq/seq_cnt）在 B=64 只有 32.5 MiB，
`WorkBuffers` 每槽 ~300 KiB，而实测涨的是 **1352 MiB**。

做法：在 `Runtime::create_tensor` / `create_tensor_f16` / `create_tensor_u32` 三个漏斗里加
env 门控（`TRACE_ALLOC=1`）的日志，只打 ≥4 MiB 的分配，连 `len`、元素数换算的 MiB 和
`std::backtrace` 里第一条本 crate 的栈帧一起输出（release 无行号，但函数名足够）。
**分配类问题不要靠推断，登记口只有一个，插这里一次就够。**

B=64 的输出决定性地指出了一条：

```
[ALLOC] f32 elems=335544320 (1280 MiB) @ rwkv_rsv::runtime::Runtime::create_tensor
```

紧接着才是 temp/counter（各 16 MiB）——即 self-loop 阶段真正的大头是这一条。

## 二、根因：槽维被乘了两遍

分解 `335 544 320`：`= 8 × 64 × 10 240 × 64`，其中

| 因子 | 含义 | 出处 |
| --- | --- | --- |
| 8 | KS 上限 | `imma_ksplit` 的 `VK_IMMA_KSPLIT_MAX` 默认 8 |
| 64 | 「一波槽数」 | 写死在 `WorkBuffers` 的字面量里 |
| 10 240 | `max(c, ffn_hidden)` | m 的最大值（fh=10240 > c=2560） |
| 64 | **batch** | `mk(len)` 助手内部的 `create_tensor(len * bsz)` |

契约在 `runtime.rs::imma_ksplit`：部分和布局 `[KS][batch][M]`，约束 `ks·batch·m ≤ partial.len`。
⇒ **正确长度 = 每槽 `8 × max(c, fh)`**，`mk` 已经乘过 `bsz`。字面量里那个 64 是第二次计入槽维，
于是尺寸 = 正确值 × 64：B=64 该是 20 MiB 却给了 1280 MiB，B=256 该是 80 MiB 却给了 5120 MiB。

改动（`src/gpu_model.rs`，一行 + 注释里把算式写清楚）：

```rust
- imma_partial: mk(&mut *backend, 8 * 64 * c.max(cfg.ffn_hidden))?,
+ imma_partial: mk(&mut *backend, 8 * c.max(cfg.ffn_hidden))?,
```

## 三、为什么可以断言逐位等价（不是「应该没变」，是不等式不 binding）

新容量 `cap' = 8·batch·max(c,fh)`。`imma_ksplit` 只在
`ks·batch·m > cap` 时缩 ks ⇒ 缩的条件等价于 `ks·m > 8·max(c,fh)`。

- 对所有 `m ≤ ffn_hidden` 的算子（key/val/out/低秩…）：`ks ≤ 8` 且 `m ≤ max(c,fh)` ⇒ `ks·m ≤ 8·max(c,fh)`
  **恒成立**，ks 与改动前完全一致。
- 对 head（`m = vocab = 65 536`）：块数 `m/bm ≥ 目标 400` ⇒ 本来就取 `ks = 1`，与容量无关。
  （若被 `VK_IMMA_KSPLIT` 强行抬高，新容量会把它缩回去——那是**更安全**的方向。）

实测核对（TOPP=1.0，`SEGS=2 NTOK=16 PAD_TO=512 WARMUP=1`，用一个临时的
`IMMA_PARTIAL_PAD=64` 旋钮把旧尺寸还原，与修正后**交替**各跑两轮）：

| batch | 尺寸 | self-loop 后登记 | 外部峰值 | decode tok/s（两轮） | 解码指纹 |
| --- | --- | --- | --- | --- | --- |
| 192 | 旧 | 15 135 MiB | 17 885 | 1486.3 / 1465.4 | `0x4f09628/0x8e94` |
| 192 | 新 | **11 335**（−3800） | **14 085** | 1491.8 / 1474.1 | `0x4f09628/0x8e94` |
| 256 | 旧 | 17 807 MiB | 20 585 | 1505.1 / 1497.6 | `0x6875c61/0x876b` |
| 256 | 新 | **12 747**（−5060） | **15 528** | 1490.0 / 1496.2 | `0x6875c61/0x876b` |

另外 B=64/128/192/256 的 `TOPK=0`（走 top-p 兜底循环）三档指纹同样与改动前相同
（`0x343cfe4/0x6f1c` 等）。「模型加载后」那一档也从 5167 → 5147 MiB——那是 batch=1 的单流
`WorkBuffers` 里同一块缓冲（20 MiB → 80 KiB），与本条同源。

## 四、被否证/订正的旧表述

1. `README` §9「`imma_partial` **不能**按 `bsz` 配……按 `bsz` 预留反而在 B=256 多要 63 MiB
   ⇒ `ERROR_OUT_OF_DEVICE_MEMORY`」：**错**，且当时并未回滚。真实约束是上面的不等式；
   那次改动既留着 `× bsz` 又多乘 64，付出的是 64 倍（不是 63 MiB）。已在 README/README_CN 订正。
2. `2026-10-10-Vulkan采样mask位图化与显存记账-实施记录.md` §二 第 1 点
   「大并发显存的第一大户是 RNN 状态」：在**修正后**成立（状态 5.3 GB ≫ 其余），
   但**修正前**它和这块超配的 `imma_partial`（5.1 GB）并列，当时的「self-loop +5471 MiB」
   其实是后者。该条已在本记录里补记。
3. `§6.2③「B≥192 吞吐断崖是显存溢出」`：本次在旧尺寸下也复现不出断崖，README 已把
   「容量」从待解释项里划掉，剩下的是 0.413× 本身的核侧差距（低秩链、批量 prefill）
   与 fp32 状态这一占用项。

## 五、门禁

`cargo fmt --all`、`BACKEND=vulkan cargo clippy --all-targets --release -- -D warnings`、
`BACKEND=vulkan cargo test --release`、`BACKEND=cuda cargo test --release` 全绿；
`rwkv-router` 侧 `cargo check --release --all-targets` 绿（vendor 同步后）。
诊断用的 `TRACE_ALLOC` 探针与 `IMMA_PARTIAL_PAD` 旋钮**用完即删**，未进入提交。

## 六、下一步

- 占用侧还剩 **fp32 RNN 状态**（每槽 20.6 MiB，B=256 = 5.3 GB，现在确实是第一大户）→ 任务 #27。
  本次修正后 B=256 只剩 12.7 GB 登记，fp16 化状态不再是「能不能跑」的前提，而是「能不能上更高批宽」。
- 核侧差距（更值钱）：低秩链批量张量核（任务 #25，占 B=128 decode 的 15.5%）、
  批量 prefill 的 int8 张量核（任务 #26，fp16 coopmat 占 prefill 86%，口径 0.59×）。
- 方法论：**登记张量的阶段增量对不上已知结构时，别推论，插分配口**。这次的 5.1 GB 靠读代码
  读了两个小时没读出来，插三行日志一次跑出。
