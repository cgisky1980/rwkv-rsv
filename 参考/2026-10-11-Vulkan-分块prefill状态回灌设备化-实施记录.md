# Vulkan 分块 prefill 的状态回灌设备化（2026-10-11）

目标（#33 / README §6.7）：上一轮（瓦片档修正）把 B=128 批量 prefill 推到「0.96× CUDA」，
并把 B=256 的残余损失归因为「每 chunk 有 ~1.3 s 墙钟落在内核与录制之外」。本轮先证伪这个
归因的一半，再消掉另一半。结果：**B=256 0.67× → 0.79×**、**B=128 0.84×**，行为逐位不变。

## 一、先复现「修复前」的失败，而不是先改代码

两条失败都得留证据：

1. **判据失败**（先于任何优化）：新写的 CI 用例第一轮就在 CUDA 上跑红——
   `槽 0：子批宽度改变了 prefill 结果（分块 prefill 的前提不成立）`。这不是实现 bug，
   而是暴露了一条此前没人查的跨后端口径（见第四节）。
2. **归因失败**：`2350 ms / chunk` 来自 `75 197 / 32`，其中 75 197 是 bench 的
   `[批量] prefill` 列。该列 `t0` 在 `create_batch_state` **之前**、`t1` 在**解码循环之后**
   ⇒ 含 tokenize、状态分配、`WARMUP` 自循环（图捕获 / NVRTC JIT）。这是这条线上第 3 个
   基准口径 bug：同一列的 CUDA/CUDA 对比把 B=128 撑成 0.96×、把 B=256 压成 0.61×。

**干净口径**（本轮所有数字的唯一来源）：

```bash
PROF_CHUNK=1 PAD_TO=512 NTOK=1 SEGS=1 WARMUP=0 SKIP_SINGLE=1 \
  ./target/release/examples/batch_decode_bench        # 读 chunk 循环自己的合计，两后端同一二进制同一会话
```

`PROF_CHUNK=1` 是本轮新加的分解仪表（`forward_seq_batch_padded` 的 chunk 循环），把每块墙钟
拆成 `reset / core / xfer` 三段——**先有分解再谈归因**。

## 二、1.3 s 到底在哪：实测分解

| B | 后端 | reset | core（GPU） | xfer | chunk 循环合计 | vs CUDA |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | CUDA | 1 212 | 14 022 | 60 | **15 294 ms** | 1.00× |
| 128 | Vulkan | 1 007 | 17 165 | 20 | **18 191 ms** | **0.84×** |
| 256 | CUDA | 2 370 | 26 078 | 62 | **28 510 ms** | 1.00× |
| 256 | Vulkan，宿主往返（旧） | 1 907 | 33 810 | **6 713** | 42 430 ms | 0.67× |
| 256 | Vulkan，设备内拷贝（新） | 1 980 | 33 999 | **44** | **36 023 ms** | **0.79×** |

每 chunk（B=256 / `PREFILL_SLOTS=8`，32 块）：旧代码 1 326 ms = reset 62 + core 1 030 +
**xfer 210**。所以那 1.3 s **不在 chunk 之间，而在 chunk 里面**，而且正好落在上一份记录候选
清单的第一条——「状态写回」。GPU 侧的 `PROF_GPU / PROF_GAPS / PROF_HOST / PROF_SUBMIT`
四个仪表之所以看不见它：它们挂在 batch 路径的提交上，而回灌走的是**每槽 194 次独立提交**。

## 三、病根：`download_part`/`upload_part` 每次都自带一次同步提交

Vulkan 侧这两个入口各自 `Runtime::copy_range_standalone`——录一个**临时** command buffer、
提交、然后 `queue_wait_idle`。而分块 prefill 的回灌原来是逐槽 `state_slot_back` +
`state_slot_load`，每槽 `2 × (3·n_layer + 1)` = **194 次**，B=256 每 chunk 1 552 次，
~1 ms 一次 ⇒ 实测 200–225 ms。CUDA 侧没有这个症状（`lens` 是另一张表），这也是为什么
「跨后端逐位一致」的历史对拍从没抓到它。

**改法**（`src/gpu_model.rs`）：`State::copy_rows_from` + `GpuModel::state_rows_copy`。
子批的前 `chunk.len()` 行连续、大 state 里目标行段 `ci·w` 起也连续 ⇒ 每类张量一条
`copy_range`，整个 chunk `3·n_layer + 1` 条命令收进**一次** `begin_batch … end_batch`
提交。补齐槽（chunk 尾不满 `w`）不回灌。chunk 循环里另加 `PROF_CHUNK` 三段计时。

`copy_range` 在 `self.recording` 时只录不提交，因此一次提交；这条依赖 backend.rs:1588 那个
「src 只读句柄必须在 `remove(dst)` 之前取」的修复（同张量区间拷贝），两者是同一个提交里
互相咬合的。

## 四、判据：新 CI 用例分两档钉住两件事

`tests/batch_parity.rs::cross_state_rows_copy_matches_host_roundtrip`

1. **产品主张（两后端都必须逐位全等）**：设备内行段回灌 ≡ 宿主往返读数；另测
   `src_slot` 非 0（源偏移不是硬编码 0）与 `rows=1` 不越界改写下一行。纯搬数据，不涉数值。
2. **分块 prefill 的前提（按后端分档，实测）**：同一 prompt、同 `pad_to`，只改批宽。

| 后端 | w=1 vs w=4 | w=2 vs w=4（槽 0） | 断言 |
| --- | --- | --- | --- |
| Vulkan | 0 / 5 409 280 位不同 | 0 | 全等 |
| CUDA | 3 996 677 位，`max_abs=6.250e-2`（`max_rel=3.027e-2`；槽 1 `1.227e-2`） | 与 w=1 **完全相同** | 只报告 |

★ 这一条是本轮最有价值的新知识：**CUDA 上 `PREFILL_SLOTS` 是影响结果的旋钮**，而且不是
舍入噪声——w=1 与 w=2 彼此逐位相同、却都与 w=4 差 ~1 %，说明批形跨过某个阈值就换了一整套
kernel（cuBLAS 按 M 选算法），不是累加顺序慢慢漂移。Vulkan 的 coopmat GEMM 单行点积与 M
无关，故逐位不变，`PREFILL_SLOTS` 在 Vulkan 上是**纯性能旋钮**。第一版把它写成
`max_rel <= 1e-3` 的容差断言，首轮即红；改成「Vulkan 断全等 + CUDA 只报告 + 公共下限：
不得出现非有限值」。

★ 为什么 `prefill seeds` 不够：`forward_seq_batch_padded` 返回的 seeds 是各槽**末 prompt
token**，与 GPU 算了什么无关——它只能证明行搬到了对的槽，证明不了数值等价。能判的只有状态
本身（这里）和下游解码 token（基准里）。

★ 一次 `assert_eq!` 两个 5.4 M 元素 `Vec<u32>` 打出 **122 MB** 日志。失败信息一律走 `bit_diff`
紧凑摘要（不同位数 / 总数 / max_abs / max_rel）。

## 五、行为等价的三层证据（本轮改动是纯改写，等价是**全部**主张）

- `prefill seeds sum` 不变：B=256 `0x372cc8`、B=128 `0x1b9664`，旧/新/两后端各臂全等。
- 256 槽贪心解码指纹不变：`0x2799db / 0x63ab`。
- 新 CI 用例（第四节第 1 条）在两后端逐位通过。

## 六、门禁

```
cargo fmt --all                                                          ✅
cargo clippy --release --all-targets -- -D warnings                      ✅ exit 0
BACKEND=cuda   cargo test --release --test batch_parity -- -j1           10 passed / 1 ignored（71.2 s）
BACKEND=vulkan cargo test --release --test batch_parity -- -j1           10 passed / 1 ignored（75.6 s）
BACKEND=cuda   cargo test --release --lib -- -j1                         77 passed / 1 ignored
BACKEND=vulkan cargo test --release --lib -- -j1                         77 passed / 1 ignored
```

## 七、遗留（按体量排序）

1. **`reset` 是同一身病换个衣裳**：`State::reset` 用 `upload` 逐段写零 ⇒ 每 chunk 97 次
   同步提交 + w=8 时每块 165 MB 主机零缓冲。实测 1.0 s（B=128）/ 2.0 s（B=256），占 chunk
   循环 5.5 %；CUDA 也付 1.2 / 2.4 s。修法是后端级**设备内填充**
   （`vkCmdFillBuffer` / `cuMemsetD32Async`，需要给 CUDA 侧补绑定），不是把子批调小。
2. **`core` 才是剩下的全部故事**：B=256 33 999 vs CUDA 26 078 ms = 0.77×，形状级差距见
   README §6.6 的内核表；其中 `pf_value`（深 K 窄 N）仍 0.45×（#34）。
3. **上一轮的 PREFILL_SLOTS × 显存表全部要按干净口径重测**（六臂的 ms 与 `vs CUDA` 列都读自
   污染窗口；只有「登记显存随宽度上涨 / 64 与不分块 OOM」的定性结论仍成立）。已在瓦片档记录
   §5.1 末尾贴了订正块。
4. `create_batch_state` 在 B=256 是 3.7 s（Vulkan）/ 6.2 s（CUDA），落在 chunk 循环之外、
   bench 的 prefill 列之内——引用任何 prefill 数字时都要说明含不含它。
