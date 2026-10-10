# Vulkan 追平 CUDA：张量核 rkv 路径修复 + 大批量显存断崖（2026-10-10）

目标：「继续优化 vulkan，全面赶上 CUDA」。本轮做了三件事，其中两件**推翻了仓库里已有的结论**。
全部读数同会话交错测得（本机 GPU 兼作显示卡，跨会话读数不可比）。

## 1. 结论摘要

| # | 事 | 之前怎么写 | 实测真相 |
|---|---|---|---|
| ① | `GEMV_BGRP` 扫描 | 注释挂着「从未在 batch≥16 做过 A/B」 | 8 就是最优；16/24/32 分别 −10%/−24%/−27%，原因是 `float acc[ROWS][BGRP]` 寄存器溢出 |
| ② | 张量核 r/k/v 路径 | README「实测净亏 3.3×，默认关闭」 | **从没跑通过**：任何 batch 都 `ERROR_DEVICE_LOST`。修好后 **+27.9%~+42.9%**，每一档都赢 |
| ③ | B=64→256 吞吐平台 | 归因给张量核 tile 参数（待扫 `VK_IMMA_NST/BM/KSPLIT`） | **是显存溢出**：B≥192 时 WDDM 静默落系统内存，采样核单次 0.87 ms → 49.99 ms → 68.0 ms |

②③ 合起来改变了「差距」的构成：可即时回收的那一部分比原先假设的大得多（只是卡在未验证的数值
口径上），而原先记在「核质量」名下的 B=256 落后里，混着一段容量效应。

## 2. ② 的根因：`Params` 26 个引用，host 只填了 24 个

`Runtime::gemv_int8_rkv_mid_batch`（`gemv_int8_rkv_stage1_batch` 的 `RKV_OFF=1` mid 投影臂）
的 `params` 数组按「直觉顺序」写：6 个 int8 占位 → v1/w1/a1/g1 → **xv/xw/xa/xg 连着放** → 6 个占位
→ 4 个 mid 输出。而 shader 的 `Params` 是：

```
0..5  in_r_idx / in_r_sz / in_k_idx / in_k_sz / in_v_idx / in_v_sz
6..9  in_v1 in_w1 in_a1 in_g1
10..12 in_xr in_xk in_xv          ← InX4 三个
13..15 in_xr16 in_xk16 in_xv16    ← InX2 三个
16..18 in_xw in_xa in_xg          ← BufF 三个
19..21 out_r out_k out_v
22..25 out_vm out_wm out_am out_gm
```

即 `in_xr/in_xk` 两个占位被漏掉、`out_r/out_k/out_v` 三个占位也只补了 6 个而不是 5+3 个，
总数 24 ≠ 26 ⇒ 从 `in_xv` 之后**整体前移两位**。后果分两种，且都符合现象：

- v/w 链移位后读到的仍是**有效指针**（只是指向别的张量）⇒ **跑通但算错**；
- a/g 链的 `out_am`/`out_gm` 落到**从未写过的 uniform 池字节**上，被当指针解引用 ⇒ `ERROR_DEVICE_LOST`。

偏移固定 ⇒ 「只在部分分支发作」，这是它六天没被看懂的全部原因。

### 定位过程（值得复用的手法）

1. **先复现再修**（用户纪律）：`BACKEND=vulkan SLOTS=8 … VK_IMMA_OPS=…,rkv` 在任何 batch 都
   `ERROR_DEVICE_LOST`；同参数 `VK_IMMA=0` 正常 ⇒ 锁定在 IMMA 分支内。
2. **臂级二分**（临时 env `VK_RKV_BISECT`，位选 1=quant+gemm / 2=SIMT 补齐 / 4=mid 臂）：
   只有位 4 崩 ⇒ 锁定 `gemv_int8_rkv_mid_batch`。
3. **链级二分**（临时 env `VK_RKV_MIDS`，位选 vm/wm/am/gm）：v/w 干净（55.1/56.4 tok/s），
   a/g 崩 ⇒ 锁定 a/g 链。
4. **换读侧**（临时 env `MID_SWAP_AW`，用 w 的权重/x 顶替 a 的输入）：仍然崩 ⇒ **读侧无罪，写侧或槽位错位**。
   这一步是关键分岔：它排除了「a1/xa 张量尺寸不对」这一整类假设。
5. **逐 dispatch 提交**（临时 env `SUBMIT_EACH`）把「哪条 dispatch 让设备丢失」钉死：
   `[SUBMIT_EACH] 本条 = gemv_int8_rkv_stage1_batch.spv grid=(96,1,1)` → `ERROR_DEVICE_LOST`，
   而对照（只开 vm 臂）exit 0。
   ⚠️ 这个探针**第一版是坏的**：只在每条后 `end_batch()` 会让后续 dispatch 落在已结束的
   command buffer 上，两种配置都 segfault（EXIT=139），输出全部作废。修成
   `end_batch(); begin_batch();` 后先验证对照配置 exit 0 才开始采信。
6. **排除两条常见误判**：(a) `.spv` 陈旧——用 `glslangValidator` 重编到临时文件 `cmp`，
   **逐字节相同**，说明编译产物与源码一致，问题在源码/参数侧；(b) 特化常量错位——
   `create_kernel_unsafe` 用 `.enumerate()` ⇒ 下标即 `constant_id`，位置映射正确。

### 修复

`params` 逐字段标注（每项写明对应 shader 字段号），并把长度约束变成编译期检查：

```rust
let params: [u64; 26] = [
    ph,   // 0  in_r_idx
    …
    xv.device.address,  // 12 in_xv
    …
    out_gm.device.address, // 25 out_gm
];
```

只改这一处；`reads`/`writes` 原本就正确。

### 修后实测（同会话交错，`VK_IMMA_OPS=…,rkv` vs 默认）

| batch | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| SIMT r/k/v | 849.0 | 921.4 | 965.1 | 679.0 / 693.0 |
| 张量核 r/k/v | 1086.1 | 1231.0 | **1378.9** | 899.0 / **906.7** |
| 比值 | +27.9% | +33.6% | **+42.9%** | +29.7% |

**决定：仍然默认关闭。** 拦路石从「性能」换成「**未验证的数值口径**」——该路径把 r/k/v 的
**激活**也量化成 int8（`quant_x_i8`），与 SIMT 的「fp32 激活 × int8 权重」不是同一个算术，
token 指纹逐档不一致（B=256：SIMT `sum=0x68ce489 xor=0x7243`，IMMA `sum=0x69298d6 xor=0x6d52`）。
在没有质量证据（同批 prompt 的 logit 偏差 / Top-1 一致率）之前把默认数值口径换掉，不属于性能优化。
⇒ 后续项 #22。

顺带澄清一条同源误记：`gemv_int8_rkv_mid_batch` 属于 `3374512`（「gate the r/k/v tensor-core path off
(net -3.3x) and keep the infra」）。那次「测出 3.3× 净亏」的运行根本没能执行这条臂——
README 里的 3.3× 是把崩溃读成了慢。

## 3. ① BGRP 扫描（补上注释里欠的 A/B）

B=256、`SEGS=1 NTOK=32 PAD_TO=128 PROMPT_REPEAT=2 SKIP_SINGLE=1`，只变 `GEMV_BGRP`：

| BGRP | 4 | **8** | 16 | 24 | 32 |
|---|---|---|---|---|---|
| tok/s | 795.2 | **986.9 / 974.3** | 890.3 | 750.7 | 718.1 |

带宽模型预测 BGRP=32 把 r/k/v 权重读 8 遍而非 32 遍，理当完胜；实测 −27%。
瓶颈是 `float acc[ROWS][BGRP]`（ROWS=4）的寄存器压力：>8 溢出到 local memory，
省下的带宽被 spill 存取吃掉。

**所有档位 token 指纹逐位一致**（`sum=0x68ce489 xor=0x7243`）⇒ BGRP 不改数值，
且 `slot_cnt = min(BGRP, BATCH - slot0)` 尾块守卫在 BGRP 不整除 batch（24）时也正确。

⇒ 「调 SIMT 分组把 B=256 的带宽账算平」这条路**已封死**；更多槽共用一次权重读取只能走张量核。
另外记下：BGRP=8 的 986.9 与同日稍后测得的 679~703 不可直接比——那是机器状态（见 ③，
当时 B=256 的工作集已把显存推满）。同会话内 `GEMV_BGRP=8`（698.4）≡ 不设（702.8），
证明我的改动没有让默认路径回退。

## 4. ③ B≥192 是显存溢出，不是核质量

`TOPP=1.0`（关掉 top-p 兜底循环，去掉数据相关项）扫并发：

| batch | 64 | 128 | 192 | 256 |
|---|---|---|---|---|
| decode tok/s | 1275.3 | **1430.1** | 1034.7 | 1019.1 |
| `sample.spv` 单次 | 0.58 ms | 0.87 ms | **49.99 ms** | **68.0 ms** |

64→128 涨 1.5 倍，128→192 跳 **57 倍**——同一段代码，这不是算法形状能给出的曲线，是内存放置。
`nvidia-smi` 在 B=256 decode 期间实测 **21 775 / 22 528 MiB**：本机是改成 22 GB 的卡，桌面与其它
进程常驻 ~8.6 GB，本进程峰值 ~13.1 GB。显存被推满后 WDDM **静默落到系统内存**，采样核那 4 张
`[batch, vocab]` 工作区（B=256 各 67 MB）变成 PCIe 访问。

旁证（同一轮 PROF_GPU 读数，B=256 单步约 275 ms 的构成）：
`sample 188.6` / `gemm_imma_bm64_nst8 186.9` / `gemv_lowrank_chain4_batch 58.9` /
`fuse_ka_dplr_norm 51.1` / `gemv_int8_rkv_stage1_batch 39.0` / `gemm_imma_reduce 16.2`（ms，2 步合计）。
⇒ 采样核占整步 34%，所以 ③ 的价值高于原计划的 tile 扫描（任务 #21 因此降级）。

排除过的两条：
- **top-p 兜底循环**（`MAX_TOP_P_ITERS = 512`，一趟全词表只消费一个候选）：`TOPP=1.0` 关掉后
  单次仍 66 ms；`TOPK=40` 反而 117.6 ms。循环只占 ~27 ms，主项在别处。
  （顺带确认 `g_cum` 是 `shared`，不存在「只有 thread0 更新导致永不收敛」的分叉 bug。）
- **批量 prefill 的 scratch 没释放**：`SLOTS=256 PAD_TO=32 + PROMPT_REPEAT=1` 峰值仍 21 736 MiB
  （对比 `PAD_TO=128` 的 21 775）⇒ 与大 `PAD_TO` 无关，占用是 batch 线性项（状态 + 每槽工作区）。

**可执行的下一步**（本轮未动，见任务 #23）：
1. Vulkan 侧 RNN 状态仍是 fp32（`supports_f16_wkv_state()` 为 false）：每层每槽
   `H·N·N = 163 840` 元素 ⇒ B=256 状态合计约 5.3 GB，转 fp16 省 2.6 GB。
2. `WorkBuffers` 每槽工作区（估 ~30 MB/槽）需要逐项核一遍 dtype。
3. **`log_vram` 是坏的**：B=256 全程打印「登记张量 0 MiB」，13 GB 的占用和「卡被推满」在我们
   自己的日志里完全不可见，只能靠外部 `nvidia-smi`。这条要先修，否则后续任何内存优化都无法验证。
4. 运维口径：**批量 decode 的实用甜点是 B≈128**；把 SLOTS 拉到 256 是负收益。

## 5. 验证与门禁

- `cargo fmt --all`、`cargo clippy --all-targets --release -- -D warnings` 均 exit 0。
- `BACKEND=vulkan cargo test --release`：77 + 8 + 1 passed，0 failed。
- `BACKEND=cuda cargo test --release`：77 + 8 + 1 passed，0 failed。
- 修复后 `VK_IMMA_OPS=…,rkv` 冒烟（B=8，`SEGS=1 NTOK=4`）exit 0，正常出 token。
- 临时诊断已全部撤除：`VK_RKV_BISECT`（gpu_model.rs）、`VK_RKV_MIDS` + `MID_PROBE`/`MID_ADDR` +
  `SUBMIT_EACH`（runtime.rs）、`MID_SWAP_AW`（backend.rs）；`git diff` 只剩 params 修复与
  bench 侧新增的一个旋钮（`TOPP`，与既有 `SAMPLE_TEMP`/`TOPK` 同风格，保留）。
  gpu_model.rs / backend.rs 用 `git checkout` 还原。还原前的临时诊断补丁与本轮全部原始日志
  （`bench/` 下的 sweep/prof/curve/spill 与 `.jsonl`）已按工作区纪律随产物一并删除——上表数值
  已完整转录进 `README_CN.md` §6.2，撤除的诊断项名称与落点在本节逐条列出，故无信息丢失。

## 6. 文档订正落点

- `README_CN.md` 新增 §6.2（三处更正，含全部表格）；§6.1 第 1 条追加两点修正；§9 的
  「净亏 3.3×」条目改为带删除线的更正叙述，并声明「B=256 的 3.8× 不能纯当核质量读」。
- `README.md` 同步英文 §6.2 与上述两处。
