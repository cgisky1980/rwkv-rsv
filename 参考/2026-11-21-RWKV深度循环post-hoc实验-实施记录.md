# RWKV 深度循环 post-hoc 实验（P0/P1 前置）实施记录

**日期** 2026-11-21 · **仓库** rwkv-rsv · **状态** 已完成 · **结论** post-hoc 方案 No-Go，训练版仍待验证

---

## 1. 本次做了什么

对 `rwkv_recurrent_depth_plan.md` 的**推理时 post-hoc 方案**（计划 §8 的 No-Go 备选路径）做了首个可复现实验：拿一个**已训练好的普通 RWKV-7**，把中间连续一段层当作**权重共享的循环块**，同一 token 跑 L 次，量质量、吞吐与 state 大小。

必须强调：这**不是**计划正文的 Stage A/B（那些需要按随机 L 采样重新训练）。本实验回答的是 narrower 的问题——「不训练、直接把现有模型 loop 起来能不能白拿深度」。

### 1.1 代码改动

- `src/model.rs`：新增 `StateCommit`（S1 naive / S2 first / S3 last）、`DepthConfig`（N_in / N_loop / N_out、L、重注入开关）、`Model::forward_recurrent`。原 `Model::forward` **未改动**。
  - 非 commit 轮使用循环开始前的冻结 state 快照，每轮独立 clone 后丢弃 ⇒ 不写回时间 state（严格对应计划附录 B）。
  - `v_first` 在 InputBlock 第 0 层建立，循环与 OutputBlock 复用。
- `examples/recurrent_depth_eval.rs`：teacher-forced 评估器（CE / PPL / top-1 / state 字节 / 每 token 墙钟），语料槽位间状态独立，按槽位开线程并行。
- 原始日志：`outputs/recurrent_depth_2026-11-21.log`（gitignore，未入库）。

### 1.2 P0 数值校验

| 配置 | CE | top-1 |
| --- | --- | --- |
| 原 `Model::forward`（4 槽 × 24 目标） | 3.717716 | 0.395833 |
| `forward_recurrent`，L=1、N_in=24、N_loop=0、N_out=0 | 3.717716 | 0.395833 |

逐位一致（6 位小数）。循环通路本身没有引入数值偏移。

---

## 2. 实验设置

| 项 | 值 |
| --- | --- |
| 模型 | `rwkv7-g1a-0.4b-20250905-ctx4096.st`（24 层，C=1024，H=16，N=64） |
| 词表 | `.ai00-x-dev/models/rwkv/vocab.json` |
| 语料 | 仓库 `README.md`（英文），32 个互不重叠槽位 |
| 每槽 | prefill 16 token + 24 个 teacher-forced 目标 ⇒ 共 **768** 个评分位置 |
| 并行 | 16 线程（槽位级），seed/顺序固定 |
| 分段 | 中层循环：N_in=6 / N_loop=12 / N_out=6（有效深度 24 / 36 / 60） |
| 后端 | CPU fp32 参考（确定性，无 GPU 噪声） |
| GPU | 机器有 RTX 2080 Ti 22GB，但 GPU 后端尚未支持层粒度循环，本次未用 |

---

## 3. 结果

CE 单位 nats/token；PPL = exp(CE)；`ms/tok` 是 16 线程汇总后的墙钟均值（非单线程）。

| 臂 | L | 提交 | 重注入 | CE | PPL | ΔPPL vs B0 | top-1 | 有效深度 | ms/tok |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| B0 baseline | 1 | — | — | 3.739433 | 42.07 | — | 0.3997 | 24 | 42.1 |
| S3_L1 sanity | 1 | last | 1 | 3.742294 | 42.19 | +0.3% | 0.4010 | 24 | 41.2 |
| **S2_L2** | 2 | **first** | 1 | **3.870874** | **47.98** | **+14.0%** | **0.3828** | 36 | 49.2 |
| S1_L2 | 2 | naive | 1 | 4.317190 | 74.98 | +78.2% | 0.3203 | 36 | 47.4 |
| S3_L2 | 2 | last | 1 | 4.481406 | 88.36 | +110.0% | 0.2865 | 36 | 48.3 |
| S2_L4 | 4 | first | 1 | 4.436925 | 84.51 | +100.9% | 0.3203 | 60 | 60.9 |
| S1_L4 | 4 | naive | 1 | 5.533749 | 253.09 | +501.7% | 0.1823 | 60 | 61.3 |
| S3_L4 | 4 | last | 1 | 5.821368 | 337.43 | +702.4% | 0.1693 | 60 | 63.2 |
| A1 无重注入 S3_L2 | 2 | last | 0 | 4.651561 | 104.75 | +148.9% | 0.2734 | 36 | 48.7 |
| A5 底层 S3_L2 | 2 | last | 1 | 7.638288 | 2076.19 | +4834% | 0.0846 | 36 | 48.8 |
| A5 顶层 S3_L2 | 2 | last | 1 | 6.064267 | 430.21 | +922.5% | 0.1406 | 36 | 48.0 |

**state 字节数**：所有臂均为 **6,488,064 B**（24 层 × (2C + H·N²) × 4），与 L 完全无关；循环块的临时快照 3,244,032 B 是每轮复用、用完即弃的 scratch，不随 L 累积。

**S3_L1 sanity 为什么不是逐位等于 B0**：L=1 时重注入仍执行了一次 `h = F(h) + h0`，这是计划公式 `h_i = F(h_{i-1}) + h_0` 在 i=1 的直接结果，属于额外残差，不是 bug。用它作为「启用重注入的 L=1 对照」才公平（见分析）。

---

## 4. 假设判定

| 假设 | 判定 | 依据 |
| --- | --- | --- |
| H1（L=2 PPL 下降 ≥3%） | **否证（post-hoc 范围）** | 最好的 S2_L2 反而 PPL +14%；S3_L2 +110%。 |
| H2（state 显存与 L 无关） | **结构性成立** | 常驻 state 恒为 6.49 MB，L=1/2/4 完全相同；权重共享、无 KV cache。GPU 实测未做（见局限）。 |
| H3（L 可作推理旋钮） | **否证（post-hoc 范围）** | 三种策略下 CE 都随 L 单调**变差**；L=4 最差。 |
| H4（重注入与提交策略影响显著） | **成立，但方向修正** | 无重注入比有重注入差 18.5% PPL（104.75 vs 88.36）；提交策略差异巨大，但**S2 first 最优、S3 last 最差**，与计划 §3.2 的预期相反。 |
| H5（收益集中在推理任务） | 未测 | 本轮只有语言建模 CE/top-1，没有下游与合成任务。 |
| A5（循环块位置） | **支持「中层最优」** | 同为 S3/L=2：中层 88.36 < 顶层 430.21 < 底层 2076.19。 |

---

## 5. 分析

1. **post-hoc 循环拿不到免费深度。** 循环块权重是按「只执行一次、输入分布固定」训练出来的；同一 token 第二次穿过它时，输入分布已经偏移，RMSNorm / DPLR state 更新都不再处在训练流形上。CE 随 L 单调恶化，与 Dory/Ouro 等**专门训练过**的 looped 模型结论不矛盾。

2. **S2 优于 S3 的机制解释（对训练版也有参考价值）。** S2 在时间 state 上只写一次，且写的是**第一轮、分布最正常**的 k/v；后续轮只精化 h，不污染时间记忆。S3 把写入推迟到最后一轮，此时 k/v 已经过一次权重重作用、分布偏移最大，等于把被扭曲的信号写进长期 state，逐 token 累积后灾难性放大。计划把 S3 设为默认、S2 视为「后几轮看不到历史」的妥协，这个先验应当**反转**：训练版应优先消融 S2。

3. **重注入是稳定器，不是收益来源。** A1 去掉重注入后 S3_L2 从 88.36 恶化到 104.75，说明它防漂移的作用确实存在；但它无法把 post-hoc 循环变成正收益。

4. **位置敏感性极高。** 底层循环（PPL 2076）几乎彻底破坏模型——底层 representation 还太"原始"，重复作用同一组权重会直接摧毁后续所有层的输入。中层最稳。

5. **吞吐代价小于 L 的线性倍数。** 只有 12/24 层被循环，L=1→2→4 的 ms/tok 为 42→49→63（×1.17 / ×1.50），符合「深度循环只加 FLOPs、不加 state」的工程预期。

---

## 6. 局限与下一步

**局限**：① 只测 0.4B 小模型、单一英语语料、768 个评分位置；② CPU fp32 后端，GPU 显存优势只做了结构性论证、没有 nvidia-smi 实测；③ 没有 H5 的任务级评估；④ 数据顺序固定，未做多 seed 置信区间。

**建议**：

1. **计划 §8 的 No-Go 分支（post-hoc）就此关闭**，不要把 L 当现成模型的推理旋钮。
2. 若要继续主线，**必须训练**：按计划 Stage A（150M，随机 L∈{1,2,4} 采样）做，且把 **S2 first-commit 提到默认/优先消融位**，S3 降为对照。
3. 训练版落地后，优先在 GPU 后端加层粒度调度，用 3B int8 模型补 H2 的 `nvidia-smi` 峰值显存曲线与 L-曲线；这是本项目相对 looped Transformer 的核心卖点，必须有实测。
4. 后续评估沿用本例的 teacher-forced 口径（确定性、跨进程可比），避免贪心轨迹放大微小 logits 差异导致假结论。

---

## 7. 复现命令

```bash
EXE=target/release/examples/recurrent_depth_eval.exe
MP="C:/work/ai00_server/rwkv7-g1a-0.4b-20250905-ctx4096.st"
VJ="C:/work/ai00-x-dev/.ai00-x-dev/models/rwkv/vocab.json"
CO="C:/work/ai00-x-dev/rwkv-rsv/README.md"
cargo build --release --example recurrent_depth_eval

# 基线
env MODEL_PATH=$MP VOCAB_JSON=$VJ RD_CORPUS=$CO \
  RD_MODE=baseline RD_SLOTS=32 RD_THREADS=16 RD_PROMPT=16 RD_TAIL=24 $EXE

# S2 / L=2（post-hoc 最优臂）
env MODEL_PATH=$MP VOCAB_JSON=$VJ RD_CORPUS=$CO \
  RD_MODE=recurrent RD_L=2 RD_COMMIT=first RD_REINJECT=1 \
  RD_IN=6 RD_LOOP=12 RD_OUT=6 \
  RD_SLOTS=32 RD_THREADS=16 RD_PROMPT=16 RD_TAIL=24 $EXE
```

原始逐臂输出见 `outputs/recurrent_depth_2026-11-21.log`。
