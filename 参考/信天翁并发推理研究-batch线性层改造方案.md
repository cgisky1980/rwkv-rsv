# 信天翁（Albatross）并发推理研究 — batch 线性层改造方案

> 日期：2026-08-22。目的：解决本实现 batch 并发（B slot 共享权重）decode 零增益问题。
> 结论来源：`C:\work\rwkv-runtime\Albatross\faster3a_2605`（BlinkDL 本地参考实现）源码分析
> + 本实现 B 扫描实测（B=1 vs B=8 decode 时间相同）。

---

## 1. 问题回顾：为什么我们的 batch 版零增益

本实现第一版 batch 并发（2026-08-22 上午）已把全部 decode kernel 加了 batch 维
（grid.y=slot 或 [batch,...] 布局），元素级 kernel（norm_lerp6/cmix_norm_lerp/
fuse_ka_dplr_norm）共享权重读一次算 B 份，但**线性层 gemv kernel 只是把 grid 乘了
batch 倍**——每个 slot 的 block 独立扫全量权重，权重带宽需求 ×B。

实测（RTX 2080 Ti，rwkv7-g1i-2.9b int8，105 场景全量）：

| batch | decode 总时间 | 吞吐 |
|---|---|---|
| B=1 | ≈17.3s（16 摘要小样本） | 0.77/s |
| B=8 | ≈18.6s | 0.72/s |

decode 是 weight-bound（每 token 读全部 ~3GB 权重），B 份独立读取把带宽吃满 → 零增益。

## 2. 信天翁的并发基础：rows = B × T

信天翁 `faster3a_2605/rwkv7_fast_v3a.py` 的核心抽象：

```python
def select_path(B: int, T: int) -> PathConfig:
    rows = B*T          # ← B 折叠进 T，两者完全对称
    use_batched_rkv = (rows == 1) or (4 <= rows <= 64)
    cmix_mode = ...按 rows 选 kernel 变体（rows==1 / rows==2 / rows<=64 sparse / dense）
```

全 forward 是 (B, T, C) 张量流，**B 和 T 无差别**。README 性能表印证：
- B1024T1（1024 路 decode）≈ 15000 tok/s
- B1T1024（单路 prefill）≈ 17000 tok/s
两者几乎相同——**batch decode 就是 M=B 的短 prefill**。

三个关键机制：

### 2.1 线性层 = GEMM M=rows（权重读一次，被 M 行摊销）

```python
# tmix() 中 r/k/v 投影：
flat = torch.stack((xr.reshape(-1, C), xk.reshape(-1, C), xv.reshape(-1, C)))  # (3, B*T, C)
rkv = torch.bmm(flat, z[p+"rkv.weight"])                                       # (3, B*T, C)
r, k, v = [t.view(B, T, C) for t in rkv.unbind(0)]
```

(B,T,C) 摊平成 (B*T, C) 后一次 GEMM。权重 tile 载入寄存器/L1 后对多行复用，
**权重只从显存读一遍**，算力与带宽按 M 摊销。

### 2.2 元素级 kernel 把 B×T 展平成 1D（权重 [C] 只按 c 索引）

```cuda
// tmix_mix6_kernel：
const int64_t bt = pair_idx / c_pairs;
const int b = static_cast<int>(bt / T);
const int t = static_cast<int>(bt - b * T);
// x_r..x_g（lerp 系数，[C] 共享权重）只按 c 索引读，无 slot 偏移
```

### 2.3 WKV 状态更新按 (B, H) 网格，每 slot 独立 state 段

```python
torch.ops.rwkv7_wkv_fp16_v2.wkv_seq(B, T, C, H, wkv_state[B,...], ...)
```

与本实现已有的 `fuse_ka_dplr_norm_batch`（grid=(H, batch)）同构。

## 3. 本实现的差距与改造方案

| 类别 | 信天翁 | 本实现第一版 batch | 本实现改造后 |
|---|---|---|---|
| 元素级 kernel | rows 展平，权重 [C] 共享 | ✅ 已同构（norm_lerp6_batch 等） | 不变 |
| 状态更新 | (B,H) 网格 | ✅ 已同构（fuse_ka_dplr_norm_batch） | 不变 |
| **线性层** | GEMM M=B×T，权重读 1 次 | ❌ gemv grid.y=B，权重读 B 次 | **GEMM M=B** |

### 改造要点

1. **batch decode 线性层切 GEMM M=B**（M 维 = batch）：
   - rkv_stage1（r/k/v + mid 7 投影融合 kernel）→ 拆为 GEMM 调用或融合 GEMM
   - att.output / ffn.key / ffn.value / head → GEMM（op: plain/add/relu2）
   - 权重读一次，B 行摊销带宽 → weight-bound 下 decode 吞吐 ≈ ×B
2. **int8 权重与 GEMM**：GEMM 走 fp16（cuBLAS GemmEx / 自研 GEMM kernel）。
   int8 模型加载时保留/重建 fp16 线性权重副本（prefill 已是"反量化→fp16→GEMM"
   模式，复用同一份）。权衡：fp16 权重字节 ×2 但只读 1 次 vs int8 读 B 次——
   B≥2 时 fp16 GEMM 即胜出（2 < B）。
3. **布局**：激活 [batch, C]（slot 主序）即 GEMM 的 (M=B, K=C) 行主序输入，
   现有 batch 工作缓冲无需改。

### 预期

B=8：线性层带宽 8×（int8 gemv×8）→ 1×（fp16 GEMM 读一次，2 字节）≈ **4× 线性层带宽削减**；
配合已就绪的共享权重元素级 kernel，decode 吞吐预期接近 ×B（上限受非权重部分与 M=16
小 GEMM 利用率约束）。

---

## 4. 实施记录（2026-08-22 下午）：kernel 内权重复用（mb 版）

### 4.1 实际方案（比第 3 节 GEMM 方案更优）

调研 prefill GEMM 路径后发现两条硬约束：
1. `gemm` 契约要求 fp16 输入 + M/N 为 TILE(256) 倍数——batch B≤16 需 pad 到 256，激活还需 to_f16 转换；
2. int8 模型 GEMM 需每 token `dequant_int8_to_f16`（prefill 按 T 摊销，decode T=1 无法摊销，
   反量化本身 = 读 int8 + 写 fp16 ≈ 3× 权重字节，比 gemv 更贵）。

**最终方案：gemv kernel 内 batch 循环（"mb" 版）**——每 block 读一次权重（int8 原生，无需
fp16 副本），寄存器累加器复用给 BGRP 个 slot：

- `GEMV_VARIANT_MB_SRC`（BGRP=8）：覆盖 att_output(mul_add) / ffn_key(relu2) /
  ffn_value 稠密回退(add) / head(plain)——经 `gemv_variant_dispatch` 自动路由
  （batch>1 → mb，batch==1 → 原版），**上层调用零改动**。
- `GEMV_INT8_RKV_STAGE1_BATCH_SRC`（BGRP=4）：r/k/v 三个 C×C int8 投影 +
  4 个 mid 投影，主循环读 idx 一次 + 反量化一次 → 逐 slot `__hfma2`；mid 分支权重行
  读一次逐 slot 累加。
- 保持不变（权重 [C] 小或已共享）：norm_lerp6_batch / cmix_norm_lerp_batch /
  fuse_ka_dplr_norm_batch（元素级，权重 [C] 共享）/ chain4（fp32 [C,mid] 仅 ~2.3MB/层
  <5% 带宽）/ ffn_value_sparse（稀疏读 ~4% 列）/ sample_batch / record_tokens。

### 4.2 重大调试事故：SEQ_MODE 空字符串污染（记录避免重蹈）

**症状**：mb 改造后 B 扫描"零增益"，且 kernel 日志显示走的全是单序列 kernel。

**根因链**：
1. 早期调试设过 `$env:SEQ_MODE='1'`，结束时 `Remove-Item Env:SEQ_MODE` 被
   PowerShell 的 safe_rm alias 劫持失败（只在最后报了个无害错）；
2. 后续 `$env:SEQ_MODE = $null` **不删除变量，而是设为空字符串**；
3. RunCommand 工具的宿主会话向每条新命令注入环境快照——`SEQ_MODE=""` 持续存在；
4. Rust `std::env::var("SEQ_MODE").is_ok()` 对空字符串返回 **true** →
   batch_summarize 的调试分支判定"走单序列"。

**近 2 小时的全部 B 扫描数据（11:01~12:16）实为单序列路径，结论全部作废。**

**修复**：① 分支判定改 `is_ok_and(|v| !v.is_empty())`（SEQ_MODE/SKIP_GRAPH 双处）；
② 测试前用 `[Environment]::GetEnvironmentVariable('SEQ_MODE','Process')` 显式验证为 null；
③ 教训：环境开关必须检查非空；PowerShell 删环境变量用
`Remove-Item Env:X -Force`（绕过 safe_rm）并回读验证。

### 4.3 实测结果（RTX 2080 Ti，rwkv7-g1i-2.9b int8，温度≈0 确定性解码）

小样本（16 场景 / 32 摘要）真 batch 路径 B 扫描：

| batch | decode | 相对 B=1(batch API) |
|---|---|---|
| 1 | 67.6s | 1× |
| 4 | 36.0s | 1.88× |
| 8 | 29.9s | 2.26× |
| 16 | **27.6s** | **2.45×** |

全量（105 场景 / 232 摘要，B=16）：**0.91 摘要/s**（单序列基线 0.74，**+23%**）；
decode 203s vs 单序列 ~263s（**1.3×**；全量低于小样本因场景链滚动使队列常不满 16）。

**数值一致性**：B=1/8/16 输出 32 条中 30 条完全一致；2 条差异源于 fp16 舍入在
batch 分组边界（BGRP）不同执行路径下的 ±1 ulp 分叉经滚动摘要链放大——与 batch
逻辑无关（温度≈0 argmax 下 1 token 分叉即改写整条摘要）。

### 4.4 剩余瓶颈（后续可做）

1. **B=1 走 batch API 反而慢 2×**（67.6 vs 单序列 34.4s）：mb kernel 的 BGRP 寄存器
   循环在 bcnt=1 时空转 + batch 版元素级 kernel 开销。生产上 B=1 应回退单序列路径
   （batch_summarize 已天然支持：batch=1 时 dispatch 走原版 gemv，但元素级仍走
   batch 版——可在 submit 层按 batch==1 切单序列 API）。
2. **prefill 串行 51.4s**（232 次单序列 prefill）：占全量总时长 20%。prefill 本身是
   GEMM 路径，B 路已有 M 维，可直接 batch 化（信天翁 B32T32 批 prefill）。
3. **chain4 / ffn_value_sparse / sample_batch 未权重复用**：合计 <10% 带宽，
   收益有限。
4. **BGRP 调参**：rkv BGRP=4（寄存器 48 half2）与 variant BGRP=8（32 half2）为保守
   值；B=16 时 rkv 读 4 次权重，可试 BGRP=8（96 half2 寄存器，occupancy 降至 ~25%）
   换带宽。

---

## 5. 第二轮优化（2026-08-22 下午）：kernel 热点修复 + batch prefill

### 5.1 per-kernel profiling 定位真瓶颈

PROF_CUDA_KERNEL=1 + SKIP_GRAPH=1（B=8，decode 段）：

| kernel | 占比 | 根因 |
|---|---|---|
| gemv_variant_mb | 43.5%（avg 0.45ms） | BGRP=8 寄存器溢出（half2[4][8]+float[4][8]）→ spill local memory |
| gemv_int8_rkv_stage1_batch | 22.8%（avg 0.48ms） | 同上（3 矩阵×half2[4][4]=96 寄存器累加器） |
| gemv_lowrank_chain4_batch | 14.9%（avg 0.31ms） | grid=(M,batch) 每 block 仅 512B 功重，6 次 syncthrees 主导 |
| ffn_value_sparse_add_batch | 6.3% | — |
| fuse_ka_dplr_norm | 4.0% | — |
| rwkv_sample_batch | 3.2%（avg 2.1ms） | 单 block/112 线程扫 65536 logits（后续可优化） |

### 5.2 修复

1. **gemv_variant_mb**：BGRP 8→4 + `__launch_bounds__(128,4)`。
2. **gemv_int8_rkv_stage1_batch**：BGRP 4→2 + `__launch_bounds__(128,4)`。
3. **gemv_lowrank_chain4_batch 重写为 warp-per-row**：每 block 8 warp 各算 1 行
   ×BGRP=4 slot（grid=(M/8, ceil(B/4))），warp shuffle 归约替代 block 树归约。
4. **B=1 回退不再需要**：修复后 B=1 batch API 32.5s ≈ 单序列 34.4s（不再慢 2×）。

### 5.3 batch prefill（信天翁 rows 模型落地）

B 个 prompt（可变长，pad 到 T_pad）一次贯穿全部层，GEMM 的 M 维 = B×T_pad，
直接更新 batch State。新增 3 个 kernel：

- `seq_shift_batch`：token shift，slot 边界 t=0 读该 slot 的 tmix_x 段。
- `copy_token_batch`：每 slot 把 ln1/ln2 的 lens[b]-1 行写回 state[b]。
- `dplr_seq_batch`：DPLR 状态更新，s 为 [batch,H,N*N]（batch State 布局），
  **lens[b] 截断 padding 段**（pad token 不进 state）。

其余 kernel（norm/GEMM/elementwise）t 参数直接换 B*T_pad，布局天然兼容。

**两个 bug（记录避免重蹈）**：
1. **漏重置状态**：旧流程每轮 `state_load(initial_state)` 零态起步；batch 版最初
   没重置 batch_state → 第 2 轮起带上轮 decode 状态 prefill → 状态污染 → 采样出
   越界 token → gather OOB（cuGraphLaunch 700）。修复：调用方每轮 `reset_state_of`。
2. **dplr_seq_batch 传参错误**：单序列版 dplr_seq 的 k/a 参数是 **fuse_ka 的融合
   输出 k_mod/kk_l2**，batch 版最初误传原始 sb.k/sb.a → 数学全错。这类"融合链
   下游吃融合输出"的契约在复制层代码时最容易踩。

### 5.4 实测（RTX 2080 Ti，rwkv7-g1i-2.9b int8）

小样本（16 场景/32 摘要）：

| batch | prefill | decode | 吞吐 |
|---|---|---|---|
| 1 | 5.7s | 31.2s | 0.87/s |
| 8 | 3.8s | 22.5s | 1.21/s |
| 16 | **3.9s** | **21.3s** | **1.27/s** |

全量（105 场景/232 摘要，B=16）：**1.21 摘要/s**（单序列基线 0.74 → **+64%**）；
prefill 51.4→25.8s（2.0×）、decode 203→166s（1.22×）。

数值一致性：batch prefill vs 逐 slot prefill 30/32 完全一致（2 条 fp16 舍入分叉，
与 decode 批间差异同水平）。

### 5.5 剩余优化空间

1. `rwkv_sample_batch`（3.2%，avg 2.1ms）：单 block 扫 vocab=65536×B，可改多 block
   归约或 warp 级 top-K。
2. `ffn_value_sparse_add_batch`（6.3%）：稀疏读已按 slot 共享 tile，进一步可把
   blockIdx.z 换 BGRP 循环。
3. prefill 仍 25.8s：GEMM M=B×T_pad 但 T_pad 取批内 max（prompt 长度不齐浪费），
   可按长度分桶分批。
4. decode graph replay 166s：B=16 时 SM 已饱和（40×16=640 block fuse + 320×8 rkv），
   进一步需 kernel 融合（如 rkv+chain4 合并）。

