# Albatross `faster3a_2607` ↔ rwkv-rsv 全流程精细对比

> 日期：2026-09-21。上游：BlinkDL/Albatross `main` 分支的 `faster3a_2607/` 目录
> （14 个文件，全部已抓取到本地 `%TEMP%\albatross2607\` 供复核）。
> 本地对照：`rwkv-rsv`（NVRTC 运行时编译内核）+ `mytown/model-server`。
> 口径：**只记录在源码里实际读到的内容**，带文件行号；未确认的明确标注。

---

## 0. 一句话结论

Albatross 的「快」不是靠某个神级 kernel，而是三件事的组合：

1. **线性层直接交给 cuBLAS（`cublasGemmEx` + tensor op）**——在 rows=8~2047 这一档
   （**正是我们的 B=8~16 区间**）它**没有**用手写内核；
2. **极致的算子融合**（LN+残差+mix、4 个 lowrank 投影、末层 indexed norm）；
3. **按 (C, rows, GPU 架构) 查表分派**，且调参门槛是「端到端 + 两种图捕获顺序都为正」。

而我们在线性层上走了**自研 int8 gemv + BGRP 槽循环**这条路——实测该内核在 B=8 时
有效带宽仅 ~97-125 GB/s、FMA 峰值占用仅 12%（见 [方向二前置分析](2026-09-21-批量线性层瓶颈定位-方向二前置分析.md)），
是当前 decode 的最大单点（消融移除收益 -1210ms）。

---

## 1. 权重与量化

| 维度 | Albatross 2607 | rwkv-rsv |
|---|---|---|
| 计算 dtype | **fp16**（`DTYPE = torch.float16`；`WKV_MODE = "fp16"` 默认，可选 `fp32io16`） | **int8 组量化**（`w[m,k] = scale[m,k/128]*idx[m,k] + zero[m,k/128]`，4 字节打包进 uint32）+ prefill 反量化 fp16 |
| 权重布局 | **orig `[N,K]` 与 transposed `[K,N]` 双份并存**：`LOWRANK_WEIGHT="both"`；`ORIG_LINEAR_GROUPS={att_c2c, ffn_key, head}` 走 orig，其余走 `.t` | 单一 int8 `[M,K]`；低秩另有 fp32（单 token 链）与 fp16（prefill）两套 |
| 低秩权重 | `att.w1/w2/a1/a2/g1/g2/v1/v2` 均双布局 | w1/w2/a1/a2/g1/g2/v1/v2 |
| 稀疏 FFN 权重 | `value.weight` 保持 `[F, C]`，SpMV 用「每线程固定 2 个 C 列、跨 F 扫行」布局（`fast_ops_fp16.cu:922-923`） | `value_tiled` 平铺 `[f_block][c_block][f_local][c_local]` |
| Embedding | `EMB_DEVICE="cpu"`（CPU 侧 gather）+ `PRECOMPUTE_EMB_LN0=True`（加载期预算整个词表的 LN0(embed)） | `emb_ln` 预算 LN0(embed) 在 GPU（同思路）+ `emb_ln_cpu` f32 镜像供 CPU 路径 |

## 2. 状态（State）

| 维度 | Albatross 2607 | rwkv-rsv |
|---|---|---|
| 张量组织 | **一整个大张量**：shift `[L,2,B,C]`、wkv `[L,B,H,N,N]`、elapsed `[B]` int32（`rwkv7_fast_v3a.py:2407-2411`） | `Vec<GpuState>` 逐层独立张量：`tmix_x[batch,C]`、`tmix_rnn[batch,H,N,N]`、`cmix_x[batch,C]`；另加 `v_first[batch,C]` |
| 精度 | wkv **fp16**（`fp32io16` 为可选高精度档） | `tmix_rnn` **fp32** |
| WKV ABI | **三套实现共享同一物理布局 `[B,H,K,V]`**，batch stride 直接复用 `C (=H*N)`（`wkv_fp32_v2.cu:46-47`、`wkv_fp16_v2.cu:125`、`deltalog:94-95`） | 每层独立、`[batch,H,N,N]` |
| dithering | `elapsed` int32 计数器每步 `advance_i32`（fp16 WKV 的数值抖动抑制，`rwkv7_fast_v3a.py:2699/2705` 注明 "IMPORTANT FOR WKV16 DITHERING"） | 无（状态用 fp32，无此问题；若将来上 fp16 状态需补） |
| 重置 | `zero_state(B)` 按 B 建，缓存键 `(B, WKV_MODE)` | `create_batch_state(slots)` 一次，`reset_state_of` 只 upload 零（张量不重建 → 图内指针稳定） |

## 3. 每 token 主循环（最大形态差异）

| 维度 | Albatross 2607 | rwkv-rsv |
|---|---|---|
| 单步形态 | **B×1，每 token 一次 replay**：`decode_x.copy_(tokens_to_x(next_tokens))` → `decode_graph.replay()`（`app4b.py:526-531`） | **段内自循环**：一次 submit 连跑 n=32 个 token，token 在 device 侧回灌（`record_tokens`），全程无 host 介入（`gpu_model.rs::submit_sample_selfloop_batch`） |
| 采样 | **host 侧纯 torch，图外**：`torch.topk(k=SAMPLER_TOP_K=500, sorted=True)` + softmax + cumsum + searchsorted + gather（`app4b.py:327-343`） | **图内 device 采样**（`rwkv_sample_batch`）+ record_tokens |
| 惩罚 | **host 侧**：`logits.sub_(occurrence_count, alpha=...)`（`app4b.py:537-540`），每 token 在图外做 | **图内**在 sample kernel 里做 rep/freq/presence（`counter` 直方图 + `powf(cnt, decay)`） |
| 停止 | host 侧 `finished[b]` 掩码，**图恒跑满 B**（`app4b.py:557-565`） | host 侧**分段收割**：每 32 token 出图做 stop 检查再续段 |
| 图生命周期 | 按 B 捕获一次 + 缓存（`state, x, graph, output` 一起缓存）；**捕获失败自动降级** `graph=None`（`app4b.py:356-369`） | 2026-09-21 已对齐（按 `(kind,batch,n)` 捕获一次 + 永久重放 + 失败降级） |

**解读**：Albatross 每 token 都有 host 往返（copy_ + replay + 采样 + 惩罚 ≈ 6-8 个 torch 算子），
但它靠**大 B（可达 320/1024）把 host 往返摊薄**。我们 B 只有 8-16，所以用「段内自循环」
把 host 往返消掉——**这个取舍方向是对的**，问题出在段内 kernel 本身不够快。

## 4. 线性层（差异最集中、且是我们最大瓶颈）

### 4.1 Albatross 的分派策略（按 rows 分档）

| 条件 | 走的实现 |
|---|---|
| `rows == 1` 且 `N % 64 == 0` | 手写 `linear_f16_m1_splitk`（split-K + fp32 partial + 独立 reduce；`v3a_ops.cu:142-215`） |
| `rows == 2` | `linear_orig_row2_exact_f16`（`use4` 白名单：线程 ∈ {64,128,256} × out_tile ∈ {1,2}） |
| `rows ≤ 16` | **`linear_orig_wmma16_f16`**：WMMA 张量核，`__launch_bounds__(32,8)`，**1 warp 算 16×16 tile**，grid=`(N/16, M/16)`（`v3a_ops.cu:745-794`） |
| **`rows = 8 ~ 2047`（我们的档）** | **cuBLAS**：`linear_f16_cuda` → `cublasGemmEx` + `CUBLAS_GEMM_DEFAULT_TENSOR_OP`；orig 布局 `linear_f16_orig_cuda` → 加 `CUBLAS_OP_T`（`v3a_ops.cu:3505/3550`） |
| 大 rows 特定 (C,rows) 门 | CUTLASS（`rows_cutlass_runtime_cuda`：`RowMajor,128,128,64,64,stages=3,SplitKSerial,SplitKSlices=5`）或 cuBLASLt（profile 出的 `workspace_mb` + `algo_index`） |

**关键**：他们**没有**为 rows=8~2047 写手写 GEMM，而是委托 cuBLAS。
自研内核只覆盖两个极端：`rows==1/2`（host 侧 `TORCH_CHECK` 强制精确匹配，落空即报错，
绝无静默回退——`v3a_ops.cu:3687-3749`）与「大批量特定形状」。

### 4.2 我们的实现

`gemv_variant_mb`（BGRP 槽循环，int8 原地反量化）+ `gemv_int8_rkv_stage1_batch`（r/k/v + mid 融合）
+ `gemv_lowrank_chain4_batch` + `gemv_variant`（batch==1 单流版）。

**实测（隔离计时，B=8）**：att_output `0.075ms`/97 GB/s、ffn_key `0.247ms`/107 GB/s；
B=1（单流内核）同权重可达 **333 GB/s**。FMA 峰值占用仅 **12%**。
三次几何实验（BGRP 4→8、ROWS 2/BGRP 8、blockDim 128→256）**均无改善**。

### 4.3 r/k/v 融合方式

- Albatross：Python 侧 `torch.stack((xr,xk,xv))` → **`torch.bmm`（batch=3）**，权重预合并成 `rkv.weight`
  （`rwkv7_fast_v3a.py:2828-2830`）。另有 `linear_wag_rank_in_f16` 用 **`blockIdx.z` 分派 w/a/g 三组**
  （同一 kernel 承载 3 个投影，避免 3 份指针参数 → 少 3 个编译条目）。
- 我们：`gemv_int8_rkv_stage1_batch` 在**一个内核内**融合 3 个 C×C int8 投影 + 4 个 mid 投影。

> 我们在「融合度」上其实**领先**；差距在单内核的访存效率与实现载体（int8 自研 vs cuBLAS fp16）。

## 5. 元素级算子与融合

| 维度 | Albatross 2607 | rwkv-rsv |
|---|---|---|
| LN + 残差 + tmix mix6 | **一个内核** `add_layer_norm_tmix_mix6_f16`（`v3a_ops.cu:2161`，gated `B==1 and T==1`） | `norm_lerp6_batch` + `cmix_norm_lerp_batch` 两个 |
| LN + 残差 + cmix mix | `add_layer_norm_cmix_mix_f16_*` + `add_ln_cmix_mix` | `cmix_norm_lerp_batch` |
| rank-in / rank-out | `linear_wagv_rank_in_f16`（一次算 w1/a1/g1/v1）、`linear_wagv_rank_out_f16`（算 w/a/g/v **并融合 v 残差门**），gated rows≤7 / ≤4 | `gemv_lowrank_chain4_batch` |
| 末层 norm | **`add_last_layer_norm_f16_indexed`**：只对 `last_indices` 指定的行做 norm（`rwkv7_fast_v3a.py:2689-2696`） | `norm`（全量 t 行）+ `copy_token`（取末行） |
| 稀疏 FFN | `cmix_sparse_spmv_relu_rows_*`：`__ballot_sync`+`__popc` 在 smem 压缩 nnz 名单，主循环只遍历有效列；TILE=128/512；`Accumulators` 1/2/4 模板拆依赖链；输出 `atomicAdd(__half2)` 前置 `zero_vec4`（`fast_ops_fp16.cu:892-925, 1176-1313`） | `ffn_value_sparse_add_batch`（平铺 + 按 f 片稀疏遍历） |
| 向量宽度 | 元素级固定 `__half2`（4B）；`zero_vec4`/`add_shift` 用 `int4`（16B） | 视内核而定（float4/half2） |
| cp.async | **仅 WKV token 向量预取用**（`wkv_fp16_v2.cu:69-197`）；**元素级刻意不用** | 无 |

## 6. CUDA graph 友好性

| 维度 | Albatross 2607 | rwkv-rsv |
|---|---|---|
| 形状→grid | 纯函数（B/T/C/F 直算 `dim3`） | 同（但 batch 几何需手工同步 ROWS/BGRP 常量——**本次已因不同步踩过一次坑**） |
| 捕获期分配 | `*_cfg_*` / `*_out_*` 变体把 tile/线程数/输出缓冲外提，捕获期零分配 | 全部预分配常驻缓冲（2026-09-21 已修） |
| 边界检查 | **设备端检查写 NaN**，而不是 host `TORCH_CHECK`（`v3a_ops.cu:2342` 注释 "Bounds are checked on device to preserve graph capture"） | host `cu_check!` 返回 Err → 图内不会走这条 |
| host 同步 | 全目录 grep 0 处 `cudaDeviceSynchronize` / `.item()` / `.cpu()` | 图路径 0 处（采样参数走 pinned 异步上传） |
| L2 persisting | `set_graph_persisting_window` 遍历 `cudaGraphGetNodes` 给 kernel 节点设 `AccessPolicyWindow`（DeltaLog 用） | 无 |
| grid.y 上限 | 断言 `B <= 65535` 保证 2D grid 可用（`v3a_ops.cpp:114,171`） | 无此约束 |

## 7. 服务端调度

| 维度 | Albatross `app4b.py` | mytown `model-server` |
|---|---|---|
| 批来源 | **单请求内大 B**：`MAX_BATCH_PREVIEWS=320`，同一 B 条 prompt 一起跑 | **多请求攒批**：3ms 窗口 + 归批签名，BATCH_SLOTS=8 |
| 上下文缓存 | `decode_cache[(B, WKV_MODE)] → (state, x, graph, output)` | `batch_state` 一次创建、常驻 |
| 并发模型 | Gradio `concurrency_limit=1`（一个请求内 B 并行） | 多 HTTP 请求 → GPU actor 线程攒批 |
| 新请求进入 | 复用同 B 的 ctx；B 变则重建 | 批 <4 走单流，≥4 走批量 |

**结构性差异**：Albatross 是「单请求 → 大 B」，天然摊薄所有 per-step 开销；
我们是「多请求 → 一个 B」，**吞吐上限被并发数绑住**。这解释了为什么他们 glamor 的
B1024T1 数字我们无法靠单点优化追上。

---

## 8. 结论：我们该抄什么（按性价比排序）

1. ★★★ **批量线性层改走 cuBLAS fp16（tensor op）** —— 这正是他们 rows=8~2047 档的做法。
   - 我们**已有通路**：prefill 已用 `cublasGemmEx`（`CUDA_R_16F` + `CUBLAS_COMPUTE_32F`，
     `backend_cuda.rs:1396-1418`），还带 int8→fp16 反量化（`dequant_int8_to_f16`）。
   - 做法：为批量 decode 用的线性权重（receptance/key/value/att_output/ffn_key/ffn_value/head）
     建**常驻 fp16 副本**（约 5.0 + 0.34 ≈ **5.4 GB**），batch 路径直接走 GEMM。
   - 预期收益：我们当前 ~97-125 GB/s 有效带宽 → cuBLAS 通常 300-500 GB/s；
     **即便 fp16 字节数 ×2 仍是净赚**（0.075ms → 估 0.03-0.04ms 量级）。
   - 代价与风险：显存 +5.4GB；需保留 int8 路径作回退（可用 A/B 开关对比）。
2. ★★ **末层 norm indexed**：只对需要的行做 LN（我们目前全量 norm + copy_token 取末行）。
3. ★★ **LN+残差+mix 融合**：把 `norm_lerp6_batch` / `cmix_norm_lerp_batch` 合并，
   减少每层串行步数（我们实测是延迟主导，减步数直接见效）。
4. ★ **设备端边界检查代替 host 断言**（图捕获友好）。
5. ★ **几何常量单一来源**：Albatross 把 rows 分档写进 `select_path`，我们的
   ROWS/BGRP 在 kernel 与 dispatcher 各写一份，**这次不同步直接导致了一次编译错误**。
6. ○ 稀疏 FFN 的 ballot+popc 压缩细节（我们思路一致，可逐条对比优化）。
7. ○ L2 persisting window（DeltaLog 场景，我们暂无对应需求）。
8. ○ `elapsed` dithering 计数器（仅当将来把状态降到 fp16 时才需要）。

---

## 9. 未确认项

- `set_graph_persisting_window` 的实现不在已抓取的 14 个文件里（应在未下载的源中）。
- `app4b.py` 中 `USE_CUDA_GRAPH=True` 时的图捕获顺序细节、以及 `WKV_MODE` 切换对图的影响，只读了主路径。
- Albatross 的 `faster3a_2607` 无 benchmark 数据表（README 只在根目录且未包含性能表）；
  本文所有「他们更快」的判断均基于**架构事实**（cuBLAS / 融合度 / 大 B），未做同机同模型实测。
- `cmix_sparse_down_relu_rows_t512_*` 系列的具体阈值函数（`cmix_t512_accumulators` 等）未逐行核对。