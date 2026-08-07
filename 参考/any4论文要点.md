# any4 论文要点（arXiv:2507.04610，Meta，ICML 2025）

> 官方仓库：https://github.com/facebookresearch/any4
> 本目录已存官方实现参考：`any4_official_quantize.py`、`any4_official_kmeans.py`（2026-08-06 下载自 main 分支）

## 1. 格式定义（论文 2.4.1 节）

- 每行 16 项 fp16/bf16 学习码本（LUT），4-bit 索引指向 LUT 项。
- 分组量化：每 g=128 个连续权重共享一组 fp16 scale + zero（不对称量化）。
- bit 预算（论文以 K=4096 为例）：4 + 0.25(scale/zero) + 0.0625(LUT) = **4.3125 bit/权重**。
  - 本项目 K=2560：LUT 开销 16×16bit/2560 = 0.1 bit → 合计 **4.35 bit/权重**。
- 默认跳过 lm_head（论文与官方 repo 均如此）。

## 2. 官方量化算法（`anyq_quantize_tensor` 默认参数路径）

1. **组归一化**（`group_q`，asymmetric / unsigned / zero_point=True）：
   - `scale = (max - min).clamp(min=1e-6) / 15`
   - `zero = min + scale * 8`（8 = 2^(4-1)）
   - `Wg = (W - min) / scale` ∈ [0, 15]
2. **per-row k-means**：对 Wg 的每一行（长度 K，1D）做 16 簇 Lloyd k-means
   - 官方默认：scikit KMeans，**k-means++ 初始化**，n_init=1，max_iter=300，tol=1e-4
   - LUT = 16 个质心（[0,15] 域），索引 = 最近质心
3. **反量化**：`w = (LUT[idx] - 8) * scale + zero`

**等价性说明**：官方 LUT∈[0,15]、scale=(max-min)/15 与本案 LUT∈[0,1]、scale=(max-min)、zero=min
数学上完全等价（差一个 1/15 常数因子），反量化简化为 `w = LUT[idx]*scale + min`，shader 少一次减法。

## 3. 与官方实现的差异（本案选择）

| 点 | 官方 | 本案 | 理由 |
|---|---|---|---|
| k-means 初始化 | k-means++（随机） | 行内 16 分位数（确定性） | 可复现；1D 问题分位数 init 接近最优；不达标可换 k-means++ |
| 迭代 | max_iter=300, tol=1e-4 | max 50, 漂移 <1e-5 早停 | 1D 收敛快，实测报告迭代数 |
| 校准加权 k-means | 论文最佳结果用 `sample_weight=calibrate, scale_sample_weight=True` | v1 不用（纯权重 k-means） | 论文基线（无校准）已优于 int4/fp4/nf4；留作精度不达标时的增强 |
| keep_outliers / bias_pow | 默认关 | 不用 | 同上 |
| GPU 实现 | tinygemm：tensor core mma + 寄存器内联 LUT | Vulkan compute GEMV + shared memory LUT | 本案 decode 单 token 是带宽受限 GEMV，无需 tensor core |

## 4. tinygemm 要点（论文 4.2，仅供后续 any4 GEMM 参考）

- 面向小 batch（1≤m≤16）GEMM/GEMV，activation 放 B 矩阵（1/8 tensor core 利用率优于 1/16）。
- 权重不经 shared memory，gmem 直读寄存器；反量化在寄存器内完成（16 项 LUT 用 4 级 2:1 mux）。
- 行主序、K 为最内维，K 须为 16/32 倍数。

## 5. 论文精度结论（Llama3.2-1B / Llama3-8B，WikiText-2 PPL）

- any4 在所有 4-bit 数值格式（int4/fp4/nf4）中精度最高。
- 1B 模型：FP16 9.76 → any4 10.63（nf4 10.99，int4 11.89）。
- 8B 模型：FP16 6.14 → any4 6.51（nf4 6.63，int4 6.87）。

## 6. 本案实现进展（2026-08-06）

### Phase B1 完成：ffn.key any4 GEMV 路径数值验证 PASS
- 链路：`tools/quantize_any4.py`（新增 `--suffixes` 过滤）→ `{key}.any4_{idx,lut,sz}` →
  `GpuLayer::load` 检测 any4 键（原 fp16 键被替换）→ GPU 存 idx/lut/sz，CPU `dequant_any4`
  重建 fp16 副本供 prefill GEMM；`model.rs` CPU 参考同样回退 any4 反量化（`linear_to_f32`）。
- 验证方法（单层 ffn.key 测试模型，量化器 53.5s/矩阵）：
  1. 权重级：cos=0.995886，rel=8.95%（int4 基线 10.03%），验收 PASS。
  2. 合成矩阵单元测试（M=8,K=256，ones/one-hot/伪随机 6 用例）：max_abs_diff ≤ 3e-8。
  3. 整网 DIAG 三方对比：seq/tok=0.134，tok/CPU=0.083，seq/CPU=0.154 —— 与原 fp16 模型
     基线（0.115/0.092/0.151）同量级。

### 教训：gemv_any4.comp 曾漏写 w7 的 `+ zr`（已修复）

- 现象：tok vs seq logits 差 1.02（基线 0.11），top5 重排。
- 定位手法：① 强制 tok 走 fp16（同一份反量化权重）→ 差异回落基线，证明 bug 在 any4
  shader 而非量化误差传播；② 合成矩阵 + one-hot 扫描，ki≡0 (mod 8) 全对、ki≡7 (mod 8)
  恒偏大 |zero|——直接锁定每 uint32 第 8 个权重（`ipack>>28` 那项）漏加 zero。
- 教训：GLSL 里为对齐写的多空格续行容易在编辑时丢尾部表达式；逐位 one-hot 用例
  （e0/e127/e128/e255）是定位打包/解包 bug 的最快手段，首轮就该跑。

### Phase B2 完成：gemv_any4_add.comp（MUL 0/1）+ ffn.value / att.output 接线 PASS

- 新增 `gemv_any4_add.comp`（残差累加融合：MUL=0 → ffn.value `y=acc+x@A`；MUL=1 →
  att.output `y=acc+(x·g)@A`，g 为 fp16 门控），runtime `gemv_any4_add` / `gemv_any4_mul_add`。
- att.output 的 fp32 DIAG 诊断副本也改由 any4 反量化重建（原键已删）；model.rs 的 `p2t`
  闭包同步走 `linear_to_f32` 回退。
- 验证（layer 0 量化 att.output + ffn.key + ffn.value 三矩阵）：权重级 avg_cos=0.9956
  rel=9.23%（int4 10.43%）PASS；整网 DIAG：seq/tok=0.157、tok/CPU=0.071、seq/CPU=0.172，
  **top5 三方完全一致** [47,45,46,96,59]。

### 隔离量化误差 vs 内核误差的方法论（重要）

- any4 文件中原 fp16 键被删除，seq(prefill GEMM) 与 CPU 参考都吃同一份反量化权重，
  两者对比**看不到量化误差**；量化误差的端到端影响要以「原 fp16 模型输出」为基准对比。
- tok(any4 shader) vs seq(fp16 反量化副本 GEMM) 才纯反映 shader 内核误差。
- 跨模型对比工具：main.rs `SAVE_LOGITS=path`（存 CPU fp32 logits f32 bin）/
  `COMPARE_LOGITS=path`（读入对比，输出 max_abs_diff/rmse/top10 重合度）。

### 全模型量化端到端验证（192 矩阵，2026-08-06 完成）PASS

**权重级**（`any4量化报告.md`，32 层 × 6 矩阵，k-means 5851s）：
- avg cos = 0.995718（≥0.995 ✓），avg rel = 9.18%（≤10.5% ✓，优于 int4 基线 10.38% ✓）
- 最差矩阵 blocks.18.att.value.weight cos=0.9947 —— 与 AUXStar FP8 项目的敏感度排序
  （att.key > att.value > att.receptance > att.output > ffn.*）一致。

**内核正确性**（any4 GPU vs any4 CPU dequant，同权重，纯内核误差）：
- single-token diff = 0.083（fp16 基线 0.092 同量级）；seq vs CPU = 0.068（基线 0.054）
- ARGMAX_VERIFY / SELFLOOP_VERIFY(N=64) 均 match=true；diag GEMM 内核误差 ~5e-6
- seq/tok/CPU 三方 top5 完全一致 [37480, 39990, 45, 46, 47]

**端到端量化误差**（any4 CPU vs fp16 CPU，prompt " Eiffel" 3 token）：
- logits max_abs_diff = 6.92，rmse = 1.26；**top1 一致**（37480），top10 集合重合 7/10
- greedy self-loop 生成首 token 即分叉（fp16 [47,11,46,...] vs any4 [9833,41,49,...]）
  —— 4-bit 固有水平，与 AUXStar 仓库 NVFP4（rel 8.8%）top-1 ~85% 一致；
  该 3B 蒸馏小模型对量化更敏感，论文 1B/8B 的 PPL 劣化幅度（+0.37~+0.87）可参考。

**teacher-forced Top-1 一致率**（AUXStar 风格指标，main.rs `TOP1_REF_SAVE`/`TOP1_REF_COMPARE`，
CPU fp32 基准，256 token）：**227/256 = 88.7%**，首次分叉在生成第 6 token。
横向对比（同指标）：AUXStar 纯 NVFP4（rel 8.8%）~85%、FP8（rel 0.2%）93.75%；
本案 any4（rel 9.18%）88.7% —— 以相近 rel 误差优于 NVFP4，与论文
「any4 精度高于 int4/fp4/nf4 所有 4-bit 格式」的结论一致。

**性能**（RTX 2080 Ti，64 token bench 去最长平均）：
- 逐 token decode：**63.0 → 105.4 tok/s（1.67×）**；argmax 路径 103.9 tok/s
- seq prefill：707.5 → 705.4 tok/s（不变，走反量化 fp16 副本 GEMM，符合设计）
- PROF_GPU 大权值流量：5.03GB → 1.48GB/token（29.5%，理论 27.2%）
- any4 kernel 带宽利用率 58~67%（fp16 版曾 82%）：字节数降后固定开销占比升高，
  进一步提速需 kernel 级优化（寄存器 LUT / 更大 ROWS），列为后续增强。

**显存**：any4 副本 1.37GB + prefill 用 fp16 反量化副本 5.03GB ≈ 6.4GB（2080 Ti 11GB 可承受）。

## 7. Phase D：校准加权 k-means（提升端到端精度，2026-08-06）

### 背景与可行性判断（重要）

- 当前权重级 rel≈9.18% 已达 **16 簇 Lloyd-Max 理论最优（~9.5%）**，故**任何 k-means 微调都无法再降权重级 rel/cos**。
- 校准加权的价值**只在端到端**：用校准激活均值当每输入通道重要性，牺牲次要通道换取重要通道误差压缩。
  因此 B 的验收必须是端到端（Top-1 一致率 / logits），**不能只看权重级 rel**（加权后权重级 rel 甚至可能上升）。

### 已实现（tools/quantize_any4.py）

- `kmeans16_rows(..., sample_weight=None)`：支持 per-column [K] 权重。实现与官方一致——
  **assign 不变（最近质心），仅 update 改为加权平均**（官方 `run_kmeans` 的 `np.average(weights=...)`）。
  bincount 加权：`sums += w*x`、`cnts += w`，质心 = sums/cnts。
- `build_calib_weight(calib, scale, group, K)`：校准激活 → 权重，实现论文 `scale_sample_weight=True`
  （反量化误差在输出域被组 scale 放大，故权重 = calib × 组scale，再归一化绝对值和=1）。
- CLI `--calib <npz>`：键=完整张量名→[K]（或 `__shared__`→[K] 全矩阵共用），启用校准加权。
- **格式零改动**：idx/lut/sz 契约、组大小、bit 预算（4.35 bit/权重）完全不变，Rust/shaders 无需改。

### 正确性验证（临时测试，已删）

- 加权质心 = 簇内加权平均（Lloyd 不动点）✓
- sample_weight=None 退化为普通平均 ✓
- build_calib_weight 形状/归一化/公式 ✓
- 加权不改变 idx/lut/sz 形状 ✓

### 待办（下一步）：校准激活采集

- 产出 `--calib` 的 npz 需一次模型前向，逐层记录收缩维 K 的 `mean|activation|`。
- 方案：新增 `tools/collect_calib.py`（或用 Rust 推理加 dump 模式）。复用现有任一推理路径即可。
- 采集后：量化 → 端到端 Top-1 一致率对比（校准 vs 无校准），达标才说明 B 有效。

> 若 B 端到端收益不明显（校准加权通常只贡献 ~1% Top-1），则转入 Phase A（Hadamard 旋转，
> 近无损 4-bit 的真正路线），B 的加权机制可叠加复用。

## 7.5 nnq 输出域 LUT 优化（逼近无损，2026-08-06 端到端验证 PASS）

### 原理
- 纯 k-means rel≈9.18% 已达 16 簇 Lloyd-Max 理论最优，**权重级无法再降**；逼近无损只剩端到端杠杆。
- nnq（Output-domain LUT optimization）：固定 k-means 索引，用真实激活样本在**输出域**最小化
  `||X@W^T - X@Ŵ^T||²`。逐行最小二乘闭式解（`tools/quantize_any4.py` `nnq_output_lut`），
  比 Adam 收敛点快 ~100×，全模型可行。以牺牲权重级精确度换取输出域精度（权重级必然退化）。

### 校准样本采集（main.rs `CALIB_SAMPLES`/`CALIB_N`，model.rs `CalibCollector`）
- 每 token 记录 6 类量化矩阵的输入激活（att.r/k/v/o + ffn.k/v），键=`blocks.{li}.{name}`，形状 [N,K]。
- 产出 `outputs/calib_samples.st`：192 矩阵，每层 [32, 2560]（32 token）。

### 全模型 nnq 量化（GPU k-means，1484s）
- 权重级 avg rel = 9.18% → **15.1%**（牺牲，符合预期；报告见 `any4量化报告_nnq.md`）。
- **输出域误差 avg 0.0661 → 0.0281（↓58%）**，如 att.output 0.049→0.0018（27×）——nnq 有效的直接证据。

### 端到端 Top-1 一致率（CPU fp32，256 token，同一 fp16 参考序列）
| 模型 | Top-1 一致率 | 首次分叉 |
|---|---|---|
| fp16 参考 | 100% | — |
| 纯 k-means any4 | 88.3% | 第 6 token |
| **nnq any4** | **93.4%（239/256）** | **第 35 token** |

**结论：nnq 端到端 +5.1 个百分点，首次分叉从第 6 推迟到第 35 token，逼近无损。**
横向：AUXStar NVFP4 ~85%、FP8 93.75% —— nnq any4 93.4% 已达 FP8 水平，而仍是 4.35 bit/权重。

> 教训：nnq 的验收**绝不能看权重级**（会误判 FAIL），必须看输出域误差 + 端到端 Top-1。
> 采集计数 bug（`count_token` 未接上，`full()` 恒 false）曾导致长 prompt 无界采集，已修复
> （token 循环末尾每 token 计一次，`CALIB_N=4` 实测恰好采 4 样本/层）。

### 7.5.1 校准 token 扩到 512（2026-08-06，胆子大一点）
- 校准样本从 32 → 512 token（`CALIB_N=512`，产出 `outputs/calib512.st`，每矩阵 [512, K]）。
- 校准采集是**单一确定性的 greedy 自回归轨迹**（固定 prompt `" Eiffel"` + top-1 采样），
  非数据集随机抽样；轨迹单一、激活高度自相关——多样性有限，但输出域 least-squares 仍受益。
- **必要改造：nnq least-squares 加 GPU(torch) 分支**（`nnq_output_lut` 收 `device`）。
  - 原因：CPU numpy 的 `einsum("nk,mk,mkj->nmj")` 在 512 token 下逐矩阵仍可行，但 192 矩阵累计
    极慢（1 小时未完成）；且三数组 einsum 的 torch 版会展开 `[N,M,K]` 中间张量，对 ffn.value
    (M=10240) 高达 **53GB 显存**（曾把 22G+46G 共享全部占满）。
  - 解法：**逐块(m, MCH=1024) + 预乘 P[m,k,:]=scale[m,k]*onehot(idx[m,k]) + 两数组 einsum
    `"nk,bkj->nbj"`**（torch 走 matmul，不展开 [N,M,K]）。峰值显存 **<1GB**，与 CPU 结果一致
    （LUT max|diff|=2e-6）。GPU k-means + GPU least-squares 全模型量化约 4 分钟。
- 端到端 Top-1（CPU fp32，256 token，同一 fp16 参考序列）：
  | 校准 token | Top-1 一致率 | 相对 32 token |
  |---|---|---|
  | 32 | 93.4%（239/256） | 基准 |
  | **512** | **94.5%（242/256）** | **+1.1 个百分点** |
- **结论：校准 token 多一点确实有效**（94.5%，逼近 FP8 的 93.75% 之上）。权重级 avg rel 受输出域
  优化影响升至 10.16%（预期，验收须看端到端而非权重级）。
- 教训：torch 多输入 einsum 会隐性展开大中间张量 → 大矩阵务必分块 + 减少 einsum 输入张量数；
  显存优先用 `torch.cuda.max_memory_allocated()` 量化验证，而非等到 OOM。

## 8.1 k-means GPU 后端（PyTorch/torch，2026-08-06）

**背景**：全模型量化中 k-means 是速度瓶颈（192 矩阵 CPU 共 5851s）。逐层垂直替换为 GPU 后端。

**实现**（`tools/quantize_any4.py`）：
- `kmeans16_rows_torch(X, iters, ...)`：PyTorch 逐行 1D Lloyd k-means，语义与 CPU 版逐条对齐——
  assign 用「排序质心→相邻中点边界→右移计数」；update 用 `index_add_`（GPU 无带权重 bincount）；
  空簇重初始化（`argmax|X-recon|`）+ 质心保序 + drift 早停，与 CPU 版一致。
- `kmeans16_rows(..., device)`：`device != "cpu"` 时透明切到 torch 后端，失败自动回退 CPU。
- CLI `--device auto|cuda|cpu`：auto 探测 `torch.cuda.is_available()`。
- 依赖：文件头 `# dependencies` 加 `torch`；配套 `tools/.venv-gpu`（Python 3.11 + `torch==2.6.0+cu124`
  从 `https://download.pytorch.org/whl/cu124` 安装，PyPI 默认 torch 是 CPU 版，必须显式 cu124 索引）。

**验证**（RTX 2080 Ti，M=4096,K=2560 合成高斯域，iters=50）：
- 提速：CPU 22.54s → GPU 1.23s，**18.3×**（全模型 k-means 5851s → 约 320s）。
- 一致性：idx 一致率 99.9955%（4096 行仅 47 行不同，均来自 GPU 空簇 reinit 的 `argmax` 平局
  取不同位点，属局部最优等价解，不影响质量）；C 最大差 1.36e-2（fp16 域，同量级）。
- 已确认 `torch.cuda.is_available()==True`，`NVIDIA GeForce RTX 2080 Ti`，CUDA 12.4。

> 教训：本机 `nvidia-smi` 不在 PATH，但 `C:\Windows\System32\nvcuda.dll` 存在即驱动可用；
> torch 是否支持 CUDA 只看 `torch.version.cuda` 与 `torch.cuda.is_available()`，与 PATH 无关。

### k-means GPU 后端显存优化（2026-08-06）

**问题**：初版峰值显存 946MB（输入本身仅 105MB），瓶颈在 assign 的 `X[:,None]>bounds` 广播
生成 `[mc,K,15]` 布尔张量（15× 放大）+ 整块 [M,K] 常驻显存。

**优化**（`tools/quantize_any4.py` `kmeans16_rows_torch`）：
1. assign 改用 `torch.searchsorted(bounds, X, right=True)`：1D 有序边界二分，数学等价于
   `(x>bounds).sum()`（right=True 返回 ≤ 边界计数），内存 O(mc·K) 而非 O(mc·K·15)。
2. X 按 chunk 分批 `.to(device)`，不再整块常驻。

**效果**（M=10240,K=2560）：峰值 946 → **195MB**（↓79%，接近输入本身）；耗时 1.5→0.8s；
idx 一致率 99.9935%，全模型量化结果与优化前逐位一致（avg_cos=0.995718 / rel=9.178%）。

**顺带修复预存 bug（加权路径）**：GPU 版 `counts` 原用全 1（未加权），与 CPU 的
`bincount(weights=Wflat)` 不一致 → 加权质心 = Σwx/n 而非 Σwx/Σw，加权一致率仅 8%（4096 行全异、
质心塌缩到 [0,1] 边界）。修复：加权时 `counts` 也用 `Wflat` 作权重（float32），未加权退化为计数。
修复后加权一致率 8% → **99.9923%**。全模型默认走未加权路径，故此前未暴露。

## 8. 精度增强路线 A：bias_pow + keep_outliers（2026-08-06 实测，未采纳）

两条官方 any4 增强已加进 `tools/quantize_any4.py`（`--bias-pow`、`--keep-outliers`，零格式改动）。
layer 0 三组配置实测（权重级 avg rel，越小越好）：

| 配置 | avg_cos | avg_rel | max_abs 最差 | 结论 |
|---|---|---|---|---|
| 基线（纯 k-means） | 0.995634 | **9.27%** | 0.0636 | 参照 |
| `--bias-pow 2` | 0.989000 | 14.87% | 0.0736 | 明显变差 |
| `--keep-outliers` | 0.995028 | 10.09% | 0.0806 | 变差 |

**结论：两法在 RWKV 上都以整体 MSE 换极值保真，权重级必然退化**（计划 §五 note 已预警）。
本模型离群值并不灾难（基线 max_abs 已 ≤0.064），16 簇 Lloyd-Max 已近最优，故不采纳 A。
key 教训：A 这类「牺牲 bulk 保极值」的增强，只有离群值真正拖累部署输出时才值得，
且须用端到端 Top-1 而非权重级验收；在本 3B 模型上无收益迹象。近无损的真正杠杆是
**C（nnq 输出域 LUT 梯度优化）**与 **D（Hadamard 旋转降内在离群值）**。

## 9. mock 推理：自由自回归跨分布精度验证（GEN_SIM）

> 目的：用**未参与校准**的真实多语言 prompt（`outputs/calib_prompts.bin`，30% 中/30% 英/40% 其他，
> 800 条，平均 43 token），对比 fp16 与 any4（`nnq512.st`）的**自由自回归**生成轨迹，验证 any4
> 是否"接近无损"。实现见 `src/main.rs` GEN_SIM 分支（`GEN_SIM_SAVE` 生成 fp16 参考序列，
> `GEN_SIM_COMPARE` 对比）；用 CPU `model::Model` 的 any4 反量化 forward（与 GPU 反量化一致）。

### 9.1 小规模实测（8 条 prompt × 8 gen token，2026-08-06）

| 指标 | 结果 | 含义 |
|---|---|---|
| teacher-forced 单步一致率 | **84.4%（54/64）** | 喂 fp16 参考 token，单步条件概率 argmax 与 fp16 一致的比例 |
| 自由自回归平均一致长度 | 3.75 / 8 token（首分叉区间 [0,4]） | 从同一 prompt 各自生成，avg 在 3.75 个 token 处首分叉 |
| 自由自回归完整序列一致 | **25.0%（2/8）** | 8 token 全生成轨迹完全一致的比例 |
| 自由自回归全程逐位重合 | 46.9%（30/64） | 分叉后含重收敛的逐位重合率 |

### 9.2 结论与解读

- **跨分布（未校准多语言集）下并非"接近无损"**：teacher-forced 单步 Top-1 降至 **84.4%**，
  远低于校准分布内报告的 94.5%（§7.5.1，greedy 单轨迹 " Eiffel" 校准/验证同分布）。自由自回归
  完整序列一致仅 **25%**，平均 4 个 token 内即分叉。这是任何低比特量化在**分布外**的普遍表现——
  校准集 predispose 了 LUT/scale 适配，代价是未见分布上的 argmax 稳定性下降。
- **probe 定性佐证**：同一 3-token probe，fp16 top-1 logit=4.44/prob=0.45，any4 top-1 logit=3.58/
  prob=0.67——any4 反量化压缩了 logit 尺度、抬高了 top-1 概率差，但 top-1 排序仍一致；分布外
  step 的排序则可能翻转（84.4% 的 10/64 翻转正源于此）。
- **口径还原**：早期日志曾用 `GEN_N`（默认 16）而非实际 `glen=8` 计算平均一致长度，导致 5.75/16
  高估；已修正为按每条实际 glen 累计（3.75/8）。详见 `src/main.rs` GEN_SIM_COMPARE 分支。
- **建议**：若需在多样化部署场景下"接近无损"，应对校准集引入多样 prompt + 采样解码（§7.5.2 曾
  试过但端到端反降，因采样分布与验证分布失配）；更稳妥是接受"校准分布内近无损、分布外有损"
  的事实，或针对目标分布重新校准。

### 9.3 多样大规模校准实验（2026-08-06 晚，nnq_multi4096）——分布外精度显著提升

> 结论：用 `calib_prompts.bin`（800 条多语言 prompt）以 **greedy** 采集 **4096 token** 重新跑 nnq
> 输出域 LUT 优化，生成 `outputs/nnq_multi4096.st`，在**同一** 32 条分布外多语言测试集上，分布外精度
> 相对 `nnq512.st`（512-token 单 greedy 轨迹校准）**全面提升**。

采集：`CALIB_SAMPLES=outputs\calib4096.st CALIB_N=4096 CALIB_PER_PROMPT=16`，256 条 prompt × 16
token greedy，CPU 前向约 58 分钟。量化：`uv run tools/quantize_any4.py --in rwkv-g1h-3B.st
--out outputs\nnq_multi4096.st --nnq-calib outputs\calib4096.st --device auto`，权重级 avg_cos=0.9951、
nnq 输出域误差逐矩阵改善（如 0.05096→0.04096）。

**同一测试集（32 条未校准多语言 prompt × 8 gen token，GEN_PROMPTS=32 GEN_N=8）对比：**

| 指标 | nnq512（基线） | nnq_multi4096 | 提升 |
|---|---|---|---|
| teacher-forced 单步一致率 | 87.9%（225/256） | **90.2%（231/256）** | +2.3pp |
| 自由自回归平均一致长度 | 3.72 / 8 | **4.62 / 8** | +0.90 |
| 自由自回归完整序列一致 | 34.4%（11/32） | **43.8%（14/32）** | +9.4pp |
| 自由自回归全程逐位重合 | 46.5%（119/256） | **57.8%（148/256）** | +11.3pp |

**解读**
- 多样 + 大规模校准把校准分布拓宽到与部署的多语言 prompt 更接近，LUT/scale 适配不再过度偏向单条
  greedy 轨迹，分布外 argmax 稳定性提升：teacher-forced Top-1 由 87.9%→90.2%，完整序列一致接近
  翻 1.3×。
- 注意 9.1（84.4%）与本节 87.9% 的 nnq512 数值差异来自**测试集不同**（9.1 为 8 条早序 prompt，
  本节为 32 条不同 prompt），口径不同，不可直接相减；**同测试集内 nnq512 vs nnq_multi4096 才是对照**。
- 仍非"无损"：分布外 teacher-forced 90.2% vs 分布内 94.5%（§7.5.1），说明校准集多样性提升有效但
  有限——绝对上界仍受 4-bit 信息容量约束。若需进一步逼近，可增大校准集到 1.5 万+ token 或引入
  多种采样解码（需与验证解码同分布）。
- 实现与产物：`outputs/calib4096.st`（12GB 校准激活，可复用重量化）、`outputs/nnq_multi4096.st`、
  `outputs/gen_sim_ref.bin`（fp16 参考参考序列）。
