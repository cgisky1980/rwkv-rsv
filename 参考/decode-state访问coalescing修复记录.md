# decode state 访问 coalescing 修复记录（fuse_ka_dplr_norm warp/subgroup-per-row 重写）

日期：2026-08-25
背景：Albatross 原版 large kernel 的 state 布局问题排查（用户提供的问题分析）。

## 一、问题来源与 rwkv-rsv 现状判定

Albatross 原版：warp lane `i` ↔ value 坐标，循环变量 `j` ↔ key 坐标；布局
`state[V=i][K=j]`（j 为 stride-1 内层）→ 固定 j 时相邻 lane 地址间隔 256B，
一条 warp 指令访问 32 个离散 sector（每 sector 仅用 4B，12.5% 利用率）。
Albatross patch 采用**转置布局** `state[K=j][V=i]` 使 value 维变为 stride-1。

rwkv-rsv 逐路径判定结论：

| 路径 | 映射 | 判定 |
|---|---|---|
| CUDA prefill `rwkv_dplr_seq(_batch)` | half-warp↔行、lane↔列（j0=subl, +16/+32/+48） | ✅ 已 coalesced（每 half-warp 64B×2 连续段），无需改 |
| CUDA decode `fuse_ka_dplr_norm` | `row = t/2`（value 每 2 lane 变一次）+ `ct = t%2` 隔列分片 | ⚠️ 同源问题的半强度版：warp 一条指令 = 16 行 × 8B 分散段，25% sector 利用率 |
| Vulkan decode `fuse_ka_dplr_norm.comp` | 同上映射 | ⚠️ 同样 25% |
| Vulkan 遗留 `dplr.comp`/`fuse_ka_dplr.comp` | invocation↔行、串行 j | 与 Albatross 原版同构（12.5%），但已不在 dispatch 路径（被 fuse_ka_dplr_norm 取代） |

## 二、方案选择：改线程映射（A）而非转置布局（B）

- **A 与 B 的 decode 性能上限数学上完全相同**：coalescing 本质是"lane 对应的坐标落在
  stride-1 维"。A 让 lane↔key（j 本来就是 stride-1）；B 转置后让 lane↔value。
  两者每 warp 指令都是 128B 连续、100% sector 利用率。
- B 会把 prefill 改坏：转置后现有 prefill 映射（half-warp↔行）地址变为
  `j*256+i*4`，lane 间隔 256B → 跌回 12.5%；救回需整行寄存器化（64 regs/线程），
  推翻现有 s0..s3 寄存器化结构。Albatross 选 B 是因为它所有 kernel 都是
  lane↔value 模式、且无 state 持久化包袱；rwkv-rsv 的 prefill 已是最优结构。
- B 另需全链路改 state 序列化（gpu_model back/load、model.rs CPU 参考），
  存量 state 文件不兼容，零性能回报。
- B 唯一理论增益：行内归约从 warp shuffle（A）变完全私有，每行每 token 省
  ~10 条 shuffle 指令——相对 decode 每 token 每层 MB 级 state 带宽是噪声；
  且 A 的 shuffle 归约本身就比旧版 shared+syncthreads 树归约更快。

结论（用户确认）：方案 A，只动 decode 两处 kernel，不动布局/不动 prefill/不动序列化。

## 三、实施内容

### 1. CUDA `FUSE_KA_DPLR_NORM_SRC`（backend_cuda.rs）

旧映射：`row = t/2, ct = t%2`（128 线程 = 64 行 × 2 列分片），Phase 2/3 的
state 访问每 2 lane 才 8B 连续；sa/y 行内归约走 shared + `__syncthreads` 树。

新映射（warp-per-row）：
- 128 线程 = 4 warp；warp 串行处理 `row = warp; row += 4` 的状态行（n=64 时每 warp 16 行）；
- lane ↔ 连续列 `j0 = lane, j1 = lane+32`：state 行访问每条 warp 指令 128B 连续
  （100% sector 利用率；旧版 25%）；
- S 行元素寄存器化（每 lane 2 个 float，s0/s1），**一遍读一遍写**（旧版读两遍）；
- sa/y 行内归约 = `__shfl_xor_sync` 蝶形（mask 16→1，全 lane 得全和），
  行循环内零 barrier（旧版每 phase 2 次全 block syncthreads）。

**数值锚点（刻意保持不变，勿"顺手修正"）**：
- Phase 0 L2 范数保持旧版冗余归约（每行 2 线程各算一次 → **2 倍和**），
  与 CPU 参考（backend_cuda.rs 测试 `sq_sum *= 2.0`）和 Vulkan 版三方锚定；
- 更新公式/Phase 4-6 树归约形状逐项保持；归约顺序变化仅 ulp 级（测试容差 1e-2）。

### 2. Vulkan `fuse_ka_dplr_norm.comp`（subgroup-per-row）

对标 CUDA 版，跨硬件自适应：
- `SUBGROUP_SIZE` 由 runtime 传入（constant_id=4，AMD=64/Intel 可变/NVIDIA=32），
  `NUM_SUBGROUPS = 128/SUBGROUP_SIZE` 做行分派模数；
- 列分派 `j = sgid + m*SUBGROUP_SIZE`（subgroup 内连续地址 → coalesced），
  S 寄存器化（`sreg[CPI_MAX=8]`，覆盖 SUBGROUP_SIZE≥8）；
- sa/y 归约 = `subgroupAdd`（subgroup 内隐式同步，行循环内零 barrier）；
- Phase 0 保持冗余树归约（2 倍和，与 CUDA/CPU 锚定）；Phase 1 改为 t<N 一人一列。

管线闭环：`create_kernel_unsafe` 用 `required_subgroup_size(设备原生值)` 固定
subgroup 大小，spec[4] 与 `gl_SubgroupID` 严格同源。

### 3. runtime.rs（双锚点）

`fuse_ka_dplr_norm` 的 spec 常量数组 4 项 → 5 项（追加 `subgroup_size`）。
⚠ spec 数组索引 = constant_id（app.rs `create_kernel_unsafe` 按索引生成
mapEntry）；漏传会使 shader 的 SUBGROUP_SIZE 静默回落默认 32，AMD 上行分派错乱。

## 四、验证

| 项 | 结果 |
|---|---|
| `cargo build --release` | 通过；glslangValidator 重编 spv 确认（spv mtime 晚于 comp，非回退） |
| `cargo test --release`（59 项全量） | 全过，含 `fuse_ka_dplr_norm_matches_cpu`（CUDA vs CPU）、`dplr_seq_matches_cpu`（prefill 无回归） |
| Vulkan 数值验证（临时集成测试，用后已删） | vs CPU 参考 max_diff：s=2.4e-7 / km=1.2e-7 / yn=7.2e-7（ulp 量级） |
| `cargo fmt --all` + `cargo clippy --all-targets -- -D warnings` | 零警告 |

## 五、实测压测数据（2026-08-25，RTX 2080 Ti，rwkv-g1h-3B，NTOKENS=256）

方法：`git stash` 切旧/新 kernel 各跑一轮 `cargo run --release --example benchmark`；
Vulkan 两轮均在 PROF_GPU=1 下（条件一致），CUDA 无 PROF_GPU。

### 端到端 tok/s

| 路径 | 后端 | 旧 kernel | 新 kernel | Δ |
|---|---|---|---|---|
| prefill（未改动，测噪声） | CUDA | 2251.3 | 1942.0 | **-14%（纯环境噪声标尺）** |
| decode 逐 token | Vulkan | 77.2 | 78.8 | +2.1% |
| decode 逐 token | CUDA | 96.3 | 99.4 | +3.2% |
| argmax_selfloop | Vulkan | 86.6 | 87.6 | +1.2% |
| argmax_selfloop | CUDA | 98.5 | 101.3 | +2.8% |
| sample_selfloop | Vulkan | 85.6 | 86.9 | +1.5% |
| sample_selfloop | CUDA | 84.7 | 87.1 | +2.8% |

### kernel 级（Vulkan PROF_GPU GPU 时间戳，252 次 dispatch 稳定值）

| 段 | 旧 kernel avg | 新 kernel avg |
|---|---|---|
| argmax_selfloop | **0.0175ms**（0.0174–0.0177 稳定） | **0.0204ms**（0.0203–0.0205 稳定） |

### decode GPU 时间分布（修复版 argmax_selfloop 段，SUM=81.0ms / 2048 kernels）

gemv_f32io_relu2（FFN 投影）32% > gemv_rkv_stage1（rkv 投影）28% > add_mul 10%
> lowrank_chain4 7% > **fuse_ka_dplr_norm 6.4%（5.2ms）** > 其余。

### 诚实结论（与预期收益的差距分析）

1. **端到端 +1~3%，方向为正但处于噪声带内**（prefill 代码未动却有 ±14% 波动，
   证明环境噪声远大于端到端差异）。
2. **Vulkan kernel 级实测慢 ~17%**（0.0204 vs 0.0175ms）：B=1 下每 kernel 仅碰
   ~640KB state，旧版 25% sector 利用率的 4× L1 事务被 L1/L2 带宽兜底（DRAM 流量
   本就最优）；而新版的串行行循环（每 subgroup 16 行）+ 每行 2 次 subgroupAdd 的
   归约开销反而成为延迟瓶颈——B=1 小 kernel 对行间并行度（延迟隐藏）比对带宽
   效率更敏感。kernel 仅占 decode 6.4%，端到端影响 ~1%。
3. CUDA 端到端 +3% 与 Vulkan kernel 变慢不矛盾：CUDA 版行内归约是 `__shfl_xor`
   蝶形（比 Vulkan subgroupAdd 编译产物更轻），且 CUDA 端到端噪声大。
4. **大 batch decode（fuse_ka_dplr_norm_batch）未测**——state 访问 ×B、L2 真实
   承压时 coalescing 收益才会显现，是本改动的理论受益场景。
5. 保留本改动的理由：数值已验证（ulp 级）、state 格式/序列化不变、prefill 不动、
   访问模式结构正确（100% coalesced + S 一遍读写）；端到端无回归。
   若后续 profile 显示 fuse_ka_dplr_norm 占比上升（大 batch/上下文增长），此结构
   是正确起点。
6. 若要继续优化 Vulkan kernel：subgroupAdd → 手动蝶形 shuffle；或折中映射
   （保留行间并行 + 列局部连续，如每 warp 8 行 × 4 lane 连续列段）——行并行
   保延迟、列连续保带宽，按 batch 大小切换。

## 七、后续路线验证：int8 量化模型实测（2026-08-25 同机同参数）

背景：PROF_GPU 显示 decode 瓶颈在权重投影 GEMV（60%，已 86-91% 峰值带宽），
state kernel 仅 6.4%——因此最大杠杆是降权重流量。int8 模型
（`rwkv-g1h-3B.int8.st`，仓库现成支持）与 fp16 同机对比：

| 路径 | 后端 | fp16 | int8 | Δ |
|---|---|---|---|---|
| prefill | Vulkan | 522.7 | 748.4 | +43% |
| prefill | CUDA | 1942.0 | 1812.4 | -7%（噪声带） |
| decode 逐 token | Vulkan | 78.8 | **116.5** | **+48%** |
| decode 逐 token | CUDA | 99.4 | **122.3** | **+23%** |
| argmax_selfloop | Vulkan | 87.6 | **120.7** | **+38%** |
| argmax_selfloop | CUDA | 101.3 | **124.4** | **+23%** |
| sample_selfloop | Vulkan | 86.9 | 117.3 | +35% |
| sample_selfloop | CUDA | 87.1 | 104.5 | +20% |

结论：int8 是当前 B=1 decode 最有效的单一提速手段（+23~48%），验证了
"瓶颈在 GEMV 带宽"的判断。注意 int8 精度损失需另行评估（参考/int8量化报告.md）。

## 八、遗留与不做

- prefill（`rwkv_dplr_seq(_batch)` / `dplr_seq.comp`）：已 coalesced，不动；
- Vulkan 遗留 `dplr.comp`/`fuse_ka_dplr.comp`：不在 dispatch 路径，不清理
  （与本次目标无关，避免扩大改动面）；
- 转置布局（方案 B）：明确否决，理由见第二节；若未来出现"state 常驻
  shared/L2 的深度 decode 融合"再评估。
