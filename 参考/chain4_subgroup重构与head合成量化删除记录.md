# chain4 subgroup-per-row 重构 + head 合成量化路径删除记录

日期：2026-08-09。硬件：RTX 2080 Ti（68 SM，subgroup_size=32）。模型：rwkv-g1h-3B fp16。
前置：《head权重量化速度复测记录.md》《decode两跳dispatch错位修复与chain4融合回退记录.md》

## 一、删除 fp16 模型单独量化 head 的合成路径

按复测结论（fp16 模型单独量化 head 端到端收益 ≈1%，淹没于热噪声）执行：

- [gpu_model.rs](file:///c:/work/niceui/rwkv-rsv/src/gpu_model.rs)：删除 `HEAD_QUANT` 环境变量与
  `synth_int8` 加载时合成量化分支；`head_a8` 仅在模型自带 `head.weight.int8_idx`（全 int8 模型）时启用，
  否则走 fp16 `gemv_f16`。调用点留注释说明决策。
- [model.rs](file:///c:/work/niceui/rwkv-rsv/src/model.rs)：删除随之失用的 `quantize_int8`，
  反量化注释归位到 `dequant_int8`。
- head 量化只作为全 int8 模型的一部分使用。

## 二、gemv_lowrank_chain4 subgroup-per-row 重构

《decode两跳dispatch错位修复与chain4融合回退记录.md》第五节记录的后续优化点，本次实施。

### 改动

- [gemv_lowrank_chain4.comp](file:///c:/work/niceui/rwkv-rsv/assets/shaders/src/gemv_lowrank_chain4.comp)：
  1 行/wg（2560 wg × 256 线程、每线程 ~2.2 MAC、shared+barrier 两级归约）
  → **subgroup-per-row**（每 subgroup 1 行、每 wg 8 行 = 256/subgroup_size、320 wg、
  lane 跨步覆盖 K、subgroupAdd 一次归约、**无 shared 无 barrier**）。
  x 读 8 行共享（L1 命中），权重按 lane 连续合并访存。
- [runtime.rs](file:///c:/work/niceui/rwkv-rsv/src/runtime.rs)：`gemv_lowrank_chain4` dispatch
  `(M,1,1)` → `(M/rows_per_wg,1,1)`，`rows_per_wg = 256/subgroup_size`，附整除断言。
- **双锚点注释**：shader 行映射（`row = wg_id*NUM_SUBGROUPS + subgroup_id`）与 runtime dispatch
  两侧互注同源关系（dispatch/行映射单边改 = 静默错误的教训已两次应验）。

### 正确性（先验证后测速）

`BACKEND=vulkan ARGMAX_VERIFY=1 SKIP_CPU=1 cargo run --release`：
- SELFLOOP_VERIFY 8 token == [47, 11, 46, 20996, 45175, 4600, 59, 327] ✓
- gpu top10 前三 == [37480, 45, 39990] ✓；single-token diff 0.072、seq vs cpu 0.053（正常 fp16 范围）

### A/B 测速（热漂移下的方法学记录）

GPU 当日连续测试热节流严重（同一会话 e2e 22~81 tok/s 漂移），跨 run 直接比 tok/s 不可信。
方法：新旧 shader 各构独立 exe（spv 经 rust-embed 内嵌），同热窗口交替跑 selfloop 100 tok，
以**同 run 内未改动的参照 kernel（relu2/rkv_stage1）归一化**热状态。

| 轮次 | 参照 kernel 热态差 | chain4 avg（旧→新） | 归一化加速 |
|---|---|---|---|
| R3（参照差 <6%，最干净） | relu2 +2.8%，rkv +5.6% | 0.0612 → 0.0338 ms/层 | **≈1.8×** |
| R2 | rkv +28.8% | 0.1629 → 0.0891 | ≈1.4× |

同窗口 e2e（R3，参照 kernel 最接近）：72.4 → 81.3 tok/s（+12%）；
保守归属（chain4 独占节省 0.88ms/token ÷ 13.8ms/token）**≈ +6%**。

### 踩坑记录

- `git stash push -- <files>` 回退到的是**最后 commit**（S3 老版本），不是改动前的工作区状态
  （本项目大量近期工作未提交）→ 编译 E0599。双 exe A/B 应手动备份/恢复文件。
- PowerShell `Copy-Item` **保留源 mtime**：恢复备份后 cargo 认为源码比构建产物旧而跳过重建，
  需 `LastWriteTime = Get-Date` 触碰后再 build。

## 三、最终状态

- fmt / clippy --release --all-targets 零警告；cargo test --release 51 lib + 1 集成全过。
- fp16 Vulkan decode（GPU 已冷却）：**81.3 tok/s**（修复后基线 60.4 tok/s 系热态所测，本次同窗口
  旧 shader 72.4 tok/s，chain4 重构净贡献同窗口 +12%）。
- 临时文件（双 A/B exe、备份目录、临时日志）已全部清理。

## 四、gemv_f32io_add_mul subgroup-per-row 尝试与回退（负结果）

按"剩余候选"第一项实施 add_mul（att output，M=K=2560，带宽 ~60%）的 subgroup-per-row 重构
（同 chain4 模式：BLOCK=256/8 行每 wg、320 wg、无 shared/barrier），双 exe 同窗口 A/B 两轮、
relu2/rkv_stage1 参照归一化：

| 轮次 | add_mul/relu2 比值（旧） | add_mul/relu2 比值（新） | 结论 |
|---|---|---|---|
| R1 | 0.202 | 0.211 | 打平（±噪声） |
| R2 | 0.167 | 0.257 | 新版劣化 |

**两轮归一化均打平或略劣，已完整回退**（shader 恢复原 ROWS=4 块级归约形态并留注释，
runtime 恢复原 dispatch，正确性复验 SELFLOOP_VERIFY 通过）。

为什么 chain4 有效而 add_mul 无效：chain4 的 K=96~320 极小、旧形态 2560 wg 严重过并行
（每线程 ~2.2 MAC），重构消除过并行与 shared 往返收益大；add_mul K=2560 时旧形态
每线程 2.5 迭代 × 4 行独立累加器已有足够 ILP，subgroup-per-row 的单累加器 80 步 FMA 链
反而不占优。**subgroup-per-row 不是通用赢点，仅适用于小 K/过并行 kernel。**

另注： Aug-5 的 "add_mul 60% 带宽" 读数本身可能含热噪声成分（GEMV_ROWS 注释区已有
"热降频噪声曾导致 18% 带宽利用率误判" 的前科）；当日冷却态 fp16 decode 81.3 tok/s 下
add_mul 并非实际瓶颈。

## 五、剩余候选

- `gemv_rkv_stage1`：ROWS=4 后 ~65-80%，仍有一定空间（3 个 C×C 投影 + 4 个 mid 融合，
  结构比 add_mul 复杂，尝试前需冷态 profile 确认真实余量）。
- FFN relu2/value：~83% 已近 fp16 带宽峰值，不建议再投。
- **测速方法学纪律**：热漂移下同 run 参照 kernel 归一化是硬要求；e2e tok/s 仅在
  参照 kernel 差 <6% 的窗口内可比。
