# decode 两跳路径 dispatch 错位修复 + chain4 深度融合回退记录

日期：2026-08-09。硬件：RTX 2080 Ti（68 SM，subgroup_size=32）。模型：rwkv-g1h-3B（fp16 / int8）。

## 一、两个 dispatch/grid 错位 bug（decode 路径全毁）

本轮做 chain4+DPLR 深度融合时，发现 Vulkan decode（单 token）路径输出与 CPU/CUDA 完全不符，
逐层二分后定位为**两处 shader 行映射常量与 runtime dispatch 网格不一致**——同一类 bug：

| # | 文件 | shader 侧 | runtime 侧 | 后果 |
|---|------|-----------|------------|------|
| 1 | `gemv_rkv_stage1.comp` | `ROWS=2`（被误改） | `GEMV_ROWS=4` → dispatch C/4=640 wg | 每 wg 算 2 行 → 仅覆盖 1280/2560 行，r/k/v 后半脏数据 |
| 2 | `gemv_lowrank_chain4` dispatch | `row = gl_WorkGroupID.x`（1 行/wg） | 按"subgroup-per-row 重构"意图 dispatch M/8=320 wg | 仅前 320 行有效，w/a/v/g 其余 2240 行脏数据 |

- bug 1 修复：`gemv_rkv_stage1.comp` ROWS 改回 4（int8 版 `gemv_int8_rkv_stage1.comp` ROWS=4 未受影响）。
- bug 2 修复：[runtime.rs](file:///c:/work/niceui/rwkv-rsv/src/runtime.rs) `gemv_lowrank_chain4` dispatch 改为 `(M, 1, 1)`，
  与 shader 的 `row = gl_WorkGroupID.x` 对齐。
- 这两个 bug 就是上一阶段"Vulkan lowrank_chain4 输出与 CUDA 不一致"调查的真正根源
  （当时怀疑 adaptive ROWS 数值，实际是 grid 错位）。

### 定位方法（可复用）

1. **跨后端对照**：CUDA 后端走 trait 默认回退（两次 dispatch）输出正确 → 问题限定在 Vulkan 侧。
2. **环境变量二分**：曾在 `gpu_model.rs` 加 `NO_FUSE_CHAIN4=1` 开关切换 融合/两跳 路径，
   发现两条路径错误形态不同 → 各自独立定位（已随回退移除该开关）。
3. **正确性基准**：`BACKEND=vulkan ARGMAX_VERIFY=1 SKIP_CPU=1 cargo run --release`，
   通过标准：`SELFLOOP_VERIFY` 8 token 序列 == `[47, 11, 46, 20996, 45175, 4600, 59, 327]`，
   且 gpu top10 前三 == cpu top10 前三 `[37480, 45, 39990]`。

### 教训

- **shader 行映射常量与 runtime dispatch 网格必须同源**：ROWS 这类常量改动必须双边同步，
  改一边即产生静默错误（不崩、不报错，只是结果错）。runtime.rs `GEMV_ROWS` 注释本就警告过这一点。
- bug 状态下测得的速度**虚高不可信**（少算 87.5% 的行）：两跳路径曾测出 fp16 65.4 / int8 87.6 tok/s，
  修复后真实值为 60.4 / 81.2。涉及 grid/行数的改动后，先验证正确性再谈性能。

## 二、chain4+DPLR 深度融合尝试（已回退删除）

尝试把 `gemv_lowrank_chain4` + `fuse_ka_dplr_norm` 深度融合为单次 dispatch
（`fuse_chain4_dplr_norm`，省 1 次 dispatch/层 × 32 层）。kernel 本身正确性验证通过
（与两跳路径、CUDA 输出序列完全一致）。

### 公平 A/B（双 bug 修复后的正确基线，200 token self-loop 同会话近热态）

| 路径 | 两跳（修复后） | 深度融合 | 结论 |
|------|---------------|----------|------|
| fp16 | 60.4 tok/s | 60.4 tok/s | 打平 |
| int8 | 81.2 tok/s | 76.9 tok/s | 融合 -5% |

### 劣化原因

融合 kernel 受 DPLR 结构约束只能 dispatch (H=40, 1, 1)；chain4 阶段并行度从
2560 workgroup（1 行/wg）塌缩进 40 workgroup，68 SM 只用 40 个，
chain4 权重读取（w2/a2/v2/g2 共 5.9MB/层）的 DRAM 延迟无法被足够并发线程掩盖。
省 1 次 dispatch 的收益（约几 µs）抵不上 chain4 阶段变慢。int8 路径整体更快、
单次 dispatch 相对开销更小，故净亏更明显。

**决策**：回退并完整删除融合实现（shader / runtime 方法 / trait 方法+覆盖 / build.rs 条目 / 调用点），
保持两跳路径。gpu_model.rs 调用点留有注释说明本次尝试结论，避免重复尝试。

## 三、head.weight int8 反量化回退（model.rs）

int8 模型删掉了 `head.weight` fp16 原键，CPU `Model::from_safetensors` 直接报错
`TensorNotFound("head.weight")`。修复：head 改走已有 `linear_to_f32` 路由
（原键 > `{key}.int8_idx/.int8_sz` 反量化），与各层权重一致。
至此 int8 模型可完整跑 CPU 参考 + GPU 对照验证。

## 四、最终状态

- 正确性（fp16 & int8，Vulkan 两跳路径）：ARGMAX_VERIFY / SELFLOOP_VERIFY 全过，
  self-loop 序列与 CUDA 逐 token 一致；`forward_seq` vs CPU fp32 max_abs_diff ≈ 0.10~0.15。
- 速度（Vulkan self-loop 200 tok）：**fp16 60.4 tok/s，int8 81.2 tok/s**。
- `cargo fmt` / `clippy --release --all-targets` 零警告；`cargo test --release` 51 lib + 1 集成全过。

## 五、后续可选优化点

- `gemv_lowrank_chain4` 的 **subgroup-per-row 重构**（runtime 注释中的原意图）：
  每 workgroup 处理 `256/subgroup_size=8` 行、dispatch 320 wg，subgroup 内一次归约免 shared 往返。
  当前是 2560 wg × 128 线程、每线程仅 ~2.5 MAC 的过并行形态，有改进空间；
  但**必须同步修改 shader 行映射**（`row = flat_idx*8 + subgroup_id`）并保持 dispatch 一致——
  这正是本次 bug 2 的反面教材，改前先加一致性断言或注释双锚点。
- 深度融合方向若未来重启：需先把 DPLR 并行度从 H=40 workgroup 提升（如按 (h, 行块) 二维 dispatch），
  否则 chain4 阶段并行度塌缩问题依旧。
