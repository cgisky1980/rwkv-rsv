# Vulkan prefill 单发 vs 稳态差异排查记录（kernel cache 机制）

日期：2026-08-25。起因：benchmark 显示 Vulkan prefill 522 tok/s vs CUDA 1942（差 3.7x），
与 decode 仅差 ~15% 不对称，遂深查。

## 一、结论（反转）

**522 是测量假象。稳态（同长度第二次 prefill 起）实测 2335-2614 tok/s，比 CUDA（1942）还快 1.2-1.35x。**

| 路径（256 tok, 2080 Ti, fp16） | benchmark 单发 | 稳态 |
|---|---|---|
| Vulkan prefill | 522（热机）/ ~750（冷机） | **2335-2614**（0.098-0.110s，3 rep × 2 进程稳定复现） |
| Vulkan prefill int8 | 748 | 1862-2257 |
| CUDA prefill | 1942 | ≈1942（无 churn 问题） |
| CUDA prefill int8 | 1812 | ≈1812 |

验证工具：`examples/prof_prefill_steady.rs`（预热一次后测第二次起）。

## 二、原因分层

### 1. 一次性 pipeline/descriptor 创建（主因，~300ms/首次）

runtime.rs `record_kernel` 的 cache key = (shader, spec, **params**)，params 含 buffer
地址；且 cache 命中时 uniform 是创建时烘焙的（不更新）。prefill 每层权重地址不同 →
每个 (kernel × layer) 组合都是独立 key → 首次 dispatch 走完整创建链
（shader module + descriptor set layout + descriptor set + pipeline layout + pipeline +
uniform），实测 ~140μs × 1722 kernels ≈ 240ms。证据：int8 单发 prefill GPU SUM
102.6ms vs wall 340ms（70% 缺口）。

benchmark 的 warmup 无法预热：warmup T=3（m_pad 桶小）≠ 计时 T=256（m_pad=256），
spec 不同 → key 全不同。decode 不受影响（M=1 spec 恒定，warmup 有效；且 token 间
地址稳定全命中）。

### 2. 驱动级管线编译缓存（次因，~90ms，跨进程后消失）

同为"稳态"的两个进程：第一个 190ms/rep，之后 100ms/rep。首次创建某 (shader, spec)
管线时驱动编译 SASS；重复运行后命中 NVIDIA GLCache。另注意：create_kernel_unsafe
每次新建 descriptor set layout / pipeline layout（不复用），即使管线编译命中驱动
缓存，layout 对象创建仍是每次 miss 的开销。

### 3. 热降频（小因素）

连续 benchmark 时 GPU 热态，单发数字再打 ~20-30% 折扣（冷机 int8 单发 752 vs 热机 748
——int8 单发被 host 开销主导故对温度不敏感；fp16 522 为热机值）。

## 三、稳态下的真实构成（fp16 256 tok，PROF_GPU）

GPU 88-92ms / wall ~100ms（**85-90% GPU busy**）：

| 项 | 时间 | 备注 |
|---|---|---|
| gemm_f16_f32(plain/affine/relu2/tanh/bias) | ~62ms | **有效算力 21-25 TFLOPS**（2080 Ti f32 累加 tensor 峰值 28.5 的 75-89%）——coopmat GEMM 效率很高，非瓶颈 |
| dplr_seq | 9-11ms | 低占用率（40 WG × 64 线程 + 每 token 2 次 barrier），带宽理想 ~1ms，**10x 浪费**——已知优化点，CUDA 版结构（320 block × 128 线程 + 无 barrier）可对齐 |
| norm/seq_shift/to_f16/杂项 | ~16ms | |
| wall-GPU gap | ~10-15ms | 1530 kernel 的记录/提交/67MB logits 下载 |

CUDA 132ms vs Vulkan 稳态 ~100ms：CUDA 的 prefill GEMM 是标量 kernel（backend_cuda
无 wmma），Vulkan 走 coopmat tensor core——**稳态下 Vulkan 反而快 30%**。

## 四、衍生发现（部署相关）

1. **int8 稳态 prefill（2257）< fp16 稳态（2614）**：int8 每次 prefill 要付 16.2ms
   dequant（6 层/每层 × 32 层，6.9GB 反量化流量）。与 decode 相反（int8 decode +48%）。
   即：**int8 提速 decode 但拖累 prefill ~14%**。可选优化：dequant 结果按层缓存驻留
   （1.9GB int8 + 3.4GB fp16 scratch 显存可容纳，22GB 卡）或 int8 原生 coopmat GEMM
   （2080 Ti 支持 16x16x32 U8 coopmat）。
2. 服务端真实场景介于单发与稳态之间：每个新 prompt 长度 → 不同 m_pad 桶 → 该桶首次
   付 pipeline 创建费（~几十-三百 ms），同桶复用后恢复稳态。桶数量有限，长跑服务
   全部预热后无影响。

## 五、修复方向（方向 1 已于同日实施，实测见第七节）

1. **kernel cache 解耦 uniform 与 pipeline**：key 去掉 params，pipeline 按 (shader,
   spec) 缓存；uniform/descriptor 每 dispatch 更新（push descriptor 或 dynamic
   offset）。消除首次 churn + 大幅缩减 cache 条目数（1722 → ~20）。这是根治。
2. m_pad 桶化 + 启动预热（治标，配合 1 可不做）。
3. dplr_seq Vulkan 版重写对齐 CUDA 结构（稳态下再省 ~10ms，~10%）。

## 六、教训

- benchmark 单发计时不等于稳态性能，尤其当 runtime 存在按地址索引的惰性创建缓存——
  **对比后端时必须先跑同形状预热**（prof_prefill_steady 模式）。
- wall 与 GPU 时间戳（PROF_GPU SUM）对不上时，先怀疑 host 侧一次性创建，再怀疑热降频。

## 七、修复实施记录（2026-08-25 同日，方向 1 落地）

### 方案：uniform 池 + UNIFORM_BUFFER_DYNAMIC

Vulkan 惯用法（dynamic offset uniform），不引入 push_descriptor 扩展依赖：

- **Runtime 持有一个大 mapped uniform 池**（默认 8MB，`UNIFORM_POOL_MB` 可调；
  host-visible + host-cached，写入即 memcpy）
- **KernelKey 去掉 params（地址）**，改为 `(shader, spec, n_params)`——同形状不同层
  的 kernel 共享一个 pipeline；cache 条目 1722 → ~30
- **descriptor 类型 UNIFORM_BUFFER → UNIFORM_BUFFER_DYNAMIC**：kernel 的 descriptor
  绑定整个池 + 定长 range（= Params 字节数），偏移由 `cmd_bind` 的 dynamic offsets
  提供（该方法本来就支持 offsets 参数）
- **每次 dispatch**：bump 分配池 slot（按 min_uniform_buffer_offset_alignment=256
  对齐）→ `copy_from_at(params, offset)` 写 slot → `cmd_bind(cmd, &[offset])`
- **批内安全**：每 dispatch 独立 slot，顺序录制不互相覆盖；**跨批安全**：
  begin/end_batch 重置游标（end_batch 已 queue_wait_idle，GPU 空闲）
- App 层新增：`Uniform::copy_from_at`（带偏移拷贝+越界检查）、
  `Binder::bind_uniform_with_range`（DYNAMIC 绑定）；descriptor pool 补
  UNIFORM_BUFFER_DYNAMIC 类型配额（原 pool 只计 UNIFORM_BUFFER，会分配失败——
  实施中发现的隐藏坑）
- NO_CACHE 诊断模式仍走池（只是不查 cache）

### 改动文件

- `src/runtime.rs`：KernelKey / Runtime 池字段 / record_kernel 重写 / batch 边界重置
- `src/vulkan/app.rs`：copy_from_at + bind_uniform_with_range + descriptor pool 配额

### 实测（RTX 2080 Ti，256 tok，同 benchmark 方法）

| 指标 | 修复前 | 修复后 |
|---|---|---|
| fp16 单发 prefill（热机） | 522 | **964**（冷 GLCache）→ **1446**（二次进程起） |
| fp16 稳态 prefill | 2335-2614 | **2677-2700**（顺带 +5%：省 key 的 params 拷贝） |
| int8 单发 prefill | 748 | **1342**（两轮稳定） |
| int8 单发 decode | 116.5 | 116.3（无回归） |
| fp16 单发 decode | 78.8 | 85.3（热态波动带内） |

### 数值一致性（stash 前后同命令对比）

- `forward_seq vs CPU fp32`：0.053183 → **0.053183（完全一致）**
- `GPU vs CPU`：8.918588 → 8.918440（fp16 模型固有量化差，噪声级）
- single-token seq vs tok：0.119522 → 0.127735（tok 路径含 atomic float 累加，
  运行间本有微差，同量级）
- 59 项单测全过（含走 DYNAMIC 路径的 Vulkan 端到端测试）

### 剩余差距说明

单发（1446）与稳态（2700）仍差 ~70ms：**该 m_pad 桶首次出现的 ~30 个 spec 各付一次
pipeline 创建**（descriptor 类型变更后 GLCache 首轮也全量 miss，属一次性）。这是
spec 首现的固有成本，服务端可在加载后按常用 T 桶各跑一次 dummy prefill 预热消除；
benchmark 层面可让 warmup 与计时同长度。

### 注意事项（后续维护）

- **descriptor 类型已全局改为 DYNAMIC**：新增算子若用 `bind_uniform`（非池路径）
  会类型不匹配——runtime 的算子统一走 `bind_uniform_with_range`；App 层两方法并存
- uniform 池默认 8MB ≈ 3 万个**不同 params 组合**/批（批内按内容去重后，
  实际负载远低于此；溢出报错含 UNIFORM_POOL_MB 提示）
- 改 descriptor 类型会使 NVIDIA GLCache 首次全量 miss（一次性重编译）

### 补遗：uniform 池批内 dedup（同日发现并修复）

README 重测（`memtest` 1000-token selfloop）暴露池设计缺陷：selfloop **单批**录制
~26 万次 dispatch（每 token ~261 kernel），逐 dispatch bump 分配 slot → 8MB 池在
数百 token 处耗尽报错。旧设计按 (shader, spec, params) 缓存 uniform，天然按
params 去重（selfloop 每 token 的 params 逐位一致），无此限制；池化时丢失了该
语义。benchmark（256 token）与既有单测恰好未越过池边界，故验证未暴露——
**教训：惰性资源池类改动必须用长批次路径（如 1000-token selfloop）回归**。

修复：`pool_slots: HashMap<Vec<u64>, usize>` 批内按 params 内容去重——相同
params 复用同一偏移（不同 shader 解读相同字节，仍与各自调用方传入值一致，共享
安全），begin/end_batch 随游标一并清空。selfloop 池占用回落至 O(kernel 种类数)，
与 token 数无关。

修复后按 README 原口径重测（GPU ~53°C 冷态起，1000 tokens）：
fp16 Vulkan **88.1** / CUDA **92.4**；int8 Vulkan **116.0** / CUDA **126.0** tok/s
（较 README 旧值 +7~14%：coalescing 修复 +1~3%，其余为环境差异）。
