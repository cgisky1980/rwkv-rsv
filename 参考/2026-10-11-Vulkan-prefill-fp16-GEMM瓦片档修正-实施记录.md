# Vulkan 批量 prefill fp16 GEMM 瓦片档修正（2026-10-11）

对应任务 #26（原立项为「Vulkan 批量 prefill 的 int8 张量核 GEMM」，口径 0.59×）。
本轮**没有**写 int8 prefill 内核——先做了一件更基础的事：把 prefill 的 GEMM 变成
可跨后端对拍的测量对象，结果发现差距的大头是**默认瓦片档选错**，一行选档逻辑
换掉就吃掉了 0.574× → 0.96×。int8 张量核 prefill 因此降级为「锦上添花的后续项」。

## 一、先修测量：两个「假胜利」

### 1. CUDA 的 `begin_batch`/`end_batch` 不提交也不同步

Vulkan 侧 `end_batch` = end command buffer + submit + `queue_wait_idle`；CUDA 侧两个
方法都不做提交（内核直接 launch 到 stream 上），`end_batch` 只 dump profiling。
于是 kernel_bench 的计时圈量到的是 **host 入队时间**，`gemm` 报出 1840~112 640 TOPS
（fp16 张量峰值只有 ~54）。不是 cuBLAS 快，是没等它。

修法：`ComputeBackend::synchronize()` 作为**必需** trait 方法加入（故意不给默认实现——
默认 no-op 会让下一个后端悄悄重现同一个 bug），Vulkan = `rt.wait_queue()`
（新增，`queue_wait_idle`），CUDA = `cuStreamSynchronize(self.stream)`。
kernel_bench 每轮 `end_batch` 之后显式同步。

⚠️ **此前所有 CUDA 的 kernel_bench 行一律作废**，任何跨后端比较都要重测。

### 2. `sum(y)` 输出指纹列

修好同步之后 CUDA 给出物理上合理的 33.5~57.8 TOPS，但 Vulkan 的瓦片扫描里
`32×32×32` 那一档报 **234 TOPS = fp16 峰值的 4.3 倍**，而且在表里跟别的档看不出
任何区别。给基准加了 `sum(y)` 列（计时圈**外**单独跑一次 + 部分读回前 4096 个元素），
该档 `sum(y) = 0.0000` ⇒ 内核从未写过输出。

根因（结构性的，值得记住）：`gemm_spec` 里 `SUBTILE_M = TILE_M/4`、
`NUM_MAT_SUBTILE_M = SUBTILE_M / MAT_M`，MAT_M=16 ⇒ **TILE_M < 64 时零次迭代**，
内核合法地什么都不做并「成功」返回。同理 TILE_N < 32。顺带发现旧的共享内存兜底
返回 `(subgroup_size, subgroup_size, 16)` = `(32,32,16)` 就是这么一个静默空转档，
已把下界抬成 `(64,32,16)`。

结论：**没有读回的 GPU 基准数字不可引用**。给基准加「结果不可验证」的显式失败分支，
比给每个配置加人工怀疑便宜得多。

## 二、瓦片扫描（`pf_*` 族，带指纹，同会话）

`pf_out` = [rows,2560]×[2560,2560]、`pf_key` = k=2560→n=10240、`pf_value` = k=10240→n=2560。
batch 列是 **token 行数**（prefill 的 B·T_pad）。`pf_*` 走稠密 fp16 `gemm`，两后端都实现
⇒ 这是唯一能跨后端对拍的 prefill 口径（`supports_int8_imma()` 在 CUDA 上为 false，
`imma_*` 族压根没法对比）。判断看 TOPS 列，不看「有效 GB/s」。

单位 TOPS（`sum(y)` 全部为 1049.0018 / 4195.4004，各档逐位相同 ⇒ 纯性能差异）：

| 行数 | 64×64×32（旧默认） | 128×128×32 | 128×64×32 | 64×128×32 |
| --- | --- | --- | --- | --- |
| pf_out 512 | 21.4 | 21.0 | **24.6** | 22.6 |
| pf_out 1024 | 22.5 | 27.1 | **30.1** | 27.7 |
| pf_out 2048 | 22.9 | **33.0** | 30.2 | 27.8 |
| pf_out 4096 | 13.8 | **31.6** | 24.4 | 23.7 |
| pf_key 256 | 23.1 | 27.2 | 29.9 | 25.9 |
| pf_key 1024 | 22.5 | **35.5** | 29.4 | 30.0 |
| pf_key 2048 | 22.8 | **44.4** | 40.1 | 35.4 |
| pf_key 4096 | 13.6 | **39.2** | 18.8 | 23.4 |

两件事：① 128×128×32 在行数 ≥ 512 上一律最快；② 行数 = 256 时两档在噪声里等价
（抖动 75~106 %，单次 0.19 ms 已低于计时分辨率）。**所以不需要按 batch 分档**——
担心的「小 M 时 128×128 只有 40 个块、68 个 SM 吃不满」在实测里不成立。

## 三、选档逻辑改两处（`src/runtime.rs`）

### 1. NVIDIA 的档位判据从「绝对阈值」改成「共享内存需求」

旧代码 `max_shared_memory >= 163840` 才给 128×128，这张 Turing 的 API 返回 48 KB
⇒ 永久锁在 64×64×32（阈值口径参照的是 AMD 的 64 KB+，对 NVIDIA 不成立）。
现在按 `gemm_shmem(tm,tn,tk)` 实际需求量与 `max_shared_memory` 比较：128×128×32 只要
36 KB（行填充 ROW=TILE_K+8 ⇒ 真实 40 KB），48 KB 的卡照样选得上。
共享内存兜底公式也统一到同一个 `gemm_shmem`，不再有两套口径。

### 2. 形状整除性守卫（这是「能不能默认打开」的关键）

`gemm.comp` 是 Khronos 风格的 coopmat 内核：三层循环全是编译期定长 trip count、
内层没有任何边界检查，而 dispatch 用 `div_ceil` ⇒ **M/N 不是 TILE_M/TILE_N 的整数倍时，
最后一个 tile 会读写越过缓冲末尾**。低秩投影正是这种形状：n = `round_up(mid, 64)`
可以恰为 64，128×128 会往 `[M_PAD, 64]` 的 v_mid 上多写 64 列、并越过 v1_16 尾部。

新增 `gemm_tile_for(app, m, n, k)`：从硬件偏好档往下**二分收缩到能整除的最大档**，
下界 = 结构约束（M 64 = 4·MAT_M，N 32 = 2·MAT_N，K 16 = MAT_K）。
`gemm_spec` 与 `gemm_grid` 都走它，且 TILE_M/TILE_N 只依赖 M/N ⇒ 两处必然一致。
收缩只降低共享内存需求，与上限校验不冲突。

前置事实（所以整除守卫今天不会把主路径降档）：`SeqBuffers::m_pad = t.div_ceil(256)*256`
（gpu_model.rs:5601）恒为 256 的倍数；c=2560 / fh=10240 / vocab=65536 都能被 128 整除；
只有低秩 n 会落到 64 档。

## 四、结果

内核级（4096 行，同会话，指纹逐位相同）：

| 形状 | CUDA(cuBLAS) | Vulkan 新默认 | Vulkan 旧默认 | 新比值 |
| --- | --- | --- | --- | --- |
| pf_out 2560×2560 | 41.1 | 36.9 | 13.8 | **0.90×** |
| pf_key 10240×2560 | 48.2 | 39.2 | 13.6 | **0.81×** |
| pf_value 2560×10240 | 56.6 | 25.6 | 12.2 | **0.45×** |

端到端单流 prefill（`prof_prefill_steady`，SKIP_DECODE=1，best-of-3）：

| PTOKENS | Vulkan 64×64 | Vulkan 128×128 | CUDA | 比值（旧→新） |
| --- | --- | --- | --- | --- |
| 512 | 2932 | 2973 | 3706 | 0.79 → **0.80×** |
| 2048 | 2670 | 3788 | 4328 | 0.62 → **0.87×** |

端到端**批量** prefill（`batch_decode_bench` B=128 PAD_TO=512，同会话三臂，
prefill seeds 指纹 `0x1b9664` 三臂全等 ⇒ 输出 token 一致）：

| 臂 | prefill | vs CUDA |
| --- | --- | --- |
| CUDA | 23 168 ms | — |
| Vulkan 64×64（旧默认） | 40 421 ms | 0.574× |
| Vulkan 128×128（新默认） | 24 119 ms | **0.961×** |

0.574× 与 README §9 记的 0.59× 对上（跨会话复现），口径可信。同一行里 CUDA 的
decode 3429.8 tok/s vs Vulkan 1602.0 ⇒ **decode 仍是 0.47×**，prefill 不再是瓶颈项。

## 五、遗留

### 5.1 批量 prefill 的残余差距在 B=256，而且是**墙钟**问题不是内核问题（新发现，下一步）

同会话（`PAD_TO=512 NTOK=1 SEGS=1 SKIP_SINGLE=1`，prefill seeds 三臂全等 `0x372cc8`）：

| B | PREFILL_SLOTS | Vulkan prefill | 登记显存 | vs CUDA |
| --- | --- | --- | --- | --- |
| 128 | 8（默认） | 24 119 ms | ~9.5 GB | **0.961×** |
| 128 | 16 | 27 910 ms | 11 638 MiB | 0.830× |
| 128 | 32 | 30 998 ms | 15 438 MiB | 0.748× |
| 128 | 64 / 0（不分块） | `ERROR_OUT_OF_DEVICE_MEMORY` | — | — |
| 256 | 8（默认） | 75 197 ms | 12 747 MiB | **0.614×** |
| 256 | 16 | 54 409 ms | 14 279 MiB | 0.849× |
| 256 | 32 | 59 670 ms | 18 079 MiB | 0.774× |
| 256 | — CUDA | 46 173 ms | — | 1.00× |

B=256 仍只有 0.61×，而 B=128 已经 0.96× ⇒ 差距不是内核。剖一个 chunk（B=256/ps=8，
bt=4096 行，32 个 chunk）：

- `[PROF_GPU] SUM 1023 ms / 1723 kernels`（gemm_affine 369 + gemm 193 + relu2 186 + tanh 6 = **74 %**）
- `[PROF_GAPS] 间隙合计 0.693 ms / 1722`（最大 159 us，`to_f16_triple → gemm_tanh`）⇒ **GPU 侧没有 inter-dispatch 停顿**
- `[PROF_HOST] host=26.1 ms [key=0.12 cache=22.71 barrier=1.33 dispatch=1.27]` ⇒ 录制也不是瓶颈（cache 查表占大头）
- `[PROF_SUBMIT] submit=0.05ms wait=1029ms` ⇒ 与 GPU SUM 对得上（wait 就是 GPU 时间，无排队损失）

而墙钟是 75 197 / 32 = **2350 ms 每 chunk**。1029(GPU) + 26(host) + 0.7(gap) ≈ 1056 ms，
**剩下约 1.3 s / chunk（占 B=256 prefill 墙钟的 55 %）落在 chunk 与 chunk 之间**，
既不在内核里也不在录制里。ps=8→16 在 B=256 上省 20.8 s ≈ 16 chunk × 1.3 s，
正好是这个量级——但同一改法在 B=128 上**反而更慢**（+3.8 s），因为登记显存
从 ~9.5 GB 涨到 11.6 GB；再涨到 15.4 GB / 18.1 GB 就分别把 B=128、B=256 拖回更慢，
64 与不分块直接 `ERROR_OUT_OF_DEVICE_MEMORY`。

所以这不是「把 PREFILL_SLOTS 调大」能白拿的收益，而是一个**每 chunk 固定开销 × 显存
占用**的双目标问题。下一步先归因那 1.3 s：候选是 chunk 循环里走 batch 路径**之外**
提交的活（状态写回 / 历史上传 / 未计时的 dispatch）、以及 seq 缓冲按桶重建。
显存侧的联动项是 #27（RNN 状态 fp16：B=256 时 5.3 GB → 2.6 GB），它把「更大子批」
重新变成可选项。

#### ⚠️ 订正（同日，见 [2026-10-11-Vulkan-分块prefill状态回灌设备化-实施记录.md](2026-10-11-Vulkan-分块prefill状态回灌设备化-实施记录.md)）

上面这段的**结论错了**，错在把污染的窗口当分母：

1. `2350 ms / chunk` 是 `75 197 / 32` 得来的，而 `75 197` 是 bench 的
   `[批量] prefill` 列——它从 `create_batch_state` 之前起表、到**解码循环之后**停表，
   含 tokenize、状态分配与 `WARMUP` 自循环（图捕获 / NVRTC JIT）。chunk 循环自己的墙钟
   加进 `PROF_CHUNK=1` 后实测为 **42 430 ms**（旧代码）/ **36 023 ms**（修好以后），
   即每 chunk 1 326 / 1 126 ms，不是 2 350 ms。
2. 因此「1.3 s 落在 chunk 与 chunk 之间」不成立。它**在 chunk 里面**，而且就在候选清单的
   第一条：状态写回走宿主往返，每槽 194 次「独立提交 + `queue_wait_idle`」
   （`download_part`/`upload_part` 各录一个临时 command buffer）。实测拆分为
   reset 62 + core 1 030 + **xfer 210** ms / chunk（B=256 / ps=8）。改成设备内行段拷贝
   （`State::copy_rows_from`，一块一次提交）后 xfer 6 713 → **44 ms**（整轮），
   B=256 对 CUDA **0.67× → 0.79×**、B=128 **0.84×**。
3. 上面那张 PREFILL_SLOTS 六臂表的全部 ms 与 `vs CUDA` 列都读自被污染的 prefill 列，
   **只有定性结论仍成立**：登记显存随子批宽度上涨、64 / 不分块放不下（OOM 两行是真实
   观测）。ps=8→16「省 20.8 s ≈ 16 × 1.3 s」的对应关系是巧合而非机制——干净口径下
   `PREFILL_SLOTS` 各臂需要重测才能引用。
4. 顺带发现一条跨后端口径差异：**CUDA 上 `PREFILL_SLOTS` 是影响结果的旋钮**
   （w=1 与 w=2 逐位相同、两者与 w=4 差 ~1 %，即批形跨阈值换 kernel），Vulkan 上逐位
   不变。已由 `tests/batch_parity.rs::cross_state_rows_copy_matches_host_roundtrip` 分档钉住。

### 5.2 其余

1. **pf_value（深 K、窄 N）0.45×** 是唯一没闭合的 prefill 形状：n=2560 ⇒ N 方向只有
   20 个块，k=10240 ⇒ 每块扫 320 个 K-tile，A 面板被反复流入。方向是 K 方向的分块
   /更大 TILE_K，或 N 方向的 persistent 分组；与 CUDA 的 56.6 TOPS 比还差 2.2×。
2. int8 张量核 prefill（本轮原目标）现在只在「省带宽」意义上还有价值（dequant 那趟
   只占 0.9 %），fp16 路径已经基本追平，优先级排到 pf_value 与 decode 之后。
3. 基准纪律：任何 GPU 计时必须显式同步 + 带回读指纹。已写进 kernel_bench 的表尾注释。

门禁：`cargo fmt --all`、`cargo clippy --release --all-targets -- -D warnings`、
`BACKEND=vulkan cargo test --release`（77 lib + 9 batch_parity + 1 ffn_sparse，0 失败）、
`BACKEND=cuda cargo test --release`。
