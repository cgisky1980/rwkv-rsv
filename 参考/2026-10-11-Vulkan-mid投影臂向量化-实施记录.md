# 2026-10-11 Vulkan：B=128 残余差距的内核级归因 + mid 投影臂向量化（S27–S29）

会话日期：2026-10-11。后端：Vulkan（RTX 2080 Ti，22 GB 改版，同时是显示适配器）。
模型：`rwkv7-3B-int8.st`（C=2560 / H=40 / N=64 / ffn_hidden=10240 / vocab=65536 / L=32）。
本轮起点：第五次 Vulkan/CUDA 成对复测（README §6.1）之后，B=128/256 还剩 ≈31 个百分点。

## 1. 目的与结论摘要

目的：把「剩下的大批量差距到底压在哪些核上」量出来，而不是继续猜。

三条结论：

1. **跨后端的分核对齐用现有两把仪器做不到**——`PROF_CUDA_KERNEL` 在它唯一能用的配置下
   把 host 发射间隙算进了内核时长（§2，已登记为死仪器路径）。
2. **Vulkan 自己的 dump 自洽**，并给出新的排序：`gemm_imma` 48.6 % /
   **mid 投影臂 `gemv_int8_rkv_stage1_batch` 18.8 %** / 状态核 13.8 %（§3）。
3. **mid 臂既不是带宽受限也不是「访存指令数」受限**：加大 `GEMV_BGRP` 省权重重读端到端 **−8.6 %**
   （§4，死路）；于是改内循环本身——四路 mid 投影由标量改 **vec4 + `dot`**（指令与载入条数 ÷4），
   同会话反序 A/B 端到端 **−6 ~ −9 %**（§5，死路，已回退）。**两根都死 ⇒ §3 的「剩下就是指令
   条数/延迟」这个判据被证伪。**
4. **CUDA 侧同一个核独立量到同一条机制**（§6）：它的 mid 臂同样是标量循环、同样只用 `BGRP=8`
   （`MID_GG=2/4/8` 实测 1.39/1.98/4.66 ms vs GG=1 的 1.03 ms），而它拿到的 2.5× 来自消除
   `xsrc[BGRP]` 指针数组落 local memory——**Vulkan 侧本来就是内联寻址，没有这个坑**。
   ⇒ CUDA 的 mid 臂形状与 Vulkan 几乎一致，「差距压在 mid 臂」这个归因**待验**。
5. 下一步（§7，任务 #53）：给 Vulkan 补一把与 CUDA 同名的 **`ABLATE=csv` 按名消融尺**
   （纯 host、`src/runtime.rs:1089` 一处），用它做同口径的跨后端逐核归因。

## 2. 跨后端分核对齐：仪器不可用（死路登记）

命令（同一会话串行，脚本 `C:/tmp/prof128both.sh`）：

- Vulkan：`PROF_GPU=1 SKIP_GRAPH=1 NTOK=1 SLOTS={64,128} SEGS=1 PAD_TO=512 PREFILL_SLOTS=8 TOPK=1`
- CUDA：`PROF_CUDA_KERNEL=1 NO_SELFLOOP_GRAPH=1 NTOK=4`（同尺同门控）

CUDA 侧读数不可用的三条独立证据（B=128）：

| 证据 | 读数 |
|---|---|
| 合计远超自身墙钟 | dump `SUM 6 249 ms`，而该次运行 decode 墙钟 ≈150 ms（4 步 × ≈38 ms） ⇒ **≈41×** |
| 分次 flush 而非按步 | 日志里 `SUM` 从 368 ms 一路涨到 6 249 ms，共 30+ 行（事件池填满即 flush） |
| 排名被发射次数压住 | `to_f16` 3 568 次 / `dequant_int8` 3 072 次 / `elementwise_sigmoid` 2 032 次，占据前列 |
| 批宽不敏感 | 同一个 `fuse_ka` 在 B=128 与 B=64 的 avg 分别是 1.169 / 1.205 ms——流量随批宽翻倍的内核不可能等价 |

机制：`backend_cuda.rs` 的事件对是**非同步**的（launch 前后各记一个 event，dump 时统一取）。
开 self-loop 图时内核不执行、计时全 0（所以必须 `NO_SELFLOOP_GRAPH=1`）；一旦关掉图，decode 变成
host 发射受限，队列一空，`begin` 事件立刻完成而 kernel 要等下一次提交才开工 ⇒
**每对事件量到的是「GPU 执行 + host 提交等待」**。这不是散布，是系统性偏大且按发射次数分配。

⇒ 跨后端归因要走**消融**（`ABLATE=`，`backend_cuda.rs` 注释里 CUDA 自己也说它「比
PROF_CUDA_KERNEL 可靠」）或 `PROF_GAPS`，不要用这两个剖析器相减。登记为死仪器路径。

## 3. Vulkan 侧 dump：B=128 每步 46.6 ms 的归属（可用）

同一进程内 GPU 时间戳自洽：`SUM 46.6 ms / 1 349 kernels`，对上未剖析时的 55.7 ms/步 ⇒
已解释 ≈84 %，余量是 dispatch 间隙。

| 内核 | 每步 ms | 份额 | 折算 |
|---|---|---|---|
| `gemm_imma_bm64_nst8` | 22.70 | 48.6 % | count=193（split-K 多形状） |
| `gemv_int8_rkv_stage1_batch` | **8.76** | **18.8 %** | count=32 ⇒ **0.2739 ms/层** |
| `fuse_ka_dplr_norm_s16` | 6.42 | 13.8 % | 0.2008 ms/层 = 83.9 MB / 0.2008 ms = **418 GB/s = 峰值 68 %**（已在屋顶，§6.10） |
| `gemm_imma_reduce` | 2.04 | 4.4 % | |
| `sample` | 1.91 | 4.1 % | |
| `gemm_f16_f32` | 1.64 | 3.5 % | |
| `quant_x_i8` | 1.44 | 3.1 % | |
| `norm_lerp6_batch` | 1.21 | 2.6 % | |

注：r/k/v 在 §6.3 之后默认已走 `gemm_imma`，所以 `gemv_int8_rkv_stage1_batch` 在这里
**只剩 mid 投影臂**（`v1/w1/a1/g1`，fp32 权重，`VM=64 / WM=96 / AM=96 / GM=320`，共 576 行），
dispatch = `(VM+WM+AM+GM, ceil(batch/BGRP))` ⇒ 每 block 一行。

两个折算都不贴屋顶，是这一臂的定性判据：

- 带宽：5.9 MB/层 × ceil(128/8)=16 次重读 = 94 MB/层 ÷ 0.2739 ms = **343 GB/s**（实际可用 ≈560）
- 算力：2 × 576 × 2560 × 128 = 377 MFLOP/层 ÷ 0.2739 ms = **1.37 TFLOP/s**（fp32 峰值 13.5 的 10 %）

⇒ 既不是带宽受限也不是算力受限，剩下的就是**每 lane 的访存/算术指令条数与延迟**。

## 4. S28：`GEMV_BGRP` 16/32 在 mid 臂上是否可行 —— 死路

脚本 `C:/tmp/bgrp_probe.sh`，B=128、SEGS=1、NTOK=64、同尺门控，3 轮交错（第 2 轮反序）。
绑定标记 = 三臂 decode token 校验和 **`sum=0x690fa29 xor=0x6b23` 逐位相同**（BGRP 只改
「哪些槽落在同一 workgroup」，累加顺序不变）⇒ 下面所有位移都是纯结构差异，没有数值混淆。

| 轮 | BGRP=8 | BGRP=16 | BGRP=32 |
|---|---|---|---|
| 1 | 2360.7 | 2143.9 | 2147.3 |
| 2（反序） | 2305.0 | 2107.7 | 2142.1 |
| 3 | 2226.7 | 1991.9 | 2016.4 |
| 中位 | **2305.0** | 2107.7（−8.6 %） | 2142.1（−7.1 %） |

三轮的**同轮配对**里 8 都赢，且 16 与 32 彼此几乎相等 ⇒ 位移不是随槽组宽线性变化的
「流量」效应，而是**一过 8 就出现的台阶**。合理解释：这个 env 同时被
`gemv_lowrank_chain4_batch` 消费，后者的 `float acc[ROWS][BGRP]` 在 BGRP>8 时溢出到
local memory（§6.2① 已在 B=256 量过同一台阶：8→986.9、16→890.3、24→750.7、32→718.1）。

⇒ **死路登记**：想靠加大 BGRP 省 mid 权重重读是错的方向（它本就不是带宽受限）；
`GEMV_BGRP>8` 会被 chain4 的溢出拖回去，端到端净亏 ≈9 %。

## 5. S29：mid 臂标量内循环 → vec4 + `dot`

改动（`assets/shaders/src/gemv_int8_rkv_stage1_batch.comp`，spv md5
`9addb312…` → `827ba705…`）：

- `Params` 里 `in_v1/in_w1/in_a1/in_g1`、`in_xw/in_xa/in_xg` 由 `BufF` 改 **`InX4`**
  （同一 buffer_reference，8 字节槽位，**host 侧 `params` 数组一个字都不用改**）；
- 四路内循环由 `for k in index..C step BLOCK_SIZE` 标量取 4 B，改成
  `for k4 in index..XSLOT step BLOCK_SIZE` 取 `vec4`，累加写成
  `accM[s] += dot(w4, x4)`。迭代次数 **20 → 5**，访存指令数 **÷4**，每 lane 每指令搬 16 B
  （一个 warp 连续 512 B，仍完全 coalesced）。

⚠️ **数值口径变化（不是逐位不变）**：lane 内求和从「每 4 个 k 标量连加」变成
「一次 `dot` 合并 4 个乘积」，`dot` 的分量求和顺序由实现决定 ⇒ 换默认档必须过质量门：
`cargo test --release`、对 CPU fp32 基准的 logit 偏差（DIAG）、贪心逐 token 分叉率。

A/B 设计（`C:/tmp/mid4_ab.sh`）：两臂**都**走 `VK_SPV_DIR`（同一代码路径，只差目录），
OLD=`/c/tmp/spv-old-mid4`（改动前快照）/ NEW=`assets/shaders/spv`；先各跑一个不计时进程
（spec 常量按 batch 特化，重建后第一个进程会在计时区内编译 pipeline），再 3 轮交错、第 2 轮反序；
每臂打印它实际绑定的 stage1 spv md5 作为绑定标记，同臂跨轮校验和必须一致。

结果：**NEW 慢 6–9 %，弃用**（已 `git checkout` 回退，spv md5 回到 `9addb312…`；实验源码留在
`C:/tmp/mid4-vec4-experiment.comp`）。

| 轮（顺序） | OLD 标量 tok/s | NEW vec4 tok/s | NEW/OLD |
|---|---|---|---|
| 1 OLD→NEW | 385.7（弃） | 582.8（弃） | — |
| 2 **NEW→OLD**（反序） | 2324.9 | 2107.2 | **0.906×** |
| 3 OLD→NEW | 2243.7 | 2108.8 | **0.940×** |

- 绑定标记齐：每臂每轮都打印它实际绑定的 stage1 md5（OLD `9addb312` / NEW `827ba705`），
  同臂跨轮 decode 校验和逐位一致（OLD `sum=0x690fa29 xor=0x6b23`；NEW `sum=0x6938dbb
  xor=0x17c3`）⇒ 两臂差只是求和结合序，不是绑错目录。
- **第 1 轮整轮作废**：两臂 prefill 116.7 / 146.4 s（正常 26 s）——即便脚本已按同一 `SLOTS`
  各跑过一个不计时进程，**pipeline 编译仍落在计时区内**（spec 常量特化的变体集比预热跑的更宽）。
  轮 2 起 prefill 全部回到 25.7–26.4 s，读数和 §6.5 记的那条坑一致。**教训升级**：预热跑的
  `NTOK` 必须与计时跑**同档**，只提 `NTOK=1` 不够；否则第一轮必被污染。
- 干净两轮**反序轮也同向**，且两臂区间不重叠（OLD 最低 2243.7 > NEW 最高 2108.8）⇒ 方向可信。
- 静态确认改动确实生效（`spirv-dis` 两臂对比）：`OpDot` 1 → 5（四个 mid 循环 + rkv 段原有
  一处），循环体「2 次标量载入 + fmul + fadd」/k 变「2 次 `v4float` 载入 + dot」，每 lane
  trip count 20 → 5。**指令条数与载入字节数都按预期降了，却还是更慢。**

质量门：**不适用**（不采纳该改动 ⇒ 无数值口径变化需要把关；本轮 NEW 的校验和差异停留在探针里）。

## 6. 为什么慢：两个后端独立量到同一条机制（本记录的主要产出）

去读 CUDA 侧同一个核的 mid 分支（`backend_cuda.rs:2731-2812` + 启动器 `:8716-8745`），发现
**这两根杠杆它都试过并且都登记为负**：

1. **CUDA 的 mid 臂同样是标量循环**（`for kk = tid; kk < c; kk += blockDim.x:
   local_dot[gg][b] += w * xbase[s*c + kk]`），没有向量化 ⇒ 「÷4 指令条数」不是 CUDA 领先的原因。
2. **`MID_GG`（=每 block 槽跨度 `BGRP*MID_GG`，等价于 Vulkan 的 `GEMV_BGRP`）实测台阶**：
   GG=1 → **1.03 ms/次**，GG=2/4/8 → 1.39 / 1.98 / 4.66 ms，原话是「累加器变多又把寄存器压回
   溢出」。**与 S28（本仓 Vulkan，BGRP 8→16→32 净亏 8.6 %）同因**——两个后端、两套编译器、
   同一个台阶 ⇒ S28 里「被 `gemv_lowrank_chain4_batch` 的溢出拖累」这个混淆解释要让位给
   「**每 block 槽数 >8 本身就溢出**」，这条 env 本来就不该往 8 以上调。
3. **CUDA 在这个臂上真正的 2.5×（2.55 → 1.03 ms）来自消除数组化基址**：原先
   `const float* xsrc[BGRP]` 预存每槽基址被编译器落到 **local memory**，改成
   「单一 `xbase` + 常数偏移直接寻址」才拿回 2.5×。Vulkan 侧写法本来就是
   `params.in_xv.data[(slot0 + s) * XSLOT + k]` 内联寻址，**没有这个坑可消**——
   所以这根已在 CUDA 身上验证过的最大杠杆对 Vulkan 无效。

⇒ **mid 臂的定性判据要改写**：§3 的「既不贴带宽也不贴算力，剩下就是指令条数/延迟」在
「指令条数」这一半被本轮证伪（条数 ÷4 仍慢）。剩下的候选是**每 lane 在飞的载入条数**
（trip count 20 → 5 削掉了 4× 访存并行度）与 L2 侧按行×槽重读的字节，两者都不是「少发指令」
能碰到的。另外 CUDA 的形状与 Vulkan 几乎一致（同 576 行、同 1 行/block、同 BGRP=8、同样标量），
所以 §9「差距压在 mid 投影 SIMT 臂」这个归因**本身待验**——它可能两边同价。

## 7. 由此决定的下一步：把消融尺补到 Vulkan 侧（任务 #53）

S27 想做的「跨后端逐核归因」被 CUDA 那把 event-pair 尺判死（§2），而 Vulkan 侧只剩自家
`PROF_GPU`，两边不同尺 ⇒ 只能看单侧排名。CUDA 后端另有 **`ABLATE=csv`**（`backend_cuda.rs:1084-1117`
+ 跳过 launch `:741`，注释原文「比 `PROF_CUDA_KERNEL` 可靠」）：按名不启动某个 kernel，用**端到端
位移**度量它的代价。这把尺是跨后端可比的（同一个环境变量、同一个 bench、同会话交错），而 Vulkan
侧只有 `gemv_int8_rkv_stage1_batch` 一个核内的 `RKV_ABLATE` 位选，**没有通用的按名消融**。

⇒ 本轮的落地动作：给 Vulkan 补一把同名尺（Vulkan 的 dispatch 集中在 `src/runtime.rs:1089` 一处，
`shader_name` 就在手边，纯 host 改动、不动 shader、不进产品路径）。有了它，「差距压在 mid 臂」
这个待验归因可以用 4 臂同会话（base/ablate × cuda/vulkan）直接量出来，而不必再靠单侧 dump 猜。

## 8. S30：Vulkan 侧 `ABLATE` 尺建成 + 校准（B=128，与 §4/§5 同尺）

实现（`src/runtime.rs`）：`ablate_hit(shader_name)` 挂在 `record_kernel` 里唯一的
`cmd_dispatch` 上——命中的 kernel **只跳 dispatch 与它的两个 GPU 时间戳**，pipeline 创建、
uniform 池 slot、barrier、`cmd_bind` 全部照录 ⇒ 基线臂与消融臂在 host 侧逐字一致，唯一差别
是这条 dispatch 上不上卡。env 名与 CUDA 后端**共用 `ABLATE`**（同一个变量、同一个 bench、
同会话交错 ⇒ 跨后端可比）。

**第一个坑就撞上了 CUDA 文档里那条**：`record_kernel` 收到的是
`shaders/spv/<name>.spv`，而人和 dump 都写裸名 ⇒ 首跑四臂全部 `hit=NONE`、校验和与 base
**逐位相同**、tok/s 也在噪声内 —— 也就是「一处都不命中」的假零读数。区别只在于本次实现里
加了**每个名字首次命中打一条 `[ABLATE] 命中 <name>`**，日志没出现就等于没匹配，才没有把
「消融后没变化」读成「这个 kernel 不值钱」。修法是名字归一化（剥目录与 `.spv`）。

校准（修好后重跑，`C:/tmp/abl_cal.out`；每臂两轮、第 2 轮反序；臂内跨轮校验和一致：
base `0x690fa29/0x6b23`、mid `0x71a728/0xaf2c`、state `0x5a000/0x0`、imma `0xfdc027c/0x0`）：

| 臂 | r1 base | r1 消融 | 位移 | r2 消融 | r2 base | 位移 | dump 预测份额 → 折算吞吐 |
|---|---|---|---|---|---|---|---|
| mid 投影 `gemv_int8_rkv_stage1_batch` | 2389.3 | 2860.6 | **+19.7 %** | 2772.9 | 2319.1 | **+19.5 %** | 15.7 % ⇒ +18.6 % ✅ |
| 状态核 `fuse_ka_dplr_norm_s16` | 2389.3 | 2659.6 | **+11.3 %** | 2664.8 | 2319.1 | **+14.9 %** | 11.5 % ⇒ +13.0 % ✅ |
| int8 GEMM `gemm_imma_bm64_nst8` | 2389.3 | 1244.5 | **−47.9 %** | 1232.5 | 2319.1 | **−46.9 %** | 48.6 % ⇒ +94 % ❌ |

⇒ **两条与 dump 对上（mid 臂、状态核），这把尺在「消融后下游只读陈旧数据、不改变任何内核
工作量」的核上是可信的**。`gemm_imma` 那条是 CUDA 注释预告过的**反向失真**：被消融的生产者
不写输出 ⇒ 下游 `relu²` 拿到非零垃圾，稀疏 FFN 从「几乎免费」变回满密度，多出来的时间盖过了
被移除的 48.6 %。所以**imma 的份额只能用 `PROF_GPU` dump 读，不能用消融读**（登记进 §9 读数
纪律：消融读数只在「该核的缺席不改变其它核的工作量」时有效）。

### 8.1 同一把坑在 CUDA 侧也踩了一次（所以两边都加了命中日志）

CUDA 首轮（`C:/tmp/abl_cal_cuda.out`）用 `ABLATE=gemv_int8_rkv_stage1_batch /
fuse_ka_dplr_norm_batch / imma_gemm_batch` 四臂跑完：**全部与 base 逐位同校验和
（`0x6934e06/0x75b4`）、tok/s 也在噪声内** ⇒ 一处都没命中。根因是 CUDA 的消融按
**kernel 缓存键**匹配，而键带几何后缀（`fn kernel(key, src, entry)` 里
`register_kernel_name(func, key)`）。从 run 日志的 `compiling kernel <key>` 直接抄出来的真键：

| 角色 | CUDA 真键 | Vulkan 名 |
|---|---|---|
| mid 投影（低秩一级） | `gemv_int8_rkv_stage1_batch_g1` | `gemv_int8_rkv_stage1_batch` |
| 状态核 | `fuse_ka_dplr_norm_s16_w4` | `fuse_ka_dplr_norm_s16` |
| int8 张量核 GEMM | `imma_gemm_batch_b128n64l2s1p0d1k1e0v0_m65536k2560` 等 5 个（含 `_z3_*` 合并的 r/k/v） | `gemm_imma_bm64_nst8`（一个键跑全部形状） |
| 低秩两级 | `lowrank_gemm_1b64n16k128_2b64n64k64_s1` / `_s2` | 一级即上表 mid 臂；二级 `lowrank_epilogue_*` |

⇒ **两边后端现在都在「每名字首次命中」打 `[ABLATE] 命中 <name>`**（CUDA 原来只在启动时
warn 一次请求集合，静默落空完全看不出来——它自己的文档注释写了这条坑，但没有配套日志）。
任何消融读数**必须先确认 hits>0 且校验和与 base 不同**，否则就是把「没匹配」读成「零收益」。

## 9. S30 跨后端逐核归因（这把尺的第一份产出）：差距压在**整条低秩链**，不在 mid 臂单独身上

口径：B=128、SEGS=1、NTOK=64、PAD_TO=512、PREFILL_SLOTS=8、SKIP_SINGLE=1、TOPK=1，两脚本相差
约 20 分钟 ⇒ **只用脚本内配对的 Δ（换算成 ms/步）**，跨脚本只比「角色」。日志
`C:/tmp/abl_cal.out`（Vulkan base/mid/state）、`C:/tmp/abl_cuda2.out`（CUDA base/mid/state/lowrank）、
`C:/tmp/abl_vk2.out`（Vulkan 补 epi 臂，见下方噪声说明）。

步时间 = 128 / tok·s⁻¹：

| 角色 | Vulkan Δms/步（两轮） | CUDA Δms/步（两轮） | 读法 |
|---|---|---|---|
| mid 投影 = 低秩**一级** | **8.83 / 9.03**（键 `gemv_int8_rkv_stage1_batch`） | **hits=0 ⇒ 该核没被启动**（键 `..._g1`） | 与 `PROF_GPU` dump 的 8.76 ms 差 +2 % ⇒ 尺校准 ✅ |
| 状态核 `fuse_ka_dplr_norm*` | 5.45 / 7.16（dump 6.42，差 −2 %） | 7.40 / 8.97（键 `..._s16_w4`） | **CUDA 这个核比 Vulkan 贵**，不是差距来源 |
| 低秩链两级（CUDA 全在 TC） | （Vulkan 一级仍 SIMT） | **4.26 / 5.93**（键 `lowrank_gemm_1b64n16k128_2b64n64k64_s1`+`_s2`） | CUDA 用**张量核**吃掉一级+二级：`_s1` = 四条链**合并成一次 launch**（`blockIdx.z` 选链）、`_s2` = 每链一个 GEMM 且 **epilogue 内联**。⚠️ **不是**「一个融合核 / 一级瓦片留在 SMEM 当二级 A 算子」——两级之间经**全局** `mid16` 传递（读 `backend_cuda.rs:3188`/`:3274`/`:8872`，2026-10-11 订正，见 §9.5） |
| 二级 epilogue（Vulkan） | 7.51 / 0.17 ⇒ **不可用**（base 两轮漂 61.7/56.1 ms） | — | 见下方「本轮脚本的失败」 |

**总量核对**：本尺下 Vulkan base ≈ 54.4 ms/步、CUDA base ≈ 37.4 ms/步 ⇒ **0.688×**，与 §6.1
第五张表 B=128 的 0.688× **逐位对上**（两条完全独立的仪器）⇒ 换算与口径自洽。

### 结论（改变了下一步的选点）

1. **CUDA 在 B≥32 的 decode 路径上不跑 SIMT mid 臂**：消融 hits=0 且校验和与 base **逐位相同**
   （`0x6934e06/0x75b4`）。这与接口注释的方向一致（`backend.rs:697/907`：`lowrank_gemm_batch`
   只有 CUDA 实现、Vulkan 的一级留在 SIMT），⚠️ 但注释里「两级都在一个 kernel 里」这句
   **措辞不准**（它是一对 kernel），已随 §9.5 一起就地订正。
2. ⇒ 两侧「同一条低秩链」的代价是 **CUDA ≈ 5.1 ms/步（一级+二级全包）** vs
   **Vulkan ≈ 8.9 ms（一级 SIMT）+ 二级（`PROF_GPU` 份额 ≈12 % ⇒ ≈6.5 ms）≈ 15.4 ms/步**。
   **差 ≈10 ms，占 17 ms/步总差距的 ≈60 %** ⇒ 残余差距的主战场是**整条低秩链**，
   而不是 §S27 说的「mid 臂单独一名」。
3. ⇒ **§9「一级 stays SIMT，实测 0.443 ms/层（TC）vs 0.274（SIMT）」这条决策是**错框**的**：
   它拿「只把一级搬上张量核」跟「SIMT 的一级」比，而 CUDA 的对照物是**一级+二级全在张量核上**
   ——键名 `1b64n16k128_2b64n64k64` 就是「一级 tile 64×16×k128、二级 64×64×k64」这一对几何。
   ⚠️ 但**不是**「一个融合核 / 一级瓦片留在 SMEM 直接喂二级」（那是本节初稿的说法，读源码后
   证伪，见 §9.5）。被优化的正确量是
   `Vulkan(一级 SIMT + 二级 TC + 4 个独立 epilogue)` vs `CUDA(一级 TC 四链合并 + 二级 TC 内联 epilogue)`。
4. **本轮脚本的失败也登记**：`abl_vk2.sh` 的 base 两轮漂到 61.7 / 56.1 ms（跨 4 分钟的热漂），
   于是它唯一新增的 `epi` 臂位移 7.51 / 0.17 ms 不可用。**教训**：一个脚本里 base 只出现在
   轮首与轮尾时，Δ 的可用性完全取决于这两点是否重合；**base 必须与消融臂在同一轮内紧邻**
   （下一轮脚本把顺序改成 base→abl→base 夹逼）。

### 9.5 零 GPU 成本的追读：CUDA 那一侧到底是什么形态（本节初稿曾写错）

归因把靶子指到「整条低秩链」之后，先读 CUDA 源码再动手——**初稿从键名 `1b64n16k128_2b64n64k64`
推出来的「一个融合核 / 一级的瓦片留在 SMEM 直接当二级的 A 算子」是错的**，实际是三次核、
两级之间经**全局** `mid16`（`[batch, Σ mid_pad]`，host 侧 `:8872` 的调用序注释 + `:8920` 的暂存切分）：

| CUDA kernel | 源位置 | 几何 | 关键点 |
|---|---|---|---|
| `cast4_f16` | `lowrank_gemm_batch` 内 | 4× `[batch,C]` f32→f16 | x 降位，不含 sigmoid（g 的 sigmoid 留给一级出口） |
| `lowrank_stage1_gemm`（`_s1`） | `:3188` | BM=64(批) × BN=16 × BK=128，`blockIdx.z` 选链 | ★ **四条链合并成一次 launch**（`LR1_MERGE`）。原话（`:3183-3186`）：分开发射时 v 链「**只用 16 个块去喂 68 个 SM**」，合并后 ≈160 个有效块一次铺开（≈5 波→≈3 波）。act（tanh/sigmoid）**内联**在累加器出口 |
| `lowrank_stage2_gemm`（`_s2`）×4 | `:3274` | BM=64 × BN=64 × BK=64 | 二级 GEMM **内联** `lr_chain_epi`（`:3165`：v 读改写 / w `exp(sc·sigmoid)` / a sigmoid / g 直出），没有独立的 epilogue kernel |

而 CUDA 的权重是**加载时已常驻的 `*_16` fp16 副本**（`gpu_model.rs:4978-4985`，两后端共用同一份
加载代码 ⇒ **Vulkan 侧这些 fp16 权重本来就在卡上**，一级上张量核不需要任何新的转换成本）。

**由此，#25 那条「一级不能上张量核」的旧结论要降级为待验**：`kernel_bench` 的 `lr1_w/lr1_v/lr1_g`
是**每个形状单独一档**（`examples/kernel_bench.rs:284-310`，`m` 直接就是 n_pad=64/128/320，`k=C`），
所以量到的 0.443 ms/层 = **0.4~0.6 TOPS** 是「块数极少」下的读数——`gpu_model.rs:2391` 自己写的
「coopmat 只有 2×1~3 个块」正是 CUDA 用 `LR1_MERGE` 专门避开的坑。Vulkan 侧从未试过
**四链合并 + 深 K 分块（split-K）**这一档：
- 合并：把 v/w/a/g 四条一级投影放进**一次 dispatch**（`grid.z = 链`），块数 ×4 级别地补回来；
- split-K：`gemm_imma` 一侧早就有 `imma_partial` + `gemm_imma_reduce` 与按全 grid 块数定 `ks`
  的规则（#38/S15），这是「窄 N 深 K」的对症药，与 S13 的 `pf_value` 是**同一个病**；
- 内联 epilogue：现有 `lowrank_epilogue_*` 四条是独立的 [batch, n_pad] 读写一遍，把它们并进二级
  coopmat 的累加器出口，一次省掉 4 条 dispatch + 4 个中间缓冲往返。

**所以 S31 的正确形态不是「造一个融合大核」，而是把 CUDA 这三处结构差别逐个搬过来**，并按
消融尺验收（一级 mid 臂位移应随 `gemv_int8_rkv_stage1_batch` 的 mid 段退休而消失）。

### 下一步（任务 #54，按 §9.5 修订后的选点）

顺序按「成本/收益」排：**① 一级四链合并 coopmat 档（先探针：`kernel_bench` 加一档 `lr1_merged`，
把四条链放进一次 dispatch 量 ms/层）** → ② 若仍打不过 SIMT 的 0.274 ms/层，再上 split-K（复用
`imma_partial`/`reduce`）→ ③ 把 `lowrank_epilogue_*` 内联进二级。任一档输了就停在那里并登记，
不许把三件事捆成一个改动去测。
判据 = 同尺 B=128/256 端到端，以及消融位移从 ≈15.4 ms 降到 ≈5-8 ms。
质量门 = `cargo test --release` + DIAG 对 CPU fp32 的 logit 偏差 + 贪心分叉率（fp16 存储/精度档
与 CUDA 一致，见 `LOWRANK_GEMM` 的「精度档与信天翁一致」）。


---

## §10 S31 步骤 ①-a：占用率瓦片规则落地（2026-10-11，纯宿主改动，已默认开启）

### 10.1 先复现「修复前失败」

`kernel_bench`（`BACKEND=vulkan KB_OPS=lr1_* KB_BATCH=128`）在**默认档**下重放了 §9 那个
判错的读数：一级三条链 0.086 / 0.131 / 0.083 ms/次（四链 ≈0.43 ms/层），而绑定标记
（`KB_LOG=1` ⇒ `gemm_spec` 每次建管线打一行）显示原因在块数：

| 形状（M=128 批行） | 绑到的瓦片 | grid | 块数 |
|---|---|---|---|
| N=64 K=2560（v 链） | (128,64,32) | 1×1 | **1** |
| N=128 K=2560（w/a 链） | (128,128,32) | 1×1 | **1** |
| N=320 K=2560（g 链） | (128,64,32) | 1×5 | **5** |

`gemm_tile_for` 过去**只管整除**（128×128 偏好档按形状折半到能整除为止），从不看块数 ⇒
68 个 SM 喂 1 个块。这就是 §9.5 说的「块饥饿」，不是张量核的天花板。

### 10.2 顺带钉死一条**静默零输出**的结构下界（潜在引擎 bug）

`gemm.comp` 的 tile-load 循环 trip count = `TILE_M*TILE_K/NUM_ELEMENT_VEC4/NUM_INVOKE_TILE`
（fp16 = `TILE_M*TILE_K/2048`），A/B 两侧都必须 `TILE_* × TILE_K ≥ 2048`，否则该侧**一片都不
载入**、累加器保持零并被原样写出，而 dispatch、 barrier、写回全都照常「成功」。
实测 (64,32,32)：`0.034 ms/次` 而 `sum(y)=0.0000` —— 若没有 `kernel_bench` 内建的带读回判据，
这就是一个 2.5× 的假胜利。`gemm_tile` 的共享内存兜底档 `(64,32,16)` 正好落在这条线下面
（`64×16=1024`），属于同一类。修法只有抬 `TILE_K`（抬 M/N 会破坏整除性）。

同理，`NUM_TILE_K = K / TILE_K` 是定长循环、无边界检查 ⇒ `TILE_K` 不整除 K 时是**静默丢尾**。
两者都写进了 `legal()`。

### 10.3 规则（`src/runtime.rs::gemm_tile_for`）

1. 整除收缩（原样保留，安全条件）；
2. **结构下界**：不合法就按 2× 抬 `TILE_K`（要求仍整除 K、仍放进共享内存），抬不动就
   `log::warn` 喊出来而不是静默出零；
3. **占用率**：`blocks = ⌈M/TILE_M⌉·⌈N/TILE_N⌋ < GEMM_MIN_BLOCKS`（默认 **32**，`0` 关掉）时，
   按 (64,32) → (64,64) → (128,32) 的顺序取**第一个仍整除形状且合法**的档，`TILE_K` 取
   「满足下界的最小档」（更大的 K 让每块 K 循环更深，实测更慢）。

★ 一并修掉一个**会错结果的耦合**：`gemm_grid` 过去用 `gemm_tile_for(app, m, n, 0)`（传 k=0），
旧注释的理由是「TILE_M/TILE_N 只依赖 M/N」。新规则里 `TILE_N` 受 `TILE_N*TILE_K ≥ 2048` 与
占用率档约束，**两个都依赖 k** ⇒ grid 与 spec 会分叉（表现为输出少写/多写若干列）。已把
四个 `gemm` 调用点的真实 `k` 传进去。

### 10.4 A/B：同一个二进制，只翻 `GEMM_MIN_BLOCKS`（交错 + 反序，`C:/tmp/s31tile_ab.out`）

| 臂（形状→块数） | off（旧规则） | on（默认 32） | push（32→门限 32 也切二级） |
|---|---|---|---|
| `lr1_v` N=64（1→4 块） | 0.0860/0.0845/0.0853/0.0834 | **0.0288/0.0297/0.0295/0.0308** | 0.0313/0.0441/0.0309/0.0306 |
| `lr1_w` N=128（1→8 块） | 0.1312/0.1314/0.1307/0.1305 | **0.0290/0.0293/0.0294/0.0310** | 0.0277/0.0312/0.0284/0.0312 |
| `lr1_g` N=320（5→20 块） | 0.0862/0.0835/0.0951/0.0831 | **0.0300/0.0306/0.0281/0.0294** | 0.0297/0.0285/0.0286/0.0288 |
| `lr_g` 二级 g（20 块，不变） | 0.0234/0.0233/0.0243/0.0251 | 0.0249/0.0239/0.0254/0.0234 | 0.0174/0.0217/0.0181/0.0175 |
| `lrb_w` 二级 w | 0.0190/0.0206/0.0193/0.0190 | 0.0191/0.0203/0.0219/0.0223 | 0.0163/0.0211/0.0166/0.0178 |
| `lrb_v` 二级 v | 0.0165/0.0158/0.0222/0.0165 | 0.0162/0.0172/0.0169/0.0164 | 0.0151/0.0174/0.0151/0.0164 |
| `pf_key`（80 块） | 0.3743/0.3734/0.3739/0.3721 | 0.3819/0.3812/0.3826/0.3805 | 0.3865/0.3843/0.3866/0.3852 |
| `pf_value`（20 块） | 0.5356/0.5271/0.5295/0.5258 | 0.5167/0.5456/0.5177/0.5185 | **0.2672/0.2768/0.2687/0.2672** |

读法：一级三条链 **2.9~4.5×**（四链 ≈0.43 → **≈0.118 ms/层**，比 SIMT 的 0.274 还快 **2.3×**）；
二级不劣化；`pf_key`（80 块）**切细反而慢 3 %** ⇒ 门限必须 ≤ 80，可用区间 (20, 80]，取 32；
`pf_value`（深 K 窄 N，20 块）切细 **1.94×** —— 与 S13/#34 是同一个病，但真实批量 prefill 的
M 是 chunk 行（≈4096，640 块），所以这条**在 kernel 层成立、端到端暂时吃不到**，登记为
#34 的线索而不是成果。

### 10.5 端到端门控（B=128 decode，交错 + 反序，`C:/tmp/s31e2e.out`）

| 轮 | off（旧） | on（默认 32） | 位移 |
|---|---|---|---|
| r1 | 2362.5 | 2377.8 | +0.65 % |
| r2 | 2326.9 | 2322.8 | −0.18 % |
| rev | 2283.8 | 2327.1 | +1.90 % |

均值 ≈ **+0.8 %**，prefill 段在两档之间无系统性差异（26.0~26.8 s 全是热漂）。量级与
§10.4 相符：二级四条 GEMM 只占 ≈3.4 ms/步，−10~−27 % 折算 0.3~0.9 ms/步。
**结论：规则本身不是收益主体，它是把一级搬上张量核的前置条件**（否则搬上去就是 1 个块、
比 SIMT 还慢 —— 正是 #25 当年踩的坑）。

质量门：`cargo fmt --all`、`cargo clippy --release --all-targets -- -D warnings` 干净；
`cargo test --release` 全绿（`runtime::tests::gemm_bias_tanh_lowrank`、`gemm_determinism`、
`gemm_ffn_k10240` 覆盖了两条新档的输出正确性）。

### 10.6 下一步（①-b，尚未做）

把 decode 的一级 mid 臂从 `gemv_int8_rkv_stage1_batch` 搬进 `gemm`/`gemm_tanh`：
`WorkBuffers` 缺 `xw16/xa16/xg16`（`SeqBuffers` 已有，走 `to_f16_triple`），且 A 操作数要
`[lr_rows, C]` 的补长副本；w 链用现成的 `gemm_f16_f32_tanh`，g 链的 sigmoid 留在原处。
预期收益 ≈0.142 ms/层 × 32 层 ≈ **4.5 ms/步**（B=128，54 ms 基线）。
