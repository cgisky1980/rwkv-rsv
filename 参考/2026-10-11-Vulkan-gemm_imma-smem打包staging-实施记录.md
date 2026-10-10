# Vulkan `gemm_imma` smem 打包 staging（2026-10-11）

**一句话**：`gemm_imma` 的开销不由「权重重读」决定，而由每 block 每 k-tile 的 **smem 落盘指令数**决定；
把 smem 从按字节存改成按 uint 存（4 字节/条 `STS`）后，四个 W8A8 形状 **1.13~1.31×** 提速，
端到端 decode **+2.4 %（B=8）/ +5.2 %（B=32）/ +7.2 %（B=128）**，输出**逐位不变**。

- 后端：Vulkan（`assets/shaders/src/gemm_imma.comp`）
- 硬件：RTX 2080 Ti（Turing，68 SM，int8 张量核）
- 门禁：`cargo fmt --all` / `cargo clippy --all-targets --release -- -D warnings` / `cargo test --release` 全绿
- 结论表：README §6.8；批量解码总表：README §6.1

---

## 一、起点：#33 之后的 profile 说的是 `gemm_imma`

§6.3/§6.4/§6.5 之后 B=128 decode 仍然只有 **0.473×** CUDA（README §6.1，同会话交错、2 轮中位数、
离散度 ≤2.2 %）。同会话对 Vulkan B=128 自循环做 `PROF_GPU=1`：

| kernel | 占比 |
|---|---:|
| `gemm_imma` | **41 %** |
| `gemv_lowrank_chain4_batch` | 18 % |
| `fuse_ka_dplr_norm` | 16 % |
| `gemv_int8_rkv_stage1_batch` | 12 % |
| `gemm_imma_reduce` | 5 % |
| `quant_x_i8` | 2 % |

（`SUM 170.6 ms over 2048 kernels`。⚠️ 口径提醒：`kernel_bench.rs` 头部就写着 `PROF_GPU` 在自循环段
只记录到一部分 dispatch，所以这张表用来**选靶子**，不能用来算 A/B 增益。）

## 二、定位：内核级对拍，CUDA 侧的读数有个坑

`kernel_bench` 直接对四个 W8A8 形状计时（量化 + GEMM 成对计时）。**坑**：
`ComputeBackend::supports_int8_imma()` 只有 Vulkan 返回 true（`backend.rs:473` 默认 false），
所以 `KB_OPS=imma_*` 在 CUDA 上会被整族跳过——而 CUDA 的 IMMA 其实一直在跑，只是藏在
`gemv_int8_*` 里面（`IMMA_MIN_BATCH = 1`、`GEMV_IMMA` 默认开，`backend_cuda.rs:1443`）。
正确的跨后端口径是**用 SIMT 算子名去读 CUDA**：`relu2`=key、`mul_add`=out、`add`=value、`plain`=head。

B=128，同一形状（ms/call，两后端各自 5 轮取最小）：

| 形状 m×k | Vulkan（改前） | CUDA | 位次 |
|---|---:|---:|---|
| key 10240×2560 | 0.2982 | 0.1816 | 1.64× 慢 |
| value 2560×10240 | 0.3229 | 0.1714 | 1.88× 慢 |
| head 65536×2560 | 1.6596 | 0.8439 | 1.97× 慢 |
| out 2560×2560 | 0.0990 | 0.0706 | 1.40× 慢 |

TOPS：Vulkan 17~26，CUDA 24~51（fp16 张量峰值 ≈54 TOPS，int8 是其 2×）。

## 三、先复现「假设失败」：NST=16 的权重重读实验

`imma_nst` 的自动规则取 `ceil(batch/8)` 的 2 的幂并**压在 8**（构建期只编了 1/2/4/8 档），
⇒ 一个 block 盖 `8·NST = 64` 槽 ⇒ **B=128 权重读两遍、B=256 读四遍**。这看起来是 §6.1 未结项 3
（「每一波重读整套 int8 权重」）的直接证据，于是先做最便宜的那一手：`build.rs` 加
`def(16, "nst16")`，只把**显式 `VK_IMMA_NST` 分支**的 clamp 抬到 16（自动规则不动 ⇒ 默认行为零变化）。

实测（B=128 / B=256，ms/call）：

| 档位 | key B=128 | head B=128 | key B=256 | head B=256 |
|---|---:|---:|---:|---:|
| 默认 `bm64/nst8` | 0.2982 | 1.6596 | 0.5735 | 3.3022 |
| `nst16`（权重单读） | 0.3209 | 1.8727 | 0.6604 | 3.6304 |

**慢 7.6 ~ 12.8 %**，四档扫描全表（B=128/256）：

| 臂 | key B=256 | value B=256 | head B=256 |
|---|---:|---:|---:|
| 默认 | 0.5735 | 0.6051 | 3.3022 |
| `VK_IMMA_BM=128` | 0.8472 | 0.7579 | 3.9956 |
| `VK_IMMA_NST=4` | 0.7545 | 0.8098 | 4.2986 |
| `VK_IMMA_NST=2` | 1.2169 | 1.2869 | 7.6176 |
| `VK_IMMA_KSPLIT=2` | 0.5987 | 0.5952 | 3.4140 |

⇒ 默认档就是整个瓦片空间的顶点；「重读」不是瓶颈（CUDA 侧同源证据：`backend_cuda.rs ≈1866` 的
`IMMA_DB`/`IM_PIPE` 两条隐藏延迟的路都实测无效）。**真正的信号是 `NST` 2→8 慢 2.07×**，
而那段区间里块数是**下降**的——只与「每槽摊到的 staging 字节数」同向。

## 四、真因：staging 是**按字节**写的

每 block 每 k-tile 要落 `(BM + 8·NST)·BK` 字节；`BM=64/NST=8` 档 = 8192 B 权重 + 8192 B 激活 = 16 KiB。
旧 `store_packed4` 把一个 `uint`（4 个量化字节）拆成 **4 条 `STS.u8`** ⇒
**每 block 每 k-tile ≈ 512 条 warp 级 store，而同期的 `mma` 只有 128 条**（8 warp × 16）。
取数/落盘指令才是发射瓶颈。

## 五、修复：smem 按 uint 存 + packed `coopMatLoad`

```glsl
const uint WPU = WS / 4u;          // 行跨距按 uint 计（WS=144 ⇒ 36）
shared uint ws[BM * WPU];          // 旧：shared int8_t ws[BM * WS]
shared uint xs[SLOT * WPU];
...
ws[(i >> 5u) * WPU + (i & 31u)] = wr[n] ^ 0x80808080u;   // 一条 STS.u32
...
coopMatLoad(a0, ws, wrow + 0u, WPU, gl_CooperativeMatrixLayoutRowMajor);   // 偏移 0/8/16/24
```

要点（都是可复现的口径，写下来免得下次再查）：

1. **GLSL 允许 8 位 coopmat 从 32 位标量数组打包载入**（`glslangValidator -V -G` 接受；
   SPIR-V 侧就是 `CooperativeMatrixLoadKHR` 的 storage type ≠ component type + pack=4）。
   此时 **`stride` 与 `offset` 一律按 `uint` 计**：一个 32 字节的 k 切片 = 8 个 uint ⇒ 偏移
   从 `0/32/64/96` 变 `0/8/16/24`，行跨距从 `WS` 变 `WPU`。
2. 打包方向沿**主轴**：RowMajor 的 A 沿行（= k）打包 ✓，ColumnMajor 的 B 沿列（也是 k）✓，
   与 `xq`/权重的全局布局（4 个连续 k 一个 `uint`）天然一致，所以**读回也是每 lane 一条 `LDS.u32`**
   （coopmat 的 int8 片段本来就是每 lane 4 个连续字节）。
3. `^0x80`（无符号量化域下移到有符号）从「逐字节」变成「整字一次」。
4. 诊断臂 `OP=9/10`（`examples/imma_probe.rs` 用它定位过三处索引 bug）要跟着改：按字节取回
   `PACKED_BYTE(arr, byte)`。写成**宏**而不是函数——GLSL 的数组形参必须带尺寸，而 `ws`/`xs`
   的尺寸按档位不同（这是本次唯一一次编译失败：`array size required`）。
5. **累加顺序不变**（每个输出元素的 k 循环顺序、每组 4 次 `mma` 的顺序都没动）⇒ 理论上就应当逐位一致，
   实测也一致（见下）。

## 六、验证

内核级（B=128 / B=256，`sum(y)` 与改前**逐位相同**）：

| 形状 | B=128 改前→改后 | B=256 改前→改后 | 改后 vs CUDA |
|---|---|---|---|
| key | 0.2982 → **0.2492**（1.20×） | 0.5735 → **0.4757**（1.21×） | 1.37× 慢 |
| value | 0.3229 → **0.2675**（1.21×） | 0.6051 → **0.5117**（1.18×） | 1.56× 慢 |
| head | 1.6596 → **1.2682**（1.31×） | 3.3022 → **2.5209**（1.31×） | 1.50× 慢 |
| out | 0.0990 → **0.0876**（1.13×） | 0.1900 → **0.1640**（1.16×） | 1.24× 慢 |

端到端（`bench_ab`，同会话交错、2 轮中位数、门控 `SEGS=4 PAD_TO=512 NTOK=32 WARMUP=1 SKIP_SINGLE=1`
`prompt_repeat=6 temp=1 top_k=0`，会话 `20261011-043110-89384`）：

| B | CUDA | 离散度 | Vulkan | 离散度 | Vulkan/CUDA | 改前 |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 585.4 | 1.2 % | 512.1 | 0.8 % | **0.875** | 0.847 |
| 32 | 1782.0 | 1.2 % | 1275.2 | 0.4 % | **0.716** | 0.678 |
| 128 | 3426.6 | 1.4 % | 1739.7 | 1.4 % | **0.508** | 0.473 |

**逐位性证据（关键）**：Vulkan 三档 decode 指纹与会话 `20261011-040726-67976`（改前）**完全一致**——
`0xd0c265`/`0x7b05`（B=8）、`0x3460382`/`0x16fa`（B=32）、`0xd1b5310`/`0x7bfc`（B=128），两轮也各自一致。
跨后端「分叉」是既有的 CUDA 批形换 kernel 现象（README §6.7），不是本次引入。

门禁：fmt 0 / clippy 0（`touch` 强制重跑确认非缓存）/ `cargo test --release` = 78+11+1 通过、3 ignored。

## 七、未结（按大小排）

1. **split-K 规则是在旧 staging 上调的，现在对 `value` 明显失配**：B=256 下自动档（`ks=8`）0.5117 ms，
   强制 `VK_IMMA_KSPLIT=2` 0.4617 ms（+11 %）；key/head 基本持平。要单独一次 ks 扫描 + 端到端确认，
   不要和本次混在一起（`ks>1` 会改求和顺序 ⇒ 不再逐位）。
2. **剩下的 1.2~1.6× 在循环内部**（片段载入/epilogue 结构、`tile` smem 往返、bank 冲突），
   不在瓦片形状上——§6.4（A 片段寄存器复用 + 槽组分 warp）是同一层的下一个候选做法。
3. ~~**同一论证适用于 prefill 的 fp16 `gemm`**~~ —— **已核对，作废（2026-10-11）**：
   `assets/shaders/src/gemm.comp` 的 staging 本来就是 `shared uvec4 sh_a/sh_b`（每次 STS 落 16 字节），
   打包这条杠杆在 prefill 路径上**不存在**；§6.6/§6.7 的 0.79× 位次是另一套机制（任务 #34）。
4. `NST=16` 的 spv 档位**保留在构建里**（只作为探针，默认选不到），并在 `Runtime::imma_nst` 的
   文档里写明已否决——避免下次再花一轮 GPU 时间去试同一条死路。

## 八、复现命令

```bash
# 内核级（CUDA 侧要用 SIMT 算子名，见 §二）
KB_OPS=imma_key,imma_out,imma_value,imma_head KB_BATCH=128,256 BACKEND=vulkan \
  ./target/release/examples/kernel_bench.exe
KB_OPS=relu2,mul_add,add,plain KB_BATCH=128,256 BACKEND=cuda \
  ./target/release/examples/kernel_bench.exe
# 瓦片扫描（每个臂一次进程，串行）
VK_IMMA_NST=16 KB_OPS=imma_key,imma_value,imma_head KB_BATCH=256 BACKEND=vulkan \
  ./target/release/examples/kernel_bench.exe
# 端到端 + 指纹
AB_BACKENDS=cuda,vulkan AB_SLOTS=8,32,128 AB_ROUNDS=2 SEGS=4 PAD_TO=512 NTOK=32 WARMUP=1 SKIP_SINGLE=1 \
  ./target/release/examples/bench_ab.exe
```

改动文件：`assets/shaders/src/gemm_imma.comp`、`assets/shaders/spv/gemm_imma_bm{64,128}_nst{1,2,4,8}.spv`
（+ 新增 `..._nst16.spv`）、`build.rs`、`src/runtime.rs`、`README.md`。

---

# 九、后续（同日）：split-K 规则重定 —— 第七节未结项 1

`imma_ksplit` 的占用目标过去只数 **`M/BM`**，漏掉 grid 的第三维
`grid.z = ceil(batch / (8·NST))`。旧 staging 下这个漏算被别的东西盖住了；打包之后每 block
便宜了 1.2~1.3 倍，于是「batch 越大越该少切」这件事第一次变成了可测的失配：
batch 上去以后并发已经被 grid.z 拿到，继续 split-K 只多付 partial 写出 + 归约那一趟
（代价随 `ks × batch × M` 涨）。

## 9.1 逐档实测（`VK_IMMA_KSPLIT` 强制，packed staging，单位 ms）

三个形状 × 八档 batch；`imma_head` 见 9.3 的对照说明。B=1…16 一组、B=32…256 一组
（各组内部串行、进程独立）。加粗 = 该 batch 的实测最优 ks。

**key（m=10240, k=2560，`M/BM` = 160 块）**

| ks | B=1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0785 | 0.0791 | 0.0790 | **0.0845** | 0.1061 | **0.1267** | **0.2111** | **0.4195** |
| 2 | **0.0721** | **0.0746** | **0.0776** | 0.0855 | **0.0995** | 0.1319 | 0.2553 | 0.4904 |
| 4 | 0.0732 | 0.0770 | 0.0809 | 0.0915 | 0.1098 | 0.1527 | 0.2888 | 0.5615 |
| 8 | 0.0737 | 0.0787 | 0.0838 | 0.0936 | 0.1138 | 0.1594 | 0.3060 | 0.5958 |

**out = rkv 的 r/k（m=2560, k=2560，40 块）**

| ks | B=1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.0281 | 0.0285 | 0.0294 | 0.0314 | 0.0404 | 0.0566 | **0.0805** | 0.1394 |
| 2 | 0.0264 | 0.0264 | 0.0266 | 0.0300 | 0.0378 | 0.0518 | **0.0795** | **0.1363** |
| 4 | **0.0244** | 0.0251 | 0.0265 | 0.0290 | 0.0350 | 0.0487 | 0.0848 | 0.1586 |
| 8 | **0.0244** | 0.0254 | **0.0260** | **0.0277** | **0.0344** | **0.0465** | 0.0895 | 0.1720 |

**value（m=2560, k=10240，40 块，`K/128` = 80 段）**

| ks | B=1 | 4 | 8 | 16 | 32 | 64 | 128 | 256 |
|---|---|---|---|---|---|---|---|---|
| 1 | 0.1040 | 0.1024 | 0.1035 | 0.1115 | 0.1264 | 0.2146 | 0.2954 | 0.5019 |
| 2 | 0.0894 | 0.0933 | 0.0931 | 0.1019 | 0.1173 | 0.1564 | 0.2649 | **0.4619** |
| 4 | 0.0809 | 0.0833 | 0.0861 | 0.0935 | 0.1081 | 0.1473 | **0.2547** | 0.4970 |
| 8 | **0.0746** | **0.0772** | **0.0803** | **0.0895** | **0.1030** | **0.1379** | 0.2745 | 0.5208 |

形状层面的结论是一句话：**optimal 全 grid 块数（`M/BM × grid.z × ks`）≈ 320**，
而不是「`M/BM` 凑到 400」。把 9.1 三张表每一格换算成全 grid 块数，最优档落在
160~320（B 小）与 320~640（B 大）之间，中心就是 320 ≈ 4.7×SM。

## 9.2 目标档扫描（旧规则口径，只改 `VK_IMMA_KSPLIT_TARGET`）

用来确认「320」不是从 9.1 反推出来的循环论证——这一组是独立扫的（B=256，ms）：

| 形状 | target 100 | 160 | 200 | 260 | 400（旧默认） |
|---|---|---|---|---|---|
| out | **0.1367** | 0.1589 | 0.1677 | 0.1663 | 0.1696 |
| value | **0.4522** | 0.4851 | 0.4893 | 0.4909 | 0.5208 |
| key | **0.4145** | **0.4210** | 0.4846 | 0.4856 | 0.4903 |
| head | 2.5188 | 2.5290 | 2.5433 | 2.5552 | 2.5643 |

（head 在 100…400 全部 ks=1，所以这一行整行都是漂移，见 9.3。）

## 9.3 口径与漂移

- 每一组 ks 扫描里，**四个臂是四个独立进程、按 1→2→4→8 的顺序串行**。同会话内
  `imma_head` 的散布：B=32 0.5085→0.5162（1.5 %）、B=256 2.5305→2.5900（2.4 %），
  且**单调**——即卡在这一串进程里缓慢升温/被桌面负载拖住。⇒ 读数里 ≤2.4 % 的部分
  不能算规则的效果；9.1/9.2 里 7~20 % 的档位差扣掉漂移后方向不变。
- 更要紧的是**跨会话不可比**：§6.8 表里 packed 后的 head B=128 是 1.2682 ms，本次同形状
  复测 1.2225 ms（head 在两规则下都是 ks=1，没有任何代码差异）⇒ 3.6 % 是会话间漂移。
  所以本节只在**同一组扫描内部**比档，绝对值一律以端到端 A/B 为准（9.5）。
- `imma_head`（M=65536）在新旧规则下恒 ks=1（1024 块 ≥ 任何目标），它同时充当
  「规则没改动到别的东西」的哨兵。

## 9.4 新规则与它选出的档

`imma_ksplit(m, k, bm, nst, cap, batch)`：`blocks = (M/BM) × ceil(batch/(8·NST))`，
`ks = ceil(320 / blocks)`，上限 `VK_IMMA_KSPLIT_MAX`=8 不变，
`(K/128) % ks == 0` 与 partial 容量收口不变。

对照 9.1 的 24 个「形状 × batch」组合：**20 个正好命中实测最优 ks**；
没命中的 4 个是 key B=16（选 2、优 1，+1.2 %）、key B=64（选 2、优 1，+4.1 %）、
out B=4（选 8、优 4，+1.2 %）、out B=128（选 4、优 2，+6.7 %）。
**没有任何一组比旧规则慢**（旧规则在这 4 组上要么给同一个 ks，要么更差）。
换到新规则的收益集中在 B≥128：key −17 %、out −20 %、value −7~−11 %（各自组内）。

⚠️ ks 变了 ⇒ 求和结合顺序变 ⇒ **落盘字节不再是 §6.8 那一组指纹**。数值契约不变
（同样的 int8 量化、同样 fp32 累加，只是加法顺序），与 v 臂接入 split-K 时同一条声明。
门：`cargo test --release --test batch_parity` 10 passed / 0 failed / 1 ignored（已知
`stochastic_reproducibility_repro`），其中 `cross_backend_batch_paths_agree`、
`slots_beyond_the_first_wave_match`、`greedy_tokens_are_reproducible` 直接覆盖这条契约。

## 9.5 端到端（同会话交错 A/B，会话 `20261011-045748-40368`，3 轮取中位数）

门控与 §6.1 那两次完全同一份快照（`SEGS=4 PAD_TO=512 NTOK=32 WARMUP=1 SKIP_SINGLE=1
prompt_repeat=6 temp=1 top_k=0`；CUDA 三个张量核开关均未设=开；`VK_IMMA_OPS` 未设=全开含 rkv）。

| 并发 B | CUDA tok/s（离散度） | Vulkan tok/s（离散度） | Vulkan/CUDA | 上一次（§6.8 后） |
|---:|---:|---:|---:|---:|
| 8 | 617.7（1.4 %） | 536.5（0.7 %） | **0.868** | 0.875 |
| 32 | 1869.8（1.5 %） | 1330.4（1.6 %） | **0.711** | 0.716 |
| 128 | 3587.6（2.0 %） | 1881.8（1.9 %） | **0.525** | 0.508 |

三个后端**同时**比 §6.8 那次会话快 ~5 %（CUDA 585.4→617.7、3426.6→3587.6）⇒ 会话整体偏冷，
比值才是可比的量。B=128 的 Vulkan 涨 8.2 % 而 CUDA 涨 4.7 % ⇒ 净 **+3.3 % 位次**。

**为什么只有 B=128 动**：新规则在 B=8/32 对每个形状选出的 ks 与旧规则**逐一相同**
（`out`：`ceil(320/40)=8` 被整除收口拖回 5，和旧的 `ceil(400/40)=10→8→…→5` 同一档；
`value`：80 段整除 8 ⇒ 仍是 8；`key`：`ceil(320/160)=2` ⇒ 仍是 2）。
指纹把这件事钉死了：**B=8/32 两档的 Vulkan decode 指纹与 §6.8 会话逐位一致**
（`0xd0c265`/`0x7b05`、`0x3460382`/`0x16fa`），所以那两行的 ±0.7 % 纯属漂移。
B=128 变了（`key` 2→1、`out`/`rkv` 5→4、`value` 8→4）⇒ 指纹变成 `0xd21f6bb`/`0x78b3`，
**3 轮全等**（确定性没破），与 §6.5 v 臂同一条契约：量化不变、fp32 累加不变、只有结合顺序。

顺带一条观察（不写成正确性证明）：重定后 B=128 的 Vulkan 与 CUDA 的 token 和**更接近**——
`|0xd21f6bb − 0xd21f638| = 131`，而 §6.8 那次 `|0xd1b5310 − 0xd21f638| = 435 496`。
方向与「少切一段 = 少一次重排」一致；单条确定性贪心 prompt，不作结论。

## 9.6 门与结论

- `cargo fmt --all` 干净；`cargo clippy --all-targets --release -- -D warnings` 干净；
  `cargo test --release --test batch_parity` 10 passed / 0 failed / 1 ignored。
- 默认改动只在 `Runtime::imma_ksplit` 的**占用目标口径**（+ 新增 `nst` 形参），
  kernel、spv、量化、`VK_IMMA_KSPLIT_MAX` 上限 8 都没动；`VK_IMMA_KSPLIT`/`_TARGET`/`_MAX`
  三个探针口径照旧可覆盖（`_TARGET` 默认值 400 ⇒ 320）。
- 位次结论：**B=128 0.508 → 0.525**，B=8/32 不变。这条杠杆到此收口；
  `gemm_imma` 相对 CUDA 还剩 1.12~1.45×（新默认档 9.1 会话的 Vulkan 读数 ÷ §6.8 会话的 CUDA 列：
  key 0.2042/0.1816、out 0.0821/0.0706、value 0.2384/0.1714、head 1.2225/0.8439
  ——⚠️ 两列**不同会话**，按 9.3 的口径这条只能看量级，不能当逐位结论），
  这部分在循环内部，不在瓦片/分块策略上。

# 十、后续探针（同日）：A 片段直接从 global 载入 —— **已证伪，且过程里抓到一个测量方法学的坑**

第七节未结项 2 说「剩下的 1.2~1.6× 在循环内部（片段载入 / epilogue 结构 …）」。这一节把其中
**最好量化的一条**做完：既然第九节确认了「每 block 每 k-tile 的 smem staging 指令数」是代价所在，
那能不能**整段删掉权重的 staging**——让 `coopMatLoad` 直接吃 global？

## 10.1 能力事实（正面结论，值得记住）

- **glslang 接受**：`coopMatLoad(a, params.in_idx.data, base, stride, RowMajor)` 里的数组是
  `GL_EXT_buffer_reference` 指向的 `uint data[]`（Generic 指针），编译通过，产出
  `OpCooperativeMatrixLoadKHR … MakePointerVisible|NonPrivatePointer`。
- **NVIDIA 驱动也接受**：pipeline 创建成功、kernel 真的跑（`sum(y)` 非零、档位切换会改数值）。
  即「tensor core 片段直吃 global」在 Turing/450core 上**可用**，不是非法构造。
  将来 #25（低秩链 fp16 张量核）若要省掉一次 staging，这条路是通的。

## 10.2 计时：直取 global **慢 32~42 %**（同会话交错 A/B，`imma_key` m=10240 k=2560）

| batch | 基线（smem staging）ms | 探针（A 直取 global）ms | 差 |
|---:|---:|---:|---:|
| 8 | 0.0762 / 0.0793 | **0.1019 / 0.1021** | **+33 % / +29 %** |
| 128 | 0.2042 / 0.2052 | **0.2910 / 0.2914** | **+42 % / +41 %** |

两轮各测两次，读数各自内部散布 ≤2.4 %，方向一致 ⇒ 结论稳。

机理：权重侧的 global 访问形态从「整块连续、每线程 1 个字、可寄存器预取并发在飞」变成
「每 warp 16 行 × 32 字节的短段、每 lane 4 个字、由片段载入驱动」——省下的 STS/LDS 条数
远不够赔上失去的**载入并发**与**行分散度**。所以 §6.8 的「staging 指令数是瓶颈」要收窄成：
**staging 是必要代价，而不是可省代价**；要再省只能省**字节数**（打包已经做过）或**重数**
（`NST` 已实测过，见 §三），不是省这条搬运本身。

## 10.3 过程里抓到的坑（写死在纪律里）

第一轮 A/B 报出「两档 ms 差 ≤2 %、`sum(y)` **逐位相同**」。当时我按「数值档没差别 ⇒ 驱动走了同一份
SASS」去解释，正准备把「证伪」写进台账——**这是错的**：两档读数太接近本身就是探针无效的征兆。
真正的因是编译命令笔误：宏名是 `IMMA_DIRECT_A`，命令行写成了 `-DIM_DIRECT_A=1`，
于是 `_da.spv` 其实是**基线代码的另一份字节**（`md5sum` 与 `gemm_imma_bm64_nst8.spv` 完全一致）。
⇒ 第一轮的「慢 2 %」是 baseline 对 baseline 的噪声，什么都没测。

**加了一条自检**：探针档除了被测改动，再写一个**与性能无关的数值标记**（这里把 `szs[]` 全量覆写成
fp16 `0x4242`=48.06，标记在 `barrier()` 之前 ⇒ 无竞争）。判据：

1. 探针档的 `sum(y)` 必须与基线**差一个量级**（实测 97 899 659 264 → 3 863 899 275 264，×39.5）；
2. 想再稳一层就 `md5sum` 两份 spv —— 相同 ⇒ define 根本没生效。

`kernel_bench` 的 `sum(y)` 列只能证「内核跑了」，**不能**证「跑的是这一档的代码」；两档 sum 相同是
探针失效的信号，不是「数据无关」的信号。

## 10.4 处置

- 探针已全部回退：`assets/shaders/src/gemm_imma.comp`、`src/runtime.rs`（`VK_IMMA_DIRECT_A` 的
  spv 名切换）、`*_da.spv` 五份编译产物都删干净；`git status` 只剩本文档。
- 第七节未结项 2 的「片段载入」子假设关闭；剩下的候选是 **`tile` smem 往返 / epilogue 结构**
  （`coopMatStore` → `subgroupBarrier` → 逐元素读回，每 warp 每组一次）和 bank 冲突层面。
