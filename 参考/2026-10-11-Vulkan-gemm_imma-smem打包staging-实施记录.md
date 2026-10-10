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
3. **同一论证适用于 prefill 的 fp16 `gemm`**：元素 2 字节 ⇒ 每字 2 个，staging 指令数可减半；
   这条与 §6.6/§6.7 的 prefill 位次（0.79×）直接相关，未做。
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
