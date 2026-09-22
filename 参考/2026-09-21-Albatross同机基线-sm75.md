# Albatross 同机基线（RTX 2080 Ti / sm_75，2026-09-21）

> 目的：给「超越信天翁」补上**裁判**。此前所有判断都基于源码事实，没有同机同模型的实测
> （上游 `faster3a_2607` 仓库不含性能表，见[全流程精细对比](2026-09-21-Albatross-faster3a_2607-全流程精细对比.md) §9）。
>
> 关联：[跨架构基线表](2026-09-21-跨架构基线表.md) · [batch线性层第二代实施记录](2026-09-21-batch线性层第二代实施记录.md) ·
> 计划：[rwkv-rsv超越信天翁-跨架构批量解码计划.md](file:///c:/work/ai00-x-dev/.trae/documents/rwkv-rsv超越信天翁-跨架构批量解码计划.md)

---

## 0. 一句话结论

**在 2080 Ti 上，B=1 双方持平（94.0 vs 86.2 tok/s），B=256 上游是我们的 6.9 倍
（2874.3 vs 419.2 tok/s）。差距是纯粹的「并发摊薄」差距，不是单点内核差距。**

---

## 1. 环境与复现（Windows + Turing 专属的坑，全部实录）

| 项 | 值 |
|---|---|
| 上游 | `https://github.com/BlinkDL/Albatross` 的 `faster3a_2607/`（`--depth 1 --filter=blob:none --sparse` 检出） |
| 本地路径 | `C:\work\albatross\faster3a_2607`（**工作区之外**，不污染 ai00-x-dev） |
| 模型 | `C:\Users\cgisk\Downloads\rwkv7-g1j-2.9b-20260831-ctx16384.pth`（6.0 GB，正是 `rwkv7-3B-int8.st` 的源头） |
| python | `uv venv --python 3.12` + `torch==2.6.0+cu124` + `ninja` + `numpy` |
| nvcc | CUDA 12.8.61（`-res-usage` / `--extra-device-vectorization` 在 12.9+ 已移除，**12.8 恰好可用**） |
| 探针 | `bench_local.py`（照抄 `app4b.py::_generate_batch_text` 的 prefill/decode/计时口径）+ `run_bench.bat`（环境包装） |

**踩坑留档（三个都会静默失败，勿重试原路径）**：

1. **必须先 `set_wkv_mode()` 再 `RWKV7()`**。`load_extensions()` 由 `set_wkv_mode` 触发
   （`rwkv7_fast_v3a.py:2173`）；直接建模会得到 `torch.ops.rwkv7_v3a_ops` 下的
   `AttributeError`——但**命名空间存在**（`torch.ops.<name>` 是惰性创建的），极易误判成"算子没编进去"。
2. **Windows 链接必须补 cuBLAS/cuBLASLt**。上游 `load()` 没传 `libraries`，Linux 上由
   `libtorch_cuda` 传递；Windows 直接 `LNK2019: unresolved external symbol cublasGemmEx`（12 个未解析）。
   做法：在 `import rwkv7_fast_v3a` **之前**劫持 `torch.utils.cpp_extension.load`，
   给 `extra_ldflags` 追加 `cublas.lib cublasLt.lib`（ninja 的链接行本来就带
   `/LIBPATH:<CUDA>\lib\x64`）。**不改上游源码**。
3. **代理 shell 会把 `ProgramFiles(x86)` 污染成 `C:\Program Files (x86)(x86)`**，`vcvars64.bat`
   因此定位不到 Windows SDK ⇒ `cl` 报 `fatal error C1083: Cannot open include file: 'stddef.h'/'corecrt.h'`。
   同时 PATH 缺 `System32`（`where`、`DOSKEY` 都不认）。修法：包装 bat 里先改回
   `set "ProgramFiles(x86)=C:\Program Files (x86)"` 并补 `System32`，再 `call vcvars64.bat`。

### ★ 最重要的环境发现：**上游 fp16 路线在 Turing 上编不出来**

`nvcc` 直接报：

```
ptxas ... error : Feature 'cp.async' requires .target sm_80 or higher
ptxas ... error : Feature 'cp.async.commit_group' requires .target sm_80 or higher
```

全目录 grep：`cp.async` 只出现在 `cuda/rwkv7_wkv_fp16_v2.cu`（11 处），且
**`cuda/` 下零个 `__CUDA_ARCH__` 守卫**（`grep -c "__CUDA_ARCH__"` = 0）。

> ⇒ 上游默认的 `WKV_MODE="fp16"`（fp16 状态 + `elapsed` 抖动计数器）**只能在 sm_80+ 用**。
> 在 2080 Ti 上只能用 `WKV_MODE="fp32io16"`（`rwkv7_wkv_fp32_v2.cu`，无 cp.async），
> 而这一档的 **wkv 状态是 fp32**（`zero_state`：`torch.float32 if WKV_MODE == "fp32io16"`，
> `rwkv7_fast_v3a.py:2404/2409`）——**它省显存的那一招在 Turing 上直接失效**。
> （旁证：`app4b.py::_generate_batch_text` 的 `wkv_mode` 默认值也是 `"fp32io16"`。）

---

## 2. 测量结果（同会话、同一台机、同一份权重）

`--wkv-mode fp32io16`；prompt 128 tok（B=1 prefill 后 `copy_state_to_batch` 展开到 B 宽）；
解码 32 步。**上游口径** = 含 host 侧采样与 `.cpu().tolist()`（照抄 app4b 的计时区间）；
**GPU-only** = 只 replay，不采样。

| B | 上游口径 tok/s | GPU-only tok/s | 上游每槽 ms/步 | GPU-only 每槽 ms/步 | VRAM used |
|---|---|---|---|---|---|
| 1 | 94.0 | 96.0 | 10.633 | 10.420 | 6.78 G |
| 8 | 447.1 | 427.9 | 2.236 | 2.337 | 6.99 G |
| 64 | 2018.6 | 2086.1 | 0.495 | 0.479 | 8.07 G |
| **256** | **2874.3** | **2963.9** | **0.348** | **0.337** | **12.03 G** |

> 两档之差 ≈3% ⇒ **host 采样不是变量**，上表就是 GPU 真实速度。
> 权重加载后常驻 5.37 GB（fp16 双布局：orig + transposed）/ used 6.65 GB。

**我们的同会话对照**（`batch_decode_bench`，PAD_TO=160 / SEGS=2 / NTOK=32 / TOPK=50，各槽不同 prompt）：

| B | rwkv-rsv tok/s | 每槽 ms/步 | VRAM（登记张量） |
|---|---|---|---|
| 单流 | 86.2 | 11.60 | — |
| 8 | 322.0 | 3.105 | 5940 MiB |
| 64 | 394.4 | 2.535 | 7315 MiB |
| **256** | **419.2** | **2.386** | 11534 MiB |

### 2.1 差距全貌

| B | 上游 tok/s | 我们 tok/s | 上游/我们 |
|---|---|---|---|
| 1 | 94.0 | 86.2（单流） | **1.09×** |
| 8 | 447.1 | 322.0 | 1.39× |
| 64 | 2018.6 | 394.4 | 5.12× |
| **256** | **2874.3** | **419.2** | **6.86×** |

**逐槽成本随 B 的收缩倍数**（B=8 → B=256）：

| | 每槽 ms/步 B=8 | B=256 | 收缩 |
|---|---|---|---|
| 上游 | 2.236 | 0.348 | **6.4×** |
| 我们 | 3.105 | 2.386 | **1.30×** |

---

## 3. 判读：差距 = 计算单元，不是带宽，不是几何

### 3.1 B=1 持平 ⇒ 单点内核不是问题

B=1 时双方每步 10.4~11.6 ms，几乎相同。**我们的 kernel 结构、融合度、访存都是合格的**
——这与 [batch线性层第二代实施记录 §8](2026-09-21-batch线性层第二代实施记录.md) 的结论一致：
我们的问题是"每槽成本不随 B 摊薄"。

### 3.2 用 FLOP 口径看，差距是**算力档位**差

RWKV-7 2.9B ≈ 2.9 G MAC/token。

| | B=256 每步实际算力 | 峰值 | 占比 |
|---|---|---|---|
| 上游 | 256×2.9G×2 = 1.48 TFLOP / 86.4 ms = **17.1 TFLOPS** | fp16 TC ~107（FC ~53） | ~32%（对 FC 峰值） |
| 我们 | 1.48 TFLOP / 610.8 ms = **2.42 TFLOPS** | fp32 SIMT 13.45 | ~18% |

**17.1 / 2.42 = 7.1×**，与端到端 6.86× 几乎相等。
⇒ 差距**全部来自计算载体的吞吐档位**：上游走 **fp16 张量核（rows≤16 WMMA /
rows 8~2047 cuBLAS `CUBLAS_GEMM_DEFAULT_TENSOR_OP`）**，我们走
**int8 反量化 + fp16 SIMT（`hfma2`）**。int8 常驻在这里**没有换来任何算力优势**
——反量化本身还要再花约 12 条 fp32 指令/4 权重（见实施记录 §8.3）。

### 3.3 带宽口径：两边都远未饱和，但差的倍数不同

| | 每步权重字节 | 带宽下限@500GB/s | 实测每步 | 距离下限 |
|---|---|---|---|---|
| 上游（fp16） | 5.4 GB | 10.8 ms | 86.4 ms | 8× |
| 我们（int8） | 2.1 GB | 4.2 ms | 610.8 ms | 145× |

⇒ 我们的 int8 只省了字节，**没省时间**；上游用两倍字节换来 7 倍速度。

### 3.4 显存：两边的"省"不是同一回事

上游在 sm_80+ 靠 fp16 状态省一半；**在本机（sm_75）只能 fp32 状态**，B=256 用 12.03 GB，
我们 11.53 GB ⇒ **Turing 上显存其实是我们略省**。
（上游真省的部分是 `EMB_DEVICE="cpu"` + prefill B=1 后 `copy_state_to_batch`——
后者我们已用 `PREFILL_SLOTS` 复刻，见实施记录 §6.3。）

---

## 4. 对「超越」的含义（下一步）

**验收基准（同机同模型，2080 Ti / fp32io16 / B=256）：2874.3 tok/s。**
我们目前 419.2，需要 **6.86×**。按 §3.2，这个倍数只能从**计算载体**来：

1. **fp16 权重副本 + WMMA/cuBLAS**（对齐上游）：直接拿到 TC 档位；代价 +5.4 GB 显存，
   违反 D1 的 8GB 下限，只能在显存充裕档启用，8GB 档保留 int8 路径。
2. **int8 IMMA `m8n8k16`**（Turing 原生 215 TOPS）：保住 8GB 下限与 int8 常驻，
   但激活需量化到 int8（W8A8）⇒ 改数值口径，需重跑数值门禁与输出质量验证。
3. 已排除：继续调 int8 SIMT 内核几何（实施记录 §8.3 五点扫描全部为负）。

> 注：上游在本机还吃了个亏——**没有 CUTLASS**（`CUTLASS_INCLUDE_DIR` 不存在，
> `rows` 特化档全部回落 cuBLAS），且用的是非最优的 `fp32io16`。
> 也就是说 **2874.3 是上游的"残血"成绩**，不是它的上限。