# rwkv-rsv

[English](#english) · [中文](#中文)

RWKV-7 推理引擎（Rust + Vulkan 计算着色器），支持三路权重量化推理（fp16 / any4-4bit / int8-8bit），按模型文件自动路由。CPU fp32 参考实现用于精度验证。

RWKV-7 inference engine in Rust with Vulkan compute shaders. Supports three weight-quantization inference paths (fp16 / any4-4bit / int8-8bit) auto-routed by model file, plus a CPU fp32 reference for accuracy verification.

---

## 中文

### 特性

- **纯 Vulkan 计算着色器**推理：GEMM / GEMV / fused r-k-v / low-rank / time-shift 等全套内核，显存与延迟均可控。
- **三路量化推理路径**，按模型文件自动路由（优先级 int8 → any4 → fp16），互斥共存：
  | 路径 | 权重位宽 | 说明 | 3B 权重文件大小 |
  |---|---|---|---|
  | fp16 | 16-bit | 不量化，无损 | 5.49 GB |
  | int8 | 8-bit 非对称 per-group(128) | 近无损（teacher-forced Top-1 一致率 ~98.8%），省 41% | 3.22 GB |
  | any4 | 4-bit LUT + per-row k-means | 极省显存（省 62%），分布外精度略降 | 2.07 GB |
- **离线量化工具**：[tools/quantize_any4.py](tools/quantize_any4.py) 从 fp16 `.st` 生成 `.any4.st` / `.int8.st`（`--bits 4|8`）。
- **精度验证工作流**：`TOP1_MULTI_SAVE/COMPARE`、`TOP1_REF_SAVE/COMPARE`、`SAVE/COMPARE_LOGITS` 等 env 驱动，对比量化模型与 fp16 参考的端到端一致率。
- **参考实现**：[src/model.rs](src/model.rs) 提供 CPU fp32 反量化与单测，作为 GPU 内核正确性基准。

### 构建

需要 Rust（edition 2024）与 Vulkan SDK（`glslangValidator`，build.rs 在构建时把 `assets/shaders/src/*.comp` 编译为 `assets/shaders/spv/*.spv`）。

```bash
cargo build --release
```

### 运行

加载模型（默认 `c:\work\niceui\rwkv-g1h-3B.st`，可用 `MODEL_PATH` 切换）：

```bash
cargo run --release
```

主要 env 变量（详见 [src/main.rs](src/main.rs)）：

| 变量 | 作用 |
|---|---|
| `MODEL_PATH` | 模型路径；`.int8_idx` → int8，`.any4_idx` → any4，否则 fp16 |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | 多 prompt 的 teacher-forced Top-1 一致率验证 |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | 单 prompt 的 CPU fp32 参考一致率 |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | 单 prompt logits 的 RMSE / Top-10 对比 |
| `GEN_TOKENS` / `REPORT_EVERY` | 连续自回归生成（`memtest` 子流程） |

### 量化

```bash
# 4-bit any4
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.any4.st --bits 4
# 8-bit int8
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

校准 prompt 采集：[tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py)（多语言分层抽样 aya_dataset）。

### 目录结构

```
src/            Rust 源码（main / model / gpu_model / runtime / vulkan）
assets/shaders/ Vulkan 计算着色器源码（*.comp，spv 为构建产物）
tools/          离线量化与校准工具（Python）
参考/           研发参考文档（量化报告、shader 记录等）
test/           开发用脚本
```

---

## English

### Features

- **Pure Vulkan compute-shader** inference: full kernel set for GEMM / GEMV / fused r-k-v / low-rank / time-shift, with controllable memory and latency.
- **Three quantization inference paths**, auto-routed by model file (priority int8 → any4 → fp16), mutually exclusive:
  | Path | Weight bits | Notes | 3B weight file size |
  |---|---|---|---|
  | fp16 | 16-bit | no quantization, lossless | 5.49 GB |
  | int8 | 8-bit asymmetric per-group(128) | near-lossless (teacher-forced Top-1 agreement ~98.8%), saves 41% | 3.22 GB |
  | any4 | 4-bit LUT + per-row k-means | maximal memory savings (62%), slightly lower out-of-distribution accuracy | 2.07 GB |
- **Offline quantizer**: [tools/quantize_any4.py](tools/quantize_any4.py) generates `.any4.st` / `.int8.st` from an fp16 `.st` (`--bits 4|8`).
- **Accuracy verification workflows**: env-driven `TOP1_MULTI_SAVE/COMPARE`, `TOP1_REF_SAVE/COMPARE`, `SAVE/COMPARE_LOGITS` compare end-to-end agreement against the fp16 reference.
- **Reference implementation**: [src/model.rs](src/model.rs) provides CPU fp32 dequantization and unit tests as the correctness baseline for GPU kernels.

### Build

Requires Rust (edition 2024) and the Vulkan SDK (`glslangValidator`; `build.rs` compiles `assets/shaders/src/*.comp` → `assets/shaders/spv/*.spv` at build time).

```bash
cargo build --release
```

### Run

Load a model (default `c:\work\niceui\rwkv-g1h-3B.st`, switchable via `MODEL_PATH`):

```bash
cargo run --release
```

Key env vars (see [src/main.rs](src/main.rs)):

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Model path; `.int8_idx` → int8, `.any4_idx` → any4, otherwise fp16 |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | Multi-prompt teacher-forced Top-1 agreement |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | Single-prompt CPU fp32 reference agreement |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | Single-prompt logits RMSE / Top-10 comparison |
| `GEN_TOKENS` / `REPORT_EVERY` | Continuous autoregressive generation (`memtest` sub-flow) |

### Quantization

```bash
# 4-bit any4
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.any4.st --bits 4
# 8-bit int8
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

Calibration prompt collection: [tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py) (multilingual stratified sampling from aya_dataset).

### Layout

```
src/            Rust sources (main / model / gpu_model / runtime / vulkan)
assets/shaders/ Vulkan compute shader sources (*.comp; spv are build artifacts)
tools/          Offline quantization & calibration tools (Python)
参考/           Research reference docs (quantization reports, shader notes, etc.)
test/           Development scripts
```

## License

See [Cargo.toml](Cargo.toml) for dependencies. This repository is research-oriented; please verify weights/licensing before commercial use.