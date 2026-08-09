# rwkv-rsv

[English](README.md) · [中文版](README_CN.md)

**RWKV-7 inference engine in Rust + Vulkan/CUDA compute shaders.** Supports **fp16 / int8(8-bit)** weight-quantization paths auto-routed by model file, coexisting exclusively, plus a **CPU fp32 reference** for accuracy verification.

> Implementation details live in [参考/技术实现细节.md](参考/技术实现细节.md).

---

## Contents

1. [What It Does](#1-what-it-does)
2. [Design Goals](#2-design-goals)
3. [Build](#3-build)
4. [Run](#4-run)
5. [Quantization Toolchain](#5-quantization-toolchain)
6. [Metrics](#6-metrics)
7. [Layout](#7-layout)
8. [Relationship with Albatross and cryscan/rosalia](#8-relationship-with-albatross-and-cryscanrosalia)
9. [Known Limitations & Future Work](#9-known-limitations--future-work)
10. [License](#10-license)

## 1. What It Does

`rwkv-rsv` is a pure-Rust **RWKV-7** inference engine that runs on **Vulkan / CUDA compute shaders** — portable across GPU vendors (NVIDIA / AMD / Intel / mobile) and platforms (Windows / Linux / macOS).

Two core goals:

1. **Cut VRAM**: offline weight quantization (int8 8-bit saves 41%) shrinks the 3B model's weights from 5.49GB (fp16) to 3.22GB with little accuracy loss.
2. **Reproducible verification**: a CPU fp32 reference plus GPU-kernel unit tests isolate kernel error from quantization error, verifiable layer-by-layer and token-by-token.

Validated against **RWKV-7 Goosed g1h-3B**; the model structure adapts to safetensors tensor shapes, so other models in the same family load without code changes.

## 2. Design Goals

- **Portability first**: Vulkan over CUDA-only, trading ~40% peak throughput for cross-vendor / cross-platform usability.
- **Two quantization paths coexist**: fp16 (lossless reference) / int8 (near-lossless), auto-routed by model file, one binary for all.
- **Runtime-compiled shaders**: `build.rs` compiles `*.comp` → SPIR-V via `glslangValidator` at build time; `constant_id` enables cross-hardware adaptation.
- **Research-oriented**: CPU fp32 reference, DIAG three-way check, logits comparison, teacher-forced Top-1 agreement, and more.

## 3. Build

Requires **Rust (edition 2024)** and the **Vulkan SDK** (`glslangValidator`; `build.rs` compiles `assets/shaders/src/*.comp` → `assets/shaders/spv/*.spv` at build time).

```bash
cargo build --release
```

## 4. Run

Load a model (default `c:\work\niceui\rwkv-g1h-3B.st`, switchable via `MODEL_PATH`; presence of `.int8_idx` → int8, otherwise fp16):

```bash
cargo run --release
```

Key env vars (see [src/main.rs](src/main.rs)):

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Model path; `.int8_idx` → int8, otherwise fp16 |
| `BACKEND` | `vulkan` / `cuda`, explicit backend selection |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | Multi-prompt teacher-forced Top-1 agreement |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | Single-prompt CPU fp32 reference agreement |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | Single-prompt logits RMSE / Top-10 comparison |
| `GEN_TOKENS` / `REPORT_EVERY` | Continuous autoregressive generation (`memtest` sub-flow) |
| `DIAG` | Three-way check (seq / tok / CPU) + dequant validation |
| `PROF_GPU` / `PROF_HOST` | GPU / host-side profiling |
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | Override cross-hardware adaptation |

### 4.1 Examples (web-rwkv style)

```bash
# Model info: load model, print ModelInfo, probe forward
cargo run --release --example model_info
# Autoregressive generation: prefill + GPU self-loop (argmax deterministic / sample)
cargo run --release --example generate          # env: NTOKENS, TEMP, TOPK, TOPP, GEN_MODE, VOCAB_JSON
# Throughput benchmark: infer_seq / infer_tokens / argmax_selfloop / sample_selfloop
cargo run --release --example benchmark
# State serialization: forward→state_back→save→state_load→state_back lossless round-trip
cargo run --release --example state_persist     # env: OUT=state.bin
```

### 4.2 Library API & GPU Sampling

`rwkv-rsv` also exports a **library** tailored for server integration (e.g. an `ai00-server`), mirroring [web-rwkv](https://github.com/cryscan/web-rwkv):

- **`ModelBuilder` + `Bundle`**: load a model and bind a zero-initialized `State`; `infer_tokens` / `infer_seq` / `infer` advance it and return logits.
- **`State`**: first-class session state — `state_back()` downloads it to a CPU `Vec<f32>`, `state_load()` restores it, `reset()` clears it. Serialization is bit-exact.
- **GPU sampling (`SamplerParams`)**: `infer_sample` / `infer_sample_selfloop` filter logits entirely on GPU and return only the sampled token index.

## 5. Quantization Toolchain

Offline quantizer (Python, run via `uv`):

```bash
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

Calibration prompt collection: [tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py) (aya_dataset multilingual stratified sampling: 30% zh / 30% en / 40% other). Full arg reference: [参考/技术实现细节.md](参考/技术实现细节.md).

## 6. Metrics

Hardware: **RTX 2080 Ti**. Model: RWKV-7 Goosed g1h-3B (weights fp16 5.49GB / int8 3.22GB, 41% saved). Full reports under [参考/](参考/).

**End-to-end accuracy (teacher-forced Top-1 agreement vs fp16 reference):**

| Path | Agreement | Notes |
|---|---|---|
| int8 | **98.8%** (506/512) | near-lossless; weight avg cos 0.999820 / rel 0.6135% |

**GPU decode self-loop throughput** (argmax self-loop, 1000 tokens, GPU 60°C cold; via `memtest` + `SELFLOOP_ONLY=1` / `SELFLOOP_N=1000`):

| Weight | Vulkan | CUDA |
|---|---|---|
| fp16 | 80.0 tok/s | 85.7 tok/s |
| int8 | 110.6 tok/s | 110.8 tok/s |

int8 ≈ +38% over fp16 (Vulkan) / +29% (CUDA).

## 7. Layout

```
src/            Rust sources (main / model / gpu_model / runtime / vulkan / backend)
assets/shaders/ Vulkan compute shader sources (*.comp; spv are build artifacts)
tools/          Offline quantization & calibration tools (Python)
参考/           Research reference docs (quantization reports, shader notes, GPU model, technical details)
test/           Development scripts
examples/       Self-contained example programs
```

## 8. Relationship with Albatross and cryscan/rosalia

This project initially referenced two existing RWKV inference implementations:

- **信天翁 [Albatross](https://github.com/BlinkDL/Albatross)** (Apache-2.0): a high-performance **CUDA** engine for RWKV-7 (claims 15000+ tps decode for 7.2B fp16 on a single 5090). Our **kernel fusion, sequence-parallel batched submission, argmax sampling (transferring only token indices), compile-once/reuse pipeline** designs draw on / benchmark against its approach.
- **[cryscan/rosalia](https://github.com/cryscan/rosalia)**: an RWKV engine in **Rust + Vulkan compute shaders**. Our **Vulkan kernel skeleton, runtime structure, and initial design of some operators** were inspired by it.

| Dimension | Albatross | rwkv-rsv |
|---|---|---|
| Compute backend | CUDA (single vendor) | Vulkan / CUDA (cross-vendor / cross-platform) |
| Philosophy | peak performance, hardware-specific | portability first, runtime-compiled shaders |
| Relative gap | baseline | about **40%** (same GPU inference path) |

The gap mainly comes from: ① Albatross's more aggressive kernel fusion; ② CUDA Graph capture cutting launch overhead; ③ CUDA's mature hardware-specific libraries. This is a **portability (Vulkan) vs peak performance (CUDA) trade-off**, not an implementation defect.

Inspired by both, this repository has since evolved independently, adding capabilities neither has: **two weight-quantization paths** (fp16 / int8 auto-routed), **CPU fp32 reference plus GPU-kernel unit tests**, and an **offline quantization toolchain** with a reproducible accuracy-verification workflow.

## 9. Known Limitations & Future Work

- **prefill dequant overhead ~10%** (int8 vs fp16 at T=512): full removal needs cooperative-matrix fused dequant+GEMM (Marlin-style zero-copy), a long-term direction.
- **prefill time grows ~quadratically with T** (T=256: 0.33ms → T=512: 0.81ms, WKV parallel intra-chunk term); consider WKV chunking for long prompts.
- **Precision enhancements** (roadmap): calibration-weighted k-means, G=64 for denser metadata.

## 10. License

MIT. See [Cargo.toml](Cargo.toml) for dependencies. Research-oriented; please verify model weights (RWKV-7 Goosed weight license) and third-party dependency licensing before commercial use.