# rwkv-rsv

[English](README.md) · [中文版](README_CN.md)

RWKV-7 inference engine in **Rust + Vulkan compute shaders** — no CUDA dependency, portable across GPU vendors and platforms. Supports **fp16 / int8(8-bit)** weight-quantization paths auto-routed by model file, coexisting exclusively, plus a **CPU fp32 reference** for accuracy verification.

> Primary language is English. A Chinese version is available at [README_CN.md](README_CN.md).

---

## English

### Contents

1. [Overview](#1-overview)
2. [Design Goals](#2-design-goals)
3. [Model Parameters](#3-model-parameters)
4. [RWKV-7 Architecture Notes](#4-rwkv-7-architecture-notes)
5. [Two Quantization Paths](#5-two-quantization-paths)
6. [Quantized File Format](#6-quantized-file-format)
7. [Cross-Hardware Adaptation](#7-cross-hardware-adaptation)
8. [Build](#8-build)
9. [Run](#9-run)
10. [Quantization Toolchain](#10-quantization-toolchain)
11. [Accuracy & Performance Benchmarks](#11-accuracy--performance-benchmarks)
12. [Layout](#12-layout)
13. [Relationship with Albatross and cryscan/rosalia](#13-relationship-with-albatross-and-cryscanrosalia)
14. [Known Limitations & Future Work](#14-known-limitations--future-work)
15. [License](#15-license)

### 1. Overview

`rwkv-rsv` is a pure-Rust **RWKV-7** inference engine that runs entirely on **Vulkan compute shaders** — no CUDA dependency, portable across GPU vendors (NVIDIA / AMD / Intel / mobile) and platforms (Windows / Linux / macOS).

Two core goals:

1. **Cut VRAM**: offline weight quantization (int8 8-bit saves 41%) shrinks the 3B model's weights from 5.49GB (fp16) to 3.22GB with little accuracy loss.
2. **Reproducible verification**: a CPU fp32 reference plus GPU-kernel unit tests isolate kernel error from quantization error, verifiable layer-by-layer and token-by-token.

It is validated against **RWKV-7 Goosed g1h-3B**, but the model structure adapts to safetensors tensor shapes, so other models in the same family load without code changes.

### 2. Design Goals

- **Portability first**: Vulkan over CUDA, trading ~40% peak throughput for cross-vendor / cross-platform usability.
- **Two quantization paths coexist**: fp16 (lossless reference) / int8 (near-lossless), auto-routed by model file, one binary for all.
- **Runtime-compiled shaders**: `build.rs` compiles `*.comp` → SPIR-V via `glslangValidator` at build time; `constant_id` enables cross-hardware adaptation.
- **Research-oriented**: CPU fp32 reference, DIAG three-way check, logits comparison, teacher-forced Top-1 agreement, and more.

### 3. Model Parameters

Default validation model **RWKV-7 Goosed `g1h-3B`**:

| Parameter | Value |
|---|---|
| Layers | 32 |
| n_embd (C) | 2560 |
| Heads | 40 |
| head_size (N) | 64 |
| ffn_hidden | 10240 |
| vocab | 65536 |
| Low-rank mid dims (w/a/v/g) | derived from safetensors shapes |

> All hyperparameters are derived from tensor shapes (layers = max `blocks.*` + 1; heads/head_size from `r_k`; low-rank mid dims from the non-C dimension of `w1/a1/v1/g1`).

**Per-layer weight tensors** (`blocks.{i}.`, stored `[M, K]`, row-major):

| Class | Tensor | Shape | Quantized |
|---|---|---|---|
| Attention | `att.receptance.weight` | 2560 × 2560 | ✅ |
| Attention | `att.key.weight` | 2560 × 2560 | ✅ |
| Attention | `att.value.weight` | 2560 × 2560 | ✅ |
| Attention | `att.output.weight` | 2560 × 2560 | ✅ |
| FFN | `ffn.key.weight` | 10240 × 2560 | ✅ |
| FFN | `ffn.value.weight` | 2560 × 10240 | ✅ |
| Token-shift | `att.x_r / x_w / x_k / x_v / x_a / x_g` | [C] | ❌ |
| Attn biases | `att.w0 / a0 / v0` | [C] | ❌ |
| Low-rank gates | `att.w1/w2, a1/a2, v1/v2, g1/g2` | [C,mid]/[mid,C] | ❌ |
| WKV params | `att.r_k` [H,N], `k_k`, `k_a` | [C] | ❌ |
| Norms | `ln1, ln2, ln_x, ln0, ln_out` | [C] | ❌ |
| FFN shift | `ffn.x_k` | [C] | ❌ |

Total quantized matrices: **6 × 32 = 192**. head/emb (directly control logits), LayerNorm, and low-rank small matrices are kept as-is.

### 4. RWKV-7 Architecture Notes

Each layer = **Time Mixing + Channel Mixing**, strictly aligned with the numpy baseline (see [src/model.rs](src/model.rs)).

**Time Mixing (token shift + linear attention / DPLR state):**

```
ln1 = LayerNorm(x)
xr…xg = lerp(ln1, prev, x_*)              // 6-way token shift lerp
r   = xr @ receptance_w;  k = xk @ key_w;  v = xv @ value_w
v   = lerp(v, v_first, sigmoid(v0 + xv @ v1 @ v2))   // gated v_first rollback
w   = exp( -sigmoid(w0 + tanh(xw @ w1) @ w2) / √e )  // decay ∈ [0.545, 1.0]
a   = sigmoid(a0 + xa @ a1 @ a2)
kk  = L2_norm( k * k_k )                    // per-head L2
k_mod = lerp(k, k * a, k_a)
S   = S * w + (S @ kk) * (-kk * a) + v ⊗ k_mod    // DPLR state update
y   = S @ r + GroupNorm · ln_x
y  += Σ_h ( Σ_j r·k_mod·r_k ) · v          // sum_rk_rk term
g   = sigmoid(xg @ g1) @ g2
x  += (y * g) @ output_w
```

**Channel Mixing:**

```
ln2 = LayerNorm(x)
xb  = lerp(ln2, prev, ffn_x_k)
x  += relu(xb @ ffn_key_w)² @ ffn_value_w
```

**Per-layer RNN state**: `tmix_x` [C], `tmix_rnn` [H, N, N] (DPLR state), `cmix_x` [C]. O(1) state, per-token autoregressive.

### 5. Two Quantization Paths

Weights are **auto-routed by model file** (priority int8 → fp16), mutually exclusive, one binary for all two:

| Path | Weight bits | Format | 3B weight file | Saved | Weight-level | End-to-end Top-1 |
|---|---|---|---|---|---|---|
| fp16 | 16-bit | none | 5.49 GB | — | lossless (ref) | 100% |
| int8 | 8-bit | asymmetric per-group(128), scale/zero, no LUT | 3.22 GB | 41% | avg cos **0.999820** / rel **0.6135%** | **98.8%** (506/512) |

- **Quantized ops**: 6 matrices per layer (att.receptance/key/value/output + ffn.key/value), 192 total; head/emb, LayerNorm, low-rank kept as-is.
- **int8 advantage**: uniform per-group quantization has no k-means cluster-boundary nonlinearity or LUT, so it is a near-lossless tier.
- **Routing**: presence of the `.int8_idx` key in the model file at `MODEL_PATH`.

### 6. Quantized File Format

Matrices stored `[M, K]`, K contraction dim, `group = 128`. Dequantization happens on-the-fly in GPU shaders / CPU reference — **no permanent fp16 copy**.

**int8 (asymmetric per-group uniform, 256 levels, no LUT):**

| safetensors key suffix | Shape | dtype | Notes |
|---|---|---|---|
| `{name}.int8_idx` | [M, K] | U8 (4 bytes packed per uint32, little-endian) | 1 byte/weight, 0..255 |
| `{name}.int8_sz` | [M, K/128] | U32 | (scale: fp16 low16 \| zero: fp16 high16) |

Dequant: `w[m,k] = scale[m, k/128] * idx[m,k] + zero[m, k/128]`, with `scale = (max-min)/255`, `zero = min`. Constant groups (scale=0) rebuild exactly as zero.

> The format contract is locked by CPU unit tests in [src/model.rs](src/model.rs) (`dequant_int8`), guaranteeing bit-exact pack (Python) → unpack (Rust/GPU).

### 7. Cross-Hardware Adaptation

At startup we detect `vendor_id`, `subgroup_size`, `max_compute_shared_memory_size`, `max_compute_work_group_size`, `device_type`, and pick optimal parameters (see [参考/GPU模型实现.md](参考/GPU模型实现.md)):

**GEMM tile (`gemm_tile()`):**

| Hardware | Condition | TILE_M | TILE_N | TILE_K |
|---|---|---|---|---|
| NVIDIA high | shared ≥ 163840 | 128 | 128 | 32 |
| NVIDIA mid | shared ≥ 98304 | 64 | 64 | 64 |
| NVIDIA low | other | 64 | 64 | 32 |
| AMD high | shared ≥ 65536 | 64 | 64 | 32 |
| AMD low | other | 32 | 32 | 32 |
| Intel | all | 32 | 32 | 32 |
| Other | all | 64 | 64 | 32 |

Shared-memory overflow auto-degrades (reduce TILE_K first, then TILE_M/TILE_N, then fall back to minimal config). All subgroup-using shaders parameterize `SUBGROUP_SIZE` via `constant_id` for AMD (wave64) / Intel (variable) correctness.

**Overridable via env:** `GEMM_TILE_M/N/K`, `GEMV_BLOCK_SIZE`, `GEMV_ROWS`.

### 8. Build

Requires **Rust (edition 2024)** and the **Vulkan SDK** (`glslangValidator`; `build.rs` compiles `assets/shaders/src/*.comp` → `assets/shaders/spv/*.spv` at build time).

```bash
cargo build --release
```

### 9. Run

Load a model (default `c:\work\niceui\rwkv-g1h-3B.st`, switchable via `MODEL_PATH`):

```bash
cargo run --release
```

Key env vars (see [src/main.rs](src/main.rs)):

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Model path; `.int8_idx` → int8, otherwise fp16 |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | Multi-prompt teacher-forced Top-1 agreement |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | Single-prompt CPU fp32 reference agreement |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | Single-prompt logits RMSE / Top-10 comparison |
| `GEN_TOKENS` / `REPORT_EVERY` | Continuous autoregressive generation (`memtest` sub-flow) |
| `DIAG` | Three-way check (seq / tok / CPU) + dequant validation |
| `PROF_GPU` / `PROF_HOST` | GPU / host-side profiling |
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | Override cross-hardware adaptation |

#### 9.1 Examples (web-rwkv style)

Self-contained runnable programs under `examples/`:

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

#### 9.2 Library API (web-rwkv style) & GPU Sampling

`rwkv-rsv` also exports a **library** tailored for server integration (e.g. an `ai00-server`), mirroring [web-rwkv](https://github.com/cryscan/web-rwkv)'s abstractions:

- **`ModelBuilder` + `Bundle`**: load a model and bind a zero-initialized `State`; `infer_tokens` / `infer_seq` / `infer` advance it and return logits.
- **`State`**: first-class session state — `state_back()` downloads it to a CPU `Vec<f32>`, `state_load()` restores it, `reset()` clears it. Serialization is bit-exact (round-trip `max_diff == 0`).
- **GPU sampling (`SamplerParams`)**: `infer_sample` / `infer_sample_selfloop` filter logits entirely on GPU — `temperature`, `top-k`, `top-p` plus OpenAI-compatible `repetition_penalty`, `frequency_penalty`, `presence_penalty` — and return only the sampled token index (no per-token logits download).

> Determinism note: GPU forward is deterministic given an identical `State`. The former "non-determinism" was a `Bundle::reset()` bug that reset the model-internal state instead of the session state; `reset()` now correctly clears the working session state.

### 10. Quantization Toolchain

Offline quantizer (Python, run via `uv`):

```bash
# 4-bit any4
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.any4.st --bits 4
# 8-bit int8
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

Key arguments:

| Arg | Default | Notes |
|---|---|---|
| `--bits` | 4 | 4=any4 (LUT+k-means); 8=int8 asymmetric per-group (near-lossless) |
| `--group` | 128 | quantization group size |
| `--iters` | 50 | k-means iteration cap |
| `--bias-pow` | 1.0 | signed power distortion: biases k-means toward heavy tails (any4 paper) |
| `--keep-outliers` | off | replace each row's LUT extremes with actual extremes for exact outlier rebuild |
| `--calib` | — | calibration activation npz → calibration-weighted k-means (`scale_sample_weight=True`) |
| `--nnq-calib` | — | nnq output-domain LUT optimization (fixed indices, min output MSE, closed form ~100× faster than Adam) |
| `--device` | auto | k-means device (cuda if available, else cpu) |

Calibration prompt collection: [tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py) (aya_dataset multilingual stratified sampling: 30% zh / 30% en / 40% other).

Built-in weight-level acceptance (PASS/FAIL): int8 requires avg cos ≥ 0.999, avg rel ≤ 1%.

### 11. Accuracy & Performance Benchmarks

**Hardware: RTX 2080 Ti.** Full reports under [参考/](参考/) (`int8量化报告.md`, `GPU模型实现.md`).

**End-to-end accuracy (teacher-forced Top-1 agreement vs fp16 reference):**

| Path | Agreement | Notes |
|---|---|---|
| int8 | **98.8%** (506/512) | near-lossless; no OOD logit compression / ranking inversion |

**GPU decode self-loop throughput** (argmax self-loop, 300 tokens, GPU 60°C cold):

| Weight | Backend | tok/s |
|---|---|---|
| fp16 | Vulkan | 79.8 |
| fp16 | CUDA | 85.6 |
| int8 | Vulkan | 105.5 |
| int8 | CUDA | 110.2 |

> Hardware: RTX 2080 Ti. Measured via `cargo run --release -- memtest` with `SELFLOOP_ONLY=1` / `SELFLOOP_N=300` (pure GPU argmax self-loop, Batch=N, re-measured 2026-08-09 GPU 60°C cold). int8 ≈ +32% over fp16 (Vulkan) / +29% (CUDA); CUDA ≈ +5% over Vulkan at same weight precision.

> VRAM note: `memtest`'s dedicated reading is system-level (includes desktop/browser noise); use "process delta = reading − idle value at the same time".

### 12. Layout

```
src/            Rust sources (main / model / gpu_model / runtime / vulkan)
assets/shaders/ Vulkan compute shader sources (*.comp; spv are build artifacts)
tools/          Offline quantization & calibration tools (Python)
参考/           Research reference docs (quantization reports, shader notes, GPU model notes)
test/           Development scripts
```

### 13. Relationship with Albatross and cryscan/rosalia

This project initially referenced two existing RWKV inference implementations:

- **信天翁 [Albatross](https://github.com/BlinkDL/Albatross)** (Apache-2.0): a high-performance **CUDA** inference engine for RWKV-7 (repo claims 15000+ tps decode for 7.2B fp16 on a single 5090). Our **kernel fusion (fuse_ka + dplr + group_norm + sum_rk_rk in one launch), sequence-parallel batched submission, argmax sampling (transferring only token indices), and compile-once/reuse pipeline** designs draw on / benchmark against its approach.
- **[cryscan/rosalia](https://github.com/cryscan/rosalia)**: an RWKV inference engine in **Rust + Vulkan compute shaders**. Our **Vulkan kernel skeleton (ROWS=4 / BLOCK_SIZE=128 / SUBGROUP_SIZE `constant_id` cross-hardware adaptation), runtime structure, and the initial design of some operators** were inspired by it.

#### 13.1 Key comparison vs Albatross

| Dimension | Albatross | rwkv-rsv |
|---|---|---|
| Compute backend | CUDA (single vendor) | Vulkan (cross-vendor / cross-platform) |
| Peak throughput | 15000+ tps decode on a single 5090 | see local benchmark below (2080 Ti) |
| Philosophy | peak performance, hardware-specific (cuBLAS/cuDNN, CUDA Graph) | portability first, runtime-compiled shaders |
| Relative gap | baseline | about **40%** (same GPU inference path) |

The gap mainly comes from: ① Albatross's more aggressive kernel fusion reducing dispatches; ② CUDA Graph capture cutting launch overhead; ③ CUDA's mature hardware-specific libraries. This is a **portability (Vulkan) vs peak performance (CUDA) trade-off**, not an implementation defect.

#### 13.2 Independent evolution

Inspired by both, this repository has since evolved independently, adding capabilities neither has:

- **Two weight-quantization paths** (fp16 / int8) auto-routed by model file;
- **CPU fp32 reference** ([src/model.rs](src/model.rs)) plus GPU-kernel correctness unit tests, as a baseline isolating kernel error from quantization error;
- **Offline quantization toolchain** (per-group int8) with a reproducible accuracy-verification workflow.

### 14. Known Limitations & Future Work

- **prefill dequant overhead ~10%** (int8 vs fp16 at T=512): full removal needs cooperative-matrix fused dequant+GEMM (Marlin-style zero-copy), a long-term direction.
- **prefill time grows ~quadratically with T** (T=256: 0.33ms → T=512: 0.81ms, WKV parallel intra-chunk term); consider WKV chunking for long prompts.
- **Precision enhancements** (roadmap): calibration-weighted k-means (`sample_weight=calibrate`), G=64 for denser metadata.

### 15. License

MIT. See [Cargo.toml](Cargo.toml) for dependencies. Research-oriented; please verify model weights (RWKV-7 Goosed weight license) and third-party dependency licensing before commercial use.