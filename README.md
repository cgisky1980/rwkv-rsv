# rwkv-rsv

[English](README.md) · [中文版](README_CN.md)

**RWKV-7 inference engine in Rust + Vulkan/CUDA compute shaders.** Supports **fp16 / int8 (8-bit)** weight-quantization paths auto-routed by model file, coexisting exclusively, plus a **CPU fp32 reference** for accuracy verification.

> Implementation details live in [参考/技术实现细节.md](参考/技术实现细节.md).
> This README holds **current numbers only**. The 23 sections of measurement logs (what was tried, what died, with raw data paths) moved to [参考/Vulkan-CUDA实测记录合集.md](参考/Vulkan-CUDA实测记录合集.md) on 2026-10-12 — nothing was deleted, and every §6.x citation below points into that file.

---

## At a glance (2026-10-12)

Same-session, paired, 4 interleaved rounds (order reversed between rounds), `power.limit` 300 W, RWKV-7 Goosed g1h-3B int8. **Throughput in tokens/s (the number a user feels); ms/step stays in §6.2 as the engineering gap. Absolute numbers first - the ratio is derived from them, not the other way round.**

| Path | Vulkan tok/s | CUDA tok/s | Vulkan / CUDA | Plain verdict |
|---|---|---|---|---|
| **Single stream, 1 token at a time** (what the desktop client does) | **113.1** | 98.7 | **1.15×** | **+14.4 tok/s** |
| Batched decode B=1 | **101.6** | 87.9 | **1.16×** | +13.7 tok/s |
| Batched decode B=8 | **681.6** | 566.9 | **1.20×** | +114.7 tok/s |
| Batched decode B=32 | **1 754.8** | 1 737.8 | **1.01×** | +17.0 tok/s (tie) |
| Batched decode B=64 | 2 201.1 | 2 610.5 | 0.84× | −409.4 tok/s |
| Batched decode B=128 | 2 432.4 | 3 218.3 | 0.76× | −785.9 tok/s |
| Batched decode B=256 | 2 648.9 | 3 374.7 | 0.78× | −725.8 tok/s |
| Batched prefill B=1 / 8 / 32 / 64 / 128 / 256 | **667 / 1 726 / 2 919 / 3 279 / 3 387 / 3 344** | 188 / 956 / 2 017 / 2 633 / 2 968 / 3 080 | **3.55× / 1.81× / 1.45× / 1.25× / 1.14× / 1.09×** | faster at **all six** widths |

⚠️ The prefill column divides the **padded** token count (every slot is computed to `T_pad=512`) by the whole preparation window, which also contains batch-state creation and one warmup segment - so it is a **floor**, on a protocol identical for both backends. The prompts themselves are 104 - 139 tokens, so 4 - 5× of that work is padding (§9).

So: **one stream and small batches are won; long prompts are won; wide-batch decode is still behind, and it is behind by one kernel.** At B=128 the int8 tensor-core GEMM alone costs 23.90 ms of a 46.19 ms GPU-busy step — more than the entire 12.84 ms/step gap that the −785.9 tok/s comes from. Details: §6.

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

1. **Cut VRAM**: offline weight quantization (int8 8-bit saves 41 %) shrinks the 3B model's weights from 5.49 GB (fp16) to 3.22 GB with little accuracy loss.
2. **Reproducible verification**: a CPU fp32 reference plus GPU-kernel unit tests isolate kernel error from quantization error, verifiable layer-by-layer and token-by-token.

Validated against **RWKV-7 Goosed g1h-3B**; the model structure adapts to safetensors tensor shapes, so other models in the same family load without code changes.

## 2. Design Goals

- **Portability first, with the cost stated honestly.** Vulkan is the primary backend, giving cross-vendor / cross-platform reach. Measured on the same card, in the same session, against this repo's own CUDA backend:
  - **Low concurrency — where real apps live — Vulkan is ahead**: 113.1 vs 98.7 tok/s (+14.4) on the single-stream path the desktop client uses, +13.7 / +114.7 tok/s at B=1 / B=8 (1.15× / 1.20×), a tie at B=32.
  - **Wide-batch decode it is behind**: −409 / −786 / −726 tok/s at B=64 / 128 / 256 (0.84× / 0.76× / 0.79×). Vulkan's decode throughput stops climbing at ~2 650 tok/s while CUDA reaches 3 375.
  - **Batched prefill it is ahead at every width** (667 vs 188 tok/s at B=1 up to 3 344 vs 3 080 at B=256; 3.55× → 1.09×), with byte-identical seeds between backends.
  - The remainder is **not** spread across the engine: §6.6 attributes it to a single kernel.
- **Two quantization paths coexist**: fp16 (lossless reference) / int8 (near-lossless), auto-routed by model file, one binary for all.
- **Runtime-compiled shaders**: `build.rs` compiles `*.comp` → SPIR-V via `glslangValidator` at build time; `constant_id` enables cross-hardware adaptation.
- **Research-oriented**: CPU fp32 reference, DIAG three-way check, logits comparison, teacher-forced Top-1 agreement, per-kernel GPU timers, same-session A/B orchestration, and a kernel-level benchmark harness that needs no model file.

## 3. Build

Requires **Rust (edition 2024)** and the **Vulkan SDK** (`glslangValidator`; `build.rs` compiles `assets/shaders/src/*.comp` → `assets/shaders/spv/*.spv` at build time).

```bash
cargo build --release
cargo build --release --examples   # ★ REQUIRED before running any example or benchmark
```

⚠️ **The second line is not optional.** `cargo build --release` does **not** rebuild `examples/*`, so a benchmark run after a code change silently executes the **old binary** and reports the old numbers as if they were the new ones. This has already produced false results in this repo's logs (§6.5, rule 10).

## 4. Run

Load a model (default `c:\work\niceui\rwkv-g1h-3B.st`, switchable via `MODEL_PATH`; presence of `.int8_idx` → int8, otherwise fp16):

```bash
cargo run --release
```

Key env vars (see [src/main.rs](src/main.rs)):

| Variable | Purpose |
|---|---|
| `MODEL_PATH` | Model path; `.int8_idx` → int8, otherwise fp16 |
| `BACKEND` | `vulkan` / `cuda`, explicit backend selection. ⚠️ Must be set **explicitly** in A/B scripts; an unset `BACKEND` follows whatever the previous arm left (§6.2 ledger note in the archive). |
| `TOP1_MULTI_SAVE` / `TOP1_MULTI_COMPARE` | Multi-prompt teacher-forced Top-1 agreement |
| `TOP1_REF_SAVE` / `TOP1_REF_COMPARE` | Single-prompt CPU fp32 reference agreement |
| `SAVE_LOGITS` / `COMPARE_LOGITS` | Single-prompt logits RMSE / Top-10 comparison |
| `GEN_TOKENS` / `REPORT_EVERY` | Continuous autoregressive generation (`memtest` sub-flow) |
| `DIAG` | Three-way check (seq / tok / CPU) + dequant validation |
| `PROF_GPU` / `PROF_HOST` / `PROF_GAPS` | Per-kernel GPU timestamps / host-side recording cost / gaps between dispatches. ⚠️ `PROF_GPU`'s query pool holds **2 048 dispatches per window** and drops the rest — the dump then prints a `⚠ 查询池已满` warning and its `SUM` is a **window prefix**, not the whole window's GPU-busy time. At wide batch one window ≈ one step. ⚠️ Mutually exclusive with `SELFLOOP_PIPE` (hard error, not fake timestamps). |
| `PROF_LOW` | Prints the low-rank chain's actual binding decisions (`stage1=true/false`, the chosen tile per shape) — the instrument that proves *which* arm ran, independent of speed |
| `SELFLOOP_PIPE` | **Vulkan-only windowed submit** (`<rounds per window>`, **default 2 = on**; set 0 to disable). Without it a whole self-loop segment is recorded into **one** command buffer and submitted once, so the GPU idles through all the host recording that `PROF_HOST` measures (3.14 ms/step at B=128, 0.69 ms/step single-stream). Windowing submits every k rounds **without waiting**, so recording round k+1 happens while the GPU eats window k. Applies to **both** self-loop paths — batched decode and the single-stream loop the desktop client uses. Two safety gates on the single-stream side: the segment must be longer than one window, and it must be on the per-round sampler-row path, otherwise an in-flight window would read a later round's sampling params. Worth a **fixed −2.25 / −3.19 / −2.93 / −3.20 ms/step** at B=8/32/64/128 and −1.08 ms/step single-stream ⇒ the relative gain grows as concurrency drops. Ledger: archive §6.23, raw paired tables in [参考/2026-10-11-Vulkan-prefill-fp16-GEMM瓦片档修正-实施记录.md](参考/2026-10-11-Vulkan-prefill-fp16-GEMM瓦片档修正-实施记录.md) §6.24. |
| `VK_IMMA_OPS` | Which batched tensor-core (W8A8 int8) ops are on; default **all** (head/out/key/val + rkv) since 2026-10-10. `VK_IMMA_BM` / `VK_IMMA_NST` pick the tile, `VK_IMMA_KSPLIT*` the split-K shape. |
| `LOWRANK_GEMM2` / `LOWRANK_GEMM1` / `LOWRANK_MIN` | Low-rank chain onto coopmat fp16 tensor cores. Level 2 default on at `batch >= 32`; level 1 (the mid projection) default on at `batch >= 64`, because B=32 measures a **−8…−10 % loss** there. `GEMM_SK` (split-K) is **default off** — it bought +2.6 % pooled with inconsistent signs, i.e. nothing, and only helps under preemption. |
| `WKV_STATE_F16` | Store the RNN state as fp16 on Vulkan (`_s16` arms). −1290 MiB registered at B=128, bit-identical greedy tokens at B=1 (0/64 divergence) — but it does **not** halve the state kernel's time, because the arithmetic stays fp32. |
| `GEMV_BGRP` / `MID_GG` | Workgroup slot count for the SIMT arms. 8 is the measured optimum on **both** backends (16/32 = −8.6 / −7.1 %); the CUDA-side analogue `MID_GG` measures 1.03 ms at GG=1 vs 4.66 ms at GG=8. |
| `SKIP_GRAPH` | **CUDA-only** control. On Vulkan it is a null operation — which is exactly why it is useful: an A/B that "wins" under `SKIP_GRAPH=1` on Vulkan is measuring nothing. |
| `ABLATE` | **Cross-backend removal ruler.** CUDA: comma-separated kernel names, launches are skipped. Vulkan: same names, implemented at the single dispatch site (`src/runtime/record_kernel`), keeping pipeline / uniforms / barriers and dropping only the dispatch + its timestamps. ⚠️ A typo silently matches nothing and reads as "zero benefit", so the first hit per name is logged. ⚠️ Only valid when the removed kernel's output does not change *other* kernels' work — ablating `gemm_imma_bm64_nst8` came back **−47 %** (garbage into `relu²` ⇒ the sparse FFN stops being sparse). |
| `VK_SPV_DIR` | **Shader A/B instrument.** Load every `*.spv` from this flat directory instead of the embedded copy. Required because release builds bake `assets/shaders/spv/*.spv` into the binary (`#[derive(RustEmbed)]` in [src/vulkan/asset.rs](src/vulkan/asset.rs)) — copying new bytes onto disk does **not** change what a running exe executes, so without this knob you silently benchmark the old kernels while believing you swapped them (this voided archive §6.10's numbers). Point it at a directory holding one variant's bytes and log the md5 of each spv actually bound. A missing file is a **hard error** — no silent fallback, which is what makes it a usable arm marker. |
| `GEMM_TILE_M` / `GEMM_TILE_N` / `GEMM_TILE_K` / `GEMM_MIN_BLOCKS` / `GEMM_SWZ` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | Override cross-hardware adaptation. ⚠️ `GEMM_TILE_*` is a **direct override** and bypasses `gemm_tile_for`'s structural floor (§6.6). |
| `UNIFORM_POOL_MB` | Vulkan uniform pool capacity in MB (default 8; usage is O(distinct params per batch) — measured peak 69 KB, so this is a fallback knob, not something to tune) |

### 4.1 Examples (web-rwkv style)

```bash
# Model info: load model, print ModelInfo, probe forward
cargo run --release --example model_info
# Autoregressive generation: prefill + GPU self-loop (argmax deterministic / sample)
cargo run --release --example generate          # env: NTOKENS, TEMP, TOPK, TOPP, GEN_MODE, VOCAB_JSON
# Throughput benchmark: infer_seq / infer_tokens / argmax_selfloop / sample_selfloop
cargo run --release --example benchmark
# Steady-state prefill benchmark (2nd run onward after same-length warmup; excludes one-time pipeline creation)
cargo run --release --example prof_prefill_steady   # env: PTOKENS
# Batched-concurrency throughput/correctness: SLOTS x SEGS x NTOK, with STEP=1 two-path parity,
# SLOT_BENCH=1 slot fan-out, BUCKET_BENCH=1 the four length-bucket cells, and BENCH_JSON=<path>
# writing "reading + full gating" as one JSONL line
cargo run --release --example batch_decode_bench    # env: MODEL_PATH, SLOTS, PAD_TO, TOPK, STEP, BENCH_JSON
# Same-session **interleaved A/B** orchestrator (the only legitimate way to compare): rotates
# measurement order, runs children strictly serially, reports median + spread + ratio +
# cross-backend fingerprint agreement. env: AB_SLOTS, AB_ROUNDS, AB_JSON, AB_SUMMARY
cargo run --release --example bench_ab
# Kernel-level A/B **without a model file** (weights + activations are synthesized):
# times individual shapes, prints `sum(y)` as a "the kernel actually ran" guard.
# env: KB_OPS (comma list), KB_BATCH, KB_ITERS, KB_REPEATS
cargo run --release --example kernel_bench
# State serialization: forward→state_back→save→state_load→state_back lossless round-trip
cargo run --release --example state_persist     # env: OUT=state.bin
```

⚠️ `kernel_bench` **exits 0 even when every arm fails**. A driver script that only checks the return code has verified nothing; `grep` the per-arm result lines.

### 4.2 Library API & GPU Sampling

`rwkv-rsv` also exports a **library** tailored for server integration (e.g. an `ai00-server`), mirroring [web-rwkv](https://github.com/cryscan/web-rwkv):

- **`ModelBuilder` + `Bundle`**: load a model and bind a zero-initialized `State`; `infer_tokens` / `infer_seq` / `infer` advance it and return logits.
- **`State`**: first-class session state — `state_back()` downloads it to a CPU `Vec<f32>`, `state_load()` restores it, `reset()` clears it. Serialization is bit-exact.
- **GPU sampling (`SamplerParams`)**: `infer_sample` / `infer_sample_selfloop` filter logits entirely on GPU and return only the sampled token index.
- **Concurrency pool primitives (batched-layout `State`)**: `create_batch_state(B)` + `forward_seq_batch_padded` / `submit_sample_selfloop_batch` / `batch_step_sample`; slot management via `state_slot_back` / `state_slot_load` (host round-trip) and `state_slot_broadcast` (**on-device** fan-out of one state to many slots — session forks / many-samples-per-prompt, saving 2× single-slot state traffic over PCIe per target). `warmup_len_buckets(state, &[T_pad..])` moves the per-length-bucket seq-buffer rebuild out of the first real request and into startup (★ the buffer cache holds **one** shape: only the last bucket stays resident). Check `supports_batch_decode()` for the batched **decode** preconditions; `batch_decode_blocker()` names *which* one fails (backend lacks the ops vs model has no int8 handles). ⚠️ That gate covers decode only — batched prefill works on fp16 models too, and the value `forward_seq_batch_padded` returns is each slot's last prompt token (the decode seed), not a predicted token.

## 5. Quantization Toolchain

Offline quantizer (Python, run via `uv`):

```bash
uv run tools/quantize_any4.py --in rwkv-g1h-3B.st --out rwkv-g1h-3B.int8.st --bits 8
```

Calibration prompt collection: [tools/prepare_calib_prompts.py](tools/prepare_calib_prompts.py) (aya_dataset multilingual stratified sampling: 30 % zh / 30 % en / 40 % other). Full arg reference: [参考/技术实现细节.md](参考/技术实现细节.md).

## 6. Metrics

Hardware for every number here: one **RTX 2080 Ti** (68 SM, 616 GB/s measured DRAM roof, also driving the display). Model: RWKV-7 Goosed g1h-3B — `C=2560 / H=40 / N=64 / V=65536 / L=32`, weights fp16 5.49 GB / int8 3.22 GB. Full reports under [参考/](参考/).

**Which quantization path to pick**: int8 — 41 % less VRAM, 98.8 % top-1 agreement versus the CPU fp32 reference (§6.4), and it is the path every tensor-core and batched-decode arm is built on (an fp16 model cannot do batched decode at all, §9). fp16 stays the lossless reference and the format a plain `.st` loads as. The old head-to-head throughput pair (2026-08, not interleaved) read int8 ≈ **+32 % (Vulkan) / +36 % (CUDA)** over fp16; it has not been re-measured since, so treat it as orientation, not as a current number.

All rows below are **same-session, both backends interleaved, order reversed between rounds, ≥ 4 rounds, matched-round medians** unless stated. Ratios are **Vulkan tok/s ÷ CUDA tok/s**, so **> 1.0 means Vulkan is faster**. `power.limit` 300 W (that is the shipping 档).

### 6.1 The path the desktop client actually uses (one stream)

The six-width table in §6.2 runs *B=1 through the batched kernel*; the desktop client generates **one stream at a time** (`submit_sample_selfloop`). That cell had never been measured head-to-head until 2026-10-12.

| 口径 | Vulkan | CUDA | ratio |
|---|---|---|---|
| matched-round median (4 rounds) | **113.1 tok/s** (8.842 ms/token) | 98.7 tok/s (10.131 ms/token) | **1.146×** |
| Δmin (best-of-rounds) | 115.0 tok/s (8.696 ms/token) | 99.3 tok/s (10.070 ms/token) | 1.158× |
| round-to-round spread | **1.9 %** | 14.9 % | — |

**Vulkan is ahead by 1.29 – 1.37 ms per token**, every one of the 4 rounds ≥ 1.14×, so the direction does not depend on one cell. All 8 arms printed the **same order-sensitive fingerprint** `sum=0x278a9c xor=0x7354 mix=0xcfeeb6baaf81b753` ⇒ the two backends are bit-identical on this path. CUDA's 14.9 % spread was one round reading 86.4 tok/s while the card moved 46 → 62 ℃.

For orientation only (**cross-session, never subtractable**): this same cell read CUDA 91.4 / Vulkan 79.2 tok/s = **0.87×, i.e. behind**, on 2026-10-10. The direction flipped with the windowed-submit lever (archive §6.23 and its single-stream addendum). Raw data `/c/tmp/ac16/s52/`, ledger [参考/2026-10-11-Vulkan-prefill-fp16-GEMM瓦片档修正-实施记录.md](参考/2026-10-11-Vulkan-prefill-fp16-GEMM瓦片档修正-实施记录.md) §6.27.

### 6.2 Batched decode, six widths (12th re-measure, 2026-10-12)

Build = archive §6.23's windowed submits **on by default**. Protocol: `SEGS=4 NTOK=32 PAD_TO=512 WARMUP=1 SKIP_SINGLE=1 PROMPT_REPEAT=6 TOPK=1 PREFILL_SLOTS=8 TEMP=1`, six widths × 4 rounds, all 48 arms clean.

| Width | Vulkan tok/s | CUDA tok/s | Vulkan / CUDA | Vulkan ms/step | CUDA ms/step | Δ ms/step | 11th → 12th (ratio) |
|---|---:|---:|---:|---:|---:|---:|---|
| B=1 | **101.6** | 87.9 | **1.155** | **9.85** | 11.37 | **−1.52** | 0.914 → 1.155 |
| B=8 | **681.6** | 566.9 | **1.202** | **11.73** | 14.11 | **−2.38** | 0.971 → 1.202 |
| B=32 | **1 754.8** | 1 737.8 | **1.010** | **18.23** | 18.41 | **−0.18** | 0.808 → 1.010 |
| B=64 | 2 201.1 | 2 610.5 | 0.843 | 29.07 | 24.52 | +4.55 | 0.741 → 0.843 |
| B=128 | 2 432.4 | 3 218.3 | 0.756 | 52.61 | 39.77 | +12.84 | 0.708 → 0.756 |
| B=256 | 2 648.9 | 3 374.7 | 0.785 | 96.64 | 75.86 | +20.78 | 0.753 → 0.785 |

The tok/s column is the log's own aggregate (`segs × NTOK × B` tokens over the measured decode window); ms/step is the same reading divided by steps, and it is the column the gap analysis uses, because the gap is a fixed cost per step. **Vulkan's throughput saturates at ~2 650 tok/s from B=64 on while CUDA keeps climbing to 3 375** — that plateau, not a per-token penalty, is what "0.76×" means.

**The three low widths crossed 1.0 inside one session** (every round ≥ 1.12 at B=1, ≥ 1.18 at B=8, three positive + one tie at B=32). That is exactly the signature the lever predicts: a **fixed** ms/step saved on host recording is ~20 % of a 9.9 ms step and ~4 % of a 96 ms one.

⚠️ **Do not bank the whole move versus the 11th row.** This session's *CUDA* column is itself 8 – 13 % slower than its own 11th-session values (11.37 vs 10.43 at B=1; 39.77 vs 35.22 at B=128), run at 66 – 70 ℃ off a 60 ℃ start — part of the apparent gain is reference-side heat. The citable facts are (a) the matched-round position **within** this session and (b) the two-arm same-session A/B of the lever itself: **−2.25 / −3.19 / −2.93 / −3.20 ms/step** at B=8/32/64/128. Vulkan absolutes are likewise not subtractable across sessions (B=128 reads 52.61 here vs 49.64 in the 11th — heat, not a regression). Raw `/c/tmp/ac16/chophole/pair12/`, aggregator `agg13.mjs`, ledger archive §6.24.7.

Historical rows (10th: 0.893 / 0.943 / 0.776 / 0.718 / 0.725 / 0.746; 11th: 0.914 / 0.971 / 0.808 / 0.741 / 0.708 / 0.753; 8th 3-round-median 口径: 0.941 / 0.945 / 0.806 / 0.730 / 0.690 / 0.762; and the six pre-300 W sessions) stay in archive §6.1. **A statistic change or a 档 change is not a regression, and the two 档 never get subtracted from each other.**

### 6.3 Batched prefill, six widths

No code changed for this round; the numbers are the same-session pair behind §6.2's decode column.

| Width | B=1 | B=8 | B=32 | B=64 | B=128 | B=256 |
|---|---|---|---|---|---|---|
| Vulkan **tok/s** | **667** | **1 726** | **2 919** | **3 279** | **3 387** | **3 344** |
| CUDA **tok/s** | 188 | 956 | 2 017 | 2 633 | 2 968 | 3 080 |
| Vulkan ms (raw window) | **768** | **2 373** | **5 613** | **9 994** | **19 350** | **39 195** |
| CUDA ms (raw window) | 2 721 | 4 284 | 8 123 | 12 445 | 22 084 | 42 561 |
| Vulkan / CUDA | **3.55×** | **1.81×** | **1.45×** | **1.25×** | **1.14×** | **1.09×** |

tok/s here = `B × 512` padded tokens over the whole preparation window (batch-state creation and one warmup segment included) ⇒ a **floor**, same protocol both arms. Since the prompts are 104 - 139 tokens but padded to 512, most of this throughput is spent on padding (§9); the shape of the curve - Vulkan rising to ~3 400 tok/s while CUDA reaches 3 080 - is the honest part, not the absolute.

Vulkan leads at **every** width, and the six prefill seed fingerprints are **byte-identical** between backends (`0x4021` / `0x1bd0d` / `0x6e599` / `0xdcb32` / `0x1b9664` / `0x372cc8`). This retires two older framings ("prefill lags further than decode", "CUDA takes over at B ≥ 64") — the crossover was fixed by one tile-selection change plus a block-order swizzle (archive §6.6, §6.13 – §6.15).

### 6.4 Accuracy and memory

| Check | Result |
|---|---|
| int8 vs CPU fp32, teacher-forced Top-1 (512 prompts' next tokens) | **98.8 %** (506/512) |
| int8 vs fp16 logits | cos **0.999820**, relative error **0.6135 %** |
| int8 weights vs fp16 | 5.49 GB → **3.22 GB** (−41 %) |
| fp16 state storage (Vulkan `_s16` arms), B=128 | **−1290 MiB** registered; logits rmse 3.227 → 3.223 (unchanged), **0/64** greedy-token divergence |
| `STEP=1` two-path parity (batched vs per-slot) | **bit-identical** state after batched prefill, on both backends |
| Vulkan vs CUDA at B ≥ 8 | **divergent by design** (different kernels/arithmetic order) — not a bug, and the reason cross-backend A/B must be read as *throughput*, never as *equality* |

**How a numeric change gets admitted here.** A change that alters the arithmetic (fp32 → fp16 at some stage, a different reduction order, split-K summation) **cannot** be admitted on "the token stream is identical", because it genuinely isn't: level 1's fp32 → fp16 diverges the greedy stream, unlike level 2's bit-identical admission. Instead the repo uses a **same-ruler control triangle** — all-old vs shipped, all-old vs new, shipped vs new — and admits on *distance*, plus an argmax-divergence count. Two metrics in this codebase are known **false witnesses** and must never be cited as evidence: `seq vs cpu max_abs_diff` (fixed by the first token, insensitive to everything after) and `deployment-summarize 2×2` (rounds two different errors into the same cell). Ledger: archive §6.9 (bit-identical admission) and §6.20 – §6.21 (bit-exact sampling folds), plus [参考/2026-10-11-Vulkan-mid投影臂向量化-实施记录.md](参考/2026-10-11-Vulkan-mid投影臂向量化-实施记录.md) §11 – §12 (the control triangle for the two arms that genuinely diverge).

### 6.5 How to read these numbers (the ten rules)

This engine's gaps are 1 – 3 % and its noise is 5 – 15 %, so the discipline **is** the measurement. These rules were each bought by a wrong result that is recorded in the archive.

1. **One process owns the GPU.** Never run two benchmarks at once, and never compare across processes that overlapped.
2. **Absolute ms never travels between sessions.** Ratios don't travel either: a per-avg for the *same* compiled shader spread ±12 % across process batches (0.1854 – 0.2219 ms). Any design that compares two different processes is **direction-only**.
3. **Matched-round medians, ≥ 4 rounds, order reversed between rounds.** "Δmin of per-round bests" is structurally biased when one arm runs first every round, and it breaks down entirely under a monotone warm-up (the card drifting 1695 → 1830 MHz). This table's resolution is ≈ 2 – 3 %.
4. **State the power 档.** This card has four distinct ones (250 W / 300 W / `-lgc` 1860 MHz / hot) and numbers taken in different 档 are different physics. Rows above the 300 W baseline cannot be subtracted from it.
5. **Run a null control.** `SKIP_GRAPH` is a no-op on Vulkan, so a Vulkan "win" under it is measuring nothing — and it caught a fabricated −8.31 % (6/6 same sign) that had been banked for a day.
6. **Grep the arm-marker lines after a paired run.** An arm that never started prints nothing and reads as a pass (`STEP_AB` without `STEP=1` silently no-ops). "Two arms give the same numbers" is the *expected* outcome of a bit-preserving change, so it cannot be the marker: use a hard error (`VK_SPV_DIR` on a missing file) or the md5 of what was actually bound.
7. **The first segment is dirty.** `record_kernel`'s cache miss goes through `create_kernel` (VkPipeline creation, 82 – 88 ms once) — a host-side measurement taken on the first segment read 5.1 – 6.1 ms/step where steady state is 3.14. The 5.15 figure in old notes was retracted for exactly this.
8. **Know which instruments are trusted.** `PROF_CUDA_KERNEL` reported 6 249 ms inside a 150 ms wall clock. Ablation (removal) reads are only valid when the removed kernel's absence doesn't change other kernels' work — tensor-core GEMM shares must come from `PROF_GPU`, never from ablation (−47 % artifact).
9. **`VK_SPV_DIR` for any shader A/B**, and log the md5 each arm actually bound (see §4).
10. **`cargo build --release --examples` before any example run** (see §3), otherwise you are measuring yesterday's binary.

### 6.6 What is left inside the gap

One kernel. A `PROF_GPU` dump of a B=128 decode step (its own session — the 46.19 ms below is GPU-busy and is **not** the 52.61 ms wall-clock step of §6.2; archive §6.19 re-verifies the shares in a later session with `gemm_imma` at 50.6 % vs the 51.7 % here):

| Kernel | ms/step | share |
|---|---|---|
| `gemm_imma_bm64_nst8` (int8 tensor-core GEMM) | **23.90** | **51.7 %** |
| `fuse_ka_dplr_norm_s16` (batched state) | 8.73 | 18.9 % |
| the whole low-rank chain (`gemm_f16_f32` 3.52 + `to_f16_triple` 0.94 + `gemm_f16_f32_tanh` 0.83 + 4 epilogues 0.95) — levels 1 and 2 are **both** on coopmat fp16 here | ≈6.2 | ≈13.4 % |
| `gemm_imma_reduce` (split-K reduce) | 1.99 | 4.3 % |
| `sample` | 1.49 | 3.2 % |
| `quant_x_i8` (activation quantization) | 1.35 | 2.9 % |
| total GPU | 46.19 | — |

**`gemm_imma` at 23.90 ms is bigger than the whole 12.84 ms gap at that width.** On the *same shape* the Vulkan arm sustains **34.8 TOPS** against CUDA's **50.9 TOPS**, and every candidate explanation has now been measured out:

- **Not bandwidth, at wide batch.** At B=1 the same kernel family reads the 2.5 GB int8 weight set once in 5.53 ms/step = **455 GB/s ≈ 74 % of the 616 GB/s roof** — there it *is* traffic-bound, and nothing can be won. At B=128 it takes 20.8 ms, where one pass of the weights needs 120 GB/s and two passes 240 GB/s ⇒ **far below the roof**: the constraint is arithmetic/occupancy, not traffic. (This is also why the `NST=16` "read the weights once instead of twice" experiment measured nothing: the re-read is real, but removing it wouldn't make it faster.)
- **Not latency hiding.** Vulkan's cross-tile software pipeline (`IM_PIPE`, the one structure copied from `rosalia/ember`) measured **13/20 cells slower** with inconsistent signs, and CUDA's own version of the same idea (`IMMA_PIPE` / `IM_NBUF`) measured **−36 %** — the resident blocks (~23.8 KB smem ⇒ 2 blocks × 8 warps per SM) already cover each other's load latency.
- **Not instruction count, not store order.** Cutting the fetch instructions by 4× lost; three re-orderings of the accumulator flush measured −1.1 / −2.0 / −5.2 % (below the 3 % kernel-level bar) with the store side and the smem read side locked against each other.

What is left is the **cooperative-matrix accumulator round-trip**: GLSL gives no handle on which lane owns which m16n8 fragment element, so `coopMatStore` → `subgroupBarrier` → scalar `LDS` read → `subgroupBarrier` → store is the *only* way to consume the accumulator, and `subgroupShuffle` cannot move it either (the source register index varies per lane, so it is not uniform). nvcc holds the mma fragment in registers and dequantizes/writes back there. **Closing this gap means the compiler/interface layer — direct SPIR-V fragment access or a private tile convention — not more GLSL scheduling.** That, not "Vulkan is slower", is the remaining work.

### 6.7 Levers shipped (and where their logs are)

Everything below is **default-on** in the current build. Each row's raw data, protocol, and rejected alternatives are in the named archive section (archive = [参考/Vulkan-CUDA实测记录合集.md](参考/Vulkan-CUDA实测记录合集.md)).

| Lever | What it bought | Archive |
|---|---|---|
| Windowed pipelined host submit (`SELFLOOP_PIPE=2`), both self-loop paths | fixed **−2.25 / −3.19 / −2.93 / −3.20 ms/step** at B=8/32/64/128, **−1.08 ms/step** single-stream ⇒ this is what put B ≤ 32 ahead | §6.23 (+ ledger §6.24) |
| `gemm_imma` W8A8 int8 tensor cores for the five big projections (head/out/key/val + **rkv**, default flipped 2026-10-10) | rkv arm **+27.9 / +33.6 / +42.9 / +29.7 %** at B=32/64/128/256 | §6.2②, §6.3 |
| `gemm_imma` fetch structure (A-fragment reuse + slot-grouped warps), bit-exact | **+12 … +16 %** at B ≥ 64 | §6.4 |
| `gemm_imma` shared-memory packed 4-bytes-per-word, bit-exact | decode **+2.4 … +7.2 %** | §6.8 |
| v-arm split-K with an fp16 reduce exit | **+2 … +4 %** at B ≤ 64 | §6.5 |
| Low-rank chain level 2 onto coopmat fp16 tensor cores | B=128 decode **66.8 → 57.2 ms/step (+16.9 %)** | §6.9 |
| Low-rank level 1 (the mid projection) onto tensor cores, **plus** the occupancy-aware tile rule that made it possible | B=128 **58.0 → 51.4 ms/step (+12.8 %)**; the same rule made deep-K narrow-N `pf_value` **1.94×** faster | §6.9 (its S31 supersession note), ledger [参考/2026-10-11-Vulkan-mid投影臂向量化-实施记录.md](参考/2026-10-11-Vulkan-mid投影臂向量化-实施记录.md) §10 – §12 |
| Group-major int8 weight layout, bit-identical | **+6.8 … +9.3 %** (shrinks with batch) | §6.1 |
| Batched-prefill fp16 tile default fixed | prefill **0.574× → 0.96×** of CUDA | §6.6 |
| Batched-prefill GEMM block-order swizzle (the A panel was re-read once per column) | chunk **1 077 → 981 ms**; B=128 clean chunk loop −8.9 % | §6.11 |
| Accumulator width as a fifth tunable; ffn-key on the fp16 accumulator | K=2560 prefill shapes **5.065 → 3.967 ms**, and **−49 ms per chunk** on ffn-key | §6.12, §6.13 |
| Low-M shape gate 2048 → 256 | every low-M rung is a win | §6.14 |
| Device-side batched zero-fill (chunked prefill's `reset` segment) | **1 409.6 ms → 4.6 ms** | §6.15 |
| fp16 RNN state on Vulkan (`_s16` arms) | **−1290 MiB** registered at B=128; the state kernel itself **−19.5 %** | `参考/2026-10-11-Vulkan-状态fp16存储-实施记录.md` |
| Host-submission collapse: 4 submits/token → 1; 97 state syncs → 1; 97×targets slot copies → 1; chunked-prefill hand-back on device | B=256 prefill **0.67× → 0.79×** of CUDA; admitted on the sign test, not on throughput | §6.7, §6.16, §6.17, §6.18 |
| Sampling kernel passes folded (temperature into the load; nucleus skipped under greedy) | **11 → 5** full-vocab passes, then **5 → 4**; bit-exact | §6.20, §6.21 |
| Sampler kernels (top-k / top-p / temperature / multinomial on GPU), state persistence | returns the sampled token instead of 256 KB of logits | — |

### 6.8 Roads that are closed (measured dead — do not re-try without new evidence)

- **`gemm_imma` restructuring, five sub-paths**: ① cross-tile software pipeline (`IM_PIPE`, 13/20 cells slower); ② split-K as a *throughput* lever (`GEMM_SK`; it is contention insurance, +34 – 41 % only under preemption); ③ weight fragment read directly from global (−32 … −42 %); ④ a wide `BN=16` column tile (+6.5 … +39 %); ⑤ epilogue store reordering (best −1.1 %). The flush-barrier structure is closed by code reading. Root cause is in §6.6.
- **The batched state kernel**: DRAM-bound at 68 % of the roof (418 GB/s of 616 peak). Two-row interleave and adjacent-column-pair access are shipped, and **no end-to-end gain is claimed for either** — the four-arm sweep that produced "−0.98 %" compared two arms holding the *same embedded bytes*.
- **The mid-projection arm's shape**: `GEMV_BGRP` 16/32 = −8.6 / −7.1 %, and rewriting its loops to `vec4` + `dot` = −6 … −9 % (CUDA measured the same cliff on its analogue). The instruction-count premise itself was falsified: 4× fewer instructions lost.
- **`pf_value` / ffn-down on the fp16 seq path**: they sit at **0.54×** of their own traffic lower bound (0.45× before the block-order swizzle), and both the mma-rate and the accumulator-width explanations were measured out — what is left is a weight-layout / DRAM-locality question, which belongs to the int8-prefill item in §9, not to a tuning sweep (archive §6.11, §6.12).
- **`SELFLOOP_CHUNK`** (the alternative pipelining scheme): a wash.
- **Cross-tile batching / small-batch tile variants for `batch ≤ 4`**: still open, and it is *unmeasured*, not disproven — `IM_BM` / `IM_NST` / KSPLIT were never swept there.
- **Reference repositories**: Albatross is gone from this machine (its +11 … 26 % column cannot be refreshed and must not be divided into any CUDA column above). `cryscan/rosalia` @ `ember` was surveyed in full: it has **no int8 path at all** (zero `int8`/`coopmate` hits tree-wide), so the one kernel that matters has no peer implementation there; its only learnable trick is the register-resident next-tile prefetch, and that trick is the ① in the list above — measured dead on our side (archive §6.22).

### 6.9 Full measurement logs

- [参考/Vulkan-CUDA实测记录合集.md](参考/Vulkan-CUDA实测记录合集.md) — English log, §6.1 – §6.23 plus the old §8 / §9 body, with the raw-data paths (`/c/tmp/ac16/…`) and aggregator scripts.
- [参考/Vulkan-CUDA实测记录合集-中文.md](参考/Vulkan-CUDA实测记录合集-中文.md) — Chinese log, numbered **independently** (it runs to §6.19). ⚠️ The two files' §6.x numbers are **not** the same section: EN §6.1–§6.23 was the English README's own numbering, CN §6.1–§6.19 the Chinese README's. Rough alignment for the 2026-10-12 block: CN §6.6 – §6.16 = EN §6.11 – §6.21 (offset +5: block order / accumulator width / ffn-key / shape gate / zero-fill / state readback / one-submit / slot move / PROF_GPU fix / sampling ×2); CN §6.17 = the 11th paired re-measure (EN: inside §6.1); CN §6.18 = EN §6.22 (ember survey); CN §6.19 = EN §6.23 (windowed submits). **When in doubt, follow the date and the kernel name, not the number.**
- Per-topic implementation records (these hold the deeper ledgers, e.g. §6.24 – §6.28 of the windowed-submit record): `参考/2026-10-10-*.md`, `参考/2026-10-11-*.md`, `参考/2026-10-12-*.md`.

⚠️ Numbers in the archive are **frozen readings from the session that took them**. They are evidence about *what was tried and what happened*, not current throughput — for that, use §6.

## 7. Layout

```
src/            Rust sources (main / model / gpu_model / runtime / backend
                / backend_vulkan-impl / backend_cuda / vulkan/)
assets/shaders/ Vulkan compute shader sources (*.comp; spv are build artifacts,
                baked into the release binary — see VK_SPV_DIR)
tools/          Offline quantization & calibration tools (Python)
bench/          Kernel-level benchmark support
参考/           Research reference docs: the measurement archive (§6.9),
                技术实现细节.md, per-day implementation records
test/           Development scripts
examples/       Self-contained example programs (see §4.1)
```

## 8. Relationship with Albatross and cryscan/rosalia

This project initially referenced two existing RWKV inference implementations:

- **信天翁 [Albatross](https://github.com/BlinkDL/Albatross)** (Apache-2.0): a high-performance **CUDA** engine for RWKV-7 (claims 15 000+ tps decode for 7.2B fp16 on a single 5090). Our **kernel fusion, sequence-parallel batched submission, argmax sampling (transferring only token indices), compile-once/reuse pipeline** designs draw on / benchmark against its approach. ⚠️ It is **no longer installed on this machine**, so its one measured pair ("parity at B=1, **+11 % … +26 %** at B=8 – 256") cannot be refreshed and **must not be divided into §6's CUDA column**.
- **[cryscan/rosalia](https://github.com/cryscan/rosalia)**: an RWKV engine in **Rust + Vulkan compute shaders**. Our **Vulkan kernel skeleton, runtime structure, and initial design of some operators** were inspired by it. The `ember` branch was surveyed operator-by-operator on 2026-10-12 and has **no int8 tensor-core path** (archive §6.22) — i.e. nothing to copy for the gap in §6.6.

The original gap versus Albatross came from ① its more aggressive kernel fusion, ② CUDA Graph capture cutting launch overhead, ③ its mature hardware-specific libraries. It has since been closed and reversed on the CUDA path by **int8 tensor cores on int8-resident weights** (half the bytes per token, 2× the tensor-core peak), split-K, merged multi-chain launches and shape-aware tiling.

Inspired by both, this repository has since evolved independently, adding capabilities neither has: **two weight-quantization paths** (fp16 / int8 auto-routed), **CPU fp32 reference plus GPU-kernel unit tests**, and an **offline quantization toolchain** with a reproducible accuracy-verification workflow.

## 9. Known Limitations & Future Work

- **Wide-batch decode is behind CUDA (0.84 / 0.76 / 0.79× at B=64/128/256) and it is one kernel's problem** — §6.6. The remaining levers are the ones not yet in §6.8: a `batch ≤ 4` tile variant, and whatever breaks the coopmat accumulator round-trip.
- **No int8 tensor-core path for batched prefill.** Prefill runs fp16 coopmat GEMMs (`gemm_f16_f32_affine` 37 % + `_relu2` 27 % + `gemm_f16_f32` 22 % = 86 %, `dequant_int8_f16` only 0.9 %) — it is *faster than CUDA anyway*, but the int8 weight traffic is being dequantized rather than consumed.
- **Prefill dequant overhead ≈ 10 %** (int8 vs fp16 at T=512): full removal needs cooperative-matrix fused dequant + GEMM (Marlin-style zero-copy), a long-term direction.
- **Prefill time grows ~quadratically with T** (T=256: 0.33 ms → T=512: 0.81 ms, WKV parallel intra-chunk term); consider WKV chunking for long prompts.
- **Padding waste in batched prefill**: with `PAD_TO` the sequence dimension is padded to a fixed bucket, costing 4 – 5× unnecessary work when slots have unequal lengths.
- **fp16 models have no batched *decode* path** (batched prefill does work): the batched decode r/k/v kernels exist only in int8 variants, so an fp16 model falls back to per-slot single-sequence decode. `batch_decode_blocker()` returns the reason as a string so callers can log *why* they went serial instead of reading a silent `supports_batch_decode() == false`.
- **`LOWRANK_MIN` is 64 for level 1, not 32** — at B=32 the tensor-core mid arm measures a **−8 … −10 % net loss** (block starvation is fixed, but the shape still doesn't pay). B=32 is the cell a dedicated small-M tile variant would have to win.
- **Vulkan has no `RKV_MERGE` across the r/k/v chains beyond the one already shipped**, so it records 3 extra `quant_x_i8` + 3 extra `gemm_imma_reduce` per layer versus CUDA's single merged launch. Expectations for further merging are revised **down** by the small-batch fact in §6.2: it cannot rescue B=1; its value is partial traffic and host recording at large batch — which is exactly where we are already behind for a different reason.
- **The first prefill of a new length bucket is expensive**: switching length rebuilds all 45 `[bt,·]` seq buffers, which costs *seconds* — cold-bucket first request 3641.0 ms vs same-bucket repeat 2493.2 ms on Vulkan (1809.0 vs 942.7 on CUDA). `warmup_len_buckets` moves that wait from the user to startup, but **the total cost is unchanged**, and because `SeqBuffers` caches **one** shape only, just the last bucket in the warmup list stays resident — requests that alternate lengths need a fixed `pad_to` or one model per bucket.
- **`dplr_seq` occupancy is low** (40 workgroups × 64 threads + 2 barriers per token, ~10 ms in steady prefill): could adopt the CUDA structure (parallel-row blocks + shuffle reduction).
- **Measurement environment ceiling**: the card drives the display, SM clocks step in ~7 % increments (1695 – 1830 MHz) and temperatures run 46 → 82 ℃, so end-to-end resolution is ~2 – 3 % and any single-cell claim below that is not evidence. See §6.5.

## 10. License

Apache-2.0.
