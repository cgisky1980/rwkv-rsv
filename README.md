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

- **Portability first, with the cost stated**: Vulkan is the primary backend, giving cross-vendor / cross-platform reach. Measured here: **no gap to CUDA at the same SIMT 口径**, but **0.87× down to 0.27× at default gates as batch grows** (see §6.1).
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
| `UNIFORM_POOL_MB` | Vulkan uniform pool capacity in MB (default 8; usage is O(distinct params per batch) — measured peak 69 KB, so this is a fallback knob, not something to tune) |
| `GEMM_TILE_*` / `GEMV_BLOCK_SIZE` / `GEMV_ROWS` | Override cross-hardware adaptation |

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
# State serialization: forward→state_back→save→state_load→state_back lossless round-trip
cargo run --release --example state_persist     # env: OUT=state.bin
```

### 4.2 Library API & GPU Sampling

`rwkv-rsv` also exports a **library** tailored for server integration (e.g. an `ai00-server`), mirroring [web-rwkv](https://github.com/cryscan/web-rwkv):

- **`ModelBuilder` + `Bundle`**: load a model and bind a zero-initialized `State`; `infer_tokens` / `infer_seq` / `infer` advance it and return logits.
- **`State`**: first-class session state — `state_back()` downloads it to a CPU `Vec<f32>`, `state_load()` restores it, `reset()` clears it. Serialization is bit-exact.
- **GPU sampling (`SamplerParams`)**: `infer_sample` / `infer_sample_selfloop` filter logits entirely on GPU and return only the sampled token index.
- **Concurrency pool primitives (batched-layout `State`)**: `create_batch_state(B)` + `forward_seq_batch_padded` / `submit_sample_selfloop_batch` / `batch_step_sample`; slot management via `state_slot_back` / `state_slot_load` (host round-trip) and `state_slot_broadcast` (**on-device** fan-out of one state to many slots - session forks / many-sample-per-prompt, saving 2x single-slot state over PCIe per target). `warmup_len_buckets(state, &[T_pad..])` moves the per-length-bucket seq-buffer rebuild out of the first real request and into startup (★ the buffer cache holds one shape: only the last bucket stays resident). Check `supports_batch_decode()` for the batched **decode** preconditions; `batch_decode_blocker()` names *which* one fails (backend lacks the ops vs model has no int8 handles). ⚠️ That gate covers decode only - batched prefill works on fp16 models too, and the value `forward_seq_batch_padded` returns is each slot's last prompt token (the decode seed), not a predicted token.

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

**GPU decode self-loop throughput** (argmax self-loop, 1000 tokens, GPU ~53°C cold start; via `memtest` + `SELFLOOP_ONLY=1` / `SELFLOOP_N=1000`):

| Weight | Vulkan | CUDA |
|---|---|---|
| fp16 | 88.1 tok/s | 92.4 tok/s |
| int8 | 116.0 tok/s | 126.0 tok/s |

int8 ≈ +32% over fp16 (Vulkan) / +36% (CUDA).

> Re-measured **interleaved in one session on 2026-10-04** (int8, same method): Vulkan **68.4** vs
> CUDA **68.3** tok/s — dead even. The absolute values move a lot between sessions (>40% here),
> which is exactly why only the interleaved ratio is meaningful; the 2026-08 table above predates
> the Vulkan r/k/v kernel work and was not measured interleaved. See §6.1.
>
> ⚠️ The **CUDA gate 口径 of that single-stream pair was never recorded**, and it decides whether the
> "parity" holds: measured interleaved on 2026-10-10 at **default gates** (CUDA's three tensor-core
> switches on) the same comparison is **91.4 vs 79.2 = 0.87×** — already unequal at B=1. Read
> "parity at B=1" as a SIMT-vs-SIMT result.

**GPU prefill throughput** (T=256 steady-state, i.e. 2nd run onward after same-length warmup; `prof_prefill_steady`, Vulkan cooperative-matrix GEMM path):

| Weight | Vulkan steady | Notes |
|---|---|---|
| fp16 | **2677-2700 tok/s** | effective GEMM 21-25 TFLOPS (75-89% of f32-accumulate tensor peak) |
| int8 | 2263-2307 tok/s | includes per-layer dequant overhead (see Known Limitations) |

Vulkan ⇄ CUDA on this path is a wash too: interleaved in one session (2026-10-04, `PTOKENS=256`) **2455.6 vs 2433.0 tok/s** (1.01×). ⚠️ Same caveat as above — that pair's CUDA gates were not recorded, so treat it as a SIMT-口径 statement. Batched prefill is a different story: at default gates Vulkan is *faster* at B=1 (1,169 ms vs 2,238 ms) and *slower* from B≥32 onward (86.5 s vs 46.7 s at B=256) — see the prefill columns in §6.1.

A cold single-shot prefill additionally pays the one-time pipeline-creation cost of that length bucket (m_pad), ~tens to a hundred ms; servers can run one dummy prefill per common length bucket after loading. Details: [参考/Vulkan-prefill单发与稳态差异排查记录.md](参考/Vulkan-prefill单发与稳态差异排查记录.md).

### 6.1 Batched-decode throughput: CUDA / Vulkan / Albatross (int8, C=2560 / H=40 / N=64 / V=65536 / L=32)

**Main table = one same-session interleaved A/B, 2026-10-10** (`bench_ab`, session
`20261010-184051-89364`; batch and backend **alternate every round**, 3 rounds per cell, **median**
taken; raw records `bench/ab-20261010-184051-89364.jsonl` + `.summary.json`):

| Batch | CUDA tok/s | spread | CUDA prefill | Vulkan tok/s | spread | Vulkan prefill | Vulkan/CUDA | Per-cell fingerprint |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 (single-stream) | 91.4 | 1.7% | 2,238 ms | 79.2 | 3.4% | 1,169 ms | **0.87** | identical `0x1acbfa` |
| 8 | 583.1 | 1.8% | 3,719 ms | 482.4 | 5.7% | 3,712 ms | **0.83** | diverged |
| 16 | 1062.8 | 1.9% | 6,153 ms | 693.3 | 2.6% | 6,605 ms | **0.65** | diverged |
| 32 | 1778.0 | 12.6% | 8,403 ms | 861.2 | 2.3% | 11,761 ms | **0.48** | diverged |
| 64 | 2633.1 | 19.4% | 13,894 ms | 935.5 | 1.4% | 22,269 ms | **0.36** | diverged |
| 96 | 2871.7 | 8.9% | 19,407 ms | 913.2 | 3.8% | 33,020 ms | **0.32** | diverged |
| 128 | 3367.6 | 1.4% | 24,427 ms | 967.4 | 1.6% | 43,670 ms | **0.29** | diverged |
| 256 | 3726.5 | 1.6% | 46,652 ms | 988.5 | 1.6% | 86,453 ms | **0.27** | diverged |

**Gate snapshot (verbatim, identical for both backends in that session)**: `SEGS=4` `PAD_TO=512`
`NTOK=32` `WARMUP=1` `SKIP_SINGLE=1` `prompt_repeat=6` `temp=1` `top_k=0`;
`LOWRANK_GEMM` / `FFN_VALUE_GEMM` / `GEMV_IMMA` **all unset = ON** (CUDA low-rank 4-chain / FFN value
/ r·k·v all on int8 tensor cores); `VK_IMMA` on, `VK_IMMA_OPS` unset = **everything but rkv on**,
`VK_IMMA_NST`/`VK_IMMA_BM` automatic by batch/M; `SKIP_GRAPH` unset = off (self-loop uses graph
capture), `STEP` off (self-loop path only), `PENALTY` off, `SELFLOOP_CHUNK` / `ABLATE` default.

**Four things this table supports**:

1. **At default gates Vulkan is behind at every batch**, and the gap widens monotonically with B:
   0.87× at B=1, 0.83× at B=8, roughly halved from B=32 on ⇒ **0.27×** at B=256. **The previous
   version of this section ("0.98×–1.03× across B=1~256, Vulkan ahead at 128/256") does not hold at
   default gates** — that was a SIMT-vs-SIMT 口径 result (CUDA with its three tensor-core gates off);
   see the 口径 paragraph below. The widening is *consistent* with rkv being the one batched kernel
   that stays SIMT on Vulkan while CUDA runs it on int8 tensor cores (`VK_IMMA_OPS` defaults to
   everything but rkv), but **no per-op ablation was run** to pin that down, so it is offered as a
   consistent explanation, not a proven one.
2. **Absolute values must never be quoted across sessions**: CUDA's *own* spread reaches **12.6% /
   19.4%** in the B=32/64 cells (3 rounds, same backend, same parameters — comparing still needs
   ~20% headroom). The card doubles as the desktop display. Any quoted reading needs a same-session,
   interleaved, gate-spelled-out re-measure, reported as median + spread.
3. **Batched throughput is emphatically not ×B**: B=256 is only **×40.8 (CUDA) / ×12.5 (Vulkan)**
   over B=1, and Vulkan is essentially flat from B=64 on (935 → 988 tok/s for a 4× larger batch).
   Mechanism: [参考/2026-10-10-批量并发收尾与基准可信度-实施记录.md](参考/2026-10-10-批量并发收尾与基准可信度-实施记录.md) §11 —
   every wave re-reads the whole int8 weight set, at a constant 64–68 ms per wave.
4. **prefill and decode rank in opposite directions**: at B=1 Vulkan prefill is *faster*
   (1,169 vs 2,238 ms), then CUDA takes over from B≥32 and pulls away (86.5 s vs 46.7 s at B=256).

**A real cliff fixed in the same session (improvement 5)**: Vulkan fell off the tensor-core path for
**batch>64** — the old `imma_ok` predicate carried a `batch <= 64` bound, and crossing it switched
**all four tensor-core GEMMs (head/out/key/val)** back to SIMT (head = vocab 65536×2560, the largest
single matmul in a layer). Measured:

| Batch | Old gate (batch>64 off IMMA) | `VK_IMMA=0` all-batch control | After the fix (same params) |
|---:|---:|---:|---:|
| 64 | 934.4 | 436.5 | 935.5 |
| 80 | 430.7 | — | — |
| 96 | 429.2 | — | 913.2 |
| 128 | 428.4 | 422.8 | 967.4 |
| 256 | 426.2 | — | 988.5 |

≈ **2.1×–2.3×** at B=96/128/256, and the old-gate readings land in the same band as turning IMMA off
for the whole batch (429 vs 436) — that is the attribution evidence. Regression test:
`tests/batch_parity.rs::slots_beyond_the_first_wave_match` (72 slots × 8 greedy steps, bit-identical).
(⚠️ Provenance: the 64/80/96 cells and the B=128 control come from the 18:01 locator probe
`bench/probe-imma.jsonl`; the 128/256 old-gate cells from the 17:26 pre-fix sweep
`bench/ab-20261010-172659-90152.jsonl`; the "after the fix" column is the 18:40 formal session median.
Four same-day different runs — **cited only to show the "off IMMA" magnitude**, never as a ratio.)

**Albatross (external reference implementation)**: ⚠️ a **different session with different bench
parameters**, so it is a magnitude reference only and must not be divided into the CUDA column above.
That implementation is no longer installed here, so **its column cannot be refreshed**. The table
keeps its original same-session interleaved comparison against CUDA (2026-09-22, `batch_decode_bench`
vs `run_bench.bat`, alternating):

| Batch | Albatross | rwkv-rsv CUDA (that session) | Speedup (valid only inside this pair) |
|---:|---:|---:|---:|
| 1 (single-stream) | 94.2 – 99.0 | 94.1 – 97.2 | parity (1.012× over 3 interleaved rounds) |
| 8 | 465.4 | 547.8 | +17.7% |
| 16 | 806.0 | 979.4 | +21.5% |
| 32 | 1294.6 | 1632.8 | +26.1% |
| 64 | 2123.3 | 2354.7 | +10.9% |
| 128 | 2704.6 | 3123.1 | +15.5% |
| 256 | 3036.9 | 3467.7 | +14.2% |

> ⚠️ The CUDA column in that table is likewise a historical reading with **no recorded gates** (the
> three tensor-core gates have been default-on since 2026-09-21). For a trustworthy comparison,
> re-measure: `cargo run --release --example bench_ab` (`AB_BACKENDS` / `AB_SLOTS` / `AB_ROUNDS` /
> `AB_JSON`), or single-backend `batch_decode_bench` with `BENCH_JSON=1` to get JSONL.

**口径 (SIMT-vs-SIMT vs default)**: the "0.98×–1.03× parity" and "bit-identical across backends"
claims of 2026-10-04 were both **aligned-gate** results — CUDA needs
`LOWRANK_GEMM=0 FFN_VALUE_GEMM=0 GEMV_IMMA=0`. Spot check on 2026-10-10 at B=8: CUDA default 624.6 /
gates off 503.5 / Vulkan 514.6; at B=32: default 1900.7 / off 583.4 / Vulkan 929.1. Default versus
aligned differs by up to 3×, so "Vulkan caught CUDA's tensor-core path" was never true.

**Correctness — same backend, two paths**: the per-step path (`batch_step_sample`, host-in-the-loop) and the GPU self-loop path produce **bit-identical** sum/xor fingerprints at every batch size on both backends. This is the regression probe `STEP=1` in `batch_decode_bench` (added 2026-10-10 after a Vulkan read-back bug made the two paths disagree — see [参考/2026-10-10-Vulkan批量残留修复-实施记录.md](参考/2026-10-10-Vulkan批量残留修复-实施记录.md)).

**Correctness — across backends**: bit-exactness holds **only when both run the same SIMT kernel sequence**. With CUDA forced onto the Vulkan path (`LOWRANK_GEMM=0 FFN_VALUE_GEMM=0 GEMV_IMMA=0`) B=8 agrees exactly (`0xd08ada`/`0x2ea` on both). CUDA's *default* path uses int8 tensor cores (those gates are default-on since 2026-09-21), and then the fingerprints differ — of the 16 cells in the main table above, **only B=1 matches** (`0x1acbfa`/`0x6d5c`); every B≥8 diverges (e.g. B=8 CUDA `0xd02e16`/`0x6d08` vs Vulkan `0xd08ada`/`0x2ea`; B=256 CUDA `0x1a493fad`/`0x7e01` vs Vulkan `0x1a407915`/`0x6713`). The logits are numerically adjacent; sampling turns a rare single-token flip into a different tail, so this is **not** a correctness defect, but any claim of cross-backend bit-identity must state its 口径. The other half of that evidence: **the same backend reproduces its fingerprint across all 3 rounds in all 16 cells (det=Y)**, so the divergence comes from the backend, not from timing jitter. The 2026-10-04 table's "bit-identical at every batch size" line was written at aligned gates and does **not** hold against defaults. (CUDA keeps the same public `gemv_int8_rkv_stage1_batch` signature but ignores the new fp16-activation inputs and stays on its original fp32 path, so at aligned gates the CUDA column is a genuine unchanged baseline.)

**What got the CUDA path ahead** of Albatross: **int8 tensor cores** (`mma.m8n8k16.s8`, Turing's int8 TC peak is 2× its fp16
TC peak) on **int8-resident weights** (2.68 GB/token read vs 5.37 GB for an fp16 engine), **split-K
with a deterministic reduction kernel** (more blocks without extra traffic), **merged multi-chain
launches** (r/k/v + the 4 low-rank chains), **shape-aware tiling** (BM by batch, BN ≈ batch,
block count filled to the 68 SMs), **warp-per-row + `__shfl` reductions** replacing per-row block
trees, **atomic-free sparse FFN accumulation**, and every scan pass in the samplers unrolled for
memory-level parallelism. Full write-ups: [int8 IMMA record](参考/2026-09-21-int8-IMMA实施记录.md) ·
[cross-arch baseline table](参考/2026-09-21-跨架构基线表.md).

**What closed the Vulkan gap (SIMT-vs-SIMT 口径; it is not parity with CUDA's tensor-core path)** — all in the r/k/v fused kernel `gemv_int8_rkv_stage1_batch`:

- **fp16 activations**: the kernel used to read a full `vec4` (16 B) of fp32 activation per slot per k-step while the weights were already int8-packed — a second large slice of L2 traffic. It now reads an `f16vec2` buffer (8 B, halved) and widens back to `vec4` for the fp32 accumulator.
- **`vec4` weight packing + `dot`**: dequantised weights are packed per row as `vec4` and combined with the activation `vec4` via `dot()` — 4 FMAs collapse to 1 instruction.
- **Tiled `x` resident in shared memory** (`XTILE=256`): each k-element of `x` was re-read by all `BGRP=8` slots × `ROWS` rows; it is now loaded once per tile. smem drops from 80 KB (over the 48 KB sm_75 limit) to 32 KB.
- **`BGRP=8` slot grouping** for weight reuse across slots.

Details and pitfalls: [参考/2026-10-04-Vulkan追平CUDA实施记录.md](参考/2026-10-04-Vulkan追平CUDA实施记录.md).

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
| Weight precision | fp16 | fp16 / **int8** (auto-routed, 8 GB-VRAM floor) |
| Relative gap (CUDA batched decode) | baseline | **ahead** — that same-session pair gave parity at B=1 and +11% to +26% at B=8–256; Albatross is gone from this machine, so **the column cannot be refreshed and must not be divided into §6.1's CUDA column** |
| Relative gap (Vulkan vs our own CUDA) | n/a | **Two 口径, and they disagree**: ① *SIMT vs SIMT* (CUDA's three tensor-core gates off) — parity, 0.98×–1.03× across decode B=1…256 and steady prefill; ② **defaults** (CUDA tensor cores on, Vulkan on for everything but rkv) — **behind at every batch**, 0.87× at B=1 down to **0.27× at B=256** (same-session interleaved medians, §6.1) |

The original gap came from ① Albatross's more aggressive kernel fusion; ② CUDA Graph capture cutting launch overhead; ③ CUDA's mature hardware-specific libraries. It has since been closed and reversed on the CUDA path by **int8 tensor cores on int8-resident weights** (half the bytes per token, 2× the tensor-core peak), split-K, merged multi-chain launches and shape-aware tiling. The **Vulkan** path was brought to the same level *at the same SIMT 口径*: halving activation traffic in the r/k/v fused kernel (fp16 activations), `vec4`+`dot` weight handling and shared-memory tiling of `x` closed the last ~40%, so Vulkan is throughput-neutral against CUDA **when CUDA is held to the same kernel sequence** while keeping cross-vendor reach. Against CUDA's *default* tensor-core path Vulkan is not neutral — the r/k/v batched kernel is the one op Vulkan keeps on SIMT by default (`VK_IMMA_OPS`), and that is where most of the B-scaling gap lives (see §6.1, 口径 paragraph).

Inspired by both, this repository has since evolved independently, adding capabilities neither has: **two weight-quantization paths** (fp16 / int8 auto-routed), **CPU fp32 reference plus GPU-kernel unit tests**, and an **offline quantization toolchain** with a reproducible accuracy-verification workflow.

## 9. Known Limitations & Future Work

- **prefill dequant overhead ~10%** (int8 vs fp16 at T=512): full removal needs cooperative-matrix fused dequant+GEMM (Marlin-style zero-copy), a long-term direction.
- **prefill time grows ~quadratically with T** (T=256: 0.33ms → T=512: 0.81ms, WKV parallel intra-chunk term); consider WKV chunking for long prompts.
- **Vulkan dplr_seq occupancy is low** (40 workgroups × 64 threads + 2 barriers per token, ~10ms in steady prefill): could adopt the CUDA structure (parallel-row blocks + shuffle reduction).
- **First prefill of a new length bucket pays pipeline creation** (kernel cache now decouples uniforms via a shared pool + DYNAMIC offsets keyed on (shader, spec): 1722 → ~30 entries; the remainder is the inherent first-appearance cost of each spec): eliminable via startup warmup. ★ Measured 2026-10-10 (`BUCKET_BENCH=1 SLOTS=8`, 3B int8, fresh bucket = `PAD_TO+128`) - and it is **more** than pipeline creation: switching length rebuilds all 45 `[bt,·]` seq buffers, which costs *seconds*, not ms - cold-bucket first request 1809.0 ms vs same-bucket repeat 942.7 ms on CUDA (864.1 ms saved), 3641.0 vs 2493.2 ms on Vulkan (1152.2 ms saved). `warmup_len_buckets` moves that wait from the user to startup; **total cost is unchanged** (warmup itself costs 1845.3 / 3623.4 ms ≈ the cold first request), and because `SeqBuffers` caches **one** shape only, just the last bucket in the warmup list stays resident - requests that alternate lengths need a fixed `pad_to` or one model per bucket.
- ~~**Vulkan uniform pool must be sized by hand for long batches**~~ — **stale, verified resolved** (2026-10-10). The claim predates `pool_slots` dedup (ba08642, 2026-08-25), which made pool use O(distinct params per batch) rather than O(dispatches). Measured high-water with `PROF_POOL=1`: 69,080 B (batched prefill) / 33,432 B (self-loop) / 31,344 B (`batch_step_sample`) — **flat across B=8/16/32/64 and NTOK=32/64/256**, including `SKIP_GRAPH=1` (all n rounds in one submit), `STEP=1 PENALTY=1` and long prompts. Default 8 MB therefore runs every batch path with ~1.2 % of the pool; `UNIFORM_POOL_MB` is only an extreme-shape fallback.
- **fp16 models have no batched *decode* path (batched prefill does work)**: the batched decode r/k/v kernels exist only in int8 variants (`gemv_int8_rkv_stage1_batch`), so an fp16 model falls back to per-slot single-sequence decode; `forward_seq_batch_padded` (batched prefill) uses the seq-parallel kernels and reads fp16/fp32 weights fine - verified on both backends 2026-10-10 by comparing the **per-slot state after batched prefill against the single-stream state, bitwise identical** (that is the assertion that actually pins "pad rows never enter the state"; the seeds the call returns are each slot's *last prompt token*, i.e. the decode seed, **not** a next-token prediction - comparing them to single-stream argmax is a category error, and an earlier version of this test did exactly that). `GpuModel::batch_decode_blocker()` returns the reason as a string so callers (client session pool, mytown engine) can log *why* they went serial instead of reading a silent `supports_batch_decode() == false` - and instead of over-reacting to "no batched forward path" and serializing prefill too.
- **Vulkan has no batched tensor-core GEMM for the low-rank / FFN-value chains** (`lowrank_gemm_batch` / `ffn_value_gemm_batch` are CUDA-only): the other batched tensor-core ops (head/out/key/val) **are** on by default (`VK_IMMA_OPS` = everything but rkv). The Vulkan tensor-core r/k/v variant was measured at a net **3.3× loss** and is off by default (`imma_op_on("rkv") == false`; infra kept, reproducible via `VK_IMMA_OPS=...,rkv`) — each extra dispatch inside a batch costs more than the kernel saves on this driver (43 ms/step lands *between* dispatches). Consequence: at large batch CUDA's default path is ~1.2× (B=8) to ~2.0× (B=32) ahead — measured 2.06× at B=32 and **3.77× at B=256** in the 2026-10-10 same-session A/B (§6.1); closing that needs **fewer, bigger** Vulkan dispatches, not more. What reaches parity today is the SIMT fused kernel *against CUDA's SIMT path*.
- ~~**Vulkan's tensor-core batched GEMM only supports batch ≤ 64**~~ — **that was a bug, fixed 2026-10-10**. `imma_ok`'s hard `(1..=64).contains(&batch)` predicate silently dropped **B>64 onto the SIMT path**. Note what that actually cost: it is *not* "rkv fell back to SIMT" (rkv never used IMMA by default anyway - see `imma_op_on`), it is **all four tensor-core GEMMs - head/out/key/val - switching off together**, and head (vocab 65536×2560) is the single largest matmul in a layer. Throughput collapsed to a B=1-sized plateau: measured cliff at the same session and parameters - B=64 tensor core **934.4** tok/s vs B=80 **430.7** / B=96 **429.2** under the old gate (control: B=64 with IMMA forced off via `VK_IMMA=0` = **436.5**, i.e. the drop was 100 % the gate). Multi-wave (`grid.z = ceil(batch/(8·NST))`) had **already been written** into `gemm_imma.comp` - load guarded by `s0base + (i>>5) < BATCH`, store by `slot >= BATCH` - it had simply never been *executed*. Lifting the gate exposed two more sizing bugs, also fixed: `WorkBuffers.xb_q`/`xb_aux` were capped at `bsz.min(64)` while the quantize kernel writes all `bsz` rows by global slot index (wave 2 wrote out of bounds → `ERROR_DEVICE_LOST` at B=128), and `imma_partial` must **not** scale with `bsz` (`imma_ksplit` shrinks `ks` when `ks·batch·m > cap`, so a `bsz`-sized reserve asks for 63 MiB more and hits `ERROR_OUT_OF_DEVICE_MEMORY` at B=256). After the fix (formal-session medians): B=96 **913.2**, B=128 **967.4**, B=256 **988.5** tok/s, i.e. **2.1×–2.3×**. Guard test: `tests/batch_parity.rs::slots_beyond_the_first_wave_match` (72 slots straddling the 64-slot wave boundary, greedy, bitwise equality against wave 0) - this class of bug is invisible to throughput sweeps and shows up as user-reported garbage above B=64.
- **Accuracy headroom**: calibration-weighted k-means, G=64 denser metadata.

## 10. License

MIT. See [Cargo.toml](Cargo.toml) for dependencies. Research-oriented; please verify model weights (RWKV-7 Goosed weight license) and third-party dependency licensing before commercial use.