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

- **Portability first, with the cost stated**: Vulkan is the primary backend, giving cross-vendor / cross-platform reach. Measured here: **no gap to CUDA at the same SIMT 口径**, and at **default gates batched decode is 0.941× (B=1) / 0.945× (B=8) / 0.806× (B=32) / 0.730× (B=64) / 0.690× (B=128) / 0.762× (B=256)** — ★ 8th same-session re-measure, 2026-10-12, 口径 = `power.limit` 300 W with SM clocks drifting 1680…1845; absolute ms/step Vulkan 12.40 / 14.71 / 22.86 / 34.28 / 58.53 / 100.48 vs CUDA 11.67 / 13.89 / 18.43 / 25.01 / 40.37 / 76.54, so the gap is 0.7–1.0 ms/step at low concurrency and 18.2 / 24.0 ms/step at B=128/256. ⚠️ Rows above this one in §6.1 are the 250 W 档 and **cannot be subtracted across 档** (the 5th round read 0.900 / 0.972 / 0.824 / 0.765 / 0.690 / 0.688; in that session CUDA's *own* absolutes drifted −6.4…−7.7 % versus the 4th, so the ratio jump there is mostly a session effect on the reference side — credit the group-major layout with its own same-session A/B, **+6.8…9.3 %**, see §6.1 "read the absolutes before the ratios"). Batched prefill is **not** monotonically behind (8th-round same-session six-width pair, §20.2): Vulkan **leads** at B=1/8/32 (2.47× / 1.44× / 1.085×) and trails 6.1 / 12.4 / 16.5 % at B=64/128/256 (0.94 / 0.89 / 0.86×) — the older "0.79…0.84×" was a large-width-only reading. ⇒ **the gap is decode-only; the two stages have different bottlenecks** (§6.1 / §6.6-§6.9).
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
| `VK_SPV_DIR` | **Shader A/B instrument.** Load every `*.spv` from this flat directory instead of the embedded copy. Required because release builds bake `assets/shaders/spv/*.spv` into the binary (`#[derive(RustEmbed)]` in [src/vulkan/asset.rs](src/vulkan/asset.rs)) — copying new bytes onto disk does **not** change what a running exe executes, so without this knob you silently benchmark the old kernels while believing you swapped them (this is what voided §6.10's numbers). Point it at a directory holding one variant's bytes and log the md5 of each spv actually bound. |
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

Vulkan ⇄ CUDA on this path is a wash too: interleaved in one session (2026-10-04, `PTOKENS=256`) **2455.6 vs 2433.0 tok/s** (1.01×). ⚠️ Same caveat as above — that pair's CUDA gates were not recorded, so treat it as a SIMT-口径 statement. Batched prefill is a different story: at default gates Vulkan was *faster* at B=1 (1,169 ms vs 2,238 ms) and *slower* from B≥32 onward (86.5 s vs 46.7 s at B=256) — see the prefill columns in §6.1. **Corrected 2026-10-11 (§6.6, re-measured clean in §6.7)**: that B≥32 lag was mostly the fp16 GEMM tile default, not batching - the tile fix is **1.68x** on batched prefill. On the clean protocol (§6.7: chunk-loop clock, no warmup segments) batched prefill is now **0.84x** of CUDA at B=128 (18.2 vs 15.3 s) and **0.79x** at B=256 (36.0 vs 28.5 s); the B=256 gap that was attributed to "1.3 s per chunk outside the kernels" was 210 ms of per-slot state hand-back, now a device-side row copy.

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

**Re-measured 2026-10-11** (same harness, same gate defaults, session `20261011-040726-67976`, 2
rounds/cell median, decode only - the table above's decode column predates §6.3's rkv tensor-core default
flip and §6.4/§6.5, so it is kept as the pre-chain baseline):

| Batch | CUDA tok/s | spread | Vulkan tok/s | spread | Vulkan/CUDA | was (2026-10-10) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 590.2 | 1.4% | 499.8 | 1.1% | **0.847** | 0.83 |
| 32 | 1786.9 | 1.5% | 1210.8 | 1.8% | **0.678** | 0.48 |
| 128 | 3440.8 | 2.2% | 1626.9 | 1.2% | **0.473** | 0.29 |

Vulkan B=128 decode went 967.4 -> **1 626.9 tok/s (+68 %)** while CUDA stayed inside its own drift
(3 367.6 -> 3 440.8), so the 2026-10-11 chain (§6.3 rkv flip + §6.4 `gemm_imma` fetch structure + §6.5
v-chain split-K) is worth **+0.18x** at B=128 and **+0.20x** at B=32. Cross-backend fingerprints still
diverge (expected - different kernel implementations, see the 口径 note in
`tests/batch_parity.rs`). **Decode remains the largest open gap** (prefill now *leads* at B≤32 and is
0.86–0.94× at B≥64, §20.2 of the 2026-10-11 ledger), and it is
still *widening* with B.

**Re-measured again the same day after §6.8** (packed `gemm_imma` smem staging; session
`20261011-043110-89384`, same harness and gates, 2 rounds/cell median):

| Batch | CUDA tok/s | spread | Vulkan tok/s | spread | Vulkan/CUDA | was (a few hours earlier) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 585.4 | 1.2% | 512.1 | 0.8% | **0.875** | 0.847 |
| 32 | 1782.0 | 1.2% | 1275.2 | 0.4% | **0.716** | 0.678 |
| 128 | 3426.6 | 1.4% | 1739.7 | 1.4% | **0.508** | 0.473 |

CUDA again inside its own drift; **Vulkan's own decode fingerprints are unchanged**
(`0xd0c265`/`0x7b05` at B=8, `0x3460382`/`0x16fa` at B=32, `0xd1b5310`/`0x7bfc` at B=128 — byte-identical
to the session above), so this is a pure speed change (§6.8).

**Re-measured a third time the same day, after the split-K rule re-tune (§6.8's open item (i), session
`20261011-045748-40368`, 3 rounds/cell median)**:

| Batch | CUDA tok/s | spread | Vulkan tok/s | spread | Vulkan/CUDA | was (after §6.8) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 617.7 | 1.4% | 536.5 | 0.7% | **0.868** | 0.875 |
| 32 | 1869.8 | 1.5% | 1330.4 | 1.6% | **0.711** | 0.716 |
| 128 | 3587.6 | 2.0% | 1881.8 | 1.9% | **0.525** | 0.508 |

Only B=128 moves, and that is exactly what the rule predicts: at B=8/32 the new
"whole-grid blocks" formula selects the **same** ks as before for every shape (`out` stays 5 because
the divisibility step drags 8 back down, `value` stays 8, `key` stays 2) — and those two rows confirm
it with **byte-identical Vulkan fingerprints** to the session above (`0xd0c265`/`0x7b05`,
`0x3460382`/`0x16fa`), so their ±0.7 % is session drift, not a result. At B=128 the rule takes
`key` 2→1, `out`/`rkv` 5→4, `value` 8→4, and **Vulkan's fingerprint changes** (`0xd1b5310`/`0x7bfc` →
`0xd21f6bb`/`0x78b3`, stable over all 3 rounds) because split-K changes the summation order — the
contract that already covers the v-chain (§6.5): same quantisation, same fp32 accumulation, different
association. Incidentally the retuned Vulkan lands *closer* to CUDA's token stream at B=128 than the
in-order variant did: `|0xd21f6bb − 0xd21f638| = 131` versus `435 496` before (§6.8's session), which is
an observation about one deterministic greedy prompt, not a proof of accuracy.

**Re-measured a fourth time, after the fp16-state arm (§9) and the low-rank level-2 tensor core (§6.9) —
session `20261011-074644-90876`, same harness, same protocol (`SEGS=4 NTOK=32 PAD_TO=512 WARMUP=1`,
graph path ON, all gates default), 3 rounds/cell median**:

| Batch | CUDA tok/s | spread | Vulkan tok/s | spread | Vulkan/CUDA | was (after §6.8's split-K retune) |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 616.7 | 1.8% | 538.8 | 0.3% | **0.874** | 0.868 |
| 32 | 1 875.6 | 1.6% | 1 388.0 | 1.2% | **0.740** | 0.711 |
| 128 | 3 600.8 | 2.0% | 2 257.2 | 2.4% | **0.627** | 0.525 |

**CUDA's three fingerprints are byte-identical to the previous session** (`0xd02e16` / `0x345e198` /
`0xd21f638`), so every move here is Vulkan-side. B=8 is the built-in control: §6.9's gate is
`batch >= 32`, so at B=8 nothing but the fp16 state changed and the ratio moves 0.868 → 0.874 = drift.
Where §6.9 *is* on, the ratio moves with it: **+19.4 % at B=128** (Vulkan absolute 1 881.8 → 2 257.2)
and +4.1 % at B=32, matching §5.3's batch-width sweep of that lever (per-step +14.4 % at B=128,
+2.8 % at B=32 — the smaller end-to-end number at B=32 is what you expect when the same fixed-cost
kernels dominate a cheaper step). Vulkan's fingerprints all changed (`0xd0c265` → `0xd03618` at B=8,
`0x3460382` → `0x34dc29f`, `0xd21f6bb` → `0xd23806b`) — two numeric levers landed since, and both are
covered by their own element-wise accuracy gates (§4.3 here, §9's four-way gate for the state arm).
⚠️ So this table's *delta* is the two levers combined; the isolated §6.9 attribution is §5.2/§5.3, not here.

**Re-measured a fifth time, after the int8 weight **group-major layout** (`参考/2026-10-11-Vulkan-gemm_imma取数结构-实施记录.md`
§6.6-§6.9, commit `0f4dbeb`; session `20261011-110103-83952`, same harness/protocol, 3 rounds/cell median,
and for the first time in this chain **all six batches** including B=1/64/256)**:

| Batch | CUDA tok/s | spread | Vulkan tok/s | spread | Vulkan/CUDA | was (4th) |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 89.5 | 0.8% | 80.6 | 3.2% | **0.900** | (never in this table) |
| 8 | 577.2 | 6.9% | 561.3 | 0.9% | **0.972** | 0.874 |
| 32 | 1 731.5 | 5.7% | 1 427.6 | 5.0% | **0.824** | 0.740 |
| 64 | 2 593.1 | 2.9% | 1 982.8 | 1.5% | **0.765** | (never in this table) |
| 128 | 3 329.9 | 1.5% | 2 296.0 | 2.0% | **0.690** | 0.627 |
| 256 | 3 713.8 | 5.1% | 2 556.1 | 25.8% | **0.688** | 0.27 (2026-10-10, stale) |

**CUDA is again a static reference**: its B=8/32/128 fingerprints are byte-identical to the 4th session
(`0xd02e16` / `0x345e198` / `0xd21f638`), which is what makes the ratio moves readable —
`int8_idx_group_major()` is Vulkan-only, so CUDA never saw the permutation.
⚠️ **Vulkan's fingerprints all moved** (`0xd03618`→`0xcfde9a`, `0x34dc29f`→`0x3488a87`,
`0xd23806b`→`0xd1e3a7b`), and that is expected rather than alarming: **three** commits landed between
the 4th session and this build — `07ba7dc` (state-kernel adjacent-column pairs + two-row interleave,
which *changes the `subgroupAdd` summation partition*, so it is ulp-level different, not bit-identical),
`0c1fae4` (the `VK_SPV_DIR` instrument + a probe tile that is off by default, numerics-neutral) and
`0f4dbeb` (the layout itself, a pure permutation of the same bytes ⇒ numerics-neutral by construction:
`int8_idx_to_group_major` writes `dst[g·M·32 + m·32 + j] = src[m·(K/4) + g·32 + j]` and every reader's
address math is the exact inverse). The gate that actually covers this is the test suite, not the
fingerprint: `cargo test --release` passes (11 + 1 tests, incl. the batch-path parity and the sparse-FFN
vs-CPU check). ⚠️ What it *cannot* prove is bitwise identity with the pre-layout build — and the two
cross-backend gates I tried first are not discriminators here: `CROSS_BACKEND=1` at *default* gates reports
first divergence at step 0 (expected — the two backends run different kernel sets, so it only becomes
meaningful once the SIMT 口径 env knobs are aligned), and `DIAG=1` against the CPU fp32 reference on the
3B **int8** model is a wide band on both sides (Vulkan seq-vs-CPU `max_abs = 0.319`, CUDA's own
`0.428`, `tok` paths up to `5.17`, all agreeing on top-1 with a 45/46 rank swap). So the correctness
argument for the layout is **structural + suite-green**: the permutation is a bijection, every reader's
address math is its exact inverse, and a `grep` audit over `assets/shaders/src/*.comp` finds **no** kernel
that reads `input_idx`/`widx` without the matching `IDX_GS` math. ⚠️ Note the corollary for §6.10: the pairs/interleave commit *is* in this build, but its measured
"−0.98 %" was produced by the broken instrument, so it contributes nothing to these cells — the layout
does, and its isolated effect is the same-session two-arm end-to-end A/B (+6.8…9.3 % at B=1/8/32).

**Read the absolutes before the ratios.** CUDA's fingerprints being byte-identical proves its *code* is
unchanged — it does **not** make its throughput comparable across sessions, and this session is the first
of the five where that mattered: CUDA came in **6.4…7.7 % below its own 4th-session values at all three
shared cells** (616.7 → 577.2, 1 875.6 → 1 731.5, 3 600.8 → 3 329.9) while Vulkan came in **above** them
(538.8 → 561.3, 1 388.0 → 1 427.6, 2 257.2 → 2 296.0). So the ratio jumps (0.874 → 0.972, 0.740 → 0.824,
0.627 → 0.690) are roughly **one third Vulkan-side and two thirds a CUDA session effect** — and the reason
the 2nd/3rd/4th tables could be compared across sessions is simply that CUDA's absolutes happened to sit
within 1 % of each other there, not because cross-session ratios are safe. ★ New discipline for §6.1:
**a ratio is only citable against the session that measured both arms; to credit a lever with a ratio
move, cite the lever's own same-session A/B** (here: +6.8…9.3 %), and use the canonical table only for
the level. Consistency check that the two do agree: the A/B's NEW arm at B=32 (1 432-1 467 tok/s) brackets
this session's 1 427.6 ✓.

What the table does establish on its own: **B=1 (0.900) and B=64 (0.765) enter it for the first time**, and
**B=256 is 0.688, not 0.27** — the §9 0.27× row measured a long-superseded build, which retires the
"capacity cliff at large batch" story for the fourth time. ⚠️ Caveat on that cell: its 3-round spread is
25.8 % (readings 2 588 / 2 058 / 2 556 — round 2 is an outlier of the same family the 4th session recorded
there); excluding it the cell is 0.688–0.697.

**Two more paired sessions ran after that, and they redefine the 口径 (measurement settings) — read this
before using any number above.** The 6th (`20261011-225925-17792`) was taken with the GPU **clock-locked**
(`nvidia-smi -lgc`, which in fact pins SM at **1860 MHz** against a 2100 MHz max, power limit still the
250 W default); the 7th (`20261012-004233-91772`) was taken after the **power limit was raised to 300 W**.
Because both sides of a pair must share one environment, the table below is now the **canonical level for
the 300 W 口径**, and everything above it belongs to 250 W sessions:

| Batch | CUDA tok/s | spread | CUDA ms/step | Vulkan tok/s | spread | Vulkan ms/step | Vulkan/CUDA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 85.7 | 9.2% | 11.67 | 80.6 | 7.7% | 12.40 | **0.941** |
| 8 | 575.9 | 8.8% | 13.89 | 544.0 | 7.8% | 14.71 | **0.945** |
| 32 | 1 735.9 | 9.7% | 18.43 | 1 399.8 | 12.7% | 22.86 | **0.806** |
| 64 | 2 558.7 | 4.0% | 25.01 | 1 866.8 | 10.8% | 34.28 | **0.730** |
| 128 | 3 170.6 | 2.5% | 40.37 | 2 186.8 | 3.0% | 58.53 | **0.690** |
| 256 | 3 344.6 | 6.3% | 76.54 | 2 547.7 | 2.2% | 100.48 | **0.762** |

All twelve fingerprints are byte-identical to the clock-locked session (`0x1acbfa`/`0xd02e16`/`0x345e198`/
`0x68f1ce4`/`0xd21f638`/`0x1a49f841` on CUDA, `0x1a747c`/`0xcfde9a`/`0x3488a87`/`0x690a7d1`/`0xd2150da`/
`0x1a45b4d4` on Vulkan) ⇒ same build, so the moves against the locked 250 W run are pure environment:
**B=1 0.865 → 0.941** (Vulkan 76.9 → 80.6 while CUDA 89.0 → 85.7 — the "single-stream regression" I
reported mid-session was the 1860 MHz pin plus that cell's own ±17 % round spread, not code: the two
gates that landed are `batch >= 32/64` and split-K is off by default, so B=1 executes identical kernels),
**B=32 0.715 → 0.806** (Vulkan +160.5 tok/s, 25.82 → 22.86 ms/step), but **B=128 0.733 → 0.690**
(Vulkan −141.9, 54.97 → 58.53 ms/step while CUDA moved only −7.6) and B=256 −225.0. 1 Hz sampling over
that session (774 points) explains the split: with the wall at 300 W the card draws up to **310.9 W** and
reaches **82 °C**, and the SM clock is no longer pinned — it floats **1680…1845** (mean 1795). So raising
the limit buys the mid-width cells (which were power-capped) and costs the heaviest cells (which are now
thermal-capped), and Vulkan's large-batch path is the more heat-sensitive of the two. ★ Discipline: from
now on every §6.1 row states its **power 档 + clock 档**, and a cell may only be compared against rows
measured in the same 档. Consequence for the ledger: the level-1 coopmat arm keeps its own accounting
(§15: +8.8 % at B=128 from a same-session two-arm A/B at 1860 MHz/250 W) — do **not** recompute it from
this table, where the same build reads 58.53 ms/step because the GPU is hot.

**Gate snapshot (verbatim, identical for both backends in that session)**: `SEGS=4` `PAD_TO=512`
`NTOK=32` `WARMUP=1` `SKIP_SINGLE=1` `prompt_repeat=6` `temp=1` `top_k=0`;
`LOWRANK_GEMM` / `FFN_VALUE_GEMM` / `GEMV_IMMA` **all unset = ON** (CUDA low-rank 4-chain / FFN value
/ r·k·v all on int8 tensor cores); `VK_IMMA` on, `VK_IMMA_OPS` unset — ⚠️ this snapshot's wording
"**everything but rkv**" is the **pre-§6.3** default and is stale for every session after the 2026-10-10
flip (including the 3rd and 4th tables above), where unset = **everything on**;
`VK_IMMA_NST`/`VK_IMMA_BM` automatic by batch/M; `SKIP_GRAPH` unset = off (self-loop uses graph
capture), `STEP` off (self-loop path only), `PENALTY` off, `SELFLOOP_CHUNK` / `ABLATE` default.
From the **4th** session onward two more gates are unset-but-ON and both are Vulkan-side:
`WKV_STATE_F16` (fp16 state storage, §9) and `LOWRANK_GEMM2` (low-rank level-2 tensor core at
`batch >= 32`, §6.9). From the **5th** session onward, one more: `LOWRANK_GEMM1` (level-1 mid
projections on coopmat fp16, `batch >= 64`, §6.9 / record §11) — and `PROF_LOW=1` is the
perf-independent binding marker that says which of the three gates actually fired.
From the **6th** session onward two more knobs exist and **both are unset in the default 口径**:
`GEMM_SK` (level-1 coopmat split-K; unset ⇒ `ks=1` = the shipped path — the arm is **in-tree but
default OFF**, its measured throughput case was falsified, record §12) and `LOWRANK_MIN` (the
level-1 batch gate, default 64; raise/lower it to re-measure B=32/48 without a rebuild).

**Four things this table supports**:

1. **At default gates Vulkan is behind at every batch**, and the gap widens monotonically with B:
   0.87× at B=1, 0.83× at B=8, roughly halved from B=32 on ⇒ **0.27×** at B=256. **The previous
   version of this section ("0.98×–1.03× across B=1~256, Vulkan ahead at 128/256") does not hold at
   default gates** — that was a SIMT-vs-SIMT 口径 result (CUDA with its three tensor-core gates off);
   see the 口径 paragraph below. The widening is *consistent* with rkv being the one batched kernel
   that stays SIMT on Vulkan while CUDA runs it on int8 tensor cores (`VK_IMMA_OPS` defaults to
   everything but rkv), but **no per-op ablation was run** to pin that down, so it is offered as a
   consistent explanation, not a proven one.
   ⚠️ **Three corrections from §6.2 / §6.1's 5th re-measure**: (i) the premise inside that explanation — "tensor-core rkv
   would be slower on Vulkan" — is false; once its crash is fixed it is faster at every batch
   (+27.9 % … +42.9 %), so the recoverable share of this gap is much larger than assumed, it is just
   blocked on an unvalidated numeric contract rather than on speed; (ii) **the B=256 cell is not a clean
   kernel-quality reading** — at B ≥ 192 the Vulkan side fills VRAM and WDDM spills to system memory
   (sampler 0.87 ms → 68 ms per call, §6.2③), so 0.27× mixes in a capacity effect. B=64/128 are unaffected;
   (iii) **and (ii)'s 0.27× no longer reproduces at all** — the 5th re-measure gets B=256 = **0.688×**, sitting
   on the same monotone curve as B=64/128, so that cell had been measuring a stale build rather than a
   capacity cliff (see the B=256 note under the tables).
2. **Absolute values must never be quoted across sessions**: CUDA's *own* spread reaches **12.6% /
   19.4%** in the B=32/64 cells (3 rounds, same backend, same parameters — comparing still needs
   ~20% headroom). The card doubles as the desktop display. Any quoted reading needs a same-session,
   interleaved, gate-spelled-out re-measure, reported as median + spread.
3. **Batched throughput is emphatically not ×B**: B=256 is only **×40.8 (CUDA) / ×12.5 (Vulkan)**
   over B=1, and Vulkan is essentially flat from B=64 on (935 → 988 tok/s for a 4× larger batch).
   Mechanism: [参考/2026-10-10-批量并发收尾与基准可信度-实施记录.md](参考/2026-10-10-批量并发收尾与基准可信度-实施记录.md) §11 —
   every wave re-reads the whole int8 weight set, at a constant 64–68 ms per wave.
   ⚠️ **That reading of the mechanism is now falsified for the tensor-core kernel (2026-10-11, §6.8)**:
   making one wave cover 128 slots instead of 64 (so the weights are read **once** instead of twice)
   made `gemm_imma` **7–14 % slower**, and the flat-throughput curve persists. Cost per wave is set by
   per-block *instruction issue*, not by weight re-read traffic — CUDA's own `IMMA_DB`/`IM_PIPE`
   dead-ends (`backend_cuda.rs` ≈1866) say the same thing from the other side.
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

### 6.2 Follow-up measurements (2026-10-10): three corrections to §6.1

All three were taken **interleaved inside one session** (this GPU is also the display adapter, so
cross-session numbers are not comparable). Two previously documented conclusions were overturned,
and one "performance loss" turned out to be a crash.

**① `GEMV_BGRP` tops out at 8 — "more slots per weight pass" does not keep paying off** (it had never
been A/B'd at batch ≥ 16). At B=256, everything else fixed: BGRP 4 → 795.2, **8 → 986.9 / 974.3**
(repeat run, 1.3 % drift), 16 → 890.3, 24 → 750.7, 32 → 718.1 tok/s. The bandwidth model says
BGRP=32 reads r/k/v weights 8 times instead of 32 and should win outright; it loses by 27 %, because
`float acc[ROWS][BGRP]` spills to local memory past 8. **Every BGRP produced the identical
fingerprint** (`sum=0x68ce489 xor=0x7243`), which also proves the `slot_cnt` tail guard is correct when
BGRP does not divide the batch (24). ⇒ the B=256 gap cannot be closed by SIMT group tuning; it needs a
path that reuses one weight read over many more slots (see ②).

**② The tensor-core r/k/v path was never "3.3× slower" — it crashed.** It raised `ERROR_DEVICE_LOST`
at every batch (256/128/64/8/4/2/1), so the 3.3× figure could not be reproduced at all. Bisection
landed on `gemv_int8_rkv_mid_batch` (the `RKV_OFF=1` mid-projection arm): its `params` array filled
**24 entries for a `Params` struct holding 26 `buffer_reference`s**, so every slot from `in_g1`
onward shifted by two and `out_am` was read out of never-written pool bytes and dereferenced as a
pointer. A fixed shift is why it looked branch-specific: the v/w arms land on still-valid pointers and
**run, computing wrong values**, while the a/g arms hit garbage and **fault**. That path had not
executed successfully once since it was written (`3374512`). After the fix, interleaved against the
default SIMT path (`VK_IMMA_OPS=…,rkv`):

| batch | 32 | 64 | 128 | 256 |
|---|---|---|---|---|
| SIMT r/k/v | 849.0 | 921.4 | 965.1 | 679.0 / 693.0 |
| tensor-core r/k/v | **1086.1** | **1231.0** | **1378.9** | **899.0 / 906.7** |
| ratio | +27.9 % | +33.6 % | **+42.9 %** | +29.7 % |

It wins at every batch and the two repeats do not overlap (so it is not drift). **It still ships
default-off**, for the one gate that has not been passed: this path quantizes the r/k/v *activations* to
int8, i.e. a different numeric contract from SIMT's "fp32 activation × int8 weight" (fingerprints differ
at every batch, as expected), and the end-to-end quality gate (logit deviation / top-1 agreement over a
prompt batch) has not been run. Swapping the default numeric contract without quality evidence is not a
performance optimisation.

**③ The B ≥ 192 throughput cliff is VRAM spill, not kernel quality.** With `TOPP=1.0` (top-p fallback
loop disabled) the concurrency curve reads:

| batch | 64 | 128 | 192 | 256 |
|---|---|---|---|---|
| decode tok/s | 1275.3 | **1430.1** | 1034.7 | 1019.1 |
| `sample.spv`, per call | 0.58 ms | 0.87 ms | **49.99 ms** | **68.0 ms** |

The sampler jumps **57×** between 128 and 192 while the same code grows only 1.5× from 64 to 128 —
a memory-placement signature, not an algorithmic one. `nvidia-smi` during B=256 decode measured
**21 775 / 22 528 MiB** (this card is a 22 GB mod; desktop plus other processes hold ~8.6 GB, our
process peaks at ~13.1 GB). Once VRAM is full WDDM **silently falls back to system memory**, so the
sampler's four `[batch, vocab]` workspaces (67 MB each at B=256) become PCIe traffic. Two consequences:
- **the practical sweet spot for batched decode is B ≈ 128**, not 256; pushing SLOTS to 256 is negative
  returns.
- the **0.27× at B=256** in §6.1's default 口径 is **partly a capacity effect**, not kernel quality. A clean
  comparison needs the per-slot footprint reduced first (candidates: the RNN state, `H·N·N = 163 840`
  elements per layer per slot ⇒ ~5.3 GB at B=256, and the per-slot `WorkBuffers` workspace).
  **State footprint: done 2026-10-11** — Vulkan now has the same fp16 *storage* arm CUDA has
  (`supports_f16_wkv_state() = true`, `_s16` spv variants, `WKV_STATE_F16=0` to switch back), measured
  **−1290 MiB of registered tensors at B=128** (7788 → 6498 MiB) and **+3.5 % decode** same-session
  interleaved; ⇒ ~2.6 GB back at B=256. `WorkBuffers` is untouched.
  ⚠️ **The premise of this bullet is retired** (5th re-measure, 2026-10-11): B=256 is **0.688×** and monotone
  with B=64/128, so the 0.27× was a stale build, not a capacity effect — the "cliff at large batch" story
  has now been retracted four times and the footprint work above stands on its own merits (VRAM, not ratio).

Also exposed by ③: `log_vram` printed "registered tensors 0 MiB" throughout the B=256 run — **the VRAM
accounting is broken**, which is why a 13 GB footprint and a full card were invisible in our own logs and
only detectable with external `nvidia-smi`.

**③-correction (2026-10-10, same day)**: that accounting sentence is now **resolved**, and the footprint
numbers in ③ were *under*-counts, so read them as "at least this big":
- `device_bytes` summed a side table (`lens`) that had no entry for `u32`/batch tensors ⇒ silent **0**; it
  now sums `GpuTensorU32::len` per tensor, and `log_vram` prints a second column
  `host 镜像` (Vulkan mirrors every tensor host-side; those bytes are **system RAM**, invisible to
  `nvidia-smi` and must never be subtracted from the GPU budget).
- With the unlogged self-loop scratch counted, **B=256 registered footprint is 17 869 MiB, not ~13.1 GB**,
  and external peak is **20 662 / 22 528 MiB** (desktop baseline 2 128-2 786 MiB — this card is also the
  display adapter). Registered vs external now closes to **+7 MiB (B=256) / +36 MiB (B=128)**, so the old
  "6 GB unaccounted" was **logging timing**, not a leak: with `WARMUP=0` the `log_vram` line fires *before*
  the self-loop scratch is allocated inside the timed region.
- Consequently **"B=256 must OOM" is refuted** — it fits with ~1.8 GB headroom. The B≥192 *sampler*
  slowdown in ③ stays as measured data, and ③'s direction ("capacity, not kernel quality") is
  **strengthened** by the larger true footprint; what changes is the budget arithmetic, which is now
  per-item: RNN state **20.6 MiB/slot** (5.3 GB at B=256) ≫ self-loop scratch ≫ sampler workspaces.
- The sampler mask is now a **bitmap** on Vulkan (`MASK_WORD_STRIDE` spec constant; the host derives the
  stride from the buffer's actual length, CUDA keeps the dense 口径 via
  `supports_compact_sample_mask() = false`): 64 MiB → 2 MiB at B=256, externally measured **−61 MiB** peak
  (theory −62 MiB). It is **throughput-neutral** (B=128 steady 1039.0 vs 1037.6 tok/s) — do **not** cite it
  as a speed win; its real saving is per-`reduce_max` mask traffic (256 KiB → 8 KiB per slot).
  Guard: `tests/batch_parity.rs::top_p_sampled_tokens_stay_inside_the_nucleus` — the mask path previously
  had **zero** CI coverage (every existing sampling test is greedy, and greedy never enters the top-p
  fallback loop). It carries a forward criterion (no sampled token may fall outside the host-computed
  nucleus ⇒ catches cross-slot contamination / stale mask bits) **and** a reverse one (a ≥2-candidate slot
  must occasionally beat a non-argmax ⇒ catches "mask never takes effect", which degrades to greedy and
  would otherwise pass the forward test). Both directions were verified by fault injection.

**③-bis (2026-10-11): the ~5.3 GB that the self-loop stage added is identified and reclaimed -
`imma_partial` was allocated 64x too large.** ③-correction left one number unexplained: `log_vram` grew by
+5471 MiB (B=256) / +2735 (B=128) / +1352 (B=64) between "prefill" and "self-loop", i.e. **21.4 MiB per
slot** - state-sized, while the sampler scratch plus `WorkBuffers` only accounts for ~0.5 MiB/slot.
Attribution method (worth keeping in the toolbox, it is ~10 lines): an env-gated `log` inside
`Runtime::create_tensor{,_f16,_u32}` printing `len`/MiB for every allocation ≥ 4 MiB. At B=64 it caught a
single **f32 tensor of 335 544 320 elements = 1280 MiB**, and
`335 544 320 = 8 (KS ceiling) × 64 (slots per wave) × max(c, ffn_hidden) × batch` - the slot dimension was
counted **twice**, because `WorkBuffers::new`'s `mk(len)` helper already multiplies `len` by `bsz`.
The layout `imma_ksplit` must satisfy is `ks·batch·m ≤ partial.len`, so the correct length is
`8 × max(c, ffn_hidden)` **per slot**.

| batch | registered after self-loop | external peak | decode tok/s (2 alternating rounds) | fingerprint |
|---|---|---|---|---|
| 192, old sizing | 15 135 MiB | 17 885 | 1486.3 / 1465.4 | `0x4f09628/0x8e94` |
| 192, fixed | **11 335** (−3800) | **14 085** | 1491.8 / 1474.1 | `0x4f09628/0x8e94` **identical** |
| 256, old sizing | 17 807 MiB | 20 585 | 1505.1 / 1497.6 | `0x6875c61/0x876b` |
| 256, fixed | **12 747** (−5060) | **15 528** | 1490.0 / 1496.2 | `0x6875c61/0x876b` **identical** |

- **Bit-exact by construction, and verified as such**: under the fixed cap `ks·m ≤ 8·10 240` never binds
  for any op with `m ≤ ffn_hidden` (ks stays 8), and head (`m = vocab 65 536`) already takes ks=1 because
  its block count exceeds the split-K target - so the kernel sequence is unchanged. The A/B above ran the
  two sizings alternately (a temporary `IMMA_PARTIAL_PAD` knob, removed after measurement) and the decode
  fingerprints match in both directions.
- **Throughput is neutral here (± 1 %)** - which is itself the finding: with 20 585 MiB of peak the old
  sizing was *not* yet spilling on this machine, so **③'s 57× sampler cliff does not reproduce at all** in
  this build (1465-1505 tok/s at B=192/256, monotone with batch). ③ was recorded in a session where other
  processes held ~8.6 GB; what the fix changes is exactly that margin - B=256 now leaves ~7 GB free instead
  of ~2 GB, so a co-resident workload no longer pushes the batch path into WDDM spill.
- **This was never Vulkan-only**: `WorkBuffers::new` is backend-agnostic, so CUDA allocated the same dead
  5.1 GB at B=256 (its own split-K partials live elsewhere) - every historical CUDA large-batch number
  includes that allocation.
- Corrects the sentence in §9 ("`imma_partial` must **not** scale with `bsz` … asks for 63 MiB more"): the
  attempted fix kept `× bsz` inside the helper and *added* another `× 64`, so it neither stopped the scaling
  nor cost 63 MiB - it cost 64x the correct size.

**④ The first process after a shader rebuild must not be used as either side of an A/B** - this
round got burned by exactly that and published a fake win.
★ Quantified 2026-10-10 (`sample.comp` + one new spec constant, same params, two fresh processes):
the decode increment is a flat **~4.5 s** regardless of batch (+4692/+4385/+4665 ms at B=128/192/256),
i.e. one-time, not per-round; process 1 read 479.3 tok/s and process 2 read **1039.0** on identical token
fingerprints. `sample.spv` alone costs 4.4-4.7 s to compile (it `[[unroll]]`s a full-vocab loop), and
**every distinct spec-constant tuple pays it separately** (`N/HIST_STRIDE/SAMPLER_ROW/BATCH/MASK_WORD_STRIDE`),
so adding a spec constant is a hidden bench-tax on the next sweep. `vulkan/app.rs` keeps the
`pipeline_cache` in **memory only** (`PipelineCacheCreateInfo` with no initial data, never
persisted), while the driver keeps its own disk cache keyed by SPIR-V hash ⇒ the **first**
process after `build.rs` recompiles shaders compiles pipelines **inside the timed region**. Measured
on `batch_summarize` batch=8 with identical input: the SIMT arm, being that first process, read
0.52 summaries/s (decode 35.3 s); re-run interleaved it reads **2.46 / 2.34** (decode 5.4 / 5.8 s)
- level with the tensor-core arm (2.36 / 2.38). The 6.4x gap was entirely artifact. Outputs are
unaffected and prove only timing was polluted: three independent processes of the same config
produce 20/20 byte-identical summaries. ⇒ Discipline: after touching `assets/shaders/`, run one
throwaway process first; never cite a single-round, single-side reading.

### 6.3 Default flip (2026-10-10): Vulkan r/k/v now run on W8A8 tensor cores too

§6.2② disproved the "3.3× net loss" claim (it was a crash, not slowness). One gate was left
open: this path quantizes the r/k/v **activations** to int8, i.e. it changes the numeric contract.
The gate has now been run and passed, so `VK_IMMA_OPS` defaults to **everything, rkv included**
(reproduce the old default with `VK_IMMA_OPS=key,out,val,head`).

**Quality (the actual reason for the flip)** — same pipeline the deployment uses
(`batch_summarize`, batch=8, 20 summaries over 10 real multi-turn scenarios), CUDA as reference:

| rkv path | mean common-prefix ratio vs CUDA | mean length (CUDA = 88.3 chars) | rejected |
| --- | --- | --- | --- |
| SIMT (old default) | 0.332 | 48.8 | 0 |
| tensor core (new default) | **0.501** | **77.9** | 0 |

The argument is a fact about the code, not a taste: on the CUDA side the r/k/v activations were
already quantized to int8 and fed to W8A8 (`backend_cuda.rs`, `use_imma`, threshold
`IMMA_MIN_BATCH = 1`), so the tensor-core path is the one that matches CUDA, and SIMT
("fp32 activations x int8 weights") was the deviant one. ⚠️ Do **not** read the byte-identical
rate here — both arms score 5/20 against CUDA, because greedy sampling amplifies any single logit
flip into a full sequence divergence (see the implementation record, §2).

**Speed (gain and cost, both stated)** — same-session interleaved A/B, one variable
(`VK_IMMA_OPS`), decode-only, B x path x 3 rounds with the order rotated per round, medians:

| B | rkv = SIMT tok/s | rkv = tensor core tok/s | tensor core / SIMT |
|---:|---:|---:|---:|
| 1 | 74.9 | 65.9 | **0.88x** |
| 8 | 462.7 | 454.5 | 0.98x |
| 32 | 832.8 | 1075.5 | **1.29x** |

The -12% at B=1 is **not** dispatch count: `PROF_GAPS=1` measures 1.3-1.8 microseconds between
adjacent dispatches and 0.5 ms total across a whole recorded batch (2047 dispatches), ~3% of a
15 ms step - and forcing `VK_IMMA_KSPLIT=1` (three fewer dispatches per layer) is **slower**.
The cost is inside the kernel: BN=8 uses one of its eight slot lanes at batch=1, plus the
split-K partial round-trip. ⇒ Decision: **no batch floor**. We accept 12% at B=1 rather than
revert the single-stream path to a numeric contract that differs from CUDA; the real fix is a
tile variant for batch<=4 (filed under §9).

Default configuration vs CUDA after the flip (`bench_ab`, 3-round medians, session
`20261010-213743-78124`; absolute values are not comparable across sessions):

| B | 1 | 8 | 32 | 128 |
| --- | ---: | ---: | ---: | ---: |
| vulkan / cuda (new default) | 0.754x | 0.804x | **0.632x** | **0.413x** |
| vulkan / cuda (old default, §6.1) | 0.87x | 0.83x | ~0.5x | ~0.3x |

I.e. the flip is a net win **inside Vulkan** (tensor core vs SIMT, the table above) from B>=32 and a
small loss at B=1/8. ⚠️ It is **not** a statement about CUDA: the ratio row still sits below 1.0 at
every batch, and 0.632x at B=32 being *lower* than 0.804x at B=8 is CUDA's IMMA path scaling faster
with batch, not this flip damaging Vulkan (the old default was ~0.5x at B=32, so the flip improved
that cell). Gates: `BACKEND=vulkan cargo test
--release` 77 + 8 + 1 passed; `cargo fmt --all` and `cargo clippy --all-targets -- -D warnings`
clean. Full raw readings, plus the two numbers this record originally got wrong, are in
`参考/2026-10-10-Vulkan-rkv张量核默认翻转-实施记录.md`.

### 6.4 `gemm_imma` fetch structure (2026-10-11): +12-16 % at B>=64, bit-exact

**How the bottleneck was found.** `examples/kernel_bench.rs` gained the W8A8 family
(`imma_key / imma_out / imma_value / imma_head`, each `quant_x_i8` + `gemm_imma` with the engine's
own partial-sum capacity so `imma_ksplit` picks the same `ks`) and a **TOPS** column. All four
shapes land in the same 14-22 TOPS band against this card's ~107 TOPS int8 peak. If the limit were
DRAM, the head (most weight bytes) would be worst and `att.output` (fewest) best - it isn't, so the
limit was the **per-warp shared-memory load instruction count**: the A fragment (`m16k32` int8,
16 B/lane ~ 4 LDS) depends only on the row block, yet was re-`coopMatLoad`ed inside the slot-group
loop, giving 2 load instructions per mma.

**Two edits, both meant to be bit-exact** (`assets/shaders/src/gemm_imma.comp`):

1. hoist the A fragment out of the slot-group loop into `a0..a3` (one load per k-tile per warp);
2. split the slot groups across warps - `SW = min(NST,2)` slot-warp columns, `RPG = NST/SW` groups
   per warp, block = `(BM/16)·SW·32` threads. Halves `acc[NST][4]` (32 accumulators at NST=8 was the
   spill threshold) and doubles the warp count, so latency has something to hide behind. `BM=128`
   already has 8 row warps (`tile`'s first dimension is the warp cap) so that arm only gets edit 1.

Consequence to respect: `wid` is now a 2-D index, so every place that used `wid * WR` as a **row**
base must use `row_warp * WR` (flush's `szs`, the epilogue's output row, the OP=9/10 diagnostics).
Missing one is silent corruption, not a crash - it is the same base that caused the 2026-10-04
"two runs, different fingerprints" race.

**A/B (same session, three arms built and swept sequentially, one bench process at a time,
2 rounds each, means)** - `base` = the files at `7dd5a15`, `#29` = base + the shader only:

| B | base tok/s | #29 tok/s | delta | token fingerprints |
|---:|---:|---:|---:|---|
| 1 | 69.2 | 70.2 | +1.5 % | identical |
| 8 | 479.2 | 494.5 | +3.2 % | identical |
| 32 | 1132.2 | 1177.7 | +4.0 % | identical |
| 64 | 1285.3 | 1497.5 | **+16.5 %** | identical |
| 128 | 1409.2 | 1604.6 | **+13.9 %** | identical |
| 256 | 1425.3 | 1597.8 | **+12.1 %** | identical |

Identical fingerprints at every batch means the numeric contract did not move - unlike §6.3's flip,
this needs no quality gate. Where it pays is exactly where §6.2① said the tensor-core path was
weakest (NST=8, i.e. B>=16): the gain tracks `RPG` shrinking from 8 to 4.

**Standing ratios after this change** (same session, clean single-instance interleaved sweep,
means over 2 rounds; the §6.3 row came from a different session and is only a rough reference):

| B | 1 | 8 | 32 | 64 | 128 | 256 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| vulkan tok/s | 68.6 | 493.5 | 1212.7 | 1525.8 | 1620.7 | 1592.5 |
| cuda tok/s | 92.1 | 590.3 | 1808.3 | 2705.4 | 3403.9 | 3825.9 |
| **vulkan / cuda** | 0.745x | 0.836x | 0.671x | 0.564x | **0.476x** | **0.416x** |

Round-to-round repeatability at B>=32 is under 1 %, but **B=1 is not** (68.6 = 71.1/66.1, a 7.5 %
spread), so the B=1 cell cannot resolve anything smaller than a few percent, and 0.745x vs §6.3's
0.754x is drift, not a regression.

**Measurement incident (worth keeping).** Two interleaved loops were found running *at the same
time* - stopping a background job kills the outer shell, its `sh -c` child and the bench process
survived. The contaminated pass reported B=64 at 570.9 tok/s where the same build single-instance
gives ~1520. Every absolute number from it was thrown out, and the sweep script now refuses to
start if a `batch_decode_bench` is already alive. Rule: if an absolute reading is 2-3x off
history, suspect concurrency before suspecting a regression.

Gates: `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`, `BACKEND=vulkan cargo test
--release` (77 + 9 + 1 passed, 0 failed). Details:
`参考/2026-10-11-Vulkan-gemm_imma取数结构-实施记录.md`.

### 6.5 v-arm split-K (2026-10-11): +2-4 % at B<=64, quality gate passed at n=232

`gemm_imma_f16` (the v chain) was the one IMMA arm that never split K: `grid.x = M/BM` = 40 blocks
at M=C=2560, below this card's 68 SMs, while the r/k arms (same shape, `gemm_imma`) had long since
been raised to 40x`ks` blocks. The reason was purely that the reduce kernel only had f32 exits, so
`OP=3` (fp16 store) forced `ks = 1`. `gemm_imma_reduce.comp` gained an `Out16` buffer_reference and
an `OP == 3` arm, so v now writes f32 partials and reduces them into fp16 in ascending `ks` order -
deterministic, but **not** the same bytes as the old path: same quantization, same fp32 accumulation,
different association order. That makes it a numeric-contract change, which is exactly the situation
§6.3 handles with a quality gate rather than a fingerprint check.

**Speed** (same-session three-arm A/B, `#29` vs `#29+#30`, 2 rounds, means): +3.3 / +2.2 / +3.5 /
+2.6 / 0.0 / +0.3 % at B = 1 / 8 / 32 / 64 / 128 / 256 - the expected shape (v is one of three chains
and only 40 blocks, so it is a small-batch occupancy fix; at B>=128 `grid.z` already supplies the
blocks). In the deployment summary pipeline (batch=8) it is 2.81/2.95 -> **3.00/3.03 summaries/s**.

**Quality** (`batch_summarize`, all 105 real multi-turn scenarios = 232 summaries, CUDA as
reference; `VK_IMMA_V_KSPLIT=1` pins the old path so the A/B is one environment variable, same
binary, same shaders):

| arm | common-prefix ratio vs CUDA (cp/min) | cp/max | byte-identical | mean chars (CUDA = 77.3) | collapsed |
| --- | --- | --- | --- | --- | --- |
| v split-K OFF (committed before this) | 0.514 | 0.426 | 69/232 | 80.8 | 18 |
| v split-K ON (new default) | **0.541** | **0.483** | **91/232** | **76.7** | 18 |

Collapsed = CUDA gave >= 80 chars and the arm gave < 35 % of that (the §6.3 failure mode). Paired:
8 collapsed only with OFF, 8 only with ON (McNemar chi2 = 0.06, not significant); every config
reproduced byte-identically across its two runs.

⚠️ **The first pass of this gate used 20 summaries and pointed the wrong way.** It showed the prefix
ratio up but the mean length dropping 77.9 -> 56.9, which reads like the SIMT truncation symptom, and
the reasonable conclusion looked like "revert". At n=232 the length column reverses (80.8 -> 76.7
against a CUDA mean of 77.3): the drop was four short draws in a 20-sample mean, not a distribution
shift. Rule for greedy-sampling outputs: a 20-sample summary gate can only detect gross damage
(§6.3's 0.332 vs 0.501); anything within ~0.1 needs the full scenario set.

Gates: `cargo fmt --all`, `cargo clippy --all-targets -- -D warnings`, `BACKEND=vulkan` and
`BACKEND=cuda cargo test --release` (both 77 + 9 + 1 passed, 0 failed).

### 6.6 Batched-prefill fp16 GEMM tile default (2026-10-11): 0.574x -> **0.96x** of CUDA

⚠️ **Read the ratios in this section through §6.7.** The tile fix itself is solid - `40,421 -> 24,119 ms`
is a within-Vulkan A/B in one window, i.e. **1.68x** - but both sides of every "vs CUDA" number here were
taken through the bench's contaminated prefill column (it includes the `WARMUP` self-loop). On the clean
protocol §6.7 defines, B=128 is **0.84x** and B=256 is **0.79x**, not 0.96x / 0.61x.

The §9 item said batched prefill ran at 0.59x of CUDA and blamed the fp16 coopmat GEMM itself
(~1.8x slower). Fixing that was supposed to need an int8 tensor-core prefill kernel. It needed a
one-line tile-selection change instead - but getting a *trustworthy* number for `gemm` across
backends first exposed two harness bugs.

**① CUDA's `begin_batch`/`end_batch` neither submit nor synchronize** (kernels launch straight onto
the stream; synchronization happens at graph replay / download). So the timed loop measured **host
enqueue**, and `gemm` reported 1,840-112,640 TOPS on a card whose fp16 tensor peak is ~54. Not
cuBLAS being fast - cuBLAS not being waited for. `ComputeBackend::synchronize()` is now a **required
trait method with no default implementation** (a default no-op would let the next backend reproduce
the bug); Vulkan = `queue_wait_idle`, CUDA = `cuStreamSynchronize`. ⚠️ **Every historical CUDA
kernel_bench row is void.**

**② A GPU timing without a read-back is not a measurement.** After ①, the tile sweep still showed
`32x32x32` at 234 TOPS - 4.3x the fp16 peak, indistinguishable from the other arms in the table. The
new `sum(y)` column (one untimed call + partial read-back) returned 0.0000: that arm never wrote an
output. Root cause is structural - `SUBTILE_M = TILE_M/4` and `NUM_MAT_SUBTILE_M = SUBTILE_M/MAT_M`,
MAT_M = 16, so **TILE_M < 64 gives zero loop iterations**: the kernel legally does nothing and
returns success. The shared-memory fallback's `(subgroup_size, subgroup_size, 16)` = `(32,32,16)` was
exactly such a silent no-op arm; floors are now `(64, 32, 16)`.

**What was actually wrong with the tile choice**: the NVIDIA branch gated 128x128 on
`max_shared_memory >= 163840`, while this Turing reports 48 KB - so it was pinned at 64x64x32
forever (that threshold is an AMD-sized-VRAM notion, it does not map to what NVIDIA's API returns).
`128x128x32` needs 36 KB by the same formula the code already uses for its guard (40 KB with the
shader's `ROW = TILE_K+8`), so the card fits it comfortably. Selection is now requirement-based
(`gemm_shmem`), and the guard uses the same single formula instead of two divergent ones.

**Shape divisibility is what makes the bigger tile safe to ship**: `gemm.comp` is Khronos-style coopmat -
fixed compile-time trip counts, no bounds checks - while the dispatch uses `div_ceil`. So if M/N is not
a multiple of TILE_M/TILE_N, the last tile **reads and writes past the end of the buffer**. The low-rank
projections are exactly that shape (n = `round_up(mid, 64)`, can be 64): 128-wide N would write 64
columns past a `[M_PAD, 64]` v_mid. `gemm_tile_for()` therefore shrinks the preferred tile to the
largest one that divides the problem (floors as above); `gemm_spec` and `gemm_grid` both go through it,
and TILE_M/TILE_N depend only on M/N so the two cannot disagree. Prerequisites that keep the main path
at 128: `SeqBuffers.m_pad = t.div_ceil(256)*256` (gpu_model.rs:5601) is always a multiple of 256, and
2560 / 10240 / 65536 are all multiples of 128.

**Kernel level** (4096 token-rows, same session, `sum(y)` bit-identical across every arm):

| shape (`pf_*` = fp16 `gemm`, the only cross-backend-comparable prefill measure) | CUDA cuBLAS | Vulkan new | Vulkan old | new ratio |
| --- | --- | --- | --- | --- |
| `pf_out` 2560x2560 | 41.1 TOPS | 36.9 | 13.8 | **0.90x** |
| `pf_key` 10240x2560 | 48.2 | 39.2 | 13.6 | **0.81x** |
| `pf_value` 2560x10240 | 56.6 | 25.6 | 12.2 | **0.45x** |

No batch-dependent gating was needed: at 512+ rows 128x128 is never worse and usually 1.5-2.9x
better, and at 256 rows the two arms are equal inside the noise (0.19 ms/call is below the timer's
resolution). Small-M under-fill (256/128 x 2560/128 = 40 blocks < 68 SMs) does not show up in practice.

**End to end**, single-sequence (`prof_prefill_steady`, best-of-3): 2048 tokens 2,670 -> **3,788 tok/s**
against CUDA's 4,328 (0.62x -> 0.87x); 512 tokens 2,932 -> 2,973 vs 3,706 (0.80x - short prompts are
RNN-bound, not GEMM-bound, hence the small move there).

**Batched** (`batch_decode_bench` B=128 `PAD_TO=512`, three arms in one session, prefill seed
fingerprint `0x1b9664` identical in all three): CUDA 23,168 ms / Vulkan 64x64 40,421 ms (**0.574x** -
the §9 0.59x, reproduced across sessions) / Vulkan 128x128 24,119 ms (**0.961x**). In the same lines,
decode is still 1,602 vs 3,430 tok/s (0.47x), so prefill is no longer the large-batch story - §6.1's
decode columns are.

**But B=256 is still 0.61x, and that is no longer a kernel problem** (same session, `PAD_TO=512`,
prefill seed fingerprint identical in every arm):

| B | `PREFILL_SLOTS` | Vulkan prefill | registered VRAM | vs CUDA |
| --- | --- | --- | --- | --- |
| 128 | 8 (default) | 24,119 ms | ~9.5 GB | **0.961x** |
| 128 | 16 | 27,910 ms | 11,638 MiB | 0.830x |
| 128 | 32 | 30,998 ms | 15,438 MiB | 0.748x |
| 128 | 64 / 0 (no chunking) | `ERROR_OUT_OF_DEVICE_MEMORY` | - | - |
| 256 | 8 (default) | 75,197 ms | 12,747 MiB | **0.614x** |
| 256 | 16 | 54,409 ms | 14,279 MiB | 0.849x |
| 256 | 32 | 59,670 ms | 18,079 MiB | 0.774x |
| 256 | - CUDA | 46,173 ms | - | 1.00x |

Profiling one chunk of the B=256 case (bt=4096 rows, 32 chunks) rules out the usual suspects:
`[PROF_GPU] SUM 1023ms over 1723 kernels`, `[PROF_GAPS] 间隙合计 0.693ms` (largest single gap 159 us -
**no inter-dispatch stalls at all**), `[PROF_HOST] host=26.1ms [cache=22.7 barrier=1.3 dispatch=1.3]`,
`[PROF_SUBMIT] submit=0.05ms wait=1029ms` (the wait *is* the GPU time; nothing is lost in queuing).
Yet the chunk appeared to cost 75,197/32 = **2,350 ms of wall clock**. That last step was the harness bug
described in §6.7 - the bench's prefill column includes the `WARMUP` self-loop, so the real chunk cost was
**1,326 ms**, of which 1,030 ms is the GPU and **210 ms is the per-slot state hand-back** (194
submits-with-sync per slot). Halving the chunk count does cut that hand-back, but the same move *slows
B=128 down*, because registered VRAM goes ~9.5 -> 11.6 GB; above ~15 GB it gets worse again, and 64 slots /
no chunking OOMs. It was a two-objective problem (per-chunk fixed cost x VRAM headroom) - and §6.7 removes
the fixed cost instead of trading chunk count against VRAM. The linked memory item is §9's fp16 RNN state
(5.3 GB -> 2.6 GB at B=256), which is what would make a larger sub-batch affordable again.

Remaining: `pf_value` (deep-K, narrow-N: n=2560 gives only 20 blocks along N while k=10240 streams 320
K-tiles per block) is the one prefill shape still 2.2x behind, and int8 tensor-core prefill is demoted
behind it - dequant is only 0.9% of the chunk, and fp16 is now at parity. Details:
[参考/2026-10-11-Vulkan-prefill-fp16-GEMM瓦片档修正-实施记录.md](参考/2026-10-11-Vulkan-prefill-fp16-GEMM瓦片档修正-实施记录.md).

### 6.7 Chunked-prefill state hand-back on the device (2026-10-11): B=256 0.67x -> **0.79x** of CUDA

§6.6 ended with "the remaining B=256 loss is ~1.3 s of per-chunk wall clock outside kernel execution".
Half of that was real, and the half that wasn't is the more important finding, so it goes first.

**The harness bug (the third one in this line of work).** `batch_decode_bench`'s `[批量] prefill N ms`
column is `t0 -> t1`, where `t0` is taken **before** `create_batch_state` and `t1` **after** the decode
loop - so it contains tokenization, state allocation and the `WARMUP` self-loop segments. With the
default `WARMUP=1` those segments carry graph capture / JIT: the 75.2 s reported for B=256 covers a chunk
loop that costs 42.4 s. CUDA's capture is the slower of the two, which is exactly why the same bug made
B=128 look like **0.96x** of CUDA and B=256 look like **0.61x**. Clean protocol: read the chunk loop's own
clock (`PROF_CHUNK=1`, new in `forward_seq_batch_padded`) under `PAD_TO=512 NTOK=1 SEGS=1 WARMUP=0
SKIP_SINGLE=1`, one binary for both backends, one session. `prefill seeds` identical in every row.

| B | backend | reset | core (GPU) | xfer | chunk-loop total | vs CUDA |
| --- | --- | --- | --- | --- | --- | --- |
| 128 | CUDA | 1,212 ms | 14,022 | 60 | **15,294 ms** | 1.00x |
| 128 | Vulkan | 1,007 ms | 17,165 | 20 | **18,191 ms** | **0.84x** |
| 256 | CUDA | 2,370 ms | 26,078 | 62 | **28,510 ms** | 1.00x |
| 256 | Vulkan, host round-trip (old) | 1,907 ms | 33,810 | **6,713** | 42,430 ms | 0.67x |
| 256 | Vulkan, device copy (new) | 1,980 ms | 33,999 | **44** | **36,023 ms** | **0.79x** |

**What `xfer` was.** Every chunk hands its sub-batch state back into the big batch state. The old code
did that per slot with `state_slot_back` + `state_slot_load`, and each of those is `3·n_layer + 1 = 97`
`download_part` / `upload_part` calls - and *each of those* records a **temporary command buffer, submits
it and `queue_wait_idle`** (`Runtime::copy_range_standalone`). So **194 submits-with-sync per slot**, 1 552
per chunk at B=256 / `PREFILL_SLOTS=8`: ~1 ms each, which is the measured 200-225 ms.

**The fix** is `State::copy_rows_from` (+ `GpuModel::state_rows_copy`): one device-side `copy_range` per
(layer, tensor), because both sides are contiguous - sub rows `[0, chunk_len)` map onto state rows
`[ci·w, ci·w + chunk_len)` - so a whole chunk is `3·n_layer + 1` copy commands in **one submit**. 200 ms
-> 1.2 ms per chunk, and B=256 prefill 42.43 -> **36.02 s** (0.67x -> **0.79x**). It is a behaviour-preserving
rewrite, verified at three levels: prefill seed sum unchanged (`0x372cc8`), 256-slot greedy decode
fingerprint unchanged (`0x2799db / 0x63ab`), and a new CI case
`tests/batch_parity.rs::cross_state_rows_copy_matches_host_roundtrip`.

★ That test had to add a check §6.6's evidence never provided: `forward_seq_batch_padded`'s **seeds are
the last prompt tokens** - numerically independent of everything the GPU computed. They prove rows land in
the right slots, not that the arithmetic is right. So the case asserts the **state bits** (device copy vs
host read-back, non-zero `src_slot`, `rows=1` not spilling into the next row) *and* the premise of chunking
itself: the same prompt prefilled at width 1 and 2 vs inside a 4-wide batch. **That premise is per-backend,
and the CUDA arm came out worse than assumed** (2026-10-11, per-slot state = 5 409 280 f32):

| backend | w=1 vs w=4 | w=2 vs w=4 (slot 0) | asserted |
| --- | --- | --- | --- |
| Vulkan | 0 differing bits | 0 differing bits | bit-exact equality |
| CUDA | 3 996 677 bits, `max_abs=6.250e-2` | identical to the w=1 row, `max_rel=3.027e-2` | report only |

Vulkan's exactness is the real property (a coopmat GEMM's per-row dot product does not depend on M, and
tiling never reorders one row's accumulation) ⇒ `PREFILL_SLOTS` there is a **pure performance knob**. On
CUDA it is **result-affecting**, and not at rounding level: w=1 and w=2 are bit-identical to each other yet
both differ from w=4 by ~1 % of the state's magnitude, i.e. a batch-shape threshold flips the kernel rather
than the accumulation drifting. The first version of this test asserted `max_rel <= 1e-3` on CUDA and was
red on the first run; report-only on both sides would have thrown away Vulkan's property, so the case is
two arms with one common floor (no non-finite values). Anyone replaying a request across chunk widths on
CUDA should treat `PREFILL_SLOTS` as part of the 口径, like the SIMT knobs. (Also learned the hard way:
never `assert_eq!` two state `Vec`s - one failure printed 122 MB of log. The case reports
`(differing bits, total, max_abs, max_rel)`.)

Still open in this item, in size order:

1. **`reset` is the same pathology in a different suit** - `State::reset` zeroes the sub-state with
   `upload`s, i.e. 97 submits-with-sync per chunk plus 165 MB of host zeros per chunk at w=8: 1.0 s (B=128)
   / 2.0 s (B=256) = 5.5 % of the chunk loop; CUDA pays 1.2 / 2.4 s. The fix is a device-side fill
   (`vkCmdFillBuffer` / `cuMemsetD32Async`) exposed as a backend op, not smaller chunks.
2. **`core` is the whole story again** - 17.2 vs 14.0 s at B=128, 34.0 vs 26.1 s at B=256. That is §6.6's
   kernel-level table (`pf_out` 0.90x, `pf_key` 0.81x, `pf_value` 0.45x) rather than plumbing.
3. §6.6's `PREFILL_SLOTS` x VRAM table needs re-measuring under this protocol (its 24.1 / 27.9 / 31.0 s
   rows all carry the warmup contamination; the ordering of the arms is the part that survives).
4. `create_batch_state` is 3.7 s (Vulkan) / 6.2 s (CUDA) at B=256 - outside the chunk loop but *inside*
   the bench's prefill column, so any quoted prefill number has to say whether it includes it.

Details: [参考/2026-10-11-Vulkan-分块prefill状态回灌设备化-实施记录.md](参考/2026-10-11-Vulkan-分块prefill状态回灌设备化-实施记录.md).

### 6.8 `gemm_imma` shared memory packed 4-bytes-per-word (2026-10-11): decode +2.4…+7.2 %, bit-exact

**Where the gap actually is.** After §6.3-§6.5 the decode ratio was still 0.47x at B=128, so the four
W8A8 tensor-core shapes were timed in isolation against CUDA's own IMMA kernel
(`kernel_bench`; CUDA reaches its imma path through `gemv_int8_*` because `IMMA_MIN_BATCH = 1` and
`GEMV_IMMA` defaults on - note `supports_int8_imma()` is Vulkan-only, so the `imma_*` op family is
skipped on CUDA and the equivalent shapes must be read off the SIMT op names:
`relu2` = key, `mul_add` = out, `add` = value, `plain` = head):

| Shape (m × k) | B | Vulkan (before) | CUDA | Vulkan/CUDA |
|---|---:|---:|---:|---:|
| key 10240×2560 | 128 | 0.2982 ms | 0.1816 ms | 1.64x slower |
| value 2560×10240 | 128 | 0.3229 ms | 0.1714 ms | 1.88x slower |
| head 65536×2560 | 128 | 1.6596 ms | 0.8439 ms | 1.97x slower |
| out 2560×2560 | 128 | 0.0990 ms | 0.0706 ms | 1.40x slower |

**The re-read hypothesis, tested and rejected.** `imma_nst` covered only `NST ∈ {1,2,4,8}` (build.rs),
so a block served `8·NST = 64` slots and **B=128 read every int8 weight twice, B=256 four times**. Adding
an `nst16` variant (128 slots per block ⇒ single read) was supposed to be the easy 2x. It is
**slower** - key 0.2982 → 0.3209, head 1.6596 → 1.8727 (+7.6…+12.8 %) - and so is every other tile
move (`BM=128`, `NST=4/2`, forced split-K): the default `BM=64 / NST=8 / split-K auto` is the top of
that whole space, which closes §6.1's item 3 as a *mechanism* (see the ⚠️ note there).

**What the knob sweep instead exposed** is the real limiter. Cost scales with the bytes each block has
to *stage per k-tile*, i.e. `(BM + 8·NST)·BK` - `NST` 2→8 bought 2.07x while block count went *down*,
which is the signature of instruction-issue pressure on the staging path, not of DRAM traffic. And the
staging path was storing one byte at a time: `store_packed4` unpacked each `uint` (4 quantised bytes)
into **four `STS.u8`**. At `BM=64/NST=8` that is 16 KiB per block per k-tile = ~512 warp-wide store
instructions against only 128 `mma` instructions.

The fix is to keep the shared memory **packed**: `shared uint ws[]/xs[]` (4 int8 per word), one
`STS.u32` per loaded word, and let `coopMatLoad` take the packed form (GLSL allows an 8-bit coopmat to
load from a 32-bit scalar array - `stride`/`offset` are then counted in words, `WS/4 = 36`). Each
fragment needs exactly 4 contiguous bytes per lane, so the read-back becomes one `LDS.u32` too. The
`^0x80` signed shift moves to the whole word (`^ 0x80808080u`).

Measured, isolated (B=128 / B=256, `sum(y)` byte-identical to the pre-change kernel on all four shapes):

| Shape | B=128 before → after | B=256 before → after | after vs CUDA |
|---|---|---|---|
| key | 0.2982 → **0.2492** (1.20x) | 0.5735 → **0.4757** (1.21x) | 1.37x slower |
| value | 0.3229 → **0.2675** (1.21x) | 0.6051 → **0.5117** (1.18x) | 1.56x slower |
| head | 1.6596 → **1.2682** (1.31x) | 3.3022 → **2.5209** (1.31x) | 1.50x slower |
| out | 0.0990 → **0.0876** (1.13x) | 0.1900 → **0.1640** (1.16x) | 1.24x slower |

End to end, same-session interleaved against CUDA (§6.1's second re-measure table): **+2.4 % (B=8) /
+5.2 % (B=32) / +7.2 % (B=128)** decode, ratio 0.875 / 0.716 / **0.508**, CUDA columns unchanged within
their own spread, and **Vulkan decode fingerprints unchanged** (`0xd0c265`/`0x7b05`,
`0x3460382`/`0x16fa`, `0xd1b5310`/`0x7bfc`) - the accumulation order per output element is untouched by
construction, so this is a pure speed change. Gates: `cargo fmt --all`, `cargo clippy --all-targets
--release -- -D warnings` clean, `cargo test --release` green (78 + 11 + 1, 3 ignored).

Left on this kernel, in size order: (i) **done the same day** — the split-K *rule* had been tuned on the
old staging and was mistuned once staging got cheap: it counted only `M/BM` blocks and ignored
`grid.z = ceil(batch/(8·NST))`, so large batches over-split (`value` auto ks 8 = 0.512 ms at B=256 versus
forced ks 2 = 0.462 ms). It now targets **320 whole-grid blocks**; kernel-level −17 % (`key` B=128),
−20 % (`out` B=256), −7…−11 % (`value` B=128/256), end to end **+3.3 % ratio at B=128** and provably
zero change at B≤32 (see §6.1's third re-measure and 参考 §九);
(ii) the remaining 1.2-1.6x per-kernel gap to CUDA is inside the loop (fragment load / epilogue
structure), not in tile shape - §6.4's fetch-structure work is the template; **(ii-a) probed and
falsified the same day**: dropping weight staging altogether and letting `coopMatLoad` read the A
fragment straight from global (glslang *and* the NVIDIA driver accept a `buffer_reference` pointer
here - a capability worth remembering) is **32-42 % slower** (`imma_key` B=8 0.076 →
0.102, B=128 0.204 → 0.291): staging is a necessary cost, not a removable one, because the packed
bulk LDG prefetch buys more than the eliminated `STS`/`LDS` instructions cost. What remains of (ii)
is the `tile[]` store→barrier→read-back round trip in the flush. Methodology note from this probe:
the **first** A/B was invalid - the shader was compiled with `-DIM_DIRECT_A` while the macro is
`IMMA_DIRECT_A`, so the "probe" spv was byte-identical to the baseline (its `md5sum` matched) and the
reported 2 % was pure noise; both arms printing the **same `sum(y)`** is the signature of a dead
probe, so every probe arm now also carries a numeric marker unrelated to performance (here: overwrite
`szs[]` with a constant scale, which moved `sum(y)` by 39.5x and proved the arm ran). (iii) ~~the same packing
argument applies to the fp16 prefill `gemm`~~ — **checked and voided**: `gemm.comp` already stages
`shared uvec4` (16 B per store), so there is nothing to pack there; §6.6's prefill gap is a different
mechanism (see 参考 §七 item 3).

Details: [参考/2026-10-11-Vulkan-gemm_imma-smem打包staging-实施记录.md](参考/2026-10-11-Vulkan-gemm_imma-smem打包staging-实施记录.md).

### 6.9 Low-rank chain, second level on coopmat fp16 tensor cores (2026-10-11): B=128 decode **+16.9 %**

The last batched chain without a Vulkan tensor core (§9 item 2) was the low-rank one: 20.9 % of B=128
decode in `gemv_lowrank_chain4_batch`, measured at **6.6 % of fp32 SIMT peak** and 40× above its
bandwidth floor — latency-bound by the per-row subgroup reductions over K=64/96, which is a
*structure* problem, so the lever was a tensor-core arm, not a tuning sweep.

**Only the second level moved**, and that is a measurement, not a hedge. `kernel_bench`'s new `lr*`
arms ran both levels at both precisions on this model's real shapes (same session, interleaved):

| level | shape (B=128) | SIMT | coopmat fp16 | verdict |
|---|---|---:|---:|---|
| 1 (mid projections, `w/a/v/g`) | M=128 × N=96/96/64/320 × K=C | 0.274 ms/layer | 0.443 ms/layer | **keep SIMT** (TC 1.7× slower) |
| 2 (out projections, `w/a/v/g`) | M=128 × N=C × K=128/128/64/320 | 0.398 ms/layer | **0.0824 ms/layer** | TC (4.8× faster) |

Level 1's N is 64..320, so coopmat's 16×16 tile leaves whole column bands idle at `TILE_M`'s floor of
64 rows; level 2 is a normal [B,C]×[mid,C] GEMM and tiles cleanly.

⚠️ **Superseded same day (S31 ①-b/②, record §10–§12)**: level 1 *did* move to coopmat fp16 — the
paragraph above is the level-2-only 口径. Divisibility-only tile selection was the real limiter
(block starvation), and split-K was then built and falsified as the follow-up throughput lever
(default OFF, `GEMM_SK` unset).

**Implementation**: level 1 stays as-is (SIMT, writes the mids with *unpadded* stride), then 4× `to_f16`
cast + `elementwise_sigmoid_inplace` on `g_mid`, 4× `gemm`/`gemm_bias` (the existing prefill coopmat
kernel — same formulation §6.6 tuned, so the fp16-accumulation semantics are already validated) into
zero-initialised f32 accumulators, and a new fused **`lowrank_epilogue`** shader that applies each
chain's tail semantics in one pass (`CHAIN` spec 0=w scale, 1=a, 2=v read-modify-write into the
attention v, 3=g gate). 14 dispatches/layer replace 1, and `PROF_GAPS` says the gaps are not the cost.
Gate: `lowrank_stage2 = batch >= 32 && c % 64 == 0 && all mid_pad % 64 == 0 &&
supports_lowrank_gemm_stage2()` — **default ON for Vulkan**, `LOWRANK_GEMM2=0` to force off; CUDA and
fp16-weight models report `false` and keep their own paths. Two invariants bite: `gemm.comp` has no
bounds checks (⇒ `rows = batch.div_ceil(64)*64`, `to_f16` zero-fills rows ≥ batch) and padded mid
*columns* must be zero-initialised at allocation, because `0 × uninitialised = NaN`.

**Correctness** — greedy divergence is useless here (it saturates: 17.9 % of 2 048 steps vs the 21.6/23.3 %
already accepted for the fp16-state arm, and a *batch-only* change is invisible to the single-stream
`ARGMAX_VERIFY` gate). So the decision was made on a **calibrated element-wise yardstick**: dump the
full B=128 logits (`DUMP_LG`) and take rmse against the SIMT arm —

| perturbation | logit rmse |
|---|---:|
| same arm re-run (noise floor) | 0.000 % |
| **this arm vs SIMT** | **6.850 %** |
| fp16-state arm (already shipped, same yardstick) | 6.646 % |

i.e. the tensor-core change costs what the already-accepted fp16 rounding costs, and 0 % of a
computation error. Supporting: argmax agrees 127/128 with every top-3 *set* intact (a coherent
distribution shift, not corruption), and each arm is bit-deterministic across rounds.

**Performance** (same session, interleaved, `SKIP_GRAPH=1`): 66.8 → **57.2 ms/step**, +16.9 % tok/s, the
two arms' three-round distributions do not overlap (worst TC round still 13 % above the best SIMT
round). Per-kernel attribution: low-rank level 2 **0.3977 → 0.1069 ms/layer** = −9.3 ms/step, which
matches the −9.6 ms/step wall change; every other kernel's per-avg stays inside cross-process noise.
⚠️ `PROF_GPU`'s `SUM`/`count` are **not** comparable across the arms — the ON arm dispatches 13 more
kernels/layer and the profile ring truncates at 2 048 entries; only per-avg is citable.

Details: [参考/2026-10-11-Vulkan-低秩二级张量核-实施记录.md](参考/2026-10-11-Vulkan-低秩二级张量核-实施记录.md).
Open on this lever: ~~the `batch >= 16` threshold is unvalidated for B=16…63~~ — **swept the same day,
and it was wrong**: `rows` floors at 64, so below batch 64 the tensor-core half does not shrink while
SIMT `chain4` does ⇒ B=16 is **6.9 % slower** (interleaved, non-overlapping), B=32 +2.8 %, B=64 +10.7 %,
B=128 +14.4 % (per-step ms). The default gate is therefore `batch >= 32`. Remaining: 17…31 sits on the
safe side without its own measurement, and a `TILE_M < 64` variant in `gemm.comp` is what would actually
extend the win downward. Also possible: merge the 8 cast/epilogue dispatches into 2 quads for ≤0.04
ms/layer. **Cross-backend standing after this lever** (§6.1's 4th re-measure): **0.874× (B=8) /
0.740× (B=32) / 0.627× (B=128)** — B=128 was 0.525× before this lever and the fp16-state arm.

### 6.10 State kernel: access width + two-row interleave (2026-10-11): shipped, **and the lever is closed**

`fuse_ka_dplr_norm{,_s16}` was the second biggest single kernel (15-17.7 % of B=128 decode) and the
last SIMT one nobody had touched this round. Two independent edits to
`assets/shaders/src/fuse_ka_dplr_norm.comp` - which one spv serves **both** the single-token and the
batch path (`runtime.rs:3320` / `:3398`, only `grid.z` differs), so one edit covers both:

1. **two-row interleaving** in the subgroup-per-row loop (mirrors the CUDA kernel's `#pragma
   unroll 2`): each row is a pure latency chain (load S → `subgroupAdd` sa → per-column update →
   store → `subgroupAdd` y), and written one-row-at-a-time adjacent chains cannot overlap. Two rows
   in flight = two independent chains. **Per-row arithmetic and reduction structure unchanged
   verbatim ⇒ bit-identical by construction.**
2. **column pairs**: `BufS` reinterpreted as `f16vec2` / `vec2`, so lane `sgid` owns columns
   `2p, 2p+1` instead of `p, p+32` - half the memory instructions, 128 B per warp instruction
   instead of 64. Alignment rests on `N` even (64) and `s_base + row·N` even (`N²=4096`). This one
   *does* change the `subgroupAdd` summation partition, so it needed the element-wise gate below.

**The measurement is the finding** — and then it turned out to be a measurement *of the profiler*.
`PROF_GPU` per-avg for the *same* spv spread ±12 % across process batches
(0.2219/0.2131/0.1982/0.1854/0.1888/0.1935/0.1901 ms), so the first two A/B passes (−4.0 %,
−4.7 %) had to be downgraded to "direction only". The citable design was meant to be a
**same-session, four-arm, order-reversed** sweep (spv hot-swapped between `C:\tmp\spv-base2|spv-pairs`,
`{base,pairs} × {fp16,fp32}`, 2 rounds each order, every arm logging the md5 of the spv it actually
bound):

| arm (mean of 4 rounds) | per-avg | vs base | valid? |
|---|---:|---:|---|
| base fp16 | 0.18945 ms | — | |
| **pairs fp16** | 0.18760 ms | ~~**−0.98 %**~~ | ❌ **void** |
| base fp32 | 0.33650 ms | — | |
| pairs fp32 | 0.32915 ms | ~~−2.18 %~~ | ❌ **void** |

⚠️ **Correction (2026-10-11, after §6.8's instrument post-mortem)**: `base` and `pairs` are two
byte-strings of the **same spv file name**, and a release build **embeds** `assets/shaders/spv/*.spv`
into the executable (`vulkan/asset.rs`, `#[derive(RustEmbed)]`) - so hot-swapping the file on disk
never changed what the running process executed. Those two rows compare one arm with itself; the
−0.98 % / −2.18 % is profiler drift, and **how much the column-pair edit actually buys has never been
measured.** Re-measuring it is not worth a round: the pre-edit bytes no longer exist anywhere
(`spv-all`, `spv-new`, `spv-cur` and the repo all carry the *post*-edit md5s, `171b91a36e` /
`ec23c548f3`), so it would first mean rebuilding the `.comp` from `07ba7dc^`. Use `VK_SPV_DIR` (§6.1
of the fetch-structure record) for any future shader A/B - it is the only way to swap shader bytes
inside one executable.

What **is** valid from that same table is the *cross-width* comparison, because fp16 and fp32 pick
**different file names** (`fuse_ka_dplr_norm_s16` vs `fuse_ka_dplr_norm`, `runtime.rs:59`) and both are
embedded: fp32 moves exactly **2× the bytes and takes 1.77× the time** ⇒ the kernel is **DRAM-bound,
not instruction-issue bound**, which falsifies the §1 hypothesis this lever started from. Position on
the roof: state traffic per dispatch = `B·H·N·N·2B·(r+w) = 83.9 MB` at B=128 (the profiler's `est`
column reads 0.1 MiB because it does not count `buffer_reference` traffic - ignore it) ⇒ 447 GB/s =
**72.5 % of this card's 616 GB/s**. A pure read+write stream on Turing tops out around 80-85 %, so
what is left here is ≤15 % of the kernel = **≤2 % of the decode window**, not the 2× anyone hoped for.
**That ceiling - not the void Δ - is why the lever is closed.**

**Correctness (the admission gate)**: `STEP=1 PROBE=1 DUMP_LG` logits, 2 097 152 elements —

| comparison | logit rmse | bit-identical |
|---|---:|---:|
| pairs vs base | **0.000e+0** | **100 %** (dump md5 identical, `F47EBB40…`) |
| calibration control: base vs `LOWRANK_GEMM2=0` (already-shipped perturbation) | 1.103e+0 = 6.558 % | 0 % |

The control proves the probe *can* see a rounding-level change of the kind being made here, so 0.000 %
is a measurement, not a broken probe - the reassociated fp32 sums differ by ~1 ulp and the state is
stored back as fp16, which absorbs it. Multi-step decode fingerprints are bit-identical across all four
arms (`0x32c026/0x18ea` at fp16, `0x32a50a/0x18ea` at fp32). Gates: `cargo fmt --all`, `clippy -D
warnings`, `BACKEND=vulkan cargo test --release` (10 passed / 0 failed in the batch suite).

Both edits are kept (bit-identical, and they double the bytes each warp instruction covers - which
§6.8's layout lever went on to prove is the right direction on this card), but **no end-to-end gain is
claimed for them, because none was ever measured**: the ceiling argument above (≤2 % of the decode
window even at the kernel's own 15 % headroom) is the reason to stop, not a number. §6.1's standing
ratios are therefore unchanged by this lever. Details (including the void-reading banner):
[参考/2026-10-11-Vulkan-状态核访存宽度与行交错-实施记录.md](参考/2026-10-11-Vulkan-状态核访存宽度与行交错-实施记录.md).
**Next lever by these measurements**: `gemm_imma` (≈48 % of the same dump, still in the 14-22 TOPS band
against ~107 TOPS int8 peak) - which is exactly what §6.8/§6.9 then did, and what actually moved
end-to-end decode by **+6.8…9.3 %**.

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
| Relative gap (Vulkan vs our own CUDA) | n/a | **Two 口径, and they disagree**: ① *SIMT vs SIMT* (CUDA's three tensor-core gates off) — parity, 0.98×–1.03× across decode B=1…256 and steady prefill; ② **defaults** (CUDA tensor cores on, Vulkan on for everything but rkv) — **behind at every batch**, 0.87× at B=1 down to **0.27× at B=256** (same-session interleaved medians, §6.1). ⚠️ **Updated 2026-10-11**: that row's "but rkv" is stale (§6.3 put r/k/v on W8A8 tensor cores by default), and after §6.3-§6.5 + §6.8 the same-session medians are **0.875× (B=8) / 0.716× (B=32) / 0.508× (B=128)**; **re-measured a 4th time the same day, after §6.9 + the fp16-state arm: 0.874× / 0.740× / 0.627×** (CUDA columns byte-identical, B=8 is the control because §6.9's gate is `batch >= 32`). B=256 has not been re-measured since. ★ **Re-measured a 5th time the same day, after the group-major int8 weight layout — for the first time across all six batches: 0.900× (B=1) / 0.972× (B=8) / 0.824× (B=32) / 0.765× (B=64) / 0.690× (B=128) / 0.688× (B=256)**, CUDA fingerprints again byte-identical — ⚠️ **but not its throughput**: that session's CUDA absolutes sat 6.4…7.7 % below its own 4th-session values at all three shared cells while Vulkan's sat above, so most of the ratio move versus the 4th table is a reference-side session effect. The layout's own credit is its same-session two-arm A/B, **+6.8…9.3 %** (B=1/8/32, shrinking with batch — at large batch the weights are already amortized across slots). **The 0.27× at B=256 in this row is retired** — it measured a stale build; B=256 is now monotone with the rest of the curve (its 3-round spread is 25.8 %, one outlier round; excluding it the cell is 0.688–0.697). Shape of what is left: ≈2.8 pt at B=8, ≈17.6 pt at B=32, ≈31 pt at B=128/256. **★ 2026-10-12: 口径 changed and re-measured (session `20261012-004233-91772`, both backends interleaved in one session, six widths × 3 rounds, median; `power.limit` 300 W = the shipping 档 now)** — every row above belongs to the old 档 (250 W; the seventh also had `-lgc` pinned at 1860) and **cannot be compared across 档**. The new-档 baseline is **0.941 (B=1) / 0.945 (B=8) / 0.806 (B=32) / 0.730 (B=64) / 0.690 (B=128) / 0.762 (B=256)**, absolute ms/step Vulkan 12.40 / 14.71 / 22.86 / 34.28 / 58.53 / 100.48 vs CUDA 11.67 / 13.89 / 18.43 / 25.01 / 40.37 / 76.54 ⇒ the gap is 0.7–1.0 ms/step at low concurrency, 4.4 ms at B=32, 9.3 at B=64, 18.2 at B=128, 24.0 at B=256. Raising the power wall is "win at mid, lose at large": B=32 +160.5 tok/s but B=128 −141.9 and B=256 −225.0 (82 ℃, SM clocks drifting 1680–1845 ⇒ the big widths now hit **heat**, not the watt cap). One narrative correction: **the gap is decode-only** — on batched prefill Vulkan *leads* CUDA at B=1/8/32 (2.47× / 1.44× / 1.085×) and trails by 6.1 / 12.4 / 16.5 % only at B=64/128/256 (§20). |

The original gap came from ① Albatross's more aggressive kernel fusion; ② CUDA Graph capture cutting launch overhead; ③ CUDA's mature hardware-specific libraries. It has since been closed and reversed on the CUDA path by **int8 tensor cores on int8-resident weights** (half the bytes per token, 2× the tensor-core peak), split-K, merged multi-chain launches and shape-aware tiling. The **Vulkan** path was brought to the same level *at the same SIMT 口径*: halving activation traffic in the r/k/v fused kernel (fp16 activations), `vec4`+`dot` weight handling and shared-memory tiling of `x` closed the last ~40%, so Vulkan is throughput-neutral against CUDA **when CUDA is held to the same kernel sequence** while keeping cross-vendor reach. Against CUDA's *default* tensor-core path Vulkan is not neutral — the r/k/v batched kernel is the one op Vulkan keeps on SIMT by default (`VK_IMMA_OPS`), and that is where most of the B-scaling gap lives (see §6.1, 口径 paragraph). ⚠️ That last sentence is the 2026-10-10 pre-flip state: §6.3 (same day) moved r/k/v onto the W8A8 tensor-core path by default, and §6.4/§6.5/§6.8 then re-measured the remaining gap as sitting inside `gemm_imma` and the low-rank chain, not in r/k/v being SIMT.

Inspired by both, this repository has since evolved independently, adding capabilities neither has: **two weight-quantization paths** (fp16 / int8 auto-routed), **CPU fp32 reference plus GPU-kernel unit tests**, and an **offline quantization toolchain** with a reproducible accuracy-verification workflow.

## 9. Known Limitations & Future Work

- **prefill dequant overhead ~10%** (int8 vs fp16 at T=512): full removal needs cooperative-matrix fused dequant+GEMM (Marlin-style zero-copy), a long-term direction.
- **prefill time grows ~quadratically with T** (T=256: 0.33ms → T=512: 0.81ms, WKV parallel intra-chunk term); consider WKV chunking for long prompts.
- **Vulkan dplr_seq occupancy is low** (40 workgroups × 64 threads + 2 barriers per token, ~10ms in steady prefill): could adopt the CUDA structure (parallel-row blocks + shuffle reduction).
- **The batched state kernel `fuse_ka_dplr_norm` is closed as a lever — it is DRAM-bound** (2026-10-11, §6.10): two-row
  interleave + adjacent-column-pair access are both shipped, **but no end-to-end gain is claimed for either, because
  none was ever measured** — the four-arm sweep that produced "−0.98 % at fp16" compared two `VK_SPV_DIR` arms that
  turned out to hold the *same embedded bytes* (release builds bake the spv into the binary; see §4's `VK_SPV_DIR`
  row), so those percentages are void. ⚠️ And "bit-identical" was wrong too: the pairs edit changes the `subgroupAdd`
  summation partition, and Vulkan's decode fingerprints did move with it (§6.1's 5th table) — the gate that covers it
  is `cargo test --release`, which passes. What survives from that session is the *qualitative* judgment, which came
  from a valid comparison (fp32 vs fp16 state are different **file names**, selected per dispatch): the kernel sits at
  83.9 MB / 0.1876 ms = **447 GB/s ≈ 72.5 % of this card's peak** (absolute reading, ±10 % cross-process), and
  **fp32 state (2× the bytes) takes 1.77× the time** — time tracks bytes, not memory-instruction count. Remaining
  headroom to a realistic copy-kernel ceiling (80-85 %) is ≤15 % of a kernel that is ~12.5 % of the decode window ⇒
  **≤2 % end-to-end, not worth another round**. Byte count is irreducible at fp16 (the state must be read and written
  in full each token); going below fp16 would break the numerics contract.
- **`gemm_imma` was the next lever by those measurements, and it has since been taken** (2× in one day): the packed
  smem staging + split-K retune (§6.8) and the **group-major int8 weight layout** `[K/128][M][32]` (§6.6-§6.9 of the
  `gemm_imma` record, commit `0f4dbeb`). ⚠️ Its credit is the **+6.8…9.3 %** same-session two-arm A/B, *not* the
  0.740→0.824 / 0.627→0.690 ratio steps between the 4th and 5th tables — those two cells moved mostly because
  CUDA's own absolutes dropped that session (§6.1, "read the absolutes before the ratios"). What the 5th table leaves
  is ≈31 pt at B=128/256.
- **S27 tried to attribute those points across backends and found the instruments cannot meet in the middle**
  (2026-10-11, one session, `batch_decode_bench`, B=64/128, data `C:/tmp/prof128-be-*.log`):
  - **`PROF_CUDA_KERNEL` absolutes are unusable in exactly the configuration it requires.** With
    `NO_SELFLOOP_GRAPH=1` the decode becomes host-launch-bound, and a non-sync event pair brackets *host submission*
    whenever the queue is empty ⇒ CUDA's dump sums to **6 249 ms** for a run whose wall clock decode window is
    ≈150 ms (≈41×), and its `avg` ranks are dominated by launch count, not GPU work (`to_f16` 3 568 launches,
    `dequant_int8` 3 072, `elementwise_sigmoid` 2 032). Two further tells: the same `fuse_ka` shows `avg` 1.17 ms
    at B=128 and 1.20 ms at B=64 (a kernel whose traffic doubles with batch cannot cost the same), and the dump
    prints a **partial flush per event-pool fill** (`SUM 368 → 6 249 ms` in 30+ growing lines) rather than per step.
    ⇒ cross-backend per-kernel attribution needs `PROF_GAPS`/ablation-style arms, not these two instruments. Registered
    as a dead instrument path.
  - **Vulkan's own dump is self-consistent and does rank what is left** at B=128 (NTOK=1, SKIP_GRAPH=1,
    `SUM 46.6 ms` GPU over 1 349 kernels against 55.7 ms/step unprofiled ⇒ ~84 % accounted, the rest is inter-dispatch):
    `gemm_imma_bm64_nst8` **22.70 ms (48.6 %)**, `gemv_int8_rkv_stage1_batch` **8.76 ms (18.8 %)**,
    `fuse_ka_dplr_norm_s16` **6.42 ms (13.8 %)**, `gemm_imma_reduce` 2.04, `sample` 1.91, `gemm_f16_f32` 1.64,
    `quant_x_i8` 1.44, `norm_lerp6_batch` 1.21, everything else <0.5 each. The state kernel is at
    **418 GB/s = 68 % of the 616 peak** (83.9 MB/layer at 0.2008 ms) — i.e. already on the roof §6.10 concluded.
    **The new #2 is not `gemm_imma` and not the state kernel: it is the mid-projection arm** (with r/k/v on tensor
    cores this kernel reduces to `v1/w1/a1/g1`, fp32 weights, `dispatch = (VM+WM+AM+GM, ceil(batch/BGRP))`, one row
    per block ⇒ the 5.9 MB/层 mid weight set is re-read `batch/BGRP` = **16×** at the default BGRP=8). §6.2①'s
    "BGRP>8 spills" verdict was measured on the *int8 r/k/v* accumulator (`float acc[ROWS][BGRP]`), which this arm no
    longer runs, so **BGRP 16/32 has never been tried on the mid arm and is numerically bit-identical to try**.
  - **Both levers that ranking implied are now measured dead on this arm (S28 + S29, 2026-10-11, same session,
    interleaved with the order reversed, `C:/tmp/bgrp-b128-*.log` / `C:/tmp/mid4-ab-b128-*.log`)**:
    `GEMV_BGRP` 16/32 = **−8.6 % / −7.1 %** e2e (medians 2107.7 / 2142.1 vs BGRP=8's 2305.0, all three arms' decode
    checksums bit-identical since BGRP only changes which slots share a workgroup); and rewriting the four mid loops
    from scalar to **`vec4` + `dot`** (Params members swapped `BufF`→`InX4` — no host change at all, since a slot is
    just a device address) = **−6…−9 %** e2e (paired same-round NEW/OLD 0.906× / 0.940×, ranges non-overlapping;
    `spirv-dis` confirms it did what it claims — `OpDot` 1→5, per-lane trip count 20→5, loads ÷4). Reverted, spv md5
    back to `9addb312…`.
  - **Why, read off CUDA's own copy of the same kernel (`backend_cuda.rs:2731-2812`) — it had already measured both**
    and named the mechanism: its mid arm is *also* a plain scalar loop with `BGRP=8`, and its slot-span knob
    `MID_GG` (= `BGRP × GG` slots per block, the exact analogue of `GEMV_BGRP`) measures **GG=1 1.03 ms vs
    GG=2/4/8 1.39/1.98/4.66 ms** — "more accumulators spill the registers right back". Two backends, two compilers,
    the same cliff ⇒ §6.2①'s spill verdict generalises to this arm, and the S28 confound (the same env also feeding
    `gemv_lowrank_chain4_batch`) is no longer needed to explain it. CUDA's actual 2.5× on this kernel came from
    **removing an array of per-slot base pointers** (`const float* xsrc[BGRP]` → local-memory spill, 2.55→1.03 ms),
    which the Vulkan arm never had — it addresses inline. ⇒ the biggest measured win available on this arm is already
    in place on both sides, and "**the mid arm is instruction-issue-bound**" (the premise S29 was built on) is
    falsified: instructions ÷4 lost. What remains unproven is the premise above it — that the arm is where the
    *cross-backend* gap sits at all, since CUDA runs the same shape. **Treat "mid arm = the #2 gap" as unverified.**
  - **So the gap can't be attributed with either old instrument, and this session built the third one**: CUDA's
    `ABLATE=csv` (skip launches by name, its own comment says it beats `PROF_CUDA_KERNEL`) had no Vulkan counterpart
    — only an intra-kernel `RKV_ABLATE` bitmask. Vulkan dispatches through exactly one site (`src/runtime.rs`,
    `record_kernel`), so the same-named ruler is now there: an ablated kernel keeps its pipeline creation, uniform
    slot, barriers and bind, and drops only `cmd_dispatch` (+ its two timestamps, so the `PROF_GPU` ring stays
    honest). Names are normalised (`shaders/spv/x.spv` → `x`) and the first hit per name logs, because the trap
    CUDA documents — a typo silently matching nothing, read as "zero benefit" — fired on the first calibration run
    and the hit log is what caught it.
- **S30 built the missing ruler and the attribution flipped** (2026-10-11, B=128, same 尺门控, data
  `C:/tmp/abl_cal.out` / `C:/tmp/abl_cuda2.out`, record §8-§9): Vulkan's ablation arm agrees with its own
  `PROF_GPU` dump to **+2 %** (mid Δ 8.83/9.03 ms vs 8.76 dumped) and **−2 %** (state 5.45/7.16 vs 6.42), and the
  totals cross-check the 5th §6.1 table independently — 54.4 vs 37.4 ms/step = **0.688×**, the same cell the paired
  re-measure got with a completely different instrument. What it then measured:
  - **CUDA does not launch the SIMT mid arm at B=128 at all** (`ABLATE=gemv_int8_rkv_stage1_batch_g1` ⇒ hits=0,
    checksum bit-identical to base). Its whole low-rank chain, **level 1 included**, runs on tensor cores — but
    ⚠️ *not* as one fused kernel: reading `backend_cuda.rs:3188/3274/8872` it is `cast4_f16` →
    `lowrank_stage1_gemm` (**the four chains merged into a single launch**, chain picked by `blockIdx.z`,
    BM=64×BN=16×BK=128 with tanh/sigmoid inlined) → 4× `lowrank_stage2_gemm` (64×64×64) with `lr_chain_epi`
    inlined into the accumulator's exit, and the two levels meet through a **global** `mid16`, not SMEM. Ablation
    Δ **4.26/5.93 ms/step** for the whole chain. Vulkan pays **8.9 ms for level 1 alone** plus ≈6.5 ms for
    level 2 + four epilogues ⇒ **≈15.4 ms vs ≈5.1 ms ≈ 10 ms = ~60 % of the 17 ms/step gap sits in this one
    chain.**
  - ⇒ the item-2 decision above ("level 1 stays SIMT — measured, not assumed: TC 0.443 vs SIMT 0.274 ms/layer") was a
    **wrong-frame comparison**: it benchmarked "level 1 moved to TC" against "level 1 in SIMT", while CUDA's peer is
    the *whole chain* on tensor cores with the four chains **merged into one launch** and the epilogues **inlined**
    into level 2. And the 0.443 itself turned out to be a **block-starvation artifact**: divisibility-only tile
    selection leaves the level-1 shape (M=128 × N=64..320 × K=2560) with **1..5 blocks for 68 SMs**. Fixing the tile
    rule host-side (occupancy + a structural floor, see S31 below) puts the same shapes at 4..20 blocks and level 1
    on TC at **0.029 ms/call ≈ 0.118 ms/layer** — 2.3× *faster* than SIMT's 0.274 ⇒ "level 1 cannot go on tensor
    cores" is retired, and the lever is the chain (task S31), not that decision.
  - **State kernel is not a gap source**: its ablation costs CUDA *more* than Vulkan (7.40/8.97 vs 5.45/7.16 ms) —
    consistent with §6.10 (Vulkan's copy already sits at 68 % of DRAM peak).
  - **Ablation validity rule learned**: the reading is a "removal benefit", so it is only usable when the absent
    kernel's output does not change *other* kernels' work. Ablating `gemm_imma_bm64_nst8` came back **−47 %**
    (garbage into `relu²` ⇒ the sparse FFN stops being sparse ⇒ downstream gets slower than the removed 48.6 %), so
    tensor-core GEMM shares must be read from `PROF_GPU`, never from ablation. Also: base arms have to sit *adjacent*
    to the ablated arm in every round — `abl_vk2.sh` put base only at the head and tail of its rounds, they drifted
    61.7 vs 56.1 ms, and its one new arm (epilogues, 7.51/0.17 ms) became unusable.
- **Measurement trap found while closing it**: `PROF_GPU` per-avg for the *same* spv spread **±12 %** across process
  batches (0.2219/0.2131/0.1982/0.1854/0.1888/0.1935/0.1901 ms). Any kernel-level A/B that compares two *different*
  processes is therefore direction-only. Citable design = same session, arms interleaved, **order reversed** between
  rounds, each arm logging the md5 of the spv it actually bound.
- **First prefill of a new length bucket pays pipeline creation** (kernel cache now decouples uniforms via a shared pool + DYNAMIC offsets keyed on (shader, spec): 1722 → ~30 entries; the remainder is the inherent first-appearance cost of each spec): eliminable via startup warmup. ★ Measured 2026-10-10 (`BUCKET_BENCH=1 SLOTS=8`, 3B int8, fresh bucket = `PAD_TO+128`) - and it is **more** than pipeline creation: switching length rebuilds all 45 `[bt,·]` seq buffers, which costs *seconds*, not ms - cold-bucket first request 1809.0 ms vs same-bucket repeat 942.7 ms on CUDA (864.1 ms saved), 3641.0 vs 2493.2 ms on Vulkan (1152.2 ms saved). `warmup_len_buckets` moves that wait from the user to startup; **total cost is unchanged** (warmup itself costs 1845.3 / 3623.4 ms ≈ the cold first request), and because `SeqBuffers` caches **one** shape only, just the last bucket in the warmup list stays resident - requests that alternate lengths need a fixed `pad_to` or one model per bucket.
- ~~**Vulkan uniform pool must be sized by hand for long batches**~~ — **stale, verified resolved** (2026-10-10). The claim predates `pool_slots` dedup (ba08642, 2026-08-25), which made pool use O(distinct params per batch) rather than O(dispatches). Measured high-water with `PROF_POOL=1`: 69,080 B (batched prefill) / 33,432 B (self-loop) / 31,344 B (`batch_step_sample`) — **flat across B=8/16/32/64 and NTOK=32/64/256**, including `SKIP_GRAPH=1` (all n rounds in one submit), `STEP=1 PENALTY=1` and long prompts. Default 8 MB therefore runs every batch path with ~1.2 % of the pool; `UNIFORM_POOL_MB` is only an extreme-shape fallback.
- **fp16 models have no batched *decode* path (batched prefill does work)**: the batched decode r/k/v kernels exist only in int8 variants (`gemv_int8_rkv_stage1_batch`), so an fp16 model falls back to per-slot single-sequence decode; `forward_seq_batch_padded` (batched prefill) uses the seq-parallel kernels and reads fp16/fp32 weights fine - verified on both backends 2026-10-10 by comparing the **per-slot state after batched prefill against the single-stream state, bitwise identical** (that is the assertion that actually pins "pad rows never enter the state"; the seeds the call returns are each slot's *last prompt token*, i.e. the decode seed, **not** a next-token prediction - comparing them to single-stream argmax is a category error, and an earlier version of this test did exactly that). `GpuModel::batch_decode_blocker()` returns the reason as a string so callers (client session pool, mytown engine) can log *why* they went serial instead of reading a silent `supports_batch_decode() == false` - and instead of over-reacting to "no batched forward path" and serializing prefill too.
- **The low-rank chain is the only batched chain left with a SIMT half on Vulkan** (level 2 is on tensor cores since 2026-10-11 — §6.9; `lowrank_gemm_batch` was CUDA-only. ⚠️ This entry used to charge **FFN value** to "SIMT only" as well, which was **wrong**: at batch >= 16 CUDA runs the dense tiled `ffn_value_gemm_batch` / `ffn_value_imma_batch`, while Vulkan runs the generic `imma_op_on("val")` path `gemm_imma(M=C, K=fh, OP=accumulate)` - both are tensor cores, just different shapes): every other batched tensor-core op (head/out/key/val **and rkv**) is now on by default - the `VK_IMMA_OPS` default flipped from "everything but rkv" to "everything" on 2026-10-10, see §6.3.
  ~~The Vulkan tensor-core r/k/v variant measured a 3.3x net loss~~ - **that was a misreading, corrected 2026-10-10**: the path hit `ERROR_DEVICE_LOST` at every batch (`gemv_int8_rkv_mid_batch` wrote only 24 of the shader's 26 `buffer_reference` params => every later slot shifted, the tail dereferenced as a pointer), so "3.3x" was a crash read as slowness. Fixed and re-measured interleaved in one session it is a **net win of +27.9%~+42.9%** (B=32/64/128/256, §6.2②), and the end-to-end quality gate passed ⇒ **the default is now the tensor core** (§6.3).
  Three separate gaps remain; stop collapsing them into "Vulkan is slower":
  1. **Small batches are slightly slower** (tensor core / SIMT: B=1 0.88x, B=8 0.98x). This is *not* dispatch count - `PROF_GAPS=1` shows 1.3-1.8 us between dispatches, 0.5 ms total per recorded batch (2047 dispatches), and forcing `VK_IMMA_KSPLIT=1` (three fewer dispatches per layer) is slower. The cost is BN=8 using 1/8 of its slot lanes at batch=1 plus the split-K partial round-trip. ⇒ a **tile variant dedicated to batch<=4** is needed (`IM_BM`/`IM_NST`/KSPLIT were never swept at small batch).
  2. ~~**The two-stage low-rank chain still only has a SIMT batched kernel** (`gemv_lowrank_chain4_batch`), where CUDA has the two-stage fp16 tensor-core `lowrank_gemm_batch`.~~ — **level 2 landed on coopmat fp16 tensor cores 2026-10-11 (§6.9): B=128 decode 66.8 → 57.2 ms/step (+16.9 %), default ON (`LOWRANK_GEMM2=0` to force off). Level 1 stays SIMT — measured, not assumed: the same `kernel_bench` arms put stage 1 at 0.443 ms/layer on TC vs 0.274 SIMT, because N=64..320 leaves coopmat tiles mostly idle.** ⚠️ **2026-10-11 (S30): that A/B is the wrong frame** — CUDA's peer is the *whole chain* on tensor cores (four level-1 chains merged into one launch, epilogues inlined into level 2 — see the S30 bullet above for the kernel geometry), and its ablation costs 5.1 ms/step for the whole chain vs Vulkan's ≈15.4 ms; the re-opened lever is task S31, not this decision. ★ **2026-10-11 (S31): and the 0.443 itself was block starvation, not a tensor-core limit** — divisibility-only tile selection gave the level-1 shape (M=128 × N=64..320 × K=2560) only **1..5 blocks for 68 SMs**; the host-side occupancy rule now picks (64,32,64) ⇒ 4..20 blocks and **0.029 ms/call (≈0.118 ms/layer, `sum(y)` valid) = 2.3× faster than SIMT's 0.274**. Routing the decode mid arm through that path is the next step; the rule by itself also made deep-K narrow-N `pf_value` **1.94×** faster at kernel level (S13). ★ **2026-10-11 (S31 ①-b) landed**: the decode mid arm now runs through `gemm`/`gemm_tanh` on coopmat fp16 behind `LOWRANK_GEMM1` (**default ON at `batch >= 64`** — one step higher than level 2's 32, because B=32 measures a net **−8~−10 %** loss, B=64 a tie, B=128 **58.0 → 51.4 ms/step = +12.8 %**), the SIMT mid arm retires, and the gate is verified by a perf-independent binding marker (`PROF_LOW=1` ⇒ `stage1=true` + the `M=64/128 N=64/128/320 K=2560` tile lines). Quality 口径: fp32→fp16 at level 1 **does** diverge the greedy token stream (unlike level 2's bit-identical admission), so it was admitted on the same-ruler control instead — all-SIMT vs shipped = 0.861 mean|Δ|, all-SIMT vs new = 0.814, shipped vs new = 0.837 with 0/128 argmax divergence. Data + the two instrument lessons (`BACKEND=vulkan` must be explicit; cross-batch-width runs must pin `PREFILL_SLOTS=8`, which otherwise slows decode ≈1.9×): `参考/2026-10-11-Vulkan-mid投影臂向量化-实施记录.md` §11. ★ **2026-10-11 (S31 ②): split-K was built for the 4/8/20-block starvation and falsified as a throughput lever — it is NOT the next step.** A coopmat fp16 split-K pair (`gemm_splitk.comp` + ascending-ks `gemm_splitk_reduce.comp`, op applied *after* summation, plumbed as a trait *provided* method so CUDA is untouched) measures, in a quiet same-session interleaved triple with `PREFILL_SLOTS=8` pinned, **+1.2 / −0.6 / +6.7 %** at B=128 and **+2.9 / −3.0 / +8.2 %** at B=64 against `ks=1` — pooled +2.6 %/+2.8 %, inside base's own 5.2 % repeat spread, with inconsistent signs. Quality passes the same ruler as ①-b (split-K vs shipped = 0.876 mean|Δ|, 0/128 argmax divergence; and A↔B reproduced §11.4's 0.8368 to the digit, so the ruler itself is stable across sessions). What it *does* buy: **34~41 %** when the card is being preempted (forced `ks=4` in a contended window) — i.e. contention insurance, not throughput. ⇒ default **OFF** (`GEMM_SK` unset ⇒ `ks=1`, which falls back to the exact pre-existing dispatch, reduce kernel never emitted); the arm and both knobs stay in-tree. Record §12. ★ 2026-10-10: measured with `PROF_GPU=1 SKIP_GRAPH=1` at B=128 decode rather than guessing - `gemm_imma` **48.7%**, low-rank chain **15.5%**, state kernel `fuse_ka_dplr_norm` **14.5%**, rkv stage-2 `gemv_int8_rkv_stage1_batch` **10.6%**, `gemm_imma_reduce` 4.4%, `quant_x_i8` 2.0% ⇒ **the three SIMT kernels together are 40.6%** while the tensor-core GEMMs take 48.7%. ⚠️ **Superseded 2026-10-11** by a fresh dump after both the `gemm_imma` rounds and the fp16 state landed (`参考/2026-10-11-Vulkan-状态fp16存储-实施记录.md` §3.7): the decode denominator fell from ≈200 ms to 122 ms per 2 steps, and the shares are now `gemm_imma` **38.6 %**, **low-rank chain 20.9 %**, state kernel 15.0 %, rkv stage-2 13.9 % - the three SIMT kernels are **49.8 %**. The fp16-state saving is fully attributed to the state kernel itself (22.79 → 18.35 ms = −19.5 %, everything else inside cross-process noise), which also records that **fp16 *storage* does not halve this kernel's time** - the arithmetic stays fp32. And the low-rank chain is now provably neither bandwidth- nor peak-bound: 40× the bandwidth floor, **6.6 % of fp32 SIMT peak** - it is latency-bound by subgroup-per-row reductions over K=64/96, which is why a tensor-core arm is the lever rather than a tuning sweep. The same dump works out to ~100 ms of GPU per step against 88 ms/step unprofiled ⇒ **the GPU is saturated; large batch is not a host/submission bottleneck** (which also lowers expectations for point 3). Data: `参考/2026-10-10-Vulkan-rkv张量核默认翻转-实施记录.md` §3.5. ⇒ After §6.9 that 20.9 % is **split**, not removed: level 2 (0.3977 → 0.1069 ms/layer on TC) leaves only level 1's 0.274 ms/layer of SIMT in this chain, so the chain's share falls to ≈12 % and the "three SIMT kernels = 49.8 %" line is ≈41 % (折算 from the same per-avgs; the pairwise CUDA re-measure is what has to confirm it, not arithmetic).
  3. **CUDA merges the three r/k/v chains into one dispatch (`RKV_MERGE`); Vulkan does not**, so it records 3 extra `quant_x_i8` + 3 extra `gemm_imma_reduce` per layer. Expectations for merging them are revised *down* by point 1: it cannot rescue B=1; the value is partial traffic and host recording at large batch.
  Current standing, default configuration vs CUDA (same-session interleaved): **0.754x (B=1) / 0.804x (B=8) / 0.632x (B=32) / 0.413x (B=128)** (§6.3). ⚠️ The B>=192 slot must not be charged entirely to kernel quality: §6.2③ recorded a 57x sampler slowdown there and read it as WDDM spill. The footprint has since been cut - `imma_partial` was 64x oversized, and reclaiming it took B=256 from 17 807 to 12 747 MiB registered / 20 585 to 15 528 MiB external, **bit-exact and throughput-neutral** (§6.2③-bis) - and in the current build the B>=192 slowdown **does not reproduce at all** (1465-1505 tok/s, monotone in batch, TOPP=1.0, alternating rounds). So capacity is no longer the open explanation; what is left is the 0.41x itself. The per-slot fp32 RNN state that used to sit here as "the remaining footprint item" **is now halved** (Vulkan `_s16` state arm, §9), so the only items left in that 0.41x are kernel-side: the low-rank chain (§9 item 2) and batched prefill. ⚠️ **§6.9 has since taken the chain's second level onto tensor cores**, and the 4th §6.1 re-measure
(session `20261011-074644-90876`, CUDA columns byte-identical to the previous one) moved it to
**0.874× (B=8) / 0.740× (B=32) / 0.627× (B=128)** — B=8 is the control (the gate is `batch >= 32`), and
the +19.4 % at B=128 is the combined step of §6.9 **and** the fp16-state arm, so the isolated §6.9
number stays §5.2/§5.3's. What the remaining 0.373 at B=128 is made of is item 1 (small-batch tile) plus
**whatever is still inside `gemm_imma` and the state/rkv kernels** — the low-rank chain is no longer the
largest SIMT item; the r·k·v stage-1 mid projection (0.274 ms/layer) and `fuse_ka_dplr_norm_s16` are.
- ~~**Batched prefill lags further than decode and is the real large-batch cost**~~ — **measured and largely fixed 2026-10-11 (§6.6)**. The old 口径 (2026-10-10 `PROF_GPU`): at B=128 `prefill 42,367 ms / decode 5,831 ms` ⇒ prefill is 7.3x the decode wall clock, CUDA at the same parameters `24,859 / 2,403 ms` ⇒ **prefill ran at 0.59x of CUDA**, and Vulkan batched prefill's GEMMs are fp16 coopmat with no int8 variant (`gemm_f16_f32_affine` 37% + `_relu2` 27% + `gemm_f16_f32` 22% = **86%**, `dequant_int8_f16` only **0.9%**) ⇒ the cost was the fp16 GEMM itself. What it was *not* was an int8-tensor-core gap: the NVIDIA tile-selection branch gated the 128x128 tile on `max_shared_memory >= 163840` while this card reports 48 KB, so prefill ran pinned at 64x64x32 even though 128x128x32 fits in 40 KB. One tile-selection change (+ a shape-divisibility guard so the low-rank n=64 shapes can't read/write OOB) took batched prefill B=128 from 40,421 ms to **24,119 ms** = **1.68x**, bit-identical seeds. ⚠️ The "= 0.96x of CUDA" attached to that pair, and the "B=256 = 0.61x" below it, were both read through a contaminated window; on the clean protocol (§6.7) the same two points are **0.84x** (18.2 vs 15.3 s) and **0.79x** (36.0 vs 28.5 s, after the state hand-back moved onto the device). What is left in this item: `pf_value` (deep-K narrow-N) is still 0.45x of cuBLAS at kernel level, and int8 prefill is demoted behind it (dequant is 0.9%, fp16 is at parity). ⚠️ 4-5x of that workload remains **padding waste** (prompts are 104..139 tokens but `PAD_TO=512`) - a bucketing problem on the consumer side, common-mode between backends.
- ~~**Vulkan still keeps the RNN state in fp32** — the only remaining large per-slot VRAM lever~~ —
  **done 2026-10-11**: Vulkan now has CUDA's arm exactly, **fp16 storage + fp32 compute**
  (`supports_f16_wkv_state()` is true). Implementation = one extra spec dimension `WKV_S16` on the three
  state kernels (`fuse_ka_dplr_norm_s16.spv` / `dplr_seq_f32_f32_s16.spv` / `dplr_seq_batch_s16.spv`,
  naming aligned with CUDA's `{base}_s16`) plus a host-side `StateTensor` enum that makes the spv file
  name follow the state tensor's dtype. `Params` slots were always just 64-bit device addresses, so
  **the dtype swap needs no change to the dispatch parameter layout** — which is why this was far
  smaller than the recorded blocker suggested (that blocker read "`BufS{float data[]}` needs a new
  `buffer_reference`/spec variant", which is the work itself, not an obstacle).
  Measured (same-session interleaved, 3 rounds, B=128): fp16 **1977.5 / 1947.0 / 1914.8** vs fp32
  **1881.5 / 1881.7 / 1863.0** tok/s ⇒ median **+3.5 %** with **non-overlapping** distributions;
  registered tensors **7788 → 6498 MiB (−1290 MiB)** and the host mirror drops by the same amount
  ⇒ ≈ **−2.6 GB** at B=256. Numeric gate: greedy per-token fork rate against the fp32 state is
  **21.59 %** at B=64×512 (CUDA shipped its fp16 state on the same config at an accepted 23.26 %) and
  26.03 % at B=8×512 — first fork at step 0 is expected there, because both arms produce
  **bit-identical prefill seeds** and only the state store-back width differs. The default flip rests
  on a **sampling-free accuracy gate** (same CPU-fp32 reference, four arms = 2 backends × 2 state
  widths): CUDA pays **+42 %** logit rmse (2.691 → 3.823) and 46/64 divergent greedy self-loop tokens
  for its fp16 state, while **Vulkan pays nothing measurable** — rmse 3.227 → 3.223 (−0.12 %, the fp16
  arm is marginally *closer* to the reference) and **0/64** self-loop divergence. Both arms bind only
  `_s16` state kernels in the default arm and only the plain ones with `WKV_STATE_F16=0`
  (`LOG_BARRIERS=1`), so the switch is provably live. ⚠️ One metric in that harness is a **false
  witness**: `seq vs cpu max_abs_diff` reads identical within each backend (Vulkan 0.100290, CUDA
  0.483808 in both arms) because the max is taken over the concatenated sequence and is set by
  **token 1**, whose input state is all zeros (fp16 is lossless on zero) — it hides everything behind
  that ceiling; only rmse and the greedy sequence are citable here. The deployment summarize pipeline
  was tried first and **rejected as non-discriminating** for this change (2×2 matrix: same-contract
  pairs are *less* similar than cross-contract ones, and both similarity ratios normalise by
  `max(len)` while length is exactly the axis the change moves).
  `WKV_STATE_F16=0` switches the whole arm back. Data and pitfalls:
  `参考/2026-10-11-Vulkan-状态fp16存储-实施记录.md`.
- **Run one throwaway process before any A/B** (§6.2④): the Vulkan pipeline cache is memory-only and the driver keys its disk cache by SPIR-V hash, so the first process after a shader rebuild compiles pipelines inside the timed region. This round it manufactured a 6.4x fake win (0.52 -> 2.41 summaries/s); interleaved, both arms are level. Single-round, single-side readings are never citable.
- ~~**Vulkan's tensor-core batched GEMM only supports batch ≤ 64**~~ — **that was a bug, fixed 2026-10-10**. `imma_ok`'s hard `(1..=64).contains(&batch)` predicate silently dropped **B>64 onto the SIMT path**. Note what that actually cost: it is *not* "rkv fell back to SIMT" (rkv never used IMMA by default anyway - see `imma_op_on`), it is **all four tensor-core GEMMs - head/out/key/val - switching off together**, and head (vocab 65536×2560) is the single largest matmul in a layer. Throughput collapsed to a B=1-sized plateau: measured cliff at the same session and parameters - B=64 tensor core **934.4** tok/s vs B=80 **430.7** / B=96 **429.2** under the old gate (control: B=64 with IMMA forced off via `VK_IMMA=0` = **436.5**, i.e. the drop was 100 % the gate). Multi-wave (`grid.z = ceil(batch/(8·NST))`) had **already been written** into `gemm_imma.comp` - load guarded by `s0base + (i>>5) < BATCH`, store by `slot >= BATCH` - it had simply never been *executed*. Lifting the gate exposed two more sizing bugs, also fixed: `WorkBuffers.xb_q`/`xb_aux` were capped at `bsz.min(64)` while the quantize kernel writes all `bsz` rows by global slot index (wave 2 wrote out of bounds → `ERROR_DEVICE_LOST` at B=128), and `imma_partial` needed a capacity bound. ⚠️ What this entry then claimed ("must **not** scale with `bsz` … a `bsz`-sized reserve asks for 63 MiB more") was **wrong and was never reverted**: `imma_ksplit` requires `ks·batch·m ≤ partial.len`, so the correct length is `8 × max(c, ffn_hidden)` **per slot**; the attempted fix left the helper's `× bsz` in place and multiplied by 64 again, which cost **64× the right size** (5.1 GB at B=256) until it was found and reclaimed on 2026-10-11 - see §6.2③-bis. After the fix (formal-session medians): B=96 **913.2**, B=128 **967.4**, B=256 **988.5** tok/s, i.e. **2.1×–2.3×**. Guard test: `tests/batch_parity.rs::slots_beyond_the_first_wave_match` (72 slots straddling the 64-slot wave boundary, greedy, bitwise equality against wave 0) - this class of bug is invisible to throughput sweeps and shows up as user-reported garbage above B=64.
- **Accuracy headroom**: calibration-weighted k-means, G=64 denser metadata.

## 10. License

MIT. See [Cargo.toml](Cargo.toml) for dependencies. Research-oriented; please verify model weights (RWKV-7 Goosed weight license) and third-party dependency licensing before commercial use.