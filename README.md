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
   ⚠️ **Two corrections from §6.2 (2026-10-10)**: (i) the premise inside that explanation — "tensor-core rkv
   would be slower on Vulkan" — is false; once its crash is fixed it is faster at every batch
   (+27.9 % … +42.9 %), so the recoverable share of this gap is much larger than assumed, it is just
   blocked on an unvalidated numeric contract rather than on speed; (ii) **the B=256 cell is not a clean
   kernel-quality reading** — at B ≥ 192 the Vulkan side fills VRAM and WDDM spills to system memory
   (sampler 0.87 ms → 68 ms per call, §6.2③), so 0.27× mixes in a capacity effect. B=64/128 are unaffected.
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
  comparison needs the per-slot footprint reduced first (candidates: Vulkan still keeps the RNN state in
  fp32 — `supports_f16_wkv_state()` is false, `H·N·N = 163 840` elements per layer per slot ⇒ ~5.3 GB of
  state at B=256, fp16 would save 2.6 GB — and the per-slot `WorkBuffers` workspace). Filed under §9;
  not touched this round.

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
- **The only batched chain still missing a tensor core on Vulkan is the low-rank one** (`lowrank_gemm_batch` is CUDA-only. ⚠️ This entry used to charge **FFN value** to "SIMT only" as well, which was **wrong**: at batch >= 16 CUDA runs the dense tiled `ffn_value_gemm_batch` / `ffn_value_imma_batch`, while Vulkan runs the generic `imma_op_on("val")` path `gemm_imma(M=C, K=fh, OP=accumulate)` - both are tensor cores, just different shapes): every other batched tensor-core op (head/out/key/val **and rkv**) is now on by default - the `VK_IMMA_OPS` default flipped from "everything but rkv" to "everything" on 2026-10-10, see §6.3.
  ~~The Vulkan tensor-core r/k/v variant measured a 3.3x net loss~~ - **that was a misreading, corrected 2026-10-10**: the path hit `ERROR_DEVICE_LOST` at every batch (`gemv_int8_rkv_mid_batch` wrote only 24 of the shader's 26 `buffer_reference` params => every later slot shifted, the tail dereferenced as a pointer), so "3.3x" was a crash read as slowness. Fixed and re-measured interleaved in one session it is a **net win of +27.9%~+42.9%** (B=32/64/128/256, §6.2②), and the end-to-end quality gate passed ⇒ **the default is now the tensor core** (§6.3).
  Three separate gaps remain; stop collapsing them into "Vulkan is slower":
  1. **Small batches are slightly slower** (tensor core / SIMT: B=1 0.88x, B=8 0.98x). This is *not* dispatch count - `PROF_GAPS=1` shows 1.3-1.8 us between dispatches, 0.5 ms total per recorded batch (2047 dispatches), and forcing `VK_IMMA_KSPLIT=1` (three fewer dispatches per layer) is slower. The cost is BN=8 using 1/8 of its slot lanes at batch=1 plus the split-K partial round-trip. ⇒ a **tile variant dedicated to batch<=4** is needed (`IM_BM`/`IM_NST`/KSPLIT were never swept at small batch).
  2. **The two-stage low-rank chain still only has a SIMT batched kernel** (`gemv_lowrank_chain4_batch`), where CUDA has the two-stage fp16 tensor-core `lowrank_gemm_batch`. ★ 2026-10-10: measured with `PROF_GPU=1 SKIP_GRAPH=1` at B=128 decode rather than guessing - `gemm_imma` **48.7%**, low-rank chain **15.5%**, state kernel `fuse_ka_dplr_norm` **14.5%**, rkv stage-2 `gemv_int8_rkv_stage1_batch` **10.6%**, `gemm_imma_reduce` 4.4%, `quant_x_i8` 2.0% ⇒ **the three SIMT kernels together are 40.6%** while the tensor-core GEMMs take 48.7%. The same dump works out to ~100 ms of GPU per step against 88 ms/step unprofiled ⇒ **the GPU is saturated; large batch is not a host/submission bottleneck** (which also lowers expectations for point 3). Data: `参考/2026-10-10-Vulkan-rkv张量核默认翻转-实施记录.md` §3.5.
  3. **CUDA merges the three r/k/v chains into one dispatch (`RKV_MERGE`); Vulkan does not**, so it records 3 extra `quant_x_i8` + 3 extra `gemm_imma_reduce` per layer. Expectations for merging them are revised *down* by point 1: it cannot rescue B=1; the value is partial traffic and host recording at large batch.
  Current standing, default configuration vs CUDA (same-session interleaved): **0.754x (B=1) / 0.804x (B=8) / 0.632x (B=32) / 0.413x (B=128)** (§6.3). ⚠️ The B>=192 slot must not be charged entirely to kernel quality: §6.2③ recorded a 57x sampler slowdown there and read it as WDDM spill. The footprint has since been cut - `imma_partial` was 64x oversized, and reclaiming it took B=256 from 17 807 to 12 747 MiB registered / 20 585 to 15 528 MiB external, **bit-exact and throughput-neutral** (§6.2③-bis) - and in the current build the B>=192 slowdown **does not reproduce at all** (1465-1505 tok/s, monotone in batch, TOPP=1.0, alternating rounds). So capacity is no longer the open explanation; what is left is the 0.41x itself, whose remaining footprint item is the fp32 RNN state (next entry) and whose remaining kernel items are the low-rank chain (§9 item 2) and batched prefill.
- **Batched prefill lags further than decode and is the real large-batch cost** (2026-10-10 `PROF_GPU`): at B=128 the same workload reads `prefill 42,367 ms / decode 5,831 ms` ⇒ prefill is **7.3x** the decode wall clock; CUDA at the same parameters reads `24,859 / 2,403 ms` ⇒ **prefill runs at 0.59x of CUDA** (decode 0.41x - the two must not be merged). Vulkan batched prefill goes through `forward_seq_batch`, whose GEMMs are **fp16 coopmat with no int8 tensor-core variant**: in one chunk `gemm_f16_f32_affine` 37% + `gemm_f16_f32_relu2` 27% + `gemm_f16_f32` 22% = **86%**, while `dequant_int8_f16` is only **0.9%** ⇒ the cost is the fp16 GEMM itself (~1.8x slower than CUDA), not dequantization. ⚠️ 4-5x of this workload is **padding waste** (prompts are 104..139 tokens but `PAD_TO=512`) - a bucketing problem, not a backend one. Prefill is chunked by `PREFILL_SLOTS=8` ⇒ 128 slots run as 16 serial chunks.
- **Vulkan still keeps the RNN state in fp32** — the **only remaining large per-slot VRAM lever** (the
  other half of this entry, `log_vram` accounting, is **fixed** 2026-10-10, see §6.2③-correction):
  `supports_f16_wkv_state()` returns false on Vulkan (CUDA true), `H·N·N = 163 840` elements per layer per
  slot × 32 layers × 4 B ⇒ **20.6 MiB per slot**, ~**5.3 GB** at B=256 — by far the largest per-slot item
  in the accounting (see §6.2③-correction for the full B=256 breakdown). fp16 halves it (−2.6 GB at B=256)
  and is what must happen before
  §6.1's B≥192 readings are comparable to CUDA on equal terms. Blocking detail: the state kernel
  `fuse_ka_dplr_norm.comp` declares its state arm as `BufS{float data[]}`, so an fp16 arm needs a new
  `buffer_reference`/spec variant before `supports_f16_wkv_state()` can flip. Numeric contract must be
  gated (fp16 state drifts vs fp32 across a long self-loop; `tests/batch_parity.rs` parity cases are the
  falsifier).
- **Run one throwaway process before any A/B** (§6.2④): the Vulkan pipeline cache is memory-only and the driver keys its disk cache by SPIR-V hash, so the first process after a shader rebuild compiles pipelines inside the timed region. This round it manufactured a 6.4x fake win (0.52 -> 2.41 summaries/s); interleaved, both arms are level. Single-round, single-side readings are never citable.
- ~~**Vulkan's tensor-core batched GEMM only supports batch ≤ 64**~~ — **that was a bug, fixed 2026-10-10**. `imma_ok`'s hard `(1..=64).contains(&batch)` predicate silently dropped **B>64 onto the SIMT path**. Note what that actually cost: it is *not* "rkv fell back to SIMT" (rkv never used IMMA by default anyway - see `imma_op_on`), it is **all four tensor-core GEMMs - head/out/key/val - switching off together**, and head (vocab 65536×2560) is the single largest matmul in a layer. Throughput collapsed to a B=1-sized plateau: measured cliff at the same session and parameters - B=64 tensor core **934.4** tok/s vs B=80 **430.7** / B=96 **429.2** under the old gate (control: B=64 with IMMA forced off via `VK_IMMA=0` = **436.5**, i.e. the drop was 100 % the gate). Multi-wave (`grid.z = ceil(batch/(8·NST))`) had **already been written** into `gemm_imma.comp` - load guarded by `s0base + (i>>5) < BATCH`, store by `slot >= BATCH` - it had simply never been *executed*. Lifting the gate exposed two more sizing bugs, also fixed: `WorkBuffers.xb_q`/`xb_aux` were capped at `bsz.min(64)` while the quantize kernel writes all `bsz` rows by global slot index (wave 2 wrote out of bounds → `ERROR_DEVICE_LOST` at B=128), and `imma_partial` needed a capacity bound. ⚠️ What this entry then claimed ("must **not** scale with `bsz` … a `bsz`-sized reserve asks for 63 MiB more") was **wrong and was never reverted**: `imma_ksplit` requires `ks·batch·m ≤ partial.len`, so the correct length is `8 × max(c, ffn_hidden)` **per slot**; the attempted fix left the helper's `× bsz` in place and multiplied by 64 again, which cost **64× the right size** (5.1 GB at B=256) until it was found and reclaimed on 2026-10-11 - see §6.2③-bis. After the fix (formal-session medians): B=96 **913.2**, B=128 **967.4**, B=256 **988.5** tok/s, i.e. **2.1×–2.3×**. Guard test: `tests/batch_parity.rs::slots_beyond_the_first_wave_match` (72 slots straddling the 64-slot wave boundary, greedy, bitwise equality against wave 0) - this class of bug is invisible to throughput sweeps and shows up as user-reported garbage above B=64.
- **Accuracy headroom**: calibration-weighted k-means, G=64 denser metadata.

## 10. License

MIT. See [Cargo.toml](Cargo.toml) for dependencies. Research-oriented; please verify model weights (RWKV-7 Goosed weight license) and third-party dependency licensing before commercial use.