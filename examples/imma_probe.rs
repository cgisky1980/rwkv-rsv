//! 临时探针：单独跑 `quant_x_i8` + `gemm_imma`，检查
//!   ① 同一输入重复执行是否**确定**（竞态排查）；
//!   ② 与 CPU 参考实现是否一致（数值正确性）。
//!
//! 用法：`cargo run --release --example imma_probe`（`OP=0|1|2`、`M=`、`K=`、`BATCH=`、`REPS=`）。
//! ⚠️ 用完即删（仅诊断用）。

use half::f16;

use rwkv_rsv::backend::{
    ComputeBackend, Int8Handle, TensorDtype, TensorId, create_backend, detect_backend,
};

const K_GROUP: usize = 128;

fn lcg(state: &mut u64) -> u32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (*state >> 33) as u32
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let m: usize = std::env::var("M")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(128);
    let k: usize = std::env::var("K")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(256);
    let batch: usize = std::env::var("BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);
    let op: u32 = std::env::var("OP")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(0);
    let reps: usize = std::env::var("REPS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(8);

    assert_eq!(k % K_GROUP, 0);
    assert_eq!(m % 64, 0);

    let mut rng = 0x12345678u64;
    let g = k / K_GROUP;

    // —— 宿主侧构造：权重 idx（0..255）/ sz（fp16 尺度|零点）/ 激活 ——
    let idx_words = m * (k / 4);
    let diag = std::env::var("DIAG").is_ok();
    // DIAG=ones：激活全 1（q 全 127）+ 权重每行常量 `(r%5)+1` ⇒ `y(slot, r) = 128*((r%5)+1)`，
    // 只与行号有关、与 k/槽无关 ⇒ **专门检验 A 片段的「行」映射**（B 是否损坏在此测试里不可见）。
    let ones = std::env::var("DIAG").map(|v| v == "ones").unwrap_or(false);
    // DIAG：权重做成「对角」——第 r 行只有 k = r % 128 处 byte=1（= q_u 1），其余 0，
    // 且 scale=1 / zero=0 ⇒ `y(slot, r) = sx * q[slot][r % 128]`，
    // 于是可以把 GPU 的输出直接反解成「它认为的 q 值」，从而定位 A/B 的行列映射。
    let idx: Vec<u32> = if ones {
        let mut v = vec![0u32; idx_words];
        for r in 0..m {
            let byte = ((r % 5) + 1) as u32;
            let word = byte * 0x0101_0101u32;
            for w in 0..k / 4 {
                v[r * (k / 4) + w] = word;
            }
        }
        v
    } else if diag {
        let mut v = vec![0u32; idx_words];
        for r in 0..m {
            let kx = r % K_GROUP;
            v[r * (k / 4) + kx / 4] |= 1u32 << (8 * (kx % 4));
        }
        v
    } else {
        // ⚠️ 权重字节用**居中**随机（`q_u ∈ [124,132]` ⇒ s8 ∈ [-4,4]）：均匀 0..255 会让
        // 每组的零点和修正项（~3e4）与结果（~0.3）严重抵消，相对误差度量失去意义。
        (0..idx_words)
            .map(|_| {
                let mut w = 0u32;
                for j in 0..4 {
                    let b = 124 + (lcg(&mut rng) % 9);
                    w |= (b & 0xFF) << (8 * j);
                }
                w
            })
            .collect()
    };
    let sz: Vec<u32> = (0..m * g)
        .map(|_| {
            // 尺度取 0.5..1.5 的正数，零点取 0（探针只验证算术路径，不追求真实分布）
            let s = if diag {
                1.0
            } else {
                0.5 + (lcg(&mut rng) % 1000) as f32 / 1000.0
            };
            (f16::from_f32(0.0).to_bits() as u32) << 16 | f16::from_f32(s).to_bits() as u32
        })
        .collect();
    let x: Vec<f32> = if ones {
        vec![1.0f32; batch * k]
    } else {
        (0..batch * k)
            .map(|_| (lcg(&mut rng) % 2000) as f32 / 1000.0 - 1.0)
            .collect()
    };

    // —— 后端张量 ——
    let mut b = create_backend(detect_backend())?;
    let mk_u32 =
        |b: &mut dyn ComputeBackend, n: usize| -> Result<TensorId, Box<dyn std::error::Error>> {
            let t = b.create_tensor(n, TensorDtype::U32)?;
            b.upload_u32(t, &vec![0u32; n])?;
            Ok(t)
        };
    let mk_f32 =
        |b: &mut dyn ComputeBackend, n: usize| -> Result<TensorId, Box<dyn std::error::Error>> {
            let t = b.create_tensor(n, TensorDtype::F32)?;
            b.upload(t, &vec![0.0f32; n])?;
            Ok(t)
        };

    let a_idx = mk_u32(&mut *b, idx_words)?;
    let a_sz = mk_u32(&mut *b, m * g)?;
    b.upload_u32(a_idx, &idx)?;
    b.upload_u32(a_sz, &sz)?;
    let a = Int8Handle {
        idx: a_idx,
        sz: a_sz,
        m,
        k,
    };

    let xt = mk_f32(&mut *b, batch * k)?;
    b.upload(xt, &x)?;
    let xq = mk_u32(&mut *b, 8 * (k / 4))?;
    let xaux = mk_f32(&mut *b, 8 * g * 4)?;
    let y0 = (0..8 * m)
        .map(|i| (i % 7) as f32 * 0.25 - 0.5)
        .collect::<Vec<f32>>();

    let partial = mk_f32(&mut *b, 8 * 8 * m)?;
    let q = quantize(&x, k, batch);

    let mut runs: Vec<Vec<f32>> = Vec::new();
    for _ in 0..reps {
        let y = mk_f32(&mut *b, 8 * m)?;
        b.upload(y, &y0)?;
        b.begin_batch()?;
        b.quant_x_i8(xt, None, xq, xaux, k, batch)?;
        b.gemm_imma(&a, xq, xaux, y, partial, m, k, batch, op)?;
        b.end_batch()?;
        runs.push(b.download(y)?);
    }

    // ⓪ 逐项核对 `quant_x_i8` 的产物（xq / xaux）——gemm 出错时先确认 A 侧输入本身对。
    {
        let xq_h = b.download_u32(xq)?;
        let aux_h = b.download(xaux)?;
        let mut bad_q = 0usize;
        let mut bad_aux = 0usize;
        let mut first = String::new();
        for s in 0..batch {
            for kx in 0..k {
                let got = ((xq_h[s * (k / 4) + kx / 4] >> (8 * (kx % 4))) & 0xFF) as i32;
                let got = if got >= 128 { got - 256 } else { got };
                if got != q.q[s * k + kx] {
                    bad_q += 1;
                    if first.is_empty() {
                        first = format!("q slot{s} k{kx}: gpu={got} cpu={}", q.q[s * k + kx]);
                    }
                }
            }
            for gi in 0..g {
                let a = &aux_h[(s * g + gi) * 4..(s * g + gi) * 4 + 4];
                let want = [
                    q.sx[s * g + gi] as f32,
                    (128.0 * q.sx[s * g + gi] * q.cs[s * g + gi]) as f32,
                    q.rs[s * g + gi] as f32,
                    0.0,
                ];
                for j in 0..4 {
                    if (a[j] - want[j]).abs() > 1e-3 * want[j].abs().max(1.0) {
                        bad_aux += 1;
                        if first.is_empty() {
                            first =
                                format!("aux slot{s} g{gi} [{j}]: gpu={} cpu={}", a[j], want[j]);
                        }
                    }
                }
            }
        }
        println!("[probe] quant_x_i8 核对：q 失配 {bad_q}，aux 失配 {bad_aux}  {first}");
    }

    // ① 确定性
    let mut nondet = 0usize;
    for r in runs.iter().skip(1) {
        if r.iter().zip(&runs[0]).any(|(a, c)| a != c) {
            nondet += 1;
        }
    }
    println!("[probe] reps={reps} 与第 1 次不同的次数 = {nondet}");
    if nondet > 0 {
        for s in 0..batch {
            let d = runs[0][s * m..(s + 1) * m]
                .iter()
                .zip(&runs[1 % runs.len()][s * m..(s + 1) * m])
                .filter(|(a, c)| a != c)
                .count();
            println!("  slot {s}: 不同元素 {d}/{m}");
        }
    }

    // ② 数值：CPU 参考（fp64 复刻 kernel 的口径）
    let mut max_rel = 0.0f64;
    let mut worst = String::new();
    let got = &runs[0];
    for s in 0..batch {
        for r in 0..m {
            let mut acc = 0.0f64;
            for gi in 0..g {
                let mut sdot = 0.0f64;
                for kk in 0..K_GROUP {
                    let kx = gi * K_GROUP + kk;
                    let byte = (idx[r * (k / 4) + kx / 4] >> (8 * (kx % 4))) & 0xFF;
                    let w = (byte as i32) - 128; // ^0x80 后的有符号解释
                    sdot += (w * q.q[s * k + kx] as i32) as f64;
                }
                let scale = f16::from_bits((sz[r * g + gi] & 0xFFFF) as u16).to_f32() as f64;
                let zero = f16::from_bits((sz[r * g + gi] >> 16) as u16).to_f32() as f64;
                let sx = q.sx[s * g + gi];
                let cs = q.cs[s * g + gi];
                let rs = q.rs[s * g + gi];
                acc += scale * (sx * sdot + 128.0 * sx * cs) + zero * rs;
            }
            let expect = match op {
                0 => {
                    if acc > 0.0 {
                        acc * acc
                    } else {
                        0.0
                    }
                }
                1 => y0[s * m + r] as f64 + acc,
                _ => acc,
            };
            let gv = got[s * m + r] as f64;
            let rel = (gv - expect).abs() / expect.abs().max(1e-3);
            if rel > max_rel {
                max_rel = rel;
                worst = format!("slot {s} row {r}: gpu={gv:.6} cpu={expect:.6}");
            }
        }
    }
    println!("[probe] 最大相对误差 = {max_rel:.3e}  最差点: {worst}");
    // 抵消会把「相对误差」放大到无意义 ⇒ 同时给出**绝对**误差与量级（fp32 判据：绝对误差
    // 应与 Σ|组贡献| × 1e-7 同阶，而不是与结果本身比）。
    {
        let (mut max_abs, mut scale) = (0.0f64, 0.0f64);
        for s in 0..batch {
            for r in 0..m {
                let mut acc = 0.0f64;
                let mut abs_terms = 0.0f64;
                for gi in 0..g {
                    let mut sdot = 0.0f64;
                    for kk in 0..K_GROUP {
                        let kx = gi * K_GROUP + kk;
                        let byte = (idx[r * (k / 4) + kx / 4] >> (8 * (kx % 4))) & 0xFF;
                        sdot += (((byte as i32) - 128) * q.q[s * k + kx] as i32) as f64;
                    }
                    let sc = f16::from_bits((sz[r * g + gi] & 0xFFFF) as u16).to_f32() as f64;
                    let z = f16::from_bits((sz[r * g + gi] >> 16) as u16).to_f32() as f64;
                    let t = sc
                        * (q.sx[s * g + gi] * sdot + 128.0 * q.sx[s * g + gi] * q.cs[s * g + gi])
                        + z * q.rs[s * g + gi];
                    acc += t;
                    abs_terms += t.abs();
                }
                let expect = match op {
                    0 => {
                        if acc > 0.0 {
                            acc * acc
                        } else {
                            0.0
                        }
                    }
                    1 => y0[s * m + r] as f64 + acc,
                    _ => acc,
                };
                max_abs = max_abs.max((got[s * m + r] as f64 - expect).abs());
                scale = scale.max(abs_terms.max(expect.abs()));
            }
        }
        println!(
            "[probe] 最大绝对误差 = {max_abs:.3e}  量级 = {scale:.3e}  ⇒ 归一到量级 = {:.3e}",
            max_abs / scale
        );
    }

    if ones {
        println!(
            "  DIAG=ones 期望 y(slot,r) = 128*((r%5)+1) ∈ {{256,384,512,640,768}}（与 slot 无关）"
        );
        for s in 0..2usize {
            let vals: Vec<String> = (0..16).map(|r| format!("{:.1}", got[s * m + r])).collect();
            println!("   slot {s} r0..15: {}", vals.join(" "));
        }
    }
    if diag {
        // y(slot, r) = sx * q[slot][r % 128] ⇒ 反解 GPU 认为的 q，并找出它在 q 网格中的位置。
        for s in 0..2usize {
            let sx = q.sx[s * g];
            for r in 0..8usize {
                let yv = got[s * m + r] as f64;
                // 权重只有 k=r%128 处非零 ⇒ 只有第 0 组的贡献非零、其余组精确相消
                // ⇒ `y = sx · q[slot][r%128]`，直接反解即可。
                let infer = (yv / sx).round() as i32;
                let want = q.q[s * k + r % K_GROUP];
                let pos = (0..k)
                    .find(|&j| q.q[s * k + j] == infer)
                    .map(|j| j as i64)
                    .unwrap_or(-1);
                println!(
                    "  DIAG s{s} r{r}: 期望 k={}  want_q={want:>5} | 反解 q={infer:>5} 出现在 k={pos}",
                    r % K_GROUP
                );
            }
        }
    }

    // ③ 失配分布：按 (slot, 行/16) 统计，用于判断是不是「行错位/warp 归属」类问题
    for s in 0..batch {
        let mut row = String::new();
        for blk in 0..(m / 16) {
            let mut bad = 0;
            let mut shown = 0;
            for r in blk * 16..(blk + 1) * 16 {
                let mut acc = 0.0f64;
                for gi in 0..g {
                    let mut sdot = 0.0f64;
                    for kk in 0..K_GROUP {
                        let kx = gi * K_GROUP + kk;
                        let byte = (idx[r * (k / 4) + kx / 4] >> (8 * (kx % 4))) & 0xFF;
                        sdot += (((byte as i32) - 128) * q.q[s * k + kx] as i32) as f64;
                    }
                    let scale = f16::from_bits((sz[r * g + gi] & 0xFFFF) as u16).to_f32() as f64;
                    let zero = f16::from_bits((sz[r * g + gi] >> 16) as u16).to_f32() as f64;
                    acc += scale
                        * (q.sx[s * g + gi] * sdot + 128.0 * q.sx[s * g + gi] * q.cs[s * g + gi])
                        + zero * q.rs[s * g + gi];
                }
                let expect = match op {
                    0 => {
                        if acc > 0.0 {
                            acc * acc
                        } else {
                            0.0
                        }
                    }
                    1 => y0[s * m + r] as f64 + acc,
                    _ => acc,
                };
                let gv = got[s * m + r] as f64;
                if (gv - expect).abs() > 1e-3 * expect.abs().max(1.0) {
                    bad += 1;
                    if shown < 2 {
                        println!("    s{s} r{r}: gpu={gv:.5} cpu={expect:.5}");
                        shown += 1;
                    }
                }
            }
            row.push_str(&format!("{bad:>3}"));
        }
        println!("  slot {s} 每 16 行失配数: {row}");
    }
    Ok(())
}

struct Q {
    q: Vec<i32>,
    sx: Vec<f64>,
    cs: Vec<f64>,
    rs: Vec<f64>,
}

/// fp64 复刻 `quant_x_i8.comp`：对称 int8 分组量化（组宽 128）。
fn quantize(x: &[f32], k: usize, batch: usize) -> Q {
    let g = k / K_GROUP;
    let mut q = vec![0i32; batch * k];
    let (mut sx, mut cs, mut rs) = (
        vec![0.0; batch * g],
        vec![0.0; batch * g],
        vec![0.0; batch * g],
    );
    for b in 0..batch {
        for gi in 0..g {
            let base = b * k + gi * K_GROUP;
            let amax = (0..K_GROUP).fold(0.0f32, |a, i| a.max(x[base + i].abs()));
            // ⚠️ 必须**逐位复刻 shader 的 fp32 路径**：`sx = amax·(1/127)`、`inv = 1/sx`、
            // `q = roundEven(v·inv)`，全部在 f32 里。任何一处换成 f64 的等价写法（如 `v/s`）
            // 都会在 .5 边界上偶发 1 LSB 差异，而 1 LSB 会让 `cs` 差 1 ⇒ `128·sx·cs` 差 ~1.0，
            // 直接把探针误差抬到 O(1)（本轮排查就栽在这上面）。
            let s = if amax > 0.0 {
                amax * (1.0f32 / 127.0)
            } else {
                1.0
            };
            let inv = 1.0f32 / s;
            let mut c = 0i32;
            let mut r = 0.0f64;
            for i in 0..K_GROUP {
                let v = x[base + i];
                let qv = (v * inv).round_ties_even().clamp(-127.0, 127.0) as i32;
                q[base + i] = qv;
                c += qv;
                r += v as f64;
            }
            let s = s as f64;
            sx[b * g + gi] = s;
            cs[b * g + gi] = c as f64;
            rs[b * g + gi] = r;
        }
    }
    Q { q, sx, cs, rs }
}
