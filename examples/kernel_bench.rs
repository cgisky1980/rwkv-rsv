//! 隔离内核计时基准（**不加载模型**）：对指定算子 + 形状输出 ms/call 与有效带宽。
//!
//! 为什么需要它：`batch_decode_bench` 是端到端的，改动某个内核后要跑完整推理才能看
//! 到收益；而 `PROF_CUDA_KERNEL` 的占比**不可信**——它在每个 kernel 后
//! `cuEventSynchronize`，实测 batch 段"内核耗时总和"≈ kernel 数 × 单次同步开销
//! （见 `backend_cuda.rs::launch_smem` 的注释）。本基准用「连续 N 次调用 + 末尾一次
//! 同步」计量，得到的是该算子的真实单次成本。
//!
//! 用法：
//! ```powershell
//! cargo run --release --example kernel_bench
//! $env:KB_OPS="relu2,mul_add"; $env:KB_BATCH="8,16"; cargo run --release --example kernel_bench
//! ```
//!
//! 环境变量（空字符串视为未设置，沿用项目惯例）：
//!   `KB_OPS`     逗号分隔算子，默认 `relu2,mul_add,plain,add`；可加 `f16`（fp16 权重）
//!   `KB_BATCH`   逗号分隔 batch，默认 `1,8,16`
//!   `KB_ITERS`   每组迭代次数，默认 300
//!   `KB_WARMUP`  预热次数，默认 5
//!   `KB_REPEATS` 重复轮数，默认 5，**取各轮最小值**（最小值最不易被桌面占卡/
//!                时钟波动污染；实测单轮均值在 0.03ms 级小内核上抖动可达 68%，
//!                取最小后降到个位数百分比）
//!   `KB_C` / `KB_FH` / `KB_VOCAB`  模型维度，默认 2560 / 10240 / 65536
//!
//! 形状按算子语义固定（与 `gpu_model.rs::forward_layer_batch` 的调用点一致）：
//!   `relu2`   = ffn.key      m=fh    k=c      （`gemv_int8_relu2`，单次最贵）
//!   `mul_add` = att.output   m=c     k=c      （`gemv_int8_mul_add`）
//!   `plain`   = head         m=vocab k=c      （`gemv_int8_plain`，读取量最大）
//!   `add`     = ffn.value    m=c     k=fh     （`gemv_int8_add`，稠密回退）
//!   `f16`     = head(fp16)   m=vocab k=c      （`gemv_f16`，对照组）
//!
//! **未覆盖**：`gemv_int8_rkv_stage1_batch`（26 参）与 `gemv_lowrank_chain4_batch`（20 参）
//! ——参数表过长，待 Phase 1 主战场（4 个平坦算子）收敛后再补。

use std::time::Instant;

use rwkv_rsv::backend::{
    ComputeBackend, Int8Handle, TensorDtype, TensorId, create_backend, detect_backend,
};

/// 逗号分隔环境变量解析（空字符串 = 未设置；历史教训：空串会被 `is_ok()` 判真）。
fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 极简 LCG（只用于填充数据，避免引入依赖）。
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// 一个待测用例：算子名 + (m, k) + 权重字节数（单次读）。
struct Case {
    op: String,
    m: usize,
    k: usize,
}

impl Case {
    /// int8 权重字节 = idx(m*k, 每元素 1B) + sz(m*(k/128), 每组 4B)。
    fn wbytes_int8(&self) -> f64 {
        (self.m * self.k + self.m * (self.k / 128) * 4) as f64
    }
    /// fp16 权重字节 = m*k*2。
    fn wbytes_f16(&self) -> f64 {
        (self.m * self.k * 2) as f64
    }
}

fn main() {
    let c = env_usize("KB_C", 2560);
    let fh = env_usize("KB_FH", 10240);
    let vocab = env_usize("KB_VOCAB", 65536);
    let batches: Vec<usize> = env_list("KB_BATCH", "1,8,16")
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let ops = env_list("KB_OPS", "relu2,mul_add,plain,add");
    let iters = env_usize("KB_ITERS", 300);
    let warmup = env_usize("KB_WARMUP", 5);
    let repeats = env_usize("KB_REPEATS", 5).max(1);

    let cases: Vec<Case> = ops
        .iter()
        .map(|op| match op.as_str() {
            "relu2" => Case {
                op: op.clone(),
                m: fh,
                k: c,
            },
            "mul_add" => Case {
                op: op.clone(),
                m: c,
                k: c,
            },
            "plain" => Case {
                op: op.clone(),
                m: vocab,
                k: c,
            },
            "add" => Case {
                op: op.clone(),
                m: c,
                k: fh,
            },
            "f16" => Case {
                op: op.clone(),
                m: vocab,
                k: c,
            },
            other => {
                eprintln!("未知算子 '{other}'（可选 relu2/mul_add/plain/add/f16）");
                std::process::exit(2);
            }
        })
        .collect();

    println!(
        "模型维度 c={c} fh={fh} vocab={vocab}；iters={iters} warmup={warmup} repeats={repeats}"
    );
    println!("kernel_bench: 隔离计时（不加载模型），有效带宽按「权重单次读」计");
    println!("口径：每轮 iters 次调用 + 一次同步，取 repeats 轮的最小值；抖动 = (最大-最小)/最小");
    println!();

    let backend = match create_backend(detect_backend()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("创建后端失败: {e:#}");
            std::process::exit(1);
        }
    };
    let mut b = backend;

    println!(
        "{:<8} {:>7} {:>6} {:>6} {:>11} {:>12} {:>10} {:>8}",
        "op", "m", "k", "batch", "ms/call", "权重 MB", "有效 GB/s", "抖动"
    );
    println!("{}", "-".repeat(76));

    for case in &cases {
        // 维度按内核假设校验：m 需是 4 的倍数（GEMV_ROWS），k 需是 128 的倍数（int8 分组）。
        if case.m % 4 != 0 || case.k % 128 != 0 {
            eprintln!(
                "跳过 {} m={} k={}：m 需为 4 的倍数、k 需为 128 的倍数",
                case.op, case.m, case.k
            );
            continue;
        }
        for &batch in &batches {
            let (ms, jitter) = match run_case(b.as_mut(), case, batch, iters, warmup, repeats) {
                Ok(v) => v,
                Err(e) => {
                    eprintln!("{} batch={} 失败: {e:#}", case.op, batch);
                    continue;
                }
            };
            let (wbytes, is_f16) = if case.op == "f16" {
                (case.wbytes_f16(), true)
            } else {
                (case.wbytes_int8(), false)
            };
            let gbs = wbytes / (ms / 1000.0) / 1e9;
            println!(
                "{:<8} {:>7} {:>6} {:>6} {:>11.4} {:>12.2} {:>10.1} {:>7.1}%{}",
                case.op,
                case.m,
                case.k,
                batch,
                ms,
                wbytes / 1e6,
                gbs,
                jitter * 100.0,
                if is_f16 { "  (fp16)" } else { "" }
            );
        }
    }
    println!();
    println!("注：int8 内核按 BGRP 分组，batch > BGRP 时权重实际被读 ceil(batch/BGRP) 遍；");
    println!("    上表「有效 GB/s」按单次读折算，故真实带宽 = 该值 × ceil(batch/BGRP)。");
    println!("    抖动超过 ~10% 时该行不可用于 A/B 判断（可用 KB_REPEATS 加大重复轮数）。");
}

/// 建张量 → 预热 → `repeats` 轮（每轮 `iters` 次调用 + 一次同步）→
/// 返回 `(最小 ms/call, 抖动)`，抖动 = (最大-最小)/最小。取最小以剔除外部占卡干扰。
fn run_case(
    b: &mut dyn ComputeBackend,
    case: &Case,
    batch: usize,
    iters: usize,
    warmup: usize,
    repeats: usize,
) -> Result<(f64, f64), Box<dyn std::error::Error>> {
    let (m, k) = (case.m, case.k);
    let mut rng = Rng(0x5A5A_1234_ABCD_0001);

    // 权重：int8（idx 打包 4×uint8 + sz 打包 scale/zero 两个 f16）或 fp16。
    let (a8, w16) = if case.op == "f16" {
        let t = b.create_tensor(m * k, TensorDtype::F16)?;
        b.upload(t, &vec![0.01f32; m * k])?;
        (None, Some(t))
    } else {
        let idx: Vec<u32> = (0..m * (k / 4)).map(|_| 0x7f7f_7f7fu32).collect();
        let sz: Vec<u32> = vec![0x3800_3c00u32; m * (k / 128)]; // 低 16=scale(1.0)，高 16=zero(0.5)
        let h = Int8Handle {
            idx: b.create_tensor(m * (k / 4), TensorDtype::U32)?,
            sz: b.create_tensor(m * (k / 128), TensorDtype::U32)?,
            m,
            k,
        };
        b.upload_u32(h.idx, &idx)?;
        b.upload_u32(h.sz, &sz)?;
        (Some(h), None)
    };

    // 激活 [batch, k] f32 / 门控 [batch, k] f16 / 输出 [batch, m] f32。
    let x = b.create_tensor(batch * k, TensorDtype::F32)?;
    let xs: Vec<f32> = (0..batch * k)
        .map(|_| 0.01 * (1.0 + rng.next_f32().abs()))
        .collect();
    b.upload(x, &xs)?;
    let g = b.create_tensor(batch * k, TensorDtype::F16)?;
    b.upload(g, &vec![0.5f32; batch * k])?;
    let y = b.create_tensor(batch * m, TensorDtype::F32)?;
    b.upload(y, &vec![0.0f32; batch * m])?;

    let call = |b: &mut dyn ComputeBackend| -> Result<(), Box<dyn std::error::Error>> {
        match case.op.as_str() {
            "relu2" => b.gemv_int8_relu2(a8.as_ref().unwrap(), x, y, m, k, batch),
            "mul_add" => b.gemv_int8_mul_add(a8.as_ref().unwrap(), x, g, y, m, k, batch),
            "plain" => b.gemv_int8_plain(a8.as_ref().unwrap(), x, y, m, k, batch),
            "add" => b.gemv_int8_add(a8.as_ref().unwrap(), x, y, m, k, batch),
            "f16" => b.gemv_f16(w16.unwrap(), x, y, m, k, batch),
            other => unreachable!("已校验: {other}"),
        }
    };

    for _ in 0..warmup {
        call(b)?;
    }
    // 预热后同步一次，避免把首次编译/首次 launch 的开销计入。
    let _ = b.download(y)?;

    // repeats 轮，每轮 iters 次调用 + 一次同步；取最小值（最不易被外部占卡污染）。
    let mut best = f64::INFINITY;
    let mut worst = 0.0f64;
    for _ in 0..repeats {
        let t0 = Instant::now();
        for _ in 0..iters {
            call(b)?;
        }
        let _ = b.download(y)?; // 每轮唯一一次同步
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        best = best.min(ms);
        worst = worst.max(ms);
    }
    let jitter = if best > 0.0 {
        (worst - best) / best
    } else {
        0.0
    };

    // 释放临时张量（避免注册表膨胀）。
    let mut all: Vec<TensorId> = vec![x, g, y];
    if let Some(h) = a8 {
        all.push(h.idx);
        all.push(h.sz);
    }
    if let Some(t) = w16 {
        all.push(t);
    }
    for t in all {
        b.free_tensor(t);
    }
    Ok((best, jitter))
}
