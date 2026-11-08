//! 推理性能基准示例：对比不同 GPU 推理路径的吞吐。
//!
//! 用法：
//!   cargo run --release --example benchmark
//!   MODEL_PATH=... NTOKENS=256 cargo run --release --example benchmark
//!
//! 覆盖路径：
//!   - infer_seq     : sequence-parallel，整段 prompt 一次贯穿各层（prefill）
//!   - infer_tokens  : 逐 token 前向（decode 单 token）
//!   - argmax_selfloop: 单次 submit 内连续 argmax 生成
//!   - sample_selfloop: 单次 submit 内连续采样生成（含 penalty/history）
//!
//! 输出每路径的 tok/s，便于对比 prefill 与 decode 的吞吐差异。

use std::error::Error;
use std::time::Instant;

use rwkv_rsv::gpu_model::{ModelBuilder, SamplerParams};

fn main() -> Result<(), Box<dyn Error>> {
    init_log();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.st".to_string());
    let n = std::env::var("NTOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128usize);

    let mut bundle = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded, vocab={}", bundle.info().num_vocab);

    // 预热：构建 kernel 缓存后再计时
    let warm: Vec<u32> = vec![304, 25740, 109];
    let _ = bundle.infer_seq(&warm)?;
    bundle.reset()?;
    let _ = bundle.infer_tokens(&warm)?;
    bundle.reset()?;
    let _ = bundle.infer_argmax_selfloop(warm[2], 8)?;
    bundle.reset()?;
    let sp = SamplerParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        seed: 514,
        ..Default::default()
    };
    let _ = bundle.infer_sample_selfloop(warm[2], 8, &sp)?;
    log::info!("预热完成\n");

    // 1) infer_seq：整段 prefill
    bundle.reset()?;
    let prompt: Vec<u32> = (0..n).map(|i| (i % 1000) as u32 + 1).collect();
    let t0 = Instant::now();
    let _ = bundle.infer_seq(&prompt)?;
    let seq_s = t0.elapsed().as_secs_f64();
    log::info!(
        "infer_seq(prefill {n} tok)      : {seq_s:.3}s → {:.1} tok/s",
        n as f64 / seq_s
    );

    // 2) infer_tokens：逐 token decode
    bundle.reset()?;
    let t0 = Instant::now();
    let _ = bundle.infer_tokens(&prompt)?;
    let tok_s = t0.elapsed().as_secs_f64();
    log::info!(
        "infer_tokens(decode {n} tok)    : {tok_s:.3}s → {:.1} tok/s",
        n as f64 / tok_s
    );

    // 3) argmax_selfloop
    bundle.reset()?;
    let t0 = Instant::now();
    let _ = bundle.infer_argmax_selfloop(prompt[0], n)?;
    let sl_s = t0.elapsed().as_secs_f64();
    log::info!(
        "argmax_selfloop({n} tok)        : {sl_s:.3}s → {:.1} tok/s",
        n as f64 / sl_s
    );

    // 4) sample_selfloop
    bundle.reset()?;
    let t0 = Instant::now();
    let _ = bundle.infer_sample_selfloop(prompt[0], n, &sp)?;
    let ss_s = t0.elapsed().as_secs_f64();
    log::info!(
        "sample_selfloop({n} tok)        : {ss_s:.3}s → {:.1} tok/s",
        n as f64 / ss_s
    );

    // 5) 单 token 解码的两条路径对照（`SINGLE_AB=1`）
    //    A：`infer_tokens([t])` —— 走 `forward_token`（SIMT GEMV 家族）
    //    B：`infer_seq([t])`    —— 走 sequence-parallel 层（张量核 GEMM 家族，t=1 时退化为窄 GEMM）
    //    两者数学等价（同一 state 语义），差值即「解码内核形态」的净开销。
    if std::env::var("SINGLE_AB").is_ok() {
        let reps = std::env::var("SINGLE_AB_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(48usize);
        for (label, seq) in [("A: forward_token", false), ("B: forward_seq", true)] {
            bundle.reset()?;
            // 预热 8 步
            for i in 0..8 {
                let tok = prompt[i % prompt.len()];
                if seq {
                    let _ = bundle.infer_seq(&[tok])?;
                } else {
                    let _ = bundle.infer_tokens(&[tok])?;
                }
            }
            let t0 = Instant::now();
            for i in 0..reps {
                let tok = prompt[(i + 8) % prompt.len()];
                if seq {
                    let _ = bundle.infer_seq(&[tok])?;
                } else {
                    let _ = bundle.infer_tokens(&[tok])?;
                }
            }
            let dt = t0.elapsed().as_secs_f64();
            log::info!(
                "单 token [{label}] : {:.2} ms/token → {:.1} tok/s",
                dt * 1e3 / reps as f64,
                reps as f64 / dt
            );
        }
    }

    // 6) 单流 vs 批量 B=1（`B1_AB=1`）：同进程、同参数、交替 4 轮取每档最小值
    //    两条路径走的是**不同的内核形态**（forward_layer vs forward_layer_batch），
    //    批宽都是 1，所以差值就是「解码内核形态」的净差。
    if std::env::var("B1_AB").is_ok() {
        let reps = std::env::var("B1_AB_REPS")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(64usize);
        let mut bstate = bundle.model.create_batch_state(1)?;
        // 三档采样参数：贪心（top_k=1）、常见采样（50/0.9）、纯 top-p（0/0.2）
        let cfgs = [
            ("top_k=1 (贪心)", 1u32, 1.0f32),
            ("top_k=50 top_p=0.9", 50, 0.9),
            ("top_k=0 top_p=0.2", 0, 0.2),
        ];
        for (label, tk, tp) in cfgs {
            let sp = SamplerParams {
                top_k: tk,
                top_p: tp,
                ..sp
            };
            let mut best_single = f64::MAX;
            let mut best_batch = f64::MAX;
            for round in 0..4 {
                let order = if round % 2 == 0 {
                    [true, false]
                } else {
                    [false, true]
                };
                for single_first in order {
                    if single_first {
                        bundle.reset()?;
                        let t0 = Instant::now();
                        let ticket = bundle.model.submit_sample_selfloop(
                            &mut bundle.state,
                            prompt[0],
                            reps,
                            &sp,
                        )?;
                        let _ = bundle.model.collect_sample_selfloop(ticket)?;
                        let dt = t0.elapsed().as_secs_f64();
                        best_single = best_single.min(dt * 1e3 / reps as f64);
                    } else {
                        bundle.model.reset_state_of(&bstate)?;
                        let t0 = Instant::now();
                        let ticket = bundle.model.submit_sample_selfloop_batch(
                            &mut bstate,
                            &[prompt[0]],
                            reps,
                            &sp,
                        )?;
                        let _ = bundle.model.collect_sample_selfloop_batch(ticket, 1)?;
                        let dt = t0.elapsed().as_secs_f64();
                        best_batch = best_batch.min(dt * 1e3 / reps as f64);
                    }
                }
            }
            log::info!(
                "单流 vs 批量 B=1 [{label}]（{reps} tok × 4 轮取最小）: 单流 {best_single:.2} ms / 批量 {best_batch:.2} ms（{:.2}×）",
                best_single / best_batch
            );
        }
        // `B1_PROF=1`：分别量两条路径的 **GPU 内核总时间/步**（PROF_CUDA_KERNEL=1 才有读数），
        // 用来区分「内核多维」与「宿主发射多」。
        if std::env::var("B1_PROF").is_ok() {
            let n_prof = 24usize;
            let sp1 = SamplerParams {
                top_k: 50,
                top_p: 0.9,
                ..sp
            };
            // 单流
            bundle.reset()?;
            let _ = bundle
                .model
                .submit_sample_selfloop(&mut bundle.state, prompt[0], 8, &sp1)?;
            bundle.clear_kernel_prof();
            let t0 = Instant::now();
            let tk =
                bundle
                    .model
                    .submit_sample_selfloop(&mut bundle.state, prompt[0], n_prof, &sp1)?;
            let _ = bundle.model.collect_sample_selfloop(tk)?;
            let wall = t0.elapsed().as_secs_f64() * 1e3 / n_prof as f64;
            log::info!("[PROF] 单流 {n_prof} 步 wall={wall:.2} ms/步，内核时间：");
            bundle.dump_kernel_prof();
            // 批量 B=1
            bundle.model.reset_state_of(&bstate)?;
            let _ =
                bundle
                    .model
                    .submit_sample_selfloop_batch(&mut bstate, &[prompt[0]], 8, &sp1)?;
            bundle.clear_kernel_prof();
            let t0 = Instant::now();
            let tk = bundle.model.submit_sample_selfloop_batch(
                &mut bstate,
                &[prompt[0]],
                n_prof,
                &sp1,
            )?;
            let _ = bundle.model.collect_sample_selfloop_batch(tk, 1)?;
            let wall = t0.elapsed().as_secs_f64() * 1e3 / n_prof as f64;
            log::info!("[PROF] 批量 B=1 {n_prof} 步 wall={wall:.2} ms/步，内核时间：");
            bundle.dump_kernel_prof();
            // 逐内核归因必须**没有 graph 捕获**（replay 只有一次 cuGraphLaunch，看不到子核）：
            // 用 `infer_tokens([t])` 走 eager 的 `forward_token`（与自循环同样的单步数学）。
            bundle.reset()?;
            for _ in 0..4 {
                let _ = bundle.infer_tokens(&[prompt[0]])?;
            }
            bundle.clear_kernel_prof();
            let t0 = Instant::now();
            for i in 0..n_prof {
                let _ = bundle.infer_tokens(&[prompt[i % prompt.len()]])?;
            }
            let wall = t0.elapsed().as_secs_f64() * 1e3 / n_prof as f64;
            log::info!("[PROF] eager forward_token {n_prof} 步 wall={wall:.2} ms/步，内核时间：");
            bundle.dump_kernel_prof();
        }
    }

    Ok(())
}

fn init_log() {
    let _ = log::set_logger(&STDOUT_LOGGER);
    log::set_max_level(log::LevelFilter::Info);
}

struct StdoutLogger;

static STDOUT_LOGGER: StdoutLogger = StdoutLogger;

impl log::Log for StdoutLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        println!(
            "[{}][{}] {}",
            chrono::Local::now().format("%H:%M:%S"),
            record.level(),
            record.args()
        );
    }
    fn flush(&self) {}
}
