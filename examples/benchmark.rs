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
