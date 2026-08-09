//! 临时剖析示例：仅测 decode（infer_tokens）路径，隔离 per-kernel 时间。
//! 用后即删（不属于长期工具）。

use std::error::Error;

use rwkv_rsv::gpu_model::{ModelBuilder, SamplerParams};

fn main() -> Result<(), Box<dyn Error>> {
    let _ = log::set_logger(&STDOUT_LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.st".to_string());
    let n: usize = std::env::var("NTOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(128);

    let mut bundle = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded, vocab={}", bundle.info().num_vocab);

    // 预热 decode（构建 kernel 缓存）
    let warm: Vec<u32> = vec![304, 25740, 109];
    let _ = bundle.infer_tokens(&warm)?;
    bundle.reset()?;

    // 清空 prof，仅测 decode
    bundle.clear_kernel_prof();
    let prompt: Vec<u32> = (0..n).map(|i| (i % 1000) as u32 + 1).collect();
    let t0 = std::time::Instant::now();
    let _ = bundle.infer_tokens(&prompt)?;
    let s = t0.elapsed().as_secs_f64();
    log::info!(
        "infer_tokens(decode {n} tok)  : {s:.3}s → {:.1} tok/s",
        n as f64 / s
    );
    bundle.dump_kernel_prof();

    // ===== prefill（forward_seq）剖析 =====
    bundle.reset()?;
    bundle.clear_kernel_prof();
    let pt: usize = std::env::var("PTOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);
    let prompt: Vec<u32> = (0..pt).map(|i| (i % 1000) as u32 + 1).collect();
    let t0 = std::time::Instant::now();
    let _ = bundle.infer_seq(&prompt)?;
    let s = t0.elapsed().as_secs_f64();
    log::info!(
        "forward_seq(prefill {pt} tok) : {s:.3}s → {:.1} tok/s",
        pt as f64 / s
    );
    bundle.dump_kernel_prof();

    // ===== argmax_selfloop 剖析（单收益，不截断：8 tok × 256 kernel = 2048 < 4096）=====
    bundle.reset()?;
    bundle.clear_kernel_prof();
    let t0 = std::time::Instant::now();
    let _ = bundle.infer_argmax_selfloop(304, 8)?;
    let s = t0.elapsed().as_secs_f64();
    log::info!(
        "argmax_selfloop(8 tok)          : {s:.3}s → {:.1} tok/s",
        8.0 / s
    );
    bundle.dump_kernel_prof();

    bundle.reset()?;
    let sp = SamplerParams {
        temperature: 0.8,
        top_k: 50,
        top_p: 0.9,
        seed: 514,
        ..Default::default()
    };
    let t0 = std::time::Instant::now();
    let _ = bundle.infer_sample_selfloop(304, n, &sp)?;
    let s = t0.elapsed().as_secs_f64();
    log::info!(
        "sample_selfloop({n} tok)      : {s:.3}s → {:.1} tok/s",
        n as f64 / s
    );

    Ok(())
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
