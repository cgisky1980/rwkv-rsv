//! 临时剖析：稳态 prefill 速度（先预热全部 prefill kernel，再测第二次）。
//! 用后即删。对比 CUDA graph 与非 graph。

use std::error::Error;

use rwkv_rsv::gpu_model::ModelBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    let _ = log::set_logger(&STDOUT_LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.int8.st".to_string());
    let pt: usize = std::env::var("PTOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(512);

    let mut bundle = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded, vocab={}", bundle.info().num_vocab);

    // 预热 decode kernel（SKIP_DECODE=1 跳过，用于隔离 decode 崩溃）
    if std::env::var("SKIP_DECODE").is_err() {
        let warm: Vec<u32> = vec![304, 25740, 109];
        let _ = bundle.infer_tokens(&warm)?;
        bundle.reset()?;
    }

    let prompt: Vec<u32> = (0..pt).map(|i| (i % 1000) as u32 + 1).collect();

    // 预热 prefill kernel（首次含 JIT 编译，不计时）
    let _ = bundle.infer_seq(&prompt)?;
    bundle.reset()?;

    // 稳态计时（第二次 prefill）
    for rep in 0..3 {
        bundle.reset()?;
        let t0 = std::time::Instant::now();
        let _ = bundle.infer_seq(&prompt)?;
        let s = t0.elapsed().as_secs_f64();
        log::info!(
            "steady prefill {pt} tok rep {rep}: {s:.4}s → {:.1} tok/s",
            pt as f64 / s
        );
    }

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
