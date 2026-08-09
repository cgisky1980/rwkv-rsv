//! 临时验证：PREFILL_GRAPH 分块推理逐块正确性。用后即删。
//! 核心问题：graph 重放是否能在跨块时正确读取上一块的 state 续写。

use std::error::Error;

use rwkv_rsv::gpu_model::ModelBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    let _ = log::set_logger(&STDOUT_LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.int8.st".to_string());
    let t: usize = std::env::var("PTOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(32);
    let nchunk: usize = std::env::var("NCHUNK")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(3);

    let mut bundle = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded, vocab={}", bundle.info().num_vocab);
    let warm: Vec<u32> = vec![304, 25740, 109];
    let _ = bundle.infer_tokens(&warm)?;
    bundle.reset()?;
    let chunk: Vec<u32> = (0..t).map(|i| (i % 1000) as u32 + 1).collect();

    // 非 graph：逐块推进 state
    unsafe {
        std::env::set_var("PREFILL_GRAPH", "0");
    }
    bundle.reset()?;
    let mut a: Vec<Vec<f32>> = Vec::new();
    for _ in 0..nchunk {
        a.push(bundle.infer_seq(&chunk)?);
    }

    // graph：逐块推进 state（首次捕获+重放，后续纯重放）
    unsafe {
        std::env::set_var("PREFILL_GRAPH", "1");
    }
    bundle.reset()?;
    let mut b: Vec<Vec<f32>> = Vec::new();
    for _ in 0..nchunk {
        b.push(bundle.infer_seq(&chunk)?);
    }

    for k in 0..nchunk {
        let mut md = 0.0f32;
        for (p, q) in a[k].iter().zip(&b[k]) {
            md = md.max((p - q).abs());
        }
        log::info!("[CMP] chunk {k}: non-graph vs graph max_abs_diff = {md:.6}");
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
