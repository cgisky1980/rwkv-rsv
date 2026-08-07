//! State 序列化示例：演示 web-rwkv 风格 `State` 的「前进 → 取态 → 存盘 → 回灌」闭环。
//!
//! 用法：
//!   cargo run --release --example state_persist
//!   MODEL_PATH=... OUT=state.bin cargo run --release --example state_persist
//!
//! 流程：
//!   1. 用 prompt 前向推理，得到一段「会话状态」；
//!   2. `state_back()` 把整态下载为连续 f32，写入文件；
//!   3. 重置状态，`state_load()` 从文件回灌；
//!   4. 再次 `state_back()`，与第 2 步的原始态逐位对比，
//!      证明序列化（下载 → 存盘 → 回灌）是**无损**的（max_diff == 0）。
//!
//! 注意：本示例只验证序列化本身无损。若想验证「回灌后继续前向」的 logits 一致性，
//! 需先保证前向是确定性的（GPU 前向存在非确定性，见库内诊断），否则 logits 对比会被
//! 前向本身的波动污染。

use std::error::Error;

use rwkv_rsv::gpu_model::ModelBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    init_log();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.st".to_string());
    let out_path = std::env::var("OUT").unwrap_or_else(|_| "state.bin".to_string());

    let mut bundle = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded, vocab={}", bundle.info().num_vocab);

    // 1) 用 prompt 前向，得到一段会话态
    let prompt: Vec<u32> = vec![304, 25740, 109];
    let _ = bundle.infer_tokens(&prompt)?;
    log::info!("prompt 前向完成");

    // 2) 取态并写盘
    let state = bundle.state_back()?;
    log::info!(
        "state_back 得到 {} f32 = {} bytes，写入 {}",
        state.len(),
        state.len() * 4,
        out_path
    );
    std::fs::write(&out_path, bytemuck::cast_slice(&state))?;

    // 3) 重置状态，再从文件回灌
    bundle.reset()?;
    let bytes = std::fs::read(&out_path)?;
    let loaded: &[f32] = bytemuck::cast_slice(&bytes);
    bundle.state_load(loaded)?;
    log::info!("state_load 回灌完成（{} f32）", loaded.len());

    // 4) 再次取态，与原始态逐位对比，验证序列化无损
    let state2 = bundle.state_back()?;
    let max_diff = state
        .iter()
        .zip(&state2)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0f32, |m, v| m.max(v));
    log::info!("序列化往返（back→存盘→load→back）max_diff = {max_diff:.6}");

    if max_diff == 0.0 {
        log::info!("✅ State 序列化滚动条无损，闭环验证通过");
    } else {
        log::info!("⚠️ 序列化存在差异，请检查 back/load 布局");
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
