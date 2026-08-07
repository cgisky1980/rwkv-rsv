//! 模型信息示例：加载模型并打印 web-rwkv 风格的 `ModelInfo`。
//!
//! 用法：
//!   cargo run --release --example model_info
//!   MODEL_PATH=path/to/model.st cargo run --release --example model_info
//!
//! 该示例演示 `ModelBuilder` + `Bundle` 的用法：构建模型、绑定状态、
//! 读取公开元信息（层数/维度/词表/头数等），并做一次 probe 推理验证可用。

use std::error::Error;

use rwkv_rsv::gpu_model::ModelBuilder;

fn main() -> Result<(), Box<dyn Error>> {
    // 简单日志（不依赖 simplelog 的复杂配置，直接打到 stdout）
    init_log();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.st".to_string());
    log::info!("loading model: {model_path}");

    // 创建 Runtime + 加载模型 + 绑定零初始 State，得到 Bundle（web-rwkv 风格）
    let mut bundle = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded");

    // 读取并打印模型元信息
    let info = bundle.info();
    log::info!("== ModelInfo ==");
    log::info!("  version    : {:?}", info.version);
    log::info!("  num_layer  : {}", info.num_layer);
    log::info!("  num_emb    : {}", info.num_emb);
    log::info!("  num_vocab  : {}", info.num_vocab);
    log::info!("  num_head   : {}", info.num_head);
    log::info!("  head_size  : {}", info.head_size);
    log::info!("  ffn_hidden : {}", info.ffn_hidden);
    log::info!(
        "  w/a/v/g mid: {}/{}/{}/{}",
        info.w_mid,
        info.a_mid,
        info.v_mid,
        info.g_mid
    );

    // probe 前向：验证模型可用（" Eiffel" → tokens [304, 25740, 109]）
    let tokens: Vec<u32> = vec![304, 25740, 109];
    let logits = bundle.infer_tokens(&tokens)?;
    log::info!("probe forward 返回 logits 长度 = {}", logits.len());

    // 打印 top-5
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
    log::info!("== top-5 logits ==");
    for (rank, (token, logit)) in indexed.iter().take(5).enumerate() {
        log::info!("  {}: token={token} logit={logit:.6}", rank + 1);
    }

    Ok(())
}

fn init_log() {
    // 避免重复初始化
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
