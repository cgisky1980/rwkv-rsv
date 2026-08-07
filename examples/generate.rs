//! 自回归生成示例：先 prefill prompt，再用 GPU self-loop 连续采样生成 token。
//!
//! 用法：
//!   cargo run --release --example generate
//!   MODEL_PATH=... cargo run --release --example generate
//!   NTOKENS=64 TEMP=0.8 TOPK=50 TOPP=0.9 GEN_MODE=argmax|sample \
//!     cargo run --release --example generate
//!
//! 演示两种 GPU 生成路径（都在单次 submit 内完成，不逐 token 下载 logits）：
//!   - argmax：每轮取 logits 最大索引（确定性，无参数）
//!   - sample：temperature / top-k / top-p / 惩罚 采样（随机，seed 递增）
//!
//! prompt 默认用 " Eiffel"（tokens [304, 25740, 109]）。若设置了 VOCAB_JSON
//! （web-rwkv 词表 JSON），可打印生成文本。

use std::error::Error;

use rwkv_rsv::gpu_model::{ModelBuilder, SamplerParams};

fn main() -> Result<(), Box<dyn Error>> {
    init_log();

    let model_path = std::env::var("MODEL_PATH")
        .unwrap_or_else(|_| r"c:\work\niceui\rwkv-g1h-3B.st".to_string());
    let n = std::env::var("NTOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(64usize);
    let mode = std::env::var("GEN_MODE").unwrap_or_else(|_| "sample".to_string());

    // 可选：web-rwkv 词表 JSON 用于文本打印
    let vocab_json = std::env::var("VOCAB_JSON").ok();
    let tokenizer = vocab_json
        .as_deref()
        .map(|p| {
            let s = std::fs::read_to_string(p)?;
            Ok::<_, Box<dyn Error>>(rwkv_rsv::tokenizer::Tokenizer::new(&s)?)
        })
        .transpose()?;

    log::info!("loading model: {model_path}");
    let mut bundle = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded, vocab={}", bundle.info().num_vocab);

    // prompt prefill
    let prompt: Vec<u32> = vec![304, 25740, 109];
    let _ = bundle.infer_tokens(&prompt)?;
    log::info!("prompt prefill 完成：{:?}", prompt);

    // 用最后一个 prompt token 作为 self-loop 的起始 seed
    let seed = *prompt.last().unwrap();

    let (generated, elapsed) = match mode.as_str() {
        "argmax" => {
            log::info!("== 生成模式: GPU argmax self-loop, {n} tokens ==");
            let t0 = std::time::Instant::now();
            let toks = bundle.infer_argmax_selfloop(seed, n)?;
            (toks, t0.elapsed().as_secs_f64())
        }
        _ => {
            // 采样参数（可从环境变量覆盖）
            let temp = std::env::var("TEMP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.8);
            let topk = std::env::var("TOPK")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(50u32);
            let topp = std::env::var("TOPP")
                .ok()
                .and_then(|s| s.parse().ok())
                .unwrap_or(0.9f32);
            let sp = SamplerParams {
                temperature: temp,
                top_k: topk,
                top_p: topp,
                seed: 514,
                ..Default::default()
            };
            log::info!("== 生成模式: GPU sample self-loop, {n} tokens ==");
            let t0 = std::time::Instant::now();
            let toks = bundle.infer_sample_selfloop(seed, n, &sp)?;
            (toks, t0.elapsed().as_secs_f64())
        }
    };

    let tokens_per_sec = n as f64 / elapsed;
    log::info!("生成完成：{n} tokens 用时 {elapsed:.3}s ({tokens_per_sec:.1} tok/s)");

    // 汇总 prompt + 生成序列
    let mut full = prompt.clone();
    full.extend_from_slice(&generated);
    log::info!("生成 token 序列 ({} 个): {:?}", full.len(), full);

    if let Some(tok) = &tokenizer {
        let bytes = tok.decode(&full)?;
        let text = String::from_utf8_lossy(&bytes);
        log::info!("== 生成文本 ==");
        log::info!("{}", text);
    } else {
        log::info!("(未设置 VOCAB_JSON，跳过文本打印)");
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
