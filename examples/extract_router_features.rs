//! 智能路由分类头训练特征提取工具（常驻训练管线，非临时测试）。
//!
//! 读取 golden 标注集（`{"text","tier"}` JSONL），对每条文本用当前引擎提取
//! mean-pooled 最后一层 hidden（state embedding），增量写出 features.jsonl
//! （每行 `{"idx","tier","hidden":[...]}`，f32 最短往返序列化）。
//!
//! 用法：
//!   DATA=<golden.jsonl> OUT=<features.jsonl> cargo run --release --example extract_router_features
//!   # 批量模式（B=16，信天翁 rows 并发 prefill + segmean，2-3× 提速）：
//!   BATCH=16 DATA=... OUT=... cargo run --release --example extract_router_features
//!
//! 支持断点续跑：启动时扫描输出文件已有 idx 并跳过。中途终止后重跑即可。
//! 训练脚本见 `client/scripts/router_head/train_mlp.py`。

use std::collections::HashSet;
use std::error::Error;
use std::io::{BufRead, BufWriter, Write};
use std::time::Instant;

use rwkv_rsv::gpu_model::{Bundle, ModelBuilder};
use rwkv_rsv::tokenizer::Tokenizer;
use serde::{Deserialize, Serialize};

const DEFAULT_DATA: &str =
    r"C:\work\ai00-x-dev\client\scripts\router_head\data\golden_balanced.jsonl";
const DEFAULT_OUT: &str = r"C:\work\ai00-x-dev\client\scripts\router_head\data\features.jsonl";
const DEFAULT_MODEL: &str =
    r"C:\work\ai00-x-dev\client\target\release\models\rwkv\rwkv7-g1i-2.9b.int8.st";
const DEFAULT_VOCAB: &str = r"C:\work\ai00-x-dev\client\target\release\models\rwkv\vocab.json";

/// 截断长度：与线上 `classify_with_engine` 保持一致（默认 256 = 会话摘要
/// ~128 + 当前请求 ~128 的双段预算；可用环境变量 MAX_TOKENS 覆盖）。
const DEFAULT_MAX_TOKENS: usize = 256;
const PAD_TOKEN: u32 = 0;

#[derive(Deserialize)]
struct GoldenRow {
    text: String,
    tier: u8,
    /// v4：上一轮路由等级（0-3；None = 首轮/未知——golden 单轮数据无此字段）。
    #[serde(default)]
    prev_tier: Option<u8>,
}

#[derive(Deserialize)]
struct FeatureRow {
    idx: usize,
}

/// 用 struct 直接序列化：f32 走最短往返表示（~9 位），
/// 比 json! 宏经 f64 中转（17 位）小约 4 倍。
#[derive(Serialize)]
struct FeatureOut {
    idx: usize,
    tier: u8,
    #[serde(skip_serializing_if = "Option::is_none")]
    prev_tier: Option<u8>,
    hidden: Vec<f32>,
}

fn main() -> Result<(), Box<dyn Error>> {
    init_log();

    let data_path = std::env::var("DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let out_path = std::env::var("OUT").unwrap_or_else(|_| DEFAULT_OUT.to_string());
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let vocab_path = std::env::var("VOCAB_JSON").unwrap_or_else(|_| DEFAULT_VOCAB.to_string());
    let max_tokens: usize = std::env::var("MAX_TOKENS")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(DEFAULT_MAX_TOKENS);

    // 断点续跑：收集已完成 idx。
    let mut done: HashSet<usize> = HashSet::new();
    if std::path::Path::new(&out_path).exists() {
        let f = std::fs::File::open(&out_path)?;
        for line in std::io::BufReader::new(f).lines() {
            let line = line?;
            if line.trim().is_empty() {
                continue;
            }
            if let Ok(row) = serde_json::from_str::<FeatureRow>(&line) {
                done.insert(row.idx);
            }
        }
        log::info!("resume: {} samples already extracted", done.len());
    }

    // 加载模型与分词器。
    log::info!("loading model: {model_path}");
    let t0 = Instant::now();
    let Bundle { mut model, state } = ModelBuilder::new(&model_path).build()?;
    log::info!("model loaded in {:.1}s", t0.elapsed().as_secs_f32());

    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tokenizer = Tokenizer::new(&vocab)?;

    // batch 并发度：BATCH>=2 启用批量 mean-hidden（信天翁 rows 并发 prefill +
    // segmean）；缺省/1 走单序列。
    let batch_size: usize = std::env::var("BATCH")
        .ok()
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(0);

    let mut classify_state = model.create_state()?;
    let initial_state = model.state_back(&state)?;
    let emb_dim = model.info().num_emb;
    log::info!(
        "hidden dim = {emb_dim}, mode = {}",
        if batch_size >= 2 {
            format!("batch(B={batch_size})")
        } else {
            "single".to_string()
        }
    );

    // 追加打开输出。
    let out_file = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)?;
    let mut writer = BufWriter::new(out_file);

    // 预读全部待处理行（批量模式需成组；行数 ~3 万内存无压力）。
    let t0 = Instant::now();
    let mut pending: Vec<(usize, GoldenRow, Vec<u32>)> = Vec::new();
    let mut skipped: usize = 0;
    for (idx, line) in std::io::BufReader::new(std::fs::File::open(&data_path)?)
        .lines()
        .enumerate()
    {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        if done.contains(&idx) {
            skipped += 1;
            continue;
        }
        let row: GoldenRow = match serde_json::from_str(&line) {
            Ok(r) => r,
            Err(e) => {
                log::warn!("skip malformed line {idx}: {e}");
                continue;
            }
        };
        let mut tokens = tokenizer.encode(row.text.as_bytes())?;
        if tokens.is_empty() {
            tokens.push(PAD_TOKEN);
        }
        tokens.truncate(max_tokens);
        pending.push((idx, row, tokens));
    }
    let total = pending.len();
    log::info!("{total} samples to extract ({skipped} already done)");

    let mut processed: usize = 0;
    let mut last_logged = 0usize;
    macro_rules! log_progress {
        () => {
            if processed / 100 > last_logged / 100 {
                writer.flush().ok();
                let elapsed = t0.elapsed().as_secs_f32();
                let throughput = processed as f32 / elapsed.max(0.001);
                let remaining = (total - processed) as f32 / throughput;
                log::info!(
                    "progress: {}/{} ({:.1}/s, ETA {:.0}m)",
                    processed,
                    total,
                    throughput,
                    remaining / 60.0
                );
                last_logged = processed;
            }
        };
    }

    if batch_size >= 2 {
        // 批量模式：pad_to=Some(max_tokens) 固定 bt = B*max_tokens，seq 缓冲只建
        // 一次（避免逐 chunk 重建 ~GB 级缓冲）；lens 按原始长度计算（pad 行不计入
        // segmean 均值——曾因预 pad 导致 lens=256，均值混入 pad 垃圾，已修）。
        // 尾 chunk 用 PAD 哑行补齐（结果被 zip 丢弃）。
        let mut batch_state = model.create_batch_state(batch_size)?;
        for chunk in pending.chunks(batch_size) {
            let mut tokens_batch: Vec<Vec<u32>> = chunk.iter().map(|(_, _, t)| t.clone()).collect();
            while tokens_batch.len() < batch_size {
                tokens_batch.push(vec![PAD_TOKEN]);
            }
            let hiddens = model.forward_seq_batch_mean_hidden(
                &mut batch_state,
                &tokens_batch,
                Some(max_tokens),
            )?;
            for ((idx, row, _), hidden) in chunk.iter().zip(hiddens) {
                let feature = FeatureOut {
                    idx: *idx,
                    tier: row.tier,
                    prev_tier: row.prev_tier,
                    hidden,
                };
                serde_json::to_writer(&mut writer, &feature)?;
                writer.write_all(b"\n")?;
            }
            processed += chunk.len();
            log_progress!();
        }
    } else {
        // 单序列模式（缺省）：与线上 classify_with_engine 同路径。
        for (idx, row, tokens) in &pending {
            model.state_load(&classify_state, &initial_state)?;
            let hidden = model.forward_seq_mean_hidden(&mut classify_state, tokens)?;
            let feature = FeatureOut {
                idx: *idx,
                tier: row.tier,
                prev_tier: row.prev_tier,
                hidden,
            };
            serde_json::to_writer(&mut writer, &feature)?;
            writer.write_all(b"\n")?;
            processed += 1;
            log_progress!();
        }
    }

    writer.flush()?;
    log::info!("done: {processed} extracted ({skipped} skipped), output: {out_path}");
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
