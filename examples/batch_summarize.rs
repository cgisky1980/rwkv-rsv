//! 批量生成真实摘要池：多轮对话 → 定稿摘要（部署同款管线）。
//!
//! 输入：JSONL，每行一个多轮对话场景：
//!   {"turns": [{"user": "...", "assistant": "..."}, ...], "tier": 0-3}
//! 输出：JSONL，每行 {"summary": "...", "tier": n, "turn": k}——
//!   对每个对话逐轮滚动生成摘要（与线上 SessionSummaryService 完全同构：
//!   同一 build_summary_prompt/clean_txt/clean_summary/采样参数）。
//!
//! 用法：
//!   DATA=scenarios.jsonl OUT=summaries.jsonl \
//!     cargo run --release --example batch_summarize
//!
//! 摘要供 build_dataset.py / gen_context_augment.py 消费：按 tier 分池、
//! 与 golden 请求组合成 `Summary: {real}\nRequest: {golden}` 训练样本。
//! 训练分布 = 线上分布（同一模型同一提示词生成）。

use std::collections::HashSet;
use std::error::Error;
use std::io::{BufRead, BufWriter, Write};

use rwkv_rsv::gpu_model::{ModelBuilder, SamplerParams};

const DEFAULT_MODEL: &str =
    r"C:\work\ai00-x-dev\client\target\release\models\rwkv\rwkv7-g1i-2.9b.int8.st";
const DEFAULT_VOCAB: &str = r"C:\work\ai00-x-dev\client\target\release\models\rwkv\vocab.json";
const DEFAULT_DATA: &str =
    r"C:\work\ai00-x-dev\client\scripts\router_head\data\multiturn\scenarios.jsonl";
const DEFAULT_OUT: &str =
    r"C:\work\ai00-x-dev\client\scripts\router_head\data\multiturn\summaries.jsonl";

// ==== 与部署端 client/src/crates/core/src/agent/routing/summary.rs 逐字一致（冻结） ====

const SUMMARY_MAX_TOKENS: usize = 96;
const SUMMARY_MAX_WORDS: usize = 120;
const ASSISTANT_HEAD_CHARS: usize = 500;

fn clean_txt(txt: &str) -> String {
    let normalized = txt.replace("\r\n", "\n").replace('\r', "\n");
    let mut out = String::with_capacity(normalized.len());
    let mut last_nl = false;
    for ch in normalized.chars() {
        if ch == '\n' {
            if !last_nl {
                out.push('\n');
            }
            last_nl = true;
        } else {
            out.push(ch);
            last_nl = false;
        }
    }
    out.trim().to_string()
}

fn build_summary_prompt(
    prev_summary: Option<&str>,
    last_user: &str,
    last_assistant: &str,
) -> String {
    let prev = prev_summary
        .map(clean_txt)
        .filter(|s| !s.is_empty())
        .unwrap_or_else(|| "(none)".to_string());
    let user = clean_txt(last_user);
    let assistant_head: String = clean_txt(last_assistant)
        .chars()
        .take(ASSISTANT_HEAD_CHARS)
        .collect();
    format!(
        "Conversation so far:\nSummary: {prev}\nUser: {user}\nAssistant: {assistant_head}\n\nQuestion: Summarize the conversation. What task is in progress?\nAnswer:",
    )
}

fn clean_summary(raw: &str) -> Option<String> {
    let mut text = raw.trim().to_string();
    for marker in ["Answer:", "Summary:", "Question:"] {
        if let Some(rest) = text.strip_prefix(marker) {
            text = rest.trim().to_string();
        }
    }
    if let Some(pos) = text.find("\n\n") {
        text.truncate(pos);
    }
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut cut = words.len();
    if words.len() >= 12 {
        'outer: for i in 3..words.len().saturating_sub(3) {
            let gram = &words[i - 3..i];
            if words[i + 3..].windows(3).any(|w| w == gram) {
                cut = (i - 1).max(1);
                break 'outer;
            }
        }
    }
    let mut text: String = words[..cut.min(words.len())].join(" ");
    // 句级截断（字符边界安全：多字节句号不能按字节 +1 截）。
    let sentence_ends: Vec<usize> = text
        .char_indices()
        .filter(|(_, c)| matches!(c, '.' | '。' | '!' | '？' | '?'))
        .map(|(i, c)| i + c.len_utf8())
        .collect();
    if let Some(&first_end) = sentence_ends.first() {
        let keep_to = match sentence_ends.get(1) {
            Some(&second_end) if second_end - first_end <= 60 && second_end < text.len() => {
                second_end
            }
            _ => first_end,
        };
        text.truncate(keep_to);
    }
    let mut words = 0usize;
    let mut out = String::new();
    for ch in text.chars() {
        if ch.is_whitespace() || ch > '\u{2E80}' {
            words += 1;
        }
        if words > SUMMARY_MAX_WORDS {
            break;
        }
        out.push(ch);
    }
    let cleaned = out.trim().to_string();
    if cleaned.is_empty() {
        None
    } else {
        Some(cleaned)
    }
}

// ==== 输入格式 ====

#[derive(serde::Deserialize)]
struct Turn {
    user: String,
    assistant: String,
}

#[derive(serde::Deserialize)]
struct Scenario {
    turns: Vec<Turn>,
    tier: u8,
}

#[derive(serde::Serialize, serde::Deserialize)]
struct SummaryOut {
    summary: String,
    tier: u8,
    turn: usize,
    scenario: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    init_log();

    let data_path = std::env::var("DATA").unwrap_or_else(|_| DEFAULT_DATA.to_string());
    let out_path = std::env::var("OUT").unwrap_or_else(|_| DEFAULT_OUT.to_string());
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let vocab_path = std::env::var("VOCAB_JSON").unwrap_or_else(|_| DEFAULT_VOCAB.to_string());

    // 断点续跑。
    let mut done: HashSet<String> = HashSet::new();
    if std::path::Path::new(&out_path).exists() {
        let f = std::fs::File::open(&out_path)?;
        for l in std::io::BufReader::new(f).lines().map_while(Result::ok) {
            if let Ok(row) = serde_json::from_str::<SummaryOut>(&l) {
                done.insert(format!("{}:{}", row.scenario, row.turn));
            }
        }
        log::info!("resume: {} summaries already generated", done.len());
    }

    log::info!("loading model: {model_path}");
    let mut bundle = ModelBuilder::new(&model_path).build()?;
    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tokenizer = rwkv_rsv::tokenizer::Tokenizer::new(&vocab)?;

    // 定稿采样参数：确定性解码（temperature≈0 / top_p 禁用 / top_k 50 走快速路径）。
    let sp = SamplerParams {
        temperature: 0.0001,
        top_k: 50,
        top_p: 1.0,
        seed: 42,
        ..Default::default()
    };

    let f = std::fs::File::open(&data_path)?;
    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)?;
    let mut writer = BufWriter::new(out);

    let mut n_ok = 0usize;
    let mut n_reject = 0usize;
    for (si, line) in std::io::BufReader::new(f).lines().enumerate() {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        let sc: Scenario = match serde_json::from_str(&line) {
            Ok(s) => s,
            Err(e) => {
                log::warn!("skip malformed scenario {si}: {e}");
                continue;
            }
        };
        // 滚动摘要：与线上 spawn_update 相同的逐轮链。
        let mut summary: Option<String> = None;
        for (ti, turn) in sc.turns.iter().enumerate() {
            let key = format!("{si}:{ti}");
            if done.contains(&key) {
                continue;
            }
            let prompt = build_summary_prompt(summary.as_deref(), &turn.user, &turn.assistant);
            let mut state = bundle.model.create_state()?;
            let tokens = tokenizer.encode(prompt.as_bytes())?;
            let _ = bundle.model.forward_seq_with_state(&mut state, &tokens)?;
            let seed = *tokens.last().unwrap();
            let generated = bundle.model.forward_sample_selfloop_with_state(
                &mut state,
                seed,
                SUMMARY_MAX_TOKENS,
                &sp,
            )?;
            let mut full = tokens.clone();
            full.extend_from_slice(&generated);
            let bytes = tokenizer.decode(&full)?;
            let text = String::from_utf8_lossy(&bytes);
            let gen_text: String = text.chars().skip(prompt.chars().count()).collect();

            match clean_summary(&gen_text) {
                Some(s) => {
                    summary = Some(s.clone());
                    let row = SummaryOut {
                        summary: s,
                        tier: sc.tier,
                        turn: ti,
                        scenario: si,
                    };
                    serde_json::to_writer(&mut writer, &row)?;
                    writer.write_all(b"\n")?;
                    n_ok += 1;
                }
                None => {
                    n_reject += 1;
                    log::warn!("summary rejected (scenario {si} turn {ti})");
                }
            }
            if (ti + 1) % 5 == 0 {
                writer.flush()?;
            }
        }
    }
    writer.flush()?;
    log::info!("done: {n_ok} summaries written, {n_reject} rejected -> {out_path}");
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
