//! 摘要提示词/参数调试工具（训练-部署保真验证）。
//!
//! 用与线上 summary.rs 完全一致的 continuation prompt + 采样参数
//! （temperature 0.3 / top_p 0.9 / 150 token）生成摘要，人工检查质量：
//! 是否回声、是否要点式、是否丢失任务上下文、中英文行为差异。
//!
//! 用法（PROMPTS 环境变量指向 JSONL，每行 {"prev","user","assistant"}）：
//!   PROMPTS=prompts.jsonl cargo run --release --example summarize_probe
//!
//! 无 PROMPTS 时跑内置的 6 组典型场景。

use std::error::Error;

use rwkv_rsv::gpu_model::{ModelBuilder, SamplerParams};

/// 与 client/src/crates/core/src/agent/routing/summary.rs 保持一致的参数。
/// （采样参数已定稿为确定性解码，见 main 里的 SamplerParams。）
const SUMMARY_MAX_TOKENS: usize = 150;
const SUMMARY_MAX_WORDS: usize = 120;
const ASSISTANT_HEAD_CHARS: usize = 500;

/// 官方 clean_txt：CRLF 归一 + 连续换行压缩为单个（RWKV 对回车/换行敏感，
/// 见 RWKV7-G1x-templates.txt）。所有进入 prompt 的文本必须先过这里。
fn clean_txt(txt: &str) -> String {
    let normalized = txt.replace("\r\n", "\n").replace('\r', "\n");
    // 连续 2+ 换行压缩为 1（官方 clean_txt 语义；段间分隔由模板另行添加 \n\n）。
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

/// 与 summary.rs 对齐的官方 RWKV7-G1 格式（材料在前 + Question/Answer）。
/// 字节级对齐官方模板："材料文本\n\nQuestion: 问题?\nAnswer:"——
/// 段间双换行、Question 与 Answer 各占一行、冒号后单空格、结尾无尾随空格。
fn build_summary_prompt(prev: Option<&str>, last_user: &str, last_assistant: &str) -> String {
    let prev = prev
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

/// 与 summary.rs 一致的清洗 + 长尾重复退化兜底：确定性解码的开头一两句
/// 最准，30 token 后可能陷入 "port 5, port 5" 式循环——按句截断到前两句。
fn clean_summary(raw: &str) -> Option<String> {
    let mut text = raw.trim().to_string();
    for marker in ["Updated summary:", "Summary:", "Current summary:"] {
        if let Some(rest) = text.strip_prefix(marker) {
            text = rest.trim().to_string();
        }
    }
    // 首个双换行截断（回声续写跟随其后）。
    if let Some(pos) = text.find("\n\n") {
        text.truncate(pos);
    }
    // 重复 n-gram 检测（宽匹配）：滑窗内任意 3-gram 在后文再次出现即视为退化，
    // 截断到首次出现处（覆盖 "the crate may be changed" 变体循环）。
    let words: Vec<&str> = text.split_whitespace().collect();
    let mut cut = words.len();
    if words.len() >= 12 {
        for n in [3usize] {
            for i in n..words.len().saturating_sub(n) {
                let gram = &words[i - n..i];
                // 在剩余窗口内找该 gram 的第二次出现。
                if words[i + n..].windows(n).any(|w| w == gram) {
                    cut = (i - 1).max(1);
                    break;
                }
            }
            if cut < words.len() {
                break;
            }
        }
    }
    let mut text: String = words[..cut.min(words.len())].join(" ");
    // 句级截断：首句必须保留（任务主题最准）；第二句仅在 ≤25 词且完整时保留。
    let sentence_ends: Vec<usize> = text
        .char_indices()
        .filter(|(_, c)| matches!(c, '.' | '。' | '!' | '？' | '?'))
        .map(|(i, _)| i)
        .collect();
    if let Some(&first_end) = sentence_ends.first() {
        let keep_to = match sentence_ends.get(1) {
            Some(&second_end) if second_end - first_end <= 30 * 2 && second_end < text.len() => {
                second_end + 1
            }
            _ => first_end + 1,
        };
        text.truncate(keep_to);
    }
    // 词数预算。
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

/// 内置典型场景（多轮递进：同一会话连续两轮更新，验证滚动摘要行为）。
const BUILTIN_SCENARIOS: &[(&str, &str, &str, &str)] = &[
    // (label, prev_summary, last_user, last_assistant)
    (
        "zh-debug-first",
        "",
        "帮我调试这段 Rust 代码，总是在 aarch64 上崩溃",
        "看起来像是 FFI 结构体布局不匹配导致的。llama.cpp 子模块从 f6f12e43f 升级到 b10369 后，llama_context_params 新增了 n_outputs_max_per_seq 字段。你可以先对比一下 llama.h 的字段定义，把 FFI 侧的结构体补齐。",
    ),
    (
        "zh-debug-second",
        "Debugging a Rust FFI crash on aarch64. Suspect struct layout mismatch after llama.cpp upgrade; need to compare llama.h field definitions and fix the FFI structs.",
        "对齐了字段还是崩",
        "如果字段对齐后仍然崩溃，请提供完整的崩溃堆栈。另外注意 c_char 的类型问题：Linux aarch64 上 c_char 是 u8 而不是 i8，如果你的 FFI 声明用了 i8，在那个平台会静默破坏布局。int8_t*（如 llama_batch.logits）保持 i8 不变。",
    ),
    (
        "en-translation-first",
        "",
        "Translate this to English: 这个分类器的准确率达到了 98.3%",
        "The classifier achieved an accuracy of 98.3%.",
    ),
    (
        "en-translation-second",
        "Translating Chinese sentences to English. Last: accuracy figures.",
        "再翻译一句：训练数据集包含 16,751 条真实请求",
        "The training dataset contains 16,751 real requests.",
    ),
    (
        "zh-chitchat",
        "",
        "今天上海天气怎么样",
        "今天上海多云转晴，气温 22-28 度，空气质量良好，适合户外活动。",
    ),
    (
        "en-code-first",
        "",
        "Write a Rust function to check if a string is a palindrome",
        "Here's a simple implementation:\n```rust\nfn is_palindrome(s: &str) -> bool {\n    let clean: Vec<char> = s.chars().filter(|c| c.is_alphanumeric()).collect();\n    clean.iter().eq(clean.iter().rev())\n}\n```\nThis ignores case and non-alphanumeric characters.",
    ),
];

#[derive(serde::Deserialize)]
struct PromptRow {
    prev: String,
    user: String,
    assistant: String,
}

fn main() -> Result<(), Box<dyn Error>> {
    init_log();

    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        r"C:\work\ai00-x-dev\client\target\release\models\rwkv\rwkv7-g1i-2.9b.int8.st".to_string()
    });
    let vocab_path = std::env::var("VOCAB_JSON").unwrap_or_else(|_| {
        r"C:\work\ai00-x-dev\client\target\release\models\rwkv\vocab.json".to_string()
    });

    log::info!("loading model: {model_path}");
    let mut bundle = ModelBuilder::new(&model_path).build()?;
    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tokenizer = rwkv_rsv::tokenizer::Tokenizer::new(&vocab)?;

    let scenarios: Vec<(String, Option<String>, String, String)> = match std::env::var("PROMPTS") {
        Ok(path) => {
            let text = std::fs::read_to_string(&path)?;
            text.lines()
                .filter(|l| !l.trim().is_empty())
                .map(|l| {
                    let row: PromptRow = serde_json::from_str(l)?;
                    Ok((
                        format!("custom-{}", l.len()),
                        Some(row.prev),
                        row.user,
                        row.assistant,
                    ))
                })
                .collect::<Result<Vec<_>, Box<dyn Error>>>()?
        }
        Err(_) => BUILTIN_SCENARIOS
            .iter()
            .map(|(label, prev, user, assistant)| {
                (
                    label.to_string(),
                    if prev.is_empty() {
                        None
                    } else {
                        Some(prev.to_string())
                    },
                    user.to_string(),
                    assistant.to_string(),
                )
            })
            .collect(),
    };

    // 采样参数（实验定稿）：全 0 确定性解码。实测温度 0.3+惩罚 1.3 会引入
    // 幻觉（乱码数字串/主题漂移），而全 0 的开头一两句最准；长尾重复退化由
    // clean_summary 的首句截断兜底（摘要只需任务主题，无需完整段落）。
    let sp = SamplerParams {
        temperature: 0.0001,
        top_k: 50,
        top_p: 1.0,
        seed: 42,
        ..Default::default()
    };

    for (label, prev, user, assistant) in &scenarios {
        let prompt = build_summary_prompt(prev.as_deref(), user, assistant);
        // 独立 State + seq 并行 prefill（避免 reset 触发 kernel 重编译，
        // 且 seq 路径比逐 token forward 快得多）。
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
        // 生成文本 = prompt 之后的部分。
        let gen_text: String = text.chars().skip(prompt.chars().count()).collect();
        let cleaned = clean_summary(&gen_text);

        println!("\n===== {} =====", label);
        println!("[raw] {}", gen_text.replace('\n', "\\n"));
        match cleaned {
            Some(s) => println!(
                "[cleaned] {} ({}/{} words)",
                s,
                s.split_whitespace().count(),
                SUMMARY_MAX_WORDS
            ),
            None => println!("[cleaned] <REJECTED>"),
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
