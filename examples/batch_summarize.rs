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
//! 并发（多进程分片）：CUDA_CTX_LOCK 是进程级全局锁（backend 存活期间持有），
//! 同进程多模型实例会死锁；改用多进程——把场景文件切成 N 片，每片一个进程：
//!   SCENARIO_OFFSET=<该片首行在原文件中的行号> 使输出的 scenario 字段为全局
//!   索引，各分片输出可直接拼接（resume 的 done 集合同样按全局索引判重）。
//! 摘要供 build_dataset.py / gen_context_augment.py 消费：按 tier 分池、
//! 与 golden 请求组合成 `Summary: {real}\nRequest: {golden}` 训练样本。
//! 训练分布 = 线上分布（同一模型同一提示词生成）。

use std::collections::{HashMap, VecDeque};
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

#[derive(serde::Deserialize, Clone)]
struct Turn {
    user: String,
    assistant: String,
}

#[derive(serde::Deserialize, Clone)]
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
    // 并发度：**单模型实例 + batch 维并发**（B 个 slot 共享一份权重，一次读权重
    // 算 B 份——weight-bound 场景吞吐 ≈ ×B；多实例方案是 B 份权重带宽，实测无增益）。
    // prefill 走单序列 seq 路径，完成后状态经 host 中转灌入 batch State 对应 slot；
    // decode 走 batch selfloop（graph 捕获一轮 batch 前向 + 逐轮重放，GPU 内 token
    // 自回写）。仅 CUDA 后端。
    let batch: usize = std::env::var("BATCH")
        .ok()
        .or_else(|| std::env::var("CONCURRENCY").ok())
        .and_then(|v| v.parse().ok())
        .filter(|n| *n >= 1)
        .unwrap_or(8);

    // 预读全部场景行（跳过空行；scenario id = 原文件行号）。
    let mut scenarios: Vec<Scenario> = Vec::new();
    for line in std::io::BufRead::lines(std::io::BufReader::new(std::fs::File::open(&data_path)?)) {
        let line = line?;
        if line.trim().is_empty() {
            continue;
        }
        match serde_json::from_str(&line) {
            Ok(sc) => scenarios.push(sc),
            Err(e) => log::warn!("skip malformed scenario: {e}"),
        }
    }
    log::info!("loaded {} scenarios from {data_path}", scenarios.len());

    // 断点续跑：已完成 (scenario, turn) → 摘要（滚动链回读，恢复最近摘要）。
    let mut done: HashMap<(usize, usize), String> = HashMap::new();
    if std::path::Path::new(&out_path).exists() {
        let f = std::fs::File::open(&out_path)?;
        for l in std::io::BufReader::new(f).lines().map_while(Result::ok) {
            if let Ok(row) = serde_json::from_str::<SummaryOut>(&l) {
                done.insert((row.scenario, row.turn), row.summary);
            }
        }
        log::info!("resume: {} summaries already generated", done.len());
    }

    // 待处理队列：场景链（scenario, next_turn, rolling_summary）。
    let mut queue: VecDeque<(usize, usize, Option<String>)> = VecDeque::new();
    for (gid, sc) in scenarios.iter().enumerate() {
        let mut next = 0usize;
        let mut summary: Option<String> = None;
        for (ti, _) in sc.turns.iter().enumerate() {
            if let Some(s) = done.get(&(gid, ti)) {
                next = ti + 1;
                summary = Some(s.clone());
            }
        }
        if next < sc.turns.len() {
            queue.push_back((gid, next, summary));
        }
    }
    let total_turns: usize = queue
        .iter()
        .map(|(gid, t, _)| scenarios[*gid].turns.len() - t)
        .sum();
    log::info!(
        "{total_turns} turns to generate across {} scenarios",
        queue.len()
    );

    // 单模型实例：权重只载一份。
    log::info!("loading model: {model_path}");
    let bundle = ModelBuilder::new(&model_path).build()?;
    let mut model = bundle.model;

    // 单序列 State（prefill 用）+ batch State（decode 用）+ 零态快照。
    let mut single_state = model.create_state()?;
    let initial_state = model.state_back(&single_state)?;
    let mut batch_state = model.create_batch_state(batch)?;

    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tokenizer = rwkv_rsv::tokenizer::Tokenizer::new(&vocab)?;

    // 定稿采样参数：确定性解码（temperature≈0 / top_k 50 快速路径）。
    let sp = SamplerParams {
        temperature: 0.0001,
        top_k: 50,
        top_p: 1.0,
        seed: 42,
        ..Default::default()
    };

    let out = std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&out_path)?;
    let mut writer = BufWriter::new(out);

    // 批内任务描述（slot → 任务）。
    struct SlotJob {
        scenario: usize,
        turn: usize,
        tier: u8,
        prompt_tokens: Vec<u32>,
        prompt_chars: usize,
        /// None = 空 slot（尾批不足，灌零态空转，产出丢弃）。
        active: bool,
    }

    let mut n_ok = 0usize;
    let mut n_reject = 0usize;
    let t_start = std::time::Instant::now();
    let mut t_prefill = std::time::Duration::ZERO;
    let mut t_decode = std::time::Duration::ZERO;

    while !queue.is_empty() {
        // 1. 填满 B 个 slot：队首任务 → **batch prefill**（B 个 prompt 一次贯穿
        // 全部层，GEMM M 维 = B*T_pad，直接更新 batch State——信天翁 rows 模型；
        // 此前逐 slot 串行 prefill 占全量 20%）。PAD_MODE=1 调试对照：逐 slot
        // 走旧单序列 prefill + 状态灌 slot。
        let t_pf = std::time::Instant::now();
        let mut jobs: Vec<SlotJob> = Vec::with_capacity(batch);
        let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(batch);
        let mut prompt_chars_list: Vec<usize> = Vec::with_capacity(batch);
        for _slot in 0..batch {
            let Some((gid, ti, summary)) = queue.pop_front() else {
                // 队列空：占位空 slot（pad token 0 空转，产出丢弃）。
                jobs.push(SlotJob {
                    scenario: 0,
                    turn: 0,
                    tier: 0,
                    prompt_tokens: Vec::new(),
                    prompt_chars: 0,
                    active: false,
                });
                prompts.push(vec![0]);
                prompt_chars_list.push(0);
                continue;
            };
            let sc = &scenarios[gid];
            let turn = &sc.turns[ti];
            let prompt = build_summary_prompt(summary.as_deref(), &turn.user, &turn.assistant);
            prompt_chars_list.push(prompt.chars().count());
            prompts.push(tokenizer.encode(prompt.as_bytes())?);
            jobs.push(SlotJob {
                scenario: gid,
                turn: ti,
                tier: sc.tier,
                prompt_tokens: Vec::new(),
                prompt_chars: 0,
                active: true,
            });
        }
        // 回填 prompt_tokens/chars 到 jobs（借用结束后）。
        for (slot, job) in jobs.iter_mut().enumerate() {
            if job.active {
                job.prompt_tokens = prompts[slot].clone();
                job.prompt_chars = prompt_chars_list[slot];
            }
        }
        // prefill：batch 一次（G1 官方模板契约：全量 prompt，selfloop 首步从重复末
        // token 开始）或逐 slot 单序列对照。
        let seeds: Vec<u32> = if std::env::var("PAD_MODE").is_ok_and(|v| !v.is_empty()) {
            let mut out = Vec::with_capacity(batch);
            for slot in 0..batch {
                if !jobs[slot].active {
                    model.state_slot_load(&batch_state, slot, &initial_state)?;
                    out.push(0);
                    continue;
                }
                model.state_load(&single_state, &initial_state)?;
                model.forward_seq_with_state(&mut single_state, &prompts[slot])?;
                let data = model.state_back(&single_state)?;
                model.state_slot_load(&batch_state, slot, &data)?;
                out.push(*prompts[slot].last().unwrap());
            }
            out
        } else {
            // 每轮 prefill 从零态开始（各 turn 是含滚动摘要的独立 prompt，
            // 与旧单序列流程 state_load(initial_state) 语义一致——漏掉会导致
            // 第 2 轮起带着上轮 decode 状态 prefill，状态污染→采样出越界 token）。
            model.reset_state_of(&batch_state)?;
            model.forward_seq_batch(&mut batch_state, &prompts)?
        };

        // 2. batch decode：B slot 并发采样 SUMMARY_MAX_TOKENS 个 token。
        t_prefill += t_pf.elapsed();
        let t_dc = std::time::Instant::now();
        // SEQ_MODE=1 调试对照：逐 slot 走旧单序列 selfloop API（隔离 batch 路径 bug）。
        // 注意：必须检查非空——PowerShell `$env:X=$null` 会设成空串而非删除，
        // `is_ok()` 对空串也返回 true，曾导致全部测试静默走单序列路径。
        let generated: Vec<Vec<u32>> = if std::env::var("SEQ_MODE").is_ok_and(|v| !v.is_empty()) {
            let mut out = Vec::with_capacity(batch);
            for slot in 0..batch {
                if !jobs[slot].active {
                    out.push(Vec::new());
                    continue;
                }
                // 单序列路径：slot 状态取回 → 单序列 State → submit 单序列 selfloop。
                let data = model.state_slot_back(&batch_state, slot)?;
                let mut st = model.create_state()?;
                model.state_load(&st, &data)?;
                let ticket =
                    model.submit_sample_selfloop(&mut st, seeds[slot], SUMMARY_MAX_TOKENS, &sp)?;
                out.push(model.collect_sample_selfloop(ticket)?);
            }
            out
        } else {
            let ticket = model.submit_sample_selfloop_batch(
                &mut batch_state,
                &seeds,
                SUMMARY_MAX_TOKENS,
                &sp,
            )?;
            model.collect_sample_selfloop_batch(ticket, batch)?
        };
        t_decode += t_dc.elapsed();

        // 3. 收割：解码 + 清洗 + 输出 + 场景链推进。
        for (slot, job) in jobs.iter().enumerate() {
            if !job.active {
                continue;
            }
            let mut full = job.prompt_tokens.clone();
            full.extend_from_slice(&generated[slot]);
            let bytes = tokenizer.decode(&full)?;
            let text = String::from_utf8_lossy(&bytes);
            let gen_text: String = text.chars().skip(job.prompt_chars).collect();

            let mut next_summary: Option<String> = None;
            match clean_summary(&gen_text) {
                Some(s) => {
                    next_summary = Some(s.clone());
                    let row = SummaryOut {
                        summary: s,
                        tier: job.tier,
                        turn: job.turn,
                        scenario: job.scenario,
                    };
                    serde_json::to_writer(&mut writer, &row)?;
                    writer.write_all(b"\n")?;
                    n_ok += 1;
                }
                None => {
                    n_reject += 1;
                    log::warn!(
                        "summary rejected (scenario {} turn {})",
                        job.scenario,
                        job.turn
                    );
                }
            }
            // 场景链推进：还有后续轮次 → 队尾续跑（滚动摘要传递）。
            // 被拒轮沿用旧摘要（与线上 spawn_update 失败保留旧行为一致）。
            let sc = &scenarios[job.scenario];
            if job.turn + 1 < sc.turns.len() {
                let summary = next_summary.or_else(|| {
                    // 拒绝时回退到该场景最近一次成功摘要（可能为 None）。
                    let mut s = None;
                    for ti in (0..job.turn).rev() {
                        if let Some(x) = done.get(&(job.scenario, ti)) {
                            s = Some(x.clone());
                            break;
                        }
                    }
                    s
                });
                queue.push_back((job.scenario, job.turn + 1, summary));
            }
        }

        if n_ok.is_multiple_of(50) && n_ok > 0 {
            writer.flush()?;
            let el = t_start.elapsed().as_secs_f32();
            log::info!(
                "progress: {n_ok} ok / {n_reject} rejected ({:.2}/s, elapsed {:.0}s)",
                n_ok as f32 / el,
                el
            );
        }
    }

    writer.flush()?;
    let el = t_start.elapsed().as_secs_f32();
    log::info!(
        "done: {n_ok} summaries written, {n_reject} rejected, batch={batch}, \
         {:.2}/s, prefill {:.1}s / decode {:.1}s -> {out_path}",
        n_ok as f32 / el.max(0.001),
        t_prefill.as_secs_f32(),
        t_decode.as_secs_f32(),
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
