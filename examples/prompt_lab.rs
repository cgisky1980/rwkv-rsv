//! 桌面 RWKV 管线复刻实验壳：与 apps/desktop/src/rwkv_llm.rs 的
//! prepare_task/advance_task/sample_token 完全一致（sanitize→User/Assistant 包裹→
//! 128 chunk prefill→CPU nucleus 采样循环→stop 序列），用于快速迭代提示词。
//!
//! 用法：
//!   PROMPT_FILE=path TEMP_DISABLE=1 cargo run --release --example prompt_lab
//!   可选：TOP_P(默认0.1) TOP_K(50) NTOKENS(350) STOP_ON_BLANK(默认1)
//!   批量（模型只加载一次，每 run 重置状态重新 prefill）：
//!   RUNS=10 OUT_PREFIX=ask OUT_DIR=test PROMPT_FILE=a.txt';b.txt ...
//!   （PROMPT_FILE 支持 ';' 分隔多文件，逐文件 × RUNS 循环）

use std::collections::HashMap;
use std::error::Error;

use rwkv_rsv::gpu_model::{Bundle, ModelBuilder};
use rwkv_rsv::tokenizer::Tokenizer;

const PREFILL_CHUNK: usize = 128;

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        r"c:\work\ai00-x-dev\client\target\release\models\rwkv\rwkv7-g1i-2.9b.int8.st".into()
    });
    let vocab_path = std::env::var("VOCAB_JSON").unwrap_or_else(|_| {
        r"c:\work\ai00-x-dev\client\target\release\models\rwkv\vocab.json".into()
    });
    let prompt_files: Vec<String> = std::env::var("PROMPT_FILE")
        .map_err(|_| "需要 PROMPT_FILE")?
        .split(';')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect();
    let runs: usize = std::env::var("RUNS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(1);
    let out_prefix = std::env::var("OUT_PREFIX").unwrap_or_else(|_| "run".into());
    let out_dir = std::env::var("OUT_DIR").ok();
    let ntokens: usize = std::env::var("NTOKENS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(350);
    let top_p: f32 = std::env::var("TOP_P")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.1);
    let top_k: usize = std::env::var("TOP_K")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    let pen: f32 = std::env::var("PEN")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0.0);
    let stop_on_blank = std::env::var("STOP_ON_BLANK")
        .ok()
        .map(|s| s != "0")
        .unwrap_or(true);

    // ---- 与桌面 prepare_task 一致的清洗 + 包裹（逐文件） ----
    let prompts: Vec<(String, String)> = prompt_files
        .iter()
        .map(|pf| -> Result<(String, String), Box<dyn Error>> {
            let raw = std::fs::read_to_string(pf)?;
            // 含英文角色标记的按 preserve_roles 原样放行；否则包 User/Assistant（生产行为）
            let has_role = [
                "User:",
                "Assistant:",
                "System:",
                "# User",
                "# Assistant",
                "# System",
            ]
            .iter()
            .any(|m| raw.contains(m));
            // 生产管线是「每条消息内部 sanitize、段落间保持 \n\n」——因此含角色标记的
            // 提示词文件必须原样放行，不能整体 sanitize（会把段落间 \n\n 折叠成 \n，
            // 与生产的 token 流失真）。
            let prompt = if has_role {
                raw.trim_end_matches(['\n', '\r']).to_string()
            } else {
                format!("User: {}\n\nAssistant: ", sanitize_rwkv_content(&raw))
            };
            Ok((pf.clone(), prompt))
        })
        .collect::<Result<_, _>>()?;

    log_step(&format!("加载模型: {}", model_path));
    let Bundle {
        mut model,
        state: _,
    } = ModelBuilder::new(&model_path).build()?;
    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tokenizer = Tokenizer::new(&vocab)?;

    // FENCE_STOP=1：生产围栏收尾方案——遇闭合 ``` 即停（与 consult.ts FENCE_STOPS 一致）
    let fence_stop = std::env::var("FENCE_STOP").ok().as_deref() == Some("1");

    for (pf, prompt) in &prompts {
        let prompt_tokens = tokenizer.encode(prompt.as_bytes())?;
        log_step(&format!(
            "[{pf}] prompt {} 字符 → {} tokens",
            prompt.chars().count(),
            prompt_tokens.len()
        ));
        let tag = std::path::Path::new(pf)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("p")
            .to_string();
        for run in 1..=runs {
            // 每 run 新建零初始态——模型权重不重载
            let mut state = model.create_state()?;
            let out = run_once(
                &mut model,
                &mut state,
                &tokenizer,
                &prompt_tokens,
                ntokens,
                top_p,
                top_k,
                pen,
                stop_on_blank,
                fence_stop,
            )?;
            let text = format!(
                "[{pf}] run {run}/{runs}\n===== OUTPUT ({}, stop={}) =====\n{}\n===== END ({:.2}s) =====\n",
                out.n_tokens, out.ended_by_stop, out.text, out.secs
            );
            print!("{text}");
            if let Some(dir) = &out_dir {
                let path = format!("{dir}/{out_prefix}-{tag}-{run}.txt");
                std::fs::write(&path, &text)?;
            }
        }
    }
    Ok(())
}

struct RunOutput {
    text: String,
    n_tokens: usize,
    ended_by_stop: bool,
    secs: f64,
}

#[allow(clippy::too_many_arguments)]
fn run_once(
    model: &mut rwkv_rsv::gpu_model::GpuModel,
    state: &mut rwkv_rsv::gpu_model::State,
    tokenizer: &Tokenizer,
    prompt_tokens: &[u32],
    ntokens: usize,
    top_p: f32,
    top_k: usize,
    pen: f32,
    stop_on_blank: bool,
    fence_stop: bool,
) -> Result<RunOutput, Box<dyn Error>> {
    // ---- 分块 prefill（forward_seq_with_state）----
    let mut last_logits: Vec<f32> = Vec::new();
    for chunk in prompt_tokens.chunks(PREFILL_CHUNK) {
        last_logits = model.forward_seq_with_state(state, chunk)?;
    }

    // ---- Decode 循环：CPU sample_token 复刻 ----
    let mut token_counts: HashMap<u32, i32> = HashMap::new();
    let mut acc_ids: Vec<u32> = Vec::new();
    let mut stop_buffer = String::new();
    let stop_seqs: Vec<String> = {
        let mut v: Vec<String> = if stop_on_blank {
            vec![
                "\n\nUser:".into(),
                "\n\nSystem:".into(),
                "\n\nInstruction:".into(),
                "\n\nInput:".into(),
                "\n\n".into(),
            ]
        } else {
            vec![
                "\n\nUser:".into(),
                "\n\nSystem:".into(),
                "\n\nInstruction:".into(),
                "\n\nInput:".into(),
            ]
        };
        if fence_stop {
            v.insert(0, "\n```".into());
        }
        v
    };
    let mut ended_by_stop = false;

    let t0 = std::time::Instant::now();
    for _ in 0..ntokens {
        let id = sample_token(
            &last_logits,
            top_p,
            top_k,
            &token_counts,
            pen,
            pen,
            0.996_540_26,
        );
        token_counts.entry(id).and_modify(|c| *c += 1).or_insert(1);

        let logits = model.forward_with_state(state, &[id])?;
        last_logits = logits;

        acc_ids.push(id);
        let decoded = tokenizer.decode(&[id]).unwrap_or_default();
        let token_str = String::from_utf8_lossy(&decoded).to_string();
        stop_buffer.push_str(&token_str);
        if stop_buffer.len() > 200 {
            let split_idx = stop_buffer.len() - 100;
            if let Some((idx, _)) = stop_buffer.char_indices().find(|(i, _)| *i >= split_idx) {
                stop_buffer = stop_buffer[idx..].to_string();
            }
        }
        if stop_seqs
            .iter()
            .any(|s| !s.is_empty() && stop_buffer.ends_with(s.as_str()))
        {
            ended_by_stop = true;
            break;
        }
    }

    let full = tokenizer.decode(&acc_ids)?;
    Ok(RunOutput {
        text: String::from_utf8_lossy(&full).to_string(),
        n_tokens: acc_ids.len(),
        ended_by_stop,
        secs: t0.elapsed().as_secs_f64(),
    })
}

/// 桌面 sample_token 复刻（无温度缩放 + top_k/top_p + 惩罚）。
fn sample_token(
    logits: &[f32],
    top_p: f32,
    top_k: usize,
    token_counts: &HashMap<u32, i32>,
    presence_penalty: f32,
    frequency_penalty: f32,
    penalty_decay: f32,
) -> u32 {
    let mut logits = logits.to_vec();
    for (&id, &count) in token_counts {
        if (id as usize) < logits.len() {
            let penalty = presence_penalty + frequency_penalty * (count as f32).powf(penalty_decay);
            logits[id as usize] -= penalty;
        }
    }
    let probs = softmax(&logits);
    let mut candidates: Vec<(usize, f32)> =
        probs.iter().enumerate().map(|(i, &p)| (i, p)).collect();
    candidates.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    let mut cumsum = 0.0f32;
    let mut picked: Vec<(usize, f32)> = Vec::new();
    for (i, p) in candidates.into_iter().take(top_k.max(1)) {
        cumsum += p;
        picked.push((i, p));
        if cumsum >= top_p {
            break;
        }
    }
    let r = fastrand::f64() as f32 * cumsum.min(1.0);
    let mut acc = 0.0f32;
    let mut selected = picked.first().map(|&(i, _)| i as u32).unwrap_or(0);
    for (i, p) in picked {
        acc += p;
        if acc >= r {
            selected = i as u32;
            break;
        }
    }
    selected
}

fn softmax(logits: &[f32]) -> Vec<f32> {
    let max = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let exps: Vec<f32> = logits.iter().map(|&x| (x - max).exp()).collect();
    let sum: f32 = exps.iter().sum();
    exps.into_iter().map(|e| e / sum).collect()
}

/// 桌面 sanitize_rwkv_content 复刻。
fn sanitize_rwkv_content(content: &str) -> String {
    let s = content.replace("\r\n", "\n").replace('\r', "\n");
    let mut result = String::with_capacity(s.len());
    let mut prev_nl = false;
    for ch in s.chars() {
        if ch == '\n' {
            if !prev_nl {
                result.push('\n');
            }
            prev_nl = true;
        } else {
            result.push(ch);
            prev_nl = false;
        }
    }
    result.trim_end_matches('\n').to_string()
}

fn log_step(msg: &str) {
    println!("[{:?}] {}", std::time::SystemTime::now(), msg);
}
