//! Teacher-forced 质量判据：**不采样**，把语料里的真 token 喂回去，量
//!   ① `CE`（交叉熵，nats/token）；② `top-1` 命中率；③ 相对参考臂的 top-1 一致率。
//!
//! 为什么需要它（台账 §3.15.8 的待办）：int6→int8 副本这类"换内核/换舍入"的改动会改变批量
//! 贪心解码的 token 轨迹，但**贪心轨迹会把 1e-3 级的 logits 差异级联放大到饱和**（跑几十步就
//! 进 2-周期），所以拿"会不会进周期"当判据判不动任何改动。teacher-forced 下**输入序列固定**，
//! 每一步的状态与参考臂完全同步 ⇒ 量出来的就是**单步数值口径**，且 CE/top-1 是**确定性**的
//! （不受 GPU 时钟/功耗噪声影响，跨进程可比）。
//!
//! 用法（每个臂一个进程，`v`=变量）：
//! ```powershell
//! $env:MODEL_PATH="...int6n.st"; $env:VOCAB_JSON="...vocab.json"
//! $env:TF_SLOTS="16"; $env:TF_TAIL="64"; $env:TF_PROMPT="32"
//! $env:OUT_I8_MINB="999"   # 参考臂：att.output 走 int6
//! cargo run --release --example teacher_forced_eval
//! $env:OUT_I8_MINB="1"     # 被测臂：att.output 走 int8 副本
//! $env:TF_TOK_OUT="target/tf_out1.txt"
//! cargo run --release --example teacher_forced_eval
//! ```
//! `TF_TOK_OUT` 落每 (slot, step) 的 top-1 token，两份文件逐行比 = **一致率**。
//! 语料默认取仓库里的一篇实施记录（`TF_CORPUS` 可换），切片后各槽互不重叠。

use std::error::Error;
use std::time::Instant;

use rwkv_rsv::backend::{create_backend, detect_backend};
use rwkv_rsv::gpu_model::GpuModel;
use rwkv_rsv::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    let _ = simplelog::TermLogger::init(
        log::LevelFilter::Warn,
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    );
    let model_path = std::env::var("MODEL_PATH")?;
    let vocab_path = std::env::var("VOCAB_JSON")?;
    let corpus_path = std::env::var("TF_CORPUS")
        .unwrap_or_else(|_| "参考/2026-11-08-int6量化与MTP-阶段记录.md".to_string());
    let slots: usize = envn("TF_SLOTS", 16);
    let prompt: usize = envn("TF_PROMPT", 32);
    let tail: usize = envn("TF_TAIL", 64);

    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tok = Tokenizer::new(&vocab)?;
    let text = std::fs::read_to_string(&corpus_path)?;
    let toks = tok.encode(text.as_bytes())?;
    assert!(
        toks.len() >= slots * (prompt + tail) + 1,
        "语料太短：{} tok，需要 {}（TF_SLOTS×(TF_PROMPT+TF_TAIL)+1）",
        toks.len(),
        slots * (prompt + tail) + 1
    );

    let backend = create_backend(detect_backend())?;
    let mut model = GpuModel::from_safetensors(backend, &model_path)?;
    let n_vocab = model.info().num_vocab;
    let mut state = model.create_batch_state(slots)?;
    model.reset_state_of(&state)?;

    // 每槽取一段互不重叠的连续语料：prefill 前 `prompt-1` 个，其余做教师目标。
    let seg = prompt + tail;
    let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(slots);
    let mut truth: Vec<Vec<u32>> = Vec::with_capacity(slots);
    for s in 0..slots {
        let base = s * seg;
        prompts.push(toks[base..base + prompt - 1].to_vec());
        truth.push(toks[base + prompt - 1..base + prompt + tail].to_vec());
    }
    let t0 = Instant::now();
    let _ = model.forward_seq_batch_padded(&mut state, &prompts, 256)?;

    let mut feed: Vec<Vec<u32>> = truth.iter().map(|t| vec![t[0]]).collect();
    let (mut ce_sum, mut n, mut top1_ok) = (0.0f64, 0usize, 0usize);
    let mut top1: Vec<u32> = Vec::with_capacity(slots * tail);
    for step in 0..tail - 1 {
        let lg = model.batch_step_logits(&mut state, &feed)?;
        for s in 0..slots {
            let row = &lg[s * n_vocab..(s + 1) * n_vocab];
            let target = truth[s][step + 1] as usize;
            // log-softmax（f64，先减 max）
            let mx = row.iter().fold(f32::MIN, |a, &v| a.max(v)) as f64;
            let mut sum = 0.0f64;
            for &v in row {
                sum += ((v as f64) - mx).exp();
            }
            let lp = (row[target] as f64 - mx) - sum.ln();
            ce_sum += -lp;
            n += 1;
            let best = row
                .iter()
                .enumerate()
                .fold(
                    (0usize, f32::MIN),
                    |a, (i, &v)| if v > a.1 { (i, v) } else { a },
                )
                .0;
            if best == target {
                top1_ok += 1;
            }
            top1.push(best as u32);
        }
        feed = truth.iter().map(|t| vec![t[step + 1]]).collect::<Vec<_>>();
    }
    let secs = t0.elapsed().as_secs_f64();
    let ce = ce_sum / n as f64;
    log::warn!(
        "[TF] B={slots} prompt={prompt} tail={tail} 评了 {n} 个位置（{secs:.1}s）\
         | CE={ce:.5} nats/tok | top1={:.4}（{top1_ok}/{n}）",
        top1_ok as f64 / n as f64
    );
    // 机器可读行（便于把多个臂拼成一张表）
    println!(
        "TF_RESULT ce={ce:.6} top1={:.6} n={n} slots={slots} prompt={prompt} tail={tail}",
        top1_ok as f64 / n as f64
    );
    if let Ok(p) = std::env::var("TF_TOK_OUT") {
        use std::io::Write as _;
        let f = std::fs::File::create(&p)?;
        let mut w = std::io::BufWriter::new(f);
        for (i, t) in top1.iter().enumerate() {
            writeln!(w, "{i} {t}")?;
        }
        log::warn!("[TF] top-1 逐位置 -> {p}");
    }
    Ok(())
}

fn envn<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
