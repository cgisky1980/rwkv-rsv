//! 深度循环（recurrent depth）teacher-forced 评估。
//!
//! 对应计划 `rwkv_recurrent_depth_plan.md` 的 P0/P1：在**不训练**的前提下，把已训练
//! RWKV-7 的连续一段层当作权重共享的循环块，量 L / state 提交策略 / 残差重注入
//! 对 CE、PPL、top-1 与 state 字节数的影响。
//!
//! 用法（一个臂一个进程）：
//! ```bash
//! MODEL_PATH=<model.st> VOCAB_JSON=<vocab.json> RD_CORPUS=<text> \
//! RD_MODE=recurrent RD_L=2 RD_COMMIT=last RD_REINJECT=1 \
//! RD_IN=6 RD_LOOP=12 RD_OUT=6 RD_SLOTS=32 RD_THREADS=8 \
//! cargo run --release --example recurrent_depth_eval
//! ```
//! `RD_MODE=baseline` 走 L=1 恒等调度（已验证与原 `Model::forward` 数值一致）。
//! 各语料槽位状态独立，按槽位开线程并行；模型只读共享。

use std::error::Error;
use std::str::FromStr;
use std::time::Instant;

use rwkv_rsv::model::{DepthConfig, Model, StateCommit};
use rwkv_rsv::tokenizer::Tokenizer;

/// 前向只读模型权重；`calib` 字段虽是 RefCell，但本评估路径完全不触碰它。
struct SharedModel(Model);
// SAFETY: forward_recurrent 只读取除 calib 外的字段，每个线程独占自己的 LayerState。
unsafe impl Sync for SharedModel {}

#[derive(Default)]
struct SlotAcc {
    ce_sum: f64,
    top1_ok: usize,
    n_eval: usize,
    n_token: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let model_path = std::env::var("MODEL_PATH")?;
    let vocab_path = std::env::var("VOCAB_JSON")?;
    let corpus_path = std::env::var("RD_CORPUS").unwrap_or_else(|_| "README.md".into());
    let mode = std::env::var("RD_MODE").unwrap_or_else(|_| "recurrent".into());
    let slots: usize = envn("RD_SLOTS", 8);
    let threads: usize = envn("RD_THREADS", 8).max(1).min(slots);
    let prompt: usize = envn("RD_PROMPT", 16);
    let tail: usize = envn("RD_TAIL", 24);
    let reinject = envn::<u8>("RD_REINJECT", 1) != 0;

    let model = SharedModel(Model::from_safetensors(&model_path)?);
    let cfg = if mode == "baseline" {
        DepthConfig::passthrough(model.0.config.n_layer)
    } else if mode == "recurrent" {
        DepthConfig {
            n_in: envn("RD_IN", 0),
            n_loop: envn("RD_LOOP", model.0.config.n_layer),
            n_out: envn("RD_OUT", 0),
            l: envn("RD_L", 1),
            commit: StateCommit::from_str(
                &std::env::var("RD_COMMIT").unwrap_or_else(|_| "last".into()),
            )?,
            reinject,
        }
    } else {
        return Err(format!("unknown RD_MODE={mode}").into());
    };
    cfg.validate(model.0.config.n_layer)
        .map_err(|e| -> Box<dyn Error> { e.into() })?;

    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tok = Tokenizer::new(&vocab)?;
    let text = std::fs::read_to_string(&corpus_path)?;
    let toks = tok.encode(text.as_bytes())?;
    let seg = prompt + tail;
    if toks.len() < slots * seg + 1 {
        return Err(format!(
            "语料太短：{} tok，需要 {}（RD_SLOTS×(RD_PROMPT+RD_TAIL)+1）",
            toks.len(),
            slots * seg + 1
        )
        .into());
    }

    // 每层 state = tmix_x[C] + tmix_rnn[H*N*N] + cmix_x[C]，f32。
    // 循环块只有一份常驻 state；非 commit 轮的临时快照不常驻，也不随 L 增长。
    let c = model.0.config.n_embd;
    let h = model.0.config.n_head;
    let n = model.0.config.head_size;
    let per_layer = (2 * c + h * n * n) * std::mem::size_of::<f32>();
    let state_bytes = model.0.config.n_layer * per_layer;
    let snapshot_bytes = cfg.n_loop * per_layer;

    let t0 = Instant::now();
    let chunk = slots.div_ceil(threads);
    let model_ref: &SharedModel = &model;
    let toks_ref: &[u32] = &toks;
    let accs: Vec<SlotAcc> = std::thread::scope(|scope| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let start = t * chunk;
                let end = (start + chunk).min(slots);
                scope.spawn(move || {
                    let mut acc = SlotAcc::default();
                    for s in start..end {
                        let base = s * seg;
                        let segment = &toks_ref[base..base + seg];
                        let mut states = model_ref.0.init_state();
                        for i in 0..segment.len() - 1 {
                            let lg =
                                model_ref
                                    .0
                                    .forward_recurrent(&[segment[i]], &mut states, &cfg)?;
                            acc.n_token += 1;
                            if i < prompt - 1 {
                                continue;
                            }
                            let target = segment[i + 1] as usize;
                            let mx = lg.iter().fold(f32::MIN, |a, &v| a.max(v)) as f64;
                            let mut sum = 0.0f64;
                            for &v in &lg {
                                sum += ((v as f64) - mx).exp();
                            }
                            acc.ce_sum += mx + sum.ln() - (lg[target] as f64);
                            let best = lg
                                .iter()
                                .enumerate()
                                .fold(
                                    (0usize, f32::MIN),
                                    |a, (j, &v)| {
                                        if v > a.1 { (j, v) } else { a }
                                    },
                                )
                                .0;
                            if best == target {
                                acc.top1_ok += 1;
                            }
                            acc.n_eval += 1;
                        }
                    }
                    Ok::<SlotAcc, String>(acc)
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().unwrap().expect("slot evaluation failed"))
            .collect()
    });

    let acc = accs.into_iter().fold(SlotAcc::default(), |mut a, b| {
        a.ce_sum += b.ce_sum;
        a.top1_ok += b.top1_ok;
        a.n_eval += b.n_eval;
        a.n_token += b.n_token;
        a
    });
    let secs = t0.elapsed().as_secs_f64();
    let ce = acc.ce_sum / acc.n_eval as f64;
    let ppl = ce.exp();
    let top1 = acc.top1_ok as f64 / acc.n_eval as f64;
    println!(
        "RD_RESULT mode={mode} L={} commit={:?} reinject={} n_in={} n_loop={} n_out={} \
         eff_depth={} ce={ce:.6} ppl={ppl:.4} top1={top1:.6} n={} \
         state_bytes={state_bytes} snapshot_bytes={snapshot_bytes} \
         ms_per_tok={:.4} wall_s={:.1} slots={slots} threads={threads} prompt={prompt} tail={tail}",
        cfg.l,
        cfg.commit,
        reinject as u8,
        cfg.n_in,
        cfg.n_loop,
        cfg.n_out,
        cfg.effective_depth(),
        acc.n_eval,
        secs * 1000.0 / acc.n_token as f64,
        secs,
    );
    Ok(())
}

fn envn<T: std::str::FromStr>(k: &str, d: T) -> T {
    std::env::var(k)
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(d)
}
