//! MTP drafter 的**接受长度**探针（P4）：不测速度，只验证「起草得对不对」。
//!
//! 为什么先测接受长度：drafter 的接线错误（tap 层号、`dep` 行、RoPE 位置、token shift、
//! draft 词表）在速度上只表现为「不加速」（1.0×，与「MTP 本来就没收益」无法区分），
//! 在接受长度上是**数量级**的变化（接线正确 ≈ 1.8–2.5，接错 ≈ 1.0）。所以先钉住这个数。
//!
//! 口径：主模型跑 **greedy**（argmax）自生成流；每步用刚算完的 `ln_out(x)` + 两个 tap
//! 起草 K 个 token，再与主模型随后真正产生的 token 逐位置比对。
//! 接受长度按真投机步的定义：`1 + 连续命中的草稿数`（第 0 个 token 恒由主模型给出）。
//! 被拒绝的草稿不会改变 drafter 已提交的流（只提交被接受的位置），所以「顺序跑、逐步比对」
//! 与真投机环的接受率等价。
//!
//! 用法：
//! ```powershell
//! $env:MODEL_PATH="...\rwkv7-g1l-2.9b-int8.st"; $env:MTP_K=3; $env:MTP_TOKENS=64
//! cargo run --release --example mtp_probe
//! ```
//! 环境变量：`MODEL_PATH`（必填）、`MTP_K`（默认 3）、`MTP_TOKENS`（默认 64）、
//! `MTP_SEED`（初始 token，默认 1）、`MTP_FORCE=1`（放行 `n_layer` 不符的 sidecar）。

use std::error::Error;

use rwkv_rsv::gpu_model::ModelBuilder;
use rwkv_rsv::mtp::{find_sidecar, load_sidecar};

fn main() -> Result<(), Box<dyn Error>> {
    init_log();
    let model_path = std::env::var("MODEL_PATH")?;
    let k = env_usize("MTP_K", 3);
    let n = env_usize("MTP_TOKENS", 64);
    let seed = env_usize("MTP_SEED", 1) as u32;
    let force = std::env::var("MTP_FORCE").is_ok_and(|v| v == "1");

    let mut bundle = ModelBuilder::new(&model_path).build()?;
    let info = bundle.info();
    let (c, l) = (info.num_emb, info.num_layer);

    let path =
        find_sidecar(&model_path)?.ok_or("找不到 MTP sidecar（RWKV_MTP_PATH=auto 也没命中）")?;
    let sc = load_sidecar(&path)?;
    sc.check_against_model(c, l, force)?;
    log::info!(
        "sidecar: {}（C={} L={} taps={:?} W={} depth={} draft_vocab={}）",
        sc.path.display(),
        sc.meta.n_embd,
        sc.meta.n_layer,
        sc.meta.fuse_layers,
        sc.meta.window,
        sc.meta.depth,
        sc.draft_vocab.len()
    );
    let model = &mut bundle.model;
    model.enable_mtp(&sc, k)?;

    // 预热（构建 kernel 缓存；drafter 的 kernel 第一轮也在其中）
    let _ = model.forward_argmax(&[seed])?;
    model.mtp_reset()?;
    model.reset_state()?;

    // 先把 prompt 逐个喂给主模型，**同时**把 drafter 的已提交位置对齐（§3.4）：
    // 每处理完位置 i 就提交 `(row_i, prompt[i+1])`，于是 `drafter.pos == 主模型已处理位置数`。
    // 不对齐会让 RoPE 位置与 KV 窗口整体错位，接受率直接归零（不是慢，是错）。
    let prompt: Vec<u32> = std::env::var("MTP_PROMPT")
        .ok()
        .filter(|v| !v.trim().is_empty())
        .map(|v| v.split(',').filter_map(|s| s.trim().parse().ok()).collect())
        .unwrap_or_else(|| vec![304, 25740, 109]);
    let mut tok = 0u32;
    for (i, &t) in prompt.iter().enumerate() {
        tok = model.forward_argmax(&[t])?;
        if i + 1 < prompt.len() {
            let _ = model.mtp_draft_host(prompt[i + 1], 1)?; // 只取提交副作用
        }
    }
    log::info!(
        "prompt {prompt:?} → drafter.pos={:?}（期望 {}）",
        model.mtp_pos(),
        prompt.len() - 1
    );

    // 主模型 greedy 自生成；每一步的草稿先记下，等真值出现后统一比对。
    let mut actual: Vec<u32> = vec![0; n + 2];
    actual[1] = tok;
    let mut drafts_by_step: Vec<Vec<u32>> = Vec::new();

    for p in 0..n.saturating_sub(1) {
        // 通路自检（前 3 步）：drafter 的 head 对同一行 x_norm 的 argmax 必须等于主模型
        if p < 3 {
            let chk = model.mtp_head_check()?;
            log::info!(
                "[自检] 步 {p}: head(x_norm)={chk} vs 主模型 tok={tok} ⇒ {}",
                if chk == tok {
                    "一致"
                } else {
                    "❌ 不一致（taps/head 通路有问题）"
                }
            );
        }
        let d = model.mtp_draft_host(tok, k)?;
        let next = model.forward_argmax(&[tok])?; // 处理位置 p+1 → 位置 p+2 的 token
        if p < 5 {
            log::info!(
                "[步 {p}] in={tok}(draft词表内={}) 草稿={d:?} 真值={next}(draft词表内={})",
                in_draft(tok, &sc),
                in_draft(next, &sc)
            );
        }
        drafts_by_step.push(d);
        actual[p + 2] = next;
        tok = next;
    }

    // 比对：步 p 的第 d 个草稿预测位置 p+1+d
    let mut hit = vec![0usize; k + 1];
    let mut seen = vec![0usize; k + 1];
    let mut acc_sum = 0.0f64;
    let mut steps = 0usize;
    for (p, drafts) in drafts_by_step.iter().enumerate() {
        let mut cnt = 0usize;
        for (i, &d) in drafts.iter().enumerate() {
            let q = p + 2 + i;
            if q > n {
                break;
            }
            seen[i + 1] += 1;
            if d == actual[q] {
                hit[i + 1] += 1;
                if cnt == i {
                    cnt += 1;
                }
            }
        }
        acc_sum += 1.0 + cnt as f64;
        steps += 1;
    }

    log::info!("—— MTP 接受长度（greedy 自生成，{steps} 步，K={k}）——");
    for d in 1..=k {
        if seen[d] > 0 {
            log::info!(
                "  depth {d}: top-1 命中 {}/{} = {:.1}%",
                hit[d],
                seen[d],
                100.0 * hit[d] as f64 / seen[d] as f64
            );
        }
    }
    log::info!(
        "  平均接受长度 = {:.3} token/主前向（门限 ≥1.8）",
        acc_sum / steps.max(1) as f64
    );
    Ok(())
}

/// 该 token 是否在 drafter 的 draft 词表内（掩码之外 ⇒ drafter 永远不可能预测它）。
fn in_draft(tok: u32, sc: &rwkv_rsv::mtp::MtpSidecar) -> bool {
    sc.draft_vocab
        .iter()
        .take(rwkv_rsv::mtp::MTP_DRAFT_VOCAB)
        .any(|&t| t == tok)
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
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
