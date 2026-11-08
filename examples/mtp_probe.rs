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

    // —— 投机步 A/B：同一贪心流，plain 逐 token vs spec（起草→验证→回滚）——
    let n_gen = env_usize("MTP_GEN", 32);
    let am = |v: &[f32]| -> u32 {
        let mut b = 0;
        for i in 1..v.len() {
            if v[i] > v[b] {
                b = i;
            }
        }
        b as u32
    };
    // ★ 两臂都必须从**同一个真实 prompt** 出发：`seed=1` 会让模型掉进「token 1 自循环」，
    // 而 token 1 不在 draft 词表内 ⇒ 接受率必然 0（见台账 §3.3 的坑 2）。
    model.mtp_reset()?;
    model.reset_state()?;
    let mut tok = model.forward_argmax(&prompt)?;
    let mut plain = Vec::with_capacity(n_gen);
    plain.push(tok); // prompt 之后的第一个 token（两臂对齐到同一位置）
    let tp = std::time::Instant::now();
    while plain.len() < n_gen {
        tok = model.forward_argmax(&[tok])?;
        plain.push(tok);
    }
    let ms_plain = tp.elapsed().as_secs_f64() * 1000.0;

    // spec 臂：prompt 逐 token 走，并同步提交 drafter（§3.4 的 prefill 提交规则）
    let seed_spec = |bundle: &mut rwkv_rsv::gpu_model::Bundle| -> Result<u32, Box<dyn Error>> {
        bundle.reset()?;
        bundle.model.mtp_reset()?;
        let mut tok = 0u32;
        for (i, &t) in prompt.iter().enumerate() {
            let lg = bundle.infer_tokens(&[t])?;
            let nt = am(&lg);
            if i + 1 < prompt.len() {
                let _ = bundle.model.mtp_draft_host(prompt[i + 1], 1)?;
            } else {
                tok = nt;
            }
        }
        Ok(tok)
    };
    // 预热：图捕获 + 各 n 的 `dplr_seq` 变体编译都在前几步，不计时
    let mut w = seed_spec(&mut bundle)?;
    for _ in 0..k + 2 {
        let out = bundle.model.spec_step(&mut bundle.state, w, k)?;
        w = *out.last().unwrap();
    }
    let mut tok = seed_spec(&mut bundle)?;
    let mut spec = Vec::with_capacity(n_gen);
    let mut steps = 0usize;
    let mut accepted = 0usize;
    let ts = std::time::Instant::now();
    while spec.len() < n_gen {
        let out = bundle.model.spec_step(&mut bundle.state, tok, k)?;
        accepted += out.len() - 2; // 被接受的草稿数（末尾那个是新采样、未进状态）
        steps += 1;
        for &t in &out[..out.len() - 1] {
            spec.push(t);
        }
        tok = *out.last().unwrap();
    }
    let ms_spec = ts.elapsed().as_secs_f64() * 1000.0;

    // 臂 C：与 spec 臂走**同一条 seq 路径**的逐 token 解码（T=1 的验证 pass）。
    // 作用是把「投机机制本身有没有错」与「seq/decode 两条路径的数值差」分开：
    // 若 B 与 C 长程一致而 B≠A，则机制正确，分歧只来自两条路径的数值档。
    let mut c = seed_spec(&mut bundle)?;
    let mut seqplain = Vec::with_capacity(n_gen);
    seqplain.push(c);
    while seqplain.len() < n_gen {
        let p1 = bundle
            .model
            .forward_seq_verify_with_state(&mut bundle.state, &[c])?;
        c = p1[0];
        seqplain.push(c);
    }
    let mut same_ac = 0usize;
    for (a, b) in spec.iter().zip(seqplain.iter()) {
        if a == b {
            same_ac += 1;
        } else {
            break;
        }
    }
    log::info!("  spec vs seq-逐token：前 {same_ac}/{n_gen} 个一致（这条决定「机制」对不对）");
    if std::env::var("MTP_PROF").is_ok() {
        log::info!("  spec    [0..8] = {:?}", &spec[..spec.len().min(8)]);
        log::info!(
            "  seqplain[0..8] = {:?}",
            &seqplain[..seqplain.len().min(8)]
        );
        log::info!("  plain   [0..8] = {:?}", &plain[..plain.len().min(8)]);
    }

    let model = &mut bundle.model;

    // 流一致性：逐位置比对（near-tie 允许分歧，与上游同样声明）
    let mut same = 0usize;
    for (a, b) in plain.iter().zip(spec.iter()) {
        if a == b {
            same += 1;
        } else {
            break;
        }
    }
    log::info!(
        "—— 投机步 A/B（贪心 {n_gen} token，K={k}）——\n  plain: {:.2} ms（{:.1} tok/s）\n  spec : {:.2} ms（{:.1} tok/s）⇒ **{:.2}×**，{steps} 步、平均接受 {:.2} 草稿/步\n  流一致性：前 {same}/{n_gen} 个一致{}",
        ms_plain,
        n_gen as f64 * 1000.0 / ms_plain,
        ms_spec,
        n_gen as f64 * 1000.0 / ms_spec,
        ms_plain / ms_spec,
        accepted as f64 / steps.max(1) as f64,
        if same == n_gen {
            "（完全一致）"
        } else {
            "（near-tie 分歧，见文档）"
        }
    );

    // —— 成本三件套：验证券化之前的先决读数（决定 K 与回滚策略）——
    // ① 单 token decode（plain 基线）② T=K+1 的因果 seq pass（验证明细）③ drafter 链
    let reps = env_usize("MTP_REPS", 16).max(1);
    let t0 = std::time::Instant::now();
    for i in 0..reps {
        let _ = model.forward_argmax(&[(i % 1000) as u32 + 1])?;
    }
    let ms_dec = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    // 小 T seq 路径自检：同一段 T 个 token，「逐 token 解码」与「一次因果 seq pass」的
    // 末行 logits 必须同 argmax（GEMV vs GEMM 累加顺序不同 ⇒ 允许小的数值差）。
    {
        let toks: Vec<u32> = (1..=k as u32 + 1).collect();
        bundle.reset()?;
        let l_tok = bundle.infer_tokens(&toks)?;
        bundle.reset()?;
        let l_seq = bundle.infer_seq(&toks)?;
        let (mut md, mut mi) = (0.0f32, (0usize, 0usize));
        for i in 0..l_tok.len() {
            let d = (l_tok[i] - l_seq[i]).abs();
            if d > md {
                md = d;
                mi = (i, i);
            }
        }
        let am = |v: &[f32]| -> usize {
            let mut b = 0;
            for i in 1..v.len() {
                if v[i] > v[b] {
                    b = i;
                }
            }
            b
        };
        let (a1, a2) = (am(&l_tok), am(&l_seq));
        log::info!(
            "[自检] T={} seq vs 解码：argmax {a1} vs {a2} ⇒ {}；max|Δlogit|={md:.4e}（idx {}）",
            toks.len(),
            if a1 == a2 { "一致" } else { "❌ 不一致" },
            mi.0
        );
    }
    let t0 = std::time::Instant::now();
    for i in 0..reps {
        let toks = vec![(i % 1000) as u32 + 1; k + 1];
        let _ = bundle.infer_seq(&toks)?;
    }
    let ms_seq = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    // ★ 带存档的验证 pass（投机环真正走的那条）与上面的 `infer_seq` 配对。
    // 两者差 = 存档（32 层 × 8 张 `[T,C]` 拷贝）+ 逐位置 head/argmax 读回 + 图键差异。
    // 必须**同进程紧邻**测量：本机无法锁频，跨进程比值会被热态漂移吃掉（见台账 §3.10）。
    let t0 = std::time::Instant::now();
    for i in 0..reps {
        let toks = vec![(i % 1000) as u32 + 1; k + 1];
        let _ = bundle
            .model
            .forward_seq_verify_with_state(&mut bundle.state, &toks)?;
    }
    let ms_verify = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;
    // 回滚自检：同一基线下「T 全 pass + 回滚到 2」必须等于「直接 T=2 pass」
    if std::env::var("MTP_ROLLBACK_TEST").is_ok() {
        let mut c0 = seed_spec(&mut bundle)?;
        let mut toks = vec![c0];
        for _ in 0..k {
            let p1 = bundle
                .model
                .forward_seq_verify_with_state(&mut bundle.state, &[c0])?;
            c0 = p1[0];
            toks.push(c0);
        }
        let (rolled, reference) =
            bundle
                .model
                .mtp_rollback_selftest(&mut bundle.state, &toks, 2)?;
        let mut md = 0.0f32;
        let mut mi = 0usize;
        for i in 0..rolled.len().min(reference.len()) {
            let d = (rolled[i] - reference[i]).abs();
            if d > md {
                md = d;
                mi = i;
            }
        }
        log::info!(
            "[回滚自检] tokens={toks:?} n=2：max|Δstate| = {md:.3e}（idx {mi} / {}）⇒ {}",
            rolled.len(),
            if md < 1e-5 {
                "逐位一致 ✅"
            } else {
                "❌ 不一致（回滚有 bug）"
            }
        );
    }
    let model = &mut bundle.model;

    let t0 = std::time::Instant::now();
    for i in 0..reps {
        let _ = model.mtp_draft_host((i % 1000) as u32 + 1, k)?;
    }
    let ms_draft = t0.elapsed().as_secs_f64() * 1000.0 / reps as f64;

    log::info!("—— 成本（{reps} 次取均值，K={k}）——");
    log::info!("  单 token decode        : {ms_dec:.3} ms");
    log::info!(
        "  T={} 因果 seq pass     : {ms_seq:.3} ms（验证券化用）",
        k + 1
    );
    log::info!("  drafter 链 {k} 深度      : {ms_draft:.3} ms");
    log::info!(
        "  T={} 验证 pass（带存档） : {ms_verify:.3} ms（比 infer_seq 多 {:.3} ms = 存档+逐位置读回）",
        k + 1,
        ms_verify - ms_seq
    );
    let acc = acc_sum / steps.max(1) as f64;
    for (name, extra) in [("免费回滚（理想）", 0.0), ("第二遍 seq 重放", ms_seq)] {
        let per_step = ms_seq + ms_draft + extra;
        log::info!(
            "  投影：{name} ⇒ {:.2} ms/token、{:.2}×",
            per_step / acc,
            ms_dec / (per_step / acc)
        );
    }
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
