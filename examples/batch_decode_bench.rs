//! 批量解码基准：隔离对比「单流自循环」vs「批量自循环」的每段时间。
//!
//! 用法：
//!   cargo run --release --example batch_decode_bench
//!   MODEL_PATH=... VOCAB_JSON=... SLOTS=8 SEGS=4 NTOK=32
//!   TOPK=50（0=无 top-k，走 top-p 兜底路径）；SAMPLE_TEMP 调高可制造平坦分布压测兜底多候选轮。
//!   PAD_TO=512（批量 prefill 的固定 T_pad）；SKIP_SINGLE=1 跳过单流基准（扫并发时省时间）。
//!   STEP=1：额外跑 `batch_step_sample` **逐步**路径（每步回主机），与上面的自循环路径
//!   对拍同一套 sum/xor 指纹——这是「采样结果陈旧」类 bug 的回归探针（见 §STEP 块注释）。
//!   DUMP_TOK_STEP=<路径>：STEP 路径逐 token 落盘，格式同 DUMP_TOK。
//!   SLOT_BENCH=1：量「把一份状态分发给多个槽」的两条做法（宿主往返 vs 设备内广播）。
//!   BUCKET_BENCH=1：量长度桶预热把多少用户可见延迟挪到启动期（见该块注释）。
//!   BENCH_JSON=<路径>：把本次**读数 + 完整门控**追加成一行 JSON（JSONL）。
//!
//! ## 为什么有 BENCH_JSON（基准可信度）
//!
//! README §6.1 的老表踩过两个坑：① 不同会话的列被并排放在一起（跨会话漂移 ±17% 足以
//! 反转结论）；② CUDA 侧三个张量核开关（`LOWRANK_GEMM` / `FFN_VALUE_GEMM` / `GEMV_IMMA`）
//! 默认全开，而配对读数没写门控 —— 于是「SIMT 对 SIMT」和「SIMT 对张量核」两种口径混在
//! 一张表里，读者无法判断比较的是什么。所以这里把**每个读数与它的门控绑成一条记录**：
//! 没有门控的读数不可引用。配合 `examples/bench_ab.rs`（同会话交错 A/B 编排）使用。
//!
//! **各槽提示词互不相同**（`prompt_for`）：全槽同 prompt 会把每层残差/状态压成同一条
//! 数值路径，测出来的是「同一份计算的 N 份拷贝」，不代表真实并发的内存/带宽行为。
//! 不同提示词还能让 `forward_seq_batch_padded` 的 lens 分布不均匀（更接近服务端实况）。

use std::error::Error;
use std::time::Instant;

use rwkv_rsv::backend::{create_backend, detect_backend};
use rwkv_rsv::gpu_model::{GpuModel, SamplerParams};
use rwkv_rsv::tokenizer::Tokenizer;

fn main() -> Result<(), Box<dyn Error>> {
    init_log();

    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| {
        r"C:\work\ai00-x-dev\client\target\release\models\rwkv\rwkv7-3B-int8.st".into()
    });
    let vocab_path = std::env::var("VOCAB_JSON").unwrap_or_else(|_| {
        r"C:\work\ai00-x-dev\client\target\release\models\rwkv\vocab.json".into()
    });
    let slots: usize = std::env::var("SLOTS")
        .unwrap_or_else(|_| "8".into())
        .parse()?;
    let segs: usize = std::env::var("SEGS")
        .unwrap_or_else(|_| "4".into())
        .parse()?;
    let ntok: usize = std::env::var("NTOK")
        .unwrap_or_else(|_| "32".into())
        .parse()?;
    let topk: u32 = std::env::var("TOPK")
        .unwrap_or_else(|_| "0".into())
        .parse()?;
    // 采样温度（调试用：调高可制造平坦分布，压测 top-p 兜底多候选轮路径）。
    // 注意：不可用 `TEMP` —— Windows 自带同名变量（临时目录路径），parse 必失败。
    let temp: f32 = std::env::var("SAMPLE_TEMP")
        .unwrap_or_else(|_| "1.0".into())
        .parse()?;
    // prompt 重复次数：默认 6（≈90 tok）；调大用于长 prompt（≥1024 tok）冒烟。
    let repeat: usize = std::env::var("PROMPT_REPEAT")
        .unwrap_or_else(|_| "6".into())
        .parse()?;
    // 预热段数（默认 1）：丢弃一段，消除 NVRTC 首次编译与首次 launch 开销。
    let warmup: usize = std::env::var("WARMUP")
        .unwrap_or_else(|_| "1".into())
        .parse()?;
    // top-p 累计阈值（默认 0.2）。`TOPP=1.0` 关闭 top-p 兜底循环——该循环是「一趟全词表
    // 只消费一个候选」，趟数由分布决定，是采样核耗时归因的主要变量。
    let topp: f32 = std::env::var("TOPP")
        .unwrap_or_else(|_| "0.2".into())
        .parse()?;
    // 批量 prefill 的固定 T_pad（跨批复用 seq 缓冲；超限时 kernel 内会自动抬升一次）。
    let pad_to: usize = std::env::var("PAD_TO")
        .unwrap_or_else(|_| "512".into())
        .parse()?;
    // 跳过单流基准（扫并发曲线时单流值是常量，重复跑纯属浪费）。
    let skip_single = std::env::var("SKIP_SINGLE").is_ok_and(|v| v != "0");
    // 逐步采样路径（batch_step_sample）对拍开关。
    let step_mode = std::env::var("STEP").is_ok_and(|v| v != "0");
    // 惩罚模式：STEP 路径按 token 追加惩罚历史（`write_u32_part`）+ 频率惩罚 > 0。
    // 只在 STEP=1 时有意义；每步每槽一次小段上传，是正确性探针而非吞吐路径。
    let penalty = std::env::var("PENALTY").is_ok_and(|v| v != "0");

    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tokenizer = Tokenizer::new(&vocab)?;
    log::info!("加载模型: {model_path}");
    let backend = create_backend(detect_backend())?;
    let mut model = GpuModel::from_safetensors(backend, &model_path)?;
    log::info!("模型就绪 vocab={}", model.info().num_vocab);
    model.log_vram("模型加载后");

    // 单流基准用的第一条提示词
    let prompt_text = prompt_for(0, repeat);
    let prompt_tokens = tokenizer.encode(prompt_text.as_bytes())?;
    let seed_token = prompt_tokens[prompt_tokens.len() - 1];
    log::info!("prompt(0) {} tok", prompt_tokens.len());

    log::info!("top_k={topk}（0=兜底 top-p 路径）");
    let mut sp = SamplerParams {
        temperature: temp,
        top_k: topk,
        top_p: topp,
        seed: 514,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        ..Default::default()
    };
    // PENALTY=1：给**两条**批量路径同时带上频率惩罚（自循环与逐步共用这一份 `sp`），
    // 这样逐步路径上「每步往 hist 行尾追加一个 token」的增量上传就有了可比的判据：
    // 与自循环（hist = 本段已生成 token、hist_len = round）仍应逐位一致。
    if penalty {
        sp.frequency_penalty = 0.2;
        log::info!("PENALTY=1：frequency_penalty=0.2，逐步路径按 token 追加惩罚历史");
    }

    // —— 单流基准 ——
    let mut single_tok_s: Option<f64> = None;
    if !skip_single {
        let mut single = model.create_state()?;
        let t0 = Instant::now();
        model.reset_state_of(&single)?;
        let prefill = &prompt_tokens[..prompt_tokens.len() - 1];
        model.forward_seq_with_state(&mut single, prefill)?;
        let mut seg_seed = seed_token;
        for _ in 0..warmup {
            let ticket = model.submit_sample_selfloop(&mut single, seg_seed, ntok, &sp)?;
            let toks = model.collect_sample_selfloop(ticket)?;
            seg_seed = toks[toks.len() - 1];
        }
        let t1 = Instant::now();
        let mut toks_total = 0usize;
        for _ in 0..segs {
            let ticket = model.submit_sample_selfloop(&mut single, seg_seed, ntok, &sp)?;
            let toks = model.collect_sample_selfloop(ticket)?;
            seg_seed = toks[toks.len() - 1];
            toks_total += toks.len();
        }
        let secs = t1.elapsed().as_secs_f64();
        let tok_s = toks_total as f64 / secs;
        single_tok_s = Some(tok_s);
        log::info!(
            "[单流] prefill {} ms | decode {} tok {} ms = {:.1} tok/s",
            t1.duration_since(t0).as_millis(),
            toks_total,
            (secs * 1000.0) as u128,
            tok_s
        );
    }

    // —— 批量基准（每槽不同 prompt）——
    {
        // 供 BENCH_JSON 汇总的可选读数（只在对应开关打开时才有值）。
        let mut step_readings: Option<(f64, u64, u64)> = None; // (tok/s, sum, xor)
        let mut slot_host_ms: Option<f64> = None;
        let mut slot_dev_ms: Option<f64> = None;
        let mut slot_bytes: Option<usize> = None;
        let mut bucket_readings: Option<serde_json::Value> = None;
        let t0 = Instant::now();
        let mut bstate = model.create_batch_state(slots)?;
        model.reset_state_of(&bstate)?;
        model.log_vram("建 batch state 后");
        let mut prompts: Vec<Vec<u32>> = Vec::with_capacity(slots);
        for s in 0..slots {
            let toks = tokenizer.encode(prompt_for(s, repeat).as_bytes())?;
            prompts.push(toks[..toks.len().saturating_sub(1)].to_vec());
        }
        let (pmin, pmax) = prompts.iter().fold((usize::MAX, 0usize), |(a, b), p| {
            (a.min(p.len()), b.max(p.len()))
        });
        log::info!("各槽 prompt 长度 {pmin}..{pmax} tok（互不相同），pad_to={pad_to}",);
        let seeds = model.forward_seq_batch_padded(&mut bstate, &prompts, pad_to)?;
        // seeds 校验和：分块 prefill（PREFILL_SLOTS）与单批 prefill 应逐位一致
        let seed_sum: u64 = seeds.iter().map(|&v| v as u64).sum();
        let seed_xor: u64 = seeds.iter().fold(0u64, |a, &v| a ^ (v as u64));
        log::info!("prefill seeds sum={seed_sum:#x} xor={seed_xor:#x}");
        model.log_vram("prefill 后");
        let mut seg_seed_next = seeds.clone();
        for _ in 0..warmup {
            let ticket =
                model.submit_sample_selfloop_batch(&mut bstate, &seg_seed_next, ntok, &sp)?;
            let outs = model.collect_sample_selfloop_batch(ticket, slots)?;
            for (slot, out) in outs.iter().enumerate() {
                if let Some(last) = out.last() {
                    seg_seed_next[slot] = *last;
                }
            }
        }
        model.log_vram("self-loop 图捕获后");
        let t1 = Instant::now();
        let mut toks_total = 0usize;
        // 解码 token 指纹（sum/xor）：**质量 A/B 的判据**——TOPK=1 贪心 + 固定 seed 时与路径无关，
        // 数值口径改动（如 LOWRANK_GEMM / IMMA 开关）若改变了判读，这两位立刻不同。
        let (mut tok_sum, mut tok_xor) = (0u64, 0u64);
        // 逐槽累积的完整 token 序列（供 `DUMP_TOK` 对拍；SEGS 段拼起来）。
        let mut all_tokens: Vec<Vec<u32>> = vec![Vec::new(); slots];
        for _ in 0..segs {
            let ticket =
                model.submit_sample_selfloop_batch(&mut bstate, &seg_seed_next, ntok, &sp)?;
            let outs = model.collect_sample_selfloop_batch(ticket, slots)?;
            for (slot, out) in outs.iter().enumerate() {
                if let Some(last) = out.last() {
                    seg_seed_next[slot] = *last;
                }
                all_tokens[slot].extend_from_slice(out);
            }
            for &t in outs.iter().flat_map(|o| o.iter()) {
                tok_sum = tok_sum.wrapping_add(t as u64);
                tok_xor ^= t as u64;
            }
            toks_total += outs.iter().map(|o| o.len()).sum::<usize>();
        }
        let b_secs = t1.elapsed().as_secs_f64();
        let batch_tok_s = toks_total as f64 / b_secs;
        let batch_prefill_ms = t1.duration_since(t0).as_millis();
        log::info!(
            "[批量 B={}] prefill {} ms | decode {} tok {} ms = {:.1} tok/s（聚合）",
            slots,
            batch_prefill_ms,
            toks_total,
            (b_secs * 1000.0) as u128,
            batch_tok_s
        );
        log::info!(
            "[批量 B={}] decode tokens sum={tok_sum:#x} xor={tok_xor:#x}",
            slots
        );
        // `DUMP_TOK=<路径>`：把每槽每步的 token id 逐行落盘（`slot step token`），供两条
        // 数值路径做**逐 token 对拍**（分叉位置/分叉率）——指纹只能判「是否相同」，
        // 判「差多少、从第几个 token 开始差」必须逐 token 比。
        if let Ok(path) = std::env::var("DUMP_TOK") {
            use std::io::Write as _;
            let f = std::fs::File::create(&path)?;
            let mut w = std::io::BufWriter::new(f);
            for (slot, out) in all_tokens.iter().enumerate() {
                for (step, &t) in out.iter().enumerate() {
                    writeln!(w, "{slot} {step} {t}")?;
                }
            }
            log::info!("[批量 B={}] tokens dumped -> {path}", slots);
        }

        // —— STEP=1：`batch_step_sample` 逐步路径对拍 ——
        //
        // 为什么要这条路径：自循环（`submit_sample_selfloop_batch`）一次提交跑完 n 轮、
        // 全程无主机介入，采样结果在 `end_batch()` **之后**才下载；逐步路径每步回主机，
        // 是客户端并发池的实际形态（stop 串判定 / 流式增量都在主机侧做）。两条路径在
        // 「同一状态 + 同一 seed 口径 + 同一惩罚」下必须产出**逐位相同**的 token 序列，
        // 指纹不同即说明逐步路径的采样读数有问题（历史上就是 download 夹在
        // `begin_batch()`/`end_batch()` 之间：Vulkan 上采样 dispatch 只是被记录进持久
        // command buffer，而 download 另开 command buffer 立即提交 → 读到上一批的陈旧值）。
        //
        // 惩罚历史在此固定为零（`hist_len` 恒 0，hist 只是一块零值占位缓冲），把上面那个
        // 正确性问题与「逐 token 追加历史需要的部分上传」解耦；`PENALTY=1` 才走追加路径。
        if step_mode {
            // PROBE=1：只跑一步批量前向并下载 logits，把「前向算错」与「采样读数错」
            // 解耦——`batch_step_logits` 与 `batch_step_sample` 共用同一前向核心。
            // 期望：末位 argmax == 自循环路径该槽的首个 token。
            if std::env::var("PROBE").is_ok_and(|v| v != "0") {
                model.reset_state_of(&bstate)?;
                let p_seeds = model.forward_seq_batch_padded(&mut bstate, &prompts, pad_to)?;
                let p_feed: Vec<Vec<u32>> = p_seeds.iter().map(|&t| vec![t]).collect();
                let p_vocab = model.info().num_vocab;
                let lg = model.batch_step_logits(&mut bstate, &p_feed)?;
                for s in 0..slots {
                    let row = &lg[s * p_vocab..(s + 1) * p_vocab];
                    let mut idx: Vec<usize> = (0..p_vocab).collect();
                    idx.sort_by(|&a, &b| row[b].total_cmp(&row[a]));
                    log::info!(
                        "[PROBE] slot{s} feed={} argmax={} top3={:?} logits={:.3} {:.3} {:.3}",
                        p_feed[s][0],
                        idx[0],
                        &idx[..3],
                        row[idx[0]],
                        row[idx[1]],
                        row[idx[2]]
                    );
                }
                // `DUMP_LG=<路径>`：把整行 logits 逐元素落盘（`slot idx value`），供两档做
                // **采样无关**的误差对拍。分叉率会被贪心级联放大到饱和（见 参考 §3.3），
                // 只有逐元素误差能分开「fp16 舍入级扰动」与「前向算错」。
                if let Ok(path) = std::env::var("DUMP_LG") {
                    use std::io::Write as _;
                    let f = std::fs::File::create(&path)?;
                    let mut w = std::io::BufWriter::new(f);
                    for s in 0..slots {
                        for (i, &v) in lg[s * p_vocab..(s + 1) * p_vocab].iter().enumerate() {
                            writeln!(w, "{s} {i} {v:.6}")?;
                        }
                    }
                    log::info!("[PROBE] logits dumped -> {path}");
                }
            }
            let st_t0 = Instant::now();
            // 复用同一 batch State（不另建，省显存）：重置 + 重新 prefill，保证与自循环
            // 路径从完全相同的状态出发。
            model.reset_state_of(&bstate)?;
            let seeds2 = model.forward_seq_batch_padded(&mut bstate, &prompts, pad_to)?;
            if seeds2 != seeds {
                log::warn!("[STEP] 两次 prefill 的 seeds 不一致，对拍前提不成立");
            }
            let prefill_ms = st_t0.elapsed().as_millis();

            // hist 行宽：默认只占 8（hist_len 恒 0，内核不会读到有效数据以外的位置）；
            // PENALTY=1 时要按自循环的口径复刻「段内已生成 token」，行宽须 ≥ 段长 ntok。
            let hist_stride = if penalty { ntok.max(8) } else { 8 };
            let hist = model.alloc_u32(slots * hist_stride)?;
            model.write_u32(hist, &vec![0u32; slots * hist_stride])?;
            // PENALTY=1 的语义与自循环逐字对齐：每段开头 hist_len 归零，段内第 r 步
            // hist_len = r（= 本段前 r 步采样出的 token，**不含**本步刚喂进去的那个），
            // 采样后立刻把新 token 追加到该行第 r 列（`write_u32_part` = 增量上传）。
            let mut hist_len = vec![0u32; slots];

            // seed 口径必须复刻自循环：段初基 = 该槽上一段的末 token，段内第 r 步
            // seed = 基 + r（见 `submit_sample_selfloop_batch` 的 `seeds_r`）。
            let mut feed = seeds2.clone();
            let mut st_sum = 0u64;
            let mut st_xor = 0u64;
            let mut st_total = 0usize;
            let mut all_tokens: Vec<Vec<u32>> = vec![Vec::new(); slots];
            let mut st_t1 = Instant::now();
            for seg in 0..warmup + segs {
                let base = feed.clone();
                if penalty {
                    // 段初清零，与 `submit_sample_selfloop_batch` 开头把 token_seq 整段
                    // upload 成零、且 sampler[7]=round 从 0 起完全同构。
                    hist_len.iter_mut().for_each(|l| *l = 0);
                }
                if seg == warmup {
                    st_t1 = Instant::now();
                }
                for r in 0..ntok {
                    let tokens: Vec<Vec<u32>> = feed.iter().map(|&t| vec![t]).collect();
                    let sps: Vec<SamplerParams> = (0..slots)
                        .map(|s| SamplerParams {
                            seed: base[s].wrapping_add(r as u32),
                            ..sp
                        })
                        .collect();
                    let out = model.batch_step_sample(
                        &mut bstate,
                        &tokens,
                        hist,
                        hist_stride,
                        &hist_len,
                        &sps,
                    )?;
                    for (s, &t) in out.iter().enumerate() {
                        feed[s] = t;
                        if penalty {
                            // 客户端池 `append_hist` 的形状：只写这一格，不重传整行。
                            model.write_u32_part(hist, s * hist_stride + r, &[t])?;
                            hist_len[s] = r as u32 + 1;
                        }
                        if seg >= warmup {
                            st_sum = st_sum.wrapping_add(t as u64);
                            st_xor ^= t as u64;
                            all_tokens[s].push(t);
                            st_total += 1;
                        }
                    }
                }
            }
            let step_secs = st_t1.elapsed().as_secs_f64();
            let step_toks = st_total as f64 / step_secs;
            step_readings = Some((step_toks, st_sum, st_xor));
            log::info!(
                "[STEP B={}] prefill {} ms | decode {} tok {} ms = {:.1} tok/s（聚合，逐token回主机）",
                slots,
                prefill_ms,
                st_total,
                (step_secs * 1000.0) as u128,
                step_toks
            );
            log::info!(
                "[STEP B={}] decode tokens sum={st_sum:#x} xor={st_xor:#x}",
                slots
            );
            log::info!(
                "[对拍 B={}] 自循环 sum={tok_sum:#x} xor={tok_xor:#x} | STEP sum={st_sum:#x} xor={st_xor:#x} => {}",
                slots,
                if (tok_sum, tok_xor) == (st_sum, st_xor) {
                    "一致"
                } else {
                    "**不一致**（逐步路径采样读数有问题）"
                }
            );
            if let Ok(path) = std::env::var("DUMP_TOK_STEP") {
                use std::io::Write as _;
                let f = std::fs::File::create(&path)?;
                let mut w = std::io::BufWriter::new(f);
                for (slot, out) in all_tokens.iter().enumerate() {
                    for (step, &t) in out.iter().enumerate() {
                        writeln!(w, "{slot} {step} {t}")?;
                    }
                }
                log::info!("[STEP B={}] tokens dumped -> {path}", slots);
            }
        }

        // —— SLOT_BENCH=1：把一份状态分发给多个槽的两条做法对比 ——
        //
        // 场景是会话 fork / 同一 prompt 多路采样。老做法：`state_slot_back` 下载 + 逐槽
        // `state_slot_load` 回灌（每个目标 2× 单槽状态大小的 PCIe 往返，且逐层同步等待）；
        // 新做法：`state_slot_broadcast` 全程设备内拷贝。等价性（逐位相同）由
        // `tests/batch_parity.rs` 的 `device_slot_broadcast_matches_host_roundtrip` 钉住，
        // 这里只量时间。
        if std::env::var("SLOT_BENCH").is_ok_and(|v| v != "0") {
            let targets: Vec<usize> = (1..slots).collect();
            let t0 = Instant::now();
            let d0 = model.state_slot_back(&bstate, 0)?;
            for &s in &targets {
                model.state_slot_load(&bstate, s, &d0)?;
            }
            let host_ms = t0.elapsed().as_secs_f64() * 1000.0;
            let t1 = Instant::now();
            model.state_slot_broadcast(&bstate, 0, &targets)?;
            let dev_ms = t1.elapsed().as_secs_f64() * 1000.0;
            slot_host_ms = Some(host_ms);
            slot_dev_ms = Some(dev_ms);
            slot_bytes = Some(d0.len() * 4);
            log::info!(
                "[SLOT_BENCH] 分发到 {} 个槽：宿主往返 {host_ms:.2} ms | 设备内广播 {dev_ms:.2} ms | \
                 单槽状态 {} B（宿主往返多走 ≈ {} B PCIe）",
                targets.len(),
                d0.len() * 4,
                d0.len() * 4 * targets.len() * 2,
            );
        }

        // —— BUCKET_BENCH=1：长度桶预热到底把代价挪去哪了 ——
        //
        // `SeqBuffers` 按 `bt = 批量 × T_pad` 定容且**只存一份**，所以每换一个长度就整表
        // 重建一次（`clear_cache` + 45 个缓冲重分配）。这四档量的是：
        //   ① 冷桶首请求（表还被别的长度占着）—— 用户等的时间含重建
        //   ② 同桶重复 —— 不重建，该桶 prefill 的下界
        //   ③ 预热该桶的成本（启动期付）
        //   ④ 预热后的首请求 —— 与 ① 相减 = 用户可见延迟的省头
        // 结论要说清楚：预热**不减少重建次数**，只是把「用户等的那一次」换成「启动期的一次」。
        // 跨桶抖动（长度来回变）每次都得重建，预热救不了 —— 要么固定 pad_to，要么按桶各留模型。
        if std::env::var("BUCKET_BENCH").is_ok_and(|v| v != "0") {
            let novel = pad_to + 128; // 没建过表的桶
            let tc = Instant::now();
            model.forward_seq_batch_padded(&mut bstate, &prompts, novel)?;
            let cold_ms = tc.elapsed().as_secs_f64() * 1000.0;
            let ts = Instant::now();
            model.forward_seq_batch_padded(&mut bstate, &prompts, novel)?;
            let steady_ms = ts.elapsed().as_secs_f64() * 1000.0;
            // 切回 pad_to，让下一次打 novel 的条件与①一致（表又被换成别的长度）。
            model.forward_seq_batch_padded(&mut bstate, &prompts, pad_to)?;
            let tw = Instant::now();
            let costs = model.warmup_len_buckets(&mut bstate, &[novel])?;
            let warm_cost_ms = tw.elapsed().as_secs_f64() * 1000.0;
            let tr = Instant::now();
            model.forward_seq_batch_padded(&mut bstate, &prompts, novel)?;
            let warmed_req_ms = tr.elapsed().as_secs_f64() * 1000.0;
            log::info!(
                "[BUCKET_BENCH] 桶 {novel} tok：冷桶首请求 {cold_ms:.1} ms | 同桶重复 {steady_ms:.1} ms | \
                 预热成本 {warm_cost_ms:.1} ms + 预热后首请求 {warmed_req_ms:.1} ms \
                 => 用户可见延迟省 {:.1} ms（重建次数没变，只是从请求里挪到启动期）",
                cold_ms - warmed_req_ms
            );
            bucket_readings = Some(serde_json::json!({
                "bucket_t_pad": novel,
                "cold_first_request_ms": cold_ms,
                "same_bucket_repeat_ms": steady_ms,
                "warmup_cost_ms": costs.first().map(|(_, m)| *m),
                "warmup_wall_ms": warm_cost_ms,
                "warmed_first_request_ms": warmed_req_ms,
                "user_visible_saving_ms": cold_ms - warmed_req_ms,
            }));
        }

        // —— BENCH_JSON：把「读数」和「门控」绑成一条记录再落盘 ——
        // 单独打印 tok/s 没有意义：同会话/跨会话、张量核开关开没开，都能把结论反转。
        bench_json_line(serde_json::json!({
            "kind": "batch_decode_bench",
            "ts": chrono::Local::now().to_rfc3339(),
            "session": std::env::var("BENCH_SESSION").unwrap_or_else(|_| "solo".into()),
            "round": std::env::var("BENCH_ROUND").unwrap_or_else(|_| "0".into()),
            "model": model_path,
            "vocab": vocab_path,
            "params": {
                "slots": slots, "segs": segs, "ntok": ntok, "pad_to": pad_to,
                "top_k": topk, "temp": temp, "prompt_repeat": repeat, "warmup": warmup,
                "skip_single": skip_single,
            },
            "gating": gating_json(),
            "readings": {
                "single_tok_s": single_tok_s,
                "batch_prefill_ms": batch_prefill_ms,
                "batch_tok_s": batch_tok_s,
                "batch_tokens_sum": format!("{tok_sum:#x}"),
                "batch_tokens_xor": format!("{tok_xor:#x}"),
                "prefill_seeds_sum": format!("{seed_sum:#x}"),
                "prefill_seeds_xor": format!("{seed_xor:#x}"),
                "step": step_readings.map(|(t, s, x)| serde_json::json!({
                    "tok_s": t, "tokens_sum": format!("{s:#x}"), "tokens_xor": format!("{x:#x}")
                })),
                "slot_broadcast": slot_host_ms.zip(slot_dev_ms).map(|(h, d)| serde_json::json!({
                    "targets": slots - 1, "host_roundtrip_ms": h, "device_broadcast_ms": d,
                    "bytes_per_slot": slot_bytes,
                })),
                "bucket_warmup": bucket_readings,
            },
        }));
    }

    Ok(())
}

/// 提示词池：32 条互不相同的开场（中文，长度相近但不相等）。
const PROMPT_POOL: [&str; 32] = [
    "你是小镇居民花子，咖啡馆店员。",
    "你是夜班铁路调度员，正在核对时刻表。",
    "你是图书馆管理员，负责整理旧书目录。",
    "你是海边灯塔的守护人，记录潮汐变化。",
    "你是一名中学物理老师，准备明天的实验课。",
    "你是中药铺的坐堂先生，替人看脉。",
    "你是城市规划局的新人，整理道路图纸。",
    "你是植物园的园艺师，照看温室里的兰花。",
    "你是旧书店老板，正在给旧书分类定价。",
    "你是山村民宿的主人，安排客人的行程。",
    "你是博物馆讲解员，准备青铜器展厅的稿子。",
    "你是面包房的学徒，凌晨四点开始和面。",
    "你是气象站的观测员，记录今天的风向。",
    "你是剧团的舞台监督，清点道具清单。",
    "你是修表匠，手里有一只停了二十年的怀表。",
    "你是自行车铺的师傅，给链条上油。",
    "你是社区医生，整理居民的健康档案。",
    "你是港口的海关关员，核对货运单据。",
    "你是乡村邮递员，今天的路线经过三个村。",
    "你是茶园的采茶人，判断今年的头春。",
    "你是档案馆的整理员，给旧照片编目。",
    "你是陶艺工坊的主人，准备开窑。",
    "你是天文台的助理，整理昨夜的观测数据。",
    "你是渔船的船长，检查出海的补给。",
    "你是老式电影院的放映员，检查胶片。",
    "你是花店的店员，给婚礼准备花束。",
    "你是巷口修鞋摊的师傅，今天下着小雨。",
    "你是少年宫的书法老师，批改学生的作业。",
    "你是小饭馆的厨师，研究一道新菜。",
    "你是林场的护林员，巡查今天的防火道。",
    "你是手工皮鞋店的学徒，学习量脚样。",
    "你是旧钟楼的维护工，检查齿轮。",
];

/// 第 `slot` 槽的提示词。**保证各槽文本互不相同**：首段按 `slot % 32` 取，
/// 尾段按 `(slot + slot/32*7) % 32` 取 —— 两者组合在 `slot < 32*32` 范围内唯一，
/// 覆盖 256 槽绰绰有余。
fn prompt_for(slot: usize, repeat: usize) -> String {
    let n = PROMPT_POOL.len();
    let head = PROMPT_POOL[slot % n];
    let body = PROMPT_POOL[(slot + slot / n * 7 + 1) % n];
    let mut s = String::with_capacity(head.len() + body.len() * (repeat + 1) + 8);
    s.push_str(head);
    for _ in 0..repeat {
        s.push_str(body);
    }
    s
}

/// 门控快照：把**会改数值路径或吞吐**的开关，连同「未设时走哪一档」一起记下来。
///
/// 值以 `(未设，默认: ` 开头 = 环境变量没设，走代码默认档。之所以连默认值都要写：
/// README §6.1 的老表正是因为只记了读数、没记门控，才让「SIMT 对 SIMT」被读成
/// 「Vulkan 追平了 CUDA 的张量核路径」。
fn gating_json() -> serde_json::Value {
    let knob = |name: &str, default_when_unset: &str| -> String {
        std::env::var(name).unwrap_or_else(|_| format!("(未设，默认: {default_when_unset})"))
    };
    serde_json::json!({
        "BACKEND": knob("BACKEND", "自动探测（本机为 vulkan 时走 Vulkan）"),
        "GEMV_IMMA": knob("GEMV_IMMA", "开 —— CUDA r/k/v 走 int8 张量核"),
        "LOWRANK_GEMM": knob("LOWRANK_GEMM", "开 —— CUDA 低秩四条走 int8 张量核"),
        "FFN_VALUE_GEMM": knob("FFN_VALUE_GEMM", "开 —— CUDA FFN value 走 int8 张量核"),
        // ★ Vulkan 低秩链的两级张量核档 + 门控绑定标记。缺这三条就会重演 2026-10-11 ①-b v1
        // 那次「两臂读数完全相同、其实两臂都没在 Vulkan 上跑」的空转。
        "LOWRANK_GEMM2": knob("LOWRANK_GEMM2", "开（batch>=32 且形状整除）—— Vulkan 低秩二级 coopmat fp16"),
        "LOWRANK_GEMM1": knob("LOWRANK_GEMM1", "开（随二级门）—— Vulkan 低秩一级 coopmat fp16，SIMT mid 臂整条退役"),
        "PROF_LOW": knob("PROF_LOW", "关 —— 每次 forward 第 0 层打一条 gemm/stage2/stage1/lr_rows 标记"),
        "GEMM_MIN_BLOCKS": knob("GEMM_MIN_BLOCKS", "32 —— coopmat 瓦片占用率档下限（0 关掉）"),
        "VK_IMMA_OPS": knob("VK_IMMA_OPS", "全开（含 rkv，2026-10-10 翻转）"),
        "VK_IMMA": knob("VK_IMMA", "开 —— Vulkan W8A8 张量核 GEMM；但受 `imma_ok` 的 batch 门约束，见下"),
        "VK_IMMA_NST": knob("VK_IMMA_NST", "按 batch 自动（≥ceil(batch/8) 的最小 2 的幂，上限 8）"),
        "VK_IMMA_BM": knob("VK_IMMA_BM", "按 M 自动（64/128 tile）"),
        "SKIP_GRAPH": knob("SKIP_GRAPH", "关 —— 自循环走图捕获/单批录制"),
        "SELFLOOP_CHUNK": knob("SELFLOOP_CHUNK", "内核默认分段"),
        "PREFILL_SLOTS": knob("PREFILL_SLOTS", "8"),
        "STEP": knob("STEP", "关 —— 只跑自循环路径"),
        "PENALTY": knob("PENALTY", "关 —— 无频率惩罚"),
        "SLOT_BENCH": knob("SLOT_BENCH", "关"),
        "ABLATE": knob("ABLATE", "空 —— 不消融任何 kernel"),
    })
}

/// 向 `BENCH_JSON` 指向的文件追加一行 JSON（JSONL）。未设该变量则什么都不做。
/// 写失败只 warn：记录层不该把基准跑断。
fn bench_json_line(rec: serde_json::Value) {
    let Ok(path) = std::env::var("BENCH_JSON") else {
        return;
    };
    use std::io::Write as _;
    match std::fs::OpenOptions::new()
        .create(true)
        .append(true)
        .open(&path)
    {
        Ok(f) => {
            let mut w = std::io::BufWriter::new(f);
            if let Err(e) = writeln!(w, "{rec}").and_then(|_| w.flush()) {
                log::warn!("BENCH_JSON 写入失败: {e}");
            }
        }
        Err(e) => log::warn!("BENCH_JSON 打开失败（{path}）: {e}"),
    }
}

fn init_log() {
    let _ = simplelog::TermLogger::init(
        log::LevelFilter::Info,
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    );
}
