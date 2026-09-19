//! 批量解码基准：隔离对比「单流自循环」vs「批量自循环」的每段时间。
//!
//! 用法：
//!   cargo run --release --example batch_decode_bench
//!   MODEL_PATH=... VOCAB_JSON=... SLOTS=8 SEGS=4 NTOK=32
//!   TOPK=50（0=无 top-k，走 top-p 兜底路径）；TEMP 调高可制造平坦分布压测兜底多候选轮。

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

    let vocab = std::fs::read_to_string(&vocab_path)?;
    let tokenizer = Tokenizer::new(&vocab)?;
    log::info!("加载模型: {model_path}");
    let backend = create_backend(detect_backend())?;
    let mut model = GpuModel::from_safetensors(backend, &model_path)?;
    log::info!("模型就绪 vocab={}", model.info().num_vocab);

    // 中文 prompt（默认 ≈90 tok；PROMPT_REPEAT 可调大）
    let prompt_text = "你是小镇居民花子，咖啡馆店员。".repeat(repeat);
    let prompt_tokens = tokenizer.encode(prompt_text.as_bytes())?;
    let seed_token = prompt_tokens[prompt_tokens.len() - 1];
    log::info!("prompt {} tok", prompt_tokens.len());

    log::info!("top_k={topk}（0=兜底 top-p 路径）");
    let sp = SamplerParams {
        temperature: temp,
        top_k: topk,
        top_p: 0.2,
        seed: 514,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        ..Default::default()
    };

    // —— 单流基准 ——
    {
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
        log::info!(
            "[单流] prefill {} ms | decode {} tok {} ms = {:.1} tok/s",
            t1.duration_since(t0).as_millis(),
            toks_total,
            t1.elapsed().as_millis(),
            toks_total as f64 / t1.elapsed().as_secs_f64()
        );
    }

    // —— 批量基准（slots 槽同 prompt）——
    {
        let t0 = Instant::now();
        let mut bstate = model.create_batch_state(slots)?;
        model.reset_state_of(&bstate)?;
        let mut prompts: Vec<Vec<u32>> = Vec::new();
        for _ in 0..slots {
            prompts.push(prompt_tokens[..prompt_tokens.len() - 1].to_vec());
        }
        let seeds = model.forward_seq_batch_padded(&mut bstate, &prompts, 512)?;
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
        let t1 = Instant::now();
        let mut toks_total = 0usize;
        for _ in 0..segs {
            let ticket =
                model.submit_sample_selfloop_batch(&mut bstate, &seg_seed_next, ntok, &sp)?;
            let outs = model.collect_sample_selfloop_batch(ticket, slots)?;
            for (slot, out) in outs.iter().enumerate() {
                if let Some(last) = out.last() {
                    seg_seed_next[slot] = *last;
                }
            }
            toks_total += outs.iter().map(|o| o.len()).sum::<usize>();
        }
        log::info!(
            "[批量 B={}] prefill {} ms | decode {} tok {} ms = {:.1} tok/s（聚合）",
            slots,
            t1.duration_since(t0).as_millis(),
            toks_total,
            t1.elapsed().as_millis(),
            toks_total as f64 / t1.elapsed().as_secs_f64()
        );
    }

    Ok(())
}

fn init_log() {
    let _ = simplelog::TermLogger::init(
        log::LevelFilter::Info,
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    );
}
