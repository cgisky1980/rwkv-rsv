//! 批量解码基准：隔离对比「单流自循环」vs「批量自循环」的每段时间。
//!
//! 用法：
//!   cargo run --release --example batch_decode_bench
//!   MODEL_PATH=... VOCAB_JSON=... SLOTS=8 SEGS=4 NTOK=32
//!   TOPK=50（0=无 top-k，走 top-p 兜底路径）；SAMPLE_TEMP 调高可制造平坦分布压测兜底多候选轮。
//!   PAD_TO=512（批量 prefill 的固定 T_pad）；SKIP_SINGLE=1 跳过单流基准（扫并发时省时间）。
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
    // 批量 prefill 的固定 T_pad（跨批复用 seq 缓冲；超限时 kernel 内会自动抬升一次）。
    let pad_to: usize = std::env::var("PAD_TO")
        .unwrap_or_else(|_| "512".into())
        .parse()?;
    // 跳过单流基准（扫并发曲线时单流值是常量，重复跑纯属浪费）。
    let skip_single = std::env::var("SKIP_SINGLE").is_ok_and(|v| v != "0");

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
        log::info!(
            "[单流] prefill {} ms | decode {} tok {} ms = {:.1} tok/s",
            t1.duration_since(t0).as_millis(),
            toks_total,
            t1.elapsed().as_millis(),
            toks_total as f64 / t1.elapsed().as_secs_f64()
        );
    }

    // —— 批量基准（每槽不同 prompt）——
    {
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
        log::info!(
            "[批量 B={}] prefill {} ms | decode {} tok {} ms = {:.1} tok/s（聚合）",
            slots,
            t1.duration_since(t0).as_millis(),
            toks_total,
            t1.elapsed().as_millis(),
            toks_total as f64 / t1.elapsed().as_secs_f64()
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

fn init_log() {
    let _ = simplelog::TermLogger::init(
        log::LevelFilter::Info,
        simplelog::Config::default(),
        simplelog::TerminalMode::Mixed,
        simplelog::ColorChoice::Auto,
    );
}
