//! 批量并发解码路径的**正确性**对拍（此前只存在于 `examples/batch_decode_bench.rs` 的
//! 一次性手工跑法，这里挪进 `cargo test` 可跑的用例）。
//!
//! 钉住的四件事，都对应历史上真实踩过的坑：
//!
//! 1. **batch prefill 的重置口径**：每轮从零态起步，同一批 prompt 连打两次必须逐位一致
//!    （`forward_seq_*_batch` 是「续跑」语义，重置由调用方负责）。
//! 2. **逐步路径的采样读数**：`batch_step_sample`（每步回主机，客户端并发池的实际形态）
//!    与 `submit_sample_selfloop_batch`（一次提交跑完整段，全程无主机介入）从同一状态出发
//!    必须产出**逐 token 相同**的序列。曾经的 bug 就是 download 夹在 `begin_batch()` /
//!    `end_batch()` 之间——Vulkan 上采样 dispatch 只被记录进持久 command buffer，而
//!    download 另开 command buffer 立即提交，于是读到上一批的陈旧值。
//! 3. **惩罚历史的部分上传**：`write_u32_part` 逐 token 追加（客户端池 `append_hist` 的
//!    形状）必须与自循环「整段 hist + hist_len=round」的口径等价。
//! 4. **贪心的可复现性**：同一路径连跑两遍逐 token 必须相同（抓「跨请求残留状态没清」
//!    这一类）。★ 这条只对 `top_k=1` 成立；随机采样下做不到，见
//!    [`stochastic_reproducibility_repro`]（`#[ignore]` 的复现用例）。
//! 5. **跨后端**：各后端**内部**两条批量路径必须一致；跨后端之间只报告不判等，实测数字
//!    与理由写在 [`cross_backend_batch_paths_agree`] 上。
//! 6. **设备内状态分发**：`state_slot_broadcast` 与「`state_slot_back` + `state_slot_load`」
//!    的宿主往返必须逐位相同，且只动目标槽（会话 fork / 同 prompt 多路采样走的就是这条）。
//!
//! 模型/词表文件缺失或建不出后端时，用例**显式跳过**（打印原因，不算失败）——没有 GPU
//! 权重的 CI 也能跑这份测试文件。
//!
//! 跑法（本仓 Rust 一律 `--release`）：
//! ```bash
//! cargo test --release --test batch_parity -- --nocapture --test-threads=1
//! MODEL_PATH=... VOCAB_JSON=... cargo test --release --test batch_parity
//! ```
//!
//! 跨后端对拍（改进项 2 的后半）默认**不跑**，因为它要求两侧同 SIMT 口径才有可比性
//! （张量核心开关不同，数值路径就不同），env 旋钮又基本是进程内 `OnceLock` 读的，只能由
//! 外部给。要跑：
//! ```bash
//! CROSS_BACKEND=1 GEMV_IMMA=0 VK_IMMA_OPS=... cargo test --release --test batch_parity \
//!     -- --nocapture --test-threads=1 cross_backend
//! ```

use std::path::Path;
use std::sync::{Mutex, MutexGuard, OnceLock};

use rwkv_rsv::backend::{BackendKind, create_backend, detect_backend};
use rwkv_rsv::gpu_model::{GpuModel, SamplerParams, State};
use rwkv_rsv::tokenizer::Tokenizer;

const DEFAULT_MODEL: &str =
    r"C:\work\ai00-x-dev\client\target\release\models\rwkv\rwkv7-3B-int8.st";
const DEFAULT_VOCAB: &str = r"C:\work\ai00-x-dev\client\target\release\models\rwkv\vocab.json";

/// 各槽互不相同的提示词（同 prompt 会把每层残差压成同一条数值路径，测不出并发行为）。
const PROMPTS: [&str; 8] = [
    "你是小镇居民花子，咖啡馆店员。",
    "你是夜班铁路调度员，正在核对时刻表。",
    "你是图书馆管理员，负责整理旧书目录。",
    "你是海边灯塔的守护人，记录潮汐变化。",
    "你是一名中学物理老师，准备明天的实验课。",
    "你是中药铺的坐堂先生，替人看脉。",
    "你是城市规划局的新人，整理道路图纸。",
    "你是植物园的园艺师，照看温室里的兰花。",
];

const SLOTS: usize = 4;
const NTOK: usize = 8;
const WARMUP_SEGS: usize = 1;
const SEGS: usize = 2;

/// GPU 用例串行：并发两个模型会在同一张卡上抢显存与带宽，指纹本身不受影响但耗时翻倍，
/// 且小显存机器上直接 OOM——串行是这里唯一稳的形态。
fn gpu_lock() -> MutexGuard<'static, ()> {
    static L: OnceLock<Mutex<()>> = OnceLock::new();
    L.get_or_init(|| Mutex::new(()))
        .lock()
        .unwrap_or_else(|e| e.into_inner())
}

/// 后端就绪 + 模型可加载；任一条件不满足返回 `None`（调用方跳过）。
fn try_load_kind(kind: BackendKind) -> Option<(GpuModel, Tokenizer)> {
    let model_path = std::env::var("MODEL_PATH").unwrap_or_else(|_| DEFAULT_MODEL.to_string());
    let vocab_path = std::env::var("VOCAB_JSON").unwrap_or_else(|_| DEFAULT_VOCAB.to_string());
    if !Path::new(&model_path).exists() || !Path::new(&vocab_path).exists() {
        eprintln!("SKIP: 模型或词表文件不存在（{model_path} / {vocab_path}）");
        return None;
    }
    let tokenizer =
        Tokenizer::new(&std::fs::read_to_string(&vocab_path).ok()?).expect("词表必须能解析");
    let backend = match create_backend(kind) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("SKIP: 建不出 {kind:?} 后端：{e}");
            return None;
        }
    };
    match GpuModel::from_safetensors(backend, &model_path) {
        Ok(m) => Some((m, tokenizer)),
        Err(e) => {
            eprintln!("SKIP: 模型加载失败：{e}");
            None
        }
    }
}

/// 本机自动探测的后端（`detect_backend()` 自己读 `BACKEND` env：cuda / vulkan）。
fn try_load() -> Option<(GpuModel, Tokenizer)> {
    try_load_kind(detect_backend())
}

/// 每槽 prompt（去掉末 token，末 token 作为解码种子，与基准 example 同形）。
fn prompts(tokenizer: &Tokenizer) -> (Vec<Vec<u32>>, Vec<u32>) {
    let mut batch = Vec::with_capacity(SLOTS);
    let mut seeds = Vec::with_capacity(SLOTS);
    for (s, text) in PROMPTS.iter().take(SLOTS).enumerate() {
        // 槽间长度也错开：同一 prompt 重复 s+1 次，lens 不均匀才接近服务端实况。
        let text = text.repeat(s + 1);
        let toks = tokenizer
            .encode(text.as_bytes())
            .expect("中文提示词必须能编码");
        assert!(toks.len() >= 2, "prompt 太短，测不出 prefill");
        seeds.push(*toks.last().unwrap());
        batch.push(toks[..toks.len() - 1].to_vec());
    }
    (batch, seeds)
}

fn sampler(top_k: u32, frequency_penalty: f32) -> SamplerParams {
    SamplerParams {
        temperature: 1.0,
        top_k,
        top_p: 0.2,
        seed: 514,
        repetition_penalty: 1.0,
        frequency_penalty,
        ..Default::default()
    }
}

/// 自循环批量路径：一次提交跑完一整段，段间以前一段末 token 续跑。
fn run_selfloop(
    model: &mut GpuModel,
    state: &mut State,
    prompts: &[Vec<u32>],
    pad_to: usize,
    sp: &SamplerParams,
) -> Vec<Vec<u32>> {
    model.reset_state_of(state).unwrap();
    let mut feed = model
        .forward_seq_batch_padded(state, prompts, pad_to)
        .unwrap();
    let mut out = vec![Vec::new(); SLOTS];
    for seg in 0..WARMUP_SEGS + SEGS {
        let ticket = model
            .submit_sample_selfloop_batch(state, &feed, NTOK, sp)
            .unwrap();
        let outs = model.collect_sample_selfloop_batch(ticket, SLOTS).unwrap();
        for (s, o) in outs.iter().enumerate() {
            if seg >= WARMUP_SEGS {
                out[s].extend_from_slice(o);
            }
            if let Some(&last) = o.last() {
                feed[s] = last;
            }
        }
    }
    out
}

/// 逐步批量路径：每步回主机一次（客户端并发池的形状）。`penalty=true` 时按
/// 每步一个 token 增量追加惩罚历史（`write_u32_part`），与自循环的「段内 hist」对齐。
///
/// 采样 seed 口径复刻自循环：段初基 = 该槽上一段末 token，段内第 r 步 seed = 基 + r。
fn run_stepwise(
    model: &mut GpuModel,
    state: &mut State,
    prompts: &[Vec<u32>],
    pad_to: usize,
    sp: &SamplerParams,
    penalty: bool,
) -> Vec<Vec<u32>> {
    model.reset_state_of(state).unwrap();
    let mut feed = model
        .forward_seq_batch_padded(state, prompts, pad_to)
        .unwrap();
    let hist_stride = if penalty { NTOK.max(8) } else { 8 };
    let hist = model.alloc_u32(SLOTS * hist_stride).unwrap();
    model
        .write_u32(hist, &vec![0u32; SLOTS * hist_stride])
        .unwrap();
    let mut hist_len = vec![0u32; SLOTS];
    let mut out = vec![Vec::new(); SLOTS];
    for seg in 0..WARMUP_SEGS + SEGS {
        let base = feed.clone();
        if penalty {
            hist_len.iter_mut().for_each(|l| *l = 0);
        }
        for r in 0..NTOK {
            let tokens: Vec<Vec<u32>> = feed.iter().map(|&t| vec![t]).collect();
            let sps: Vec<SamplerParams> = (0..SLOTS)
                .map(|s| SamplerParams {
                    seed: base[s].wrapping_add(r as u32),
                    ..*sp
                })
                .collect();
            let got = model
                .batch_step_sample(state, &tokens, hist, hist_stride, &hist_len, &sps)
                .unwrap();
            for (s, &t) in got.iter().enumerate() {
                feed[s] = t;
                if penalty {
                    model
                        .write_u32_part(hist, s * hist_stride + r, &[t])
                        .unwrap();
                    hist_len[s] = r as u32 + 1;
                }
                if seg >= WARMUP_SEGS {
                    out[s].push(t);
                }
            }
        }
    }
    model.release_tensor(hist);
    out
}

fn fingerprint(out: &[Vec<u32>]) -> (u64, u64) {
    out.iter()
        .flat_map(|v| v.iter())
        .fold((0u64, 0u64), |(sum, xor), &t| {
            (sum.wrapping_add(t as u64), xor ^ (t as u64))
        })
}

/// 1：batch prefill 从零态起步必须可复现，且固定 `pad_to` 与自动取批内最大长度等价。
#[test]
fn batch_prefill_is_reset_deterministic() {
    let _guard = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        return;
    };
    let (batch, _) = prompts(&tokenizer);
    let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
    let mut state = model.create_batch_state(SLOTS).unwrap();

    let mut runs = Vec::new();
    for _ in 0..2 {
        model.reset_state_of(&state).unwrap();
        runs.push(
            model
                .forward_seq_batch_padded(&mut state, &batch, pad_to)
                .unwrap(),
        );
    }
    assert_eq!(
        runs[0], runs[1],
        "同一批 prompt 两次 prefill 的 seeds 必须一致"
    );

    // 不固定 pad_to 的公共入口走同一核心（lens 口径也一致 → seeds 必须相同）。
    model.reset_state_of(&state).unwrap();
    let unpadded = model.forward_seq_batch(&mut state, &batch).unwrap();
    assert_eq!(
        unpadded, runs[0],
        "forward_seq_batch 与 forward_seq_batch_padded(pad=批内最长) 必须等价"
    );
}

/// 2 + 3：贪心（top_k=1，与数值路径无关的判读）下，逐步路径与自循环路径逐 token 一致；
/// 带频率惩罚 + 逐 token 增量上传历史时仍须一致。
#[test]
fn stepwise_and_selfloop_paths_agree_greedy() {
    let _guard = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        return;
    };
    let (batch, _) = prompts(&tokenizer);
    let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
    let mut state = model.create_batch_state(SLOTS).unwrap();

    for (label, freq) in [("无惩罚", 0.0f32), ("频率惩罚 0.2", 0.2)] {
        let sp = sampler(1, freq);
        let looped = run_selfloop(&mut model, &mut state, &batch, pad_to, &sp);
        let stepped = run_stepwise(&mut model, &mut state, &batch, pad_to, &sp, freq > 0.0);
        assert_eq!(
            fingerprint(&looped),
            fingerprint(&stepped),
            "[{label}] 两条批量路径的 token 指纹必须一致"
        );
        for s in 0..SLOTS {
            assert_eq!(
                looped[s],
                stepped[s],
                "[{label}] slot{s} 逐 token 必须一致（首个分叉 index={:?}）",
                looped[s].iter().zip(&stepped[s]).position(|(a, b)| a != b)
            );
        }
    }
}

/// 4：**贪心可复现性**——同一路径连跑两遍逐 token 必须相同（抓「跨请求残留状态没清」
/// 这一类：状态没重置、常驻采样缓冲被上一段污染等）。
///
/// ⚠ 只用贪心（`top_k=1`）断言这条。**随机采样（`top_k>0`）做不到跨进程逐 token 复现**，
/// 2026-10-10 实测（复现用例见 [`stochastic_reproducibility_repro`]）：同一 example、同一
/// env 连跑两次，自循环与逐步两条路径的 token 序列**各自**都不同（`SKIP_GRAPH=1` 关掉
/// graph 重放也一样），而 `top_k=1` 时两侧 64 个 token 全一致、batch prefill 的 seeds 也
/// 逐位一致（用例 1）。这个形状指向「logits 有末位级 run-to-run 差异 ⇒ argmax 不放大、
/// 轮盘赌放大」，不是采样读数 bug（那个的表现是每步返回喂进去的 token）。
#[test]
fn greedy_tokens_are_reproducible() {
    let _guard = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        return;
    };
    let (batch, _) = prompts(&tokenizer);
    let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
    let mut state = model.create_batch_state(SLOTS).unwrap();

    let sp = sampler(1, 0.0);
    let stepped = run_stepwise(&mut model, &mut state, &batch, pad_to, &sp, false);
    assert_eq!(
        stepped,
        run_stepwise(&mut model, &mut state, &batch, pad_to, &sp, false),
        "逐步路径贪心必须可复现"
    );
    let looped = run_selfloop(&mut model, &mut state, &batch, pad_to, &sp);
    assert_eq!(
        looped,
        run_selfloop(&mut model, &mut state, &batch, pad_to, &sp),
        "自循环路径贪心必须可复现"
    );
}

/// 随机采样不可复现的**复现用例**。默认 `#[ignore]`：它现在必然失败，而失败本身就是记录。
/// 复看现状：`cargo test --release --test batch_parity -- --ignored --nocapture`。
#[test]
#[ignore = "已知不成立（2026-10-10 实测）：top_k>0 时同一路径两次跑逐 token 不同；归约顺序钉成确定之后再取消 ignore"]
fn stochastic_reproducibility_repro() {
    let _guard = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        return;
    };
    let (batch, _) = prompts(&tokenizer);
    let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
    let mut state = model.create_batch_state(SLOTS).unwrap();

    let sp = sampler(8, 0.0);
    let a = run_selfloop(&mut model, &mut state, &batch, pad_to, &sp);
    let b = run_selfloop(&mut model, &mut state, &batch, pad_to, &sp);
    let c = run_stepwise(&mut model, &mut state, &batch, pad_to, &sp, false);
    eprintln!(
        "自循环两次 {:?} vs {:?}；逐步一次 {:?}",
        fingerprint(&a),
        fingerprint(&b),
        fingerprint(&c)
    );
    assert_eq!(
        a, b,
        "同一路径同一 seed 的随机采样应当可复现（当前不可复现 ⇒ 见本用例 doc）"
    );
}

/// 5：设备内 slot 广播与「下载 + 回灌」的宿主往返必须给出**逐位相同**的状态，且只动
/// 目标槽（会话 fork / 同 prompt 多路采样的分发口径）。
#[test]
fn device_slot_broadcast_matches_host_roundtrip() {
    let _guard = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        return;
    };
    let (batch, _) = prompts(&tokenizer);
    let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
    let mut state = model.create_batch_state(SLOTS).unwrap();
    model.reset_state_of(&state).unwrap();
    model
        .forward_seq_batch_padded(&mut state, &batch, pad_to)
        .unwrap();

    let slot0 = model.state_slot_back(&state, 0).unwrap();
    // 前提：各槽状态本来互不相同，否则「拷过去了」根本没有判据。
    for s in 1..SLOTS {
        assert_ne!(
            model.state_slot_back(&state, s).unwrap(),
            slot0,
            "slot{s} 的初始状态应当与 slot0 不同（各槽 prompt 不同）"
        );
    }

    // slot1 走设备内广播，slot2 走宿主往返，两者必须一致。
    model.state_slot_broadcast(&state, 0, &[1]).unwrap();
    model.state_slot_load(&state, 2, &slot0).unwrap();
    let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<u32>>();
    let via_device = bits(&model.state_slot_back(&state, 1).unwrap());
    let via_host = bits(&model.state_slot_back(&state, 2).unwrap());
    assert_eq!(
        via_device,
        via_host,
        "设备内广播与宿主往返必须逐位相同（{} 个 f32）",
        via_device.len()
    );
    // 没列进目标的槽不受影响。
    assert_ne!(
        model.state_slot_back(&state, 3).unwrap(),
        slot0,
        "slot3 不在目标列表里，必须保持原状"
    );
}
///
/// 断言的是**各后端内部**两条批量路径必须一致（这是正确性口径，跨后端都该成立）；
/// 跨后端之间只**报告**贪心 token 指纹，不判等：2026-10-10 实测同 SIMT 口径
/// （`GEMV_IMMA=0 LOWRANK_GEMM=0`）下 cuda=`0xce5f5/0xfb9`、vulkan=`0xcd73a/0x7d96`
/// 已经分叉，而两侧内部 step-vs-selfloop 全等——即「跨后端逐位一致」从来不是本引擎的
/// 承诺（两侧是不同的 kernel 实现），把它写成断言只会得到一条永远红的 CI。
/// 跨后端（CUDA vs Vulkan）同 scenario 对拍。**默认跳过**，见文件头的口径说明。
#[test]
fn cross_backend_batch_paths_agree() {
    if !std::env::var("CROSS_BACKEND").is_ok_and(|v| v != "0") {
        eprintln!("SKIP: 未设 CROSS_BACKEND=1（跨后端对拍需要外部给齐 SIMT 口径 env 旋钮）");
        return;
    }
    let _guard = gpu_lock();
    let mut per_backend: Vec<(&'static str, (u64, u64))> = Vec::new();
    for (name, kind) in [("cuda", BackendKind::Cuda), ("vulkan", BackendKind::Vulkan)] {
        let Some((mut model, tokenizer)) = try_load_kind(kind) else {
            return;
        };
        let (batch, _) = prompts(&tokenizer);
        let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
        let mut state = model.create_batch_state(SLOTS).unwrap();
        let sp = sampler(1, 0.0);
        let looped = run_selfloop(&mut model, &mut state, &batch, pad_to, &sp);
        let stepped = run_stepwise(&mut model, &mut state, &batch, pad_to, &sp, false);
        for s in 0..SLOTS {
            assert_eq!(
                looped[s], stepped[s],
                "[{name}] 后端内部两条批量路径必须逐 token 一致（跨后端比较的前提）"
            );
        }
        let fp = fingerprint(&looped);
        eprintln!("[{name}] greedy tokens sum={:x} xor={:x}", fp.0, fp.1);
        per_backend.push((name, fp));
        drop(model);
    }
    let (a, b) = (per_backend[0], per_backend[1]);
    eprintln!(
        "[跨后端报告] {}={:x}/{:x} vs {}={:x}/{:x} => {}",
        a.0,
        a.1.0,
        a.1.1,
        b.0,
        b.1.0,
        b.1.1,
        if a.1 == b.1 {
            "逐位一致"
        } else {
            "分叉（预期：跨后端不保证逐位一致）"
        }
    );
}
