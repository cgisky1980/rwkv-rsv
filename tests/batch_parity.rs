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
//! 7. **长度桶预热**：`warmup_len_buckets` 是纯性能路径，预热过 / 没预热过的 seeds 与
//!    贪心 token 必须逐位一致（否则就是把首请求的代价换成了错的结果）。
//! 8. **fp16 模型的批量路径**：批量 prefill 必须可用且逐槽与单流口径一致（证明不是静默垃圾）；
//!    批量 **decode** 必须返回**带原因**的错误（缺 int8 句柄）而不是静默降级或崩；
//!    同架构 int8 模型批量必须可用，才证明归因落在权重格式上。
//!    ⚠️ 本用例初版把「没有批量 decode」写成了「没有批量前向路径」并据此断言 prefill 也该报错，
//!    两后端实测都把它推翻了（详见该用例 doc）。
//! 9. **跨波槽位**：批宽超过一个 tensor-core 波（`8·NST`=64 槽）时，第二波、第三波的槽必须
//!    与第一波逐位一致（全槽同 prompt + 贪心，零容差）——动 batch 门/tile 形状时的守门用例。
//! 10. **top-p 兜底循环的候选集**（`sample.comp` 里 mask 的读/置/清路径）：上面所有采样用例
//!     都走贪心（`top_k=1`），而贪心**根本不进这个循环** ⇒ 一条都钉不住 mask。mask 从稠密档
//!     压成位图档（2026-10-10，`MASK_WORD_STRIDE`）之后，「位偏移/行步长算错」的具体表现是
//!     **别的槽的位被当成本槽的位**或**上一轮的位没清干净** ⇒ 高概率候选被误标已选、`g_cutoff`
//!     落到尾部 ⇒ 轮盘赌能选到概率极低的 token。判据不依赖 RNG（随机采样做不到逐 token 复现，
//!     见用例 4 的 ★ 注）：采到的 token 必须落在宿主侧按同一口径算出的 top-p 候选集内，
//!     见 [`top_p_sampled_tokens_stay_inside_the_nucleus`]。
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
    /// `(后端名, (sum, xor), 各槽 token 序列)`
    type BackendRun = (&'static str, (u64, u64), Vec<Vec<u32>>);
    let mut per_backend: Vec<BackendRun> = Vec::new();
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
        per_backend.push((name, fp, looped));
        drop(model);
    }
    let (a, b) = (&per_backend[0], &per_backend[1]);
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
    // **分叉定位**：只报「分叉」没用，要报**从第几步分叉**——这才区分得开两种成因：
    // 首步（step 0）就分叉 = 前向/logits 层面就不一致（数值口径没对齐，值得查）；
    // 首步一致、后面才分叉 = 前向数值相邻、采样把罕见的单点翻转放大成序列分叉
    // （README §6.1 记的就是这种，属预期行为，不是缺陷）。
    if a.1 != b.1 {
        let mut first_divergence = usize::MAX;
        let mut diverged_slots = 0usize;
        let mut diverged_tokens = 0usize;
        for s in 0..SLOTS {
            let (x, y) = (&a.2[s], &b.2[s]);
            let n = x.len().min(y.len());
            let mut d = 0usize;
            while d < n && x[d] == y[d] {
                d += 1;
            }
            if d < n || x.len() != y.len() {
                diverged_slots += 1;
                diverged_tokens += n - d;
                first_divergence = first_divergence.min(d);
            }
        }
        eprintln!(
            "[跨后端分叉定位] 首个分叉步={first_divergence}（0 = 前向层面就不一致）\
             分叉槽={diverged_slots}/{SLOTS} 分叉 token={diverged_tokens}/{}\
             长度 {} vs {}",
            SLOTS * NTOK,
            a.2.iter().map(|v| v.len()).sum::<usize>(),
            b.2.iter().map(|v| v.len()).sum::<usize>(),
        );
    }
}

/// 长度桶预热**不得改变结果**（改进项 8 的预热那半）。
///
/// `warmup_len_buckets` 干的事是把 `SeqBuffers`（按 `bt = 批量 × T_pad` 定容的 45 个缓冲）
/// 提前建好，让首个真实请求不付整表重建的代价。这是纯性能路径，所以判据只有一条：
/// 预热过 / 没预热过，同一批 prompt 的 seeds 与贪心 token 序列必须**逐位一致**
/// （预热用的提示词是 `t_pad` 本身，与真实请求同宽 ⇒ 命中同一个 `bt`）。
/// 顺带打印每桶耗时，供「值不值得在启动期付」的判断；时间不断言（本机 ±3% 常态）。
#[test]
fn warmup_buckets_do_not_change_results() {
    let _guard = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        return;
    };
    let (batch, _) = prompts(&tokenizer);
    let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
    let sp = sampler(1, 0.0);

    // 对照：从不预热的 state。
    let mut s_cold = model.create_batch_state(SLOTS).unwrap();
    let seeds_cold = model
        .forward_seq_batch_padded(&mut s_cold, &batch, pad_to)
        .unwrap();
    let tok_cold = run_selfloop(&mut model, &mut s_cold, &batch, pad_to, &sp);

    // 预热多个桶（含一个比 pad_to 更长的，模拟真实服务里的长 prompt 桶）。
    let mut s_warm = model.create_batch_state(SLOTS).unwrap();
    let buckets = [pad_to, pad_to + 37, 0usize];
    let t0 = std::time::Instant::now();
    let costs = model.warmup_len_buckets(&mut s_warm, &buckets).unwrap();
    let warm_ms = t0.elapsed().as_secs_f64() * 1000.0;
    assert_eq!(
        costs.len(),
        2,
        "warmup_len_buckets 应跳过 t_pad<1 的桶，其余逐桶给耗时"
    );

    let seeds_warm = model
        .forward_seq_batch_padded(&mut s_warm, &batch, pad_to)
        .unwrap();
    assert_eq!(
        seeds_cold, seeds_warm,
        "预热后同批 prompt 的 seeds 必须与未预热逐位一致"
    );
    let tok_warm = run_selfloop(&mut model, &mut s_warm, &batch, pad_to, &sp);
    assert_eq!(
        tok_cold, tok_warm,
        "预热不得改变贪心 token 序列（逐槽逐 token）"
    );
    let per: Vec<String> = costs
        .iter()
        .map(|(t, ms)| format!("{t}tok:{ms:.1}ms"))
        .collect();
    eprintln!(
        "[BUCKET] 预热 {} 个桶共 {warm_ms:.1} ms（{}）—— 这笔开销从首请求挪到启动期",
        costs.len(),
        per.join(" ")
    );
}

/// fp16 模型走**批量 decode** 时**必须给出带原因的错误**，而不是静默降级或崩（改进项 4）。
///
/// 事实口径（2026-10-10 实测订正，两后端一致）：批量前向分两段，只有 decode 段要 int8——
/// 批量 **prefill** 走 seq 并行核，读 fp16/fp32 权重照常工作；批量 **decode** 的融合 r/k/v 核
/// 只有 int8 变体（`gemv_int8_rkv_stage1_batch`），走到那里返回
/// `batch 并发路径要求 r/k/v 为 int8 量化权重（fp16 模型暂不支持）`。
/// 所以本用例钉四条：① 批量 prefill 可用，且**每槽状态与单流逐位一致**（两后端实测
/// `to_bits` 全等 ⇒ 「pad 行不进 state」这条契约成立，证明它不是静默垃圾）；
/// ② `forward_seq_batch_padded` 的返回值是**各槽提示词的末 token**（解码种子），
///    **不是** next-token 预测——本用例初版把它当成预测值去比单流 argmax，两后端都「失败」，
///    那是用例口径错，不是路径 bug（种子 `[11098,16417]` vs 单流 next `[19137,10080]`）；
/// ③ 批量 decode 显式拒绝且原因写明 int8；④ 同架构 int8 模型批量可用（归因落在权重格式）。
/// 这条 API 存在的意义就是让消费方**一眼分清**两种前提失败：
/// 「后端缺批量算子」（只能升级后端）vs「模型没有 int8 句柄」（换量化权重即可）——
/// 裸 bool 会让客户端池静默回落到逐槽串行，事后无从判断该改哪边。
///
/// ⚠️ 早前的 README 与本用例口径都写成「fp16 模型没有批量前向路径」，把 prefill 也说成了
/// 不行；消费方照那句决策会连批量 prefill 都退成逐槽串行。
///
/// 用 `router-0.1B.st` / `router-0.1B-int8.st`（同架构、加载快）；缺文件则显式跳过。
/// 单流路径在 fp16 上必须照常工作 —— 这是「换 int8 权重」这条指引成立的前提。
///
/// ⚠️ 这是本文件**最慢**的一个测试：两次 `CudaBackend::new()` 的首次 NVRTC 内核编译占了几乎
/// 全部时间（模型本身只有 0.1B，加载几秒）。它测的是**拒绝路径的口径**，与吞吐无关，
/// 只想跑批量对拍时用 `--skip fp16_batch_path` 跳过。Vulkan 后端没有 NVRTC，明显更快。
#[test]
fn fp16_batch_path_reports_reason_not_silently() {
    let _guard = gpu_lock();
    let vocab_path = std::env::var("VOCAB_JSON").unwrap_or_else(|_| DEFAULT_VOCAB.to_string());
    let dir = Path::new(&DEFAULT_MODEL).parent().map(|p| p.to_path_buf());
    let Some(dir) = dir else { return };
    let fp16 = std::env::var("FP16_MODEL")
        .unwrap_or_else(|_| dir.join("router-0.1B.st").display().to_string());
    let int8 = std::env::var("INT8_SMALL_MODEL")
        .unwrap_or_else(|_| dir.join("router-0.1B-int8.st").display().to_string());
    if !Path::new(&fp16).exists() || !Path::new(&int8).exists() || !Path::new(&vocab_path).exists()
    {
        eprintln!("SKIP: 缺少 fp16/int8 小模型或词表（{fp16} / {int8}）");
        return;
    }
    let kind = detect_backend();
    let vocab = std::fs::read_to_string(&vocab_path).unwrap();
    let tokenizer = Tokenizer::new(&vocab).unwrap();
    let (batch, _) = prompts(&tokenizer);
    let two = &batch[..2];
    let pad_to = two.iter().map(|p| p.len()).max().unwrap();

    // —— fp16：批量路径报「需要 int8」，单流路径照常 ——
    let mut m = GpuModel::from_safetensors(create_backend(kind).unwrap(), &fp16).unwrap();
    assert!(
        !m.supports_batch_decode(),
        "fp16 模型不应报告支持批量并发路径"
    );
    let blocker = m
        .batch_decode_blocker()
        .expect("fp16 模型的 blocker 必须有内容");
    eprintln!("[FP16] blocker = {blocker}");
    assert!(
        blocker.contains("int8"),
        "fp16 模型的阻塞原因必须指向 int8 权重（而不是后端缺算子），实测：{blocker}"
    );
    let mut single = m.create_state().unwrap();
    let argmax = |l: &[f32]| -> u32 {
        l.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i as u32)
            .unwrap_or(u32::MAX)
    };
    // 单流逐槽的参照值：下一 token（argmax）**加上整份状态**。每次从零态起步，槽间互不串味。
    let mut next_single = Vec::with_capacity(2);
    let mut state_single: Vec<Vec<f32>> = Vec::with_capacity(2);
    for p in two {
        m.reset_state_of(&single).unwrap();
        let logits = m
            .forward_seq_with_state(&mut single, p)
            .expect("fp16 模型的单流路径必须可用（否则「换 int8 权重」这条指引无意义）");
        assert_eq!(
            logits.len(),
            m.info().num_vocab,
            "单流 logits 长度应为词表大小"
        );
        next_single.push(argmax(&logits));
        state_single.push(m.state_back(&single).expect("单流状态下载必须可用"));
    }
    // 实测口径（2026-10-10，两后端一致）：要求 int8 的是**批量 decode** 的融合 r/k/v 核
    // （`forward_layer_batch`），而**批量 prefill** 走 seq 并行核、读 fp16/fp32 权重照常工作。
    // 早前 README/本用例把这条写成「fp16 没有批量前向路径」，是把 prefill 也说成了不行——
    // 消费方照那句做决策会连批量 prefill 都回退成逐槽串行。
    match m.create_batch_state(2) {
        Ok(mut bstate) => {
            m.reset_state_of(&bstate).unwrap();
            let seeds = m
                .forward_seq_batch_padded(&mut bstate, two, pad_to)
                .expect("fp16 模型的批量 prefill 应可用（seq 并行核不依赖 int8 句柄）");
            // ★ 返回值是**各槽提示词的末 token**（解码种子），不是 next-token 预测。
            // 本用例初版把它当成预测值去和单流 argmax 比，两边必然不同——那是**用例的口径错**，
            // 不是路径的 bug（两后端读数一致：种子 = 提示词末位）。
            let last_tok: Vec<u32> = two.iter().map(|p| *p.last().unwrap()).collect();
            assert_eq!(
                seeds, last_tok,
                "批量 prefill 必须返回各槽提示词的末 token 作为解码种子"
            );
            // 「不是静默垃圾」的真正判据在这里：**批量 prefill 之后每槽的状态**必须与单流逐槽
            // 一致。「pad 行不进 state」这条契约比种子看不见、比长度看不见，只有比状态看得见。
            for (i, a) in state_single.iter().enumerate() {
                let b = m
                    .state_slot_back(&bstate, i)
                    .expect("批量状态单槽下载必须可用");
                assert_eq!(
                    b.len(),
                    a.len(),
                    "slot {i}：单流与批量槽状态的布局长度必须一致（slot_back/slot_load 的契约）"
                );
                let bitwise = a.iter().zip(&b).all(|(x, y)| x.to_bits() == y.to_bits());
                let max_abs = a
                    .iter()
                    .zip(&b)
                    .fold(0f32, |acc, (x, y)| acc.max((x - y).abs()));
                let scale = a.iter().fold(0f32, |acc, x| acc.max(x.abs())).max(1e-6);
                let rel = max_abs / scale;
                eprintln!(
                    "[FP16] slot {i} 批量 vs 单流状态：逐位一致={bitwise} max_abs={max_abs:.3e} rel={rel:.3e}"
                );
                assert!(
                    bitwise,
                    "slot {i}：批量 prefill 之后的状态必须与单流**逐位**一致（2026-10-10 两后端实测全等）；\
                     现在 max_abs={max_abs:.3e} rel={rel:.3e} ⇒ prefill 的数值口径变了，先归因再谈放宽"
                );
            }
            eprintln!(
                "[FP16] 批量 prefill OK seeds={seeds:?}（=提示词末位）；单流 next={next_single:?}；\
                 状态与单流对齐见上面逐槽读数"
            );
            // `SelfloopTicket` 没有 Debug，所以拿不到 `expect_err`（它要求 Ok 侧可调试）。
            match m.submit_sample_selfloop_batch(&mut bstate, &seeds, 4, &sampler(1, 0.0)) {
                Err(err) => assert!(
                    err.to_string().contains("int8"),
                    "fp16 模型的批量 decode 必须显式拒绝且原因写明 int8（不许静默产出垃圾），\
                     实际: {err}"
                ),
                Ok(_) => panic!("fp16 模型的批量 decode 竟然成功了——门控失效或被绕过"),
            }
            eprintln!("[FP16] 批量 decode 显式拒绝 ✓");
        }
        Err(e) => assert!(
            e.to_string().contains("int8"),
            "批量状态都建不起来，但原因必须写明，实际: {e}"
        ),
    }

    // —— int8（同架构）：批量路径可用，证明确实是权重格式这一条前提在起作用 ——
    let mut m2 = GpuModel::from_safetensors(create_backend(kind).unwrap(), &int8).unwrap();
    if !m2.supports_batch_decode() {
        // 后端本身没有批量算子时，fp16 侧的 blocker 会是「后端缺算子」而不是「缺 int8」，
        // 上面那条断言已经把这种情形挡掉了；这里只做 int8 侧的说明性跳过。
        eprintln!("SKIP: 该后端未实现批量算子，int8 侧对照不做归因结论");
        return;
    }
    let mut s2 = m2.create_batch_state(2).unwrap();
    m2.reset_state_of(&s2).unwrap();
    let seeds = m2
        .forward_seq_batch_padded(&mut s2, two, pad_to)
        .expect("int8 模型的批量 prefill 应成功");
    assert_eq!(seeds.len(), 2);
    eprintln!("[INT8] 批量 prefill OK seeds={seeds:?}");
}

/// 9：**批宽跨过「一波」边界时，第二波的槽必须和第一波算得一模一样**（改进项 5 的守门用例）。
///
/// 为什么要单独钉这条：Vulkan 的 W8A8 张量核 GEMM 每 block 只覆盖 `8·NST` 个槽（NST 构建期
/// 只编 1/2/4/8 四档 ⇒ 一波 64 槽），批宽超过一波时靠 `grid.z = ceil(batch / (8·NST))` 起
/// 第二波、第三波。CUDA 侧也是按 BM/BN tile 分块。**波与波之间唯一的差别就是槽行的地址偏移**，
/// 一旦某处的行步长用了「本波内槽号」而不是「全局槽号」，后果是批宽 ≤64 时全对、>64 时后半
/// 槽静默错——正是最难被吞吐基准发现、又最容易在上线后被用户报出来的那类 bug（2026-10-10 的
/// 同会话扫描实测：Vulkan 每步耗时 B=64 68ms → B=128 299ms，因为 `imma_ok` 的 batch 门把
/// >64 整段踢回 SIMT 反量化路径；把门放开就是在动这段形状，所以先把这条钉上）。
///
/// 判据不含任何数值容差：**全槽同 prompt + 同 seed + 贪心（`top_k=1`）** ⇒ 每条槽是同一份计算
/// 的 72 份拷贝，token 序列必须逐位相同。72 = 64 + 8 ⇒ 第二波是**不满波**，越界读写也一并覆盖。
/// 贪心是这里唯一正确的判读口径：随机采样连跑两遍都做不到逐 token 复现（见用例 4 的 ★ 注）。
#[test]
fn slots_beyond_the_first_wave_match() {
    const WAVE_SLOTS: usize = 72;
    const PAD_TO: usize = 96;
    let _gpu = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        eprintln!("SKIP: 模型或后端不可用，跨波槽位对拍跳过");
        return;
    };

    // 同一条 prompt 喂满所有槽：任何槽间差异都只可能来自寻址，不可能来自输入。
    let toks = tokenizer
        .encode(PROMPTS[0].repeat(2).as_bytes())
        .expect("中文提示词必须能编码");
    let seed = *toks.last().unwrap();
    let prompt = toks[..toks.len() - 1].to_vec();
    let prompts: Vec<Vec<u32>> = vec![prompt; WAVE_SLOTS];
    let feeds = vec![seed; WAVE_SLOTS];

    let mut state = match model.create_batch_state(WAVE_SLOTS) {
        Ok(s) => s,
        Err(e) => {
            eprintln!("SKIP: {WAVE_SLOTS} 槽批量状态建不出来（显存不足？），跳过: {e}");
            return;
        }
    };
    model.reset_state_of(&state).unwrap();
    model
        .forward_seq_batch_padded(&mut state, &prompts, PAD_TO)
        .unwrap();

    let sp = sampler(1, 0.0);
    let ticket = model
        .submit_sample_selfloop_batch(&mut state, &feeds, NTOK, &sp)
        .unwrap();
    let outs = model
        .collect_sample_selfloop_batch(ticket, WAVE_SLOTS)
        .unwrap();

    let bad: Vec<usize> = (0..WAVE_SLOTS).filter(|&s| outs[s] != outs[0]).collect();
    if !bad.is_empty() {
        // 把「从哪一槽开始分叉」打出来：≥64 说明是第二波，<64 说明与波无关（另一类 bug）。
        eprintln!(
            "分叉槽: {:?}（首波边界 64）槽0={:?} 分叉槽={:?}",
            bad, outs[0], outs[bad[0]]
        );
    }
    assert!(
        bad.is_empty(),
        "{WAVE_SLOTS} 槽同输入同贪心，必须逐位一致；实际 {}/{} 槽分叉，首个分叉槽 {}（波边界 64）",
        bad.len(),
        WAVE_SLOTS,
        bad[0],
    );
    eprintln!(
        "[WAVE] {WAVE_SLOTS} 槽 × {NTOK} 步贪心逐位一致（首 token {}）",
        outs[0][0]
    );
}

/// 宿主侧复刻 `sample.comp` §3–§6 的 top-p 口径（惩罚全关 ⇒ 只有 temperature + softmax +
/// 「按 p 降序累计到 top_p，末位即 `g_cutoff`」）：返回 `(候选集大小, cutoff 概率)`。
///
/// 用 f64 累积：GPU 侧是 f32 树形归约，边界可能差一个元素 —— 所以判据只取 cutoff 的
/// **量级**（见用例里的 `P_SLACK`），不取精确位次。
///
/// ★ 必须复刻 kernel 的趟数上限 `MAX_TOP_P_ITERS`（=512）：阈值很高（temp 抬高 + top_p 调大）
/// 时宿主侧会算出上千个候选，而 GPU 只消费前 512 个就停 ⇒ 真实 cutoff 是**第 512 名**的概率，
/// 比第 N 名高。不夹这一刀，宿主侧候选集是 GPU 的**超集**，判据白白放松。
fn top_p_nucleus(logit_row: &[f32], temperature: f32, top_p: f32) -> (usize, f64) {
    const MAX_TOP_P_ITERS: usize = 512;
    let inv_t = if temperature > 0.0 {
        1.0 / f64::from(temperature)
    } else {
        1.0
    };
    let m = logit_row
        .iter()
        .map(|&v| f64::from(v))
        .fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    let mut probs: Vec<f64> = logit_row
        .iter()
        .map(|&v| {
            let e = (f64::from(v) * inv_t - m).exp();
            sum += e;
            e
        })
        .collect();
    for p in &mut probs {
        *p /= sum;
    }
    let mut desc = probs;
    desc.sort_by(|a, b| b.total_cmp(a));
    let mut cum = 0.0;
    let mut size = 0usize;
    let mut cutoff = 0.0;
    for &p in &desc {
        cum += p;
        size += 1;
        cutoff = p;
        if cum >= f64::from(top_p) || size >= MAX_TOP_P_ITERS {
            break;
        }
    }
    // probs 已被 move 进 desc，这里返回它需要的信息只有 (size, cutoff)。
    (size, cutoff)
}

/// 10：**mask 路径（top-p 兜底循环）的候选集守门用例**。
///
/// 为什么必须单独有这一条：本文件其它采样判据全是贪心（`top_k=1`），贪心不进 top-p 循环 ⇒
/// mask 的读/置/清在现有 CI 里**一次都没被执行过**。位图化改动（`MASK_WORD_STRIDE`，
/// 2026-10-10）动的正是这个循环：行步长或位偏移写错的后果是「跨槽位污染 / 上一轮残留」，
/// 那会把真正的高概率候选标成已选，`g_cutoff` 掉到分布尾部，于是能采出概率极低（甚至接近 0）
/// 的 token —— 用户侧表现为偶发乱码，而吞吐基准完全看不出来。
///
/// 判据刻意**不依赖随机数**（同 seed 连跑两遍逐 token 都不可复现，见用例 4 的 ★ 注）：
/// 只要求「采到的 token 的概率 ≥ 宿主侧算出的 cutoff × `P_SLACK`」。合法的顶格候选之间
/// 差不到一个数量级，而泄漏进来的尾部候选比 cutoff 低**几个数量级**，所以这条既不会因
/// f32/f64 边界抖动而假失败，又能确实抓住位偏移类 bug。
///
/// 覆盖性自证：断言里同时要求**至少一半槽的候选集 ≥ 2**（否则循环一趟就退出，mask 读路径
/// 没被走到 ⇒ 本用例退化成空判），分布若哪天变得过分尖锐，这里会直接报出来而不是静默失去效力。
#[test]
fn top_p_sampled_tokens_stay_inside_the_nucleus() {
    const DRAWS: usize = 8;
    /// 判据容差：概率相对 cutoff 的下限。取 1e-2 是为了「绝不误报」优先——它仍然比
    /// 尾部泄漏（1e-5..1e-8 量级）高三个数量级。
    const P_SLACK: f64 = 1e-2;

    let _guard = gpu_lock();
    let Some((mut model, tokenizer)) = try_load() else {
        return;
    };
    let (batch, _) = prompts(&tokenizer);
    let pad_to = batch.iter().map(|p| p.len()).max().unwrap();
    let vocab = model.info().num_vocab;
    let mut state = model.create_batch_state(SLOTS).unwrap();

    // top_k=0 ⇒ 跳过 top-k 基数选择，直接进 top-p 兜底循环（mask 每步都被清/置/读）。
    //
    // ★ 口径必须**故意把分布摊平**（temp 1.25 + top_p 0.3），否则本用例是空判：
    // 2026-10-10 用默认 `sampler()`（temp 1.0 / top_p 0.2）实测四条槽的 top-1 概率分别是
    // 0.229 / 0.443 / 0.793 / 0.488 —— 全都在第一趟就累计过阈值 ⇒ 循环只跑一趟，
    // mask 只被清和被写、**从没被读**（下面的覆盖性断言正是为此而设，它当场把这条报了出来）。
    // 反过来也不能摊得太平：temp 2.0 / top_p 0.5 实测候选集 828–1412 个，cutoff 落到
    // 1e-4 量级，判据下限（cutoff×P_SLACK）就贴到分布尾部了。1.25/0.3 是「候选集 ≥ 2
    // 且 cutoff 仍在 1e-2 量级」的窗口，覆盖性与区分度同时拿到。
    let sp = SamplerParams {
        temperature: 1.25,
        top_k: 0,
        top_p: 0.3,
        seed: 514,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        ..Default::default()
    };
    assert!(
        (sp.repetition_penalty - 1.0).abs() < f32::EPSILON
            && sp.frequency_penalty == 0.0
            && sp.presence_penalty == 0.0,
        "本用例的宿主侧复刻不含惩罚项，sampler 口径若加惩罚必须同步"
    );

    // —— 参考分布：从零态 prefill 后跑第一步前向，取 logits ——
    model.reset_state_of(&state).unwrap();
    let feed = model
        .forward_seq_batch_padded(&mut state, &batch, pad_to)
        .unwrap();
    model.reset_state_of(&state).unwrap();
    let feed2 = model
        .forward_seq_batch_padded(&mut state, &batch, pad_to)
        .unwrap();
    assert_eq!(
        feed, feed2,
        "prefill 必须可复现，否则 logits 与抽样不同分布"
    );
    let tokens_in: Vec<Vec<u32>> = feed.iter().map(|&t| vec![t]).collect();
    let logits = model
        .batch_step_logits(&mut state, &tokens_in)
        .expect("batch_step_logits 必须可用");

    let mut cutoffs = Vec::with_capacity(SLOTS);
    let mut sizes = Vec::with_capacity(SLOTS);
    let mut argmaxes = Vec::with_capacity(SLOTS);
    for s in 0..SLOTS {
        let row = &logits[s * vocab..(s + 1) * vocab];
        let (size, cutoff) = top_p_nucleus(row, sp.temperature, sp.top_p);
        let am = row
            .iter()
            .enumerate()
            .fold((0usize, f32::NEG_INFINITY), |(bi, bv), (i, &v)| {
                if v > bv { (i, v) } else { (bi, bv) }
            })
            .0 as u32;
        eprintln!("[NUCLEUS] slot{s} 候选集={size} cutoff={cutoff:.3e} argmax={am}");
        sizes.push(size);
        cutoffs.push(cutoff);
        argmaxes.push(am);
    }
    assert!(
        sizes.iter().filter(|&&n| n >= 2).count() * 2 >= SLOTS,
        "top-p 循环全程只走一趟（候选集全为 1）⇒ mask 读路径没被覆盖，本用例已失效；\
         调高 temperature 或调低 top_p 再造出多候选分布。实测 sizes={sizes:?}"
    );

    // —— 多 seed 独立抽样：每次从零态重跑到同一步，保证与 logits 参考同分布 ——
    let hist = model.alloc_u32(SLOTS * 8).unwrap();
    model.write_u32(hist, &[0u32; SLOTS * 8]).unwrap();
    let hist_len = vec![0u32; SLOTS];
    // 宿主侧按 token id 查概率只需要参考行；排序已丢掉位次，这里重算单点概率。
    let prob_of = |s: usize, tok: u32| -> f64 {
        let row = &logits[s * vocab..(s + 1) * vocab];
        let inv_t = if sp.temperature > 0.0 {
            1.0 / f64::from(sp.temperature)
        } else {
            1.0
        };
        let m = row
            .iter()
            .map(|&v| f64::from(v))
            .fold(f64::NEG_INFINITY, f64::max);
        let mut sum = 0.0;
        let ps: Vec<f64> = row
            .iter()
            .map(|&v| {
                let e = (f64::from(v) * inv_t - m).exp();
                sum += e;
                e
            })
            .collect();
        ps[tok as usize] / sum
    };

    let mut drawn = 0usize;
    // mask **失效**类故障的反向判据：见函数末尾的断言说明。
    let mut off_greedy = 0usize;
    let mut greedy_eligible = 0usize;
    for r in 0..DRAWS {
        model.reset_state_of(&state).unwrap();
        let seeds_r = model
            .forward_seq_batch_padded(&mut state, &batch, pad_to)
            .unwrap();
        assert_eq!(seeds_r, feed, "第 {r} 次抽样的起点状态必须与参考一致");
        let sps: Vec<SamplerParams> = (0..SLOTS)
            .map(|s| SamplerParams {
                seed: 7919u32.wrapping_mul(r as u32 + 1).wrapping_add(s as u32),
                ..sp
            })
            .collect();
        let got = model
            .batch_step_sample(&mut state, &tokens_in, hist, 8, &hist_len, &sps)
            .unwrap();
        for (s, &t) in got.iter().enumerate() {
            let p = prob_of(s, t);
            let floor = cutoffs[s] * P_SLACK;
            assert!(
                p >= floor,
                "[NUCLEUS] slot{s} 第{r}抽采出候选集外的 token {t}：p={p:.3e} < cutoff×{P_SLACK}={floor:.3e}（cutoff={:.3e} 候选集={}）\
                 ⇒ mask 位偏移/行步长/清除范围有错（跨槽污染或上一轮残留）",
                cutoffs[s],
                sizes[s]
            );
            if sizes[s] >= 2 {
                greedy_eligible += 1;
                if t != argmaxes[s] {
                    off_greedy += 1;
                }
            }
            drawn += 1;
        }
    }
    model.release_tensor(hist);
    // 反向判据（与上面的「不许掉出候选集」成对，缺一不可）：
    // mask 若**整体失效**（比如行步长算错到读不到任何置位、或置位写到读不到的位置），
    // 循环就每趟都重新挑中同一个 argmax，`g_cum` 反复累加同一个 p，最终 `g_cutoff = p_argmax`
    // ⇒ 截断后只剩 argmax ⇒ 采样**退化成贪心**。这种故障不会掉出宿主侧候选集（argmax 当然在
    // 集合内），所以只靠上一条判据是抓不住的 —— 2026-10-10 注入 `mask_wstride /= 2` 实测正是
    // 如此：上一条判据照样通过。这里要求「候选集 ≥2 的槽里至少出现过一次非 argmax」。
    // 概率上极宽松：slot0 候选集 11、argmax 占比 ~1/11..1/3 ⇒ 32 次全中的概率 < 1e-6。
    assert!(
        greedy_eligible > 0 && off_greedy > 0,
        "[NUCLEUS] {greedy_eligible} 次候选集≥2 的抽样**全部**等于 argmax ⇒ top-p 循环没真的消费掉\
         已选候选（mask 失效 / 置位写到读不到的位置），采样已退化成贪心。sizes={sizes:?}"
    );
    eprintln!(
        "[NUCLEUS] {drawn} 次 top-p 抽样全部落在候选集内（top_p={} temp={}）",
        sp.top_p, sp.temperature
    );
}
