//! RWKV-7 GPU 推理模块
//!
//! 使用 Vulkan runtime 在 GPU 上执行前向推理，数值对齐 CPU model.rs 实现。
//! 所有线性权重保持 PyTorch [out, in] 行主序布局（直接对齐 gemv 着色器期望）；
//! 低秩权重在加载时从 safetensors 的 [in, out] 转置为 [out, in]。
//!
//! 注: GpuModel 在下一步验证流程中接入 main.rs，当前模块允许 dead_code 警告。

#![allow(dead_code)]

use half::f16;
use safetensors::tensor::TensorView;

use crate::backend::{ComputeBackend, Int8Handle, TensorDtype, TensorId};
use crate::runtime::R;

/// LayerNorm eps（与 CPU model.rs 一致）
pub const LN_EPS: f32 = 1.0e-5;
/// L2 范数防除零（与 fuse_ka_dplr 的 1e-12 一致）
pub const EPS_L2: f32 = 1.0e-12;
/// GroupNorm eps（注意：当前 norm 着色器固定使用 1.0e-5，存在轻微数值差异）
pub const GN_EPS: f32 = 64.0e-5;

/// FFN hidden 维度、低秩中间维度随模型变化，由 safetensors 形状推导，不在此硬编码
///
/// 模型超参数
#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layer: usize,
    pub n_embd: usize, // C
    pub vocab: usize,
    pub head_size: usize, // N = 64
    pub n_head: usize,    // H = C / N
    // 低秩中间维度（从 safetensors 形状推导）
    pub ffn_hidden: usize,
    pub w_mid: usize,
    pub a_mid: usize,
    pub v_mid: usize,
    pub g_mid: usize,
    // 低秩中间维度补齐到 TILE_N=64 的倍数（tensor-core GEMM 的 N/K 维度要求）
    pub w_mid_pad: usize,
    pub a_mid_pad: usize,
    pub v_mid_pad: usize,
    pub g_mid_pad: usize,
}

impl ModelConfig {
    /// 派生 web-rwkv 风格的公开模型元信息（供服务端读取/展示）。
    pub fn info(&self) -> ModelInfo {
        ModelInfo {
            version: ModelVersion::V7,
            num_layer: self.n_layer,
            num_emb: self.n_embd,
            // 对齐 web-rwkv：num_hidden 用于 buffer 计算（num_emb × num_hidden = 每层 att 线性投影权重大小），RWKV-7 为 C×C
            num_hidden: self.n_embd,
            num_vocab: self.vocab,
            num_head: self.n_head,
            head_size: self.head_size,
            ffn_hidden: self.ffn_hidden,
            w_mid: self.w_mid,
            a_mid: self.a_mid,
            v_mid: self.v_mid,
            g_mid: self.g_mid,
        }
    }
}

/// 模型架构版本（对标 `web-rwkv::ModelVersion`）。rwkv-rsv 专用于 RWKV-7。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVersion {
    V7,
}

/// web-rwkv 风格的公开模型元信息（对标 `web-rwkv::ModelInfo`）。
/// 服务端（ai00-server）据此获取模型规模、初始化 State、展示模型信息，无需访问内部 `ModelConfig`。
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ModelInfo {
    /// 模型架构版本
    pub version: ModelVersion,
    /// 层数
    pub num_layer: usize,
    /// 嵌入维度 C
    pub num_emb: usize,
    /// 隐藏维度（=num_emb，用于 buffer 尺寸计算）
    pub num_hidden: usize,
    /// vocab 大小
    pub num_vocab: usize,
    /// 头数
    pub num_head: usize,
    /// 每头维度 N
    pub head_size: usize,
    /// FFN hidden 维度
    pub ffn_hidden: usize,
    /// 低秩中间维度（w/a/v/g）
    pub w_mid: usize,
    pub a_mid: usize,
    pub v_mid: usize,
    pub g_mid: usize,
}

/// 分块 prefill 的**子批宽度**（`PREFILL_SLOTS`，默认 8；0 = 关闭分块、沿用旧行为）。
///
/// prefill 工作区 `SeqBuffers` 大小 ∝ `子批宽度 × T_pad`（~45 个缓冲、多数 fp32），
/// 而它**在解码期完全不用却常驻**。把宽度与解码并发解耦后，B=256 也能在 22GB 卡上跑通；
/// 代价是 prefill 墙钟时间按 `ceil(B/子批宽度)` 叠加（解码吞吐不受影响）。
fn prefill_chunk_slots() -> usize {
    static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
    *N.get_or_init(|| {
        std::env::var("PREFILL_SLOTS")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(8)
    })
}

/// RWKV-7 GPU 模型：权重上传一次，forward 时复用工作缓冲与状态
pub struct GpuModel {
    backend: Box<dyn ComputeBackend>,
    config: ModelConfig,
    // 预计算的 GPU 权重
    emb_ln: TensorId, // [vocab, C] fp16 — 预计算 ln0(embed) 后的 embedding
    emb_ln_cpu: std::sync::Arc<Vec<f32>>, // [vocab, C] CPU 缓存（f16 舍入值，与 GPU 表逐位一致），避免每次 forward_seq 下载；Arc 供权重共享实例零复制
    ln_out_w: TensorId,
    ln_out_b: TensorId,
    head_w16: Option<TensorId>, // [vocab, C] fp16 变体（head 未 int8 量化时用）
    head_a8: Option<Int8Handle>, // [vocab, C] int8 量化 head（模型含 head.weight.int8_* 时启用，省 4× 带宽）
    layers: Vec<GpuLayer>,
    // 工作缓冲区（forward 时复用，避免每 token 重新分配）
    bufs: WorkBuffers,
    // batch 并发工作缓冲区（[batch, ...] 布局；B 变化时重建。None = 未用 batch 路径）
    batch_bufs: Option<(usize, WorkBuffers)>,
    // batch prefill 的 lens 缓冲（[batch] u32；batch 变化时重建）
    batch_lens: Option<(usize, TensorId)>,
    // 序列并行工作缓冲区（forward_seq 时按需创建/复用）
    seq_bufs: Option<SeqBuffers>,
    // 批量单步（batch_step_sample）专用序列缓冲：与 seq_bufs 隔离，避免与单序列
    // prefill / 分类路径的 T 不同反复 clear_cache + 重建。
    batch_seq_bufs: Option<SeqBuffers>,
    // 分块 prefill 用的小状态（宽度 = `prefill_chunk_slots()`）。
    // 动机：解码并发可以到 256，但 prefill 工作区不该跟着涨（见 forward_seq_batch_padded）。
    prefill_state: Option<(usize, State)>,
    // 批量单步的采样临时缓冲（vocab×batch 级大缓冲，按 batch 缓存复用）。
    batch_step_tmp: Option<(usize, BatchStepTmp)>,
    /// self-loop 常驻缓冲（按 (kind,batch,n) 缓存）。常驻原因见 [`SelfloopBufs`]。
    selfloop_bufs: Vec<SelfloopBufs>,
    // 内部推理状态（web-rwkv 风格 `State`）。
    // 拆分为外部传入式 `forward_with_state`/`forward_seq_with_state`（主 API），
    // 旧 `forward`/`forward_seq`/`forward_argmax` 等便捷封装复用此内部态。
    state: Option<State>,
    // 标量缓冲区: -1/sqrt(e)，用于 w = exp(w_sig * scale)
    scale_w: TensorId,
}

/// 单层权重（全部已上传到 GPU）
#[derive(Clone)]
pub struct GpuLayer {
    ln1_w: TensorId,
    ln1_b: TensorId,
    ln2_w: TensorId,
    ln2_b: TensorId,
    ln_x_w: TensorId,
    ln_x_b: TensorId,
    // token shift 系数 [C]
    x_r: TensorId,
    x_w: TensorId,
    x_k: TensorId,
    x_v: TensorId,
    x_a: TensorId,
    x_g: TensorId,
    // att biases [C]
    w0: TensorId,
    a0: TensorId,
    v0: TensorId,
    // 低秩权重（单 token 低秩链用 fp32；prefill 用 fp16，见 *_16）
    w1: TensorId,
    w2: TensorId,
    a1: TensorId,
    a2: TensorId,
    v1: TensorId,
    v2: TensorId,
    g1: TensorId,
    g2: TensorId,
    // element-wise 参数
    r_k: TensorId,     // [H, N]
    k_k: TensorId,     // [C]
    k_a: TensorId,     // [C]
    ffn_x_k: TensorId, // [C]
    /// 诊断用 fp32 输出权重（仅 GEMM_DIAG 参考路径使用；其余线性权重已删 fp32 副本省显存）。
    /// 仅在设置 GEMM_DIAG 时创建，正式推理不持有，省 ~26MB/层。
    output_w: Option<TensorId>, // [C, C]
    // fp16 线性权重（prefill tensor-core GEMM 用，fp32io16 模式）——
    // 不常驻：prefill（forward_seq）前由 host 反量化临时创建，decode 前释放归还显存。
    receptance_w16: Option<TensorId>, // [C, C]
    key_w16: Option<TensorId>,
    value_w16: Option<TensorId>,
    output_w16: Option<TensorId>,
    ffn_key_w16: Option<TensorId>,   // [ffn_hidden, C]
    ffn_value_w16: Option<TensorId>, // [C, ffn_hidden]（稠密，稀疏内核回退用）
    // fp16 ffn_value 的平铺布局 [fh, C]（CudaBackend 稀疏 FFN 用；Vulkan 回退稠密）。
    // `FFN_VALUE_GEMM`（默认开）时**不建**（改由 ffn_value_dense16 承担），省 1.68GB。
    ffn_value_tiled: Option<TensorId>,
    /// ffn_value 的稠密 fp16 `[C, fh]`（Phase 2-3 稠密张量核 GEMM 用）。
    /// 与 `ffn_value_tiled` **互斥**（同一份数据、同一体积 52.4MB/层），故显存净变化为 0。
    ffn_value_dense16: Option<TensorId>,
    // int8 量化权重（decode 单 token GEMV 用；None 表示该矩阵未量化，走 fp16 路径）
    ffn_key_a8: Option<Int8Handle>,    // [ffn_hidden, C]
    ffn_value_a8: Option<Int8Handle>,  // [C, ffn_hidden]
    att_output_a8: Option<Int8Handle>, // [C, C]
    receptance_a8: Option<Int8Handle>, // [C, C]
    key_a8: Option<Int8Handle>,        // [C, C]
    value_a8: Option<Int8Handle>,      // [C, C]
    // fp16 低秩权重（tensor-core GEMM 用，已补齐到 [mid_pad, C] / [C, mid_pad]）—— 仅保留 fp16
    w1_16: TensorId, // [wm_pad, C]
    w2_16: TensorId, // [C, wm_pad]
    a1_16: TensorId, // [am_pad, C]
    a2_16: TensorId, // [C, am_pad]
    v1_16: TensorId, // [vm_pad, C]
    v2_16: TensorId, // [C, vm_pad]
    g1_16: TensorId, // [gm_pad, C]
    g2_16: TensorId, // [C, gm_pad]
}

/// 每层 RNN 状态（GPU 端）
pub struct GpuState {
    tmix_x: TensorId,   // [C] token shift
    tmix_rnn: TensorId, // [H, N, N] DPLR state
    cmix_x: TensorId,   // [C] token shift
}

/// WKV 状态（`tmix_rnn`）的**存储** dtype：`WKV_STATE_F16=0` 时退回 fp32，否则 fp16。
///
/// ★ **默认开**（Phase 2-2，2026-09-22 翻默认）。依据 = **与信天翁同精度档**：
/// `faster3a_2607/rwkv7_fast_v3a.py:2404` 的 `WKV_MODE="fp16"`（上游**默认值**）下
/// 状态就是 `torch.float16`；且 `cuda/rwkv7_wkv_fp16_v2.cu` 连状态**运算**都是 fp16
/// （`__hfma2` / `__hmul2` / `half2 state[]`）。本实现是**fp16 存 + fp32 算**
/// ⇒ 比上游默认档**更准**。我们测的 2874.3 基线反而是上游更准的 `fp32io16` 档
/// （fp16 档用 `cp.async`，sm_75 编不出来）。
///
/// ⚠️ 精度实测（`TOPK=1` 贪心逐 token 对拍 fp32 状态）：B=64 × 512 步 = 23.26% 分叉、
/// B=256 × 64 步 = 17.69%；**B=8 × 512 步 = 0% 是假阳性**（batch<16 走 SIMT 路径，
/// 输入值不同恰好不翻 argmax）——**长程数值门禁必须在生产批量档（≥64）上做**。
///
/// ⚠️ 三处状态内核（`fuse_ka_dplr_norm` / `dplr_seq_batch` / `dplr_seq`）由
/// host 侧 `any_ptr` + `dplr_variant` 自动跟随本 dtype，无需手工同步。
pub(crate) fn wkv_state_dtype() -> TensorDtype {
    if std::env::var("WKV_STATE_F16").is_ok_and(|v| v == "0") {
        TensorDtype::F32
    } else {
        TensorDtype::F16
    }
}

/// 在途 self-loop 句柄：submit 后全部轮次已在 GPU 流上排队（零 host 同步），
/// collect 时一次同步取回生成序列。
/// 单线程使用约束：同一后端至多一个在途 ticket（pinned 行号复用前提）。
pub struct SelfloopTicket {
    /// 生成序列缓冲。**常驻**：由 [`SelfloopBufs`] 形状缓存持有，collect 不释放
    /// （graph 内已烘焙其指针，释放会导致重放野指针）。
    token_seq: TensorId,
    n: usize,
}

/// 批量单步采样（`batch_step_sample`）的采样临时缓冲：vocab 级大缓冲
///（temp/mask 各 [batch, vocab] f32 + counter [batch, vocab] u32 直方图），
/// 按 batch 缓存复用——逐 token 主机介入路径每步都要用，不能每步重建。
#[derive(Clone)]
struct BatchStepTmp {
    temp: TensorId,    // [batch, vocab]
    mask: TensorId,    // [batch, vocab]
    counter: TensorId, // [batch, vocab] u32
    sampler: TensorId, // [batch, 10]
}

/// self-loop 的形状 key：`(kind, batch, n)` 折叠为 u64。
/// kind：0 = 单流（`selfloop_step`），1 = 批量（`sample_selfloop_step_batch`）。
/// 两者 kernel 序列不同，即使 batch 同为 1 也必须分开缓存。
fn selfloop_shape_key(kind: u64, batch: usize, n: usize) -> u64 {
    (kind << 62) | ((batch as u64) << 32) | (n as u64)
}

/// self-loop 常驻缓冲的形状数上限。超出时整体重置（释放缓冲并作废所有已捕获图）——
/// 每个形状约 `6 × batch × vocab × 4` 字节，不设限会被 max_tokens 余数组合撑爆显存。
const MAX_SELFLOOP_SHAPES: usize = 8;

/// self-loop 的常驻工作缓冲，按形状缓存、**跨 submit 复用**。
///
/// 为什么必须常驻：这些缓冲的设备指针被烘焙进已捕获的 CUDA graph。原先每次
/// submit 都 `create_tensor` 并在 collect 里释放，导致指针每次都变 → graph 每段
/// 都要重新 capture + instantiate + destroy；实测这种高频 churn 会在
/// `cuGraphInstantiate` 驱动代码内部触发 0xC0000005（nvcuda64.dll+0x297ca0）。
/// 常驻后每个形状只需捕获一次，此后永久重放。
/// ★ 2026-09-22：那个 `0xC0000005` 的**根因已定位** —— `cuGraphInstantiate` 的符号
/// 解析取到了 v1（五参数）而按三参数调用（见 `backend_cuda.rs` 的 `cu_graph_instantiate`）。
/// 常驻仍是好设计（省去重复实例化），但"churn 才崩"只是概率性 ABI 踩空的表象。
#[derive(Clone)]
struct SelfloopBufs {
    kind: u64,
    batch: usize,
    n: usize,
    temp: TensorId,      // [batch, vocab]
    mask: TensorId,      // [batch, vocab]
    counter: TensorId,   // [batch, vocab] u32
    sampler: TensorId,   // [batch, 10]
    token_seq: TensorId, // [batch, n]
    seq_cnt: TensorId,   // [batch]
}

impl SelfloopBufs {
    fn key(&self) -> u64 {
        selfloop_shape_key(self.kind, self.batch, self.n)
    }
    fn all(&self) -> [TensorId; 6] {
        [
            self.temp,
            self.mask,
            self.counter,
            self.sampler,
            self.token_seq,
            self.seq_cnt,
        ]
    }
}

/// 工作缓冲区：forward 期间复用，避免反复创建 GpuTensor
///（TensorId 全 Copy，Clone 零开销——batch 路径取副本避开 &self 与 &mut self 借用冲突）
#[derive(Clone)]
pub struct WorkBuffers {
    // [C] 大小
    x: TensorId,
    ln1: TensorId,
    xr: TensorId,
    xw: TensorId,
    xk: TensorId,
    xv: TensorId,
    xa: TensorId,
    xg: TensorId,
    prev_x: TensorId,
    r: TensorId,
    k: TensorId,
    v: TensorId,
    v_full: TensorId,
    gate: TensorId,
    w_full: TensorId,
    w_sig: TensorId,
    w: TensorId,
    a_full: TensorId,
    a: TensorId,
    kk_l2: TensorId,
    k_mod: TensorId,
    b_vec: TensorId,
    y: TensorId,
    y_norm: TensorId,
    g: TensorId,
    y_g: TensorId,
    y_out: TensorId,
    ln2: TensorId,
    prev_c: TensorId,
    xb: TensorId,
    v2: TensorId,
    x_norm: TensorId,
    tmp_c: TensorId, // 临时缓冲，用于 in-place 操作中转
    // 其他大小
    v_mid: TensorId,         // [V_MID]
    w_mid: TensorId,         // [W_MID]
    a_mid: TensorId,         // [A_MID]
    g_mid: TensorId,         // [G_MID]
    r2: TensorId,            // [FFN_HIDDEN]
    logits: TensorId,        // [vocab]
    token_argmax: TensorId,  // [1] GPU argmax 采样的 token 索引（字节存 uint）
    current_token: TensorId, // [1] 当前待 gather 的 token 索引（f32 位模式存 uint，供 gather_row 读取）
}

/// 序列并行工作缓冲区（forward_seq 用）：所有激活均为 [T, C]（token 主序）
pub struct SeqBuffers {
    /// 序列长度 T（缓冲大小固定，T 变化时重建）
    t: usize,
    /// 补齐到 TILE_M=256 倍数的 token 数（GEMM 输出缓冲大小）
    m_pad: usize,
    // [T, C] 大小
    x: TensorId,
    ln1: TensorId,
    xr: TensorId,
    xw: TensorId,
    xk: TensorId,
    xv: TensorId,
    xa: TensorId,
    xg: TensorId,
    // [M_PAD, C] 大小（tensor-core GEMM 输出）
    r: TensorId,
    k: TensorId,
    v: TensorId,
    v_first: TensorId, // [M_PAD, C] 每 token 来自 layer 0 的 v（copy_device 需与 v 同尺寸）
    v_full: TensorId,  // [M_PAD, C]（tensor-core GEMM 输出）
    gate: TensorId,    // [T, C]
    w_full: TensorId,  // [M_PAD, C]（tensor-core GEMM 输出）
    w_sig: TensorId,   // [T, C]
    w: TensorId,       // [T, C]
    a_full: TensorId,  // [M_PAD, C]（tensor-core GEMM 输出）
    a: TensorId,       // [T, C]
    kk_l2: TensorId,
    k_mod: TensorId,
    b_vec: TensorId,
    y: TensorId,
    y_norm: TensorId,
    g: TensorId, // [M_PAD, C]（tensor-core GEMM 输出）
    y_g: TensorId,
    y_out: TensorId,        // [M_PAD, C]
    diag_out_ref: TensorId, // [M_PAD, C] gemv 参考输出（仅诊断用）
    ln2: TensorId,
    xb: TensorId,
    v2: TensorId, // [M_PAD, C]
    x_norm: TensorId,
    tmp_c: TensorId,
    // 低秩中间缓冲 [M_PAD, mid_pad]（tensor-core GEMM 输出，mid_pad 补齐到 64）
    v_mid: TensorId,
    w_mid: TensorId,
    a_mid: TensorId,
    g_mid: TensorId,
    r2: TensorId,      // [M_PAD, fh]
    head_in: TensorId, // [C] 最后 token 的 x_norm 行（head 只算末 token）
    logits: TensorId,  // [vocab] 末 token logits
    // fp16 激活（tensor-core GEMM 输入）
    xr16: TensorId, // [M_PAD, C]
    xk16: TensorId,
    xv16: TensorId,
    xw16: TensorId, // 低秩 w/a/g 第一级投影输入
    xa16: TensorId,
    xg16: TensorId,
    y_g16: TensorId,
    xb16: TensorId,
    r2_16: TensorId, // [M_PAD, fh]
    // fp16 低秩中间缓冲（第二级投影的 GEMM 输入）
    v_mid16: TensorId, // [M_PAD, vm_pad]
    w_mid16: TensorId, // [M_PAD, wm_pad]
    a_mid16: TensorId, // [M_PAD, am_pad]
    g_mid16: TensorId, // [M_PAD, gm_pad]
    /// int8→fp16 反量化共享 scratch（方案A prefill）：大小取 6 个 int8 矩阵最大值
    /// [ffn_hidden, C] = 26.2M 元素（52.4MB）。每矩阵 GEMM 前由 dequant 全量覆写，
    /// 顺序复用（barrier 由 record_kernel 读写序保证），替代旧逐层 3.4GB fp16 副本。
    w_scratch: TensorId,
}

impl SeqBuffers {
    /// 释放全部工作缓冲的设备内存（T 变化重建前调用，防泄漏：
    /// 旧缓冲的 TensorId 被结构体 Drop 后，后端注册表仍持有设备内存）。
    fn free(&mut self, backend: &mut dyn ComputeBackend) {
        let tensors = [
            self.x,
            self.ln1,
            self.xr,
            self.xw,
            self.xk,
            self.xv,
            self.xa,
            self.xg,
            self.r,
            self.k,
            self.v,
            self.v_first,
            self.v_full,
            self.gate,
            self.w_full,
            self.w_sig,
            self.w,
            self.a_full,
            self.a,
            self.kk_l2,
            self.k_mod,
            self.b_vec,
            self.y,
            self.y_norm,
            self.g,
            self.y_g,
            self.y_out,
            self.diag_out_ref,
            self.ln2,
            self.xb,
            self.v2,
            self.x_norm,
            self.tmp_c,
            self.v_mid,
            self.w_mid,
            self.a_mid,
            self.g_mid,
            self.r2,
            self.head_in,
            self.logits,
            self.xr16,
            self.xk16,
            self.xv16,
            self.xw16,
            self.xa16,
            self.xg16,
            self.y_g16,
            self.xb16,
            self.r2_16,
            self.v_mid16,
            self.w_mid16,
            self.a_mid16,
            self.g_mid16,
            self.w_scratch,
        ];
        for t in tensors {
            backend.free_tensor(t);
        }
    }
}

impl GpuModel {
    /// 从 safetensors 文件加载模型并上传到 GPU
    pub fn from_safetensors(mut backend: Box<dyn ComputeBackend>, path: &str) -> R<Self> {
        let file = std::fs::File::open(path)?;
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        let st = safetensors::SafeTensors::deserialize(&mmap)?;

        let n_layer = max_layer(&st) + 1;
        let emb = st.tensor("emb.weight")?;
        let (vocab, n_embd) = (emb.shape()[0], emb.shape()[1]);
        // head_size / n_head 从 r_k 形状 [H, N] 推导（跨模型通用）
        let rk = st.tensor("blocks.0.att.r_k")?;
        let n_head = rk.shape()[0];
        let head_size = rk.shape()[1];

        // 低秩中间维度从各 low-rank 权重形状推导
        let mid_of = |t: &TensorView| -> usize {
            let s = t.shape();
            if s[0] == n_embd { s[1] } else { s[0] }
        };
        let w_mid = mid_of(&st.tensor("blocks.0.att.w1")?);
        let a_mid = mid_of(&st.tensor("blocks.0.att.a1")?);
        let v_mid = mid_of(&st.tensor("blocks.0.att.v1")?);
        let g_mid = mid_of(&st.tensor("blocks.0.att.g1")?);
        // 原 fp16 键可能被 int8 量化键替换，此时从量化张量推导 M（int8_idx [M,K]）
        let ffn_hidden = match st.tensor("blocks.0.ffn.key.weight") {
            Ok(t) => t.shape()[0],
            Err(_) => st.tensor("blocks.0.ffn.key.weight.int8_idx")?.shape()[0],
        };

        let config = ModelConfig {
            n_layer,
            n_embd,
            vocab,
            head_size,
            n_head,
            ffn_hidden,
            w_mid,
            a_mid,
            v_mid,
            g_mid,
            w_mid_pad: round_up(w_mid, 64),
            a_mid_pad: round_up(a_mid, 64),
            v_mid_pad: round_up(v_mid, 64),
            g_mid_pad: round_up(g_mid, 64),
        };
        log::info!(
            "gpu model: n_layer={n_layer} n_embd={n_embd} vocab={vocab} n_head={n_head} head_size={head_size} \
             ffn_hidden={ffn_hidden} w_mid={w_mid} a_mid={a_mid} v_mid={v_mid} g_mid={g_mid}"
        );

        // 批量上传模式：环形 pinned 暂存 + 按槽事件同步（消除逐 tensor 全流同步）
        backend.upload_bulk_begin();

        // 加载 ln0 并在 CPU 端预计算 emb_ln = ln0(embed)
        // 键名: blocks.0.ln0.weight
        let emb_f32 = tensor_to_f32(&emb);
        let ln0_w = tensor_to_f32(&st.tensor("blocks.0.ln0.weight")?);
        let ln0_b = tensor_to_f32(&st.tensor("blocks.0.ln0.bias")?);
        let emb_ln = layer_norm_rows(&emb_f32, &ln0_w, &ln0_b, n_embd, vocab, LN_EPS);
        // GPU 表存 fp16（省 335MB 显存）；CPU 缓存用同一 f16 舍入值回读 f32，
        // 保证 seq（CPU 上传）与 tok（GPU gather）两条路径输入逐位一致
        let emb_ln_t = backend.create_tensor(vocab * n_embd, TensorDtype::F16)?;
        backend.upload(emb_ln_t, &emb_ln)?;
        let emb_ln: Vec<f32> = emb_ln.iter().map(|&v| f16::from_f32(v).to_f32()).collect();

        // ln_out
        let ln_out_w = tensor_to_f32(&st.tensor("ln_out.weight")?);
        let ln_out_b = tensor_to_f32(&st.tensor("ln_out.bias")?);
        let ln_out_w_t = backend.create_tensor(n_embd, TensorDtype::F32)?;
        backend.upload(ln_out_w_t, &ln_out_w)?;
        let ln_out_b_t = backend.create_tensor(n_embd, TensorDtype::F32)?;
        backend.upload(ln_out_b_t, &ln_out_b)?;

        // head：仅当模型自带 int8 量化（全 int8 模型的一部分）时用 int8（省 4× 带宽，
        // head 是 decode 单 kernel 读取量最大者）；否则保留 fp16 变体。
        // 2026-08-09 复测结论：fp16 模型单独量化 head 端到端收益 ≈1%（淹没于热噪声），
        // 已删除加载时合成 int8 head 路径（HEAD_QUANT），head 量化只随全 int8 模型使用。
        let head_a8 = if st.tensor("head.weight.int8_idx").is_ok() {
            let idx_t = st.tensor("head.weight.int8_idx")?; // U8 [vocab, C]
            let sz_t = st.tensor("head.weight.int8_sz")?; // U32 [vocab, C/128]
            let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
            let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
            assert_eq!(idx_u32.len(), vocab * n_embd / 4, "head.int8_idx 形状不符");
            assert_eq!(sz_u32.len(), vocab * n_embd / 128, "head.int8_sz 形状不符");
            let idx_gpu = backend.create_tensor(vocab * n_embd / 4, TensorDtype::U32)?;
            backend.upload_u32(idx_gpu, idx_u32)?;
            let sz_gpu = backend.create_tensor(vocab * n_embd / 128, TensorDtype::U32)?;
            backend.upload_u32(sz_gpu, sz_u32)?;
            log::info!("head 使用 int8 量化权重（{vocab}x{n_embd}）");
            Some(Int8Handle {
                idx: idx_gpu,
                sz: sz_gpu,
                m: vocab,
                k: n_embd,
            })
        } else {
            None
        };
        let head_w16_t = if head_a8.is_none() {
            let head_w = tensor_to_f32(&st.tensor("head.weight")?);
            let t = backend.create_tensor(vocab * n_embd, TensorDtype::F16)?;
            backend.upload(t, &head_w)?;
            Some(t)
        } else {
            None
        };

        // 加载每一层
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(GpuLayer::load(backend.as_mut(), &st, i, n_embd, &config)?);
        }

        // 工作缓冲区
        let bufs = WorkBuffers::new(backend.as_mut(), n_embd, vocab, &config, 1)?;

        // 内部推理状态（可序列化 `State`：每层 RNN 状态 + v_first）
        let state = State::new(backend.as_mut(), n_embd, n_head, head_size, n_layer)?;

        // scale_w: -1/sqrt(e)
        let scale_w = backend.create_tensor(1, TensorDtype::F32)?;
        backend.upload(scale_w, &[-1.0f32 / std::f32::consts::E.sqrt()])?;

        // 权重上传完成：释放模型级权重的 host（系统内存）缓冲，仅保留 device 拷贝
        backend.drop_host(emb_ln_t);
        backend.drop_host(ln_out_w_t);
        backend.drop_host(ln_out_b_t);
        if let Some(t) = head_w16_t {
            backend.drop_host(t);
        }
        backend.drop_host(scale_w);
        backend.upload_bulk_end();

        Ok(Self {
            backend,
            config,
            emb_ln: emb_ln_t,
            emb_ln_cpu: std::sync::Arc::new(emb_ln),
            ln_out_w: ln_out_w_t,
            ln_out_b: ln_out_b_t,
            head_w16: head_w16_t,
            head_a8,
            layers,
            bufs,
            batch_bufs: None,
            batch_lens: None,
            seq_bufs: None,
            batch_seq_bufs: None,
            prefill_state: None,
            batch_step_tmp: None,
            selfloop_bufs: Vec::new(),
            state: Some(state),
            scale_w,
        })
    }

    /// 重置 RNN 状态为零（回到首次推理前的初始状态）
    pub fn reset_state(&mut self) -> R<()> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;
        self.state
            .as_ref()
            .unwrap()
            .reset(&*self.backend, c, h, n)?;
        Ok(())
    }

    /// 清空累计的 per-kernel profiling 时间（仅诊断）。
    pub fn clear_kernel_prof(&mut self) {
        self.backend.clear_kernel_prof();
    }

    /// 打印累计的 per-kernel profiling 时间（仅诊断）。
    pub fn dump_kernel_prof(&mut self) {
        self.backend.dump_kernel_prof();
    }

    /// 重置一个**外部** `State`（例如 `Bundle` 持有的会话状态）为零。
    /// 与 `reset_state`（重置模型内部 `Option<State>`）不同，此方法作用于调用方传入的状态，
    /// 供 `Bundle::reset` 使用——否则 reset 会重置到模型内部态而遗漏会话态，导致跨次状态残留。
    pub fn reset_state_of(&self, state: &State) -> R<()> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;
        state.reset(&*self.backend, c, h, n)?;
        Ok(())
    }

    /// 创建零初始化的推理状态（web-rwkv 风格，供 `forward_with_state` 使用）。
    /// `State` 是会话持久化与 state tuning 的第一等公民：`state_back` 取态 → 存盘 →
    /// `state_load` 回灌。
    pub fn create_state(&mut self) -> R<State> {
        State::new(
            self.backend.as_mut(),
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
            self.config.n_layer,
        )
    }

    /// 权重共享并发实例：新后端（独立 CUDA stream）导入本实例的全部设备张量
    ///（同一 primary ctx，设备指针跨实例有效；权重显存零复制），工作缓冲/seq
    /// 缓冲/状态为本实例私有。配合 `submit/collect_sample_selfloop` 的拆分式
    /// API，单线程即可让 K 个实例的流并发执行（提交即返回，收集时各自同步）。
    /// 仅 CUDA 后端支持；实例须在创建它的线程上使用（cuCtxSetCurrent per-thread）。
    pub fn build_shared(&self) -> R<GpuModel> {
        let mut backend = crate::backend::create_backend(crate::backend::detect_backend())?;
        backend.import_tensors_from(self.backend.as_ref())?;
        let config = self.config.clone();
        let bufs = WorkBuffers::new(backend.as_mut(), config.n_embd, config.vocab, &config, 1)?;
        let state = State::new(
            backend.as_mut(),
            config.n_embd,
            config.n_head,
            config.head_size,
            config.n_layer,
        )?;
        Ok(GpuModel {
            backend,
            config,
            emb_ln: self.emb_ln,
            emb_ln_cpu: std::sync::Arc::clone(&self.emb_ln_cpu),
            ln_out_w: self.ln_out_w,
            ln_out_b: self.ln_out_b,
            head_w16: self.head_w16,
            head_a8: self.head_a8,
            layers: self.layers.clone(),
            bufs,
            batch_bufs: None,
            batch_lens: None,
            seq_bufs: None,
            batch_seq_bufs: None,
            prefill_state: None,
            batch_step_tmp: None,
            selfloop_bufs: Vec::new(),
            state: Some(state),
            scale_w: self.scale_w,
        })
    }

    /// 公开模型元信息（web-rwkv 风格 `ModelInfo`），供服务端读取模型规模。
    pub fn info(&self) -> ModelInfo {
        self.config.info()
    }

    /// 后端登记在册张量的设备字节总数（诊断用：分阶段比较显存增量）。
    pub fn device_bytes(&self) -> usize {
        self.backend.device_bytes()
    }

    /// 分阶段显存报告（诊断用，单位 MiB）。调用点在关键节点，便于定位大户。
    pub fn log_vram(&self, stage: &str) {
        log::info!(
            "[VRAM] {stage:<22} 登记张量 {} MiB",
            self.device_bytes() / (1024 * 1024)
        );
    }

    /// 把 `State` 整态下载到 CPU 为连续 `Vec<f32>`（布局与 `state_load` 一一对应）。
    pub fn state_back(&self, state: &State) -> R<Vec<f32>> {
        state.back(
            &*self.backend,
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
        )
    }

    /// 从 CPU `Vec<f32>` 回灌 `State`（与 `state_back` 布局对应）。
    pub fn state_load(&self, state: &State, data: &[f32]) -> R<()> {
        state.load(
            &*self.backend,
            data,
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
        )
    }

    /// 前向推理：返回最后一个 token 的 logits [vocab]。
    /// 便捷封装：复用内部状态（等价于 `forward_with_state(&mut self.state, tokens)`）。
    pub fn forward(&mut self, tokens: &[u32]) -> R<Vec<f32>> {
        let mut state = self.state.take().unwrap();
        let out = self.forward_with_state(&mut state, tokens);
        self.state = Some(state);
        out
    }

    /// 前向推理（web-rwkv 风格）：接受外部 `State`，返回最后一个 token 的 logits [vocab]。
    /// `State` 由调用方持有/保存，`forward_with_state` 在其上累积 RNN 状态，
    /// 支撑会话持久化与 state tuning（`State::back` → 存盘 → `State::load`）。
    pub fn forward_with_state(&mut self, state: &mut State, tokens: &[u32]) -> R<Vec<f32>> {
        for &token in tokens {
            self.forward_token(token, false, None, &[], state)?;
        }
        let logits = self.backend.download(self.bufs.logits)?;
        Ok(logits)
    }

    /// 前向推理 + GPU 采样：返回最后一个 token 的 argmax 索引（全 GPU，不下载 logits）。
    /// 对标 albatross 的 torch.argmax：只把 4 字节的 token 索引回传，省去每 token 下载
    /// 65536 个 f32 logits（256KB）与 CPU 遍历。
    pub fn forward_argmax(&mut self, tokens: &[u32]) -> R<u32> {
        let mut state = self.state.take().unwrap();
        let out = self.forward_argmax_with_state(&mut state, tokens);
        self.state = Some(state);
        out
    }

    /// forward_argmax 的 state 显式版本（供内部封装复用）。
    fn forward_argmax_with_state(&mut self, state: &mut State, tokens: &[u32]) -> R<u32> {
        for &token in tokens {
            self.forward_token(token, true, None, &[], state)?;
        }
        let t = self.backend.download(self.bufs.token_argmax)?;
        // shader 向 f32 缓冲写入 uint，回读时按位解释为 u32
        Ok(t[0].to_bits())
    }

    /// 前向推理 + GPU 采样：推进整段 tokens，返回最后一个 token 的采样索引（全 GPU）。
    /// 在 logits 上做 penalty/temperature/top-k/top-p 过滤后按概率采样，只回传 4 字节索引。
    /// 采样参数由 `SamplerParams` 携带；`seed` 由调用方控制（每 token 生成应递增）。
    /// `history` 为已生成 token（惩罚计数用，空则跳过惩罚）。
    fn forward_sample_with_state(
        &mut self,
        state: &mut State,
        tokens: &[u32],
        sp: &SamplerParams,
        history: &[u32],
    ) -> R<u32> {
        for &token in tokens {
            self.forward_token(token, false, Some(sp), history, state)?;
        }
        let t = self.backend.download(self.bufs.token_argmax)?;
        Ok(t[0].to_bits())
    }

    /// GPU self-loop 批量采样生成：在**单次 submit** 内连续采样 n 个 token。
    /// 与 argmax 版 self-loop 同构，但每轮用带参数（temperature/top-k/top-p）的采样替换
    /// argmax；采样临时缓冲（temp/mask/sampler）预建一次存活到 batch 提交后，每轮
    /// 用 pinned 异步上传更新 seed 后复用。返回生成的 n 个 token 索引。
    pub fn forward_sample_selfloop_with_state(
        &mut self,
        state: &mut State,
        seed: u32,
        n: usize,
        sp: &SamplerParams,
    ) -> R<Vec<u32>> {
        let ticket = self.submit_sample_selfloop(state, seed, n, sp)?;
        self.collect_sample_selfloop(ticket)
    }

    /// 在途 self-loop 句柄：submit 后全部轮次已在 GPU 流上排队（零 host 同步），
    /// collect 时一次同步取回生成序列并释放临时缓冲。
    /// 单线程使用约束：同一后端至多一个在途 ticket（pinned 行号复用前提）。
    pub fn submit_sample_selfloop(
        &mut self,
        state: &mut State,
        seed: u32,
        n: usize,
        sp: &SamplerParams,
    ) -> R<SelfloopTicket> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let ns = self.config.head_size;

        // 常驻缓冲（按 (kind=0, batch=1, n) 缓存）：graph 内烘焙了这些指针，不能每段重建。
        let b = self.selfloop_bufs(0, 1, n)?;
        // 序列缓冲每段清零（内容每段覆盖；graph 内指针不变）。
        self.backend.upload(b.token_seq, &vec![0.0; n])?;
        self.backend.upload(b.seq_cnt, &[0.0; 1])?;

        // 开启批处理：整段 self-loop 所有 kernel 一次性记录 + 提交
        self.backend.begin_batch()?;

        // 首个 token 由 CPU 写入 host-visible 缓冲
        self.backend
            .store_token_host(self.bufs.current_token, seed)?;

        // 预置 round 0 的 sampler（graph 捕获时 sample kernel 从 sampler 缓冲读参数）
        self.backend.store_sampler_host(
            b.sampler,
            sp.temperature,
            sp.top_k,
            sp.top_p,
            seed,
            sp.repetition_penalty,
            sp.frequency_penalty,
            sp.presence_penalty,
            0,
        )?;

        if self.backend.supports_graph_capture() {
            // CUDA graph：捕获一轮完整前向（gather→层→ln+head→GPU采样→record_token），
            // 之后每 token 重放，消除 258 次/层的 cuLaunchKernel 启动开销。
            // **每个形状只捕获一次**，此后跨 submit 永久重放（原先每段重捕获）。
            // 每轮重放前更新 seed/hist_len，sample kernel 在重放时读取最新设备参数。
            // 逐轮更新走 pinned 异步上传（流序，零 host 同步）——同步版每轮
            // cuStreamSynchronize 会把整段 selfloop 退化成逐轮 CPU⟷GPU 往返。
            let key = b.key();
            let graph_ok = self.ensure_selfloop_graph(key, |m| {
                m.sample_selfloop_step(
                    state,
                    c,
                    h,
                    ns,
                    b.temp,
                    b.mask,
                    b.counter,
                    b.sampler,
                    b.token_seq,
                    b.seq_cnt,
                )
            });
            let use_async = n <= self.backend.sampler_async_rows();
            for round in 0..n {
                state.v_first_set = false;
                if use_async {
                    self.backend.store_sampler_async(
                        b.sampler,
                        round,
                        sp.temperature,
                        sp.top_k,
                        sp.top_p,
                        seed + round as u32,
                        sp.repetition_penalty,
                        sp.frequency_penalty,
                        sp.presence_penalty,
                        round as u32,
                    )?;
                } else {
                    self.backend.store_sampler_host(
                        b.sampler,
                        sp.temperature,
                        sp.top_k,
                        sp.top_p,
                        seed + round as u32,
                        sp.repetition_penalty,
                        sp.frequency_penalty,
                        sp.presence_penalty,
                        round as u32,
                    )?;
                }
                if graph_ok {
                    self.backend.selfloop_graph_replay(key)?;
                } else {
                    // 降级：无 graph（不支持或捕获失败），逐轮直接提交本步 kernel。
                    self.sample_selfloop_step(
                        state,
                        c,
                        h,
                        ns,
                        b.temp,
                        b.mask,
                        b.counter,
                        b.sampler,
                        b.token_seq,
                        b.seq_cnt,
                    )?;
                }
            }
        } else {
            // Vulkan：无 graph 捕获，把 n 轮完整前向逐 token 记录进同一批次；
            // 每轮用 store_sampler_host 更新 seed/hist_len（sample kernel 执行时读取最新参数）。
            for round in 0..n {
                self.backend.store_sampler_host(
                    b.sampler,
                    sp.temperature,
                    sp.top_k,
                    sp.top_p,
                    seed + round as u32,
                    sp.repetition_penalty,
                    sp.frequency_penalty,
                    sp.presence_penalty,
                    round as u32,
                )?;
                self.sample_selfloop_step(
                    state,
                    c,
                    h,
                    ns,
                    b.temp,
                    b.mask,
                    b.counter,
                    b.sampler,
                    b.token_seq,
                    b.seq_cnt,
                )?;
            }
        }

        // 一次性提交整段 self-loop（异步：不等 GPU 执行完）
        self.backend.end_batch()?;
        Ok(SelfloopTicket {
            token_seq: b.token_seq,
            n,
        })
    }

    /// 取回 self-loop 结果：同步本实例流（只等这一条流，其它实例的流不受影响），
    /// 下载生成序列。缓冲常驻（由形状缓存持有），此处不释放。
    pub fn collect_sample_selfloop(&mut self, ticket: SelfloopTicket) -> R<Vec<u32>> {
        let t = self.backend.download(ticket.token_seq)?;
        let n = ticket.n;
        Ok(t[..n].iter().map(|x| x.to_bits()).collect())
    }

    // ==== batch 并发 API（单实例多序列：1 份权重 + 1 个 batch State，一次读权重算 B 份）====

    /// 创建 batch 布局的推理状态（[batch, ...]）。仅 CUDA 后端支持 batch>1。
    pub fn create_batch_state(&mut self, batch: usize) -> R<State> {
        State::new_batched(
            self.backend.as_mut(),
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
            self.config.n_layer,
            batch,
        )
    }

    /// 分配一个外部持有的 U32 设备缓冲（并发池的惩罚历史 `hist` 等）。
    /// 由调用方负责在用完后 `release_tensor`。
    pub fn alloc_u32(&mut self, len: usize) -> R<TensorId> {
        self.backend.create_tensor(len, TensorDtype::U32)
    }

    /// 上传外部 U32 缓冲（长度须 ≤ 分配长度）。
    pub fn write_u32(&self, t: TensorId, data: &[u32]) -> R<()> {
        self.backend.upload_u32(t, data)
    }

    /// 部分上传外部 U32 缓冲：把 data 写入 [offset, offset+len) 段。
    /// 惩罚历史逐 token 追加用（O(1) 增量上传，避免每步重传整行）。
    pub fn write_u32_part(&self, t: TensorId, offset: usize, data: &[u32]) -> R<()> {
        self.backend.upload_u32_part(t, offset, data)
    }

    /// 该模型是否支持 batch 并发解码路径（要求 r/k/v 为 int8 量化权重——
    /// batch 层 kernel 仅有 int8 变体；fp16 模型须回退逐槽单序列路径）。
    pub fn supports_batch_decode(&self) -> bool {
        self.layers.first().is_some_and(|l| {
            l.receptance_a8.is_some() && l.key_a8.is_some() && l.value_a8.is_some()
        })
    }

    /// 释放外部缓冲（`alloc_u32` 的配对操作）。
    pub fn release_tensor(&mut self, t: TensorId) {
        self.backend.free_tensor(t);
    }

    /// 单 slot 状态回灌（batch State）：把单序列布局数据写入 batch_state 的
    /// 第 `slot` 段（prefill 单序列 → 状态灌入 batch slot 的桥）。
    pub fn state_slot_load(&self, batch_state: &State, slot: usize, data: &[f32]) -> R<()> {
        batch_state.slot_load(
            &*self.backend,
            slot,
            data,
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
        )
    }

    /// 单 slot 状态下载（batch State）。
    pub fn state_slot_back(&self, batch_state: &State, slot: usize) -> R<Vec<f32>> {
        batch_state.slot_back(
            &*self.backend,
            slot,
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
        )
    }

    /// 设备内单行迁移：把 batch 状态第 `from` 行拷到第 `to` 行
    /// （连续批调度槽位紧凑化用，全程设备内，不过主机）。
    pub fn state_slot_move(&mut self, batch_state: &State, from: usize, to: usize) -> R<()> {
        batch_state.slot_move(
            self.backend.as_mut(),
            from,
            to,
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
        )
    }

    /// batch prefill 公共核心：B 个 prompt（可变长，pad 到 T_pad）一次贯穿全部层，
    /// 直接更新 batch State（[batch, ...] 布局）——信天翁 B×T rows 模型：GEMM 的
    /// M 维 = B*T_pad，权重读一份算全部行。
    /// `pad_to`：Some(n) 强制 T_pad = n（调用方保证 n >= 各 prompt 长度）——固定 bt
    /// 让 seq 缓冲只建一次；None 取批内最大长度。lens 恒按**原始长度**计算
    ///（pad 行不进 state、不计入 segmean 均值）。
    /// 返回 (seeds, seq_bufs[已 take，调用方归还], lens_t, t_pad, batch, tok[调用方释放])。
    /// 仅 CUDA（seq_shift_batch/dplr_seq_batch/copy_token_batch）。状态重置由调用方负责。
    /// 取（并在 T 变化时重建）序列缓冲：`dedicated=true` 走 `batch_seq_bufs`
    /// （批量单步路径专用，避免与单序列 prefill 的 T 抖动互相 clear_cache），
    /// false 走共享的 `seq_bufs`。返回的缓冲已被 take，调用方负责归还。
    fn take_seq_bufs(&mut self, bt: usize, dedicated: bool) -> R<SeqBuffers> {
        let c = self.config.n_embd;
        let vocab = self.config.vocab;
        let mut cur = if dedicated {
            self.batch_seq_bufs.take()
        } else {
            self.seq_bufs.take()
        };
        if cur.as_ref().is_none_or(|sb| sb.t != bt) {
            self.backend.clear_cache();
            if let Some(mut old) = cur.take() {
                old.free(self.backend.as_mut());
            }
            cur = Some(SeqBuffers::new(
                self.backend.as_mut(),
                bt,
                c,
                vocab,
                &self.config,
            )?);
        }
        Ok(cur.expect("seq_bufs 已重建"))
    }

    fn forward_seq_batch_core(
        &mut self,
        state: &mut State,
        tokens_batch: &[Vec<u32>],
        pad_to: Option<usize>,
        dedicated_bufs: bool,
    ) -> R<(Vec<u32>, SeqBuffers, TensorId, usize, usize, TensorId)> {
        let batch = tokens_batch.len();
        if batch == 0 {
            return Err("forward_seq_batch: 空 batch".into());
        }
        if state.batch() != batch {
            return Err(format!(
                "forward_seq_batch: state batch {} != tokens {}",
                state.batch(),
                batch
            )
            .into());
        }
        let (c, n) = (self.config.n_embd, self.config.head_size);
        let t_pad = pad_to
            .filter(|p| *p >= 1)
            .unwrap_or_else(|| tokens_batch.iter().map(|v| v.len()).max().unwrap_or(1))
            .max(tokens_batch.iter().map(|v| v.len()).max().unwrap_or(1))
            .max(1);
        let bt = batch * t_pad;

        // 摊平 tokens（slot 主序，pad 0）与 lens（恒为原始长度——pad 行不进 state）。
        let mut flat = vec![0u32; bt];
        let lens: Vec<u32> = tokens_batch
            .iter()
            .map(|v| v.len().max(1).min(t_pad) as u32)
            .collect();
        for (b, toks) in tokens_batch.iter().enumerate() {
            let l = toks.len().max(1).min(t_pad);
            flat[b * t_pad..b * t_pad + l].copy_from_slice(&toks[..l]);
        }
        // seeds：每 slot 末 token（pad 段之前的实际末位）。
        let seeds: Vec<u32> = tokens_batch.iter().map(|v| *v.last().unwrap()).collect();

        // lens 缓冲（batch 变化时重建；内容每次覆盖上传）。
        let lens_t = match self.batch_lens {
            Some((b, t)) if b == batch => t,
            _ => {
                if let Some((_, old)) = self.batch_lens.take() {
                    self.backend.free_tensor(old);
                }
                let t = self.backend.create_tensor(batch, TensorDtype::U32)?;
                self.batch_lens = Some((batch, t));
                t
            }
        };
        self.backend.upload_u32(lens_t, &lens)?;

        // token 缓冲 + embedding gather（一次取 B*T_pad 行）。
        let tok = self.backend.create_tensor(bt, TensorDtype::U32)?;
        self.backend.upload_u32(tok, &flat)?;

        // 序列缓冲按 bt 重建（t 维换 B*T_pad，布局完全兼容）。
        let mut seq_bufs = self.take_seq_bufs(bt, dedicated_bufs)?;

        self.backend.begin_batch()?;
        self.backend
            .gather_rows_device_f16(self.emb_ln, seq_bufs.x, tok, c, bt)?;
        for i in 0..self.config.n_layer {
            self.forward_seq_layer_batch(i, t_pad, c, n, &mut seq_bufs, state, lens_t, batch)?;
        }
        self.backend.end_batch()?;

        // 调用方负责：释放 tok、归还 seq_bufs（mean-hidden 路径在归还前还需 segmean）。
        Ok((seeds, seq_bufs, lens_t, t_pad, batch, tok))
    }

    /// batch prefill（公共入口，decode 用）：更新 batch State，返回每 slot 末
    /// token（decode seed）。调用方须自行重置状态（每轮独立 prompt 从零态起步）。
    pub fn forward_seq_batch(
        &mut self,
        state: &mut State,
        tokens_batch: &[Vec<u32>],
    ) -> R<Vec<u32>> {
        let (seeds, seq_bufs, _lens_t, _t_pad, _batch, tok) =
            self.forward_seq_batch_core(state, tokens_batch, None, false)?;
        self.backend.free_tensor(tok);
        self.seq_bufs = Some(seq_bufs);
        Ok(seeds)
    }

    /// 批量 prefill（固定 T_pad 版）：seq 缓冲按 `pad_to` 建一次、跨批复用，
    /// 消除变长 prompt 逐批 clear_cache + kernel 重编译。lens 仍按各槽原始
    /// 长度（pad 行不进 state）；返回各槽末 token（解码种子）。
    /// 服务端应固定 pad_to（如 512）并保证 prompt 不超限，超限时 T_pad 自动
    /// 抬升一次（重建一次缓冲，后续复用）。
    pub fn forward_seq_batch_padded(
        &mut self,
        state: &mut State,
        tokens_batch: &[Vec<u32>],
        pad_to: usize,
    ) -> R<Vec<u32>> {
        let limit = prefill_chunk_slots();
        if limit == 0 || tokens_batch.len() <= limit {
            let (seeds, seq_bufs, _lens_t, _t_pad, _batch, tok) =
                self.forward_seq_batch_core(state, tokens_batch, Some(pad_to), false)?;
            self.backend.free_tensor(tok);
            self.seq_bufs = Some(seq_bufs);
            return Ok(seeds);
        }
        // —— 分块 prefill：把 **prefill 宽度** 与 **解码宽度** 解耦 ——
        //
        // 动机（2026-09-21 实测）：`SeqBuffers` 有 ~45 个 `[batch×T_pad, ·]` 缓冲（多数 fp32），
        // 大小 ∝ 解码并发 × T_pad，且**解码期完全不用却常驻**。B=64/T_pad=160 时实测
        // 占 4.4 GB（模型 5.1 GB + 状态 1.3 GB 之外），B=256 时必然 OOM
        //（实测 B=128 峰值已 20.7 GB / 22 GB）。
        //
        // 做法与 Albatross 的 `copy_state_to_batch` 同构：各槽 prefill 相互独立，
        // 故可按宽度 `limit` 逐块 prefill（用一块小状态），再把状态行灌进大 batch state。
        // 峰值工作区从 `解码宽度×T_pad` 降到 `limit×T_pad`。
        let w = limit.min(tokens_batch.len());
        let mut sub = match self.prefill_state.take() {
            Some((w0, s)) if w0 == w => s,
            _ => self.create_batch_state(w)?,
        };
        let mut seeds: Vec<u32> = Vec::with_capacity(tokens_batch.len());
        for (ci, chunk) in tokens_batch.chunks(w).enumerate() {
            // 末块不足 w：用最后一个 prompt 补齐到 w（`forward_seq_batch_core` 要求
            // state.batch()==batch）；补齐槽的结果不参与状态回灌，只是少量冗余计算。
            let mut padded: Vec<Vec<u32>> = chunk.to_vec();
            while padded.len() < w {
                padded.push(chunk.last().cloned().unwrap_or_else(|| vec![0]));
            }
            self.reset_state_of(&sub)?;
            let (s, seq_bufs, _l, _t, _b, tok) =
                self.forward_seq_batch_core(&mut sub, &padded, Some(pad_to), false)?;
            self.backend.free_tensor(tok);
            self.seq_bufs = Some(seq_bufs); // 下块同 bt 直接复用
            seeds.extend(s.into_iter().take(chunk.len()));
            for i in 0..chunk.len() {
                let d = self.state_slot_back(&sub, i)?;
                self.state_slot_load(state, ci * w + i, &d)?;
            }
        }
        self.prefill_state = Some((w, sub));
        Ok(seeds)
    }

    /// 批量 mean-hidden 特征提取：batch prefill 全层前向后，对每 slot 的末层残差
    /// x 取 lens 感知 token 均值（语义对齐单序列 forward_seq_mean_hidden，仅 fp32
    /// 累加顺序不同）。内部先重置状态（t=0 边界读取 tmix_x 须零态起步）。
    /// `pad_to`：Some(n) 固定 T_pad（bt 恒定 → seq 缓冲只建一次；lens 仍按原始
    /// 长度，pad 行不计入均值）；None 取批内最大长度。
    pub fn forward_seq_batch_mean_hidden(
        &mut self,
        state: &mut State,
        tokens_batch: &[Vec<u32>],
        pad_to: Option<usize>,
    ) -> R<Vec<Vec<f32>>> {
        self.reset_state_of(state)?;
        let (_seeds, seq_bufs, lens_t, t_pad, batch, tok) =
            self.forward_seq_batch_core(state, tokens_batch, pad_to, false)?;
        let c = self.config.n_embd;
        let out = self.backend.create_tensor(batch * c, TensorDtype::F32)?;
        self.backend
            .segmean(seq_bufs.x, out, lens_t, c, t_pad, batch)?;
        let flat = self.backend.download(out)?;
        self.backend.free_tensor(out);
        self.backend.free_tensor(tok);
        self.seq_bufs = Some(seq_bufs);
        Ok(flat.chunks(c).map(<[f32]>::to_vec).collect())
    }

    /// **批量单步采样**（并发池的 host-in-the-loop 路径）：B 个 slot 在同一步各自
    /// 喂入 `tokens[b]`（长度 1 = decode；>1 = chunked prefill），一次批量前向
    ///（权重读一份算 B 份）+ 一次批量 GPU 采样，返回每 slot 采样的下一个 token。
    ///
    /// 与 `submit_sample_selfloop_batch` 的区别：那个是「一次提交跑完 n 轮、全程无
    /// 主机介入」（graph 捕获），这个是「每步回主机一次」——stop 串判定 / 流式增量 /
    /// 会话缓存回写都在主机侧（D5），因此不能捕获 CUDA graph。
    ///
    /// `hist`：[batch, hist_stride] U32 惩罚历史（每 slot 前 `hist_len[b]` 个有效）。
    /// 本步采样**不计入**惩罚（语义与客户端池 `sample_token` 一致），调用方须在
    /// 返回后自行把新 token 追加到历史。`sp[b].seed` 每步应递增（保证多样性）。
    ///
    /// 仅 CUDA（batch 层 kernel 与 `sample_into_host_seeded_batch` 均仅 CUDA）。
    /// `sp[b]` 为该 slot 的采样参数（并发池各任务参数不同）；`sp.len()` 须 = batch。
    /// `batch ≤ state.batch()`：只用 batch State 的前 batch 行（并发池活跃槽前缀）。
    pub fn batch_step_sample(
        &mut self,
        state: &mut State,
        tokens: &[Vec<u32>],
        hist: TensorId,
        hist_stride: usize,
        hist_len: &[u32],
        sp: &[SamplerParams],
    ) -> R<Vec<u32>> {
        let batch = tokens.len();
        if hist_len.len() != batch || sp.len() != batch {
            return Err(format!(
                "batch_step_sample: hist_len {} / sp {} != tokens {}",
                hist_len.len(),
                sp.len(),
                batch
            )
            .into());
        }
        if hist_len.iter().any(|&l| l as usize > hist_stride) {
            return Err(
                format!("batch_step_sample: hist_len 超出 hist_stride {hist_stride}").into(),
            );
        }
        let (vocab, tmp) = (self.config.vocab, self.batch_step_tmp(batch)?);
        let bb = self.batch_step_forward(state, tokens)?;
        // 采样参数行 [batch, 10]（hist_len 走 sampler[7]，seed 走 sampler[3]）。
        let mut sdata = Vec::with_capacity(batch * 10);
        for (b, &hl) in hist_len.iter().enumerate() {
            let p = &sp[b];
            sdata.extend_from_slice(&[
                p.temperature,
                f32::from_bits(p.top_k),
                p.top_p,
                f32::from_bits(p.seed),
                p.repetition_penalty,
                p.frequency_penalty,
                p.presence_penalty,
                f32::from_bits(hl),
                p.penalty_decay,
                0.0,
            ]);
        }
        self.backend.upload(tmp.sampler, &sdata)?;
        self.backend.begin_batch()?;
        self.backend.sample_into_host_seeded_batch(
            bb.logits,
            bb.current_token,
            vocab,
            tmp.temp,
            tmp.mask,
            tmp.counter,
            tmp.sampler,
            hist,
            batch,
            hist_stride,
        )?;
        let raw = self.backend.download(self.bufs_current_token_of(&bb))?;
        self.backend.end_batch()?;
        Ok(raw.iter().map(|v| f32::to_bits(*v)).collect())
    }

    /// 批量单步前向（`batch_step_sample` 的前半段）：喂入各 slot 的 token、
    /// 批量贯穿全部层、写 ln_out+head，返回 batch 工作缓冲（logits 在 `bb.logits`
    /// [batch, vocab]）。诊断 / 测试用（`batch_step_logits`）。
    /// `batch ≤ state.batch()`：只用 batch State 的前 batch 行，多余行不动
    ///（并发池活跃槽恒为前缀，槽位回收/新增只改前缀长度）。
    fn batch_step_forward(&mut self, state: &mut State, tokens: &[Vec<u32>]) -> R<WorkBuffers> {
        use crate::model::LN_EPS;
        let batch = tokens.len();
        if batch == 0 {
            return Err("batch_step_forward: 空 batch".into());
        }
        if batch > state.batch() {
            return Err(format!(
                "batch_step_forward: state batch {} < tokens {}",
                state.batch(),
                batch
            )
            .into());
        }
        if tokens.iter().any(|t| t.is_empty()) {
            return Err("batch_step_forward: 每 slot 至少 1 个 token".into());
        }
        let (c, h, ns, vocab) = (
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
            self.config.vocab,
        );
        let bb = self.batch_buffers(batch)?;
        // 全 slot 单 token = decode 快路径（逐 slot 单步 kernel，无 pad 浪费）；
        // 否则走变长 seq 路径（prefill 分块 / 混批），从各 slot 末 token 取残差。
        let decode_only = tokens.iter().all(|t| t.len() == 1);

        self.backend.begin_batch()?;
        if decode_only {
            let flat: Vec<f32> = tokens.iter().map(|t| f32::from_bits(t[0])).collect();
            self.backend
                .upload(self.bufs_current_token_of(&bb), &flat)?;
            state.v_first_set = false;
            self.backend
                .gather_rows_device_f16(self.emb_ln, bb.x, bb.current_token, c, batch)?;
            for i in 0..self.config.n_layer {
                self.forward_layer_batch(i, c, h, ns, batch, state, &bb)?;
            }
        } else {
            // 变长 seq 路径：用专用 batch_seq_bufs（与单序列 prefill 的 T 抖动隔离）。
            let (_seeds, seq_bufs, lens_t, t_pad, _b, tok) =
                self.forward_seq_batch_core(state, tokens, None, true)?;
            // 各 slot 末 token 的残差 → bb.x（batch 布局），后续 norm/head/采样共用。
            self.backend
                .copy_token_batch(seq_bufs.x, bb.x, lens_t, c, t_pad, batch)?;
            self.backend.free_tensor(tok);
            self.batch_seq_bufs = Some(seq_bufs);
        }

        self.backend.norm(
            bb.x,
            self.ln_out_w,
            self.ln_out_b,
            bb.x_norm,
            c,
            1,
            LN_EPS,
            batch,
        )?;
        if let Some(a8) = &self.head_a8 {
            self.backend
                .gemv_int8_plain(a8, bb.x_norm, bb.logits, vocab, c, batch)?;
        } else {
            let w16 = self.head_w16.expect("head 既无 int8 也无 fp16 权重");
            self.backend
                .gemv_f16(w16, bb.x_norm, bb.logits, vocab, c, batch)?;
        }
        self.backend.end_batch()?;
        Ok(bb)
    }

    /// 诊断 / 测试：批量单步前向的 logits（[batch, vocab]，slot 主序）。
    /// 与 `batch_step_sample` 共用前向核心（采样之前的中间量一致）。
    pub fn batch_step_logits(&mut self, state: &mut State, tokens: &[Vec<u32>]) -> R<Vec<f32>> {
        let bb = self.batch_step_forward(state, tokens)?;
        self.backend.download(bb.logits)
    }

    /// batch prefill 单层：全 kernel 的 token 维 = B*T_pad（信天翁 rows 模型），
    /// 状态类 kernel 走 batch 变体（slot 边界 / lens 截断）。
    #[allow(clippy::too_many_arguments)]
    fn forward_seq_layer_batch(
        &mut self,
        i: usize,
        t_pad: usize,
        c: usize,
        n: usize,
        sb: &mut SeqBuffers,
        state: &mut State,
        lens: TensorId,
        batch: usize,
    ) -> R<()> {
        use crate::model::LN_EPS;
        let fh = self.config.ffn_hidden;
        let (wwp, aap, vvp, ggp) = (
            self.config.w_mid_pad,
            self.config.a_mid_pad,
            self.config.v_mid_pad,
            self.config.g_mid_pad,
        );
        let layer = &self.layers[i];
        let bt = batch * t_pad;
        let m_pad = sb.m_pad;

        // ===== Time Mixing =====
        self.backend
            .norm(sb.x, layer.ln1_w, layer.ln1_b, sb.ln1, c, 1, LN_EPS, bt)?;

        // token shift（slot 边界 t=0 读该 slot 的 tmix_x 段）
        self.backend.seq_shift_batch(
            sb.ln1,
            state.layers[i].tmix_x,
            layer.x_r,
            sb.xr,
            c,
            t_pad,
            c,
            c,
            batch,
        )?;
        self.backend.seq_shift_batch(
            sb.ln1,
            state.layers[i].tmix_x,
            layer.x_w,
            sb.xw,
            c,
            t_pad,
            c,
            c,
            batch,
        )?;
        self.backend.seq_shift_batch(
            sb.ln1,
            state.layers[i].tmix_x,
            layer.x_k,
            sb.xk,
            c,
            t_pad,
            c,
            c,
            batch,
        )?;
        self.backend.seq_shift_batch(
            sb.ln1,
            state.layers[i].tmix_x,
            layer.x_v,
            sb.xv,
            c,
            t_pad,
            c,
            c,
            batch,
        )?;
        self.backend.seq_shift_batch(
            sb.ln1,
            state.layers[i].tmix_x,
            layer.x_a,
            sb.xa,
            c,
            t_pad,
            c,
            c,
            batch,
        )?;
        self.backend.seq_shift_batch(
            sb.ln1,
            state.layers[i].tmix_x,
            layer.x_g,
            sb.xg,
            c,
            t_pad,
            c,
            c,
            batch,
        )?;
        // state.tmix_x = 每 slot 的 ln1 末行（lens-1）
        self.backend
            .copy_token_batch(sb.ln1, state.layers[i].tmix_x, lens, c, t_pad, batch)?;

        // r/k/v = token 并行 GEMM（M = B*T_pad；int8 → dequant scratch → TC GEMM）
        if let (Some(ra8), Some(ka8), Some(va8)) =
            (&layer.receptance_a8, &layer.key_a8, &layer.value_a8)
        {
            self.backend.to_f16_triple(
                sb.xr, sb.xk, sb.xv, sb.xr16, sb.xk16, sb.xv16, c, bt, m_pad, c, c,
            )?;
            self.backend.dequant_int8_to_f16(ra8, sb.w_scratch, c, c)?;
            self.backend
                .gemm(sb.xr16, sb.w_scratch, sb.r, m_pad, c, c)?;
            self.backend.dequant_int8_to_f16(ka8, sb.w_scratch, c, c)?;
            self.backend
                .gemm(sb.xk16, sb.w_scratch, sb.k, m_pad, c, c)?;
            self.backend.dequant_int8_to_f16(va8, sb.w_scratch, c, c)?;
            self.backend
                .gemm(sb.xv16, sb.w_scratch, sb.v, m_pad, c, c)?;
        } else {
            self.backend.to_f16_triple(
                sb.xr, sb.xk, sb.xv, sb.xr16, sb.xk16, sb.xv16, c, bt, m_pad, c, c,
            )?;
            let r16 = layer
                .receptance_w16
                .as_ref()
                .ok_or("receptance_w16 missing")?;
            let k16 = layer.key_w16.as_ref().ok_or("key_w16 missing")?;
            let v16 = layer.value_w16.as_ref().ok_or("value_w16 missing")?;
            self.backend.gemm(sb.xr16, *r16, sb.r, m_pad, c, c)?;
            self.backend.gemm(sb.xk16, *k16, sb.k, m_pad, c, c)?;
            self.backend.gemm(sb.xv16, *v16, sb.v, m_pad, c, c)?;
        }

        // 低秩 w/a/g 第一级投影输入转 fp16
        self.backend.to_f16_triple(
            sb.xw, sb.xa, sb.xg, sb.xw16, sb.xa16, sb.xg16, c, bt, m_pad, c, c,
        )?;

        // v_first 逻辑（layer 0 快照；layer>0 交叉混合）
        if i == 0 {
            self.backend.copy_device(sb.v, sb.v_first)?;
        } else {
            self.backend
                .gemm(sb.xv16, layer.v1_16, sb.v_mid, m_pad, vvp, c)?;
            self.backend
                .to_f16(sb.v_mid, sb.v_mid16, vvp, bt, m_pad, vvp, vvp)?;
            self.backend
                .gemm_bias(sb.v_mid16, layer.v2_16, layer.v0, sb.v_full, m_pad, c, vvp)?;
            self.backend
                .elementwise_sigmoid(sb.v_full, sb.gate, c, bt)?;
            self.backend
                .v_first_lerp(sb.v, sb.gate, sb.v_first, c, bt, c)?;
        }

        // w 链：tanh(xw@w1)@w2 + w0 → sigmoid → exp(scale)
        self.backend
            .gemm_tanh(sb.xw16, layer.w1_16, sb.w_mid, m_pad, wwp, c)?;
        self.backend
            .to_f16(sb.w_mid, sb.w_mid16, wwp, bt, m_pad, wwp, wwp)?;
        self.backend
            .gemm_bias(sb.w_mid16, layer.w2_16, layer.w0, sb.w_full, m_pad, c, wwp)?;
        self.backend
            .elementwise_sigmoid(sb.w_full, sb.w_sig, c, bt)?;
        self.backend
            .elementwise_scale_exp(sb.w_sig, self.scale_w, sb.w, c, bt)?;

        // a 链：xa@a1@a2 + a0 → sigmoid
        self.backend
            .gemm(sb.xa16, layer.a1_16, sb.a_mid, m_pad, aap, c)?;
        self.backend
            .to_f16(sb.a_mid, sb.a_mid16, aap, bt, m_pad, aap, aap)?;
        self.backend
            .gemm_bias(sb.a_mid16, layer.a2_16, layer.a0, sb.a_full, m_pad, c, aap)?;
        self.backend.elementwise_sigmoid(sb.a_full, sb.a, c, bt)?;

        // 融合 k/k_a
        self.backend.fuse_ka(
            sb.k,
            layer.k_k,
            sb.a,
            layer.k_a,
            sb.k_mod,
            sb.kk_l2,
            sb.b_vec,
            self.config.n_head,
            n,
            bt,
        )?;

        // DPLR（batch 版：每 slot 独立 T 循环，lens 截断 padding）。
        // 注意 k/a 参数是 fuse_ka 的融合输出 k_mod/kk_l2（与单序列版一致），
        // 而非原始 sb.k/sb.a——曾误传原始值导致状态污染（采样出越界 token）。
        self.backend.dplr_seq_batch(
            state.layers[i].tmix_rnn,
            sb.r,
            sb.w,
            sb.k_mod,
            sb.v,
            sb.kk_l2,
            sb.b_vec,
            sb.y,
            lens,
            self.config.n_head,
            n,
            t_pad,
            c,
            batch,
        )?;

        // y_norm = group_norm(y) + sum(r*k_mod*r_k)*v
        self.backend.norm(
            sb.y,
            layer.ln_x_w,
            layer.ln_x_b,
            sb.y_norm,
            n,
            self.config.n_head,
            GN_EPS,
            bt,
        )?;
        self.backend.sum_rk_rk(
            sb.r,
            sb.k_mod,
            layer.r_k,
            sb.v,
            sb.y_norm,
            self.config.n_head,
            n,
            bt,
        )?;

        // g 链：sigmoid(xg@g1)@g2
        self.backend
            .gemm(sb.xg16, layer.g1_16, sb.g_mid, m_pad, ggp, c)?;
        self.backend
            .elementwise_sigmoid_inplace(sb.g_mid, ggp, bt)?;
        self.backend
            .to_f16(sb.g_mid, sb.g_mid16, ggp, bt, m_pad, ggp, ggp)?;
        self.backend
            .gemm(sb.g_mid16, layer.g2_16, sb.g, m_pad, c, ggp)?;

        // y_g = y_norm * g；y_out = GEMM(output_w, y_g) + x
        self.backend
            .elementwise_mul(sb.y_norm, sb.g, sb.y_g, c, bt)?;
        if let Some(a8) = &layer.att_output_a8 {
            self.backend.to_f16(sb.y_g, sb.y_g16, c, bt, m_pad, c, c)?;
            self.backend.dequant_int8_to_f16(a8, sb.w_scratch, c, c)?;
            self.backend
                .gemm_add(sb.y_g16, sb.w_scratch, sb.x, sb.y_out, m_pad, c, c)?;
        } else {
            self.backend.to_f16(sb.y_g, sb.y_g16, c, bt, m_pad, c, c)?;
            self.backend.gemm_add(
                sb.y_g16,
                *layer.output_w16.as_ref().ok_or("output_w16 missing")?,
                sb.x,
                sb.y_out,
                m_pad,
                c,
                c,
            )?;
        }
        std::mem::swap(&mut sb.x, &mut sb.y_out);

        // ===== Channel Mixing =====
        self.backend
            .norm(sb.x, layer.ln2_w, layer.ln2_b, sb.ln2, c, 1, LN_EPS, bt)?;
        self.backend.seq_shift_batch(
            sb.ln2,
            state.layers[i].cmix_x,
            layer.ffn_x_k,
            sb.xb,
            c,
            t_pad,
            c,
            c,
            batch,
        )?;
        self.backend
            .copy_token_batch(sb.ln2, state.layers[i].cmix_x, lens, c, t_pad, batch)?;

        // FFN key/value（token 并行 GEMM）
        if let Some(fka8) = &layer.ffn_key_a8 {
            self.backend.to_f16(sb.xb, sb.xb16, c, bt, m_pad, c, c)?;
            self.backend
                .dequant_int8_to_f16(fka8, sb.w_scratch, fh, c)?;
            self.backend
                .gemm_relu2(sb.xb16, sb.w_scratch, sb.r2, m_pad, fh, c)?;
        } else {
            self.backend.to_f16(sb.xb, sb.xb16, c, bt, m_pad, c, c)?;
            let fk16 = layer.ffn_key_w16.as_ref().ok_or("ffn_key_w16 missing")?;
            self.backend
                .gemm_relu2(sb.xb16, *fk16, sb.r2, m_pad, fh, c)?;
        }
        if let Some(fva8) = &layer.ffn_value_a8 {
            self.backend
                .to_f16(sb.r2, sb.r2_16, fh, bt, m_pad, fh, fh)?;
            self.backend
                .dequant_int8_to_f16(fva8, sb.w_scratch, c, fh)?;
            self.backend
                .gemm_add(sb.r2_16, sb.w_scratch, sb.x, sb.v2, m_pad, c, fh)?;
        } else {
            self.backend
                .to_f16(sb.r2, sb.r2_16, fh, bt, m_pad, fh, fh)?;
            let fv16 = layer
                .ffn_value_w16
                .as_ref()
                .ok_or("ffn_value_w16 missing")?;
            self.backend
                .gemm_add(sb.r2_16, *fv16, sb.x, sb.v2, m_pad, c, fh)?;
        }
        std::mem::swap(&mut sb.x, &mut sb.v2);

        Ok(())
    }

    /// 获取 batch 工作缓冲（B 变化时重建；B>1 仅 CUDA）。返回零开销 Clone 副本。
    fn batch_buffers(&mut self, batch: usize) -> R<WorkBuffers> {
        if self.batch_bufs.as_ref().is_none_or(|(b, _)| *b != batch) {
            if let Some((_, mut old)) = self.batch_bufs.take() {
                old.free(self.backend.as_mut());
            }
            // 旧工作缓冲已释放 → 烘焙了这些指针的 **batch** 图（kind=1）全部作废。
            // **必须**清掉，否则重放旧图 = 对已释放显存做 use-after-free。
            // （顺序不可颠倒：先释放缓冲、后清图会留下悬空图。）
            // 单流图（kind 0/2）依赖常驻的单序列工作缓冲，不受影响，故只清 kind=1。
            if !self.selfloop_bufs.is_empty() {
                self.backend.clear_selfloop_graphs(Some(1));
            }
            let bufs = WorkBuffers::new(
                self.backend.as_mut(),
                self.config.n_embd,
                self.config.vocab,
                &self.config,
                batch,
            )?;
            self.batch_bufs = Some((batch, bufs));
        }
        Ok(self.batch_bufs.as_ref().unwrap().1.clone())
    }

    /// 获取批量单步采样临时缓冲（batch 变化时重建）。返回零开销 Clone 副本。
    fn batch_step_tmp(&mut self, batch: usize) -> R<BatchStepTmp> {
        if self
            .batch_step_tmp
            .as_ref()
            .is_none_or(|(b, _)| *b != batch)
        {
            if let Some((_, old)) = self.batch_step_tmp.take() {
                for t in [old.temp, old.mask, old.counter, old.sampler] {
                    self.backend.free_tensor(t);
                }
            }
            let vocab = self.config.vocab;
            let temp = self
                .backend
                .create_tensor(vocab * batch, TensorDtype::F32)?;
            let mask = self
                .backend
                .create_tensor(vocab * batch, TensorDtype::F32)?;
            let counter = self
                .backend
                .create_tensor(vocab * batch, TensorDtype::U32)?;
            let sampler = self.backend.create_tensor(10 * batch, TensorDtype::F32)?;
            self.batch_step_tmp = Some((
                batch,
                BatchStepTmp {
                    temp,
                    mask,
                    counter,
                    sampler,
                },
            ));
        }
        Ok(self.batch_step_tmp.as_ref().unwrap().1.clone())
    }

    /// 取（或新建）self-loop 常驻缓冲，按 `(kind, batch, n)` 缓存。
    ///
    /// 形状数超过 [`MAX_SELFLOOP_SHAPES`] 时整体重置：先作废全部已捕获图，再释放旧
    /// 缓冲（顺序不可颠倒——图里烘焙了缓冲指针，先释放会留下悬空图）。
    fn selfloop_bufs(&mut self, kind: u64, batch: usize, n: usize) -> R<SelfloopBufs> {
        if let Some(b) = self
            .selfloop_bufs
            .iter()
            .find(|b| b.kind == kind && b.batch == batch && b.n == n)
        {
            return Ok(b.clone());
        }
        if self.selfloop_bufs.len() >= MAX_SELFLOOP_SHAPES {
            self.backend.clear_selfloop_graphs(None);
            let old = std::mem::take(&mut self.selfloop_bufs);
            for b in old {
                for t in b.all() {
                    self.backend.free_tensor(t);
                }
            }
        }
        let vocab = self.config.vocab;
        let temp = self
            .backend
            .create_tensor(vocab * batch, TensorDtype::F32)?;
        let mask = self
            .backend
            .create_tensor(vocab * batch, TensorDtype::F32)?;
        let counter = self
            .backend
            .create_tensor(vocab * batch, TensorDtype::U32)?;
        let sampler = self.backend.create_tensor(10 * batch, TensorDtype::F32)?;
        let token_seq = self.backend.create_tensor(n * batch, TensorDtype::F32)?;
        let seq_cnt = self.backend.create_tensor(batch, TensorDtype::F32)?;
        let bufs = SelfloopBufs {
            kind,
            batch,
            n,
            temp,
            mask,
            counter,
            sampler,
            token_seq,
            seq_cnt,
        };
        self.selfloop_bufs.push(bufs.clone());
        Ok(bufs)
    }

    /// 捕获该形状的 self-loop 图（若尚未捕获）。
    ///
    /// 返回 `true` = 图可用（本次及后续同形状直接重放）；`false` = 捕获不可用，
    /// 调用方应降级为逐轮直接提交。**任何失败都只在本次告警并永久禁用该形状**，
    /// 绝不向上抛错打断服务。
    fn ensure_selfloop_graph(
        &mut self,
        key: u64,
        capture: impl FnOnce(&mut Self) -> R<()>,
    ) -> bool {
        if self.backend.selfloop_graph_ready(key) {
            return true;
        }
        if self.backend.selfloop_graph_skip(key) {
            return false;
        }
        if let Err(e) = self.backend.begin_selfloop_capture(key) {
            log::warn!("self-loop 图捕获不可用（key=0x{key:x}）：{e:#}；该形状降级为逐轮提交");
            return false;
        }
        if let Err(e) = capture(self) {
            // 捕获区间内出错：必须 abort 让 stream 退出捕获态，否则后续 CUDA 调用全废。
            self.backend.abort_selfloop_capture();
            log::warn!("self-loop 捕获中出错（key=0x{key:x}）：{e:#}；该形状降级为逐轮提交");
            return false;
        }
        if let Err(e) = self.backend.end_selfloop_capture() {
            log::warn!("self-loop 图实例化失败（key=0x{key:x}）：{e:#}；该形状降级为逐轮提交");
            return false;
        }
        true
    }

    /// batch 版单层前向：全部 kernel 走 batch 变体（B slot 共享权重一次读）。
    /// `bb` 为 [batch, ...] 布局的 batch 工作缓冲；`state` 为 batch State。
    #[allow(clippy::too_many_arguments)]
    fn forward_layer_batch(
        &mut self,
        i: usize,
        c: usize,
        h: usize,
        n: usize,
        batch: usize,
        state: &mut State,
        bb: &WorkBuffers,
    ) -> R<()> {
        use crate::model::LN_EPS;
        let (wm, am, vm, gm, fh) = (
            self.config.w_mid,
            self.config.a_mid,
            self.config.v_mid,
            self.config.g_mid,
            self.config.ffn_hidden,
        );
        let (wmp, amp, vmp, gmp) = (
            self.config.w_mid_pad,
            self.config.a_mid_pad,
            self.config.v_mid_pad,
            self.config.g_mid_pad,
        );
        // ===== Time Mixing =====
        self.backend.norm_lerp6_batch(
            bb.x,
            state.layers[i].tmix_x,
            self.layers[i].ln1_w,
            self.layers[i].ln1_b,
            self.layers[i].x_r,
            self.layers[i].x_w,
            self.layers[i].x_k,
            self.layers[i].x_v,
            self.layers[i].x_a,
            self.layers[i].x_g,
            bb.xr,
            bb.xw,
            bb.xk,
            bb.xv,
            bb.xa,
            bb.xg,
            c,
            LN_EPS,
            batch,
        )?;

        // r/k/v + mid 融合 gemv（batch 版；fp16 权重模型暂不支持 batch 路径）。
        // ★ `LOWRANK_GEMM`（Phase 2，默认开，见 [Phase 2 计划]）：低秩链两级改走 **fp16 张量核
        // tiled GEMM**（复用加载时已常驻的 `*_16` 权重），替代 fp32 SIMT 的
        // 「mid 投影段 + chain4」两处（97ms → 24.6ms/步，端到端 1025.5 → 1466.5 tok/s = 1.43×）。
        // 依赖：GEMV_IMMA 开（r/k/v 由 IMMA 代劳，本函数只剩 r/k/v 三段）。
        // 两个开关均**默认开**（显式 `=0` 关闭）；低秩链精度档与信天翁一致（同为 fp16 张量核）。
        let lowrank_gemm = batch >= 16
            && c.is_multiple_of(64)
            && [wmp, amp, vmp, gmp].iter().all(|p| p.is_multiple_of(64))
            && std::env::var("LOWRANK_GEMM")
                .map(|v| v != "0")
                .unwrap_or(true)
            && std::env::var("GEMV_IMMA").map(|v| v != "0").unwrap_or(true);
        if let (Some(r_a8), Some(k_a8), Some(v_a8)) = (
            &self.layers[i].receptance_a8,
            &self.layers[i].key_a8,
            &self.layers[i].value_a8,
        ) {
            // LOWRANK_GEMM 时 mid 伪行数传 0 ⇒ `gemv_int8_rkv_stage1_batch` 只发 r/k/v。
            let (mvm, mwm, mam, mgm) = if lowrank_gemm {
                (0, 0, 0, 0)
            } else {
                (vm, wm, am, gm)
            };
            self.backend.gemv_int8_rkv_stage1_batch(
                r_a8,
                k_a8,
                v_a8,
                self.layers[i].v1,
                self.layers[i].w1,
                self.layers[i].a1,
                self.layers[i].g1,
                bb.xr,
                bb.xk,
                bb.xv,
                bb.xw,
                bb.xa,
                bb.xg,
                bb.r,
                bb.k,
                bb.v,
                bb.v_mid,
                bb.w_mid,
                bb.a_mid,
                bb.g_mid,
                c,
                mvm,
                mwm,
                mam,
                mgm,
                batch,
            )?;
        } else {
            return Err("batch 并发路径要求 r/k/v 为 int8 量化权重（fp16 模型暂不支持）".into());
        }

        // v_first 跨层快照（[batch, C] 全量拷贝，len 自适应）。
        if !state.v_first_set {
            self.backend.copy_device_f16(bb.v, state.v_first)?;
            state.v_first_set = true;
        }

        // 低秩链第二级 + fuse_ka_dplr_norm（batch 版）。
        if lowrank_gemm {
            // Phase 2：两级都走 fp16 张量核 GEMM（一级 mid + 二级 chain4 epilogue）。
            self.backend.lowrank_gemm_batch(
                self.layers[i].v1_16,
                self.layers[i].w1_16,
                self.layers[i].a1_16,
                self.layers[i].g1_16,
                self.layers[i].w2_16,
                self.layers[i].a2_16,
                self.layers[i].v2_16,
                self.layers[i].g2_16,
                bb.xw,
                bb.xa,
                bb.xv,
                bb.xg,
                self.layers[i].w0,
                self.layers[i].a0,
                self.layers[i].v0,
                self.scale_w,
                state.v_first,
                bb.w,
                bb.a,
                bb.v,
                bb.g,
                c,
                batch,
                wm,
                am,
                vm,
                gm,
                wmp,
                amp,
                vmp,
                gmp,
            )?;
        } else {
            self.backend.gemv_lowrank_chain4_batch(
                self.layers[i].w2,
                self.layers[i].a2,
                self.layers[i].v2,
                self.layers[i].g2,
                bb.w_mid,
                bb.a_mid,
                bb.v_mid,
                bb.g_mid,
                self.layers[i].w0,
                self.layers[i].a0,
                self.layers[i].v0,
                self.scale_w,
                state.v_first,
                bb.w,
                bb.a,
                bb.v,
                bb.g,
                h * n,
                wm,
                am,
                vm,
                gm,
                batch,
            )?;
        }
        self.backend.fuse_ka_dplr_norm_batch(
            state.layers[i].tmix_rnn,
            bb.k,
            self.layers[i].k_k,
            bb.a,
            self.layers[i].k_a,
            bb.r,
            bb.v,
            bb.w,
            self.layers[i].ln_x_w,
            self.layers[i].ln_x_b,
            self.layers[i].r_k,
            bb.k_mod,
            bb.y,
            bb.y_norm,
            h,
            n,
            EPS_L2,
            GN_EPS,
            batch,
        )?;

        // x += (y_norm .* g) @ output_w（gemv_variant 已支持 batch 维）。
        if let Some(a8) = &self.layers[i].att_output_a8 {
            self.backend
                .gemv_int8_mul_add(a8, bb.y_norm, bb.g, bb.x, c, c, batch)?;
        } else {
            let w16 = *self.layers[i]
                .output_w16
                .as_ref()
                .ok_or("output_w16 missing when not int8")?;
            self.backend
                .gemv_f16_mul_add(w16, bb.y_norm, bb.g, bb.x, c, c, batch)?;
        }

        // ===== Channel Mixing =====
        self.backend.cmix_norm_lerp_batch(
            bb.x,
            state.layers[i].cmix_x,
            self.layers[i].ln2_w,
            self.layers[i].ln2_b,
            self.layers[i].ffn_x_k,
            bb.xb,
            c,
            LN_EPS,
            batch,
        )?;

        // r2 = relu²(xb @ ffn_key.T)
        if let Some(a8) = &self.layers[i].ffn_key_a8 {
            self.backend
                .gemv_int8_relu2(a8, bb.xb, bb.r2, fh, c, batch)?;
        } else {
            let w16 = *self.layers[i]
                .ffn_key_w16
                .as_ref()
                .ok_or("ffn_key_w16 missing when not int8")?;
            self.backend
                .gemv_f16_relu2(w16, bb.xb, bb.r2, fh, c, batch)?;
        }
        // x += r2 @ ffn_value。★ Phase 2-3：**大 batch 走稠密 fp16 张量核 GEMM**，
        // 小 batch 才回退稀疏/量化 GEMV——稀疏 gather 的权重流量随 batch 线性涨
        // （B=256 时 13.4GB/层），而稠密 tiled 只读 `batch/BM` 遍（105MB/层），
        // 多算的 FLOP 在张量核上近乎免费（详见 `ffn_value_gemm` 内推导）。
        let dense_ffn = batch >= 16
            && fh.is_multiple_of(64)
            && c.is_multiple_of(16)
            && self.layers[i].ffn_value_dense16.is_some()
            && std::env::var("FFN_VALUE_GEMM")
                .map(|v| v != "0")
                .unwrap_or(true);
        if dense_ffn {
            // ★ Phase 2-5（2026-09-22）：优先走 **int8 IMMA**（`FFN_VALUE_IMMA`，默认开）。
            // 与 fp16 版同语义，但权重字节减半（26.2 vs 52.4 MB/层，且 `batch/BM` 遍重读
            // ⇒ 105 vs 210 MB/层），A 侧改由 `quant_x_i8` 在核内量化 ⇒ 顺带省掉 `cast_f16`。
            let use_imma = self.layers[i].ffn_value_a8.is_some()
                && self.backend.supports_ffn_value_imma(c, fh, batch);
            if use_imma {
                let a8 = self.layers[i].ffn_value_a8.as_ref().unwrap();
                self.backend
                    .ffn_value_imma_batch(a8, bb.r2, bb.x, c, fh, batch)?;
            } else {
                let w16 = self.layers[i].ffn_value_dense16.unwrap();
                self.backend
                    .ffn_value_gemm_batch(w16, bb.r2, bb.x, c, fh, batch)?;
            }
        } else {
            let sparse_ok =
                self.backend.supports_sparse_ffn() && std::env::var("FFN_SPARSE_OFF").is_err();
            let sparse_vt = if sparse_ok {
                self.layers[i].ffn_value_tiled
            } else {
                None
            };
            if let Some(vt) = sparse_vt {
                self.backend
                    .ffn_value_sparse_add_batch(vt, bb.r2, bb.x, c, fh, batch)?;
            } else if let Some(a8) = &self.layers[i].ffn_value_a8 {
                self.backend.gemv_int8_add(a8, bb.r2, bb.x, c, fh, batch)?;
            } else {
                let w16 = *self.layers[i]
                    .ffn_value_w16
                    .as_ref()
                    .ok_or("ffn_value_w16 missing")?;
                self.backend.gemv_f16_add(w16, bb.r2, bb.x, c, fh, batch)?;
            }
        }

        Ok(())
    }

    /// batch 采样 self-loop 单步：gather_rows → 全部层 → ln_out+head → batch 采样
    /// → batch record_token。供 CUDA graph 捕获/重放（kernel 序列固定，seed 经
    /// sampler 缓冲逐轮异步更新）。
    #[allow(clippy::too_many_arguments)]
    fn sample_selfloop_step_batch(
        &mut self,
        state: &mut State,
        batch: usize,
        bb: &WorkBuffers,
        temp: TensorId,
        mask: TensorId,
        counter: TensorId,
        sampler: TensorId,
        token_seq: TensorId,
        seq_cnt: TensorId,
        seq_stride: usize,
    ) -> R<()> {
        use crate::model::LN_EPS;
        let (c, h, ns, vocab) = (
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
            self.config.vocab,
        );
        state.v_first_set = false;
        self.backend
            .gather_rows_device_f16(self.emb_ln, bb.x, bb.current_token, c, batch)?;
        for i in 0..self.config.n_layer {
            self.forward_layer_batch(i, c, h, ns, batch, state, bb)?;
        }
        self.backend.norm(
            bb.x,
            self.ln_out_w,
            self.ln_out_b,
            bb.x_norm,
            c,
            1,
            LN_EPS,
            batch,
        )?;
        // head：logits [batch, vocab]（gemv_int8_plain/gemv_f16 已支持 batch 维）。
        if let Some(a8) = &self.head_a8 {
            self.backend
                .gemv_int8_plain(a8, bb.x_norm, bb.logits, vocab, c, batch)?;
        } else {
            let w16 = self.head_w16.expect("head 既无 int8 也无 fp16 权重");
            self.backend
                .gemv_f16(w16, bb.x_norm, bb.logits, vocab, c, batch)?;
        }
        // batch 采样（token 写回 current_token[b]，下一轮 gather 自动跟随）。
        // hist = token_seq（[batch, n] 布局，前 round 个已生成 token 在各 slot 段首；
        // 每 slot 实际历史长度 = sampler[7] = round，stride 恒为 n）。
        self.backend.sample_into_host_seeded_batch(
            bb.logits,
            bb.current_token,
            vocab,
            temp,
            mask,
            counter,
            sampler,
            token_seq,
            batch,
            seq_stride,
        )?;
        self.backend
            .record_tokens(bb.current_token, token_seq, seq_cnt, seq_stride, batch)
    }

    /// batch 采样 self-loop：B 个 slot 在**同一 GpuModel/同一 batch State** 上并发
    /// 生成各 n 个 token——一次读权重算 B 份（weight-bound 场景吞吐 ≈ ×B）。
    /// CUDA graph 捕获一轮 batch 前向后逐轮重放，seed 经 pinned 宽行异步更新。
    /// `seeds`：每 slot 首个 token（如 prompt 末 token）；返回 ticket 供 collect。
    #[allow(clippy::too_many_arguments)]
    pub fn submit_sample_selfloop_batch(
        &mut self,
        state: &mut State,
        seeds: &[u32],
        n: usize,
        sp: &SamplerParams,
    ) -> R<SelfloopTicket> {
        let batch = seeds.len();
        // 注：曾放开为 batch ≤ state.batch()（解码按实际批宽跑，避免补齐空转槽）。
        // 实测在 conc 16 + 长 prompt 下触发 access violation（0xC0000005），
        // 尚未定位到具体越界点，故回退为严格相等。要做动态批宽需先审计
        // WorkBuffers/pinned sampler 行宽/State 行步长在 batch < 分配宽 时的全部
        // 索引假设（见参考/2026-09-19-batch内核权重零摊薄根因修复记录.md 第 8 节）。
        if state.batch() != batch {
            return Err(format!(
                "submit_sample_selfloop_batch: state batch {} != seeds {}",
                state.batch(),
                batch
            )
            .into());
        }
        // batch 路径依赖 CUDA 专有的 batch kernel 变体（Vulkan 未实现）——保留原语义。
        // 例外：SKIP_GRAPH 是逐 kernel 剖析/排障用的非图路径，须允许它在
        // RWKV_GRAPH=0（supports_graph_capture()==false）下继续走通。
        let skip_graph = std::env::var("SKIP_GRAPH").is_ok_and(|v| !v.is_empty());
        if !skip_graph && !self.backend.supports_graph_capture() {
            return Err(
                "submit_sample_selfloop_batch: 后端不支持 graph 捕获（batch 路径仅 CUDA）".into(),
            );
        }

        // batch 工作缓冲（B 变化时重建；重建会作废所有已捕获图，见 batch_buffers）。
        let bb = self.batch_buffers(batch)?;
        // 常驻采样缓冲（按 (kind=1, batch, n) 缓存）：graph 内烘焙了这些指针，
        // 不能每段重建（否则每段都必须重新捕获，即 churn 根因）。
        let b = self.selfloop_bufs(1, batch, n)?;
        self.backend.upload(b.token_seq, &vec![0.0; n * batch])?;
        self.backend
            .upload(b.seq_cnt, &vec![f32::from_bits(0); batch])?;

        // 首轮 token：B 个 seed 写入 current_token [batch]（位模式）。
        self.backend.upload(
            self.bufs_current_token_of(&bb),
            &seeds.iter().map(|s| f32::from_bits(*s)).collect::<Vec<_>>(),
        )?;
        // 预置 round 0 的 sampler（graph 捕获时 sample kernel 从 sampler 缓冲读参数）。
        let mut sampler0 = Vec::with_capacity(batch * 10);
        for &s in seeds {
            sampler0.extend_from_slice(&[
                sp.temperature,
                f32::from_bits(sp.top_k),
                sp.top_p,
                f32::from_bits(s),
                sp.repetition_penalty,
                sp.frequency_penalty,
                sp.presence_penalty,
                f32::from_bits(0),
                sp.penalty_decay,
                0.0,
            ]);
        }
        self.backend.upload(b.sampler, &sampler0)?;

        self.backend.begin_batch()?;
        let key = b.key();
        let graph_ok = !skip_graph
            && self.ensure_selfloop_graph(key, |m| {
                m.sample_selfloop_step_batch(
                    state,
                    batch,
                    &bb,
                    b.temp,
                    b.mask,
                    b.counter,
                    b.sampler,
                    b.token_seq,
                    b.seq_cnt,
                    n,
                )
            });
        // 逐轮：更新本轮的 seed/hist_len，然后执行（重放图，或降级直接提交本步 kernel）。
        // 异步更新只在重放路径用（降级路径本来就每轮同步上传 sampler）。
        let use_async = graph_ok && n <= self.backend.sampler_async_rows() / batch.max(1);
        for round in 0..n {
            state.v_first_set = false;
            let seeds_r: Vec<u32> = seeds.iter().map(|s| s.wrapping_add(round as u32)).collect();
            if use_async {
                self.backend.store_sampler_async_batch(
                    b.sampler,
                    round,
                    sp.temperature,
                    sp.top_k,
                    sp.top_p,
                    &seeds_r,
                    sp.repetition_penalty,
                    sp.frequency_penalty,
                    sp.presence_penalty,
                    round as u32,
                    sp.penalty_decay,
                )?;
            } else {
                let mut data = Vec::with_capacity(batch * 10);
                for &s in &seeds_r {
                    data.extend_from_slice(&[
                        sp.temperature,
                        f32::from_bits(sp.top_k),
                        sp.top_p,
                        f32::from_bits(s),
                        sp.repetition_penalty,
                        sp.frequency_penalty,
                        sp.presence_penalty,
                        f32::from_bits(round as u32),
                        sp.penalty_decay,
                        0.0,
                    ]);
                }
                self.backend.upload(b.sampler, &data)?;
            }
            if graph_ok {
                self.backend.selfloop_graph_replay(key)?;
            } else {
                // 降级：无图（SKIP_GRAPH 调试开关 / 不支持 / 捕获失败），逐轮直接提交。
                self.sample_selfloop_step_batch(
                    state,
                    batch,
                    &bb,
                    b.temp,
                    b.mask,
                    b.counter,
                    b.sampler,
                    b.token_seq,
                    b.seq_cnt,
                    n,
                )?;
            }
        }
        self.backend.end_batch()?;
        Ok(SelfloopTicket {
            token_seq: b.token_seq,
            n,
        })
    }

    /// batch collect：下载 [batch, n] 序列并按 slot 切分。
    /// 缓冲常驻（由形状缓存持有），此处不释放。
    pub fn collect_sample_selfloop_batch(
        &mut self,
        ticket: SelfloopTicket,
        batch: usize,
    ) -> R<Vec<Vec<u32>>> {
        let t = self.backend.download(ticket.token_seq)?;
        let n = ticket.n;
        let mut out = Vec::with_capacity(batch);
        for b in 0..batch {
            let seg = &t[b * n..b * n + n];
            out.push(seg.iter().map(|x| x.to_bits()).collect());
        }
        Ok(out)
    }

    /// 取 batch 工作缓冲的 current_token（submit 时写 seeds 用）。
    fn bufs_current_token_of(&self, bb: &WorkBuffers) -> TensorId {
        bb.current_token
    }

    /// GPU self-loop 批量生成：在**单次 submit** 内连续采样 n 个 token。
    /// 首个 token 由 CPU 写 host-visible 缓冲（seed），随后每轮 argmax 结果直接写回
    /// 同一 host 缓冲，下一轮 gather 自动跟随——全程无 CPU 回读/回传 token、
    /// 无每 token 一次 submit+wait，消除 CPU⟷GPU 交换与 dispatch 间同步开销。
    /// 每轮 argmax 的 token 用 record_token 追加到序列缓冲，结束后一次性下载验证。
    /// 返回生成的 n 个 token 索引（按位解释为 u32）。
    pub fn forward_argmax_selfloop(&mut self, seed: u32, n: usize) -> R<Vec<u32>> {
        let mut state = self.state.take().unwrap();
        let out = self.forward_argmax_selfloop_with_state(&mut state, seed, n);
        self.state = Some(state);
        out
    }

    /// self-loop 的 state 显式版本（供内部封装复用）。
    fn forward_argmax_selfloop_with_state(
        &mut self,
        state: &mut State,
        seed: u32,
        n: usize,
    ) -> R<Vec<u32>> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let ns = self.config.head_size;

        // 常驻缓冲（按 (kind=2, batch=1, n) 缓存）：graph 内烘焙了这些指针，不能每次重建。
        // kind=2 与 sample_selfloop（kind=0）区分——两者 kernel 序列不同（argmax vs 采样）。
        let b = self.selfloop_bufs(2, 1, n)?;
        self.backend.upload(b.token_seq, &vec![0.0; n])?;
        self.backend.upload(b.seq_cnt, &[0.0; 1])?;

        // 开启批处理：整段 self-loop 所有 kernel 一次性记录 + 提交
        self.backend.begin_batch()?;

        // 首个 token 由 CPU 写入 host-visible 缓冲
        self.backend
            .store_token_host(self.bufs.current_token, seed)?;

        if self.backend.supports_graph_capture() {
            // CUDA：捕获首个 token 的完整前向 kernel 序列，之后每 token 重放，
            // 消除 258 次/层的 cuLaunchKernel 启动开销。
            // **每个形状只捕获一次**，此后跨调用永久重放（原先每次调用重捕获）。
            let key = b.key();
            let graph_ok = self.ensure_selfloop_graph(key, |m| {
                m.selfloop_step(state, c, h, ns, b.token_seq, b.seq_cnt)
            });

            // 执行 token 0，随后重放 n-1 次
            for round in 0..n {
                if round > 0 {
                    state.v_first_set = false;
                }
                if graph_ok {
                    self.backend.selfloop_graph_replay(key)?;
                } else {
                    self.selfloop_step(state, c, h, ns, b.token_seq, b.seq_cnt)?;
                }
            }
        } else {
            // Vulkan：无 graph 捕获，把 n 轮完整前向逐 token 记录进同一批次。
            // selfloop_step 内部已重置 v_first_set，每轮会重新快照 v_first。
            for _ in 0..n {
                self.selfloop_step(state, c, h, ns, b.token_seq, b.seq_cnt)?;
            }
        }

        // 一次性提交整段 self-loop
        self.backend.end_batch()?;

        // 下载序列缓冲，按位解释为 u32
        let t = self.backend.download(b.token_seq)?;
        Ok(t[..n].iter().map(|x| x.to_bits()).collect())
    }

    /// self-loop 单步：gather → 全部层 → ln_out+head → argmax → record_token。
    /// 供 CUDA graph 捕获/重放使用（kernel 序列固定，仅数据随 token 变化）。
    #[allow(clippy::too_many_arguments)]
    fn selfloop_step(
        &mut self,
        state: &mut State,
        c: usize,
        h: usize,
        ns: usize,
        token_seq: TensorId,
        seq_cnt: TensorId,
    ) -> R<()> {
        use crate::model::LN_EPS;
        state.v_first_set = false;
        self.backend
            .gather_row_device_f16(self.emb_ln, self.bufs.x, self.bufs.current_token, c)?;
        for i in 0..self.config.n_layer {
            self.forward_layer(i, c, h, ns, state)?;
        }
        self.backend.norm(
            self.bufs.x,
            self.ln_out_w,
            self.ln_out_b,
            self.bufs.x_norm,
            c,
            1,
            LN_EPS,
            1,
        )?;
        self.head_gemv(self.bufs.x_norm, self.bufs.logits, self.config.vocab, c)?;
        self.backend.argmax_into_host(
            self.bufs.logits,
            self.bufs.current_token,
            self.config.vocab,
        )?;
        self.backend
            .record_token(self.bufs.current_token, token_seq, seq_cnt)
    }

    /// 采样 self-loop 单步：gather → 全部层 → ln_out+head → GPU 采样 → record_token。
    /// 供 CUDA graph 捕获/重放使用（kernel 序列固定，seed/hist_len 经 sampler 缓冲逐轮更新）。
    #[allow(clippy::too_many_arguments)]
    fn sample_selfloop_step(
        &mut self,
        state: &mut State,
        c: usize,
        h: usize,
        ns: usize,
        temp: TensorId,
        mask: TensorId,
        counter: TensorId,
        sampler: TensorId,
        token_seq: TensorId,
        seq_cnt: TensorId,
    ) -> R<()> {
        use crate::model::LN_EPS;
        state.v_first_set = false;
        self.backend
            .gather_row_device_f16(self.emb_ln, self.bufs.x, self.bufs.current_token, c)?;
        for i in 0..self.config.n_layer {
            self.forward_layer(i, c, h, ns, state)?;
        }
        self.backend.norm(
            self.bufs.x,
            self.ln_out_w,
            self.ln_out_b,
            self.bufs.x_norm,
            c,
            1,
            LN_EPS,
            1,
        )?;
        self.head_gemv(self.bufs.x_norm, self.bufs.logits, self.config.vocab, c)?;
        self.backend.sample_into_host_seeded(
            self.bufs.logits,
            self.bufs.current_token,
            self.config.vocab,
            temp,
            mask,
            counter,
            sampler,
            token_seq, // 前 round 个已生成 token 作为惩罚历史
        )?;
        self.backend
            .record_token(self.bufs.current_token, token_seq, seq_cnt)
    }

    /// 单 token 前向：把该 token 的 logits 写入 bufs.logits（含 batch 记录与提交）。
    /// `do_argmax` 为真时，在提交前把 logits 的 GPU argmax 索引写入 bufs.token_argmax；
    /// `sampler` 为 Some 时用带参数（temperature/top-k/top-p/penalty）的 GPU 采样写回同一缓冲；
    /// `history` 为已生成 token（惩罚计数用，采样时透传，非采样调用传空）。
    /// （均与本次前向同批记录，避免额外一次 submit；二选一，两者都提供时 sampler 优先。）
    fn forward_token(
        &mut self,
        token: u32,
        do_argmax: bool,
        sampler: Option<&SamplerParams>,
        history: &[u32],
        state: &mut State,
    ) -> R<()> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;
        let vocab = self.config.vocab;

        // 每个 token 重置 v_first
        state.v_first_set = false;

        // 开启批处理记录：整层 forward 的所有 kernel + 拷贝一次性记录
        self.backend.begin_batch()?;

        // 参数化 embedding gather：token 索引由 CPU 直接写入 host-visible 缓冲（无 kernel、
        // 无 spec constant，避免每 token 重建 pipeline），再由 gather kernel 从 emb_ln 表
        // 按索引取行 → x。索引来自 host 缓冲，循环体不依赖具体 token 值，
        // 为 GPU self-loop（argmax 直接写索引）铺路。
        self.backend
            .store_token_host(self.bufs.current_token, token)?;
        self.backend
            .gather_row_device_f16(self.emb_ln, self.bufs.x, self.bufs.current_token, c)?;

        for i in 0..self.config.n_layer {
            self.forward_layer(i, c, h, n, state)?;
            if let Ok(sl) = std::env::var("SNAP_LAYER")
                && sl.parse::<usize>().unwrap() == i
            {
                self.backend.end_batch()?;
                let xd = self.backend.download(self.bufs.x)?;
                log::info!("[SNAP] tok  layer {i} x[0..8] = {:?}", &xd[..8]);
                self.backend.begin_batch()?;
            }
        }

        // ln_out + head
        // x_norm = layer_norm(x, ln_out_w, ln_out_b)
        self.backend.norm(
            self.bufs.x,
            self.ln_out_w,
            self.ln_out_b,
            self.bufs.x_norm,
            c,
            1,
            LN_EPS,
            1,
        )?;
        // logits = x_norm @ head.T = gemv_f16(head_w16, x_norm, M=vocab, K=c)
        self.head_gemv(self.bufs.x_norm, self.bufs.logits, vocab, c)?;
        // 需要采样时，GPU 端 argmax / 带参数采样归约（与本次前向同批）
        if let Some(sp) = sampler {
            self.backend.sample(
                self.bufs.logits,
                self.bufs.token_argmax,
                vocab,
                sp.temperature,
                sp.top_k,
                sp.top_p,
                sp.seed,
                sp.repetition_penalty,
                sp.frequency_penalty,
                sp.presence_penalty,
                history,
            )?;
        } else if do_argmax {
            self.backend
                .argmax(self.bufs.logits, self.bufs.token_argmax, vocab)?;
        }

        // 一次性提交整 token 的所有计算到 GPU
        self.backend.end_batch()?;
        Ok(())
    }

    /// logits = x_in @ head.T，输出 [vocab]。
    /// 模型含 int8 量化 head 时走 `gemv_int8_plain`（省 4× 带宽，head 是 decode 单 kernel 读取量最大者），
    /// 否则走 fp16 `gemv_f16`。
    fn head_gemv(&mut self, x_in: TensorId, out: TensorId, vocab: usize, c: usize) -> R<()> {
        if let Some(a8) = &self.head_a8 {
            self.backend.gemv_int8_plain(a8, x_in, out, vocab, c, 1)
        } else {
            let w16 = self.head_w16.expect("head 既无 int8 也无 fp16 权重");
            self.backend.gemv_f16(w16, x_in, out, vocab, c, 1)
        }
    }

    /// 单层前向：time mixing + channel mixing
    fn forward_layer(
        &mut self,
        i: usize,
        c: usize,
        h: usize,
        n: usize,
        state: &mut State,
    ) -> R<()> {
        // decode 走 int8/fp16 GEMV，无需 fp16 副本；prefill 已逐层释放临时 fp16，decode 不再持有。
        let (wm, am, vm, gm, fh) = (
            self.config.w_mid,
            self.config.a_mid,
            self.config.v_mid,
            self.config.g_mid,
            self.config.ffn_hidden,
        );
        // 阶段剖析（PROFILE=1 时每个阶段独立 submit 并计时）
        let pf = std::env::var("PROFILE").is_ok();
        #[allow(unused_assignments)]
        let mut t0 = std::time::Instant::now();
        #[allow(unused_assignments)]
        macro_rules! pf_phase {
            ($name:expr) => {
                if pf {
                    self.backend.end_batch()?;
                    log::info!(
                        "[P] tok t=0 layer {i} {}: {:.3}ms",
                        $name,
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                    self.backend.begin_batch()?;
                    t0 = std::time::Instant::now();
                }
            };
        }
        // ===== Time Mixing =====
        // 深度融合：ln1 = layer_norm(x) + 6 次 lerp(xr/xw/xk/xv/xa/xg) + state.tmix_x 写回
        self.backend.norm_lerp6(
            self.bufs.x,
            state.layers[i].tmix_x,
            self.layers[i].ln1_w,
            self.layers[i].ln1_b,
            self.layers[i].x_r,
            self.layers[i].x_w,
            self.layers[i].x_k,
            self.layers[i].x_v,
            self.layers[i].x_a,
            self.layers[i].x_g,
            self.bufs.xr,
            self.bufs.xw,
            self.bufs.xk,
            self.bufs.xv,
            self.bufs.xa,
            self.bufs.xg,
            c,
            LN_EPS,
        )?;
        pf_phase!("norm_lerp6");

        // r/k/v + v_mid/w_mid/a_mid/g_mid = 融合 gemv（一次 dispatch 算 r/k/v 三个 C×C 投影
        // + v1/w1/a1/g1 四个 mid 投影）。量化权重自动路由：int8 → fp16
        // （r/k/v 三者同量化格式才走对应融合版；权重带宽 fp16 2B / int8 1B）。
        if let (Some(r_a8), Some(k_a8), Some(v_a8)) = (
            &self.layers[i].receptance_a8,
            &self.layers[i].key_a8,
            &self.layers[i].value_a8,
        ) {
            self.backend.gemv_int8_rkv_stage1(
                r_a8,
                k_a8,
                v_a8,
                self.layers[i].v1,
                self.layers[i].w1,
                self.layers[i].a1,
                self.layers[i].g1,
                self.bufs.xr,
                self.bufs.xk,
                self.bufs.xv,
                self.bufs.xw,
                self.bufs.xa,
                self.bufs.xg,
                self.bufs.r,
                self.bufs.k,
                self.bufs.v,
                self.bufs.v_mid,
                self.bufs.w_mid,
                self.bufs.a_mid,
                self.bufs.g_mid,
                c,
                vm,
                wm,
                am,
                gm,
            )?;
        } else {
            self.backend.gemv_rkv_stage1(
                *self.layers[i]
                    .receptance_w16
                    .as_ref()
                    .ok_or("receptance_w16 missing when not int8")?,
                *self.layers[i]
                    .key_w16
                    .as_ref()
                    .ok_or("key_w16 missing when not int8")?,
                *self.layers[i]
                    .value_w16
                    .as_ref()
                    .ok_or("value_w16 missing when not int8")?,
                self.layers[i].v1,
                self.layers[i].w1,
                self.layers[i].a1,
                self.layers[i].g1,
                self.bufs.xr,
                self.bufs.xk,
                self.bufs.xv,
                self.bufs.xw,
                self.bufs.xa,
                self.bufs.xg,
                self.bufs.r,
                self.bufs.k,
                self.bufs.v,
                self.bufs.v_mid,
                self.bufs.w_mid,
                self.bufs.a_mid,
                self.bufs.g_mid,
                c,
                vm,
                wm,
                am,
                gm,
            )?;
        }
        pf_phase!("gemv_rkv_stage1");
        Self::dump_tensors(
            self.backend.as_mut(),
            "tok",
            i,
            &[
                ("xr", self.bufs.xr),
                ("xk", self.bufs.xk),
                ("xv", self.bufs.xv),
                ("r", self.bufs.r),
                ("k", self.bufs.k),
            ],
        )?;

        // ===== v_first 跨层逻辑（首层把 v 快照到 v_first）=====
        if !state.v_first_set {
            self.backend.copy_device_f16(self.bufs.v, state.v_first)?;
            state.v_first_set = true;
        }
        // ===== 低秩链第二级（w/a/g/v 二级投影+激活）+ fuse_ka + dplr + group_norm + sum_rk_rk。
        // 注：曾尝试把两者深度融合为单次 dispatch（fuse_chain4_dplr_norm），实测 fp16 -7.6%、
        // int8 -12%（chain4 阶段并行度从 C/ROWS=640 workgroup 塌缩到 DPLR 的 H=40，SM 利用率
        // 不足），已回退为两次 dispatch（2026-08-09，详见参考目录融合记录）。
        //   k_mod_i = k_i * (1 + k_a_i * (a_i - 1))
        //   kk_l2_i = normalize(k_i * k_k_i)
        //   b_i     = -kk_l2_i * a_i
        //   S 更新 + y = S @ r
        //   y_norm  = group_norm(y, ln_x_w, ln_x_b) + sum(r*k_mod*r_k)*v
        self.backend.gemv_lowrank_chain4(
            self.layers[i].w2,
            self.layers[i].a2,
            self.layers[i].v2,
            self.layers[i].g2,
            self.bufs.w_mid,
            self.bufs.a_mid,
            self.bufs.v_mid,
            self.bufs.g_mid,
            self.layers[i].w0,
            self.layers[i].a0,
            self.layers[i].v0,
            self.scale_w,
            state.v_first,
            self.bufs.w,
            self.bufs.a,
            self.bufs.v,
            self.bufs.g,
            h * n,
            wm,
            am,
            vm,
            gm,
        )?;
        pf_phase!("gemv_lowrank_chain4");
        self.backend.fuse_ka_dplr_norm(
            state.layers[i].tmix_rnn,
            self.bufs.k,
            self.layers[i].k_k,
            self.bufs.a,
            self.layers[i].k_a,
            self.bufs.r,
            self.bufs.v,
            self.bufs.w,
            self.layers[i].ln_x_w,
            self.layers[i].ln_x_b,
            self.layers[i].r_k,
            self.bufs.k_mod,
            self.bufs.y,
            self.bufs.y_norm,
            h,
            n,
            EPS_L2,
            GN_EPS,
        )?;
        pf_phase!("fuse_ka_dplr_norm");
        Self::dump_tensors(
            self.backend.as_mut(),
            "tok",
            i,
            &[
                ("w", self.bufs.w),
                ("a", self.bufs.a),
                ("v_chain", self.bufs.v),
                ("g", self.bufs.g),
            ],
        )?;
        Self::dump_tensors(self.backend.as_mut(), "tok", i, &[("y", self.bufs.y)])?;
        Self::dump_tensors(
            self.backend.as_mut(),
            "tok",
            i,
            &[("y_norm", self.bufs.y_norm)],
        )?;

        // x += (y_norm .* g) @ output_w（mul + 残差累加都折叠进 gemv，省 2 次 dispatch）；
        // 量化权重自动路由：int8 → fp16
        if let Some(a8) = &self.layers[i].att_output_a8 {
            self.backend.gemv_int8_mul_add(
                a8,
                self.bufs.y_norm,
                self.bufs.g,
                self.bufs.x,
                c,
                c,
                1,
            )?;
        } else {
            self.backend.gemv_f16_mul_add(
                *self.layers[i]
                    .output_w16
                    .as_ref()
                    .ok_or("output_w16 missing when not int8")?,
                self.bufs.y_norm,
                self.bufs.g,
                self.bufs.x,
                c,
                c,
                1,
            )?;
        }
        pf_phase!("gemv_f16_mul_add");
        Self::dump_tensors(
            self.backend.as_mut(),
            "tok",
            i,
            &[("y_g", self.bufs.y_g), ("x_out", self.bufs.x)],
        )?;

        // ===== Channel Mixing =====
        // 深度融合：ln2 = layer_norm(x, ln2_w, ln2_b) + prev_c 读入 + state.cmix_x 写回 + lerp(xb)
        // xb = ln2 + ffn_x_k * (prev_c - ln2)，一次 dispatch 完成（原 4 跳）。
        self.backend.cmix_norm_lerp(
            self.bufs.x,
            state.layers[i].cmix_x,
            self.layers[i].ln2_w,
            self.layers[i].ln2_b,
            self.layers[i].ffn_x_k,
            self.bufs.xb,
            c,
            LN_EPS,
        )?;
        pf_phase!("cmix_norm_lerp");

        // r2 = relu²(xb @ ffn_key.T) — M=ffn_hidden, K=C；量化权重自动路由：int8 → fp16
        if let Some(a8) = &self.layers[i].ffn_key_a8 {
            self.backend
                .gemv_int8_relu2(a8, self.bufs.xb, self.bufs.r2, fh, c, 1)?;
        } else {
            self.backend.gemv_f16_relu2(
                *self.layers[i]
                    .ffn_key_w16
                    .as_ref()
                    .ok_or("ffn_key_w16 missing when not int8")?,
                self.bufs.xb,
                self.bufs.r2,
                fh,
                c,
                1,
            )?;
        }
        pf_phase!("gemv_f16_relu2");
        // x += r2 @ ffn_value（残差累加折叠进 gemv，省 1 次 dispatch）。
        // 优先稀疏 FFN（r2 96% 稀疏，只读非零列，省带宽）：CUDA 上 fp16/int8 均反量化出
        // fp16 平铺权重走稀疏内核（只读 ~4% 列，远优于稠密 int8 全量读取）。
        // 不支持稀疏或 FFN_SPARSE_OFF 时退回稠密 gemv（int8 → fp16）。
        let sparse_ok =
            self.backend.supports_sparse_ffn() && std::env::var("FFN_SPARSE_OFF").is_err();
        let sparse_vt = if sparse_ok {
            self.layers[i].ffn_value_tiled
        } else {
            None
        };
        if let Some(vt) = sparse_vt {
            self.backend.ffn_value_sparse_add(
                self.layers[i].ffn_value_w16,
                vt,
                self.bufs.r2,
                self.bufs.x,
                c,
                fh,
            )?;
        } else if let Some(a8) = &self.layers[i].ffn_value_a8 {
            self.backend
                .gemv_int8_add(a8, self.bufs.r2, self.bufs.x, c, fh, 1)?;
        } else {
            self.backend.gemv_f16_add(
                *self.layers[i]
                    .ffn_value_w16
                    .as_ref()
                    .ok_or("ffn_value_w16 missing")?,
                self.bufs.r2,
                self.bufs.x,
                c,
                fh,
                1,
            )?;
        }
        pf_phase!("ffnv_sparse");
        let _ = t0; // 引用最后一次赋值，避免 unused_assignments 警告

        Ok(())
    }

    /// 前向推理（sequence-parallel）：把整段 T 个 token 一次贯穿各层，返回最后 token 的 logits [vocab]。
    /// 对标 albatross forward_seq：线性投影用批量 GEMM（token 并行），WKV 顺序更新用单次 launch 的
    /// dplr_seq（内部循环 T），最大程度减少逐 token 的 dispatch 开销。
    pub fn forward_seq(&mut self, tokens: &[u32]) -> R<Vec<f32>> {
        let mut state = self.state.take().unwrap();
        let out = self.forward_seq_with_state(&mut state, tokens);
        self.state = Some(state);
        out
    }

    /// 前向推理（sequence-parallel，web-rwkv 风格）：接受外部 `State`。
    /// 把整段 T 个 token 一次贯穿各层，返回最后 token 的 logits [vocab]。
    pub fn forward_seq_with_state(&mut self, state: &mut State, tokens: &[u32]) -> R<Vec<f32>> {
        let t = tokens.len();
        assert!(t >= 1, "forward_seq requires at least 1 token");
        let (c, n, vocab) = (self.config.n_embd, self.config.head_size, self.config.vocab);

        // 按需创建/复用序列并行缓冲（T 变化时重建）
        if self.seq_bufs.as_ref().is_none_or(|sb| sb.t != t) {
            // T 变化 → 新缓冲地址 + 新 spec，旧 kernel 缓存失效，清空避免耗尽 descriptor pool
            self.backend.clear_cache();
            // 释放旧缓冲的设备内存（防泄漏：注册表残留分配）
            if let Some(mut old) = self.seq_bufs.take() {
                old.free(self.backend.as_mut());
            }
            self.seq_bufs = Some(SeqBuffers::new(
                self.backend.as_mut(),
                t,
                c,
                vocab,
                &self.config,
            )?);
        }
        // 临时取出 seq_bufs，避免 forward_seq_layer 的 &mut self 与 &mut self.seq_bufs 冲突
        let mut seq_bufs = self.seq_bufs.take().unwrap();
        let sb = &mut seq_bufs;

        // 收集整段嵌入行 [t, c]（用 CPU 缓存，避免每次下载 671MB 的 emb_ln 表）
        let emb_ln = &self.emb_ln_cpu;
        let mut x_data = vec![0.0; t * c];
        for (ti, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            x_data[ti * c..(ti + 1) * c].copy_from_slice(&emb_ln[tok * c..(tok + 1) * c]);
        }
        self.backend.upload(sb.x, &x_data)?;

        let seq_pf = std::env::var("SEQ_PROFILE").is_ok();
        let mut spf_t0 = std::time::Instant::now();
        // CUDA prefill graph：整段各层 + head 一次捕获为 graph，之后同 T 直接重放，
        // 消除跨层 launch 开销。Vulkan 为 no-op（逐一提交）。诊断路径（GEMM_DIAG/PROFILE）
        // 含下载/逐段 submit，与 graph 捕获互斥，故仅在普通路径启用。
        let prefill_graph = std::env::var("PREFILL_GRAPH").as_deref() == Ok("1")
            && std::env::var("GEMM_DIAG").is_err()
            && std::env::var("PROFILE").is_err();
        if prefill_graph && self.backend.prefill_graph_valid(t)? {
            self.backend.prefill_graph_replay()?;
        } else {
            if prefill_graph {
                self.backend.begin_prefill_capture(t)?;
            }
            // 单批处理全部层：任何层都走 int8 GEMM（零 fp16 副本），无需逐层 begin/end_batch 隔离。
            self.backend.begin_batch()?;
            for i in 0..self.config.n_layer {
                if std::env::var("GEMM_DIAG").is_ok() {
                    log::info!("[FS] layer {i} start");
                }
                self.forward_seq_layer(i, t, c, n, sb, state)?;
                if std::env::var("GEMM_DIAG").is_ok() {
                    log::info!("[FS] layer {i} done");
                }
                if seq_pf {
                    log::info!("[SP] layer {i}: {:.4}s", spf_t0.elapsed().as_secs_f64());
                    spf_t0 = std::time::Instant::now();
                }
            }
            self.backend.end_batch()?;

            // ln_out + head（只算最后 token，避免 [T, vocab] 全量 GEMM 与 67MB 下载）
            self.backend.begin_batch()?;
            if std::env::var("HEAD_DIAG").is_ok() {
                log::info!("[HEAD] norm start");
            }
            // x_norm = layer_norm(x, ln_out_w, ln_out_b) over T 个 token
            self.backend.norm(
                sb.x,
                self.ln_out_w,
                self.ln_out_b,
                sb.x_norm,
                c,
                1,
                LN_EPS,
                t,
            )?;
            if std::env::var("HEAD_DIAG").is_ok() {
                log::info!("[HEAD] copy_token start");
            }
            // head_in = x_norm[T-1]（末 token 行）
            self.backend
                .copy_token(sb.x_norm, sb.head_in, c, c, t - 1)?;
            if std::env::var("HEAD_DIAG").is_ok() {
                log::info!("[HEAD] head_gemv start");
            }
            // logits = head_in @ head.T，输出 [vocab]（int8 量化 head 省 4× 带宽）
            self.head_gemv(sb.head_in, sb.logits, vocab, c)?;
            if std::env::var("HEAD_DIAG").is_ok() {
                log::info!("[HEAD] head_gemv done");
            }

            self.backend.end_batch()?;
            if prefill_graph {
                self.backend.end_prefill_capture()?;
                // 捕获不执行 kernel，立即重放一次以产生本次输出（同 T 后续调用直接走重放）。
                self.backend.prefill_graph_replay()?;
            }
        }

        // 诊断：比较最后层 output GEMM vs gemv 参考
        if std::env::var("GEMM_DIAG").is_ok() {
            let go = self.backend.download(sb.y_out)?;
            let gr = self.backend.download(sb.diag_out_ref)?;
            let mut md = 0.0f32;
            for (a, b) in go.iter().zip(&gr) {
                md = md.max((a - b).abs());
            }
            log::info!("[FS] last-layer output GEMM vs gemv max_abs_diff: {md:.6}");
            // 诊断：to_f16 是否正确（y_g16 vs fp16(y_g)）
            let yg = self.backend.download(sb.y_g)?;
            let yg16 = self.backend.download(sb.y_g16)?;
            let mut md16 = 0.0f32;
            for i in 0..c {
                let want = half::f16::from_f32(yg[i]).to_f32();
                md16 = md16.max((yg16[i] - want).abs());
            }
            // 检查 y_g 中是否有非零fp16源（排除整行0的padding）
            let nonzero = yg.iter().take(c).filter(|v| v.abs() > 1e-6).count();
            log::info!(
                "[FS] to_f16 y_g16 vs fp16(y_g) max_abs_diff: {md16:.6} (nonzero yg={nonzero}/{c})"
            );
            log::info!("[FS] y_g[0..4] = {:?}", &yg[..4]);
            log::info!("[FS] y_g16[0..4] = {:?}", &yg16[..4]);
        }

        // 返回最后 token 的 logits（先下载，再归还 seq_bufs）
        let last_logits = self.backend.download(sb.logits)?;

        // 归还 seq_bufs
        self.seq_bufs = Some(seq_bufs);

        Ok(last_logits)
    }

    /// 序列 prefill 后返回最后一层 FFN 输出（残差流 sb.x）在 T 维的 mean pooling。
    ///
    /// 与 `forward_seq_with_state` 共享层循环语义，但跳过 ln_out/head
    /// （省去 [vocab] GEMM 与全量 logits 下载），仅下载 [T, C] 后在 CPU 端
    /// 对 T 维求均值 → [C]。供智能路由的 state-embedding 分类器提取特征。
    pub fn forward_seq_mean_hidden(&mut self, state: &mut State, tokens: &[u32]) -> R<Vec<f32>> {
        let t = tokens.len();
        assert!(t >= 1, "forward_seq_mean_hidden requires at least 1 token");
        let (c, n, vocab) = (self.config.n_embd, self.config.head_size, self.config.vocab);

        // 按需创建/复用序列并行缓冲（与 forward_seq_with_state 一致）
        if self.seq_bufs.as_ref().is_none_or(|sb| sb.t != t) {
            self.backend.clear_cache();
            // 释放旧缓冲的设备内存（防泄漏）
            if let Some(mut old) = self.seq_bufs.take() {
                old.free(self.backend.as_mut());
            }
            self.seq_bufs = Some(SeqBuffers::new(
                self.backend.as_mut(),
                t,
                c,
                vocab,
                &self.config,
            )?);
        }
        let mut seq_bufs = self.seq_bufs.take().unwrap();
        let sb = &mut seq_bufs;

        // 收集整段嵌入行 [t, c]（CPU 缓存的 emb_ln 表）
        let emb_ln = &self.emb_ln_cpu;
        let mut x_data = vec![0.0; t * c];
        for (ti, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            x_data[ti * c..(ti + 1) * c].copy_from_slice(&emb_ln[tok * c..(tok + 1) * c]);
        }
        self.backend.upload(sb.x, &x_data)?;

        // 单批处理全部层：prefill 后 sb.x 即最后一层 FFN 输出（残差后）
        self.backend.begin_batch()?;
        for i in 0..self.config.n_layer {
            self.forward_seq_layer(i, t, c, n, sb, state)?;
        }
        self.backend.end_batch()?;

        // 下载 [t, c] 后对 T 维求均值 → [c]
        let x_all = self.backend.download(sb.x)?;
        let mut mean_hidden = vec![0.0f32; c];
        for ti in 0..t {
            let row = &x_all[ti * c..(ti + 1) * c];
            for (acc, &v) in mean_hidden.iter_mut().zip(row) {
                *acc += v;
            }
        }
        let inv_t = 1.0 / t as f32;
        for v in mean_hidden.iter_mut() {
            *v *= inv_t;
        }

        // 归还 seq_bufs
        self.seq_bufs = Some(seq_bufs);

        Ok(mean_hidden)
    }

    /// 诊断：逐层快照 x（前 D 维），用于定位 seq 路径两次运行的首发发散层。
    /// 返回 Vec<Vec<f32>>，第 i 项为第 i+1 层输出 x 的快照（i==0 为 layer 0 之后）。
    pub fn snapshot_seq_layers(&mut self, tokens: &[u32], d: usize) -> R<Vec<Vec<f32>>> {
        let t = tokens.len();
        let (c, n, vocab) = (self.config.n_embd, self.config.head_size, self.config.vocab);
        self.reset_state()?;
        self.backend.clear_cache();
        if self.seq_bufs.as_ref().is_none_or(|sb| sb.t != t) {
            self.backend.clear_cache();
            // 释放旧缓冲的设备内存（防泄漏）
            if let Some(mut old) = self.seq_bufs.take() {
                old.free(self.backend.as_mut());
            }
            self.seq_bufs = Some(SeqBuffers::new(
                self.backend.as_mut(),
                t,
                c,
                vocab,
                &self.config,
            )?);
        }
        let mut seq_bufs = self.seq_bufs.take().unwrap();
        let sb = &mut seq_bufs;
        let emb_ln = &self.emb_ln_cpu;
        let mut x_data = vec![0.0; t * c];
        for (ti, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            x_data[ti * c..(ti + 1) * c].copy_from_slice(&emb_ln[tok * c..(tok + 1) * c]);
        }
        self.backend.upload(sb.x, &x_data)?;
        let mut snaps = Vec::new();
        // 取出内部状态，逐层快照（与 forward_seq 一致：int8 GEMM 零 fp16 副本）
        let mut state = self.state.take().unwrap();
        for i in 0..self.config.n_layer {
            self.backend.begin_batch()?;
            self.forward_seq_layer(i, t, c, n, sb, &mut state)?;
            self.backend.end_batch()?;
            let xd = self.backend.download(sb.x)?;
            snaps.push(xd[..d].to_vec());
        }
        self.state = Some(state);
        self.seq_bufs = Some(seq_bufs);
        Ok(snaps)
    }

    /// 诊断：逐 token 跑 forward，逐层快照 x（取前 d 个元素）。用于和 seq 逐层对比定位 bug。
    pub fn snapshot_layers(&mut self, tokens: &[u32], d: usize) -> R<Vec<Vec<f32>>> {
        let (c, h, n, _vocab) = (
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
            self.config.vocab,
        );
        self.reset_state()?;
        // 预填充输入 token（仅取第 0 个 token，因为 snapshot_layers 是逐 token）
        let mut snaps = Vec::new();
        // 初始 emb_ln 输入
        let emb_ln = &self.emb_ln_cpu;
        let token = tokens[0] as usize;
        let mut x_host = emb_ln[token * c..(token + 1) * c].to_vec();
        self.backend.upload(self.bufs.x, &x_host)?;
        // 取出内部状态，与 forward 一致：逐层前向并快照
        let mut state = self.state.take().unwrap();
        for i in 0..self.config.n_layer {
            state.v_first_set = false;
            self.backend.begin_batch()?;
            self.forward_layer(i, c, h, n, &mut state)?;
            self.backend.end_batch()?;
            x_host = self.backend.download(self.bufs.x)?;
            snaps.push(x_host[..d].to_vec());
        }
        self.state = Some(state);
        Ok(snaps)
    }

    /// 诊断：下载 layer idx 的 tmix_rnn 状态的前 n 个元素（用于对比 seq 与 tok 路径 dplr 状态差异）。
    pub fn download_state_rnn(&mut self, idx: usize, n: usize) -> R<Vec<f32>> {
        let s = self
            .backend
            .download(self.state.as_ref().unwrap().layers[idx].tmix_rnn)?;
        Ok(s[..n.min(s.len())].to_vec())
    }

    /// 诊断：下载 layer idx 的 tmix_x（time-mix 插值状态）前 n 个元素。
    pub fn download_state_tmix_x(&mut self, idx: usize, n: usize) -> R<Vec<f32>> {
        let s = self
            .backend
            .download(self.state.as_ref().unwrap().layers[idx].tmix_x)?;
        Ok(s[..n.min(s.len())].to_vec())
    }

    /// 诊断：下载 layer idx 的 cmix_x（channel-mix 插值状态）前 n 个元素。
    pub fn download_state_cmix_x(&mut self, idx: usize, n: usize) -> R<Vec<f32>> {
        let s = self
            .backend
            .download(self.state.as_ref().unwrap().layers[idx].cmix_x)?;
        Ok(s[..n.min(s.len())].to_vec())
    }

    pub fn layers_len(&self) -> usize {
        self.layers.len()
    }

    /// 词表大小（采样/解码用）。
    pub fn vocab_len(&self) -> usize {
        self.config.vocab
    }

    /// 模型维度 n_embd（C）。
    pub fn n_embd(&self) -> usize {
        self.config.n_embd
    }

    /// 诊断：逐 token 路径，跑 up_to_layer 层之前的 x_norm（ln1）和 x（输入），下载前 n 个元素。
    pub fn diag_tok_x_and_xnorm_before_layer(
        &mut self,
        tokens: &[u32],
        up_to_layer: usize,
        n: usize,
    ) -> R<(Vec<f32>, Vec<f32>)> {
        let (c, h, n_head) = (
            self.config.n_embd,
            self.config.n_head,
            self.config.head_size,
        );
        self.reset_state()?;
        let emb_ln = &self.emb_ln_cpu;
        let token = tokens[0] as usize;
        let x_host = emb_ln[token * c..(token + 1) * c].to_vec();
        self.backend.upload(self.bufs.x, &x_host)?;
        let mut state = self.state.take().unwrap();
        for i in 0..up_to_layer {
            state.v_first_set = false;
            self.backend.begin_batch()?;
            self.forward_layer(i, c, h, n_head, &mut state)?;
            self.backend.end_batch()?;
        }
        self.state = Some(state);
        // 下载输入 x（layer up_to_layer 的输入）
        let x_inp = self.backend.download(self.bufs.x)?;
        // 计算 ln1
        self.backend.begin_batch()?;
        let li = up_to_layer.min(self.layers.len() - 1);
        let l = &self.layers[li];
        self.backend.norm(
            self.bufs.x,
            l.ln1_w,
            l.ln1_b,
            self.bufs.ln1,
            c,
            1,
            LN_EPS,
            1,
        )?;
        self.backend.end_batch()?;
        let ln = self.backend.download(self.bufs.ln1)?;
        let nn = n.min(ln.len()).min(x_inp.len());
        Ok((x_inp[..nn].to_vec(), ln[..nn].to_vec()))
    }

    /// 诊断：seq 路径，跑 up_to_layer 层之后的 x 和 x_norm（ln1）第 0 个 token 的前 n 个元素。
    pub fn diag_seq_x_and_xnorm_before_layer(
        &mut self,
        tokens: &[u32],
        up_to_layer: usize,
        n: usize,
    ) -> R<(Vec<f32>, Vec<f32>)> {
        let t = tokens.len();
        let (c, n_head, vocab) = (self.config.n_embd, self.config.head_size, self.config.vocab);
        self.backend.clear_cache();
        if self.seq_bufs.as_ref().is_none_or(|sb| sb.t != t) {
            self.backend.clear_cache();
            self.seq_bufs = Some(SeqBuffers::new(
                self.backend.as_mut(),
                t,
                c,
                vocab,
                &self.config,
            )?);
        }
        let mut seq_bufs = self.seq_bufs.take().unwrap();
        let sb = &mut seq_bufs;
        let emb_ln = &self.emb_ln_cpu;
        let mut x_data = vec![0.0; t * c];
        for (ti, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            x_data[ti * c..(ti + 1) * c].copy_from_slice(&emb_ln[tok * c..(tok + 1) * c]);
        }
        self.backend.upload(sb.x, &x_data)?;
        // 取出内部状态，跑 [0, up_to_layer) 层
        let mut state = self.state.take().unwrap();
        for i in 0..up_to_layer {
            self.backend.begin_batch()?;
            self.forward_seq_layer(i, t, c, n_head, sb, &mut state)?;
            self.backend.end_batch()?;
        }
        self.state = Some(state);
        // 下载输入 x（第 0 token：sb.x[0..c]，因为 mk_pad(c) 布局 [M_PAD, C] 第 0 行偏移 = 0 * C = 0）
        let x_all = self.backend.download(sb.x)?;
        let x_inp = x_all[..c.min(x_all.len())].to_vec();
        // 跑 norm
        self.backend.begin_batch()?;
        let li = up_to_layer.min(self.layers.len() - 1);
        let l = &self.layers[li];
        self.backend
            .norm(sb.x, l.ln1_w, l.ln1_b, sb.ln1, c, 1, LN_EPS, t)?;
        self.backend.end_batch()?;
        let xn = self.backend.download(sb.ln1)?;
        let nn = n.min(xn.len()).min(x_inp.len());
        self.seq_bufs = Some(seq_bufs);
        Ok((x_inp[..nn].to_vec(), xn[..nn].to_vec()))
    }

    /// 诊断：dump 指定层的中间张量（需在批处理上下文中，吞吐低，仅调试用）。
    fn dump_tensors(
        backend: &mut dyn ComputeBackend,
        tag: &str,
        i: usize,
        items: &[(&str, TensorId)],
    ) -> R<()> {
        let want = std::env::var("DUMP_LAYER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        if want == Some(i) {
            backend.end_batch()?;
            for (name, t) in items {
                let d = backend.download(*t)?;
                let n = d.len().min(8);
                log::info!("[DBG] {tag} layer {i} {name} [0..{n}] = {:?}", &d[..n]);
            }
            backend.begin_batch()?;
        }
        Ok(())
    }

    /// 诊断：用 CPU fp16 参考验证 GPU GEMM 的 r/k/v 输出（GEMM_DIAG_VERIFY=层号时触发）。
    /// 下载 xk16/xv16 与 key_w16/value_w16，计算 k[0]/v[0] 的 fp16 精确参考，与 GPU 输出对比。
    fn verify_gemm_rkv(
        backend: &mut dyn ComputeBackend,
        layer: &GpuLayer,
        i: usize,
        sb: &SeqBuffers,
        c: usize,
    ) -> R<()> {
        let want = std::env::var("GEMM_DIAG_VERIFY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        if want != Some(i) {
            return Ok(());
        }
        backend.end_batch()?;
        // int8 模型无常驻 w16（方案A 下参考为 w_scratch 反量化结果，此处直接跳过）
        let (Some(key_w16), Some(value_w16)) = (&layer.key_w16, &layer.value_w16) else {
            log::info!("[GEMMV] layer {i} skipped: int8 模型无常驻 w16 参考");
            backend.begin_batch()?;
            return Ok(());
        };
        let xk16 = backend.download(sb.xk16)?;
        let xv16 = backend.download(sb.xv16)?;
        let xk = backend.download(sb.xk)?;
        let kw16 = backend.download(*key_w16)?;
        let vw16 = backend.download(*value_w16)?;
        let gk = backend.download(sb.k)?;
        let gv = backend.download(sb.v)?;
        backend.begin_batch()?;

        // 校验 xk16 == fp16(xk)，揭穿 GEMM 输入是否被竞态污染
        for e in 0..8 {
            let ref16 = half::f16::from_f32(xk[e]).to_f32();
            let seq16 = xk16[e];
            log::info!(
                "[GEN16] layer {i} xk[{e}] fp32={:.6} fp16_ref={:.6} xk16={:.6} diff={:.6}",
                xk[e],
                ref16,
                seq16,
                (seq16 - ref16).abs()
            );
        }

        // k[0] = sum_z xk16[0,z] * key_w16[0,z]；v[0] = sum_z xv16[0,z] * value_w16[0,z]
        for j in 0..4 {
            let mut kref = 0.0f32;
            let mut vref = 0.0f32;
            for z in 0..c {
                kref += xk16[z] * kw16[j * c + z];
                vref += xv16[z] * vw16[j * c + z];
            }
            log::info!(
                "[GEMMV] layer {i} j={j} k: gpu={:.6} cpu_fp16={:.6} diff={:.6} | v: gpu={:.6} cpu_fp16={:.6} diff={:.6}",
                gk[j],
                kref,
                (gk[j] - kref).abs(),
                gv[j],
                vref,
                (gv[j] - vref).abs()
            );
        }
        Ok(())
    }

    /// 单层前向（sequence-parallel）：time mixing + channel mixing，对整段 [T, C] 并行。
    #[allow(clippy::too_many_arguments, unused_assignments)]
    fn forward_seq_layer(
        &mut self,
        i: usize,
        t: usize,
        c: usize,
        n: usize,
        sb: &mut SeqBuffers,
        state: &mut State,
    ) -> R<()> {
        let fh = self.config.ffn_hidden;
        let (wwp, aap, vvp, ggp) = (
            self.config.w_mid_pad,
            self.config.a_mid_pad,
            self.config.v_mid_pad,
            self.config.g_mid_pad,
        );
        let layer = &self.layers[i];
        // 阶段剖析（PROFILE=1 时每个阶段独立 submit 并计时）
        let pf = std::env::var("PROFILE").is_ok();
        let mut t0 = std::time::Instant::now();
        macro_rules! pf_phase {
            ($name:expr) => {
                if pf {
                    self.backend.end_batch()?;
                    log::info!(
                        "[P] t={t} layer {i} {}: {:.4}s",
                        $name,
                        t0.elapsed().as_secs_f64()
                    );
                    self.backend.begin_batch()?;
                    t0 = std::time::Instant::now();
                }
            };
        }

        // ===== Time Mixing =====
        // ln1 = layer_norm(x, ln1_w, ln1_b) over T 个 token
        self.backend
            .norm(sb.x, layer.ln1_w, layer.ln1_b, sb.ln1, c, 1, LN_EPS, t)?;

        // token shift + time-mix：xr/xw/xk/xv/xa/xg（t=0 用旧 state，t>0 用前一 token）
        self.backend
            .seq_shift(sb.ln1, state.layers[i].tmix_x, layer.x_r, sb.xr, c, t, c, c)?;
        self.backend
            .seq_shift(sb.ln1, state.layers[i].tmix_x, layer.x_w, sb.xw, c, t, c, c)?;
        self.backend
            .seq_shift(sb.ln1, state.layers[i].tmix_x, layer.x_k, sb.xk, c, t, c, c)?;
        self.backend
            .seq_shift(sb.ln1, state.layers[i].tmix_x, layer.x_v, sb.xv, c, t, c, c)?;
        self.backend
            .seq_shift(sb.ln1, state.layers[i].tmix_x, layer.x_a, sb.xa, c, t, c, c)?;
        self.backend
            .seq_shift(sb.ln1, state.layers[i].tmix_x, layer.x_g, sb.xg, c, t, c, c)?;

        // state[i].tmix_x = ln1[T-1]（须在 seq_shift 之后，避免覆盖 t=0 读取的旧 state）
        self.backend
            .copy_token(sb.ln1, state.layers[i].tmix_x, c, c, t - 1)?;

        // r/k/v = token 并行 GEMM。
        // 方案A（默认）：int8 权重 dequant 到共享 w_scratch，走 fp16 tensor-core GEMM；
        // 非 int8 模型：直接用常驻 fp16 w16。
        let m_pad = sb.m_pad;
        if std::env::var("GEMM_DIAG").is_ok() {
            log::info!("[FS] layer {i} gemm rkv start");
        }
        if let (Some(ra8), Some(ka8), Some(va8)) =
            (&layer.receptance_a8, &layer.key_a8, &layer.value_a8)
        {
            self.backend.to_f16_triple(
                sb.xr, sb.xk, sb.xv, sb.xr16, sb.xk16, sb.xv16, c, t, m_pad, c, c,
            )?;
            self.backend.dequant_int8_to_f16(ra8, sb.w_scratch, c, c)?;
            self.backend
                .gemm(sb.xr16, sb.w_scratch, sb.r, m_pad, c, c)?;
            self.backend.dequant_int8_to_f16(ka8, sb.w_scratch, c, c)?;
            self.backend
                .gemm(sb.xk16, sb.w_scratch, sb.k, m_pad, c, c)?;
            self.backend.dequant_int8_to_f16(va8, sb.w_scratch, c, c)?;
            self.backend
                .gemm(sb.xv16, sb.w_scratch, sb.v, m_pad, c, c)?;
        } else {
            self.backend.to_f16_triple(
                sb.xr, sb.xk, sb.xv, sb.xr16, sb.xk16, sb.xv16, c, t, m_pad, c, c,
            )?;
            let r16 = layer
                .receptance_w16
                .as_ref()
                .ok_or("receptance_w16 missing")?;
            let k16 = layer.key_w16.as_ref().ok_or("key_w16 missing")?;
            let v16 = layer.value_w16.as_ref().ok_or("value_w16 missing")?;
            self.backend.gemm(sb.xr16, *r16, sb.r, m_pad, c, c)?;
            self.backend.gemm(sb.xk16, *k16, sb.k, m_pad, c, c)?;
            self.backend.gemm(sb.xv16, *v16, sb.v, m_pad, c, c)?;
        }
        if std::env::var("GEMM_DIAG").is_ok() {
            log::info!("[FS] layer {i} gemm rkv done");
        }
        pf_phase!("shift+rkv_gemm");
        Self::dump_tensors(
            self.backend.as_mut(),
            "seq",
            i,
            &[
                ("xr", sb.xr),
                ("xk", sb.xk),
                ("xv", sb.xv),
                ("r", sb.r),
                ("k", sb.k),
                ("v", sb.v),
            ],
        )?;
        Self::verify_gemm_rkv(self.backend.as_mut(), layer, i, sb, c)?;

        // 低秩 w/a/g 第一级投影输入转 fp16（xw/xa/xg → [M_PAD, C]）
        self.backend.to_f16_triple(
            sb.xw, sb.xa, sb.xg, sb.xw16, sb.xa16, sb.xg16, c, t, m_pad, c, c,
        )?;

        // v_first 逻辑：layer 0 存 v_first = v；layer>0 交叉混合 v
        if i == 0 {
            self.backend.copy_device(sb.v, sb.v_first)?;
        } else {
            // v_mid = tensor-core GEMM(xv @ v1)：[M_PAD, vm_pad]
            self.backend
                .gemm(sb.xv16, layer.v1_16, sb.v_mid, m_pad, vvp, c)?;
            // v_mid → fp16（第二级投影输入）
            self.backend
                .to_f16(sb.v_mid, sb.v_mid16, vvp, t, m_pad, vvp, vvp)?;
            // v_full = gemm_bias(v_mid @ v2 + v0)：[M_PAD, C]
            self.backend
                .gemm_bias(sb.v_mid16, layer.v2_16, layer.v0, sb.v_full, m_pad, c, vvp)?;
            // gate = sigmoid(v_full)
            self.backend.elementwise_sigmoid(sb.v_full, sb.gate, c, t)?;
            // v = v + gate*(v_first - v)（原地，含 t=0）
            self.backend
                .v_first_lerp(sb.v, sb.gate, sb.v_first, c, t, c)?;
        }

        // w = exp(-sigmoid(w0 + tanh(xw@w1)@w2)/sqrt(e))
        // w_mid = tanh(xw @ w1)：[M_PAD, wm_pad]
        self.backend
            .gemm_tanh(sb.xw16, layer.w1_16, sb.w_mid, m_pad, wwp, c)?;
        self.backend
            .to_f16(sb.w_mid, sb.w_mid16, wwp, t, m_pad, wwp, wwp)?;
        // w_full = gemm_bias(w_mid @ w2 + w0)：[M_PAD, C]
        self.backend
            .gemm_bias(sb.w_mid16, layer.w2_16, layer.w0, sb.w_full, m_pad, c, wwp)?;
        self.backend
            .elementwise_sigmoid(sb.w_full, sb.w_sig, c, t)?;
        self.backend
            .elementwise_scale_exp(sb.w_sig, self.scale_w, sb.w, c, t)?;

        // a = sigmoid(a0 + xa@a1@a2)
        // a_mid = xa @ a1：[M_PAD, am_pad]
        self.backend
            .gemm(sb.xa16, layer.a1_16, sb.a_mid, m_pad, aap, c)?;
        self.backend
            .to_f16(sb.a_mid, sb.a_mid16, aap, t, m_pad, aap, aap)?;
        // a_full = gemm_bias(a_mid @ a2 + a0)：[M_PAD, C]
        self.backend
            .gemm_bias(sb.a_mid16, layer.a2_16, layer.a0, sb.a_full, m_pad, c, aap)?;
        self.backend.elementwise_sigmoid(sb.a_full, sb.a, c, t)?;

        // 融合 k/k_a：k_mod、kk_l2、b_vec 一次 launch
        self.backend.fuse_ka(
            sb.k,
            layer.k_k,
            sb.a,
            layer.k_a,
            sb.k_mod,
            sb.kk_l2,
            sb.b_vec,
            self.config.n_head,
            n,
            t,
        )?;
        pf_phase!("lowrank+fuse_ka");

        // DPLR：单次 launch 处理整段 T（内部循环），S 跨 token 传递
        self.backend.dplr_seq(
            state.layers[i].tmix_rnn,
            sb.r,
            sb.w,
            sb.k_mod,
            sb.v,
            sb.kk_l2,
            sb.b_vec,
            sb.y,
            self.config.n_head,
            n,
            t,
            c,
        )?;
        pf_phase!("lowrank+dplr_seq");
        Self::dump_tensors(
            self.backend.as_mut(),
            "seq",
            i,
            &[("y", sb.y), ("y_norm", sb.y_norm)],
        )?;

        // y_norm = group_norm(y, ln_x_w, ln_x_b)
        self.backend.norm(
            sb.y,
            layer.ln_x_w,
            layer.ln_x_b,
            sb.y_norm,
            n,
            self.config.n_head,
            GN_EPS,
            t,
        )?;

        // extra: y_norm += sum(r*k_mod*r_k, head) * v
        self.backend.sum_rk_rk(
            sb.r,
            sb.k_mod,
            layer.r_k,
            sb.v,
            sb.y_norm,
            self.config.n_head,
            n,
            t,
        )?;

        // g = sigmoid(xg@g1)@g2（tensor-core GEMM）
        // g_mid = xg @ g1：[M_PAD, gm_pad]
        self.backend
            .gemm(sb.xg16, layer.g1_16, sb.g_mid, m_pad, ggp, c)?;
        // sigmoid 原地 + 转 fp16（第二级投影输入）
        self.backend.elementwise_sigmoid_inplace(sb.g_mid, ggp, t)?;
        self.backend
            .to_f16(sb.g_mid, sb.g_mid16, ggp, t, m_pad, ggp, ggp)?;
        // g = g_mid @ g2：[M_PAD, C]
        self.backend
            .gemm(sb.g_mid16, layer.g2_16, sb.g, m_pad, c, ggp)?;
        Self::dump_tensors(
            self.backend.as_mut(),
            "seq",
            i,
            &[("w", sb.w), ("a", sb.a), ("v", sb.v), ("g", sb.g)],
        )?;

        // y_g = y_norm * g
        self.backend
            .elementwise_mul(sb.y_norm, sb.g, sb.y_g, c, t)?;

        // y_out = GEMM(output_w, y_g) + x（融合残差相加）；int8 默认走方案A（dequant→scratch→TC GEMM）
        if let Some(a8) = &layer.att_output_a8 {
            self.backend.to_f16(sb.y_g, sb.y_g16, c, t, m_pad, c, c)?;
            self.backend.dequant_int8_to_f16(a8, sb.w_scratch, c, c)?;
            self.backend
                .gemm_add(sb.y_g16, sb.w_scratch, sb.x, sb.y_out, m_pad, c, c)?;
        } else {
            self.backend.to_f16(sb.y_g, sb.y_g16, c, t, m_pad, c, c)?;
            self.backend.gemm_add(
                sb.y_g16,
                *layer.output_w16.as_ref().ok_or("output_w16 missing")?,
                sb.x,
                sb.y_out,
                m_pad,
                c,
                c,
            )?;
        }
        // 交换 x 与 y_out：x 现在持有 block 输出残差，y_out 持有旧 x（下轮被覆盖）。
        std::mem::swap(&mut sb.x, &mut sb.y_out);
        if std::env::var("GEMM_DIAG").is_ok() {
            // gemv 参考输出（诊断用）
            let output_w = layer
                .output_w
                .as_ref()
                .ok_or("output_w missing when GEMM_DIAG")?;
            self.backend
                .gemv_seq(*output_w, sb.y_g, sb.diag_out_ref, c, c, c, c, t)?;
        }

        pf_phase!("y_norm+sum+g+output");
        Self::dump_tensors(
            self.backend.as_mut(),
            "seq",
            i,
            &[("g", sb.g), ("y_g", sb.y_g), ("x_out", sb.x)],
        )?;

        // ===== Channel Mixing =====
        // ln2 = layer_norm(x, ln2_w, ln2_b)
        self.backend
            .norm(sb.x, layer.ln2_w, layer.ln2_b, sb.ln2, c, 1, LN_EPS, t)?;

        // xb = token shift（t=0 用旧 cmix_x state，t>0 用前一轮 ln2）
        self.backend.seq_shift(
            sb.ln2,
            state.layers[i].cmix_x,
            layer.ffn_x_k,
            sb.xb,
            c,
            t,
            c,
            c,
        )?;
        // state[i].cmix_x = ln2[T-1]（须在 seq_shift 之后）
        self.backend
            .copy_token(sb.ln2, state.layers[i].cmix_x, c, c, t - 1)?;

        // FFN = token 并行 GEMM；int8 默认走方案A（dequant→scratch→TC GEMM）
        if std::env::var("GEMM_DIAG").is_ok() {
            log::info!("[FS] layer {i} ffn start");
        }
        if let Some(fka8) = &layer.ffn_key_a8 {
            self.backend.to_f16(sb.xb, sb.xb16, c, t, m_pad, c, c)?;
            self.backend
                .dequant_int8_to_f16(fka8, sb.w_scratch, fh, c)?;
            self.backend
                .gemm_relu2(sb.xb16, sb.w_scratch, sb.r2, m_pad, fh, c)?;
        } else {
            self.backend.to_f16(sb.xb, sb.xb16, c, t, m_pad, c, c)?;
            let fk16 = layer.ffn_key_w16.as_ref().ok_or("ffn_key_w16 missing")?;
            self.backend
                .gemm_relu2(sb.xb16, *fk16, sb.r2, m_pad, fh, c)?;
        }
        if let Some(fva8) = &layer.ffn_value_a8 {
            self.backend.to_f16(sb.r2, sb.r2_16, fh, t, m_pad, fh, fh)?;
            self.backend
                .dequant_int8_to_f16(fva8, sb.w_scratch, c, fh)?;
            self.backend
                .gemm_add(sb.r2_16, sb.w_scratch, sb.x, sb.v2, m_pad, c, fh)?;
        } else {
            self.backend.to_f16(sb.r2, sb.r2_16, fh, t, m_pad, fh, fh)?;
            let fv16 = layer
                .ffn_value_w16
                .as_ref()
                .ok_or("ffn_value_w16 missing")?;
            self.backend
                .gemm_add(sb.r2_16, *fv16, sb.x, sb.v2, m_pad, c, fh)?;
        }
        // 交换 x 与 v2：x 现在持有 block 最终输出残差，v2 持有旧 x（下轮被覆盖）。
        std::mem::swap(&mut sb.x, &mut sb.v2);
        if std::env::var("GEMM_DIAG").is_ok() {
            log::info!("[FS] layer {i} ffn done");
        }
        pf_phase!("channel_mix+ffn");

        Ok(())
    }
}

impl GpuLayer {
    /// 从 safetensors 加载单层权重并上传到 GPU
    fn load(
        backend: &mut dyn ComputeBackend,
        st: &safetensors::SafeTensors,
        idx: usize,
        c: usize,
        cfg: &ModelConfig,
    ) -> R<Self> {
        // 一维参数
        let load1 = |backend: &mut dyn ComputeBackend, name: &str| -> R<TensorId> {
            let key = format!("blocks.{idx}.{name}");
            let data = tensor_to_f32(&st.tensor(&key)?);
            let t = backend.create_tensor(data.len(), TensorDtype::F32)?;
            backend.upload(t, &data)?;
            Ok(t)
        };
        // 线性权重：只保留 output_w fp32 用于 GEMM_DIAG 诊断，其余已删 fp32 副本省显存
        let load_linear_diag = |backend: &mut dyn ComputeBackend, key: String| -> R<TensorId> {
            let data = tensor_to_f32(&st.tensor(&key)?);
            let t = backend.create_tensor(data.len(), TensorDtype::F32)?;
            backend.upload(t, &data)?;
            Ok(t)
        };
        // 线性权重 → fp16（tensor-core GEMM 用）
        let load_linear_f16 = |backend: &mut dyn ComputeBackend, key: String| -> R<TensorId> {
            let data = tensor_to_f32(&st.tensor(&key)?);
            let t = backend.create_tensor(data.len(), TensorDtype::F16)?;
            backend.upload(t, &data)?;
            Ok(t)
        };
        // ffn.value 权重 → 稀疏 FFN 平铺布局（对齐 Albatross cmix_sparse_down 的 value_weight_tiled）。
        // 布局：元素 (f, c) 映射到 [f_block][c_block][f_local][c_local]，
        //   f_block=f/FFN_TILE, f_local=f%FFN_TILE, c_block=c/FFN_SPMV_C_TILE, c_local=c%FFN_SPMV_C_TILE。
        // 该布局使稀疏内核按「固定 f、连续 c」读取，命中合并访问。
        // 三种量化（fp16/int8）均构建：CUDA 稀疏内核只读 r2 非零列（~4%），
        // 反量化出 fp16 平铺权重远优于稠密 int8 全量读取。
        let load_ffn_value_tiled = |backend: &mut dyn ComputeBackend,
                                    key: String,
                                    c: usize,
                                    fh: usize|
         -> R<(Option<TensorId>, Option<TensorId>)> {
            const FFN_SPMV_TILE: usize = 128;
            const FFN_SPMV_C_TILE: usize = 256;
            // 从任意量化形式取原始 f32 数据（fp16 / int8），返回 (data, shape=[M,K])。
            // int8 张量可能为 2D [M,K] 或 3D（group 打包，如 [M,K/128,128]），
            // 故 M=shape[0]，K 由展平字节按量化比特数反推（与 load_int8 一致）。
            let (data, shape) = if let Ok(idx_t) = st.tensor(&format!("{key}.int8_idx")) {
                let sz_t = st.tensor(&format!("{key}.int8_sz"))?;
                let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
                let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
                let m = idx_t.shape()[0];
                let k = idx_u32.len() * 4 / m;
                (dequant_int8(idx_u32, sz_u32, m, k), vec![m, k])
            } else {
                let t = st.tensor(&key)?;
                let shape = t.shape().to_vec();
                (tensor_to_f32(&t), shape)
            };
            // 定向到 [c, fh]（与 load_linear_f16 一致，解码 gemv 按 [C, fh] 使用）。
            let oriented = if shape[0] == c && shape[1] == fh {
                data
            } else if shape[0] == fh && shape[1] == c {
                transpose(&data, fh, c)
            } else {
                panic!("{key}: unexpected shape {shape:?}, want [{c},{fh}] or [{fh},{c}]")
            };
            let c_blocks = c / FFN_SPMV_C_TILE;
            assert_eq!(c % FFN_SPMV_C_TILE, 0, "{key}: C 需为 C_TILE 整数倍");
            assert_eq!(fh % FFN_SPMV_TILE, 0, "{key}: fh 需为 TILE 整数倍");
            // ★ Phase 2-3：稠密 fp16 路径（`FFN_VALUE_GEMM`，默认开）**只需要 oriented 本身**
            // （`[C, fh]` 行主序 = 稠密 GEMM 的 B 操作数），于是：
            // ① 省掉整块 tiled 构建（加载期平铺 + 1.68GB 显存）；
            // ② 与 `ffn_value_tiled` **互斥**（同一份数据、同一体积）⇒ 显存净变化为 0。
            let dense16 = {
                let tg = backend.create_tensor(oriented.len(), TensorDtype::F16)?;
                backend.upload(tg, &oriented)?;
                Some(tg)
            };
            if std::env::var("FFN_VALUE_GEMM")
                .map(|v| v != "0")
                .unwrap_or(true)
            {
                return Ok((None, dense16));
            }
            let mut tiled = vec![0.0f32; fh * c];
            // 加载关键路径：按 f_block 分块多线程平铺（各块写不相交区域，结果与串行一致）
            {
                let f_block_len = FFN_SPMV_TILE * c; // 每 f_block 的 tiled 区域长
                let f_blocks = fh / FFN_SPMV_TILE;
                let threads = std::thread::available_parallelism()
                    .map(|n| n.get())
                    .unwrap_or(4)
                    .min(f_blocks.max(1));
                if threads > 1 {
                    let blocks_per = f_blocks.div_ceil(threads);
                    std::thread::scope(|scope| {
                        for (chunk_idx, fbs) in
                            tiled.chunks_mut(blocks_per * f_block_len).enumerate()
                        {
                            let fb0 = chunk_idx * blocks_per;
                            let oriented = &oriented;
                            scope.spawn(move || {
                                for (fb_local, fbs) in fbs.chunks_mut(f_block_len).enumerate() {
                                    let f_block = fb0 + fb_local;
                                    for (fl, f) in (f_block * FFN_SPMV_TILE
                                        ..(f_block + 1) * FFN_SPMV_TILE)
                                        .enumerate()
                                    {
                                        for cc in 0..c {
                                            let c_block = cc / FFN_SPMV_C_TILE;
                                            let c_local = cc % FFN_SPMV_C_TILE;
                                            fbs[(c_block * FFN_SPMV_TILE + fl) * FFN_SPMV_C_TILE
                                                + c_local] = oriented[cc * fh + f];
                                        }
                                    }
                                }
                            });
                        }
                    });
                } else {
                    for f in 0..fh {
                        let f_block = f / FFN_SPMV_TILE;
                        let f_local = f % FFN_SPMV_TILE;
                        for cc in 0..c {
                            let c_block = cc / FFN_SPMV_C_TILE;
                            let c_local = cc % FFN_SPMV_C_TILE;
                            tiled[((f_block * c_blocks + c_block) * FFN_SPMV_TILE)
                                * FFN_SPMV_C_TILE
                                + f_local * FFN_SPMV_C_TILE
                                + c_local] = oriented[cc * fh + f];
                        }
                    }
                }
            }
            let tg = backend.create_tensor(tiled.len(), TensorDtype::F16)?;
            backend.upload(tg, &tiled)?;
            Ok((Some(tg), dense16))
        };
        // 低秩权重：gemv 需要 [out, in] 行主序。
        // 各模型原始布局不同（g1h=[out,in]、g1d=[in,out]），按实际形状自适应转置。
        let load_lowrank = |backend: &mut dyn ComputeBackend,
                            name: &str,
                            out_dim: usize,
                            in_dim: usize|
         -> R<TensorId> {
            let key = format!("blocks.{idx}.att.{name}");
            let t = st.tensor(&key)?;
            let shape = t.shape();
            let data = tensor_to_f32(&t);
            let len = data.len();
            let oriented = if shape[0] == out_dim && shape[1] == in_dim {
                data // 已是 [out, in]
            } else if shape[0] == in_dim && shape[1] == out_dim {
                transpose(&data, in_dim, out_dim) // [in, out] → [out, in]
            } else {
                panic!(
                    "{key}: unexpected shape {shape:?}, want [{out_dim},{in_dim}] or [{in_dim},{out_dim}]"
                )
            };
            let t = backend.create_tensor(len, TensorDtype::F32)?;
            backend.upload(t, &oriented)?;
            Ok(t)
        };
        // 低秩权重 → fp16（tensor-core GEMM 用，补齐到 [pad_out, pad_in]，不满的行/列填 0）。
        // 第一级投影 pad=[mid_pad, C]（n=mid_pad, k=C），第二级投影 pad=[C, mid_pad]（n=C, k=mid_pad）。
        let load_lowrank_f16 = |backend: &mut dyn ComputeBackend,
                                name: &str,
                                out_dim: usize,
                                in_dim: usize,
                                pad_out: usize,
                                pad_in: usize|
         -> R<TensorId> {
            let key = format!("blocks.{idx}.att.{name}");
            let t = st.tensor(&key)?;
            let shape = t.shape();
            let data = tensor_to_f32(&t);
            let oriented = if shape[0] == out_dim && shape[1] == in_dim {
                data // 已是 [out, in]
            } else if shape[0] == in_dim && shape[1] == out_dim {
                transpose(&data, in_dim, out_dim) // [in, out] → [out, in]
            } else {
                panic!(
                    "{key}: unexpected shape {shape:?}, want [{out_dim},{in_dim}] or [{in_dim},{out_dim}]"
                )
            };
            let mut padded = vec![0.0f32; pad_out * pad_in];
            for r in 0..out_dim {
                for cc in 0..in_dim {
                    padded[r * pad_in + cc] = oriented[r * in_dim + cc];
                }
            }
            let t = backend.create_tensor(padded.len(), TensorDtype::F16)?;
            backend.upload(t, &padded)?;
            Ok(t)
        };

        let (wm, am, vm, gm) = (cfg.w_mid, cfg.a_mid, cfg.v_mid, cfg.g_mid);
        let (wwp, aap, vvp, ggp) = (cfg.w_mid_pad, cfg.a_mid_pad, cfg.v_mid_pad, cfg.g_mid_pad);

        // int8 量化权重：探测 {key}.int8_idx 是否存在。
        // 存在 → 上传 idx/sz 二路张量（idx 为 U8 [M,K]，重解释为 uint32 [M,K/4]）；
        // 不存在 → None，走原 fp16 加载路径（单二进制兼容两种模型文件）。
        let load_int8 = |backend: &mut dyn ComputeBackend,
                         key: &str,
                         m: usize,
                         k: usize|
         -> R<Option<Int8Handle>> {
            if st.tensor(&format!("{key}.int8_idx")).is_err() {
                return Ok(None);
            }
            let idx_t = st.tensor(&format!("{key}.int8_idx"))?; // U8 [M, K]
            let sz_t = st.tensor(&format!("{key}.int8_sz"))?; // U32 [M, K/128]
            let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
            let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
            assert_eq!(idx_u32.len(), m * k / 4, "{key}.int8_idx 形状不符");
            assert_eq!(sz_u32.len(), m * k / 128, "{key}.int8_sz 形状不符");
            let idx_gpu = backend.create_tensor(m * k / 4, TensorDtype::U32)?;
            backend.upload_u32(idx_gpu, idx_u32)?;
            let sz_gpu = backend.create_tensor(m * k / 128, TensorDtype::U32)?;
            backend.upload_u32(sz_gpu, sz_u32)?;
            log::info!("layer {idx}: {key} 使用 int8 量化权重（{m}x{k}）");
            Ok(Some(Int8Handle {
                idx: idx_gpu,
                sz: sz_gpu,
                m,
                k,
            }))
        };
        // 二路量化权重统一加载：优先 int8 → fp16（按模型文件内容自动路由）。
        // 返回 (int8, fp16)，二者至多一个为 Some。
        let load_linear = |backend: &mut dyn ComputeBackend,
                           key: &str,
                           m: usize,
                           k: usize|
         -> R<(Option<Int8Handle>, Option<TensorId>)> {
            if let Some(a8) = load_int8(backend, key, m, k)? {
                return Ok((Some(a8), None));
            }
            Ok((None, Some(load_linear_f16(backend, key.to_string())?)))
        };
        let (ffn_key_a8, ffn_key_w16) = load_linear(
            backend,
            &format!("blocks.{idx}.ffn.key.weight"),
            cfg.ffn_hidden,
            c,
        )?;
        let (ffn_value_a8, ffn_value_w16) = load_linear(
            backend,
            &format!("blocks.{idx}.ffn.value.weight"),
            c,
            cfg.ffn_hidden,
        )?;
        // 稀疏 FFN 平铺权重（fp16，CUDA 稀疏内核用）：两种量化均构建，解码按 r2 非零列只读。
        // `FFN_VALUE_GEMM`（默认开）时返回 (None, dense16)——改用稠密张量核 GEMM。
        let (ffn_value_tiled, ffn_value_dense16) = load_ffn_value_tiled(
            backend,
            format!("blocks.{idx}.ffn.value.weight"),
            c,
            cfg.ffn_hidden,
        )?;
        let (receptance_a8, receptance_w16) = load_linear(
            backend,
            &format!("blocks.{idx}.att.receptance.weight"),
            c,
            c,
        )?;
        let (key_a8, key_w16) =
            load_linear(backend, &format!("blocks.{idx}.att.key.weight"), c, c)?;
        let (value_a8, value_w16) =
            load_linear(backend, &format!("blocks.{idx}.att.value.weight"), c, c)?;
        // att.output：int8 时走对应 GEMM（零 fp16 副本）；
        // fp32 诊断副本仅 GEMM_DIAG 时用反量化权重创建（正式推理不创建，省 ~26MB/层）
        let want_diag = std::env::var("GEMM_DIAG").is_ok();
        let (att_output_a8, output_w16, output_w) = if let Some(a8) =
            load_int8(backend, &format!("blocks.{idx}.att.output.weight"), c, c)?
        {
            let ow = if want_diag {
                let key = format!("blocks.{idx}.att.output.weight");
                let idx_t = st.tensor(&format!("{key}.int8_idx"))?;
                let sz_t = st.tensor(&format!("{key}.int8_sz"))?;
                let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
                let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
                let w32 = dequant_int8(idx_u32, sz_u32, c, c);
                let t32 = backend.create_tensor(w32.len(), TensorDtype::F32)?;
                backend.upload(t32, &w32)?;
                Some(t32)
            } else {
                None
            };
            (Some(a8), None, ow)
        } else {
            (
                None,
                Some(load_linear_f16(
                    backend,
                    format!("blocks.{idx}.att.output.weight"),
                )?),
                if want_diag {
                    Some(load_linear_diag(
                        backend,
                        format!("blocks.{idx}.att.output.weight"),
                    )?)
                } else {
                    None
                },
            )
        };

        let mut layer = Self {
            ln1_w: load1(backend, "ln1.weight")?,
            ln1_b: load1(backend, "ln1.bias")?,
            ln2_w: load1(backend, "ln2.weight")?,
            ln2_b: load1(backend, "ln2.bias")?,
            ln_x_w: load1(backend, "att.ln_x.weight")?,
            ln_x_b: load1(backend, "att.ln_x.bias")?,
            x_r: load1(backend, "att.x_r")?,
            x_w: load1(backend, "att.x_w")?,
            x_k: load1(backend, "att.x_k")?,
            x_v: load1(backend, "att.x_v")?,
            x_a: load1(backend, "att.x_a")?,
            x_g: load1(backend, "att.x_g")?,
            w0: load1(backend, "att.w0")?,
            a0: load1(backend, "att.a0")?,
            v0: load1(backend, "att.v0")?,
            // 低秩权重转置为 [out, in]
            w1: load_lowrank(backend, "w1", wm, c)?, // [out=wm, in=C]
            w2: load_lowrank(backend, "w2", c, wm)?, // [out=C, in=wm]
            a1: load_lowrank(backend, "a1", am, c)?,
            a2: load_lowrank(backend, "a2", c, am)?,
            v1: load_lowrank(backend, "v1", vm, c)?,
            v2: load_lowrank(backend, "v2", c, vm)?,
            g1: load_lowrank(backend, "g1", gm, c)?,
            g2: load_lowrank(backend, "g2", c, gm)?,
            r_k: load1(backend, "att.r_k")?,
            k_k: load1(backend, "att.k_k")?,
            k_a: load1(backend, "att.k_a")?,
            ffn_x_k: load1(backend, "ffn.x_k")?,
            // 诊断用 fp32 输出权重（仅 GEMM_DIAG 参考路径）
            output_w,
            // fp16 线性权重（非 int8 矩阵常驻；int8 矩阵为 None，走 int8 GEMM/GEMV 零副本）
            receptance_w16,
            key_w16,
            value_w16,
            output_w16,
            ffn_key_w16,
            ffn_value_w16,
            ffn_value_tiled,
            ffn_value_dense16,
            ffn_key_a8,
            ffn_value_a8,
            att_output_a8,
            receptance_a8,
            key_a8,
            value_a8,
            // fp16 低秩权重（补齐到 mid_pad）
            w1_16: load_lowrank_f16(backend, "w1", wm, c, wwp, c)?, // [wm_pad, C]
            w2_16: load_lowrank_f16(backend, "w2", c, wm, c, wwp)?, // [C, wm_pad]
            a1_16: load_lowrank_f16(backend, "a1", am, c, aap, c)?, // [am_pad, C]
            a2_16: load_lowrank_f16(backend, "a2", c, am, c, aap)?, // [C, am_pad]
            v1_16: load_lowrank_f16(backend, "v1", vm, c, vvp, c)?, // [vm_pad, C]
            v2_16: load_lowrank_f16(backend, "v2", c, vm, c, vvp)?, // [C, vm_pad]
            g1_16: load_lowrank_f16(backend, "g1", gm, c, ggp, c)?, // [gm_pad, C]
            g2_16: load_lowrank_f16(backend, "g2", c, gm, c, ggp)?, // [C, gm_pad]
        };
        // 权重上传完成，释放全部 host（系统内存）缓冲，仅保留 device 拷贝
        layer.drop_weight_hosts(backend);
        Ok(layer)
    }

    /// 释放本层所有权重的 host（系统内存）缓冲。权重上传完成后调用，运行期只读 device。
    fn drop_weight_hosts(&mut self, backend: &mut dyn ComputeBackend) {
        // fp32 一维/低秩权重
        for t in [
            &mut self.ln1_w,
            &mut self.ln1_b,
            &mut self.ln2_w,
            &mut self.ln2_b,
            &mut self.ln_x_w,
            &mut self.ln_x_b,
            &mut self.x_r,
            &mut self.x_w,
            &mut self.x_k,
            &mut self.x_v,
            &mut self.x_a,
            &mut self.x_g,
            &mut self.w0,
            &mut self.a0,
            &mut self.v0,
            &mut self.w1,
            &mut self.w2,
            &mut self.a1,
            &mut self.a2,
            &mut self.v1,
            &mut self.v2,
            &mut self.g1,
            &mut self.g2,
            &mut self.r_k,
            &mut self.k_k,
            &mut self.k_a,
            &mut self.ffn_x_k,
        ] {
            backend.drop_host(*t);
        }
        // 诊断用 output_w（fp32，Option：仅 GEMM_DIAG 时存在）
        if let Some(t) = self.output_w.take() {
            backend.drop_host(t);
        }
        // fp16 低秩权重（常驻）
        for t in [
            &mut self.w1_16,
            &mut self.w2_16,
            &mut self.a1_16,
            &mut self.a2_16,
            &mut self.v1_16,
            &mut self.v2_16,
            &mut self.g1_16,
            &mut self.g2_16,
        ] {
            backend.drop_host(*t);
        }
        // fp16 线性权重（Option）：非 int8 矩阵常驻，释放其 host；int8 矩阵此时为 None 跳过
        for t in [
            &mut self.receptance_w16,
            &mut self.key_w16,
            &mut self.value_w16,
            &mut self.output_w16,
            &mut self.ffn_key_w16,
            &mut self.ffn_value_w16,
            &mut self.ffn_value_tiled,
            &mut self.ffn_value_dense16,
        ] {
            if let Some(t) = t.as_mut() {
                backend.drop_host(*t);
            }
        }
    }
}

impl GpuState {
    fn new(backend: &mut dyn ComputeBackend, c: usize, h: usize, n: usize) -> R<Self> {
        Self::new_batched(backend, c, h, n, 1)
    }

    /// batch 版：每张量为 [batch, ...]（slot 主序）。
    fn new_batched(
        backend: &mut dyn ComputeBackend,
        c: usize,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<Self> {
        let bsz = batch.max(1);
        let tmix_x = backend.create_tensor(c * bsz, TensorDtype::F32)?;
        backend.upload(tmix_x, &vec![0.0; c * bsz])?;
        let tmix_rnn = backend.create_tensor(h * n * n * bsz, wkv_state_dtype())?;
        backend.upload(tmix_rnn, &vec![0.0; h * n * n * bsz])?;
        let cmix_x = backend.create_tensor(c * bsz, TensorDtype::F32)?;
        backend.upload(cmix_x, &vec![0.0; c * bsz])?;
        Ok(Self {
            tmix_x,
            tmix_rnn,
            cmix_x,
        })
    }

    /// 重置 RNN 状态为零（用于多次独立 forward）
    fn reset(
        &self,
        backend: &dyn ComputeBackend,
        c: usize,
        h: usize,
        n: usize,
        batch: usize,
    ) -> R<()> {
        backend.upload(self.tmix_x, &vec![0.0; c * batch])?;
        backend.upload(self.tmix_rnn, &vec![0.0; h * n * n * batch])?;
        backend.upload(self.cmix_x, &vec![0.0; c * batch])?;
        Ok(())
    }
}

/// web-rwkv 风格的模型加载器（对标 `web-rwkv::ModelBuilder`）。
/// 封装「创建 Runtime + 从 safetensors 加载模型 + 绑定 State」，
/// 供服务端（ai00-server）以最小配置加载模型并得到 `Bundle`。
#[derive(Debug, Clone)]
pub struct ModelBuilder {
    path: String,
}

impl ModelBuilder {
    /// 指定模型文件路径（safetensors `.st`，按文件名自动路由 fp16/int8）。
    pub fn new(path: impl Into<String>) -> Self {
        Self { path: path.into() }
    }

    /// 构建模型并绑定一个零初始化的 `State`，得到 `Bundle`。
    /// 后端自动选择（`detect_backend()`）：CUDA 算子就绪前默认为 Vulkan。
    pub fn build(self) -> R<Bundle> {
        let backend = crate::backend::create_backend(crate::backend::detect_backend())?;
        let mut model = GpuModel::from_safetensors(backend, &self.path)?;
        let state = model.create_state()?;
        Ok(Bundle { model, state })
    }
}

/// web-rwkv 风格的模型+状态聚合（对标 `web-rwkv::Bundle`）。
/// 服务端持有单个 Bundle 即可完成一次会话的完整推理：`infer` 推进文本、
/// `state_back`/`state_load` 存取会话态、`info` 读取模型规模。
pub struct Bundle {
    /// 模型（权重已上传 GPU）
    pub model: GpuModel,
    /// 会话推理状态（可 `state_back`/`state_load` 持久化）
    pub state: State,
}

impl Bundle {
    /// 用给定模型创建一个 Bundle（内部为模型新建零初始态）。
    pub fn new(mut model: GpuModel) -> R<Self> {
        let state = model.create_state()?;
        Ok(Self { model, state })
    }

    /// 公开展示模型元信息。
    pub fn info(&self) -> ModelInfo {
        self.model.info()
    }

    /// 前向推理：推进整段 tokens，返回最后 token 的 logits [vocab]。
    /// 在内部 `state` 上累积 RNN 状态（等价 `model.forward_with_state(&mut state, tokens)`）。
    pub fn infer_tokens(&mut self, tokens: &[u32]) -> R<Vec<f32>> {
        self.model.forward_with_state(&mut self.state, tokens)
    }

    /// 前向推理（sequence-parallel）：把整段 tokens 一次贯穿各层。
    pub fn infer_seq(&mut self, tokens: &[u32]) -> R<Vec<f32>> {
        self.model.forward_seq_with_state(&mut self.state, tokens)
    }

    /// 清空累计的 per-kernel profiling 时间（仅诊断；CUDA 覆盖，其余为 no-op）。
    pub fn clear_kernel_prof(&mut self) {
        self.model.clear_kernel_prof();
    }

    /// 打印累计的 per-kernel profiling 时间（仅诊断）。
    pub fn dump_kernel_prof(&mut self) {
        self.model.dump_kernel_prof();
    }

    /// 把当前会话态整态下载到 CPU（存盘/state tuning 用）。
    pub fn state_back(&self) -> R<Vec<f32>> {
        self.model.state_back(&self.state)
    }

    /// 从 CPU 数据回灌会话态（与 `state_back` 布局对应）。
    pub fn state_load(&self, data: &[f32]) -> R<()> {
        self.model.state_load(&self.state, data)
    }

    /// 清零会话态（回到首次推理前）。
    pub fn reset(&mut self) -> R<()> {
        // 注意：必须重置 Bundle 自身的 `self.state`（infer_tokens/infer 实际使用的会话态），
        // 而非模型内部态。此前错误地调用 `model.reset_state()`，只重置了模型内部 `Option<State>`，
        // 导致会话态残留、同样的输入多次前向结果不一致（伪“GPU 非确定性”）。
        self.model.reset_state_of(&self.state)
    }

    /// 分块增量推理（对标 web-rwkv 的 chunked `infer`）。
    /// 把长 token 序列按 `chunk_size` 切成块，逐块推进内部 `state`，返回 logits。
    /// 长 prompt 不再一次性 prefill（避免单块超大缓冲/显存峰值），块间 RNN 状态自然累积。
    /// 注意：块大小变化会重建 seq 缓冲（clear_cache），服务端应尽量用固定块大小。
    ///
    /// 输出模式：
    /// - `RnnOption::Last`：返回最后一个 token 的 logits [vocab]（自回归生成用）
    /// - `RnnOption::Full`：返回每个 token 位置的 logits，拼接为 `Vec<f32>`（长度 num_tok × vocab）
    pub fn infer(&mut self, input: &RnnInput) -> R<Vec<f32>> {
        let cs = input.chunk_size.max(1);
        let vocab = self.info().num_vocab;
        match input.option {
            RnnOption::Last => {
                let mut logits = vec![0.0f32; vocab];
                for block in input.tokens.chunks(cs) {
                    logits = self.model.forward_seq_with_state(&mut self.state, block)?;
                }
                Ok(logits)
            }
            // Full：每个 token 单独前向，收集每位置 logits（prompt 一次性打分/state tuning 用）
            RnnOption::Full => {
                let mut out = Vec::with_capacity(input.tokens.len() * vocab);
                for &tok in &input.tokens {
                    let lg = self.model.forward_seq_with_state(&mut self.state, &[tok])?;
                    out.extend_from_slice(&lg);
                }
                Ok(out)
            }
        }
    }

    /// 便捷封装：Last 输出模式的 chunked 推理（等价 `infer(&RnnInput{ tokens, chunk_size, ..})`）。
    pub fn infer_chunked(&mut self, tokens: &[u32], chunk_size: usize) -> R<Vec<f32>> {
        let input = RnnInput {
            tokens: tokens.to_vec(),
            chunk_size,
            option: RnnOption::Last,
        };
        self.infer(&input)
    }

    /// GPU 采样：推进整段 tokens，返回最后一个 token 的 argmax 索引（全 GPU，不下载 logits）。
    /// 对标 albatross 的 torch.argmax：只回传 4 字节索引，省去每 token 下载 65536 个 f32。
    pub fn infer_argmax(&mut self, tokens: &[u32]) -> R<u32> {
        self.model
            .forward_argmax_with_state(&mut self.state, tokens)
    }

    /// GPU self-loop 批量生成：在**单次 submit** 内连续采样 n 个 token。
    /// 首个 token 由 CPU 写入（seed），随后每轮 argmax 结果直接写回同一 host 缓冲，
    /// 全程无 CPU 回读/回传 token、无每 token 一次 submit+wait，消除 CPU⟷GPU 交换开销。
    /// 返回生成的 n 个 token 索引。
    pub fn infer_argmax_selfloop(&mut self, seed: u32, n: usize) -> R<Vec<u32>> {
        self.model
            .forward_argmax_selfloop_with_state(&mut self.state, seed, n)
    }

    /// GPU 采样：推进整段 tokens，返回最后一个 token 的采样索引（penalty/temperature/top-k/top-p）。
    /// 全 GPU 过滤+采样，只回传 4 字节索引；`sp.seed` 由调用方控制（每生成递增）。
    /// `history` 为已生成 token（惩罚计数用，空则跳过惩罚）。
    pub fn infer_sample(&mut self, tokens: &[u32], sp: &SamplerParams, history: &[u32]) -> R<u32> {
        self.model
            .forward_sample_with_state(&mut self.state, tokens, sp, history)
    }

    /// GPU self-loop 批量采样生成：在**单次 submit** 内连续采样 n 个 token。
    /// 每轮用带参数（temperature/top-k/top-p）的采样替换 argmax，seed 随轮次递增。
    /// 返回生成的 n 个 token 索引。
    pub fn infer_sample_selfloop(
        &mut self,
        seed: u32,
        n: usize,
        sp: &SamplerParams,
    ) -> R<Vec<u32>> {
        self.model
            .forward_sample_selfloop_with_state(&mut self.state, seed, n, sp)
    }
}

/// web-rwkv 风格的推理输出模式（对标 `web-rwkv::RnnOption`）。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RnnOption {
    /// 仅输出最后一个 token 的预测（自回归生成用）
    #[default]
    Last,
    /// 输出所有 token 的预测（分别返回每个位置的 logits）
    Full,
}

/// web-rwkv 风格的推理输入（对标 `web-rwkv::RnnInput`）。
/// 携带待推理的 token 序列、分块大小与输出模式；`Bundle::infer` 据此做 chunked 增量推理。
#[derive(Debug, Clone)]
pub struct RnnInput {
    /// 待前向的 token 序列
    pub tokens: Vec<u32>,
    /// 分块大小（长 prompt 分块，避免单次超大 prefill）
    pub chunk_size: usize,
    /// 输出模式（Last=只返回最后 token 的 logits；Full=返回全部每位置 logits）
    pub option: RnnOption,
}

impl RnnInput {
    /// 构造推理输入。`chunk_size` 默认 0（表示不强制分块，交给实现选择）。
    pub fn new(tokens: Vec<u32>, chunk_size: usize) -> Self {
        Self {
            tokens,
            chunk_size,
            option: RnnOption::Last,
        }
    }
}

/// GPU 采样参数（penalty / temperature / top-k / top-p）。
/// 传给 `Bundle::infer_sample*`，在 GPU 上对 logits 过滤后按概率采样。
/// 惩罚公式与客户端池 / ai00-server 对齐（作用于 softmax 前的 logits）：
///   repetition_penalty（缩放，1.0=禁用）、
///   presence_penalty + frequency_penalty × cnt^penalty_decay（0.0=禁用）。
#[derive(Debug, Clone, Copy)]
pub struct SamplerParams {
    /// 温度 >0；logits 除以 temperature 后做 softmax。0 或负视为 1（不缩放）。
    pub temperature: f32,
    /// top-k：仅保留概率/ logits 前 top_k 个候选。0 表示禁用。
    pub top_k: u32,
    /// top-p：按概率从高到低累积，达到 top_p 后截断。1.0（或 >=1）表示禁用。
    pub top_p: f32,
    /// 采样随机种子；每生成一个 token 应递增以保证多样性。
    pub seed: u32,
    /// repetition_penalty：对历史中出现的 token，logit>0 时 /=rp，logit<0 时 *=rp。1.0 表示禁用。
    pub repetition_penalty: f32,
    /// frequency_penalty：logit 减去 fp × 出现次数^penalty_decay。0.0 表示禁用。
    pub frequency_penalty: f32,
    /// presence_penalty：出现过的 token 一律 logit 减去 pp。0.0 表示禁用。
    pub presence_penalty: f32,
    /// frequency_penalty 的次数衰减指数：`cnt^decay`。1.0 = 现行线性行为
    /// （单流 kernel 与历史调用方语义不变）；<1.0 = 高频 token 惩罚次线性增长
    /// （客户端池默认 0.99654026）。
    pub penalty_decay: f32,
}

impl Default for SamplerParams {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,
            top_p: 1.0,
            seed: 0,
            repetition_penalty: 1.0,
            frequency_penalty: 0.0,
            presence_penalty: 0.0,
            penalty_decay: 1.0,
        }
    }
}

/// 可序列化的推理状态（对标 web-rwkv 的 `State`）。
///
/// 持有每层 RNN 状态（tmix_x / tmix_rnn / cmix_x）与跨层共享的 v_first，
/// 支持 `reset`（清零）、`back`（下载到 CPU `Vec<f32>`）与 `load`（从 CPU 回灌）。
/// 是会话保存与 state tuning 的基础：`forward` 前进文本 → `back()` 取态 → 持久化 → `load()` 回灌。
pub struct State {
    layers: Vec<GpuState>,
    v_first: TensorId,
    v_first_set: bool,
    /// batch 并发布局（batch > 1 时张量为 [batch, ...]；1 = 单序列）。
    batch: usize,
}

impl State {
    /// 创建零初始化的状态（单序列）。
    pub fn new(
        backend: &mut dyn ComputeBackend,
        c: usize,
        h: usize,
        n: usize,
        n_layer: usize,
    ) -> R<Self> {
        Self::new_batched(backend, c, h, n, n_layer, 1)
    }

    /// 创建零初始化的 batch 状态：每层张量为 [batch, C] / [batch, H, N, N]，
    /// v_first 为 [batch, C]（slot 主序；单序列 batch=1 与 `new` 等价）。
    pub fn new_batched(
        backend: &mut dyn ComputeBackend,
        c: usize,
        h: usize,
        n: usize,
        n_layer: usize,
        batch: usize,
    ) -> R<Self> {
        let bsz = batch.max(1);
        let mut layers = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            layers.push(GpuState::new_batched(backend, c, h, n, bsz)?);
        }
        let v_first = backend.create_tensor(c * bsz, TensorDtype::F16)?;
        backend.upload(v_first, &vec![0.0; c * bsz])?;
        Ok(Self {
            layers,
            v_first,
            v_first_set: false,
            batch: bsz,
        })
    }

    /// 该状态的 batch 数（1 = 单序列）。
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// 把整态下载到 CPU 为连续 `Vec<f32>`（布局见 `load`）。
    /// batch>1 时返回 [batch] 份单序列布局顺序拼接（slot 主序）。
    pub fn back(&self, backend: &dyn ComputeBackend, c: usize, h: usize, n: usize) -> R<Vec<f32>> {
        let per_layer = c + h * n * n + c;
        let mut out = Vec::with_capacity(self.batch * (self.layers.len() * per_layer + c));
        for b in 0..self.batch {
            let off_x = b * c;
            let off_rnn = b * h * n * n;
            for s in &self.layers {
                let x = backend.download(s.tmix_x)?;
                out.extend_from_slice(&x[off_x..off_x + c]);
                let rnn = backend.download(s.tmix_rnn)?;
                out.extend_from_slice(&rnn[off_rnn..off_rnn + h * n * n]);
                let cx = backend.download(s.cmix_x)?;
                out.extend_from_slice(&cx[off_x..off_x + c]);
            }
            let vf = backend.download(self.v_first)?;
            out.extend_from_slice(&vf[b * c..b * c + c]);
        }
        Ok(out)
    }

    /// 从 CPU 数据回灌整态（与 `back` 布局一一对应）。
    pub fn load(
        &self,
        backend: &dyn ComputeBackend,
        data: &[f32],
        c: usize,
        h: usize,
        n: usize,
    ) -> R<()> {
        let per = c + h * n * n + c;
        let expect = self.batch * (self.layers.len() * per + c);
        if data.len() != expect {
            return Err(format!("State::load 长度不符: 得 {} 期望 {expect}", data.len()).into());
        }
        if self.batch == 1 {
            // 单序列：整段覆盖上传（Vulkan/CUDA 通用线上路径）。
            let mut pos = 0usize;
            for s in &self.layers {
                backend.upload(s.tmix_x, &data[pos..pos + c])?;
                pos += c;
                backend.upload(s.tmix_rnn, &data[pos..pos + h * n * n])?;
                pos += h * n * n;
                backend.upload(s.cmix_x, &data[pos..pos + c])?;
                pos += c;
            }
            backend.upload(self.v_first, &data[pos..pos + c])?;
            return Ok(());
        }
        // batch：逐 slot 段上传（仅 CUDA 支持 upload_part）。
        let mut pos = 0usize;
        for b in 0..self.batch {
            for s in &self.layers {
                backend.upload_part(s.tmix_x, b * c, &data[pos..pos + c])?;
                pos += c;
                backend.upload_part(s.tmix_rnn, b * h * n * n, &data[pos..pos + h * n * n])?;
                pos += h * n * n;
                backend.upload_part(s.cmix_x, b * c, &data[pos..pos + c])?;
                pos += c;
            }
            backend.upload_part(self.v_first, b * c, &data[pos..pos + c])?;
            pos += c;
        }
        Ok(())
    }

    /// 单 slot 状态下载（batch 布局）：返回该 slot 的单序列布局数据。
    pub fn slot_back(
        &self,
        backend: &dyn ComputeBackend,
        slot: usize,
        c: usize,
        h: usize,
        n: usize,
    ) -> R<Vec<f32>> {
        if slot >= self.batch {
            return Err(format!("slot_back: slot {slot} >= batch {}", self.batch).into());
        }
        // 只下载该 slot 的行段（整表下载在 batch=16 时 ~260MB，单行 ~15MB）。
        let rnn = h * n * n;
        let mut out = Vec::with_capacity(self.layers.len() * (2 * c + rnn) + c);
        for s in &self.layers {
            out.extend_from_slice(&backend.download_part(s.tmix_x, slot * c, c)?);
            out.extend_from_slice(&backend.download_part(s.tmix_rnn, slot * rnn, rnn)?);
            out.extend_from_slice(&backend.download_part(s.cmix_x, slot * c, c)?);
        }
        out.extend_from_slice(&backend.download_part(self.v_first, slot * c, c)?);
        Ok(out)
    }

    /// 设备内单 slot 行迁移：把 slot `from` 的全部状态行拷到 slot `to`
    /// （连续批调度的槽位紧凑化用——全程设备内拷贝，不经过主机）。
    pub fn slot_move(
        &self,
        backend: &mut dyn ComputeBackend,
        from: usize,
        to: usize,
        c: usize,
        h: usize,
        n: usize,
    ) -> R<()> {
        if from >= self.batch || to >= self.batch {
            return Err(format!("slot_move: {from}->{to} 超出 batch {}", self.batch).into());
        }
        if from == to {
            return Ok(());
        }
        let rnn = h * n * n;
        for s in &self.layers {
            backend.copy_range(s.tmix_x, from * c, s.tmix_x, to * c, c)?;
            backend.copy_range(s.tmix_rnn, from * rnn, s.tmix_rnn, to * rnn, rnn)?;
            backend.copy_range(s.cmix_x, from * c, s.cmix_x, to * c, c)?;
        }
        backend.copy_range(self.v_first, from * c, self.v_first, to * c, c)?;
        Ok(())
    }

    /// 单 slot 状态回灌（batch 布局）：把单序列布局数据写入指定 slot 段。
    pub fn slot_load(
        &self,
        backend: &dyn ComputeBackend,
        slot: usize,
        data: &[f32],
        c: usize,
        h: usize,
        n: usize,
    ) -> R<()> {
        if slot >= self.batch {
            return Err(format!("slot_load: slot {slot} >= batch {}", self.batch).into());
        }
        let per = c + h * n * n + c;
        let expect = self.layers.len() * per + c;
        if data.len() != expect {
            return Err(
                format!("State::slot_load 长度不符: 得 {} 期望 {expect}", data.len()).into(),
            );
        }
        let mut pos = 0usize;
        for s in &self.layers {
            backend.upload_part(s.tmix_x, slot * c, &data[pos..pos + c])?;
            pos += c;
            backend.upload_part(s.tmix_rnn, slot * h * n * n, &data[pos..pos + h * n * n])?;
            pos += h * n * n;
            backend.upload_part(s.cmix_x, slot * c, &data[pos..pos + c])?;
            pos += c;
        }
        backend.upload_part(self.v_first, slot * c, &data[pos..pos + c])?;
        Ok(())
    }

    /// 清零全部状态（含 v_first 缓冲）。
    pub fn reset(&self, backend: &dyn ComputeBackend, c: usize, h: usize, n: usize) -> R<()> {
        for s in &self.layers {
            s.reset(backend, c, h, n, self.batch)?;
        }
        backend.upload(self.v_first, &vec![0.0; c * self.batch])?;
        Ok(())
    }
}

impl WorkBuffers {
    /// 创建 decode 工作缓冲。`batch`：单序列路径传 1；batch 并发路径传 B
    ///（所有 per-slot 缓冲为 [batch, ...] 布局，kernel 一次读权重算 B 份）。
    fn new(
        backend: &mut dyn ComputeBackend,
        c: usize,
        vocab: usize,
        cfg: &ModelConfig,
        batch: usize,
    ) -> R<Self> {
        let bsz = batch.max(1);
        // 创建 C 大小的 buffer 并初始化为 0（×batch）
        let mk_c = |b: &mut dyn ComputeBackend| -> R<TensorId> {
            let t = b.create_tensor(c * bsz, TensorDtype::F32)?;
            b.upload(t, &vec![0.0; c * bsz])?;
            Ok(t)
        };
        let mk = |b: &mut dyn ComputeBackend, len: usize| -> R<TensorId> {
            let t = b.create_tensor(len * bsz, TensorDtype::F32)?;
            b.upload(t, &vec![0.0; len * bsz])?;
            Ok(t)
        };
        // fp16 零缓冲（w/a/g/v 输出，链第二级下游以 fp16 读取减半带宽）
        let mk_c16 = |b: &mut dyn ComputeBackend| -> R<TensorId> {
            let t = b.create_tensor(c * bsz, TensorDtype::F16)?;
            b.upload(t, &vec![0.0f32; c * bsz])?;
            Ok(t)
        };

        Ok(Self {
            x: mk_c(&mut *backend)?,
            ln1: mk_c(&mut *backend)?,
            xr: mk_c(&mut *backend)?,
            xw: mk_c(&mut *backend)?,
            xk: mk_c(&mut *backend)?,
            xv: mk_c(&mut *backend)?,
            xa: mk_c(&mut *backend)?,
            xg: mk_c(&mut *backend)?,
            prev_x: mk_c(&mut *backend)?,
            r: mk_c(&mut *backend)?,
            k: mk_c(&mut *backend)?,
            v: mk_c16(&mut *backend)?,
            v_full: mk_c(&mut *backend)?,
            gate: mk_c(&mut *backend)?,
            w_full: mk_c(&mut *backend)?,
            w_sig: mk_c(&mut *backend)?,
            w: mk_c16(&mut *backend)?,
            a_full: mk_c(&mut *backend)?,
            a: mk_c16(&mut *backend)?,
            kk_l2: mk_c(&mut *backend)?,
            k_mod: mk_c(&mut *backend)?,
            b_vec: mk_c(&mut *backend)?,
            y: mk_c(&mut *backend)?,
            y_norm: mk_c(&mut *backend)?,
            g: mk_c16(&mut *backend)?,
            y_g: mk_c(&mut *backend)?,
            y_out: mk_c(&mut *backend)?,
            ln2: mk_c(&mut *backend)?,
            prev_c: mk_c(&mut *backend)?,
            xb: mk_c(&mut *backend)?,
            v2: mk_c(&mut *backend)?,
            x_norm: mk_c(&mut *backend)?,
            tmp_c: mk_c(&mut *backend)?,
            // 其他大小
            v_mid: mk(&mut *backend, cfg.v_mid)?,
            w_mid: mk(&mut *backend, cfg.w_mid)?,
            a_mid: mk(&mut *backend, cfg.a_mid)?,
            g_mid: mk(&mut *backend, cfg.g_mid)?,
            r2: mk(&mut *backend, cfg.ffn_hidden)?,
            logits: mk(&mut *backend, vocab)?,
            token_argmax: mk(&mut *backend, 1)?,
            current_token: mk(&mut *backend, 1)?,
        })
    }

    /// 释放全部工作缓冲的设备内存（batch 变化重建时调用，防泄漏）。
    fn free(&mut self, backend: &mut dyn ComputeBackend) {
        for t in [
            self.x,
            self.ln1,
            self.xr,
            self.xw,
            self.xk,
            self.xv,
            self.xa,
            self.xg,
            self.prev_x,
            self.r,
            self.k,
            self.v,
            self.v_full,
            self.gate,
            self.w_full,
            self.w_sig,
            self.w,
            self.a_full,
            self.a,
            self.kk_l2,
            self.k_mod,
            self.b_vec,
            self.y,
            self.y_norm,
            self.g,
            self.y_g,
            self.y_out,
            self.ln2,
            self.prev_c,
            self.xb,
            self.v2,
            self.x_norm,
            self.tmp_c,
            self.v_mid,
            self.w_mid,
            self.a_mid,
            self.g_mid,
            self.r2,
            self.logits,
            self.token_argmax,
            self.current_token,
        ] {
            backend.free_tensor(t);
        }
    }
}

impl SeqBuffers {
    /// 创建序列并行的 [T, *] 工作缓冲区
    fn new(
        backend: &mut dyn ComputeBackend,
        t: usize,
        c: usize,
        vocab: usize,
        cfg: &ModelConfig,
    ) -> R<Self> {
        let m_pad = t.div_ceil(256) * 256;
        let mk = |b: &mut dyn ComputeBackend, len: usize| -> R<TensorId> {
            let buf = b.create_tensor(len, TensorDtype::F32)?;
            b.upload(buf, &vec![0.0; len])?;
            Ok(buf)
        };
        let mk_c = |b: &mut dyn ComputeBackend, len: usize| -> R<TensorId> {
            let buf = b.create_tensor(t * len, TensorDtype::F32)?;
            b.upload(buf, &vec![0.0; t * len])?;
            Ok(buf)
        };
        let mk_pad = |b: &mut dyn ComputeBackend, len: usize| -> R<TensorId> {
            let buf = b.create_tensor(m_pad * len, TensorDtype::F32)?;
            b.upload(buf, &vec![0.0; m_pad * len])?;
            Ok(buf)
        };
        let mk_f16 = |b: &mut dyn ComputeBackend, len: usize| -> R<TensorId> {
            let buf = b.create_tensor(m_pad * len, TensorDtype::F16)?;
            b.upload(buf, &vec![0.0; m_pad * len])?;
            Ok(buf)
        };

        Ok(Self {
            t,
            m_pad,
            x: mk_pad(&mut *backend, c)?,
            ln1: mk_c(&mut *backend, c)?,
            xr: mk_c(&mut *backend, c)?,
            xw: mk_c(&mut *backend, c)?,
            xk: mk_c(&mut *backend, c)?,
            xv: mk_c(&mut *backend, c)?,
            xa: mk_c(&mut *backend, c)?,
            xg: mk_c(&mut *backend, c)?,
            r: mk_pad(&mut *backend, c)?,
            k: mk_pad(&mut *backend, c)?,
            v: mk_pad(&mut *backend, c)?,
            v_first: mk_pad(&mut *backend, c)?,
            v_full: mk_pad(&mut *backend, c)?,
            gate: mk_c(&mut *backend, c)?,
            w_full: mk_pad(&mut *backend, c)?,
            w_sig: mk_c(&mut *backend, c)?,
            w: mk_c(&mut *backend, c)?,
            a_full: mk_pad(&mut *backend, c)?,
            a: mk_c(&mut *backend, c)?,
            kk_l2: mk_c(&mut *backend, c)?,
            k_mod: mk_c(&mut *backend, c)?,
            b_vec: mk_c(&mut *backend, c)?,
            y: mk_c(&mut *backend, c)?,
            y_norm: mk_c(&mut *backend, c)?,
            g: mk_pad(&mut *backend, c)?,
            y_g: mk_c(&mut *backend, c)?,
            y_out: mk_pad(&mut *backend, c)?,
            diag_out_ref: mk_pad(&mut *backend, c)?,
            ln2: mk_c(&mut *backend, c)?,
            xb: mk_c(&mut *backend, c)?,
            v2: mk_pad(&mut *backend, c)?,
            x_norm: mk_c(&mut *backend, c)?,
            tmp_c: mk_c(&mut *backend, c)?,
            v_mid: mk_pad(&mut *backend, cfg.v_mid_pad)?,
            w_mid: mk_pad(&mut *backend, cfg.w_mid_pad)?,
            a_mid: mk_pad(&mut *backend, cfg.a_mid_pad)?,
            g_mid: mk_pad(&mut *backend, cfg.g_mid_pad)?,
            r2: mk_pad(&mut *backend, cfg.ffn_hidden)?,
            head_in: mk(&mut *backend, c)?,
            logits: mk(&mut *backend, vocab)?,
            xr16: mk_f16(&mut *backend, c)?,
            xk16: mk_f16(&mut *backend, c)?,
            xv16: mk_f16(&mut *backend, c)?,
            xw16: mk_f16(&mut *backend, c)?,
            xa16: mk_f16(&mut *backend, c)?,
            xg16: mk_f16(&mut *backend, c)?,
            y_g16: mk_f16(&mut *backend, c)?,
            xb16: mk_f16(&mut *backend, c)?,
            r2_16: mk_f16(&mut *backend, cfg.ffn_hidden)?,
            v_mid16: mk_f16(&mut *backend, cfg.v_mid_pad)?,
            w_mid16: mk_f16(&mut *backend, cfg.w_mid_pad)?,
            a_mid16: mk_f16(&mut *backend, cfg.a_mid_pad)?,
            g_mid16: mk_f16(&mut *backend, cfg.g_mid_pad)?,
            // dequant 每次 GEMM 前全量覆写，无需零初始化上传
            w_scratch: backend.create_tensor(cfg.ffn_hidden * c, TensorDtype::F16)?,
        })
    }
}

// ===== 辅助函数 =====

/// safetensors view → Vec<f32>（支持 F32 和 F16）
fn tensor_to_f32(data: &TensorView) -> Vec<f32> {
    match data.dtype() {
        safetensors::tensor::Dtype::F32 => bytemuck::cast_slice::<u8, f32>(data.data()).to_vec(),
        safetensors::tensor::Dtype::F16 => {
            let f16s: &[f16] = bytemuck::cast_slice(data.data());
            f16s.iter().map(|x| x.to_f32()).collect()
        }
        d => panic!("unsupported dtype: {d:?}"),
    }
}

/// 矩阵转置: [rows, cols] → [cols, rows]（行主序）
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    // 加载关键路径：大矩阵转置按行分块多线程（各输出行写不相交区域）
    if rows * cols >= 1 << 22 {
        let mut out = vec![0.0f32; rows * cols];
        let threads = std::thread::available_parallelism()
            .map(|n| n.get())
            .unwrap_or(4)
            .min(cols.max(1));
        if threads > 1 {
            let rows_per = cols.div_ceil(threads);
            std::thread::scope(|scope| {
                for (chunk_idx, rows_out) in out.chunks_mut(rows_per * rows).enumerate() {
                    let c_start = chunk_idx * rows_per;
                    scope.spawn(move || {
                        for (i, col) in rows_out.chunks_mut(rows).enumerate() {
                            let cc = c_start + i;
                            for r in 0..rows {
                                col[r] = data[r * cols + cc];
                            }
                        }
                    });
                }
            });
            return out;
        }
    }
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// int8 CPU 反量化（非对称 per-group=128，与 tools/quantize_any4.py --bits 8 及 model.rs 一致）。
///
/// 供 GEMM_DIAG 诊断时重建 fp32 权重（正式推理直接走 int8 GEMV/GEMM，不创建此副本）
fn dequant_int8(idx: &[u32], sz: &[u32], m: usize, k: usize) -> Vec<f32> {
    let kg = k / 128;
    let mut w = vec![0.0f32; m * k];
    // 加载是启动关键路径：按行分组多线程反量化（各行写不相交区域，结果与单线程一致）
    let threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
        .min(m.max(1));
    if threads <= 1 || m < 8 {
        for r in 0..m {
            dequant_int8_row(idx, sz, kg, k, r, &mut w[r * k..(r + 1) * k]);
        }
        return w;
    }
    let rows_per = m.div_ceil(threads);
    std::thread::scope(|scope| {
        for (chunk_idx, rows) in w.chunks_mut(rows_per * k).enumerate() {
            let r_start = chunk_idx * rows_per;
            scope.spawn(move || {
                for (i, row) in rows.chunks_mut(k).enumerate() {
                    dequant_int8_row(idx, sz, kg, k, r_start + i, row);
                }
            });
        }
    });
    w
}

fn dequant_int8_row(idx: &[u32], sz: &[u32], kg: usize, k: usize, r: usize, row: &mut [f32]) {
    for (g, chunk) in row.chunks_mut(128).enumerate() {
        let szv = sz[r * kg + g];
        let scale = f16::from_bits((szv & 0xFFFF) as u16).to_f32();
        let zero = f16::from_bits((szv >> 16) as u16).to_f32();
        for (j, wv) in chunk.iter_mut().enumerate() {
            let ki = g * 128 + j;
            let pack = idx[r * (k / 4) + ki / 4];
            let q = ((pack >> ((ki % 4) * 8)) & 0xFF) as f32;
            *wv = scale * q + zero;
        }
    }
}

/// 向上取整到 a 的倍数（tensor-core GEMM 维度对齐用）
fn round_up(x: usize, a: usize) -> usize {
    x.div_ceil(a) * a
}

/// LayerNorm 按行处理多行（CPU 端预计算 emb_ln 用）
fn layer_norm_rows(
    data: &[f32],
    w: &[f32],
    b: &[f32],
    c: usize,
    rows: usize,
    eps: f32,
) -> Vec<f32> {
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        let row = &data[r * c..(r + 1) * c];
        let normed = layer_norm(row, w, b, eps);
        out[r * c..(r + 1) * c].copy_from_slice(&normed);
    }
    out
}

/// 单行 LayerNorm
fn layer_norm(x: &[f32], w: &[f32], b: &[f32], eps: f32) -> Vec<f32> {
    let n = x.len() as f32;
    let mean = x.iter().sum::<f32>() / n;
    let var = x.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n + eps;
    let inv_std = var.sqrt().recip();
    x.iter()
        .zip(w)
        .zip(b)
        .map(|((&xi, &wi), &bi)| (xi - mean) * inv_std * wi + bi)
        .collect()
}

/// 解析 safetensors 中最大层索引
fn max_layer(st: &safetensors::SafeTensors) -> usize {
    st.names()
        .iter()
        .filter_map(|k| {
            if k.starts_with("blocks.") {
                k.split('.').nth(1).and_then(|s| s.parse::<usize>().ok())
            } else {
                None
            }
        })
        .max()
        .unwrap_or(0)
}
