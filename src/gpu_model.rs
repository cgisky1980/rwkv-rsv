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

use crate::runtime::{GpuTensor, GpuTensor16, GpuTensorAny4, GpuTensorInt8, R, Runtime};

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

/// RWKV-7 GPU 模型：权重上传一次，forward 时复用工作缓冲与状态
pub struct GpuModel {
    rt: Runtime,
    config: ModelConfig,
    // 预计算的 GPU 权重
    emb_ln: GpuTensor16,  // [vocab, C] fp16 — 预计算 ln0(embed) 后的 embedding
    emb_ln_cpu: Vec<f32>, // [vocab, C] CPU 缓存（f16 舍入值，与 GPU 表逐位一致），避免每次 forward_seq 下载
    ln_out_w: GpuTensor,
    ln_out_b: GpuTensor,
    head_w16: GpuTensor16, // [vocab, C] fp16 变体（单 token 用 fp16 gemv）
    layers: Vec<GpuLayer>,
    // 工作缓冲区（forward 时复用，避免每 token 重新分配）
    bufs: WorkBuffers,
    // 序列并行工作缓冲区（forward_seq 时按需创建/复用）
    seq_bufs: Option<SeqBuffers>,
    // 每层 RNN 状态（GPU 端维护）
    state: Vec<GpuState>,
    // v_first 跨层共享，每个 token 重置（fp16，与 v 缓冲同精度）
    v_first: GpuTensor16,
    v_first_set: bool,
    // 标量缓冲区: -1/sqrt(e)，用于 w = exp(w_sig * scale)
    scale_w: GpuTensor,
}

/// 单层权重（全部已上传到 GPU）
pub struct GpuLayer {
    ln1_w: GpuTensor,
    ln1_b: GpuTensor,
    ln2_w: GpuTensor,
    ln2_b: GpuTensor,
    ln_x_w: GpuTensor,
    ln_x_b: GpuTensor,
    // token shift 系数 [C]
    x_r: GpuTensor,
    x_w: GpuTensor,
    x_k: GpuTensor,
    x_v: GpuTensor,
    x_a: GpuTensor,
    x_g: GpuTensor,
    // att biases [C]
    w0: GpuTensor,
    a0: GpuTensor,
    v0: GpuTensor,
    // 低秩权重（单 token 低秩链用 fp32；prefill 用 fp16，见 *_16）
    w1: GpuTensor,
    w2: GpuTensor,
    a1: GpuTensor,
    a2: GpuTensor,
    v1: GpuTensor,
    v2: GpuTensor,
    g1: GpuTensor,
    g2: GpuTensor,
    // element-wise 参数
    r_k: GpuTensor,     // [H, N]
    k_k: GpuTensor,     // [C]
    k_a: GpuTensor,     // [C]
    ffn_x_k: GpuTensor, // [C]
    /// 诊断用 fp32 输出权重（仅 GEMM_DIAG 参考路径使用；其余线性权重已删 fp32 副本省显存）。
    /// 仅在设置 GEMM_DIAG 时创建，正式推理不持有，省 ~26MB/层。
    output_w: Option<GpuTensor>, // [C, C]
    // any4 量化矩阵的 fp16 线性权重（prefill tensor-core GEMM 用，fp32io16 模式）——
    // 不常驻：prefill（forward_seq）前由 host 反量化临时创建，decode 前释放归还显存。
    // 非 any4 矩阵此处为 Some 常驻（直接读模型 fp16 键）；any4 矩阵为 None/临时。
    receptance_w16: Option<GpuTensor16>, // [C, C]
    key_w16: Option<GpuTensor16>,
    value_w16: Option<GpuTensor16>,
    output_w16: Option<GpuTensor16>,
    ffn_key_w16: Option<GpuTensor16>,   // [ffn_hidden, C]
    ffn_value_w16: Option<GpuTensor16>, // [C, ffn_hidden]
    // any4 量化权重（decode 单 token GEMV 用；None 表示该矩阵未量化，走 fp16 路径）
    ffn_key_a4: Option<GpuTensorAny4>,    // [ffn_hidden, C]
    ffn_value_a4: Option<GpuTensorAny4>,  // [C, ffn_hidden]
    att_output_a4: Option<GpuTensorAny4>, // [C, C]
    receptance_a4: Option<GpuTensorAny4>, // [C, C]
    key_a4: Option<GpuTensorAny4>,        // [C, C]
    value_a4: Option<GpuTensorAny4>,      // [C, C]
    // int8 量化权重（decode 单 token GEMV 用；None 表示该矩阵未量化，走 fp16/any4 路径）
    ffn_key_a8: Option<GpuTensorInt8>,    // [ffn_hidden, C]
    ffn_value_a8: Option<GpuTensorInt8>,  // [C, ffn_hidden]
    att_output_a8: Option<GpuTensorInt8>, // [C, C]
    receptance_a8: Option<GpuTensorInt8>, // [C, C]
    key_a8: Option<GpuTensorInt8>,        // [C, C]
    value_a8: Option<GpuTensorInt8>,      // [C, C]
    // fp16 低秩权重（tensor-core GEMM 用，已补齐到 [mid_pad, C] / [C, mid_pad]）—— 仅保留 fp16
    w1_16: GpuTensor16, // [wm_pad, C]
    w2_16: GpuTensor16, // [C, wm_pad]
    a1_16: GpuTensor16, // [am_pad, C]
    a2_16: GpuTensor16, // [C, am_pad]
    v1_16: GpuTensor16, // [vm_pad, C]
    v2_16: GpuTensor16, // [C, vm_pad]
    g1_16: GpuTensor16, // [gm_pad, C]
    g2_16: GpuTensor16, // [C, gm_pad]
}

/// 每层 RNN 状态（GPU 端）
pub struct GpuState {
    tmix_x: GpuTensor,   // [C] token shift
    tmix_rnn: GpuTensor, // [H, N, N] DPLR state
    cmix_x: GpuTensor,   // [C] token shift
}

/// 工作缓冲区：forward 期间复用，避免反复创建 GpuTensor
pub struct WorkBuffers {
    // [C] 大小
    x: GpuTensor,
    ln1: GpuTensor,
    xr: GpuTensor,
    xw: GpuTensor,
    xk: GpuTensor,
    xv: GpuTensor,
    xa: GpuTensor,
    xg: GpuTensor,
    prev_x: GpuTensor,
    r: GpuTensor,
    k: GpuTensor,
    v: GpuTensor16,
    v_full: GpuTensor,
    gate: GpuTensor,
    w_full: GpuTensor,
    w_sig: GpuTensor,
    w: GpuTensor16,
    a_full: GpuTensor,
    a: GpuTensor16,
    kk_l2: GpuTensor,
    k_mod: GpuTensor,
    b_vec: GpuTensor,
    y: GpuTensor,
    y_norm: GpuTensor,
    g: GpuTensor16,
    y_g: GpuTensor,
    y_out: GpuTensor,
    ln2: GpuTensor,
    prev_c: GpuTensor,
    xb: GpuTensor,
    v2: GpuTensor,
    x_norm: GpuTensor,
    tmp_c: GpuTensor, // 临时缓冲，用于 in-place 操作中转
    // 其他大小
    v_mid: GpuTensor,         // [V_MID]
    w_mid: GpuTensor,         // [W_MID]
    a_mid: GpuTensor,         // [A_MID]
    g_mid: GpuTensor,         // [G_MID]
    r2: GpuTensor,            // [FFN_HIDDEN]
    logits: GpuTensor,        // [vocab]
    token_argmax: GpuTensor,  // [1] GPU argmax 采样的 token 索引（字节存 uint）
    current_token: GpuTensor, // [1] 当前待 gather 的 token 索引（f32 位模式存 uint，供 gather_row 读取）
}

/// 序列并行工作缓冲区（forward_seq 用）：所有激活均为 [T, C]（token 主序）
pub struct SeqBuffers {
    /// 序列长度 T（缓冲大小固定，T 变化时重建）
    t: usize,
    /// 补齐到 TILE_M=256 倍数的 token 数（GEMM 输出缓冲大小）
    m_pad: usize,
    // [T, C] 大小
    x: GpuTensor,
    ln1: GpuTensor,
    xr: GpuTensor,
    xw: GpuTensor,
    xk: GpuTensor,
    xv: GpuTensor,
    xa: GpuTensor,
    xg: GpuTensor,
    // [M_PAD, C] 大小（tensor-core GEMM 输出）
    r: GpuTensor,
    k: GpuTensor,
    v: GpuTensor,
    v_first: GpuTensor, // [M_PAD, C] 每 token 来自 layer 0 的 v（copy_device 需与 v 同尺寸）
    v_full: GpuTensor,  // [M_PAD, C]（tensor-core GEMM 输出）
    gate: GpuTensor,    // [T, C]
    w_full: GpuTensor,  // [M_PAD, C]（tensor-core GEMM 输出）
    w_sig: GpuTensor,   // [T, C]
    w: GpuTensor,       // [T, C]
    a_full: GpuTensor,  // [M_PAD, C]（tensor-core GEMM 输出）
    a: GpuTensor,       // [T, C]
    kk_l2: GpuTensor,
    k_mod: GpuTensor,
    b_vec: GpuTensor,
    y: GpuTensor,
    y_norm: GpuTensor,
    g: GpuTensor, // [M_PAD, C]（tensor-core GEMM 输出）
    y_g: GpuTensor,
    y_out: GpuTensor,        // [M_PAD, C]
    diag_out_ref: GpuTensor, // [M_PAD, C] gemv 参考输出（仅诊断用）
    ln2: GpuTensor,
    xb: GpuTensor,
    v2: GpuTensor, // [M_PAD, C]
    x_norm: GpuTensor,
    tmp_c: GpuTensor,
    // 低秩中间缓冲 [M_PAD, mid_pad]（tensor-core GEMM 输出，mid_pad 补齐到 64）
    v_mid: GpuTensor,
    w_mid: GpuTensor,
    a_mid: GpuTensor,
    g_mid: GpuTensor,
    r2: GpuTensor,      // [M_PAD, fh]
    head_in: GpuTensor, // [C] 最后 token 的 x_norm 行（head 只算末 token）
    logits: GpuTensor,  // [vocab] 末 token logits
    // fp16 激活（tensor-core GEMM 输入）
    xr16: GpuTensor16, // [M_PAD, C]
    xk16: GpuTensor16,
    xv16: GpuTensor16,
    xw16: GpuTensor16, // 低秩 w/a/g 第一级投影输入
    xa16: GpuTensor16,
    xg16: GpuTensor16,
    y_g16: GpuTensor16,
    xb16: GpuTensor16,
    r2_16: GpuTensor16, // [M_PAD, fh]
    // fp16 低秩中间缓冲（第二级投影的 GEMM 输入）
    v_mid16: GpuTensor16, // [M_PAD, vm_pad]
    w_mid16: GpuTensor16, // [M_PAD, wm_pad]
    a_mid16: GpuTensor16, // [M_PAD, am_pad]
    g_mid16: GpuTensor16, // [M_PAD, gm_pad]
    /// any4→fp16 反量化共享 scratch（方案A prefill）：大小取 6 个 any4 矩阵最大值
    /// [ffn_hidden, C] = 26.2M 元素（52.4MB）。每矩阵 GEMM 前由 dequant 全量覆写，
    /// 顺序复用（barrier 由 record_kernel 读写序保证），替代旧逐层 3.4GB fp16 副本。
    w_scratch: GpuTensor16,
}

impl GpuModel {
    /// 从 safetensors 文件加载模型并上传到 GPU
    pub fn from_safetensors(rt: Runtime, path: &str) -> R<Self> {
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
        // 原 fp16 键可能被 any4 量化键替换，此时从 any4_idx [M, K/2] 推导 M
        let ffn_hidden = match st.tensor("blocks.0.ffn.key.weight") {
            Ok(t) => t.shape()[0],
            Err(_) => st.tensor("blocks.0.ffn.key.weight.any4_idx")?.shape()[0],
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

        // 加载 ln0 并在 CPU 端预计算 emb_ln = ln0(embed)
        // 键名: blocks.0.ln0.weight
        let emb_f32 = tensor_to_f32(&emb);
        let ln0_w = tensor_to_f32(&st.tensor("blocks.0.ln0.weight")?);
        let ln0_b = tensor_to_f32(&st.tensor("blocks.0.ln0.bias")?);
        let emb_ln = layer_norm_rows(&emb_f32, &ln0_w, &ln0_b, n_embd, vocab, LN_EPS);
        // GPU 表存 fp16（省 335MB 显存）；CPU 缓存用同一 f16 舍入值回读 f32，
        // 保证 seq（CPU 上传）与 tok（GPU gather）两条路径输入逐位一致
        let mut emb_ln_t = rt.create_tensor_f16(vocab * n_embd)?;
        rt.upload_f16(&emb_ln_t, &emb_ln)?;
        let emb_ln: Vec<f32> = emb_ln.iter().map(|&v| f16::from_f32(v).to_f32()).collect();

        // ln_out
        let ln_out_w = tensor_to_f32(&st.tensor("ln_out.weight")?);
        let ln_out_b = tensor_to_f32(&st.tensor("ln_out.bias")?);
        let mut ln_out_w_t = rt.create_tensor(n_embd)?;
        rt.upload(&ln_out_w_t, &ln_out_w)?;
        let mut ln_out_b_t = rt.create_tensor(n_embd)?;
        rt.upload(&ln_out_b_t, &ln_out_b)?;

        // head.weight 原始 [vocab, C] = [out, in]，直接用
        // 仅保留 fp16 变体（单 token 用 fp16 gemv 减半带宽）；fp32 副本已删省显存
        let head_w = tensor_to_f32(&st.tensor("head.weight")?);
        let mut head_w16_t = rt.create_tensor_f16(vocab * n_embd)?;
        rt.upload_f16(&head_w16_t, &head_w)?;

        // 加载每一层
        let mut layers = Vec::with_capacity(n_layer);
        for i in 0..n_layer {
            layers.push(GpuLayer::load(&rt, &st, i, n_embd, &config)?);
        }

        // 工作缓冲区
        let bufs = WorkBuffers::new(&rt, n_embd, vocab, &config)?;

        // 状态
        let mut state = Vec::with_capacity(n_layer);
        for _ in 0..n_layer {
            state.push(GpuState::new(&rt, n_embd, n_head, head_size)?);
        }

        // v_first 跨层共享
        let mut v_first = rt.create_tensor_f16(n_embd)?;

        // scale_w: -1/sqrt(e)
        let mut scale_w = rt.create_tensor(1)?;
        rt.upload(&scale_w, &[-1.0f32 / std::f32::consts::E.sqrt()])?;

        // 权重上传完成：释放模型级权重的 host（系统内存）缓冲，仅保留 device 拷贝
        rt.drop_host_f16(&mut emb_ln_t);
        rt.drop_host(&mut ln_out_w_t);
        rt.drop_host(&mut ln_out_b_t);
        rt.drop_host_f16(&mut head_w16_t);
        rt.drop_host_f16(&mut v_first);
        rt.drop_host(&mut scale_w);

        Ok(Self {
            rt,
            config,
            emb_ln: emb_ln_t,
            emb_ln_cpu: emb_ln,
            ln_out_w: ln_out_w_t,
            ln_out_b: ln_out_b_t,
            head_w16: head_w16_t,
            layers,
            bufs,
            seq_bufs: None,
            state,
            v_first,
            v_first_set: false,
            scale_w,
        })
    }

    /// 重置 RNN 状态为零（回到首次推理前的初始状态）
    pub fn reset_state(&mut self) -> R<()> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;
        for s in &self.state {
            s.reset(&self.rt, c, h, n)?;
        }
        self.v_first_set = false;
        Ok(())
    }

    /// 前向推理：返回最后一个 token 的 logits [vocab]
    pub fn forward(&mut self, tokens: &[u32]) -> R<Vec<f32>> {
        for &token in tokens {
            self.forward_token(token, false)?;
        }
        let logits = self.rt.download(&self.bufs.logits)?;
        Ok(logits)
    }

    /// 前向推理 + GPU 采样：返回最后一个 token 的 argmax 索引（全 GPU，不下载 logits）。
    /// 对标 albatross 的 torch.argmax：只把 4 字节的 token 索引回传，省去每 token 下载
    /// 65536 个 f32 logits（256KB）与 CPU 遍历。
    pub fn forward_argmax(&mut self, tokens: &[u32]) -> R<u32> {
        for &token in tokens {
            self.forward_token(token, true)?;
        }
        let t = self.rt.download(&self.bufs.token_argmax)?;
        // shader 向 f32 缓冲写入 uint，回读时按位解释为 u32
        Ok(t[0].to_bits())
    }

    /// GPU self-loop 批量生成：在**单次 submit** 内连续采样 n 个 token。
    /// 首个 token 由 CPU 写 host-visible 缓冲（seed），随后每轮 argmax 结果直接写回
    /// 同一 host 缓冲，下一轮 gather 自动跟随——全程无 CPU 回读/回传 token、
    /// 无每 token 一次 submit+wait，消除 CPU⟷GPU 交换与 dispatch 间同步开销。
    /// 每轮 argmax 的 token 用 record_token 追加到序列缓冲，结束后一次性下载验证。
    /// 返回生成的 n 个 token 索引（按位解释为 u32）。
    pub fn forward_argmax_selfloop(&mut self, seed: u32, n: usize) -> R<Vec<u32>> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let ns = self.config.head_size;
        let vocab = self.config.vocab;

        // 序列缓冲 [n] 与原子计数器 [1]（record_token 用，spec 恒空不重建 pipeline）
        let token_seq = self.rt.create_tensor(n)?;
        let mut seq_cnt = self.rt.create_tensor(1)?;
        self.rt.upload(&token_seq, &vec![0.0; n])?;
        self.rt.upload(&seq_cnt, &[0.0; 1])?;

        // 开启批处理：整段 self-loop 所有 kernel 一次性记录 + 提交
        self.rt.begin_batch()?;

        // 首个 token 由 CPU 写入 host-visible 缓冲
        self.rt.store_token_host(&self.bufs.current_token, seed)?;

        for _ in 0..n {
            // RNN 状态跨 token 保持，仅每个 token 重置 v_first
            self.v_first_set = false;

            // 参数化 gather：读 host 缓冲中的 token 索引 → 取 embedding 行
            self.rt.gather_row_device_f16(
                &self.emb_ln,
                &mut self.bufs.x,
                &self.bufs.current_token,
                c,
            )?;

            for i in 0..self.config.n_layer {
                self.forward_layer(i, c, h, ns)?;
            }

            // ln_out + head
            self.rt.norm(
                &self.bufs.x,
                &self.ln_out_w,
                &self.ln_out_b,
                &mut self.bufs.x_norm,
                c,
                1,
                LN_EPS,
                1,
            )?;
            self.rt.gemv_f16(
                &self.head_w16,
                &self.bufs.x_norm,
                &mut self.bufs.logits,
                vocab,
                c,
                1,
            )?;

            // GPU argmax 直接写回 host-visible current_token（self-loop 关键：下一轮 gather 自动跟随）
            let tok_host = self
                .bufs
                .current_token
                .host
                .as_ref()
                .ok_or("forward_argmax_selfloop: current_token host dropped")?;
            self.rt
                .argmax_into_host(&self.bufs.logits, tok_host, vocab)?;
            // 把本轮 token 追加到序列缓冲（供一次性下载验证）
            self.rt.record_token(tok_host, &token_seq, &mut seq_cnt)?;
        }

        // 一次性提交整段 self-loop
        self.rt.end_batch()?;

        // 下载序列缓冲，按位解释为 u32
        let t = self.rt.download(&token_seq)?;
        Ok(t[..n].iter().map(|x| x.to_bits()).collect())
    }

    /// 单 token 前向：把该 token 的 logits 写入 bufs.logits（含 batch 记录与提交）。
    /// `do_argmax` 为真时，在提交前把 logits 的 GPU argmax 索引写入 bufs.token_argmax
    /// （与本次前向同批记录，避免额外一次 submit）。
    fn forward_token(&mut self, token: u32, do_argmax: bool) -> R<()> {
        let c = self.config.n_embd;
        let h = self.config.n_head;
        let n = self.config.head_size;
        let vocab = self.config.vocab;

        // 每个 token 重置 v_first
        self.v_first_set = false;

        // 开启批处理记录：整层 forward 的所有 kernel + 拷贝一次性记录
        self.rt.begin_batch()?;

        // 参数化 embedding gather：token 索引由 CPU 直接写入 host-visible 缓冲（无 kernel、
        // 无 spec constant，避免每 token 重建 pipeline），再由 gather kernel 从 emb_ln 表
        // 按索引取行 → x。索引来自 host 缓冲，循环体不依赖具体 token 值，
        // 为 GPU self-loop（argmax 直接写索引）铺路。
        self.rt.store_token_host(&self.bufs.current_token, token)?;
        self.rt.gather_row_device_f16(
            &self.emb_ln,
            &mut self.bufs.x,
            &self.bufs.current_token,
            c,
        )?;

        for i in 0..self.config.n_layer {
            self.forward_layer(i, c, h, n)?;
            if let Ok(sl) = std::env::var("SNAP_LAYER")
                && sl.parse::<usize>().unwrap() == i
            {
                self.rt.end_batch()?;
                let xd = self.rt.download(&self.bufs.x)?;
                log::info!("[SNAP] tok  layer {i} x[0..8] = {:?}", &xd[..8]);
                self.rt.begin_batch()?;
            }
        }

        // ln_out + head
        // x_norm = layer_norm(x, ln_out_w, ln_out_b)
        self.rt.norm(
            &self.bufs.x,
            &self.ln_out_w,
            &self.ln_out_b,
            &mut self.bufs.x_norm,
            c,
            1,
            LN_EPS,
            1,
        )?;
        // logits = x_norm @ head.T = gemv_f16(head_w16, x_norm, M=vocab, K=c)
        self.rt.gemv_f16(
            &self.head_w16,
            &self.bufs.x_norm,
            &mut self.bufs.logits,
            vocab,
            c,
            1,
        )?;
        // 需要采样时，GPU 端 argmax 归约（与本次前向同批）
        if do_argmax {
            self.rt
                .argmax(&self.bufs.logits, &mut self.bufs.token_argmax, vocab)?;
        }

        // 一次性提交整 token 的所有计算到 GPU
        self.rt.end_batch()?;
        Ok(())
    }

    /// 单层前向：time mixing + channel mixing
    fn forward_layer(&mut self, i: usize, c: usize, h: usize, n: usize) -> R<()> {
        // decode 走 any4 GEMV，无需 fp16 副本；prefill 已逐层释放临时 fp16，decode 不再持有。
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
                    self.rt.end_batch()?;
                    log::info!(
                        "[P] tok t=0 layer {i} {}: {:.3}ms",
                        $name,
                        t0.elapsed().as_secs_f64() * 1000.0
                    );
                    self.rt.begin_batch()?;
                    t0 = std::time::Instant::now();
                }
            };
        }
        // ===== Time Mixing =====
        // 深度融合：ln1 = layer_norm(x) + 6 次 lerp(xr/xw/xk/xv/xa/xg) + state.tmix_x 写回
        self.rt.norm_lerp6(
            &self.bufs.x,
            &mut self.state[i].tmix_x,
            &self.layers[i].ln1_w,
            &self.layers[i].ln1_b,
            &self.layers[i].x_r,
            &self.layers[i].x_w,
            &self.layers[i].x_k,
            &self.layers[i].x_v,
            &self.layers[i].x_a,
            &self.layers[i].x_g,
            &mut self.bufs.xr,
            &mut self.bufs.xw,
            &mut self.bufs.xk,
            &mut self.bufs.xv,
            &mut self.bufs.xa,
            &mut self.bufs.xg,
            c,
            LN_EPS,
        )?;
        pf_phase!("norm_lerp6");

        // r/k/v + v_mid/w_mid/a_mid/g_mid = 融合 gemv（一次 dispatch 算 r/k/v 三个 C×C 投影
        // + v1/w1/a1/g1 四个 mid 投影）。三路量化权重自动路由：int8 → any4 → fp16
        // （r/k/v 三者同量化格式才走对应融合版；权重带宽 fp16 2B / int8 1B / any4 ~0.5B）。
        if let (Some(r_a8), Some(k_a8), Some(v_a8)) = (
            &self.layers[i].receptance_a8,
            &self.layers[i].key_a8,
            &self.layers[i].value_a8,
        ) {
            self.rt.gemv_int8_rkv_stage1(
                r_a8,
                k_a8,
                v_a8,
                &self.layers[i].v1,
                &self.layers[i].w1,
                &self.layers[i].a1,
                &self.layers[i].g1,
                &self.bufs.xr,
                &self.bufs.xk,
                &self.bufs.xv,
                &self.bufs.xw,
                &self.bufs.xa,
                &self.bufs.xg,
                &mut self.bufs.r,
                &mut self.bufs.k,
                &mut self.bufs.v,
                &mut self.bufs.v_mid,
                &mut self.bufs.w_mid,
                &mut self.bufs.a_mid,
                &mut self.bufs.g_mid,
                c,
                vm,
                wm,
                am,
                gm,
            )?;
        } else if let (Some(r_a4), Some(k_a4), Some(v_a4)) = (
            &self.layers[i].receptance_a4,
            &self.layers[i].key_a4,
            &self.layers[i].value_a4,
        ) {
            self.rt.gemv_any4_rkv_stage1(
                r_a4,
                k_a4,
                v_a4,
                &self.layers[i].v1,
                &self.layers[i].w1,
                &self.layers[i].a1,
                &self.layers[i].g1,
                &self.bufs.xr,
                &self.bufs.xk,
                &self.bufs.xv,
                &self.bufs.xw,
                &self.bufs.xa,
                &self.bufs.xg,
                &mut self.bufs.r,
                &mut self.bufs.k,
                &mut self.bufs.v,
                &mut self.bufs.v_mid,
                &mut self.bufs.w_mid,
                &mut self.bufs.a_mid,
                &mut self.bufs.g_mid,
                c,
                vm,
                wm,
                am,
                gm,
            )?;
        } else {
            self.rt.gemv_rkv_stage1(
                self.layers[i]
                    .receptance_w16
                    .as_ref()
                    .ok_or("receptance_w16 missing when not any4")?,
                self.layers[i]
                    .key_w16
                    .as_ref()
                    .ok_or("key_w16 missing when not any4")?,
                self.layers[i]
                    .value_w16
                    .as_ref()
                    .ok_or("value_w16 missing when not any4")?,
                &self.layers[i].v1,
                &self.layers[i].w1,
                &self.layers[i].a1,
                &self.layers[i].g1,
                &self.bufs.xr,
                &self.bufs.xk,
                &self.bufs.xv,
                &self.bufs.xw,
                &self.bufs.xa,
                &self.bufs.xg,
                &mut self.bufs.r,
                &mut self.bufs.k,
                &mut self.bufs.v,
                &mut self.bufs.v_mid,
                &mut self.bufs.w_mid,
                &mut self.bufs.a_mid,
                &mut self.bufs.g_mid,
                c,
                vm,
                wm,
                am,
                gm,
            )?;
        }
        pf_phase!("gemv_rkv_stage1");
        Self::dump_tensors(
            &mut self.rt,
            "tok",
            i,
            &[
                ("xr", &self.bufs.xr),
                ("xk", &self.bufs.xk),
                ("xv", &self.bufs.xv),
                ("r", &self.bufs.r),
                ("k", &self.bufs.k),
            ],
        )?;

        // ===== v_first 跨层逻辑（首层把 v 快照到 v_first）=====
        if !self.v_first_set {
            self.rt.copy_device_f16(&self.bufs.v, &mut self.v_first)?;
            self.v_first_set = true;
        }
        // ===== 低秩链第二级融合（w/a/g/v 二级投影 + 激活，1 次 dispatch）=====
        // 首层 v_first==v，v 链 lerp 因子 sigmoid*(v_first-v)=0，v 保持不变（=value 投影）
        self.rt.gemv_lowrank_chain4(
            &self.layers[i].w2,
            &self.layers[i].a2,
            &self.layers[i].v2,
            &self.layers[i].g2,
            &self.bufs.w_mid,
            &self.bufs.a_mid,
            &self.bufs.v_mid,
            &self.bufs.g_mid,
            &self.layers[i].w0,
            &self.layers[i].a0,
            &self.layers[i].v0,
            &self.scale_w,
            &self.v_first,
            &mut self.bufs.w,
            &mut self.bufs.a,
            &mut self.bufs.v,
            &mut self.bufs.g,
            c,
            wm,
            am,
            vm,
            gm,
        )?;
        pf_phase!("lowrank_chain4");

        // 融合 kernel（对标 albatross）：fuse_ka + dplr + group_norm + sum_rk_rk 一次 launch
        //   k_mod_i = k_i * (1 + k_a_i * (a_i - 1))
        //   kk_l2_i = normalize(k_i * k_k_i)
        //   b_i     = -kk_l2_i * a_i
        //   S 更新 + y = S @ r
        //   y_norm  = group_norm(y, ln_x_w, ln_x_b) + sum(r*k_mod*r_k)*v
        // （省 1 次 dispatch/层，替代 fuse_ka_dplr + norm_sum_rk_rk 两次）
        self.rt.fuse_ka_dplr_norm(
            &mut self.state[i].tmix_rnn,
            &self.bufs.k,
            &self.layers[i].k_k,
            &self.bufs.a,
            &self.layers[i].k_a,
            &self.bufs.r,
            &self.bufs.v,
            &self.bufs.w,
            &self.layers[i].ln_x_w,
            &self.layers[i].ln_x_b,
            &self.layers[i].r_k,
            &mut self.bufs.k_mod,
            &mut self.bufs.y,
            &mut self.bufs.y_norm,
            h,
            n,
            EPS_L2,
            GN_EPS,
        )?;
        pf_phase!("fuse_ka_dplr_norm");
        Self::dump_tensors(&mut self.rt, "tok", i, &[("y", &self.bufs.y)])?;
        Self::dump_tensors(&mut self.rt, "tok", i, &[("y_norm", &self.bufs.y_norm)])?;

        // x += (y_norm .* g) @ output_w（mul + 残差累加都折叠进 gemv，省 2 次 dispatch）；
        // 三路量化权重自动路由：int8 → any4 → fp16
        if let Some(a8) = &self.layers[i].att_output_a8 {
            self.rt.gemv_int8_mul_add(
                a8,
                &self.bufs.y_norm,
                &self.bufs.g,
                &mut self.bufs.x,
                c,
                c,
                1,
            )?;
        } else if let Some(a4) = &self.layers[i].att_output_a4 {
            self.rt.gemv_any4_mul_add(
                a4,
                &self.bufs.y_norm,
                &self.bufs.g,
                &mut self.bufs.x,
                c,
                c,
                1,
            )?;
        } else {
            self.rt.gemv_f16_mul_add(
                self.layers[i]
                    .output_w16
                    .as_ref()
                    .ok_or("output_w16 missing when not any4")?,
                &self.bufs.y_norm,
                &self.bufs.g,
                &mut self.bufs.x,
                c,
                c,
                1,
            )?;
        }
        pf_phase!("gemv_f16_mul_add");
        Self::dump_tensors(
            &mut self.rt,
            "tok",
            i,
            &[("y_g", &self.bufs.y_g), ("x_out", &self.bufs.x)],
        )?;

        // ===== Channel Mixing =====
        // 深度融合：ln2 = layer_norm(x, ln2_w, ln2_b) + prev_c 读入 + state.cmix_x 写回 + lerp(xb)
        // xb = ln2 + ffn_x_k * (prev_c - ln2)，一次 dispatch 完成（原 4 跳）。
        self.rt.cmix_norm_lerp(
            &self.bufs.x,
            &mut self.state[i].cmix_x,
            &self.layers[i].ln2_w,
            &self.layers[i].ln2_b,
            &self.layers[i].ffn_x_k,
            &mut self.bufs.xb,
            c,
            LN_EPS,
        )?;
        pf_phase!("cmix_norm_lerp");

        // r2 = relu²(xb @ ffn_key.T) — M=ffn_hidden, K=C；三路量化权重自动路由：int8 → any4 → fp16
        if let Some(a8) = &self.layers[i].ffn_key_a8 {
            self.rt
                .gemv_int8_relu2(a8, &self.bufs.xb, &mut self.bufs.r2, fh, c, 1)?;
        } else if let Some(a4) = &self.layers[i].ffn_key_a4 {
            self.rt
                .gemv_any4_relu2(a4, &self.bufs.xb, &mut self.bufs.r2, fh, c, 1)?;
        } else {
            self.rt.gemv_f16_relu2(
                self.layers[i]
                    .ffn_key_w16
                    .as_ref()
                    .ok_or("ffn_key_w16 missing when not any4")?,
                &self.bufs.xb,
                &mut self.bufs.r2,
                fh,
                c,
                1,
            )?;
        }
        pf_phase!("gemv_f16_relu2");
        // x += r2 @ ffn_value（残差累加折叠进 gemv，省 1 次 dispatch）；
        // 三路量化权重自动路由：int8 → any4 → fp16
        if let Some(a8) = &self.layers[i].ffn_value_a8 {
            self.rt
                .gemv_int8_add(a8, &self.bufs.r2, &mut self.bufs.x, c, fh, 1)?;
        } else if let Some(a4) = &self.layers[i].ffn_value_a4 {
            self.rt
                .gemv_any4_add(a4, &self.bufs.r2, &mut self.bufs.x, c, fh, 1)?;
        } else {
            self.rt.gemv_f16_add(
                self.layers[i]
                    .ffn_value_w16
                    .as_ref()
                    .ok_or("ffn_value_w16 missing when not any4")?,
                &self.bufs.r2,
                &mut self.bufs.x,
                c,
                fh,
                1,
            )?;
        }
        pf_phase!("gemv_f16_add");
        let _ = t0; // 引用最后一次赋值，避免 unused_assignments 警告

        Ok(())
    }

    /// 前向推理（sequence-parallel）：把整段 T 个 token 一次贯穿各层，返回最后 token 的 logits [vocab]。
    /// 对标 albatross forward_seq：线性投影用批量 GEMM（token 并行），WKV 顺序更新用单次 launch 的
    /// dplr_seq（内部循环 T），最大程度减少逐 token 的 dispatch 开销。
    pub fn forward_seq(&mut self, tokens: &[u32]) -> R<Vec<f32>> {
        let t = tokens.len();
        assert!(t >= 1, "forward_seq requires at least 1 token");
        let (c, n, vocab) = (self.config.n_embd, self.config.head_size, self.config.vocab);

        // 按需创建/复用序列并行缓冲（T 变化时重建）
        if self.seq_bufs.as_ref().is_none_or(|sb| sb.t != t) {
            // T 变化 → 新缓冲地址 + 新 spec，旧 kernel 缓存失效，清空避免耗尽 descriptor pool
            self.rt.clear_cache();
            self.seq_bufs = Some(SeqBuffers::new(&self.rt, t, c, vocab, &self.config)?);
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
        self.rt.upload(&sb.x, &x_data)?;

        let seq_pf = std::env::var("SEQ_PROFILE").is_ok();
        let mut spf_t0 = std::time::Instant::now();
        // 单批处理全部层：任何层都走 any4 GEMM（零 fp16 副本），无需逐层 begin/end_batch 隔离。
        self.rt.begin_batch()?;
        for i in 0..self.config.n_layer {
            if std::env::var("GEMM_DIAG").is_ok() {
                log::info!("[FS] layer {i} start");
            }
            self.forward_seq_layer(i, t, c, n, sb)?;
            if std::env::var("GEMM_DIAG").is_ok() {
                log::info!("[FS] layer {i} done");
            }
            if seq_pf {
                log::info!("[SP] layer {i}: {:.4}s", spf_t0.elapsed().as_secs_f64());
                spf_t0 = std::time::Instant::now();
            }
        }
        self.rt.end_batch()?;

        // ln_out + head（只算最后 token，避免 [T, vocab] 全量 GEMM 与 67MB 下载）
        self.rt.begin_batch()?;
        // x_norm = layer_norm(x, ln_out_w, ln_out_b) over T 个 token
        self.rt.norm(
            &sb.x,
            &self.ln_out_w,
            &self.ln_out_b,
            &mut sb.x_norm,
            c,
            1,
            LN_EPS,
            t,
        )?;
        // head_in = x_norm[T-1]（末 token 行）
        self.rt
            .copy_token(&sb.x_norm, &mut sb.head_in, c, c, t - 1)?;
        // logits = head_in @ head.T，输出 [vocab]（fp16 权重减半带宽）
        self.rt
            .gemv_f16(&self.head_w16, &sb.head_in, &mut sb.logits, vocab, c, 1)?;

        self.rt.end_batch()?;

        // 诊断：比较最后层 output GEMM vs gemv 参考
        if std::env::var("GEMM_DIAG").is_ok() {
            let go = self.rt.download(&sb.y_out)?;
            let gr = self.rt.download(&sb.diag_out_ref)?;
            let mut md = 0.0f32;
            for (a, b) in go.iter().zip(&gr) {
                md = md.max((a - b).abs());
            }
            log::info!("[FS] last-layer output GEMM vs gemv max_abs_diff: {md:.6}");
            // 诊断：to_f16 是否正确（y_g16 vs fp16(y_g)）
            let yg = self.rt.download(&sb.y_g)?;
            let yg16 = self.rt.download_f16(&sb.y_g16)?;
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
        let last_logits = self.rt.download(&sb.logits)?;

        // 归还 seq_bufs
        self.seq_bufs = Some(seq_bufs);

        Ok(last_logits)
    }

    /// 诊断：逐层快照 x（前 D 维），用于定位 seq 路径两次运行的首发发散层。
    /// 返回 Vec<Vec<f32>>，第 i 项为第 i+1 层输出 x 的快照（i==0 为 layer 0 之后）。
    pub fn snapshot_seq_layers(&mut self, tokens: &[u32], d: usize) -> R<Vec<Vec<f32>>> {
        let t = tokens.len();
        let (c, n, vocab) = (self.config.n_embd, self.config.head_size, self.config.vocab);
        self.reset_state()?;
        self.rt.clear_cache();
        if self.seq_bufs.as_ref().is_none_or(|sb| sb.t != t) {
            self.rt.clear_cache();
            self.seq_bufs = Some(SeqBuffers::new(&self.rt, t, c, vocab, &self.config)?);
        }
        let mut seq_bufs = self.seq_bufs.take().unwrap();
        let sb = &mut seq_bufs;
        let emb_ln = &self.emb_ln_cpu;
        let mut x_data = vec![0.0; t * c];
        for (ti, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            x_data[ti * c..(ti + 1) * c].copy_from_slice(&emb_ln[tok * c..(tok + 1) * c]);
        }
        self.rt.upload(&sb.x, &x_data)?;
        let mut snaps = Vec::new();
        // 与 forward_seq 一致：any4 GEMM 零 fp16 副本，逐层独立 batch 以便下载快照。
        for i in 0..self.config.n_layer {
            self.rt.begin_batch()?;
            self.forward_seq_layer(i, t, c, n, sb)?;
            self.rt.end_batch()?;
            let xd = self.rt.download(&sb.x)?;
            snaps.push(xd[..d].to_vec());
        }
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
        self.rt.upload(&self.bufs.x, &x_host)?;
        // 与 forward 一致：逐层前向并快照
        for i in 0..self.config.n_layer {
            self.v_first_set = false;
            self.rt.begin_batch()?;
            self.forward_layer(i, c, h, n)?;
            self.rt.end_batch()?;
            x_host = self.rt.download(&self.bufs.x)?;
            snaps.push(x_host[..d].to_vec());
        }
        Ok(snaps)
    }

    /// 诊断：下载 layer idx 的 tmix_rnn 状态的前 n 个元素（用于对比 seq 与 tok 路径 dplr 状态差异）。
    pub fn download_state_rnn(&mut self, idx: usize, n: usize) -> R<Vec<f32>> {
        let s = self.rt.download(&self.state[idx].tmix_rnn)?;
        Ok(s[..n.min(s.len())].to_vec())
    }

    pub fn layers_len(&self) -> usize {
        self.layers.len()
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
        self.rt.upload(&self.bufs.x, &x_host)?;
        for i in 0..up_to_layer {
            self.v_first_set = false;
            self.rt.begin_batch()?;
            self.forward_layer(i, c, h, n_head)?;
            self.rt.end_batch()?;
        }
        // 下载输入 x（layer up_to_layer 的输入）
        let x_inp = self.rt.download(&self.bufs.x)?;
        // 计算 ln1
        self.rt.begin_batch()?;
        let li = up_to_layer.min(self.layers.len() - 1);
        let l = &self.layers[li];
        self.rt.norm(
            &self.bufs.x,
            &l.ln1_w,
            &l.ln1_b,
            &mut self.bufs.ln1,
            c,
            1,
            LN_EPS,
            1,
        )?;
        self.rt.end_batch()?;
        let ln = self.rt.download(&self.bufs.ln1)?;
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
        self.rt.clear_cache();
        if self.seq_bufs.as_ref().is_none_or(|sb| sb.t != t) {
            self.rt.clear_cache();
            self.seq_bufs = Some(SeqBuffers::new(&self.rt, t, c, vocab, &self.config)?);
        }
        let mut seq_bufs = self.seq_bufs.take().unwrap();
        let sb = &mut seq_bufs;
        let emb_ln = &self.emb_ln_cpu;
        let mut x_data = vec![0.0; t * c];
        for (ti, &tok) in tokens.iter().enumerate() {
            let tok = tok as usize;
            x_data[ti * c..(ti + 1) * c].copy_from_slice(&emb_ln[tok * c..(tok + 1) * c]);
        }
        self.rt.upload(&sb.x, &x_data)?;
        // 跑 [0, up_to_layer) 层
        for i in 0..up_to_layer {
            self.rt.begin_batch()?;
            self.forward_seq_layer(i, t, c, n_head, sb)?;
            self.rt.end_batch()?;
        }
        // 下载输入 x（第 0 token：sb.x[0..c]，因为 mk_pad(c) 布局 [M_PAD, C] 第 0 行偏移 = 0 * C = 0）
        let x_all = self.rt.download(&sb.x)?;
        let x_inp = x_all[..c.min(x_all.len())].to_vec();
        // 跑 norm
        self.rt.begin_batch()?;
        let li = up_to_layer.min(self.layers.len() - 1);
        let l = &self.layers[li];
        self.rt
            .norm(&sb.x, &l.ln1_w, &l.ln1_b, &mut sb.ln1, c, 1, LN_EPS, t)?;
        self.rt.end_batch()?;
        let xn = self.rt.download(&sb.ln1)?;
        let nn = n.min(xn.len()).min(x_inp.len());
        self.seq_bufs = Some(seq_bufs);
        Ok((x_inp[..nn].to_vec(), xn[..nn].to_vec()))
    }

    /// 诊断：dump 指定层的中间张量（需在批处理上下文中，吞吐低，仅调试用）。
    fn dump_tensors(rt: &mut Runtime, tag: &str, i: usize, items: &[(&str, &GpuTensor)]) -> R<()> {
        let want = std::env::var("DUMP_LAYER")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        if want == Some(i) {
            rt.end_batch()?;
            for (name, t) in items {
                let d = rt.download(t)?;
                let n = d.len().min(8);
                log::info!("[DBG] {tag} layer {i} {name} [0..{n}] = {:?}", &d[..n]);
            }
            rt.begin_batch()?;
        }
        Ok(())
    }

    /// 诊断：校验 GPU dequant_any4_to_f16 输出与 CPU dequant_any4 参考一致
    ///（GEMM_DIAG_VERIFY=层号 时对该层 receptance 触发，kernel 单测）。
    /// 下载 any4 idx/lut/sz 与 w_scratch，CPU 端按同一公式（含 fp16 舍入）重算并比对。
    fn verify_dequant(
        rt: &mut Runtime,
        a4: &GpuTensorAny4,
        scratch: &GpuTensor16,
        i: usize,
        m: usize,
        k: usize,
    ) -> R<()> {
        let want = std::env::var("GEMM_DIAG_VERIFY")
            .ok()
            .and_then(|s| s.parse::<usize>().ok());
        if want != Some(i) {
            return Ok(());
        }
        rt.end_batch()?;
        let idx = rt.download_u32(&a4.idx, m * (k / 8))?;
        let lut = rt.download_f16(&a4.lut)?;
        let sz = rt.download_u32(&a4.sz, m * (k / 128))?;
        let gpu = rt.download_f16(scratch)?;
        rt.begin_batch()?;
        let w_ref = dequant_any4(&idx, &lut, &sz, m, k);
        let mut max_diff = 0.0f32;
        let mut bad = 0usize;
        for (e, &wr) in w_ref.iter().enumerate() {
            let d = (gpu[e] - f16::from_f32(wr).to_f32()).abs();
            if d > max_diff {
                max_diff = d;
            }
            if d > 1e-3 {
                bad += 1;
            }
        }
        log::info!(
            "[DEQ] layer {i} dequant scratch vs cpu: max_diff={max_diff:.6} bad(>1e-3)={bad}/{}",
            m * k
        );
        Ok(())
    }

    /// 诊断：用 CPU fp16 参考验证 GPU GEMM 的 r/k/v 输出（GEMM_DIAG_VERIFY=层号时触发）。
    /// 下载 xk16/xv16 与 key_w16/value_w16，计算 k[0]/v[0] 的 fp16 精确参考，与 GPU 输出对比。
    fn verify_gemm_rkv(
        rt: &mut Runtime,
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
        rt.end_batch()?;
        // any4 模型无常驻 w16（方案A 下参考为 w_scratch 反量化结果，此处直接跳过）
        let (Some(key_w16), Some(value_w16)) = (&layer.key_w16, &layer.value_w16) else {
            log::info!("[GEMMV] layer {i} skipped: any4 模型无常驻 w16 参考");
            rt.begin_batch()?;
            return Ok(());
        };
        if std::env::var("ADDR_DIAG").is_ok() {
            log::info!(
                "[ADDR] layer{i} key_w16 addr={:#x} len={} | value_w16 addr={:#x} len={}",
                key_w16.device.address,
                key_w16.len,
                value_w16.device.address,
                value_w16.len
            );
            log::info!(
                "[ADDR] layer{i} r addr={:#x} len={} | k addr={:#x} len={} | v addr={:#x} len={}",
                sb.r.device.address,
                sb.r.len,
                sb.k.device.address,
                sb.k.len,
                sb.v.device.address,
                sb.v.len
            );
            log::info!(
                "[ADDR] layer{i} key_w16[0..3]={:?}",
                rt.download_f16(key_w16)
                    .map(|d| d[..3].to_vec())
                    .unwrap_or_default()
            );
        }
        let xk16 = rt.download_f16(&sb.xk16)?;
        let xv16 = rt.download_f16(&sb.xv16)?;
        let xk = rt.download(&sb.xk)?;
        let kw16 = rt.download_f16(key_w16)?;
        let vw16 = rt.download_f16(value_w16)?;
        let gk = rt.download(&sb.k)?;
        let gv = rt.download(&sb.v)?;
        rt.begin_batch()?;

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
                    self.rt.end_batch()?;
                    log::info!(
                        "[P] t={t} layer {i} {}: {:.4}s",
                        $name,
                        t0.elapsed().as_secs_f64()
                    );
                    self.rt.begin_batch()?;
                    t0 = std::time::Instant::now();
                }
            };
        }

        // ===== Time Mixing =====
        // ln1 = layer_norm(x, ln1_w, ln1_b) over T 个 token
        self.rt.norm(
            &sb.x,
            &layer.ln1_w,
            &layer.ln1_b,
            &mut sb.ln1,
            c,
            1,
            LN_EPS,
            t,
        )?;

        // token shift + time-mix：xr/xw/xk/xv/xa/xg（t=0 用旧 state，t>0 用前一 token）
        self.rt.seq_shift(
            &sb.ln1,
            &self.state[i].tmix_x,
            &layer.x_r,
            &mut sb.xr,
            c,
            t,
            c,
            c,
        )?;
        self.rt.seq_shift(
            &sb.ln1,
            &self.state[i].tmix_x,
            &layer.x_w,
            &mut sb.xw,
            c,
            t,
            c,
            c,
        )?;
        self.rt.seq_shift(
            &sb.ln1,
            &self.state[i].tmix_x,
            &layer.x_k,
            &mut sb.xk,
            c,
            t,
            c,
            c,
        )?;
        self.rt.seq_shift(
            &sb.ln1,
            &self.state[i].tmix_x,
            &layer.x_v,
            &mut sb.xv,
            c,
            t,
            c,
            c,
        )?;
        self.rt.seq_shift(
            &sb.ln1,
            &self.state[i].tmix_x,
            &layer.x_a,
            &mut sb.xa,
            c,
            t,
            c,
            c,
        )?;
        self.rt.seq_shift(
            &sb.ln1,
            &self.state[i].tmix_x,
            &layer.x_g,
            &mut sb.xg,
            c,
            t,
            c,
            c,
        )?;

        // state[i].tmix_x = ln1[T-1]（须在 seq_shift 之后，避免覆盖 t=0 读取的旧 state）
        self.rt
            .copy_token(&sb.ln1, &mut self.state[i].tmix_x, c, c, t - 1)?;

        // r/k/v = token 并行 GEMM。
        // 方案A（默认）：any4 权重 dequant 到共享 w_scratch，走 fp16 tensor-core GEMM；
        // ANY4_GEMM_PREFILL=1：保留的标量 any4 GEMM 路径（显存极限场景备用/方案C基线，慢 ~5.7×）；
        // 非 any4 模型：直接用常驻 fp16 w16。
        let m_pad = sb.m_pad;
        let any4_gemm_prefill = std::env::var("ANY4_GEMM_PREFILL").is_ok();
        if std::env::var("GEMM_DIAG").is_ok() {
            log::info!("[FS] layer {i} gemm rkv start");
        }
        if let (Some(ra8), Some(ka8), Some(va8)) =
            (&layer.receptance_a8, &layer.key_a8, &layer.value_a8)
        {
            self.rt.to_f16_triple(
                &sb.xr,
                &sb.xk,
                &sb.xv,
                &mut sb.xr16,
                &mut sb.xk16,
                &mut sb.xv16,
                c,
                t,
                m_pad,
                c,
                c,
            )?;
            self.rt.dequant_int8_to_f16(ra8, &sb.w_scratch, c, c)?;
            self.rt
                .gemm(&sb.xr16, &sb.w_scratch, &mut sb.r, m_pad, c, c)?;
            self.rt.dequant_int8_to_f16(ka8, &sb.w_scratch, c, c)?;
            self.rt
                .gemm(&sb.xk16, &sb.w_scratch, &mut sb.k, m_pad, c, c)?;
            self.rt.dequant_int8_to_f16(va8, &sb.w_scratch, c, c)?;
            self.rt
                .gemm(&sb.xv16, &sb.w_scratch, &mut sb.v, m_pad, c, c)?;
        } else if let (Some(ra4), Some(ka4), Some(va4)) =
            (&layer.receptance_a4, &layer.key_a4, &layer.value_a4)
        {
            if any4_gemm_prefill {
                self.rt
                    .gemm_any4(ra4, &sb.xr, None, &mut sb.r, c, c, t, false)?;
                self.rt
                    .gemm_any4(ka4, &sb.xk, None, &mut sb.k, c, c, t, false)?;
                self.rt
                    .gemm_any4(va4, &sb.xv, None, &mut sb.v, c, c, t, false)?;
            } else {
                self.rt.to_f16_triple(
                    &sb.xr,
                    &sb.xk,
                    &sb.xv,
                    &mut sb.xr16,
                    &mut sb.xk16,
                    &mut sb.xv16,
                    c,
                    t,
                    m_pad,
                    c,
                    c,
                )?;
                self.rt.dequant_any4_to_f16(ra4, &sb.w_scratch, c, c)?;
                Self::verify_dequant(&mut self.rt, ra4, &sb.w_scratch, i, c, c)?;
                self.rt
                    .gemm(&sb.xr16, &sb.w_scratch, &mut sb.r, m_pad, c, c)?;
                self.rt.dequant_any4_to_f16(ka4, &sb.w_scratch, c, c)?;
                self.rt
                    .gemm(&sb.xk16, &sb.w_scratch, &mut sb.k, m_pad, c, c)?;
                self.rt.dequant_any4_to_f16(va4, &sb.w_scratch, c, c)?;
                self.rt
                    .gemm(&sb.xv16, &sb.w_scratch, &mut sb.v, m_pad, c, c)?;
            }
        } else {
            self.rt.to_f16_triple(
                &sb.xr,
                &sb.xk,
                &sb.xv,
                &mut sb.xr16,
                &mut sb.xk16,
                &mut sb.xv16,
                c,
                t,
                m_pad,
                c,
                c,
            )?;
            let r16 = layer
                .receptance_w16
                .as_ref()
                .ok_or("receptance_w16 missing")?;
            let k16 = layer.key_w16.as_ref().ok_or("key_w16 missing")?;
            let v16 = layer.value_w16.as_ref().ok_or("value_w16 missing")?;
            self.rt.gemm(&sb.xr16, r16, &mut sb.r, m_pad, c, c)?;
            self.rt.gemm(&sb.xk16, k16, &mut sb.k, m_pad, c, c)?;
            self.rt.gemm(&sb.xv16, v16, &mut sb.v, m_pad, c, c)?;
        }
        if std::env::var("GEMM_DIAG").is_ok() {
            log::info!("[FS] layer {i} gemm rkv done");
        }
        pf_phase!("shift+rkv_gemm");
        Self::dump_tensors(
            &mut self.rt,
            "seq",
            i,
            &[
                ("xr", &sb.xr),
                ("xk", &sb.xk),
                ("xv", &sb.xv),
                ("r", &sb.r),
                ("k", &sb.k),
                ("v", &sb.v),
            ],
        )?;
        Self::verify_gemm_rkv(&mut self.rt, layer, i, sb, c)?;

        // 低秩 w/a/g 第一级投影输入转 fp16（xw/xa/xg → [M_PAD, C]）
        self.rt.to_f16_triple(
            &sb.xw,
            &sb.xa,
            &sb.xg,
            &mut sb.xw16,
            &mut sb.xa16,
            &mut sb.xg16,
            c,
            t,
            m_pad,
            c,
            c,
        )?;

        // v_first 逻辑：layer 0 存 v_first = v；layer>0 交叉混合 v
        if i == 0 {
            self.rt.copy_device(&sb.v, &mut sb.v_first)?;
        } else {
            // v_mid = tensor-core GEMM(xv @ v1)：[M_PAD, vm_pad]
            self.rt
                .gemm(&sb.xv16, &layer.v1_16, &mut sb.v_mid, m_pad, vvp, c)?;
            // v_mid → fp16（第二级投影输入）
            self.rt
                .to_f16(&sb.v_mid, &mut sb.v_mid16, vvp, t, m_pad, vvp, vvp)?;
            // v_full = gemm_bias(v_mid @ v2 + v0)：[M_PAD, C]
            self.rt.gemm_bias(
                &sb.v_mid16,
                &layer.v2_16,
                &layer.v0,
                &mut sb.v_full,
                m_pad,
                c,
                vvp,
            )?;
            // gate = sigmoid(v_full)
            self.rt
                .elementwise_sigmoid(&sb.v_full, &mut sb.gate, c, t)?;
            // v = v + gate*(v_first - v)（原地，含 t=0）
            self.rt
                .v_first_lerp(&sb.v, &sb.gate, &sb.v_first, c, t, c)?;
        }

        // w = exp(-sigmoid(w0 + tanh(xw@w1)@w2)/sqrt(e))
        // w_mid = tanh(xw @ w1)：[M_PAD, wm_pad]
        self.rt
            .gemm_tanh(&sb.xw16, &layer.w1_16, &mut sb.w_mid, m_pad, wwp, c)?;
        self.rt
            .to_f16(&sb.w_mid, &mut sb.w_mid16, wwp, t, m_pad, wwp, wwp)?;
        // w_full = gemm_bias(w_mid @ w2 + w0)：[M_PAD, C]
        self.rt.gemm_bias(
            &sb.w_mid16,
            &layer.w2_16,
            &layer.w0,
            &mut sb.w_full,
            m_pad,
            c,
            wwp,
        )?;
        self.rt
            .elementwise_sigmoid(&sb.w_full, &mut sb.w_sig, c, t)?;
        self.rt
            .elementwise_scale_exp(&sb.w_sig, &self.scale_w, &mut sb.w, c, t)?;

        // a = sigmoid(a0 + xa@a1@a2)
        // a_mid = xa @ a1：[M_PAD, am_pad]
        self.rt
            .gemm(&sb.xa16, &layer.a1_16, &mut sb.a_mid, m_pad, aap, c)?;
        self.rt
            .to_f16(&sb.a_mid, &mut sb.a_mid16, aap, t, m_pad, aap, aap)?;
        // a_full = gemm_bias(a_mid @ a2 + a0)：[M_PAD, C]
        self.rt.gemm_bias(
            &sb.a_mid16,
            &layer.a2_16,
            &layer.a0,
            &mut sb.a_full,
            m_pad,
            c,
            aap,
        )?;
        self.rt.elementwise_sigmoid(&sb.a_full, &mut sb.a, c, t)?;

        // 融合 k/k_a：k_mod、kk_l2、b_vec 一次 launch
        self.rt.fuse_ka(
            &sb.k,
            &layer.k_k,
            &sb.a,
            &layer.k_a,
            &mut sb.k_mod,
            &mut sb.kk_l2,
            &mut sb.b_vec,
            self.config.n_head,
            n,
            t,
        )?;
        pf_phase!("lowrank+fuse_ka");

        // DPLR：单次 launch 处理整段 T（内部循环），S 跨 token 传递
        self.rt.dplr_seq(
            &mut self.state[i].tmix_rnn,
            &sb.r,
            &sb.w,
            &sb.k_mod,
            &sb.v,
            &sb.kk_l2,
            &sb.b_vec,
            &mut sb.y,
            self.config.n_head,
            n,
            t,
            c,
        )?;
        pf_phase!("lowrank+dplr_seq");
        Self::dump_tensors(
            &mut self.rt,
            "seq",
            i,
            &[("y", &sb.y), ("y_norm", &sb.y_norm)],
        )?;

        // y_norm = group_norm(y, ln_x_w, ln_x_b)
        self.rt.norm(
            &sb.y,
            &layer.ln_x_w,
            &layer.ln_x_b,
            &mut sb.y_norm,
            n,
            self.config.n_head,
            GN_EPS,
            t,
        )?;

        // extra: y_norm += sum(r*k_mod*r_k, head) * v
        self.rt.sum_rk_rk(
            &sb.r,
            &sb.k_mod,
            &layer.r_k,
            &sb.v,
            &mut sb.y_norm,
            self.config.n_head,
            n,
            t,
        )?;

        // g = sigmoid(xg@g1)@g2（tensor-core GEMM）
        // g_mid = xg @ g1：[M_PAD, gm_pad]
        self.rt
            .gemm(&sb.xg16, &layer.g1_16, &mut sb.g_mid, m_pad, ggp, c)?;
        // sigmoid 原地 + 转 fp16（第二级投影输入）
        self.rt.elementwise_sigmoid_inplace(&mut sb.g_mid, ggp, t)?;
        self.rt
            .to_f16(&sb.g_mid, &mut sb.g_mid16, ggp, t, m_pad, ggp, ggp)?;
        // g = g_mid @ g2：[M_PAD, C]
        self.rt
            .gemm(&sb.g_mid16, &layer.g2_16, &mut sb.g, m_pad, c, ggp)?;
        Self::dump_tensors(
            &mut self.rt,
            "seq",
            i,
            &[("w", &sb.w), ("a", &sb.a), ("v", &sb.v), ("g", &sb.g)],
        )?;

        // y_g = y_norm * g
        self.rt
            .elementwise_mul(&sb.y_norm, &sb.g, &mut sb.y_g, c, t)?;

        // y_out = GEMM(output_w, y_g) + x（融合残差相加）；int8/any4 默认走方案A（dequant→scratch→TC GEMM）
        if let Some(a8) = &layer.att_output_a8 {
            self.rt.to_f16(&sb.y_g, &mut sb.y_g16, c, t, m_pad, c, c)?;
            self.rt.dequant_int8_to_f16(a8, &sb.w_scratch, c, c)?;
            self.rt
                .gemm_add(&sb.y_g16, &sb.w_scratch, &sb.x, &mut sb.y_out, m_pad, c, c)?;
        } else if let Some(a4) = &layer.att_output_a4 {
            if any4_gemm_prefill {
                self.rt
                    .gemm_any4(a4, &sb.y_g, Some(&sb.x), &mut sb.y_out, c, c, t, false)?;
            } else {
                self.rt.to_f16(&sb.y_g, &mut sb.y_g16, c, t, m_pad, c, c)?;
                self.rt.dequant_any4_to_f16(a4, &sb.w_scratch, c, c)?;
                self.rt
                    .gemm_add(&sb.y_g16, &sb.w_scratch, &sb.x, &mut sb.y_out, m_pad, c, c)?;
            }
        } else {
            self.rt.to_f16(&sb.y_g, &mut sb.y_g16, c, t, m_pad, c, c)?;
            self.rt.gemm_add(
                &sb.y_g16,
                layer.output_w16.as_ref().ok_or("output_w16 missing")?,
                &sb.x,
                &mut sb.y_out,
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
            self.rt
                .gemv_seq(output_w, &sb.y_g, &mut sb.diag_out_ref, c, c, c, c, t)?;
        }

        pf_phase!("y_norm+sum+g+output");
        Self::dump_tensors(
            &mut self.rt,
            "seq",
            i,
            &[("g", &sb.g), ("y_g", &sb.y_g), ("x_out", &sb.x)],
        )?;

        // ===== Channel Mixing =====
        // ln2 = layer_norm(x, ln2_w, ln2_b)
        self.rt.norm(
            &sb.x,
            &layer.ln2_w,
            &layer.ln2_b,
            &mut sb.ln2,
            c,
            1,
            LN_EPS,
            t,
        )?;

        // xb = token shift（t=0 用旧 cmix_x state，t>0 用前一轮 ln2）
        self.rt.seq_shift(
            &sb.ln2,
            &self.state[i].cmix_x,
            &layer.ffn_x_k,
            &mut sb.xb,
            c,
            t,
            c,
            c,
        )?;
        // state[i].cmix_x = ln2[T-1]（须在 seq_shift 之后）
        self.rt
            .copy_token(&sb.ln2, &mut self.state[i].cmix_x, c, c, t - 1)?;

        // FFN = token 并行 GEMM；any4 默认走方案A（dequant→scratch→TC GEMM）
        if std::env::var("GEMM_DIAG").is_ok() {
            log::info!("[FS] layer {i} ffn start");
        }
        if let Some(fka8) = &layer.ffn_key_a8 {
            self.rt.to_f16(&sb.xb, &mut sb.xb16, c, t, m_pad, c, c)?;
            self.rt.dequant_int8_to_f16(fka8, &sb.w_scratch, fh, c)?;
            self.rt
                .gemm_relu2(&sb.xb16, &sb.w_scratch, &mut sb.r2, m_pad, fh, c)?;
        } else if let Some(fka4) = &layer.ffn_key_a4 {
            if any4_gemm_prefill {
                self.rt
                    .gemm_any4(fka4, &sb.xb, None, &mut sb.r2, fh, c, t, true)?;
            } else {
                self.rt.to_f16(&sb.xb, &mut sb.xb16, c, t, m_pad, c, c)?;
                self.rt.dequant_any4_to_f16(fka4, &sb.w_scratch, fh, c)?;
                self.rt
                    .gemm_relu2(&sb.xb16, &sb.w_scratch, &mut sb.r2, m_pad, fh, c)?;
            }
        } else {
            self.rt.to_f16(&sb.xb, &mut sb.xb16, c, t, m_pad, c, c)?;
            let fk16 = layer.ffn_key_w16.as_ref().ok_or("ffn_key_w16 missing")?;
            self.rt
                .gemm_relu2(&sb.xb16, fk16, &mut sb.r2, m_pad, fh, c)?;
        }
        if let Some(fva8) = &layer.ffn_value_a8 {
            self.rt
                .to_f16(&sb.r2, &mut sb.r2_16, fh, t, m_pad, fh, fh)?;
            self.rt.dequant_int8_to_f16(fva8, &sb.w_scratch, c, fh)?;
            self.rt
                .gemm_add(&sb.r2_16, &sb.w_scratch, &sb.x, &mut sb.v2, m_pad, c, fh)?;
        } else if let Some(fva4) = &layer.ffn_value_a4 {
            if any4_gemm_prefill {
                self.rt
                    .gemm_any4(fva4, &sb.r2, Some(&sb.x), &mut sb.v2, c, fh, t, false)?;
            } else {
                self.rt
                    .to_f16(&sb.r2, &mut sb.r2_16, fh, t, m_pad, fh, fh)?;
                self.rt.dequant_any4_to_f16(fva4, &sb.w_scratch, c, fh)?;
                self.rt
                    .gemm_add(&sb.r2_16, &sb.w_scratch, &sb.x, &mut sb.v2, m_pad, c, fh)?;
            }
        } else {
            self.rt
                .to_f16(&sb.r2, &mut sb.r2_16, fh, t, m_pad, fh, fh)?;
            let fv16 = layer
                .ffn_value_w16
                .as_ref()
                .ok_or("ffn_value_w16 missing")?;
            self.rt
                .gemm_add(&sb.r2_16, fv16, &sb.x, &mut sb.v2, m_pad, c, fh)?;
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
        rt: &Runtime,
        st: &safetensors::SafeTensors,
        idx: usize,
        c: usize,
        cfg: &ModelConfig,
    ) -> R<Self> {
        // 一维参数
        let load1 = |name: &str| -> R<GpuTensor> {
            let key = format!("blocks.{idx}.{name}");
            let data = tensor_to_f32(&st.tensor(&key)?);
            let t = rt.create_tensor(data.len())?;
            rt.upload(&t, &data)?;
            Ok(t)
        };
        // 线性权重：只保留 output_w fp32 用于 GEMM_DIAG 诊断，其余已删 fp32 副本省显存
        let load_linear_diag = |key: String| -> R<GpuTensor> {
            let data = tensor_to_f32(&st.tensor(&key)?);
            let t = rt.create_tensor(data.len())?;
            rt.upload(&t, &data)?;
            Ok(t)
        };
        // 线性权重 → fp16（tensor-core GEMM 用）
        let load_linear_f16 = |key: String| -> R<GpuTensor16> {
            let data = tensor_to_f32(&st.tensor(&key)?);
            let t = rt.create_tensor_f16(data.len())?;
            rt.upload_f16(&t, &data)?;
            Ok(t)
        };
        // 低秩权重：gemv 需要 [out, in] 行主序。
        // 各模型原始布局不同（g1h=[out,in]、g1d=[in,out]），按实际形状自适应转置。
        let load_lowrank = |name: &str, out_dim: usize, in_dim: usize| -> R<GpuTensor> {
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
            let t = rt.create_tensor(len)?;
            rt.upload(&t, &oriented)?;
            Ok(t)
        };
        // 低秩权重 → fp16（tensor-core GEMM 用，补齐到 [pad_out, pad_in]，不满的行/列填 0）。
        // 第一级投影 pad=[mid_pad, C]（n=mid_pad, k=C），第二级投影 pad=[C, mid_pad]（n=C, k=mid_pad）。
        let load_lowrank_f16 = |name: &str,
                                out_dim: usize,
                                in_dim: usize,
                                pad_out: usize,
                                pad_in: usize|
         -> R<GpuTensor16> {
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
                for c in 0..in_dim {
                    padded[r * pad_in + c] = oriented[r * in_dim + c];
                }
            }
            let t = rt.create_tensor_f16(padded.len())?;
            rt.upload_f16(&t, &padded)?;
            Ok(t)
        };

        let (wm, am, vm, gm) = (cfg.w_mid, cfg.a_mid, cfg.v_mid, cfg.g_mid);
        let (wwp, aap, vvp, ggp) = (cfg.w_mid_pad, cfg.a_mid_pad, cfg.v_mid_pad, cfg.g_mid_pad);

        // any4 量化权重：探测 {key}.any4_idx 是否存在。
        // 存在 → 上传 idx/lut/sz 三路张量，并 CPU 反量化出 f32 供 fp16 副本（prefill GEMM 用）；
        // 不存在 → None，走原 fp16 加载路径（单二进制兼容两种模型文件）。
        let load_any4 = |key: &str, m: usize, k: usize| -> R<Option<GpuTensorAny4>> {
            if st.tensor(&format!("{key}.any4_idx")).is_err() {
                return Ok(None);
            }
            let idx_t = st.tensor(&format!("{key}.any4_idx"))?; // U8 [M, K/2]
            let lut_t = st.tensor(&format!("{key}.any4_lut"))?; // F16 [M, 16]
            let sz_t = st.tensor(&format!("{key}.any4_sz"))?; // U32 [M, K/128]
            let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
            let lut32 = tensor_to_f32(&lut_t);
            let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
            assert_eq!(idx_u32.len(), m * k / 8, "{key}.any4_idx 形状不符");
            assert_eq!(lut32.len(), m * 16, "{key}.any4_lut 形状不符");
            assert_eq!(sz_u32.len(), m * k / 128, "{key}.any4_sz 形状不符");
            let idx_gpu = rt.create_tensor_u32(m * k / 8)?;
            rt.upload_u32(&idx_gpu, idx_u32)?;
            let lut_gpu = rt.create_tensor_f16(m * 16)?;
            rt.upload_f16(&lut_gpu, &lut32)?;
            let sz_gpu = rt.create_tensor_u32(m * k / 128)?;
            rt.upload_u32(&sz_gpu, sz_u32)?;
            log::info!("layer {idx}: {key} 使用 any4 量化权重（{m}x{k}）");
            Ok(Some(GpuTensorAny4 {
                idx: idx_gpu,
                lut: lut_gpu,
                sz: sz_gpu,
                m,
                k,
            }))
        };
        // int8 量化权重：探测 {key}.int8_idx 是否存在。
        // 存在 → 上传 idx/sz 二路张量（idx 为 U8 [M,K]，重解释为 uint32 [M,K/4]）；
        // 不存在 → None，走原 fp16/any4 加载路径（单二进制兼容三种模型文件）。
        let load_int8 = |key: &str, m: usize, k: usize| -> R<Option<GpuTensorInt8>> {
            if st.tensor(&format!("{key}.int8_idx")).is_err() {
                return Ok(None);
            }
            let idx_t = st.tensor(&format!("{key}.int8_idx"))?; // U8 [M, K]
            let sz_t = st.tensor(&format!("{key}.int8_sz"))?; // U32 [M, K/128]
            let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
            let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
            assert_eq!(idx_u32.len(), m * k / 4, "{key}.int8_idx 形状不符");
            assert_eq!(sz_u32.len(), m * k / 128, "{key}.int8_sz 形状不符");
            let idx_gpu = rt.create_tensor_u32(m * k / 4)?;
            rt.upload_u32(&idx_gpu, idx_u32)?;
            let sz_gpu = rt.create_tensor_u32(m * k / 128)?;
            rt.upload_u32(&sz_gpu, sz_u32)?;
            log::info!("layer {idx}: {key} 使用 int8 量化权重（{m}x{k}）");
            Ok(Some(GpuTensorInt8 {
                idx: idx_gpu,
                sz: sz_gpu,
                m,
                k,
            }))
        };
        // 三路量化权重统一加载：优先 int8 → any4 → fp16（按模型文件内容自动路由）。
        // 返回 (int8, any4, fp16)，三者至多一个为 Some。
        let load_linear = |key: &str,
                           m: usize,
                           k: usize|
         -> R<(
            Option<GpuTensorInt8>,
            Option<GpuTensorAny4>,
            Option<GpuTensor16>,
        )> {
            if let Some(a8) = load_int8(key, m, k)? {
                return Ok((Some(a8), None, None));
            }
            match load_any4(key, m, k)? {
                Some(a4) => Ok((None, Some(a4), None)),
                None => Ok((None, None, Some(load_linear_f16(key.to_string())?))),
            }
        };
        let (ffn_key_a8, ffn_key_a4, ffn_key_w16) =
            load_linear(&format!("blocks.{idx}.ffn.key.weight"), cfg.ffn_hidden, c)?;
        let (ffn_value_a8, ffn_value_a4, ffn_value_w16) =
            load_linear(&format!("blocks.{idx}.ffn.value.weight"), c, cfg.ffn_hidden)?;
        let (receptance_a8, receptance_a4, receptance_w16) =
            load_linear(&format!("blocks.{idx}.att.receptance.weight"), c, c)?;
        let (key_a8, key_a4, key_w16) = load_linear(&format!("blocks.{idx}.att.key.weight"), c, c)?;
        let (value_a8, value_a4, value_w16) =
            load_linear(&format!("blocks.{idx}.att.value.weight"), c, c)?;
        // att.output：int8/any4 时走对应 GEMM（零 fp16 副本）；
        // fp32 诊断副本仅 GEMM_DIAG 时用反量化权重创建（正式推理不创建，省 ~26MB/层）
        let want_diag = std::env::var("GEMM_DIAG").is_ok();
        let (att_output_a8, att_output_a4, output_w16, output_w) =
            if let Some(a8) = load_int8(&format!("blocks.{idx}.att.output.weight"), c, c)? {
                let ow = if want_diag {
                    let key = format!("blocks.{idx}.att.output.weight");
                    let idx_t = st.tensor(&format!("{key}.int8_idx"))?;
                    let sz_t = st.tensor(&format!("{key}.int8_sz"))?;
                    let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
                    let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
                    let w32 = dequant_int8(idx_u32, sz_u32, c, c);
                    let t32 = rt.create_tensor(w32.len())?;
                    rt.upload(&t32, &w32)?;
                    Some(t32)
                } else {
                    None
                };
                (Some(a8), None, None, ow)
            } else if let Some(a4) = load_any4(&format!("blocks.{idx}.att.output.weight"), c, c)? {
                let ow = if want_diag {
                    let key = format!("blocks.{idx}.att.output.weight");
                    let idx_t = st.tensor(&format!("{key}.any4_idx"))?;
                    let lut_t = st.tensor(&format!("{key}.any4_lut"))?;
                    let sz_t = st.tensor(&format!("{key}.any4_sz"))?;
                    let idx_u32: &[u32] = bytemuck::cast_slice(idx_t.data());
                    let lut32 = tensor_to_f32(&lut_t);
                    let sz_u32: &[u32] = bytemuck::cast_slice(sz_t.data());
                    let w32 = dequant_any4(idx_u32, &lut32, sz_u32, c, c);
                    let t32 = rt.create_tensor(w32.len())?;
                    rt.upload(&t32, &w32)?;
                    Some(t32)
                } else {
                    None
                };
                (None, Some(a4), None, ow)
            } else {
                (
                    None,
                    None,
                    Some(load_linear_f16(format!("blocks.{idx}.att.output.weight"))?),
                    if want_diag {
                        Some(load_linear_diag(format!("blocks.{idx}.att.output.weight"))?)
                    } else {
                        None
                    },
                )
            };

        let mut layer = Self {
            ln1_w: load1("ln1.weight")?,
            ln1_b: load1("ln1.bias")?,
            ln2_w: load1("ln2.weight")?,
            ln2_b: load1("ln2.bias")?,
            ln_x_w: load1("att.ln_x.weight")?,
            ln_x_b: load1("att.ln_x.bias")?,
            x_r: load1("att.x_r")?,
            x_w: load1("att.x_w")?,
            x_k: load1("att.x_k")?,
            x_v: load1("att.x_v")?,
            x_a: load1("att.x_a")?,
            x_g: load1("att.x_g")?,
            w0: load1("att.w0")?,
            a0: load1("att.a0")?,
            v0: load1("att.v0")?,
            // 低秩权重转置为 [out, in]
            w1: load_lowrank("w1", wm, c)?, // [out=wm, in=C]
            w2: load_lowrank("w2", c, wm)?, // [out=C, in=wm]
            a1: load_lowrank("a1", am, c)?,
            a2: load_lowrank("a2", c, am)?,
            v1: load_lowrank("v1", vm, c)?,
            v2: load_lowrank("v2", c, vm)?,
            g1: load_lowrank("g1", gm, c)?,
            g2: load_lowrank("g2", c, gm)?,
            r_k: load1("att.r_k")?,
            k_k: load1("att.k_k")?,
            k_a: load1("att.k_a")?,
            ffn_x_k: load1("ffn.x_k")?,
            // 诊断用 fp32 输出权重（仅 GEMM_DIAG 参考路径）
            output_w,
            // fp16 线性权重（非 any4 矩阵常驻；any4 矩阵为 None，走 any4 GEMM/GEMV 零副本）
            receptance_w16,
            key_w16,
            value_w16,
            output_w16,
            ffn_key_w16,
            ffn_value_w16,
            ffn_key_a4,
            ffn_value_a4,
            att_output_a4,
            receptance_a4,
            key_a4,
            value_a4,
            ffn_key_a8,
            ffn_value_a8,
            att_output_a8,
            receptance_a8,
            key_a8,
            value_a8,
            // fp16 低秩权重（补齐到 mid_pad）
            w1_16: load_lowrank_f16("w1", wm, c, wwp, c)?, // [wm_pad, C]
            w2_16: load_lowrank_f16("w2", c, wm, c, wwp)?, // [C, wm_pad]
            a1_16: load_lowrank_f16("a1", am, c, aap, c)?, // [am_pad, C]
            a2_16: load_lowrank_f16("a2", c, am, c, aap)?, // [C, am_pad]
            v1_16: load_lowrank_f16("v1", vm, c, vvp, c)?, // [vm_pad, C]
            v2_16: load_lowrank_f16("v2", c, vm, c, vvp)?, // [C, vm_pad]
            g1_16: load_lowrank_f16("g1", gm, c, ggp, c)?, // [gm_pad, C]
            g2_16: load_lowrank_f16("g2", c, gm, c, ggp)?, // [C, gm_pad]
        };
        // 权重上传完成，释放全部 host（系统内存）缓冲，仅保留 device 拷贝
        layer.drop_weight_hosts(rt);
        Ok(layer)
    }

    /// 释放本层所有权重的 host（系统内存）缓冲。权重上传完成后调用，运行期只读 device。
    fn drop_weight_hosts(&mut self, rt: &Runtime) {
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
            rt.drop_host(t);
        }
        // 诊断用 output_w（fp32，Option：仅 GEMM_DIAG 时存在）
        if let Some(mut t) = self.output_w.take() {
            rt.drop_host(&mut t);
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
            rt.drop_host_f16(t);
        }
        // fp16 线性权重（Option）：非 any4 矩阵常驻，释放其 host；any4 矩阵此时为 None 跳过
        for t in [
            &mut self.receptance_w16,
            &mut self.key_w16,
            &mut self.value_w16,
            &mut self.output_w16,
            &mut self.ffn_key_w16,
            &mut self.ffn_value_w16,
        ] {
            if let Some(t) = t.as_mut() {
                rt.drop_host_f16(t);
            }
        }
        // any4 量化权重
        for a4 in [
            &mut self.ffn_key_a4,
            &mut self.ffn_value_a4,
            &mut self.att_output_a4,
            &mut self.receptance_a4,
            &mut self.key_a4,
            &mut self.value_a4,
        ]
        .into_iter()
        .flatten()
        {
            rt.drop_host_u32(&mut a4.idx);
            rt.drop_host_f16(&mut a4.lut);
            rt.drop_host_u32(&mut a4.sz);
        }
    }
}

impl GpuState {
    fn new(rt: &Runtime, c: usize, h: usize, n: usize) -> R<Self> {
        let tmix_x = rt.create_tensor(c)?;
        rt.upload(&tmix_x, &vec![0.0; c])?;
        let tmix_rnn = rt.create_tensor(h * n * n)?;
        rt.upload(&tmix_rnn, &vec![0.0; h * n * n])?;
        let cmix_x = rt.create_tensor(c)?;
        rt.upload(&cmix_x, &vec![0.0; c])?;
        Ok(Self {
            tmix_x,
            tmix_rnn,
            cmix_x,
        })
    }

    /// 重置 RNN 状态为零（用于多次独立 forward）
    fn reset(&self, rt: &Runtime, c: usize, h: usize, n: usize) -> R<()> {
        rt.upload(&self.tmix_x, &vec![0.0; c])?;
        rt.upload(&self.tmix_rnn, &vec![0.0; h * n * n])?;
        rt.upload(&self.cmix_x, &vec![0.0; c])?;
        Ok(())
    }
}

impl WorkBuffers {
    fn new(rt: &Runtime, c: usize, vocab: usize, cfg: &ModelConfig) -> R<Self> {
        // 创建 C 大小的 buffer 并初始化为 0
        let mk_c = || -> R<GpuTensor> {
            let t = rt.create_tensor(c)?;
            rt.upload(&t, &vec![0.0; c])?;
            Ok(t)
        };
        let mk = |len: usize| -> R<GpuTensor> {
            let t = rt.create_tensor(len)?;
            rt.upload(&t, &vec![0.0; len])?;
            Ok(t)
        };
        // fp16 零缓冲（w/a/g/v 输出，链第二级下游以 fp16 读取减半带宽）
        let mk_c16 = || -> R<GpuTensor16> {
            let t = rt.create_tensor_f16(c)?;
            rt.upload_f16(&t, &vec![0.0f32; c])?;
            Ok(t)
        };

        Ok(Self {
            x: mk_c()?,
            ln1: mk_c()?,
            xr: mk_c()?,
            xw: mk_c()?,
            xk: mk_c()?,
            xv: mk_c()?,
            xa: mk_c()?,
            xg: mk_c()?,
            prev_x: mk_c()?,
            r: mk_c()?,
            k: mk_c()?,
            v: mk_c16()?,
            v_full: mk_c()?,
            gate: mk_c()?,
            w_full: mk_c()?,
            w_sig: mk_c()?,
            w: mk_c16()?,
            a_full: mk_c()?,
            a: mk_c16()?,
            kk_l2: mk_c()?,
            k_mod: mk_c()?,
            b_vec: mk_c()?,
            y: mk_c()?,
            y_norm: mk_c()?,
            g: mk_c16()?,
            y_g: mk_c()?,
            y_out: mk_c()?,
            ln2: mk_c()?,
            prev_c: mk_c()?,
            xb: mk_c()?,
            v2: mk_c()?,
            x_norm: mk_c()?,
            tmp_c: mk_c()?,
            // 其他大小
            v_mid: mk(cfg.v_mid)?,
            w_mid: mk(cfg.w_mid)?,
            a_mid: mk(cfg.a_mid)?,
            g_mid: mk(cfg.g_mid)?,
            r2: mk(cfg.ffn_hidden)?,
            logits: mk(vocab)?,
            token_argmax: mk(1)?,
            current_token: mk(1)?,
        })
    }
}

impl SeqBuffers {
    /// 创建序列并行的 [T, *] 工作缓冲区
    fn new(rt: &Runtime, t: usize, c: usize, vocab: usize, cfg: &ModelConfig) -> R<Self> {
        let m_pad = t.div_ceil(256) * 256;
        let mk = |len: usize| -> R<GpuTensor> {
            let buf = rt.create_tensor(len)?;
            rt.upload(&buf, &vec![0.0; len])?;
            Ok(buf)
        };
        let mk_c = |len: usize| -> R<GpuTensor> {
            let buf = rt.create_tensor(t * len)?;
            rt.upload(&buf, &vec![0.0; t * len])?;
            Ok(buf)
        };
        let mk_pad = |len: usize| -> R<GpuTensor> {
            let buf = rt.create_tensor(m_pad * len)?;
            rt.upload(&buf, &vec![0.0; m_pad * len])?;
            Ok(buf)
        };
        let mk_f16 = |len: usize| -> R<GpuTensor16> {
            let buf = rt.create_tensor_f16(m_pad * len)?;
            rt.upload_f16(&buf, &vec![0.0; m_pad * len])?;
            Ok(buf)
        };

        Ok(Self {
            t,
            m_pad,
            x: mk_pad(c)?,
            ln1: mk_c(c)?,
            xr: mk_c(c)?,
            xw: mk_c(c)?,
            xk: mk_c(c)?,
            xv: mk_c(c)?,
            xa: mk_c(c)?,
            xg: mk_c(c)?,
            r: mk_pad(c)?,
            k: mk_pad(c)?,
            v: mk_pad(c)?,
            v_first: mk_pad(c)?,
            v_full: mk_pad(c)?,
            gate: mk_c(c)?,
            w_full: mk_pad(c)?,
            w_sig: mk_c(c)?,
            w: mk_c(c)?,
            a_full: mk_pad(c)?,
            a: mk_c(c)?,
            kk_l2: mk_c(c)?,
            k_mod: mk_c(c)?,
            b_vec: mk_c(c)?,
            y: mk_c(c)?,
            y_norm: mk_c(c)?,
            g: mk_pad(c)?,
            y_g: mk_c(c)?,
            y_out: mk_pad(c)?,
            diag_out_ref: mk_pad(c)?,
            ln2: mk_c(c)?,
            xb: mk_c(c)?,
            v2: mk_pad(c)?,
            x_norm: mk_c(c)?,
            tmp_c: mk_c(c)?,
            v_mid: mk_pad(cfg.v_mid_pad)?,
            w_mid: mk_pad(cfg.w_mid_pad)?,
            a_mid: mk_pad(cfg.a_mid_pad)?,
            g_mid: mk_pad(cfg.g_mid_pad)?,
            r2: mk_pad(cfg.ffn_hidden)?,
            head_in: mk(c)?,
            logits: mk(vocab)?,
            xr16: mk_f16(c)?,
            xk16: mk_f16(c)?,
            xv16: mk_f16(c)?,
            xw16: mk_f16(c)?,
            xa16: mk_f16(c)?,
            xg16: mk_f16(c)?,
            y_g16: mk_f16(c)?,
            xb16: mk_f16(c)?,
            r2_16: mk_f16(cfg.ffn_hidden)?,
            v_mid16: mk_f16(cfg.v_mid_pad)?,
            w_mid16: mk_f16(cfg.w_mid_pad)?,
            a_mid16: mk_f16(cfg.a_mid_pad)?,
            g_mid16: mk_f16(cfg.g_mid_pad)?,
            // dequant 每次 GEMM 前全量覆写，无需零初始化上传
            w_scratch: rt.create_tensor_f16(cfg.ffn_hidden * c)?,
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
    let mut out = vec![0.0; data.len()];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// int8 CPU 反量化（非对称 per-group=128，与 tools/quantize_any4.py --bits 8 及 model.rs 一致）。
/// `w[m,k] = scale[m,k/128] * idx[m,k] + zero[m,k/128]`。
/// - idx: [M*K/4] uint32，每 uint32 打包 4 个 uint8 权重（低位字节在前：b0=byte0 … b3=byte3）
/// - sz:  [M*K/128] uint32（scale fp16 低 16 位 | zero fp16 高 16 位）
///
/// 供 GEMM_DIAG 诊断时重建 fp32 权重（正式推理直接走 int8 GEMV/GEMM，不创建此副本）
fn dequant_int8(idx: &[u32], sz: &[u32], m: usize, k: usize) -> Vec<f32> {
    let kg = k / 128;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        let row = &mut w[r * k..(r + 1) * k];
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
    w
}

/// any4 CPU 反量化（arXiv:2507.04610，group=128）：
/// `w[m,k] = scale[m,k/128] * lut[m, idx[m,k]] + zero[m,k/128]`
/// - idx: [M*K/8] uint32（每 uint32 打包 8 个 4-bit 索引，低位在前）
/// - lut: [M*16] f32（每行学习码本）
/// - sz:  [M*K/128] uint32（scale fp16 低 16 位 | zero fp16 高 16 位）
///
/// 加载时由 any4 权重重建 fp32 权重（转 fp16 供 prefill GEMM 用）
fn dequant_any4(idx: &[u32], lut: &[f32], sz: &[u32], m: usize, k: usize) -> Vec<f32> {
    let kg = k / 128;
    let mut w = vec![0.0f32; m * k];
    for r in 0..m {
        let row_lut = &lut[r * 16..r * 16 + 16];
        let row = &mut w[r * k..(r + 1) * k];
        for (g, chunk) in row.chunks_mut(128).enumerate() {
            let szv = sz[r * kg + g];
            let scale = f16::from_bits((szv & 0xFFFF) as u16).to_f32();
            let zero = f16::from_bits((szv >> 16) as u16).to_f32();
            for (j, wv) in chunk.iter_mut().enumerate() {
                let ki = g * 128 + j;
                let pack = idx[r * (k / 8) + ki / 8];
                let q = ((pack >> ((ki % 8) * 4)) & 0xF) as usize;
                *wv = scale * row_lut[q] + zero;
            }
        }
    }
    w
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
