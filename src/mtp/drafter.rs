//! MTP drafter：权重上传 + per-slot 状态 + 单步编排（上游 `docs/mtp.md` §1.4 / §3.2 / §3.3）。
//!
//! 只有两个专用核（[`ComputeBackend::mtp_rms_rope`] 与
//! [`ComputeBackend::mtp_kv_write`]/[`ComputeBackend::mtp_attn`]），其余全部复用既有
//! GEMV / LayerNorm / 逐元素原语。
//!
//! **边界**：本模块只产 draft，**不决定输出**。接受/拒绝与主模型回滚在 P5（`gpu_model`）。
//!
//! 单步（§1.4，深度 k 在绝对位置 p = j + k − 1）：
//! ```text
//! e  = ln0(emb(token))                       token = 被起草的那个 token
//! u  = (k == 1) ? fproj·[LN_lh(h); LN_lf0(m1); LN_lf1(m2)] : LN_lhd(prev_out)
//! x  = LN_ln0( proj·[u; LN_le(e)] + dep[k-1] )
//! xl = LN_ln1(x); q = RoPE(RMS(Wq·xl)); kk = RoPE(RMS(Wk·xl)); v = Wv·xl
//! 写 (kk,v) 到 p，再 GQA 注意力（可见性 0 ≤ sp ≤ p 且 p − sp < W，query 见自身）
//! x  = x + Wo·( y ⊙ sigmoid(Wg·xl) )
//! x2 = LN_ln2(x); kx = x2 + (shift − x2) ⊙ x_k; x = x + Wvalue·relu(Wkey·kx)²
//! out = LN_lo(x); draft = argmax(head·out) 限定在 draft_vocab[:N] 内
//! ```
//!
//! `shift`（channel-mix 的 token shift）= **drafter 上一位置** 的 `x2`；深度 1 用状态里
//! 最后一个已提交位置的 `x2`（[`MtpState::ffn_x`]）。
//!
//! draft 词表截断的实现方式：不另存一份 head 副本，而是给 logits 加一张**静态掩码**
//! （draft 内 0、其余 −1e30），再在全词表上 argmax ⇒ 索引即 token id，不需要回映射，
//! 也**不需要**任何 host 往返（链内 token 全程留在设备上）。

use crate::backend::{ComputeBackend, Int6Handle, Int8Handle, TensorDtype, TensorId, Wq};
use crate::mtp::sidecar::{MtpSidecar, MtpTensor};
use crate::mtp::{MTP_DRAFT_VOCAB, MTP_HEAD_DIM, MTP_N_HEAD, MTP_N_KV_HEAD, MTP_ROPE_DIM};
use crate::runtime::R;

/// 主模型 head 的三种形态：drafter 复用它算 draft logits，不另存副本。
#[derive(Clone, Copy)]
pub enum HeadSrc {
    I8(Int8Handle),
    I6(Int6Handle),
    F16(TensorId),
}

impl HeadSrc {
    fn gemv(
        &self,
        be: &mut dyn ComputeBackend,
        x: TensorId,
        y: TensorId,
        vocab: usize,
        c: usize,
    ) -> R<()> {
        match self {
            HeadSrc::I8(h) => be.w_plain(&Wq::I8(*h), x, y, vocab, c, 1),
            HeadSrc::I6(h) => be.w_plain(&Wq::I6(*h), x, y, vocab, c, 1),
            HeadSrc::F16(t) => be.gemv_f16(*t, x, y, vocab, c, 1),
        }
    }
}

/// 一个 Linear：fp16 或 int6（N-major）量化权重，`[out, in]` 行主序语义。
///
/// 为什么要 int6 档：drafter 每步要读 ~1 GB 权重（fp16），实测 3.0 ms 全是带宽
/// （329 GB/s，已接近本机上限）。权重压到 int6（0.75 B/权重）后字节数减半 ⇒ 这部分
/// 直接砍到 ~1.6 ms。上游 `docs/mtp.md` §3.1 也是这么做的（"weight-only int6, 0.5 GB → 0.19 GB"），
/// 草稿只影响接受率、不影响输出分布，所以量化是安全的。
#[derive(Clone, Copy)]
enum Lin {
    F16(TensorId),
    I6(Int6Handle),
}

impl Lin {
    /// `y = x @ W^T`
    fn plain(
        self,
        be: &mut dyn ComputeBackend,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
    ) -> R<()> {
        match self {
            Lin::F16(w) => be.gemv_f16(w, x, y, m, k, 1),
            Lin::I6(h) => be.w_plain(&Wq::I6(h), x, y, m, k, 1),
        }
    }

    /// `y += x @ W^T`
    fn add(
        self,
        be: &mut dyn ComputeBackend,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
    ) -> R<()> {
        match self {
            Lin::F16(w) => be.gemv_f16_add(w, x, y, m, k, 1),
            Lin::I6(h) => be.w_add(&Wq::I6(h), x, y, m, k, 1),
        }
    }

    /// `y = relu²(x @ W^T)`
    fn relu2(
        self,
        be: &mut dyn ComputeBackend,
        x: TensorId,
        y: TensorId,
        m: usize,
        k: usize,
    ) -> R<()> {
        match self {
            Lin::F16(w) => be.gemv_f16_relu2(w, x, y, m, k, 1),
            Lin::I6(h) => be.w_relu2(&Wq::I6(h), x, y, m, k, 1),
        }
    }
}

/// 一组 LayerNorm 的仿射参数。
#[derive(Clone, Copy)]
struct Ln {
    w: TensorId,
    b: TensorId,
}

struct Weights {
    fproj: Lin,
    proj: Lin,
    q: Lin,
    g: Lin,
    k: Lin,
    v: Lin,
    o: Lin,
    ffn_key: Lin,
    ffn_value: Lin,
    lh: Ln,
    lf0: Ln,
    lf1: Ln,
    lhd: Ln,
    le: Ln,
    ln0: Ln,
    ln1: Ln,
    ln2: Ln,
    lo: Ln,
    /// `dep[8, C]` 拆成 8 行（`elementwise_add` 需要独立的 `[C]` 张量）。
    dep: Vec<TensorId>,
    x_k: TensorId,
    q_norm: TensorId,
    k_norm: TensorId,
}

/// 工作缓冲（单流 batch=1，全部 f32）。
#[derive(Clone, Copy)]
struct Bufs {
    e: TensorId,
    h_n: TensorId,
    m1_n: TensorId,
    m2_n: TensorId,
    cat3: TensorId,
    u: TensorId,
    e_n: TensorId,
    cat2: TensorId,
    x0: TensorId,
    x: TensorId,
    xl: TensorId,
    q: TensorId,
    g: TensorId,
    kk: TensorId,
    vv: TensorId,
    y: TensorId,
    gs: TensorId,
    x2: TensorId,
    shift: TensorId,
    kx: TensorId,
    hh: TensorId,
    out: TensorId,
    /// `[1]`：当前 token id（f32 位模式，供 gather 与 argmax 写回）。
    tok: TensorId,
    /// `[max_k]`：本次链产出的 draft token（链尾一次读回）。
    drafts: TensorId,
    logits: TensorId,
}

/// drafter 的持久状态（§3.2；单流 = 槽 0）。
pub struct MtpState {
    /// KV ring，布局 `[S][n_kv*D]`（与内核的 `slot*nkv_d + kvh*d` 一致）。
    pub kc: TensorId,
    pub vc: TensorId,
    /// `[S]` u32：每个 ring 槽的绝对位置（`u32::MAX` = 空）。
    pub ring_pos: TensorId,
    /// `[C]`：最后一个**已提交**位置的 `x2`（channel-mix 的 token shift）。
    pub ffn_x: TensorId,
    /// 已提交的位置数（= 下一个深度 1 步的 RoPE 位置）。
    pub pos: u32,
    s_ring: usize,
}

/// MTP drafter（权重 + 状态 + 工作缓冲）。
pub struct MtpDrafter {
    pub c: usize,
    pub vocab: usize,
    pub window: usize,
    /// 训练深度（`dep` 的有效行数）。
    pub depth: usize,
    /// 一步最多起草几个 token（`S = window + max_k + 1`）。
    pub max_k: usize,
    ln_eps: f32,
    rope_dim: usize,
    rope_theta: f32,
    emb_ln: TensorId,
    head: HeadSrc,
    w: Weights,
    b: Bufs,
    /// `[vocab]` f32：draft 词表内 0，其余 −1e30（argmax 只在 draft 词表内选）。
    draft_mask: TensorId,
    pub draft_vocab: Vec<u32>,
    pub st: MtpState,
}

impl MtpDrafter {
    /// 上传权重 + 分配状态/缓冲。`emb_ln` 是主模型的 `ln0(emb)` fp16 表（共享，不复制）。
    pub fn new(
        be: &mut dyn ComputeBackend,
        sc: &MtpSidecar,
        emb_ln: TensorId,
        head: HeadSrc,
        vocab: usize,
        max_k: usize,
    ) -> R<Self> {
        let c = sc.meta.n_embd;
        let d = MTP_HEAD_DIM;
        let max_k = max_k.max(1);
        let s_ring = sc.meta.window + max_k + 1;

        let get = |name: &str| -> R<&MtpTensor> {
            sc.tensors
                .get(name)
                .ok_or_else(|| format!("MTP sidecar 缺张量 {name}").into())
        };
        let f16t = |be: &mut dyn ComputeBackend, name: &str, m: usize, k: usize| -> R<Lin> {
            let t = get(name)?;
            // int6 档**默认关**（`MTP_DRAFTER_I6=1` 才开）：实测本机只从 3.26 → 2.93 ms
            // （权重减半却只省 10 %，因为 drafter 的矩阵窄、`gemv_int6nm` 在此形状上
            // 效率远低于 `gemv_f16`），而 int6 的草稿略差、端到端反而 1.13× → 1.11×。
            // 保留这条路是因为它已经跑通（含 CPU 对拍规则的打包器），等 head 子表
            // 那一半（378 MB/步）一起做时再评估。
            if std::env::var("MTP_DRAFTER_I6").is_ok() {
                return Ok(Lin::I6(up_i6(be, t, m, k)?));
            }
            Ok(Lin::F16(up_f16(be, t)?))
        };
        let ln = |be: &mut dyn ComputeBackend, name: &str| -> R<Ln> {
            Ok(Ln {
                w: up_f32(be, get(&format!("{name}.weight"))?)?,
                b: up_f32(be, get(&format!("{name}.bias"))?)?,
            })
        };

        let w = Weights {
            fproj: f16t(be, "fproj.weight", c, 3 * c)?,
            proj: f16t(be, "proj.weight", c, 2 * c)?,
            q: f16t(be, "attn.q.weight", MTP_N_HEAD * d, c)?,
            g: f16t(be, "attn.g.weight", MTP_N_HEAD * d, c)?,
            k: f16t(be, "attn.k.weight", MTP_N_KV_HEAD * d, c)?,
            v: f16t(be, "attn.v.weight", MTP_N_KV_HEAD * d, c)?,
            o: f16t(be, "attn.o.weight", c, MTP_N_HEAD * d)?,
            ffn_key: f16t(be, "blk.ffn.key.weight", 4 * c, c)?,
            ffn_value: f16t(be, "blk.ffn.value.weight", c, 4 * c)?,
            lh: ln(be, "lh")?,
            lf0: ln(be, "lf.0")?,
            lf1: ln(be, "lf.1")?,
            lhd: ln(be, "lhd")?,
            le: ln(be, "le")?,
            ln0: ln(be, "blk.ln0")?,
            ln1: ln(be, "blk.ln1")?,
            ln2: ln(be, "blk.ln2")?,
            lo: ln(be, "lo")?,
            dep: {
                let dep = get("dep")?;
                (0..dep.shape[0])
                    .map(|r| {
                        let id = be.create_tensor(c, TensorDtype::F32)?;
                        be.upload(id, &dep.data[r * c..(r + 1) * c])?;
                        Ok(id)
                    })
                    .collect::<R<Vec<_>>>()?
            },
            x_k: up_f32(be, get("blk.ffn.x_k")?)?,
            q_norm: up_f32(be, get("attn.q_norm.weight")?)?,
            k_norm: up_f32(be, get("attn.k_norm.weight")?)?,
        };

        let mk = |be: &mut dyn ComputeBackend, n: usize| -> R<TensorId> {
            be.create_tensor(n, TensorDtype::F32)
        };
        let b = Bufs {
            e: mk(be, c)?,
            h_n: mk(be, c)?,
            m1_n: mk(be, c)?,
            m2_n: mk(be, c)?,
            cat3: mk(be, 3 * c)?,
            u: mk(be, c)?,
            e_n: mk(be, c)?,
            cat2: mk(be, 2 * c)?,
            x0: mk(be, c)?,
            x: mk(be, c)?,
            xl: mk(be, c)?,
            q: mk(be, MTP_N_HEAD * d)?,
            g: mk(be, MTP_N_HEAD * d)?,
            kk: mk(be, MTP_N_KV_HEAD * d)?,
            vv: mk(be, MTP_N_KV_HEAD * d)?,
            y: mk(be, MTP_N_HEAD * d)?,
            gs: mk(be, MTP_N_HEAD * d)?,
            x2: mk(be, c)?,
            shift: mk(be, c)?,
            kx: mk(be, c)?,
            hh: mk(be, 4 * c)?,
            out: mk(be, c)?,
            tok: mk(be, 1)?,
            drafts: mk(be, max_k)?,
            logits: mk(be, vocab)?,
        };

        // draft 词表掩码（`draft_vocab[:N]` 内 0，其余 −1e30；用 `-1e30` 与内核哨兵一致）。
        let n_draft = sc.draft_vocab.len().min(MTP_DRAFT_VOCAB).min(vocab);
        if n_draft == 0 {
            return Err("MTP sidecar 的 draft_vocab 为空".into());
        }
        let mut mask = vec![-1.0e30f32; vocab];
        for &t in &sc.draft_vocab[..n_draft] {
            if let Some(m) = mask.get_mut(t as usize) {
                *m = 0.0;
            }
        }
        let draft_mask = be.create_tensor(vocab, TensorDtype::F32)?;
        be.upload(draft_mask, &mask)?;
        be.drop_host(draft_mask);

        let st = MtpState {
            kc: be.create_tensor(s_ring * MTP_N_KV_HEAD * d, TensorDtype::F32)?,
            vc: be.create_tensor(s_ring * MTP_N_KV_HEAD * d, TensorDtype::F32)?,
            ring_pos: be.create_tensor(s_ring, TensorDtype::U32)?,
            ffn_x: be.create_tensor(c, TensorDtype::F32)?,
            pos: 0,
            s_ring,
        };

        let mut this = Self {
            c,
            vocab,
            window: sc.meta.window,
            depth: sc.meta.depth,
            max_k,
            ln_eps: sc.meta.ln_eps,
            rope_dim: if sc.meta.rope_dim == 0 {
                MTP_ROPE_DIM
            } else {
                sc.meta.rope_dim
            },
            rope_theta: if sc.meta.rope_theta > 0.0 {
                sc.meta.rope_theta
            } else {
                1.0e6
            },
            emb_ln,
            head,
            w,
            b,
            draft_mask,
            draft_vocab: sc.draft_vocab[..n_draft].to_vec(),
            st,
        };
        this.reset(be)?;
        Ok(this)
    }

    /// 清空状态（KV ring / ring_pos / ffn_x / pos），等价于新会话。
    pub fn reset(&mut self, be: &mut dyn ComputeBackend) -> R<()> {
        let n = self.st.s_ring * MTP_N_KV_HEAD * MTP_HEAD_DIM;
        be.fill_zeros(&[(self.st.kc, n), (self.st.vc, n), (self.st.ffn_x, self.c)])?;
        be.upload_u32(self.st.ring_pos, &vec![u32::MAX; self.st.s_ring])?;
        self.st.pos = 0;
        Ok(())
    }

    /// 已提交的位置数。
    pub fn pos(&self) -> u32 {
        self.st.pos
    }

    /// 诊断：对给定 `[C]` 输入跑一次 head（**不加** draft 词表掩码）并回读 argmax。
    /// 用途：把 `ln_out(x)` 喂进来，结果必须等于主模型这一步的 argmax（同一份 head、
    /// 同一行 x_norm）——用来把「taps/head 通路错」与「drafter 块错」分开。
    pub fn head_argmax(&mut self, be: &mut dyn ComputeBackend, x: TensorId) -> R<u32> {
        let b = self.b;
        self.head.gemv(be, x, b.logits, self.vocab, self.c)?;
        be.argmax(b.logits, b.tok, self.vocab)?;
        let t = be.download(b.tok)?;
        Ok(t[0].to_bits())
    }

    /// 深度 1 一步（§3.3 的 `commit_rows`）：消费 `(H, token)`，写 KV 到位置 `pos`，
    /// 推进 `pos` 并把 `ffn_x` 更新为这一步的 `x2`。
    ///
    /// `h/m1/m2` 是主模型在位置 `pos` 的三个残差（`ln_out(x)` 之前的两层 tap 与末层），
    /// 均为设备上 `[C]` f32；`tok` 是**该位置之后**那个 token 的设备 `[1]` 索引缓冲。
    pub fn commit(
        &mut self,
        be: &mut dyn ComputeBackend,
        h: TensorId,
        m1: TensorId,
        m2: TensorId,
        tok: TensorId,
    ) -> R<()> {
        let p = self.st.pos;
        self.depth1(be, h, m1, m2, tok, p, true)?;
        be.copy_device(self.b.x2, self.st.ffn_x)?;
        self.st.pos += 1;
        Ok(())
    }

    /// 只提交、不产草稿（§3.3 的 `accept` 用）：跳过 head + 掩码 + argmax（那一步要读 168 MB
    /// int8 head，是整块最贵的一段）。纯提交 ≈ 0.1 ms，带 head ≈ 1.1 ms ⇒ 接受后补提交
    /// `j` 步的代价可以忽略。
    pub fn commit_only(
        &mut self,
        be: &mut dyn ComputeBackend,
        h: TensorId,
        m1: TensorId,
        m2: TensorId,
        tok: TensorId,
    ) -> R<()> {
        let p = self.st.pos;
        self.depth1(be, h, m1, m2, tok, p, false)?;
        be.copy_device(self.b.x2, self.st.ffn_x)?;
        self.st.pos += 1;
        Ok(())
    }

    /// 起草 `k` 个 token（§3.3 的 `chain`）：先 `commit`（深度 1），再用自身上一轮输出
    /// 追加 `k−1` 个深度。返回 `k` 个 draft token id（本步一次读回）。
    ///
    /// `h/m1/m2/tok` 与 [`Self::commit`] 同义；`k` 会被夹到 `max_k`。
    pub fn draft_chain(
        &mut self,
        be: &mut dyn ComputeBackend,
        h: TensorId,
        m1: TensorId,
        m2: TensorId,
        tok: TensorId,
        k: usize,
    ) -> R<Vec<u32>> {
        let k = k.clamp(1, self.max_k);
        let b = self.b;
        let p = self.st.pos;
        self.commit(be, h, m1, m2, tok)?;
        be.copy_part(b.drafts, 0, b.tok, 1)?;
        for i in 1..k {
            // u = LN_lhd(上一深度的 out)；token = 上一个 draft（两者都在设备上）
            be.norm(
                b.out,
                self.w.lhd.w,
                self.w.lhd.b,
                b.u,
                self.c,
                1,
                self.ln_eps,
                1,
            )?;
            let u = b.u;
            // 深度 i+1 ⇒ `dep[i]`（`dep[k-1]` 在深度 k 加）
            self.block(be, u, b.tok, p + i as u32, i, true)?;
            be.copy_part(b.drafts, i, b.tok, 1)?;
        }
        let d = be.download(b.drafts)?;
        Ok(d[..k].iter().map(|x| x.to_bits()).collect())
    }

    /// 深度 1：`u = fproj·[LN_lh(h); LN_lf0(m1); LN_lf1(m2)]`，然后跑一个块。
    #[allow(clippy::too_many_arguments)]
    fn depth1(
        &mut self,
        be: &mut dyn ComputeBackend,
        h: TensorId,
        m1: TensorId,
        m2: TensorId,
        tok: TensorId,
        p: u32,
        with_draft: bool,
    ) -> R<()> {
        let (b, c) = (self.b, self.c);
        be.norm(h, self.w.lh.w, self.w.lh.b, b.h_n, c, 1, self.ln_eps, 1)?;
        be.norm(m1, self.w.lf0.w, self.w.lf0.b, b.m1_n, c, 1, self.ln_eps, 1)?;
        be.norm(m2, self.w.lf1.w, self.w.lf1.b, b.m2_n, c, 1, self.ln_eps, 1)?;
        be.copy_part(b.cat3, 0, b.h_n, c)?;
        be.copy_part(b.cat3, c, b.m1_n, c)?;
        be.copy_part(b.cat3, 2 * c, b.m2_n, c)?;
        self.w.fproj.plain(be, b.cat3, b.u, c, 3 * c)?;
        // 深度 1 的 token shift = 最后一个已提交位置的 x2
        be.copy_device(self.st.ffn_x, b.shift)?;
        self.block(be, b.u, tok, p, 0, with_draft)
    }

    /// 一个 drafter 块（§1.4 的 `proj` 之后全部），结果 `out`/`x2`/`tok`(argmax) 留缓冲。
    /// `dep_row` = 深度 − 1（深度 1 ⇒ 0）。
    fn block(
        &mut self,
        be: &mut dyn ComputeBackend,
        u: TensorId,
        tok: TensorId,
        p: u32,
        dep_row: usize,
        with_draft: bool,
    ) -> R<()> {
        let (b, c) = (self.b, self.c);
        let (d, nh, nkv) = (MTP_HEAD_DIM, MTP_N_HEAD, MTP_N_KV_HEAD);

        // e = ln0(emb(token))  —— 直接用主模型的 ln0(emb) 表
        be.gather_row_device_f16(self.emb_ln, b.e, tok, c)?;
        be.norm(b.e, self.w.le.w, self.w.le.b, b.e_n, c, 1, self.ln_eps, 1)?;
        be.copy_part(b.cat2, 0, u, c)?;
        be.copy_part(b.cat2, c, b.e_n, c)?;
        self.w.proj.plain(be, b.cat2, b.x0, c, 2 * c)?;
        // + dep[k-1]
        let row = dep_row.min(self.w.dep.len() - 1);
        be.elementwise_add(b.x0, self.w.dep[row], b.x0, c)?;
        be.norm(b.x0, self.w.ln0.w, self.w.ln0.b, b.x, c, 1, self.ln_eps, 1)?;
        be.norm(b.x, self.w.ln1.w, self.w.ln1.b, b.xl, c, 1, self.ln_eps, 1)?;

        self.w.q.plain(be, b.xl, b.q, nh * d, c)?;
        self.w.g.plain(be, b.xl, b.g, nh * d, c)?;
        self.w.k.plain(be, b.xl, b.kk, nkv * d, c)?;
        self.w.v.plain(be, b.xl, b.vv, nkv * d, c)?;
        be.mtp_rms_rope(b.q, self.w.q_norm, nh, d, p, self.rope_dim, self.rope_theta)?;
        be.mtp_rms_rope(
            b.kk,
            self.w.k_norm,
            nkv,
            d,
            p,
            self.rope_dim,
            self.rope_theta,
        )?;
        be.mtp_kv_write(
            self.st.kc,
            self.st.vc,
            self.st.ring_pos,
            b.kk,
            b.vv,
            p,
            nkv,
            d,
            self.st.s_ring,
        )?;
        be.mtp_attn(
            b.q,
            self.st.kc,
            self.st.vc,
            self.st.ring_pos,
            b.y,
            p,
            self.window,
            nh,
            nkv,
            d,
            self.st.s_ring,
        )?;
        // x = x + Wo·(y ⊙ sigmoid(Wg·xl))
        be.elementwise_sigmoid(b.g, b.gs, nh * d, 1)?;
        be.elementwise_mul(b.y, b.gs, b.y, nh * d, 1)?;
        self.w.o.add(be, b.y, b.x, c, nh * d)?;

        // channel-mix
        be.norm(b.x, self.w.ln2.w, self.w.ln2.b, b.x2, c, 1, self.ln_eps, 1)?;
        be.elementwise_lerp(b.x2, b.shift, self.w.x_k, b.kx, c)?;
        self.w.ffn_key.relu2(be, b.kx, b.hh, 4 * c, c)?;
        self.w.ffn_value.add(be, b.hh, b.x, c, 4 * c)?;

        // out → draft（`with_draft=false` 时整段跳过：`accept` 的补提交不需要草稿）
        be.norm(b.x, self.w.lo.w, self.w.lo.b, b.out, c, 1, self.ln_eps, 1)?;
        if with_draft {
            self.head.gemv(be, b.out, b.logits, self.vocab, c)?;
            be.elementwise_add(b.logits, self.draft_mask, b.logits, self.vocab)?;
            be.argmax(b.logits, b.tok, self.vocab)?;
        }
        // 下一深度 / 下一次提交的 token shift
        be.copy_device(b.x2, b.shift)
    }
}

fn up_f16(be: &mut dyn ComputeBackend, t: &MtpTensor) -> R<TensorId> {
    let id = be.create_tensor(t.numel(), TensorDtype::F16)?;
    be.upload(id, &t.data)?;
    be.drop_host(id);
    Ok(id)
}

/// 把一个 `[m, k]` 行主序 f32 权重就地量化成 **int6 N-major** 平面并上传。
///
/// 打包规则必须与 CUDA 内核（`i6nm_dot` / `i6nm_dot_split`）的读法逐字节一致
/// （`backend_cuda.rs` 的 `int6_nmajor_kernels_match_cpu` 就是对拍门禁）：
/// · 每 128 个 k 一个对称 scale（`max|w|/32`，fp16 存进 `sz` 低 16 位）；
/// · `e = round(w/scale) + 32 ∈ [0,63]`（6-bit，`-32` 偏置）；
/// · **64-k 一个 chunk，chunk 内**：
///   `lo[32c+j][col]` 低 nibble = e(j)、高 nibble = e(j+32)；
///   `lo[32c+16+j][col]` 低/高 = e(j+16) / e(j+48)；
///   `hi[16c+j][col]` 的 4 个 2-bit 域依次 = e(j) / e(j+16) / e(j+32) / e(j+48) 的高 2 位。
///
/// 之所以在这里自己量化而不是读文件：sidecar 只有 fp16/bf16，drafter 的 int6 档是**加载期**
/// 的一次性转换（上游 `MTPDrafter.quantize` 同一做法）。
fn up_i6(be: &mut dyn ComputeBackend, t: &MtpTensor, m: usize, k: usize) -> R<Int6Handle> {
    if !k.is_multiple_of(128) || t.data.len() != m * k {
        return Err(format!(
            "up_i6: 形状不符（m={m} k={k}，数据 {}，k 需为 128 的倍数）",
            t.data.len()
        )
        .into());
    }
    let (kg, lo_rows, hi_rows) = (k / 128, k / 2, k / 4);
    let mut lo = vec![0u8; lo_rows * m];
    let mut hi = vec![0u8; hi_rows * m];
    let mut sz = vec![0u32; kg * m];
    let mut e = vec![0u8; k];
    for r in 0..m {
        let row = &t.data[r * k..(r + 1) * k];
        for g in 0..kg {
            let blk = &row[g * 128..(g + 1) * 128];
            let max = blk.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let scale = if max > 0.0 { max / 32.0 } else { 1.0 };
            sz[g * m + r] = half::f16::from_f32(scale).to_bits() as u32;
            for (i, &w) in blk.iter().enumerate() {
                let q = (w / scale).round() + 32.0;
                e[g * 128 + i] = q.clamp(0.0, 63.0) as u8;
            }
        }
        for c in 0..k / 64 {
            let base = c * 64;
            for j in 0..16 {
                let (e0, e1) = (e[base + j], e[base + j + 16]);
                let (e2, e3) = (e[base + j + 32], e[base + j + 48]);
                lo[(c * 32 + j) * m + r] = (e0 & 15) | ((e2 & 15) << 4);
                lo[(c * 32 + 16 + j) * m + r] = (e1 & 15) | ((e3 & 15) << 4);
                hi[(c * 16 + j) * m + r] = ((e0 >> 4) & 3)
                    | (((e1 >> 4) & 3) << 2)
                    | (((e2 >> 4) & 3) << 4)
                    | (((e3 >> 4) & 3) << 6);
            }
        }
    }
    let as_u32 = |x: &[u8]| -> Vec<u32> {
        x.chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()
    };
    let (lo_w, hi_w) = (as_u32(&lo), as_u32(&hi));
    let lo_t = be.create_tensor(lo_w.len(), TensorDtype::U32)?;
    be.upload_u32(lo_t, &lo_w)?;
    let hi_t = be.create_tensor(hi_w.len(), TensorDtype::U32)?;
    be.upload_u32(hi_t, &hi_w)?;
    let sz_t = be.create_tensor(sz.len(), TensorDtype::U32)?;
    be.upload_u32(sz_t, &sz)?;
    be.drop_host(lo_t);
    be.drop_host(hi_t);
    be.drop_host(sz_t);
    Ok(Int6Handle {
        lo: lo_t,
        hi: hi_t,
        sz: sz_t,
        m,
        k,
        nmajor: true,
    })
}

fn up_f32(be: &mut dyn ComputeBackend, t: &MtpTensor) -> R<TensorId> {
    let id = be.create_tensor(t.numel(), TensorDtype::F32)?;
    be.upload(id, &t.data)?;
    be.drop_host(id);
    Ok(id)
}
