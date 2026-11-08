//! MTP（multi-token prediction）投机解码。
//!
//! 规范来源：上游 `OpenMOSE/RWKV-Infer-Next` 的 `docs/mtp.md`（v1，2026-11-08 冻结）。
//! 本模块只负责**读 sidecar + 在 GPU 上跑 drafter**；投机步的「验证 / 回滚」在
//! `gpu_model.rs` 的 spec 路径里（P5）。
//!
//! 边界（与上游一致，别在这里加东西）：
//! - drafter **不决定输出**：它给的 draft 只影响速度，最终 token 一律由主模型采样得到；
//! - drafter 权重可量化（上游 int6），量化只改接受率、不改输出。

pub mod sidecar;

pub use sidecar::{MtpMeta, MtpSidecar, find_sidecar, load_sidecar};

/// drafter 固定超参（上游 §1.2：对任何 C 都一样）。
pub const MTP_N_HEAD: usize = 16;
pub const MTP_N_KV_HEAD: usize = 4;
pub const MTP_HEAD_DIM: usize = 128;
pub const MTP_ROPE_DIM: usize = 32;
pub const MTP_ROPE_THETA: f32 = 1.0e6;
pub const MTP_WINDOW: usize = 512;
pub const MTP_DEPTH_ROWS: usize = 8;
/// 默认 draft 词表截断（上游 §1.2：引擎只用前 16384 个高频 token）
pub const MTP_DRAFT_VOCAB: usize = 16384;
