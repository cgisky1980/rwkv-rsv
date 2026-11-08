//! MTP sidecar 读取与校验（上游 `docs/mtp.md` §2 的 v1 格式）。
//!
//! 只做「文件 → 主机侧张量 + meta」；显存上传与推理在 `drafter.rs`（后续阶段）。
//!
//! 支持三种形态（与上游 `rwkv_next/mtp.py: load_sidecar` 一致）：
//! 1. 目录：`mtp/mtp.safetensors` + `mtp/mtp.json`（`{"meta": {...}}`）；
//! 2. 单文件 `.safetensors`：header metadata 的 `mtp_meta` 键里放 meta JSON；
//! 3. 训练器 `.pt`：**明确报错**并提示用上游 `tools/export_mtp.py` 转换。
//!
//! 张量表（§1.3，C = 主模型宽度，H=16 / Hk=4 / D=128）：
//! `lh.{weight,bias}`、`lf.{0,1}.{weight,bias}`、`fproj.weight[C,3C]`、
//! `lhd.{weight,bias}`、`dep[8,C]`、`le.{weight,bias}`、`proj.weight[C,2C]`、
//! `blk.ln{0,1,2}.{weight,bias}`、`attn.{q,g}[2048,C]`、`attn.{k,v}[512,C]`、
//! `attn.o[C,2048]`、`attn.{q,k}_norm[128]`、`blk.ffn.x_k[1,1,C]`、
//! `blk.ffn.{key[4C,C],value[C,4C]}.weight`、`lo.{weight,bias}`、`draft_vocab[V] i64`。

use std::collections::HashMap;
use std::path::{Path, PathBuf};

use crate::runtime::R;

/// sidecar meta（上游 §2.2）。缺省值按规范填。
#[derive(Debug, Clone)]
pub struct MtpMeta {
    pub n_embd: usize,
    pub n_layer: usize,
    /// 两个 tap 层（0-based，取该层**输出残差**）。32 层模型 = [9, 19]，61 层 = [17, 36]。
    pub fuse_layers: [usize; 2],
    /// 注意力窗口 W（meta 里 `window` 或 `ctx`，缺省 512）。
    pub window: usize,
    /// 训练深度（`dep` 的有效行数，0 = 未知）。
    pub depth: usize,
    pub ln_eps: f32,
    pub rms_eps: f32,
    pub rope_dim: usize,
    pub rope_theta: f32,
    pub base_model: Option<String>,
}

impl Default for MtpMeta {
    fn default() -> Self {
        Self {
            n_embd: 0,
            n_layer: 0,
            fuse_layers: [0, 0],
            window: 512,
            depth: 3,
            ln_eps: 1.0e-5,
            rms_eps: 1.0e-6,
            rope_dim: 32,
            rope_theta: 1.0e6,
            base_model: None,
        }
    }
}

/// 一个 fp 权重（形状 + 行主序 f32 数据，bf16/f16/f32 都已展开成 f32）。
#[derive(Debug, Clone)]
pub struct MtpTensor {
    pub shape: Vec<usize>,
    pub data: Vec<f32>,
}

impl MtpTensor {
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// 读出来的 sidecar：meta + 全部 fp 张量 + draft 词表（**原始顺序**，未截断）。
#[derive(Debug)]
pub struct MtpSidecar {
    pub meta: MtpMeta,
    pub tensors: HashMap<String, MtpTensor>,
    /// `draft_vocab[0..]` = 按训练频率排序的 token id（引擎只用前 `MTP_DRAFT_VOCAB`）。
    pub draft_vocab: Vec<u32>,
    /// 实际读取的文件（诊断用）
    pub path: PathBuf,
}

/// 必备张量名（`blk.ln*` 等按 §1.3 的固定名）。`draft_vocab` 单独存。
pub const REQUIRED: [&str; 31] = [
    "lh.weight",
    "lh.bias",
    "lf.0.weight",
    "lf.0.bias",
    "lf.1.weight",
    "lf.1.bias",
    "fproj.weight",
    "lhd.weight",
    "lhd.bias",
    "dep",
    "le.weight",
    "le.bias",
    "proj.weight",
    "blk.ln0.weight",
    "blk.ln0.bias",
    "blk.ln1.weight",
    "blk.ln1.bias",
    "blk.ln2.weight",
    "blk.ln2.bias",
    "attn.q.weight",
    "attn.g.weight",
    "attn.k.weight",
    "attn.v.weight",
    "attn.o.weight",
    "attn.q_norm.weight",
    "attn.k_norm.weight",
    "blk.ffn.x_k",
    "blk.ffn.key.weight",
    "blk.ffn.value.weight",
    "lo.weight",
    "lo.bias",
];

/// 按 `RWKV_MTP_PATH` / 模型目录自动找 sidecar（上游 §2.4 的 `auto` 规则子集）。
///
/// `model_path` 是**主模型文件**的路径；sidecar 在它的同目录下按顺序找：
/// `mtp/mtp.safetensors` → 首个 `*.mtp.safetensors` → 首个 `*.mtp.pt`。
/// 返回 `None` 表示「没有 sidecar ⇒ MTP 关闭」（不是错误）。
pub fn find_sidecar(model_path: &str) -> R<Option<PathBuf>> {
    let spec = std::env::var("RWKV_MTP_PATH").unwrap_or_else(|_| "auto".into());
    let spec = spec.trim().to_string();
    match spec.as_str() {
        "" | "off" | "none" | "0" => return Ok(None),
        "auto" => {}
        p => {
            let pb = PathBuf::from(p);
            let resolved = if pb.is_dir() {
                pb.join("mtp.safetensors")
            } else {
                pb
            };
            if !resolved.exists() {
                return Err(format!("RWKV_MTP_PATH={p}: 找不到 sidecar（{resolved:?}）").into());
            }
            return Ok(Some(resolved));
        }
    }
    let dir = Path::new(model_path)
        .parent()
        .ok_or("find_sidecar: 模型路径没有父目录")?;
    let direct = dir.join("mtp").join("mtp.safetensors");
    if direct.exists() {
        return Ok(Some(direct));
    }
    let mut entries: Vec<PathBuf> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .collect();
    entries.sort();
    for pat in [".mtp.safetensors", ".mtp.pt"] {
        if let Some(p) = entries.iter().find(|p| p.to_string_lossy().ends_with(pat)) {
            return Ok(Some(p.clone()));
        }
    }
    Ok(None)
}

/// 读 sidecar（目录 / 单 `.safetensors` / `.pt`）。
pub fn load_sidecar(path: &Path) -> R<MtpSidecar> {
    if path.extension().and_then(|e| e.to_str()) == Some("pt") {
        return Err(format!(
            "MTP sidecar 是训练器格式（{}）：请先用上游 tools/export_mtp.py 转成 mtp/mtp.safetensors",
            path.display()
        )
        .into());
    }
    let st_path = if path.is_dir() {
        path.join("mtp.safetensors")
    } else {
        path.to_path_buf()
    };
    let bytes = std::fs::read(&st_path)
        .map_err(|e| format!("读 sidecar 失败 {}: {e}", st_path.display()))?;
    let st = safetensors::SafeTensors::deserialize(&bytes)
        .map_err(|e| format!("sidecar 不是合法 safetensors（{}）: {e}", st_path.display()))?;

    // meta：优先同目录 mtp.json，其次 safetensors header 的 `mtp_meta`（上游 §2.3 的单文件形态）。
    let meta_raw = read_meta_json(&st_path, &bytes)?;
    let meta = parse_meta(&meta_raw.unwrap_or_else(|| "{}".into()))?;

    let mut tensors = HashMap::new();
    let mut draft_vocab = Vec::new();
    for (name, view) in st.tensors() {
        if name == "draft_vocab" {
            draft_vocab = tensor_to_u32(&view)?;
            continue;
        }
        let shape: Vec<usize> = view.shape().to_vec();
        let data = tensor_to_f32(&view, &name)?;
        tensors.insert(name, MtpTensor { shape, data });
    }
    if draft_vocab.is_empty() {
        return Err("sidecar 缺少 draft_vocab（int64 词表顺序）".into());
    }
    let sc = MtpSidecar {
        meta,
        tensors,
        draft_vocab,
        path: st_path,
    };
    sc.validate_shapes()?;
    Ok(sc)
}

impl MtpSidecar {
    /// 校验与主模型的一致性（上游 §2.4 的加载期硬拒项）。
    pub fn check_against_model(&self, n_embd: usize, n_layer: usize, force: bool) -> R<()> {
        if self.meta.n_embd != n_embd {
            return Err(format!(
                "MTP sidecar 宽度不符：sidecar {} vs 模型 {n_embd}",
                self.meta.n_embd
            )
            .into());
        }
        if self.meta.n_layer != 0 && self.meta.n_layer != n_layer {
            let msg = format!(
                "MTP sidecar 层数不符：sidecar {} vs 模型 {n_layer}",
                self.meta.n_layer
            );
            if force {
                log::warn!("{msg}（MTP_FORCE=1，继续；仅速度受影响）");
            } else {
                return Err(msg.into());
            }
        }
        for &l in &self.meta.fuse_layers {
            if l >= n_layer {
                return Err(format!(
                    "MTP sidecar 的 fuse_layers={:?} 超出模型层数 {n_layer}",
                    self.meta.fuse_layers
                )
                .into());
            }
        }
        Ok(())
    }

    /// 形状契约（§1.3）。`n_embd` 来自 meta。
    fn validate_shapes(&self) -> R<()> {
        let c = self.meta.n_embd;
        if c == 0 {
            return Err("sidecar meta 缺 n_embd".into());
        }
        let want: Vec<(&str, Vec<usize>)> = vec![
            ("lh.weight", vec![c]),
            ("lh.bias", vec![c]),
            ("lf.0.weight", vec![c]),
            ("lf.0.bias", vec![c]),
            ("lf.1.weight", vec![c]),
            ("lf.1.bias", vec![c]),
            ("fproj.weight", vec![c, 3 * c]),
            ("lhd.weight", vec![c]),
            ("lhd.bias", vec![c]),
            ("dep", vec![8, c]),
            ("le.weight", vec![c]),
            ("le.bias", vec![c]),
            ("proj.weight", vec![c, 2 * c]),
            ("blk.ln0.weight", vec![c]),
            ("blk.ln0.bias", vec![c]),
            ("blk.ln1.weight", vec![c]),
            ("blk.ln1.bias", vec![c]),
            ("blk.ln2.weight", vec![c]),
            ("blk.ln2.bias", vec![c]),
            ("attn.q.weight", vec![2048, c]),
            ("attn.g.weight", vec![2048, c]),
            ("attn.k.weight", vec![512, c]),
            ("attn.v.weight", vec![512, c]),
            ("attn.o.weight", vec![c, 2048]),
            ("attn.q_norm.weight", vec![128]),
            ("attn.k_norm.weight", vec![128]),
            ("blk.ffn.x_k", vec![1, 1, c]),
            ("blk.ffn.key.weight", vec![4 * c, c]),
            ("blk.ffn.value.weight", vec![c, 4 * c]),
            ("lo.weight", vec![c]),
            ("lo.bias", vec![c]),
        ];
        for (name, shape) in want {
            let t = self
                .tensors
                .get(name)
                .ok_or_else(|| format!("sidecar 缺张量 {name}"))?;
            if t.shape != shape {
                return Err(format!(
                    "sidecar 张量 {name} 形状不符：{:?} vs 期望 {:?}",
                    t.shape, shape
                )
                .into());
            }
        }
        Ok(())
    }
}

/// 读 meta JSON 原文：同目录 `mtp.json` 优先，其次 safetensors header 的 `mtp_meta`。
///
/// safetensors 头 = 前 8 字节 little-endian 的 JSON 长度 + 那段 JSON（`__metadata__` 是字符串表）。
fn read_meta_json(st_path: &Path, bytes: &[u8]) -> R<Option<String>> {
    let json_path = st_path.with_file_name("mtp.json");
    if let Ok(s) = std::fs::read_to_string(&json_path) {
        return Ok(Some(s));
    }
    if bytes.len() < 8 {
        return Ok(None);
    }
    let n = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    if bytes.len() < 8 + n {
        return Ok(None);
    }
    let hdr = std::str::from_utf8(&bytes[8..8 + n])
        .map_err(|e| format!("safetensors header 非 UTF-8: {e}"))?;
    let v: serde_json::Value =
        serde_json::from_str(hdr).map_err(|e| format!("safetensors header JSON 解析失败: {e}"))?;
    Ok(v.get("__metadata__")
        .and_then(|m| m.get("mtp_meta"))
        .and_then(|s| s.as_str())
        .map(String::from))
}

/// 解析 `{"meta": {...}}`（也接受直接给 meta 对象）。
fn parse_meta(raw: &str) -> R<MtpMeta> {
    let v: serde_json::Value =
        serde_json::from_str(raw).map_err(|e| format!("sidecar meta JSON 解析失败: {e}"))?;
    let m = v.get("meta").unwrap_or(&v);
    let mut meta = MtpMeta::default();
    let get_usize = |k: &str| m.get(k).and_then(|x| x.as_u64()).map(|x| x as usize);
    let get_f32 = |k: &str| m.get(k).and_then(|x| x.as_f64()).map(|x| x as f32);
    meta.n_embd = get_usize("n_embd").ok_or("sidecar meta 缺 n_embd")?;
    meta.n_layer = get_usize("n_layer").unwrap_or(0);
    if let Some(fl) = m.get("fuse_layers").and_then(|x| x.as_array()) {
        if fl.len() != 2 {
            return Err("sidecar meta 的 fuse_layers 必须是两个元素".into());
        }
        meta.fuse_layers = [
            fl[0].as_u64().ok_or("fuse_layers[0] 非整数")? as usize,
            fl[1].as_u64().ok_or("fuse_layers[1] 非整数")? as usize,
        ];
    }
    meta.window = get_usize("window")
        .or_else(|| get_usize("ctx"))
        .unwrap_or(512);
    meta.depth = get_usize("depth").unwrap_or(3);
    meta.ln_eps = get_f32("ln_eps").unwrap_or(1.0e-5);
    meta.rms_eps = get_f32("rms_eps").unwrap_or(1.0e-6);
    if let Some(a) = m.get("attn") {
        meta.rope_dim = a
            .get("rope_dim")
            .and_then(|x| x.as_u64())
            .map(|x| x as usize)
            .unwrap_or(32);
        meta.rope_theta = a
            .get("rope_theta")
            .and_then(|x| x.as_f64())
            .map(|x| x as f32)
            .unwrap_or(1.0e6);
    }
    meta.base_model = m
        .get("base_model")
        .and_then(|x| x.as_str())
        .map(String::from);
    Ok(meta)
}

/// bf16/f16/f32 → f32（safetensors 只给我们原始字节）。
fn tensor_to_f32(v: &safetensors::tensor::TensorView, name: &str) -> R<Vec<f32>> {
    use safetensors::Dtype;
    let d = v.data();
    Ok(match v.dtype() {
        Dtype::F32 => d
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F16 => d
            .chunks_exact(2)
            .map(|c| half::f16::from_bits(u16::from_le_bytes([c[0], c[1]])).to_f32())
            .collect(),
        Dtype::BF16 => d
            .chunks_exact(2)
            .map(|c| {
                // bf16 → f32：低 16 位补零
                f32::from_bits((u16::from_le_bytes([c[0], c[1]]) as u32) << 16)
            })
            .collect(),
        other => {
            return Err(
                format!("张量 {name} 的 dtype 不支持: {other:?}（期望 bf16/f16/f32）").into(),
            );
        }
    })
}

/// int64 词表 → u32（越界/负数视为格式错误）。
fn tensor_to_u32(v: &safetensors::tensor::TensorView) -> R<Vec<u32>> {
    use safetensors::Dtype;
    let d = v.data();
    let to_u32 = |x: i64| -> R<u32> {
        u32::try_from(x).map_err(|_| format!("draft_vocab 里出现越界值 {x}").into())
    };
    let out: R<Vec<u32>> = match v.dtype() {
        Dtype::I64 => d
            .chunks_exact(8)
            .map(|c| {
                to_u32(i64::from_le_bytes([
                    c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                ]))
            })
            .collect(),
        Dtype::I32 => d
            .chunks_exact(4)
            .map(|c| to_u32(i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as i64))
            .collect(),
        other => return Err(format!("draft_vocab 的 dtype 不支持: {other:?}").into()),
    };
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    /// 用本机真实 sidecar 验证解析（文件不存在则跳过）。
    ///
    /// 钉住四件事：① meta 关键字段（宽度/层数/taps/window/depth）；② 31 个张量齐全且形状
    /// 与 §1.3 完全一致；③ bf16 展开后的数值范围合理（LayerNorm 权重均值≈1、无 NaN）；
    /// ④ draft_vocab 是 0..V-1 的**排列**（引擎靠它把 argmax 映射回 token id）。
    #[test]
    fn load_real_sidecar_if_present() {
        let p = r"c:\work\ai00-x-dev\.ai00-x-dev\rwkv-build\mtp";
        let dir = Path::new(p);
        if !dir.join("mtp.safetensors").exists() {
            log::info!("MTP sidecar 不在本机，跳过");
            return;
        }
        let sc = load_sidecar(dir).expect("load sidecar");
        assert_eq!(sc.meta.n_embd, 2560);
        assert_eq!(sc.meta.n_layer, 32);
        assert_eq!(sc.meta.fuse_layers, [9, 19]);
        assert_eq!(sc.meta.window, 512);
        assert_eq!(sc.meta.depth, 3);
        for name in REQUIRED {
            assert!(sc.tensors.contains_key(name), "缺张量 {name}");
        }
        // LayerNorm 权重：训练后均值应接近 1 且全为正
        let ln = &sc.tensors["blk.ln0.weight"];
        assert_eq!(ln.shape, vec![2560]);
        assert!(ln.data.iter().all(|v| v.is_finite() && *v > 0.0));
        let mean = ln.data.iter().map(|v| *v as f64).sum::<f64>() / ln.data.len() as f64;
        assert!((0.5..2.0).contains(&mean), "ln0.weight 均值异常: {mean}");
        // draft_vocab 必须是 0..V-1 的排列
        let v = sc.draft_vocab.len();
        let mut sorted = sc.draft_vocab.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), v, "draft_vocab 有重复");
        assert_eq!(*sorted.first().unwrap(), 0);
        assert_eq!(*sorted.last().unwrap() as usize, v - 1);
        log::info!(
            "MTP sidecar OK：{} 个张量，draft_vocab {v}，{:.1} MB(fp32 展开)",
            sc.tensors.len(),
            sc.tensors.values().map(|t| t.numel() * 4).sum::<usize>() as f64 / 1e6
        );
    }
}
