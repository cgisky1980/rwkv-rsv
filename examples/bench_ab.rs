//! 同会话**交错 A/B** 编排器：把 `batch_decode_bench` 在多个后端 × 多个并发档之间交替跑，
//! 每档重复若干轮，聚合成带门控的可信读数表（Markdown + JSON 摘要）。
//!
//! ## 为什么需要它（改进项 1：基准可信度）
//!
//! README §6.1 的老表是**两次不同会话**的读数拼起来的（信天翁+CUDA 一次、Vulkan+CUDA 一次），
//! 而这块卡兼作桌面显示卡 —— 跨会话漂移实测可达 ±17%，单次读数足以把结论反转 ±3%。
//! 更糟的是配对读数**没记门控**：CUDA 侧 `LOWRANK_GEMM` / `FFN_VALUE_GEMM` / `GEMV_IMMA`
//! 三个张量核开关默认全开，于是「SIMT 对 SIMT」被读成「Vulkan 追平了张量核路径」。
//!
//! 本编排器针对这两点做了三件事：
//! 1. **一次会话内**跑完所有后端与并发档，且**每轮轮换测量顺序**（第 r 轮把列表旋转 r 格、
//!    奇数轮反向）—— 这样「谁先被测」不固定，桌面负载/升温带来的偏置不会系统性压在某一方。
//! 2. **严格串行**执行子进程：一张物理 GPU 上并发跑两个后端，读数必然是垃圾。
//! 3. 每个读数都由 `BENCH_JSON` 与**它自己的门控**绑成一条记录（见 `gating_json()`），
//!    摘要表里把门控原样打印出来，引用时不可能漏。
//!
//! 另外汇总**跨后端指纹是否一致**（`batch_tokens_sum`）—— 这正是 README 里那句
//! 「每一档批量两后端逐位一致」需要的证据，口径（门控）随表一起出。
//!
//! ## 用法
//! ```bash
//! # 先建好 example 产物
//! cargo build --release --examples
//! # 三档并发 × 两个后端 × 3 轮，父进程的环境（含门控旋钮）原样透传
//! AB_BACKENDS=cuda,vulkan AB_SLOTS=8,16,32 AB_ROUNDS=3 \
//!   cargo run --release --example bench_ab
//! ```
//!
//! | 环境变量 | 默认 | 说明 |
//! |---|---|---|
//! | `AB_BACKENDS` | `cuda,vulkan` | 参与 A/B 的后端，第一个是比值基准 |
//! | `AB_SLOTS` | `8,16,32` | 并发档（每档一个子进程） |
//! | `AB_ROUNDS` | `3` | 每档重复轮数（≥3 才谈得上中位数与离散度） |
//! | `AB_JSON` | `bench/ab-<会话 id>.jsonl` | 全部单条读数（含门控）落盘位置 |
//! | `AB_SUMMARY` | 同上后缀 `.summary.json` | 聚合摘要 JSON |
//! | `AB_BENCH_BIN` | 自动在 `target/release/examples/` 里找 | 显式指定 bench 可执行文件 |
//!
//! 其余基准参数（`SEGS` / `NTOK` / `PAD_TO` / `TOPK` / `PROMPT_REPEAT` / `WARMUP` / `STEP` /
//! `GEMV_IMMA` / `LOWRANK_GEMM` / `FFN_VALUE_GEMM` …）**由父进程环境透传** —— 编排器不改门控，
//! 只改 `BACKEND` / `BENCH_JSON` / `BENCH_SESSION` / `BENCH_ROUND` / `SLOTS`。

use std::collections::BTreeMap;
use std::error::Error;
use std::path::{Path, PathBuf};
use std::process::Command;

/// 单条读数的键：(后端, 并发档)。
type Key = (String, usize);

fn main() -> Result<(), Box<dyn Error>> {
    let backends = parse_list::<String>(
        &std::env::var("AB_BACKENDS").unwrap_or_else(|_| "cuda,vulkan".into()),
        |s| s.to_string(),
    );
    let slots_list: Vec<usize> = std::env::var("AB_SLOTS")
        .unwrap_or_else(|_| "8,16,32".into())
        .split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(|s| {
            s.parse::<usize>()
                .map_err(|e| format!("AB_SLOTS 非法项 {s}: {e}").into())
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;
    let rounds: usize = std::env::var("AB_ROUNDS")
        .unwrap_or_else(|_| "3".into())
        .parse()?;
    if backends.is_empty() || slots_list.is_empty() || rounds == 0 {
        return Err("AB_BACKENDS / AB_SLOTS 非空且 AB_ROUNDS ≥ 1".into());
    }
    let session = format!(
        "{}-{}",
        chrono::Local::now().format("%Y%m%d-%H%M%S"),
        std::process::id()
    );
    let out_dir = PathBuf::from("bench");
    std::fs::create_dir_all(&out_dir)?;
    let json_path = PathBuf::from(std::env::var("AB_JSON").unwrap_or_else(|_| {
        out_dir
            .join(format!("ab-{session}.jsonl"))
            .display()
            .to_string()
    }));
    let summary_path = std::env::var("AB_SUMMARY")
        .unwrap_or_else(|_| format!("{}.summary.json", json_path.display()));

    // 每个子进程写自己的 JSONL，避免并发追加交错（这里其实串行，但分文件更好排错）。
    let bench_bin = locate_bench_bin()?;
    println!(
        "同会话交错 A/B：session={session}\n  bench={}",
        bench_bin.display()
    );
    println!(
        "  backends={:?} slots={:?} rounds={rounds}",
        backends, slots_list
    );
    println!("  门控（父进程环境透传，未设=走代码默认档）:");
    for (k, d) in GATED_KNOBS {
        println!(
            "    {k} = {}",
            std::env::var(k).unwrap_or_else(|_| format!("(未设，默认: {d})"))
        );
    }

    let mut plan: Vec<(String, usize)> = Vec::new();
    for b in &backends {
        for &s in &slots_list {
            plan.push((b.clone(), s));
        }
    }

    let mut cells: BTreeMap<Key, Vec<serde_json::Value>> = BTreeMap::new();
    for r in 1..=rounds {
        // 轮换顺序：第 r 轮旋转 r-1 格，奇数轮反向 —— 「谁先被测」不固定。
        let mut order = plan.clone();
        let rot = (r - 1) % order.len();
        order.rotate_left(rot);
        if r % 2 == 0 {
            order.reverse();
        }
        for (backend, slots) in &order {
            print!("  round{r} {backend} B={slots}: ");
            // 单档失败（显存不够、子进程崩）只弃这一格，不 abort 整轮 —— 一趟 40 多个
            // 子进程的扫描跑到最后一档才崩，前面的读数就全废了。
            match run_one(&bench_bin, backend, *slots, r, &session, &json_path) {
                Ok(line) => {
                    let tok_s = line
                        .get("readings")
                        .and_then(|v| v.get("batch_tok_s"))
                        .and_then(|v| v.as_f64());
                    match tok_s {
                        Some(v) => {
                            println!("{v:.1} tok/s");
                            cells
                                .entry((backend.clone(), *slots))
                                .or_default()
                                .push(line);
                        }
                        None => println!("读数缺 batch_tok_s（弃此格）"),
                    }
                }
                Err(e) => println!("失败: {e}"),
            }
        }
    }

    let (md, summary) = aggregate(&backends, &cells, &session, rounds);
    std::fs::write(&json_path, append_gating_header(&json_path, &summary)?)?;
    println!("\n单条读数（含门控）: {}", json_path.display());
    println!("{md}");
    std::fs::write(&summary_path, serde_json::to_string_pretty(&summary)?)?;
    println!("聚合摘要: {summary_path}");
    Ok(())
}

/// 会被「忘记记录」从而让读数变得不可解释的门控旋钮（与 `gating_json()` 同一份口径）。
const GATED_KNOBS: [(&str, &str); 6] = [
    ("GEMV_IMMA", "开 —— CUDA r/k/v 走 int8 张量核"),
    ("LOWRANK_GEMM", "开 —— CUDA 低秩四条走 int8 张量核"),
    ("FFN_VALUE_GEMM", "开 —— CUDA FFN value 走 int8 张量核"),
    ("VK_IMMA_OPS", "除 rkv 外全开"),
    ("SKIP_GRAPH", "关 —— 自循环走图捕获/单批录制"),
    ("SELFLOOP_CHUNK", "内核默认分段"),
];

/// 跑一个子进程并把它的 JSONL 行读回来。
fn run_one(
    bench_bin: &Path,
    backend: &str,
    slots: usize,
    round: usize,
    session: &str,
    json_path: &Path,
) -> Result<serde_json::Value, Box<dyn Error>> {
    let before = file_len(json_path);
    let status = Command::new(bench_bin)
        .env("BACKEND", backend)
        .env("SLOTS", slots.to_string())
        .env("BENCH_JSON", json_path)
        .env("BENCH_SESSION", session)
        .env("BENCH_ROUND", round.to_string())
        // 单流值是常量，扫并发时重复跑纯属浪费（编排器只关心批量列）。
        .env("SKIP_SINGLE", "1")
        .status()?;
    if !status.success() {
        return Err(format!("{backend} B={slots} round{round} 子进程退出 {status}").into());
    }
    let text = read_from(json_path, before)?;
    let line = text
        .lines()
        .filter_map(|l| serde_json::from_str::<serde_json::Value>(l).ok())
        .next_back()
        .ok_or_else(|| format!("{backend} B={slots} round{round} 没有产出 JSON 读数"))?;
    Ok(line)
}

/// 聚合：每格给 中位数 / 轮数 / 离散度，并给相对基准后端的比值；跨后端指纹一致性单列。
fn aggregate(
    backends: &[String],
    cells: &BTreeMap<Key, Vec<serde_json::Value>>,
    session: &str,
    rounds: usize,
) -> (String, serde_json::Value) {
    let mut md = String::new();
    let mut summary = serde_json::json!({
        "session": session,
        "rounds": rounds,
        "backends": backends,
        "gating": gating_snapshot(),
        "cells": [],
        "cross_backend_fingerprints": [],
    });
    let mut cells_out = Vec::new();
    // 逐并发档一行，列 = 各后端中位数 tok/s + 相对基准后端倍率。
    let mut header = String::from("| 并发 B |");
    for b in backends {
        header.push_str(&format!(" {b} tok/s（中位数） | 离散度 |"));
    }
    if backends.len() > 1 {
        for i in 1..backends.len() {
            header.push_str(&format!(" {} / {} |", backends[i], backends[0]));
        }
    }
    header.push_str(" 跨后端指纹 |");
    md.push_str(&header);
    md.push_str("\n|---:|");
    for _ in 0..backends.len() {
        md.push_str("---:|---:|");
    }
    if backends.len() > 1 {
        for _ in 1..backends.len() {
            md.push_str("---:|");
        }
    }
    md.push_str("---|\n");

    let mut slots_seen: Vec<usize> = cells.keys().map(|k| k.1).collect();
    slots_seen.sort_unstable();
    slots_seen.dedup();
    for &s in &slots_seen {
        let mut row = format!("| {s} |");
        let mut medians: Vec<f64> = Vec::new();
        let mut fps: Vec<String> = Vec::new();
        for b in backends {
            let Some(recs) = cells.get(&(b.clone(), s)) else {
                row.push_str(" - | - |");
                medians.push(f64::NAN);
                continue;
            };
            let vals: Vec<f64> = recs
                .iter()
                .filter_map(|r| r.get("readings")?.get("batch_tok_s")?.as_f64())
                .collect();
            let med = median(&vals);
            let spread = spread_pct(&vals);
            medians.push(med);
            row.push_str(&format!(" {med:.1} | {spread:.1}% |"));
            let fp = recs
                .last()
                .and_then(|r| r.get("readings")?.get("batch_tokens_sum")?.as_str())
                .unwrap_or("-")
                .to_string();
            fps.push(format!("{b}={fp}"));
            cells_out.push(serde_json::json!({
                "backend": b, "slots": s, "rounds": vals.len(),
                "median_tok_s": med, "min_tok_s": vals.iter().cloned().fold(f64::INFINITY, f64::min),
                "max_tok_s": vals.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
                "spread_pct": spread, "readings": vals,
                "tokens_sum": fp,
            }));
        }
        if backends.len() > 1 {
            for i in 1..backends.len() {
                let ratio = if medians[0] > 0.0 && !medians[i].is_nan() {
                    medians[i] / medians[0]
                } else {
                    f64::NAN
                };
                row.push_str(&format!(" {ratio:.3}× |"));
            }
        }
        // 指纹一致性：只比较**同一次会话内**的各家指纹（口径见表头门控）。
        let all_same = fps.len() > 1
            && fps
                .iter()
                .all(|f| f.split('=').nth(1) == Some(fps[0].split('=').nth(1).unwrap_or("")));
        let fp_cell = match fps.len() {
            0 | 1 => "—".to_string(),
            _ if all_same => "一致".to_string(),
            _ => "**分叉**".to_string(),
        };
        row.push_str(&format!(" {fp_cell} |"));
        md.push_str(&row);
        md.push('\n');
        if let Some(a) = summary
            .get_mut("cross_backend_fingerprints")
            .and_then(|v| v.as_array_mut())
        {
            a.push(serde_json::json!({ "slots": s, "fingerprints": fps, "agree": all_same }));
        }
    }
    if let Some(a) = summary.get_mut("cells").and_then(|v| v.as_array_mut()) {
        *a = std::mem::take(&mut cells_out);
    }
    // 离散度警戒：README 说单次读数能把结论反转 ±3%。
    let worst = summary
        .get("cells")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|c| c.get("spread_pct").and_then(|v| v.as_f64()))
                .fold(0.0f64, f64::max)
        })
        .unwrap_or(0.0);
    md.push_str(&format!(
        "\n> 会话 `{session}`，每档 {rounds} 轮取中位数；最大离散度 {worst:.1}%\
         （>3% 说明单次读数不足以定结论）。\n"
    ));
    (md, summary)
}

fn gating_snapshot() -> serde_json::Value {
    let mut m = serde_json::Map::new();
    for (k, d) in GATED_KNOBS {
        m.insert(
            k.to_string(),
            serde_json::Value::String(
                std::env::var(k).unwrap_or_else(|_| format!("(未设，默认: {d})")),
            ),
        );
    }
    for k in [
        "SEGS",
        "NTOK",
        "PAD_TO",
        "TOPK",
        "PROMPT_REPEAT",
        "WARMUP",
        "STEP",
        "PENALTY",
        "SAMPLE_TEMP",
    ] {
        if let Ok(v) = std::env::var(k) {
            m.insert(k.to_string(), serde_json::Value::String(v));
        }
    }
    serde_json::Value::Object(m)
}

/// 在文件头补一行门控说明（JSONL 里以 `_comment` 键的 JSON 行表示，解析器忽略非读数行）。
fn append_gating_header(
    json_path: &Path,
    summary: &serde_json::Value,
) -> Result<String, Box<dyn Error>> {
    let body = std::fs::read_to_string(json_path).unwrap_or_default();
    let head =
        serde_json::json!({ "_comment": "门控快照（本会话）", "gating": summary.get("gating") });
    Ok(format!("{head}\n{body}"))
}

/// 定位 `batch_decode_bench` 的可执行文件（Windows 名字带 cargo 的哈希后缀，只能枚举）。
fn locate_bench_bin() -> Result<PathBuf, Box<dyn Error>> {
    if let Ok(p) = std::env::var("AB_BENCH_BIN") {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe()?;
    let dir = exe
        .parent()
        .filter(|p| p.file_name().map(|n| n == "examples").unwrap_or(false))
        .map(|p| p.to_path_buf())
        .ok_or_else(|| {
            format!(
                "无法从 current_exe({}) 推断 examples 目录；请设 AB_BENCH_BIN，\
                 或先 `cargo build --release --examples`",
                exe.display()
            )
        })?;
    let mut best: Option<(std::time::SystemTime, PathBuf)> = None;
    for entry in std::fs::read_dir(&dir)? {
        let entry = entry?;
        let name = entry.file_name().to_string_lossy().to_string();
        // 名字有两种形态：`batch_decode_bench.exe`（本机工具链）与
        // `batch_decode_bench-<hash>.exe`（带元数据哈希的构建）。都要接住。
        if !name.starts_with("batch_decode_bench") {
            continue;
        }
        let is_exe = entry
            .path()
            .extension()
            .map(|e| e == "exe")
            .unwrap_or(!cfg!(windows));
        if !is_exe {
            continue;
        }
        if let Ok(m) = entry.metadata().and_then(|md| md.modified())
            && best.as_ref().is_none_or(|(t, _)| m > *t)
        {
            best = Some((m, entry.path()));
        }
    }
    best.map(|(_, p)| p).ok_or_else(|| {
        format!(
            "{} 下找不到 batch_decode_bench*.exe，先跑 `cargo build --release --examples`",
            dir.display()
        )
        .into()
    })
}

fn file_len(p: &Path) -> u64 {
    std::fs::metadata(p).map(|m| m.len()).unwrap_or(0)
}

/// 从给定字节偏移读到文件末尾（只取本次子进程新追加的那几行）。
fn read_from(p: &Path, offset: u64) -> Result<String, Box<dyn Error>> {
    use std::io::{Read, Seek, SeekFrom};
    let mut f = std::fs::File::open(p)?;
    f.seek(SeekFrom::Start(offset))?;
    let mut buf = Vec::new();
    f.read_to_end(&mut buf)?;
    Ok(String::from_utf8_lossy(&buf).into_owned())
}

fn parse_list<T>(spec: &str, conv: impl Fn(&str) -> T) -> Vec<T> {
    spec.split(',')
        .map(|s| s.trim())
        .filter(|s| !s.is_empty())
        .map(conv)
        .collect()
}

fn median(v: &[f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    let mut x = v.to_vec();
    x.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = x.len();
    if n % 2 == 1 {
        x[n / 2]
    } else {
        (x[n / 2 - 1] + x[n / 2]) / 2.0
    }
}

/// 离散度 = (max − min) / min × 100%。
fn spread_pct(v: &[f64]) -> f64 {
    if v.len() < 2 {
        return 0.0;
    }
    let min = v.iter().cloned().fold(f64::INFINITY, f64::min);
    let max = v.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    if min > 0.0 {
        (max - min) / min * 100.0
    } else {
        0.0
    }
}
