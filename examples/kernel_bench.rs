//! 隔离内核计时基准（**不加载模型**）：对指定算子 + 形状输出 ms/call 与有效带宽。
//!
//! 为什么需要它：`batch_decode_bench` 是端到端的，改动某个内核后要跑完整推理才能看
//! 到收益；而 `PROF_CUDA_KERNEL` 的占比**不可信**——它在每个 kernel 后
//! `cuEventSynchronize`，实测 batch 段"内核耗时总和"≈ kernel 数 × 单次同步开销
//! （见 `backend_cuda.rs::launch_smem` 的注释）。本基准用「连续 N 次调用 + 末尾一次
//! 同步」计量，得到的是该算子的真实单次成本。
//!
//! 用法：
//! ```powershell
//! cargo run --release --example kernel_bench
//! $env:KB_OPS="relu2,mul_add"; $env:KB_BATCH="8,16"; cargo run --release --example kernel_bench
//! ```
//!
//! 环境变量（空字符串视为未设置，沿用项目惯例）：
//!   `KB_OPS`     逗号分隔算子，默认 `relu2,mul_add,plain,add`；可加 `f16`（fp16 权重）
//!                与 `imma_key,imma_out,imma_value,imma_head`（W8A8 张量核）
//!   `KB_BATCH`   逗号分隔 batch，默认 `1,8,16`
//!   `KB_ITERS`   每组迭代次数，默认 300
//!   `KB_WARMUP`  预热次数，默认 5
//!   `KB_REPEATS` 重复轮数，默认 5，**取各轮最小值**（最小值最不易被桌面占卡/
//!                时钟波动污染；实测单轮均值在 0.03ms 级小内核上抖动可达 68%，
//!                取最小后降到个位数百分比）
//!   `KB_C` / `KB_FH` / `KB_VOCAB`  模型维度，默认 2560 / 10240 / 65536
//!
//! 形状按算子语义固定（与 `gpu_model.rs::forward_layer_batch` 的调用点一致）：
//!   `relu2`   = ffn.key      m=fh    k=c      （`gemv_int8_relu2`，单次最贵）
//!   `mul_add` = att.output   m=c     k=c      （`gemv_int8_mul_add`）
//!   `plain`   = head         m=vocab k=c      （`gemv_int8_plain`，读取量最大）
//!   `add`     = ffn.value    m=c     k=fh     （`gemv_int8_add`，稠密回退）
//!   `f16`     = head(fp16)   m=vocab k=c      （`gemv_f16`，对照组）
//!
//! ★ `imma_*` = **W8A8 张量核路径**（`quant_x_i8` + `gemm_imma`，含 split-K 时的
//! `gemm_imma_reduce`），与 SIMT 的四个平坦算子一一对应：
//!   `imma_key`   m=fh    k=c    OP=0   `imma_out`   m=c k=c OP=1（带门控）
//!   `imma_value` m=c     k=fh   OP=1   `imma_head`  m=vocab k=c OP=2
//! 部分和容量按**引擎真实口径**给（`8·max(c,fh)·batch`，即 `WorkBuffers::imma_partial`），
//! 这样 `imma_ksplit` 在这个基准里选出的 ks 与真实推理一致。
//! 为什么需要它：端到端 `batch_decode_bench` 里改一个内核要跑完整推理才看得到收益，
//! 而 `PROF_GPU` 在自循环段只能记录到一部分 dispatch（占比不可用于 A/B）。
//! 大 batch 的差距主体在这些 GEMM 形状上，所以这里直接量 **TOPS**
//! （`2·m·k·batch / ms`）——它是张量核内核唯一的可比口径。
//!
//! **未覆盖**：`gemv_int8_rkv_stage1_batch`（26 参）与 `gemv_lowrank_chain4_batch`（20 参）
//! ——参数表过长，待 Phase 1 主战场（4 个平坦算子）收敛后再补。

use std::time::Instant;

use rwkv_rsv::backend::{
    ComputeBackend, Int8Handle, TensorDtype, TensorId, create_backend, detect_backend,
};

/// 逗号分隔环境变量解析（空字符串 = 未设置；历史教训：空串会被 `is_ok()` 判真）。
fn env_list(key: &str, default: &str) -> Vec<String> {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .unwrap_or_else(|| default.to_string())
        .split(',')
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
        .collect()
}

fn env_usize(key: &str, default: usize) -> usize {
    std::env::var(key)
        .ok()
        .filter(|v| !v.trim().is_empty())
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

/// 极简 LCG（只用于填充数据，避免引入依赖）。
struct Rng(u64);
impl Rng {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) * 2.0 - 1.0
    }
}

/// 一个待测用例：算子名 + (m, k) + 权重字节数（单次读）。
struct Case {
    op: String,
    m: usize,
    k: usize,
    /// `true` ⇒ 走 W8A8 张量核（`quant_x_i8` + `gemm_imma`）。
    imma: bool,
    /// `gemm_imma` 的 epilogue 选择（0 relu²、1 累加、2 覆盖写）。
    icode: u32,
    /// `quant_x_i8` 是否带门控（`att.output` 的 `y_norm · g`）。
    xgate: bool,
    /// `true` ⇒ 走**批量 prefill 的稠密 fp16 GEMM**（`gemm`：`C[rows,n] = A[rows,k] @ B[n,k]^T`）。
    /// 这一族与 `imma_*` 的语义不同：`batch` 那一列在这里是**行数 = token 数**（`B·T_pad`），
    /// `m` 是输出特征数。之所以要有它：`supports_int8_imma()` 在 CUDA 后端是 false，
    /// `imma_*` 只能量 Vulkan，跨后端比不了；而 prefill 的 86 % 时间在这条 `gemm` 上，
    /// **两后端都实现** ⇒ 这是唯一能在同精度、同形状下直接对拍 Vulkan/CUDA 的口径。
    pf16: bool,
}

impl Case {
    /// int8 权重字节 = idx(m*k, 每元素 1B) + sz(m*(k/128), 每组 4B)。
    fn wbytes_int8(&self) -> f64 {
        (self.m * self.k + self.m * (self.k / 128) * 4) as f64
    }
    /// fp16 权重字节 = m*k*2。
    fn wbytes_f16(&self) -> f64 {
        (self.m * self.k * 2) as f64
    }
    /// 张量核的标量吞吐口径：`2·m·k·batch / ms`（含 split-K 归约，是「端到端有效 TOPS」）。
    fn tops(&self, batch: usize, ms: f64) -> f64 {
        2.0 * self.m as f64 * self.k as f64 * batch as f64 / (ms / 1000.0) / 1e12
    }
}

fn main() {
    let c = env_usize("KB_C", 2560);
    let fh = env_usize("KB_FH", 10240);
    let vocab = env_usize("KB_VOCAB", 65536);
    let batches: Vec<usize> = env_list("KB_BATCH", "1,8,16")
        .iter()
        .filter_map(|s| s.parse().ok())
        .collect();
    let ops = env_list("KB_OPS", "relu2,mul_add,plain,add");
    let iters = env_usize("KB_ITERS", 300);
    let warmup = env_usize("KB_WARMUP", 5);
    let repeats = env_usize("KB_REPEATS", 5).max(1);

    let cases: Vec<Case> = ops
        .iter()
        .map(|op| match op.as_str() {
            "relu2" => Case {
                op: op.clone(),
                m: fh,
                k: c,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: false,
            },
            "mul_add" => Case {
                op: op.clone(),
                m: c,
                k: c,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: false,
            },
            "plain" => Case {
                op: op.clone(),
                m: vocab,
                k: c,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: false,
            },
            "add" => Case {
                op: op.clone(),
                m: c,
                k: fh,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: false,
            },
            "f16" => Case {
                op: op.clone(),
                m: vocab,
                k: c,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: false,
            },
            // ---- W8A8 张量核四个形状（与上面 SIMT 口径一一对应）----
            "imma_key" => Case {
                op: op.clone(),
                m: fh,
                k: c,
                imma: true,
                icode: 0,
                xgate: false,
                pf16: false,
            },
            "imma_out" => Case {
                op: op.clone(),
                m: c,
                k: c,
                imma: true,
                icode: 1,
                xgate: true,
                pf16: false,
            },
            "imma_value" => Case {
                op: op.clone(),
                m: c,
                k: fh,
                imma: true,
                icode: 1,
                xgate: false,
                pf16: false,
            },
            "imma_head" => Case {
                op: op.clone(),
                m: vocab,
                k: c,
                imma: true,
                icode: 2,
                xgate: false,
                pf16: false,
            },
            // ---- 批量 prefill 的稠密 fp16 GEMM（`gemm`）：与上面 imma_* 同形状，但那一列
            //      在这里是 token 行数 ----
            "pf_out" => Case {
                op: op.clone(),
                m: c,
                k: c,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: true,
            },
            "pf_key" => Case {
                op: op.clone(),
                m: fh,
                k: c,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: true,
            },
            "pf_value" => Case {
                op: op.clone(),
                m: c,
                k: fh,
                imma: false,
                icode: 0,
                xgate: false,
                pf16: true,
            },
            other => {
                eprintln!(
                    "未知算子 '{other}'（可选 relu2/mul_add/plain/add/f16 \
                     与 imma_key/imma_out/imma_value/imma_head、pf_out/pf_key/pf_value）"
                );
                std::process::exit(2);
            }
        })
        .collect();

    println!(
        "模型维度 c={c} fh={fh} vocab={vocab}；iters={iters} warmup={warmup} repeats={repeats}"
    );
    println!("kernel_bench: 隔离计时（不加载模型），有效带宽按「权重单次读」计");
    println!("口径：每轮 iters 次调用 + 一次同步，取 repeats 轮的最小值；抖动 = (最大-最小)/最小");
    println!();

    let backend = match create_backend(detect_backend()) {
        Ok(b) => b,
        Err(e) => {
            eprintln!("创建后端失败: {e:#}");
            std::process::exit(1);
        }
    };
    let mut b = backend;

    println!(
        "{:<11} {:>7} {:>6} {:>6} {:>11} {:>12} {:>10} {:>9} {:>8} {:>12}",
        "op", "m", "k", "batch", "ms/call", "权重 MB", "有效 GB/s", "TOPS", "抖动", "sum(y)"
    );
    println!("{}", "-".repeat(86));

    // split-K 部分和容量与引擎一致：`8·max(c,fh)·batch`（`WorkBuffers::imma_partial`）。
    let cap_rows = 8 * fh.max(c);
    for case in &cases {
        // 维度按内核假设校验：m 需是 4 的倍数（GEMV_ROWS），k 需是 128 的倍数（int8 分组）。
        if case.m % 4 != 0 || case.k % 128 != 0 {
            eprintln!(
                "跳过 {} m={} k={}：m 需为 4 的倍数、k 需为 128 的倍数",
                case.op, case.m, case.k
            );
            continue;
        }
        // 张量核口径：`gemm_imma` 每 block 固定 BM=64 行 ⇒ m 需为 64 的倍数；
        // 设备不支持 coopmat i8 时整族跳过（CUDA 后端走它自己的同名内核，不经这个 trait）。
        if case.imma {
            if case.m % 64 != 0 {
                eprintln!("跳过 {} m={}：gemm_imma 需 m 为 64 的倍数", case.op, case.m);
                continue;
            }
            if !b.supports_int8_imma() {
                eprintln!("跳过 {}：本后端 supports_int8_imma() = false", case.op);
                continue;
            }
        }
        for &batch in &batches {
            let (ms, jitter, fp) =
                match run_case(b.as_mut(), case, batch, cap_rows, iters, warmup, repeats) {
                    Ok(v) => v,
                    Err(e) => {
                        eprintln!("{} batch={} 失败: {e:#}", case.op, batch);
                        continue;
                    }
                };
            let (wbytes, is_f16) = if case.op == "f16" || case.pf16 {
                (case.wbytes_f16(), true)
            } else {
                (case.wbytes_int8(), false)
            };
            let gbs = wbytes / (ms / 1000.0) / 1e9;
            println!(
                "{:<11} {:>7} {:>6} {:>6} {:>11.4} {:>12.2} {:>10.1} {:>9.1} {:>7.1}% {:>12.4}{}",
                case.op,
                case.m,
                case.k,
                batch,
                ms,
                wbytes / 1e6,
                gbs,
                case.tops(batch, ms),
                jitter * 100.0,
                fp,
                if is_f16 { "  (fp16)" } else { "" }
            );
        }
    }
    println!();
    println!("注：int8 内核按 BGRP 分组，batch > BGRP 时权重实际被读 ceil(batch/BGRP) 遍；");
    println!("    上表「有效 GB/s」按单次读折算，故真实带宽 = 该值 × ceil(batch/BGRP)。");
    println!("    TOPS = 2·m·k·batch/ms，含 split-K 归约与量化那一趟（端到端口径）。");
    println!("    抖动超过 ~10% 时该行不可用于 A/B 判断（可用 KB_REPEATS 加大重复轮数）。");
    println!("    sum(y) = 计时后单独一次调用的输出前 4096 个元素之和；**为 0 ⇒ 内核没跑**，");
    println!("      该行的 ms/call 与 TOPS 全部作废（这一列是 2026-10-11 加的，起因是");
    println!("      GEMM_TILE=32/32/32 报出 234 TOPS = fp16 峰值的 4.3 倍这种假胜利）。");
    println!("    pf_* 族的 batch 是 **token 行数**（prefill 的 B·T_pad）、m 是输出特征数；");
    println!("      它走稠密 fp16 `gemm`（两后端都实现 ⇒ 可跨后端对拍），「有效 GB/s」只是");
    println!("      权重单次读的下界口径，判断请看 TOPS 列（fp16 张量峰值 ≈ 54 TOPS）。");
}

/// 建张量 → 预热 → `repeats` 轮（每轮 `iters` 次调用 + 一次同步）→
/// 返回 `(最小 ms/call, 抖动, 结果指纹)`，抖动 = (最大-最小)/最小。取最小以剔除外部占卡干扰。
fn run_case(
    b: &mut dyn ComputeBackend,
    case: &Case,
    batch: usize,
    cap_rows: usize,
    iters: usize,
    warmup: usize,
    repeats: usize,
) -> Result<(f64, f64, f64), Box<dyn std::error::Error>> {
    let (m, k) = (case.m, case.k);
    let mut rng = Rng(0x5A5A_1234_ABCD_0001);

    // 权重：int8（idx 打包 4×uint8 + sz 打包 scale/zero 两个 f16）或 fp16。
    let (a8, w16) = if case.op == "f16" || case.pf16 {
        let t = b.create_tensor(m * k, TensorDtype::F16)?;
        b.upload(t, &vec![0.01f32; m * k])?;
        (None, Some(t))
    } else {
        let idx: Vec<u32> = (0..m * (k / 4)).map(|_| 0x7f7f_7f7fu32).collect();
        let sz: Vec<u32> = vec![0x3800_3c00u32; m * (k / 128)]; // 低 16=scale(1.0)，高 16=zero(0.5)
        let h = Int8Handle {
            idx: b.create_tensor(m * (k / 4), TensorDtype::U32)?,
            sz: b.create_tensor(m * (k / 128), TensorDtype::U32)?,
            m,
            k,
        };
        b.upload_u32(h.idx, &idx)?;
        b.upload_u32(h.sz, &sz)?;
        (Some(h), None)
    };

    // 激活 [batch, k] f32 / 门控 [batch, k] f16 / 输出 [batch, m] f32。
    // ★ pf16 三族**不建 f32 的 x**：它的「batch」是 token 行，rows=65536 时 x 是 671 MB 的
    //   上传（在计时之外，但会把一次基准的准备拖到分钟级），而 `gemm` 的 A 本来就要求 fp16。
    let (x, g, a16) = if case.pf16 {
        let a = b.create_tensor(batch * k, TensorDtype::F16)?;
        b.upload(a, &vec![0.01f32; batch * k])?;
        (None, None, Some(a))
    } else {
        let x = b.create_tensor(batch * k, TensorDtype::F32)?;
        let xs: Vec<f32> = (0..batch * k)
            .map(|_| 0.01 * (1.0 + rng.next_f32().abs()))
            .collect();
        b.upload(x, &xs)?;
        let g = b.create_tensor(batch * k, TensorDtype::F16)?;
        b.upload(g, &vec![0.5f32; batch * k])?;
        (Some(x), Some(g), None)
    };
    let y = b.create_tensor(batch * m, TensorDtype::F32)?;
    b.upload(y, &vec![0.0f32; batch * m])?;

    // 张量核路径的三个附加张量：量化激活 xq（u32 打包 4×int8）、每组统计 xaux（vec4/组）、
    // split-K 部分和（容量 = `cap_rows·batch`，与引擎的 `imma_partial` 同口径 ⇒ 这里选出的
    // ks 与真实推理一致）。
    let (xq, xaux, partial) = if case.imma {
        let xq = b.create_tensor(batch * (k / 4), TensorDtype::U32)?;
        b.upload_u32(xq, &vec![0u32; batch * (k / 4)])?;
        let xaux = b.create_tensor(batch * (k / 128) * 4, TensorDtype::F32)?;
        b.upload(xaux, &vec![1.0f32; batch * (k / 128) * 4])?;
        let partial = b.create_tensor(cap_rows * batch, TensorDtype::F32)?;
        b.upload(partial, &vec![0.0f32; cap_rows * batch])?;
        (Some(xq), Some(xaux), Some(partial))
    } else {
        (None, None, None)
    };

    let call = |b: &mut dyn ComputeBackend| -> Result<(), Box<dyn std::error::Error>> {
        match case.op.as_str() {
            "relu2" => b.gemv_int8_relu2(a8.as_ref().unwrap(), x.unwrap(), y, m, k, batch),
            "mul_add" => {
                b.gemv_int8_mul_add(a8.as_ref().unwrap(), x.unwrap(), g.unwrap(), y, m, k, batch)
            }
            "plain" => b.gemv_int8_plain(a8.as_ref().unwrap(), x.unwrap(), y, m, k, batch),
            "add" => b.gemv_int8_add(a8.as_ref().unwrap(), x.unwrap(), y, m, k, batch),
            "f16" => b.gemv_f16(w16.unwrap(), x.unwrap(), y, m, k, batch),
            "pf_out" | "pf_key" | "pf_value" => {
                // `gemm(a, w, out, rows, n, k)`：`case.m` 在这里是**输出特征数 n**。
                b.gemm(a16.unwrap(), w16.unwrap(), y, batch, m, k)
            }
            "imma_key" | "imma_out" | "imma_value" | "imma_head" => {
                let (xq, xaux, partial) = (xq.unwrap(), xaux.unwrap(), partial.unwrap());
                // 与 `gpu_model.rs` 一致：量化与 GEMM 成对计时（门控在量化那一步做掉）。
                b.quant_x_i8(
                    x.unwrap(),
                    case.xgate.then_some(g.unwrap()),
                    xq,
                    xaux,
                    k,
                    batch,
                )?;
                b.gemm_imma(
                    a8.as_ref().unwrap(),
                    xq,
                    xaux,
                    y,
                    partial,
                    m,
                    k,
                    batch,
                    case.icode,
                )?;
                Ok(())
            }
            other => unreachable!("已校验: {other}"),
        }
    };

    // ★ 计时圈必须**两边都收口**，而两个后端的收口方式相反：
    //   - Vulkan 的 `record_kernel` 只把 dispatch 记进持久 command buffer，**不提交**
    //     （`end_batch` 才 submit + wait_idle）⇒ 漏 `end_batch` 量到的是纯 host 记录时间
    //     （`add`（27 MB 权重）报 0.0053 ms/call = 5100 GB/s，物理不可能）；
    //   - CUDA 的 `begin_batch`/`end_batch` **既不提交也不等待**（内核当场 launch 到 stream，
    //     真正的收口在 graph replay / download）⇒ 只靠 `end_batch` 量到的是 host 入队时间
    //     （`gemm` 报 1840~112000 TOPS，而 fp16 张量峰值只有 ~54）。
    //   ⇒ 统一走 `synchronize()`，它才是跨后端可比的口径。
    for _ in 0..warmup {
        b.begin_batch()?;
        call(b)?;
        b.end_batch()?;
    }
    // 预热后同步一次，避免把首次编译/首次 launch 的开销计入。
    // ★ 这里原来用 `download(y)` 兜同步：`pf_*` 的 y 在 rows=16384 时是 671 MB 的下载
    //   （纯浪费）。`synchronize` 就是为这件事存在的原语。
    b.synchronize()?;

    let mut sum = 0.0f64;
    // ★ 一次「带读回」的调用（放在计时圈**外面**，圈保持纯净）。
    //   `sum == 0` ⇒ 输出从未落地 ⇒ 下面那一行的 ms/call 是「内核根本没跑」的速度，不可用。
    //   为什么必须内建：tile=32×32×32 那一档一开始报出 234 TOPS（fp16 张量峰值 54 的 4.3 倍），
    //   而它在表里与其他档看不出任何区别——只有读回才暴露得出来。
    {
        b.begin_batch()?;
        call(b)?;
        b.end_batch()?;
        b.synchronize()?;
        let mut chunk = 1usize << 16; // 先试 65536 元素
        loop {
            match b.download_part(y, 0, chunk) {
                Ok(v) => {
                    sum = v.iter().take(4096).map(|&x| x as f64).sum();
                    break;
                }
                Err(e) => {
                    if chunk > 16 {
                        chunk /= 2; // 后端对下载长度有约束（对齐/页）⇒ 逐半退让
                    } else {
                        eprintln!("读回失败（结果不可验证）: {e}");
                        break;
                    }
                }
            }
        }
    }

    // repeats 轮，每轮 iters 次调用 + 一次同步；取最小值（最不易被外部占卡污染）。
    let mut best = f64::INFINITY;
    let mut worst = 0.0f64;
    for _ in 0..repeats {
        b.begin_batch()?;
        let t0 = Instant::now();
        for _ in 0..iters {
            call(b)?;
        }
        b.end_batch()?; // Vulkan：submit + wait_idle
        // ★ CUDA 的 end_batch **不等**（内核直接 launch 到 stream 上），少了这一行就会
        //   把 host 入队时间当 GPU 时间报出去（实测 `gemm` 报 1840~112000 TOPS，而 fp16
        //   张量峰值只有 ~54）⇒ 跨后端可比的唯一办法是两边都显式 synchronize。
        b.synchronize()?;
        let ms = t0.elapsed().as_secs_f64() * 1000.0 / iters as f64;
        best = best.min(ms);
        worst = worst.max(ms);
    }
    let jitter = if best > 0.0 {
        (worst - best) / best
    } else {
        0.0
    };

    // 结果指纹：由上面那次带读回的调用填的 `sum`（见其注释）。
    let fp = sum;

    // 释放临时张量（避免注册表膨胀）。
    let mut all: Vec<TensorId> = vec![y];
    all.extend([x, g, a16].into_iter().flatten());
    all.extend([xq, xaux, partial].into_iter().flatten());
    if let Some(h) = a8 {
        all.push(h.idx);
        all.push(h.sz);
    }
    if let Some(t) = w16 {
        all.push(t);
    }
    for t in all {
        b.free_tensor(t);
    }
    Ok((best, jitter, fp))
}
