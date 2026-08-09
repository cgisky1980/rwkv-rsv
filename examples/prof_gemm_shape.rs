//! 临时诊断：实测 cuBLAS 对 prefill 大 GEMM shape 的峰值吞吐。
//! 用后即删。对比不同 CUBLAS_GEMM 设置与 shapes。

use std::error::Error;

use rwkv_rsv::backend::{TensorDtype, create_backend, detect_backend};

fn main() -> Result<(), Box<dyn Error>> {
    let _ = log::set_logger(&STDOUT_LOGGER);
    log::set_max_level(log::LevelFilter::Info);

    let mut b = create_backend(detect_backend())?;

    // FFN value GEMM: C[512, 2560] = X[512, 10240] @ W[2560, 10240]^T
    // 即 a=[m,k], b=[n,k], c=[m,n]
    let shapes: Vec<(usize, usize, usize, &str)> = vec![
        (512, 2560, 10240, "value M512"),
        (1024, 2560, 10240, "value M1024"),
        (2048, 2560, 10240, "value M2048"),
        (4096, 2560, 10240, "value M4096"),
        (512, 10240, 2560, "key   M512"),
        (1024, 10240, 2560, "key   M1024"),
        (2048, 10240, 2560, "key   M2048"),
        (4096, 10240, 2560, "key   M4096"),
    ];
    for (m, n, k, name) in shapes {
        let a = b.create_tensor(m * k, TensorDtype::F16)?;
        let wt = b.create_tensor(n * k, TensorDtype::F16)?;
        let c = b.create_tensor(m * n, TensorDtype::F32)?;
        let x: Vec<f32> =
            (0..m * k)
                .map(|i| ((i as f32) * 0.001).sin())
                .fold(vec![], |mut v, f| {
                    v.push(f);
                    v
                });
        let w: Vec<f32> =
            (0..n * k)
                .map(|i| ((i as f32) * 0.0007).cos())
                .fold(vec![], |mut v, f| {
                    v.push(f);
                    v
                });
        b.upload(a, &x)?;
        b.upload(wt, &w)?;
        // 预热
        b.gemm(a, wt, c, m, n, k)?;
        b.end_batch()?;
        // 计时
        let t0 = std::time::Instant::now();
        let reps = 10;
        for _ in 0..reps {
            b.begin_batch()?;
            b.gemm(a, wt, c, m, n, k)?;
            b.end_batch()?;
            let _ = b.download(c)?; // 强制同步
        }
        let s = t0.elapsed().as_secs_f64() / reps as f64;
        let flop = 2.0 * m as f64 * n as f64 * k as f64;
        log::info!("{name}: {:.4}s → {:.1} GFLOPS", s, flop / s / 1e9);
    }
    Ok(())
}

struct StdoutLogger;
static STDOUT_LOGGER: StdoutLogger = StdoutLogger;

impl log::Log for StdoutLogger {
    fn enabled(&self, _metadata: &log::Metadata) -> bool {
        true
    }
    fn log(&self, record: &log::Record) {
        println!(
            "[{}][{}] {}",
            chrono::Local::now().format("%H:%M:%S"),
            record.level(),
            record.args()
        );
    }
    fn flush(&self) {}
}
