# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors"]
# ///
"""概念验证：Hadamard/随机正交旋转能否降低 any4 权重级量化误差。

原理（QuaRot/QuIP）：对 W[M,K] 沿收缩维 K 施加正交旋转 Q：
  W_rot = W @ Q，量化 W_rot，反解 W_hat = W_hat_rot @ Q^T。
Q 正交 ⇒ 输出不变（Wx = W_rot @ (Q^T x)），但 W_rot 去相关、离群值被抹平，
per-row k-means 能更精确。此处只测权重级误差（转回去后 vs 原 W），
不涉及推理端 x 旋转（那是 shader 的事）。

三种 Q：
  - none        基线
  - randorth    随机正交（QR of 高斯）——去相关上限
  - had30       块对角 Hadamard（块=30）+ 随机符号——可快速变换（部署用候选）
用法：uv run test/rotation_concept.py [--layers 1] [--suffix ffn.key.weight|att.output.weight]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from safetensors import safe_open

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.quantize_any4 import quantize_matrix  # type: ignore


def random_orthogonal(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    A = rng.standard_normal((n, n))
    Q, _ = np.linalg.qr(A)
    return Q.astype(np.float32)


def block_hadamard_perm(n: int, block: int, seed: int) -> np.ndarray:
    """块对角 Hadamard（每块 K=block 的 Walsh-Hadamard）+ 整体随机置换 + 随机符号。
    返回正交矩阵 [n,n]；可 O(n log n) 快速应用（部署端 shader 用）。
    Q = P @ (I⊗H) @ S，H 为 block 阶 Walsh-Hadamard（含 1/sqrt(block) 归一化）。"""
    assert block & (block - 1) == 0, "block 需为 2 的幂"
    assert n % block == 0, "block 须整除 n"
    rng = np.random.default_rng(seed)
    # Walsh-Hadamard[block,block]（Sylvester 倍增）
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < block:
        H = np.block([[H, H], [H, -H]])
    H = (H * np.sqrt(1.0 / block)).astype(np.float32)
    nb = n // block
    M = np.kron(np.eye(nb, dtype=np.float32), H)  # [n,n] 正交
    perm = rng.permutation(n)
    sign = rng.choice([-1.0, 1.0], size=n).astype(np.float32)
    P = np.eye(n, dtype=np.float32)[perm]  # 行置换（正交）
    S = np.diag(sign)  # 随符号（正交）
    return (S @ M @ P).astype(np.float32)  # Q = S M P ⇒ Q^T = P^T M S，x@Q^T = fwht(x[perm])·sign


def fwht_block(x: np.ndarray, block: int) -> np.ndarray:
    """对 x[K] 做块对角 FWHT（每 block 独立，含 1/sqrt(block)）。即 x @ (I⊗H)。"""
    out = x.astype(np.float32).copy()
    k = out.shape[0]
    for b0 in range(0, k, block):
        h = out[b0 : b0 + block].copy()
        ln = 1
        while ln < block:
            for i in range(0, block, ln * 2):
                for j in range(ln):
                    a, c = h[i + j], h[i + j + ln]
                    h[i + j] = a + c
                    h[i + j + ln] = a - c
            ln *= 2
        out[b0 : b0 + block] = h / np.sqrt(block)
    return out


def fast_apply_qT(x: np.ndarray, perm: np.ndarray, sign: np.ndarray, block: int) -> np.ndarray:
    """x_rot = x @ Q^T，Q = S M P ⇒ x@Q^T = fwht_block(x[perm]) * sign。"""
    return fwht_block(x[perm], block) * sign


def rel_err(W: np.ndarray, W_hat: np.ndarray) -> float:
    return float(np.linalg.norm(W_hat - W) / max(np.linalg.norm(W), 1e-12))


def verify_fast_rotation(n: int, block: int, seed: int) -> None:
    """验证快路径 x@Q^T == 稠密 Q^T·x，且 W@x == W_rot@(Q^T·x)。"""
    Q = block_hadamard_perm(n, block, seed)
    # 从 Q=P M S 反解 perm/sign（Q 的分解：P M S，故 Q^T=S M P^T，x@Q^T=fwht(x[perm])*sign）
    rng = np.random.default_rng(seed)
    # 直接重建分解验证：Q 用已知 perm/sign/H 构造
    rng2 = np.random.default_rng(seed)
    perm = rng2.permutation(n)
    sign = rng2.choice([-1.0, 1.0], size=n).astype(np.float32)
    H = np.array([[1.0]], dtype=np.float32)
    while H.shape[0] < block:
        H = np.block([[H, H], [H, -H]])
    H = (H * np.sqrt(1.0 / block)).astype(np.float32)
    M = np.kron(np.eye(n // block, dtype=np.float32), H)
    P = np.eye(n, dtype=np.float32)[perm]
    S = np.diag(sign)
    Q2 = S @ M @ P  # 与 block_hadamard_perm 一致（S M P）
    assert np.allclose(Q, Q2, atol=1e-5), "Q 分解不一致"
    x = rng.standard_normal(n).astype(np.float32)
    dense = x @ Q2.T  # x@Q^T
    fast = fast_apply_qT(x, perm, sign, block)
    assert np.allclose(dense, fast, atol=1e-4), f"fast/dense max diff {np.abs(dense-fast).max()}"
    # 正交性 + W@x == W_rot@(Q^T x)
    W = rng.standard_normal((n, n)).astype(np.float32)
    W_rot = W @ Q2
    lhs = W @ x
    rhs = W_rot @ fast
    assert np.allclose(lhs, rhs, atol=1e-3), f"W@x vs W_rot@QT x max diff {np.abs(lhs-rhs).max()}"
    print(f"verify(n={n},block={block}) PASS: fast==dense(≤1e-4), ortho+W@x==W_rot@QTx(≤1e-3)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true", help="只跑快速变换数值验证")
    ap.add_argument("--in", dest="inp", default=r"c:\work\niceui\rwkv-g1h-3B.st")
    ap.add_argument("--suffix", default="att.output.weight")
    args = ap.parse_args()
    if args.verify:
        verify_fast_rotation(2560, 256, 0)
        verify_fast_rotation(10240, 256, 1)
        return 0

    key = f"blocks.0.{args.suffix}"
    with safe_open(args.inp, framework="numpy") as f:
        W = f.get_tensor(key).astype(np.float32)
    M, K = W.shape
    print(f"{key}  {M}x{K}")

    # 基线
    _, s0 = quantize_matrix(W, group=128, iters=50)
    print(f"基线          rel={s0['rel']:.4%}  cos={s0['cos']:.6f}  max_abs={s0['max_abs']:.4f}")

    for name, build in [
        ("randorth", lambda n: random_orthogonal(n, 0)),
        ("had256", lambda n: block_hadamard_perm(n, 256, 0)),
    ]:
        Q = build(K).astype(np.float32)
        W_rot = (W @ Q).astype(np.float32)
        # Q 正交 ⇒ ||W_hat-W|| == ||W_hat_rot-W_rot||，故 quantize_matrix(W_rot) 的 rel 即部署误差
        _, s = quantize_matrix(W_rot, group=128, iters=50)
        print(
            f"{name:9s} rotate rel={s['rel']:.4%}  cos={s['cos']:.6f}  max_abs={s['max_abs']:.4f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())