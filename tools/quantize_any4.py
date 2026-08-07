# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors", "torch"]
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# priority = "explicit"
# ///
"""any4 离线量化器：RWKV-7 .st (fp16) → .any4.st

格式（对齐 any4 论文 arXiv:2507.04610，group_size=128）：
  对大权值矩阵 W[M, K]（行主序，K 为收缩维）：
    {name}.any4_idx  U8  [M, K/2]   每字节 2 个 4-bit 索引（低 nibble=偶数 k，高 nibble=奇数 k）
    {name}.any4_lut  F16 [M, 16]    每行 16 项学习码本（per-row k-means 质心，定义在组归一化域）
    {name}.any4_sz   U32 [M, K/128] 每元素 = (scale:fp16 低16位 | zero:fp16 高16位)
  反量化：w[m,k] = scale[m,k/128] * lut[m, idx] + zero[m,k/128]

量化对象白名单（每层 6 大矩阵）：
  blocks.{i}.att.{receptance,key,value,output}.weight  [C,C]
  blocks.{i}.ffn.{key,value}.weight                    [4C,C] / [C,4C]
不量化：head/emb/ln/lerp/低秩小矩阵（保持原样拷贝）。

用法：
  uv run tools/quantize_any4.py [--in x.st] [--out y.st] [--layers N] [--iters 50] [--group 128]
"""

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file

# 量化白名单后缀（blocks.{i}. 之后部分）
QUANT_SUFFIXES = frozenset(
    [
        "att.receptance.weight",
        "att.key.weight",
        "att.value.weight",
        "att.output.weight",
        "ffn.key.weight",
        "ffn.value.weight",
    ]
)

K_CLUSTERS = 16  # 4-bit → 16 项 LUT


def quant_target_layer(key: str) -> int | None:
    """若 key 是量化目标，返回层号；否则 None。"""
    parts = key.split(".")
    # blocks.{i}.att.receptance.weight → ["blocks", "0", "att", ...]
    if len(parts) < 3 or parts[0] != "blocks":
        return None
    suffix = ".".join(parts[2:])
    if suffix not in QUANT_SUFFIXES:
        return None
    return int(parts[1])


def kmeans16_rows_torch(
    X: np.ndarray,
    iters: int,
    tol: float = 1e-4,
    chunk: int = 2048,
    sample_weight: np.ndarray | None = None,
    device: str = "cuda",
):
    """per-row Lloyd k-means 的 PyTorch GPU 后端（语义与 kmeans16_rows 完全一致）。

    逐行 1D 16 簇：assign 用「排序质心→相邻中点边界→右移计数」；update 用 index_add_
    替代 CPU 的 bincount（GPU 无带权重 bincount）。空簇重初始化 + 质心保序 + drift 早停
    与 CPU 版逐条对齐，保证两种后端产出可复现、可互核。
    显存优化：assign 用 torch.searchsorted（边界二分）替代 `X[:,None]>bounds` 的
    [mc,K,15] 广播布尔张量（15× 放大），并把整块 X 按 chunk 分批载入，峰值≈输入本身。
    返回 (C [M,16] float32, idx [M,K] uint8, 平均迭代数)。
    """
    import torch

    torch.backends.cuda.matmul.allow_tf32 = False
    M, K = X.shape
    C = np.empty((M, K_CLUSTERS), dtype=np.float32)
    idx = np.empty((M, K), dtype=np.uint8)
    qs = torch.linspace(
        0.5 / K_CLUSTERS, 1.0 - 0.5 / K_CLUSTERS, K_CLUSTERS, device=device
    ).to(torch.float32)
    sw_t = (
        torch.from_numpy(np.ascontiguousarray(sample_weight, dtype=np.float32)).to(device)
        if sample_weight is not None
        else None
    )
    iter_sum = 0
    nchunks = 0
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        # 分批载入，避免整块 [M,K] 常驻显存（ffn 大矩阵可省数百 MB）
        Xc = torch.from_numpy(
            np.ascontiguousarray(X[s:e], dtype=np.float32)
        ).to(device)  # [mc, K]
        mc = e - s
        nchunks += 1
        offsets = (torch.arange(mc, device=device) * K_CLUSTERS).to(torch.int64)  # [mc]
        # 分位数初始化（确定性，与 CPU 版一致）
        Cc = torch.quantile(Xc, qs, dim=1).t().contiguous()  # [mc, 16]
        Wflat = sw_t.repeat(mc) if sw_t is not None else None
        for it in range(iters):
            Cs, _ = torch.sort(Cc, dim=1)  # [mc,16] 有序质心
            bounds = (Cs[:, :-1] + Cs[:, 1:]) * 0.5  # [mc,15]
            # 1D assign：x 落在哪个区间 = 边界的插入位置（searchsorted 返回 ≤ 边界计数，
            # 用 right=True 等价于 `(x>bounds).sum()`，但内存 O(mc*K) 而非 O(mc*K*15)）
            Ic = torch.searchsorted(bounds.contiguous(), Xc, right=True).to(torch.uint8)
            flat = (Ic.to(torch.int64) + offsets[:, None]).ravel()  # [mc*K]
            xr = Xc.ravel()
            # update：加权时 counts 也用 Wflat 作权重（对齐 CPU 的 bincount(weights=Wflat)，
            # 质心 = Σwx/Σw；未加权时退化为计数）。float32 以容纳浮点权重。
            if Wflat is not None:
                wflat = Wflat
            else:
                wflat = torch.ones(flat.numel(), dtype=torch.float32, device=device)
            counts = torch.zeros(mc * K_CLUSTERS, dtype=torch.float32, device=device)
            counts.index_add_(0, flat, wflat)
            sums = torch.zeros(mc * K_CLUSTERS, dtype=torch.float32, device=device)
            sums.index_add_(0, flat, xr * wflat)
            sums = sums.view(mc, K_CLUSTERS)
            counts = counts.view(mc, K_CLUSTERS)
            C_new = sums / counts.clamp_min(1e-12)
            empty = counts == 0
            if empty.any():
                recon = Cs.gather(1, Ic.long())  # [mc,K] 重建
                worst = (Xc - recon).abs().argmax(dim=1)
                em, ej = empty.nonzero(as_tuple=True)
                C_new[em, ej] = Xc[em, worst[em]]
            C_new, _ = torch.sort(C_new, dim=1)
            drift = (C_new - Cs).abs().max().item()
            Cc = C_new
            if drift < tol:
                break
        iter_sum += it + 1
        bounds = (Cc[:, :-1] + Cc[:, 1:]) * 0.5
        Ic = torch.searchsorted(bounds.contiguous(), Xc, right=True).to(torch.uint8)
        C[s:e] = Cc.cpu().numpy()
        idx[s:e] = Ic.cpu().numpy()
    return C, idx, iter_sum / max(1, nchunks)


def kmeans16_rows(
    X: np.ndarray,
    iters: int,
    tol: float = 1e-4,
    chunk: int = 512,
    sample_weight: np.ndarray | None = None,
    device: str = "cpu",
):
    """per-row Lloyd k-means（1D，k=16），可选校准加权。

    X: [M, K] float32（组归一化域，∈[0,1] 附近）
    sample_weight: None 或 [K] float32（每输入通道重要性，跨行共享）。
      实现与 any4 官方一致：**assign 不变**（最近质心），仅 **update 改为加权平均**，
      让重要通道的质心更贴近其值，压缩重要通道量化误差（牺牲次要通道）。
    sample_weight 为 None 时退化为纯权重 k-means（当前默认行为）。
    返回 (C [M,16] float32 质心, idx [M,K] uint8 簇号, 平均迭代数)
    1D k-means 的 assign 用排序+边界比较，比 16 路 |x-c| 广播快。
    update 用行偏移 bincount，比 one-hot einsum 快一个数量级。
    device != "cpu" 时透明切换到 kmeans16_rows_torch（GPU 加速）。
    """
    if device != "cpu":
        try:
            return kmeans16_rows_torch(
                X, iters, tol, chunk=2048, sample_weight=sample_weight, device=device
            )
        except Exception as e:  # GPU 失败（无 torch/无 CUDA）回退 CPU
            print(f"[any4] 警告：GPU k-means 失败 ({e})，回退 CPU")
            pass
    M, K = X.shape
    C = np.empty((M, K_CLUSTERS), dtype=np.float32)
    idx = np.empty((M, K), dtype=np.uint8)
    qs = (np.arange(K_CLUSTERS) + 0.5) / K_CLUSTERS  # 16 等分位点初始化
    iter_sum = 0
    # 校准权重平铺为 [mc, K] 的连续数组（bincount 需按元素对齐）
    tile_w = (
        np.broadcast_to(sample_weight, (chunk, K)).ravel()
        if sample_weight is not None
        else None
    )
    for s in range(0, M, chunk):
        e = min(s + chunk, M)
        Xc = np.ascontiguousarray(X[s:e])
        mc = e - s
        rows = np.arange(mc)
        offsets = (rows * K_CLUSTERS)[:, None]  # [mc, 1] bincount 行偏移
        # 分位数初始化（确定性，免 k-means++ 串行采样）
        Cc = np.quantile(Xc, qs, axis=1).astype(np.float32).T  # [mc, 16]
        Wflat = tile_w[: mc * K] if tile_w is not None else None
        for it in range(iters):
            # assign：质心排序 → 相邻中点边界 → x 落在第几个区间
            order = Cc.argsort(axis=1)
            Cs = np.take_along_axis(Cc, order, axis=1)  # [mc, 16] 有序
            bounds = (Cs[:, :-1] + Cs[:, 1:]) * 0.5  # [mc, 15]
            Ic = (Xc[:, :, None] > bounds[:, None, :]).sum(axis=2).astype(np.uint8)
            # update：行偏移 bincount 求（加权）簇内和/权重和（远快于 one-hot einsum）
            flat = (Ic + offsets).ravel()
            if Wflat is not None:
                sums = np.bincount(
                    flat, weights=(Xc.ravel() * Wflat), minlength=mc * K_CLUSTERS
                )
                counts = np.bincount(flat, weights=Wflat, minlength=mc * K_CLUSTERS)
            else:
                sums = np.bincount(flat, weights=Xc.ravel(), minlength=mc * K_CLUSTERS)
                counts = np.bincount(flat, minlength=mc * K_CLUSTERS)
            sums = sums.reshape(mc, K_CLUSTERS)
            counts = counts.reshape(mc, K_CLUSTERS)
            C_new = sums / np.maximum(counts, 1e-12)
            # 空簇 → 重初始化为该行当前重建误差最大的点
            empty = counts == 0
            if empty.any():
                recon = Cs[rows[:, None], Ic]
                err = np.abs(Xc - recon)
                worst = err.argmax(axis=1)
                em, ej = np.nonzero(empty)
                C_new[em, ej] = Xc[em, worst[em]]
            # 质心保持排序（1D k-means 单调性：有序质心更新后仍近似有序，重排保险）
            C_new.sort(axis=1)
            drift = np.abs(C_new - Cs).max()
            Cc = C_new
            if drift < tol:
                break
        iter_sum += it + 1
        # 最终 assign（Cc 已有序）
        bounds = (Cc[:, :-1] + Cc[:, 1:]) * 0.5
        Ic = (Xc[:, :, None] > bounds[:, None, :]).sum(axis=2).astype(np.uint8)
        C[s:e] = Cc
        idx[s:e] = Ic
    return C, idx, iter_sum / max(1, (M + chunk - 1) // chunk)


def quantize_int8_matrix(W32: np.ndarray, group: int):
    """W32: [M, K] float32 → int8 非对称 per-group 三量化（256 级均匀，无 LUT、无 k-means）。

    格式（与 any4 的 sz 结构对齐，仅 idx 从 [M,K/2] nibble 变为 [M,K] 逐字节）：
      {name}.int8_idx  U8   [M, K]    每权重 1 字节（0..255）
      {name}.int8_sz   U32  [M, K/128] 每元素 = (scale: fp16 低16位 | zero: fp16 高16位)
    反量化：w[m,k] = scale[m,k/128] * idx[m,k] + zero[m,k/128]

    精度：256 级均匀量化，rel ≈ 0.35%（高斯域），远优于 any4 的 ~9%，接近 fp16。
    """
    M, K = W32.shape
    assert K % group == 0, f"K={K} 须被 group={group} 整除"
    KG = K // group
    Wr = W32.reshape(M, KG, group)
    wmin = Wr.min(axis=2)
    wmax = Wr.max(axis=2)
    # 反量化 w = scale*q + zero（q∈[0,255]），故 scale 存 step = (wmax-wmin)/255
    span = wmax - wmin
    zero_scale = span == 0
    scale = np.where(zero_scale, 1.0, span) / 255.0
    scale_safe = np.where(zero_scale, 1.0, scale)
    zero = wmin.copy()
    # 256 级：q = clip(round((w-zero)/scale), 0, 255)
    q = np.clip(np.round((Wr - zero[..., None]) / scale_safe[..., None]), 0, 255).astype(
        np.uint8
    )
    scale[zero_scale] = 0.0

    # 实际存储精度（fp16 化 scale/zero）后的重建，用于报告真实部署误差
    scale_h = scale.astype(np.float16).astype(np.float32)  # [M, KG]
    zero_h = zero.astype(np.float16).astype(np.float32)
    W_hat = (q.astype(np.float32) * scale_h[..., None] + zero_h[..., None]).reshape(M, K)

    diff = W_hat - W32
    mse = float((diff**2).mean())
    max_abs = float(np.abs(diff).max())
    norm_w = float(np.linalg.norm(W32))
    rel = float(np.linalg.norm(diff) / max(norm_w, 1e-12))
    cos = float(
        np.dot(W32.ravel(), W_hat.ravel())
        / max(norm_w * float(np.linalg.norm(W_hat)), 1e-12)
    )

    s16 = scale.astype(np.float16).view(np.uint16).astype(np.uint32)
    z16 = zero.astype(np.float16).view(np.uint16).astype(np.uint32)
    sz = ((z16 << 16) | s16).astype(np.uint32)  # [M, KG]

    tensors = {"idx": q, "sz": sz}
    stats = {
        "mse": mse,
        "max_abs": max_abs,
        "rel": rel,
        "rel_int4": float("nan"),
        "cos": cos,
        "iters": 0.0,
        "out_rel_before": float("nan"),
        "out_rel_after": float("nan"),
        "shape": (M, K),
    }
    return tensors, stats


def quantize_matrix(
    W32: np.ndarray,
    group: int,
    iters: int,
    sample_weight=None,
    bias_pow: float = 1.0,
    keep_outliers: bool = False,
    X: np.ndarray | None = None,
    nnq_iters: int = 300,
    device: str = "cpu",
):
    """W32: [M, K] float32 → any4 三张量 + 权重级误差统计（按实际存储的 fp16 精度评估）。

    sample_weight: None 或 [K] float32——每输入通道重要性（已含 scale_sample_weight 缩放，
      见 main 里 build_calib_weight）。传给 k-means 做加权质心更新。
    bias_pow: >=1.0，有符号幂失真（官方 any4 同名参数）。把归一化域去中心后做
      sign(x)|x|^pow，k-means 偏重重尾极值，之后反变换回原域。==1.0 时无效果。
    keep_outliers: 布尔，官方 any4 同名参数。把每行 LUT 的最大/最小项替换为该行
      实际最大/最小权重，保证极值精确重建（4-bit 误差主因正是离群值）。
    """
    M, K = W32.shape
    assert K % group == 0 and K % 2 == 0, f"K={K} 须被 group={group} 与 2 整除"
    KG = K // group
    Wr = W32.reshape(M, KG, group)
    wmin = Wr.min(axis=2)
    wmax = Wr.max(axis=2)
    scale = wmax - wmin
    # 常数组（scale==0）：scale 存 0，反量化 lut*0+zero=zero 精确重建；归一化时用 1 防除零
    zero_scale = scale == 0
    scale_safe = np.where(zero_scale, 1.0, scale)
    zero = wmin.copy()
    Ws = ((Wr - zero[..., None]) / scale_safe[..., None]).reshape(M, K)
    Ws = np.ascontiguousarray(Ws, dtype=np.float32)
    scale[zero_scale] = 0.0

    # bias_pow：去中心 + 有符号幂（单调，放大极值间距），k-means 后反变换回原域
    if bias_pow != 1.0:
        xc = Ws - 0.5
        Ws_k = (np.abs(xc) ** bias_pow) * np.sign(xc)
    else:
        Ws_k = Ws
    C, idx, avg_iters = kmeans16_rows(
        Ws_k, iters=iters, sample_weight=sample_weight, device=device
    )
    if bias_pow != 1.0:
        C = (np.abs(C) ** (1.0 / bias_pow)) * np.sign(C) + 0.5
    if keep_outliers:
        # 每行 LUT 极值项用该行实际极值权重替换（C 已有序：argmax→末项，argmin→首项）
        C[np.arange(M), C.argmax(axis=1)] = Ws.max(axis=1)
        C[np.arange(M), C.argmin(axis=1)] = Ws.min(axis=1)
        C = np.ascontiguousarray(C, dtype=np.float32)

    # nnq：固定索引，用真实激活在输出域 Adam 微调 LUT（X 为 [N,K] 校准激活）
    out_rel_before = out_rel_after = float("nan")
    if X is not None:
        C, out_rel_before, out_rel_after = nnq_output_lut(
            X, W32, idx, scale, zero, C, group, iters=nnq_iters, device=device
        )

    # 实际存储精度（fp16 化）后的重建，用于报告真实部署误差
    scale_h = scale.astype(np.float16).astype(np.float32)  # [M, KG]
    zero_h = zero.astype(np.float16).astype(np.float32)
    C_h = C.astype(np.float16).astype(np.float32)  # [M, 16]
    recon_n = C_h[np.arange(M)[:, None], idx].reshape(M, KG, group)
    W_hat = (recon_n * scale_h[..., None] + zero_h[..., None]).reshape(M, K)

    diff = W_hat - W32
    mse = float((diff**2).mean())
    max_abs = float(np.abs(diff).max())
    norm_w = float(np.linalg.norm(W32))
    rel = float(np.linalg.norm(diff) / max(norm_w, 1e-12))
    cos = float(
        np.dot(W32.ravel(), W_hat.ravel())
        / max(norm_w * float(np.linalg.norm(W_hat)), 1e-12)
    )

    # int4 g128 基线（同组 min-max 均匀 16 级）：any4 须明显优于它才有意义
    # （理论：高斯权重 int4 rel≈11%，any4 rel≈9.5%，k-means 尺度等变故组缩放不改变 k-means 误差）
    scale4 = scale_safe / (K_CLUSTERS - 1)
    q4 = np.clip(np.round((Wr - zero[..., None]) / scale4[..., None]), 0, K_CLUSTERS - 1)
    W4 = (q4 * scale4[..., None] + zero[..., None]).reshape(M, K)
    rel4 = float(np.linalg.norm(W4 - W32) / max(norm_w, 1e-12))
    del W4, q4

    # 打包：nibble（低=偶数 k，高=奇数 k）
    packed = (idx[:, 0::2] | (idx[:, 1::2] << 4)).astype(np.uint8)  # [M, K/2]
    lut16 = C.astype(np.float16)  # [M, 16]
    s16 = scale.astype(np.float16).view(np.uint16).astype(np.uint32)
    z16 = zero.astype(np.float16).view(np.uint16).astype(np.uint32)
    sz = ((z16 << 16) | s16).astype(np.uint32)  # [M, KG]

    tensors = {"idx": packed, "lut": lut16, "sz": sz}
    stats = {
        "mse": mse,
        "max_abs": max_abs,
        "rel": rel,
        "rel_int4": rel4,
        "cos": cos,
        "iters": avg_iters,
        "out_rel_before": out_rel_before,
        "out_rel_after": out_rel_after,
        "shape": (M, K),
    }
    return tensors, stats


def nnq_output_lut(
    X: np.ndarray,
    W32: np.ndarray,
    idx: np.ndarray,
    scale: np.ndarray,
    zero: np.ndarray,
    lut0: np.ndarray,
    group: int,
    iters: int = 300,
    lr: float = 0.05,
    weight_reg: float = 0.0,
    device: str = "cpu",
):
    """nnq 输出域 LUT 优化：固定 k-means 索引，最小化输出域 MSE（逐行最小二乘闭式解）。

    对每行 m：Y_hat[:,m] = b[m] + A[m] @ lut[m,:]，其中
      A[m][n,j] = sum_{k: idx[m,k]==j} scale[m,k/G]*X[n,k]  （[N,16]）
      b[m][n]   = sum_k X[n,k]*zero[m,k/G]                  （[N]）
    最优 lut[m,:] = pinv(A[m]) (Y[:,m]-b[m])，即 ||A@lut+b-Y||² 的精确极小点，
    与「Adam 收敛点」数学等价，但 O(N·16³) 闭式求解快 ~100×，全模型可行。

    weight_reg>0 时解带 Tikhonov 脊的正规方程（抑制病态/空簇），默认 0（实证最优）。
    device="cuda" 时用 torch GPU 加速（A_full 三维 einsum 大，GPU 显著快于 CPU numpy）。
    返回 (lut_opt, out_rel_before, out_rel_after)。"""
    N, K = X.shape
    M = W32.shape[0]
    # target 输出 Y = X @ W^T [N, M]
    Y = X @ W32.T
    scale_exp = np.repeat(scale, group, axis=1)  # [M, K]
    zero_exp = np.repeat(zero, group, axis=1)  # [M, K]
    row_idx = np.arange(M)[:, None]
    Wh0 = scale_exp * lut0[row_idx, idx] + zero_exp
    yn = float(np.linalg.norm(Y))
    before = float(np.linalg.norm(X @ Wh0.T - Y) / max(yn, 1e-12))

    if device == "cuda":
        import torch

        Xt = torch.from_numpy(X).contiguous().to("cuda")  # [N,K]
        scale_t = torch.from_numpy(scale_exp).contiguous().to("cuda")  # [M,K]
        idx_t = torch.from_numpy(idx).long().contiguous().to("cuda")  # [M,K]
        Tt = torch.from_numpy(Y - X @ zero_exp.T).contiguous().to("cuda")  # [N,M]
        # 逐块求解，避免构造 [N,M,K]/[M,K,16] 大中间张量（对 M=10240 会爆几十 GB 显存）。
        # A_full[n,m,:] = sum_k scale[m,k]*X[n,k]*[idx[m,k]==j] → 预乘 P[m,k,:]=scale[m,k]*onehot(idx[m,k])
        # 再两数组 einsum "nk,bkj->nbj"（torch 走 matmul，不展开 [N,M,K]）。
        MCH = 1024
        lut_t = torch.empty(M, K_CLUSTERS, device="cuda")  # [M,16]
        bad_t = torch.zeros(M, dtype=torch.bool, device="cuda")
        for m0 in range(0, M, MCH):
            m1 = min(m0 + MCH, M)
            idf_b = torch.nn.functional.one_hot(
                idx_t[m0:m1], num_classes=K_CLUSTERS
            ).float()  # [blk,K,16]
            P_b = scale_t[m0:m1][:, :, None] * idf_b  # [blk,K,16]
            A_b = torch.einsum("nk,bkj->nbj", Xt, P_b)  # [N,blk,16]
            T_b = Tt[:, m0:m1]  # [N,blk]
            AtA_b = torch.einsum("nbj,nbk->bjk", A_b, A_b)  # [blk,16,16]
            AtT_b = torch.einsum("nbj,nb->bj", A_b, T_b)  # [blk,16]
            if weight_reg > 0:
                AtA_b = AtA_b + weight_reg * torch.eye(K_CLUSTERS, device="cuda")
            lut_b = torch.linalg.pinv(AtA_b) @ AtT_b.unsqueeze(-1)  # [blk,16,1]
            lut_b = lut_b.squeeze(-1)  # [blk,16]
            lut_t[m0:m1] = lut_b
            bad_t[m0:m1] = (~torch.isfinite(lut_b)).any(dim=1)
        lut_np = lut_t.cpu().numpy()
        lut = np.ascontiguousarray(lut_np, dtype=np.float32)
        bad_np = bad_t.cpu().numpy()
        if bad_np.any():
            lut[bad_np] = lut0[bad_np]
    else:
        # A_full[m,n,j] = sum_{k: idx[m,k]==j} scale[m,k/G]*X[n,k]；b_full = X @ zero_exp.T
        idx_oh = np.eye(K_CLUSTERS, dtype=np.float32)[idx]  # [M, K, 16]
        A_full = np.einsum("nk,mk,mkj->nmj", X, scale_exp, idx_oh)  # [N, M, 16]
        b_full = X @ zero_exp.T  # [N, M]
        T = Y - b_full  # [N, M]
        # 正规方程：AtA[m,:,:] = A^T A [M,16,16]，AtT[m,:] = A^T T [M,16]
        AtA = np.einsum("nmj,nmk->mjk", A_full, A_full)
        AtT = np.einsum("nmj,nm->mj", A_full, T)
        if weight_reg > 0:
            AtA = AtA + weight_reg * np.eye(K_CLUSTERS, dtype=np.float32)
        # 每行 16×16 求解；用 pinv 对空簇/欠采样导致的奇异 AtA 稳健（solve 会直接抛 Singular）
        lut = np.einsum("mjk,mk->mj", np.linalg.pinv(AtA), AtT)  # [M,16]
        # 数值兜底：若某行解异常（NaN/Inf），回退该行 k-means LUT
        bad = ~np.isfinite(lut).all(axis=1)
        if bad.any():
            lut[bad] = lut0[bad]
        lut = np.ascontiguousarray(lut, dtype=np.float32)

    Wh = scale_exp * lut[row_idx, idx] + zero_exp
    after = float(np.linalg.norm(X @ Wh.T - Y) / max(yn, 1e-12))
    return lut, before, after


def build_calib_weight(calib: np.ndarray, scale: np.ndarray, group: int, K: int):
    """校准激活均值 → k-means per-column 权重（实现论文 `scale_sample_weight=True`）。

    calib: [K] float32，某层的输入通道激活均值（跨校准 token 的 mean|activation|）。
    scale: [M, KG] float32，组 scale（重构投影到输出域时被 scale 放大）。
    论文理由：反量化误差在输出域为 `scale_g * lut_err`，故重要通道权重 = calib * group_scale。
    返回 [K] float32，作为 kmeans16_rows 的 sample_weight。
    """
    Kc = calib.shape[0]
    assert Kc == K, f"calib 长度 {Kc} 须等于 K={K}"
    KG = K // group
    # 每组的 scale 平铺到该组内每个列（跨行取均值，K 为收缩维/输入维）
    gscale = scale.mean(axis=0)  # [KG]
    sw = np.repeat(gscale, group)  # [K]
    w = calib * sw
    # 归一化避免尺度影响 bincount 权重（仅相对大小有意义）
    denom = float(np.abs(w).sum())
    if denom > 0:
        w = w / denom
    return w.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="any4/int8 offline quantizer for RWKV-7 .st")
    ap.add_argument("--in", dest="inp", default=r"c:\work\niceui\rwkv-g1h-3B.st")
    ap.add_argument("--out", dest="out", default=None, help="默认 {stem}.any4.st")
    ap.add_argument(
        "--bits",
        type=int,
        default=4,
        choices=[4, 8],
        help="量化位宽：4=any4（LUT+k-means，默认）；8=int8 非对称 per-group（无 LUT，近无损）",
    )
    ap.add_argument("--layers", type=int, default=None, help="只量化前 N 层（调试用）")
    ap.add_argument(
        "--suffixes",
        default=None,
        help="只量化这些后缀（逗号分隔，如 ffn.key.weight；默认全部 6 类）",
    )
    ap.add_argument("--iters", type=int, default=50, help="k-means 迭代上限")
    ap.add_argument("--group", type=int, default=128, help="量化组大小")
    ap.add_argument(
        "--bias-pow",
        type=float,
        default=1.0,
        help="有符号幂失真（>=1.0，官方 any4 同名参数）：k-means 偏重重尾极值。==1.0 关闭",
    )
    ap.add_argument(
        "--keep-outliers",
        action="store_true",
        help="每行 LUT 极值项替换为该行实际极值权重（官方 any4 同名参数），保证离群值精确重建",
    )
    ap.add_argument(
        "--calib",
        default=None,
        help="校准激活 npz（键=完整张量名→[K] 激活均值；或 __shared__→[K] 全矩阵共用）。"
        "启用校准加权 k-means（scale_sample_weight=True）。产出见 tools/collect_calib.py",
    )
    ap.add_argument(
        "--nnq-calib",
        default=None,
        help="nnq 输出域 LUT 优化用的校准激活 safetensors（键=blocks.{li}.{name}→[N,K] 样本矩阵，"
        "产出见 main.rs CALIB_SAMPLES）。指定后对每个量化矩阵做输出域 Adam 微调 LUT",
    )
    ap.add_argument("--nnq-iters", type=int, default=300, help="nnq Adam 迭代上限")
    ap.add_argument(
        "--device",
        default="auto",
        help="k-means 计算设备：auto（有 CUDA 用 cuda 否则 cpu）/ cuda / cpu。"
        "GPU 用 PyTorch 后端显著加速 Lloyd 迭代",
    )
    ap.add_argument(
        "--report",
        default=r"c:\work\niceui\rwkv-rsv\参考\any4量化报告.md",
        help="权重级误差报告输出路径",
    )
    args = ap.parse_args()
    if args.device == "auto":
        try:
            import torch

            args.device = "cuda" if torch.cuda.is_available() else "cpu"
        except Exception:
            args.device = "cpu"
    print(f"[any4] k-means 设备：{args.device}")

    inp = Path(args.inp)
    suffix = ".int8" if args.bits == 8 else ".any4"
    out = Path(args.out) if args.out else inp.with_name(inp.stem + suffix + ".st")

    t_start = time.time()
    out_tensors: dict[str, np.ndarray] = {}
    report_rows: list[tuple[str, dict]] = []

    with safe_open(str(inp), framework="numpy") as f:
        keys = list(f.keys())
        # 分类：量化目标 vs 原样拷贝
        suffix_allow = (
            set(args.suffixes.split(",")) if args.suffixes else QUANT_SUFFIXES
        )
        targets: list[tuple[str, int]] = []
        for k in keys:
            li = quant_target_layer(k)
            if li is None or (args.layers is not None and li >= args.layers):
                continue
            if k.split(".", 2)[2] not in suffix_allow:
                continue
            targets.append((k, li))
        target_set = {k for k, _ in targets}

        # 校准激活 npz（可选）：键=完整张量名→[K]，或 __shared__→[K] 全矩阵共用
        calib_map: dict[str, np.ndarray] = {}
        if args.calib:
            calib_path = Path(args.calib)
            if not calib_path.exists():
                print(f"[any4] 警告：校准文件不存在，忽略：{calib_path}")
            else:
                with np.load(calib_path) as z:
                    calib_map = {k2: np.asarray(z[k2], dtype=np.float32) for k2 in z.files}
        # nnq 校准激活 safetensors（可选）：键=blocks.{li}.{name}→[N,K]
        nnq_map: dict[str, np.ndarray] = {}
        if args.nnq_calib:
            nnq_path = Path(args.nnq_calib)
            if not nnq_path.exists():
                print(f"[any4] 警告：nnq 校准文件不存在，忽略：{nnq_path}")
            else:
                with safe_open(str(nnq_path), framework="numpy") as nf:
                    nnq_map = {k2: np.asarray(nf.get_tensor(k2), dtype=np.float32) for k2 in nf.keys()}
        print(
            f"[any4] 输入 {inp} 共 {len(keys)} 张量，量化目标 {len(targets)} 个"
            + (f"（仅前 {args.layers} 层）" if args.layers is not None else "")
            + (f"（仅后缀 {args.suffixes}）" if args.suffixes else "")
            + (f"，校准加权 {len(calib_map)} 个" if calib_map else "")
            + (f"，nnq 输出域 {len(nnq_map)} 个" if nnq_map else "")
        )

        # 1) 非量化张量原样拷贝
        for k in keys:
            if k not in target_set:
                out_tensors[k] = f.get_tensor(k)

        # 2) 逐矩阵量化（及时释放原 fp16，控制内存峰值）
        for n, (k, li) in enumerate(sorted(targets), 1):
            t0 = time.time()
            W = f.get_tensor(k)  # [M, K] fp16（safetensors 存储 [out, in] = [M, K]）
            assert W.ndim == 2, f"{k}: 期望 2D，得 {W.shape}"
            W32 = W.astype(np.float32)
            del W
            if args.bits == 8:
                tensors, stats = quantize_int8_matrix(W32, args.group)
                del W32
                out_tensors[f"{k}.int8_idx"] = tensors["idx"]
                out_tensors[f"{k}.int8_sz"] = tensors["sz"]
                report_rows.append((k, stats))
                print(
                    f"[int8] ({n}/{len(targets)}) {k} {stats['shape']} "
                    f"cos={stats['cos']:.6f} rel={stats['rel']:.4%} "
                    f"max_abs={stats['max_abs']:.4f} ({time.time() - t0:.1f}s)",
                    flush=True,
                )
                continue
            # 校准加权：合并该矩阵的组 scale（scale_sample_weight=True）；无则纯权重 k-means
            sw = None
            cb = calib_map.get(k) or calib_map.get("__shared__")
            if cb is not None and cb.shape[0] == W32.shape[1]:
                M_, K_ = W32.shape
                Wr = W32.reshape(M_, K_ // args.group, args.group)
                gscale = Wr.max(axis=2) - Wr.min(axis=2)  # [M, KG]
                sw = build_calib_weight(cb, gscale, args.group, K_)
            tensors, stats = quantize_matrix(
                W32,
                group=args.group,
                iters=args.iters,
                sample_weight=sw,
                bias_pow=args.bias_pow,
                keep_outliers=args.keep_outliers,
                X=nnq_map.get(k),
                nnq_iters=args.nnq_iters,
                device=args.device,
            )
            del W32
            out_tensors[f"{k}.any4_idx"] = tensors["idx"]
            out_tensors[f"{k}.any4_lut"] = tensors["lut"]
            out_tensors[f"{k}.any4_sz"] = tensors["sz"]
            report_rows.append((k, stats))
            print(
                f"[any4] ({n}/{len(targets)}) {k} {stats['shape']} "
                f"cos={stats['cos']:.6f} rel={stats['rel']:.4%} "
                f"(int4: {stats['rel_int4']:.4%}) "
                f"max_abs={stats['max_abs']:.4f} iters={stats['iters']:.0f} "
                f"({time.time() - t0:.1f}s)",
                flush=True,
            )

    print(f"[any4] 写出 {out} ...")
    save_file(out_tensors, str(out))

    # 汇总报告
    avg_cos = float(np.mean([s["cos"] for _, s in report_rows])) if report_rows else 1.0
    avg_rel = float(np.mean([s["rel"] for _, s in report_rows])) if report_rows else 0.0
    avg_rel4 = float(np.mean([s["rel_int4"] for _, s in report_rows])) if report_rows else 0.0
    worst = min(report_rows, key=lambda r: r[1]["cos"]) if report_rows else None
    nnq_rows = [s for _, s in report_rows if not np.isnan(s["out_rel_before"])]
    avg_out_before = float(np.mean([s["out_rel_before"] for s in nnq_rows])) if nnq_rows else None
    avg_out_after = float(np.mean([s["out_rel_after"] for s in nnq_rows])) if nnq_rows else None

    lines = [
        f"# {'int8' if args.bits == 8 else 'any4'} 量化权重级误差报告",
        "",
        f"- 输入：`{inp}`",
        f"- 输出：`{out}`",
        f"- 位宽：{args.bits}-bit，group_size = {args.group}",
        f"- 量化矩阵数：{len(report_rows)}"
        + (f"（仅前 {args.layers} 层）" if args.layers is not None else ""),
        f"- 耗时：{time.time() - t_start:.1f}s",
        "",
        f"## 汇总",
        "",
        f"- 平均余弦相似度：**{avg_cos:.6f}**（目标 ≥ 0.995）",
        (
            f"- 平均相对误差：**{avg_rel:.4%}**（目标 ≤ 10.5%，且须优于 int4 基线 **{avg_rel4:.4%}**）"
            if args.bits == 4
            else f"- 平均相对误差：**{avg_rel:.4%}**（int8 近无损，rel 应 ≪1%）"
        ),
        (
            f"- nnq 输出域相对误差：**{avg_out_before:.5f} → {avg_out_after:.5f}**"
            + ("（逼近无损目标：≪1e-2）" if avg_out_after < 1e-2 else "")
            if avg_out_after is not None
            else "- nnq 输出域优化：关闭"
        ),
        "",
        "> 验收阈值的依据：k-means 是尺度等变的，per-row 16 簇量化的误差 ≈ 16 值 Lloyd-Max",
        "> 量化高斯分布的理论最优（rel ≈ 9.5%）；int4 g128 基线 ≈ 11%。组 scale/zero 对 any4",
        "> 不降低 k-means 误差（仅适配 tinygemm 式存储布局），故 2% 一类阈值不可达，",
        "> 正确验收方式是与 int4 基线对比 + 端到端 logits/文本验证。",
    ]
    if worst:
        lines.append(
            f"- 最差矩阵：`{worst[0]}` cos={worst[1]['cos']:.6f} rel={worst[1]['rel']:.4%}"
        )
    lines += [
        "",
        "| 矩阵 | 形状 | 余弦 | 相对误差 | int4 基线 | max_abs | 输出域误差(nnq前→后) | 迭代 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, s in report_rows:
        out_col = (
            f"{s['out_rel_before']:.5f}→{s['out_rel_after']:.5f}"
            if not np.isnan(s["out_rel_before"])
            else "—"
        )
        rel4_col = f"{s['rel_int4']:.4%}" if not np.isnan(s["rel_int4"]) else "—"
        lines.append(
            f"| `{k}` | {s['shape'][0]}×{s['shape'][1]} | {s['cos']:.6f} "
            f"| {s['rel']:.4%} | {rel4_col} | {s['max_abs']:.4f} | {out_col} | {s['iters']:.0f} |"
        )
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[any4] 报告 → {report_path}")
    print(f"[any4] 汇总: avg_cos={avg_cos:.6f} avg_rel={avg_rel:.4%} int4={avg_rel4:.4%}")
    # 权重级验收：cos ≥ 0.995，rel ≤ 10.5%，且优于 int4 基线（依据见报告注）
    if args.bits == 8:
        ok = avg_cos >= 0.999 and avg_rel <= 0.01
    else:
        ok = avg_cos >= 0.995 and avg_rel <= 0.105 and avg_rel < avg_rel4
    print(f"[any4] 权重级验收: {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
