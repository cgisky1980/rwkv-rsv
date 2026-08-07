# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors"]
# ///
"""nnq holdout 泛化验证：训练 16 token → 留出 16 token，看输出域误差是否都降。
若留出集也降，说明 nnq 不（完全）过拟合校准集；否则需更大校准或约束。用完即删。"""
import numpy as np
from safetensors import safe_open
from safetensors.numpy import save_file
import sys
sys.path.insert(0, r"c:\work\niceui\rwkv-rsv\tools")
from quantize_any4 import nnq_output_lut, kmeans16_rows

MODEL = r"c:\work\niceui\rwkv-g1h-3B.st"
CALIB = r"c:\work\niceui\rwkv-rsv\outputs\calib128.st"
GROUP = 128

# 数据集：训练前 96 token，留出后 32 token
init_tokens = np.arange(128, dtype=np.int64)
train_idx = init_tokens[:96]
val_idx = init_tokens[96:]

with safe_open(MODEL, framework="numpy") as f, safe_open(CALIB, framework="numpy") as c:
    for key in [
        "blocks.0.att.key.weight",
        "blocks.0.att.output.weight",
        "blocks.0.ffn.key.weight",
    ]:
        W = f.get_tensor(key).astype(np.float32)
        Xall = c.get_tensor(key).astype(np.float32)  # [128, K]
        M, K = W.shape
        KG = K // GROUP
        Wr = W.reshape(M, KG, GROUP)
        wmin = Wr.min(axis=2); wmax = Wr.max(axis=2)
        scale = np.where(wmax == wmin, 1.0, wmax - wmin)
        zero = wmin.copy()
        Ws = ((Wr - zero[..., None]) / scale[..., None]).reshape(M, K)
        C, idx, _ = kmeans16_rows(np.ascontiguousarray(Ws, dtype=np.float32), iters=50)
        def out_rel(X, lut):
            Y = X @ W.T
            Wh = np.repeat(scale, GROUP, axis=1) * lut[np.arange(M)[:, None], idx] + np.repeat(zero, GROUP, axis=1)
            return float(np.linalg.norm(X @ Wh.T - Y) / max(np.linalg.norm(Y), 1e-12))
        val_base = out_rel(Xall[val_idx], C)
        print(f"--- {key}  留出集基线={val_base:.5f}")
        for wreg in [0.0, 0.2, 1.0, 4.0]:
            lut, tr_b, tr_a = nnq_output_lut(Xall[train_idx], W, idx, scale, zero, C, GROUP, iters=300, weight_reg=wreg)
            val_a = out_rel(Xall[val_idx], lut)
            print(f"  wreg={wreg:5.2f}: 训练 {tr_b:.4f}→{tr_a:.4f} | 留出 {val_base:.4f}→{val_a:.4f}")