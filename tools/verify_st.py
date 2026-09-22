# /// script
# requires-python = ">=3.10"
# dependencies = ["safetensors", "numpy"]
# ///
"""Verify int8 .st structure (int8_idx 允许 2D [M,K] 或 3D [M,K/128,128] group 打包)."""

import sys

import numpy as np
from safetensors import safe_open

path = sys.argv[1]
with safe_open(path, framework="numpy") as f:
    keys = list(f.keys())
    emb = f.get_tensor("emb.weight")
    vocab, c = emb.shape
    print(f"tensors: {len(keys)}, emb {emb.shape} {emb.dtype}")
    assert emb.dtype == np.float16 and len(emb.shape) == 2

    def check_idx_sz(base: str, m: int, k: int):
        idx = f.get_tensor(f"{base}.int8_idx")
        sz = f.get_tensor(f"{base}.int8_sz")
        # 字节量校验（形状 2D/3D 皆可，M=shape[0]，总字节 = M*K / M*K/128）
        assert idx.dtype == np.uint8 and idx.shape[0] == m and idx.size == m * k, (
            f"{base}.int8_idx: shape={idx.shape} size={idx.size} expect M={m} M*K={m*k}"
        )
        assert sz.dtype == np.uint32 and sz.size == m * (k // 128), (
            f"{base}.int8_sz: shape={sz.shape} size={sz.size}"
        )

    check_idx_sz("blocks.0.att.key.weight", c, c)
    rk = f.get_tensor("blocks.0.att.r_k")
    n_head, head_size = rk.shape
    check_idx_sz("blocks.0.ffn.key.weight", f.get_tensor("blocks.0.ffn.key.weight.int8_idx").size // c, c)
    check_idx_sz("blocks.0.ffn.value.weight", c, f.get_tensor("blocks.0.ffn.value.weight.int8_idx").size // c)
    check_idx_sz("head.weight", vocab, c)
    for k in ["blocks.0.ln0.weight", "ln_out.weight"]:
        t = f.get_tensor(k)
        assert t.shape == (c,) and t.dtype == np.float16, f"{k}: {t.shape} {t.dtype}"
    assert rk.dtype == np.float16, f"r_k dtype {rk.dtype}"
    n_quant = sum(1 for k in keys if k.endswith(".int8_idx"))
    print(f"config: vocab={vocab} n_embd={c} n_head={n_head} head_size={head_size}")
    print(f"quantized matrices: {n_quant}")
    print("STRUCTURE OK")
