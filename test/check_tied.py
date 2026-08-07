"""临时验证：head.weight 与 emb.weight 是否共享(tied)。"""
import sys
import numpy as np
from safetensors import safe_open

path = "c:\\work\\niceui\\rwkv-g1h-3B.st"
with safe_open(path, framework="numpy") as f:
    keys = list(f.keys())
    has_head = any("head.weight" in k or k == "head.weight" for k in keys)
    has_emb = any("emb.weight" in k or k == "emb.weight" for k in keys)
    print("keys containing head:", [k for k in keys if "head" in k])
    print("keys containing emb :", [k for k in keys if "emb" in k])
    if has_head and has_emb:
        hk = [k for k in keys if k == "head.weight"]
        ek = [k for k in keys if k == "emb.weight"]
        hk = hk[0] if hk else [k for k in keys if "head.weight" in k][0]
        ek = ek[0] if ek else [k for k in keys if "emb.weight" in k][0]
        h = f.get_tensor(hk)
        e = f.get_tensor(ek)
        print(f"head: {hk} shape={h.shape} dtype={h.dtype}")
        print(f"emb : {ek} shape={e.shape} dtype={e.dtype}")
        if h.shape == e.shape:
            same = np.array_equal(h, e)
            print("== TIED (逐字节相同) ==" if same else "== NOT tied ==")
            if not same:
                print(f"max abs diff = {np.abs(h.astype(np.float32)-e.astype(np.float32)).max():.6f}")
        else:
            print("shape 不同，非 tied")