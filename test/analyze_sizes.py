"""临时分析脚本：统计 any4 量化模型文件的张量体积构成。
用法: uv run python test/analyze_sizes.py <model.st>
"""
import sys
from collections import defaultdict
from safetensors import safe_open

path = sys.argv[1]
cat = defaultdict(lambda: [0, 0])  # name -> [count, bytes]
total = 0
with safe_open(path, framework="numpy") as f:
    for k in f.keys():
        t = f.get_slice(k)
        shape = t.get_shape()
        dt = t.get_dtype()
        item = {"F16": 2, "F32": 4, "U8": 1, "U32": 4, "U16": 2, "I8": 1, "I32": 4}[dt]
        n = 1
        for s in shape:
            n *= s
        b = n * item
        total += b
        # 分类
        if k.endswith((".any4_idx", ".any4_lut", ".any4_sz")):
            type_key = "any4_" + k.rsplit(".", 1)[1]
        elif ".att." in k and any(x in k for x in ["w1", "w2", "a1", "a2", "v1", "v2", "g1", "g2"]):
            type_key = "lowrank"
        elif k.startswith("blocks.") and ("weight" in k or "bias" in k):
            type_key = "layer_param"
        elif k.startswith("emb."):
            type_key = "embedding"
        elif k.startswith("head."):
            type_key = "head"
        elif k.startswith("ln_out") or k.startswith("blocks.0.ln0"):
            type_key = "ln/emb"
        else:
            type_key = "other"
        cat[type_key][0] += 1
        cat[type_key][1] += b

print(f"{'分类':<14} {'数量':>5} {'字节':>12} {'MB':>9} {'占比':>6}")
for name, (cnt, b) in sorted(cat.items(), key=lambda x: -x[1][1]):
    print(f"{name:<14} {cnt:>5} {b:>12} {b/1e6:>9.1f} {b/total*100:>5.1f}%")
print(f"{'总计':<14} {'':>5} {total:>12} {total/1e6:>9.1f}")