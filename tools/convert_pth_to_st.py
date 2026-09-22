# /// script
# requires-python = ">=3.10"
# dependencies = ["numpy", "safetensors", "torch"]
# [[tool.uv.index]]
# name = "pytorch-cu124"
# url = "https://download.pytorch.org/whl/cu124"
# priority = "explicit"
# ///
"""RWKV 原生 .pth → rwkv-rsv .st (safetensors) 转换器。

BlinkDL 原生检查点（如 rwkv7-g1i-7.2b-20260805-ctx16384.pth）为 torch
序列化的 {key: tensor} 字典，键名即 rwkv-rsv 原生键（emb/blocks.N.att....
/head）。本脚本仅做「加载 → dtype 归一化为 F16 → safetensors 落盘」，
不改动任何键名与张量数值。

用法：
  uv run tools/convert_pth_to_st.py --in rwkv7-g1i-7.2b-....pth --out rwkv7-g1i-7.2b.st
"""

import argparse
import sys
import time
from pathlib import Path

import torch
from safetensors.numpy import save_file


def main() -> int:
    ap = argparse.ArgumentParser(description="RWKV .pth → .st converter")
    ap.add_argument("--in", dest="inp", required=True)
    ap.add_argument("--out", dest="out", required=True)
    args = ap.parse_args()

    t0 = time.time()
    print(f"loading {args.inp} ...", file=sys.stderr)
    state = torch.load(args.inp, map_location="cpu", weights_only=True)
    if hasattr(state, "state_dict"):
        state = state.state_dict()

    out = {}
    skipped = []
    for key, tensor in state.items():
        if not torch.is_tensor(tensor):
            skipped.append(key)
            continue
        if tensor.dim() == 0:
            # 标量参数（如个别 legacy 缩放项）展平为 [1]
            tensor = tensor.reshape(1)
        # fp16 归一化（与 rwkv-g1h-3B.st 同基准）；整型保持原样
        if tensor.is_floating_point():
            tensor = tensor.to(torch.float16)
        out[key] = tensor.contiguous().numpy()

    if skipped:
        print(f"skipped non-tensor keys: {skipped}", file=sys.stderr)

    print(f"converting {len(out)} tensors -> {args.out} ...", file=sys.stderr)
    save_file(out, args.out)
    size_gb = Path(args.out).stat().st_size / (1024**3)
    print(f"done: {args.out} ({size_gb:.2f} GB) in {time.time() - t0:.1f}s", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
