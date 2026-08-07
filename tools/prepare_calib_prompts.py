"""下载 HF 多语言指令数据集 → trie BPE tokenize → 存 .bin 供 Rust 校准采集。

数据源：CohereForAI/aya_dataset（100+ 语言，streaming 走 parquet）。
按 language_code 分层抽样：中文 30% / 英文 30% / 其他语种 40%。

输出格式（Rust 侧 main.rs 读取）：
  u32 magic = 0xC411B0C0
  u32 n_prompts
  n_prompts × { u32 len; len × u32 token }

用法：
  python tools/prepare_calib_prompts.py --out outputs/calib_prompts.bin --n 800
"""
import argparse
import random
import struct
import sys
from pathlib import Path

sys.path.insert(0, r"c:\work\niceui\dspark-rwkv-repo")
from rwkv_tokenizer import TRIE_TOKENIZER  # noqa: E402

VOCAB_TXT = r"c:\work\niceui\dspark-rwkv-repo\rwkv_vocab_v20230424.txt"
MAGIC = 0xC411B0C0

# aya_dataset 语言分层：zho=中文，eng=英文，其余=其他语种
ZHO = {"zho"}
ENG = {"eng"}


def classify(code: str) -> str:
    if code in ZHO:
        return "zh"
    if code in ENG:
        return "en"
    return "other"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=r"c:\work\niceui\rwkv-rsv\outputs\calib_prompts.bin")
    ap.add_argument("--dataset", default="CohereForAI/aya_dataset")
    ap.add_argument("--n", type=int, default=800, help="采集的 prompt 总条数")
    ap.add_argument("--max-len", type=int, default=256, help="每条 prompt 最大 token 数（截断）")
    ap.add_argument("--seed", type=int, default=514, help="合并后打乱随机种子")
    args = ap.parse_args()

    tok = TRIE_TOKENIZER(VOCAB_TXT)
    print(f"[prepare] tokenizer 加载完成（{len(tok.idx2token)} 词条）")

    from datasets import load_dataset  # noqa: PLC0415

    ds = load_dataset(args.dataset, split="train", streaming=True)
    print(f"[prepare] 数据集 {args.dataset}（streaming）")

    # 分层目标：中 30% / 英 30% / 其他 40%
    n_zh = round(args.n * 0.30)
    n_en = round(args.n * 0.30)
    n_other = args.n - n_zh - n_en
    targets = {"zh": n_zh, "en": n_en, "other": n_other}
    buckets: dict[str, list[list[int]]] = {"zh": [], "en": [], "other": []}
    print(f"[prepare] 分层目标：中文 {n_zh} / 英文 {n_en} / 其他 {n_other}")

    for x in ds:
        bucket = classify(str(x.get("language_code", "")))
        if len(buckets[bucket]) >= targets[bucket]:
            continue
        text = str(x.get("inputs", "")).strip()
        if not text:
            continue
        toks = tok.encode(text)
        if not toks:
            continue
        if len(toks) > args.max_len:
            toks = toks[: args.max_len]
        buckets[bucket].append(toks)
        if all(len(buckets[b]) >= targets[b] for b in buckets):
            break

    for b, target in targets.items():
        got = len(buckets[b])
        if got < target:
            print(f"[prepare] 警告：{b} 仅采到 {got}/{target}（语料不足，不足部分将略过）")

    prompts = buckets["zh"] + buckets["en"] + buckets["other"]
    random.Random(args.seed).shuffle(prompts)
    if not prompts:
        print("[prepare] 未采到任何 prompt，退出")
        return

    print(f"[prepare] 采集 {len(prompts)} 条 prompt，平均 {sum(len(p) for p in prompts)/len(prompts):.1f} token")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("wb") as f:
        f.write(struct.pack("<II", MAGIC, len(prompts)))
        for p in prompts:
            f.write(struct.pack("<I", len(p)))
            f.write(struct.pack(f"<{len(p)}I", *p))
    print(f"[prepare] 写出 {out}（{out.stat().st_size} 字节）")
    # 抽查
    sample = prompts[0]
    print(f"[prepare] 样例 decode: {tok.decode(sample)[:80]!r}")


if __name__ == "__main__":
    main()