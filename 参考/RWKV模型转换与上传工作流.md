# RWKV 模型转换与上传工作流（pth → st → int8 → MS/HF）

> 2026-08-31 首次整理（g1j 2.9b → rwkv7-3B-int8.st 实战记录）。工具链已收敛到本仓库，
> 此后按本文档操作即可，无需再找 ai00_server / niceui 的散落脚本。

## 工具链（全部在 `rwkv-rsv/tools/`，uv 直跑）

| 脚本 | 用途 | 来源 |
|---|---|---|
| `convert_pth_to_st.py` | RWKV7 原生 `.pth`（bf16）→ `.st`（fp16 safetensors）。**键名即 rwkv-rsv 原生键，不转置不改名**（RWKV7 的 att.w1/w2/a1/a2/g1/g2/v1/v2 低秩矩阵按 pth 原始 `[in,out]` 方向保留，运行时按该方向计算） | 本仓库自有 |
| `quantize_any4.py` | `.st` 离线量化：`--bits 8` = int8 非对称 per-group(128)（近无损，纯 numpy）；`--bits 4` = any4（k-means LUT，可 GPU 加速） | 本仓库自有 |
| `convert_safetensors.py` | ai00_server 旧版转换器（带 time_maa 改名/低秩转置），**RWKV7 不要用**（会转置错方向），仅老版本模型参考 | 2026-08-31 自 `C:\work\ai00_server\assets\scripts` 归档 |
| `convert_tokenizer.py` / `json2kbnf.py` | tokenizer / 语法转换（伴生工具） | 同上 |
| `client/scripts/sync-models.py` | 本地 models 目录 → ModelScope + HuggingFace 全量同步 + manifest 重建 | client 仓库 |

依赖运行方式：脚本带 PEP 723 头（pytorch-cu124 index），`uv run tools/xxx.py` 即可。

## 标准流水线（以 g1j 2.9b → rwkv7-3B-int8.st 为例）

```powershell
# 1. pth → fp16 st（~5.5GB，2-3 分钟）
uv run rwkv-rsv/tools/convert_pth_to_st.py `
  --in "$env:USERPROFILE\Downloads\rwkv7-g1j-2.9b-20260831-ctx16384.pth" `
  --out test\rwkv-work\rwkv7-3B.st

# 2. fp16 st → int8 st（CUDA 下 ~40s；产出 .ai00-x-dev/models/rwkv/）
uv run rwkv-rsv/tools/quantize_any4.py --bits 8 `
  --in test\rwkv-work\rwkv7-3B.st `
  --out .ai00-x-dev\models\rwkv\rwkv7-3B-int8.st `
  --report test\rwkv-work\int8-report-3B.md

# 3. 结构校验（int8_idx 允许 2D [M,K] 或 3D [M,K/128,128]，运行时按字节量反推 K）
uv run rwkv-rsv/tools/verify_st.py .ai00-x-dev\models\rwkv\rwkv7-3B-int8.st
```

验收标准（int8）：量化器输出 `权重级验收: PASS`（avg_cos ≥ 0.999 且 avg_rel ≤ 1%）。

## 上传（MS + HF 替换旧文件）

全量同步用 `sync-models.py`（要求本地 models 目录为完整镜像，否则 manifest 全量重建会
**抹掉未在本地目录中的组件**——危险）。单文件替换/改名用补丁式脚本（本轮为
`client/test/upload-rwkv-rename.py`，一次性脚本，用后已删）：

1. 拉远端 `manifest.json`（MS base）→ 只增删目标条目、本地实算 sha256 → 绝不全量重建
2. HF：`HfApi.upload_file`（新文件 + manifest）+ `delete_file`（旧文件）
3. MS：git clone（`GIT_LFS_SKIP_SMUDGE=1` + `lfs.concurrenttransfers=1` 等 LFS 稳定参数，
   凭据在 `~/.modelscope/credentials`）→ 拷新文件 + `git rm` 旧文件 + 写 manifest → push
4. 先 `--dry-run` 核对清单再实跑

## 档位统一命名（进行中）

| 档位 | 旧文件 | 新文件（统一名） | 状态 |
|---|---|---|---|
| 3B | `rwkv/rwkv7-g1i-2.9b.int8.st` | `rwkv/rwkv7-3B-int8.st` | ✅ 2026-08-31 完成（g1j 20260831, ctx16384；3,295,783,296 B；avg_cos=0.999881 / avg_rel=0.6139%，详见 [int8量化报告-g1j-3B-20260831.md](int8量化报告-g1j-3B-20260831.md)） |
| 7B | `rwkv/rwkv7-g1i-7.2b/rwkv7-g1i-7.2b.int8.st` | `rwkv/rwkv7-7B-int8.st` | ✅ 2026-08-31 完成（g1j 20260831, ctx16384；7,898,815,456 B；avg_cos=0.999846 / avg_rel=0.6063%，详见 [int8量化报告-g1j-7B-20260831.md](int8量化报告-g1j-7B-20260831.md)；旧 7.2b 子目录两文件已从双端删除） |
| 13B | （仓库尚无文件） | `rwkv/rwkv7-13B-int8.st` | ⏳ 等 g1j 13B 权重 |

配套代码：`client/src/apps/desktop/src/rwkv_llm.rs` 的 `BUILTIN_RWKV`（st_rel/vocab_rel）。
vocab 三档共用顶层 `rwkv/vocab.json`（65536 World 词表；7B 独立 vocab 子目录待 7B 迁移时一并收敛）。

## 坑位记录

- **低秩转置**：ai00_server 的 `convert_safetensors.py` 会转置 w1/w2/...，RWKV7 原生 pth
  键名已与 rwkv-rsv 一致且方向正确，**再转置会算错**。用 `convert_pth_to_st.py`。
- **manifest 全量重建**：`sync-models.py` 从本地目录重建 manifest，本地目录不全 = 组件丢失。
  单文件操作必须走补丁式更新。
- **int8_idx 形状**：量化器产出 3D `[M, K/128, 128]`（group 打包）与 2D `[M, K]` 字节布局等价，
  rwkv-rsv 加载按字节量处理，两者皆可。
- **新旧文件同体积**：g1i/g1j 2.9b int8 均为 3,295,783,296 B（同架构），本地扫描按大小区间
  判定就绪，旧文件用户不受改名影响。
