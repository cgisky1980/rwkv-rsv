# 去掉 any4，只保留 fp16 / int8 完成记录

> 状态：完成。代码库中已移除全部 any4（4-bit 学习码本量化）推理路径，
> 量化仅保留 fp16 与 int8 两种。`cargo build --release`、`cargo clippy --release
> --all-targets` 零警告，`cargo test --release` 全量通过（lib 51 + 集成 1）。

---

## 1. 背景

此前推理路径按模型文件自动路由 fp16 / int8 / any4 三路。any4 为
arXiv:2507.04610 的 4-bit 学习码本（idx/lut/sz 三路张量）量化。为收敛量化面、
降低维护与 dispatch 开销，现仅保留 fp16 与 int8。

---

## 2. 改动清单

### 2.1 Rust 源码

| 文件 | 改动 |
| --- | --- |
| `src/backend.rs` | 删除 `Any4Handle` 结构体及 trait 中所有 `gemv_any4_*` / `gemm_any4` / `dequant_any4_to_f16` 方法（前轮已完成） |
| `src/runtime.rs` | 删除 `GpuTensorAny4` 及任何 any4 运行时实现（前轮已完成） |
| `src/backend_cuda.rs` | 删除 any4 kernel 定义与 trait 实现；GEMV 泛型 `wtype` 仅保留 fp16(0)/int8(2)（前轮已完成） |
| `src/gpu_model.rs` | 删除 `load_any4` 闭包；`load_linear` 收敛为 int8→fp16 二路；`load_ffn_value_tiled` 去掉 any4 分支；att.output 加载去掉 any4 分支；`GpuLayer` 构造/`drop_weight_hosts` 去掉 `*_a4` 字段；删除 `dequant_any4` 函数；forward_seq 中 att_output / ffn_value 去掉 any4 分支 |
| `src/model.rs` | 删除 `dequant_any4` 函数、`linear_to_f32` 的 any4 回退、`ffn_hidden` 推导的 any4 键回退；测试模块 `any4_tests` 收敛为 `quant_tests`（仅保留 int8 测试），删除 any4 专属测试与 k-means 合成器 |
| `src/main.rs` | 注释清理（int8/any4 → int8） |
| `src/lib.rs` | 文档更新为 fp16 / int8 |

### 2.2 Shader / 构建

- `assets/shaders/src/*.comp` 与 `assets/shaders/spv/*.spv` 中已不存在 any4 内核；
- `build.rs` 编译清单已无 any4 引用。

### 2.3 保留项

- `tools/quantize_any4.py` 仍保留（`--bits 8` 负责 int8 量化，idx/sz 打包契约与
  int8 GEMV/GEMM 一致）；
- `model.rs` 的 `quant_tests` 内 int8 格式契约 / 精度 / 常数组测试保留并通过。

---

## 3. 验证

```
cargo build --release            # 通过，无警告
cargo clippy --release --all-targets  # 通过，零警告
cargo test --release             # lib 51 passed + 集成 1 passed
cargo test --release int8        # quant_tests int8 3 项 + backend_cuda int8 3 项全过
```

后续 kernel 融合优化（fp16/int8）从当前基线继续。