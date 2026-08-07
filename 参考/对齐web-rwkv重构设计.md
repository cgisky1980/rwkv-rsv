# rwkv-rsv 对齐 web-rwkv 重构设计

> 日期：2026-08-07
> 目标：把 rwkv-rsv 从「单态 GpuModel」重构为 web-rwkv 风格的推理库，供
> ai00-server 作为后端集成（对标 web-rwkv 的 library 分层）。
> 参考：`参考/web-rwkv/src/{context,runtime/model,runtime/infer,runtime/{v7},tokenizer}.rs`

## 1. 对标要点（web-rwkv 的公共抽象，服务端直接使用）

| web-rwkv 抽象 | 作用 | rwkv-rsv 现状 | 差距 |
|---|---|---|---|
| `ModelInfo` | 模型元信息（num_layer/num_emb/num_vocab/num_head…） | `ModelConfig`（私有字段） | 需公开 + 语义对齐 |
| `State` trait | 状态一等公民：`init`/`load`/`back`（CPU↔GPU 序列化） | `Vec<GpuState>` 内嵌于 GpuModel，不可导出 | **核心缺口**，支撑 state tuning 与会话保存 |
| `Model`/`Bundle`/`ModelBuilder` | 加载模型 + 绑定状态 | `GpuModel::from_safetensors` | 需对齐分层 |
| `Runtime` + `infer(RnnInput)` | chunked 增量/批量推理 | `forward`/`forward_seq` | 需包装 |
| `Tokenizer` | 文本↔token | 已移植 `src/tokenizer.rs` | ✅ |
| softmax + sampler | 采样在服务端，不入库 | 无 | 服务端实现 |

## 2. 阶段划分（每阶段保持 `cargo build --release` + clippy 零警告）

### A. State 独立化 + 序列化（最高价值，先做）✅ 已完成
- 新增 `pub struct State`：持有 `Vec<GpuState>`（每层 tmix_x/tmix_rnn/cmix_x）+ `v_first`/`v_first_set`
- `State::new(rt,c,h,n,n_layer)` / `reset` / `back(rt)->Vec<f32>`（全态下载）/ `load(rt,&[f32])`
- `GpuModel::create_state` / `state_back` / `state_load` 公开封装
- `forward`/`forward_seq` 改为接受外部 `&mut State`（`forward_with_state`/`forward_seq_with_state`），旧签名作为便捷封装复用内部态
- 演示：`cargo run --release -- statetune`（前进→`state_back`→扰动→`state_load`→前向，验证闭环）
  - 运行结果：state 尺寸 5,409,280 f32（=32×(2560+40·64·64+2560)+2560），tuned 后 logits top-5 显著变化，闭环可用
- 编译 `cargo build --release` + `cargo clippy --release --all-targets` 零警告 ✅

### B. ModelInfo + Bundle + 公开 API 对齐 ✅ 已完成
- `pub struct ModelInfo`（对齐 web-rwkv 字段语义）：version(ModelVersion::V7)/num_layer/num_emb/num_hidden(=num_emb)/num_vocab/num_head/head_size/ffn_hidden/低秩 mid；`ModelConfig::info()` 派生，`GpuModel::info()` 公开
- `pub struct ModelBuilder`：`new(path)` + `build() -> Bundle`（封装 Runtime 创建 + safetensors 加载 + 绑定 State）
- `pub struct Bundle`：聚合 `model` + `state`，提供 `infer(&RnnInput)`/`infer_tokens`/`infer_seq`/`infer_chunked`/`state_back`/`state_load`/`reset`/`info`
- 编译 + clippy 零警告 ✅

### C. infer / RnnInput chunked 增量推理 ✅ 已完成
- `pub enum RnnOption`（对标 web-rwkv）：Last/Full（输出模式）
- `pub struct RnnInput`（对标 web-rwkv）：`tokens` + `chunk_size` + `option`
- `Bundle::infer(&RnnInput)`：Last 模式按块用 `forward_seq_with_state` 逐块推进同一 `state`（块间 RNN 状态自然累积，避免单次超大 prefill）；Full 模式逐 token 前向收集每位置 logits（prompt 打分/state tuning 用）
- 块大小固定可复用 seq 缓冲；`infer_tokens`/`infer_seq`/`infer_chunked` 为便捷封装
- 基准读取真实 web-rwkv 源码（`参考/web-rwkv/src/runtime/{model,v7,infer/rnn}.rs`）逐字段对齐
- 编译 + clippy + fmt 零警告 ✅

### 阶段全部完成 ✅
- A（State 序列化）、B（ModelInfo/Bundle/ModelBuilder）、C（chunked infer）均已落地，库已具备服务端集成所需的全部原语。
- 下一步：升级 ai00-server 用 rwkv-rsv 作后端，完善 OpenAI 接口协议（/v1/chat/completions + SSE 流式）。

## 3. 约束与风险
- 不破坏现有推理内核（kernels/shader 不动），只做外层状态解耦
- State 序列化格式需跨版本稳定（供 ai00-server 存会话/state tuning 文件）
- 每阶段编译 + clippy 零警告后进入下一阶段