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

### D. GPU 采样接入 Bundle（全 GPU，不下载 logits） ✅ 已完成
- `GpuModel` 已有 GPU 采样内核：`forward_argmax`（单 token argmax，只回传 4 字节索引）与 `forward_argmax_selfloop`（单次 submit 内连续采样 n 个 token，结果直接写回 host 缓冲，无 CPU↔GPU 交换）
- 新增 `Bundle::infer_argmax(&mut self, tokens) -> R<u32>`：复用外部 `state` 的 `forward_argmax_with_state`
- 新增 `Bundle::infer_argmax_selfloop(&mut self, seed, n) -> R<Vec<u32>>`：复用 `forward_argmax_selfloop_with_state`
- 采样全在 GPU 完成，只在 self-loop 结束时一次性下载 n 个 token；服务端生成循环无需下载每 token 的 logits
- 编译 + clippy + fmt 零警告 ✅

### E. GPU 采样参数化（temperature / top-k / top-p） ✅ 已完成
- 新增 `sample.comp` shader：单 workgroup 内做 temperature 缩放 → top-k 阈值过滤 → softmax → top-p 累积截断 → splitmix 随机数按概率采样，输出 token 索引
- 采样参数放 host-visible `sampler` 缓冲 `[temperature, top_k, top_p, seed]`，CPU 用 `store_sampler_host` 直写，shader 读 device address（无 spec constant，无每参数重建 pipeline）
- `runtime`：`sample`（自建临时缓冲，单次调用）、`sample_into_host_seeded`（复用预建缓冲，self-loop）、`store_sampler_host`
- `gpu_model`：新增 `SamplerParams`（temperature/top_k/top_p/seed），`forward_token` 增采样分支，新增 `forward_sample_with_state` / `forward_sample_selfloop_with_state`
- `Bundle`：新增 `infer_sample(tokens, &SamplerParams)` 与 `infer_sample_selfloop(seed, n, &SamplerParams)`
- 采样全程 GPU，只回传 4 字节索引；temperature>0，top_k=0 禁用，top_p>=1 禁用
- 编译 + clippy + fmt 零警告 ✅

### F. GPU 采样惩罚（repetition / frequency / presence，兼容 OpenAI 参数） ✅ 已完成
- 新增三个与 OpenAI / vLLM / llama.cpp 主流一致的惩罚，作用于 softmax 前的 logits：
  - `repetition_penalty`（缩放，1.0=禁用）：出现过的 token，logit>0 时 /=rp，logit<0 时 *=rp
  - `frequency_penalty`（次数偏移，0.0=禁用）：logit -= fp × 出现次数
  - `presence_penalty`（存在偏移，0.0=禁用）：出现过的 token 一律 logit -= pp
- `sample.comp` 新增惩罚阶段：由历史 token 直方图计数（`counter` 为 vocab 长 u32 缓冲，`atomicAdd` 统计），再按计数施加三惩罚；顺序为 载入 logits → 惩罚 → temperature → top-k → softmax → top-p → 采样
- `sampler` 缓冲扩为 8 元素 `[temperature, top_k, top_p, seed, rep, freq, pres, hist_len]`；并修正打包：float 读回用 `uintBitsToFloat` 直接写 f32，uint 读回用 `f32::from_bits` 保持位（修复了原 4 元素版 `to_bits() as f32` 数值转换导致的参数错位）
- `runtime`：`sample` 增 `repetition_penalty/frequency_penalty/presence_penalty/history`；`sample_into_host_seeded` 增 `counter` 与 `hist`；`record_sample` 透传 counter/hist 地址
- `gpu_model`：`SamplerParams` 增三惩罚字段；`forward_token`/`forward_sample_with_state` 透传历史；self-loop 以累积的 `token_seq`（前 round 个）作惩罚历史，hist_len=round
- `Bundle`：`infer_sample(tokens, &SamplerParams, history)` 增历史参数；`infer_sample_selfloop` 历史由内部累积
- 编译 + 测试 28 passed + clippy + fmt 零警告 ✅

### 阶段全部完成 ✅
- A（State 序列化）、B（ModelInfo/Bundle/ModelBuilder）、C（chunked infer）、D（GPU 采样接入 Bundle）、E（采样参数化 temperature/top-k/top-p）、F（采样惩罚 repetition/frequency/presence）均已落地，库已具备服务端集成所需的全部原语。
- 下一步：升级 ai00-server 用 rwkv-rsv 作后端，完善 OpenAI 接口协议（/v1/chat/completions + SSE 流式）。

### G. 参考 web-rwkv 的示例/跑测程序 ✅ 已完成
向 `examples/` 新增 4 个独立跑测程序（`MODEL_PATH` 缺省 `c:\work\niceui\rwkv-g1h-3B.st`）：
- `model_info`：加载模型打印 `ModelInfo`（version/layers/emb/vocab/head/ffn_hidden/w-a-v-g mid）+ probe 前向 top-5 logits
- `generate`：prefill prompt 后 GPU self-loop 生成（`GEN_MODE=argmax|sample`，`TEMP/TOPK/TOPP` 可调，`VOCAB_JSON` 打印文本）
- `benchmark`：对比 `infer_seq`(prefill) / `infer_tokens`(decode) / `argmax_selfloop` / `sample_selfloop` 的 tok/s
- `state_persist`：`前进→state_back→存盘→state_load→state_back` 闭环，验证序列化逐位无损
- 编译 + fmt + clippy 零警告 ✅；四个示例均实际跑通（3B 模型）

**重要发现：GPU 前向“非确定性”已被定位并修复（根因是 `Bundle::reset` 重置错对象）**
- 现象：同一输入连续前向两次，logits `max_abs_diff ≈ 10`；但 `back→load→back` 整态 `max_diff = 0`（序列化本身逐位无损）
- 因此 state_persist 只断言「序列化往返无损」，不比较回灌后 logits（曾被误认为前向波动污染）
- 定位过程（决定性实验）：
  - 两个**独立** `Bundle`（各自全新 State）跑同一 prompt，state_back 逐层**完全一致** → 证明 GPU kernel 本身**确定性**，问题不在 shader/原子/竞态
  - 同一 `Bundle` 先跑一次、`reset()` 后再跑一次，第一个 token 起 layer 0 的 `tmix_rnn`/`cmix_x` 即发散（`tmix_x` 因被 ln1 覆盖而仍一致）
  - `FULL_BARRIER=1` 全量屏障无效 → 排除 kernel 间缺 barrier
  - 根因：`Bundle::infer_tokens`/`infer` 使用 **Bundle 自身的 `self.state`**，但 `Bundle::reset()` 误调用 `model.reset_state()`（只重置 **GpuModel 内部 `Option<State>`**），导致会话态残留、reset 后仍从上一 run 的 RNN 状态继续 → 伪“非确定性”
- 修复：新增 `GpuModel::reset_state_of(&State)`（重置外部会话态），`Bundle::reset()` 改调它重置 `self.state`
- 验证：同一 `Bundle` 的 fresh vs reset 现在逐层 `完全一致`；`cargo fmt` + `cargo clippy --release` 零警告
- 结论：串行化/确定性无问题；此前的非确定性纯为 reset API bug，非 GPU 数值波动

### H. 修复 `--all-features` 编译错误（rayon 依赖缺失） ✅ 已完成
- 症状：`Cargo.toml` 声明了 `rayon` feature 但未引入 `rayon` 依赖，`cargo build --all-features` 报 `cannot find module or crate 'rayon'`（`vulkan/layout.rs:672`）
- 修复：`[features] rayon = ["dep:rayon"]` 绑定可选依赖；`[dependencies] rayon = { version = "1.12", optional = true }`（本地缓存为 1.x，故用 1.12 而非 2）
- 验证：`cargo build --release --all-features` + `cargo clippy --release --all-features --all-targets` + `cargo fmt` 零警告；默认 features 构建无回归 ✅
- 备注：本次因 rsproxy.cn 镜像网络超时无法在线解析新依赖，改用 `cargo build --offline` 命中本地缓存完成验证（临时项目级 `.cargo/config.toml` 已删除，未改动用户级镜像配置）

## 3. 约束与风险
- 不破坏现有推理内核（kernels/shader 不动），只做外层状态解耦
- State 序列化格式需跨版本稳定（供 ai00-server 存会话/state tuning 文件）
- 每阶段编译 + clippy 零警告后进入下一阶段