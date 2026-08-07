# rwkv-rsv Vulkan Compute 着色器完成记录

> RWKV-7 推理专用的 Vulkan GLSL compute 着色器集合。参考 `rosalia/assets/shaders/src/` 风格（extensions、buffer_reference、I_TYPE/O_TYPE 宏、specialization constants）。

## 着色器源码位置

- **目录**：`c:\work\niceui\rwkv-rsv\assets\shaders\src\`
- **编译产物目录**（与 rosalia 一致）：`c:\work\niceui\rwkv-rsv\assets\shaders\spv\`

## 通用约定

### Extension 头部声明（3 个新着色器共用）

```glsl
#version 450 core
#extension GL_KHR_shader_subgroup_basic : enable
#extension GL_KHR_shader_subgroup_arithmetic : enable
#extension GL_EXT_scalar_block_layout : enable
#extension GL_KHR_memory_scope_semantics : enable
#extension GL_EXT_shader_explicit_arithmetic_types : enable
#extension GL_EXT_buffer_reference : enable
#extension GL_EXT_control_flow_attributes : enable
```

> 注：与 rosalia 的 `gemv.comp` / `norm.comp` 相比，去掉了 `GL_EXT_shader_subgroup_extended_types_float16`。本批次着色器未直接使用 f16 subgroup 扩展类型，按需可再加。

### 编译时宏（-D）

| 宏 | 含义 |
|----|------|
| `I_TYPE` | 输入 buffer 元素类型（`f16` / `f32`