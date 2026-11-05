# Qwen3.5 / Wan2.1 视频 CQ 接口

模型运行实现分为 `backbone_*`、`generator_*` 和 `modeling_umm.py`。
`models/references/` 只保存原生源码，训练和推理不导入该目录。
当前训练入口只使用 Qwen3.5 / Wan2.1 视频主链路；不再使用的旧 Qwen3 / Z2 图像路线备份已清理。原生对照源码仍保留在 `models/references/`。
统一数据管线与启动方式见 [视频训练说明](video_training.md)。

## 双流约定

- Content 压紧所有有效 token。训练只构造受监督文本和目标视频的 query；调试输出还原为 `[B,L,D]`，未计算的 query 位置为零。
- Content 默认在图像组内、视频原生时间块内双向可见，块间 causal。GDN 在同一块全部写入后读取该块，短卷积保持真实 C 历史。`model.content_visibility=causal` 用于原生对照。
- 文本 query `Q_i` 读取 `C_<i`，预测同位置 `labels[i]`，不 shift。
- 同一视频所有 visual query 读取第一个 visual token 之前的前缀。
  帧间字符仍处于原始 content 位置，其 text query 可以读取此前视觉内容。
- `query_read_end[B,L]` 表示物理位置的排他边界；输入该字段时会验证它与视频分组一致。
  旧 `sigma` 不控制 Qwen3.5 的计算顺序。
- 核为 4 的 query 短卷积使用最后三个允许的 content 原始 q 投影和当前 query 的 q。
  不包含其他 query，不修改 content 卷积历史。
- GDN 将只读 query 事件插在对应 content 写入之前，query 的 K/V/beta/g 为零。
  Content 更新相对顺序不变。Full Attention 的 query 只读取 content K/V。
- CFG 只删除选定完整视频 query 的前缀读取，不改变 content。
- video 的文本生成不缓存跨步 KV、卷积或 GDN 状态；Wan 条件投影与 cross-attention K/V 按请求缓存。

`models/qwen35_layers.py`、`models/wan_layers.py`、`ops/` 和 `distributed/` 使用 main 的实现；视频包装层只负责输入、原生位置和采样接口。原生源码仅用于测试对照。

## 装配与权重

```python
from models import UMM, UMMConfig

# 从原生本地组件加载。两个 mask ID 必须来自已经扩展的 tokenizer。
model = UMM.from_pretrained_components(
    "/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Qwen3.5-2B",
    "/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Wan2.1-T2V-1.3B",
    mask_token_id=text_query_token_id,
    video_mask_token_id=video_query_token_id,
    wan_attention_backend="auto",  # NPU tensor 自动选择 NPU backend
)

# 保存配置后，可通过统一入口重建同一结构；权重另外加载。
config = UMMConfig.from_dict(model.config.to_dict())
restored = UMM(config)
restored.load_state_dict(model.state_dict(), strict=True)
```

视频模型配置使用 `backbone_type="qwen35"`、`generator_type="wan21"`，
并包含原生 `qwen_config`、`wan_config`、query token ID、视觉冻结与 checkpoint 配置。
训练 YAML 的 `model` 保存组件路径和训练控制，加载后在 `model.architecture` 保存上述原生结构配置。
旧 `cq_it/z2` 配置不再属于运行入口。

视觉编码器现在属于 `backbone.vision_encoder`，参数仅注册一次。
与早期独立 `vision_encoder` 的 UMM 视频检查点相比，该部分 state-dict 前缀发生变化；
原生 Qwen/Wan 组件加载映射保持有效。`model.vision_encoder` 提供只读访问属性。

## 前向输入

Batch 使用原生 processor 的 `input_ids`、`attention_mask`、`mm_token_type_ids`、
`video_grid_thw` 和 `pixel_values_videos`。每帧占位与标记遵循原生布局。

可用 `video_features` 替代像素 patches：它是按视频顺序排列的 post-merger 特征列表，
每项形状为 `[visual_tokens,D]`。未知目标视频对应 `None`，并通过
`video_observed_mask[videos]` 标记为 False。同一 batch 的 Qwen 表示统一使用缓存或在线像素，不接受 `pixel_video_ids`。

`target_video_ids` 指定送入 Wan 的视频及顺序；省略时选择全部视频。
文本理解无需调用 Wan，可以显式传入空目标列表。
`labels[B,L]` 中 `-100` 表示不监督；只对文本和特殊 token 计算文本输出。

```python
predictions = model(
    batch,
    noisy_latents=latents,  # list of [C,T,H,W]，顺序与 target_video_ids 相同
    timesteps=times,       # [target_videos]，已经是 Wan 模型使用的时间单位
    return_hidden_states=True,
)
# query_hidden/content_hidden: [B,L,D]
# text_logits/text_labels: 对齐的监督文本位置
# condition/condition_mask: 按目标视频收集并补齐的 query 条件
# velocity: list of [C,T,H,W]，FP32 noise-minus-clean 预测
```

训练设置 `compute_text_loss=True`，通过共享分块交叉熵返回 `text_loss`，不保留完整词表 logits；flow loss 及其归一化仍由训练层计算。Qwen visual token 数量与
Wan latent token 数量不要求相同。

## 无缓存推理

```python
tokens = model.generate_text(
    prompt_batch, max_new_tokens=128, use_cache=False,
    temperature=0.0,
)

latents = model.generate_video(
    target_layout_batch,
    latent_shapes=[(16, latent_t, latent_h, latent_w)],
    steps=30, solver="euler", cfg=3.5, time_shift=5.0,
)
```

文本每步在实际前缀末尾放置一个占位，执行完整等长双流前向，取该位置的 query
经 LM head 预测，然后用采样 token 替换占位。只使用一个输出不代表只计算一个 query。
输入视觉特征只提取一次；不保存跨步的语言模型 KV、卷积或 GDN 状态。
采样排除内部 query/视觉占位 token，原始 logits 可通过 `return_logits=True` 返回供对齐检查。

T2V 输入包含目标视频的完整原生占位布局和已知网格，不需要目标像素。
`generate_video` 排除所选目标的 pixels/features，保留其它已知视频。
一次 backbone 前向产生固定条件；CFG 使用同一批次中的有条件／无条件两份输入。
随后仅运行 Wan 采样循环，条件投影与每层 cross-attention K/V 在该请求内缓存一次。输入 latent shapes 是 Wan VAE 坐标，不从 Qwen grid 猜测。

采样支持 Euler/Heun，积分方向是 sigma=1 到 0，模型预测为 noise-clean，
传给模型的 timestep 为 `sigma * timestep_scale`（默认 1000）。
`time_shift` 显式控制时间网格，默认 1。该接口是普通 flow 积分器，不是原版 UniPC。
也可传入 `initial_noise` 保证可复现；它不会被原地修改。
VAE 留在编解码层，通过可选 `decode(latents)` 回调完成像素视频输出和 latent 缩放。

## 验证

```bash
python -m pytest tests/test_backbone_qwen35.py tests/test_modeling_umm_video.py tests/test_generator_wan21.py
UMM_VIDEO_DEVICE=npu:0 python -m pytest tests/test_modeling_umm_video.py
python tests/validation/validate_wan21.py --help
```

测试覆盖共享层的原生因果数值对照、时间块双向可见性、跨 chunk 梯度、整视频与帧间字符无泄漏、
query 独立性、CFG 不改 content、同位置训练与无缓存推理 logits 对齐、Wan 积分方向、
原生分片权重加载与序列化，以及运行时禁止导入 references。
CPU 小配置验证不能代替真实 Ascend 设备和完整预训练模型的性能验证。
