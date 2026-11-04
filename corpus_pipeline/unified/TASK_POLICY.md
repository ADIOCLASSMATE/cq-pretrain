# 按来源与任务方向处理（2026-11-03）

新运行使用 `policy.profile = "task_aware_v1"`，绑定规则版本
`corpus.task_policy.v1`。已有运行、原始文件和已发行索引不就地修改。

| 任务 | 来源与检测 | 图像生成 loss | 文本监督 |
|---|---|---|---|
| T2I，来源已确认 AI 生成图 | 直接排除，不调用检测模型 | 不参与 | 该 T2I 记录不进入训练 |
| 其他 T2I | 延续 CF384/UniFD、疑似图复核及原门禁；完成复核仍不确定的排除 | 通过后允许，显式禁用标记仍优先 | 保留原 caption |
| I2T，包括 AI 生成图或未知来源图 | 跳过 AI 来源排除、检测及视觉复核，只进行去重 | **统一禁用** | 保留原文本监督及视觉输入 |

图像解码、引用绑定和有序监督的完整性仍须成立；损坏或错配不能作为有效样本。
评测集的隔离继续保留。交错是结构分类，不自动等同于 T2I；I2T 方向必须有明确合同或用户侧图像位置证明。
无法证明生成方向的混合/未知任务不自动取得图像 loss 权限。

同一张图可以同时关联被排除的 T2I 和被保留的 I2T。来源、检测和质量结果不跨任务连带删除 I2T。
权限附在监督记录上，`image_loss_allowed=false` 优先于 20% 或任何图像生成抽样概率；
训练端同时兼容旧 `dcvlm_image_loss_allowed`，记录级和图片级权限按布尔 AND 合并。
禁用的是图像生成目标及对应 VAE/DiT loss，视觉理解输入和文本 loss 保留。

## 配置

在新输出目录的配置中添加：

```json
"policy": {"profile": "task_aware_v1", "t2i_uncertain": "exclude"}
```

`genund_caption_tar` 输入必须显式指定 `direction: "i2t"` / `"t2i"`，
或用 `source_directions` 为该根目录内每个来源声明方向。两种写法互斥。
例如 `source_directions: {"BLIP3o-Pretrain-Short-Caption": "i2t", "laion-aesthetics-12m": "t2i"}`；
映射必须精确覆盖本次枚举来源，不能用缺省值猜任务方向。
新 profile 自动要求 `preserve_all_captions=true`。

已注册字段中的不同有效 caption 会各自生成可训练监督；完全相同的文本折叠并保留所有来源字段。
对话、角色、多图位置及交错顺序保持不变，不把多个 caption 拼成一条句子。
未知字段和未支持的载体需要明确适配，不默默忽略。

## 发行与读取

`source_view` 阶段发布两种相互绑定的视图：

- `training_candidate_source_view`：原 metadata 的逐行选择与权限 sidecar。
  `unified.release.iter_batches()` 读取时附上任务和 loss 标记，原 Parquet 不改写。
- `training_asset_supervision_view`：`assets.parquet` 对每个精确图像身份仅一行；
  `supervisions.parquet` 保留不同的完整有序监督；关系表连接图像与监督，来源表保留被精确去重的记录证据。

精确编码或解码像素一致用于图像身份去重。SSCD 相似候选不自动删除，也不随意合并不同裁剪。
这里只统一图像资产索引，不复制或删除原图；训练 sampler 的每图权重未改变。

## 本次验证范围

实现包含归一化、前置路由、对账、筛选、候选去重、最终读取权限，以及现有训练输入链路的权限消费。
CPU 小样本测试使用模拟检测输出；它证明控制流和索引关系，不代表真实检测器准确率或全量运行完成。
新策略需要新的执行目录和当前代码的运行/数值资格。

六个优先子集及 DenseFusion 补充源的载体已由 `production/genund_collections.py` 和 `unified/genund_loose.py` 接入。全流程入口见 `../README.md`；全量完成以各运行的最终合并回执为准。
当前训练 `Catalog` 接受 `dcvlm.pretrain.v1`，尚未直接接入 `corpus.native.v2`/unified 发行格式。
后续训练导入必须保留上述布尔权限，不能把本次读取测试视为实际训练已经消费新发行。
输入流水线的恢复契约也绑定了权限语义；旧契约 checkpoint 不能直接恢复为新语义，需显式开始新的训练阶段。
