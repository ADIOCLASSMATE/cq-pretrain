# 来源适配结论

本文件只归纳既有审计证据；没有重新挂载 CPU 数据、读取图像或把历史文件存在性当成本轮检查结果。

`config/adapters.json` 启用一条 Infinity-MM 路径规则：将旧前缀 `/inspire/hdd/project/deepgen/public/data_zoo/source_images/Infinity-MM` 替换为运行时的 `{collection_root}/Infinity-MM`。匹配必须是完整前缀及 `/` 边界，保留后续 stage、子集、分片和文件名。没有按 basename 搜索。

2026-10-21 的 `annotations.json` 中三个 JSON 文件各抽查三条记录，共九条图像引用，都与该规则产生的路径一致，且当时 stat 确认存在。本次对九条历史证据完成了离线字符串映射验证。`annotation_semantics.json` 另确认对应记录有非空 assistant 回答，但这些证据不证明图像内容与回答语义一致，也不证明其他记录完整。运行时仍须重新检查文件存在。

LLaVA-OneVision-Mid-Data 的示例图像路径相对于来源根目录；llava_ov_data_new 的示例路径已包含来源名，相对于整个 gen-und 根目录。现有显式 `source_root` / `collection_root` 候选覆盖这两种情形，无须再加旧前缀替换。

OpenUni/Megalith 已观察到 `caption_internlm2[_short]`、`caption_florence2[_short]`、`caption_sharecap[_short]` 六种描述字段，而通用 `caption` 与部分 TXT 只是 Flickr URL。这些字段应作为后续明确配置的 caption 视图保留，不能把同图六种重描述计为六张独立图。还需建立 `captions` 与真实图片的确定关联，因此配置中只记录为未启用的适配建议，不会自动猜列或把 URL 当描述。

OpenUni/laion6m 的 manifest 包含 `image` 和 `annotation` 引用，独立 caption JSON 有 `caption`、`detailed_caption`；未来应联结并检查两侧文件，不能将 annotation 文件名当正文。CC12M_and_Imagenet21K_Recap 与 Stable_Diffusion_3_Recaption 在既有有界审计中未找到抽样标注文件，当前没有证据可配置本地字段映射。

证据文件的绝对路径、Infinity-MM 九个映射实例、未启用的字段适配建议，以及官方数据卡链接均保存在 `config/adapters.json`。
