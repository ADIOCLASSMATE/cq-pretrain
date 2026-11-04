# 结合原始语境的训练质量复核

本模块针对此前提交人工复核的688张DCVLM图像，恢复原始caption、对话或交错正文，再对图像和原始任务一起评估。它是独立质量层，不修改旧模型、人审结果、源图像，也不自动发布训练集。

## 当前生效规则

最新用户规则为“带模糊的直接不要”。语义质量建议只是独立证据：即使caption仍然有用，确认的局部模糊、整体失焦、运动模糊或明显背景虚化都排除。绘画笔触、材料纹理、平滑渐变和分割掩膜不能仅凭外观形式判为模糊；边界图在源TAR原图和上下文中复核，仍不确定则pending。

本次严格处理覆盖原人工队列688张，实际有效索引由旧复核根目录的`QUALITY_POLICY_CURRENT.json`指向。`HUMAN_REVIEW_CURRENT.json`保留历史人审；此前人审AI排除不能被质量意见解除。原始TAR保留，排除作用于训练候选索引，不物理删除原文件。

严格流程为`blur_run`完整推理/审计 → 独立原图复核 → `strict_policy`绑定语义、模糊和来源证据 → `apply_quality`写入独立不可变版本并原子更新指针 → `strict_report`从已应用的真实索引生成报告。当前报告在`context_review_20261025/strict_blur_20261025/report/`；`report/`及`superseded_report_v1/`是历史证据，不能当作最新处理结论。

生产`process`入口默认加载`config/quality_policy.json`，默认作用范围是本次已复核的688张。范围内未确定的图像留在待处理区，clear仍必须通过独立AI筛选；其他图像记录为范围外、未复核，继续原有流程，不自动声称模糊检查通过。发布及读取时再次校验绑定。本次688张不能代表全DCVLM已经完成模糊过滤。详见[生产入口说明](../production/README.md)。

## 范围与含义

本次包含583个caption pair、9个conversation、96个交错记录中的目标图像。每个图像ID只检查人工队列中指向的那个源记录；不代表该图所有重复出现位置，也不代表DCVLM全库随机样本。

- `keep / good`：图文对应，适合当前通用知识预训练；不等同于后续高分辨率精修阶段的高质量认证。
- `flag_low_information / usable_low_information`：描述或视觉信息弱，但没有明确错误，建议保留并标记。
- `exclude_pair / unusable`：独立图文/对话配对明显不可用。
- `drop_image_keep_text / unusable`：建议移除交错记录中的这个目标图像位置；文字仅作为独立保留候选，没有验证其他未显示图像或整篇序列。
- `needs_context`：现有证据仍不足；不伪装成坏图或技术失败。

图像损伤与真实错配分别统计：`image_quality=unusable`优先按图像损伤计，不因为主体模糊无法对应caption就声称原配对一定错误。只有图像可辨、文本却明确不相符的情况才单列错配。人工AI排除与本轮语义质量判断独立；新质量keep不能推翻AI来源决定。

## 执行

先运行`extract.py --root <旧复核根目录> --output <新根目录>`，验证原图SHA和同record TAR成员，保留完整原始文本、源字节证据、其他图像位置和人工决定（仅审计字段）。提取器拒绝覆盖已有产物。

`run.py`复用已验证的SQLite单写入者、WAL、逐项重试、断点续跑与自适应并发调度。新ContextFarm额外固定本模块源码哈希，并将实际prompt逐项绑定到冻结的源语境。使用真实解码图像附件，固定`gpt-5.6-sol`、`low`，每批最多4图、最多32并发；合计文字超过80000字符的候选批次自动减为单图。本批每项100000字符预算下688项均没有截断。

```bash
.venv/bin/python -m corpus_pipeline.context_review.run init --root <新根目录> --manifest <新根目录>/candidates.jsonl
.venv/bin/python -m corpus_pipeline.context_review.run run --root <新根目录> --workers 32 --initial-workers 4 --batch-size 4
.venv/bin/python -m corpus_pipeline.context_review.run export --root <新根目录>
```

先运行两批8图的真实试验并执行严格部分导出、核看实际附件，再扩大并发。SIGTERM停止补充任务、等待在途响应提交；成功项不会重跑。本次已实际验证24个在途图像安全排空后继续。

## 审计和产物

`candidates.jsonl`是冻结模型输入，`source_evidence.jsonl`包含原始TAR文本、成员偏移和哈希。模型prompt只采用白名单语境字段，历史人工/模型判断不进入推理。

每次调用保存真实图像附件、解码证据、prompt、schema、命令、原始回复、完整事件流、线程和用量。严格审计检查全部成功项与源输入、原始回复、事件、解码附件、导出分片的对应关系及SHA；只有完整通过才生成新运行的`COMPLETE.json`。

报告保留历史决定和新任务质量建议两列，提供逐项索引与完整语境浏览，便于解释为什么某张图适合或不适合原任务。最终产物说明以运行目录内的报告为准。
