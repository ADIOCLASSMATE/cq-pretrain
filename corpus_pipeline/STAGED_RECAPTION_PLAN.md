# DCVLM／gen-und 的分阶段补充描述策略

日期：2026-10-23（Asia/Shanghai）。本文保留分阶段策略的设计依据。**后续已实现可执行 v2 管道**，当前实现与限制见 [PIPELINE_V2.md](PIPELINE_V2.md)，真实验收见 [重构报告](REFACTOR_REPORT_20261023.md)。下文第 6 节描述的是重构前的代码差距，不代表当前实现状态。

用户希望前期混合原生监督与已有可用描述，只补必要缺口；后期约 1M 像素阶段再完整 recaption。本设计将 1M 理解为约 1024² 的训练图像总像素预算。原图尺寸、训练目标尺寸和标注模型实际看到的尺寸分别记录；上采样不能增加原图信息量。

## 1. 阶段目标

| 阶段 | 所需描述 | 保留方式 | 新生成的触发条件 |
|---|---|---|---|
| `early_mixed` | 至少一个符合原任务、正确配对且按当前质量政策可使用的文本视图；不要求三档齐全 | 原文、已有合格 recaption 均可训练，已有多粒度样本继续参与 | 确认缺可用描述、确认实质性内容错误、或原指令所需形式确实缺失 |
| `highres_1mp` | 对冻结的高分辨率 T2I／caption 型 I2T 图像池完成合格短、中、长描述覆盖 | 原文留存；同一图的不同描述为候选，依阶段配置选取 | 缺少符合该阶段图像细节、语言、任务及质量契约的描述；已有达标候选可计入覆盖 |

这里建议将后期“完整”定义为**实际进入该阶段描述任务池的完整覆盖**，不强迫所有原始文件重新调用一遍模型。若需要统一新模型的全新输出，应另开 `fresh_generation_coverage` 指标与新 annotation revision，不能和可用描述覆盖率混淆。

QA、OCR、分类、数学、编辑、多轮和交错记录保持自己的监督目标。若这些图片明确需要辅助 caption，可另选辅助图像池，生成 sidecar 并单独计数；不自动替换答案、编辑指令、对话或网页正文。纯文本不进入视觉 recaption。

前期仍需修复明显错误监督，不能将已确认错配和不可用文本统一推迟到高分辨率阶段。建议保留现有合格多粒度样本，让前期训练也接触这些条件形式；具体比例与分阶段收益要用训练对照验证。MOSS 公布的是每图三档描述及有条件的原文／recaption 采样，没有证明“仅最后阶段补全”与其原配方效果等价。[MOSS §3.2–3.3](https://arxiv.org/html/2610.23374v1#S3.SS2)、[附录 C.1](https://arxiv.org/html/2610.23374v1#A3.SS1)

## 2. “需要补充”的判定对象

判定对象是**某阶段、某个精确图像视图、某项原生任务所需的描述候选**。不能只给整个来源打一个 `needs_recaption` 布尔值。

概念上：

`需要补充 = 已进入当前描述任务池 AND 原图/原记录准入通过 AND 关联可靠 AND 当前没有满足任务与阶段要求的可复用描述`

如果关联、来源或语义状态不明，结果是“待恢复／待复核”，不是自动生成或自动通过。生成成功也不能代替后续质量验收。

| 观察到的情况 | 应采取的动作 | 判定方式 |
|---|---|---|
| 图片缺失、来源拒绝／未决、必需图片未通过尺寸或其他准入门 | 阻塞本记录，不做无效补写 | 继承现有来源、尺寸、AI 筛选与记录完整性规则 |
| 标注文件还没找到、字段未知、key 关联不明或错误 | `repair_metadata`：先恢复标注和关联 | adapter、source/shard/key、快照及实际文件证据；不能按行号 zip |
| 真正 caption 字段已恢复并检查，全部为空／缺失，且图像关联可靠 | `generate_missing`：补新描述 | 确定性规则；无需先花一次模型调用判断是否缺失 |
| generic caption 实际是路径、URL 或索引 | 先查该源真实描述字段；确实没有可用候选再补 | 字段合同证明其用途；含 URL 的有效 OCR／正文不能用同一规则删除 |
| 只需修正已确定的编码、包装等表面问题 | `clean_surface`：可追溯清理 | 仅执行来源限定且不改变语义的规则；保留原始文本，保护 OCR 数字、符号与换行 |
| 已有正确的原 caption 或已验收 recaption | `reuse_original`／`reuse_annotation` | 核对任务、语言、图像身份、质量证据与当前阶段适用性 |
| caption 简短、只具备一个长度，或来源为 AI 文本 | 前期继续保留符合任务的候选，不因这三项强制重写 | 长度、AI 来源与事实质量分别记录 |
| 相似度低、模板重复、可能说错主体／数量／位置、疑似 OCR 幻觉 | `review_caption` | 这些是风险信号；模型只判可用／需纠正／不确定，不把分数当真值 |
| 可靠复核确认 caption 有实质错误，图像和关联本身可靠，且无其他可用候选 | `regenerate_verified_failure` | 停用错误文本视图，另加正确描述；不删除原始记录或伪称恢复了错误关联 |
| 后期池中缺合格短中长描述 | `complete_highres_caption` | 看该阶段所需可见细节补齐，验收后计入覆盖；简单图片允许自然较短 |
| QA、OCR、标签、多轮、编辑、交错、纯文本 | `preserve_native_task` | 按原任务质量流程检查；“cat”“42”可能是合格类别或答案，不是劣质 caption |

没有满足规则的情况保留待审。低 CLIP／图文相似度只作候选召回，不设跨照片、图表、文档和多语言通用的删除阈值。长描述同样可能错误；短句也可能完全满足原指令。

## 3. 对现有来源的具体路由

| 当前集合／来源 | 先做什么 | 什么情况下才补 |
|---|---|---|
| DCVLM DataComp、ReLAION、GRIT | 读取已有 `.txt`／`segments.text`，保持原生 I2T；按来源和文本风险抽检 | 真正缺文本、无关模板、已确认描述错误；前期不因缺长描述全部重写 |
| DCVLM PixMo-Cap、COCO、OpenImages caption | 从原 conversation 的 assistant 描述恢复候选，保留用户指令 | 任务不匹配或内容错误才修；仅支持短回答的指令不能套长描述 |
| DCVLM InternVL-SA1B caption | 保留已有机器描述并标记 AI 来源，检查图文与事实 | 内容或当前阶段契约不合格才补；AI 文本本身不触发重写 |
| gen-und BLIP3o Long/Short、LAION-aesthetics | 验证同 shard/stem 的 TXT/JSON.caption，复用合格描述 | 缺失／错误；Short 与 Long 版本是否同图需要真实资产匹配 |
| OpenUni laion6m | 恢复已观察到的 `caption`、`detailed_caption` 及 image/annotation 关联 | 先用已有短／详细候选，只补确实不足的部分 |
| OpenUni Megalith | 读取独立标注里的六种 caption 候选；比较来源与质量 | generic caption/TXT 可能只是 Flickr URL，不能因此直接认定全图缺描述；六候选都不适用才生成 |
| CC12M+ImageNet Recap、SD3 Recaption、DenseFusion、ScaleCap 等 | 优先定位、配对原标注；目前“未定位”不等于“没有” | 恢复后判断；SD3 Recaption 不按名字当作全 AI 图拒绝 |
| FineVision、LLaVA 图片包、PixMo AMA 等 | 先恢复原 QA、多轮、多图或其他标注 | 无法恢复时不能用 caption 假造原任务；gen-und 如明确另建独立 caption 视图，须有新记录身份和完整准入 |
| JourneyDB 两个版本、MJHQ | 按已确认 AI 图／评测隔离规则排除训练用途 | 不投入训练 recaption |
| UniWorld、OpenUni 混合分支 | 使用窄路径规则、source/target/control 角色和真实父图编辑血缘 | 明确生成图排除；允许的编辑图仍须描述实际目标视图，保留编辑任务 |

这些是基于既有合同和有界样本的来源建议。PixMo/COCO 等说明 **`format=conversation` 不等于不是 caption**；必须补齐明确的来源＋指令＋结构映射，不能仅按目录名猜。完整字段、未决项及证据见 [逐来源路由](research/RECAPTION_SOURCE_ROUTING_20261023.md)。

## 4. 如何判断质量而不先把全库逐条 judge 一遍

先做廉价的全量元数据检查：字段语义、空值、实际引用、语言／格式约束、角色／轮次、重复候选、精确图像身份。已明确需要生成的 caption 不再先付一次“是否需要生成”的判断成本。

对已有监督建立来源／子源／任务／语言／领域／文本长度的分层抽检。随机基线与风险召回样本分别统计，保留抽样概率；风险集合的错误率不能当总体错误率。抽检发现某个 adapter 系统性配错，优先修 adapter 和重建关联；发现整个子源描述系统性失真，再决定对该分支重做，避免先给每条都调用一次 judge。

评审按明确问题输出证据：主要主体是否对应；动作、数量和关系是否有依据；是否擅自推断材料、身份、地点；原指令的语言和格式是否保持；该任务要求的可读文字是否忠实；当前阶段是否确实缺少重要信息。保留 `uncertain`，不强迫模型给出肯定结论。复杂 OCR、图表、编辑目标以及高分辨率新增细节需要相应领域的验收。

`source_stratum_audit` 与 `per_annotation_visual_review` 必须分开保存。来源抽检是批次风险控制，不证明每条文本已经看图验真。以来源抽检支持原文复用的发布政策须独立版本化、校准，不能伪造当前逐记录 Gemma QC 的 pass 回执。

## 5. 记录、复用与训练读取

每个需求计划至少保存以下内容：

```json
{
  "training_phase": "early_mixed",
  "source_snapshot_sha256": "...",
  "original_sample_id": "...",
  "asset_view_identity": "...",
  "native_task": "single_i2t",
  "semantic_task": "caption",
  "caption_contract_id": "...",
  "pairing_state": "verified",
  "caption_state": "usable_original",
  "caption_origin": "unknown",
  "quality_evidence_scope": "source_stratum_audit",
  "action": "reuse_original",
  "candidate_refs": ["source_annotation_reference"],
  "reason_codes": ["EARLY_TASK_HAS_USABLE_DESCRIPTION"],
  "training_admission": "requires_separate_existing_gates_and_release_policy"
}
```

候选目录独立于模型响应缓存。原文、上游 AI 描述、我们的新描述分别保留 provenance；未知文本来源保持 unknown。缓存结构合格不等于质量合格，人工隔离、质量撤销与当前来源门仍然有效。

同一精确图像可以共享合格图像侧描述，但语言、实际视野、任务约束及质量标准也须匹配。裁剪、编辑、改变文字的相似图不能仅凭 SSCD 共享 caption；多图记录和 QA 的质量证据仍绑定完整有序上下文。请求缓存继续保留既有模型、processor、prompt、上下文及 token 预算的严格身份，不能为了提高复用率放松。

训练时分别控制任务混合、图像总曝光、描述候选选择。原文／已有／新增候选的阶段权重单独冻结，每次读取选一个符合原指令的候选，不因同图有六条 caption 就变成六倍曝光。DCVLM 仍保留原方向，T2I 来自 gen-und 的适用视图。早、晚阶段各自冻结 release 与候选配置，后期更新不得改写已开始训练的早期版本。

1MP 池应同时报告唯一资产视图覆盖率、任务记录覆盖率、训练采样权重覆盖率，并按来源／领域分解。需要复核、生成失败或没有合格三档的项目保留 pending；不能悄悄从分母删掉再宣称 100%。在实际入选高分辨率池确定前，当前 8,891 万条 caption 的全重写耗时只是一种容量情景。

## 6. 当前代码差距及本次产物

已核对现有执行器：每条进入 annotation 分支的记录都走生成＋复核，成功缓存可能避免实际新请求；纯文本也未跳过 LLM。reader 对 caption 直接选生成版本的长短，尚未实现原文／已有／新增候选混合。现发布协议也不能把“保留原文”直接当作新模型生成结果。

最小接入顺序是：

1. 扩充固定来源的 caption 字段／精确指令合同，恢复 gen-und 原标注关联；建立候选目录。
2. 在模型调用前增加阶段需求计划，明确保留、复用、补缺、待审、后期覆盖分支；计划不授予训练准入。
3. 将保留／导入／复用回执与真实模型生成回执分开；按任务路由质量检查。Layout 保留独立候选选择入口。
4. 为 reader 与 release 增加阶段候选选择、原文混合、原生任务不可变约束和历史撤销处理。
5. 用固定小批验证任务保真、断点续跑、覆盖分母和零多余生成，再按当前挂载实际做分片统计。

本次落盘的 [阶段策略 JSON](prepared/config/staged_recaption_policy.json) 是 `production_execution_enabled=false` 的设计契约，**不是已运行的调度器，也没有产生全量需补写比例**。详细代码位置与缓存／发布边界见 [接入审计](research/STAGED_RECAPTION_INTEGRATION_AUDIT_20261023.md)。

gen-und 的 21 个父源目前都没有全量标注覆盖率；不能根据“8 个来源抽样见到文本”推算有多少图必须重写。下一步数量统计应先区分真实缺失、标注待恢复、可复用、质量待审、明确需重写与后期才需补充，按记录与唯一图像分别计量。
