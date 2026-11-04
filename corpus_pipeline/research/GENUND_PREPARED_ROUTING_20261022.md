# gen-und 21 个一级来源的准备路由合同

日期：2026-10-22。此次仅从已保存的 CPU 挂载审计、字段复查、现行来源规则与 adapter 配置准备合同；没有新 API 调用、集群操作或重新扫描挂载。

机器可读配置：[genund_source_contracts.json](../prepared/config/genund_source_contracts.json)。格式为 `corpus_pipeline.genund_parent_contracts.v1`，版本 `2026-10-22.prepared.1`，`contracts` 恰好覆盖 21 个一级目录，`source_id` 保留目录大小写。

这些合同把“已经知道如何分流”与“运行时仍须证实可用”分开。21 个一级来源都在历史审计中发现图像；其中 8 个来源抽样发现自然语言或对话文本，13 个在检查预算内未定位标注。上述数字不表示全量标注覆盖率、内容完整率或解码成功率。每条合同的 `full_coverage`、`unique_image_count` 和 `usable_record_count` 都保留 `null`，整体 `training_ready=false`。

## 使用边界

父源合同只提供准备知识，不替代 634 条来源/路径规则编译结果。每条 `source_filter` 引用现有规则 ID，并记录输入文件 SHA256。遇到具体路径/子源，仍由完整 policy resolver 判定；父目录任务家族只是先验，不给所有记录强行贴同一个类型。

现行口径是：AI 生成图像排除；确有自然父图血缘的 AI 编辑图可以保留并标注；程序绘制图表、渲染文字、CAD/3D 等非 AI 图像允许保留；AI 合成 caption 只标记。旧 `dataset_selection.json` 中“严格真实原图池排除 SynthDog/所有编辑目标”等历史建议已经被用户后续要求覆盖，本合同没有沿用这些旧建议。

DCVLM 始终保持原生任务方向和图文顺序。gen-und 内的 QA/OCR/文档理解也不能统一改成 T2I；只有明确的图文 caption 视图在配置指定后才生成 T2I 长、中、短描述。缺失原始问答时，模型新写的 caption 必须标成新派生视图，不能冒充恢复了原 QA。

## 逐父源路由

“上游任务”只说明来源定义，不证明本地已经恢复对应样本；实际轮次、方向和交错顺序由恢复后的原始 schema 决定。

| 一级来源 | 本地文本 | 上游任务家族/条件 | 处理路线 |
|---|---|---|---|
| BLIP3o-Pretrain-JourneyDB | 预算内未定位 | caption_pair、T2I_prompt、I2T_caption | 整源拒绝；不做训练 recaption |
| BLIP3o-Pretrain-Long-Caption | 抽样已见 | caption_pair | 复用已配对标注；按实际任务修订 |
| BLIP3o-Pretrain-Short-Caption | 抽样已见 | caption_pair | 复用已配对标注；按实际任务修订 |
| CC12M_and_Imagenet21K_Recap | 预算内未定位 | caption_pair、classification_if_original_class_retained | 先恢复标注；缺标注只能另建新 caption |
| DenseFusion-1M | 预算内未定位 | I2T_dense_caption、caption_pair | 先恢复标注；缺标注只能另建新 caption |
| DenseFusion-4V-100K | 预算内未定位 | I2T_dense_caption、caption_pair | 先恢复标注；缺标注只能另建新 caption |
| FineVision | 预算内未定位 | I2T_QA、I2T_caption、multi_image_understanding、multiturn_I2T、OCR、grounding、text_only | 先恢复标注；缺标注只能另建新 caption |
| Infinity-MM | 抽样已见 | I2T_caption、I2T_QA、multiturn_I2T、OCR、document_understanding、text_only | 复用已配对标注；按实际任务修订 |
| JourneyDB-train | 预算内未定位 | caption_pair、T2I_prompt、I2T_caption、I2T_QA | 整源拒绝；不做训练 recaption |
| LLaVA-OneVision-1.5-Instruct-Data-Images | 预算内未定位 | I2T_QA、I2T_caption、multiturn_I2T、multi_image_understanding、OCR、grounding、text_only_in_parent_an… | 先恢复标注；缺标注只能另建新 caption |
| LLaVA-OneVision-1.5-Mid-Training-85M | 预算内未定位 | caption_pair、interleaved_only_if_original_document_and_order_recovered | 先恢复标注；缺标注只能另建新 caption |
| LLaVA-OneVision-Mid-Data | 抽样已见 | I2T_caption、I2T_QA、OCR、document_understanding、text_only | 复用已配对标注；按实际任务修订 |
| Salmonnn | 预算内未定位 | caption_pair、multi_image_caption_if_original_annotations_recovered、classification_if_original_l… | 先恢复标注；缺标注只能另建新 caption |
| ScaleCap450k_image | 预算内未定位 | I2T_dense_caption、caption_pair | 先恢复标注；缺标注只能另建新 caption |
| Stable_Diffusion_3_Recaption | 预算内未定位 | caption_pair、classification_if_original_class_retained | 先恢复标注；缺标注只能另建新 caption |
| UniWorld-V1-new | 抽样已见 | T2I、image_editing、multiturn_image_editing、conditional_generation、structured_prediction | 逐图角色与编辑血缘；保留原编辑/控制任务 |
| laion-aesthetics-12m | 抽样已见 | caption_pair | 复用已配对标注；按实际任务修订 |
| llava_ov_data_new | 抽样已见 | I2T_QA、multiturn_I2T、OCR、document_understanding、caption_if_original_task_is_caption、grounding_i… | 复用已配对标注；按实际任务修订 |
| mjhq30k_imgs | 预算内未定位 | T2I_evaluation | 整源拒绝；不做训练 recaption |
| openuni | 抽样已见 | T2I_caption_pair | 六子集分别配对；排 Flux/DALL·E 分支 |
| pixmo-ask-model-anything | 预算内未定位 | single_turn_I2T_QA | 先恢复标注；缺标注只能另建新 caption |

## 已准备好的关键例外

1. **三个整源拒绝**：`BLIP3o-Pretrain-JourneyDB`、`JourneyDB-train`、`mjhq30k_imgs`。JourneyDB 两个版本来自 Midjourney 生成图；MJHQ 同时带评测隔离原因。这些来源不用再逐记录询问 API 是否 AI 图，也不用为训练付费重 caption。
2. **Stable_Diffusion_3_Recaption 不按名称拒绝**：发布者使用 ImageNet2012 与 CC12M 原图/重描述。当前本地 caption 尚未定位，不能直接启用上游 schema。ImageNet 部分上游 ID 恒为 `999999999`，不能用它当唯一图像键。需按源路径、原始图像 ID 的适用范围及内容哈希配对。
3. **UniWorld 逐目标处理**：`data/BLIP3o-60k/` 是明确生成图分支；其 Human/COCO 提示来源不是目标真实来源证明。其余编辑和控制任务不能整源拒绝。必须保留 source、target、control 的角色，并证实具体 target 的自然父图血缘；仅目录名为 OpenImages、编辑前图看起来像照片或生成器说“编辑”，都不足以建立该关系。任意必要图像不可用时隔离整条任务，不静默删段。
4. **StyleBooth 是专门配对合同**：已有 `Source:FILE`、`Target:FILE` 与风格字段；样本路径以 `stylebooth_dataset` 为根。末列 `SearchResult:DICT` 内逗号未正确转义，历史验证按前五个逗号拆分、再解析 JSON；正式运行仍需验证完整 schema 和错误行。样本 3 条、6 张图片存在不等于全量通过。
5. **OpenUni 六路独立准备**：`cc12m`、`laion6m`、`megalith-10m`、`redcaps5m`、`redcaps5m_resized`、`text-to-image-2M` 均在 `subcontracts` 中列出。`image/annotation` 是索引路径而非描述文本；图片和 caption TAR 不能按记录行号直接 zip。必须处理 TAR 成员定位、解包副本、分片命名空间和孤立配对。
6. **Megalith 六 caption 与 URL 陷阱**：已观察 `caption_internlm2{,_short}`、`caption_florence2{,_short}`、`caption_sharecap{,_short}`。它们是同一图像的六种文本视图，不是六张独立图片。原 metadata Parquet 无图像 bytes，且 generic `caption`/TXT 可是 Flickr 页面 URL。可从独立 `captions/<shard>/<key>.json` 选择真实描述；`caption_internlm2_short` 只是依据 OpenUni 用法提出的候选默认值，质量验证前不自动激活。
7. **OpenUni T2I-2M 分支拒绝**：现行规则只对已知 Flux、DALL·E 来源路径生效，并引用具体规则 ID。没有这些关键词不等于真实图；混合分支仍需恢复组件身份。自然图片组件也需逐图准入，不从 512/1024 尺寸或提示来源推断。
8. **Infinity 精确前缀映射**：只替换 `/inspire/hdd/project/deepgen/public/data_zoo/source_images/Infinity-MM` 到 `{collection_root}/Infinity-MM`，并要求路径边界，保留 stage/subset/shard/member 后缀。不能替换更宽泛旧根来影响其他集合。历史 9 个 stat 命中不注入运行时 `exists=true`。
9. **两个 LLaVA 路径基准不同**：`LLaVA-OneVision-Mid-Data` 的 image 相对 source root；`llava_ov_data_new` 的 image 已含该一级目录名，相对 collection root。后者不能二次加前缀。聚合 JSON 与组成 JSON 的重复记录须去重。
10. **图片来源不恢复序列**：85M 集中的 `recs_obelics_*` 或 MINT/OBELICS 名称不能证明原网页顺序尚在。网页截图问答也不是文本/图片交错文档。只有原 doc ID、segment/order 与图像引用恢复后才允许标为交错序列。

## 复用 caption 与去重单位

可以优先复用的本地文本包括 BLIP3o Long/Short 的同 stem TXT、LAION 的 TXT/JSON.caption、OpenUni 已配对的独立描述，以及确属 caption 的对话子集。现有文本仍需图像对应、语言、事实性与长度质量检查；不能因“很长”就认定质量高。

只有图片的 FineVision、LLaVA 图片包、PixMo 等，应先寻找对应原任务标注；找不到时，可以在图像通过准入后另建 grounded caption，但不能用 caption 代替原问题答案、多轮、定位框或网页正文。新文本统一保留 `caption_origin=ai`、生成器版本和原图像 ID。

合同列出需优先检查的同源族：CC12M、ImageNet、LAION/OpenUni、DenseFusion 1M/4V、SA-1B、COCO 多控制任务、LLaVA-OneVision 与 Infinity stage3、Docmatix，以及 RedCaps raw/resized。`actual_content_overlap_verified=false` 明确表示尚未逐图匹配。已知共享来源不等于每个子集都有交集，也不等于可以删去所有相同来源记录。

资产去重与监督去重分开：相同图像可以共享存储和一次视觉计算，多个真实不同 QA、不同任务或长中短 caption 仍可作为合法视图。同一图像的六 caption 不能计成六张唯一图片；相同图像的三个独立 QA 也不能被强行当成原生三轮对话。

## 哪些无需逐记录调用 API

已知来源/路径规则、文件载体、尺寸、精确哈希、显式 role/turn/image 数量、原始 segment 顺序、明确 key 配对、已知 prefix 映射，都由代码确定。API 只处理规则和 schema 无法确定的语义、内容质量及图像来源疑难，保留 unknown 与人工队列。合同不根据单个目录名幻想未知字段，也不把模型判断当原始来源证据。

正式运行仍必须验证挂载可见性、每个分片 schema、图像解码、孤立配对比例、source/target 血缘、split、去重候选、序列守恒和生成文本质量。抽样结构知识的作用是减少重复识别工作，不是跳过这些检查。

## 证据与交付检查

- [原始 gen-und 报告](../../GENUND_DATASET_REPORT.md)
- [2026-10-21 汇总](../../reports/genund_20261021/dataset_summary.json)
- [2026-10-22 字段复查](../../reports/unified_mm_20261022/deep_metadata_verified.json)
- [对话/StyleBooth 语义检查](../../reports/genund_20261021/annotation_semantics.json)
- [当前来源规则](../config/sources.json)
- [当前 adapter 及待启用字段覆盖](../config/adapters.json)
- [Megalith 配对证据](../../reports/genund_20261021/megalith_caption_pairs.json)
- [TAR 配对证据](../../reports/genund_20261021/tar_pairs.json)

交付检查：JSON 可解析；21 个父源唯一且与现有来源表完全一致；3 个整源拒绝；8 个父源只有“抽样文本存在”结论；Stable_Diffusion_3_Recaption 非整源拒绝；OpenUni 6 个子合同及 Megalith 6 caption 字段；所有覆盖率/实际重叠与训练就绪状态均未夸大为已验证。此文件是静态准备合同，不声明已经实现或运行全量转换。
