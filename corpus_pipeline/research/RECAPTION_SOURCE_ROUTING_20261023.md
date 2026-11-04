# DCVLM 与 gen-und：早期复用、补缺及 1MP 阶段 caption 覆盖

本次仅审计已有准备合同、来源规则摘要与 2,396 张历史样本的元数据；未读取全库、重新挂载 gen-und、调用模型或修改生产配置。日期按 Asia/Shanghai 为 2026-10-23。

建议将“需要补充”定义为**某个资产或任务缺少当前阶段需要的、正确配对且经过验收的文本视图**，而不是“目录名字不像高质量数据”或“caption 不够长”。早期阶段复用合格原文，优先补没有可用监督的记录，并处理明确错误；在约 1MP（例如 1024×1024 像素面积）阶段，对最终入选的 caption 图像池完成所需描述视图的覆盖。已有达标描述可以贡献覆盖，不必仅为了换生成模型重新写一遍。

这是一项分阶段工程建议，不是 MOSS 已验证过的训练阶段对照结论。MOSS 的每图三档生成配方、原文/生成描述训练采样、精选 layout 子集是不同概念；不能由 20:80 采样率推导只需给 80% 图片生成描述。[MOSS C.1](https://arxiv.org/html/2610.23374v1#A3.SS1)

## 路由的具体含义

所有 caption 路由都必须先经过既有图像准入：原始短边至少 128、实际资产与引用完整、来源与 AI 图规则、去重和任务解析。来源状态 `unknown/mixed` 的处理仍遵守现行规则；复用文本不授予图像准入。AI caption 只标注，AI 生成图像仍排除；有可验证自然父图血缘的 AI 编辑图可保留并标注，非 AI 程序图表和渲染图不因“不像照片”而排除。

| 路由 | 触发条件 | 早期动作 | 不能据此声称 |
|---|---|---|---|
| `reuse` | 图文匹配成立，原任务明确，已有文本满足该任务和当前阶段要求 | 保留原文、原任务和来源；分层抽检后按已通过的来源/记录规则复用 | 目录内所有图文都正确，或文本都为人工标注 |
| `recover_annotation` | 图片有了，但本地标注未找到、字段未核实或配对未恢复 | 优先定位已有标注、建立 source/shard/key 命名空间和孤立配对统计 | “原数据没有 caption”，或可按文件顺序配对 |
| `repair` | 已证实缺文本、仅 URL/路径、格式损坏，或已核实的内容错误 | 无损格式错误先用代码；缺描述或内容修订才对准入图片生成新视图 | 模型生成的 caption 恢复了原问答、多轮或网页正文 |
| `review` | 语义配对不明、过度具体、可疑 OCR/数量/关系、来源混合或抽检异常 | 对问题样本或异常分层做有针对性的看图复核；保留 uncertain/人工队列 | schema 合格等于事实正确；“很长”就等于高质量 |
| `late_coverage` | 已纳入最终高分辨率 caption 池，但缺少该阶段规定的合格描述视图 | 按唯一资产/有效视图版本补齐，逐项验收；保留原文，不覆盖原任务 | 需要把所有 QA、纯文本及交错文档改成 caption |
| `reject` | 命中已证实 AI 生成图来源/路径规则 | 排除训练记录；不为这些图付费 recaption | 把原始生成 prompt 当成真实图像来源证明 |

只有“正确但简短”的 caption 不应仅因长度不足进入早期 repair；它可以是合格短视图，长视图留到后期补。反过来，URL、文件名或无关网页模板不是可复用 caption。图文事实检查与格式检查分别记录，不能借“复用”绕过内容验收。

## DCVLM 的来源建议

下表字段由原始父版 `dcvlm_pretrain_v1_20261016T062238Z` 的既有校准样本核对。样本记录中的 `parquet_path`、`segments`、`source_member` 是证据；没有据此声称全量覆盖。`prepared_profiles.json` 的历史统计引用已经退役的 T2I50 快照，明确设置 `retired_t2i50_snapshot=true`、`caption_direction_usable=false`，本报告不使用其旧方向分布。

| DCVLM 来源 | 已见本地文本/原生任务 | 早期路线 | 1MP 阶段如何补 | 重要边界 |
|---|---|---|---|---|
| `datacomp_1b` | 同图 `.txt` 已归一到 `segments[].text`；`format=caption`，图像段在文本段前 | 原网页描述通过配对和质量检查后 reuse；URL-only/空白/模板垃圾 repair；商品属性、名称等不可见细节 review | 入选 caption 资产缺少细节视图时补 grounded 描述；原文仍保留 | 来源图像为 mixed，文本来源先验 unknown；不能整源自动通过，也不能假定已有 recaption |
| ReLAION 三源：`relaion-2b-en-vol-1-unpacked-TEST`、`...vol-2...`、`...research-addon...` | `.txt`→`segments[].text`，已有短网页标题/商品描述样本；`format=caption` | 与 DataComp 相同，先复用可用原文；明显非视觉内容进入补缺/复核 | 对后期实际选入池补齐缺少的中长细节，而非重复生成所有已有合格短句 | 名称中的 `research/TEST` 不证明本地训练/评测用途或 caption 质量；需已有 split/来源审计 |
| `grit_caption` | 同 stem `.txt` caption 已见 | reuse 可用句子；图像无从支持的功能/推断性语句 review | 缺主体、动作、关系、可读文字的 caption 资产按需补 | 本地读到 caption 不等于保留了 GRIT 全部 grounding 框；不能因源名生成虚构原框 |
| `pixmo_cap` | `.conversations.txt` 映射到 user 指令＋assistant 描述；`format=conversation` | 从原任务中读取 assistant caption 并保留指令；已有细致描述优先 reuse；身份/地点等不可见细节重点 review | 已满足 late 描述契约的文本可计入覆盖，其余补缺少视图 | 不能仅用 `format=caption` 搜索 caption 任务；来源标记 mixed，不能把整个 PixMo 家族当同一任务 |
| `coco_caption` | conversation 的 assistant 简短描述，原生单轮 I2T caption | 以短句原任务为主 reuse；不为追求长度替换良好短监督 | 需要时另加详细 I2T caption 视图；保留原短句/用户请求 | “要求简短”任务不应塞入长回答；COCO 不等于该镜像每条文本都是人工真值 |
| `openimages_caption` | conversation 的 assistant caption | 与 COCO 相同；来源证据与内容质检分别做 | 补合格细节视图并保存原始监督 | 当前来源真实性与文本来源先验均 unknown，不能由名字直接升级 |
| `internvl_sa1b_caption` | `.txt`→`segments[].text`，既有 AI caption | 标记 `caption_origin=ai` 后优先复用；复杂 OCR、身份和解释性细节重点 review | 达标内容复用；缺失内容再看图补充 | “AI caption”不触发图像排除；SA-1B 与其他集合的重叠需资产验证 |
| 原生 QA、OCR、数学、多轮及交错来源 | 显式 role/turn 与 ordered segments | 保留任务；有证据的问题答案才修订。纯文本跳过视觉 recaption | 可追加资产级 caption sidecar，但不替换题目答案、轮次或正文 | 同一张图多个 QA 不能合并成虚构多轮；交错顺序不能由图片文件夹恢复 |

DCVLM 的方向保护仍有效：**不再从 DCVLM 构建 T2I50 变体**。caption 增补属于保留原生 I2T 语义的回答视图/sidecar。T2I 从 gen-und 中通过配对与任务合同的 caption/生成监督构建，不能把所有 gen-und QA 自动反转。

## gen-und 的来源建议

### 本地已抽样见到可复用文本

| 来源/子源 | 已确认的字段与配对线索 | 早期 reuse/repair/review | 后期覆盖 |
|---|---|---|---|
| `BLIP3o-Pretrain-Long-Caption` | 同目录精确 stem 的 JPG/TXT；`TXT_contents`；合同记录已有 AI caption | 先验证逐条 stem 配对和来源，复用现有长描述；问题内容 review；只补缺失或不合格项 | 已通过的长描述可以贡献覆盖；必要时导出保守短/中视图，仍需检查不能添加新事实 |
| `BLIP3o-Pretrain-Short-Caption` | 同目录精确 stem 的 JPG/TXT；`TXT_contents` | 简短且忠实可直接用于早期；长度短不是错误 | 入选后期池再补中/长信息；不能假定 Long/Short 两个版本是同一批图片，须内容哈希或可靠 ID 对齐 |
| `laion-aesthetics-12m` | 已见 `TXT_contents`、`JSON.caption`；同 shard/key；JSON 有 status、width/height、sha256 等 | 复用可用原文，URL/模板及错配进入 repair/review；metadata.sha256 须与 bytes 核验 | 补详细描述并保留原文；aesthetics 名称不是文本质量保证 |
| `openuni/data/laion6m` | 本地见 `caption` 与 `detailed_caption`；清单 `image/annotation` 是路径，需连到 raw/captions TAR 或提取文件 | 已有短/详细文本分别作为候选视图，优先复用；先验证同 key 和文本事实性，不重新全写 | 补缺少或未达标的视图即可；上游文档中的 `re_caption` 尚未在该本地分支确认，不能覆盖实际 schema |
| `openuni/data/megalith-10m` | 独立 `captions/<shard>/<key>.json` 六字段：`caption_internlm2`、`caption_internlm2_short`、`caption_florence2`、`caption_florence2_short`、`caption_sharecap`、`caption_sharecap_short` | 保留六视图来源，先做分层比较和问题样本 review，再选可复用视图；不因已有多模型文字就全量再生成 | 六视图中有满足要求的候选可贡献覆盖；缺少可靠细节或字段不合格才补 |
| `Infinity-MM`、`LLaVA-OneVision-Mid-Data`、`llava_ov_data_new` 中确属 caption 的任务 | 已抽样见对话文本；用显式角色与真实任务读取，不按父目录统一归类 | caption 子任务可复用；QA/OCR/多轮保持本来语义 | 仅补 caption 缺口或添加资产 sidecar；不把整合集改成 T2I |

Megalith 的关键陷阱：原 metadata Parquet 没有图像 bytes；generic `caption` 或 TXT 有时只是 Flickr 页面 URL，不能作为真实描述的兜底。六字段是**同一图像的六种文本视图**，不是六张独立图；`caption_internlm2_short` 只是原合同根据 OpenUni 用法提出的候选默认值，尚未经过我们质量比较，不能自动当最佳。字段名能追溯视图来源，不等于已知完整模型版本或事实正确。

### 有图片，但已有 recaption/原任务标注尚未在本地恢复

| 来源/子源 | 字段现状与待恢复内容 | 早期动作 | 后期动作 |
|---|---|---|---|
| `CC12M_and_Imagenet21K_Recap` | 本地 `local_fields=null`；上游候选 `recaption`、`recaption_short`、`class` | 优先找相同 revision 的标注并恢复 image/id 配对；class 标签不是完整视觉描述 | 复用恢复出的合格长短描述；找不到标注但图像准入后，可另建新 caption，并标记非原始标注 |
| `Stable_Diffusion_3_Recaption` | 本地未定位 caption；上游候选 `caption`；合同说明涉及 ImageNet/CC12M 原图 | 不按名称当 AI 生成图整源拒绝；先恢复标注与原图身份 | 同上；ImageNet 分支上游 ID 可能恒为 `999999999`，不能当唯一键 |
| `DenseFusion-1M`、`DenseFusion-4V-100K` | 本地未定位标注；上游候选 `caption`，待找匹配 image_id/path/url 的 JSONL | 优先恢复已有 dense caption，再做 grounding review | 已达标的细节描述不必重写；缺少视图按需补；不得从 `4V` 名称推断本地记录必有四张图 |
| `ScaleCap450k_image` | 本地未定位 `ScaleCap450k.json`；上游 `images[]`、`conversations[].value`/`from` | 优先恢复超详细描述及多图关系，再验实际配对/事实性 | 复用达标内容；按明确单图/多图任务补齐视图，不拆散原多图记录 |
| `Salmonnn/InternVL-SA-1B-Caption` | 本地标注 pending；上游 `id/image/conversations`，image 可能是列表 | 找 EN/CN/多图原标注，与 DCVLM 同源资产匹配后复用 | 同资产可复用已验 sidecar；不同语言/多图条件保留分别的任务视图 |
| `Salmonnn/imagenet-21k` | 未恢复自然语言标注；类别目录不是 caption | 恢复 class 监督；没有 caption 不代表没有训练价值 | 需要 caption 时在保留分类任务的同时新增，不伪造“原始长描述” |
| `Salmonnn/sharecaption`、`Salmonnn/densefusion-1m` | 上游身份/具体 revision 对应仍未解决 | 先识别来源；不把 `sharecaption` 自动映射为 ShareGPT4V | 身份和配对确认后再按既有 caption 或新建视图处理 |
| `openuni/data/cc12m` | 已见 `image/annotation` 索引；实际 caption 字段尚未确认 | 恢复引用指向的 captions TAR 与 key，不能假定字段叫 caption | 达标既有描述优先，否则补；先与其他 CC12M 分支按资产去重 |
| `openuni/data/redcaps5m`、`redcaps5m_resized` | 本地 caption 字段未确认；上游候选 `re_caption`；观察到 `.aria2` 状态文件，需查完整性 | 恢复原/缩放图关系和标注；两目录不应按名字累计成独立 10M | 按内容与变换关系共享描述；裁切/文字变化不得直接继承原描述 |
| `FineVision`、LLaVA-OneVision 图片包与 85M、`pixmo-ask-model-anything` | 多数本地原标注未定位；PixMo AMA 上游是 question/answer，不是单纯 caption | 优先找原任务标注；image-only 不能进入不存在的原 QA 监督 | 仅在明确需要时另建 caption；不重造原问题、轮次、定位框或网页次序 |

这里的“未定位”只是历史审计预算内没找到，不是证明文件不存在。恢复已有标注可以比调用大模型更便宜，但只有严格配对后的文本才能计入可用覆盖。

### 明确排除和必须逐组件处理的来源

| 来源 | 路由 |
|---|---|
| `BLIP3o-Pretrain-JourneyDB`、`JourneyDB-train`、`mjhq30k_imgs` | 合同已确认为 AI 生成图的整源拒绝；不做训练 recaption。MJHQ 另有评测用途隔离。 |
| `openuni/data/text-to-image-2M` | 本地可见 `prompt` 不代表可保留图像。按当前 Flux-dev / DALL-E_3 路径规则排除；可能存在的自然图组件必须恢复逐记录来源，不能把不含关键词的路径自动判为真实。 |
| `UniWorld-V1-new` | 保留 source/target/control 角色；排明确生成图分支。真实父图的 AI 编辑目标只有具体血缘证据成立才保留；caption 不应取代编辑指令和目标关系。 |

## “后期完整 recaption”应如何验收

先冻结 `late_caption_pool` 的成员，记录阶段、版本及资产身份：只统计通过图像准入、已正确配对且实际会在该阶段训练的 caption 资产。1MP 是训练目标分辨率/面积，不能由升采样后的尺寸倒推原图具备细节，也不能据原图短边 128 自动承诺高分辨率文本质量。

建议同时报告三个独立分母：

1. **资产覆盖率**：该池中具有所需合格描述视图的唯一图像资产数 / 池内唯一图像资产数。
2. **监督记录覆盖率**：引用这些视图且保留原任务/方向的就绪 caption 记录数 / 池内 caption 记录数。同图不同问题、语言或条件不能仅凭资产 coverage 判全记录完成。
3. **采样权重覆盖率**：后期训练采样权重落在就绪记录上的比例。稀有来源遗漏不能被大量网页照片掩盖，需同时按来源、领域、任务统计。

`late_ready` 必须绑定原图哈希/几何变换、caption 字段和文本哈希、来源或生成模型版本、验证版本及结论。短/中/长缺项、不确定事实未解决、原图已更换、裁切变化、配对不明均不计入对应的完整三档覆盖。已有人工短句可以贡献 short；已有高质量 detailed caption 可以贡献 long；medium 的压缩也不能引入原图不支持的新断言。三档不是固定字数配额。

若后期要求的是同一新模型统一生成，应另外统计 `fresh_generation_coverage`；它与 `usable_caption_coverage` 是两件事。目前建议以**达到契约的内容覆盖**为目标，避免把已有合格 recaption 也无条件重写。进入训练池需满足冻结的 late gate；尚未完成的记录保留 pending，不为了覆盖率把 uncertain 改成通过。也可以先用明确冻结的就绪子集开始该阶段，但必须如实报告其相对候选池的覆盖，不能称为全候选池完成。

资产级去重先于昂贵补写：DCVLM 与 gen-und 共享的 CC12M、LAION、SA-1B、COCO、DenseFusion 等图像先通过原图 hash/可靠 ID 或验证后的近重复映射对齐；一次视觉生成可被多个允许的任务视图引用。**同源名称并不证明实际内容重叠**，不同裁切或编辑目标也不能盲目共用描述。已有标注与已验新视图放在旁路版本中，原文不删除。

## 当前证据边界与文件回执

- gen-und 合同覆盖 21 个一级来源，8 个来源抽样发现文本，3 个整源拒绝；21 个来源的 `full_coverage` 都为 null。不能据此给出现在需要 recaption 的全量条数或比例。
- OpenUni 的 caption 字段、Megalith 六字段及 BLIP3o/LAION 同 stem 文本是有界本地审计事实；全量配对率、质量通过率、来源真实性、与 DCVLM 的实际重叠率尚未量化。
- DCVLM 表格通过既有 2,396 图样本核对当前父版的真实 `segments` 和任务载体，没有扫描 119M 行。它证明相关字段在样本中存在，不证明任何一个来源全部达到早期/后期要求。
- 当前 prepared 合同仍为准备状态，`training_ready=false`/`automatic_accept=false`。本文是阶段路由建议，没有启用新的生产规则。

本次读取的主要输入及 SHA256：

| 文件 | SHA256 |
|---|---|
| [genund_source_contracts.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/prepared/config/genund_source_contracts.json) | `8a5d3f9c0289c46e5f51b155f68406cbe916d4fe392b6a8d6cb14be63411d39c` |
| [prepared_profiles.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/prepared/config/prepared_profiles.json) | `0d906a242ae82489161e1419c6d6331b56eb730486ab60eced3e758a4c9e32b5` |
| [GENUND_PREPARED_ROUTING_20261022.md](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/research/GENUND_PREPARED_ROUTING_20261022.md) | `c6ad63d6453d4a6f7f436a1ea6ccb0fd19a96fc61a91c3aef40699b8f7e0dcf9` |
| [samples.jsonl](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/runs/detector_calibration_20261022/samples.jsonl) | `1c2fc90cef24496b8b4a9c59d83a2dd9e00df6fafdde76317c7bdcdb1a7dc8b7` |

详细原始配对证据由父合同引用，包括 `reports/genund_20261021/megalith_caption_pairs.json`、`tar_pairs.json`、`reports/unified_mm_20261022/deep_metadata_verified.json`。本轮没有把这些历史路径存在性升级为未来 Ascend 挂载可用性证明。
