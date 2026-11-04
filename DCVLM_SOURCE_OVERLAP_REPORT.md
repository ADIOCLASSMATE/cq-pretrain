# DCVLM 来源、gen-und 同源关系与 T2I 排 AI 图规模估算

> 2026-10-22 更新：本文中 T2I50 的实测计数是历史审计结果。用户已退役该派生版；当前 DCVLM 使用原版 v1 并保留原生类型，T2I 仅从 gen-und 构建。最新入口与验证见 `corpus_pipeline/README.md`、`corpus_pipeline/VALIDATION.md`。

核验日期：2026-10-22（北京时间）。本报告回答：交错来自哪里；图片描述与单轮／多轮视觉对话来自哪里；哪些与 gen-und 共享上游；排除明确 AI 图来源会减少多少 T2I 候选。

**主要结论：DCVLM 的交错集中在 OmniCorpus-CC 和 MINT-HTML；独立 caption 集中在 ReLAION 和 DataComp。gen-und 与它共享多种原始图像／指令来源，但“共享来源”并不等于已经测出相同图片或相同图文记录的重复率。T2I 删除比例必须绑定候选来源清单；当前没有一个经过完整配对、去重和 AI 判定的 gen-und T2I 总数。**

## 1. 核验对象与统计口径

使用当前实际快照 `dcvlm_pretrain_t2i50_20261017T093600Z`：**119,010,208 条文档、12,205 个 Parquet、165 个正式来源**。从全部 12,205 份发布分片的 `done.json` 汇总来源×格式，逐份与 catalog 的身份、行数、bytes、SHA256 对齐，并与全局及逐源发布统计对平。这里的“精确”指**既有发布构建统计的精确汇总**，本轮没有重新扫描 119M 行或重新解码图像。

原始 `dcvlm_pool_small` 是未混合的候选池，官方论文／大池的来源数与规模不能替代当前 small 子集及本地处理结果。官方原始卡记录 166 源，当前快照 165 源；论文按上游家族归纳为 160 源，是不同口径。[原始 small 数据卡](https://huggingface.co/datasets/mlfoundations/dcvlm_pool_small)、[官方来源分类](https://arxiv.org/html/2606.28551v3#A5)

| 本地 format | 精确文档数 | 含该格式的物理来源数 | 含义 |
|---|---:|---:|---|
| caption | 88,908,841 | 8 | 独立图文 pair；当前构建分成 T2I/I2T 两方向 |
| conversation | 3,507,428 | 115 | 有图像的描述、QA、OCR、定位、代码等指令；包含单轮及多轮 |
| interleaved | 4,497,702 | 9 | 5 个上游家族的有序图文文档 |
| text_conversation | 22,096,237 | 36 | 纯文本对话，不能算视觉 caption／视觉多轮 |

来源数不能直接相加为 165：有 3 个混合来源同时包含视觉与纯文本。`allava` 的视觉／纯文分别为 19,353／109,921；`tqa` 为 135／2,918；`visualwebinstruct` 为 12,741／22,300。后文对视觉来源的计数使用 conversation 列，而非 source 总行数。

本轮共查 4,321 条诊断记录，覆盖 286 个分片、719 个 row group；115 个视觉对话来源均有样本，其中视觉对话 2,663 条。单／多轮通过 Parquet 的角色与 turn 字段作分层诊断抽样：一轮指一组 user 问与 assistant 答；20 条交替消息是 10 轮。每源最多取首／中／尾分片与 row group 的部分记录，**不是均匀随机样本，不把抽样轮数比例外推成全源精确比例**。已知来源总数、抽样轮数和证据强度分别列示。

精确清单：[dcvlm_sources.csv](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/dcvlm_sources_20261022/dcvlm_sources.csv)；完整证据：[dcvlm_sources.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/dcvlm_sources_20261022/dcvlm_sources.json)。

## 2. 交错序列：9 个物理来源，5 个上游家族

| 上游家族 | 本地来源 | 文档数 | 占交错文档 | 图像出现次数 | 内容 |
|---|---|---:|---:|---:|---|
| [OmniCorpus-CC](https://huggingface.co/datasets/OpenGVLab/OmniCorpus-CC) | omnicc_matteo、omnicc_mehdi、omnicc_vishaal | 2,428,596 | 54.00% | 7,898,967 | 网页原文中的图文序列 |
| [MINT-1T-HTML](https://huggingface.co/datasets/mlfoundations/MINT-1T-HTML) | mint_html_matteo、mint_html_mehdi、mint_html_vishaal | 1,943,913 | 43.22% | 5,947,225 | Common Crawl HTML 中的交错图文 |
| [MINT-1T-PDF](https://huggingface.co/collections/mlfoundations/mint-1t) | mint_pdf | 81,707 | 1.82% | 406,410 | PDF提取的文本与插图 |
| [WanJuan1.0](https://github.com/opendatalab/WanJuan1.0) | wanjuan_matteo | 24,797 | 0.55% | 371,419 | 公共网页的图文文档，新闻人物社会生活等 |
| [Multimodal Textbook](https://github.com/DAMO-NLP-SG/multimodal_textbook) | multimodal_textbook | 18,689 | 0.42% | 199,445 | 教学视频关键帧与ASR/OCR整理的知识序列 |

总计 **4,497,702 篇文档、14,823,466 次图像出现**。其中 OmniCorpus-CC＋MINT-HTML 为 **4,372,509 篇，占 97.22%**；MINT-PDF、WanJuan、Multimodal Textbook 合计约 2.78%。这说明当前自然交错的覆盖主要由通用网页决定，不能仅凭“有交错”推断教材／PDF／中文文档已得到均衡训练。

`matteo/mehdi/vishaal` 三类后缀在本地目录层面分开，应按对应同一上游家族聚合；它们不是新的语义数据类型。也不能据后缀断言三个目录图片互不重复。

几个内容上的区别值得保留：

- **OmniCorpus-CC、MINT-HTML**：网页正文与图片位置的共同上下文，有文章、说明、产品等网页内容；具体域比例本次没有全量测量。
- **MINT-PDF**：PDF 提取文本与插图，不能一概称为整页截图、带 bbox 的版面真值，或全部都是学术论文。本地各 PDF 子来源覆盖未进一步拆分。
- **WanJuan**：上游的图文部分来自公共网页，涵盖新闻、人物、自然与社会生活。原始公开版以 URL 引用图像；当前 DCVLM 已有下载图片索引，不能将上游分发形式与当前落盘状态混淆。[原作者说明](https://github.com/opendatalab/WanJuan1.0)
- **Multimodal Textbook**：原作者说明是教学视频关键帧与音频转写／OCR 形成的知识序列，**不是简单的 PDF 教材集合**。它对步骤连续性和课程语义有价值，但本地仅 18,689 篇。[原始仓库](https://github.com/DAMO-NLP-SG/multimodal_textbook)

上一轮已经核对 OmniCC/MINT 的原 TAR 与 Parquet：9 篇文档的 105 个文本成员逐字节一致，并保留成员位置顺序。本轮进一步确认全部来源×格式计数。它们是实际共同文档序列，不是 packing 时把无关 pair 拼接出来的结果；但未据此证明所有原始网页 DOM、全部文档身份和几何坐标均保留。

**与 gen-und 的关系：** LLaVA-OneVision-1.5-Mid-Training-85M 上游包含 MINT、OBELICS 单图来源，可能复用交错文档中的图像。原图来源相交，不代表 gen-und 当前的平铺图片／重描述能替代 DCVLM 的原序列。OBELICS 也不是这份 DCVLM catalog 中一个独立同名交错来源，不能把两个名字直接合并为同一数据集。

## 3. Caption 和单轮／多轮视觉对话的来源

你提到的“caption 类型（单轮、多轮）”在实际文件中跨越两种格式：简单图文 pair 记为 caption；“看图描述”也可能包装成 user/assistant，记为 conversation。下面同时按**存储格式**和**任务用途**列出，避免只筛 format=caption 后漏掉描述数据。

### 3.1 独立 caption：8 个物理来源

| 本地来源 | caption 数 | 占 caption |
|---|---:|---:|
| datacomp_1b | 42,175,337 | 47.437% |
| relaion-2b-en-vol-1-unpacked-TEST | 29,744,774 | 33.455% |
| relaion-2b-en-vol-2-unpacked-TEST | 14,482,382 | 16.289% |
| relaion-2b-en-research-addon-unpacked-TEST | 1,247,291 | 1.403% |
| grit_caption | 449,436 | 0.506% |
| internvl_sa1b_caption | 372,030 | 0.418% |
| facecaption15m | 348,849 | 0.392% |
| as100m | 88,742 | 0.100% |

ReLAION 三个目录合计 **45,474,447**；DataComp 为 **42,175,337**。两者合计 **87,649,784，占该格式的 98.58%**。其余 4 个源合计 1,259,057：GRIT、InternVL-SA1B、FaceCaption、AS-100M。

这些是图文 pair，包装后可形成单轮描述／生成任务；它们不是原生多轮对话。当前 T2I50 在全部 caption 中分配 T2I=44,454,420、I2T=44,454,421。不要据 source 名称或“一半方向”推断文本都是长描述、图像全部真实，或每图同时具有双方向视图。

### 3.2 官方归为 Captioning、本地存为 conversation 的 7 个来源

官方将以下来源归为 captioning；在当前本地格式中，它们落入 conversation，合计 **95,545** 条视觉记录。这是来源级归类，不是逐行检验后确认全是纯 caption 或全部可直接反转为 T2I。

| 语义上属于描述／caption 的来源 | 本地 conversation 数 | 抽样：单轮／多轮 |
|---|---:|---|
| pixmo_cap | 17,940 | 36 / 0 |
| sharegpt4o | 1,663 | 12 / 0 |
| textocr_gpt4v | 702 | 6 / 0 |
| textcaps | 3,387 | 18 / 0 |
| coco_caption | 15,977 | 36 / 0 |
| openimages_caption | 15,807 | 36 / 0 |
| sea-vl | 40,069 | 42 / 0 |

这些来源本次样本均为单轮。表中的 0 表示未在这些样本观察到多轮，不能作为全源没有多轮的证明。COCO Caption、PixMo-Cap、TextCaps 等属于你要找的图片描述来源，不能因 format=conversation 而遗漏。[官方类别依据](https://arxiv.org/html/2606.28551v3#A5.SS1)

### 3.3 Caption & Knowledge 指令来源

官方另将以下来源归为描述／知识类视觉指令，本地合计 **191,573** 条。它们并非每条都是可直接反转为 T2I 的 caption，也可能是知识问答或多轮指令。

| Caption & Knowledge 指令来源 | 本地视觉记录数 | 抽样：单轮／多轮 | 抽到的问答轮数 |
|---|---:|---|---|
| art500k | 14,642 | 36 / 0 | 1–1 |
| llava_595k | 18,543 | 36 / 0 | 1–1 |
| mminstruct | 12,646 | 30 / 6 | 1–5 |
| sharegpt4v | 31,183 | 35 / 19 | 1–5 |
| svit | 114,559 | 54 / 0 | 1–1 |

其中 `llava_595k` 对应 CC3M 原图衍生的 LLaVA 预训练数据；`sharegpt4v` 和 `mminstruct` 本轮同时观察到单轮与多轮。SVIT 虽然上游有多种任务，本次取到的本地样本全部为单轮，不能按上游“存在多轮”直接替代当前记录结构。

### 3.4 更广义视觉对话：115 个来源中，18 个抽到多轮

除描述外，视觉 conversation 覆盖通用 QA、图表／表格、OCR、文档、定位计数、数学、医学／科学等。按视觉记录数最大的来源包括 PlotQA 549,642、ProVision10M 505,027、RenderedText 375,884、FinTabNet 267,441、MultiUI 221,414、UniChart 167,437、DocStruct4M 135,106、SBSFigures 127,801、SVIT 114,559；这些名称与规模不能作为全部是 caption 或全部是真实照片的证明。

**多轮来源的直接样本证据如下。第二列是该源全部视觉记录数，不是多轮记录数。**

| 已观察到多轮的视觉来源 | 本地视觉记录数（包含单轮） | 抽样：单轮／多轮 | 抽到的问答轮数 |
|---|---:|---|---|
| docmatix | 39,770 | 1 / 53 | 1–32 |
| sharegpt4v | 31,183 | 35 / 19 | 1–5 |
| som_llava | 16,977 | 0 / 36 | 4–5 |
| cosyn400k | 13,705 | 9 / 27 | 1–10 |
| mminstruct | 12,646 | 30 / 6 | 1–5 |
| llavar | 11,535 | 14 / 16 | 1–3 |
| scigraphqa | 9,266 | 6 / 12 | 1–5 |
| pixmo_docs | 7,936 | 0 / 18 | 8–11 |
| lvis_instruct_4v | 4,509 | 0 / 18 | 6–10 |
| visdial | 3,795 | 0 / 18 | 10–10 |
| as_v2 | 3,119 | 0 / 18 | 4–4 |
| mmdu | 1,832 | 0 / 12 | 10–13 |
| refcoco | 1,780 | 2 / 10 | 1–4 |
| cambrian_gpt4o | 1,768 | 8 / 4 | 1–2 |
| llava_med_finetune | 1,592 | 0 / 12 | 3–3 |
| visual7w | 933 | 0 / 6 | 3–6 |
| kvqa | 762 | 0 / 6 | 6–15 |
| visualmrc | 328 | 0 / 6 | 3–3 |

VisDial 是最明确的单图连续问答候选；MM-DU 提供多图多轮。Docmatix、LLaVAR、ShareGPT4V、MMInstruct 等本地有混合轮数；其他来源可能把多个问题串成多轮字段，仍需区分“角色上多轮”与“具有强历史依赖的自然对话”。这轮只判断来源与结构，不宣称抽样已验证全部语义连贯性。

当前只有样本层面的单／多轮判定，**尚无全量 3,507,428 条视觉对话的单／多轮精确计数**。若之后需要精确混合权重，可只扫描这部分记录的 segments.role/turn，不必重扫 88.9M caption 或图像 bytes。

## 4. 与 gen-und 的重合：已能确认共享来源，不能报实际重复率

这里分清三个层级：①同一原图来源／公开任务；②当前选中同一张图；③图、文字、角色轮次均相同的重复记录。本轮能确认多项①，尚未对两库执行全量 hash/ID join 来量化②③。

已整理 17 个来源家族，以及 63 个 DCVLM 任务源在 gen-und 中的可对齐分发入口。它们可能共享原图，却使用不同 caption／QA、split 或处理版本；不能把这些来源的行数加起来称为重复样本数。

| 原始家族／任务 | DCVLM 中的入口 | gen-und 中的对应入口 | 当前判断 |
|---|---|---|---|
| LAION | ReLAION 三目录，共 45,474,447 条 caption | laion-aesthetics、OpenUni/laion6m、DenseFusion、部分 ScaleCap | 共同 LAION 家族明确；并非证明 gen-und 子集全在本地 ReLAION small 中 |
| DataComp | datacomp_1b：42,175,337 | LLaVA-1.5 Mid-Training-85M 的上游配方列 DataComp-1B | 上游同源已披露；本地平铺副本中该分支覆盖与交集待确认 |
| SA-1B | internvl_sa1b_caption：372,030；部分 ShareGPT4V | Salmonnn/InternVL-SA-1B-Caption；FineVision/LLaVA 的 sharegpt4v(sam)；85M 配方 | 同原图家族明确，文本版本不同也可能值得保留 |
| CC3M | llava_595k：18,543 | Infinity/LLaVA-CC3M-Pretrain-595K；Mid-Data/cc3m_llava_recap；Salmonnn/sharecaption/cc3m | 共同 CC3M 来源明确；不能与 CC12M 混同 |
| MS-COCO | coco_caption、VQAv2、RefCOCO、VisDial、部分 ShareGPT4V/LVIS 等 | FineVision、Infinity、LLaVA、UniWorld COCO 分支 | 同原图、多种 caption／QA／定位任务；优先统一 image_id 和 split |
| ShareGPT4V | sharegpt4v：31,183 | Infinity、FineVision、LLaVA 多个 ShareGPT4V 分支；ScaleCap 部分原图 | 同产品家族明确，100K／1.2M／SFT mix 等版本不等同 |
| Visual Genome | GQA、SVIT、Visual7W 等 | LLaVA 的 vg/gqa/svit；FineVision Visual7W 等 | 共同原图家族，问答／描述可能不同 |
| OpenImages／TextVQA 家族 | openimages_caption、TextVQA、TextCaps、TextOCR/GPT4V | FineVision、LLaVA 的 TextVQA/TextCaps/TextOCR 分支 | 同原图可带不同文字与任务；不能只按记录文本去重 |
| 文档／图表／OCR | DocStruct4M、Docmatix、DocVQA、PlotQA、ChartQA、OCR-VQA 等 | Infinity、FineVision、LLaVA 对应子源 | 明确同任务／产品分发，多轮拆合方式和具体子集交集待测 |
| CC12M、ImageNet、RedCaps | 当前没有独立同名 source | 多套 Recap、Salmonnn、OpenUni 中明确存在 | **跨库重合尚未证实**；不能据此说完全不重合，也不能说已经确认重复 |
| PixMo | Cap／Points／Docs | Pixmo AMA；LLaVA 中 PixMo Docs | Docs 有同名分支；AMA 与 Cap/Points 不因同发布者就成为同一图集 |

关键上游依据：[LLaVA-1.5 85M 的来源声明](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)、[ShareGPT4V 原图与版本说明](https://github.com/ShareGPT4Omni/ShareGPT4V/blob/master/docs/Data.md)、[LAION 家族](https://laion.ai/blog/relaion-5b/)、[LLaVA-CC3M-595K](https://huggingface.co/datasets/liuhaotian/LLaVA-CC3M-Pretrain-595K)。本地目录路径及全部别名见 [来源重合专项表](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/dcvlm_sources_20261022/source_overlap.md)。

对你要合并的 caption/单多轮数据，最直接的影响是：**gen-und 中不少 FineVision／LLaVA／Infinity 分支会再次带入 DCVLM 已有的任务和原图家族。** 不建议将三个大合集按行直接相加；应保留真正新增的图像、有效长短描述、问题或对话上下文，并合并重复记录与身份。

本次未发现可支持“跨库实际图片重复率为 X%”的完整索引，所以报告中实际同图数和相同标注数保持未知，而不是 0。CC12M／ImageNet／RedCaps 可优先作为潜在新增原图候选核查，但没有通过真实 ID/hash 比对前也不能保证是净增量。

## 5. gen-und 的 T2I 排除 AI 生成图后，会少多少

### 5.1 能给出的数值结论

**按目前有本地计数／条件外推依据的 8 个 caption/prompt 来源，已知 AI 来源剔除量约 838 万，占该子集候选目标容量约 14.74%。** 这不是整个 gen-und 的比例，也不是去重后的独立图片比例。

这 8 源的清单和分母如下。单位均为**百万（M）图片目标出现次数**，按一份图片文件补齐标注后最多作为一个 T2I 候选来估算；不是现成通过配对验收的训练行数。多 caption 不再乘倍数。

| 纳入的来源 | 本地目标容量估计 M | 是否属于明确 AI 图专属来源 |
|---|---:|---|
| BLIP3o-Pretrain-JourneyDB | 4.190 | 是，Midjourney |
| JourneyDB-train | 4.190 | 是，Midjourney |
| BLIP3o-Pretrain-Short-Caption | 4.785 | 未确认其全量图像来源，不按名称排除；不代表已无 AI |
| CC12M_and_Imagenet21K_Recap | 22.681 | 原图重描述来源；仍需来源与图像筛选 |
| DenseFusion-1M | 1.100 | 模型生成 caption 不能当作生成图片 |
| DenseFusion-4V-100K | 0.117 | 116,720 个 PNG 文件名，不等于全部有匹配标签 |
| Stable_Diffusion_3_Recaption | 10.864 | 发布者使用 ImageNet/CC12M；不是 SD3 生成图专属源 |
| laion-aesthetics-12m | 8.931 | 网页混合源，未知内部 AI 比例 |
| 合计 | **56.857** | 已知生成来源 **8.380M** |

除 DenseFusion-4V 文件名完整清点外，上表主要依赖“完整分片目录数 × 少数完整叶目录均值”的条件外推。分片差异会影响结果，因此结论取约数。算式为 **8.3798 / 56.8567 ≈ 14.74%**。Journey 两个分发按源内记录分别计算，并没有宣称它们是 838 万张互不相同图片。[JourneyDB 官方来源](https://github.com/JourneyDB/JourneyDB)

MJHQ 的 30,000 张本地图片同样是 Midjourney 图，但它本来就是评测集；这里在形成训练分母前已经排除，不把它当作额外训练容量。[MJHQ 数据卡](https://huggingface.co/datasets/playgroundai/MJHQ-30K)

### 5.2 为什么不能将 14.74% 直接套到整个 gen-und

Long-Caption、85M 多来源训练数据等大源尚未完成本地全量配对清点。为了说明分母影响，可以使用以下**显式来源清单扩展情景**；新增规模来自上游标称，假设对应数据和标注都能恢复，且只执行“已知生成来源／分支”排除。85M 行还额外假设每条标称记录可形成至多一个合格单图 caption 目标，仅用于容量演示；实际计算须替换为经过任务资格筛选的记录数，恢复标注本身不代表全部适用于 T2I。

| 逐步扩展的候选清单 | 候选容量 M | 已知生成来源删除量 M | 删除比例 |
|---|---:|---:|---:|
| 上述 8 个本地可估量来源 | 56.857 | 8.380 | **14.74%** |
| 再加入 Long-Caption 的标称 27M | 83.857 | 8.380 | **9.99%** |
| 再加入 LLaVA-1.5 Mid 的名称标称 85M | 168.857 | 8.380 | **4.96%** |
| 再加入 OpenUni/text-to-image 的标称 2.01M 与 UniWorld/BLIP3o-60k | 170.927 | 9.813 | **5.74%** |

**四行都不是全合集的实测删除比例，也不是统计上下界。** 最后一行仍没有包含 OpenUni 的其他大分支、ScaleCap、Salmonnn 等全部来源。它说明：仅从已经识别的专属生成图来源看，分母中是否加入大规模重描述数据，会把占比从约 15% 改到约 6% 的量级；目前不能选其中一个数作为整个合集的确定答案。

Long 的 27M 是上游标称规模；LLaVA 的 85M 在此仅按名称标称进行容量演示，当前上游 viewer 已显示约 91.5M 行，两者都不是该挂载副本的本地实测。不能以标题替代精确数量。[Long-Caption 数据卡](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption)、[85M 数据卡](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)

OpenUni text-to-image 分支包含原图和 Flux/DALL·E 图。上游各成分近似相加约 2.2M，与 2M 标题并不严格一致：本模型按 512px 部分约 1500K/2200K 的生成占比归一到 2M，再加 10K 高分辨率生成图，得到约 **1.374M** 待排除；连同 BLIP3o-60k，新增约 **1.434M**。若直接采用各近似成分相加，最后一行情景约为 **5.81%**，而不是 5.74%；这种小差异远小于目前本地覆盖和未知图源的不确定性。[OpenUni 数据说明](https://github.com/wusize/OpenUni/blob/main/docs/DATASETS.md)、[text-to-image-2M 组成](https://huggingface.co/datasets/jackyhate/text-to-image-2M)

### 5.3 真正排干净 AI 图，还要处理混合来源内部的 AI 图

上面的 K/D 仅是来源规则删除占比。若未被整源排除部分还含未知比例 p 的生成图，则理想识别下：

`全部 AI 图删除比例 = K/D + p × (1 − K/D)`

例如最后一行的已知来源占比为 5.74%；**仅作敏感性假设**，若其余候选里还有 5% AI 图，则总删除比例变为 **10.45%**，约 1,786.91 万次目标出现。p=5% 不是本次测得的污染率，不是默认事实；实际检测的漏检、误判也尚未评估。

严格准入还可能因来源不明、caption 无法对应而隔离数据。这些损失要单独记账，不能全部叫作“AI 图被删除”。真实图配模型 caption 可以保留；不能因为 caption 是 GPT／Qwen 生成，就把该记录算进 AI 图分子。

### 5.4 去重和任务定义，会进一步改变分母

- 两套 JourneyDB 共享原始图源，必须在去重后重新计数；真实候选侧 CC12M/ImageNet/LAION/DenseFusion 也存在重复。因此不能只将 Journey 分子除以二、却维持其他分母不变。
- QA 答案、视觉多轮、纯文本不能整包反转成 T2I；需要独立生成并验收描述才会变成新的 caption→image 视图。
- UniWorld 的编辑、风格迁移、试穿和控制图任务不是全部纯 T2I。本报告没有把整个 2.7M 任务量加进 T2I 分母；生成编辑目标仍应排除。
- resized 图和原图、TAR 和解包目录不重复累加；多 caption 视图数、独立图片数、实际采样曝光应分列。

可用于规划的结论是：**明确 AI 专属源的删除量主要由 JourneyDB 两套驱动，已在可估量子集中达到约 838 万次目标出现；整个合集最终比例仍需建立选定 T2I manifest 后重算。** 更复杂的容量、配对可恢复率与背景 AI 情景已保存在 [估算专项报告](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/dcvlm_sources_20261022/genund_t2i_estimate.md) 和 JSON，均明确标记假设，没有把未知来源默认认证为真实图。

## 6. 对合并顺序的直接建议

1. **DCVLM 优先提供原生交错和既有单／多轮结构。** 不用 gen-und 的重 caption 平铺图替代原文档；也不需要把无关 pair 重新拼成所谓交错。
2. **gen-und 优先补有效的长短描述、图像域和结构标注。** 先查是否共享图像，新增 caption／QA 可以是有价值的新监督；不把相同图像再出现一次算成新增视觉覆盖。
3. **对共享来源建立统一身份映射。** 依次使用上游 image_id、字节 hash、pixel hash、原 URL 与近重关系；同图不同标注保留为视图，完全相同记录合并，所有视图继承 split。
4. **T2I 比例只在选定候选 manifest 上统计。** 明确图片目标、所用 caption、已知 AI 标签、配对状态和重复组；再报告删除行数／比例、独立图数／比例与实际采样曝光变化。样本条数比例不是文件字节比例或训练算力比例。

本轮仅执行本地元数据／文本结构只读分析与上游资料核对，未启动 CPU／GPU 实例，未修改现有语料、执行全库 AI 图过滤或训练。

交付：本报告、[165 来源精确 CSV](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/dcvlm_sources_20261022/dcvlm_sources.csv)、[来源重合清单](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/dcvlm_sources_20261022/source_overlap.json)、[T2I 估算模型](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/dcvlm_sources_20261022/genund_t2i_estimate.json)。
