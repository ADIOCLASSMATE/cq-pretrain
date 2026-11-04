# 来源优先的 layout 潜力索引

本轮按数据集官方页面、作者代码及现有来源合同建立 `prepared/config/layout_source_contracts.json`。它只回答“哪些来源值得优先恢复结构标注或抽样判断”，不认证图像质量、不触发逐图 API、不改变当前图像准入或原始监督。来源名、长 caption、原图像素面积都不能单独证明适合 layout，更不能把当前集合自动定义成未来 1MP 阶段的数据池。

## 路由及覆盖范围

| 集合 | 明确来源条目 | layout_candidate | not_default_layout | sample_to_decide |
|---|---:|---:|---:|---:|
| DCVLM | 165 | 12 | 34 | 119 |
| gen-und 一级来源 | 21 | 0 | 8 | 13 |
| 合计 | 186 | 12 | 42 | 132 |

- `layout_candidate`：优先找原有结构标注或做有界适用性抽样；不是“该来源每张图都高质量”。
- `not_default_layout`：当前不默认开展 layout 工作；保留原任务训练，也不禁止以后明确选中的抽样。
- `sample_to_decide`：混合来源、版本或本地字段未解决，先恢复来源／抽样，不凭目录名分类。

本轮核对了 **21 个来源的第一方页面，加 1 个来源的第一方搜索摘要**；没有声称对其余所有网站重新审计。DCVLM 的 34 个非默认来源包括按已发布格式统计确认的 33 个纯文本来源，以及主要提供详细 caption 的 PixMo-Cap。119 个待定来源中，118 个尚无专门核实的 layout 领域，Chart2Text 则领域已知、具体版本和本地结构仍待核实。gen-und 的来源级大类来自已有合同，其 13 个混合来源仍需子来源身份与本地标注判断。

**186 个条目的 `local_binding_verified` 均为 false。** 这里指“具体原图／版本／裁切与结构标注的 ID、坐标对应已经验证”，与普通图文配对是否成功是两回事。本轮没有挂载 gen-und、读取新的结构标注包或调用模型。

## 哪些来源优先恢复现有结构

| DCVLM 来源 | 官方证据 | 处理建议及边界 |
|---|---|---|
| `chartqa` | 完整版本提供图表标注和底层表格；早期版本不带 annotations，作者提示部分框有噪声。[官方仓库](https://github.com/vis-nlp/ChartQA) | 先确定版本并按图像 ID 恢复；不能因原始 QA 可读就认为图表框也已恢复。 |
| `plotqa` | 独立 annotations 包含图元框、轴、标题、图例及数值；与 QA 文件分开。[格式说明](https://github.com/NiteshMethani/PlotQA/blob/master/PlotQA_Dataset.md) | 优先恢复精确 split/image_index 对应的原始标注。 |
| `dvqa`、`figureqa` | DVQA 提供柱、文本框、标签及底层表格；FigureQA 生成器同时输出框与源数值。[DVQA](https://github.com/kushalkafle/DVQA_dataset)、[FigureQA](https://github.com/Maluuba/FigureQA) | 程序绘制不应造成排除；现成结构可减少重新推断，但转换镜像可能只保留 QA。 |
| `docstruct4m` | 作者描述了结构文本与多粒度文本框，覆盖文档、网页、表格、图表和自然图像。[论文](https://arxiv.org/abs/2403.12895)、[官方项目](https://github.com/X-PLUG/mPLUG-DocOwl/tree/main/DocOwl1.5) | 优先恢复原任务字段；保留解析／定位目标，而不是统一改成 caption。 |
| `fintabnet` | IBM 原始数据预览展示表格／单元格几何与 HTML 结构。[IBM 数据预览](https://dax-cdn.cdn.appdomain.cloud/dax-fintabnet/1.0.0/data_preview/exploring_FinTabNet_dataset.html) | 本轮网页正文访问失败，证据限第一方索引摘要；还需确认本地表格裁切、坐标原点和对应版本。 |
| `websight` | 公开格式为截图＋HTML/CSS；v0.1/v0.2 内容不同。[官方数据卡](https://huggingface.co/datasets/HuggingFaceM4/WebSight) | 原代码可作为结构来源；若要算像素框，需要匹配渲染版本和几何，不把 HTML 直接冒充坐标真值。 |
| `clevr_math` | CLEVR-Math 引用 CLEVR 生成流程；CLEVR 输出 scene JSON、问题和功能程序。[CLEVR-Math](https://github.com/dali-does/clevr-math)、[CLEVR 生成器](https://github.com/facebookresearch/clevr-dataset-gen) | 有场景结构潜力，但当前数学 QA 镜像是否保存 scene、对应哪个渲染版本尚未确认；3D 坐标也不能直接当 2D 框。 |
| `coco_caption` | COCO 上游包含 caption、检测、分割、关键点等独立标注资源。[官方 API](https://github.com/cocodataset/cocoapi) | 摄影图片同样可能有对象布局价值，不能只把文档／图表算作 layout 候选；当前 caption 镜像未证明带有实例框。 |
| `renderedtext` | 官方 schema 可见 `ocr_annotation.bb_character_level` 字符几何。[官方数据卡](https://huggingface.co/datasets/wendlerc/RenderedText) | 优先恢复原字符多边形；是否值得扩成更复杂 layout 仍按实际内容决定。 |

另有两类只具备领域上的 layout 潜力，不能声称已有完整结构真值：

- `docvqa`：官方任务是文档图片上的问答，答案为文本跨度；这不是完整页面区域布局。[官方说明](https://www.docvqa.org/datasets/docvqa)
- `synthdog`：官方保存实现导出 `ground_truth.gt_parse.text_sequence`，因此目前能确认阅读文本，不能确认输出了完整区域框。[生成器代码](https://github.com/clovaai/donut/blob/master/synthdog/template.py)

PixMo-Cap 公开字段主要是详细 caption 与语音转写，长描述不等于结构标注；已有文本应复用。Chart2Text 官方项目提供源数据、标题及描述，但存在早期／后续版本，需先定位本地版本再定路由。[PixMo-Cap](https://huggingface.co/datasets/allenai/pixmo-cap)、[Chart2Text](https://github.com/JasonObeid/Chart2Text)

## gen-und 21 个来源怎样预先准备

| 路线 | 一级来源 | 依据与限制 |
|---|---|---|
| 不默认 layout，先保留原有图文监督 | BLIP3o Long、BLIP3o Short、CC12M_and_Imagenet21K_Recap、Stable_Diffusion_3_Recaption、laion-aesthetics-12m | 已有合同和本轮官方页面支持图文／类别描述字段，没有建立完整结构 JSON 合同。此标签不代表没有任何值得挑选的图片。[BLIP3o Long](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption)、[BLIP3o Short](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption)、[CC12M/ImageNet Recap](https://huggingface.co/datasets/gmongaras/CC12M_and_Imagenet21K_Recap)、[SD3 Recaption](https://huggingface.co/datasets/gmongaras/Stable_Diffusion_3_Recaption)、[LAION](https://laion.ai/blog/laion-aesthetics/) |
| 按现有 AI 图规则先排除 | BLIP3o-Pretrain-JourneyDB、JourneyDB-train、mjhq30k_imgs | 继承现有来源合同的整源拒绝，不做 layout 开销；没有调整用户的 AI 图规则。 |
| 按子来源与内容抽样决定 | FineVision、Infinity-MM、三个 LLaVA-OneVision 名称来源、llava_ov_data_new | 混合 QA／OCR／文档／图表／照片不能在父目录统一认定；FineVision 官方格式还含来源与评分字段，需恢复真实子集。[FineVision](https://huggingface.co/datasets/HuggingFaceM4/FineVision) |
| 恢复已有描述后按需抽样 | DenseFusion-1M、DenseFusion-4V-100K、ScaleCap450k_image、Salmonnn | DenseFusion 的描述含 OCR、对象和位置，但文字叙述不是机器可用的框标注；其余沿用现有合同，未在本轮全量重查。[DenseFusion 官方项目](https://github.com/baaivision/DenseFusion) |
| 必须拆分具体任务／分支 | openuni、UniWorld-V1-new、pixmo-ask-model-anything | OpenUni 六种 caption 不等于六种 layout；UniWorld 的控制／源／目标图需保留角色；PixMo AMA 的问答不能因为名字与 PixMo-Cap 相近就改成描述。 |

上述混合来源的子分支是否有 layout，只能在对应 schema 与图像关联恢复后升级，不能把上游网页的某个子集功能推广到整个本地目录。

## 在当前阶段怎样使用

先给记录附加来源级潜力和上游结构线索，再恢复已有标注；需要进一步判断的来源做可追踪的有界抽样。只有明确选中的样本才进入后续 layout 生成与独立验收。初期不因“有 layout 潜力”“分辨率高”“caption 不够详细”重写或丢弃可用监督；caption／答案确有实质错误才走对应纠错流程。未来高分辨率训练池需要另行选择和冻结，当前集合及本索引都不预先代表该池。
