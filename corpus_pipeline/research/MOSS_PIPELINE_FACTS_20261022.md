# MOSS-Image 对增量数据管道的可复用事实与边界

核验日期：2026-10-22（Asia/Shanghai）。范围：arXiv v1、OpenMOSS 官方主页与仓库、SSCD 原作者仓库。本次重新读取官方材料；MOSS 当前 `main` 仍为 `52b87fca23acedc399d062d593bec388aee9926a`，递归目录树 167 项且 `truncated=false`。本报告未运行模型、API 或部署任务。

结论：可以采用 MOSS 的图像去重、多粒度图像描述和精选 layout 标注；其公开材料不足以直接复刻一个处理 T2I、I2T、多轮和文档交错的生产管道。后者需要我们的任务保真、增量索引、质量验收和数据发布协议。

## 1. 论文披露了什么

| 事项 | 核验结果 | 使用边界 |
| --- | --- | --- |
| 图像清理 | 来源自身预处理后，使用 SSCD 去重、自研 AI 图检测器 | 未公开 SSCD 型号、阈值、检索配置、冲突选择或完整生产脚本 |
| caption | Gemma 4；短句、中段、长描述；图像与原 caption 作输入 | 没有披露 Gemma 具体尺寸；粒度按内容定义，不是固定 token 配额 |
| 多视图采样 | 原 caption 与 recaption 都存在时 20:80；三种有效 recaption 均匀采样 | 这是 caption 视图比例，不是 T2I:I2T，也不是复制三份独立图像 |
| layout | 对精选高质量子集追加 bbox JSON | 没有给出可复刻的高质量评分阈值、子集规模或全库比例 |
| 结构验证 | 校验 JSON、字段、坐标和引用等，无效输出最多重试三次 | 语法通过不保证 OCR、物体与位置正确 |

上述来自 [§3.1–3.3](https://arxiv.org/html/2610.23374v1#S3.SS1)、[附录 C](https://arxiv.org/html/2610.23374v1#A3)。MOSS 是 T2I 模型，不能把其文本条件包装或样本 packing 当成已验证的多轮／交错训练配方。

另一个容易误用的数是 **192 px**：它属于附录 B.2 中多条件叠加的早期过滤消融；不是最终管道公布的统一尺寸门槛。最终方案省略该额外启发式过滤，强调保持覆盖面。该实验不能推导成“不能做尺寸检查”，也不能证明任意一个单独过滤条件有害。[附录 B.2](https://arxiv.org/html/2610.23374v1#A2.SS2)

## 2. Qwen 与 Gemma：官方推荐与本项目增强应分开

当前 [USERGUIDE / Caption Model](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/USERGUIDE.md#caption-model) 推荐 **Gemma 4、Qwen 3.5 27B、Qwen 3.5 35B-A3B**，建议 thinking。示例请求为 `max_tokens=16384`、`temperature=1`、`top_p=0.95`；这不是必须采用的吞吐或准确率最优值，也不证明全部预训练标注都由这些精确模型完成。Gemma 4 未指定大小。

公开 recaption 协议包含 `caption.long/medium/short`、`watermark[]`、`borders[]`。长描述关注可见主体、构图和原样 OCR；中描述压缩长描述，短描述概括主要内容。原 caption 仅供参考；水印单独记录，不写入三个 caption。[USERGUIDE / Image Captioning](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/USERGUIDE.md#image-captioning)

**本项目建议（不是论文结论）：**若采用更大的 Qwen 作为主标注器、Gemma 作为异家族复核器，应以 DCVLM 的分域人工标注集比较错误率、覆盖率和吞吐。模型大只决定候选配置，不替代质量验收。不要把同一张图两个模型的一致意见视为独立事实证明。

recaption 必须保持任务语义：T2I caption 可追加三粒度条件；I2T caption 可追加等价回答视图；已有 VQA、OCR、数学问答和对话不应统一改成图片描述。交错文档需要保存段落和图片次序；新增 caption 应作为图片侧标注保存，是否插入序列由任务配置决定。

## 3. SSCD 如何迁移，而不误删有价值监督

[SSCD 官方实现](https://github.com/facebookresearch/sscd-copy-detection) 推荐默认 `sscd_disc_mixup`，512 维；较高精度选项 `sscd_disc_large` 为 1024 维。描述子 L2 归一化，可用点积余弦距离。官方允许短边缩放 288 或直接缩放到 320×320 等固定方形，以利批处理；这是 SSCD 自己的预处理建议，不应与 AI 检测器或 VLM 的预处理混用。

官方给出的 **余弦相似度 >0.75、90% precision** 只是 `sscd_disc_mixup` 在 DISC 图像对上的示例。它既不是 MOSS 披露的阈值，也不足以作为 OCR／图表／文档的全局删除门槛。[SSCD / Using SSCD descriptors](https://github.com/facebookresearch/sscd-copy-detection#using-sscd-descriptors)

**工程建议：**

1. 先做文件哈希和规范化像素哈希，复用既有图像处理结果；将 SSCD 留给裁剪、缩放、压缩等近重复。
2. 每次入库同时检索“已发布索引”和“本批次索引”；仅查历史库会遗漏同批重复。索引写入需有批次提交和并发冲突控制。
3. SSCD 建立候选图像簇。对文档、图表、UI、手写题目，增加文字／数字、版面与局部差异检查，避免把同模板不同答案合并。
4. **图像资产去重和监督记录去重分开。**同图的不同问答、长短 caption、多轮上下文可引用同一资产；不能因为图像已存在就删除整个训练记录。相反，完全相同任务与文本也应去重，防止多来源复制造成过采样。
5. 近重复簇内选代表时考虑分辨率、清晰度、来源证据、是否裁切／编辑；保留成员与变换关系。不要未经核验将不同裁剪的 bbox 或 caption 原样挂到代表图。
6. 同一个重复簇尽量放在同一训练／验证分割中；多图记录通过共享资产形成的关联也要检查，避免跨分割泄漏。

阈值通过“确定副本、非副本但模板相同、不同裁剪、真实编辑前后”的分域图像对校准。检索 top-k、阈值、后处理版本都写入审计记录；不存在一个可从 MOSS 直接抄来的生产门槛。

## 4. 官方高质量案例能怎样用于 few-shot

官方主页是 [MOSS-Image project page](https://openmoss.github.io/MOSS-Image/)。仓库 [website/README](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/website/README.md) 说明主页布局图展示输入规格，showcase 来自仓库素材。

当前 `assets/images/001` 至 `020` 共 20 组，每组包含图像与 `parameters.json`；[assets/README](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/assets/README.md) 说明它们是带提示词、种子和采样设置的 Gradio 示例。核验的具体条目均注明 `model_repo=OpenMOSS-Team/MOSS-Image-Base`。它们属于**生成案例**，不是公开的非 AI 原始训练图与人审标注集合。

| 可参考条目 | 提示词体现的能力目标 | few-shot 使用限制 |
| --- | --- | --- |
| [001 参数](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/assets/images/001/parameters.json) | 摄影场景内的长中文路牌、透视与局部折叠 | 目标提示词不是已验证的输出 OCR／bbox；不宜直接当标注 gold |
| [003 参数](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/assets/images/003/parameters.json) | 纵向八格漫画、时序及气泡文字 | 可参考多面板组织；需要自己的非 AI 漫画来源与真实序列标签 |
| [006 参数](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/assets/images/006/parameters.json) | 书页照片、中文正文与脚注 | 可作为内容密度和排版目标；不能用其生成文字作为真实 OCR 标注 |
| [012 参数](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/assets/images/012/parameters.json) | 试卷扫描外观、题目和物理图 | 尤其需要数字、公式和图文对应复核 |
| [014 参数](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/assets/images/014/parameters.json) | 多人物健身摄影、人物关系与空间位置 | 可参考多实体定位需求；不是来源真实性示范 |

以上是对已读取**提示词元数据**的分类，不是本次对生成输出逐像素完成的质量判定。图像文件位于各条目同目录的 `image.png`。没有核实可直接下载的 MOSS 高质量真实训练图－标注 gold 子集。

建议把 MOSS few-shot 设为 `synthetic_reference_only`，用于解释“希望达到的描述与布局精细程度”；与训练资产、AI 检测校准集完全分离。然后从 DCVLM 建立自己的少量人审真实正负样例：

- 正例：清晰海报／多区域文档、文字可辨的图表、多实体空间关系明确的摄影图、多面板漫画。
- 反例：需要猜测的模糊字、主体被截断、几乎没有局部结构的简单图片、强压缩导致框无法稳定定位。
- 边界例：审美普通但结构监督丰富的表单；漂亮但局部语义稀少的照片。避免把“超高质量”仅等价为摄影审美。

这套 few-shot 应让模型输出不同维度：`visual_quality`、`text_legibility`、`layout_supervision_value`、`annotation_feasibility`、`uncertainty`。质量和 layout 适配度分别评分；不适合 layout 的图仍可留作普通 T2I／I2T。

## 5. layout 的可执行边界

当前 [layout 字段文档](https://github.com/OpenMOSS/MOSS-Image/blob/52b87fca23acedc399d062d593bec388aee9926a/skills/moss-image-prompt-enhancer/references/layout.md) 使用 `summary + items`；草稿为像素 `xyxy`，公开编译器转成以中心为原点的 `yxyx`。本项目应保存一个统一的原图坐标表示，所有模型输入坐标由代码转换；同时保存原图宽高和几何变换。官方生成草稿要求的 16 倍数画布不等于原始训练图片必须先裁成 16 倍数。

推理 prompt enhancer 会规划尚未生成的画面；真实图像标注则必须忠于现有像素。这两个 prompt 不能混用。将 JSON 生成安排在精选候选之后合理，但应先独立检查物体存在、OCR、框覆盖与遮挡，校验后才发布。建议先保留 `layout_candidate` 和评分证据，后续生产分支再生成并验收 JSON。

## 6. 给当前 DCVLM 小试验的建议

先以不改写原始记录的影子运行统计各阶段去除量和任务覆盖变化，再逐项开放入库。重点检查：尺寸门槛对字符/OCR 的损失、SSCD 在相同图表模板上的误合并、同图多监督是否保留、recaption 是否引入不可见事实、交错序列是否被拆坏。输出图像数、记录数、任务视图数三种口径；最终预训练收益仍需固定数据规模与训练预算的对照验证，不能由 API 一致率或 JSON 合格率直接断言。

## 补充核验：是否全图 recaption（2026-10-23，Asia/Shanghai）

附录 C.1 的生成配方明确按每张图生成短、中、长三档描述，因此应理解为面向图像的广覆盖标注设计，不能说论文仅对少量高质量图片 recaption。但这不构成全库 100% 成功覆盖的统计保证：论文从 94 个分片抽样 5,792 张图，其中 4,335 张具有完整三档，约 74.84%。这是该标注样本的完整 triplet 比例，不能外推为全库覆盖率；其他样本也可能已有部分描述，不能判定完全未 recaption。[附录 C.1](https://arxiv.org/html/2610.23374v1#A3.SS1)

原 caption : recaption 的 20:80 是两者同时可用时的训练采样概率；三档齐全时再均匀采样。它不是 80% 图片执行重描述，也不是完全替换原 caption。明确只用于精选高质量子集的是额外 layout 标注。[§3.3](https://arxiv.org/html/2610.23374v1#S3.SS3)

我们的“仅对缺少合格描述的资产补充 recaption”是适配现有多任务语料的工程选择，不应归因为 MOSS 已证明只需重写一个小子集。API 容量报告的约 204 天按 DCVLM 88,908,841 条 caption 全部重生成计算；不包含把 QA、多轮、交错或纯文本统一改写成 caption 的工作，也不是 MOSS 的实际处理工期。
