# CASIA 与 ChartX：用于来源知情复核的补充证据

核验日期：2026-10-22。本文记录来源级背景，以及随后完成的五张 ChartX 图像的上游精确匹配；不修改核心过滤策略、原始数据或冻结盲审输入。第二轮已使用并冻结的 [source_enrichment.json](source_enrichment.json) 只包含较早的来源级背景，其“未完成逐图映射”限制描述的是当时证据状态，文件保持不变。新增逐图证据另见 [ChartX 核对报告](CHARTX_MAPPING.md)及下文。

## 当前记录为何不足

本次来源补充开始时，`corpus_pipeline/config/sources.json` 中 `dcvlm:casia`、`dcvlm:chartx` 均为 `unknown` / `unresolved`，只有 DCVLM 总来源表的引用。这表示此前登记的证据不足，不是检测到了 AI 图像。冻结 pilot 的两类记录也没有上游原始图像 ID 或逐图真实性标签。这些原有字段和冻结输入保持不变；新增 ChartX 逐图匹配写在独立记录中，不能把来源级背景本身当成逐图验证结果。

## 可核实的来源事实

| 来源 | 上游明确说明的生产过程 | 对复核的意义 |
|---|---|---|
| CASIA | 官方介绍称，CASIA-OLHWDB / CASIA-HWDB 的样本由 1,020 名书写者在 2007–2010 年用 Anoto 笔在纸上书写；离线数据来自扫描、分割和标注后的字符／文本图像。离线说明还明确背景为 255，前景为 0–254 的灰度值。 | 小幅字符图、白底和灰度轮廓本身与上述流程相容，不能仅据此判断 AI 像素生成。这里说的是上游数据库的过程，并未证明当前某张包装图片属于哪个官方原始文件。 |
| ChartX | 原论文 §3.2 描述 matplotlib、seaborn、plotly、mplfinance 等绘图库；§3.3 说明 GPT-4 生成数据、文本及绘图代码，随后执行绘图代码渲染图像。作者数据卡提供图像、CSV、Python 代码和文本四类内容。 | 在本项目允许程序绘图、包括 LLM 辅助代码的口径下，应区分“AI 生成数值／代码／文本”和“生成模型产生图像像素”。数值失真、单位不合理或不合现实的人口规模可以是数据质量问题，不能单独证明图像由生成式图像模型绘制。 |

## DCVLM 到上游的公开映射与边界

- **CASIA：** DCVLM 官方论文来源表将 CASIA 列为 Naive OCR 数据，并引用 Liu 等人的 2011 年 CASIA 数据库论文；其下载来源链接指向 `pascalbliem/handwritten-chinese-character-hanzi-datasets` Kaggle 镜像。这个链路支持“DCVLM 声明使用该 CASIA 来源”。本次没有审计 Kaggle 镜像的全部文件、转换脚本或生成历史，也未将本地图片匹配到 writer、GNT 文件及字符序号。CASIA 名称及该链接不是逐图真值证书。
- **ChartX：** DCVLM 官方来源表直接链接 `U4R/ChartX`，当前重定向到作者组织 `InternScience/ChartX`；相应作者代码仓库从 `UniModal4Reasoning/ChartVLM` 重定向到 `InternScience/ChartVLM`。作者数据集包含 `imgname`、`img`、CSV、description、summarization、redrawing 和 QA 字段。这构成明确的公开来源声明链路。随后完成的五张 pilot 图像精确匹配见下一节；全量 DCVLM ChartX 包装仍未逐项核对。
- **本地只读抽查：** CASIA 包装 key `15273` 的文本是识别字符、答案为 `7`；ChartX 包装 key `845` 保留了一段关于 USA/UK/Germany/France 手机及互联网使用人数的摘要。其 TAR 配套成员有对话、过滤指标及哈希等，没有在这两条记录中找到原始 CASIA writer/GNT 索引或 ChartX `imgname`／绘图代码。观察仅覆盖这两条配套记录，不代表已经遍历整个来源。

初次来源检查仅提出用 ChartX 保留的文本反查官方 JSON 的 `imgname`，再验证原图与包装图的像素关系；该步骤现在已对本轮五张 ChartX 图像完成。CASIA 仍需找回镜像转换和原始字符索引，尚无精确匹配。两类来源都不能据此宣称全源无污染。

## 后续完成：五张 ChartX 图像的精确上游匹配

检索固定作者仓库提交 `31d1c7bd971a5209a45988fb6d706e1767176b6a` 的全部 6,000 条元数据后，本轮五张 ChartX 图像均获得唯一的元数据匹配，并与对应官方 PNG 的尺寸和解码 RGBA 像素逐字节一致。

| 本轮编号 | DCVLM 包装 key | 官方图像 | 精确元数据匹配 |
|---|---|---|---|
| A017 | `845` | `bar_483.png` | 完整 summarization 文本 |
| A030 | `887` | `bar_375.png` | QA 完整问题与答案 |
| A040 | `288` | `bar_350.png` | 完整 description 文本 |
| A048 | `626` | `bar_43.png` | 完整 redrawing Python 代码 |
| A050 | `962` | `bar_381.png` | 完整 redrawing Python 代码 |

这是**像素一致，不是文件字节一致**：五组文件 SHA256 均不同，上游 PNG 带有 `Software` / `dpi` 元数据而本地没有；官方 `Software` 为 Matplotlib 3.7.1。核对没有执行绘图代码。结合作者明确的代码渲染流程，这五张已有逐图程序绘图来源证据；GPT-4 对数据、代码和文本的参与仍需单独标注。

完整记录见 [chartx_mapping.jsonl](../runs/api_review_pilot_20261022/chartx_mapping.jsonl) 和 [CHARTX_MAPPING.md](CHARTX_MAPPING.md)。核对通过固定提交的元数据与 ZIP 指定成员完成，记录了成员 CRC、尺寸、文件哈希和像素哈希；没有下载完整图像 ZIP，因此不声称验证了全包 LFS 哈希。结论仅覆盖上述五张，不能推广到整个 ChartX 包装，更不覆盖 CASIA。

## 一手引用

1. [CASIA 官方数据库介绍](https://nlpr.ia.ac.cn/databases/handwriting/home.html)：采集人员、时间、纸笔采集，以及在线／离线数据库的说明。
2. [CASIA 官方离线数据库说明](https://nlpr.ia.ac.cn/databases/handwriting/Offline_database.html)：页面扫描、分割、标注和灰度背景／前景格式。
3. [CASIA 原作者 ICDAR 2011 论文](https://nlpr.ia.ac.cn/databases/download/ICDAR2011-CASIA%20databases.pdf)：由人书写的在线与离线数据库及标准子集。
4. [ChartX 原论文 §3.2–3.3](https://arxiv.org/html/2402.12185#S3.SS3)：绘图库、GPT-4 数据／文本／代码生成与代码渲染流程。
5. [ChartX 作者数据集](https://huggingface.co/datasets/InternScience/ChartX)：数据模态、文件布局、原始字段及与论文的关联。
6. [ChartX / ChartVLM 作者仓库](https://github.com/InternScience/ChartVLM)：论文和作者数据集链接。
7. [DCVLM 官方来源表](https://arxiv.org/html/2606.28551v3#A5.SS3)：CASIA、ChartX 的引用与下载来源；[表中 ChartX 原链接](https://huggingface.co/datasets/U4R/ChartX)当前重定向到作者组织名下。
8. [DCVLM 表中 CASIA 的 Kaggle 镜像链接](https://www.kaggle.com/datasets/pascalbliem/handwritten-chinese-character-hanzi-datasets)：这里只引用 DCVLM 声明的下载目标；本次未审计镜像内容。

## 第二轮使用方式

补充 JSON 只用于来源知情的复核上下文，避免将“低分辨率”或“虚构统计量”误当成生成像素的充分证据。三模型第一次盲审输入和结果保持原样；第二轮应明确记录实际看到的来源事实。新增 ChartX 精确匹配不能反写到已经冻结或发送的第二轮输入中，也不能声称模型当时看到了尚未提供的证据。来源级推断和逐图匹配应分别留档。本次文档更新不改变 `source_image_origin`、核心过滤结论或训练资格。
