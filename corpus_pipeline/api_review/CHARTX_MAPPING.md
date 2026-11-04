# ChartX 逐图来源核对

核对时间：2026-10-22。本轮 `samples.jsonl` 实际含 **5 张** `source=chartx` 的图像。它们全部唯一匹配到官方元数据，并与官方发布 PNG **尺寸及 RGBA 像素逐字节完全相同**。这比只凭数据集名称、caption 或视觉风格判断制作来源更强。

| 本轮编号 | 官方图像 | 元数据匹配方式 | 文件字节相同 | RGBA 像素相同 |
|---|---|---|---|---|
| A017 | `bar_483.png` | 完整 summarization 文本精确匹配 | 否 | 是 |
| A030 | `bar_375.png` | QA 的完整问题与答案精确匹配 | 否 | 是 |
| A040 | `bar_350.png` | 完整 description 文本精确匹配 | 否 | 是 |
| A048 | `bar_43.png` | 完整 redrawing Python 代码精确匹配 | 否 | 是 |
| A050 | `bar_381.png` | 完整 redrawing Python 代码精确匹配 | 否 | 是 |

五组本地及上游 PNG 都是 RGBA。上游文件带有 `Software` 和 `dpi` 元数据，本地文件没有这些字段；上游 `Software` 均为 `Matplotlib version3.7.1, https://matplotlib.org/`。因此不能声称文件哈希相同；实际证明的是全部解码像素相同，文件容器/编码存在差异。未通过重新运行不受信任的绘图代码来建立对应关系。

## 证据链

官方来源为 [InternScience/ChartX](https://huggingface.co/datasets/InternScience/ChartX)，锁定仓库提交 `31d1c7bd971a5209a45988fb6d706e1767176b6a`。下载时使用 HF 镜像的对应仓库及固定提交；记录保留官方仓库 URL。检索了官方两个元数据 JSON 的全部 6,000 条记录，每个本地样本都有唯一匹配。

随后对该提交的 `ChartX_png.zip` 发起 HTTP Range 请求，读取 ZIP 目录及五个匹配成员，未下载完整 437 MB 图像包。每个提取成员检查了未压缩长度和 ZIP CRC32，再分别计算本地/上游文件 SHA256、RGBA 像素 SHA256，并逐字节比较像素。没有声称验证了完整 ZIP 的 LFS SHA256，因为未下载全包。

官方 [论文第 3.2、3.3 节](https://arxiv.org/html/2402.12185#S3.SS3) 说明，ChartX 使用绘图库渲染图表，并使用 GPT-4 产生数据和认知任务内容；生成的绘图代码被用于渲染最终图像。因此这些图应细分为**LLM 辅助生成数据/代码、传统程序渲染像素**。不能将它们简单标成“完全没有 AI 参与”，也不能混同于扩散模型等直接生成的栅格图像。

在当前“允许程序图表，排除 AI 直接生成图像”的规则下，这五张已有足够明确的程序绘图来源证据可供保护规则审核；caption、代码及数据的 AI 参与应另外标注。若未来政策要排除 AI 编写代码形成的图表，那是另一个明确的规则变更，应保留这项制作方式差异。

## 本地记录与范围

- 逐图机器记录：`corpus_pipeline/runs/api_review_pilot_20261022/chartx_mapping.jsonl`，含原图哈希、官方成员路径、元数据行号/任务、双方文件及像素哈希、尺寸、CRC 检查和匹配元数据。
- 下载的官方元数据、ZIP 目录和五张 PNG：`corpus_pipeline/runs/api_review_pilot_20261022/chartx_upstream/`。
- 验证集元数据文件 SHA256：`49bff662b564a2fa8396564b2357495ba5ce5beb45008b4db5114c916450393e`。
- 测试集元数据文件 SHA256：`a729f56d75911baf23e949feb515c1a808ccc54733ff5906477418506c9be157`。

本次没有修改来源注册表、核心筛选规则、原始样本或 API 判断结果。上述逐图证据只覆盖这五张；不能据此宣称 DCVLM 中所有名为 ChartX 的样本都已经逐图核验。官方 ChartX 原本是评测集，这种精确匹配也应进入后续评测污染登记；是否保留为预训练数据和是否允许用于评测，需要分别决定。
