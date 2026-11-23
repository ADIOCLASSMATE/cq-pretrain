# gen-und-mm-pretrain:01 数据集实测报告

审计日期：2026-10-21（UTC；平台界面时间为 UTC+8）。对象：`/inspire/dataset/gen-und-mm-pretrain/01`。本报告使用当日 CPU 挂载实测与上游发布资料；旧审计仅提供待复查路径，未沿用旧存在性结论。

## 结论

**这套合集已经包含大量实际图片，不是只给图片 URL 的清单。21 个一级数据集都找到本地图像文件或 TAR 图像成员；对这些已落盘样本，不需要再从原站下载图像 URL。**

但“已有图片”不等于“整个合集可直接送进训练”：

- **8/21 个一级集合确认有图像和自然语言/对话文本；13/21 的文本在本轮有界检查中未确认。** 8个集合内部也可能只有部分任务标注已齐，不能视作全量覆盖率。第一轮presence是7个图文均有，补查UniWorld的TXT后综合为8个。
- **CPU分区与当前昇腾工程的存储不互通。** 本次由CPU Notebook执行脚本，通过远程传输取回结果，未使用“同路径即共享”的假设。平台对昇腾空间的挂载校验明确返回 `available replica not found in given storage`。因此后续仍要跨分区传输选定图片/训练包；这不是重新抓取公网图片URL。
- **合集包含不完整或经过改造的子副本。** `Ghibli-36k` 实际只有24组源图/结果图，共48张；`DenseFusion-4V-100K` 实际有116,720个PNG文件名；`redcaps5m/raw`还有135个`.aria2`状态文件。不能把上游标称数直接当本地可用数。
- **原始目录尚未适配本工程训练格式。** 需要图文关联、路径重写、来源/任务拆分与去重，再产出当前读取器要求的TAR图像池和`dcvlm.pretrain.v1`语料索引。此次没有执行大规模搬运或语料重建。

数据广场当前版本显示 **91.7 TiB、479,913,290 个文件**。这包含图片、标注、压缩包及可能的解包副本，既不是图片数，也不是去重后的训练样本量。证据：[平台版本快照](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/platform_dataset.json)、[昇腾挂载校验](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/ascend_mount_validation.json)。

## 本次检查范围与计数口径

使用已有专用CPU Notebook `genund-audit-1015-01`，4 CPU / 16 GiB，Python 3.10.12，只读挂载数据集。基础检查覆盖21个一级目录，追加检查37个stage/子集，再对UniWorld 23项、Salmonnn 4项、openuni 6项做路径和叶目录抽样。未下载公网图片、未解码图片、未验证图片内容能否正常读取。

共检查去重后的21个TAR和3个ZIP的有限成员清单；Parquet初次因无依赖未读，后在CPU临时目录安装PyArrow 21.0.0，补验3个Megalith Parquet的schema、footer及各3行。大JSON只取前1 MiB、最多3条记录，基础脚本最多8条；标注文件专项检查对每个一级数据集最多取3份文件，另外执行TAR内标注及定点补验。图片路径检查只做文件存在性检查。取回的结果用SHA256核对传输完整性。

`YES`表示观察到；`UNKNOWN`表示在预算内未确认，不能解释成没有。路径命中/同stem配对也不等于语义内容正确。目录抽样不是全数据均匀抽样；“条件外推”仅适用于假设其他分片与所抽分片大小近似的量级参考，不是统计置信区间。精确计数只对明确写出的浅层目录范围成立，未做全集递归扫描或图像去重。

原始证据：[annotation_semantics.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/annotation_semantics.json)、[presence.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/presence.json)、[inventory.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/inventory.json)、[annotations.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/annotations.json)、[nested_presence.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/nested_presence.json)、[parquet_followup.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/parquet_followup.json)、[exact_counts.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/exact_counts.json)、[shallow_counts.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/shallow_counts.json)。综合表：[CSV总表](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/dataset_summary.csv)。

## 21个一级数据集总览

下表“文本有”只说明集合中已发现文本，不保证每张图都配有文本。M=百万，K=千；“约”与“条件外推”均不能作精确计数。

| 数据集 | 图片 | 文本 | 本地量级/结构 | 上游规模（不可替代本地数） |
|---|---|---|---|---|
| BLIP3o-Pretrain-JourneyDB | 有 | 未确认 | 约419万文件（条件外推） | 4M图像 |
| BLIP3o-Pretrain-Long-Caption | 有 | 有 | 大目录；本轮未估全量 | 27M图像 |
| BLIP3o-Pretrain-Short-Caption | 有 | 有 | 约478万文件（条件外推） | 5M图像 |
| CC12M_and_Imagenet21K_Recap | 有 | 未确认 | 约2,268万文件（条件外推） | 约22.7M记录 |
| DenseFusion-1M | 有 | 未确认 | 约110万文件（条件外推） | 1.059M样本 |
| DenseFusion-4V-100K | 有 | 未确认 | 116,720个PNG文件名（精确） | 100K标注样本 |
| FineVision | 有 | 未确认 | 169个子目录；总图片量未实测 | 17.3M图像/24.3M样本 |
| Infinity-MM | 有 | 有 | 4个stage；总记录量未实测 | 44.8M阶段记录，含纯文本 |
| JourneyDB-train | 有 | 未确认 | 约419万文件（条件外推） | 4.453M原图/4.190M有标注图 |
| LLaVA-OneVision-1.5-Instruct-Data-Images | 有 | 未确认 | 157个子目录；总图片量未实测 | 父指令集约22M记录，非图片数 |
| LLaVA-OneVision-1.5-Mid-Training-85M | 有 | 未确认 | 平铺大目录；本轮未估全量 | 85M训练样本标称 |
| LLaVA-OneVision-Mid-Data | 有 | 有 | 7个图片子目录；8份JSON | 完整混合规模未确认 |
| Salmonnn | 有 | 未确认 | 4个子集；总量未实测 | 各子集分别说明 |
| ScaleCap450k_image | 有 | 未确认 | 图片目录+ZIP；总量未实测 | 450K标称/455,011标注行 |
| Stable_Diffusion_3_Recaption | 有 | 未确认 | 约1,086万文件（条件外推） | 10,862,932记录 |
| UniWorld-V1-new | 有 | 有 | 23个任务子集；含明显小副本 | 约2.7M任务样本 |
| laion-aesthetics-12m | 有 | 有 | 约893万文件（条件外推） | 约12M图文对的来源名称 |
| llava_ov_data_new | 有 | 有 | 79个子目录+聚合JSON | 当前上游约3.94M记录 |
| mjhq30k_imgs | 有 | 未确认 | 30,000张图片（精确） | 30K评测图像 |
| openuni | 有 | 有 | 6个子目录；总图像量未实测 | 6M/5M/CC12M/约10M/2M来源 |
| pixmo-ask-model-anything | 有 | 未确认 | 约15.4万文件（条件外推） | 161,737 QA行 |

## 已确认的整理障碍

1. **图片路径规则不统一。** Infinity-MM抽样9个引用需替换旧绝对前缀后才能命中；LLaVA-OneVision-Mid-Data的9个引用相对自己的数据集根目录；llava_ov_data_new的3个引用相对整个合集根目录。不能统一机械地拼一个根路径。
2. **对话与CSV另做了语义字段检查。** Infinity与两组LLaVA的7份JSON，共21条样本都有非空gpt/assistant回答，非仅有`<image>`占位符。StyleBooth前3条含非空EnStyle/ZhStyle/ShortStyleName，6个源图/目标图路径全部存在；末列SearchResult:DICT中的JSON逗号未按CSV转义，本次按前5个逗号切分后解析末列成功，不能直接套默认CSV读法。
3. **有URL字段不代表只有URL。** LAION的同stem JPG/TXT/JSON已在本地；Megalith的3个Parquet都是下载元数据，无图像binary列，但抽样9/9记录找到本地JPG。相反，Megalith的`caption`与部分TXT只是Flickr网页URL，不能充当描述；应关联独立JSON的`caption_internlm2_short`等字段。本次另外两份真实caption JSON也都找到对应本地JPG。
4. **有图文不代表索引开箱即用。** OpenUni索引中的相对图片路径有的指向TAR成员，普通文件路径检查会失败。TAR内已有原图，读取器需要TAR成员/偏移索引或重新打包。原始caption与recaption应按key关联，不能跨分片按行号直接zip。
5. **不能仅按目录名承诺完整性。** Ghibli只有48张图；DenseFusion-4V比100K名称多出16,720个PNG文件名，原因尚未确认；RedCaps的`.aria2`表明还留有下载状态，TAR存在并不证明整包完成。此次按要求只检查存在性，不做全包解码、CRC或下载完整性认证。
6. **重复来源很多。** JourneyDB两个版本、CC12M/ImageNet的多个recaption版本、DenseFusion1M/4V、LAION与OpenUni、RedCaps原尺寸/重采样版本、UniWorld多个COCO任务都可能复用图片。现有统计不能相加为“新增独立图片量”。

## 逐数据集说明

以下每项先给本次挂载证据，再给上游的内容、领域与规模；上游资料用于解释来源，不证明挂载版本等同于当前上游main。

### 1. BLIP3o-Pretrain-JourneyDB

**本次挂载：** 419个一级分片目录；3个完整样本各10,000 JPG，419×10,000≈419万。抽样只发现图片，caption未定位。

**内容与上游规模：**

这是 JourneyDB 衍生的合成图像与英文 caption 预训练数据，内容覆盖 Midjourney 场景、人物与物体、艺术风格及设计。官方数据卡标称 **4M 张图像**；资料查询时 HF viewer 显示约 **4.28M 条记录，属于估计值**。上游使用 WebDataset，字段包括 `jpg`、`txt`、`__key__`、`__url__`。

训练使用时应标记其合成图像来源。它与 JourneyDB-train 共享 JourneyDB 来源，合并时需按图像 ID 或内容核对重叠；caption 文本不同不能证明图片独立。数据卡概数与 viewer 估计不是精确清点。

来源：[官方数据卡](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-JourneyDB)、[BLIP3o 官方仓库](https://github.com/JiuhaiChen/BLIP3o)。

### 2. BLIP3o-Pretrain-Long-Caption

**本次挂载：** raw/ 下发现同stem JPG/TXT；抽取3份TXT均有自然语言，匹配图像文件存在。该单目录计到50,000条目即截断（25,001 JPG、24,999 TXT），不能把这些当全量。

**内容与上游规模：**

通用图像长描述数据，覆盖人物、物体、环境、场景与视觉细节。官方标称 **27M 张图像**；HF viewer 约 **27.2M 条记录，为估计值**。数据卡披露使用 `Qwen/Qwen2.5-VL-7B-Instruct` 生成约 **120 tokens** 的长 caption，以 JPG/TXT 配对的 WebDataset 分发。

官方未充分披露完整图像来源配比、重标注失败率和过滤阈值，不能把全部样本归为同一种来源。其与 Short-Caption 的图像交集尚未核实，不能默认是相同图像集合的长短两个版本，也不能默认完全不重叠。

来源：[官方数据卡](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Long-Caption)、[BLIP3o 官方仓库](https://github.com/JiuhaiChen/BLIP3o)。

### 3. BLIP3o-Pretrain-Short-Caption

**本次挂载：** 1,832个一级分片目录；3个完整样本分别2,577、2,642、2,616 JPG，TXT数量逐目录相同，均值外推约478万图像文件。已验证抽样同stem图文。

**内容与上游规模：**

短描述图文预训练数据，公开样例包括通用照片、商品、室内场景和插画；这些样例不能代表完整来源配比。官方标称 **5M 张图像**；HF viewer 约 **4.77M 条记录，为估计值**。数据卡披露使用 `Qwen/Qwen2.5-VL-7B-Instruct` 生成约 **20 tokens** 的短 caption，采用 JPG/TXT WebDataset 配对。

它提供较简洁的图像条件，但与 Long-Caption 的图像重叠关系未确认。混合训练和统计独立图像量时，应通过图像 ID 或内容核对，不能按两个发布名称直接相加。

来源：[官方数据卡](https://huggingface.co/datasets/BLIP3o/BLIP3o-Pretrain-Short-Caption)、[BLIP3o 官方仓库](https://github.com/JiuhaiChen/BLIP3o)。

### 4. CC12M_and_Imagenet21K_Recap

**本次挂载：** 追加浅层清点精确8,700个分片目录；3个完整样本各2,607 JPG，条件外推22,680,900个图像文件。当前挂载抽样为解包JPG，未确认上游Parquet中的recaption字段同时落盘。

**内容与上游规模：**

将 ImageNet 物体类别图像与 CC12M 网页图像组合，并提供长短 caption。作者正文描述约 **13M 张 ImageNet21K 图像和约 10M 张可用 CC12M 图像**；HF viewer 约 **22.7M 条记录**，均为上游近似口径。Parquet 内嵌 PNG 图片，字段为 `image`、`class`、`id`、`recaption`、`recaption_short`。ImageNet 部分的标注模型链接指向 LLaVA-NeXT 8B；短描述由 Meta-Llama-3-8B-Instruct 压成约 40 词以内的单句。

CC12M 部分按 ID 合并 LLaVA-ReCap-CC12M 图片与 CaptionEmporium 的 LLaVA-NeXT 描述，可用图片少于源名称中的 12M。README YAML 残留的 `num_examples=39108` 与正文和完整 viewer 不一致，不应作全集规模。另有 `_Highqual` 仓库，不能将其过滤结论用于本集合。与其他 CC12M、ImageNet 衍生集混合时应核对图片重叠。

来源：[发布者数据卡](https://huggingface.co/datasets/gmongaras/CC12M_and_Imagenet21K_Recap)。

### 5. DenseFusion-1M

**本次挂载：** 110个一级分片目录；3个完整样本各10,000 PNG，条件外推约110万。未定位caption JSONL，不等同于确认缺失。

**内容与上游规模：**

从 LAION 选取图像并经语义聚类、去重，利用标签、目标检测和 OCR 等专家线索生成稠密自然语言描述。领域覆盖通用照片、视觉艺术、商业设计与商品、信息图、文档和带文字图像。论文统计为 **1,059K 个样本**，HF viewer 约 **1.06M 条记录**；论文报告平均 caption 为 **1,130 字符、191 词、11.0 句**。上游 JSONL 包括 `image_id`、`caption`、`url`、`image_path`，PNG 图像独立分片保存。

构建流程先从约 1M 选图池抽取 100K 图像，由 GPT-4V 标注，用于训练 LLaVA-1.6-7B caption engine，再生成大集合描述。因此 DenseFusion-4V-100K 不能当成额外独立的 100K 图像直接累加。专家检测信息用于生成 caption，最终文本并不等于结构化检测标注；论文提示要求最终描述不包含检测坐标。

来源：[官方数据卡](https://huggingface.co/datasets/BAAI/DenseFusion-1M)、[官方仓库](https://github.com/baaivision/DenseFusion)、[论文 v2](https://arxiv.org/html/2407.08303v2)。

### 6. DenseFusion-4V-100K

**本次挂载：** 对48个浅层分片做文件名计数，精确116,720个.png条目，无截断；不是100,000，也不是三分片外推的120,000。未验证去重、标注覆盖或图片解码；不能直接当116,720个GPT-4V标注样本。

**内容与上游规模：**

这是 DenseFusion 的 GPT-4V 稠密描述子集，用于训练后续 caption engine，覆盖通用照片、视觉艺术、商品、OCR 和信息密集图像。官方名称与论文采用 **100K 个标注样本**口径；论文报告平均 **1,253 字符、206 词、11.2 句**。它是 `BAAI/DenseFusion-1M` 仓库内的独立配置，JSONL 标注与 PNG 图像分开，字段包括 `image_id`、`caption`、`url`、`image_path`。

该子集图像从 DenseFusion 的约 1M 选图池抽取，图像重叠是明确的构建关系。它可以作为不同标注质量或 captioner 训练用途的数据，但不能在独立图片量上与 DenseFusion-1M 直接相加。

来源：[包含该配置的官方数据卡](https://huggingface.co/datasets/BAAI/DenseFusion-1M)、[官方仓库](https://github.com/baaivision/DenseFusion)、[论文 v2](https://arxiv.org/html/2407.08303v2)。

### 7. FineVision

**本次挂载：** 一级169个子目录；抽样mapqa目录有37,417 JPG、CoSyn_400k_circuit有10,470 JPG，robut_wikisql计至50,000 JPG截断。未定位对话/评分标注。上游185 configs与本地169目录不是可直接对比的计数单位。

**内容与上游规模：**

多来源、多任务视觉指令集合，覆盖 caption 与知识、图表与表格、通用 VQA、定位与计数、数学、OCR、OCR QA、科学和纯文本。官方数据卡列出 **17.3M 图像、24.3M 样本、88.9M 对话轮次、9.5B 回答 tokens**；HF viewer 展示 **185 个配置**。这些统计单位不同，不能将样本数视为唯一图片数。

上游 Parquet 主要字段是 `images`、`texts`（user/assistant 列表）、`source`，一个样本可以含多图和多轮对话；还提供 `relevance`、`visual_dependency`、`image_correspondence`、`formatting` 及 `min` 质量评分。完整训练样本需要图像、对话及相应评分的关联。LLaVA-OneVision-1.5 官方建议与 FineVision 合并时去重。出现基准来源名称本身不证明评测污染，仍需检查实际 split 与原始 ID。

来源：[官方数据卡](https://huggingface.co/datasets/HuggingFaceM4/FineVision)、[官方浏览器](https://huggingface.co/spaces/HuggingFaceM4/FineVision)、[论文](https://arxiv.org/abs/2510.17269)、[LLaVA-OneVision-1.5 合并建议](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)。

### 8. Infinity-MM

**本次挂载：** stage1–4均存在。3个抽样JSON、每文件3条，9个图像引用均能在重映射后找到本地图片。JSON字段包括id、conversations、image，某些有qw2vl_loss。原image为旧绝对路径，需要将 /inspire/hdd/project/deepgen/public/data_zoo/source_images/ 前缀换成本次挂载根目录；此处证明路径命中，不证明语义配对或全集完整。

**内容与上游规模：**

四阶段视觉语言预训练与指令混合，覆盖图像描述、通用视觉指令、OCR、文档/图表/屏幕、数学推理、生成指令及纯文本。官方阶段规模为 **Stage 1：10M；Stage 2：25.8M；Stage 3：6M；Stage 4：3M**，算术合计 **44.8M 条阶段记录**。其中含纯文本和跨阶段复用，不能称为 44.8M 张独立图片。

Stage 1 是 10M 图像 caption；Stage 2 包含通用视觉 7.1M、OCR 2.6M、文档/图表/屏幕 5.8M、数学/推理 1.3M、纯文本 9M。Stage 3 包含 LLaVA-OneVision 3.5M、Infinity-Instruct 1.3M、Docmatix 1.2M；Stage 4 包含 GPT-4 生成 1.7M、合成 0.8M、专项任务 0.4M、Infinity-Preference 0.1M。常见字段是 `id`、`source`、`image`、`conversations`，另有标签、分数、phash 等辅助字段。

官方 2024-11-05 更新说明指出 `stage2/7M_0712_math_plus_system_release_0802` 不完整，替换为不带 `_0802` 的版本；准备数据时应核对发布版本。Stage 3 明确复用 LLaVA-OneVision、Docmatix，跨集合合并需要核对来源与样本 ID。

来源：[官方数据卡与更新记录](https://huggingface.co/datasets/BAAI/Infinity-MM)、[论文](https://arxiv.org/abs/2410.18558)。

### 9. JourneyDB-train

**本次挂载：** 200个一级分片目录；3个完整样本各20,949 JPG，条件外推4,189,800个文件，与上游有标注训练图像量级接近。但本轮未发现对应prompt/caption文件，数量接近不等于标注已齐。

**内容与上游规模：**

Midjourney 合成图像及生成 prompt、caption、风格/内容问答，适用于 prompt 反推、风格检索、图像描述和 VQA。官方 train split 统计为 **4,453,193 张原始图像、4,189,737 张有标注图像**；另有 **1,643,375 条原始 prompts、1,385,317 条有标注 prompts、7,056,394 条风格 QA、8,775,971 条内容 QA**。图像、prompt 与 QA 是不同单位，一个 prompt 可对应多图。

全数据集的 4,692,751 张原图、4,429,295 张有标注图像包含验证和测试，不能用于标称 train split。它是合成图像集合，且与 BLIP3o-Pretrain-JourneyDB 共享来源，合并时需核对图像重叠。

来源：[官方仓库与数据统计](https://github.com/JourneyDB/JourneyDB)、[官方项目页](https://journeydb.github.io)。

### 10. LLaVA-OneVision-1.5-Instruct-Data-Images

**本次挂载：** 一级157个子目录。基础广度优先扫描达到目录预算，image为UNKNOWN；追加随机路径清点在IconQA、wikipedia_2m、svit等叶目录实际发现JPG，因此综合结论为图片存在。示例叶目录分别有32,284、8,383、2,463 JPG；未定位指令标注。

**内容与上游规模：**

可核实的官方上游是父指令集 `LLaVA-OneVision-1.5-Instruct-Data`，包含文档/OCR、图表、数学几何、科学、通用 VQA、网页截图、百科知识及纯文本指令。数据卡标称 **22M 条指令记录**，HF viewer 约 **21.85M 条、属于估计值**；论文摘要另称 **26M**，存在版本/口径差异。**这些均为父指令集记录数，不能作为一个 `-Images` 图片目录的图像数。** 未取得与该目录名完全一致的独立官方数据卡与规模。

父集为多配置 Parquet，常见字段 `id`、`image`、`conversations`、`data_source`；对话字段随子集可能是 `role/content` 或 `from/value`。官方配置中存在整体名称和 `part-*` 名称，未核实行级交集前不能默认互斥后相加。官方建议与 FineVision 合并时去重；纯文本记录和多任务组织也使记录数不等于图像数。

来源：[官方父指令集数据卡](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Instruct-Data)、[论文摘要](https://arxiv.org/abs/2509.23661)。

### 11. LLaVA-OneVision-1.5-Mid-Training-85M

**本次挂载：** 抽样为平铺JPG，包含recs_obelics_*文件名；目录计至50,000 JPG仍截断。不能由名字85M认定本地恰好85M，也不能由来源名推断完整网页交错文档。caption未定位。

**内容与上游规模：**

多来源图像重描述与多模态中期训练集合，上游标称 **85M 个训练样本**。主分支列出 ImageNet-21k、LAIONCN、DataComp-1B、Zero250M、COYO700M、SA-1B、MINT、OBELICS 共 8 个来源，覆盖通用视觉概念、自然场景及网页/文档来源图像。当前 HF 发布文件体积约 **46.7 TB**，这是物理体积而非样本计数。

上游按来源、`EN/CN`、`partXX` 组织 Parquet。来源名称中的 1B、250M、700M 是原来源名称，不能当成该混合实际采用的份额；MINT/OBELICS 来源也不能单独证明发布版本保留了原始网页交错序列。引用主分支资料时，不应把 `refs/pr/2/README.md` 的未确认合并内容当成当前正式说明。

来源：[主数据卡](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M)、[主分支来源目录](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/tree/main)、[OBELICS Parquet 示例](https://huggingface.co/datasets/mvp-lab/LLaVA-OneVision-1.5-Mid-Training-85M/tree/main/obelics/EN/part01)。

### 12. LLaVA-OneVision-Mid-Data

**本次挂载：** 本地7个图片目录与8份JSON；聚合LLaVA-OneVision-Mid-Data.json为4,041,951,479 bytes，另有ureader、synthdog、ReCap等处理后JSON。3份JSON各取3条，共9个相对图像路径均命中；以该子集根目录解析相对路径。未全量计JSON记录。

**内容与上游规模：**

官方中期训练资源包含 `evol_instruct`、`synthdog_en`、`synthdog_zh`、`ureader_tr`，覆盖指令、合成文档 OCR、文档理解和中英文文本。官方明确直接上传 **processed JSON 标注与图片 TAR.GZ，二者独立分发**。已读数据卡没有给出可作为完整训练 mixture 精确总量的数字；HF 的 **60,200 预览行、563,196 估计行**属于首 5GB 自动图像预览口径，发布体积约 **45.1 GB**。

不能将该有限 viewer 统计写成整个中期训练集合的规模，也不能因为 viewer 只显示图像字段就判定没有文本。构建训练数据时需将独立 JSON 与图像路径关联；单独下载某一种资源不等于获得完整图文样本。

来源：[官方数据卡与分发说明](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data)、[官方文件目录](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Mid-Data/tree/main)。

### 13. Salmonnn

**本次挂载：** InternVL-SA-1B-Caption、imagenet-21k、sharecaption、densefusion-1m四项均发现图片。前三个平铺抽样大目录或类别目录不宜按几个叶目录外推。sharecaption还有11个来源目录。四个子集均未在本轮预算内确认自然语言标注。

**内容与上游规模：**

Salmonnn 是社区分发者，其数据应按具体来源分别描述，不能将社区重打包规模与原作者的图文标注规模混用。该集合没有可直接采用的统一独立图片总数。

| 子集名称 | 内容与领域 | 上游规模及单位 | 标注分发与训练注意 | 直接来源 |
|---|---|---|---|---|
| InternVL-SA-1B-Caption | SA-1B 自然场景；InternVL2-Llama3-76B 生成中英文描述 | 原作者标称 12M 图文对；文件分别标称 11M 英文单图、1M 中文单图、77K 英文多图、77K 中文多图；社区 512 版本 viewer 约 12M 行 | 原作者只分发 JSONL，图片从 Meta SA-1B 另取；字段含 id/image/conversations，多图 image 为列表。不同语言和多图记录会复用图像，不能相加为唯一图片；原作者文件数与 HF 有限预览估计不一致 | [原作者数据卡](https://huggingface.co/datasets/OpenGVLab/InternVL-SA-1B-Caption)、[社区 512 版本](https://huggingface.co/datasets/Salmonnn/InternVL-SA-1B-Caption-512/viewer) |
| imagenet-21k | ImageNet 物体类别图像 | 社区 viewer 约 9.58M 行，为估计展示；文件约 1.1 TB | TAR 图像分片；viewer 显示 jpeg.jpg/__key__/__url__，未显示 caption。WordNet 类别目录不等于自然语言描述 | [分发页](https://huggingface.co/datasets/Salmonnn/imagenet-21k)、[TAR 目录](https://huggingface.co/datasets/Salmonnn/imagenet-21k/tree/main) |
| sharecaption | 上游来源映射尚未确认 | 未确认，不能赋予 100K/1.2M 等规模 | 当前公开仓库列表没有直接同名项；不能仅凭名称认定 ShareGPT4V，也不能据此断言资源不存在 | [公开数据集列表](https://huggingface.co/Salmonnn/datasets) |
| densefusion-1m | 密集图片描述来源的图片重打包；候选对应 Salmonnn/dense | 候选分发 viewer 约 948K 行，为估计值 | 示例 key 前缀为 densefusion-1m，viewer 只显示 jpg/__key__/__url__。原始 DenseFusion 有 caption 不证明这个重打包 TAR 同含 caption；来源对应尚未做 revision/逐条核验 | [社区分发页](https://huggingface.co/datasets/Salmonnn/dense) |

这里应将“原作者标注数据”和“社区图片分发”作为两层资源管理，不能因名称对应就认定图文配套完整。

### 14. ScaleCap450k_image

**本次挂载：** 发现解包图片和ZIP并存；抽3个ZIP，均有图片成员、未确认文本成员，抽样成员数2,000／1,222／2,000。images/laion_filter_image和share4v_image均达到50,000条目截断。未定位ScaleCap450k.json，不能把目录中图片数当标注条数。

**内容与上游规模：**

上游 ScaleCap-450K 为单图超详细描述集合，面向多模态理解和 T2I，关注图像多样性、内容丰富度、高分辨率与适度复杂性。数据卡标称 **450K 图文对**，构成为约 ShareGPT4V 100K 加 LAION-5B 筛选 350K；当前 HF 标注表显示 **455,011 行**。

上游 `images/` 与 `ScaleCap450k.json` 独立分发，后者约 **1.46 GB**。schema 为 `images` 路径列表与 `conversations[{from,value}]`，通常是 human 图像占位符加 gpt 长描述。因此使用图片部分时必须另关联 JSON；450K 是数据卡称谓，455,011 是当前标注表口径，均不能仅由图片目录名推定。

来源：[数据卡](https://huggingface.co/datasets/long-xing1/ScaleCap-450k)、[文件目录](https://huggingface.co/datasets/long-xing1/ScaleCap-450k/tree/main)、[标注 viewer](https://huggingface.co/datasets/long-xing1/ScaleCap-450k/viewer)。

### 15. Stable_Diffusion_3_Recaption

**本次挂载：** 1,870个分片目录；3个完整样本5,809、5,810、5,810 JPG，条件外推约10,864,077个文件。本地抽样为解包图片，caption字段尚未定位。

**内容与上游规模：**

这是 gmongaras 社区整合的 ImageNet2012 train/val 与 CC12M 图文数据，**不是 Stability AI 官方原始训练数据发布**。当前上游为 **10,862,932 条记录**，Parquet 字段为 `id`、`image`、`caption`，`image` 为真实图片 bytes。ImageNet 使用类别文本模板；CC12M 从 LLaVA-ReCap-CC12M 获取图片，并按 ID 合并 CaptionEmporium 的 LLaVA-NeXT 描述。

ImageNet 部分的 `id` 统一为 `999999999`，不能用这个字段直接去重或估计唯一图片数。上游 CC12M 图片为 JPEG、ImageNet 为 PNG；与其他 CC12M/ImageNet 衍生集合混合时，重写 caption 不会消除原图重叠。

来源：[发布者数据卡](https://huggingface.co/datasets/gmongaras/Stable_Diffusion_3_Recaption)。

### 16. UniWorld-V1-new

**本次挂载：** 全部23个子集均通过追加目录清点发现图片。BLIP3o-60k的JPG/TXT样本可配对；StyleBooth有CSV，补查前3条的6个源图/目标图路径均命中，并有中英文风格标签；其末列JSON未正确转义逗号，需要专门解析。其他任务的完整指令/源图/目标图关系未确认。Ghibli-36k实际只有24组original.png/result.png，共48 PNG；不能视为完整36K版本。详细本地目录计数见下方表。

**内容与上游规模：**

可核实的官方上游是 UniWorld-V1，涵盖 T2I、图像编辑/修复、风格转换、商品提取/试穿以及边缘、深度、姿态、分割等视觉条件转换。官方称模型训练使用 **2.7M 个样本**，这不是全部发布子集规模之和或唯一图片数。数据分成 **图片资源与 annotation JSON**，训练配置分别指定图片根目录、JSON 和 region-weighting 开关。HF 首 5GB 的约 **7.11K 行**仅是部分 TAR 自动预览，不能当成全集规模。

下表为官方项目说明的子集/任务标称规模，不是任何本地目录的计数。双向转换及多任务可共享原图。

| 子集或来源 | 内容与领域 | 上游标称规模及单位 | 训练使用注意 |
|---|---|---|---|
| BLIP3o-60k | T2I | 约 60K 个样本；配置示例文件名含 58,859 | 官方为约一半样本增加 T2I 指令；名称概数与文件名计数不要混写 |
| OSP1024-286k | 高分辨率长 caption | 286K 个样本 | Open-Sora Plan 来源，Qwen2-VL-72B 标注；长宽比 3:4–4:3，美学分数至少 6，短边至少 1024 |
| imgedit-724k | 图像编辑 | 724K 个编辑样本 | GPT-4o 筛选，官方称保留原数据约一半 |
| OmniEdit-368k | 图像编辑 | 368K 个编辑样本 | 去除编辑区域小于画面 1/100 的样本；短边至少 1024 |
| SEED-Data-Edit-Part1-Openimages-65k | 图像编辑 | 65K 个编辑样本 | 输入/输出图片与指令配对 |
| SEED-Data-Edit-Part2-3-12k | 图像编辑 | 12K 个编辑样本 | 输入/输出图片与指令配对 |
| PromptfixData-18k | 图像修复与编辑 | 18K 个任务样本 | 应保留对应修复/编辑指令 |
| StyleBooth-11k | 风格转换 | 11K 个任务样本 | 需要关联源图、目标图及指令 |
| Ghibli-36k | 风格转换 | 36K 个任务样本 | 官方明确未做质量过滤 |
| viton_hd | 商品提取与姿态 | 提取约 23K、姿态约 11K 条任务记录 | 同源多任务，不能当成互不重叠的新增图片 |
| deepfashion | 商品提取与姿态 | 提取约 27K、姿态约 13K 条任务记录 | 同源多任务，不能当成互不重叠的新增图片 |
| shop_product-23k | 商品提取与虚拟试穿 | 23K 个任务样本 | 来自 Open-Sora Plan 内部数据 |
| coco2017_caption_canny-236k | 图像与 Canny 双向转换 | 236K 条转换任务记录 | 双向任务与其他 COCO 任务可共享原图 |
| coco2017_caption_depth-236k | 图像与深度双向转换 | 236K 条转换任务记录 | 同上 |
| coco2017_caption_hed-236k | 图像与 HED 双向转换 | 236K 条转换任务记录 | 同上 |
| coco2017_caption_mlsd-236k | 图像与 MLSD 双向转换 | 236K 条转换任务记录 | 同上 |
| coco2017_caption_normal-236k | 图像与法线双向转换 | 236K 条转换任务记录 | 同上 |
| coco2017_caption_openpose-62k | 图像与姿态双向转换 | 62K 条转换任务记录 | 同上 |
| coco2017_caption_sketch-236k | 图像与素描双向转换 | 236K 条转换任务记录 | 同上 |
| unsplash_canny-20k | 图像与 Canny 双向转换 | 20K 条转换任务记录 | 转换任务数不是唯一原图数 |
| open_pose-40k | 图像与姿态双向转换 | 40K 条转换任务记录 | 转换任务数不是唯一原图数 |
| mscoco-controlnet-canny-less-colors-236k | 图像与 Canny 双向转换 | 236K 条转换任务记录 | 与其他 COCO 图像可能重复 |
| coco2017_seg_box-448k | 检测与分割 | 448K 条任务记录 | 过滤区域小于 1/100 的实例，以原图上的 mask 可视化作为目标；实例/任务数不是唯一原图数 |

来源：[官方项目的数据组成与准备说明](https://huggingface.co/LanguageBind/UniWorld-V1)、[官方数据发布](https://huggingface.co/datasets/LanguageBind/UniWorld-V1)。

### 17. laion-aesthetics-12m

**本次挂载：** 精确2,420个一级分片目录；3个完整样本3,656、3,693、3,722 JPG，各有同数量TXT和JSON。按样本均值外推约8,930,607图像文件，不能当精确全量，亦不能直接认作12M。JSON含caption/url/key/status/width/height/sha256；URL是来源记录，旁边已有图片。caption样本含多语言网页短文本，未必都是英文或高质量描述。

**内容与上游规模：**

与名称最相符的官方定义是 LAION-Aesthetics V2 英文子集：约 **12M 个图文对**，预测审美评分 **至少 6**。它是从 LAION-5B 筛选的网页图文，侧重审美质量，文本通常来自网页图片描述，不保证是详细人工 caption。

不同阈值的 aesthetic 子集存在包含关系，不能相加；也不能与 V1 的 8M/120M 子集混淆。仅凭目录名尚不足以严格确认所用镜像、阈值或 revision。URL/metadata 记录与已下载图片是两层资源，需分别核对。

来源：[LAION 官方 Aesthetics V1/V2 说明](https://laion.ai/blog/laion-aesthetics/)。

### 18. llava_ov_data_new

**本次挂载：** 79个图片目录；llava_ov_all_with_image.json为3,453,296,293 bytes。抽3条均含id/image/conversations，3个图像路径均存在；image已经包含llava_ov_data_new/，应相对合集根解析，不能再重复拼该目录名。

**内容与上游规模：**

对应的公开上游是 `lmms-lab/LLaVA-OneVision-Data`，将 Cambrian、Cauldron、UReader 的部分子集及其他单源任务清洗、重标注后混合。覆盖通用 VQA、OCR、文档、图表、数学几何、网页截图和代码。当前发布页为 **89 个 configs、3,938,627 条记录**；这不是所有原始来源全量之和。

多数上游 Parquet 含 `id`、`image`、`conversations`、`data_source`，官方训练格式可转换为 JPG 与 JSON。`ureader_kg` 和 `ureader_qa` 则独立提供 processed JSON 与图片 TAR.GZ。完整训练数据需恢复图片与对话的关联；截图→HTML、截图问答等任务，不应仅凭网页来源名称归类成原网页连续正文与独立配图的交错语料。

来源：[官方数据卡、规模与格式说明](https://huggingface.co/datasets/lmms-lab/LLaVA-OneVision-Data)。

### 19. mjhq30k_imgs

**本次挂载：** 10个类别各3,000 JPG，精确30,000文件，未读取图片内容。本轮未定位官方meta_data.json，不能认定prompts随图片一起挂载。

**内容与上游规模：**

MJHQ-30K 是生成质量与审美**评测集**，官方为 **30,000 张图片、10 个类别**：animals、art、fashion、food、indoor、landscape、logo、people、plants、vehicles。`mjhq30k_imgs.zip` 分发图片，独立的 `meta_data.json` 以图片 stem 为 key，给出 `category` 和 `prompt`。

HF Parquet 主 schema 为 `image/label`，只检查它或图片目录可能漏掉独立 prompt 元数据。该集合应在数据台账中保留评测用途标记，不能与预训练集合无区别累加；如用于训练，需要另外处理评测重叠问题。

来源：[官方 README](https://huggingface.co/datasets/playgroundai/MJHQ-30K/blob/main/README.md)、[独立 prompt/category 元数据](https://huggingface.co/datasets/playgroundai/MJHQ-30K/blob/main/meta_data.json)。

### 20. openuni

**本次挂载：** 6个子目录全部抽到图像和文本。TAR、解包目录、JSON索引并存，不能把TAR和解包副本累计为独立图片。megalith的3个Parquet各有10,000元数据行、无binary图像列，9条样本全部在本地找到JPG；caption却是Flickr页面URL，真实描述需取captions/中的caption_internlm2等字段，另验证2份此类JSON与本地JPG配对。redcaps5m/raw有342个TAR及135个.aria2状态文件，是需要核对下载完成状态的线索。

**内容与上游规模：**

OpenUni 将图像源、重 caption 源和路径索引分别准备，既有自然网页图像，也有合成图像。多个目录不自动代表互不重叠的独立数据源；尤其不能将原始与 resized 版本、替代训练路线的标称量直接相加。

| 子集名称 | 内容与领域 | 上游规模及单位 | 标注分发与训练注意 | 直接来源 |
|---|---|---|---|---|
| laion6m | LAION 审美网页图片重描述 | 官方称从 laion-aesthetics-12m-umap 选取并重描述约 6M 个样本 | wusize/laion6m_recap 只发布 URL/caption，图片另下载；索引为 image/annotation 路径列表，实际 caption 字段为 re_caption | [准备说明](https://github.com/wusize/OpenUni/blob/main/docs/datasets/laion6m.md) |
| redcaps5m | Reddit 自然图片重描述 | 官方称从 RedCaps12M 成功下载并重描述约 5M 个样本 | 只发布 URL/caption，图片另下载；字段 re_caption，另有 image/annotation 索引。文档标题误写 6m，正文及仓库名为 5m | [准备说明](https://github.com/wusize/OpenUni/blob/main/docs/datasets/redcaps5m.md) |
| cc12m | 通用网页图像及面向生成的简洁描述 | CC12M 是来源名称；已读文档未给出可采用的精确成功配对数 | 图像来自 pixparse/cc12m-wds，文本由 PixelProse 再改写，发布于 wusize/cc12m_recap；图像 TAR、caption TAR、索引分开。官方将其作为获取 RedCaps 困难时的替代路线 | [准备说明](https://github.com/wusize/OpenUni/blob/main/docs/datasets/cc12m.md)、[总说明](https://github.com/wusize/OpenUni/blob/main/docs/DATASETS.md) |
| redcaps5m_resized | 疑似尺寸处理副本，关系待核实 | 未找到官方独立数据源定义或规模 | 不可直接当作新增独立 5M 数据集；与 redcaps5m 的关系需另行核对索引 | [官方总说明](https://github.com/wusize/OpenUni/blob/main/docs/DATASETS.md) |
| megalith-10m | Flickr 摄影与独立重 caption | 当前 drawthingsai metadata viewer 约 9.58M 行；源名称为 10m | 原图在独立 TAR，OpenUni 的 caption TAR/index 独立提供，训练字段为 caption_internlm2_short。原 metadata 示例的 caption 实际是 Flickr 页面 URL，url 为图片 URL，且含 success/error 状态；metadata 行数不能直接当成功图像数 | [准备说明](https://github.com/wusize/OpenUni/blob/main/docs/datasets/megalith10m.md)、[原 metadata](https://huggingface.co/datasets/drawthingsai/megalith-10m) |
| text-to-image-2M | 真实和合成图片混合的 T2I 数据 | 标称约 2M 个 512² 图文样本，另有 10K 个 1024² 样本 | WebDataset TAR 含图像与标注，OpenUni 另给路径索引，文本字段为 prompt；含大量 Flux-dev/DALL·E 3 图像，不能归为纯自然图片 | [数据卡](https://huggingface.co/datasets/jackyhate/text-to-image-2M)、[准备说明](https://github.com/wusize/OpenUni/blob/main/docs/datasets/text2image2m.md) |

T2I-2M 的公开组成约为：700K LLaVA-NeXT 原图加 Qwen2-VL 重描述、500K LLaVA-pretrain 提示加 Flux-dev 图像、900K DALL·E 3 合成图、100K GPT-4o 提示加 Flux-dev 图；另有 10K 高分辨率 Flux-dev 样本。各项是近似统计，不应加总后包装成精确总量。对 Megalith 等资源尤其需要区分**字段名叫 caption**与**字段值确为描述文本**。

来源：[OpenUni 官方数据准备总说明](https://github.com/wusize/OpenUni/blob/main/docs/DATASETS.md)。

### 21. pixmo-ask-model-anything

**本次挂载：** 77个一级分片目录；3个完整样本各2,000 JPG，条件外推约154,000个图片文件。即使一条QA生成一个图片文件，也可能反复保存同一原图，不能据此计算唯一图像数。本地QA标注未定位。

**内容与上游规模：**

Ai2 的视觉指令问答数据，内容涉及多样图片、物体识别、细节理解及较长回答。当前官方为 **161,737 条 Q/A 记录**；同一图片 URL 可以对应多个问答，所以不是 161,737 张唯一图片。

上游 Parquet 字段为 `image_url`、`image_sha256`、`question`、`answer`，**没有 image bytes 列**。图片需另行下载，官方提供 sha256 用于确认下载内容对应被标注图像。HF viewer 能显示远程图片也不证明图片文件已经下载；图片物化结果与问答标注应分别核实并配对。

来源：[Ai2 官方数据卡、格式与下载校验说明](https://huggingface.co/datasets/allenai/pixmo-ask-model-anything)。

## 内部子集的本地实测

下列计数只属于标出的目录。叶目录“≥50,000”表示枚举预算截断，不是整个子集只有50,000张；异构任务不能按少数叶目录直接乘倍数。UniWorld的上游任务规模见上文表，下面列真实挂载目录名。

### Salmonnn

| 本地子目录 | 图片 | 文本证据 | 局部文件/容器计数示例 |
|---|---|---|---|
| InternVL-SA-1B-Caption | 已见 | 未确认 | `.`：.jpg: 50,000（截断） |
| densefusion-1m | 已见 | 未确认 | `.`：.jpg: 50,000（截断） |
| imagenet-21k | 已见 | 未确认 | `n12344483`：.jpeg: 1,400；`n11906917`：.jpeg: 1,108 |
| sharecaption | 已见 | 未确认 | `sogou`：.jpg: 50,000（截断）；`laion400m`：.jpg: 50,000（截断） |

### openuni/data

| 本地子目录 | 图片 | 文本证据 | 局部文件/容器计数示例 |
|---|---|---|---|
| cc12m | 已见 | 已见 | `captions`：.tar: 2,176；`raw/0771`：.jpg: 5,041、.json: 5,041、.txt: 5,041 |
| laion6m | 已见 | 已见 | `captions`：.tar: 2,420；`raw`：.tar: 2,420 |
| megalith-10m | 已见 | 已见 | `captions`：.tar: 959；`raw`：.json: 1,918、.parquet: 959、.tar: 959 |
| redcaps5m | 已见 | 已见 | `raw`：.aria2: 135、.tar: 342 |
| redcaps5m_resized | 已见 | 已见 | `raw`：.tar: 584；`.`：.json: 1 |
| text-to-image-2M | 已见 | 已见 | `data`：.json: 3；`raw/data_1024_10K/data_000000`：.jpg: 10,000、.json: 10,000 |

### UniWorld-V1-new/data

| 本地子目录 | 图片 | 文本证据 | 局部文件/容器计数示例 |
|---|---|---|---|
| BLIP3o-60k | 已见 | 已见 | `occupation_2`：.jpg: 2,490、.txt: 2,490；`object_2`：.jpg: 951、.txt: 951 |
| Ghibli-36k | 已见 | 本目录完整枚举未见文本 | 全目录精确48 PNG，24组original/result |
| OSP1024-286k | 已见 | 未确认 | `images/laion-nolang/00086/00015`：.jpg: 11；`images/coyo2/00081/00161`：.jpg: 1 |
| OmniEdit-368k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| PromptfixData-18k | 已见 | 未确认 | `images`：.jpg: 36,746 |
| SEED-Data-Edit-Part1-Openimages-65k | 已见 | 未确认 | `auto_editing/openimages/images/source_images`：.jpg: 50,000（截断）；`auto_editing/openimages/images/target_images`：.jpg: 50,000（截断） |
| SEED-Data-Edit-Part2-3-12k | 已见 | 未确认 | `multi_turn_editing/images/data/20240308_235P_931turns/Data/275`：.jpg: 2；`multi_turn_editing/images/data/20240318_100P_410turns/Data/101475`：.jpg: 2 |
| StyleBooth-11k | 已见 | CSV含风格标签；3条源/目标6图均存在 | `stylebooth_dataset/BatchA/SimpleVectorArt/images`：.jpg: 216；`stylebooth_dataset/BatchA2/Impressionism/images`：.jpg: 203 |
| coco2017_caption_canny-236k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| coco2017_caption_depth-236k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| coco2017_caption_hed-236k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| coco2017_caption_mlsd-236k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| coco2017_caption_normal-236k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| coco2017_caption_openpose-62k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| coco2017_caption_sketch-236k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| coco2017_seg_box-448k | 已见 | 未确认 | `images/OpenDataLab___COCO_2017/raw/Images/train2017`：.jpg: 50,000（截断）；`images/drawed_coco/train_2017/bbox`：.jpg: 50,000（截断） |
| deepfashion-41k | 已见 | 未确认 | `images`：.jpg: 41,010 |
| imgedit | 已见 | 未确认 | `replace/replace_part4/00047_00000_000001441`：.png: 2；`remove/remove_part5/00047_00013_000132510`：.png: 2 |
| mscoco-controlnet-canny-less-colors-236k | 已见 | 未确认 | `images`：.jpg: 50,000（截断） |
| open_pose-40k | 已见 | 未确认 | `images`：.jpg: 40,000 |
| shop_product-23k | 已见 | 未确认 | `images`：.jpg: 23,901 |
| unsplash_canny-20k | 已见 | 未确认 | `images`：.jpg: 20,000 |
| viton_hd-34k | 已见 | 未确认 | `images`：.jpg: 34,941 |

OpenUni 6个子集的图片和文本存在性均已确认，但Megalith原始TAR的TXT可为URL；需用recaption JSON。15个原始TAR抽样中检出了同stem图像/文本成员，其中Megalith那份文本是URL，不能计作自然语言caption。其余5类原始TAR抽样的自然语言字段为`caption`、`original_caption`或`prompt`。LAION6M本地独立caption样本字段还出现`caption/detailed_caption`，与部分上游文档的`re_caption`有差异，转换程序应以本地schema为准。

证据：[tar_pairs.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/tar_pairs.json)、[megalith_caption_pairs.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/megalith_caption_pairs.json)、[uniworld_inventory.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/uniworld_inventory.json)、[openuni_inventory.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/openuni_inventory.json)、[salmonnn_inventory.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/salmonnn_inventory.json)。

## 用于当前训练工程的建议

**可先从已验证存在配对的样本建立小规模正式语料，再扩大覆盖。** BLIP3o Long/Short、LAION、OpenUni中的caption样本可分别转为I2T和T2I；网页短标题需要质量过滤，Megalith要选真实recaption列。Infinity与两组已找到JSON的LLaVA更适合保留原图像理解/问答任务，不应把问答答案一律反转成图像生成prompt。

FineVision、LLaVA-1.5 Instruct Images、85M、CC12M/Imagenet Recap、DenseFusion、ScaleCap、JourneyDB、PixMo等图片已见但标注未确认的集合，应先定位/补齐标注，并用原ID关联现有图片；不必默认重新下载原图。若确实无可匹配标注，再考虑重caption。UniWorld编辑/条件生成需要保留源图、目标图、任务指令与控制图含义；当前不能只抽一个JPG配任意一句文本。MJHQ-30K主要是评测图像，若用其做评测，应保留独立。

建议实施顺序：

1. 在CPU分区为选定子集建立统一manifest，记录来源、原始ID、任务类型、caption来源、图像本地路径或TAR成员、split和配对状态。将URL保留为溯源字段，数据读取不依赖公网。
2. 按上述schema规则建立图文对应，区分纯文本、caption、VQA、多图、编辑/控制任务；按原始图像ID及后续内容哈希去重。此次没有做哈希去重或全量有效性筛选。
3. 为当前工程输出TAR图像池与Parquet文档索引（`sample_id/source_shard/segments/images/text_tokens/format`），生成`manifest.json/catalog.json/sources.json/COMPLETE.json`及验证报告；固定tokenizer版本。当前`Catalog`会拒绝未完成或未验证的语料，因此原始目录不能直接代替正式数据路径。
4. 跨分区搬运选定数据包与索引，校验传输哈希，再在昇腾空间用真实数据做小规模加载/训练验证。不要假设CPU端的`/inspire/...`同名路径能被昇腾任务看到。

工程依据：[Catalog](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/ascend_pretrain/input_pipeline/catalog.py)、[本地TAR图片读取器](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/ascend_pretrain/input_pipeline/images.py)、[I2T/T2I构造](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/ascend_pretrain/input_pipeline/chat.py)。本次仅完成数据审计与报告，未更改训练配置或启动训练。

## 附件与复查

综合机器可读总表：[dataset_summary.csv](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/dataset_summary.csv)、[dataset_summary.json](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/dataset_summary.json)。已枚举的子集目录清单见[子目录清单](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/reports/genund_20261021/subset_directory_inventory.md)，包含FineVision、LLaVA-1.5 Instruct、llava_ov_data_new的实际目录名。

通用存在性脚本：[check_genund_presence.py](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/scripts/check_genund_presence.py)；用法：[说明](/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/scripts/check_genund_presence.md)。复查时应把脚本传到已挂载数据集的CPU环境运行，报告写到挂载目录之外。

```bash
python3 check_genund_presence.py /inspire/dataset/gen-und-mm-pretrain/01 \
  --sample-shards 3 --output /tmp/genund_presence_new.json
```

本次CPU Notebook审计结束后已停止，原有实例保留；临时HPC尝试均已结束或停止并删除任务条目。审计产物已取回当前工程目录，后续不依赖CPU容器的临时文件。

环境缺PyArrow时，Parquet结果会保留未确认与错误原因；不能把列名或URL字段当作已嵌入图片。此次原始presence文件保留初次缺依赖的记录，补验结论另存parquet_followup.json；综合报告使用补验后的证据。
