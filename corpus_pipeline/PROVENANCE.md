# 第一阶段来源准入与证据口径

此阶段将图像来源和文本标注来源分开管理。按用户最新确认，默认执行 `ai_only`，并启用 `allow_ai_edits_from_natural_sources: true`：AI 生成图像排除；有经确认真实源图血缘的 AI 编辑图保留并标记；程序绘图、合成文档和 3D 渲染可保留。来源不明的图片进入隔离索引；AI 生成的 caption、问答或 OCR 文本只记录来源，不因文本来自模型而删除。

这是一套**来源证据规则**，不是已经部署的逐像素 AI 图检测器。通过本阶段意味着“满足当前来源规则”，不能解释为“全部像素已验证、没有漏掉任何 AI 图”。未知数据不自动放行，因此网页大池会产生大量隔离记录。这种保守结果是当前证据不足的表现，不是测得网页全部为合成图。

## 1. 配置与可复核范围

- `config/sources.json`：634 条来源规则，包括现有 DCVLM 的全部 165 个正式来源、gen-und 的全部 21 个一级集合，以及 448 个已发现的语义子目录和明确生成分片规则。
- `config/sources_inventory.csv`：同一注册表的人类可读版本，列出来源、子集、图像/文本来源状态、证据等级、范围、链接和依据。
- `config/rules.json`：默认启用上述 AI 图像政策，并规定未知图像隔离、AI caption 保留并标记。可显式选择 `natural_only` 严格摄影图模式。

DCVLM 来源清单来自此前核对的 12,205 份发布记录，合计 119,010,208 条文档；这些是现有构建统计，不是本管道本轮已扫描和筛选的条数。gen-und 来源清单来自此前 CPU 分区只读挂载的目录与标注审计。本目录记录了原始证据文件的相对路径和 SHA256；规则运行不要求历史报告随代码部署。

165+21 是顶级来源覆盖数；634 是包含子目录的规则数，不能当成 634 个互不重复的数据集。子目录数量也不是图片数，历史文件量外推不参与准入判断。

## 2. 图像来源的含义

| `image_origin` | 解释 | 默认处理 |
|---|---|---|
| `natural` | 有明确来源构建证据的摄影图像；当前主要为 COCO、Visual Genome、SA-1B、iNaturalist 等已识别分支 | 来源门暂准入，仍需格式和图像引用有效 |
| `ai_generated` | 明确由生成模型产生目标像素 | 拒绝整条记录 |
| `ai_edited` | 生成式编辑、风格重绘等产生新语义像素 | 真实源图血缘经确认时标记保留；父图未知隔离，明确非自然父图拒绝 |
| `programmatic` / `rendered` | 程序合成、3D 渲染、代码绘图、合成文档等非自然图 | 来源门可保留，仍需通过引用与结构等检查 |
| `mixed` | 集合有多种来源，当前没有逐图区分证据 | 隔离，等待更具体证据 |
| `unknown` | 缺少足够来源证据 | 隔离，不能当自然图 |

本版本依用户澄清，将程序渲染与 AI 图像生成分开。PlotQA 等图表仍不标为摄影图像，但默认不因图表来自程序渲染而排除；CoSyn、PixMo-Docs 的代码即使由 LLM 产生，图像来源仍标记为 `programmatic`，文本标注来源另记。这些标签不会为了适配准入政策而改成 `natural`。

`natural_only` 保留为显式可选的严格摄影图模式，额外排除程序渲染等非自然图，且不启用 AI 编辑例外。注册表中历史依据里的“严格自然图策略排除”只描述该可选模式，不代表当前默认决策。政策调整不改变来源证据；旧版本试跑的拒绝数量也不能直接用于新版本。

`natural` 允许正常 resize、JPEG 编码、隐私模糊等不生成新语义内容的处理。照片拍到文字、图表或屏幕并不因为画面内容而自动成为生成图；相反，直接渲染的图表、网页截图或控制图不能仅因包含真实文字/真实数据而标为摄影图。本阶段未能确定具体生成方式的扫描件、图表、截图和混合艺术数据仍隔离，不凭模糊目录名批量贴 AI 标签。

SA-1B 的[官方数据卡](https://ai.meta.com/datasets/segment-anything)明确说明原图由摄影供应商提供，掩码由模型生成。因此原始照片可作为来源级自然图候选，模型掩码不意味着原始照片是 AI 图。独立导出的彩色 mask/深度/Canny 图片则是派生 raster，不能继承原照片的 `natural`；有明确程序派生证据时可按相应来源类型保留，不能因父图是真实照片就跳过来源与任务角色核对。

AI 编辑例外要求逐目标图记录可核对的父图血缘。输出的 `provenance.images[*].origin` 继续保留 `ai_edited`，并记下 `edit_source_origin`、`edit_source_evidence` 与 `policy_exception=ai_edit_from_natural_source`。源图只要尚未证明为自然来源，就不能启用例外；已明确属于生成图或程序图的父图也不符合这个例外。注册表没有为 Ghibli、UniWorld 等混合编辑分支猜测或补造父图声明。记录中有另一张自然图、文件名叫 original 或 prompt 来自 COCO，单独都不足以确认目标编辑血缘。`ai_generated` 不因参考图来自真实世界而改成 `ai_edited`。

## 3. 图像门作用于整条记录

无论最终分类为单轮 T2I、单轮 I2T、多轮生成、多轮理解或交错序列，都检查**全部图像引用**，包括条件图、目标图、历史轮次和多图输入。任何一张图触发当前政策的拒绝条件，就拒绝整个原始记录；一个符合例外的编辑目标不能替其他生成图豁免。

不自动剪掉坏图并继续使用其余对话或正文。那样可能留下“第二张图”“如下图”、编辑前后指代等失效上下文。若以后要回收真实 source 图，应建立有独立文本依据的新视图，并保留原记录和派生关系。

纯文本没有图像来源需要判断；不能因为所属混合集合 `image_origin=mixed/unknown` 而错误删除纯文本。也不能因为文件里只剩 `<image>` 占位符却缺少引用，就把缺图视觉记录误当纯文本放行。

## 4. 已编码的重点来源结论

| 来源或分支 | 图像处理依据 | 文本处理 |
|---|---|---|
| JourneyDB-train、BLIP3o-Pretrain-JourneyDB | [JourneyDB](https://github.com/JourneyDB/JourneyDB) 的 Midjourney 图像，明确拒绝 | 原始 prompt 与生成 caption 需分别追踪；不反推所有文本都是 AI |
| MJHQ-30K | [官方评测集](https://huggingface.co/datasets/playgroundai/MJHQ-30K)，同时属于 Midjourney 图像；AI 与评测隔离记录为独立原因 | 不混入训练 |
| UniWorld/BLIP3o-60k | [GPT-4o 蒸馏生成图](https://huggingface.co/datasets/BLIP3o/BLIP3o-60k)，拒绝；Human/COCO prompt 标签不证明当前目标图是自然照片 | 提示和目标像素来源分开 |
| OpenUni/text-to-image-2M | [官方组成](https://huggingface.co/datasets/jackyhate/text-to-image-2M)含 Flux、DALL·E 分支；明确分片拒绝，未拆分混合部分隔离 | 根据实际重描述/提示来源标记 |
| Stable_Diffusion_3_Recaption | [发布者](https://huggingface.co/datasets/gmongaras/Stable_Diffusion_3_Recaption)说明来自 ImageNet/CC12M，**不按 SD3 名称判生成图**；整体混合来源先隔离 | 模板和模型文本混合 |
| BLIP3o Long/Short、DenseFusion、CC12M/ImageNet Recap、ScaleCap | 模型生成 caption 不等于图片由模型生成；未知/混合原图仍隔离 | 已有明确重描述依据的来源标 `ai`，保留 |
| InternVL-SA1B | [原图加模型描述](https://huggingface.co/datasets/OpenGVLab/InternVL-SA-1B-Caption)，自然来源依据与 AI caption 分开 | 标 `ai`，保留 |
| PixMo-Cap/Points/AMA | 人工标注不能证明自然图；[官方例子](https://docs.allenai.org/training_data/pixmo)包括 AI 场景，网页图池不能全源放行 | 人工、转写、模型整理分为 `human`/`mixed`，不按文本排图 |
| CoSyn、PixMo-Docs | [CoSyn 论文](https://arxiv.org/abs/2502.14846)与[PixMo-Docs 卡](https://huggingface.co/datasets/allenai/pixmo-docs)明确代码渲染图片，默认可通过来源门；严格模式才按非自然图拒绝 | AI 指令独立标记 |
| SynthDoG、SynthText、CLEVR、FigureQA、DVQA、RenderedText、WebSight、PlotQA | 程序生成、渲染或合成产品，默认不因该来源类型排除；各条规则附上游依据 | 不能因为 synthetic 图像推定全部文本也是 LLM 文本 |
| UniWorld 其他编辑与控制任务 | 不把整个集合全标生成；编辑目标需逐图确认自然父图才能启用保留例外，程序控制图单独标记；未能拆清的记录隔离 | 等待 source/target 与标注配对 |

网页大池 DataComp、ReLAION、LAION、OmniCorpus、MINT 等不会因为“没有生成模型字段”就被当作自然图。后续应补充逐图来源、生成器记录、经过校准的检测结果或人工复核。

## 5. 文本标注的含义

`caption_origin` 的 `ai/human/mixed/unknown` 表示文本标注来源。对 QA/对话记录，它同样记录回答或指令标注的来源，字段名不代表该记录就能反转成 T2I。来源级标签和逐文本证据不同，输出必须保留其 scope/evidence。

- 明确 VLM/LLM 重描述：`ai`，保留。
- 人类语音描述再由转写/LLM 整理：`mixed`，保留。
- 原始网页 alt text：通常为 `unknown`，不能仅凭它来自旧网页就标人工。
- 缺少证据：保留 `unknown`；不得按 caption 中的 “AI” 字样、目录中的 “GPT/SD” 词或答案文风猜测来源。
- “AI caption” 与“AI image”是两个独立字段；两者都不能用一个 `synthetic=true` 替代。

来源级描述只覆盖有公开依据的已识别产品。未确认的 FaceCaption/AS100M 等文本生成链不会被编造为人工或模型标注。后续可用更具体的逐条证据更新标签，无需修改原始图像或正文。

## 6. 扩展、冲突与复现

新增规则应给出稳定 ID、集合/来源/路径范围、状态、证据 URL、证据等级、判断日期或版本、适用对象。优先收窄到原子子集或图片资产。已知 AI 生成来源的拒绝证据不能被泛化的自然来源标签覆盖；同一记录的另一张自然图也不能替未知图片背书或自动证明编辑父图身份。AI 编辑例外需要独立血缘证据，不能靠改写 `image_origin` 实现。

默认政策、来源注册表和运行参数应参与运行指纹。修改规则后应生成新的运行版本，保留上一版索引与原因，不能把旧运行的通过计数冒充新规则结果。隔离索引的意义是未来证据补齐后可重新评估，而非物理删除数据。

本阶段尚未完成的工作：逐像素 AI 图检测、生成式编辑检测、唯一图像/近重复识别、全部原始 split 恢复、评测去污染、缺失标注恢复、caption 忠实度与图片解码质量检查。公开来源白名单本身也可能受错误镜像或未披露加工影响。后续规则应能以逐图反证撤销暂定准入。

## 7. 编辑原图溯源字段

单图记录可在记录级声明；多图记录必须将字段绑定到对应编辑目标，或者使用明确覆盖该目标的来源规则。支持 `edit_source_origin`、`source_image_origin`、`original_image_origin` 和结构化 `edit_source_provenance`。例如：

```json
{
  "image": {
    "path": "edited.jpg",
    "image_origin": "ai_edited",
    "edit_source_provenance": {
      "origin": "natural",
      "source_id": "photo-123",
      "source_path": "original.jpg",
      "evidence": "受信标注或来源构建记录中的明确编辑关系"
    }
  },
  "caption": "编辑后的图像描述",
  "caption_origin": "ai"
}
```

这些是溯源元数据声明，不是像素鉴伪的结果。不同字段、内嵌 metadata 或来源规则如有反证，不能仅凭 `origin=natural` 放行；缺少足够证据时隔离。原图路径作为溯源信息保存，不自动加入训练图像序列。

`synthetic/generated` 或 `is_generated/is_synthetic_image` 等泛称不能单独证明使用了 AI 生成模型；明确的程序渲染来源可保留，否则作为未明确生成方式处理。`is_ai_generated` 等明确 AI 证据仍触发排除。
