# 多模态训练语料管道

本目录包含代码、来源规则、测试、运行说明和实际试跑结论。目标是对现有 DCVLM 和 gen-und 建立可追溯的候选索引，按任务分类，筛除 AI 生成图像，并单独处理有真实源图依据的 AI 编辑图。源文件只读；“排除”指不进入保留索引，不删除原数据。

**2026-10-23 重构入口：** 使用 `python -m corpus_pipeline process` 或 `production.staged_run`，说明见 [分阶段管道 v2](PIPELINE_V2.md)，实际结果见 [重构与真实验证报告](REFACTOR_REPORT_20261023.md)。早期已有文本没有确认错误就保留，不确定性标记；仅确实缺失或确认错误才补写。默认 `--audit-fraction 0 --layout source`，按原图分辨率和来源标记后期 recaption／layout 潜力，不逐图调用 layout API，不自动选入未来高分池。当前图像筛选使用 `image_screening.v3`：DataComp、ReLAION 等 mixed／unknown 来源进入检测，不再由来源标签本身提前隔离；通过筛选后仍保留原始来源标签。

旧前置门下的真实 DCVLM 验证为 33 条记录中发布并读回 17 条、15 条来源待处理、1 条原图尺寸排除，0 次 recaption／原文审核／layout API。这些数字保留为历史证据，不代表 v3 的最终放行数量；下文其他旧先导数字也只属于对应历史运行。

Serving 后端使用平台模型部署端点，部署与队列状态以 [2026-10-23 执行记录](EXECUTION_20261023_SERVING.md) 对应时间的快照为准；该文不代表此刻实时状态。本次重构验证使用已授权的 Qwen API，没有新建训练 Job 或模型部署。此前的 [2026-10-22 记录](EXECUTION_20261022.md) 保留为历史快照。

**当前范围固定为 gen-und 与 DCVLM：提前准备每个来源／子集的证据、适配器和任务规则，运行时按配置校验并执行，不让 API 重复识别已知来源和格式。** 当前 DCVLM 用于先导验证，未来 DCVLM-balance-200B 挂载后必须比较版本和 schema，不能直接继承小池统计。逐源配置及只读试验见 [prepared](prepared/README.md)，优化建议见 [本次报告](prepared/REPORT.md)。最终发布引用原集的训练索引、决策和新增标注，不复制完整图片数据集。

后续阶段已新增 [真实 SSCD](dedup/README.md) 与 [完整记录先导执行器](production/README.md)：后者包含模型标注、独立质检、精选 layout、增量发布与训练读取。**19 条 DCVLM 原生记录已完成真实端到端验收，最终发布并读回 9 条记录、8 图和 1 份 layout**，详见 [实测结果](production/CANARY_STATUS_20261022.md)；这不代表全库已处理或质量准确率已验证。当前统一硬门槛是 **原始图片短边至少 128，无领域例外，禁止放大后通过；多图记录全部必需图片均须合格。** 下文 `python -m corpus_pipeline run/export` 仍指来源与结构索引入口，不自动执行这些后续阶段。
API 方案已完成 Qwen 的真实图像并发基准，128 并发主结果、格式恢复与容量算例见 [Qwen API 吞吐报告](API_QWEN_THROUGHPUT_REPORT.md)，可复现工具见 [api_benchmark](api_benchmark/README.md)。这些结果只衡量 API 输出，不代表质量通过或全库处理完成。
分阶段 caption 补充已接入 planner、执行器与发布读取，当前政策及命令以 [PIPELINE_V2.md](PIPELINE_V2.md) 为准。[早期复用与 1MP 阶段覆盖方案](STAGED_RECAPTION_PLAN.md) 保留设计背景；[策略 JSON](prepared/config/staged_recaption_policy.json) 记录实现合同，不是可任意执行谓词的规则语言。



**当前输入与任务分工：DCVLM 使用原始规范化父版，保持原生类型、方向与序列；T2I 训练候选仅来自 gen-und。** 默认 DCVLM 路径为 `/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z`，不再使用 T2I50 派生版。

**已有完整分阶段执行器和有界真实验证，尚未全量处理或完成亿级吞吐验收。** 来源不能保证每张图的制作方式时，通过检测和必要的 API 复核决定筛选准入，不要求先证明自然来源。明确 AI、尺寸不合格、关联错误、未确立真实父图的编辑和仍不能判断的警报样本继续排除或待处理。下文旧 `run/export` 的 `accepted` 仅表示来源、引用和结构候选，其 mixed／unknown `quarantine` 表示等待后续筛选；两者都不等于完整流程的最终训练准入。[检测器校准](detector_calibration/REPORT.md)、[API 复核试验](api_review/REPORT.md) 与 `VALIDATION.md` 保留各自历史范围。

历史 `VALIDATION.md`、`runs/` 与 `validation/` 中基于 T2I50 的结果保留为历史快照，其方向分布与双向候选数量不代表当前父版。此任务分工更新后必须新建运行目录；当前代码尚未重新全量处理父版或 gen-und。

## 文件与入口

| 文件 | 用途 |
|---|---|
| `pipeline.py`、`__main__.py` | 命令行、工作单元、检查点、索引发布、导出 |
| `adapters.py` | DCVLM、Parquet、JSON/JSONL、CSV、TAR、散图与同名标注适配 |
| `policy.py` | 规范化、任务分类、图像／文本来源判定 |
| `config/sources.json` | 165 个 DCVLM、21 个 gen-und 一级来源及嵌套来源规则 |
| `config/sources_inventory.csv` | 可读的逐来源台账 |
| `config/rules.json` | 本轮启用的筛选策略 |
| `config/adapters.json`、`ADAPTER_FINDINGS.md` | 已验证路径修正及尚待专用适配的标注结构 |
| `prepared/config/layout_source_contracts.json` | 186 个精确来源的 layout 路由与上游标注线索，默认不调用逐图 API |
| `PROVENANCE.md` | 来源结论、证据、AI 图像与编辑血缘口径 |
| `PIPELINE_DESIGN.md` | 总体目标：自动识别与路由、只读原集、增量标注、存储预算及训练引用读取；区分设计与已实现能力 |
| `LAYOUT_PIPELINE_PLAN.md` | layout 阶段设计：候选来源与分域质量、Qwen/Gemma 标注复核、索引契约和验收 |
| `dedup/` | 原图身份、官方 SSCD、历史／本批近重候选与真实验证 |
| `production/` | 完整记录分阶段执行器、原生保留、潜力索引、必要增量标注及发布读取 |
| `production/image_screening.py`、`image_screening_prompt_v3.txt` | 将实际视觉复核映射为筛选准入／排除／待处理；不改写图像来源 |
| `serving/` | HIGH / LOW Serving 部署、健康检查、资格验证与动态端点 |
| `test_*.py` | 分类、来源规则、配对、断点与失败恢复测试 |
| `runs/` | 实际索引、状态、试跑日志；不纳入 Git |

以下来源／结构索引入口依赖 Python 3.10+、PyArrow 和 ijson，不需要模型推理。后续 SSCD、检测器、API 和模型部署各有额外运行依赖，见对应子目录说明；不要把基础 `requirements.txt` 当作所有阶段的完整环境。CPU 分区需要单独准备依赖与代码。

```bash
python -m pip install -r corpus_pipeline/requirements.txt
python -m corpus_pipeline --help
python -m unittest corpus_pipeline.test_adapters corpus_pipeline.test_pipeline corpus_pipeline.test_policy -v
```

从 `corpus_pipeline` 的父目录执行命令，或把该父目录加入 `PYTHONPATH`。整个子目录可独立传输，不依赖本工程的训练模块或历史审计脚本。

## 分类与任务视图

| `category` | 定义 |
|---|---|
| `single_t2i` | 一轮文本条件生成图像，或原生文本→图像 pair |
| `single_i2t` | 一轮图像条件输出文本，含 caption、QA、OCR 等子任务 |
| `multi_t2i` | 多轮会话，图像出现在 assistant 输出，保留历史文本条件 |
| `multi_i2t` | 多轮会话，图像作为输入、assistant 输出文本 |
| `interleaved` | 具备原始顺序的图文交错文档 |
| `pure_text` | 无图像的文本或文本对话 |
| `mixed_multimodal` | 图像同时作为会话输入与输出，或无法简化为单方向的多图 pair |
| `unclassified` | 结构不明确、缺少必要关联或未支持的记录形态 |

一轮指实际的一组 user/assistant 交互。多图不等于多轮；不把若干独立图文 pair 拼成“自然交错”。I2T 是方向分类，不能把所有 I2T 回答都称为描述。

gen-und 的单图 caption pair 记录一次，其 `eligible_views` 可同时包含 `single_t2i`、`single_i2t`。DCVLM 的 caption 仅保留原生 I2T 训练视图，不再反转构造 T2I；若输入中存在原生 T2I，仍保留其 `category` 与顺序证据，但不授予 T2I 训练视图。不会复制图像或自动增加采样权重。普通 QA 仅保留自身方向，不会把“是的”“3 个”等回答反转成 T2I。多轮、交错和纯文本保持原始分类与完整上下文；gen-und 原生多轮 T2I 仍可作为对应训练候选。

## 当前完整流程的图像筛选

`process`／`production.staged_run` 的 v3 筛选先检查所有必需图片的原始字节身份、关联、解码和短边 128 门槛，再根据来源路由：

| 情况 | production 处理 |
|---|---|
| 明确 AI 生成 | 排除，明确负面证据优先 |
| 已验证的程序渲染等保护来源，或有真实父图证据的 AI 编辑 | 按来源保护并标记，仍须满足尺寸和完整性检查 |
| mixed／unknown 网页来源或其他未获保护图片 | Community Forensics＋UniFD；有效检测无警报则按筛选策略准入 |
| 检测警报 | 使用 [v3 提示词](production/image_screening_prompt_v3.txt) 调用 Qwen；有具体非 AI 证据、中高置信且本次无需外部证据时可准入；高置信 AI 且至少两条具体独立线索时排除 |
| 复核 uncertain、编辑真实父图未确立、检测或 API 失败 | pending，保留原因和实际回执 |

准入表示“通过当前筛选”，不会把 mixed／unknown 改成 natural，也不认证真实编辑父图。全部必需图片都须有准入回执；同一精确 SHA 可复用裁决和所选输入中的已验证保护证据，不同图片不能互相借用。API 排除资产形成撤销记录，作用于本次其他批次及合并的 base release，防止同图通过另一条记录回流。

## 基础 stage1 来源与结构规则（`run/export`）

以下是基础索引入口的既有规则；根 `policy.py` 未改成模型执行器。来源 `quarantine` 表示待后续筛选，production 只移除 `image_provenance_mixed`、`image_provenance_unknown` 两类前置原因并接上上述检测。其他结构、明确 AI、编辑父图和尺寸阻断仍保留。不能只导出 stage1 accepted 就认为已经覆盖全部可筛选网页数据。

按用户最新确认，默认执行 `ai_only`，并启用 `allow_ai_edits_from_natural_sources: true`：

1. 任一图像明确为 `ai_generated`，整条记录 `rejected`。真实图片提供了参考或 prompt，也不能使新生成的目标图自动获得编辑例外。规则作用于所有输入图、输出图、历史轮次和交错图片。
2. stage1 中图像来源 `unknown/mixed`、引用缺失／仅有 URL、格式歧义进入 `quarantine`。production 对其中仅因 `unknown/mixed` 暂存的图片继续执行检测；引用缺失和格式歧义仍须先解决。来源不明不等于 AI 图，不能把这里的隔离数量当成最终 pending 或 AI 比例。
3. `ai_edited` 仅在该目标图有经确认的 `natural` 编辑源图血缘时保留并标记；源图未知时隔离，源图明确非自然时拒绝。不能因同一记录恰好还有一张自然图，就推断它是编辑目标的父图。输出保留 `provenance.images[*].origin=ai_edited`，并记录 `edit_source_origin`、`edit_source_evidence` 和 `policy_exception=ai_edit_from_natural_source`。
4. 有来源依据的 `natural`、`programmatic`、`rendered` 图片可以通过来源门，仍检查格式和引用。程序绘图、合成文档、3D 渲染不会仅因不是摄影图而被排除；来源级证据不等于逐图验真。
5. AI caption、AI QA 和 AI 文本保留，分别记录 `provenance.text.origin`、`ai_generated`、证据与作用范围。`mixed/unknown` 不强行改成“人工”。
6. 已知评测来源有独立排除原因。尚未完成全量评测去污染。

`--policy natural_only` 是可选的严格摄影图模式：额外拒绝程序渲染等非自然图，也不应用 AI 编辑例外。默认使用上述 `ai_only` 政策。切换规则必须新建输出目录；旧版严格政策的计数不能当作新政策结果，实际验证口径见 `VALIDATION.md`。

图像来源与 caption 来源完全独立，例如 InternVL-SA1B 的原照片可暂准入，VLM caption 标记 AI；不会因 `Stable_Diffusion_3_Recaption` 的名称把原始 CC/ImageNet 图像全部判为生成图。原始网页 alt text 不自动标成人工。

## 两个存储分区分别执行

CPU 与昇腾存储不互通。`gen-und` 在 CPU 只读挂载，当前 DCVLM 在昇腾项目盘。两边运行同一份代码和规则，分别输出索引，再传回索引／汇总；传索引不等于图片已经跨分区搬运。

先执行逐来源诊断抽样：

```bash
# 在当前昇腾工程目录运行。
.venv/bin/python -m corpus_pipeline run \
  --collection dcvlm \
  --root /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z \
  --output corpus_pipeline/runs/dcvlm_native_tasks_sample \
  --mode sample --sample-units 2 --sample-rows 16

# 将 corpus_pipeline 整个目录传到 CPU 分区后，在它的父目录运行。
python3 -m corpus_pipeline run \
  --collection genund \
  --root /inspire/dataset/gen-und-mm-pretrain/01 \
  --output corpus_pipeline/runs/genund_native_tasks_sample \
  --mode sample --sample-files 64 --sample-units 2 --sample-rows 8
```

`sample` 是目录前缀／分片前部的诊断抽样，不能外推全库比例。它限制工作单元数与读取行数；TAR 为正确关联乱序成员，仍需要完整扫描选中归档的成员头和标注。压缩 TAR 的跳过也需要解压数据流，但不解码图片或提取文件。

全量入口已经具备；下面命令会实际遍历输入，不应把它的计划规模当作已经处理的样本数：

```bash
python3 -m corpus_pipeline run \
  --collection genund --root /inspire/dataset/gen-und-mm-pretrain/01 \
  --output /path/on/cpu/storage/genund_stage1_full --mode full

.venv/bin/python -m corpus_pipeline run \
  --collection dcvlm \
  --output corpus_pipeline/runs/dcvlm_native_tasks_full --mode full
```

同一命令、同一输出目录再次执行会校验并跳过已提交单元，重新处理失败单元。输入、代码、依赖版本、规则或参数变化时需新输出目录。一个目录同时只允许一个写进程。

## 输出与索引契约

每次运行输出：

- `run.json`：代码／规则／参数与数据快照身份；用于复现。
- `state.sqlite`：逐单元状态、输入指纹、提交回执；断点恢复依据。
- `index/<前缀>/<单元ID>.jsonl.gz`：流式压缩记录索引，每条均有决策和分类。
- `manifest.jsonl`：已成功提交的索引分片、校验和、逐分类计数。
- `summary.json`：按来源、类别、决策、原因与文本来源汇总。
- `progress.json`：最近完成的单元；处理中不反复扫描全部历史索引。
- `COMPLETE.json`：运行成功与否、是否抽样、是否单 worker 全量。解析失败或中断不会标为全量完成。

索引保留 `sample_id`、`collection/source`、原始记录 `locator`、顺序 `segments`、图像引用 `images`、轮数、`category`、`eligible_views`、`decision/reason_codes`、图像和文本来源证据。TAR 引用保留成员及偏移；压缩 TAR 明确区分解压后偏移，不能把它作为压缩文件 seek 偏移。

默认文本保留长度、SHA256 和原始记录位置，不再次复制全文；使用 `--include-text` 可内联文本。图像始终不复制。`sample_id` 是来源内记录身份，不是跨来源唯一图像 ID；已有图像 SHA256 单列为去重候选键。本阶段没有声称执行图片去重。

按类别或候选视图导出：

```bash
python -m corpus_pipeline export \
  --run corpus_pipeline/runs/genund_native_tasks_sample \
  --decision accepted --view single_t2i \
  --output corpus_pipeline/runs/exports/genund_t2i.jsonl.gz

python -m corpus_pipeline export \
  --run corpus_pipeline/runs/dcvlm_native_tasks_sample \
  --decision accepted --view single_i2t \
  --output corpus_pipeline/runs/exports/dcvlm_native_i2t.jsonl.gz

python -m corpus_pipeline export \
  --run corpus_pipeline/runs/dcvlm_native_tasks_sample \
  --decision quarantine --category interleaved \
  --output corpus_pipeline/runs/exports/dcvlm_interleaved_review.jsonl.gz
```

导出输出 `selected_view`，**只筛选引用，不重排原文或构造模型 loss**。下一阶段读取原始记录后，按选定视图构造训练样本。导出 manifest 带 `source_mode/partial_export`，防止把抽样导出误当全库；失败／未完成的运行不能导出。多个视图计数有交集，不能相加为唯一图片数。

`--decision accepted` 为训练候选导出：`single_t2i/multi_t2i` 必须同时来自 gen-und 运行和 gen-und 记录；对 DCVLM 显式请求 T2I 的 `--view` 或 `--category` 会报错，历史索引中的双向候选标记也无法绕过。默认导出若遇到非 gen-und 的原生 T2I，会跳过并在 manifest 的 `skipped_reason_counts.t2i_training_requires_genund` 记录条数。历史 DCVLM 的反转视图会从导出 `eligible_views` 中移除，旧值存于 `indexed_eligible_views`。

`--decision rejected/quarantine` 是审计导出，保留原生 T2I 等全部被审计类别，支持按原生类别筛选，不因训练来源限制静默漏掉记录。`export_purpose` 区分 `training_candidates` 与 `audit`；原始运行索引不被改写。

## 扩展规则与规模边界

v2 默认生成 `annotation_potential.jsonl`：原图面积至少 1,048,576 像素且短边至少 128 时标记分辨率潜力，面积可配；按来源页面与固定合同标记 layout 路由。潜力不等于质量、训练准入或未来池选择，本地结构标注尚需恢复验证。只有显式 `--layout select` 才启动图级布局选择、生成和复核；默认不生成 layout JSON。旧 `run/export` 仍仅负责来源与结构索引；历史设计见 [Layout 阶段方案](LAYOUT_PIPELINE_PLAN.md)，当前实现以 [PIPELINE_V2.md](PIPELINE_V2.md) 为准。

新增来源、嵌套子集或逐图来源证据可添加到 `config/sources.json`，填写稳定 rule ID、匹配范围、来源状态、依据和证据等级，更新版本后新建 run。明确的负面证据优先于宽泛自然来源标签；一张自然图的证据不能传播给同一记录的另一张未知图。

新增筛选阶段建议保持 `normalize → classify → evaluate → index` 的边界：在 `evaluate` 中加入独立规则函数，追加可审计原因，保留原始引用和已有判断，添加覆盖误放行／误拒绝的测试。以后可增加真实度检测、caption 忠实度、分辨率、OCR、去重、split 去污染及采样配比。

基础 `run/export` 入口目前适合建立首轮可复核索引，仍有明确的工程边界；下列范围不包含独立的 `production` 先导执行器：

- 全量 gen-und 发现使用磁盘 SQLite，临时盘需求随文件量增长；可通过 `TMPDIR` 放到合适的本地／项目盘。发现阶段目前需重新遍历，尚未做可恢复的持久化目录清单。
- 散图以 1,024 条组成提交单元。Parquet／TAR／JSON 标注以输入文件为提交单元；超大单文件中断后从该文件重跑，尚未做文件内行游标恢复。
- `--worker-index/--worker-count` 支持确定性单元分区，各 worker 必须用独立输出目录；当前每个 worker 会重复发现输入，不应直接在数亿文件上启动大量 worker。后续应先发布共享 inventory，再分发单元。
- ZIP、Arrow、LMDB 等未适配容器显式报单元失败；不会悄悄跳过并声称全量成功。需要关联外部标注、特殊 CSV、旧路径或不明 schema 的记录隔离／报错，等待专用适配器。
- 没有读取完整图像字节做验真、解码质量检查或新图像哈希；没有下载公网图片，也没有恢复尚未找到的标注。
- 两个集合的独立索引尚未进行全局去重、训练切分和采样混合，不能把两边条数简单相加为新增独立图片量。


### 来源规则与 API 复核试验（2026-10-22）

实测报告见 [api_review/REPORT.md](api_review/REPORT.md)，代码与命令见 [api_review/README.md](api_review/README.md)。本轮使用三家视觉模型审查 DCVLM 样本，并通过官方像素匹配保护五张 ChartX 程序图；人工队列和画廊在 `runs/api_review_pilot_20261022/`。这是独立审计模块，未修改原始训练数据或既有筛选结论。
