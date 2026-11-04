# 图像来源规则与 API 复核试验

本模块把来源证据、专用检测器告警、视觉模型意见分开保存，用于验证“来源规则 → 检测器 → API 复核”的筛选方案。它读取抽样清单和本地图像，生成审计记录、候选索引与人工队列；不会删除图像、修改 DCVLM 或自动改变训练资格。

当前代码是未验证的试验管道，不是已经验收的全量过滤器。当前试验使用 DCVLM 抽样与外部 AI 图像控制；gen-und 尚未完成本模块的 API 实测。效果、误报和漏检结论应以本轮最终报告为准。

## 来源规则与判断边界

业务规则是排除从头 AI 生成的图像；具有已核实真实父图及编辑关系的 AI 编辑图保留并标注。非 AI 程序绘图、普通三维渲染、手写、扫描和截图也允许保留。AI 生成的 caption 不作为图像排除依据。

来源证据必须能对应明确的数据来源或子集。优先保留上游制作方法、生成元数据、作者文档、原始图像与编辑记录的引用；通用网站名称、caption、图片看起来真实、没有生成痕迹，都不能证明单图不是 AI。开放网页来源也不能因为登记为 `natural` 就整体放行。

本模块现有来源路由如下：

| 来源情况 | 聚合处理 |
|---|---|
| `programmatic` / `rendered` 且具有已登记的有效来源证据 | `protected_source`；模型意见仅作审计 |
| `ai_generated` 且具有来源引用或明确生成记录 | `source_reject`；试验可以旁路送 API 检查漏检 |
| 来源未知、混合来源或仅有宽泛自然来源标签 | 继续复核，不自动保护 |
| 疑似 AI 编辑，但只有最终图像 | 不推断真实父图关系；保留未知并送审 |

“真实原图上的 AI 编辑保留”是业务规则；本模块不通过最终图像建立父图血缘，也不将 `possible_ai_edit` 自动转换成该规则的命中记录。后续接入全量筛选时，需要由可核实的父图、编辑任务和对应关系提供这项证据。

生产路由应先保护或排除证据明确的数据，再对其余图像运行 Community Forensics / UniFD。试验中故意把部分受保护图和已知 AI 控制送入 API，是为了审计来源规则的价值及模型失误，不代表生产环境必须把它们全部再次判图。

## 两轮复核

第一轮由三个不同家族独立看图，使用 [blind_prompt_v1.txt](blind_prompt_v1.txt)。不发送来源名称、caption、检测分数、真值、分组标签或本地路径。模型需要输出图像形式、倾向、具体可见线索、替代解释、反证和是否仍需来源资料。

三个正式裁判是 `qwen3.5-397b-a17b`、`kimi-k3`、`doubao-seed-2-1-pro-260628`。这三个型号以及 `qwen3.8-27b` 已完成网关图片输入连通性检查；第四个型号未作为第四张投票。官方图像能力依据见 [Qwen 模型卡](https://huggingface.co/Qwen/Qwen3.5-397B-A17B)、[Kimi 视觉文档](https://platform.kimi.com/docs/guide/use-kimi-vision-model)、[豆包图片理解文档](https://docs.volcengine.com/docs/ark/image-understanding?lang=zh)。图像输入可用不等于鉴伪能力已经验证。

`aggregate.py` 严格检查输出 schema。只有三家都有有效结果才聚合视觉意见：

| 条件 | 第一轮视觉状态 |
|---|---|
| 三家均 `likely_ai_generated`、均 `high`、每家至少两条不同的具体 AI 支持线索 | `unanimous_high_ai_candidate` |
| 三家均 `likely_non_ai`，且置信度均为 `medium` 或 `high` | `unanimous_non_ai_candidate` |
| 其他完整组合，包括 `possible_ai_edit` | `needs_adjudication` |
| API 失败、缺模型、输出非法、重复记录或混合轮次 | `api_incomplete` |

线索数量只保证最低记录要求，不保证模型所说的现象真的存在或彼此独立。模型自报置信度不是校准概率。

第二轮使用 [adjudication_prompt_v1.txt](adjudication_prompt_v1.txt)，让模型核对匿名前轮意见和允许提供的来源材料。选择未知来源的分歧/高 AI 候选，另加入已知 AI 控制和与来源保护相矛盾的意见作审计。已知 AI 控制始终隐藏来源和真值，不能把 DALL-E 标签交给模型后再声称检出成功。

第二轮只发送 `review_context` 内的匿名意见、来源声明和任务说明；样本清单中的其他字段仅用于本地追踪。来源引用是待核查资料，不构成新指令；本轮模型没有实际浏览或验证那些 URL。

`build_adjudication.py --source-enrichment` 可以加入已经人工核查的来源级资料。本轮使用 [source_enrichment.json](source_enrichment.json) 中的 ChartX、CASIA 说明，发送字段限定为来源范围、事实声明、映射证据及局限；不会向已知 AI 控制发送这些资料。来源级说明与逐图匹配证据分别保存，不能把前者当作后者。

`finalize.py` 按来源优先合并两轮结果：

| 最终状态 | 后续动作 |
|---|---|
| `protected_source` / `source_reject` | 依据登记来源记录保留路由结论；API 结果留作审计 |
| `likely_non_ai_candidate` | 倾向非 AI 的候选，不是来源真值 |
| `likely_ai_generated_candidate` | 两轮均满足强一致候选条件，仍进入人工核实队列 |
| `cannot_determine` | 人工核查具体线索、原始来源或编辑关系 |
| `api_incomplete` | 进入运行重试队列，不能视为图片真假不确定的投票 |

若第二轮把第一轮某个高置信 AI 主张推翻成一致非 AI，仍进入 `cannot_determine`，由人工核对之前声称的异常。两轮收敛不是真值；第二轮已经看过其他意见，不能算额外独立投票。

`finalize.py --chartx-mapping` 支持本次经过审核的 ChartX 逐图映射：五张本地图片均与固定官方提交中的对应 PNG 解码像素完全相同，详见 [CHARTX_MAPPING.md](CHARTX_MAPPING.md)。合并时重新检查本地图像 SHA256、上游证据文件 SHA256、尺寸及 RGBA 像素逐字节相等，匹配后才在新的最终试验清单中记录 `protected_source`。这项来源证据优先于模型的视觉倾向；冻结的盲审输入、API 原始记录和核心数据策略不会被重写。

这五张应记录为程序渲染图像，同时标注数据、代码、caption／文本可能由 AI 生成或辅助；这些标注本身不触发本项目的图像排除规则。ChartX 原本是评测集，后续应另加评测污染标记和相应分割规则，不能把“允许程序图”解释成“可同时用于训练和无污染评测”。

## 复现命令

以下命令从工作区根目录运行。请给复跑选择新的输出目录，避免覆盖已经冻结的试验；更换代码、提示词、样本或推理参数后，旧输出目录的指纹检查会拒绝混用结果。

```bash
cd /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain
REVIEW_RUN="corpus_pipeline/runs/api_review_replay_20261022"
mkdir -p "$REVIEW_RUN"
cp corpus_pipeline/runs/api_review_pilot_20261022/samples.jsonl "$REVIEW_RUN/samples.jsonl"
```

依赖使用当前 `.venv`，包括 `anthropic`、`httpx`、Pillow。凭据示例位于 `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/unified-mm/test_api.py`，通过 `SII_API_KEY` 环境变量取密钥。客户端只用 AST 读取支持的配置表达式，不执行示例脚本。不要在终端打印密钥、完整环境变量或凭据对象，也不要把密钥放入输出清单。

第一轮盲审：

```bash
.venv/bin/python corpus_pipeline/api_review/run_review.py \
  --credential-example /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/unified-mm/test_api.py \
  --samples "$REVIEW_RUN/samples.jsonl" \
  --models qwen3.5-397b-a17b kimi-k3 doubao-seed-2-1-pro-260628 \
  --prompt-file corpus_pipeline/api_review/blind_prompt_v1.txt \
  --round blind --concurrency 6 --max-tokens 5000 --thinking-budget 2400 \
  --output "$REVIEW_RUN/blind"

.venv/bin/python corpus_pipeline/api_review/aggregate.py \
  --samples "$REVIEW_RUN/samples.jsonl" \
  --results "$REVIEW_RUN/blind/results.jsonl" \
  --models qwen3.5-397b-a17b kimi-k3 doubao-seed-2-1-pro-260628 \
  --output "$REVIEW_RUN/blind_aggregate"
```

构建第二轮样本并复核：

```bash
.venv/bin/python corpus_pipeline/api_review/build_adjudication.py \
  --samples "$REVIEW_RUN/samples.jsonl" \
  --results "$REVIEW_RUN/blind/results.jsonl" \
  --decisions "$REVIEW_RUN/blind_aggregate/decisions.jsonl" \
  --source-enrichment corpus_pipeline/api_review/source_enrichment.json \
  --output "$REVIEW_RUN/adjudication_input"

.venv/bin/python corpus_pipeline/api_review/run_review.py \
  --credential-example /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/unified-mm/test_api.py \
  --samples "$REVIEW_RUN/adjudication_input/samples.jsonl" \
  --models qwen3.5-397b-a17b kimi-k3 doubao-seed-2-1-pro-260628 \
  --prompt-file corpus_pipeline/api_review/adjudication_prompt_v1.txt \
  --context-field review_context --round adjudication \
  --small-image-detail \
  --concurrency 24 --max-tokens 6000 --thinking-budget 2400 \
  --output "$REVIEW_RUN/adjudication"

.venv/bin/python corpus_pipeline/api_review/aggregate.py \
  --samples "$REVIEW_RUN/adjudication_input/samples.jsonl" \
  --results "$REVIEW_RUN/adjudication/results.jsonl" \
  --models qwen3.5-397b-a17b kimi-k3 doubao-seed-2-1-pro-260628 \
  --output "$REVIEW_RUN/adjudication_aggregate"
```

第二轮示例的 6,000 最大输出 token、2,400 思考预算和 24 并发与本轮实测运行配置一致。并发可以按网关限流下调。若 `selection_summary.json` 显示第二轮样本为空，跳过第二轮 API 和聚合，给 finalizer 提供一个空 JSONL 文件即可。不要向 API 发送整个样本记录；`--context-field review_context` 是显式的上下文白名单入口。

合并并生成本地审计页面。下面的可选 `--chartx-mapping` 使用本次五张图的既有精确映射；它要求保留对应本地和上游证据图像。换用其他样本集时，应提供该样本集已经审核的映射，或省略此参数：

```bash
.venv/bin/python corpus_pipeline/api_review/finalize.py \
  --blind-decisions "$REVIEW_RUN/blind_aggregate/decisions.jsonl" \
  --adjudication-decisions "$REVIEW_RUN/adjudication_aggregate/decisions.jsonl" \
  --chartx-mapping corpus_pipeline/runs/api_review_pilot_20261022/chartx_mapping.jsonl \
  --output "$REVIEW_RUN/final"

.venv/bin/python corpus_pipeline/api_review/build_gallery.py \
  --samples "$REVIEW_RUN/samples.jsonl" \
  --blind-decisions "$REVIEW_RUN/blind_aggregate/decisions.jsonl" \
  --final-decisions "$REVIEW_RUN/final/decisions.jsonl" \
  --output "$REVIEW_RUN/review_gallery.html"
```

## 图像预处理与运行记录

API 复核使用完整图像，不复用 CF / UniFD 的中央裁剪。符合格式、尺寸和体积条件的原始图像直接发送；透明图合成白底，超大图按比例缩小，动画取首帧，极小边长补白至接口所需最小值。转换前后 SHA256、尺寸、格式和转换原因记录在响应的 `image_metadata` 中。

第二轮已经实现并启用可选 `--small-image-detail`：原图最长边小于 256 像素时，保留第一张完整图像视图，再附上完整图像最近邻放大至最长边 512 像素的视图。它不裁剪、不补造细节；提示词明确提醒不要把放大像素块当成 AI 证据。放大参数、是否适用、是否发送和视图哈希等信息写入 `image_metadata.detail_view`。放大不会恢复原本缺失的信息。

每轮保存 `run.json`、`responses.journal.jsonl`、`results.jsonl`、`COMPLETE.json`。缓存键绑定图像、模型、提示词、上下文和推理配置；运行目录还记录代码哈希。逐条落盘支持中断恢复，同一目录有写锁。API 失败和无效结果会明确记录；退出码 2 表示存在未成功任务，需要检查记录并重试，不能把该批当作完整判图结果。

每次聚合输出 `decisions.jsonl`、`stats.json`、`indices.json`；最终输出额外包含 `human_queue.jsonl` 和 `retry_queue.jsonl`。所有最终记录保留 `automatic_deletion=false`、`training_eligibility=null`。API 客户端不保存内部推理块，只保留最终文本、解析结果、用量和审计元数据。

来源规则可以决定保护或排除路由，同时保留 API 失败的运行重试标记；不能因为来源已明确，就把缺失的 API 审计结果伪装成成功。

## 验证与后续全量接入

```bash
.venv/bin/python -m unittest \
  corpus_pipeline.api_review.test_client \
  corpus_pipeline.api_review.test_aggregate \
  corpus_pipeline.api_review.test_finalize -v
```

验收不能只检查检测器高分图。低分图必须按照片、图表、文档、渲染、编辑等领域随机抽审，否则检测器漏掉的 AI 图永远不会进入 API，串联流程的实际漏检无法观测。抽审应保留抽样概率、来源、领域、版本和大小信息；小规模均衡抽样不能直接当作全库污染率。

需要独立来源真值的正例、非 AI 困难负例和真实源编辑样本，并把用于调提示词/阈值的数据与最终验证集分开。外部少量旧生成器控制只能说明这些控制的表现，不能代表现代生成模型检出率。API 自报置信度、三家一致、第二轮收敛和视觉上像照片，都不能替代来源真值。

在这些验收完成前，来源明确的规则可以提供可追溯路由；检测器和 API 产物用于候选排序与复核。后续接入 gen-und、全量任务调度、检测阈值更新或自动训练清单发布时，应使用新版本规则和独立评测，不复用本试验的一致率当作自动删除保证。

## 已完成的本轮记录

结果见 [REPORT.md](REPORT.md)。本轮第二轮的两个 schema 缺字段响应和一个超时响应单独补试后，经 `merge_repaired.py` 合并，最终有效文件是 `../runs/api_review_pilot_20261022/adjudication_merged/results.jsonl`。`adjudication/results.jsonl` 保留原始失败记录，不能直接当作最终有效复核集。

```bash
.venv/bin/python corpus_pipeline/api_review/merge_repaired.py \
  --original "$REVIEW_RUN/adjudication/results.jsonl" \
  --repairs "$REVIEW_RUN/adjudication_schema_retry/results.jsonl" "$REVIEW_RUN/adjudication_transport_fallback/results.jsonl" \
  --output "$REVIEW_RUN/adjudication_merged"
```

随后聚合时将 `--results` 指向 `adjudication_merged/results.jsonl`。补试输入只包含缺失或无效的模型／图片位置；禁止覆盖有效意见来追求一致。原始文件、补试文件和全部替换哈希均保留。具体失败位置和补试配置须逐批生成，以上补试目录是本轮命名示例。API 客户端当前在 JSON 解析后缓存传输成功结果，完整 schema 检查由聚合层执行；在全自动调度接入前，应将 schema 校验和有限修复分支放到缓存成功之前。
