# 分阶段数据处理管道 v2

当前入口为 `python -m corpus_pipeline process`，完整流程、命令和能力边界见 [PIPELINE_V2.md](../PIPELINE_V2.md)，真实验收见 [REFACTOR_REPORT_20261023.md](../REFACTOR_REPORT_20261023.md)。

## 当前图像质量门控

按最新要求“带模糊的直接不要”，`process`／`production.staged_run` 默认启用 [quality_policy.json](../config/quality_policy.json)，在原图完整性和尺寸检查之后、AI 检测之前执行 [quality_gate.py](quality_gate.py)。正常纹理、绘画笔触或渐变本身不构成模糊证据；其他图像质量、来源和图文配对规则仍独立生效。

**默认 `scope: reviewed_subset`，本次只改变原人工队列中 688 个图像 SHA256 的质量门控。DCVLM 全库尚未完成这套逐图模糊检查。** 相同字节的图像可复用其图像质量结论；范围外 SHA 明确标记 `blur_status: unreviewed`、`policy_decision: not_reviewed`、`quality_scope: out_of_scope`，继续原 AI 及其他准入流程，不凭这次复核新加质量阻断，也不宣称已通过模糊检查。原完整 overlay 的其余 69,804 项不会因这次策略被改成 pending。这次有界复核的通过率不能外推到全库。

| 逐图证据 | 当前记录处理 |
| --- | --- |
| 默认 reviewed_subset：范围外未审核 SHA | 标注未审核，继续原 AI 及其他门控 |
| 显式 all_images：未审核 SHA | pending：`image_blur_review_missing` |
| 审核仍不确定／来源待核实／语境待修复 | pending：`image_quality_review_pending` |
| 明确模糊或其他已确认的质量排除 | reject：`required_image_quality_excluded` |
| 已有明确 AI 生成排除 | reject：`required_image_policy_ai_excluded` |
| 质量决策 admit 且 blur 为 clear | 继续原 AI 筛选、去重和其他准入步骤，不代表真实来源已认证 |

多图记录的任何必需图槽被阻断，当前完整训练视图都不能发布。`quality_gate_evidence.jsonl` 与 `admission.evidence.image_quality` 保留逐槽原因和 `remove_image_and_repair_references` 等后续动作；**这些是待执行的修复说明，尚未删除图槽或改写引用**。原图、原始记录和其他文本保留。需要完成引用及任务结构修复后，才能另外构建合格视图。

`--quality-policy` 可指定另一份同协议配置，默认路径如下。配置必须启用 `enabled: true` 和 `require_blur_evidence: true`，证据要求在所选范围内生效。`scope: reviewed_subset` 是默认；后续显式改为 `scope: all_images`，才要求所有输入图片都有模糊复核证据，并将未审核 SHA 置为 pending。配置中的 `current_overlay` 相对于配置文件所在目录解析，指向独立的 `QUALITY_POLICY_CURRENT.json`，不覆盖 `HUMAN_REVIEW_CURRENT.json` 或旧人工决定。当前指针缺失、发布未完成、SHA 不符或 688 项绑定不完整都会报错，不会自动退回旧策略。

```bash
.venv/bin/python -m corpus_pipeline process \
  --manifest /absolute/path/to/selected_manifest.json \
  --output /absolute/path/to/new_run_directory \
  --quality-policy corpus_pipeline/config/quality_policy.json
```

只检查当前质量版本、统计其有界覆盖，不启动模型或发布训练集：

```python
from collections import Counter
from corpus_pipeline.production.quality_gate import load_quality_policy

state = load_quality_policy("corpus_pipeline/config/quality_policy.json")
print("reviewed_images", len(state["decisions"]))
print(Counter(row["report_category"] for row in state["decisions"].values()))
```

运行身份绑定 scope 以及配置、CURRENT 指针、输入 manifest 和 688 项决策文件的 SHA256；恢复运行不能换范围或策略后沿用旧身份。发布器和 reader 独立重读冻结的证据文件，重新推导每个图槽应为已审核、范围外未审核或阻断。质量 clear 和范围外未审核都仍须提供逐图 AI screening admit 收据；`--base-release` 也不能把缺少新门控收据的旧 admitted 视图混入新策略发布。后续切换 CURRENT 只影响新运行，旧冻结 release 仍验证其绑定的历史文件；源图与既有历史产物不会被改写。

相关测试：`python -m unittest corpus_pipeline.production.test_quality_gate corpus_pipeline.production.test_staged_run corpus_pipeline.production.test_staged_publish`。测试 fixture 只验证门控和发布逻辑，不计为真实图像审核或全库处理证据。

下文保留的是旧 `production.run` 固定 canary 的历史说明；它不是当前按需补写管道。旧回执用于溯源，不应将其无条件生成行为或 Kimi 默认参数套用到新入口。

# 完整原生记录的端到端先导执行器

实际完成回执见 `CANARY_STATUS_20261022.md`：19 条完整记录已跑通全部分支，最终可见 9 条／8 图／1 layout，疑点保留隔离；不代表全库质量或通过率。

该目录提供可执行的 DCVLM 先导，而不是将单图抽样结果直接当成完整训练记录。当前正式大规模输入仍须在 gen-und／DCVLM-balance-200B 挂载后，验证对应 profile、适配器和存储解析；本入口的固定 canary 不代表已处理整个集合。

## 行为边界

- 原图只读；从实际 Parquet 恢复完整记录，重新读取所有引用的 TAR 图片字节并校验 SHA256。
- **所有图片原始短边必须 ≥128，无领域例外。** 任一必需图片未通过，整条记录不可发布。纯文本跳过视觉阶段。
- 来源规则仍使用现有 `policy.py`：明确 AI 图拒绝、真实父图编辑按现行例外、未知／混合来源隔离。检测器低分和 API 判断都不把未知变成真实来源。
- CF384／UniFD 外部缓存仅接受明确冻结的 scores、modelinfo 和 inference_run 文件 hash；同时绑定权重、预处理、代码及 Python／torch／torchvision／timm 版本。缺失图实际推理并核对相同身份。告警按需调用保守 Kimi 图片复核。
- 真正执行 SSCD，与本批及指定历史特征库查询；精确资产共用、不同监督保留，近似候选进入图像对复核。特征库不是训练准入白名单。
- Qwen generator 按原生任务生成增量标注；Gemma verifier 独立检查事实与任务保真。单轮 QA 不改成通用描述，多轮与交错不重排，DCVLM 不生成 T2I 反转视图。
- 精选 layout 为独立生成＋视觉复核分支；坐标／schema 合法不等于事实正确。失败 layout 不被发布；原始任务仍保留。
- 发布仅包含通过完整来源、图片、模型质量门的记录。unknown canary 可用于明确标注的审计生成，但不会进入训练发布。
- 发布包含原图引用和新增标注，先实际读回，再原子切换 `current.json`。读取整个 parent 链，并按 `(collection, original_sample_id)` upsert；同批重跑复用同一 release。

## 命令

```bash
.venv/bin/python -m corpus_pipeline.production.run prepare \
  --output corpus_pipeline/production/canary_manifest.json

# registry.api_key_env 由启动进程安全注入；原生 Serving 当前使用 INF_API_KEY。
# 不能把密钥写到命令参数或日志。
.venv/bin/python -m corpus_pipeline.production.run run \
  --manifest corpus_pipeline/production/canary_manifest.json \
  --output corpus_pipeline/production/runs/dcvlm_canary_20261022 \
  --endpoints corpus_pipeline/serving/endpoints.json \
  --record-concurrency 4 --model-concurrency 6 --io-workers 8

.venv/bin/python -m unittest corpus_pipeline.production.test_production -v

.venv/bin/python -m corpus_pipeline.production.reader --release /absolute/path/to/release
```

同一输出目录只允许一个执行器。源快照、manifest 或来源规则改变须使用新 run；失败模型响应只保留失败审计，不能成为成功缓存。成功缓存包含模型、processor、runtime、precision、prompt、上下文、图像 SHA 和请求预算。服务 URL 改变但这些身份不变时可以复用。

原生 Serving 使用 OpenAI `base_url`、`Authorization: Bearer` 与 registry 指定的 `INF_API_KEY` 环境变量，仍严格检查返回模型名称。传输方式变化不会替换模型推理证据。已有 CPU 检测阶段先核对完成回执、分数 checksum、权重/预处理/运行时和完整资产范围；Kimi 历史复核可在 prompt/model/预算及实际发送图像完全相同的条件下复用，并保留原始推理身份及独立复用回执，不再发起相同付费请求。

需要跨资源排队持续运行时，使用 `python -m corpus_pipeline.production.supervise --manifest ... --output ...`。supervisor 在两个端点都通过部署验证后才启动 canary，等待本身不消耗记录的模型失败重试次数。`supervisor.json` 记录 PID/start_ticks、状态、心跳和子进程日志；只有 transport／暂不可用失败且端点新代际 ready 才自动续跑。schema、数据、质量失败不会无限重试。该进程不申请或更改集群资源。

`--publish-dir` 可指向已有 release store；`--dedup-history` 可指向已有兼容特征库。后者仅复用图像身份／向量，不能继承来源通过状态。跨作业共用时由 SSCD catalog 自身的事务处理并发增量；发布 store 使用单写锁及原子目录提交。

模型请求默认保留可直接发送的原始 PNG／JPEG 字节。需要方向修正、透明底处理或超尺寸缩放时，使用完整图片、白色透明底、无损 PNG，记录发送 hash 和实际变换；不裁剪或生成新的图像内容。过长记录不会偷偷丢掉图片。截断响应受控增加 token 预算，仍不完整就失败。

## 训练读取

原生 Serving 必须有未到期的 `ready_expires_at` 与 `expected_boots`；实际 HTTP 返回的 boot 必须匹配已验收 incarnation。lease 续租不构成新的模型代际。正式入口拒绝顶层 `qualification_only=true` 或 `published_training=false` 的临时资格 registry。Qwen 已验收的 `mm_processor_kwargs.min_pixels/max_pixels=1048576` 随请求发送并进入缓存身份；原图 128 门槛仍先独立执行。

```python
from corpus_pipeline.production.reader import iter_training_records

for record in iter_training_records(release_path, epoch_seed=42):
    segments = record["segments"]  # 保留 native roles/order，叠加已通过的目标修订
    image_bytes = record["images"] # 全部必需图片，原始 SHA256 再验证
```

原生 caption pair 每次按资产＋epoch 从 short/medium/long 选择一个描述，权重为 0.25/0.5/0.25，可显式指定 `caption_length` 做对照。当前另注册了 coco_caption 中实际观察到的两种描述指令，严格核对来源、单图、单轮和指令原文，保留用户指令及原目标语言；其中 terse 指令只允许 short 替换原 assistant，其他长度仅是 sidecar。没有命中已注册指令的 conversation 保持 QA。三种描述不展开成三倍图像记录。上述局部选择不是全库任务混合采样器；跨记录资产曝光上限仍需训练数据调度层统一执行。

## 运行产物与状态

`run.json` 保存不可变输入／规则身份；`attempts/` 保存每次执行的代码、冻结检测器和配置身份，`attempt_records/`、`attempt_summaries/` 保存对应历史，不覆盖旧执行回执。`asset_index.jsonl` 是全部图片检查；`ai_evidence.jsonl` 是来源独立的检测／API 证据；`dedup/` 是真实 SSCD；`model_cache/` 保存规范化生成和复核结果；`record_results/` 为最新逐记录状态；`pair_reviews.jsonl` 保存候选对判断；`releases/` 为已完成读取验证的版本。

以实际 `summary.json` 和 `COMPLETE.json` 为完成依据。`required_stage_acceptance` 明确检查真实生成、独立复核、付费告警复核、SSCD pair 复核、layout 生成且通过、发布读回等分支；仅有全部 terminal 状态不等于 canary 通过。部署还没 ready 时，执行器等待动态端点；这种状态不代表生成、质检或发布成功。测试中的 fixture 也不计入真实模型运行数量。

## 尚存的规模边界

`quality_review_requests.json` 可绑定确切原图及外部审阅回执，触发保留原始模型结果的新生成／独立复核。`admission_overrides.json` 是持久 quarantine-only 决策，按原始记录与图片 SHA 绑定；禁止人工配置把 unknown 自动放行。离线 `python -m corpus_pipeline.production.review --output ... --manifest ...` 仅支持已完成 run 内无 parent 的 canary 候选，重建可见版本后实际读回，不新增模型请求。旧候选、原始模型回执和摘要留存于 `pre_manual_adjudication/`，最终 `manual_adjudication.json` 与 `final_reader_acceptance.json` 明确排除项及发布 ancestry。

当前执行器面向有界完整记录 canary，清单和元数据一次载入；不是数亿条记录的全量调度器。大规模需由共享 inventory 按不可变工作单元调用，复用模型缓存、特征库和 release store；不能直接把全库清单一次塞入本进程。当前只实现 DCVLM 完整记录载入，gen-und 需要将已准备的真实格式适配结果绑定到同一契约后接入。

自动来源判决始终受原规则限制，不为提高发布率放行未知。单次 VLM 的 pair 相似判断也不会直接删图；所有原监督可追溯。质量指标和最终训练收益需要独立校准及对照，canary 的通过数量不能外推到全库。

当前 pair 复核只产出保留／候选共曝光簇的审计建议，尚未将近重 sampling cluster 应用到训练读取权重；`sampling_cluster_applied_to_training=false` 明确记录这一边界。精确同图资产共享已经生效，不同原监督仍然保留。
