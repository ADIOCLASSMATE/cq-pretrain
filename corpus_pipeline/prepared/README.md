# 固定来源准备与 DCVLM 先导检查

范围固定为 gen-und 与 DCVLM。未来正式 DCVLM-balance-200B 挂载后先验证新快照，不继承当前小池数量。没有任意数据集 API 自动发现入口。

历史准备建议见 [REPORT.md](REPORT.md)。**来源配置、并发样本 preflight、真实 [SSCD](../dedup/README.md) 及 [分阶段完整记录执行器](../PIPELINE_V2.md) 均已接入。** 最新真实 DCVLM 验证与边界见 [重构报告](../REFACTOR_REPORT_20261023.md)：当前默认保留没有确认错误的原文，不确定性标记；仅必要时补写，layout 按来源标记潜力。已完成有界端到端验证，不代表全量生产；候选标签也不等于最终训练准入。

当前用户确定的硬规则为：**所有原图短边至少 128，无领域例外；不能放大后通过。多图记录的全部必需图片都须通过。** 该规则已写入生成 profile、pipeline plan 和完整记录执行器。历史 `preflight` 的尺寸 sweep 仍保留原来的只读统计含义。

**图像来源合同用于路由，不能让混合网页来源绕过检测后直接积压。** 当前 production `image_screening.v3` 将 DataComp、ReLAION 等 mixed／unknown 图片送入 Community Forensics＋UniFD；有效检测无警报按筛选策略准入，警报调用 [Qwen 筛选提示词](../production/image_screening_prompt_v3.txt)。有具体非 AI 证据、中高置信且本次无需外部证据时可准入；高置信 AI 且至少两条具体独立线索时排除；uncertain、编辑父图未确立和模型失败仍待处理。筛选通过不会把来源标签改成 natural，也不认证制作历史。

明确 AI、关联／结构错误和尺寸门继续阻断。已验证的程序渲染等保护来源和真实父图编辑保持保护；同一精确 SHA 可继承所选输入中的保护证据并复用裁决，全部必需图片均须有准入回执。API 排除资产的撤销跨本次批次和合并的 base release 生效。旧前置门下 33 条记录发布 17 条的结果属于历史验证，不能当作新流程的准入率。

| 文件 | 作用 |
|---|---|
| [profiles.py](profiles.py) | 编译原始来源／适配器规则和已有调查，不丢窄路径覆盖 |
| [prepared_profiles.json](config/prepared_profiles.json) | 634 条规则，165 DCVLM 父源、21 gen-und 父源；保留证据及准备状态 |
| [prepared_profiles.csv](config/prepared_profiles.csv) | 可筛选的逐来源／路径摘要；详细证据以 JSON 为准 |
| [genund_source_contracts.json](config/genund_source_contracts.json) | 21 个父源逐项准备：载体、关联、字段、任务、例外、缺口、去重家族 |
| [layout_source_contracts.json](config/layout_source_contracts.json) | 165 个 DCVLM、21 个 gen-und 精确来源的 layout 路由与页面依据；上游标注声明不等于本地已恢复 |
| [staged_recaption_policy.json](config/staged_recaption_policy.json) | 当前已实现的阶段合同：早期保留并标记、必要补写、分辨率潜力与未来所选高分池分离 |
| [pipeline_plan.json](config/pipeline_plan.json) | 目标阶段与模型分工的设计合同，包含原图短边 128 硬规则；不是执行入口 |
| [preflight.py](preflight.py) | 有界线程并发读取原始 TAR，验证 hash／解码尺寸，保留原始监督关联 |
| [test_profiles.py](test_profiles.py)、[test_preflight.py](test_preflight.py) | 来源规则、原生方向、未知处理、哈希错误和多图范围等边界测试 |

在仓库根目录运行：

```bash
.venv/bin/python -m corpus_pipeline.prepared.profiles \
  --output-dir corpus_pipeline/prepared/config

.venv/bin/python -m unittest \
  corpus_pipeline.prepared.test_profiles \
  corpus_pipeline.prepared.test_preflight -v

.venv/bin/python -m corpus_pipeline.prepared.preflight \
  --output corpus_pipeline/runs/prepared_dcvlm_preflight_next \
  --workers 8 \
  --api-decisions corpus_pipeline/runs/api_review_pilot_20261022/final/decisions.jsonl
```

编译器确定性覆盖生成的 JSON/CSV；这是单进程离线准备工具，不是并发线上配置发布器。输入与配置内容有 hash。`preflight` 要求新输出目录，已有目录拒绝覆盖；当前实现是有界样本审计，不支持全库流式调度或中途续跑。原始图片只读，不复制图片，不调用新模型／API，不删除源文件。

`preflight` 默认复用此前的固定视觉样本、关联、检测分数和阈值，可通过 `--samples`、`--associations`、`--scores`、`--thresholds` 指定同一协议的输入。样本关联必须每原记录仅选择一张图，这属于先导抽样协议，不能用来声称全部多图记录已完成检查。尺寸 sweep 不设置生产阈值。

已经完成的实际运行见 [prepared_dcvlm_preflight_20261022](../runs/prepared_dcvlm_preflight_20261022/summary.json)：2,396 个图片资产、2,428 个监督记录、132 个来源。原始 TAR 字节 hash 全部匹配；单轮 I2T 1,958、多轮 I2T 245、交错 225。170 个多图记录尚有 1,038 图槽未检查；没有最终训练发布。

输出文件：

- `asset_index.jsonl`：原始定位、hash／解码、尺寸候选、冻结检测路由和已有 API 证据引用。
- `record_index.jsonl`：原记录定位、顺序 segments、原生任务、资产引用和未检查图槽。
- `summary.json`：资产／记录不同计数口径，按来源和领域统计尺寸影响。
- `receipt.json`、`COMPLETE.json`：输入、代码、产物 hash 及明确的样本审计完成范围。

`match_profiles(registry, collection, source, paths)` 返回所有匹配来源规则；不能只拿最后一个父源标签。`validate_record` 复用现有来源与结构规则，返回 stage1 候选，`release_eligibility=None`。此基础层的 mixed／unknown `quarantine` 只表示待后续图像筛选，不是最终排除；production 仅将 `image_provenance_mixed`、`image_provenance_unknown` 移出前置门并接上模型检测，没有修改根 `policy.py` 的 stage1 语义。`validate_snapshot` 只校验调用方提供的快照绑定回执，并不自行扫描数据；`validation_receipts_present` 也不是准入。

根规则仍以 `../config/sources.json`、`rules.json`、`adapters.json` 为准；本目录增加准备合同，不修改旧冻结 API／检测器试验。生成配置不意味着尚未解决的 caption 配对、来源证明或未来挂载已经可用。`pipeline_plan.json` 是设计合同；完整记录先导的独立运行入口见 `../production/README.md`，当前不支持把此设计 JSON 直接作为运行配置。

分阶段补充策略保存在 [staged_recaption_policy.json](config/staged_recaption_policy.json)，已由 `production.planner`、`staged_run` 和发布读取实现；JSON 记录约束，不作为动态谓词语言执行。当前政策及命令见 [PIPELINE_V2.md](../PIPELINE_V2.md)，[STAGED_RECAPTION_PLAN.md](../STAGED_RECAPTION_PLAN.md) 保留设计背景。

默认 `--audit-fraction 0 --layout source` 不做逐图原文基线审核或 layout API。输出 `annotation_potential.jsonl` 使用实际原图面积 ≥ 1,048,576、短边 ≥ 128 标记潜力；面积阈值可配，`future_highres_pool_selected=false`。186 个来源合同分为候选／不默认做 layout／待来源抽样，全部本地结构绑定尚未验证。未来高分池须另行选择，当前潜力标签不会自动改变图像来源、尺寸、关联或训练准入结论。
