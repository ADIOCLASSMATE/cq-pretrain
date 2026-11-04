# DCVLM 未决图像并发复核

此模块用真实 Codex CLI 图像附件复核上一轮 Qwen 标注中的未决项和技术失败项。它新增筛选结果侧表，不覆盖原始数据或上一轮判断。

本次范围是 70,492 个唯一图像：70,462 个模型未决项和 30 个技术失败项。运行根目录为：

`corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z/postprocess_20261024/codex_uncertain_20261024/`

## 判断口径

- `admit`：通过当前视觉筛查，不代表已证明来源真实。
- `reject / ai_generated`：高置信生成判断，至少两处独立可见异常。
- `reject / quality_unusable`：主体或主要文字因严重模糊、遮挡等失去训练价值；与 AI 排除分别统计。
- `pending`：仍需外部证据或人工判断，不强迫二分类。
- `failed`：技术处理失败，不能伪装成模型不确定或坏质量。

常规图表、文档、手写、绘画、传统渲染与合成图不因风格而排除。普通压缩、老照片、可读的低分辨率图可以保留。AI 编辑的真实图像维持原来源例外规则；AI caption 不触发图像排除。

## 执行

`inputs.py` 冻结只读输入，按 TAR 原始字节 SHA 校验并完整解码。所有图像 EXIF 归正、必要时等比缩放，不裁剪、不以文件名代替看图。原字节、解码、附件、命令、prompt/schema、原始回复、events、线程 ID 和 usage 均保留。

每次调用是独立 ephemeral Codex 线程，显式固定 `gpt-5.6-sol` 与 `low` reasoning。每批默认 8 张，CLI 并发从 4 逐步增加到硬上限 64；遇到相关传输失败会熔断，随后单探针恢复。部分成功逐项提交，只重试未成功项。SQLite WAL 和独占锁保证一个写入者；收到 SIGTERM 停止补充任务并收完在途结果。

`supervise.py` 负责后台运行、异常退出重启、精确身份接管已有 farm，以及完成后的严格导出和报告。它不创建 NPU 部署。手动运行状态查询：

```bash
.venv/bin/python -m corpus_pipeline.codex_review.farm status --root <运行根目录>
```

恢复前确认现有 supervisor 状态；不要重复启动。普通恢复入口：

```bash
.venv/bin/python -m corpus_pipeline.codex_review.supervise \
  --root <运行根目录> --workers 64 --initial-workers 4 --batch-size 8
```

2026-10-25 对主批次耗尽五次重试的最后 3 条 HTTP 429 失败进行了单独恢复。
`recover_transport.py` 核验原始 events 中的限流证据，归档旧 item/batch 行及部分导出回执，
在独占锁和事务内为指定条目各授权一次额外尝试；不重置累计 attempts，不重跑成功项。
恢复累计最多 3 条，每个 ID 仅一次；内容拒绝、图像解码失败不能通过此入口重排。
随后用单并发 supervisor 补跑，并重新执行完整严格审计与导出。
授权、代码哈希与历史快照位于运行目录 `recovery/http429_once_20261025/`。

## 产物与完成定义

- `candidates.jsonl` / `candidates.manifest.json`：唯一输入和固定范围。
- `state.sqlite`、`HEARTBEAT.json`、`SUPERVISOR_STATUS.json`：进度和恢复状态。
- `batches/`：逐次推理、解码附件和原始证据，保留失败历史。
- `exports/<版本>/`：原子 gzip 分片、manifest，以及不可变严格审计。
- `final_report.json` / `.md`：AI 排除、质量排除、保留、视觉未决、技术失败分别计数。
- `screening_overlay.jsonl.gz`：按 asset_id 连接到原数据的筛选侧表。
- `manual_review.jsonl.gz`：模型仍无法确定的项。
- `technical_failures.jsonl.gz`：技术失败或未完成项，保持状态区分。
- `raw_provenance.jsonl.gz`：原始推理证据索引。

只有全部候选成功完成真实推理、逐项严格审计和分片核验之后才生成完整 `COMPLETE.json`。视觉结论仍为 pending 的有效结果会进入人工队列；它们不属于技术失败。部分导出明确记录 `complete=false`，不声称全量完成。新结果不自动认证来源或发布训练集。

本次上线前进行了 8 张真实图像的两批验证、在途 SIGTERM 排空与恢复、严格导出和报告验证；另有生产附件视觉抽查，记录在运行目录的 `spotchecks/`。这些小样验证流程和判断合理性，不构成准确率评测。

## 导入人工复核

`human_import.py` 导入离线复核页面导出的 JSON。先验证队列 SHA、图像 ID 与编号映射、严格模型审计和提交格式，再生成不可变的独立结果层。只应用用户明确提交的决定；遗漏项与 `unsure` 继续 pending。质量排除与 AI 排除分别统计。

```bash
.venv/bin/python -m corpus_pipeline.codex_review.human_import \
  --root <运行根目录> --submission <用户导出的JSON路径>
```

产物位于 `human_reviews/<提交SHA256>/`，包含原始提交、全量合并侧表、人工作用层、剩余待复核队列、报告和审计收据。`HUMAN_REVIEW_CURRENT.json` 在全量核验后原子发布，指向当前版本。消费方应读取此指针及 `resolved_screening_overlay.jsonl.gz` 的顶层决定；`model_screening` 保留原模型意见，不能用它覆盖人工决定。原模型数据库、报告、页面与图像保持不变。同一提交可幂等重跑；不同提交不会静默替换已有版本。

用户在会话中明确补判时，保留原提交，生成包含完整原队列的修订提交，并通过 `--supersedes <当前提交SHA>` 与 `--correction-evidence <补判证据JSON>` 导入。证据记录原话、图像 ID、编号和前后决定；核验器要求未补判项逐字段不变，拒绝删项、混入未确认变更或错用父版本。新版本的 `REPORT.json.correction` 连接父版本与证据，历史版本仍保留。

本模块的 `COMPLETE.json` 只表示人工结果导入完成；是否仍有待复核项见 `all_reviews_resolved`，不等于最终训练集已发布。
