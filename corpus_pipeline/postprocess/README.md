> 本文件的运行状态和旧入口是历史快照。当前联合处理使用 [统一管道](../README.md)；已完成 DCVLM 的准确结果见 [全量结果](../dedup/README.md)。不要把下文排队、未完成或旧模型池描述当作当前状态。

# DCVLM frontpass 后处理

输入是已封存的 `FRONTPASS_COMPLETE.json` / `frontpass_index.json`。不重跑冻结检测器，不修改原始数据或问答方向。

1. `reconcile.py` 校验最终任务回执，一次分区后按图片 SHA 汇集所有来源证据。跨来源已知非 AI 保护仅作用于同一 SHA，不从相似图传播。损坏引用、像素冲突和推理异常单独保留。输出 256 个桶的资产索引、API 候选和异常，支持中断续跑。
2. `api_screen.py --mode annotate` 使用原生 Serving 池做盲视觉复核。原始检测器疑似状态不因模型意见而解除：顶层仍为 `pending`、`detector_alarm_cleared=false`。完整意见、模型建议、图片 SHA、提示词 SHA、模型版本、运行 boot 和响应回执保存在 sidecar。
3. `screen_supervisor.py --config <absolute-config>` 等待全量来源对账、分配请求、识别抢占后的暂时不可用、恢复同一 SQLite 任务。完成所有标注后执行配置中的一次性清理命令。标注完成与来源确认完成是两个不同状态。
4. `sscd_index.py` 提供 catalog/train/build/merge/calibrate/search，复用已有 FP32 特征，以共享 IVF-PQ 全局检索、原始 FP32 精确重排，仅导出候选边。`sscd_supervisor.py` 串联完整阶段，绑定冻结源码、配置与筛选快照，保存后台状态和可恢复的阶段回执。全量索引仍必须独立完成全库召回校准，不能复用有界试验资格。

## 为什么正式使用 annotate

2026-10-24 的真实对照中，5 张有仓库来源声明和 EXIF 支持的 DALL·E 2 示例，Qwen397B 的 v4 提示词漏检 4 张，付费 qwen3.8-max 也漏检相同 4 张；原 v3 仍漏检 3 张。这个小集合不是全库准确率估计，但证明“两个视觉模型都认为非 AI”不足以解除检测器报警。原 `primary` / `both` / `cascade` 模式保留用于隔离实验，当前全量配置明确使用 `annotate`。

示例中透明掩码被误当作 AI 结构异常，进一步说明实物模型、传统绘画、图表、CAD、透明背景与生成式 AI 不能仅按外观划分。v4 用于减少无根据的强断言；它不是经过召回率校准的 AI 真伪检测器。后续最终过滤仍需可靠来源证据、带标签的领域校准以及未决样本复核。

## 当前运行入口

运行目录：`../cluster/runs/dcvlm_frontpass_20261022T2129Z/postprocess_20261024/`。

- `screen_annotations_config.json`：正式全量标注配置，最多 512 并发请求。
- `ai_annotations_v4/supervisor/STATE.json`：CPU supervisor 状态。
- `ai_annotations_v4/{PROGRESS,SUMMARY,COMPLETE}.json`：处理进度与阶段完成状态。
- `reconciliation/PROGRESS.json`：图片与来源汇总进度。
- `../serving/ai_review_20261024/STATUS.json`：模型池状态，最多 4 × 32 卡，包含未就绪预约。
- `serving_canary/`：DCVLM 真实图片、已知 AI 正例、付费二审及对照实验。

使用仓库 `.venv/bin/python`，不要对解释器路径调用 `Path.resolve()`，否则可能绕过 venv。CPU supervisor 与模型池 guardian 独立运行；模型池重启后必须重新取得当前 boot 的图像资格。账号私钥只从受限文件注入子进程环境，不写入回执。

停止全量标注：创建 `ai_annotations_v4/supervisor/STOP`；先停止取新任务再等待在途调用落盘。停止本池模型资源：使用 Inspire SDK 解释器执行 `python -m corpus_pipeline.serving.ai_review_pool stop`。该命令只操作持久清单绑定的服务，并与自动恢复通过跨进程文件锁互斥。

所有输出仍为 `published_training=false`。没有执行全量 recaption、生成 layout JSON 或发布最终训练集。
