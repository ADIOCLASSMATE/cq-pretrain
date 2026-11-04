# 数据预处理 pipeline

新任务统一从 `python -m corpus_pipeline` 启动。维护代码在 `orchestration/`、`unified/`、`production/`、`resource_manager/` 和 `serving/`；运行目录只保存配置、冻结代码和产物，不再从上一轮 `ops/` 拷贝补丁。

当前入口处理 **GenUND 六个指定子集及 DenseFusion 补充分支，与已有 DCVLM-baseline 发行合并**。它不表示支持整个 GenUND 的所有来源，也不重新下载 DCVLM。

| 阶段 | 行为 |
|---|---|
| 准备与验收 | 冻结维护代码，注册每个来源的已核验计划，绑定外置挂载映射、模型资格和 CPU 资源 ID。 |
| CPU 归一化 | 并行读取原监督与图片引用，原子领取工作单元，完成后封存 native 索引；完成的 CPU 不再被守护进程拉起。 |
| NPU 前置处理 | LOW 单卡实例，分片租约队列；检查绑定、图片字节和任务方向，仅对 T2I 执行 AI 检测。 |
| 对账与审查 | 汇总精确图像身份；释放前置卡后预热 LOW TP2 审查服务，并行完成 CPU 对账。审查进程必须先用同一客户端成功请求一张真实图片。 |
| 去重与发行 | 保留不同 caption、问答及有序交错监督；精确重复折叠，图像资产只保留一份索引。与既有 baseline 合并后验证所有产物。 |

本次 profile 固定使用精确去重，不执行 SSCD 全对相似搜索；保留已有特征。最终完成只认运行根目录 `COMPLETE.json` 及其绑定的 `joint_release/COMPLETE.json`，实例 RUNNING、任务数完成或单源完成都不能替代。

## 数据规则

- I2T 跳过 AI 来源筛选、检测和视觉审查，统一 `image_loss_allowed=false`；保留图像输入和文本 loss。引用损坏、错配仍不能进入训练。
- ScaleCap、DenseFusion、BLIP3o Long Caption 同时保留 I2T/T2I。已知 AI 来源直接排除 T2I；其余 T2I 按检测与复核结果筛选，完成复核后仍不确定的排除。基础设施失败留待恢复，不能冒充审查结果。
- OpenUni 按原生 T2I 解析；PixMo 和 FineVision 保持问答、对话原结构。不同 caption 均保留，不把多个 caption 拼成假交错。
- 精确去重同时比较图像身份、有序监督、原生格式、任务方向和 loss 权限。同图不同监督仍是不同样本。
- 原图与旧发行不改写。图片引用采用固定 dataset ID 和相对路径／TAR 范围；文件夹原样复制后仅调整外置 `source_roots.json`。

完整字段与训练读取约束见 [TASK_POLICY.md](unified/TASK_POLICY.md)。

## 启动与恢复

使用 [配置模板](orchestration/examples/genund.json) 和 [操作说明](unified/OPERATIONS.md)：

```bash
.venv/bin/python -m corpus_pipeline prepare --config /ABS/request.json
.venv/bin/python -m corpus_pipeline register-sources --config /ABS/run/ops/prepared_config.json
.venv/bin/python -m corpus_pipeline preflight --config /ABS/run/ops/prepared_config.json
.venv/bin/python -m corpus_pipeline run --config /ABS/run/ops/prepared_config.json
.venv/bin/python -m corpus_pipeline status --config /ABS/run/ops/prepared_config.json
```

`prepare`、`register-sources`、`preflight` 不申请资源；`run` 管理已登记的 CPU 并按阶段申请 NPU。CPU Notebook 及来源计划须事先完成准备，不能把缺标注、缺图或未知绑定当作已完成计划。

NPU 上限显式填写，当前授权最高 **512 卡**，包含排队和未确认申请；前置实例固定 1 卡，审查固定 2 卡。NPU 为 LOW 1，CPU 可为 HIGH 6。使用当前已验证的 Qwen3.6-35B-A3B W8A8 / vLLM-Ascend v0.23.0rc1 配方；卡数上限不是可获得卡数或完成时间保证。

新任务运行冻结副本，仓库后续修改不会热替换正在运行的代码。旧任务继续使用自己的绑定入口；不把旧运行目录交给 `prepare`。`joint` 仅保留为底层阶段工具，旧 `process` 已禁止作为默认入口。9 个一次性启动／迁移脚本已移出 `scripts/`，追溯见 [退役清单](validation/retired_entrypoints_20261104/MANIFEST.json)。

修复与验证范围见 [重构报告](orchestration/REFACTOR_REPORT.md)。
