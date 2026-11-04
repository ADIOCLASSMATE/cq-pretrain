> 本文件的运行状态和旧入口是历史快照。当前联合处理使用 [统一管道](../README.md)；已完成 DCVLM 的准确结果见 [全量结果](../dedup/README.md)。不要把下文排队、未完成或旧模型池描述当作当前状态。

# DCVLM 离线分片前置处理

本目录从原始 DCVLM 快照重新运行图像检查、AI 检测和 SSCD 特征提取，产出可恢复的筛选索引。当前入口不调用 API、不全量 recaption、不生成 layout JSON、不改写原始文本或任务方向。**产物仍是前置索引，尚未完成全局去重和疑似图片复核，不能直接称为最终训练发布。**

本次运行状态见 [STATUS.md](runs/dcvlm_frontpass_20261022T2129Z/STATUS.md)。旧筛选输出已按清单清理；代码、规则、来源文档、模型权重与原始数据保留。新的模型推理不读取旧检测分数或旧 API 结果。

截至 2026-10-22 22:01:36 UTC，已提交 4 个 LOW 作业，每个申请 16 张 Ascend 910B 和 128 CPU，合计 64 卡、512 CPU；四个 accepted ID 均已确认 QUEUING／LOW、未分配节点，正式队列仍为已处理 0 条，不能把申请资源或提交成功当成正在满速处理。NPU float32 已通过有界数值资格；跨节点 RPC 的 HTTPS GET／POST 已实际通过。独立 CPU 验证已处理 64 条真实记录；它不计入全量队列完成数。24 小时仍是目标，尚无生产持续吞吐证明。

## 输入与处理规则

当前原始父版位于 `/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z`。其发布统计为 119,010,208 条记录、108,026,384 次图像引用、99,856,885 张独立已解码图片。最后一项是原快照的构建统计，不是本轮全库去重结果。

`frontpass.prepare` 只读取快照与分片 catalog 元数据，将原生 Parquet 按最多 8,192 行拆成任务。每个任务保存 Parquet 路径、来源、文件指纹、起始行和不含末行的终点；不会预先把 1.19 亿条记录展开进队列。worker 每次有界读取一小批原记录，保留多图槽位、原生任务类别和原文引用。

1. **来源与完整性。** 明确 AI 生成来源排除。具有依据的程序图表、文档渲染、3D 渲染，以及已有真实父图依据的 AI 编辑按来源保护；编辑父图未知仍待处理。图片关联、原始字节 SHA、解码和尺寸仍须有效。原始短边小于 128 排除，不靠放大通过。
2. **混合网页来源正常检测。** DataComp、ReLAION 等 mixed／unknown 标签只决定进入检测，不再提前隔离。未获得来源保护的图片使用常驻 Community Forensics CF384＋UniFD；原始 `origin` 保留，不将低分改写成“已证实自然图”。
3. **警报不直接删除。** CF384 分数 ≥ 0.90 或 UniFD 分数 ≥ 0.95 时标记 `ai_suspect_pending_api`。本轮 API 关闭，这些图片进入待复核输出。检测失败、完整性异常或编辑父图未确定分别保留 pending 原因。阈值是当前冻结配置，不是概率校准后的 AI 准确率。
4. **SSCD 特征。** 对符合读取条件的图片提取 512 维 float32 特征，保存原图 SHA、像素身份和向量偏移。当前没有执行全局 ANN 检索、近重复组聚合或全库监督去重。同一任务内按 SHA 复用推理，不把跨任务推理次数当作全局唯一图片数。
5. **文本与潜力。** 保留原生 caption、QA、多轮、交错、纯文本及已有文本来源标记。面积达到 1,048,576 像素且短边至少 128 时标记分辨率潜力；layout 按固定来源合同打标签，不调用逐图 API，也不自动选入未来高分训练池。
6. **完整记录决策。** 任一必需图像被排除，记录不能通过；任一必需图像 pending，整条记录保留 pending。没有剩余阻断的记录标记 `frontpass_pass_pending_global_dedup`。纯文本保留原生结构，不运行图像模型。

来源保护需要既有证据；不能只凭图像“像图表”或“像渲染图”授予保护。当前前置 worker 保存各次来源关联，跨任务同图的来源证据与负面证据还需全局汇总。因此前置通过不能替代完整 production 发布验收。

## 文件与职责

| 文件 | 职责 |
|---|---|
| `frontpass.py` | 原生 Parquet 任务计划、有界读入、来源路由、输出前置索引 |
| `inference.py` | 每个进程常驻 CF384、UniFD、SSCD；CPU／NPU 显式设备；解码线程与有界批推理 |
| `queue.py` | 控制主机上的任务状态、租约、心跳、回收、原子完成回执 |
| `queue_rpc.py` | 单控制主机 HTTP 队列服务与带 token 的客户端；远端 worker 只通过 RPC 修改队列 |
| `node_worker.py` | 每个分配的加速器启动独立 worker，写节点／进程日志；正式 NPU 执行需要资格回执 |
| `npu_env.sh` | CANN／驱动路径和离线模型环境，不下载权重 |
| `test_cluster_queue.py` | 多进程唯一领取、抢占后旧 token 拒绝、有限重试、完整与幂等提交 |
| `test_cluster_inference.py` | 推理输入、原图尺寸、预处理及结果边界测试 |

模型目录为 `corpus_pipeline/models/ai_detectors` 与 `corpus_pipeline/models/sscd`。NPU 或降精度执行具有独立模型身份；不能把 CPU float32 的验证直接冒充 NPU／其他精度资格，也不会静默退回 CPU。

## 小任务队列与抢占恢复

**该挂载的跨主机 `flock` 已实测失败，远端 worker 禁止直接调用 `queue.py` 或写入 queue states。** 当前使用控制主机集中式 HTTP 队列服务，经平台官方 Notebook HTTPS proxy 向远端提供带认证 token 的 RPC；worker 通过它领取、心跳、完成、失败及查询，共享盘承载图片引用和输出。`queue.py` 的持久 `.queue.lock` 只在服务所在的同一主机内串行化元数据操作。

服务启动时原子创建 `.queue_rpc_owner/` 确保单实例归属；已有 owner 时拒绝第二个服务，不会根据超时自动抢占 server 所有权。服务崩溃后须先确认原服务已停止，再恢复 owner，避免两个控制主机同时写队列。认证配置 `queue_rpc.json` 权限为 0600，不打印或复制其 token 到日志。客户端不静默重发结果不明的 claim 请求；交付状态不确定时依靠租约恢复和回执核对。

每个任务领取时仍原子创建 `claims/<task_id>/`，每个 attempt 生成新随机 token。任务 catalog 与顺序游标避免每次领取扫描全部待处理任务；过期检查只遍历当前 active claims。失败重试使用独立 retry 标记。

任务结果写入 `artifacts/<task_id>/<token>/`。旧 worker 即使在抢占后恢复，也不能用旧 token 心跳、失败或完成当前任务，且不能覆盖新 attempt 的输出目录。租约过期后任务可重新领取；重试耗尽成为 failed，不算完成。worker 心跳期间必须保持运行；任务重启可能重算最后一个尚未提交的小分片。

完成提交要求：输出文件存在，位于当前 token 的目录，SHA256 与字节数匹配，处理记录数等于整个任务行范围。大文件 hash 在元数据锁外执行，提交前再次检查当前租约和文件身份；最终状态通过原子 JSON 替换提交。已完成任务同一回执可重复提交，统计不会重复增加；缺失或损坏输出不能提交完成。

默认 prepare 设置 900 秒租约、最多 5 次 attempt，并登记 24 小时目标时间。deadline 只是进度目标；超过目标不会把未完成任务标成完成，也不会静默丢弃。全量耗时必须由集群持续有效吞吐、排队和抢占情况决定。

真实 [跨主机验证](runs/dcvlm_frontpass_20261022T2129Z/cross_host_lock_validation.json) 中，本地进程持有锁时 node310 仍成功获得同一 inode 的锁，而本机竞争进程正确被阻止；因此跨主机 flock 不能使用。远端原子 rename、nonce 与 SHA 回读成功。切换 RPC 后，node310 经官方 HTTPS proxy 的 [GET health](runs/dcvlm_frontpass_20261022T2129Z/remote_queue_health_https.json) 和 [POST RPC](runs/dcvlm_frontpass_20261022T2129Z/remote_queue_rpc_post.json) 均已实际成功。

当前队列服务、路由刷新进程和进度监控已后台运行。官方代理路由每 1,200 秒刷新，进度每 120 秒写入 `PROGRESS.json`；认证路由配置以 0600 保存，日志不包含代理 URL 或 token。GPU 探测 Notebook `dev-wjx-ascend` 已停止；队列由 CPU 控制主机维护。代理刷新和服务可用性仍是运行依赖，不能因一次 GET／POST 成功而假定整日网络不会中断。

## 命令与状态解释

以下命令从仓库根目录运行。`prepare` 只执行一次；断点恢复启动 worker，不能重新覆盖同一运行的计划。

```bash
.venv/bin/python -m corpus_pipeline.cluster.frontpass prepare \
  --root corpus_pipeline/cluster/runs/NEW_RUN \
  --rows-per-task 8192

# 控制主机单独运行，保持服务存活；禁止在每个 worker 节点重复启动。
.venv/bin/python -m corpus_pipeline.cluster.queue_rpc serve \
  --root corpus_pipeline/cluster/runs/NEW_RUN --port 18783

# 使用 Inspire SDK Python 在控制主机维护官方 HTTPS 代理路由。
/root/.local/share/uv/tools/inspire-skill/bin/python -m corpus_pipeline.cluster.queue_rpc refresh-route \
  --root corpus_pipeline/cluster/runs/NEW_RUN \
  --notebook cpu-wjx-ascend --loop --interval 1200

# 在另一个进程中运行；worker 通过 RPC 修改队列。
.venv/bin/python -m corpus_pipeline.cluster.frontpass worker \
  --root corpus_pipeline/cluster/runs/NEW_RUN \
  --device cpu --dtype float32 --batch-size 8 --threads 8 --max-tasks 1

.venv/bin/python -m corpus_pipeline.cluster.queue_rpc status \
  --root corpus_pipeline/cluster/runs/NEW_RUN
```

CPU 示例用于有界验证，不表示建议用单机 CPU 执行全库。已完成 NPU 资格验证后，实际调度使用已分配设备，worker 通过共享队列动态领取小任务。

RPC `status` 返回：

| 字段 | 含义 |
|---|---|
| `counts.pending/running/failed/completed` | 任务数量；failed 是不可继续或重试耗尽的任务 |
| `failed_attempts` | 历次失败／过期 attempt 数，不等于最终失败任务数 |
| `processed_records` | 只有已完成回执内的处理记录数；包含通过、排除和待复核记录 |
| `processed_images` | 已完成回执内的图像引用次数，不是唯一图片数或最终准入图数 |
| `elapsed_seconds`、`deadline_at`、`seconds_to_deadline` | 从队列初始化计时及目标剩余时间；负值表示超时 |
| `all_tasks_completed` | 所有任务完成；仍不表示全局去重或最终训练发布完成 |
| `all_tasks_terminal` | 没有 pending／running；可能存在失败，必须同时查看 failed |

`claim()` 返回空值只表示当前没有可领取的任务，其他 worker 可能仍持有全部剩余租约。空闲 worker 应继续等待并带抖动重试；所有任务完成才正常退出，所有任务终止但存在 failed 应明确报告失败。状态检查会回收过期租约，但完整状态聚合要读取已有任务状态；不要让每个空闲 worker 高频扫描全队列。

每个已完成任务的主要输出：

- `records.parquet`：原记录身份、类别、文本来源、图槽和前置状态。
- `images.parquet`：原图引用、来源、尺寸、检测分数、SSCD 偏移和潜力标签。
- `suspects.parquet`：待 API 复核或推理／完整性待处理的图片引用。
- `sscd.f32`：512 维小端 float32 向量；尚未形成全局去重结果。
- `summary.json`：本任务计数、原生类别、实际耗时、模型身份及 `api_calls=0`、`training_release=false`。

正式汇总必须逐任务读取队列中已提交的当前 token 回执，不能递归收集 artifacts 下所有文件；失败或抢占 attempt 的残留文件不可混进有效结果。

当前 cluster 测试共 74 项：CPU 环境运行 68 项通过、6 项因 SDK 环境跳过；这 6 项在 Inspire SDK 环境另行全部通过。独立环境的测试覆盖与实际 NPU／跨节点验收分别记录，测试通过不等于全量生产已完成。

## 可独立准备的纯文本来源

仅按原始 `manifest.statistics.sources` 中**整个来源的 image_occurrences 为 0**选择候选，并与 catalog 行数逐源核对。当前得到 33 个来源、21,961,098 条记录、2,221 个 Parquet 分片，按 8,192 行划分为 4,409 个任务；名单和文件指纹见 [text_only_sources.json](runs/dcvlm_frontpass_20261022T2129Z/text_only_sources.json)。这是全库约 18.45% 的记录，不需要图像模型，可供后续 CPU 专用 worker 提前处理。

该候选范围不包括 allava 等混合来源，即使其中部分记录没有图片。只检查了 manifest 与 catalog 元数据，没有据此声称逐条检查完成。执行 CPU 专用路径时仍须核对实际每条记录无图，保留结构和来源规则；不能仅凭候选名单跳过异常图槽，也不应加载无须使用的图像模型。

实时作业状态由只读监控每 120 秒写入 [JOB_STATUS.json](runs/dcvlm_frontpass_20261022T2129Z/JOB_STATUS.json)，监控自身状态见 [JOB_WATCH_RUNTIME.json](runs/dcvlm_frontpass_20261022T2129Z/JOB_WATCH_RUNTIME.json)。排队资源不足不触发重复提交。

自动轮询空闲 NPU、按合法小规格扩容并恢复抢占实例的工具已运行，详见 [AUTO_SCALING.md](AUTO_SCALING.md)；它接管现有碎片实例并保留原 Job 排队。
