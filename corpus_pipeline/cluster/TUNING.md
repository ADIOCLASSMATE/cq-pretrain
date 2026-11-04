# NPU 前置处理调优与受控发布

适用运行：`runs/dcvlm_frontpass_20261022T2129Z`。本阶段保留原始图文，AI 检测疑似图只标注，SSCD 只提取特征；全局去重和付费 API 复核仍是后续工作。调优不改变上述规则。

**当前运行 `dcvlm_frontpass_20261022T2129Z` 已于 2026-10-23 完成 tuned 验证与滚动发布。** 详细配置、数值比较、生产观测和限制见 [NPU_TUNING_REPORT.md](runs/dcvlm_frontpass_20261022T2129Z/NPU_TUNING_REPORT.md)。其他运行是否启用，仍以其有效 `TUNED_QUALIFIED.json`、实际 worker 身份和 rollout 回执为准；不要跨运行复制资格回执。

## 配置与执行身份

当前已验证的默认配置如下。DataComp、ReLAION vol-1/vol-2 和 AnyWord 四个来源通过 `source_profiles` 覆盖为 batch32、解码4、CPU线程4，其他设置相同。后续调整任何参数都应重新完成与该配置绑定的验证，不能只修改生产回执中的数字。

| 参数 | 当前默认 | 作用 |
| --- | ---: | --- |
| 每设备进程 | 1 | 每个 NPU 一个模型常驻进程 |
| `batch_size` | 64 | 模型批量 |
| `decode_workers` | 16 | 图像读取、解码与预处理线程池 |
| `cpu_threads` | 1 | PyTorch CPU 算子线程，进程级设置 |
| `prefetch_batches` | 2 | 当前批之外最多两个未来批次 |
| `fast_prepare` | `true` | 保持像素和张量一致的准备路径 |
| `target_images` | 1024 | 按新增图像数量聚合记录 |
| `max_records` | 1024 | 单组记录数上限 |
| `max_record_bytes` | 67,108,864 | 单组序列化记录 64 MiB 上限 |
| 模型精度 | strict FP32 | 沿用原模型数学和精度设置 |

预取容量包括当前批，因此聚合目标至少满足 `batch_size * (prefetch_batches + 1)`；以上默认为 `1024 >= 192`。记录字节上限不是进程 RSS 上限；JPEG 解码、PIL 图像、RGBA 哈希字节和模型输入会另外占用内存。显存占用率本身不是优化目标，关注稳定图像吞吐、准备等待时间和输出一致性。

`fast_prepare.py` 仅对不带透明信息、EXIF 方向缺省或为 1 的 RGB 图使用等价捷径，减少全尺寸拷贝与不必要的白底透明合成；其他模式和方向回退原实现。像素 SHA256 分别更新头部和 RGBA 字节，哈希输入字节序列不变。Community Forensics、UniFD 官方预处理以及 SSCD resize / normalize 均保留。

原 `inference.py`、`optimized_inference.py`、`frontpass.py` 三个冻结文件不修改。新增组件分工：

- `tuned_inference.py`：独立设置解码线程、CPU 算子线程和有界预取；一次仅允许一个调用流。
- `tuned_frontpass.py`：沿用原任务处理与租约循环，替换推理 worker 和经验证的聚合配置。
- `tuned_configuration.py`：核验资格回执与真实验证证据；`node_worker.py` 据此选择入口。
- `benchmark_launcher.py`：脱离 Jupyter 终端生命周期的有时限诊断启动器。
- `tuned_rollout.py` / `worker_maintenance.py`：核验本任务 worker 身份后进行更新，不停止 Notebook 资源。

Notebook / `node_worker` 命令中的 `--batch-size 32 --threads 4` 仍是 legacy 入口兼容参数。只有通过 tuned 资格门槛的入口才读取实际新配置。检查 `TUNED_QUALIFIED.json`、`workers/<worker_id>/execution_identity.json` 以及每项产物 `summary.json` 中的 `execution_identity` / `performance.aggregation`，不要用命令行上的旧数字推断实际批量。原入口直接执行仍按其原参数运行。

若回执包含 `source_profiles`，其来源名称对应经过完整验证的配置和聚合覆盖项；未覆盖来源使用默认配置。worker 仅在前一任务推理完全结束后切换，仍复用同一组常驻模型。每项任务保存实际 `source`、`source_profile`、配置和聚合参数，worker 级身份文件表示最近使用的配置。不能因为某个源的小样本较快，就绕过该来源的正式验证直接新增覆盖项。

## 独立基准的运行方法

下面是复测示例，应在已经分配的专用 NPU Notebook 中运行。不要与同卡生产 worker 同时压测；该文档和这些命令不会自动申请平台资源。每次使用新的输出目录，已有目录不能复用。

```bash
export CORPUS_REPO=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain
export CORPUS_RUN="$CORPUS_REPO/corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z"
export CORPUS_TUNING="$CORPUS_RUN/tuning_high_r01"
export CORPUS_RECHECK="$CORPUS_RUN/tuning_recheck_NEW_UNIQUE"
```

初始串行网格包含 batch / decode / CPU / prefetch 组合、交替重复和模型计算上限估计：

```bash
bash "$CORPUS_REPO/corpus_pipeline/cluster/npu_env.sh" \
  -m corpus_pipeline.cluster.benchmark_launcher \
  --directory "$CORPUS_RECHECK/grid_execution" --module tuning_probe --timeout 1800 -- \
  --root "$CORPUS_RUN" --output "$CORPUS_RECHECK/serial_grid" --count 512
```

精细网格使用 JSON 数组，每个元素包含 `batch_size`、`decode_workers`、`cpu_threads`、`prefetch_batches`、`fast_prepare` 五个字段。以下引用本轮已保存的候选数组：

```bash
bash "$CORPUS_REPO/corpus_pipeline/cluster/npu_env.sh" \
  -m corpus_pipeline.cluster.benchmark_launcher \
  --directory "$CORPUS_RECHECK/refine_execution" --module tuning_probe --timeout 1800 -- \
  --root "$CORPUS_RUN" --output "$CORPUS_RECHECK/refined_grid" --count 1024 \
  --configs "$CORPUS_TUNING/refine_configs.json"
```

单卡 1 / 2 / 3 进程对照，每种情景处理相同总图像数。下面重现本轮并发诊断的 `decode_workers=8`，与首轮完整验证候选的 16 不同：

```bash
bash "$CORPUS_REPO/corpus_pipeline/cluster/npu_env.sh" \
  -m corpus_pipeline.cluster.benchmark_launcher \
  --directory "$CORPUS_RECHECK/concurrency_execution" --module concurrency_probe --timeout 1800 -- \
  --root "$CORPUS_RUN" --output "$CORPUS_RECHECK/concurrency" --count 1024 \
  --workers 1 2 3 --batch-size 64 --decode-workers 8 --cpu-threads 1 --prefetch-batches 2 \
  --startup-timeout 300 --run-timeout 300
```

主力源和聚合诊断读取既有完整验证的 baseline，并对 DataComp、两个 ReLAION 子源各取首个 parquet 的前 1024 条新建原 Optimized baseline。其后单个 Tuned worker 常驻，串行 configure、同源预热、完整 `process_task` 计时并比较产物。包含主力源 batch 32 / 64 / 128 / 256，DataComp 额外 batch 512，以及 AS100M、AnyWord、Allava 边界。该工具只提供选型依据，不能替代正式完整验证。

```bash
bash "$CORPUS_REPO/corpus_pipeline/cluster/npu_env.sh" \
  -m corpus_pipeline.cluster.benchmark_launcher \
  --directory "$CORPUS_RECHECK/aggregation_execution" --module aggregation_probe --timeout 900 -- \
  --root "$CORPUS_RUN" --output "$CORPUS_RECHECK/aggregation" \
  --reference-root "$CORPUS_TUNING/frontpass_qualification" \
  --config "$CORPUS_TUNING/selected.json" --timeout-seconds 900
```

正式完整前置处理对照示例；配置 JSON 是单个对象，聚合目标必须与拟发布参数完全一致：

```bash
bash "$CORPUS_REPO/corpus_pipeline/cluster/npu_env.sh" \
  -m corpus_pipeline.cluster.benchmark_launcher \
  --directory "$CORPUS_RECHECK/qualification_execution" --module tuned_frontpass_probe --timeout 1800 -- \
  --root "$CORPUS_RUN" --output "$CORPUS_RECHECK/frontpass_qualification" \
  --config "$CORPUS_TUNING/selected.json" \
  --source-profiles "$CORPUS_TUNING/source_profiles.json" --device npu:0 --target-images 1024
```

各命令应串行执行，前一个结束后再启动下一个。launcher 返回 PID 只表示启动请求完成；必须检查对应启动目录的 `EXIT.json` 为成功且 returncode 为 0，以及探针输出的 `COMPLETE.json`。查看 `PROCESS.log`、`PARTIAL.json` 可以获得中间进度。超时、异常或信号退出会清理其子进程并记录失败；SIGKILL / 平台抢占可能来不及写 EXIT，应核验原进程与平台状态，不盲目重复启动。主机本地锁和子进程父进程死亡保护用于避免重叠诊断，不是跨主机分布式锁。

## 完整验证与发布门槛

基准是 `OptimizedResidentInferenceWorker` 的 batch 32、threads 4、聚合目标 128。候选必须跑真实任务的完整输入、推理、策略判断和产物写出流程，并保持以下条件：

- 记录与图像覆盖、原始绑定、元数据、策略状态完全一致；阈值决策分歧为 0。
- 两个检测器概率最大绝对误差不超过 0.001，SSCD 最小余弦相似度不低于 0.999，结果均为有限值。
- 包含 AS100M 2048 条的两轮相反执行顺序、AnyWord 1024、Allava 8192，以及 chartqa、CLEVR、DocVQA、IAM、COCO caption 的各 16 条边界；主力 DataComp / ReLAION 覆盖要求以正式 probe 和 `EXPECTED_CASES` 为准，主力源诊断本身不算正式覆盖。
- 实际配置、聚合参数、FP32 模型身份、冻结 plan、执行代码和验证代码的哈希与完整证据匹配。CPU 张量 bitwise 对照和单模型吞吐网格均不能单独授权生产切换。

`TUNED_QUALIFIED.json` 绑定实际 configuration、aggregation、每卡进程数、代码哈希、完整验证回执的绝对路径及 SHA256。生成前先通过相同门槛校验，再原子发布；保留旧 `QUALIFIED.json` 和 `OPTIMIZED_QUALIFIED.json` 及其全部证据。资格回执不存在时原路径继续可用；回执存在但格式错误、悬空链接、哈希漂移或验证不足时启动失败，不能静默忽略。

有界样本上的一致性只说明这次优化没有改变已测样本的处理结果；它不证明 AI 判别准确率，也不直接证明全库一天完成。全库估算应按源图像量与该源持续吞吐计算，并考虑分片、存储和抢占。

## 已验证后的 worker 更新

仅在有效新资格回执发布后，从原队列控制主机运行：

```bash
/root/.local/share/uv/tools/inspire-skill/bin/python \
  -m corpus_pipeline.cluster.tuned_rollout \
  --root "$CORPUS_RUN" --output "$CORPUS_RUN/tuned_rollout_NEW_UNIQUE"
```

该入口只处理本次初始 accepted 清单和动态 allocator 清单中的 LOW Notebook。第一阶段先按精确 ID 复核资源合同与 RUNNING 状态，再核验 supervisor、node worker 和各 NPU 子进程的 PID / 启动时间 / 完整命令。收齐全部选定旧 worker 的退出证据并写入 `STOPS_VERIFIED.json` 后，第二阶段才对仍属于相同 worker 和 token 的当前活租约执行可重试失败；过期或已被其他 worker 接管的租约跳过。第一阶段任一停止失败时不主动重排或启动，以免刚重排的任务再次被其他旧 worker 领取和中断。

随后在同一个 Notebook 幂等启动替代 worker。控制器如果抢先启动了替代 worker，launch 应返回已有实例，不重复开进程。检查 rollout 的逐实例 STOPPED、ROLLOUT、COMPLETE 和新 worker 身份；不操作高优基准 Notebook、无关 Job、Serving 或用户的其他实例。更新不删除任何已完成数据；抢占按原队列的租约机制恢复。

## 回滚

删除新回执只改变未来进程的入口选择，**不会**改变正在运行的 tuned worker。回滚应作为一组有记录的操作执行：

1. 在原控制主机核验当前控制器进程身份并优雅停止，等待退出，防止维护过程中自动重启 worker。不要把 `--no-auto-scale` 当作 worker 暂停开关，它仅禁止新增资源。保留队列 RPC、代理刷新和纯文本等独立服务。
2. 验证原 `QUALIFIED.json`、`OPTIMIZED_QUALIFIED.json` 及冻结代码仍有效；将 `TUNED_QUALIFIED.json` 原子移到唯一审计路径保存，不能留下同名悬空链接。不删除生产结果、源数据、队列状态或旧资格证据。
3. 对本任务精确 owned Notebook 调用 `worker_maintenance`，核验并停止同一组已归属 worker；停止成功后再启动原路径。不得以模糊进程名批量 kill，也不停止 Notebook 资源。
4. 原 worker 的未完成工作可等待租约过期后恢复。若要立即重试，必须沿用 `requeue_stopped_leases` 的停止证据、控制主机、活租约和 token 校验，禁止直接改状态文件。重新排队仍受原任务重试预算限制。
5. 在同一 Notebook 使用原 `notebook_worker launch` 幂等入口恢复，核验实际模块回到 `frontpass`、原资格配置有效后，再按原控制器启动记录恢复自动管理。

在已确认归属的目标 Notebook 内，维护入口形式如下；`--name`、设备数及 Notebook ID 必须来自 accepted 合同，远程调用也需使用其精确 ID：

```bash
bash "$CORPUS_REPO/corpus_pipeline/cluster/npu_env.sh" \
  -m corpus_pipeline.cluster.worker_maintenance \
  --root "$CORPUS_RUN" --name OWNED_NOTEBOOK --output "$CORPUS_RUN/rollback_NEW_UNIQUE/OWNED_NOTEBOOK"

bash "$CORPUS_REPO/corpus_pipeline/cluster/npu_env.sh" \
  -m corpus_pipeline.cluster.notebook_worker launch \
  --root "$CORPUS_RUN" --name OWNED_NOTEBOOK --devices ACCEPTED_DEVICE_COUNT
```

`tuned_rollout` 要求有效 tuned 回执，不能在移除回执后直接把它当作回滚命令。保持控制器冻结直到已核验的替代 worker 和租约状态恢复正常；所有异常停止点先查回执再继续。
