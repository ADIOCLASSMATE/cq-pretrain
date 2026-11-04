# 原生 Serving 预算控制器

`budget_pool.py` 管理固定大小的原生 Serving 副本。模型加载、模型身份、每次启动后的就绪判断由模型 backend 实现；控制器负责卡数预约、并行观察、抢占恢复、服务发现和停止。它不处理数据，也不使用训练 Job / Notebook 承载模型。

配置 `corpus.native_budget_pool.v1` 指定绝对状态目录、backend Python 文件、构造参数、每副本卡数、总卡数上限、LOW 优先级、最大副本数和每副本请求上限。`external_reservations` 可将独立管理的调试 HIGH 实例计入同一预算，但不把它的停止权转交给控制器。该资源仅在其所有者已禁用恢复、保存释放状态，且 backend 再次从平台核实释放后，才不占预算。

待调度、被抢占、响应丢失和暂时无法观察的副本均占预约卡数。Create 前先落盘预约账本及意图；结果未知时仅查询恢复。抢占只重启原 Serving ID，不额外申请替代卡。失去账本、配置身份变化或资源归属不符时拒绝新增申请。停止标记先写入，停止和重启经同一副本锁串行执行。精确节点放置由原生 Serving 平台决定，控制器不虚构指定节点能力。

通用入口（使用安装了模型 backend 所需 SDK 的 Python）：

```text
python -m corpus_pipeline.serving.budget_pool guard --config /absolute/pool/config.json
python -m corpus_pipeline.serving.budget_pool status --config /absolute/pool/config.json
python -m corpus_pipeline.serving.budget_pool stop --config /absolute/pool/config.json
python -m corpus_pipeline.serving.budget_pool retire --config /absolute/pool/config.json --slot 20
```

`guard` 维持单个管理进程；PID、启动时间和完整 argv 都匹配时才视为同一进程。管理进程每个副本使用独立观察线程。NPU 实例被抢占后，CPU 请求端应根据到期租约撤掉流量，并幂等重试未完成的样本。

`stop` 会发送本池的停止请求并观察一次；管理进程继续观察，直到 `STOPPED.json.complete=true`。停止控制进程本身不会释放平台实例。外部 HIGH 实例需由它自己的控制器停止。状态目录和意图文件不可删除后重建，否则会丢失资源账本。

`pool.json` 遵循 `corpus.screening_pool.v1`，列出动态 registry 路径及 `max_inflight`。registry 的端点租约、当前 boot 身份和图像支持由 backend 提供；数据工作端负责总请求并发和供图。模型特有配置不进入通用调度模块。

当前 Qwen3.6 W8A8 实例配置在 `qwen36_w8a8_20261024/pool/config.json`：每副本 2 卡、总上限 128 卡、最多 64 个 LOW。首个 HIGH 两卡仍运行时，LOW 最多 63 个。推理直接采用 `official_v023/CONTRACT.json` 固定的官方命令；本控制器位于模型实例之外。

永久停用的已知副本可用 `retire` 回收其预算预约。此命令不发送 Stop、Start 或 Create：必须已有本槽 DISABLED 和完整释放记录，再由已审计 adapter 现场核验同一服务 ID 的 STOPPED 及实例终态。请求、释放快照、原预约绑定及退休证明保存在本槽 retirement 目录，提交到完整历史账本之后才免计。普通 STOPPED、抢占、未知申请或单独写一个 retired 标志均不会释放预算。

补齐目标按未退休预约数计算，新增编号始终取历史最大编号加一，不重用旧身份。例如原有64槽中20号永久退休后，可以新增65号，仍是64个活跃预约、128卡。已发送未知申请优先占用目标；TARGET降低时不会因为晚编号已运行而错误恢复更多早编号未发送申请。STATUS分别展示历史、退休和活跃预约；外部资源另作实时预算核验。最终清理和watchdog仍覆盖全部历史槽，包括退休槽，不删除原始账本。

`REQUEST_LIMIT.json` 的 `max_inflight` 可动态设置每副本请求准入，范围为 1 到配置中的 `per_replica_max_inflight`。当前配置上限为 128；请求准入上限与 NPU 卡数预算分别管理。CPU 全局请求准入由工作链独立限制，实际在途数不会因为副本增加而越过该全局上限。调整请求准入不会重启模型，也不会改变推理参数或标注缓存身份。

`TARGET.json` 的 `replicas: 0` 仅暂停新建及未发送预约的恢复，现有服务和观察继续工作。已审计的官方 adapter 可以恢复发送前失败的原预约：必须仍无 service ID、无 `CREATE_TRANSMISSION.json`，并在原 slot 锁及 provision 锁内重新通过资源身份、精确同名查询、预算和禁用检查。其他 adapter 默认不启用此恢复；有发送标记的未知申请永远不按此路径重发。

管理接口的真实 `TransportError` 可以消耗已有且尚未到期的就绪租约，但不能延长租约；已知停止、身份变化、绑定错误和过期租约仍撤销就绪。这样单次管理查询失败不会立刻丢弃本来仍有效的在途响应。错误类型、阶段、时间保存在 `pool_last_error`，观察恢复后也保留。

管理观察最多 4 个并发，开始间隔至少 0.35 秒；控制器重载后先恢复已有副本的观察，再尝试新建。64 个副本的理论启动间隔合计为 22.4 秒，实际周期还受 SDK 延迟影响，不能据此保证所有租约始终有效。请求错误的 `cause_chain` 仅保存异常类型和数值状态码，不保存 URL、凭证或异常全文。

管理接口明确返回 HTTP 429 时，`PROVISION_BACKOFF.json` 持久记录全池扩容退避：首次 60–72 秒，后续指数增长至最多 15 分钟。同一冷却期的并发错误不会反复加倍；已有服务继续观察，不以成功续租清除扩容冷却。新建、未发送预约恢复、抢占重启都在实际操作前复核冷却，原预算和身份检查照常执行。已达到目标或 TARGET=0 时，不做仅为新分配所需的外部预算查询。

停止过程由 `cleanup.lock` 串行协调不同控制进程，每轮最多 4 个清理调用、开始间隔至少 0.35 秒。已审计 adapter 的预约若从未写入 Create 发送标记、没有服务 ID，且在禁用及生命周期锁内核实精确同名资源不存在，可以写 `CANCELLED_NEVER_SENT.json` 并结束该预约；保留账本及原始意图哈希，不声称平台曾创建或停止该服务。已发送或身份未知的申请仍须正常查询确认。CLI `stop` 的退出码不能替代 `STOPPED.json.complete=true` 及逐项释放证明。

验证：`python -m unittest corpus_pipeline.serving.test_budget_pool` 覆盖未知创建响应、外部 HIGH 预算、抢占复用、停止竞态、账本丢失、身份冲突和部分清理失败。模型 backend 的运行验证单独记录在对应 scope。

`test_budget_retirement` 另验证退休后的补齐、slot65恢复、证据篡改、错误重新启用、提交前崩溃，以及现有watchdog对完整历史清理的校验。
