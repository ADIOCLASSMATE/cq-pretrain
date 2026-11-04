# Pipeline 重构记录（2026-11-04）

新任务统一使用 `python -m corpus_pipeline`。把本轮运行目录中验证过的修复收回维护代码，复用原有阶段执行器与资源管理器；没有新建插件框架或通用调度系统。

| 已出现的问题 | 维护代码中的处理 | 验证 |
|---|---|---|
| 修复只留在某轮 ops，新任务重新踩坑 | `orchestration/prepare.py` 冻结当前维护代码，生成统一入口和模板；不复制旧运行目录 | 全新运行目录、冻结文件篡改、真实 worker 导入测试 |
| 冻结包漏掉间接依赖 | 纳入 detector_calibration 与 api_benchmark，排除旧 fragment 调度入口 | 静态包内导入检查；从冻结目录实际导入 CPU/NPU worker 模块 |
| 控制端允许 512 卡，远端仍拒绝超过 128 的预算 | `unified/fleet.py` 统一上限检查；预算包含排队与未知创建 | 512 卡配置通过远端 preflight；超限拒绝 |
| 整节点排队、远端终端查询阻塞 | `orchestration/fleet.py` 固定 LOW 单卡，优先已绑定共享心跳；未知结果保留预约 | 单卡规格、提交节流、抢占、未知创建与终端失败测试 |
| CPU 完成后空转、被守护程序重新启动 | `orchestration/cpu.py` 按已封存回执持久退休，保留后续 review CPU | 退休不可逆、审查 CPU 保留、未知启动复用 operation ID |
| GPFS 上只用 flock 不能充分协调跨节点 | `coordination.py` 使用线程锁、flock 和 POSIX 记录锁；ingest 原子 mkdir 领取 | 线程/进程互斥与并行领取测试 |
| HTTP 接收开销高、计时混淆 | `production/http_transport.py` 直接使用 aiohttp 接收完整原始响应；HTTPX 继续序列化及验证 | 本地真实 HTTP/SSE、压缩、超时和错误处理测试 |
| 单独 canary 成功但正式进程持续失败 | `postprocess/review_health.py` 同进程真实图片预检；滚动零有效返回中断 | 预检失败不得消费全队列；历史成功不能遮盖新失败 |
| Workspace 枚举接口变更，SDK 查不到资源 | `serving/platform_client.py` 使用实时 ListWorkspaces，保留工作空间和公平调度字段 | 分页/缺页测试，真实 CPU 名称与 ID 查询成功 |
| Notebook 网关连接过早超时 | `cluster/notebook_bound_exec.py` 仅在当前连接上延长官方网关连接时间 | 限定域名、不重试发送、错误 workspace 拒绝；真实 CPU 终端执行成功 |
| 服务历史查询和排队轮询挤占运行服务观察 | 已知服务 ID 直查；运行/排队独立观察通道；预算无需逐次查物理实例 | 归属校验、观察并发及等待服务不阻塞运行服务测试 |
| 停止回执丢失导致资源长期占用 | `resource_manager/engine.py` 重新观察后仅对绑定的 Stop 重试；不盲目重试 Create | 未确认停止、释放确认和无双重创建测试 |
| 每轮重复串行读取大产物 | `unified/workflow.py` 有界并行验证与进程内文件身份缓存 | 内容篡改、冲突绑定仍拒绝 |
| 任务方向、去重与最终完成容易混淆 | `unified/exact_release.py` 与 `baseline_union.py` 保持原监督和权限，合并回执决定最终完成 | I2T 无图像 loss、不同 caption 保留、未决基础设施失败不能发布 |

## 验证结果

- 核心回归：**685 项通过，183 个子测试通过**；最后一次完整输出保存在本次 validation 目录。
- 最终入口、冻结包、平台及阶段回归：41 项、8 个子测试通过；随后新增审查恢复锁冲突用例，入口 8 项测试全部通过。
- 使用微型真实图片完成七个执行来源的注册、并行归一化、封存及 native 采用，断言最终发行不会提前出现；不同来源在此使用相同微型载体验证调度覆盖，不能当作真实六子集的数据量验收。
- 正式 SDK 查询、CPU 终端执行及真实图片模型推理通过；探针确认无缓存、正式队列不变。回执见 `validation/pipeline_refactor_20261104/REAL_IMAGE_PROBE.json`。

## 入口与旧代码

9 个已经完成用途的一次性启动/迁移脚本已从 `scripts/` 移到 `validation/retired_entrypoints_20261104/`，扩展名改为 `.disabled`。原路径与 SHA256 见该目录 `MANIFEST.json`。旧默认 `process` 入口已禁止，新快照不包含旧 fragment 调度、旧模型池或 staged_run 入口。

当前生产运行仍使用自己的冻结代码、账本和审查队列。验收时发现控制 Notebook 已重建，旧 host 的控制心跳中断；核验相同 Notebook ID、平台 uptime、控制锁及前置 144 个资源全部退休后，只重绑定控制 host 并恢复原版本控制器和模型池。同时修复审查恢复重复争抢 workflow 锁的问题：复用已封存且绑定的对账与 API selection，仅恢复模型池守护。维护入口对应测试通过，运行层补丁及旧 SHA 备份在 `ops/controller_host_recovery_20261104/`；冻结模型和数据 worker 未改。原始数据、模型权重和既有 DCVLM 发行未移动或重写。

## 边界

这次验证没有重新跑全量语料、申请新的大规模 NPU 池或完成 512 卡并发压测；不能据此声称端到端时限或 NPU 利用率已经达标。模型配方、精度和筛选门槛保持当前已验证约定。新任务仍需提供已核验来源计划、已登记 CPU Notebook 以及当前模型资格；训练 Catalog 导入联合发行是后续独立工作。
