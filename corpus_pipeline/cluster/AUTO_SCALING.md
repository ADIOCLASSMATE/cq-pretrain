# NPU 碎片用卡上限与自动补齐

运行记录：`dcvlm_frontpass_20261022T2129Z` 已于 2026-10-23 20:41:50（北京时间）完成全量离线前置处理。2026-10-24 平台回查本轮 63 个受管历史 Notebook 均 STOPPED，控制器已退出、当前占用 0 卡。下文为本轮执行策略及复用说明。AI 检测只标注疑似图，付费模型 API 关闭；保留原始数据与文本，前置结果不是最终训练发布。

## 用户当前资源限制

2026-10-23：用户更新资源策略，当前管道上限为 **48 卡**，轮询发现资源并在上限内补齐。控制器使用 `--auto-scale --max-cards 48`；自动申请必须显式给出 `--max-cards`，不再允许省略上限启动扩容。`--no-auto-scale` 仍可用于暂停新申请、维护已有实例。

计入上限的是当前运行保留的全部卡数：已接收的运行、排队、重启中或状态暂不明的实例，加创建结果未确认的申请。按实例 ID 去重；只有绑定到精确实例 ID 的停止确认才能释放已接收实例的额度。明确拒绝的创建申请释放额度，超时或响应不明不会触发重复申请。账本跨控制器重启保留。

48 卡是申请上限和补齐目标，不保证平台始终有 48 卡可用。抢占后优先恢复原实例；排队超时则撤回，确认停止后释放额度再从其他候选节点补齐。规格按剩余额度选择，剩 3 卡时不会申请 4/8/16 卡。若后续把上限降到已保留卡数以下，暂停新申请与实例重启，不自动强停正在工作的实例。

## 集中节点与合并

新申请使用独立的 `corpus_pipeline.resource_manager.planning` 模块：优先已有节点，同时核算 NPU、CPU、内存与合法配额。节点剩余 7 卡时，可以在同一轮按 4＋2＋1 连续提交；每次提交后重新读取资源和预算。已接收但尚未运行的申请只预占对应三种资源，不封锁整个节点；平台已经计入 RUNNING 的资源不会重复扣除。资源不足时继续选择其他节点。

默认启用 `--consolidate`。已占用节点出现足够空闲 GPU、CPU、内存时，规划一个能完整腾空至少一个本管道来源节点的迁移。目标可以是一个或多个合法规格实例，例如将 6 卡迁为 4＋2，将 7 卡迁为 4＋2＋1。仅选择身份已确认、近期观察为 RUNNING、无其他待处理操作的本管道 LOW 实例。规划器使用有界搜索，不宣称全局最优。

一次只执行一个持久迁移计划，每轮最多发送一次停止或创建：先重新确认目标容量，再逐个停止来源实例，全部精确回查 STOPPED 后才能在目标节点逐项创建同等总卡数的新实例。每个目标配额都有独立迁移标签，响应不明确时只对账，不重复创建。目标被抢占或容量变化时退出迁移计划，转回正常补齐；未确认释放的卡仍占额度。迁移后至少冷却 300 秒，任务完成或关闭扩容时不再发起迁移。

合并会暂时降低运行卡数；停止实例上的未完成分片由原有队列租约恢复。已提交的结果继续保留。48 卡在当前 16 卡节点上理论上至少需要 3 个节点，实际节点数取决于哪些节点有可用资源。

当前启动参数和资源策略见运行目录 `fragment_notebooks/execution-r02/CONTROLLER_LAUNCH.json` 与 `RESOURCE_POLICY.json`。最新切换证据见 `fragment_notebooks/execution-r02/card_cap_48/`。之前关闭新申请的回执保留为历史。原高优调优实例保持 STOPPED，原四个分布式 Job 保持 CANCELLED。

图像处理已启用按图像数量聚合、有界下一批预取以及每任务性能计数。真实数据对照与发布依据见 [优化报告](runs/dcvlm_frontpass_20261022T2129Z/OPTIMIZATION_REPORT.md)。实时状态以运行 JSON 为准，不以本文件快照为准。

## 行为

- 约每 10 秒检查一次空闲 NPU；每 60 秒检查所管理实例和 worker。平台请求耗时、超时退避会延长实际周期。
- 使用当前合法的 1/2/4/8/16 卡规格，检查 NPU、CPU、内存、Ready 状态与维护状态。节点统计仅用于挑选候选，实际是否运行由平台状态与 worker 回执确认。
- 所有新实例均为 LOW（priority=3），只使用指定昇腾工作区、910B 资源组和 high-dimensionaldata 项目。排除 047、248、295 节点。
- 每次申请同时检查本管道 `--max-cards` 上限与当前账号可见的 256 卡总额度，规划后、发送前再次检查；本次管道上限为 48。256 卡计入其他实验的可见申请，但不会操作它们。这不是全项目所有成员资源用量的证明。
- 已提交但尚未 RUNNING 的实例保留节点预约。PENDING 等待至少 90 秒且最新资源不再满足，或累计 PENDING 达到 240 秒，将撤回申请；停止前再次按精确 ID 复查，已经 RUNNING 则保留。确认 STOPPED 后才释放预留并允许新尝试。退休实例不再重启，节点冷却 120→240→480→600 秒。创建与停止均先保存意图、单次发送，不确定结果只读对账。
- 实例 RUNNING 后幂等启动检测 worker。所有 worker 共用原分片队列，租约及 token 防止重复提交，抢占后可恢复未完成分片。
- 连续 5 个不同 worker attempt 出错时停止该实例的自动 worker 重启并标记 action_required；控制循环继续记录状态。
- 全量任务与完成索引都验收后，停止本工具管理的 Notebook 和本次原四个 Job，保留对象与回执。不会因为纯文本先完成或到达 24 小时目标就宣布全库完成。

原有与新增输出共用队列；不修改冻结任务计划，不复用旧筛选结果。SSCD 仍只提取特征，全局去重和疑似图集中 API 复核属于后续阶段。

## 代码与运行入口

主入口：`fragment_controller.py`；扩容分配：`fragment_allocator.py`；自动合并：`fragment_consolidation.py`；容器内幂等启动：`notebook_worker.py`。CPU 纯文本路径：`text_worker.py`。

预算与三资源规划已抽到 [通用资源管理模块](../resource_manager/README.md)，通过 `resource_planning.py` 对接本管道的持久回执。现有控制器继续负责数据队列完成判断和已验证 worker；新工作负载可用通用 `engine`＋平台适配器，输入预算、argv 命令和运行配置。不要同时启动两个控制器接管当前 DCVLM。Native Serving 通过专用后端接入，当前 SDK 无精确节点选择能力，不能承诺同样的节点集中行为。

```bash
/root/.local/share/uv/tools/inspire-skill/bin/python -m corpus_pipeline.cluster.fragment_controller \
  --root /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z \
  --submission /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z/fragment_notebooks/execution-r02/SUBMITTED.json \
  --interval 60 --auto-scale --max-cards 48 --consolidate
```

本轮已完成，控制器已经退出；上述命令保留作运行记录，不应为已完成的本轮重新申请资源。容器本地锁阻止同控制主机重复启动；变更控制主机前须确认旧控制器已停止，不依赖跨主机 flock。

最新模块化测试、真实平台只读核验与控制器切换见 [模块化上线记录](runs/dcvlm_frontpass_20261022T2129Z/fragment_notebooks/execution-r02/modular_resource_manager/REPORT.md)。此前 105 项测试及 33 卡从 7 个节点合并到 4 个节点的证据保留在 [首次集中上线记录](runs/dcvlm_frontpass_20261022T2129Z/fragment_notebooks/execution-r02/card_cap_48/REPORT.md)。图像处理的数值与吞吐对照仍见此前优化报告。

## 状态与回执

- [全量进度](runs/dcvlm_frontpass_20261022T2129Z/PROGRESS.json)：分别展示记录与图像吞吐；任一未完成通道没有有效速率时，总 ETA 为未知。
- [控制器状态](runs/dcvlm_frontpass_20261022T2129Z/fragment_notebooks/execution-r02/CONTROLLER_RUNTIME.json)：实例、worker、扩容健康与操作结果。
- [启动回执](runs/dcvlm_frontpass_20261022T2129Z/fragment_notebooks/execution-r02/CONTROLLER_LAUNCH.json)：进程与命令。
- [最近资源发现](runs/dcvlm_frontpass_20261022T2129Z/fragment_notebooks/execution-r02/dynamic/LAST_DISCOVERY.json)：候选配额、节点预约、新接收实例。
- `FRAGMENT_RESERVATIONS.json`：本次运行的持久卡数账本；缺失账本或身份不一致时拒绝新申请。
- `FRAGMENT_CONSOLIDATION.json` 与 `fragment_consolidation/`：当前迁移计划与历史回执。
- `fragment_notebooks/execution-r02/dynamic/`：自动创建的 INTENT、ACCEPTED、UNCONFIRMED 或 REJECTED 审计。
- [原四个 Job 状态](runs/dcvlm_frontpass_20261022T2129Z/JOB_STATUS.json)。

节点统计只作候选依据；平台实际 RUNNING 和 worker 提交结果才证明获得了资源。平台事件中的 CPU 提示可能由汇总事件派生，不单凭该提示宣称硬件故障。一日目标仍取决于实际分配与持续图像吞吐。

## 可选的 tuned 执行路径

新一轮 CPU 准备、预取和批量调优的运行说明见 [TUNING.md](TUNING.md)。当前 `dcvlm_frontpass_20261022T2129Z` 已完成独立真实 NPU 完整验证并滚动发布，详情见该运行的 `NPU_TUNING_REPORT.md`；其他运行仍须取得自己的 `TUNED_QUALIFIED.json`。原有 `inference.py`、`optimized_inference.py`、`frontpass.py` 三个冻结文件保持不变。

`node_worker.py` 在启动时校验回执绑定的配置、聚合参数、代码哈希和完整验证产物，成功后才选择 `tuned_frontpass`。没有该回执时继续使用原入口；存在但无效时拒绝启动，不静默切换。原 `QUALIFIED.json` 和 `OPTIMIZED_QUALIFIED.json` 保留。

Notebook 启动命令中仍可能出现 `--batch-size 32 --threads 4`。这是兼容旧入口的参数；进入 tuned 路径后的实际 batch、解码线程、CPU 线程、预取和聚合配置以资格回执及 `workers/<worker_id>/execution_identity.json`、任务 `summary.json` 为准。该机制只作用于 tuned 入口，直接运行原入口不会自动使用新配置。

发布或移除新回执仅影响后续启动的进程。活跃 worker 不会自动热切换；上线应使用经过归属核验的 `tuned_rollout.py`，回滚按 TUNING.md 的控制器冻结、精确 worker 停止和租约处理流程执行。不要用删除结果、批量杀进程或停止其他 Notebook 的方式更新执行配置。
