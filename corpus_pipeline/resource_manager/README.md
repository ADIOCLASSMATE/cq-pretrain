# 通用资源管理模块

给出**用卡上限、命令 argv、平台配置**，管理器负责轮询资源、申请合适规格、启动命令、记录状态以及在中断后恢复。业务程序负责实际工作。这一层没有数据集、模型框架或特定账户绑定。

现有 DCVLM 流水线仍由原 `fragment_controller` 管理，通过桥接复用这里的规划核心。**不要另外启动通用 manager 接管同一批资源**；两套控制器不能同时拥有同一个任务。

## 模块边界

| 模块 | 职责 |
| --- | --- |
| `models.py` | 明确 workload、预算、节点空闲量、配额和预约的配置契约。 |
| `planning.py` | 无平台调用的纯规划：三资源约束、预算、优先已有节点、弹性拆分和迁移建议。 |
| `engine.py` | 持久状态、轮询、创建意图、恢复、健康检查、完成和停止流程。 |
| `inspire_adapter.py` | 实现平台适配接口，查询配额和资源，创建 Notebook / 原生 Serving，校验资源归属。 |
| `command_runner.py` | Notebook 内运行 argv 命令、保存进程身份、日志与操作回执，执行就绪和完成探针。 |
| `__main__.py` | `plan`、`status`、`run`、`stop` 命令入口。 |

其他平台可以实现 `engine.Adapter` 接口，替换配置中的 `adapter` 工厂。已有调度器也可以只调用 `plan_allocations` / `plan_consolidation`，自行执行返回的计划；无需同时采用通用 engine。

## 配置示例

- [弹性 Notebook](examples/notebook.json)：48 卡上限，可申请 1、2、4、8、16 卡实例；适合能够分片、断点续跑的批处理命令。
- [固定规格 Serving](examples/serving.json)：16 卡上限，每个副本固定单节点 8 卡，最多两个独立 Serving allocation。

示例的账户、工作空间、项目、计算组、镜像、模型名称和绝对路径都是占位值。使用前替换，并确认镜像包含依赖、代码和探针；控制主机与容器中的路径不是同一套路径。

`workload.scope` 标识任务，`profile` 标识运行配置，两者都必须显式指定。`adapter_config` 决定具体账户、平台资源组和镜像；仅修改 `profile` 字符串不会自动选择另一套平台配置。

`state_dir` 和 `registry_dir` 位于控制主机，分别保存控制器状态和适配器操作/资源绑定。Notebook 的 `repository`、`remote_python`、`worker_state_dir` 是远端容器路径。`repository` 必须能够导入 `corpus_pipeline`；工作命令、数据与检查点路径须在对应容器内可用。配置不会自动上传代码或同步两个分区的文件。

### 预算与放置

`max_cards` 是该 scope/profile 的申请上限，`global_max_cards` 是适配器在指定工作空间内对该账户可观测资源应用的另一道上限；示例值 256 可按实际约束修改。规划同时检查 GPU、CPU、内存和当前平台合法配额。预算不保证立即取得相应资源，也不会自动把命令改造成数据并行程序。

弹性任务优先复用已有节点，然后减少新增节点。例如已有 1 卡实例的节点又空出 7 卡，在 CPU、内存与配额均满足时，可继续放入 4+2+1 卡实例。PENDING / UNKNOWN 申请仍占预算，也占目标节点的相应三种资源，但不封锁该节点剩余空位。资源 ID / 操作 ID 用于去重；仅确证停止并退休后释放预约，列表中暂时消失不等于释放。创建结果不确定时保留意图并查询恢复，不盲目重发。

固定模型副本使用 `replica_cards`，如 8 卡副本不会被拆成 4+2+1。降低运行中任务的预算会限制后续申请，不能把它当作立即缩容命令。

### 命令与生命周期

`command`、`readiness_command`、`drain_command`、`completion_command` 都是 JSON argv 数组，不是隐式 shell 字符串。重定向、管道和变量展开需要业务程序自行实现，或明确以 shell 作为 argv 的可执行程序。

适配器给每个工作命令设置：

| 环境变量 | 含义 |
| --- | --- |
| `RESOURCE_MANAGER_CARDS` | 当前 allocation 的卡数，而不是整个任务上限。 |
| `RESOURCE_MANAGER_ALLOCATION_ID` | 该次分配的稳定操作 ID，可用于日志和业务租约标识；重新分配会有新 ID。 |

工作命令需要读取卡数并启动合适的本地进程。数据分片、队列领取、租约、幂等输出、断点续跑、跨实例通信和模型张量并行均由业务层实现。管理器不生成全局 rank，也不根据 allocation ID 自动划分数据。

Notebook 示例的 `completion_command` 必须在**整个任务完成**时返回 0，其余情况返回非零。它会在运行中的容器内执行，不应只检查当前 worker 的本地分片。未配置完成探针时，engine 使用已启动 worker 的成功退出判断结束，因此短暂空队列不能被业务程序误报为全局完成。`readiness_command` 返回 0 表示该实例已就绪，不能仅检查进程存在。

迁移策略为：

- `disabled`：关闭主动合并迁移。
- `restartable`：适合可安全中断恢复的批处理；确证源实例停止后，再申请目标配额，可将 3 卡或 7 卡整组迁移为多个合法实例。
- `drain`：需要显式 drain 命令与适配器支持。服务迁移还要求至少一个就绪副本和迁移后的就绪数量边界；执行前会重新检查状态。该边界不是平台抢占时的可用性保证。

## CLI

从能导入本仓库且安装了对应平台 SDK 的控制环境执行。先复制一份示例到任务自己的配置文件，完成占位值替换。以下命令路径同样是示例：

```bash
python -m corpus_pipeline.resource_manager plan \
  --config /srv/example-resource-manager/notebook.json

python -m corpus_pipeline.resource_manager run \
  --config /srv/example-resource-manager/notebook.json \
  --max-cards 48 \
  --command-json '["/opt/example/venv/bin/python", "/workspace/example/worker.py", "--config", "/workspace/example/workload.json"]'

python -m corpus_pipeline.resource_manager status \
  --config /srv/example-resource-manager/notebook.json

python -m corpus_pipeline.resource_manager stop \
  --config /srv/example-resource-manager/notebook.json
```

`plan` 查询当前平台资源并输出计划，不创建实例。`status` 读取本地已保存状态，不是实时平台查询。`run` 会实际申请和管理资源，默认持续轮询；`run --once` 只执行一个管理周期，也可能申请资源。

`stop` 只写入指定 scope/profile 的停止请求，由仍在运行的控制器执行资源释放；控制器不在运行时，需要使用原配置恢复 `run` 才会处理。SIGINT / SIGTERM 退出的是管理循环，不等于释放平台资源。恢复必须使用同一状态目录，不能删掉账本后重新申请。

`--max-cards`、`--command-json`、`--profile` 覆盖配置中的对应字段，`--state-dir` 覆盖状态目录。命令、scope、profile 等身份字段一旦写入状态便受校验；只有预算和轮询间隔可以不改变任务身份。更换工作负载应使用独立状态目录，并先处理旧任务的资源，不能借新目录重复接管。相同状态目录要求同一控制主机和单个运行控制器；状态和适配器绑定记录都应保留。

## 原生 Serving 的边界

Serving 示例走平台的**模型部署**接口。每个 allocation 创建一个 `replicas=1`、`nodes_per_replica=1` 的独立 Serving，原生自动扩缩容关闭，由此管理器在预算内管理 allocation 数量。示例的张量并行参数属于占位模型服务命令，必须替换为所用推理框架实际支持的参数。

当前 SDK 的原生 Serving 创建接口没有 `node` 字段，适配器报告 `node_placement=False`，因此不能承诺将多个 Serving 集中到指定物理节点。规划中的 `native-scheduler` 只是预算容量表示，不是实际节点。该后端没有 drain 合约，示例明确设为 `migration_policy="disabled"`、`drain_command=null`。

Notebook 默认继续使用实际空闲节点。显式设置 `adapter_config.notebook_placement="native_scheduler"` 时，按指定 LOW/HIGH 优先级交给平台排队，创建请求不绑定节点；此时 `native-scheduler` 也只表示请求预算，不能解释为空闲物理资源。应设置 `migration_policy="disabled"`；待调度、运行和结果未知的请求均计入总卡数上限。平台选中排除节点时禁止启动 worker，但保留已验证所有权的停止能力。

多个 allocation 会产生多个独立 endpoint，统一地址、发现、负载均衡、请求重试和流量迁移需要业务层路由实现。`min_ready_replicas=1` 不会替代路由或提供零停机保证；停止请求会停止本任务拥有的服务。

通用 Serving 路径已提供接口适配与离线测试，**尚未通过真实模型部署验收**。正式采用前需要用实际镜像、模型、命令和就绪探针验证完整生命周期；这份示例不代表已经部署成功。
