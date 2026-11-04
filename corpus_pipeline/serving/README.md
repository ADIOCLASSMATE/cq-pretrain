# 数据管道原生 Serving

当前实现位于 `inspire_serving/`。旧 GPU Job 实现已退役，原始记录保留在 `RETIRED_GPUJOB_20261022.md`；旧控制器 `runtime/WATCHDOG_DISABLED` 保持存在，不得与新发布器同时运行。

## 资源与身份

账户 wjx-ascend，空间昇腾卡公共空间，项目多模态大模型新架构评测探索与scaling-law，资源组910B资源。已 live 核实标准调度项目 HIGH=6、LOW=3。

| 模型 | HIGH | LOW | 当前精度与拓扑 |
|---|---|---|---|
| Qwen3.5-397B-A17B | 32卡 | 32卡 | BF16，单replica双16卡节点，TP8/DP4/EP32 |
| Gemma4-31B-IT | 4卡 | 4卡 | BF16，单replica单节点TP4 |

验收阶段固定申请72卡；验收及五层BF16基线冻结后，已按计划停止两项HIGH并释放其36卡。当前只保留原LOW服务：Qwen LOW32排队，Gemma LOW4运行且已验证，registry发布generator LOW ready=false、verifier LOW ready=true。两HIGH对象保留为STOPPED、自动恢复仍关闭；既有LOW ID不变。完整回执为 `inspire_serving/runtime/planned-high-release.json` 与 `post-high-release-live-20261023.json`。四个自有原生 Serving ID 存在 `inspire_serving/runtime/{qwen,gemma}-{high,low}.json`；不接管历史64卡Qwen、其他项目或其他用户资源。Qwen本项目登记名 `corpus-qwen397b-bf16-ref-20261023`，V1是公共权重只读引用，未复制806GB权重；平台必须使用同项目Model Registry。

Serving 创建合同没有 `exclude_nodes` / `specified_nodes`。ScaleServing只改变replica数，不改变每replica节点/卡数；资源档位变化必须另建完整配置的Serving并重新验收。不能把旧Job的节点排除能力声称为Serving能力。原生16卡W8A8档已准备并通过live dry-plan，`downsize.py`实现恢复档位选择，但尚未发生真实抢占、未执行16卡量化推理，不能称为实机验证完成。

## 独立监控、恢复和端点

`inspire_serving/watchdog.py` 守护 `monitor.py`，各服务使用常驻工作线程及同线程SDK Client，正常2秒查询状态，10秒查询事件，异常退避至30秒。watchdog每5秒以PID、Linux starttime、完整argv核验monitor；控制主机关机仍需要外部入口，不能称为跨主机高可用。

平台负责原生实例排队/重调度/重启。PENDING、模型加载、HTTP超时、503、观察失败均不触发服务重建或停止。HIGH不参加LOW恢复。只有自有LOW在最近RUNNING→STOPPED转换中有明确近期“被抢占”受害者事件，才允许对同ID执行一次start；“This pod will preempt pod ...”是主动抢占别人，不作受害者证据。没有真实平台抢占恢复实测时，只报告状态机测试结果。

`downsize.py`只在该LOW明确被抢占且旧ID仍STOPPED时评估完整空闲节点：至少2个则原BF16档start；两次相隔至少15秒的新live观测均只有1个完整节点，才创建单节点W8A8_16原生Serving。0个则继续等待。每次恢复意图重置采样，原始PENDING不会因等待过久切档。旧ID停止状态需在create前再次核实，full plan/payload/CREATE_INTENT先持久，响应丢失仅exact-name与不可变字段唯一匹配reconcile。缩卡前另核验当前账户跨项目可见Serving/Notebook/活跃Job：单独记录实际有节点分配卡数与排队申请卡数，后者保守计入256卡上限；缺少身份/资源/分页证据则不新建。当前两项HIGH已停止，活跃LOW申请总量为36卡；若Qwen LOW由32卡缩至16卡，活跃LOW申请总量为20卡，STOPPED的HIGH不计入；每精度独立revision、图片资格与绑定incarnation的真实数据质量门，后者未过不发布ready。量化真实数据质量门由 `quantization_quality.py` 自动执行：W8图片资格通过且冻结BF16基线READY存在后，由monitor启动独立qualification-only进程，覆盖照片caption、图表QA、文档OCR、渲染QA、多轮/多图五层，Gemma看原图独立复核；所有case通过且完整身份/基线hash/worker代码hash绑定才发布。当前尚未部署W8或发生真实抢占，因此只证明接口与状态机，不能称为实机全链路恢复验证。平台仍控制节点选择，不支持排除占用/故障节点字段。

新monitor只发布经过模型身份、processor/runtime版本、所有实例新鲜心跳、boot UUID、Qwen当前档位卡数对应的HCCL回执（BF16为32rank已实测；W8A8为16rank，尚未实测）和图片/并发验证的服务。同service重启后，platform实例created/started/restarts/term与boot变动立即撤销旧资格。HTTP响应boot也必须属于本次验收；不能从旧共享文件复用成功。

统一 `endpoints.json` 仍为 `corpus_pipeline.endpoints.v1`，含模型/精度/权重、processor与runtime revision、role、base_url、expected_boots、ready及30秒ready_expires_at。消费者每次重试重新读取，并校验返回model与boot。HIGH验收期间优先HIGH；HIGH不可用时可选已完整验收的LOW。只有明确写入 `inspire_serving/runtime/HIGH_ACCEPTANCE_COMPLETE` 才允许在HIGH可用时主动优先LOW；该标记须由实际全流程验收完成后产生。

## 鉴权与图片验收

原生平台入口使用 `Authorization: Bearer $INF_API_KEY`；指定key存 `private/inspire-serving.key`（0600），生产进程从文件安全注入环境，registry和日志不存明文。网关依赖平台外层鉴权，仅向引擎注入独立私钥，避免两个不同Bearer令牌冲突。所有网络请求明确禁用代理。

`vision_probe.py --api-key-env INF_API_KEY` 做无效token拒绝、模型身份、每请求两张随机数字/色块真实PNG、严格有序JSON和并发1/4/8验证，共13请求26图。全部请求/响应/图像/延迟存 `inspire_serving/runtime/probes/`。独立probe异常退出或1800秒超时只重试probe，最多5次退避，绝不重启模型。此验收不代替真实DCVLM照片、图表、文档语义QC。

## 观察与测试

- `inspire_serving/runtime/watchdog-heartbeat.json`、`monitor-heartbeat.json`：活跃进程身份。
- `inspire_serving/runtime/*-live.json`：当前权威平台服务、实例和事件。
- `inspire_serving/runtime/*-attestation.json`：运行版本与本次容器身份证明。
- `inspire_serving/runtime/*-{plan,payload}.json`：完整计划及原始创建载荷。
- `inspire_serving/runtime/probes/`：模型多图并发验收原始证据。
- `/root/.local/share/uv/tools/inspire-skill/bin/python -m unittest discover -s corpus_pipeline/serving/inspire_serving -p 'test_monitor.py' -v`。
- `.venv/bin/python -m unittest corpus_pipeline.serving.inspire_serving.test_quantization_quality -v`。

本次planned HIGH stop通过 `retire_high.py` 执行：先验证最终COMPLETE、reader、16/32并发和冻结基线证据，写HIGH_ACCEPTANCE_COMPLETE，确认Gemma已切合格LOW，再停止两项HIGH并确认实例终态。HIGH自动恢复一直关闭，因此此计划停机保留LOW watchdog，不需要停掉整个监控器；不是抢占实验。

若用户另行要求停止仍处于自动恢复管理的LOW服务，应先禁用新watchdog并确认monitor已停，再处理精确自有服务，避免与恢复控制冲突。停止监控本身不会停止任何Serving。不要通过停HTTP、杀模型或操作无关资源测试抢占。

当前测试：19项native monitor/恢复/质量接口测试、13项量化worker测试通过。当前原生watchdog PID3545592、monitor PID3714986的PID/starttime/argv已在 `post-high-release-live-20261023.json` 实核；动态最新身份仍以heartbeat为准。新watchdog的实际受控恢复为6.10秒，四个Serving ID无变化、无额外create；这不等同于NPU抢占恢复实测。
