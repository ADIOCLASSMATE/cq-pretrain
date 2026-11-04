# Retired GPU Job implementation — not the current deployment

2026-10-23用户明确改为原生Serving；本文仅保留历史。旧Job均已停止，旧watchdog禁用，不能据本文重启。

# 数据管道推理服务与抢占恢复

此目录只管理本轮 `corpus_pipeline` 创建的 Qwen / Gemma 服务。历史 64 卡 Qwen Serving、其他项目任务和其他用户任务不在控制范围内。运行状态以 `runtime/*-state.json`、平台 live 回执和 `endpoints.json` 为准，不能把提交成功或平台 RUNNING 当作图片推理已通过。

## 当前部署合同

账户 `wjx-ascend`；空间 `昇腾卡公共空间`；项目 `多模态大模型新架构评测探索与scaling-law`；资源组 `910B资源`。2026-10-22 live 核实本空间不是 fair scheduling，提交 priority=3 对应 LOW；平台状态的数值 priority=13 属另一刻度，核对 `priority_level=LOW`。

GPU Job 的当前最低 quota 是 16 卡。Serving 有 4 卡 quota，但没有 `exclude_nodes` 创建字段，因此正式服务用 GPU Job 承载认证 HTTP 网关。固定排除从用户现有 `node_policy.apply_fixed_exclusions` 合并；不删除既有故障节点记录。

| 档位 | 资源 | 权重和运行参数 | 发布条件 |
|---|---|---|---|
| Qwen BF16 首选 | 2×16 卡，shm128 GiB/节点 | 本地 `Qwen3.5-397B-A17B`，vllm-ascend:v0.23.0rc1，TP8/DP4/EP32，65536上下文，32序列/DP | HCCL、模型和独立多图 canary 通过 |
| Qwen W8A8 备用 | 1×16 卡，shm128 GiB | 本地 `Qwen3.5-397B-A17B-w8a8-mtp`，TP8/DP2/EP16，`--quantization ascend`，32768上下文，16序列/DP | 作为独立精度版本重新验证，不能沿用 BF16 成绩 |
| Gemma BF16 | 1×16 卡，shm128 GiB | Gemma4-31B-IT，TP4×4 本地副本，认证网关8000，后端9000–9003 | 四个引擎健康且新多图 canary 通过 |

Qwen 的 16 卡量化路线有 [vLLM Ascend 官方模型文档](https://docs.vllm.ai/projects/ascend/en/latest/tutorials/models/Qwen3.5-397B-A17B.html) 的容量与参数依据。本机已检查 BF16 94 分片及 W8A8 99 分片存在；这不是文件内容逐字节校验，也不是本轮已经完成了量化推理。`profiles.json` 的 revision 使用配置与权重索引摘要；量化档另包含量化说明摘要。

本轮经协调的最大申请量为 Qwen32 + Gemma16 = 48 卡，仍受用户项目256卡上限约束。旧代全部确认终态以后才建新代，控制器不会为竞速同时保留两份 Qwen。Gemma 16 卡 Job 内减少副本数不会将平台申请量变为4卡；当前 quota 不支持声称这种缩卡。

## 恢复行为

`supervisor.py` 对两个角色分别使用工作线程，正常每2秒观察 live job 状态、约5秒补取事件。每个角色最多一个在途检查；模型图片测试是独立进程。节点快照共享缓存2秒，运行期间不高频扫描所有节点；平台失败温和退避，不阻塞另一个角色。

HTTP超时、网关503、缺失记录、平台查询超时均不能证明抢占。缺失/异常观察只取消 readiness，不创建或停止任务。`will preempt` 表示抢占别人的事件不算受害者证据；明确 `is/was preempted` 等证据与排队/终态结合后才进入恢复。

恢复时按完整实时空闲节点选择 BF1632 → W8A8_16 档位，并在新 payload 排除所有占用节点，包括平台未提供其优先级的占用节点。最初尚未启动的32卡任务可以在两个相隔至少15秒的 live 快照均只能容纳16卡、且任务实例仍无节点落点时，切换到16卡资格验证。这是明确的容量选择，不以API超时或等待时长作为停止依据。停止前再次核实 live 状态；运行中的实例优先保留。

每次 create 先写 `CREATE_INTENT` 和完整 payload。创建响应丢失时仅按精确名字、账户和不可变配置核对对象，不盲重发；0个或多个候选时保持暂停。原始完整请求不从精简状态回填，所有新代另存请求文件。仅已登记的自有 job ID 可以停止。

平台当前会把 image_type 从 SOURCE_PRIVATE 归一成 SOURCE_PUBLIC，并把提交 envs 置空；控制器只容许这两项已观测归一，镜像完整 URL、资源、命令、项目、排除列表和 LOW 身份仍须一致。后续代将无敏感信息的 deployment ID 明写在 command，避免依赖丢失的 envs。首代 Qwen launcher 有仅针对已提交首代的固定 fallback。

`watchdog.py` 位于工作负载外的控制主机上，使用 `/proc` PID、进程 starttime 和完整 argv 确认身份，控制器退出后重新启动。已经真实终止过一次 supervisor 验证恢复：新控制器恢复了同一 job ID 和 owned_jobs；回执为 `runtime/watchdog-restart-validation.json`。它能跨本轮交互结束持续运行，但控制主机本身关机仍需由外部启动入口恢复，不能将同机守护描述成跨主机高可用。

## 图片验证与接口

统一注册表 `endpoints.json`，schema `corpus_pipeline.endpoints.v1`，按角色 generator/verifier 提供兼容 OpenAI 的 `/v1` base_url、模型名、权重revision、precision、processor/runtime revision、支持图片数、`request_overrides` 和 ready。消费者每批和每次重试重读注册表。新代/新精度的 URL 可以在 ready=false 时公布，消费者必须等 ready=true，且缓存身份必须包含模型和全部 revision。

`vision_probe.py` 使用真实 PNG 字节而不是图片路径占位符；每请求两张随机数字/色块图片，答案只存在于图片中。并发1、4、8，共13请求、26张图，严格比对字符、颜色、图片次序，同时验证模型身份和认证。保存全部请求、响应、图片和延迟/吞吐。过关才发布 ready；403、内容错误进入诊断；暂时网络/429/5xx仅重试 canary，最多5次指数退避，不重启模型。

这个 canary 验证图片输入能力、基本视觉准确性和有限并发，不能证明照片、图表、文档等全域 recaption 质量或量化无损。真实 DCVLM 的多领域生成与语义 QC 必须另由 `production/` 的人工审查/复核门执行。

网关8000仅转发规定的 OpenAI 路由到引擎9000；明文令牌不写 registry 或日志。共享密钥在 `private/client.key`（目录0700，文件0600）；生产进程由 Python 读取后注入 `CORPUS_INFERENCE_API_KEY`。上游引擎密钥独立，下载和请求都逐调用禁用代理。

## 检查入口

- `runtime/supervisor-heartbeat.json`、`watchdog-heartbeat.json`：进程和更新时间。
- `runtime/qwen-state.json`、`gemma-state.json`：当前身份、档位、平台状态、资格验证。
- `runtime/live-resource-summary.json`：最近人工刷新的资源摘要；瞬时负数 available 不能用作可分配资源。
- `runtime/*-observation.json`：完整观测回执；`runtime/*payload*.json`：原始/后继请求。
- `runtime/probes/`：每代图片与并发验证，完整失败也保留。
- `python -m unittest discover -s corpus_pipeline/serving -p test_supervisor.py -v`：恢复状态机测试。

不要通过关闭 inference API 或制造超时测试抢占，不要停止无关任务。停止控制器只影响控制进程，不会停止模型任务；外层 watchdog 会恢复它。显式关闭外层守护需创建 `runtime/WATCHDOG_DISABLED`，再停止经 PID+starttime核实的控制进程，并另行处理要停止的具体自有模型任务。
