# 模型部署迁移与 DCVLM 验收记录

本次已改为平台 **模型部署 / Serving**。Qwen 高优 32 卡、Gemma 高优／低优各 4 卡均已通过真实图片和并发资格验证；Qwen 低优 32 卡已提交排队。**19 条真实 DCVLM 记录的端到端先导验收已完成，最终发布 9 条记录、8 张图片和 1 份 layout，并全部读回验证。** 这是小样本流程验收，不是全库处理或质量准确率评估。时间使用北京时间。

**02:12 收尾状态：两项临时 HIGH 已确认 STOPPED、实例终态，共释放本任务 36 张卡；Gemma LOW 已运行并接替 verifier，Qwen LOW 保持原 ID 排队，当前 generator 为 `ready=false`。** 这是验收后的计划停机切换，不是无缝交接或实际抢占恢复。原生 watchdog／monitor 仍在运行，Qwen LOW 获得资源后会先做资格验证，再发布可用端点；不会启动旧训练 Job。

| 当前 Serving | ID | 状态 |
|---|---|---|
| Qwen LOW，32 卡 BF16 | `sv-c6c87717-ac42-4ffe-addb-c54463ea27a4` | PENDING |
| Gemma LOW，4 卡 BF16 | `sv-48788dbf-f3b7-41e0-8265-88b8b4cb8939` | RUNNING，图片资格通过 |
| Qwen HIGH，32 卡 BF16 | `sv-1e8ef255-6134-4b37-b6ab-554e6490df24` | STOPPED，验收历史保留 |
| Gemma HIGH，4 卡 BF16 | `sv-c4d1aea3-e2b1-4170-82f7-ee87715295a1` | STOPPED，验收历史保留 |

切换先核实 Gemma LOW 的当前资格与注册表，再停止精确两项自有 HIGH；两项 LOW 的 ID 未变。回执绑定最终 COMPLETE、reader、并发测试与 BF16 基线的 SHA256，见 [planned-high-release.json](serving/inspire_serving/runtime/planned-high-release.json)。不处理无关队列或工作负载。以上是观察快照，实时状态见控制器回执。

## 已落实的迁移

2026-10-23 01:21:47，两项本任务创建的训练 Job 已确认 `CANCELLED`：

- Qwen：`job-216206dd-efee-495d-a063-5cf20ffe8da1`。
- Gemma：`job-be3f0805-30e0-48c2-95e9-4e38a6a7ace8`。

旧 Job 的 watchdog 与 supervisor 已按 PID、启动时间和命令核实后停止，状态文件已标记停用；`serving/runtime/WATCHDOG_DISABLED` 保留，防止旧控制器再拉起训练 Job。迁移回执为 [migration_to_serving.json](serving/runtime/migration_to_serving.json)。历史 Job 记录保留，不处理其他任务。

| 模型 | 高优验收部署 | 低优排队部署 | 精度 |
|---|---|---|---|
| Qwen3.5-397B-A17B | 单副本，2 节点 × 16 张 910B | 同规格独立 Serving | BF16 |
| Gemma4-31B-IT | 单副本，4 张 910B，TP4 | 同规格独立 Serving | BF16 |

四项合计申请 72 张卡，分别提交 priority=6 / 3，对应本项目 HIGH / LOW。验收期间实际分配 40 张卡，另 32 张为 Qwen LOW 排队申请，不能把排队申请数当作实际占用。部署前检查账户、项目、Serving quota、镜像与实时用量；保留每项 dry-run、完整请求和创建回执。生成及复核使用高优；低优通过图片与并发验收后才可接替。

Gemma 高优 `corpus-gemma4-serving-high-20261023-r01` 在 01:25 创建并调度到 `infra-gpu-npu-286` 拉取镜像；低优 `corpus-gemma4-serving-low-20261023-r01` 在 01:26 创建。这里只记录当时状态，后续以 live 回执与资格验证结果为准。

Qwen 高优 `corpus-qwen-serving-high-20261023-r01` 已创建（`sv-1e8ef255-6134-4b37-b6ab-554e6490df24`）。首次提交暴露模型登记属于另一项目，平台拒绝创建；精确名称核对不存在后，在当前项目新登记公共权重引用，再提交成功，没有复制权重或创建重复服务。

Gemma 首次真实运行确认 `vllm=0.30.0+empty`、`vllm-ascend=0.30.0rc2`、`torch-npu=2.10.0.post7`、`transformers=5.14.1` 和四张 `Ascend910B2C`。BF16 权重加载完成后，两项 4 卡服务都遇到 HCCL EI0010：碎片卡跨通信平面。根据运行库报错启用 `HCCL_INTRA_ROCE_ENABLE=1`，并显式配置通信接口与本机 IP；确认各自 STOPPED 后启动同一 Serving，随后推理成功，无需增加卡数。

最终带进程启动身份、请求参数和鉴权校验的资格测试，Gemma 高优与低优各 13/13 通过，覆盖并发 1 / 4 / 8、每请求两张图。高优 8 并发约 39.05 输出 token/s，低优约 37.23 输出 token/s；这只是每请求约 69 输出 token 的小型 OCR 测试，不能外推真实长 caption 或全库吞吐。实际 DCVLM 照片、CLEVR 与双图顺序另有验证。详见 [Gemma 实测报告](serving/gemma4/VERIFIED_SERVING_20261023.md)。

Qwen 高优获得 `infra-gpu-npu-481` 和 `infra-gpu-npu-400` 两节点，32 rank 的 HCCL AllReduce / AllGather 正确性测试全部通过；实际 `vllm-ascend=0.23.0rc1`、`torch-npu=2.10.0.post2`、`transformers=5.5.4`，BF16。首轮 512×512 数字图 OCR 失败，未放行。参照既有部署实证，为每次请求显式设置 `mm_processor_kwargs={min_pixels:1048576,max_pixels:1048576}`，两图请求的输入 token 从 569 升至 2105，证明配置生效；重验 13/13 严格通过，8 并发约 39.53 输出 token/s。生产客户端同步修复参数白名单，参数与预处理身份进入缓存键。原图 128 门槛仍在任何视觉上采样前执行。

完整资格回执、两次 Qwen 测试和三轮 live 状态见 [Serving 实测报告](serving/inspire_serving/QUALIFICATION_20261023.md)。低优 Qwen 尚未分卡。平台日志中的“将抢占其他实例以释放资源”表示本服务正在获得资源，不能解释为本服务被抢占。

另对两项 HIGH 各执行 16 与 32 并发的双图 OCR 请求，每模型 48/48 请求通过数字、颜色、图序、JSON、模型身份、鉴权和当前 boot 校验，合计 96 个请求。结果如下：

| 模型 | 客户端并发 | 输出 token/s | 请求/s |
|---|---:|---:|---:|
| Qwen BF16 | 16 | 71.75 | 1.040 |
| Qwen BF16 | 32 | 88.99 | 1.290 |
| Gemma BF16 | 16 | 40.08 | 0.581 |
| Gemma BF16 | 32 | 40.61 | 0.588 |

这是每请求约 69 输出 token 的短任务压力测试；Gemma 配置 `max_num_seqs=8`，增大客户端并发主要增加排队。不能把该数值当作长 caption 吞吐或模型最大性能。原始请求与回执见 [16／32 并发结果](serving/inspire_serving/runtime/benchmark_16_32_20261023T020158/summary.json)。

## 服务与恢复合同

客户端使用平台要求的 `Authorization: Bearer INF_API_KEY`。密钥保存在权限 0600 的私有文件，通过进程环境注入；不写入公开注册表、报告或请求日志。引擎内部密钥与平台调用密钥分开。上述三项已运行服务均已实际验证错误凭据拒绝、正确凭据成功、模型身份、实际运行库版本、多图顺序和内容。

Serving 的已核实 SDK / Console 创建合同没有 `exclude_nodes` / `specified_nodes` 字段；`ScaleServing` 只能改变副本数，不能改变单副本 GPU 规格。不能宣称训练 Job 的节点排除机制已经迁移成功。后台恢复只可使用实际支持的 Serving 操作；更换资源规格或精度时必须另建 Serving、验证后更新端点。HTTP 超时本身不是抢占证据。

动态端点注册表为 [serving/endpoints.json](serving/endpoints.json)。`ready` 绑定真实图片验收、平台实例、进程 boot ID 和 30 秒租约；生产客户端核查响应 boot ID，每次重试重读端点。模型、权重版本、精度、处理器、运行参数、prompt 和输入身份参与成功结果缓存；续租不能触发无意义重跑。

新 Serving watchdog 已真实测试：仅终止按 PID／启动时间／命令核实的 monitor，约 6.10 秒恢复；前后四项服务 ID 不变，没有新 create，也没有停止或重启模型服务。回执为 [native-watchdog-restart-validation-20261023.json](serving/inspire_serving/runtime/native-watchdog-restart-validation-20261023.json)。这是监控进程恢复测试，仍不是实际 NPU 抢占恢复测试。

低优缩卡只在明确近期受害者事件、旧实例已释放、节点和预算检查通过后执行。16 卡 W8A8 属独立精度版本，须重新通过图片和真实原生任务质量门；本轮尚未分配该档位，因此不宣称量化推理或实机抢占切换已经验证。新 Serving 创建合同仍不支持指定排除节点。

已从最终准入数据离线冻结照片 caption、多轮对话、图表、文档 OCR、渲染 QA 五类 BF16 基线，每类 1 条。冻结前重新核对原始 TAR 图像 SHA、实际生成与复核回执、最终发布及手动隔离决定；没有新增模型请求。基线 SHA256 为 `cbe0154e2414f4b3b1765bd3d85dcf56fe2467db323bda18a6de023403211720`，见 [基线](serving/inspire_serving/runtime/BF16_BASELINE_READY.json) 与 [读回校验](serving/inspire_serving/runtime/BF16_BASELINE_READBACK.json)。W8 候选只有在同请求、同图像、同预处理下通过实际质量复核才能进入生产；该小基线不能证明全域量化等价。

## DCVLM 执行状态

本轮继续使用 19 条完整原始记录先导，保留此前实际完成的尺寸检查、CF / UniFD、SSCD 和一次 Kimi 复核证据。尺寸硬门槛保持原始短边至少 128；DCVLM 保留原生任务方向与上下文。AI 图像来源不明保持隔离，AI 文本单独标记。

检测器结果复用核对图片身份、检查点、预处理、运行库和完成文件；Kimi 复用核对完整 prompt、模型、token 参数及实际发送图片字节，不因仅运输代码改变而重复调用。模型注释、独立质检、layout、SSCD 复核与引用发布、读回均已在首轮真实运行。本轮真实 release 均为无 parent 的候选；历史链追加与更新由测试覆盖，尚未真实追加到已接受的历史语料。

首轮 19 条记录中，16 条取得真实生成／质检，2 条多轮记录因模型数错 segment 索引而被拦截，1 条因原图短边小于 128 不标注。首轮候选 release 已撤回可见入口并保留审计产物；第二轮修正可编辑索引与“未修改则保留原答案”的 delta 协议，避免静默改写用户文本。最终 18 条可处理记录（17 条含图、1 条纯文本）均有真实生成／复核终态，含图记录涉及 18 张合格图片；5 份实际 layout 中 1 份通过。纯文本保留自身任务，没有虚构图片。

根代理逐图复核了两张 caption、一个 CLEVR 算术题、图表、唯一通过的 Websight layout 和 SSCD 棋盘候选。夜景 caption 中无法确认的“church”推断触发单条真实修订；修订后仍有不确定性则整条隔离，不强行清空 concerns。Websight 文本与区域结构吻合，但框仅为近似区域，不是像素级 OCR 真值。SSCD 相似度 0.954 的棋盘分别为 8 列和 6 列、缺格位置也不同，Gemma 和实际看图均支持不合并。

上述视觉复查由本助手实际看图完成，不是人类标注或全库质量保证。夜景记录和另一条要求从图像判断 race 的 VisDial 记录均已持久隔离；可见肤色不能验证身份属性，双模型通过也不等于每个细节正确。原始标签、模型输出和失败回执保留，不篡改为通过。

最终 release 为 `49d1405c20a7d80efecd504eb9facbd26486ea51bf3765e0db923ab2cce27210`：**7 条单轮 I2T、1 条多轮 I2T、1 条纯文本，共 8 图，另有 1 份 layout**。交错／多图未知来源保留审计结果，未冒充通过来源准入。最终 release 不继承撤回候选；独立 reader 确认两项隔离记录不再可见，全部原图 SHA、方向和完整上下文验证成功，离线 finalize 重复执行幂等。`COMPLETE.json` 明确 `canary_completed=true`、`full_corpus_processed=false`。

完整计数、真实新请求／缓存复用、质量反例和剩余边界见 [DCVLM 实测报告](production/CANARY_STATUS_20261022.md) 与 [最终 reader 证据](production/runs/dcvlm_canary_v2_20261022/final_reader_acceptance.json)。生产相关 28 项测试、Serving 监控恢复相关 19 项测试、量化质量 worker 13 项测试通过。

当前是可恢复的小样本流程：尚不是数亿记录的分布式调度器；近似 SSCD pair 只产生审计建议，未将近重 family 应用到训练采样权重；历史正式 release 的通用撤销尚需实现。gen-und 与 DCVLM-balance-200B 挂载后仍须验证具体版本、schema 和逐来源适配。本次 9/19 的通过比例不能外推正式语料。

此前的 [2026-10-22 执行记录](EXECUTION_20261022.md) 是迁移前的历史状态；其中训练 Job 的部署与控制进程描述已被本次 Serving 方案替代。
