# 原生 Serving 实测资格（北京时间2026-10-23）

Qwen3.5-397B-A17B BF16 HIGH32 已通过真实图片资格；Gemma4-31B-IT BF16 HIGH4、LOW4 也通过。Qwen LOW32仍在原生调度队列，未因等待超时重建。所有原始请求、响应、图片、运行版本与boot证据保存在runtime子目录及同名JSON汇总中。

- Qwen：双节点32rank HCCL allreduce/allgather正确；原始512px探针有数字OCR及JSON结构错误，完整失败保留在attempt01。恢复既有验证路线的请求`mm_processor_kwargs={min_pixels:1048576,max_pixels:1048576}`后，attempt02通过全部13请求26图，包含两图顺序、六位数字、色块、JSON结构、模型身份、平台无效token拒绝及当前boot绑定。每请求prompt_tokens从569升至2105。
- Qwen有限并发：1请求19.68秒、4请求40.62秒、8请求13.96秒；最后一档约39.53输出token/秒总吞吐。这包含不同预热阶段，属于短OCR任务探针，不能据此推算长recaption生产吞吐或宣称线性加速。
- Gemma：HIGH与LOW均经对应当前boot的13请求26图验证。HIGH额外真实DCVLM照片、CLEVR及有序双图测试由gemma4目录记录。
- 原生控制器固定4个服务线程，live SDK client同线程使用。native-live-validation-20261023.json保存3轮直接平台观察及watchdog/monitor PID+starttime+argv。watchdog自愈已观察到多次受控monitor重载；这是控制进程恢复，不是模型被抢占后的实机恢复验证。
- 原生Serving不暴露exclude_nodes/specified_nodes。缩卡代码另建W8A8_16独立精度Serving，仅明确被抢占LOW停态才能触发；live dry-plan已通过，但本轮未启动该量化档、未发生实抢占。自动量化质量执行器已接入并通过边界测试；尚无W8真机执行，因此当前不放行量化档。

所有HTTP请求禁代理；外部平台鉴权使用0600私有文件注入INF_API_KEY，明文不进入资格结果。生产管道只选择ready且30秒租约有效、HTTP boot与已验证boot一致的端点。

后续独立16/32并发实测由主控执行：两个HIGH各48/48请求、96图全部严格正确且auth/boot通过。Qwen并发16/32分别约71.75/88.99输出token/秒、1.04/1.29请求/秒；Gemma约40.08/40.61输出token/秒、0.581/0.588请求/秒。Gemma当前8序列配置已接近此探针负载的吞吐上限，继续增加客户端并发主要增加等待。证据为 `runtime/benchmark_16_32_20261023T020158/summary.json`；这些仍是短OCR任务，不能直接替代长caption生产测量。

最终计划切换：在19条DCVLM流程结束、9条正式记录/8图/1 layout已独立读回、冻结五层BF16回归基线后，先把Gemma指向已严格验收的LOW，再停止本任务Qwen HIGH32与Gemma HIGH4。平台确认两HIGH及其实例终态，释放其36卡；未操作其他任务。现Qwen LOW32仍在原队列等待、ready=false，Gemma LOW4在运行且ready=true；存在明确的Qwen等待窗口，不能称无缝切换。原LOW IDs保持不变，独立watchdog持续观察，待Qwen获资源后自动做新的boot/HCCL/图片资格。详见 `runtime/planned-high-release.json`、`runtime/post-high-release-live-20261023.json`。

上述32rank通信是本次BF16 HIGH的实测结果；16rank仅对应将来W8A8缩卡档的检查逻辑与单元测试，当前没有16rank量化实机成绩。19项monitor/恢复接口测试和13项quality worker测试通过，自动量化质量执行器已接入冻结基线，尚未启动W8或制造真实抢占。
