# 多模态 API 吞吐基准

本工具使用已提取的真实 DCVLM 图像，测量 API 阶段的整批耗时、响应延迟、失败与协议通过率。生成结果只用于基准和审计，不会加入训练库。

## 本次固定条件

- 按用户要求，后续测试仅调用 Qwen 系列。每个 caption 请求生成短、中、长三条英文描述。
- 显式关闭 thinking；检查接口报告的 reasoning tokens 与 thinking block 数量。非流式延迟是完整返回耗时，不是首 token 延迟。
- 凭证从 `../unified-mm/test_api.py` 通过 AST 安全读取；不执行示例、不记录密钥、不记录请求 headers 或图像 base64。使用直连，关闭环境代理。
- 默认 SDK 与应用重试均为零，失败和超时计入结果。请求具有唯一 ID；没有本地响应缓存，保留服务端缓存计数。
- 预加载及图像编码单独计时；API 整批时间从开始发请求到最后一次尝试结束，包含长尾。超时是整个请求的 wall deadline。
- 多于 manifest 图像数量的请求会显式循环样本，并在结果中记录重复数量；不能称为独立图像。

## 运行

在仓库根目录使用 `.venv/bin/python -m corpus_pipeline.api_benchmark.bench --help` 查看参数。每次使用新的输出目录。`--task caption` 生成三档描述，`--task review` 复核已有描述，`--task vision_probe` 检查带已知真值的图片。

运行产生 `run.json`、逐次 `requests.jsonl` 和 `summary.json`。原始协议通过要求 HTTP、返回模型标识、完整 JSON/schema 和 request ID 全部合格；caption 与 review 的协议通过均不等于事实正确。退出码 1 表示有请求失败；若完整 `summary.json` 已生成，该批次仍有有效的失败率测量。退出码 2 表示配置或执行错误。

原始输出必须保留。任何字段规范化只能单独报告 direct/repaired，不能覆盖原成功计数。样本是分层速度测试集，并非全库频率加权样本；短时吞吐外推只作为容量算例。

## 离线验证

`.venv/bin/python -m unittest discover -s corpus_pipeline/api_benchmark -t . -p 'test_*.py'`

测试使用本地 fixtures，不会调用付费 API。

本轮完整实测与解释见 [Qwen API 吞吐报告](../API_QWEN_THROUGHPUT_REPORT.md)。

离线重算已有记录（不调用 API）：

```bash
.venv/bin/python -m corpus_pipeline.api_benchmark.analyze_runs
.venv/bin/python -m corpus_pipeline.api_benchmark.audit_format_recovery --root corpus_pipeline/production/runs/api_throughput_20261023
```
