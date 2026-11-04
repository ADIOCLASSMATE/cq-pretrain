# Qwen3.6-35B-A3B Ascend W8A8 资产

本目录只负责 CPU 下载与权重完整性检查，不创建 NPU、Notebook 或 Serving。`WEIGHTS_READY.json` 必须存在且 `complete=true`，方可将目的目录交给部署程序；其含义是资产完整，不能替代当前模型 boot 的图片推理验收。

发布方为 ModelScope `Eco-Tech/Qwen3.6-35B-A3B-w8a8`，固定 commit `1a118d717bcbd59480f4a110fe22181d21711b4d`。下载目的目录为：

`/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Qwen3.6-35B-A3B-W8A8-1a118d71`

该目录位于本项目昇腾分区的共享模型盘；不假设 CPU 分区同名路径互通。下载进程在已有、能访问此路径的 CPU 主机运行，不申请平台资源。

官方 vLLM-Ascend 的 [v0.23.0rc1 部署文档](https://docs.vllm.ai/projects/ascend/en/v0.23.0rc1/tutorials/models/Qwen3.6-35B-A3B.html)引用这个发布方仓库，双卡示例使用 `--quantization ascend`。固定 tag 原文及 SHA 保存在 `OFFICIAL_VLLM_ASCEND_RECIPE.md` / `OFFICIAL_SOURCE.json`。这说明该量化格式有官方部署路径，不表示本项目已通过新部署的功能、质量或吞吐验收。

公共盘原有 `Qwen3.6-35B-A3B` 和 `Qwen3.6-35B-A3B-nomtp-ascend-FlagOS` 经实际 safetensors header 检查均为 BF16，无 Ascend 量化描述，故没有将它们冒称 W8A8。下载版本的 `config.json` 同样保留未量化层的 BF16 dtype；实际量化由 `quant_model_description.json` 与 INT8 tensor 共同证明，不能只看 config 的 dtype 或文件夹名称。

这个发布版本含 10 个 `quant_model_weights-*.safetensors`，以及量化描述、索引、processor/tokenizer、chat template 和来源说明。整个资产 39,810,885,219 字节。量化描述包含 92,251 项 `W8A8_DYNAMIC`、1,701 项 `FLOAT`；另有版本等元数据。十个 header 的 tensor 分布为：

| dtype | tensor 数量 | payload 字节 |
| --- | ---: | ---: |
| I8 | 30,750 | 32,400,998,400 |
| F32 | 61,500 | 252,395,520 |
| BF16 | 1,701 | 7,101,648,608 |

这些是 header/描述的实际统计；只有最终 READY 才同时证明全部 payload 的完整 SHA 已验证。它是混合保留浮点层的 Ascend `W8A8_DYNAMIC`，不是 FP8，也不是全部 tensor 都 INT8。发布方 README 将量化工具绑定至 msmodelslim commit `99394e9404ed72d811205d7a30a6ae883653200a`，并报告其 A3 环境测试；本目录不会把发布方的测试结果当作我们的 910B 双卡实测。

`SOURCE_MANIFEST.json` 固定仓库 commit、逐文件大小和发布方 SHA256。准备阶段已重新读取同一 immutable commit 的仓库清单并核对，响应保存在 `REPOSITORY_FILES.json`；下载脚本消费这份固定清单并使用带精确 Content-Range 的分块下载。分块落盘并 fsync 后写持久状态。恢复时要求 commit、文件 SHA、大小和分块大小一致。最终文件逐个重算完整 SHA256，通过后才从 `.part` 原子改名。

READY 另检查所有 safetensors header 与 index 一一对应、offset 连续且无截断/尾部多余字节、总 tensor bytes 与索引一致、存在真实 I8 weight 且逐项有 `W8A8_DYNAMIC` 声明、没有 FP8。`test_download.py` 使用独立微型离线夹具，覆盖正确 INT8 以及 BF16/FP8 冒充、缺 tensor、错误 shard、截断和索引大小不符。

`processor_revision` 由 9 个已核验的 config、processor、tokenizer 和模板文件的名称→SHA256 字典规范 JSON 再求 SHA256 得到；具体映射在 `PROCESSOR_IDENTITY.json` 和最终 READY。统一模型名建议 `Qwen3.6-35B-A3B-W8A8`，精度字符串为 `W8A8_DYNAMIC`；运行时权重 revision 应保留完整 ModelScope commit，不使用浮动 `master`。

下载/恢复命令（不会申请 NPU）：

```bash
.venv/bin/python corpus_pipeline/serving/qwen36_w8a8_20261024/weights/download.py --workers 24 --chunk-mib 64
```

同一目录有排他的下载进程锁；已有最终文件若 SHA 错误会拒绝继续，不静默覆盖。网络异常只记录类型，日志不打印重定向签名或凭据。发布 READY 后不得改写该目录文件来迁就运行时；需要调整资产时应建立独立版本与回执。
