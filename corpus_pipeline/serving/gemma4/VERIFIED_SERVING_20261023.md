# Gemma4 原生 Serving 实测验收

2026-10-23 01:45（Asia/Shanghai）记录：`Gemma4-31B-IT` BF16 已在 Ascend 910B2C 上完成真实推理，HIGH4 和 LOW4 均为原生 Serving、TP4 单引擎。两者的模型资产、启动选项相同；HIGH 用于当前流程，LOW 为已验证的后备服务。资格属于当前 boot，后续重启必须重新验收。此记录不代表完整 DCVLM 数据管道已处理完成。

| 项目 | HIGH | LOW |
|---|---|---|
| 服务 | `corpus-gemma4-serving-high-20261023-r01` | `corpus-gemma4-serving-low-20261023-r01` |
| 平台 ID | `sv-c4d1aea3-e2b1-4170-82f7-ee87715295a1` | `sv-48788dbf-f3b7-41e0-8265-88b8b4cb8939` |
| priority / 卡数 | 6 / 4 | 3 / 4 |
| 当前 boot | `9a2bf8dbd9c24b5c89015255aaf287a7` | `9c0d5a9b229c4cc8ad2f84865a5fb216` |
| 真实照片描述 | 2.015 秒 | 2.129 秒 |
| CLEVR 严格 JSON | 3.019 秒 | 3.361 秒 |
| 双图顺序 JSON | 5.314 秒 | 5.474 秒 |

实际镜像 `vllm-ascend:v0.30.0rc2` 中，vLLM 为 `0.30.0+empty`，vllm-ascend 为 `0.30.0rc2`，torch 为 `2.10.0+cpu`，torch-npu 为 `2.10.0.post7`，transformers 为 `5.14.1`。torch 的 cpu 版本字符串未妨碍 torch-npu 扩展实际使用 4 张 Ascend910B2C。模型架构实际解析为 `Gemma4ForConditionalGeneration`，权重为固定 revision `842da3794eaa0b77d5f08bae87a17459d91ff475`，并非量化替代。

## 已通过的功能与质量边界

真实输入来自 DCVLM 抽样原图，调用前重新验证图像 SHA256。摄影图包含多只陶瓷杯和饮品；CLEVR 图准确含两个圆柱、一个球。Gemma 返回严格 JSON `{"object_count":3,"shapes":["cylinder","cylinder","sphere"]}`。双图输出正确保持杯子在先、几何图在后，无串图。HIGH、LOW 三条最终回答逐字相同，全部 `finish_reason=stop`，返回 model 均为请求的 `Gemma4-31B-IT`。

已逐张查看两幅原图并核对输出。模型对主要物体和数量判断正确，但把青色圆柱粗称为 blue，把杯中饮品概括为不同 coffee。后者无法仅靠图片证明具体饮品种类。因此这组结果证明视觉、JSON 与多图功能可用，不能证明所有 caption 细节准确，也不能用于判断图像是否 AI 生成。正式复述与审阅仍需保留不确定性、细节核查和人工兜底。

真实图结果及人工核对分别在：

- `canary/native_high_real_dcvlm_20261023_r01/{receipt,visual_review}.json`。
- `canary/native_low_real_dcvlm_20261023_r01/{receipt,visual_review}.json`。

LOW 的新 canary 持久化了每次响应的 `X-UMM-Boot-ID`，与调用前后 `/health` boot 完全一致。HIGH 真实图 canary 的旧脚本未逐响应存该头；共享控制器的独立视觉资格已绑定相同 HIGH boot，并保留了逐请求证据。

## 并发实测

共享控制器的随机双图数字/颜色严格 JSON 测试，在两个服务均 13/13 请求通过。以下引用每个服务 `attempt02/result.json`，每个请求约 2,236 输入 token、69 输出 token，不能直接当成长 caption 的吞吐。

| 并发 | HIGH 总输出 token/s | LOW 总输出 token/s | HIGH req/s | LOW req/s |
|---:|---:|---:|---:|---:|
| 1 | 11.50 | 12.24 | 0.167 | 0.177 |
| 4 | 29.17 | 28.24 | 0.423 | 0.409 |
| 8 | 37.95 | 38.73 | 0.550 | 0.561 |

证据位于上层 `inspire_serving/runtime/probes/corpus-gemma4-serving-{high,low}-20261023-r01/` 对应当前 boot 资格目录。初始 cold/warm 阶段的不同 attempt 有不同性能，不应混合求平均或夸大稳定吞吐。当前为 eager；未为跑分改动已验证 HIGH。8 图及完整 32,768 token 是配置允许上限，此次功能 canary 尚未覆盖这两个极限。长文审阅与最大输出预算必须另看正式 pipeline 实测。

## 真实故障与修复

最初 4 卡分配跨 NPU plane，HIGH、LOW 都在加载权重后遇到 HCCL EI0010，实际错误明确要求 `HCCL_INTRA_ROCE_ENABLE=1`。`launch.sh` 已设置该选项，并从容器默认路由选取 NIC/IP，配置 HCCL/Gloo/TP 网络接口。对精确归属的两个 Serving 各执行 stop、确认 STOPPED、再 start；没有创建重叠服务，没有停止无关任务。重启后的日志及以上真实推理证明这次通信问题已解决。历史错误仍保留在追加日志中，排查时需用 boot 时间区分。

## 代码、配置和身份证据

`runtime/<deployment>/versions-node0.json` 保存真实包版本、processor hash、运行选项、通信环境及 boot。两个服务还通过容器内只读 shell 运行 `attest_running.py`，把实际 supervisor/engine/gateway 进程 argv、允许列表环境变量及源文件 SHA 保存为：

- `runtime/gemma4-serving-high-20261023-r01/live-attestation-9a2bf8dbd9c24b5c89015255aaf287a7.json`。
- `runtime/gemma4-serving-low-20261023-r01/live-attestation-9c0d5a9b229c4cc8ad2f84865a5fb216.json`。

四文件 `config.json`、`gateway.py`、`launch.sh`、`serve.py` 的 mtime 均早于两个实际 boot，engine 真实 argv 与启动回执逐项一致，两个子进程的 boot 环境也与 heartbeat 相同。共同代码摘要为 `sha256:2e76d7cb7e6be5f6a078b66ae87aa00e3398173fb5821af57e8e8f02d6d35355`。这证明当前挂载代码和本 boot 的关系，但不是签名容器镜像证明。独立 receipt 未修改当前 versions 或消费者 cache identity，避免正在处理的生产任务中途改变缓存身份。

API 使用平台原生 Bearer 认证，客户端通过 `INF_API_KEY` 或受限私钥文件读取，不在源码和回执记录密钥。HTTP 客户端 `trust_env=False`。平台 Serving 没有原生 `exclude_nodes` 参数，节点落点检查不能替代创建排除。新 Serving 控制器负责持续健康观察和 boot 变化撤销资格，当前运行状态应以动态 registry 为准。
