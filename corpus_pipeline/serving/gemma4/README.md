# Gemma4-31B-IT 语料校验服务

模型权重已实际下载并逐文件校验；2026-10-23（Asia/Shanghai）HIGH4 和 LOW4 两个原生 Serving 均已通过真实 Ascend 视觉、严格 JSON、双图顺序与并发 1/4/8 canary。逐项证据和限制见 `VERIFIED_SERVING_20261023.md`。运行时是否仍可用以共享动态 registry 的当前 boot 资格为准。

- 官方模型：[`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it)，revision `842da3794eaa0b77d5f08bae87a17459d91ff475`，BF16。
- 持久资产：`/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Gemma4-31B-IT-842da379`。9 文件共 62,578,656,403 字节，`WEIGHTS_READY.json` 保存各文件大小与实际 SHA256。
- 下载移除大小写代理变量，HTTP 客户端 `trust_env=False`。官方 HF 直连 CDN 过慢后，改用[官方 Google ModelScope 镜像](https://www.modelscope.cn/models/google/gemma-4-31B-it)，两片权重与 tokenizer 的 LFS SHA256 必须匹配锁定的 HF 版本；其余元数据与已经直连取得的 HF 文件逐字节摘要匹配。按块恢复，最后完整文件 SHA256 校验后才原子发布就绪回执。
- processor revision 来自 `config.json`、`generation_config.json`、`processor_config.json`、`tokenizer.json`、`tokenizer_config.json`、`chat_template.jinja` 的实际文件 SHA256 规范列表，再对该列表取 SHA256。明细在 `processor_provenance.json`，启动时重新核验文件内容。
- 模型部署必须使用原生 **Serving**。当前为 HIGH4 与 LOW4 各一个 TP4 单副本，服务名分别为 `corpus-gemma4-serving-high-20261023-r01` 与 `corpus-gemma4-serving-low-20261023-r01`；时间按 Asia/Shanghai。HIGH priority6、LOW priority3 来自 live 项目与工作空间规则。8000 容器网关转发到 9000 单引擎，平台使用 `INF_API_KEY` 认证，内部引擎另用私钥。
- 两个 Serving 的精确请求原件、身份、端点、runtime 目录见 `serving_high_payload_20261023.json`、`serving_low_payload_20261023.json` 及各自 handoff。旧推理 Job 已停止、旧 Job 控制器停用，`prepare_job.py` 已禁止再提交。旧文件只作历史审计。
- 新 Serving 监控位于上层 `corpus_pipeline/serving/inspire_serving/`；HIGH 不绑定 LOW Job 调度器。Serving 不支持原生 `exclude_nodes`，实际落点检查不能冒充创建时节点排除。
- 候选镜像 `vllm-ascend:v0.30.0rc2` 来自 live 可见镜像目录。Gemma4 Ascend 官方指南提供 A2/A3/950、TP4/BF16/eager 基线，但不能替代实际容器验证。[官方 Ascend 指南](https://docs.vllm.ai/projects/ascend/en/main/tutorials/models/Gemma4.html)。

`serve.py` 在 `runtime/<deployment>/versions-node0.json` 记录实际 `vllm`、`vllm-ascend`、`torch`、`torch-npu`、`transformers` 版本、NPU 数量及本代 model/processor 身份；`node0.json` 每 3 秒更新。配置上限为 8 张图、32,768 token，图片处理使用 `max_soft_tokens=1120`；当前功能验收覆盖 1–2 张图，上限不等于已做满负载验收。每次启动产生新的 boot UUID，网关在推理响应头返回 `X-UMM-Boot-ID`，防止容器重启后复用旧视觉资格。

4 卡可能分配在不同 NPU plane，首次启动实际遇到 HCCL EI0010。`launch.sh` 现显式启用 `HCCL_INTRA_ROCE_ENABLE=1` 并选取容器默认 NIC/IP；两个自有 Serving 受控重启后完成真实推理。保持当前 BF16/TP4/eager 稳定配置，性能优化应在单独验证后更新资格。

真实端点分配后运行：

```bash
.venv/bin/python corpus_pipeline/serving/gemma4/canary.py \
  --base-url https://<actual-serving-domain>/v1 \
  --api-key-file corpus_pipeline/serving/private/inspire-serving.key \
  --expect-replicas 1 \
  --output corpus_pipeline/serving/gemma4/canary/<new-run>
```

canary 使用实际 DCVLM 图片，校验原图 SHA256，测试照片描述、三物体 CLEVR 的严格 JSON schema与两张不同图片的顺序。并发能力另外使用上层 `vision_probe.py --expected-replicas 1 --concurrency 1 4 8` 检查。另可加 `--thinking` 检查思考模式；日志只保留最终回答及是否存在 reasoning 的布尔值。HTTP 200 仅表示调用成功，需复核实际视觉回答、schema、图片顺序与并发响应，才能给该代服务合格结论。

下载完整性测试：

```bash
.venv/bin/python -m unittest corpus_pipeline.serving.gemma4.test_download -v
```

测试覆盖中断文件恢复、同大小篡改文件拒绝、错误 Content-Range 拒绝。测试通过不代表真实 NPU 推理通过。
