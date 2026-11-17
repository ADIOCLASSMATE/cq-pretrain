CUDA / L40S 移植验证，2026-11-17

原任务指定的 16-head GDN 门禁、CPU 回归、完整模型 TP1/TP2/DP2、两节点 16 卡训练及清洁进程恢复均已完成。另一个独立随机初始化的 8-head 压力用例存在未解决的精度超差，完整失败证据随报告保留，不能将本报告解释为所有额外精度测试通过。

代码位于 `cq-pretrain-l40s` 的 `l40s` 分支。多节点实际执行的 91 个代码/配置文件与 commit `1f18c0ed9c9e79185fec5e52ba5c90a880c9504d` 的 SHA256 一致，报告产物目录不参与这一代码一致性判断。提交均带用户要求的 co-author trailer，未 push。

只接入上游 FLA、PyTorch SDPA、NVIDIA Apex fused wgrad、Megatron-Core 和 NCCL，没有编写 GPU kernel。CPU 计划文件、全部 configs 和 YAML 的 diff 为空；整个 Qwen backbone 的 AST 除 CUDA dispatch 外与 `1545ef7` 一致。具体哈希见 `invariants.json`。

运行时复用 NGC 25.02 镜像的 PyTorch 2.7.0a0+ecf3bae40a.nv25.02 / CUDA 12.8 / NVIDIA Triton 3.2.0 / Apex，额外依赖由 CPU 预下载 wheelhouse 离线安装，FLA 0.4.0、MCore 0.17.0 固定版本。共享 venv 和 Triton cache 供所有节点复用。环境明细见 `gpu-runtime.json`；安装命令见仓库 `scripts/CUDA.md`。

所有新建 Notebook 和 Job 都显式提交 `--priority 3`，平台回读 LOW。Notebook priority 为 3；Job status 暴露的内部 priority 为 13，提交 JSON 仍为 3。两节点 Job 为 `cq-cuda-2n16-p3-1117`，每节点 quota `8,210,800`、共享内存 128 GiB，自动重试最多 3 次、间隔 60 秒，最长运行 1 小时。任务状态 SUCCEEDED，两实例均 instance_success。

CPU 使用 PyTorch 2.8.0+cu128，无 torch_npu。完整输出：

```text
..................................................s................................................ [ 61%]
.............s.............................................s.        [100%]
=========================== short test summary info ============================
SKIPPED [1] tests/test_document_packing.py:202: Pinned MCore source unavailable
SKIPPED [1] ../unified-mm/.venv/lib/python3.12/site-packages/_pytest/unittest.py:523: Pinned official MOSS source is not installed
SKIPPED [1] tests/test_wan_unified.py:199: Local native tokenizer fixture required
157 passed, 3 skipped, 49 subtests passed in 39.74s
```

`tests/test_qwen35_cq.py` 的 5 项测试包含在上述完整回归中；单独基线输出保存在 `cpu-cq-before.log`，更新后的 CQ/backend/CE 小组合为 `cpu-cuda-final.log`（9 passed）。三个跳过项是缺少固定 MCore 源码夹具、官方 MOSS 源码夹具及本地 native tokenizer 夹具，未将其算作通过。

GDN 对照 FP64 CPU recurrence，包含完整 GatedDelta 层的输出、输入梯度和所有参数梯度，阈值保持 0.035。RMSNorm 输出/dx/dw 相对误差分别为 0.00167716 / 0.00232190 / 1.14917e-7。

| 模式 | 长度 | 最大相对误差 | query 隔离 max abs |
|---|---:|---:|---:|
| CQ_IT | 61 | 0.009172577 | 0.0 |
| CQ_IT | 64 | 0.010426376 | 0.0 |
| CQ_IT | 65 | 0.009410594 | 0.0 |
| Bi_I+CQ_T | 61 | 0.007980112 | 0.0 |
| Bi_I+CQ_T | 64 | 0.007888833 | 0.0 |
| Bi_I+CQ_T | 65 | 0.008758094 | 0.0 |

把同一完整 16-head 测试层的 FLA 调用拆为两个 8-head 调用，结果也全部通过，最大误差与上表相同，见 `gdn_tp2_kernel_shards.json`。这是单 GPU 上的 TP 局部 kernel 形状验证；实际分布式 TP2 在独立双卡训练中验证。

额外的 `--key-heads 8 --value-heads 8` 重新随机初始化了一个不同的完整测试层。在 `Bi_I+CQ_T / length=64`，A_log / dt_bias 梯度误差为 0.1138772 / 0.1035430，超过 0.035；纯 CPU BF16 recurrence 在同一输入上分别为 0.0512694 / 0.0447439，CPU FP32 为 2.309e-6 / 3.086e-6。BF16 舍入贡献了误差，FLA 的额外误差仍未消除。没有放宽阈值、修改 seed 或隐藏失败。详见 `gdn_8head_precision_failure.json` 和 `bf16_rounding_diagnostic.txt`。

融合 wgrad 两块累加对 FP64 的误差 5.86055e-8；CUDA CE 普通/直接 main_grad 累加最大误差 0.00230950；attention 布尔屏蔽、空 query 行、输出及 Q/K/V 梯度最大误差 0.00390954。完整值见 `operators.json`。

训练使用固定官方 Qwen3.5-2B-Base + Wan2.1-T2V-1.3B 预训练权重，revision 和校验哈希见 `assets.json`。模型共 3,629,092,736 参数，其中 3,297,676,160 可训练；ViT/merger 按原配置冻结。保留原 YAML，通过 CLI 指定 synthetic、序列 2048、每个 microbatch 一个图像、短步数和 EMA=0。fake pixels/latents 不覆盖真实语料 I/O 或 VAE 编码。这是短训练验证，不是完整 8192 上下文生产吞吐。

| 布局 | 成功更新 | tokens/step | 预热后秒/步 | tokens/s | 最大 allocated GiB |
|---|---:|---:|---:|---:|---:|
| 1 GPU TP1 + CPU optimizer offload | 3 | 2048 | 3.3664 | 608.36 | 36.3897 |
| 2 GPU TP2/DP1 | 5 | 4096 | 1.8076 | 2265.98 | 31.2773 |
| 2 GPU TP1/DP2 | 5 | 4096 | 1.4196 | 2885.32 | 42.5636 |
| 2 节点 TP1/DP16 | 4 | 32768 | 1.4994 | 21854.05 | 26.4675 |

双卡比较使用相同 global batch=2 / 4096 tokens，TP2 的 GA=2，DP2 的 GA=1；均为一个预热 step 加四个计时 step。TP1/DP2 吞吐高 27.33%，显存更高，因此本次多节点使用 TP1/DP16。TP1 单卡为容纳完整训练状态使用 MCore 原生 optimizer offload，其速度不与纯 GPU 布局混为同一口径。

16 卡测量为一个预热 step 加三个计时 step。表中是训练更新时间；第 4 步 checkpoint 用时 13.8216 秒，本次含保存开销的计时段吞吐为 5361.09 tokens/s。保存 17 个文件、46,866,117,790 字节，实际大小与 manifest 完全一致。

随后所有训练进程退出并重新启动，从第 4 步恢复 model、optimizer、FP32 masters 和 RNG，16 个 rank 均完成第 5 步，且没有重新加载 base 预训练权重。恢复后的首步包含重新预热，不作为稳定吞吐。平台自动容错已启用；这里实际验证的是清洁进程重启，未注入节点故障，也未验证长程精确数值重放。

| rank | 节点 | step 4 loss | 恢复后 step 5 loss | 最大 GiB |
|---:|---|---:|---:|---:|
| 0 | qb-prod-y40s-gpu041 | 1.33729243 | 1.28141975 | 26.467450 |
| 1 | qb-prod-y40s-gpu041 | 1.30773354 | 1.25885355 | 26.467454 |
| 2 | qb-prod-y40s-gpu041 | 1.29494047 | 1.24512637 | 26.467451 |
| 3 | qb-prod-y40s-gpu041 | 1.36169183 | 1.30421376 | 26.467453 |
| 4 | qb-prod-y40s-gpu041 | 1.23763967 | 1.19610238 | 26.467451 |
| 5 | qb-prod-y40s-gpu041 | 1.30389154 | 1.25435257 | 26.467453 |
| 6 | qb-prod-y40s-gpu041 | 1.32941735 | 1.27660787 | 26.467453 |
| 7 | qb-prod-y40s-gpu041 | 1.29117072 | 1.24312162 | 26.467450 |
| 8 | qb-prod-y40s-gpu199 | 1.25172150 | 1.20823860 | 26.467450 |
| 9 | qb-prod-y40s-gpu199 | 1.31192780 | 1.26176071 | 26.467451 |
| 10 | qb-prod-y40s-gpu199 | 1.24559307 | 1.20563126 | 26.467451 |
| 11 | qb-prod-y40s-gpu199 | 1.35758615 | 1.30181110 | 26.467453 |
| 12 | qb-prod-y40s-gpu199 | 1.29647171 | 1.24816251 | 26.467453 |
| 13 | qb-prod-y40s-gpu199 | 1.29799092 | 1.24925756 | 26.467451 |
| 14 | qb-prod-y40s-gpu199 | 1.34114707 | 1.28714705 | 26.467454 |
| 15 | qb-prod-y40s-gpu199 | 1.31402934 | 1.26457131 | 26.467431 |

全部 80 条成功更新记录在 `multinode_steps.jsonl`，每条含 rank、hostname、step、loss、梯度范数、optimizer_updated、耗时、显存。`multinode/` 和 `multinode-resume/` 保留原始 run.json、summary.json、每 rank JSONL 和 backend provenance。`acceptance.json` 记录独立复核结果，显式包含额外 8-head 门禁失败。

观察按 Events → Logs → Metrics → Instances/Status 执行，随后各保留一个阻塞 CLI job wait，终态后验收。两节点平台 GPU 指标峰值分别为 46.5% 和 72.5625%（一分钟采样会漏掉短步峰值）。CLI `job logs --all` 仍由平台截成 1000/3971 条，`multi-logs-final.json` 明确带 truncated=true；共享盘的 80 条每-rank 更新记录完整，验收不依赖这份截断的平台日志。

精确命令、GPU 任务提交及等待命令保存在 `commands.txt`；训练正文分别是 `tune-two-gpu.sh` 和 `multinode.sh`。dry-run JSON、平台终态、Events/指标和 checkpoint manifest 一并保留。本次及旧开发 Notebook 已停止/删除，连接缓存已清理；Job 已终态，代码、共享权重和第 4 步 checkpoint 保留，未改动其他任务。
