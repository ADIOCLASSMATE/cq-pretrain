# CQ Pretraining

主线使用 **Qwen3.5-2B-Base + Wan2.1-T2V-1.3B**。图像按单帧视频进入同一个 Wan 生成器，I2T、T2I、V2T、T2V 共用数据流、CQ 主干、训练循环和 checkpoint 协议。

## 模型约定

- 生成条件来自视觉 query，经可训练投影与 Wan 原生 text embedding 后注入 cross-attention。默认每张图／整段视频 **64 个** query。
- `model.image_condition_queries` 设置任意正整数；`visual_query_mode: native` 则按每个输入的最终视觉 token 数分配。不同目标可有不同 query 数量。
- I2T/V2T 及对话中的用户视觉输入只训练理解分支：保留 content，跳过生成 query、目标 VAE 与 Wan loss。数据中的图像 loss 禁用标记继续生效。
- 同一图像组 content 双向可见。视频按 Qwen 原生 temporal patch（相邻 2 帧）划分，块内双向、块间 causal。query 只读历史，不写入 KV、GDN recurrent 或卷积状态；生成 query 不读取自己的目标图像／视频。
- ViT **及 merger 默认冻结**；可缓存 merger 后的最终视觉特征。Wan VAE 使用冻结的 posterior mean 和原生归一化。
- Wan 使用原生 3D RoPE、Q/K 全宽 RMSNorm 与 FP32 时间调制。TP 时跨 rank 汇总 RMS 方差，不能用每个 head 的归一化代替。

| 目录 | 职责 |
|---|---|
| `models/` | Qwen、视觉编码、CQ 与统一 Wan；NPU kernel 适配 |
| `data/` | Parquet/JSONL 来源、时间戳、packing、缓存、异步设备预处理 |
| `training/` | TP/DP、分布式优化器、梯度调度、EMA 与训练循环 |
| `configs/` | 模型／训练／运行环境 YAML 与依赖锁 |
| `evaluations/` | I2T/T2I/V2T/T2V 定性评估与实测记录 |
| `utils/` | checkpoint、重分片、实验身份和运行协调 |
| `scripts/` | 启动、依赖准备、缓存准备与验证命令 |
| `tests/` | CPU 数值、语义、packing 和断点回归 |

`configs/legacy/` 与 MOSS 参考模块仅保留旧实验的数值回归；新生产入口使用 Wan 配置。9B/MOSS checkpoint 与此架构不兼容，需新开训练阶段。

## 环境与启动

在仓库根目录执行。CANN 9.0、PyTorch/torch_npu、Megatron/MindSpeed 的版本见 `configs/deps.core17.lock.json`、`environment.core17.txt`；Wan/视频处理的 Python 依赖见 `environment.wan.txt`。`scripts/launch.sh` 加载项目内 CANN、AscendC/FLA 和 Wan 依赖，训练过程不下载模型。

```bash
python -m scripts.setup_mindspeed --help
NPROC_PER_NODE=2 bash scripts/launch.sh \
  --config configs/qwen35_2b_wan.yaml \
  --data-root /path/to/published-corpus \
  --output /path/to/run --until-data-end
```

`configs/qwen35_2b_wan.yaml` 集中定义 model、training、environment。CLI 参数覆盖 YAML 的 training 设置。模型路径可在 model 中改写；必须提供经过校验的 **2B-Base**，不能用 2B 对话权重代替。

多节点／低优抢占恢复继续使用 `train_production.py` 与 `launch_job.py`：每次启动新 rendezvous，恢复 optimizer、FP32 masters、renderer EMA、RNG 和已消费的数据游标。完整模型加 Adam、EMA 的单卡测试在首次 optimizer 分配时超出 64 GiB；默认采用 TP2，资源预算需按实际序列／视频长度验证。

生产入口要求显式传入 `--data-root`，避免误用已退役的 small-pool 索引。MC2 只在本机 TP 组具有完整 HCCS 互联时启用；PHB 碎卡自动使用普通 TP 通信。环境变量可覆盖 YAML 中的运行环境默认值。

## 数据与缓存

已发布的 DCVLM/GenUND Parquet 目录直接作为 `--data-root`。预处理仍使用 `python -m corpus_pipeline`，规则见 [数据预处理说明](corpus_pipeline/README.md)。DCVLM 保留原生任务结构，不重新生成 T2I50。

也可指定 JSONL 或混合来源 YAML：

```yaml
sources:
  - path: /path/to/published-corpus
  - path: videos.jsonl
    video:
      mode: native
      qwen_num_frames: 8
```

JSONL 简单记录使用 `task: t2i/i2t/t2v/v2t`、`caption` 和 `image_path`／`video_path`；NPY 视频使用 `frames_path` 与 `fps`。也支持原生 `messages` 和 `videos` 数组。媒体路径相对 JSONL 解析。视频保留解码 PTS；native 模式仅将末尾裁到 Wan 的 4k+1 帧，空间尺寸必须满足原生 patch 对齐。需要固定抽帧／裁剪时，在 source.video 中设置 `mode: fixed`、`num_frames`、`fps`、`height`、`width`。

```bash
# 可用 --rank / --world 将相同来源分给多个 CPU/NPU 实例。
python -m scripts.prepare_latents --config configs/qwen35_2b_wan.yaml \
  --data-root /path/to/sources.yaml --cache /path/to/visual-cache --device npu:0
```

训练时设置 `model.visual_cache` 为同一目录。缓存身份包含来源内容 hash、真实帧／时间戳与预处理几何、编码器身份；每个条目有 tensor 校验和并原子发布。重跑跳过已有条目；缺失条目在线计算。缓存独立于原始数据，复制整个数据目录后不依赖旧绝对路径；改变数据、编码器或处理几何会使用新条目。原始文件不被改写。

保留 main 的 locality shuffle、批量 tokenize、有限预取、完整文档 packing、consumed-only 游标、H2D/VAE 独立 stream、全 DP/GA 目标归一化、DiT 分块反向与单次 CQ 梯度桥接、固定 collective 顺序、FP32 fused weight-gradient、MC2、激活重算和 renderer-only EMA。Wan attention 使用紧凑 TND 布局，不构造整张 padding mask。

## 验证与评估

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 .venv/bin/python -m pytest -q tests
NPROC_PER_NODE=1 PRETRAIN_MODULE=scripts.benchmark_gdn bash scripts/launch.sh --output /path/to/gdn.json
NPROC_PER_NODE=2 PRETRAIN_MODULE=scripts.validation.validate_wan_tp PRETRAIN_MC2_ROW=1 \
  bash scripts/launch.sh --output /path/to/wan-tp.json
```

Ascend 910B 图像 CQ 的 GDN 层前向加反向实测（含 warmup，12 次计时中位数）：

| Content 长度 | 自定义 AscendC | 官方 MindSpeed |
|---|---:|---:|
| 2048 | 10.46 ms | 15.72 ms |
| 8192 | 38.60 ms | 56.60 ms |

当前默认 `gdn_backend: ascendc`；保留 `mindspeed` 对照。这里只比较 GDN recurrence 提供方，卷积／门控归一化、输入、参数和 CQ 事件计划相同；不代表整模型吞吐。原始记录在 `evaluations/benchmarks/`。

统一路径已通过 126 项 CPU 回归（另有 46 个子测试）、官方 Diffusers 前向／反向对照、NPU BF16 与 TP2 梯度对照、完整权重两步训练、断点续训、四任务评估、缓存重跑及不均匀 DP 目标检查。完整权重测试使用 128 token 和小尺寸媒体，单卡峰值 33.75 GiB；尚未据此承诺正式长序列吞吐。TorchAIR 调制融合已适配 Wan 的 FP32 语义并通过数值检查，默认仍关闭。

训练支持 `--validation-suite` 的固定定性样例；T2V 加 `num_frames`、`fps`，V2T 指定视频路径与校验值，视频输出为 MP4。结果保留 suite hash、权重来源、耗时、显存和 HTML。此套件不声称提供官方 benchmark 分数。

Wan 默认采样 `guidance: 1`；此次 I2T 不再提供 null 图像生成训练目标，因此不沿用旧 MOSS 的 CFG=3 默认值。已有独立 null 条件训练的实验可在评估样例中显式设置 guidance。

每 1000 个绝对更新默认保存独立推理快照（`--eval-snapshot-every`）；训练恢复 checkpoint 的频率、保留策略和完整性校验独立管理。`--eval-snapshot-dir` 指定持久评估权重目录，零间隔关闭快照。
