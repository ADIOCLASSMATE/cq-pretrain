# CQ Pretraining

主线使用 **Qwen3.5-2B-Base + Wan2.1-T2V-1.3B**。图像按单帧视频进入同一个 Wan 生成器，I2T、T2I、V2T、T2V 共用数据流、CQ 主干、训练循环和 checkpoint 协议。

## 模型约定

- 生成条件来自视觉 query，经 **2048 → 1536 → 1536** 两层 adapter（中间 GELU）注入 Wan cross-attention。沿用 Wan 的两层 text embedding 结构，删除额外的 2048→4096 投影。加载原生 Wan 时只重新初始化第一层输入矩阵，保留该层 bias 和第二层权重。
- 默认 `visual_query_mode: native`：每个生成目标的 query 数等于 Qwen merger 输出 token 数，即 `grid_t × grid_h × grid_w / 4`。例如处理后 512×512 图像对应 256 个 query；不同目标使用各自长度。`visual_query_mode: fixed` 时由 `image_condition_queries` 指定数量。
- I2T/V2T 及对话中的用户视觉输入只训练理解分支：保留 content，跳过生成 query、目标 VAE 与 Wan loss。数据中的图像 loss 禁用标记继续生效。
- `content_visibility: visual_block`（默认）让同一图像组 content 双向可见。视频按 Qwen 原生 temporal patch（相邻 2 帧）划分，块内双向、块间 causal。query 只读历史，不写入 KV、GDN recurrent 或卷积状态；生成 query 不读取自己的目标图像／视频。`content_visibility: causal` 使用逐 token causal；两种模式共享同一规划和执行路径。Wan DiT 始终在整段 clip 内双向可见。
- ViT **及 merger 默认冻结**；可缓存 merger 后的最终视觉特征。Wan VAE 使用冻结的 posterior mean 和原生归一化。
- Wan 使用原生 3D RoPE、Q/K 全宽 RMSNorm 与 FP32 时间调制。TP 时跨 rank 汇总 RMS 方差，不能用每个 head 的归一化代替。

| 目录 | 职责 |
|---|---|
| `models/` | Qwen、视觉编码、CQ、统一 Wan 与推理缓存 |
| `ops/` | attention/GDN/CE 算子、算子布局与 TorchAIR 编译 |
| `distributed/` | Megatron 运行时、TP、MC2、参数聚合与梯度同步 |
| `data/` | Parquet/JSONL 来源、时间戳、packing、视觉预处理、CQ 规划、缓存和异步设备预处理 |
| `training/` | 目标加权与 DP 归一化、DiT 分块反向、梯度调度、EMA 与训练循环 |
| `configs/` | 模型／训练／运行环境 YAML 与依赖锁 |
| `evaluations/` | I2T/T2I/V2T/T2V 定性评估与实测记录 |
| `utils/` | checkpoint、重分片、实验身份和运行协调 |
| `scripts/` | 启动、依赖准备、缓存准备与验证命令 |
| `tests/` | CPU 数值、语义、packing 和断点回归 |

核心模块按职责命名：

- `models/cq_backbone.py`、`models/cq_layout.py`、`models/vision_encoder.py`、`models/decoding.py` 分别负责 CQ 执行、布局定义、视觉编码与增量解码。
- `ops/functional.py`、`ops/attention.py`、`ops/gated_delta_net.py`、`ops/cross_entropy.py` 放算子；`attention_layout.py`、`gated_delta_layout.py` 放对应布局，`torchair.py` 放编译实现。
- `distributed/megatron.py`、`distributed/communication.py`、`distributed/mc2.py` 分别负责 Megatron 接入、通信调度与计算通信融合。
- `utils/dependencies.py` 管理依赖环境，`utils/file_sync.py` 管理文件同步。

`configs/legacy/` 与 MOSS 参考模块仅保留旧实验的数值回归；新生产入口使用 Wan 配置。9B/MOSS checkpoint 与此架构不兼容，需新开训练阶段。

`JointModel.forward` 返回未加权的 `text_loss`、可选 `image_loss`；`compute_image_loss=False` 返回公开的 `image_conditions`。`training.loss.compose_losses` 组合目标，`training.renderer_schedule` 调度分块反向；模型不读取任务采样概率、loss 权重或 DP 缩放值。`ops/` 和 `distributed/` 不导入数据、模型或训练循环；模型也不导入数据／训练模块。CQ 布局定义在 `models/cq_layout.py`，CPU 批次布局由 `data/batch_layout.py` 构建。

## 环境与启动

在仓库根目录执行。CANN 9.0、PyTorch/torch_npu、Megatron/MindSpeed 的版本见 `configs/deps.core17.lock.json`、`environment.core17.txt`；Wan/视频处理的 Python 依赖见 `environment.wan.txt`。`scripts/launch.sh` 加载项目内 CANN、AscendC/FLA 和 Wan 依赖，训练过程不下载模型。

```bash
python -m scripts.setup_mindspeed --help
NPROC_PER_NODE=2 bash scripts/launch.sh \
  --config configs/qwen35_2b_wan.yaml \
  --data-root /path/to/published-corpus \
  --output /path/to/run --until-data-end
```

`configs/qwen35_2b_wan.yaml` 按 `model`、`data`、`loss`、`runtime`、`training`、`environment` 分组。`RunConfig` 由有明确职责的 dataclass 组成，保留直接字段访问；旧的平面 JSON 配置仍可读取。CLI 参数覆盖 YAML 的 training 设置。模型路径可在 model 中改写；必须提供经过校验的 **2B-Base**，不能用 2B 对话权重代替。

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

训练时设置 `data.visual_cache` 为同一目录。缓存身份包含来源内容 hash、真实帧／时间戳与预处理几何、编码器身份；每个条目有 tensor 校验和并原子发布。重跑跳过已有条目；缺失条目在线计算。缓存独立于原始数据，复制整个数据目录后不依赖旧绝对路径；改变数据、编码器或处理几何会使用新条目。原始文件不被改写。

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

本次 main 重构通过 **156 项 CPU 测试＋46 个子测试**，包括两种可见性、缓存解码、官方 Diffusers 梯度对照和新 adapter 加载。16 卡 TP2×DP8 已通过 Wan 前后向、不均匀分块反传、推理缓存，以及小模型训练恢复的权重／FP32 优化器／EMA／RNG 逐位一致性检查。完整预训练权重的 8K 训练 step 对照：I2T 1.628→1.629 秒，joint 1.952→1.946 秒，速度基本持平；joint 峰值显存 16.50→16.43 GiB。计时使用驻留的合成视觉特征与 latent，不包含数据 I/O、ViT/VAE 编码；记录见 [本次验证](evaluations/benchmarks/main_infra_refactor_20261105.json)。

重构前的统一路径曾通过 CPU 回归、官方 Diffusers 前向／反向对照、NPU BF16 与 TP2 梯度对照、完整权重两步训练、断点续训、四任务评估、缓存重跑及不均匀 DP 目标检查。完整权重测试使用 128 token 和小尺寸媒体，单卡峰值 33.75 GiB；尚未据此承诺正式长序列吞吐。TorchAIR 调制融合已适配 Wan 的 FP32 语义并通过数值检查，默认仍关闭。

训练支持 `--validation-suite` 的固定定性样例；T2V 加 `num_frames`、`fps`，V2T 指定视频路径与校验值，视频输出为 MP4。结果保留 suite hash、权重来源、耗时、显存和 HTML。此套件不声称提供官方 benchmark 分数。

I2T 全部保留，仅监督原始文本，不创建视觉生成 query、不运行目标 VAE/DiT。原生单／多轮 T2I 和自然交错文档也全部保留，监督原始文本及允许生成的图像；输入角色的图像和已有 `image_loss_allowed=false` 标记仍不作为生成目标。原始文本包含用户轮和助手轮，人工添加的模板不计文本 loss。

原生 T2I 的每个生成目标按 `visual_uncond_probability: 0.1` 使用空条件，其余 90% 使用正常前缀，图文 loss 都保留。自然交错文档不做随机 CFG dropout：`I₁ T I₂` 中 `I₁` 无前缀，使用空条件；`I₂` 读取之前的 `I₁ + T`。`T I₁ T I₂` 则两张图都按各自真实前缀生成。原生视频生成目标沿用相同的 10% 规则。

模型只接收生成目标索引和 `Batch.generation_unconditional` 标记，不判断 T2I、交错顺序或采样比例。标记为 True 的视觉 Q 在每层不读取任何 C（空卷积历史、零 GDN 状态、无 attention 读取），保留自身残差和 MLP，输出经 Wan cross-attention 作为 DiT condition；False 或未提供标记时使用正常可见性。标记不改变目标位置、RoPE、query 数量或其他目标的计算，图像与视频共用实现。布局在 CPU 预先准备，逐层 forward 不再构造 mask／事件布局，也不判断数据任务。

数据侧允许在样本或单个视觉目标设置布尔值 `non_cfg`，目标标记优先；显式 True/False 均覆盖默认采样策略。未指定时，数据侧按上述原生 10%／交错真实前缀规则生成标记；I2T 始终不创建生成目标。原生随机选择以样本、图像位置、seed 和 epoch 固定，可精确恢复。评估不随机选择，`make_batch(..., non_cfg=True)` 与 `cfg_null_condition: empty` 的负分支使用同一路径。采样 `guidance` 仍由评估样例指定，默认 1。

query 数量和空条件策略本身不改变参数形状；本次两层 adapter 改变了权重与优化器形状，旧训练结果不再兼容：训练 checkpoint 仅接受 schema 7，推理快照仅接受 v2；不提供旧三层 adapter 转换。新 checkpoint 的精确 `--resume` 同时校验配置、数据游标、源码和实际启用的 kernel 选项。dynamic query 的语义、梯度、交错前缀及恢复规则有 CPU 回归覆盖，原固定 64 query 的 NPU 吞吐数字不能直接当作新配置的测速结果。

每 1000 个绝对更新默认保存独立推理快照（`--eval-snapshot-every`）；训练恢复 checkpoint 的频率、保留策略和完整性校验独立管理。`--eval-snapshot-dir` 指定持久评估权重目录，零间隔关闭快照。

## video 共享执行层

video 的通用算子和分布式执行已经接入 main 的实现，保留视频的数据、原生时间戳、任务归一化和实验管理。

| 共享部分 | video 接入位置 |
|---|---|
| CQ 层、融合投影、Q 卷积、AscendC/MindSpeed GDN、双向时间块 | `models/qwen35_layers.py`；`backbone_qwen35.py` 只适配视频输入和布局 |
| Wan 层、TND attention、FP32 时间层、两层 adapter | `models/wan_layers.py`；`generator_wan21.py` 适配变长视频与采样 |
| attention/GDN/CE、CPU 布局、编译与通信 | `ops/`、`distributed/` 与 main 保持相同实现 |
| 分布式 Adam、FP32 梯度、参数聚合和梯度同步顺序 | `training/runtime.py` 调用共享运行时，保留 video 的参数分组、DP/GA 归一化和恢复生命周期 |

video 保留 Qwen3.5-2B 初始权重；main 继续使用 Base。双方均使用直接的 2048→1536→1536 adapter，拒绝旧参数结构和旧训练结果。video 默认 AscendC GDN、重入式 checkpoint；backbone 原生时间块内双向、块间 causal，Wan 整个 clip 双向。

Wan 条件投影和 cross-attention K/V 已按请求缓存。video 的 V2T 文本生成仍走原生前缀接口，未接入 main 的逐 token KV/GDN 缓存；视频训练拓扑仍为 DP，TP=PP=CP=1。数据 packing、DiT 分块反向和任务采样策略属于各自训练流程。

此次共享层迁移通过 main 157 项 CPU 测试及 51 个子测试、video 270 项 CPU 测试，以及 16 卡算子、混合任务训练和精确恢复检查；完整 Qwen3.5-2B + Wan1.3B 权重完成两次更新，共享 Wan 的 TP2 前后向对照通过。结果及共享源码摘要见 [迁移验证报告](evaluations/benchmarks/video_shared_runtime_20261105.json)，详细说明在 video 分支的 `docs/validation.md`。完整权重测试使用短合成视频，不包含真实数据 I/O、VAE 编解码或长视频吞吐评测；main 的历史速度数据也不能代替 video 的端到端吞吐。
