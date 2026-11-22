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

六源混训使用 `--data-root data/imagenet_gqa_blip3o_intersyn_climbmix.yaml`。默认容量为 4,096，打乱后的文档按流中顺序装包，文档之间隔离 attention。装包后按有效长度向上对齐到 2,048 或 4,096。用 `--sequence-length 8192` 可启用 8,192 档。

显式标注图像生成目标的 DCVLM/GenUND Parquet 发行目录也可作为 `--data-root`。预处理入口为 `python -m corpus_pipeline`。[数据预处理说明](corpus_pipeline/README.md)

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

原生视觉训练发行版在每次图像出现的位置用 `generation_target` 指定生成监督。ImageNet I2T 和 GQA 的图像提供上下文，文本 CE 覆盖原始语料中允许监督的位置和终止／起图标记。ImageNet T2I、BLIP3o 和 InterSyn 的目标图参与 Wan 的 rectified-flow loss。

每个生成目标默认采样 4 组独立的噪声和时间步，用配置项 `data.rf` 或 `--rf` 设置正整数。各次采样共用 VAE latent、backbone 产生的 CQ 条件和 DiT 每层的 cross-attention K/V。DiT 在现有 token 和图数预算内合批，超出预算时分批计算，同图各次采样的 flow loss 取平均。每篇文档只参与一次 packing 和文本 CE，跨卡归一化按原始目标图数计算。

模型只接收生成目标索引和 `Batch.generation_unconditional` 标记，不判断 T2I、交错顺序或采样比例。标记为 True 的视觉 Q 在每层不读取任何 C（空卷积历史、零 GDN 状态、无 attention 读取），保留自身残差和 MLP，输出经 Wan cross-attention 作为 DiT condition；False 或未提供标记时使用正常可见性。标记不改变目标位置、RoPE、query 数量或其他目标的计算，图像与视频共用实现。布局在 CPU 预先准备，逐层 forward 不再构造 mask／事件布局，也不判断数据任务。

样本或单个目标的 `non_cfg` 可以显式指定空条件（True）或正常前缀（False），目标标记优先。未指定时，无历史的目标使用空条件，有历史的目标按 `visual_uncond_probability` 抽取空条件，默认 10%，InterSyn 同样适用。选择由样本、目标位置、seed 和 epoch 决定，同一目标的 RF 次采样共用这次选择。评估的负分支显式使用空条件，`guidance` 默认 1。

两个消融轴只改变视觉 Q 读取的内容，不增加参数，默认值保持原行为和原数据游标：

- `model.query_interaction`（`--query-interaction`）：`none` 时每个视觉 Q 只读 C；`bidirectional` 时同一目标的 Q 还互相可见。attention 层对「该目标可见的 C 前缀＋同目标全部 Q」做一次 softmax；GDN 层为每个目标另开一条序列，重放其前缀写入后让全部 Q 先写后读，与内容流中图像块的规则相同。C 和文本 Q 不读视觉 Q，不同目标的 Q 互不可见。
- `model.null_condition`（`--null-condition`）：`masked` 时空条件的 Q 不读任何 C；`bos` 时只读文档起始标记，Q 的坐标改为相对该标记的固定位置，因此空条件与文档内容和长度无关。无起始标记的自然交错文档仍按 `masked` 处理。

本次三轮训练将原 bos 实验臂替换为 `--null-condition token`。训练和采样共用 tokenizer 中的 `<null_condition>` 和一个隔离的单 token 内容段。目标图的空条件 query 只读这一位置；双向模式下还读取同一目标图的 query。对话和自然交错格式使用同一条路径。token 使用预训练词表的空余行，由独立随机数生成器初始化，在 checkpoint 中保存。`--until-data-end --data-epochs 3` 按 epoch 更新目录打乱顺序，所有 DP rank 在各自语料耗尽后共同进入下一轮，学习率、优化器和 EMA 贯穿全程。

位置只通过 rotary 进入注意力分数，分数只对 value 加权。因此相互独立的空条件 Q，以及只能互相读取的空条件 Q，输出都是同一个向量的重复；只有 `bidirectional` 与单 token 条件（`bos` 或 `token`）同时启用时，空条件才随位置变化。验证见 `tests/test_query_interaction.py` 和 `scripts/validation/validate_query_blocks_cuda.py`。

query 数量和空条件策略本身不改变参数形状；本次两层 adapter 改变了权重与优化器形状，旧训练结果不再兼容：训练 checkpoint 仅接受 schema 7，推理快照仅接受 v2；不提供旧三层 adapter 转换。新 checkpoint 的精确 `--resume` 同时校验配置、数据游标、源码和实际启用的 kernel 选项。dynamic query 的语义、梯度、交错前缀及恢复规则有 CPU 回归覆盖，原固定 64 query 的 NPU 吞吐数字不能直接当作新配置的测速结果。

每 1000 个绝对更新默认保存独立推理快照（`--eval-snapshot-every`）；训练恢复 checkpoint 的频率、保留策略和完整性校验独立管理。`--eval-snapshot-dir` 指定持久评估权重目录，零间隔关闭快照。

### 训练中的 benchmark 评测

推理快照由 `evaluations/suites` 评测，指标为 GQA、MME 感知与认知、POPE F1、TextVQA、VQAv2、ImageNet FID-50K 和 GenEval。配置在 `configs/evaluation.yaml`，包括评测数据与 GenEval 运行环境的路径、生图协议和评测作业规格。

- 一个作业评测一个快照的一项任务（`qa` 为 GQA、MME、POPE、TextVQA 四项，另有 `vqav2`、`geneval`、`fid`），使用本机全部 GPU。作业命令是 `scripts/evaluate.sh TASK WEIGHTS OUTPUT`。结果分块写入，被抢占后重新提交会从已完成的部分继续。`EVAL_LIMIT=N` 只跑前 N 项，用作冒烟测试。
- 控制器在 CPU 主机上运行，不需要 torch：`.venv-cuda/bin/python -m evaluations.suites.controller --run RUN --source SRC`。它为每个已提交的快照和每项任务提交一个低优先级作业，跟踪状态。作业未产出 `RESULTS.json` 就结束时，控制器会再次提交。四项任务都有结果后，控制器删除该快照，并在 `RUN/benchmarks/RESULTS.md` 和 `results.jsonl` 汇总分数。在 `RUN/benchmarks` 下创建 `STOP` 文件，控制器即停止提交新作业。
- 生图统一使用 512×512、30 步 Euler、CFG 3.0、shift 3.0。问答均为单图零样本贪心解码，计分使用 lmms-eval 插件和官方 VQA 脚本。FID 与 ADM 的 ImageNet-512 参考统计量对比，Inception 特征在采样的同时由 CPU 提取。GenEval 使用固定版本的官方检测器。权重除了评测快照，也可以是带渲染器 EMA 的 TP1 训练 checkpoint。


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
