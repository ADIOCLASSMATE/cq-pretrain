# UMM v3：Qwen3.5 + Wan2.1

当前主链路支持单视频样本的 T2V/V2T 持续预训练和无缓存推理。Content/query 等长，文本 query 预测同位置 token，整段视频 query 只读取第一个 visual token 之前的前缀。Qwen 原生 ViT 提供 content；Wan 对整段视频 latent 联合建模。

模型代码收敛为三个模块：

| 模块 | 职责 |
| --- | --- |
| `models/backbone_qwen35.py` | 原生 Qwen3.5、ViT、原生 M-RoPE、标注的 CQ 分支和视频输入适配 |
| `models/generator_wan21.py` | 原生 Wan2.1 权重布局、条件投影、视频 flow 前向和采样 |
| `models/modeling_umm.py` | 组装两者，将文本 query 接 LM head、视频 query 接 Wan，提供无缓存生成 |

原生参考文件保留在 `models/references/`，运行时不导入。已移除当前主链路不再使用的旧 Qwen3/Z2 图像路线备份。

训练调度、参数分组、EMA 和 checkpoint 位于 `training/`、`utils/`；数据读取、原生序列化、缓存和 padding 位于 `data/`。没有跨样本 packing，不自动划分训练/验证集或分配任务视图。

`scripts/` 只保留训练、评估、推理、离线特征/latent 准备和环境准备入口。开发用的四卡冒烟、连续/恢复对照、CQ/Wan 验证与 checkpoint 比较统一放在 `tests/validation/`，用法见 [测试说明](tests/README.md)。Triton 启动完整性检查位于 `utils/triton_integrity.py`，由环境和安装入口调用。

Ascend 主链路采用 MindSpeed + Megatron Core 的 DP 与分布式优化器，TP=PP=CP=1；默认开启梯度归约和参数收集重叠。低层环境按 `cq-pretrain` 固定：Python 3.12.13、PyTorch 2.7.1+cpu、torch_npu 2.7.1.post4、CANN 9.0.0，源码提交见 `configs/deps.core17.lock.json`。Qwen3.5 仍使用 Transformers 5.19.0 / Diffusers 0.41.0。

```bash
# 安装到 .venv-npu 和 .deps/Ascend90；首次安装备份旧 venv，不覆盖系统 CANN。
python3 scripts/setup_mindspeed.py --sources --python --cann
# 若主机尚无 g++/make，可由 root 单独执行 --host-tools 安装 build-essential。

# 正式配置需要填写 data.sources。
.venv-npu/bin/python -m training.train --config configs/video.yaml --dry-run

# 910B 四卡短程验证：9 帧、4 fps、768×432，T2V/V2T 各一次微批。
bash tests/validation/test_train.sh

# 正式启动；学习率、loss 权重、冻结和采样预算均可通过 YAML/--set 覆盖。
bash scripts/train.sh --config /path/to/run.yaml
```

910B 测试脚本默认使用仓库的 `.venv-npu/bin/python`、项目级 CANN、`HCCL_INTRA_ROCE_ENABLE=1` 和 `wan_attention_backend=npu`，由 torchrun 启动四个进程。全局默认训练后端为 MindSpeed。安装与测试依赖统一见 [requirements-mindspeed.txt](requirements-mindspeed.txt)，不再维护单独的 CPU 环境、旧 DeepSpeed 环境或 ZeRO-2 启动配置；启动器不再读取 `ACCELERATE_CONFIG`。

CPU 回归也使用 `.venv-npu`，通过环境变量禁用 torch_npu 自动加载，无需加载 CANN：

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-npu/bin/python -m pytest tests -q
```

CPU 小模型测试显式选择 `distributed.backend=accelerate`、`training.device=cpu` 和原生计算后端；这条训练接口仅用于 CPU 回归。`accelerate` 包仍用于 MCore 的数据分片和 CPU 测试，不是 DeepSpeed 回退入口。需要真实设备的测试由各自的显式环境变量启用。

视频配置还显式选择 `model.qwen_gdn_backend=mindspeed`，在 CQ 分支调用固定提交的官方 Triton GDN；通用配置的 `native` 保留原生 Torch 参考计算。两者没有静默回退。环境新增 Triton 3.2.0 / Triton-Ascend 3.2.1，安装脚本保证 Ascend wheel 最后覆盖共享文件。上游 NumPy 约束存在冲突：保留 CQ 的 1.26.0，并在锁文件记录 Triton-Ascend 要求 1.26.4 的元数据例外，`pip check` 并非全绿。

`scripts/env_mindspeed.sh` 支持在 Bash 和 zsh 中直接 source。安装后及每次环境加载都会运行 `utils/triton_integrity.py`，按 Ascend 的 RECORD 校验共享文件及 NPU 自有文件的大小、SHA-256 和导入路径，发现覆盖或删除就报错返回。可用 `.venv-npu/bin/python utils/triton_integrity.py` 单独检查，无需初始化 NPU。

Ascend 算子编译的工作文件默认通过 `ASCEND_WORK_PATH` 放入 `.cache/ascend-work/<主机名>/`；环境入口和 pytest 都保留用户显式指定的路径。它包含 CANN 的编译调试/运行文件，与训练日志和 checkpoint 分开。根目录偶发的 `kernel_meta*` 是可再生成的编译工作目录，已加入 Git 忽略；绕过项目环境和测试入口直接启动 NPU 程序时，CANN 仍可能在当前目录创建它。

训练日志默认放在 `outputs/<exp_id>/logs/train/<attempt_id>/`，checkpoint 放在 `outputs/<exp_id>/checkpoints/`。`--set training.exp_id=实验名` 选择实验，`--auto-resume` 自动恢复该实验最近完整 checkpoint；默认保留最近 3 个。`output_dir` 可覆盖完整实验目录，`log_dir` / `checkpoint_dir` 可独立覆盖。推理权重导出位于 `checkpoints/exports/`。每次启动都有独立日志目录；恢复允许调整管理参数，训练参数仍严格校验。processor 随导出保存到 `checkpoints/exports/processor/`。

[训练与数据说明](docs/video_training.md) 包含 JSONL、缓存生成、混合数据、参数来源、推理和恢复命令；[CQ 模型接口](docs/video_cq.md) 说明双流可见性；[验证记录](docs/validation.md) 区分 CPU、NPU 验证及尚未覆盖的检查。

统一采样与评估通过 `scripts/evaluate.sh` 调度 `sample` / `score` 两个动作。任务 YAML 位于 `configs/evaluations/`：loss 调用与训练内 validation 相同的监督前向，T2V/V2T 负责生成，VBench 通过协议适配器接入。每个任务分别定义数据、采样参数、采集产物和评分参数；一次启动可选择多份任务配置。`evaluations/benchmarks/__init__.py` 的 `Benchmark` 注册项声明采样规划函数、scorer 和评分是否需要 UMM；新增协议与 YAML 不需要修改公共调度器或模型。`scripts/generate.py` 保留为旧命令的兼容转发，不再维护独立推理实现。

```bash
# 同一 checkpoint 计算 loss、生成 T2V/V2T 预览；数据在 run/task YAML 中设置。
bash scripts/evaluate.sh --config /path/to/run.yaml --checkpoint /path/to/weights.pt \
  --task-config configs/evaluations/loss.yaml \
  --task-config configs/evaluations/t2v_preview.yaml \
  --task-config configs/evaluations/v2t_preview.yaml --eval-id step2000
```

评估结果放在 `outputs/<exp_id>/evaluations/<eval_id>/`，日志放在 `logs/eval/<eval_id>/<attempt_id>/`；续跑复用结果但新建日志 attempt。评估进程数默认继承 `distributed.num_processes`，可由 `evaluation.num_processes` 覆盖。详细参数、输入视图和续跑规则见 [统一评估说明](docs/video_training.md#统一采样与评估)。[VBench 运行说明](evaluations/benchmarks/vbench/README.md) 保留原协议参考；本阶段明确接口，官方评分仍需单独配置 CUDA/VBench 环境，不声明已完成端到端评分适配。

官方 MindSpeed GDN 与标准行索引接入后，四卡真实权重连续训练 4 步和中断后自动恢复至第 4 步已逐位一致：模型、FP32 master weights、Adam、RNG 和数据游标共 5,480 个张量全部相同。最近 N 个 checkpoint 清理、日志隔离及 CQ 无泄漏检查通过。完整设置、数值判据和依赖例外见验证记录；这属于短程链路验证，不代表长期收敛或生成质量评估。
