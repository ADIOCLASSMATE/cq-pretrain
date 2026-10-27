# UMM VBench

独立于 val loss 的视频生成质量评测。训练内验证和 `evaluations.eval` 继续使用现有逻辑。
这里仅提供 UMM 生成脚本和调用官方 VBench 的评分脚本，没有 benchmark 注册器或通用 suite。

## 清单与版本

- 官方仓库：[Vchitect/VBench](https://github.com/Vchitect/VBench)。固定提交：`fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490`，原始 T2V VBench 的 16 个维度。
- `full.json` 与该提交的 `vbench/VBench_full_info.json` 逐字节一致；SHA-256：`5dd2de80ee43cda750b2b72ea7023657c0b90d3702041c7e4608c65dbe50dccd`。
- `subset.json` 是本项目的诊断子集。将官方记录按 `dimension[0]` 分组，对排序后的 11 个分组依次用 `random.Random(42).sample(..., 2)` 选取记录，再按原始记录顺序保存。保留 `auxiliary_info` 等全部字段。清单固定，不在每次运行时重新抽样。
- 子集 SHA-256：`d08544a1f6a4ce0b686219ec40ebf0088c5824170cd4cdb3c60b9811bcbb64c5`。
- 元数据遵循上游 Apache-2.0，许可证见 `LICENSE.vbench`。本项目仅裁剪子集，完整清单未修改。

| 模式 | Prompt 记录 | 默认生成候选 | 筛选后目标视频数 | 输出 |
| --- | ---: | ---: | ---: | --- |
| subset | 22 | 150 | 110 | 16 个维度分数及样本量；不计算完整榜单总分 |
| full | 946 | 6,230 | 4,730 | 16 个维度及官方 quality / semantic / total 分数 |

946 条记录中有 944 个不同的 prompt 文本；不同分组中的同文记录不能去重，否则会缺视频。
普通 prompt 生成 5 条；temporal_flickering 每条生成 25 个候选，使用官方静态筛选后取 5 条。
不足 5 条时评分报错，补生成后重跑。规则来自[官方采样说明](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/prompts/README.md)。

## 生成：UMM 环境

输入为完整推理导出 `weights.pt` 或 `weights-ema.pt`，不是训练恢复目录。
模型、processor、VAE、设备、精度和采样参数均使用项目 YAML / `--set`。
每个进程加载一个模型和 VAE，按全局视频任务编号取步长切片；不创建训练用 DDP 或优化器。
随机种子为 `sampling.seed + video_index`，与进程数、子集/全量、续跑顺序无关，沿用 UniVideo 的逐视频设种方式。

```bash
UMM_ROOT=/inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3
UMM_CONFIG="$UMM_ROOT/configs/video.yaml"
UMM_EXPORT="$UMM_ROOT/outputs/my_exp/checkpoints/exports/weights.pt"
UMM_VBENCH_OUTPUT="$UMM_ROOT/outputs/my_exp/benchmarks/vbench_subset"

# 只查看数量和最终生成配置，不加载模型、不创建输出。
bash "$UMM_ROOT/evaluations/scripts/vbench_generate.sh" \
  --config "$UMM_CONFIG" --checkpoint "$UMM_EXPORT" \
  --mode subset --output "$UMM_VBENCH_OUTPUT" --dry-run

# 先生成一条，检查完整的 UMM -> Wan VAE -> MP4 链路。
bash "$UMM_ROOT/evaluations/scripts/vbench_generate.sh" \
  --config "$UMM_CONFIG" --checkpoint "$UMM_EXPORT" \
  --mode subset --output "$UMM_VBENCH_OUTPUT" --max-videos 1

# 四卡生成；省略 max-videos 即补齐本模式，已经完成的文件会复用。
NPROC_PER_NODE=4 bash "$UMM_ROOT/evaluations/scripts/vbench_generate.sh" \
  --config "$UMM_CONFIG" --checkpoint "$UMM_EXPORT" \
  --mode subset --output "$UMM_VBENCH_OUTPUT"
```

全量运行使用 `--mode full`，并选择新的输出目录，例如 `vbench_full`。
`--max-videos N` 仅访问本模式的前 N 个任务，供冒烟检查；它不会将一次未完成的运行标记为完整评测。

默认 shell 使用 NPU，加载项目 CANN/MindSpeed 环境；`PYTHON_BIN` 可指定生成环境的解释器。
CUDA/CPU 调试可设置 `UMM_DEVICE=cuda` / `cpu`、`PYTHON_BIN`，并显式选择兼容的
`model.qwen_gdn_backend=native`、`model.wan_attention_backend=sdpa`、`distributed.backend=accelerate`。
`--set training.device=...` 应与 `UMM_DEVICE` 一致。

帧数、分辨率、fps、CFG、采样步数均遵循 UMM 配置，而非 HunyuanVideo 的固定参数。
当前项目默认是 9 帧、4 fps、768×432、50 步。`full` 表示完整 prompt 和维度覆盖；
用于正式模型比较时还需明确相同的视频时长、尺寸与生成设置。subset/full 应使用同一套生成设置。

生成目录保存 `run.json`、原样的 `prompts.json`、逐 rank 的 `generation.rank-N.jsonl`，以及
`videos/<官方分组>/<prompt>-<index>.mp4`。日志记录每个视频的 seed 和包含解码/写盘的耗时。
MP4 在 staging 目录写完并检查几何、帧数、fps 与首帧可解码后原子移动到目标路径。
续跑复用通过同样检查的文件，损坏文件重新生成；checkpoint 路径/大小/修改时间、配置或清单改变时要求新目录。
权重身份检查使用文件元数据，不是对大权重文件逐字节计算哈希。

如果静态筛选不足，可对同一目录增加候选预算；原有 seed/视频保持一致，只补新增编号：

```bash
NPROC_PER_NODE=4 bash "$UMM_ROOT/evaluations/scripts/vbench_generate.sh" \
  --config "$UMM_CONFIG" --checkpoint "$UMM_EXPORT" \
  --mode subset --output "$UMM_VBENCH_OUTPUT" --flicker-candidates 50
```

该参数会增加本次清单中所有 flickering prompt 的候选数。续跑须保留已经提高的预算。
不同启动任务不要同时写同一输出目录。

## 评分：独立官方 CUDA 环境

评分脚本使用 Python 标准库负责调度；视频检查需要 PyAV。实际指标全部由官方仓库计算。
在有 CUDA 的机器上按[官方安装说明](https://github.com/Vchitect/VBench/blob/fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490/README.md#installation)准备独立环境，安装固定提交的 VBench、Detectron2 和 PyAV。
上游该提交检查 CUDA 11.6 / 11.7 / 11.8 / 12.1，并固定 Transformers 4.33.2；不要将其依赖装进 UMM 环境。

```bash
# 在独立 CUDA 环境中执行；VBENCH_ROOT 与解释器路径按实际位置填写。
VBENCH_ROOT=/path/to/VBench
git clone https://github.com/Vchitect/VBench.git "$VBENCH_ROOT"
git -C "$VBENCH_ROOT" checkout fd18b3d055cb0fc6f066ca90fe2c3c8cbb698490
# 先按官方说明安装匹配的 CUDA PyTorch / torchvision。
/path/to/vbench-env/bin/python -m pip install -e "$VBENCH_ROOT"
/path/to/vbench-env/bin/python -m pip install 'detectron2@git+https://github.com/facebookresearch/detectron2.git' av

VBENCH_PYTHON=/path/to/vbench-env/bin/python \
  bash "$UMM_ROOT/evaluations/scripts/vbench_score.sh" \
  --vbench-root "$VBENCH_ROOT" --run-dir "$UMM_VBENCH_OUTPUT"
```

生成和评分可以在不同机器运行：复制或共享整个生成目录，并在评分机器准备 UMM 代码及官方 VBench checkout。
评分不需要加载 UMM 权重或 NPU 库，也不要求原 checkpoint 路径在评分机器上存在。
`--dry-run` 只校验清单/官方源码版本并列出评分命令，不检查 CUDA、不下载权重、不创建评分目录。

评分流程：

1. 检查固定的官方 Git 提交与源码未修改、所有候选视频完整。
2. 调用官方 `init_submodules(['dynamic_degree'])` 准备 RAFT，再调用官方 `static_filter.py`。
   subset 使用 `--filter_scope all`，评分读取筛选输出中编号 0–4 的视频；full 使用官方默认 flickering 筛选范围。
3. 根据官方 `dimension_to_folder.json` 对 16 个维度分别调用 `evaluate.py`。
   使用 `--mode vbench_standard --full_json_dir <本次 prompts.json>`，保留语义指标所需标签。
   这里的上游 mode 名称不代表子集变成了完整榜单评测。
4. 检查官方实际消费的 prompt/video 清单和有限数值结果。只有 full 且 16 个维度齐全时，
   才将评分 JSON 打包给官方 `scripts/cal_final_score.py`。上游缺失维度会补零，因此本地入口提前拒绝缺项。

每次评分写入新的 `scores/<UTC时间>/`，保存官方命令、日志、逐视频结果和 `summary.json`。
首次评分会由官方代码下载评估模型；请预留网络与缓存空间。失败保留日志，重跑创建独立结果目录。

## NPU 状态

当前实现支持 UMM 在 NPU 生成，官方评分保留 CUDA 路径。未做 CUDA-to-NPU 静默替换。
截至 2026-10-22 的源码检查：上游安装入口检查 CUDA；`evaluate.py`、静态筛选和通信辅助使用 CUDA/NCCL；
GRiT 使用 Detectron2，另外还有 RAFT、AMT、UMT、ViCLIP、CLIP 等模型。
完整迁移需要单独验证依赖安装、所需算子及 16 个维度的数值一致性，不能仅以修改设备字符串确认支持。
因此本入口在没有 CUDA 时给出明确错误，NPU 评分迁移范围需另行讨论。

## 本次验证（2026-10-22）

- CPU 回归覆盖清单、分片、seed、UMM 小模型生成、MP4 编码、续跑及损坏文件重建、评分命令和缺项拒绝；另运行现有训练/导出/验证 CLI 回归。
- 使用固定提交的官方元数据构建函数核对 subset/full 的全部 16 个维度，验证标签和每条 5 个输入路径均保留；用合成分数验证官方汇总脚本的输入格式。这不是对真实视频的官方指标计算。
- 在 Ascend 910B2C 上加载真实 UMM 导出，完成 MindSpeed GDN、Wan 采样、真实 Wan VAE 解码和 MP4 写盘：9 帧、4 fps、768×432、2 步、CFG=1，单条生成日志为 56.50 秒，未包含模型加载。这里只证明链路可用，不用于外推 50 步或更长视频吞吐。
- 完整 GPU 评分及 NPU 评分迁移尚未执行；没有产生真实 VBench 质量分数。
