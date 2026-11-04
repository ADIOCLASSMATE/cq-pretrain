# CQ Pretraining

Qwen3.5 backbone、双流 content/query（CQ）建模与 MOSS-Image DiT 的多模态预训练实现。
支持图像理解（I2T）与图像生成（T2I），包含数据管线、训练、checkpoint、EMA 和定性评估。

## 模型

- Qwen3.5 语言主干交替使用 GatedDeltaNet 与 full attention。
- Content 流更新历史状态；query 流读取允许访问的 content 状态，不写回 GatedDeltaNet 的 recurrent/conv state。
- Qwen3.5 ViT 编码器冻结，spatial merger、语言主干和 DiT 参与训练。
- MOSS renderer 接收 CQ 生成的 condition queries；VAE 冻结并产生 posterior latent。
- Q=64 配置使用 8×8 query 网格，另有 Q=256 配置和 cross-attention 变体。
- cross-attention 变体使用 `x + to_out(attention)`，无额外 gate；输出投影采用非零的小初始化（标准差 `0.02 / sqrt(2 × DiT 层数)`），从第一步传递条件梯度。旧 gated 变体的 checkpoint 不兼容，需要重新训练。

主要入口：

| 文件 | 用途 |
|---|---|
| `ascend_pretrain/model.py` | 联合模型与损失 |
| `ascend_pretrain/qwen35.py` | Qwen3.5 与 GatedDeltaNet |
| `ascend_pretrain/qwen35_layout.py` | 双流布局与位置编码 |
| `ascend_pretrain/mainline.py` | CQ 执行计划 |
| `ascend_pretrain/moss.py` | MOSS renderer |
| `ascend_pretrain/input_pipeline/` | 数据读取、打包与设备预处理 |
| `ascend_pretrain/train.py` | 训练入口 |
| `ascend_pretrain/configs/` | 模型配置 |

## 数据来源约定

DCVLM 保持原生 I2T、对话、交错和纯文本类型，不再把一半 caption 改为 T2I。新预处理复用已发行的 DCVLM-baseline，与 GenUND 指定子集合并；实际训练使用的版本以训练配置为准。新任务统一使用 `python -m corpus_pipeline`，入口与筛选规则见 [数据预处理说明](corpus_pipeline/README.md)。T2I50 已退役，旧实验的指标与数据游标不能当作新语料的结果。

## I2T 的 CFG 无条件图像目标

新训练配方使用 `--i2t-image-loss-probability 0.2 --cfg-null-condition i2t_prefix`。
每个 image-first I2T caption 样本独立以 20% 概率参与图像 flow loss；所有样本
仍保留图像理解输入和文本 loss。T2I 生成目标全部保留。对话和交错文档中的图片
可能具有有效上下文，不作为这种固定前缀 null 样本抽样。

抽样在 CPU 打包阶段完成，由样本 ID、数据 seed 和 epoch 确定，保持 TP 和断点
恢复一致；新 epoch 重新抽样，不要求每个小 batch 恰好选中 20%。未选图片跳过
生成 VAE、DiT，以及 native CQ_IT 中只用于图像生成的 query 计算。全部图片仍走
ViT/content 流。选中目标按 DiT token budget 独立分块；图像 loss 按整个 DP/GA
更新实际选中的图片数求平均，不做 5 倍逆概率补偿。零选中 batch 只训练理解分支。

`i2t_prefix` 推理 null 复用训练的 I2T 图像 CQ 可见前缀。旧配置字段缺省保持
`1.0 + empty_t2i`；本地生产入口默认采用新配方。改变配方需要新训练阶段，不能
当成旧 checkpoint 的精确续训。日志记录 `generation_images`、
`i2t_null_candidates`、`i2t_null_images` 和实际 null 比例。若 I2T/T2I 各半，
选中生成目标中的 null 比例期望为 `0.2 / (1 + 0.2) = 16.7%`。

## 环境与资产

依赖版本见 `environment.core17.txt`、`requirements-data.lock` 和
`ascend_pretrain/deps.core17.lock.json`。`ascend_pretrain/setup_deps.py`
提供固定版本依赖的准备入口；Ascend 训练还需要匹配的 CANN、torch_npu 与算子扩展。

模型权重、tokenizer、数据、评估图片和运行产物需单独准备。
配置中的 `text_assets`、`vision_assets`、`moss_assets` 使用 `assets/` 下的相对路径，
可改为实际资产目录。`assets/` 中随代码保留轻量参考配置、参考源码及其许可证。
定性评估样例定义、图片来源和校验值见 `validation_assets/suite.json`。
请在仓库根目录运行命令，使相对路径保持一致。

```bash
python -m ascend_pretrain.train --help
python -m ascend_pretrain.setup_deps --help
```

## 评测权重快照

训练默认每 **1000 个全局更新 step** 单独保存推理权重，不启动 benchmark。
`--eval-snapshot-every` 可调整间隔，设为 `0` 可关闭；恢复训练后仍按绝对 step 计数。
现有恢复 checkpoint 的保存频率、`--keep-checkpoints` 和清理策略不变。

通过 `--eval-snapshot-dir` 指定持久目录。未指定时，若使用
`--checkpoint-dir /path/to/run/checkpoints`，快照写到
`/path/to/run/eval_snapshots/step_0001000/`；否则写到
`<output>/eval_snapshots/step_0001000/`。后续 step 各自保留，不自动轮转删除。

每份快照包含完整模型的 `model_tp_*.pt` 分片，以及最后提交的 `manifest.json`。
启用 renderer EMA 时保存 **online backbone + EMA renderer**，与定性验证一致；
未启用 EMA 时保存 online 模型。保留推理所需参数和 buffers，不包含优化器、
FP32 master、RNG 或数据游标。冻结 VAE 和 tokenizer 等外部资产不重复复制，
在 manifest 中记录其版本/校验清单、模型配置、源码摘要及权重来源。

后续评测只读取存在完整 `manifest.json` 的快照，模型装载可复用
`checkpoint.initialize_model()`（支持 TP 重分片），不能用 `--resume` 将其作为
完整训练恢复点。如果在 checkpoint 提交后、对应快照完成前中断，恢复到该
1000-step 边界时会补存；已完整提交的快照验证后跳过，不覆盖。

采用固定源码快照的运行中作业不会自动获得此功能，需要在后续启动/恢复时使用新版代码。

## 检查

数据管线测试需要与其测试词表匹配的 tokenizer 目录，可通过 `DCVLM_TOKENIZER` 指定，
默认使用 `assets/tokenizer`。MOSS 参考对照测试需要配置所指向的原始 MOSS 资产。

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s ascend_pretrain -p 'test_*.py' -t .
```

测试覆盖双流可见性、GatedDeltaNet 状态与梯度、位置编码、视觉编码与合并投影、
数据打包、通信调度、EMA、checkpoint 恢复及 renderer 参考实现。

## 版本管理

```bash
git clone https://github.com/ADIOCLASSMATE/cq-pretrain.git
cd cq-pretrain
git switch -c feature/my-change
git add <修改的文件>
git diff --cached
git commit -m "Describe the change"
git push -u origin HEAD
```

`.gitignore` 使用明确的文件允许列表。新增源码或配置时需同步更新允许列表，
再检查暂存内容；外部资产与本地运行文件不纳入版本管理。
