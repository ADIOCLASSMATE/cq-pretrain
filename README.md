# CQ Pretraining

Qwen3.5 backbone、双流 content/query（CQ）建模与 MOSS-Image DiT 的多模态预训练实现。
支持图像理解（I2T）与图像生成（T2I），包含数据管线、训练、checkpoint、EMA 和定性评估。

## 模型

- Qwen3.5 语言主干交替使用 GatedDeltaNet 与 full attention。
- Content 流更新历史状态；query 流读取允许访问的 content 状态，不写回 GatedDeltaNet 的 recurrent/conv state。
- Qwen3.5 ViT 编码器冻结，spatial merger、语言主干和 DiT 参与训练。
- MOSS renderer 接收 CQ 生成的 condition queries；VAE 冻结并产生 posterior latent。
- Q=64 配置使用 8×8 query 网格，另有 Q=256 配置和 cross-attention 变体。

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
