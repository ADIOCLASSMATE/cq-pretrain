# UMM v3 迁移记录

本文汇总重构方案、源码来源和基线复核记录。运行时不读取本文；验证快照见 [validation.md](validation.md)。

## 当前基线与源码来源

当前迁移路线是 **CQ_IT backbone + 固定逐位置 condition + 全图 Z2 DiT flow**。Z2 使用 AdaLN-Zero 和 noisy-token self-attention。现有迁移测试的结论限定于所选源码；最佳实验 run、配置和 checkpoint 尚待确认。

| 项目 | 记录 |
| --- | --- |
| 来源仓库 | `../unified-mm`（相对 UMM_v3 根目录） |
| 固定分支 | `origin/npu` |
| 固定提交 | `193ec90300af5fee4bc4b7e8a426492a908a0447` |
| 选定架构 | CQ_IT + fixed per-position condition + Z2 joint-image flow |
| 运行时依赖原仓库 | 否 |
| 最佳版本状态 | `unconfirmed_best_version` |
| 待完成事项 | `needs_best_run_identification` |

原来的选择依据是研究方向“B → Z2”和可用代码，尚缺少负责人指定的最佳 run 与可比较的正式评测。因此该依据不足以确认 Z2 是最佳实现。

## 重构目标与职责

围绕固定 CQ_IT + 全图 flow 实现进行独立架构重构，保留选定路线的训练和推理数学语义。

目标目录：`UMM_v3`。

- backbone 负责多模态 content/query；generator 只接收 condition、x_t、t
- modeling_umm 显式组合，不再依赖旧父类的整个实验分支集合
- loss 与 sampler 从模型分离，trainer 与对象初始化分离
- checkpoint 只保留一份；EMA 位于 training；配置解析位于 configs/config.py
- KL16 codec 与全图 flow sampling 位于 models；未使用的 diagnostics 工具已删除
- 不把 Z2 的逐位置 AdaLN 误称为旧 clean-content cross-attention head
- 权重迁移与完整训练恢复分别设计，旧 optimizer 不提供未经验证的转换

## 实施阶段

下表保留原迁移记录中的阶段状态和实施标记，表示当时的交付记录。当前实现的正确性仍需结合具体验证范围判断。

| 顺序 | 范围 | 验收项 | 实施标记 | 记录状态 |
| --- | --- | --- | --- | --- |
| 1 | 固定源码/架构、接口与训练协议 | 固定 source commit 与文件指纹；明确 image-only 几何及视频边界 | `root` | `complete` |
| 2 | backbone/generator/composition、全图 sampler 与文本 cache | 源码前向/梯度对照；CQ 无目标泄漏；Euler/Heun/CFG 与单次 backbone；cached/full 文本输出一致 | `model_core` | `complete` |
| 3 | 数据、序列、packing、调度、已消费游标 | 来源逐字段零容差对照；真实多worker恢复；单任务/混合GA；文本loader不消耗模型RNG | `data_pipeline` | `complete` |
| 4 | trainer/loss/optimizer、EMA/checkpoint、评估 | 混合梯度与AdamW更新对照；双rank DDP与分片EMA；训练中断恢复逐元素一致；独立验证不读训练资产 | `root + runtime_eval` | `complete` |
| 5 | 配置、CLI、转换工具、文档 | 真实tiny模型train/eval/generate命令链；来源权重及EMA转换；KL16接口shard缓存转换；可安装包在仓库外导入 | `root + runtime_eval` | `complete` |

### 已交付范围

- 纯文本、T2I、I2T
- class/caption posterior 数据及packing
- 完整图像token网格flow
- Euler/Heun、CFG、文本KV-cache
- AdamW/WSD、混合GA、DDP接口、ZeRO-2配置
- 分片FP32 EMA与事务恢复
- loss/PPL/生成指标与冻结候选评分入口

### 交付边界

- 视频数据/时空位置/codec支持
- 旧全部架构消融和全部benchmark资产适配器
- 真实大模型质量/显存/吞吐等价
- GPU/NPU或DeepSpeed实际集群验收

### 后续设备验收

1. 同一来源权重和固定batch，在目标BF16设备对照condition/velocity/loss/gradient
2. 同一seed和solver对照生成latents及KL16解码
3. 短程ZeRO-2训练与中断恢复对照
4. 正式样本集FID/IS及文本/I2T指标对照后再替换原训练入口

## 基线复核

复核日期：2026-10-13。范围为本地原仓库的 master 与 origin/npu 源码、配置及已跟踪实验文档；未查询远程报告或训练任务。

| 引用 | 固定提交 |
| --- | --- |
| `master` | `933ae59acc6cc08e9d40f5b1f6316bcd7ac99900` |
| `origin/npu` | `193ec90300af5fee4bc4b7e8a426492a908a0447` |

复核结论：Z2 的迁移实现已完成并经过当时的 CPU 验证，但其作为最佳实验基线的选择尚未确认。源码迁移一致性与模型质量排名需要分别判断。

先前基线选择的问题：采用“B → Z2”研究方向及 CQ_IT + 全图 flow 的架构匹配作为依据时，没有对应到最佳 run 和同协议的终点评测。

### 证据

以下源码路径和行号对应上表中的固定提交。

| 引用 | 文件 | 行号 | 发现 |
| --- | --- | --- | --- |
| `master` | `docs/ABLATION_CONCLUSIONS.md` | 8、12、61、65 | Retained dynamic dual-stream contextual head, 100-step Heun and parallel_rate=1; not the Z2 joint-flow experiment. |
| `origin/npu` | `docs/PUBLIC_EVALUATION_REPORT.md` | 3、4、36、40 | B -> Z2 describes research direction; training validation and one-epoch screening are separate from formal final EMA evaluation; use actual saved run configs. |
| `origin/npu` | `scripts/build_current_evaluation_site.py` | 304、308 | Z2 final evaluation and CFG sweep explicitly not archived; one-epoch screen is step1251/raw/CFG3.5/Heun10/FID50K. |
| `origin/npu` | `docs/Z2_EXPERIMENT.md` | 7、16、49、58 | Z2 changes head depth/width/FFN/normalization/QKV modulation; LR screen selects a short-budget starting point, not proven long-run optimum. |
| `origin/npu` | `models/modeling_model/image_flow_loss.py` | 445、469 | Original contextual block combines AdaLN and cross-attention; these mechanisms are not mutually exclusive. |
| `origin/npu` | `models/modeling_model/image_flow_loss_joint_b.py` | 45、59 | Z+B reuses cross-cache APIs but constructs K/V from the same noisy state as Q; it is self-attention by data source. |

### 原仓库中的相关实现

该表用于解释来源架构差异；v3 当前只迁移 Z2。

| 实现 | 配置 | 图像顺序 | 条件与 attention | 深度 | 宽度 |
| --- | --- | --- | --- | --- | --- |
| `B` | `configs/selfless/unified_b_33b_ascend64.yaml` | random / native serialized reveal | dynamic content/query cross-attention plus AdaLN | 8 | 1280 |
| `Z` | `configs/selfless/unified_z_33b_ascend64.yaml` | joint | condition input addition plus time AdaLN and noisy self-attention | 8 | 1280 |
| `Z+B` | `configs/selfless/unified_z_b_head_33b_ascend64.yaml` | joint | same-noisy-stream attention with separate Q/KV norms and Q-only attention modulation | 8 | 1280 |
| `Z2` | `configs/selfless/unified_z2_33b_ascend64.yaml` | joint | AdaLN-Zero plus common Q/K/V noisy self-attention | 12 | 864 |

### 当时缺少的资产

- `unified-mm/output/evaluation/summary.json`
- `unified-mm/output/experiments/`
- `Formal evaluation outputs and selected actual checkpoint config`

### 关闭最佳基线问题所需工作

1. Identify the owner-selected best run/config/checkpoint and its source commit.
2. Read the actual saved config and trace Q/K/V sources, image ordering and whole-image sampling.
3. Verify formal quality results and comparison protocol: training budget/data, raw or EMA, sample count, CFG/solver/steps and matched metrics.
4. Correct the generator implementation if the identified head is not Z2, then run parity against that implementation.

基线复核时仅修正文档与状态，未修改模型、训练或采样实现。已有 Z2 迁移与 CPU 检查结果仍只适用于 Z2。

## 验证参考与署名

- 模型参考：tests/reference/pinned_model.py; frozen class bodies with local CPU kernel/framework adapters。
- 数据参考：tests/test_data_source_parity.py; optional explicit UMM_SOURCE_SNAPSHOT of this commit。
- 训练协议参考：tests/test_training_alignment.py 直接提取固定源码中的冻结与验证抽样函数；加噪复用上述模型参考。验证结果见 [validation.md](validation.md)。
- 保留原作者署名与上游声明；MAR 声明见 [MAR.txt](../licenses/MAR.txt)，完整说明见 [NOTICE.txt](../licenses/NOTICE.txt)。

## 固定来源文件的 SHA-256

这些指纹针对固定来源提交中的文件，用于追溯迁移依据，不是当前 v3 文件的校验值。

| 来源文件 | SHA-256 |
| --- | --- |
| `models/modeling_model/modeling_selfless_flow.py` | `e9fdd0a279cd0fe4b425a85f36a4b9ade1324c6afc3e84f23e5cab8680137954` |
| `models/modeling_model/modeling_joint_dit.py` | `6a6f5d5322fcb30798b8ea1e0d1035c6c9e5f16103e8b261fe98c6cf0a85d02c` |
| `models/modeling_model/image_flow_loss_z2.py` | `18b33dd527a865641369a943150b9b489b0cd7e754617661ce4444c85196e6f7` |
| `models/modeling_model/image_flow_loss.py` | `92c20428f082b315907dd1052b05e687decb0cd330d58f3c031712ab381c5bc5` |
| `models/modeling_model/image_position_utils.py` | `f5f4dbff4898688d9a316cd844da16a16e71808e7610921dca9d53497e837cdd` |
| `pretrain/train_selfless_flow.py` | `cd102aa17298af48a59496c3728b5f25d288470740b51bc7d385ec7e17c98b11` |
| `utils/dataset_imagenet_flow_cache.py` | `4dbadf31c439ed444fd040ee602e2145681b71ff5fd1302bbca1e1ebdcdcf48e` |
| `utils/multimodal_segment_packing.py` | `e2936fcfd874807af46ad5c4abbbb6d170c51727729e07efea1f319eeae0fdb2` |
| `utils/climbmix_online_dataset.py` | `6840b816a7301b62134d3ce782be5ba4d97f64944eccf9923b4b41cdcc0c6162` |
| `utils/sharded_ema.py` | `32d2c230740740471397f2d6dc5859de858c96a8f6f6774d9caa11547482ba40` |
| `configs/selfless/unified_z2_33b_ascend64.yaml` | `8e8f69b048d6e4721e871947f3d8a864c906e4e1a71f6444506bb04f14c368b4` |
| `utils/training_unified_loss_validation.py` | `e0c12eafc0fea22e09e0b5eef2db78a616896ee7a1d640baa920b06dfe449743` |
| `utils/training_downstream_validation.py` | `739254e8c06fa35e959149f93cef1d0fc81502cb79ab62fcf243ed7a4387a2bb` |
| `utils/selfless_flow_optimizer.py` | `723a5b25bd4fcec4a83dda047cf58f2dbc76afd9f705259721e130a915977e8e` |
| `utils/evaluation/text_benchmarks.py` | `036c31e9f6bfbf54b6daf980d5500fd6a4c797dcff5b40a96a88de1b68b2ce67` |
| `utils/evaluation/multimodal_likelihood.py` | `5255d6079ea0f3b1443a52348390cd8dbdc36dcc195aec699fbb470cb3818576` |
| `utils/evaluation/calibration.py` | `f9726961b9c8dbf3a1099f6c50d87d302c027fb2602c8fdb0a4908bd0c7eac6d` |
