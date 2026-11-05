# Qwen3.5 + Wan2.1 主链路验证

本记录收录视频主链路及后续调整的验证。旧图像路线的独立源码备份及其验证记录已按用户要求清理，不用于证明当前视频路线。

## 2026-11-05 UTC：共享 main 执行层与 16 卡验证

通用 attention、GDN、分块交叉熵、融合投影、Wan 层和分布式执行改为 main 的实现；16 个共享模块逐文件 SHA-256 相同。视频包装层保留原生时间戳、动态 query、任务归一化和采样接口。video 继续从 Qwen3.5-2B 初始化，使用直接的 2048→1536→1536 adapter，拒绝旧结构的训练结果。backbone 在原生时间块内双向、块间 causal，Wan 在整个 clip 内双向。

本轮结果与源码、kernel 身份见 [机器可读报告](shared_runtime_validation_20261105.json)。设备为节点 224 的 16 张 Ascend 910B2C；最终联合验证任务 `cq-video-shared16-1106-r7` 正常退出，退出码 0。

| 检查 | 结果 |
| --- | --- |
| CPU 回归 | video 270 passed、18 项设备测试跳过；main 157 passed、51 subtests passed |
| 实际 NPU 算子与 CQ | 16 rank 各 14 项通过，共 224 项，无失败或跳过；覆盖 AscendC 与 MindSpeed，检查实际 kernel 身份 |
| 分布式更新对照 | 16 卡、GA=2、3 次更新，与显式全局 batch AdamW 的参数最大绝对差 `1.4901161193847656e-8`；副本及优化器恢复逐位一致 |
| 混合 V2T/T2V 与恢复 | 16 卡、GA=2、重入式 checkpoint；小模型连续训练与恢复后的模型、FP32 master、优化器、scheduler 和通信就绪状态逐位一致 |
| 完整初始权重训练 | Qwen3.5-2B + Wan2.1-T2V-1.3B，16 卡、GA=2，完成两次优化器更新；在线 ViT 与缓存 post-merger 特征均覆盖，loss/参数有限，所检查的副本参数完全相同；峰值显存 26.31 GiB |
| 共享 Wan 的 TP2 回归 | 16 卡通过前后向对照；输出相对 RMS 0.279%，参数梯度最大相对 RMS 1.251% |

设备测试发现并修复了视觉特征原地写入 Megatron 自定义 autograd view 的问题：现在将全部视觉位置合并为一次非原地 `index_copy`，并加入 CPU 回归。恢复检查还修复了首个更新后的通信就绪计数保存。独立 MindSpeed 对照显式传入与参考相同的 Q/K 归一化设置，数值阈值未放宽。失败尝试与节点诊断保留在报告和原始日志中，不计入通过结果。

完整权重测试使用短合成视频布局和 VAE latent，不包含真实数据 I/O、VAE 编解码、完整分辨率长视频吞吐或生成质量评估；记录的 step 时间包含验证操作。video 训练仍为 DP、TP=PP=CP=1；独立 TP2 测试验证的是 main 的共享 Wan 层。Wan 条件投影与 K/V 已缓存，video 文本生成仍采用完整前缀接口，未接入 main 的增量 KV/GDN 解码。下方历史结果只适用于各自日期的实现。

## 2026-10-23 UTC：统一 sample/score 与四卡验证

本轮以任务 YAML 驱动 loss、T2V/V2T 和 VBench 的 sample/score 调度。`models/` 的 8 个 Python 文件及原 `evaluations/validation.py` 与修改前逐字相同；不增加逐步采样轨迹。VBench 的规划、产物协议和外部 scorer 接口已接入，实际 CUDA 官方评分未执行。

证据目录：`outputs/eval_unification_20261023T070213Z/`。真实设备为四卡 910B，使用现有 `.venv-npu`、CANN 9、MindSpeed GDN、BF16 参数与 NPU Wan attention。真实本地 Qwen3.5-2B＋Wan2.1-1.3B 完成两步训练及导出；输入为 `temp_data/slider_stop.jsonl` 的 T2V/V2T 两种视图，9 帧、4 fps、768×432。采样检查使用两步积分、CFG=1、两 token，仅检验数值与工程链路，不是生成质量评估。

| 检查 | 本轮结果 | 证据 |
| --- | --- | --- |
| 实际训练 validation 与独立 loss CLI | 同一导出 checkpoint、输入、seed，全部 14 项指标完全一致 | `actual-training-loss-comparison.json` |
| 逐批 loss | 实际 `Trainer.validate()` 路径、公共 scorer 和另起进程的独立 CLI，T2V/V2T 每批各项 loss/count 完全一致；本数据每类只有一批，按规则均分配给 rank 0，其他 rank 参与归约 | `parity-comparison.json`、`parity/trainer-batches.rank-*.jsonl`、eval attempt 下的批记录 |
| 原生成计算与公共 sample | 四个 rank 均比较 T2V latent、V2T token/logits，最大绝对差为 0 | `parity/report.rank-*.json` |
| 四卡采样 | 4 个 T2V＋4 个 V2T 请求，rank 0–3 各处理两种任务一个结果，无重复 | `fourcard-samples.json` |
| 中断与续跑 | 先保存 3 个结果，再重新启动补齐其余 5 个；已有文件 mtime 不变；latent、token/logits 逐元素一致，MP4/文本文件逐字节一致。一个 T2V 请求由 rank 3 改到 rank 0，结果仍相同 | `resume-files-comparison.json`、`sampling-tensor-comparison.json` |
| 日志隔离与恢复 | train/eval 分开，续跑新建 attempt；校验权重、配置、输入及产物摘要；损坏产物重新采样，失败前已提交产物可继续使用 | `experiment_final/logs/`、dispatcher 回归 |

原始旧 CLI 没有应用配置中的确定性开关：直接对比 T2V latent 的最大差为 0.015625，V2T 文本相同。只为原旧代码补齐 `torch.use_deterministic_algorithms(True)`、`HCCL_DETERMINISTIC=True` 和 `CLOSE_MATMUL_K_SHIFT=1` 后，T2V latent 与新入口逐元素相同，见 `legacy-deterministic-comparison.json`。没有修改模型或旧参考代码来消除差异。新入口按训练配置设置这些开关，并使用每请求固定 RNG；CPU 旧脚本的构造阶段 RNG 与生成 RNG 需区分。

验证途中首次训练遇到 AICPU `507018`，独立设备初始化另出现一次 `507033` 超时；没有获得稳定的模型层复现，也没有修改模型绕过。布尔索引最小测试、实际 batch 布局以及之后顺序执行的完整训练和所有 NPU 评估均通过。另一次测试把全局验证上限设为 1，仅覆盖 T2V，被原有“各任务均需有指标”的检查拒绝；测试配置改为 2 后通过，loss 规则未变。

配置与管理基础测试此前 27 项通过，CPU shell 自动恢复、失败元数据保护与 checkpoint 清理也通过。本轮最终回归 **125 passed、0 failed（219.71 秒）**，覆盖配置、采样、loss、调度器、VBench 协议、视频管线、shell 与训练管理；证据为 `final-tests.xml`、`final-tests.log`，不把历史测试计入本次结果。日常配置与命令见 `docs/video_training.md`；真实四卡比较入口为 `tests/validation/verify_unified_evaluation.py`。

## 2026-10-22 UTC：验证工具归档位置与历史产物清理

开发验证命令统一迁到 `tests/validation/`，本文命令路径同步到新位置；历史日志内部的旧路径保持原样作为记录。Triton 文件校验属于启动依赖，迁到 `utils/triton_integrity.py`，环境和安装入口继续执行校验。`scripts/` 仅保留日常入口，完整命令见 `tests/README.md`。

清理 64 个历史诊断张量、3 个 CANN `.run` 安装包和无运行/测试引用的 `archive/`，保留历史日志、报告、VBench 及 `models/references/`。已安装的 `.deps/Ascend90`、`.deps/core17`、`.python` 和 `.venv-npu` 不受影响；重新安装 CANN 时，setup 会按锁文件重新下载并校验安装包。

根目录 `kernel_meta` 当时仅有一个 0 字节 `*_bt.log`，已清理。Ascend 会自动产生此类算子编译/调试目录；环境入口和 pytest 现默认将 `ASCEND_WORK_PATH` 指向 `.cache/ascend-work/<主机名>/`，尊重显式覆盖，不改变模型或算子计算。该变量也影响 CANN 自身运行日志等工作文件的默认位置，与 UMM 实验日志分开；没有改变算子共享缓存 `ASCEND_CACHE_PATH`。

本轮验证：28 项针对性回归通过，5 个迁移后 Python 模块的 CLI 入口通过，恢复 shell 的编排检查通过；从 `/tmp` 使用真实环境执行迁移后的训练 shell dry-run 成功，仍为四进程 MindSpeed。Bash/zsh 环境加载、Triton 92 个文件校验、NPU 小矩阵计算和四卡识别正常，未在根目录再产生 `kernel_meta`。模型和 VBench 的 22 个文件摘要不变；本轮没有重新执行四卡完整训练。清单和日志见 `outputs/cleanup_tools_20261022T131342Z/`。

## 2026-10-22 UTC：环境与测试产物收敛

按用户确认，CPU 回归统一使用 `.venv-npu`，移除独立 `.venv` 和旧 Torch 2.8/DeepSpeed 环境备份；依赖由 `requirements-mindspeed.txt` 与 `configs/deps.core17.lock.json` 统一管理。默认训练后端改为 MindSpeed，移除 ZeRO-2 YAML、DeepSpeed 可选依赖和旧启动分支。CPU 回归显式使用轻量 Accelerator，MCore 仍依赖 accelerate 的数据分片工具。

本轮 `.venv-npu` 全量 CPU 回归 **213 passed、18 skipped**，包含 CQ、原生一致性、数据管线、训练/恢复、独立评估和 VBench 接口测试；18 项需要显式启用真实设备。第一次运行的 4 项失败来自测试隐式依赖旧默认后端，改为显式 CPU/GDN 测试配置后复测通过。Triton 完整性检查覆盖 92 个文件；生产 shell dry-run 确认 MindSpeed、四进程及两种通信重叠，shell 语法和 Ruff 检查通过。本轮不重新执行四卡完整训练。

历史测试 checkpoint、优化器状态和导出权重已按授权删除，清单约 1,030 GiB；历史日志、验证报告、VBench 代码和样例视频保留。**下文历史 checkpoint 路径、旧 `.venv` 命令及旧依赖清单仅记录当时状态，现已清理，不能直接用于恢复或评估。** 后续 VBench/独立评估需指定新的完整导出。原始模型权重目录、`models/` 源码和原生参考实现未修改。

清理清单、模型/VBench 摘要校验和本轮测试日志位于 `outputs/workspace_cleanup_20261022T125105Z/`。项目级 CANN、`.python` 解释器、源码依赖、验证工具和旧图像源码备份保留。NumPy/Triton 的上游元数据例外保持原状，没有在清理过程中更换已验证版本。

## 2026-10-22 UTC：zsh 环境加载与 Triton 文件完整性

修复 `scripts/env_mindspeed.sh` 在 zsh 中依赖 Bash 专有变量导致项目路径错误的问题，按 shell 分别解析当前脚本，保留调用者工作目录、nounset 状态及 Python 覆盖设置；CANN/ATB 初始化和完整性检查失败均向调用者返回非零。新增 `utils/triton_integrity.py`，在 setup 安装后和环境加载时检查版本、导入路径及 Ascend RECORD 中所有 `triton/` 文件的大小和 SHA-256。共享文件以 Ascend 记录为准，生成字节码不参与检查。

- 针对性测试 **27 passed**：Bash/zsh、相对/绝对路径、带空格目录、严格模式、失败传播，以及共享文件覆盖、删除、截断、NPU 自有文件损坏、记录缺失、版本错误和 PYTHONPATH 遮蔽。
- 实际 `.venv-npu` 完整性校验通过：92 个文件、38 个带哈希的共享文件、705,129,052 字节。
- 在 `/tmp` 分别由 Bash 和 zsh、`set -eu`、清空外部 CANN/Python 路径后加载真实环境，均成功导入 torch_npu 并执行小型 NPU 矩阵乘法；确认 CANN/HCCL 来自项目 9.0.0，未加载系统 8.1 toolkit 库。
- 本轮未改 models、依赖版本和系统 CANN；没有重跑四卡完整训练。证据目录：`outputs/env_integrity_20261022/`。

## 2026-10-22 UTC：官方 MindSpeed GDN 与 CQ 行索引适配

**CPU 回归、四卡 GDN/CQ、真实权重连续/恢复逐位一致性和 checkpoint 管理对照通过。** 本节是当前适配状态，下方历史记录中的“当前实现”和设备边界描述均指各自记录日期。

证据目录为 `outputs/gdn_backend_20261022T0728/`。本轮在保留原生 Qwen3.5 代码结构的前提下，只修改 `models/backbone_qwen35.py`：CQ 扫描显式选择 `native` 或 `mindspeed`，事件排序及逆排序使用带 batch 偏移的整行 `index_select`。短卷积窗口、content 写入、query 只读事件、Attention、LM head 和 Wan 条件对齐规则未改变；没有新增自定义反向或复制上游 GDN kernel。去掉标注的 CQ 区块仍可恢复原生参考源码。

`modeling_umm.py`、`generator_wan21.py`、包初始化及 `models/references/` 共 7 个文件与本轮修改前的 SHA-256 完全一致，见 `unchanged_models_verification.json`。模型之外新增 backend 配置、加载时内外配置同步、实际 kernel 身份审计及严格恢复合同；生产和 Ascend 测试配置显式选择 MindSpeed，通用配置默认 native。没有静默后端回退；更换 backend、源码身份或 CQ 索引实现版本不能作为相同计算条件下的精确恢复。

| 验证 | 设置与结果 | 证据 |
| --- | --- | --- |
| CPU 回归 | 最终 167 passed、18 skipped；跳过项为显式要求 NPU 的测试，含新增 launcher 环境回归 | `cpu-final-complete.xml`、`cpu-final-complete.log` |
| 四卡官方 GDN | 每 rank 8 项通过；Qwen3.5-2B 实际 GDN 形状 H=16、D=128、B=2，长度 1/63/64/65/127/128/129；比较输出和 Q/K/V/g/beta 梯度，另覆盖跨 chunk 只读事件 | `cq-final/rank_0.xml` 至 `rank_3.xml` |
| 四卡 CQ 集成 | 每 rank 6 项通过；小型原生 Qwen/ViT/Wan 联合模型使用实际 MindSpeed kernel，覆盖 content 原生数值一致性、无泄漏、query 独立性、同位置 LM/Wan 对齐、梯度检查点及文本训练/推理对齐 | `cq-final/`，每 rank 合计 14 passed、0 skipped |
| 环境实际运行 | 官方 GDN 前向及全部输入反向有限；普通推理命令行的包导入通过 | `environment_gdn_smoke.json` |
| 跨后端数值诊断 | FP32 原生扫描、独立 FP64 递推及官方 BF16 kernel 对照；固定输入重复运行 20 次输出与输入梯度均逐位相同 | `parity65_diagnostic.json` |
| 只读梯度诊断 | 未来位置的反向累加输入及独立 FP64 反向求和均为精确零；官方 FP32 反向累加末端残差受舍入误差界约束 | `readonly_g_roundoff_diagnostic.json` |
| 四卡真实权重训练及恢复 | 连续 4 步与中断 2 步后自动恢复至 4 步，全部指标及 5,480 个状态张量逐位一致；最近 N 个清理、日志隔离通过 | `four_rank/verification_report.json`、`checkpoint-comparison.log` |
| 导出权重推理入口 | 两个命令均退出 0，实际使用 MindSpeed/BF16；T2V 两步采样得到有限 latent `[16,3,54,96]`；V2T 两 token 输出 `00`，仅确认链路，不代表生成质量已解决 | `inference_report.json`、`t2v-inference.log`、`v2t-inference.log` |

四卡 kernel 测试与四卡联合模型测试均显式选择 MindSpeed，并检查实际调用，不能由 native fallback 代替通过。真实权重短程训练另行验证，与小模型单元测试分别留存证据；不把历史后端的验证结果计作本轮通过证据。

### 环境与依赖边界

使用固定 MindSpeed 提交 `ff884722a3ce7079fbafe3cb97a3287311f58a5b` 中的 [`chunk_gated_delta_rule`](https://github.com/Ascend/MindSpeed/blob/ff884722a3ce7079fbafe3cb97a3287311f58a5b/mindspeed/core/ssm/npu_chunk_gated_delta_rule.py)，没有修改其源码。新增社区 Triton 3.2.0 与 Triton-Ascend 3.2.1，Ascend wheel 按锁定 SHA-256 最后安装，以保留两者共享目录中的 NPU 实现；补齐宿主机 `build-essential`，供 Triton 编译使用。

原有 CQ 环境 Python 3.12.13、CANN 9.0.0、PyTorch 2.7.1+cpu、torch_npu 2.7.1.post4 保持不变。保留 NumPy 1.26.0 以满足 MindSpeed 的版本要求；Triton-Ascend 的元数据要求 NumPy 1.26.4，两者冲突作为明确例外写入依赖锁。因此 `pip check` **不是全绿**；安装脚本只接受这一已记录冲突，其他依赖冲突仍报错。当前通过的是这台 910B 主机的测试组合，不据此声明该元数据冲突对所有功能均无影响。

### 首轮失败定位与数值判据

首轮检查有两个失败，均保留原始日志并完成独立诊断后才调整对应断言；没有改动上游 kernel 或把逐位训练恢复比较改成近似比较。

1. **长度 65 的跨后端 g 梯度比较。** 一个接近零的梯度点绝对差约 `3.8e-3`，原来的单点相对/绝对误差断言失败。与原生 FP32 扫描和独立 FP64 递推比较后，整体输出及输入梯度相对 L2 误差约为 0.36%–0.48%，而固定 kernel 重复 20 次全部逐位相等。这支持该用例的差异来自 BF16 中间舍入与抵消，不能据此归因为运行间非确定性。最终检查同时采用 MindSpeed-MM 的逐元素 `atol=rtol=0.01` 与 FLA 的整体相对 RMS 限制：输出 `<0.005`、Q/K/V 梯度 `<0.008`、g/beta 梯度 `<0.02`。两种条件必须同时满足；收紧整体误差约束是为了避免近零点的绝对容差掩盖全张量的误差。此判据用于不同实现间的数值一致性，不用于同实现连续/恢复的逐位比较。

2. **只读事件之后的原始 g 梯度。** 官方反向使用 FP32 `-cumsum(g) + sum(g) + g` 实现反向累加，未来位置出现最大 `2.9802322387695312e-8` 的消减残差。直接捕获该操作的输入后，未来位置全为精确零；独立 FP64 反向求和也为精确零，说明残差产生于最后的求和组合。测试只对该操作输出的 g 梯度采用 `gamma_130 * sum(abs(chunk))` 舍入界，其中 `gamma_n = n*u/(1-n*u)`、`u = FP32 eps/2`，130 对应两个长度 64 的归约及加减。其余梯度仍要求精确零；模型输入的目标梯度、改动目标内容后的 query 输出、改动单个 query 后其他 query 的输出与梯度仍保持零容差要求。诊断中所有残差均在该界内，最大误差界约 `6.13e-6`。

此外，首次四卡测试暴露了 UMM 与 Megatron 同名 `tests` 包的导入冲突；通过显式的 UMM `tests/__init__.py` 和两个 fixture 的绝对导入解决。该修正只影响验证代码的模块定位。

### 真实权重连续训练与自动恢复

使用既有 `tests/validation/test_mindspeed_resume.sh`、真实本地 Qwen3.5/Wan 权重及 `temp_data/slider_stop.jsonl`，4 张 910B2C，TP=PP=CP=1、DP=4，BF16、batch=1、累积=2、EMA 关闭、确定性模式，两种通信重叠均开启。输入为 9 帧、4 fps、768×432，在线 Qwen ViT/Wan VAE，累积窗口包含 T2V 与 V2T。

- 连续训练 4 步；独立实验第 2 步中断，自动恢复至第 4 步。4 步的全局目标、文本/视频 loss 和梯度范数全部相同；第 4 步全局目标为 `2.0616281270980834`。
- 第 2 步仅恢复不更新的对照，以及第 4 步继续训练对照，各比较 5,480 个张量、14,076,855,040 个元素与所有其他状态，差异均为零。包含 BF16 模型、FP32 master weights、Adam、scheduler、四卡 RNG、通信状态及数据游标。严格逐位断言通过，未使用近似容差。
- 两个实验只保留 checkpoint-3/4，latest 指向 4；伪造的未完成 checkpoint-3 被跳过并隔离。旧日志 16 个文件哈希不变，恢复新增 attempt，显式日志目录覆盖生效。

训练子脚本全部正常退出后，最初的父 shell 在运行比较器时因没有继承子进程的 CANN 环境而报 `libhccl.so` 缺失。已将 `source env_mindspeed.sh` 提前到父脚本，统一使用 `$PYTHON_BIN`；新增隔离 launcher 回归，并验证从空动态库路径加载 torch_npu/HCCL。实际状态比较加载环境后单独重跑并以 0 退出，没有重复或更改训练结果。原始失败日志仍保留在 `four_rank.log`。

此前裸模型梯度问题在本轮完整训练对照中未再复现。该结论限定于上述输入、种子、四步预算与锁定环境，不等同于所有形状或长期训练的确定性保证。日志仍有上游参数收集顺序影响重叠效率、ArgSort 使用 AiCPU、短卷积使用 Torch 参考实现的提示；本轮验证正确性，不据此声称通信与算子性能已充分优化。

额外对导出权重运行普通 `scripts.generate`：先在前向前受控导入后端，再将模型参数与 MCore 训练对齐为 BF16，避免仅 autocast 时保留 FP32 embedding/残差流。真实 T2V/V2T 命令运行通过；T2V 本次只保存 latent，没有复验 VAE 解码。V2T 仍输出 `00`，本轮未进行质量训练或据此判定其唯一原因。小模型的同位置 query/labels/LM head 及训练推理对齐检查另有独立测试通过。

## 2026-10-21 UTC：MindSpeed / Megatron Core 适配与四卡验证

**主链路训练、状态恢复、管理功能与 CQ 检查通过；继续训练的严格逐位一致性未通过。** 没有放宽比较容差或将后者标为通过。

本轮证据目录为 `outputs/mcore_migration_20261021T182149Z/`，总报告为其中 `verification_report.json`；原始日志、逐张量比较、环境记录、修改前源码状态和诊断脚本均保留。

本次新增 `distributed/megatron.py`、`distributed/communication.py`，通过 MCore DDP/分布式 AdamW 接入普通 UMM 模块，TP=PP=CP=1、DP=4。默认开启 `overlap_grad_reduce` 和 `overlap_param_gather`，固定条件分支的通信次序；最后一个微批未使用的参数只补就绪事件，不清空此前累积的梯度。Qwen 函数式短卷积的参数读取由训练侧父模块 hook 等待收集完成。

训练 loss 改为整个 DP/累积窗口内按有效文本 token 数、目标视频数归一化，不再额外除以累积次数；原有角色学习率、WSD、CFG、flow repeats、冻结控制保留。checkpoint 新增完整 BF16 模型与各 rank 的 FP32 master、Adam、scheduler、通信就绪状态，复用既有事务保存、自动恢复、最近 N 个清理和分类校验。

`models/` 仅修改获准的 `generator_wan21.py`：三个时间分支 Linear 使用保持 FP32 算术的子类，通过子模块 forward 触发参数收集 hook。FP32/BF16 下与原函数式计算的输出和梯度零容差一致。其余 7 个 Python 文件（含 backbone、modeling_umm 和 references）SHA-256 与本轮修改前完全相同，见 `models-verification.json`。

环境按 CQ 锁定为 Python 3.12.13、PyTorch 2.7.1+cpu、torch_npu 2.7.1.post4、CANN 9.0.0；四个上游源码提交与 `configs/deps.core17.lock.json` 相同且无本地源码补丁。UMM 保留 Transformers 5.19.0、Diffusers 0.41.0；依赖一致性检查通过。旧环境保留在 `.venv-npu-before-core17-20261021T182305`，训练脚本选择 `.deps/Ascend90/cann`，没有覆盖系统 CANN 8.1.RC1。

| 验证 | 设置与结果 |
| --- | --- |
| CPU 回归 | 150 passed；包括全局监督数归一化、时间分支 FP32 数值/梯度、通信顺序及管理回归 |
| 四卡通信/优化器 | 每 rank 三次更新，含 rank 间不同分支、整个窗口缺席的模块；与显式全局 batch AdamW 参考的 BF16 参数最大差为 0，DP 副本一致，master/Adam 恢复完全一致 |
| 四卡 CQ | MindSpeed 初始化后每卡 26 passed；覆盖 content 原生一致性、整段视频无泄漏、query 独立性、同位置标签及训练/推理对齐 |
| 默认训练入口 | `tests/validation/test_train.sh`，仅覆盖输出目录，真实 Qwen/Wan 权重和 temp_data，四卡、9 帧、4 fps、768×432、BF16、batch=1、GA=2、在线 ViT/VAE、EMA 关闭；2 步，退出码 0 |
| 连续/自动恢复 | 连续 4 步；另一实验第 2 步停止后自动恢复至第 4 步，均退出码 0，两种重叠始终开启 |
| 精确状态恢复 | 第 2 步加载后直接保存，5,480 个张量、14,076,855,040 个元素及其余标量全部相等，包含四卡 RNG、游标、master 和优化器状态 |
| 保留与隔离 | continuous/interrupted 均只保留 checkpoint-3/4；latest 指向 4；伪造的不完整 checkpoint-3 被跳过并隔离；旧日志 16 个文件哈希不变，恢复新建 attempt，显式 log_dir 覆盖生效 |

默认入口的 T2V 视频 loss 为 `2.569178 → 1.527452`，V2T 文本 loss 为 `8.735689 → 6.574328`。这是链路检查，不是收敛/质量评估。NPU 真实权重验证关闭 EMA；没有将 CPU EMA 回归作为 NPU EMA 验证。

严格连续训练对照的结果：

- 独立启动的连续/中断实验，首步目标完全相同，但首步梯度范数已为 `907.6857299804688` 与 `907.68017578125`，差异发生在任何恢复之前。
- 第 4 步全局目标为 `1.9304541110992433` 与 `1.9015068769454957`。模型权重最大绝对差为 `4.8828125e-4`；backbone/generator 的整体相对 L2 差分别为 `2.6940e-4`、`1.7238e-4`，冻结 ViT 完全一致。
- 从完全相同的第 2 步 checkpoint 再次重放，第 3 步目标均为 `1.9037850379943848`，梯度范数仍有细微差异；第 4 步目标为 `1.9015068769454957` 与 `1.9025204181671143`，权重最大绝对差同为 `4.8828125e-4`。
- 上述比较的 RNG、已消费游标、调度器和其他非张量状态均一致；优化器浮点状态也有差异。`tests/validation/verify_mcore_run.py` 保留严格断言，当前会因继续训练不完全相等而返回非零，详见 `full_validation_v2/*-comparison.json`，不是静默通过。

进一步诊断没有修改模型源码：固定权重、输入和 RNG，不做更新，分别重复单次 T2V/V2T 反向均完全一致；四卡两任务累积重复仍有差异，关闭两种重叠也可复现。去掉 DDP、优化器和 checkpoint，用裸模型交替两个任务并累积 FP32 梯度，仍有 297 个梯度张量不同，loss 完全相同。原生 Qwen eager 配置下仍有 244 个梯度张量不同，因此目前无法只归因于 SDPA。当前 torch_npu 下请求 `sdpa_kernel(MATH)` 的小探针仍显示 `NpuFusionAttentionV3Backward0`，不能据该开关声称实际切换了数学内核。独立的四卡 FP32 HCCL ReduceScatter 探针在 40,000,000/512,778,496 元素、同步/异步各次重复均完全一致。**具体数值差异算子尚未定位，严格继续训练一致性仍是未完成项。**

本轮修复了 MegatronAdaptor 注册的占位 FLA 被 Transformers fallback 捕获、误入 MCore GDN 并触发 `NameError: l2norm` 的导入次序问题：在 adaptor 初始化前加载 UMM 原生实现，保留其既有 kernel dispatch。另有一次早期冷启动出现 AICPU `IndexPutV3`/507018，后续普通异步训练及重放未复现，不能声称已定位其原因。临时诊断脚本的训练模式/JSON 输出错误已修正，不涉及生产模型。

复验命令（新实验名避免覆盖已有记录）：

```bash
bash tests/validation/test_train.sh --set training.exp_id=my_mcore_smoke
# 包括训练、恢复、清理、隔离及逐张量严格比较；当前会如实报告一致性失败。
bash tests/validation/test_mindspeed_resume.sh outputs/my_mcore_resume_check
source scripts/env_mindspeed.sh
python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.validation.validate_cq_mcore --output outputs/my_cq_check
```

Ruff、compileall、shell 语法及 `git diff --check` 通过。

## 2026-10-21：实验日志与 checkpoint 管理、四卡恢复对照

实现默认 `outputs/<exp_id>/logs/` 与 `outputs/<exp_id>/checkpoints/`，完整实验目录及两个分支可分别覆盖。新增每次启动独立 attempt、控制台捕获与退出码、自动恢复、版本化训练契约、最近 N 个 checkpoint 保留和写入锁。原生模型及 CQ 计算未改，`models/` 全部 8 个 Python 文件（含 references）修改前后 SHA-256 一致。

验证目录：`outputs/management_verify_20261021T224900/`；完整结果为其中 `verification_report.json`，原始控制台、JSONL 指标、状态比较脚本及逐张量差异均保留。

四卡均为 910B2C、64 GiB；使用既有 `tests/validation/test_train.sh` 和 `temp_data/slider_stop.jsonl`，在线 ViT/VAE，9 帧、4 fps、768×432、BF16、ZeRO-2、每卡 batch=1、累积=2。NPU 验证配置关闭 EMA。

| 对照 | 操作 | 结果 |
| --- | --- | --- |
| continuous | 连续训练到第 2 步，keep=2 | 正常退出，保留 checkpoint-1、checkpoint-2 |
| split | 独立启动，第 1 步保存退出；同实验 --auto-resume 到第 2 步，keep=1 | 两次均退出码 0，只保留 checkpoint-2；旧日志 8 个文件哈希不变，新日志为独立 attempt |
| replay | 将 continuous 的完整 checkpoint-1 作为不可变快照，--auto-resume 到第 2 步 | 排除初始状态差异；显式日志/checkpoint 目录覆盖生效，keep=1 清理成功 |
| restore_only | 从 continuous 的 checkpoint-1 加载并重新保存，不执行更新 | 1,846 个训练状态张量及所有标量完全一致，含模型、优化器、调度器、各 rank RNG 和数据游标 |

继续训练后的比较必须区分数值与状态：

- 四个 rank 的 RNG、已消费的数据游标、步数、调度器与优化器标量配置一致。
- 同一个 checkpoint 起点：第 2 步总 loss 均为 `1.1028074324131012`；导出权重最大绝对差 `2.44140625e-4`，相对 L2 差 `8.491381186084957e-6`；优化器浮点状态也有差异。
- 独立启动的连续/中断两组：第 2 步总 loss 分别为 `1.1028074324131012`、`1.1023704558610916`；权重最大绝对差 `2.44140625e-4`，相对 L2 差 `5.422561624487938e-5`。两组在尚未恢复的第 1 步就已有浮点差异。
- **加载/保存本身逐位保持训练状态，但当前四卡 BF16 更新未达到逐位确定性。** 控制实验支持差异出现在后续计算，未定位具体算子；不声称长期收敛、生成质量或 NPU EMA 等价性已经验证。

CPU 与故障回归：相关 41 项完整运行通过；新增 3 个损坏 JSON 类型的回退用例后，对 checkpoint/管理相关 18 项局部复验全部通过（与前 41 项有重叠）。覆盖训练参数变更拒绝、管理参数变更接受、恢复失败不覆盖旧元数据、最新损坏回退/全部损坏拒绝、保留 N=1/N=2/null、失败保存不删旧 checkpoint、隔离损坏文件、写入锁和符号链接保护。CPU 连续/中断训练权重、优化器、调度器、RNG 和游标零容差一致，独立 EMA 恢复测试及双进程 Gloo 恢复通过。Ruff、shell 语法和 diff 空白检查通过。

本次修正了 NPU 环境下 CPU/Gloo 测试最后一个冗余 Accelerator barrier 的设备误判；导出阶段已有 `run_io_phase` 全 rank 同步，因此移除冗余 barrier。训练前向、loss、content/query 规则保持原样。

## 2026-10-21：910B 默认测试入口复验

已将上一轮实际验证通过的 Python 依赖补齐到 `.venv-npu`，共 70 个包，版本记录在 `requirements-npu-tested.txt`；`uv pip check` 通过。保留原有 PyTorch 2.8.0+cpu、torch_npu 2.8.0.post2 和主机 CANN 8.1.RC1，补齐 Accelerate 1.15.0、DeepSpeed 0.19.8、Transformers 5.19.0、Diffusers 0.41.0 等依赖。这里记录的是当前主机实测组合，并非通用 CANN/torch_npu 配套建议。

直接执行以下命令，未额外指定 Python、HCCL、attention 后端或训练参数：

```bash
bash tests/validation/test_train.sh
```

- 脚本默认选择 `.venv-npu/bin/python`、`HCCL_INTRA_ROCE_ENABLE=1` 和四卡 ZeRO-2 配置；测试 YAML 设置 `wan_attention_backend=npu`。四个训练 rank 的进程环境均已核对。
- 使用 `temp_data/slider_stop.jsonl` 的 T2V/V2T 视图，9 帧、4 fps、768×432，BF16、每卡 batch=1、累积=2、EMA 关闭，完成 **2 个 optimizer 更新，退出码 0**。
- T2V 视频 loss：`2.560015 → 1.581914`；V2T 文本 loss：`8.730907 → 6.023696`。这只是合成样本链路验证，不代表质量或长期收敛。
- 默认目录 `output/ascend_smoke` 已保存 config、processor、完整 `checkpoint-2` 和 `weights.pt`。重跑时使用新输出目录，或使用 `--resume`；保留原有防覆盖检查。
- 本次只修改测试入口、测试配置、依赖记录和文档；`models/` 源码与原生参考文件哈希保持一致。

日志与环境证据：`output/ascend_default_verify_20261021T110710Z/`；训练指标：`output/ascend_smoke/metrics.jsonl`。上一轮临时环境的四卡恢复、在线/缓存评估、T2V/V2T 推理及 NPU 数值检查见 `output/ascend_verify_20261021T093257Z/verification_report.json`，本次没有重复这些检查。

## 2026-10-21：批内统一表示与 flow 重复抽样

**131 通过，4 跳过，0 失败，58.02 秒。** 本次仍为 CPU 测试；四项跳过为 Wan 加速器测试，没有执行 Ascend / DeepSpeed ZeRO-2 验证。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo \
  .venv/bin/python -m pytest tests -q --junitxml=/tmp/umm-video-batch-flow-regression.xml
```

- Qwen/Wan 各自统一缓存或在线编码，四种跨分支组合通过单样本和多样本 batch 检查；同一分支的批内混用明确拒绝，`prefer_cache` 不例外。
- `flow.repeats=3` 的多视频、不同 latent 尺寸、混合 T2V/V2T 及不同 CFG 标记检查通过：CQ backbone、LM head 各执行一次；重复条件与原视频顺序对应，时间/噪声独立，视频 loss 与逐组独立计算的均值及梯度一致。文本 CE 不重复，视频计数不膨胀。
- 默认 `repeats=1` 另与本次修改前的 loss、`UMM.forward` 和 backbone 输入适配函数作小模型对照。缓存与在线输入均得到完全相同的 loss、随机数状态及全部 99 个有梯度参数张量的梯度（零容差）。
- V2T 不受重复次数影响；双进程 CPU/Gloo 混合训练、EMA、checkpoint/恢复以 `repeats=2` 运行通过。旧 checkpoint 缺省重复次数按 1 兼容，显式改变次数仍拒绝精确恢复。
- 原有 content 原生一致性、整段视频无泄漏、query 独立性、训练/推理对齐检查全部通过。
- 模型源码严格等于用户确认的补丁：仅改 `backbone_qwen35.prepare_inputs`、`UMM.generate_video` 的像素筛选及 `UMM.forward` 的条件重复。Wan、原生参考源码、CFG 和短卷积/GDN/Attention 核心没有修改；Ruff 与 `git diff --check` 通过。

上述测试验证实现与数值契约，不代表完整权重在 910B 上的显存、吞吐或视频质量结果。

## 2026-10-20：主链路回归（历史记录）

**110 通过，4 跳过，0 失败，57.97 秒。** 环境为 Python 3.12、PyTorch 2.8.0+cpu、Transformers 5.19.0、Diffusers 0.41.0、torchvision 0.23.0+cpu、PyAV 19.0.1。四项跳过是未配置设备时的 Wan 加速器测试；本机没有 torch_npu / NPU，也未执行 DeepSpeed ZeRO-2。

```bash
OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 GLOO_SOCKET_IFNAME=lo \
  .venv/bin/python -m pytest tests -q --junitxml=/tmp/umm-video-final.xml
```

完整日志 `/tmp/umm-video-final.log`，JUnit `/tmp/umm-video-final.xml`。双进程测试需要本地 loopback；受限沙箱内的首次 Gloo 连接失败，开放本地通信后完整通过。

| 检查 | 证据 |
| --- | --- |
| 保留原生源码 | 去掉 `CQ BEGIN/END` 区块后逐字恢复 Qwen3.5 参考源码；Wan 实现与本次改造前逐字一致 |
| content 一致性 | 与原生模型逐层比较，覆盖 Full Attention、GDN 跨 chunk、不同后端和 BF16；CFG 不改变 content |
| query 独立性 | 改动单个 query 并检查其他 query 的输出及梯度；独立卷积窗口与原生 qkv 的 q 计算对照 |
| 整段视频无泄漏 | 改变目标视觉特征及前缀之后的所有 content embedding，视频 query 不变；V2T 文本输出仍对已知视频敏感 |
| 训练/推理对齐 | 同位置文本解码 logits、真实目标/空目标视频条件、固定初始噪声下的 Wan 输出一致 |
| 原生序列化 | native Processor 的 input_ids、pixels 与缓存路径逐项对照；labels 与 query 同位置，视觉 token/padding 忽略 |
| 在线/缓存组合 | Qwen/Wan 四种组合、批内不同表示和不同任务、非连续 pixel_video_ids、目标 latent 顺序 |
| 冻结与梯度 | 冻结 ViT 不进入优化器，在线解冻 ViT 可获得梯度；语言、Wan、condition_proj 分组正确 |
| 训练 CLI | 小型原生 Qwen/ViT/Wan 完成训练、导出、T2V latent 生成、V2T 文本生成与独立 loss 评估 |
| 精确恢复 | 单进程中断后继续训练，与连续训练逐参数零容差相等；恢复已消费游标，不计入预取 batch |
| 数据并行 | 两个真实 CPU/Gloo rank 运行混合 T2V/V2T、梯度累积、分片 EMA、checkpoint/恢复 |
| 数据与配置拒绝 | 缓存编码器/片段不一致、多视频记录、旧模型配置、解冻 ViT 配缓存等错误显式拒绝 |

另完成 Ruff、compileall、shell 语法及 `git diff --check`。运行时禁止导入 `models.references` 的测试通过。备份中 76 个源码文件的 SHA-256 均重新校验通过，原生参考文件没有改动。

## 本地真实资产检查

使用用户提供的本地 Qwen3.5-2B、Wan2.1-T2V-1.3B 权重，而非下载模型：

1. 原生 Wan VAE 权重经官方转换后严格加载；CPU 上完成 `[3,5,16,16] → [16,2,2,2] → [3,5,16,16]` 编解码，结果有限。
2. `temp_data/slider_stop.jsonl` 的真实 9 帧、4 fps、768×432 视频完成 Qwen ViT 与 Wan VAE 缓存生成及读回：Qwen `[1680,2048]`、grid `[5,28,48]`；Wan `[16,3,54,96]`。该 T2V 样本序列长 1790，padding 后 1792。
3. 真实 Qwen tokenizer/聊天模板下，V2T 去除 assistant 答案后的生成 prompt 与训练序列前缀逐 token 完全相同。
4. 通过新的 `training.train.load_model` 严格加载全部 **3,640,631,168** 个模型参数，CPU 上完成小尺寸联合 CE/flow 前向；loss 有限。相同视频布局在给出真实 content 特征和移除目标 content 时，Wan 条件零容差相等。这是加载/数值检查，不是全分辨率训练或质量评估。

缓存检查命令：

```bash
OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 .venv/bin/python -m scripts.prepare_latents \
  --config configs/tests/ascend_smoke.yaml \
  --manifest ../temp_data/slider_stop.jsonl --task t2v --kind both \
  --cache-dir /tmp/umm-video-real-cache --output /tmp/umm-video-real-cache/t2v.jsonl \
  --set training.device=cpu --set training.mixed_precision=no
```

临时证据：`/tmp/umm-video-real-cache.log`、`/tmp/umm-video-pretrained-forward.log`。这些路径是本次运行产物，不是代码的运行依赖。

## 设备验证边界

`configs/accelerate/zero2.yaml` 和 `tests/validation/test_train.sh` 的 910B 单节点四卡默认入口已按上文完成真实权重、真实视频的两步训练及保存验证。此前临时环境还完成四卡 ZeRO-2 checkpoint 恢复并验证重新导出权重哈希一致，以及 CQ/Wan 的 NPU 数值与梯度检查；不据此声称跨机器兼容、EMA 恢复或长期训练已验证。

当前 GDN 使用原生可移植 PyTorch 后备实现；没有声称其已达到 NPU 优化内核的吞吐。保留原生结构和数值约定后，可在目标设备性能数据基础上单独优化。正式视频质量、长期收敛和任务比例调优尚未验证。
