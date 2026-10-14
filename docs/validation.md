# UMM v3 验证记录

本文保存实际执行的验证快照。结果对应记录时的代码与环境，不代表所有设备、运行配置或后续修改都已验证。

## 模块归类验证：150 项通过

执行时间：`2026-10-14T03:55:38.203548+00:00`（UTC）。`codec.py`、`sampling.py` 已归入 `models/`，
配置解析已归入 `configs/config.py`，无调用的 `diagnostics.py` 已删除。
`configs/` 现在是 Python 包；安装包包含配置解析代码和四份实验/Accelerate YAML。
训练、评估和生成的命令行用法保持一致，导入路径已更新。

| 验证 | 结果 |
| --- | --- |
| 迁移前后实现 | 40 个保留的生产代码和入口文件，剔除导入语句后的 AST 完全一致 |
| 完整 CPU 回归 | 150 通过，0 失败，0 跳过，约 45.8 秒 |
| 训练 | CE/flow 联合目标、梯度与参数更新、checkpoint/EMA 恢复、真实 DataLoader worker 与双 rank Gloo 测试通过 |
| 生成 | 真实小模型 CLI 生成、一次 backbone condition、整图多步积分、测试 VAE 编解码及 posterior 协议检查通过 |
| 采样原版对照 | Euler/Heun × 无 CFG/固定 CFG/线性 CFG 共六组结果，均以 rtol=0、atol=0 通过 |
| 可编辑安装与入口 | 已离线重装；保留现有 Ascend PYTHONPATH，从项目外导入 8 个代表模块并完成 7 项入口检查 |
| wheel 安装 | 37 个 Python 模块、4 份 YAML 与源码逐字节一致；全部模块从 wheel 安装位置导入，两个实验配置校验通过 |
| 静态检查 | Ruff、compileall、shell 语法检查通过；AST 扫描未发现旧根目录模块导入 |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo \
PYTHONPATH=/tmp/umm-v3-vision-check UMM_SOURCE_SNAPSHOT=/tmp/umm-v3-source-193ec90 \
.venv/bin/python -m pytest tests -q --junitxml=/tmp/umm-v3-module-grouping-verified.xml
```

日志：`/tmp/umm-v3-module-grouping-verified.log`；JUnit：`/tmp/umm-v3-module-grouping-verified.xml`。
打包日志：`/tmp/umm-v3-module-grouping-wheel-build.log`。以上临时路径是本次运行证据。
本次仅调整模块归属，既有训练对齐修复与采样计算保持一致；原 `unified-mm` 仓库保持 clean。
验证使用 CPU 和测试资产，未执行真实 NPU/GPU 或生产大模型训练。下方为保留旧路径的历史快照。

## 目录展开验证：150 项通过

执行时间：`2026-10-14T03:35:55.769775+00:00`（UTC）。内层 `unified_mm/` 已移除；
`models/`、`data/`、`training/`、`evaluations/`、`utils/` 和四个通用模块直接位于项目根目录。
启动入口改为 `python -m training.train` 和 `python -m evaluations.eval`。

| 验证 | 结果 |
| --- | --- |
| 完整 CPU 回归（含原版数据对照、真实 DataLoader worker 与双 rank Gloo） | 150 通过，0 失败，0 跳过，约 44.8 秒 |
| 迁移前后计算实现 | 38 个核心 Python 文件剔除导入语句后的 AST 完全相同 |
| 可编辑安装 | 已离线重装；从项目外导入 9 个代表模块，执行 7 项训练/评估/工具入口检查 |
| 当前 Ascend 环境的 config 命名冲突 | 显式配置根目录 package-dir 后，保留环境 PYTHONPATH 的目录外启动检查通过 |
| wheel 构建与安装 | 包内 37 个 Python 模块与源码清单完全一致；安装后全部从 wheel 目录导入，无旧包、测试或虚拟环境文件 |
| 静态检查 | Ruff、compileall、shell 语法检查通过 |

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo \
PYTHONPATH=/tmp/umm-v3-vision-check UMM_SOURCE_SNAPSHOT=/tmp/umm-v3-source-193ec90 \
.venv/bin/python -m pytest tests -q --junitxml=/tmp/umm-v3-flat-verified.xml
```

日志：`/tmp/umm-v3-flat-verified.log`；JUnit：`/tmp/umm-v3-flat-verified.xml`。
打包日志：`/tmp/umm-v3-flat-wheel-build.log`。以上临时路径是本次运行证据。
可选原版数据对照测试仅在 fixture 内扩展包搜索路径，加载原版特有的 helper；
生产模块仍不依赖旧仓库。此前五项训练对齐修复保持有效。
本次未执行真实 NPU/GPU 训练；硬件验证边界与下方记录相同。

## 五项训练对齐修复：150 项通过

执行时间：`2026-10-14T03:13:20.978162+00:00`（UTC）。来源仍为
`unified-mm@193ec90300af5fee4bc4b7e8a426492a908a0447`，backbone、Z2 generator 和模型组装实现未改动。

| 修复 | 实现与验证 |
| --- | --- |
| EMA 类型冲突 | 在后端 prepare 完成后构建 layout；真实训练入口分别以 FP32、模拟后端 BF16 转换运行训练、保存、恢复，EMA 影子权重保持 FP32 |
| 输入扰动精度 | FP16/FP32 cache 分别配合 FP32/BF16 投影，与固定原版 `_build_x0_inputs_embeds` 零容差对照，并检查后续 RF 输入、目标与随机状态 |
| rank 随机数 | 入口使用 `device_specific=True`；双 rank Gloo 使用真实 Accelerate rank 与全局随机数，对照显式 `seed + rank` 的梯度、参数更新 |
| 验证协议 | 默认每类两张；直接执行原版抽样函数，对照含两位数 class_index 和重新排序 cache 的样本顺序；训练内/独立 loss 指标一致；分布式与单进程验证一致 |
| generator 冻结范围 | 直接执行原版 `_apply_trainable_scope` 对照可训练集合和优化器覆盖，确认更新后整个 backbone 保持冻结、condition projector 和 generator 正常更新 |

测试总数 **150**，通过 **150**，失败 **0**，跳过 **0**，耗时约 **45.1 秒**。
下方 138 项快照中的全部测试继续通过，新增 `tests.test_training_alignment` 的 12 项。
`tests.test_cli` 补充训练内与独立 raw 权重 loss 的逐项一致性检查；
`tests.test_ddp_training` 补充 rank 独立 RNG 与不等长任务的分布式验证。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo \
PYTHONPATH=/tmp/umm-v3-vision-check UMM_SOURCE_SNAPSHOT=/tmp/umm-v3-source-193ec90 \
.venv/bin/python -m pytest tests -q --junitxml=/tmp/umm-v3-five-fixes-verified.xml
```

本次使用与下方快照相同的 CPU 环境，开放本地 IPC/loopback 完成真实 DataLoader worker 和 Gloo 测试。
日志：`/tmp/umm-v3-five-fixes-verified.log`；JUnit：`/tmp/umm-v3-five-fixes-verified.xml`。
临时路径仅为此次运行证据，复现时需准备对应源码快照与 torchvision 依赖。

Ruff、compileall、shell 语法检查通过；unified、image、joint 和 generator-only 配置 dry-run 通过。
原仓库保持 clean，HEAD 仍为 `933ae59acc6cc08e9d40f5b1f6316bcd7ac99900`。

loss 验证的 `evaluation.image_classes` 需指向原版 `imagenet1k_synthetic_v1/t2i/classes.json`。
两个示例配置已配置该资产位置。`loss_image_per_class=null` 显式选择全量 loss 验证，
生成评估保持全量；统一验证种子和可选上限见 README。

BF16 测试模拟了后端的参数类型转换，**未实际执行 NPU/GPU 或 DeepSpeed ZeRO-2**。
真实大模型、生产语料、官方 KL16/Inception 和质量/性能结论仍适用本文末尾的未验证边界。

## 上一次完整测试快照：138 项通过

记录时间：2026-10-13 23:22:41.501（Asia/Shanghai）。

原始 UTC 时间：`2026-10-13T15:22:41.501200+00:00`。

来源参考提交：`193ec90300af5fee4bc4b7e8a426492a908a0447`。

| 项目 | 结果 |
| --- | --- |
| 测试总数 | 138 |
| 通过 | 138 |
| 失败 | 0 |
| 跳过 | 0 |
| 真实本地多进程 | 已执行 |

### 环境

| 项目 | 版本或说明 |
| --- | --- |
| Python | 3.12.3 |
| 平台 | `Linux-5.15.0-119-generic-x86_64-with-glibc2.39` |
| 设备 | CPU |
| torch | 2.8.0+cpu |
| transformers | 5.19.0 |
| accelerate | 1.15.0 |
| numpy | 1.26.4 |
| pytest | 9.1.1 |
| safetensors | 0.8.0 |
| PyYAML | 6.0.3 |
| torchvision | 0.23.0+cpu |

### 原始执行命令

命令中的 `/tmp` 路径是当时的临时依赖、源码快照和测试输出位置。复现时需要准备对应文件并替换路径。

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 GLOO_SOCKET_IFNAME=lo PYTHONPATH=/tmp/umm-v3-vision-check UMM_SOURCE_SNAPSHOT=/tmp/umm-v3-source-193ec90 .venv/bin/python -m pytest tests -q -rA --junitxml=/tmp/umm-v3-four-fixes-verified.xml
```

Run with local IPC/loopback enabled. Restricted sandbox blocks Gloo and worker communication.

CPU torchvision 0.23.0+cpu was installed into /tmp/umm-v3-vision-check from the official PyTorch index for the direct source preprocessing comparison; base package dependencies unchanged.

### 模块结果与上次快照对比

| 模块 | 最近一次通过 | 上次通过 |
| --- | --- | --- |
| `tests.test_checkpoint_runtime` | 7 | 4 |
| `tests.test_cli` | 9 | 8 |
| `tests.test_data` | 23 | 21 |
| `tests.test_data_source_parity` | 7 | 5 |
| `tests.test_ddp_training` | 2 | 1 |
| `tests.test_evaluation_runtime` | 11 | 11 |
| `tests.test_joint_training` | 8 | — |
| `tests.test_mask_initialization` | 7 | — |
| `tests.test_model_core` | 21 | 21 |
| `tests.test_preprocessing` | 8 | — |
| `tests.test_training` | 20 | 19 |
| `tests.test_weights` | 15 | 15 |
| 总计 | 138 | 105 |

### 其他检查

| 检查 | 最近一次记录 | 上次记录 |
| --- | --- | --- |
| Ruff | passed | passed |
| compileall | passed | passed |
| Shell 语法 | passed | passed |
| 配置 dry-run | passed: unified.yaml, image.yaml, unified.yaml with schedule=[joint] and accumulation=1 | passed |
| 可编辑安装 | 本次快照未单独记录 | passed |
| 仓库外导入 | 本次快照未单独记录 | passed |
| 原仓库状态 | clean, original HEAD 933ae59acc6cc08e9d40f5b1f6316bcd7ac99900 unchanged | clean, original HEAD 933ae59acc6cc08e9d40f5b1f6316bcd7ac99900 unchanged |

## 四项对齐工作的验证范围

### 同一前向的 CE 与 flow 联合监督

One backbone forward supports CE and flow with independent masks and denominators; joint loader, trainer, validation, CLI, resume and DDP exercised.

对应测试：

- `tests/test_joint_training.py`
- `tests/test_training.py`
- `tests/test_ddp_training.py`
- `tests/test_cli.py`

### 图像预处理

No BOX stages; source Bicubic Resize, CenterCrop rounding and VAE normalization retained.

对应测试：

- `tests/test_preprocessing.py`
- `tests/test_cli.py::test_prepare_latents_cli_produces_readable_scaled_posterior_shards`

直接对照 torchvision 的案例数：24。

Fixed pixel cases use a small VAE fixture; official KL16 asset execution remains unverified.

### Mask 初始化

copy_text by default, independent random optional, only for newly added base-model image tokens; full checkpoint embeddings preserved.

对应测试：

- `tests/test_mask_initialization.py`
- `tests/test_checkpoint_runtime.py`

Sharded native HF weights, tied/untied heads, grown/reserved vocab rows, and full-model restore.

### 独立 prompt 来源

Both presets explicitly configure separate train/val synthetic text indexes; T2I prompts and I2T captions tested through loaders.

对应测试：

- `tests/test_data.py`
- `tests/test_data_source_parity.py`

## 补充检查

| 范围 | 测试数 | 通过 | 生产代码变化 |
| --- | --- | --- | --- |
| evaluation weights label after final report correction | 11 | 11 | 未单独记录 |
| persistent worker test cleanup; workers explicitly joined with exit code zero; no abort stderr | 3 | 3 | 否 |

## 尚未验证

- GPU/NPU and DeepSpeed ZeRO-2 hardware execution
- Real full-size pretrained checkpoints and production corpus
- Official KL16 and Inception asset execution
- Full FID/IS quality, convergence, throughput and memory comparison
- Video support (not implemented)

CPU 小模型结果不能证明完整模型的 FID/IS、收敛、吞吐或显存等价。设备验收步骤见 [migration.md](migration.md#后续设备验收)。

## 上一次完整测试快照

记录时间：2026-10-13 00:33:42.103（Asia/Shanghai）。

原始 UTC 时间：`2026-10-12T16:33:42.103272+00:00`。

| 测试总数 | 通过 | 失败 | 跳过 | 真实本地多进程 |
| --- | --- | --- | --- | --- |
| 105 | 105 | 0 | 0 | 已执行 |

原始执行命令：

```bash
OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 UMM_SOURCE_SNAPSHOT=/tmp/umm-v3-source-193ec90 .venv/bin/python -m pytest tests -q -rA
```

网络与通信要求：None; local IPC/loopback enabled for workers and Gloo.

模块计数及其他检查保留在上方对比表中；最新快照补充了 joint 监督、mask 初始化、预处理和 prompt 来源等覆盖。
