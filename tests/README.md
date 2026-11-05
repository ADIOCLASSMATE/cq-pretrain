# 开发验证

日常训练、评估、推理和环境准备使用 `scripts/`；这里保留开发回归及需要真实设备的显式验证。
所有命令从项目根目录执行，统一使用 `.venv-npu`。

## CPU 回归

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-npu/bin/python -m pytest tests -q
```

默认不运行需要显式指定设备的测试。`conftest.py` 将偶发 Ascend 编译工作文件放到
`.cache/ascend-work/<主机名>/`；显式 `ASCEND_WORK_PATH` 优先，不要求为 CPU 测试加载 CANN。

## 910B 验证

```bash
# 两步真实权重训练，默认使用 temp_data。
bash tests/validation/test_train.sh

# 连续训练与中断/恢复对照，同时检查 checkpoint 清理和日志隔离。
bash tests/validation/test_mindspeed_resume.sh outputs/my_resume_check

source scripts/env_mindspeed.sh
# CQ / GDN 不变量；4 个 rank 各自执行相同的检查。
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.validation.validate_cq_mcore --output outputs/my_cq_check
# 小模型通信与分布式优化器验证。
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.validation.validate_mcore --output outputs/my_transport_check
# 16 卡混合 V2T/T2V、GA=2、重计算和精确恢复；加 --full-weights 加载正式初始权重。
"$PYTHON_BIN" -m torch.distributed.run --standalone --nproc-per-node=16 \
  -m tests.validation.validate_shared_runtime --output outputs/my_shared_check
# Wan 原生权重和完整生成器前向/反向，不加载 backbone 或 VAE。
"$PYTHON_BIN" -m tests.validation.validate_wan21 \
  --checkpoint /inspire/sj-ssd3/project/high-dimensionaldata/public/models/Wan2.1-T2V-1.3B \
  --device npu:0 --report outputs/my_wan_check/report.json
```

验证会产生新的测试 checkpoint。旧测试权重和诊断张量已清理，历史报告仍保留。
`validate_mcore.py` 的检查对象是小模型状态；正式模型连续/恢复对照使用 `test_mindspeed_resume.sh`。

| 文件 | 用途 |
| --- | --- |
| `validation/test_train.sh` | 真实数据、真实权重四卡短程训练 |
| `validation/test_mindspeed_resume.sh` | 连续、恢复、保留数量和日志隔离验证 |
| `validation/validate_cq_mcore.py` | 加载 NPU 算子后运行 AscendC / MindSpeed CQ / GDN 检查 |
| `validation/validate_mcore.py` | 通信、条件分支和优化器更新对照 |
| `validation/validate_shared_runtime.py` | 16 卡真实 UMM 混合分支、梯度累积、优化器和恢复 |
| `validation/validate_wan21.py` | Wan 权重和生成器设备验证 |
| `validation/verify_mcore_run.py` | 校验恢复脚本产生的完整结果 |
| `validation/compare_mcore_checkpoints.py` | 逐张量比较两个 MCore checkpoint |

后两个工具通过 `python -m tests.validation.<模块名> --help` 查看参数。
环境启动必须执行的 Triton 完整性校验在 `utils/triton_integrity.py`，不依赖 tests：

```bash
.venv-npu/bin/python -m utils.triton_integrity
```

## 统一 sample/score 回归

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 \
  .venv-npu/bin/python -m pytest \
  tests/test_evaluation_config.py tests/test_eval_sampling.py tests/test_eval_scoring.py \
  tests/test_eval_dispatcher.py -q
```

配置测试检查任务文件相对路径、训练/评估配置校验分离、日志层级和管理参数恢复合同。采样测试比较旧接口与公共 sample 的最终 latent/token、产物选择、稳定 seed 和任务分片；loss 测试比较训练 validation 与独立 scorer 的逐批统计和指标。真实四卡与 temp_data 验证需明确选择 NPU 配置，CPU 通过不能替代实际设备结果。

日常评估统一使用 `scripts/evaluate.sh` 和 `configs/evaluations/*.yaml`。采样续跑测试可通过 `--max-new-samples` 有限生成，再对同一 `--run-dir` 执行 `--resume`；检查结果无重复、旧产物摘要保持不变、每次启动有独立 `logs/eval/<eval_id>/<attempt_id>`。训练日志检查使用 `logs/train/<attempt_id>`，processor 检查位置为 `checkpoints/exports/processor/`。逐批 loss 对照读取评估 attempt 下 `<任务名>/batches.rank-<N>.jsonl`，汇总结果读取 `evaluations/<eval_id>/<任务名>/metrics.json`。仅 score 的测试需要确认不加载 UMM、不读取原 checkpoint，并拒绝修改计算协议。

VBench 仍位于 `evaluations/benchmarks/vbench/`。本阶段接口与协议测试不代表官方评分环境已完成适配；实际评分要求单独的 CUDA/VBench 环境。

真实 NPU 数值对照使用下面的独立验证工具，需已有完整导出；它运行实际 `Trainer.validate()` 路径和原生成调用对照，不创建优化器或修改模型：

```bash
source scripts/env_mindspeed.sh
.venv-npu/bin/python -m torch.distributed.run --standalone --nproc-per-node=4 \
  -m tests.validation.verify_unified_evaluation \
  --config /data/run.yaml --checkpoint /data/weights.pt --output /data/parity \
  --set sampling.steps=2 --set sampling.max_new_tokens=2
```

配置需包含 T2V/V2T 两类 validation sources；全局 max_batches 必须覆盖两类任务。CPU 回归与真实 NPU 验证建议顺序执行，避免不同进程同时初始化设备。
