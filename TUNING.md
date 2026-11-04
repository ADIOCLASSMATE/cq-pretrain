# Q=64 吞吐调优记录（2026-10-18）

全部为 64 × Ascend 910B2C、真实 T2I50 数据、官方预训练权重、真实冻结 FP32 VAE。
`docs/s` 为计时区间内实际完成的文档数，非容量槽位。证据目录
`/inspire/sj-ssd3/project/high-dimensionaldata/public/training/cq_speed_20261017/`。

## 起点与结论

| 配置 | mean s/更新 | docs/s | 峰值 GiB |
|---|---:|---:|---:|
| Q=256 TP2 全重计算（旧） | 49.8 | 81.8 | 56.73 |
| **Q=64 TP2 全重计算（选定）** | **36.1** | **113.1** | **52.22** |
| Q=64 TP4 全重计算 | 42.8 | 95.1 | 31.52 |

20 步确认复测 36.4 s / 109.8 docs/s；独立 sweep 的参考档复现 36.0 s / 113.6 docs/s，
说明测量可复现。128 卡实测约 200 docs/s。

## sweep01：TP × 重计算 2×3 网格

TP 与重计算通过显存相互作用，因此做完整网格而非"先选 TP 再调重计算"的贪心链。

| stage | 结果 |
|---|---|
| tp2_full | **36.1 s / 113.1 docs/s / 52.22 GiB** |
| tp4_full | 42.8 s / 95.1 docs/s / 31.52 GiB |
| tp2_none / tp4_none | **真实 OOM**（约 57.9 GiB / 容量 60.96） |
| tp2_sel_dit21 / tp4_sel_dit21 | 真实 OOM |
| best_mc2_off | 37.3 s / 109.4 docs/s —— **关闭 MC2 慢 3%** |

## sweep02：重计算深度梯度

| stage | 结果 |
|---|---|
| tp2_full_ref | 36.0 s / 113.6 docs/s / 52.22 GiB |
| bb32_dit42（等价全开） | 36.2 s / 113.0 docs/s / 52.22 GiB |
| bb32_dit36 | 峰值 54.96 GiB → OOM |
| bb32_dit30 / dit24 | OOM |
| bb28_dit42 | 峰值 56.38 GiB → OOM |
| bb24_dit42 | OOM |

**结论：8192 token 下全量重计算已是最优解，不是可调项。** 42 层 DiT 持有绝大部分
激活；任何减少重计算的尝试都越过 60.96 GiB 的真实容量。

## sweep03：profile 与其余旋钮

| stage | docs/s | 相对基线 |
|---|---:|---|
| 基线 tp2_full | 113.1 | — |
| prefetch2（device_prefetch 2 / vae_chunk 2） | 114.1 | 略优，已采用 |
| pad8192 | 112.6 | 噪声内 |
| ce_chunk4096 | 112.6 | 噪声内，且多耗 3 GiB |
| bucket80 | 112.3 | 噪声内 |
| situ_all（affine/gated_add 也融合） | 109.6 | **更慢**，不采用 |

### Q=64 下的 profile（rank0，每步）

整步 wall 35.27 s：Computing 27.87 s（79.0%）、未被覆盖的通信 7.06 s（20.0%）、
Free 0.34 s（1.0%）。算子时长总和 52.22 s/步 = wall 的 1.5 倍，说明重叠充分，
**算子时长总和不能当作 wall time 占比**。

`MatmulAllReduceMc2AicpuKernel` 占算子时长总和的 44.55%（23.26 s/步），但若它真的
串行就会超过 Computing 总量——它是与计算并发的控制任务，不在关键路径上。这与
"关闭 MC2 反而慢 3%"的实测一致。

### 不是瓶颈的部分

dataloader 等待 **0.03 s/步**，设备准备等待 0.18 s/步。注意力
（FlashAttentionScore + Grad）合计仅占算子时长的 2.8%。继续优化 dataloader 或
注意力都没有意义；当前是 **GEMM / 激活显存** 双重约束。

## 删除的历史逻辑

动态 work-budget 组批（按预测计算量决定每次更新的 pack 数，会因 padding 未填满目标
token 数而增大 GA）已从 `train.py`、`input_pipeline/device.py`、`tune64.py` 与测试中
整体删除。训练合同是固定 global batch + 固定 GA。

同时删除 `train_recommended.py` 与 `train_resumable.py`（均为 Q=256 旧配方入口），
由 `train_production.py` 取代。

## 基础设施订正

- 提交 Job 必须显式 `--shm-size 128`；缺省 0 GiB 会让 HCCL 首个 AllReduce 直接超时。
- `tune64.py` 的 55.5 GiB 显存上限原本硬编码，已改为读取 plan 的 `memory_guard_gib`；
  卡的真实可用容量是 **60.96 GiB**，不是 64。
- 正式训练 `HCCL_EXEC_TIMEOUT=1800`（launch.sh 默认 300，对可被抢占的低优作业过紧）。
