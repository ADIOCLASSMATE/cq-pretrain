# 当前全流程操作

使用仓库根目录的 `.venv/bin/python -m corpus_pipeline`。`joint` 是底层阶段工具，不负责六子集 CPU 生命周期、全局卡数协调与 baseline 合并。

## 输入准备

从 [新任务模板](../orchestration/examples/genund.json) 填写本次独立运行目录，替换所有 `REPLACE`。本 profile 绑定账号 `wjx-ascend`、昇腾公共 Workspace、现有项目与 910B 资源组，不接受把 CPU 与审查分配到不同项目。

必需输入：

- 六个逻辑子集及 DenseFusion-4V-100k 补充源的完整 `PLAN.json`。OpenUni、ScaleCap、FineVision 使用 `production/genund_collections.py`；BLIP、DenseFusion、PixMo 使用 `unified/genund_loose.py`。缺失标注或恢复图片须先取得精确字节绑定，不能按文件名猜对应关系。
- 已封存的 baseline selected-reference index 和 reconciliation，不能用未完成中间目录替代。
- 已验收的 detector/SSCD 资格文件、模型资产目录、私密 API key 文件。CPU 数据依赖见 `corpus_pipeline/requirements.txt`；NPU 使用已具备匹配 torch_npu/CANN 的环境，管理端用已安装 Inspire SDK 的独立 Python。
- 已创建且属于本项目的 CPU Notebook 名称、资源 ID、workspace ID 和 `ingest`/`review` 角色。唯一 review CPU 在归一化完成后继续保留。模板中的 CPU 优先级应与已创建资源一致。
- `dataset_roots`：计划中的固定 dataset ID 到实际挂载根目录的映射。数据原件在运行目录之外；控制端与 CPU/NPU 的共享代码、队列和输出路径必须相通。远端 worker 还会检查真实挂载与内容绑定。

准备与注册：

```bash
.venv/bin/python -m corpus_pipeline prepare --config /ABS/request.json
.venv/bin/python -m corpus_pipeline register-sources --config /ABS/run/ops/prepared_config.json
.venv/bin/python -m corpus_pipeline preflight --config /ABS/run/ops/prepared_config.json
```

准备生成冻结 `runtime/RUNTIME.json`、统一启动器和单卡/TP2 模板。重新准备必须与原绑定一致，修改代码或输入需新建运行目录。`preflight` 不执行真实模型推理；运行时仍须通过模型数值资格、图片字节校验与同进程真实审查预检。

## 执行与观察

在持久控制终端执行：

```bash
.venv/bin/python -m corpus_pipeline run --config /ABS/run/ops/prepared_config.json
```

入口校验后自动切换到该运行的冻结代码。CPU 归一化与已有完整来源的后续处理可以并行；一个来源的 NPU 阶段拥有全局预算，释放确认后才交给下一阶段。所有新前置实例使用真实 `1,8,64` 规格，不做整节点集中迁移。

```bash
.venv/bin/python -m corpus_pipeline status --config /ABS/run/ops/prepared_config.json
```

重点检查 `actual_review_responses` 是否增加、最后更新时间、CPU 退休状态和阶段回执。审查的计时、HTTP 尝试、失败及真实返回保存在各源 `api_review/PROGRESS.json`。审查队列等待无卡不算 HTTP 故障；持续 120 秒且至少 64 次尝试无有效返回会中断并保留可恢复账本。

运行服务与排队服务分开轮询，已绑定服务按 ID 查询；预算按已申请卡数计算，未确认 Create 保留占用。共享文件系统锁同时使用进程锁和记录锁，禁止删除锁文件或账本来恢复预算。队列 SQLite 只由拥有它的进程访问；远端 worker 使用 RPC。

## 停止与恢复

```bash
.venv/bin/python -m corpus_pipeline stop --config /ABS/run/ops/prepared_config.json
```

该命令发出停止请求，控制器等待正在执行的操作结束并确认平台释放。返回 `stop_requested` 不等于已经释放资源。普通进程崩溃可使用同一冻结配置重新运行；显式 STOP 不自动清除。不要启动第二个 owner，也不要用新代码覆盖旧 runtime。

归一化结束后，纯 ingest CPU 会持久退休；后续审查 CPU 在最终合并完成后释放。完成条件是运行根目录 `COMPLETE.json` 绑定正确的 `joint_release/COMPLETE.json`，且发布验证通过。模型输出不确定与网络/平台失败有不同语义，后者不得作为丢弃样本或结束流水线的理由。

新联合发行使用 `unified.baseline_union.ReferenceReader`，先 `iter_references()` 再 `read_record()`，须保留返回的图像 loss 权限。训练 Catalog 的格式接入是单独步骤，不能因数据发行完成就认定现有训练已消费它。

控制 Notebook 若重建为新 host，旧 owner 绑定不会自动放行。先核验相同平台资源 ID、旧心跳是否早于新实例 uptime、控制锁和已有资源退休账本，再保存旧绑定并显式接管；不要删除账本或仅按旧 PID 不存在就重建资源池。
