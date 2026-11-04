# 第一阶段当前验证：DCVLM 原生类型与 gen-und T2I

更新：2026-10-22（北京时间）。规则版本 `2026-10-22.stage1.3`。

## 当前来源与任务约定

按用户最新要求，T2I50 派生版已删除。DCVLM 默认入口恢复为：

```text
/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z
```

DCVLM 保留原生 I2T、对话、交错与纯文本类型和顺序，不再反转 caption 构造 T2I。T2I 训练候选仅从 gen-und 构建，gen-und 的合格 caption pair 可提供 T2I 视图，原生多轮 T2I 也可保留。索引中的原生分类证据与训练视图分开管理；排除／隔离审计导出不因为该训练来源限制遗漏记录。

AI 来源规则未改变：生成图排除；AI 编辑目标若有明确真实原图来源，保留并标注；未知编辑来源隔离；非 AI 程序绘图／渲染不因非自然属性排除；AI caption 保留并标注。具体证据要求见 `PROVENANCE.md`。

## 数据退役与保留核验

仅删除 `dcvlm_pretrain_t2i50_20261017T093600Z` 整个目录，包含 58,475 个普通文件，合计 **106,223,678,972 bytes（约 106.2 GB，逻辑大小）**。这不是共享存储配额即时回收量的测量。

删除前逐一核对 12,205 对父子发布 Parquet：两边独立，无共享 inode，大小符合各自 catalog。子版 `runtime`、`dedup` 两个符号链接指向父版，删除时只解除链接，不沿链接删除父版文件。父版 catalog SHA256 删除前后均为：

```text
e36afd5d89c3539d5d8e14b8b7d0ea76e296ec0dee9b37e029f0e1c5fb4cbb59
```

父版的 119,010,208 条文档、12,205 个发布分片为既有发布统计；本次校验 catalog、文件存在性和大小，不声称重新读取全部行。12,206 个原始 TAR 已逐项确认存在且大小符合 sources；原始图片池、annotations、父版 runtime/dedup、训练 checkpoint 和实验源码快照均保留。

平台查询当时无活动训练 Job，但两条旧实验的自动重启绑定仍指向 T2I50。为防止读取已删除输入，已通过 Inspire 控制器的正常关闭入口停用这两条绑定，最终状态均为 `complete`。没有将旧 checkpoint 的数据游标改成父版游标，也没有启动替代训练。相关记录：

- `validation/t2i50_retirement_20261022/deletion.json`：目标、删除前统计、删除后校验。
- `validation/t2i50_retirement_20261022/controllers.json`：两条绑定的关闭回执。
- `validation/t2i50_retirement_20261022/retired_metadata/`：少量旧发布清单和策略证据，不包含训练数据。

当前 `train_production.py`、`ascend_pretrain/configs/recommended.json`、管道默认路径及两工程 `INSPIRE.md` 已更新；data-syn 的旧 `active_run.json` 已撤销，并保留退役记录。历史 `.runs` 快照和既有审计报告中的旧路径保留真实历史身份。

## 原版 DCVLM 实际抽样重跑

命令：

```bash
.venv/bin/python -m corpus_pipeline run \
  --collection dcvlm --output corpus_pipeline/runs/dcvlm_native_stage1_3 \
  --mode sample --sample-units 2 --sample-rows 16
```

覆盖 **165 个来源、238 个工作单元、3,808 条记录**，解析失败为 0。每来源最多两个分片，每分片前 16 条；这是诊断抽样，不代表全库分布。

| 原生类别 | 保留 | 排除 | 待核实 |
|---|---:|---:|---:|
| 单轮 I2T | 519 | 0 | 1,950 |
| 多轮 I2T | 105 | 0 | 191 |
| 交错图文 | 0 | 0 | 288 |
| 纯文本 | 755 | 0 | 0 |
| 合计 | 1,379 | 0 | 2,429 |

本次样本 **T2I 分类为 0，T2I 候选视图为 0**。上一轮 T2I50 抽样中的 89 条 T2I 在父版恢复为 I2T；旧的 32 条 DCVLM caption 反向 T2I 候选也不再提供。来源筛选的保留／排除／待核实总数未改变。

实际导出 1,379 条保留记录，均选择其原生类别。显式尝试从该 DCVLM run 导出 `single_t2i` 已被拒绝，且未留下输出文件。所有 238 个索引分片的 SHA256、行数与运行完成状态已核对，运行快照核心代码 SHA256 与最终代码一致。

## gen-und 范围

本轮没有启动 CPU 或重新读取 gen-und 原始数据。此前的 21 来源、322 条历史样本政策复核仍为 **0 保留、48 排除、274 待核实**，详见 `validation/stage1_2/`；这些不是 stage1.3 的新原始数据运行。gen-und 缺少图文关联的散图诊断样本不能推断整源没有 caption。

T2I 来源约束、caption 反向视图和多轮 T2I 的实现由构造测试覆盖；这不表示 gen-und 全量 T2I 已完成建库。CPU 与昇腾存储仍不互通。

## 测试与产物

**84 项自动化测试通过，无跳过**；测试命令和日志：

```bash
.venv/bin/python -m unittest discover -s corpus_pipeline -p 'test_*.py' -q
# 日志：corpus_pipeline/runs/tests_stage1_3.log
```

新增测试覆盖 DCVLM 原生方向、gen-und 的 T2I 视图、历史索引不能绕过来源限制、原生多轮／交错保持、非训练审计导出完整性。此前图像来源、编辑原图证据冲突、容器适配、恢复与原子发布测试继续通过。

- `runs/dcvlm_native_stage1_3/`：当前原版 DCVLM 样本索引与快照。
- `runs/exports/dcvlm_native_stage1_3_accepted.jsonl.gz`：1,379 条原生任务候选索引。
- `validation/dcvlm_native_stage1_3_checks.json`：实际分片、视图和导出核验。
- `validation/source_sample_counts.csv`：当前 DCVLM 与历史 gen-und 逐来源计数，显式区分验证方式和版本。
- `validation/summary.json`：当前机器可读验证范围。
- `validation/artifact_checksums.json`：当前代码、配置、文档与验证产物校验和。
- `validation/stage1_1/`、`validation/stage1_2/`：历史政策与 T2I50 抽样报告，已存档；旧 run 和旧导出不能当作当前训练入口。

## 限制

本阶段尚未全量筛选、未部署像素级 AI 检测器，也未完成近重复去重、caption 忠实度检查、解码质量筛选或完整评测去污染。来源规则通过不等于逐像素鉴真。后续检测器调研见 `research/AI_DETECTOR_SURVEY_20261022.md`，其中建议尚未接入当前筛选决策。
