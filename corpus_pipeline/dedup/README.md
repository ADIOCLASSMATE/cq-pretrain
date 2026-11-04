> 后续 DCVLM-balanced-200b＋gen-und 联合处理统一使用 [新管道入口](../README.md)。下文为已经完成且保持不可变的 DCVLM 全量结果。

# DCVLM 全量 SSCD 与训练候选索引

本目录包含正式亿级处理的资产策略、图候选和训练记录发布模块，也保留早期 `sscd.py` 小批量 ANN 实验。两者的运行规模和完成凭据不同。2026-10-26 的正式结果已经完成全量相似度候选计算、资产筛选映射、119,010,208 条记录的保留／排除／隔离索引、原文件顺序读取视图，以及确认占位图的独立质量覆盖层。**当前最终训练入口是 `quality_overlay/source_view`**，基础索引和原始数据保持不变。

## 正式结果及边界

| 层级 | 已完成结果 |
| --- | --- |
| 原始记录 | 24,235 个 committed parts，119,010,208 条记录，12,205 个原 metadata Parquet |
| 资产身份 | 99,930,890 个编码资产，99,867,014 个精确视觉身份；63,876 个不同编码资产共享已确认像素身份 |
| SSCD 全量计算 | 90,535,667 个可用向量，4,098,353,454,299,611 个无序图像对全部覆盖，755,773,464 条候选边 |
| 最终训练候选 keep | 108,166,671 条，90.88857% |
| 最终明确 exclude | 10,794,086 条，9.06988% |
| 完整性 quarantine | 49,451 条，0.04155% |
| 原文件读取视图 | 12,205 个 uint8 mask，合计 119,010,208 字节；全部原行覆盖，未索引行数为 0 |

`keep` 是当前阶段的训练候选资格。它不代表每张图都完成严格模糊复核或完整质量认证；产物明确写入 `published_training=false`、`source_certification=false`。其中 31,054 条最终保留记录仍带未决筛选标注。现有严格模糊复核覆盖的是上游明确绑定的 688 张审核图，不能扩张为整个语料已完成该检查。

最终排除记录由 10,618,694 条既有 frontpass 拒绝记录、132,704 条明确图像筛选排除记录和 42,688 条新增确认占位图记录组成。本次严格的相同监督签名没有产生额外重复记录排除，不能把 SSCD 的候选边数或候选代表数量当成已删除记录数。所有原始文件均未删除或改写。

最终保留记录按原类别为：`single_i2t` 82,211,153；`pure_text` 22,096,237；`interleaved` 3,735,950；`multi_i2t` 123,330；`unclassified` 1。最后这 1 条保留待后续标注，不以类别不明为由自动删除。DCVLM 原任务方向保持不变，没有把 I2T 转成 T2I。

## 正式模块与顺序

1. [`../postprocess/sscd_exact_npu.py`](../postprocess/sscd_exact_npu.py)：对正式 FP32 向量做 NPU 分块全对计算、保守阈值筛选和 CPU 高精度复核。完成凭据绑定校准、计划和全部任务回执。当前 `.75` 是候选阈值，不能直接用作删除阈值。
2. [`decisions.py`](decisions.py)：绑定既有 Qwen 与最新人工／质量覆盖策略，输出原资产的明确排除、未决或带标注保留结果。模型疑似 AI 不自动变成确认排除。
3. [`asset_release.py`](asset_release.py)：合并全量资产状态和筛选策略；只有确认的文件／规范化像素身份才可共享 canonical identity。AI／质量决策始终关联原 `asset_id`，不会通过 canonical identity 传播到另一编码资产。
4. [`graph_policy.py`](graph_policy.py) 和 [`sscd_graph.py`](sscd_graph.py)：保留 SSCD 候选图与直接边代表关系用于审计。连通分量不是传递合并授权，相似图表中的不同数字、不同题目、裁剪或编辑内容仍可能是有效的不同监督。
5. [`training_release.py`](training_release.py)：消费 committed records 和最终资产映射，冻结原 metadata 文件 SHA，生成 `keep`／`exclude`／`quarantine` 记录指针。不读取图像字节，不重写文本。
6. [`source_view.py`](source_view.py)：将上述结果按原文件／原行号排序，生成每个原文件的 selection mask，提供高吞吐的顺序训练读取入口。
7. [`selection_overlay.py`](selection_overlay.py)：对已完成基础视图施加独立、可审计的质量排除。当前仅匹配原 `image_ids` 中的 exact encoded SHA `2637f446bc6640220c9b726c624f2156836bb7a67b754c098f7fda5f126c7fcc`。原 TAR 实图证实它是商品缺图占位徽标；对应商品 caption 与画面不符。命中记录 42,738 条，其中 42,688 条由 keep 改为 exclude，既有 41 条 exclude 和 9 条 quarantine 保持不变。该规则不传播到 canonical 资产或 SSCD 相似图；多图记录只要含此 exact 资产就整条排除。

训练记录只有同时满足相同 `category`、原 `native_format`、角色／轮次／图像位置感知的 `text_hash`、相同顺序的精确 canonical 图像身份，才被视为相同监督。相同图像的不同 caption、QA、多轮或交错记录继续保留。代表优先选择可保留且完整性正常的记录；重复指针必须指向同签名的 keep 记录。缺失资产或未解决完整性问题进入 quarantine，不能直接喂给训练。

## 真实路径与可重启入口

在仓库根目录执行；以下是本次完成产物的实际路径：

```bash
DCVLM_FRONT="/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z"
DCVLM_RELEASE="$DCVLM_FRONT/postprocess_20261024/sscd_release_20261026"
DCVLM_EXACT="$DCVLM_FRONT/postprocess_20261024/sscd_parallel_20261025/exact_npu_20261025/run"

# 完成凭据：
# $DCVLM_EXACT/COMPLETE.json
# $DCVLM_RELEASE/assets/COMPLETE.json
# $DCVLM_RELEASE/graph/COMPLETE.json
# $DCVLM_RELEASE/training_candidate_index/COMPLETE.json
# $DCVLM_RELEASE/training_candidate_source_view/COMPLETE.json
# $DCVLM_RELEASE/quality_overlay/COMPLETE.json
# $DCVLM_RELEASE/quality_overlay/source_view/COMPLETE.json

.venv/bin/python -m corpus_pipeline.dedup.training_release \
  --run-root "$DCVLM_FRONT" \
  --asset-manifest "$DCVLM_RELEASE/assets/COMPLETE.json" \
  --output "$DCVLM_RELEASE/training_candidate_index" \
  --threads 3 --memory-limit 128GB --batch-parts 128

.venv/bin/python -m corpus_pipeline.dedup.source_view \
  --candidate "$DCVLM_RELEASE/training_candidate_index" \
  --output "$DCVLM_RELEASE/training_candidate_source_view" \
  --threads 2 --memory-limit 32GB

DCVLM_PLACEHOLDER_ASSET="2637f446bc6640220c9b726c624f2156836bb7a67b754c098f7fda5f126c7fcc"
.venv/bin/python -m corpus_pipeline.dedup.selection_overlay \
  --candidate "$DCVLM_RELEASE/training_candidate_index" \
  --source-view "$DCVLM_RELEASE/training_candidate_source_view" \
  --output "$DCVLM_RELEASE/quality_overlay" \
  --asset-id "$DCVLM_PLACEHOLDER_ASSET" \
  --evidence "$DCVLM_RELEASE/audits/caption_alignment_spotcheck.json" \
  --evidence "$DCVLM_RELEASE/audits/caption_alignment_spotcheck/extraction_evidence.json" \
  --evidence "$DCVLM_RELEASE/audits/caption_alignment_spotcheck/$DCVLM_PLACEHOLDER_ASSET.jpg" \
  --threads 2 --memory-limit 32GB
```

已完成目录再次执行会校验绑定并复用结果；输入、策略或代码身份变化需要新的输出目录，不能覆盖既有发布产物。初次 6 线程／64GB 的全局列表聚合触及 DuckDB 内存限额；同目录以 3 线程／128GB 恢复后复用全部已准备记录，并完成正式索引。这是 SQL 内存限额失败，不是系统 OOM，恢复没有修改筛选语义。运行与恢复账本位于上述 release 目录的 `TRAINING_CANDIDATE_PROCESS.json`、`TRAINING_CANDIDATE_RECOVERY_01.json` 和 `SOURCE_VIEW_PROCESS.json`。

基础记录索引 manifest SHA256 为 `59992bd797d2271376466be2f5b2fe0e21b6c3cc72024ef8237b88bda253235b`，基础顺序视图为 `c944b714d6bd9c85d6769365ee3a297663790da3f7ca4bece3b65c7474ebbd64`。质量覆盖 manifest 为 `19fcc31261685020172d6270c9aa8c34b979f586169f75d5047d0bb9efd45976`；**最终顺序读取视图** manifest 为 `b973635ae49bc30fe4fd93f4f386ec7032fc2ab8723c9d599c84be7255559ef1`。原文件与 mask 的 SHA 绑定在相应 manifest 中。每个原行都有唯一 locator；重复 locator、覆盖不足或三类计数不一致都会使 source view 发布失败。

覆盖层保存全部命中与变更记录的 Parquet、资产规则、证据 SHA、基础输入 SHA、原／最终计数和按来源／类别的差值。仅 4,892 个原文件的 mask 发生改变，其余 mask 复用已验证内容。最终视图明确标记 `selection_overlay=true`，不会声称其决策计数仍等于基础索引。独立全量审计验证了全部 12,205 个基础／最终 mask 的 SHA 与逐行差异，差异集合恰好为 42,688 个原 keep 命中；审计凭据为 `audits/quality_overlay.json`，SHA256 `a0a43b4be091832dc7c95af417ec5400c3d7f72cfe569886362e3365f35fe48d`。

## 训练读取

全量训练使用 `iter_native_batches`，按原 Parquet 顺序读取并选出 keep 行。每个所读原文件校验一次完整 SHA，不会在按签名分桶的随机记录间来回重复加载同一原文件。返回原始 Arrow 字段，包括原任务格式、segments、图像槽位和图像位置元数据，图像字节仍交给现有原始数据解码器读取。

```python
from pathlib import Path
from corpus_pipeline.dedup.source_view import iter_native_batches

view = Path(
    "/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/"
    "corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z/"
    "postprocess_20261024/sscd_release_20261026/quality_overlay/source_view"
)
for native_batch in iter_native_batches(
    view, batch_size=8192, epoch_seed=0,
    worker_id=0, num_workers=1,
):
    # 将原始 metadata batch 交给现有训练数据解码器。
    pass
```

多 worker 必须使用相同 `epoch_seed` 和全局一致的 `num_workers`，各自使用不同 `worker_id`。该实现按文件打乱和分配，文件内保持原行顺序；筛选后每批大小可能小于 `batch_size`。mask 编码为 `0=exclude`、`1=keep`、`2=quarantine`、`255=not_indexed`，正式全量结果没有 255。

[`candidate_reader.py`](candidate_reader.py) 用于按基础记录定位和深入审计，逐条校验原记录及图槽，不作为全量顺序训练入口；基础记录的 keep 标签还须叠加质量覆盖才能代表当前选择。真实 423 条 DCVLM canary 已全部恢复并核对完整原记录；source view 的同一 canary 也已逐条对齐。完成后另对基础全量 view 做了 771 条／3 批有界启动验证，全部原 Arrow 字段一致。相关凭据为 release 目录中的 `training_candidate_real_canary/READER_CANARY_COMPLETE.json`、`training_candidate_real_canary/SOURCE_VIEW_CANARY_COMPLETE.json`、`SOURCE_VIEW_FULL_RELEASE_READER_CANARY.json`。最终覆盖视图的全量差异验证见上述独立审计。

## 历史小批量 ANN 验证

以下 `sscd.py` API 和 2026-10-22 验证只针对有界工作单元，不是上述正式全量结果的实现或完成证据。

实现 `process_assets()`，读取原图验证 SHA256、原始短边至少 128、单帧解码及规范化 RGBA 像素身份，运行官方 `sscd_disc_mixup` 并检索本批与历史资产。图片和原监督不会被删除。

```python
from corpus_pipeline.dedup.sscd import process_assets

result = process_assets(
    assets,  # [{asset_id, sha256, image_path OR blob_loader, width, height, domain, source_refs}]
    output_dir="new_work_unit_dir",  # 已存在则拒绝覆盖
    history_path="shared_feature_catalog.sqlite",
    batch_size=16,
    threads=8,
)
```

源图不得先放大来通过尺寸门。多帧图片需要独立的帧协议，当前返回明确失败。透明图像特征使用白底合成，像素身份仍保留完整 RGBA，防止不同透明度被误合并。

文件 SHA／规范化像素一致可以共享 canonical asset，同时保留所有来源观察和上层监督记录。SSCD 相似度只创建 `requires_pair_review` 边；默认 0.75 是候选召回起点，不是删除阈值，不代表 MOSS 公开了这个值。图表文字／数字、题目格子、裁剪与编辑差异必须复核。

历史 SQLite 是特征与身份缓存，**不是已通过所有筛选的训练库**。训练资格由来源、AI 筛选、任务质量及发布器另外判断。来源冲突不因共享资产而消失。生产管道的正式索引与研究样本索引应分开。

数据库使用 WAL、FULL 同步和事务锁；提交时更新内存索引中的新增条目，再检查本批，因此两个 worker 不会各自忽略另一批的已提交资产。异常回滚会重建 ANN 视图，避免幻影条目。权重／预处理／运行精度身份不一致会拒绝复用旧特征。

该历史 API 的检索使用 FAISS HNSW（512维、M32、efConstruction160、efSearch128、top20）。这是近似候选检索；精确字节／像素匹配不依赖 ANN。每个 `process_assets` 处理有界工作单元，内存索引按 SQLite 重建；不能将小试验吞吐和内存外推为亿级能力。上面的正式生产路径使用独立的分块计算、校准和完整覆盖回执。

早期实验记录（原 run 目录当前未保留，下面的小样本统计不作为正式全量完成凭据）：

- 官方 TorchScript 直连下载，约98.8MB，权重 SHA256 `9f26bd4c848cc19b73d2ae92eea6e04886f61a7b764ceb7a13aeee62e6a6db56`。
- 20资产实际 canary：精确副本正确共享；已知 JPEG 重压对相似度0.99525，送入候选复核。
- DCVLM 小样本实跑：2,213 个尺寸合格输入，2,208 单帧完成，5多帧明确失败；838候选边，无自动近重合并。8CPU线程约173秒，不能外推全部原始数据的吞吐或损坏率。
- [ANN 检查](ann_validation_20261022.json)：本批随机128查询对比精确top20，召回率99.73%；其中阈值以上的149个top20邻居均被召回。只是当前样本的局部验证。
- [实际看图记录](visual_pair_review_20261022.json)：0.954的两张棋盘题内容不同，0.801的两张统计图主题／数字不同，均不可合并。

运行边界测试：

```bash
.venv/bin/python -m unittest corpus_pipeline.dedup.test_sscd -v
```

CPU环境若安装了 `torch_npu` 却没有 NPU 驱动，编码器在 import torch 前关闭自动设备后端加载；不因此修改 NPU 部署容器。依赖为当前 `.venv` 的 CPU PyTorch、NumPy、Pillow 和 `faiss-cpu==1.15.1`。

官方权重与预处理依据：[SSCD 原作者仓库](https://github.com/facebookresearch/sscd-copy-detection)。
