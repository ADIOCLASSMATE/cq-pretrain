# DCVLM 完整记录 canary：实际完成结果与质量边界

截至 2026-10-22 18:05 UTC（北京时间 2026-10-23 02:05），**真实端到端 canary 已完成**。Qwen3.5-397B-A17B BF16 生成、Gemma4-31B-IT BF16 独立复核、Kimi-k3 告警复核、SSCD、精选 layout、发布及原始图片读回均有实际回执；不是 fixture 或仅 dry-run。最终可见集合为 **9 条记录、8 张图片、1 个独立 layout**。这是覆盖关键分支的小样本验收，不能用于估计全库准确率或通过率。

产物根目录：`runs/dcvlm_canary_v2_20261022/`。最终 release 为 `49d1405c20a7d80efecd504eb9facbd26486ea51bf3765e0db923ab2cce27210`，`summary.json`、`COMPLETE.json`、`manual_adjudication.json`、`final_reader_acceptance.json` 彼此绑定校验值。

## 实际输入与阶段结果

输入清单 `canary_manifest_v2.json` 包含 19 条完整原生 Parquet 记录，19 个图片引用、19 个不同原始图像 SHA；MMDU 的两张关联图全部恢复，纯文本没有图片。原始任务为单轮 I2T 14 条、多轮 I2T 3 条、交错 1 条、纯文本 1 条。没有将 DCVLM 反转为 T2I，也没有从多图记录只取一张冒充完整处理。

| 阶段 | 实际结果 |
|---|---|
| 原始图像检查 | 19 图全部按 TAR byte range 读取、校验 SHA 并解码；18 图短边 ≥128，1 张 CASIA 短边 30 被拒绝，OCR 没有例外 |
| CF384＋UniFD | 18 图绑定真实分数；17 图严格核验冻结历史回执并复用，额外 MMDU 图片实际执行新推理 |
| AI 路由 | 17 个 no_model_alert、1 个 ai_suspect_review；这些分数不是来源真实性证明 |
| 付费复核 | pixmo_points 图实际调用 Kimi-k3 一次，返回 likely_non_ai／medium／needs_provenance=true；mixed 来源仍隔离，续跑没有再次付费调用 |
| SSCD | 官方权重、FP32 CPU、512 维；18 个合格资产、1 对近似候选、0 失败；本轮无精确重复 |
| 类型保持生成与独立 QC | 18 条记录有真实 Qwen/Gemma 结果，另 1 条尺寸不合格不生成；未知来源 audit-only 生成不等于准入 |
| layout | 最终执行结果包含 5 个真实生成候选，1 个通过独立视觉 QC；其余不发布，原生任务独立判断 |
| 发布及读回 | 最终 9 条记录／8 图／1 layout；逐条恢复原始 Parquet、核对原图 SHA，所有原生方向保持 |

第二轮在已有成功生成缓存上用时约 96 秒：18 个最终 annotation 中 14 个复用相同请求／模型／预处理身份的真实成功结果，4 个为新生成；18 个最终质量复核均为更新 delta 协议后的新请求。这不是冷启动吞吐基准。28 项单元／接口测试另行通过，不计入真实模型推理数量。

## 实际遇到的问题与修复

第一轮两条 VisDial 请求把 `segment_index` 当作文本回合序号，忽略 image segment，严格验证器拒绝了指向 user 段的修订。现将原始零基索引和可修改 assistant 索引显式传入，多轮和多图顺序不变。另有 Gemma 把空 `assistant_revisions` 当成无答案；现明确 annotation 是原记录的 delta，复核上下文列出实际保留或修改后的 assistant 目标。空修订仍须检查原答案是否正确，不自动通过。

Qwen 服务验收确认需要 `mm_processor_kwargs={min_pixels:1048576,max_pixels:1048576}`。正式客户端真实传递并将其纳入缓存身份；这只影响模型处理器，尺寸门始终检查未放大的原图。客户端保留原始可用 PNG/JPEG，必要时做 EXIF、白底透明合成与完整图缩放，不裁剪。每次请求检查动态 endpoint 的有效 lease、已验收 boot、模型、权重、processor、runtime 与 precision；同请求并发 single-flight，不把失败写成成功缓存。

初轮双模型通过的夜景描述仍存在“远处尖塔是教堂”和时间表述矛盾。实际保守重生成去掉了教堂身份和 medium 的确定时间，但 Qwen 仍保留明确 concerns，因此整条隔离，未清空 concerns 强行发布。另一条 VisDial 原问题询问 race；可见肤色不能验证该身份属性，原标签和两模型原回执均保留，发布决定持久改为隔离，不再调用模型证明它。

这些外部审阅记录在 `quality_review_requests.json`、`admission_overrides.json`。后者仅允许 quarantine，不允许把未知或已拒绝来源改成白名单。`review.py` 离线重建了本 run 的无 parent 候选 release，没有新增模型请求；旧候选目录和回执不可变保留，但 current 已切换。实际 reader 确认 `1529c981…` 与 `01673f758…` 均不可见，且旧候选不是最终 release 的祖先。

## 最终可见数据

| 来源 | 最终记录数 | 保留任务 |
|---|---:|---|
| coco_caption | 1 | 单轮 caption I2T；原 terse 指令只用 short，medium/long 为 sidecar |
| visdial | 1 | 原生完整多轮 I2T |
| plotqa | 1 | 图表 QA；保留来源精确数值标签，只声称像素近似一致 |
| synthdog | 2 | OCR QA，保留原语言；文本规范化需区别于凭空补字 |
| clevr_math | 1 | 渲染场景算术 QA |
| websight | 2 | 截图到代码的原生 I2T；其中 1 个额外结构 layout |
| dolly | 1 | 纯文本 |

最终为单轮 I2T 7 条、多轮 I2T 1 条、纯文本 1 条。交错、多图 MMDU、两张棋盘图和 pixmo_points 仍因 unknown/mixed 来源或未通过质量门隔离；保留审计输出不表示它们已获训练准入。不要将 9/19 当作正式语料的预计通过率，因为这里故意选择了难例与控制例。

## 独立看图结论

- SSCD 相似度约 0.954 的两张棋盘图，实际分别 8 列与 6 列，缺格位置也不同。Gemma 真实 pair 复核与独立看图均支持保留不同资产，未按高相似度误合并。
- Websight 的 11 元素 layout 与原始 2560×1440 截图的标题、段落、地址和 2022 footer 相符，阅读顺序支持；框是近似区域，不是像素级 OCR 真值。这只能证明一个结构 layout 案例通过小样本复核。
- CLEVR 已查看案例的小紫色哑光方块数量与“加 2 后为 3”一致。另一条模型间判断不一致的渲染 QA 留在隔离队列，没有为了保留渲染数据而豁免质量门。
- PlotQA 保留原答案 0.8247；图像支持约 0.82 的一致性，不能宣称由图像独立读出四位小数。原始 source 标签精度与视觉复核精度分开记录。

相关独立观察：`manual_pair_review_20261023.json`、`manual_native_review_20261023.json`、`manual_layout_review_20261023.json`。这些是 assistant 视觉复查，不称人工真值；双模型通过也不保证每个细节正确。

## 剩余边界

当前完整载入器仅覆盖 DCVLM 固定 canary；gen-und 和未来 DCVLM-balance-200B 仍需真实挂载后逐来源验证 profile、格式和 schema。进程内有界并发、阶段缓存与增量引用发布已经可用，但不是数亿记录的分布式调度器。

精确同图资产共享与不同监督保留已有实现；**近似 pair 目前只生成审计建议，尚未将近重 family／曝光簇应用到训练权重**，`sampling_cluster_applied_to_training=false`。本次 offline review 仅替换本 run 无 parent 的未验收候选，不能代替已接受历史语料的通用撤销／tombstone。检测阈值、全域质量、实际训练收益仍需独立校准和后续对照。
