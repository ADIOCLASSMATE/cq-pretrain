# 分阶段 recaption 接入审计（只读）

审计日期：2026-10-23；对象为当前工作区 `corpus_pipeline/production/`、prepared profile 与已有 gen-und 来源契约。没有调用 API、修改生产代码或改变任何入库结论。本文给离线 planner 的接入边界；不是已经接通生产的能力声明。

## 当前行为与用户目标之间的差距

用户提出的“前期混合原监督与已有合格 recaption，只补必要；到 1M 像素训练阶段完整 recaption”可以节省大量重复生成，但当前 production 尚未实现这一调度方式。

1. **所有通过 annotation eligibility 的记录都会走 generator + verifier。** `production/run.py:239–275` 先检查来源、尺寸、图像完整性、去重与人工隔离，再调用 `annotate_record`。`run.py:66–85` 无条件提出 typed annotation、独立 QC，并可能额外修订一轮。硬拒绝或非 audit-only 的未决记录不会走这一分支。因此准确表述是“每条符合该分支准入条件的记录都进入生成/QC 请求路径”，不是“整个输入集合全部实际重新调用模型”。成功缓存可能避免新请求。
2. **纯文本也会调用两个模型。** 没有图片时只是 `bound=[]`，并未绕过 `annotate_record`。`validation.py:68–69` 只禁止虚构视觉内容。prepared 的 `profiles.py:98` 所说 `skip_visual_recaption` 不能解释为已经跳过所有 LLM。
3. **没有训练 stage、按需 caption planning 或候选复用入口。** `run.py:350–372` 的 CLI 和 `model_context`（41–53）均没有这些字段。prepared 的 `_contracts`（`prepared/profiles.py:74–109`）是分类、任务和 caption 行为契约；production 实际执行还没有消费一份逐记录 recaption 需求计划。
4. **现有 reader 不会混合原 caption 与生成 caption。** `publish.py:33–55` 对 caption view 直接替换目标文本；25%/50%/25% 是短、中、长之间的权重，不是原文与 recaption 之间的比例。原始记录仍可通过 locator 找回，但不会被这条 caption view 的 reader 当作一个可采样候选。
5. **“无需生成”不能伪装成一次成功的空输出。** `validation.py:49–50` 要求 caption 记录覆盖所有图片，每图三个非空长度字段（42–48）。当前 `commit_release` 又依赖 annotation/quality 的 cache key、identity，并统一写 `ai_generated_text=True`（`publish.py:114–141`）。必须显式表示 retained original / imported qualified caption，不能填假的 generator/QC 回执，也不能将一条原 caption 复制三次冒充三档。

## 任务语义的现有保护及不足

| 任务 | 当前代码依据 | 离线 planner 的允许动作 |
|---|---|---|
| 真正 caption、已有合法 T2I prompt | `validation.py:15–25`、`prompts.py:4–5`；DCVLM 仍禁止 T2I 转换 | 原 caption 保留为独立候选；已有 recaption 满足质量证据时复用；只补该阶段所缺的有效候选。gen-und T2I 还必须满足其已验证配对与方向契约。 |
| QA、OCR、数学/图表答案 | `prompts.py:6,17–19`、`run.py:56–63` 保留源标签/上下文的规则 | 原问题与答案保留；只有具体问题需要核验或修订时进入任务专属 QC。图片描述是附加 sidecar，不能覆盖 OCR 转录、数值答案或 QA。图像只能近似验证时不可宣称复验了源标签的精确小数。 |
| 多轮、多图 | `semantic_contract` 保持 dialogue；`model_context` 显式 segment indices；`records.py:86–92` 校验所有图像绑定 | 以完整记录为语义单元，保留全部角色、轮次、顺序和图像；不能将一轮的缓存结论复用到不同对话。仅确有必要才整段复核或修订。 |
| 交错序列 | `prompts.py:7` 规定正文/顺序保留；caption 可为 sidecar | 原正文与交错顺序直接保留；图片级描述可按需补充，不把它们插入或覆盖原文，不凭文件名重建网页。 |
| 纯文本 | `semantic_contract` 固定 pure_text；当前仍双模型 | 视觉 recaption 计划应明确 `not_applicable`；原监督保留。独立文本质量流程若需要调用模型，另计预算与回执，不能归入视觉 recaption。 |

需要注意两个实际边界：

- `records.py:25–39` 仅为 `coco_caption` 的两种**精确已观察指令**加 caption contract；其他视觉 conversation 默认 QA（`validation.py:23–25`）。不能只见 source 名称含 caption 或 assistant 文本很长就升级为 caption。固定来源新增映射仍需 adapter/精确结构证据。
- `annotation_validator` 当前对任意任务都允许修改合法 assistant text segment（`validation.py:51–58`）；因此 interleaved/pure_text 的“原文完全不变”更多依赖 prompt，而非专门的类型断言。planner 的 preserve-original 分支应直接使用未修改的 source record，未来接线再增加明确的任务级 immutable segment 检查。

## 推荐的最小接入点

### 1. 先产出离线计划，保持生产不自动读取

位置：完整原记录与 source profile 已绑定后、实际 `annotate_record` 之前。现入口是 `run.py:159` 的 `load_record` 和 `run.py:268` 的 annotation 分支。planner 可以预先决定工作需求，但其输出不能绕过后续图像/来源/人工隔离 gate。

建议每项计划至少绑定：

```json
{
  "plan_schema": "staged_caption_plan.v1",
  "training_stage": "early_mix | highres_1m",
  "policy_version": "explicit-frozen-version",
  "collection": "dcvlm | genund",
  "source_snapshot_sha256": "...",
  "source_record_sha256": "...",
  "original_sample_id": "...",
  "ordered_asset_sha256": ["..."],
  "native_task": "...",
  "semantic_task": "caption | qa | dialogue | interleaved | pure_text",
  "caption_contract_sha256": "...",
  "action": "preserve_original | reuse_qualified | fill_missing | regenerate_highres | defer_review",
  "target_segment_indices": [],
  "required_caption_views": [],
  "reusable_candidate_ids": [],
  "missing_candidate_specs": [],
  "eligibility_status": "pending_runtime_gates",
  "reason_codes": []
}
```

这些是建议字段，不是现有 runtime 接口。训练 stage、caption 候选需求、保留原任务三个概念应分开；`highres_1m` 不能把 QA/OCR/对话/纯文本统一改造成描述图片的任务。

### 2. 候选目录和“合格”证据独立于 response cache

来源本身已带 caption 字段不代表质量通过；原 caption、来源模型生成 caption、我们新生成 caption、未知文本来源必须分别标注。gen-und 已有契约明确先恢复/复用现有 recaption（`prepared/config/genund_source_contracts.json:3242–3252`），但 local join 和来源/模型身份仍可能 pending（3235–3239）。`config/adapters.json:108–135` 的 megalith 规则仍 disabled：通用 `caption` 字段可能只是 URL，六个 recaption 字段不能算成六张图。

建议候选目录记录：精确图像/有序图片集合、原记录/目标 segment、语言/指令约束、候选文本 hash、文本 origin、生成模型/版本/预处理、输入像素与裁剪状态、质量回执 hash/状态/适用范围、未决 concerns、人工隔离/撤销版本。没有质量证据的已有文本只是 `existing_unverified`，不能被 `reuse_qualified` 自动选中。

前期不要求所有图都有三档完整描述。保留已经合格的原短 caption 是有效路径；该阶段没有训练需求的 medium/long 不应该为了补齐当前 schema 就额外生成。后期完整覆盖的计数应是**合格图片及目标视图的覆盖率**，不能用重复图片或不同 caption 数量冒充。

### 3. 调度结果与模型回执分离

位置：`run.py:66–113` 和 `publish.py:114–150`。

- preserve-original：保留原 segments，生成一个真实的 deterministic retention receipt；不构造模型生成记录、不填 `real_model_inference=True`。
- reuse-qualified：引用原候选及原质量证据，校验当前适用性；是否补 QC 由变更/质量策略决定，不因“复用”默认重跑 generator。
- fill-missing：只生成所需候选；现 `annotation_validator` 要求全部短中长，因此应新增有版本的 candidate/coverage schema，而非悄悄放宽旧 schema。
- regenerate-highres：依据高分辨率计划执行描述任务；不改变 native QA/OCR/多轮语义。
- defer-review：保留可审计输入和缺口，不标通过或入库。

现有 layout 分支依赖本次 annotation 的 `layout_candidates`（`run.py:98–109`），跳过 generator 就不会得到新的候选。因此 layout 应有独立选择计划/证据入口；无需为少量 layout 强迫每条原监督先生成 caption。

### 4. reader 添加“原文/已有候选/新候选”选择，保持一次曝光

位置：`publish.materialize`（15–57）、`reader.iter_training_records`（9–13）。先按训练阶段选可用候选来源，再在其许可长度集合内选择；概率由独立冻结配置给出，本审计不假设 50% 混合。保留语言、短回答指令和唯一 target segment 绑定。每次 materialize 只产出一个训练视图，同一资产的多候选不增加默认曝光次数。

QA/OCR/多轮/交错的原任务不得因候选选择被 caption 替代。附加图片 caption 是另一种显式视图；DCVLM 仍不新增 T2I。当前 `validate_release`（`publish.py:95–96`）只接受 DCVLM 非 T2I，未来 gen-und 发布必须另有已验证的 release contract。

训练阶段需要冻结自己的 release/视图选择策略。当前 parent chain 以 `(collection, original_sample_id)` upsert（`publish.py:82–86`），不是多 stage 同时可选的视图目录；直接把晚阶段写进同一个 current pointer 会替换早阶段可见记录。现通用 reader 也没有历史 tombstone；人工隔离不能只通过“本批不写它”来撤销父版本。现 `review.py` 的补救只针对 parentless canary，不适用于全量历史回收。

## 缓存怎样复用才不破坏保真

当前 `DynamicClient` 已有有用的严格身份：`client.py:172–184` 绑定模型 revision、precision、prompt、完整 context、**有序原图 SHA**、预处理版本、max_tokens、processor/runtime revision、request_overrides 及 client/validation 代码；成功缓存再次通过 validator，schema/请求失败不会变成功。URL 单独变化不应造成语义身份改变，但服务当前仍须 ready 才能走到 cache lookup（152–184）。这意味着离线的复用计划应读取已核验候选回执，不要为了只复用旧产物启动新服务。

缓存 `status=ok` 仅表示模型响应结构合格，不等于 verifier 的 verdict=pass，更不等于来源通过。generator 的 concerns、QC fail/uncertain、人工隔离和当前 source gate 均须重新检查。缓存里 `real_model_inference=True` 是历史事实；当前 summary（`run.py:320–321`）会连 cache hit 一起计入，新增 planner 统计必须区分 `fresh_calls`、`cache_reused`、`existing_imported`、`retained_original`，不能据此夸大本次调用量。

复用按三层分离：

1. 图像资产层：字节、decode、尺寸及相同模型版本的 SSCD 特征可复用；来源冲突与当前政策仍独立评估。SSCD 相似候选不授权 caption 互换。
2. 图片描述层：同一精确图像、相同裁剪/方向/输入视野、适用语言与质量标准的候选可以共享；图表文字不同、编辑图或裁剪不同不能仅因相似度高而共享。
3. 记录监督层：QA/OCR/多轮/交错必须绑定完整有序上下文、全部图片及约束。相同图片不能复用另一条问题/另一段对话的 QC。

## “1M 像素”需要明确记录的三个量

训练阶段的目标像素预算、原图真实尺寸、标注模型实际看到的分辨率/processor 预算必须分别存储。当前 Serving generator registry 已含 `min_pixels=max_pixels=1048576`，client 会应用并纳入身份（`client.py:56–69,172–177`）；这不是未来“1M 训练阶段”已经执行，也不证明低分辨率原图获得了真实 1M 细节。现 image payload 超长边会缩到 2048（72–98），应保留实际发送尺寸/字节 hash 与 processor 配置的区别。

后期 highres 完整 recaption 的计划应显式区分“完整覆盖”与“强制全部新调用”：已有完全相同高分辨率证据、任务契约和质量标准的候选能否满足覆盖，是冻结策略的一项；若要求强制更新，须记录新的 annotation revision/force-refresh epoch。无论采用哪项，都不能把早期低细节候选自动升级成 highres QC，也不能因原图短边 ≥128 就声称达到 1M 原生信息量。

## 最小验收建议

- 离线 planner 不加载服务、不调用 API；同一输入/政策产出相同计划与摘要。
- 原生 QA、OCR、多轮、交错、纯文本在 preserve-original 计划下逐 segment 不变；图像顺序/数量不变；DCVLM 无 T2I 视图。
- 已有短 caption 足够时不要求中长；合格已有候选只引用原证据，未知/URL-only/缺图配对不能升级。
- QA 上下文、语言、图片顺序、输入像素策略、质量政策或人工撤销一旦改变，相关复用失效；只有 URL 变化不影响内容身份。
- reader 能以固定 seed 重现原文/recaption 混合，选中符合原指令的长度；多候选仍一次曝光。
- 阶段覆盖率、模型调用、复用、保留原文、隔离分别计数；本 canary 的 COMPLETE 中“必须有真实生成/layout”条件（`run.py:335–345`）不能直接当成只做保留/复用的生产批次成功条件。
