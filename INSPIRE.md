# 平台资产

工程：`/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain`。

账户 `wjx-ascend`，Workspace `昇腾卡公共空间`，Compute Group `910B资源`，
项目 `多模态大模型新架构评测探索与scaling-law`（`high-dimensionaldata`）。
可观察并发上限 256 卡；提交前重新核对实时 quota、image、资源及可见运行任务。
不停止或修改其他实验。排除故障节点 `infra-gpu-npu-248.host.shzhisuan.com`。

基础镜像 `docker-t.sii.shaipower.online/inspire-studio/dev-wjx-ascend:v-1.4`。训练显式
加载工程内 CANN/NNAL 9.0.0、`.venv-core17`、MCore 0.17 与锁定的 GDN AscendC backend，
不使用镜像中的旧 CANN。永久 Notebook `dev-wjx-ascend` 保留。

**提交 GPU Job 必须显式 `--shm-size 128`**：缺省为 0 GiB，会让 HCCL 的首个 AllReduce
直接超时，表现为所有 rank 卡在 `dist.barrier()` 后报 507048。

## 既有训练资产

- Job `cq-q64-prod128-p3-20261018`，8 实例 × 16 卡 = 128 卡，**优先级 3**（低优不消耗
  点券），平台自动容错：最多重试 100 次、间隔 60 秒。
- 源码快照（不可编辑，历史实验依据）：`.runs/cq-production-q64-20261018/source`。
- 持久产物：`/inspire/sj-ssd3/project/high-dimensionaldata/public/training/cq_q64_production_128_20261018`。
  属持久训练资产，普通实验清理不得删除。
- 原实验的源码快照和持久目录保留用于审计；其 T2I50 输入已退役，不再直接启动旧命令或复用旧数据游标。

## 数据与权重

- 候选合集 `gen-und-mm-pretrain:01` 于 2026-11-03 在昇腾空间通过挂载校验，
  已在该分区 CPU 实例实读 `/inspire/dataset/gen-und-mm-pretrain/01`（只读 GPFS）。
  本次副本为 10 个一级子集，不能沿用 2026-10-21 CPU 空间 21 个子集的完整性结论；
  Short-Caption 仅有辅助文件，LLaVA 两组标注抽样图片引用未命中。现有生产读取器仅支持三个来源的 TAR，
  不能直接覆盖当前散图、独立 JSON 和 OpenUni 的格式。新审计位于
  `/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_mount_preflight_20261103_v1`。
  **CPU 与昇腾分区的项目公共目录不能按同名路径视为共享。** 跨分区仍须实测可见性；
  历史 CPU 审计见 `GENUND_DATASET_REPORT.md`，其中“昇腾无副本”仅适用于 2026-10-21。
- DCVLM small-pool 旧索引：`/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z`，119,010,208 文档；索引与标注保留。其原始池已按用户要求退役，依赖旧 TAR 的图片读取不可用，不再作为可直接训练的正式输入；原生图文顺序与类型的历史产物不改写。
- DCVLM 离线处理运行 `dcvlm_frontpass_20261022T2129Z` 的 LOW Notebook 总申请上限为 **48 卡**（2026-10-23 用户更新，含排队和未确认申请）；轮询补齐、优先已有节点并合并分散实例。启动必须显式 `--max-cards 48`；该限制独立于账号可见 256 卡总额度。运行策略、控制器回执与持久计卡账本见 `corpus_pipeline/cluster/`，不会控制其他实验。
- 通用预算与节点规划模块：`corpus_pipeline/resource_manager/`。DCVLM 通过 `cluster/resource_planning.py` 复用，允许同节点组合 4＋2＋1；队列完成与 worker 资格仍由数据管道负责。新任务通过独立 scope/profile、状态目录、argv 命令及平台适配器接入。模型服务使用原生 Serving 后端，固定每副本卡数；当前 SDK 不提供 Serving 精确节点选择，禁止以 Notebook/Job 替代模型部署来绕过该限制。
- T2I 来源仅为 `gen-und-mm-pretrain:01`。2026-10-22 用户要求删除 `dcvlm_pretrain_t2i50_20261017T093600Z`；该派生版不再作为输入。退役证据见 `corpus_pipeline/validation/t2i50_retirement_20261022/`。
- 旧原始图片 TAR 池 `.../datasets/mlfoundations/dcvlm_pool_small` 已于 2026-10-29 按用户明确授权删除。独立标注目录 `.../datasets/mlfoundations/dcvlm_pool_small_annotations`、旧预处理索引和审查证据保留；删除与保留目录核验见 `corpus_pipeline/validation/small_pool_raw_retirement_20261029/COMPLETE.json`。baseline 原始池不受此次退役影响。
- DCVLM baseline 200B 原始仓库下载目标：
  `/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/mlfoundations/dcvlm-baseline-200b`，
  固定 `mlfoundations/dcvlm-baseline-200b@251523f5a04d7a852223454182cea63bc9f95063`。
  完整清单为 10,756 文件、25,707,038,443,285 字节；下载控制与校验清单位于
  `.../datasets/_downloads/dcvlm_baseline_200b_20261027/`。
  只有该控制目录的 `COMPLETE.json` 与 `manifest.json`、逐文件校验回执相符才算完整可用；
  保留原始 TAR，不解包或生成第二份下载缓存。此资产属于上述昇腾分区项目盘。
- DCVLM baseline 的 DataComp 原图恢复任务：
  **2026-11-04 用户已另行明确授权在现有 baseline 路径替换 DataComp 的模糊图片。**
  本次恢复任务允许修改 TAR，覆盖上文及旧索引合同中针对此任务的原始只读限制。
  控制目录为 `.../datasets/_repairs/dcvlm_datacomp_unblur_20261104/`，实现位于
  `scripts/dcvlm_unblur/`；CPU 实例前缀 `dcvlm-unblur-cpu-1104-`，限定本项目昇腾空间，
  申请最多 256 个槽位，每个 0 GPU / 8 CPU / 128 GiB，故障节点上的槽位排除。
  用户已明确改为完整恢复优先，允许超过 1 小时，不再按时限截断。
  只替换与上游原图 SHA256 完全一致的下载；
  网络失败或哈希不符不计为恢复，也不默认删除样本。`STATUS.json` 跟踪阶段；
  `APPLIED.json` 表示可恢复项已写回，只有 `all_datacomp_blur_removed=true` 才表示无待处理项。
  新索引在该控制目录的 `native/`；旧索引、图像特征和上传清单在写回时失效，不能沿用。
  原模糊成员、替换日志和原始上游版本凭据保留。修复不自动恢复数据集市上传。
- 原始 baseline 的数据集市 CLI 上传目标为 `dcvlm-baseline-200b:v2`，
  SFTP 目录 `/dcvlm-baseline-200b/v2`，副本策略 `all`（QB 七宝与 SJ 松江）。
  独立上传控制目录为 `.../datasets/_uploads/dcvlm_baseline_200b_v2_20261104/`，
  `source_inventory.json` 绑定上述原始 revision、全部文件 SHA256、大小和修改时间；
  上传已按用户要求暂停：用户正在排查 DCVLM 图像模糊问题。原 `transfer.py` 在
  `cpu-wjx-ascend` 的 supervisor 与 rclone 子进程均停止；`PAUSED_BY_USER.json`
  阻止 `start.py` / `transfer.py` 再启动。等待用户明确要求恢复；若源文件变化，
  必须重新核对源清单和目标版本，不能沿用旧校验结论。`status.py` 只读查看。
  凭据由当前账号在进程内加载，不写入控制目录。全量上传与内容校验后才确认版本完成。
  `UPLOAD_COMPLETE.json` 只表示上传校验及提交完成；只有 QB/SJ 副本均为 `active`，
  且 `CPU资源空间`、`昇腾卡公共空间` 均通过挂载校验，才写根目录 `COMPLETE.json`。
  上传或副本验收期间保留该 CPU 和控制目录；训练数据根路径切换仍只修改映射。
- 数据集市上传 MCP 源码与独立依赖位于全局个人目录
  `/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/mcp/inspire-datasquare/`，
  Codex 全局注册名 `inspire-datasquare`。使用前读资源 `datasquare://workflow`。
  `datasquare_upload_status` 可观察现有上传，job_id 为 `dcvlm-baseline-200b-v2-20261104`；
  该任务仍由上述原 supervisor 管理，MCP 不重复启动或接管。
  新上传通过 `datasquare_start_upload` 创建持久任务，完整内容校验后再确认；
  `datasquare_validate_mounts.ready` 仅在 QB/SJ 副本均 active 且 CPU/Ascend 空间均通过平台挂载校验时为真。
  MCP 文件与新任务状态保存在 global_user，旧上传控制目录保持原位。
- baseline 的前序独立索引快照为
  `/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/dcvlm_baseline_200b_20261028_v2/`。
  原始下载目录只读，索引引用固定 `repo@revision`、相对 TAR 路径、成员范围和内容 SHA；
  实际数据根目录通过运行时 `CORPUS_DATASET_ROOTS` 指向的 `source_roots.json` 映射。
  原目录完整复制到平台后只需更新此映射，不能改写原始 TAR 或把新挂载绝对路径写回索引。
  `native/COMPLETE.json` 仅表示全量索引完成；完整检测、复核、去重与训练发布仍需后续阶段回执。
- baseline 全流程独立输出为
  `/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/dcvlm_baseline_full_20261028_v3/`。
  `runtime/` 为冻结实现；`native_index/` 绑定完整上游清单并校验复用前序索引，
  `native/` 仅合并派生索引小文件，不复制图片或改变原始监督。
  `pipeline.json`、`ops.json` 分别绑定数据流程与持久阶段控制；独立 `review_pool/`
  使用上述项目的 Qwen3.6 官方 W8A8 Serving 资产，不复活其他池。
  当前阶段控制入口由 `ORCHESTRATOR_CURRENT.json` 指向，运维策略及无损交接回执保存在
  `operations/`；可独立调整部署目标和截止时间，保留冻结数据实现、既有资源账本与在途工作。
  baseline 的资源预算由该入口绑定的策略管理，不沿用其他实验的固定预算；
  必须验证上一阶段资源释放后切换。
  只有 `CURRENT.json` 所绑定的完整发布回执、全部阶段覆盖和训练引用校验成功才算全流程完成。
  原始数据迁移仍只更新 `source_roots.json`；不改 TAR、样本配对或内容摘要。
- Qwen：`.../models/Qwen3.5-9B`，`Qwen/Qwen3.5-9B@c202236235762e1c871ad0ccb60c8ee5ba337b9a`。
- MOSS：`.../models/MOSS-Image-Base`，`OpenMOSS-Team/MOSS-Image-Base@1ceede80badddd6a7812069619edaa5633587192`。

两组权重均有固定 revision 与文件校验清单。在线分词使用上述 Qwen3.5-9B 的 tokenizer。
MOSS 自带的重复 Qwen 文本权重不另存一份。使用真实本地图片、真实冻结 VAE、RF1。

语料处理使用的 Gemma 独立权重资产：
`/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Gemma4-31B-IT-842da379`，
官方 `google/gemma-4-31B-it@842da3794eaa0b77d5f08bae87a17459d91ff475`，BF16。
9 个文件共 62,578,656,403 字节；完整文件 SHA256 与下载回执见
`corpus_pipeline/serving/gemma4/WEIGHTS_READY.json`，processor/tokenizer/chat template
等输入处理资产内容摘要见 `corpus_pipeline/serving/gemma4/processor_provenance.json`。
此资产属于上述昇腾 Workspace/Project；文件齐备不表示在线推理服务已通过资格验证。

语料生成使用公共 BF16 权重 `/inspire/sj-ssd3/global_public/public_models/Qwen3.5-397B-A17B`。
本项目 Serving 模型登记名为 `corpus-qwen397b-bf16-ref-20261023`，版本 1，
model ID 为 `7ec0bf73-3a39-401b-8c7c-ea3dab1c1d0d`；登记仅引用公共权重，不另存一份。
Serving 要求模型登记与服务属于同一项目，不能直接复用另一项目的登记。
具体推理精度、处理器摘要与在线验收见 `corpus_pipeline/serving/`，登记成功不表示图片推理通过。

DCVLM AI 复核改用 Qwen3.6-35B-A3B Ascend **W8A8_DYNAMIC、TP2 / EP / DP1**。
资产路径为 `/inspire/sj-ssd3/project/high-dimensionaldata/public/models/Qwen3.6-35B-A3B-W8A8-1a118d71`，
来源 `Eco-Tech/Qwen3.6-35B-A3B-w8a8@1a118d717bcbd59480f4a110fe22181d21711b4d`；
25 个文件、10 个权重分片的完整 SHA256 回执见
`corpus_pipeline/serving/qwen36_w8a8_20261024/weights/WEIGHTS_READY.json`。
同项目模型登记为 `corpus-qwen36-35ba3b-w8a8-1a118d71`、版本 1，
model ID `51162c97-22a5-4f27-9f00-57b2556bb04d`。该量化资产与公共 BF16 版本不同。
部署采用 vLLM-Ascend **v0.23.0rc1 的 A2/A3 官方 W8A8 双卡配方**，配套平台镜像
`vllm-ascend:v0.23.0rc1`（`image-03380f97-1e02-4ed9-8c04-1e1797828ce9`）。
固定版本依据、完整启动参数和权重绑定位于
`corpus_pipeline/serving/qwen36_w8a8_20261024/official_v023/CONTRACT.json`；
容器直接运行官方 `vllm serve`，仅将模型仓库名替换为本地已校验权重路径。
推理参数不经过自写模型入口或网关；外部 SDK 管理资源，CPU 使用原生 OpenAI SSE 接口。
共享节点的主机级 sysctl / CPU governor 由平台管理员负责，不在服务容器中修改。
原生 Serving 总预算仍为 **128 卡**；通用 `corpus_pipeline/serving/budget_pool.py`
配合模型 backend 管理最多 64 个双卡 LOW，独立 HIGH 若仍占卡也计入总上限。
排队、未确认申请和抢占重启均保留原预约，不创建超额替代实例。
配置、资源账本与就绪租约位于 `corpus_pipeline/serving/qwen36_w8a8_20261024/pool/`；
通用契约见 `corpus_pipeline/serving/BUDGET_POOL.md`。旧 397B 池和旧标注仅保留历史。
本轮按用户要求只做当前启动的基础图片请求/身份检查，随后直接用正式队列测试 CPU 并发；
该基础检查不等于模型 AI 识别质量验证。
全量来源对账与可恢复复核执行器见 `corpus_pipeline/postprocess/README.md`。
当前正式策略为视觉注释 `annotate`：模型意见不自动清除检测器报警，不改写来源事实；
后续策略变更须保留已知标签验证与独立输出版本。

DCVLM 全量 SSCD 索引的独立持久目录为
`corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z/postprocess_20261024/sscd_full_20261025/`。
输入绑定上游完整 `reconciliation/COMPLETE.json`，源码冻结在该目录的 `runtime/`；
通过 `launch.sh` 启动或恢复，`status.sh` 只读查询。仅复用昇腾分区共享盘上的既有 FP32 特征，
CPU 计算不需要 NPU，也不依赖 CPU 资源空间的隔离存储。索引、召回校准和全量查询各有检查点；
是否完成只看 `index/supervisor/COMPLETE.json` 及其绑定产物。候选边不自动删除图片或监督，
代表图筛选必须尊重绑定的 AI／质量结果。原始特征、已提交分片和输入快照均为后续消费者资产。

活动模型配置 `ascend_pretrain/configs/qwen35_9b_moss_q64.json`：CQ queries **64**，
理解与生成面积上限均 262,144 pixels（生成含补边），ViT 编码层冻结、merger 可训练。

## Base 与 ungated cross 对照资产

- Base 的固定源码：`.runs/cq-resume-compat-20261020/source`；入口：
  `.runs/cq-base-q64-20261019/run.sh`；持久产物：
  `/inspire/sj-ssd3/project/high-dimensionaldata/public/training/cq_base_q64_prod128_20261019`。
  保留完整 checkpoint，用原 TP2 配置恢复；不得跟随 cross 清理。
- Ungated cross 的固定源码：`.runs/cq-base-crossattn-ungated-20261021/source`；入口：
  `.runs/cq-base-crossattn-ungated-20261021/run.sh`；持久产物：
  `/inspire/sj-ssd3/project/high-dimensionaldata/public/training/cq_base_crossattn128_ungated_20261021`。
  使用 TP4、Q=64、global batch 128。每层为 `x + to_out(y)`，输出权重初始标准差
  `0.02 / sqrt(2 * 42)`，新投影在 `no_init_weights()` 外初始化并按 TP 分片。
  该版本从官方预训练权重开始；只从自己的输出目录恢复，不兼容旧 gated checkpoint。
- 上述两项依赖 T2I50 的自动重启绑定随数据退役停用；checkpoint 保留。后续改用新语料须建立新运行身份，不改写历史快照。
- 上述源码快照视为不可编辑。完整提交 payload、源码摘要和控制器回执保存在各自
  `.runs` 目录，仅作本地运行记录；不得加入 Git。

## 更早的资产

- `cq_qwen35_9b_moss_128_20261017`：Q=256 + 旧数据集的 128 卡训练，已于 step 1200
  停止（`cq-qwen9b-moss128-p3-20261017`，CANCELLED）。checkpoint 保留。
- `.runs/cq-speed-20261017`：Q=64 调优证据（见 TUNING.md）。
- `.runs/cq-t2i50-tune-20261017`、`.runs/cq-document-tune-20261017`：t2i50 与整文档
  打包的前序调优快照。

API 和权重下载按用户要求直连：调用级移除 `HTTP_PROXY`、`HTTPS_PROXY`、`ALL_PROXY`
及其小写形式；不改变全局代理配置。训练设置 Hugging Face offline。

## GenUND 六子集与 DCVLM-baseline 合并（2026-11-03）

**新任务入口（2026-11-04）：**统一使用 `python -m corpus_pipeline`，配置模板见
`corpus_pipeline/orchestration/examples/genund.json`，操作说明见 `corpus_pipeline/unified/OPERATIONS.md`。
CPU 生命周期、LOW 单卡调度、512 卡预算、同进程审查预检、HTTP 接收和精确发行已进入维护代码；
新任务冻结当前代码，不再复制旧 `ops/`、`sdk_compat` 或带版本号的运行补丁。
下文的运行目录与脚本是**已在运行任务的冻结绑定与历史追溯**，不是新任务模板。

独立运行目录为
`/inspire/sj-ssd3/project/high-dimensionaldata/public/preprocessing/genund_six_with_dcvlm_20261103_v1`。
绑定官方 `gen-und-mm-pretrain:01`，在上述昇腾 Workspace/Project 的 Notebook 内只读访问
`/inspire/dataset/gen-und-mm-pretrain/01`；输入身份见 `DATASET_BINDING.json`，
移动挂载后仅调整 `source_roots.json`，原始图片与旧 DCVLM 发行内容不改写。
六个显式范围是 OpenUni、ScaleCap450k_image、DenseFusion-1M、pixmo-ask-model-anything、
FineVision、BLIP3o-Pretrain-Long-Caption，不代表整个 GenUND 上游集合。

本轮 I2T 统一关闭图像 loss；三套 caption 集保留 I2T/T2I 两个共享图片的视图，
T2I 排除已知 AI 来源并执行检测与复核，完成复核后仍不确定者不进入 T2I。
OpenUni 按官方原生 T2I 配置解析，问答/对话保留原结构。
按相同图像、相同有序监督及任务方向精确去重，保留不同 caption；
本轮不执行仅用于建议的全库 SSCD 近邻搜索，前向 SSCD 特征仍保留。
现有 DCVLM-baseline 的 70,004,441 条已选记录由 `baseline_reuse/` 的校验索引复用。

持久入口为 `ops/stage_controller.py`，配置 `ops/stage_controller.json`；
CPU 恢复与完成后回收由 `ops/cpu_watchdog.py` 管理，只操作本轮登记实例。
全部七个执行来源的原生整理回执封存后，`ops/CPU_RETIREMENTS.json` 绑定其 SHA，
允许回收已完成整理的 CPU，并禁止守护程序重新拉起它们；保留配置指定的审查 CPU。
资源、截止时间与任务规则见 `ops/POLICY.json`。分片进度见 `ingest_workers/`、
`normalized_sources/`，后续进度见 `ops/stage_control/STATE.json`。
本轮 CPU 可用高优；NPU 前向使用大量 LOW 单卡实例领取独立任务，
总预约上限包含排队申请。不得为了等待完整节点而保留未运行的多卡前向申请。
用户已将本轮总上限调整为 512 卡，排队和未确认申请同样计入。
审查池最多 256 个 LOW TP2 副本；后续前向继续使用单卡，按数据量分配，最多 512 卡。
`runtime_512/` 与 `ops/one_card_manager_512.py` 承担后续来源；已开始的来源保留
其原有冻结运行环境。预算迁移凭据在 `ops/ceiling512_migration/`，原始创建意图保留。

GenUND 的独立图片审查使用 `runtime_review_portable/` 中的 API 审查实现，
由 `ops/portable_review_bootstrap.py` 在 CPU 审查入口加载；冻结的初筛与汇总实现保留。
图片仍使用固定 dataset ID 和相对路径，通过 `source_roots.json` 定位并校验原始字节 SHA256。
实现变更与验证凭据位于 `ops/portable_review_fix/`。审查凭据仅在进程内加载。
正式审查入口必须在同一进程中完成一张真实图片的成功预检，随后才能领取全量队列；
独立 canary 成功不能替代正式进程验证。入口等待期间若脚本发生变化，必须重启加载。
`ops/review_health_watchdog.py` 对启动后持续零有效返回且大量请求失败的进程中止重试；
失败重排凭据保存在 `ops/credential_recovery/`，基础设施失败不能作为审查通过。
`ops/review_http_transport.py` 用 aiohttp 接收原生 SSE 完整字节，仍由原 HTTPX 序列化请求，
原客户端执行图片、平台身份、SSE 结束标记与审查结果校验；实现 SHA 和依赖版本绑定在
正式进程的 `REVIEW_IMPLEMENTATION_AMENDMENT.json`。切换须先等在途请求结束，并重新通过同进程预检。
预算准入按全部已申请卡数计数；物理分配情况由模型池独立观察，避免在每次创建前重复查询实例。
`ops/sdk_compat/genund_bound_serving.py` 对已绑定服务使用原生 ID 直接查询并保留完整归属校验，
未知创建仍按原创建意图核对；管理查询按动作限速，避免重复遍历服务历史触发平台限流。
`ops/sdk_compat/genund_utilization_pool.py` 为本轮原生模型池区分运行与排队观察通道，
优先刷新运行服务，排队服务降低轮询频率；原生归属、预算、30 秒就绪租约与抢占恢复校验保留。
模型复核保持 LOW，使用已验证的模型运行规格；资源合并须在检查点与实际释放确认后进行。
DenseFusion 官方额外 caption 与可修复原图作为独立 supplemental 执行源，
在最终统计中仍归入 DenseFusion-1M；六个逻辑来源的最终完成条件包含该补充分支。
不能根据实例 RUNNING 或局部 COMPLETE 判断六子集合并完成；最终只认本轮根目录
`COMPLETE.json` 与其绑定的 `joint_release/COMPLETE.json`。

## DCVLM SSCD 并行执行

上述昇腾 Workspace/Project 的当前 SSCD 执行目录为
`corpus_pipeline/cluster/runs/dcvlm_frontpass_20261022T2129Z/postprocess_20261024/sscd_parallel_20261025/`。
共享原始特征与冻结 runtime，`launch.sh` / `status.sh` 为入口；旧 `sscd_full_20261025` 产物保留用于审核和分片复用。
并行打包、共享全局索引的查询以及每阶段完整性验证均有可恢复回执。
本共享盘已实测：跨主机互斥须用 POSIX `lockf`，不能使用仅本机有效的 `flock`；同进程多线程须另有线程锁。
本次新建资源由资源账本约束，完成后回收；不会停止常驻 CPU 或其他任务。
