# 视频训练与统一数据管线

## 数据契约

每行一个样本、一个视频。`task` 必须是 `t2v` 或 `v2t`，可写在记录中，也可由配置的 source 指定；两处冲突时拒绝加载。所有媒体、缓存相对路径均相对 JSONL 文件，source 的 manifest 路径相对启动工作目录。

```json
{"sample_id":"clip-001-t2v","task":"t2v","messages":[{"role":"user","content":[{"type":"text","text":"A blue block moves right."}]},{"role":"assistant","content":[{"type":"video","video_id":"v0"}]}],"videos":[{"id":"v0","path":"clips/001.mp4","qwen_cache":"cache/qwen/001.pt","wan_cache":"cache/wan/001.pt"}]}
```

V2T 把 video 放在 user 消息中，assistant 是待监督的描述。两类任务都监督序列中所有非 visual、非 padding token，包括用户文本、对话模板、时间戳和帧间特殊 token；这是本阶段持续预训练目标，不是 assistant-only SFT。标签与 query 同位置，不做 next-token shift。

文本 query 使用 `<|text_mask|>`，视觉 query 使用 `<|vision_mask|>`，分别对应现有配置字段 `mask_token_id`、`video_mask_token_id`。它们是 backbone 同一张 embedding 表中的两个独立行；原生 `<|image_pad|>`、`<|video_pad|>` 等 content 占位符保留原名。

训练、推理和缓存准备共用 `load_processor()`。加载旧 processor 时，`<|mdm_mask|>`、`<|video_mask|>` 会在内存中重命名，并保留各自原有 ID，继续对应 checkpoint 中原来的 embedding 行；原 processor 文件不会被改写，新保存的 processor 使用新名称。若旧名和对应新名同时存在，会明确报错，避免选择错误的权重行。这一映射仅处理 processor 的 token 命名；旧训练 checkpoint 和权重导出不再支持恢复。

简单标注也可直接使用：

```json
{"sample_id":"clip-001","video_path":"clips/001.mp4","caption":"A blue block moves right."}
```

简单标注由 source.task 决定视图：T2V 以 caption 为 prompt；V2T 默认指令为 `Describe this video.`，答案为 caption，可设置 source.instruction。`temp_data` 的物理标注不会自动加入输入。可以直接给同一 manifest 配两个 source；这由配置明确表达，代码不自行拆分数据。JSON sidecar 也兼容。

NPY 输入可用 `frames_path` 和明确的原始 `fps`，格式为 uint8 `[T,H,W,3]`。同时提供 MP4 和 NPY 时优先使用 NPY。MP4 使用 PyAV 读取真实 PTS。按目标 fps 选择最近的源帧，超出源视频时长则报错。共同的空间预处理是等比例 bicubic resize + center crop，随后 Qwen 和 Wan 使用同一片段；Qwen 再执行自身原生 resize/normalize/temporal merge。

首期设置为 9 帧、4 fps、432×768，Qwen 使用全部 9 帧。Qwen 原生 temporal merge 会把最后一帧补齐为一对，输出 5 个时间网格；本地原生 Processor 得到 `[5,28,48]` 网格，即 1,680 个 visual tokens。Wan VAE 得到 `[16,3,54,96]` latent。这两个 token 网格无需相同，Wan 使用 cross-attention 接收 CQ 条件。Qwen 帧数可通过 `data.video.qwen_num_frames` 后续单独调整，像素预算通过 `qwen_processor` 传给原生 video processor；不允许在它内部再次抽帧。

`data.max_seq_length` 超限会报错，不截断半个视频。批内仅右侧 padding，一个样本对应一行。batch size 控制数据并行的独立样本数，不表示一个样本含多个视频。

## 数据混合

分别提供来源、默认按任务 1:1 交替：

```yaml
data:
  mixing: balanced
  task_weights: {t2v: 1, v2t: 1}
  sources:
    - {name: generation, path: /data/t2v.jsonl, task: t2v}
    - {name: understanding, path: /data/v2t.jsonl, task: v2t}
  validation_sources: []
```

`task_weights` 是整数微批次数，如 2:1 会产生 `[t2v,t2v,v2t]` 周期；梯度累积必须包含完整周期。单任务将另一权重设为 0。同一任务的多个 source 默认按记录数混合；显式 source.weight 表示该来源在该任务内的抽样权重（放回采样）。未设置 weight 的来源权重为它的记录数。

已经混合好的清单保留其样本比例：

```yaml
data:
  mixing: preserve
  sources: [{path: /data/mixed.jsonl}]
```

此时不再根据 task_weights 重采样；允许一个微批内同时出现两种任务。多个 manifest 也可用 preserve 拼接后打乱，不能同时设置 source.weight。任务比例与 `lambda_text` / `lambda_video` 独立。纯文本数据和多视频单样本暂未接入。

## 缓存与在线编码

```yaml
data:
  representations: {qwen: prefer_cache, wan: prefer_cache}
model:
  freeze_vision: true
```

每套表示独立选择 `cache`（必须有缓存）、`online`（从原视频即时编码）、`prefer_cache`（记录有缓存引用则使用，否则在线）。支持整批两者都缓存、都在线，或 Qwen/Wan 只有一条分支使用缓存。**同一分支的一个 batch 内必须统一表示来源，不能混用缓存和在线视频**；Wan 只检查该 batch 中的 T2V 目标，V2T 不读取 Wan latent。`prefer_cache` 若使同一 batch 出现两种来源，会明确报错；应补齐缓存或将该分支统一设为 `online`，不自动重排样本或切换模式。坏缓存或不匹配的缓存直接报错，不静默回退。这里的“在线”指运行时编码，与网络、视频流式生成和 KV/GDN cache 无关。

缓存生成入口默认复用训练时相同的固定片段预处理和原生 Processor：

```bash
python -m scripts.prepare_latents --config configs/tests/ascend_smoke.yaml \
  --manifest ../temp_data/slider_stop.jsonl --task t2v --kind both \
  --cache-dir /data/cache --output /data/t2v.cached.jsonl
```

`--kind qwen` / `wan` 可单独生成。省略 manifest 时处理配置中的全部 sources，输出一个带显式 task 的规范清单。已有缓存经校验后复用；不覆盖已有输出清单。

已经完成隔帧、缩放和裁剪的视频，使用 `configs/prepare_latents_native.yaml` 的 `data.video.mode: native`。此模式读取文件现有分辨率、容器帧率和真实 PTS，仅删除末尾 0～3 帧以满足 Wan 的 `4k+1`，不限制为固定长度；至少需要 5 帧，Wan 的 H/W 必须整除 16。`num_frames/fps/height/width` 设为 `null`，实际值记录在每条缓存的 `clip` 中。

Qwen 在上述对齐片段上独立调用原生 `sample_frames()`。示例按 `qwen_processor.fps: 2.0` 计算抽帧数，再限制在 `min_frames: 4`、`max_frames: 16` 和实际帧数以内，均匀选帧；这不会改变 Wan 的帧率或输入。固定帧数时设置 `qwen_num_frames`，并将 `qwen_processor.fps` 设为 `null`；原生固定帧数路径不使用 min/max 限制，超过片段帧数则报错。Qwen 保留实际采样 PTS，关闭二次抽帧和 resize，H/W 必须整除 patch_size × merge_size（原生 Qwen3.5 为 32）；奇数帧的最后一帧仍由原生 temporal patchify 补齐。

原生模式的离线提取和在线读取共用 `data/video.py`，支持两条分支独立选择 cache/online/prefer_cache。已有 v1 缓存仍要求相同的完整 `data.video` 配置；v2 索引可自动读取 profile。`cache` 模式缺缓存时报错。生成评估仍需在 `sample.video` 显式指定生成参数。示例的 Qwen 抽帧上限为可调预算，并不保证任意源分辨率都能满足 `max_seq_length`。

Qwen 缓存为 post-merger `[N,D]` 特征和 `grid_thw`；Wan 缓存为 posterior **mean** 经原生均值/标准差归一化的 `[16,T,H,W]` latent。VAE 使用 Diffusers 的官方 Wan 权重转换，严格加载本地 `Wan2.1_VAE.pth`，始终冻结，在 `data/codec.py` 管理，不属于训练模型参数。

两个缓存都记录采样帧、实际时间戳、空间变换、源文件 SHA-256、预处理配置和编码器标识。旧 v1 编码器标识由本地资产路径、大小和修改时间计算。新 v2 使用模型内容 SHA-256，并记录计算精度、实现版本和保存策略；Wan 只依赖 VAE 资产，不因 Qwen 抽帧设置变化而失效。不同缓存或在线片段的元数据必须一致。缓存和在线编码使用不同精度时可能有舍入差异；部署时应统一编码精度。

`freeze_vision: false` 必须搭配 `qwen: online`，梯度才可回到 ViT。曾训练 ViT 的 UMM 导出也必须使用在线 Qwen 输入；当前缓存脚本只从原生 qwen_path 编码，不生成已微调 ViT 的缓存。

## 原位扩充 OpenVid 81 帧索引

两个分辨率各自保留唯一的 `samples.jsonl`，原字段和行顺序保留，不新增第二份样本索引。每行新增 `qwen_cache`、`wan_cache`（相对该索引目录）、`latent_profile`（共享配置 JSON 路径）和 `latent_info`。info 按分支记录 format、shape、dtype、file_bytes、文件 SHA-256；Qwen 另含 grid_thw、实际抽帧数及 T2V/V2T 序列长度。真实 PTS、源视频 SHA-256、帧索引、完整编码契约放在 `.pt` metadata 中。缓存 tensor 只保存一份。

```bash
source scripts/env_mindspeed.sh
.venv-npu/bin/python -m scripts.prepare_latents \
  --config configs/prepare_openvid81.yaml \
  --manifest /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_81frame/832x480/samples.jsonl \
  --in-place --device-index 0
```

对 `960x544/samples.jsonl` 使用相同命令。小试通过后可运行 `python -m scripts.prepare_openvid81 --root /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_81frame --devices 0,1,2,3`：它检查两种分辨率的小试/续跑报告，启动独立 worker，并在提取后执行全量只读复核。总状态保存在根目录 `metadata/latent_validation/full_status.json`。该配置要求真实解码恰好 81 帧，保留原尺寸、容器帧率和 PTS，不裁帧或缩放。Qwen 使用原生 2 fps、4～16 帧抽样。模型以 BF16 计算时保存其实际 BF16 输出；切换计算精度时仍直接保存实际输出，不强制降精度。Wan 使用 FP32 posterior mean 和官方归一化。

latent 默认位于索引旁 `latents/{qwen,wan}/<分支契约摘要>/<前缀>/<样本及源内容摘要>.pt`。配置在 `metadata/latent_profiles/`；进度、原始字段摘要在 `metadata/latent_runs/`。摘要用于检查，不能替代数据备份。原视频不变，旧子集验证报告中的原索引哈希作为历史记录保留；当前索引哈希在每次 latent 运行报告中更新。

每个文件先写临时文件并 fsync，再原子提交；索引使用文件锁和原子替换。不同 worker 的已完成项会合并，默认每 16 个样本写回一次，退出时也写回已完成项。只有两条请求分支都成功且序列检查通过才添加该行引用。断电可能留下无引用的完整文件或 `.tmp`；续跑会校验并复用完整文件，临时文件不会被当作 latent。已有损坏文件明确报错，不静默覆盖。每次续跑仍校验源文件 SHA-256，但可跳过视频解码和编码器加载。

小试可以加 `--sample-ids /path/pilot_ids.json`（样本 ID 数组）或 `--limit 4`；`--verify-online` 比较实际模型重新编码及 T2V/V2T Dataset 读取；`--stop-after-files 1` 用于模拟文件保存后中断。`--verify-only` 要求所有文件已存在，执行内容/配置/序列校验。正常重跑同一命令即可续跑。并行时每个进程显式设置 `--num-shards N --shard-index 0..N-1 --device-index 0..N-1`；按原索引行号取模划分，两个分辨率的同一 ID 分给同一 worker。不同运行用 `--run-id` 区分报告。

训练时只需要索引路径、任务以及所需表示，不必重复视频尺寸和抽帧配置：

```yaml
data:
  representations: {qwen: cache, wan: cache}
  sources:
    - {path: /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_81frame/832x480/samples.jsonl, task: t2v}
    - {path: /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_81frame/832x480/samples.jsonl, task: v2t}
```

`load_config()` 从索引引用的 profile 推导 `data.video`，显式配置仍可覆盖，但实际使用缓存的分支会校验契约。Qwen 在线且 Wan 缓存时可以改变 Qwen 抽帧；改变 Wan 输入尺寸/帧数则不能复用旧 Wan latent。训练恢复契约同时包含索引、媒体文件状态和 profile 内容。原始文件读取可使用 `read_manifest()` / `resolve_video()`，无需附加任务标签；训练视图仍通过 source.task 构造。

## 目标、优化与默认参数

Flow 定义 `z_s=(1-s)*z_clean+s*noise`，训练目标 `noise-z_clean`，Wan 时间为 `1000*s`；生成从 s=1 积分到 0。每个视频先对 latent 元素求 MSE，再按目标视频求均值。文本按有效 token 求 CE。微批目标为 `0.05*CE+1.0*video_MSE`，梯度累积对微批目标取均值；日志分别统计文本 token 数和视频数。

`flow.repeats` 控制每个目标视频的 flow 重复抽样次数，必须是正整数，默认 `1`。例如 `--set flow.repeats=4` 会复用一次 CQ backbone 前向得到的视频条件，为每个目标抽取 4 组独立的时间与噪声，并将重复条件和加噪 latent 一起送入 Wan。视频 loss 对全部视频和重复次数取平均，文本 CE 只计算一次，日志的视频数仍统计原始目标数，不乘重复次数。CFG 仍按原始目标视频采样一次，所有重复共用该决定；content/query 计算和可见性不变。重复次数增加 Wan 的计算及激活显存，不增加 CQ 前向次数，也不改变推理采样步数。V2T 没有视频 flow 目标，因此不受该参数影响。

```yaml
flow:
  repeats: 1
```

默认值在 `configs/config.py`，来源及选择如下：

| 项目 | 默认值 | 依据 |
| --- | --- | --- |
| 语言骨干/embedding LR | 1e-5 | `cq-pretrain/ascend_pretrain/train.py:40` |
| Wan/条件投影 LR | 1e-4 | 参考 `cosmos-framework/examples/toml/sft_config/vision_sft_nano_mapstyle_dataloader.toml:61` 的全量微调 LR；并非该仓库针对 Wan 的结论 |
| Adam betas/eps | (0.9,0.95) / 1e-6 | 同一 Cosmos 配置第 52–53 行 |
| 语言 weight decay / Wan weight decay | 0.1 / 0 | CQ `ascend_pretrain/parallel.py` 的 optimizer 默认值；Cosmos 同配置第 62 行。bias/norm/embedding/投影不 decay |
| 文本/视频 loss 权重 | 0.05 / 1 | 参考 CQ `ascend_pretrain/config.py:51`；将其图像权重作为视频初值 |
| 时间采样 | sigmoid(N(0,1))，eps=1e-5 | CQ `ascend_pretrain/input_pipeline/vae.py:138`，默认不混均匀分布 |
| 总更新步数 | 100,000 | 采用 CQ 配置的 100k 衰减时间尺度作为可覆盖预算，不沿用其 12 步基准测试 |
| warmup / 最后 cosine decay | 1,000 / 10,000 步，最低 0.1 LR | 本项目的初始调度选择，可覆盖；不是照搬参考的短程 SFT/benchmark |
| per-rank batch / accumulation | 1 / 8 | 四卡时每次更新 32 个样本，默认 T2V/V2T 各 16 个 |
| 精度 / EMA | BF16 autocast / 开启，0.9997 | 保留 FP32 优化参数/累积和原分片 EMA 设施 |

这些是可运行的初始值，未经过视频质量调参。T2V CFG 条件丢弃概率默认 0.1，只删除视频 query 的前缀读取，content 不变。`train_language`、`train_wan`、`train_adapter`、`freeze_vision` 分别控制更新范围；冻结语言及 ViT 时不能安排 V2T 更新。

正式配置 `configs/video.yaml` 不指向合成样本。`configs/tests/ascend_smoke.yaml` 单独提供两步验证：四卡、每卡 batch=1、累积=2、EMA 关闭，使用明确指定的 slider_stop 两种视图。设备默认 NPU，若 torch_npu/设备不可用会报错，不静默切换 CPU。910B 测试配置明确选择 `model.wan_attention_backend=npu`，展开 NPU attention 所需的 mask query 维度；Qwen Full Attention 保持使用支持显式 CQ mask 的 SDPA。其他配置仍可选择 `auto/npu/sdpa`。

## 训练、恢复与推理

在仓库根目录运行 shell，或用绝对路径指定 source。

```bash
# 短程 910B 测试
bash tests/validation/test_train.sh

# 正式训练
bash scripts/train.sh --config /data/run.yaml

# 恢复模型、optimizer、scheduler、EMA、各 rank RNG、已消费的数据游标
bash scripts/train.sh --config /data/run.yaml --resume outputs/umm_video/checkpoints/checkpoint-2000
```

`tests/validation/test_train.sh` 默认选择 `.venv-npu/bin/python`、项目级 CANN 9.0.0 和 `HCCL_INTRA_ROCE_ENABLE=1`，支持当前四卡跨通信平面的单机拓扑。环境准备命令为 `python3 scripts/setup_mindspeed.py --sources --python --cann`；手动执行 Python/NPU 检查前先 `source scripts/env_mindspeed.sh`，Bash 和 zsh 均支持；从其他目录调用时使用脚本绝对路径。首次准备会备份旧环境，系统 CANN 不被覆盖。默认测试输出为 `outputs/ascend_smoke/`，正式配置为 `outputs/umm_video/`。

### MindSpeed / Megatron Core

`configs/config.py` 的默认后端和两个视频 YAML 均选择 `distributed.backend=mindspeed`，使用 torchrun、HCCL、MCore DDP 和分布式 AdamW；`distributed.num_processes` 默认 4。当前只支持 TP=PP=CP=1，保留模型层和权重布局。BF16 模型权重在 DP rank 间复制，FP32 master weights 和 Adam 状态分片保存。`overlap_grad_reduce`、`overlap_param_gather` 默认均为 true；bucket_size 默认 40000000 个元素。冻结范围、各角色学习率/weight decay、WSD 和 flow repeats 保留独立配置。

视频默认使用 `model.qwen_gdn_backend=ascendc`，与 main 共用投影、短卷积、只读 query 事件和 AscendC 反向。`mindspeed` 保留作为显式对照。`model.content_visibility` 可选 `visual_block`（默认，原生时间块内双向、块间 causal）或 `causal`。Wan clip attention 始终双向，变长序列以 TND 紧凑执行，不构造 padding attention mask。

Triton-Ascend 3.2.1 和社区 Triton 3.2.0 共享部分安装路径，必须通过 `scripts/setup_mindspeed.py --python` 保证 Ascend wheel 最后安装；首次编译需要主机 g++/make，可用 `--host-tools` 单独安装。CANN/Torch/torch_npu 仍保持 CQ 版本。MindSpeed 与 Triton-Ascend 的 NumPy 元数据约束无交集：保留 1.26.0，并在 `configs/deps.core17.lock.json` 记录单个例外；setup 只接受这项已知冲突，其他依赖冲突仍报错。首次不同形状会触发 Triton 编译，冷启动时间不代表常态吞吐。

安装结束及 `source scripts/env_mindspeed.sh` 时，都会执行标准库实现的 `utils/triton_integrity.py`。共享文件以最后安装的 Ascend RECORD 为准；检查锁定版本、关键记录、导入路径，以及全部 Ascend 所有的 `triton/` 文件大小和 SHA-256，跳过生成的 `.pyc`。当前检查 92 个文件，其中 38 个为带哈希的共享文件，约读取 705 MB；本机读取并计算哈希约 0.8 秒。缺失、覆盖或导入遮蔽会返回非零并报告路径，不自动重装或忽略错误。直接解析 RECORD，以免 metadata API 过滤已被删除的文件；这是防止安装覆盖的检查，不是对 RECORD 本身被篡改的安全认证。

无需先加载 CANN，也可单独运行 `.venv-npu/bin/python utils/triton_integrity.py`。若文件受损，使用 `python3 scripts/setup_mindspeed.py --python` 按锁重装；若错误是导入路径遮蔽，先清理对应 PYTHONPATH。这里的完整性通过不代表 NumPy 元数据冲突消失。

环境入口默认设置 `ASCEND_WORK_PATH=.cache/ascend-work/<主机名>/` 的绝对路径，并保留用户覆盖；pytest 使用相同规则。CANN 的 `kernel_meta` 编译调试文件以及受该变量控制的运行日志等写入此工作目录，不混入 UMM 的实验日志。共享算子缓存 `ASCEND_CACHE_PATH` 没有改动。未加载项目环境时，CANN 仍可能在当前目录创建 `kernel_meta`；这不是模型源码或 checkpoint。含义与优先级见 [CANN 官方说明](https://www.hiascend.com/document/detail/zh/CANNCommunityEdition/900beta2/API/appdevgapi/aclcppdevg_03_1371.html)。

运行日志和 checkpoint 合同记录实际 GDN provider、源码摘要、Triton 版本和 CQ 行索引实现版本。改变后端不属于管理参数调整，不能作为精确恢复继续旧实验；需要用导出权重初始化新实验。旧训练结果不做后端字段补齐，必须按当前完整配置重新开始实验。跨后端比较允许 BF16 舍入差，连续/恢复对照则必须在相同后端、配置和环境下执行。

统一评估的 `evaluations/runtime.py` 在第一次前向前导入并审计所选 GDN 后端，避免在扫描中途安装 MindSpeed adaptor 补丁；推理保留原来的 `torch.compile`。选择 MCore 配置时，加载后的模型显式转为 BF16，与训练时的参数和残差流 dtype 一致，不能仅依靠 autocast 将 FP32 模型当作同一计算设置。

训练侧按固定顺序派发通信桶，防止不同 rank 的 T2V/V2T 分支跳过模块后发生 collective 顺序错位。最后一次 backward 后，只为缺席参数补齐就绪标记，不清空此前累积的梯度。整窗口未使用但仍可训练的参数采用零梯度，因此 Adam 动量/weight decay 仍可更新；要停更模块需显式冻结。Qwen 短卷积直接调用共享 CausalConv 模块以触发权重收集；Wan 时间分支保持 FP32。重算允许同一参数多次触发梯度就绪事件。

每个优化器窗口先统计所有 DP rank 的有效文本 token 数 N 和目标视频数 M，再给本地均值 loss 乘 D*n/N、D*m/M。梯度归约自身包含 1/D，不再额外除以累积次数。flow.repeats 对同一视频的独立 flow 抽样取均值，不放大视频权重。日志的 `train/objective_global` 使用全局 token/video 计数重建目标；分任务指标仍分别记录各任务的原始均值。

`distributed/megatron.py` 只负责运行时、优化器和分片状态；`distributed/communication.py` 负责条件分支通信。EMA、评估和导出前完成参数收集；评估暂时解除参数收集 hooks，避免覆盖 EMA 或在纯推理中触发训练通信。低层版本与源码提交固定在 `configs/deps.core17.lock.json`，恢复时同时校验 backend、拓扑、通信设置和 optimizer bucket 布局。

MCore checkpoint 的 `megatron/model.pt` 保存一份完整模型，各 rank 的 `optimizer_rank_XXXXX.pt` 保存 FP32 master、Adam 状态、scheduler 与通信就绪状态；外层各 rank runtime 文件保存 RNG 和已消费数据游标。旧 DeepSpeed 训练、启动配置和恢复能力已移除。CPU 回归在同一个 `.venv-npu` 中运行，显式选择 `distributed.backend=accelerate`、CPU 和原生计算后端，不与 MCore 组合；启动前设置 `TORCH_DEVICE_BACKEND_AUTOLOAD=0`。`accelerate` 仍是 MCore 数据分片的依赖，不能因移除 DeepSpeed 而卸载。依赖统一由 `requirements-mindspeed.txt` 和源码锁管理。

### 实验、日志和权重目录

```yaml
training:
  exp_id: my_experiment
  output_root: ./outputs
  output_dir: null
  log_dir: null
  checkpoint_dir: null
  keep_checkpoints: 3
```

默认实验布局如下。旧日志不搬移；新训练和评估启动采用新层级。

```text
outputs/<exp_id>/
├── config.yaml
├── logs/
│   ├── train/<attempt_id>/
│   └── eval/<eval_id>/<attempt_id>/
├── checkpoints/
│   ├── checkpoint-<step>/
│   └── exports/             # weights.pt、可选 weights-ema.pt、processor/
└── evaluations/<eval_id>/
    ├── run.json
    ├── config.yaml
    ├── validation/
    ├── preview/
    └── vbench_subset/
```

`exp_id` 隔离实验，`eval_id` 隔离固定权重和评测协议，`attempt_id` 标识每次启动。任务目录名称来自任务 YAML 的 `name`，上图仅作示例。`run.json`、`config.yaml`、`.writer.lock` 是保留名称，不能用作任务名。输出相对路径以 UMM_v3 根目录为基准。`training.output_dir` 指定完整实验目录，不再追加 exp_id；`training.log_dir` 指定公共日志根，训练继续追加 `train/<attempt_id>`，评估追加 `eval/<eval_id>/<attempt_id>`；checkpoint 根可独立覆盖。`evaluation.output_dir` 是完整评估结果目录，不再追加 eval_id，`evaluation.log_dir` 是本次评估的日志父目录，只追加 attempt_id。结果、日志、checkpoint 目录不得互相嵌套。

实验根 `config.yaml` 仅由成功初始化或恢复的训练写入；评估保存自己的配置，不改训练配置。每个启动另存配置快照。processor 放在 `checkpoints/exports/processor/`：加载优先采用显式 `model.processor_path`，其次是导出权重旁的 processor，最后是原生 `model.qwen_path`；mask token ID 仍与导出严格校验。

每个 attempt 包含 config.yaml、run.json（启动参数、代码 SHA-256、环境/拓扑、恢复来源、允许变化的管理参数）、metrics.jsonl、checkpoints.jsonl 和 summary.json。指标以实际 optimizer step 为横轴，包含时间戳、统计窗口起止步数和更新次数；保存和正常结束前补写未满 log_every 的窗口。经 scripts/train.sh/test_train.sh 启动时，console.log 合并 launcher 和所有 rank 的 stdout/stderr，launch_result.json 保存真实退出码；直接 `python -m training.train` 仍有结构化日志，但控制台输出由调用者捕获。

每次恢复创建新的 attempt，原日志保留。日志不参与 checkpoint 清理。完整训练状态放在 `checkpoints/checkpoint-N/`；最终推理导出放在 `checkpoints/exports/weights.pt` 和可选 weights-ema.pt，不参与最近 N 个 checkpoint 的统计。

```bash
# 同实验自动恢复；全新实验目录可直接开始训练。
bash tests/validation/test_train.sh --auto-resume --set training.exp_id=slider_experiment

# 两条输出分支独立覆盖。
bash tests/validation/test_train.sh --auto-resume --set training.exp_id=slider_experiment \
  --set training.log_dir=/data/logs/slider_experiment \
  --set training.checkpoint_dir=/data/weights/slider_experiment

# 从旧实验 checkpoint 派生新实验。
bash tests/validation/test_train.sh --set training.exp_id=slider_branch \
  --resume outputs/slider_experiment/checkpoints/checkpoint-1
```

`--resume PATH` 与 `--auto-resume` 互斥。自动恢复按目录步数倒序检查完成标记、元数据、文件清单及大小，最新损坏则退回最近完整 checkpoint；不依赖可能过期的 latest.json，不跨实验搜索。已有训练历史却没有完整 checkpoint 时拒绝从初始化重新开始。明确指定的 checkpoint 无效或最新完整 checkpoint 训练契约不兼容时直接报错，不绕过参数校验。这里只自动选择恢复点，不负责崩溃后重新拉起进程。

训练持有实验根目录和 checkpoint 根目录的排他写锁。每次新 checkpoint 在临时目录写完、所有 rank 完成并校验后发布，再原子更新 latest.json，最后清理超出 keep_checkpoints 的旧完整 checkpoint。默认保留 3 个，正整数 N 至少为 1，null 表示全部保留。新保存失败时不删除旧 checkpoint。自动回退时，损坏 checkpoint/残留临时目录在所有 rank 恢复成功后移到 checkpoints/quarantine/，便于检查且避免重存同一步时路径冲突；隔离文件需人工清理。显式从较早 checkpoint 分叉时需使用新 checkpoint_dir，不能覆盖已有较新历史。

### 恢复校验

checkpoint 保存完整配置用于追溯，同时保存版本化的 resume_contract：

| 类别 | 参数 | 恢复行为 |
| --- | --- | --- |
| 管理 | exp_id、各输出目录、日志/保存/验证频率、保留数量、validation_sources、validation_max_batches、evaluation 的已登记管理字段、已有 sampling 字段 | 允许修改并记录差异 |
| 训练 | 模型结构/冻结范围、loss/CFG/flow、优化器、LR 调度、batch/累积、训练数据/表示/采样、种子、EMA | 必须一致，错误列出具体字段 |
| 运行状态 | world size、设备/精度、MCore 并行拓扑、通信设置及优化器 bucket 布局 | 新 checkpoint 严格校验，禁止精确恢复时改变拓扑 |
| 尚未分类 | 新增配置字段、workers/prefetch | 默认严格校验 |

max_train_steps 参与 WSD 学习率曲线，属于训练参数；`--stop-after-steps N` 用于在绝对第 N 次更新处停止，不改变训练预算。数据管线另校验各 rank 已消费的数据游标、数据清单及资产身份，预取未消费的 batch 不计入游标。验证流程保留训练 RNG 和独立验证游标，因而验证管理参数可调整。

先校验并恢复模型、optimizer、scheduler、EMA、各 rank RNG 和数据游标，再更新共享 config.yaml 与 exports/processor；恢复失败只留下本次 attempt 的错误记录，不覆盖旧运行元数据。旧 checkpoint 仍可通过 --resume 读取；自动发现旧布局时将 checkpoint_dir 指向旧目录，不搬移文件；若仍沿用旧 output_dir，需另外指定 log_dir，使日志不嵌套在旧 checkpoint 根目录内。旧 checkpoint 的配置会投影到同一分类契约，缺少 flow.repeats 时仅补历史默认值 1；旧文件没有记录的新 runtime_contract 无法追溯验证，run.json 会标注 legacy_runtime，此时仍检查原有 backend/world size/配置/数据约束。

文件完整性检查是清单和大小检查，不包含大权重文件的逐字节哈希。`weights.pt` / `weights-ema.pt` 是推理导出，不包含 optimizer/RNG/数据游标，不能用于训练状态恢复。

### 统一采样与评估

`eval` 由评测任务 YAML 驱动，只有 `sample` 和 `score` 两个动作。两者不是固定流水线：loss 可以直接 score，benchmark 也可以对已有生成结果单独 score。训练内 validation 继续调用 `validate_losses()`；独立 loss scorer 复用同一目标、模式、随机数及归约，不定义另一套 loss。

| 任务 YAML | 动作 | 输入与结果 |
| --- | --- | --- |
| `configs/evaluations/loss.yaml` | score | 监督数据 → 文本 CE、视频 flow loss、文本 PPL |
| `configs/evaluations/t2v_preview.yaml` | sample | prompt/描述清单 → MP4、最终 latent |
| `configs/evaluations/v2t_preview.yaml` | sample | 视频与指令 → 文本、可选 token IDs/logits |
| `configs/evaluations/vbench_subset.yaml` | sample/score | 固定 subset 协议 → 视频与外部评分结果 |
| `configs/evaluations/vbench_full.yaml` | sample/score | full 协议；评分环境需另行配置 |

任务 YAML 的共同结构是 `name`、`benchmark`、`actions`、`data`、`sample`、`score`。嵌套字段由对应适配器校验。多个 benchmark 可以在同一次 eval 中独立执行；只有输入、采样协议、checkpoint 和种子都兼容时才可复用结果。

扩展接口集中在 `evaluations/benchmarks/__init__.py` 的 `BENCHMARKS` 注册表：`Benchmark(sampler, scorer, score_requires_model)`。`sampler` 是请求规划函数 `planner(task, config, directory=None)`，返回带稳定 ID、task、seed 和输入信息的请求列表；实际 T2V/V2T 前向和产物保存仍由公共 `evaluations/sampling.py` 执行。`directory=None` 用于 dry-run，规划器不能写结果。`scorer` 是 `evaluations/scoring.py` 中 `SCORERS` 的注册名称；评分函数采用 `scorer(runtime, task, output_dir, *, distributed=False)`。`score_requires_model=True` 表示评分需要 UMM 监督前向（如 loss），为 false 时只消费已有采样产物或外部评分模型，不初始化 UMM/NPU。

当 benchmark 的 `sample` 有专用字段时，注册可选 `normalize_sample(task, config)`，将其转换为公共采样参数；原任务保留给 scorer，转换结果写入本次 plan。公共采样器只接受明确支持的参数，避免专用参数被悄悄忽略。

新增评测只需增加协议规划函数、注册 `Benchmark` 与 scorer，再编写任务 YAML，不修改调度器或 models。只有采样需求的任务可以不注册 scorer，只有 loss 等监督评分需求的任务可以不注册 sampler。适配器负责确认 sample 产物满足 score 协议，不能通过公共入口偷偷改变采样次数或参考答案可见性。

下面的 T2V 任务可以保存为 `/data/eval/t2v.yaml`：

```yaml
name: preview
benchmark: t2v
actions: [sample]
data:
  prompts:
    - A blue block moves right.
sample:
  samples_per_input: 1
  generation: {steps: 50, solver: euler, cfg: 5.0, time_shift: 5.0}
  video: {num_frames: 9, fps: 4.0, height: 432, width: 768}
  outputs: [mp4, latent]
  collect: {initial_noise: false}
score: {}
```

V2T 使用 `benchmark: v2t`、`data.sources: [{path: ./v2t.jsonl, task: v2t}]`、`outputs: [text]`，`generation` 可以覆盖 `max_new_tokens`、`temperature` 等。简单 caption 清单通过 source 的 task 选择视图；规范 JSONL 保留自身任务。没有参考回答的输入也能采样，loss 则必须有监督目标。

`sample.generation` 覆盖公共 `sampling`，`sample.video` 覆盖 `data.video`，不会改动其他任务配置。T2V 可以同时输出 MP4 和 latent，只进行一次采样；只要 latent 时不调用 VAE 解码。可选采集初始噪声、文本 token IDs、文本原始 logits；逐步视频 latent/轨迹暂不支持，不新增模型接口。`samples_per_input` 是结果数，`generation.steps` 是积分步数，两者与 loss 的 `flow.repeats` 相互独立。

loss 默认 `score.input: dataset`、`objective: training`，使用 `data.validation_sources`，也可以在任务的 `data.sources` 单独指定。`score.metrics` 可选 `text_ce`、`video_flow`、`text_ppl`；`seed`、`max_batches` 为 null 时继承运行配置；`evaluation.max_batches` 为 null 时继续继承原有 `training.validation_max_batches`。`max_batches` 限制的是跨所有验证流和 rank 的全局 batch 总数，不是每种任务各取 N 批；选定范围必须覆盖 schedule 中每种监督任务，否则指标归约会报错，不会以缺失任务的零值替代。汇总结果与有效 token/视频数保存到 `evaluations/<eval_id>/<任务名>/metrics.json`；默认开启的 `score.save_batches` 将逐 batch 诊断保存在 `logs/eval/<eval_id>/<attempt_id>/<任务名>/batches.rank-<N>.jsonl`，同目录保存本次 metrics 快照。任务目录的 metrics 是最近一次成功计算的结果，历史启动诊断互不覆盖。首期不支持 loss 的离线充分统计采集或 `sample.mode: supervised`。评估加载不检查训练总步数与 warmup/decay 预算，也不要求训练清单；共享模型、dtype、后端及数据格式仍严格校验。

```bash
# 执行同一 checkpoint 的多个任务。省略 --task-config 时默认 loss。
bash scripts/evaluate.sh --config /data/run.yaml --checkpoint /data/weights.pt \
  --task-config configs/evaluations/loss.yaml \
  --task-config /data/eval/t2v.yaml --eval-id step2000

# 只选择其中一个任务，或者只执行某个动作。
bash scripts/evaluate.sh --config /data/run.yaml --checkpoint /data/weights.pt \
  --task-config configs/evaluations/vbench_subset.yaml \
  --eval-id step2000_vbench --job vbench_subset --actions sample

# 检查执行计划，不加载模型或创建输出。
bash scripts/evaluate.sh --config /data/run.yaml --checkpoint /data/weights.pt \
  --task-config /data/eval/t2v.yaml --eval-id preview_check --dry-run

# 续跑已有计划；使用已有 result/config，新建日志 attempt。
bash scripts/evaluate.sh --run-dir outputs/umm_video/evaluations/step2000 --resume
```

也可在运行配置中写 `evaluation.tasks: [evaluations/loss.yaml, evaluations/t2v_preview.yaml]`。这些路径相对运行配置所在目录；任务内部 sources 等文件路径相对任务 YAML；CLI `--task-config` 相对当前目录解析。数据记录中视频和缓存路径仍遵循原数据契约。实际生效配置保存到评估目录，避免后续修改任务文件改变已有结果的含义。

`evaluation.num_processes` 为 null 时继承 `distributed.num_processes`（默认 4），可显式改为 1 等其他进程数。四卡采样采用每卡完整模型、按请求分片，不创建训练优化器。样本和重复编号确定 seed，不随 rank 改变；不使用会补齐重复项的 DistributedSampler。`--max-new-samples N` 可限制本次 attempt 跨全部任务和 rank 的新生成总数，用于验证中断与续跑，不修改结果身份。续跑检查 checkpoint、配置和输入身份，跳过完整产物；不完整或损坏产物重新生成。同一个 eval_id 不允许多个写入者同时运行。新增日志 attempt 不覆盖旧日志。loss 中断后重新计算该任务，不恢复到某个 batch。

采样输入与监督输入分开：T2V 不读取真实目标视频、ViT 特征或 Wan latent；V2T 去掉参考回答；评分器可以读取参考信息，但不将其加入生成条件。没有视频内部自回归生成，也不提供 KV/GDN cache；原生时间戳用于 content、文本监督和位置布局，不代表已实现按秒可控生成。

VBench 本阶段以稳定适配接口为重点，保留现有 prompt、seed、候选数量及官方命名协议。NPU 只负责生成；score 的 `python` 和 `vbench_root` 需指向独立 CUDA/VBench 环境及匹配的源码版本。仅评分通过 `--run-dir ... --actions score` 读取已提交产物，不加载 UMM/CANN，也不要求原权重和数据路径在评分机器上存在。此时允许调整评分解释器、源码目录和日志等管理参数；模型、数据、采样及其他计算协议覆盖会被拒绝，防止已有结果与配置不一致。缺少环境时明确报错，不把接口检查当作实际评分验证；尚未宣称官方评分链路已完成验证。旧 `evaluations/scripts/` 命令和 `scripts/generate.py` 只保留兼容用途，新任务统一使用 `scripts/evaluate.sh`。
