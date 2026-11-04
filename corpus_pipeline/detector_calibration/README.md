# DCVLM 图像检测器抽样与阈值诊断

本工具在原版 DCVLM 的真实图像字节上运行 Community Forensics 224／384 和 UniFD，保存逐图分数、来源信息和供人工观察的图片页。它用于比较检测器在照片、图表、文档、渲染／截图、混合网页等来源中的行为，**不自动删除数据**，不改写原版 DCVLM，也不生成 T2I50。

当前实验目录为 `corpus_pipeline/runs/detector_calibration_20261022/`。它包含来自 132 个视觉来源的 **2,396 张唯一图片**，按来源和领域分层抽样，评分前固定为 calibration 1,943 张、holdout 453 张；并非全库扫描或可据此推断全库 AI 图比例的随机样本。实际数量、输入哈希和抽样方法以 `sampling_summary.json` 为准。

## 解释边界

来源注册表提供来源层面的证据，**不是逐图真实性真值**。当前样本没有经过确认的 AI 生成正例集合，也没有经过确认的“真实原图→AI 编辑结果”配对标签；混合与未知来源中仍可能包含这两类图片。因此本次不能给出可信的召回率、精确率、AUC，或能保证保留真实来源编辑图的最终自动删除阈值。

`prob = sigmoid(logit)` 是检测器的 fake 类分数，**不是在 DCVLM 上校准好的真实性概率**。高分图片仍需结合原始来源、图像内容和编辑关系核实；图表、文档、非 AI 渲染图与真实来源的 AI 编辑图属于用户允许保留的数据。图片页上的视觉观察也不是确证生成过程的依据。

`analyze.py` 的阈值候选控制的是“有来源支持的保留类样本中，有多少超过分数线”，不是已证明的真实误报率。分层配额、来源聚集和缺少逐图标签使其区间不能外推为全库保证。先看 calibration 图片选择复核规则，再冻结规则检查 holdout；不要看完 holdout 后继续调参并称之为独立验证。

## 从仓库根目录复现

以下所有相对路径均基于仓库根目录 `ascend-pretrain/`。使用已有 `.venv`，依赖包括 torch、torchvision、timm、safetensors、Pillow、numpy、pyarrow 和 requests。当前执行环境与具体库版本自动保存在 `inference_run.json`。下列命令创建一个新实验目录，不覆盖已锁定的当前样本：

```bash
CALIBRATION_RUN=corpus_pipeline/runs/detector_calibration_repro
```

### 1. 抽取原版 DCVLM 的图像字节

```bash
.venv/bin/python corpus_pipeline/detector_calibration/sample_dcvlm.py \
  --root /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/dcvlm_pretrain_v1_20261016T062238Z \
  --sources corpus_pipeline/config/sources.json \
  --output "$CALIBRATION_RUN" \
  --seed 20261022 --target 2400 --shards-per-source 3
```

抽样器验证原版 DCVLM 的 COMPLETE/catalog，拒绝 T2I50。它在每个来源中按 shard 行数进行不放回抽样，再随机选记录，每条记录随机取一张图，按原始 TAR 中的 offset/length 读取并验证 SHA-256。输出保存原始字节，避免重新编码影响检测。去重后数量可能低于目标数。原始 TAR 池必须在当前执行环境可访问；本工具不会假定 CPU 与 Ascend 分区共享存储，也不会自动访问图片 URL。

`samples.jsonl` 创建后锁定；需要修改 seed、样本数、来源配置时使用新目录。按图片字节 SHA 分配 calibration/holdout，保证同一字节内容不会同时进入两侧。不同编码或近重复图片尚未做视觉去重，分割不等于内容级去重。

### 2. 下载固定作者模型

```bash
.venv/bin/python corpus_pipeline/detector_calibration/prepare_models.py \
  --output "$CALIBRATION_RUN/models" \
  --hf-endpoint https://hf-mirror.com
```

本次 HF 主站直连超时，采用直连镜像获取**作者仓库固定 revision**，用发布的 LFS SHA-256 验证内容；不是第三方微调模型。若所在环境能直连主站，可将 endpoint 改为 `https://huggingface.co`。下载器的独立 HTTP 会话不读取代理环境变量，也不修改全局代理配置。

官方来源已核对：

- Community Forensics [作者代码](https://github.com/JeongsooP/Community-Forensics)、[224 检查点](https://huggingface.co/OwensLab/commfor-model-224)、[384 检查点](https://huggingface.co/OwensLab/commfor-model-384)。
- UniFD [作者代码](https://github.com/WisconsinAIVision/UniversalFakeDetect)、[固定版本的检测分类头](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/pretrained_weights/fc_weights.pth)。
- UniFD 的视觉骨干来自 [OpenAI CLIP ViT-L/14](https://huggingface.co/openai/clip-vit-large-patch14)。本实现加载作者检测头，不把通用 CLIP 相似度当检测分数。

固定 revision、SHA-256、参数映射与加载方式详见本目录 `MODELS.md`，实际下载清单见实验目录 `models/manifest.json`。不使用 Hub 远程自定义代码；骨干读取 safetensors，UniFD head 使用 `weights_only=True`，全部严格校验键与 shape。

### 3. 执行三个模型的 CPU FP32 推理

```bash
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python \
  corpus_pipeline/detector_calibration/infer.py \
  --samples "$CALIBRATION_RUN/samples.jsonl" \
  --models-dir "$CALIBRATION_RUN/models" \
  --output "$CALIBRATION_RUN/full_cpu_fp32" \
  --models cf224 cf384 unifd \
  --device cpu --dtype float32 --threads 8 --batch-size 8
```

`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 避免 CPU 机器自动加载缺少 Ascend runtime 的 torch_npu；只作用于该命令。每张图解码前再次校验 SHA-256。逐模型逐批保存 `scores.<model>.jsonl`，相同命令可断点继续；失败保留 `stage/error`，可显式 `--retry-errors` 重试。代码、选取样本、设备、精度或 batch 改变会拒绝复用原输出目录。

只有指定样本和三个模型全部成功，`COMPLETE.json.success` 才为 true。`scores.jsonl` 按图片 SHA 汇总三模型的原始 logit、fake 分数与状态；`modelinfo.*.json` 保存权重与预处理；`timing.*.json` 保存该次执行新增部分的耗时。断点恢复后的耗时不是此前所有进程的累计运行时间。

### 4. 汇总阈值与生成待看图片页

```bash
.venv/bin/python corpus_pipeline/detector_calibration/analyze.py \
  --samples "$CALIBRATION_RUN/samples.jsonl" \
  --scores "$CALIBRATION_RUN/full_cpu_fp32/scores.jsonl" \
  --output "$CALIBRATION_RUN/review_calibration" \
  --review-split calibration
```

`analysis.json` 包含各来源领域、来源标签、分割、输入尺寸的分数分布，固定阈值超线数量，模型之间的分歧，以及来源支持保留类的负对照候选。`review_candidates.jsonl` 选择高分、阈值附近、模型分歧、各领域随机与低分对照图片；`contact_sheets/` 图片页只显示 review ID，避免先看分数和来源标签影响视觉判断。`review_pages.json` 提供图片页清单。程序生成候选后仍必须实际查看图片，并记录观察和不确定性。

冻结拟采用的复核规则后，再单独生成 holdout 图片页：

```bash
.venv/bin/python corpus_pipeline/detector_calibration/analyze.py \
  --samples "$CALIBRATION_RUN/samples.jsonl" \
  --scores "$CALIBRATION_RUN/full_cpu_fp32/scores.jsonl" \
  --output "$CALIBRATION_RUN/review_holdout" \
  --review-split holdout
```

`--review-split` 控制图片候选与图片页，汇总仍包含两部分固定阈值统计。不要把在 holdout 上重新挑选出的阈值报告为未参与选择的验证结果。本工具不会把候选阈值写进训练数据删除规则。

### 5. 评估冻结后的送审规则

本次实测的最终规则见 `thresholds.json`，结论见 [REPORT.md](REPORT.md)。它只定义送审工作点，不是逐图真实性概率校准，也不是自动删除规则。换模型、预处理或数据领域后应重新验证。

```bash
.venv/bin/python -m corpus_pipeline.detector_calibration.evaluate_policy \
  --samples "$CALIBRATION_RUN/samples.jsonl" \
  --scores "$CALIBRATION_RUN/full_cpu_fp32/scores.jsonl" \
  --thresholds corpus_pipeline/detector_calibration/thresholds.json \
  --output "$CALIBRATION_RUN/policy_evaluation" --split all
```

输出逐图 `review_index.jsonl` 和按来源／领域／来源类型×split 汇总的 `policy_evaluation.json`。CF384 或 UniFD 超线触发送审，同时超线优先；CF224 不投票。程序绘图、渲染来源和明确真实源 AI 编辑转为非阻断的来源审计。缺失／错误／非有限分数保留为 `inference_error`，未超线仅为 `no_model_alert`，都不会自动生成训练准入决定。

对照模型实际看到的空间区域可使用 `crop_review.py --review <review_candidates.jsonl> --ids R001 R002 --output <新目录>`；加 `TORCH_DEVICE_BACKEND_AUTOLOAD=0` 以在本机 CPU 环境加载 torchvision。此工具仅生成审查拼图，不修改原图。

## 预处理与已验证的等价范围

| 模型 | 官方评估预处理 | 对图片覆盖的限制 |
|---|---|---|
| CF224 | RGB；短边 Resize256，BILINEAR；中心 Crop224；ImageNet normalize | 缩放后中心区域 |
| CF384 | RGB；短边 Resize440，BILINEAR；中心 Crop384；ImageNet normalize | 缩放后中心区域 |
| UniFD | RGB；**不 resize**；中心 Crop224；CLIP normalize | 原始分辨率中心 224×224，可能遗漏边缘编辑或文档内容 |

实现依据 [CF 官方 dataloader](https://github.com/JeongsooP/Community-Forensics/blob/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4/dataloader.py) 与 [UniFD 官方 validate.py](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/validate.py)。UniFD 没有使用 `clip.load` 返回的通用 CLIP resize 预处理，视觉投影也不做 L2 normalize。三个模型均保持官方 RGB 读取口径，不增加 EXIF 转正、JPEG 重编码或多裁剪；UniFD 遇到小图时由 CenterCrop 补零。

本地 CLIP 视觉实现已与固定版本、已阅读的官方 UniFD `VisionTransformer` 做 FP32 对照：相同映射权重、2 个固定随机输入，视觉特征和检测头 logit 的最大绝对误差均为 **0.0**。结果在当前实验目录的 `unifd_architecture_equivalence.json`。这验证所测架构与前向实现一致，不代表针对所有输入、其他硬件或 FP16/BF16 的数值证明，也没有另外下载 Azure TorchScript 权重进行逐字节比较。作者权重由固定仓库 revision 与 LFS 哈希核实。
