# Community Forensics 与 UniFD：固定模型与前向口径

本目录只运行检测器并输出分数；不据此删除原图、改写来源标签或覆盖既有来源过滤规则。`prob` 是 `sigmoid(logit)`，表示模型训练时的 fake 类分数，**没有经过我们 DCVLM 上的概率校准**。真实来源上的 AI 编辑是否保留仍由来源关系决定。

## 权重与代码

全部模型保存在 `corpus_pipeline/runs/detector_calibration_20261022/models/`，完整文件列表、下载地址、字节数与 SHA-256 见 `manifest.json`。下载器使用独立 `requests.Session(trust_env=False)`，不修改全局代理。HF 主站在本机直连超时，因此本次通过 `hf-mirror.com` 直连获取**作者仓库的固定 revision**，并校验其发布的 LFS SHA-256；未使用第三方微调模型。

| 名称 | 作者仓库／代码版本 | 权重 SHA-256 |
|---|---|---|
| CF224 | [OwensLab/commfor-model-224](https://huggingface.co/OwensLab/commfor-model-224/tree/26afc31e6b40c312c3fd42c05a758be62446215b) @ `26afc31e6b40c312c3fd42c05a758be62446215b` | `a6cc439d5a6d2dfadd60c77d27a2838ad55b34e601ecd30f46ad97266d6ac4e0` |
| CF384 | [OwensLab/commfor-model-384](https://huggingface.co/OwensLab/commfor-model-384/tree/6076002bf0d9dd37537f965ee2f06f826c333b61) @ `6076002bf0d9dd37537f965ee2f06f826c333b61` | `b89f36275f3bf5e2b040eee36597a8f19db051bff9a473a9cf7b2466284fb387` |
| UniFD 的 CLIP 视觉骨干 | [openai/clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14/tree/32bd64288804d66eefd0ccbe215aa642df71cc41) @ `32bd64288804d66eefd0ccbe215aa642df71cc41` | `a2bf730a0c7debf160f7a6b50b3aaf3703e7e88ac73de7a314903141db026dcb` |
| UniFD 的检测分类头 | [作者 fc_weights.pth](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/pretrained_weights/fc_weights.pth) @ `030495aea3300a8b54c0ec37ec7fe1dd7e63c619` | `477100745713bcc957beb2b40859536859b6483fd6301b3b9293151b194c7847` |

CF 官方代码固定为 [ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4](https://github.com/JeongsooP/Community-Forensics/tree/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4)，UniFD 官方代码固定为 [030495aea3300a8b54c0ec37ec7fe1dd7e63c619](https://github.com/WisconsinAIVision/UniversalFakeDetect/tree/030495aea3300a8b54c0ec37ec7fe1dd7e63c619)。相关源码仅保留用于核对与复现；推理不用 Hub 的远程自定义代码。

CF 在本机 `timm` 中创建与作者相同的 `vit_small_patch16_{224|384}.augreg_in21k_ft_in1k`、单 logit head，`pretrained=False` 后严格加载完整检测权重。224 版 21,666,049 参数，384 版 21,811,969 参数；实际参数数以运行 `modelinfo` 为准。

UniFD 使用 OpenAI ViT-L/14 视觉骨干和作者的真实检测分类头，不使用通用 CLIP 相似度替代检测器。安全张量中的 393 个视觉键（包含 24 层）全部映射为 OpenAI 原结构的 296 个键，Q/K/V 按此顺序拼接，视觉投影转置；严格检查键、shape、24 层／1024 维／16 头／14 patch／QuickGELU。199 个文本参数键明确排除，因为 `encode_image` 不使用文本塔。输出 768 维视觉投影**不做 L2 归一化**，直接接作者 `Linear(768,1)`；总计 303,966,977 参数。骨干以 safetensors 读取，检测头以 `torch.load(..., weights_only=True)` 读取，全部 `strict=True`，每次运行验证固定 SHA-256。

## 与官方评估一致的预处理

| 项目 | CF224 | CF384 | UniFD |
|---|---|---|---|
| 输入 | PIL RGB | PIL RGB | PIL RGB |
| 缩放 | 短边 256，BILINEAR | 短边 440，BILINEAR | **不缩放** |
| 裁剪 | 中心 224×224 | 中心 384×384 | 中心 224×224 |
| 转换 | ToTensor，0–1 | ToTensor，0–1 | ToTensor，0–1 |
| 均值 | `[.485,.456,.406]` | 同左 | `[.48145466,.4578275,.40821073]` |
| 标准差 | `[.229,.224,.225]` | 同左 | `[.26862954,.26130258,.27577711]` |
| EXIF 自动转正 | 无 | 无 | 无 |
| fake 类分数 | sigmoid(logit)，1=fake | 同左 | 同左 |

依据：CF [dataloader.py 中 `determine_resize_crop_sizes/get_transform`](https://github.com/JeongsooP/Community-Forensics/blob/ee5b71d43db0f3779e1edd64ee927b13f2dd6ad4/dataloader.py)；UniFD [validate.py 中 `RealFakeDataset`](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/validate.py) 与 [clip_models.py](https://github.com/WisconsinAIVision/UniversalFakeDetect/blob/030495aea3300a8b54c0ec37ec7fe1dd7e63c619/models/clip_models.py)。UniFD 的 `clip.load` 虽返回默认 CLIP resize 预处理，但官方检测评估并未使用它，本实现也未替换为该预处理。

这意味着 UniFD 对高分辨率图像只观察中心原生像素 224×224 的区域；它不能代表整张文档或编辑图的所有局部。小于 224 的图片按 torchvision CenterCrop 默认补零。CF 同样只看缩放后的中心区域。没有增加多裁剪、JPEG 重编码或增强测试，以免混淆官方基线与本次阈值比较。

## 验证与本机运行

本机使用 CPU FP32、8 个线程、batch=8，`TORCH_DEVICE_BACKEND_AUTOLOAD=0` 避免无 Ascend runtime 的 CPU 环境自动加载 torch_npu。执行库版本见 `inference_run.json`，本次为 torch 2.6.0+cpu、torchvision 0.21.0+cpu、timm 1.0.26。推理入口支持显式 CUDA／NPU 设备，但本次没有用它们，FP16/BF16 也没有校准或等价性保证。

`unifd_architecture_equivalence.json` 保存本地适配器与已阅读、固定版本的官方 `VisionTransformer` 在相同权重下的数值对照：2 个固定随机输入，FP32 视觉特征和检测头 logit 最大绝对误差均为 **0.0**。此测试验证架构实现和前向一致；不把它描述为本地 HF 文件与另外下载的 Azure TorchScript 检查点逐字节比较。骨干真实性由作者仓库 revision 与 LFS 内容哈希固定。

16 张真实样本预跑（含解码，不含模型加载）：CF224 0.63 秒、CF384 1.08 秒、UniFD 4.03 秒；三个模型均 0 失败。预跑只用于决定 CPU 足够，完整运行的吞吐应从最终 `timing.*.json` 读取。

```bash
.venv/bin/python corpus_pipeline/detector_calibration/prepare_models.py \
  --output corpus_pipeline/runs/detector_calibration_20261022/models \
  --hf-endpoint https://hf-mirror.com

TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv/bin/python \
  corpus_pipeline/detector_calibration/infer.py \
  --samples corpus_pipeline/runs/detector_calibration_20261022/samples.jsonl \
  --models-dir corpus_pipeline/runs/detector_calibration_20261022/models \
  --output corpus_pipeline/runs/detector_calibration_20261022/full \
  --models cf224 cf384 unifd --device cpu --threads 8 --batch-size 8
```

每次输入解码前检查图片 SHA-256 与抽样清单一致。每模型写 `scores.<model>.jsonl`，每批 flush/fsync，按 `image_sha256` 断点跳过已完成结果。代码、样本、设备、精度、batch 等签名不一致会拒绝复用输出目录；改变设置时应创建新目录。可用 `--retry-errors` 重试错误记录，最终合并使用同一图片的最后记录。

每模型记录包含 `model/image_sha256/image_path/status/logit/prob/timing_s`。最终 `scores.jsonl` 用图片 SHA 汇合到 `scores:{cf224:{...},cf384:{...},unifd:{...}}`。失败有 `stage/error`，不会伪造成功分数；`COMPLETE.json.success` 只有全部指定图片和模型成功才为 true。`modelinfo.*.json` 和 `timing.*.json` 保存模型、预处理和本轮增量耗时。
