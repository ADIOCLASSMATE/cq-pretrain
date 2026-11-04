# OpenVid / OpenVidHD 全量统计

## 当前统计范围：OpenVidHD 子目录

本次仅统计 `/inspire/dataset/openvid-1m/v2/OpenVidHD` 中的全部视频，
caption 来自配套 `/inspire/dataset/openvid-1m/v2/data/train/OpenVidHD.csv`。
程序先逐包核对实际视频文件名、目录内 `OpenVidHD.json` 和配套 CSV，
再直接扫描该目录所有视频的元数据和完整帧时间表。
实际归档与 JSON 共 433,523 个视频，CSV 有 433,509 条标注；
14 个视频缺少 CSV 标注，仍计入全部视频统计。
caption 均值、分布的分母为有标注的 433,509 条，不把缺失标注计为零长度。
本次结果位于 `results/OpenVidHD/`，图标题标注 OpenVidHD。
`results/` 根目录中的 OpenVid-1M 结果属于之前的完整主数据集统计。

```bash
cd /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3/data/analysis
.venv/bin/python analyze_openvidhd.py all --workers 32
```

可以分别运行 `inventory`、`metadata`、`scan`、`aggregate`、`plots`、`validate`。
其中 `inventory.csv` 保存全部视频在 HD 归档中的实际位置，
`inventory_summary.json` 记录归档/JSON/CSV 三方对账，
`scope_validation.json` 核对最终结果是否仍对应相同的全部文件。
`missing_caption_videos.csv` 保存 14 个缺失标注的视频及其归档位置。
其他结果文件的名称和统计定义与下文一致。

## 主数据集的原有入口

只统计数据集事实，不修改或运行训练代码。默认输入为
`/inspire/dataset/openvid-1m/v2`，输出为本目录的 `results/`。
主数据集包含 OpenVidHD 子集；不会重复扫描子集归档。

## 运行

```bash
cd /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3/data/analysis
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r requirements.lock.txt
.venv/bin/python analyze_openvid.py all --workers 32
.venv/bin/python -m unittest discover -p 'test_*.py' -v
.venv/bin/python validate_results.py
```

可分别运行 `metadata`、`scan`、`aggregate`、`plots` 四个阶段。
视频扫描按压缩包保存结果；归档文件身份与解析器源码未变化时，
重新运行 `scan` 会跳过已完成的包。中断时正在处理的包会重新扫描。
遇到读取错误或 CSV/视频集合不一致，汇总会报错，不会把部分数据画成全量图。
程序直接读取普通 ZIP 和 aa/ab 分片，无需全量解压到磁盘。

## 统计口径

- 总体：所选视频目录中的全部视频，与配套 CSV 做一对一核对。
  OpenVidHD 中缺少标注的 14 个视频保留在视频统计中，caption 统计单列缺失。
- 帧数：主表 `frame_count` 为 MP4 完整 `stts` 帧数，有标注的视频与全量对应 CSV 对账；
  `encoded_sample_count` 为 `stsz` 编码样本数。两者不一致时分别保留，
  并对这些文件完整解码，保存实际解码帧数，不能把标注帧数当作无条件的解码事实。
- 时长：完整 `stts` 帧持续时间之和除以时间基；不是首末帧时间差。
  `mdhd` 声明的媒体轨道时长另存为 `media_duration_seconds`，保留两者不一致的记录。
- 平均 FPS：帧数除以上述 `stts` 总时长。所有原始 FPS 值都保存在频数表。
- 单视频内部帧率：用完整 `stts` 和 `ctts` 重建呈现时间戳，排序后比较间隔。
  `constant` 表示严格等间隔；`quantized_uniform` 表示不严格相等，但每个时间戳
  距端点拟合等间隔网格不超过一个时间基单位；`variable` 表示偏差超过该单位。
  最后一帧持续时间不被误算为相邻帧的时间间隔。B 帧重排序、起始空白编辑段均有处理。
  CTTS 偏移按照 FFmpeg/PyAV 的有符号 32 位规则读取，兼容 version 0 中的负偏移。
  实现依据：[FFmpeg mov_read_ctts](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/mov.c)。
  `invalid_timing` 单列两份时间表计数不一致的文件；它们仍计入尺寸、帧数、时长、caption
  等全量统计，但不能归入已确认恒定帧率的视频。
- 分辨率：视频样本描述中的实际编码宽×高。另检查旋转矩阵，旋转数量写入汇总。
  480p/1080p 明确按高度 480/1080 统计，同时单列精确 1920×1080 及短边口径。
  512×512 不归入 480p。完整尺寸列表另存 CSV，并全部绘入多页 PDF。
- Caption 字符数：Python `len(caption)`，包括空格和标点。
- Caption 英文词数：正则 `\b\w+(?:['’-]\w+)*\b`。
- Caption token 数：本地 Qwen3.5-2B `tokenizer.json`，
  `add_special_tokens=False`，不加聊天模板或视频 token。对全量 caption 执行。

这是完整元数据及帧时间表扫描，不宣称逐帧 RGB 图像均经过解码或图像质量审核。
PyAV 验证样本仅用来独立验证解析器；统计分母始终是全部视频。

## 文件

| 文件 | 内容 |
|---|---|
| `summary.json` / `numeric_summary.csv` | 全量统计、分位数、来源与定义 |
| `all_video_metrics.csv` | 全部视频逐条数据，包括尺寸、FPS、帧数、时长、caption 长度 |
| `caption_and_csv_metrics.csv` | 全量 CSV 和 caption 的逐条统计 |
| `*_distribution.csv` | 全部 FPS、尺寸、帧数、时间间隔类别的精确频数 |
| `*_bins.csv` | 图表实际使用的区间及完整计数 |
| `video_timing.{png,svg,pdf}` | 帧率、帧数、时长、视频内部帧间隔 |
| `resolution_distribution.{png,svg,pdf}` | 常见尺寸及 480p/1080p 比例 |
| `all_resolutions.pdf` | 每一种尺寸的多页图表，无省略 |
| `caption_lengths.{png,svg,pdf}` | 字符、词、token 的全量分布 |
| `video_shards/` | 每个归档的逐条检查结果和续跑状态 |
| `validation.json` | 全量对账及独立解析器验证 |
| `timing_anomalies.csv` / `timing_anomaly_decode.json` | 时间表不一致的视频及逐帧解码记录 |

初次执行时，解析器对少数带起始空白编辑段的视频保守报错。
`repair_scan_errors.py` 只重读这些记录，保留原扫描与修复解析器的身份及日志。
当前解析器已支持该结构，正常重跑无需修复步骤。
`refresh_timing.py` 记录初次统计时对全部非等间隔记录的 CTTS 兼容性复核。
