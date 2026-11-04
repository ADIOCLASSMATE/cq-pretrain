# OpenVidHD 双分辨率离线预处理（v3）

本目录保存筛选、解码、隔帧、缩放、编码、校验和续跑实现。训练代码不变；
不截断到 81 帧，不产生 Qwen/ViT 或 VAE latent。

## 数据范围和处理规则

仅处理 /inspire/dataset/openvid-1m/v2/OpenVidHD。关联原始 OpenVidHD.csv、
OpenVidHD.json 和 data/analysis/results/OpenVidHD 中的全量普查结果。
选择全部原始帧率中分辨率恰为 1920×1080、至少 161 帧、caption.strip() 非空的样本。
strip 仅用于筛选，保存的 caption 不变。当前普查对应 215,442 个候选样本、
98 个归档组、207 种精确源帧率；代码重新计算这些数量，不将它们作为筛选条件。

每个原视频仅解码一次，在显示顺序中保留第 0、2、4、… 帧直到结尾，输出帧数为
(N+1)//2。每个选中帧仅做一次源 RGB 转换，再独立生成两种尺寸：

| 版本 | 覆盖目标尺寸的缩放 | 中心裁剪 |
|---|---|---|
| 960x544 | 1920×1080 → 967×544 | x=3, y=0，960×544（左3、右4） |
| 832x480 | 1920×1080 → 853×480 | x=10, y=0，832×480（左10、右11） |

整数尺寸存在至多半像素的缩放取整。两个版本直接来自原帧，不串行缩放。使用
CPU torch float32 bicubic、antialias=True、align_corners=False，随后
clamp、round、uint8。两个尺寸的宽高均为 32 的倍数，因此也满足 16 的倍数。

保存视频轨单独的 MP4：原生 MPEG-4 Part 2（mpeg4 / mp4v），yuv420p，q=2，
单编码线程。PyAV 使用 QSCALE 和 global_quality=2*118。这是有损编码，
RGB→YUV420 的色度采样和颜色转换误差也包括在质量评估中。

精确输出帧率用 Fraction(source_fps)/2 表达。若约分后为 p/q，MP4 时间基为
1/p，第 j 帧 PTS 为 j*q。逐帧验证：

    output_time(j) = source_time(2*j) - source_time(0)

不进行小数舍入、补帧、时间插值或变速。奇数源帧数的最后一个保留帧在 MP4 中
显示一个输出间隔，因此容器结束时刻比原视频多一个源间隔；首末帧时间戳仍精确对应。
duration_seconds 和 pts_span_seconds 分别记录显示时长和首末帧跨度。

## 输出目录和样本标识

正式根目录：

    /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected

10 样本验证根目录为上述路径加 _pilot10。

    metadata/OpenVidHD.csv              原始完整 CSV，逐字节复制
    metadata/OpenVidHD.json             原始完整分组索引，逐字节复制
    metadata/selected.jsonl             筛选结果、源位置、原始字段，无哈希
    metadata/source_integrity.jsonl     辅助校验：源 ZIP CRC
    metadata/run_integrity.json         辅助校验：代码、配置、输入及验证文件
    metadata/integrity_parts/partN.jsonl 辅助校验：各版本逐条提交记录
    metadata/integrity.jsonl            辅助校验：汇总后的有效输出记录
    960x544/videos/partN/<原文件名>.mp4
    960x544/samples.jsonl
    832x480/videos/partN/<原文件名>.mp4
    832x480/samples.jsonl
    pairs.jsonl                        以原始 sample_id 连接两个版本
    processing_config.json             可读配置、版本、运行环境，无哈希
    _state/partN.jsonl                  可读处理日志，无哈希
    _state/partN.status.json            各归档组实时进度
    _scratch/partN/                     当前源文件的临时解压目录
    logs/<UTC启动时间>.log
    logs/exit_code                     batch.sh 最终退出码
    progress.json                      最多每30秒更新一次的全局进度
    run.json                           running / succeeded / incomplete
    summary.json
    failures.jsonl
    commit_validation.json             每个提交前已完成输出全解码校验
    validation.json                    显式 verify 的验证报告

sample_id 和 video 均为原 CSV 的完整 video 字段，含扩展名和全部片段后缀；
文件名不改。原后缀用于追溯身份，不重新解释为输出帧数。source 中保留归档路径、
ZIP member、part、源尺寸、帧数、精确 FPS、原始首帧 PTS 和时间基；
source.annotation 保留全部 9 个原始 CSV 字段。caption 不改写、不截短、不自动重标注。

每个版本的 samples.jsonl 位于该版本目录，video_path 相对其所在目录，例如
videos/part1/<原文件名>.mp4。pairs.jsonl 的两个路径相对数据集根目录。
两份公开 manifest 仅包含双方均成功的样本，ID 和顺序完全一致；单侧成功仍保存
视频和处理日志，续跑时只补另一侧。成功文件数可能因此大于完整配对数的两倍。

业务 manifest、配置、进度、报告都不保存 SHA、CRC 或 fingerprint。
所有校验值仅写入 metadata/ 下明确标记为 integrity 或 resume_baseline 的辅助文件。
JSONL 是本项目的适配格式，不是 OpenVid 官方原始标注格式。UMM_v3 的 read_records
可以读取它并解析路径；实际训练的时长、帧率和模型分辨率仍由训练配置控制。

## 实现与提交顺序

- archive.py：可 seek 的分卷 ZIP 读取、单成员解压、CRC 和源文件校验。
- video.py：一次源解码、两条独立空间分支、精确 PTS、MPEG-4 编码和完整回读。
- manifest.py：原子 JSON/JSONL、fsync 日志、尾部恢复、独占运行锁、辅助校验。
- openvidhd.py：全量筛选、按归档组进程池调度、配对汇总、验证与 CLI。
- pilot_report.py：已通过独立审计的 10 样本质量和大小汇总。
- validate_dual_pilot.py：重新执行已完成 pilot，验证 20 文件全部跳过且内容、
  mtime、处理日志和 manifest 不变。

每个版本编码后先完整解码，检查 codec、像素格式、每帧尺寸、帧数、每个 PTS、
平均 FPS 和 stream duration，再 fsync 视频、原子 rename、fsync 目录、提交
辅助校验日志，最后提交可读成功日志。恢复依据为 (sample_id, variant)。
跳过已成功文件前再次检查其文件内容。缺失、损坏或没有完整提交记录的文件重建；
一个版本失败不撤销另一个已经验证成功的版本。

代码、输入和运行环境在 select 时冻结。全量启动要求同版本的 10 源样本 /
20 输出文件独立审计通过，并检查报告和实际视频未被更改。运行中不要修改四个
核心实现文件或更新环境。worker 数可变。旧单机入口在本机使用 flock；当前共享盘
实测不支持用这把锁做跨主机互斥，多实例必须使用下文的 distributed.py。

## 测试、pilot 和全量运行

使用已有 .venv-npu，关闭 torch NPU 自动加载，全部在 CPU 执行。run.sh 固定
OMP/MKL/OpenBLAS 单线程。当前 cgroup 配额为 32 个 CPU 核，正式使用 16 workers。

在 UMM_v3 根目录执行：

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
      .venv-npu/bin/python -m unittest discover -s data/preprocess/tests -v

14 项测试覆盖选样边界、完整 ID/caption、分卷 ZIP 与 CRC 损坏、损坏日志尾部、
奇偶长度、非零起始 PTS、B 帧显示顺序、编号帧、双分辨率裁剪、单分支编码失败、
仅修复缺失/损坏版本、孤立文件、运行锁和 batch 退出码。207 种有理数 FPS 均在
两个真实尺寸上执行了短片编码与时间戳校验。日志中的 injected variant failure
是故意注入的恢复用例。

pilot 确定性覆盖 5 种常见 FPS、161/162 帧边界、最长源视频、最低源 FPS，以及
分卷 ZIP 中的少见 FPS。独立第二次读取全部原帧，逐一核对每个输出的源帧 2*j、
对应时间和编码前 RGB，统计全帧 RGB PSNR/MAE，检查 caption 和全部原标注字段。
另外使用 ffmpeg 检查全部 20 个输出首帧宏块的实际 q=2，并通过 UMM read_records
验证两份 manifest 的路径能被解析；这些检查不等价于执行模型训练。

    tmux new-session -d -s openvidhd-dual-pilot -c /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3 \
      'bash data/preprocess/batch.sh pilot /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_pilot10 4'
    tmux set-window-option -t openvidhd-dual-pilot remain-on-exit on

    TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv-npu/bin/python -m data.preprocess.pilot_report \
      --output /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_pilot10
    TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv-npu/bin/python -m data.preprocess.validate_dual_pilot \
      --output /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_pilot10

确认 validation.json 和 resume_validation.json 均通过后启动：

    tmux new-session -d -s openvidhd-dual -c /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3 \
      'bash data/preprocess/batch.sh full /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected 16 /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_pilot10'
    tmux set-window-option -t openvidhd-dual remain-on-exit on
    tmux attach -t openvidhd-dual

Ctrl-b d 分离终端。tmux 可防止终端/SSH 断连结束任务，主机或容器退出后仍需手动续跑。
未切换到分布式模式时，重复相同 batch.sh 命令即可续跑。创建输出根目录的 STOP
文件会在当前样本完成后优雅停止；单机续跑前移除 STOP。分布式模式须一直保留
根目录 STOP，并使用下文入口。剩余空间低于 100 GiB 时保留待处理记录并停止 worker。

    bash data/preprocess/run.sh finalize --output /absolute/dataset/root
    bash data/preprocess/run.sh verify --output /absolute/dataset/root
    bash data/preprocess/run.sh verify --output /absolute/dataset/root --source-correspondence

正常全量结束时自动汇总 manifest；途中 finalize 须先停止处理并取得同一把锁。
全量每个文件已经回读，通常无需再额外全量 verify。processing_finished 仅表示
没有待尝试版本；all_candidates_succeeded 还要求所有候选都配对成功。不要将进程启动
视为全量完成。10 样本是边界覆盖集合，按其压缩率推算全量空间仅用于示意。

## 多 CPU 实例调度

`distributed.py` 复用冻结的 `part_work`、`transcode` 和 `reconcile`；四个核心文件
不修改。只有调度、批次日志和最终汇总层发生变化，不重新选样或改变像素处理。

切换时先创建根目录 STOP，等待旧进程退出。`prepare` 检查原配置和通过的 pilot，
复制旧日志到 `_distributed/baseline/`，记录每个成功 MP4 的路径、大小和修改时间。
已经完成两个版本的样本从任务中排除；只完成一个版本时，把该成功记录带入批次，
原转换函数会校验并跳过它。备份旧日志和原视频一直保留。

待处理样本在 ZIP 内按物理顺序分批，上限 32000 原始帧或 128 视频。每个样本
只属于一批；16 个四视频小批次用于真实数据验证，其成品直接留在最终数据集。

目录约定：

```text
_distributed/
  plan.json, tasks.jsonl           # 可读的固定任务清单
  plan_integrity.json             # 独立校验信息
  task_integrity.jsonl
  baseline/                       # 原完成日志备份和文件状态快照
  batches/batch_000000/
    metadata/selected.jsonl       # 本批原始样本及完整 caption
    metadata/source_integrity.jsonl
    metadata/integrity_parts/     # 本批输出校验日志
    _state/                      # 本批成功/失败日志
    _scratch/                    # 一次仅解压正在处理的源视频
    960x544 -> 原数据集/960x544
    832x480 -> 原数据集/832x480
    STOP -> _distributed/STOP
  claims/batch_000000/            # 原子 mkdir 领取，不使用 exist_ok
    owner.json                   # 实例、主机、PID、进程启动时间
    done.json                    # 成功后保留，阻止再次领取
  attempts/                      # 确认原进程死亡后归档的领取记录
  sessions/, logs/               # 各实例会话、日志和退出码
  validation/report.json         # 两机验证通过后才能运行 full
  progress.json                  # 分布式实时进度
  finalized/                     # 汇总视图，原批次日志仍保留
  finished.json                  # 全量覆盖和最终发布通过
```

各实例用 `distributed_tmux.sh` 启动。后台 session 名为 `openvidhd-full`；本机另有
`openvidhd-monitor`，每分钟更新进度，每五分钟检查已退出的处理进程。恢复前在
原主机检查 PID 和启动时钟，确认原执行者已退出才回收领取；网络不可达或心跳
超时不会触发抢占。容器被平台停止/替换时需先确认平台终态，再人工恢复其任务。
普通进程失败最多自动重试三次，达到上限的批次保留错误供检查。

手动查询和恢复（在 UMM_v3 根目录执行，output 为数据集绝对路径）：

```bash
bash data/preprocess/distributed.sh progress --output "$dataset_root"
# recover 必须在该批原进程所在主机运行，活进程和已完成批次会被拒绝。
bash data/preprocess/distributed.sh recover --output "$dataset_root" --batch batch_000000
# 恢复后启动正常 worker，只会领取未完成且未被占用的批次。
bash data/preprocess/distributed_tmux.sh "$dataset_root" openvidhd-cpu-20261030-01 full 16
```

暂停分布式处理应创建 `_distributed/STOP`，等待各 worker 完成当前样本后退出；
恢复时移除该文件并回收已经退出的领取，再启动 worker。不要删除根目录 STOP。

两机验证包含同时运行、跨实例互斥、真实 SIGKILL 中断、跨实例接续、已有成功文件
大小/修改时间不变、成功日志不重复、全帧解码及时间戳检查。`fault-test/fault-watch`
只允许验证阶段使用，正式全量开始后不可执行。

监控器仅管理本次创建并登记的 8 台 CPU 实例；原工作实例继续使用 16 workers。
全部批次完成后，单个汇总进程检查旧文件未改变和样本覆盖，在独立视图合并日志，
再原子发布两份 samples.jsonl、pairs.jsonl、summary.json 和辅助 integrity 文件。
最后停止本次新增的 CPU 实例，不停止原工作实例。最终 MP4 路径和文件名不变。

## 历史版本

旧无损 RGB 540p 和单尺寸 MPEG-4 540p 数据目录均保留并暂停，STOP 文件不移除。
对应代码分别保存在 versions/lossless_rgb_v1/ 和 versions/mpeg4_960x540_v2/。
本次从原始源视频生成，不使用旧 540p 成品作为缩放输入。

compare_codecs.py、compare_codecs.sh、export_codec_frames.py、
plot_codec_comparison.py 和 validate_codec_comparison.py 属于既有 540p 编码对比实验；
结果在 data/analysis/results/OpenVidHD/codec_comparison_10/。新分辨率的像素已经改变，
不能要求新视频与旧 540p 基准逐字节一致；新版用各自缩放裁剪后的原始 RGB 做质量参照。
