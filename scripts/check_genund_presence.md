# 图文存在性抽样检查

`check_genund_presence.py` 默认检查 `/inspire/dataset/gen-und-mm-pretrain/01`
中的每个一级数据集目录，各输出一行。只检查存在性，不解码图片、不访问 URL、
不解包到磁盘、不统计全量样本数、不校验图文逐条配对。

## 运行

在已经挂载数据集的 CPU Notebook/Job 中运行，Python 3.10+：

```bash
# 检查 Parquet 时需要；版本与项目 requirements-data.lock 一致。
python -m pip install pyarrow==21.0.0

python /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/scripts/check_genund_presence.py \
  /inspire/dataset/gen-und-mm-pretrain/01 \
  --sample-shards 3 \
  --output /tmp/genund_presence.json
```

输出终端表格、`/tmp/genund_presence.json` 和 `/tmp/genund_presence.csv`。
JSON 保留图片、标注的证据路径、检查过的容器、读取上限和错误原因。
CSV 使用 UTF-8 BOM，便于表格软件打开。输出必须位于数据根目录之外，
文件已存在时会拒绝覆盖；再次运行请使用新的输出文件名。

只查一个或几个嵌套子集：

```bash
python /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/scripts/check_genund_presence.py \
  /inspire/dataset/gen-und-mm-pretrain/01 \
  --dataset openuni/data/cc12m \
  --dataset BLIP3o-Pretrain-Long-Caption \
  --output /tmp/genund_selected.json
```

传 `--dataset .` 可把 ROOT 自身作为一个数据集检查。
选择嵌套子集时，检查范围是该子树，不会把父目录共享标注自动归属给子集。
默认一级汇总也不表示其每一个子集都同时含图文。

## 判定含义

| 列/结果 | 含义 |
|---|---|
| `image=YES` | 发现图片文件、归档中的图片成员、Parquet 非空图片 bytes，或存在的本地图片引用 |
| `annotation=YES` | 发现 TXT/Caption 文件，或抽样记录中的非空 caption/text/conversations 等文本字段 |
| `NO` | 已检查完所选子树的普通文件，未发现相应类型，且无抽样、容器或错误导致的不确定性 |
| `UNKNOWN` | 抽样没有确认；可能是扫描上限、外链/未解析路径、缺少依赖、未知格式或访问错误 |
| `BOTH_PRESENT` | 该目录范围内图像和文本各有存在证据；不证明一一配对或全库完整 |
| `MISSING_TYPE` | 有类型为 NO，且没有 UNKNOWN |
| `UNCONFIRMED` | 有类型为 UNKNOWN |

图片文件即使为空或内容损坏，只要存在也会计入，符合“只查存在”的用途。
Parquet 中 `image` 字段名本身不构成图片已下载的证据：只含 URL、
不存在的路径、全空 `image.bytes` 都不会判为图片 YES。
README、许可证、常见配置文件排除；JSON 文件仅存在但只含 URL/id 等元数据，
不会被自动当成文本标注。问答/指令文本也属于文本存在证据，不代表适合作为生成 caption。

## 抽样及开销

- 目录使用有界广度遍历；默认每个数据集最多 20,000 个目录项、128 个目录，
  每个目录最多 2,000 个条目，深度最多 8。找到普通图片和 TXT 后可提前结束。
- 对发现的候选文件按后缀分别随机抽最多 3 个（`--sample-shards`），
  固定 `--seed 0`。例如存在 TAR 和 Parquet，会分别抽 3 个。
  这是有界目录前缀内的抽样，不是对整库的均匀随机抽样。
- TAR/TAR.GZ/TGZ/TAR.BZ2/TAR.XZ/ZIP 检查成员名，通常找到图文后即结束；
  每包最多检查 2,000 个成员（`--max-members`）。不读取图片以做解码验证。
- 未压缩 TAR 可跳过成员内容。压缩 TAR 为跳到下一成员，需要流式解压，
  默认最多消耗 64 MiB 解压数据（`--max-archive-bytes`）；超过后标记未确认。
  ZIP 库会读取中央目录，但不解压图片成员。
- JSON/JSONL/CSV/TSV 只读最多 1 MiB 前缀（`--max-probe-bytes`），
  解析少量完整记录，避免加载数 GB 的标注。只从解析出的字段判断文本存在。
- Parquet 投影相关列，从第一个 row group 取最多 8 行（`--sample-rows`），
  不调用 Hugging Face 自动图像解码。底层页可能大于这些样本，
  因此小 batch 不是严格的磁盘读取字节上限。
- `.tar.zst` 另需 `python -m pip install zstandard`。
  缺少可选依赖或遇到 Arrow/LMDB/7z 等未支持容器时，保留错误/原因并继续其他数据集。
- 不跟随遍历中遇到的目录符号链接，避免循环；受影响的缺失类型记为 UNKNOWN。

抽样确认了图文存在即可作为该数据集“已含图文”的证据；没有抽到时不推断整库没有。
需要补查时，可指定更精确的 `--dataset` 或提高相应预算，例如
`--max-dir-entries 10000 --sample-shards 5`。`--max-entries 0` 仅取消总目录项限制，
`--max-dir-entries 0` 仅取消单目录项限制，其他预算仍生效。

## 验证

```bash
python -m unittest discover \
  -s /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/scripts \
  -p test_check_genund_presence.py -v
```

测试使用临时文件，覆盖不能解码但存在的图片、URL-only、空图片 bytes、
不存在的路径、多分片 JSON 标注、压缩包和扫描预算；无 PyArrow 时跳过 Parquet 用例。
