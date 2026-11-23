# 公开图片标签采集小测

## 中文标签对照入口

`Start Chinese Tags.command` 运行 `instagram_chinese_batch.py`，复用现有专用 profile，依次从 `#写真`、`#人像摄影`、`#生活记录` 搜索新帖，每组目标 5 张。逐组保留标签证据，统一生成图库、总索引及 `results.zip`。跨标签重复图片单独计数；单组结果不足仍继续，登录/限流/网络/解析故障停止后续标签。批次输出与 profile 强制使用互不包含的目录。使用方法见 `CHINESE_README.txt`。

用户回传的 v2.1 成功结果：24 个候选、处理 13 帖、保存 5 张静态图，图片哈希、尺寸及无 EXIF 已核验。第 4 张为绘画，原帖也带 `ink`、`inktober` 标签；其余图的风格不能证明是否相机实拍。`GraphImage`/轮播照片类型只区分静态图和视频，未提供真实性或绘画筛除。这次中文对照同样不加语义或 AI 筛选。

主采集脚本保持用户成功运行的 SHA256 `57fef81617d3dd9eb17d5dd4da2d208f7292f9eeb8a709814aba04d48b0cb8da`。本次新增 4 项离线测试及 1 项 Chromium 中文模拟页面集成测试通过；中文实时采集待用户本机运行。

## 当前入口：macOS 普通账号登录后自动发现

`Start Portrait v2.1.command` 在 Mac 本机打开专用浏览器，用户在 Instagram 官网手动登录。之后 `instagram_browser_collect.py` 从 `#portrait` 标签页及标签搜索出发，滚动发现新的 `/p/` 帖子，再核对作者正文的精确标签并下载照片。没有种子链接输入；不读取历史候选集，不调用模型 API，不需要 Meta 开发者令牌。

轮播只接受明确 `is_video=false` 的照片子项，每帖最多一张；视频封面不计数。默认目标 5 张，最多检查 60 个帖子，每入口最多滚动 12 次，连续三次无新链接停止。平台要求验证、限流、登录过期时停止并报告实际进度。登录资料留在本机专用 profile；图片下载用独立 Cookie 容器。结果 ZIP 只打包输出目录，不含 profile。图片会重编码并去除元数据，但没有处理人物面部或进行 AI 图检测。

启动方式见 `MAC_README.txt`。需要 Python 3.11+，启动器自动安装固定版本依赖。已有 Chrome 时使用其程序配合独立 profile；否则下载 Chromium。输出位于 `~/Pictures/InstagramTagSamples/portrait_日期时间/`，包含图库、图片、搜索发现记录、标签证据、`summary.json` 和 `results.zip`。原帖 URL 与 CDN URL 仅存于本机来源记录，不上传。

**v1 用户实跑：** 附件 `results.zip` 显示从 `/explore/tags/portrait/` 跳转搜索页后取得 24 个 `/p/` 链接，首帖 `CMpJ_IMFctW` 开始处理后发生 `BrowserError`，保存 0 张。旧版异常处理仅保留类型，无法判断具体网络原因；逐条 yield 时才写发现记录，导致摘要只计 1 个已发现帖。

**第二次回传与 v2.1：** 用户确认仍启动了旧的 `InstagramPortrait` 文件夹。此次发现并处理 60 帖，均为 HTTP 200，但所有标签及媒体类型字段为空。旧版将未识别页面误判为 `exact_hashtag_missing`。v2.1 只有读到媒体结构及作者正文容器后才判断标签不匹配；否则分别记为 `post_data_unrecognized` 或 `caption_unavailable`，并最多等待 5 秒后读取渲染 DOM。连续三帖仍无法解析便停止。日志仅保存结构计数和已提取证据，不保存原始登录态 HTML。新启动器有独立文件名及版本检查，结果目录包含 `v2_1`，摘要记录版本和脚本哈希。

**v2.1 验证：20 项离线测试及 4 项 Chromium 模拟页面集成测试通过。** 新增未识别页面与标签缺失的区别、初始页面空壳的渲染重读、HTTP 200 登录表单和连续未解析提前停止。未取得用户 Mac 上 v2.1 实跑结果；不能从旧版空字段确定实际返回的是登录页、应用壳还是变化的嵌入页结构。

**v2 修复：** 帖子嵌入页及 CDN 图片通过 Chromium 页面导航读取响应，取代独立的 Playwright APIRequestContext 请求；图片仍使用不含 Instagram 登录 Cookie 的独立浏览器上下文。单帖错误分类记录到 `errors.jsonl` 并继续，失败后更换下载标签页，避免 Chrome 错误页跳转干扰下一帖。连续三帖网络错误或平台明确限流时停止。发现整批候选先落盘，`attempted_posts.jsonl` 单独记录实际尝试；摘要带版本、阶段和尝试数量。此调整解决可复现的程序行为，不代表已经确认用户网络报错的根因。

**验证边界：17 项离线测试与 3 项真实 Chromium 模拟页面集成测试通过；macOS 启动器通过 Bash 语法检查。** 集成测试覆盖首帖失败后取得后续 5 张测试图、连续失败停止、HTTP 429 停止、图片请求无登录 Cookie、结果包无登录凭据。模拟图不属于 Instagram 样本。v2 未在用户 Mac 上实跑。远程匿名浏览器补测用户发现的首帖超时，不能外推用户本机也有同样原因。记录位于 `macos_v2_live_probe/summary.json`。

浏览器读取响应及上下文隔离分别依据 [Playwright Response.body](https://playwright.dev/python/docs/api/class-response#response-body) 与 [BrowserContext](https://playwright.dev/python/docs/api/class-browsercontext)。集成测试需设置 `IG_TEST_BROWSER` 为 Chromium 可执行文件绝对路径；未设置时明确跳过这 3 项。

## 另一路径：已获授权的 Meta Hashtag Search

`collect_instagram_hashtag.py` 的业务输入只有标签、目标数量和输出目录，不读取旧候选集，不需要输入摄影师账号、文章或帖子链接。它实现的是授权 Meta Hashtag Search 适配器，仍须使用有对应接口访问权限的配置；普通 Instagram 登录账号并不自动等同于这种接口授权。

流程：

1. 将 `#portrait` 规范化为 `portrait`。
2. 请求 `ig_hashtag_search?q=portrait&user_id=...`，取得标签 ID。
3. 请求该标签的 `top_media`、`recent_media`，通过返回的游标翻页，自动发现新帖子。
4. 对每个新帖核对作者正文的精确 `#portrait`；`#portraitphotography` 不代替。
5. 提取 `IMAGE`，轮播只取明确标为 `IMAGE` 的子项，跳过视频。默认每个帖子最多保存一张，防止一个轮播占满样本。
6. 下载、解码、检查短边、去除元数据、按文件哈希去重，保存图片和来源证据。
7. 满 5 张即停止；达到页数/帖子数上限、没有下一页或接口报错时报告实际数量，不补入旧图。

此流程不调用模型 API，不按画面语义或生活感筛图。Instagram/Meta 平台接口不是模型 API。

启动前在运行环境配置 `IG_ACCESS_TOKEN`、`IG_USER_ID`、`IG_API_VERSION`。版本使用 Meta 应用实际配置的版本；令牌仅通过请求头发送，不写入命令行、索引或日志，不转发给图片 CDN。脚本不会帮用户申请接口权限。

```bash
.venv/bin/python scripts/social_image_probe/collect_instagram_hashtag.py \
  --tag portrait --limit 5 --max-pages 3 --max-posts 100 \
  --output corpus_pipeline/runs/social_image_probe_20261024/portrait_by_hashtag_001
```

必须使用新的输出目录。`--max-pages` 是每个 feed 的页数上限；只查询一个标签。输出 `discovered_posts.jsonl` 记录每条新帖子来自哪个标签、feed、页码，`discovery_events.jsonl` 记录查询和翻页，`manifest.jsonl` 记录实际保存的照片，`summary.json` 区分完成、数量不足、缺配置、接口失败。缺配置或未完成时退出码为 2。

**真实验证状态：流程代码与模拟分页测试已通过，授权接口尚未实跑。** 当前环境三个配置均缺失，预检记录为 `hashtag_flow_preflight/summary.json`，`live_network_run=false`、下载数 0；不能将模拟测试中的 5 个测试图片称为 Instagram 采集结果。正式接口的权限、字段支持和当前返回内容仍需接入授权后核验。

匿名发现补测证据位于 `hashtag_discovery_probe/`：标签页重定向 Reels；公开标签接口请求超时；Google、DuckDuckGo 返回人机验证；Bing 未返回有效 Instagram 帖子；Imginn 本机请求为 403；Yahoo 超时。没有处理验证码、复用用户登录态或将文章候选冒充标签搜索。这些失败只描述当前环境，不证明其他合法接入方式不可行。

接口结构参考 Meta 官方 SDK 的 [Hashtag feeds](https://github.com/facebook/facebook-python-business-sdk/blob/main/facebook_business/adobjects/shadowighashtag.py) 与 [IG media](https://github.com/facebook/facebook-python-business-sdk/blob/main/facebook_business/adobjects/igmedia.py)。文档站本次读取受限，SDK 核验不能替代真实授权接口测试。

```bash
.venv/bin/python -m unittest discover -s scripts/social_image_probe \
  -p 'test_instagram*.py' -v
```

以下保留历史采集尝试。`collect_instagram_photos.py` 只是已知 URL 的下游照片下载器，不能单独作为“从标签发现帖子”的流程。

采集脚本不调用模型 API，不运行检测或标注模型。`collect.py` 探测 Instagram、X、小红书匿名页面，并实现 Flickr 标签采集；新增的 `collect_instagram_photos.py` 可从已知公开帖子链接下载明确标为静态图片的 Instagram 照片，详见文末补测。Instagram 按标签自动发现照片帖尚未打通。

当前 Instagram 照片采集器强制核对作者正文里的精确标签：输入种子的 `category` 是要求匹配的标签（`portrait` 或 `landscape`），不再沿用文章类别决定收录。`portraitphotography`、`selfportrait`、评论区标签及正文中的普通单词不算 `portrait`。只有公开嵌入页作者正文中存在对应的 Instagram 标签链接，且媒体类型为 `GraphImage`，才会下载；不使用模型判断题材或生活感。旧静态照片实验的 10 张没有使用这项条件，其中仅 1 张符合 `#portrait`、0 张符合 `#landscape`，不能作为满足标签要求的样本集。

在仓库根目录运行（当前 `.venv` 已有 requests/Pillow）：

```bash
.venv/bin/python scripts/social_image_probe/collect.py \
  --output corpus_pipeline/runs/social_image_probe_20261024/run_001 \
  --tags landscape portrait --limit 5 --min-side 512
```

每次必须指定新的输出目录。需要独立环境时安装本目录 `requirements.txt`。默认串行、每次请求至少间隔 1 秒、每组最多 5 张；单次响应设字节上限，不自动重试。使用环境已有的网络代理配置，不加载账号凭据或浏览器 Cookie。

只复跑已实现的 Flickr 采集：

```bash
.venv/bin/python scripts/social_image_probe/collect.py \
  --sources flickr --tags landscape portrait --limit 5 --min-side 512 \
  --output corpus_pipeline/runs/social_image_probe_20261024/run_002
```

输出：`summary.json` 是实测结果；`requests.jsonl` 是请求回执；`manifest.jsonl` 是无作者字段的图片索引；`images/` 按 SHA256 命名；`provenance.private.jsonl` 单独保存来源及标签；`evidence/` 保存公开响应以便复查。输出目录权限为 700，文件为 600。图片使用副本去除 EXIF/XMP 等嵌入元数据并重编码为 JPEG，不是逐字节原图，也不保证可辨认人物已匿名化。

Flickr 精确匹配返回的标签列表，不用文字描述代替标签；从照片页真实 `og:image` 读取较大版本，不猜测 CDN 原图地址。它仍可能是平台缩放图。这里只取一页最近公开 feed，未实现历史翻页或全站检索。图片和来源记录保持 `ai_status=not_checked`、`rights_status=not_reviewed`，用于后续清理，不代表已获训练许可或通过既有语料管道。

Instagram 的主题页封面、小红书的 logo、X 的登录页图片不计入有效样本。初轮 HTTP 模式失败也不证明浏览器方式一定不可行；后续浏览器与单帖补测见下文，均未使用登录态。小红书入口是关键词发现，仅探测是否返回笔记，不把关键词命中称为精确标签通过。

Flickr feed 文档：https://www.flickr.com/services/feeds/docs/photos_public/

## 2026-10-24 实测与浏览器补测

实际结果保存在 `corpus_pipeline/runs/social_image_probe_20261024/`：

| 平台 | 风景 | 写真 | 样本与限制 |
|---|---:|---:|---|
| Flickr | 5 | 5 | 原帖图片的大图版本，约 1024 像素；不保证原始分辨率 |
| Instagram | 5 | 5 | 360/361×640 的 Reels 视频封面，不能计作原始照片 |
| X | 0 | 0 | 匿名搜索重定向登录 |
| 小红书 | 0 | 0 | HTTP 搜索列表为空；浏览器返回安全限制 300012 / IP at risk |

Flickr 使用精确 `landscape`、`portrait` 标签。Instagram 初始 `landscape` 主题页包含园艺、绘画等噪声，因此补测 `landscapephotography`、`portraitphotography`，核对图片 alt 中的摄影标签后各保存 5 张。规则仅匹配标签，排除明确绘画标签，不进行模型推理、语义分类或 AI 检测。共 20 张的文件哈希、解码、尺寸、标签来源与元数据检查通过；`preview/gallery.html` 为完全离线图库，四张拼图位于同目录。采集图片、证据与来源均在被 Git 忽略的运行目录内。

`browser_probe.mjs` 使用 Node 22 自带 WebSocket 调用本地 Chromium CDP，每页等待 12 秒，无登录、滚动或模型 API。`collect_browser_images.py` 从保存的浏览器输出下载有匹配标签的帖子预览，明确记录 `sample_kind=reel_cover`，这一步的短边门槛是 256。浏览器诊断不是已完成的全站翻页采集器。

```bash
node scripts/social_image_probe/browser_probe.mjs \
  /root/.cache/ms-playwright/chromium-1248/chrome-linux64/chrome \
  corpus_pipeline/runs/social_image_probe_20261024/new_browser \
  landscapephotography portraitphotography

.venv/bin/python scripts/social_image_probe/collect_browser_images.py \
  --input landscape=corpus_pipeline/runs/social_image_probe_20261024/new_browser/instagram_landscapephotography.json \
  --input portrait=corpus_pipeline/runs/social_image_probe_20261024/new_browser/instagram_portraitphotography.json \
  --output corpus_pipeline/runs/social_image_probe_20261024/new_instagram --limit 5

.venv/bin/python scripts/social_image_probe/make_preview.py \
  --runs corpus_pipeline/runs/social_image_probe_20261024/run_001 \
         corpus_pipeline/runs/social_image_probe_20261024/instagram_previews \
  --output corpus_pipeline/runs/social_image_probe_20261024/new_preview
```

Chromium 路径须使用本机实际安装位置。省略末尾两个 Instagram 标签参数时，浏览器脚本执行预置的 Instagram portrait、小红书两类搜索诊断。实时标签页会变化，不保证复跑仍返回相同的图片或数量。`make_preview.py` 检查每组恰好 5 张并生成预览拼图；预览缩放使用 Pillow，不使用图像生成或模型 API。

## Instagram 静态照片补测

同日后续已取得风景 5 张、人像 5 张静态照片，长边 996–1800 像素。保存在 `photo_attempt/photos/` 和 `photo_attempt/photos_extra/`，预览位于 `photo_attempt/preview/`，均相对于上述运行目录。前文的 Instagram Reels 封面仍为初次实验结果，不与这次照片混算。

可用链路是：已知公开 `/p/SHORTCODE/` 链接 → `/embed/captioned/` 公开嵌入页 → 验证 `data-media-type="GraphImage"` → 下载 `EmbeddedMediaImage` 对应的 Instagram CDN 图片。仅接受明确的单张静态图片帖；轮播 `GraphSidecar`、视频及未知类型全部跳过。保存响应证据、来源和校验结果，图片去除嵌入元数据。没有使用账号凭据或模型 API。

这次帖子链接来自公开摄影文章中的 Instagram 链接，分组沿用文章类别，记录 `tag_filter_used=false`，不冒充按标签搜索结果。标签页及摄影师主页仍未提供可用的匿名照片搜索列表。`GraphImage` 只能确认静态图片类型，不能证明非 AI 或未经后期处理；下载的也是平台提供的版本，不保证相机原始分辨率。

上述是修改筛选器之前的历史实验。当前版本已强制精确标签过滤，以下命令复跑时成功数将不同。标签核验与标签入口发现是两个独立步骤，前者已经实现，后者仍未打通。

种子 JSON 格式为 `[{"category":"portrait","post_url":"https://www.instagram.com/p/SHORTCODE/","source_url":"来源文章地址"}]`。每次输入不超过 40 条，每类最多 10 张，串行请求间隔至少 1.5 秒，不自动重试。示例中输出目录必须尚不存在：

```bash
.venv/bin/python scripts/social_image_probe/collect_instagram_photos.py \
  --seeds corpus_pipeline/runs/social_image_probe_20261024/photo_attempt/photo_seeds.json \
  --output corpus_pipeline/runs/social_image_probe_20261024/photo_attempt/photos_retry \
  --limit 5 --min-side 512

.venv/bin/python scripts/social_image_probe/make_preview.py \
  --runs corpus_pipeline/runs/social_image_probe_20261024/photo_attempt/photos \
         corpus_pipeline/runs/social_image_probe_20261024/photo_attempt/photos_extra \
  --output corpus_pipeline/runs/social_image_probe_20261024/photo_attempt/preview_new
```

`summary.json` 报告实际成功数；失效、限制或类型不符时可能不足目标数量。CDN 链接会过期，复跑应重新读取嵌入页。`browser_probe.mjs` 也支持用 `--targets=PATH` 传入至多 12 个 `{ "tag": "本地文件名", "url": "https://www.instagram.com/..." }` 目标用于诊断；本次真正下载照片只需要 HTTP 脚本。

## 精确标签纠正与复测

`photo_attempt/exact_tag_audit.json` 逐条核对旧 10 张的作者正文标签。当前版本以 `category_basis=exact_caption_hashtag`、`tag_filter_used=true` 记录通过样本，并在私有来源索引保存 `matched_tag_href`。不通过时记录 `exact_hashtag_missing`，不会下载图片凑数。

`exact_tag_photos/` 用 15 条已知帖子链接复测（含无标签和仅有相似标签的对照）。`tag_retry_browser/` 保存两次匿名标签入口访问的证据：两者均转到 `/popular/.../`，返回的帖子链接是 Reels，没有静态 `/p/` 帖子链接。因此该结果仍不代表实现了按标签自动获取各 5 张照片。

回归检查覆盖相似标签、评论/页面其他区域、大小写、换行和外站链接：

```bash
.venv/bin/python -m unittest discover -s scripts/social_image_probe \
  -p 'test_instagram_caption_tags.py' -v
```
