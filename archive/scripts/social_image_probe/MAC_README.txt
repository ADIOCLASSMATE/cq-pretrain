Instagram #portrait 自动发现与照片采集（macOS v2.1）

升级：解压到 InstagramPortrait-v2.1 文件夹，双击其中的 Start Portrait v2.1.command。
启动器会核对版本，并显示“v2.1：版本检查通过”；请使用这一个新入口。
它会复用之前的专用浏览器登录资料和依赖环境；登录过期时按官网提示重新登录。
v2 通过 Chromium 页面读取帖子和图片；单帖网络失败会记录诊断后继续。
失败后新建下载标签页，避免错误页的残留跳转影响下一个帖子。连续三帖网络失败停止。
搜索页上的整批候选会在下载前保存，分别统计“发现”“尝试”“保存”，不再把已发现数量计成处理数量。
v2.1 进一步区分“页面未解析成功”和“已读正文没有目标标签”。初始 HTML 未识别时，最多等待浏览器渲染 5 秒后重读。
连续三帖无法解析就停止并记录页面结构计数，不再重复检查 60 帖后笼统报告标签不匹配。
记录只包括页面类型、元素计数和已核验的标签/图片字段，不导出原始 HTML、登录凭据或浏览器截图。

1. 将压缩包解压到本机文件夹。
2. 双击 Start Portrait v2.1.command。
3. 首次运行会安装依赖；若没有 Google Chrome，会下载 Chromium。
4. 在新打开的 Instagram 官网窗口输入账号、密码和验证码。
5. 检测到登录后，脚本自动从 #portrait 标签页和 #portrait 搜索出发，滚动发现新帖子。
6. 只下载原帖正文带精确 #portrait、且明确是静态照片的图片。轮播取明确的照片子项，每帖最多一张。
7. 收集到 5 张后停止并打开图库；若搜索结果不足、登录过期、验证码或限流，则报告实际进度。

需要 Python 3.11+ 和能访问 Instagram 的网络。启动器自动识别常见的 macOS Python 安装位置。
手动从 Terminal 启动的等效方式：bash 后面加一个空格，再把 Start Portrait v2.1.command 拖进 Terminal，按回车。

结果：本机 Pictures/InstagramTagSamples/portrait_v2_1_日期时间/
- gallery.html：可离线打开的照片预览。
- images/：照片副本，已去除嵌入元数据。
- discovered_posts.jsonl：只从本次标签搜索发现的新帖及发现位置。
- attempted_posts.jsonl：本次实际开始处理的帖子。
- errors.jsonl：失败阶段、网络错误代码和分类，不保存原始异常文本、Cookie 或认证头。
- evidence/：原帖标签和照片类型证据；不保存登录页、Cookie、请求认证头或浏览器截图。
- manifest.jsonl：图片索引与哈希。
- summary.json：实际结果和停止原因，包含 collector_version=2.1 和脚本哈希。
- results.zip：可带回当前对话的结果包，不包含浏览器登录资料。

登录资料仅由专用浏览器保存在：
~/Library/Application Support/InstagramTagCollector/browser-profile/
脚本不会读取你平时 Chrome 的浏览器资料，不导出 Cookie，不上传结果。
若不想保留登录，请在该专用 Instagram 窗口登出；关闭脚本后也可删除上述专用目录。

流程没有候选链接输入，不读取之前的摄影文章或样本。没有模型 API，不按生活感、画面题材或 AI 判断筛图。
一次最多浏览 60 个新帖子，每个搜索入口最多滚动 12 次；连续三次没有新链接停止该入口。
只访问 Instagram 页面及其图片 CDN；不做私信、关注、点赞或发布。

状态说明：
- complete：已保存目标数量。
- discovery_exhausted：已结束有界搜索，实际图片可能不足 5 张。
- needs_user_login / needs_user_login_or_verification：需要你在官网完成登录或验证。
- rate_limited / http_access_limited：平台限制，自动停止。
- browser_or_network_error：检查本机网络和浏览器，再运行。
- repeated_browser_errors：连续三帖网络失败或浏览器已关闭，详情见 errors.jsonl。
- post_parser_unavailable：连续三帖无法识别页面/正文，详情见 evidence/ 与 errors.jsonl；不代表这些帖子没有目标标签。

v1 用户实跑结果：标签搜索返回 24 个帖子链接，首帖请求失败，下载 0 张。
原日志缺少具体报错，无法据此确定是代理、证书、连接还是其他错误。
第二次用户实跑仍用了旧启动器：60 帖返回 HTTP 200，但全部未识别媒体类型和正文，旧脚本错误地计作 exact_hashtag_missing。
v2.1 验证：20 项离线测试、4 项真实 Chromium 的模拟页面集成测试，以及启动器 Bash 语法和版本检查。
模拟测试覆盖首帖失败后继续保存 5 张、连续失败停止、平台限流停止、未解析提前停止、渲染后重读、HTTP 200 登录表单识别、凭据隔离。
测试图片是人工构造的数据，不属于 Instagram 采集结果。v2.1 尚未在你的 Mac 上实跑；不保证当前平台页面结构始终受支持。
