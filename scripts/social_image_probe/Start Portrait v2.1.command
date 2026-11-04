#!/bin/bash
set -u
umask 077

IG_TASK_DIR="$(cd "$(dirname "$0")" && pwd)"
IG_TASK_RUNTIME="$HOME/Library/Application Support/InstagramTagCollector"
IG_TASK_PYTHON=""

ig_task_stop() {
  printf '\n%s\n' "$1"
  read -r -p '按回车关闭窗口。' IG_TASK_REPLY
  exit 1
}

for IG_TASK_CANDIDATE in /opt/homebrew/bin/python3 /usr/local/bin/python3 "$(command -v python3 || true)"; do
  if [ -n "$IG_TASK_CANDIDATE" ] && [ -x "$IG_TASK_CANDIDATE" ] && "$IG_TASK_CANDIDATE" -c 'import sys; sys.exit(0 if sys.version_info >= (3,11) else 1)' 2>/dev/null; then
    IG_TASK_PYTHON="$IG_TASK_CANDIDATE"
    break
  fi
done
if [ -z "$IG_TASK_PYTHON" ]; then
  ig_task_stop '需要 Python 3.11 或更新版本。请安装 macOS Python：https://www.python.org/downloads/macos/ ，然后重新双击本文件。'
fi

mkdir -p "$IG_TASK_RUNTIME" || ig_task_stop '无法创建本机运行目录。'
chmod 700 "$IG_TASK_RUNTIME"
if [ ! -x "$IG_TASK_RUNTIME/venv/bin/python" ]; then
  "$IG_TASK_PYTHON" -m venv "$IG_TASK_RUNTIME/venv" || ig_task_stop '创建 Python 环境失败。'
fi
IG_TASK_PYTHON="$IG_TASK_RUNTIME/venv/bin/python"
if ! "$IG_TASK_PYTHON" -c 'import importlib.metadata as m; assert m.version("playwright")=="1.64.1"; assert m.version("Pillow")=="13.0.0"; assert m.version("requests")=="2.34.2"' 2>/dev/null; then
  printf '首次启动：安装浏览器控制和图片处理依赖。\n'
  "$IG_TASK_PYTHON" -m pip install -r "$IG_TASK_DIR/browser_requirements.txt" || ig_task_stop '依赖下载失败，请检查网络后重新启动。'
fi
if [ ! -x '/Applications/Google Chrome.app/Contents/MacOS/Google Chrome' ]; then
  "$IG_TASK_PYTHON" -m playwright install chromium || ig_task_stop '浏览器下载失败，请检查网络后重试。'
fi

IG_TASK_VERSION="$("$IG_TASK_PYTHON" "$IG_TASK_DIR/instagram_browser_collect.py" --version)" || ig_task_stop '读取采集器版本失败。'
if [ "$IG_TASK_VERSION" != '2.1' ]; then
  ig_task_stop '文件版本混用。请重新解压完整的 InstagramPortrait-v2.1 文件夹，再运行其中的 Start Portrait v2.1.command。'
fi
IG_TASK_OUTPUT="$HOME/Pictures/InstagramTagSamples/portrait_v2_1_$(date +%Y%m%d_%H%M%S)"
printf '\nInstagram 标签采集 v2.1：版本检查通过，复用本机专用浏览器。\n若登录已过期，请在 Instagram 官网登录。\n登录后自动搜索 #portrait，目标为 5 个不同帖子中的照片。\n'
"$IG_TASK_PYTHON" "$IG_TASK_DIR/instagram_browser_collect.py" \
  --tag portrait --limit 5 \
  --profile "$IG_TASK_RUNTIME/browser-profile" \
  --output "$IG_TASK_OUTPUT"
IG_TASK_STATUS=$?
if [ "$IG_TASK_STATUS" -eq 0 ]; then
  open "$IG_TASK_OUTPUT/gallery.html"
fi
if [ -d "$IG_TASK_OUTPUT" ]; then
  open "$IG_TASK_OUTPUT"
fi
printf '\n结果目录：%s\n可反馈 summary.json 或 results.zip；登录资料不在结果包中。\n' "$IG_TASK_OUTPUT"
read -r -p '按回车关闭窗口。' IG_TASK_REPLY
exit "$IG_TASK_STATUS"
