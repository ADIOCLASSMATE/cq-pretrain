#!/usr/bin/env bash
set -euo pipefail
export CORPUS_CONFIG=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/serving/qwen/config-w8a8.json
exec bash /inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain/corpus_pipeline/serving/qwen/launch.sh
