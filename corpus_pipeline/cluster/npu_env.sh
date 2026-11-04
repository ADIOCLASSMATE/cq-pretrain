#!/usr/bin/env bash
set -eo pipefail
CORPUS_REPO=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/code/ascend-pretrain
source /usr/local/Ascend/ascend-toolkit/set_env.sh
for CORPUS_DRIVER_LIB in /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
  if [[ -d "$CORPUS_DRIVER_LIB" ]]; then
    export LD_LIBRARY_PATH="$CORPUS_DRIVER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
export PYTHONPATH="$CORPUS_REPO${PYTHONPATH:+:$PYTHONPATH}"
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
cd "$CORPUS_REPO"
exec "$CORPUS_REPO/.venv/bin/python" "$@"
