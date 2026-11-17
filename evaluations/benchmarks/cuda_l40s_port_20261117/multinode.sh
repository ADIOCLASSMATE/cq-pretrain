#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd/global_user/wanjiaxin-253108030048/code/cq-pretrain-l40s
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=1 NPROC_PER_NODE=8
: "${PET_NNODES:?Missing platform node count}"
[[ "$PET_NNODES" == 2 ]]
.venv-cuda/bin/python - <<'PY'
import json,os,socket,torch
print(json.dumps(dict(launch_node=socket.gethostname(), visible_gpus=torch.cuda.device_count(),
    rendezvous={k:os.environ.get(k) for k in ('PET_NNODES','PET_NODE_RANK','PET_MASTER_ADDR','PET_MASTER_PORT')})),flush=True)
assert torch.cuda.device_count()==8
PY
common=(--config configs/qwen35_2b_wan.yaml --synthetic
  --text-assets .assets/Qwen3.5-2B-Base --vision-assets .assets/Qwen3.5-2B-Base
  --wan-assets .assets/Wan2.1-T2V-1.3B --tp 1 --global-batch 16 --sequence-length 2048
  --images 1 --renderer-ema-decay 0 --eval-snapshot-every 0 --log-all-ranks
  --auto-resume --checkpoint-dir .runs/cuda-port-20261117/multi-checkpoints --keep-checkpoints 1)
bash scripts/launch.sh "${common[@]}" --steps 4 --total-steps 4 --warmup 1 --save-final \
  --output .runs/cuda-port-20261117/multinode
bash scripts/launch.sh "${common[@]}" --steps 5 --total-steps 5 --warmup 0 \
  --output .runs/cuda-port-20261117/multinode-resume
if [[ "$PET_NODE_RANK" == 0 ]]; then
  .venv-cuda/bin/python -m scripts.validation.summarize_cuda_run .runs/cuda-port-20261117/multinode --nodes 2 --steps 4
  .venv-cuda/bin/python -m scripts.validation.summarize_cuda_run .runs/cuda-port-20261117/multinode-resume --nodes 2 --steps 1
fi
