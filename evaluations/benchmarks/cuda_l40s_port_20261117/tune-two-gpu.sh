#!/usr/bin/env bash
set -euo pipefail
cd /inspire/hdd/global_user/wanjiaxin-253108030048/code/cq-pretrain-l40s
export OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 NPROC_PER_NODE=2
common=(--config configs/qwen35_2b_wan.yaml --synthetic
  --text-assets .assets/Qwen3.5-2B-Base --vision-assets .assets/Qwen3.5-2B-Base
  --wan-assets .assets/Wan2.1-T2V-1.3B --global-batch 2 --sequence-length 2048
  --images 1 --steps 5 --warmup 1 --renderer-ema-decay 0 --eval-snapshot-every 0 --log-all-ranks)
bash scripts/launch.sh "${common[@]}" --tp 2 --output .runs/cuda-port-20261117/tp2-dp1
.venv-cuda/bin/python -m scripts.validation.summarize_cuda_run .runs/cuda-port-20261117/tp2-dp1 --nodes 1 --steps 5
bash scripts/launch.sh "${common[@]}" --tp 1 --output .runs/cuda-port-20261117/tp1-dp2
.venv-cuda/bin/python -m scripts.validation.summarize_cuda_run .runs/cuda-port-20261117/tp1-dp2 --nodes 1 --steps 5
