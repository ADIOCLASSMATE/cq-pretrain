#!/usr/bin/env bash
set -euo pipefail

cd /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3

exec bash scripts/train.sh --config configs/openvid81_832x480_trial.yaml \
  --set training.exp_id=openvid81_832x480_smoke \
  --set training.max_train_steps=2 \
  --set training.gradient_accumulation_steps=2 \
  --set optimizer.warmup_steps=0 \
  --set optimizer.decay_steps=0 \
  "$@"
