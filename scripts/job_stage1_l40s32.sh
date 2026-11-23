#!/usr/bin/env bash
# Stage-1 512-area pretraining on 32 L40S (4 nodes x 8, TP1 DP32, global batch
# 128 = 4 packs per card), 8K packs, RF=1, logit-normal(0,1) with shift 3, WSD
# learning rate. The Ascend sibling is scripts/job_stage1_npu32.sh; the two
# differ only in runtime, geometry and corpus paths.
#
# Runs on every node. The platform restarts the whole job after a failure or a
# preemption and --auto-resume continues from the newest committed checkpoint
# (weights, optimizer, FP32 masters, EMA, RNG and the consumed data cursor).
#
# This file is re-read on every restart, so a fix rolls out without stopping
# the job: freeze a new snapshot and switch SRC below with a rename, never an
# in-place edit, so running shells keep reading the old file.
#
# Two earlier failures this configuration is shaped by:
#   * a corpus record that cannot bind its image used to abort the step, and a
#     resumed cursor reached it every restart -- five restarts in 90 minutes
#     with no progress. data/upstream.py now passes over it; the controller
#     stops paying after three restarts that commit nothing.
#   * every image read hashed its entire containing TAR (~440 MiB per 210 KiB
#     image). Reads now verify the member, which is what pins the bytes.
set -uo pipefail
RUN=/inspire/hdd/global_user/wanjiaxin-253108030048/runs/cq-stage1-8k-l40s32
SRC="$RUN/SOURCE"
DATA=/inspire/hdd/global_user/wanjiaxin-253108030048/datasets/mm_pretrain/stage1-512-v1/release
DECAY_START="${DECAY_START:-73000}"   # first cosine step; an epoch is about 91K updates
DECAY_END="${DECAY_END:-91000}"
NODE="${PET_NODE_RANK:-${NODE_RANK:-0}}"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN/logs"
LOG="$RUN/logs/node${NODE}.${ATTEMPT}.log"
cd "$SRC"
{
  echo "host=$(hostname) node=$NODE nnodes=${PET_NNODES:-?} master=${PET_MASTER_ADDR:-?} commit=$(cat SOURCE_COMMIT)"
  nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | sort | uniq -c
  for mount in dcvlm-baseline-200b/v2 gen-und-mm-pretrain/01 imgedit/v1 infovis/v1 textground4m/v1 textatlas5m/v1 poster100k/v1 tide/v1; do
    [[ -d "/inspire/dataset/$mount" ]] || { echo "MISSING MOUNT /inspire/dataset/$mount"; exit 21; }
  done
} >>"$LOG" 2>&1 || exit 21
# Kernel caches survive restarts here rather than in the container, so a
# preempted job does not recompile them.
export TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCHINDUCTOR_CACHE_DIR="$RUN/cache/inductor"
export NPROC_PER_NODE=8 PRETRAIN_BACKEND=cuda PRETRAIN_STACK=cuda
bash scripts/launch.sh --config configs/qwen35_2b_wan.yaml \
  --text-assets .assets/Qwen3.5-2B-Base --vision-assets .assets/Qwen3.5-2B-Base \
  --wan-assets .assets/Wan2.1-T2V-1.3B --wan-vae-assets .assets/Wan2.1-VAE \
  --data-root "$DATA" --long-document-policy error \
  --tp 1 --global-batch 128 --sequence-length 8192 --until-data-end --warmup 0 \
  --rf 1 --timestep-shift 3.0 --dit-microbatch-tokens 8192 \
  --lr 5e-5 --lr-decay-style wsd --lr-warmup-steps 1000 --lr-decay-start "$DECAY_START" --lr-decay-steps "$DECAY_END" \
  --bf16-grad-reduce --checkpoint --keep-attention-activations 6 --phase-timing --log-all-ranks \
  --data-workers 8 --image-workers 12 --image-prefetch 48 --data-prefetch 24 \
  --output "$RUN/train" --checkpoint-dir "$RUN/checkpoints" --auto-resume \
  --save-every 50 --keep-checkpoints 3 --save-first --save-final \
  --eval-snapshot-every 500 --eval-snapshot-dir "$RUN/eval_snapshots" \
  --validation-suite "$SRC/validation_assets/suite.json" \
  --validate-every 500 >>"$LOG" 2>&1
code=$?
echo "node=$NODE exit=$code $(date -u +%FT%TZ)" >>"$LOG"
exit $code
