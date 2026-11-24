#!/usr/bin/env bash
# Stage-1 512-area pretraining on 32 Ascend 910B (2 nodes x 16, TP1 DP32,
# global batch 128 = 4 packs per card), 8K packs, RF=1, logit-normal(0,1) with
# shift 3, WSD learning rate.
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
#     image), which starved the cards on a 12-CPU-per-card node. Reads now
#     verify the member, which is what pins the bytes.
set -uo pipefail
RUN=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/runs/cq-stage1-8k-npu32
SRC="$RUN/SOURCE"
# Both live under global_user, not under /inspire/sj-ssd3/project/<project>/.
# A project-scoped mount is only present for jobs billed to that project, so
# reading the corpus from one tied this run to whoever paid for it: billing the
# public project made the corpus vanish and every attempt failed on a missing
# path. These copies are byte-identical, and catalog and asset identity are
# content hashes rather than paths, so the resume contract is unaffected.
DATA=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/datasets/mm_pretrain/stage1-512-v1/release
ASSETS=/inspire/sj-ssd3/global_user/wanjiaxin-253108030048/models
DECAY_START="${DECAY_START:-73000}"   # first cosine step; an epoch is about 91K updates
DECAY_END="${DECAY_END:-91000}"
NODE="${PET_NODE_RANK:-${NODE_RANK:-0}}"
ATTEMPT="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN/logs"
LOG="$RUN/logs/node${NODE}.${ATTEMPT}.log"
cd "$SRC"
{
  echo "host=$(hostname) node=$NODE nnodes=${PET_NNODES:-?} master=${PET_MASTER_ADDR:-?} commit=$(cat SOURCE_COMMIT)"
  npu-smi info -l 2>/dev/null | head -5
  for mount in dcvlm-baseline-200b/v2 gen-und-mm-pretrain/01 imgedit/v1 infovis/v1 textground4m/v1 textatlas5m/v1 poster100k/v1 tide/v1; do
    [[ -d "/inspire/dataset/$mount" ]] || { echo "MISSING MOUNT /inspire/dataset/$mount"; exit 21; }
  done
} >>"$LOG" 2>&1 || exit 21
# Kernel caches survive restarts here rather than in the container, so a
# preempted job does not recompile them.
export TRITON_CACHE_DIR="$RUN/cache/triton"
export TORCH_EXTENSIONS_DIR="$RUN/cache/extensions"
export NPROC_PER_NODE=16 PRETRAIN_BACKEND=npu PRETRAIN_STACK=core17
# A silent fall back to a portable expression would be both slower and a
# different implementation from the one these numbers were gated against.
export PRETRAIN_REQUIRE_FUSED=1
bash scripts/launch.sh --config configs/qwen35_2b_wan.yaml \
  --text-assets "$ASSETS/Qwen3.5-2B-Base" --vision-assets "$ASSETS/Qwen3.5-2B-Base" \
  --wan-assets "$ASSETS/Wan2.1-T2V-1.3B" --wan-vae-assets "$ASSETS/Wan2.1-VAE" \
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
