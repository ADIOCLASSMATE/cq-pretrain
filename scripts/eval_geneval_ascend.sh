#!/usr/bin/env bash
# GenEval sampling on Ascend, in one scheduling slot.
#
# Every Ascend quota row allocates a whole 16-card node, so a job holds 16
# cards whether it uses one or sixteen, and it must gang-schedule onto a fully
# free node. On a cluster that runs at 100% those slots are scarce, so this
# script earns the slot twice:
#
#   1. A one-prompt smoke. The evaluation runtime was CUDA-only until it was
#      ported, so if that port is wrong this exits in about a minute and gives
#      the node back instead of discovering it twenty minutes in.
#   2. The full 553 x 4 run, into the same directory. Sampling is seeded per
#      (prompt, sample) and skips files that exist, so the smoke's four images
#      are kept rather than redrawn, and a preempted attempt resumes.
#
# Only the full run writes GENERATED.json, and only that receipt lets the H100
# detector score the images: it symlinks all 553 prompt folders and refuses a
# partial set. That is why a smoke alone could never close this loop.
set -uo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${1:?Usage: eval_geneval_ascend.sh OUTPUT_DIR WEIGHTS}"
WEIGHTS="${2:?missing WEIGHTS}"
mkdir -p "$OUT"
LOG="$OUT/stage.log"
echo "=== $(date -u +%FT%TZ) host=$(hostname) weights=$WEIGHTS" >>"$LOG"

if [[ -f "$OUT/GENERATED.json" ]]; then
  echo "already complete" >>"$LOG"
else
  if [[ ! -f "$OUT/SMOKE.json" ]]; then
    echo "--- stage 1: one-prompt smoke" >>"$LOG"
    EVAL_LIMIT=1 bash "$ROOT/scripts/evaluate_ascend.sh" geneval "$WEIGHTS" "$OUT" >>"$LOG" 2>&1
    if [[ ! -f "$OUT/SMOKE.json" ]]; then
      echo "SMOKE FAILED: the Ascend sampling path is broken; not holding the node" >>"$LOG"
      tail -40 "$OUT/job.log" 2>/dev/null
      exit 3
    fi
    echo "smoke ok: $(find "$OUT/images" -name '*.png' | wc -l) images" >>"$LOG"
  fi
  echo "--- stage 2: full 553 x 4" >>"$LOG"
  bash "$ROOT/scripts/evaluate_ascend.sh" geneval "$WEIGHTS" "$OUT" >>"$LOG" 2>&1
  [[ -f "$OUT/GENERATED.json" ]] || { echo "full run incomplete" >>"$LOG"; tail -40 "$OUT/job.log"; exit 4; }
fi

# Pack for the handoff: relative paths so the detector can unpack anywhere.
tar czf "$OUT.tar.gz" -C "$(dirname "$OUT")" "$(basename "$OUT")"
cd "$(dirname "$OUT")" && md5sum "$(basename "$OUT").tar.gz" > "$OUT.tar.gz.md5"
{ echo "packed $(du -h "$OUT.tar.gz" | cut -f1)"; cat "$OUT.tar.gz.md5"; } >>"$LOG"
echo "READY $OUT.tar.gz"
