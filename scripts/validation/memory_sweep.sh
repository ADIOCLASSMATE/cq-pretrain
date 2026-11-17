#!/usr/bin/env bash
# Production-shape boundary sweep on L40S (45 GiB usable).
# Finds where TP1 stops fitting and whether GA>1 keeps the ordered-bucket /
# deferred-renderer path healthy. Records peak memory and s/step per config.
set -u
cd "$(dirname "${BASH_SOURCE[0]}")/../.."
ROOT=.runs/memsweep-$(date +%m%d)
mkdir -p "$ROOT"
ASSETS=(--text-assets .assets/Qwen3.5-2B-Base --vision-assets .assets/Qwen3.5-2B-Base
        --wan-assets .assets/Wan2.1-T2V-1.3B)
for spec in "1 2048 1" "1 4096 1" "1 8192 1" "2 8192 1" "2 8192 2" "2 4096 2"; do
  set -- $spec
  TP=$1; SEQ=$2; GA=$3
  GPUS=$TP
  GB=$(( GPUS / TP * GA ))
  NAME="tp${TP}-seq${SEQ}-ga${GA}"
  echo "=== $NAME (gpus=$GPUS global_batch=$GB) ==="
  OMP_NUM_THREADS=2 NPROC_PER_NODE=$GPUS timeout 900 \
  bash scripts/launch.sh --config configs/qwen35_2b_wan.yaml --synthetic "${ASSETS[@]}" \
    --tp "$TP" --sequence-length "$SEQ" --global-batch "$GB" --images 1 \
    --steps 4 --warmup 1 --renderer-ema-decay 0 --eval-snapshot-every 0 --log-all-ranks \
    --output "$ROOT/$NAME" > "$ROOT/$NAME.log" 2>&1
  CODE=$?
  if [ $CODE -eq 0 ] && [ -f "$ROOT/$NAME/summary.json" ]; then
    python3 - "$ROOT/$NAME/summary.json" "$NAME" <<'PY'
import json,sys
d=json.load(open(sys.argv[1]))
print("  OK   %-18s peak=%.2f GiB  mean=%.3f s  tok/s=%.0f  ga=%s" % (
    sys.argv[2], d.get("peak_memory_gib",0), d.get("mean_seconds",0),
    d.get("tokens_per_second") or 0, d.get("grad_accum")))
PY
  else
    if grep -qiE "out of memory|OutOfMemoryError" "$ROOT/$NAME.log"; then
      echo "  OOM  $NAME (exit $CODE)"
    else
      echo "  FAIL $NAME (exit $CODE) -- last lines:"
      tail -6 "$ROOT/$NAME.log" | sed 's/^/        /'
    fi
  fi
done
echo "=== sweep root: $ROOT ==="
