#!/usr/bin/env bash
# Real-weight four-card validation. All test overrides stay in this script.
set -euo pipefail
ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
# Child training scripts cannot export CANN back into this parent. The final
# checkpoint comparator imports torch_npu too, so initialize every stage here.
source "$ROOT/scripts/env_mindspeed.sh"
UMM_VERIFY_ROOT="${1:-$ROOT/outputs/mcore_verify_$(date -u +%Y%m%dT%H%M%SZ)}"
mkdir -p "$UMM_VERIFY_ROOT"
UMM_VERIFY_ROOT="$(cd "$UMM_VERIFY_ROOT" && pwd)"
common=(--set "training.output_root=$UMM_VERIFY_ROOT"
  --set training.max_train_steps=4 --set training.save_every=1
  --set training.keep_checkpoints=2 --set training.deterministic=true)
bash tests/validation/test_train.sh "${common[@]}" --set training.exp_id=continuous
bash tests/validation/test_train.sh "${common[@]}" --set training.exp_id=interrupted --stop-after-steps 2
# Exact restore without updates, also into an independently overridden log dir.
bash tests/validation/test_train.sh "${common[@]}" --set training.exp_id=restored \
  --set "training.log_dir=$UMM_VERIFY_ROOT/restore_logs" \
  --resume "$UMM_VERIFY_ROOT/interrupted/checkpoints/checkpoint-2" --stop-after-steps 2
"$PYTHON_BIN" - "$UMM_VERIFY_ROOT" <<'PY'
import hashlib,json,os,shutil,sys
from pathlib import Path
root=Path(sys.argv[1])
checkpoint=root/'interrupted/checkpoints/checkpoint-2'
shutil.copytree(checkpoint, root/'before_resume_checkpoint', copy_function=os.link)
hashes={str(p.relative_to(root)):hashlib.sha256(p.read_bytes()).hexdigest()
        for p in root.rglob('*') if p.is_file() and '/logs/' in str(p)}
(root/'logs-before-resume.json').write_text(json.dumps(hashes,indent=2)+'\n')
# An interrupted write must be ignored and quarantined after valid recovery.
broken=root/'interrupted/checkpoints/checkpoint-3'
broken.mkdir(); (broken/'partial').write_text('incomplete test payload\n')
PY
bash tests/validation/test_train.sh "${common[@]}" --set training.exp_id=interrupted --auto-resume
echo "Validation payloads: $UMM_VERIFY_ROOT"
"$PYTHON_BIN" -m tests.validation.verify_mcore_run "$UMM_VERIFY_ROOT"
