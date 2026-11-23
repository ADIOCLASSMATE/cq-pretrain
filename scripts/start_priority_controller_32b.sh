#!/bin/bash
# Start the priority controller for the 32-card stage-1 run unless it is already up.
# Pause: touch RUN/priority/STOP. Stop: kill $(cat RUN/priority/controller.pid).
R=/inspire/hdd/global_user/wanjiaxin-253108030048/runs/cq-stage1-8k-wsd-32b
SRC="$R/source_8df3e82"
mkdir -p "$R/priority"
if [[ -f "$R/priority/controller.pid" ]] && kill -0 "$(cat "$R/priority/controller.pid")" 2>/dev/null; then
  echo "already running: $(cat "$R/priority/controller.pid")"
  exit 0
fi
if [[ -f "$R/priority/ATTENTION.json" ]]; then
  echo "ATTENTION.json present; read it and delete it before restarting the controller" >&2
  exit 2
fi
cd "$R" && nohup python3 -u "$SRC/scripts/priority_controller.py" --run "$R" --prefix cq8k32b \
  >>"$R/priority/controller.out" 2>&1 &
echo "started $!"
