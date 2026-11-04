#!/usr/bin/env bash
set -euo pipefail
cd /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3
export TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONUNBUFFERED=1
probe_root=/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_selected_parallel_probe_20261030
mkdir -p "$probe_root/logs"
exec > "$probe_root/logs/remote_probe.log" 2>&1
trap 'probe_status=$?; printf "%s\n" "$probe_status" > "$probe_root/logs/exit_code"; exit "$probe_status"' EXIT
.venv-npu/bin/python -m data.preprocess.parallel_feasibility.probe preflight
bash data/preprocess/batch.sh pilot "$probe_root/pilot10" 4
.venv-npu/bin/python -m data.preprocess.parallel_feasibility.probe compare-pilot
.venv-npu/bin/python -m data.preprocess.parallel_feasibility.probe benchmark
