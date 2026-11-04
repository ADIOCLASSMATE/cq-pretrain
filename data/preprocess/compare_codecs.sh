#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
comparison_root="$PWD/data/analysis/results/OpenVidHD/codec_comparison_10"
pilot_root=/inspire/sj-ssd3/global_user/lipengfei-253107010003/datasets/openvidhd_960x540_stride2_pilot10
mkdir -p "$comparison_root"
exec > >(tee -a "$comparison_root/comparison.log") 2>&1
trap 'comparison_status=$?; printf "%s\n" "$comparison_status" > "$comparison_root/exit_code"; exit "$comparison_status"' EXIT
.venv-npu/bin/python -m data.preprocess.compare_codecs --pilot "$pilot_root" --output "$comparison_root" --repeats 3
data/analysis/.venv/bin/python -m data.preprocess.plot_codec_comparison --output "$comparison_root"
