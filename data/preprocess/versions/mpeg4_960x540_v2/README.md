# OpenVidHD offline preprocessing

This directory owns the preprocessing implementation. It does not modify the
training loader, invoke Qwen/ViT or VAE, create latent caches, or truncate to 81 frames.

## Frozen transformation

Input scope is `/inspire/dataset/openvid-1m/v2/OpenVidHD` only. Join original
`data/train/OpenVidHD.csv`, `OpenVidHD/OpenVidHD.json`, and the full census under
`data/analysis/results/OpenVidHD`. Select **all original FPS** with exactly
1920×1080, at least 161 frames, and a caption that is nonempty after stripping.
Stripping is only a presence check: the stored caption remains unchanged.
The inspected input version contains **215,442 eligible videos**, **98 archive
groups**, and **207 exact source rates**. These counts are observed, not used as
hard-coded selection filters.

For each video:

1. Extract one ZIP member to temporary storage and check ZIP CRC and SHA256.
   A seekable concatenation reads split ZIPs without joining them on disk.
2. Decode every original frame in presentation order. Require its resolution,
   frame count and exact rational PTS intervals to match the frozen census.
3. Keep original frame indices `0, 2, 4, ...` to the end. Output count is
   `(original_count + 1) // 2`, so odd/even 161/162-frame inputs both give 81 frames.
4. Convert selected decoded frames to RGB24. Resize to 960×540 with CPU torch
   float32 bicubic interpolation, `antialias=True`, `align_corners=False`, then
   clamp to [0,255], round and cast to uint8. No cropping or padding.
5. Encode video-only MP4 using native `mpeg4` (MPEG-4 Part 2), `yuv420p`, fixed
   quantizer `q=2`, one encoder thread. PyAV uses the QSCALE flag and
   `global_quality=2*118`, matching the measured `mpeg4_q2` comparison. This is
   lossy encoding; the resized RGB is converted to YUV420 before compression.
6. Let the reduced output FPS be `p/q = original_FPS/2`. Set the MP4 time base to
   `1/p` and output frame `j` PTS to `j*q`. Rebase the original first timestamp to
   zero. This gives `output_time(j) = original_time(2*j) - original_time(0)`.
   No decimal FPS rounding, frame duplication, interpolation or speed change.
7. Decode the entire output before committing it. Check actual frame count,
   geometry, codec/pixel format, stream duration and every PTS. Store separate
   SHA256 values for resized pre-encoder RGB and decoded output RGB. Later
   readback checks compare the output to its decoded hash, not to the lossy
   encoder input. The pilot independently reconstructs the input hash from
   original frames `2*j` and measures RGB PSNR/MAE against the encoded output.
8. Fsync and atomically rename the MP4, then fsync its success journal entry.

For odd original frame counts the final retained frame is displayed for one
output interval. Thus the encoded end duration grows by one *source* interval;
the first-to-last-frame timestamps still align exactly. Both `duration_seconds`
and `pts_span_seconds` are recorded to distinguish these conventions.

## Output layout and identity

Default production root:
`/inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_960x540_selected`

```text
videos/part1/<unchanged-original-filename>.mp4
videos/part2/...
metadata/OpenVidHD.csv         # full original annotation CSV, byte-for-byte copy
metadata/OpenVidHD.json        # full original part index, byte-for-byte copy
metadata/selected.jsonl        # immutable candidates, captions and original fields
processing_config.json        # input identities, code hashes, runtime, exact transform
_state/partN.jsonl             # durable latest-attempt journal, one writer per part
_state/partN.status.json       # live per-part progress
_scratch/partN/                # one source MP4 per active worker; temporary only
logs/<UTC-start-time>.log
logs/exit_code                # written when batch.sh exits; 0 means success
progress.json                 # live global counts, refreshed at least every 30 s
run.json                      # running / succeeded / incomplete
samples.jsonl                 # successes only, reconciled after processing finishes
failures.jsonl                # latest unresolved sample failures after reconciliation
summary.json                  # candidate = success + failure + pending
commit_validation.json        # full per-file readback completed before every commit
validation.json               # explicit verify result (pilot includes source comparison)
```

`sample_id` and `video` equal the original CSV `video` field, including the full
filename and any clip suffix. Output filenames are unchanged. Original clip
suffixes are identifiers, not claims about the new frame count. `video_path` is
relative to the dataset root. The original part group, actual archive filenames,
ZIP member, CRC, source MP4 SHA256, frame count, exact FPS and original first PTS
remain under `source`; **all nine original CSV fields** remain under
`source.annotation`. Original quality scores are not recomputed or relabeled.

Top-level processed metadata includes `width`, `height`, `num_frames`, `fps_num`,
`fps_den`, `fps`, `duration_seconds`, `pts_span_seconds`, `output_sha256`,
`output_bytes`, `processing`, and `validation`. JSONL is an adapter for UMM_v3,
not the official OpenVid annotation format. The current loader reads JSONL but
its later sampling still uses training configuration; this corpus alone does
not enable per-sample-FPS training.

## Runtime and tests

Use the existing `.venv-npu/bin/python` with automatic torch NPU loading disabled.
`run.sh` selects that interpreter and sets CPU library thread counts to one.
Despite the environment name, this pipeline uses CPU only. The exact Python,
torch, numpy, PyAV and FFmpeg library versions are frozen in processing_config.
`requirements.lock.txt` records the three directly used third-party packages.

```bash
cd /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3
TORCH_DEVICE_BACKEND_AUTOLOAD=0 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
  .venv-npu/bin/python -m unittest discover -s data/preprocess/tests -v
```

Tests cover selection boundaries, exact identity/caption preservation, split
ZIP reads and CRC corruption, torn journal recovery, numbered-frame selection,
nonzero source time origins, odd/even lengths, B-frame presentation order,
pre-encoder/decoded RGB hash checks, lossy numbered-frame correspondence, and
actual MPEG-4 Part 2 encoding/MP4 timing for all 207 eligible rational rates.
The nine tests also exercise committed-file skipping, orphan rebuilding,
corrupted-output retries, exclusive locking, and batch-script exit status.
`tests/mpeg4.log` records the executed MPEG-4 test run; its injected failure
message belongs to the deliberate retry test.

## Pilot, production and tmux

The ten-case pilot deterministically includes the five common FPS groups,
161-frame and 162-frame boundaries, the longest source, the slowest source, and
an uncommon FPS from a split ZIP. It uses a separate `_pilot10` output root.
After normal per-file validation, an independent second source decode compares
every output frame `j` with resized original frame `2*j`, checks matching times,
reconstructs the exact pre-encoder input hash, measures compression error, and
compares every original annotation field. Full processing refuses to start
without ten passing pilot records and matching code, inputs and runtime.
The completed pilot has 2,560 output frames in ten MP4 files. Its full independent
audit is in `_pilot10/validation.json`; `pilot_samples.csv` lists the ten cases and
`pilot_report.json` records observed size and the limitations of extrapolation.
These files live inside the sibling pilot dataset directory, not this code folder.
The `data.preprocess.validate_mpeg4_pilot` module checks all ten source IDs, captions,
pre-encoder RGB hashes, encoded bytes and RGB errors against the reviewed
`codec_comparison_10/videos/mpeg4_q2` experiment. Its result is `benchmark_validation.json`.

```bash
tmux new-session -d -s openvidhd-mpeg4-pilot -c /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3 \
  'bash data/preprocess/batch.sh pilot /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_960x540_selected_pilot10 4'
tmux set-window-option -t openvidhd-mpeg4-pilot remain-on-exit on

# After pilot completion, check the reviewed MPEG-4 comparison and write its report.
TORCH_DEVICE_BACKEND_AUTOLOAD=0 .venv-npu/bin/python -m data.preprocess.validate_mpeg4_pilot \
  --output /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_960x540_selected_pilot10

# Run only after inspecting the passed pilot validation.json and benchmark_validation.json.
tmux new-session -d -s openvidhd-mpeg4 -c /inspire/sj-ssd3/global_user/lipengfei-253107010003/projs/unified_mm/UMM_v3 \
  'bash data/preprocess/batch.sh full /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_960x540_selected 16 /inspire/sj-ssd3/project/high-dimensionaldata/public/datasets/openvidhd_960x540_selected_pilot10'
tmux set-window-option -t openvidhd-mpeg4 remain-on-exit on
tmux attach -t openvidhd-mpeg4
```

Detach with `Ctrl-b d`. tmux preserves the process across terminal/SSH closure;
host/container termination still requires restarting the command. `progress.json`
and per-part journals survive restart. The actual cgroup CPU quota here is 32
cores, even though the host reports 192 CPUs; the production example uses 16
workers, with one decode/resize/encode thread per worker. No GPU/NPU job is created.

## Recovery and audit commands

Rerun the same `batch.sh` command in tmux to resume. Do not edit implementation
files or update the runtime during a run: source/config/runtime fingerprints
must match, otherwise the command refuses to mix versions. Worker count can change.
Only one process command can hold the output root's flock lock.

On resume, each committed output is checked by file SHA256 before skipping.
Missing or changed files are rebuilt. An output left behind after rename but
before journal commit is rebuilt from the source, never trusted by existence.
An incomplete last journal append is discarded; malformed complete lines fail.
Latest attempt wins, so failure retries do not duplicate samples. Controlled
per-part scratch directories from interrupted extraction are cleaned on restart.

Create `<output>/STOP` to request a graceful stop after current samples. Delete
that file before resuming. Workers also stop if free storage drops below the
100 GiB reserve. Pending samples remain pending rather than being called successful.

```bash
bash data/preprocess/run.sh finalize --output /absolute/dataset/root
bash data/preprocess/run.sh verify --output /absolute/dataset/root
# Expensive independent re-read of the original source videos as well:
bash data/preprocess/run.sh verify --output /absolute/dataset/root --source-correspondence
```

`finalize` needs the run lock and rebuilds final manifests from journals. Full
processing already reads every encoded output back before commit, so another
full `verify` is optional unless corruption or a changed file is suspected.
Selection/processing failures are explicit; `processing_finished` means no
pending records, whereas `all_candidates_succeeded` additionally requires zero
failures. Inspect both rather than treating process creation as completion.

Pilot sizes and timing are estimates only: the deliberately diverse ten-case
set is not a random sample and cannot establish an accurate full-dataset size
or completion time. Use live throughput and disk usage during production.

## Historical RGB run and codec comparison (2026-10-29)

The user requested a pause before choosing between MPEG-4 Part 2 q=2 and H.264
CRF18. The old RGB root remains under the personal datasets directory as
`openvidhd_960x540_stride2`; its STOP marker remains present. The stopped run preserved
3,382 successful videos, zero failures and 212,060 pending videos. `pause.json`
records the user-requested state; the existing `run.json` says `incomplete` because
the full population has not been processed.

The user subsequently selected MPEG-4 Part 2 and the shared output path above.
Active transformation version 2 rebuilds videos from the original ZIP sources;
old RGB files are not mixed into the new corpus. The four original semantic
source files and verified hashes are preserved in `versions/lossless_rgb_v1/`.
The active implementation intentionally refuses the old dataset fingerprint.

At comparison time, the independent experiment left the then-frozen production
implementation unchanged. Its original artifacts are retained. Its scripts are:

- `compare_codecs.py`: all-frame RGB PSNR/MAE/SSIM, exact media checks, repeated
  encoder/decoder timings, preserved IDs/captions, and per-frame/per-clip evidence.
- `compare_codecs.sh`: the tmux-compatible experiment runner with log/exit status.
- `export_codec_frames.py`: exports exact benchmark-decoder pixels for figures,
  rechecking their squared/absolute errors against the measured frame rows.
- `plot_codec_comparison.py`: scientific comparison plots and ten identical-region
  visual comparisons, using the analysis venv's matplotlib.
- `validate_codec_comparison.py`: independently reconciles frame/clip/global
  metrics, all file hashes, codec formats, original captions and source mappings.
- `tests/test_compare_codecs.py`: three analytical controls for error arithmetic,
  aggregation, exact rational media timing and SSIM identity.

Run from the project root with `bash data/preprocess/compare_codecs.sh`. The tmux
session used for this experiment is `openvidhd-codec-compare`. Results are under
`data/analysis/results/OpenVidHD/codec_comparison_10/`, including the 20 encoded
videos, all 5,120 per-frame comparisons, 150 timing measurements, charts and
methodology notes. Three timings are recorded per clip/stage/mode, using one CPU
codec thread and warm file cache; aggregate throughput uses total frames divided
by the sum of per-clip median times. ZIP extraction/resizing/model computation are
outside these codec timings. Quality uses the final encoded repetition; raw file
sizes and hashes of every repetition are retained without assuming bit-identical
encoder outputs. The shared input RGB digest must stay unchanged.

PSNR and SSIM refer to RGB, including chroma-subsampling/colour-conversion error;
they are not a direct measure of downstream model quality. The small purposive
pilot does not establish full-dataset size, speed or training-quality effects.
