"""Run the pinned official GenEval evaluator in its own Python 3.9 runtime.

Same settings as the evaluator's original harness: in-process data loading and
a fixed thread count. The pinned runtime ships torch 1.13 built for sm_37-sm_86,
so on newer GPUs (H100/H200, sm_90) it runs the unchanged evaluator on the CPU:
GENEVAL_DEVICE=cpu rewrites only the script's device selection, which is
checked by exact match so a different evaluator revision fails loudly.
Usage: python geneval_worker.py EVALUATE_IMAGES_PY IMAGE_DIR --outfile ... --model-path ...
"""
import os
import sys

THREADS = int(os.environ.get('GENEVAL_THREADS', '1'))
for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[name] = str(THREADS)

import cv2
import torch

cv2.setNumThreads(1)
torch.set_num_threads(THREADS)
torch.set_num_interop_threads(1)
OriginalDataLoader = torch.utils.data.DataLoader


class InProcessDataLoader(OriginalDataLoader):
    def __init__(self, *args, **kwargs):
        # The official color-crop loader passes num_workers by keyword.
        assert len(args) < 6
        kwargs['num_workers'] = 0
        super().__init__(*args, **kwargs)


torch.utils.data.DataLoader = InProcessDataLoader
DEVICE_LINES = ('DEVICE = "cuda" if torch.cuda.is_available() else "cpu"\nassert DEVICE == "cuda"\n')

script = sys.argv[1]
sys.argv = sys.argv[1:]
source = open(script).read()
if os.environ.get('GENEVAL_DEVICE', 'cuda') == 'cpu':
    if source.count(DEVICE_LINES) != 1:
        raise SystemExit('Unexpected GenEval evaluator: its device selection lines changed')
    source = source.replace(DEVICE_LINES, 'DEVICE = "cpu"\n')
code = compile(source, script, 'exec')
exec(code, {'__name__': '__main__', '__file__': script, '__builtins__': __builtins__})
