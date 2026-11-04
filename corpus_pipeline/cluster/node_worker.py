"""Fixed offline worker pool, one process per assigned accelerator, no HCCL."""
from __future__ import annotations
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time


def worker_module(root):
    from .tuned_configuration import configuration
    return ('corpus_pipeline.cluster.tuned_frontpass' if configuration(root) is not None
            else 'corpus_pipeline.cluster.frontpass')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--devices', type=int, default=16)
    parser.add_argument('--dtype', default='float32')
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    # A submission may wait in LOW queue. It must not start unqualified code.
    qualification = args.root / 'QUALIFIED.json'
    if not qualification.is_file():
        raise RuntimeError('qualification receipt missing; refusing production inference')
    import json
    receipt = json.loads(qualification.read_text())
    if receipt.get('success') is not True or receipt.get('dtype') != args.dtype:
        raise RuntimeError('requested production dtype is not qualified')
    from corpus_pipeline.production.common import file_sha
    if receipt.get('inference_code_sha256') != file_sha(Path(__file__).with_name('inference.py')):
        raise RuntimeError('inference implementation changed after numerical qualification')
    if not (args.root / 'queue_rpc.json').is_file():
        raise RuntimeError('central queue service must be configured before node startup')
    module = worker_module(args.root)
    import torch
    import torch_npu
    if torch.npu.device_count() != args.devices:
        raise RuntimeError('assigned NPU count differs from requested worker count')
    node_dir = args.root / 'nodes' / (socket.gethostname() + '-' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ'))
    node_dir.mkdir(parents=True, exist_ok=False)
    children = []; files = []; stop = False
    def terminate(*_):
        nonlocal stop
        stop = True
        for child in children:
            if child.poll() is None:
                child.terminate()
    signal.signal(signal.SIGTERM, terminate); signal.signal(signal.SIGINT, terminate)
    try:
        for index in range(args.devices):
            log = (node_dir / f'worker-{index:02d}.log').open('w')
            files.append(log)
            command = [sys.executable, '-m', module, 'worker',
                       '--root', str(args.root.resolve()), '--device', f'npu:{index}',
                       '--dtype', args.dtype, '--batch-size', str(args.batch_size), '--threads', str(args.threads)]
            children.append(subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT))
            # Spread large weight reads and model initialization across the node.
            time.sleep(0.5)
        while any(child.poll() is None for child in children):
            if stop or any(child.poll() not in (None, 0) for child in children):
                terminate(); break
            time.sleep(2)
        codes = [child.wait(timeout=60) for child in children]
        if stop or any(code != 0 for code in codes):
            raise RuntimeError('worker failed or node preempted; task leases enable recovery: ' + str(codes))
    finally:
        terminate()
        for child in children:
            if child.poll() is None:
                child.kill()
        for stream in files:
            stream.close()


if __name__ == '__main__':
    main()
