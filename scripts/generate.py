#!/usr/bin/env python3
"""Compatibility arguments for single-sample callers; computation lives in evaluations."""

import argparse
import json
from pathlib import Path
import shutil
import tempfile

from configs.config import load_config


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--task', choices=['t2v', 'v2t'], required=True)
    parser.add_argument('--prompt')
    parser.add_argument('--manifest')
    parser.add_argument('--index', type=int, default=0)
    parser.add_argument('--output', required=True)
    parser.add_argument('--set', action='append', default=[])
    args = parser.parse_args(argv)
    output = Path(args.output)
    if output.exists():
        raise FileExistsError(output)
    suffixes = {'.pt': 'latent', '.mp4': 'mp4'} if args.task == 't2v' else {'.json': 'text'}
    if output.suffix not in suffixes:
        raise ValueError(f'Expected one of {list(suffixes)}')
    from evaluations.runtime import EvaluationRuntime
    from evaluations.sampling import build_requests, sample_request

    config = load_config(args.config, args.set, purpose='eval')
    config['model']['checkpoint'] = args.checkpoint
    task = dict(name='preview', benchmark=args.task, actions=['sample'],
                sample={'outputs': [suffixes[output.suffix]]}, data={}, score={})
    if args.task == 't2v':
        if not args.prompt:
            raise ValueError('T2V requires --prompt')
        task['data']['prompts'] = [args.prompt]
    else:
        if not args.manifest:
            raise ValueError('V2T requires --manifest')
        task['data']['sources'] = [{'path': args.manifest, 'task': 'v2t'}]
    requests = build_requests(task, config)
    request = requests[args.index if args.task == 'v2t' else 0]
    # Legacy --index selects an input; its random seed remains the configured seed.
    request['seed'] = config['sampling']['seed']
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix='.sample-', dir=output.parent) as staging:
        row = sample_request(EvaluationRuntime(config), task, request, staging)
        shutil.move(str(Path(staging) / row['artifacts'][suffixes[output.suffix]]), str(output))
    print(json.dumps({'output': str(output), 'seed_policy': 'per_request_v1'}))


# Existing imports can migrate independently; there is only one implementation.
def video_layout(*args, **kwargs):
    from evaluations.sampling import video_layout as implementation
    return implementation(*args, **kwargs)


def write_video(*args, **kwargs):
    from evaluations.sampling import write_video as implementation
    return implementation(*args, **kwargs)


if __name__ == '__main__':
    main()
