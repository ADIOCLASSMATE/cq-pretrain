"""Q=64 production entrypoint: fresh rendezvous and auto-resume on every restart.

Low-priority jobs are preempted. Each platform restart negotiates a new launch
generation, logs under its own attempt directory, and resumes every rank from
the newest committed checkpoint, including optimizer, FP32 masters, renderer
EMA, RNG and the consumed-data cursor. Model, training and runtime settings
come from one experiment YAML; previous MOSS throughput is not a Wan benchmark.
"""
import argparse
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import uuid

from utils.distributed import atomic_json
from utils.checkpoint_catalog import durable_json

ROOT = Path(__file__).resolve().parent

def experiment(config):
    import yaml
    value=yaml.safe_load(Path(config).read_text())
    return value['training'],{k:str(v) for k,v in value.get('environment',{}).items()}


def read_json(path):
    try:
        return json.loads(path.read_text())
    except FileNotFoundError:
        return None


def join_attempt(root, node_rank, nodes, timeout=1800, interval=1.):
    """A fresh per-process nonce prevents acceptance of stale peer receipts."""
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    token = uuid.uuid4().hex
    generation = uuid.uuid4().hex if node_rank == 0 else None
    if node_rank == 0:
        atomic_json(root/'current.json', dict(generation=generation, nodes=nodes))
    deadline = time.monotonic()+timeout
    acknowledged = None
    while time.monotonic() < deadline:
        current = read_json(root/'current.json')
        if current is not None:
            if current['nodes'] != nodes:raise ValueError('Restart node topology changed')
            active = current['generation']
            if node_rank == 0 and active != generation:
                raise RuntimeError('Another controller replaced this launch generation')
            if active != acknowledged:
                atomic_json(root/f'ack_{node_rank:03d}.json', dict(generation=active, token=token))
                acknowledged = active
            cohort_path = root/'attempts'/active/'cohort.json'
            if node_rank == 0:
                peers = [read_json(root/f'ack_{i:03d}.json') for i in range(nodes)]
                if all(peer and peer['generation'] == active for peer in peers):
                    atomic_json(cohort_path, dict(generation=active, members=[peer['token'] for peer in peers]))
            cohort = read_json(cohort_path)
            if cohort and cohort['members'][node_rank] == token:
                return active
        time.sleep(interval)
    raise TimeoutError('Platform restart did not start every node of the fixed-size job')


def training_command(args, attempt):
    recipe,_=experiment(args.config)
    world = args.nodes*args.nproc_per_node
    if world % recipe['tp']:raise ValueError('World size must divide the recipe TP')
    dp = world//recipe['tp']
    if recipe['global_batch'] % (dp*recipe['micro_batch']):
        raise ValueError(f"global_batch {recipe['global_batch']} must divide DP={dp}")
    training = ['--config', str(args.config), '--output', str(attempt/'train'),
        '--data-root', str(args.data_root), '--data-mode', 'A',
        '--i2t-image-loss-probability',str(args.i2t_image_loss_probability),
        '--cfg-null-condition',args.cfg_null_condition,
        *(['--steps', str(args.steps)] if args.steps else ['--until-data-end']),
        '--warmup', str(min(10,max(0,args.steps-1)) if args.steps else 10),
        '--checkpoint',
        '--checkpoint-dir', str(args.output/'checkpoints'), '--auto-resume',
        '--save-every', str(args.save_every), '--keep-checkpoints', str(args.keep_checkpoints),
        '--save-first', '--save-final',
        '--validation-suite', str(args.validation_suite),
        '--validate-first', '--validate-final', '--validate-every', str(args.validate_every)]
    return [str(ROOT/'.venv-core17/bin/python'), str(ROOT/'launch_job.py'),
            '--nodes', str(args.nodes), '--node-rank', str(args.node_rank),
            '--nproc-per-node', str(args.nproc_per_node),
            '--rendezvous', str(attempt/'rendezvous'), '--', *training]


def run(args):
    recipe,environment_overrides=experiment(args.config)
    args.output.mkdir(parents=True, exist_ok=True)
    coordination = args.output/'coordination'
    coordination.mkdir(exist_ok=True)
    # Locks are released by the kernel even after SIGKILL. Never let two
    # platform attempts write the same checkpoint stream concurrently.
    with (coordination/f'node_{args.node_rank:03d}.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (args.output/'complete.json').exists():
            print(json.dumps(dict(status='already_complete', output=str(args.output))), flush=True)
            return 0
        generation = join_attempt(coordination, args.node_rank, args.nodes)
        attempt = args.output/'attempts'/generation
        attempt.mkdir(parents=True, exist_ok=True)
        command = training_command(args, attempt)
        atomic_json(attempt/f'launch_node_{args.node_rank:03d}.json',
                    dict(command=command, generation=generation, recipe=recipe))
        print(json.dumps(dict(status='launching', generation=generation, command=command)), flush=True)
        environment = dict(environment_overrides, **os.environ)
        with (attempt/f'node_{args.node_rank:03d}.log').open('a') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT,
                                     start_new_session=True, env=environment, cwd=str(ROOT))
            stopped = [False]
            def terminate(signum, frame):
                stopped[0] = True
                if child.poll() is None:
                    try:os.killpg(child.pid, signal.SIGTERM)
                    except ProcessLookupError:pass
            old_handlers = {sig:signal.signal(sig, terminate) for sig in (signal.SIGTERM, signal.SIGINT)}
            try:
                while child.poll() is None:
                    current = read_json(coordination/'current.json')
                    peer_states = [read_json(attempt/f'exit_node_{i:03d}.json') for i in range(args.nodes)]
                    if stopped[0] or current['generation'] != generation or any(p and p['exit_code'] for p in peer_states):
                        terminate(signal.SIGTERM, None)
                        try:child.wait(timeout=30)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                        break
                    time.sleep(1)
                code = child.wait()
                if stopped[0] and code == 0:code = 143
            finally:
                for sig, handler in old_handlers.items():signal.signal(sig, handler)
        atomic_json(attempt/f'exit_node_{args.node_rank:03d}.json', dict(exit_code=code))
        if code:return code if code > 0 else 128-code
        deadline = time.monotonic()+1800
        while time.monotonic() < deadline:
            peers = [read_json(attempt/f'exit_node_{i:03d}.json') for i in range(args.nodes)]
            if any(p and p['exit_code'] for p in peers):return 1
            if all(p is not None for p in peers):
                if args.node_rank == 0:
                    durable_json(args.output/'complete.json',
                                 dict(generation=generation, status='finite_dataset_complete'))
                print(json.dumps(dict(status='complete', generation=generation)), flush=True)
                return 0
            time.sleep(1)
        raise TimeoutError('Training exited but peer nodes did not complete')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--config', type=Path,
                        default=ROOT/'configs/qwen35_2b_wan.yaml')
    parser.add_argument('--data-root', type=Path, required=True,
                        help='Published corpus directory or mixed-source YAML')
    parser.add_argument('--validation-suite', type=Path, default=ROOT/'validation_assets/suite.json')
    parser.add_argument('--i2t-image-loss-probability',type=float,default=0.,
                        help='Legacy MOSS sampling only; Wan always disables I2T image loss')
    parser.add_argument('--cfg-null-condition',choices=('empty_t2i','i2t_prefix'),default='empty_t2i',
                        help='Wan requires empty_t2i; i2t_prefix is for legacy MOSS checkpoints')
    parser.add_argument('--nodes', type=int, default=int(os.environ.get('PET_NNODES', '8')))
    parser.add_argument('--node-rank', type=int, default=int(os.environ.get('PET_NODE_RANK', '0')))
    parser.add_argument('--nproc-per-node', type=int, default=16)
    parser.add_argument('--save-every', type=int, default=100)
    parser.add_argument('--keep-checkpoints', type=int, default=3)
    parser.add_argument('--validate-every', type=int, default=500)
    parser.add_argument('--steps', type=int, default=0,
                        help='Stop after this many steps instead of consuming the dataset; '
                             'for bounded probes such as verifying a resume end to end')
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    args.output = args.output.resolve()
    recipe,environment_overrides=experiment(args.config)
    if not math.isfinite(args.i2t_image_loss_probability) or not 0<=args.i2t_image_loss_probability<=1:
        parser.error('Require finite 0 <= i2t-image-loss-probability <= 1')
    if args.nodes < 1 or not 0 <= args.node_rank < args.nodes or args.nproc_per_node < 1:
        parser.error('Invalid fixed-size node topology')
    if args.save_every < 1 or args.keep_checkpoints < 1 or args.validate_every < 1:
        parser.error('Require positive checkpoint interval, retention and validation interval')
    for path in (args.config, args.data_root, args.validation_suite):
        if not path.exists():parser.error(f'Missing required path: {path}')
    if args.dry_run:
        command = training_command(args, args.output/'attempts'/'DRY_RUN')
        command.insert(command.index('--'), '--dry-run')
        print(json.dumps(dict(recipe=recipe, environment=environment_overrides,
            world=args.nodes*args.nproc_per_node, dp=args.nodes*args.nproc_per_node//recipe['tp'],
            grad_accum=recipe['global_batch']//(args.nodes*args.nproc_per_node//recipe['tp']),
            output=str(args.output), command=command), indent=2), flush=True)
        return subprocess.call(command, env=dict(environment_overrides, **os.environ), cwd=str(ROOT))
    return run(args)


if __name__ == '__main__':
    sys.exit(main())
