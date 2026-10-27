"""YAML-driven sample/score dispatcher, with isolated attempts and resumable artifacts."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import uuid

from configs.config import load_config, save_config
from configs.evaluation import load_tasks, resolve_evaluation_paths
from evaluations.artifacts import digest, file_identity, read_receipt, consolidate
from evaluations.benchmarks import get_benchmark
from utils.checkpoint_catalog import DirectoryLock, durable_json
from utils.run_logging import create_attempt, RunLogger, timestamp


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument('--config', help='Model/data/runtime YAML; optional with --run-dir')
    result.add_argument('--checkpoint', help='Exported UMM weights, not optimizer checkpoint directory')
    result.add_argument('--task-config', action='append', default=[], help='Evaluation task YAML (repeatable)')
    result.add_argument('--job', action='append', default=[], help='Run only these task names')
    result.add_argument('--actions', nargs='+', choices=['sample', 'score'])
    result.add_argument('--eval-id')
    result.add_argument('--run-dir', help='Exact existing evaluation root for score or resume')
    result.add_argument('--resume', action='store_true', help='Reuse verified artifacts from the same eval run')
    result.add_argument('--set', action='append', default=[])
    result.add_argument('--dry-run', action='store_true', help='Resolve task plan without loading models or writing results')
    result.add_argument('--max-new-samples', type=int, help='Stop this attempt after this many new samples globally')
    result.add_argument('--print-environment', action='store_true', help=argparse.SUPPRESS)
    result.add_argument('--worker-plan', help=argparse.SUPPRESS)
    return result


def resolve(args):
    saved = None
    if args.run_dir:
        run_dir = Path(args.run_dir).expanduser().resolve()
        saved = json.loads((run_dir / 'run.json').read_text())
        args.config = args.config or str(run_dir / 'config.yaml')
        args.eval_id = args.eval_id or saved['eval_id']
    if not args.config:
        raise ValueError('Provide --config or an existing --run-dir')
    config = load_config(args.config, args.set, purpose='eval')
    if args.checkpoint:
        config['model']['checkpoint'] = str(Path(args.checkpoint).expanduser().resolve())
    tasks = (deepcopy(saved['tasks']) if saved and not args.task_config
             else load_tasks(args.task_config, config))
    if args.run_dir:
        config['evaluation']['output_dir'] = str(run_dir)
    eval_id = args.eval_id or config['evaluation'].get('eval_id') or (
        'eval_' + datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '_' + uuid.uuid4().hex[:8])
    config['evaluation']['eval_id'] = eval_id
    paths = resolve_evaluation_paths(config, eval_id)
    selected = set(args.job or [task['name'] for task in tasks])
    if selected - {task['name'] for task in tasks}:
        raise ValueError(f'Unknown task names: {sorted(selected)}')
    if args.max_new_samples is not None and args.max_new_samples < 1:
        raise ValueError('--max-new-samples must be positive')
    for task in tasks:
        adapter = get_benchmark(task['benchmark'])
        task['_actions'] = list(args.actions or task['actions']) if task['name'] in selected else []
        if 'score' in task['_actions'] and adapter.scorer is None:
            raise ValueError(f"{task['benchmark']} has no scorer; configure a benchmark adapter")
        if 'sample' in task['_actions'] and adapter.sampler is None:
            raise ValueError(f"{task['benchmark']} has no sampling action; loss uses teacher-forced score on dataset")
    needs_model = any('sample' in task['_actions'] or (
        'score' in task['_actions'] and get_benchmark(task['benchmark']).score_requires_model) for task in tasks)
    return config, tasks, paths, needs_model


def semantic_identity(config, tasks):
    """Management paths/rank count/actions do not change the generated experiment."""
    model = deepcopy(config['model'])
    checkpoint = model.pop('checkpoint', None)
    if checkpoint:
        checkpoint = file_identity(checkpoint)
    inputs = {}
    for task in tasks:
        sources = task.get('data', {}).get('sources')
        if sources is None:
            direct = any(key in task.get('data', {}) for key in ('prompt', 'prompts'))
            sources = ([] if direct or task['benchmark'] == 'vbench' else
                       config['data'].get('validation_sources', []))
        for source in sources:
            path = str(Path(source['path']).resolve())
            inputs[path] = file_identity(path)
            # T2V sampling needs captions/layout only. V2T and supervised loss
            # also depend on observed content; detect replacement at the same path.
            if task['benchmark'] == 'loss' or task['benchmark'] == 'v2t' or task.get('sample', {}).get('mode') in {'v2t', 'mixed'}:
                raw = Path(path).read_text()
                records = (json.loads(raw) if Path(path).suffix == '.json' else
                           [json.loads(line) for line in raw.splitlines() if line.strip()])
                if isinstance(records, dict):
                    records = [records]
                for row in records:
                    row_task = row.get('task', source.get('task'))
                    if task['benchmark'] != 'loss' and row_task == 't2v':
                        continue
                    videos = row.get('videos', [dict(path=row.get('video_path'), frames_path=row.get('frames_path'),
                                                  qwen_cache=row.get('qwen_cache'), wan_cache=row.get('wan_cache'))])
                    for video in videos:
                        keys = ['qwen_cache']
                        if task['benchmark'] == 'loss' and row_task == 't2v':
                            keys.append('wan_cache')
                        # Fingerprint available raw content as well: prefer_cache
                        # can fall back to it after cache validation.
                        keys.append('frames_path' if video.get('frames_path') else 'path')
                        for key in keys:
                            if video.get(key):
                                asset = (Path(path).parent / video[key]).resolve()
                                if asset.is_file() and str(asset) not in inputs:
                                    inputs[str(asset)] = file_identity(asset)
    training = config['training']
    return dict(checkpoint=checkpoint, model=model, data=config['data'], flow=config['flow'],
                sampling=config['sampling'], codec=config['codec'],
                evaluation={key: config['evaluation'][key] for key in ('seed', 'max_batches')},
                runtime={key: training[key] for key in ('device', 'mixed_precision', 'batch_size', 'deterministic')},
                tasks=[{**{key: task.get(key) for key in ('name', 'benchmark', 'data', 'sample')},
                        'score': {key: value for key, value in task.get('score', {}).items()
                                  if get_benchmark(task['benchmark']).score_requires_model
                                  if key not in {'python', 'vbench_root', 'environment', 'dry_run'}}}
                       for task in tasks], inputs=inputs)


def make_plan(config, tasks, directory, attempt, maximum):
    jobs = []
    remaining = maximum
    for original in tasks:
        task = deepcopy(original)
        target = directory / task['name']
        job = dict(task=task, directory=str(target), requests=[], pending=[])
        # Plan sampling even in score-only mode when an original manifest exists.
        if 'sample' in task['_actions']:
            adapter = get_benchmark(task['benchmark'])
            job['requests'] = adapter.sampler(task, config, target)
            job['sampling_task'] = (adapter.normalize_sample(deepcopy(task), config)
                                    if adapter.normalize_sample else task)
            from evaluations.sampling import validate_sampling_task
            validate_sampling_task(job['sampling_task'])
        elif (target / 'requests.json').exists():
            job['requests'] = json.loads((target / 'requests.json').read_text())
        if job['requests']:
            target.mkdir(parents=True, exist_ok=True)
            durable_json(target / 'requests.json', job['requests'])
            pending = [request for request in job['requests'] if read_receipt(target, request) is None]
            if 'sample' in task['_actions']:
                job['pending'] = pending if remaining is None else pending[:remaining]
                if remaining is not None:
                    remaining -= len(job['pending'])
            job['sample_complete'] = not pending
        jobs.append(job)
    return dict(config=config, jobs=jobs, output_dir=str(directory), attempt=str(attempt))


def worker(plan):
    import torch.distributed as dist
    from evaluations.runtime import EvaluationRuntime
    from evaluations.sampling import sample_request
    from evaluations.artifacts import commit_receipt
    from evaluations.scoring import get_scorer

    config = plan['config']
    rank, world = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1))
    runtime = EvaluationRuntime(config)
    # Sampling has independent replicas and no collectives. Loss alone needs DP reduction.
    reduction = world > 1 and any('score' in j['task']['_actions'] and get_benchmark(j['task']['benchmark']).score_requires_model
                                 for j in plan['jobs'])
    if reduction:
        device = runtime.device
        dist.init_process_group(backend='hccl' if device.type == 'npu' else 'gloo')
    try:
        for job in plan['jobs']:
            task = job['task']
            for request in job['pending'][rank::world]:
                result = sample_request(runtime, job.get('sampling_task', task), request, job['directory'])
                row = commit_receipt(job['directory'], request, result, rank=rank,
                                     attempt_id=Path(plan['attempt']).name)
                print(json.dumps(dict(event='sample', job=task['name'], id=request['id'], rank=rank,
                                      artifacts=row['artifacts'])), flush=True)
            adapter = get_benchmark(task['benchmark'])
            if adapter.score_requires_model and 'score' in task['_actions']:
                diagnostics = Path(plan['attempt']) / task['name']
                result = get_scorer(adapter.scorer)(runtime, task, diagnostics, distributed=reduction)
                if rank == 0:
                    durable_json(Path(plan['attempt']) / f"{task['name']}.metrics.json", result)
    finally:
        if dist.is_initialized():
            dist.destroy_process_group()


def launch(plan, count, attempt):
    plan_path = attempt / 'plan.json'
    durable_json(plan_path, plan)
    command = [sys.executable]
    if count > 1:
        command += ['-m', 'torch.distributed.run', '--standalone', '--nnodes=1', f'--nproc-per-node={count}']
    command += ['-m', 'evaluations.eval', '--worker-plan', str(plan_path)]
    durable_json(attempt / 'launch.json', dict(command=command, started_at=timestamp()))
    env = dict(os.environ, PYTHONUNBUFFERED='1')
    with (attempt / 'console.log').open('a') as log:
        process = subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                   text=True, errors='replace', bufsize=1, start_new_session=True)
        def forward(signum, _frame):
            if process.poll() is None:
                os.killpg(process.pid, signum)
        previous = {sig: signal.signal(sig, forward) for sig in (signal.SIGINT, signal.SIGTERM)}
        try:
            for line in process.stdout:
                log.write(line)
                log.flush()
                print(line, end='', file=sys.stderr, flush=True)
            code = process.wait()
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
    durable_json(attempt / 'launch_result.json', dict(exit_code=code, finished_at=timestamp()))
    if code:
        raise RuntimeError(f'Evaluation workers exited with {code}; see {attempt / "console.log"}')


def dispatch(args, config, tasks, paths, needs_model):
    directory, eval_id = paths['output_dir'], config['evaluation']['eval_id']
    if args.dry_run:
        for task in tasks:
            if 'sample' in task['_actions']:
                adapter = get_benchmark(task['benchmark'])
                adapter.sampler(task, config)
                from evaluations.sampling import validate_sampling_task
                validate_sampling_task(adapter.normalize_sample(deepcopy(task), config)
                                       if adapter.normalize_sample else task)
        print(json.dumps(dict(eval_id=eval_id, directories={k: str(v) for k, v in paths.items()},
                              tasks=tasks, needs_model=needs_model), indent=2))
        return 0
    if needs_model and not config['model'].get('checkpoint'):
        raise ValueError('Model evaluation requires an exported UMM --checkpoint')
    with DirectoryLock(directory):
        run_path = directory / 'run.json'
        previous = json.loads(run_path.read_text()) if run_path.exists() else None
        if not needs_model and previous:
            # Offline scoring consumes saved artifacts, never the original checkpoint
            # or dataset. Its environment may be a different machine entirely.
            original_tasks = {task['name']: task for task in previous['tasks']}
            if {task['name'] for task in tasks} != set(original_tasks):
                raise ValueError('Offline scoring must retain saved task definitions; use --job to select a task')
            stored_config = load_config(directory / 'config.yaml', purpose='eval')
            for section in ('model', 'data', 'flow', 'sampling', 'codec'):
                if digest(config[section]) != digest(stored_config[section]):
                    raise ValueError(f'Offline scoring cannot change {section}; use a new evaluation')
            for section, keys in {'training': ('device', 'mixed_precision', 'batch_size', 'deterministic'),
                                  'evaluation': ('seed', 'max_batches')}.items():
                if any(config[section][key] != stored_config[section][key] for key in keys):
                    raise ValueError(f'Offline scoring cannot change {section} computation settings')
            for task in tasks:
                original = original_tasks.get(task['name'], {})
                if any(task.get(key) != original.get(key) for key in ('benchmark', 'data', 'sample')):
                    raise ValueError('Offline scoring cannot change the saved sampling protocol')
            identity = previous['identity']
        else:
            identity = semantic_identity(config, tasks)
        if run_path.exists():
            if not (args.resume or args.run_dir):
                raise ValueError('Evaluation already exists; use --resume or a new --eval-id')
            if previous['identity_sha256'] != digest(identity):
                raise ValueError('Checkpoint, data, sampling or model settings changed; use a new eval_id')
        else:
            previous = dict(schema='umm_evaluation_v1', eval_id=eval_id, attempts=[])
        attempt = create_attempt(config, purpose='eval', eval_id=eval_id)
        logger = RunLogger(attempt, purpose='eval', eval_id=eval_id)
        count = config['evaluation'].get('num_processes') or config['distributed']['num_processes']
        logger.start(config, sys.argv[1:], {'num_processes': count})
        print(f'Evaluation results: {directory}\nEvaluation logs: {attempt}', file=sys.stderr, flush=True)
        save_config(config, directory / 'config.yaml')
        stored_tasks = [{key: value for key, value in task.items() if key != '_actions'} for task in tasks]
        run = dict(previous, identity=identity, identity_sha256=digest(identity), tasks=stored_tasks,
                   status='running', updated_at=timestamp())
        run.pop('error', None)
        run['attempts'].append(str(attempt))
        durable_json(run_path, run)
        results = deepcopy(previous.get('results', {}))
        try:
            plan = make_plan(config, tasks, directory, attempt, args.max_new_samples)
            compute = any(j['pending'] or ('score' in j['task']['_actions'] and get_benchmark(j['task']['benchmark']).score_requires_model)
                          for j in plan['jobs'])
            if compute:
                launch(plan, count, attempt)
            else:
                (attempt / 'console.log').touch()
            for job in plan['jobs']:
                task, target = job['task'], Path(job['directory'])
                if not task['_actions']:
                    continue
                target.mkdir(parents=True, exist_ok=True)
                completed = consolidate(target, job['requests']) if job['requests'] else []
                state = dict(benchmark=task['benchmark'], samples_expected=len(job['requests']),
                             samples_completed=len(completed), attempt_id=attempt.name,
                             completed_actions=list(results.get(task['name'], {}).get('completed_actions', [])))
                if job['pending'] and 'score' in state['completed_actions']:
                    # Scores depend on the committed artifact set; a repair or
                    # extension requires fresh scoring, even if generation is deterministic.
                    state['completed_actions'].remove('score')
                if job['requests'] and len(completed) == len(job['requests']) and 'sample' not in state['completed_actions']:
                    state['completed_actions'].append('sample')
                if job['requests'] and len(completed) != len(job['requests']):
                    state['status'] = 'incomplete'
                elif 'score' in task['_actions']:
                    adapter = get_benchmark(task['benchmark'])
                    if adapter.score_requires_model:
                        state['metrics'] = json.loads((attempt / task['name'] / 'metrics.json').read_text())
                        durable_json(target / 'metrics.json', state['metrics'])
                    else:
                        from evaluations.scoring import get_scorer
                        scoring_task = dict(task, _score_log_dir=str(attempt / task['name'] / 'score'))
                        state['metrics'] = get_scorer(adapter.scorer)(None, scoring_task, target)
                    state['score_signature'] = digest(dict(settings=task.get('score'),
                                                           artifacts=[row['integrity'] for row in completed]))
                    state['status'] = 'completed'
                    if 'score' not in state['completed_actions']:
                        state['completed_actions'].append('score')
                else:
                    state['status'] = 'completed'
                if set(task['actions']) - set(state['completed_actions']):
                    state['status'] = 'incomplete'
                durable_json(target / 'state.json', state)
                results[task['name']] = state
                run['results'] = results
                durable_json(run_path, run)
            run['status'] = ('incomplete' if len(results) != len(tasks) or
                             any(r['status'] == 'incomplete' for r in results.values()) else 'completed')
            run['results'] = results
            logger.finish(status=run['status'], step=None, results=results)
        except BaseException as exc:
            run.update(status='failed', error=f'{type(exc).__name__}: {exc}')
            logger.finish(status='failed', step=None, error=run['error'])
            raise
        finally:
            run['results'] = results
            run['updated_at'] = timestamp()
            durable_json(run_path, run)
        # Keep the pre-existing loss-only CLI JSON contract.
        only_loss = len(results) == 1 and next(iter(results.values())).get('benchmark') == 'loss'
        print(json.dumps(next(iter(results.values()))['metrics'] if only_loss else
                         dict(eval_id=eval_id, status=run['status'], output_dir=str(directory), tasks=results), indent=2))
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    if args.worker_plan:
        worker(json.loads(Path(args.worker_plan).read_text()))
        return 0
    config, tasks, paths, needs_model = resolve(args)
    if args.print_environment:
        print(config['distributed']['backend'] if needs_model else 'score')
        return 0
    return dispatch(args, config, tasks, paths, needs_model)


if __name__ == '__main__':
    sys.exit(main())
