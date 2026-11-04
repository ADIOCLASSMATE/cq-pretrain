"""Qualified tuned launcher; the frozen frontpass retains policy and task I/O.

The lease loop matches the frozen launcher, preserving heartbeat, retry and
commit semantics. Keeping a separate entrypoint permits audited rollback without
altering the historical qualified implementation or completed outputs.
"""
from __future__ import annotations
import argparse, json, os, random, socket, threading, time, traceback, uuid
from pathlib import Path
from .frontpass import BASE, POLICY, load_rules, process_task
from corpus_pipeline.production.common import atomic_json, digest, file_sha
from corpus_pipeline.production.records import snapshot
from corpus_pipeline.production.potential import compile_source_contracts

def apply_source_profile(worker, qualified, source, qualification_sha256):
    """Switch only between drained tasks and retain each task's actual settings."""
    from .tuned_configuration import select_source_profile
    profile = select_source_profile(qualified, source)
    config, aggregation = profile['configuration'], profile['aggregation']
    if any(worker.execution_identity.get(key) != value for key, value in config.items()):
        worker.configure(**config)
    worker.execution_identity.update(qualification_sha256=qualification_sha256,
        configuration=config, aggregation=aggregation, source=source,
        source_profile=source if source in qualified.get('source_profiles', {}) else 'default')
    return aggregation


def run_worker(root, *, device, dtype='float32', batch_size=32, threads=4, max_tasks=None, queue_backend='rpc'):
    from .tuned_inference import TunedResidentInferenceWorker
    from .tuned_configuration import configuration
    qualified = configuration(root)
    if qualified is None:
        raise RuntimeError("tuned execution receipt is required")
    if (dtype, batch_size, threads) != ("float32", 32, 4):
        raise RuntimeError("unexpected launcher arguments")
    if queue_backend == 'rpc':
        from . import queue_rpc as queue
        queue.health(root)
    elif queue_backend == 'local' and device == 'cpu':
        from . import queue
    else:
        raise ValueError('production workers require the central queue RPC')
    root = Path(root).resolve()
    plan = json.loads((root / 'plan.json').read_text())
    rules = load_rules(); snap = snapshot(plan['snapshot_root'])
    if plan['policy'] != POLICY or plan['snapshot_fingerprint'] != snap['fingerprint'] or plan['rules_sha256'] != digest(rules):
        raise ValueError('run/input/rule identity changed')
    if plan['thresholds_sha256'] != file_sha(BASE / 'detector_calibration/thresholds.json') or plan['layout_contracts_sha256'] != file_sha(BASE / 'prepared/config/layout_source_contracts.json'):
        raise ValueError('run threshold/layout identity changed')
    contracts = compile_source_contracts(json.loads((BASE / 'prepared/config/layout_source_contracts.json').read_text()))
    thresholds = json.loads((BASE / 'detector_calibration/thresholds.json').read_text())['review_thresholds']
    worker_id = f'{socket.gethostname()}-{device.replace(":", "_")}-{os.getpid()}-{uuid.uuid4().hex[:8]}'
    worker_root = root / 'workers' / worker_id
    worker_root.mkdir(parents=True)
    aggregation = qualified['aggregation']
    qualification_sha256 = file_sha(root / 'TUNED_QUALIFIED.json')
    with TunedResidentInferenceWorker(device=device, dtype=dtype, **qualified['configuration']) as worker:
        worker.execution_identity['qualification_sha256'] = qualification_sha256
        worker.execution_identity['aggregation'] = aggregation
        worker.execution_identity['configuration'] = qualified['configuration']
        atomic_json(worker_root / 'model_identity.json', worker.identity)
        if getattr(worker, 'execution_identity', None):
            atomic_json(worker_root / 'execution_identity.json', worker.execution_identity)
        done = 0
        while max_tasks is None or done < max_tasks:
            claim = queue.claim(root, worker_id)
            if claim is None:
                state = queue.status(root)
                atomic_json(worker_root / 'last_queue_status.json', state)
                if state['all_tasks_completed']:
                    break
                if state['counts']['pending'] or state['counts']['running']:
                    # Other workers may still own the tail tasks. Stay alive
                    # so a preempted owner's expired lease can be reclaimed.
                    # The global status scan only runs when claim found no work.
                    time.sleep(random.uniform(30, 45))
                    continue
                if state['counts']['failed']:
                    raise RuntimeError('queue ended with terminal failed tasks: ' + str(state['counts']['failed']))
                raise RuntimeError('queue has no claimable work but is not complete')
            stop = threading.Event(); heartbeat_errors = []
            def beat():
                while not stop.wait(30):
                    try:
                        queue.heartbeat(root, claim['task_id'], claim['token'])
                    except Exception as error:
                        heartbeat_errors.append(repr(error)); return
            thread = threading.Thread(target=beat, daemon=True); thread.start()
            try:
                aggregation = apply_source_profile(worker, qualified, claim['task']['source'], qualification_sha256)
                atomic_json(worker_root / 'execution_identity.json', worker.execution_identity)
                summary, artifacts = process_task(root, claim, worker, snap, rules, contracts, thresholds,
                                                  aggregation=aggregation)
                if heartbeat_errors:
                    raise RuntimeError('lost lease: ' + heartbeat_errors[0])
                queue.complete(root, claim['task_id'], claim['token'], artifacts=artifacts,
                               processed_records=summary['counts']['records'],
                               processed_images=summary['counts'].get('image_occurrences', 0))
                done += 1
                print(json.dumps({'worker': worker_id, 'task': claim['task_id'], **summary['counts'],
                                  'elapsed_seconds': summary['elapsed_seconds']}), flush=True)
            except Exception as error:
                traceback.print_exc()
                queue.fail(root, claim['task_id'], claim['token'], repr(error), retryable=True)
                raise
            finally:
                stop.set(); thread.join(timeout=2)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=['worker'])
    parser.add_argument('--root',required=True,type=Path)
    parser.add_argument('--device',required=True)
    parser.add_argument('--dtype',default='float32')
    parser.add_argument('--batch-size',type=int,default=32)
    parser.add_argument('--threads',type=int,default=4)
    parser.add_argument('--max-tasks',type=int)
    args=vars(parser.parse_args());args.pop('command');run_worker(**args)

if __name__=='__main__':main()
