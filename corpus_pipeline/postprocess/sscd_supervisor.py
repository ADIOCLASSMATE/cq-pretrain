"""Resumable, locally detached SSCD orchestration; never deletes supervision.

Only one stage child is owned at a time. The index module owns its own immutable
checkpoints and global-index lock. A child inherits the supervisor lock so a
killed supervisor cannot accidentally leave an unprotected overlapping writer.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
import signal
import socket
import subprocess
import sys
import tempfile
import time

SCHEMA = 'corpus.sscd_supervisor.v1'
CONFIG_SCHEMA = 'corpus.sscd_supervisor.config.v1'
INDEX_SCHEMA = 'corpus.sscd_index.v1'
MODULE = 'corpus_pipeline.postprocess.sscd_index'
MODULE_PATH = Path('corpus_pipeline/postprocess/sscd_index.py')
SELF_PATH = Path('corpus_pipeline/postprocess/sscd_supervisor.py')
STAGES = ('catalog', 'train', 'build', 'merge', 'calibrate', 'search')


def require(condition, message):
    if not condition:
        raise ValueError(message)


def canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def digest(value):
    return hashlib.sha256(canonical(value).encode()).hexdigest()


def file_sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text())


def atomic_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            stream.write(canonical(value) + '\n'); stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


@contextmanager
def exclusive(path):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+') as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield stream


def checked_config(value):
    value = dict(value)
    require(value.get('schema_version') == CONFIG_SCHEMA, 'Invalid supervisor config schema')
    for key in ('reconciliation_root', 'output', 'python', 'workdir'):
        require(isinstance(value.get(key), str) and Path(value[key]).is_absolute()
                and '\0' not in value[key], 'Absolute configuration path required: ' + key)
    # Do not resolve the interpreter symlink: that would bypass its venv.
    require(Path(value['python']).is_file() and os.access(value['python'], os.X_OK), 'Python executable unavailable')
    require(not Path(value['output']).resolve().is_relative_to(Path(value['reconciliation_root']).resolve()),
            'Output must not overlap sealed reconciliation inputs')
    defaults = dict(nlist=16384, pq_m=64, pq_bits=8, coarse_hnsw=32, niter=20,
                    threads=14, train_count=1_000_000, block_rows=250_000,
                    checkpoint_shards=16, threshold=.75, query_count=128, top_k=10,
                    min_recall=.95, min_positive_recall=.99, search_batch_size=1024,
                    heartbeat_seconds=15, stop_grace_seconds=60)
    for key, default in defaults.items():
        value.setdefault(key, default)
    bounds = {'nlist': (1, 1_048_576), 'pq_m': (1, 512), 'pq_bits': (4, 8), 'coarse_hnsw': (0, 64),
              'niter': (1, 100), 'threads': (1, 64), 'train_count': (1, 100_000_000),
              'block_rows': (1, 1_000_000), 'checkpoint_shards': (1, 4096), 'query_count': (1, 4096),
              'top_k': (1, 65536), 'search_batch_size': (1, 1024), 'heartbeat_seconds': (1, 60),
              'stop_grace_seconds': (1, 300)}
    for key, (low, high) in bounds.items():
        require(type(value[key]) is int and low <= value[key] <= high, 'Invalid supervisor limit: ' + key)
    require(512 % value['pq_m'] == 0 and (value['coarse_hnsw'] == 0 or value['coarse_hnsw'] >= 4), 'Invalid PQ/HNSW config')
    for key, low in (('min_recall', .95), ('min_positive_recall', .99)):
        require(type(value[key]) in (int, float) and math.isfinite(value[key]) and low <= value[key] <= 1,
                'Recall gate cannot be lowered: ' + key)
    require(type(value['threshold']) in (int, float) and math.isfinite(value['threshold'])
            and -1 <= value['threshold'] <= 1, 'Invalid candidate threshold')
    value.setdefault('calibration_candidates', [
        {'nprobe': n, 'candidate_k': k, 'ef_search': max(128, n * 2)}
        for n, k in ((32, 256), (64, 256), (64, 512), (128, 512), (128, 1024), (256, 1024))])
    trials = value['calibration_candidates']
    require(isinstance(trials, list) and 1 <= len(trials) <= 32, 'Bounded calibration candidates required')
    previous = (0, 0, 0)
    for trial in trials:
        require(isinstance(trial, dict) and set(trial) == {'nprobe', 'candidate_k', 'ef_search'}, 'Invalid calibration candidate')
        n, k, ef = (trial[x] for x in ('nprobe', 'candidate_k', 'ef_search'))
        require(all(type(x) is int for x in (n, k, ef)) and 1 <= n <= value['nlist']
                and value['top_k'] <= k <= 65536 and n <= ef <= 1_048_576, 'Invalid calibration search bounds')
        require(all(a >= b for a, b in zip((n, k, ef), previous)) and (n, k, ef) != previous,
                'Calibration candidates must monotonically strengthen search')
        previous = (n, k, ef)
    value.setdefault('disk_headroom_bytes', dict(catalog=400_000_000_000, train=16_000_000_000,
        build=32_000_000_000, merge=32_000_000_000, calibrate=16_000_000_000, search=64_000_000_000))
    require(isinstance(value['disk_headroom_bytes'], dict) and set(value['disk_headroom_bytes']) == set(STAGES)
            and all(type(v) is int and v >= 0 for v in value['disk_headroom_bytes'].values()), 'Invalid stage disk headroom')
    value.setdefault('screening_snapshot', {})
    require(isinstance(value['screening_snapshot'], dict), 'screening_snapshot must be metadata object')
    return value


def runtime_identity():
    return {'host': socket.gethostname(), 'boot_id': Path('/proc/sys/kernel/random/boot_id').read_text().strip()}


def process_identity(pid):
    try:
        fields = Path(f'/proc/{pid}/stat').read_text().rsplit(')', 1)[1].split()
        argv = [part.decode() for part in Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0') if part]
        if fields[0] == 'Z':
            return None
        return {**runtime_identity(), 'pid': pid, 'start_ticks': int(fields[19]), 'argv': argv, 'argv_sha256': digest(argv)}
    except (OSError, ValueError, IndexError):
        return None


def process_alive(identity):
    if not isinstance(identity, dict) or identity.get('host') != socket.gethostname():
        return False
    current = process_identity(identity.get('pid', -1))
    return current is not None and all(current.get(k) == identity.get(k)
        for k in ('host', 'boot_id', 'pid', 'start_ticks', 'argv_sha256'))


def launched_identity(child, command):
    # Popen can return before exec; do not bind the temporary fork's old argv.
    deadline = time.monotonic() + 2
    while child.poll() is None and time.monotonic() < deadline:
        value = process_identity(child.pid)
        if value is not None and value['argv'] == command:
            return value
        time.sleep(.02)
    return None


def snapshot_metadata(value):
    references = []
    def visit(item):
        if isinstance(item, dict):
            if isinstance(item.get('path'), str) and isinstance(item.get('sha256'), str):
                path = Path(item['path'])
                require(path.is_absolute() and file_sha(path) == item['sha256'], 'Screening snapshot artifact changed')
                entry = {'path': str(path), 'sha256': item['sha256'], 'bytes': path.stat().st_size}
                if path.suffix == '.json' and entry['bytes'] <= 4 << 20:
                    entry['frozen_json'] = read_json(path)
                references.append(entry)
            for nested in item.values():
                visit(nested)
        elif isinstance(item, list):
            for nested in item:
                visit(nested)
    visit(value)
    return {'declared': value, 'verified_references': references,
            'semantics': 'Input screening metadata snapshot only; no source data or source screening label is rewritten.'}


def bind_inputs(config_path, config, directory):
    workdir = Path(config['workdir'])
    source_files = [workdir / MODULE_PATH, workdir / SELF_PATH]
    require((workdir / SELF_PATH).resolve() == Path(__file__).resolve(), 'Supervisor must run from the configured frozen runtime')
    completion = Path(config['reconciliation_root']) / 'COMPLETE.json'
    upstream = read_json(completion)
    require(upstream.get('schema') == 'corpus.asset_reconciliation.v1' and upstream.get('complete') is True
            and upstream.get('scope') == 'full_run', 'A sealed full reconciliation is required')
    identity = {'schema_version': SCHEMA, 'config': config, 'config_path': str(Path(config_path).resolve()),
        'config_sha256': file_sha(config_path),
        'runtime_modules': {str(path.relative_to(workdir)): file_sha(path) for path in source_files},
        'reconciliation': {'path': str(completion), 'sha256': file_sha(completion), 'scope': upstream['scope']},
        'candidate_edges_only': True, 'automatic_merge': False, 'training_records_deleted': 0}
    path = directory / 'INPUT.json'
    if path.exists():
        existing = read_json(path)
        require(existing['identity'] == identity, 'Supervisor input/config/runtime changed; use a new output directory')
        return existing
    bound = {'identity': identity, 'screening_snapshot': snapshot_metadata(config['screening_snapshot'])}
    atomic_json(path, bound)
    return bound


class Supervisor:
    def __init__(self, config_path):
        self.config_path = Path(config_path).absolute()
        self.config = checked_config(read_json(self.config_path))
        self.output = Path(self.config['output']).resolve()
        self.directory = self.output / 'supervisor'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.stop_requested = False
        self.child = None
        self.child_identity = None
        self.stage = None
        self.stage_key = None
        self.started = time.time()
        self.current_log = None
        self.input = None
        self.lock_stream = None
        self.completed = []
        self.last_verification_heartbeat = 0

    def unchanged(self):
        require(file_sha(self.config_path) == self.input['identity']['config_sha256'], 'Supervisor configuration changed while running')
        for relative, sha in self.input['identity']['runtime_modules'].items():
            require(file_sha(Path(self.config['workdir']) / relative) == sha, 'Runtime code changed while running')
        ref = self.input['identity']['reconciliation']
        require(file_sha(ref['path']) == ref['sha256'], 'Reconciliation completion changed while running')

    def progress(self, status, **extra):
        value = {'schema_version': SCHEMA, 'status': status, 'pid': os.getpid(),
            'process': process_identity(os.getpid()), 'child': self.child_identity,
            'stage': self.stage, 'stage_key': self.stage_key, 'stage_log': self.current_log,
            'updated_at_epoch': time.time(), 'started_at_epoch': self.started,
            'elapsed_seconds': time.time() - self.started, 'completed_stages': self.completed,
            'input_sha256': file_sha(self.directory / 'INPUT.json') if self.input is not None else None,
            'full_query_completed': False, 'candidate_edges_only': True,
            'near_duplicates_resolved': False, 'automatic_merge': False, 'training_records_deleted': 0, **extra}
        atomic_json(self.directory / 'PROGRESS.json', value)
        return value

    def stage_identity(self, stage, **kwargs):
        return {'schema': INDEX_SCHEMA, 'stage': stage,
                'code_sha256': self.input['identity']['runtime_modules'][str(MODULE_PATH)], **kwargs}

    def spec(self, stage, trial=None, calibration=None):
        cfg, out = self.config, self.output
        args = [cfg['python'], '-m', MODULE, stage, '--output', str(out), '--threads', str(cfg['threads'])]
        if stage == 'catalog':
            identity = self.stage_identity(stage, input=str(Path(cfg['reconciliation_root']).resolve()),
                input_sha256=self.input['identity']['reconciliation']['sha256'], scope='full_run', block_rows=cfg['block_rows'])
            args += ['--input', cfg['reconciliation_root'], '--block-rows', str(cfg['block_rows'])]
            path = out / 'catalog/COMPLETE.json'
        else:
            catalog_path = out / 'catalog/COMPLETE.json'
            data = read_json(catalog_path); catalog_sha = file_sha(catalog_path)
            require(data.get('complete') is True and data.get('scope') == 'full_run', 'Catalog is not fully sealed')
            if stage == 'train':
                params = {k: cfg[k] for k in ('nlist', 'pq_m', 'pq_bits', 'coarse_hnsw', 'niter', 'threads')}
                params['seed'] = 20261023
                identity = self.stage_identity(stage, catalog_sha256=catalog_sha, config=params,
                                               train_count=min(cfg['train_count'], data['vectors']))
                for key in ('nlist', 'pq_m', 'pq_bits', 'coarse_hnsw', 'niter', 'train_count'):
                    args += ['--' + key.replace('_', '-'), str(cfg[key])]
                path = out / 'train/COMPLETE.json'
            elif stage == 'build':
                identity = self.stage_identity(stage, catalog_sha256=catalog_sha, train_sha256=file_sha(out / 'train/COMPLETE.json'))
                path = out / 'shards/COMPLETE.json'
            elif stage == 'merge':
                identity = self.stage_identity(stage, catalog_sha256=catalog_sha, shards_sha256=file_sha(out / 'shards/COMPLETE.json'))
                args += ['--checkpoint-shards', str(cfg['checkpoint_shards'])]
                path = out / 'merged/COMPLETE.json'
            elif stage == 'calibrate':
                params = {k: cfg[k] for k in ('threshold', 'query_count', 'top_k', 'min_recall', 'min_positive_recall', 'threads')}
                params.update(trial, seed=8273)
                identity = self.stage_identity(stage, catalog_sha256=catalog_sha, index_sha256=file_sha(out / 'merged/COMPLETE.json'), parameters=params)
                for key in ('threshold', 'query_count', 'top_k', 'min_recall', 'min_positive_recall'):
                    args += ['--' + key.replace('_', '-'), str(cfg[key])]
                for key, value in trial.items():
                    args += ['--' + key.replace('_', '-'), str(value)]
                path = out / 'calibration' / digest(identity)[:16] / 'COMPLETE.json'
            else:
                identity = self.stage_identity(stage, catalog_sha256=catalog_sha, index_sha256=file_sha(out / 'merged/COMPLETE.json'),
                    calibration_sha256=file_sha(calibration), query_start=0, query_end=data['vectors'], batch_size=cfg['search_batch_size'])
                args += ['--calibration', str(calibration), '--query-start', '0', '--batch-size', str(cfg['search_batch_size'])]
                path = out / 'search' / digest(identity)[:16] / 'COMPLETE.json'
        return args, path, identity

    def validate_stage(self, stage, path, identity):
        value = read_json(path)
        require(value.get('complete') is True, 'Stage has no successful completion: ' + stage)
        if stage == 'catalog':
            require(value.get('identity_sha256') == digest(identity) and value.get('input_sha256') == identity['input_sha256']
                    and value.get('scope') == 'full_run' and value.get('vectors', 0) > 0, 'Catalog completion binding mismatch')
        else:
            require(value.get('identity') == identity, 'Stage completion identity mismatch: ' + stage)
        if stage == 'search':
            require(value.get('scope') == 'full_run' and value.get('query_count') == identity['query_end']
                    and identity['query_start'] == 0 and value.get('automatic_merge') is False
                    and value.get('training_records_deleted') == 0, 'Search did not cover the full candidate query scope')
        return value

    def calibration_passed(self, value):
        params = value['identity']['parameters']
        require(params['min_recall'] >= .95 and params['min_positive_recall'] >= .99, 'Calibration recall gate was lowered')
        return (value.get('complete') is True and value.get('passed') is True
            and value.get('qualified_for_full_run') is True and value.get('scope') == 'full_run'
            and type(value.get('recall_at_k')) in (int, float)
            and value['recall_at_k'] >= self.config['min_recall']
            and type(value.get('threshold_positive_recall')) in (int, float)
            and value['threshold_positive_recall'] >= self.config['min_positive_recall']
            and value.get('threshold_positive_count', 0) > 0)

    def verify_checkpoint_tree(self, path, root, expected_sha=None):
        """Verify leaves, not just COMPLETE files, only at reuse/finalization."""
        from .sscd_index import bound_path
        root = Path(root).resolve()
        seen = {}
        def checksum(file):
            hasher = hashlib.sha256()
            with file.open('rb') as stream:
                for block in iter(lambda: stream.read(8 << 20), b''):
                    hasher.update(block)
                    if self.stop_requested or (self.directory / 'STOP').exists():
                        raise InterruptedError('Stop requested during checkpoint verification')
                    if time.monotonic() - self.last_verification_heartbeat >= self.config['heartbeat_seconds']:
                        self.last_verification_heartbeat = time.monotonic()
                        self.progress('verifying_checkpoint', verifying_artifact=str(file), receipts_verified=len(seen))
            return hasher.hexdigest()
        def visit(file, expected=None):
            file = Path(file).resolve()
            require(file.is_relative_to(root), 'Checkpoint receipt escaped output root')
            actual = checksum(file)
            require(expected is None or actual == expected, 'Checkpoint receipt SHA mismatch: ' + str(file))
            if file in seen:
                require(seen[file] == actual, 'Checkpoint changed during validation')
                return
            seen[file] = actual
            receipt = read_json(file)
            for item in receipt.get('artifacts', []):
                artifact_path = bound_path(root, item, verify=False)
                require(checksum(artifact_path) == item['sha256'], 'Checkpoint artifact SHA mismatch: ' + str(artifact_path))
            for collection in ('blocks', 'shards', 'buckets'):
                for entry in receipt.get(collection, []):
                    visit(root / entry['receipt_path'], entry['receipt_sha256'])
            for entry in receipt.get('batches', []):
                visit(root / entry['path'], entry['sha256'])
        visit(path, expected_sha)

    def run_child(self, command, log_path):
        environment = os.environ.copy()
        environment['PYTHONPATH'] = self.config['workdir']
        for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
            environment[key] = str(self.config['threads'])
        environment.setdefault('TORCH_DEVICE_BACKEND_AUTOLOAD', '0')
        signal_time = None
        with log_path.open('ab', buffering=0) as stream:
            self.child = subprocess.Popen(command, cwd=self.config['workdir'], env=environment,
                stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                start_new_session=True, pass_fds=(self.lock_stream.fileno(),))
            self.child_identity = launched_identity(self.child, command)
            atomic_json(self.directory / 'CHILD.json', {'process': self.child_identity, 'command': command,
                        'stage_key': self.stage_key, 'log': str(log_path), 'input_sha256': file_sha(self.directory / 'INPUT.json')})
            while self.child.poll() is None:
                self.stop_requested |= (self.directory / 'STOP').exists()
                if self.stop_requested and signal_time is None:
                    # This Popen child/process group belongs to this supervisor.
                    try:
                        os.killpg(self.child.pid, signal.SIGTERM)
                    except ProcessLookupError:
                        pass
                    signal_time = time.monotonic()
                if signal_time is not None and time.monotonic() - signal_time >= self.config['stop_grace_seconds']:
                    try:
                        os.killpg(self.child.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass
                self.progress('stopping' if self.stop_requested else 'running', child_pid=self.child.pid)
                time.sleep(min(self.config['heartbeat_seconds'], 1 if self.stop_requested else 60))
            code = self.child.returncode
            atomic_json(self.directory / 'CHILD_EXIT.json', {'stage_key': self.stage_key, 'returncode': code,
                        'exited_at_epoch': time.time(), 'process': self.child_identity})
            self.child = None
            self.child_identity = None
            return code

    def stage_run(self, stage, key, *, trial=None, calibration=None):
        self.stage, self.stage_key = stage, key
        self.unchanged()
        command, receipt_path, identity = self.spec(stage, trial=trial, calibration=calibration)
        journal = self.directory / 'stages' / (key + '.json')
        if journal.exists():
            saved = read_json(journal)
            require(saved['input_sha256'] == file_sha(self.directory / 'INPUT.json') and saved['command'] == command
                    and saved['receipt']['path'] == str(receipt_path) and saved['receipt']['sha256'] == file_sha(receipt_path),
                    'Successful stage checkpoint changed: ' + key)
            if stage == 'catalog':
                upstream = self.input['identity']['reconciliation']
                self.verify_checkpoint_tree(upstream['path'], Path(upstream['path']).parent, upstream['sha256'])
            self.verify_checkpoint_tree(receipt_path, self.output, saved['receipt']['sha256'])
            value = self.validate_stage(stage, receipt_path, identity)
            self.completed.append(key)
            self.progress('running', reused_stage=key)
            return value, receipt_path
        require(shutil.disk_usage(self.output).free >= self.config['disk_headroom_bytes'][stage],
                'Insufficient disk headroom before stage: ' + stage)
        logs = self.directory / 'logs'; logs.mkdir(exist_ok=True)
        log_path = logs / (key + '.' + str(time.time_ns()) + '.log')
        self.current_log = str(log_path)
        self.progress('starting_stage', command=command)
        code = self.run_child(command, log_path)
        if self.stop_requested:
            raise InterruptedError('Stop requested; committed index checkpoints remain resumable')
        require(code == 0, 'Stage child exited unsuccessfully: ' + key + ', exit=' + str(code))
        self.unchanged()
        value = self.validate_stage(stage, receipt_path, identity)
        if stage == 'search':
            self.verify_checkpoint_tree(receipt_path, self.output)
        atomic_json(journal, {'schema_version': SCHEMA, 'complete': True,
            'input_sha256': file_sha(self.directory / 'INPUT.json'), 'command': command,
            'receipt': {'path': str(receipt_path), 'sha256': file_sha(receipt_path)},
            'child_returncode': 0, 'log': str(log_path), 'completed_at_epoch': time.time()})
        self.completed.append(key)
        self.progress('running', completed_stage=key)
        return value, receipt_path

    def execute(self):
        for stage in ('catalog', 'train', 'build', 'merge'):
            if self.stop_requested or (self.directory / 'STOP').exists():
                raise InterruptedError('Stop requested before next stage')
            self.stage_run(stage, stage)
        attempts = []
        selected = None
        for number, trial in enumerate(self.config['calibration_candidates']):
            if self.stop_requested or (self.directory / 'STOP').exists():
                raise InterruptedError('Stop requested before calibration')
            value, path = self.stage_run('calibrate', f'calibrate_{number:02d}', trial=trial)
            passed = self.calibration_passed(value)
            attempts.append({'parameters': trial, 'receipt': {'path': str(path), 'sha256': file_sha(path)},
                             'passed': passed, 'recall_at_k': value.get('recall_at_k'),
                             'threshold_positive_recall': value.get('threshold_positive_recall')})
            atomic_json(self.directory / 'CALIBRATION.json', {'input_sha256': file_sha(self.directory / 'INPUT.json'),
                'attempts': attempts, 'selected': str(path) if passed else None,
                'min_recall': self.config['min_recall'], 'min_positive_recall': self.config['min_positive_recall']})
            if passed:
                selected = path
                break
        if selected is None:
            return self.progress('await_calibration', reason='Configured stronger search candidates exhausted; recall gate remains unchanged',
                                 calibration_attempts=attempts)
        if self.stop_requested or (self.directory / 'STOP').exists():
            raise InterruptedError('Stop requested before full search')
        result, path = self.stage_run('search', 'search', calibration=selected)
        complete = {'schema_version': SCHEMA, 'complete': True,
            'input_sha256': file_sha(self.directory / 'INPUT.json'), 'full_query_completed': True,
            'query_count': result['query_count'], 'edge_rows': result['edge_rows'],
            'search_receipt': {'path': str(path), 'sha256': file_sha(path)},
            'calibration_receipt': {'path': str(selected), 'sha256': file_sha(selected)},
            'candidate_edges_only': True, 'near_duplicates_resolved': False,
            'automatic_merge': False, 'training_records_deleted': 0,
            'edge_rows_are_not_unique_pair_count': True, 'completed_at_epoch': time.time()}
        atomic_json(self.directory / 'COMPLETE.json', complete)
        return self.progress('complete', **{k: v for k, v in complete.items() if k != 'schema_version'})

    def run(self):
        with exclusive(self.directory / '.LOCK') as lock:
            self.lock_stream = lock
            try:
                self.input = bind_inputs(self.config_path, self.config, self.directory)
            except Exception as exc:
                return self.progress('failed', error_type=type(exc).__name__, error=str(exc), automatic_retry=False)
            atomic_json(self.directory / 'PROCESS.json', process_identity(os.getpid()))
            self.progress('starting')
            old = {}
            def stop(signum, frame):
                self.stop_requested = True
            for number in (signal.SIGTERM, signal.SIGINT):
                old[number] = signal.signal(number, stop)
            try:
                return self.execute()
            except InterruptedError as exc:
                return self.progress('stopped', error=str(exc), resumable=True)
            except Exception as exc:
                return self.progress('failed', error_type=type(exc).__name__, error=str(exc),
                                     automatic_retry=False, resumable_from_committed_checkpoints=True)
            finally:
                for number, handler in old.items():
                    signal.signal(number, handler)
                self.lock_stream = None


def status(config_path):
    config = checked_config(read_json(config_path))
    directory = Path(config['output']) / 'supervisor'
    progress = read_json(directory / 'PROGRESS.json') if (directory / 'PROGRESS.json').exists() else {'status': 'not_started'}
    process = read_json(directory / 'PROCESS.json') if (directory / 'PROCESS.json').exists() else None
    child = read_json(directory / 'CHILD.json').get('process') if (directory / 'CHILD.json').exists() else None
    return {**progress, 'supervisor_alive': process_alive(process), 'stage_child_alive': process_alive(child)}


def launch(config_path):
    path = Path(config_path).absolute(); config = checked_config(read_json(path))
    directory = Path(config['output']) / 'supervisor'; directory.mkdir(parents=True, exist_ok=True)
    with exclusive(directory / '.LAUNCH_LOCK'):
        current = status(path)
        if current.get('supervisor_alive') or current.get('stage_child_alive'):
            return {**current, 'launched': False, 'reason': 'Bound supervisor or stage child is already running'}
        # An inherited lock also protects a child on another host/PID namespace.
        try:
            with exclusive(directory / '.LOCK'):
                pass
        except BlockingIOError:
            return {**current, 'launched': False, 'reason': 'Supervisor/stage lock remains held'}
        command = [config['python'], '-m', 'corpus_pipeline.postprocess.sscd_supervisor', 'run', '--config', str(path)]
        environment = os.environ.copy(); environment['PYTHONPATH'] = config['workdir']
        with (directory / 'supervisor.log').open('ab', buffering=0) as log:
            child = subprocess.Popen(command, cwd=config['workdir'], env=environment, stdin=subprocess.DEVNULL,
                                     stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        identity = launched_identity(child, command)
        atomic_json(directory / 'PROCESS.json', identity)
        return {'launched': True, 'pid': child.pid, 'process': identity, 'log': str(directory / 'supervisor.log')}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('launch', 'run', 'status'))
    parser.add_argument('--config', required=True)
    args = parser.parse_args(argv)
    if args.action == 'launch':
        result = launch(args.config)
    elif args.action == 'status':
        result = status(args.config)
    else:
        result = Supervisor(args.config).run()
    print(canonical(result), flush=True)
    if args.action == 'run' and result.get('status') != 'complete':
        raise SystemExit(3 if result.get('status') == 'await_calibration' else 130 if result.get('status') == 'stopped' else 1)


if __name__ == '__main__':
    main()
