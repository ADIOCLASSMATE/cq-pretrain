"""Bounded, deadline-aware CPU additions to an existing frozen SSCD fleet.

Only this controller's create intents can become owned extra notebooks. Original
fleet notebooks are never restarted or stopped here. Create is submitted at most
once per name: uncertain results are reconciled against the live platform.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shlex
import signal
import socket
import subprocess
import threading
import time
import uuid

VERSION = 'corpus.sscd_cpu_autoscale.v2'
_LOCAL_LOCKS = {}
_LOCAL_GUARD = threading.Lock()


def read(path): return json.loads(Path(path).read_text())
def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def save(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.' + uuid.uuid4().hex)
    with temporary.open('x') as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def lock(path, blocking=False):
    path = Path(path).resolve(); path.parent.mkdir(parents=True, exist_ok=True)
    with _LOCAL_GUARD: mutex = _LOCAL_LOCKS.setdefault(str(path), threading.Lock())
    acquired = mutex.acquire(blocking=blocking)
    if not acquired:
        yield False; return
    try:
        with path.open('a+') as stream:
            owned = False
            try:
                try:
                    fcntl.lockf(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB)); owned = True
                except BlockingIOError: pass
                yield owned
            finally:
                if owned: fcntl.lockf(stream, fcntl.LOCK_UN)
    finally: mutex.release()


def option(argv, name):
    if argv.count(name) != 1: raise ValueError('Resource option must occur exactly once: ' + name)
    return argv[argv.index(name) + 1]


def create_template(c, ledger):
    initial = {x['name']: x for x in ledger['instances'] if x.get('owned') and x['name'] in c['notebooks']}
    if set(initial) != set(c['notebooks']): raise ValueError('Initial fleet ownership missing')
    required = ('--account', '--workspace', '--project', '--group', '--quota', '--image')
    rows = [{flag: option(initial[name]['argv'], flag) for flag in required} for name in c['notebooks']]
    if any(x != rows[0] for x in rows): raise ValueError('Initial fleet resource profile differs')
    value = rows[0]
    if value['--account'] != c['account'] or value['--workspace'] != c['workspace']:
        raise ValueError('Fleet account/workspace mismatch')
    if value['--quota'] != '0,16,256' or '910B' not in value['--group']:
        raise ValueError('Autoscaler only supports this Ascend CPU 0,16,256 profile')
    argv = list(initial[c['notebooks'][0]]['argv'])
    if argv[:4] != ['inspire', '--json', 'notebook', 'create']: raise ValueError('Invalid create template')
    return argv


def scaling_decision(progress, *, now, reserved_instances, active_workers, max_instances,
                     last_scale_at=0, baseline_instances=8, initial_instances_target=None):
    """Linear throughput estimate is a trigger, not a completion guarantee."""
    result = {'add': 0, 'target_instances': reserved_instances, 'reason': 'observe',
              'reserved_instances': reserved_instances, 'active_workers': active_workers,
              'max_instances': max_instances, 'linear_scaling_assumption': True}
    if progress.get('phase') not in ('catalog', 'search'):
        return dict(result, reason='phase_not_parallel_scaling_target')
    if now - progress.get('updated_at', 0) > 180: return dict(result, reason='controller_heartbeat_stale')
    forecast = progress.get('forecast', {}); rate = forecast.get('throughput_units_per_second')
    valid_rate = (forecast.get('measurement_seconds', 0) >= 300
                  and forecast.get('measurement_observations', 0) >= 2
                  and type(rate) in (int, float) and math.isfinite(rate) and rate >= 0)
    initial_target = baseline_instances if initial_instances_target is None else initial_instances_target
    if type(initial_target) is not int or not baseline_instances <= initial_target <= max_instances:
        raise ValueError('Initial target must cover original fleet and fit maximum')
    if progress.get('phase') == 'search' and not valid_rate and reserved_instances < initial_target:
        # An explicit, benchmark-informed initial target is a warmup allowance,
        # not an estimated rate. No five-minute delay is needed between its
        # bounded four-instance waves; all other creation safeguards still apply.
        stats = progress.get('stats', {})
        remaining = forecast.get('phase_remaining')
        if stats.get('complete') or stats.get('ready_to_finalize') or (remaining is not None and remaining <= 0):
            return dict(result, reason='no_remaining_phase_work')
        target = initial_target
        total, complete = stats.get('total_units'), stats.get('complete_units', 0)
        if type(total) is int and type(complete) is int:
            if total <= complete: return dict(result, reason='no_remaining_phase_work')
            target = min(target, max(baseline_instances, total - complete - 1))
        add = min(4, max(0, target - reserved_instances))
        return dict(result, add=add, target_instances=target,
                    reason='explicit_initial_search_target' if add else 'capacity_or_useful_parallelism_limit',
                    initial_instances_target=initial_target, linear_scaling_assumption=False)
    if now - last_scale_at < 300: return dict(result, reason='five_minute_post_scale_observation')
    if not valid_rate:
        return dict(result, reason='need_five_minute_rate')
    if rate <= 0: return dict(result, reason='stalled_rate_restore_workers_before_expansion')
    if active_workers < 1: return dict(result, reason='no_live_worker_heartbeat')
    remaining = forecast.get('phase_remaining'); eta = forecast.get('phase_remaining_seconds')
    if remaining is None or remaining <= 0 or eta is None: return dict(result, reason='no_remaining_phase_work')
    seconds = progress['deadline_epoch'] - now
    budget = seconds - 10 * 3600 if progress['phase'] == 'catalog' else seconds * .7
    result.update(phase_remaining_seconds=eta, phase_budget_seconds=budget, rate=rate)
    if eta <= budget: return dict(result, reason='estimated_phase_fits_budget')
    target_workers = math.ceil(active_workers * eta / budget) if budget > 0 else max_instances + 1
    # active_workers includes the original coordinator-local worker. It is not a notebook reservation.
    target_instances = max(reserved_instances, target_workers - 1, baseline_instances)
    tasks = progress.get('stats', {}).get('total_units', 0) - progress.get('stats', {}).get('complete_units', 0)
    if tasks > 0: target_instances = min(target_instances, max(baseline_instances, tasks - 1))
    target_instances = min(max_instances, target_instances)
    add = min(4, max(0, target_instances - reserved_instances))
    return dict(result, add=add, target_instances=target_instances,
                reason='deadline_requires_capacity' if add else 'capacity_or_useful_parallelism_limit')


class Platform:
    """No raw CLI/SDK output or access URLs are persisted or printed."""
    def __init__(self, c): self.c = c
    def invoke(self, argv, timeout=120, **kwargs):
        try:
            value = subprocess.run(argv, capture_output=True, text=True, timeout=timeout, **kwargs)
            try: document = json.loads(value.stdout)
            except (ValueError, TypeError): document = None
            return {'returncode': value.returncode, 'document': document}
        except subprocess.TimeoutExpired: return {'returncode': None, 'uncertain': True, 'error_type': 'TimeoutExpired'}
        except OSError as exc: return {'returncode': None, 'error_type': type(exc).__name__}
    def list_name(self, name):
        c = self.c
        result = self.invoke(['inspire', '--json', 'notebook', 'list', '--workspace', c['workspace'],
                              '--account', c['account'], '--keyword', name, '--all'], timeout=60)
        doc = result.get('document')
        if result['returncode'] != 0 or not isinstance(doc, dict) or doc.get('success') is not True:
            raise RuntimeError('Live notebook lookup failed')
        items = doc.get('data', {}).get('items')
        if not isinstance(items, list): raise ValueError('Unexpected live notebook list schema')
        return [{k: row.get(k) for k in ('name', 'status', 'workspace', 'project', 'compute_group')}
                for row in items if str(row.get('name', '')).rstrip().casefold() == name.rstrip().casefold()]
    def create(self, argv): return self.invoke(argv)
    def transition(self, name, action):
        c = self.c; argv = ['inspire', '--json', 'notebook', action, name, '--account', c['account'], '--workspace', c['workspace']]
        if action == 'start': argv.append('--no-wait')
        return self.invoke(argv)
    def launch_worker(self, name, config_path):
        c = self.c
        argv = [c['python'], '-m', 'corpus_pipeline.postprocess.sscd_cluster_runner', 'launch-worker',
                '--config', str(config_path), '--worker-id', name]
        shell = 'cd ' + shlex.quote(c['runtime']) + ' && PYTHONPATH=' + shlex.quote(c['runtime']) + ' ' + shlex.join(argv)
        call = [c['sdk_python'], '-m', 'corpus_pipeline.cluster.notebook_exec', '--notebook', name,
                '--account', c['account'], '--workspace', c['workspace'], '--timeout', '45',
                '--max-output-bytes', '8192', '--command', shell]
        env = dict(os.environ, PYTHONPATH=c['runtime'])
        return self.invoke(call, timeout=100, cwd=c['runtime'], env=env)


def sanitized(result):
    doc = result.get('document') or {}
    return {'returncode': result.get('returncode'), 'uncertain': bool(result.get('uncertain')),
            'success': doc.get('success') is True, 'completed': doc.get('completed') is True,
            'error_type': result.get('error_type')}


class Autoscaler:
    def __init__(self, cluster_config, *, max_instances=None, initial_instances_target=None, platform=None):
        self.cp = Path(cluster_config).absolute(); self.c = read(self.cp); self.root = Path(self.c['root'])
        self.d = self.root / 'autoscale'; self.d.mkdir(parents=True, exist_ok=True); self.stop_event = threading.Event()
        for key in ('root', 'output', 'runtime', 'python', 'sdk_python', 'input', 'adopt'):
            if not Path(self.c[key]).is_absolute(): raise ValueError('Absolute path required: ' + key)
        for relative, expected in self.c['runtime_sha256'].items():
            if sha(Path(self.c['runtime']) / relative) != expected: raise ValueError('Frozen worker runtime changed')
        ledger = read(self.root / 'resources' / 'PLAN.json'); self.template = create_template(self.c, ledger)
        self.base_names = self.c['notebooks']; self.base_count = len(self.base_names)
        bound = self.d / 'INPUT.json'
        with lock(self.d / 'SETUP.lock', blocking=True):
            if bound.exists():
                value = read(bound)
                if value['cluster_config_sha256'] != sha(self.cp) or value['code_sha256'] != sha(__file__):
                    raise ValueError('Autoscaler input/runtime changed')
                if max_instances is not None and max_instances != value['max_instances']: raise ValueError('Autoscale budget changed')
                if (initial_instances_target is not None
                        and initial_instances_target != value.get('initial_instances_target', self.base_count)):
                    raise ValueError('Autoscale initial target changed')
                self.binding = value
            else:
                maximum = 16 if max_instances is None else max_instances
                if type(maximum) is not int or not self.base_count <= maximum <= 128:
                    raise ValueError('Maximum must cover original fleet and be <=128')
                initial_target = self.base_count if initial_instances_target is None else initial_instances_target
                if type(initial_target) is not int or not self.base_count <= initial_target <= maximum:
                    raise ValueError('Initial target must cover original fleet and fit maximum')
                self.binding = {'schema': VERSION, 'cluster_config': str(self.cp), 'cluster_config_sha256': sha(self.cp),
                                'code_sha256': sha(__file__), 'initial_notebooks': self.base_names,
                                'max_instances': maximum, 'initial_instances_target': initial_target, 'template': self.template,
                                'name_prefix': 'sscd-auto-' + hashlib.sha256(str(self.root).encode()).hexdigest()[:8],
                                'deadline_epoch': self.c['deadline_epoch'], 'minimum_measurement_seconds': 300,
                                'max_additions_per_round': 4, 'catalog_reserved_later_seconds': 36000}
                save(bound, self.binding)
        self.maximum = self.binding['max_instances']; self.platform = platform or Platform(self.c)
        self.initial_target = self.binding.get('initial_instances_target', self.base_count)
        self.state = read(self.d / 'STATE.json') if (self.d / 'STATE.json').exists() else {'instances': {}, 'last_scale_at': 0}
    def persist(self): save(self.d / 'STATE.json', self.state)
    def live(self, name):
        found = self.platform.list_name(name)
        if len(found) > 1: raise ValueError('Ambiguous duplicate notebook name')
        if found:
            expected = {'workspace': self.c['workspace'], 'project': option(self.template, '--project'),
                        'compute_group': option(self.template, '--group')}
            if any(found[0].get(k) != v for k, v in expected.items()): raise ValueError('Live notebook resource identity mismatch')
            return found[0]
        return None
    def add_owned(self, name, intent):
        ledger_path = self.root / 'resources' / 'PLAN.json'
        with lock(self.root / 'resources' / 'PLAN.autoscale.lock', blocking=True):
            ledger = read(ledger_path); existing = [x for x in ledger['instances'] if x['name'] == name]
            expected = {'name': name, 'argv': intent['argv'], 'owned': True, 'cpu': 16, 'memory_gib': 256, 'gpu': 0,
                        'autoscale_owner': sha(self.d / 'INPUT.json'), 'intent_path': str(self.d / 'intents' / (name + '.json'))}
            if existing and existing != [expected]: raise ValueError('Ownership ledger collision')
            if not existing:
                ledger['instances'].append(expected); ledger['max_instances'] = max(ledger.get('max_instances', 0), self.maximum)
                save(ledger_path, ledger)
        cfg = dict(self.c, notebooks=[name], max_instances=1)
        config_path = self.d / 'worker_configs' / (name + '.json')
        if config_path.exists() and read(config_path) != cfg: raise ValueError('Extra worker config changed')
        if not config_path.exists(): save(config_path, cfg)
        return config_path
    def create_one(self, now):
        index = self.base_count + len(self.state['instances']) + 1
        if index > self.maximum: return
        name = self.binding['name_prefix'] + f'-{index:02d}'; path = self.d / 'intents' / (name + '.json')
        argv = list(self.template); argv[argv.index('--name') + 1] = name
        # Crash after intent publication is reconciled at the same name, never blindly resubmitted.
        intent = read(path) if path.exists() else {'name': name, 'argv': argv, 'created_at': now,
                                                   'attempted': False, 'preflight_absent': False}
        self.state['instances'][name] = {'intent': str(path), 'owned': False, 'state': 'intent', 'last_action_at': 0}
        save(path, intent); self.persist()
        if intent.get('attempted'): return
        if self.live(name) is not None:
            self.state['instances'][name].update(state='name_collision', error='Existing name was not created by this intent')
            self.persist(); return
        intent.update(preflight_absent=True, attempted=True, submitted_at=now); save(path, intent)
        result = self.platform.create(argv)
        intent['create_result'] = sanitized(result); save(path, intent)
        self.state['instances'][name]['state'] = 'await_live_confirmation'; self.persist()
        self.reconcile_one(name, now)
    def reconcile_one(self, name, now, *, stopping=False):
        entry = self.state['instances'][name]; intent = read(entry['intent'])
        if entry.get('state') == 'name_collision': return
        expected_argv = list(self.template); expected_argv[expected_argv.index('--name') + 1] = name
        if intent.get('name') != name or intent.get('argv') != expected_argv:
            raise ValueError('Creation intent resource binding changed')
        if not intent.get('attempted'):
            if stopping:
                entry.update(state='unsubmitted_intent', error='No submission or cleanup authority'); return
            # Safe recovery only before the durable attempted bit. A crash after
            # that bit is ambiguous and can only be reconciled, never resubmitted.
            if self.live(name) is not None:
                entry.update(state='name_collision', error='Existing name predates submission'); return
            intent.update(preflight_absent=True, attempted=True, submitted_at=now); save(entry['intent'], intent)
            intent['create_result'] = sanitized(self.platform.create(intent['argv'])); save(entry['intent'], intent)
        if not intent.get('preflight_absent'):
            raise ValueError('Creation intent lacks successful absent-name preflight')
        live = self.live(name)
        if live is None:
            entry.update(state='missing_or_creation_uncertain', last_observed_at=now); return
        cp = self.add_owned(name, intent); entry.update(owned=True, last_observed_at=now, platform_status=live['status'])
        status = str(live['status']).upper()
        if stopping:
            if status in ('STOPPED', 'FAILED', 'DELETED'):
                entry.update(state='released'); return
            if now - entry.get('last_stop_at', 0) >= 120:
                entry['last_stop_at'] = now; entry['stop_result'] = sanitized(self.platform.transition(name, 'stop'))
            entry['state'] = 'release_pending'; return
        hb = self.root / 'controller' / 'workers' / (name + '.json')
        value = read(hb) if hb.exists() else {}
        if value.get('fatal'):
            entry.update(state='await_worker_repair', error='Worker retries exhausted; automatic relaunch disabled')
            return
        if now - value.get('updated_at', 0) < 180 and not value.get('fatal') and value.get('status') not in ('failed', 'stopped'):
            entry.update(state='worker_live', consecutive_recovery_attempts=0); return
        attempts = entry.get('consecutive_recovery_attempts', 0)
        delay = min(1800, 120 * 2 ** min(attempts, 4))
        if now - entry.get('last_action_at', 0) < delay: return
        if status in ('STOPPED', 'FAILED'):
            entry['last_action_at'] = now; entry['start_result'] = sanitized(self.platform.transition(name, 'start'))
            entry.update(state='restart_requested', consecutive_recovery_attempts=attempts + 1)
        elif status == 'RUNNING':
            entry['last_action_at'] = now; entry['launch_result'] = sanitized(self.platform.launch_worker(name, cp))
            entry.update(state='worker_launch_requested', consecutive_recovery_attempts=attempts + 1)
        else: entry['state'] = 'await_platform_running'
    def active_workers(self, now, phase):
        names = self.base_names + list(self.state['instances']) + ['coordinator-local']; active = 0
        for name in names:
            path = self.root / 'controller' / 'workers' / (name + '.json')
            if not path.exists(): continue
            hb = read(path)
            if (now - hb.get('updated_at', 0) < 180 and hb.get('phase') == phase and hb.get('child_pid')
                    and hb.get('status') == 'running' and not hb.get('fatal')):
                active += 1
        return active
    def tick(self, now=None):
        now = time.time() if now is None else now
        with lock(self.d / 'TICK.lock') as acquired:
            if not acquired: return {'state': 'another_tick_active'}
            self.state = read(self.d / 'STATE.json') if (self.d / 'STATE.json').exists() else self.state
            # Intents survive a crash before STATE publication; count all reservations toward the cap.
            for path in (self.d / 'intents').glob('*.json') if (self.d / 'intents').exists() else []:
                self.state['instances'].setdefault(path.stem, {'intent': str(path), 'owned': False, 'state': 'recovered_intent'})
            progress_path = self.root / 'controller' / 'PROGRESS.json'
            if not progress_path.exists(): return {'state': 'await_controller_progress'}
            progress = read(progress_path); terminal = progress.get('phase') in ('complete', 'failed', 'stopped')
            stopping = terminal or (self.root / 'controller' / 'STOP').exists() or (self.d / 'STOP').exists()
            errors = {}
            def reconcile(name):
                try: self.reconcile_one(name, now, stopping=stopping)
                except Exception as exc: errors[name] = type(exc).__name__  # never persist possibly sensitive exception text
            with ThreadPoolExecutor(max_workers=4) as pool: list(pool.map(reconcile, list(self.state['instances'])))
            self.persist()
            reserved = self.base_count + len(self.state['instances'])
            decision = scaling_decision(progress, now=now, reserved_instances=reserved,
                                        active_workers=self.active_workers(now, progress['phase']), max_instances=self.maximum,
                                        last_scale_at=self.state.get('last_scale_at', 0), baseline_instances=self.base_count,
                                        initial_instances_target=self.initial_target)
            if not stopping and not errors and decision['add']:
                self.state['last_scale_at'] = now; self.persist()
                for _ in range(decision['add']):
                    try: self.create_one(now)
                    except Exception as exc: errors['create'] = type(exc).__name__; break
            self.persist()
            pending = [n for n, e in self.state['instances'].items()
                       if e.get('state') not in ('released', 'name_collision', 'unsubmitted_intent')]
            value = {'schema': VERSION, 'updated_at': now, 'pid': os.getpid(), 'host': socket.gethostname(),
                     'phase': progress['phase'], 'decision': decision, 'errors': errors,
                     'initial_instances': self.base_count, 'additional_reservations': len(self.state['instances']),
                     'max_instances': self.maximum, 'initial_instances_target': self.initial_target,
                     'stopping': stopping, 'cleanup_complete': stopping and not pending,
                     'original_instances_managed_here': False, 'instances': self.state['instances']}
            save(self.d / 'PROGRESS.json', value); return value
    def run(self):
        if self.c.get('coordinator_host') and self.c['coordinator_host'] != socket.gethostname():
            raise ValueError('Autoscaler must run on the configured coordinator host')
        with lock(self.d / 'RUN.lock') as acquired:
            if not acquired: return {'already_running': True}
            save(self.d / 'DAEMON.json', {'pid': os.getpid(), 'host': socket.gethostname(), 'started_at': time.time()})
            signal.signal(signal.SIGTERM, lambda *_: self.stop_event.set())
            signal.signal(signal.SIGINT, lambda *_: self.stop_event.set())
            while not self.stop_event.is_set():
                try:
                    result = self.tick()
                    if result.get('cleanup_complete'): return result
                except Exception as exc:
                    save(self.d / 'ERROR.json', {'at': time.time(), 'error_type': type(exc).__name__})
                self.stop_event.wait(60)
            return {'stopped': True, 'resources_retained_for_resume': True}
    def launch(self):
        with lock(self.d / 'RUN.lock') as acquired:
            if not acquired: return {'already_running': True}
        argv = [self.c['python'], str(Path(__file__).absolute()), 'run', '--cluster-config', str(self.cp),
                '--max-instances', str(self.maximum), '--initial-instances-target', str(self.initial_target)]
        with (self.d / 'daemon.log').open('ab', buffering=0) as stream:
            child = subprocess.Popen(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=stream,
                                     cwd=self.root, start_new_session=True, close_fds=True)
        return {'submitted_pid': child.pid, 'log': str(self.d / 'daemon.log'), 'max_instances': self.maximum}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('prepare', 'launch', 'run', 'once', 'status'))
    parser.add_argument('--cluster-config', required=True); parser.add_argument('--max-instances', type=int)
    parser.add_argument('--initial-instances-target', type=int,
                        help='Explicit benchmark-informed initial search fleet; defaults to original fleet size')
    args = parser.parse_args(argv); runner = Autoscaler(args.cluster_config, max_instances=args.max_instances,
                                                      initial_instances_target=args.initial_instances_target)
    if args.action == 'prepare': result = runner.binding
    elif args.action == 'launch': result = runner.launch()
    elif args.action == 'run': result = runner.run()
    elif args.action == 'once': result = runner.tick()
    else:
        result = read(runner.d / 'PROGRESS.json') if (runner.d / 'PROGRESS.json').exists() else {'state': 'not_started'}
    # This output contains only our whitelist receipts, never CLI connection data.
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__': main()
