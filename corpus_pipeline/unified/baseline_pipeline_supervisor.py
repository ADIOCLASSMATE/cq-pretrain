"""Explicit, persistent orchestration of one baseline preprocessing run.

All resource actions come from a frozen external ops file. Business receipts,
not process exit codes or operator STOP, authorize stage transitions. Each
command has a durable runner so an orchestrator restart retains its exit status.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from . import baseline_supervisor as process

SCHEMA = 'corpus.baseline_pipeline_supervisor.v1'
MODULE = 'corpus_pipeline.unified.baseline_pipeline_supervisor'
COMMANDS = ('normalize', 'queue_rpc', 'frontpass_workflow', 'frontpass_manager',
            'screening_workflow', 'sscd_workflow', 'sscd_manager', 'publish_workflow',
            'review_start', 'review_stop', 'review_released', 'index_stop',
            'frontpass_stop', 'sscd_stop')
PROBES = ('frontpass_released', 'sscd_released', 'index_released', 'review_ready')
RESOURCE_STARTS = {'frontpass_manager', 'sscd_manager', 'review_start'}
RECEIPTS = ('normalize', 'frontpass', 'screening', 'sscd_search', 'publication')
GATES = ('frontpass_plan', 'frontpass_released', 'sscd_plan', 'sscd_released', 'index_released')
WORKFLOWS = {'normalize', 'frontpass_workflow', 'screening_workflow', 'sscd_workflow', 'publish_workflow'}
STOP_COMMANDS = ('index_stop', 'frontpass_stop', 'sscd_stop', 'review_stop')
read_json, atomic_json = process.read_json, process.atomic_json


def sha(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 20), b''):
            result.update(block)
    return result.hexdigest()


def nested(value, key):
    for part in key.split('.'):
        if not isinstance(value, dict) or part not in value:
            return None
        value = value[part]
    return value


def checked_ops(path):
    path = Path(path).absolute()
    config = read_json(path)
    if not isinstance(config, dict) or config.get('schema') != SCHEMA:
        raise ValueError('Explicit valid ops.json is required')
    for key in ('run_root', 'control', 'repository'):
        if not isinstance(config.get(key), str) or not Path(config[key]).is_absolute():
            raise ValueError('Explicit absolute path required: ' + key)
        config[key] = str(Path(config[key]).resolve())
    root, control = Path(config['run_root']), Path(config['control'])
    if control == root or not control.is_relative_to(root):
        raise ValueError('Controller control directory must be inside this run root')
    if control.is_symlink() or any((control/name).is_symlink() for name in ('commands', 'logs')):
        raise ValueError('Unsafe controller state symlink')
    try:
        deadline = datetime.fromisoformat(config['deadline_utc'].replace('Z', '+00:00'))
        if deadline.tzinfo is None:
            raise ValueError('Deadline requires timezone')
        config['deadline_epoch'] = deadline.timestamp()
    except (KeyError, TypeError, ValueError):
        raise ValueError('Explicit ISO UTC deadline required') from None
    if type(config.get('queue_port')) is not int or not 1024 <= config['queue_port'] <= 65535:
        raise ValueError('Explicit independent queue_port required')
    for key, low, high in (('poll_seconds', 1, 60), ('retry_seconds', 1, 300),
                           ('max_retry_seconds', 1, 3600), ('max_failures', 1, 100),
                           ('stop_grace_seconds', 1, 300)):
        if type(config.get(key)) not in (int, float) or not low <= config[key] <= high:
            raise ValueError('Explicit bounded controller setting required: ' + key)
    commands = config.get('commands', {})
    if not set(COMMANDS) <= set(commands) or set(commands) - set(COMMANDS) - set(PROBES):
        raise ValueError('All named command interfaces must be explicitly configured')
    signatures = set()
    for name, command in commands.items():
        argv = command.get('argv')
        cwd = command.get('cwd')
        if (not isinstance(argv, list) or not argv or any(not isinstance(x, str) or not x for x in argv)
                or not Path(argv[0]).is_absolute() or not os.access(argv[0], os.X_OK)
                or not isinstance(cwd, str) or not Path(cwd).is_absolute() or not Path(cwd).is_dir()
                or not isinstance(command.get('env'), dict)
                or any(not isinstance(k, str) or not isinstance(v, str) for k, v in command['env'].items())):
            raise ValueError('Invalid explicit argv/cwd/env: ' + name)
        # Every child must name this run, making exact-argv adoption unambiguous.
        if not any(arg == str(root) or arg.startswith(str(root) + '/')
                   or ('=' + str(root)) in arg for arg in argv):
            raise ValueError('Command must explicitly bind its own run: ' + name)
        signature = (tuple(argv), str(Path(cwd).resolve()))
        if signature in signatures:
            raise ValueError('Different command roles cannot share identical argv/cwd')
        signatures.add(signature)
        command['cwd'] = signature[1]
    rpc = commands['queue_rpc']['argv']
    port = str(config['queue_port'])
    if not any(value == port or value.endswith('=' + port) for value in rpc):
        raise ValueError('queue_rpc argv does not bind the declared independent port')
    receipts = config.get('receipts', {})
    if set(receipts) != set(RECEIPTS):
        raise ValueError('All business receipt paths are required')
    for name, value in receipts.items():
        if not isinstance(value, str) or not Path(value).is_absolute() or not Path(value).resolve().is_relative_to(root):
            raise ValueError('Receipt must belong to this run: ' + name)
    gates = config.get('gates', {})
    if not set(GATES) <= set(gates) or set(gates) - set(GATES) - {'review_ready'}:
        raise ValueError('All readiness and resource-release gates are required')
    for name, gate in gates.items():
        if 'command' in gate:
            if name not in PROBES or gate['command'] != name or name not in commands:
                raise ValueError('Release probe must bind its named explicit command')
            continue
        checks = gate.get('checks')
        if not isinstance(checks, list) or not checks:
            raise ValueError('Gate requires explicit JSON checks: ' + name)
        for check in checks:
            if (not isinstance(check.get('path'), str) or not Path(check['path']).is_absolute()
                    or not Path(check['path']).resolve().is_relative_to(root)
                    or not isinstance(check.get('equals'), dict) or not check['equals']):
                raise ValueError('Invalid run-bound gate: ' + name)
        if name.endswith('_released'):
            zero = gate.get('zero_key')
            if not isinstance(zero, str) or not any(type(c['equals'].get(zero)) is int and c['equals'][zero] == 0 for c in checks):
                raise ValueError('Release gate must explicitly prove zero reservations: ' + name)
            if not any(any(key != zero and (value is True or isinstance(value, str) and value in ('complete', 'released', 'stopped'))
                           for key, value in c['equals'].items()) for c in checks):
                raise ValueError('Release gate also needs a terminal manager state: ' + name)
    config['ops_path'] = str(path)
    return config


def gate_passed(config, name):
    for check in config['gates'][name]['checks']:
        value = read_json(check['path'])
        if not isinstance(value, dict):
            return False
        for key, expected in check['equals'].items():
            actual = nested(value, key)
            if type(actual) is not type(expected) or actual != expected:
                return False
    return True


def receipt_passed(config, name):
    path = Path(config['receipts'][name])
    value = read_json(path)
    if value is None:
        return False
    if name == 'publication':
        if not isinstance(value, dict) or not isinstance(value.get('complete'), dict):
            raise ValueError('Publication requires a bound CURRENT pointer, not a completion flag')
        for field in ('complete', 'manifest'):
            artifact = value.get(field, {})
            target = Path(artifact.get('path', ''))
            if (not target.is_absolute() or not target.resolve().is_relative_to(Path(config['run_root'])/'releases')
                    or not target.is_file() or sha(target) != artifact.get('sha256')):
                raise ValueError('Published release artifact binding mismatch')
        complete, manifest = (read_json(value[field]['path']) for field in ('complete', 'manifest'))
        if (complete.get('complete') is not True or manifest.get('complete') is not True
                or complete.get('scope') != 'full' or manifest.get('scope') != 'full'
                or value.get('scope') != 'full'
                or complete.get('manifest_sha256') != value['manifest']['sha256']):
            raise ValueError('Full publication is not complete')
        return True
    if not isinstance(value, dict) or value.get('complete') is not True or value.get('stage') != name:
        raise ValueError('Invalid business receipt: ' + name)
    coverage = value.get('coverage', {})
    if (type(coverage.get('expected')) is not int or coverage['expected'] < 0
            or coverage.get('processed') != coverage['expected']
            or coverage.get('missing') != 0 or coverage.get('pending_infra') != 0
            or value.get('semantics', {}).get('scope') != 'full'):
        raise ValueError('Business receipt has incomplete full coverage: ' + name)
    return True


def runner_command(config, name, attempt):
    return [sys.executable, '-m', MODULE, 'execute', '--ops', config['ops_path'],
            '--name', name, '--attempt', str(attempt)]


def command_area(config, name, attempt):
    return Path(config['control'])/'commands'/name/str(attempt)


def execute(config, name, attempt):
    """Durably capture one child's result independently of the orchestrator."""
    if name not in config['commands'] or attempt < 1:
        raise ValueError('Unknown command or invalid attempt')
    area = command_area(config, name, attempt)
    area.mkdir(parents=True, exist_ok=True)
    command = config['commands'][name]
    with process.exclusive(area/'runner.lock'):
        if (area/'RESULT.json').exists():
            return 0
        matches = process.matching_processes(command['argv'], command['cwd'])
        if matches:
            # A previous runner may have died without its child. Never duplicate
            # a possibly resource-creating operation when its result is unknown.
            atomic_json(area/'RESULT.json', dict(exit_code=None, error='existing_command_without_durable_result',
                                                matching_pids=[x['pid'] for x in matches], time=time.time()))
            return 1
        env = dict(os.environ); env.update(command['env'])
        stopped = [False]
        for number in (signal.SIGTERM, signal.SIGINT):
            signal.signal(number, lambda *_: stopped.__setitem__(0, True))
        with (area/'stdout.log').open('ab', buffering=0) as stdout, (area/'stderr.log').open('ab', buffering=0) as stderr:
            child = subprocess.Popen(command['argv'], cwd=command['cwd'], env=env,
                stdin=subprocess.DEVNULL, stdout=stdout, stderr=stderr, start_new_session=True)
            identity = None
            for _ in range(200):
                candidate = process.process_identity(child.pid)
                if candidate and candidate['argv'] == command['argv'] and candidate['cwd'] == command['cwd']:
                    identity = candidate; break
                if child.poll() is not None:
                    break
                time.sleep(.01)
            atomic_json(area/'CHILD.json', dict(identity=identity, argv=command['argv'], cwd=command['cwd']))
            signalled = False
            stopping_at = None
            while child.poll() is None:
                if stopped[0] and not signalled:
                    if identity:
                        process.signal_owned(identity, command['argv'], command['cwd'], signal.SIGTERM)
                    signalled = True
                    stopping_at = time.monotonic()
                if stopping_at is not None and time.monotonic()-stopping_at > config['stop_grace_seconds'] and identity:
                    process.signal_owned(identity, command['argv'], command['cwd'], signal.SIGKILL)
                time.sleep(.2)
            stderr_tail = (area/'stderr.log').read_bytes()[-2000:].decode('utf-8', errors='replace')
            atomic_json(area/'RESULT.json', dict(exit_code=child.returncode, time=time.time(),
                                                stopped=stopped[0], stderr_tail=stderr_tail))
    return 0


class Supervisor:
    def __init__(self, config):
        self.config = config
        self.control = Path(config['control'])
        self.control.mkdir(parents=True, exist_ok=True)
        self.children = {}
        self.stop_requested = False
        self.state = read_json(self.control/'STATE.json', {})
        self.state.setdefault('phase', 'normalize')
        if self.state['phase'] == 'complete':
            self.state['phase'] = 'publish'  # Revalidate publication and released resources on restart.
        self.state.setdefault('commands', {})
        self.state.setdefault('errors', [])

    def bind(self):
        expected = Path(self.config['repository'])/'corpus_pipeline/unified/baseline_pipeline_supervisor.py'
        if expected.resolve() != Path(__file__).resolve():
            raise ValueError('Launch from the explicitly frozen repository')
        identity = dict(schema=SCHEMA, config=self.config, code_sha256=sha(__file__),
                        process_helpers_sha256=sha(process.__file__), host=process.process_identity(os.getpid())['host'])
        old = read_json(self.control/'INPUT.json')
        if old is not None and old != identity:
            raise ValueError('Ops/runtime/host changed; use a new explicit control directory')
        if old is None:
            atomic_json(self.control/'INPUT.json', identity)
        self.save()

    def save(self, now=None):
        now = time.time() if now is None else now
        self.state.update(schema=SCHEMA, time=now, supervisor=process.process_identity(os.getpid()),
                          deadline_utc=self.config['deadline_utc'],
                          deadline_seconds_remaining=self.config['deadline_epoch']-now,
                          deadline_overdue=now > self.config['deadline_epoch'],
                          complete=self.state['phase'] == 'complete')
        atomic_json(self.control/'STATE.json', self.state)
        atomic_json(self.control/'STATUS.json', self.state)

    def reconcile(self, name, now):
        row = self.state['commands'].setdefault(name, dict(attempts=0, failures=0, next_start=0, phase='idle'))
        attempt = row['attempts']
        if not attempt:
            return row
        argv = runner_command(self.config, name, attempt)
        cwd = self.config['repository']
        child = self.children.get(name)
        if child is not None and child.poll() is not None:
            self.children.pop(name)
        alive = process.same_process(row.get('identity'), argv, cwd)
        if not alive:
            matches = process.matching_processes(argv, cwd)
            if len(matches) > 1:
                raise RuntimeError('Multiple exact owned runners: ' + name)
            if matches:
                row.update(identity=matches[0], adopted_at=now); alive = True
        row['alive'] = alive
        if alive:
            row['phase'] = 'running'
            return row
        area = command_area(self.config, name, attempt)
        result = read_json(area/'RESULT.json')
        if row.get('handled_attempt') != attempt:
            if result is None:
                raw = self.config['commands'][name]
                matches = process.matching_processes(raw['argv'], raw['cwd'])
                if matches:
                    row.update(phase='orphan_running', orphan_identities=matches)
                    return row
                result = dict(exit_code=None, error='runner_exited_without_durable_result')
            row.update(result=result, handled_attempt=attempt, finished_at=now)
            waiting_allowed = name in WORKFLOWS or name in PROBES + ('review_released',)
            ok = result.get('exit_code') in ((0, 2) if waiting_allowed else (0,))
            row['phase'] = 'succeeded' if ok else 'backoff'
            if ok:
                row['failures'] = 0
            else:
                row['failures'] += 1
                self.error(name, result.get('error') or ('command exit ' + str(result.get('exit_code')) + ': ' + result.get('stderr_tail', '')), now)
            row['next_start'] = now + min(self.config['max_retry_seconds'],
                self.config['retry_seconds'] * 2 ** min(row['failures'], 10))
            if name in PROBES + ('review_released',):
                row.update(released=False, ready=False)
            if name in PROBES + ('review_released',) and result.get('exit_code') == 0:
                try:
                    payload = json.loads((area/'stdout.log').read_text())
                    row['released'] = (payload.get('released') is True and type(payload.get('reserved')) is int
                                       and payload['reserved'] == 0)
                    row['ready'] = (payload.get('ready') is True and type(payload.get('ready_replicas')) is int
                                    and payload['ready_replicas'] >= 1)
                except (OSError, ValueError, AttributeError):
                    row['released'] = False
                    row['ready'] = False
        return row

    def ensure(self, name, now, *, repeat=False, cleanup=False):
        row = self.reconcile(name, now)
        if row.get('alive') or row['phase'] == 'orphan_running':
            return row
        if row['phase'] == 'succeeded' and not repeat:
            return row
        if now < row['next_start']:
            return row
        if ((name not in PROBES + ('review_released',) and row['failures'] >= self.config['max_failures'])
                or (name in RESOURCE_STARTS and not cleanup and row['attempts'] >= self.config['max_failures'])):
            row['phase'] = 'retry_exhausted'
            return row
        row.update(attempts=row['attempts']+1, phase='launching', started_at=now, identity=None, released=False, ready=False)
        self.save(now)  # Durable intent before creating any process/resources.
        argv = runner_command(self.config, name, row['attempts'])
        log = self.control/'logs'/(name + '.log'); log.parent.mkdir(parents=True, exist_ok=True)
        env = dict(os.environ, PYTHONPATH=self.config['repository'], PYTHONUNBUFFERED='1')
        with log.open('ab', buffering=0) as stream:
            child = subprocess.Popen(argv, cwd=self.config['repository'], env=env, stdin=subprocess.DEVNULL,
                                     stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        self.children[name] = child
        for _ in range(200):
            identity = process.process_identity(child.pid)
            if identity and identity['argv'] == argv and identity['cwd'] == self.config['repository']:
                row.update(identity=identity, alive=True, phase='running'); break
            if child.poll() is not None:
                break
            time.sleep(.01)
        return row

    def stop(self, names, now):
        done = True
        for name in names:
            row = self.reconcile(name, now)
            identities = list(row.get('orphan_identities', []))
            raw = self.config['commands'][name]
            for identity in identities:
                if process.same_process(identity, raw['argv'], raw['cwd']):
                    process.signal_owned(identity, raw['argv'], raw['cwd'], signal.SIGTERM); done = False
            if row.get('alive'):
                process.signal_owned(row['identity'], runner_command(self.config, name, row['attempts']),
                                     self.config['repository'], signal.SIGTERM)
                row['stop_requested_at'] = row.get('stop_requested_at', now)
                done = False
        return done

    def error(self, operation, error, now):
        self.state['errors'].append(dict(time=now, operation=operation, error=str(error)))
        self.state['errors'] = self.state['errors'][-40:]
        self.state['last_error'] = self.state['errors'][-1]

    def gate(self, name, now):
        gate = self.config['gates'][name]
        if 'command' not in gate:
            return gate_passed(self.config, name)
        row = self.reconcile(gate['command'], now)
        field = 'ready' if name == 'review_ready' else 'released'
        if row.get(field) is True and now - row.get('finished_at', 0) <= self.config['max_retry_seconds']:
            return True
        self.ensure(gate['command'], now, repeat=True)
        return False

    def released_review(self, now):
        stop = self.reconcile('review_stop', now)
        if stop['phase'] != 'succeeded':
            self.ensure('review_stop', now, repeat=True)
            return False
        row = self.reconcile('review_released', now)
        if row.get('released') is True and now - row.get('finished_at', 0) <= self.config['max_retry_seconds']:
            return True
        self.ensure('review_released', now, repeat=True)
        # stop_once may return exit 0 while deletions still await platform state.
        # Reissue its idempotent cleanup until the independent release probe wins.
        self.ensure('review_stop', now, repeat=True)
        return False

    def stopping(self, now):
        self.state['phase'] = 'stopping'
        workflows_stopped = self.stop(tuple(WORKFLOWS), now)
        self.ensure('index_stop', now)
        released = self.gate('index_released', now)
        for manager, stop, gate in [('frontpass_manager', 'frontpass_stop', 'frontpass_released'),
                                    ('sscd_manager', 'sscd_stop', 'sscd_released')]:
            if not self.state['commands'].get(manager, {}).get('attempts', 0):
                continue
            stopped = self.ensure(stop, now)
            stage_released = self.gate(gate, now)
            if not stage_released and stopped['phase'] == 'succeeded':
                # The stop interface has written its durable stop request. Keep
                # its owning manager alive to finish exact platform retirement.
                self.ensure(manager, now, repeat=True, cleanup=True)
                if manager == 'frontpass_manager': self.ensure('queue_rpc', now, repeat=True)
            released = stage_released and released
        if self.state['commands'].get('review_start', {}).get('attempts', 0):
            released = self.released_review(now) and released
        if workflows_stopped and released and self.stop(tuple(self.config['commands']), now):
            self.state['phase'] = 'stopped'
            atomic_json(self.control/'STOPPED.json', dict(time=now, complete=False, resources_released=True))
            return True
        return False

    def step(self, now=None):
        now = time.time() if now is None else now
        done = False
        try:
            if (self.stop_requested or (self.control/'STOP').exists()
                    or (Path(self.config['run_root'])/'STOP').exists()
                    or self.state['phase'] in ('stopping', 'stopped')):
                done = self.stopping(now)
            elif now < self.state.get('retry_after', 0):
                pass
            else:
                phase = self.state['phase']
                if phase == 'normalize':
                    if receipt_passed(self.config, 'normalize'):
                        if self.stop(('normalize',), now): self.state['phase'] = 'frontpass'
                    else: self.ensure('normalize', now, repeat=True)
                elif phase == 'frontpass':
                    if receipt_passed(self.config, 'frontpass'): self.state['phase'] = 'frontpass_release'
                    else:
                        self.ensure('queue_rpc', now, repeat=True)
                        self.ensure('frontpass_workflow', now, repeat=True)
                        if self.gate('frontpass_plan', now): self.ensure('frontpass_manager', now, repeat=True)
                elif phase == 'frontpass_release':
                    requested = self.ensure('frontpass_stop', now)
                    if self.gate('frontpass_released', now):
                        if self.stop(('frontpass_workflow', 'frontpass_manager', 'queue_rpc'), now): self.state['phase'] = 'review_start'
                    elif requested['phase'] == 'succeeded':
                        self.ensure('frontpass_manager', now, repeat=True, cleanup=True)
                        self.ensure('queue_rpc', now, repeat=True)
                elif phase == 'review_start':
                    if not self.gate('frontpass_released', now):
                        raise RuntimeError('Frontpass reservations reappeared before review')
                    row = self.ensure('review_start', now)
                    ready = (self.gate('review_ready', now) if 'review_ready' in self.config['gates']
                             else row['phase'] == 'succeeded')
                    if ready and (row.get('alive') or row['phase'] == 'succeeded'): self.state['phase'] = 'screening'
                elif phase == 'screening':
                    if receipt_passed(self.config, 'screening'):
                        if self.stop(('screening_workflow',), now): self.state['phase'] = 'review_release'
                    else:
                        if 'review_ready' in self.config['gates']: self.ensure('review_start', now, repeat=True)
                        self.ensure('screening_workflow', now, repeat=True)
                elif phase == 'review_release':
                    if self.released_review(now) and self.stop(('review_start',), now): self.state['phase'] = 'sscd'
                elif phase == 'sscd':
                    if receipt_passed(self.config, 'sscd_search'): self.state['phase'] = 'sscd_release'
                    else:
                        self.ensure('sscd_workflow', now, repeat=True)
                        if self.gate('sscd_plan', now): self.ensure('sscd_manager', now, repeat=True)
                elif phase == 'sscd_release':
                    requested = self.ensure('sscd_stop', now)
                    if self.gate('sscd_released', now):
                        if self.stop(('sscd_workflow', 'sscd_manager'), now): self.state['phase'] = 'publish'
                    elif requested['phase'] == 'succeeded':
                        self.ensure('sscd_manager', now, repeat=True, cleanup=True)
                elif phase == 'publish':
                    if receipt_passed(self.config, 'publication'):
                        all_receipts = all(receipt_passed(self.config, name) for name in RECEIPTS[:-1])
                        all_released = all(self.gate(name, now) for name in ('index_released', 'frontpass_released', 'sscd_released'))
                        if all_receipts and all_released and self.released_review(now):
                            if self.stop(tuple(self.config['commands']), now):
                                self.state['phase'] = 'complete'; done = True
                                atomic_json(self.control/'COMPLETE.json', dict(complete=True, time=now,
                                    publication=self.config['receipts']['publication'],
                                    publication_sha256=sha(self.config['receipts']['publication']),
                                    deadline_met=now <= self.config['deadline_epoch']))
                    else: self.ensure('publish_workflow', now, repeat=True)
                elif phase == 'complete': done = True
                else: raise ValueError('Unknown persisted pipeline phase')
        except Exception as exc:
            self.error(self.state['phase'], type(exc).__name__ + ': ' + str(exc), now)
            self.state['retry_after'] = now + self.config['retry_seconds']
        self.save(now)
        return done


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='action', required=True)
    for name in ('run', 'status', 'execute'):
        child = sub.add_parser(name); child.add_argument('--ops', required=True)
        if name == 'execute':
            child.add_argument('--name', required=True, choices=COMMANDS + PROBES)
            child.add_argument('--attempt', type=int, required=True)
    args = parser.parse_args(argv)
    try:
        config = checked_ops(args.ops)
        if args.action == 'execute': return execute(config, args.name, args.attempt)
        if args.action == 'status':
            print(json.dumps(read_json(Path(config['control'])/'STATUS.json', {'phase': 'not_started', 'complete': False})))
            return 0
        supervisor = Supervisor(config)
        with process.exclusive(supervisor.control/'controller.lock'):
            supervisor.bind()
            for number in (signal.SIGTERM, signal.SIGINT):
                signal.signal(number, lambda *_: setattr(supervisor, 'stop_requested', True))
            while not supervisor.step():
                time.sleep(config['poll_seconds'])
        return 0 if supervisor.state['phase'] == 'complete' else 2
    except Exception as exc:
        print(json.dumps(dict(complete=False, error_type=type(exc).__name__, error=str(exc))), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
