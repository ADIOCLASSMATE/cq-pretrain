"""Local process/fencing tests only; no dataset or platform resources are used."""
from argparse import Namespace
from copy import deepcopy
import fcntl
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from unittest.mock import patch

from . import baseline_supervisor as sup


FAKE_ENGINE = r'''
import argparse, fcntl, json, os, pathlib, signal, socket, time
p=argparse.ArgumentParser()
p.add_argument('action'); p.add_argument('--download-control'); p.add_argument('--output')
p.add_argument('--worker'); p.add_argument('--watch',action='store_true')
a=p.parse_args(); output=pathlib.Path(a.output); output.mkdir(exist_ok=True)
state=json.loads((pathlib.Path(a.download_control)/'fixture.json').read_text())
if a.action=='status':
    print(json.dumps(state))
elif a.action=='finalize':
    time.sleep(state.get('finalize_delay',0))
    if state.get('finalize_pending'):
        print(json.dumps({'error':'waiting_for_inputs'})); raise SystemExit(2)
    result={'complete':True,'validation_passed':True}
    (output/'COMPLETE.json').write_text(json.dumps(result)); print(json.dumps(result))
elif a.action=='worker':
    area=output/'state'; (area/'locks').mkdir(parents=True,exist_ok=True)
    (area/'workers').mkdir(exist_ok=True)
    lock=(area/'locks'/('worker-'+a.worker+'.lock')).open('a+')
    fcntl.lockf(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    stopped=[False]
    signal.signal(signal.SIGTERM,lambda *args:stopped.__setitem__(0,True))
    while not stopped[0]:
        (area/'workers'/(a.worker+'.json')).write_text(json.dumps({'pid':os.getpid(),'host':socket.gethostname(),'time':time.time(),'phase':'indexing'}))
        time.sleep(.02)
'''


class SupervisorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.repo = self.root / 'repository'
        package = self.repo / 'corpus_pipeline/unified'
        package.mkdir(parents=True)
        (self.repo / 'corpus_pipeline/__init__.py').touch()
        (package / '__init__.py').touch()
        (package / 'baseline_index.py').write_text(FAKE_ENGINE)
        (package / 'baseline_supervisor.py').write_text('# fixture')
        self.download = self.root / 'download'
        self.download.mkdir()
        self.raw = self.root / 'raw'
        self.raw.mkdir()
        self.run = self.root / 'run'
        self.run.mkdir()
        self.roots = self.run / 'source_roots.json'
        self.roots.write_text(json.dumps({'dataset@revision': str(self.raw)}))
        self.config = dict(repository=str(self.repo), python=sys.executable,
            download_control=str(self.download), output=str(self.run / 'native'),
            control=str(self.run / 'index_control'), source_roots=str(self.roots), workers=1,
            poll_seconds=1, status_seconds=1, stop_grace_seconds=1, command_timeout_seconds=10)
        self.write_engine_status('indexing')
        self.processes = []
        self.addCleanup(self.cleanup_processes)

    def cleanup_processes(self):
        for child in self.processes:
            if child.poll() is None:
                child.terminate()
            try:
                child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                child.kill(); child.wait()

    def write_engine_status(self, phase, **extra):
        (self.download / 'fixture.json').write_text(json.dumps(dict(phase=phase, **extra)))

    def prepared(self):
        controller = sup.Supervisor(self.config)
        controller.step()
        self.processes.extend(controller.children.values())
        return controller

    def test_explicit_venv_and_thread_environment(self):
        config = sup.checked_config(Namespace(command='run', **self.config))
        self.assertEqual(config['python'], sys.executable)
        env = sup.child_environment(config)
        for name in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
            self.assertEqual(env[name], '1')
        self.assertEqual(env['CORPUS_DATASET_ROOTS'], str(self.roots))
        self.assertEqual(sup.command_for(config, 'worker', 7)[-3:], ['--worker', '7', '--watch'])

    def test_state_and_logs_cannot_overlap_raw_or_download_or_output(self):
        for key, value in (('control', str(self.raw / 'control')), ('output', str(self.raw / 'native')),
                           ('control', str(self.download / 'control')),
                           ('control', str(self.run / 'native/control'))):
            cfg = dict(self.config, **{key: value})
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                sup.checked_config(Namespace(command='run', **cfg))

    def test_lockf_excludes_another_supervisor_process(self):
        lock = self.root / 'supervisor.lock'
        code = "import fcntl,sys; f=open(sys.argv[1],'a+');\ntry: fcntl.lockf(f,fcntl.LOCK_EX|fcntl.LOCK_NB)\nexcept BlockingIOError: sys.exit(17)\n"
        with sup.exclusive(lock):
            result = subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True)
            self.assertEqual(result.returncode, 17)
        result = subprocess.run([sys.executable, '-c', code, str(lock)], capture_output=True)
        self.assertEqual(result.returncode, 0)

    def test_restart_adopts_live_worker_and_finishes_only_after_worker_exit(self):
        first = self.prepared()
        child = first.children[0]
        identity = first.state['workers']['0']['identity']
        self.assertEqual(identity['pid'], child.pid)
        self.assertEqual(os.getsid(child.pid), child.pid)
        self.assertFalse(Path(first.state['workers']['0']['log']).is_relative_to(Path(self.config['output'])))
        second = sup.Supervisor(self.config)
        second.state['last_status_time'] = 0
        second.step()
        self.assertEqual(second.state['workers']['0']['identity'], identity)
        self.assertEqual(second.state['workers']['0']['attempts'], 1)
        self.assertFalse(second.children)
        self.write_engine_status('ready_to_finalize')
        second.state['last_status_time'] = 0
        self.assertFalse(second.step())
        self.assertEqual(second.state['phase'], 'draining')
        self.assertFalse((second.control / 'COMPLETE.json').exists())
        child.wait(timeout=3)
        self.assertFalse(second.step())
        self.processes.append(second.finalizer_child)
        second.finalizer_child.wait(timeout=3)
        self.assertTrue(second.step())
        self.assertEqual(second.state['phase'], 'complete')
        self.assertTrue((second.control / 'COMPLETE.json').exists())

    def test_recovers_unrecorded_popen_window_without_duplicate(self):
        first = self.prepared()
        child = first.children[0]
        row = first.state['workers']['0']
        row['identity'] = None
        row['phase'] = 'launching'
        first.save()
        second = sup.Supervisor(self.config)
        second.step()
        self.assertEqual(second.state['workers']['0']['identity']['pid'], child.pid)
        self.assertFalse(second.children)

    def test_exited_worker_backs_off_then_restarts(self):
        controller = self.prepared()
        old = controller.children[0]
        old.terminate(); old.wait(timeout=3)
        now = time.time()
        controller.step(now)
        row = controller.state['workers']['0']
        self.assertEqual(row['phase'], 'backoff')
        self.assertEqual(row['attempts'], 1)
        self.assertGreater(row['next_start'], now)
        controller.step(row['next_start'] - .1)
        self.assertEqual(row['attempts'], 1)
        controller.step(row['next_start'] + .1)
        self.processes.extend(controller.children.values())
        self.assertEqual(row['attempts'], 2)
        self.assertNotEqual(row['identity']['pid'], old.pid)

    def test_stale_identity_does_not_signal_reused_or_unrelated_pid(self):
        controller = self.prepared()
        identity = deepcopy(controller.state['workers']['0']['identity'])
        command = sup.command_for(self.config, 'worker', 0)
        identity['start_ticks'] += 1
        with patch.object(os, 'kill') as kill, patch.object(signal, 'pidfd_send_signal', create=True) as pidfd_signal:
            self.assertFalse(sup.signal_owned(identity, command, self.config['repository'], signal.SIGTERM))
            kill.assert_not_called(); pidfd_signal.assert_not_called()
        self.assertIsNone(controller.children[0].poll())

    def test_stop_drains_only_bound_worker_and_preserves_engine_stop_state(self):
        controller = self.prepared()
        unrelated = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])
        self.processes.append(unrelated)
        (controller.control / 'STOP').touch()
        self.assertFalse(controller.step())
        controller.children[0].wait(timeout=3)
        self.assertTrue(controller.step())
        self.assertEqual(controller.state['phase'], 'stopped')
        self.assertIsNone(unrelated.poll())
        self.assertFalse((Path(self.config['output']) / 'state/STOP').exists())
        self.assertFalse((controller.control / 'COMPLETE.json').exists())

    def test_all_units_without_source_completion_never_finalizes(self):
        controller = sup.Supervisor(self.config)
        self.write_engine_status('waiting_for_download', indexed_units=10752, source_complete=False)
        with patch.object(controller, 'launch') as launch, patch.object(controller, 'invoke', wraps=controller.invoke) as invoke:
            controller.step()
            self.assertEqual([call.args[0] for call in invoke.call_args_list], ['status'])
            launch.assert_called_once()
        self.assertFalse((controller.control / 'COMPLETE.json').exists())

    def test_finalizer_pending_and_false_success_do_not_mark_complete(self):
        controller = sup.Supervisor(self.config)
        self.write_engine_status('ready_to_finalize', finalize_pending=True)
        self.assertFalse(controller.step())
        self.processes.append(controller.finalizer_child)
        controller.finalizer_child.wait(timeout=3)
        self.assertFalse(controller.step())
        self.assertEqual(controller.state['phase'], 'waiting_for_inputs')
        self.assertFalse((controller.control / 'COMPLETE.json').exists())
        self.write_engine_status('ready_to_finalize')
        controller.state['finalizer']['next_start'] = 0
        self.assertFalse(controller.step())
        self.processes.append(controller.finalizer_child)
        controller.finalizer_child.wait(timeout=3)
        output = Path(self.config['output'])
        (output / 'COMPLETE.json').write_text(json.dumps({'complete': True}))
        self.assertFalse(controller.step())
        self.assertFalse((controller.control / 'COMPLETE.json').exists())

    def test_finalizer_survives_supervisor_restart_and_remains_observable(self):
        self.write_engine_status('ready_to_finalize', finalize_delay=.4)
        first = sup.Supervisor(self.config)
        self.assertFalse(first.step())
        child = first.finalizer_child
        self.processes.append(child)
        second = sup.Supervisor(self.config)
        self.assertFalse(second.step())
        self.assertEqual(second.state['phase'], 'finalizing')
        self.assertEqual(second.state['finalizer']['identity']['pid'], child.pid)
        self.assertEqual(second.state['finalizer']['attempts'], 1)
        self.assertGreater(second.state['time'], 0)
        self.assertIsNone(second.finalizer_child)
        child.wait(timeout=3)
        self.assertTrue(second.step())
        self.assertEqual(second.state['phase'], 'complete')

    def test_status_failure_does_not_spawn_or_claim_completion(self):
        controller = sup.Supervisor(self.config)
        with patch.object(controller, 'invoke', side_effect=RuntimeError('temporary status error')), patch.object(controller, 'launch') as launch:
            controller.step()
            launch.assert_not_called()
        self.assertEqual(controller.state['phase'], 'status_unavailable')
        self.assertFalse((controller.control / 'COMPLETE.json').exists())

    def test_runtime_binding_refuses_changed_code_or_roots(self):
        controller = sup.Supervisor(self.config)
        pretend_module = self.repo / 'corpus_pipeline/unified/baseline_supervisor.py'
        with patch.object(sup, '__file__', str(pretend_module)):
            controller.bind()
            controller.bind()
            self.roots.write_text(json.dumps({'dataset@revision': str(self.root / 'different')}))
            with self.assertRaisesRegex(ValueError, 'changed'):
                controller.bind()


if __name__ == '__main__':
    unittest.main()
