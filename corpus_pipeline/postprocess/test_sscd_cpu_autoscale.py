from pathlib import Path
import tempfile
import threading
import unittest

from . import sscd_cpu_autoscale as a


def progress(now=10000, phase='catalog', eta=40000, rate=1000):
    return {'phase': phase, 'updated_at': now, 'deadline_epoch': now + 60000,
            'stats': {'total_units': 512, 'complete_units': 10},
            'forecast': {'throughput_units_per_second': rate, 'measurement_seconds': 300,
                         'measurement_observations': 6, 'phase_remaining': 40000000,
                         'phase_remaining_seconds': eta}}


class ScalingTests(unittest.TestCase):
    def test_explicit_initial_target_only_warms_search_in_four_instance_waves(self):
        value = progress(phase='search'); value['forecast']['measurement_seconds'] = 0
        kwargs = dict(now=10000, reserved_instances=8, active_workers=0, max_instances=128,
                      initial_instances_target=64, last_scale_at=9999)
        result = a.scaling_decision(value, **kwargs)
        self.assertEqual((result['add'], result['target_instances']), (4, 64))
        self.assertEqual(result['reason'], 'explicit_initial_search_target')
        for phase in ('catalog', 'build', 'train', 'failed', 'calibrate_512_4096'):
            value['phase'] = phase
            self.assertEqual(a.scaling_decision(value, **kwargs)['add'], 0)
        value['phase'] = 'search'; kwargs['reserved_instances'] = 63
        self.assertEqual(a.scaling_decision(value, **kwargs)['add'], 1)
        kwargs['reserved_instances'] = 64
        self.assertEqual(a.scaling_decision(value, **kwargs)['add'], 0)

    def test_initial_target_respects_stale_complete_and_valid_rate_evidence(self):
        value = progress(phase='search'); value['forecast']['measurement_seconds'] = 0
        kwargs = dict(now=10000, reserved_instances=8, active_workers=9, max_instances=128,
                      initial_instances_target=64)
        value['updated_at'] = 9800
        self.assertEqual(a.scaling_decision(value, **kwargs)['add'], 0)
        value['updated_at'] = 10000; value['stats']['ready_to_finalize'] = True
        self.assertEqual(a.scaling_decision(value, **kwargs)['add'], 0)
        value['stats'].pop('ready_to_finalize'); value['forecast']['phase_remaining'] = 0
        self.assertEqual(a.scaling_decision(value, **kwargs)['add'], 0)
        value = progress(phase='search', eta=1000)
        self.assertEqual(a.scaling_decision(value, **kwargs)['reason'], 'estimated_phase_fits_budget')
        value['forecast']['throughput_units_per_second'] = 0
        self.assertEqual(a.scaling_decision(value, **kwargs)['reason'], 'stalled_rate_restore_workers_before_expansion')
        value['forecast']['measurement_seconds'] = 0; value['stats'] = {'total_units': 13, 'complete_units': 2}
        self.assertEqual(a.scaling_decision(value, **kwargs)['target_instances'], 10)

    def test_deadline_reserve_bounded_growth_cooldown_and_warmup(self):
        kwargs = dict(now=10000, reserved_instances=8, active_workers=9, max_instances=32)
        result = a.scaling_decision(progress(), **kwargs)
        self.assertEqual(result['add'], 4); self.assertEqual(result['target_instances'], 14)
        self.assertEqual(result['phase_budget_seconds'], 24000)
        self.assertEqual(a.scaling_decision(progress(), last_scale_at=9800, **kwargs)['add'], 0)
        value = progress(); value['forecast']['measurement_seconds'] = 299
        self.assertEqual(a.scaling_decision(value, **kwargs)['reason'], 'need_five_minute_rate')
        self.assertEqual(a.scaling_decision(progress(rate=0), **kwargs)['add'], 0)
        self.assertEqual(a.scaling_decision(progress(phase='build'), **kwargs)['add'], 0)
        self.assertEqual(a.scaling_decision(progress(eta=1000), **kwargs)['add'], 0)
        kwargs['reserved_instances'] = 31
        self.assertEqual(a.scaling_decision(progress(eta=1000000), **kwargs)['add'], 1)

    def test_search_seventy_percent_budget_and_completed_work(self):
        kwargs = dict(now=10000, reserved_instances=8, active_workers=9, max_instances=32)
        self.assertEqual(a.scaling_decision(progress(phase='search', eta=40000), **kwargs)['add'], 0)
        self.assertGreater(a.scaling_decision(progress(phase='search', eta=60000), **kwargs)['add'], 0)
        value = progress(); value['forecast']['phase_remaining'] = 0
        self.assertEqual(a.scaling_decision(value, **kwargs)['add'], 0)
        value = progress(); value['updated_at'] -= 181
        self.assertEqual(a.scaling_decision(value, **kwargs)['reason'], 'controller_heartbeat_stale')


class FakePlatform:
    def __init__(self): self.live = {}; self.calls = []; self.timeout = False; self.visible = True
    def list_name(self, name):
        self.calls.append(('list', name)); return self.live.get(name, [])
    def create(self, argv):
        name = a.option(argv, '--name'); self.calls.append(('create', name))
        if self.visible:
            self.live[name] = [{'name': name, 'status': 'RUNNING', 'workspace': '昇腾卡公共空间',
                               'project': 'project', 'compute_group': '910B资源'}]
        if self.timeout: return {'returncode': None, 'uncertain': True, 'error_type': 'TimeoutExpired'}
        return {'returncode': 0, 'document': {'success': True, 'data': {'url': 'https://SECRET/token'}}}
    def transition(self, name, action):
        self.calls.append((action, name))
        self.live[name][0]['status'] = 'STOPPED' if action == 'stop' else 'RUNNING'
        return {'returncode': 0, 'document': {'success': True}}
    def launch_worker(self, name, config):
        self.calls.append(('launch', name)); return {'returncode': 0, 'document': {'completed': True}}


class AutoscaleTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(); self.root = Path(self.tmp.name)
        names = [f'base-{i}' for i in range(8)]; entries = []
        for name in names:
            argv = ['inspire', '--json', 'notebook', 'create', '--name', name, '--account', 'account',
                    '--workspace', '昇腾卡公共空间', '--project', 'project', '--group', '910B资源',
                    '--quota', '0,16,256', '--image', 'approved-image', '--no-wait', '--no-auto-stop']
            entries.append({'name': name, 'argv': argv, 'owned': True})
        a.save(self.root / 'resources' / 'PLAN.json', {'instances': entries, 'max_instances': 8})
        self.cfg = {'root': str(self.root), 'output': str(self.root / 'index'), 'python': '/usr/bin/python3',
                    'sdk_python': '/sdk/python', 'runtime': str(self.root / 'runtime'), 'input': '/input', 'adopt': '/adopt',
                    'runtime_sha256': {}, 'notebooks': names, 'max_instances': 8, 'account': 'account',
                    'workspace': '昇腾卡公共空间', 'threads': 14, 'deadline_epoch': 70000}
        self.cp = self.root / 'config.json'; a.save(self.cp, self.cfg); self.platform = FakePlatform()
        self.runner = a.Autoscaler(self.cp, max_instances=32, platform=self.platform)
        a.save(self.root / 'controller' / 'PROGRESS.json', progress())
        for name in names + ['coordinator-local']:
            a.save(self.root / 'controller' / 'workers' / (name + '.json'),
                   {'phase': 'catalog', 'child_pid': 1, 'updated_at': 10000, 'status': 'running'})
    def tearDown(self): self.tmp.cleanup()

    def extra_runner(self, name, **kwargs):
        root = self.root / name
        cfg = dict(self.cfg, root=str(root))
        a.save(root / 'resources' / 'PLAN.json', a.read(self.root / 'resources' / 'PLAN.json'))
        cp = root / 'config.json'; a.save(cp, cfg)
        return a.Autoscaler(cp, platform=self.platform, **kwargs)

    def test_initial_target_binding_defaults_128_cap_and_validation(self):
        default = self.extra_runner('default')
        self.assertEqual((default.maximum, default.initial_target), (16, 8))
        large = self.extra_runner('large', max_instances=128, initial_instances_target=64)
        self.assertEqual((large.maximum, large.initial_target), (128, 64))
        resumed = a.Autoscaler(large.cp, platform=self.platform)
        self.assertEqual(resumed.initial_target, 64)
        with self.assertRaisesRegex(ValueError, 'initial target changed'):
            a.Autoscaler(large.cp, initial_instances_target=65, platform=self.platform)
        for name, maximum, target in [('large_target', 128, 129), ('small_target', 16, 7),
                                      ('large_max', 129, 64), ('boolean_max', True, 8)]:
            with self.assertRaises(ValueError):
                self.extra_runner(name, max_instances=maximum, initial_instances_target=target)

    def test_initial_search_target_reserves_in_bounded_waves_without_cooldown(self):
        runner = self.extra_runner('warming', max_instances=128, initial_instances_target=20)
        for number, now in enumerate((10000, 10060, 10120), start=1):
            value = progress(now, phase='search'); value['forecast']['measurement_seconds'] = 0
            a.save(runner.root / 'controller' / 'PROGRESS.json', value)
            result = runner.tick(now)
            self.assertEqual(result['additional_reservations'], number * 4)
            self.assertEqual(result['decision']['reason'], 'explicit_initial_search_target')
        value = progress(10180, phase='search'); value['forecast']['measurement_seconds'] = 0
        a.save(runner.root / 'controller' / 'PROGRESS.json', value)
        self.assertEqual(runner.tick(10180)['additional_reservations'], 12)
        self.assertEqual(sum(action == 'create' for action, _ in self.platform.calls), 12)

    def test_create_wave_owned_ledger_worker_configs_and_no_secret_output(self):
        result = self.runner.tick(10000)
        self.assertEqual(result['additional_reservations'], 4)
        created = [name for action, name in self.platform.calls if action == 'create']
        self.assertEqual(len(created), 4)
        ledger = a.read(self.root / 'resources' / 'PLAN.json'); self.assertEqual(len(ledger['instances']), 12)
        for name in created:
            cfg = a.read(self.runner.d / 'worker_configs' / (name + '.json'))
            self.assertEqual(cfg['notebooks'], [name]); self.assertEqual(cfg['max_instances'], 1)
            self.assertEqual(cfg['runtime_sha256'], self.cfg['runtime_sha256'])
        self.runner.tick(10030)
        self.assertEqual(sum(x[0] == 'create' for x in self.platform.calls), 4)
        for path in self.runner.d.rglob('*.json'):
            self.assertNotIn('SECRET', path.read_text())

    def test_uncertain_create_reconciles_once_without_duplicate_submission(self):
        self.platform.timeout = True; self.platform.visible = False
        self.runner.create_one(10000); name = next(iter(self.runner.state['instances']))
        self.assertEqual(self.runner.state['instances'][name]['state'], 'missing_or_creation_uncertain')
        resumed = a.Autoscaler(self.cp, platform=self.platform)
        resumed.reconcile_one(name, 10301)
        self.assertEqual(sum(x[0] == 'create' for x in self.platform.calls), 1)
        self.assertEqual(len(a.read(self.root / 'resources' / 'PLAN.json')['instances']), 8)
        self.platform.live[name] = [{'name': name, 'status': 'RUNNING', 'workspace': '昇腾卡公共空间',
                                    'project': 'project', 'compute_group': '910B资源'}]
        resumed.reconcile_one(name, 10602)
        self.assertTrue(resumed.state['instances'][name]['owned'])
        self.assertEqual(sum(x[0] == 'create' for x in self.platform.calls), 1)

    def test_unsubmitted_intent_recovers_before_attempt_marker(self):
        name = self.runner.binding['name_prefix'] + '-09'
        argv = list(self.runner.template); argv[argv.index('--name') + 1] = name
        path = self.runner.d / 'intents' / (name + '.json')
        a.save(path, {'name': name, 'argv': argv, 'attempted': False, 'preflight_absent': False})
        # Simulate termination before STATE.json, which must not lose the intent.
        value = progress(); value['forecast']['measurement_seconds'] = 0
        a.save(self.root / 'controller' / 'PROGRESS.json', value)
        result = self.runner.tick(10000)
        self.assertEqual(result['additional_reservations'], 1)
        self.assertTrue(self.runner.state['instances'][name]['owned'])
        self.assertEqual(sum(x[0] == 'create' for x in self.platform.calls), 1)

    def test_collision_never_adopted_or_stopped(self):
        name = self.runner.binding['name_prefix'] + '-09'
        self.platform.live[name] = [{'name': name, 'status': 'RUNNING', 'workspace': '昇腾卡公共空间',
                                    'project': 'project', 'compute_group': '910B资源'}]
        self.runner.create_one(10000); self.runner.reconcile_one(name, 11000, stopping=True)
        self.assertEqual(self.runner.state['instances'][name]['state'], 'name_collision')
        self.assertFalse(any(x[0] in ('create', 'stop', 'launch') for x in self.platform.calls))
        self.assertEqual(len(a.read(self.root / 'resources' / 'PLAN.json')['instances']), 8)

    def test_restart_and_completion_cleanup_excludes_initial_fleet(self):
        self.runner.create_one(10000); name = next(iter(self.runner.state['instances']))
        self.platform.live[name][0]['status'] = 'STOPPED'
        self.runner.reconcile_one(name, 11000)
        self.assertIn(('start', name), self.platform.calls)
        value = progress(12000, phase='complete'); a.save(self.root / 'controller' / 'PROGRESS.json', value)
        self.runner.persist(); self.runner.tick(12000)
        self.assertIn(('stop', name), self.platform.calls)
        final = self.runner.tick(12130); self.assertTrue(final['cleanup_complete'])
        self.assertFalse(any(action == 'stop' and target in self.cfg['notebooks'] for action, target in self.platform.calls))

    def test_fatal_worker_is_not_relaunched_but_still_released_on_completion(self):
        self.runner.create_one(10000); name = next(iter(self.runner.state['instances']))
        a.save(self.root / 'controller' / 'workers' / (name + '.json'),
               {'phase': 'search', 'child_pid': None, 'updated_at': 10000, 'status': 'failed', 'fatal': True})
        before = len(self.platform.calls)
        self.runner.reconcile_one(name, 20000)
        self.assertEqual(self.runner.state['instances'][name]['state'], 'await_worker_repair')
        self.assertFalse(any(action in ('start', 'launch') for action, _ in self.platform.calls[before:]))
        self.runner.reconcile_one(name, 20000, stopping=True)
        self.assertIn(('stop', name), self.platform.calls)

    def test_budget_binding_foreign_profile_and_live_identity_fail_closed(self):
        with self.assertRaisesRegex(ValueError, 'budget changed'): a.Autoscaler(self.cp, max_instances=16, platform=self.platform)
        with self.assertRaisesRegex(ValueError, 'Maximum'):
            other = dict(self.cfg, root=str(self.root / 'new'))
            a.save(self.root / 'new' / 'resources' / 'PLAN.json', a.read(self.root / 'resources' / 'PLAN.json'))
            a.save(self.root / 'new.json', other); a.Autoscaler(self.root / 'new.json', max_instances=129, platform=self.platform)
        self.runner.create_one(10000); name = next(iter(self.runner.state['instances']))
        self.platform.live[name][0]['project'] = 'foreign'
        with self.assertRaisesRegex(ValueError, 'identity mismatch'):
            self.runner.reconcile_one(name, 11000, stopping=True)
        self.assertNotIn(('stop', name), self.platform.calls)

    def test_singleton_thread_lock(self):
        results = []
        def second():
            with a.lock(self.root / 'singleton') as held: results.append(held)
        with a.lock(self.root / 'singleton') as held:
            self.assertTrue(held); thread = threading.Thread(target=second); thread.start(); thread.join(timeout=5)
            self.assertEqual(results, [False])
        second(); self.assertEqual(results, [False, True])


if __name__ == '__main__': unittest.main()
