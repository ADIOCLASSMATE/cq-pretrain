"""Offline official launch and external-observer contracts; no SDK requests."""
import copy
import json
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import MagicMock, patch

from inspire.services.execution.remote_exec import ExecResult, select_exec_instance
from corpus_pipeline.serving.qwen36_w8a8_20261024.official_v023 import backend as mod


class OfficialBackendTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.assets = self.root / 'assets'
        (self.assets / 'weights').mkdir(parents=True)
        self.ready = dict(complete=True, precision='W8A8_DYNAMIC', quantization='ascend',
            model_path='/verified/local/model', revision='fixed-revision',
            processor_revision='sha256:processor')
        mod.native.save(self.assets / 'weights/WEIGHTS_READY.json', self.ready)
        mod.native.save(self.assets / 'MODEL_REGISTERED.json', {
            'model_path': self.ready['model_path'], 'revision': self.ready['revision'], 'model_id': 'model-owned'})
        for item in (patch.object(mod, 'ASSETS', self.assets),
                     patch.object(mod, 'asset_receipt', return_value=self.ready),
                     patch.object(mod.pool, 'configure_management_proxy_bypass')):
            item.start()
            self.addCleanup(item.stop)
        self.contract = mod.official_contract()
        self.binding = {'launch_contract_sha256': mod.digest(self.contract), 'frozen': True}
        self.backend = mod.Backend()
        item = patch.object(self.backend, 'binding', return_value=(self.contract, self.binding))
        item.start(); self.addCleanup(item.stop)

    def state(self, priority=3):
        def plan(spec):
            return NS(create_kwargs={'mirror_id': mod.IMAGE_ID, 'model_id': 'model-owned',
                'task_priority': spec.priority, 'replicas': 1, 'node_num_per_replica': 1,
                'resource_spec_price': self.contract['quota']})
        client = MagicMock()
        client.servings.plan.side_effect = plan
        with patch.object(mod, 'InspireClient') as factory, patch.object(mod.pool, 'observable_project_budget'):
            factory.return_value.__enter__.return_value = client
            return self.backend.prepare(self.root / ('high' if priority in (4, 6) else 'low'), 1,
                                        'owned-high' if priority in (4, 6) else 'owned-low', priority)

    def test_current_priorities_preserve_tier_recovery_and_payload(self):
        for priority, tier, recover in ((1, 'low', True), (3, 'low', True),
                                        (4, 'high', False), (6, 'high', False)):
            with self.subTest(priority=priority):
                state = self.state(priority)
                self.assertEqual(state['tier'], tier)
                self.assertEqual(state['auto_recovery_enabled'], recover)
                self.assertEqual(state['payload']['task_priority'], priority)
                self.assertEqual(state['create_spec']['priority'], priority)
                other = copy.deepcopy(state)
                other['tier'] = 'high' if tier == 'low' else 'low'
                with self.assertRaises(ValueError):
                    self.backend.validate(other)
        for priority in (True, 0, 2, 5, '1'):
            with self.subTest(rejected_priority=priority), self.assertRaises(ValueError):
                self.state(priority)

    def test_platform_running_alone_never_publishes_ready(self):
        state = self.state(1)
        state.update(service_id='sv-owned', endpoint='https://invalid.example')
        client = MagicMock()
        detail = NS(ref=NS(key='sv-owned'), status='RUNNING',
                    to_dict=lambda: {'endpoint': state['endpoint']})
        client.servings.instances.return_value = [NS(status='INSTANCE_RUNNING')]
        client.servings.events.return_value = NS(items=[])
        proof = {'incarnation': 'current', 'platform_instance_id': 'project/pod',
                 'runtime_revision': 'official-native-sha256:runtime'}
        for health, smoke, qualified in ((False, True, False), (True, False, False), (True, True, True)):
            with self.subTest(health=health, smoke=smoke), patch.object(mod, 'InspireClient') as factory, \
                    patch.object(self.backend, 'reconcile', return_value=(dict(state), detail)), \
                    patch.object(self.backend, 'attest', return_value=proof), \
                    patch.object(self.backend, 'health', return_value=health), \
                    patch.object(self.backend, 'smoke', return_value=smoke) as request:
                factory.return_value.__enter__.return_value = client
                observed = self.backend.observe(self.root, state)
                self.assertEqual(observed['canary_passed'], qualified)
                self.assertEqual(mod.native.load(self.root / 'registry.json')['endpoints'][0]['ready'], qualified)
                self.assertEqual(request.called, health)

    def test_literal_official_command_has_only_local_asset_substitution(self):
        raw = (mod.AUDIT / 'OFFICIAL_VLLM_SERVE_V023.txt').read_text()
        expected = shlex.split(raw.replace('\\\n', ' '))
        expected[2] = self.ready['model_path']
        self.assertEqual(self.contract['argv'], expected)
        argv = self.contract['argv']
        for flag in ('--dtype', '--speculative-config', '--enforce-eager'):
            self.assertNotIn(flag, argv)
        graph = json.loads(argv[argv.index('--compilation-config') + 1])
        self.assertEqual(graph, {'cudagraph_mode': 'FULL_DECODE_ONLY'})
        self.assertEqual(argv[argv.index('--tensor-parallel-size') + 1], '2')
        self.assertEqual(argv[argv.index('--data-parallel-size') + 1], '1')
        self.assertIn('--enable-expert-parallel', argv)
        self.assertIn('--async-scheduling', argv)
        self.assertNotIn('gateway.py', self.contract['command'])
        self.assertNotIn('serve.py', self.contract['command'])
        self.assertNotIn('sysctl', self.contract['command'])
        self.assertNotIn('scaling_governor', self.contract['command'])

    def test_high_low_share_launch_contract_and_differ_only_resource_identity(self):
        high, low = self.state(6), self.state(3)
        for key in ('model', 'revision', 'precision', 'processor_revision', 'binding',
                    'launch_contract_sha256', 'request_overrides'):
            self.assertEqual(high[key], low[key])
        self.assertEqual(high['create_spec']['command'], low['create_spec']['command'])
        self.assertEqual(high['cards'], 2)
        self.assertFalse(high['auto_recovery_enabled'])
        self.assertTrue(low['auto_recovery_enabled'])
        self.assertEqual(high['priority'], 6)
        self.assertEqual(low['priority'], 3)

    def test_wrong_resolved_image_or_quota_rejected_before_create(self):
        for key, value in (('mirror_id', 'wrong-image'), ('resource_spec_price',
                           {'gpu_count': 4, 'cpu_count': 16, 'memory_size_gib': 128})):
            client = MagicMock()
            payload = dict(mirror_id=mod.IMAGE_ID, model_id='model-owned', task_priority=3,
                           replicas=1, node_num_per_replica=1, resource_spec_price=self.contract['quota'])
            payload[key] = value
            client.servings.plan.return_value = NS(create_kwargs=payload)
            with self.subTest(key=key), patch.object(mod, 'InspireClient') as factory, \
                    patch.object(mod.pool, 'observable_project_budget'):
                factory.return_value.__enter__.return_value = client
                with self.assertRaises(ValueError):
                    self.backend.prepare(self.root / 'bad', 1, 'owned-bad', 3)
                client.servings.create.assert_not_called()

    def test_native_registry_declares_platform_attestation_without_boot_header(self):
        state = self.state()
        state.update(service_id='sv-owned', endpoint='https://invalid.example', incarnation='incarnation',
                     platform_instance_id='project/pod', runtime_revision='official-native-sha256:runtime',
                     qualification_receipt='/fake/basic-smoke.json')
        self.backend.publish(self.root, state, True)
        row = mod.native.load(self.root / 'registry.json')['endpoints'][0]
        self.assertTrue(row['ready'])
        self.assertGreater(row['ready_expires_at'], 0)
        self.assertEqual(row['transport_mode'], 'native_openai')
        self.assertEqual(row['attestation_scope'], 'platform_instance_and_registry')
        self.assertTrue(row['native_stream'])
        self.assertFalse(row['response_boot_attested'])
        self.assertEqual(row['expected_boots'], [])
        self.assertEqual(row['qualification_kind'], 'basic_smoke')
        state.pop('endpoint')
        self.backend.publish(self.root, state, False)
        row = mod.native.load(self.root / 'registry.json')['endpoints'][0]
        self.assertFalse(row['ready'])
        self.assertEqual(row['ready_expires_at'], 0)

    def attest_fixture(self):
        state = self.state(); state['service_id'] = 'sv-owned'
        instance = NS(status='INSTANCE_RUNNING', raw={'name': 'project/pod', 'node': 'node-owned',
            'created_at': 1000, 'started_at': 1001, 'restarts': 0, 'term': 1})
        versions = {'vllm': '0.23.0', 'vllm-ascend': '0.23.0rc1', 'torch': '2.10.0+cpu',
                    'torch-npu': '2.10.0.post2', 'transformers': '5.0.0'}
        client = MagicMock()
        output = json.dumps(versions) + '\n'
        client.servings.exec.return_value = ExecResult(0, output, output, '', True, 'pty')
        return state, instance, versions, client

    def test_current_instance_metadata_checked_once_and_rechecked_after_restart(self):
        state, instance, versions, client = self.attest_fixture()
        detail = NS(ref=NS(key='sv-owned'))
        first = self.backend.attest(client, detail, [instance], state, self.root)
        cached = self.backend.attest(client, detail, [instance], state, self.root)
        self.assertEqual(first, cached)
        self.assertEqual(client.servings.exec.call_count, 1)
        command = client.servings.exec.call_args.kwargs['command']
        self.assertNotIn('import torch', command)
        instance.raw.update(started_at=2000, restarts=1)
        restarted = self.backend.attest(client, detail, [instance], state, self.root)
        self.assertNotEqual(first['incarnation'], restarted['incarnation'])
        self.assertEqual(first['runtime_revision'], restarted['runtime_revision'])
        self.assertEqual(client.servings.exec.call_count, 2)
        self.assertFalse(restarted['response_boot_attested'])

    def test_wrong_package_pair_rejected_without_receipt(self):
        state, instance, versions, client = self.attest_fixture()
        versions['vllm'] = '0.30.0'
        text = json.dumps(versions)
        client.servings.exec.return_value = ExecResult(0, text, text, '', True, 'pty')
        with self.assertRaises(ValueError):
            self.backend.attest(client, NS(ref='ref'), [instance], state, self.root)
        self.assertFalse((self.root / 'attestations').exists())

    def test_smoke_receipt_rejects_old_incarnation_and_false_process_attestation(self):
        state = self.state()
        state.update(incarnation='new', platform_instance_id='project/pod', runtime_revision='runtime')
        path = self.root / 'smoke/new/RESULT.json'
        result = {key: state[key] for key in ('model', 'revision', 'precision', 'processor_revision',
            'runtime_revision', 'incarnation', 'platform_instance_id', 'launch_contract_sha256')}
        result.update(passed=True, response_boot_attested=False, qualification_kind='basic_smoke')
        mod.native.save(path, result)
        self.assertTrue(self.backend.smoke(self.root, state))
        for key, value in (('incarnation', 'old'), ('response_boot_attested', True)):
            changed = copy.deepcopy(result); changed[key] = value; mod.native.save(path, changed)
            with self.subTest(key=key), self.assertRaises(ValueError):
                self.backend.smoke(self.root, state)

    def test_real_sdk_selector_accepts_exact_owned_handle(self):
        rows = [{'name': 'project/sv-owned-0', 'status': 'INSTANCE_RUNNING', 'rank': 0}]
        self.assertEqual(select_exec_instance('serving', rows, 'project/sv-owned-0'), 'project/sv-owned-0')

    def test_explicit_asset_root_keeps_original_verified_launch(self):
        audit = self.assets / 'weights/official_version_audit'
        audit.mkdir()
        for name in ('guide-v023.md', 'OFFICIAL_VLLM_SERVE_V023.txt'):
            (audit / name).write_bytes((mod.AUDIT / name).read_bytes())
        expected = self.contract
        mod.native.save(self.assets / 'official_v023/CONTRACT.json', expected)
        (self.assets / 'official_v023/smoke.py').write_bytes((mod.ROOT / 'smoke.py').read_bytes())
        # The explicit root must work without machine-local deployment receipts.
        with patch.object(mod, 'ASSETS', self.root / 'unused'), \
             patch.object(mod, 'ROOT', self.root / 'unused'), \
             patch.object(mod, 'AUDIT', self.root / 'unused'):
            backend = mod.Backend(asset_root=self.assets)
            contract, binding = backend.binding()
        self.assertEqual(contract, expected)
        self.assertEqual(backend.assets, self.assets)
        self.assertEqual(binding['launch_contract_sha256'], mod.digest(expected))
        self.assertEqual(binding['backend_sha256'], mod.file_sha(mod.__file__))


if __name__ == '__main__':
    unittest.main()
