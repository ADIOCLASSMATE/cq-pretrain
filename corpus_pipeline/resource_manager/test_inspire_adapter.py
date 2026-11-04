"""Fake SDK contract tests: no platform requests or real resource operations."""
from __future__ import annotations

from contextlib import contextmanager, nullcontext
import copy
import json
from pathlib import Path
import shlex
import tempfile
from types import SimpleNamespace as NS
import unittest
from unittest.mock import Mock, patch

from .inspire_adapter import InspireAdapter


class FakeTransport:
    session = object()
    def __init__(self): self.sends = []; self.sending = False
    def scope(self, **kwargs): return nullcontext()
    @contextmanager
    def single_send(self, operation_id, **kwargs):
        if self.sending: raise RuntimeError('single_send blocks cannot be nested')
        self.sends.append((operation_id, kwargs))
        self.sending = True
        try: yield
        finally: self.sending = False


class FakeService:
    def __init__(self, backend):
        self.backend = backend
        self.details = {}
        self.raws = {}
        self.last_spec = None
        self.wrong_id = None
        self.exec = Mock(return_value=NS(completed=True, returncode=0))
        self.stop = Mock()

    def quotas(self, *args, **kwargs):
        return NS(items=[NS(to_dict=lambda: {'compute_group': 'group', 'allowed_priority_levels': ['LOW'],
                                            'quota': '2,16,128'})])

    def plan(self, spec):
        self.last_spec = spec
        gpu, cpu, memory = map(int, spec.quota.split(','))
        payload = {'name': spec.name, 'workspace_id': 'workspace-id', 'project_id': 'project-id',
                   'logic_compute_group_id': 'group-id', 'task_priority': spec.priority,
                   'is_publicpath_readonly': spec.public_path_readonly}
        if self.backend == 'notebook':
            payload.update(gpu_count=gpu, cpu_count=cpu, memory_size=memory,
                           node_id=spec.node, shared_memory_size=spec.shm_gib, quota_id='quota-id',
                           auto_stop=spec.auto_stop, image_id='image-id', image_url='fixture-image-url')
            if spec.datasets:
                from inspire.services.catalog.datasets import parse_dataset_specs
                payload['dataset_info'] = [mount.as_payload('/fixture/datasets/' + mount.dataset + '/' + mount.version)
                                           for mount in parse_dataset_specs(spec.datasets)]
        else:
            payload.update(model_id='model-id', model_version=spec.model_version, mirror_id='image-id',
                           command=spec.command, port=spec.port, replicas=spec.replicas,
                           node_num_per_replica=spec.nodes_per_replica,
                           resource_spec_price={'gpu_count': gpu, 'cpu_count': cpu,
                                                'memory_size_gib': memory, 'quota_id': 'quota-id',
                                                'logic_compute_group_id': 'group-id'})
        return NS(create_kwargs=payload, workspace=NS(name='workspace'), project=NS(name='project'),
                  group=NS(name='group'), priority=spec.priority)

    def register(self, payload, identifier='resource-id'):
        raw = dict(payload)
        if self.backend == 'notebook':
            raw.update(creator={'id': 'owner-id'}, workspace={'id': 'workspace-id'}, project={'id': 'project-id'},
                       start_config={key: payload[key] for key in ('logic_compute_group_id', 'gpu_count', 'cpu_count',
                         'memory_size', 'node_id', 'shared_memory_size', 'quota_id', 'auto_stop')})
            raw['start_config'].update(mirror_id=payload['image_id'], mirror_url=payload['image_url'])
        else:
            raw['user_id'] = 'owner-id'
        self.raws[identifier] = raw
        detail = NS(ref=NS(key=identifier, workspace_id='workspace-id'), name=payload['name'],
                    project='project', workspace='workspace', priority=3, priority_level='LOW',
                    status='RUNNING', node={'name': payload.get('node_id')}, raw=raw)
        self.details[identifier] = detail
        return identifier

    def get(self, ref, **kwargs):
        detail = self.details[ref.key]
        if self.wrong_id:
            value = copy.copy(detail)
            value.ref = NS(key=self.wrong_id, workspace_id='workspace-id')
            return value
        return detail

    def iter(self, **kwargs): return iter(self.details.values())


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(); self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def adapter(self, backend='notebook', **overrides):
        config = {'account': 'fixture-account', 'workspace': 'workspace', 'project': 'project',
                  'group': 'group', 'image': 'image', 'priority': 3,
                  'registry_dir': str(self.root / backend), 'backend': backend}
        if backend == 'notebook':
            config.update(repository='/fixture/repo', remote_python='/fixture/python', worker_state_dir='/fixture/state')
        else:
            config.update(model='fixture-model', model_version=1, port=8000)
        config.update(overrides)
        adapter = InspireAdapter(config)
        service = FakeService(backend)
        image = NS(status='SUCCESS', ref=NS(key='image-id'), url='fixture-image-url')
        transport = FakeTransport()
        client = NS(_transport=transport, base_url='https://fixture.invalid', cache=NS(clear=lambda: None),
                    images=NS(get=lambda *args, **kwargs: image), notebooks=service if backend == 'notebook' else None,
                    servings=service if backend == 'serving' else None)
        adapter._client = client
        adapter._owner = lambda: 'owner-id'
        adapter._global_budget = lambda: {'global_remaining_cards': 256, 'resources': {}, 'observable_requested_cards': 0}
        return adapter, service, transport

    def intent(self, backend='notebook'):
        return {'operation_id': 'operation-id', 'name': 'managed-fixture', 'scope': 'scope', 'profile': 'profile',
                'placement': {'node': 'node-a' if backend == 'notebook' else 'native-scheduler',
                              'quota': {'cards': 2, 'cpu': 16, 'memory': 128}},
                'command': ['python', 'worker.py', 'literal;$(not-shell)'],
                'spec': {'replica_cards': 2 if backend == 'serving' else None,
                         'workload_kind': 'service' if backend == 'serving' else 'batch'}}

    def create(self, adapter, service, intent, *, uncertain=False):
        method = 'create_notebook' if adapter.backend == 'notebook' else 'create_serving'
        module = 'notebooks' if adapter.backend == 'notebook' else 'servings'
        def send(**payload):
            payload.pop('session')
            identifier = service.register(payload)
            if uncertain:
                raise TimeoutError('fixture unknown result')
            return {'id': identifier}
        with patch('inspire.platform.web.browser_api.' + module + '.' + method, side_effect=send) as call:
            if uncertain:
                with self.assertRaises(TimeoutError): adapter.create(intent)
                return call
            identifier = adapter.create(intent)
            self.assertEqual(call.call_count, 1)
            return identifier

    def raw_patch(self, service):
        return patch('inspire.platform.web.browser_api.notebooks.get_notebook_detail',
                     side_effect=lambda identifier, **kwargs: service.raws[identifier])

    def test_explicit_configuration_and_exact_node_single_send(self):
        with self.assertRaisesRegex(ValueError, 'Explicit adapter field'):
            InspireAdapter({})
        adapter, service, transport = self.adapter()
        identifier = self.create(adapter, service, self.intent())
        self.assertEqual(service.last_spec.node, 'node-a')
        self.assertEqual(service.last_spec.quota, '2,16,128')
        self.assertTrue(service.last_spec.public_path_readonly)
        self.assertFalse(service.last_spec.auto_stop)
        self.assertEqual(service.last_spec.datasets, [])
        self.assertEqual(len(transport.sends), 1)
        with self.raw_patch(service):
            self.assertEqual(adapter.observe(identifier)['state'], 'RUNNING')

    def test_fair_low_and_explicit_writable_derived_public_output(self):
        adapter, service, transport = self.adapter(priority=1, public_path_readonly=False)
        original_plan = service.plan
        def planning(spec):
            plan = original_plan(spec)
            plan.create_kwargs['gpu_type'] = 'planning-only'
            return plan
        service.plan = planning
        self.create(adapter, service, self.intent())
        self.assertEqual(service.last_spec.priority, 1)
        self.assertFalse(service.last_spec.public_path_readonly)
        self.assertNotIn('gpu_type', service.raws['resource-id'])
        self.assertEqual(len(transport.sends), 1)
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            self.adapter(public_path_readonly='false')

    def test_native_notebook_scheduler_supports_explicit_low_and_preserves_payload_checks(self):
        for priority in (1, 3):
            adapter, _, _ = self.adapter(priority=priority, notebook_placement='native_scheduler')
            self.assertFalse(adapter.capabilities['node_placement'])
        with self.assertRaisesRegex(ValueError, 'Notebook backend'):
            self.adapter(backend='serving', notebook_placement='native_scheduler')
        with self.assertRaisesRegex(ValueError, 'placement'):
            self.adapter(notebook_placement='automatic')
        adapter, service, _ = self.adapter(priority=1, notebook_placement='native_scheduler',
                                           excluded_nodes=['bad-node'], global_max_cards=128)
        service.quotas = lambda *a, **kw: NS(items=[NS(to_dict=lambda: {
            'compute_group': 'group', 'allowed_priority_levels': ['LOW'], 'quota': '2,16,128'})])
        intent = self.intent(); intent['placement']['node'] = 'native-scheduler'
        identifier = self.create(adapter, service, intent)
        self.assertIsNone(service.last_spec.node)
        self.assertFalse(service.last_spec.auto_stop)
        self.assertFalse(adapter.capabilities['node_placement'])
        detail = service.details[identifier]; detail.priority = 1; detail.node = {'name': 'chosen-node'}
        with self.raw_patch(service):
            service.raws[identifier]['start_config']['node_id'] = ''
            self.assertEqual(adapter.observe(identifier)['state'], 'RUNNING')
            service.raws[identifier]['start_config']['node_id'] = 'unexpected-bound-node'
            with self.assertRaisesRegex(ValueError, 'payload and owner'):
                adapter.observe(identifier)
            service.raws[identifier]['start_config']['node_id'] = ''
            service.raws[identifier]['start_config']['cpu_count'] = 999
            with self.assertRaisesRegex(ValueError, 'payload and owner'):
                adapter.observe(identifier)
            service.raws[identifier]['start_config']['cpu_count'] = 16
            detail.node = {'name': 'bad-node'}
            with self.assertRaisesRegex(RuntimeError, 'excluded worker node'):
                adapter.launch(identifier, intent['command'], 'launch-id')
            adapter.stop(identifier, 'stop-id')
            service.stop.assert_called_once()

    def test_native_scheduler_snapshot_is_budget_capacity_without_physical_availability_claim(self):
        adapter, service, _ = self.adapter(priority=6, notebook_placement='native_scheduler',
                                           global_max_cards=128)
        service.quotas = lambda *a, **kw: NS(items=[NS(to_dict=lambda: {
            'compute_group': 'group', 'allowed_priority_levels': ['HIGH'], 'quota': '2,16,128'})])
        adapter._global_budget = lambda: {'global_remaining_cards': 120, 'resources': {},
                                          'observable_requested_cards': 8}
        with patch('inspire.platform.web.browser_api.availability.api.list_node_dimension',
                   side_effect=AssertionError('Native queue does not bind current free nodes')):
            result = adapter.snapshot()
        self.assertEqual(result['global_remaining_cards'], 120)
        self.assertEqual(result['nodes'][0]['name'], 'native-scheduler')
        self.assertFalse(result['capabilities']['node_placement'])
        self.assertEqual(result['capacity_scope'], 'request_budget_not_physical_availability')

    def test_optional_unsupported_public_path_field_is_omitted_without_weakening_other_bindings(self):
        adapter, service, _ = self.adapter(omit_unsupported_public_path_flag=True)
        identifier = self.create(adapter, service, self.intent())
        self.assertNotIn('is_publicpath_readonly', service.raws[identifier])
        with self.raw_patch(service):
            self.assertEqual(adapter.observe(identifier)['state'], 'RUNNING')
            service.raws[identifier]['start_config']['gpu_count'] = 99
            with self.assertRaisesRegex(ValueError, 'payload and owner'):
                adapter.observe(identifier)
        with self.assertRaisesRegex(ValueError, 'must be boolean'):
            self.adapter(omit_unsupported_public_path_flag='true')

    def test_official_dataset_mounts_are_explicit_copied_and_bound(self):
        datasets = ['dcvlm-balanced-200b:v1', 'fixture-extra:v2']
        adapter, service, _ = self.adapter(datasets=datasets)
        datasets.append('unrequested:v3')
        identifier = self.create(adapter, service, self.intent())
        self.assertEqual(service.last_spec.datasets, ['dcvlm-balanced-200b:v1', 'fixture-extra:v2'])
        service.last_spec.datasets.append('later-sdk-mutation:v4')
        self.assertEqual(adapter.config['datasets'], ['dcvlm-balanced-200b:v1', 'fixture-extra:v2'])
        raw = service.raws[identifier]
        self.assertEqual(raw['dataset_info'][0], {'dataset_id': 'dcvlm-balanced-200b', 'version_id': 'v1',
                         'path': '/fixture/datasets/dcvlm-balanced-200b/v1'})
        raw['dataset_info'].reverse()
        with self.raw_patch(service):
            self.assertEqual(adapter.observe(identifier)['state'], 'RUNNING')
        original = copy.deepcopy(raw['dataset_info'])
        for changed in (None, [], [original[0]], [*original, original[0]],
                        [dict(original[0], version_id='wrong'), original[1]],
                        [dict(original[0], path='/another/storage/path'), original[1]]):
            with self.subTest(mounts=changed):
                raw['dataset_info'] = changed
                with self.raw_patch(service), self.assertRaisesRegex(ValueError, 'dataset mounts'):
                    adapter.stop(identifier, 'stop-operation')
        service.stop.assert_not_called()

    def test_dataset_mount_plan_cannot_drop_or_replace_requested_mounts(self):
        for mode in ('drop', 'replace', 'unresolved'):
            with self.subTest(mode=mode):
                adapter, service, transport = self.adapter(datasets=['dcvlm-balanced-200b:v1'])
                plan = service.plan
                def wrong_plan(spec):
                    result = plan(spec)
                    if mode == 'drop':
                        result.create_kwargs['dataset_info'] = []
                    elif mode == 'replace':
                        result.create_kwargs['dataset_info'][0]['dataset_id'] = 'other-dataset'
                    else:
                        result.create_kwargs['dataset_info'][0].pop('path')
                    return result
                service.plan = wrong_plan
                with self.assertRaisesRegex(ValueError, 'dataset mounts'):
                    adapter.create(self.intent())
                self.assertEqual(transport.sends, [])
                self.assertFalse(adapter._operation_path('operation-id').exists())

    def test_invalid_dataset_configuration_and_serving_mounts_are_rejected(self):
        for datasets in ('dataset:v1', None, [None], ['']):
            with self.subTest(datasets=datasets), self.assertRaisesRegex(ValueError, 'datasets must be a list'):
                self.adapter(datasets=datasets)
        with self.assertRaisesRegex(ValueError, 'Notebook backend'):
            self.adapter('serving', datasets=['dcvlm-balanced-200b:v1'])

    def test_running_notebook_remains_reserved_until_node_capacity_lists_exact_allocation(self):
        adapter, _, _ = self.adapter()
        adapter._global_budget = lambda: {'global_remaining_cards': 100, 'resources': {
            'accounted': {'cards': 8, 'state': 'RUNNING', 'node': 'node-a', 'capacity_accounted': True},
            'ahead-of-inventory': {'cards': 4, 'state': 'RUNNING', 'node': 'node-a', 'capacity_accounted': True},
            'wrong-node': {'cards': 2, 'state': 'RUNNING', 'node': 'node-b', 'capacity_accounted': True},
            'pending': {'cards': 1, 'state': 'PENDING', 'node': 'node-a', 'capacity_accounted': False}},
            'observable_requested_cards': 15}
        nodes = [{'name': 'node-a', 'status': 'Ready', 'gpu_type': 'ASCEND_910B_64G',
                  'gpu': {'available': 8}, 'cpu': {'available': 100}, 'memory': {'available': 1000},
                  'tasks_associated': {'tasks': [{'id': 'accounted'}, {'id': 'wrong-node'}, {'id': 'pending'}]}},
                 {'name': 'node-b', 'status': 'Ready', 'gpu_type': 'ASCEND_910B_64G',
                  'gpu': {'available': 8}, 'cpu': {'available': 100}, 'memory': {'available': 1000}}]
        with patch('inspire.platform.web.browser_api.availability.api.list_node_dimension',return_value=nodes):
            values = adapter.snapshot()['resources']
        self.assertTrue(values['accounted']['capacity_accounted'])
        for key in ('ahead-of-inventory', 'wrong-node', 'pending'):
            self.assertFalse(values[key]['capacity_accounted'],key)

    def test_public_stop_owns_single_send_without_nested_transport_context(self):
        for backend in ('notebook', 'serving'):
            with self.subTest(backend=backend):
                adapter, service, transport = self.adapter(backend)
                identifier = self.create(adapter, service, self.intent(backend))
                def public_sdk_stop(ref, **kwargs):
                    with transport.single_send('sdk-native-stop', inspect=backend):
                        self.assertEqual(ref.key,identifier)
                        self.assertEqual(kwargs['workspace'],'workspace')
                service.stop.side_effect = public_sdk_stop
                bound_context = self.raw_patch(service) if backend=='notebook' else nullcontext()
                with bound_context:adapter.stop(identifier,'manager-durable-stop')
                service.stop.assert_called_once()
                self.assertEqual(len(transport.sends),2)

    def test_owner_quota_and_profile_mismatches_fail_before_lifecycle_action(self):
        adapter, service, _ = self.adapter()
        identifier = self.create(adapter, service, self.intent())
        original = copy.deepcopy(service.raws[identifier])
        changes = [lambda raw: raw.update(creator={'id': 'other-owner'}),
                   lambda raw: raw['start_config'].update(gpu_count=4),
                   lambda raw: raw['start_config'].update(node_id='other-node'),
                   lambda raw: raw.update(project={'id': 'other-project'})]
        for change in changes:
            with self.subTest(change=change):
                service.raws[identifier] = copy.deepcopy(original)
                change(service.raws[identifier])
                with self.raw_patch(service), self.assertRaisesRegex(ValueError, 'submitted payload and owner'):
                    adapter.stop(identifier, 'stop-operation')
        self.assertEqual(service.stop.call_count, 0)
        service.raws[identifier] = original
        service.details[identifier].project = 'other-project'
        with self.raw_patch(service), self.assertRaisesRegex(ValueError, 'profile differs'):
            adapter.observe(identifier)

    def test_actual_assigned_node_and_exact_lookup_id_are_verified(self):
        adapter, service, _ = self.adapter()
        identifier = self.create(adapter, service, self.intent())
        service.details[identifier].node = {'name': 'wrong-actual-node'}
        with self.raw_patch(service), self.assertRaises(ValueError):
            adapter.observe(identifier)
        service.details[identifier].node = {'name': 'node-a'}
        service.wrong_id = 'wrong-resource-id'
        with self.raw_patch(service), self.assertRaisesRegex(ValueError, 'different.*ID'):
            adapter.observe(identifier)

    def test_unknown_create_reconciles_exact_identity_without_resending(self):
        adapter, service, transport = self.adapter()
        intent = self.intent()
        call = self.create(adapter, service, intent, uncertain=True)
        with self.assertRaisesRegex(ValueError, 'durable intent'):
            adapter.create(intent)
        with self.raw_patch(service):
            self.assertEqual(adapter.reconcile(intent), 'resource-id')
            self.assertEqual(adapter.reconcile(intent), 'resource-id')
        self.assertEqual(call.call_count, 1)
        self.assertEqual(len(transport.sends), 1)
        changed = copy.deepcopy(intent); changed['profile'] = 'other-profile'
        with self.assertRaisesRegex(ValueError, 'intent changed'):
            adapter.reconcile(changed)

    def test_reconcile_rejects_get_response_with_another_id(self):
        adapter, service, _ = self.adapter()
        intent = self.intent(); self.create(adapter, service, intent, uncertain=True)
        service.wrong_id = 'wrong-resource-id'
        with self.raw_patch(service), self.assertRaisesRegex(ValueError, 'different.*ID'):
            adapter.reconcile(intent)

    def test_worker_command_is_bound_and_passed_as_argv(self):
        adapter, service, _ = self.adapter()
        intent = self.intent(); identifier = self.create(adapter, service, intent)
        adapter._runner = Mock(return_value={'running': True})
        with self.raw_patch(service):
            with self.assertRaisesRegex(ValueError, 'command differs'):
                adapter.launch(identifier, ['another-command'], 'launch')
            adapter.launch(identifier, intent['command'], 'launch')
        adapter._runner.assert_called_once_with(identifier, ['launch', '--operation-id', 'launch', '--',
            'env', 'RESOURCE_MANAGER_CARDS=2', 'RESOURCE_MANAGER_ALLOCATION_ID=operation-id', *intent['command']])

    def test_serving_requires_fixed_replica_and_rejects_exact_node(self):
        adapter, service, _ = self.adapter('serving')
        intent = self.intent('serving')
        wrong = copy.deepcopy(intent); wrong['spec']['replica_cards'] = None
        with self.assertRaisesRegex(ValueError, 'fixed replica_cards'):
            adapter.create(wrong)
        wrong = copy.deepcopy(intent); wrong['placement']['node'] = 'node-a'
        with self.assertRaisesRegex(ValueError, 'no exact-node placement'):
            adapter.create(wrong)
        identifier = self.create(adapter, service, intent)
        self.assertFalse(adapter.capabilities['node_placement'])
        self.assertEqual(service.last_spec.replicas, 1)
        self.assertEqual(service.last_spec.nodes_per_replica, 1)
        self.assertEqual(shlex.split(service.last_spec.command), ['env', 'RESOURCE_MANAGER_CARDS=2',
            'RESOURCE_MANAGER_ALLOCATION_ID=operation-id', *intent['command']])
        self.assertEqual(adapter.observe(identifier)['state'], 'RUNNING')

    def test_native_running_is_not_readiness_and_quota_command_are_validated(self):
        adapter, service, _ = self.adapter('serving')
        intent = self.intent('serving'); identifier = self.create(adapter, service, intent)
        adapter.launch(identifier, intent['command'], 'launch')
        result = adapter.health(identifier)
        self.assertTrue(result['running']); self.assertFalse(result['ready'])
        self.assertEqual(service.exec.call_count, 0)
        self.assertTrue(adapter.health(identifier, ['curl', '-f', 'http://localhost/ready'])['ready'])
        service.exec.return_value = NS(completed=True, returncode=1)
        self.assertFalse(adapter.health(identifier, ['check-ready'])['ready'])
        raw = service.raws[identifier]
        raw['command'] = 'other command'
        with self.assertRaisesRegex(ValueError, 'submitted payload and owner'):
            adapter.observe(identifier)
        raw['command'] = shlex.join(['env', 'RESOURCE_MANAGER_CARDS=2',
            'RESOURCE_MANAGER_ALLOCATION_ID=operation-id', *intent['command']])
        raw['resource_spec_price']['gpu_count'] = 4
        with self.assertRaisesRegex(ValueError, 'quota changed'):
            adapter.observe(identifier)


if __name__ == '__main__':
    unittest.main()
