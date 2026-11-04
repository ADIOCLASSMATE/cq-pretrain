"""Inspire backend for the generic manager; no dataset or account defaults.

Notebook supports node placement and argv workers. Native Serving owns its
model process and placement: each allocation is one fixed-size, one-node
Serving replica. It does not masquerade as a Notebook or a training Job.
"""
from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import logging
import os
from pathlib import Path
import shlex
import tempfile
import time


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                     allow_nan=False).encode()).hexdigest()


def _dataset_info(value):
    """Compare resolved mount identities without depending on response order."""
    if not isinstance(value, list) or any(not isinstance(row, dict) or any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in ('dataset_id', 'version_id', 'path')) for row in value):
        raise ValueError('Notebook dataset mounts require resolved names, versions and storage paths')
    mounts = sorted((row['dataset_id'], row['version_id'], row['path']) for row in value)
    if len({row[:2] for row in mounts}) != len(mounts):
        raise ValueError('Notebook dataset mounts contain duplicate versions')
    return mounts


def _save(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=path.parent)
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, ensure_ascii=False, sort_keys=True, allow_nan=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


class InspireAdapter:
    def __init__(self, config):
        self.config = deepcopy(config)
        for key in ('account', 'workspace', 'project', 'group', 'image', 'registry_dir'):
            if not isinstance(config.get(key), str) or not config[key].strip():
                raise ValueError('Explicit adapter field required: ' + key)
        self.backend = config.get('backend', 'notebook')
        if self.backend not in ('notebook', 'serving'):
            raise ValueError('backend must be notebook or serving')
        datasets = config.get('datasets', [])
        if not isinstance(datasets, list) or any(not isinstance(value, str) or not value.strip()
                                                for value in datasets):
            raise ValueError('datasets must be a list of dataset:version strings')
        if datasets and self.backend != 'notebook':
            raise ValueError('Official dataset mounts require the Notebook backend')
        if config.get('priority') not in (1, 3, 4, 6):
            raise ValueError('Explicit supported LOW(1/3) or HIGH(4/6) priority is required')
        self.notebook_placement = config.get('notebook_placement', 'exact_node')
        if self.notebook_placement not in ('exact_node', 'native_scheduler'):
            raise ValueError('Unknown Notebook placement mode')
        if self.notebook_placement == 'native_scheduler' and self.backend != 'notebook':
            raise ValueError('Native Notebook scheduler requires the Notebook backend')
        if type(config.get('public_path_readonly', True)) is not bool:
            raise ValueError('public_path_readonly must be boolean')
        if type(config.get('omit_unsupported_public_path_flag', False)) is not bool:
            raise ValueError('omit_unsupported_public_path_flag must be boolean')
        self.ceiling = config.get('global_max_cards', 256)
        if type(self.ceiling) is not int or self.ceiling < 1:
            raise ValueError('global_max_cards must be positive')
        self.registry = Path(config['registry_dir']).resolve()
        self.capabilities = {'node_placement': self.backend == 'notebook' and self.notebook_placement == 'exact_node',
                             'drain': self.backend == 'notebook',
                             'retry_owned_stop': True,
                             'native_command': self.backend == 'serving'}
        if self.backend == 'notebook':
            for key in ('repository', 'remote_python', 'worker_state_dir'):
                if not config.get(key) or not Path(config[key]).is_absolute():
                    raise ValueError('Notebook requires an absolute ' + key)
        elif (not config.get('model') or type(config.get('model_version')) is not int
              or type(config.get('port')) is not int or not 1 <= config['port'] <= 65535):
            raise ValueError('Native Serving requires model, model_version and port')
        self._client = None

    @property
    def client(self):
        if self._client is None:
            # Only this manager process is affected; no persistent account or
            # network configuration changes are made.
            for key in ('HTTP_PROXY', 'HTTPS_PROXY', 'ALL_PROXY', 'http_proxy', 'https_proxy', 'all_proxy'):
                os.environ.pop(key, None)
            logging.getLogger('inspire').setLevel(logging.CRITICAL)
            from corpus_pipeline.serving.platform_client import InspireClient
            self._client = InspireClient(account=self.config['account'])
        return self._client

    @property
    def service(self):
        return self.client.notebooks if self.backend == 'notebook' else self.client.servings

    def _owner(self):
        from inspire.platform.web.browser_api.jobs import get_current_user
        with self.client._transport.scope(timeout=30):
            return get_current_user(session=self.client._transport.session)['id']

    def _catalog(self):
        self.client.cache.clear()
        image = self.client.images.get(self.config['image'], workspace=self.config['workspace'])
        if image.status != 'SUCCESS' or not image.ref.key:
            raise ValueError('Configured image is not ready')
        rows = self.service.quotas(self.config['workspace'], group=self.config['group'], limit=100).items
        quotas = []
        priority = 'LOW' if self.config['priority'] in (1, 3) else 'HIGH'
        for row in rows:
            value = row.to_dict()
            if value.get('compute_group') != self.config['group']:
                continue
            levels = value.get('allowed_priority_levels')
            if levels and priority not in {str(level).upper() for level in levels}:
                continue
            cards, cpu, memory = map(int, value['quota'].split(','))
            if cards > 0 and cpu > 0 and memory > 0:
                quotas.append({'cards': cards, 'cpu': cpu, 'memory': memory})
        if not quotas:
            raise ValueError('No current legal quota for this profile')
        return image, quotas

    def _global_budget(self):
        """This account's observable active requests in the explicit workspace."""
        client, workspace, owner = self.client, self.config['workspace'], self._owner()
        terminal = {'STOPPED', 'FAILED', 'DELETED', 'CANCELLED', 'SUCCEEDED', 'COMPLETED'}
        records = {}
        # SDK Notebook listing is scoped to the current account.
        for row in client.notebooks.iter(workspace=workspace):
            if row.status not in terminal:
                records[row.ref.key] = {'cards': int(row.resource['gpu_count']),
                    'state': row.status, 'node': (row.node or {}).get('name'),
                    'capacity_accounted': row.status == 'RUNNING'}
        for row in client.servings.iter(workspace=workspace):
            if row.status in terminal:
                continue
            detail = client.servings.get(row.ref, workspace=workspace)
            raw = detail.raw
            if not raw.get('user_id'):
                raise ValueError('Serving ownership is unknown')
            if raw['user_id'] == owner:
                cards = int(raw['resource_spec_price']['gpu_count']) * int(raw['replicas']) * int(raw['node_num_per_replica'])
                records[row.ref.key] = {'cards': cards, 'state': detail.status,
                    'node': None, 'capacity_accounted': detail.status == 'RUNNING'}
        for row in client.jobs.iter(workspace=workspace):
            if row.status in terminal:
                continue
            detail = client.jobs.get(row.ref, workspace=workspace)
            if detail.status in terminal:
                continue
            raw = detail.raw
            creator = raw.get('user_id') or (raw.get('created_by') or {}).get('id')
            if not creator:
                raise ValueError('Job ownership is unknown')
            if creator == owner:
                cards = sum(int(item['gpu_count']) * int(item['instance_count']) for item in raw['framework_config'])
                records[row.ref.key] = {'cards': cards, 'state': detail.status,
                    'node': None, 'capacity_accounted': detail.status == 'RUNNING'}
        used = sum(row['cards'] for row in records.values())
        return {'resources': records, 'global_remaining_cards': max(0, self.ceiling - used),
                'observable_requested_cards': used}

    def snapshot(self):
        image, quotas = self._catalog()
        budget = self._global_budget()
        if self.backend == 'serving' or self.notebook_placement == 'native_scheduler':
            # No physical-node placement claim: this is native scheduler capacity.
            maximum = max([self.ceiling, *[row['cards'] for row in quotas]])
            nodes = [{'name': 'native-scheduler', 'gpu': maximum,
                      'cpu': max(row['cpu'] / row['cards'] for row in quotas) * maximum,
                      'memory': max(row['memory'] / row['cards'] for row in quotas) * maximum}]
        else:
            from inspire.platform.web.browser_api.availability.api import list_node_dimension
            probe = {'name': 'resource-manager-discovery', 'placement': {'node': None, 'quota': quotas[-1]},
                     'command': ['true']}
            plan = self.service.plan(self._spec(probe, image))
            payload = plan.create_kwargs
            with self.client._transport.scope(timeout=30):
                rows = list_node_dimension(payload['logic_compute_group_id'],
                    workspace_id=payload['workspace_id'], session=self.client._transport.session)
            excluded = set(self.config.get('excluded_nodes', []))
            nodes = []
            accounted_on_node = {}
            for row in rows:
                # Notebook RUNNING can lead node free-capacity accounting by
                # minutes. Only release the local reservation when the same
                # capacity response also lists this allocation on this node.
                # Otherwise a create burst reuses stale free cards repeatedly.
                tasks = (row.get('tasks_associated') or {}).get('tasks') or []
                accounted_on_node[row['name']] = {task['id'] for task in tasks
                                                  if isinstance(task, dict) and task.get('id')}
                gpu_type = row.get('gpu_type') or (row.get('gpu_info') or {}).get('gpu_type')
                eligible = (row.get('status') == 'Ready' and row.get('cordon_type') in ('', None)
                            and row['name'] not in excluded
                            and gpu_type == self.config.get('gpu_type', 'ASCEND_910B_64G'))
                nodes.append({'name': row['name'], 'gpu': max(0, int(float(row.get('gpu', {}).get('available', 0)))),
                    'cpu': max(0, float(row.get('cpu', {}).get('available', 0))),
                    'memory': max(0, float(row.get('memory', {}).get('available', 0))), 'eligible': eligible})
            for identifier, resource in budget['resources'].items():
                resource['capacity_accounted'] = (resource['state'] == 'RUNNING'
                    and identifier in accounted_on_node.get(resource.get('node'), set()))
        result = {**budget, 'nodes': nodes, 'quotas': quotas, 'observed_at': time.time(),
                  'capabilities': self.capabilities}
        if self.notebook_placement == 'native_scheduler':
            result['capacity_scope'] = 'request_budget_not_physical_availability'
        return result

    def _spec(self, intent, image):
        cfg, quota = self.config, intent['placement']['quota']
        shared = {'name': intent['name'], 'workspace': cfg['workspace'], 'project': cfg['project'],
                  'group': cfg['group'], 'quota': ','.join(str(int(quota[key])) for key in ('cards', 'cpu', 'memory')),
                  'image': image.ref, 'priority': cfg['priority'],
                  'public_path_readonly': cfg.get('public_path_readonly', True),
                  'shm_gib': min(cfg.get('shm_gib', 128), int(quota['cpu']), int(quota['memory']))}
        if any(float(quota[key]) != int(quota[key]) for key in ('cards', 'cpu', 'memory')):
            raise ValueError('Inspire quota units must be integer catalog entries')
        if self.backend == 'notebook':
            from inspire.sdk.models_notebooks import NotebookCreateSpec
            node = intent['placement']['node']
            if self.notebook_placement == 'native_scheduler':
                if node not in (None, 'native-scheduler'):
                    raise ValueError('Native Notebook scheduler cannot bind a physical node')
                node = None
            if node in self.config.get('excluded_nodes', []):
                raise ValueError('Requested node is excluded')
            return NotebookCreateSpec(**shared, node=node, auto_stop=False, enable_notification=False,
                                      datasets=list(cfg.get('datasets', [])))
        from inspire.sdk.models_serving import ServingCreateSpec
        if intent['placement']['node'] != 'native-scheduler':
            raise ValueError('Native Serving has no exact-node placement API')
        command = self._workload_command(intent)
        return ServingCreateSpec(**shared, model=cfg['model'], model_version=cfg['model_version'],
            command=shlex.join(command), port=cfg['port'], replicas=1, nodes_per_replica=1,
            auto_scaling=False, description='Managed fixed-size replica; resource-manager allocation ' + intent.get('operation_id', ''))

    def _operation_path(self, operation_id):
        return self.registry / 'operations' / (_digest(operation_id) + '.json')

    def _workload_command(self, intent):
        return ['env', 'RESOURCE_MANAGER_CARDS=' + str(intent['placement']['quota']['cards']),
                'RESOURCE_MANAGER_ALLOCATION_ID=' + intent.get('operation_id', 'discovery'),
                *intent['command']]

    def _binding_path(self, resource_id):
        return self.registry / 'resources' / (_digest(resource_id) + '.json')

    def _bind(self, identifier, operation):
        value = {**operation, 'resource_id': identifier}
        path = self._binding_path(identifier)
        if path.exists() and json.loads(path.read_text()).get('intent') != operation['intent']:
            raise ValueError('Resource already belongs to a different intent')
        _save(path, value)

    def create(self, intent):
        path = self._operation_path(intent['operation_id'])
        if path.exists():
            raise ValueError('Create already has a durable intent; use reconcile')
        if self.backend == 'serving' and (intent['spec'].get('replica_cards') != intent['placement']['quota']['cards']
                                         or intent['spec'].get('workload_kind') != 'service'):
            raise ValueError('Native Serving requires a fixed replica_cards service contract')
        image, quotas = self._catalog()
        if intent['placement']['quota'] not in quotas:
            raise ValueError('Selected quota is no longer in the live catalog')
        plan = self.service.plan(self._spec(intent, image))
        if (plan.workspace.name != self.config['workspace'] or plan.project.name != self.config['project']
                or plan.group.name != self.config['group'] or plan.priority != self.config['priority']):
            raise ValueError('Live plan escaped the explicit profile')
        payload = dict(plan.create_kwargs)
        if self.backend == 'notebook':
            payload.update(image_id=image.ref.key, image_url=image.url)
            # The live quota already binds the card type. SDK 7.1.13 includes
            # a planning-only key rejected by its browser create function.
            payload.pop('gpu_type', None)
            if self.config.get('omit_unsupported_public_path_flag', False):
                # Some live Notebook API revisions reject this optional SDK
                # field. Omission uses platform permissions; the worker's
                # shared-output preflight still verifies actual write access.
                payload.pop('is_publicpath_readonly', None)
            if self.config.get('datasets'):
                from inspire.services.catalog.datasets import parse_dataset_specs
                requested = sorted((row.dataset, row.version) for row in
                                   parse_dataset_specs(self.config['datasets'], field='datasets'))
                if [row[:2] for row in _dataset_info(payload.get('dataset_info'))] != requested:
                    raise ValueError('Live plan differs from the requested dataset mounts')
        budget = self._global_budget()
        if budget['global_remaining_cards'] < intent['placement']['quota']['cards']:
            raise ValueError('Fresh account ceiling leaves insufficient capacity')
        operation = {'intent': intent, 'payload': payload, 'payload_sha256': _digest(payload),
                     'config_sha256': _digest(self.config), 'owner_id': self._owner(), 'backend': self.backend}
        _save(path, operation)
        with self.client._transport.scope(timeout=60):
            with self.client._transport.single_send(intent['operation_id'], create=True,
                                                    inspect='notebooks' if self.backend == 'notebook' else 'servings'):
                if self.backend == 'notebook':
                    from inspire.platform.web.browser_api.notebooks import create_notebook
                    from inspire.services.notebook.notebooks import extract_notebook_id
                    identifier = extract_notebook_id(create_notebook(**payload, session=self.client._transport.session))
                else:
                    from inspire.platform.web.browser_api.servings import create_serving
                    from inspire.sdk.servings import core
                    identifier = core.created_serving_id(create_serving(**payload, session=self.client._transport.session))
        if not identifier:
            raise RuntimeError('Create result is unknown; reconcile exact identity')
        self._bind(identifier, operation)
        return identifier

    def _ref(self, operation, identifier):
        if self.backend == 'notebook':
            from inspire.sdk.models_notebooks import NotebookRef as Ref
        else:
            from inspire.sdk.models_serving import ServingRef as Ref
        return Ref(name=operation['intent']['name'], key=identifier, account=self.config['account'],
                   workspace_id=operation['payload']['workspace_id'], base_url=self.client.base_url)

    def _validate(self, operation, detail):
        payload = operation['payload']
        if (_digest(payload) != operation['payload_sha256'] or operation['config_sha256'] != _digest(self.config)
                or detail.name != payload['name'] or detail.project != self.config['project']
                or detail.workspace != self.config['workspace'] or str(detail.priority) != str(self.config['priority'])):
            raise ValueError('Live resource identity or profile differs from its receipt')
        if self.backend == 'notebook':
            from inspire.platform.web.browser_api.notebooks import get_notebook_detail
            actual_node = (detail.node or {}).get('name')
            if self.notebook_placement == 'native_scheduler' and payload['node_id'] is not None:
                raise ValueError('Native Notebook payload unexpectedly binds a node')
            if self.notebook_placement == 'exact_node' and actual_node and (
                    actual_node != payload['node_id'] or actual_node in self.config.get('excluded_nodes', [])):
                raise ValueError('Allocated Notebook escaped its exact node constraint')
            raw = get_notebook_detail(detail.ref.key, session=self.client._transport.session)
            if payload.get('dataset_info') and (
                    _dataset_info(raw.get('dataset_info')) != _dataset_info(payload['dataset_info'])):
                raise ValueError('Live Notebook dataset mounts differ from the submitted payload')
            start = raw.get('start_config') or {}
            owner = (raw.get('creator') or {}).get('id', raw.get('user_id'))
            actual = {'workspace_id': (raw.get('workspace') or {}).get('id', raw.get('workspace_id')),
                'project_id': (raw.get('project') or {}).get('id', raw.get('project_id')),
                'image_id': start.get('mirror_id', raw.get('mirror_id')),
                'image_url': start.get('mirror_url', raw.get('mirror_url')),
                'is_publicpath_readonly': raw.get('is_publicpath_readonly')}
            if self.config.get('omit_unsupported_public_path_flag', False):
                if 'is_publicpath_readonly' in payload:
                    raise ValueError('Unsupported public path flag remained in the submitted payload')
                actual.pop('is_publicpath_readonly')
            for key in ('logic_compute_group_id', 'gpu_count', 'cpu_count', 'memory_size', 'node_id',
                        'shared_memory_size', 'quota_id', 'auto_stop'):
                actual[key] = start.get(key, raw.get(key))
            if self.notebook_placement == 'native_scheduler' and actual['node_id'] in (None, ''):
                # The live API persists an unbound None submission as ''.
                actual['node_id'] = payload['node_id']
        else:
            raw, actual = detail.raw, {}
            owner = raw.get('user_id')
            for key in ('workspace_id', 'project_id', 'logic_compute_group_id', 'model_id', 'model_version',
                        'mirror_id', 'command', 'port', 'replicas', 'node_num_per_replica'):
                actual[key] = raw.get(key)
            wanted = payload['resource_spec_price']
            current = raw.get('resource_spec_price') or {}
            if any(current.get(key) != wanted.get(key) for key in ('gpu_count', 'cpu_count', 'memory_size_gib', 'quota_id', 'logic_compute_group_id')):
                raise ValueError('Serving resource quota changed')
        if owner != operation['owner_id'] or any(value != payload[key] for key, value in actual.items()):
            raise ValueError('Live resource no longer matches the submitted payload and owner')

    def _bound(self, identifier):
        operation = json.loads(self._binding_path(identifier).read_text())
        if operation['resource_id'] != identifier:
            raise ValueError('Resource receipt ID changed')
        detail = self.service.get(self._ref(operation, identifier), workspace=self.config['workspace'])
        if detail.ref.key != identifier:
            raise ValueError('Resource lookup returned a different ID')
        self._validate(operation, detail)
        return operation, detail

    def reconcile(self, intent):
        path = self._operation_path(intent['operation_id'])
        if not path.exists():
            return None
        operation = json.loads(path.read_text())
        if operation['intent'] != intent:
            raise ValueError('Create intent changed before reconciliation')
        candidates = [row for row in self.service.iter(workspace=self.config['workspace'], keyword=intent['name'])
                      if row.name == intent['name']]
        if len(candidates) != 1:
            return None
        detail = self.service.get(candidates[0].ref, workspace=self.config['workspace'])
        if detail.ref.key != candidates[0].ref.key:
            raise ValueError('Reconciliation returned a different exact resource ID')
        self._validate(operation, detail)
        self._bind(detail.ref.key, operation)
        return detail.ref.key

    def observe(self, identifier):
        operation, detail = self._bound(identifier)
        state = detail.status
        if state in ('CREATING', 'QUEUING', 'RESTARTING', 'RETRYING', 'INITIALIZING'):
            state = 'PENDING'
        if state not in ('PENDING', 'RUNNING', 'STOPPED', 'FAILED'):
            state = 'UNKNOWN'
        return {'state': state, 'node': operation['intent']['placement']['node'],
                'capacity_accounted': detail.status == 'RUNNING'}

    def stop(self, identifier, operation_id):
        operation, detail = self._bound(identifier)
        with self.client._transport.scope(timeout=30):
            # Public SDK stop owns its single-send context. Unlike raw create
            # below, wrapping it again rejects before sending any request.
            self.service.stop(detail.ref, workspace=self.config['workspace'])

    def _runner(self, identifier, arguments):
        _, detail = self._bound(identifier)
        if detail.status != 'RUNNING':
            raise RuntimeError('Worker command requires a RUNNING resource')
        if (self.notebook_placement == 'native_scheduler' and
                (detail.node or {}).get('name') in self.config.get('excluded_nodes', [])):
            # Keep ownership validation and stop available, but never execute
            # workers on an excluded placement selected by the native scheduler.
            raise RuntimeError('Native scheduler selected an excluded worker node')
        from corpus_pipeline.cluster.notebook_exec import execute
        argv = [self.config['remote_python'], '-m', 'corpus_pipeline.resource_manager.command_runner',
                '--root', self.config['worker_state_dir'], '--resource-key', identifier, *arguments]
        command = 'cd ' + shlex.quote(self.config['repository']) + ' && ' + shlex.join(argv)
        result = execute(command, account=self.config['account'], workspace=self.config['workspace'],
            notebook=detail.name, notebook_id=identifier, workspace_id=detail.ref.workspace_id,
            timeout=40, max_output_bytes=8192)
        if result.get('completed') is not True or result.get('returncode') != 0:
            raise RuntimeError('Remote worker operation was not confirmed')
        candidates = []
        for line in result.get('output', '').splitlines():
            try:
                value = json.loads(line)
            except ValueError:
                continue
            if isinstance(value, dict) and 'running' in value:
                candidates.append(value)
        if len(candidates) != 1:
            raise ValueError('Ambiguous remote worker receipt')
        return candidates[0]

    def launch(self, identifier, command, operation_id):
        operation, _ = self._bound(identifier)
        if list(command) != operation['intent']['command']:
            raise ValueError('Worker command differs from the Create intent')
        if self.backend == 'serving':
            operation['launch_operation_id'] = operation_id
            _save(self._binding_path(identifier), operation)
            return
        self._runner(identifier, ['launch', '--operation-id', operation_id, '--',
                                  *self._workload_command(operation['intent'])])

    def _native_probe(self, identifier, command):
        _, detail = self._bound(identifier)
        result = self.service.exec(detail.ref, command=shlex.join(command),
            workspace=self.config['workspace'], timeout=30, max_output_bytes=1024)
        return bool(result.completed and result.returncode == 0)

    def health(self, identifier, readiness_command=None):
        operation, detail = self._bound(identifier)
        if self.backend == 'serving':
            running = detail.status == 'RUNNING'
            ready = running and bool(readiness_command) and self._native_probe(identifier, readiness_command)
            return {'running': running, 'ready': ready, 'completed': False, 'exit_code': None,
                    'launch_operation_id': operation.get('launch_operation_id')}
        args = ['status']
        if readiness_command:
            args += ['--readiness', json.dumps(list(readiness_command))]
        return self._runner(identifier, args)

    def drain(self, identifier, command, operation_id):
        if not self.capabilities['drain']:
            raise ValueError('This backend has no configured request-draining contract')
        self._runner(identifier, ['drain', '--operation-id', operation_id, '--', *command])

    def completion(self, identifier, command):
        if self.backend == 'serving':
            return self._native_probe(identifier, command)
        return self._runner(identifier, ['probe', '--', *command]).get('probe_success') is True
