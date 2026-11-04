"""Conservative dynamic LOW Notebook allocation for the existing frontpass.

No daemon or CLI entry point: the controller explicitly calls Allocator. Every
Create is single-send after a durable exact intent. An unconfirmed Create keeps
its card/node reservation indefinitely and is resolved only by read-only exact
name, owner, project and payload reconciliation. Declared node availability is
only a placement candidate, never evidence of an actually running Notebook.
"""
from __future__ import annotations

import fcntl
import json
from pathlib import Path
import re
import tempfile
import time
import uuid

from corpus_pipeline.production.common import atomic_json, digest, file_sha
from .submit_jobs import ACCOUNT, WORKSPACE, PROJECT, GROUP, IMAGE_NAME, IMAGE_URL, save_new

EXCLUDED = {f'infra-gpu-npu-{n}.host.shzhisuan.com' for n in ('047', '248', '295')}
LEGAL_CARDS = {1, 2, 4, 8, 16}
CEILING = 256


def verified_retired(item, state):
    """Only an exact STOPPED observation can release an accepted reservation."""
    retirement = state.get('retirement') or {}
    return (retirement.get('phase') == 'verified_stopped'
            and retirement.get('notebook_id') == item['notebook_id']
            and retirement.get('node') == item['node']
            and isinstance(retirement.get('verified_stopped_at'), (int, float))
            and state.get('platform_status') == 'STOPPED')


def node_cooldowns(states, now):
    result = {}
    for state in states.values():
        retirement = state.get('retirement') or {}
        node = retirement.get('node')
        until = retirement.get('cooldown_until', 0)
        if node and isinstance(until, (int, float)) and until > now:
            result[node] = max(result.get(node, 0), until)
    return result


def quota_options(rows):
    """Use complete live quota triples, preferring fewer CPUs for equal GPUs."""
    options = []
    for row in rows:
        if row.get('compute_group') != GROUP:
            continue
        levels = row.get('allowed_priority_levels')
        # SDK exposes [] for legacy "any" quotas. Its plan() still performs
        # the authoritative LOW priority validation before any Create.
        if levels and 'LOW' not in {str(value).upper() for value in levels}:
            continue
        try:
            gpu, cpu, memory = [int(value) for value in row['quota'].split(',')]
        except (KeyError, ValueError, TypeError):
            continue
        if gpu in LEGAL_CARDS and cpu > 0 and memory > 0:
            options.append({'quota': row['quota'], 'gpu': gpu, 'cpu': cpu, 'memory': memory})
    return sorted(options, key=lambda row: (-row['gpu'], row['cpu'], row['memory'], row['quota']))


def fit(node, options, remaining):
    gpu_type = node.get('gpu_type') or node.get('gpu_info', {}).get('gpu_type')
    if (node.get('name') in EXCLUDED or node.get('status') != 'Ready'
            or node.get('cordon_type') not in ('', None)
            or gpu_type != 'ASCEND_910B_64G'):
        return None
    for option in options:
        if (option['gpu'] <= remaining
                and float(node.get('gpu', {}).get('available', -1)) >= option['gpu']
                and float(node.get('cpu', {}).get('available', -1)) >= option['cpu']
                and float(node.get('memory', {}).get('available', -1)) >= option['memory']):
            return option
    return None


def definite_rejection(error):
    """Only a typed SDK rejection with a known business-error envelope counts."""
    try:
        from inspire.platform.errors import ValidationError
    except ImportError:
        return None
    if isinstance(error, ValidationError):
        match = re.match(r'^API error: (InvalidParameter):', str(error))
        if match:
            return match.group(1)
    return None


def unresolved(directory):
    return not (directory / 'ACCEPTED.json').exists() and not (directory / 'REJECTED.json').exists()


def validate_max_cards(value, *, required=False):
    if value is None and not required:
        return None
    if type(value) is not int or not 1 <= value <= CEILING:
        raise ValueError('max_cards must be an integer between 1 and 256 before allocation')
    return value


def validate_allocation_tag(value):
    if value is not None and (not isinstance(value, str)
                              or re.fullmatch(r'[A-Za-z0-9_.:-]{1,128}', value) is None):
        raise ValueError('allocation_tag must be a bounded plan identifier')
    return value


def _unique_managed(managed):
    unique = {}
    identity_fields = ('devices', 'node', 'name', 'priority', 'workspace_id',
                       'project_id', 'group_id', 'create_kwargs_sha256', 'allocation_tag')
    for item in managed:
        identifier = item['notebook_id']
        if (not isinstance(identifier, str) or not identifier
                or type(item['devices']) is not int or not 1 <= item['devices'] <= CEILING
                or not isinstance(item['node'], str) or not item['node']):
            raise ValueError('invalid managed Notebook reservation')
        previous = unique.get(identifier, {})
        if any(key in previous and key in item and previous[key] != item[key]
               for key in identity_fields):
            raise ValueError('managed Notebook reservation identity changed')
        unique[identifier] = previous | item
    return unique


def _uncertain_cards(uncertain):
    unique = {}
    for index, intent in enumerate(uncertain):
        cards = intent['create_kwargs']['gpu_count']
        if type(cards) is not int or cards not in LEGAL_CARDS:
            raise ValueError('invalid unconfirmed Create reservation')
        key = intent.get('operation_id', ('unidentified', index))
        if key in unique and unique[key] != intent:
            raise ValueError('unconfirmed Create reservation identity changed')
        unique[key] = intent
    return sum(intent['create_kwargs']['gpu_count'] for intent in unique.values())


def managed_reserved_budget(managed, uncertain=(), states=None, max_cards=None, live=None):
    """Count accepted IDs until verified retirement, independent of live lists.

    STOPPED, FAILED, PENDING and absent observations keep their reservation.
    This pure helper does not discover receipts; Allocator.managed_budget adds
    the durable per-run ledger and every still-unconfirmed Create intent.
    """
    validate_max_cards(max_cards)
    unique, states = _unique_managed(managed), states or {}
    live_records = live['records'] if live is not None else {}
    accepted = sum(item['devices'] for key, item in unique.items()
                   if key in live_records or not verified_retired(item, states.get(key, {})))
    unknown = _uncertain_cards(uncertain)
    used = accepted + unknown
    return {'accepted_reserved_cards': accepted, 'uncertain_cards': unknown,
            'reserved_total_cards': used, 'max_cards': max_cards,
            'remaining_cards': None if max_cards is None else max(0, max_cards - used),
            'over_cap': max_cards is not None and used > max_cards,
            'checked_at': time.time()}


def reserved_budget(live, managed, uncertain, states=None, max_cards=None):
    records = live['records']
    unique = _unique_managed(managed)
    states = states or {}
    missing = sum(item['devices'] for key, item in unique.items()
                  if key not in records and not verified_retired(item, states.get(key, {})))
    unknown = _uncertain_cards(uncertain)
    used = int(live['observable_requested_cards']) + missing + unknown
    run_budget = managed_reserved_budget(unique.values(), uncertain, states, max_cards, live)
    remaining = max(0, CEILING - used)
    if max_cards is not None:
        remaining = min(remaining, run_budget['remaining_cards'])
    return {'live_cards': int(live['observable_requested_cards']),
            'managed_missing_from_live_cards': missing, 'uncertain_cards': unknown,
            'reserved_total_cards': used, 'ceiling': CEILING, 'checked_at': time.time(),
            'managed_budget': run_budget, 'remaining_cards': remaining}


class Live:
    """Small SDK boundary kept separate so safety tests never reach a platform."""
    def __init__(self, client):
        self.client = client

    def discover(self):
        from inspire.platform.web.browser_api.jobs import get_current_user
        # This is an in-memory catalog cache only; a new observation must not
        # inherit a stale image/quota/project resolution from the last cycle.
        self.client.cache.clear()
        image = self.client.images.get(IMAGE_NAME, workspace=WORKSPACE)
        if image.status != 'SUCCESS' or image.url != IMAGE_URL or not image.ref.key.startswith('image-'):
            raise ValueError('live qualified image identity is unavailable')
        rows = self.client.notebooks.quotas(WORKSPACE, group=GROUP, limit=100).items
        options = quota_options([row.to_dict() for row in rows])
        if not options:
            raise ValueError('no legal live LOW quota')
        # A plan is read-only and resolves current workspace/project/group IDs.
        payload = self.plan('corpus-fragment-discovery', options[-1], None, image)
        with self.client._transport.scope(timeout=30):
            owner = get_current_user(session=self.client._transport.session)['id']
        context = {'image': image, 'owner_id': owner, 'options': options,
                   'workspace_id': payload['workspace_id'], 'project_id': payload['project_id'],
                   'group_id': payload['logic_compute_group_id']}
        return context, self.nodes(context)

    def nodes(self, context):
        from inspire.platform.web.browser_api.availability.api import list_node_dimension
        with self.client._transport.scope(timeout=30):
            return list_node_dimension(context['group_id'], workspace_id=context['workspace_id'],
                                       session=self.client._transport.session)

    def plan(self, name, option, node, image):
        from inspire.sdk.models_notebooks import NotebookCreateSpec
        spec = NotebookCreateSpec(name=name, workspace=WORKSPACE, project=PROJECT,
            group=GROUP, quota=option['quota'], image=image.ref, priority=3, node=node,
            shm_gib=min(128, option['cpu'], option['memory']), auto_stop=False,
            enable_notification=False, public_path_readonly=True)
        plan = self.client.notebooks.plan(spec)
        if (plan.workspace.name != WORKSPACE or plan.project.name != PROJECT
                or plan.group.name != GROUP or plan.priority != 3):
            raise ValueError('live plan resolved outside the approved LOW scope')
        payload = dict(plan.create_kwargs)
        # Explicitly bind the actual registry ID. The URL must never be used as
        # mirror_id; both fields stay independently represented in the receipt.
        payload['image_id'], payload['image_url'] = image.ref.key, image.url
        if (payload['gpu_count'] != option['gpu'] or payload['cpu_count'] != option['cpu']
                or payload['memory_size'] != option['memory'] or payload['task_priority'] != 3
                or payload['is_publicpath_readonly'] is not True or payload.get('node_id') != node):
            raise ValueError('live create payload differs from selected quota or placement')
        return payload

    def budget(self):
        from corpus_pipeline.serving.inspire_serving.downsize import observable_project_budget
        return observable_project_budget(self.client, 0)

    def create(self, payload, operation_id):
        from inspire.platform.web.browser_api.notebooks import create_notebook
        from inspire.services.notebook.notebooks import extract_notebook_id
        with self.client._transport.scope(timeout=60):
            with self.client._transport.single_send(operation_id, create=True, inspect='notebooks'):
                result = create_notebook(**payload, session=self.client._transport.session)
        identifier = extract_notebook_id(result)
        if not identifier:
            raise RuntimeError('Create did not confirm an ID')
        return identifier

    def reconcile(self, intent):
        from inspire.platform.web.browser_api.notebooks import get_notebook_detail
        payload = intent['create_kwargs']
        candidates = [row for row in self.client.notebooks.iter(workspace=WORKSPACE,
                          keyword=payload['name']) if row.name == payload['name']]
        if len(candidates) != 1:
            return None
        candidate = candidates[0]
        with self.client._transport.scope(timeout=30):
            raw = get_notebook_detail(candidate.ref.key, session=self.client._transport.session)
        # GetNotebook's verified native shape stores identity in nested objects
        # and the submitted resources in start_config. Internal priority 13 is
        # normalized by the SDK candidate to priority 3 / LOW.
        start = raw.get('start_config') or {}
        actual = {'name': raw.get('name'),
            'project_id': (raw.get('project') or {}).get('id', raw.get('project_id')),
            'workspace_id': (raw.get('workspace') or {}).get('id', raw.get('workspace_id')),
            'image_id': start.get('mirror_id', raw.get('mirror_id')),
            'image_url': start.get('mirror_url', raw.get('mirror_url')),
            'is_publicpath_readonly': raw.get('is_publicpath_readonly')}
        for key in ('logic_compute_group_id', 'gpu_count', 'cpu_count', 'memory_size',
                    'node_id', 'shared_memory_size', 'quota_id', 'auto_stop'):
            actual[key] = start.get(key, raw.get(key))
        owner = (raw.get('creator') or {}).get('id', raw.get('user_id'))
        if (owner != intent['owner_id'] or str(candidate.priority) != str(payload['task_priority'])
                or candidate.priority_level != 'LOW'
                or any(actual[key] != payload[key] for key in actual)):
            return None
        return candidate.ref.key


class Allocator:
    def __init__(self, client, root, execution, max_cards: int | None = None):
        self.root = Path(root).resolve()
        self.execution = Path(execution).resolve()
        self.max_cards = validate_max_cards(max_cards)
        self.directory = self.execution / 'dynamic'
        self.directory.mkdir(parents=True, exist_ok=True)
        self.reservations_path = self.root / 'FRAGMENT_RESERVATIONS.json'
        self.lock_path = Path(tempfile.gettempdir()) / (
            'corpus-fragment-allocator-' + digest(str(self.root))[:24] + '.lock')
        self.live = Live(client)

    def pending_fit(self, item):
        """Fresh declared capacity for the exact accepted quota, not allocation proof."""
        context = {'workspace_id': item['workspace_id'], 'group_id': item['group_id']}
        nodes = self.live.nodes(context)
        fresh = next((row for row in nodes if row.get('name') == item['node']), None)
        option = {'quota': f"{item['devices']},{item['cpu']},{item['memory']}",
                  'gpu': item['devices'], 'cpu': item['cpu'], 'memory': item['memory']}
        return {'observed_at': time.time(), 'node': item['node'],
                'fits': bool(fresh and fit(fresh, [option], CEILING)),
                'requested_quota': option,
                'node_snapshot': {key: fresh.get(key) for key in (
                    'name', 'status', 'cordon_type', 'gpu', 'cpu', 'memory')} if fresh else None}

    def _intents(self):
        result = []
        paths = set(self.directory.glob('*/INTENT.json'))
        if self.reservations_path.exists():
            ledger = json.loads(self.reservations_path.read_text())
            body = {key: ledger[key] for key in ('schema', 'root', 'notebooks', 'intents')}
            if (body['root'] != str(self.root) or ledger.get('records_sha256') != digest(body)):
                raise ValueError('per-run reservation ledger identity or digest changed')
            # A new execution directory must still reconcile earlier sends for
            # this run. Missing old paths remain reserved in the ledger.
            paths.update(Path(entry['directory']) / 'INTENT.json'
                         for entry in body['intents'].values()
                         if (Path(entry['directory']) / 'INTENT.json').exists())
        for path in sorted(paths):
            item = json.loads(path.read_text())
            if (item.get('root') != str(self.root)
                    or item.get('create_kwargs_sha256') != digest(item['create_kwargs'])
                    or item['create_kwargs'].get('task_priority') != 3
                    or item['create_kwargs'].get('gpu_count') not in LEGAL_CARDS
                    or item['create_kwargs'].get('project_name') != PROJECT
                    or item['create_kwargs'].get('image_url') != IMAGE_URL
                    or item['create_kwargs'].get('is_publicpath_readonly') is not True
                    or item['create_kwargs'].get('node_id') in EXCLUDED):
                raise ValueError('dynamic create intent identity changed')
            rejected = path.parent / 'REJECTED.json'
            if rejected.exists():
                evidence = json.loads(rejected.read_text())
                if (evidence.get('operation_id') != item['operation_id']
                        or evidence.get('create_kwargs_sha256') != item['create_kwargs_sha256']
                        or evidence.get('business_error_code') != 'InvalidParameter'
                        or (path.parent / 'ACCEPTED.json').exists()):
                    raise ValueError('explicit rejection evidence is not bound to this intent')
            result.append((path.parent, item))
        return result

    def _disk_managed(self, intents):
        rows = []
        for directory, intent in intents:
            accepted = directory / 'ACCEPTED.json'
            if accepted.exists():
                row = json.loads(accepted.read_text())
                if (row['create_kwargs_sha256'] != intent['create_kwargs_sha256']
                        or row['name'] != intent['create_kwargs']['name']
                        or row['devices'] != intent['create_kwargs']['gpu_count']
                        or row['node'] != intent['create_kwargs']['node_id']
                        or row['priority'] != 3
                        or row.get('allocation_tag') != intent.get('allocation_tag')
                        or row['workspace_id'] != intent['create_kwargs']['workspace_id']
                        or row['project_id'] != intent['create_kwargs']['project_id']
                        or row['group_id'] != intent['create_kwargs']['logic_compute_group_id']):
                    raise ValueError('accepted dynamic identity differs from exact intent')
                # Earlier accepted receipts omit these two quota fields. They
                # are restored from the immutable, hash-bound create intent.
                if (row.get('cpu', intent['create_kwargs']['cpu_count']) != intent['create_kwargs']['cpu_count']
                        or row.get('memory', intent['create_kwargs']['memory_size']) != intent['create_kwargs']['memory_size']):
                    raise ValueError('accepted dynamic CPU/memory quota differs from intent')
                rows.append(row | {'cpu': intent['create_kwargs']['cpu_count'],
                                   'memory': intent['create_kwargs']['memory_size']})
        return list(_unique_managed(rows).values())

    def _reservation_view(self, managed=()):
        """Monotonic accounting survives a missing receipt or intent directory.

        One controller owns this run under its host-local lock. The ledger is
        per run (not execution directory); it preserves known accepted IDs and
        unresolved sends across controller restarts. A missing/corrupt ledger
        after initialization fails closed rather than starting an empty budget.
        Terminal intent resolutions never regress when a receipt disappears.
        """
        marker = self.root / 'FRAGMENT_RESERVATIONS_REQUIRED.json'
        if self.reservations_path.exists():
            ledger = json.loads(self.reservations_path.read_text())
            body = {key: ledger[key] for key in ('schema', 'root', 'notebooks', 'intents')}
            if (body['schema'] != 'corpus_fragment_reservations.v1'
                    or body['root'] != str(self.root)
                    or ledger.get('records_sha256') != digest(body)):
                raise ValueError('per-run reservation ledger identity or digest changed')
        else:
            if marker.exists():
                raise ValueError('required per-run reservation ledger is unavailable')
            body = {'schema': 'corpus_fragment_reservations.v1', 'root': str(self.root),
                    'notebooks': [], 'intents': {}}
        previous = digest(body)
        intents = self._intents()
        disk_managed = self._disk_managed(intents)
        owned = _unique_managed([*body['notebooks'], *managed, *disk_managed])
        accepted_operations = {item['operation_id']: item for item in disk_managed}
        entries = body['intents']
        for operation_id, entry in entries.items():
            intent = entry['intent']
            if (intent['operation_id'] != operation_id or intent['root'] != str(self.root)
                    or intent['create_kwargs_sha256'] != digest(intent['create_kwargs'])
                    or entry['status'] not in ('pending', 'accepted', 'rejected')):
                raise ValueError('persisted Create reservation identity changed')
            if entry['status'] == 'accepted' and entry['notebook_id'] not in owned:
                raise ValueError('accepted Create reservation lost its Notebook identity')
        for directory, intent in intents:
            operation_id = intent['operation_id']
            entry = entries.get(operation_id)
            if entry is None:
                entry = {'intent': intent, 'directory': str(directory.resolve()), 'status': 'pending'}
                entries[operation_id] = entry
            elif entry['intent'] != intent or entry['directory'] != str(directory.resolve()):
                raise ValueError('persisted Create intent identity changed')
            if operation_id in accepted_operations:
                identifier = accepted_operations[operation_id]['notebook_id']
                if (entry['status'] == 'rejected'
                        or entry.get('notebook_id', identifier) != identifier):
                    raise ValueError('Create resolution changed')
                entry.update(status='accepted', notebook_id=identifier)
            elif (directory / 'REJECTED.json').exists():
                if entry['status'] == 'accepted':
                    raise ValueError('accepted Create became rejected')
                entry['status'] = 'rejected'
        body['notebooks'] = list(owned.values())
        if not self.reservations_path.exists() or digest(body) != previous:
            atomic_json(self.reservations_path, body | {
                'records_sha256': digest(body), 'updated_at': time.time()})
        if not marker.exists():
            atomic_json(marker, {'root': str(self.root), 'ledger': str(self.reservations_path)})
        uncertain = [entry['intent'] for entry in entries.values() if entry['status'] == 'pending']
        return list(owned.values()), uncertain

    def load_managed(self):
        return self._reservation_view()[0]

    def managed_budget(self, managed, states=None, *, live=None):
        owned, uncertain = self._reservation_view(managed)
        return managed_reserved_budget(owned, uncertain, states, self.max_cards, live)

    def reserved_budget(self, live, managed, states=None):
        """Both limits, including ledger entries absent from live/receipt lists."""
        owned, uncertain = self._reservation_view(managed)
        return reserved_budget(live, owned, uncertain, states, self.max_cards)

    def reserved_unconfirmed_cards(self):
        return _uncertain_cards(self._reservation_view()[1])

    def allocation_tag_status(self, allocation_tag):
        """Local durable status; no platform call and no implicit Create retry."""
        if validate_allocation_tag(allocation_tag) is None:
            raise ValueError('allocation_tag is required')
        self._reservation_view()
        ledger = json.loads(self.reservations_path.read_text())
        matches = [entry for entry in ledger['intents'].values()
                   if entry['intent'].get('allocation_tag') == allocation_tag]
        if not matches:
            return {'status': 'not_sent', 'allocation_tag': allocation_tag}
        if len(matches) != 1:
            raise ValueError('allocation_tag is bound to more than one Create intent')
        entry = matches[0]
        payload = entry['intent']['create_kwargs']
        return {'status': entry['status'], 'allocation_tag': allocation_tag,
                'operation_id': entry['intent']['operation_id'],
                'target_node': payload['node_id'], 'target_cards': payload['gpu_count'],
                'target_quota': {'gpu': payload['gpu_count'], 'cpu': payload['cpu_count'],
                    'memory': payload['memory_size'],
                    'quota': ','.join(str(payload[key]) for key in ('gpu_count', 'cpu_count', 'memory_size'))},
                'notebook_id': entry.get('notebook_id')}

    def _accept(self, directory, intent, notebook_id, reconciled=False):
        payload = intent['create_kwargs']
        row = {'notebook_id': notebook_id, 'name': payload['name'], 'devices': payload['gpu_count'],
               'cpu': payload['cpu_count'], 'memory': payload['memory_size'],
               'node': payload['node_id'], 'priority': 3, 'workspace_id': payload['workspace_id'],
               'project_id': payload['project_id'], 'group_id': payload['logic_compute_group_id'],
               'accepted_at': time.time(), 'create_kwargs_sha256': intent['create_kwargs_sha256'],
               'operation_id': intent['operation_id'], 'reconciled_read_only': reconciled}
        if intent.get('allocation_tag') is not None:
            row['allocation_tag'] = intent['allocation_tag']
        save_new(directory / 'ACCEPTED.json', row)
        return row

    def discover_and_allocate(self, managed, states, *, allow_create=True, should_stop=None,
                              target_node=None, target_cards=None, allocation_tag=None, target_quota=None):
        # All allocator calls belong to the same stable CPU controller; enforce
        # that premise before relying on its host-local flock.
        from .text_worker import verify_control_host
        verify_control_host(self.root)
        # The reservation ledger is per run, including across execution
        # directories. Its send lock must have the same scope on this verified
        # control host; shared-storage locks do not establish cross-host safety.
        with self.lock_path.open('a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return self._discover_and_allocate(managed, states, allow_create=allow_create,
                                               should_stop=should_stop, target_node=target_node,
                                               target_cards=target_cards, allocation_tag=allocation_tag,
                                               target_quota=target_quota)

    def _discover_and_allocate(self, managed, states, *, allow_create=True, should_stop=None,
                               target_node=None, target_cards=None, allocation_tag=None, target_quota=None):
        should_stop = should_stop or (lambda: False)
        validate_allocation_tag(allocation_tag)
        if target_quota is not None:
            if (target_node is None or not isinstance(target_quota, dict)
                    or target_quota.get('gpu') != target_cards
                    or any(type(target_quota.get(key)) is not int or target_quota[key] <= 0
                           for key in ('gpu', 'cpu', 'memory'))
                    or target_quota.get('quota') != ','.join(str(target_quota[key]) for key in ('gpu', 'cpu', 'memory'))):
                raise ValueError('target_quota must be one exact legal catalog triple')
        if ((target_node is None) != (target_cards is None)
                or (allocation_tag is not None and target_node is None)
                or (target_node is not None and (
                    not isinstance(target_node, str) or not target_node or target_node in EXCLUDED
                    or type(target_cards) is not int or target_cards not in LEGAL_CARDS))):
            raise ValueError('target_node and one exact legal target_cards quota must be supplied together')
        if allow_create:
            validate_max_cards(self.max_cards, required=True)
            qualification = json.loads((self.root / 'QUALIFIED.json').read_text())
            if (qualification.get('success') is not True or qualification.get('dtype') != 'float32'
                    or qualification.get('inference_code_sha256') != file_sha(Path(__file__).with_name('inference.py'))):
                raise ValueError('production inference no longer qualified')
        added = []
        owned_rows, uncertain = self._reservation_view(managed)
        uncertain_ids = {intent['operation_id'] for intent in uncertain}
        # Reconcile uncertain submissions before considering any replacement.
        for directory, intent in self._intents():
            if intent['operation_id'] in uncertain_ids:
                identifier = self.live.reconcile(intent)
                if identifier:
                    added.append(self._accept(directory, intent, identifier, reconciled=True))
        owned_rows, uncertain = self._reservation_view(managed)
        run_budget = managed_reserved_budget(owned_rows, uncertain, states, self.max_cards)
        tag_status = self.allocation_tag_status(allocation_tag) if allocation_tag is not None else None
        tag_sent = tag_status is not None and tag_status['status'] != 'not_sent'
        if tag_sent:
            if tag_status['target_node'] != target_node or tag_status['target_cards'] != target_cards:
                raise ValueError('allocation_tag destination or exact quota changed')
            if target_quota is not None and tag_status['target_quota'] != target_quota:
                raise ValueError('allocation_tag CPU or memory quota changed')
            if tag_status['status'] == 'accepted':
                added = list(_unique_managed([*added, *[
                    item for item in owned_rows if item['notebook_id'] == tag_status['notebook_id']]]).values())
        if not allow_create or should_stop() or run_budget['remaining_cards'] == 0 or tag_sent:
            atomic_json(self.directory / 'REGISTRY.json', {'notebooks': self.load_managed(),
                'updated_at': time.time(), 'unconfirmed_cards': self.reserved_unconfirmed_cards(),
                'managed_budget': run_budget})
            return added
        from corpus_pipeline.resource_manager.planning import plan_allocations
        from .resource_planning import make_spec, reservations, capacities, quotas
        owned = _unique_managed(owned_rows)
        now = time.time()
        cooldowns = node_cooldowns(states, now)
        blocked = set(cooldowns) | EXCLUDED
        # A retry without a binding tag cannot identify an earlier unknown
        # targeted Create. Keep that legacy interface fail-closed.
        if target_node and any(intent['create_kwargs']['node_id'] == target_node for intent in uncertain):
            return added
        context, nodes = self.live.discover()
        options = context['options']
        if target_node is not None:
            options = [option for option in options if option['gpu'] == target_cards]
        if target_quota is not None:
            options = [option for option in options if option == target_quota]
        decisions = []
        owned_by_node = {}
        # Every iteration uses the same generic three-resource planner. A
        # pending four-card request reserves four cards, not its whole node.
        for _ in range(self.max_cards):
            if should_stop():
                break
            if not any(row['name'] not in blocked
                       and (target_node is None or row['name'] == target_node)
                       and fit(row, options, CEILING) for row in nodes):
                break
            live_budget = self.live.budget()
            budget = self.reserved_budget(live_budget, list(owned.values()), states)
            if budget['remaining_cards'] == 0:
                break
            owned_rows, uncertain = self._reservation_view(owned.values())
            owned = _unique_managed(owned_rows)
            effective_states = {key: dict(value) for key, value in states.items()}
            for identifier in owned:
                active = live_budget['records'].get(identifier)
                if active and active.get('status'):
                    state = effective_states.setdefault(identifier, {})
                    state.update(platform_status=active.get('status', 'UNKNOWN'), observed_at=time.time())
                elif active and verified_retired(owned[identifier], effective_states.get(identifier, {})):
                    effective_states.setdefault(identifier, {})['platform_status'] = 'UNKNOWN'
            reserved = reservations(owned.values(), effective_states, uncertain, root=self.root)
            owned_by_node = {}
            for item in reserved:
                if not item.is_retired:
                    owned_by_node[item.node] = owned_by_node.get(item.node, 0) + item.cards
            nodes = self.live.nodes(context)
            candidates = [row for row in nodes if target_node is None or row['name'] == target_node]
            proposal = plan_allocations(make_spec(self.root, self.max_cards),
                capacities(candidates, blocked), quotas(options), reserved,
                global_remaining_cards=budget['remaining_cards'])
            if not proposal.placements:
                break
            placement = proposal.placements[0]
            name = placement.node
            option = next(row for row in options if (row['gpu'], row['cpu'], row['memory']) ==
                          (placement.quota.cards, placement.quota.cpu, placement.quota.memory))
            fresh = next(row for row in nodes if row['name'] == name)
            operation_id = uuid.uuid4().hex
            short_node = name.split('.')[0].removeprefix('infra-gpu-npu-')
            notebook_name = 'corpus-fp-dyn-' + short_node + '-' + operation_id[:10]
            payload = self.live.plan(notebook_name, option, name, context['image'])
            if (payload['workspace_id'] != context['workspace_id'] or payload['project_id'] != context['project_id']
                    or payload['logic_compute_group_id'] != context['group_id']):
                raise ValueError('live project/workspace/group changed before Create')
            # Budget gets one final fresh read after SDK planning latency.
            budget = self.reserved_budget(self.live.budget(), list(owned.values()), states)
            if option['gpu'] > budget['remaining_cards']:
                decisions.append({'node': name, 'reason': 'fresh_budget_exhausted'})
                break
            if should_stop():
                decisions.append({'node': name, 'reason': 'controller_stop_requested'})
                break
            directory = self.directory / operation_id
            directory.mkdir()
            intent = {'schema': 'corpus_fragment_create.v1', 'operation_id': operation_id,
                'root': str(self.root), 'created_at': time.time(), 'owner_id': context['owner_id'],
                'create_kwargs': payload, 'create_kwargs_sha256': digest(payload),
                'node_snapshot': {key: fresh.get(key) for key in ('name', 'status', 'cordon_type', 'gpu', 'cpu', 'memory')},
                'budget': budget, 'retry_allowed': False}
            if allocation_tag is not None:
                intent['allocation_tag'] = allocation_tag
            save_new(directory / 'INTENT.json', intent)
            # The per-run ledger must also be durable before a single send.
            self._reservation_view(owned.values())
            uncertain.append(intent)
            try:
                identifier = self.live.create(payload, operation_id)
            except Exception as error:
                rejected_code = definite_rejection(error)
                save_new(directory / ('REJECTED.json' if rejected_code else 'UNCONFIRMED.json'),
                    {'at': time.time(), 'error_type': type(error).__name__,
                     'business_error_code': rejected_code, 'operation_id': operation_id,
                     'create_kwargs_sha256': intent['create_kwargs_sha256'], 'retry_allowed': False})
                atomic_json(self.directory / 'REGISTRY.json', {'notebooks': self.load_managed(),
                    'updated_at': time.time(), 'unconfirmed_cards': self.reserved_unconfirmed_cards()})
                # Stop this discovery cycle, including clear validation errors;
                # the controller applies backoff. Never spray more creates
                # after any failed/uncertain single-send operation.
                raise RuntimeError(('dynamic Create explicitly rejected; reservation released; '
                                    if rejected_code else 'dynamic Create unconfirmed; reservation retained; ')
                                   + type(error).__name__) from None
            row = self._accept(directory, intent, identifier)
            added.append(row)
            owned[identifier] = row
            uncertain.remove(intent)
            if target_node is not None:
                break  # One exact, tagged placement per consolidation call.
        atomic_json(self.directory / 'REGISTRY.json', {'notebooks': self.load_managed(),
            'updated_at': time.time(), 'uncertain_intents': len(uncertain),
            'managed_budget': self.managed_budget(owned.values(), states)})
        atomic_json(self.directory / 'LAST_DISCOVERY.json', {'observed_at': time.time(),
            'nodes_observed': len(nodes), 'blocked_nodes': sorted(blocked), 'decisions': decisions,
            'node_cooldowns': cooldowns,
            'placement_policy': 'resource_manager_three_resource_packing',
            'owned_cards_by_node': owned_by_node,
            'target_node': target_node, 'target_cards': target_cards,
            'newly_accepted': [item['notebook_id'] for item in added],
            'live_legal_quotas': context['options'], 'running_not_inferred_from_available': True})
        return added
