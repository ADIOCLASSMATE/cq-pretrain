"""Small, explicit Inspire SDK boundary for the current native Serving API.

No process-wide monkeypatches. Read pacing is shared by clients in this process;
Create remains a single-send SDK operation with its original uncertain outcome.
"""
import threading
import time
import re
from uuid import uuid4

from inspire.sdk import InspireClient as SDKClient
from inspire.sdk.servings import Servings
from inspire.sdk.resources import operation, Workspaces
from inspire.sdk.models_serving import ServingRef, ServingHandle
from inspire.sdk.exceptions import SubmissionUncertainError, ValidationError
from inspire.platform.web import browser_api as api
from inspire.services.serving import serving_submission
from inspire.platform.web.browser_api.core import _request_json, _v2_result


def workspace_entries(session):
    """The live platform removed userWorkspaceList from GetRoutes."""
    entries={};page=1
    while True:
        result=_v2_result(_request_json(session,'POST','/api/v2/workspace?Action=ListWorkspaces',
            referer=session.base_url.rstrip('/')+'/workspace',
            body={'PageNumber':page,'page_size':100,'filter':{}},timeout=30))
        rows=result.get('items');total=result.get('total')
        # ListWorkspaces returns total as a decimal string on this platform.
        if not isinstance(rows,list) or not str(total).isdigit():
            raise ValidationError('Incomplete live workspace catalog')
        total=int(total)
        for row in rows:
            key=row.get('id','');name=row.get('name')
            if (not re.fullmatch(r'ws-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}',key)
                    or not name or key in entries or type(row.get('is_fair_workspace')) is not bool):
                raise ValidationError('Invalid live workspace identity or scheduling policy')
            entries[key]={'id':key,'name':name,'is_fair_workspace':row['is_fair_workspace']}
        if len(entries)==total:break
        if not rows or len(entries)>total:raise ValidationError('Incomplete workspace pagination')
        page+=1
    session.all_workspace_ids=list(entries)
    session.all_workspace_names={key:row['name'] for key,row in entries.items()}
    session.all_workspace_fair_scheduling={key:row['is_fair_workspace'] for key,row in entries.items()}
    return list(entries.values())


class NativeWorkspaces(Workspaces):
    def _routes(self):
        rows=self._catalog('workspaces',(),lambda:workspace_entries(self.session))
        if self.client._catalog_context is not None:
            self.client._catalog_context.update({row['id']:row['is_fair_workspace'] for row in rows})
        return rows

_locks = {key: threading.Lock() for key in ('get', 'list')}
_last = {key: float('-inf') for key in _locks}


def pace(key):
    with _locks[key]:
        delay = _last[key] + {'get': .25, 'list': .35}[key] - time.monotonic()
        if delay > 0:
            time.sleep(delay)
        _last[key] = time.monotonic()


def create_arguments(value):
    # CreateServingConsole rejects this SDK field. Model-serving commands only
    # read the fixed weights; corpus dataset mounts belong to CPU/NPU workers.
    value = dict(value)
    value.pop('is_publicpath_readonly', None)
    return value


class NativeServings(Servings):
    def _detail(self, key):
        pace('get')
        return super()._detail(key)

    def _fetch(self, *args, **kwargs):
        pace('list')
        return super()._fetch(*args, **kwargs)

    @operation
    def create(self, spec, *, operation_id=None):
        identifier = uuid4().hex if operation_id is None else operation_id
        if not isinstance(identifier, str) or not identifier:
            raise ValidationError('operation_id must be a non-empty string')
        plan = self.plan(spec)
        with self.client._transport.single_send(identifier, create=True, inspect='servings'):
            result = api.create_serving(**create_arguments(plan.create_kwargs), session=self.session)
        key = serving_submission.created_serving_id(result)
        if not key:
            raise SubmissionUncertainError(identifier, inspect='servings')
        return ServingHandle(plan.name, self._make_ref(ServingRef, plan.name, key, plan.workspace.ref.key), identifier)


class InspireClient(SDKClient):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.workspaces = NativeWorkspaces(self)
        self.servings = NativeServings(self)
