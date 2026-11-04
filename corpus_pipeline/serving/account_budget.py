"""Native requested-card admission and management network configuration."""
import os
import time

WORKSPACE = "昇腾卡公共空间"


def configure_management_proxy_bypass(environ=None):
    """Keep existing bypasses and route the two verified platform hosts directly.

    Requests prefers lowercase no_proxy when both forms exist. Synchronize the
    union before clients or guardian children start; leave other proxy settings
    intact, including those needed for paid APIs and unrelated services.
    """
    environ = os.environ if environ is None else environ
    items = []
    seen = set()
    for raw in (environ.get('NO_PROXY', ''), environ.get('no_proxy', ''),
                'qz.sii.edu.cn,keycloak-inspire-prod.sii.edu.cn'):
        for value in raw.split(','):
            item = value.strip()
            if item and item.casefold() not in seen:
                items.append(item)
                seen.add(item.casefold())
    environ['NO_PROXY'] = environ['no_proxy'] = ','.join(items)

def observable_project_budget(client,requested,*,ceiling=256):
    """Count this account's visible active Ascend workloads across projects."""
    if type(ceiling) is not int or ceiling < 1 or type(requested) is not int or requested < 0:
        raise ValueError('Explicit positive card ceiling and nonnegative request required')
    from inspire.platform.web.browser_api.jobs import get_current_user
    with client._transport.scope(timeout=20):
        user_id=get_current_user(session=client._transport.session)['id']
    records={}
    terminal={'STOPPED','FAILED','DELETED','CANCELLED','SUCCEEDED','COMPLETED'}
    for service in client.servings.iter(workspace=WORKSPACE):
        if service.status in terminal:
            continue
        detail=client.servings.get(service.ref,workspace=WORKSPACE);raw=detail.raw
        assert raw.get('user_id'), 'Missing Serving ownership evidence'
        if raw['user_id']!=user_id:
            continue
        cards=int(raw['resource_spec_price']['gpu_count'])*int(raw['replicas'])*int(raw['node_num_per_replica'])
        allocated=None  # Admission counts requests, including unplaced ones.
        records[detail.ref.key]={'kind':'serving','name':detail.name,'cards':cards,'requested_cards':cards,'allocated_cards':allocated,'status':detail.status}
    for notebook in client.notebooks.iter(workspace=WORKSPACE):
        if notebook.status in terminal:
            continue
        cards=int(notebook.resource['gpu_count'])
        allocated=cards if notebook.status=='RUNNING' or notebook.node.get('name') else 0
        records[notebook.ref.key]={'kind':'notebook','name':notebook.name,'cards':cards,'requested_cards':cards,'allocated_cards':allocated,'status':notebook.status}
    # The current SDK filters job status locally; scan once and retain only
    # nonterminal records rather than re-fetching history for every active state.
    for job in client.jobs.iter(workspace=WORKSPACE):
        if job.status in terminal or job.ref.key in records:
            continue
        detail=client.jobs.get(job.ref,workspace=WORKSPACE)
        if detail.status in terminal:
            continue
        owner=detail.raw.get('user_id') or (detail.raw.get('created_by') or {}).get('id')
        assert owner, 'Missing Job ownership evidence'
        if owner!=user_id:
            continue
        configs=detail.raw['framework_config']
        if not configs:
            raise ValueError('Visible active Job has no resource contract')
        cards=sum(int(x['gpu_count'])*int(x['instance_count']) for x in configs)
        gpu_counts={int(x['gpu_count']) for x in configs}
        if len(gpu_counts)!=1:
            raise ValueError('Heterogeneous active Job needs explicit per-instance allocation accounting')
        allocated=None
        records[detail.ref.key]={'kind':'job','name':detail.name,'cards':cards,'requested_cards':cards,'allocated_cards':allocated,'status':detail.status}
    used=sum(x['cards'] for x in records.values())
    if used+requested>ceiling:
        raise ValueError('Observable account Ascend GPU ceiling would be exceeded')
    return {'time':time.time(),'records':records,'observable_requested_cards':used,
            'observable_allocated_cards':None,'allocation_telemetry':'not_queried_for_admission','requested_cards':requested,
            'observable_total_with_request':used+requested,'ceiling':ceiling,
            'scope':'current account visible active Ascend workloads across projects, queued requests included conservatively; unrelated resources are read only'}
