"""External SDK lifecycle adapter for the unmodified official v0.23 A2 server.

The allocation runs the upstream `vllm serve` command directly. No model runner,
gateway, kernel patch, response header injection, or in-container observer exists.
"""
from __future__ import annotations
import json
from pathlib import Path
import shlex
import subprocess
import time
import urllib.request
from corpus_pipeline.serving.platform_client import InspireClient
from inspire.sdk.models_serving import ServingCreateSpec
from corpus_pipeline.serving.qwen36_w8a8_20261024 import backend as lifecycle
from corpus_pipeline.serving.qwen36_w8a8_20261024.runtime.contract import check, digest, file_sha

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT.parent
AUDIT = ASSETS / 'weights/official_version_audit'
native = lifecycle.native
pool = lifecycle.pool
IMAGE = 'vllm-ascend:v0.23.0rc1'
IMAGE_ID = 'image-03380f97-1e02-4ed9-8c04-1e1797828ce9'
OFFICIAL_SHA = '3f8f063ef8d7861f46fa31d9289f833d2a5c3454dac4afb2f4e8d48f7cd1848f'
ENVIRONMENT = {'VLLM_USE_MODELSCOPE':'True', 'PYTORCH_NPU_ALLOC_CONF':'expandable_segments:True',
    'HCCL_OP_EXPANSION_MODE':'AIV', 'HCCL_BUFFSIZE':'1024', 'OMP_NUM_THREADS':'1', 'TASK_QUEUE_ENABLE':'1'}


def asset_receipt(asset_root=None):
    assets = ASSETS if asset_root is None else Path(asset_root).resolve()
    path = assets / 'weights/WEIGHTS_READY.json'; ready = native.load(path)
    check(ready.get('complete') is True and ready['precision'] == 'W8A8_DYNAMIC'
          and ready['quantization'] == 'ascend', 'Verified W8A8 asset required')
    model = Path(ready['model_path']).resolve()
    for entry in ready['files']:
        file = (model / entry['name']).resolve()
        check(file.is_relative_to(model) and entry['verified'] is True and file.stat().st_size == entry['size_bytes'],
              'Verified asset changed')
        if not file.name.endswith('.safetensors'):
            check(file_sha(file) == entry['sha256'], 'Verified metadata changed')
    return ready


def official_contract(asset_root=None):
    assets = ASSETS if asset_root is None else Path(asset_root).resolve()
    audit = AUDIT if asset_root is None else assets / 'weights/official_version_audit'
    ready = asset_receipt(asset_root)
    check(file_sha(audit / 'guide-v023.md') == OFFICIAL_SHA, 'Fixed-tag official guide changed')
    raw = (audit / 'OFFICIAL_VLLM_SERVE_V023.txt').read_text()
    check(raw.strip() in (audit / 'guide-v023.md').read_text(), 'Command is not the literal upstream A2 recipe')
    argv = shlex.split(raw.replace('\\\n', ' '))
    check(argv[:3] == ['vllm','serve','Eco-Tech/Qwen3.6-35B-A3B-w8a8'], 'Wrong upstream model')
    argv[2] = ready['model_path']  # The sole model-command substitution.
    environment = '; '.join('export ' + key + '=' + shlex.quote(value) for key,value in ENVIRONMENT.items())
    environment += '; export LD_PRELOAD=/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}'
    command = shlex.join(['bash','-lc', environment + '; exec ' + shlex.join(argv)])
    return {'schema':'corpus.official_vllm_launch.v1', 'official_version':'v0.23.0rc1',
        'official_source_sha256':OFFICIAL_SHA, 'official_command_sha256':file_sha(audit / 'OFFICIAL_VLLM_SERVE_V023.txt'),
        'image':IMAGE, 'image_id':IMAGE_ID, 'argv':argv, 'environment':ENVIRONMENT,
        'ld_preload':'/usr/lib/aarch64-linux-gnu/libjemalloc.so.2:${LD_PRELOAD:-}', 'command':command,
        'weights_ready_sha256':file_sha(assets / 'weights/WEIGHTS_READY.json'), 'revision':ready['revision'],
        'model_path':ready['model_path'], 'precision':'W8A8_DYNAMIC', 'processor_revision':ready['processor_revision'],
        'quota':{'gpu_count':2,'cpu_count':16,'memory_size_gib':128}, 'nodes':1, 'replicas':1,
        'host_administrator_prerequisites_not_executed':['cpu scaling governor','vm.swappiness','kernel.numa_balancing','kernel.sched_migration_cost_ns']}


def prepare_contract():
    contract = official_contract(); path = ROOT / 'CONTRACT.json'
    if path.exists():
        check(native.load(path) == contract, 'Official contract changed')
    else:
        native.save(path, contract)
    return contract


class Backend(lifecycle.Backend):
    def __init__(self, asset_root=None, account_card_ceiling=256):
        pool.configure_management_proxy_bypass()
        self.asset_root = None if asset_root is None else Path(asset_root).resolve()
        self.assets = ASSETS if self.asset_root is None else self.asset_root
        self.contract_root = ROOT if self.asset_root is None else self.assets / 'official_v023'
        if type(account_card_ceiling) is not int or account_card_ceiling < 1:
            raise ValueError('Explicit positive account card ceiling required')
        self.account_card_ceiling = account_card_ceiling

    def binding(self):
        contract = native.load(self.contract_root / 'CONTRACT.json')
        check(contract == official_contract(self.asset_root), 'Official immutable launch changed')
        return contract, {'launch_contract_sha256':digest(contract), 'backend_sha256':file_sha(__file__),
            'lifecycle_sha256':file_sha(lifecycle.__file__), 'smoke_sha256':file_sha(self.contract_root / 'smoke.py'),
            'weights_ready_sha256':contract['weights_ready_sha256']}

    def prepare(self, slot_dir, slot, name, priority):
        slot_dir = Path(slot_dir).resolve(); contract,binding = self.binding()
        check(type(priority) is int and priority in (1,3,4,6)
              and name.replace('-','').isalnum(), 'Invalid allocation identity')
        registration = native.load(self.assets / 'MODEL_REGISTERED.json')
        check(registration['model_path'] == contract['model_path'] and registration['revision'] == contract['revision'],
              'Wrong registered asset')
        fields = dict(name=name, model=lifecycle.MODEL_REGISTRY_NAME, model_version=1,
            command=contract['command'], port=8000, workspace=native.WORKSPACE, project=native.PROJECT,
            group='910B资源', quota='2,16,128', image=IMAGE, replicas=1, nodes_per_replica=1,
            shm_gib=128, priority=priority, auto_scaling=False, public_path_readonly=True,
            description='Official v0.23.0rc1 A2 unmodified vllm serve; W8A8 TP2; external lifecycle only')
        with InspireClient(account='wjx-ascend') as client:
            plan = client.servings.plan(ServingCreateSpec(**fields))
            pool.observable_project_budget(client,2,ceiling=self.account_card_ceiling)
        payload = plan.create_kwargs
        check(payload['mirror_id'] == IMAGE_ID and payload['model_id'] == registration['model_id']
              and payload['task_priority'] == priority and payload['replicas'] == payload['node_num_per_replica'] == 1,
              'Resolved native identity changed')
        check(all(payload['resource_spec_price'][k] == v for k,v in contract['quota'].items()), 'Resolved quota changed')
        state = {'schema_version':'corpus_pipeline.native_serving.v1','scope':str(slot_dir),'slot':slot,
            'role':'verifier','tier':'high' if priority in (4,6) else 'low','priority':priority,'cards':2,
            'name':name,'service_name':name,'operation_id':name,'deployment_id':name,
            'create_spec':fields,'payload':payload,'binding':binding,'model':'qwen3.6',
            'revision':contract['revision'],'precision':'W8A8_DYNAMIC','processor_revision':contract['processor_revision'],
            'expected_nodes':1,'expected_replicas':1,'api_key_file':str(self.assets.parent/'private/inspire-serving.key'),
            'api_key_env':'INF_API_KEY','auto_recovery_enabled':priority in (1,3),'phase':'PREPARED','created_at':time.time(),
            'release_verified':False,'canary_passed':False,'max_images':1,'qualification_kind':'basic_smoke',
            'launch_contract_sha256':binding['launch_contract_sha256'], 'transport_mode':'native_openai',
            'attestation_scope':'platform_instance_and_registry', 'native_stream':True, 'expected_boots':[],
            'request_overrides':{'chat_template_kwargs':{'enable_thinking':False},
                                 'mm_processor_kwargs':{'max_pixels':1048576}}}
        self.validate(state)
        return state

    def publish(self, slot_dir, state, ready):
        path = Path(slot_dir)/'registry.json'
        previous = native.load(path) if path.exists() else {'registry_generation':0,'endpoints':[]}
        state = dict(state)
        if not state.get('endpoint') and previous['endpoints']:
            state['endpoint'] = previous['endpoints'][0]['base_url'].removesuffix('/v1')
        rows = []
        if state.get('service_id') and state.get('endpoint'):
            row = native.candidate(state, bool(ready))
            row.update(max_output_tokens=3000, qualification_kind='basic_smoke', transport_mode='native_openai',
                native_stream=True, attestation_scope='platform_instance_and_registry',
                platform_instance_id=state.get('platform_instance_id'), launch_contract_sha256=state['launch_contract_sha256'],
                response_boot_attested=False, expected_boots=[])
            rows.append(row)
        native.save(path, {'schema_version':'corpus_pipeline.endpoints.v1','registry_generation':previous['registry_generation']+1,
            'updated_at':time.time(),'endpoints':rows,'publisher':'official_v023_external_sdk'})

    def attest(self, client, detail, instances, state, slot_dir):
        check(len(instances)==1 and instances[0].status=='INSTANCE_RUNNING', 'Native instance is not running')
        check(self.binding()[1] == state['binding'], 'External launch binding changed')
        signature = native.instance_signature(instances)
        incarnation = digest({'service_id':state['service_id'],'platform':signature})
        path = Path(slot_dir)/'attestations'/(incarnation+'.json')
        if path.exists():
            proof = native.load(path)
            check(proof['platform_instances']==signature and proof['launch_contract_sha256']==state['launch_contract_sha256'],
                  'Native instance receipt changed')
            return proof
        # Read-only package metadata; no torch/model imports or device context.
        code = "import importlib.metadata as m,json; print(json.dumps({x:m.version(x) for x in ['vllm','vllm-ascend','torch','torch-npu','transformers']}))"
        result = client.servings.exec(detail.ref, workspace=native.WORKSPACE, instance=instances[0].raw['name'],
            command=shlex.join(['python3','-c',code]), timeout=15, max_output_bytes=4096)
        check(result.completed and result.returncode==0 and not result.truncated, 'Package metadata observation incomplete')
        packages = next(json.loads(line) for line in result.stdout.splitlines() if line.strip().startswith('{'))
        check(packages['vllm'].split('+')[0]=='0.23.0' and packages['vllm-ascend']=='0.23.0rc1'
              and packages['torch'].split('+')[0]=='2.10.0' and packages['torch-npu']=='2.10.0.post2',
              'Actual image does not match the official version pair')
        proof = {'incarnation':incarnation,'platform_instance_id':instances[0].raw['name'],
            'platform_instances':signature,'packages':packages,'launch_contract_sha256':state['launch_contract_sha256'],
            'runtime_revision':'official-native-sha256:'+digest({'packages':packages,'launch':state['launch_contract_sha256']}),
            'attestation_scope':'platform_instance_and_registry','response_boot_attested':False,'time':time.time()}
        native.save(path,proof)
        return proof

    @staticmethod
    def health(state):
        opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        headers={'Authorization':'Bearer '+Path(state['api_key_file']).read_text().strip()}
        try:
            request=urllib.request.Request(state['endpoint'].rstrip('/')+'/health',headers=headers)
            with opener.open(request,timeout=4) as response:
                if response.status!=200:return False
            request=urllib.request.Request(state['endpoint'].rstrip('/')+'/v1/models',headers=headers)
            with opener.open(request,timeout=4) as response:
                body=json.load(response)
            return [row['id'] for row in body['data']]==['qwen3.6']
        except Exception:
            return False

    def smoke(self, slot_dir, state):
        directory=Path(slot_dir)/'smoke'/state['incarnation'];directory.mkdir(parents=True,exist_ok=True)
        result_path=directory/'RESULT.json'
        if result_path.exists():
            result=native.load(result_path)
            check(all(result.get(k)==state.get(k) for k in ('model','revision','precision','processor_revision',
                'runtime_revision','incarnation','platform_instance_id','launch_contract_sha256')), 'Stale smoke identity')
            check(result.get('response_boot_attested') is False and result.get('qualification_kind')=='basic_smoke',
                  'Smoke attestation scope changed')
            state['qualification_receipt']=str(result_path);state['smoke_status']='PASSED' if result['passed'] else 'FAILED'
            return result['passed'] is True
        marker=directory/'INTENT.json'
        if marker.exists():
            state['smoke_status']='RUNNING_OR_UNKNOWN';return False
        with (self.assets.parents[1]/'model_comparison_20261023/manifest_all140.jsonl').open() as stream:
            sample=json.loads(next(stream))
        request={key:state[key] for key in ('model','revision','precision','processor_revision','runtime_revision',
            'incarnation','platform_instance_id','launch_contract_sha256','api_key_file','request_overrides')}
        request.update(base_url=state['endpoint'].rstrip('/')+'/v1',asset={k:sample[k] for k in ('asset_id','sha256','original_ref')})
        native.save(directory/'INPUT.json',request);native.save(marker,{'time':time.time(),'input_sha256':file_sha(directory/'INPUT.json')})
        command=[str(self.assets.parents[2]/'.venv/bin/python'),'-m',
            'corpus_pipeline.serving.qwen36_w8a8_20261024.official_v023.smoke','--input',str(directory/'INPUT.json')]
        with (directory/'stdout.log').open('ab') as output:
            child=subprocess.Popen(command,cwd=self.assets.parents[2],stdin=subprocess.DEVNULL,stdout=output,
                stderr=subprocess.STDOUT,start_new_session=True)
        native.save(directory/'PROCESS.json',native.identity(child.pid));state['smoke_status']='RUNNING'
        return False

    def observe(self, slot_dir, state):
        state=dict(state);ready=False
        with InspireClient(account='wjx-ascend') as client:
            state,detail=self.reconcile(client,state)
            instances=client.servings.instances(detail.ref,workspace=native.WORKSPACE)
            previous=state.get('platform_status');now=time.time()
            state.update(platform_status=detail.status,endpoint=detail.to_dict()['endpoint'],last_observed_at=now,
                instance_statuses=[x.status for x in instances],observation_error_type=None)
            if detail.status=='RUNNING':state.update(was_running=True,last_running_at=now,stopped_transition_at=None)
            elif detail.status=='STOPPED' and state.get('was_running') and previous!='STOPPED':state['stopped_transition_at']=now
            state['events']=list(client.servings.events(detail.ref,workspace=native.WORKSPACE,limit=30).items)
            if detail.status=='RUNNING' and not lifecycle.disabled(slot_dir,state):
                try:
                    proof=self.attest(client,detail,instances,state,slot_dir)
                    if state.get('incarnation')!=proof['incarnation']:
                        state.update(qualification_receipt=None,canary_passed=False,smoke_status=None)
                    state.update({k:proof[k] for k in ('incarnation','platform_instance_id','runtime_revision')})
                    if self.health(state):ready=self.smoke(slot_dir,state)
                    state['readiness_reason']='BASIC_SMOKE_PASSED' if ready else state.get('smoke_status') or 'OFFICIAL_ENGINE_LOADING'
                except Exception as error:
                    state.update(readiness_reason='ATTESTATION_OR_LOADING_FAILED',observation_error_type=type(error).__name__)
            else:state['readiness_reason']='DISABLED_OR_NOT_RUNNING'
        ready=ready and not lifecycle.disabled(slot_dir,state)
        state.update(canary_passed=ready,heartbeat_at=time.time())
        if ready:state.update(phase='QUALIFIED',first_qualified_at=state.get('first_qualified_at',time.time()))
        self.publish(slot_dir,state,ready)
        return state
