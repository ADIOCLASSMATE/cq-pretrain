"""Compile a new run from maintained modules; never copy another run's ops/."""
from __future__ import annotations
from pathlib import Path
import shlex
import sys

from .config import (SCHEMA, SOURCES, LOGICAL_SOURCES, SUPPLEMENTAL_PARENTS,
                     check_config, validate_config, immutable, read, sha)

PACKAGE = Path(__file__).resolve().parents[1]
CODE_DIRS = ('orchestration', 'unified', 'production', 'cluster', 'postprocess',
            'dedup', 'api_review', 'api_benchmark', 'detector_calibration', 'resource_manager', 'serving',
            'serving/inspire_serving', 'serving/qwen36_w8a8_20261024',
            'serving/qwen36_w8a8_20261024/official_v023', 'serving/qwen36_w8a8_20261024/runtime')
RETIRED = {'pipeline.py', 'production/staged_run.py',
           'cluster/fragment_allocator.py', 'cluster/fragment_controller.py', 'cluster/submit_jobs.py',
           'cluster/fragment_consolidation.py', 'cluster/tuned_rollout.py',
           'serving/ai_review_pool.py', 'serving/configured_pool.py',
           'serving/inspire_serving/downsize.py'}


def source_files():
    files = set(PACKAGE.glob('*.py'))
    for directory in CODE_DIRS:
        files.update((PACKAGE/directory).glob('*.py'))
    for directory in ('config', 'prepared/config', 'detector_calibration', 'unified/config'):
        files.update((PACKAGE/directory).glob('*.json'))
    files.add(PACKAGE/'production/image_screening_prompt_v4.txt')
    files.add(PACKAGE/'production/image_screening_prompt_v3.txt')
    files.add(PACKAGE/'postprocess/requirements.txt')
    files.add(PACKAGE/'requirements.txt')
    files.add(PACKAGE/'api_review/requirements.txt')
    return sorted(p for p in files if not p.name.startswith('test_')
                  and str(p.relative_to(PACKAGE)) not in RETIRED)


def snapshot(runtime):
    """Code/config only: no checkpoints, data, private keys, probes or old runs."""
    runtime = Path(runtime)
    expected = {str(p.relative_to(PACKAGE)): sha(p) for p in source_files()}
    manifest = runtime/'RUNTIME.json'
    if manifest.exists():
        if read(manifest).get('files') != expected:
            raise ValueError('Existing runtime is immutable; choose a new run')
        verify_runtime(runtime)
        return
    if runtime.exists() and any(runtime.iterdir()):
        raise ValueError('Unbound runtime directory is not empty')
    for source in source_files():
        target = runtime/'corpus_pipeline'/source.relative_to(PACKAGE)
        target.parent.mkdir(parents=True, exist_ok=True)
        # A copy is intentional: later canonical edits must not change a run.
        with target.open('xb') as stream:
            stream.write(source.read_bytes())
    immutable(manifest, {'schema': 'corpus.runtime_snapshot.v1', 'files': expected})


def verify_runtime(runtime):
    runtime = Path(runtime)
    proof = read(runtime/'RUNTIME.json')
    if not proof or proof.get('schema') != 'corpus.runtime_snapshot.v1' or not proof.get('files'):
        raise ValueError('Missing maintained runtime snapshot')
    for name, checksum in proof['files'].items():
        path = runtime/'corpus_pipeline'/name
        if Path(name).is_absolute() or '..' in Path(name).parts or sha(path) != checksum:
            raise ValueError('Frozen runtime changed')
    return proof


def launcher(cfg, *, npu=False):
    q = shlex.quote
    text = '#!/usr/bin/env bash\nset -eo pipefail\n'
    if npu:
        text += 'source '+q(cfg.get('cann_env', '/usr/local/Ascend/ascend-toolkit/set_env.sh'))+'\n'
        text += '''for CORPUS_DRIVER_LIB in /usr/local/Ascend/driver/lib64/driver /usr/local/Ascend/driver/lib64/common /usr/local/Ascend/driver/lib64; do
  if [[ -d "$CORPUS_DRIVER_LIB" ]]; then
    export LD_LIBRARY_PATH="$CORPUS_DRIVER_LIB${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
  fi
done
'''
    # CANN appends essential Python paths: put our package first without erasing them.
    text += 'export PYTHONPATH='+q(cfg['runtime'])+'"${PYTHONPATH:+:$PYTHONPATH}"\n'
    text += 'export CORPUS_DATASET_ROOTS='+q(cfg['source_roots'])+'\n'
    text += 'export TORCH_DEVICE_BACKEND_AUTOLOAD=0 PYTHONUNBUFFERED=1 HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1\n'
    text += 'export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=1 TOKENIZERS_PARALLELISM=false\n'
    if npu: text += 'export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True\n'
    text += 'cd '+q(cfg['runtime'])+'\nexec '+q(cfg['python'])+' "$@"\n'
    return text


def prepare(path):
    cfg = read(path)
    if not isinstance(cfg, dict) or cfg.get('schema') != SCHEMA:
        raise ValueError('New tasks require corpus.orchestration.v1')
    run = Path(cfg['run']).resolve()
    if (run/'EXECUTION.json').exists() or (run/'sources').exists():
        raise ValueError('prepare requires a fresh run, never an existing corpus run')
    cfg.setdefault('repository', str(PACKAGE.parent))
    cfg.setdefault('runtime', str(run/'runtime'))
    cfg.setdefault('python', sys.executable)
    cfg.setdefault('source_roots', str(run/'source_roots.json'))
    cfg.setdefault('review_template', str(run/'ops/review_pool_template.json'))
    cfg.setdefault('sources', list(SOURCES))
    cfg.setdefault('logical_sources', list(LOGICAL_SOURCES))
    cfg.setdefault('supplemental_logical_parents', SUPPLEMENTAL_PARENTS)
    cfg.setdefault('dedup_mode', 'exact_identity_ordered_supervision_v1')
    cfg.setdefault('priority', 1)
    cfg.setdefault('cpu_priority', 6)
    # Budget is explicit even though 512 is this project's approved upper bound.
    if 'max_cards' not in cfg: raise ValueError('Explicit max_cards required')
    validate_config(cfg)
    from .cpu import validate_workers
    validate_workers(cfg)
    platform = cfg['platform']
    for key in ('account','workspace','project','group','image'):
        if not platform.get(key): raise ValueError('Explicit live platform selection required: '+key)
    if platform['account'] != cfg['account'] or platform['workspace'] != cfg['workspace']:
        raise ValueError('Platform execution scope differs')
    roots = cfg.pop('dataset_roots', None)
    if roots is None: roots = read(cfg['source_roots'])
    if not isinstance(roots, dict) or not roots: raise ValueError('Missing portable dataset roots')
    for value in roots.values():
        source = Path(value).resolve()
        if not Path(value).is_absolute() or run.is_relative_to(source) or source.is_relative_to(run):
            raise ValueError('Raw dataset roots must be absolute and outside the derived run')
    workers=[]
    for row in cfg['cpu_workers']:
        if 'ingest' not in row['roles']:continue
        count=row.get('workers',12)
        if type(count) is not int or not 1<=count<=32:raise ValueError('Invalid CPU worker count')
        workers.extend(row['name']+'-'+str(i) for i in range(count))
    if not workers or len(workers)!=len(set(workers)):raise ValueError('Explicit unique CPU ingestion partitions required')
    immutable(cfg['source_roots'], roots)
    run.mkdir(parents=True, exist_ok=True)
    temporary = run/'ops/prepared_config.json'
    immutable(temporary, cfg)
    cfg = check_config(temporary)
    snapshot(cfg['runtime'])
    for name, npu in [('cpu_python.sh', False), ('npu_python.sh', True)]:
        target = run/'ops'/name
        content = launcher(cfg, npu=npu)
        if target.exists() and target.read_text() != content: raise ValueError('Immutable launcher changed')
        target.write_text(content); target.chmod(0o755)
    template = {'adapter': 'corpus_pipeline.resource_manager.inspire_adapter:InspireAdapter',
        'state_dir': str(run/'fleet/template/controller'),
        'adapter_config': {**platform, 'backend':'notebook', 'priority':1,
            'global_max_cards':cfg['max_cards'], 'notebook_placement':'native_scheduler',
            'omit_unsupported_public_path_flag':True, 'public_path_readonly':False,
            'repository':cfg['runtime'], 'remote_python':str(run/'ops/npu_python.sh'),
            'shm_gib':64, 'registry_dir':str(run/'fleet/template/adapter'),
            'worker_state_dir':str(run/'fleet/remote_workers')},
        'workload': {'scope':cfg['name_prefix']+'-frontpass', 'profile':'frontpass',
            'workload_kind':'batch', 'max_cards':cfg['max_cards'], 'allowed_cards':[1],
            'replica_cards':1, 'migration_policy':'disabled', 'min_ready_replicas':0,
            'poll_seconds':15, 'command':[str(run/'ops/npu_python.sh'),'-m','corpus_pipeline.unified','--help']}}
    immutable(run/'ops/fleet_template.json', template)
    assets = cfg['model_assets']
    review = {'schema_version':'corpus.native_budget_pool.v1', 'root':'{source_run}/review_pool',
        'backend_file':'{runtime}/corpus_pipeline/serving/qwen36_w8a8_20261024/official_v023/backend.py',
        'backend_kwargs':{'asset_root':assets, 'account_card_ceiling':cfg['max_cards']},
        'cards_per_replica':2, 'replicas':cfg['max_cards']//2, 'max_cards':cfg['max_cards'],
        'priority':1, 'name_prefix':cfg['name_prefix']+'-{source_slug}',
        'per_replica_max_inflight':128, 'poll_seconds':8, 'provision_seconds':3, 'external_reservations':[]}
    immutable(cfg['review_template'], review)
    immutable(run/'ops/INGEST_WORKERS.json',{'workers':workers})
    return {'prepared':True, 'config':str(temporary), 'runtime':cfg['runtime'],
            'resources_started':False, 'max_cards':cfg['max_cards'], 'priority':1}
