"""Immutable run configuration and explicit execution environment."""
import hashlib,json,os
from pathlib import Path
from datetime import datetime

SCHEMA='corpus.orchestration.v1'
LOGICAL_SOURCES=('OpenUni','ScaleCap450k_image','DenseFusion-1M','pixmo-ask-model-anything','FineVision','BLIP3o-Pretrain-Long-Caption')
SUPPLEMENTAL_PARENTS={'DenseFusion-4V-100k':'DenseFusion-1M'}
SOURCES=(*LOGICAL_SOURCES,*SUPPLEMENTAL_PARENTS)

PROXY_KEYS=('HTTPS_PROXY','HTTP_PROXY','ALL_PROXY','https_proxy','http_proxy','all_proxy')

def read(path,default=None):
    path=Path(path)
    return json.loads(path.read_text()) if path.exists() else default


def sha(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for part in iter(lambda:stream.read(1<<20),b''):h.update(part)
    return h.hexdigest()


def save(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    temporary=path.with_name('.'+path.name+'.'+str(os.getpid()))
    with temporary.open('w') as stream:
        json.dump(value,stream,ensure_ascii=False,sort_keys=True);stream.write('\n');stream.flush();os.fsync(stream.fileno())
    os.replace(temporary,path)


def immutable(path,value):
    old=read(path)
    if old is not None and old!=value:raise ValueError('Immutable configuration changed: '+str(path))
    if old is None:save(path,value)


def check_config(path):
    cfg=read(path)
    validate_config(cfg)
    cfg['config_path']=str(Path(path).resolve())
    return cfg


def validate_config(cfg):
    if cfg.get('schema')!=SCHEMA or tuple(cfg.get('sources',()))!=SOURCES:raise ValueError('Explicit logical-six execution-source controller required')
    if cfg.get('logical_sources')!=list(LOGICAL_SOURCES) or cfg.get('supplemental_logical_parents')!=SUPPLEMENTAL_PARENTS:
        raise ValueError('Supplemental source logical coverage binding differs')
    for k in ('run','repository','runtime','python','sdk_python','review_template','source_roots',
              'sscd_qualification','frontpass_qualification','baseline_index','baseline_reconciliation','api_key_file','model_assets'):
        if not isinstance(cfg.get(k),str) or not Path(cfg[k]).is_absolute():raise ValueError('Absolute '+k+' required')
    if cfg.get('dedup_mode')!='exact_identity_ordered_supervision_v1':raise ValueError('Explicit exact-only mode required')
    if type(cfg.get('max_cards')) is not int or not 2<=cfg['max_cards']<=512 or type(cfg.get('priority')) is not int or cfg['priority']!=1:
        raise ValueError('Explicit LOW 1 NPU budget in [2,512] required')
    if type(cfg.get('cpu_priority')) is not int or cfg['cpu_priority'] not in (1,6):raise ValueError('CPU priority must be explicit LOW 1 or HIGH 6')
    if not isinstance(cfg.get('review_notebook'),str) or not cfg['review_notebook']:
        raise ValueError('Explicit retained review Notebook required')
    if type(cfg.get('queue_port_base')) is not int or not 1024<=cfg['queue_port_base']<=65535-len(SOURCES):
        raise ValueError('Independent queue port range required')
    deadline=datetime.fromisoformat(cfg['deadline_utc'].replace('Z','+00:00'))
    if deadline.tzinfo is None:raise ValueError('Deadline timezone required')
    cfg['deadline_epoch']=deadline.timestamp()
    return cfg


def environment(cfg):
    value=dict(os.environ,PYTHONPATH=cfg['runtime'],CORPUS_DATASET_ROOTS=cfg['source_roots'],
        OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',OPENBLAS_NUM_THREADS='1',TORCH_DEVICE_BACKEND_AUTOLOAD='0',PYTHONUNBUFFERED='1')
    for name in PROXY_KEYS:value.pop(name,None)
    return value
