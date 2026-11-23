"""Recover original FineVision supervision with full upstream/local byte binding.

Inventory reads only the image path/list-level column. Recovery downloads one
pinned Parquet into bounded memory, verifies its official LFS SHA, binds every
image against mounted bytes, then persists annotation metadata only. File locks
and atomic receipts allow many independent CPU notebooks to share the queue.
"""
import argparse
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor,as_completed
from contextlib import contextmanager
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import random
import shutil
import socket
import sys
import tempfile
import threading
import time

import pyarrow.parquet as pq

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from corpus_pipeline.production.common import atomic_json,canonical,digest,file_sha
from recover_genund_annotation_ranges import RangeFile,fetch
from finevision_local_assets import build_partition,lookup

SCHEMA='genund.finevision.recovery_queue.v1'


class CachedRangeFile(RangeFile):
    """Prefetched physical column chunks avoid serial WAN calls per row group."""
    def __init__(self,url,size):
        super().__init__(url,size); self.chunks=[]
    def read(self,size=-1):
        size=self.size-self.position if size<0 else min(size,self.size-self.position)
        for start,data in self.chunks:
            if start<=self.position and self.position+size<=start+len(data):
                offset=self.position-start; self.position+=size; return data[offset:offset+size]
        return super().read(size)


@contextmanager
def lock(path):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a') as stream:
        try: fcntl.flock(stream,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError:
            yield False; return
        try: yield True
        finally: fcntl.flock(stream,fcntl.LOCK_UN)


def prepare(run,root):
    run=Path(run); root=Path(root); area=run/'finevision_recovery'; area.mkdir(parents=True,exist_ok=True)
    info_path=run/'annotations/FineVision/hub_info.json'; info=json.loads(info_path.read_text())
    subsets=sorted(path.name for path in (root/'FineVision').iterdir() if path.is_dir())
    upstream={row['config_name']:row for row in info['cardData']['dataset_info']}
    if len(subsets)!=169 or any(name not in upstream for name in subsets): raise ValueError('mounted FineVision subset contract differs')
    tasks=[]
    for item in sorted(info['siblings'],key=lambda row:row['rfilename']):
        name=item['rfilename']; subset=name.split('/')[0]
        if subset not in subsets or not name.endswith('.parquet'): continue
        task={'subset':subset,'path':name,'size':item['size'],'sha256':item['lfs']['sha256'],'revision':info['sha']}
        task['task_id']=digest(task); tasks.append(task)
    expected={name:next(split['num_examples'] for split in upstream[name]['splits'] if split['name']=='train') for name in subsets}
    plan={'schema':SCHEMA,'complete':True,'root':str(root),'source':'FineVision','revision':info['sha'],
          'source_info_sha256':file_sha(info_path),'tasks':tasks,'subsets':subsets,'expected_rows':expected,
          'source_bytes':sum(task['size'] for task in tasks),'task_count':len(tasks),
          'mapping_rule':'every encoded source image SHA must match a mounted candidate; ordinals never authorize a match'}
    path=area/'PLAN.json'
    if path.exists() and json.loads(path.read_text())!=plan: raise ValueError('FineVision recovery plan changed')
    if not path.exists(): atomic_json(path,plan)
    return plan


def source_url(task):
    return f"https://huggingface.co/datasets/HuggingFaceM4/FineVision/resolve/{task['revision']}/{task['path']}?download=true"


def download_parquet(task,directory,connections=64):
    """Restartable ranges: bounded scratch, complete LFS SHA before any use."""
    parts=directory/'download_ranges'; parts.mkdir(parents=True,exist_ok=True); block=4<<20
    starts=list(range(0,task['size'],block)); todo=[]; cached=0; transferred=0; started=time.time()
    for start in starts:
        path=parts/f'{start:012d}.part'; size=min(block,task['size']-start)
        if path.exists() and path.stat().st_size==size: cached+=size
        else: todo.append(start)
    url=source_url(task)
    if todo: fetch(url,max(0,task['size']-65536),task['size'])  # inexpensive signed-CDN warmup
    def one(start):
        data=fetch(url,start,min(task['size'],start+block)); path=parts/f'{start:012d}.part'
        temporary=path.with_suffix('.partial'); temporary.write_bytes(data); os.replace(temporary,path); return len(data)
    last=0
    with ThreadPoolExecutor(max_workers=connections) as pool:
        for future in as_completed([pool.submit(one,start) for start in todo]):
            transferred+=future.result(); now=time.time()
            if now-last>=10 or cached+transferred==task['size']:
                atomic_json(directory/'DOWNLOAD_PROGRESS.json',{'task_id':task['task_id'],'source_bytes':task['size'],
                    'cached_bytes':cached,'transferred_bytes':transferred,'available_bytes':cached+transferred,
                    'started_at':started,'updated_at':now,'MB_per_second':transferred/max(now-started,.001)/1e6,
                    'connections':connections}); last=now
    blob=b''.join((parts/f'{start:012d}.part').read_bytes() for start in starts)
    if len(blob)!=task['size'] or hashlib.sha256(blob).hexdigest()!=task['sha256']:
        for path in parts.glob('*.part'): path.unlink()
        raise ValueError('complete upstream Parquet does not match pinned official LFS SHA')
    return blob


def inventory(task,area):
    target=area/'inventory'/task['task_id']; target.mkdir(parents=True,exist_ok=True)
    with CachedRangeFile(source_url(task),task['size']) as stream:
        reader=pq.ParquetFile(stream)
        columns=[column.path for column in reader.schema]
        if 'images.list.element.path' not in columns: raise ValueError('original images.path physical column missing')
        index=columns.index('images.list.element.path'); ranges=[]
        for i in range(reader.metadata.num_row_groups):
            column=reader.metadata.row_group(i).column(index)
            start=min(value for value in [column.dictionary_page_offset,column.data_page_offset] if value is not None and value>=0)
            ranges.append((start,start+column.total_compressed_size))
        with ThreadPoolExecutor(max_workers=8) as pool:
            stream.chunks=list(pool.map(lambda pair:(pair[0],fetch(stream.url,*pair)),ranges))
        stream.bytes_read+=sum(len(data) for _,data in stream.chunks); stream.requests+=len(ranges)
        table=reader.read(columns=['images.list.element.path'])
        counts=[len(images) if images is not None else 0 for images in table['images'].to_pylist()]
        if len(counts)!=reader.metadata.num_rows: raise ValueError('original image-list coverage differs')
        record={'complete':True,'task':task,'rows':len(counts),'images':sum(counts),
                'row_groups':reader.metadata.num_row_groups,'row_group_rows':[reader.metadata.row_group(i).num_rows for i in range(reader.metadata.num_row_groups)],
                'network_bytes':stream.bytes_read,'range_requests':stream.requests,'created_by':reader.metadata.created_by}
    # Only row counts are needed for global offsets; every image byte is checked later.
    atomic_json(target/'COMPLETE.json',record)
    return record


def binding_plan(plan,area):
    marker=area/'BINDING_PLAN.json'
    if marker.exists(): return json.loads(marker.read_text())
    with lock(area/'binding_plan.lock') as acquired:
        if not acquired: return None
        if marker.exists(): return json.loads(marker.read_text())
        by_subset=defaultdict(list)
        for task in plan['tasks']:
            path=area/'inventory'/task['task_id']/'COMPLETE.json'
            if not path.exists(): return None
            value=json.loads(path.read_text())
            if value.get('complete') is not True or value['task']!=task: raise ValueError('inventory receipt identity differs')
            by_subset[task['subset']].append((task,value))
        tasks=[]
        for subset,values in sorted(by_subset.items()):
            row_offset=image_offset=0
            for task,value in values:
                tasks.append({**task,'row_start':row_offset,'image_start':image_offset,'rows':value['rows'],'images':value['images']})
                row_offset+=value['rows']; image_offset+=value['images']
            if row_offset!=plan['expected_rows'][subset]: raise ValueError('pinned FineVision split row coverage differs: '+subset)
        result={'schema':SCHEMA,'complete':True,'source':'FineVision','plan_sha256':file_sha(area/'PLAN.json'),
                'tasks':tasks,'records':sum(task['rows'] for task in tasks),'image_slots':sum(task['images'] for task in tasks)}
        atomic_json(marker,result); return result


def recover(task,plan,area):
    directory=area/'recovered'/task['task_id']; directory.mkdir(parents=True,exist_ok=True)
    started=time.time(); blob=download_parquet(task,directory)
    reader=pq.ParquetFile(io.BytesIO(blob)); temporary=directory/'annotations.partial.jsonl'
    rows=images=0; missing=[]; ambiguous=0; restored_count=restored_bytes=0
    recovery_binding_path=area/'RECOVERED_ASSET_BINDING.json'
    recovery_binding=json.loads(recovery_binding_path.read_text()) if recovery_binding_path.exists() else None
    subset_root=Path(plan['root'])/'FineVision'/task['subset']
    with temporary.open('w',encoding='utf-8') as output:
        for batch in reader.iter_batches(batch_size=32):
            for original in batch.to_pylist():
                originals=original.get('images') or []; bound=[]; record_misses=[]
                row_index=task['row_start']+rows
                for image_index,image in enumerate(originals):
                    source=image.get('bytes')
                    if not isinstance(source,bytes) or not source: raise ValueError('original image bytes missing')
                    sha=hashlib.sha256(source).hexdigest(); flat=task['image_start']+images
                    candidates=[f'{flat}.jpg',f'{row_index}_{image_index}.jpg']
                    if len(originals)==1: candidates.append(f'{row_index}.jpg')
                    matched=[]
                    for name in dict.fromkeys(candidates):
                        path=subset_root/name
                        if path.is_file() and file_sha(path)==sha: matched.append(path)
                    if not matched and (area/'local_assets'/task['subset']/'COMPLETE.json').exists():
                        found=lookup(area,task['subset'],sha)
                        if found:
                            path=Path(plan['root'])/found[0]
                            if file_sha(path)!=sha: raise ValueError('mounted asset changed after full SHA indexing')
                            matched.append(path)
                    if not matched:
                        if recovery_binding is None:
                            record_misses.append({'row_index':row_index,'image_index':image_index,'image_sha256':sha,'candidates':candidates})
                        else:
                            # Missing is established only by a complete local SHA index.
                            if not (area/'local_assets'/task['subset']/'COMPLETE.json').exists():
                                raise RuntimeError('local SHA index incomplete; cannot declare a source image missing')
                            rel=Path('FineVision')/sha[:2]/(sha+'.img'); target=Path(recovery_binding['root'])/rel
                            target.parent.mkdir(parents=True,exist_ok=True)
                            if target.exists():
                                if file_sha(target)!=sha: raise ValueError('recovered content-addressed image changed')
                            else:
                                descriptor,name=tempfile.mkstemp(prefix='.'+sha+'.',suffix='.partial',dir=target.parent)
                                os.close(descriptor); temporary_image=Path(name)
                                temporary_image.write_bytes(source)
                                if file_sha(temporary_image)!=sha: raise ValueError('recovered original image SHA differs')
                                os.replace(temporary_image,target)
                            bound.append({'path':str(rel),'sha256':sha,'dataset_id':recovery_binding['dataset_id'],
                                'upstream_parquet':task['path'],'upstream_row_index':rows,'upstream_image_index':image_index,
                                'recovery_reason':'not_present_in_complete_mounted_subset_sha_index'})
                            restored_count+=1; restored_bytes+=len(source)
                    else:
                        path=matched[0]
                        bound.append({'path':str(path.relative_to(plan['root'])),'sha256':sha,
                                      'upstream_parquet':task['path'],'upstream_row_index':rows,'upstream_image_index':image_index})
                        ambiguous+=len(matched)>1
                    images+=1
                if record_misses:
                    missing.extend(record_misses)
                else:
                    recovered={**original,'images':bound,'_original_binding':{'repo':'HuggingFaceM4/FineVision',
                        'revision':task['revision'],'parquet':task['path'],'parquet_sha256':task['sha256'],
                        'row_index_in_parquet':rows,'global_row_index':row_index,
                        'original_image_paths':[value.get('path') for value in originals]}}
                    output.write(canonical(recovered)+'\n')
                rows+=1
    if rows!=task['rows'] or images!=task['images']: raise ValueError('full-source image/row counts differ from inventory')
    if missing:
        with temporary.open() as stream: written=sum(1 for _ in stream)
        atomic_json(directory/'UNRESOLVED.json',{'complete':False,'task':task,'records':rows,'missing_images':missing,
                                               'written_verified_records':written})
        raise ValueError(f'{len(missing)} original image slots lack verified mounted binding')
    target=directory/'annotations.jsonl'; os.replace(temporary,target)
    result={'complete':True,'task':task,'records':rows,'image_slots':images,'multiple_equal_candidates':ambiguous,
            'path':str(target),'sha256':file_sha(target),'size_bytes':target.stat().st_size,
            'source_parquet_sha_verified':True,'every_image_sha_verified':True,'source_modified':False,
            'image_payloads_persisted':bool(restored_count),'recovered_missing_image_slots':restored_count,
            'recovered_missing_image_slot_bytes':restored_bytes,'verified_upstream_bytes':task['size'],
            'seconds':time.time()-started}
    atomic_json(directory/'COMPLETE.json',result)
    shutil.rmtree(directory/'download_ranges',ignore_errors=True)
    return result


@contextmanager
def file_claim(area,phase,task,worker_name):
    path=area/'atomic_claims'/(phase+'-'+task['task_id'])
    path.parent.mkdir(parents=True,exist_ok=True)
    try: path.mkdir()
    except FileExistsError:
        yield False; return
    try:
        atomic_json(path/'OWNER.json',{'worker':worker_name,'pid':os.getpid(),'host':socket.gethostname(),'at':time.time()})
        yield True
    finally: shutil.rmtree(path)


def work_one(task,plan,area,phase,worker_name):
    directory=area/('inventory' if phase=='inventory' else 'recovered')/task['task_id']
    if (directory/'COMPLETE.json').exists(): return 'complete'
    if phase=='recover' and not (area/'local_assets'/task['subset']/'COMPLETE.json').exists(): return 'index_wait'
    error=directory/'ERROR.json'
    if error.exists() and time.time()-error.stat().st_mtime<300: return 'backoff'
    with file_claim(area,phase,task,worker_name) as acquired:
        if not acquired: return 'busy'
        if (directory/'COMPLETE.json').exists(): return 'complete'
        directory.mkdir(parents=True,exist_ok=True)
        try:
            value=inventory(task,area) if phase=='inventory' else recover(task,plan,area)
            error.unlink(missing_ok=True); return 'processed'
        except Exception as exc:
            atomic_json(error,{'task':task,'phase':phase,'host':socket.gethostname(),'at':time.time(),
                               'error_type':type(exc).__name__,'error':str(exc)[:1000]})
            return 'failed'


def worker(run,workers=8,max_tasks=None,heartbeat=None,worker_name=None,partition_index=0,partitions=1):
    area=Path(run)/'finevision_recovery'; plan=json.loads((area/'PLAN.json').read_text())
    if (area/'PAUSE').exists(): return {'paused':True}
    partition_contract=area/'WORKER_PARTITIONS.json'
    if partition_contract.exists():
        contract=json.loads(partition_contract.read_text())
        if partitions!=contract['partitions'] or worker_name!=contract['owners'][partition_index]:
            raise ValueError('worker fixed partition differs from current sealed assignment')
    if not 0<=partition_index<partitions: raise ValueError('invalid fixed worker partition')
    worker_name=worker_name or socket.gethostname()
    owns=lambda task:int(task['task_id'],16)%partitions==partition_index
    tasks=[task for task in plan['tasks'] if owns(task)]
    # The same-notebook process flock is already held. A restarted fixed owner
    # may clear only its own abandoned claims; it never steals another owner.
    for task in tasks:
        for phase in ('inventory','recover'):
            claim=area/'atomic_claims'/(phase+'-'+task['task_id'])
            if claim.exists():
                owner=claim/'OWNER.json'
                if owner.exists() and json.loads(owner.read_text()).get('worker')!=worker_name:
                    raise ValueError('fixed partition claim belongs to a different owner')
                shutil.rmtree(claim)
    random.Random(socket.gethostname()+str(os.getpid())).shuffle(tasks)
    completed=0; status_path=area/'workers'/(socket.gethostname()+'-'+str(os.getpid())+'.json')
    heartbeat=Path(heartbeat) if heartbeat else status_path.with_suffix('.heartbeat.json')
    current={'pid':os.getpid(),'host':socket.gethostname(),'workers':workers,'state':'running',
             'partition_index':partition_index,'partitions':partitions,'worker':worker_name}
    stopped=threading.Event()
    def pulse():
        while not stopped.is_set():
            atomic_json(heartbeat,{**current,'updated_at':time.time()}); stopped.wait(30)
    threading.Thread(target=pulse,daemon=True).start()
    threading.Thread(target=build_partition,args=(plan,area,partition_index,partitions,current),daemon=True).start()
    while True:
        binding=binding_plan(plan,area); phase='recover' if binding else 'inventory'
        current['phase']=phase
        selected=[task for task in binding['tasks'] if owns(task)] if binding else tasks
        random.Random(socket.gethostname()+str(os.getpid())).shuffle(selected)
        directory=area/('recovered' if binding else 'inventory')
        todo=[task for task in selected if not (directory/task['task_id']/'COMPLETE.json').exists()]
        if not todo:
            if binding: return {'complete':True,'processed':completed}
            time.sleep(15)
            continue
        if max_tasks is not None: todo=todo[:max_tasks-completed]
        if not todo: return {'complete':False,'processed':completed}
        progressed=0
        with ThreadPoolExecutor(max_workers=workers) as pool:
            for result in pool.map(lambda task:work_one(task,plan,area,phase,worker_name),todo):
                progressed+=result=='processed'; completed+=result=='processed'
                current['processed']=completed
                atomic_json(status_path,{'pid':os.getpid(),'host':socket.gethostname(),'phase':phase,
                    'updated_at':time.time(),'processed':completed,'last_result':result,'workers':workers})
        if max_tasks is not None and completed>=max_tasks: return {'complete':False,'processed':completed}
        if progressed==0: time.sleep(15)


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('command',choices=['prepare','worker','status'])
    parser.add_argument('--run',required=True); parser.add_argument('--root',default='/inspire/dataset/gen-und-mm-pretrain/01')
    parser.add_argument('--workers',type=int,default=8); parser.add_argument('--max-tasks',type=int)
    parser.add_argument('--heartbeat'); parser.add_argument('--process-lock')
    parser.add_argument('--worker-name'); parser.add_argument('--partition-index',type=int,default=0)
    parser.add_argument('--partitions',type=int,default=1)
    args=parser.parse_args(); area=Path(args.run)/'finevision_recovery'
    if args.command=='prepare': result=prepare(args.run,args.root); result={k:v for k,v in result.items() if k!='tasks'}
    elif args.command=='worker':
        process_lock=Path(args.process_lock) if args.process_lock else area/'process_locks'/(socket.gethostname()+'.lock')
        with lock(process_lock) as acquired:
            if not acquired: result={'already_running':True}
            else: result=worker(args.run,args.workers,args.max_tasks,args.heartbeat,args.worker_name,args.partition_index,args.partitions)
    else:
        result={'inventory_complete':len(list((area/'inventory').glob('*/COMPLETE.json'))),
                'recovered_complete':len(list((area/'recovered').glob('*/COMPLETE.json'))),
                'errors':len(list(area.glob('*/*/ERROR.json'))),'binding_plan_ready':(area/'BINDING_PLAN.json').exists()}
    print(json.dumps(result,ensure_ascii=False))


if __name__=='__main__': main()
