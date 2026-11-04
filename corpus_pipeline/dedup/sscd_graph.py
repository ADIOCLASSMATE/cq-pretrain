"""Full SSCD graph audit and advisory, direct-edge representative candidates.

Connected components are leakage/audit groups, NEVER a transitive deletion
rule. Every candidate member has a direct threshold-qualified edge to a kept
candidate representative. Even those groups remain advisory: no image,
caption, QA, interleaved sequence, or source record is deleted by this module.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import ctypes
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time

import numpy as np

from ..postprocess import sscd_index as s
from ..postprocess import sscd_parallel_search as lock_helper
from ..postprocess.sscd_parallel_search import claim

VERSION = 'corpus.sscd_graph.v1'
EDGE_DTYPE = np.dtype([('a','<u4'),('b','<u4'),('score','<f4')])
INVALID = np.uint32(2**32-1)
NATIVE_PATH = Path(__file__).with_name('_sscd_graph.cpp')


def runtime():
    return {'code_sha256': s.file_sha(__file__), 'native_code_sha256': s.file_sha(NATIVE_PATH),
            'numpy': np.__version__, 'helper_sha256': s.file_sha(s.__file__),
            'lock_helper_sha256': s.file_sha(lock_helper.__file__)}


def digest_file(path):
    raw=Path(path).read_bytes(); return json.loads(raw),hashlib.sha256(raw).hexdigest()


def stage_binding(plan, stage, **values):
    return {'plan_sha256': s.digest(plan), 'stage': stage, **values}


def fsync_directory(path):
    fd=os.open(path,os.O_RDONLY)
    try: os.fsync(fd)
    finally: os.close(fd)


def checkpoint(root, stage, **values):
    s.atomic_json(root / 'progress' / (stage+'.json'), {'stage':stage,'updated_at':time.time(),**values})


def reserve(root, minimum=5*2**30):
    s.require(shutil.disk_usage(root).free >= minimum, 'Insufficient disk reserve for graph output')


def prepare(source, output, *, threshold=.75, allow_bounded=False):
    source,root=Path(source).resolve(),Path(output).resolve()
    s.require(source!=root and not root.is_relative_to(source), 'Graph output must be separate from source search')
    complete,complete_sha=digest_file(source/'COMPLETE.json'); search,search_sha=digest_file(source/'PLAN.json')
    s.require(complete.get('complete') is True and complete.get('all_pairs_covered') is True,
              'Input search has not covered every pair')
    s.require(complete['plan_sha256']==search_sha and complete['vectors']==search['vectors'], 'Search completion binding mismatch')
    scope=search['identity']['scope']; s.require(scope=='full_run' or allow_bounded,'Bounded graph needs explicit opt-in')
    s.require(search['identity']['threshold']<=threshold<=1 and np.isfinite(threshold), 'Graph threshold outside retrieved range')
    blocks=[{'start_id':b['start_id'],'count':b['count']} for b in search['blocks']]
    end=0
    for b in blocks:
        s.require(b['start_id']==end and b['count']>0,'Input block coverage gap/overlap');end+=b['count']
    s.require(end==search['vectors'] and 0<end<int(INVALID),'Graph requires contiguous uint32 vector IDs')
    s.require(complete['total_tasks']==len(blocks)*(len(blocks)+1)//2,'Input task count mismatch')
    s.require(complete['total_pairs']==end*(end-1)//2,'Input global pair coverage mismatch')
    plan={'identity':{'schema':VERSION,'runtime':runtime(),'source':str(source),
          'source_complete_sha256':complete_sha,'source_plan_sha256':search_sha,
          'catalog_sha256':search['identity']['catalog_sha256'],'index_root':search['identity']['index_root'],
          'source_threshold':search['identity']['threshold'],'threshold':threshold,'scope':scope,
          'allow_bounded':allow_bounded,'advisory':True,'automatic_drop':False},
          'vectors':end,'blocks':blocks,'source_edges':complete['candidate_edges'],
          'source_task_receipts_sha256':complete['task_receipts_sha256'],'pack_segments':len(blocks)}
    root.mkdir(parents=True,exist_ok=True)
    with claim(root/'.prepare.lock',blocking=True):
        path=root/'PLAN.json'
        if path.exists(): s.require(s.read_json(path)==plan,'Graph source/config/runtime changed; use new output')
        else:
            s.require(not (root/'COMPLETE.json').exists(),'Completion without graph plan')
            s.atomic_json(path,plan)
    return plan


def load_plan(output):
    root=Path(output).resolve();plan=s.read_json(root/'PLAN.json');ident=plan['identity']
    s.require(ident['runtime']==runtime(),'Graph runtime changed')
    source=Path(ident['source'])
    s.require(s.file_sha(source/'COMPLETE.json')==ident['source_complete_sha256'] and
              s.file_sha(source/'PLAN.json')==ident['source_plan_sha256'],'Source search manifests changed')
    return root,plan


def tasks_for_left(plan,left):
    blocks=plan['blocks'];n=len(blocks);a=blocks[left]
    first=left*n-left*(left-1)//2
    for right in range(left,n):
        b=blocks[right]
        yield {'number':first+right-left,'left':left,'right':right,
               'pairs':a['count']*b['count'] if left!=right else a['count']*(a['count']-1)//2}


def segment_path(root,left): return root/'packed'/f'{left:06d}'


def validate_segment(root,plan,left,*,full_hash=True,source_hashes=False):
    path=segment_path(root,left)/'COMPLETE.json';value=s.read_json(path)
    s.require(value['complete'] is True and value['binding']==stage_binding(plan,'pack',left=left),'Packed segment binding changed')
    for item in value['artifacts']:s.bound_path(root,item,verify=full_hash)
    source=Path(plan['identity']['source']);shas=[]
    for task,entry in zip(tasks_for_left(plan,left),value['source_tasks']):
        source_path=source/'tasks'/f'{left:06d}'/f"{task['right']:06d}"/'COMPLETE.json'
        receipt,sha=digest_file(source_path);s.require(sha==entry['receipt_sha256'],'Source task receipt changed')
        s.require(receipt['binding']=={'plan_sha256':plan['identity']['source_plan_sha256'],'task':task},'Source task identity changed')
        for item,stamp in zip(receipt['artifacts'],entry['artifact_stamps']):
            target=s.bound_path(source,item,verify=source_hashes)
            s.require(s.stamp(target)==stamp,'Source edge artifact changed')
        shas.append(sha)
    s.require(len(value['source_tasks'])==len(plan['blocks'])-left,'Incomplete packed source task list')
    return value,shas


def pack_segment(root,plan,left):
    import pyarrow.parquet as pq
    destination=segment_path(root,left);source=Path(plan['identity']['source'])
    with claim(root/'locks'/f'pack-{left:06d}.lock') as acquired:
        if not acquired:return False
        if (destination/'COMPLETE.json').exists():
            validate_segment(root,plan,left);return False
        destination.parent.mkdir(parents=True,exist_ok=True)
        s.require(not destination.exists(),'Uncommitted packed destination exists')
        for stale in destination.parent.glob('.'+destination.name+'.*.tmp'):
            if stale.is_dir():shutil.rmtree(stale)
        reserve(root,0 if plan['identity']['allow_bounded'] else 5*2**30)
        temporary=Path(tempfile.mkdtemp(prefix='.'+destination.name+'.',suffix='.tmp',dir=destination.parent))
        digest=hashlib.sha256();count=source_count=0;source_tasks=[];started=time.monotonic()
        try:
            target=temporary/'edges.bin'
            with target.open('wb',buffering=8<<20) as outgoing:
                for task in tasks_for_left(plan,left):
                    source_receipt=source/'tasks'/f'{left:06d}'/f"{task['right']:06d}"/'COMPLETE.json'
                    receipt,receipt_sha=digest_file(source_receipt)
                    s.require(receipt.get('complete') is True and receipt['pairs_covered']==task['pairs'] and
                              receipt['binding']=={'plan_sha256':plan['identity']['source_plan_sha256'],'task':task},
                              'Invalid complete source task')
                    observed=0;stamps=[];a,b=plan['blocks'][left],plan['blocks'][task['right']]
                    for item in receipt['artifacts']:
                        path=s.bound_path(source,item,verify=True);before=s.stamp(path)
                        for batch in pq.ParquetFile(path).iter_batches(batch_size=262144,
                              columns=['vector_id_a','vector_id_b','exact_cosine'],use_threads=False):
                            aa=batch.column(0).to_numpy(zero_copy_only=False)
                            bb=batch.column(1).to_numpy(zero_copy_only=False)
                            score=batch.column(2).to_numpy(zero_copy_only=False)
                            s.require(((aa>=a['start_id'])&(aa<a['start_id']+a['count'])&
                                       (bb>=b['start_id'])&(bb<b['start_id']+b['count'])&(aa<bb)).all(),
                                      'Edge outside task range or noncanonical/self edge')
                            s.require(np.isfinite(score).all() and (score>=plan['identity']['source_threshold']).all() and
                                      (score<=1.00001).all(),'Invalid original SSCD candidate score')
                            observed+=len(aa);keep=score>=plan['identity']['threshold']
                            packed=np.empty(int(keep.sum()),dtype=EDGE_DTYPE)
                            packed['a'],packed['b'],packed['score']=aa[keep],bb[keep],score[keep]
                            raw=packed.tobytes();outgoing.write(raw);digest.update(raw);count+=len(packed)
                        s.require(s.stamp(path)==before,'Source edges changed during projection');stamps.append(before)
                    s.require(observed==receipt['edges'],'Source edge count differs from complete receipt')
                    source_count+=observed;source_tasks.append({'receipt_sha256':receipt_sha,'artifact_stamps':stamps})
                    if len(source_tasks)%32==0:
                        checkpoint(root,f'pack-{left:06d}',complete=False,source_tasks_done=len(source_tasks),
                                   source_tasks_total=len(plan['blocks'])-left,edges=count,seconds=time.monotonic()-started)
                outgoing.flush();os.fsync(outgoing.fileno())
            artifact={'path':str((destination/'edges.bin').relative_to(root)),'sha256':digest.hexdigest(),'size_bytes':count*12}
            result={'complete':True,'binding':stage_binding(plan,'pack',left=left),'edges':count,'source_edges':source_count,
                    'source_tasks':source_tasks,'artifacts':[artifact],'seconds':time.monotonic()-started}
            s.atomic_json(temporary/'COMPLETE.json',result);os.replace(temporary,destination);fsync_directory(destination.parent)
            checkpoint(root,f'pack-{left:06d}',complete=True,edges=count,seconds=result['seconds'])
        finally:
            if temporary.exists():shutil.rmtree(temporary)
    return True


def pack(output,*,workers=4,max_segments=None):
    root,plan=load_plan(output);s.require(1<=workers<=16,'Worker count outside bounds')
    numbers=list(range(len(plan['blocks'])))
    if max_segments is not None:s.require(max_segments>0,'Invalid segment limit');numbers=numbers[:max_segments]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        changed=list(pool.map(lambda left:pack_segment(root,plan,left),numbers))
    return {'new_segments':sum(changed),**status(root)}


def finalize_pack(root,plan):
    with claim(root/'.finalize-pack.lock',blocking=True):
        count=source_count=0;manifest=hashlib.sha256();segments=[]
        for left in range(len(plan['blocks'])):
            value,shas=validate_segment(root,plan,left)
            for sha in shas:manifest.update(sha.encode())
            count+=value['edges'];source_count+=value['source_edges']
            segments.append({'receipt_path':str((segment_path(root,left)/'COMPLETE.json').relative_to(root)),
                             'receipt_sha256':s.file_sha(segment_path(root,left)/'COMPLETE.json'),
                             'edges':value['edges'],'artifact':value['artifacts'][0]})
        s.require(manifest.hexdigest()==plan['source_task_receipts_sha256'],'Source task aggregate SHA mismatch')
        s.require(source_count==plan['source_edges'],'Not all retrieved edges were consumed')
        value={'complete':True,'binding':stage_binding(plan,'packed'),'edges':count,'source_edges':source_count,
               'source_task_receipts_sha256':manifest.hexdigest(),'segments':segments}
        path=root/'packed/COMPLETE.json'
        if path.exists():s.require(s.read_json(path)==value,'Packed completion changed')
        else:s.atomic_json(path,value)
        return value


def load_packed(root,plan):
    path=root/'packed/COMPLETE.json';value=s.read_json(path)
    s.require(value['complete'] is True and value['binding']==stage_binding(plan,'packed'),'Packed graph binding changed')
    s.require(len(value['segments'])==len(plan['blocks']),'Missing packed segments')
    for entry in value['segments']:
        s.require(s.file_sha(root/entry['receipt_path'])==entry['receipt_sha256'],'Packed segment receipt changed')
        s.bound_path(root,entry['artifact'])
    return value


class Native:
    def __init__(self,root):
        directory=root/'native';directory.mkdir(parents=True,exist_ok=True)
        target=directory/(s.file_sha(NATIVE_PATH)+'.so')
        with claim(directory/'.build.lock',blocking=True):
            if not target.exists():
                temporary=target.with_suffix('.tmp.so')
                command=['c++','-O3','-std=c++17','-fPIC','-shared',str(NATIVE_PATH),'-o',str(temporary)]
                subprocess.run(command,check=True,capture_output=True,text=True);os.replace(temporary,target)
                s.atomic_json(target.with_suffix('.json'),{'source_sha256':s.file_sha(NATIVE_PATH),
                              'binary_sha256':s.file_sha(target),'command':command})
            meta=s.read_json(target.with_suffix('.json'))
            s.require(meta['source_sha256']==s.file_sha(NATIVE_PATH) and meta['binary_sha256']==s.file_sha(target),
                      'Native graph library changed')
        self.lib=ctypes.CDLL(str(target))

    def call(self,name,*values):
        args=[]
        for value in values:
            if isinstance(value,np.ndarray):
                s.require(value.flags.c_contiguous,'Native array must be contiguous');args.append(ctypes.c_void_p(value.ctypes.data))
            elif isinstance(value,(int,np.integer)):args.append(ctypes.c_uint64(int(value)))
            elif isinstance(value,(float,np.floating)):args.append(ctypes.c_float(float(value)))
            else:raise TypeError(type(value))
        result=getattr(self.lib,name)(*args);s.require(result==0,f'Native graph invariant {name} failed: {result}')


def mapped(path,dtype,count,mode='w+'):
    if count==0:
        if mode=='w+':Path(path).write_bytes(b'')
        return np.empty(0,dtype=dtype)
    return np.memmap(path,dtype=dtype,mode=mode,shape=(count,))


def packed_arrays(root,packed):
    for entry in packed['segments']:
        if entry['edges']:
            yield np.memmap(root/entry['artifact']['path'],dtype=EDGE_DTYPE,mode='r',shape=(entry['edges'],))


def phase_existing(root,name,bound):
    directory=root/name
    if (directory/'COMPLETE.json').exists():
        value=s.read_json(directory/'COMPLETE.json')
        s.require(value['complete'] is True and value['binding']==bound,'Graph stage binding changed')
        for item in value['artifacts']:s.bound_path(root,item)
        return value
    return None


def stage_temporary(root,name):
    s.require(not (root/name).exists(),'Uncommitted graph stage destination exists')
    for stale in root.glob('.'+name+'.*.tmp'):
        if stale.is_dir():shutil.rmtree(stale)
    return Path(tempfile.mkdtemp(prefix='.'+name+'.',suffix='.tmp',dir=root))


def publish_phase(root,name,temporary,bound,fields):
    arts=[]
    for path in sorted(temporary.iterdir()):
        if path.is_file():
            with path.open('rb') as stream:os.fsync(stream.fileno())
            item=s.artifact(path,temporary);item['path']=str(Path(name)/path.name);arts.append(item)
    result={'complete':True,'binding':bound,'artifacts':arts,**fields}
    s.atomic_json(temporary/'COMPLETE.json',result);os.replace(temporary,root/name);fsync_directory(root)
    return result


def structure(output):
    root,plan=load_plan(output);native=Native(root);n=plan['vectors']
    with claim(root/'.structure.lock',blocking=True):
        packed=finalize_pack(root,plan);packed_sha=s.file_sha(root/'packed/COMPLETE.json')
        bound=stage_binding(plan,'components',packed_sha256=packed_sha)
        counts=phase_existing(root,'components',bound)
        if counts is None:
            temporary=stage_temporary(root,'components');started=time.monotonic()
            try:
                parent=mapped(temporary/'parent.work','<u4',n);sizes=mapped(temporary/'sizes.work','<u4',n)
                degree=mapped(temporary/'degree.work','<u8',n)
                native.call('graph_initialize',n,parent,sizes,degree)
                seen=0
                for number,edges in enumerate(packed_arrays(root,packed)):
                    native.call('graph_count',n,len(edges),edges,parent,sizes,degree,float(plan['identity']['threshold']))
                    seen+=len(edges);checkpoint(root,'components',edges_processed=seen,edges_total=packed['edges'])
                component=mapped(temporary/'component_id.u4','<u4',n)
                component_size=mapped(temporary/'component_size.u4','<u4',n)
                offsets=mapped(temporary/'offsets.u8','<u8',n+1);summary=np.zeros(6,np.uint64);hist=np.zeros(32,np.uint64)
                native.call('graph_components',n,parent,sizes,degree,component,component_size,offsets,summary,hist)
                s.require(int(offsets[-1])==2*packed['edges'] and seen==packed['edges'],'Degree totals differ from packed graph')
                for array in (component,component_size,offsets):array.flush()
                del parent,sizes,degree,component,component_size,offsets
                for name in ('parent.work','sizes.work','degree.work'):(temporary/name).unlink()
                fields=dict(zip(('components','nontrivial_components','largest_component','vertices_with_edges','isolated_vertices','max_degree'),map(int,summary)))
                fields.update(edges=seen,vectors=n,component_size_log2_histogram=list(map(int,hist)),seconds=time.monotonic()-started,
                              connected_components_are_audit_only=True,advisory=True)
                counts=publish_phase(root,'components',temporary,bound,fields)
            finally:
                if temporary.exists():shutil.rmtree(temporary)
        bound=stage_binding(plan,'adjacency',packed_sha256=packed_sha,components_sha256=s.file_sha(root/'components/COMPLETE.json'))
        adjacency=phase_existing(root,'adjacency',bound)
        if adjacency is None:
            temporary=stage_temporary(root,'adjacency');started=time.monotonic()
            try:
                offsets=mapped(root/'components/offsets.u8','<u8',n+1,'r')
                # Random first writes to new shared-filesystem mmap pages can
                # trigger one allocation RPC per page. Build in ordinary RAM,
                # then write sequentially. At 755M edges this uses ~13GB RAM.
                cursor=np.array(offsets[:-1],copy=True)
                neighbors=np.empty(2*packed['edges'],dtype='<u4')
                scores=np.empty(2*packed['edges'],dtype='<f4');seen=0
                for edges in packed_arrays(root,packed):
                    native.call('graph_fill',n,len(edges),edges,offsets,cursor,neighbors,scores)
                    seen+=len(edges);checkpoint(root,'adjacency',edges_processed=seen,edges_total=packed['edges'],
                                               memory_mode='ram_then_sequential_write')
                s.require(np.array_equal(cursor,offsets[1:]),'Adjacency cursor coverage mismatch')
                fill_seconds=time.monotonic()-started
                neighbors.tofile(temporary/'neighbors.u4');scores.tofile(temporary/'scores.f4')
                del cursor,neighbors,scores,offsets
                adjacency=publish_phase(root,'adjacency',temporary,bound,{'edges':seen,'directed_entries':2*seen,
                                        'vectors':n,'seconds':time.monotonic()-started,'fill_seconds':fill_seconds,
                                        'memory_mode':'ram_then_sequential_write','full_cursor_coverage_verified':True,'advisory':True})
            finally:
                if temporary.exists():shutil.rmtree(temporary)
    return {'components':{k:v for k,v in counts.items() if k not in ('artifacts','binding')},
            'adjacency':{k:v for k,v in adjacency.items() if k not in ('artifacts','binding')}}


def policy_array(path,dtype,n,default):
    if path is None:return np.full(n,default,dtype=dtype),None
    target=Path(path).resolve();s.require(target.stat().st_size==n*np.dtype(dtype).itemsize,'Policy array size mismatch')
    stamp=s.stamp(target);sha=s.file_sha(target)
    s.require(s.stamp(target)==stamp,'Policy changed during verification')
    return np.memmap(target,dtype=dtype,mode='r',shape=(n,)),{'path':str(target),'sha256':sha,'stamp':stamp,'dtype':dtype}


def representatives(output,*,eligible=None,protected=None,priority=None):
    import pyarrow as pa
    import pyarrow.parquet as pq
    root,plan=load_plan(output);n=plan['vectors'];native=Native(root)
    with claim(root/'.representatives.lock',blocking=True):
        counts=s.read_json(root/'components/COMPLETE.json');adj=s.read_json(root/'adjacency/COMPLETE.json')
        packed_sha=s.file_sha(root/'packed/COMPLETE.json')
        s.require(counts['binding']==stage_binding(plan,'components',packed_sha256=packed_sha),
                  'Component graph binding changed')
        s.require(adj['binding']==stage_binding(plan,'adjacency',packed_sha256=packed_sha,
                  components_sha256=s.file_sha(root/'components/COMPLETE.json')),'Adjacency graph binding changed')
        for value in (counts,adj):
            for item in value['artifacts']:s.bound_path(root,item)
        allowed,allow_id=policy_array(eligible,'u1',n,1)
        protect,protect_id=policy_array(protected,'u1',n,0)
        s.require(((allowed==0)|(allowed==1)).all() and ((protect==0)|(protect==1)).all(),'Policies must be binary uint8')
        rank,rank_id=policy_array(priority,'<u8',n,0)
        policies={'eligible':allow_id,'protected':protect_id,'priority':rank_id}
        bound=stage_binding(plan,'representatives',components_sha256=s.file_sha(root/'components/COMPLETE.json'),
                            adjacency_sha256=s.file_sha(root/'adjacency/COMPLETE.json'),policies=policies,
                            ordering='priority_desc_vector_id_asc',protected_semantics='retain_self_do_not_suppress_others')
        existing=phase_existing(root,'representatives',bound)
        if existing:return existing
        temporary=stage_temporary(root,'representatives');started=time.monotonic()
        try:
            # Traversal and representative assignments are random accesses.
            # Ordinary RAM avoids shared-filesystem mmap write amplification;
            # immutable inputs have already been verified above.
            offsets=np.fromfile(root/'components/offsets.u8',dtype='<u8')
            neighbors=np.fromfile(root/'adjacency/neighbors.u4',dtype='<u4')
            scores=np.fromfile(root/'adjacency/scores.f4',dtype='<f4')
            order=np.arange(n,dtype=np.uint32) if priority is None else np.argsort(np.bitwise_xor(rank,np.uint64(2**64-1)),kind='stable').astype(np.uint32)
            rep=np.empty(n,dtype='<u4')
            direct=np.empty(n,dtype='<f4')
            groups=np.empty(n,dtype='<u4');summary=np.zeros(5,np.uint64)
            native.call('graph_representatives',n,offsets,neighbors,scores,order,allowed,protect,rep,direct,groups,summary,float(plan['identity']['threshold']))
            valid=rep!=INVALID
            s.require(int(valid.sum())==int(summary[0]),'Representative eligible coverage mismatch')
            s.require(np.array_equal(rep[rep[valid]],rep[valid]),'Representative roots must retain themselves')
            s.require((direct[valid]>=plan['identity']['threshold']).all(),'Representative score below threshold')
            compute_seconds=time.monotonic()-started
            rep.tofile(temporary/'candidate_rep_vector_id.u4')
            direct.tofile(temporary/'direct_score.f4')
            groups.tofile(temporary/'candidate_group_size.u4')
            del offsets,neighbors,scores,order,valid
            comp=mapped(root/'components/component_id.u4','<u4',n,'r');sizes=mapped(root/'components/component_size.u4','<u4',n,'r')
            parquet=temporary/'membership.parquet'
            schema=pa.schema([('vector_id',pa.uint32()),('component_id',pa.uint32()),('component_size',pa.uint32()),
                   ('candidate_rep_vector_id',pa.uint32()),('direct_score',pa.float32()),
                   ('is_candidate_representative',pa.bool_()),('eligible',pa.bool_()),('protected',pa.bool_())])
            with pq.ParquetWriter(parquet,schema,compression='zstd') as writer:
                for start in range(0,n,1048576):
                    end=min(start+1048576,n);ids=np.arange(start,end,dtype=np.uint32);invalid=rep[start:end]==INVALID
                    arrays=[pa.array(ids),pa.array(comp[start:end]),pa.array(sizes[start:end]),
                            pa.array(rep[start:end],mask=invalid),pa.array(direct[start:end],mask=invalid),
                            pa.array(rep[start:end]==ids),pa.array(allowed[start:end].astype(bool)),pa.array(protect[start:end].astype(bool))]
                    writer.write_table(pa.Table.from_arrays(arrays,schema=schema))
                    checkpoint(root,'representatives',vertices_exported=end,vertices_total=n)
            for item in policies.values():
                if item:s.require(s.stamp(item['path'])==item['stamp'],'Policy changed during representative selection')
            fields=dict(zip(('eligible_vertices','candidate_representatives','candidate_members','excluded_vertices','protected_vertices'),map(int,summary)))
            fields.update(vectors=n,seconds=time.monotonic()-started,compute_seconds=compute_seconds,
                          memory_mode='ram_then_sequential_write',full_representative_root_invariant_verified=True,
                          advisory=True,provisional=True,
                          automatic_drop=False,supervision_records_dropped=0,retained_eligible_assets=int(summary[0]),
                          direct_edge_invariant=True,connected_components_are_audit_only=True,
                          self_representative_score_is_conventional_identity=True,
                          policy_note='Every near-duplicate group is provisional; all eligible images and distinct supervision remain available.')
            del rep,direct,groups
            return publish_phase(root,'representatives',temporary,bound,fields)
        finally:
            if temporary.exists():shutil.rmtree(temporary)


def finalize(output,*,verify_source_hashes=True):
    root,plan=load_plan(output)
    with claim(root/'.finalize.lock',blocking=True):
        packed=finalize_pack(root,plan)
        if verify_source_hashes:
            for left in range(len(plan['blocks'])):validate_segment(root,plan,left,source_hashes=True)
        values={}
        for name in ('components','adjacency','representatives'):
            value=s.read_json(root/name/'COMPLETE.json');s.require(value['complete'] is True,'Incomplete graph phase')
            s.require(value['binding']['plan_sha256']==s.digest(plan),'Graph phase plan mismatch')
            for item in value['artifacts']:s.bound_path(root,item)
            values[name]=value
        reps=values['representatives']
        packed_sha=s.file_sha(root/'packed/COMPLETE.json')
        s.require(values['components']['binding']==stage_binding(plan,'components',packed_sha256=packed_sha),
                  'Final component graph binding changed')
        s.require(values['adjacency']['binding']==stage_binding(plan,'adjacency',packed_sha256=packed_sha,
                  components_sha256=s.file_sha(root/'components/COMPLETE.json')),'Final adjacency graph binding changed')
        for item in reps['binding']['policies'].values():
            if item:s.require(s.file_sha(item['path'])==item['sha256'],'Representative policy content changed')
        s.require(reps['binding']['components_sha256']==s.file_sha(root/'components/COMPLETE.json') and
                  reps['binding']['adjacency_sha256']==s.file_sha(root/'adjacency/COMPLETE.json'),'Representative upstream graph changed')
        result={'schema':VERSION,'complete':True,'advisory':True,'provisional':True,'automatic_drop':False,
                'near_duplicates_resolved':False,'supervision_records_dropped':0,'source_records_modified':False,
                'vectors':plan['vectors'],'edges':packed['edges'],'source_edges':plan['source_edges'],
                'components':values['components']['components'],'largest_component':values['components']['largest_component'],
                'candidate_representatives':reps['candidate_representatives'],'candidate_members':reps['candidate_members'],
                'retained_eligible_assets':reps['retained_eligible_assets'],'direct_edge_invariant':True,
                'plan_sha256':s.file_sha(root/'PLAN.json'),
                'phases':{name:s.file_sha(root/name/'COMPLETE.json') for name in values},
                'source_artifact_hashes_reverified':verify_source_hashes}
        path=root/'COMPLETE.json'
        if path.exists():s.require(s.read_json(path)==result,'Final graph completion changed')
        else:s.atomic_json(path,result)
        return result


def status(output):
    root,plan=load_plan(output);segments=edges=0
    for left in range(len(plan['blocks'])):
        path=segment_path(root,left)/'COMPLETE.json'
        if path.exists():segments+=1;edges+=s.read_json(path)['edges']
    result={'vectors':plan['vectors'],'packed_segments':segments,'total_segments':len(plan['blocks']),
            'packed_edges':edges,'source_edges':plan['source_edges'],'updated_at':time.time(),
            'complete':(root/'COMPLETE.json').exists(),'observational_only':True}
    for name in ('components','adjacency','representatives'):
        path=root/name/'COMPLETE.json';result[name+'_complete']=path.exists()
        if path.exists():result[name]={k:v for k,v in s.read_json(path).items() if k not in ('artifacts','binding')}
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('stage',choices=('prepare','pack','structure','representatives','finalize','status'))
    parser.add_argument('--input');parser.add_argument('--output',required=True);parser.add_argument('--threshold',type=float,default=.75)
    parser.add_argument('--allow-bounded',action='store_true');parser.add_argument('--workers',type=int,default=4)
    parser.add_argument('--max-segments',type=int);parser.add_argument('--eligible');parser.add_argument('--protected');parser.add_argument('--priority')
    args=parser.parse_args()
    if args.stage=='prepare':
        s.require(args.input,'prepare requires --input');result=prepare(args.input,args.output,threshold=args.threshold,allow_bounded=args.allow_bounded)
        result={k:v for k,v in result.items() if k!='blocks'}
    elif args.stage=='pack':result=pack(args.output,workers=args.workers,max_segments=args.max_segments)
    elif args.stage=='structure':result=structure(args.output)
    elif args.stage=='representatives':result=representatives(args.output,eligible=args.eligible,protected=args.protected,priority=args.priority)
    elif args.stage=='finalize':result=finalize(args.output)
    else:result=status(args.output)
    print(s.canonical(result),flush=True)


if __name__=='__main__':main()
