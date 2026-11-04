"""Bind graph representative preferences to existing asset screening evidence.

This controls advisory representatives only. It never certifies a SSCD pair as
equivalent, changes encoded-image references, or propagates a screening label.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import duckdb
import numpy as np
from .asset_release import artifact, verify, require
from corpus_pipeline.postprocess.reconcile import atomic_json, quoted, file_sha


def build(index, assets_manifest, output, threads=4):
    index, assets_manifest, output=map(lambda p:Path(p).absolute(),(index,assets_manifest,output))
    output.mkdir(parents=True,exist_ok=True)
    catalog_path=index/'catalog/COMPLETE.json'
    catalog=json.loads(catalog_path.read_text()); assets=json.loads(assets_manifest.read_text())
    require(assets.get('complete') and assets.get('canonical_equivalence_only'), 'Unqualified asset map')
    identity={'schema':'corpus.sscd_representative_policy.v1',
              'catalog':artifact(catalog_path),'asset_map':artifact(assets_manifest),
              'code_sha256':file_sha(__file__),'advisory_only':True}
    identity_path=output/'IDENTITY.json'
    if identity_path.exists():require(json.loads(identity_path.read_text())==identity,'Policy identity changed')
    else:atomic_json(identity_path,identity)
    if (output/'COMPLETE.json').exists():
        result=json.loads((output/'COMPLETE.json').read_text())
        for row in result['files']:verify(row)
        return result
    mappings=[];expected=0
    for block in catalog['blocks']:
        r=index/block['receipt_path'];require(file_sha(r)==block['receipt_sha256'],'Catalog receipt changed')
        b=json.loads(r.read_text())
        item=next(i for i in b['artifacts'] if Path(i['path']).name=='mapping.parquet')
        verify({**item,'path':str(index/item['path'])});mappings.append(str(index/item['path']))
        expected+=block['count']
    for item in assets['files']:verify(item)
    db=duckdb.connect();db.execute(f'SET threads={threads}');db.execute("SET memory_limit='40GB'")
    db.execute('SET temp_directory='+quoted(output/'spill'))
    paths='['+','.join(quoted(p) for p in mappings)+']'
    afiles='['+','.join(quoted(p['path']) for p in assets['files'])+']'
    atomic_json(output/'PROGRESS.json',{'phase':'join_policy','updated_at':time.time(),'vectors':expected})
    arrays={name:np.memmap(output/name,mode='w+',dtype=dtype,shape=(expected,))
            for name,dtype in [('eligible.u1','u1'),('protected.u1','u1'),('priority.u8','<u8')]}
    for a in arrays.values():a[:]=0
    seen=np.zeros(expected,dtype=bool);total=0;excluded=0
    query='''SELECT m.vector_id,a.asset_id,a.screening_action,a.global_status,
               a.original_width,a.original_height FROM read_parquet('''+paths+''') m
               LEFT JOIN read_parquet('''+afiles+''',hive_partitioning=false) a USING(asset_id)'''
    for batch in db.execute(query).fetch_record_batch(262144):
        ids=batch.column(0).to_numpy(zero_copy_only=False).astype(np.int64,copy=False)
        require(len(ids)==len(np.unique(ids)) and np.all((ids>=0)&(ids<expected))
                and not seen[ids].any(),'Duplicate/out-of-range vector ID')
        require(batch.column(1).null_count==0,'Missing asset policy')
        action=np.array(batch.column(2).to_pylist());status=np.array(batch.column(3).to_pylist())
        eligible=(action!='exclude') & (status!='pending_integrity')
        w=batch.column(4).to_numpy(zero_copy_only=False).astype(np.uint64)
        h=batch.column(5).to_numpy(zero_copy_only=False).astype(np.uint64)
        # Screening rank first, then pixel area. Identifier breaks ties in graph.
        rank=np.where(action=='retain_annotated',3,np.where(eligible,1,0)).astype(np.uint64)
        priority=(rank<<np.uint64(60))+np.minimum(w*h,np.uint64((1<<60)-1))
        arrays['eligible.u1'][ids]=eligible;arrays['priority.u8'][ids]=priority
        seen[ids]=True;total+=len(ids);excluded+=int((~eligible).sum())
    require(total==expected and seen.all(),'Vector policy coverage mismatch')
    for a in arrays.values():a.flush()
    result={'complete':True,'schema':identity['schema'],'identity_sha256':file_sha(identity_path),
            'vectors':expected,'ineligible_representatives':excluded,'advisory_only':True,
            'files':[artifact(output/name) for name in arrays]}
    atomic_json(output/'COMPLETE.json',result);db.close()
    return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--index',required=True)
    p.add_argument('--asset-manifest',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=4);a=p.parse_args()
    print(json.dumps(build(a.index,a.asset_manifest,a.output,a.threads)))

if __name__=='__main__':main()
