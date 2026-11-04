"""Expose verified SSCD advisory membership by original image SHA, not vector ID."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import duckdb
from .asset_release import artifact, verify, require
from corpus_pipeline.postprocess.reconcile import file_sha, atomic_json, quoted


def build(assets_manifest,index,graph,output,threads=4,memory='48GB'):
    assets_manifest,index,graph,output=map(lambda p:Path(p).absolute(),(assets_manifest,index,graph,output))
    output.mkdir(parents=True,exist_ok=True)
    assets=json.loads(assets_manifest.read_text())
    completed=json.loads((graph/'COMPLETE.json').read_text())
    require(assets.get('complete') and completed.get('complete'),'Upstream incomplete')
    require(assets.get('canonical_equivalence_only') is True,'Asset canonical mapping must contain exact equivalences only')
    require(completed.get('advisory') and not completed.get('automatic_drop'),'Unexpected SSCD semantics')
    catalog_path=index/'catalog/COMPLETE.json';catalog=json.loads(catalog_path.read_text())
    graph_plan_path=graph/'PLAN.json'
    require(file_sha(graph_plan_path)==completed['plan_sha256'],'Graph plan changed')
    graph_plan=json.loads(graph_plan_path.read_text())
    require(graph_plan['identity']['catalog_sha256']==file_sha(catalog_path),
            'Graph and asset vector catalogs differ')
    identity={'schema':'corpus.asset_similarity_index.v1','assets':artifact(assets_manifest),
              'graph':artifact(graph/'COMPLETE.json'),'catalog':artifact(catalog_path),
              'code_sha256':file_sha(__file__),'advisory_only':True}
    identity_path=output/'IDENTITY.json'
    if identity_path.exists():require(json.loads(identity_path.read_text())==identity,'Identity changed')
    else:atomic_json(identity_path,identity)
    if (output/'COMPLETE.json').exists():
        old=json.loads((output/'COMPLETE.json').read_text())
        for item in old['files']:verify(item)
        return old
    mappings=[]
    for block in catalog['blocks']:
        receipt=index/block['receipt_path'];require(file_sha(receipt)==block['receipt_sha256'],'Catalog changed')
        value=json.loads(receipt.read_text())
        item=next(i for i in value['artifacts'] if Path(i['path']).name=='mapping.parquet')
        verify({**item,'path':str(index/item['path'])});mappings.append(index/item['path'])
    for item in assets['files']:verify(item)
    rep_receipt=graph/'representatives/COMPLETE.json'
    require(file_sha(rep_receipt)==completed['phases']['representatives'],'Representatives changed')
    rep=json.loads(rep_receipt.read_text())
    item=next(i for i in rep['artifacts'] if Path(i['path']).name=='membership.parquet')
    membership=graph/item['path'];verify({**item,'path':str(membership)})
    db=duckdb.connect();db.execute(f'SET threads={threads}');db.execute('SET memory_limit='+quoted(memory))
    db.execute('SET temp_directory='+quoted(output/'spill'));db.execute('SET preserve_insertion_order=false')
    db.execute('SET partitioned_write_max_open_files=32')
    def view(name,files):
        db.execute('CREATE TEMP VIEW '+name+' AS SELECT * FROM read_parquet(['+
                   ','.join(quoted(f) for f in files)+'],hive_partitioning=false)')
    view('a',[i['path'] for i in assets['files']]);view('m',mappings);view('g',[membership])
    require(db.execute('SELECT count(*)=count(DISTINCT vector_id) AND count(*)=? FROM m',
                       [completed['vectors']]).fetchone()[0],'Vector mapping incomplete')
    dest=output/'assets'
    if dest.exists():
        import shutil
        shutil.rmtree(dest)
    atomic_json(output/'PROGRESS.json',{'phase':'joining_asset_similarity','updated_at':time.time()})
    db.execute('''COPY (SELECT a.*,m.vector_id,g.component_id AS sscd_component_id,
        g.component_size AS sscd_component_size,r.asset_id AS sscd_candidate_rep_asset_id,
        g.direct_score AS sscd_direct_score,g.is_candidate_representative,g.eligible AS sscd_rep_eligible,
        true AS sscd_candidate_only,false AS sscd_automatic_merge,substr(a.asset_id,1,2) AS bucket
        FROM a LEFT JOIN m USING(asset_id) LEFT JOIN g USING(vector_id)
        LEFT JOIN m r ON r.vector_id=g.candidate_rep_vector_id) TO '''+quoted(dest)+
        ' (FORMAT PARQUET,PARTITION_BY(bucket),COMPRESSION ZSTD,ROW_GROUP_SIZE 65536)')
    paths=sorted(dest.glob('bucket=*/*.parquet'));view('result',paths)
    rows,unique,covered,invalid=db.execute('''SELECT count(*),count(DISTINCT asset_id),
        count(vector_id),count(*) FILTER(WHERE vector_id IS NOT NULL AND sscd_component_id IS NULL)
        FROM result''').fetchone()
    require(rows==unique==assets['counts']['assets'] and covered==completed['vectors'] and invalid==0,
            'Combined asset/SSCD index lost or multiplied rows')
    result={'complete':True,'schema':identity['schema'],'identity_sha256':file_sha(identity_path),
            'assets':rows,'sscd_vectors':covered,'canonical_equivalence_only':True,
            'sscd_candidate_only':True,'files':[artifact(p) for p in paths],
            'screening_propagates_across_similarity':False}
    atomic_json(output/'COMPLETE.json',result);db.close();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('--asset-manifest',required=True);p.add_argument('--index',required=True)
    p.add_argument('--graph',required=True);p.add_argument('--output',required=True)
    p.add_argument('--threads',type=int,default=4);p.add_argument('--memory',default='48GB')
    a=p.parse_args();r=build(a.asset_manifest,a.index,a.graph,a.output,a.threads,a.memory)
    print(json.dumps({k:v for k,v in r.items() if k!='files'}))

if __name__=='__main__':main()
