"""Real-data readback and identifier checks for a candidate release bundle."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import time
import duckdb
from .asset_release import artifact, verify, require
from .candidate_reader import CandidateReader
from .source_view import iter_native_batches
from corpus_pipeline.postprocess.reconcile import atomic_json, file_sha, quoted


def connect(root):
    con=duckdb.connect();con.execute('SET threads=2');con.execute("SET memory_limit='32GB'")
    con.execute('SET temp_directory='+quoted(root/'audits/spill'))
    return con


def parquet(files):
    return 'read_parquet(['+','.join(quoted(f) for f in files)+'],hive_partitioning=false)'


def reader_audit(root):
    root=Path(root).absolute();started=time.time();index=root/'training_candidate_index'
    manifest=json.loads((index/'manifest.json').read_text())
    complete=json.loads((index/'COMPLETE.json').read_text())
    require(complete['manifest_sha256']==file_sha(index/'manifest.json'),'Candidate manifest changed')
    expected={(r['source'],r['category'],r['decision']) for r in manifest['source_category_counts'] if r['records']}
    found={};con=connect(root)
    fields='record_id,source,category,decision,original_parquet_path,row_index,sample_id,source_record_sha256,text_hash,native_format,image_ids'
    for bucket in manifest['buckets']:
        for item in bucket['artifacts']:
            rows=con.execute('SELECT '+fields+' FROM '+parquet([index/item['path']])+'''
                QUALIFY row_number() OVER(PARTITION BY source,category,decision ORDER BY record_id)=1''').fetch_arrow_table().to_pylist()
            for r in rows:found.setdefault((r['source'],r['category'],r['decision']),r)
        if set(found)==expected:break
    require(set(found)==expected,'Reader sample does not cover all source/category/decision strata')
    samples=sorted(found.values(),key=lambda r:(r['original_parquet_path'],r['row_index']))
    readback=[]
    with CandidateReader(index) as reader:
        for ref in samples:
            raw=reader.read(ref,allow_excluded=True)
            readback.append({k:ref[k] for k in ['record_id','source','category','decision','sample_id']})
        if samples:
            bad={**samples[0],'source_record_sha256':'0'*64}
            try:reader.read(bad,allow_excluded=True)
            except ValueError:pass
            else:raise ValueError('Reader accepted a changed source-record identity')
    view=root/'training_candidate_source_view';vm=json.loads((view/'manifest.json').read_text())
    vc=json.loads((view/'COMPLETE.json').read_text())
    require(vc['manifest_sha256']==file_sha(view/'manifest.json') and
            vm['input_candidate_manifest_sha256']==complete['manifest_sha256'],'Source view binding differs')
    chosen=[];seen_sources=set()
    for i,item in enumerate(vm['files']):
        # Select a few complete native files, preserving their original format.
        if item['counts']['keep']>0:
            chosen.append((i,item))
        if len(chosen)==4:break
    batch_rows=0
    for i,item in chosen:
        count=sum(b.num_rows for b in iter_native_batches(view,batch_size=2048,worker_id=i,num_workers=len(vm['files'])))
        require(count==item['counts']['keep'],'Production reader selection differs')
        batch_rows+=count
    result={'passed':True,'schema':'corpus.dedup.real_reader_audit.v1','elapsed_seconds':time.time()-started,
            'input_manifest':artifact(index/'manifest.json'),'source_view_manifest':artifact(view/'manifest.json'),
            'strata':len(expected),'exact_original_records_rehashed':len(samples),'samples':readback,
            'production_native_files_read':len(chosen),'production_native_rows_read':batch_rows,
            'tampered_reference_rejected':True,'images_decoded':0,
            'scope':'All source/category/decision strata sampled; four complete source files read using production batch loader.'}
    atomic_json(root/'audits/reader.json',result);con.close();return result


def combined_audit(root):
    root=Path(root).absolute();started=time.time();out=root/'asset_similarity_index'
    complete=json.loads((out/'COMPLETE.json').read_text())
    identity=json.loads((out/'IDENTITY.json').read_text())
    require(complete.get('complete') is True and identity.get('advisory_only') is True,
            'Combined index incomplete or not advisory')
    require(file_sha(out/'IDENTITY.json')==complete['identity_sha256'],'Combined index identity differs')
    verify(identity['graph']);verify(identity['assets'])
    require(Path(identity['graph']['path'])==root/'graph/COMPLETE.json' and
            Path(identity['assets']['path'])==root/'assets/COMPLETE.json','Combined input paths differ')
    for item in complete['files']:verify(item)
    catalog_path=Path(identity['catalog']['path']);verify(identity['catalog'])
    catalog=json.loads(catalog_path.read_text());index=catalog_path.parent.parent
    mappings=[]
    for block in catalog['blocks']:
        path=index/block['receipt_path'];require(file_sha(path)==block['receipt_sha256'],'Catalog receipt differs')
        receipt=json.loads(path.read_text())
        item=next(i for i in receipt['artifacts'] if Path(i['path']).name=='mapping.parquet')
        verify({**item,'path':str(index/item['path'])})
        mappings.append(index/item['path'])
    graph=root/'graph';gc=json.loads((graph/'COMPLETE.json').read_text())
    require(gc.get('complete') is True and gc.get('advisory') is True and
            gc.get('automatic_drop') is False and gc.get('source_artifact_hashes_reverified') is True,
            'Graph incomplete or not verified advisory output')
    require(gc['plan_sha256']==file_sha(graph/'PLAN.json'),'Graph plan differs')
    plan=json.loads((graph/'PLAN.json').read_text())
    require(plan['identity']['catalog_sha256']==file_sha(catalog_path),'Graph catalog differs')
    rep_path=graph/'representatives/COMPLETE.json'
    require(file_sha(rep_path)==gc['phases']['representatives'],'Graph representative receipt differs')
    rep=json.loads(rep_path.read_text())
    member_item=next(i for i in rep['artifacts'] if Path(i['path']).name=='membership.parquet')
    membership=graph/member_item['path'];verify({**member_item,'path':str(membership)})
    con=connect(root)
    for name,paths in [('a',[x['path'] for x in complete['files']]),('m',mappings),
                       ('g',[membership])]:
        con.execute('CREATE TEMP VIEW '+name+' AS SELECT * FROM '+parquet(paths))
    counts=con.execute('''SELECT count(*),count(DISTINCT asset_id),count(vector_id),
        count(*) FILTER(WHERE NOT sscd_candidate_only OR sscd_automatic_merge OR
            (screening_action='exclude' AND sscd_rep_eligible)) FROM a''').fetchone()
    require(counts[:3]==(complete['assets'],complete['assets'],gc['vectors']) and counts[3]==0,
            'Combined coverage or advisory/eligibility differs')
    mismatches=con.execute('''SELECT count(*) FROM a
        LEFT JOIN m ON m.asset_id=a.asset_id LEFT JOIN g ON g.vector_id=m.vector_id
        LEFT JOIN m r ON r.asset_id=a.sscd_candidate_rep_asset_id
        WHERE a.vector_id IS DISTINCT FROM m.vector_id
           OR a.sscd_component_id IS DISTINCT FROM g.component_id
           OR a.sscd_component_size IS DISTINCT FROM g.component_size
           OR a.sscd_direct_score IS DISTINCT FROM g.direct_score
           OR a.sscd_rep_eligible IS DISTINCT FROM g.eligible
           OR r.vector_id IS DISTINCT FROM g.candidate_rep_vector_id''').fetchone()[0]
    require(mismatches==0,'Combined asset/graph identity mapping mismatch')
    result={'passed':True,'schema':'corpus.dedup.combined_index_audit.v1','elapsed_seconds':time.time()-started,
            'input':artifact(out/'COMPLETE.json'),'graph':artifact(graph/'COMPLETE.json'),
            'assets':counts[0],'vectors':counts[2],'mismatches':mismatches,
            'all_output_hashes_verified':True,'all_mapping_hashes_verified':True,
            'graph_membership_hash_verified':True,'advisory_not_equivalence':True}
    atomic_json(root/'audits/combined_index.json',result);con.close();return result


def main():
    p=argparse.ArgumentParser();p.add_argument('stage',choices=['reader','combined']);p.add_argument('--root',required=True)
    a=p.parse_args();r=(reader_audit if a.stage=='reader' else combined_audit)(a.root)
    print(json.dumps({k:v for k,v in r.items() if k!='samples'}))

if __name__=='__main__':main()
