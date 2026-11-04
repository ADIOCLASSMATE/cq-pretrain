"""One additional, SHA-verified HTTP retry for unresolved official PixMo images."""
from __future__ import annotations
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import threading
import time
from urllib.parse import urljoin

from corpus_pipeline.production.common import atomic_json, canonical, file_sha
from .pixmo_recovery import (CONTRACT, MAX_BYTES, _claim, _public_url, _save_official_image,
                            describe, finalize_bindings, stamp)

HEADERS={'User-Agent':'Mozilla/5.0','Accept':'image/avif,image/webp,image/apng,image/*,*/*;q=0.8'}


def fetch(url):
    import requests
    session=requests.Session();session.trust_env=False
    try:
        for _ in range(6):
            _public_url(url)
            with session.get(url,headers=HEADERS,allow_redirects=False,stream=True,timeout=(6,15)) as response:
                if response.is_redirect:url=urljoin(url,response.headers['Location']);continue
                response.raise_for_status();blocks=[];size=0
                for block in response.iter_content(1<<18):
                    size+=len(block)
                    if size>MAX_BYTES:raise ValueError('official image byte budget exceeded')
                    blocks.append(block)
                return b''.join(blocks)
        raise ValueError('original image redirect limit')
    finally:session.close()


def retry(item,root,dataset_id):
    sha,urls=item;errors=[]
    for url in urls[:3]:
        try:
            blob=fetch(url);observed=hashlib.sha256(blob).hexdigest()
            if observed!=sha:
                errors.append({'reason':'source_changed','observed_sha256':observed});continue
            describe(blob)  # Reject unsupported geometry/frames and invalid encodings.
            result=_save_official_image(blob,sha,root,dataset_id)
            return {**result,'original_sha256':sha,'source_url':url,'recovery_pass':'browser_headers_retry_v1'}
        except Exception as error:
            value={'error_type':type(error).__name__}
            if getattr(error,'response',None) is not None:value['http_status']=error.response.status_code
            errors.append(value)
    return {'status':'unbound','original_sha256':sha,'recovery_pass':'browser_headers_retry_v1','attempts':errors}


def finalize(primary,output,final_output,workers=4):
    primary,output,final_output=map(Path,(primary,output,final_output));final_output.mkdir(parents=True,exist_ok=True)
    receipt=final_output/'COMPLETE.json'
    if receipt.exists():return json.loads(receipt.read_text())
    original=finalize_bindings(primary,workers)
    if original is None or not all((output/f'worker-{i}'/'COMPLETE.json').exists() for i in range(workers)):return None
    claim=final_output/'.claim'
    try:_claim(claim)
    except RuntimeError:return None
    try:
        if file_sha(original['path'])!=original['sha256']:raise ValueError('primary PixMo results changed')
        target=final_output/'bindings.sqlite';temporary=target.with_suffix('.building');shutil.copyfile(original['path'],temporary)
        db=sqlite3.connect(temporary);added=0;receipts=[]
        for i in range(workers):
            complete=json.loads((output/f'worker-{i}'/'COMPLETE.json').read_text());receipts.append(complete)
            if file_sha(complete['path'])!=complete['sha256']:raise ValueError('PixMo retry result changed')
            part=sqlite3.connect('file:'+complete['path']+'?mode=ro',uri=True)
            try:
                for sha,status,value in part.execute('SELECT original_sha256,status,result_json FROM results'):
                    previous=db.execute('SELECT status,result_json FROM bindings WHERE original_sha256=?',(sha,)).fetchone()
                    if previous is None:raise ValueError('retry introduced an unregistered original image SHA')
                    if previous[0]=='bound':continue
                    if status=='bound':
                        result=json.loads(value)
                        if result['proof'].get('source_sha256')!=sha or result['local_sha256']!=sha:
                            raise ValueError('retry result lacks the official original image SHA')
                        db.execute('UPDATE bindings SET status=?,result_json=? WHERE original_sha256=?',(status,value,sha));added+=1
                    else:
                        result=json.loads(previous[1]);result['additional_retry']=json.loads(value)
                        db.execute('UPDATE bindings SET result_json=? WHERE original_sha256=?',(canonical(result),sha))
            finally:part.close()
            db.commit()
        counts=dict(db.execute('SELECT status,count(*) FROM bindings GROUP BY status'));db.close();os.replace(temporary,target)
        result={**{k:v for k,v in original.items() if k not in ('path','sha256','fingerprint','counts')},
                'path':str(target),'sha256':file_sha(target),'fingerprint':stamp(target),'counts':counts,
                'retry_added_bound_images':added,'retry_worker_receipts':receipts,'primary_binding_sha256':original['sha256']}
        atomic_json(receipt,result);return result
    finally:shutil.rmtree(claim)


def run(primary,annotations,output,final_output,recovered_root,recovered_dataset_id,worker_index,workers=4,threads=16):
    import pyarrow.parquet as pq
    primary,annotations,output=map(Path,(primary,annotations,output));work=output/f'worker-{worker_index}';work.mkdir(parents=True,exist_ok=True)
    if (work/'COMPLETE.json').exists():finalize(primary,output,final_output,workers);return
    claim=_claim(work/'.claim');stop=threading.Event();state={'status':'running','phase':'retrying_unresolved_images','pid':os.getpid(),'worker':worker_index}
    def beat():
        while not stop.is_set():
            value={**state,'updated_unix':time.time()};atomic_json(work/'HEARTBEAT.json',value);atomic_json(claim/'HEARTBEAT.json',value);stop.wait(15)
    heartbeat=threading.Thread(target=beat,daemon=True);heartbeat.start()
    try:
        identity={'contract':CONTRACT,'pass':'browser_headers_retry_v1','primary':str(primary.resolve()),
                  'annotations_sha256':file_sha(annotations),'worker':worker_index,'workers':workers,
                  'recovered_root':str(Path(recovered_root).resolve()),'recovered_dataset_id':recovered_dataset_id,
                  'implementation_sha256':file_sha(__file__),'headers':HEADERS}
        saved=work/'IDENTITY.json'
        if saved.exists() and json.loads(saved.read_text())!=identity:raise ValueError('PixMo retry identity changed')
        atomic_json(saved,identity);urls={}
        for row in pq.read_table(annotations,columns=['image_sha256','image_url']).to_pylist():
            sha=row['image_sha256']
            if int(sha[:8],16)%workers==worker_index:
                values=urls.setdefault(sha,[])
                if row['image_url'] not in values:values.append(row['image_url'])
        db=sqlite3.connect(work/'results.sqlite');db.execute('CREATE TABLE IF NOT EXISTS results(original_sha256 TEXT PRIMARY KEY,status TEXT,result_json TEXT)')
        done={row[0] for row in db.execute('SELECT original_sha256 FROM results')};counts=Counter(dict(db.execute('SELECT status,count(*) FROM results GROUP BY status')))
        with ThreadPoolExecutor(threads) as pool:
            while True:
                path=primary/f'worker-{worker_index}'/'results.sqlite';pending=[]
                if path.exists():
                    reader=sqlite3.connect('file:'+str(path)+'?mode=ro',uri=True,timeout=30)
                    try:pending=[sha for sha, in reader.execute("SELECT original_sha256 FROM results WHERE status='unbound'") if sha not in done]
                    finally:reader.close()
                for start in range(0,len(pending),threads*4):
                    futures=[pool.submit(retry,(sha,urls[sha]),recovered_root,recovered_dataset_id) for sha in pending[start:start+threads*4]]
                    for future in as_completed(futures):
                        result=future.result();sha=result['original_sha256'];db.execute('INSERT INTO results VALUES(?,?,?)',(sha,result['status'],canonical(result)));db.commit()
                        done.add(sha);counts[result['status']]+=1;state.update(completed=len(done),counts=dict(counts))
                if not pending and (primary/f'worker-{worker_index}'/'COMPLETE.json').exists():break
                stop.wait(15)
        db.close();result={'status':'complete','path':str(work/'results.sqlite'),'sha256':file_sha(work/'results.sqlite'),'counts':dict(counts),'identity':identity}
        atomic_json(work/'COMPLETE.json',result);state.update(status='complete',phase='complete');finalize(primary,output,final_output,workers)
    except BaseException as error:
        state.update(status='failed',error_type=type(error).__name__,error=str(error));atomic_json(work/'FAILED.json',state);raise
    finally:
        stop.set();heartbeat.join();atomic_json(work/'HEARTBEAT.json',{**state,'updated_unix':time.time()});shutil.rmtree(claim)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('primary','annotations','output','final_output','recovered_root','recovered_dataset_id'):
        parser.add_argument('--'+name.replace('_','-'),required=True)
    parser.add_argument('--worker-index',type=int,required=True);parser.add_argument('--workers',type=int,default=4);parser.add_argument('--threads',type=int,default=16)
    run(**vars(parser.parse_args()))

if __name__=='__main__':main()
