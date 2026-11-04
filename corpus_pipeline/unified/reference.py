"""Bounded-memory exact SSCD calibration reference over the joint image catalog."""
from pathlib import Path
import os
import numpy as np
from threadpoolctl import threadpool_limits
from corpus_pipeline.postprocess import sscd_index as s


def build(index,*,threshold=.75,query_count=128,top_k=10,batch_size=32768,threads=3,allow_bounded=False):
    data=s.Catalog(index,allow_bounded=allow_bounded)
    root=Path(index).resolve();out=root/'calibration/unified_reference';out.mkdir(parents=True,exist_ok=True)
    identity={'catalog_sha256':data.sha256,'code_sha256':s.file_sha(__file__),
              'parameters':{'threshold':threshold,'query_count':query_count,'top_k':top_k}}
    if (out/'COMPLETE.json').exists():
        receipt=s.read_json(out/'COMPLETE.json')
        s.require(receipt['identity']==identity,'Reference input/implementation changed')
        for item in receipt['artifacts']:s.bound_path(root,item)
        return out/'evidence.npz'
    s.require(data.count>=2,'SSCD reference needs at least two distinct image assets')
    ids=np.unique(np.linspace(0,data.count-1,min(query_count,data.count),dtype=np.int64))
    k=min(top_k,data.count-1);queries=s.normalized(data.fetch(ids)).astype(np.float64)
    best_ids=np.empty((len(ids),0),np.int64);best_scores=np.empty((len(ids),0),np.float32)
    positive=np.zeros(len(ids),np.int64)
    with threadpool_limits(limits=threads):
        for start,raw in data.iter_vectors(batch_size=batch_size):
            scores=(queries@s.normalized(raw).astype(np.float64).T).astype(np.float32)
            columns=np.arange(start,start+len(raw),dtype=np.int64)
            for row,qid in enumerate(ids):
                if start<=qid<start+len(raw):scores[row,qid-start]=-np.inf
            positive+=(scores>=threshold).sum(axis=1)
            next_ids=[];next_scores=[]
            for row in range(len(ids)):
                names=np.r_[best_ids[row],columns];values=np.r_[best_scores[row],scores[row]]
                order=np.lexsort((names,-values))[:k]
                next_ids.append(names[order]);next_scores.append(values[order])
            best_ids=np.asarray(next_ids);best_scores=np.asarray(next_scores)
    temp=out/'.evidence.partial.npz';target=out/'evidence.npz'
    np.savez(temp,query_ids=ids,exact_ids=best_ids,exact_scores=best_scores,threshold_positive_count=positive)
    os.replace(temp,target)
    s.atomic_json(out/'COMPLETE.json',{'complete':True,'identity':identity,'exact_reference_vectors':data.count,
        'query_count':len(ids),'threshold_positive_count':int(positive.sum()),
        'artifacts':[s.artifact(target,root)]})
    return target
