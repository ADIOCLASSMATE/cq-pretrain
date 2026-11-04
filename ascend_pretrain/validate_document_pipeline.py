"""Read a bounded real-data prefix; compare coverage and CPU packing, not NPU speed."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time

import numpy as np
import torch

from .config import ModelConfig
from .input_pipeline.stream import DocumentStream
from .input_pipeline.packing import Packer,token_contract


class PrefixStream:
    def __init__(self,source,limit):self.source,self.limit,self.consumed=source,limit,0
    def __getattr__(self,name):return getattr(self.source,name)
    def peek(self):return self.source.peek() if self.consumed<self.limit else None
    def advance(self):self.source.advance();self.consumed+=1
    def state_dict(self):return dict(self.source.state_dict(),audit_documents=self.consumed)


def audit(args,preserve):
    cfg=ModelConfig.load(args.config);cfg.preserve_documents=preserve
    docs=DocumentStream(args.data_root,'A',0,32,args.seed,workers=2,prefetch=4,
        image_threads=2,image_prefetch=16,block_rows=4,tokenize_rows=32,cfg=cfg,
        native_images=True,sequence_length=8192)
    source=PrefixStream(docs,args.documents)
    packer=Packer(source,cfg,8192,token_contract(args.data_root,cfg),
        max_images=0 if preserve else 16,padding_multiple=128)
    result=dict(preserve_documents=preserve,documents=args.documents,images=0,body_tokens=0,
        occupied_tokens=0,physical_slots=0,backbone_slots_including_queries=0,
        generation_tokens=0,dit_microbatches=0,document_fragments=0,document_continuations=0,
        packs=[],document_details={})
    digests={};began=time.perf_counter()
    try:
        while (pair:=packer.next()) is not None:
            batch,receipt=pair
            detail=result['document_details']
            for r in receipt['documents']:
                doc=detail.setdefault(r['id'],dict(body_tokens=0,images=0,fragments=0,completed=0,
                    format=r['format'],direction=r['direction']))
                doc['fragments']+=1;doc['completed']+=int(r['completed'])
                doc['images']+=r['images'];doc['body_tokens']+=r['body_tokens']
                h=digests.setdefault(r['id'],hashlib.sha256())
                for start,n in r['body_spans']:h.update(batch.ids[0,start:start+n].numpy().astype('<i8').tobytes())
                if r['element'] or r['offset']:result['document_continuations']+=1
                result['document_fragments']+=1
            layout=batch.generation
            sizes=[p.ends[-1] for p in layout.microbatches] if layout and layout.microbatches else [layout.ends[-1]] if layout else []
            if preserve and any(n>cfg.generation_token_budget for n in sizes):raise AssertionError('Unbounded DiT chunk')
            if layout and any(h*w*256>262144 for h,w in layout.grids):raise AssertionError('Generation pixel cap exceeded')
            if batch.vision and any(h*w*256>262144 for _,h,w in batch.vision.grid_thw):raise AssertionError('Vision pixel cap exceeded')
            for key,amount in dict(images=receipt['images'],body_tokens=receipt['body_tokens'],
                    occupied_tokens=receipt['occupied'],physical_slots=batch.ids.shape[1],
                    backbone_slots_including_queries=batch.ids.shape[1]+len(batch.mainline.query_positions),
                    generation_tokens=receipt['generation_tokens'],dit_microbatches=len(sizes)).items():
                result[key]+=amount
            result['packs'].append(dict(occupied=receipt['occupied'],length=batch.ids.shape[1],
                queries=len(batch.mainline.query_positions),images=receipt['images'],
                generation_tokens=receipt['generation_tokens'],dit_chunk_tokens=sizes))
        result['cpu_seconds']=time.perf_counter()-began
        result['packing_stats']=copy.deepcopy(packer.stats)
        result['stream_stats']=copy.deepcopy(docs.stats)
        result['body_digests']={key:h.hexdigest() for key,h in digests.items()}
    finally:docs.close()
    if len(result['document_details'])!=args.documents:raise AssertionError('Missing documents')
    if any(x['completed']!=1 for x in result['document_details'].values()):raise AssertionError('Document not completed exactly once')
    result['pack_count']=len(result['packs'])
    result['context_capacity_utilization']=result['occupied_tokens']/(8192*result['pack_count'])
    result['cpu_documents_per_second']=args.documents/result['cpu_seconds']
    return result


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--config',default=str(Path(__file__).parent/'configs/qwen35_9b_moss_documents.json'))
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--documents',type=int,default=256)
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args();torch.set_num_threads(1)
    runs=[]
    for preserve in (False,True):
        run=audit(args,preserve);runs.append(run)
        print(json.dumps({k:run[k] for k in ('preserve_documents','pack_count','images','body_tokens',
            'document_continuations','context_capacity_utilization','cpu_seconds','cpu_documents_per_second')}),flush=True)
    old,new=runs
    if old['body_digests']!=new['body_digests']:raise AssertionError('Corpus text changed')
    for key,a in old['document_details'].items():
        b=new['document_details'][key]
        if any(a[k]!=b[k] for k in ('body_tokens','images','completed','format','direction')):
            raise AssertionError('Coverage changed: '+key)
    report=dict(passed=True,scope='CPU real-image preprocessing and packing only; no VAE or NPU training',
        data_root=args.data_root,identical_documents_body_tokens_and_images=True,
        caveat='Sequential local runs share filesystem caches; CPU timings are not a training-speed comparison.',runs=runs)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(dict(passed=True,report=str(args.output))),flush=True)


if __name__=='__main__':main()
