"""TP2 validation and timing of CQ state/KV caches and Wan condition KV caches."""
import argparse
import json
from pathlib import Path
import time
import numpy as np
import torch
import torch.distributed as dist
from configs.config import RunConfig
from evaluations.sampling import make_batch
from models.backbone_qwen35 import cast_for_training
from models.cq_backbone import backbone
from models.decoding import CQDecoder
from data.vision import NativeImageProcessor
from scripts.benchmark_backbones import Model
from data.generation import WanGenerationInputs
import distributed.megatron as P


def maximum(values):
    values=torch.tensor(values,device='npu',dtype=torch.float32)
    dist.all_reduce(values,op=dist.ReduceOp.MAX)
    return values.cpu().tolist()


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();cfg=RunConfig()
    P.initialize(2,cfg,initialize_weights=False)
    torch.manual_seed(73);torch.npu.manual_seed_all(73)
    from transformers.initialization import no_init_weights
    with no_init_weights():model=cast_for_training(Model(cfg,None,True))
    model.load_weights();model.to('npu').eval()
    report=dict(world_size=dist.get_world_size(),tp=2,dp=P.dp_size(),cq=[],wan={},complete=False)
    def save():
        if dist.get_rank()==0:
            args.output.write_text(json.dumps(report,indent=2)+'\n')
            print(json.dumps(report),flush=True)
    tokens=dict(bos=1,eos=2,boi=3,eoi=4,image=0)
    processed=NativeImageProcessor(cfg,4096)(np.zeros((256,256,3),np.uint8))
    for length in (256,2048):
        prefix=torch.randint(5,230000,(length,)).tolist();extra=list(range(21,29))
        batches=[make_batch(cfg,tokens,prefix+extra[:i],processed,'I2T') for i in range(len(extra)+1)]
        features=torch.randn(len(batches[0].vision.content_indices),cfg.width,device='npu',dtype=torch.bfloat16)
        device_batches=[b.to('npu') for b in batches]
        expected=[backbone(model,b,vision_features=features)[b.mainline.output_positions] for b in device_batches]
        decoder=CQDecoder(model,batches[0],len(extra),vision_features=features)
        actual=[decoder.hidden.clone()]+[decoder.advance(t).clone() for t in extra]
        errors=[float((x.float()-y.float()).norm()/y.float().norm()) for x,y in zip(actual,expected)]
        error=max(errors)
        max_abs=max(float((x.float()-y.float()).abs().max()) for x,y in zip(actual,expected))
        error,max_abs=maximum([error,max_abs])
        row=dict(prefix_tokens=batches[0].ids.shape[1]-1,decode_tokens=len(extra),relative_rms=error,
                 max_abs=max_abs,rank0_error_by_token=errors)
        report['cq'].append(row);save()
        assert error<.025, row
        elapsed={}
        for cached in (False,True):
            timings=[]
            for _ in range(3):
                if cached:decoder=CQDecoder(model,batches[0],len(extra),vision_features=features)
                dist.barrier();torch.npu.synchronize();start=time.perf_counter()
                if cached:
                    for token in extra:decoder.advance(token)
                else:
                    for batch in device_batches[1:]:backbone(model,batch,vision_features=features)
                torch.npu.synchronize();timings.append(time.perf_counter()-start)
            elapsed['cached' if cached else 'full_prefix']=sorted(timings)[1]
        old,new=maximum([elapsed['full_prefix'],elapsed['cached']])
        row.update(full_prefix_ms_per_token=old*1000/len(extra),cached_ms_per_token=new*1000/len(extra),speedup=old/new)
        save()
    layout=WanGenerationInputs.prepare([np.zeros((256,256,3),np.uint8),
        np.zeros((5,256,256,3),np.uint8)],[64,64],cfg.dit_width//cfg.dit_heads).to('npu')
    condition=torch.randn(2,64,cfg.width,device='npu',dtype=torch.bfloat16)
    latent=torch.randn(sum(layout.lengths),cfg.latent_channels*4,device='npu')
    features=torch.randn(2,cfg.wan_freq_dim,device='npu')
    context=model.renderer.project_conditions(condition,layout)
    cached=model.renderer.cache_condition_kv(context)
    expected=model.renderer.predict(latent,None,features,layout,context)
    actual=model.renderer.predict(latent,None,features,layout,context,cached)
    error=maximum([float((actual-expected).norm()/expected.norm())])[0]
    report['wan']=dict(relative_rms=error,cache_bytes=sum(t.numel()*t.element_size() for kv in cached for t in kv))
    save();assert error==0,report['wan']
    for label,kv in [('uncached',None),('cached',cached)]:
        times=[]
        for _ in range(7):
            dist.barrier();torch.npu.synchronize();start=time.perf_counter()
            model.renderer.predict(latent,None,features,layout,context,kv)
            torch.npu.synchronize();times.append(time.perf_counter()-start)
        report['wan'][label+'_ms']=maximum([sorted(times[2:])[2]*1000])[0]
    report['wan']['speedup']=report['wan']['uncached_ms']/report['wan']['cached_ms']
    report['complete']=True;save()
    dist.barrier();dist.destroy_process_group()


if __name__=='__main__':main()
