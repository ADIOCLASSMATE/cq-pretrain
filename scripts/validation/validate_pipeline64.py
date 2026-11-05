"""64-rank NPU stream, pinned-source lifetime, restart and coverage oracle."""
import argparse
import gc
import json
import os
from pathlib import Path
import torch
import torch.distributed as dist
import torch_npu

from configs.config import RunConfig
from data.stream import DocumentStream
from data.packing import Packer, ConsumedLoader, token_contract
from data.device import DeviceLoader
from data.codec import DevicePreparer


def main():
    p=argparse.ArgumentParser();p.add_argument('--data-root',required=True);p.add_argument('--output',required=True)
    p.add_argument('--config',default='configs/legacy/qwen35_cq.json')
    p.add_argument('--shuffle-block-rows',type=int,default=64)
    a=p.parse_args();device=f'npu:{os.environ["LOCAL_RANK"]}';torch.npu.set_device(device)
    torch.set_num_threads(1);dist.init_process_group('hccl');rank,world=dist.get_rank(),dist.get_world_size()
    if world!=64:raise ValueError('This oracle must run on exactly 64 NPUs')
    cfg=RunConfig.load(a.config);tokens=token_contract(a.data_root,cfg)
    out=Path(a.output)
    if rank==0:out.mkdir(parents=True,exist_ok=True)
    dist.barrier()
    def loader(depth,state=None,workers=2):
        docs=DocumentStream(a.data_root,'A',rank,world,42,workers=workers,prefetch=3,
            state=state['documents'] if state else None,allow_development=True,
            native_images=True,cfg=cfg,sequence_length=2048,block_rows=a.shuffle_block_rows,tokenize_rows=16 if workers==2 else 7)
        pack=Packer(docs,cfg,2048,tokens,state,pin_device=device,max_images=2,padding_multiple=128)
        host=ConsumedLoader(pack,3,state)
        return DeviceLoader(host,DevicePreparer(cfg,device=device,encoder='fake'),depth)
    saved=[];ids=[];restart=None
    before=torch.random.get_rng_state().clone();npu_before=torch.npu.get_rng_state().clone()
    # CPU BF16 rounding must match the old device-copy/cast exactly.
    probe=torch.randn(1<<18,generator=torch.Generator().manual_seed(31))
    torch.testing.assert_close(probe.to(device,dtype=torch.bfloat16),probe.bfloat16().to(device),atol=0,rtol=0)
    for depth in (0,2):
        pipeline=loader(depth)
        try:
            for group_index in range(2):
                for item in pipeline.next_group(3):
                    batch=pipeline.acquire(item)
                    # Force allocator churn while the transfer stream and
                    # consuming stream run asynchronously, then copy for oracle.
                    junk=torch.empty(8<<20,device=device,dtype=torch.uint8);junk.fill_(7);del junk
                    result={k:getattr(batch,k).cpu() for k in ('ids','latents','noise','times')}
                    if batch.vision is not None:result['vision']=batch.vision.pixel_values.cpu()
                    if depth==0:
                        saved.append((item[1],result))
                        ids.extend(r['id'] for r in item[1]['documents'] if r['completed'])
                    else:
                        receipt,expected=saved.pop(0)
                        assert receipt==item[1]
                        for k in result:torch.testing.assert_close(result[k],expected[k],atol=0,rtol=0)
                    pipeline.acknowledge(item[1])
                    if depth==0 and len(saved)==3:restart=pipeline.state_dict()
                    del batch,result
                torch.npu.synchronize();pipeline.device_wait_seconds()
        finally:pipeline.close()
        if depth==0:resume_expected=list(saved[3:])
    assert torch.equal(before,torch.random.get_rng_state())
    assert torch.equal(npu_before,torch.npu.get_rng_state())
    # Restart from consumed step 3 with changed worker/task granularity.
    pipeline=loader(2,restart,1)
    try:
        group=pipeline.next_group(3)
        for item,(receipt,expected) in zip(group,resume_expected):
            batch=pipeline.acquire(item);assert torch.isfinite(batch.latents).all().item()
            assert item[1]==receipt
            for key in ('ids','latents','noise','times'):
                torch.testing.assert_close(getattr(batch,key).cpu(),expected[key],atol=0,rtol=0)
            pipeline.acknowledge(item[1])
        assert pipeline.state_dict()['serial']==6
    finally:pipeline.close()
    gathered=[None]*world;dist.all_gather_object(gathered,ids)
    flat=[x for part in gathered for x in part];assert len(flat)==len(set(flat))
    report=dict(passed=True,world=world,rank=rank,completed_documents=len(ids),
        tests=['sync_async_bitwise','private_cpu_and_npu_rng','allocator_lifetime','worker_change_restart','dp_disjointness'])
    (out/f'rank_{rank:05d}.json').write_text(json.dumps(report))
    dist.barrier()
    if rank==0:(out/'summary.json').write_text(json.dumps(dict(report,completed_documents_all_ranks=len(flat)),indent=2))
    dist.destroy_process_group()


if __name__=='__main__':main()
