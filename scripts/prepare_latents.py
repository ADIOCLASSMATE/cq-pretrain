"""Resume immutable Qwen merger-feature and Wan latent caches using the training loader."""
import argparse
import json
import math
import torch
from configs.config import ModelConfig
from data.cache import caches
from data.stream import DocumentStream
from data.packing import Packer, token_contract
from data.generation import patch_latent
from models.vision import QwenVisionInput


@torch.no_grad()
def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/qwen35_2b_wan.yaml')
    parser.add_argument('--data-root',required=True)
    parser.add_argument('--cache',required=True)
    parser.add_argument('--device',default='npu:0')
    parser.add_argument('--rank',type=int,default=0)
    parser.add_argument('--world',type=int,default=1)
    parser.add_argument('--sequence-length',type=int,default=8192)
    args=parser.parse_args()
    if args.device.startswith('npu'):
        import torch_npu
        torch.npu.set_device(args.device)
    cfg=ModelConfig.load(args.config);cfg.visual_cache=args.cache
    cfg.validate();feature_cache,latent_cache=caches(cfg)
    documents=DocumentStream(args.data_root,'A',args.rank,args.world,42,cfg=cfg,
        native_images=True,sequence_length=args.sequence_length)
    packer=Packer(documents,cfg,args.sequence_length,token_contract(args.data_root,cfg))
    vision=vae=None;written=dict(features=0,latents=0,batches=0)
    dtype=torch.bfloat16 if args.device.startswith('npu') else torch.float32
    try:
        while (item:=packer.next()) is not None:
            batch,_=item
            if batch.vision is not None:
                layout=batch.vision
                missing=[feature_cache.load(key) is None for key in layout.cache_keys]
                if any(missing):
                    if vision is None:
                        vision=QwenVisionInput(cfg);vision.load_pretrained(cfg.vision_assets)
                        vision=vision.to(device=args.device,dtype=dtype).eval().requires_grad_(False)
                    features=vision(layout.to(args.device));offset=0
                    for key,grid,needed in zip(layout.cache_keys,layout.grid_thw,missing):
                        count=math.prod(grid)//4
                        if needed:
                            feature_cache.save(key,features=features[offset:offset+count]);written['features']+=1
                        offset+=count
            if batch.generation is not None:
                for key,pixels in zip(batch.generation.cache_keys,batch.generation.pixels):
                    if latent_cache.load(key) is not None:continue
                    if vae is None:
                        from data.wan_codec import WanVideoCodec
                        vae=WanVideoCodec.from_pretrained(cfg.wan_vae_assets,device=args.device)
                    latent=vae.encode(pixels.to(args.device).float().div(127.5).sub(1))
                    latent_cache.save(key,latent=patch_latent(latent));written['latents']+=1
            written['batches']+=1
            if written['batches']%100==0:print(json.dumps(written),flush=True)
    finally:documents.close()
    print(json.dumps(dict(status='complete',**written)),flush=True)


if __name__=='__main__':main()
