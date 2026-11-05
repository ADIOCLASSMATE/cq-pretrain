"""Synthetic native-resolution inputs for the production packer."""
import torch


def fake_vision_batch(cfg,length,batch_size,images,seed,device,dtype):
    """Native rectangular synthetic inputs through the same production packer."""
    import numpy as np
    from data.packing import Packer
    if batch_size!=1:
        raise ValueError('Native-resolution packing currently uses micro_batch=1; use gradient accumulation')
    class Documents:
        def __init__(self):
            rng=np.random.default_rng(seed);self.index=0
            self.pixels=[rng.integers(0,256,size=(cfg.image_size,cfg.image_size*(1+i%2),3),dtype=np.uint8)
                         for i in range(images)]
            elements=[]
            for i in range(images):elements.extend((dict(type='text',input_ids=[5,6,7]),dict(type='image',index=i)))
            elements.append(dict(type='text',input_ids=[8]*length))
            self.doc=dict(id='synthetic',elements=elements)
        def peek(self):return self.doc if self.index==0 else None
        def advance(self):self.index+=1
        def prefetch_images(self,*args):pass
        def image(self,doc,element):return self.pixels[doc['elements'][element]['index']]
        def state_dict(self):return dict(index=self.index)
    packed=Packer(Documents(),cfg,length,dict(bos=1,eos=2,boi=3,eoi=4,image=cfg.image_mask_token_id)).next()
    batch,_=packed
    if len(batch.image_indices)!=images:
        raise ValueError('Synthetic sequence is too short to fit the requested native images')
    if cfg.renderer=='wan':
        from data.codec import DevicePreparer
        receipt=dict(serial=0,next_state=dict(documents=dict(signature=dict(
            snapshot='synthetic',seed=seed,epoch=0,rank=0))))
        return DevicePreparer(cfg,device=device,encoder='fake')(batch,receipt)
    generator=torch.Generator().manual_seed(seed)
    batch.pixels=batch.pixels.to(dtype).div(127.5).sub(1)
    batch.latents=torch.randn(images,cfg.image_tokens,cfg.latent_channels,generator=generator,dtype=dtype)
    batch.noise=torch.randn(images*cfg.flow_draws,cfg.image_tokens,cfg.latent_channels,generator=generator,dtype=dtype)
    batch.times=torch.randn(images*cfg.flow_draws,generator=generator).sigmoid().clamp(1e-5,1-1e-5)
    return batch.to(device)
