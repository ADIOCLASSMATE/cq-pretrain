"""Immutable, relocatable visual feature/latent caches with exact identities."""
import hashlib
import json
import os
from pathlib import Path
import tempfile
from functools import lru_cache
import torch


def digest(value):
    return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=False).encode()).hexdigest()


@lru_cache(maxsize=16)
def encoder_identity(root):
    root=Path(root)
    if (root/'manifest.json').is_file():
        value=json.loads((root/'manifest.json').read_text())
        return digest({k:value[k] for k in ('model_id','revision','files')})
    # Wan's unused T5 checkpoint and bundled VAE are not generator weights.
    if root.is_file():names=[root]
    else:
        weights=sorted(root.glob('diffusion_pytorch_model*.safetensors')) or sorted(root.glob('*.pth'))
        names=weights+sorted(root.glob('*config*.json'))
    return digest([(p.name,hashlib.file_digest(p.open('rb'),'sha256').hexdigest()) for p in names])


class VisualCache:
    def __init__(self,root,encoder,preprocessing):
        self.root=Path(root);self.encoder=encoder;self.preprocessing=preprocessing

    def identity(self,source,clip):
        return dict(schema='cq.visual.v1',source=source,clip=clip,
                    encoder=self.encoder,preprocessing=self.preprocessing)

    def path(self,identity):
        key=digest(identity)
        return self.root/key[:2]/(key+'.pt')

    def load(self,identity):
        path=self.path(identity)
        if not path.exists():return None
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload['identity']!=identity:raise ValueError(f'Visual cache identity differs: {path}')
        tensors=payload['tensors']
        checksums={k:hashlib.sha256(v.contiguous().view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in tensors.items()}
        if payload['sha256']!=checksums:raise ValueError(f'Visual cache checksum differs: {path}')
        return tensors

    def save(self,identity,**tensors):
        existing=self.load(identity)
        if existing is not None:return self.path(identity)
        path=self.path(identity);path.parent.mkdir(parents=True,exist_ok=True)
        tensors={k:v.detach().cpu().contiguous() for k,v in tensors.items()}
        checksums={k:hashlib.sha256(v.view(torch.uint8).numpy().tobytes()).hexdigest() for k,v in tensors.items()}
        fd,tmp=tempfile.mkstemp(dir=path.parent,prefix='.'+path.name)
        try:
            with os.fdopen(fd,'wb') as out:
                torch.save(dict(identity=identity,tensors=tensors,sha256=checksums),out)
                out.flush();os.fsync(out.fileno())
            try:os.link(tmp,path)
            except FileExistsError:pass
        finally:Path(tmp).unlink(missing_ok=True)
        self.load(identity)
        return path


def caches(cfg):
    if not cfg.visual_cache:return None,None
    if not (cfg.freeze_vision and cfg.freeze_vision_merger):raise ValueError('Visual caches require frozen ViT and merger')
    from models.vision import vision_config
    if vision_config(cfg.vision_assets)['out_hidden_size']!=cfg.width:
        raise ValueError('Final visual caches require the native merger width, without a random projection')
    common={'version':'native-qwen-wan-v1'}
    qwen=VisualCache(Path(cfg.visual_cache)/'qwen',encoder_identity(cfg.vision_assets),
        dict(common,min_pixels=cfg.vision_min_pixels,max_pixels=cfg.vision_max_pixels))
    wan=VisualCache(Path(cfg.visual_cache)/'wan',encoder_identity(cfg.wan_vae_assets),
        dict(common,max_pixels=cfg.generation_max_pixels,posterior='mode',spatial_stride=8,temporal_stride=4))
    return qwen,wan
