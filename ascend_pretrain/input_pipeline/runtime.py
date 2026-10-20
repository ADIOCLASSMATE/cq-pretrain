"""Small real-loader runtime helpers, kept out of CPU worker imports."""
import torch
from ..data import Batch
from ..mainline import build_plan


def empty_batch(cfg, tokens):
    batch=Batch(ids=torch.tensor([[tokens['bos']]]),pixels=torch.empty(0,3,cfg.image_size,cfg.image_size,dtype=torch.bfloat16),
        latents=torch.empty(0,cfg.image_tokens,cfg.latent_channels,dtype=torch.bfloat16),
        image_indices=torch.empty(0,cfg.image_tokens,dtype=torch.long),text_indices=torch.empty(0,dtype=torch.long),
        image_queries=torch.zeros(1,1,dtype=torch.bool),sigma=torch.zeros(1,1,dtype=torch.long),
        segments=torch.full((1,1),-1,dtype=torch.long),row_col=torch.zeros(2,1,1,dtype=torch.long),
        noise=torch.empty(0),times=torch.empty(0),mask=torch.empty(0,dtype=torch.bool),valid_rows=torch.empty(0,dtype=torch.bool))
    batch.mainline=build_plan(batch,cfg)
    return batch
