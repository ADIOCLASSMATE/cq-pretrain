"""Small real-loader runtime helpers, kept out of CPU worker imports."""
import torch
from data.batch import Batch
from data.batch_layout import build_plan


def empty_batch(cfg, tokens):
    batch=Batch(ids=torch.tensor([[tokens['bos']]]),generation_image_indices=torch.empty(0,dtype=torch.long),
        latents=torch.empty(0,cfg.latent_channels*4,dtype=torch.bfloat16),
        image_indices=torch.empty(0,0,dtype=torch.long),text_indices=torch.empty(0,dtype=torch.long),
        image_queries=torch.zeros(1,1,dtype=torch.bool),sigma=torch.zeros(1,1,dtype=torch.long),
        segments=torch.full((1,1),-1,dtype=torch.long),row_col=torch.zeros(3,1,1,dtype=torch.long),
        noise=torch.empty(0),times=torch.empty(0))
    batch.mainline=build_plan(batch,cfg)
    return batch
