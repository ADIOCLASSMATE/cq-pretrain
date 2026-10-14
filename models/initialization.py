"""Keep initializers in FP32 even when modules are constructed in BF16."""
import torch
from torch import nn


def initialize_normal(tensor, std):
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        nn.init.normal_(value, std=std)
        tensor.copy_(value.to(tensor.dtype))


def initialize_xavier(tensor):
    if tensor.is_meta:
        return
    with torch.no_grad():
        value = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        nn.init.xavier_uniform_(value)
        tensor.copy_(value.to(tensor.dtype))
