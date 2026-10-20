"""Optional, strict TorchAIR graphs for the DiT's bandwidth-heavy pointwise work."""
from functools import lru_cache
import torch
from torch.nn import functional as F


def modulated_norm_reference(hidden, scale, shift):
    normalized = F.layer_norm(hidden, (hidden.shape[-1],), eps=1e-6)
    return normalized * (1 + scale) + shift


def gated_add_reference(hidden, branch, gate):
    return hidden + gate * branch


@lru_cache(None)
def compiled(name):
    import torchair
    functions = dict(modulated_norm=modulated_norm_reference, gated_add=gated_add_reference)
    config = torchair.CompilerConfig()
    # Parameters stay mutable and gradients are compiled by AOTAutograd.
    config.experimental_config.frozen_parameter = False
    return torch.compile(functions[name], backend=torchair.get_npu_backend(compiler_config=config),
                         fullgraph=True, dynamic=False)
