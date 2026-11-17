"""Inductor fusion for the Wan DiT's bandwidth-bound pointwise chain.

Same math as ops.functional.modulated_norm_reference / gated_add_reference:
this only lets Inductor fuse the existing elementwise chain (layer_norm, the
affine modulation, the gate multiply and the residual add) into one pass
instead of several full round trips through GDDR6. No kernel is written here
and no numerical contract changes.

Packed Wan token counts vary per batch, so dynamic=True is mandatory: with
static shapes every new packed length triggers a fresh Inductor compilation.
"""

from functools import lru_cache

import torch


@lru_cache(None)
def compiled(name):
    from ops.functional import gated_add_reference, modulated_norm_reference

    functions = dict(modulated_norm=modulated_norm_reference, gated_add=gated_add_reference)
    return torch.compile(functions[name], dynamic=True, fullgraph=True)
