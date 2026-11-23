"""Strict TorchAIR graphs for the Wan DiT's bandwidth-bound pointwise chain.

Same math as ops.functional.modulated_norm_reference / gated_add_reference:
this only lets the graph compiler fuse the existing elementwise chain into one
pass instead of several round trips through HBM. No kernel is written here and
no numerical contract changes. The CUDA counterpart is ``ops.cuda_compile``.
"""

from functools import lru_cache

import torch


@lru_cache(None)
def compiled(name):
    import torchair

    from ops.functional import gated_add_reference, modulated_norm_reference

    functions = dict(modulated_norm=modulated_norm_reference, gated_add=gated_add_reference)
    config = torchair.CompilerConfig()
    # Parameters stay mutable and gradients are compiled by AOTAutograd.
    config.experimental_config.frozen_parameter = False
    return torch.compile(
        functions[name],
        backend=torchair.get_npu_backend(compiler_config=config),
        fullgraph=True,
        dynamic=False,
    )
