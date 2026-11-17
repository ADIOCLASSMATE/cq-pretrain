"""Explicit versioned runtime selection; never silently fall back to an old stack."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def name():
    value = os.environ.get('PRETRAIN_STACK', 'cuda')
    if value != 'cuda':
        raise ValueError(f'Unsupported PRETRAIN_STACK {value!r}; this branch is CUDA-only')
    return value


def lock_path():
    return PACKAGE / 'configs' / f'deps.{name()}.lock.json'


def installed_versions():
    """Record the actual environment alongside the declared dependency lock."""
    from importlib.metadata import PackageNotFoundError, version

    result = {}
    for package in ('torch', 'triton', 'pytorch-triton', 'flash-linear-attention', 'fla-core',
                    'megatron-core', 'apex', 'transformer-engine',
                    'transformers', 'diffusers', 'pyarrow', 'av'):
        try:
            result[package] = version(package)
        except PackageNotFoundError:
            result[package] = None
    return result
