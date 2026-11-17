"""Explicit versioned runtime selection; never silently fall back to an old stack."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def name():
    from distributed.accel import IS_NPU

    value = os.environ.get('PRETRAIN_STACK', 'core17' if IS_NPU else 'cuda')
    # Existing model YAMLs describe the Ascend stack. Device dispatch selects
    # CUDA without rewriting their model or CPU planning configuration.
    if value == 'core17' and not IS_NPU:
        value = 'cuda'
    if value not in ('core17', 'cuda'):
        raise ValueError(f'Unsupported PRETRAIN_STACK {value!r}; expected core17 or cuda')
    return value


def source_directories():
    deps = PACKAGE / '.deps'
    name()
    return [deps / 'core17' / component for component in
            ('Megatron-LM', 'MegatronAdaptor', 'TransformerEngineNPU', 'MindSpeed')]


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
