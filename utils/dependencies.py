"""Explicit versioned runtime selection; never silently fall back to an old stack."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parents[1]


def name():
    value = os.environ.get('PRETRAIN_STACK', 'core17')
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
