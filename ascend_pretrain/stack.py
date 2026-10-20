"""Explicit versioned runtime selection; never silently fall back to an old stack."""
import os
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent


def name():
    value = os.environ.get('PRETRAIN_STACK', 'core17')
    if value != 'core17':
        raise ValueError(f'This project requires the pinned CANN 9.0/Core 0.17 stack; got {value!r}')
    return value


def source_directories():
    deps = PACKAGE / '.deps'
    name()
    return [deps / 'core17' / component for component in
            ('Megatron-LM', 'MegatronAdaptor', 'TransformerEngineNPU', 'MindSpeed')]


def lock_path():
    return PACKAGE / f'deps.{name()}.lock.json'
