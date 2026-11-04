"""Manage an explicit workload budget and argv through a configured adapter."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib
import json
from pathlib import Path
import signal
import threading

from .engine import Controller, request_stop, summary
from .models import WorkloadSpec


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('plan', 'status', 'run', 'stop'))
    parser.add_argument('--config', required=True, type=Path)
    parser.add_argument('--state-dir', type=Path)
    parser.add_argument('--profile')
    parser.add_argument('--max-cards', type=int)
    parser.add_argument('--command-json', help='Explicit JSON argv array; never an implicit shell string')
    parser.add_argument('--once', action='store_true', help='Run one durable management cycle')
    return parser.parse_args(argv)


def read_config(args):
    config = json.loads(args.config.read_text())
    raw = dict(config['workload'])
    for key in ('profile', 'max_cards'):
        value = getattr(args, key)
        if value is not None:
            raw[key] = value
    if args.command_json is not None:
        raw['command'] = json.loads(args.command_json)
    spec = WorkloadSpec(**raw)
    directory = args.state_dir or Path(config['state_dir'])
    if not directory.is_absolute():
        directory = args.config.resolve().parent / directory
    return config, spec, directory.resolve()


def main(argv=None):
    args = parse_args(argv)
    config, spec, directory = read_config(args)
    if args.action == 'status':
        path = directory / 'STATE.json'
        result = summary(json.loads(path.read_text())) if path.exists() else {
            'scope': spec.scope, 'profile': spec.profile, 'phase': 'not_started', 'max_cards': spec.max_cards}
    elif args.action == 'stop':
        request_stop(directory, scope=spec.scope, profile=spec.profile)
        result = {'scope': spec.scope, 'profile': spec.profile, 'stop_requested': True,
                  'note': 'The owning run controller will drain/release; this command performs no platform write.'}
    else:
        module_name, separator, factory_name = config['adapter'].partition(':')
        if not separator or not module_name or not factory_name:
            raise ValueError('adapter must be an explicit importable module:factory')
        factory = getattr(importlib.import_module(module_name), factory_name)
        adapter = factory(config.get('adapter_config', {}))
        controller = Controller(adapter, spec, directory)
        if args.action == 'plan':
            result = controller.plan()
        else:
            stop = threading.Event()
            signal.signal(signal.SIGTERM, lambda *_: stop.set())
            signal.signal(signal.SIGINT, lambda *_: stop.set())
            result = controller.run(should_stop=stop.is_set, once=args.once)
    print(json.dumps(result, ensure_ascii=False, sort_keys=True))


if __name__ == '__main__':
    main()
