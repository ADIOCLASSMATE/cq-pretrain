"""Small protocol registry: task-specific sample planning and score execution.

A benchmark adds a planner plus scorer entry, without changing model execution
or the common scheduler. Planners return stable serializable sample requests;
scorers consume the dataset or committed artifacts as declared here.
"""

from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class Benchmark:
    sampler: Callable | None = None
    scorer: str | None = None
    score_requires_model: bool = False
    normalize_sample: Callable | None = None


def ordinary_samples(task, config, directory=None):
    from evaluations.sampling import build_requests
    return build_requests(task, config)


def vbench_samples(task, config, directory=None):
    import json
    from pathlib import Path
    from evaluations.benchmarks.vbench.adapter import plan_vbench, prepare_vbench

    requests, metadata, prompts = plan_vbench(task, config)
    if directory is not None:
        target = Path(directory)
        if (target / 'run.json').exists():
            # Root identity has already checked computational settings. Retain
            # native metadata's original management snapshot across attempts.
            saved = json.loads((target / 'run.json').read_text())
            metadata['config'] = saved['config']
            metadata['checkpoint'] = saved['checkpoint']
        prepare_vbench(target, metadata, prompts)
    return requests


BENCHMARKS = {
    'loss': Benchmark(scorer='loss', score_requires_model=True),
    't2v': Benchmark(sampler=ordinary_samples),
    'v2t': Benchmark(sampler=ordinary_samples),
    'vbench': Benchmark(sampler=vbench_samples, scorer='vbench'),
}


def get_benchmark(name):
    try:
        return BENCHMARKS[name]
    except KeyError as exc:
        raise ValueError(f'Unknown benchmark {name!r}; available: {list(BENCHMARKS)}') from exc
